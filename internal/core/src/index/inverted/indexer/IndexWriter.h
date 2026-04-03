// Ported from tantivy's indexer/index_writer.rs
//
// tantivy architecture:
//   IndexWriter holds mpsc::SyncSender. add_document() sends
//   AddOperation to a bounded channel. Background indexing worker
//   receives operations, accumulates in SegmentWriter (our
//   PostingsWriter), and flushes when memory budget is exceeded.
//   commit() sends CommitOperation, blocks until worker completes
//   flush + SegmentUpdater::commit().
//
// Phase 1: single worker thread (tantivy supports N workers
// with round-robin dispatch).

#pragma once

#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "index/inverted/indexer/IndexDirectory.h"
#include "index/inverted/indexer/IndexMeta.h"
#include "index/inverted/indexer/LogMergePolicy.h"
#include "index/inverted/indexer/MergePolicy.h"
#include "index/inverted/indexer/SegmentEntry.h"
#include "index/inverted/indexer/SegmentManager.h"
#include "index/inverted/indexer/SegmentUpdater.h"
#include "index/inverted/indexer/Stamper.h"
#include "index/inverted/postings/PostingsWriter.h"
#include "index/inverted/postings/SegmentSerializer.h"
#include "index/inverted/termdict/SortedBlockKeyMap.h"

namespace milvus::index::inverted {

static constexpr size_t kDefaultMemoryBudget = 15 * 1024 * 1024;  // 15 MB
static constexpr size_t kDefaultChannelCapacity = 4096;

// --- Operations sent through channel ---
// Port of tantivy's AddOperation / WriterOperation.

struct AddOperation {
    uint32_t doc_id;
    std::vector<uint8_t> term;  // owned copy
};

struct CommitOperation {
    Opstamp opstamp;
    std::promise<Opstamp> result;
};

struct ShutdownOperation {};

using WriterOperation =
    std::variant<AddOperation, CommitOperation, ShutdownOperation>;

// --- Bounded channel ---
// Port of Rust's std::sync::mpsc::sync_channel.

template <typename T>
class BoundedChannel {
 public:
    explicit BoundedChannel(size_t capacity) : capacity_(capacity) {
    }

    void
    send(T item) {
        std::unique_lock<std::mutex> lock(mu_);
        not_full_.wait(lock,
                       [&] { return queue_.size() < capacity_ || closed_; });
        if (closed_)
            return;
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }

    std::optional<T>
    recv() {
        std::unique_lock<std::mutex> lock(mu_);
        not_empty_.wait(lock, [&] { return !queue_.empty() || closed_; });
        if (queue_.empty())
            return std::nullopt;
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    void
    close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

 private:
    std::queue<T> queue_;
    size_t capacity_;
    bool closed_ = false;
    std::mutex mu_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
};

// --- IndexWriter ---

class IndexWriter {
 public:
    explicit IndexWriter(const std::string& dir,
                         size_t memory_budget = kDefaultMemoryBudget)
        : directory_(dir),
          memory_budget_(memory_budget),
          updater_(&directory_,
                   &segment_manager_,
                   std::make_shared<LogMergePolicy>()),
          channel_(kDefaultChannelCapacity) {
        // If meta exists, load committed segments into segment manager.
        if (directory_.has_meta()) {
            auto meta = directory_.load_meta();
            for (auto& seg : meta.segments) {
                segment_manager_.add_segment(SegmentEntry(seg));
            }
            auto entries = segment_manager_.segment_entries();
            segment_manager_.commit(std::move(entries));
            stamper_.revert(meta.opstamp);
        }
        // Start background worker.
        worker_ = std::thread(&IndexWriter::worker_loop, this);
    }

    ~IndexWriter() {
        channel_.send(ShutdownOperation{});
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    // Non-copyable, non-movable (owns thread).
    IndexWriter(const IndexWriter&) = delete;
    IndexWriter&
    operator=(const IndexWriter&) = delete;
    IndexWriter(IndexWriter&&) = delete;
    IndexWriter&
    operator=(IndexWriter&&) = delete;

    // Send a (doc_id, term) pair to the worker.
    // Blocks only if channel is full (back-pressure).
    void
    add_document(uint32_t doc_id, const uint8_t* term, size_t term_len) {
        AddOperation op;
        op.doc_id = doc_id;
        op.term.assign(term, term + term_len);
        channel_.send(std::move(op));
    }

    // Commit: send commit op to worker, block until complete.
    Opstamp
    commit() {
        auto opstamp = stamper_.stamp();
        CommitOperation op;
        op.opstamp = opstamp;
        std::future<Opstamp> future = op.result.get_future();
        channel_.send(std::move(op));
        return future.get();
    }

    IndexDirectory&
    directory() {
        return directory_;
    }

    const IndexDirectory&
    directory() const {
        return directory_;
    }

    void
    set_merge_policy(std::shared_ptr<MergePolicy> policy) {
        updater_.set_merge_policy(std::move(policy));
    }

 private:
    // Background worker loop: process operations from channel.
    // Port of tantivy's indexing worker thread.
    void
    worker_loop() {
        auto writer = std::make_unique<PostingsWriter<DocIdRecorder>>();
        uint32_t num_docs = 0;
        uint32_t max_doc_id = 0;

        while (true) {
            auto op = channel_.recv();
            if (!op.has_value())
                break;

            auto& operation = op.value();
            if (auto* add = std::get_if<AddOperation>(&operation)) {
                writer->subscribe(
                    add->doc_id, 0, add->term.data(), add->term.size());
                num_docs++;
                if (add->doc_id > max_doc_id) {
                    max_doc_id = add->doc_id;
                }

                // Flush if memory budget exceeded.
                if (writer->mem_usage() >= memory_budget_) {
                    flush_segment(*writer, num_docs, max_doc_id);
                    writer = std::make_unique<PostingsWriter<DocIdRecorder>>();
                    num_docs = 0;
                    max_doc_id = 0;
                }
            } else if (auto* commit =
                           std::get_if<CommitOperation>(&operation)) {
                if (num_docs > 0) {
                    flush_segment(*writer, num_docs, max_doc_id);
                    writer = std::make_unique<PostingsWriter<DocIdRecorder>>();
                    num_docs = 0;
                    max_doc_id = 0;
                }
                updater_.commit(commit->opstamp);
                commit->result.set_value(commit->opstamp);
            } else {
                // ShutdownOperation
                break;
            }
        }
    }

    void
    flush_segment(PostingsWriter<DocIdRecorder>& writer,
                  uint32_t num_docs,
                  uint32_t max_doc_id) {
        SegmentId seg_id = SegmentId::generate_random();
        directory_.create_segment_dir(seg_id);
        std::string seg_dir = directory_.segment_dir(seg_id);

        // Serialize postings + term dictionary.
        LocalFileWriter pst_writer(seg_dir + "/" + kPstFileName);
        LocalFileWriter idx_writer(seg_dir + "/" + kIdxFileName);
        LocalFileWriter dct_writer(seg_dir + "/" + kDctFileName);

        SegmentSerializer serializer(&pst_writer,
                                     std::make_unique<SortedBlockKeyMapWriter>(
                                         &idx_writer, &dct_writer));
        serialize_postings(writer, serializer);
        serializer.close();

        pst_writer.flush();
        idx_writer.flush();
        dct_writer.flush();

        // Write segment meta.
        SegmentMeta meta;
        meta.segment_id = seg_id;
        meta.num_docs = num_docs;
        meta.max_doc_id = max_doc_id;

        LocalFileWriter meta_writer(seg_dir + "/" + kMetaFileName);
        meta.serialize(&meta_writer);
        meta_writer.flush();

        // Add to segment manager via updater.
        updater_.add_segment(SegmentEntry(meta));
    }

    IndexDirectory directory_;
    SegmentManager segment_manager_;
    SegmentUpdater updater_;
    Stamper stamper_;
    size_t memory_budget_;

    BoundedChannel<WriterOperation> channel_;
    std::thread worker_;
};

}  // namespace milvus::index::inverted
