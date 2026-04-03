// Ported from tantivy's reader/mod.rs
//
// IndexReader provides snapshot-based reading of committed segments.
// Each call to searcher() returns a consistent snapshot.
// reload() creates a new snapshot from the latest meta.
//
// Simplified: no file watcher (manual reload), no warming,
// no ArcSwap (uses mutex + shared_ptr).

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "index/inverted/indexer/IndexDirectory.h"
#include "index/inverted/indexer/IndexMeta.h"
#include "index/inverted/indexer/Searcher.h"
#include "index/inverted/segment/SegmentReader.h"

namespace milvus::index::inverted {

class IndexReader {
 public:
    explicit IndexReader(IndexDirectory* directory)
        : directory_(directory) {
        reload();
    }

    // Returns the current searcher snapshot.
    // Thread-safe: multiple readers can hold different snapshots.
    std::shared_ptr<Searcher>
    searcher() const {
        std::lock_guard<std::mutex> lock(mu_);
        return current_searcher_;
    }

    // Reload from disk: read meta → open SegmentReaders → swap searcher.
    // Port of InnerIndexReader::reload().
    void
    reload() {
        auto meta = directory_->load_meta();

        std::vector<std::unique_ptr<SegmentReader>> readers;
        readers.reserve(meta.segments.size());
        uint32_t max_doc_id = 0;

        for (const auto& seg_meta : meta.segments) {
            std::string dir = directory_->segment_dir(seg_meta.segment_id);
            readers.push_back(std::make_unique<SegmentReader>(dir));
            if (seg_meta.max_doc_id > max_doc_id) {
                max_doc_id = seg_meta.max_doc_id;
            }
        }

        auto new_searcher =
            std::make_shared<Searcher>(std::move(readers), max_doc_id);

        std::lock_guard<std::mutex> lock(mu_);
        current_searcher_ = std::move(new_searcher);
    }

 private:
    IndexDirectory* directory_;
    mutable std::mutex mu_;
    std::shared_ptr<Searcher> current_searcher_;
};

}  // namespace milvus::index::inverted
