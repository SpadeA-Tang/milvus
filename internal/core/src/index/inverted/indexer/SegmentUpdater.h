// Ported from tantivy's indexer/segment_updater.rs
//
// Coordinates segment lifecycle: adding segments, committing,
// triggering merges, and garbage collection.
//
// Simplified: no delete support, no FutureResult (sync execution),
// no active_index_meta cache, merge runs synchronously.

#pragma once

#include <mutex>
#include <memory>
#include <unordered_set>

#include "index/inverted/indexer/IndexDirectory.h"
#include "index/inverted/indexer/IndexMeta.h"
#include "index/inverted/indexer/MergePolicy.h"
#include "index/inverted/indexer/SegmentEntry.h"
#include "index/inverted/indexer/SegmentManager.h"
#include "index/inverted/indexer/SegmentMerger.h"
#include "index/inverted/indexer/Stamper.h"
#include "index/inverted/postings/SegmentSerializer.h"
#include "index/inverted/segment/SegmentReader.h"
#include "index/inverted/termdict/SortedBlockKeyMap.h"

namespace milvus::index::inverted {

class SegmentUpdater {
 public:
    SegmentUpdater(IndexDirectory* directory,
                   SegmentManager* segment_manager,
                   std::shared_ptr<MergePolicy> merge_policy)
        : directory_(directory),
          segment_manager_(segment_manager),
          merge_policy_(std::move(merge_policy)) {
    }

    // Port of schedule_add_segment: add to uncommitted + consider merge.
    void
    add_segment(SegmentEntry entry) {
        std::lock_guard<std::mutex> lock(mu_);
        segment_manager_->add_segment(std::move(entry));
    }

    // Port of schedule_commit:
    // 1. commit segment manager (uncommitted → committed)
    // 2. save meta.json
    // 3. garbage collect
    // 4. consider merge
    Opstamp
    commit(Opstamp opstamp) {
        std::lock_guard<std::mutex> lock(mu_);

        // Commit: move all segments to committed.
        auto entries = segment_manager_->segment_entries();
        segment_manager_->commit(std::move(entries));

        // Save IndexMeta to disk.
        save_meta(opstamp);

        // GC stale files.
        auto living = list_living_files();
        directory_->garbage_collect(living);

        // Evaluate merge candidates.
        consider_merge_options_locked();

        return opstamp;
    }

    // Port of consider_merge_options.
    void
    consider_merge_options() {
        std::lock_guard<std::mutex> lock(mu_);
        consider_merge_options_locked();
    }

    void
    set_merge_policy(std::shared_ptr<MergePolicy> policy) {
        std::lock_guard<std::mutex> lock(mu_);
        merge_policy_ = std::move(policy);
    }

 private:
    void
    consider_merge_options_locked() {
        auto [committed, uncommitted] =
            segment_manager_->get_mergeable_segments(in_merge_);

        auto candidates =
            merge_policy_->compute_merge_candidates(committed);

        for (const auto& candidate : candidates) {
            start_merge(candidate);
        }
    }

    // Port of start_merge + end_merge (synchronous in Phase 1).
    void
    start_merge(const MergeCandidate& candidate) {
        // Mark segments as in-merge.
        auto merge_entries =
            segment_manager_->start_merge(candidate.segment_ids);
        if (!merge_entries.has_value()) {
            return;
        }

        for (const auto& id : candidate.segment_ids) {
            in_merge_.insert(id);
        }

        // Execute merge.
        auto merged_meta = execute_merge(candidate.segment_ids);

        // End merge: replace old segments with merged segment.
        if (merged_meta.has_value()) {
            SegmentEntry merged_entry(merged_meta.value());
            segment_manager_->end_merge(
                candidate.segment_ids, std::move(merged_entry));
        } else {
            // Merge produced empty result.
            segment_manager_->end_merge(candidate.segment_ids, std::nullopt);
        }

        for (const auto& id : candidate.segment_ids) {
            in_merge_.erase(id);
        }
    }

    // Execute merge: open readers → merge → write new segment.
    std::optional<SegmentMeta>
    execute_merge(const std::vector<SegmentId>& segment_ids) {
        // Open readers for segments to merge.
        std::vector<SegmentReader> readers;
        readers.reserve(segment_ids.size());
        for (const auto& id : segment_ids) {
            std::string dir = directory_->segment_dir(id);
            readers.emplace_back(dir);
        }

        // Create output segment.
        SegmentId merged_id = SegmentId::generate_random();
        directory_->create_segment_dir(merged_id);
        std::string merged_dir = directory_->segment_dir(merged_id);

        // Write merged segment.
        LocalFileWriter pst_writer(merged_dir + "/" + kPstFileName);
        LocalFileWriter idx_writer(merged_dir + "/" + kIdxFileName);
        LocalFileWriter dct_writer(merged_dir + "/" + kDctFileName);

        SegmentSerializer serializer(
            &pst_writer,
            std::make_unique<SortedBlockKeyMapWriter>(
                &idx_writer, &dct_writer));

        SegmentMerger merger(readers);
        auto meta = merger.write(serializer, merged_id);

        pst_writer.flush();
        idx_writer.flush();
        dct_writer.flush();

        // Write segment meta.
        LocalFileWriter meta_writer(merged_dir + "/" + kMetaFileName);
        meta.serialize(&meta_writer);
        meta_writer.flush();

        if (meta.num_docs == 0) {
            return std::nullopt;
        }
        return meta;
    }

    void
    save_meta(Opstamp opstamp) {
        auto committed_metas = segment_manager_->committed_segment_metas();
        IndexMeta index_meta;
        index_meta.opstamp = opstamp;
        index_meta.segments = std::move(committed_metas);
        directory_->save_meta(index_meta);
    }

    // Compute the set of files that should not be deleted.
    std::unordered_set<std::string>
    list_living_files() const {
        std::unordered_set<std::string> files;
        // All segments in both registers are living.
        auto entries = segment_manager_->segment_entries();
        for (const auto& entry : entries) {
            std::string prefix =
                "seg_" + std::to_string(entry.segment_id().id) + "/";
            files.insert(prefix + kPstFileName);
            files.insert(prefix + kIdxFileName);
            files.insert(prefix + kDctFileName);
            files.insert(prefix + kMetaFileName);
        }
        return files;
    }

    IndexDirectory* directory_;
    SegmentManager* segment_manager_;
    std::shared_ptr<MergePolicy> merge_policy_;
    std::mutex mu_;
    std::unordered_set<SegmentId, SegmentIdHash> in_merge_;
};

}  // namespace milvus::index::inverted
