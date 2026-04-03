// Ported from tantivy's indexer/log_merge_policy.rs
//
// LogMergePolicy tries to merge segments that have a similar number
// of documents.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "index/inverted/indexer/MergePolicy.h"

namespace milvus::index::inverted {

class LogMergePolicy : public MergePolicy {
 public:
    void
    set_min_num_segments(size_t min_num_segments) {
        min_num_segments_ = min_num_segments;
    }

    void
    set_max_docs_before_merge(size_t max_docs_before_merge) {
        max_docs_before_merge_ = max_docs_before_merge;
    }

    void
    set_min_layer_size(uint32_t min_layer_size) {
        min_layer_size_ = min_layer_size;
    }

    void
    set_level_log_size(double level_log_size) {
        level_log_size_ = level_log_size;
    }

    std::vector<MergeCandidate>
    compute_merge_candidates(
        const std::vector<SegmentMeta>& segments) const override {
        // Filter and sort by size (descending).
        std::vector<const SegmentMeta*> size_sorted;
        for (const auto& seg : segments) {
            if (seg.num_docs <= static_cast<uint32_t>(max_docs_before_merge_)) {
                size_sorted.push_back(&seg);
            }
        }
        if (size_sorted.empty()) {
            return {};
        }
        std::sort(size_sorted.begin(),
                  size_sorted.end(),
                  [](const SegmentMeta* a, const SegmentMeta* b) {
                      return a->num_docs > b->num_docs;
                  });

        // Group segments into levels by exponential size.
        double current_max_log_size = std::numeric_limits<double>::max();
        std::vector<std::vector<const SegmentMeta*>> levels;
        std::vector<const SegmentMeta*> current_level;

        for (const auto* seg : size_sorted) {
            double segment_log_size =
                std::log2(static_cast<double>(clip_min_size(seg->num_docs)));
            if (segment_log_size < (current_max_log_size - level_log_size_)) {
                if (!current_level.empty()) {
                    levels.push_back(std::move(current_level));
                    current_level.clear();
                }
                current_max_log_size = segment_log_size;
            }
            current_level.push_back(seg);
        }
        if (!current_level.empty()) {
            levels.push_back(std::move(current_level));
        }

        // Filter levels that meet the minimum segment count.
        std::vector<MergeCandidate> candidates;
        for (const auto& level : levels) {
            if (level.size() >= min_num_segments_) {
                MergeCandidate candidate;
                for (const auto* seg : level) {
                    candidate.segment_ids.push_back(seg->id());
                }
                candidates.push_back(std::move(candidate));
            }
        }
        return candidates;
    }

 private:
    uint32_t
    clip_min_size(uint32_t size) const {
        return std::max(min_layer_size_, size);
    }

    size_t min_num_segments_ = 8;
    size_t max_docs_before_merge_ = 10'000'000;
    uint32_t min_layer_size_ = 10'000;
    double level_log_size_ = 0.75;
};

}  // namespace milvus::index::inverted
