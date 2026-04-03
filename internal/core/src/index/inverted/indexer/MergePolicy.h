// Ported from tantivy's indexer/merge_policy.rs
//
// The MergePolicy defines which segments should be merged.
// Every time the list of segments changes, the segment updater
// asks the merge policy if some segments should be merged.

#pragma once

#include <vector>

#include "index/inverted/segment/SegmentMeta.h"

namespace milvus::index::inverted {

// Set of segment ids suggested for a merge.
struct MergeCandidate {
    std::vector<SegmentId> segment_ids;
};

// The MergePolicy defines which segments should be merged.
//
// This call happens on the segment updater thread, and will block
// other segment updates, so all implementations should happen rapidly.
class MergePolicy {
 public:
    virtual ~MergePolicy() = default;
    virtual std::vector<MergeCandidate>
    compute_merge_candidates(
        const std::vector<SegmentMeta>& segments) const = 0;
};

// Never merge segments.
class NoMergePolicy : public MergePolicy {
 public:
    std::vector<MergeCandidate>
    compute_merge_candidates(
        const std::vector<SegmentMeta>& /*segments*/) const override {
        return {};
    }
};

}  // namespace milvus::index::inverted
