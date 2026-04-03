// Ported from tantivy's indexer/segment_entry.rs
//
// A segment entry describes the state of a given segment at a given instant.
// Simplified: no delete support (no alive_bitset, no delete_cursor).

#pragma once

#include "index/inverted/segment/SegmentMeta.h"

namespace milvus::index::inverted {

struct SegmentEntry {
    SegmentMeta meta_;

    explicit SegmentEntry(SegmentMeta meta) : meta_(std::move(meta)) {
    }

    SegmentId
    segment_id() const {
        return meta_.id();
    }

    const SegmentMeta&
    meta() const {
        return meta_;
    }
};

}  // namespace milvus::index::inverted
