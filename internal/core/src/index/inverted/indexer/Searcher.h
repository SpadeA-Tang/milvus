// Ported from tantivy's core/searcher.rs
//
// A Searcher holds a snapshot of SegmentReaders and dispatches
// queries across all segments, merging results.
//
// Simplified: no Executor (single-threaded search), no doc store.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "index/inverted/query/Query.h"
#include "index/inverted/segment/SegmentReader.h"

namespace milvus::index::inverted {

class Searcher {
 public:
    explicit Searcher(
        std::vector<std::unique_ptr<SegmentReader>> segment_readers,
        uint32_t max_doc_id)
        : segment_readers_(std::move(segment_readers)),
          max_doc_id_(max_doc_id) {
    }

    // Term query across all segments.
    Bitset
    term_query(const uint8_t* term, size_t term_len) const {
        Bitset result(num_bits());
        for (const auto& reader : segment_readers_) {
            auto segment_result = reader->term_query(term, term_len);
            // OR the segment result into the combined bitset.
            // segment_result may be smaller, so resize if needed.
            if (segment_result.size() > result.size()) {
                result.resize(segment_result.size());
            }
            for (size_t i = 0; i < segment_result.size(); i++) {
                if (segment_result[i]) {
                    result.set(i);
                }
            }
        }
        return result;
    }

    // Execute an arbitrary Query across all segments.
    Bitset
    search(const Query& query) const {
        Bitset result(num_bits());
        for (const auto& reader : segment_readers_) {
            auto weight = query.weight();
            auto segment_result = weight->execute(*reader);
            if (segment_result.size() > result.size()) {
                result.resize(segment_result.size());
            }
            for (size_t i = 0; i < segment_result.size(); i++) {
                if (segment_result[i]) {
                    result.set(i);
                }
            }
        }
        return result;
    }

    size_t
    num_segments() const {
        return segment_readers_.size();
    }

    uint32_t
    num_bits() const {
        return max_doc_id_ + 1;
    }

 private:
    std::vector<std::unique_ptr<SegmentReader>> segment_readers_;
    uint32_t max_doc_id_;
};

}  // namespace milvus::index::inverted
