// Ported from tantivy's indexer/stamper.rs
//
// Provides globally unique, monotonically increasing operation timestamps.
// Used to track commit points.

#pragma once

#include <atomic>
#include <cstdint>

namespace milvus::index::inverted {

using Opstamp = uint64_t;

class Stamper {
 public:
    explicit Stamper(Opstamp first_opstamp = 0)
        : counter_(first_opstamp) {
    }

    // Returns a new unique opstamp.
    Opstamp
    stamp() {
        return counter_.fetch_add(1, std::memory_order_seq_cst);
    }

    // Reserves n consecutive opstamps. Returns the first.
    Opstamp
    stamps(uint64_t n) {
        return counter_.fetch_add(n, std::memory_order_seq_cst);
    }

    // Revert the stamper to a previous value.
    Opstamp
    revert(Opstamp to) {
        counter_.store(to, std::memory_order_seq_cst);
        return to;
    }

 private:
    std::atomic<uint64_t> counter_;
};

}  // namespace milvus::index::inverted
