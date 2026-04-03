// Ported from tantivy's termdict/fst_termdict/merger.rs
//
// Given a list of sorted term streams (one per segment),
// returns an iterator over sorted unique terms.
//
// For each term, provides which segments contain it and their PostingsInfo.

#pragma once

#include <cstdint>
#include <functional>
#include <queue>
#include <string>
#include <vector>

#include "index/inverted/termdict/SortedBlockKeyMap.h"

namespace milvus::index::inverted {

// tantivy: type TermStreamer = sstable::Streamer<TermSSTable>
using TermStreamer = Streamer;

struct HeapItem {
    TermStreamer* streamer;
    size_t segment_ord;

    // Min-heap: smaller key first, then smaller segment_ord.
    bool
    operator>(const HeapItem& other) const {
        int cmp = streamer->key().compare(other.streamer->key());
        if (cmp != 0) return cmp > 0;
        return segment_ord > other.segment_ord;
    }
};

// K-way merge of sorted term dictionaries from multiple segments.
class TermMerger {
 public:
    explicit TermMerger(std::vector<TermStreamer>& streamers) {
        for (size_t i = 0; i < streamers.size(); i++) {
            if (streamers[i].is_valid()) {
                heap_.push(HeapItem{&streamers[i], i});
            }
        }
    }

    // Advance to the next unique term.
    // Returns true if there is another term, false if done.
    bool
    advance() {
        // Push back current_streamers from previous call.
        for (auto& item : current_streamers_) {
            if (item.streamer->advance()) {
                heap_.push(item);
            }
        }
        current_streamers_.clear();

        if (heap_.empty()) {
            return false;
        }

        // Pop the smallest term.
        auto head = heap_.top();
        heap_.pop();
        current_streamers_.push_back(head);

        // Collect all segments with the same term.
        while (!heap_.empty() &&
               heap_.top().streamer->key() ==
                   current_streamers_[0].streamer->key()) {
            current_streamers_.push_back(heap_.top());
            heap_.pop();
        }
        return true;
    }

    // Returns the current term bytes.
    const std::string&
    key() const {
        return current_streamers_[0].streamer->key();
    }

    // Returns segment ordinals and their PostingsInfo for the current term.
    const std::vector<HeapItem>&
    current_streamers() const {
        return current_streamers_;
    }

 private:
    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<>>
        heap_;
    std::vector<HeapItem> current_streamers_;
};

}  // namespace milvus::index::inverted
