// Ported from tantivy's indexer/segment_register.rs
//
// The segment register keeps track of the list of segments,
// their size as well as the state they are in.

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "index/inverted/indexer/SegmentEntry.h"

namespace milvus::index::inverted {

class SegmentRegister {
 public:
    void
    clear() {
        segment_states_.clear();
    }

    std::vector<SegmentMeta>
    get_mergeable_segments(
        const std::unordered_set<SegmentId, SegmentIdHash>& in_merge) const {
        std::vector<SegmentMeta> result;
        for (const auto& [id, entry] : segment_states_) {
            if (in_merge.find(id) == in_merge.end()) {
                result.push_back(entry.meta());
            }
        }
        return result;
    }

    std::vector<SegmentId>
    segment_ids() const {
        std::vector<SegmentId> ids;
        ids.reserve(segment_states_.size());
        for (const auto& [id, _] : segment_states_) {
            ids.push_back(id);
        }
        return ids;
    }

    std::vector<SegmentEntry>
    segment_entries() const {
        std::vector<SegmentEntry> entries;
        entries.reserve(segment_states_.size());
        for (const auto& [_, entry] : segment_states_) {
            entries.push_back(entry);
        }
        return entries;
    }

    std::vector<SegmentMeta>
    segment_metas() const {
        std::vector<SegmentMeta> metas;
        metas.reserve(segment_states_.size());
        for (const auto& [_, entry] : segment_states_) {
            metas.push_back(entry.meta());
        }
        return metas;
    }

    bool
    contains_all(const std::vector<SegmentId>& ids) const {
        for (const auto& id : ids) {
            if (segment_states_.find(id) == segment_states_.end()) {
                return false;
            }
        }
        return true;
    }

    void
    add_segment_entry(const SegmentEntry& entry) {
        segment_states_.emplace(entry.segment_id(), entry);
    }

    void
    remove_segment(const SegmentId& id) {
        segment_states_.erase(id);
    }

    std::optional<SegmentEntry>
    get(const SegmentId& id) const {
        auto it = segment_states_.find(id);
        if (it != segment_states_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    size_t
    size() const {
        return segment_states_.size();
    }

 private:
    std::unordered_map<SegmentId, SegmentEntry, SegmentIdHash> segment_states_;
};

}  // namespace milvus::index::inverted
