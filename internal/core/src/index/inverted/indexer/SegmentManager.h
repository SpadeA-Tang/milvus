// Ported from tantivy's indexer/segment_manager.rs
//
// The segment manager stores the list of segments as well as their state.
// It guarantees the atomicity of the changes (merges especially).

#pragma once

#include <cassert>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

#include "index/inverted/indexer/SegmentRegister.h"

namespace milvus::index::inverted {

enum class SegmentsStatus { Committed, Uncommitted };

struct SegmentRegisters {
    SegmentRegister uncommitted;
    SegmentRegister committed;

    // Check if all the segments are committed or uncommitted.
    std::optional<SegmentsStatus>
    segments_status(const std::vector<SegmentId>& segment_ids) const {
        if (uncommitted.contains_all(segment_ids)) {
            return SegmentsStatus::Uncommitted;
        }
        if (committed.contains_all(segment_ids)) {
            return SegmentsStatus::Committed;
        }
        return std::nullopt;
    }
};

class SegmentManager {
 public:
    SegmentManager() = default;

    explicit SegmentManager(std::vector<SegmentMeta> committed_metas) {
        for (auto& meta : committed_metas) {
            registers_.committed.add_segment_entry(
                SegmentEntry(std::move(meta)));
        }
    }

    std::pair<std::vector<SegmentMeta>, std::vector<SegmentMeta>>
    get_mergeable_segments(
        const std::unordered_set<SegmentId, SegmentIdHash>& in_merge) const {
        std::shared_lock lock(mu_);
        return {registers_.committed.get_mergeable_segments(in_merge),
                registers_.uncommitted.get_mergeable_segments(in_merge)};
    }

    std::vector<SegmentEntry>
    segment_entries() const {
        std::shared_lock lock(mu_);
        auto entries = registers_.uncommitted.segment_entries();
        auto committed = registers_.committed.segment_entries();
        entries.insert(entries.end(), committed.begin(), committed.end());
        return entries;
    }

    void
    commit(std::vector<SegmentEntry> segment_entries) {
        std::unique_lock lock(mu_);
        registers_.committed.clear();
        registers_.uncommitted.clear();
        for (auto& entry : segment_entries) {
            registers_.committed.add_segment_entry(entry);
        }
    }

    // Marks a list of segments as in merge.
    // Returns the segment entries if all segments are found
    // in committed or uncommitted register.
    std::optional<std::vector<SegmentEntry>>
    start_merge(const std::vector<SegmentId>& segment_ids) const {
        std::shared_lock lock(mu_);
        auto status = registers_.segments_status(segment_ids);
        if (!status.has_value()) {
            return std::nullopt;
        }
        const SegmentRegister& reg =
            (*status == SegmentsStatus::Uncommitted)
                ? registers_.uncommitted
                : registers_.committed;
        std::vector<SegmentEntry> entries;
        for (const auto& id : segment_ids) {
            entries.push_back(reg.get(id).value());
        }
        return entries;
    }

    void
    add_segment(SegmentEntry entry) {
        std::unique_lock lock(mu_);
        registers_.uncommitted.add_segment_entry(entry);
    }

    // Replace a list of segments with their merged equivalent.
    std::optional<SegmentsStatus>
    end_merge(const std::vector<SegmentId>& before_merge_ids,
              std::optional<SegmentEntry> after_merge_entry) {
        std::unique_lock lock(mu_);
        auto status = registers_.segments_status(before_merge_ids);
        if (!status.has_value()) {
            return std::nullopt;
        }
        SegmentRegister& target =
            (*status == SegmentsStatus::Uncommitted)
                ? registers_.uncommitted
                : registers_.committed;
        for (const auto& id : before_merge_ids) {
            target.remove_segment(id);
        }
        if (after_merge_entry.has_value()) {
            target.add_segment_entry(after_merge_entry.value());
        }
        return status;
    }

    std::vector<SegmentMeta>
    committed_segment_metas() const {
        std::unique_lock lock(mu_);
        // Remove empty segments, then return.
        auto entries = registers_.committed.segment_entries();
        for (const auto& entry : entries) {
            if (entry.meta().num_docs == 0) {
                registers_.committed.remove_segment(entry.segment_id());
            }
        }
        return registers_.committed.segment_metas();
    }

    void
    remove_all_segments() {
        std::unique_lock lock(mu_);
        registers_.committed.clear();
        registers_.uncommitted.clear();
    }

 private:
    mutable std::shared_mutex mu_;
    mutable SegmentRegisters registers_;
};

}  // namespace milvus::index::inverted
