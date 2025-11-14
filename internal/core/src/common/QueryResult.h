// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <map>
#include <limits>
#include <string>
#include <queue>
#include <utility>
#include <vector>
#include <boost/align/aligned_allocator.hpp>
#include <boost/dynamic_bitset.hpp>
#include <NamedType/named_type.hpp>

#include "common/FieldMeta.h"
#include "common/ArrayOffsets.h"
#include "pb/schema.pb.h"
#include "knowhere/index/index_node.h"

namespace milvus {

// scan cost in each search/query
struct StorageCost {
    int64_t scanned_remote_bytes = 0;
    int64_t scanned_total_bytes = 0;

    StorageCost() = default;

    StorageCost(int64_t scanned_remote_bytes, int64_t scanned_total_bytes)
        : scanned_remote_bytes(scanned_remote_bytes),
          scanned_total_bytes(scanned_total_bytes) {
    }

    StorageCost
    operator+(const StorageCost& rhs) const {
        return {scanned_remote_bytes + rhs.scanned_remote_bytes,
                scanned_total_bytes + rhs.scanned_total_bytes};
    }

    void
    operator+=(const StorageCost& rhs) {
        scanned_remote_bytes += rhs.scanned_remote_bytes;
        scanned_total_bytes += rhs.scanned_total_bytes;
    }

    StorageCost
    operator*(const double factor) const {
        return {static_cast<int64_t>(scanned_remote_bytes * factor),
                static_cast<int64_t>(scanned_total_bytes * factor)};
    }

    void
    operator*=(const double factor) {
        scanned_remote_bytes =
            static_cast<int64_t>(scanned_remote_bytes * factor);
        scanned_total_bytes =
            static_cast<int64_t>(scanned_total_bytes * factor);
    }

    void
    operator=(const StorageCost& rhs) {
        scanned_remote_bytes = rhs.scanned_remote_bytes;
        scanned_total_bytes = rhs.scanned_total_bytes;
    }

    std::string
    ToString() const {
        return fmt::format("scanned_remote_bytes: {}, scanned_total_bytes: {}",
                           scanned_remote_bytes,
                           scanned_total_bytes);
    }
};

inline std::ostream&
operator<<(std::ostream& os, const StorageCost& cost) {
    os << cost.ToString();
    return os;
}

struct OffsetDisPair {
 private:
    std::pair<int64_t, float> off_dis_;
    int iterator_idx_;

 public:
    OffsetDisPair(std::pair<int64_t, float> off_dis, int iter_idx)
        : off_dis_(off_dis), iterator_idx_(iter_idx) {
    }

    const std::pair<int64_t, float>&
    GetOffDis() const {
        return off_dis_;
    }

    int
    GetIteratorIdx() const {
        return iterator_idx_;
    }
};

struct OffsetDisPairComparator {
    bool
    operator()(const std::shared_ptr<OffsetDisPair>& left,
               const std::shared_ptr<OffsetDisPair>& right) const {
        if (left->GetOffDis().second != right->GetOffDis().second) {
            return left->GetOffDis().second < right->GetOffDis().second;
        }
        return left->GetOffDis().first < right->GetOffDis().first;
    }
};
/**
 * @brief Abstract base class for vector iterators
 *
 * Provides a unified interface for iterating over (offset, distance) pairs
 * in sorted order. Implementations can be:
 * - ChunkMergeIterator: Merges multiple chunk iterators
 * - ElementFilterIterator: Filters elements based on expressions
 */
class VectorIterator {
 public:
    virtual ~VectorIterator() = default;

    /**
     * @brief Check if there are more elements
     * @return true if Next() will return a valid result
     */
    virtual bool
    HasNext() = 0;

    /**
     * @brief Get the next (offset, distance) pair
     * @return Optional pair of (offset, distance), or nullopt if exhausted
     */
    virtual std::optional<std::pair<int64_t, float>>
    Next() = 0;
};

/**
 * @brief Multi-way merge iterator for vector search results from multiple chunks
 *
 * Merges knowhere iterators from different chunks using a min-heap,
 * returning results in distance-sorted order.
 */
class ChunkMergeIterator : public VectorIterator {
 public:
    ChunkMergeIterator(int chunk_count,
                       const std::vector<int64_t>& total_rows_until_chunk = {})
        : total_rows_until_chunk_(total_rows_until_chunk) {
        iterators_.reserve(chunk_count);
    }

    bool
    HasNext() override {
        return !heap_.empty();
    }

    std::optional<std::pair<int64_t, float>>
    Next() override {
        if (!heap_.empty()) {
            auto top = heap_.top();
            heap_.pop();
            if (iterators_[top->GetIteratorIdx()]->HasNext()) {
                auto origin_pair = iterators_[top->GetIteratorIdx()]->Next();
                auto off_dis_pair = std::make_shared<OffsetDisPair>(
                    origin_pair, top->GetIteratorIdx());
                heap_.push(off_dis_pair);
            }
            return top->GetOffDis();
        }
        return std::nullopt;
    }

    bool
    AddIterator(knowhere::IndexNode::IteratorPtr iter) {
        if (!sealed && iter != nullptr) {
            iterators_.emplace_back(iter);
            return true;
        }
        return false;
    }

    void
    seal() {
        sealed = true;
        int idx = 0;
        for (auto& iter : iterators_) {
            if (iter->HasNext()) {
                auto origin_pair = iter->Next();
                auto off_dis_pair =
                    std::make_shared<OffsetDisPair>(origin_pair, idx++);
                heap_.push(off_dis_pair);
            }
        }
    }

 private:
    int64_t
    convert_to_segment_offset(int64_t chunk_offset, int chunk_idx) {
        if (total_rows_until_chunk_.size() == 0) {
            AssertInfo(
                iterators_.size() == 1,
                "Wrong state for vectorIterators, which having incorrect "
                "kw_iterator count:{} "
                "without setting value for chunk_rows, "
                "cannot convert chunk_offset to segment_offset correctly",
                iterators_.size());
            return chunk_offset;
        }
        return total_rows_until_chunk_[chunk_idx] + chunk_offset;
    }

 private:
    std::vector<knowhere::IndexNode::IteratorPtr> iterators_;
    std::priority_queue<std::shared_ptr<OffsetDisPair>,
                        std::vector<std::shared_ptr<OffsetDisPair>>,
                        OffsetDisPairComparator>
        heap_;
    bool sealed = false;
    std::vector<int64_t> total_rows_until_chunk_;
    //currently, ChunkMergeIterator is guaranteed to be used serially without concurrent problem, in the future
    //we may need to add mutex to protect the variable sealed
};

struct SearchResult {
    SearchResult() = default;

    int64_t
    get_total_result_count() const {
        if (topk_per_nq_prefix_sum_.empty()) {
            return 0;
        }
        AssertInfo(topk_per_nq_prefix_sum_.size() == total_nq_ + 1,
                   "wrong topk_per_nq_prefix_sum_ size {}",
                   topk_per_nq_prefix_sum_.size());
        return topk_per_nq_prefix_sum_[total_nq_];
    }

 public:
    void
    AssembleChunkVectorIterators(
        int64_t nq,
        int chunk_count,
        const std::vector<int64_t>& total_rows_until_chunk,
        const std::vector<knowhere::IndexNode::IteratorPtr>& kw_iterators) {
        AssertInfo(kw_iterators.size() == nq * chunk_count,
                   "kw_iterators count:{} is not equal to nq*chunk_count:{}, "
                   "wrong state",
                   kw_iterators.size(),
                   nq * chunk_count);
        std::vector<std::shared_ptr<VectorIterator>> vector_iterators;
        vector_iterators.reserve(nq);
        for (int i = 0, vec_iter_idx = 0; i < kw_iterators.size(); i++) {
            vec_iter_idx = vec_iter_idx % nq;
            if (vector_iterators.size() < nq) {
                auto chunk_merge_iter = std::make_shared<ChunkMergeIterator>(
                    chunk_count, total_rows_until_chunk);
                vector_iterators.emplace_back(chunk_merge_iter);
            }
            const auto& kw_iterator = kw_iterators[i];
            // Cast to ChunkMergeIterator to call AddIterator
            auto chunk_merge_iter =
                std::static_pointer_cast<ChunkMergeIterator>(
                    vector_iterators[vec_iter_idx++]);
            chunk_merge_iter->AddIterator(kw_iterator);
        }
        for (const auto& vector_iter : vector_iterators) {
            // Cast to ChunkMergeIterator to call seal
            auto chunk_merge_iter =
                std::static_pointer_cast<ChunkMergeIterator>(vector_iter);
            chunk_merge_iter->seal();
        }
        this->vector_iterators_ = vector_iterators;
    }

 public:
    int64_t total_nq_;
    int64_t unity_topK_;
    int64_t total_data_cnt_;
    void* segment_;

    // first fill data during search, and then update data after reducing search results
    std::vector<float> distances_;
    std::vector<int64_t> seg_offsets_;
    std::optional<std::vector<GroupByValueType>> group_by_values_;
    std::optional<int64_t> group_size_;

    // first fill data during fillPrimaryKey, and then update data after reducing search results
    std::vector<PkType> primary_keys_;
    DataType pk_type_;

    // fill data during reducing search result
    std::vector<int64_t> result_offsets_;
    // after reducing search result done, size(distances_) = size(seg_offsets_) = size(primary_keys_) =
    // size(primary_keys_)

    // set output fields data when fill target entity
    std::map<FieldId, std::unique_ptr<milvus::DataArray>> output_fields_data_;

    // used for reduce, filter invalid pk, get real topks count
    std::vector<size_t> topk_per_nq_prefix_sum_{};

    //Vector iterators, used for group by
    std::optional<std::vector<std::shared_ptr<VectorIterator>>>
        vector_iterators_;
    // record the storage usage in search
    StorageCost search_storage_cost_;

    // ========== Element-level Search Support ==========

    /**
     * @brief Indicates if this SearchResult contains element-level results
     *
     * - true: element_indices_ and element_iterators_ are valid
     * - false: seg_offsets_ and vector_iterators_ are valid (doc-level)
     */
    bool is_element_level_{false};

    /**
     * @brief Element indices within arrays (when is_element_level_=true)
     *
     * Each value represents the index within the array (0-based).
     * Used together with seg_offsets_ to identify specific array elements:
     * - seg_offsets_[i] = document ID
     * - element_indices_[i] = index within that document's array
     */
    std::vector<int32_t> element_indices_;

    /**
     * @brief Element-level iterators (used in iterative filter mode)
     *
     * Similar to vector_iterators_ but operates on element granularity.
     * Each iterator yields (element_id, distance) pairs.
     */
    std::optional<std::vector<std::shared_ptr<VectorIterator>>>
        element_iterators_;

    /**
     * @brief Array field offsets for element ↔ doc conversion
     *
     * Required for all element-level operations.
     * Shared across the entire query lifecycle.
     */
    std::shared_ptr<ArrayOffsets> array_offsets_;

    // ========== Helper Methods ==========

    /**
     * @brief Check if result has iterators (either doc or element level)
     */
    bool
    HasIterators() const {
        return (is_element_level_ && element_iterators_.has_value()) ||
               (!is_element_level_ && vector_iterators_.has_value());
    }

    /**
     * @brief Get the appropriate iterators based on result type
     */
    std::optional<std::vector<std::shared_ptr<VectorIterator>>>
    GetIterators() {
        if (is_element_level_) {
            return element_iterators_;
        } else {
            return vector_iterators_;
        }
    }
};

using SearchResultPtr = std::shared_ptr<SearchResult>;
using SearchResultOpt = std::optional<SearchResult>;

struct RetrieveResult {
    RetrieveResult() = default;

 public:
    int64_t total_data_cnt_;
    void* segment_;
    std::vector<int64_t> result_offsets_;
    std::vector<DataArray> field_data_;
    bool has_more_result = true;
    // record the storage usage in retrieve
    StorageCost retrieve_storage_cost_;
};

using RetrieveResultPtr = std::shared_ptr<RetrieveResult>;
using RetrieveResultOpt = std::optional<RetrieveResult>;
}  // namespace milvus
