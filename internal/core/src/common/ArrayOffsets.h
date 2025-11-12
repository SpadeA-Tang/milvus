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

#include <vector>
#include <algorithm>
#include "common/Types.h"
#include "common/EasyAssert.h"

namespace milvus {

/**
 * @brief Array field offset information for element-level operations
 *
 * Stores cumulative element counts for an array field across all documents.
 * Used for bidirectional conversion between ElementID and (DocID, ElementIndex).
 *
 * Example:
 *   doc0: [elem0, elem1, elem2]       -> 3 elements
 *   doc1: [elem3, elem4]              -> 2 elements
 *   doc2: []                          -> 0 elements
 *   doc3: [elem5, elem6, elem7]       -> 3 elements
 *
 *   offsets = [0, 3, 5, 5, 8]
 *   total elements = 8 (elem0~elem7)
 */
struct ArrayOffsets {
    /**
     * Cumulative element counts
     * offsets[i] = total number of elements in docs [0, i)
     * offsets[0] = 0 (always)
     * offsets[doc_count] = total_element_count
     *
     * Size = doc_count + 1
     */
    std::vector<int64_t> offsets;

    // ========== Basic Info ==========

    int64_t
    GetDocCount() const {
        AssertInfo(!offsets.empty(), "ArrayOffsets not initialized");
        return offsets.size() - 1;
    }

    int64_t
    GetTotalElementCount() const {
        AssertInfo(!offsets.empty(), "ArrayOffsets not initialized");
        return offsets.back();
    }

    int64_t
    GetDocElementCount(int64_t doc_id) const {
        AssertInfo(doc_id >= 0 && doc_id < GetDocCount(),
                   "doc_id out of range: {}", doc_id);
        return offsets[doc_id + 1] - offsets[doc_id];
    }

    // ========== ElementID ↔ (DocID, ElementIndex) Conversion ==========

    /**
     * @brief Convert ElementID to (DocID, ElementIndex)
     *
     * @param element_id Global element ID (0-based)
     * @return (doc_id, element_index_in_doc)
     *
     * Time complexity: O(log N) where N = doc_count
     */
    std::pair<int64_t, int64_t>
    ElementIDToDoc(int64_t element_id) const {
        AssertInfo(!offsets.empty(), "ArrayOffsets not initialized");
        AssertInfo(element_id >= 0 && element_id < GetTotalElementCount(),
                   "element_id out of range: {}", element_id);

        // Binary search to find which doc this element belongs to
        auto it = std::upper_bound(offsets.begin(), offsets.end(), element_id);
        int64_t doc_id = std::distance(offsets.begin(), it) - 1;
        int64_t element_index = element_id - offsets[doc_id];

        return {doc_id, element_index};
    }

    /**
     * @brief Convert (DocID, ElementIndex) to ElementID
     *
     * @param doc_id Document ID
     * @param element_index Element index within the document
     * @return Global element ID
     *
     * Time complexity: O(1)
     */
    int64_t
    DocToElementID(int64_t doc_id, int64_t element_index) const {
        AssertInfo(doc_id >= 0 && doc_id < GetDocCount(),
                   "doc_id out of range: {}", doc_id);
        AssertInfo(element_index >= 0 &&
                       element_index < GetDocElementCount(doc_id),
                   "element_index out of range for doc {}: {}",
                   doc_id,
                   element_index);

        return offsets[doc_id] + element_index;
    }

    // ========== Batch Conversion ==========

    /**
     * @brief Batch convert ElementIDs to (DocID, ElementIndex) pairs
     */
    std::vector<std::pair<int64_t, int64_t>>
    ElementIDsToDocPairs(const std::vector<int64_t>& element_ids) const {
        std::vector<std::pair<int64_t, int64_t>> result;
        result.reserve(element_ids.size());

        for (int64_t elem_id : element_ids) {
            result.push_back(ElementIDToDoc(elem_id));
        }

        return result;
    }

    // ========== Factory Methods ==========

    /**
     * @brief Build ArrayOffsets from element counts per document
     *
     * @param element_counts element_counts[i] = number of elements in doc_i
     * @return ArrayOffsets with cumulative offsets
     */
    static ArrayOffsets
    BuildFromElementCounts(const std::vector<int64_t>& element_counts) {
        ArrayOffsets result;
        result.offsets.reserve(element_counts.size() + 1);

        int64_t cumulative = 0;
        result.offsets.push_back(cumulative);

        for (int64_t count : element_counts) {
            cumulative += count;
            result.offsets.push_back(cumulative);
        }

        return result;
    }

    /**
     * @brief Build ArrayOffsets from segment's array field
     *
     * @param segment Segment interface
     * @param array_field_name Name of the array field
     * @return ArrayOffsets for the array field
     *
     * Note: Implementation to be provided by segment layer
     */
    static ArrayOffsets
    BuildFromSegment(const void* segment, const std::string& array_field_name);
};

}  // namespace milvus
