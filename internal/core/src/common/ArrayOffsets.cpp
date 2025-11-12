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

#include "ArrayOffsets.h"
#include "segcore/SegmentInterface.h"
#include "log/Log.h"

namespace milvus {

ArrayOffsets
ArrayOffsets::BuildFromSegment(const void* segment,
                                const std::string& array_field_name) {
    auto seg =
        static_cast<const segcore::SegmentInternalInterface*>(segment);

    // TODO: Full implementation requires:
    // 1. Find StructArrayFieldSchema from schema using array_field_name
    // 2. Determine the field_id of the offsets column or access method
    // 3. Read offsets data from segment to build ArrayOffsets
    //
    // Current POC implementation:
    // - Assumes each document has a fixed number of elements (3 elements per doc)
    // - This allows us to verify the entire element-level filtering pipeline
    // - Will be replaced with actual segment data access later

    int64_t row_count = seg->get_row_count();
    std::vector<int64_t> element_counts;
    element_counts.reserve(row_count);

    // POC: Assume each doc has 3 elements
    for (int64_t i = 0; i < row_count; ++i) {
        element_counts.push_back(3);
    }

    LOG_INFO(
        "ArrayOffsets::BuildFromSegment (POC): struct_name={}, doc_count={}, "
        "elements_per_doc=3, total_elements={}",
        array_field_name,
        row_count,
        row_count * 3);

    return BuildFromElementCounts(element_counts);
}

}  // namespace milvus
