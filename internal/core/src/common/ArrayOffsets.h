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
#include <utility>
#include "common/Types.h"
#include "common/EasyAssert.h"

namespace milvus {

struct ArrayOffsets {
    std::vector<std::pair<int32_t, int32_t>> element_info;
    int64_t doc_count = 0;

    int64_t
    GetDocCount() const {
        return doc_count;
    }

    int64_t
    GetTotalElementCount() const {
        return element_info.size();
    }

    std::pair<int64_t, int64_t>
    ElementIDToDoc(int64_t elem_id) const {
        assert(elem_id >= 0 && elem_id < GetTotalElementCount());

        const auto& [doc_id, elem_idx] = element_info[elem_id];
        return {doc_id, elem_idx};
    }

    static ArrayOffsets
    BuildFromSegment(const void* segment, const std::string& array_field_name);
};

}  // namespace milvus
