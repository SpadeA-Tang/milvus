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
#include <map>
#include <shared_mutex>
#include "common/Types.h"
#include "common/EasyAssert.h"

namespace milvus {

class IArrayOffsets {
 public:
    virtual ~IArrayOffsets() = default;

    virtual int64_t
    GetDocCount() const = 0;

    virtual int64_t
    GetTotalElementCount() const = 0;

    virtual std::pair<int64_t, int64_t>
    ElementIDToDoc(int64_t elem_id) const = 0;
};

struct ArrayOffsets : public IArrayOffsets {
    std::vector<std::pair<int32_t, int32_t>> element_info;
    int64_t doc_count = 0;

    int64_t
    GetDocCount() const override {
        return doc_count;
    }

    int64_t
    GetTotalElementCount() const override {
        return element_info.size();
    }

    std::pair<int64_t, int64_t>
    ElementIDToDoc(int64_t elem_id) const override {
        assert(elem_id >= 0 && elem_id < GetTotalElementCount());

        const auto& [doc_id, elem_idx] = element_info[elem_id];
        return {doc_id, elem_idx};
    }

    static ArrayOffsets
    BuildFromSegment(const void* segment, const std::string& array_field_name);
};

class GrowingArrayOffsets : public IArrayOffsets {
 public:
    GrowingArrayOffsets() = default;

    void
    Insert(int64_t doc_id_start, const int32_t* array_lengths, int64_t count);

    int64_t
    GetDocCount() const override {
        std::shared_lock lock(mutex_);
        return committed_doc_count_;
    }

    int64_t
    GetTotalElementCount() const override {
        std::shared_lock lock(mutex_);
        return element_info_.size();
    }

    std::pair<int64_t, int64_t>
    ElementIDToDoc(int64_t elem_id) const override {
        std::shared_lock lock(mutex_);
        assert(elem_id >= 0 &&
               elem_id < static_cast<int64_t>(element_info_.size()));
        const auto& [doc_id, elem_idx] = element_info_[elem_id];
        return {doc_id, elem_idx};
    }

 private:
    struct PendingDoc {
        int64_t doc_id;
        int32_t array_len;
    };

    void
    DrainPendingDocs();

 private:
    // Committed element mappings
    std::vector<std::pair<int32_t, int32_t>> element_info_;

    // Number of documents committed (contiguous from 0)
    int64_t committed_doc_count_ = 0;

    // Pending documents waiting for earlier docs to complete
    // Key: doc_id, automatically sorted
    std::map<int64_t, PendingDoc> pending_docs_;

    // Protects all member variables
    mutable std::shared_mutex mutex_;
};

}  // namespace milvus
