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
#include "common/EasyAssert.h"

namespace milvus {

namespace {
ArrayOffsets
BuildArrayOffsetsFromColumn(const segcore::SegmentInternalInterface* segment,
                            const FieldMeta& field_meta,
                            int64_t row_count) {
    FieldId field_id = field_meta.get_id();
    auto data_type = field_meta.get_data_type();

    ArrayOffsets result;
    result.doc_count = row_count;

    auto temp_op_ctx = std::make_unique<OpContext>();
    auto op_ctx_ptr = temp_op_ctx.get();

    int64_t num_chunks = segment->num_chunk(field_id);
    int64_t current_doc_id = 0;

    if (data_type == DataType::VECTOR_ARRAY) {
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            auto pin_wrapper = segment->chunk_view<VectorArrayView>(
                op_ctx_ptr, field_id, chunk_id);
            const auto& [vector_array_views, valid_flags] = pin_wrapper.get();

            for (size_t i = 0; i < vector_array_views.size(); ++i) {
                int64_t array_len = 0;
                if (valid_flags.empty() || valid_flags[i]) {
                    array_len = vector_array_views[i].length();
                }
                // Add (doc_id, element_index) for each element
                for (int32_t j = 0; j < array_len; ++j) {
                    result.element_info.emplace_back(current_doc_id, j);
                }
                current_doc_id++;
            }
        }
    } else {
        for (int64_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
            auto pin_wrapper =
                segment->chunk_view<ArrayView>(op_ctx_ptr, field_id, chunk_id);
            const auto& [array_views, valid_flags] = pin_wrapper.get();

            for (size_t i = 0; i < array_views.size(); ++i) {
                int64_t array_len = 0;
                if (valid_flags.empty() || valid_flags[i]) {
                    array_len = array_views[i].length();
                }
                // Add (doc_id, element_index) for each element
                for (int32_t j = 0; j < array_len; ++j) {
                    result.element_info.emplace_back(current_doc_id, j);
                }
                current_doc_id++;
            }
        }
    }

    AssertInfo(current_doc_id == row_count,
               "Document count mismatch: expected {}, got {}",
               row_count,
               current_doc_id);

    return result;
}

}  // anonymous namespace

ArrayOffsets
ArrayOffsets::BuildFromSegment(const void* segment,
                               const std::string& array_field_name) {
    auto seg = static_cast<const segcore::SegmentInternalInterface*>(segment);

    int64_t row_count = seg->get_row_count();
    if (row_count == 0) {
        LOG_INFO(
            "ArrayOffsets::BuildFromSegment: empty segment for struct '{}'",
            array_field_name);
        ArrayOffsets result;
        result.doc_count = 0;
        return result;
    }

    const auto& schema = seg->get_schema();
    const auto& array_field_meta =
        FindFirstArrayFieldInStruct(schema, array_field_name);

    ArrayOffsets result =
        BuildArrayOffsetsFromColumn(seg, array_field_meta, row_count);

    int64_t total_elements = result.GetTotalElementCount();

    LOG_INFO(
        "ArrayOffsets::BuildFromSegment: struct_name='{}', "
        "field_id={}, row_count={}, total_elements={}",
        array_field_name,
        array_field_meta.get_id().get(),
        row_count,
        total_elements);

    return result;
}

void
GrowingArrayOffsets::Insert(int64_t doc_id_start,
                            const int32_t* array_lengths,
                            int64_t count) {
    std::unique_lock lock(mutex_);

    for (int64_t i = 0; i < count; ++i) {
        int64_t doc_id = doc_id_start + i;
        int32_t array_len = array_lengths[i];

        if (doc_id == committed_doc_count_) {
            for (int32_t j = 0; j < array_len; ++j) {
                element_info_.emplace_back(static_cast<int32_t>(doc_id), j);
            }
            committed_doc_count_++;

            // Try to drain pending documents
            DrainPendingDocs();
        } else {
            // Cache this document for later
            pending_docs_[doc_id] = {doc_id, array_len};
        }
    }
}

void
GrowingArrayOffsets::DrainPendingDocs() {
    while (true) {
        auto it = pending_docs_.find(committed_doc_count_);
        if (it == pending_docs_.end()) {
            break;
        }

        // Commit this pending document
        const auto& pending = it->second;
        for (int32_t j = 0; j < pending.array_len; ++j) {
            element_info_.emplace_back(static_cast<int32_t>(pending.doc_id), j);
        }
        committed_doc_count_++;

        // Remove from pending
        pending_docs_.erase(it);
    }
}

}  // namespace milvus
