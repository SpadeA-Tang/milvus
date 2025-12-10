// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <shared_mutex>
#include <vector>
#include <atomic>

#include "common/Types.h"
#include "common/VectorArray.h"
#include "segcore/ConcurrentVector.h"
#include "storage/MmapManager.h"

namespace milvus::segcore {

// ConcurrentVectorArray: A specialized storage for VectorArray data type
// that stores all vectors contiguously with an offset array for row-to-vector mapping.
//
// Unlike ConcurrentVector<VectorArray> which stores each VectorArray object separately
// (causing memory fragmentation and requiring data copies during queries),
// this implementation stores all vector data in a contiguous buffer:
//
// Memory Layout:
//   data_:    [v0_0][v0_1][v0_2][v1_0][v1_1][v2_0][v2_1][v2_2][v2_3]...
//              ^^^^^^^^^^^       ^^^^^       ^^^^^^^^^^^^^^^^^^^^^
//              row 0 (3 vecs)    row 1       row 2 (4 vecs)
//
//   offsets_: [0, 3, 5, 9, ...]
//              |  |  |  |
//              |  |  |  └── row 2 ends at vector index 9
//              |  |  └── row 1 ends / row 2 starts at vector index 5
//              |  └── row 0 ends / row 1 starts at vector index 3
//              └── row 0 starts at vector index 0
//
// Benefits:
//   - O(1) access to any vector by element_id: data_ + element_id * bytes_per_vector_
//   - No memory copy needed during search/query
//   - Cache-friendly contiguous memory layout
//   - Efficient view_data implementation for index building
//
class ConcurrentVectorArray : public VectorBase {
public:
    ConcurrentVectorArray(int64_t dim,
                          int64_t size_per_chunk,
                          DataType element_type,
                          storage::MmapChunkDescriptorPtr mmap_descriptor = nullptr);

    ~ConcurrentVectorArray() override = default;

    // Disable copy
    ConcurrentVectorArray(const ConcurrentVectorArray&) = delete;
    ConcurrentVectorArray& operator=(const ConcurrentVectorArray&) = delete;

    // ========== VectorBase interface implementation ==========

    void
    set_data_raw(ssize_t element_offset,
                 const void* source,
                 ssize_t element_count) override;

    void
    set_data_raw(ssize_t element_offset,
                 const std::vector<FieldDataPtr>& data) override;

    void
    set_data_raw(ssize_t element_offset,
                 ssize_t element_count,
                 const DataArray* data,
                 const FieldMeta& field_meta) override;

    void
    fill_chunk_data(const std::vector<FieldDataPtr>& data) override;

    SpanBase
    get_span_base(int64_t chunk_id) const override;

    const void*
    get_chunk_data(ssize_t chunk_index) const override;

    int64_t
    get_chunk_size(ssize_t chunk_index) const override;

    int64_t
    get_element_size() const override;

    int64_t
    get_element_offset(ssize_t chunk_index) const override;

    ssize_t
    num_chunk() const override;

    bool
    empty() override;

    void
    clear() override;

    // ========== VectorArray-specific interfaces ==========

    // Get pointer to a single vector by global element_id (flattened index)
    // This is O(1) operation used by view_data in index building
    const void*
    get_element_by_id(int64_t element_id) const;

    // Get pointer to the j-th vector in row i
    const void*
    get_vector(int64_t row, int64_t j) const;

    // Get number of vectors in a specific row
    int64_t
    get_row_length(int64_t row) const;

    // Get total number of vectors across all rows
    int64_t
    total_vectors() const;

    // Get pointer to contiguous vector data (for search without copy)
    const void*
    get_raw_data() const;

    // Get pointer to offsets array (for search)
    const int64_t*
    get_offsets() const;

    // Get number of rows
    int64_t
    num_rows() const;

    // Get vector dimension
    int64_t
    dim() const;

    // Get element type (VECTOR_FLOAT, VECTOR_FLOAT16, etc.)
    DataType
    element_type() const;

    // Get VectorArray view for a specific row (for compatibility)
    VectorArrayView
    view_row(int64_t row) const;

private:
    // Vector dimension
    int64_t dim_;

    // Element type of vectors (VECTOR_FLOAT, VECTOR_FLOAT16, VECTOR_BINARY, etc.)
    DataType element_type_;

    // Bytes per single vector
    int64_t bytes_per_vector_;

    // Contiguous storage for all vector data
    // Size = total_vectors * bytes_per_vector_
    std::vector<char> data_;

    // Row offset array
    // offsets_[i] = starting vector index of row i
    // offsets_[num_rows] = total number of vectors
    // Size = num_rows + 1
    std::vector<int64_t> offsets_;

    // Thread safety
    mutable std::shared_mutex mutex_;

    // Mmap support (optional, for future use)
    storage::MmapChunkDescriptorPtr mmap_descriptor_;
};

}  // namespace milvus::segcore
