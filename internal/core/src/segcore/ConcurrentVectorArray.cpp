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

#include "segcore/ConcurrentVectorArray.h"

#include "common/EasyAssert.h"
#include "common/Utils.h"

namespace milvus::segcore {

ConcurrentVectorArray::ConcurrentVectorArray(
    int64_t dim,
    int64_t size_per_chunk,
    DataType element_type,
    storage::MmapChunkDescriptorPtr mmap_descriptor)
    : VectorBase(size_per_chunk),
      dim_(dim),
      element_type_(element_type),
      mmap_descriptor_(std::move(mmap_descriptor)) {
    // TODO: Initialize bytes_per_vector_ based on element_type and dim
    // TODO: Initialize offsets_ with [0]
}

// ========== VectorBase interface implementation ==========

void
ConcurrentVectorArray::set_data_raw(ssize_t element_offset,
                                    const void* source,
                                    ssize_t element_count) {
    // TODO: Implement
    // source is VectorArray* array
    // Need to:
    // 1. Extract vectors from each VectorArray
    // 2. Append to data_ contiguously
    // 3. Update offsets_
}

void
ConcurrentVectorArray::set_data_raw(ssize_t element_offset,
                                    const std::vector<FieldDataPtr>& data) {
    // TODO: Implement
}

void
ConcurrentVectorArray::set_data_raw(ssize_t element_offset,
                                    ssize_t element_count,
                                    const DataArray* data,
                                    const FieldMeta& field_meta) {
    // TODO: Implement
    // This is the main entry point from Insert()
    // Need to parse VectorFieldProto from DataArray and store contiguously
}

void
ConcurrentVectorArray::fill_chunk_data(const std::vector<FieldDataPtr>& data) {
    // TODO: Implement
}

SpanBase
ConcurrentVectorArray::get_span_base(int64_t chunk_id) const {
    // TODO: Implement
    // May need special handling since data layout is different
}

const void*
ConcurrentVectorArray::get_chunk_data(ssize_t chunk_index) const {
    // TODO: Implement
    // Note: chunk concept may not apply directly to this storage
}

int64_t
ConcurrentVectorArray::get_chunk_size(ssize_t chunk_index) const {
    // TODO: Implement
}

int64_t
ConcurrentVectorArray::get_element_size() const {
    // TODO: Implement
    // Return bytes_per_vector_ or throw NotImplemented?
}

int64_t
ConcurrentVectorArray::get_element_offset(ssize_t chunk_index) const {
    // TODO: Implement
}

ssize_t
ConcurrentVectorArray::num_chunk() const {
    // TODO: Implement
    // With contiguous storage, may always return 1 or compute based on size
}

bool
ConcurrentVectorArray::empty() {
    // TODO: Implement
}

void
ConcurrentVectorArray::clear() {
    // TODO: Implement
}

// ========== VectorArray-specific interfaces ==========

const void*
ConcurrentVectorArray::get_element_by_id(int64_t element_id) const {
    // TODO: Implement
    // Return: data_.data() + element_id * bytes_per_vector_
}

const void*
ConcurrentVectorArray::get_vector(int64_t row, int64_t j) const {
    // TODO: Implement
    // Return: get_element_by_id(offsets_[row] + j)
}

int64_t
ConcurrentVectorArray::get_row_length(int64_t row) const {
    // TODO: Implement
    // Return: offsets_[row + 1] - offsets_[row]
}

int64_t
ConcurrentVectorArray::total_vectors() const {
    // TODO: Implement
    // Return: offsets_.back() (or offsets_[num_rows()])
}

const void*
ConcurrentVectorArray::get_raw_data() const {
    // TODO: Implement
    // Return: data_.data()
}

const int64_t*
ConcurrentVectorArray::get_offsets() const {
    // TODO: Implement
    // Return: offsets_.data()
}

int64_t
ConcurrentVectorArray::num_rows() const {
    // TODO: Implement
    // Return: offsets_.size() - 1
}

int64_t
ConcurrentVectorArray::dim() const {
    // TODO: Implement
    // Return: dim_
}

DataType
ConcurrentVectorArray::element_type() const {
    // TODO: Implement
    // Return: element_type_
}

VectorArrayView
ConcurrentVectorArray::view_row(int64_t row) const {
    // TODO: Implement
    // Create VectorArrayView pointing to the row's data
}

}  // namespace milvus::segcore
