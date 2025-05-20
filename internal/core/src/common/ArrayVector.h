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

#include "FieldMeta.h"
#include "Types.h"
#include "common/EasyAssert.h"
#include "common/VectorTrait.h"

namespace milvus {
class ArrayVector : public milvus::VectorTrait {
 public:
    ArrayVector() = default;

    ~ArrayVector() {
        delete[] data_;
    }

    ArrayVector(
        char* data, int len, int dim, size_t size, DataType element_type)
        : size_(size), length_(len), dim_(dim), element_type_(element_type) {
        data_ = new char[size];
        std::copy(data, data + size, data_);
    }

    explicit ArrayVector(const VectorArray& field_data) {
        dim_ = field_data.dim();
        switch (field_data.data_case()) {
            case VectorArray::kFloatVector: {
                element_type_ = DataType::VECTOR_FLOAT;
                length_ = field_data.float_vector().data().size() / dim_;
                auto data = new float[length_ * dim_];
                size_ = field_data.float_vector().data().size() * sizeof(float);
                std::copy(field_data.float_vector().data().begin(),
                          field_data.float_vector().data().end(),
                          data);
                data_ = reinterpret_cast<char*>(data);
                break;
            }
            case VectorArray::kBinaryVector: {
                element_type_ = DataType::VECTOR_BINARY;
                break;
            }
            case VectorArray::kFloat16Vector: {
                element_type_ = DataType::VECTOR_FLOAT16;
                break;
            }
            case VectorArray::kBfloat16Vector: {
                element_type_ = DataType::VECTOR_BFLOAT16;
                break;
            }
            case VectorArray::kSparseFloatVector: {
                // todo(SpadeA): it should be here(it can be a element in Array)?
                PanicInfo(NotImplemented,
                          "Sparse float vector is not supported");
                break;
            }
            case VectorArray::kInt8Vector: {
                element_type_ = DataType::VECTOR_INT8;
                break;
            }
            default: {
                // empty array
            }
        }
    }

    ArrayVector&
    operator=(const ArrayVector& other) {
        delete[] data_;
        length_ = other.length_;
        size_ = other.size_;
        dim_ = other.dim_;
        element_type_ = other.element_type_;
        data_ = new char[size_];
        std::copy(other.data_, other.data_ + size_, data_);
        return *this;
    }

    bool
    operator==(const ArrayVector& other) const {
        if (element_type_ != other.element_type_ || length_ != other.length_ ||
            size_ != other.size_) {
            return false;
        }

        if (length_ == 0) {
            return true;
        }

        switch (element_type_) {
            case DataType::VECTOR_FLOAT: {
                for (int i = 0; i < length_; ++i) {
                    auto a = get_data<float>(i);
                    auto b = other.get_data<float>(i);
                    for (int j = 0; j < dim_; ++j) {
                        if (a[j] != b[j]) {
                            return false;
                        }
                    }
                }
                return true;
            }
            default: {
                PanicInfo(NotImplemented, "Unsupported vector type");
            }
        }
    }

    template <typename T>
    T*
    get_data(const int index) const {
        AssertInfo(index >= 0 && index < length_,
                   "index out of range, index={}, length={}",
                   index,
                   length_);
        switch (element_type_) {
            case DataType::VECTOR_FLOAT: {
                const float* base =
                    reinterpret_cast<const float*>(data_) + index * dim_;
                return const_cast<T*>(base);
            }
            default: {
                PanicInfo(NotImplemented, "Unsupported vector type");
            }
        }
    }

    VectorArray
    output_data() const {
        VectorArray vector_array;
        vector_array.set_dim(dim_);
        switch (element_type_) {
            case DataType::VECTOR_FLOAT: {
                auto data = reinterpret_cast<const float*>(data_);
                vector_array.mutable_float_vector()->mutable_data()->Add(
                    data, data + length_ * dim_);
                break;
            }
            default: {
                PanicInfo(NotImplemented, "Unsupported vector type");
            }
        }
        return vector_array;
    }

    int
    length() const {
        return length_;
    }

    size_t
    byte_size() const {
        return size_;
    }

    int64_t
    dim() const {
        return dim_;
    }

    DataType
    element_type() const {
        return element_type_;
    }

    const char*
    data() const {
        return data_;
    }

 private:
    int64_t dim_ = 0;
    char* data_{nullptr};
    int length_ = 0;
    int size_ = 0;
    DataType element_type_ = DataType::NONE;
};
}  // namespace milvus