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

#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "common/ArrayVector.h"

std::vector<float>
generate_float_vector(int64_t seed, int64_t N, int64_t dim) {
    std::vector<float> final(dim * N);
    for (int n = 0; n < N; ++n) {
        // generate random float vector
        std::vector<float> data(dim);
        std::default_random_engine er2(seed + n);
        std::normal_distribution<> distr2(0, 1);
        for (auto& x : data) {
            x = distr2(er2);
        }

        std::copy(data.begin(), data.end(), final.begin() + dim * n);
    }
    return final;
};

TEST(VectorArray, TestConstructVectorArray) {
    using namespace milvus;

    int N = 10;
    // 1. test float vector

    int64_t dim = 128;
    milvus::proto::schema::VectorField field_float_vector_array;
    field_float_vector_array.set_dim(dim);

    auto data = generate_float_vector(100, N, dim);
    field_float_vector_array.mutable_float_vector()->mutable_data()->Add(
        data.begin(), data.end());

    auto float_vector_array = ArrayVector(field_float_vector_array);
    ASSERT_EQ(float_vector_array.length(), N);
    ASSERT_EQ(float_vector_array.dim(), dim);
    ASSERT_EQ(float_vector_array.element_type(), DataType::VECTOR_FLOAT);
    ASSERT_EQ(float_vector_array.byte_size(), N * dim * sizeof(float));

    for (int i = 0; i < N; i++) {
        auto floats = float_vector_array.get_data<float>(i);
        for (int j = 0; j < dim; j++) {
            ASSERT_EQ(floats[j], data[i * dim + j]);
        }
    }

    auto float_vector_array_tmp =
        ArrayVector(const_cast<char*>(float_vector_array.data()),
                    float_vector_array.length(),
                    float_vector_array.dim(),
                    float_vector_array.byte_size(),
                    float_vector_array.element_type());

    ASSERT_EQ(float_vector_array_tmp, float_vector_array);
}
