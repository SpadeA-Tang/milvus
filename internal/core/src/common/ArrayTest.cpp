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

#include <stdint.h>
#include <string>
#include <string_view>
#include <vector>

#include "common/Array.h"
#include "common/Types.h"
#include "filemanager/InputStream.h"
#include "gtest/gtest.h"
#include "pb/plan.pb.h"
#include "pb/schema.pb.h"

TEST(Array, TestConstructArray) {
    using namespace milvus;

    int N = 10;
    // 1. test int
    milvus::proto::schema::ScalarField field_int_data;
    milvus::proto::plan::Array field_int_array;
    field_int_array.set_same_type(true);
    for (int i = 0; i < N; i++) {
        field_int_data.mutable_int_data()->add_data(i);
        field_int_array.mutable_array()->Add()->set_int64_val(i);
    }
    auto int_array = Array(field_int_data);
    ASSERT_EQ(N, int_array.length());
    ASSERT_EQ(N * sizeof(int), int_array.byte_size());
    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(int_array.get_data_unchecked<int>(i), i);
    }
    ASSERT_TRUE(int_array.is_same_array(field_int_array));
    auto int_array_tmp = Array(const_cast<char*>(int_array.data()),
                               int_array.length(),
                               int_array.byte_size(),
                               int_array.get_element_type(),
                               int_array.get_offsets_data());
    auto int_8_array = Array(const_cast<char*>(int_array.data()),
                             int_array.length(),
                             int_array.byte_size(),
                             DataType::INT8,
                             int_array.get_offsets_data());
    ASSERT_EQ(int_array.length(), int_8_array.length());
    auto int_16_array = Array(const_cast<char*>(int_array.data()),
                              int_array.length(),
                              int_array.byte_size(),
                              DataType::INT16,
                              int_array.get_offsets_data());
    ASSERT_EQ(int_array.length(), int_16_array.length());
    ASSERT_TRUE(int_array_tmp == int_array);
    auto int_array_view = ArrayView(const_cast<char*>(int_array.data()),
                                    int_array.length(),
                                    int_array.byte_size(),
                                    int_array.get_element_type(),
                                    int_array.get_offsets_data());
    ASSERT_EQ(int_array.length(), int_array_view.length());
    ASSERT_EQ(int_array.byte_size(), int_array_view.byte_size());
    ASSERT_EQ(int_array.get_element_type(), int_array_view.get_element_type());

    // 2. test long
    milvus::proto::schema::ScalarField field_long_data;
    milvus::proto::plan::Array field_long_array;
    field_long_array.set_same_type(true);
    for (int i = 0; i < N; i++) {
        field_long_data.mutable_long_data()->add_data(i);
        field_long_array.mutable_array()->Add()->set_int64_val(i);
    }
    auto long_array = Array(field_long_data);
    ASSERT_EQ(N, long_array.length());
    ASSERT_EQ(N * sizeof(int64_t), long_array.byte_size());
    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(long_array.get_data_unchecked<int64_t>(i), i);
    }
    ASSERT_TRUE(long_array.is_same_array(field_int_array));
    auto long_array_tmp = Array(const_cast<char*>(long_array.data()),
                                long_array.length(),
                                long_array.byte_size(),
                                long_array.get_element_type(),
                                long_array.get_offsets_data());
    ASSERT_TRUE(long_array_tmp == long_array);
    auto long_array_view = ArrayView(const_cast<char*>(long_array.data()),
                                     long_array.length(),
                                     long_array.byte_size(),
                                     long_array.get_element_type(),
                                     long_array.get_offsets_data());
    ASSERT_EQ(long_array.length(), long_array_view.length());
    ASSERT_EQ(long_array.byte_size(), long_array_view.byte_size());
    ASSERT_EQ(long_array.get_element_type(),
              long_array_view.get_element_type());

    // 3. test string
    milvus::proto::schema::ScalarField field_string_data;
    milvus::proto::plan::Array field_string_array;
    field_string_array.set_same_type(true);
    for (int i = 0; i < N; i++) {
        field_string_data.mutable_string_data()->add_data(std::to_string(i));
        proto::plan::GenericValue string_val;
        string_val.set_string_val(std::to_string(i));
        field_string_array.mutable_array()->Add()->CopyFrom(string_val);
    }
    auto string_array = Array(field_string_data);
    ASSERT_EQ(N, string_array.length());
    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(string_array.get_data_unchecked<std::string_view>(i),
                  std::to_string(i));
    }
    ASSERT_TRUE(string_array.is_same_array(field_string_array));
    auto string_array_tmp = Array(const_cast<char*>(string_array.data()),
                                  string_array.length(),
                                  string_array.byte_size(),
                                  string_array.get_element_type(),
                                  string_array.get_offsets_data());
    ASSERT_TRUE(string_array_tmp == string_array);
    auto string_array_view = ArrayView(const_cast<char*>(string_array.data()),
                                       string_array.length(),
                                       string_array.byte_size(),
                                       string_array.get_element_type(),
                                       string_array.get_offsets_data());
    ASSERT_EQ(string_array.length(), string_array_view.length());
    ASSERT_EQ(string_array.byte_size(), string_array_view.byte_size());
    ASSERT_EQ(string_array.get_element_type(),
              string_array_view.get_element_type());

    // 4. test bool
    milvus::proto::schema::ScalarField field_bool_data;
    milvus::proto::plan::Array field_bool_array;
    field_bool_array.set_same_type(true);
    for (int i = 0; i < N; i++) {
        field_bool_data.mutable_bool_data()->add_data(bool(i));
        field_bool_array.mutable_array()->Add()->set_bool_val(bool(i));
    }
    auto bool_array = Array(field_bool_data);
    ASSERT_EQ(N, bool_array.length());
    ASSERT_EQ(N * sizeof(bool), bool_array.byte_size());
    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(bool_array.get_data_unchecked<bool>(i), bool(i));
    }
    ASSERT_TRUE(bool_array.is_same_array(field_bool_array));
    auto bool_array_tmp = Array(const_cast<char*>(bool_array.data()),
                                bool_array.length(),
                                bool_array.byte_size(),
                                bool_array.get_element_type(),
                                bool_array.get_offsets_data());
    ASSERT_TRUE(bool_array_tmp == bool_array);
    auto bool_array_view = ArrayView(const_cast<char*>(bool_array.data()),
                                     bool_array.length(),
                                     bool_array.byte_size(),
                                     bool_array.get_element_type(),
                                     bool_array.get_offsets_data());
    ASSERT_EQ(bool_array.length(), bool_array_view.length());
    ASSERT_EQ(bool_array.byte_size(), bool_array_view.byte_size());
    ASSERT_EQ(bool_array.get_element_type(),
              bool_array_view.get_element_type());

    //5. test float
    milvus::proto::schema::ScalarField field_float_data;
    milvus::proto::plan::Array field_float_array;
    field_float_array.set_same_type(true);
    for (int i = 0; i < N; i++) {
        field_float_data.mutable_float_data()->add_data(float(i) * 0.1);
        field_float_array.mutable_array()->Add()->set_float_val(float(i * 0.1));
    }
    auto float_array = Array(field_float_data);
    ASSERT_EQ(N, float_array.length());
    ASSERT_EQ(N * sizeof(float), float_array.byte_size());
    for (int i = 0; i < N; ++i) {
        ASSERT_DOUBLE_EQ(float_array.get_data_unchecked<float>(i),
                         float(i * 0.1));
    }
    ASSERT_TRUE(float_array.is_same_array(field_float_array));
    auto float_array_tmp = Array(const_cast<char*>(float_array.data()),
                                 float_array.length(),
                                 float_array.byte_size(),
                                 float_array.get_element_type(),
                                 float_array.get_offsets_data());
    ASSERT_TRUE(float_array_tmp == float_array);
    auto float_array_view = ArrayView(const_cast<char*>(float_array.data()),
                                      float_array.length(),
                                      float_array.byte_size(),
                                      float_array.get_element_type(),
                                      float_array.get_offsets_data());
    ASSERT_EQ(float_array.length(), float_array_view.length());
    ASSERT_EQ(float_array.byte_size(), float_array_view.byte_size());
    ASSERT_EQ(float_array.get_element_type(),
              float_array_view.get_element_type());

    //6. test double
    milvus::proto::schema::ScalarField field_double_data;
    milvus::proto::plan::Array field_double_array;
    field_double_array.set_same_type(true);
    for (int i = 0; i < N; i++) {
        field_double_data.mutable_double_data()->add_data(double(i) * 0.1);
        field_double_array.mutable_array()->Add()->set_float_val(
            double(i * 0.1));
    }
    auto double_array = Array(field_double_data);
    ASSERT_EQ(N, double_array.length());
    ASSERT_EQ(N * sizeof(double), double_array.byte_size());
    for (int i = 0; i < N; ++i) {
        ASSERT_DOUBLE_EQ(double_array.get_data_unchecked<double>(i),
                         double(i * 0.1));
    }
    ASSERT_TRUE(double_array.is_same_array(field_double_array));
    auto double_array_tmp = Array(const_cast<char*>(double_array.data()),
                                  double_array.length(),
                                  double_array.byte_size(),
                                  double_array.get_element_type(),
                                  double_array.get_offsets_data());
    ASSERT_TRUE(double_array_tmp == double_array);
    auto double_array_view = ArrayView(const_cast<char*>(double_array.data()),
                                       double_array.length(),
                                       double_array.byte_size(),
                                       double_array.get_element_type(),
                                       double_array.get_offsets_data());
    ASSERT_EQ(double_array.length(), double_array_view.length());
    ASSERT_EQ(double_array.byte_size(), double_array_view.byte_size());
    ASSERT_EQ(double_array.get_element_type(),
              double_array_view.get_element_type());

    milvus::proto::schema::ScalarField field_empty_data;
    milvus::proto::plan::Array field_empty_array;
    auto empty_array = Array(field_empty_data);
    ASSERT_EQ(0, empty_array.length());
    ASSERT_EQ(0, empty_array.byte_size());
    ASSERT_TRUE(empty_array.is_same_array(field_empty_array));
}

namespace {

milvus::NullableScalarArrayValueProto
BuildNullableIntArrayValue(const std::vector<int>& values,
                           const std::vector<bool>& valid_data) {
    milvus::NullableScalarArrayValueProto proto;
    for (auto value : values) {
        proto.mutable_data()->mutable_int_data()->add_data(value);
    }
    for (auto valid : valid_data) {
        proto.add_valid_data(valid);
    }
    return proto;
}

milvus::NullableScalarArrayValueProto
BuildNullableStringArrayValue(const std::vector<std::string>& values,
                              const std::vector<bool>& valid_data) {
    milvus::NullableScalarArrayValueProto proto;
    for (const auto& value : values) {
        proto.mutable_data()->mutable_string_data()->add_data(value);
    }
    for (auto valid : valid_data) {
        proto.add_valid_data(valid);
    }
    return proto;
}

milvus::proto::plan::Array
BuildIntPlanArray(const std::vector<int>& values) {
    milvus::proto::plan::Array plan_array;
    plan_array.set_same_type(true);
    for (auto value : values) {
        plan_array.mutable_array()->Add()->set_int64_val(value);
    }
    return plan_array;
}

void
AssertElementValidity(const milvus::Array& array,
                      const std::vector<bool>& valid_data) {
    ASSERT_EQ(valid_data.size(), array.get_element_valid_data_length());
    for (auto i = 0; i < valid_data.size(); ++i) {
        ASSERT_EQ(valid_data[i], array.is_element_valid(i));
    }
}

}  // namespace

TEST(Array, TestNullableScalarArrayValueProtoConstructArray) {
    using namespace milvus;

    auto proto =
        BuildNullableIntArrayValue({10, 20, 30, 40}, {true, false, true, true});
    auto array = Array(proto);

    ASSERT_TRUE(array.is_element_nullable());
    ASSERT_EQ(4, array.length());
    ASSERT_EQ(4, array.get_element_valid_data_length());
    ASSERT_EQ(array.get_element_valid_data().size_in_bytes(),
              array.get_element_valid_data_byte_size());
    ASSERT_TRUE(array.is_element_valid(0));
    ASSERT_FALSE(array.is_element_valid(1));
    ASSERT_TRUE(array.is_element_valid(2));
    ASSERT_TRUE(array.is_element_valid(3));
    ASSERT_TRUE(array.has_invalid_element());

    ASSERT_EQ(10, array.get_data_unchecked<int>(0));
    ASSERT_EQ(30, array.get_data_unchecked<int>(2));
    ASSERT_EQ(40, array.get_data_unchecked<int>(3));
}

TEST(Array, TestScalarFieldProtoConstructArrayIsNotElementNullable) {
    using namespace milvus;

    ScalarFieldProto proto;
    proto.mutable_int_data()->add_data(10);
    proto.mutable_int_data()->add_data(20);

    auto array = Array(proto);

    ASSERT_FALSE(array.is_element_nullable());
    ASSERT_EQ(0, array.get_element_valid_data_length());
    ASSERT_TRUE(array.is_element_valid(0));
    ASSERT_TRUE(array.is_element_valid(1));
    ASSERT_FALSE(array.has_invalid_element());
}

TEST(Array, TestNullableArrayCopyPreservesElementValidity) {
    using namespace milvus;

    auto proto =
        BuildNullableIntArrayValue({10, 20, 30, 40}, {true, false, true, true});
    auto array = Array(proto);
    auto copied = Array(array);

    ASSERT_TRUE(copied.is_element_nullable());
    ASSERT_EQ(array.length(), copied.length());
    ASSERT_EQ(array.get_element_valid_data_length(),
              copied.get_element_valid_data_length());
    AssertElementValidity(copied, {true, false, true, true});
    ASSERT_TRUE(copied == array);

    auto assigned = Array();
    assigned = array;
    ASSERT_TRUE(assigned.is_element_nullable());
    AssertElementValidity(assigned, {true, false, true, true});
    ASSERT_TRUE(assigned == array);
}

TEST(Array, TestNullableArrayConstructFromBitmapView) {
    using namespace milvus;

    int values[] = {10, 20, 30, 40};
    TargetBitmap valid_data(4, false);
    valid_data.set(0);
    valid_data.set(2);

    auto array = Array(reinterpret_cast<char*>(values),
                       4,
                       sizeof(values),
                       DataType::INT32,
                       nullptr,
                       valid_data.view(),
                       true);

    ASSERT_TRUE(array.is_element_nullable());
    ASSERT_TRUE(array.is_element_valid(0));
    ASSERT_FALSE(array.is_element_valid(1));
    ASSERT_TRUE(array.is_element_valid(2));
    ASSERT_FALSE(array.is_element_valid(3));
    ASSERT_TRUE(array.has_invalid_element());
}

TEST(Array, TestNullableArrayViewPreservesElementValidity) {
    using namespace milvus;

    auto proto =
        BuildNullableIntArrayValue({10, 20, 30, 40}, {true, false, true, true});
    auto array = Array(proto);

    auto view = ArrayView(const_cast<char*>(array.data()),
                          array.length(),
                          array.byte_size(),
                          array.get_element_type(),
                          array.get_offsets_data(),
                          array.get_element_valid_data().view(),
                          array.is_element_nullable());

    ASSERT_TRUE(view.is_element_nullable());
    ASSERT_TRUE(view.is_element_valid(0));
    ASSERT_FALSE(view.is_element_valid(1));
    ASSERT_TRUE(view.is_element_valid(2));
    ASSERT_TRUE(view.is_element_valid(3));
    ASSERT_TRUE(view.has_invalid_element());

    Array restored;
    view.output_data(restored);
    ASSERT_TRUE(restored.is_element_nullable());
    AssertElementValidity(restored, {true, false, true, true});
    ASSERT_TRUE(restored == array);
}

TEST(Array, TestNullableArrayOutputNullableProto) {
    using namespace milvus;

    auto proto =
        BuildNullableIntArrayValue({10, 20, 30, 40}, {true, false, true, true});
    auto array = Array(proto);

    auto output = array.output_nullable_data();

    ASSERT_EQ(4, output.data().int_data().data_size());
    ASSERT_EQ(4, output.valid_data_size());
    ASSERT_EQ(10, output.data().int_data().data(0));
    ASSERT_EQ(20, output.data().int_data().data(1));
    ASSERT_EQ(30, output.data().int_data().data(2));
    ASSERT_EQ(40, output.data().int_data().data(3));
    ASSERT_TRUE(output.valid_data(0));
    ASSERT_FALSE(output.valid_data(1));
    ASSERT_TRUE(output.valid_data(2));
    ASSERT_TRUE(output.valid_data(3));
}

TEST(Array, TestNullableStringArrayPreservesOffsetsAndValidity) {
    using namespace milvus;

    auto proto = BuildNullableStringArrayValue({"alpha", "", "gamma"},
                                               {true, false, true});
    auto array = Array(proto);

    ASSERT_TRUE(array.is_element_nullable());
    ASSERT_EQ(3, array.length());
    ASSERT_EQ("alpha", array.get_data_unchecked<std::string_view>(0));
    ASSERT_EQ("gamma", array.get_data_unchecked<std::string_view>(2));
    ASSERT_TRUE(array.is_element_valid(0));
    ASSERT_FALSE(array.is_element_valid(1));
    ASSERT_TRUE(array.is_element_valid(2));

    auto view = ArrayView(const_cast<char*>(array.data()),
                          array.length(),
                          array.byte_size(),
                          array.get_element_type(),
                          array.get_offsets_data(),
                          array.get_element_valid_data().view(),
                          array.is_element_nullable());
    auto output = view.output_nullable_data();

    ASSERT_EQ("alpha", output.data().string_data().data(0));
    ASSERT_EQ("", output.data().string_data().data(1));
    ASSERT_EQ("gamma", output.data().string_data().data(2));
    ASSERT_TRUE(output.valid_data(0));
    ASSERT_FALSE(output.valid_data(1));
    ASSERT_TRUE(output.valid_data(2));
}

TEST(Array, TestNullableArrayEqualityIgnoresInvalidElementPayload) {
    using namespace milvus;

    auto left =
        Array(BuildNullableIntArrayValue({10, 20, 30}, {true, false, true}));
    auto right =
        Array(BuildNullableIntArrayValue({10, 99, 30}, {true, false, true}));
    auto different_validity =
        Array(BuildNullableIntArrayValue({10, 20, 30}, {true, true, true}));

    ASSERT_TRUE(left == right);
    ASSERT_FALSE(left == different_validity);
}

TEST(Array, TestNullableArrayDoesNotMatchPlanArrayWithInvalidElements) {
    using namespace milvus;

    auto nullable_array =
        Array(BuildNullableIntArrayValue({10, 20, 30}, {true, false, true}));
    auto all_valid_nullable_array =
        Array(BuildNullableIntArrayValue({10, 20, 30}, {true, true, true}));
    auto plan_array = BuildIntPlanArray({10, 20, 30});

    ASSERT_FALSE(nullable_array.is_same_array(plan_array));
    ASSERT_TRUE(all_valid_nullable_array.is_same_array(plan_array));
}

TEST(Array, TestNullableArrayRejectsMismatchedValidityLength) {
    using namespace milvus;

    auto proto = BuildNullableIntArrayValue({10, 20, 30}, {true, false});

    ASSERT_ANY_THROW({
        auto array = Array(proto);
        (void)array;
    });
}
