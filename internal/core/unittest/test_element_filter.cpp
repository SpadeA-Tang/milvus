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
#include <memory>
#include <vector>
#include <boost/format.hpp>

#include "common/Schema.h"
#include "query/Plan.h"
#include "test_utils/DataGen.h"
#include "test_utils/storage_test_utils.h"
#include "test_utils/cachinglayer_test_utils.h"

using namespace milvus;
using namespace milvus::query;
using namespace milvus::segcore;

TEST(ElementFilter, IterativeFilterWithArrayField) {
    // Step 1: Prepare schema with array field
    int dim = 8;
    auto schema = std::make_shared<Schema>();
    auto vec_fid = schema->AddDebugVectorArrayField("structA[array_float_vec]",
                                                    DataType::VECTOR_FLOAT,
                                                    dim,
                                                    knowhere::metric::L2);
    auto int_array_fid = schema->AddDebugArrayField(
        "structA[price_array]", DataType::INT32, false);

    auto int64_fid = schema->AddDebugField("id", DataType::INT64);
    schema->set_primary_field_id(int64_fid);

    size_t N = 100;
    int array_len = 3;

    // Step 2: Generate test data
    auto raw_data = DataGen(schema, N, 42, 0, 1, array_len);

    for (int i = 0; i < raw_data.raw_->fields_data_size(); i++) {
        auto* field_data = raw_data.raw_->mutable_fields_data(i);
        if (field_data->field_id() == int_array_fid.get()) {
            field_data->mutable_scalars()
                ->mutable_array_data()
                ->mutable_data()
                ->Clear();

            for (int row = 0; row < N; row++) {
                auto* array_data = field_data->mutable_scalars()
                                       ->mutable_array_data()
                                       ->mutable_data()
                                       ->Add();

                for (int elem = 0; elem < array_len; elem++) {
                    int value = row * array_len + elem + 1;
                    array_data->mutable_int_data()->mutable_data()->Add(value);
                }
            }
            break;
        }
    }

    // Step 3: Create sealed segment with field data
    auto segment = CreateSealedWithFieldDataLoaded(schema, raw_data);

    // Step 4: Load vector index for element-level search
    auto array_vec_values = raw_data.get_col<VectorFieldProto>(vec_fid);

    // DataGen generates VECTOR_ARRAY with data in float_vector (flattened),
    // not in vector_array (nested structure)
    std::vector<float> vector_data(dim * N * array_len);
    for (int i = 0; i < N; i++) {
        const auto& float_vec = array_vec_values[i].float_vector().data();
        // float_vec contains array_len * dim floats
        for (int j = 0; j < array_len * dim; j++) {
            vector_data[i * array_len * dim + j] = float_vec[j];
        }
    }

    // For element-level search, index all elements (N * array_len vectors)
    auto indexing = GenVecIndexing(N * array_len,
                                   dim,
                                   vector_data.data(),
                                   knowhere::IndexEnum::INDEX_HNSW);
    LoadIndexInfo load_index_info;
    load_index_info.field_id = vec_fid.get();
    load_index_info.index_params = GenIndexParams(indexing.get());
    load_index_info.cache_index =
        CreateTestCacheIndex("test", std::move(indexing));
    load_index_info.index_params["metric_type"] = knowhere::metric::L2;
    load_index_info.field_type = DataType::VECTOR_ARRAY;
    load_index_info.element_type = DataType::VECTOR_FLOAT;
    segment->LoadIndex(load_index_info);

    int topK = 5;

    // Step 5: Test with element-level filter
    // Query: Search array elements, filter by element_value < 10
    {
        std::string raw_plan =
            boost::str(boost::format(R"(vector_anns: <
                                        field_id: %1%
                                        predicates: <
                                          element_filter_expr: <
                                            element_expr: <
                                              binary_range_expr: <
                                                column_info: <
                                                  field_id: %2%
                                                  data_type: Int32
                                                  element_type: Int32
                                                  is_element_level: true
                                                >
                                                lower_inclusive: true
                                                upper_inclusive: false
                                                lower_value: <
                                                  int64_val: -100
                                                >
                                                upper_value: <
                                                  int64_val: 100
                                                >
                                              >
                                            >
                                            predicate: <
                                              binary_arith_op_eval_range_expr: <
                                                column_info: <
                                                  field_id: %3%
                                                  data_type: Int64
                                                >
                                                arith_op: Mod
                                                right_operand: <
                                                  int64_val: 2
                                                >
                                                op: Equal
                                                value: <
                                                  int64_val: 0
                                                >
                                              >
                                            >
                                            struct_name: "structA"
                                          >
                                        >
                                        query_info: <
                                          topk: 5
                                          metric_type: "L2"
                                          hints: "iterative_filter"
                                          search_params: "{\"ef\": 50}"
                                        >
                                        placeholder_tag: "$0">)") %
                       vec_fid.get() % int_array_fid.get() % int64_fid.get());

        proto::plan::PlanNode plan_node;
        auto ok =
            google::protobuf::TextFormat::ParseFromString(raw_plan, &plan_node);
        ASSERT_TRUE(ok) << "Failed to parse element-level filter plan";

        auto plan = CreateSearchPlanFromPlanNode(schema, plan_node);
        ASSERT_NE(plan, nullptr);

        auto num_queries = 1;
        auto seed = 1024;
        auto ph_group_raw =
            CreatePlaceholderGroup(num_queries, dim, seed, true);
        auto ph_group =
            ParsePlaceholderGroup(plan.get(), ph_group_raw.SerializeAsString());

        auto search_result =
            segment->Search(plan.get(), ph_group.get(), 1L << 63);

        // Verify results
        ASSERT_NE(search_result, nullptr);

        // In element-level mode, results should be element indices, not doc offsets
        ASSERT_TRUE(search_result->is_element_level_);
        ASSERT_FALSE(search_result->element_indices_.empty());
        // Also check seg_offsets_ which stores the doc IDs
        ASSERT_FALSE(search_result->seg_offsets_.empty());
        ASSERT_EQ(search_result->element_indices_.size(),
                  search_result->seg_offsets_.size());

        // Should have topK results per query
        ASSERT_LE(search_result->element_indices_.size(), topK * num_queries);

        std::cout << "Element-level search returned:" << std::endl;
        for (auto i = 0; i < search_result->seg_offsets_.size(); i++) {
            std::cout << "doc_id: " << search_result->seg_offsets_[i]
                      << ", element_index: "
                      << search_result->element_indices_[i] << std::endl;
            std::cout << "distance: " << search_result->distances_[i]
                      << std::endl;
        }

        // Verify distances are sorted (ascending for L2)
        for (size_t i = 1; i < search_result->distances_.size(); ++i) {
            ASSERT_LE(search_result->distances_[i - 1],
                      search_result->distances_[i])
                << "Distances should be sorted in ascending order";
        }
    }
}
