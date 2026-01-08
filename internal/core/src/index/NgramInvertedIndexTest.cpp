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
#include <iomanip>
#include <map>
#include <random>
#include <string>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "common/Schema.h"
#include "test_utils/GenExprProto.h"
#include "query/PlanProto.h"
#include "query/ExecPlanNodeVisitor.h"
#include "expr/ITypeExpr.h"
#include "test_utils/storage_test_utils.h"
#include "index/IndexFactory.h"
#include "index/NgramInvertedIndex.h"
#include "segcore/load_index_c.h"
#include "test_utils/cachinglayer_test_utils.h"
#include "expr/ITypeExpr.h"

using namespace milvus;
using namespace milvus::query;
using namespace milvus::segcore;
using namespace milvus::exec;

void
test_ngram_with_data(const boost::container::vector<std::string>& data,
                     const std::string& literal,
                     proto::plan::OpType op_type,
                     const std::vector<bool>& expected_result,
                     bool forward_to_br = false) {
    int64_t collection_id = 1;
    int64_t partition_id = 2;
    int64_t segment_id = 3;
    int64_t index_build_id = 4000;
    int64_t index_version = 4000;
    int64_t index_id = 5000;

    auto schema = std::make_shared<Schema>();
    auto field_id = schema->AddDebugField("ngram", DataType::VARCHAR);

    auto field_meta = milvus::segcore::gen_field_meta(collection_id,
                                                      partition_id,
                                                      segment_id,
                                                      field_id.get(),
                                                      DataType::VARCHAR,
                                                      DataType::NONE,
                                                      false);
    auto index_meta = gen_index_meta(
        segment_id, field_id.get(), index_build_id, index_version);

    std::string root_path = "/tmp/test-inverted-index/";
    auto storage_config = gen_local_storage_config(root_path);
    auto cm = CreateChunkManager(storage_config);
    auto fs = storage::InitArrowFileSystem(storage_config);

    size_t nb = data.size();

    auto field_data =
        storage::CreateFieldData(DataType::VARCHAR, DataType::NONE, false);
    field_data->FillFieldData(data.data(), data.size());

    auto segment = CreateSealedSegment(schema);
    auto field_data_info = PrepareSingleFieldInsertBinlog(collection_id,
                                                          partition_id,
                                                          segment_id,
                                                          field_id.get(),
                                                          {field_data},
                                                          cm);
    segment->LoadFieldData(field_data_info);

    auto payload_reader =
        std::make_shared<milvus::storage::PayloadReader>(field_data);
    storage::InsertData insert_data(payload_reader);
    insert_data.SetFieldDataMeta(field_meta);
    insert_data.SetTimestamps(0, 100);

    auto serialized_bytes = insert_data.Serialize(storage::Remote);

    auto get_binlog_path = [=](int64_t log_id) {
        return fmt::format("{}/{}/{}/{}/{}",
                           collection_id,
                           partition_id,
                           segment_id,
                           field_id.get(),
                           log_id);
    };

    auto log_path = get_binlog_path(0);

    auto cm_w = ChunkManagerWrapper(cm);
    cm_w.Write(log_path, serialized_bytes.data(), serialized_bytes.size());

    storage::FileManagerContext ctx(field_meta, index_meta, cm, fs);
    std::vector<std::string> index_files;

    auto index_size = 0;
    {
        Config config;
        config[milvus::index::INDEX_TYPE] = milvus::index::INVERTED_INDEX_TYPE;
        config[INSERT_FILES_KEY] = std::vector<std::string>{log_path};

        auto ngram_params = index::NgramParams{
            .loading_index = false,
            .min_gram = 2,
            .max_gram = 4,
        };
        auto index =
            std::make_shared<index::NgramInvertedIndex>(ctx, ngram_params);
        index->Build(config);

        auto create_index_result = index->Upload();
        auto memSize = create_index_result->GetMemSize();
        index_size = create_index_result->GetSerializedSize();
        ASSERT_GT(memSize, 0);
        ASSERT_GT(index_size, 0);
        index_files = create_index_result->GetIndexFiles();
    }

    {
        Config config;
        config[milvus::index::INDEX_FILES] = index_files;
        config[milvus::LOAD_PRIORITY] =
            milvus::proto::common::LoadPriority::HIGH;

        auto ngram_params = index::NgramParams{
            .loading_index = true,
            .min_gram = 2,
            .max_gram = 4,
        };
        auto index =
            std::make_unique<index::NgramInvertedIndex>(ctx, ngram_params);
        index->Load(milvus::tracer::TraceContext{}, config);

        auto cnt = index->Count();
        ASSERT_EQ(cnt, nb);

        exec::SegmentExpr segment_expr(std::move(std::vector<exec::ExprPtr>{}),
                                       "SegmentExpr",
                                       nullptr,
                                       segment.get(),
                                       field_id,
                                       {},
                                       DataType::VARCHAR,
                                       nb,
                                       8192,
                                       0);
        if (op_type != proto::plan::OpType::Equal) {
            std::optional<TargetBitmap> bitset_opt =
                index->ExecuteQuery(literal, op_type, &segment_expr);
            if (forward_to_br) {
                ASSERT_TRUE(!bitset_opt.has_value());
            } else {
                auto bitset = std::move(bitset_opt.value());
                for (size_t i = 0; i < nb; i++) {
                    ASSERT_EQ(bitset[i], expected_result[i]);
                }
            }
        }
    }

    {
        std::map<std::string, std::string> index_params{
            {milvus::index::INDEX_TYPE, milvus::index::NGRAM_INDEX_TYPE},
            {milvus::index::MIN_GRAM, "2"},
            {milvus::index::MAX_GRAM, "4"},
            {milvus::LOAD_PRIORITY, "HIGH"},
        };
        milvus::segcore::LoadIndexInfo load_index_info{
            .collection_id = collection_id,
            .partition_id = partition_id,
            .segment_id = segment_id,
            .field_id = field_id.get(),
            .field_type = DataType::VARCHAR,
            .enable_mmap = true,
            .mmap_dir_path = "/tmp/test-ngram-index-mmap-dir",
            .index_id = index_id,
            .index_build_id = index_build_id,
            .index_version = index_version,
            .index_params = index_params,
            .index_files = index_files,
            .schema = field_meta.field_schema,
            .index_size = index_size,
        };

        uint8_t trace_id[16] = {0};
        uint8_t span_id[8] = {0};
        trace_id[0] = 1;
        span_id[0] = 2;
        CTraceContext trace{
            .traceID = trace_id,
            .spanID = span_id,
            .traceFlags = 0,
        };
        auto cload_index_info = static_cast<CLoadIndexInfo>(&load_index_info);
        AppendIndexV2(trace, cload_index_info);
        UpdateSealedSegmentIndex(segment.get(), cload_index_info);

        auto unary_range_expr = test::GenUnaryRangeExpr(op_type, literal);
        auto column_info = test::GenColumnInfo(
            field_id.get(), proto::schema::DataType::VarChar, false, false);
        unary_range_expr->set_allocated_column_info(column_info);
        auto expr = test::GenExpr();
        expr->set_allocated_unary_range_expr(unary_range_expr);
        auto parser = ProtoParser(schema);
        auto typed_expr = parser.ParseExprs(*expr);
        auto parsed = std::make_shared<plan::FilterBitsNode>(
            DEFAULT_PLANNODE_ID, typed_expr);
        BitsetType final;
        final = ExecuteQueryExpr(parsed, segment.get(), nb, MAX_TIMESTAMP);
        for (size_t i = 0; i < nb; i++) {
            if (final[i] != expected_result[i]) {
                std::cout << "final[" << i << "] = " << final[i]
                          << ", expected_result[" << i
                          << "] = " << expected_result[i] << std::endl;
            }
            ASSERT_EQ(final[i], expected_result[i]);
        }
    }
}

TEST(NgramIndex, TestNgramWikiEpisode) {
    boost::container::vector<std::string> data;
    data.push_back(
        "'Indira Davelba Murillo Alvarado (Tegucigalpa, "
        "the youngest of eight siblings. She attended primary school at the "
        "Escuela 14 de Julio, and her secondary studies at the Instituto "
        "school called \"Indi del Bosque\", where she taught the children of "
        "Honduran women'");
    data.push_back(
        "Richmond Green Secondary School is a public secondary school in "
        "Richmond Hill, Ontario, Canada.");
    data.push_back(
        "The Gymnasium in 2002 Gymnasium Philippinum or Philippinum High "
        "School is an almost 500-year-old secondary school in Marburg, Hesse, "
        "Germany.");
    data.push_back(
        "Sir Winston Churchill Secondary School is a Canadian secondary school "
        "located in St. Catharines, Ontario.");
    data.push_back("Sir Winston Churchill Secondary School");

    // within min-max_gram
    {
        // equal, all should fail
        std::vector<bool> expected_result{false, false, false, false, false};
        test_ngram_with_data(
            data, "ary", proto::plan::OpType::Equal, expected_result);

        // inner match
        expected_result = {true, true, true, true, true};
        test_ngram_with_data(
            data, "ary", proto::plan::OpType::InnerMatch, expected_result);

        expected_result = {false, true, false, true, true};
        test_ngram_with_data(
            data, "y S", proto::plan::OpType::InnerMatch, expected_result);

        expected_result = {true, true, true, true, false};
        test_ngram_with_data(
            data, "y s", proto::plan::OpType::InnerMatch, expected_result);

        // prefix
        expected_result = {false, false, false, true, true};
        test_ngram_with_data(
            data, "Sir", proto::plan::OpType::PrefixMatch, expected_result);

        // postfix
        expected_result = {false, false, false, false, true};
        test_ngram_with_data(
            data, "ool", proto::plan::OpType::PostfixMatch, expected_result);

        // match
        expected_result = {true, false, false, false, false};
        test_ngram_with_data(
            data, "%Alv%y s%", proto::plan::OpType::Match, expected_result);
    }

    // exceeds max_gram
    {
        // inner match
        std::vector<bool> expected_result{false, true, true, true, false};
        test_ngram_with_data(data,
                             "secondary school",
                             proto::plan::OpType::InnerMatch,
                             expected_result);

        // prefix
        expected_result = {false, false, false, true, true};
        test_ngram_with_data(data,
                             "Sir Winston",
                             proto::plan::OpType::PrefixMatch,
                             expected_result);

        // postfix
        expected_result = {false, false, true, false, false};
        test_ngram_with_data(data,
                             "Germany.",
                             proto::plan::OpType::PostfixMatch,
                             expected_result);

        // match
        expected_result = {true, true, true, true, false};
        test_ngram_with_data(data,
                             "%secondary%school%",
                             proto::plan::OpType::Match,
                             expected_result);
    }
}

TEST(NgramIndex, TestNgramSimple) {
    boost::container::vector<std::string> data(10000,
                                               "elementary school secondary");

    // all can be hit by ngram tantivy but will be filterred out by the second phase
    test_ngram_with_data(data,
                         "secondary school",
                         proto::plan::OpType::InnerMatch,
                         std::vector<bool>(10000, false));

    test_ngram_with_data(data,
                         "ele",
                         proto::plan::OpType::PrefixMatch,
                         std::vector<bool>(10000, true));

    test_ngram_with_data(data,
                         "%ary%sec%",
                         proto::plan::OpType::Match,
                         std::vector<bool>(10000, true));

    // should be forwarded to brute force
    test_ngram_with_data(data,
                         "%ary%s%",
                         proto::plan::OpType::Match,
                         std::vector<bool>(10000, true),
                         true);

    test_ngram_with_data(data,
                         "ary",
                         proto::plan::OpType::PostfixMatch,
                         std::vector<bool>(10000, true));
}

// Test that ngram index should only be used for like operations
// (Match, InnerMatch, PrefixMatch, PostfixMatch)
// and NOT for other operations (Equal, NotEqual, In, NotIn, etc.)
// Issue: https://github.com/milvus-io/milvus/issues/44020
TEST(NgramIndex, TestNonLikeExpressionsWithNgram) {
    boost::container::vector<std::string> data = {"apple",
                                                  "banana",
                                                  "cherry",
                                                  "date",
                                                  "elderberry",
                                                  "fig",
                                                  "grape",
                                                  "honeydew",
                                                  "kiwi",
                                                  "lemon"};

    int64_t collection_id = 1;
    int64_t partition_id = 2;
    int64_t segment_id = 3;
    int64_t index_build_id = 4000;
    int64_t index_version = 4000;
    int64_t index_id = 5000;

    auto schema = std::make_shared<Schema>();
    auto field_id = schema->AddDebugField("ngram", DataType::VARCHAR);

    auto field_meta = milvus::segcore::gen_field_meta(collection_id,
                                                      partition_id,
                                                      segment_id,
                                                      field_id.get(),
                                                      DataType::VARCHAR,
                                                      DataType::NONE,
                                                      false);
    auto index_meta = gen_index_meta(
        segment_id, field_id.get(), index_build_id, index_version);

    std::string root_path = "/tmp/test-inverted-index/";
    auto storage_config = gen_local_storage_config(root_path);
    auto cm = CreateChunkManager(storage_config);
    auto fs = storage::InitArrowFileSystem(storage_config);

    size_t nb = data.size();

    auto field_data =
        storage::CreateFieldData(DataType::VARCHAR, DataType::NONE, false);
    field_data->FillFieldData(data.data(), data.size());

    auto segment = CreateSealedSegment(schema);
    auto field_data_info = PrepareSingleFieldInsertBinlog(collection_id,
                                                          partition_id,
                                                          segment_id,
                                                          field_id.get(),
                                                          {field_data},
                                                          cm);
    segment->LoadFieldData(field_data_info);

    auto payload_reader =
        std::make_shared<milvus::storage::PayloadReader>(field_data);
    storage::InsertData insert_data(payload_reader);
    insert_data.SetFieldDataMeta(field_meta);
    insert_data.SetTimestamps(0, 100);

    auto serialized_bytes = insert_data.Serialize(storage::Remote);

    auto get_binlog_path = [=](int64_t log_id) {
        return fmt::format("{}/{}/{}/{}/{}",
                           collection_id,
                           partition_id,
                           segment_id,
                           field_id.get(),
                           log_id);
    };

    auto log_path = get_binlog_path(0);

    auto cm_w = ChunkManagerWrapper(cm);
    cm_w.Write(log_path, serialized_bytes.data(), serialized_bytes.size());

    storage::FileManagerContext ctx(field_meta, index_meta, cm, fs);
    std::vector<std::string> index_files;

    // Build ngram index
    {
        Config config;
        config[milvus::index::INDEX_TYPE] = milvus::index::INVERTED_INDEX_TYPE;
        config[INSERT_FILES_KEY] = std::vector<std::string>{log_path};

        auto ngram_params = index::NgramParams{
            .loading_index = false,
            .min_gram = 2,
            .max_gram = 4,
        };
        auto index =
            std::make_shared<index::NgramInvertedIndex>(ctx, ngram_params);
        index->Build(config);

        auto create_index_result = index->Upload();
        index_files = create_index_result->GetIndexFiles();
    }

    // Load index and test
    {
        std::map<std::string, std::string> index_params{
            {milvus::index::INDEX_TYPE, milvus::index::NGRAM_INDEX_TYPE},
            {milvus::index::MIN_GRAM, "2"},
            {milvus::index::MAX_GRAM, "4"},
            {milvus::LOAD_PRIORITY, "HIGH"},
        };
        milvus::segcore::LoadIndexInfo load_index_info{
            .collection_id = collection_id,
            .partition_id = partition_id,
            .segment_id = segment_id,
            .field_id = field_id.get(),
            .field_type = DataType::VARCHAR,
            .enable_mmap = true,
            .mmap_dir_path = "/tmp/test-ngram-index-mmap-dir",
            .index_id = index_id,
            .index_build_id = index_build_id,
            .index_version = index_version,
            .index_params = index_params,
            .index_files = index_files,
            .schema = field_meta.field_schema,
            .index_size = 1024 * 1024 * 1024,
        };

        uint8_t trace_id[16] = {0};
        uint8_t span_id[8] = {0};
        trace_id[0] = 1;
        span_id[0] = 2;
        CTraceContext trace{
            .traceID = trace_id,
            .spanID = span_id,
            .traceFlags = 0,
        };
        auto cload_index_info = static_cast<CLoadIndexInfo>(&load_index_info);
        AppendIndexV2(trace, cload_index_info);
        UpdateSealedSegmentIndex(segment.get(), cload_index_info);

        // Test: TermFilterExpr (IN operator)
        {
            std::vector<proto::plan::GenericValue> values;
            proto::plan::GenericValue val1;
            val1.set_string_val("apple");
            values.push_back(val1);
            proto::plan::GenericValue val2;
            val2.set_string_val("banana");
            values.push_back(val2);
            proto::plan::GenericValue val3;
            val3.set_string_val("cherry");
            values.push_back(val3);

            auto term_expr = std::make_shared<milvus::expr::TermFilterExpr>(
                milvus::expr::ColumnInfo(field_id, DataType::VARCHAR), values);
            auto plan = std::make_shared<plan::FilterBitsNode>(
                DEFAULT_PLANNODE_ID, term_expr);

            BitsetType final =
                ExecuteQueryExpr(plan, segment.get(), nb, MAX_TIMESTAMP);
            // Only apple, banana, cherry should match
            for (size_t i = 0; i < nb; i++) {
                if (i < 3) {
                    ASSERT_TRUE(final[i]) << "Expected true at index " << i;
                } else {
                    ASSERT_FALSE(final[i]) << "Expected false at index " << i;
                }
            }
        }

        // Test: UnaryRangeExpr with Equal operator
        {
            auto unary_range_expr =
                test::GenUnaryRangeExpr(proto::plan::OpType::Equal, "apple");
            auto column_info = test::GenColumnInfo(
                field_id.get(), proto::schema::DataType::VarChar, false, false);
            unary_range_expr->set_allocated_column_info(column_info);
            auto expr = test::GenExpr();
            expr->set_allocated_unary_range_expr(unary_range_expr);
            auto parser = ProtoParser(schema);
            auto typed_expr = parser.ParseExprs(*expr);
            auto parsed = std::make_shared<plan::FilterBitsNode>(
                DEFAULT_PLANNODE_ID, typed_expr);
            BitsetType final =
                ExecuteQueryExpr(parsed, segment.get(), nb, MAX_TIMESTAMP);
            // Only apple should match (exact match)
            for (size_t i = 0; i < nb; i++) {
                if (i == 0) {
                    ASSERT_TRUE(final[i]) << "Expected true at index " << i;
                } else {
                    ASSERT_FALSE(final[i]) << "Expected false at index " << i;
                }
            }
        }

        // Test: BinaryRangeFilterExpr
        {
            proto::plan::GenericValue lower_val;
            lower_val.set_string_val("cherry");
            proto::plan::GenericValue upper_val;
            upper_val.set_string_val("grape");

            auto binary_range_expr =
                std::make_shared<milvus::expr::BinaryRangeFilterExpr>(
                    milvus::expr::ColumnInfo(field_id, DataType::VARCHAR),
                    lower_val,
                    upper_val,
                    true,
                    true);
            auto plan = std::make_shared<plan::FilterBitsNode>(
                DEFAULT_PLANNODE_ID, binary_range_expr);

            BitsetType final =
                ExecuteQueryExpr(plan, segment.get(), nb, MAX_TIMESTAMP);
            // Strings between "cherry" and "grape" inclusive: cherry, date, elderberry, fig, grape
            for (size_t i = 0; i < nb; i++) {
                if (i >= 2 && i <= 6) {
                    ASSERT_TRUE(final[i]) << "Expected true at index " << i;
                } else {
                    ASSERT_FALSE(final[i]) << "Expected false at index " << i;
                }
            }
        }

        // Test: LogicalBinaryExpr with AND
        {
            // Create Equal expression
            auto unary_range_expr1 =
                test::GenUnaryRangeExpr(proto::plan::OpType::Equal, "apple");
            auto column_info1 = test::GenColumnInfo(
                field_id.get(), proto::schema::DataType::VarChar, false, false);
            unary_range_expr1->set_allocated_column_info(column_info1);
            auto expr1 = test::GenExpr();
            expr1->set_allocated_unary_range_expr(unary_range_expr1);
            auto parser1 = ProtoParser(schema);
            auto typed_expr1 = parser1.ParseExprs(*expr1);

            // Create NotEqual expression
            auto unary_range_expr2 = test::GenUnaryRangeExpr(
                proto::plan::OpType::NotEqual, "banana");
            auto column_info2 = test::GenColumnInfo(
                field_id.get(), proto::schema::DataType::VarChar, false, false);
            unary_range_expr2->set_allocated_column_info(column_info2);
            auto expr2 = test::GenExpr();
            expr2->set_allocated_unary_range_expr(unary_range_expr2);
            auto parser2 = ProtoParser(schema);
            auto typed_expr2 = parser2.ParseExprs(*expr2);

            // Create LogicalBinaryExpr with AND
            auto logical_and_expr =
                std::make_shared<milvus::expr::LogicalBinaryExpr>(
                    milvus::expr::LogicalBinaryExpr::OpType::And,
                    typed_expr1,
                    typed_expr2);
            auto plan = std::make_shared<plan::FilterBitsNode>(
                DEFAULT_PLANNODE_ID, logical_and_expr);

            BitsetType final =
                ExecuteQueryExpr(plan, segment.get(), nb, MAX_TIMESTAMP);
            // Only apple should match (apple == "apple" AND apple != "banana")
            for (size_t i = 0; i < nb; i++) {
                if (i == 0) {
                    ASSERT_TRUE(final[i]) << "Expected true at index " << i;
                } else {
                    ASSERT_FALSE(final[i]) << "Expected false at index " << i;
                }
            }
        }

        // Test: LogicalUnaryExpr with NOT
        {
            // Create Equal expression
            auto unary_range_expr =
                test::GenUnaryRangeExpr(proto::plan::OpType::Equal, "apple");
            auto column_info = test::GenColumnInfo(
                field_id.get(), proto::schema::DataType::VarChar, false, false);
            unary_range_expr->set_allocated_column_info(column_info);
            auto expr = test::GenExpr();
            expr->set_allocated_unary_range_expr(unary_range_expr);
            auto parser = ProtoParser(schema);
            auto typed_expr = parser.ParseExprs(*expr);

            // Create LogicalUnaryExpr with NOT
            auto logical_not_expr =
                std::make_shared<milvus::expr::LogicalUnaryExpr>(
                    milvus::expr::LogicalUnaryExpr::OpType::LogicalNot,
                    typed_expr);
            auto plan = std::make_shared<plan::FilterBitsNode>(
                DEFAULT_PLANNODE_ID, logical_not_expr);

            BitsetType final =
                ExecuteQueryExpr(plan, segment.get(), nb, MAX_TIMESTAMP);
            // All except apple should match (NOT (field == "apple"))
            for (size_t i = 0; i < nb; i++) {
                if (i != 0) {
                    ASSERT_TRUE(final[i]) << "Expected true at index " << i;
                } else {
                    ASSERT_FALSE(final[i]) << "Expected false at index " << i;
                }
            }
        }

        // Test: LogicalBinaryExpr with OR
        {
            // Create Equal expression
            auto unary_range_expr1 =
                test::GenUnaryRangeExpr(proto::plan::OpType::Equal, "apple");
            auto column_info1 = test::GenColumnInfo(
                field_id.get(), proto::schema::DataType::VarChar, false, false);
            unary_range_expr1->set_allocated_column_info(column_info1);
            auto expr1 = test::GenExpr();
            expr1->set_allocated_unary_range_expr(unary_range_expr1);
            auto parser1 = ProtoParser(schema);
            auto typed_expr1 = parser1.ParseExprs(*expr1);

            // Create Equal expression for "banana"
            auto unary_range_expr2 =
                test::GenUnaryRangeExpr(proto::plan::OpType::Equal, "banana");
            auto column_info2 = test::GenColumnInfo(
                field_id.get(), proto::schema::DataType::VarChar, false, false);
            unary_range_expr2->set_allocated_column_info(column_info2);
            auto expr2 = test::GenExpr();
            expr2->set_allocated_unary_range_expr(unary_range_expr2);
            auto parser2 = ProtoParser(schema);
            auto typed_expr2 = parser2.ParseExprs(*expr2);

            // Create LogicalBinaryExpr with OR
            auto logical_or_expr =
                std::make_shared<milvus::expr::LogicalBinaryExpr>(
                    milvus::expr::LogicalBinaryExpr::OpType::Or,
                    typed_expr1,
                    typed_expr2);
            auto plan = std::make_shared<plan::FilterBitsNode>(
                DEFAULT_PLANNODE_ID, logical_or_expr);

            BitsetType final =
                ExecuteQueryExpr(plan, segment.get(), nb, MAX_TIMESTAMP);
            // Apple and banana should match (apple == "apple" OR field == "banana")
            for (size_t i = 0; i < nb; i++) {
                if (i == 0 || i == 1) {
                    ASSERT_TRUE(final[i]) << "Expected true at index " << i;
                } else {
                    ASSERT_FALSE(final[i]) << "Expected false at index " << i;
                }
            }
        }

        // Test: NullExpr with IS_NULL
        {
            auto null_expr = std::make_shared<milvus::expr::NullExpr>(
                milvus::expr::ColumnInfo(field_id, DataType::VARCHAR),
                proto::plan::NullExpr_NullOp_IsNull);
            auto plan = std::make_shared<plan::FilterBitsNode>(
                DEFAULT_PLANNODE_ID, null_expr);

            BitsetType final =
                ExecuteQueryExpr(plan, segment.get(), nb, MAX_TIMESTAMP);
            // None should match since we have no null values
            for (size_t i = 0; i < nb; i++) {
                ASSERT_FALSE(final[i]) << "Expected false at index " << i;
            }
        }

        // Test: NullExpr with IS_NOT_NULL
        {
            auto null_expr = std::make_shared<milvus::expr::NullExpr>(
                milvus::expr::ColumnInfo(field_id, DataType::VARCHAR),
                proto::plan::NullExpr_NullOp_IsNotNull);
            auto plan = std::make_shared<plan::FilterBitsNode>(
                DEFAULT_PLANNODE_ID, null_expr);

            BitsetType final =
                ExecuteQueryExpr(plan, segment.get(), nb, MAX_TIMESTAMP);
            // All should match since we have no null values
            for (size_t i = 0; i < nb; i++) {
                ASSERT_TRUE(final[i]) << "Expected true at index " << i;
            }
        }

        // // Test: ExistsExpr
        // {
        //     auto exists_expr = std::make_shared<milvus::expr::ExistsExpr>(
        //         milvus::expr::ColumnInfo(field_id, DataType::VARCHAR));
        //     auto plan = std::make_shared<plan::FilterBitsNode>(
        //         DEFAULT_PLANNODE_ID, exists_expr);

        //     BitsetType final = ExecuteQueryExpr(plan, segment.get(), nb, MAX_TIMESTAMP);
        //     // All should match since the field exists for all rows
        //     for (size_t i = 0; i < nb; i++) {
        //         ASSERT_TRUE(final[i]) << "Expected true at index " << i;
        //     }
        // }

        // Test: AlwaysTrueExpr
        {
            auto always_true_expr =
                std::make_shared<milvus::expr::AlwaysTrueExpr>();
            auto plan = std::make_shared<plan::FilterBitsNode>(
                DEFAULT_PLANNODE_ID, always_true_expr);

            BitsetType final =
                ExecuteQueryExpr(plan, segment.get(), nb, MAX_TIMESTAMP);
            // All should match
            for (size_t i = 0; i < nb; i++) {
                ASSERT_TRUE(final[i]) << "Expected true at index " << i;
            }
        }
    }
}

TEST(NgramIndex, TestNgramJson) {
    std::vector<std::string> json_raw_data = {
        R"(1)",
        R"({"a": "Milvus project"})",
        R"({"a": "Zilliz cloud"})",
        R"({"a": "Query Node"})",
        R"({"a": "Data Node"})",
        R"({"a": [1, 2, 3]})",
        R"({"a": {"b": 1}})",
        R"({"a": 1001})",
        R"({"a": true})",
        R"({"a": "Milvus", "b": "Zilliz cloud"})",
    };

    auto json_path = "/a";
    auto schema = std::make_shared<Schema>();
    auto json_fid = schema->AddDebugField("json", DataType::JSON);

    auto file_manager_ctx = storage::FileManagerContext();
    file_manager_ctx.fieldDataMeta.field_schema.set_data_type(
        milvus::proto::schema::JSON);
    file_manager_ctx.fieldDataMeta.field_schema.set_fieldid(json_fid.get());
    file_manager_ctx.fieldDataMeta.field_id = json_fid.get();

    index::CreateIndexInfo create_index_info{
        .index_type = index::INVERTED_INDEX_TYPE,
        .json_cast_type = JsonCastType::FromString("VARCHAR"),
        .json_path = json_path,
        .ngram_params = std::optional<index::NgramParams>{index::NgramParams{
            .loading_index = false,
            .min_gram = 2,
            .max_gram = 3,
        }},
    };
    auto inv_index = index::IndexFactory::GetInstance().CreateJsonIndex(
        create_index_info, file_manager_ctx);

    auto ngram_index = std::unique_ptr<index::NgramInvertedIndex>(
        static_cast<index::NgramInvertedIndex*>(inv_index.release()));

    std::vector<milvus::Json> jsons;
    for (auto& json : json_raw_data) {
        jsons.push_back(milvus::Json(simdjson::padded_string(json)));
    }

    auto json_field =
        std::make_shared<FieldData<milvus::Json>>(DataType::JSON, false);
    json_field->add_json_data(jsons);
    ngram_index->BuildWithFieldData({json_field});
    ngram_index->finish();
    ngram_index->create_reader(milvus::index::SetBitsetSealed);

    auto segment = segcore::CreateSealedSegment(schema);
    segcore::LoadIndexInfo load_index_info;
    load_index_info.field_id = json_fid.get();
    load_index_info.field_type = DataType::JSON;
    load_index_info.cache_index =
        CreateTestCacheIndex("", std::move(ngram_index));

    std::map<std::string, std::string> index_params{
        {milvus::index::INDEX_TYPE, milvus::index::NGRAM_INDEX_TYPE},
        {milvus::index::MIN_GRAM, "2"},
        {milvus::index::MAX_GRAM, "3"},
        {milvus::LOAD_PRIORITY, "HIGH"},
        {JSON_PATH, json_path},
        {JSON_CAST_TYPE, "VARCHAR"}};
    load_index_info.index_params = index_params;

    segment->LoadIndex(load_index_info);

    auto cm = milvus::storage::RemoteChunkManagerSingleton::GetInstance()
                  .GetRemoteChunkManager();
    auto load_info = PrepareSingleFieldInsertBinlog(
        0, 0, 0, json_fid.get(), {json_field}, cm);
    segment->LoadFieldData(load_info);

    std::vector<std::tuple<proto::plan::GenericValue,
                           std::vector<int64_t>,
                           proto::plan::OpType>>
        test_cases;
    proto::plan::GenericValue value;
    value.set_string_val("liz");
    test_cases.push_back(std::make_tuple(
        value, std::vector<int64_t>{}, proto::plan::OpType::Equal));

    value.set_string_val("nothing");
    test_cases.push_back(std::make_tuple(
        value, std::vector<int64_t>{}, proto::plan::OpType::InnerMatch));

    value.set_string_val("il");
    test_cases.push_back(std::make_tuple(
        value, std::vector<int64_t>{1, 2, 9}, proto::plan::OpType::InnerMatch));

    value.set_string_val("lliz");
    test_cases.push_back(std::make_tuple(
        value, std::vector<int64_t>{2}, proto::plan::OpType::InnerMatch));

    value.set_string_val("Zi");
    test_cases.push_back(std::make_tuple(
        value, std::vector<int64_t>{2}, proto::plan::OpType::PrefixMatch));

    value.set_string_val("Zilliz");
    test_cases.push_back(std::make_tuple(
        value, std::vector<int64_t>{2}, proto::plan::OpType::PrefixMatch));

    value.set_string_val("de");
    test_cases.push_back(std::make_tuple(
        value, std::vector<int64_t>{3, 4}, proto::plan::OpType::PostfixMatch));

    value.set_string_val("Node");
    test_cases.push_back(std::make_tuple(
        value, std::vector<int64_t>{3, 4}, proto::plan::OpType::PostfixMatch));

    value.set_string_val("%ery%ode%");
    test_cases.push_back(std::make_tuple(
        value, std::vector<int64_t>{3}, proto::plan::OpType::Match));

    for (auto& test_case : test_cases) {
        auto value = std::get<0>(test_case);
        auto expr = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"a"}, true),
            std::get<2>(test_case),
            value,
            std::vector<proto::plan::GenericValue>{});

        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);

        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), json_raw_data.size(), MAX_TIMESTAMP);
        auto expect_result = std::get<1>(test_case);
        EXPECT_EQ(result.count(), expect_result.size());
        for (auto& id : expect_result) {
            EXPECT_TRUE(result[id]);
        }
    }
}

// Test that ngram index should only be used for like operations on JSON fields
// and NOT for other operations (Equal, NotEqual, In, etc.)
TEST(NgramIndex, TestJsonNonLikeExpressionsWithNgram) {
    std::vector<std::string> json_raw_data = {R"({"name": "apple"})",
                                              R"({"name": "banana"})",
                                              R"({"name": "cherry"})",
                                              R"({"name": "date"})",
                                              R"({"name": "elderberry"})",
                                              R"({"name": "fig"})",
                                              R"({"name": "grape"})",
                                              R"({"name": "honeydew"})",
                                              R"({"name": "kiwi"})",
                                              R"({"name": "lemon"})"};

    auto json_path = "/name";
    auto schema = std::make_shared<Schema>();
    auto json_fid = schema->AddDebugField("json", DataType::JSON);

    auto file_manager_ctx = storage::FileManagerContext();
    file_manager_ctx.fieldDataMeta.field_schema.set_data_type(
        milvus::proto::schema::JSON);
    file_manager_ctx.fieldDataMeta.field_schema.set_fieldid(json_fid.get());
    file_manager_ctx.fieldDataMeta.field_id = json_fid.get();

    index::CreateIndexInfo create_index_info{
        .index_type = index::INVERTED_INDEX_TYPE,
        .json_cast_type = JsonCastType::FromString("VARCHAR"),
        .json_path = json_path,
        .ngram_params = std::optional<index::NgramParams>{index::NgramParams{
            .loading_index = false,
            .min_gram = 2,
            .max_gram = 4,
        }},
    };
    auto inv_index = index::IndexFactory::GetInstance().CreateJsonIndex(
        create_index_info, file_manager_ctx);

    auto ngram_index = std::unique_ptr<index::NgramInvertedIndex>(
        static_cast<index::NgramInvertedIndex*>(inv_index.release()));

    std::vector<milvus::Json> jsons;
    for (auto& json : json_raw_data) {
        jsons.push_back(milvus::Json(simdjson::padded_string(json)));
    }

    auto json_field =
        std::make_shared<FieldData<milvus::Json>>(DataType::JSON, false);
    json_field->add_json_data(jsons);
    ngram_index->BuildWithFieldData({json_field});
    ngram_index->finish();
    ngram_index->create_reader(milvus::index::SetBitsetSealed);

    auto segment = segcore::CreateSealedSegment(schema);
    segcore::LoadIndexInfo load_index_info;
    load_index_info.field_id = json_fid.get();
    load_index_info.field_type = DataType::JSON;
    load_index_info.cache_index =
        CreateTestCacheIndex("", std::move(ngram_index));

    std::map<std::string, std::string> index_params{
        {milvus::index::INDEX_TYPE, milvus::index::NGRAM_INDEX_TYPE},
        {milvus::index::MIN_GRAM, "2"},
        {milvus::index::MAX_GRAM, "4"},
        {milvus::LOAD_PRIORITY, "HIGH"},
        {JSON_PATH, json_path},
        {JSON_CAST_TYPE, "VARCHAR"}};
    load_index_info.index_params = index_params;

    segment->LoadIndex(load_index_info);

    auto cm = milvus::storage::RemoteChunkManagerSingleton::GetInstance()
                  .GetRemoteChunkManager();
    auto load_info = PrepareSingleFieldInsertBinlog(
        0, 0, 0, json_fid.get(), {json_field}, cm);
    segment->LoadFieldData(load_info);

    size_t nb = json_raw_data.size();

    // Test: JSON Equal operation
    {
        proto::plan::GenericValue value;
        value.set_string_val("apple");
        auto expr = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"name"}, true),
            proto::plan::OpType::Equal,
            value,
            std::vector<proto::plan::GenericValue>{});

        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), nb, MAX_TIMESTAMP);

        // Only first record should match (exact match for "apple")
        EXPECT_EQ(result.count(), 1);
        EXPECT_TRUE(result[0]);
    }

    // Test: JSON NotEqual operation
    {
        proto::plan::GenericValue value;
        value.set_string_val("apple");
        auto expr = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"name"}, true),
            proto::plan::OpType::NotEqual,
            value,
            std::vector<proto::plan::GenericValue>{});

        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), nb, MAX_TIMESTAMP);

        // All except first record should match
        EXPECT_EQ(result.count(), 9);
        EXPECT_FALSE(result[0]);
        for (size_t i = 1; i < nb; i++) {
            EXPECT_TRUE(result[i]);
        }
    }

    // Test: JSON GreaterThan operation
    {
        proto::plan::GenericValue value;
        value.set_string_val("fig");
        auto expr = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"name"}, true),
            proto::plan::OpType::GreaterThan,
            value,
            std::vector<proto::plan::GenericValue>{});

        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), nb, MAX_TIMESTAMP);

        // Records with names > "fig": grape, honeydew, kiwi, lemon
        EXPECT_EQ(result.count(), 4);
        for (size_t i = 6; i < nb; i++) {
            EXPECT_TRUE(result[i]);
        }
    }

    // Test: JSON LessThan operation
    {
        proto::plan::GenericValue value;
        value.set_string_val("date");
        auto expr = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"name"}, true),
            proto::plan::OpType::LessThan,
            value,
            std::vector<proto::plan::GenericValue>{});

        auto plan =
            std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID, expr);
        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), nb, MAX_TIMESTAMP);

        // Records with names < "date": apple, banana, cherry
        EXPECT_EQ(result.count(), 3);
        for (size_t i = 0; i < 3; i++) {
            EXPECT_TRUE(result[i]);
        }
    }

    // Test: JSON TermFilterExpr (IN operation)
    {
        std::vector<proto::plan::GenericValue> values;
        proto::plan::GenericValue val1, val2, val3;
        val1.set_string_val("apple");
        val2.set_string_val("cherry");
        val3.set_string_val("grape");
        values.push_back(val1);
        values.push_back(val2);
        values.push_back(val3);

        auto term_expr = std::make_shared<milvus::expr::TermFilterExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"name"}, true),
            values);
        auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                           term_expr);

        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), nb, MAX_TIMESTAMP);

        // Only apple, cherry, grape should match
        EXPECT_EQ(result.count(), 3);
        EXPECT_TRUE(result[0]);  // apple
        EXPECT_TRUE(result[2]);  // cherry
        EXPECT_TRUE(result[6]);  // grape
    }

    // Test: JSON BinaryRangeFilterExpr
    {
        proto::plan::GenericValue lower_val;
        lower_val.set_string_val("cherry");
        proto::plan::GenericValue upper_val;
        upper_val.set_string_val("grape");

        auto binary_range_expr =
            std::make_shared<milvus::expr::BinaryRangeFilterExpr>(
                milvus::expr::ColumnInfo(
                    json_fid, DataType::JSON, {"name"}, true),
                lower_val,
                upper_val,
                true,
                true);
        auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                           binary_range_expr);

        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), nb, MAX_TIMESTAMP);

        // Strings between "cherry" and "grape" inclusive: cherry, date, elderberry, fig, grape
        EXPECT_EQ(result.count(), 5);
        for (size_t i = 2; i <= 6; i++) {
            EXPECT_TRUE(result[i]);
        }
    }

    // Test: JSON NullExpr IS_NULL
    {
        auto null_expr = std::make_shared<milvus::expr::NullExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"name"}, true),
            proto::plan::NullExpr_NullOp_IsNull);
        auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                           null_expr);

        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), nb, MAX_TIMESTAMP);

        // None should match since all have non-null names
        EXPECT_EQ(result.count(), 0);
    }

    // Test: JSON NullExpr IS_NOT_NULL
    {
        auto null_expr = std::make_shared<milvus::expr::NullExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"name"}, true),
            proto::plan::NullExpr_NullOp_IsNotNull);
        auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                           null_expr);

        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), nb, MAX_TIMESTAMP);

        // All should match since all have non-null names
        EXPECT_EQ(result.count(), 10);
        for (size_t i = 0; i < nb; i++) {
            EXPECT_TRUE(result[i]);
        }
    }

    // Test: JSON ExistsExpr
    {
        auto exists_expr = std::make_shared<milvus::expr::ExistsExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"name"}, true));
        auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                           exists_expr);

        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), nb, MAX_TIMESTAMP);

        // All should match since all have the "name" field
        EXPECT_EQ(result.count(), 10);
        for (size_t i = 0; i < nb; i++) {
            EXPECT_TRUE(result[i]);
        }
    }

    // Test: JSON LogicalBinaryExpr with AND
    {
        // Create Equal expression for "apple"
        proto::plan::GenericValue val1;
        val1.set_string_val("apple");
        auto expr1 = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"name"}, true),
            proto::plan::OpType::Equal,
            val1,
            std::vector<proto::plan::GenericValue>{});

        // Create NotEqual expression for "banana"
        proto::plan::GenericValue val2;
        val2.set_string_val("banana");
        auto expr2 = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"name"}, true),
            proto::plan::OpType::NotEqual,
            val2,
            std::vector<proto::plan::GenericValue>{});

        // Create LogicalBinaryExpr with AND
        auto logical_and_expr =
            std::make_shared<milvus::expr::LogicalBinaryExpr>(
                milvus::expr::LogicalBinaryExpr::OpType::And, expr1, expr2);
        auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                           logical_and_expr);

        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), nb, MAX_TIMESTAMP);

        // Only apple should match (name == "apple" AND name != "banana")
        EXPECT_EQ(result.count(), 1);
        EXPECT_TRUE(result[0]);
    }

    // Test: JSON LogicalUnaryExpr with NOT
    {
        proto::plan::GenericValue value;
        value.set_string_val("apple");
        auto equal_expr = std::make_shared<milvus::expr::UnaryRangeFilterExpr>(
            milvus::expr::ColumnInfo(json_fid, DataType::JSON, {"name"}, true),
            proto::plan::OpType::Equal,
            value,
            std::vector<proto::plan::GenericValue>{});

        // Create LogicalUnaryExpr with NOT
        auto logical_not_expr =
            std::make_shared<milvus::expr::LogicalUnaryExpr>(
                milvus::expr::LogicalUnaryExpr::OpType::LogicalNot, equal_expr);
        auto plan = std::make_shared<plan::FilterBitsNode>(DEFAULT_PLANNODE_ID,
                                                           logical_not_expr);

        auto result = milvus::query::ExecuteQueryExpr(
            plan, segment.get(), nb, MAX_TIMESTAMP);

        // All except apple should match (NOT (name == "apple"))
        EXPECT_EQ(result.count(), 9);
        EXPECT_FALSE(result[0]);
        for (size_t i = 1; i < nb; i++) {
            EXPECT_TRUE(result[i]);
        }
    }
}

// Performance test for pre_filter optimization.
// This test measures the benefit of applying pre_filter from previous expressions
// before the ngram post-filtering phase.
//
// Scenario: `int_field < threshold AND varchar_field LIKE '%pattern%'`
// - 1,000,000 rows with random unique sentences
// - Tests phase1 hit rates: 10%, 25%, 50%, 75% (different keywords)
// - Tests pre_filter selectivities: 10%, 25%, 50%, 75%
// - 3 warmup runs, 5 test runs averaged
// - Optimization: candidate_set &= pre_filter before post-filtering
TEST(NgramIndex, TestPreFilterOptimizationPerformance) {
    // Configuration
    const size_t nb = 1000000;  // Total number of rows

    int64_t collection_id = 1;
    int64_t partition_id = 2;
    int64_t segment_id = 3;
    int64_t index_build_id = 4000;
    int64_t index_version = 4000;

    auto schema = std::make_shared<Schema>();
    auto varchar_field_id = schema->AddDebugField("text", DataType::VARCHAR);
    auto int_field_id = schema->AddDebugField("filter_val", DataType::INT64);

    auto field_meta = milvus::segcore::gen_field_meta(collection_id,
                                                      partition_id,
                                                      segment_id,
                                                      varchar_field_id.get(),
                                                      DataType::VARCHAR,
                                                      DataType::NONE,
                                                      false);
    auto index_meta = gen_index_meta(
        segment_id, varchar_field_id.get(), index_build_id, index_version);

    std::string root_path = "/tmp/test-ngram-prefilter-perf/";
    auto storage_config = gen_local_storage_config(root_path);
    auto cm = CreateChunkManager(storage_config);
    auto fs = storage::InitArrowFileSystem(storage_config);

    // Generate test data with random sentences and multiple keywords:
    // - keyword_10: appears in ~10% of rows
    // - keyword_25: appears in ~25% of rows
    // - keyword_50: appears in ~50% of rows
    // - keyword_75: appears in ~75% of rows
    // Each sentence is unique with random words to create diverse ngram tokens
    boost::container::vector<std::string> text_data;
    std::vector<int64_t> int_data;
    text_data.reserve(nb);
    int_data.reserve(nb);

    // Keywords with different hit rates
    struct KeywordConfig {
        std::string keyword;
        int percent;
        size_t count = 0;
    };
    std::vector<KeywordConfig> keywords = {
        {"alpha_keyword", 10},
        {"beta_keyword", 25},
        {"gamma_keyword", 50},
        {"delta_keyword", 75},
    };

    // Word pools for generating random sentences
    std::vector<std::string> adjectives = {"quick",
                                           "lazy",
                                           "bright",
                                           "dark",
                                           "smooth",
                                           "rough",
                                           "warm",
                                           "cold",
                                           "large",
                                           "small",
                                           "happy",
                                           "sad",
                                           "ancient",
                                           "modern",
                                           "simple",
                                           "complex"};
    std::vector<std::string> nouns = {"database",
                                      "server",
                                      "network",
                                      "system",
                                      "process",
                                      "memory",
                                      "storage",
                                      "cluster",
                                      "node",
                                      "index",
                                      "query",
                                      "table",
                                      "record",
                                      "field",
                                      "cache"};
    std::vector<std::string> verbs = {"processes",
                                      "handles",
                                      "manages",
                                      "stores",
                                      "retrieves",
                                      "updates",
                                      "deletes",
                                      "creates",
                                      "monitors",
                                      "optimizes",
                                      "scales",
                                      "replicates"};

    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<size_t> adj_dist(0, adjectives.size() - 1);
    std::uniform_int_distribution<size_t> noun_dist(0, nouns.size() - 1);
    std::uniform_int_distribution<size_t> verb_dist(0, verbs.size() - 1);
    std::uniform_int_distribution<int> percent_dist(0, 99);

    for (size_t i = 0; i < nb; i++) {
        // Build a random sentence
        std::string sentence =
            "The " + adjectives[adj_dist(rng)] + " " + nouns[noun_dist(rng)] +
            " " + verbs[verb_dist(rng)] + " the " + adjectives[adj_dist(rng)] +
            " " + nouns[noun_dist(rng)] + " id_" + std::to_string(i);

        // Randomly add keywords based on their target percentages
        for (auto& kw : keywords) {
            if (percent_dist(rng) < kw.percent) {
                sentence += " " + kw.keyword;
                kw.count++;
            }
        }

        text_data.push_back(sentence);
        int_data.push_back(static_cast<int64_t>(i));
    }

    std::cout << "Generated data with keywords:" << std::endl;
    for (const auto& kw : keywords) {
        std::cout << "  - " << kw.keyword << ": " << kw.count << " rows ("
                  << (100.0 * kw.count / nb) << "%)" << std::endl;
    }

    auto text_field_data =
        storage::CreateFieldData(DataType::VARCHAR, DataType::NONE, false);
    text_field_data->FillFieldData(text_data.data(), text_data.size());

    auto int_field_data =
        storage::CreateFieldData(DataType::INT64, DataType::NONE, false);
    int_field_data->FillFieldData(int_data.data(), int_data.size());

    auto segment = CreateSealedSegment(schema);

    // Load varchar field
    auto text_field_info =
        PrepareSingleFieldInsertBinlog(collection_id,
                                       partition_id,
                                       segment_id,
                                       varchar_field_id.get(),
                                       {text_field_data},
                                       cm);
    segment->LoadFieldData(text_field_info);

    // Load int field
    auto int_field_info = PrepareSingleFieldInsertBinlog(collection_id,
                                                         partition_id,
                                                         segment_id,
                                                         int_field_id.get(),
                                                         {int_field_data},
                                                         cm);
    segment->LoadFieldData(int_field_info);

    // Build and upload ngram index
    auto payload_reader =
        std::make_shared<milvus::storage::PayloadReader>(text_field_data);
    storage::InsertData insert_data(payload_reader);
    insert_data.SetFieldDataMeta(field_meta);
    insert_data.SetTimestamps(0, 100);
    auto serialized_bytes = insert_data.Serialize(storage::Remote);

    auto log_path = fmt::format("{}/{}/{}/{}/{}",
                                collection_id,
                                partition_id,
                                segment_id,
                                varchar_field_id.get(),
                                0);

    auto cm_w = ChunkManagerWrapper(cm);
    cm_w.Write(log_path, serialized_bytes.data(), serialized_bytes.size());

    storage::FileManagerContext ctx(field_meta, index_meta, cm, fs);
    std::vector<std::string> index_files;

    {
        Config config;
        config[milvus::index::INDEX_TYPE] = milvus::index::INVERTED_INDEX_TYPE;
        config[INSERT_FILES_KEY] = std::vector<std::string>{log_path};

        auto ngram_params = index::NgramParams{
            .loading_index = false,
            .min_gram = 2,
            .max_gram = 3,
        };
        auto index =
            std::make_shared<index::NgramInvertedIndex>(ctx, ngram_params);
        index->Build(config);

        auto create_index_result = index->Upload();
        index_files = create_index_result->GetIndexFiles();
    }

    // Load ngram index
    Config config;
    config[milvus::index::INDEX_FILES] = index_files;
    config[milvus::LOAD_PRIORITY] = milvus::proto::common::LoadPriority::HIGH;

    auto ngram_params = index::NgramParams{
        .loading_index = true,
        .min_gram = 2,
        .max_gram = 3,
    };
    auto index = std::make_unique<index::NgramInvertedIndex>(ctx, ngram_params);
    index->Load(milvus::tracer::TraceContext{}, config);

    // Test configuration
    const int warmup_runs = 3;
    const int test_runs = 2;
    std::vector<int> pre_filter_percentages = {50, 20, 10, 5, 2};

    // Helper lambda to create SegmentExpr
    auto create_segment_expr = [&]() {
        return exec::SegmentExpr(std::move(std::vector<exec::ExprPtr>{}),
                                 "SegmentExpr",
                                 nullptr,
                                 segment.get(),
                                 varchar_field_id,
                                 {},
                                 DataType::VARCHAR,
                                 nb,
                                 8192,
                                 0);
    };

    // Helper lambda to create pre_filter bitmap
    auto create_pre_filter = [&](int selectivity_pct) {
        TargetBitmap pre_filter(nb, false);
        std::mt19937 pre_rng(123 + selectivity_pct);
        std::uniform_int_distribution<int> pre_dist(0, 99);
        for (size_t i = 0; i < nb; i++) {
            if (pre_dist(pre_rng) < selectivity_pct) {
                pre_filter[i] = true;
            }
        }
        return pre_filter;
    };

    // Structure to store test results
    struct TestResult {
        int phase1_hit_pct;
        int pre_filter_pct;
        double selectivity;
        double avg_ms;
        double speedup;
    };
    std::vector<TestResult> results;

    std::cout << "\n=== Pre-filter Optimization Performance Test ==="
              << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  - Total rows: " << nb << std::endl;
    std::cout << "  - Warmup runs: " << warmup_runs << std::endl;
    std::cout << "  - Test runs: " << test_runs << std::endl;

    // Warmup runs (discard results)
    std::cout << "\nWarming up..." << std::endl;
    for (int w = 0; w < warmup_runs; w++) {
        auto seg_expr = create_segment_expr();
        index->ExecuteQuery(keywords[0].keyword,
                            proto::plan::OpType::InnerMatch,
                            &seg_expr,
                            nullptr);
    }
    std::cout << "Warmup complete.\n" << std::endl;
    std::cout << "Running tests (debug output below)...\n" << std::endl;

    // Test each keyword (different phase1 hit rates) with each pre_filter selectivity
    for (const auto& kw : keywords) {
        // First, measure baseline (no pre_filter) for this keyword
        std::cout << "\n============ Phase1 Hit: " << kw.percent
                  << "%, Pre-filter: 100% (baseline) ============" << std::endl;
        double baseline_total_ms = 0;
        size_t baseline_count = 0;
        for (int r = 0; r < test_runs; r++) {
            auto seg_expr = create_segment_expr();
            auto start = std::chrono::high_resolution_clock::now();
            auto result = index->ExecuteQuery(kw.keyword,
                                              proto::plan::OpType::InnerMatch,
                                              &seg_expr,
                                              nullptr);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start)
                    .count() /
                1000.0;
            baseline_total_ms += duration_ms;
            if (r == 0) {
                ASSERT_TRUE(result.has_value());
                baseline_count = result->count();
            }
        }
        double baseline_avg_ms = baseline_total_ms / test_runs;
        double baseline_selectivity = 100.0 * baseline_count / nb;

        // Store baseline result
        results.push_back(
            {kw.percent, 100, baseline_selectivity, baseline_avg_ms, 1.0});

        // Test with different pre_filter selectivities
        for (int pct : pre_filter_percentages) {
            std::cout << "\n============ Phase1 Hit: " << kw.percent
                      << "%, Pre-filter: " << pct
                      << "% ============" << std::endl;
            auto pre_filter = create_pre_filter(pct);

            double total_ms = 0;
            size_t result_count = 0;
            for (int r = 0; r < test_runs; r++) {
                auto seg_expr = create_segment_expr();
                auto start = std::chrono::high_resolution_clock::now();
                auto result =
                    index->ExecuteQuery(kw.keyword,
                                        proto::plan::OpType::InnerMatch,
                                        &seg_expr,
                                        &pre_filter);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration_ms =
                    std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                        .count() /
                    1000.0;
                total_ms += duration_ms;
                if (r == 0) {
                    ASSERT_TRUE(result.has_value());
                    result_count = result->count();
                }
            }
            double avg_ms = total_ms / test_runs;
            double speedup = avg_ms > 0 ? baseline_avg_ms / avg_ms : 0;
            double selectivity = 100.0 * result_count / nb;

            // Store result
            results.push_back({kw.percent, pct, selectivity, avg_ms, speedup});
        }
    }

    // Print all results at the end
    std::cout << "\n\n";
    std::cout << "============================================================="
                 "====\n";
    std::cout
        << "                        FINAL RESULTS                           \n";
    std::cout << "============================================================="
                 "====\n";
    std::cout << std::setw(12) << "Phase1 Hit" << std::setw(18)
              << "PF Selectivity" << std::setw(14) << "Selectivity"
              << std::setw(12) << "Avg (ms)" << std::setw(12) << "Speedup\n";
    std::cout << std::string(68, '-') << "\n";

    int last_phase1_pct = -1;
    for (const auto& r : results) {
        if (last_phase1_pct != -1 && last_phase1_pct != r.phase1_hit_pct) {
            std::cout << std::string(68, '-') << "\n";
        }
        last_phase1_pct = r.phase1_hit_pct;

        std::cout << std::setw(12) << std::to_string(r.phase1_hit_pct) + "%"
                  << std::setw(18) << std::to_string(r.pre_filter_pct) + "%"
                  << std::setw(13) << std::fixed << std::setprecision(2)
                  << r.selectivity << "%" << std::setw(12) << std::fixed
                  << std::setprecision(2) << r.avg_ms << std::setw(12)
                  << std::fixed << std::setprecision(2) << r.speedup << "x\n";
    }
    std::cout << std::string(68, '-') << std::endl;
}

// Helper function to truncate string at valid UTF-8 boundary
std::string
TruncateUtf8(const std::string& text, size_t max_len) {
    if (text.size() <= max_len) {
        return text;
    }
    // Find valid UTF-8 boundary by backing up from max_len
    size_t len = max_len;
    while (len > 0 && (static_cast<unsigned char>(text[len]) & 0xC0) == 0x80) {
        // Current byte is a UTF-8 continuation byte (10xxxxxx), back up
        --len;
    }
    return text.substr(0, len);
}

// Helper function to run wiki data performance test with optional text truncation
void
RunWikiDataPerformanceTest(size_t max_text_length) {
    // Read wiki JSON files from the specified directory
    const std::string wiki_dir = "/home/spadea/working4/wiki-jsons";

    boost::container::vector<std::string> text_data;
    size_t total_text_length = 0;

    std::string test_name =
        max_text_length > 0
            ? fmt::format("Truncated to {} bytes", max_text_length)
            : "Full text";
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "Loading wiki data from: " << wiki_dir << std::endl;

    for (const auto& entry : std::filesystem::directory_iterator(wiki_dir)) {
        if (entry.path().extension() == ".json") {
            std::ifstream file(entry.path());
            if (!file.is_open()) {
                std::cerr << "Failed to open: " << entry.path() << std::endl;
                continue;
            }

            nlohmann::json json_data = nlohmann::json::parse(file);
            for (const auto& item : json_data) {
                if (item.contains("text")) {
                    std::string text = item["text"].get<std::string>();
                    // Truncate text if max_text_length is specified
                    if (max_text_length > 0 && text.size() > max_text_length) {
                        text = TruncateUtf8(text, max_text_length);
                    }
                    total_text_length += text.size();
                    text_data.push_back(std::move(text));
                }
            }
            std::cout << "  Loaded: " << entry.path().filename()
                      << " (total rows so far: " << text_data.size() << ")"
                      << std::endl;
        }
    }

    const size_t nb = text_data.size();
    double avg_length =
        nb > 0 ? static_cast<double>(total_text_length) / nb : 0;

    std::cout << "\n=== Wiki Data Statistics ===" << std::endl;
    std::cout << "  Total rows: " << nb << std::endl;
    std::cout << "  Total text length: " << total_text_length << " bytes"
              << std::endl;
    std::cout << "  Average text length: " << std::fixed << std::setprecision(2)
              << avg_length << " bytes" << std::endl;
    if (max_text_length > 0) {
        std::cout << "  Max text length (truncated): " << max_text_length
                  << " bytes" << std::endl;
    }

    if (nb == 0) {
        std::cout << "No data loaded, skipping test." << std::endl;
        return;
    }

    // Setup segment and index
    int64_t collection_id = 1;
    int64_t partition_id = 2;
    int64_t segment_id = 3;
    int64_t index_build_id = 4000;
    int64_t index_version = 4000;

    auto schema = std::make_shared<Schema>();
    auto varchar_field_id = schema->AddDebugField("text", DataType::VARCHAR);

    auto field_meta = milvus::segcore::gen_field_meta(collection_id,
                                                      partition_id,
                                                      segment_id,
                                                      varchar_field_id.get(),
                                                      DataType::VARCHAR,
                                                      DataType::NONE,
                                                      false);
    auto index_meta = gen_index_meta(
        segment_id, varchar_field_id.get(), index_build_id, index_version);

    std::string root_path =
        max_text_length > 0
            ? fmt::format("/tmp/test-ngram-wiki-perf-{}b/", max_text_length)
            : "/tmp/test-ngram-wiki-perf/";
    auto storage_config = gen_local_storage_config(root_path);
    auto cm = CreateChunkManager(storage_config);
    auto fs = storage::InitArrowFileSystem(storage_config);

    auto text_field_data =
        storage::CreateFieldData(DataType::VARCHAR, DataType::NONE, false);
    text_field_data->FillFieldData(text_data.data(), text_data.size());

    auto segment = CreateSealedSegment(schema);

    auto text_field_info =
        PrepareSingleFieldInsertBinlog(collection_id,
                                       partition_id,
                                       segment_id,
                                       varchar_field_id.get(),
                                       {text_field_data},
                                       cm);
    segment->LoadFieldData(text_field_info);

    // Build and upload ngram index
    auto payload_reader =
        std::make_shared<milvus::storage::PayloadReader>(text_field_data);
    storage::InsertData insert_data(payload_reader);
    insert_data.SetFieldDataMeta(field_meta);
    insert_data.SetTimestamps(0, 100);
    auto serialized_bytes = insert_data.Serialize(storage::Remote);

    auto log_path = fmt::format("{}/{}/{}/{}/{}",
                                collection_id,
                                partition_id,
                                segment_id,
                                varchar_field_id.get(),
                                0);

    auto cm_w = ChunkManagerWrapper(cm);
    cm_w.Write(log_path, serialized_bytes.data(), serialized_bytes.size());

    storage::FileManagerContext ctx(field_meta, index_meta, cm, fs);
    std::vector<std::string> index_files;

    std::cout << "\nBuilding ngram index..." << std::endl;
    auto build_start = std::chrono::high_resolution_clock::now();
    {
        Config config;
        config[milvus::index::INDEX_TYPE] = milvus::index::INVERTED_INDEX_TYPE;
        config[INSERT_FILES_KEY] = std::vector<std::string>{log_path};

        auto ngram_params = index::NgramParams{
            .loading_index = false,
            .min_gram = 2,
            .max_gram = 3,
        };
        auto index =
            std::make_shared<index::NgramInvertedIndex>(ctx, ngram_params);
        index->Build(config);

        auto create_index_result = index->Upload();
        index_files = create_index_result->GetIndexFiles();
    }
    auto build_end = std::chrono::high_resolution_clock::now();
    auto build_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(build_end -
                                                              build_start)
            .count();
    std::cout << "Index build time: " << build_duration_ms << " ms"
              << std::endl;

    // Load ngram index
    Config config;
    config[milvus::index::INDEX_FILES] = index_files;
    config[milvus::LOAD_PRIORITY] = milvus::proto::common::LoadPriority::HIGH;

    auto ngram_params = index::NgramParams{
        .loading_index = true,
        .min_gram = 2,
        .max_gram = 3,
    };
    auto index = std::make_unique<index::NgramInvertedIndex>(ctx, ngram_params);
    index->Load(milvus::tracer::TraceContext{}, config);

    // Test configuration
    const int warmup_runs = 3;
    const int test_runs = 2;
    std::vector<int> pre_filter_percentages = {20, 10, 5, 2};

    // Search patterns: {literal, op_type}
    // All patterns extracted from wiki-jsons first 50 bytes
    // All Match pattern literals are >= 6 chars
    std::vector<std::pair<std::string, proto::plan::OpType>> search_patterns = {
        // ============ InnerMatch - short patterns (6-10 chars) ============
        {"species", proto::plan::OpType::InnerMatch},   // 4.54%, 7 chars
        {"american", proto::plan::OpType::InnerMatch},  // 3.84%, 8 chars
        {"village", proto::plan::OpType::InnerMatch},   // 2.10%, 7 chars

        // ============ InnerMatch - medium patterns (11-20 chars) ============
        {"is a species", proto::plan::OpType::InnerMatch},   // 4.24%, 12 chars
        {"a species of", proto::plan::OpType::InnerMatch},   // 4.22%, 12 chars
        {"may refer to", proto::plan::OpType::InnerMatch},   // 3.30%, 12 chars
        {"a village in", proto::plan::OpType::InnerMatch},   // 1.02%, 12 chars
        {"also known as", proto::plan::OpType::InnerMatch},  // 1.17%, 13 chars
        {"is a genus of", proto::plan::OpType::InnerMatch},  // 0.93%, 13 chars
        {"in the family", proto::plan::OpType::InnerMatch},  // 0.90%, 13 chars
        {"railway station",
         proto::plan::OpType::InnerMatch},                    // 0.96%, 15 chars
        {"rural locality", proto::plan::OpType::InnerMatch},  // 0.82%, 14 chars
        {"notable people", proto::plan::OpType::InnerMatch},  // 0.61%, 14 chars

        // ============ InnerMatch - long patterns (21-35 chars) ============
        {"is a species of plant",
         proto::plan::OpType::InnerMatch},  // 2.03%, 21 chars
        {"a species of plant in",
         proto::plan::OpType::InnerMatch},  // 1.98%, 21 chars
        {"is a species of beetle",
         proto::plan::OpType::InnerMatch},  // 1.09%, 22 chars
        {"species of plant in the",
         proto::plan::OpType::InnerMatch},  // 1.56%, 23 chars
        {"species of beetle in the",
         proto::plan::OpType::InnerMatch},  // 0.68%, 24 chars
        {"unincorporated community",
         proto::plan::OpType::InnerMatch},  // 0.40%, 24 chars
        {"a species of beetle in the",
         proto::plan::OpType::InnerMatch},  // 0.68%, 26 chars
        {"is an unincorporated community",
         proto::plan::OpType::InnerMatch},  // 0.38%, 30 chars

        // ============ InnerMatch - extra long patterns (36-50 chars) ============
        {"is a species of plant in the family",
         proto::plan::OpType::InnerMatch},  // 0.12%, 35 chars
        {"is a species of beetle in the family",
         proto::plan::OpType::InnerMatch},  // 0.04%, 36 chars

        // ============ Match - short patterns (each literal >= 6 chars) ============
        {"%species%family%", proto::plan::OpType::Match},  // 0.18%, 16 chars
        {"%species%beetle%", proto::plan::OpType::Match},  // 1.16%, 16 chars
        {"%surname%people%", proto::plan::OpType::Match},  // 0.61%, 16 chars

        // ============ Match - medium patterns (each literal >= 6 chars) ============
        {"%railway%station%", proto::plan::OpType::Match},   // 0.96%, 17 chars
        {"%village%district%", proto::plan::OpType::Match},  // 0.22%, 18 chars
        {"%station%located%", proto::plan::OpType::Match},   // 0.45%, 17 chars
        {"%village%located%", proto::plan::OpType::Match},   // 0.13%, 17 chars
        {"%football%season%", proto::plan::OpType::Match},   // 0.04%, 17 chars

        // ============ Match - long patterns (each literal >= 6 chars) ============
        {"%surname%notable%people%",
         proto::plan::OpType::Match},  // 0.56%, 24 chars
        {"%railway%station%located%",
         proto::plan::OpType::Match},  // 0.44%, 25 chars
        {"%unincorporated%community%",
         proto::plan::OpType::Match},  // 0.40%, 26 chars
        {"%species%beetle%family%",
         proto::plan::OpType::Match},  // 0.04%, 23 chars
        {"%village%located%district%",
         proto::plan::OpType::Match},  // 0.03%, 26 chars
    };

    // Helper lambda to create SegmentExpr
    auto create_segment_expr = [&]() {
        return exec::SegmentExpr(std::move(std::vector<exec::ExprPtr>{}),
                                 "SegmentExpr",
                                 nullptr,
                                 segment.get(),
                                 varchar_field_id,
                                 {},
                                 DataType::VARCHAR,
                                 nb,
                                 8192,
                                 0);
    };

    // Helper lambda to create pre_filter bitmap
    auto create_pre_filter = [&](int selectivity_pct) {
        TargetBitmap pre_filter(nb, false);
        std::mt19937 pre_rng(123 + selectivity_pct);
        std::uniform_int_distribution<int> pre_dist(0, 99);
        for (size_t i = 0; i < nb; i++) {
            if (pre_dist(pre_rng) < selectivity_pct) {
                pre_filter[i] = true;
            }
        }
        return pre_filter;
    };

    // Helper to convert OpType to string
    auto op_type_to_string = [](proto::plan::OpType op) -> std::string {
        switch (op) {
            case proto::plan::OpType::InnerMatch:
                return "InnerMatch";
            case proto::plan::OpType::Match:
                return "Match";
            default:
                return "Unknown";
        }
    };

    // Structure to store test results
    struct TestResult {
        std::string pattern;
        std::string op_type;
        int pre_filter_pct;
        double selectivity;
        double avg_ms;
        double speedup;
    };
    std::vector<TestResult> results;

    std::cout << "\n=== Wiki Data Pre-filter Optimization Performance Test ==="
              << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  - Total rows: " << nb << std::endl;
    std::cout << "  - Avg text length: " << avg_length << " bytes" << std::endl;
    std::cout << "  - Warmup runs: " << warmup_runs << std::endl;
    std::cout << "  - Test runs: " << test_runs << std::endl;

    // Warmup runs
    std::cout << "\nWarming up..." << std::endl;
    for (int w = 0; w < warmup_runs; w++) {
        auto seg_expr = create_segment_expr();
        index->ExecuteQuery(search_patterns[0].first,
                            search_patterns[0].second,
                            &seg_expr,
                            nullptr);
    }
    std::cout << "Warmup complete.\n" << std::endl;

    // Test each pattern with each pre_filter selectivity
    for (const auto& [pattern, op_type] : search_patterns) {
        std::string op_str = op_type_to_string(op_type);

        // First, measure baseline (no pre_filter)
        std::cout << "\n============ Pattern: \"" << pattern << "\" (" << op_str
                  << "), Pre-filter: 100% (baseline) ============" << std::endl;
        double baseline_total_ms = 0;
        size_t baseline_count = 0;
        for (int r = 0; r < test_runs; r++) {
            auto seg_expr = create_segment_expr();
            auto start = std::chrono::high_resolution_clock::now();
            auto result =
                index->ExecuteQuery(pattern, op_type, &seg_expr, nullptr);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                      start)
                    .count() /
                1000.0;
            baseline_total_ms += duration_ms;
            if (r == 0) {
                ASSERT_TRUE(result.has_value());
                baseline_count = result->count();
            }
        }
        double baseline_avg_ms = baseline_total_ms / test_runs;
        double baseline_selectivity = 100.0 * baseline_count / nb;

        results.push_back(
            {pattern, op_str, 100, baseline_selectivity, baseline_avg_ms, 1.0});

        // Test with different pre_filter selectivities
        for (int pct : pre_filter_percentages) {
            std::cout << "\n============ Pattern: \"" << pattern << "\" ("
                      << op_str << "), Pre-filter: " << pct
                      << "% ============" << std::endl;
            auto pre_filter = create_pre_filter(pct);

            double total_ms = 0;
            size_t result_count = 0;
            for (int r = 0; r < test_runs; r++) {
                auto seg_expr = create_segment_expr();
                auto start = std::chrono::high_resolution_clock::now();
                auto result = index->ExecuteQuery(
                    pattern, op_type, &seg_expr, &pre_filter);
                auto end = std::chrono::high_resolution_clock::now();
                auto duration_ms =
                    std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          start)
                        .count() /
                    1000.0;
                total_ms += duration_ms;
                if (r == 0) {
                    ASSERT_TRUE(result.has_value());
                    result_count = result->count();
                }
            }
            double avg_ms = total_ms / test_runs;
            double speedup = avg_ms > 0 ? baseline_avg_ms / avg_ms : 0;
            double selectivity = 100.0 * result_count / nb;

            results.push_back(
                {pattern, op_str, pct, selectivity, avg_ms, speedup});
        }
    }

    // Print all results
    std::cout << "\n\n";
    std::cout << "============================================================="
                 "=====================================\n";
    std::cout << "                                        FINAL RESULTS        "
                 "                                    \n";
    std::cout << "============================================================="
                 "=====================================\n";
    std::cout << std::setw(25) << "Pattern" << std::setw(12) << "OpType"
              << std::setw(16) << "PF Selectivity" << std::setw(18)
              << "Like Selectivity" << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Speedup\n";
    std::cout << std::string(95, '-') << "\n";

    std::string last_pattern = "";
    for (const auto& r : results) {
        if (!last_pattern.empty() && last_pattern != r.pattern) {
            std::cout << std::string(95, '-') << "\n";
        }
        last_pattern = r.pattern;

        std::cout << std::setw(25) << r.pattern << std::setw(12) << r.op_type
                  << std::setw(16) << std::to_string(r.pre_filter_pct) + "%"
                  << std::setw(17) << std::fixed << std::setprecision(2)
                  << r.selectivity << "%" << std::setw(12) << std::fixed
                  << std::setprecision(2) << r.avg_ms << std::setw(12)
                  << std::fixed << std::setprecision(2) << r.speedup << "x\n";
    }
    std::cout << std::string(95, '-') << std::endl;
}

TEST(NgramIndex, TestWikiDataPerformance) {
    RunWikiDataPerformanceTest(0);  // 0 means no truncation (full text)
}

TEST(NgramIndex, TestWikiDataPerformance_2000bytes) {
    RunWikiDataPerformanceTest(3500);
}

TEST(NgramIndex, TestWikiDataPerformance_1000bytes) {
    RunWikiDataPerformanceTest(1900);
}

TEST(NgramIndex, TestWikiDataPerformance_500bytes) {
    RunWikiDataPerformanceTest(600);
}

TEST(NgramIndex, TestWikiDataPerformance_100bytes) {
    RunWikiDataPerformanceTest(100);
}

TEST(NgramIndex, TestWikiDataPerformance_50bytes) {
    RunWikiDataPerformanceTest(50);
}