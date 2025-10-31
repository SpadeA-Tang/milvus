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

#include "MatchAllExpr.h"
#include <utility>
#include <unordered_set>
#include "common/Types.h"
#include "query/PlanProto.h"
#include "exec/expression/CompareExpr.h"
#include "exec/expression/LogicalBinaryExpr.h"
#include "exec/expression/ColumnExpr.h"
#include "exec/expression/ValueExpr.h"
#include "exec/expression/TantivyNestedFFI.h"

namespace milvus {
namespace exec {

// Forward declarations
static proto::plan::TantivyQueryExpr
ConvertPhyExprToTantivyQuery(const std::shared_ptr<Expr>& expr);

// Helper: Extract field name from nested_path (e.g., ["intField"] -> "intField")
static std::string
ExtractFieldNameFromNestedPath(const std::vector<std::string>& nested_path) {
    if (nested_path.empty()) {
        return "";
    }
    // For MATCH_ALL, nested_path[0] is the sub-field name
    return nested_path[0];
}

// Helper: Extract value from PhyValueExpr
static bool
ExtractValueI64(const std::shared_ptr<Expr>& expr, int64_t& value) {
    auto value_expr = std::dynamic_pointer_cast<PhyValueExpr>(expr);
    if (!value_expr) {
        return false;
    }

    auto logical_expr = value_expr->GetLogicalExpr();
    auto val_expr = std::dynamic_pointer_cast<const milvus::expr::ValueExpr>(logical_expr);
    if (!val_expr) {
        return false;
    }

    // Try to extract i64 value
    if (val_expr->val_.val_case() == proto::plan::GenericValue::kInt64Val) {
        value = val_expr->val_.int64_val();
        return true;
    }
    return false;
}

static bool
ExtractValueF64(const std::shared_ptr<Expr>& expr, double& value) {
    auto value_expr = std::dynamic_pointer_cast<PhyValueExpr>(expr);
    if (!value_expr) {
        return false;
    }

    auto logical_expr = value_expr->GetLogicalExpr();
    auto val_expr = std::dynamic_pointer_cast<const milvus::expr::ValueExpr>(logical_expr);
    if (!val_expr) {
        return false;
    }

    if (val_expr->val_.val_case() == proto::plan::GenericValue::kFloatVal) {
        value = val_expr->val_.float_val();
        return true;
    }
    return false;
}

static bool
ExtractValueBool(const std::shared_ptr<Expr>& expr, bool& value) {
    auto value_expr = std::dynamic_pointer_cast<PhyValueExpr>(expr);
    if (!value_expr) {
        return false;
    }

    auto logical_expr = value_expr->GetLogicalExpr();
    auto val_expr = std::dynamic_pointer_cast<const milvus::expr::ValueExpr>(logical_expr);
    if (!val_expr) {
        return false;
    }

    if (val_expr->val_.val_case() == proto::plan::GenericValue::kBoolVal) {
        value = val_expr->val_.bool_val();
        return true;
    }
    return false;
}

static bool
ExtractValueString(const std::shared_ptr<Expr>& expr, std::string& value) {
    auto value_expr = std::dynamic_pointer_cast<PhyValueExpr>(expr);
    if (!value_expr) {
        return false;
    }

    auto logical_expr = value_expr->GetLogicalExpr();
    auto val_expr = std::dynamic_pointer_cast<const milvus::expr::ValueExpr>(logical_expr);
    if (!val_expr) {
        return false;
    }

    if (val_expr->val_.val_case() == proto::plan::GenericValue::kStringVal) {
        value = val_expr->val_.string_val();
        return true;
    }
    return false;
}

// Convert CompareExpr to TantivyFieldCondition
static proto::plan::TantivyFieldCondition
ConvertCompareExprToFieldCondition(const std::shared_ptr<PhyCompareFilterExpr>& compare_expr) {
    proto::plan::TantivyFieldCondition condition;

    // Get logical expression to access operator type
    auto logical_expr = compare_expr->GetLogicalExpr();
    auto compare_logical = std::dynamic_pointer_cast<const milvus::expr::CompareExpr>(logical_expr);
    AssertInfo(compare_logical, "CompareExpr logical expression is null");

    auto op_type = compare_logical->op_type_;

    // Get inputs (left and right operands)
    auto& inputs = compare_expr->GetInputsRef();
    AssertInfo(inputs.size() == 2, "CompareExpr should have exactly 2 inputs");

    // Left should be ColumnExpr, right should be ValueExpr
    auto left_column = std::dynamic_pointer_cast<PhyColumnExpr>(inputs[0]);
    AssertInfo(left_column, "Left operand of CompareExpr should be ColumnExpr");

    auto column_info = left_column->GetColumnInfo();
    AssertInfo(column_info.has_value(), "ColumnExpr should have column info");

    std::string field_name = ExtractFieldNameFromNestedPath(column_info->nested_path_);
    condition.set_field_name(field_name);

    auto right_value = inputs[1];

    // Handle different operator types
    if (op_type == proto::plan::OpType::Equal) {
        // Term query
        int64_t i64_val;
        double f64_val;
        bool bool_val;
        std::string str_val;

        if (ExtractValueI64(right_value, i64_val)) {
            condition.set_term_i64(i64_val);
        } else if (ExtractValueF64(right_value, f64_val)) {
            condition.set_term_f64(f64_val);
        } else if (ExtractValueBool(right_value, bool_val)) {
            condition.set_term_bool(bool_val);
        } else if (ExtractValueString(right_value, str_val)) {
            condition.set_term_keyword(str_val);
        } else {
            ThrowInfo(DataTypeInvalid, "Unsupported value type in CompareExpr");
        }
    } else if (op_type == proto::plan::OpType::GreaterThan ||
               op_type == proto::plan::OpType::GreaterEqual ||
               op_type == proto::plan::OpType::LessThan ||
               op_type == proto::plan::OpType::LessEqual) {
        // Range query
        int64_t i64_val;
        double f64_val;
        std::string str_val;

        if (ExtractValueI64(right_value, i64_val)) {
            auto* range = condition.mutable_range_i64();
            if (op_type == proto::plan::OpType::GreaterThan) {
                range->set_lower(i64_val);
                range->set_lower_inclusive(false);
            } else if (op_type == proto::plan::OpType::GreaterEqual) {
                range->set_lower(i64_val);
                range->set_lower_inclusive(true);
            } else if (op_type == proto::plan::OpType::LessThan) {
                range->set_upper(i64_val);
                range->set_upper_inclusive(false);
            } else {  // LessEqual
                range->set_upper(i64_val);
                range->set_upper_inclusive(true);
            }
        } else if (ExtractValueF64(right_value, f64_val)) {
            auto* range = condition.mutable_range_f64();
            if (op_type == proto::plan::OpType::GreaterThan) {
                range->set_lower(f64_val);
                range->set_lower_inclusive(false);
            } else if (op_type == proto::plan::OpType::GreaterEqual) {
                range->set_lower(f64_val);
                range->set_lower_inclusive(true);
            } else if (op_type == proto::plan::OpType::LessThan) {
                range->set_upper(f64_val);
                range->set_upper_inclusive(false);
            } else {  // LessEqual
                range->set_upper(f64_val);
                range->set_upper_inclusive(true);
            }
        } else if (ExtractValueString(right_value, str_val)) {
            auto* range = condition.mutable_range_keyword();
            if (op_type == proto::plan::OpType::GreaterThan) {
                range->set_lower(str_val);
                range->set_lower_inclusive(false);
            } else if (op_type == proto::plan::OpType::GreaterEqual) {
                range->set_lower(str_val);
                range->set_lower_inclusive(true);
            } else if (op_type == proto::plan::OpType::LessThan) {
                range->set_upper(str_val);
                range->set_upper_inclusive(false);
            } else {  // LessEqual
                range->set_upper(str_val);
                range->set_upper_inclusive(true);
            }
        } else {
            ThrowInfo(DataTypeInvalid, "Unsupported value type in range CompareExpr");
        }
    } else {
        ThrowInfo(OpTypeInvalid, fmt::format("Unsupported OpType for Tantivy: {}", op_type));
    }

    return condition;
}

// Recursive conversion function
static proto::plan::TantivyQueryExpr
ConvertPhyExprToTantivyQuery(const std::shared_ptr<Expr>& expr) {
    proto::plan::TantivyQueryExpr query_expr;

    // Try CompareExpr
    if (auto compare_expr = std::dynamic_pointer_cast<PhyCompareFilterExpr>(expr)) {
        auto* condition = query_expr.mutable_condition();
        *condition = ConvertCompareExprToFieldCondition(compare_expr);
        return query_expr;
    }

    // Try LogicalBinaryExpr
    if (auto logical_expr = std::dynamic_pointer_cast<PhyLogicalBinaryExpr>(expr)) {
        auto* logical = query_expr.mutable_logical();

        // Get logical expression to access operator type
        auto logical_typed_expr = logical_expr->GetLogicalExpr();
        auto logical_binary = std::dynamic_pointer_cast<const milvus::expr::LogicalBinaryExpr>(logical_typed_expr);
        AssertInfo(logical_binary, "LogicalBinaryExpr logical expression is null");

        // Set operator
        if (logical_binary->op_type_ == milvus::expr::LogicalBinaryExpr::OpType::And) {
            logical->set_op(proto::plan::TantivyLogicalExpr_Op_AND);
        } else if (logical_binary->op_type_ == milvus::expr::LogicalBinaryExpr::OpType::Or) {
            logical->set_op(proto::plan::TantivyLogicalExpr_Op_OR);
        } else {
            ThrowInfo(OpTypeInvalid, "Unsupported LogicalBinaryExpr OpType for Tantivy");
        }

        // Recursively convert children
        for (auto& child : logical_expr->GetInputsRef()) {
            auto* child_expr = logical->add_children();
            *child_expr = ConvertPhyExprToTantivyQuery(child);
        }

        return query_expr;
    }

    ThrowInfo(ExprInvalid, fmt::format("Unsupported expression type for Tantivy conversion: {}", expr->name()));
}

PhyMatchAllFilterExpr::PhyMatchAllFilterExpr(
    const std::vector<std::shared_ptr<Expr>>& input,
    const std::shared_ptr<const milvus::expr::MatchAllExpr>& expr,
    const std::string& name,
    milvus::OpContext* op_ctx,
    const segcore::SegmentInternalInterface* segment,
    int64_t active_count,
    int64_t batch_size,
    int32_t consistency_level)
    : SegmentExpr(input,
                  name,
                  op_ctx,
                  segment,
                  FieldId(expr->get_column_info().field_id()),
                  {},  // nested_path
                  DataType::BOOL,  // output type
                  active_count,
                  batch_size,
                  consistency_level,
                  false,  // use_index
                  true),  // is_optional
      expr_(expr) {
    // The predicate expression has already been compiled by CompileInputs
    // and passed in the input vector (since we added it to inputs_ in MatchAllExpr)
    AssertInfo(input.size() == 1,
               "MatchAllExpr should have exactly one input (the predicate)");
    predicate_expr_ = input[0];
}

void
PhyMatchAllFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
    // Get batch size
    auto input = context.get_offset_input();
    SetHasOffsetInput((input != nullptr));
    auto real_batch_size = has_offset_input_ ? context.get_offset_input()->size()
                                              : GetNextBatchSize();
    if (real_batch_size == 0) {
        result = nullptr;
        return;
    }

    // Create result bitmap
    auto res_vec = std::make_shared<ColumnVector>(
        TargetBitmap(real_batch_size), TargetBitmap(real_batch_size));
    TargetBitmapView res(res_vec->GetRawData(), real_batch_size);
    TargetBitmapView valid(res_vec->GetValidRawData(), real_batch_size);
    valid.set();  // All results are valid

    // Get struct field information
    auto field_id = expr_->get_column_info().field_id();

    // TODO: Get Tantivy nested index reader from segment
    // This requires extending the segment interface to:
    // 1. Check if a field has a Tantivy nested index
    // 2. Get the IndexReaderNestedWrapper pointer
    //
    // For now, we check if Tantivy nested index is available
    // If not available, fall back to returning false (no matches)

    void* tantivy_reader = nullptr;  // TODO: segment->GetTantivyNestedReader(field_id);

    if (tantivy_reader == nullptr) {
        // No Tantivy nested index available, return no matches
        // This is a safe default - won't incorrectly match any data
        for (int i = 0; i < real_batch_size; i++) {
            res[i] = false;
        }
        result = res_vec;
        return;
    }

    // Convert predicate expression tree to TantivyQueryExpr protobuf
    proto::plan::TantivyQueryExpr tantivy_query;
    try {
        tantivy_query = ConvertPhyExprToTantivyQuery(predicate_expr_);
    } catch (const std::exception& e) {
        ThrowInfo(ExprInvalid,
                  fmt::format("Failed to convert expression to Tantivy query: {}", e.what()));
    }

    // Serialize protobuf to bytes
    std::string query_bytes;
    if (!tantivy_query.SerializeToString(&query_bytes)) {
        ThrowInfo(ExprInvalid, "Failed to serialize Tantivy query protobuf");
    }

    // Call Tantivy FFI to execute nested query
    int64_t* result_row_ids = nullptr;
    size_t result_count = 0;

    char* error = tantivy_search_nested(
        tantivy_reader,
        reinterpret_cast<const uint8_t*>(query_bytes.data()),
        query_bytes.size(),
        &result_row_ids,
        &result_count);

    if (error != nullptr) {
        std::string error_msg(error);
        tantivy_free_error(error);
        ThrowInfo(ExprInvalid,
                  fmt::format("Tantivy nested query failed: {}", error_msg));
    }

    // Convert row_ids to a set for fast lookup
    std::unordered_set<int64_t> matching_rows;
    if (result_row_ids != nullptr && result_count > 0) {
        matching_rows.insert(result_row_ids, result_row_ids + result_count);
        tantivy_free_row_ids(result_row_ids, result_count);
    }

    // Fill result bitmap
    // For each row in the current batch, check if it matches
    auto current_pos = GetCurrentRows();
    for (int i = 0; i < real_batch_size; i++) {
        int64_t row_id = has_offset_input_ ? (*input)[i] : (current_pos + i);
        res[i] = matching_rows.count(row_id) > 0;
    }

    result = res_vec;
}

}  // namespace exec
}  // namespace milvus
