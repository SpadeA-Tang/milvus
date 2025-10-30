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
#include "common/Types.h"
#include "query/PlanProto.h"

namespace milvus {
namespace exec {

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

    // Evaluate the predicate expression
    // The predicate contains column references to struct sub-fields (via nested_path)
    // which will be loaded from the segment automatically
    VectorPtr predicate_result;
    predicate_expr_->Eval(context, predicate_result);

    // The predicate should return a boolean vector
    AssertInfo(predicate_result != nullptr,
               "MATCH_ALL predicate evaluation returned null");
    AssertInfo(predicate_result->size() == real_batch_size,
               "MATCH_ALL predicate result size mismatch");

    // Convert the predicate result to our output format
    auto res_vec = std::make_shared<ColumnVector>(
        TargetBitmap(real_batch_size), TargetBitmap(real_batch_size));
    TargetBitmapView res(res_vec->GetRawData(), real_batch_size);
    TargetBitmapView valid(res_vec->GetValidRawData(), real_batch_size);

    // Copy the predicate results
    auto pred_data = predicate_result->GetRawData();
    auto pred_valid = predicate_result->GetValidRawData();
    for (int i = 0; i < real_batch_size; i++) {
        res[i] = pred_data[i];
        valid[i] = pred_valid[i];
    }

    result = res_vec;
}

}  // namespace exec
}  // namespace milvus
