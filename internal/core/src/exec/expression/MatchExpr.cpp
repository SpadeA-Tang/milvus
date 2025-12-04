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

#include "MatchExpr.h"
#include <utility>
#include "common/Types.h"

namespace milvus {
namespace exec {

PhyMatchFilterExpr::PhyMatchFilterExpr(
    std::vector<std::shared_ptr<Expr>> input,
    const std::shared_ptr<const milvus::expr::MatchExpr>& expr,
    const std::string& name,
    milvus::OpContext* op_ctx,
    const segcore::SegmentInternalInterface* segment,
    int64_t active_count,
    int64_t batch_size,
    int32_t consistency_level)
    : SegmentExpr(std::move(input),
                  name,
                  op_ctx,
                  segment,
                  FieldId(-1),
                  {},
                  DataType::BOOL,
                  active_count,
                  batch_size,
                  consistency_level,
                  false,
                  true),
      expr_(expr) {
}

void
PhyMatchFilterExpr::Eval(EvalCtx& context, VectorPtr& result) {
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
    valid.set();

    // Get serialized predicate proto bytes
    const std::string& predicate_bytes = expr_->get_predicate_proto_bytes();

    // TODO: Call Tantivy FFI to execute nested query
    // 1. Get Tantivy nested index reader from segment
    // 2. Call tantivy_search_nested(reader, predicate_bytes.data(), predicate_bytes.size(), ...)
    // 3. Convert matching row IDs to bitmap

    // For now, return all false since Tantivy integration is not complete
    for (int i = 0; i < real_batch_size; i++) {
        res[i] = false;
    }

    result = res_vec;
}

}  // namespace exec
}  // namespace milvus
