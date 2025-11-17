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

#include "ElementFilterBitsNode.h"
#include "common/Tracer.h"

namespace milvus {
namespace exec {

PhyElementFilterBitsNode::PhyElementFilterBitsNode(
    int32_t operator_id,
    DriverContext* driverctx,
    const expr::TypedExprPtr& element_expr,
    const std::string& struct_name)
    : Operator(driverctx,
               DataType::NONE,  // TODO: Define proper output type
               operator_id,
               "element_filter_bits_plan_node",  // TODO: Get from plan node
               "PhyElementFilterBitsNode"),
      struct_name_(struct_name) {
    ExecContext* exec_context = operator_context_->get_exec_context();
    query_context_ = exec_context->get_query_context();

    // Build expression set from element-level expression
    std::vector<expr::TypedExprPtr> exprs;
    exprs.emplace_back(element_expr);
    element_exprs_ = std::make_unique<ExprSet>(exprs, exec_context);
}

void
PhyElementFilterBitsNode::AddInput(RowVectorPtr& input) {
    input_ = std::move(input);
}

RowVectorPtr
PhyElementFilterBitsNode::GetOutput() {
    if (is_finished_ || !no_more_input_) {
        return nullptr;
    }

    tracer::AutoSpan span(
        "PhyElementFilterBitsNode::GetOutput", tracer::GetRootSpan(), true);

    DeferLambda([&]() { is_finished_ = true; });

    if (input_ == nullptr) {
        return nullptr;
    }

    // ========== Step 1: Extract doc-level bitset from input ==========
    TargetBitmap doc_bitset = ExtractDocBitset();

    auto segment = query_context_->get_segment();
    auto field_meta = milvus::FindFirstArrayFieldInStruct(segment->get_schema(),
                                                          struct_name_);
    auto field_id = field_meta.get_id();
    auto array_offsets = segment->GetArrayOffsets(field_id);
    if (array_offsets == nullptr) {
        ThrowInfo(ErrorCode::UnexpectedError,
                  "ArrayOffsets not found for field {}",
                  field_id.get());
    }
    query_context_->set_array_offsets(array_offsets);

    // ========== Step 3: Convert to element-level bitset ==========
    TargetBitmap element_bitset = DocBitsetToElementBitset(doc_bitset);

    // ========== Step 4: Evaluate element-level expression ==========
    TargetBitmap expr_result = EvaluateElementExpression(element_bitset);

    // ========== Step 5: Combine results (AND operation) ==========
    // Only elements that pass both doc-level and element-level filters
    element_bitset &= expr_result;

    // ========== Step 6: Write to QueryContext ==========
    query_context_->set_element_level_bitset(std::move(element_bitset));
    query_context_->set_is_element_level_query(true);
    query_context_->set_struct_name(struct_name_);

    tracer::AddEvent(fmt::format("struct_name: {}, total_elements: {}",
                                 struct_name_,
                                 array_offsets->GetTotalElementCount()));

    // ========== Step 7: Return input (passthrough) ==========
    return input_;
}

// =============================================================================
// Private Methods - Implementation Stubs
// =============================================================================

TargetBitmap
PhyElementFilterBitsNode::ExtractDocBitset() {
    // TODO: Implement doc bitset extraction from input_ RowVector
    //
    // The doc-level bitset should be encoded in the input RowVector.
    // This depends on how FilterBitsNode/MvccNode encodes the bitset.
    //
    // POC implementation: Assume all docs pass (all bits set to true)
    int64_t doc_count = query_context_->get_active_count();
    TargetBitmap bitset(doc_count);
    bitset.set();  // All docs pass for now

    return bitset;
}

TargetBitmap
PhyElementFilterBitsNode::DocBitsetToElementBitset(
    const TargetBitmap& doc_bitset) {
    auto array_offsets = query_context_->get_array_offsets();
    AssertInfo(array_offsets != nullptr, "Array offsets not available");

    int64_t total_elements = array_offsets->GetTotalElementCount();
    int64_t doc_count = array_offsets->GetDocCount();

    AssertInfo(doc_bitset.size() == doc_count,
               "Doc bitset size mismatch: {} vs {}",
               doc_bitset.size(),
               doc_count);

    TargetBitmap element_bitset(total_elements);

    for (int64_t elem_id = 0; elem_id < total_elements; ++elem_id) {
        int32_t doc_id = array_offsets->element_info[elem_id].first;
        if (doc_bitset[doc_id]) {
            element_bitset[elem_id] = true;
        }
    }

    return element_bitset;
}

TargetBitmap
PhyElementFilterBitsNode::EvaluateElementExpression(
    const TargetBitmap& valid_elements) {
    auto array_offsets = query_context_->get_array_offsets();
    int64_t total_elements = array_offsets->GetTotalElementCount();

    // TODO: Implement element-level expression evaluation
    //
    // This requires:
    // 1. EvalCtx extension to support element-level evaluation mode
    // 2. Expr extension to handle array[sub_field] syntax
    // 3. Segment interface to provide element-level data access
    //
    // POC implementation: Assume all valid elements pass the expression
    // Create result bitmap and copy valid_elements
    TargetBitmap result(total_elements);
    for (int64_t i = 0; i < total_elements; ++i) {
        result[i] = valid_elements[i];
    }

    return result;
}

}  // namespace exec
}  // namespace milvus
