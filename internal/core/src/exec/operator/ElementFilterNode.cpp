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

#include "ElementFilterNode.h"
#include "common/Tracer.h"
#include "common/ElementFilterIterator.h"
#include "monitor/Monitor.h"

namespace milvus {
namespace exec {

PhyElementFilterNode::PhyElementFilterNode(
    int32_t operator_id,
    DriverContext* driverctx,
    const std::shared_ptr<const plan::ElementFilterNode>& element_filter_node)
    : Operator(driverctx,
               element_filter_node->output_type(),
               operator_id,
               element_filter_node->id(),
               "PhyElementFilterNode") {
    ExecContext* exec_context = operator_context_->get_exec_context();
    query_context_ = exec_context->get_query_context();
    std::vector<expr::TypedExprPtr> exprs;
    exprs.emplace_back(element_filter_node->element_filter());
    element_exprs_ = std::make_unique<ExprSet>(exprs, exec_context);
}

void
PhyElementFilterNode::AddInput(RowVectorPtr& input) {
    input_ = std::move(input);
}

void
PhyElementFilterNode::LoadArrayOffsets() {
    // Check if already loaded
    if (query_context_->get_array_offsets() != nullptr) {
        return;
    }

    // Load array offsets from segment
    const auto* segment = query_context_->get_segment();

    // POC: BuildFromSegment currently assumes 3 elements per document
    // TODO: Implement full segment interface extension to read actual offsets
    auto array_offsets = std::make_shared<ArrayOffsets>(
        ArrayOffsets::BuildFromSegment(segment, struct_name_));

    query_context_->set_array_offsets(array_offsets);
}

RowVectorPtr
PhyElementFilterNode::GetOutput() {
    if (is_finished_ || !no_more_input_) {
        return nullptr;
    }

    tracer::AutoSpan span(
        "PhyElementFilterNode::GetOutput", tracer::GetRootSpan(), true);

    DeferLambda([&]() { is_finished_ = true; });

    if (input_ == nullptr) {
        return nullptr;
    }

    std::chrono::high_resolution_clock::time_point start_time =
        std::chrono::high_resolution_clock::now();

    // ========== Step 1: Get search result with iterators ==========
    milvus::SearchResult search_result = query_context_->get_search_result();

    if (!search_result.is_element_level_) {
        ThrowInfo(ExprInvalid,
                  "PhyElementFilterNode expects element-level search result");
    }

    if (!search_result.vector_iterators_.has_value()) {
        ThrowInfo(
            ExprInvalid,
            "PhyElementFilterNode expects vector_iterators in search result");
    }

    LoadArrayOffsets();

    // ========== Step 2: Wrap each iterator with ElementFilterIterator ==========
    // For element-level search, iterators should be in vector_iterators_
    // (they will be moved to element_iterators_ downstream if needed)
    auto& base_iterators = search_result.vector_iterators_.value();
    std::vector<std::shared_ptr<VectorIterator>> wrapped_iterators;
    wrapped_iterators.reserve(base_iterators.size());

    ExecContext* exec_context = operator_context_->get_exec_context();

    for (auto& base_iter : base_iterators) {
        // Wrap each iterator with ElementFilterIterator
        // ElementFilterIterator inherits from VectorIterator, so can be stored directly
        // Note: element_exprs_ contains the element-level filter expression
        auto wrapped_iter = std::make_shared<ElementFilterIterator>(
            base_iter, exec_context, element_exprs_.get(), 256  // batch_size
        );

        wrapped_iterators.push_back(wrapped_iter);
    }

    // ========== Step 3: Update search result with wrapped iterators ==========
    // Keep them in vector_iterators_ (element-level flag is already set)
    search_result.vector_iterators_ = std::move(wrapped_iterators);
    query_context_->set_search_result(std::move(search_result));

    // ========== Step 4: Record metrics ==========
    std::chrono::high_resolution_clock::time_point end_time =
        std::chrono::high_resolution_clock::now();
    double cost =
        std::chrono::duration<double, std::micro>(end_time - start_time)
            .count();

    tracer::AddEvent(
        fmt::format("PhyElementFilterNode: wrapped {} iterators, struct_name: "
                    "{}, cost_us: {}",
                    wrapped_iterators.size(),
                    struct_name_,
                    cost));

    // Pass through input to downstream
    return input_;
}

}  // namespace exec
}  // namespace milvus
