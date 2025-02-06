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

#include "RandomSampleNode.h"

namespace milvus {
namespace exec {
PhyRandomSampleNode::PhyRandomSampleNode(
    int32_t operator_id,
    DriverContext* ctx,
    const std::shared_ptr<const plan::RandomSampleNode>& random_sample_node)
    : Operator(ctx,
               random_sample_node->output_type(),
               operator_id,
               random_sample_node->id(),
               "PhyRandomSampleNode"),
      factor_(random_sample_node->factor()) {
    // We should intercept unexpected number of factor at proxy level, so it's impossible to trigger
    // the panic here theoretically.
    AssertInfo(
        factor_ > 0.0 && factor_ < 1.0, "Unexpected sample factor {}", factor_);
    active_count_ = operator_context_->get_exec_context()
                        ->get_query_context()
                        ->get_active_count();
}

void
PhyRandomSampleNode::AddInput(RowVectorPtr& input) {
    PanicInfo(UnexpectedError,
              "PhyRandomSampleNode should be the leaf source node");
}

FixedVector<uint32_t>
PhyRandomSampleNode::HashsetSample(const uint32_t N,
                                   const uint32_t M,
                                   std::mt19937& gen) {
    std::uniform_int_distribution<> dis(0, N - 1);
    std::unordered_set<uint32_t> sampled;
    sampled.reserve(N);
    while (sampled.size() < M) {
        sampled.insert(dis(gen));
    }

    return FixedVector<uint32_t>(sampled.begin(), sampled.end());
}

FixedVector<uint32_t>
PhyRandomSampleNode::StandardSample(const uint32_t N,
                                    const uint32_t M,
                                    std::mt19937& gen) {
    FixedVector<uint32_t> inputs(N);
    FixedVector<uint32_t> outputs(M);
    std::iota(inputs.begin(), inputs.end(), 0);

    std::sample(inputs.begin(), inputs.end(), outputs.begin(), M, gen);
    return outputs;
}

FixedVector<uint32_t>
PhyRandomSampleNode::Sample(const uint32_t N, const float factor) {
    const uint32_t M = N * factor;
    std::random_device rd;
    std::mt19937 gen(rd());
    float epsilon = std::numeric_limits<float>::epsilon();
    // It's derived from some experiments
    if (factor <= 0.02 + epsilon ||
        (N <= 10000000 && factor <= 0.045 + epsilon) ||
        (N <= 60000000 && factor <= 0.025 + epsilon)) {
        return HashsetSample(N, M, gen);
    }
    return StandardSample(N, M, gen);
}

RowVectorPtr
PhyRandomSampleNode::GetOutput() {
    if (is_finished_) {
        return nullptr;
    }

    if (active_count_ == 0) {
        is_finished_ = true;
        return nullptr;
    }

    auto col_input = std::make_shared<ColumnVector>(
        TargetBitmap(active_count_), TargetBitmap(active_count_));
    TargetBitmapView data(col_input->GetRawData(), col_input->size());
    // True in TargetBitmap means we don't want this row, while for readability, we set the relevant row be true
    // if it's sampled. So we need to flip the bits at last.
    // However, if sample rate is larger than 0.5, we use 1-factor for sampling so that in some cases, the sampling
    // performance would be better. In that case, we don't need to flip at last.
    bool need_flip = true;
    float factor = factor_;
    if (factor > 0.5) {
        need_flip = false;
        factor = 1.0 - factor;
    }
    auto sampled = Sample(active_count_, factor);
    for (auto n : sampled) {
        data[n] = true;
    }

    is_finished_ = true;
    if (need_flip) {
        data.flip();
    }
    return std::make_shared<RowVector>(std::vector<VectorPtr>{col_input});
}

bool
PhyRandomSampleNode::IsFinished() {
    return is_finished_;
}

}  // namespace exec
}  // namespace milvus