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

#pragma once

#include "indexbuilder/IndexCreatorBase.h"

namespace milvus::indexbuilder {

class NestedIndexCreator : public IndexCreatorBase {
 public:
    explicit NestedIndexCreator(
        Config& config,
        const storage::FileManagerContext& file_manager_context);

    void
    Build(const milvus::DatasetPtr& dataset) override;

    void
    Build() override;

    milvus::BinarySet
    Serialize() override;

    void
    Load(const milvus::BinarySet& binary_set) override;

    index::IndexStatsPtr
    Upload() override;

 private:
    milvus::index::IndexBasePtr index_ = nullptr;
};

using NestedIndexCreatorPtr = std::unique_ptr<NestedIndexCreator>;

inline NestedIndexCreatorPtr
CreateNestedIndexCreator(
    Config& config, const storage::FileManagerContext& file_manager_context) {
    return std::make_unique<NestedIndexCreator>(config, file_manager_context);
}

}  // namespace milvus::indexbuilder