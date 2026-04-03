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

#include <string>
#include <memory>
#include <utility>

#include "index/inverted/query/Query.h"
#include "index/inverted/segment/SegmentReader.h"

namespace milvus::index::inverted {

class TermWeight : public Weight {
 public:
    explicit TermWeight(std::string term) : term_(std::move(term)) {
    }

    Bitset
    execute(const SegmentReader& reader) const override {
        return reader.term_query(reinterpret_cast<const uint8_t*>(term_.data()),
                                 term_.size());
    }

 private:
    std::string term_;
};

class TermQuery : public Query {
 public:
    explicit TermQuery(std::string term) : term_(std::move(term)) {
    }

    std::unique_ptr<Weight>
    weight() const override {
        return std::make_unique<TermWeight>(term_);
    }

 private:
    std::string term_;
};

}  // namespace milvus::index::inverted
