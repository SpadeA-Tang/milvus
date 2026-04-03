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

#include <memory>
#include <utility>
#include <vector>

#include "index/inverted/query/Query.h"
#include "index/inverted/segment/SegmentReader.h"

namespace milvus::index::inverted {

enum class Occur { Must, Should, MustNot };

class BoolWeight : public Weight {
 public:
    BoolWeight(std::vector<std::pair<Occur, std::unique_ptr<Weight>>> clauses)
        : clauses_(std::move(clauses)) {
    }

    Bitset
    execute(const SegmentReader& reader) const override {
        std::vector<const Weight*> must, should, must_not;
        for (const auto& [occur, w] : clauses_) {
            switch (occur) {
                case Occur::Must:
                    must.push_back(w.get());
                    break;
                case Occur::Should:
                    should.push_back(w.get());
                    break;
                case Occur::MustNot:
                    must_not.push_back(w.get());
                    break;
            }
        }

        Bitset result(reader.num_bits());

        if (!must.empty()) {
            result = must[0]->execute(reader);
            for (size_t i = 1; i < must.size(); i++) {
                result &= must[i]->execute(reader);
            }
        } else if (!should.empty()) {
            result = should[0]->execute(reader);
            for (size_t i = 1; i < should.size(); i++) {
                result |= should[i]->execute(reader);
            }
            should.clear();  // already consumed
        }

        // When must is present, should clauses only affect scoring
        // (no scoring in Phase 1, so they are ignored).

        for (const auto* w : must_not) {
            result &= ~w->execute(reader);
        }

        return result;
    }

 private:
    std::vector<std::pair<Occur, std::unique_ptr<Weight>>> clauses_;
};

class BoolQuery : public Query {
 public:
    void
    add(Occur occur, std::unique_ptr<Query> sub_query) {
        clauses_.push_back({occur, std::move(sub_query)});
    }

    std::unique_ptr<Weight>
    weight() const override {
        std::vector<std::pair<Occur, std::unique_ptr<Weight>>> weights;
        for (const auto& [occur, q] : clauses_) {
            weights.push_back({occur, q->weight()});
        }
        return std::make_unique<BoolWeight>(std::move(weights));
    }

 private:
    std::vector<std::pair<Occur, std::unique_ptr<Query>>> clauses_;
};

}  // namespace milvus::index::inverted
