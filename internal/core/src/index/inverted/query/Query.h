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

#include <boost/dynamic_bitset.hpp>

namespace milvus::index::inverted {

using Bitset = boost::dynamic_bitset<>;

class SegmentReader;

// Weight: Query bound to a segment, produces a bitset of matching doc_ids.
class Weight {
 public:
    virtual ~Weight() = default;
    virtual Bitset
    execute(const SegmentReader& reader) const = 0;
};

// Query: segment-independent query description.
class Query {
 public:
    virtual ~Query() = default;
    virtual std::unique_ptr<Weight>
    weight() const = 0;
};

}  // namespace milvus::index::inverted
