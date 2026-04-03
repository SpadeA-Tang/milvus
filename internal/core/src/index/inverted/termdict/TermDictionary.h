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

#include <cstdint>
#include <optional>
#include <string_view>

#include "index/inverted/postings/PostingFormat.h"

namespace milvus::index::inverted {

// --- TermDictionaryWriter ---
// Interface for building a term dictionary.
// Terms must be inserted in lexicographic order.

class TermDictionaryWriter {
 public:
    virtual void
    insert_key(std::string_view key) = 0;

    virtual void
    insert_value(const PostingsInfo& info) = 0;

    virtual void
    finish() = 0;

    virtual ~TermDictionaryWriter() = default;
};

// --- TermDictionaryReader ---
// Interface for reading a term dictionary.

class TermDictionaryReader {
 public:
    virtual std::optional<PostingsInfo>
    lookup(const uint8_t* token, size_t token_len) = 0;

    virtual ~TermDictionaryReader() = default;
};

}  // namespace milvus::index::inverted
