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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

namespace milvus::index::inverted {

struct CacheKey {
    uint64_t segment_id;
    uint32_t file_type;  // 0=idx, 1=dct, 2=pst
    uint64_t offset;

    bool
    operator==(const CacheKey& other) const {
        return segment_id == other.segment_id && file_type == other.file_type &&
               offset == other.offset;
    }
};

struct CacheKeyHash {
    size_t
    operator()(const CacheKey& k) const {
        size_t h = std::hash<uint64_t>{}(k.segment_id);
        h ^= std::hash<uint32_t>{}(k.file_type) + 0x9e3779b9 + (h << 6) +
             (h >> 2);
        h ^= std::hash<uint64_t>{}(k.offset) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct CacheEntry {
    std::shared_ptr<void> data;
    size_t size_bytes;
};

using CacheHandle = std::shared_ptr<void>;

class BlockCache {
 public:
    virtual CacheEntry
    get_or_load(const CacheKey& key, std::function<CacheEntry()> loader) = 0;
    virtual void
    evict_to(size_t target_bytes) = 0;
    virtual size_t
    memory_usage() const = 0;
    virtual ~BlockCache() = default;
};

// Phase 1: no caching, every call goes to disk.
// Same interface so SegmentReader code won't change in Phase 2.
class PassthroughCache : public BlockCache {
 public:
    CacheEntry
    get_or_load(const CacheKey& /*key*/,
                std::function<CacheEntry()> loader) override {
        return loader();
    }

    void
    evict_to(size_t /*target_bytes*/) override {
    }

    size_t
    memory_usage() const override {
        return 0;
    }
};

}  // namespace milvus::index::inverted
