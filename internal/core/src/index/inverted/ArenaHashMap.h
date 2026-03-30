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

#include "SharedArenaHashMap.h"

namespace milvus::index::inverted {

// Convenience wrapper: SharedArenaHashMap + owned MemoryArena.
// Ported from tantivy stacker/src/arena_hashmap.rs
class ArenaHashMap {
 public:
    explicit ArenaHashMap(size_t table_size = 4) : map_(table_size) {
    }

    template <typename V>
    std::optional<V>
    get(const uint8_t* key, size_t key_len) const {
        return map_.get<V>(key, key_len, arena_);
    }

    template <typename V, typename Updater>
    V
    mutate_or_create(const uint8_t* key, size_t key_len, Updater&& updater) {
        return map_.mutate_or_create<V>(
            key, key_len, arena_, std::forward<Updater>(updater));
    }

    template <typename Callback>
    void
    iter(Callback&& callback) const {
        map_.iter(arena_, std::forward<Callback>(callback));
    }

    size_t
    mem_usage() const {
        return map_.mem_usage() + arena_.mem_usage();
    }

    bool
    is_empty() const {
        return map_.is_empty();
    }

    size_t
    len() const {
        return map_.len();
    }

    MemoryArena&
    memory_arena() {
        return arena_;
    }

    const MemoryArena&
    memory_arena() const {
        return arena_;
    }

 private:
    SharedArenaHashMap map_;
    MemoryArena arena_;
};

}  // namespace milvus::index::inverted
