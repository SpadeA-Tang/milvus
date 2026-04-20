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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

#include "index/inverted/indexing/MemoryArena.h"

namespace milvus::index::inverted {

// MurmurHash2 (32-bit, seed=0).
// Matches tantivy's murmurhash32::murmurhash2.
inline uint32_t
murmurhash2(const uint8_t* key, size_t len) {
    constexpr uint32_t m = 0x5bd1e995;
    constexpr int r = 24;
    uint32_t h = static_cast<uint32_t>(len);  // seed=0, so h = 0 ^ len

    while (len >= 4) {
        uint32_t k;
        std::memcpy(&k, key, 4);
        k *= m;
        k ^= k >> r;
        k *= m;
        h *= m;
        h ^= k;
        key += 4;
        len -= 4;
    }

    switch (len) {
        case 3:
            h ^= static_cast<uint32_t>(key[2]) << 16;
            [[fallthrough]];
        case 2:
            h ^= static_cast<uint32_t>(key[1]) << 8;
            [[fallthrough]];
        case 1:
            h ^= static_cast<uint32_t>(key[0]);
            h *= m;
    }

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;
    return h;
}

// Hash table slot: address in arena + cached hash.
struct KeyValue {
    Addr key_value_addr = Addr::null_pointer();
    uint32_t hash = 0;

    bool
    is_empty() const {
        return key_value_addr.is_null();
    }
    bool
    is_not_empty() const {
        return !key_value_addr.is_null();
    }
};

// Linear probing helper.
struct LinearProbing {
    size_t pos;
    size_t mask;

    static LinearProbing
    compute(uint32_t hash, size_t mask) {
        return LinearProbing{static_cast<size_t>(hash), mask};
    }

    size_t
    next_probe() {
        pos = pos + 1;
        return pos & mask;
    }
};

// Greatest power of two <= n. Panics if n == 0.
inline size_t
compute_previous_power_of_two(size_t n) {
    if (n == 0) {
        throw std::runtime_error("n must be > 0");
    }
    int msb = 63 - __builtin_clzll(static_cast<uint64_t>(n));
    return size_t(1) << msb;
}

// Arena-based hash map with byte-slice keys.
//
// Keys and values are stored inline in an external MemoryArena.
// Arena layout per entry: [key_len: u16 LE] [key_bytes] [value_bytes]
//
// The hash table uses open addressing with linear probing.
// Multiple SharedArenaHashMap instances can share one MemoryArena.
class SharedArenaHashMap {
 public:
    explicit SharedArenaHashMap(size_t table_size = 4) {
        size_t cap = compute_previous_power_of_two(table_size);
        table_.resize(cap);
        mask_ = cap - 1;
    }

    size_t
    mem_usage() const {
        return table_.size() * sizeof(KeyValue);
    }

    bool
    is_empty() const {
        return len_ == 0;
    }

    size_t
    len() const {
        return len_;
    }

    // Get a value associated to a key.
    template <typename V>
    std::optional<V>
    get(const uint8_t* key, size_t key_len, const MemoryArena& arena) const {
        uint32_t h = get_hash(key, key_len);
        LinearProbing probe = LinearProbing::compute(h, mask_);
        while (true) {
            size_t bucket = probe.next_probe();
            const KeyValue& kv = table_[bucket];
            if (kv.is_empty()) {
                return std::nullopt;
            }
            if (kv.hash == h) {
                auto val_addr = get_value_addr_if_key_match(
                    key, key_len, kv.key_value_addr, arena);
                if (val_addr) {
                    return arena.read<V>(*val_addr);
                }
            }
        }
    }

    // Create a new entry or update an existing one.
    // `updater` receives std::optional<V> (nullopt if new) and returns V.
    // The key is truncated to u16::MAX bytes.
    template <typename V, typename Updater>
    V
    mutate_or_create(const uint8_t* key,
                     size_t key_len,
                     MemoryArena& arena,
                     Updater&& updater) {
        if (is_saturated()) {
            resize();
        }
        // Truncate key to u16 max.
        key_len = std::min(key_len, static_cast<size_t>(UINT16_MAX));

        uint32_t h = get_hash(key, key_len);
        LinearProbing probe = LinearProbing::compute(h, mask_);
        size_t bucket = probe.next_probe();
        KeyValue kv = table_[bucket];

        while (true) {
            if (kv.is_empty()) {
                // Key does not exist: create new entry.
                V val = updater(std::optional<V>(std::nullopt));
                // [key_len: u16 LE] [key_bytes] [value_bytes]
                size_t num_bytes = sizeof(uint16_t) + key_len + sizeof(V);
                Addr key_addr = arena.allocate_space(num_bytes);
                {
                    uint8_t* data = arena.slice_mut(key_addr, num_bytes);
                    uint16_t kl = static_cast<uint16_t>(key_len);
                    std::memcpy(data, &kl, sizeof(uint16_t));
                    std::memcpy(data + 2, key, key_len);
                    std::memcpy(data + 2 + key_len, &val, sizeof(V));
                }
                set_bucket(h, key_addr, bucket);
                return val;
            }
            if (kv.hash == h) {
                auto val_addr = get_value_addr_if_key_match(
                    key, key_len, kv.key_value_addr, arena);
                if (val_addr) {
                    V v = arena.read<V>(*val_addr);
                    V new_v = updater(std::optional<V>(v));
                    arena.write_at(*val_addr, new_v);
                    return new_v;
                }
            }
            // Fetch next bucket before loop jump (matches tantivy).
            bucket = probe.next_probe();
            kv = table_[bucket];
        }
    }

    // Iterate all entries: callback(key_ptr, key_len, value_addr).
    template <typename Callback>
    void
    iter(const MemoryArena& arena, Callback&& callback) const {
        for (const auto& kv : table_) {
            if (kv.is_not_empty()) {
                auto info = get_key_value(kv.key_value_addr, arena);
                callback(info.key, info.key_len, info.value_addr);
            }
        }
    }

 private:
    static uint32_t
    get_hash(const uint8_t* key, size_t len) {
        return murmurhash2(key, len);
    }

    struct KeyInfo {
        const uint8_t* key;
        size_t key_len;
        Addr value_addr;
    };

    // Read key and compute value address from arena.
    // Arena layout: [key_len: u16 LE] [key_bytes] [value starts here]
    static KeyInfo
    get_key_value(Addr addr, const MemoryArena& arena) {
        const uint8_t* data = arena.slice_from(addr);
        uint16_t key_bytes_len;
        std::memcpy(&key_bytes_len, data, sizeof(uint16_t));
        const uint8_t* key = data + 2;
        Addr value_addr = addr.offset(2 + key_bytes_len);
        return {key, key_bytes_len, value_addr};
    }

    // Check if stored key matches target; return value Addr if so.
    static std::optional<Addr>
    get_value_addr_if_key_match(const uint8_t* target_key,
                                size_t target_len,
                                Addr addr,
                                const MemoryArena& arena) {
        auto info = get_key_value(addr, arena);
        if (info.key_len == target_len &&
            std::memcmp(info.key, target_key, target_len) == 0) {
            return info.value_addr;
        }
        return std::nullopt;
    }

    void
    set_bucket(uint32_t hash, Addr key_value_addr, size_t bucket) {
        len_++;
        table_[bucket] = KeyValue{key_value_addr, hash};
    }

    bool
    is_saturated() const {
        return table_.size() <= len_ * 2;
    }

    void
    resize() {
        size_t new_len = std::max(table_.size() * 2, size_t(1) << 13);
        size_t new_mask = new_len - 1;
        mask_ = new_mask;
        std::vector<KeyValue> new_table(new_len);
        std::vector<KeyValue> old_table = std::move(table_);
        table_ = std::move(new_table);
        for (const auto& kv : old_table) {
            if (kv.is_not_empty()) {
                LinearProbing probe = LinearProbing::compute(kv.hash, new_mask);
                while (true) {
                    size_t bucket = probe.next_probe();
                    if (table_[bucket].is_empty()) {
                        table_[bucket] = kv;
                        break;
                    }
                }
            }
        }
    }

    std::vector<KeyValue> table_;
    size_t mask_ = 0;
    size_t len_ = 0;
};

}  // namespace milvus::index::inverted
