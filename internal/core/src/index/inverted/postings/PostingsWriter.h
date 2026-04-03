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
#include <string>
#include <vector>

#include "index/inverted/indexing/ArenaHashMap.h"
#include "index/inverted/postings/Recorder.h"
#include "index/inverted/postings/SegmentSerializer.h"

namespace milvus::index::inverted {

struct TermAddr {
    std::string term;
    Addr addr;
};

// PostingsWriter accumulates (term, doc_id) pairs in memory.
//
// Templated on Recorder type to avoid dynamic dispatch.
// Phase 1: DocIdRecorder (doc_id only)
// Phase 2: TfAndPositionRecorder (doc_id + position)
//
// Usage:
//   PostingsWriter<DocIdRecorder> writer;
//   writer.subscribe(doc_id, 0, term_bytes, term_len);
//   ...
//   serialize_postings(writer, serializer);

template <typename Rec>
class PostingsWriter {
 public:
    explicit PostingsWriter(size_t table_size = 4) : map_(table_size) {
    }

    void
    subscribe(uint32_t doc_id,
              uint32_t position,
              const uint8_t* term,
              size_t term_len) {
        MemoryArena& arena = map_.memory_arena();
        map_.template mutate_or_create<Rec>(
            term,
            term_len,
            [&arena, doc_id, position](std::optional<Rec> opt) -> Rec {
                if (opt.has_value()) {
                    Rec rec = opt.value();
                    if (rec.current_doc() != doc_id) {
                        rec.close_doc(arena);
                        rec.new_doc(doc_id, arena);
                    }
                    rec.record_position(position, arena);
                    return rec;
                } else {
                    Rec rec{};
                    rec.new_doc(doc_id, arena);
                    rec.record_position(position, arena);
                    return rec;
                }
            });
        max_doc_id_ = std::max(max_doc_id_, doc_id);
    }

    // Serialize sorted terms through the serializer.
    void
    serialize(const std::vector<TermAddr>& term_addrs,
              SegmentSerializer& serializer) {
        BufferLender lender;
        for (const auto& entry : term_addrs) {
            serialize_one_term(
                reinterpret_cast<const uint8_t*>(entry.term.data()),
                entry.term.size(),
                entry.addr,
                lender,
                serializer);
        }
    }

    size_t
    mem_usage() const {
        return map_.mem_usage();
    }

    size_t
    num_terms() const {
        return map_.len();
    }

    uint32_t
    max_doc_id() const {
        return max_doc_id_;
    }

    ArenaHashMap&
    hash_map() {
        return map_;
    }

    const ArenaHashMap&
    hash_map() const {
        return map_;
    }

 private:
    void
    serialize_one_term(const uint8_t* term,
                       size_t term_len,
                       Addr addr,
                       BufferLender& lender,
                       SegmentSerializer& serializer) {
        const MemoryArena& arena = map_.memory_arena();
        Rec rec = arena.read<Rec>(addr);
        uint32_t term_doc_freq = rec.term_doc_freq().value_or(0);
        serializer.new_term(term, term_len, term_doc_freq);
        rec.serialize(arena, serializer, lender);
        serializer.close_term();
    }

    ArenaHashMap map_;
    uint32_t max_doc_id_ = 0;
};

// Top-level serialization function.
//
// Collects terms from the writer's hashmap, sorts lexicographically,
// and drives PostingsWriter::serialize through the provided serializer.
template <typename Rec>
void
serialize_postings(PostingsWriter<Rec>& writer, SegmentSerializer& serializer) {
    std::vector<TermAddr> term_addrs;
    term_addrs.reserve(writer.num_terms());
    writer.hash_map().iter(
        [&](const uint8_t* key, size_t key_len, Addr val_addr) {
            term_addrs.push_back(
                {std::string(reinterpret_cast<const char*>(key), key_len),
                 val_addr});
        });

    std::sort(
        term_addrs.begin(),
        term_addrs.end(),
        [](const TermAddr& a, const TermAddr& b) { return a.term < b.term; });

    writer.serialize(term_addrs, serializer);
}

}  // namespace milvus::index::inverted
