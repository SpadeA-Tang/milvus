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
#include <vector>

#include "index/inverted/indexing/ExpUnrolledLinkedList.h"
#include "index/inverted/indexing/MemoryArena.h"
#include "index/inverted/postings/SegmentSerializer.h"
#include "index/inverted/postings/PostingFormat.h"

namespace milvus::index::inverted {

//
// Recorder is in charge of recording relevant information about
// the presence of a term in a document.
//
// Recorder types are stored by value in ArenaHashMap (must be trivially copyable,
// no vtable). Uses CRTP to define the interface at compile time.

// CRTP base class for Recorder types.
// Derived must implement:
//   uint32_t current_doc() const;
//   void new_doc(uint32_t doc_id, MemoryArena& arena);
//   void record_position(uint32_t position, MemoryArena& arena);
//   void close_doc(MemoryArena& arena);
//   std::optional<uint32_t> term_doc_freq() const;
//   void serialize(const MemoryArena& arena, SegmentSerializer& serializer,
//                  BufferLender& lender) const;
template <typename Derived>
struct Recorder {
    // Static dispatch helpers (optional, for generic code).
    uint32_t
    current_doc() const {
        return static_cast<const Derived*>(this)->current_doc();
    }
    void
    new_doc(uint32_t doc_id, MemoryArena& arena) {
        static_cast<Derived*>(this)->new_doc(doc_id, arena);
    }
    void
    record_position(uint32_t position, MemoryArena& arena) {
        static_cast<Derived*>(this)->record_position(position, arena);
    }
    void
    close_doc(MemoryArena& arena) {
        static_cast<Derived*>(this)->close_doc(arena);
    }
};

// Reusable buffer to avoid repeated allocations during serialization.
struct BufferLender {
    std::vector<uint8_t> buffer_u8;
    std::vector<uint32_t> buffer_u32;

    std::vector<uint8_t>&
    lend_u8() {
        buffer_u8.clear();
        return buffer_u8;
    }

    std::pair<std::vector<uint8_t>&, std::vector<uint32_t>&>
    lend_all() {
        buffer_u8.clear();
        buffer_u32.clear();
        return {buffer_u8, buffer_u32};
    }
};

// Only records the doc ids.
struct DocIdRecorder : Recorder<DocIdRecorder> {
    ExpUnrolledLinkedList stack;
    uint32_t current_doc_ = 0;
    uint32_t doc_count_ = 0;

    uint32_t
    current_doc() const {
        return current_doc_;
    }

    void
    new_doc(uint32_t doc_id, MemoryArena& arena) {
        uint32_t delta = doc_id - current_doc_;
        current_doc_ = doc_id;
        doc_count_++;
        stack.writer(arena).write_u32_vint(delta);
    }

    void
    record_position(uint32_t /*position*/, MemoryArena& /*arena*/) {
    }

    void
    close_doc(MemoryArena& /*arena*/) {
    }

    std::optional<uint32_t>
    term_doc_freq() const {
        return doc_count_;
    }

    bool
    has_term_freq() const {
        return false;
    }

    void
    serialize(const MemoryArena& arena,
              SegmentSerializer& serializer,
              BufferLender& lender) const {
        auto& buffer = lender.lend_u8();
        stack.read_to_end(arena, buffer);

        auto iter = get_sum_reader(
            VInt32Reader(buffer.data(), buffer.data() + buffer.size()));
        uint32_t doc_id;
        while (iter.next(doc_id)) {
            serializer.write_doc(doc_id, 0, nullptr, 0);
        }
    }
};

// Records doc ids, term frequencies, and positions.
// Phase 2: uncomment and use when position storage is needed.
//
// struct TfAndPositionRecorder {
//     ExpUnrolledLinkedList stack;
//     uint32_t current_doc_ = 0;
//     uint32_t term_doc_freq_ = 0;
//
//     static constexpr uint32_t kPositionEnd = 0;
//
//     uint32_t current_doc() const { return current_doc_; }
//
//     void new_doc(uint32_t doc_id, MemoryArena& arena) {
//         uint32_t delta = doc_id - current_doc_;
//         current_doc_ = doc_id;
//         term_doc_freq_++;
//         stack.writer(arena).write_u32_vint(delta);
//     }
//
//     void record_position(uint32_t position, MemoryArena& arena) {
//         stack.writer(arena).write_u32_vint(position + 1);
//     }
//
//     void close_doc(MemoryArena& arena) {
//         stack.writer(arena).write_u32_vint(kPositionEnd);
//     }
//
//     std::optional<uint32_t> term_doc_freq() const { return term_doc_freq_; }
//     bool has_term_freq() const { return true; }
//
//     void serialize(const MemoryArena& arena, SegmentSerializer& serializer,
//                    BufferLender& lender) const {
//         auto& [buffer_u8, buffer_positions] = lender.lend_all();
//         stack.read_to_end(arena, buffer_u8);
//         const uint8_t* ptr = buffer_u8.data();
//         const uint8_t* end = ptr + buffer_u8.size();
//         uint32_t prev_doc = 0;
//         while (ptr < end) {
//             uint32_t delta = static_cast<uint32_t>(decode_varuint(ptr));
//             uint32_t doc_id = prev_doc + delta;
//             prev_doc = doc_id;
//             buffer_positions.clear();
//             uint32_t prev_pos_plus_one = 1;
//             while (ptr < end) {
//                 uint32_t val = static_cast<uint32_t>(decode_varuint(ptr));
//                 if (val == kPositionEnd) break;
//                 uint32_t delta_pos = val - prev_pos_plus_one;
//                 buffer_positions.push_back(delta_pos);
//                 prev_pos_plus_one = val;
//             }
//             serializer.write_doc(doc_id);  // Phase 2: add tf, positions
//         }
//     }
// };

}  // namespace milvus::index::inverted
