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
#include <vector>

#include "MemoryArena.h"

namespace milvus::index::inverted {

// Ported from tantivy stacker/src/expull.rs
//
// Exponential unrolled linked list stored in MemoryArena.
// Block sizes grow exponentially: first block 8B (1<<3),
// then 16B, 32B, ..., capped at 32KB (1<<15).
//
// Each block layout: [data: block_size bytes] [next_addr: 4 bytes]
// The tail pointer advances as data is written within the current block.
// When a block is full, a new block is allocated and linked via the next pointer.

static constexpr uint16_t kFirstBlockNum = 2;

// Block size = 1 << min(block_num, 15).
// First block (block_num=3 after increment): 1<<3 = 8 bytes.
inline uint16_t
get_block_size(uint16_t block_num) {
    return static_cast<uint16_t>(
        1u << std::min(block_num, static_cast<uint16_t>(15)));
}

class ExpUnrolledLinkedList {
 public:
    ExpUnrolledLinkedList() = default;

    void
    increment_num_blocks() {
        block_num_++;
    }

    // Read all stored data into output vector.
    void
    read_to_end(const MemoryArena& arena, std::vector<uint8_t>& output) const {
        Addr addr = head_;
        if (addr.is_null()) {
            return;
        }

        uint16_t last_block_len = get_block_size(block_num_) - remaining_cap_;

        // Full blocks.
        for (uint16_t bn = kFirstBlockNum + 1; bn < block_num_; bn++) {
            uint16_t cap = get_block_size(bn);
            const uint8_t* data = arena.slice(addr, cap);
            output.insert(output.end(), data, data + cap);
            addr = arena.read<Addr>(addr.offset(cap));
        }

        // Last block (may be partial).
        const uint8_t* data = arena.slice(addr, last_block_len);
        output.insert(output.end(), data, data + last_block_len);
    }

    bool
    is_empty() const {
        return head_.is_null();
    }

    // Writer for appending data to the list.
    // Ported from tantivy ExpUnrolledLinkedListWriter.
    class Writer {
     public:
        Writer(ExpUnrolledLinkedList& eull, MemoryArena& arena)
            : eull_(eull), arena_(arena) {
        }

        void
        extend_from_slice(const uint8_t* buf, size_t len) {
            while (len > 0) {
                if (eull_.remaining_cap_ == 0) {
                    eull_.increment_num_blocks();
                    uint16_t block_size = get_block_size(eull_.block_num_);
                    ensure_capacity(block_size);
                }

                uint8_t* output =
                    arena_.slice_mut(eull_.tail_, eull_.remaining_cap_);
                size_t add_len =
                    std::min(len, static_cast<size_t>(eull_.remaining_cap_));
                std::memcpy(output, buf, add_len);

                eull_.remaining_cap_ -= static_cast<uint16_t>(add_len);
                eull_.tail_ =
                    eull_.tail_.offset(static_cast<uint32_t>(add_len));
                buf += add_len;
                len -= add_len;
            }
        }

        void
        write_u32_vint(uint32_t val) {
            uint8_t buf[5];
            size_t pos = 0;
            while (val >= 0x80) {
                buf[pos++] = static_cast<uint8_t>(val | 0x80);
                val >>= 7;
            }
            buf[pos++] = static_cast<uint8_t>(val);
            extend_from_slice(buf, pos);
        }

     private:
        void
        ensure_capacity(uint32_t allocate) {
            Addr new_block = arena_.allocate_space(allocate + sizeof(Addr));
            if (eull_.head_.is_null()) {
                eull_.head_ = new_block;
            } else {
                // Write next pointer at current tail position.
                arena_.write_at(eull_.tail_, new_block);
            }
            eull_.tail_ = new_block;
            eull_.remaining_cap_ = static_cast<uint16_t>(allocate);
        }

        ExpUnrolledLinkedList& eull_;
        MemoryArena& arena_;
    };

    Writer
    writer(MemoryArena& arena) {
        return Writer(*this, arena);
    }

 private:
    // remaining_cap starts at 0 to trigger initial allocation.
    uint16_t remaining_cap_ = 0;
    uint16_t block_num_ = kFirstBlockNum;
    Addr head_ = Addr::null_pointer();
    Addr tail_ = Addr::null_pointer();
};

}  // namespace milvus::index::inverted
