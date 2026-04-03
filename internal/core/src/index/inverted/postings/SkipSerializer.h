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
#include <cstring>
#include <vector>

#include "index/inverted/postings/PostingFormat.h"

namespace milvus::index::inverted {

// Matches tantivy's encode_bitwidth / decode_bitwidth.
// Encodes the bit width and strict_delta flag into a single byte.
inline uint8_t
encode_bitwidth(uint8_t num_bits, bool strict_delta) {
    return strict_delta ? num_bits + 33 : num_bits;
}

inline std::pair<uint8_t, bool>
decode_bitwidth(uint8_t encoded) {
    if (encoded >= 33) {
        return {static_cast<uint8_t>(encoded - 33), true};
    }
    return {encoded, false};
}

// --- SkipSerializer ---
//
// Writes per-block skip entries for full blocks (128 docs).
// Phase 1: doc_id only (no term freq, positions, or block WAND).
//
// Skip entry format (per full block):
//   [last_doc: uint32 LE] [encoded_bitwidth: uint8]
//
// The skip data is written before the postings data in .pst:
//   [skip_data_len: VarUInt] [skip_data] [postings_data]
// (Only when doc_freq >= kBitpackBlockSize.)

class SkipSerializer {
 public:
    void
    write_doc(uint32_t last_doc, uint8_t doc_num_bits) {
        write_u32(last_doc);
        buffer_.push_back(encode_bitwidth(doc_num_bits, true));
    }

    const uint8_t*
    data() const {
        return buffer_.data();
    }

    size_t
    size() const {
        return buffer_.size();
    }

    void
    clear() {
        buffer_.clear();
    }

 private:
    void
    write_u32(uint32_t val) {
        uint8_t buf[4];
        std::memcpy(buf, &val, 4);
        buffer_.insert(buffer_.end(), buf, buf + 4);
    }

    std::vector<uint8_t> buffer_;
};

// --- BlockInfo ---
// Describes one block's encoding type and parameters.

enum class BlockType : uint8_t {
    kBitPacked = 0,
    kVInt = 1,
};

struct BlockInfo {
    BlockType type = BlockType::kVInt;
    // For BitPacked:
    uint8_t doc_num_bits = 0;
    // For VInt:
    uint32_t num_docs = 0;

    bool
    operator==(const BlockInfo& other) const {
        if (type != other.type)
            return false;
        if (type == BlockType::kBitPacked)
            return doc_num_bits == other.doc_num_bits;
        return num_docs == other.num_docs;
    }
};

// Skip entry size for Phase 1 (doc_id only): 4 bytes last_doc + 1 byte bits = 5
static constexpr size_t kSkipEntrySize = 5;

// --- SkipReader ---
//
// Reads skip entries and supports block-level seeking within a posting list.
// Phase 1: doc_id only (IndexRecordOption::Basic equivalent).

class SkipReader {
 public:
    SkipReader(const uint8_t* skip_data,
               size_t skip_data_len,
               uint32_t doc_freq)
        : data_(skip_data),
          data_len_(skip_data_len),
          read_pos_(0),
          remaining_docs_(doc_freq),
          byte_offset_(0) {
        if (doc_freq >= kBitpackBlockSize) {
            last_doc_in_block_ = 0;
            read_block_info();
        } else {
            // No full blocks — everything is VInt tail
            last_doc_in_block_ = UINT32_MAX;  // TERMINATED
            block_info_ = {BlockType::kVInt, 0, doc_freq};
        }
    }

    uint32_t
    last_doc_in_block() const {
        return last_doc_in_block_;
    }

    uint32_t
    last_doc_in_previous_block() const {
        return last_doc_in_previous_block_;
    }

    size_t
    byte_offset() const {
        return byte_offset_;
    }

    BlockInfo
    block_info() const {
        return block_info_;
    }

    // Advance to the next block.
    void
    advance() {
        if (block_info_.type == BlockType::kBitPacked) {
            remaining_docs_ -= static_cast<uint32_t>(kBitpackBlockSize);
            byte_offset_ += compressed_block_size(block_info_.doc_num_bits);
        } else {
            // VInt block — consuming remaining docs
            remaining_docs_ = 0;
            byte_offset_ = SIZE_MAX;
        }

        last_doc_in_previous_block_ = last_doc_in_block_;

        if (remaining_docs_ >= kBitpackBlockSize) {
            read_block_info();
        } else {
            // Remaining docs form the VInt tail
            last_doc_in_block_ = UINT32_MAX;  // TERMINATED
            block_info_ = {BlockType::kVInt, 0, remaining_docs_};
        }
    }

    // Seek to the block that may contain `target`.
    // Returns true if we advanced past at least one block.
    bool
    seek(uint32_t target) {
        if (last_doc_in_block_ >= target) {
            return false;
        }
        while (true) {
            advance();
            if (last_doc_in_block_ >= target) {
                return true;
            }
        }
    }

 private:
    void
    read_block_info() {
        // Read [last_doc: u32 LE][encoded_bitwidth: u8]
        std::memcpy(&last_doc_in_block_, data_ + read_pos_, 4);
        auto [doc_num_bits, strict_delta] =
            decode_bitwidth(data_[read_pos_ + 4]);
        read_pos_ += kSkipEntrySize;

        block_info_ = {BlockType::kBitPacked, doc_num_bits, 0};
    }

    // Size of a compressed full block (bitpacked data only, no header).
    static size_t
    compressed_block_size(uint8_t doc_num_bits) {
        return (kBitpackBlockSize * doc_num_bits + 7) / 8;
    }

    const uint8_t* data_;
    size_t data_len_;
    size_t read_pos_;

    uint32_t last_doc_in_block_ = 0;
    uint32_t last_doc_in_previous_block_ = 0;
    uint32_t remaining_docs_;
    size_t byte_offset_;
    BlockInfo block_info_;
};

}  // namespace milvus::index::inverted
