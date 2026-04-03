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

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

#include "index/inverted/postings/PostingFormat.h"

namespace milvus::index::inverted {

// Matches tantivy's BlockEncoder (postings/compression/mod.rs).
// Reusable encoder for compressing blocks of 128 values.
// Holds an internal buffer that is reused across calls.

struct BlockEncoder {
    std::vector<uint8_t> output;

    // Delta-encode and bitpack a sorted block of doc_ids.
    // Matches tantivy's compress_block_sorted(block, offset).
    // Uses strictly sorted encoding: stores delta-1 since doc_ids are unique
    // (gaps >= 1). When offset=0, first delta is block[0] unchanged
    // (matches tantivy's None case).
    // Returns (num_bits, pointer to compressed data, compressed data size).
    std::tuple<uint8_t, const uint8_t*, size_t>
    compress_block_sorted(const std::array<uint32_t, kBitpackBlockSize>& block,
                          uint32_t offset) {
        uint32_t deltas[kBitpackBlockSize];
        if (offset == 0) {
            deltas[0] = block[0];
        } else {
            deltas[0] = block[0] - offset - 1;
        }
        for (size_t i = 1; i < kBitpackBlockSize; i++) {
            deltas[i] = block[i] - block[i - 1] - 1;
        }

        uint32_t max_val = 0;
        for (size_t i = 0; i < kBitpackBlockSize; i++) {
            max_val |= deltas[i];
        }
        uint8_t num_bits = bit_width(max_val);

        output.clear();
        bitpack(deltas, kBitpackBlockSize, num_bits, output);

        return {num_bits, output.data(), output.size()};
    }
};

// Matches tantivy's BlockDecoder (postings/compression/mod.rs).
// Reusable decoder for decompressing blocks of 128 values.

struct BlockDecoder {
    std::array<uint32_t, kBitpackBlockSize> output{};
    size_t output_len = 0;

    // Decompress a sorted (strictly sorted delta-encoded) block.
    // Matches tantivy's uncompress_block_sorted(data, offset, num_bits).
    // Returns number of bytes consumed from compressed_data.
    size_t
    uncompress_block_sorted(const uint8_t* compressed_data,
                            uint32_t offset,
                            uint8_t num_bits) {
        size_t consumed = (kBitpackBlockSize * num_bits + 7) / 8;
        bitunpack(compressed_data, kBitpackBlockSize, num_bits, output.data());

        // Undo strictly sorted delta encoding.
        // offset=0 (None): val[0] = delta[0]
        // offset>0: val[0] = offset + delta[0] + 1
        if (offset != 0) {
            output[0] += offset + 1;
        }
        for (size_t i = 1; i < kBitpackBlockSize; i++) {
            output[i] += output[i - 1] + 1;
        }
        output_len = kBitpackBlockSize;
        return consumed;
    }
};

}  // namespace milvus::index::inverted
