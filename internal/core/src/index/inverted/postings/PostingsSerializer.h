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
#include <array>
#include <cstdint>
#include <vector>

#include "index/inverted/storage/FileIO.h"
#include "index/inverted/postings/PostingFormat.h"
#include "index/inverted/postings/BlockEncoder.h"
#include "index/inverted/postings/SkipSerializer.h"

namespace milvus::index::inverted {

// --- Block ---
// Accumulates doc_ids for one block (128 entries).

struct Block {
    std::array<uint32_t, kBitpackBlockSize> doc_ids{};
    size_t len = 0;

    void
    append_doc(uint32_t doc_id) {
        doc_ids[len++] = doc_id;
    }

    bool
    is_full() const {
        return len == kBitpackBlockSize;
    }

    bool
    is_empty() const {
        return len == 0;
    }

    uint32_t
    last_doc() const {
        return doc_ids[kBitpackBlockSize - 1];
    }

    void
    clear() {
        len = 0;
    }
};

// --- PostingsSerializer ---
// Phase 1: doc_id only (no term freq, positions, or block WAND).
//
// Accumulates doc_ids in blocks of 128. Full blocks are delta-encoded
// and bitpacked. The tail block (< 128) uses VInt encoding.
//
// Per-term output format in .pst:
//   If doc_freq >= 128:
//     [skip_data_len: VarUInt] [skip_data] [postings_data]
//   If doc_freq < 128:
//     [postings_data]   (VInt-encoded deltas only)
//
// Postings data format:
//   For each full block:
//     [bit_width: uint8] [bitpacked deltas: ceil(128 * bits / 8) bytes]
//   Tail block (if any):
//     [VInt-encoded deltas]
//
// Usage:
//   serializer.clear();            // reset for new term
//   for (doc_id : sorted_docs)
//       serializer.write_doc(doc_id);
//   serializer.close_term(doc_freq);  // flush to output

class PostingsSerializer {
 public:
    explicit PostingsSerializer(FileWriter* output) : output_(output) {
    }

    // Reset state for a new term.
    void
    clear() {
        block_.clear();
        last_doc_id_encoded_ = 0;
        postings_buffer_.clear();
        skip_serializer_.clear();
    }

    // Add a doc_id. Doc_ids must be pushed in increasing order.
    void
    write_doc(uint32_t doc_id) {
        block_.append_doc(doc_id);
        if (block_.is_full()) {
            write_block();
        }
    }

    // Flush the current term's postings to the output writer.
    // Must be called after all doc_ids for this term have been written.
    void
    close_term(uint32_t doc_freq) {
        // Encode tail block (remaining < 128 docs) as VInt
        if (!block_.is_empty()) {
            vint_encode_sorted(block_.doc_ids.data(),
                               block_.len,
                               last_doc_id_encoded_,
                               postings_buffer_);
            block_.clear();
        }

        // Write to output: [skip_data_len][skip_data][postings_data]
        if (doc_freq >= static_cast<uint32_t>(kBitpackBlockSize)) {
            write_varuint(output_, skip_serializer_.size());
            output_->write(skip_serializer_.data(), skip_serializer_.size());
        }
        output_->write(postings_buffer_.data(), postings_buffer_.size());

        // Clear buffers for next term
        skip_serializer_.clear();
        postings_buffer_.clear();
    }

    // Number of bytes written to the output so far.
    uint64_t
    written_bytes() const {
        return output_->offset();
    }

 private:
    void
    write_block() {
        auto [num_bits, encoded, encoded_size] =
            block_encoder_.compress_block_sorted(block_.doc_ids,
                                                 last_doc_id_encoded_);

        last_doc_id_encoded_ = block_.last_doc();
        skip_serializer_.write_doc(last_doc_id_encoded_, num_bits);

        postings_buffer_.insert(
            postings_buffer_.end(), encoded, encoded + encoded_size);

        block_.clear();
    }

    FileWriter* output_;
    uint32_t last_doc_id_encoded_ = 0;

    Block block_;
    BlockEncoder block_encoder_;
    std::vector<uint8_t> postings_buffer_;
    SkipSerializer skip_serializer_;
};

// --- PostingsDecoder ---
// Decode helper for reading posting lists written by PostingsSerializer.
// Uses SkipReader for block-level navigation.
//
// This is a simplified decode-all utility for testing and simple queries.
// Step 4 will introduce BlockSegmentPostings for lazy block-at-a-time decoding.

class PostingsDecoder {
 public:
    // Decode a complete posting list from the .pst file.
    // `offset` is the start position in the file.
    // `len` is the total byte length of this term's posting data.
    // `doc_freq` is the number of documents.
    static void
    decode(FileReader* reader,
           uint64_t offset,
           uint32_t len,
           uint32_t doc_freq,
           std::vector<uint32_t>& output) {
        std::vector<uint8_t> buf(len);
        reader->read(offset, len, buf.data());
        const uint8_t* ptr = buf.data();

        output.resize(doc_freq);

        // Parse skip data to get per-block num_bits
        const uint8_t* skip_ptr = nullptr;
        if (doc_freq >= static_cast<uint32_t>(kBitpackBlockSize)) {
            uint64_t skip_data_len = decode_varuint(ptr);
            skip_ptr = ptr;
            ptr += skip_data_len;
        }

        // Decode full blocks (num_bits comes from skip data)
        BlockDecoder decoder;
        size_t pos = 0;
        uint32_t last_doc = 0;
        while (pos + kBitpackBlockSize <= doc_freq) {
            // Read skip entry: [last_doc: u32 LE][encoded_bitwidth: u8]
            skip_ptr += 4;  // skip last_doc field
            auto [bits, strict_delta] = decode_bitwidth(*skip_ptr++);

            size_t consumed =
                decoder.uncompress_block_sorted(ptr, last_doc, bits);
            ptr += consumed;
            std::copy(decoder.output.begin(),
                      decoder.output.begin() + kBitpackBlockSize,
                      output.data() + pos);
            last_doc = decoder.output[kBitpackBlockSize - 1];
            pos += kBitpackBlockSize;
        }

        // Decode VInt tail
        size_t remainder = doc_freq - pos;
        if (remainder > 0) {
            vint_decode_sorted(ptr, remainder, last_doc, output.data() + pos);
        }
    }
};

}  // namespace milvus::index::inverted
