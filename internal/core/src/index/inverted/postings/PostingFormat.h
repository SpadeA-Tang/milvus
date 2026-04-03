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
#include <cstring>
#include <optional>
#include <vector>

#include "index/inverted/storage/FileIO.h"

namespace milvus::index::inverted {

// --- VarInt encoding (unsigned LEB128) ---

inline size_t
encode_varuint(uint8_t* buf, uint64_t val) {
    size_t pos = 0;
    while (val >= 0x80) {
        buf[pos++] = static_cast<uint8_t>(val | 0x80);
        val >>= 7;
    }
    buf[pos++] = static_cast<uint8_t>(val);
    return pos;
}

inline uint64_t
decode_varuint(const uint8_t*& ptr) {
    uint64_t val = 0;
    uint32_t shift = 0;
    while (true) {
        uint8_t byte = *ptr++;
        val |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0)
            return val;
        shift += 7;
    }
}

// Iterator over varint-encoded uint32 values.
struct VInt32Reader {
    const uint8_t* ptr;
    const uint8_t* end;

    VInt32Reader(const uint8_t* data, const uint8_t* end)
        : ptr(data), end(end) {
    }

    bool
    next(uint32_t& out) {
        if (ptr >= end)
            return false;
        out = static_cast<uint32_t>(decode_varuint(ptr));
        return true;
    }
};

// Wraps VInt32Reader to yield prefix sums (delta → absolute).
struct SumReader {
    VInt32Reader inner;
    uint32_t state = 0;

    explicit SumReader(VInt32Reader reader) : inner(reader) {
    }

    bool
    next(uint32_t& out) {
        uint32_t delta;
        if (!inner.next(delta))
            return false;
        state += delta;
        out = state;
        return true;
    }
};

inline SumReader
get_sum_reader(VInt32Reader reader) {
    return SumReader(reader);
}

inline void
write_varuint(FileWriter* writer, uint64_t val) {
    uint8_t buf[10];
    size_t len = encode_varuint(buf, val);
    writer->write(buf, len);
}

// --- Posting encoding types ---

enum class PostingEncoding : uint8_t {
    kInline = 0,
    kBitpacking = 1,
    kRoaring = 2,
};

// --- PostingsInfo ---
// Metadata for one term's posting list.
// Inline postings (cardinality <= kInlineMax) are stored directly here.
// Larger postings reference a range in the .pst file.

struct PostingsInfo {
    uint32_t cardinality = 0;
    PostingEncoding encoding = PostingEncoding::kInline;

    static constexpr size_t kInlineMax = 6;
    std::array<uint32_t, kInlineMax> inline_docs{};

    uint64_t file_offset = 0;
    uint32_t data_size = 0;

    void
    serialize(FileWriter* writer) const {
        write_varuint(writer, cardinality);
        uint8_t enc = static_cast<uint8_t>(encoding);
        writer->write(&enc, 1);

        if (encoding == PostingEncoding::kInline) {
            for (uint32_t i = 0; i < cardinality; i++) {
                write_varuint(writer, inline_docs[i]);
            }
        } else {
            write_varuint(writer, file_offset);
            write_varuint(writer, data_size);
        }
    }

    static PostingsInfo
    deserialize(const uint8_t*& ptr) {
        PostingsInfo info;
        info.cardinality = static_cast<uint32_t>(decode_varuint(ptr));
        info.encoding = static_cast<PostingEncoding>(*ptr++);

        if (info.encoding == PostingEncoding::kInline) {
            for (uint32_t i = 0; i < info.cardinality; i++) {
                info.inline_docs[i] =
                    static_cast<uint32_t>(decode_varuint(ptr));
            }
        } else {
            info.file_offset = decode_varuint(ptr);
            info.data_size = static_cast<uint32_t>(decode_varuint(ptr));
        }
        return info;
    }
};

// --- Bitpacking helpers ---

static constexpr size_t kBitpackBlockSize = 128;

inline uint8_t
bit_width(uint32_t max_val) {
    if (max_val == 0)
        return 0;
    return static_cast<uint8_t>(32 - __builtin_clz(max_val));
}

// Pack `count` values of `bits` width each into output buffer.
inline void
bitpack(const uint32_t* values,
        size_t count,
        uint8_t bits,
        std::vector<uint8_t>& output) {
    if (bits == 0)
        return;

    size_t total_bytes = (count * bits + 7) / 8;
    size_t start = output.size();
    output.resize(start + total_bytes, 0);

    uint8_t* dst = output.data() + start;
    uint64_t buffer = 0;
    int buffer_bits = 0;
    size_t byte_pos = 0;

    for (size_t i = 0; i < count; i++) {
        buffer |= static_cast<uint64_t>(values[i]) << buffer_bits;
        buffer_bits += bits;
        while (buffer_bits >= 8) {
            dst[byte_pos++] = static_cast<uint8_t>(buffer & 0xFF);
            buffer >>= 8;
            buffer_bits -= 8;
        }
    }
    if (buffer_bits > 0) {
        dst[byte_pos] = static_cast<uint8_t>(buffer & 0xFF);
    }
}

// Unpack `count` values of `bits` width each from src buffer.
inline void
bitunpack(const uint8_t* src, size_t count, uint8_t bits, uint32_t* output) {
    if (bits == 0) {
        std::memset(output, 0, count * sizeof(uint32_t));
        return;
    }

    uint32_t mask = (bits == 32) ? UINT32_MAX : ((1u << bits) - 1);
    uint64_t buffer = 0;
    int buffer_bits = 0;
    size_t byte_pos = 0;

    for (size_t i = 0; i < count; i++) {
        while (buffer_bits < bits) {
            buffer |= static_cast<uint64_t>(src[byte_pos++]) << buffer_bits;
            buffer_bits += 8;
        }
        output[i] = static_cast<uint32_t>(buffer) & mask;
        buffer >>= bits;
        buffer_bits -= bits;
    }
}

// --- VInt helpers for delta-encoded sorted arrays ---

inline void
vint_encode_sorted(const uint32_t* values,
                   size_t count,
                   uint32_t offset,
                   std::vector<uint8_t>& output) {
    for (size_t i = 0; i < count; i++) {
        uint32_t delta = values[i] - offset;
        offset = values[i];
        uint8_t buf[5];
        size_t len = encode_varuint(buf, delta);
        output.insert(output.end(), buf, buf + len);
    }
}

inline size_t
vint_decode_sorted(const uint8_t* data,
                   size_t count,
                   uint32_t offset,
                   uint32_t* output) {
    const uint8_t* ptr = data;
    for (size_t i = 0; i < count; i++) {
        uint32_t delta = static_cast<uint32_t>(decode_varuint(ptr));
        offset += delta;
        output[i] = offset;
    }
    return static_cast<size_t>(ptr - data);
}

// --- PostingListCodec ---
// Codec interface for encoding/decoding posting lists to/from .pst file.

class PostingListCodec {
 public:
    virtual void
    encode(const uint32_t* doc_ids, size_t count, FileWriter* writer) = 0;
    virtual void
    decode(FileReader* reader,
           uint64_t offset,
           uint32_t len,
           size_t count,
           std::vector<uint32_t>& output) = 0;
    virtual ~PostingListCodec() = default;
};

// Delta + block-128 bitpacking codec.
// with skip list, term frequency, and block WAND removed.
//
// Format:
//   For each full block (128 doc_ids):
//     [bit_width: uint8] [bitpacked deltas: ceil(128 * bits / 8) bytes]
//   If remainder > 0 (< 128 doc_ids):
//     [VInt-encoded deltas]
//
// Full blocks use bitpacking for speed; the tail block uses VInt
// because bitpacking a partial block wastes bits. This matches tantivy.
//
// NOTE: This codec is used by InvertedStorageTest for low-level encode/decode
// tests. The main serialization path uses PostingsSerializer + BlockEncoder.
class BitpackingPostingCodec : public PostingListCodec {
 public:
    void
    encode(const uint32_t* doc_ids, size_t count, FileWriter* writer) override {
        if (count == 0)
            return;

        // Delta encode
        std::vector<uint32_t> deltas(count);
        deltas[0] = doc_ids[0];
        for (size_t i = 1; i < count; i++) {
            deltas[i] = doc_ids[i] - doc_ids[i - 1];
        }

        std::vector<uint8_t> buffer;
        buffer.reserve(count * 4);

        // Full blocks: bitpacking
        size_t pos = 0;
        while (pos + kBitpackBlockSize <= count) {
            uint32_t max_delta = *std::max_element(
                deltas.data() + pos, deltas.data() + pos + kBitpackBlockSize);
            uint8_t bits = bit_width(max_delta);
            buffer.push_back(bits);
            bitpack(deltas.data() + pos, kBitpackBlockSize, bits, buffer);
            pos += kBitpackBlockSize;
        }

        // Tail block: VInt
        size_t remainder = count - pos;
        if (remainder > 0) {
            uint32_t offset = (pos > 0) ? doc_ids[pos - 1] : 0;
            vint_encode_sorted(doc_ids + pos, remainder, offset, buffer);
        }

        writer->write(buffer.data(), buffer.size());
    }

    void
    decode(FileReader* reader,
           uint64_t offset,
           uint32_t len,
           size_t count,
           std::vector<uint32_t>& output) override {
        std::vector<uint8_t> buf(len);
        reader->read(offset, len, buf.data());

        output.resize(count);
        const uint8_t* ptr = buf.data();

        // Full blocks: bitpacking
        size_t pos = 0;
        uint32_t last_doc = 0;
        while (pos + kBitpackBlockSize <= count) {
            uint8_t bits = *ptr++;
            bitunpack(ptr, kBitpackBlockSize, bits, output.data() + pos);
            ptr += (kBitpackBlockSize * bits + 7) / 8;

            // Undo delta within block
            output[pos] += last_doc;
            for (size_t i = pos + 1; i < pos + kBitpackBlockSize; i++) {
                output[i] += output[i - 1];
            }
            last_doc = output[pos + kBitpackBlockSize - 1];
            pos += kBitpackBlockSize;
        }

        // Tail block: VInt
        size_t remainder = count - pos;
        if (remainder > 0) {
            vint_decode_sorted(ptr, remainder, last_doc, output.data() + pos);
        }
    }
};

}  // namespace milvus::index::inverted
