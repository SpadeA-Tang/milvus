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
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "index/inverted/storage/FileIO.h"
#include "index/inverted/postings/PostingFormat.h"
#include "index/inverted/termdict/TermDictionary.h"

namespace milvus::index::inverted {

static constexpr size_t kDictBlockSize = 128;

// --- Sparse Index (in-memory) ---

struct SparseIndexEntry {
    std::string first_token;
    uint64_t block_offset;  // offset in .dct file
};

struct SparseIndex {
    std::vector<SparseIndexEntry> entries;

    // Find the block that may contain the token.
    // Returns the index of the last block whose first_token <= token,
    // or -1 if token < all first tokens.
    int
    find_block(const uint8_t* token, size_t token_len) const {
        std::string_view target(reinterpret_cast<const char*>(token),
                                token_len);
        int lo = 0, hi = static_cast<int>(entries.size()) - 1;
        int result = -1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (entries[mid].first_token <= target) {
                result = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return result;
    }
};

// --- Dictionary Block (loaded from .dct) ---

struct DictBlock {
    std::vector<std::string> tokens;
    std::vector<PostingsInfo> infos;

    // Binary search for a token within the block.
    int
    find(const uint8_t* token, size_t token_len) const {
        std::string_view target(reinterpret_cast<const char*>(token),
                                token_len);
        int lo = 0, hi = static_cast<int>(tokens.size()) - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (tokens[mid] == target)
                return mid;
            if (tokens[mid] < target)
                lo = mid + 1;
            else
                hi = mid - 1;
        }
        return -1;
    }
};

// --- SSTable Writer ---
// Implements TermDictionaryWriter with SSTable format:
// .idx (sparse index) + .dct (dictionary blocks of 128 entries each).
// Terms must be inserted in lexicographic order.

class SortedBlockKeyMapWriter : public TermDictionaryWriter {
 public:
    SortedBlockKeyMapWriter(FileWriter* idx_writer, FileWriter* dct_writer)
        : idx_writer_(idx_writer), dct_writer_(dct_writer) {
    }

    void
    insert_key(std::string_view key) override {
        pending_.push_back({std::string(key), {}});
    }

    void
    insert_value(const PostingsInfo& info) override {
        assert(!pending_.empty());
        pending_.back().info = info;
    }

    void
    finish() override {
        size_t num_blocks =
            (pending_.size() + kDictBlockSize - 1) / kDictBlockSize;

        // .idx: [num_blocks] then per block [token_len][token][offset]
        write_varuint(idx_writer_, num_blocks);

        for (size_t block = 0; block < num_blocks; block++) {
            size_t start = block * kDictBlockSize;
            size_t end = std::min(start + kDictBlockSize, pending_.size());
            size_t count = end - start;

            // Sparse index entry
            const std::string& first_token = pending_[start].key;
            write_varuint(idx_writer_, first_token.size());
            idx_writer_->write(first_token.data(), first_token.size());
            write_varuint(idx_writer_, dct_writer_->offset());

            // Dictionary block: [count] then per token [token_len][token][PostingsInfo]
            write_varuint(dct_writer_, count);
            for (size_t i = start; i < end; i++) {
                write_varuint(dct_writer_, pending_[i].key.size());
                dct_writer_->write(pending_[i].key.data(),
                                   pending_[i].key.size());
                pending_[i].info.serialize(dct_writer_);
            }
        }
    }

 private:
    struct Entry {
        std::string key;
        PostingsInfo info;
    };

    FileWriter* idx_writer_;
    FileWriter* dct_writer_;
    std::vector<Entry> pending_;
};

// --- SSTable Reader ---
// Implements TermDictionaryReader.
// Loads sparse index from .idx, reads dictionary blocks from .dct on demand.

class SortedBlockKeyMapReader : public TermDictionaryReader {
 public:
    SortedBlockKeyMapReader(FileReader* idx_reader, FileReader* dct_reader)
        : dct_reader_(dct_reader) {
        load_sparse_index(idx_reader);
    }

    std::optional<PostingsInfo>
    lookup(const uint8_t* token, size_t token_len) override {
        int block_idx = sparse_index_.find_block(token, token_len);
        if (block_idx < 0)
            return std::nullopt;

        DictBlock block = load_dict_block(block_idx);
        int pos = block.find(token, token_len);
        if (pos < 0)
            return std::nullopt;

        return block.infos[pos];
    }

    const SparseIndex&
    sparse_index() const {
        return sparse_index_;
    }

 private:
    void
    load_sparse_index(FileReader* idx_reader) {
        size_t file_size = static_cast<size_t>(idx_reader->file_size());
        std::vector<uint8_t> buf(file_size);
        idx_reader->read(0, file_size, buf.data());
        const uint8_t* ptr = buf.data();

        uint64_t num_blocks = decode_varuint(ptr);
        sparse_index_.entries.resize(num_blocks);

        for (uint64_t i = 0; i < num_blocks; i++) {
            uint64_t token_len = decode_varuint(ptr);
            sparse_index_.entries[i].first_token.assign(
                reinterpret_cast<const char*>(ptr), token_len);
            ptr += token_len;
            sparse_index_.entries[i].block_offset = decode_varuint(ptr);
        }
    }

    DictBlock
    load_dict_block(int block_idx) {
        uint64_t offset = sparse_index_.entries[block_idx].block_offset;

        // Block extends from its offset to the next block's offset (or EOF).
        uint64_t end_offset;
        if (block_idx + 1 < static_cast<int>(sparse_index_.entries.size())) {
            end_offset = sparse_index_.entries[block_idx + 1].block_offset;
        } else {
            end_offset = dct_reader_->file_size();
        }

        size_t block_size = static_cast<size_t>(end_offset - offset);
        std::vector<uint8_t> buf(block_size);
        dct_reader_->read(offset, block_size, buf.data());

        const uint8_t* ptr = buf.data();
        uint64_t count = decode_varuint(ptr);

        DictBlock block;
        block.tokens.resize(count);
        block.infos.resize(count);

        for (uint64_t i = 0; i < count; i++) {
            uint64_t token_len = decode_varuint(ptr);
            block.tokens[i].assign(reinterpret_cast<const char*>(ptr),
                                   token_len);
            ptr += token_len;
            block.infos[i] = PostingsInfo::deserialize(ptr);
        }

        return block;
    }

    SparseIndex sparse_index_;
    FileReader* dct_reader_;
};

}  // namespace milvus::index::inverted
