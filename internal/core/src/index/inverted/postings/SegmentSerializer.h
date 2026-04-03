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

#include <cassert>
#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "index/inverted/postings/PostingFormat.h"
#include "index/inverted/postings/PostingsSerializer.h"
#include "index/inverted/termdict/TermDictionary.h"

namespace milvus::index::inverted {

// SegmentSerializer drives term dictionary and postings file writing.
//
// In tantivy this is split into InvertedIndexSerializer (per-segment,
// manages per-field writers) and FieldSerializer (per-field, wraps term
// dict builder + postings serializer). Since we have no Field concept,
// these are merged into one class.
//
// File writers and dict writer are passed in from outside (caller owns
// the lifetime), matching tantivy's pattern where InvertedIndexSerializer
// owns the writers and FieldSerializer borrows them.

class SegmentSerializer {
 public:
    SegmentSerializer(FileWriter* postings_writer,
                      std::unique_ptr<TermDictionaryWriter> dict_writer)
        : postings_serializer_(postings_writer),
          dict_writer_(std::move(dict_writer)) {
    }

    void
    new_term(const uint8_t* term, size_t term_len, uint32_t term_doc_freq) {
        assert(!term_open_ && "new_term called without closing previous term");
        term_open_ = true;
        postings_serializer_.clear();
        dict_writer_->insert_key(
            std::string_view(reinterpret_cast<const char*>(term), term_len));

        current_info_ = PostingsInfo{};
        current_info_.cardinality = term_doc_freq;
        inline_idx_ = 0;

        if (term_doc_freq <= PostingsInfo::kInlineMax) {
            current_info_.encoding = PostingEncoding::kInline;
        } else {
            current_info_.encoding = PostingEncoding::kBitpacking;
            current_info_.file_offset = postings_serializer_.written_bytes();
        }
    }

    void
    write_doc(uint32_t doc_id,
              uint32_t term_freq,
              const uint32_t* position_deltas,
              size_t positions_len) {
        if (current_info_.encoding == PostingEncoding::kInline) {
            current_info_.inline_docs[inline_idx_++] = doc_id;
        } else {
            postings_serializer_.write_doc(doc_id);
        }
        // Phase 2: store term_freq and position_deltas
    }

    void
    close_term() {
        if (!term_open_) {
            return;
        }

        if (current_info_.encoding == PostingEncoding::kBitpacking) {
            postings_serializer_.close_term(current_info_.cardinality);
            current_info_.data_size = static_cast<uint32_t>(
                postings_serializer_.written_bytes() -
                current_info_.file_offset);
        }

        dict_writer_->insert_value(current_info_);
        term_open_ = false;
    }

    void
    close() {
        close_term();
        dict_writer_->finish();
    }

 private:
    PostingsSerializer postings_serializer_;
    std::unique_ptr<TermDictionaryWriter> dict_writer_;

    PostingsInfo current_info_;
    uint32_t inline_idx_ = 0;
    bool term_open_ = false;
};

}  // namespace milvus::index::inverted
