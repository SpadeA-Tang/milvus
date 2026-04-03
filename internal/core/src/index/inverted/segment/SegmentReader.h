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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include "index/inverted/segment/SegmentMeta.h"
#include "index/inverted/postings/PostingsSerializer.h"
#include "index/inverted/termdict/SortedBlockKeyMap.h"

namespace milvus::index::inverted {

using Bitset = boost::dynamic_bitset<>;

// SegmentReader loads a serialized segment and answers term lookups.
// Thread-safe for concurrent reads (all I/O via pread, no shared mutable state).

class SegmentReader {
 public:
    explicit SegmentReader(const std::string& dir)
        : pst_reader_(
              std::make_unique<LocalFileReader>(dir + "/" + kPstFileName)),
          idx_reader_(
              std::make_unique<LocalFileReader>(dir + "/" + kIdxFileName)),
          dct_reader_(
              std::make_unique<LocalFileReader>(dir + "/" + kDctFileName)) {
        // Load metadata
        LocalFileReader meta_reader(dir + "/" + kMetaFileName);
        meta_ = SegmentMeta::deserialize(&meta_reader);

        // Load term dictionary (sparse index loaded eagerly)
        dict_reader_ = std::make_unique<SortedBlockKeyMapReader>(
            idx_reader_.get(), dct_reader_.get());
    }

    // Lookup a term and return matching doc_ids as a Bitset.
    Bitset
    term_query(const uint8_t* term, size_t term_len) const {
        Bitset bitset(num_bits());

        auto info = dict_reader_->lookup(term, term_len);
        if (!info.has_value()) {
            return bitset;
        }

        if (info->encoding == PostingEncoding::kInline) {
            for (uint32_t i = 0; i < info->cardinality; i++) {
                bitset.set(info->inline_docs[i]);
            }
        } else {
            std::vector<uint32_t> doc_ids;
            PostingsDecoder::decode(pst_reader_.get(),
                                    info->file_offset,
                                    info->data_size,
                                    info->cardinality,
                                    doc_ids);
            for (uint32_t id : doc_ids) {
                bitset.set(id);
            }
        }

        return bitset;
    }

    // Lookup a term and return its PostingsInfo.
    std::optional<PostingsInfo>
    lookup_term(const uint8_t* term, size_t term_len) const {
        return dict_reader_->lookup(term, term_len);
    }

    const SegmentMeta&
    meta() const {
        return meta_;
    }

    // Bitset size: max_doc_id + 1.
    uint32_t
    num_bits() const {
        return meta_.max_doc_id + 1;
    }

 private:
    SegmentMeta meta_;
    std::unique_ptr<LocalFileReader> pst_reader_;
    std::unique_ptr<LocalFileReader> idx_reader_;
    std::unique_ptr<LocalFileReader> dct_reader_;
    std::unique_ptr<SortedBlockKeyMapReader> dict_reader_;
};

}  // namespace milvus::index::inverted
