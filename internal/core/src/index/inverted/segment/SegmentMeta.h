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
#include <string>
#include <vector>

#include "index/inverted/storage/FileIO.h"
#include "index/inverted/postings/PostingFormat.h"

namespace milvus::index::inverted {

struct SegmentMeta {
    uint32_t num_docs = 0;
    uint32_t max_doc_id = 0;

    void
    serialize(FileWriter* writer) const {
        write_varuint(writer, num_docs);
        write_varuint(writer, max_doc_id);
    }

    static SegmentMeta
    deserialize(FileReader* reader) {
        size_t file_size = static_cast<size_t>(reader->file_size());
        std::vector<uint8_t> buf(file_size);
        reader->read(0, file_size, buf.data());
        const uint8_t* ptr = buf.data();

        SegmentMeta meta;
        meta.num_docs = static_cast<uint32_t>(decode_varuint(ptr));
        meta.max_doc_id = static_cast<uint32_t>(decode_varuint(ptr));
        return meta;
    }
};

// File name constants for segment directory.
inline const std::string kDctFileName = "term.dct";
inline const std::string kIdxFileName = "term.idx";
inline const std::string kPstFileName = "posting.pst";
inline const std::string kMetaFileName = "segment.meta";

}  // namespace milvus::index::inverted
