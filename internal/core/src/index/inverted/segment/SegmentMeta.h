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

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "index/inverted/storage/FileIO.h"
#include "index/inverted/postings/PostingFormat.h"

namespace milvus::index::inverted {

// File name constants for segment directory.
inline const std::string kDctFileName = "term.dct";
inline const std::string kIdxFileName = "term.idx";
inline const std::string kPstFileName = "posting.pst";
inline const std::string kMetaFileName = "segment.meta";

// Uuid identifying a segment.
//
// Ported from tantivy's SegmentId (segment_id.rs).
// Uses a simple auto-increment counter for reproducibility.
struct SegmentId {
    uint64_t id;

    bool
    operator==(const SegmentId& other) const {
        return id == other.id;
    }
    bool
    operator!=(const SegmentId& other) const {
        return id != other.id;
    }
    bool
    operator<(const SegmentId& other) const {
        return id < other.id;
    }

    static SegmentId
    generate_random() {
        static std::atomic<uint64_t> counter{0};
        return SegmentId{counter.fetch_add(1, std::memory_order_relaxed)};
    }
};

struct SegmentIdHash {
    size_t
    operator()(const SegmentId& sid) const {
        return std::hash<uint64_t>{}(sid.id);
    }
};

// SegmentMeta contains simple meta information about a segment.
//
// Ported from tantivy's SegmentMeta (index_meta.rs).
// Simplified: no delete support, no TrackedObject.
struct SegmentMeta {
    SegmentId segment_id{};
    uint32_t num_docs = 0;
    uint32_t max_doc_id = 0;

    SegmentId
    id() const {
        return segment_id;
    }

    // num_docs returns the number of documents (no deletes, so same as max_doc).
    uint32_t
    num_docs_alive() const {
        return num_docs;
    }

    void
    serialize(FileWriter* writer) const {
        write_varuint(writer, segment_id.id);
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
        meta.segment_id.id = decode_varuint(ptr);
        meta.num_docs = static_cast<uint32_t>(decode_varuint(ptr));
        meta.max_doc_id = static_cast<uint32_t>(decode_varuint(ptr));
        return meta;
    }
};

}  // namespace milvus::index::inverted
