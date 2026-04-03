// Ported from tantivy's index/index_meta.rs
//
// IndexMeta is the top-level metadata for an index, serialized to meta.json.
// Contains the list of committed segments and the last commit opstamp.
//
// Simplified: no schema, no index_settings, no payload.

#pragma once

#include <cstdint>
#include <vector>

#include "index/inverted/indexer/Stamper.h"
#include "index/inverted/segment/SegmentMeta.h"

namespace milvus::index::inverted {

inline const std::string kIndexMetaFileName = "meta";

struct IndexMeta {
    std::vector<SegmentMeta> segments;
    Opstamp opstamp = 0;

    void
    serialize(FileWriter* writer) const {
        write_varuint(writer, opstamp);
        write_varuint(writer, segments.size());
        for (const auto& seg : segments) {
            seg.serialize(writer);
        }
    }

    static IndexMeta
    deserialize(FileReader* reader) {
        size_t file_size = static_cast<size_t>(reader->file_size());
        std::vector<uint8_t> buf(file_size);
        reader->read(0, file_size, buf.data());
        const uint8_t* ptr = buf.data();

        IndexMeta meta;
        meta.opstamp = decode_varuint(ptr);
        uint64_t num_segments = decode_varuint(ptr);
        meta.segments.reserve(num_segments);
        for (uint64_t i = 0; i < num_segments; i++) {
            SegmentMeta seg;
            seg.segment_id.id = decode_varuint(ptr);
            seg.num_docs = static_cast<uint32_t>(decode_varuint(ptr));
            seg.max_doc_id = static_cast<uint32_t>(decode_varuint(ptr));
            meta.segments.push_back(seg);
        }
        return meta;
    }
};

}  // namespace milvus::index::inverted
