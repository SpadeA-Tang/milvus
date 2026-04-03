// Ported from tantivy's indexer/merger.rs
//
// Merges multiple segments into one.
//
// Simplified for our use case:
// - External doc_ids (no remapping needed)
// - No deletes (no alive_bitset filtering)
// - No term_freq / positions (Phase 1: doc_id only)
// - No store / fast fields / fieldnorms
//
// Core algorithm: K-way merge sorted terms via TermMerger,
// for each term merge posting lists via min-heap on doc_ids,
// write through SegmentSerializer.

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <string>
#include <vector>

#include "index/inverted/postings/PostingsSerializer.h"
#include "index/inverted/postings/SegmentSerializer.h"
#include "index/inverted/segment/SegmentMeta.h"
#include "index/inverted/segment/SegmentReader.h"
#include "index/inverted/termdict/SortedBlockKeyMap.h"
#include "index/inverted/termdict/TermMerger.h"

namespace milvus::index::inverted {

class SegmentMerger {
 public:
    explicit SegmentMerger(std::vector<SegmentReader>& readers)
        : readers_(readers) {
    }

    // Write the merged segment through the serializer.
    // Returns the SegmentMeta for the merged segment.
    //
    // Corresponds to tantivy's IndexMerger::write() +
    // write_postings_for_field().
    SegmentMeta
    write(SegmentSerializer& serializer, SegmentId merged_id) {
        // Build TermStreamers from all readers' dictionaries.
        std::vector<TermStreamer> streamers;
        streamers.reserve(readers_.size());
        for (auto& reader : readers_) {
            streamers.emplace_back(reader.dict_reader());
        }

        TermMerger merger(streamers);

        uint32_t total_docs = 0;
        uint32_t max_doc_id = 0;
        for (const auto& reader : readers_) {
            total_docs += reader.meta().num_docs;
            max_doc_id = std::max(max_doc_id, reader.meta().max_doc_id);
        }

        // K-way merge terms.
        while (merger.advance()) {
            const std::string& term_key = merger.key();
            const auto& current = merger.current_streamers();

            // Collect all doc_ids for this term across segments.
            // Use a min-heap to merge-sort doc_ids (they are sorted
            // within each segment's posting list).
            //
            // This mirrors serialize_merged_terms_for_user_id in tantivy.
            std::vector<uint32_t> merged_doc_ids;
            uint32_t total_doc_freq = 0;

            for (const auto& item : current) {
                const PostingsInfo& info = item.streamer->value();
                total_doc_freq += info.cardinality;

                if (info.encoding == PostingEncoding::kInline) {
                    for (uint32_t i = 0; i < info.cardinality; i++) {
                        merged_doc_ids.push_back(info.inline_docs[i]);
                    }
                } else {
                    std::vector<uint32_t> doc_ids;
                    PostingsDecoder::decode(
                        readers_[item.segment_ord].pst_reader(),
                        info.file_offset,
                        info.data_size,
                        info.cardinality,
                        doc_ids);
                    merged_doc_ids.insert(
                        merged_doc_ids.end(), doc_ids.begin(), doc_ids.end());
                }
            }

            // Sort doc_ids (merge from multiple segments).
            std::sort(merged_doc_ids.begin(), merged_doc_ids.end());

            // Write through serializer.
            serializer.new_term(
                reinterpret_cast<const uint8_t*>(term_key.data()),
                term_key.size(),
                static_cast<uint32_t>(merged_doc_ids.size()));

            for (uint32_t doc_id : merged_doc_ids) {
                serializer.write_doc(doc_id, 0, nullptr, 0);
            }

            serializer.close_term();
        }

        serializer.close();

        SegmentMeta meta;
        meta.segment_id = merged_id;
        meta.num_docs = total_docs;
        meta.max_doc_id = max_doc_id;
        return meta;
    }

 private:
    std::vector<SegmentReader>& readers_;
};

}  // namespace milvus::index::inverted
