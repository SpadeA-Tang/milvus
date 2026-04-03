// NativeTextIndex — text indexing with tokenization.
//
// Wraps IndexWriter/IndexReader + Tokenizer to provide a complete
// text search pipeline: tokenize → index → query.
//
// Corresponds to milvus::index::TextMatchIndex but uses our native
// inverted index instead of tantivy Rust FFI for storage/search.

#pragma once

#include <memory>
#include <string>

#include "index/inverted/Tokenizer.h"
#include "index/inverted/indexer/IndexDirectory.h"
#include "index/inverted/indexer/IndexReader.h"
#include "index/inverted/indexer/IndexWriter.h"
#include "index/inverted/indexer/MergePolicy.h"

namespace milvus::index::inverted {

class NativeTextIndex {
 public:
    // Write mode: creates IndexWriter for building.
    NativeTextIndex(const std::string& path,
                    std::shared_ptr<Tokenizer> tokenizer,
                    size_t memory_budget = kDefaultMemoryBudget)
        : path_(path),
          tokenizer_(std::move(tokenizer)),
          writer_(std::make_unique<IndexWriter>(path, memory_budget)) {
    }

    // Read-only mode: opens existing index for queries.
    static std::unique_ptr<NativeTextIndex>
    open(const std::string& path, std::shared_ptr<Tokenizer> tokenizer) {
        auto index = std::unique_ptr<NativeTextIndex>(
            new NativeTextIndex(path, std::move(tokenizer), ReadOnly{}));
        return index;
    }

    // Build from array of texts. Tokenizes each, indexes, commits.
    void
    build(size_t n, const std::string* texts) {
        for (size_t i = 0; i < n; i++) {
            add_text(static_cast<uint32_t>(i), texts[i]);
        }
        commit();
    }

    // Add single text (for growing segment incremental insert).
    void
    add_text(uint32_t doc_id, const std::string& text) {
        auto stream = tokenizer_->token_stream(text);
        while (stream->advance()) {
            const auto& tok = stream->token();
            writer_->add_document(
                doc_id,
                reinterpret_cast<const uint8_t*>(tok.data()),
                tok.size());
        }
    }

    // Commit current data and open/refresh reader.
    void
    commit() {
        writer_->commit();
        if (!reader_) {
            reader_ =
                std::make_unique<IndexReader>(&writer_->directory());
        } else {
            reader_->reload();
        }
    }

    // Reload reader to see latest committed data.
    void
    reload() {
        if (reader_) {
            reader_->reload();
        }
    }

    // Exact term query.
    Bitset
    term_query(const std::string& term) const {
        auto searcher = reader_->searcher();
        return searcher->term_query(
            reinterpret_cast<const uint8_t*>(term.data()), term.size());
    }

    // Match query: tokenize query text, require min_should_match terms
    // to match per document.
    // Corresponds to TextMatchIndex::MatchQuery(query, min_should_match).
    //   min_should_match=1 → OR (any term matches)
    //   min_should_match=N → at least N query terms must match
    Bitset
    match_query(const std::string& query_text,
                uint32_t min_should_match = 1) const {
        auto searcher = reader_->searcher();
        auto stream = tokenizer_->token_stream(query_text);

        std::vector<Bitset> term_results;
        while (stream->advance()) {
            const auto& tok = stream->token();
            term_results.push_back(searcher->term_query(
                reinterpret_cast<const uint8_t*>(tok.data()),
                tok.size()));
        }

        if (term_results.empty()) {
            return Bitset(searcher->num_bits());
        }

        size_t num_bits = searcher->num_bits();
        for (const auto& tr : term_results) {
            if (tr.size() > num_bits) {
                num_bits = tr.size();
            }
        }

        Bitset result(num_bits);
        for (size_t i = 0; i < num_bits; i++) {
            uint32_t count = 0;
            for (const auto& tr : term_results) {
                if (i < tr.size() && tr[i]) {
                    count++;
                }
            }
            if (count >= min_should_match) {
                result.set(i);
            }
        }
        return result;
    }

    void
    set_merge_policy(std::shared_ptr<MergePolicy> policy) {
        writer_->set_merge_policy(std::move(policy));
    }

    IndexWriter&
    writer() {
        return *writer_;
    }

 private:
    struct ReadOnly {};

    // Private constructor for read-only mode.
    NativeTextIndex(const std::string& path,
                    std::shared_ptr<Tokenizer> tokenizer,
                    ReadOnly)
        : path_(path),
          tokenizer_(std::move(tokenizer)),
          directory_(std::make_unique<IndexDirectory>(path)),
          reader_(std::make_unique<IndexReader>(directory_.get())) {
    }

    static void
    merge_or(Bitset& dst, const Bitset& src) {
        if (src.size() > dst.size()) {
            dst.resize(src.size());
        }
        for (size_t i = 0; i < src.size(); i++) {
            if (src[i]) {
                dst.set(i);
            }
        }
    }

    std::string path_;
    std::shared_ptr<Tokenizer> tokenizer_;

    // Read-only mode: standalone directory.
    std::unique_ptr<IndexDirectory> directory_;

    // Write mode: writer owns its directory.
    std::unique_ptr<IndexWriter> writer_;

    // Reader (created after first commit or in read-only mode).
    std::unique_ptr<IndexReader> reader_;
};

}  // namespace milvus::index::inverted
