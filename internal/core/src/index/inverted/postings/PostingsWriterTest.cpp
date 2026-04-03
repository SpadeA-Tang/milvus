#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "index/inverted/postings/PostingsWriter.h"
#include "index/inverted/segment/SegmentReader.h"
#include "index/inverted/segment/SegmentMeta.h"
#include "index/inverted/termdict/SortedBlockKeyMap.h"

using namespace milvus::index::inverted;

namespace {

// Helper: create a temp directory and clean up on destruction.
class TempDir {
 public:
    TempDir() {
        path_ =
            std::filesystem::temp_directory_path() /
            ("seg_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
        std::filesystem::create_directories(path_);
        path_str_ = path_.string();
    }
    ~TempDir() {
        std::filesystem::remove_all(path_);
    }
    const std::string&
    path() const {
        return path_str_;
    }
    TempDir(const TempDir&) = delete;
    TempDir&
    operator=(const TempDir&) = delete;

 private:
    std::filesystem::path path_;
    std::string path_str_;
};

using Writer = PostingsWriter<DocIdRecorder>;

void
add_term(Writer& writer, const std::string& term, uint32_t doc_id) {
    writer.subscribe(
        doc_id, 0, reinterpret_cast<const uint8_t*>(term.data()), term.size());
}

void
flush_segment(Writer& writer, const std::string& dir, uint32_t num_docs) {
    LocalFileWriter pst_writer(dir + "/" + kPstFileName);
    LocalFileWriter idx_writer(dir + "/" + kIdxFileName);
    LocalFileWriter dct_writer(dir + "/" + kDctFileName);

    SegmentSerializer serializer(
        &pst_writer,
        std::make_unique<SortedBlockKeyMapWriter>(&idx_writer, &dct_writer));
    serialize_postings(writer, serializer);
    serializer.close();

    pst_writer.flush();
    idx_writer.flush();
    dct_writer.flush();

    LocalFileWriter meta_writer(dir + "/" + kMetaFileName);
    SegmentMeta meta;
    meta.num_docs = num_docs;
    meta.max_doc_id = writer.max_doc_id();
    meta.serialize(&meta_writer);
    meta_writer.flush();
}

}  // namespace

// --- PostingsWriter ---

TEST(PostingsWriterTest, BasicAccumulation) {
    Writer writer;

    add_term(writer, "hello", 0);
    add_term(writer, "world", 0);
    add_term(writer, "hello", 1);
    add_term(writer, "hello", 5);

    EXPECT_EQ(writer.num_terms(), 2u);
    EXPECT_EQ(writer.max_doc_id(), 5u);
    EXPECT_GT(writer.mem_usage(), 0u);
}

// --- Round-trip: Writer -> Serializer -> Reader ---

TEST(SegmentRoundTripTest, SmallPostings) {
    TempDir dir;
    uint32_t num_docs = 10;

    // Write: 3 terms, all inline (cardinality <= 6)
    {
        Writer writer;
        add_term(writer, "apple", 1);
        add_term(writer, "apple", 3);
        add_term(writer, "banana", 0);
        add_term(writer, "banana", 2);
        add_term(writer, "banana", 4);
        add_term(writer, "cherry", 9);
        flush_segment(writer, dir.path(), num_docs);
    }

    // Read and verify
    SegmentReader reader(dir.path());
    EXPECT_EQ(reader.meta().num_docs, num_docs);
    EXPECT_EQ(reader.meta().max_doc_id, 9u);
    EXPECT_EQ(reader.num_bits(), 10u);

    // apple: {1, 3}
    auto apple =
        reader.term_query(reinterpret_cast<const uint8_t*>("apple"), 5);
    EXPECT_EQ(apple.count(), 2u);
    EXPECT_TRUE(apple[1]);
    EXPECT_TRUE(apple[3]);

    // banana: {0, 2, 4}
    auto banana =
        reader.term_query(reinterpret_cast<const uint8_t*>("banana"), 6);
    EXPECT_EQ(banana.count(), 3u);
    EXPECT_TRUE(banana[0]);
    EXPECT_TRUE(banana[2]);
    EXPECT_TRUE(banana[4]);

    // cherry: {9}
    auto cherry =
        reader.term_query(reinterpret_cast<const uint8_t*>("cherry"), 6);
    EXPECT_EQ(cherry.count(), 1u);
    EXPECT_TRUE(cherry[9]);

    // Non-existent term
    auto none =
        reader.term_query(reinterpret_cast<const uint8_t*>("missing"), 7);
    EXPECT_EQ(none.count(), 0u);
}

TEST(SegmentRoundTripTest, LargePostings) {
    TempDir dir;
    uint32_t num_docs = 500;

    // Write: one term with 500 doc_ids (exercises bitpacking + skip data)
    {
        Writer writer;
        for (uint32_t i = 0; i < num_docs; i++) {
            add_term(writer, "frequent", i * 3);
        }
        flush_segment(writer, dir.path(), num_docs);
    }

    SegmentReader reader(dir.path());
    auto result =
        reader.term_query(reinterpret_cast<const uint8_t*>("frequent"), 8);
    EXPECT_EQ(result.count(), num_docs);
    for (uint32_t i = 0; i < num_docs; i++) {
        EXPECT_TRUE(result[i * 3]) << "missing doc_id=" << i * 3;
    }
}

TEST(SegmentRoundTripTest, MixedPostings) {
    TempDir dir;
    uint32_t max_doc = 2000;

    {
        Writer writer;

        // Inline term (3 docs)
        add_term(writer, "rare", 10);
        add_term(writer, "rare", 500);
        add_term(writer, "rare", 1999);

        // Medium term (50 docs, vint only, no skip)
        for (uint32_t i = 0; i < 50; i++) {
            add_term(writer, "medium", i * 40);
        }

        // Large term (300 docs, bitpacking + skip)
        for (uint32_t i = 0; i < 300; i++) {
            add_term(writer, "common", i * 5);
        }

        flush_segment(writer, dir.path(), max_doc);
    }

    SegmentReader reader(dir.path());

    // rare: inline
    auto rare = reader.term_query(reinterpret_cast<const uint8_t*>("rare"), 4);
    EXPECT_EQ(rare.count(), 3u);
    EXPECT_TRUE(rare[10]);
    EXPECT_TRUE(rare[500]);
    EXPECT_TRUE(rare[1999]);

    // medium: vint-encoded
    auto medium =
        reader.term_query(reinterpret_cast<const uint8_t*>("medium"), 6);
    EXPECT_EQ(medium.count(), 50u);
    for (uint32_t i = 0; i < 50; i++) {
        EXPECT_TRUE(medium[i * 40]);
    }

    // common: bitpacking + skip
    auto common =
        reader.term_query(reinterpret_cast<const uint8_t*>("common"), 6);
    EXPECT_EQ(common.count(), 300u);
    for (uint32_t i = 0; i < 300; i++) {
        EXPECT_TRUE(common[i * 5]);
    }
}

TEST(SegmentRoundTripTest, ManyTerms) {
    TempDir dir;
    uint32_t num_terms = 1000;
    uint32_t docs_per_term = 10;
    uint32_t num_docs = num_terms * docs_per_term;

    {
        Writer writer;
        // Generate 1000 terms: "term_0000", "term_0001", ...
        // Each term has 10 docs: term_i -> {i*10, i*10+1, ..., i*10+9}
        for (uint32_t t = 0; t < num_terms; t++) {
            char buf[16];
            snprintf(buf, sizeof(buf), "term_%04u", t);
            std::string term(buf);
            for (uint32_t d = 0; d < docs_per_term; d++) {
                add_term(writer, term, t * docs_per_term + d);
            }
        }
        flush_segment(writer, dir.path(), num_docs);
    }

    SegmentReader reader(dir.path());
    EXPECT_EQ(reader.meta().num_docs, num_docs);

    // Verify a few random terms
    for (uint32_t t : {0u, 42u, 500u, 999u}) {
        char buf[16];
        snprintf(buf, sizeof(buf), "term_%04u", t);
        auto result = reader.term_query(reinterpret_cast<const uint8_t*>(buf),
                                        strlen(buf));
        EXPECT_EQ(result.count(), docs_per_term) << "term=" << buf;
        for (uint32_t d = 0; d < docs_per_term; d++) {
            EXPECT_TRUE(result[t * docs_per_term + d]) << "term=" << buf;
        }
    }
}
