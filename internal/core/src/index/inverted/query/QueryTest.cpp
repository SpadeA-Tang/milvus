#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include "index/inverted/postings/PostingsWriter.h"
#include "index/inverted/segment/SegmentReader.h"
#include "index/inverted/segment/SegmentMeta.h"
#include "index/inverted/termdict/SortedBlockKeyMap.h"
#include "index/inverted/query/Query.h"
#include "index/inverted/query/TermQuery.h"
#include "index/inverted/query/BoolQuery.h"

using namespace milvus::index::inverted;

namespace {

class TempDir {
 public:
    TempDir() {
        path_ =
            std::filesystem::temp_directory_path() /
            ("query_test_" +
             std::to_string(reinterpret_cast<uintptr_t>(this)));
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

// --- TermQuery ---

TEST(TermQueryTest, Execute) {
    TempDir dir;

    {
        Writer writer;
        add_term(writer, "hello", 0);
        add_term(writer, "hello", 3);
        add_term(writer, "world", 1);
        add_term(writer, "world", 3);
        flush_segment(writer, dir.path(), 4);
    }

    SegmentReader reader(dir.path());

    TermQuery q("hello");
    auto result = q.weight()->execute(reader);
    EXPECT_EQ(result.count(), 2u);
    EXPECT_TRUE(result[0]);
    EXPECT_TRUE(result[3]);
}

// --- BoolQuery ---

TEST(BoolQueryTest, Must) {
    TempDir dir;

    {
        Writer writer;
        add_term(writer, "hello", 0);
        add_term(writer, "hello", 1);
        add_term(writer, "hello", 3);
        add_term(writer, "world", 0);
        add_term(writer, "world", 2);
        add_term(writer, "world", 3);
        flush_segment(writer, dir.path(), 4);
    }

    SegmentReader reader(dir.path());

    // hello AND world -> {0, 3}
    BoolQuery bq;
    bq.add(Occur::Must, std::make_unique<TermQuery>("hello"));
    bq.add(Occur::Must, std::make_unique<TermQuery>("world"));
    auto result = bq.weight()->execute(reader);
    EXPECT_EQ(result.count(), 2u);
    EXPECT_TRUE(result[0]);
    EXPECT_TRUE(result[3]);
}

TEST(BoolQueryTest, Should) {
    TempDir dir;

    {
        Writer writer;
        add_term(writer, "cat", 0);
        add_term(writer, "cat", 1);
        add_term(writer, "dog", 2);
        add_term(writer, "dog", 3);
        flush_segment(writer, dir.path(), 4);
    }

    SegmentReader reader(dir.path());

    // cat OR dog -> {0, 1, 2, 3}
    BoolQuery bq;
    bq.add(Occur::Should, std::make_unique<TermQuery>("cat"));
    bq.add(Occur::Should, std::make_unique<TermQuery>("dog"));
    auto result = bq.weight()->execute(reader);
    EXPECT_EQ(result.count(), 4u);
}

TEST(BoolQueryTest, MustNot) {
    TempDir dir;

    {
        Writer writer;
        add_term(writer, "hello", 0);
        add_term(writer, "hello", 1);
        add_term(writer, "hello", 2);
        add_term(writer, "hello", 3);
        add_term(writer, "spam", 1);
        add_term(writer, "spam", 3);
        flush_segment(writer, dir.path(), 4);
    }

    SegmentReader reader(dir.path());

    // hello AND NOT spam -> {0, 2}
    BoolQuery bq;
    bq.add(Occur::Must, std::make_unique<TermQuery>("hello"));
    bq.add(Occur::MustNot, std::make_unique<TermQuery>("spam"));
    auto result = bq.weight()->execute(reader);
    EXPECT_EQ(result.count(), 2u);
    EXPECT_TRUE(result[0]);
    EXPECT_TRUE(result[2]);
}

TEST(BoolQueryTest, Combined) {
    TempDir dir;

    {
        Writer writer;
        add_term(writer, "a", 0);
        add_term(writer, "a", 1);
        add_term(writer, "a", 2);
        add_term(writer, "b", 0);
        add_term(writer, "b", 2);
        add_term(writer, "b", 3);
        add_term(writer, "c", 1);
        add_term(writer, "c", 2);
        flush_segment(writer, dir.path(), 4);
    }

    SegmentReader reader(dir.path());

    // (a AND b) AND NOT c -> {0}
    BoolQuery bq;
    bq.add(Occur::Must, std::make_unique<TermQuery>("a"));
    bq.add(Occur::Must, std::make_unique<TermQuery>("b"));
    bq.add(Occur::MustNot, std::make_unique<TermQuery>("c"));
    auto result = bq.weight()->execute(reader);
    EXPECT_EQ(result.count(), 1u);
    EXPECT_TRUE(result[0]);
}
