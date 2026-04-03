#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include "index/inverted/indexer/SegmentMerger.h"
#include "index/inverted/postings/PostingsWriter.h"
#include "index/inverted/segment/SegmentMeta.h"
#include "index/inverted/segment/SegmentReader.h"
#include "index/inverted/termdict/SortedBlockKeyMap.h"

using namespace milvus::index::inverted;

namespace {

class TempDir {
 public:
    TempDir(const std::string& prefix) {
        path_ = std::filesystem::temp_directory_path() /
                (prefix + "_" +
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
    TempDir& operator=(const TempDir&) = delete;

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

TEST(SegmentMergerTest, TwoSegments) {
    TempDir seg1_dir("merge_seg1");
    TempDir seg2_dir("merge_seg2");
    TempDir merged_dir("merge_result");

    // Segment 1: hello -> {0, 1}, world -> {0, 2}
    {
        Writer writer;
        add_term(writer, "hello", 0);
        add_term(writer, "hello", 1);
        add_term(writer, "world", 0);
        add_term(writer, "world", 2);
        flush_segment(writer, seg1_dir.path(), 3);
    }

    // Segment 2: hello -> {3, 5}, foo -> {4}
    {
        Writer writer;
        add_term(writer, "hello", 3);
        add_term(writer, "hello", 5);
        add_term(writer, "foo", 4);
        flush_segment(writer, seg2_dir.path(), 3);
    }

    // Merge
    SegmentReader reader1(seg1_dir.path());
    SegmentReader reader2(seg2_dir.path());
    std::vector<SegmentReader> readers;
    readers.push_back(std::move(reader1));
    readers.push_back(std::move(reader2));

    {
        LocalFileWriter pst_writer(merged_dir.path() + "/" + kPstFileName);
        LocalFileWriter idx_writer(merged_dir.path() + "/" + kIdxFileName);
        LocalFileWriter dct_writer(merged_dir.path() + "/" + kDctFileName);

        SegmentSerializer serializer(
            &pst_writer,
            std::make_unique<SortedBlockKeyMapWriter>(&idx_writer,
                                                      &dct_writer));

        SegmentMerger merger(readers);
        auto meta = merger.write(serializer, SegmentId::generate_random());

        pst_writer.flush();
        idx_writer.flush();
        dct_writer.flush();

        LocalFileWriter meta_writer(merged_dir.path() + "/" + kMetaFileName);
        meta.serialize(&meta_writer);
        meta_writer.flush();

        EXPECT_EQ(meta.num_docs, 6u);
        EXPECT_EQ(meta.max_doc_id, 5u);
    }

    // Verify merged segment
    SegmentReader merged_reader(merged_dir.path());

    // foo -> {4}
    auto foo = merged_reader.term_query(
        reinterpret_cast<const uint8_t*>("foo"), 3);
    EXPECT_EQ(foo.count(), 1u);
    EXPECT_TRUE(foo[4]);

    // hello -> {0, 1, 3, 5}
    auto hello = merged_reader.term_query(
        reinterpret_cast<const uint8_t*>("hello"), 5);
    EXPECT_EQ(hello.count(), 4u);
    EXPECT_TRUE(hello[0]);
    EXPECT_TRUE(hello[1]);
    EXPECT_TRUE(hello[3]);
    EXPECT_TRUE(hello[5]);

    // world -> {0, 2}
    auto world = merged_reader.term_query(
        reinterpret_cast<const uint8_t*>("world"), 5);
    EXPECT_EQ(world.count(), 2u);
    EXPECT_TRUE(world[0]);
    EXPECT_TRUE(world[2]);
}

TEST(SegmentMergerTest, ThreeSegmentsWithOverlappingTerms) {
    TempDir d1("merge3_s1"), d2("merge3_s2"), d3("merge3_s3");
    TempDir dm("merge3_result");

    // Segment 1: a -> {0,1}, b -> {0}
    {
        Writer w;
        add_term(w, "a", 0);
        add_term(w, "a", 1);
        add_term(w, "b", 0);
        flush_segment(w, d1.path(), 2);
    }
    // Segment 2: a -> {2}, c -> {3}
    {
        Writer w;
        add_term(w, "a", 2);
        add_term(w, "c", 3);
        flush_segment(w, d2.path(), 2);
    }
    // Segment 3: b -> {4}, c -> {5}
    {
        Writer w;
        add_term(w, "b", 4);
        add_term(w, "c", 5);
        flush_segment(w, d3.path(), 2);
    }

    std::vector<SegmentReader> readers;
    readers.emplace_back(d1.path());
    readers.emplace_back(d2.path());
    readers.emplace_back(d3.path());

    {
        LocalFileWriter pw(dm.path() + "/" + kPstFileName);
        LocalFileWriter iw(dm.path() + "/" + kIdxFileName);
        LocalFileWriter dw(dm.path() + "/" + kDctFileName);

        SegmentSerializer s(
            &pw,
            std::make_unique<SortedBlockKeyMapWriter>(&iw, &dw));

        SegmentMerger merger(readers);
        auto meta = merger.write(s, SegmentId::generate_random());

        pw.flush(); iw.flush(); dw.flush();

        LocalFileWriter mw(dm.path() + "/" + kMetaFileName);
        meta.serialize(&mw);
        mw.flush();
    }

    SegmentReader mr(dm.path());

    // a -> {0,1,2}
    auto a = mr.term_query(reinterpret_cast<const uint8_t*>("a"), 1);
    EXPECT_EQ(a.count(), 3u);

    // b -> {0,4}
    auto b = mr.term_query(reinterpret_cast<const uint8_t*>("b"), 1);
    EXPECT_EQ(b.count(), 2u);

    // c -> {3,5}
    auto c = mr.term_query(reinterpret_cast<const uint8_t*>("c"), 1);
    EXPECT_EQ(c.count(), 2u);
}
