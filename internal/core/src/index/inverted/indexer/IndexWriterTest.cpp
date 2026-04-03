#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include "index/inverted/indexer/IndexReader.h"
#include "index/inverted/indexer/IndexWriter.h"

using namespace milvus::index::inverted;

namespace {

class TempDir {
 public:
    explicit TempDir(const std::string& prefix) {
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

 private:
    std::filesystem::path path_;
    std::string path_str_;
};

}  // namespace

// --- IndexWriter basic ---

TEST(IndexWriterTest, WriteAndCommit) {
    TempDir dir("iw_basic");
    {
        IndexWriter writer(dir.path());
        writer.set_merge_policy(std::make_shared<NoMergePolicy>());

        writer.add_document(0, reinterpret_cast<const uint8_t*>("hello"), 5);
        writer.add_document(1, reinterpret_cast<const uint8_t*>("hello"), 5);
        writer.add_document(2, reinterpret_cast<const uint8_t*>("world"), 5);

        auto opstamp = writer.commit();
        EXPECT_GE(opstamp, 0u);
    }

    // Verify meta exists.
    IndexDirectory directory(dir.path());
    EXPECT_TRUE(directory.has_meta());
    auto meta = directory.load_meta();
    EXPECT_EQ(meta.segments.size(), 1u);
    EXPECT_EQ(meta.segments[0].num_docs, 3u);
}

TEST(IndexWriterTest, EmptyCommit) {
    TempDir dir("iw_empty");
    IndexWriter writer(dir.path());
    writer.set_merge_policy(std::make_shared<NoMergePolicy>());
    auto opstamp = writer.commit();
    EXPECT_GE(opstamp, 0u);

    auto meta = writer.directory().load_meta();
    EXPECT_TRUE(meta.segments.empty());
}

// --- IndexReader basic ---

TEST(IndexReaderTest, ReadAfterCommit) {
    TempDir dir("ir_basic");

    IndexWriter writer(dir.path());
    writer.set_merge_policy(std::make_shared<NoMergePolicy>());

    writer.add_document(0, reinterpret_cast<const uint8_t*>("hello"), 5);
    writer.add_document(1, reinterpret_cast<const uint8_t*>("hello"), 5);
    writer.add_document(2, reinterpret_cast<const uint8_t*>("world"), 5);
    writer.commit();

    IndexReader reader(&writer.directory());
    auto searcher = reader.searcher();
    EXPECT_EQ(searcher->num_segments(), 1u);

    auto result = searcher->term_query(
        reinterpret_cast<const uint8_t*>("hello"), 5);
    EXPECT_EQ(result.count(), 2u);
    EXPECT_TRUE(result[0]);
    EXPECT_TRUE(result[1]);

    auto world = searcher->term_query(
        reinterpret_cast<const uint8_t*>("world"), 5);
    EXPECT_EQ(world.count(), 1u);
    EXPECT_TRUE(world[2]);
}

TEST(IndexReaderTest, Reload) {
    TempDir dir("ir_reload");

    IndexWriter writer(dir.path());
    writer.set_merge_policy(std::make_shared<NoMergePolicy>());

    // First commit.
    writer.add_document(0, reinterpret_cast<const uint8_t*>("foo"), 3);
    writer.commit();

    IndexReader reader(&writer.directory());
    auto s1 = reader.searcher();
    EXPECT_EQ(s1->term_query(
                      reinterpret_cast<const uint8_t*>("foo"), 3)
                  .count(),
              1u);

    // Second commit (new segment).
    writer.add_document(1, reinterpret_cast<const uint8_t*>("bar"), 3);
    writer.commit();

    // Before reload, old snapshot still valid.
    auto foo1 = s1->term_query(
        reinterpret_cast<const uint8_t*>("bar"), 3);
    EXPECT_EQ(foo1.count(), 0u);

    // After reload, sees new data.
    reader.reload();
    auto s2 = reader.searcher();
    auto bar2 = s2->term_query(
        reinterpret_cast<const uint8_t*>("bar"), 3);
    EXPECT_EQ(bar2.count(), 1u);
    EXPECT_TRUE(bar2[1]);
}

// --- Merge ---

TEST(IndexWriterTest, MergeAfterCommit) {
    TempDir dir("iw_merge");

    IndexWriter writer(dir.path());
    // Use LogMergePolicy with low threshold to trigger merge.
    auto policy = std::make_shared<LogMergePolicy>();
    policy->set_min_num_segments(2);
    policy->set_min_layer_size(1);
    writer.set_merge_policy(policy);

    // Create 3 small segments by committing with tiny memory budget.
    // Each add_document + commit creates one segment.
    writer.add_document(0, reinterpret_cast<const uint8_t*>("a"), 1);
    writer.commit();
    writer.add_document(1, reinterpret_cast<const uint8_t*>("b"), 1);
    writer.commit();
    writer.add_document(2, reinterpret_cast<const uint8_t*>("a"), 1);
    // This commit should trigger merge of the 3 segments.
    writer.commit();

    // Verify merged result.
    IndexReader reader(&writer.directory());
    auto searcher = reader.searcher();

    auto a = searcher->term_query(
        reinterpret_cast<const uint8_t*>("a"), 1);
    EXPECT_EQ(a.count(), 2u);
    EXPECT_TRUE(a[0]);
    EXPECT_TRUE(a[2]);

    auto b = searcher->term_query(
        reinterpret_cast<const uint8_t*>("b"), 1);
    EXPECT_EQ(b.count(), 1u);
    EXPECT_TRUE(b[1]);
}

// --- Multiple terms per document ---

TEST(IndexWriterTest, MultipleTermsPerDoc) {
    TempDir dir("iw_multi_terms");

    IndexWriter writer(dir.path());
    writer.set_merge_policy(std::make_shared<NoMergePolicy>());

    // Doc 0 has two terms.
    writer.add_document(0, reinterpret_cast<const uint8_t*>("hello"), 5);
    writer.add_document(0, reinterpret_cast<const uint8_t*>("world"), 5);
    // Doc 1 has one term.
    writer.add_document(1, reinterpret_cast<const uint8_t*>("hello"), 5);
    writer.commit();

    IndexReader reader(&writer.directory());
    auto searcher = reader.searcher();

    auto hello = searcher->term_query(
        reinterpret_cast<const uint8_t*>("hello"), 5);
    EXPECT_EQ(hello.count(), 2u);

    auto world = searcher->term_query(
        reinterpret_cast<const uint8_t*>("world"), 5);
    EXPECT_EQ(world.count(), 1u);
    EXPECT_TRUE(world[0]);
}

// --- Memory budget flush ---

TEST(IndexWriterTest, MemoryBudgetFlush) {
    TempDir dir("iw_mem_flush");

    // Tiny budget to force multiple segments.
    IndexWriter writer(dir.path(), 1024);
    writer.set_merge_policy(std::make_shared<NoMergePolicy>());

    // Add many documents to exceed budget.
    for (uint32_t i = 0; i < 500; i++) {
        std::string term = "term_" + std::to_string(i % 50);
        writer.add_document(
            i,
            reinterpret_cast<const uint8_t*>(term.data()),
            term.size());
    }
    writer.commit();

    auto meta = writer.directory().load_meta();
    // Should have multiple segments due to memory limit.
    EXPECT_GT(meta.segments.size(), 1u);

    // All documents should be queryable.
    IndexReader reader(&writer.directory());
    auto searcher = reader.searcher();
    auto result = searcher->term_query(
        reinterpret_cast<const uint8_t*>("term_0"), 6);
    // term_0 appears for doc_ids 0, 50, 100, 150, 200, 250, 300, 350, 400, 450
    EXPECT_EQ(result.count(), 10u);
}
