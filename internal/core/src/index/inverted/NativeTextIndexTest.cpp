#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <vector>

#include "index/inverted/NativeTextIndex.h"

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

auto
make_tokenizer() {
    return std::make_shared<SimpleTokenizer>();
}

}  // namespace

// --- Tokenizer ---

TEST(TokenizerTest, SimpleTokenizer) {
    SimpleTokenizer tokenizer;
    auto stream = tokenizer.token_stream("Hello, World! foo123");

    std::vector<std::string> tokens;
    while (stream->advance()) {
        tokens.push_back(stream->token());
    }
    ASSERT_EQ(tokens.size(), 3u);
    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], "world");
    EXPECT_EQ(tokens[2], "foo123");
}

TEST(TokenizerTest, EmptyText) {
    SimpleTokenizer tokenizer;
    auto stream = tokenizer.token_stream("");

    EXPECT_FALSE(stream->advance());
}

// --- NativeTextIndex: build + term_query ---

TEST(NativeTextIndexTest, BuildAndTermQuery) {
    TempDir dir("nti_basic");

    std::vector<std::string> texts = {
        "the quick brown fox",
        "the lazy dog",
        "quick brown dog",
    };

    NativeTextIndex index(dir.path(), make_tokenizer());
    index.set_merge_policy(std::make_shared<NoMergePolicy>());
    index.build(texts.size(), texts.data());

    // "quick" appears in doc 0, 2.
    auto quick = index.term_query("quick");
    EXPECT_EQ(quick.count(), 2u);
    EXPECT_TRUE(quick[0]);
    EXPECT_TRUE(quick[2]);

    // "dog" appears in doc 1, 2.
    auto dog = index.term_query("dog");
    EXPECT_EQ(dog.count(), 2u);
    EXPECT_TRUE(dog[1]);
    EXPECT_TRUE(dog[2]);

    // "the" appears in doc 0, 1.
    auto the = index.term_query("the");
    EXPECT_EQ(the.count(), 2u);
    EXPECT_TRUE(the[0]);
    EXPECT_TRUE(the[1]);

    // "missing" not found.
    auto missing = index.term_query("missing");
    EXPECT_EQ(missing.count(), 0u);
}

// --- NativeTextIndex: match_query (multi-term OR) ---

TEST(NativeTextIndexTest, MatchQuery) {
    TempDir dir("nti_match");

    std::vector<std::string> texts = {
        "the quick brown fox",   // doc 0
        "the lazy dog",          // doc 1
        "quick brown dog",       // doc 2
    };

    NativeTextIndex index(dir.path(), make_tokenizer());
    index.set_merge_policy(std::make_shared<NoMergePolicy>());
    index.build(texts.size(), texts.data());

    // "quick dog" → tokenize to ["quick", "dog"] → OR → docs 0, 1, 2.
    auto result = index.match_query("quick dog");
    EXPECT_EQ(result.count(), 3u);
    EXPECT_TRUE(result[0]);
    EXPECT_TRUE(result[1]);
    EXPECT_TRUE(result[2]);

    // "fox lazy" → docs 0, 1.
    auto result2 = index.match_query("fox lazy");
    EXPECT_EQ(result2.count(), 2u);
    EXPECT_TRUE(result2[0]);
    EXPECT_TRUE(result2[1]);

    // "nonexistent" → no results.
    auto result3 = index.match_query("nonexistent");
    EXPECT_EQ(result3.count(), 0u);
}

// --- Persistence: build → close → open read-only → query ---

TEST(NativeTextIndexTest, Persistence) {
    TempDir dir("nti_persist");

    std::vector<std::string> texts = {
        "alpha beta gamma",
        "beta gamma delta",
        "gamma delta epsilon",
    };

    // Build and destroy.
    {
        NativeTextIndex index(dir.path(), make_tokenizer());
        index.set_merge_policy(std::make_shared<NoMergePolicy>());
        index.build(texts.size(), texts.data());
    }

    // Open read-only and query.
    auto index = NativeTextIndex::open(dir.path(), make_tokenizer());

    auto beta = index->term_query("beta");
    EXPECT_EQ(beta.count(), 2u);
    EXPECT_TRUE(beta[0]);
    EXPECT_TRUE(beta[1]);

    auto gamma = index->match_query("gamma epsilon");
    EXPECT_EQ(gamma.count(), 3u);
}

// --- Incremental: add_text + commit + reload ---

TEST(NativeTextIndexTest, IncrementalAddAndReload) {
    TempDir dir("nti_incremental");

    NativeTextIndex index(dir.path(), make_tokenizer());
    index.set_merge_policy(std::make_shared<NoMergePolicy>());

    // First batch.
    index.add_text(0, "hello world");
    index.add_text(1, "hello there");
    index.commit();

    auto hello = index.term_query("hello");
    EXPECT_EQ(hello.count(), 2u);

    auto world = index.term_query("world");
    EXPECT_EQ(world.count(), 1u);
    EXPECT_TRUE(world[0]);

    // "goodbye" not indexed yet.
    auto goodbye = index.term_query("goodbye");
    EXPECT_EQ(goodbye.count(), 0u);

    // Second batch.
    index.add_text(2, "goodbye world");
    index.commit();

    // After commit, should see new data.
    goodbye = index.term_query("goodbye");
    EXPECT_EQ(goodbye.count(), 1u);
    EXPECT_TRUE(goodbye[2]);

    // "world" now in doc 0 and 2.
    world = index.term_query("world");
    EXPECT_EQ(world.count(), 2u);
    EXPECT_TRUE(world[0]);
    EXPECT_TRUE(world[2]);
}

// --- Text match: case / punctuation / min_should_match ---

TEST(NativeTextIndexTest, TextMatchCaseAndPunctuation) {
    TempDir dir("nti_case");

    std::vector<std::string> texts = {
        "Hello, World!",
        "hello there",
        "WORLD domination",
    };

    NativeTextIndex index(dir.path(), make_tokenizer());
    index.set_merge_policy(std::make_shared<NoMergePolicy>());
    index.build(texts.size(), texts.data());

    // Tokenizer lowercases + strips punctuation.
    // "HELLO" → "hello" matches doc 0 ("hello") and doc 1 ("hello").
    auto hello = index.term_query("hello");
    EXPECT_EQ(hello.count(), 2u);
    EXPECT_TRUE(hello[0]);
    EXPECT_TRUE(hello[1]);

    // "world" matches doc 0 ("world") and doc 2 ("world").
    auto world = index.term_query("world");
    EXPECT_EQ(world.count(), 2u);
    EXPECT_TRUE(world[0]);
    EXPECT_TRUE(world[2]);

    // Query with punctuation: "Hello!" → tokenized to "hello".
    auto result = index.match_query("Hello!");
    EXPECT_EQ(result.count(), 2u);
    EXPECT_TRUE(result[0]);
    EXPECT_TRUE(result[1]);
}

TEST(NativeTextIndexTest, TextMatchMinShouldMatch) {
    TempDir dir("nti_msm");

    std::vector<std::string> texts = {
        "apple banana cherry",    // doc 0: all three
        "apple banana",           // doc 1: two of three
        "apple",                  // doc 2: one of three
        "date elderberry",        // doc 3: none
    };

    NativeTextIndex index(dir.path(), make_tokenizer());
    index.set_merge_policy(std::make_shared<NoMergePolicy>());
    index.build(texts.size(), texts.data());

    // min_should_match=1 (OR): docs 0, 1, 2.
    auto or_result = index.match_query("apple banana cherry", 1);
    EXPECT_EQ(or_result.count(), 3u);
    EXPECT_TRUE(or_result[0]);
    EXPECT_TRUE(or_result[1]);
    EXPECT_TRUE(or_result[2]);
    EXPECT_FALSE(or_result[3]);

    // min_should_match=2: docs 0, 1.
    auto two = index.match_query("apple banana cherry", 2);
    EXPECT_EQ(two.count(), 2u);
    EXPECT_TRUE(two[0]);
    EXPECT_TRUE(two[1]);

    // min_should_match=3 (AND-like): only doc 0.
    auto all = index.match_query("apple banana cherry", 3);
    EXPECT_EQ(all.count(), 1u);
    EXPECT_TRUE(all[0]);

    // min_should_match > number of query terms: no results.
    auto none = index.match_query("apple banana cherry", 4);
    EXPECT_EQ(none.count(), 0u);
}

TEST(NativeTextIndexTest, TextMatchEmptyQuery) {
    TempDir dir("nti_empty_q");

    std::vector<std::string> texts = {"hello world"};

    NativeTextIndex index(dir.path(), make_tokenizer());
    index.set_merge_policy(std::make_shared<NoMergePolicy>());
    index.build(texts.size(), texts.data());

    // Empty query → no tokens → no matches.
    auto result = index.match_query("");
    EXPECT_EQ(result.count(), 0u);

    // Punctuation-only query → no tokens after tokenization.
    auto result2 = index.match_query("!!!");
    EXPECT_EQ(result2.count(), 0u);
}

TEST(NativeTextIndexTest, TextMatchLargeCorpus) {
    TempDir dir("nti_large");

    // 100 documents, each with a common term + unique term.
    std::vector<std::string> texts;
    texts.reserve(100);
    for (int i = 0; i < 100; i++) {
        texts.push_back("common unique" + std::to_string(i));
    }

    NativeTextIndex index(dir.path(), make_tokenizer());
    index.set_merge_policy(std::make_shared<NoMergePolicy>());
    index.build(texts.size(), texts.data());

    // "common" matches all 100.
    auto common = index.match_query("common");
    EXPECT_EQ(common.count(), 100u);

    // "unique42" matches only doc 42.
    auto unique = index.term_query("unique42");
    EXPECT_EQ(unique.count(), 1u);
    EXPECT_TRUE(unique[42]);

    // "common unique42" min_should_match=2 → only doc 42.
    auto both = index.match_query("common unique42", 2);
    EXPECT_EQ(both.count(), 1u);
    EXPECT_TRUE(both[42]);
}

// --- Merge with tokenized data ---

TEST(NativeTextIndexTest, MergeWithTokenizedData) {
    TempDir dir("nti_merge");

    NativeTextIndex index(dir.path(), make_tokenizer());
    auto policy = std::make_shared<LogMergePolicy>();
    policy->set_min_num_segments(2);
    policy->set_min_layer_size(1);
    index.set_merge_policy(policy);

    // Create 3 segments.
    index.add_text(0, "apple banana");
    index.commit();
    index.add_text(1, "banana cherry");
    index.commit();
    index.add_text(2, "apple cherry");
    // This commit should trigger merge.
    index.commit();

    auto apple = index.term_query("apple");
    EXPECT_EQ(apple.count(), 2u);
    EXPECT_TRUE(apple[0]);
    EXPECT_TRUE(apple[2]);

    auto banana = index.term_query("banana");
    EXPECT_EQ(banana.count(), 2u);
    EXPECT_TRUE(banana[0]);
    EXPECT_TRUE(banana[1]);

    auto cherry = index.term_query("cherry");
    EXPECT_EQ(cherry.count(), 2u);
    EXPECT_TRUE(cherry[1]);
    EXPECT_TRUE(cherry[2]);
}
