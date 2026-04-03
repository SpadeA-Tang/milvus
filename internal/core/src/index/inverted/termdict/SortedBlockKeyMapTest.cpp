#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "index/inverted/postings/PostingFormat.h"
#include "index/inverted/storage/FileIO.h"
#include "index/inverted/termdict/SortedBlockKeyMap.h"

using namespace milvus::index::inverted;

TEST(SortedBlockKeyMapTest, BasicLookup) {
    std::string idx_path = "/tmp/test_keymap.idx";
    std::string dct_path = "/tmp/test_keymap.dct";

    // Build 10 entries with inline postings.
    struct TestEntry {
        std::string key;
        PostingsInfo info;
    };
    std::vector<TestEntry> entries;
    for (int i = 0; i < 10; i++) {
        PostingsInfo info;
        info.cardinality = 1;
        info.encoding = PostingEncoding::kInline;
        info.inline_docs[0] = static_cast<uint32_t>(i * 100);

        std::string key = "term_" + std::string(1, 'a' + i);
        entries.push_back({key, info});
    }
    // Entries must be sorted.
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.key < b.key;
    });

    // Write via streaming interface
    {
        LocalFileWriter idx_writer(idx_path);
        LocalFileWriter dct_writer(dct_path);
        SortedBlockKeyMapWriter writer(&idx_writer, &dct_writer);
        for (const auto& e : entries) {
            writer.insert_key(e.key);
            writer.insert_value(e.info);
        }
        writer.finish();
        idx_writer.flush();
        dct_writer.flush();
    }

    // Read + lookup
    {
        LocalFileReader idx_reader(idx_path);
        LocalFileReader dct_reader(dct_path);
        SortedBlockKeyMapReader reader(&idx_reader, &dct_reader);

        // Lookup existing keys.
        for (const auto& e : entries) {
            auto result = reader.lookup(
                reinterpret_cast<const uint8_t*>(e.key.data()), e.key.size());
            ASSERT_TRUE(result.has_value()) << "key=" << e.key;
            EXPECT_EQ(result->cardinality, e.info.cardinality);
            EXPECT_EQ(result->inline_docs[0], e.info.inline_docs[0]);
        }

        // Lookup missing key.
        std::string missing = "zzz_not_found";
        auto result = reader.lookup(
            reinterpret_cast<const uint8_t*>(missing.data()), missing.size());
        EXPECT_FALSE(result.has_value());

        // Lookup key smaller than all entries.
        std::string too_small = "aaa";
        result =
            reader.lookup(reinterpret_cast<const uint8_t*>(too_small.data()),
                          too_small.size());
        EXPECT_FALSE(result.has_value());
    }

    std::remove(idx_path.c_str());
    std::remove(dct_path.c_str());
}

TEST(SortedBlockKeyMapTest, MultipleBlocks) {
    std::string idx_path = "/tmp/test_keymap_multi.idx";
    std::string dct_path = "/tmp/test_keymap_multi.dct";

    // Build 300 entries spanning multiple dictionary blocks.
    struct TestEntry {
        std::string key;
        PostingsInfo info;
    };
    std::vector<TestEntry> entries;
    for (int i = 0; i < 300; i++) {
        PostingsInfo info;
        info.cardinality = 2;
        info.encoding = PostingEncoding::kInline;
        info.inline_docs[0] = static_cast<uint32_t>(i);
        info.inline_docs[1] = static_cast<uint32_t>(i + 1000);

        char buf[16];
        snprintf(buf, sizeof(buf), "key_%05d", i);
        entries.push_back({std::string(buf), info});
    }

    // Write via streaming interface
    {
        LocalFileWriter idx_writer(idx_path);
        LocalFileWriter dct_writer(dct_path);
        SortedBlockKeyMapWriter writer(&idx_writer, &dct_writer);
        for (const auto& e : entries) {
            writer.insert_key(e.key);
            writer.insert_value(e.info);
        }
        writer.finish();
        idx_writer.flush();
        dct_writer.flush();
    }

    // Read + verify sparse index + lookup
    {
        LocalFileReader idx_reader(idx_path);
        LocalFileReader dct_reader(dct_path);
        SortedBlockKeyMapReader reader(&idx_reader, &dct_reader);

        // 300 / 128 = 2 full + 1 partial = 3 blocks.
        EXPECT_EQ(reader.sparse_index().entries.size(), 3u);

        // Sample lookups across blocks.
        for (int i : {0, 1, 50, 127, 128, 129, 200, 255, 256, 299}) {
            char buf[16];
            snprintf(buf, sizeof(buf), "key_%05d", i);
            auto result = reader.lookup(reinterpret_cast<const uint8_t*>(buf),
                                        std::strlen(buf));
            ASSERT_TRUE(result.has_value()) << "i=" << i;
            EXPECT_EQ(result->cardinality, 2u);
            EXPECT_EQ(result->inline_docs[0], static_cast<uint32_t>(i));
            EXPECT_EQ(result->inline_docs[1], static_cast<uint32_t>(i + 1000));
        }

        // Missing key between blocks.
        std::string gap = "key_00050_x";
        auto result = reader.lookup(
            reinterpret_cast<const uint8_t*>(gap.data()), gap.size());
        EXPECT_FALSE(result.has_value());
    }

    std::remove(idx_path.c_str());
    std::remove(dct_path.c_str());
}

// --- End-to-end: KeyMap + PostingList codec ---

TEST(EndToEndTest, WriteAndReadPostings) {
    std::string idx_path = "/tmp/test_e2e.idx";
    std::string dct_path = "/tmp/test_e2e.dct";
    std::string pst_path = "/tmp/test_e2e.pst";

    // Simulate building an index with mixed inline and bitpacked postings.
    struct TermData {
        std::string term;
        std::vector<uint32_t> doc_ids;
    };

    std::vector<TermData> terms = {
        {"apple", {1, 5, 9}},                      // inline (3 docs)
        {"banana", {2, 4, 6, 8, 10, 12, 14}},      // bitpacking (7 docs)
        {"cherry", {100}},                         // inline (1 doc)
        {"date", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}},  // bitpacking (10 docs)
    };

    BitpackingPostingCodec codec;

    struct TermEntry {
        std::string term;
        PostingsInfo info;
    };
    std::vector<TermEntry> entries;

    // Write .pst and build entries.
    {
        LocalFileWriter pst_writer(pst_path);
        for (auto& t : terms) {
            PostingsInfo info;
            info.cardinality = static_cast<uint32_t>(t.doc_ids.size());

            if (t.doc_ids.size() <= PostingsInfo::kInlineMax) {
                info.encoding = PostingEncoding::kInline;
                for (size_t i = 0; i < t.doc_ids.size(); i++) {
                    info.inline_docs[i] = t.doc_ids[i];
                }
            } else {
                info.encoding = PostingEncoding::kBitpacking;
                info.file_offset = pst_writer.offset();
                codec.encode(t.doc_ids.data(), t.doc_ids.size(), &pst_writer);
                info.data_size = static_cast<uint32_t>(pst_writer.offset() -
                                                       info.file_offset);
            }
            entries.push_back({t.term, info});
        }
        pst_writer.flush();
    }

    // Write .idx + .dct via streaming interface
    {
        LocalFileWriter idx_writer(idx_path);
        LocalFileWriter dct_writer(dct_path);
        SortedBlockKeyMapWriter writer(&idx_writer, &dct_writer);
        for (const auto& e : entries) {
            writer.insert_key(e.term);
            writer.insert_value(e.info);
        }
        writer.finish();
        idx_writer.flush();
        dct_writer.flush();
    }

    // Read and verify.
    {
        LocalFileReader idx_reader(idx_path);
        LocalFileReader dct_reader(dct_path);
        LocalFileReader pst_reader(pst_path);
        SortedBlockKeyMapReader keymap(&idx_reader, &dct_reader);

        for (const auto& t : terms) {
            auto info = keymap.lookup(
                reinterpret_cast<const uint8_t*>(t.term.data()), t.term.size());
            ASSERT_TRUE(info.has_value()) << "term=" << t.term;
            EXPECT_EQ(info->cardinality,
                      static_cast<uint32_t>(t.doc_ids.size()));

            std::vector<uint32_t> decoded;
            if (info->encoding == PostingEncoding::kInline) {
                for (uint32_t i = 0; i < info->cardinality; i++) {
                    decoded.push_back(info->inline_docs[i]);
                }
            } else {
                codec.decode(&pst_reader,
                             info->file_offset,
                             info->data_size,
                             info->cardinality,
                             decoded);
            }
            EXPECT_EQ(decoded, t.doc_ids) << "term=" << t.term;
        }
    }

    std::remove(idx_path.c_str());
    std::remove(dct_path.c_str());
    std::remove(pst_path.c_str());
}
