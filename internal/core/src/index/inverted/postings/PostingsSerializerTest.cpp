#include <gtest/gtest.h>

#include <cstdio>
#include <string>
#include <vector>

#include "index/inverted/postings/PostingsSerializer.h"
#include "index/inverted/storage/FileIO.h"

using namespace milvus::index::inverted;

TEST(PostingsSerializerTest, SmallPostingList) {
    // < 128 docs: no skip data, VInt only.
    std::string path = "/tmp/test_ps_small.pst";
    std::vector<uint32_t> doc_ids = {1, 5, 10, 20, 100};

    {
        LocalFileWriter writer(path);
        PostingsSerializer ps(&writer);
        ps.clear();
        for (uint32_t id : doc_ids) {
            ps.write_doc(id);
        }
        ps.close_term(static_cast<uint32_t>(doc_ids.size()));
        writer.flush();
    }
    {
        LocalFileReader reader(path);
        std::vector<uint32_t> decoded;
        PostingsDecoder::decode(&reader,
                                0,
                                static_cast<uint32_t>(reader.file_size()),
                                static_cast<uint32_t>(doc_ids.size()),
                                decoded);
        EXPECT_EQ(decoded, doc_ids);
    }
    std::remove(path.c_str());
}

TEST(PostingsSerializerTest, ExactOneBlock) {
    // Exactly 128 docs: one full block + skip data, no tail.
    std::string path = "/tmp/test_ps_128.pst";
    std::vector<uint32_t> doc_ids;
    for (uint32_t i = 0; i < 128; i++) {
        doc_ids.push_back(i * 3);
    }

    {
        LocalFileWriter writer(path);
        PostingsSerializer ps(&writer);
        ps.clear();
        for (uint32_t id : doc_ids) {
            ps.write_doc(id);
        }
        ps.close_term(static_cast<uint32_t>(doc_ids.size()));
        writer.flush();
    }
    {
        LocalFileReader reader(path);
        std::vector<uint32_t> decoded;
        PostingsDecoder::decode(&reader,
                                0,
                                static_cast<uint32_t>(reader.file_size()),
                                static_cast<uint32_t>(doc_ids.size()),
                                decoded);
        EXPECT_EQ(decoded, doc_ids);
    }
    std::remove(path.c_str());
}

TEST(PostingsSerializerTest, MultipleBlocks) {
    // 300 docs: 2 full blocks + 44 tail.
    std::string path = "/tmp/test_ps_300.pst";
    std::vector<uint32_t> doc_ids;
    for (uint32_t i = 0; i < 300; i++) {
        doc_ids.push_back(i * 7 + 1);
    }

    {
        LocalFileWriter writer(path);
        PostingsSerializer ps(&writer);
        ps.clear();
        for (uint32_t id : doc_ids) {
            ps.write_doc(id);
        }
        ps.close_term(static_cast<uint32_t>(doc_ids.size()));
        writer.flush();
    }
    {
        LocalFileReader reader(path);
        std::vector<uint32_t> decoded;
        PostingsDecoder::decode(&reader,
                                0,
                                static_cast<uint32_t>(reader.file_size()),
                                static_cast<uint32_t>(doc_ids.size()),
                                decoded);
        EXPECT_EQ(decoded, doc_ids);
    }
    std::remove(path.c_str());
}

TEST(PostingsSerializerTest, LargeGaps) {
    // Large gaps between doc IDs.
    std::string path = "/tmp/test_ps_gaps.pst";
    std::vector<uint32_t> doc_ids;
    for (uint32_t i = 0; i < 200; i++) {
        doc_ids.push_back(i * 10000);
    }

    {
        LocalFileWriter writer(path);
        PostingsSerializer ps(&writer);
        ps.clear();
        for (uint32_t id : doc_ids) {
            ps.write_doc(id);
        }
        ps.close_term(static_cast<uint32_t>(doc_ids.size()));
        writer.flush();
    }
    {
        LocalFileReader reader(path);
        std::vector<uint32_t> decoded;
        PostingsDecoder::decode(&reader,
                                0,
                                static_cast<uint32_t>(reader.file_size()),
                                static_cast<uint32_t>(doc_ids.size()),
                                decoded);
        EXPECT_EQ(decoded, doc_ids);
    }
    std::remove(path.c_str());
}

TEST(PostingsSerializerTest, MultipleTerms) {
    // Verify that clear() properly resets state between terms.
    std::string path = "/tmp/test_ps_multi_term.pst";

    struct TermData {
        std::vector<uint32_t> doc_ids;
        uint64_t offset;
        uint32_t len;
    };

    std::vector<TermData> terms;
    // Term 1: small (no skip)
    terms.push_back({{3, 7, 15}, 0, 0});
    // Term 2: large (with skip)
    {
        std::vector<uint32_t> ids;
        for (uint32_t i = 0; i < 200; i++) ids.push_back(i * 5);
        terms.push_back({ids, 0, 0});
    }
    // Term 3: exact block
    {
        std::vector<uint32_t> ids;
        for (uint32_t i = 0; i < 128; i++) ids.push_back(i * 2 + 1);
        terms.push_back({ids, 0, 0});
    }

    {
        LocalFileWriter writer(path);
        PostingsSerializer ps(&writer);
        for (auto& t : terms) {
            ps.clear();
            t.offset = ps.written_bytes();
            for (uint32_t id : t.doc_ids) {
                ps.write_doc(id);
            }
            ps.close_term(static_cast<uint32_t>(t.doc_ids.size()));
            t.len = static_cast<uint32_t>(ps.written_bytes() - t.offset);
        }
        writer.flush();
    }
    {
        LocalFileReader reader(path);
        for (const auto& t : terms) {
            std::vector<uint32_t> decoded;
            PostingsDecoder::decode(&reader,
                                    t.offset,
                                    t.len,
                                    static_cast<uint32_t>(t.doc_ids.size()),
                                    decoded);
            EXPECT_EQ(decoded, t.doc_ids);
        }
    }
    std::remove(path.c_str());
}
