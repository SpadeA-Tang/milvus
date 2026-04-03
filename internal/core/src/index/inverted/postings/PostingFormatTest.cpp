#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "index/inverted/postings/PostingFormat.h"
#include "index/inverted/storage/FileIO.h"

using namespace milvus::index::inverted;

// --- VarInt ---

TEST(VarIntTest, Roundtrip) {
    std::vector<uint64_t> values = {
        0, 1, 127, 128, 255, 256, 16383, 16384, UINT32_MAX, UINT64_MAX};
    for (uint64_t val : values) {
        uint8_t buf[10];
        size_t len = encode_varuint(buf, val);
        const uint8_t* ptr = buf;
        uint64_t decoded = decode_varuint(ptr);
        EXPECT_EQ(decoded, val);
        EXPECT_EQ(static_cast<size_t>(ptr - buf), len);
    }
}

// --- Bitpacking ---

TEST(BitpackTest, PackUnpack) {
    std::vector<uint32_t> values = {3, 7, 1, 0, 5, 2, 6, 4};
    uint8_t bits = bit_width(*std::max_element(values.begin(), values.end()));
    EXPECT_EQ(bits, 3);

    std::vector<uint8_t> packed;
    bitpack(values.data(), values.size(), bits, packed);

    std::vector<uint32_t> unpacked(values.size());
    bitunpack(packed.data(), values.size(), bits, unpacked.data());

    EXPECT_EQ(unpacked, values);
}

TEST(BitpackTest, ZeroBitWidth) {
    std::vector<uint32_t> values(10, 0);
    std::vector<uint8_t> packed;
    bitpack(values.data(), values.size(), 0, packed);
    EXPECT_TRUE(packed.empty());

    std::vector<uint32_t> unpacked(10, 99);
    bitunpack(nullptr, 10, 0, unpacked.data());
    EXPECT_EQ(unpacked, values);
}

TEST(BitpackTest, FullWidth32) {
    std::vector<uint32_t> values = {0, UINT32_MAX, 12345, 0xDEADBEEF};
    std::vector<uint8_t> packed;
    bitpack(values.data(), values.size(), 32, packed);

    std::vector<uint32_t> unpacked(values.size());
    bitunpack(packed.data(), values.size(), 32, unpacked.data());
    EXPECT_EQ(unpacked, values);
}

// --- PostingsInfo ---

TEST(PostingsInfoTest, InlineRoundtrip) {
    PostingsInfo info;
    info.cardinality = 3;
    info.encoding = PostingEncoding::kInline;
    info.inline_docs = {10, 20, 30, 0, 0, 0};

    std::string path = "/tmp/test_postings_info_inline.dat";
    {
        LocalFileWriter writer(path);
        info.serialize(&writer);
        writer.flush();
    }
    {
        LocalFileReader reader(path);
        std::vector<uint8_t> buf(reader.file_size());
        reader.read(0, buf.size(), buf.data());
        const uint8_t* ptr = buf.data();
        PostingsInfo decoded = PostingsInfo::deserialize(ptr);

        EXPECT_EQ(decoded.cardinality, 3u);
        EXPECT_EQ(decoded.encoding, PostingEncoding::kInline);
        EXPECT_EQ(decoded.inline_docs[0], 10u);
        EXPECT_EQ(decoded.inline_docs[1], 20u);
        EXPECT_EQ(decoded.inline_docs[2], 30u);
    }
    std::remove(path.c_str());
}

TEST(PostingsInfoTest, BitpackingRoundtrip) {
    PostingsInfo info;
    info.cardinality = 1000;
    info.encoding = PostingEncoding::kBitpacking;
    info.file_offset = 4096;
    info.data_size = 512;

    std::string path = "/tmp/test_postings_info_bp.dat";
    {
        LocalFileWriter writer(path);
        info.serialize(&writer);
        writer.flush();
    }
    {
        LocalFileReader reader(path);
        std::vector<uint8_t> buf(reader.file_size());
        reader.read(0, buf.size(), buf.data());
        const uint8_t* ptr = buf.data();
        PostingsInfo decoded = PostingsInfo::deserialize(ptr);

        EXPECT_EQ(decoded.cardinality, 1000u);
        EXPECT_EQ(decoded.encoding, PostingEncoding::kBitpacking);
        EXPECT_EQ(decoded.file_offset, 4096u);
        EXPECT_EQ(decoded.data_size, 512u);
    }
    std::remove(path.c_str());
}

// --- BitpackingPostingCodec ---

TEST(BitpackingCodecTest, SmallPostingList) {
    std::vector<uint32_t> doc_ids = {1, 5, 10, 20, 100};

    std::string path = "/tmp/test_bp_codec_small.pst";
    {
        LocalFileWriter writer(path);
        BitpackingPostingCodec codec;
        codec.encode(doc_ids.data(), doc_ids.size(), &writer);
        writer.flush();
    }
    {
        LocalFileReader reader(path);
        BitpackingPostingCodec codec;
        std::vector<uint32_t> decoded;
        codec.decode(&reader,
                     0,
                     static_cast<uint32_t>(reader.file_size()),
                     doc_ids.size(),
                     decoded);
        EXPECT_EQ(decoded, doc_ids);
    }
    std::remove(path.c_str());
}

TEST(BitpackingCodecTest, ExactOneBlock) {
    // Exactly 128 docs.
    std::vector<uint32_t> doc_ids;
    for (uint32_t i = 0; i < 128; i++) {
        doc_ids.push_back(i * 3);
    }

    std::string path = "/tmp/test_bp_codec_128.pst";
    {
        LocalFileWriter writer(path);
        BitpackingPostingCodec codec;
        codec.encode(doc_ids.data(), doc_ids.size(), &writer);
        writer.flush();
    }
    {
        LocalFileReader reader(path);
        BitpackingPostingCodec codec;
        std::vector<uint32_t> decoded;
        codec.decode(&reader,
                     0,
                     static_cast<uint32_t>(reader.file_size()),
                     doc_ids.size(),
                     decoded);
        EXPECT_EQ(decoded, doc_ids);
    }
    std::remove(path.c_str());
}

TEST(BitpackingCodecTest, MultipleBlocks) {
    // 300 docs: 2 full blocks + remainder of 44.
    std::vector<uint32_t> doc_ids;
    for (uint32_t i = 0; i < 300; i++) {
        doc_ids.push_back(i * 7 + 1);
    }

    std::string path = "/tmp/test_bp_codec_300.pst";
    {
        LocalFileWriter writer(path);
        BitpackingPostingCodec codec;
        codec.encode(doc_ids.data(), doc_ids.size(), &writer);
        writer.flush();
    }
    {
        LocalFileReader reader(path);
        BitpackingPostingCodec codec;
        std::vector<uint32_t> decoded;
        codec.decode(&reader,
                     0,
                     static_cast<uint32_t>(reader.file_size()),
                     doc_ids.size(),
                     decoded);
        EXPECT_EQ(decoded, doc_ids);
    }
    std::remove(path.c_str());
}

TEST(BitpackingCodecTest, LargeGaps) {
    // Large gaps between doc IDs (tests wider bit widths).
    std::vector<uint32_t> doc_ids = {
        0, 100, 10000, 50000, 100000, 1000000, 2000000, 3000000};

    std::string path = "/tmp/test_bp_codec_gaps.pst";
    {
        LocalFileWriter writer(path);
        BitpackingPostingCodec codec;
        codec.encode(doc_ids.data(), doc_ids.size(), &writer);
        writer.flush();
    }
    {
        LocalFileReader reader(path);
        BitpackingPostingCodec codec;
        std::vector<uint32_t> decoded;
        codec.decode(&reader,
                     0,
                     static_cast<uint32_t>(reader.file_size()),
                     doc_ids.size(),
                     decoded);
        EXPECT_EQ(decoded, doc_ids);
    }
    std::remove(path.c_str());
}
