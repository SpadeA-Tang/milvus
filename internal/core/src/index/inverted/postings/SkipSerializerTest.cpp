#include <gtest/gtest.h>

#include <vector>

#include "index/inverted/postings/SkipSerializer.h"

using namespace milvus::index::inverted;

TEST(SkipSerializerTest, WriteAndRead) {
    // Write 3 skip entries (simulating 3 full blocks).
    SkipSerializer skip;
    skip.write_doc(127, 7);   // block 0: last_doc=127, 7 bits
    skip.write_doc(300, 10);  // block 1: last_doc=300, 10 bits
    skip.write_doc(500, 8);   // block 2: last_doc=500, 8 bits

    EXPECT_EQ(skip.size(), 3 * kSkipEntrySize);

    // Read back with SkipReader.
    // doc_freq = 3 * 128 + 5 = 389 (3 full blocks + 5 tail).
    uint32_t doc_freq = 3 * 128 + 5;
    SkipReader reader(skip.data(), skip.size(), doc_freq);

    // First block info
    EXPECT_EQ(reader.last_doc_in_block(), 127u);
    EXPECT_EQ(reader.block_info().type, BlockType::kBitPacked);
    EXPECT_EQ(reader.block_info().doc_num_bits, 7);
    EXPECT_EQ(reader.byte_offset(), 0u);

    // Advance to block 1
    reader.advance();
    EXPECT_EQ(reader.last_doc_in_block(), 300u);
    EXPECT_EQ(reader.block_info().type, BlockType::kBitPacked);
    EXPECT_EQ(reader.block_info().doc_num_bits, 10);
    EXPECT_EQ(reader.last_doc_in_previous_block(), 127u);

    // Advance to block 2
    reader.advance();
    EXPECT_EQ(reader.last_doc_in_block(), 500u);
    EXPECT_EQ(reader.block_info().type, BlockType::kBitPacked);
    EXPECT_EQ(reader.block_info().doc_num_bits, 8);
    EXPECT_EQ(reader.last_doc_in_previous_block(), 300u);

    // Advance to VInt tail
    reader.advance();
    EXPECT_EQ(reader.last_doc_in_block(), UINT32_MAX);
    EXPECT_EQ(reader.block_info().type, BlockType::kVInt);
    EXPECT_EQ(reader.block_info().num_docs, 5u);
    EXPECT_EQ(reader.last_doc_in_previous_block(), 500u);
}

TEST(SkipSerializerTest, ExactMultipleOfBlockSize) {
    // doc_freq = 128: exactly one full block, no tail.
    SkipSerializer skip;
    skip.write_doc(200, 5);

    SkipReader reader(skip.data(), skip.size(), 128);
    EXPECT_EQ(reader.last_doc_in_block(), 200u);
    EXPECT_EQ(reader.block_info().type, BlockType::kBitPacked);

    reader.advance();
    EXPECT_EQ(reader.last_doc_in_block(), UINT32_MAX);
    EXPECT_EQ(reader.block_info().type, BlockType::kVInt);
    EXPECT_EQ(reader.block_info().num_docs, 0u);
}

TEST(SkipSerializerTest, NoFullBlocks) {
    // doc_freq < 128: no skip data at all.
    SkipReader reader(nullptr, 0, 50);
    EXPECT_EQ(reader.last_doc_in_block(), UINT32_MAX);
    EXPECT_EQ(reader.block_info().type, BlockType::kVInt);
    EXPECT_EQ(reader.block_info().num_docs, 50u);
}

TEST(SkipReaderTest, Seek) {
    SkipSerializer skip;
    skip.write_doc(100, 5);
    skip.write_doc(250, 8);
    skip.write_doc(400, 6);

    uint32_t doc_freq = 3 * 128 + 10;
    SkipReader reader(skip.data(), skip.size(), doc_freq);

    // Seek to target within block 0 — no movement needed.
    EXPECT_FALSE(reader.seek(50));
    EXPECT_EQ(reader.last_doc_in_block(), 100u);

    // Seek to target in block 2.
    EXPECT_TRUE(reader.seek(300));
    EXPECT_EQ(reader.last_doc_in_block(), 400u);
    EXPECT_EQ(reader.last_doc_in_previous_block(), 250u);
}
