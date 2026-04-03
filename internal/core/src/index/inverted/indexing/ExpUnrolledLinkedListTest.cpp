#include <gtest/gtest.h>

#include <vector>

#include "index/inverted/indexing/ExpUnrolledLinkedList.h"
#include "index/inverted/indexing/MemoryArena.h"

using namespace milvus::index::inverted;

namespace {

static uint32_t
read_u32_vint(const uint8_t*& ptr, const uint8_t* end) {
    uint32_t val = 0;
    uint32_t shift = 0;
    while (ptr < end) {
        uint8_t byte = *ptr++;
        val |= static_cast<uint32_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) {
            return val;
        }
        shift += 7;
    }
    return val;
}

}  // namespace

TEST(ExpUnrolledLinkedListTest, Empty) {
    MemoryArena arena;
    ExpUnrolledLinkedList stack;
    std::vector<uint8_t> buffer;
    stack.read_to_end(arena, buffer);
    EXPECT_TRUE(buffer.empty());
}

TEST(ExpUnrolledLinkedListTest, BasicWrite) {
    MemoryArena arena;
    ExpUnrolledLinkedList stack;

    uint8_t d1[] = {1};
    stack.writer(arena).extend_from_slice(d1, 1);
    uint8_t d2[] = {2};
    stack.writer(arena).extend_from_slice(d2, 1);
    uint8_t d3[] = {3, 4};
    stack.writer(arena).extend_from_slice(d3, 2);
    uint8_t d4[] = {5};
    stack.writer(arena).extend_from_slice(d4, 1);

    std::vector<uint8_t> buffer;
    stack.read_to_end(arena, buffer);
    EXPECT_EQ(buffer, (std::vector<uint8_t>{1, 2, 3, 4, 5}));
}

TEST(ExpUnrolledLinkedListTest, FirstWriteExtendsCap) {
    MemoryArena arena;
    ExpUnrolledLinkedList stack;

    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    stack.writer(arena).extend_from_slice(data, 9);

    std::vector<uint8_t> buffer;
    stack.read_to_end(arena, buffer);
    EXPECT_EQ(buffer, (std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(ExpUnrolledLinkedListTest, LongVints) {
    MemoryArena arena;
    ExpUnrolledLinkedList eull;

    std::vector<uint32_t> data;
    for (uint32_t i = 0; i < 100; i++) {
        data.push_back(i);
    }
    for (auto val : data) {
        eull.writer(arena).write_u32_vint(val);
    }

    std::vector<uint8_t> buffer;
    eull.read_to_end(arena, buffer);

    std::vector<uint32_t> result;
    const uint8_t* ptr = buffer.data();
    const uint8_t* end = ptr + buffer.size();
    while (ptr < end) {
        result.push_back(read_u32_vint(ptr, end));
    }
    EXPECT_EQ(result, data);
}

TEST(ExpUnrolledLinkedListTest, BlockSizeLimit) {
    ExpUnrolledLinkedList eull;
    for (int i = 0; i < 100; i++) {
        eull.increment_num_blocks();
    }
    // After 100 increments, block_num = 102, capped at 1<<15 = 32768.
    EXPECT_EQ(get_block_size(static_cast<uint16_t>(kFirstBlockNum + 100)),
              1 << 15);
}

TEST(ExpUnrolledLinkedListTest, Interlaced) {
    MemoryArena arena;
    ExpUnrolledLinkedList stack1;
    ExpUnrolledLinkedList stack2;

    for (uint32_t i = 0; i < 9; i++) {
        stack1.writer(arena).write_u32_vint(i);
        if (i % 2 == 0) {
            stack2.writer(arena).write_u32_vint(i);
        }
    }

    std::vector<uint8_t> buf1, buf2;
    stack1.read_to_end(arena, buf1);
    stack2.read_to_end(arena, buf2);

    // Decode stack1: expect 0..9
    std::vector<uint32_t> res1;
    const uint8_t* p1 = buf1.data();
    const uint8_t* e1 = p1 + buf1.size();
    while (p1 < e1) {
        res1.push_back(read_u32_vint(p1, e1));
    }
    std::vector<uint32_t> expected1 = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ(res1, expected1);

    // Decode stack2: expect 0, 2, 4, 6, 8
    std::vector<uint32_t> res2;
    const uint8_t* p2 = buf2.data();
    const uint8_t* e2 = p2 + buf2.size();
    while (p2 < e2) {
        res2.push_back(read_u32_vint(p2, e2));
    }
    std::vector<uint32_t> expected2 = {0, 2, 4, 6, 8};
    EXPECT_EQ(res2, expected2);
}
