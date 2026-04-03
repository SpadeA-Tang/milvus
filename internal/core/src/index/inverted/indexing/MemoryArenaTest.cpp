#include <gtest/gtest.h>

#include <cstring>

#include "index/inverted/indexing/MemoryArena.h"

using namespace milvus::index::inverted;

TEST(MemoryArenaTest, AllocateAndReadWrite) {
    MemoryArena arena;
    EXPECT_EQ(arena.num_pages(), 1u);

    Addr addr = arena.allocate_space(sizeof(uint32_t));
    arena.write_at<uint32_t>(addr, 12345u);
    EXPECT_EQ(arena.read<uint32_t>(addr), 12345u);

    arena.write_at<uint32_t>(addr, 99999u);
    EXPECT_EQ(arena.read<uint32_t>(addr), 99999u);
}

TEST(MemoryArenaTest, AllocateSlice) {
    MemoryArena arena;
    const uint8_t a[] = {'h', 'e', 'l', 'l', 'o'};
    const uint8_t b[] = {'h',
                         'a',
                         'p',
                         'p',
                         'y',
                         ' ',
                         't',
                         'a',
                         'x',
                         ' ',
                         'p',
                         'a',
                         'y',
                         'e',
                         'r'};

    Addr addr_a = arena.allocate_space(sizeof(a));
    std::memcpy(arena.slice_mut(addr_a, sizeof(a)), a, sizeof(a));

    Addr addr_b = arena.allocate_space(sizeof(b));
    std::memcpy(arena.slice_mut(addr_b, sizeof(b)), b, sizeof(b));

    EXPECT_EQ(std::memcmp(arena.slice(addr_a, sizeof(a)), a, sizeof(a)), 0);
    EXPECT_EQ(std::memcmp(arena.slice(addr_b, sizeof(b)), b, sizeof(b)), 0);
}

TEST(MemoryArenaTest, CrossPageAllocation) {
    MemoryArena arena;

    size_t chunk = 64 * 1024;  // 64 KB chunks
    while (arena.num_pages() == 1) {
        arena.allocate_space(chunk);
    }
    EXPECT_EQ(arena.num_pages(), 2u);

    // Can still allocate and use data on page 2.
    Addr addr = arena.allocate_space(sizeof(uint64_t));
    arena.write_at<uint64_t>(addr, 0xDEADBEEF);
    EXPECT_EQ(addr.page_id(), 1u);
    EXPECT_EQ(arena.read<uint64_t>(addr), 0xDEADBEEF);
}

TEST(MemoryArenaTest, StoreObject) {
    struct MyTest {
        uint64_t a;
        uint8_t b;
        uint32_t c;
    };

    MemoryArena arena;
    MyTest a{143, 21, 32};
    MyTest b{113, 221, 12};

    Addr addr_a = arena.allocate_space(sizeof(MyTest));
    arena.write_at(addr_a, a);

    Addr addr_b = arena.allocate_space(sizeof(MyTest));
    arena.write_at(addr_b, b);

    auto ra = arena.read<MyTest>(addr_a);
    EXPECT_EQ(ra.a, 143u);
    EXPECT_EQ(ra.b, 21u);
    EXPECT_EQ(ra.c, 32u);

    auto rb = arena.read<MyTest>(addr_b);
    EXPECT_EQ(rb.a, 113u);
    EXPECT_EQ(rb.b, 221u);
    EXPECT_EQ(rb.c, 12u);
}
