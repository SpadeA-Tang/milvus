#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "Allocator.h"
#include "ArenaHashMap.h"
#include "BlockCache.h"
#include "ExpUnrolledLinkedList.h"
#include "FileIO.h"
#include "MemoryArena.h"

using namespace milvus::index::inverted;

// --- Allocator ---

TEST(AllocatorTest, MallocAllocator) {
    MallocAllocator alloc;
    EXPECT_EQ(alloc.allocated_bytes(), 0u);

    void* p1 = alloc.allocate(100);
    EXPECT_NE(p1, nullptr);
    EXPECT_EQ(alloc.allocated_bytes(), 100u);

    void* p2 = alloc.allocate(200);
    EXPECT_NE(p2, nullptr);
    EXPECT_EQ(alloc.allocated_bytes(), 300u);

    alloc.deallocate(p1, 100);
    EXPECT_EQ(alloc.allocated_bytes(), 200u);

    alloc.deallocate(p2, 200);
    EXPECT_EQ(alloc.allocated_bytes(), 0u);
}

// --- FileIO ---

TEST(FileIOTest, WriteAndRead) {
    std::string path = "/tmp/test_inverted_fileio.dat";

    // Write
    {
        LocalFileWriter writer(path);
        uint32_t val1 = 42;
        writer.write(&val1, sizeof(val1));
        EXPECT_EQ(writer.offset(), sizeof(uint32_t));

        uint32_t val2 = 99;
        writer.write(&val2, sizeof(val2));
        EXPECT_EQ(writer.offset(), 2 * sizeof(uint32_t));
        writer.flush();
    }

    // Read
    {
        LocalFileReader reader(path);
        EXPECT_EQ(reader.file_size(), 2 * sizeof(uint32_t));

        uint32_t val1 = 0;
        reader.read(0, sizeof(val1), &val1);
        EXPECT_EQ(val1, 42u);

        uint32_t val2 = 0;
        reader.read(sizeof(uint32_t), sizeof(val2), &val2);
        EXPECT_EQ(val2, 99u);
    }

    std::remove(path.c_str());
}

// --- BlockCache ---

TEST(BlockCacheTest, PassthroughAlwaysLoads) {
    PassthroughCache cache;
    int load_count = 0;

    CacheKey key{1, 0, 0};
    auto loader = [&]() -> CacheEntry {
        load_count++;
        return CacheEntry{std::make_shared<int>(42), sizeof(int)};
    };

    auto entry1 = cache.get_or_load(key, loader);
    EXPECT_EQ(load_count, 1);
    EXPECT_EQ(*std::static_pointer_cast<int>(entry1.data), 42);

    // Passthrough: same key still calls loader.
    auto entry2 = cache.get_or_load(key, loader);
    EXPECT_EQ(load_count, 2);

    EXPECT_EQ(cache.memory_usage(), 0u);
}

// --- MemoryArena ---
// Tests ported from tantivy stacker/src/memory_arena.rs

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
    // Ported from tantivy test_arena_allocate_slice.
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
    // Ported from tantivy test_arena_allocate_end_of_page.
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
    // Ported from tantivy test_store_object.
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

// --- ExpUnrolledLinkedList ---
// Tests ported from tantivy stacker/src/expull.rs

TEST(ExpUnrolledLinkedListTest, Empty) {
    MemoryArena arena;
    ExpUnrolledLinkedList stack;
    std::vector<uint8_t> buffer;
    stack.read_to_end(arena, buffer);
    EXPECT_TRUE(buffer.empty());
}

TEST(ExpUnrolledLinkedListTest, BasicWrite) {
    // Ported from tantivy test_eull1.
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
    // Ported from tantivy test_eull_first_write_extends_cap.
    MemoryArena arena;
    ExpUnrolledLinkedList stack;

    uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    stack.writer(arena).extend_from_slice(data, 9);

    std::vector<uint8_t> buffer;
    stack.read_to_end(arena, buffer);
    EXPECT_EQ(buffer, (std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

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

TEST(ExpUnrolledLinkedListTest, LongVints) {
    // Ported from tantivy test_eull_long.
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
    // Ported from tantivy test_eull_limit.
    ExpUnrolledLinkedList eull;
    for (int i = 0; i < 100; i++) {
        eull.increment_num_blocks();
    }
    // After 100 increments, block_num = 102, capped at 1<<15 = 32768.
    EXPECT_EQ(get_block_size(static_cast<uint16_t>(kFirstBlockNum + 100)),
              1 << 15);
}

TEST(ExpUnrolledLinkedListTest, Interlaced) {
    // Ported from tantivy test_eull_interlaced.
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

// --- SharedArenaHashMap ---
// Tests ported from tantivy stacker/src/shared_arena_hashmap.rs

TEST(SharedArenaHashMapTest, BasicInsertAndUpdate) {
    // Ported from tantivy test_hash_map.
    MemoryArena arena;
    SharedArenaHashMap map;

    map.mutate_or_create<uint32_t>(reinterpret_cast<const uint8_t*>("abc"),
                                   3,
                                   arena,
                                   [](std::optional<uint32_t> opt) -> uint32_t {
                                       EXPECT_FALSE(opt.has_value());
                                       return 3u;
                                   });

    map.mutate_or_create<uint32_t>(reinterpret_cast<const uint8_t*>("abcd"),
                                   4,
                                   arena,
                                   [](std::optional<uint32_t> opt) -> uint32_t {
                                       EXPECT_FALSE(opt.has_value());
                                       return 4u;
                                   });

    map.mutate_or_create<uint32_t>(reinterpret_cast<const uint8_t*>("abc"),
                                   3,
                                   arena,
                                   [](std::optional<uint32_t> opt) -> uint32_t {
                                       EXPECT_TRUE(opt.has_value());
                                       EXPECT_EQ(*opt, 3u);
                                       return 5u;
                                   });

    // Verify via iter.
    std::map<std::string, uint32_t> collected;
    map.iter(arena, [&](const uint8_t* key, size_t key_len, Addr val_addr) {
        uint32_t val = arena.read<uint32_t>(val_addr);
        collected[std::string(reinterpret_cast<const char*>(key), key_len)] =
            val;
    });
    EXPECT_EQ(collected.size(), 2u);
    EXPECT_EQ(collected["abc"], 5u);
    EXPECT_EQ(collected["abcd"], 4u);
}

TEST(SharedArenaHashMapTest, EmptyGet) {
    // Ported from tantivy test_empty_hashmap.
    MemoryArena arena;
    SharedArenaHashMap map;
    EXPECT_FALSE(
        map.get<uint32_t>(reinterpret_cast<const uint8_t*>("abc"), 3, arena)
            .has_value());
}

TEST(SharedArenaHashMapTest, ManyTerms) {
    // Ported from tantivy test_many_terms.
    MemoryArena arena;
    SharedArenaHashMap map;

    std::vector<std::string> terms;
    for (int i = 0; i < 20000; i++) {
        terms.push_back(std::to_string(i));
    }

    for (const auto& term : terms) {
        map.mutate_or_create<uint32_t>(
            reinterpret_cast<const uint8_t*>(term.data()),
            term.size(),
            arena,
            [](std::optional<uint32_t>) -> uint32_t { return 5u; });
    }

    std::vector<std::string> terms_back;
    map.iter(arena, [&](const uint8_t* key, size_t key_len, Addr) {
        terms_back.emplace_back(reinterpret_cast<const char*>(key), key_len);
    });

    std::sort(terms_back.begin(), terms_back.end());
    std::sort(terms.begin(), terms.end());
    EXPECT_EQ(terms, terms_back);
}

TEST(SharedArenaHashMapTest, ComputePreviousPowerOfTwo) {
    // Ported from tantivy test_compute_previous_power_of_two.
    EXPECT_EQ(compute_previous_power_of_two(8), 8u);
    EXPECT_EQ(compute_previous_power_of_two(9), 8u);
    EXPECT_EQ(compute_previous_power_of_two(7), 4u);
    EXPECT_EQ(compute_previous_power_of_two(UINT64_MAX), size_t(1) << 63);
}

// --- ArenaHashMap (owned arena wrapper) ---

TEST(ArenaHashMapTest, BasicUsage) {
    // Ported from tantivy arena_hashmap test_hash_map.
    ArenaHashMap map;

    map.mutate_or_create<uint32_t>(reinterpret_cast<const uint8_t*>("abc"),
                                   3,
                                   [](std::optional<uint32_t> opt) -> uint32_t {
                                       EXPECT_FALSE(opt.has_value());
                                       return 3u;
                                   });

    map.mutate_or_create<uint32_t>(reinterpret_cast<const uint8_t*>("abcd"),
                                   4,
                                   [](std::optional<uint32_t> opt) -> uint32_t {
                                       EXPECT_FALSE(opt.has_value());
                                       return 4u;
                                   });

    map.mutate_or_create<uint32_t>(reinterpret_cast<const uint8_t*>("abc"),
                                   3,
                                   [](std::optional<uint32_t> opt) -> uint32_t {
                                       EXPECT_TRUE(opt.has_value());
                                       EXPECT_EQ(*opt, 3u);
                                       return 5u;
                                   });

    EXPECT_EQ(map.len(), 2u);

    // Verify via get.
    auto abc_val =
        map.get<uint32_t>(reinterpret_cast<const uint8_t*>("abc"), 3);
    EXPECT_TRUE(abc_val.has_value());
    EXPECT_EQ(*abc_val, 5u);

    auto abcd_val =
        map.get<uint32_t>(reinterpret_cast<const uint8_t*>("abcd"), 4);
    EXPECT_TRUE(abcd_val.has_value());
    EXPECT_EQ(*abcd_val, 4u);

    auto missing =
        map.get<uint32_t>(reinterpret_cast<const uint8_t*>("xyz"), 3);
    EXPECT_FALSE(missing.has_value());
}
