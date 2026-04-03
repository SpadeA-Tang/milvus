#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "index/inverted/indexing/ArenaHashMap.h"

using namespace milvus::index::inverted;

// --- SharedArenaHashMap ---

TEST(SharedArenaHashMapTest, BasicInsertAndUpdate) {
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
    MemoryArena arena;
    SharedArenaHashMap map;
    EXPECT_FALSE(
        map.get<uint32_t>(reinterpret_cast<const uint8_t*>("abc"), 3, arena)
            .has_value());
}

TEST(SharedArenaHashMapTest, ManyTerms) {
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
    EXPECT_EQ(compute_previous_power_of_two(8), 8u);
    EXPECT_EQ(compute_previous_power_of_two(9), 8u);
    EXPECT_EQ(compute_previous_power_of_two(7), 4u);
    EXPECT_EQ(compute_previous_power_of_two(UINT64_MAX), size_t(1) << 63);
}

// --- ArenaHashMap (owned arena wrapper) ---

TEST(ArenaHashMapTest, BasicUsage) {
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
