#include <gtest/gtest.h>

#include "index/inverted/storage/BlockCache.h"

using namespace milvus::index::inverted;

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
