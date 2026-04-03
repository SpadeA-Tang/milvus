#include <gtest/gtest.h>

#include "index/inverted/storage/Allocator.h"

using namespace milvus::index::inverted;

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
