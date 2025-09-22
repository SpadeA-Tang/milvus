// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <atomic>
#include "syncpoint/sync_point.h"

namespace milvus {

class SyncPointTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef MILVUS_FAILPOINT
        SyncPoint::GetInstance()->Reset();
        SyncPoint::GetInstance()->DisableProcessing();
#endif
    }

    void TearDown() override {
#ifdef MILVUS_FAILPOINT
        SyncPoint::GetInstance()->Reset();
        SyncPoint::GetInstance()->DisableProcessing();
#endif
    }
};

#ifdef MILVUS_FAILPOINT

TEST_F(SyncPointTest, BasicBlocking) {
    auto* sync_point = SyncPoint::GetInstance();

    std::atomic<int> counter{0};

    // Block at test point
    sync_point->BlockAtPoint("TestPoint");
    sync_point->EnableProcessing();

    // Start thread that will block
    std::thread t([&counter]() {
        counter = 1;
        TEST_SYNC_POINT("TestPoint");
        counter = 2;
    });

    // Give thread time to reach sync point
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check thread is blocked
    EXPECT_EQ(counter, 1);

    // Unblock the point
    sync_point->UnblockPoint("TestPoint");

    // Wait for thread to complete
    t.join();

    // Check thread completed
    EXPECT_EQ(counter, 2);
}

TEST_F(SyncPointTest, Callback) {
    auto* sync_point = SyncPoint::GetInstance();

    bool callback_executed = false;

    sync_point->SetCallBack("TestCallback", [&callback_executed](void*) {
        callback_executed = true;
    });
    sync_point->EnableProcessing();

    TEST_SYNC_POINT("TestCallback");

    EXPECT_TRUE(callback_executed);
}

TEST_F(SyncPointTest, Dependency) {
    auto* sync_point = SyncPoint::GetInstance();

    std::atomic<int> order{0};

    // Setup dependency: Point2 must wait for Point1
    sync_point->LoadDependency({
        {"Point1", "Point2"}
    });
    sync_point->EnableProcessing();

    // Thread 2 tries to pass Point2 first
    std::thread t2([&order]() {
        TEST_SYNC_POINT("Point2");
        order = 2;
    });

    // Give t2 time to reach Point2
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check t2 is blocked
    EXPECT_EQ(order, 0);

    // Thread 1 passes Point1
    std::thread t1([&order]() {
        TEST_SYNC_POINT("Point1");
        order = 1;
    });

    t1.join();
    t2.join();

    // Check order is correct
    EXPECT_EQ(order, 2);
}

TEST_F(SyncPointTest, DisableProcessing) {
    auto* sync_point = SyncPoint::GetInstance();

    std::atomic<bool> point_reached{false};

    // Block at test point
    sync_point->BlockAtPoint("BlockPoint");
    sync_point->EnableProcessing();

    // Start thread
    std::thread t([&point_reached]() {
        TEST_SYNC_POINT("BlockPoint");
        point_reached = true;
    });

    // Give thread time to block
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Disable processing should unblock
    sync_point->DisableProcessing();

    t.join();

    // Thread should have exited without reaching the point
    EXPECT_FALSE(point_reached);
}

#else

TEST_F(SyncPointTest, FailpointDisabled) {
    // When MILVUS_FAILPOINT is not defined, TEST_SYNC_POINT should be no-op
    bool executed = true;
    TEST_SYNC_POINT("TestPoint");
    EXPECT_TRUE(executed);
    std::cout << "Failpoint is disabled. Enable with -DMILVUS_FAILPOINT=ON" << std::endl;
}

#endif // MILVUS_FAILPOINT

} // namespace milvus