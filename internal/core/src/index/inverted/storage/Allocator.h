// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License

#pragma once

#include <cstddef>
#include <cstdlib>
#include <atomic>

namespace milvus::index::inverted {

class Allocator {
 public:
    virtual void*
    allocate(size_t size) = 0;
    virtual void
    deallocate(void* ptr, size_t size) = 0;
    virtual size_t
    allocated_bytes() const = 0;
    virtual ~Allocator() = default;
};

class MallocAllocator : public Allocator {
 public:
    void*
    allocate(size_t size) override {
        void* ptr = std::malloc(size);
        if (ptr) {
            allocated_.fetch_add(size, std::memory_order_relaxed);
        }
        return ptr;
    }

    void
    deallocate(void* ptr, size_t size) override {
        if (ptr) {
            allocated_.fetch_sub(size, std::memory_order_relaxed);
            std::free(ptr);
        }
    }

    size_t
    allocated_bytes() const override {
        return allocated_.load(std::memory_order_relaxed);
    }

 private:
    std::atomic<size_t> allocated_{0};
};

}  // namespace milvus::index::inverted
