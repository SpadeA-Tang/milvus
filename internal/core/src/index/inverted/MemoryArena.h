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

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace milvus::index::inverted {

// 32-bit address: 12 bits page_id + 20 bits local offset.
// Ported from tantivy stacker/src/memory_arena.rs
class Addr {
 public:
    static constexpr uint32_t kPageBits = 20;
    static constexpr uint32_t kPageSize = 1u << kPageBits;  // 1 MB
    static constexpr uint32_t kPageMask = kPageSize - 1;
    static constexpr uint32_t kMaxPages = 1u << (32 - kPageBits);  // 4096
    static constexpr uint32_t kNull = UINT32_MAX;

    Addr() : value_(kNull) {
    }
    explicit Addr(uint32_t value) : value_(value) {
    }
    Addr(uint32_t page_id, uint32_t local_offset)
        : value_((page_id << kPageBits) | local_offset) {
    }

    static Addr
    null_pointer() {
        return Addr();
    }

    Addr
    offset(uint32_t off) const {
        return Addr(value_ + off);
    }

    uint32_t
    page_id() const {
        return value_ >> kPageBits;
    }

    uint32_t
    local_offset() const {
        return value_ & kPageMask;
    }

    uint32_t
    raw() const {
        return value_;
    }

    bool
    is_null() const {
        return value_ == kNull;
    }

 private:
    uint32_t value_;
};

// 32-bit memory arena for types that are trivially copyable.
// Ported from tantivy stacker/src/memory_arena.rs
//
// - Allocations are very cheap.
// - Consecutive allocations have great locality.
// - Addresses (Addr) are 32 bits, max capacity 4GB.
// - Objects are stored unaligned (read/write via memcpy).
class MemoryArena {
 public:
    MemoryArena() {
        add_page();
    }

    // Upper-bound estimate of resident memory in bytes.
    size_t
    mem_usage() const {
        return pages_.size() * Addr::kPageSize;
    }

    // Total bytes allocated (approximation: intermediate pages counted as full).
    size_t
    len() const {
        if (pages_.empty()) {
            return 0;
        }
        return (pages_.size() - 1) * Addr::kPageSize + current_len_;
    }

    bool
    is_empty() const {
        return len() == 0;
    }

    // Write a trivially-copyable value at the given address (unaligned).
    template <typename T>
    void
    write_at(Addr addr, const T& val) {
        std::memcpy(slice_mut(addr, sizeof(T)), &val, sizeof(T));
    }

    // Read a trivially-copyable value from the given address (unaligned).
    template <typename T>
    T
    read(Addr addr) const {
        T val;
        std::memcpy(&val, slice(addr, sizeof(T)), sizeof(T));
        return val;
    }

    // Get const pointer to `len` bytes starting at `addr`.
    const uint8_t*
    slice(Addr addr, size_t len) const {
        return page_data(addr.page_id()) + addr.local_offset();
    }

    // Get const pointer from `addr` to end of its page.
    const uint8_t*
    slice_from(Addr addr) const {
        return page_data(addr.page_id()) + addr.local_offset();
    }

    // Get mutable pointer to `len` bytes starting at `addr`.
    uint8_t*
    slice_mut(Addr addr, size_t len) {
        return page_data_mut(addr.page_id()) + addr.local_offset();
    }

    // Get mutable pointer from `addr` to end of its page.
    uint8_t*
    slice_from_mut(Addr addr) {
        return page_data_mut(addr.page_id()) + addr.local_offset();
    }

    // Allocate `size` bytes and return the address.
    Addr
    allocate_space(size_t size) {
        uint32_t page_id = static_cast<uint32_t>(pages_.size() - 1);
        if (current_len_ + size <= Addr::kPageSize) {
            Addr addr(page_id, current_len_);
            current_len_ += static_cast<uint32_t>(size);
            return addr;
        }
        return add_page(size);
    }

    size_t
    num_pages() const {
        return pages_.size();
    }

 private:
    Addr
    add_page(size_t initial_len = 0) {
        if (pages_.size() >= Addr::kMaxPages) {
            throw std::runtime_error("MemoryArena: exceeded max pages (4GB)");
        }
        pages_.push_back(
            std::unique_ptr<uint8_t[]>(new uint8_t[Addr::kPageSize]()));
        uint32_t new_page_id = static_cast<uint32_t>(pages_.size() - 1);
        current_len_ = static_cast<uint32_t>(initial_len);
        return Addr(new_page_id, 0);
    }

    const uint8_t*
    page_data(uint32_t page_id) const {
        return pages_[page_id].get();
    }

    uint8_t*
    page_data_mut(uint32_t page_id) {
        return pages_[page_id].get();
    }

    std::vector<std::unique_ptr<uint8_t[]>> pages_;
    uint32_t current_len_ = 0;
};

}  // namespace milvus::index::inverted
