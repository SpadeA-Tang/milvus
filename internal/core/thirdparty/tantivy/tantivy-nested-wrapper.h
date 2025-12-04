#pragma once

#include <assert.h>
#include <string>
#include <vector>
#include <type_traits>
#include <fmt/format.h>

#include "common/EasyAssert.h"
#include "tantivy-binding.h"
#include "index/Utils.h"

namespace milvus::tantivy {

static constexpr uintptr_t DEFAULT_NESTED_NUM_THREADS = 1;
static constexpr uintptr_t DEFAULT_NESTED_MEMORY_BUDGET = 15 * 1024 * 1024;

/// C++ wrapper for Tantivy nested index (for Struct array fields)
struct TantivyNestedIndexWrapper {
    using IndexWriter = void*;
    using IndexReader = void*;

    NO_COPY_OR_ASSIGN(TantivyNestedIndexWrapper);

    TantivyNestedIndexWrapper() = default;

    TantivyNestedIndexWrapper(TantivyNestedIndexWrapper&& other) noexcept {
        writer_ = other.writer_;
        reader_ = other.reader_;
        path_ = other.path_;
        other.writer_ = nullptr;
        other.reader_ = nullptr;
        other.path_ = "";
    }

    TantivyNestedIndexWrapper&
    operator=(TantivyNestedIndexWrapper&& other) noexcept {
        if (this != &other) {
            free();
            writer_ = other.writer_;
            reader_ = other.reader_;
            path_ = other.path_;
            other.writer_ = nullptr;
            other.reader_ = nullptr;
            other.path_ = "";
        }
        return *this;
    }

    /// Create nested index writer for struct fields
    TantivyNestedIndexWrapper(
        const char* struct_name,
        const std::vector<std::string>& field_names,
        const std::vector<TantivyDataType>& data_types,
        const char* path,
        uintptr_t num_threads = DEFAULT_NESTED_NUM_THREADS,
        uintptr_t overall_memory_budget_in_bytes = DEFAULT_NESTED_MEMORY_BUDGET) {

        AssertInfo(field_names.size() == data_types.size(),
                   "field_names and data_types must have same size");

        std::vector<const char*> field_name_ptrs;
        field_name_ptrs.reserve(field_names.size());
        for (const auto& name : field_names) {
            field_name_ptrs.push_back(name.c_str());
        }

        char* error = tantivy_create_nested_index(
            struct_name,
            field_name_ptrs.data(),
            data_types.data(),
            field_names.size(),
            path,
            num_threads,
            overall_memory_budget_in_bytes,
            &writer_);

        AssertInfo(error == nullptr,
                   "failed to create nested index: {}",
                   error ? error : "unknown error");
        if (error) {
            tantivy_free_error(error);
        }

        path_ = std::string(path);
    }

    ~TantivyNestedIndexWrapper() {
        free();
    }

    /// Add nested documents for a single row
    /// @param row_id The parent row ID
    /// @param field_data Array of pointers to field data. Each pointer points to
    ///                   an array of values for that field (length = array_count)
    /// @param field_count Number of fields
    /// @param array_count Number of array elements (nested documents)
    void
    add_nested_documents(int64_t row_id,
                         const void* const* field_data,
                         uintptr_t field_count,
                         uintptr_t array_count) {
        AssertInfo(writer_ != nullptr, "writer is null");

        char* error = tantivy_nested_index_add_documents(
            writer_, row_id, field_data, field_count, array_count);

        AssertInfo(error == nullptr,
                   "failed to add nested documents: {}",
                   error ? error : "unknown error");
        if (error) {
            tantivy_free_error(error);
        }
    }

    /// Commit the index
    void
    commit() {
        AssertInfo(writer_ != nullptr, "writer is null");

        char* error = tantivy_nested_index_commit(writer_);

        AssertInfo(error == nullptr,
                   "failed to commit nested index: {}",
                   error ? error : "unknown error");
        if (error) {
            tantivy_free_error(error);
        }
    }

    /// Create reader from writer
    void
    create_reader(SetBitsetFn set_bitset) {
        AssertInfo(writer_ != nullptr, "writer is null");

        char* error = tantivy_nested_create_reader_from_writer(
            writer_, set_bitset, &reader_);

        AssertInfo(error == nullptr,
                   "failed to create nested index reader: {}",
                   error ? error : "unknown error");
        if (error) {
            tantivy_free_error(error);
        }
    }

    /// Search nested documents with protobuf-encoded query
    /// Results are written directly to the bitset via the set_bitset callback
    void
    search_nested(const uint8_t* query_proto_data,
                  uintptr_t query_proto_len,
                  void* bitset) {
        AssertInfo(reader_ != nullptr, "reader is null");

        char* error = tantivy_search_nested(
            reader_, query_proto_data, query_proto_len, bitset);

        AssertInfo(error == nullptr,
                   "failed to search nested index: {}",
                   error ? error : "unknown error");
        if (error) {
            tantivy_free_error(error);
        }
    }

    /// Search nested documents with protobuf-encoded query (string version)
    void
    search_nested(const std::string& query_proto, void* bitset) {
        search_nested(reinterpret_cast<const uint8_t*>(query_proto.data()),
                      query_proto.size(),
                      bitset);
    }

    IndexWriter
    get_writer() {
        return writer_;
    }

    IndexReader
    get_reader() {
        return reader_;
    }

    void
    free() {
        if (writer_ != nullptr) {
            tantivy_free_nested_index_writer(writer_);
            writer_ = nullptr;
        }

        if (reader_ != nullptr) {
            tantivy_free_nested_index_reader(reader_);
            reader_ = nullptr;
        }
    }

 private:
    IndexWriter writer_ = nullptr;
    IndexReader reader_ = nullptr;
    std::string path_;
};

}  // namespace milvus::tantivy
