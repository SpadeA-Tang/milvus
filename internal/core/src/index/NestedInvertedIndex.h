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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/FieldDataInterface.h"
#include "common/Types.h"
#include "index/Index.h"
#include "storage/FileManager.h"
#include "storage/DiskFileManagerImpl.h"
#include "storage/MemFileManagerImpl.h"
#include "tantivy-binding.h"
#include "tantivy-nested-wrapper.h"
#include "index/Utils.h"

namespace milvus::index {

constexpr const char* NESTED_INDEX_TYPE = "NESTED_INVERTED";

/// NestedInvertedIndex builds a Tantivy index for struct array fields.
/// It allows nested queries like: MATCH_ALL(struct_array, $[intField] == 1 && $[strField] == "aaa")
class NestedInvertedIndex : public IndexBase {
 public:
    using MemFileManager = storage::MemFileManagerImpl;
    using MemFileManagerPtr = std::shared_ptr<MemFileManager>;
    using DiskFileManager = storage::DiskFileManagerImpl;
    using DiskFileManagerPtr = std::shared_ptr<DiskFileManager>;

    /// Constructor for building index
    /// @param struct_name Name of the struct field
    /// @param field_names Names of fields within the struct
    /// @param field_types Data types of each field
    /// @param ctx File manager context
    NestedInvertedIndex(const std::string& struct_name,
                        const std::vector<std::string>& field_names,
                        const std::vector<DataType>& field_types,
                        const storage::FileManagerContext& ctx);

    /// Constructor for loading index
    explicit NestedInvertedIndex(const storage::FileManagerContext& ctx);

    ~NestedInvertedIndex() override;

    // IndexBase interface
    BinarySet
    Serialize(const Config& config) override;

    void
    Load(const BinarySet& binary_set, const Config& config = {}) override {
        ThrowInfo(ErrorCode::NotImplemented, "load v1 should be deprecated");
    }

    void
    Load(milvus::tracer::TraceContext ctx, const Config& config = {}) override;

    void
    BuildWithRawDataForUT(size_t n,
                          const void* values,
                          const Config& config = {}) override {
        ThrowInfo(ErrorCode::NotImplemented,
                  "BuildWithRawDataForUT not implemented for nested index");
    }

    void
    BuildWithDataset(const DatasetPtr& dataset,
                     const Config& config = {}) override {
        ThrowInfo(ErrorCode::NotImplemented,
                  "BuildWithDataset should be deprecated");
    }

    void
    Build(const Config& config = {}) override;

    int64_t
    Count() override {
        return count_;
    }

    IndexStatsPtr
    Upload(const Config& config = {}) override;

    const bool
    HasRawData() const override {
        return false;
    }

    bool
    IsMmapSupported() const override {
        return false;  // TODO: Add mmap support
    }

    /// Add nested documents for a single row
    /// @param row_id The parent row ID
    /// @param field_data Array of pointers to field data arrays
    /// @param array_count Number of array elements (nested documents)
    void
    AddNestedDocuments(int64_t row_id,
                       const std::vector<const void*>& field_data,
                       size_t array_count);

    /// Execute nested query and populate bitset with matching row IDs
    /// @param query_proto_data Serialized protobuf query (plan.Expr)
    /// @param query_proto_len Length of query bytes
    /// @param bitset Output bitset to receive results
    void
    SearchNested(const uint8_t* query_proto_data,
                 size_t query_proto_len,
                 void* bitset);

    /// Commit changes to index
    void
    Commit();

    /// Create reader for searching
    void
    CreateReader();

    /// Get struct name
    const std::string&
    GetStructName() const {
        return struct_name_;
    }

    /// Get field names
    const std::vector<std::string>&
    GetFieldNames() const {
        return field_names_;
    }

 private:
    void
    InitForBuild();

    static TantivyDataType
    ConvertToTantivyType(DataType milvus_type);

 private:
    std::string struct_name_;
    std::vector<std::string> field_names_;
    std::vector<DataType> field_types_;
    std::vector<TantivyDataType> tantivy_types_;

    std::unique_ptr<milvus::tantivy::TantivyNestedIndexWrapper> wrapper_;
    std::string path_;
    int64_t count_ = 0;

    MemFileManagerPtr mem_file_manager_;
    DiskFileManagerPtr disk_file_manager_;
};

using NestedInvertedIndexPtr = std::unique_ptr<NestedInvertedIndex>;

}  // namespace milvus::index
