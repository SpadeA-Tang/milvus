#include "index/NgramInvertedIndex.h"
#include "exec/expression/Expr.h"

namespace milvus::index {
constexpr const char* TMP_NGRAM_INVERTED_LOG_PREFIX =
    "/tmp/milvus/ngram-inverted-index-log/";

NgramInvertedIndex::NgramInvertedIndex(const storage::FileManagerContext& ctx,
                                       bool for_loading_index,
                                       uintptr_t min_gram,
                                       uintptr_t max_gram)
    : min_gram_(min_gram), max_gram_(max_gram) {
    schema_ = ctx.fieldDataMeta.field_schema;
    field_id_ = ctx.fieldDataMeta.field_id;
    mem_file_manager_ = std::make_shared<MemFileManager>(ctx);
    disk_file_manager_ = std::make_shared<DiskFileManager>(ctx);

    if (for_loading_index) {
        path_ = disk_file_manager_->GetLocalNgramIndexPrefix();
    } else {
        auto prefix = disk_file_manager_->GetNgramIndexIdentifier();
        path_ = std::string(TMP_NGRAM_INVERTED_LOG_PREFIX) + prefix;
        boost::filesystem::create_directories(path_);
        d_type_ = TantivyDataType::Keyword;
        std::string field_name =
            std::to_string(disk_file_manager_->GetFieldDataMeta().field_id);
        wrapper_ = std::make_shared<TantivyIndexWrapper>(
            field_name.c_str(), path_.c_str(), min_gram, max_gram);
    }
}

void
NgramInvertedIndex::BuildWithFieldData(const std::vector<FieldDataPtr>& datas) {
    AssertInfo(schema_.data_type() == proto::schema::DataType::String ||
                   schema_.data_type() == proto::schema::DataType::VarChar,
               "schema data type is {}",
               schema_.data_type());
    index_build_begin_ = std::chrono::system_clock::now();
    InvertedIndexTantivy<std::string>::BuildWithFieldData(datas);
}

IndexStatsPtr
NgramInvertedIndex::Upload(const Config& config) {
    finish();
    auto index_build_end = std::chrono::system_clock::now();
    auto index_build_duration =
        std::chrono::duration<double>(index_build_end - index_build_begin_)
            .count();
    LOG_INFO("index build done for ngram index, field id: {}, duration: {}s",
             field_id_,
             index_build_duration);
    return InvertedIndexTantivy<std::string>::Upload(config);
}

void
NgramInvertedIndex::Load(milvus::tracer::TraceContext ctx,
                         const Config& config) {
    auto index_files =
        GetValueFromConfig<std::vector<std::string>>(config, "index_files");
    AssertInfo(index_files.has_value(),
               "index file paths is empty when load ngram index");

    auto files_value = index_files.value();
    auto it = std::find_if(
        files_value.begin(), files_value.end(), [](const std::string& file) {
            return file.substr(file.find_last_of('/') + 1) ==
                   "index_null_offset";
        });
    if (it != files_value.end()) {
        std::vector<std::string> file;
        file.push_back(*it);
        files_value.erase(it);
        auto index_datas = mem_file_manager_->LoadIndexToMemory(file);
        BinarySet binary_set;
        AssembleIndexDatas(index_datas, binary_set);
        auto index_valid_data = binary_set.GetByName("index_null_offset");
        folly::SharedMutex::WriteHolder lock(mutex_);
        null_offset_.resize((size_t)index_valid_data->size / sizeof(size_t));
        memcpy(null_offset_.data(),
               index_valid_data->data.get(),
               (size_t)index_valid_data->size);
    }

    disk_file_manager_->CacheNgramIndexToDisk(files_value);
    AssertInfo(
        tantivy_index_exist(path_.c_str()), "index not exist: {}", path_);
    wrapper_ = std::make_shared<TantivyIndexWrapper>(path_.c_str(),
                                                     milvus::index::SetBitset);
    LOG_INFO(
        "load ngram index done for field id:{} with dir:{}", field_id_, path_);
}

std::optional<TargetBitmap>
NgramInvertedIndex::InnerMatchQuery(const std::string& literal,
                                    exec::SegmentExpr* segment) {
    LOG_INFO(
        "debug=== InnerMatchQuery, literal: {}, min_gram: {}, max_gram: {}, "
        "Count {}",
        literal,
        min_gram_,
        max_gram_,
        Count());
    if (literal.length() < min_gram_) {
        return std::nullopt;
    }

    TargetBitmap bitset{static_cast<size_t>(Count())};
    wrapper_->inner_match_ngram(literal, min_gram_, max_gram_, &bitset);

    // Post filtering: if the literal length is larger than the max_gram
    // we need to filter out the bitset
    if (literal.length() > max_gram_) {
        LOG_INFO("debug=== post filtering, literal length: {}, max_gram: {}",
                 literal.length(),
                 max_gram_);
        auto bitset_off = 0;
        TargetBitmapView res(bitset);
        TargetBitmap valid(res.size(), true);
        TargetBitmapView valid_res(valid.data(), valid.size());

        auto execute_sub_batch = [&literal](const std::string_view* data,
                                            const bool* valid_data,
                                            const int32_t* offsets,
                                            const int size,
                                            TargetBitmapView res,
                                            TargetBitmapView valid_res) {
            auto next_off_option = res.find_first();
            while (next_off_option.has_value()) {
                auto next_off = next_off_option.value();
                if (next_off >= size) {
                    break;
                }
                if (data[next_off].find(literal) == std::string::npos) {
                    res[next_off] = false;
                }
                next_off_option = res.find_next(next_off);
            }
        };

        segment->ProcessAllDataChunk<std::string_view>(
            execute_sub_batch, std::nullptr_t{}, res, valid_res);
    }

    return std::optional<TargetBitmap>(std::move(bitset));
}

}  // namespace milvus::index
