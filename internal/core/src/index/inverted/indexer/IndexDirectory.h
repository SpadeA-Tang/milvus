// Combines tantivy's ManagedDirectory + save_metas/load_metas.
//
// Manages the file lifecycle for an inverted index directory:
// - Tracks which files are created (.managed)
// - Persists index metadata atomically (meta)
// - Garbage collects stale segment files after merge
//
// No Directory interface (no polymorphism needed).
// No mmap (we use pread + BlockCache).
// No file watcher (commit/merge directly notifies reader).

#pragma once

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>

#include "index/inverted/indexer/IndexMeta.h"
#include "index/inverted/segment/SegmentMeta.h"

namespace milvus::index::inverted {

inline const std::string kManagedFileName = ".managed";

class IndexDirectory {
 public:
    explicit IndexDirectory(const std::string& root_path)
        : root_path_(root_path) {
        std::filesystem::create_directories(root_path_);
        load_managed();
    }

    // --- Segment path helpers ---

    // Returns subdirectory path for a segment: root/seg_<id>/
    std::string
    segment_dir(SegmentId id) const {
        return root_path_ + "/seg_" + std::to_string(id.id);
    }

    // Create a segment subdirectory and register its files.
    void
    create_segment_dir(SegmentId id) {
        std::string dir = segment_dir(id);
        std::filesystem::create_directories(dir);
        // Register the four segment files.
        register_file("seg_" + std::to_string(id.id) + "/" + kPstFileName);
        register_file("seg_" + std::to_string(id.id) + "/" + kIdxFileName);
        register_file("seg_" + std::to_string(id.id) + "/" + kDctFileName);
        register_file("seg_" + std::to_string(id.id) + "/" + kMetaFileName);
    }

    // --- File tracking (port of ManagedDirectory) ---

    void
    register_file(const std::string& relative_path) {
        managed_files_.insert(relative_path);
        save_managed();
    }

    // --- Meta persistence (port of save_metas / load_metas) ---

    // Atomic write: write to tmp → fsync → rename.
    void
    save_meta(const IndexMeta& meta) const {
        std::string meta_path = root_path_ + "/" + kIndexMetaFileName;
        std::string tmp_path = meta_path + ".tmp";

        // Write to temp file.
        {
            LocalFileWriter writer(tmp_path);
            meta.serialize(&writer);
            writer.flush();
        }

        // fsync the file.
        {
            int fd = ::open(tmp_path.c_str(), O_RDONLY);
            if (fd >= 0) {
                ::fsync(fd);
                ::close(fd);
            }
        }

        // Atomic rename.
        std::rename(tmp_path.c_str(), meta_path.c_str());

        // fsync parent directory.
        {
            int fd = ::open(root_path_.c_str(), O_RDONLY | O_DIRECTORY);
            if (fd >= 0) {
                ::fsync(fd);
                ::close(fd);
            }
        }
    }

    IndexMeta
    load_meta() const {
        std::string meta_path = root_path_ + "/" + kIndexMetaFileName;
        if (!std::filesystem::exists(meta_path)) {
            return IndexMeta{};
        }
        LocalFileReader reader(meta_path);
        return IndexMeta::deserialize(&reader);
    }

    bool
    has_meta() const {
        return std::filesystem::exists(root_path_ + "/" + kIndexMetaFileName);
    }

    // --- GC (port of ManagedDirectory::garbage_collect) ---

    void
    garbage_collect(
        const std::unordered_set<std::string>& living_files) {
        std::vector<std::string> to_delete;
        for (const auto& f : managed_files_) {
            if (living_files.find(f) == living_files.end()) {
                to_delete.push_back(f);
            }
        }
        for (const auto& f : to_delete) {
            std::string full_path = root_path_ + "/" + f;
            std::filesystem::remove(full_path);
            managed_files_.erase(f);
        }
        // Clean up empty segment directories.
        for (const auto& entry :
             std::filesystem::directory_iterator(root_path_)) {
            if (entry.is_directory() &&
                std::filesystem::is_empty(entry.path())) {
                std::filesystem::remove(entry.path());
            }
        }
        save_managed();
    }

    const std::string&
    root_path() const {
        return root_path_;
    }

    const std::unordered_set<std::string>&
    managed_files() const {
        return managed_files_;
    }

 private:
    // Persist managed file list to .managed.
    // Simple format: one relative path per line.
    void
    save_managed() const {
        std::string path = root_path_ + "/" + kManagedFileName;
        std::ofstream out(path, std::ios::trunc);
        for (const auto& f : managed_files_) {
            out << f << "\n";
        }
    }

    // Load managed file list from .managed.
    void
    load_managed() {
        std::string path = root_path_ + "/" + kManagedFileName;
        if (!std::filesystem::exists(path)) {
            return;
        }
        std::ifstream in(path);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty()) {
                managed_files_.insert(line);
            }
        }
    }

    std::string root_path_;
    std::unordered_set<std::string> managed_files_;
};

}  // namespace milvus::index::inverted
