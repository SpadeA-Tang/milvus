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

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace milvus::index::inverted {

class FileReader {
 public:
    virtual void
    read(uint64_t offset, size_t len, void* buf) = 0;
    virtual uint64_t
    file_size() const = 0;
    virtual ~FileReader() = default;
};

class FileWriter {
 public:
    virtual void
    write(const void* data, size_t len) = 0;
    virtual uint64_t
    offset() const = 0;
    virtual void
    flush() = 0;
    virtual ~FileWriter() = default;
};

class LocalFileReader : public FileReader {
 public:
    explicit LocalFileReader(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open file for reading: " +
                                     path);
        }
        struct stat st;
        if (::fstat(fd_, &st) < 0) {
            ::close(fd_);
            throw std::runtime_error("Failed to stat file: " + path);
        }
        file_size_ = static_cast<uint64_t>(st.st_size);
    }

    ~LocalFileReader() override {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    LocalFileReader(const LocalFileReader&) = delete;
    LocalFileReader&
    operator=(const LocalFileReader&) = delete;

    void
    read(uint64_t offset, size_t len, void* buf) override {
        auto* dst = static_cast<char*>(buf);
        size_t remaining = len;
        uint64_t pos = offset;
        while (remaining > 0) {
            ssize_t n = ::pread(fd_, dst, remaining, static_cast<off_t>(pos));
            if (n < 0) {
                throw std::runtime_error("pread failed");
            }
            if (n == 0) {
                throw std::runtime_error("Unexpected EOF in pread");
            }
            dst += n;
            pos += static_cast<uint64_t>(n);
            remaining -= static_cast<size_t>(n);
        }
    }

    uint64_t
    file_size() const override {
        return file_size_;
    }

 private:
    int fd_ = -1;
    uint64_t file_size_ = 0;
};

class LocalFileWriter : public FileWriter {
 public:
    explicit LocalFileWriter(const std::string& path) {
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open file for writing: " +
                                     path);
        }
    }

    ~LocalFileWriter() override {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    LocalFileWriter(const LocalFileWriter&) = delete;
    LocalFileWriter&
    operator=(const LocalFileWriter&) = delete;

    void
    write(const void* data, size_t len) override {
        auto* src = static_cast<const char*>(data);
        size_t remaining = len;
        while (remaining > 0) {
            ssize_t n = ::write(fd_, src, remaining);
            if (n < 0) {
                throw std::runtime_error("write failed");
            }
            src += n;
            remaining -= static_cast<size_t>(n);
        }
        offset_ += len;
    }

    uint64_t
    offset() const override {
        return offset_;
    }

    void
    flush() override {
        if (fd_ >= 0) {
            ::fsync(fd_);
        }
    }

 private:
    int fd_ = -1;
    uint64_t offset_ = 0;
};

}  // namespace milvus::index::inverted
