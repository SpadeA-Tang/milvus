#include <gtest/gtest.h>

#include <cstdio>
#include <string>

#include "index/inverted/storage/FileIO.h"

using namespace milvus::index::inverted;

TEST(FileIOTest, WriteAndRead) {
    std::string path = "/tmp/test_inverted_fileio.dat";

    // Write
    {
        LocalFileWriter writer(path);
        uint32_t val1 = 42;
        writer.write(&val1, sizeof(val1));
        EXPECT_EQ(writer.offset(), sizeof(uint32_t));

        uint32_t val2 = 99;
        writer.write(&val2, sizeof(val2));
        EXPECT_EQ(writer.offset(), 2 * sizeof(uint32_t));
        writer.flush();
    }

    // Read
    {
        LocalFileReader reader(path);
        EXPECT_EQ(reader.file_size(), 2 * sizeof(uint32_t));

        uint32_t val1 = 0;
        reader.read(0, sizeof(val1), &val1);
        EXPECT_EQ(val1, 42u);

        uint32_t val2 = 0;
        reader.read(sizeof(uint32_t), sizeof(val2), &val2);
        EXPECT_EQ(val2, 99u);
    }

    std::remove(path.c_str());
}
