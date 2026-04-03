# Details

Date : 2026-04-03 17:53:36

Directory /home/spadea/working2/milvus/internal/core/src/index/inverted

Total : 52 files,  5427 codes, 870 comments, 1296 blanks, all 7593 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [indexer/IndexDirectory.h](/indexer/IndexDirectory.h) | C++ | 131 | 26 | 28 | 185 |
| [indexer/IndexMeta.h](/indexer/IndexMeta.h) | C++ | 39 | 6 | 11 | 56 |
| [indexer/IndexReader.h](/indexer/IndexReader.h) | C++ | 44 | 12 | 14 | 70 |
| [indexer/IndexWriter.h](/indexer/IndexWriter.h) | C++ | 214 | 30 | 40 | 284 |
| [indexer/IndexWriterTest.cpp](/indexer/IndexWriterTest.cpp) | C++ | 162 | 22 | 49 | 233 |
| [indexer/LogMergePolicy.h](/indexer/LogMergePolicy.h) | C++ | 84 | 7 | 16 | 107 |
| [indexer/MergePolicy.h](/indexer/MergePolicy.h) | C++ | 23 | 11 | 9 | 43 |
| [indexer/Searcher.h](/indexer/Searcher.h) | C++ | 61 | 10 | 12 | 83 |
| [indexer/SegmentEntry.h](/indexer/SegmentEntry.h) | C++ | 17 | 4 | 9 | 30 |
| [indexer/SegmentManager.h](/indexer/SegmentManager.h) | C++ | 121 | 10 | 20 | 151 |
| [indexer/SegmentManagerTest.cpp](/indexer/SegmentManagerTest.cpp) | C++ | 138 | 10 | 34 | 182 |
| [indexer/SegmentMerger.h](/indexer/SegmentMerger.h) | C++ | 78 | 27 | 21 | 126 |
| [indexer/SegmentMergerTest.cpp](/indexer/SegmentMergerTest.cpp) | C++ | 168 | 13 | 40 | 221 |
| [indexer/SegmentRegister.h](/indexer/SegmentRegister.h) | C++ | 83 | 4 | 17 | 104 |
| [indexer/SegmentUpdater.h](/indexer/SegmentUpdater.h) | C++ | 143 | 30 | 35 | 208 |
| [indexer/Stamper.h](/indexer/Stamper.h) | C++ | 27 | 7 | 11 | 45 |
| [indexing/ArenaHashMap.h](/indexing/ArenaHashMap.h) | C++ | 48 | 11 | 15 | 74 |
| [indexing/ArenaHashMapTest.cpp](/indexing/ArenaHashMapTest.cpp) | C++ | 111 | 4 | 26 | 141 |
| [indexing/ExpUnrolledLinkedList.h](/indexing/ExpUnrolledLinkedList.h) | C++ | 102 | 26 | 25 | 153 |
| [indexing/ExpUnrolledLinkedListTest.cpp](/indexing/ExpUnrolledLinkedListTest.cpp) | C++ | 110 | 3 | 24 | 137 |
| [indexing/MemoryArena.h](/indexing/MemoryArena.h) | C++ | 134 | 26 | 30 | 190 |
| [indexing/MemoryArenaTest.cpp](/indexing/MemoryArenaTest.cpp) | C++ | 72 | 1 | 20 | 93 |
| [indexing/SharedArenaHashMap.h](/indexing/SharedArenaHashMap.h) | C++ | 232 | 34 | 33 | 299 |
| [native\_inverted\_index\_design.md](/native_inverted_index_design.md) | Markdown | 320 | 0 | 133 | 453 |
| [native\_inverted\_index\_impl.md](/native_inverted_index_impl.md) | Markdown | 68 | 0 | 15 | 83 |
| [postings/BlockEncoder.h](/postings/BlockEncoder.h) | C++ | 51 | 27 | 16 | 94 |
| [postings/PostingFormat.h](/postings/PostingFormat.h) | C++ | 264 | 44 | 53 | 361 |
| [postings/PostingFormatTest.cpp](/postings/PostingFormatTest.cpp) | C++ | 192 | 7 | 31 | 230 |
| [postings/PostingsSerializer.h](/postings/PostingsSerializer.h) | C++ | 132 | 57 | 34 | 223 |
| [postings/PostingsSerializerTest.cpp](/postings/PostingsSerializerTest.cpp) | C++ | 163 | 8 | 16 | 187 |
| [postings/PostingsWriter.h](/postings/PostingsWriter.h) | C++ | 113 | 26 | 21 | 160 |
| [postings/PostingsWriterTest.cpp](/postings/PostingsWriterTest.cpp) | C++ | 191 | 19 | 43 | 253 |
| [postings/Recorder.h](/postings/Recorder.h) | C++ | 86 | 82 | 21 | 189 |
| [postings/SegmentSerializer.h](/postings/SegmentSerializer.h) | C++ | 72 | 21 | 18 | 111 |
| [postings/SkipSerializer.h](/postings/SkipSerializer.h) | C++ | 149 | 40 | 35 | 224 |
| [postings/SkipSerializerTest.cpp](/postings/SkipSerializerTest.cpp) | C++ | 62 | 11 | 19 | 92 |
| [query/BoolQuery.h](/query/BoolQuery.h) | C++ | 68 | 12 | 18 | 98 |
| [query/Query.h](/query/Query.h) | C++ | 19 | 12 | 10 | 41 |
| [query/QueryTest.cpp](/query/QueryTest.cpp) | C++ | 162 | 6 | 36 | 204 |
| [query/TermQuery.h](/query/TermQuery.h) | C++ | 31 | 10 | 12 | 53 |
| [segment/SegmentMeta.h](/segment/SegmentMeta.h) | C++ | 71 | 20 | 17 | 108 |
| [segment/SegmentReader.h](/segment/SegmentReader.h) | C++ | 78 | 18 | 21 | 117 |
| [storage/Allocator.h](/storage/Allocator.h) | C++ | 40 | 10 | 10 | 60 |
| [storage/AllocatorTest.cpp](/storage/AllocatorTest.cpp) | C++ | 17 | 0 | 8 | 25 |
| [storage/BlockCache.h](/storage/BlockCache.h) | C++ | 58 | 12 | 14 | 84 |
| [storage/BlockCacheTest.cpp](/storage/BlockCacheTest.cpp) | C++ | 18 | 1 | 8 | 27 |
| [storage/FileIO.h](/storage/FileIO.h) | C++ | 121 | 10 | 21 | 152 |
| [storage/FileIOTest.cpp](/storage/FileIOTest.cpp) | C++ | 29 | 2 | 11 | 42 |
| [termdict/SortedBlockKeyMap.h](/termdict/SortedBlockKeyMap.h) | C++ | 222 | 32 | 50 | 304 |
| [termdict/SortedBlockKeyMapTest.cpp](/termdict/SortedBlockKeyMapTest.cpp) | C++ | 201 | 18 | 35 | 254 |
| [termdict/TermDictionary.h](/termdict/TermDictionary.h) | C++ | 23 | 15 | 14 | 52 |
| [termdict/TermMerger.h](/termdict/TermMerger.h) | C++ | 64 | 16 | 17 | 97 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)