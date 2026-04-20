# Native Inverted Index — Code Review Checklist

总代码量：37 个头文件 (~4,600 行) + 16 个测试文件 (~2,500 行)，共 ~7,100 行。
划分为 **8 个 Review Session**，每个 session 控制在 300-600 行代码，约 30-60 分钟。


/port-review storage        → Allocator, FileIO, BlockCache, MemoryArena

/port-review indexing        → ExpUnrolledLinkedList, SharedArenaHashMap

/port-review encoding        → Recorder, PostingFormat, BlockEncoder

/port-review postings        → Skip, Serializer, Writer, SegmentSerializer

/port-review termdict        → SortedBlockKeyMap, TermMerger, SegmentReader

/port-review query           → Query, TermQuery, BoolQuery

/port-review segment-mgmt    → Stamper, SegmentManager, MergePolicy

/port-review writer          → IndexWriter, IndexReader, SegmentUpdater, Merger

/port-review integration     → Tokenizer, NativeTextIndex

**Review 原则：** 移植 tantivy，不自己发明。每个模块对照 tantivy 源码检查是否忠实移植。

---

## Session 1: 内存基础设施 (~400 行代码 + ~160 行测试)

**目标：** 确认内存分配和文件 I/O 的正确性。这是所有上层模块的地基。

### 文件

| 文件 | 行数 | 核心类 |
|------|------|--------|
| `storage/Allocator.h` | 59 | Allocator, MallocAllocator |
| `storage/FileIO.h` | 151 | FileReader/Writer, LocalFileReader/Writer |
| `storage/BlockCache.h` | 83 | CacheKey, BlockCache, PassthroughCache |
| `indexing/MemoryArena.h` | 189 | Addr, MemoryArena |

### 测试

| 文件 | 行数 |
|------|------|
| `storage/AllocatorTest.cpp` | 24 |
| `storage/FileIOTest.cpp` | 41 |
| `storage/BlockCacheTest.cpp` | 26 |
| `indexing/MemoryArenaTest.cpp` | 92 |

### Review 要点

**Allocator.h**
- [ ] atomic counter 的内存序是否正确（relaxed 够了吗？）
- [ ] deallocate 时 size 不匹配是否会导致计数错误
- [ ] 与 tantivy 的 `MemoryArena` 计量方式对照

**FileIO.h**
- [ ] partial read/write 循环是否正确处理 EINTR
- [ ] pread 的 thread-safety（无状态，不共享 file offset）
- [ ] 错误处理：文件打开失败、读写失败的异常路径
- [ ] flush 实现是否足够（fsync vs fflush）

**BlockCache.h**
- [ ] PassthroughCache 是否真的 passthrough（每次都调 loader）
- [ ] CacheKey 的 hash 是否合理（segment_id + file_type + offset 组合）
- [ ] 接口是否为 Phase 2 的 LRU/两级缓存预留了足够空间

**MemoryArena.h**
- [ ] 32-bit 地址编码：12-bit page_id + 20-bit offset，1MB page → 最大 4096 pages = 4GB，是否够用
- [ ] page 分配的边界条件：对象跨 page 边界时是否正确处理（是否允许跨页？）
- [ ] read<T> / write_at<T> 使用 memcpy 处理 unaligned access，是否对所有 T 正确
- [ ] slice() 返回的数据是否连续（多 page 时怎么处理）
- [ ] 与 tantivy 的 `stacker::MemoryArena` 对照：page 大小、地址编码方式

---

## Session 2: 核心数据结构 (~520 行代码 + ~280 行测试)

**目标：** 确认内存中的倒排表构建逻辑。这两个结构是写入热路径的核心。

### 文件

| 文件 | 行数 | 核心类 |
|------|------|--------|
| `indexing/ExpUnrolledLinkedList.h` | 152 | ExpUnrolledLinkedList, Writer |
| `indexing/SharedArenaHashMap.h` | 298 | SharedArenaHashMap, KeyValue, LinearProbing |
| `indexing/ArenaHashMap.h` | 73 | ArenaHashMap (convenience wrapper) |

### 测试

| 文件 | 行数 |
|------|------|
| `indexing/ExpUnrolledLinkedListTest.cpp` | 136 |
| `indexing/ArenaHashMapTest.cpp` | 140 |

### Review 要点

**ExpUnrolledLinkedList.h**
- [ ] 块大小增长序列：8 → 16 → … → 32KB，与 tantivy `expull` 对照
- [ ] Writer 写满当前块后分配新块，链接指针是否正确
- [ ] read_to_end 遍历链表时，各块的有效数据长度如何确定
- [ ] 空链表的边界处理
- [ ] varint 编码的数据是否可能在块边界被截断？如何处理？

**SharedArenaHashMap.h**
- [ ] MurmurHash2 实现是否标准
- [ ] Linear probing：load factor 阈值是多少？resize 时 rehash 逻辑是否正确
- [ ] key 存储：key bytes 内联在 arena 里，KeyValue 布局是否紧凑
- [ ] mutate_or_create：不存在时 create、存在时 mutate，原子性保证（单线程即可）
- [ ] 与 tantivy 的 `stacker::HashMap` 对照：hash 函数、probing 策略、resize 策略
- [ ] Power-of-2 table size 的 mask 运算是否正确

**ArenaHashMap.h**
- [ ] 只是 SharedArenaHashMap + MemoryArena 的 wrapper，确认 lifetime 管理正确

---

## Session 3: Postings 编码 (~670 行代码 + ~500 行测试)

**目标：** 确认 posting list 的编码和解码正确性。这是数据完整性的关键。

### 文件

| 文件 | 行数 | 核心类 |
|------|------|--------|
| `postings/Recorder.h` | 188 | Recorder<Derived> (CRTP), DocIdRecorder, BufferLender |
| `postings/PostingFormat.h` | 360 | PostingsInfo, BitpackingPostingCodec, PostingListCodec |
| `postings/BlockEncoder.h` | 93 | BlockEncoder, BlockDecoder |

### 测试

| 文件 | 行数 |
|------|------|
| `postings/PostingFormatTest.cpp` | 229 |

### Review 要点

**Recorder.h**
- [ ] CRTP 模式：Recorder<DocIdRecorder> 的静态分发是否正确
- [ ] DocIdRecorder 存储在 HashMap value 里（trivially copyable），确认 sizeof 和 alignment
- [ ] BufferLender 的 buffer 复用逻辑，避免重复分配
- [ ] 与 tantivy 的 `Recorder` trait 对照

**PostingFormat.h**
- [ ] **inline 编码**：≤6 docs 直接存 PostingsInfo，阈值是否与 tantivy 一致
- [ ] **bitpacking 编码**：128-block + VInt tail 的分界逻辑
- [ ] varint 编码/解码是否标准（与 tantivy 的 VInt 兼容）
- [ ] PostingsInfo 的 serialize/deserialize 是否 round-trip 正确
- [ ] delta 编码：doc_ids 必须严格递增，delta-1 优化是否正确（允许 delta=0？）

**BlockEncoder.h**
- [ ] compress_block_sorted：128 个 doc_id → delta → bitpack，bit width 计算是否正确
- [ ] uncompress_block_sorted：反向 unpack → 累加还原，是否与 compress 完全对称
- [ ] 输出 buffer 的 size 计算：`128 * num_bits / 8`，边界情况（num_bits=0 或 32）
- [ ] 与 tantivy `BlockEncoder` 对照：bitpacking 库的选择

---

## Session 4: Postings 序列化管道 (~490 行代码 + ~530 行测试)

**目标：** 确认从内存 posting list 到磁盘文件的完整写入管道。

### 文件

| 文件 | 行数 | 核心类 |
|------|------|--------|
| `postings/SkipSerializer.h` | 223 | SkipSerializer, SkipReader, BlockInfo |
| `postings/PostingsSerializer.h` | 222 | PostingsSerializer, PostingsDecoder |
| `postings/PostingsWriter.h` | 159 | PostingsWriter<Rec>, TermAddr |
| `postings/SegmentSerializer.h` | 110 | SegmentSerializer |

### 测试

| 文件 | 行数 |
|------|------|
| `postings/SkipSerializerTest.cpp` | 91 |
| `postings/PostingsSerializerTest.cpp` | 186 |
| `postings/PostingsWriterTest.cpp` | 252 |

### Review 要点

**SkipSerializer.h**
- [ ] skip entry 格式：4 字节 last_doc (LE) + 1 字节 bit_width = 5 字节/block
- [ ] SkipReader::seek(target)：二分还是线性扫描？性能考虑
- [ ] 空 skip list（0 个满 block）的处理
- [ ] 与 tantivy skip serialization 对照

**PostingsSerializer.h**
- [ ] 128-block 积累：满 block → flush (bitpack) → skip entry，尾部 → VInt
- [ ] close_term 时 flush 尾部的逻辑是否正确
- [ ] PostingsDecoder：先读 skip data 定位 block，再解码，流程是否正确
- [ ] Block 结构体的 offset/size 计算

**PostingsWriter.h**
- [ ] ArenaHashMap<DocIdRecorder> 的 memory 估算是否准确（用于 budget 判断）
- [ ] serialize_postings：按 term 排序 → 遍历 → 逐个写入 SegmentSerializer
- [ ] subscribe(term, doc_id) 的去重逻辑（同一 term 同一 doc 是否需要去重？）
- [ ] 与 tantivy 的 SegmentWriter 对照

**SegmentSerializer.h**
- [ ] 驱动 TermDictionaryWriter + PostingsSerializer 的协调逻辑
- [ ] new_term / write_doc / close_term 的调用顺序约束
- [ ] inline vs bitpacking 的决策点（cardinality ≤ 6）
- [ ] close() 时 finalize 所有 writer 的顺序

---

## Session 5: 词典与 Segment 读取 (~570 行代码 + ~250 行测试)

**目标：** 确认词典的写入、查找、合并，以及 segment 的完整读取流程。

### 文件

| 文件 | 行数 | 核心类 |
|------|------|--------|
| `termdict/TermDictionary.h` | 51 | TermDictionaryWriter/Reader (interfaces) |
| `termdict/SortedBlockKeyMap.h` | 303 | SortedBlockKeyMapWriter/Reader, Streamer, SparseIndex |
| `termdict/TermMerger.h` | 96 | TermMerger, HeapItem |
| `segment/SegmentMeta.h` | 107 | SegmentId, SegmentMeta |
| `segment/SegmentReader.h` | 116 | SegmentReader |

### 测试

| 文件 | 行数 |
|------|------|
| `termdict/SortedBlockKeyMapTest.cpp` | 253 |

### Review 要点

**SortedBlockKeyMap.h（最复杂的词典实现）**
- [ ] Writer：128-entry block 切分，sparse index 每 block 记录首 key + offset
- [ ] Reader::lookup：binary search sparse index → 加载 dict block → block 内线性搜索
- [ ] Streamer：逐 block 惰性加载，advance() 跨 block 时是否正确
- [ ] DictBlock 的序列化格式：key_len + key_bytes + value_bytes 的布局
- [ ] 空词典的处理
- [ ] 与 tantivy SSTable 对照：block 大小、sparse index 策略

**TermMerger.h**
- [ ] min-heap 的比较函数：按 term key 字典序
- [ ] 同名 term 出现在多个 segment 时，current_streamers() 返回所有包含该 term 的 streamer
- [ ] advance() 后 heap 的 re-heapify 是否正确
- [ ] 与 tantivy merger 对照

**SegmentMeta.h**
- [ ] SegmentId 的全局唯一性：atomic counter，是否进程级别够用
- [ ] serialize / deserialize 的 round-trip

**SegmentReader.h**
- [ ] 加载 4 个文件 (.pst, .idx, .dct, .meta) 的顺序和错误处理
- [ ] term_query：lookup term → 根据 PostingsInfo 选 inline/bitpacking 解码 → 返回 bitset
- [ ] bitset 大小 = max_doc_id + 1 还是 num_docs？确认语义
- [ ] thread-safety：pread 是无状态的，多线程并发查询是否安全

---

## Session 6: Query 层 (~190 行代码 + ~200 行测试)

**目标：** 确认查询的组合与执行逻辑。这是面向用户的查询接口。

### 文件

| 文件 | 行数 | 核心类 |
|------|------|--------|
| `query/Query.h` | 40 | Query, Weight (abstract) |
| `query/TermQuery.h` | 52 | TermQuery, TermWeight |
| `query/BoolQuery.h` | 97 | BoolQuery, BoolWeight |

### 测试

| 文件 | 行数 |
|------|------|
| `query/QueryTest.cpp` | 203 |

### Review 要点

**Query.h**
- [ ] Query → Weight 两级抽象是否与 tantivy 一致
- [ ] Weight::execute 返回 Bitset，生命周期是否清晰

**TermQuery.h**
- [ ] TermWeight 持有 term string 的 copy 还是 reference？lifetime 是否安全
- [ ] execute 直接调 reader.term_query()，异常传播是否正确

**BoolQuery.h**
- [ ] Must / Should / MustNot 三种 Occur 的 bitset 运算：
  - Must: AND (intersect)
  - Should: OR (union)
  - MustNot: AND NOT (difference)
- [ ] 组合顺序：先算 Must → 再 OR Should → 最后 AND NOT MustNot？
- [ ] 边界情况：只有 Should 没有 Must 时的行为
- [ ] 边界情况：空 clause 列表
- [ ] 与 tantivy BooleanQuery 对照

---

## Session 7: Indexer — 生命周期管理 (~1,480 行代码 + ~630 行测试)

**目标：** 这是最大的一层，核心是写入的线程模型和 segment 生命周期。建议分两次 review。

### Session 7a: Segment 管理 (~530 行)

| 文件 | 行数 | 核心类 |
|------|------|--------|
| `indexer/Stamper.h` | 44 | Stamper |
| `indexer/IndexMeta.h` | 55 | IndexMeta |
| `indexer/SegmentEntry.h` | 29 | SegmentEntry |
| `indexer/SegmentRegister.h` | 103 | SegmentRegister |
| `indexer/SegmentManager.h` | 150 | SegmentManager, SegmentRegisters |
| `indexer/MergePolicy.h` | 42 | MergePolicy, NoMergePolicy |
| `indexer/LogMergePolicy.h` | 106 | LogMergePolicy |

**测试：** `SegmentManagerTest.cpp` (181 行)

**Review 要点**
- [ ] SegmentManager 的状态机：Uncommitted → Committed，转换是否 atomic
- [ ] shared_mutex 的使用：读锁 vs 写锁的粒度是否合理
- [ ] LogMergePolicy：按 log2(num_docs) 分层，≥8 个 segment 触发合并，参数是否与 tantivy 一致
- [ ] start_merge / end_merge 的锁协议：合并期间新 segment 能否继续创建
- [ ] Stamper 的 stamps(n) 批量预留是否正确

### Session 7b: Writer / Reader / Merge / Directory (~950 行)

| 文件 | 行数 | 核心类 |
|------|------|--------|
| `indexer/IndexDirectory.h` | 184 | IndexDirectory |
| `indexer/SegmentUpdater.h` | 207 | SegmentUpdater |
| `indexer/SegmentMerger.h` | 125 | SegmentMerger |
| `indexer/Searcher.h` | 82 | Searcher |
| `indexer/IndexWriter.h` | 283 | IndexWriter, BoundedChannel, WriterOperation |
| `indexer/IndexReader.h` | 69 | IndexReader |

**测试：** `IndexWriterTest.cpp` (232 行), `SegmentMergerTest.cpp` (220 行)

**Review 要点**

**IndexWriter.h（最关键）**
- [ ] BoundedChannel：send 阻塞当队列满、recv 阻塞当队列空、close 后 recv 返回 nullopt
- [ ] condition_variable 的 wait 条件是否正确（spurious wakeup 安全）
- [ ] WriterOperation = variant<Add, Commit, Shutdown>，move 语义是否正确（promise 不可 copy）
- [ ] worker_loop：收到 AddOperation 累积到 PostingsWriter → 超 budget 时 flush → CommitOperation 时 flush + 设 promise
- [ ] 析构函数：发 ShutdownOperation → join worker，是否保证所有数据都 flush
- [ ] memory budget 的计算：PostingsWriter 的内存估算是否准确
- [ ] 与 tantivy IndexWriter 的 sender/worker 模型对照

**SegmentUpdater.h**
- [ ] commit()：SegmentManager::commit() + save_meta + consider_merge
- [ ] consider_merge_options：调 MergePolicy → 执行 merge → end_merge → GC
- [ ] 同步 merge（Phase 1），是否会阻塞写入
- [ ] 死锁风险：mutex_ 与 SegmentManager 的锁是否有嵌套

**SegmentMerger.h**
- [ ] K-way merge：TermMerger 归并 term → 每个 term 合并 doc_id 列表 → 写新 segment
- [ ] doc_id 重映射：合并后 doc_id 如何重新编号
- [ ] 合并后旧 segment 的清理由 GC 负责，这里只生成新 segment

**IndexDirectory.h**
- [ ] atomic rename 保存 meta（先写 tmp 再 rename）是否正确
- [ ] .managed 文件跟踪机制
- [ ] garbage_collect：删除不在 committed segments 中的文件
- [ ] 并发安全：GC 与读取是否冲突

**Searcher.h + IndexReader.h**
- [ ] Searcher 持有 SegmentReader 快照，immutable
- [ ] IndexReader::reload 重建 Searcher，通过 shared_ptr + mutex 实现 snapshot 切换
- [ ] 旧 Searcher 的生命周期：有查询在用时不能释放

---

## Session 8: Integration + 全部测试 (~330 行代码 + ~360 行测试)

**目标：** 确认 Tokenizer 桥接和 NativeTextIndex 的端到端正确性。

### 文件

| 文件 | 行数 | 核心类 |
|------|------|--------|
| `Tokenizer.h` | 78 | TokenStream, Tokenizer, SimpleTokenizer |
| `TantivyTokenizer.h` | 65 | TantivyTokenStreamAdapter, TantivyTokenizer |
| `NativeTextIndex.h` | 184 | NativeTextIndex |

### 测试

| 文件 | 行数 |
|------|------|
| `NativeTextIndexTest.cpp` | 360 |

### Review 要点

**Tokenizer.h**
- [ ] TokenStream 接口：advance() + token()，是否与 tantivy 的 TokenStream trait 一致
- [ ] SimpleTokenizer 的分词逻辑：按非字母数字分割 + 小写化，是否足够作为测试 tokenizer

**TantivyTokenizer.h**
- [ ] FFI 桥接：milvus::tantivy::Tokenizer → inverted::Tokenizer，adapter 是否正确
- [ ] mutable string cache 的 thread-safety（token_stream 返回的 stream 是否单线程使用）
- [ ] 构造函数接收 analyzer_params，传递给 Rust 侧

**NativeTextIndex.h**
- [ ] 两种模式：write mode (IndexWriter) vs read-only mode (IndexDirectory + IndexReader)
- [ ] build()：遍历 texts → tokenize → add_document → commit，流程是否正确
- [ ] match_query(query, min_should_match)：tokenize query → 逐 term 查 bitset → 按 doc 计数 ≥ threshold
- [ ] min_should_match > 查询 term 数时返回空 bitset
- [ ] open() 静态工厂：read-only 模式的创建
- [ ] set_merge_policy 传递给 SegmentUpdater

**NativeTextIndexTest.cpp（11 个测试）**
- [ ] 基础：TokenizerSimple, TokenizerEmpty
- [ ] 构建与查询：BuildAndTermQuery, MatchQuery
- [ ] 持久化：Persistence（写入 → 重新打开 → 查询）
- [ ] 增量写入：IncrementalAddAndReload
- [ ] 合并：MergeWithTokenizedData
- [ ] Text Match：CaseAndPunctuation, MinShouldMatch, EmptyQuery, LargeCorpus
- [ ] 是否覆盖了边界情况：空文档、空查询、单 term、全匹配、零匹配

---

## 跨模块 Review 要点（贯穿所有 Session）

在每个 session 中都需要关注的共性问题：

### 正确性
- [ ] 与 tantivy 对应模块的 **语义一致性**（不要求代码 1:1，但行为必须一致）
- [ ] 所有 serialize / deserialize 是否 round-trip 正确
- [ ] 整数溢出：doc_id (u32), offset (u32/u64), 地址 (u32) 的边界
- [ ] 空输入的处理：空 term、空 posting list、空 segment、空查询

### 内存安全
- [ ] Arena 分配的对象不能单独释放，只能整体释放 — 是否有误用
- [ ] shared_ptr / unique_ptr 的所有权是否清晰
- [ ] move 语义：variant 中的 promise、channel 中的 operation

### 线程安全
- [ ] IndexWriter 的 channel + worker 模型
- [ ] SegmentManager 的 shared_mutex
- [ ] IndexReader 的 snapshot 切换
- [ ] SegmentReader 的 pread（天然线程安全）

### 性能（标记但不阻塞）
- [ ] 标记可优化点，但 Phase 1 不要求优化
- [ ] 例如：SkipReader 线性扫描 → 二分、BlockCache passthrough → LRU

---

## Review 进度

| Session | 范围 | 状态 | 备注 |
|---------|------|------|------|
| 1 | 内存基础设施 | ⬜ 未开始 | |
| 2 | 核心数据结构 | ⬜ 未开始 | |
| 3 | Postings 编码 | ⬜ 未开始 | |
| 4 | Postings 序列化管道 | ⬜ 未开始 | |
| 5 | 词典与 Segment 读取 | ⬜ 未开始 | |
| 6 | Query 层 | ⬜ 未开始 | |
| 7a | Segment 管理 | ⬜ 未开始 | |
| 7b | Writer/Reader/Merge | ⬜ 未开始 | |
| 8 | Integration | ⬜ 未开始 | |
