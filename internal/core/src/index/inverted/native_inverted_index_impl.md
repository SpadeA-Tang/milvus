# Native Inverted Index — Phase 1 实现计划

> 对应设计文档：`native_inverted_index_design.md`
> 代码位置：`internal/core/src/index/inverted/`

## 实现步骤

### Step 1 — Storage 基础 ✅
- [x] Allocator 接口 + MallocAllocator → `Allocator.h`
- [x] FileReader（pread 封装）→ `FileIO.h`
- [x] FileWriter（write 封装）→ `FileIO.h`
- [x] BlockCache 接口 + PassthroughCache → `BlockCache.h`

### Step 2 — 写入数据结构 ✅
- [x] MemoryArena（1MB 分页，32-bit 地址）→ `MemoryArena.h`
- [x] ArenaHashMap（线性探测，key/value inline 存储在 MemoryArena）→ `ArenaHashMap.h`
- [x] ExpUnrolledLinkedList（指数增长块链表，delta-encoded vint）→ `ExpUnrolledLinkedList.h`

### Step 3 — 磁盘格式 ✅
- [x] VarInt 编码（unsigned LEB128）→ `postings/PostingFormat.h`
- [x] PostingsInfo 序列化/反序列化（inline + file reference）→ `postings/PostingFormat.h`
- [x] PostingList 编码：Inline（≤6 docs 内嵌 PostingsInfo）→ `postings/PostingFormat.h`
- [x] PostingList 编码：Bitpacking（delta + block-128 bitpacking）→ `postings/PostingFormat.h`
- [x] BlockEncoder / BlockDecoder（strictly sorted delta-1 编码，对齐 tantivy postings/compression）→ `postings/BlockEncoder.h`
- [x] SkipSerializer / SkipReader（per-block skip entries，encode_bitwidth/decode_bitwidth）→ `postings/SkipSerializer.h`
- [x] PostingsSerializer（block 累积 + skip + buffer-then-write）→ `postings/PostingsSerializer.h`
- [x] PostingsDecoder（skip-aware 解码，num_bits 从 skip data 读取）→ `postings/PostingsSerializer.h`
- [x] TermDictWriter / TermDictReader 接口 → `termdict/TermDictionary.h`
- [x] SortedBlockKeyMap（SSTable 实现）序列化 + 反序列化 + 查询 → `termdict/SortedBlockKeyMap.h`
- [x] 目录重组：storage/ indexing/ postings/ termdict/ segment/ query/

### Step 4 — 单 Segment ✅
- [x] SegmentMeta（segment 元信息序列化/反序列化）→ `segment/SegmentMeta.h`
- [x] SegmentWriter（ArenaHashMap + TermRecorder 累积 doc_id）→ `segment/SegmentWriter.h`
- [x] SegmentSerializer（遍历 → 排序 → PostingsSerializer + SortedBlockKeyMap 序列化到磁盘，upfront encoding decision）→ `postings/SegmentSerializer.h`
- [x] SegmentReader（加载 sparse index → 字典块 → PostingList 解码 → Bitset）→ `segment/SegmentReader.h`
- [x] Query / Weight 接口 → `query/Query.h`
- [x] TermQuery + TermWeight → `query/TermQuery.h`
- [x] BoolQuery（Must / Should / MustNot）→ `query/BoolQuery.h`

### Step 5 — 多 Segment ✅
- [x] SegmentId（auto-increment）→ `segment/SegmentMeta.h`
- [x] SegmentEntry（segment meta wrapper）→ `indexer/SegmentEntry.h`
- [x] SegmentRegister（HashMap 容器）→ `indexer/SegmentRegister.h`
- [x] SegmentManager（uncommitted/committed 双 register + shared_mutex）→ `indexer/SegmentManager.h`
- [x] MergePolicy 接口 + NoMergePolicy → `indexer/MergePolicy.h`
- [x] LogMergePolicy（按 log2 分层，最小段数触发合并）→ `indexer/LogMergePolicy.h`
- [x] TermMerger（K 路归并有序 term 流，min-heap）→ `termdict/TermMerger.h`
- [x] TermStreamer（Streamer 惰性 block-by-block 迭代，port of tantivy sstable::Streamer）→ `termdict/SortedBlockKeyMap.h`
- [x] SegmentMerger（K 路归并 term + merge-sort doc_ids + SegmentSerializer 写入）→ `indexer/SegmentMerger.h`

### Step 6 — Index API ✅
- [x] Stamper（atomic 操作时间戳，port of tantivy indexer/stamper.rs）→ `indexer/Stamper.h`
- [x] IndexMeta（segment 列表 + opstamp 二进制序列化，port of tantivy index/index_meta.rs）→ `indexer/IndexMeta.h`
- [x] IndexDirectory（文件追踪 + GC + meta 原子持久化，合并 ManagedDirectory + meta.json）→ `indexer/IndexDirectory.h`
  - 不需要 mmap（我们用 pread + BlockCache 手动管理内存）
  - 不需要 file watcher（commit/merge 完成后直接通知 reader）
  - 不需要 Directory 多态（单一具体类）
- [x] SegmentUpdater（segment 生命周期协调：add/commit/merge/GC，port of tantivy segment_updater.rs）→ `indexer/SegmentUpdater.h`
  - Phase 1 merge 同步执行，后续替换为线程池异步
- [x] Searcher（跨 segment 查询快照，OR 合并结果，port of tantivy core/searcher.rs）→ `indexer/Searcher.h`
- [x] IndexWriter（sender/worker 模型 + bounded channel + 内存预算自动 flush，port of tantivy index_writer.rs）→ `indexer/IndexWriter.h`
  - Phase 1 单 worker 线程，后续优化为 N workers round-robin 分发
- [x] IndexReader（mutex + shared_ptr snapshot，手动 reload，port of tantivy reader/mod.rs）→ `indexer/IndexReader.h`

### Step 7 — 集成 ✅
- [x] Tokenizer 接口 + SimpleTokenizer（port of tantivy tokenizer::Tokenizer/TokenStream）→ `Tokenizer.h`
  - 生产环境通过 `milvus::tantivy::Tokenizer` Rust FFI 桥接
- [x] NativeTextIndex（tokenize → IndexWriter → IndexReader，build/load/match_query）→ `NativeTextIndex.h`
  - 对应 `milvus::index::TextMatchIndex`，替换 tantivy Rust FFI 存储/查询层
- [x] 端到端测试：term_query + match_query + persistence + incremental + merge → `NativeTextIndexTest.cpp`

## 当前进度

**已完成**：Step 1–7（75 个单元测试全部通过，16 个测试文件放置在对应模块目录下）

## 优化 TODO

- [ ] **异步 I/O 调度**：当前使用线程池，I/O 阻塞线程，上千 index 实例共存时排队严重。目标方案：Milvus C++ core 升级 C++20 后，切换为 **C++20 coroutines + io_uring**（对标 Tokio：stackless 协作调度 + 内核异步 I/O）。C++17 下无完美替代，备选：io_uring + callback 事件循环（无协程，纯 callback 模型）。
- [ ] **Bitpacking SIMD 加速**：用 [simdcomp](https://github.com/lemire/simdcomp) 替换手写 bitpack/bitunpack。simdcomp 的 integrated delta API（`simdpackwithoutmaskd1` / `simdunpackd1` / `simdmaxbitsd1`）与我们的 block-128 设计完全对齐，pack/unpack + delta 一步完成。通过 FetchContent 引入，参考 serenedb 的集成方式。
- [ ] **VByte SIMD 加速**：考虑用 [streamvbyte](https://github.com/lemire/streamvbyte) 替换手写 VInt 编码（tail block + 其他场景）。
- [ ] **自适应编码策略**：参考 serenedb `format_block_128.hpp`，per-block 选择最优编码（all-same / bitpack / streamvbyte / bitset）。
