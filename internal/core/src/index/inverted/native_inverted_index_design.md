> 目标：替换 tantivy FFI 倒排索引，用 C++ 原生实现，支持读路径内存手工控制。

# 一、设计约束

| 约束 | 决策 |
|------|------|
| 语言 | C++，tokenizer 保留，通过 Rust FFI 调用 |
| 磁盘格式 | 全新设计，不兼容 tantivy |
| Doc ID | 外部指定，直接存入 posting list （当前除 json key stats 外的默认方案） |
| Delete | 不支持 |
| Fast Field | 不需要，外部维护 （针对 json key stats） |
| BM25 Scoring | 不需要 |
| Segment | 多 segment 建索引 + growing 索引内部自动 merge + seal 最终合并成一个 segment |
| 并发模型 | 前台单线程写 + 后台多 worker 构建 segment + 多线程读，merge 期间可并发查询 |
| 内存控制 | 读路径基于 memory controller 统一控制，写路径基于预算控制 |
| 内存分配 | Allocator 接口抽象，初期标准分配器，后续接入统一内存管理 |

# 二、为什么不继续用 tantivy

1. **内存不可控**：tantivy 读路径重度依赖 mmap，物理内存占用由 OS page cache 决定，应用层无法精确控制
2. **FST 必须全量加载**：Term Dictionary 使用 FST，无法部分加载，大词表时占用几十 MB 且不可淘汰
3. **功能冗余**：Store、BM25 scoring、WAND block-max、columnar fast field 等 Milvus 不需要的功能占据大量代码

# 三、参考系统

## tantivy

写入框架参考 tantivy：
- 多 worker 后台构建 segment（channel 发送写请求 + 独立 SegmentWriter）
- 内存累积与磁盘序列化解耦（序列化格式可基于数据选择）
- per-worker 内存预算，达到阈值自动 flush
- 多 segment + MergePolicy + 后台 merge
- commit / reload 语义，snapshot 隔离读写并发
- growing segment 支持（高频 flush 到磁盘后查询，实现近实时搜索）

tantivy 不适合的部分：读路径依赖 mmap，内存不可控。

## ClickHouse Text Index v3

缓存系统参考 ClickHouse：
- **应用级缓存**：缓存反序列化后的内存对象（非文件字节），避免重复解压 + 反序列化
- **分级缓存**：不同类型数据独立缓存，各自管理容量和淘汰
- **精确内存控制**：应用层完全掌控缓存内存用量，不依赖 OS page cache

ClickHouse 的 term dictionary 使用 FST（与 tantivy/Lucene 同一思路），不过是分块的，易于缓存。磁盘格式与缓存系统正交——不同格式通过同一套缓存系统管理内存。

ClickHouse 不适合的部分：单线程写入，无 merge 系统（总是对 sealed 使用），无 phrase query 等。

整体架构来看，可以理解为 tantivy + clickhouse 的缓存系统：tantivy 的写入系统更符合 milvus 的需求，query 更加丰富且易于扩展，clickhouse 的缓存系统可以实现内存可控，并且在热路径下有更好的性能（省掉了 deserialize 的过程）。

# 四、分层架构

```
┌─────────────────────────────────────────────────────┐
│                   Index API Layer                    │
│         IndexWriter    IndexReader    IndexConfig    │
├─────────────────────────────────────────────────────┤
│                  Query Layer                         │
│    Query → Weight → Collector (Bitset/Facet/TopN)   │
├─────────────────────────────────────────────────────┤
│                  Segment Layer                       │
│    SegmentWriter   SegmentReader   SegmentMerger    │
│                  MergePolicy                        │
│                  SegmentManager (snapshot / commit)  │
├──────────────────┬──────────────────────────────────┤
│    KeyMap        │         PostingList              │
│  (接口)          │         (接口)                    │
│  ┌────────────┐  │  ┌──────────────────────────┐    │
│  │ SortedBlock│  │  │ InlinePostings (≤N)      │    │
│  │ FSTKeyMap  │  │  │ RoaringPostings          │    │
│  └────────────┘  │  │ BitpackingPostings       │    │
│                  │  └──────────────────────────┘    │
├──────────────────┴──────────────────────────────────┤
│                 Storage Layer                        │
│    Allocator (接口)     IO (接口)                     │
│    BlockCache                                       │
├─────────────────────────────────────────────────────┤
│              Tokenizer (Rust FFI)                    │
└─────────────────────────────────────────────────────┘
```

# 五、Storage Layer

## 5.1 Allocator

读路径所有内存分配经过此接口，使上层可以追踪和控制内存用量。

- 初期实现 `MallocAllocator`（封装 malloc/free）
- 后续接入 Milvus 统一内存管理
- 写路径不受约束，直接用标准分配器

## 5.2 BlockCache

读路径的缓存层。缓存反序列化后的内存对象（非文件原始字节），通过 Allocator 分配内存。参考 ClickHouse TextIndexCache 的分级缓存设计。

三级缓存各自是全局单例 LRU cache，所有 segment、所有倒排索引的数据统一管理和淘汰。高频查询的数据自然常驻，冷数据自动淘汰，不需要按 segment 或 index 单独管理缓存容量。

### 三级缓存

| 缓存 | Key | Value | 特点 |
|------|-----|-------|------|
| **Postings Cache** | (segment, index, token) | PostingList（完整的 doc_id 集合） | 最终数据，命中即结束 |
| **Tokens Cache** | (segment, index, token) | TokenPostingsInfo（offset、cardinality） | posting 的元信息，指向 .pst 文件位置 |
| **Header Cache** | (segment, index) | DictionarySparseIndex（块边界 token + 块偏移量） | 整个索引的目录，小，基本常驻 |

### 查询路径

```
term 进来
  │
  ▼
1. 查 Postings Cache (segment, index, token)
   → 命中: 直接返回 doc_id 集合，结束
   → 未命中: 继续
  │
  ▼
2. 查 Tokens Cache (segment, index, token)
   → 命中: 拿到 TokenPostingsInfo（offset），跳到 Step 4
   → 未命中: 继续
  │
  ▼
3. 查 Header Cache (segment, index)
   → 命中: 拿到 DictionarySparseIndex
   → 未命中: 从 .idx 文件读取，放入 Header Cache
   → upperBound(token) 定位字典块
   → 从 .dct 文件加载字典块，块内二分查找
   → 拿到 TokenPostingsInfo，放入 Tokens Cache
  │
  ▼
4. 根据 TokenPostingsInfo.offset 从 .pst 文件读取 posting list
   → 放入 Postings Cache
   → 返回 doc_id 集合
```

### 与 ClickHouse 的差异

ClickHouse 按 granule 范围分块缓存 posting list（key 含 block_index），因为查询可以只关注部分 row 范围。Milvus 查询始终需要完整的 doc_id 集合，因此 posting list 整体缓存，Postings Cache 可以提到最前面作为第一级快捷路径。

ClickHouse 的查询顺序是 Tokens Cache → Header Cache → Postings Cache，因为 Postings Cache 按 block 粒度缓存，必须先拿到 TokenPostingsInfo 才知道读哪些 block。

### 内存和性能考量
缓存解压出来的数据会带来额外的内存压力（目前 tantivy 通过 mmap 在内存中保留的是 bitpacking 后的内容），考虑到直接缓存解压后的 posting 会带来性能提升（特别是当 doc id 足够多时，我们可以直接缓存 bitset），我们可以基于配置或者数据情况（doc id 足够多时，bitset 可能比 bitpacking 小）决定缓存的是解压前的还是之后的数据。

## 5.3 Cell-Based Lazy Loading

### 问题

当前索引加载时会全量从 S3 拉取所有文件再加载到内存，对于大索引这会导致加载时间过长。Milvus 已有 `CacheSlot<CellT>` / `Translator<CellT>` 缓存框架（见 `cachinglayer/`），列存索引通过该框架实现了按需加载——将数据切分为 cell，每个 cell 是缓存管理的最小单元，查询时只加载被访问到的 cell，冷 cell 可被 LRU 淘汰。倒排索引齐这一机制。

### 核心思路

索引数据按 ~2MB 切分为 chunk，每个 chunk 对应一个 cell。切分对齐到数据的自然边界（字典块边界、posting list 边界），不会切断任何一个完整的数据单元。索引的编码格式不变，只额外存储一份 ChunkManifest 记录各 chunk 的偏移和归属关系。

### Cell 布局

所有 cell 统一编号，按数据类型分段：

- **Cell 0**（常驻）：稀疏索引 + ChunkManifest。包含查询所需的全部路由信息，加载后即可确定任意 term 所在的 cell
- **字典 chunk**：对齐到字典块边界，每个 chunk 包含若干完整的字典块
- **Posting chunk**：对齐到 posting list 边界，每个 chunk 包含若干完整的 posting 数据

### ChunkManifest

记录每个 chunk 在原始数据中的偏移和归属关系。通过 manifest，可以在 O(1) 时间内将字典块 index 或 posting offset 映射到对应的 cell id。

### 查询路径

单次 term lookup 最多触发 2 次远程读取（一次字典 chunk，一次 posting chunk）。热数据命中缓存后为零 I/O。

```
term 进来
  → 稀疏索引 (Cell 0, 常驻) 定位字典块
  → ChunkManifest 映射到字典 cell → PinCells → 缓存命中或远程加载
  → 解析字典块，拿到 PostingsInfo
  → 若非 inline，ChunkManifest 映射到 posting cell → PinCells → 加载
  → 解码返回 doc_id 列表
```

### 与 CacheSlot 框架的适配

倒排索引实现 Milvus 的 `Translator<CellT>` 接口，cell 是统一的 chunk wrapper，不区分数据类型。CacheSlot 框架负责 LRU 淘汰和内存统计，倒排索引只需提供 cell 数量、大小估算、按 cell id 范围加载数据等基本实现。Warmup 策略采用按访问加载，不预加载。

### 与 5.2 BlockCache 的关系

两层缓存正交互补，职责不同：

- **Cell 层（存储缓存）**：管理原始字节的按需加载与淘汰，解决"数据从 S3 到本地"
- **BlockCache 层（应用缓存）**：缓存解码后的可用对象，解决"数据从压缩格式到查询可用"

BlockCache 缓存的内容形态取决于 5.2 中的内存和性能考量——可能是解压后的 doc_id 数组、bitset，也可能是压缩态的原始字节，由配置或数据特征决定。

查询路径上，先查 BlockCache——命中则直接返回可用数据；未命中时通过 CacheSlot 获取原始字节（可能命中本地缓存，也可能触发远程加载），解码后放入 BlockCache。

# 六、KeyMap + PostingList

## 6.1 KeyMap 实现

将 term dictionary 抽象为接口，支持多种实现，基于配置或数据特征选择。

| 实现 | 特点 | 适用场景 |
|------|------|----------|
| `SortedBlockKeyMap` | SSTable，稀疏索引 + 字典块 + 块内有序 | 简单查询（exact / range） |
| `FSTKeyMap` | 分块 FST，prefix/regex 高效 | prefix/regex 密集场景 |


## 6.2 SortedBlockKeyMap（SSTable）磁盘格式

稀疏索引 + 字典块模式：

```
.idx 文件（稀疏索引）:
┌──────────────────────────────────┐
│  num_blocks: VarUInt             │
│  block_0_first_token: String     │
│  block_0_offset: UInt64          │
│  block_1_first_token: String     │
│  block_1_offset: UInt64          │
│  ...                             │
└──────────────────────────────────┘

.dct 文件（字典块）:
┌──────────────────────────────────┐
│  Block 0:                        │
│    num_tokens: VarUInt           │
│    tokens: [String × N]         │
│    postings_infos: [PostingsInfo × N] │
├──────────────────────────────────┤
│  Block 1: ...                    │
├──────────────────────────────────┤
│  ...                             │
└──────────────────────────────────┘
```

查询路径：稀疏索引二分 → 定位字典块 → 加载字典块（缓存） → 块内二分 → PostingsInfo。

## 6.3 PostingsInfo

一个 term 的 posting list 元信息，描述去哪里找实际的 doc id 序列。

```cpp
struct PostingsInfo {
    uint32_t cardinality;           // doc 数量
    PostingEncoding encoding;       // inline / roaring / bitpacking

    // 小 posting 直接内嵌（零额外 I/O）
    static constexpr size_t kInlineMax = 6;
    std::array<uint32_t, kInlineMax> inline_docs;

    // 大 posting：文件偏移 + 长度
    uint64_t file_offset;           // .pst 文件偏移
    uint32_t data_size;             // 压缩数据长度
};
```

## 6.4 PostingList 编码

PostingList 编码抽象为接口，不同编码实现可在同一 segment 内共存——写入时根据 data pattern 选择最优编码，读取时根据 header 标识选对应 decoder。

```cpp
class PostingListCodec {
public:
    virtual void encode(const uint32_t* doc_ids, size_t count, FileWriter* writer) = 0;
    virtual void decode(FileReader* reader, uint64_t offset, size_t len, Bitset* output) = 0;
    virtual ~PostingListCodec() = default;
};
```

编码选择策略（基于 data pattern 自动决定）：

| 编码 | 特点 | 适用场景 |
|------|------|----------|
| Inline | 直接嵌入 PostingsInfo，零堆分配 | cardinality ≤ 6 |
| Bitpacking | delta + 128-doc block SIMD bitpacking，压缩率高 | doc_id 间距均匀，delta 范围小 |
| Roaring Bitmap | 自适应容器（array/bitset/run），集合运算原生支持 | doc_id 分布不均匀，或密集区间多 |

同一 segment 内不同 term 可以使用不同编码。PostingsInfo.encoding 字段标识该 term 使用的编码类型。

Milvus 查询场景是全 segment 匹配（无 row range 过滤），因此 posting list 不做分块——每个 term 的 posting list 是一段连续的压缩数据，读取时全量解压。缓存和淘汰的粒度也是整个 term 的 posting list。

## 6.5 Position 存储

Phrase query 需要 term 在文档中的位置信息。

```cpp
struct PostingsInfoWithPositions : PostingsInfo {
    uint64_t pos_file_offset;       // .pos 文件偏移
    uint32_t pos_data_size;         // position 数据长度
};
```

Position 编码：delta encoding（同一文档内 position 递增），bitpacking 或 VarInt。与 posting list 同理，不做分块，全量读取解压。

# 七、查询模型与功能支撑

## 7.1 KeyMap × PostingList × 类型系统

查询能力由三个维度共同决定：

**PostingList 决定存什么，约束查询的表达力**：

| PostingList 存储内容 | 支持的查询 |
|---------------------|-----------|
| doc_id only | term query, bool query |
| doc_id + position | 上述 + phrase match |

**KeyMap 决定怎么找 term，约束查询的效率**：

| 查询类型 | SortedBlockKeyMap | FSTKeyMap |
|---------|:-:|:-:|
| exact match | ✓ 二分查找 | ✓ |
| range (>=, <=) | ✓ 有序遍历 | ✓ 但不如前者自然 |
| prefix | ✓ 等价于 range 扫描 | ✓ automaton |
| regex | ✗ 需扫描 | ✓ automaton |
| fuzzy (edit distance) | ✗ 需扫描 | ✓ Levenshtein automaton |

**KeyMap 不感知类型**：key 统一是 `bytes`。类型语义由上层的 Encoder/Tokenizer 层处理：

```
写入：typed value → Encoder → bytes → KeyMap → PostingList
查询：typed query → Encoder → bytes query → KeyMap lookup → PostingList decode
```

唯一要求：**编码后的 bytes 必须保序**（原始值的大小关系在字节序上保持一致），这样 KeyMap 的有序遍历才能正确支持 range query。

**Geo / Facet 等结构化类型**是编码层的事：
- Geo：经纬度 → geohash 编码为多精度 term，查询转为 prefix/range query
- Facet：层级路径 `"/electronics/phones"` 拆成多级 term，查询转为 prefix query

索引层始终只做 `bytes → posting list`，不感知上层类型。这是 tantivy / Lucene 的通用做法。

## 7.2 功能需求与索引层支撑

上层功能需求如何映射到 7.1 的底层能力。各功能的详细设计在后续讨论中逐步完善。

### 7.2.1 模糊搜索（纠错）

FSTKeyMap + Levenshtein automaton。7.1 KeyMap 能力表已覆盖。

### 7.2.2 PhrasePrefixQuery / 补全

**查询思路**：对于输入 `"wordA wordB prefixC"`，前面的 term 做 exact lookup，最后一个 term 做 prefix scan 展开为多个候选 term。每个候选 term 分别与前面的 exact term 组成完整短语做 phrase match（检查 position 连续性），命中任何一个候选即算匹配。

**查询 vs 展示**：查询本身（"哪些文档匹配"）只需返回 doc_id 集合。但结果展示（"匹配在原文什么位置"）需要知道命中的字节偏移（offset），否则 caller 只能拿 doc_id 回表重新分词匹配。这与 7.2.3 高亮共享同一个 trade-off：PostingList 是否需要存储 offset。

### 7.2.3 高亮

高亮需要定位匹配 term 在原文中的字节偏移（offset）。获取 offset 有两种方式：

- **查询时重新分词**：拿到 top-k doc_id 后反查原文，重新分词 + 匹配 query 定位偏移。不需要索引层额外存储。
- **索引中预存 offset**：PostingList 额外存储每次出现的字节偏移，查询时直接返回。代价是 PostingList 膨胀，且所有查询都要承担额外 I/O。

ES/Lucene 的 UnifiedHighlighter 支持三级 offset 来源：POSTINGS（`index_options: offsets`）→ TERM_VECTORS（`term_vector: with_positions_offsets`）→ ANALYSIS（重新分词），均需显式配置，默认走 ANALYSIS。

**决策**：不在 PostingList 中存 offset。高亮只针对 top-k 结果（10~100 条），重新分词开销可控。

### 7.2.4 Facet（GroupBy）

Facet 是对搜索结果的分类计数。例如搜索"笔记本电脑"返回 500 条结果，Facet 同时告诉用户：品牌维度下联想 120 条、华为 95 条、苹果 80 条。用户点击某个分类值可追加过滤条件收窄结果。本质是对查询结果按某个字段做 group by + count。

有两条执行路径：

- **列存路径**：拿到 doc_id 集合后，逐个查列存取字段值，group by count。结果集小时很快。
- **倒排路径**：编码层把层级路径拆成多级 term 写入倒排（如 `/electronics/phones` 拆为 `/electronics` 和 `/electronics/phones` 两个 term），聚合时对目标前缀做 prefix scan，每个 term 的 posting list 与查询结果做交集计数。结果集大时比列存遍历更快。

倒排层面的支撑：编码层处理多级 term 拆分，索引层只做普通的 term 写入 + prefix query，无需特殊结构。


# 八、Segment 与 Index API

整体设计保留 tantivy 的架构：多 segment + merge + snapshot 隔离。以下列出关键结构和与 tantivy 的差异点。

## 8.1 磁盘文件结构

```
segment_{id}/
├── .dct    ← 字典（KeyMap 序列化：token + PostingsInfo）
├── .idx    ← 稀疏索引（字典块的二级索引）
├── .pst    ← Posting List（压缩的 doc id 序列）
├── .pos    ← Positions（phrase query 用）
└── .meta   ← segment 元信息（doc count, min/max doc_id, 创建时间等）
```

## 8.2 写入路径

**SegmentWriter 内存数据结构**（移植自 tantivy）：

- **ArenaHashMap**：线性探测哈希表，term → Recorder 映射，key/value 均 inline 存储在 MemoryArena 中
- **MemoryArena**：1MB 分页，32-bit 地址（12 bit page_id + 20 bit offset），最大 4GB per worker
- **ExpUnrolledLinkedList**：指数增长块链表（8B → 16B → ... → 32KB 封顶），存储 delta-encoded vint 的 doc_id 和 position
- 所有分配在 MemoryArena 上完成，零堆分配

**IndexWriter 并发模型**：

- 前台 `add()` 单线程调用，保证 doc_id 严格递增
- 后台多 worker 各自持有独立 SegmentWriter，从 bounded channel 消费并构建 segment
- commit 时关闭 channel → worker flush → publish snapshot → 重建 worker

## 8.3 读取路径

**SegmentReader**：多线程安全，所有数据访问经过 BlockCache（见 5.2）。

**IndexReader**：持有 snapshot，遍历所有 segment 执行查询，合并 bitset 返回。reload() 获取最新 snapshot。

## 8.4 Query 抽象

参考 tantivy 的 Query → Weight → Collector 三层抽象。与 tantivy 的差异：去掉 Scorer，不做 BM25 评分。

- **Query**：查询描述，与 segment 无关。具体类型：TermQuery、BooleanQuery、PhraseQuery、PrefixQuery、RegexQuery 等，每种是 Query 子类，新增查询不改接口。
- **Weight**：Query 绑定到具体 segment 后的执行体。沿用业界标准命名（Lucene / tantivy），本设计中不用于 scoring。
- **Collector**：决定如何收集结果。Bitset 是默认收集策略，但 Facet 聚合（7.2.4）需要 FacetCollector。

## 8.5 Merge

- **SegmentMerger**：K-路归并排序，按 token 字典序归并，相同 token 的 posting list 做 union（doc_id 外部指定，不需要重映射）
- **MergePolicy**：可插拔接口，默认 LogMergePolicy（类似 tantivy / Lucene），按 segment 大小分层，同层达到阈值时触发合并。也可强制全部合并（sealed segment 最后触发）。
- **SegmentManager**：管理 segment 集合的可见性，snapshot 隔离 reader/writer。writer commit 后原子切换，旧 reader 不受影响


# 九、Phase 规划

## Phase 1 — tantivy 核心移植 + Milvus 集成

从 tantivy Rust → C++ 移植完整的写入、读取、多 segment、merge 框架（不含 position/phrase），并完成最简 Milvus 集成，端到端跑通 TermQuery。

- Allocator 接口 + MallocAllocator
- IO 接口 + 本地文件实现（pread/write）
- ArenaHashMap + MemoryArena + ExpUnrolledLinkedList（写入缓冲）
- SortedBlockKeyMap（SSTable 磁盘格式）
- PostingList 编码（inline + bitpacking）
- BlockCache 接口 + PassthroughCache（不缓存，直接读盘；预留接口供 Phase 2 替换为真正缓存实现）
- SegmentWriter / SegmentReader
- 多 segment + SegmentManager（snapshot）
- MergePolicy + SegmentMerger
- commit / reload 语义
- TermQuery / BoolQuery
- IndexWriter / IndexReader
- Tokenizer FFI 桥接
- 与 Milvus 集成（ScalarIndex 接口、growing / sealed segment）

## Phase 2 — Phrase + 缓存优化

- Position 存储和读取（独立 `.pos` 文件）
- PhraseQuery（带 slop）
- BlockCache 实现（应用级缓存，缓存形态由配置或数据特征决定）
- Cell-Based Lazy Loading（Translator 适配，ChunkManifest，按需从 S3 加载）

## Phase 3 — 高级查询

- FSTKeyMap 实现
- PrefixQuery / RegexQuery / FuzzyQuery
- PhrasePrefixQuery
- Facet（编码层 + FacetCollector）
