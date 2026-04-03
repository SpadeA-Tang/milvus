#include <gtest/gtest.h>

#include "index/inverted/indexer/SegmentManager.h"
#include "index/inverted/indexer/MergePolicy.h"
#include "index/inverted/indexer/LogMergePolicy.h"

using namespace milvus::index::inverted;

// --- SegmentEntry ---

TEST(SegmentEntryTest, Basic) {
    SegmentMeta meta;
    meta.segment_id = SegmentId::generate_random();
    meta.num_docs = 100;
    meta.max_doc_id = 99;

    SegmentEntry entry(meta);
    EXPECT_EQ(entry.segment_id(), meta.segment_id);
    EXPECT_EQ(entry.meta().num_docs, 100u);
}

// --- SegmentRegister ---

TEST(SegmentRegisterTest, AddAndRemove) {
    SegmentRegister reg;
    auto id_a = SegmentId::generate_random();
    auto id_b = SegmentId::generate_random();

    SegmentMeta meta_a;
    meta_a.segment_id = id_a;
    meta_a.num_docs = 10;
    reg.add_segment_entry(SegmentEntry(meta_a));

    SegmentMeta meta_b;
    meta_b.segment_id = id_b;
    meta_b.num_docs = 20;
    reg.add_segment_entry(SegmentEntry(meta_b));

    EXPECT_EQ(reg.size(), 2u);
    EXPECT_TRUE(reg.contains_all({id_a, id_b}));

    reg.remove_segment(id_a);
    EXPECT_EQ(reg.size(), 1u);
    EXPECT_FALSE(reg.get(id_a).has_value());
    EXPECT_TRUE(reg.get(id_b).has_value());
}

TEST(SegmentRegisterTest, GetMergeableSegments) {
    SegmentRegister reg;
    auto id_a = SegmentId::generate_random();
    auto id_b = SegmentId::generate_random();
    auto id_c = SegmentId::generate_random();

    for (auto id : {id_a, id_b, id_c}) {
        SegmentMeta m;
        m.segment_id = id;
        m.num_docs = 10;
        reg.add_segment_entry(SegmentEntry(m));
    }

    std::unordered_set<SegmentId, SegmentIdHash> in_merge = {id_b};
    auto mergeable = reg.get_mergeable_segments(in_merge);
    EXPECT_EQ(mergeable.size(), 2u);
    for (const auto& m : mergeable) {
        EXPECT_NE(m.id(), id_b);
    }
}

// --- SegmentManager ---

TEST(SegmentManagerTest, CommitAndMerge) {
    SegmentManager mgr;

    // Add two uncommitted segments.
    SegmentMeta m1;
    m1.segment_id = SegmentId::generate_random();
    m1.num_docs = 100;
    mgr.add_segment(SegmentEntry(m1));

    SegmentMeta m2;
    m2.segment_id = SegmentId::generate_random();
    m2.num_docs = 200;
    mgr.add_segment(SegmentEntry(m2));

    // Commit moves uncommitted to committed.
    auto entries = mgr.segment_entries();
    EXPECT_EQ(entries.size(), 2u);
    mgr.commit(std::move(entries));

    auto committed = mgr.committed_segment_metas();
    EXPECT_EQ(committed.size(), 2u);

    // Start merge.
    std::vector<SegmentId> merge_ids = {m1.segment_id, m2.segment_id};
    auto merge_entries = mgr.start_merge(merge_ids);
    ASSERT_TRUE(merge_entries.has_value());
    EXPECT_EQ(merge_entries->size(), 2u);

    // End merge: replace with merged segment.
    SegmentMeta merged_meta;
    merged_meta.segment_id = SegmentId::generate_random();
    merged_meta.num_docs = 300;
    auto status = mgr.end_merge(merge_ids, SegmentEntry(merged_meta));
    ASSERT_TRUE(status.has_value());
    EXPECT_EQ(*status, SegmentsStatus::Committed);

    committed = mgr.committed_segment_metas();
    EXPECT_EQ(committed.size(), 1u);
    EXPECT_EQ(committed[0].num_docs, 300u);
}

// --- MergePolicy ---

TEST(NoMergePolicyTest, NeverMerges) {
    NoMergePolicy policy;
    std::vector<SegmentMeta> segments(5);
    auto candidates = policy.compute_merge_candidates(segments);
    EXPECT_TRUE(candidates.empty());
}

// --- LogMergePolicy ---

TEST(LogMergePolicyTest, Empty) {
    LogMergePolicy policy;
    std::vector<SegmentMeta> empty;
    auto result = policy.compute_merge_candidates(empty);
    EXPECT_TRUE(result.empty());
}

TEST(LogMergePolicyTest, SameLevel) {
    LogMergePolicy policy;
    policy.set_min_num_segments(3);
    policy.set_max_docs_before_merge(100000);
    policy.set_min_layer_size(2);

    std::vector<SegmentMeta> segments;
    for (int i = 0; i < 3; i++) {
        SegmentMeta m;
        m.segment_id = SegmentId::generate_random();
        m.num_docs = 10;
        segments.push_back(m);
    }
    auto result = policy.compute_merge_candidates(segments);
    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].segment_ids.size(), 3u);
}

TEST(LogMergePolicyTest, MultipleLevels) {
    LogMergePolicy policy;
    policy.set_min_num_segments(3);
    policy.set_max_docs_before_merge(100000);
    policy.set_min_layer_size(2);

    std::vector<SegmentMeta> segments;
    // 6 small segments + 3 medium + 2 large (not enough for merge)
    for (uint32_t n : {10u, 10u, 10u, 10u, 10u, 10u,
                       1000u, 1000u, 1000u,
                       10000u, 10000u}) {
        SegmentMeta m;
        m.segment_id = SegmentId::generate_random();
        m.num_docs = n;
        segments.push_back(m);
    }
    auto result = policy.compute_merge_candidates(segments);
    EXPECT_EQ(result.size(), 2u);
}

TEST(LogMergePolicyTest, AllTooLarge) {
    LogMergePolicy policy;
    policy.set_max_docs_before_merge(100000);

    std::vector<SegmentMeta> segments;
    for (int i = 0; i < 8; i++) {
        SegmentMeta m;
        m.segment_id = SegmentId::generate_random();
        m.num_docs = 100001;
        segments.push_back(m);
    }
    auto result = policy.compute_merge_candidates(segments);
    EXPECT_TRUE(result.empty());
}
