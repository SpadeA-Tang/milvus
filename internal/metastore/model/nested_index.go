package model

import (
	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/indexpb"
)

type NestedIndex struct {
	CollectionID         int64
	StructFieldID        int64
	SubFieldIDs          []int64
	IndexName            string
	IndexID              int64
	IndexedRows          int64
	TotalRows            int64
	State                commonpb.IndexState
	FailReason           string
	IndexStateFailReason string

	IsDeleted  bool
	CreateTime uint64
}

func MarshalNestedIndexModel(index *NestedIndex) *indexpb.NestedIndex {
	if index == nil {
		return nil
	}

	return &indexpb.NestedIndex{
		IndexInfo: &indexpb.NestedIndexInfo{
			CollectionID:  index.CollectionID,
			StructFieldID: index.StructFieldID,
			SubFieldIDs:   index.SubFieldIDs,
			IndexName:     index.IndexName,
			IndexID:       index.IndexID,
			IndexedRows:   index.IndexedRows,
			TotalRows:     index.TotalRows,
			State:         index.State,
			FailReason:    index.FailReason,
		},
		Deleted:    index.IsDeleted,
		CreateTime: index.CreateTime,
	}
}

func UnmarshalNestedIndexModel(indexInfo *indexpb.NestedIndex) *NestedIndex {
	if indexInfo == nil {
		return nil
	}

	return &NestedIndex{
		CollectionID:  indexInfo.GetIndexInfo().GetCollectionID(),
		StructFieldID: indexInfo.GetIndexInfo().GetStructFieldID(),
		SubFieldIDs:   indexInfo.GetIndexInfo().GetSubFieldIDs(),
		IndexName:     indexInfo.GetIndexInfo().GetIndexName(),
		IndexID:       indexInfo.GetIndexInfo().GetIndexID(),
		IndexedRows:   indexInfo.GetIndexInfo().GetIndexedRows(),
		TotalRows:     indexInfo.GetIndexInfo().GetTotalRows(),
		State:         indexInfo.GetIndexInfo().GetState(),
		FailReason:    indexInfo.GetIndexInfo().GetFailReason(),
		IsDeleted:     indexInfo.GetDeleted(),
		CreateTime:    indexInfo.GetCreateTime(),
	}
}

func CloneNestedIndex(index *NestedIndex) *NestedIndex {
	if index == nil {
		return nil
	}

	clonedIndex := &NestedIndex{
		CollectionID:  index.CollectionID,
		StructFieldID: index.StructFieldID,
		SubFieldIDs:   make([]int64, len(index.SubFieldIDs)),
		IndexName:     index.IndexName,
		IndexID:       index.IndexID,
		IndexedRows:   index.IndexedRows,
		TotalRows:     index.TotalRows,
		State:         index.State,
		FailReason:    index.FailReason,
		IsDeleted:     index.IsDeleted,
		CreateTime:    index.CreateTime,
	}
	copy(clonedIndex.SubFieldIDs, index.SubFieldIDs)
	return clonedIndex
}
