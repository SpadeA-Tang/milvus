package model

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/indexpb"
)

var (
	nestedIndexCollID int64 = 1

	nestedIndexModel = &NestedIndex{
		CollectionID:  nestedIndexCollID,
		StructFieldID: 100,
		SubFieldIDs:   []int64{101, 102, 103},
		IndexName:     "nested_idx",
		IndexID:       1,
		IndexedRows:   1000,
		TotalRows:     2000,
		State:         commonpb.IndexState_Finished,
		FailReason:    "",
		IsDeleted:     false,
		CreateTime:    1,
	}

	nestedIndexPb = &indexpb.NestedIndex{
		IndexInfo: &indexpb.NestedIndexInfo{
			CollectionID:  nestedIndexCollID,
			StructFieldID: 100,
			SubFieldIDs:   []int64{101, 102, 103},
			IndexName:     "nested_idx",
			IndexID:       1,
			IndexedRows:   1000,
			TotalRows:     2000,
			State:         commonpb.IndexState_Finished,
			FailReason:    "",
		},
		Deleted:    false,
		CreateTime: 1,
	}
)

func TestMarshalNestedIndexModel(t *testing.T) {
	ret := MarshalNestedIndexModel(nestedIndexModel)
	assert.Equal(t, nestedIndexPb.IndexInfo.IndexID, ret.IndexInfo.IndexID)
	assert.Equal(t, nestedIndexPb.IndexInfo.CollectionID, ret.IndexInfo.CollectionID)
	assert.Equal(t, nestedIndexPb.IndexInfo.StructFieldID, ret.IndexInfo.StructFieldID)
	assert.Equal(t, nestedIndexPb.IndexInfo.SubFieldIDs, ret.IndexInfo.SubFieldIDs)
	assert.Equal(t, nestedIndexPb.IndexInfo.IndexName, ret.IndexInfo.IndexName)
	assert.Equal(t, nestedIndexPb.IndexInfo.IndexedRows, ret.IndexInfo.IndexedRows)
	assert.Equal(t, nestedIndexPb.IndexInfo.TotalRows, ret.IndexInfo.TotalRows)
	assert.Equal(t, nestedIndexPb.IndexInfo.State, ret.IndexInfo.State)
	assert.Equal(t, nestedIndexPb.Deleted, ret.Deleted)
	assert.Equal(t, nestedIndexPb.CreateTime, ret.CreateTime)
	assert.Nil(t, MarshalNestedIndexModel(nil))
}

func TestUnmarshalNestedIndexModel(t *testing.T) {
	ret := UnmarshalNestedIndexModel(nestedIndexPb)
	assert.Equal(t, nestedIndexModel.IndexID, ret.IndexID)
	assert.Equal(t, nestedIndexModel.CollectionID, ret.CollectionID)
	assert.Equal(t, nestedIndexModel.StructFieldID, ret.StructFieldID)
	assert.Equal(t, nestedIndexModel.SubFieldIDs, ret.SubFieldIDs)
	assert.Equal(t, nestedIndexModel.IndexName, ret.IndexName)
	assert.Equal(t, nestedIndexModel.IndexedRows, ret.IndexedRows)
	assert.Equal(t, nestedIndexModel.TotalRows, ret.TotalRows)
	assert.Equal(t, nestedIndexModel.State, ret.State)
	assert.Equal(t, nestedIndexModel.IsDeleted, ret.IsDeleted)
	assert.Equal(t, nestedIndexModel.CreateTime, ret.CreateTime)
	assert.Nil(t, UnmarshalNestedIndexModel(nil))
}

func TestCloneNestedIndex(t *testing.T) {
	ret := CloneNestedIndex(nestedIndexModel)
	assert.Equal(t, nestedIndexModel.IndexID, ret.IndexID)
	assert.Equal(t, nestedIndexModel.CollectionID, ret.CollectionID)
	assert.Equal(t, nestedIndexModel.StructFieldID, ret.StructFieldID)
	assert.Equal(t, nestedIndexModel.SubFieldIDs, ret.SubFieldIDs)
	assert.Equal(t, nestedIndexModel.IndexName, ret.IndexName)
	assert.Equal(t, nestedIndexModel.IndexedRows, ret.IndexedRows)
	assert.Equal(t, nestedIndexModel.TotalRows, ret.TotalRows)
	assert.Equal(t, nestedIndexModel.State, ret.State)
	assert.Equal(t, nestedIndexModel.FailReason, ret.FailReason)
	assert.Equal(t, nestedIndexModel.IsDeleted, ret.IsDeleted)
	assert.Equal(t, nestedIndexModel.CreateTime, ret.CreateTime)

	// Verify it's a deep copy - modifying clone should not affect original
	ret.SubFieldIDs[0] = 999
	assert.NotEqual(t, nestedIndexModel.SubFieldIDs[0], ret.SubFieldIDs[0])

	assert.Nil(t, CloneNestedIndex(nil))
}
