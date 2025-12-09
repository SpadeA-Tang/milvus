// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package proxy

import (
	"context"
	"testing"

	"github.com/cockroachdb/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

func TestCreateNestedIndexTask_PreExecute(t *testing.T) {
	collectionName := "test_collection"
	collectionID := UniqueID(100)
	structFieldName := "structField"
	subFieldName := "structField.subField1"

	paramtable.Init()
	ctx := context.Background()

	t.Run("get collection id error", func(t *testing.T) {
		mockCache := NewMockCache(t)
		mockCache.On("GetCollectionID",
			mock.Anything,
			mock.AnythingOfType("string"),
			mock.AnythingOfType("string"),
		).Return(UniqueID(0), errors.New("mock error"))
		globalMetaCache = mockCache

		task := &createNestedIndexTask{
			ctx: ctx,
			req: &milvuspb.CreateNestedIndexRequest{
				Base:            &commonpb.MsgBase{},
				CollectionName:  collectionName,
				StructFieldName: structFieldName,
				SubFieldNames:   []string{subFieldName},
				IndexName:       "nested_idx",
			},
		}

		err := task.PreExecute(ctx)
		assert.Error(t, err)
	})

	t.Run("empty sub field names", func(t *testing.T) {
		mockCache := NewMockCache(t)
		mockCache.On("GetCollectionID",
			mock.Anything,
			mock.AnythingOfType("string"),
			mock.AnythingOfType("string"),
		).Return(collectionID, nil)
		globalMetaCache = mockCache

		task := &createNestedIndexTask{
			ctx: ctx,
			req: &milvuspb.CreateNestedIndexRequest{
				Base:            &commonpb.MsgBase{},
				CollectionName:  collectionName,
				StructFieldName: structFieldName,
				SubFieldNames:   []string{}, // empty
				IndexName:       "nested_idx",
			},
		}

		err := task.PreExecute(ctx)
		assert.Error(t, err)
	})

	t.Run("invalid index name", func(t *testing.T) {
		mockCache := NewMockCache(t)
		mockCache.On("GetCollectionID",
			mock.Anything,
			mock.AnythingOfType("string"),
			mock.AnythingOfType("string"),
		).Return(collectionID, nil)
		globalMetaCache = mockCache

		task := &createNestedIndexTask{
			ctx: ctx,
			req: &milvuspb.CreateNestedIndexRequest{
				Base:            &commonpb.MsgBase{},
				CollectionName:  collectionName,
				StructFieldName: structFieldName,
				SubFieldNames:   []string{subFieldName},
				IndexName:       "$invalid", // invalid name starts with $
			},
		}

		err := task.PreExecute(ctx)
		assert.Error(t, err)
	})
}

func TestCreateNestedIndexTask_Methods(t *testing.T) {
	paramtable.Init()
	ctx := context.Background()

	task := &createNestedIndexTask{
		baseTask: baseTask{},
		ctx:      ctx,
		req: &milvuspb.CreateNestedIndexRequest{
			Base: &commonpb.MsgBase{
				MsgID:     100,
				Timestamp: 1000,
			},
			CollectionName:  "test_collection",
			StructFieldName: "structField",
			SubFieldNames:   []string{"structField.subField1"},
			IndexName:       "nested_idx",
		},
	}

	t.Run("Name", func(t *testing.T) {
		assert.Equal(t, CreateNestedIndexTaskName, task.Name())
	})

	t.Run("ID and SetID", func(t *testing.T) {
		assert.Equal(t, int64(100), task.ID())
		task.SetID(200)
		assert.Equal(t, int64(200), task.ID())
	})

	t.Run("BeginTs and EndTs", func(t *testing.T) {
		assert.Equal(t, Timestamp(1000), task.BeginTs())
		assert.Equal(t, Timestamp(1000), task.EndTs())
	})

	t.Run("SetTs", func(t *testing.T) {
		task.SetTs(2000)
		assert.Equal(t, Timestamp(2000), task.BeginTs())
	})

	t.Run("Type", func(t *testing.T) {
		task.req.Base.MsgType = commonpb.MsgType_CreateIndex
		assert.Equal(t, commonpb.MsgType_CreateIndex, task.Type())
	})

	t.Run("TraceCtx", func(t *testing.T) {
		assert.Equal(t, ctx, task.TraceCtx())
	})

	t.Run("OnEnqueue", func(t *testing.T) {
		task.req.Base = nil
		err := task.OnEnqueue()
		assert.NoError(t, err)
		assert.NotNil(t, task.req.Base)
		assert.Equal(t, commonpb.MsgType_CreateIndex, task.req.Base.MsgType)
	})

	t.Run("PostExecute", func(t *testing.T) {
		err := task.PostExecute(ctx)
		assert.NoError(t, err)
	})
}
