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
	"strings"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus/internal/types"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v2/util/commonpbutil"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
	"go.uber.org/zap"
)

const (
	CreateNestedIndexTaskName = "CreateNestedIndexTask"
)

// createNestedIndexTask handles creating nested index for struct array fields
type createNestedIndexTask struct {
	baseTask
	Condition
	req      *milvuspb.CreateNestedIndexRequest
	ctx      context.Context
	mixCoord types.MixCoordClient
	result   *commonpb.Status

	collectionID UniqueID

	subFieldIDs []int64
}

func (t *createNestedIndexTask) TraceCtx() context.Context {
	return t.ctx
}

func (t *createNestedIndexTask) ID() UniqueID {
	return t.req.GetBase().GetMsgID()
}

func (t *createNestedIndexTask) SetID(uid UniqueID) {
	t.req.GetBase().MsgID = uid
}

func (t *createNestedIndexTask) Name() string {
	return CreateNestedIndexTaskName
}

func (t *createNestedIndexTask) Type() commonpb.MsgType {
	return t.req.GetBase().GetMsgType()
}

func (t *createNestedIndexTask) BeginTs() Timestamp {
	return t.req.GetBase().GetTimestamp()
}

func (t *createNestedIndexTask) EndTs() Timestamp {
	return t.req.GetBase().GetTimestamp()
}

func (t *createNestedIndexTask) SetTs(ts Timestamp) {
	t.req.Base.Timestamp = ts
}

func (t *createNestedIndexTask) OnEnqueue() error {
	if t.req.Base == nil {
		t.req.Base = commonpbutil.NewMsgBase()
	}
	t.req.Base.MsgType = commonpb.MsgType_CreateIndex
	t.req.Base.SourceID = paramtable.GetNodeID()
	return nil
}

func (t *createNestedIndexTask) validateFields(ctx context.Context) error {
	subFieldNames := t.req.GetSubFieldNames()
	if len(subFieldNames) == 0 {
		return merr.WrapErrParameterInvalidMsg("sub_field_names cannot be empty")
	}

	schema, err := globalMetaCache.GetCollectionSchema(ctx, t.req.GetDbName(), t.req.GetCollectionName())
	if err != nil {
		return err
	}

	structFieldName := t.req.GetStructFieldName()
	structField := schema.schemaHelper.GetStructArrayFieldFromName(structFieldName)
	if structField == nil {
		return merr.WrapErrParameterInvalidMsg("struct field not found, name: %s", structFieldName)
	}

	subFields := typeutil.NewSet[string]()
	subFieldIDs := make([]int64, 0, len(subFieldNames))
	for _, subField := range subFieldNames {
		if subField == "" {
			return merr.WrapErrParameterInvalidMsg("empty sub field name found")
		}
		if subFields.Contain(subField) {
			return merr.WrapErrParameterInvalidMsg("duplicated sub field found: %s", subField)
		}
		subFields.Insert(subField)

		field, err := schema.schemaHelper.GetFieldFromName(subField)
		if err != nil {
			return err
		}

		// verify the sub field name is a sub field of the struct field
		if !strings.HasPrefix(field.Name, structFieldName) {
			return merr.WrapErrParameterInvalidMsg("sub field does not belong to struct field: struct_field=%s, sub_field=%s", structFieldName, subField)
		}

		subFieldIDs = append(subFieldIDs, field.GetFieldID())
	}
	t.subFieldIDs = subFieldIDs

	return nil
}

func (t *createNestedIndexTask) PreExecute(ctx context.Context) error {
	collName := t.req.GetCollectionName()

	collID, err := globalMetaCache.GetCollectionID(ctx, t.req.GetDbName(), collName)
	if err != nil {
		return err
	}
	t.collectionID = collID

	if err = validateIndexName(t.req.GetIndexName()); err != nil {
		return err
	}

	if err = t.validateFields(ctx); err != nil {
		return err
	}

	return nil
}

func (t *createNestedIndexTask) Execute(ctx context.Context) error {
	log.Ctx(ctx).Info("proxy create nested index", zap.Int64("collectionID", t.collectionID),
		zap.String("structFieldName", t.req.GetStructFieldName()),
		zap.Any("subFieldNames", t.req.GetSubFieldNames()),
		zap.Any("subFieldIDs", t.subFieldIDs),
		zap.String("indexName", t.req.GetIndexName()),
	)

	var err error
	req := &indexpb.CreateNestedIndexRequest{
		CollectionID:    t.collectionID,
		StructFieldName: t.req.GetStructFieldName(),
		SubFieldNames:   t.req.GetSubFieldNames(),
		SubFieldIds:     t.subFieldIDs,
		IndexName:       t.req.GetIndexName(),
		Timestamp:       t.BeginTs(),
	}
	t.result, err = t.mixCoord.CreateNestedIndex(ctx, req)
	if err = merr.CheckRPCCall(t.result, err); err != nil {
		return err
	}
	return nil
}

func (t *createNestedIndexTask) PostExecute(ctx context.Context) error {
	return nil
}
