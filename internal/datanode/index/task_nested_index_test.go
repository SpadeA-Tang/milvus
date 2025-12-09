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

package index

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/workerpb"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
)

type NestedIndexBuildTaskSuite struct {
	suite.Suite
	collectionID  int64
	partitionID   int64
	segmentID     int64
	structFieldID int64
	childFieldID  int64
	numRows       int64
}

func (suite *NestedIndexBuildTaskSuite) SetupSuite() {
	paramtable.Init()
	suite.collectionID = 1000
	suite.partitionID = 1001
	suite.segmentID = 1002
	suite.structFieldID = 200
	suite.childFieldID = 201
	suite.numRows = 100
}

func (suite *NestedIndexBuildTaskSuite) TestPreExecute_BuildDataPaths() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req := &workerpb.CreateNestedIndexJobRequest{
		ClusterID:       "test-cluster",
		BuildID:         1,
		CollectionID:    suite.collectionID,
		PartitionID:     suite.partitionID,
		SegmentID:       suite.segmentID,
		StructFieldId:   suite.structFieldID,
		StructFieldName: "structField",
		ChildFields: []*workerpb.ChildFieldInfo{
			{
				FieldId:   suite.childFieldID,
				FieldName: "childField1",
				FieldType: schemapb.DataType_Int64,
				DataIds:   []int64{10001, 10002},
				Field: &schemapb.FieldSchema{
					FieldID:  suite.childFieldID,
					Name:     "childField1",
					DataType: schemapb.DataType_Int64,
				},
			},
		},
		IndexVersion:              1,
		NumRows:                   suite.numRows,
		CurrentScalarIndexVersion: 0,
		StorageConfig: &indexpb.StorageConfig{
			RootPath:    "root",
			StorageType: "local",
		},
		IndexFilePrefix: "test/index",
	}

	manager := NewTaskManager(ctx)
	task := NewNestedIndexBuildTask(ctx, cancel, req, nil, manager, nil)

	// Test OnEnqueue
	err := task.OnEnqueue(ctx)
	suite.NoError(err)

	// Test PreExecute - this should build data paths from data_ids
	err = task.PreExecute(ctx)
	suite.NoError(err)

	// Verify data paths are built correctly
	suite.Equal(1, len(req.ChildFields))
	suite.Equal(2, len(req.ChildFields[0].DataPaths))
	suite.Contains(req.ChildFields[0].DataPaths[0], "root")
	suite.Contains(req.ChildFields[0].DataPaths[0], "1000") // collectionID
	suite.Contains(req.ChildFields[0].DataPaths[0], "1001") // partitionID
	suite.Contains(req.ChildFields[0].DataPaths[0], "1002") // segmentID
	suite.Contains(req.ChildFields[0].DataPaths[0], "201")  // fieldID
}

func (suite *NestedIndexBuildTaskSuite) TestPreExecute_SkipIfPathsExist() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create request with pre-filled DataPaths
	req := &workerpb.CreateNestedIndexJobRequest{
		ClusterID:       "test-cluster",
		BuildID:         2,
		CollectionID:    suite.collectionID,
		PartitionID:     suite.partitionID,
		SegmentID:       suite.segmentID,
		StructFieldId:   suite.structFieldID,
		StructFieldName: "structField",
		ChildFields: []*workerpb.ChildFieldInfo{
			{
				FieldId:   suite.childFieldID,
				FieldName: "childField1",
				FieldType: schemapb.DataType_Int64,
				DataIds:   []int64{10001},
				DataPaths: []string{"existing/path/file1"}, // Pre-filled paths
				Field: &schemapb.FieldSchema{
					FieldID:  suite.childFieldID,
					Name:     "childField1",
					DataType: schemapb.DataType_Int64,
				},
			},
		},
		IndexVersion: 1,
		NumRows:      suite.numRows,
		StorageConfig: &indexpb.StorageConfig{
			RootPath:    "root",
			StorageType: "local",
		},
	}

	manager := NewTaskManager(ctx)
	task := NewNestedIndexBuildTask(ctx, cancel, req, nil, manager, nil)

	err := task.PreExecute(ctx)
	suite.NoError(err)

	// Verify paths are not modified
	suite.Equal(1, len(req.ChildFields[0].DataPaths))
	suite.Equal("existing/path/file1", req.ChildFields[0].DataPaths[0])
}

func (suite *NestedIndexBuildTaskSuite) TestTaskMethods() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req := &workerpb.CreateNestedIndexJobRequest{
		ClusterID:       "test-cluster",
		BuildID:         3,
		CollectionID:    suite.collectionID,
		PartitionID:     suite.partitionID,
		SegmentID:       suite.segmentID,
		StructFieldId:   suite.structFieldID,
		StructFieldName: "structField",
		IndexVersion:    1,
		NumRows:         suite.numRows,
		TaskSlot:        2,
		StorageConfig: &indexpb.StorageConfig{
			RootPath:    "root",
			StorageType: "local",
		},
	}

	manager := NewTaskManager(ctx)
	task := NewNestedIndexBuildTask(ctx, cancel, req, nil, manager, nil)

	// Test Name()
	suite.Equal("test-cluster/3", task.Name())

	// Test Ctx()
	suite.Equal(ctx, task.Ctx())

	// Test GetSlot()
	suite.Equal(int64(2), task.GetSlot())

	// Test SetState and GetState
	task.SetState(indexpb.JobState_JobStateInProgress, "")
	suite.Equal(indexpb.JobState_JobStateInProgress, task.GetState())

	task.SetState(indexpb.JobState_JobStateFinished, "")
	suite.Equal(indexpb.JobState_JobStateFinished, task.GetState())

	task.SetState(indexpb.JobState_JobStateFailed, "test error")
	suite.Equal(indexpb.JobState_JobStateFailed, task.GetState())
}

func (suite *NestedIndexBuildTaskSuite) TestReset() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req := &workerpb.CreateNestedIndexJobRequest{
		ClusterID: "test-cluster",
		BuildID:   4,
	}

	manager := NewTaskManager(ctx)
	task := NewNestedIndexBuildTask(ctx, cancel, req, nil, manager, nil)

	// Reset should clear all fields
	task.Reset()

	suite.Equal("", task.ident)
	suite.Nil(task.cancel)
	suite.Nil(task.ctx)
	suite.Nil(task.cm)
	suite.Nil(task.req)
	suite.Nil(task.tr)
	suite.Nil(task.manager)
	suite.Nil(task.pluginContext)
}

func TestNestedIndexBuildTaskSuite(t *testing.T) {
	suite.Run(t, new(NestedIndexBuildTaskSuite))
}
