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
	"fmt"
	"strconv"
	"strings"
	"time"

	"go.uber.org/zap"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus/internal/datanode/util"
	"github.com/milvus-io/milvus/internal/storage"
	"github.com/milvus-io/milvus/internal/util/indexcgowrapper"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/metrics"
	"github.com/milvus-io/milvus/pkg/v2/proto/indexcgopb"
	"github.com/milvus-io/milvus/pkg/v2/proto/indexpb"
	"github.com/milvus-io/milvus/pkg/v2/proto/workerpb"
	"github.com/milvus-io/milvus/pkg/v2/util/metautil"
	"github.com/milvus-io/milvus/pkg/v2/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v2/util/timerecord"
)

var _ Task = (*nestedIndexBuildTask)(nil)

// nestedIndexBuildTask is used to build Tantivy-based nested index for struct array fields.
type nestedIndexBuildTask struct {
	ident  string
	cancel context.CancelFunc
	ctx    context.Context

	cm            storage.ChunkManager
	index         indexcgowrapper.CodecIndex
	req           *workerpb.CreateNestedIndexJobRequest
	tr            *timerecord.TimeRecorder
	queueDur      time.Duration
	manager       *TaskManager
	pluginContext *indexcgopb.StoragePluginContext
}

func NewNestedIndexBuildTask(
	ctx context.Context,
	cancel context.CancelFunc,
	req *workerpb.CreateNestedIndexJobRequest,
	cm storage.ChunkManager,
	manager *TaskManager,
	pluginContext *indexcgopb.StoragePluginContext,
) *nestedIndexBuildTask {
	return &nestedIndexBuildTask{
		ident:         fmt.Sprintf("%s/%d", req.GetClusterID(), req.GetBuildID()),
		cancel:        cancel,
		ctx:           ctx,
		cm:            cm,
		req:           req,
		tr:            timerecord.NewTimeRecorder(fmt.Sprintf("NestedIndexBuildID: %d, ClusterID: %s", req.GetBuildID(), req.GetClusterID())),
		manager:       manager,
		pluginContext: pluginContext,
	}
}

func (it *nestedIndexBuildTask) Ctx() context.Context {
	return it.ctx
}

func (it *nestedIndexBuildTask) Name() string {
	return it.ident
}

func (it *nestedIndexBuildTask) SetState(state indexpb.JobState, failReason string) {
	it.manager.StoreIndexTaskState(it.req.GetClusterID(), it.req.GetBuildID(), commonpb.IndexState(state), failReason)
}

func (it *nestedIndexBuildTask) GetState() indexpb.JobState {
	return indexpb.JobState(it.manager.LoadIndexTaskState(it.req.GetClusterID(), it.req.GetBuildID()))
}

func (it *nestedIndexBuildTask) GetSlot() int64 {
	return it.req.GetTaskSlot()
}

func (it *nestedIndexBuildTask) OnEnqueue(ctx context.Context) error {
	it.queueDur = 0
	it.tr.RecordSpan()
	log.Ctx(ctx).Info("NestedIndexBuildTask Enqueue",
		zap.Int64("buildID", it.req.GetBuildID()),
		zap.Int64("segmentID", it.req.GetSegmentID()),
		zap.Int64("structFieldID", it.req.GetStructFieldId()),
	)
	return nil
}

func (it *nestedIndexBuildTask) PreExecute(ctx context.Context) error {
	it.queueDur = it.tr.RecordSpan()
	log.Ctx(ctx).Info("Begin to prepare nestedIndexBuildTask",
		zap.Int64("buildID", it.req.GetBuildID()),
		zap.Int64("collectionID", it.req.GetCollectionID()),
		zap.Int64("segmentID", it.req.GetSegmentID()),
		zap.Int64("structFieldID", it.req.GetStructFieldId()),
	)

	// Build data paths for each child field from data_ids
	for _, childField := range it.req.GetChildFields() {
		if len(childField.DataPaths) == 0 {
			for _, id := range childField.GetDataIds() {
				path := metautil.BuildInsertLogPath(
					it.req.GetStorageConfig().GetRootPath(),
					it.req.GetCollectionID(),
					it.req.GetPartitionID(),
					it.req.GetSegmentID(),
					childField.GetFieldId(),
					id,
				)
				childField.DataPaths = append(childField.DataPaths, path)
			}
		}
	}

	it.req.CurrentScalarIndexVersion = getCurrentScalarIndexVersion(it.req.GetCurrentScalarIndexVersion())
	log.Ctx(ctx).Info("Successfully prepare nestedIndexBuildTask", zap.Int64("buildID", it.req.GetBuildID()),
		zap.Int64("collectionID", it.req.GetCollectionID()), zap.Int64("segmentID", it.req.GetSegmentID()),
		zap.Int64("taskVersion", it.req.GetIndexVersion()),
		zap.Int32("currentScalarIndexVersion", it.req.GetCurrentScalarIndexVersion()),
	)
	return nil
}

func (it *nestedIndexBuildTask) Execute(ctx context.Context) error {
	log := log.Ctx(ctx).With(
		zap.String("clusterID", it.req.GetClusterID()),
		zap.Int64("buildID", it.req.GetBuildID()),
		zap.Int64("collectionID", it.req.GetCollectionID()),
		zap.Int64("segmentID", it.req.GetSegmentID()),
		zap.Int64("structFieldID", it.req.GetStructFieldId()),
		zap.Int32("currentScalarIndexVersion", it.req.GetCurrentScalarIndexVersion()),
	)

	log.Info("Start building nested index")

	storageConfig := &indexcgopb.StorageConfig{
		Address:           it.req.GetStorageConfig().GetAddress(),
		AccessKeyID:       it.req.GetStorageConfig().GetAccessKeyID(),
		SecretAccessKey:   it.req.GetStorageConfig().GetSecretAccessKey(),
		UseSSL:            it.req.GetStorageConfig().GetUseSSL(),
		BucketName:        it.req.GetStorageConfig().GetBucketName(),
		RootPath:          it.req.GetStorageConfig().GetRootPath(),
		UseIAM:            it.req.GetStorageConfig().GetUseIAM(),
		IAMEndpoint:       it.req.GetStorageConfig().GetIAMEndpoint(),
		StorageType:       it.req.GetStorageConfig().GetStorageType(),
		UseVirtualHost:    it.req.GetStorageConfig().GetUseVirtualHost(),
		Region:            it.req.GetStorageConfig().GetRegion(),
		CloudProvider:     it.req.GetStorageConfig().GetCloudProvider(),
		RequestTimeoutMs:  it.req.GetStorageConfig().GetRequestTimeoutMs(),
		SslCACert:         it.req.GetStorageConfig().GetSslCACert(),
		GcpCredentialJSON: it.req.GetStorageConfig().GetGcpCredentialJSON(),
	}

	childFields := make([]*indexcgopb.ChildFieldInfo, 0, len(it.req.GetChildFields()))
	for _, childField := range it.req.GetChildFields() {
		childFields = append(childFields, &indexcgopb.ChildFieldInfo{
			InsertFiles: childField.GetDataPaths(),
			FieldSchema: childField.GetField(),
		})
	}

	buildNestedIndexParams := &indexcgopb.BuildNestedIndexInfo{
		ClusterID:                 it.req.GetClusterID(),
		BuildID:                   it.req.GetBuildID(),
		CollectionID:              it.req.GetCollectionID(),
		PartitionID:               it.req.GetPartitionID(),
		SegmentID:                 it.req.GetSegmentID(),
		IndexVersion:              it.req.GetIndexVersion(),
		NumRows:                   it.req.GetNumRows(),
		StructFieldId:             it.req.GetStructFieldId(),
		StructFieldName:           it.req.GetStructFieldName(),
		ChildFields:               childFields,
		IndexFilePrefix:           it.req.GetIndexFilePrefix(),
		StorageConfig:             storageConfig,
		CurrentScalarIndexVersion: it.req.GetCurrentScalarIndexVersion(),
		LackBinlogRows:            it.req.GetLackBinlogRows(),
		StorageVersion:            it.req.GetStorageVersion(),
	}

	if it.pluginContext != nil {
		buildNestedIndexParams.StoragePluginContext = it.pluginContext
	}

	if buildNestedIndexParams.StorageVersion == storage.StorageV2 {
		buildNestedIndexParams.SegmentInsertFiles = util.GetSegmentInsertFiles(
			it.req.GetInsertLogs(),
			it.req.GetStorageConfig(),
			it.req.GetCollectionID(),
			it.req.GetPartitionID(),
			it.req.GetSegmentID())
		buildNestedIndexParams.Manifest = it.req.GetManifest()
	}
	log.Info("create nested index", zap.Any("buildNestedIndexParams", buildNestedIndexParams), zap.Int64("buildID", it.req.GetBuildID()))

	var err error
	it.index, err = indexcgowrapper.CreateNestedIndex(ctx, buildNestedIndexParams)
	if err != nil {
		if it.index != nil && it.index.CleanLocalData() != nil {
			log.Warn("failed to clean cached data on disk after build nested index failed")
		}
		log.Warn("failed to build nested index", zap.Error(err))
		return err
	}

	buildIndexLatency := it.tr.RecordSpan()
	metrics.DataNodeKnowhereBuildIndexLatency.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10)).Observe(buildIndexLatency.Seconds())

	log.Info("Successfully build nested index", zap.Int64("buildID", it.req.GetBuildID()))
	return nil
}

func (it *nestedIndexBuildTask) PostExecute(ctx context.Context) error {
	log := log.Ctx(ctx).With(
		zap.String("clusterID", it.req.GetClusterID()),
		zap.Int64("buildID", it.req.GetBuildID()),
		zap.Int64("collectionID", it.req.GetCollectionID()),
		zap.Int64("segmentID", it.req.GetSegmentID()),
	)

	gcIndex := func() {
		if err := it.index.Delete(); err != nil {
			log.Warn("nestedIndexBuildTask Execute CIndexDelete failed", zap.Error(err))
		}
	}
	indexStats, err := it.index.UpLoad()
	if err != nil {
		log.Warn("failed to upload nested index", zap.Error(err))
		gcIndex()
		return err
	}
	encodeIndexFileDur := it.tr.Record("nested index serialize and upload done")
	metrics.DataNodeEncodeIndexFileLatency.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10)).Observe(encodeIndexFileDur.Seconds())

	// early release index for gc, and we can ensure that Delete is idempotent.
	gcIndex()

	// use serialized size before encoding
	var serializedSize uint64
	saveFileKeys := make([]string, 0)
	for _, indexInfo := range indexStats.GetSerializedIndexInfos() {
		serializedSize += uint64(indexInfo.FileSize)
		parts := strings.Split(indexInfo.FileName, "/")
		fileKey := parts[len(parts)-1]
		saveFileKeys = append(saveFileKeys, fileKey)
	}

	it.manager.StoreIndexFilesAndStatistic(
		it.req.GetClusterID(),
		it.req.GetBuildID(),
		saveFileKeys,
		serializedSize,
		uint64(indexStats.MemSize),
		0,
		it.req.GetCurrentScalarIndexVersion(),
	)
	saveIndexFileDur := it.tr.RecordSpan()
	metrics.DataNodeSaveIndexFileLatency.WithLabelValues(strconv.FormatInt(paramtable.GetNodeID(), 10)).Observe(saveIndexFileDur.Seconds())
	it.tr.Elapse("nested index building all done")
	log.Info("Successfully save nested index files",
		zap.Uint64("serializedSize", serializedSize),
		zap.Int64("memSize", indexStats.MemSize),
		zap.Strings("indexFiles", saveFileKeys))
	return nil
}

func (it *nestedIndexBuildTask) Reset() {
	it.ident = ""
	it.cancel = nil
	it.ctx = nil
	it.cm = nil
	it.req = nil
	it.tr = nil
	it.manager = nil
	it.pluginContext = nil
}
