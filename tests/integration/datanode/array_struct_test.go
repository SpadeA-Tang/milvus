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

package datanode

import (
	"context"
	"fmt"
	"math/rand"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/log"
	"github.com/milvus-io/milvus/pkg/v2/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v2/util/merr"
	"github.com/milvus-io/milvus/pkg/v2/util/metric"
	"github.com/milvus-io/milvus/tests/integration"
)

type ArrayStructDataNodeSuite struct {
	integration.MiniClusterSuite
	maxGoRoutineNum   int
	dim               int
	numCollections    int
	rowsPerCollection int
	waitTimeInSec     time.Duration
	prefix            string

	generatedFieldData map[int64]*schemapb.FieldData
}

func (s *ArrayStructDataNodeSuite) setupParam() {
	s.maxGoRoutineNum = 100
	s.dim = 128
	s.numCollections = 2
	s.rowsPerCollection = 100
	s.waitTimeInSec = time.Second * 1
	s.generatedFieldData = make(map[int64]*schemapb.FieldData)
}

func (s *ArrayStructDataNodeSuite) loadCollection(collectionName string) {
	c := s.Cluster
	dbName := ""
	schema := integration.ConstructSchema(collectionName, s.dim, true)

	sId := &schemapb.FieldSchema{
		FieldID:      103,
		Name:         integration.StructSubInt32Field,
		IsPrimaryKey: false,
		Description:  "",
		DataType:     schemapb.DataType_Array,
		ElementType:  schemapb.DataType_Int32,
		TypeParams: []*commonpb.KeyValuePair{
			{
				Key:   common.MaxCapacityKey,
				Value: "100",
			},
		},
		IndexParams: nil,
		AutoID:      false,
	}
	sVec := &schemapb.FieldSchema{
		FieldID:      104,
		Name:         integration.StructSubFloatVecField,
		IsPrimaryKey: false,
		Description:  "",
		DataType:     schemapb.DataType_ArrayOfVector,
		ElementType:  schemapb.DataType_FloatVector,
		TypeParams: []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: strconv.Itoa(s.dim),
			},
			{
				Key:   common.MaxCapacityKey,
				Value: "100",
			},
		},
		IndexParams: nil,
		AutoID:      false,
	}
	structF := &schemapb.StructFieldSchema{
		FieldID:            105,
		Name:               integration.StructField,
		EnableDynamicField: false,
		Fields:             []*schemapb.FieldSchema{sId, sVec},
	}
	schema.StructFields = []*schemapb.StructFieldSchema{structF}

	marshaledSchema, err := proto.Marshal(schema)
	s.NoError(err)

	createCollectionStatus, err := c.Proxy.CreateCollection(context.TODO(), &milvuspb.CreateCollectionRequest{
		DbName:         dbName,
		CollectionName: collectionName,
		Schema:         marshaledSchema,
		ShardsNum:      common.DefaultShardsNum,
	})
	s.NoError(err)

	err = merr.Error(createCollectionStatus)
	s.NoError(err)

	showCollectionsResp, err := c.Proxy.ShowCollections(context.TODO(), &milvuspb.ShowCollectionsRequest{})
	s.NoError(err)
	s.True(merr.Ok(showCollectionsResp.GetStatus()))

	rowNum := s.rowsPerCollection
	fVecColumn := integration.NewFloatVectorFieldData(integration.FloatVecField, rowNum, s.dim)
	hashKeys := integration.GenerateHashKeys(rowNum)
	structColumn := integration.NewStructFieldData(schema.StructFields[0], integration.StructField, rowNum, s.dim)

	s.generatedFieldData[101] = fVecColumn
	// s.generatedFieldData[structColumn.FieldId] = structColumn
	s.generatedFieldData[structColumn.GetArrayStruct().Fields[0].FieldId] = structColumn.GetArrayStruct().Fields[0]
	s.generatedFieldData[structColumn.GetArrayStruct().Fields[1].FieldId] = structColumn.GetArrayStruct().Fields[1]

	insertResult, err := c.Proxy.Insert(context.TODO(), &milvuspb.InsertRequest{
		DbName:         dbName,
		CollectionName: collectionName,
		FieldsData:     []*schemapb.FieldData{fVecColumn, structColumn},
		HashKeys:       hashKeys,
		NumRows:        uint32(rowNum),
	})
	s.NoError(err)
	s.True(merr.Ok(insertResult.GetStatus()))
	log.Info("=========================Data insertion finished=========================")

	// flush
	flushResp, err := c.Proxy.Flush(context.TODO(), &milvuspb.FlushRequest{
		DbName:          dbName,
		CollectionNames: []string{collectionName},
	})
	s.NoError(err)
	segmentIDs, has := flushResp.GetCollSegIDs()[collectionName]
	ids := segmentIDs.GetData()
	s.Require().NotEmpty(segmentIDs)
	s.Require().True(has)
	flushTs, has := flushResp.GetCollFlushTs()[collectionName]
	s.True(has)

	segments, err := c.MetaWatcher.ShowSegments()
	s.NoError(err)
	s.NotEmpty(segments)
	s.WaitForFlush(context.TODO(), ids, flushTs, dbName, collectionName)
	log.Info("=========================Data flush finished=========================")

	// create index
	createIndexStatus, err := c.Proxy.CreateIndex(context.TODO(), &milvuspb.CreateIndexRequest{
		CollectionName: collectionName,
		FieldName:      integration.FloatVecField,
		IndexName:      "_default",
		ExtraParams:    integration.ConstructIndexParam(s.dim, integration.IndexFaissIvfFlat, metric.IP),
	})
	s.NoError(err)
	err = merr.Error(createIndexStatus)
	s.NoError(err)
	s.WaitForIndexBuilt(context.TODO(), collectionName, integration.FloatVecField)
	log.Info("=========================Index created=========================")

	// load
	loadStatus, err := c.Proxy.LoadCollection(context.TODO(), &milvuspb.LoadCollectionRequest{
		DbName:         dbName,
		CollectionName: collectionName,
	})
	s.NoError(err)
	err = merr.Error(loadStatus)
	s.NoError(err)
	s.WaitForLoad(context.TODO(), collectionName)
	log.Info("=========================Collection loaded=========================")
}

func (s *ArrayStructDataNodeSuite) checkCollections() bool {
	req := &milvuspb.ShowCollectionsRequest{
		DbName:    "",
		TimeStamp: 0, // means now
	}
	resp, err := s.Cluster.Proxy.ShowCollections(context.TODO(), req)
	s.NoError(err)
	s.Equal(len(resp.CollectionIds), s.numCollections)
	notLoaded := 0
	loaded := 0
	for _, name := range resp.CollectionNames {
		loadProgress, err := s.Cluster.Proxy.GetLoadingProgress(context.TODO(), &milvuspb.GetLoadingProgressRequest{
			DbName:         "",
			CollectionName: name,
		})
		s.NoError(err)
		if loadProgress.GetProgress() != int64(100) {
			notLoaded++
		} else {
			loaded++
		}
	}
	log.Info(fmt.Sprintf("loading status: %d/%d", loaded, len(resp.GetCollectionNames())))
	return notLoaded == 0
}

func (s *ArrayStructDataNodeSuite) checkFieldsData(fieldsData []*schemapb.FieldData) {
	for _, fieldData := range fieldsData {
		for i := 0; i < s.rowsPerCollection; i++ {
			switch fieldData.FieldName {
			case integration.Int64Field:
				break
			case integration.FloatVecField:
				for j := 0; j < s.dim; j++ {
					s.Equal(fieldData.GetVectors().GetFloatVector().Data[i*s.dim+j],
						s.generatedFieldData[fieldData.FieldId].GetVectors().GetFloatVector().Data[j])
				}
			case integration.StructSubInt32Field:
				getData := fieldData.GetScalars().GetArrayData().Data[i]
				generatedData := s.generatedFieldData[fieldData.FieldId].GetScalars().GetArrayData().Data[i]

				arrayLen := len(getData.GetIntData().Data)
				s.Equal(arrayLen, len(generatedData.GetIntData().Data))

				for j := 0; j < arrayLen; j++ {
					s.Equal(getData.GetIntData().Data[j], generatedData.GetIntData().Data[j])
				}
			case integration.StructSubFloatVecField:
				getData := fieldData.GetVectors().GetArrayVector().Data[i]
				generatedData := s.generatedFieldData[fieldData.FieldId].GetVectors().GetArrayVector().Data[i]

				length := len(getData.GetFloatVector().Data)
				s.Equal(length, len(generatedData.GetFloatVector().Data))

				for j := 0; j < length; j++ {
					s.Equal(getData.GetFloatVector().Data[j], generatedData.GetFloatVector().Data[j])
				}
			case integration.StructField:
				for _, field := range fieldData.GetArrayStruct().Fields {
					if field.FieldName == integration.StructSubInt32Field {
						getData := field.GetScalars().GetArrayData().Data[i]
						generatedData := s.generatedFieldData[field.FieldId].GetScalars().GetArrayData().Data[i]

						arrayLen := len(getData.GetIntData().Data)
						s.Equal(arrayLen, len(generatedData.GetIntData().Data))

						for j := 0; j < arrayLen; j++ {
							s.Equal(getData.GetIntData().Data[j], generatedData.GetIntData().Data[j])
						}

					} else if field.FieldName == integration.StructSubFloatVecField {
						getData := field.GetVectors().GetArrayVector().Data[i]
						generatedData := s.generatedFieldData[field.FieldId].GetVectors().GetArrayVector().Data[i]

						length := len(getData.GetFloatVector().Data)
						s.Equal(length, len(generatedData.GetFloatVector().Data))

						for j := 0; j < length; j++ {
							s.Equal(getData.GetFloatVector().Data[j], generatedData.GetFloatVector().Data[j])
						}

					}
				}
			default:
				s.Fail(fmt.Sprintf("unsupported field type: %s", fieldData.FieldName))
			}

		}
	}
}

func (s *ArrayStructDataNodeSuite) search(collectionName string) {
	c := s.Cluster
	var err error
	// Query
	queryReq := &milvuspb.QueryRequest{
		Base:               nil,
		CollectionName:     collectionName,
		PartitionNames:     nil,
		Expr:               "",
		OutputFields:       []string{"*"},
		TravelTimestamp:    0,
		GuaranteeTimestamp: 0,
		QueryParams: []*commonpb.KeyValuePair{
			{
				Key:   "limit",
				Value: strconv.Itoa(s.rowsPerCollection),
			},
		},
	}
	queryResult, err := c.Proxy.Query(context.TODO(), queryReq)
	s.NoError(err)
	s.Equal(len(queryResult.FieldsData), 2)
	s.checkFieldsData(queryResult.FieldsData)

	// Search
	expr := fmt.Sprintf("%s > 0", integration.Int64Field)
	nq := 10
	topk := 10
	roundDecimal := -1
	radius := 10

	params := integration.GetSearchParams(integration.IndexFaissIvfFlat, metric.IP)
	params["radius"] = radius
	searchReq := integration.ConstructSearchRequest("", collectionName, expr,
		integration.FloatVecField, schemapb.DataType_FloatVector, nil, metric.IP, params, nq, s.dim, topk, roundDecimal)

	searchResult, _ := c.Proxy.Search(context.TODO(), searchReq)

	err = merr.Error(searchResult.GetStatus())
	s.NoError(err)
}

func (s *ArrayStructDataNodeSuite) insertBatchCollections(prefix string, collectionBatchSize, idxStart int, wg *sync.WaitGroup) {
	for idx := 0; idx < collectionBatchSize; idx++ {
		collectionName := prefix + "_" + strconv.Itoa(idxStart+idx)
		s.loadCollection(collectionName)
	}
	wg.Done()
}

func (s *ArrayStructDataNodeSuite) setupData() {
	// Add the second data node
	s.Cluster.AddDataNode()
	goRoutineNum := s.maxGoRoutineNum
	if goRoutineNum > s.numCollections {
		goRoutineNum = s.numCollections
	}
	collectionBatchSize := s.numCollections / goRoutineNum
	log.Info(fmt.Sprintf("=========================test with dim=%d, s.rowsPerCollection=%d, s.numCollections=%d, goRoutineNum=%d==================", s.dim, s.rowsPerCollection, s.numCollections, goRoutineNum))
	log.Info("=========================Start to inject data=========================")
	s.prefix = "TestDataNodeUtil" + funcutil.GenRandomStr()
	searchName := s.prefix + "_0"
	wg := sync.WaitGroup{}
	for idx := 0; idx < goRoutineNum; idx++ {
		wg.Add(1)
		go s.insertBatchCollections(s.prefix, collectionBatchSize, idx*collectionBatchSize, &wg)
	}
	wg.Wait()
	log.Info("=========================Data injection finished=========================")
	s.checkCollections()
	log.Info(fmt.Sprintf("=========================start to search %s=========================", searchName))
	s.search(searchName)
	log.Info("=========================Search finished=========================")
	time.Sleep(s.waitTimeInSec)
	s.checkCollections()
	log.Info(fmt.Sprintf("=========================start to search2 %s=========================", searchName))
	s.search(searchName)
	log.Info("=========================Search2 finished=========================")
	s.checkAllCollectionsReady()
}

func (s *ArrayStructDataNodeSuite) checkAllCollectionsReady() {
	goRoutineNum := s.maxGoRoutineNum
	if goRoutineNum > s.numCollections {
		goRoutineNum = s.numCollections
	}
	collectionBatchSize := s.numCollections / goRoutineNum
	for i := 0; i < goRoutineNum; i++ {
		for idx := 0; idx < collectionBatchSize; idx++ {
			collectionName := s.prefix + "_" + strconv.Itoa(i*collectionBatchSize+idx)
			s.search(collectionName)
			queryReq := &milvuspb.QueryRequest{
				CollectionName: collectionName,
				Expr:           "",
				OutputFields:   []string{"count(*)"},
			}
			_, err := s.Cluster.Proxy.Query(context.TODO(), queryReq)
			s.NoError(err)
		}
	}
}

func (s *ArrayStructDataNodeSuite) checkQNRestarts(idx int) {
	// Stop all data nodes
	s.Cluster.StopAllDataNodes()
	// Add new data nodes.
	qn1 := s.Cluster.AddDataNode()
	qn2 := s.Cluster.AddDataNode()
	time.Sleep(s.waitTimeInSec)
	cn := fmt.Sprintf("new_collection_r_%d", idx)
	s.loadCollection(cn)
	s.search(cn)
	// Randomly stop one data node.
	if rand.Intn(2) == 0 {
		qn1.Stop()
	} else {
		qn2.Stop()
	}
	time.Sleep(s.waitTimeInSec)
	cn = fmt.Sprintf("new_collection_x_%d", idx)
	s.loadCollection(cn)
	s.search(cn)
}

func (s *ArrayStructDataNodeSuite) TestSwapQN() {
	s.setupParam()
	s.setupData()
	cn := "new_collection_a"
	s.loadCollection(cn)
	s.search(cn)
}

func TestArrayStructDataNodeUtil(t *testing.T) {
	suite.Run(t, new(ArrayStructDataNodeSuite))
}
