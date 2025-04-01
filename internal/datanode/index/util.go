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

/*
#cgo pkg-config: milvus_core

#include <stdlib.h>
#include <stdint.h>
#include "common/init_c.h"
#include "segcore/segcore_init_c.h"
#include "indexbuilder/init_c.h"
*/
import "C"

import (
	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/common"
)

func getCurrentIndexVersion(v int32) int32 {
	cCurrent := int32(C.GetCurrentIndexVersion())
	if cCurrent < v {
		return cCurrent
	}
	return v
}

func getCurrentScalarIndexVersion(v int32) int32 {
	cCurrent := common.CurrentScalarIndexEngineVersion
	if cCurrent < v {
		return cCurrent
	}
	return v
}

func estimateFieldDataSize(dim int64, numRows int64, dataType schemapb.DataType) (uint64, error) {
	switch dataType {
	case schemapb.DataType_BinaryVector:
		return uint64(dim) / 8 * uint64(numRows), nil
	case schemapb.DataType_FloatVector:
		return uint64(dim) * uint64(numRows) * 4, nil
	case schemapb.DataType_Float16Vector, schemapb.DataType_BFloat16Vector:
		return uint64(dim) * uint64(numRows) * 2, nil
	case schemapb.DataType_SparseFloatVector:
		return 0, errors.New("could not estimate field data size of SparseFloatVector")
	default:
		return 0, nil
	}
}

func mapToKVPairs(m map[string]string) []*commonpb.KeyValuePair {
	kvs := make([]*commonpb.KeyValuePair, 0, len(m))
	for k, v := range m {
		kvs = append(kvs, &commonpb.KeyValuePair{
			Key:   k,
			Value: v,
		})
	}
	return kvs
}
