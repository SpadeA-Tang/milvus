package testutils

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus-proto/go-api/v2/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v2/schemapb"
	"github.com/milvus-io/milvus/pkg/v2/common"
	"github.com/milvus-io/milvus/pkg/v2/util/typeutil"
)

// this tests should be placed in package typeutil but we want to use GenerateStructFieldData in testutils
// where testutils imports typeutil so that typeutil cannot import testutils to use GenerateStructFieldData
func TestEstimateEntitySizeForStruct(t *testing.T) {
	dim := 128
	rowCount := 100
	sId := &schemapb.FieldSchema{
		FieldID:      103,
		Name:         "id",
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
		IndexParams:   nil,
		AutoID:        false,
		IsStructField: true,
	}
	sVec := &schemapb.FieldSchema{
		FieldID:      104,
		Name:         "structVec",
		IsPrimaryKey: false,
		Description:  "",
		DataType:     schemapb.DataType_Array,
		ElementType:  schemapb.DataType_FloatVector,
		TypeParams: []*commonpb.KeyValuePair{
			{
				Key:   common.DimKey,
				Value: strconv.Itoa(dim),
			},
			{
				Key:   common.MaxCapacityKey,
				Value: "10",
			},
		},
		IndexParams:   nil,
		AutoID:        false,
		IsStructField: true,
	}
	schema := &schemapb.StructFieldSchema{
		FieldID:            105,
		Name:               "struct",
		EnableDynamicField: false,
		Fields:             []*schemapb.FieldSchema{sId, sVec},
	}
	data := GenerateStructFieldData(schema, "struct", rowCount, dim)

	totalSize := 0
	for offset := 0; offset < rowCount; offset++ {
		size, error := typeutil.EstimateEntitySize([]*schemapb.FieldData{data}, offset)
		assert.NoError(t, error)
		totalSize += size
	}
	// it is hard coded in GenerateStructFieldData
	arrayNum := 10
	sizeExpected := rowCount * arrayNum * (dim*4 + 4)
	assert.Equal(t, totalSize, sizeExpected)
}

func TestEstimateEntitySize(t *testing.T) {
	samples := []*schemapb.FieldData{
		{
			FieldId:   111,
			FieldName: "float16_vector",
			Type:      schemapb.DataType_Float16Vector,
			Field: &schemapb.FieldData_Vectors{
				Vectors: &schemapb.VectorField{
					Dim:  64,
					Data: &schemapb.VectorField_Float16Vector{},
				},
			},
		},
		{
			FieldId:   112,
			FieldName: "bfloat16_vector",
			Type:      schemapb.DataType_BFloat16Vector,
			Field: &schemapb.FieldData_Vectors{
				Vectors: &schemapb.VectorField{
					Dim:  128,
					Data: &schemapb.VectorField_Bfloat16Vector{},
				},
			},
		},
	}
	size, error := typeutil.EstimateEntitySize(samples, int(0))
	assert.NoError(t, error)
	assert.True(t, size == 384)
}

// todo: Add more test for array of vector
