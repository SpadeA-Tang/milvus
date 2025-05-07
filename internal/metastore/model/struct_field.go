package model

import "github.com/milvus-io/milvus-proto/go-api/v2/schemapb"

type StructField struct {
	FieldID            int64
	Name               string
	Fields             []*Field
	Functions          []*Function
	EnableDynamicField bool
}

func UnmarshalStructFieldModel(fieldSchema *schemapb.StructSchema) *StructField {
	if fieldSchema == nil {
		return nil
	}

	return &StructField{
		FieldID:            fieldSchema.FieldID,
		Name:               fieldSchema.Name,
		Fields:             UnmarshalFieldModels(fieldSchema.Fields),
		Functions:          UnmarshalFunctionModels(fieldSchema.Functions),
		EnableDynamicField: fieldSchema.EnableDynamicField,
	}
}

func UnmarshalStructFieldModels(fieldSchemas []*schemapb.StructSchema) []*StructField {
	if fieldSchemas == nil {
		return nil
	}

	structFields := make([]*StructField, len(fieldSchemas))
	for idx, structFieldSchema := range fieldSchemas {
		structFields[idx] = UnmarshalStructFieldModel(structFieldSchema)
	}
	return structFields
}
