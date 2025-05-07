package model

import "github.com/milvus-io/milvus-proto/go-api/v2/schemapb"

type StructField struct {
	FieldID            int64
	Name               string
	Fields             []*Field
	Functions          []*Function
	EnableDynamicField bool
}

func MarshalStructFieldModel(structField *StructField) *schemapb.StructField {
	if structField == nil {
		return nil
	}

	return &schemapb.StructField{
		FieldID:            structField.FieldID,
		Name:               structField.Name,
		Fields:             MarshalFieldModels(structField.Fields),
		Functions:          MarshalFunctionModels(structField.Functions),
		EnableDynamicField: structField.EnableDynamicField,
	}
}

func MarshalStructFieldModels(fieldSchemas []*StructField) []*schemapb.StructField {
	if fieldSchemas == nil {
		return nil
	}

	structFields := make([]*schemapb.StructField, len(fieldSchemas))
	for idx, structField := range fieldSchemas {
		structFields[idx] = MarshalStructFieldModel(structField)
	}
	return structFields
}

func UnmarshalStructFieldModel(fieldSchema *schemapb.StructField) *StructField {
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

func UnmarshalStructFieldModels(fieldSchemas []*schemapb.StructField) []*StructField {
	if fieldSchemas == nil {
		return nil
	}

	structFields := make([]*StructField, len(fieldSchemas))
	for idx, structFieldSchema := range fieldSchemas {
		structFields[idx] = UnmarshalStructFieldModel(structFieldSchema)
	}
	return structFields
}
