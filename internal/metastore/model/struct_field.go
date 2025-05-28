package model

import "github.com/milvus-io/milvus-proto/go-api/v2/schemapb"

type StructField struct {
	FieldID            int64
	Name               string
	Fields             []*Field
	Functions          []*Function
	EnableDynamicField bool
}

func (s *StructField) Clone() *StructField {
	return &StructField{
		FieldID:            s.FieldID,
		Name:               s.Name,
		Fields:             CloneFields(s.Fields),
		Functions:          CloneFunctions(s.Functions),
		EnableDynamicField: s.EnableDynamicField,
	}
}

func CloneStructFields(structFields []*StructField) []*StructField {
	clone := make([]*StructField, len(structFields))
	for i, structField := range structFields {
		clone[i] = structField.Clone()
	}
	return clone
}

func (s *StructField) Equal(other StructField) bool {
	return s.FieldID == other.FieldID &&
		s.Name == other.Name &&
		CheckFieldsEqual(s.Fields, other.Fields) &&
		s.EnableDynamicField == other.EnableDynamicField
}

func CheckStructFieldsEqual(structFieldsA, structFieldsB []*StructField) bool {
	if len(structFieldsA) != len(structFieldsB) {
		return false
	}

	mapA := make(map[int64]*StructField)
	for _, f := range structFieldsA {
		mapA[f.FieldID] = f
	}

	for _, f := range structFieldsB {
		if other, exists := mapA[f.FieldID]; !exists || !f.Equal(*other) {
			return false
		}
	}
	return true
}

func MarshalStructFieldModel(structField *StructField) *schemapb.StructArrayFieldSchema {
	if structField == nil {
		return nil
	}

	return &schemapb.StructArrayFieldSchema{
		FieldID:            structField.FieldID,
		Name:               structField.Name,
		Fields:             MarshalFieldModels(structField.Fields),
		Functions:          MarshalFunctionModels(structField.Functions),
		EnableDynamicField: structField.EnableDynamicField,
	}
}

func MarshalStructFieldModels(fieldSchemas []*StructField) []*schemapb.StructArrayFieldSchema {
	if fieldSchemas == nil {
		return nil
	}

	structFields := make([]*schemapb.StructArrayFieldSchema, len(fieldSchemas))
	for idx, structField := range fieldSchemas {
		structFields[idx] = MarshalStructFieldModel(structField)
	}
	return structFields
}

func UnmarshalStructFieldModel(fieldSchema *schemapb.StructArrayFieldSchema) *StructField {
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

func UnmarshalStructFieldModels(fieldSchemas []*schemapb.StructArrayFieldSchema) []*StructField {
	if fieldSchemas == nil {
		return nil
	}

	structFields := make([]*StructField, len(fieldSchemas))
	for idx, StructArrayFieldSchema := range fieldSchemas {
		structFields[idx] = UnmarshalStructFieldModel(StructArrayFieldSchema)
	}
	return structFields
}
