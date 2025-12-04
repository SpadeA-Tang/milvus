type NestedIndexChecker struct {
	scalarIndexChecker
}

func newNestedIndexChecker() *NestedIndexChecker {
	return &NestedIndexChecker{}
}

func (c *NestedIndexChecker) CheckTrain(dataType schemapb.DataType, elementType schemapb.DataType, params map[string]string) error {
	return nil
}

func (c *NestedIndexChecker) CheckValidDataType(indexType IndexType, field *schemapb.FieldSchema) error {
	return nil
}
