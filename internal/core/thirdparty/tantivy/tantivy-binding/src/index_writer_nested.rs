use std::ffi::{c_char, c_void};
use std::sync::Arc;

use tantivy::{
    schema::{Field, Schema, FAST},
    Index, IndexWriter, TantivyDocument,
};

use crate::index_reader_c::SetBitsetFn;
use crate::index_reader_nested::IndexReaderNestedWrapper;
use crate::{
    data_type::TantivyDataType,
    error::{Result, TantivyBindingError},
    index_writer_v7::index_writer::schema_builder_add_field,
    util::c_ptr_to_str,
};
use log::info;

/// A wrapper for indexing nested documents for fields in a milvus Struct.
pub struct IndexWriterNestedWrapper {
    struct_name: String,
    fields: Vec<Field>,
    field_types: Vec<TantivyDataType>,
    index_writer: IndexWriter,
    index: Arc<Index>,
    id_field: Field,
}

impl IndexWriterNestedWrapper {
    pub(crate) fn new(
        struct_name: &str,
        field_name: &[&str],
        data_types: &[TantivyDataType],
        path: &str,
        num_threads: usize,
        overall_memory_budget_in_bytes: usize,
    ) -> Result<IndexWriterNestedWrapper> {
        assert_eq!(field_name.len(), data_types.len());

        info!(
            "create nested index writer, struct_name: {}, field_names: {:?}, data_types: {:?}, tantivy_index_version 7",
            struct_name,
            field_name, data_types
        );

        let mut schema_builder = Schema::builder();
        for (field_name, data_type) in field_name.iter().zip(data_types.iter()) {
            schema_builder_add_field(&mut schema_builder, field_name, *data_type);
        }
        let id_field = schema_builder.add_i64_field("doc_id", FAST);
        let schema = schema_builder.build();
        let fields: Vec<Field> = schema.fields().map(|(field, _)| field).collect();
        let index = Index::create_in_dir(path, schema)?;
        let index_writer =
            index.writer_with_num_threads(num_threads, overall_memory_budget_in_bytes)?;

        Ok(IndexWriterNestedWrapper {
            struct_name: struct_name.to_string(),
            fields,
            field_types: data_types.iter().cloned().collect(),
            index_writer,
            index: Arc::new(index),
            id_field,
        })
    }

    pub fn add_nested_documents(
        &mut self,
        row_id: i64,
        field_data: &[*const c_void],
        field_count: usize,
        array_count: usize,
    ) -> Result<()> {
        assert_eq!(field_count, self.field_types.len());

        // Create a document for each array element
        for i in 0..array_count {
            let mut doc = TantivyDocument::default();
            doc.add_i64(self.id_field, row_id);

            // Add all field values
            for field_idx in 0..field_count {
                let field_type = self.field_types[field_idx];
                let field = self.fields[field_idx];

                unsafe {
                    match field_type {
                        TantivyDataType::I64 => {
                            let data_ptr = field_data[field_idx] as *const i64;
                            let value = *data_ptr.add(i);
                            doc.add_i64(field, value);
                        }
                        TantivyDataType::F64 => {
                            let data_ptr = field_data[field_idx] as *const f64;
                            let value = *data_ptr.add(i);
                            doc.add_f64(field, value);
                        }
                        TantivyDataType::Keyword | TantivyDataType::Text => {
                            let str_array = field_data[field_idx] as *const *const c_char;
                            let str_ptr = *str_array.add(i);
                            let value = c_ptr_to_str(str_ptr)?;
                            doc.add_text(field, value);
                        }
                        TantivyDataType::Bool => {
                            let data_ptr = field_data[field_idx] as *const bool;
                            let value = *data_ptr.add(i);
                            doc.add_bool(field, value);
                        }
                        _ => {
                            return Err(TantivyBindingError::InvalidArgument(format!(
                                "unsupported field type for nested index: {:?}, struct: {}",
                                field_type, self.struct_name
                            )));
                        }
                    }
                }
            }

            // Add document after all fields are processed
            self.index_writer.add_document(doc)?;
        }

        Ok(())
    }

    pub fn create_reader(&self, set_bitset: SetBitsetFn) -> Result<IndexReaderNestedWrapper> {
        IndexReaderNestedWrapper::from_index(self.index.clone(), set_bitset)
    }

    pub fn commit(&mut self) -> Result<()> {
        self.index_writer.commit()?;
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.index_writer.commit()?;
        self.index_writer.wait_merging_threads()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use tempfile::tempdir;

    use crate::util::set_bitset;

    #[test]
    fn test_add_nested_documents_basic() {
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let field_names = vec!["int_field", "str_field"];
        let data_types = vec![TantivyDataType::I64, TantivyDataType::Keyword];

        let mut wrapper = IndexWriterNestedWrapper::new(
            "test_struct",
            &field_names,
            &data_types,
            path,
            1,
            15_000_000,
        )
        .unwrap();

        // Prepare test data
        let int_data: Vec<i64> = vec![1, 2, 3];
        let str_data: Vec<CString> = vec![
            CString::new("hello").unwrap(),
            CString::new("world").unwrap(),
            CString::new("test").unwrap(),
        ];
        let str_ptrs: Vec<*const c_char> = str_data.iter().map(|s| s.as_ptr()).collect();

        let field_data: Vec<*const c_void> = vec![
            int_data.as_ptr() as *const c_void,
            str_ptrs.as_ptr() as *const c_void,
        ];

        // Add documents
        let result = wrapper.add_nested_documents(
            100, // row_id
            &field_data,
            2, // field_count
            3, // array_count
        );

        assert!(result.is_ok());
        assert!(wrapper.commit().is_ok());

        let reader = wrapper.create_reader(set_bitset).unwrap();
    }

    #[test]
    fn test_add_nested_documents_empty_array() {
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let field_names = vec!["field1"];
        let data_types = vec![TantivyDataType::I64];

        let mut wrapper = IndexWriterNestedWrapper::new(
            "test_struct",
            &field_names,
            &data_types,
            path,
            1,
            15_000_000,
        )
        .unwrap();

        let int_data: Vec<i64> = vec![];
        let field_data: Vec<*const c_void> = vec![int_data.as_ptr() as *const c_void];

        // Adding empty array should succeed (no documents added)
        let result = wrapper.add_nested_documents(
            100,
            &field_data,
            1,
            0, // array_count = 0
        );

        assert!(result.is_ok());
    }

    #[test]
    #[should_panic]
    fn test_add_nested_documents_field_count_mismatch() {
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let field_names = vec!["field1", "field2"];
        let data_types = vec![TantivyDataType::I64, TantivyDataType::I64];

        let mut wrapper = IndexWriterNestedWrapper::new(
            "test_struct",
            &field_names,
            &data_types,
            path,
            1,
            15_000_000,
        )
        .unwrap();

        let int_data: Vec<i64> = vec![1, 2];
        let field_data: Vec<*const c_void> = vec![int_data.as_ptr() as *const c_void];

        // Should panic (field_count mismatch)
        wrapper
            .add_nested_documents(
                100,
                &field_data,
                1, // Wrong field_count (should be 2)
                2,
            )
            .unwrap();
    }

    #[test]
    fn test_add_nested_documents_all_types() {
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let field_names = vec!["bool_field", "int_field", "float_field", "keyword_field"];
        let data_types = vec![
            TantivyDataType::Bool,
            TantivyDataType::I64,
            TantivyDataType::F64,
            TantivyDataType::Keyword,
        ];

        let mut wrapper = IndexWriterNestedWrapper::new(
            "test_struct",
            &field_names,
            &data_types,
            path,
            1,
            15_000_000,
        )
        .unwrap();

        // Prepare data for various types
        let bool_data: Vec<bool> = vec![true, false];
        let int_data: Vec<i64> = vec![42, 84];
        let float_data: Vec<f64> = vec![3.14, 2.71];
        let str_data: Vec<CString> = vec![
            CString::new("test1").unwrap(),
            CString::new("test2").unwrap(),
        ];
        let str_ptrs: Vec<*const c_char> = str_data.iter().map(|s| s.as_ptr()).collect();

        let field_data: Vec<*const c_void> = vec![
            bool_data.as_ptr() as *const c_void,
            int_data.as_ptr() as *const c_void,
            float_data.as_ptr() as *const c_void,
            str_ptrs.as_ptr() as *const c_void,
        ];

        let result = wrapper.add_nested_documents(200, &field_data, 4, 2);

        assert!(result.is_ok());
        assert!(wrapper.finish().is_ok());
    }

    #[test]
    fn test_multiple_rows() {
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let field_names = vec!["value"];
        let data_types = vec![TantivyDataType::I64];

        let mut wrapper = IndexWriterNestedWrapper::new(
            "test_struct",
            &field_names,
            &data_types,
            path,
            1,
            15_000_000,
        )
        .unwrap();

        // Add multiple rows
        for row_id in 0..10 {
            let int_data: Vec<i64> = vec![row_id * 10, row_id * 10 + 1, row_id * 10 + 2];
            let field_data: Vec<*const c_void> = vec![int_data.as_ptr() as *const c_void];

            let result = wrapper.add_nested_documents(row_id, &field_data, 1, 3);

            assert!(result.is_ok());
        }

        assert!(wrapper.commit().is_ok());
    }
}
