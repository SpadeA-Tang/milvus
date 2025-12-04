use std::ffi::{c_char, c_void};
use std::ptr;

use crate::{
    error::{Result, TantivyBindingError},
    index_reader_nested::IndexReaderNestedWrapper,
    nested_query::NestedQuery,
    util::create_error,
};

/// Search nested documents using a protobuf-encoded query
///
/// # Arguments
/// * `reader_ptr` - Pointer to IndexReaderNestedWrapper
/// * `query_proto_data` - Pointer to protobuf bytes
/// * `query_proto_len` - Length of protobuf bytes
/// * `bitset` - Pointer to bitset to receive matching parent row IDs
///
/// # Returns
/// * Error message string if failed, NULL if succeeded
///
/// # Safety
/// The caller must:
/// - Free the error message (if not NULL) using `tantivy_free_error`
#[no_mangle]
pub extern "C" fn tantivy_search_nested(
    reader_ptr: *mut c_void,
    query_proto_data: *const u8,
    query_proto_len: usize,
    bitset: *mut c_void,
) -> *mut c_char {
    let result = || -> Result<()> {
        // Validate inputs
        if reader_ptr.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "reader_ptr is null".to_string(),
            ));
        }
        if query_proto_data.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "query_proto_data is null".to_string(),
            ));
        }
        if bitset.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "bitset is null".to_string(),
            ));
        }

        // Cast reader pointer
        let reader = unsafe { &*(reader_ptr as *const IndexReaderNestedWrapper) };

        // Convert protobuf bytes to NestedQuery
        let query_bytes = unsafe { std::slice::from_raw_parts(query_proto_data, query_proto_len) };
        let query = NestedQuery::from_proto_bytes(query_bytes)?;

        // Execute nested query - results are written directly to bitset
        reader.search_nested(&query, bitset)?;

        Ok(())
    };

    match result() {
        Ok(_) => ptr::null_mut(),
        Err(e) => create_error(&e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data_type::TantivyDataType,
        index_writer_nested::IndexWriterNestedWrapper,
        util::set_bitset,
    };
    use std::collections::HashSet;
    use std::ffi::CString;
    use tempfile::tempdir;

    #[test]
    fn test_search_nested_ffi() {
        // Create temporary directory for index
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        // Create nested index
        let field_names = vec!["int_field", "str_field"];
        let data_types = vec![TantivyDataType::I64, TantivyDataType::Keyword];

        let mut writer = IndexWriterNestedWrapper::new(
            "test_struct",
            &field_names,
            &data_types,
            path,
            1,
            15_000_000,
        )
        .unwrap();

        // Add test data: row 100 with two struct elements
        // Element 0: {int_field: 1, str_field: "aaa"}
        // Element 1: {int_field: 2, str_field: "bbb"}
        let int_data: Vec<i64> = vec![1, 2];
        let str_data: Vec<CString> = vec![
            CString::new("aaa").unwrap(),
            CString::new("bbb").unwrap(),
        ];
        let str_ptrs: Vec<*const i8> = str_data.iter().map(|s| s.as_ptr()).collect();

        let field_data: Vec<*const c_void> = vec![
            int_data.as_ptr() as *const c_void,
            str_ptrs.as_ptr() as *const c_void,
        ];

        writer
            .add_nested_documents(100, &field_data, 2, 2)
            .unwrap();
        writer.commit().unwrap();

        // Create reader
        let reader = writer.create_reader(set_bitset).unwrap();
        let reader_ptr = &reader as *const _ as *mut c_void;

        // Build protobuf query using Expr: (int_field == 1) AND (str_field == "aaa")
        // This should match element 0 of row 100
        use crate::nested_query::proto;
        use prost::Message;

        let query_proto = proto::Expr {
            expr: Some(proto::expr::Expr::BinaryExpr(Box::new(proto::BinaryExpr {
                op: proto::binary_expr::BinaryOp::LogicalAnd as i32,
                left: Some(Box::new(proto::Expr {
                    expr: Some(proto::expr::Expr::UnaryRangeExpr(proto::UnaryRangeExpr {
                        column_info: Some(proto::ColumnInfo {
                            nested_path: vec!["int_field".to_string()],
                            ..Default::default()
                        }),
                        op: proto::OpType::Equal as i32,
                        value: Some(proto::GenericValue {
                            val: Some(proto::generic_value::Val::Int64Val(1)),
                        }),
                        ..Default::default()
                    })),
                    is_template: false,
                })),
                right: Some(Box::new(proto::Expr {
                    expr: Some(proto::expr::Expr::UnaryRangeExpr(proto::UnaryRangeExpr {
                        column_info: Some(proto::ColumnInfo {
                            nested_path: vec!["str_field".to_string()],
                            ..Default::default()
                        }),
                        op: proto::OpType::Equal as i32,
                        value: Some(proto::GenericValue {
                            val: Some(proto::generic_value::Val::StringVal("aaa".to_string())),
                        }),
                        ..Default::default()
                    })),
                    is_template: false,
                })),
            }))),
            is_template: false,
        };

        let mut query_bytes = Vec::new();
        query_proto.encode(&mut query_bytes).unwrap();

        // Call FFI function with HashSet as bitset
        let mut result: HashSet<u32> = HashSet::new();
        let error = tantivy_search_nested(
            reader_ptr,
            query_bytes.as_ptr(),
            query_bytes.len(),
            &mut result as *mut _ as *mut c_void,
        );

        // Check no error
        assert!(error.is_null());

        // Check results - should find row 100
        assert_eq!(result.len(), 1);
        assert!(result.contains(&100));
    }

    #[test]
    fn test_search_nested_no_match() {
        // Similar test but with non-matching query
        let temp_dir = tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let field_names = vec!["int_field"];
        let data_types = vec![TantivyDataType::I64];

        let mut writer = IndexWriterNestedWrapper::new(
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

        writer
            .add_nested_documents(100, &field_data, 1, 2)
            .unwrap();
        writer.commit().unwrap();

        let reader = writer.create_reader(set_bitset).unwrap();
        let reader_ptr = &reader as *const _ as *mut c_void;

        // Query for non-existent value using Expr: int_field == 999
        use crate::nested_query::proto;
        use prost::Message;

        let query_proto = proto::Expr {
            expr: Some(proto::expr::Expr::UnaryRangeExpr(proto::UnaryRangeExpr {
                column_info: Some(proto::ColumnInfo {
                    nested_path: vec!["int_field".to_string()],
                    ..Default::default()
                }),
                op: proto::OpType::Equal as i32,
                value: Some(proto::GenericValue {
                    val: Some(proto::generic_value::Val::Int64Val(999)),
                }),
                ..Default::default()
            })),
            is_template: false,
        };

        let mut query_bytes = Vec::new();
        query_proto.encode(&mut query_bytes).unwrap();

        let mut result: HashSet<u32> = HashSet::new();
        let error = tantivy_search_nested(
            reader_ptr,
            query_bytes.as_ptr(),
            query_bytes.len(),
            &mut result as *mut _ as *mut c_void,
        );

        assert!(error.is_null());
        assert!(result.is_empty());
    }
}
