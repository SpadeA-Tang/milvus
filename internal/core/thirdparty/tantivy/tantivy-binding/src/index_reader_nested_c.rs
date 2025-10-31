use std::ffi::{c_char, c_void};
use std::ptr;

use crate::{
    error::{Result, TantivyBindingError},
    index_reader_nested::IndexReaderNestedWrapper,
    nested_query::NestedQuery,
    util::{create_error, free_error},
};

/// Search nested documents using a protobuf-encoded query
///
/// # Arguments
/// * `reader_ptr` - Pointer to IndexReaderNestedWrapper
/// * `query_proto_data` - Pointer to protobuf bytes
/// * `query_proto_len` - Length of protobuf bytes
/// * `result_row_ids` - Output pointer to receive array of matching parent row IDs
/// * `result_count` - Output pointer to receive count of matching row IDs
///
/// # Returns
/// * Error message string if failed, NULL if succeeded
///
/// # Safety
/// The caller must:
/// - Free the returned row_ids array using `tantivy_free_row_ids`
/// - Free the error message (if not NULL) using `tantivy_free_error`
#[no_mangle]
pub extern "C" fn tantivy_search_nested(
    reader_ptr: *mut c_void,
    query_proto_data: *const u8,
    query_proto_len: usize,
    result_row_ids: *mut *mut i64,
    result_count: *mut usize,
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
        if result_row_ids.is_null() || result_count.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "result pointers are null".to_string(),
            ));
        }

        // Cast reader pointer
        let reader = unsafe { &*(reader_ptr as *const IndexReaderNestedWrapper) };

        // Convert protobuf bytes to NestedQuery
        let query_bytes = unsafe { std::slice::from_raw_parts(query_proto_data, query_proto_len) };
        let query = NestedQuery::from_proto_bytes(query_bytes)?;

        // Execute nested query
        let row_ids = reader.search_nested(&query)?;

        // Allocate result array
        let count = row_ids.len();
        if count == 0 {
            unsafe {
                *result_row_ids = ptr::null_mut();
                *result_count = 0;
            }
            return Ok(());
        }

        // Convert Vec to heap-allocated array
        let boxed_slice = row_ids.into_boxed_slice();
        let raw_ptr = Box::into_raw(boxed_slice);

        unsafe {
            *result_row_ids = raw_ptr as *mut i64;
            *result_count = count;
        }

        Ok(())
    };

    match result() {
        Ok(_) => ptr::null_mut(),
        Err(e) => create_error(&e.to_string()),
    }
}

/// Free the row IDs array returned by tantivy_search_nested
///
/// # Safety
/// Must only be called once with a pointer returned by tantivy_search_nested
#[no_mangle]
pub extern "C" fn tantivy_free_row_ids(row_ids: *mut i64, count: usize) {
    if !row_ids.is_null() && count > 0 {
        unsafe {
            // Reconstruct the Box to drop it
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(row_ids, count));
        }
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

        // Build protobuf query: int_field == 1 AND str_field == "aaa"
        // This should match element 0 of row 100
        use crate::nested_query::proto;
        use prost::Message;

        let query_proto = proto::TantivyQueryExpr {
            expr: Some(proto::tantivy_query_expr::Expr::Logical(
                proto::TantivyLogicalExpr {
                    op: proto::tantivy_logical_expr::Op::And as i32,
                    children: vec![
                        proto::TantivyQueryExpr {
                            expr: Some(proto::tantivy_query_expr::Expr::Condition(
                                proto::TantivyFieldCondition {
                                    field_name: "int_field".to_string(),
                                    condition_type: Some(
                                        proto::tantivy_field_condition::ConditionType::TermI64(1),
                                    ),
                                },
                            )),
                        },
                        proto::TantivyQueryExpr {
                            expr: Some(proto::tantivy_query_expr::Expr::Condition(
                                proto::TantivyFieldCondition {
                                    field_name: "str_field".to_string(),
                                    condition_type: Some(
                                        proto::tantivy_field_condition::ConditionType::TermKeyword(
                                            "aaa".to_string(),
                                        ),
                                    ),
                                },
                            )),
                        },
                    ],
                },
            )),
        };

        let mut query_bytes = Vec::new();
        query_proto.encode(&mut query_bytes).unwrap();

        // Call FFI function
        let mut result_row_ids: *mut i64 = ptr::null_mut();
        let mut result_count: usize = 0;

        let error = tantivy_search_nested(
            reader_ptr,
            query_bytes.as_ptr(),
            query_bytes.len(),
            &mut result_row_ids,
            &mut result_count,
        );

        // Check no error
        assert!(error.is_null());

        // Check results
        assert_eq!(result_count, 1);
        assert!(!result_row_ids.is_null());

        let row_ids_slice = unsafe { std::slice::from_raw_parts(result_row_ids, result_count) };
        assert_eq!(row_ids_slice[0], 100);

        // Free result
        tantivy_free_row_ids(result_row_ids, result_count);
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

        // Query for non-existent value
        use crate::nested_query::proto;
        use prost::Message;

        let query_proto = proto::TantivyQueryExpr {
            expr: Some(proto::tantivy_query_expr::Expr::Condition(
                proto::TantivyFieldCondition {
                    field_name: "int_field".to_string(),
                    condition_type: Some(proto::tantivy_field_condition::ConditionType::TermI64(
                        999,
                    )),
                },
            )),
        };

        let mut query_bytes = Vec::new();
        query_proto.encode(&mut query_bytes).unwrap();

        let mut result_row_ids: *mut i64 = ptr::null_mut();
        let mut result_count: usize = 0;

        let error = tantivy_search_nested(
            reader_ptr,
            query_bytes.as_ptr(),
            query_bytes.len(),
            &mut result_row_ids,
            &mut result_count,
        );

        assert!(error.is_null());
        assert_eq!(result_count, 0);
        assert!(result_row_ids.is_null());
    }
}
