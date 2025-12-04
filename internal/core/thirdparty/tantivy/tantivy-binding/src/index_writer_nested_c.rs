use std::ffi::{c_char, c_void};
use std::ptr;

use crate::{
    data_type::TantivyDataType,
    error::{Result, TantivyBindingError},
    index_reader_c::SetBitsetFn,
    index_writer_nested::IndexWriterNestedWrapper,
    util::{c_ptr_to_str, create_binding, create_error, free_binding},
};

/// Create a nested index writer for struct fields
///
/// # Arguments
/// * `struct_name` - Name of the struct (for logging/debugging)
/// * `field_names` - Array of field name pointers
/// * `data_types` - Array of data types for each field
/// * `num_fields` - Number of fields
/// * `path` - Index directory path
/// * `num_threads` - Number of writer threads
/// * `overall_memory_budget_in_bytes` - Memory budget
///
/// # Returns
/// * Error message string if failed, NULL if succeeded
/// * `result_ptr` receives the writer pointer
///
/// # Safety
/// Caller must free the writer using `tantivy_free_nested_index_writer`
#[no_mangle]
pub extern "C" fn tantivy_create_nested_index(
    struct_name: *const c_char,
    field_names: *const *const c_char,
    data_types: *const TantivyDataType,
    num_fields: usize,
    path: *const c_char,
    num_threads: usize,
    overall_memory_budget_in_bytes: usize,
    result_ptr: *mut *mut c_void,
) -> *mut c_char {
    let result = || -> Result<*mut c_void> {
        // Validate inputs
        if struct_name.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "struct_name is null".to_string(),
            ));
        }
        if field_names.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "field_names is null".to_string(),
            ));
        }
        if data_types.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "data_types is null".to_string(),
            ));
        }
        if path.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "path is null".to_string(),
            ));
        }
        if result_ptr.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "result_ptr is null".to_string(),
            ));
        }

        let struct_name_str = c_ptr_to_str(struct_name)?;
        let path_str = c_ptr_to_str(path)?;

        // Convert field names
        let field_name_strs: Vec<&str> = (0..num_fields)
            .map(|i| unsafe {
                let name_ptr = *field_names.add(i);
                c_ptr_to_str(name_ptr)
            })
            .collect::<Result<Vec<&str>>>()?;

        // Convert data types
        let data_type_vec: Vec<TantivyDataType> =
            unsafe { std::slice::from_raw_parts(data_types, num_fields).to_vec() };

        let wrapper = IndexWriterNestedWrapper::new(
            struct_name_str,
            &field_name_strs,
            &data_type_vec,
            path_str,
            num_threads,
            overall_memory_budget_in_bytes,
        )?;

        Ok(create_binding(wrapper))
    };

    match result() {
        Ok(ptr) => {
            unsafe { *result_ptr = ptr };
            ptr::null_mut()
        }
        Err(e) => create_error(&e.to_string()),
    }
}

/// Add nested documents for a row
///
/// # Arguments
/// * `writer_ptr` - Pointer to IndexWriterNestedWrapper
/// * `row_id` - The parent row ID
/// * `field_data` - Array of pointers to field data arrays
/// * `field_count` - Number of fields
/// * `array_count` - Number of array elements (nested docs) to add
///
/// # Returns
/// * Error message string if failed, NULL if succeeded
#[no_mangle]
pub extern "C" fn tantivy_nested_index_add_documents(
    writer_ptr: *mut c_void,
    row_id: i64,
    field_data: *const *const c_void,
    field_count: usize,
    array_count: usize,
) -> *mut c_char {
    let result = || -> Result<()> {
        if writer_ptr.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "writer_ptr is null".to_string(),
            ));
        }
        if field_data.is_null() && field_count > 0 {
            return Err(TantivyBindingError::InvalidArgument(
                "field_data is null".to_string(),
            ));
        }

        let writer = unsafe { &mut *(writer_ptr as *mut IndexWriterNestedWrapper) };
        let field_data_slice = unsafe { std::slice::from_raw_parts(field_data, field_count) };

        writer.add_nested_documents(row_id, field_data_slice, field_count, array_count)
    };

    match result() {
        Ok(_) => ptr::null_mut(),
        Err(e) => create_error(&e.to_string()),
    }
}

/// Commit the nested index
#[no_mangle]
pub extern "C" fn tantivy_nested_index_commit(writer_ptr: *mut c_void) -> *mut c_char {
    let result = || -> Result<()> {
        if writer_ptr.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "writer_ptr is null".to_string(),
            ));
        }

        let writer = unsafe { &mut *(writer_ptr as *mut IndexWriterNestedWrapper) };
        writer.commit()
    };

    match result() {
        Ok(_) => ptr::null_mut(),
        Err(e) => create_error(&e.to_string()),
    }
}

/// Create a reader from the nested index writer
///
/// # Arguments
/// * `writer_ptr` - Pointer to IndexWriterNestedWrapper
/// * `set_bitset` - Callback function to set bitset
/// * `result_ptr` - Receives the reader pointer
///
/// # Returns
/// * Error message string if failed, NULL if succeeded
#[no_mangle]
pub extern "C" fn tantivy_nested_create_reader_from_writer(
    writer_ptr: *mut c_void,
    set_bitset: SetBitsetFn,
    result_ptr: *mut *mut c_void,
) -> *mut c_char {
    let result = || -> Result<*mut c_void> {
        if writer_ptr.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "writer_ptr is null".to_string(),
            ));
        }
        if result_ptr.is_null() {
            return Err(TantivyBindingError::InvalidArgument(
                "result_ptr is null".to_string(),
            ));
        }

        let writer = unsafe { &*(writer_ptr as *const IndexWriterNestedWrapper) };
        let reader = writer.create_reader(set_bitset)?;

        Ok(create_binding(reader))
    };

    match result() {
        Ok(ptr) => {
            unsafe { *result_ptr = ptr };
            ptr::null_mut()
        }
        Err(e) => create_error(&e.to_string()),
    }
}

/// Free the nested index writer
#[no_mangle]
pub extern "C" fn tantivy_free_nested_index_writer(ptr: *mut c_void) {
    free_binding::<IndexWriterNestedWrapper>(ptr);
}

/// Free the nested index reader
#[no_mangle]
pub extern "C" fn tantivy_free_nested_index_reader(ptr: *mut c_void) {
    use crate::index_reader_nested::IndexReaderNestedWrapper;
    free_binding::<IndexReaderNestedWrapper>(ptr);
}
