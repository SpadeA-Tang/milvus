use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use tantivy::{
    collector::Count,
    query::{AllQuery, Query},
    schema::{Field, FieldType},
    Index, IndexReader, ReloadPolicy, Term,
};

use crate::{
    bitset_wrapper::BitsetWrapper,
    docid_collector::DocIdCollector,
    error::{Result, TantivyBindingError},
    index_reader_c::SetBitsetFn,
    milvus_id_collector::MilvusIdCollector,
    nested_query::{FieldCondition, NestedQuery, QueryLogic},
};

pub(crate) struct IndexReaderNestedWrapper {
    pub(crate) fields: Vec<Field>,
    pub(crate) field_types: Vec<FieldType>,
    pub(crate) reader: IndexReader,
    pub(crate) index: Arc<Index>,
    pub(crate) id_field: Field,
    pub(crate) set_bitset: SetBitsetFn,
}

impl IndexReaderNestedWrapper {
    pub fn from_index(index: Arc<Index>, set_bitset: SetBitsetFn) -> Result<Self> {
        let schema = index.schema();
        let fields = schema.fields().map(|(field, _)| field).collect::<Vec<_>>();
        let field_types = schema
            .fields()
            .map(|(_, field_type)| field_type.field_type().clone())
            .collect::<Vec<_>>();
        let id_field = schema.get_field("doc_id").unwrap();

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay) // OnCommitWithDelay serve for growing segment.
            .try_into()?;
        reader.reload()?;

        Ok(IndexReaderNestedWrapper {
            fields,
            field_types,
            reader,
            index: index.clone(),
            id_field,
            set_bitset,
        })
    }
}
