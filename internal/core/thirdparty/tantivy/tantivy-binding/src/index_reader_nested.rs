use std::ffi::c_void;
use std::sync::Arc;

use tantivy::{
    schema::{Field, FieldType},
    Index, IndexReader, ReloadPolicy,
};

use crate::{
    bitset_wrapper::BitsetWrapper,
    docid_collector::DocIdCollector,
    error::{Result, TantivyBindingError},
    index_reader_c::SetBitsetFn,
    nested_query::NestedQuery,
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

    /// Execute nested query and return parent document row IDs
    pub fn search_nested(&self, query: &NestedQuery, bitset: *mut c_void) -> Result<()> {
        let searcher = self.reader.searcher();
        let schema = self.index.schema();

        // Build field name to Field mapping
        let field_map: Vec<(String, Field)> = schema
            .fields()
            .map(|(field, entry)| (entry.name().to_string(), field))
            .collect();

        // Convert NestedQuery to Tantivy Query
        let tantivy_query = query.to_query(&field_map)?;

        // Execute query to collect all matching documents
        // We use TopDocs with a large limit to collect all matches
        // In production, you might want to add pagination or streaming
        searcher
            .search(
                &*tantivy_query,
                &DocIdCollector {
                    bitset_wrapper: BitsetWrapper::new(bitset, self.set_bitset),
                },
            )
            .map_err(TantivyBindingError::TantivyError)
    }
}
