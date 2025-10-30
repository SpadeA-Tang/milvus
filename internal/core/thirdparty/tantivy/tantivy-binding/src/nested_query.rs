use std::ffi::{c_char, c_void};
use std::ops::Bound;

use tantivy::query::{BooleanQuery, Query, RangeQuery, TermQuery};
use tantivy::schema::{Field, IndexRecordOption};
use tantivy::{Term, TantivyDocument};

use crate::error::Result;
use crate::util::c_ptr_to_str;

/// Represents a query condition for a single field
#[derive(Debug, Clone)]
pub enum FieldCondition {
    TermI64 {
        field_name: String,
        value: i64
    },
    TermF64 {
        field_name: String,
        value: f64
    },
    TermBool {
        field_name: String,
        value: bool
    },
    TermKeyword {
        field_name: String,
        value: String
    },
    RangeI64 {
        field_name: String,
        lower: Option<i64>,
        upper: Option<i64>,
        lb_inclusive: bool,
        ub_inclusive: bool,
    },
    RangeF64 {
        field_name: String,
        lower: Option<f64>,
        upper: Option<f64>,
        lb_inclusive: bool,
        ub_inclusive: bool,
    },
    RangeKeyword {
        field_name: String,
        lower: Option<String>,
        upper: Option<String>,
        lb_inclusive: bool,
        ub_inclusive: bool,
    },
}

/// Logic for combining multiple field conditions
#[derive(Debug, Clone)]
pub enum QueryLogic {
    And,  // All conditions must be satisfied
    Or,   // At least one condition must be satisfied
}

/// Represents a nested query with multiple field conditions
#[derive(Debug, Clone)]
pub struct NestedQuery {
    pub conditions: Vec<FieldCondition>,
    pub logic: QueryLogic,
}

impl FieldCondition {
    /// Convert field condition to tantivy Query
    pub fn to_query(&self, fields: &[(String, Field)]) -> Result<Box<dyn Query>> {
        let field_map: std::collections::HashMap<_, _> = fields.iter()
            .map(|(name, field)| (name.as_str(), *field))
            .collect();

        match self {
            FieldCondition::TermI64 { field_name, value } => {
                let field = field_map.get(field_name.as_str())
                    .ok_or_else(|| crate::error::TantivyBindingError::InvalidArgument(
                        format!("Field {} not found", field_name)
                    ))?;
                let term = Term::from_field_i64(*field, *value);
                Ok(Box::new(TermQuery::new(term, IndexRecordOption::Basic)))
            },
            FieldCondition::TermF64 { field_name, value } => {
                let field = field_map.get(field_name.as_str())
                    .ok_or_else(|| crate::error::TantivyBindingError::InvalidArgument(
                        format!("Field {} not found", field_name)
                    ))?;
                let term = Term::from_field_f64(*field, *value);
                Ok(Box::new(TermQuery::new(term, IndexRecordOption::Basic)))
            },
            FieldCondition::TermBool { field_name, value } => {
                let field = field_map.get(field_name.as_str())
                    .ok_or_else(|| crate::error::TantivyBindingError::InvalidArgument(
                        format!("Field {} not found", field_name)
                    ))?;
                let term = Term::from_field_bool(*field, *value);
                Ok(Box::new(TermQuery::new(term, IndexRecordOption::Basic)))
            },
            FieldCondition::TermKeyword { field_name, value } => {
                let field = field_map.get(field_name.as_str())
                    .ok_or_else(|| crate::error::TantivyBindingError::InvalidArgument(
                        format!("Field {} not found", field_name)
                    ))?;
                let term = Term::from_field_text(*field, value);
                Ok(Box::new(TermQuery::new(term, IndexRecordOption::Basic)))
            },
            FieldCondition::RangeI64 { field_name, lower, upper, lb_inclusive, ub_inclusive } => {
                let field = field_map.get(field_name.as_str())
                    .ok_or_else(|| crate::error::TantivyBindingError::InvalidArgument(
                        format!("Field {} not found", field_name)
                    ))?;

                let lower_bound = match lower {
                    Some(val) => {
                        let term = Term::from_field_i64(*field, *val);
                        if *lb_inclusive {
                            Bound::Included(term)
                        } else {
                            Bound::Excluded(term)
                        }
                    },
                    None => Bound::Unbounded,
                };

                let upper_bound = match upper {
                    Some(val) => {
                        let term = Term::from_field_i64(*field, *val);
                        if *ub_inclusive {
                            Bound::Included(term)
                        } else {
                            Bound::Excluded(term)
                        }
                    },
                    None => Bound::Unbounded,
                };

                Ok(Box::new(RangeQuery::new(lower_bound, upper_bound)))
            },
            FieldCondition::RangeF64 { field_name, lower, upper, lb_inclusive, ub_inclusive } => {
                let field = field_map.get(field_name.as_str())
                    .ok_or_else(|| crate::error::TantivyBindingError::InvalidArgument(
                        format!("Field {} not found", field_name)
                    ))?;

                let lower_bound = match lower {
                    Some(val) => {
                        let term = Term::from_field_f64(*field, *val);
                        if *lb_inclusive {
                            Bound::Included(term)
                        } else {
                            Bound::Excluded(term)
                        }
                    },
                    None => Bound::Unbounded,
                };

                let upper_bound = match upper {
                    Some(val) => {
                        let term = Term::from_field_f64(*field, *val);
                        if *ub_inclusive {
                            Bound::Included(term)
                        } else {
                            Bound::Excluded(term)
                        }
                    },
                    None => Bound::Unbounded,
                };

                Ok(Box::new(RangeQuery::new(lower_bound, upper_bound)))
            },
            FieldCondition::RangeKeyword { field_name, lower, upper, lb_inclusive, ub_inclusive } => {
                let field = field_map.get(field_name.as_str())
                    .ok_or_else(|| crate::error::TantivyBindingError::InvalidArgument(
                        format!("Field {} not found", field_name)
                    ))?;

                let lower_bound = match lower {
                    Some(val) => {
                        let term = Term::from_field_text(*field, val);
                        if *lb_inclusive {
                            Bound::Included(term)
                        } else {
                            Bound::Excluded(term)
                        }
                    },
                    None => Bound::Unbounded,
                };

                let upper_bound = match upper {
                    Some(val) => {
                        let term = Term::from_field_text(*field, val);
                        if *ub_inclusive {
                            Bound::Included(term)
                        } else {
                            Bound::Excluded(term)
                        }
                    },
                    None => Bound::Unbounded,
                };

                Ok(Box::new(RangeQuery::new(lower_bound, upper_bound)))
            },
        }
    }
}

impl NestedQuery {
    /// Convert nested query to tantivy BooleanQuery
    pub fn to_boolean_query(&self, fields: &[(String, Field)]) -> Result<Box<dyn Query>> {
        let mut queries: Vec<Box<dyn Query>> = Vec::new();

        for condition in &self.conditions {
            queries.push(condition.to_query(fields)?);
        }

        let boolean_query = match self.logic {
            QueryLogic::And => BooleanQuery::intersection(queries),
            QueryLogic::Or => BooleanQuery::union(queries),
        };

        Ok(Box::new(boolean_query))
    }
}

/// Parse a nested query from C-style parameters
/// This is a simplified version - you may want to use protobuf or JSON for more complex queries
pub unsafe fn parse_nested_query_from_c(
    field_names: *const *const c_char,
    condition_types: *const u8,  // 0=TermI64, 1=TermKeyword, 2=RangeI64, etc.
    values: *const *const c_void,
    num_conditions: usize,
    logic: u8,  // 0=And, 1=Or
) -> Result<NestedQuery> {
    let mut conditions = Vec::new();

    for i in 0..num_conditions {
        let field_name = c_ptr_to_str(*field_names.add(i))?;
        let condition_type = *condition_types.add(i);
        let value_ptr = *values.add(i);

        let condition = match condition_type {
            0 => { // TermI64
                let value = *(value_ptr as *const i64);
                FieldCondition::TermI64 {
                    field_name: field_name.to_string(),
                    value,
                }
            },
            1 => { // TermKeyword
                let value = c_ptr_to_str(value_ptr as *const c_char)?;
                FieldCondition::TermKeyword {
                    field_name: field_name.to_string(),
                    value: value.to_string(),
                }
            },
            // Add more condition types as needed...
            _ => {
                return Err(crate::error::TantivyBindingError::InvalidArgument(
                    format!("Unknown condition type: {}", condition_type)
                ));
            }
        };

        conditions.push(condition);
    }

    let logic = match logic {
        0 => QueryLogic::And,
        1 => QueryLogic::Or,
        _ => {
            return Err(crate::error::TantivyBindingError::InvalidArgument(
                format!("Unknown query logic: {}", logic)
            ));
        }
    };

    Ok(NestedQuery { conditions, logic })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tantivy::schema::{Schema, Field, FAST};

    #[test]
    fn test_field_condition_to_query() {
        let mut schema = Schema::builder();
        let int_field = schema.add_i64_field("int_field", FAST);
        let str_field = schema.add_text_field("str_field", tantivy::schema::STRING);
        let _schema = schema.build();

        let fields = vec![
            ("int_field".to_string(), int_field),
            ("str_field".to_string(), str_field),
        ];

        // Test TermI64
        let condition = FieldCondition::TermI64 {
            field_name: "int_field".to_string(),
            value: 42,
        };
        let query = condition.to_query(&fields);
        assert!(query.is_ok());

        // Test RangeI64
        let condition = FieldCondition::RangeI64 {
            field_name: "int_field".to_string(),
            lower: Some(10),
            upper: Some(100),
            lb_inclusive: true,
            ub_inclusive: false,
        };
        let query = condition.to_query(&fields);
        assert!(query.is_ok());
    }

    #[test]
    fn test_nested_query_to_boolean_query() {
        let mut schema = Schema::builder();
        let int_field = schema.add_i64_field("int_field", FAST);
        let str_field = schema.add_text_field("str_field", tantivy::schema::STRING);
        let _schema = schema.build();

        let fields = vec![
            ("int_field".to_string(), int_field),
            ("str_field".to_string(), str_field),
        ];

        let nested_query = NestedQuery {
            conditions: vec![
                FieldCondition::TermI64 {
                    field_name: "int_field".to_string(),
                    value: 42,
                },
                FieldCondition::TermKeyword {
                    field_name: "str_field".to_string(),
                    value: "hello".to_string(),
                },
            ],
            logic: QueryLogic::And,
        };

        let query = nested_query.to_boolean_query(&fields);
        assert!(query.is_ok());
    }
}