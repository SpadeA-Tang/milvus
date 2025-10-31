use std::ops::Bound;

use tantivy::query::{BooleanQuery, Query, RangeQuery, TermQuery};
use tantivy::schema::{Field, IndexRecordOption};
use tantivy::Term;

use crate::error::Result;

// Include generated protobuf code
#[allow(clippy::all)]
pub mod proto {
    include!("proto/milvus.proto.plan.rs");
}

/// Represents a query expression tree (recursive definition)
#[derive(Debug, Clone)]
pub enum QueryExpr {
    /// Leaf node: single field condition
    Condition(FieldCondition),

    /// Internal node: logical combination
    Logical {
        op: LogicalOp,
        children: Vec<QueryExpr>,
    },
}

/// Represents a query condition for a single field
#[derive(Debug, Clone)]
pub enum FieldCondition {
    TermI64 {
        field_name: String,
        value: i64,
    },
    TermF64 {
        field_name: String,
        value: f64,
    },
    TermBool {
        field_name: String,
        value: bool,
    },
    TermKeyword {
        field_name: String,
        value: String,
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

/// Logic for combining multiple expressions
#[derive(Debug, Clone)]
pub enum LogicalOp {
    And,
    Or,
}

/// Represents a nested query (root of query expression tree)
#[derive(Debug, Clone)]
pub struct NestedQuery {
    pub expr: QueryExpr,
}

impl QueryExpr {
    /// Convert from protobuf TantivyQueryExpr
    pub fn from_proto(proto: &proto::TantivyQueryExpr) -> Result<Self> {
        match &proto.expr {
            Some(proto::tantivy_query_expr::Expr::Condition(c)) => {
                Ok(QueryExpr::Condition(FieldCondition::from_proto(c)?))
            }
            Some(proto::tantivy_query_expr::Expr::Logical(l)) => {
                let op = match proto::tantivy_logical_expr::Op::try_from(l.op) {
                    Ok(proto::tantivy_logical_expr::Op::And) => LogicalOp::And,
                    Ok(proto::tantivy_logical_expr::Op::Or) => LogicalOp::Or,
                    _ => {
                        return Err(crate::error::TantivyBindingError::InvalidArgument(
                            format!("Unknown logical op: {}", l.op)
                        ));
                    }
                };

                let mut children = Vec::new();
                for child_proto in &l.children {
                    children.push(QueryExpr::from_proto(child_proto)?);
                }

                Ok(QueryExpr::Logical { op, children })
            }
            None => Err(crate::error::TantivyBindingError::InvalidArgument(
                "Empty TantivyQueryExpr".to_string()
            )),
        }
    }

    /// Recursively convert to Tantivy Query
    pub fn to_query(&self, fields: &[(String, Field)]) -> Result<Box<dyn Query>> {
        match self {
            QueryExpr::Condition(condition) => condition.to_query(fields),
            QueryExpr::Logical { op, children } => {
                let mut sub_queries = Vec::new();
                for child in children {
                    sub_queries.push(child.to_query(fields)?);
                }

                let boolean_query = match op {
                    LogicalOp::And => BooleanQuery::intersection(sub_queries),
                    LogicalOp::Or => BooleanQuery::union(sub_queries),
                };

                Ok(Box::new(boolean_query))
            }
        }
    }
}

impl FieldCondition {
    /// Convert from protobuf TantivyFieldCondition
    pub fn from_proto(proto: &proto::TantivyFieldCondition) -> Result<Self> {
        use proto::tantivy_field_condition::ConditionType;

        let field_name = proto.field_name.clone();

        match &proto.condition_type {
            Some(ConditionType::TermI64(value)) => {
                Ok(FieldCondition::TermI64 { field_name, value: *value })
            }
            Some(ConditionType::TermF64(value)) => {
                Ok(FieldCondition::TermF64 { field_name, value: *value })
            }
            Some(ConditionType::TermBool(value)) => {
                Ok(FieldCondition::TermBool { field_name, value: *value })
            }
            Some(ConditionType::TermKeyword(value)) => {
                Ok(FieldCondition::TermKeyword { field_name, value: value.clone() })
            }
            Some(ConditionType::RangeI64(r)) => {
                Ok(FieldCondition::RangeI64 {
                    field_name,
                    lower: r.lower,
                    upper: r.upper,
                    lb_inclusive: r.lower_inclusive,
                    ub_inclusive: r.upper_inclusive,
                })
            }
            Some(ConditionType::RangeF64(r)) => {
                Ok(FieldCondition::RangeF64 {
                    field_name,
                    lower: r.lower,
                    upper: r.upper,
                    lb_inclusive: r.lower_inclusive,
                    ub_inclusive: r.upper_inclusive,
                })
            }
            Some(ConditionType::RangeKeyword(r)) => {
                Ok(FieldCondition::RangeKeyword {
                    field_name,
                    lower: r.lower.clone(),
                    upper: r.upper.clone(),
                    lb_inclusive: r.lower_inclusive,
                    ub_inclusive: r.upper_inclusive,
                })
            }
            None => Err(crate::error::TantivyBindingError::InvalidArgument(
                "Empty condition_type in TantivyFieldCondition".to_string()
            )),
        }
    }

    /// Convert field condition to tantivy Query
    pub fn to_query(&self, fields: &[(String, Field)]) -> Result<Box<dyn Query>> {
        let field_map: std::collections::HashMap<_, _> = fields
            .iter()
            .map(|(name, field)| (name.as_str(), *field))
            .collect();

        match self {
            FieldCondition::TermI64 { field_name, value } => {
                let field = field_map.get(field_name.as_str()).ok_or_else(|| {
                    crate::error::TantivyBindingError::InvalidArgument(format!(
                        "Field {} not found",
                        field_name
                    ))
                })?;
                let term = Term::from_field_i64(*field, *value);
                Ok(Box::new(TermQuery::new(term, IndexRecordOption::Basic)))
            }
            FieldCondition::TermF64 { field_name, value } => {
                let field = field_map.get(field_name.as_str()).ok_or_else(|| {
                    crate::error::TantivyBindingError::InvalidArgument(format!(
                        "Field {} not found",
                        field_name
                    ))
                })?;
                let term = Term::from_field_f64(*field, *value);
                Ok(Box::new(TermQuery::new(term, IndexRecordOption::Basic)))
            }
            FieldCondition::TermBool { field_name, value } => {
                let field = field_map.get(field_name.as_str()).ok_or_else(|| {
                    crate::error::TantivyBindingError::InvalidArgument(format!(
                        "Field {} not found",
                        field_name
                    ))
                })?;
                let term = Term::from_field_bool(*field, *value);
                Ok(Box::new(TermQuery::new(term, IndexRecordOption::Basic)))
            }
            FieldCondition::TermKeyword { field_name, value } => {
                let field = field_map.get(field_name.as_str()).ok_or_else(|| {
                    crate::error::TantivyBindingError::InvalidArgument(format!(
                        "Field {} not found",
                        field_name
                    ))
                })?;
                let term = Term::from_field_text(*field, value);
                Ok(Box::new(TermQuery::new(term, IndexRecordOption::Basic)))
            }
            FieldCondition::RangeI64 {
                field_name,
                lower,
                upper,
                lb_inclusive,
                ub_inclusive,
            } => {
                let field = field_map.get(field_name.as_str()).ok_or_else(|| {
                    crate::error::TantivyBindingError::InvalidArgument(format!(
                        "Field {} not found",
                        field_name
                    ))
                })?;

                let lower_bound = match lower {
                    Some(val) => {
                        let term = Term::from_field_i64(*field, *val);
                        if *lb_inclusive {
                            Bound::Included(term)
                        } else {
                            Bound::Excluded(term)
                        }
                    }
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
                    }
                    None => Bound::Unbounded,
                };

                Ok(Box::new(RangeQuery::new(lower_bound, upper_bound)))
            }
            FieldCondition::RangeF64 {
                field_name,
                lower,
                upper,
                lb_inclusive,
                ub_inclusive,
            } => {
                let field = field_map.get(field_name.as_str()).ok_or_else(|| {
                    crate::error::TantivyBindingError::InvalidArgument(format!(
                        "Field {} not found",
                        field_name
                    ))
                })?;

                let lower_bound = match lower {
                    Some(val) => {
                        let term = Term::from_field_f64(*field, *val);
                        if *lb_inclusive {
                            Bound::Included(term)
                        } else {
                            Bound::Excluded(term)
                        }
                    }
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
                    }
                    None => Bound::Unbounded,
                };

                Ok(Box::new(RangeQuery::new(lower_bound, upper_bound)))
            }
            FieldCondition::RangeKeyword {
                field_name,
                lower,
                upper,
                lb_inclusive,
                ub_inclusive,
            } => {
                let field = field_map.get(field_name.as_str()).ok_or_else(|| {
                    crate::error::TantivyBindingError::InvalidArgument(format!(
                        "Field {} not found",
                        field_name
                    ))
                })?;

                let lower_bound = match lower {
                    Some(val) => {
                        let term = Term::from_field_text(*field, val);
                        if *lb_inclusive {
                            Bound::Included(term)
                        } else {
                            Bound::Excluded(term)
                        }
                    }
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
                    }
                    None => Bound::Unbounded,
                };

                Ok(Box::new(RangeQuery::new(lower_bound, upper_bound)))
            }
        }
    }
}

impl NestedQuery {
    /// Create from protobuf bytes
    pub fn from_proto_bytes(bytes: &[u8]) -> Result<Self> {
        use prost::Message;

        let proto = proto::TantivyQueryExpr::decode(bytes)
            .map_err(|e| crate::error::TantivyBindingError::InvalidArgument(
                format!("Failed to decode protobuf: {}", e)
            ))?;

        let expr = QueryExpr::from_proto(&proto)?;
        Ok(NestedQuery { expr })
    }

    /// Convert to Tantivy Query
    pub fn to_query(&self, fields: &[(String, Field)]) -> Result<Box<dyn Query>> {
        self.expr.to_query(fields)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tantivy::schema::{Field, Schema, FAST};

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
    fn test_recursive_query_expr() {
        let mut schema = Schema::builder();
        let int_field = schema.add_i64_field("int_field", FAST);
        let str_field = schema.add_text_field("str_field", tantivy::schema::STRING);
        let _schema = schema.build();

        let fields = vec![
            ("int_field".to_string(), int_field),
            ("str_field".to_string(), str_field),
        ];

        // Test: (int_field == 42) AND (str_field == "hello")
        let nested_query = NestedQuery {
            expr: QueryExpr::Logical {
                op: LogicalOp::And,
                children: vec![
                    QueryExpr::Condition(FieldCondition::TermI64 {
                        field_name: "int_field".to_string(),
                        value: 42,
                    }),
                    QueryExpr::Condition(FieldCondition::TermKeyword {
                        field_name: "str_field".to_string(),
                        value: "hello".to_string(),
                    }),
                ],
            },
        };

        let query = nested_query.to_query(&fields);
        assert!(query.is_ok());
    }
}
