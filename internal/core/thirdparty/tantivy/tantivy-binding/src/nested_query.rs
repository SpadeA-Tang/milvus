use std::ops::Bound;

use tantivy::query::{AllQuery, BooleanQuery, Occur, Query, RangeQuery, TermQuery};
use tantivy::schema::{Field, IndexRecordOption};
use tantivy::Term;

use crate::error::Result;

// Use generated protobuf code from proto module
pub use crate::proto::plan as proto;

/// Extract field name from ColumnInfo's nested_path
fn extract_field_name(column_info: &proto::ColumnInfo) -> Result<String> {
    if column_info.nested_path.is_empty() {
        return Err(crate::error::TantivyBindingError::InvalidArgument(
            "nested_path is empty in ColumnInfo".to_string(),
        ));
    }
    // The field name is the first element of nested_path (e.g., ["sub_int"] -> "sub_int")
    Ok(column_info.nested_path[0].clone())
}

/// Get field from field map by name
fn get_field<'a>(
    field_name: &str,
    field_map: &std::collections::HashMap<&'a str, Field>,
) -> Result<Field> {
    field_map.get(field_name).copied().ok_or_else(|| {
        crate::error::TantivyBindingError::InvalidArgument(format!(
            "Field '{}' not found in schema",
            field_name
        ))
    })
}

/// Check if value is numeric type (uses RangeQuery instead of TermQuery)
fn is_numeric_value(value: &proto::GenericValue) -> bool {
    use proto::generic_value::Val;
    matches!(
        &value.val,
        Some(Val::Int64Val(_)) | Some(Val::FloatVal(_)) | Some(Val::BoolVal(_))
    )
}

/// Convert GenericValue to Term for the given field
fn generic_value_to_term(field: Field, value: &proto::GenericValue) -> Result<Term> {
    use proto::generic_value::Val;
    match &value.val {
        Some(Val::BoolVal(v)) => Ok(Term::from_field_bool(field, *v)),
        Some(Val::Int64Val(v)) => Ok(Term::from_field_i64(field, *v)),
        Some(Val::FloatVal(v)) => Ok(Term::from_field_f64(field, *v)),
        Some(Val::StringVal(v)) => Ok(Term::from_field_text(field, v)),
        _ => Err(crate::error::TantivyBindingError::InvalidArgument(
            "Unsupported value type in GenericValue".to_string(),
        )),
    }
}

/// Convert UnaryRangeExpr to Tantivy Query
fn convert_unary_range_expr(
    expr: &proto::UnaryRangeExpr,
    field_map: &std::collections::HashMap<&str, Field>,
) -> Result<Box<dyn Query>> {
    let column_info = expr.column_info.as_ref().ok_or_else(|| {
        crate::error::TantivyBindingError::InvalidArgument(
            "Missing column_info in UnaryRangeExpr".to_string(),
        )
    })?;

    let field_name = extract_field_name(column_info)?;
    let field = get_field(&field_name, field_map)?;

    let value = expr.value.as_ref().ok_or_else(|| {
        crate::error::TantivyBindingError::InvalidArgument(
            "Missing value in UnaryRangeExpr".to_string(),
        )
    })?;

    let op = proto::OpType::try_from(expr.op).unwrap_or(proto::OpType::Invalid);

    match op {
        proto::OpType::Equal => {
            let term = generic_value_to_term(field, value)?;
            // For numeric types, use RangeQuery for equality (more reliable with indexed numeric fields)
            // For string types, use TermQuery
            if is_numeric_value(value) {
                Ok(Box::new(RangeQuery::new(
                    Bound::Included(term.clone()),
                    Bound::Included(term),
                )))
            } else {
                Ok(Box::new(TermQuery::new(term, IndexRecordOption::Basic)))
            }
        }
        proto::OpType::NotEqual => {
            // NotEqual: ALL AND NOT(Equal)
            // Must have a positive clause (AllQuery) for MustNot to work
            let term = generic_value_to_term(field, value)?;
            let equal_query: Box<dyn Query> = if is_numeric_value(value) {
                Box::new(RangeQuery::new(
                    Bound::Included(term.clone()),
                    Bound::Included(term),
                ))
            } else {
                Box::new(TermQuery::new(term, IndexRecordOption::Basic))
            };
            Ok(Box::new(BooleanQuery::new(vec![
                (Occur::Must, Box::new(AllQuery) as Box<dyn Query>),
                (Occur::MustNot, equal_query),
            ])))
        }
        proto::OpType::GreaterThan => {
            let term = generic_value_to_term(field, value)?;
            Ok(Box::new(RangeQuery::new(Bound::Excluded(term), Bound::Unbounded)))
        }
        proto::OpType::GreaterEqual => {
            let term = generic_value_to_term(field, value)?;
            Ok(Box::new(RangeQuery::new(Bound::Included(term), Bound::Unbounded)))
        }
        proto::OpType::LessThan => {
            let term = generic_value_to_term(field, value)?;
            Ok(Box::new(RangeQuery::new(Bound::Unbounded, Bound::Excluded(term))))
        }
        proto::OpType::LessEqual => {
            let term = generic_value_to_term(field, value)?;
            Ok(Box::new(RangeQuery::new(Bound::Unbounded, Bound::Included(term))))
        }
        _ => Err(crate::error::TantivyBindingError::InvalidArgument(format!(
            "Unsupported OpType {:?} in UnaryRangeExpr",
            op
        ))),
    }
}

/// Convert BinaryRangeExpr to Tantivy Query (range with both bounds)
fn convert_binary_range_expr(
    expr: &proto::BinaryRangeExpr,
    field_map: &std::collections::HashMap<&str, Field>,
) -> Result<Box<dyn Query>> {
    let column_info = expr.column_info.as_ref().ok_or_else(|| {
        crate::error::TantivyBindingError::InvalidArgument(
            "Missing column_info in BinaryRangeExpr".to_string(),
        )
    })?;

    let field_name = extract_field_name(column_info)?;
    let field = get_field(&field_name, field_map)?;

    let lower_value = expr.lower_value.as_ref().ok_or_else(|| {
        crate::error::TantivyBindingError::InvalidArgument(
            "Missing lower_value in BinaryRangeExpr".to_string(),
        )
    })?;

    let upper_value = expr.upper_value.as_ref().ok_or_else(|| {
        crate::error::TantivyBindingError::InvalidArgument(
            "Missing upper_value in BinaryRangeExpr".to_string(),
        )
    })?;

    let lower_term = generic_value_to_term(field, lower_value)?;
    let upper_term = generic_value_to_term(field, upper_value)?;

    let lower_bound = if expr.lower_inclusive {
        Bound::Included(lower_term)
    } else {
        Bound::Excluded(lower_term)
    };

    let upper_bound = if expr.upper_inclusive {
        Bound::Included(upper_term)
    } else {
        Bound::Excluded(upper_term)
    };

    Ok(Box::new(RangeQuery::new(lower_bound, upper_bound)))
}

/// Convert TermExpr (IN query) to Tantivy Query
fn convert_term_expr(
    expr: &proto::TermExpr,
    field_map: &std::collections::HashMap<&str, Field>,
) -> Result<Box<dyn Query>> {
    let column_info = expr.column_info.as_ref().ok_or_else(|| {
        crate::error::TantivyBindingError::InvalidArgument(
            "Missing column_info in TermExpr".to_string(),
        )
    })?;

    let field_name = extract_field_name(column_info)?;
    let field = get_field(&field_name, field_map)?;

    if expr.values.is_empty() {
        return Err(crate::error::TantivyBindingError::InvalidArgument(
            "Empty values in TermExpr".to_string(),
        ));
    }

    // Convert to OR of queries: field IN [v1, v2] -> (field == v1) OR (field == v2)
    // Use RangeQuery for numeric types, TermQuery for string types
    let mut sub_queries: Vec<Box<dyn Query>> = Vec::new();
    for value in &expr.values {
        let term = generic_value_to_term(field, value)?;
        let query: Box<dyn Query> = if is_numeric_value(value) {
            Box::new(RangeQuery::new(
                Bound::Included(term.clone()),
                Bound::Included(term),
            ))
        } else {
            Box::new(TermQuery::new(term, IndexRecordOption::Basic))
        };
        sub_queries.push(query);
    }

    Ok(Box::new(BooleanQuery::union(sub_queries)))
}

/// Convert Expr to Tantivy Query (recursive)
fn convert_expr(
    expr: &proto::Expr,
    field_map: &std::collections::HashMap<&str, Field>,
) -> Result<Box<dyn Query>> {
    use proto::expr::Expr as ExprType;

    let expr_inner = expr.expr.as_ref().ok_or_else(|| {
        crate::error::TantivyBindingError::InvalidArgument("Empty Expr".to_string())
    })?;

    match expr_inner {
        ExprType::UnaryRangeExpr(e) => convert_unary_range_expr(e, field_map),

        ExprType::BinaryRangeExpr(e) => convert_binary_range_expr(e, field_map),

        ExprType::TermExpr(e) => convert_term_expr(e, field_map),

        ExprType::BinaryExpr(e) => {
            let left = e.left.as_ref().ok_or_else(|| {
                crate::error::TantivyBindingError::InvalidArgument(
                    "Missing left in BinaryExpr".to_string(),
                )
            })?;
            let right = e.right.as_ref().ok_or_else(|| {
                crate::error::TantivyBindingError::InvalidArgument(
                    "Missing right in BinaryExpr".to_string(),
                )
            })?;

            let left_query = convert_expr(left, field_map)?;
            let right_query = convert_expr(right, field_map)?;

            let op = proto::binary_expr::BinaryOp::try_from(e.op)
                .unwrap_or(proto::binary_expr::BinaryOp::Invalid);

            match op {
                proto::binary_expr::BinaryOp::LogicalAnd => {
                    Ok(Box::new(BooleanQuery::intersection(vec![left_query, right_query])))
                }
                proto::binary_expr::BinaryOp::LogicalOr => {
                    Ok(Box::new(BooleanQuery::union(vec![left_query, right_query])))
                }
                _ => Err(crate::error::TantivyBindingError::InvalidArgument(format!(
                    "Unsupported BinaryOp {:?}",
                    op
                ))),
            }
        }

        ExprType::UnaryExpr(e) => {
            let child = e.child.as_ref().ok_or_else(|| {
                crate::error::TantivyBindingError::InvalidArgument(
                    "Missing child in UnaryExpr".to_string(),
                )
            })?;

            let child_query = convert_expr(child, field_map)?;

            let op = proto::unary_expr::UnaryOp::try_from(e.op)
                .unwrap_or(proto::unary_expr::UnaryOp::Invalid);

            match op {
                proto::unary_expr::UnaryOp::Not => {
                    // NOT: ALL AND NOT(child)
                    // Must have a positive clause (AllQuery) for MustNot to work
                    Ok(Box::new(BooleanQuery::new(vec![
                        (Occur::Must, Box::new(AllQuery) as Box<dyn Query>),
                        (Occur::MustNot, child_query),
                    ])))
                }
                _ => Err(crate::error::TantivyBindingError::InvalidArgument(format!(
                    "Unsupported UnaryOp {:?}",
                    op
                ))),
            }
        }

        _ => Err(crate::error::TantivyBindingError::InvalidArgument(format!(
            "Unsupported expression type for nested query"
        ))),
    }
}

/// Represents a nested query parsed from Expr protobuf
#[derive(Debug)]
pub struct NestedQuery {
    expr: proto::Expr,
}

impl NestedQuery {
    /// Create from protobuf bytes
    pub fn from_proto_bytes(bytes: &[u8]) -> Result<Self> {
        use prost::Message;

        let expr = proto::Expr::decode(bytes).map_err(|e| {
            crate::error::TantivyBindingError::InvalidArgument(format!(
                "Failed to decode Expr protobuf: {}",
                e
            ))
        })?;

        Ok(NestedQuery { expr })
    }

    /// Convert to Tantivy Query
    pub fn to_query(&self, fields: &[(String, Field)]) -> Result<Box<dyn Query>> {
        let field_map: std::collections::HashMap<_, _> = fields
            .iter()
            .map(|(name, field)| (name.as_str(), *field))
            .collect();

        convert_expr(&self.expr, &field_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use tantivy::schema::{Schema, FAST, STRING};

    fn create_test_schema() -> (Schema, Vec<(String, Field)>) {
        let mut schema_builder = Schema::builder();
        let int_field = schema_builder.add_i64_field("sub_int", FAST);
        let str_field = schema_builder.add_text_field("sub_str", STRING);
        let schema = schema_builder.build();

        let fields = vec![
            ("sub_int".to_string(), int_field),
            ("sub_str".to_string(), str_field),
        ];

        (schema, fields)
    }

    #[test]
    fn test_unary_range_equal() {
        let (_schema, fields) = create_test_schema();

        // Create UnaryRangeExpr: sub_int == 100
        let expr = proto::Expr {
            expr: Some(proto::expr::Expr::UnaryRangeExpr(proto::UnaryRangeExpr {
                column_info: Some(proto::ColumnInfo {
                    field_id: 0,
                    data_type: 0,
                    nested_path: vec!["sub_int".to_string()],
                    ..Default::default()
                }),
                op: proto::OpType::Equal as i32,
                value: Some(proto::GenericValue {
                    val: Some(proto::generic_value::Val::Int64Val(100)),
                }),
                ..Default::default()
            })),
            is_template: false,
        };

        let bytes = expr.encode_to_vec();
        let nested_query = NestedQuery::from_proto_bytes(&bytes).unwrap();
        let query = nested_query.to_query(&fields);
        assert!(query.is_ok());
    }

    #[test]
    fn test_binary_expr_and() {
        let (_schema, fields) = create_test_schema();

        // Create: (sub_int > 10) AND (sub_str == "hello")
        let expr = proto::Expr {
            expr: Some(proto::expr::Expr::BinaryExpr(Box::new(proto::BinaryExpr {
                op: proto::binary_expr::BinaryOp::LogicalAnd as i32,
                left: Some(Box::new(proto::Expr {
                    expr: Some(proto::expr::Expr::UnaryRangeExpr(proto::UnaryRangeExpr {
                        column_info: Some(proto::ColumnInfo {
                            nested_path: vec!["sub_int".to_string()],
                            ..Default::default()
                        }),
                        op: proto::OpType::GreaterThan as i32,
                        value: Some(proto::GenericValue {
                            val: Some(proto::generic_value::Val::Int64Val(10)),
                        }),
                        ..Default::default()
                    })),
                    is_template: false,
                })),
                right: Some(Box::new(proto::Expr {
                    expr: Some(proto::expr::Expr::UnaryRangeExpr(proto::UnaryRangeExpr {
                        column_info: Some(proto::ColumnInfo {
                            nested_path: vec!["sub_str".to_string()],
                            ..Default::default()
                        }),
                        op: proto::OpType::Equal as i32,
                        value: Some(proto::GenericValue {
                            val: Some(proto::generic_value::Val::StringVal("hello".to_string())),
                        }),
                        ..Default::default()
                    })),
                    is_template: false,
                })),
            }))),
            is_template: false,
        };

        let bytes = expr.encode_to_vec();
        let nested_query = NestedQuery::from_proto_bytes(&bytes).unwrap();
        let query = nested_query.to_query(&fields);
        assert!(query.is_ok());
    }

    #[test]
    fn test_term_expr_in() {
        let (_schema, fields) = create_test_schema();

        // Create: sub_int IN [1, 2, 3]
        let expr = proto::Expr {
            expr: Some(proto::expr::Expr::TermExpr(proto::TermExpr {
                column_info: Some(proto::ColumnInfo {
                    nested_path: vec!["sub_int".to_string()],
                    ..Default::default()
                }),
                values: vec![
                    proto::GenericValue {
                        val: Some(proto::generic_value::Val::Int64Val(1)),
                    },
                    proto::GenericValue {
                        val: Some(proto::generic_value::Val::Int64Val(2)),
                    },
                    proto::GenericValue {
                        val: Some(proto::generic_value::Val::Int64Val(3)),
                    },
                ],
                ..Default::default()
            })),
            is_template: false,
        };

        let bytes = expr.encode_to_vec();
        let nested_query = NestedQuery::from_proto_bytes(&bytes).unwrap();
        let query = nested_query.to_query(&fields);
        assert!(query.is_ok());
    }

    #[test]
    fn test_unary_range_not_equal() {
        let (_schema, fields) = create_test_schema();

        // Create: sub_int != 100
        let expr = proto::Expr {
            expr: Some(proto::expr::Expr::UnaryRangeExpr(proto::UnaryRangeExpr {
                column_info: Some(proto::ColumnInfo {
                    nested_path: vec!["sub_int".to_string()],
                    ..Default::default()
                }),
                op: proto::OpType::NotEqual as i32,
                value: Some(proto::GenericValue {
                    val: Some(proto::generic_value::Val::Int64Val(100)),
                }),
                ..Default::default()
            })),
            is_template: false,
        };

        let bytes = expr.encode_to_vec();
        let nested_query = NestedQuery::from_proto_bytes(&bytes).unwrap();
        let query = nested_query.to_query(&fields);
        assert!(query.is_ok());
    }

    #[test]
    fn test_unary_expr_not() {
        let (_schema, fields) = create_test_schema();

        // Create: NOT(sub_int == 100)
        let expr = proto::Expr {
            expr: Some(proto::expr::Expr::UnaryExpr(Box::new(proto::UnaryExpr {
                op: proto::unary_expr::UnaryOp::Not as i32,
                child: Some(Box::new(proto::Expr {
                    expr: Some(proto::expr::Expr::UnaryRangeExpr(proto::UnaryRangeExpr {
                        column_info: Some(proto::ColumnInfo {
                            nested_path: vec!["sub_int".to_string()],
                            ..Default::default()
                        }),
                        op: proto::OpType::Equal as i32,
                        value: Some(proto::GenericValue {
                            val: Some(proto::generic_value::Val::Int64Val(100)),
                        }),
                        ..Default::default()
                    })),
                    is_template: false,
                })),
            }))),
            is_template: false,
        };

        let bytes = expr.encode_to_vec();
        let nested_query = NestedQuery::from_proto_bytes(&bytes).unwrap();
        let query = nested_query.to_query(&fields);
        assert!(query.is_ok());
    }

    #[test]
    fn test_binary_range_expr() {
        let (_schema, fields) = create_test_schema();

        // Create: 10 <= sub_int < 100
        let expr = proto::Expr {
            expr: Some(proto::expr::Expr::BinaryRangeExpr(proto::BinaryRangeExpr {
                column_info: Some(proto::ColumnInfo {
                    nested_path: vec!["sub_int".to_string()],
                    ..Default::default()
                }),
                lower_inclusive: true,
                upper_inclusive: false,
                lower_value: Some(proto::GenericValue {
                    val: Some(proto::generic_value::Val::Int64Val(10)),
                }),
                upper_value: Some(proto::GenericValue {
                    val: Some(proto::generic_value::Val::Int64Val(100)),
                }),
                ..Default::default()
            })),
            is_template: false,
        };

        let bytes = expr.encode_to_vec();
        let nested_query = NestedQuery::from_proto_bytes(&bytes).unwrap();
        let query = nested_query.to_query(&fields);
        assert!(query.is_ok());
    }
}
