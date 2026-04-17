use super::types::{SqlLogic, SqlOperator, SqlStatementInternal, SqlStatementType, SqlValue};
use anyhow::Result;

/// Converts a SqlOperator enum to its SQL string representation
pub fn sql_operator_to_string(op: &SqlOperator) -> &'static str {
    match op {
        SqlOperator::Equals => "=",
        SqlOperator::NotEquals => "!=",
        SqlOperator::GreaterThan => ">",
        SqlOperator::LessThan => "<",
        SqlOperator::GreaterThanOrEquals => ">=",
        SqlOperator::LessThanOrEquals => "<=",
        SqlOperator::Like => "LIKE",
        SqlOperator::NotLike => "NOT LIKE",
        SqlOperator::Exists => "IS NOT NULL",
    }
}

/// Builds a SQL WHERE clause string and parameters from a SqlStatement structure
pub fn build_sql_from_statement(
    statement: &SqlStatementInternal,
    params: &mut Vec<Box<dyn rusqlite::ToSql>>,
) -> Result<String> {
    match statement.statement_type {
        SqlStatementType::Empty => Ok(String::new()),
        SqlStatementType::Condition => {
            if let Some(condition) = &statement.condition {
                let operator = sql_operator_to_string(&condition.operator);
                // Keys starting with "$." are JSON paths into metadata;
                // bare names are direct column references (must be simple identifiers).
                let is_json = condition.key.starts_with("$.");
                let col_expr = if is_json {
                    params.push(Box::new(condition.key.clone()));
                    "json_extract(metadata, ?)".to_string()
                } else {
                    if !matches!(condition.key.as_str(), "date" | "uuid") {
                        return Err(anyhow::anyhow!(
                            "Invalid column name: {}",
                            condition.key
                        ));
                    }
                    format!("\"{}\"", condition.key)
                };

                if matches!(condition.operator, SqlOperator::Exists) {
                    Ok(format!("{col_expr} {operator}"))
                } else if let Some(value) = &condition.value {
                    match value {
                        SqlValue::String(s) => {
                            params.push(Box::new(s.clone()));
                        }
                        SqlValue::Number(n) => {
                            params.push(Box::new(*n));
                        }
                    }
                    Ok(format!("{col_expr} {operator} ?"))
                } else {
                    Err(anyhow::anyhow!(
                        "Condition with operator {:?} requires a value",
                        operator
                    ))
                }
            } else {
                Err(anyhow::anyhow!("Condition type requires a condition field"))
            }
        }
        SqlStatementType::Group => {
            if let (Some(logic), Some(statements)) = (&statement.logic, &statement.statements) {
                let logic_op = match logic {
                    SqlLogic::And => "AND",
                    SqlLogic::Or => "OR",
                };

                let mut sql_parts = Vec::new();
                for stmt in statements {
                    let sql = build_sql_from_statement(stmt, params)?;
                    if !sql.is_empty() {
                        sql_parts.push(sql);
                    }
                }

                if sql_parts.is_empty() {
                    Ok(String::new())
                } else if sql_parts.len() == 1 {
                    Ok(sql_parts[0].clone())
                } else {
                    Ok(format!("({})", sql_parts.join(&format!(" {} ", logic_op))))
                }
            } else {
                Err(anyhow::anyhow!(
                    "Group type requires logic and statements fields"
                ))
            }
        }
    }
}

/// Builds a complete SQL filter clause and parameters from an optional SqlStatement
pub fn build_filter_sql_and_params(
    sql_filter: Option<&SqlStatementInternal>,
) -> Result<(String, Vec<Box<dyn rusqlite::ToSql>>)> {
    let mut params = Vec::new();

    let sql = if let Some(filter) = sql_filter {
        build_sql_from_statement(filter, &mut params)?
    } else {
        String::new()
    };

    Ok((sql, params))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sql_operator_to_string() {
        assert_eq!(sql_operator_to_string(&SqlOperator::Equals), "=");
        assert_eq!(sql_operator_to_string(&SqlOperator::NotEquals), "!=");
        assert_eq!(sql_operator_to_string(&SqlOperator::GreaterThan), ">");
        assert_eq!(sql_operator_to_string(&SqlOperator::LessThan), "<");
        assert_eq!(
            sql_operator_to_string(&SqlOperator::GreaterThanOrEquals),
            ">="
        );
        assert_eq!(sql_operator_to_string(&SqlOperator::LessThanOrEquals), "<=");
        assert_eq!(sql_operator_to_string(&SqlOperator::Like), "LIKE");
        assert_eq!(sql_operator_to_string(&SqlOperator::NotLike), "NOT LIKE");
        assert_eq!(sql_operator_to_string(&SqlOperator::Exists), "IS NOT NULL");
    }

    #[test]
    fn test_build_sql_from_statement_empty() {
        let statement = SqlStatementInternal {
            statement_type: SqlStatementType::Empty,
            condition: None,
            logic: None,
            statements: None,
        };

        let mut params = Vec::new();
        let sql = build_sql_from_statement(&statement, &mut params).unwrap();

        assert_eq!(sql, "");
        assert_eq!(params.len(), 0);
    }

    #[test]
    fn test_build_sql_from_statement_simple_condition() {
        use super::super::types::SqlConditionInternal;
        let statement = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.field".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::String("value".to_string())),
            }),
            logic: None,
            statements: None,
        };

        let mut params = Vec::new();
        let sql = build_sql_from_statement(&statement, &mut params).unwrap();

        assert_eq!(sql, "json_extract(metadata, ?) = ?");
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_build_sql_from_statement_exists_condition() {
        use super::super::types::SqlConditionInternal;
        let statement = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.field".to_string(),
                operator: SqlOperator::Exists,
                value: None,
            }),
            logic: None,
            statements: None,
        };

        let mut params = Vec::new();
        let sql = build_sql_from_statement(&statement, &mut params).unwrap();

        assert_eq!(sql, "json_extract(metadata, ?) IS NOT NULL");
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn test_build_sql_from_statement_numeric_value() {
        use super::super::types::SqlConditionInternal;
        let statement = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.count".to_string(),
                operator: SqlOperator::GreaterThan,
                value: Some(SqlValue::Number(42.0)),
            }),
            logic: None,
            statements: None,
        };

        let mut params = Vec::new();
        let sql = build_sql_from_statement(&statement, &mut params).unwrap();

        assert_eq!(sql, "json_extract(metadata, ?) > ?");
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_build_sql_from_statement_and_group() {
        use super::super::types::SqlConditionInternal;
        let cond1 = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.field1".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::String("value1".to_string())),
            }),
            logic: None,
            statements: None,
        };

        let cond2 = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.field2".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::String("value2".to_string())),
            }),
            logic: None,
            statements: None,
        };

        let group = SqlStatementInternal {
            statement_type: SqlStatementType::Group,
            condition: None,
            logic: Some(SqlLogic::And),
            statements: Some(vec![cond1, cond2]),
        };

        let mut params = Vec::new();
        let sql = build_sql_from_statement(&group, &mut params).unwrap();

        assert_eq!(
            sql,
            "(json_extract(metadata, ?) = ? AND json_extract(metadata, ?) = ?)"
        );
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_build_sql_from_statement_or_group() {
        use super::super::types::SqlConditionInternal;
        let cond1 = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.field1".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::String("value1".to_string())),
            }),
            logic: None,
            statements: None,
        };

        let cond2 = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.field2".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::String("value2".to_string())),
            }),
            logic: None,
            statements: None,
        };

        let group = SqlStatementInternal {
            statement_type: SqlStatementType::Group,
            condition: None,
            logic: Some(SqlLogic::Or),
            statements: Some(vec![cond1, cond2]),
        };

        let mut params = Vec::new();
        let sql = build_sql_from_statement(&group, &mut params).unwrap();

        assert_eq!(
            sql,
            "(json_extract(metadata, ?) = ? OR json_extract(metadata, ?) = ?)"
        );
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_build_sql_from_statement_nested_groups() {
        use super::super::types::SqlConditionInternal;
        // Build: (field1 = "value1" OR field2 = "value2") AND field3 > 100
        let or_group = SqlStatementInternal {
            statement_type: SqlStatementType::Group,
            condition: None,
            logic: Some(SqlLogic::Or),
            statements: Some(vec![
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "$.field1".to_string(),
                        operator: SqlOperator::Equals,
                        value: Some(SqlValue::String("value1".to_string())),
                    }),
                    logic: None,
                    statements: None,
                },
                SqlStatementInternal {
                    statement_type: SqlStatementType::Condition,
                    condition: Some(SqlConditionInternal {
                        key: "$.field2".to_string(),
                        operator: SqlOperator::Equals,
                        value: Some(SqlValue::String("value2".to_string())),
                    }),
                    logic: None,
                    statements: None,
                },
            ]),
        };

        let cond3 = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.field3".to_string(),
                operator: SqlOperator::GreaterThan,
                value: Some(SqlValue::Number(100.0)),
            }),
            logic: None,
            statements: None,
        };

        let and_group = SqlStatementInternal {
            statement_type: SqlStatementType::Group,
            condition: None,
            logic: Some(SqlLogic::And),
            statements: Some(vec![or_group, cond3]),
        };

        let mut params = Vec::new();
        let sql = build_sql_from_statement(&and_group, &mut params).unwrap();

        assert_eq!(
            sql,
            "((json_extract(metadata, ?) = ? OR json_extract(metadata, ?) = ?) AND json_extract(metadata, ?) > ?)"
        );
        assert_eq!(params.len(), 6);
    }

    #[test]
    fn test_build_sql_direct_column_reference() {
        use super::super::types::SqlConditionInternal;
        let statement = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "date".to_string(),
                operator: SqlOperator::GreaterThanOrEquals,
                value: Some(SqlValue::String("2026-04-17T00:00:00Z".to_string())),
            }),
            logic: None,
            statements: None,
        };

        let mut params = Vec::new();
        let sql = build_sql_from_statement(&statement, &mut params).unwrap();

        assert_eq!(sql, "\"date\" >= ?");
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn test_build_filter_sql_and_params_none() {
        let (sql, params) = build_filter_sql_and_params(None).unwrap();

        assert_eq!(sql, "");
        assert_eq!(params.len(), 0);
    }

    #[test]
    fn test_build_filter_sql_and_params_with_filter() {
        use super::super::types::SqlConditionInternal;
        let statement = SqlStatementInternal {
            statement_type: SqlStatementType::Condition,
            condition: Some(SqlConditionInternal {
                key: "$.field".to_string(),
                operator: SqlOperator::Equals,
                value: Some(SqlValue::String("value".to_string())),
            }),
            logic: None,
            statements: None,
        };

        let (sql, params) = build_filter_sql_and_params(Some(&statement)).unwrap();

        assert_eq!(sql, "json_extract(metadata, ?) = ?");
        assert_eq!(params.len(), 2);
    }
}
