#[cfg(feature = "napi")]
use napi::bindgen_prelude::Either;
#[cfg(feature = "napi")]
use napi_derive::napi;

/// SQL operators for filtering
#[cfg_attr(feature = "napi", napi(string_enum))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum SqlOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEquals,
    LessThanOrEquals,
    Like,
    NotLike,
    Exists,
}

/// Logical operators for combining conditions
#[cfg_attr(feature = "napi", napi(string_enum))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum SqlLogic {
    And,
    Or,
}

/// Type of SQL statement
#[cfg_attr(feature = "napi", napi(string_enum))]
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum SqlStatementType {
    Condition,
    Group,
    Empty,
}

/// A single condition in a SQL filter (internal representation)
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum SqlValue {
    String(String),
    Number(f64),
}

/// A single condition in a SQL filter (internal representation)
#[derive(Debug, Clone)]
pub struct SqlConditionInternal {
    pub key: String,
    pub operator: SqlOperator,
    pub value: Option<SqlValue>,
}

/// A SQL filter statement (internal representation)
#[derive(Debug, Clone)]
pub struct SqlStatementInternal {
    pub statement_type: SqlStatementType,
    pub condition: Option<SqlConditionInternal>,
    pub logic: Option<SqlLogic>,
    pub statements: Option<Vec<SqlStatementInternal>>,
}

// NAPI-specific types and conversions
#[cfg(feature = "napi")]
#[allow(unused_imports)]
pub use napi_types::*;

#[cfg(feature = "napi")]
mod napi_types {
    use super::*;

    /// A single condition in a SQL filter (NAPI export)
    #[napi(object)]
    #[derive(Debug, Clone)]
    pub struct SqlCondition {
        pub key: String,
        pub operator: super::SqlOperator,
        pub value: Option<Either<String, f64>>,
    }

    /// A SQL filter statement (NAPI export)
    #[napi(object)]
    #[derive(Debug, Clone)]
    pub struct SqlStatement {
        pub r#type: super::SqlStatementType,
        pub condition: Option<SqlCondition>,
        pub logic: Option<super::SqlLogic>,
        pub statements: Option<Vec<SqlStatement>>,
    }

    // Conversions from NAPI types to internal types
    impl From<SqlCondition> for super::SqlConditionInternal {
        fn from(cond: SqlCondition) -> Self {
            super::SqlConditionInternal {
                key: cond.key,
                operator: cond.operator,
                value: cond.value.map(|v| match v {
                    Either::A(s) => super::SqlValue::String(s),
                    Either::B(n) => super::SqlValue::Number(n),
                }),
            }
        }
    }

    impl From<SqlStatement> for super::SqlStatementInternal {
        fn from(stmt: SqlStatement) -> Self {
            super::SqlStatementInternal {
                statement_type: stmt.r#type,
                condition: stmt.condition.map(Into::into),
                logic: stmt.logic,
                statements: stmt
                    .statements
                    .map(|stmts| stmts.into_iter().map(Into::into).collect()),
            }
        }
    }
}
