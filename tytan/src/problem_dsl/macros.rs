//! Macro system for the problem DSL.

use super::ast::{Expression, Statement};

/// Macro definition
#[derive(Debug, Clone)]
pub struct Macro {
    pub name: String,
    pub parameters: Vec<String>,
    pub body: MacroBody,
}

#[derive(Debug, Clone)]
pub enum MacroBody {
    /// Text substitution
    Text(String),
    /// Expression macro
    Expression(Box<Expression>),
    /// Statement macro
    Statement(Box<Statement>),
}
