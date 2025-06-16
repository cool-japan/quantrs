//! Abstract syntax tree definitions for the problem DSL.

use super::types::VarType;
use std::collections::HashMap;

/// Abstract syntax tree
#[derive(Debug, Clone)]
pub enum AST {
    /// Program root
    Program {
        declarations: Vec<Declaration>,
        objective: Objective,
        constraints: Vec<Constraint>,
    },

    /// Variable declaration
    VarDecl {
        name: String,
        var_type: VarType,
        domain: Option<Domain>,
        attributes: HashMap<String, Value>,
    },

    /// Expression
    Expr(Expression),

    /// Statement
    Stmt(Statement),
}

/// Declaration types
#[derive(Debug, Clone)]
pub enum Declaration {
    /// Variable declaration
    Variable {
        name: String,
        var_type: VarType,
        domain: Option<Domain>,
        attributes: HashMap<String, Value>,
    },

    /// Parameter declaration
    Parameter {
        name: String,
        value: Value,
        description: Option<String>,
    },

    /// Set declaration
    Set { name: String, elements: Vec<Value> },

    /// Function declaration
    Function {
        name: String,
        params: Vec<String>,
        body: Box<Expression>,
    },
}

/// Variable domain
#[derive(Debug, Clone)]
pub enum Domain {
    /// Range domain
    Range { min: f64, max: f64 },
    /// Set domain
    Set { values: Vec<Value> },
    /// Index set
    IndexSet { set_name: String },
}

/// Value types
#[derive(Debug, Clone)]
pub enum Value {
    Number(f64),
    Boolean(bool),
    String(String),
    Array(Vec<Value>),
    Tuple(Vec<Value>),
}

/// Objective function
#[derive(Debug, Clone)]
pub enum Objective {
    Minimize(Expression),
    Maximize(Expression),
    MultiObjective {
        objectives: Vec<(ObjectiveType, Expression, f64)>,
    },
}

#[derive(Debug, Clone)]
pub enum ObjectiveType {
    Minimize,
    Maximize,
}

/// Constraint
#[derive(Debug, Clone)]
pub struct Constraint {
    pub name: Option<String>,
    pub expression: ConstraintExpression,
    pub tags: Vec<String>,
}

/// Constraint expression
#[derive(Debug, Clone)]
pub enum ConstraintExpression {
    /// Simple comparison
    Comparison {
        left: Expression,
        op: ComparisonOp,
        right: Expression,
    },

    /// Logical combination
    Logical {
        op: LogicalOp,
        operands: Vec<ConstraintExpression>,
    },

    /// Quantified constraint
    Quantified {
        quantifier: Quantifier,
        variables: Vec<(String, String)>, // (var, set)
        constraint: Box<ConstraintExpression>,
    },

    /// Implication
    Implication {
        condition: Box<ConstraintExpression>,
        consequence: Box<ConstraintExpression>,
    },

    /// Counting constraint
    Counting {
        variables: Vec<String>,
        op: ComparisonOp,
        count: Expression,
    },
}

/// Comparison operators
#[derive(Debug, Clone, PartialEq)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
}

/// Logical operators
#[derive(Debug, Clone)]
pub enum LogicalOp {
    And,
    Or,
    Not,
    Xor,
}

/// Quantifiers
#[derive(Debug, Clone)]
pub enum Quantifier {
    ForAll,
    Exists,
    ExactlyOne,
    AtMostOne,
    AtLeastOne,
}

/// Expression
#[derive(Debug, Clone)]
pub enum Expression {
    /// Literal value
    Literal(Value),

    /// Variable reference
    Variable(String),

    /// Indexed variable
    IndexedVar {
        name: String,
        indices: Vec<Expression>,
    },

    /// Binary operation
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },

    /// Unary operation
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression>,
    },

    /// Function call
    FunctionCall { name: String, args: Vec<Expression> },

    /// Aggregation
    Aggregation {
        op: AggregationOp,
        variables: Vec<(String, String)>, // (var, set)
        expression: Box<Expression>,
    },

    /// Conditional
    Conditional {
        condition: Box<ConstraintExpression>,
        then_expr: Box<Expression>,
        else_expr: Box<Expression>,
    },
}

/// Binary operators
#[derive(Debug, Clone)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Modulo,
}

/// Unary operators
#[derive(Debug, Clone)]
pub enum UnaryOperator {
    Negate,
    Abs,
    Sqrt,
    Exp,
    Log,
}

/// Aggregation operators
#[derive(Debug, Clone)]
pub enum AggregationOp {
    Sum,
    Product,
    Min,
    Max,
    Count,
}

/// Statement
#[derive(Debug, Clone)]
pub enum Statement {
    /// Assignment
    Assignment { target: String, value: Expression },

    /// Conditional
    If {
        condition: ConstraintExpression,
        then_branch: Vec<Statement>,
        else_branch: Option<Vec<Statement>>,
    },

    /// Loop
    For {
        variable: String,
        set: String,
        body: Vec<Statement>,
    },
}
