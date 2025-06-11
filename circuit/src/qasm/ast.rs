//! Abstract Syntax Tree for OpenQASM 3.0

use std::collections::HashMap;
use std::fmt;

/// Represents a complete QASM program
#[derive(Debug, Clone, PartialEq)]
pub struct QasmProgram {
    /// Version declaration
    pub version: String,
    /// Include statements
    pub includes: Vec<String>,
    /// Global declarations
    pub declarations: Vec<Declaration>,
    /// Statements in the program
    pub statements: Vec<QasmStatement>,
}

/// Declaration types
#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    /// Quantum register declaration
    QuantumRegister(QasmRegister),
    /// Classical register declaration
    ClassicalRegister(QasmRegister),
    /// Gate definition
    GateDefinition(GateDefinition),
    /// Constant declaration
    Constant(String, Expression),
}

/// Register declaration
#[derive(Debug, Clone, PartialEq)]
pub struct QasmRegister {
    /// Register name
    pub name: String,
    /// Size of the register
    pub size: usize,
}

/// Custom gate definition
#[derive(Debug, Clone, PartialEq)]
pub struct GateDefinition {
    /// Gate name
    pub name: String,
    /// Parameter names
    pub params: Vec<String>,
    /// Qubit arguments
    pub qubits: Vec<String>,
    /// Gate body
    pub body: Vec<QasmStatement>,
}

/// QASM statement types
#[derive(Debug, Clone, PartialEq)]
pub enum QasmStatement {
    /// Gate application
    Gate(QasmGate),
    /// Measurement
    Measure(Measurement),
    /// Reset operation
    Reset(Vec<QubitRef>),
    /// Barrier
    Barrier(Vec<QubitRef>),
    /// Classical assignment
    Assignment(String, Expression),
    /// If statement
    If(Condition, Box<QasmStatement>),
    /// For loop
    For(ForLoop),
    /// While loop
    While(Condition, Vec<QasmStatement>),
    /// Function call
    Call(String, Vec<Expression>),
    /// Delay
    Delay(Expression, Vec<QubitRef>),
}

/// Gate application
#[derive(Debug, Clone, PartialEq)]
pub struct QasmGate {
    /// Gate name
    pub name: String,
    /// Parameters (angles, etc.)
    pub params: Vec<Expression>,
    /// Qubit operands
    pub qubits: Vec<QubitRef>,
    /// Control modifier
    pub control: Option<usize>,
    /// Inverse modifier
    pub inverse: bool,
    /// Power modifier
    pub power: Option<Expression>,
}

/// Qubit reference
#[derive(Debug, Clone, PartialEq)]
pub enum QubitRef {
    /// Single qubit: reg[index]
    Single { register: String, index: usize },
    /// Register slice: reg[start:end]
    Slice {
        register: String,
        start: usize,
        end: usize,
    },
    /// Entire register: reg
    Register(String),
}

/// Measurement operation
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    /// Qubits to measure
    pub qubits: Vec<QubitRef>,
    /// Classical bits to store results
    pub targets: Vec<ClassicalRef>,
}

/// Classical bit reference
#[derive(Debug, Clone, PartialEq)]
pub enum ClassicalRef {
    /// Single bit: reg[index]
    Single { register: String, index: usize },
    /// Register slice: reg[start:end]
    Slice {
        register: String,
        start: usize,
        end: usize,
    },
    /// Entire register: reg
    Register(String),
}

/// Expression types
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Literal value
    Literal(Literal),
    /// Variable reference
    Variable(String),
    /// Binary operation
    Binary(BinaryOp, Box<Expression>, Box<Expression>),
    /// Unary operation
    Unary(UnaryOp, Box<Expression>),
    /// Function call
    Function(String, Vec<Expression>),
    /// Array index
    Index(String, Box<Expression>),
}

/// Literal values
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// Integer
    Integer(i64),
    /// Floating point
    Float(f64),
    /// Boolean
    Bool(bool),
    /// String
    String(String),
    /// Pi constant
    Pi,
    /// Euler's number
    Euler,
    /// Tau (2*pi)
    Tau,
}

/// Binary operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Xor,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Unary operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Exp,
    Ln,
    Sqrt,
}

/// Condition for control flow
#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    /// Left operand
    pub left: Expression,
    /// Comparison operator
    pub op: ComparisonOp,
    /// Right operand
    pub right: Expression,
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// For loop structure
#[derive(Debug, Clone, PartialEq)]
pub struct ForLoop {
    /// Loop variable
    pub variable: String,
    /// Start value
    pub start: Expression,
    /// End value
    pub end: Expression,
    /// Step value (optional)
    pub step: Option<Expression>,
    /// Loop body
    pub body: Vec<QasmStatement>,
}

// Display implementations for pretty printing
impl fmt::Display for QasmProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "OPENQASM {};", self.version)?;

        for include in &self.includes {
            writeln!(f, "include \"{}\";", include)?;
        }

        if !self.includes.is_empty() {
            writeln!(f)?;
        }

        for decl in &self.declarations {
            writeln!(f, "{}", decl)?;
        }

        if !self.declarations.is_empty() {
            writeln!(f)?;
        }

        for stmt in &self.statements {
            writeln!(f, "{}", stmt)?;
        }

        Ok(())
    }
}

impl fmt::Display for Declaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Declaration::QuantumRegister(reg) => write!(f, "qubit[{}] {};", reg.size, reg.name),
            Declaration::ClassicalRegister(reg) => write!(f, "bit[{}] {};", reg.size, reg.name),
            Declaration::GateDefinition(def) => {
                write!(f, "gate {}", def.name)?;
                if !def.params.is_empty() {
                    write!(f, "({})", def.params.join(", "))?;
                }
                write!(f, " {}", def.qubits.join(", "))?;
                writeln!(f, " {{")?;
                for stmt in &def.body {
                    writeln!(f, "  {}", stmt)?;
                }
                write!(f, "}}")
            }
            Declaration::Constant(name, expr) => write!(f, "const {} = {};", name, expr),
        }
    }
}

impl fmt::Display for QasmStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QasmStatement::Gate(gate) => write!(f, "{}", gate),
            QasmStatement::Measure(meas) => {
                write!(f, "measure ")?;
                for (i, (q, c)) in meas.qubits.iter().zip(&meas.targets).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} -> {}", q, c)?;
                }
                write!(f, ";")
            }
            QasmStatement::Reset(qubits) => {
                write!(f, "reset ")?;
                for (i, q) in qubits.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", q)?;
                }
                write!(f, ";")
            }
            QasmStatement::Barrier(qubits) => {
                write!(f, "barrier")?;
                if !qubits.is_empty() {
                    write!(f, " ")?;
                    for (i, q) in qubits.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", q)?;
                    }
                }
                write!(f, ";")
            }
            QasmStatement::Assignment(var, expr) => write!(f, "{} = {};", var, expr),
            QasmStatement::If(cond, stmt) => write!(f, "if ({}) {}", cond, stmt),
            QasmStatement::For(for_loop) => {
                write!(
                    f,
                    "for {} in [{}:{}] {{\n",
                    for_loop.variable, for_loop.start, for_loop.end
                )?;
                for stmt in &for_loop.body {
                    writeln!(f, "  {}", stmt)?;
                }
                write!(f, "}}")
            }
            QasmStatement::While(cond, body) => {
                writeln!(f, "while ({}) {{", cond)?;
                for stmt in body {
                    writeln!(f, "  {}", stmt)?;
                }
                write!(f, "}}")
            }
            QasmStatement::Call(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ");")
            }
            QasmStatement::Delay(duration, qubits) => {
                write!(f, "delay[{}]", duration)?;
                if !qubits.is_empty() {
                    write!(f, " ")?;
                    for (i, q) in qubits.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", q)?;
                    }
                }
                write!(f, ";")
            }
        }
    }
}

impl fmt::Display for QasmGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ctrl) = self.control {
            write!(f, "ctrl({}) ", ctrl)?;
        }
        if self.inverse {
            write!(f, "inv ")?;
        }
        if let Some(power) = &self.power {
            write!(f, "pow({}) ", power)?;
        }

        write!(f, "{}", self.name)?;

        if !self.params.is_empty() {
            write!(f, "(")?;
            for (i, param) in self.params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", param)?;
            }
            write!(f, ")")?;
        }

        write!(f, " ")?;
        for (i, qubit) in self.qubits.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", qubit)?;
        }
        write!(f, ";")
    }
}

impl fmt::Display for QubitRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QubitRef::Single { register, index } => write!(f, "{}[{}]", register, index),
            QubitRef::Slice {
                register,
                start,
                end,
            } => write!(f, "{}[{}:{}]", register, start, end),
            QubitRef::Register(name) => write!(f, "{}", name),
        }
    }
}

impl fmt::Display for ClassicalRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClassicalRef::Single { register, index } => write!(f, "{}[{}]", register, index),
            ClassicalRef::Slice {
                register,
                start,
                end,
            } => write!(f, "{}[{}:{}]", register, start, end),
            ClassicalRef::Register(name) => write!(f, "{}", name),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Literal(lit) => write!(f, "{}", lit),
            Expression::Variable(name) => write!(f, "{}", name),
            Expression::Binary(op, left, right) => write!(f, "({} {} {})", left, op, right),
            Expression::Unary(op, expr) => write!(f, "({}{})", op, expr),
            Expression::Function(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Expression::Index(name, idx) => write!(f, "{}[{}]", name, idx),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Integer(n) => write!(f, "{}", n),
            Literal::Float(x) => write!(f, "{}", x),
            Literal::Bool(b) => write!(f, "{}", b),
            Literal::String(s) => write!(f, "\"{}\"", s),
            Literal::Pi => write!(f, "pi"),
            Literal::Euler => write!(f, "e"),
            Literal::Tau => write!(f, "tau"),
        }
    }
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Pow => "**",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::And => "&&",
            BinaryOp::Or => "||",
            BinaryOp::Xor => "^^",
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
        };
        write!(f, "{}", op)
    }
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = match self {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "!",
            UnaryOp::BitNot => "~",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Tan => "tan",
            UnaryOp::Asin => "asin",
            UnaryOp::Acos => "acos",
            UnaryOp::Atan => "atan",
            UnaryOp::Exp => "exp",
            UnaryOp::Ln => "ln",
            UnaryOp::Sqrt => "sqrt",
        };
        write!(f, "{}", op)
    }
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.left, self.op, self.right)
    }
}

impl fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = match self {
            ComparisonOp::Eq => "==",
            ComparisonOp::Ne => "!=",
            ComparisonOp::Lt => "<",
            ComparisonOp::Le => "<=",
            ComparisonOp::Gt => ">",
            ComparisonOp::Ge => ">=",
        };
        write!(f, "{}", op)
    }
}
