//! Problem modeling DSL for quantum optimization.
//!
//! This module provides a domain-specific language for defining
//! optimization problems in a high-level, intuitive way.

use crate::sampler::{Sampler, SampleResult, SamplerError, SamplerResult};
#[cfg(feature = "dwave")]
use crate::compile::{Compile, CompiledModel};
use ndarray::{Array, Array1, Array2, IxDyn};
use std::collections::{HashMap, HashSet};
use std::fmt;
use quantrs2_anneal::qubo::Variable;
use std::str::FromStr;

/// DSL parser for optimization problems
pub struct ProblemDSL {
    /// Parser state
    parser: Parser,
    /// Type checker
    type_checker: TypeChecker,
    /// Standard library
    stdlib: StandardLibrary,
    /// Compiler options
    options: CompilerOptions,
    /// Macro definitions
    macros: HashMap<String, Macro>,
    /// Import resolver
    import_resolver: ImportResolver,
    /// Optimization hints
    optimization_hints: Vec<OptimizationHint>,
}

/// Parser for DSL syntax
#[derive(Debug, Clone)]
pub struct Parser {
    /// Current tokens
    tokens: Vec<Token>,
    /// Current position
    position: usize,
    /// Error messages
    errors: Vec<ParseError>,
}

/// Token types
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Number(f64),
    String(String),
    Boolean(bool),
    Identifier(String),
    
    // Keywords
    Var,
    Param,
    Constraint,
    Minimize,
    Maximize,
    Subject,
    To,
    Binary,
    Integer,
    Continuous,
    In,
    ForAll,
    Exists,
    Sum,
    Product,
    If,
    Then,
    Else,
    Let,
    Define,
    Macro,
    Import,
    From,
    As,
    Domain,
    Range,
    Symmetry,
    Hint,
    
    // Operators
    Plus,
    Minus,
    Times,
    Divide,
    Power,
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    And,
    Or,
    Not,
    Implies,
    Mod,
    Xor,
    
    // Delimiters
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Comma,
    Semicolon,
    Colon,
    Arrow,
    Dot,
    DoubleDot,
    Pipe,
    
    // Special
    Eof,
    NewLine,
    Comment(String),
}

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

/// Import resolver
#[derive(Debug, Clone)]
pub struct ImportResolver {
    /// Import paths
    pub paths: Vec<String>,
    /// Loaded modules
    pub modules: HashMap<String, Module>,
    /// Symbol table
    pub symbols: HashMap<String, ImportedSymbol>,
}

#[derive(Debug, Clone)]
pub struct Module {
    pub name: String,
    pub exports: HashMap<String, ExportedItem>,
}

#[derive(Debug, Clone)]
pub enum ExportedItem {
    Variable(Variable),
    Function(BuiltinFunction),
    Template(Template),
    Macro(Macro),
}

#[derive(Debug, Clone)]
pub struct ImportedSymbol {
    pub module: String,
    pub original_name: String,
    pub local_name: String,
}

/// Optimization hint
#[derive(Debug, Clone)]
pub enum OptimizationHint {
    /// Variable ordering hint
    VariableOrder(Vec<String>),
    /// Symmetry breaking
    Symmetry(SymmetryType),
    /// Decomposition hint
    Decomposition(DecompositionHint),
    /// Solver preference
    SolverPreference(String),
    /// Custom hint
    Custom { name: String, value: String },
}

#[derive(Debug, Clone)]
pub enum SymmetryType {
    /// Permutation symmetry
    Permutation(Vec<String>),
    /// Reflection symmetry
    Reflection { axis: String },
    /// Rotation symmetry
    Rotation { order: usize },
}

#[derive(Debug, Clone)]
pub struct DecompositionHint {
    pub method: String,
    pub parameters: HashMap<String, Value>,
}

/// Parse error
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

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
    Set {
        name: String,
        elements: Vec<Value>,
    },
    
    /// Function declaration
    Function {
        name: String,
        params: Vec<String>,
        body: Box<Expression>,
    },
}

/// Variable types
#[derive(Debug, Clone, PartialEq)]
pub enum VarType {
    Binary,
    Integer,
    Continuous,
    Spin,
    Array { element_type: Box<VarType>, dimensions: Vec<usize> },
    Matrix { element_type: Box<VarType>, rows: usize, cols: usize },
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
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
    
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
    Assignment {
        target: String,
        value: Expression,
    },
    
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

/// Type checker
#[derive(Debug, Clone)]
pub struct TypeChecker {
    /// Variable types
    var_types: HashMap<String, VarType>,
    /// Function signatures
    func_signatures: HashMap<String, FunctionSignature>,
    /// Type errors
    errors: Vec<TypeError>,
}

/// Function signature
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub param_types: Vec<VarType>,
    pub return_type: VarType,
}

/// Type error
#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub location: String,
}

/// Standard library
#[derive(Debug, Clone)]
pub struct StandardLibrary {
    /// Built-in functions
    functions: HashMap<String, BuiltinFunction>,
    /// Common patterns
    patterns: HashMap<String, Pattern>,
    /// Problem templates
    templates: HashMap<String, Template>,
}

/// Built-in function
#[derive(Debug, Clone)]
pub struct BuiltinFunction {
    pub name: String,
    pub signature: FunctionSignature,
    pub description: String,
    pub implementation: FunctionImpl,
}

/// Function implementation
#[derive(Debug, Clone)]
pub enum FunctionImpl {
    /// Native Rust implementation
    Native,
    /// DSL implementation
    DSL { body: Expression },
}

/// Common pattern
#[derive(Debug, Clone)]
pub struct Pattern {
    pub name: String,
    pub description: String,
    pub parameters: Vec<String>,
    pub expansion: AST,
}

/// Problem template
#[derive(Debug, Clone)]
pub struct Template {
    pub name: String,
    pub description: String,
    pub parameters: Vec<TemplateParam>,
    pub body: String,
}

#[derive(Debug, Clone)]
pub struct TemplateParam {
    pub name: String,
    pub param_type: String,
    pub default: Option<Value>,
}

/// Compiler options
#[derive(Debug, Clone)]
pub struct CompilerOptions {
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Target backend
    pub target: TargetBackend,
    /// Debug information
    pub debug_info: bool,
    /// Warnings as errors
    pub warnings_as_errors: bool,
}

#[derive(Debug, Clone)]
pub enum OptimizationLevel {
    None,
    Basic,
    Full,
}

#[derive(Debug, Clone)]
pub enum TargetBackend {
    QUBO,
    Ising,
    HigherOrder,
    Constrained,
}

impl ProblemDSL {
    /// Create new DSL instance
    pub fn new() -> Self {
        Self {
            parser: Parser::new(),
            type_checker: TypeChecker::new(),
            stdlib: StandardLibrary::default(),
            options: CompilerOptions::default(),
            macros: HashMap::new(),
            import_resolver: ImportResolver::new(),
            optimization_hints: Vec::new(),
        }
    }
    
    /// Define a macro
    pub fn define_macro(&mut self, name: String, params: Vec<String>, body: MacroBody) {
        self.macros.insert(name.clone(), Macro {
            name,
            parameters: params,
            body,
        });
    }
    
    /// Add optimization hint
    pub fn add_hint(&mut self, hint: OptimizationHint) {
        self.optimization_hints.push(hint);
    }
    
    /// Import module
    pub fn import(&mut self, module_path: &str, alias: Option<String>) -> Result<(), String> {
        self.import_resolver.import_module(module_path, alias)
    }
    
    /// Parse DSL source code
    pub fn parse(&mut self, source: &str) -> Result<AST, Vec<ParseError>> {
        // Tokenize
        let tokens = self.tokenize(source)?;
        self.parser.tokens = tokens;
        self.parser.position = 0;
        
        // Parse program
        self.parser.parse_program()
    }
    
    /// Tokenize source code
    fn tokenize(&self, source: &str) -> Result<Vec<Token>, Vec<ParseError>> {
        let mut tokens = Vec::new();
        let mut chars = source.chars().peekable();
        let mut line = 1;
        let mut column = 1;
        
        while let Some(&ch) = chars.peek() {
            match ch {
                // Skip whitespace
                ' ' | '\t' => {
                    chars.next();
                    column += 1;
                }
                '\n' => {
                    chars.next();
                    line += 1;
                    column = 1;
                }
                
                // Numbers
                '0'..='9' => {
                    let (number, len) = self.scan_number(&mut chars)?;
                    tokens.push(Token::Number(number));
                    column += len;
                }
                
                // Identifiers and keywords
                'a'..='z' | 'A'..='Z' | '_' => {
                    let (ident, len) = self.scan_identifier(&mut chars);
                    let token = self.keyword_or_identifier(&ident);
                    tokens.push(token);
                    column += len;
                }
                
                // String literals
                '"' => {
                    let (string, len) = self.scan_string(&mut chars)?;
                    tokens.push(Token::String(string));
                    column += len;
                }
                
                // Operators and delimiters
                '+' => {
                    chars.next();
                    tokens.push(Token::Plus);
                    column += 1;
                }
                '-' => {
                    chars.next();
                    if chars.peek() == Some(&'>') {
                        chars.next();
                        tokens.push(Token::Arrow);
                        column += 2;
                    } else {
                        tokens.push(Token::Minus);
                        column += 1;
                    }
                }
                '*' => {
                    chars.next();
                    tokens.push(Token::Times);
                    column += 1;
                }
                '/' => {
                    chars.next();
                    if chars.peek() == Some(&'/') {
                        // Comment - skip to end of line
                        while chars.peek() != Some(&'\n') && chars.peek().is_some() {
                            chars.next();
                            column += 1;
                        }
                    } else {
                        tokens.push(Token::Divide);
                        column += 1;
                    }
                }
                '^' => {
                    chars.next();
                    tokens.push(Token::Power);
                    column += 1;
                }
                '=' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::Equal);
                        column += 2;
                    } else {
                        return Err(vec![ParseError {
                            message: "Expected '==' for equality".to_string(),
                            line,
                            column,
                        }]);
                    }
                }
                '<' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::LessEqual);
                        column += 2;
                    } else {
                        tokens.push(Token::Less);
                        column += 1;
                    }
                }
                '>' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::GreaterEqual);
                        column += 2;
                    } else {
                        tokens.push(Token::Greater);
                        column += 1;
                    }
                }
                '!' => {
                    chars.next();
                    if chars.peek() == Some(&'=') {
                        chars.next();
                        tokens.push(Token::NotEqual);
                        column += 2;
                    } else {
                        tokens.push(Token::Not);
                        column += 1;
                    }
                }
                '&' => {
                    chars.next();
                    if chars.peek() == Some(&'&') {
                        chars.next();
                        tokens.push(Token::And);
                        column += 2;
                    } else {
                        return Err(vec![ParseError {
                            message: "Expected '&&' for logical AND".to_string(),
                            line,
                            column,
                        }]);
                    }
                }
                '|' => {
                    chars.next();
                    if chars.peek() == Some(&'|') {
                        chars.next();
                        tokens.push(Token::Or);
                        column += 2;
                    } else {
                        return Err(vec![ParseError {
                            message: "Expected '||' for logical OR".to_string(),
                            line,
                            column,
                        }]);
                    }
                }
                '(' => {
                    chars.next();
                    tokens.push(Token::LeftParen);
                    column += 1;
                }
                ')' => {
                    chars.next();
                    tokens.push(Token::RightParen);
                    column += 1;
                }
                '[' => {
                    chars.next();
                    tokens.push(Token::LeftBracket);
                    column += 1;
                }
                ']' => {
                    chars.next();
                    tokens.push(Token::RightBracket);
                    column += 1;
                }
                '{' => {
                    chars.next();
                    tokens.push(Token::LeftBrace);
                    column += 1;
                }
                '}' => {
                    chars.next();
                    tokens.push(Token::RightBrace);
                    column += 1;
                }
                ',' => {
                    chars.next();
                    tokens.push(Token::Comma);
                    column += 1;
                }
                ';' => {
                    chars.next();
                    tokens.push(Token::Semicolon);
                    column += 1;
                }
                ':' => {
                    chars.next();
                    tokens.push(Token::Colon);
                    column += 1;
                }
                
                _ => {
                    return Err(vec![ParseError {
                        message: format!("Unexpected character: '{}'", ch),
                        line,
                        column,
                    }]);
                }
            }
        }
        
        tokens.push(Token::Eof);
        Ok(tokens)
    }
    
    /// Scan number
    fn scan_number(&self, chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<(f64, usize), ParseError> {
        let mut number_str = String::new();
        
        // Integer part
        while let Some(&ch) = chars.peek() {
            if ch.is_ascii_digit() {
                number_str.push(ch);
                chars.next();
            } else {
                break;
            }
        }
        
        // Decimal part
        if chars.peek() == Some(&'.') {
            number_str.push('.');
            chars.next();
            
            while let Some(&ch) = chars.peek() {
                if ch.is_ascii_digit() {
                    number_str.push(ch);
                    chars.next();
                } else {
                    break;
                }
            }
        }
        
        // Exponent part
        if chars.peek() == Some(&'e') || chars.peek() == Some(&'E') {
            number_str.push('e');
            chars.next();
            
            if chars.peek() == Some(&'+') || chars.peek() == Some(&'-') {
                number_str.push(chars.next().unwrap());
            }
            
            while let Some(&ch) = chars.peek() {
                if ch.is_ascii_digit() {
                    number_str.push(ch);
                    chars.next();
                } else {
                    break;
                }
            }
        }
        
        let len = number_str.len();
        let number = number_str.parse::<f64>().map_err(|_| ParseError {
            message: format!("Invalid number: {}", number_str),
            line: 0,
            column: 0,
        })?;
        
        Ok((number, len))
    }
    
    /// Scan identifier
    fn scan_identifier(&self, chars: &mut std::iter::Peekable<std::str::Chars>) -> (String, usize) {
        let mut ident = String::new();
        
        while let Some(&ch) = chars.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                chars.next();
            } else {
                break;
            }
        }
        
        let len = ident.len();
        (ident, len)
    }
    
    /// Scan string literal
    fn scan_string(&self, chars: &mut std::iter::Peekable<std::str::Chars>) -> Result<(String, usize), ParseError> {
        let mut string = String::new();
        let mut len = 1;
        
        chars.next(); // Skip opening quote
        
        while let Some(ch) = chars.next() {
            len += 1;
            match ch {
                '"' => return Ok((string, len)),
                '\\' => {
                    if let Some(escaped) = chars.next() {
                        len += 1;
                        match escaped {
                            'n' => string.push('\n'),
                            't' => string.push('\t'),
                            'r' => string.push('\r'),
                            '\\' => string.push('\\'),
                            '"' => string.push('"'),
                            _ => {
                                return Err(ParseError {
                                    message: format!("Invalid escape sequence: \\{}", escaped),
                                    line: 0,
                                    column: 0,
                                });
                            }
                        }
                    }
                }
                _ => string.push(ch),
            }
        }
        
        Err(ParseError {
            message: "Unterminated string literal".to_string(),
            line: 0,
            column: 0,
        })
    }
    
    /// Check if identifier is keyword
    fn keyword_or_identifier(&self, ident: &str) -> Token {
        match ident {
            "var" => Token::Var,
            "constraint" => Token::Constraint,
            "minimize" => Token::Minimize,
            "maximize" => Token::Maximize,
            "subject" => Token::Subject,
            "to" => Token::To,
            "binary" => Token::Binary,
            "integer" => Token::Integer,
            "continuous" => Token::Continuous,
            "in" => Token::In,
            "forall" => Token::ForAll,
            "exists" => Token::Exists,
            "sum" => Token::Sum,
            "product" => Token::Product,
            "if" => Token::If,
            "then" => Token::Then,
            "else" => Token::Else,
            "true" => Token::Boolean(true),
            "false" => Token::Boolean(false),
            _ => Token::Identifier(ident.to_string()),
        }
    }
    
    /// Type check AST
    pub fn type_check(&mut self, ast: &AST) -> Result<(), Vec<TypeError>> {
        self.type_checker.check(ast)
    }
    
    /// Compile to QUBO
    pub fn compile_to_qubo(&self, ast: &AST) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        let compiler = QUBOCompiler::new(self.options.clone());
        compiler.compile(ast)
    }
}

impl Parser {
    /// Create new parser
    fn new() -> Self {
        Self {
            tokens: Vec::new(),
            position: 0,
            errors: Vec::new(),
        }
    }
    
    /// Parse program
    fn parse_program(&mut self) -> Result<AST, Vec<ParseError>> {
        let mut declarations = Vec::new();
        let mut constraints = Vec::new();
        let mut objective = None;
        
        while !self.is_at_end() {
            match self.peek() {
                Token::Var => {
                    declarations.push(self.parse_declaration()?);
                }
                Token::Minimize | Token::Maximize => {
                    if objective.is_some() {
                        return Err(vec![ParseError {
                            message: "Multiple objectives not allowed".to_string(),
                            line: 0,
                            column: 0,
                        }]);
                    }
                    objective = Some(self.parse_objective()?);
                }
                Token::Subject => {
                    self.advance(); // subject
                    self.consume(Token::To, "Expected 'to' after 'subject'")?;
                    while !self.is_at_end() {
                        constraints.push(self.parse_constraint()?);
                    }
                }
                Token::Constraint => {
                    constraints.push(self.parse_constraint()?);
                }
                _ => {
                    return Err(vec![ParseError {
                        message: format!("Unexpected token: {:?}", self.peek()),
                        line: 0,
                        column: 0,
                    }]);
                }
            }
        }
        
        if objective.is_none() {
            return Err(vec![ParseError {
                message: "No objective function specified".to_string(),
                line: 0,
                column: 0,
            }]);
        }
        
        Ok(AST::Program {
            declarations,
            objective: objective.unwrap(),
            constraints,
        })
    }
    
    /// Parse declaration
    fn parse_declaration(&mut self) -> Result<Declaration, ParseError> {
        self.consume(Token::Var, "Expected 'var'")?;
        
        let name = self.parse_identifier()?;
        
        // Parse variable type
        let var_type = if self.peek() == &Token::LeftBracket {
            // Array type
            self.advance(); // [
            let dimensions = self.parse_dimensions()?;
            self.consume(Token::RightBracket, "Expected ']'")?;
            
            let element_type = self.parse_var_type()?;
            VarType::Array {
                element_type: Box::new(element_type),
                dimensions,
            }
        } else {
            self.parse_var_type()?
        };
        
        // Parse domain
        let domain = if self.peek() == &Token::In {
            self.advance(); // in
            Some(self.parse_domain()?)
        } else {
            None
        };
        
        // Parse attributes
        let attributes = HashMap::new(); // TODO: implement attribute parsing
        
        self.consume(Token::Semicolon, "Expected ';' after variable declaration")?;
        
        Ok(Declaration::Variable {
            name,
            var_type,
            domain,
            attributes,
        })
    }
    
    /// Parse variable type
    fn parse_var_type(&mut self) -> Result<VarType, ParseError> {
        match self.peek() {
            Token::Binary => {
                self.advance();
                Ok(VarType::Binary)
            }
            Token::Integer => {
                self.advance();
                Ok(VarType::Integer)
            }
            Token::Continuous => {
                self.advance();
                Ok(VarType::Continuous)
            }
            _ => Err(ParseError {
                message: "Expected variable type".to_string(),
                line: 0,
                column: 0,
            }),
        }
    }
    
    /// Parse dimensions
    fn parse_dimensions(&mut self) -> Result<Vec<usize>, ParseError> {
        let mut dimensions = Vec::new();
        
        loop {
            if let Token::Number(n) = self.peek() {
                dimensions.push(*n as usize);
                self.advance();
                
                if self.peek() == &Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            } else {
                return Err(ParseError {
                    message: "Expected dimension size".to_string(),
                    line: 0,
                    column: 0,
                });
            }
        }
        
        Ok(dimensions)
    }
    
    /// Parse domain
    fn parse_domain(&mut self) -> Result<Domain, ParseError> {
        if self.peek() == &Token::LeftBrace {
            // Set domain
            self.advance(); // {
            let mut values = Vec::new();
            
            loop {
                values.push(self.parse_value()?);
                
                if self.peek() == &Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            
            self.consume(Token::RightBrace, "Expected '}'")?;
            Ok(Domain::Set { values })
        } else if let Token::Identifier(set_name) = self.peek().clone() {
            // Index set
            self.advance();
            Ok(Domain::IndexSet { set_name })
        } else {
            // Range domain
            let min = self.parse_number()?;
            self.consume(Token::Minus, "Expected '-' in range")?;
            self.consume(Token::Minus, "Expected second '-' in range")?;
            let max = self.parse_number()?;
            
            Ok(Domain::Range { min, max })
        }
    }
    
    /// Parse value
    fn parse_value(&mut self) -> Result<Value, ParseError> {
        match self.peek().clone() {
            Token::Number(n) => {
                self.advance();
                Ok(Value::Number(n))
            }
            Token::Boolean(b) => {
                self.advance();
                Ok(Value::Boolean(b))
            }
            Token::String(s) => {
                self.advance();
                Ok(Value::String(s))
            }
            _ => Err(ParseError {
                message: "Expected value".to_string(),
                line: 0,
                column: 0,
            }),
        }
    }
    
    /// Parse objective
    fn parse_objective(&mut self) -> Result<Objective, ParseError> {
        match self.peek() {
            Token::Minimize => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume(Token::Semicolon, "Expected ';' after objective")?;
                Ok(Objective::Minimize(expr))
            }
            Token::Maximize => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume(Token::Semicolon, "Expected ';' after objective")?;
                Ok(Objective::Maximize(expr))
            }
            _ => Err(ParseError {
                message: "Expected 'minimize' or 'maximize'".to_string(),
                line: 0,
                column: 0,
            }),
        }
    }
    
    /// Parse constraint
    fn parse_constraint(&mut self) -> Result<Constraint, ParseError> {
        let name = if self.peek() == &Token::Constraint {
            self.advance(); // constraint
            if let Token::Identifier(name) = self.peek().clone() {
                self.advance();
                self.consume(Token::Colon, "Expected ':' after constraint name")?;
                Some(name)
            } else {
                None
            }
        } else {
            None
        };
        
        let expression = self.parse_constraint_expression()?;
        self.consume(Token::Semicolon, "Expected ';' after constraint")?;
        
        Ok(Constraint {
            name,
            expression,
            tags: Vec::new(),
        })
    }
    
    /// Parse constraint expression
    fn parse_constraint_expression(&mut self) -> Result<ConstraintExpression, ParseError> {
        self.parse_logical_or()
    }
    
    /// Parse logical OR
    fn parse_logical_or(&mut self) -> Result<ConstraintExpression, ParseError> {
        let mut left = self.parse_logical_and()?;
        
        while self.peek() == &Token::Or {
            self.advance();
            let right = self.parse_logical_and()?;
            left = ConstraintExpression::Logical {
                op: LogicalOp::Or,
                operands: vec![left, right],
            };
        }
        
        Ok(left)
    }
    
    /// Parse logical AND
    fn parse_logical_and(&mut self) -> Result<ConstraintExpression, ParseError> {
        let mut left = self.parse_comparison()?;
        
        while self.peek() == &Token::And {
            self.advance();
            let right = self.parse_comparison()?;
            left = ConstraintExpression::Logical {
                op: LogicalOp::And,
                operands: vec![left, right],
            };
        }
        
        Ok(left)
    }
    
    /// Parse comparison
    fn parse_comparison(&mut self) -> Result<ConstraintExpression, ParseError> {
        let left = self.parse_expression()?;
        
        let op = match self.peek() {
            Token::Equal => {
                self.advance();
                ComparisonOp::Equal
            }
            Token::NotEqual => {
                self.advance();
                ComparisonOp::NotEqual
            }
            Token::Less => {
                self.advance();
                ComparisonOp::Less
            }
            Token::Greater => {
                self.advance();
                ComparisonOp::Greater
            }
            Token::LessEqual => {
                self.advance();
                ComparisonOp::LessEqual
            }
            Token::GreaterEqual => {
                self.advance();
                ComparisonOp::GreaterEqual
            }
            _ => {
                return Err(ParseError {
                    message: "Expected comparison operator".to_string(),
                    line: 0,
                    column: 0,
                });
            }
        };
        
        let right = self.parse_expression()?;
        
        Ok(ConstraintExpression::Comparison { left, op, right })
    }
    
    /// Parse expression
    fn parse_expression(&mut self) -> Result<Expression, ParseError> {
        self.parse_addition()
    }
    
    /// Parse addition/subtraction
    fn parse_addition(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_multiplication()?;
        
        while matches!(self.peek(), Token::Plus | Token::Minus) {
            let op = match self.peek() {
                Token::Plus => {
                    self.advance();
                    BinaryOperator::Add
                }
                Token::Minus => {
                    self.advance();
                    BinaryOperator::Subtract
                }
                _ => unreachable!(),
            };
            
            let right = self.parse_multiplication()?;
            left = Expression::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse multiplication/division
    fn parse_multiplication(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_power()?;
        
        while matches!(self.peek(), Token::Times | Token::Divide) {
            let op = match self.peek() {
                Token::Times => {
                    self.advance();
                    BinaryOperator::Multiply
                }
                Token::Divide => {
                    self.advance();
                    BinaryOperator::Divide
                }
                _ => unreachable!(),
            };
            
            let right = self.parse_power()?;
            left = Expression::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse power
    fn parse_power(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_unary()?;
        
        if self.peek() == &Token::Power {
            self.advance();
            let right = self.parse_power()?; // Right associative
            left = Expression::BinaryOp {
                op: BinaryOperator::Power,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse unary
    fn parse_unary(&mut self) -> Result<Expression, ParseError> {
        match self.peek() {
            Token::Minus => {
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expression::UnaryOp {
                    op: UnaryOperator::Negate,
                    operand: Box::new(operand),
                })
            }
            Token::Sum => {
                self.advance();
                self.parse_aggregation(AggregationOp::Sum)
            }
            Token::Product => {
                self.advance();
                self.parse_aggregation(AggregationOp::Product)
            }
            _ => self.parse_primary(),
        }
    }
    
    /// Parse aggregation
    fn parse_aggregation(&mut self, op: AggregationOp) -> Result<Expression, ParseError> {
        self.consume(Token::LeftParen, "Expected '(' after aggregation operator")?;
        
        let mut variables = Vec::new();
        
        // Parse index variables
        loop {
            let var = self.parse_identifier()?;
            self.consume(Token::In, "Expected 'in' after variable")?;
            let set = self.parse_identifier()?;
            
            variables.push((var, set));
            
            if self.peek() == &Token::Comma {
                self.advance();
            } else {
                break;
            }
        }
        
        self.consume(Token::Colon, "Expected ':' after variables")?;
        
        let expression = self.parse_expression()?;
        
        self.consume(Token::RightParen, "Expected ')' after aggregation")?;
        
        Ok(Expression::Aggregation {
            op,
            variables,
            expression: Box::new(expression),
        })
    }
    
    /// Parse primary expression
    fn parse_primary(&mut self) -> Result<Expression, ParseError> {
        match self.peek().clone() {
            Token::Number(n) => {
                self.advance();
                Ok(Expression::Literal(Value::Number(n)))
            }
            Token::Boolean(b) => {
                self.advance();
                Ok(Expression::Literal(Value::Boolean(b)))
            }
            Token::String(s) => {
                self.advance();
                Ok(Expression::Literal(Value::String(s)))
            }
            Token::Identifier(name) => {
                self.advance();
                
                if self.peek() == &Token::LeftBracket {
                    // Indexed variable
                    self.advance(); // [
                    let mut indices = Vec::new();
                    
                    loop {
                        indices.push(self.parse_expression()?);
                        
                        if self.peek() == &Token::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    
                    self.consume(Token::RightBracket, "Expected ']'")?;
                    
                    Ok(Expression::IndexedVar { name, indices })
                } else if self.peek() == &Token::LeftParen {
                    // Function call
                    self.advance(); // (
                    let mut args = Vec::new();
                    
                    if self.peek() != &Token::RightParen {
                        loop {
                            args.push(self.parse_expression()?);
                            
                            if self.peek() == &Token::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    
                    self.consume(Token::RightParen, "Expected ')'")?;
                    
                    Ok(Expression::FunctionCall { name, args })
                } else {
                    // Simple variable
                    Ok(Expression::Variable(name))
                }
            }
            Token::LeftParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume(Token::RightParen, "Expected ')'")?;
                Ok(expr)
            }
            Token::If => {
                self.advance();
                let condition = self.parse_constraint_expression()?;
                self.consume(Token::Then, "Expected 'then'")?;
                let then_expr = self.parse_expression()?;
                self.consume(Token::Else, "Expected 'else'")?;
                let else_expr = self.parse_expression()?;
                
                Ok(Expression::Conditional {
                    condition: Box::new(condition),
                    then_expr: Box::new(then_expr),
                    else_expr: Box::new(else_expr),
                })
            }
            _ => Err(ParseError {
                message: format!("Unexpected token in expression: {:?}", self.peek()),
                line: 0,
                column: 0,
            }),
        }
    }
    
    /// Helper functions
    fn peek(&self) -> &Token {
        &self.tokens[self.position]
    }
    
    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.position += 1;
        }
        &self.tokens[self.position - 1]
    }
    
    fn is_at_end(&self) -> bool {
        matches!(self.peek(), Token::Eof)
    }
    
    fn consume(&mut self, token: Token, message: &str) -> Result<(), ParseError> {
        if self.peek() == &token {
            self.advance();
            Ok(())
        } else {
            Err(ParseError {
                message: message.to_string(),
                line: 0,
                column: 0,
            })
        }
    }
    
    fn parse_identifier(&mut self) -> Result<String, ParseError> {
        if let Token::Identifier(name) = self.peek().clone() {
            self.advance();
            Ok(name)
        } else {
            Err(ParseError {
                message: "Expected identifier".to_string(),
                line: 0,
                column: 0,
            })
        }
    }
    
    fn parse_number(&mut self) -> Result<f64, ParseError> {
        if let Token::Number(n) = self.peek() {
            let n = *n;
            self.advance();
            Ok(n)
        } else {
            Err(ParseError {
                message: "Expected number".to_string(),
                line: 0,
                column: 0,
            })
        }
    }
}

impl TypeChecker {
    /// Create new type checker
    fn new() -> Self {
        Self {
            var_types: HashMap::new(),
            func_signatures: HashMap::new(),
            errors: Vec::new(),
        }
    }
    
    /// Type check AST
    fn check(&mut self, ast: &AST) -> Result<(), Vec<TypeError>> {
        match ast {
            AST::Program { declarations, objective, constraints } => {
                // Check declarations
                for decl in declarations {
                    self.check_declaration(decl)?;
                }
                
                // Check objective
                self.check_objective(objective)?;
                
                // Check constraints
                for constraint in constraints {
                    self.check_constraint(constraint)?;
                }
                
                if self.errors.is_empty() {
                    Ok(())
                } else {
                    Err(self.errors.clone())
                }
            }
            _ => Ok(()),
        }
    }
    
    /// Check declaration
    fn check_declaration(&mut self, decl: &Declaration) -> Result<(), TypeError> {
        match decl {
            Declaration::Variable { name, var_type, .. } => {
                self.var_types.insert(name.clone(), var_type.clone());
                Ok(())
            }
            _ => Ok(()), // TODO: implement other declaration types
        }
    }
    
    /// Check objective
    fn check_objective(&mut self, objective: &Objective) -> Result<(), TypeError> {
        match objective {
            Objective::Minimize(expr) | Objective::Maximize(expr) => {
                let expr_type = self.infer_expression_type(expr)?;
                // Objective must be numeric
                match expr_type {
                    VarType::Continuous | VarType::Integer => Ok(()),
                    _ => {
                        self.errors.push(TypeError {
                            message: "Objective must be numeric".to_string(),
                            location: "objective".to_string(),
                        });
                        Err(TypeError {
                            message: "Type error".to_string(),
                            location: "objective".to_string(),
                        })
                    }
                }
            }
            _ => Ok(()), // TODO: multi-objective
        }
    }
    
    /// Check constraint
    fn check_constraint(&mut self, constraint: &Constraint) -> Result<(), TypeError> {
        self.check_constraint_expression(&constraint.expression)
    }
    
    /// Check constraint expression
    fn check_constraint_expression(&mut self, expr: &ConstraintExpression) -> Result<(), TypeError> {
        match expr {
            ConstraintExpression::Comparison { left, right, .. } => {
                let left_type = self.infer_expression_type(left)?;
                let right_type = self.infer_expression_type(right)?;
                
                // Types must be compatible
                if !self.types_compatible(&left_type, &right_type) {
                    self.errors.push(TypeError {
                        message: format!("Incompatible types in comparison: {:?} and {:?}", left_type, right_type),
                        location: "constraint".to_string(),
                    });
                }
                
                Ok(())
            }
            ConstraintExpression::Logical { operands, .. } => {
                for operand in operands {
                    self.check_constraint_expression(operand)?;
                }
                Ok(())
            }
            _ => Ok(()), // TODO: other constraint types
        }
    }
    
    /// Infer expression type
    fn infer_expression_type(&self, expr: &Expression) -> Result<VarType, TypeError> {
        match expr {
            Expression::Literal(value) => {
                match value {
                    Value::Number(_) => Ok(VarType::Continuous),
                    Value::Boolean(_) => Ok(VarType::Binary),
                    _ => Ok(VarType::Continuous), // Default
                }
            }
            Expression::Variable(name) => {
                self.var_types.get(name).cloned().ok_or(TypeError {
                    message: format!("Unknown variable: {}", name),
                    location: name.clone(),
                })
            }
            Expression::BinaryOp { left, right, .. } => {
                let left_type = self.infer_expression_type(left)?;
                let right_type = self.infer_expression_type(right)?;
                
                // Result type depends on operand types
                match (&left_type, &right_type) {
                    (VarType::Continuous, _) | (_, VarType::Continuous) => Ok(VarType::Continuous),
                    (VarType::Integer, VarType::Integer) => Ok(VarType::Integer),
                    (VarType::Binary, VarType::Binary) => Ok(VarType::Binary),
                    _ => Ok(VarType::Continuous),
                }
            }
            _ => Ok(VarType::Continuous), // TODO: other expression types
        }
    }
    
    /// Check if types are compatible
    fn types_compatible(&self, t1: &VarType, t2: &VarType) -> bool {
        match (t1, t2) {
            (VarType::Binary, VarType::Binary) => true,
            (VarType::Integer, VarType::Integer) => true,
            (VarType::Continuous, VarType::Continuous) => true,
            (VarType::Integer, VarType::Continuous) | (VarType::Continuous, VarType::Integer) => true,
            (VarType::Binary, VarType::Integer) | (VarType::Integer, VarType::Binary) => true,
            (VarType::Binary, VarType::Continuous) | (VarType::Continuous, VarType::Binary) => true,
            _ => false,
        }
    }
}

impl Default for StandardLibrary {
    fn default() -> Self {
        let mut stdlib = Self {
            functions: HashMap::new(),
            patterns: HashMap::new(),
            templates: HashMap::new(),
        };
        
        // Add built-in functions
        stdlib.add_builtin_functions();
        
        // Add common patterns
        stdlib.add_common_patterns();
        
        // Add problem templates
        stdlib.add_problem_templates();
        
        stdlib
    }
}

impl ImportResolver {
    /// Create new import resolver
    pub fn new() -> Self {
        Self {
            paths: vec![".".to_string(), "./lib".to_string()],
            modules: HashMap::new(),
            symbols: HashMap::new(),
        }
    }
    
    /// Import module
    pub fn import_module(&mut self, module_path: &str, alias: Option<String>) -> Result<(), String> {
        // Try to find module in paths
        for path in &self.paths {
            let full_path = format!("{}/{}.dsl", path, module_path);
            if std::path::Path::new(&full_path).exists() {
                // Load module (simplified - in real implementation would parse file)
                let module_name = alias.unwrap_or_else(|| module_path.to_string());
                let module = Module {
                    name: module_name.clone(),
                    exports: HashMap::new(),
                };
                self.modules.insert(module_name, module);
                return Ok(());
            }
        }
        Err(format!("Module '{}' not found", module_path))
    }
    
    /// Resolve symbol
    pub fn resolve_symbol(&self, name: &str) -> Option<&ImportedSymbol> {
        self.symbols.get(name)
    }
}

impl StandardLibrary {
    /// Add built-in functions
    fn add_builtin_functions(&mut self) {
        // Mathematical functions
        self.functions.insert("abs".to_string(), BuiltinFunction {
            name: "abs".to_string(),
            signature: FunctionSignature {
                param_types: vec![VarType::Continuous],
                return_type: VarType::Continuous,
            },
            description: "Absolute value".to_string(),
            implementation: FunctionImpl::Native,
        });
        
        self.functions.insert("sqrt".to_string(), BuiltinFunction {
            name: "sqrt".to_string(),
            signature: FunctionSignature {
                param_types: vec![VarType::Continuous],
                return_type: VarType::Continuous,
            },
            description: "Square root".to_string(),
            implementation: FunctionImpl::Native,
        });
        
        // Logical functions
        self.functions.insert("implies".to_string(), BuiltinFunction {
            name: "implies".to_string(),
            signature: FunctionSignature {
                param_types: vec![VarType::Binary, VarType::Binary],
                return_type: VarType::Binary,
            },
            description: "Logical implication".to_string(),
            implementation: FunctionImpl::Native,
        });
    }
    
    /// Add common patterns
    fn add_common_patterns(&mut self) {
        // One-hot encoding pattern
        self.patterns.insert("one_hot".to_string(), Pattern {
            name: "one_hot".to_string(),
            description: "Exactly one variable is true".to_string(),
            parameters: vec!["variables".to_string()],
            expansion: AST::Program {
                declarations: vec![],
                objective: Objective::Minimize(Expression::Literal(Value::Number(0.0))),
                constraints: vec![],
            },
        });
        
        // At-most-k pattern
        self.patterns.insert("at_most_k".to_string(), Pattern {
            name: "at_most_k".to_string(),
            description: "At most k variables are true".to_string(),
            parameters: vec!["variables".to_string(), "k".to_string()],
            expansion: AST::Program {
                declarations: vec![],
                objective: Objective::Minimize(Expression::Literal(Value::Number(0.0))),
                constraints: vec![],
            },
        });
    }
    
    /// Add problem templates
    fn add_problem_templates(&mut self) {
        // TSP template
        self.templates.insert("tsp".to_string(), Template {
            name: "tsp".to_string(),
            description: "Traveling Salesman Problem template".to_string(),
            parameters: vec![
                TemplateParam {
                    name: "n_cities".to_string(),
                    param_type: "integer".to_string(),
                    default: None,
                },
                TemplateParam {
                    name: "distances".to_string(),
                    param_type: "matrix".to_string(),
                    default: None,
                },
            ],
            body: r#"
                var x[n_cities, n_cities] binary;
                
                minimize sum(i in 0..n_cities, j in 0..n_cities: distances[i,j] * x[i,j]);
                
                subject to
                    // Each city visited exactly once
                    forall(i in 0..n_cities): sum(j in 0..n_cities: x[i,j]) == 1;
                    forall(j in 0..n_cities): sum(i in 0..n_cities: x[i,j]) == 1;
                    
                    // Subtour elimination (simplified)
                    forall(i in 0..n_cities, j in 0..n_cities | i != j):
                        u[i] - u[j] + n_cities * x[i,j] <= n_cities - 1;
                    forall(j in 0..n_cities): sum(i in 0..n_cities: x[i,j]) == 1;
                    
                    // Subtour elimination would go here
            "#.to_string(),
        });
        
        // Knapsack template
        self.templates.insert("knapsack".to_string(), Template {
            name: "knapsack".to_string(),
            description: "0-1 Knapsack Problem template".to_string(),
            parameters: vec![
                TemplateParam {
                    name: "n_items".to_string(),
                    param_type: "integer".to_string(),
                    default: None,
                },
                TemplateParam {
                    name: "values".to_string(),
                    param_type: "array".to_string(),
                    default: None,
                },
                TemplateParam {
                    name: "weights".to_string(),
                    param_type: "array".to_string(),
                    default: None,
                },
                TemplateParam {
                    name: "capacity".to_string(),
                    param_type: "number".to_string(),
                    default: None,
                },
            ],
            body: r#"
                var x[n_items] binary;
                
                maximize sum(i in 0..n_items: values[i] * x[i]);
                
                subject to
                    sum(i in 0..n_items: weights[i] * x[i]) <= capacity;
            "#.to_string(),
        });
    }
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Basic,
            target: TargetBackend::QUBO,
            debug_info: false,
            warnings_as_errors: false,
        }
    }
}

/// QUBO compiler
pub struct QUBOCompiler {
    options: CompilerOptions,
    var_mapping: HashMap<String, usize>,
    qubo_matrix: Array2<f64>,
}

impl QUBOCompiler {
    /// Create new compiler
    fn new(options: CompilerOptions) -> Self {
        Self {
            options,
            var_mapping: HashMap::new(),
            qubo_matrix: Array2::zeros((0, 0)),
        }
    }
    
    /// Compile AST to QUBO
    fn compile(&self, ast: &AST) -> Result<(Array2<f64>, HashMap<String, usize>), String> {
        match ast {
            AST::Program { declarations, objective, constraints } => {
                // Create variable mapping
                let mut var_mapping = HashMap::new();
                let mut var_count = 0;
                
                for decl in declarations {
                    if let Declaration::Variable { name, var_type, .. } = decl {
                        match var_type {
                            VarType::Binary => {
                                var_mapping.insert(name.clone(), var_count);
                                var_count += 1;
                            }
                            VarType::Array { dimensions, .. } => {
                                let total_vars: usize = dimensions.iter().product();
                                for i in 0..total_vars {
                                    var_mapping.insert(format!("{}_{}", name, i), var_count);
                                    var_count += 1;
                                }
                            }
                            _ => {
                                return Err(format!("Unsupported variable type for QUBO: {:?}", var_type));
                            }
                        }
                    }
                }
                
                // Initialize QUBO matrix
                let mut qubo = Array2::zeros((var_count, var_count));
                
                // Compile objective
                self.compile_objective(&mut qubo, &var_mapping, objective)?;
                
                // Compile constraints
                for constraint in constraints {
                    self.compile_constraint(&mut qubo, &var_mapping, constraint)?;
                }
                
                Ok((qubo, var_mapping))
            }
            _ => Err("Invalid AST structure".to_string()),
        }
    }
    
    /// Compile objective function
    fn compile_objective(
        &self,
        qubo: &mut Array2<f64>,
        var_mapping: &HashMap<String, usize>,
        objective: &Objective,
    ) -> Result<(), String> {
        match objective {
            Objective::Minimize(expr) => {
                self.compile_expression(qubo, var_mapping, expr, 1.0)
            }
            Objective::Maximize(expr) => {
                self.compile_expression(qubo, var_mapping, expr, -1.0)
            }
            _ => Err("Multi-objective not supported for QUBO".to_string()),
        }
    }
    
    /// Compile expression
    fn compile_expression(
        &self,
        qubo: &mut Array2<f64>,
        var_mapping: &HashMap<String, usize>,
        expr: &Expression,
        coefficient: f64,
    ) -> Result<(), String> {
        match expr {
            Expression::Variable(name) => {
                if let Some(&idx) = var_mapping.get(name) {
                    qubo[[idx, idx]] += coefficient;
                    Ok(())
                } else {
                    Err(format!("Unknown variable: {}", name))
                }
            }
            Expression::BinaryOp { op, left, right } => {
                match op {
                    BinaryOperator::Add => {
                        self.compile_expression(qubo, var_mapping, left, coefficient)?;
                        self.compile_expression(qubo, var_mapping, right, coefficient)?;
                        Ok(())
                    }
                    BinaryOperator::Subtract => {
                        self.compile_expression(qubo, var_mapping, left, coefficient)?;
                        self.compile_expression(qubo, var_mapping, right, -coefficient)?;
                        Ok(())
                    }
                    BinaryOperator::Multiply => {
                        // Handle quadratic terms
                        if let (Expression::Variable(v1), Expression::Variable(v2)) = (left.as_ref(), right.as_ref()) {
                            if let (Some(&idx1), Some(&idx2)) = (var_mapping.get(v1), var_mapping.get(v2)) {
                                if idx1 == idx2 {
                                    qubo[[idx1, idx1]] += coefficient;
                                } else {
                                    qubo[[idx1, idx2]] += coefficient / 2.0;
                                    qubo[[idx2, idx1]] += coefficient / 2.0;
                                }
                                Ok(())
                            } else {
                                Err("Variable not found in multiplication".to_string())
                            }
                        } else {
                            Err("Only variable multiplication supported in QUBO".to_string())
                        }
                    }
                    _ => Err(format!("Unsupported operation for QUBO: {:?}", op)),
                }
            }
            Expression::Literal(Value::Number(n)) => {
                // Constant term - ignored in QUBO
                Ok(())
            }
            _ => Err("Expression type not supported for QUBO".to_string()),
        }
    }
    
    /// Compile constraint
    fn compile_constraint(
        &self,
        qubo: &mut Array2<f64>,
        var_mapping: &HashMap<String, usize>,
        constraint: &Constraint,
    ) -> Result<(), String> {
        // Convert constraint to penalty term
        let penalty = 1000.0; // Default penalty weight
        
        match &constraint.expression {
            ConstraintExpression::Comparison { left, op, right } => {
                match op {
                    ComparisonOp::Equal => {
                        // (left - right)^2 penalty
                        self.compile_expression(qubo, var_mapping, left, penalty)?;
                        self.compile_expression(qubo, var_mapping, right, -penalty)?;
                        // TODO: Add quadratic penalty terms
                        Ok(())
                    }
                    _ => Err(format!("Constraint type not supported for QUBO: {:?}", op)),
                }
            }
            _ => Err("Complex constraints not yet supported for QUBO".to_string()),
        }
    }
}

/// Example DSL programs
pub mod examples {
    /// Simple binary optimization
    pub const SIMPLE_BINARY: &str = r#"
        var x binary;
        var y binary;
        var z binary;
        
        minimize -2*x - 3*y - 4*z + 5*x*y + 6*y*z;
        
        subject to
            x + y + z <= 2;
    "#;
    
    /// Traveling salesman problem
    pub const TSP: &str = r#"
        param n = 4;
        param distances = [
            [0, 10, 15, 20],
            [10, 0, 25, 30],
            [15, 25, 0, 35],
            [20, 30, 35, 0]
        ];
        
        var x[n, n] binary;
        
        minimize sum(i in 0..n, j in 0..n: distances[i][j] * x[i,j]);
        
        subject to
            // Each city visited exactly once
            forall(i in 0..n): sum(j in 0..n: x[i,j]) == 1;
            forall(j in 0..n): sum(i in 0..n: x[i,j]) == 1;
    "#;
    
    /// Graph coloring
    pub const GRAPH_COLORING: &str = r#"
        param n_vertices = 5;
        param n_colors = 3;
        param edges = [(0,1), (1,2), (2,3), (3,4), (4,0)];
        
        var color[n_vertices, n_colors] binary;
        
        minimize sum(v in 0..n_vertices, c in 0..n_colors: c * color[v,c]);
        
        subject to
            // Each vertex has exactly one color
            forall(v in 0..n_vertices): sum(c in 0..n_colors: color[v,c]) == 1;
            
            // Adjacent vertices have different colors
            forall((u,v) in edges, c in 0..n_colors):
                color[u,c] + color[v,c] <= 1;
    "#;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenizer() {
        let dsl = ProblemDSL::new();
        let source = "var x binary;";
        let tokens = dsl.tokenize(source);
        
        assert!(tokens.is_ok());
        let tokens = tokens.unwrap();
        assert_eq!(tokens.len(), 5); // var, x, binary, ;, EOF
    }
    
    #[test]
    fn test_parser() {
        let mut dsl = ProblemDSL::new();
        let source = examples::SIMPLE_BINARY;
        let ast = dsl.parse(source);
        
        assert!(ast.is_ok());
    }
    
    #[test]
    fn test_type_checker() {
        let mut dsl = ProblemDSL::new();
        let source = examples::SIMPLE_BINARY;
        
        if let Ok(ast) = dsl.parse(source) {
            let result = dsl.type_check(&ast);
            assert!(result.is_ok());
        }
    }
}