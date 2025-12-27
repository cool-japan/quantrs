//! Parser for `OpenQASM` 3.0

use super::ast::{
    BinaryOp, ClassicalRef, ComparisonOp, Condition, Declaration, Expression, ForLoop,
    GateDefinition, Literal, Measurement, QasmGate, QasmProgram, QasmRegister, QasmStatement,
    QubitRef, UnaryOp,
};
use std::collections::HashMap;
use std::str::FromStr;
use thiserror::Error;

/// Parser error types
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("Unexpected token: {0}")]
    UnexpectedToken(String),

    #[error("Expected {expected}, found {found}")]
    ExpectedToken { expected: String, found: String },

    #[error("Invalid syntax: {0}")]
    InvalidSyntax(String),

    #[error("Undefined identifier: {0}")]
    UndefinedIdentifier(String),

    #[error("Type mismatch: {0}")]
    TypeMismatch(String),

    #[error("Invalid number: {0}")]
    InvalidNumber(String),

    #[error("Unexpected end of input")]
    UnexpectedEof,

    #[error("Version mismatch: expected 3.0, found {0}")]
    VersionMismatch(String),
}

/// Token types for lexing
#[derive(Debug, Clone, PartialEq)]
enum Token {
    // Keywords
    OpenQasm,
    Include,
    Qubit,
    Bit,
    Gate,
    Measure,
    Reset,
    Barrier,
    If,
    Else,
    For,
    While,
    In,
    Const,
    Def,
    Return,
    Delay,
    Ctrl,
    Inv,
    Pow,

    // Identifiers and literals
    Identifier(String),
    Integer(i64),
    Float(f64),
    String(String),

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Power,
    Assign,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Not,
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,
    Arrow,

    // Delimiters
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Semicolon,
    Comma,
    Colon,
    Dot,

    // Special
    Eof,
}

/// Lexer for tokenizing QASM input
struct Lexer<'a> {
    input: &'a str,
    position: usize,
    current: Option<char>,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        let mut lexer = Lexer {
            input,
            position: 0,
            current: None,
        };
        lexer.advance();
        lexer
    }

    fn advance(&mut self) {
        self.current = self.input.chars().nth(self.position);
        if self.current.is_some() {
            self.position += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.chars().nth(self.position)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current {
            if ch.is_whitespace() {
                self.advance();
            } else if ch == '/' && self.peek() == Some('/') {
                // Skip line comment
                while self.current.is_some() && self.current != Some('\n') {
                    self.advance();
                }
            } else if ch == '/' && self.peek() == Some('*') {
                // Skip block comment
                self.advance(); // skip '/'
                self.advance(); // skip '*'
                while self.current.is_some() {
                    if self.current == Some('*') && self.peek() == Some('/') {
                        self.advance(); // skip '*'
                        self.advance(); // skip '/'
                        break;
                    }
                    self.advance();
                }
            } else {
                break;
            }
        }
    }

    fn read_identifier(&mut self) -> String {
        let mut result = String::new();
        while let Some(ch) = self.current {
            if ch.is_alphanumeric() || ch == '_' {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        result
    }

    fn read_number(&mut self) -> Result<Token, ParseError> {
        let mut result = String::new();
        let mut has_dot = false;

        while let Some(ch) = self.current {
            if ch.is_numeric() {
                result.push(ch);
                self.advance();
            } else if ch == '.' && !has_dot && self.peek().is_some_and(char::is_numeric) {
                has_dot = true;
                result.push(ch);
                self.advance();
            } else if ch == 'e' || ch == 'E' {
                result.push(ch);
                self.advance();
                if let Some(sign_ch) = self.current {
                    if sign_ch == '+' || sign_ch == '-' {
                        result.push(sign_ch);
                        self.advance();
                    }
                }
            } else {
                break;
            }
        }

        if has_dot || result.contains('e') || result.contains('E') {
            result
                .parse::<f64>()
                .map(Token::Float)
                .map_err(|_| ParseError::InvalidNumber(result))
        } else {
            result
                .parse::<i64>()
                .map(Token::Integer)
                .map_err(|_| ParseError::InvalidNumber(result))
        }
    }

    fn read_string(&mut self) -> Result<String, ParseError> {
        let mut result = String::new();
        self.advance(); // skip opening quote

        while let Some(ch) = self.current {
            if ch == '"' {
                self.advance(); // skip closing quote
                return Ok(result);
            } else if ch == '\\' {
                self.advance();
                match self.current {
                    Some('n') => result.push('\n'),
                    Some('t') => result.push('\t'),
                    Some('r') => result.push('\r'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    _ => return Err(ParseError::InvalidSyntax("Invalid escape sequence".into())),
                }
                self.advance();
            } else {
                result.push(ch);
                self.advance();
            }
        }

        Err(ParseError::UnexpectedEof)
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace();

        match self.current {
            None => Ok(Token::Eof),
            Some(ch) => match ch {
                '+' => {
                    self.advance();
                    Ok(Token::Plus)
                }
                '*' => {
                    self.advance();
                    if self.current == Some('*') {
                        self.advance();
                        Ok(Token::Power)
                    } else {
                        Ok(Token::Star)
                    }
                }
                '/' => {
                    self.advance();
                    Ok(Token::Slash)
                }
                '%' => {
                    self.advance();
                    Ok(Token::Percent)
                }
                '(' => {
                    self.advance();
                    Ok(Token::LeftParen)
                }
                ')' => {
                    self.advance();
                    Ok(Token::RightParen)
                }
                '[' => {
                    self.advance();
                    Ok(Token::LeftBracket)
                }
                ']' => {
                    self.advance();
                    Ok(Token::RightBracket)
                }
                '{' => {
                    self.advance();
                    Ok(Token::LeftBrace)
                }
                '}' => {
                    self.advance();
                    Ok(Token::RightBrace)
                }
                ';' => {
                    self.advance();
                    Ok(Token::Semicolon)
                }
                ',' => {
                    self.advance();
                    Ok(Token::Comma)
                }
                ':' => {
                    self.advance();
                    Ok(Token::Colon)
                }
                '.' => {
                    self.advance();
                    Ok(Token::Dot)
                }
                '~' => {
                    self.advance();
                    Ok(Token::BitNot)
                }
                '"' => self.read_string().map(Token::String),
                '-' => {
                    self.advance();
                    if self.current == Some('>') {
                        self.advance();
                        Ok(Token::Arrow)
                    } else {
                        Ok(Token::Minus)
                    }
                }
                '=' => {
                    self.advance();
                    if self.current == Some('=') {
                        self.advance();
                        Ok(Token::Eq)
                    } else {
                        Ok(Token::Assign)
                    }
                }
                '!' => {
                    self.advance();
                    if self.current == Some('=') {
                        self.advance();
                        Ok(Token::Ne)
                    } else {
                        Ok(Token::Not)
                    }
                }
                '<' => {
                    self.advance();
                    match self.current {
                        Some('=') => {
                            self.advance();
                            Ok(Token::Le)
                        }
                        Some('<') => {
                            self.advance();
                            Ok(Token::Shl)
                        }
                        _ => Ok(Token::Lt),
                    }
                }
                '>' => {
                    self.advance();
                    match self.current {
                        Some('=') => {
                            self.advance();
                            Ok(Token::Ge)
                        }
                        Some('>') => {
                            self.advance();
                            Ok(Token::Shr)
                        }
                        _ => Ok(Token::Gt),
                    }
                }
                '&' => {
                    self.advance();
                    if self.current == Some('&') {
                        self.advance();
                        Ok(Token::And)
                    } else {
                        Ok(Token::BitAnd)
                    }
                }
                '|' => {
                    self.advance();
                    if self.current == Some('|') {
                        self.advance();
                        Ok(Token::Or)
                    } else {
                        Ok(Token::BitOr)
                    }
                }
                '^' => {
                    self.advance();
                    Ok(Token::BitXor)
                }
                _ if ch.is_alphabetic() || ch == '_' => {
                    let ident = self.read_identifier();
                    Ok(match ident.as_str() {
                        "OPENQASM" => Token::OpenQasm,
                        "include" => Token::Include,
                        "qubit" => Token::Qubit,
                        "bit" => Token::Bit,
                        "gate" => Token::Gate,
                        "measure" => Token::Measure,
                        "reset" => Token::Reset,
                        "barrier" => Token::Barrier,
                        "if" => Token::If,
                        "else" => Token::Else,
                        "for" => Token::For,
                        "while" => Token::While,
                        "in" => Token::In,
                        "const" => Token::Const,
                        "def" => Token::Def,
                        "return" => Token::Return,
                        "delay" => Token::Delay,
                        "ctrl" => Token::Ctrl,
                        "inv" => Token::Inv,
                        "pow" => Token::Pow,
                        "pi" => Token::Identifier("pi".into()),
                        "e" => Token::Identifier("e".into()),
                        "tau" => Token::Identifier("tau".into()),
                        _ => Token::Identifier(ident),
                    })
                }
                _ if ch.is_numeric() => self.read_number(),
                _ => Err(ParseError::UnexpectedToken(ch.to_string())),
            },
        }
    }
}

/// QASM parser
pub struct QasmParser<'a> {
    lexer: Lexer<'a>,
    current_token: Token,
    /// Symbol table for tracking declarations
    symbols: HashMap<String, SymbolType>,
}

#[derive(Debug, Clone)]
enum SymbolType {
    QuantumRegister(usize),
    ClassicalRegister(usize),
    Gate(Vec<String>, Vec<String>), // params, qubits
    Constant,
    Variable,
}

impl<'a> QasmParser<'a> {
    /// Create a new parser for the given input
    pub fn new(input: &'a str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let current_token = lexer.next_token()?;

        Ok(QasmParser {
            lexer,
            current_token,
            symbols: HashMap::new(),
        })
    }

    /// Parse a complete QASM program
    pub fn parse_program(&mut self) -> Result<QasmProgram, ParseError> {
        // Parse version declaration
        self.expect_token(&Token::OpenQasm)?;
        let version = self.parse_version()?;
        self.expect_token(&Token::Semicolon)?;

        // Parse includes
        let mut includes = Vec::new();
        while self.current_token == Token::Include {
            includes.push(self.parse_include()?);
        }

        // Parse declarations and statements
        let mut declarations = Vec::new();
        let mut statements = Vec::new();

        while self.current_token != Token::Eof {
            match &self.current_token {
                Token::Qubit => declarations.push(self.parse_quantum_register()?),
                Token::Bit => declarations.push(self.parse_classical_register()?),
                Token::Gate => declarations.push(self.parse_gate_definition()?),
                Token::Const => declarations.push(self.parse_constant()?),
                _ => statements.push(self.parse_statement()?),
            }
        }

        Ok(QasmProgram {
            version,
            includes,
            declarations,
            statements,
        })
    }

    fn advance(&mut self) -> Result<(), ParseError> {
        self.current_token = self.lexer.next_token()?;
        Ok(())
    }

    fn expect_token(&mut self, expected: &Token) -> Result<(), ParseError> {
        if std::mem::discriminant(&self.current_token) == std::mem::discriminant(expected) {
            self.advance()
        } else {
            Err(ParseError::ExpectedToken {
                expected: format!("{expected:?}"),
                found: format!("{:?}", self.current_token),
            })
        }
    }

    fn parse_version(&mut self) -> Result<String, ParseError> {
        match &self.current_token {
            Token::Float(v) => {
                let version = if *v == 3.0 {
                    "3.0".to_string()
                } else {
                    format!("{v}")
                };
                if !version.starts_with("3.") {
                    return Err(ParseError::VersionMismatch(version));
                }
                self.advance()?;
                Ok(version)
            }
            Token::Integer(v) if *v == 3 => {
                // Check if next token is a dot followed by a number
                self.advance()?;
                if self.current_token == Token::Dot {
                    self.advance()?;
                    if let Token::Integer(minor) = self.current_token.clone() {
                        let minor_val = minor;
                        self.advance()?;
                        Ok(format!("3.{minor_val}"))
                    } else {
                        Ok("3.0".to_string())
                    }
                } else {
                    Ok("3.0".to_string())
                }
            }
            _ => Err(ParseError::ExpectedToken {
                expected: "version number".into(),
                found: format!("{:?}", self.current_token),
            }),
        }
    }

    fn parse_include(&mut self) -> Result<String, ParseError> {
        self.expect_token(&Token::Include)?;

        match &self.current_token {
            Token::String(s) => {
                let include = s.clone();
                self.advance()?;
                self.expect_token(&Token::Semicolon)?;
                Ok(include)
            }
            _ => Err(ParseError::ExpectedToken {
                expected: "string".into(),
                found: format!("{:?}", self.current_token),
            }),
        }
    }

    fn parse_quantum_register(&mut self) -> Result<Declaration, ParseError> {
        self.expect_token(&Token::Qubit)?;

        let (size, name) = if self.current_token == Token::LeftBracket {
            // qubit[size] name format
            self.advance()?; // consume [

            let size = match &self.current_token {
                Token::Integer(n) => {
                    let size = *n as usize;
                    self.advance()?;
                    size
                }
                Token::Identifier(_) => {
                    // For constants like 'n', we'll use a default size of 4
                    // In a full implementation, we'd evaluate the constant
                    self.advance()?;
                    4 // placeholder
                }
                _ => {
                    return Err(ParseError::ExpectedToken {
                        expected: "integer or identifier".into(),
                        found: format!("{:?}", self.current_token),
                    })
                }
            };

            self.expect_token(&Token::RightBracket)?;

            let name = match &self.current_token {
                Token::Identifier(s) => s.clone(),
                _ => {
                    return Err(ParseError::ExpectedToken {
                        expected: "identifier".into(),
                        found: format!("{:?}", self.current_token),
                    })
                }
            };
            self.advance()?;

            (size, name)
        } else {
            // qubit name format (single qubit)
            let name = match &self.current_token {
                Token::Identifier(s) => s.clone(),
                _ => {
                    return Err(ParseError::ExpectedToken {
                        expected: "identifier".into(),
                        found: format!("{:?}", self.current_token),
                    })
                }
            };
            self.advance()?;

            (1, name)
        };

        self.expect_token(&Token::Semicolon)?;

        // Add to symbol table
        self.symbols
            .insert(name.clone(), SymbolType::QuantumRegister(size));

        Ok(Declaration::QuantumRegister(QasmRegister { name, size }))
    }

    fn parse_classical_register(&mut self) -> Result<Declaration, ParseError> {
        self.expect_token(&Token::Bit)?;

        let (size, name) = if self.current_token == Token::LeftBracket {
            // bit[size] name format
            self.advance()?; // consume [

            let size = match &self.current_token {
                Token::Integer(n) => {
                    let size = *n as usize;
                    self.advance()?;
                    size
                }
                Token::Identifier(_) => {
                    // For constants like 'n', we'll use a default size of 4
                    // In a full implementation, we'd evaluate the constant
                    self.advance()?;
                    4 // placeholder
                }
                _ => {
                    return Err(ParseError::ExpectedToken {
                        expected: "integer or identifier".into(),
                        found: format!("{:?}", self.current_token),
                    })
                }
            };

            self.expect_token(&Token::RightBracket)?;

            let name = match &self.current_token {
                Token::Identifier(s) => s.clone(),
                _ => {
                    return Err(ParseError::ExpectedToken {
                        expected: "identifier".into(),
                        found: format!("{:?}", self.current_token),
                    })
                }
            };
            self.advance()?;

            (size, name)
        } else {
            // bit name format (single bit)
            let name = match &self.current_token {
                Token::Identifier(s) => s.clone(),
                _ => {
                    return Err(ParseError::ExpectedToken {
                        expected: "identifier".into(),
                        found: format!("{:?}", self.current_token),
                    })
                }
            };
            self.advance()?;

            (1, name)
        };

        self.expect_token(&Token::Semicolon)?;

        // Add to symbol table
        self.symbols
            .insert(name.clone(), SymbolType::ClassicalRegister(size));

        Ok(Declaration::ClassicalRegister(QasmRegister { name, size }))
    }

    fn parse_gate_definition(&mut self) -> Result<Declaration, ParseError> {
        self.expect_token(&Token::Gate)?;

        let name = match &self.current_token {
            Token::Identifier(s) => s.clone(),
            _ => {
                return Err(ParseError::ExpectedToken {
                    expected: "identifier".into(),
                    found: format!("{:?}", self.current_token),
                })
            }
        };
        self.advance()?;

        // Parse parameters
        let mut params = Vec::new();
        if self.current_token == Token::LeftParen {
            self.advance()?;

            while self.current_token != Token::RightParen {
                match &self.current_token {
                    Token::Identifier(s) => {
                        params.push(s.clone());
                        self.advance()?;
                    }
                    _ => {
                        return Err(ParseError::ExpectedToken {
                            expected: "identifier".into(),
                            found: format!("{:?}", self.current_token),
                        })
                    }
                }

                if self.current_token == Token::Comma {
                    self.advance()?;
                }
            }

            self.expect_token(&Token::RightParen)?;
        }

        // Parse qubit arguments
        let mut qubits = Vec::new();
        while self.current_token != Token::LeftBrace {
            match &self.current_token {
                Token::Identifier(s) => {
                    qubits.push(s.clone());
                    self.advance()?;
                }
                _ => {
                    return Err(ParseError::ExpectedToken {
                        expected: "identifier".into(),
                        found: format!("{:?}", self.current_token),
                    })
                }
            }

            if self.current_token == Token::Comma {
                self.advance()?;
            }
        }

        // Parse body
        self.expect_token(&Token::LeftBrace)?;
        let mut body = Vec::new();

        while self.current_token != Token::RightBrace {
            body.push(self.parse_statement()?);
        }

        self.expect_token(&Token::RightBrace)?;

        // Add to symbol table
        self.symbols.insert(
            name.clone(),
            SymbolType::Gate(params.clone(), qubits.clone()),
        );

        Ok(Declaration::GateDefinition(GateDefinition {
            name,
            params,
            qubits,
            body,
        }))
    }

    fn parse_constant(&mut self) -> Result<Declaration, ParseError> {
        self.expect_token(&Token::Const)?;

        let name = match &self.current_token {
            Token::Identifier(s) => s.clone(),
            _ => {
                return Err(ParseError::ExpectedToken {
                    expected: "identifier".into(),
                    found: format!("{:?}", self.current_token),
                })
            }
        };
        self.advance()?;

        self.expect_token(&Token::Assign)?;

        let expr = self.parse_expression()?;

        self.expect_token(&Token::Semicolon)?;

        // Add to symbol table
        self.symbols.insert(name.clone(), SymbolType::Constant);

        Ok(Declaration::Constant(name, expr))
    }

    fn parse_statement(&mut self) -> Result<QasmStatement, ParseError> {
        match &self.current_token {
            Token::Measure => self.parse_measure(),
            Token::Reset => self.parse_reset(),
            Token::Barrier => self.parse_barrier(),
            Token::If => self.parse_if(),
            Token::For => self.parse_for(),
            Token::While => self.parse_while(),
            Token::Delay => self.parse_delay(),
            Token::Identifier(_) => {
                // Could be gate application, assignment, or function call
                self.parse_identifier_statement()
            }
            Token::Ctrl | Token::Inv | Token::Pow => self.parse_modified_gate(),
            _ => Err(ParseError::UnexpectedToken(format!(
                "{:?}",
                self.current_token
            ))),
        }
    }

    // Simplified implementations for brevity - full parser would implement all methods

    fn parse_measure(&mut self) -> Result<QasmStatement, ParseError> {
        self.expect_token(&Token::Measure)?;

        let mut qubits = Vec::new();
        let mut targets = Vec::new();

        // Parse first qubit -> classical pair
        qubits.push(self.parse_qubit_ref()?);
        self.expect_token(&Token::Arrow)?;
        targets.push(self.parse_classical_ref()?);

        // Parse additional pairs
        while self.current_token == Token::Comma {
            self.advance()?;
            qubits.push(self.parse_qubit_ref()?);
            self.expect_token(&Token::Arrow)?;
            targets.push(self.parse_classical_ref()?);
        }

        self.expect_token(&Token::Semicolon)?;

        Ok(QasmStatement::Measure(Measurement { qubits, targets }))
    }

    fn parse_reset(&mut self) -> Result<QasmStatement, ParseError> {
        self.expect_token(&Token::Reset)?;

        let mut qubits = Vec::new();
        qubits.push(self.parse_qubit_ref()?);

        while self.current_token == Token::Comma {
            self.advance()?;
            qubits.push(self.parse_qubit_ref()?);
        }

        self.expect_token(&Token::Semicolon)?;

        Ok(QasmStatement::Reset(qubits))
    }

    fn parse_barrier(&mut self) -> Result<QasmStatement, ParseError> {
        self.expect_token(&Token::Barrier)?;

        let mut qubits = Vec::new();

        if self.current_token != Token::Semicolon {
            qubits.push(self.parse_qubit_ref()?);

            while self.current_token == Token::Comma {
                self.advance()?;
                qubits.push(self.parse_qubit_ref()?);
            }
        }

        self.expect_token(&Token::Semicolon)?;

        Ok(QasmStatement::Barrier(qubits))
    }

    fn parse_if(&mut self) -> Result<QasmStatement, ParseError> {
        self.expect_token(&Token::If)?;
        self.expect_token(&Token::LeftParen)?;

        let condition = self.parse_condition()?;

        self.expect_token(&Token::RightParen)?;

        let statement = Box::new(self.parse_statement()?);

        Ok(QasmStatement::If(condition, statement))
    }

    fn parse_for(&mut self) -> Result<QasmStatement, ParseError> {
        self.expect_token(&Token::For)?;

        let variable = match &self.current_token {
            Token::Identifier(s) => s.clone(),
            _ => {
                return Err(ParseError::ExpectedToken {
                    expected: "identifier".into(),
                    found: format!("{:?}", self.current_token),
                })
            }
        };
        self.advance()?;

        self.expect_token(&Token::In)?;
        self.expect_token(&Token::LeftBracket)?;

        let start = self.parse_expression()?;
        self.expect_token(&Token::Colon)?;
        let end = self.parse_expression()?;

        let step = if self.current_token == Token::Colon {
            self.advance()?;
            Some(self.parse_expression()?)
        } else {
            None
        };

        self.expect_token(&Token::RightBracket)?;
        self.expect_token(&Token::LeftBrace)?;

        let mut body = Vec::new();
        while self.current_token != Token::RightBrace {
            body.push(self.parse_statement()?);
        }

        self.expect_token(&Token::RightBrace)?;

        Ok(QasmStatement::For(ForLoop {
            variable,
            start,
            end,
            step,
            body,
        }))
    }

    fn parse_while(&mut self) -> Result<QasmStatement, ParseError> {
        self.expect_token(&Token::While)?;
        self.expect_token(&Token::LeftParen)?;

        let condition = self.parse_condition()?;

        self.expect_token(&Token::RightParen)?;
        self.expect_token(&Token::LeftBrace)?;

        let mut body = Vec::new();
        while self.current_token != Token::RightBrace {
            body.push(self.parse_statement()?);
        }

        self.expect_token(&Token::RightBrace)?;

        Ok(QasmStatement::While(condition, body))
    }

    fn parse_delay(&mut self) -> Result<QasmStatement, ParseError> {
        self.expect_token(&Token::Delay)?;
        self.expect_token(&Token::LeftBracket)?;

        let duration = self.parse_expression()?;

        self.expect_token(&Token::RightBracket)?;

        let mut qubits = Vec::new();

        if self.current_token != Token::Semicolon {
            qubits.push(self.parse_qubit_ref()?);

            while self.current_token == Token::Comma {
                self.advance()?;
                qubits.push(self.parse_qubit_ref()?);
            }
        }

        self.expect_token(&Token::Semicolon)?;

        Ok(QasmStatement::Delay(duration, qubits))
    }

    fn parse_identifier_statement(&mut self) -> Result<QasmStatement, ParseError> {
        let name = match &self.current_token {
            Token::Identifier(s) => s.clone(),
            _ => return Err(ParseError::InvalidSyntax("Expected identifier".into())),
        };
        self.advance()?;

        match &self.current_token {
            Token::LeftParen => {
                // Function call or gate with parameters
                self.parse_gate_or_call(name)
            }
            Token::LeftBracket | Token::Identifier(_) => {
                // Gate application
                let mut gate = QasmGate {
                    name,
                    params: Vec::new(),
                    qubits: Vec::new(),
                    control: None,
                    inverse: false,
                    power: None,
                };

                // Parse qubits
                gate.qubits.push(self.parse_qubit_ref()?);

                while self.current_token == Token::Comma {
                    self.advance()?;
                    gate.qubits.push(self.parse_qubit_ref()?);
                }

                self.expect_token(&Token::Semicolon)?;

                Ok(QasmStatement::Gate(gate))
            }
            _ => Err(ParseError::InvalidSyntax("Invalid statement".into())),
        }
    }

    fn parse_modified_gate(&mut self) -> Result<QasmStatement, ParseError> {
        let mut control = None;
        let mut inverse = false;
        let mut power = None;

        // Parse modifiers
        loop {
            match &self.current_token {
                Token::Ctrl => {
                    self.advance()?;
                    if self.current_token == Token::LeftParen {
                        self.advance()?;
                        control = Some(match &self.current_token {
                            Token::Integer(n) => *n as usize,
                            _ => {
                                return Err(ParseError::ExpectedToken {
                                    expected: "integer".into(),
                                    found: format!("{:?}", self.current_token),
                                })
                            }
                        });
                        self.advance()?;
                        self.expect_token(&Token::RightParen)?;
                    } else {
                        control = Some(1);
                    }
                }
                Token::Inv => {
                    inverse = true;
                    self.advance()?;
                }
                Token::Pow => {
                    self.advance()?;
                    self.expect_token(&Token::LeftParen)?;
                    power = Some(self.parse_expression()?);
                    self.expect_token(&Token::RightParen)?;
                }
                _ => break,
            }
        }

        // Parse gate name
        let name = match &self.current_token {
            Token::Identifier(s) => s.clone(),
            _ => {
                return Err(ParseError::ExpectedToken {
                    expected: "gate name".into(),
                    found: format!("{:?}", self.current_token),
                })
            }
        };
        self.advance()?;

        // Parse parameters if present
        let mut params = Vec::new();
        if self.current_token == Token::LeftParen {
            self.advance()?;

            while self.current_token != Token::RightParen {
                params.push(self.parse_expression()?);

                if self.current_token == Token::Comma {
                    self.advance()?;
                }
            }

            self.expect_token(&Token::RightParen)?;
        }

        // Parse qubits
        let mut qubits = Vec::new();
        qubits.push(self.parse_qubit_ref()?);

        while self.current_token == Token::Comma {
            self.advance()?;
            qubits.push(self.parse_qubit_ref()?);
        }

        self.expect_token(&Token::Semicolon)?;

        Ok(QasmStatement::Gate(QasmGate {
            name,
            params,
            qubits,
            control,
            inverse,
            power,
        }))
    }

    fn parse_gate_or_call(&mut self, name: String) -> Result<QasmStatement, ParseError> {
        self.expect_token(&Token::LeftParen)?;

        let mut args = Vec::new();

        while self.current_token != Token::RightParen {
            args.push(self.parse_expression()?);

            if self.current_token == Token::Comma {
                self.advance()?;
            }
        }

        self.expect_token(&Token::RightParen)?;

        // Check if this is followed by qubits (gate) or semicolon (function call)
        match &self.current_token {
            Token::Identifier(_) | Token::LeftBracket => {
                // Gate with parameters
                let mut qubits = Vec::new();
                qubits.push(self.parse_qubit_ref()?);

                while self.current_token == Token::Comma {
                    self.advance()?;
                    qubits.push(self.parse_qubit_ref()?);
                }

                self.expect_token(&Token::Semicolon)?;

                Ok(QasmStatement::Gate(QasmGate {
                    name,
                    params: args,
                    qubits,
                    control: None,
                    inverse: false,
                    power: None,
                }))
            }
            Token::Semicolon => {
                // Function call
                self.advance()?;
                Ok(QasmStatement::Call(name, args))
            }
            _ => Err(ParseError::InvalidSyntax(
                "Expected qubits or semicolon".into(),
            )),
        }
    }

    fn parse_qubit_ref(&mut self) -> Result<QubitRef, ParseError> {
        let register = match &self.current_token {
            Token::Identifier(s) => s.clone(),
            _ => {
                return Err(ParseError::ExpectedToken {
                    expected: "register name".into(),
                    found: format!("{:?}", self.current_token),
                })
            }
        };
        self.advance()?;

        if self.current_token == Token::LeftBracket {
            self.advance()?;

            let start = match &self.current_token {
                Token::Integer(n) => *n as usize,
                _ => {
                    return Err(ParseError::ExpectedToken {
                        expected: "integer".into(),
                        found: format!("{:?}", self.current_token),
                    })
                }
            };
            self.advance()?;

            if self.current_token == Token::Colon {
                // Slice
                self.advance()?;
                let end = match &self.current_token {
                    Token::Integer(n) => *n as usize,
                    _ => {
                        return Err(ParseError::ExpectedToken {
                            expected: "integer".into(),
                            found: format!("{:?}", self.current_token),
                        })
                    }
                };
                self.advance()?;
                self.expect_token(&Token::RightBracket)?;

                Ok(QubitRef::Slice {
                    register,
                    start,
                    end,
                })
            } else {
                // Single index
                self.expect_token(&Token::RightBracket)?;
                Ok(QubitRef::Single {
                    register,
                    index: start,
                })
            }
        } else {
            // Entire register
            Ok(QubitRef::Register(register))
        }
    }

    fn parse_classical_ref(&mut self) -> Result<ClassicalRef, ParseError> {
        let register = match &self.current_token {
            Token::Identifier(s) => s.clone(),
            _ => {
                return Err(ParseError::ExpectedToken {
                    expected: "register name".into(),
                    found: format!("{:?}", self.current_token),
                })
            }
        };
        self.advance()?;

        if self.current_token == Token::LeftBracket {
            self.advance()?;

            let start = match &self.current_token {
                Token::Integer(n) => *n as usize,
                _ => {
                    return Err(ParseError::ExpectedToken {
                        expected: "integer".into(),
                        found: format!("{:?}", self.current_token),
                    })
                }
            };
            self.advance()?;

            if self.current_token == Token::Colon {
                // Slice
                self.advance()?;
                let end = match &self.current_token {
                    Token::Integer(n) => *n as usize,
                    _ => {
                        return Err(ParseError::ExpectedToken {
                            expected: "integer".into(),
                            found: format!("{:?}", self.current_token),
                        })
                    }
                };
                self.advance()?;
                self.expect_token(&Token::RightBracket)?;

                Ok(ClassicalRef::Slice {
                    register,
                    start,
                    end,
                })
            } else {
                // Single index
                self.expect_token(&Token::RightBracket)?;
                Ok(ClassicalRef::Single {
                    register,
                    index: start,
                })
            }
        } else {
            // Entire register
            Ok(ClassicalRef::Register(register))
        }
    }

    fn parse_expression(&mut self) -> Result<Expression, ParseError> {
        self.parse_or_expression()
    }

    fn parse_or_expression(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_and_expression()?;

        while self.current_token == Token::Or {
            self.advance()?;
            let right = self.parse_and_expression()?;
            left = Expression::Binary(BinaryOp::Or, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_and_expression(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_equality_expression()?;

        while self.current_token == Token::And {
            self.advance()?;
            let right = self.parse_equality_expression()?;
            left = Expression::Binary(BinaryOp::And, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_equality_expression(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_relational_expression()?;

        loop {
            let op = match &self.current_token {
                Token::Eq => BinaryOp::Eq,
                Token::Ne => BinaryOp::Ne,
                _ => break,
            };
            self.advance()?;

            let right = self.parse_relational_expression()?;
            left = Expression::Binary(op, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_relational_expression(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_additive_expression()?;

        loop {
            let op = match &self.current_token {
                Token::Lt => BinaryOp::Lt,
                Token::Le => BinaryOp::Le,
                Token::Gt => BinaryOp::Gt,
                Token::Ge => BinaryOp::Ge,
                _ => break,
            };
            self.advance()?;

            let right = self.parse_additive_expression()?;
            left = Expression::Binary(op, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_additive_expression(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_multiplicative_expression()?;

        loop {
            let op = match &self.current_token {
                Token::Plus => BinaryOp::Add,
                Token::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.advance()?;

            let right = self.parse_multiplicative_expression()?;
            left = Expression::Binary(op, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_multiplicative_expression(&mut self) -> Result<Expression, ParseError> {
        let mut left = self.parse_unary_expression()?;

        loop {
            let op = match &self.current_token {
                Token::Star => BinaryOp::Mul,
                Token::Slash => BinaryOp::Div,
                Token::Percent => BinaryOp::Mod,
                _ => break,
            };
            self.advance()?;

            let right = self.parse_unary_expression()?;
            left = Expression::Binary(op, Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_unary_expression(&mut self) -> Result<Expression, ParseError> {
        match &self.current_token {
            Token::Minus => {
                self.advance()?;
                Ok(Expression::Unary(
                    UnaryOp::Neg,
                    Box::new(self.parse_unary_expression()?),
                ))
            }
            Token::Not => {
                self.advance()?;
                Ok(Expression::Unary(
                    UnaryOp::Not,
                    Box::new(self.parse_unary_expression()?),
                ))
            }
            Token::BitNot => {
                self.advance()?;
                Ok(Expression::Unary(
                    UnaryOp::BitNot,
                    Box::new(self.parse_unary_expression()?),
                ))
            }
            _ => self.parse_postfix_expression(),
        }
    }

    fn parse_postfix_expression(&mut self) -> Result<Expression, ParseError> {
        let mut expr = self.parse_primary_expression()?;

        loop {
            match &self.current_token {
                Token::LeftBracket => {
                    self.advance()?;
                    let index = self.parse_expression()?;
                    self.expect_token(&Token::RightBracket)?;

                    match expr {
                        Expression::Variable(name) => {
                            expr = Expression::Index(name, Box::new(index));
                        }
                        _ => {
                            return Err(ParseError::InvalidSyntax(
                                "Cannot index non-variable".into(),
                            ))
                        }
                    }
                }
                Token::LeftParen => {
                    // Function call
                    self.advance()?;
                    let mut args = Vec::new();

                    while self.current_token != Token::RightParen {
                        args.push(self.parse_expression()?);
                        if self.current_token == Token::Comma {
                            self.advance()?;
                        }
                    }

                    self.expect_token(&Token::RightParen)?;

                    match expr {
                        Expression::Variable(name) => {
                            expr = Expression::Function(name, args);
                        }
                        _ => {
                            return Err(ParseError::InvalidSyntax(
                                "Cannot call non-function".into(),
                            ))
                        }
                    }
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary_expression(&mut self) -> Result<Expression, ParseError> {
        match &self.current_token {
            Token::Integer(n) => {
                let value = *n;
                self.advance()?;
                Ok(Expression::Literal(Literal::Integer(value)))
            }
            Token::Float(f) => {
                let value = *f;
                self.advance()?;
                Ok(Expression::Literal(Literal::Float(value)))
            }
            Token::String(s) => {
                let value = s.clone();
                self.advance()?;
                Ok(Expression::Literal(Literal::String(value)))
            }
            Token::Identifier(s) => {
                let name = s.clone();
                self.advance()?;

                // Check for special constants
                match name.as_str() {
                    "pi" => Ok(Expression::Literal(Literal::Pi)),
                    "e" => Ok(Expression::Literal(Literal::Euler)),
                    "tau" => Ok(Expression::Literal(Literal::Tau)),
                    _ => Ok(Expression::Variable(name)),
                }
            }
            Token::LeftParen => {
                self.advance()?;
                let expr = self.parse_expression()?;
                self.expect_token(&Token::RightParen)?;
                Ok(expr)
            }
            _ => Err(ParseError::UnexpectedToken(format!(
                "{:?}",
                self.current_token
            ))),
        }
    }

    fn parse_condition(&mut self) -> Result<Condition, ParseError> {
        let left = self.parse_expression()?;

        let op = match &self.current_token {
            Token::Eq => ComparisonOp::Eq,
            Token::Ne => ComparisonOp::Ne,
            Token::Lt => ComparisonOp::Lt,
            Token::Le => ComparisonOp::Le,
            Token::Gt => ComparisonOp::Gt,
            Token::Ge => ComparisonOp::Ge,
            _ => {
                return Err(ParseError::ExpectedToken {
                    expected: "comparison operator".into(),
                    found: format!("{:?}", self.current_token),
                })
            }
        };
        self.advance()?;

        let right = self.parse_expression()?;

        Ok(Condition { left, op, right })
    }
}

/// Parse a QASM 3.0 string into an AST
pub fn parse_qasm3(input: &str) -> Result<QasmProgram, ParseError> {
    let mut parser = QasmParser::new(input)?;
    parser.parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_circuit() {
        let input = r#"
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
measure q -> c;
"#;

        let result = parse_qasm3(input);
        assert!(result.is_ok());

        let program = result.expect("parse_qasm3 should succeed for valid input");
        assert_eq!(program.version, "3.0");
        assert_eq!(program.includes, vec!["stdgates.inc"]);
        assert_eq!(program.declarations.len(), 2);
        assert_eq!(program.statements.len(), 3);
    }

    #[test]
    fn test_parse_gate_definition() {
        let input = r"
OPENQASM 3.0;

gate mygate(theta) q {
    rx(theta) q;
    ry(theta/2) q;
}

qubit q;
mygate(pi/4) q;
";

        let result = parse_qasm3(input);
        assert!(result.is_ok());
    }
}
