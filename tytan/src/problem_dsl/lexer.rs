//! Lexical analysis for the problem DSL.

use super::error::ParseError;
use std::fmt;

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

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::String(s) => write!(f, "\"{s}\""),
            Self::Boolean(b) => write!(f, "{b}"),
            Self::Identifier(id) => write!(f, "{id}"),
            Self::Var => write!(f, "var"),
            Self::Param => write!(f, "param"),
            Self::Constraint => write!(f, "constraint"),
            Self::Minimize => write!(f, "minimize"),
            Self::Maximize => write!(f, "maximize"),
            Self::Subject => write!(f, "subject"),
            Self::To => write!(f, "to"),
            Self::Binary => write!(f, "binary"),
            Self::Integer => write!(f, "integer"),
            Self::Continuous => write!(f, "continuous"),
            Self::In => write!(f, "in"),
            Self::ForAll => write!(f, "forall"),
            Self::Exists => write!(f, "exists"),
            Self::Sum => write!(f, "sum"),
            Self::Product => write!(f, "product"),
            Self::If => write!(f, "if"),
            Self::Then => write!(f, "then"),
            Self::Else => write!(f, "else"),
            Self::Let => write!(f, "let"),
            Self::Define => write!(f, "define"),
            Self::Macro => write!(f, "macro"),
            Self::Import => write!(f, "import"),
            Self::From => write!(f, "from"),
            Self::As => write!(f, "as"),
            Self::Domain => write!(f, "domain"),
            Self::Range => write!(f, "range"),
            Self::Symmetry => write!(f, "symmetry"),
            Self::Hint => write!(f, "hint"),
            Self::Plus => write!(f, "+"),
            Self::Minus => write!(f, "-"),
            Self::Times => write!(f, "*"),
            Self::Divide => write!(f, "/"),
            Self::Power => write!(f, "^"),
            Self::Equal => write!(f, "=="),
            Self::NotEqual => write!(f, "!="),
            Self::Less => write!(f, "<"),
            Self::Greater => write!(f, ">"),
            Self::LessEqual => write!(f, "<="),
            Self::GreaterEqual => write!(f, ">="),
            Self::And => write!(f, "&&"),
            Self::Or => write!(f, "||"),
            Self::Not => write!(f, "!"),
            Self::Implies => write!(f, "=>"),
            Self::Mod => write!(f, "%"),
            Self::Xor => write!(f, "xor"),
            Self::LeftParen => write!(f, "("),
            Self::RightParen => write!(f, ")"),
            Self::LeftBracket => write!(f, "["),
            Self::RightBracket => write!(f, "]"),
            Self::LeftBrace => write!(f, "{{"),
            Self::RightBrace => write!(f, "}}"),
            Self::Comma => write!(f, ","),
            Self::Semicolon => write!(f, ";"),
            Self::Colon => write!(f, ":"),
            Self::Arrow => write!(f, "->"),
            Self::Dot => write!(f, "."),
            Self::DoubleDot => write!(f, ".."),
            Self::Pipe => write!(f, "|"),
            Self::Eof => write!(f, "EOF"),
            Self::NewLine => write!(f, "\\n"),
            Self::Comment(c) => write!(f, "// {c}"),
        }
    }
}

/// Tokenize source code
pub fn tokenize(source: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let mut chars = source.chars().peekable();
    let mut line = 1;
    let mut column = 1;

    while let Some(&ch) = chars.peek() {
        match ch {
            // Whitespace
            ' ' | '\t' | '\r' => {
                chars.next();
                column += 1;
            }
            '\n' => {
                chars.next();
                tokens.push(Token::NewLine);
                line += 1;
                column = 1;
            }

            // Numbers
            '0'..='9' => {
                let mut number = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch.is_ascii_digit() || ch == '.' {
                        if let Some(c) = chars.next() {
                            number.push(c);
                            column += 1;
                        }
                    } else {
                        break;
                    }
                }
                let value = number.parse::<f64>().map_err(|_| ParseError {
                    message: format!("Invalid number: {number}"),
                    line,
                    column,
                })?;
                tokens.push(Token::Number(value));
            }

            // Strings
            '"' => {
                chars.next(); // consume opening quote
                column += 1;
                let mut string = String::new();
                while let Some(ch) = chars.next() {
                    column += 1;
                    if ch == '"' {
                        break;
                    } else if ch == '\\' {
                        if let Some(escaped) = chars.next() {
                            column += 1;
                            match escaped {
                                'n' => string.push('\n'),
                                't' => string.push('\t'),
                                'r' => string.push('\r'),
                                '\\' => string.push('\\'),
                                '"' => string.push('"'),
                                _ => {
                                    string.push('\\');
                                    string.push(escaped);
                                }
                            }
                        }
                    } else {
                        string.push(ch);
                    }
                }
                tokens.push(Token::String(string));
            }

            // Identifiers and keywords
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut identifier = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch.is_alphanumeric() || ch == '_' {
                        if let Some(c) = chars.next() {
                            identifier.push(c);
                            column += 1;
                        }
                    } else {
                        break;
                    }
                }

                let token = match identifier.as_str() {
                    "var" => Token::Var,
                    "param" => Token::Param,
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
                    "let" => Token::Let,
                    "define" => Token::Define,
                    "macro" => Token::Macro,
                    "import" => Token::Import,
                    "from" => Token::From,
                    "as" => Token::As,
                    "domain" => Token::Domain,
                    "range" => Token::Range,
                    "symmetry" => Token::Symmetry,
                    "hint" => Token::Hint,
                    "true" => Token::Boolean(true),
                    "false" => Token::Boolean(false),
                    "xor" => Token::Xor,
                    _ => Token::Identifier(identifier),
                };
                tokens.push(token);
            }

            // Comments
            '/' => {
                chars.next();
                column += 1;
                if chars.peek() == Some(&'/') {
                    chars.next();
                    column += 1;
                    let mut comment = String::new();
                    while let Some(&ch) = chars.peek() {
                        if ch == '\n' {
                            break;
                        }
                        if let Some(c) = chars.next() {
                            comment.push(c);
                            column += 1;
                        }
                    }
                    tokens.push(Token::Comment(comment));
                } else {
                    tokens.push(Token::Divide);
                }
            }

            // Operators and delimiters
            '+' => {
                chars.next();
                column += 1;
                tokens.push(Token::Plus);
            }
            '-' => {
                chars.next();
                column += 1;
                if chars.peek() == Some(&'>') {
                    chars.next();
                    column += 1;
                    tokens.push(Token::Arrow);
                } else {
                    tokens.push(Token::Minus);
                }
            }
            '*' => {
                chars.next();
                column += 1;
                tokens.push(Token::Times);
            }
            '^' => {
                chars.next();
                column += 1;
                tokens.push(Token::Power);
            }
            '%' => {
                chars.next();
                column += 1;
                tokens.push(Token::Mod);
            }
            '=' => {
                chars.next();
                column += 1;
                if chars.peek() == Some(&'=') {
                    chars.next();
                    column += 1;
                    tokens.push(Token::Equal);
                } else if chars.peek() == Some(&'>') {
                    chars.next();
                    column += 1;
                    tokens.push(Token::Implies);
                } else {
                    return Err(ParseError {
                        message: "Expected '==' or '=>'".to_string(),
                        line,
                        column,
                    });
                }
            }
            '!' => {
                chars.next();
                column += 1;
                if chars.peek() == Some(&'=') {
                    chars.next();
                    column += 1;
                    tokens.push(Token::NotEqual);
                } else {
                    tokens.push(Token::Not);
                }
            }
            '<' => {
                chars.next();
                column += 1;
                if chars.peek() == Some(&'=') {
                    chars.next();
                    column += 1;
                    tokens.push(Token::LessEqual);
                } else {
                    tokens.push(Token::Less);
                }
            }
            '>' => {
                chars.next();
                column += 1;
                if chars.peek() == Some(&'=') {
                    chars.next();
                    column += 1;
                    tokens.push(Token::GreaterEqual);
                } else {
                    tokens.push(Token::Greater);
                }
            }
            '&' => {
                chars.next();
                column += 1;
                if chars.peek() == Some(&'&') {
                    chars.next();
                    column += 1;
                    tokens.push(Token::And);
                } else {
                    return Err(ParseError {
                        message: "Expected '&&'".to_string(),
                        line,
                        column,
                    });
                }
            }
            '|' => {
                chars.next();
                column += 1;
                if chars.peek() == Some(&'|') {
                    chars.next();
                    column += 1;
                    tokens.push(Token::Or);
                } else {
                    tokens.push(Token::Pipe);
                }
            }
            '(' => {
                chars.next();
                column += 1;
                tokens.push(Token::LeftParen);
            }
            ')' => {
                chars.next();
                column += 1;
                tokens.push(Token::RightParen);
            }
            '[' => {
                chars.next();
                column += 1;
                tokens.push(Token::LeftBracket);
            }
            ']' => {
                chars.next();
                column += 1;
                tokens.push(Token::RightBracket);
            }
            '{' => {
                chars.next();
                column += 1;
                tokens.push(Token::LeftBrace);
            }
            '}' => {
                chars.next();
                column += 1;
                tokens.push(Token::RightBrace);
            }
            ',' => {
                chars.next();
                column += 1;
                tokens.push(Token::Comma);
            }
            ';' => {
                chars.next();
                column += 1;
                tokens.push(Token::Semicolon);
            }
            ':' => {
                chars.next();
                column += 1;
                tokens.push(Token::Colon);
            }
            '.' => {
                chars.next();
                column += 1;
                if chars.peek() == Some(&'.') {
                    chars.next();
                    column += 1;
                    tokens.push(Token::DoubleDot);
                } else {
                    tokens.push(Token::Dot);
                }
            }

            _ => {
                return Err(ParseError {
                    message: format!("Unexpected character: '{ch}'"),
                    line,
                    column,
                });
            }
        }
    }

    tokens.push(Token::Eof);
    Ok(tokens)
}
