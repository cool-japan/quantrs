//! Expression parser for mathematical expressions.
//!
//! This module provides a parser for converting string representations
//! of mathematical expressions into `Expression` objects.
//!
//! ## Supported Syntax
//!
//! - Numbers: `42`, `3.14`, `-2.5`, `1e-10`
//! - Variables: `x`, `theta`, `alpha_1`
//! - Constants: `pi`, `e`, `I` (imaginary unit)
//! - Operators: `+`, `-`, `*`, `/`, `^` (power)
//! - Functions: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`
//! - Parentheses: `(`, `)`
//!
//! ## Examples
//!
//! ```ignore
//! use quantrs2_symengine_pure::parser::parse;
//!
//! let expr = parse("sin(x) + cos(y)").unwrap();
//! let expr2 = parse("x^2 + 2*x + 1").unwrap();
//! ```

use crate::error::{SymEngineError, SymEngineResult};
use crate::expr::Expression;
use crate::ops::trig;

/// Token types for the lexer
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Identifier(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    Eof,
}

/// Lexer for tokenizing mathematical expressions
struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek();
        self.pos += 1;
        c
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self) -> Token {
        let mut s = String::new();
        let mut has_dot = false;
        let mut has_exp = false;

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.advance();
            } else if c == '.' && !has_dot && !has_exp {
                has_dot = true;
                s.push(c);
                self.advance();
            } else if (c == 'e' || c == 'E') && !has_exp {
                has_exp = true;
                s.push(c);
                self.advance();
                // Handle optional sign after exponent
                if let Some(next) = self.peek() {
                    if next == '+' || next == '-' {
                        s.push(next);
                        self.advance();
                    }
                }
            } else {
                break;
            }
        }

        let value = s.parse::<f64>().unwrap_or(0.0);
        Token::Number(value)
    }

    fn read_identifier(&mut self) -> Token {
        let mut s = String::new();

        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }

        Token::Identifier(s)
    }

    fn next_token(&mut self) -> SymEngineResult<Token> {
        self.skip_whitespace();

        match self.peek() {
            None => Ok(Token::Eof),
            Some(c) => {
                if c.is_ascii_digit()
                    || (c == '.'
                        && self
                            .input
                            .get(self.pos + 1)
                            .is_some_and(|n| n.is_ascii_digit()))
                {
                    Ok(self.read_number())
                } else if c.is_alphabetic() || c == '_' {
                    Ok(self.read_identifier())
                } else {
                    self.advance();
                    match c {
                        '+' => Ok(Token::Plus),
                        '-' => Ok(Token::Minus),
                        '*' => Ok(Token::Star),
                        '/' => Ok(Token::Slash),
                        '^' => Ok(Token::Caret),
                        '(' => Ok(Token::LParen),
                        ')' => Ok(Token::RParen),
                        ',' => Ok(Token::Comma),
                        _ => Err(SymEngineError::parse(format!("unexpected character: {c}"))),
                    }
                }
            }
        }
    }
}

/// Parser for mathematical expressions
struct Parser {
    lexer: Lexer,
    current: Token,
}

impl Parser {
    fn new(input: &str) -> SymEngineResult<Self> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        Ok(Self { lexer, current })
    }

    fn advance(&mut self) -> SymEngineResult<()> {
        self.current = self.lexer.next_token()?;
        Ok(())
    }

    fn expect(&mut self, expected: Token) -> SymEngineResult<()> {
        if std::mem::discriminant(&self.current) == std::mem::discriminant(&expected) {
            self.advance()
        } else {
            Err(SymEngineError::parse(format!(
                "expected {:?}, got {:?}",
                expected, self.current
            )))
        }
    }

    /// Parse a complete expression
    fn parse_expression(&mut self) -> SymEngineResult<Expression> {
        self.parse_additive()
    }

    /// Parse additive expressions: a + b, a - b
    fn parse_additive(&mut self) -> SymEngineResult<Expression> {
        let mut left = self.parse_multiplicative()?;

        loop {
            match &self.current {
                Token::Plus => {
                    self.advance()?;
                    let right = self.parse_multiplicative()?;
                    left = left + right;
                }
                Token::Minus => {
                    self.advance()?;
                    let right = self.parse_multiplicative()?;
                    left = left - right;
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Parse multiplicative expressions: a * b, a / b
    fn parse_multiplicative(&mut self) -> SymEngineResult<Expression> {
        let mut left = self.parse_power()?;

        loop {
            match &self.current {
                Token::Star => {
                    self.advance()?;
                    let right = self.parse_power()?;
                    left = left * right;
                }
                Token::Slash => {
                    self.advance()?;
                    let right = self.parse_power()?;
                    left = left / right;
                }
                _ => break,
            }
        }

        Ok(left)
    }

    /// Parse power expressions: a ^ b (right associative)
    fn parse_power(&mut self) -> SymEngineResult<Expression> {
        let base = self.parse_unary()?;

        if matches!(self.current, Token::Caret) {
            self.advance()?;
            let exp = self.parse_power()?; // Right associative
            Ok(base.pow(&exp))
        } else {
            Ok(base)
        }
    }

    /// Parse unary expressions: -a, +a
    fn parse_unary(&mut self) -> SymEngineResult<Expression> {
        match &self.current {
            Token::Minus => {
                self.advance()?;
                let expr = self.parse_unary()?;
                Ok(expr.neg())
            }
            Token::Plus => {
                self.advance()?;
                self.parse_unary()
            }
            _ => self.parse_primary(),
        }
    }

    /// Parse primary expressions: numbers, variables, function calls, parentheses
    fn parse_primary(&mut self) -> SymEngineResult<Expression> {
        match self.current.clone() {
            Token::Number(n) => {
                self.advance()?;
                Expression::float(n)
            }
            Token::Identifier(name) => {
                self.advance()?;

                // Check if this is a function call
                if matches!(self.current, Token::LParen) {
                    self.parse_function_call(&name)
                } else {
                    // It's a variable or constant
                    Ok(Self::get_constant_or_symbol(&name))
                }
            }
            Token::LParen => {
                self.advance()?;
                let expr = self.parse_expression()?;
                self.expect(Token::RParen)?;
                Ok(expr)
            }
            _ => Err(SymEngineError::parse(format!(
                "unexpected token: {:?}",
                self.current
            ))),
        }
    }

    /// Parse a function call: func(args...)
    fn parse_function_call(&mut self, name: &str) -> SymEngineResult<Expression> {
        self.expect(Token::LParen)?;

        let mut args = Vec::new();
        if !matches!(self.current, Token::RParen) {
            args.push(self.parse_expression()?);
            while matches!(self.current, Token::Comma) {
                self.advance()?;
                args.push(self.parse_expression()?);
            }
        }

        self.expect(Token::RParen)?;

        // Match known functions
        match name {
            "sin" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("sin requires 1 argument"));
                }
                Ok(trig::sin(&args[0]))
            }
            "cos" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("cos requires 1 argument"));
                }
                Ok(trig::cos(&args[0]))
            }
            "tan" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("tan requires 1 argument"));
                }
                Ok(trig::tan(&args[0]))
            }
            "exp" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("exp requires 1 argument"));
                }
                Ok(trig::exp(&args[0]))
            }
            "log" | "ln" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("log requires 1 argument"));
                }
                Ok(trig::log(&args[0]))
            }
            "sqrt" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("sqrt requires 1 argument"));
                }
                Ok(trig::sqrt(&args[0]))
            }
            "abs" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("abs requires 1 argument"));
                }
                Ok(trig::abs(&args[0]))
            }
            "sinh" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("sinh requires 1 argument"));
                }
                Ok(trig::sinh(&args[0]))
            }
            "cosh" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("cosh requires 1 argument"));
                }
                Ok(trig::cosh(&args[0]))
            }
            "tanh" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("tanh requires 1 argument"));
                }
                Ok(trig::tanh(&args[0]))
            }
            "asin" | "arcsin" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("asin requires 1 argument"));
                }
                Ok(trig::asin(&args[0]))
            }
            "acos" | "arccos" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("acos requires 1 argument"));
                }
                Ok(trig::acos(&args[0]))
            }
            "atan" | "arctan" => {
                if args.len() != 1 {
                    return Err(SymEngineError::parse("atan requires 1 argument"));
                }
                Ok(trig::atan(&args[0]))
            }
            "pow" => {
                if args.len() != 2 {
                    return Err(SymEngineError::parse("pow requires 2 arguments"));
                }
                Ok(args[0].pow(&args[1]))
            }
            _ => Err(SymEngineError::parse(format!("unknown function: {name}"))),
        }
    }

    /// Get a constant or create a symbol
    fn get_constant_or_symbol(name: &str) -> Expression {
        match name {
            "pi" | "PI" => Expression::pi(),
            "e" | "E" => Expression::e(),
            "i" | "I" => Expression::i(),
            _ => Expression::symbol(name),
        }
    }
}

/// Parse a mathematical expression from a string.
///
/// # Arguments
/// * `input` - The expression string to parse
///
/// # Returns
/// The parsed `Expression` or an error if parsing fails.
///
/// # Examples
///
/// ```ignore
/// use quantrs2_symengine_pure::parser::parse;
///
/// let expr = parse("x^2 + 2*x + 1").unwrap();
/// let expr2 = parse("sin(pi/2)").unwrap();
/// ```
///
/// # Errors
/// Returns `SymEngineError::ParseError` if the input is not a valid expression.
pub fn parse(input: &str) -> SymEngineResult<Expression> {
    if input.trim().is_empty() {
        return Err(SymEngineError::parse("empty expression"));
    }

    let mut parser = Parser::new(input)?;
    let expr = parser.parse_expression()?;

    // Ensure we consumed all input
    if !matches!(parser.current, Token::Eof) {
        return Err(SymEngineError::parse(format!(
            "unexpected token at end: {:?}",
            parser.current
        )));
    }

    Ok(expr)
}

/// Parse multiple expressions separated by semicolons.
///
/// # Arguments
/// * `input` - String containing multiple expressions separated by `;`
///
/// # Returns
/// Vector of parsed expressions.
pub fn parse_many(input: &str) -> SymEngineResult<Vec<Expression>> {
    input
        .split(';')
        .filter(|s| !s.trim().is_empty())
        .map(parse)
        .collect()
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_parse_number() {
        let expr = parse("42").expect("should parse");
        assert!(expr.is_number());
        assert!((expr.to_f64().expect("is number") - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_float() {
        let expr = parse("3.14").expect("should parse");
        assert!(expr.is_number());
        assert!((expr.to_f64().expect("is number") - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_parse_scientific() {
        let expr = parse("1e-10").expect("should parse");
        assert!(expr.is_number());
        assert!((expr.to_f64().expect("is number") - 1e-10).abs() < 1e-20);
    }

    #[test]
    fn test_parse_variable() {
        let expr = parse("x").expect("should parse");
        assert_eq!(expr.as_symbol(), Some("x"));
    }

    #[test]
    fn test_parse_constant_pi() {
        let expr = parse("pi").expect("should parse");
        assert_eq!(expr.as_symbol(), Some("pi"));
    }

    #[test]
    fn test_parse_addition() {
        let expr = parse("x + y").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);
        values.insert("y".to_string(), 4.0);

        let result = expr.eval(&values).expect("should eval");
        assert!((result - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_subtraction() {
        let expr = parse("x - y").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 10.0);
        values.insert("y".to_string(), 3.0);

        let result = expr.eval(&values).expect("should eval");
        assert!((result - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_multiplication() {
        let expr = parse("x * y").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);
        values.insert("y".to_string(), 4.0);

        let result = expr.eval(&values).expect("should eval");
        assert!((result - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_division() {
        let expr = parse("x / y").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 12.0);
        values.insert("y".to_string(), 4.0);

        let result = expr.eval(&values).expect("should eval");
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_power() {
        let expr = parse("x ^ 2").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let result = expr.eval(&values).expect("should eval");
        assert!((result - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_power_right_associative() {
        // 2^3^2 should be 2^(3^2) = 2^9 = 512, not (2^3)^2 = 64
        let expr = parse("2^3^2").expect("should parse");
        let result = expr.eval(&HashMap::new()).expect("should eval");
        assert!((result - 512.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_unary_minus() {
        let expr = parse("-x").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 5.0);

        let result = expr.eval(&values).expect("should eval");
        assert!((result - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_parse_parentheses() {
        let expr = parse("(x + y) * z").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 2.0);
        values.insert("y".to_string(), 3.0);
        values.insert("z".to_string(), 4.0);

        let result = expr.eval(&values).expect("should eval");
        assert!((result - 20.0).abs() < 1e-10); // (2+3)*4 = 20
    }

    #[test]
    fn test_parse_complex_expression() {
        let expr = parse("x^2 + 2*x + 1").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 3.0);

        let result = expr.eval(&values).expect("should eval");
        assert!((result - 16.0).abs() < 1e-10); // 9 + 6 + 1 = 16
    }

    #[test]
    fn test_parse_sin() {
        let expr = parse("sin(x)").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = expr.eval(&values).expect("should eval");
        assert!(result.abs() < 1e-10); // sin(0) = 0
    }

    #[test]
    fn test_parse_cos() {
        let expr = parse("cos(x)").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = expr.eval(&values).expect("should eval");
        assert!((result - 1.0).abs() < 1e-10); // cos(0) = 1
    }

    #[test]
    fn test_parse_exp() {
        let expr = parse("exp(x)").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = expr.eval(&values).expect("should eval");
        assert!((result - 1.0).abs() < 1e-10); // exp(0) = 1
    }

    #[test]
    fn test_parse_sqrt() {
        let expr = parse("sqrt(x)").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 4.0);

        let result = expr.eval(&values).expect("should eval");
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_nested_functions() {
        let expr = parse("sin(cos(x))").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 0.0);

        let result = expr.eval(&values).expect("should eval");
        // sin(cos(0)) = sin(1) ≈ 0.8414
        assert!((result - 0.841_470_984_8).abs() < 1e-6);
    }

    #[test]
    fn test_parse_combined() {
        let expr = parse("sin(x)^2 + cos(x)^2").expect("should parse");

        let mut values = HashMap::new();
        values.insert("x".to_string(), 1.5); // any value should give 1

        let result = expr.eval(&values).expect("should eval");
        assert!((result - 1.0).abs() < 1e-10); // sin²+cos² = 1
    }

    #[test]
    fn test_parse_many() {
        let exprs = parse_many("x + 1; y * 2; z ^ 3").expect("should parse");
        assert_eq!(exprs.len(), 3);
    }

    #[test]
    fn test_parse_empty_error() {
        let result = parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_syntax() {
        let result = parse("x + + y");
        // This might parse depending on the implementation
        // but at least it shouldn't panic
        let _ = result;
    }
}
