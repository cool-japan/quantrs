//! Serialization support for symbolic expressions.
//!
//! This module provides serialization and deserialization of expressions
//! using oxicode (following COOLJAPAN policy - not bincode).
//!
//! ## Example
//!
//! ```ignore
//! use quantrs2_symengine_pure::{Expression, serialize};
//!
//! let expr = Expression::symbol("x");
//!
//! // Serialize to bytes
//! let bytes = serialize::to_bytes(&expr)?;
//!
//! // Deserialize from bytes
//! let decoded: Expression = serialize::from_bytes(&bytes)?;
//! ```

use crate::error::{SymEngineError, SymEngineResult};
use crate::expr::Expression;
use crate::matrix::SymbolicMatrix;
use crate::parser;

/// A serializable form of an Expression.
///
/// Since Expression contains egg's RecExpr which doesn't implement
/// standard serialization traits, we serialize via the string representation.
#[derive(Clone, Debug)]
pub struct SerializedExpression {
    /// The expression as a string
    repr: String,
}

impl SerializedExpression {
    /// Create a serialized expression from an Expression
    #[must_use]
    pub fn from_expr(expr: &Expression) -> Self {
        Self {
            repr: expr.to_string(),
        }
    }

    /// Convert back to an Expression
    ///
    /// # Errors
    /// Returns error if parsing fails
    pub fn to_expr(&self) -> SymEngineResult<Expression> {
        // For simple expressions, try parsing
        // For complex egg s-expressions, we need special handling
        if self.repr.starts_with('(') {
            // It's an s-expression from egg, use Expression::new
            Ok(Expression::new(&self.repr))
        } else {
            // Try parsing as a mathematical expression
            parser::parse(&self.repr)
        }
    }
}

/// Serialize an Expression to bytes using oxicode.
///
/// # Arguments
/// * `expr` - The expression to serialize
///
/// # Returns
/// A vector of bytes containing the serialized expression.
///
/// # Errors
/// Returns error if serialization fails.
pub fn to_bytes(expr: &Expression) -> SymEngineResult<Vec<u8>> {
    let repr = expr.to_string();
    let len = repr.len() as u32;

    let mut bytes = Vec::with_capacity(4 + repr.len());

    // Write length as little-endian u32
    bytes.extend_from_slice(&len.to_le_bytes());

    // Write string bytes
    bytes.extend_from_slice(repr.as_bytes());

    Ok(bytes)
}

/// Deserialize an Expression from bytes.
///
/// # Arguments
/// * `bytes` - The bytes to deserialize from
///
/// # Returns
/// The deserialized expression.
///
/// # Errors
/// Returns error if deserialization or parsing fails.
pub fn from_bytes(bytes: &[u8]) -> SymEngineResult<Expression> {
    if bytes.len() < 4 {
        return Err(SymEngineError::parse("buffer too short for expression"));
    }

    let len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;

    if bytes.len() < 4 + len {
        return Err(SymEngineError::parse(
            "buffer too short for expression data",
        ));
    }

    let repr = std::str::from_utf8(&bytes[4..4 + len])
        .map_err(|e| SymEngineError::parse(format!("invalid UTF-8: {e}")))?;

    // Parse the representation
    SerializedExpression {
        repr: repr.to_string(),
    }
    .to_expr()
}

/// Serialize multiple expressions to bytes.
///
/// # Arguments
/// * `exprs` - The expressions to serialize
///
/// # Returns
/// A vector of bytes containing the serialized expressions.
///
/// # Errors
/// Returns error if serialization fails.
pub fn to_bytes_many(exprs: &[Expression]) -> SymEngineResult<Vec<u8>> {
    let mut bytes = Vec::new();

    // Write count as u32
    let count = exprs.len() as u32;
    bytes.extend_from_slice(&count.to_le_bytes());

    // Serialize each expression
    for expr in exprs {
        let expr_bytes = to_bytes(expr)?;
        bytes.extend_from_slice(&expr_bytes);
    }

    Ok(bytes)
}

/// Deserialize multiple expressions from bytes.
///
/// # Arguments
/// * `bytes` - The bytes to deserialize from
///
/// # Returns
/// The deserialized expressions.
///
/// # Errors
/// Returns error if deserialization or parsing fails.
pub fn from_bytes_many(bytes: &[u8]) -> SymEngineResult<Vec<Expression>> {
    if bytes.len() < 4 {
        return Err(SymEngineError::parse("buffer too short for count"));
    }

    let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let mut offset = 4;
    let mut exprs = Vec::with_capacity(count);

    for _ in 0..count {
        if offset + 4 > bytes.len() {
            return Err(SymEngineError::parse("unexpected end of buffer"));
        }

        let len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;

        let total_size = 4 + len;
        if offset + total_size > bytes.len() {
            return Err(SymEngineError::parse("unexpected end of buffer"));
        }

        let expr = from_bytes(&bytes[offset..offset + total_size])?;
        exprs.push(expr);
        offset += total_size;
    }

    Ok(exprs)
}

// =========================================================================
// Matrix Serialization
// =========================================================================

/// Serialize a SymbolicMatrix to bytes.
///
/// # Arguments
/// * `matrix` - The matrix to serialize
///
/// # Returns
/// A vector of bytes containing the serialized matrix.
///
/// # Errors
/// Returns error if serialization fails.
pub fn matrix_to_bytes(matrix: &SymbolicMatrix) -> SymEngineResult<Vec<u8>> {
    let mut bytes = Vec::new();

    // Write dimensions
    let rows = matrix.nrows() as u32;
    let cols = matrix.ncols() as u32;
    bytes.extend_from_slice(&rows.to_le_bytes());
    bytes.extend_from_slice(&cols.to_le_bytes());

    // Serialize each element
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let expr_bytes = to_bytes(matrix.get(i, j))?;
            bytes.extend_from_slice(&expr_bytes);
        }
    }

    Ok(bytes)
}

/// Deserialize a SymbolicMatrix from bytes.
///
/// # Arguments
/// * `bytes` - The bytes to deserialize from
///
/// # Returns
/// The deserialized matrix.
///
/// # Errors
/// Returns error if deserialization or parsing fails.
pub fn matrix_from_bytes(bytes: &[u8]) -> SymEngineResult<SymbolicMatrix> {
    if bytes.len() < 8 {
        return Err(SymEngineError::parse(
            "buffer too short for matrix dimensions",
        ));
    }

    let rows = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let cols = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;

    let mut offset = 8;
    let mut elements = Vec::with_capacity(rows * cols);

    for _ in 0..(rows * cols) {
        if offset + 4 > bytes.len() {
            return Err(SymEngineError::parse("unexpected end of buffer"));
        }

        let len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;

        let total_size = 4 + len;
        if offset + total_size > bytes.len() {
            return Err(SymEngineError::parse("unexpected end of buffer"));
        }

        let expr = from_bytes(&bytes[offset..offset + total_size])?;
        elements.push(expr);
        offset += total_size;
    }

    SymbolicMatrix::from_flat(elements, rows, cols)
}

// =========================================================================
// JSON-like Human Readable Format
// =========================================================================

/// Serialize an Expression to a JSON-like human-readable format.
///
/// This produces a simple JSON object with the expression representation.
#[must_use]
pub fn to_json(expr: &Expression) -> String {
    format!("{{\"expr\":\"{}\"}}", escape_json(&expr.to_string()))
}

/// Deserialize an Expression from JSON-like format.
///
/// # Errors
/// Returns error if parsing fails.
pub fn from_json(json: &str) -> SymEngineResult<Expression> {
    // Simple JSON parsing - extract the "expr" field
    let json = json.trim();

    if !json.starts_with('{') || !json.ends_with('}') {
        return Err(SymEngineError::parse("invalid JSON: expected object"));
    }

    let inner = &json[1..json.len() - 1];

    // Find "expr":"..."
    if let Some(start) = inner.find("\"expr\":\"") {
        let value_start = start + 8;
        if let Some(end) = inner[value_start..].find('"') {
            let value = &inner[value_start..value_start + end];
            let unescaped = unescape_json(value);
            return SerializedExpression { repr: unescaped }.to_expr();
        }
    }

    Err(SymEngineError::parse("invalid JSON: missing 'expr' field"))
}

/// Escape a string for JSON
fn escape_json(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            _ => result.push(c),
        }
    }
    result
}

/// Unescape a JSON string
fn unescape_json(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => result.push('"'),
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('\\') | None => result.push('\\'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_serialize_simple() {
        let expr = Expression::symbol("x");
        let bytes = to_bytes(&expr).expect("should serialize");

        let decoded = from_bytes(&bytes).expect("should deserialize");

        // Verify by evaluation
        let mut values = HashMap::new();
        values.insert("x".to_string(), 5.0);

        let orig = expr.eval(&values).expect("should eval");
        let dec = decoded.eval(&values).expect("should eval");

        assert!((orig - dec).abs() < 1e-10);
    }

    #[test]
    fn test_serialize_number() {
        let expr = Expression::float_unchecked(3.14);
        let bytes = to_bytes(&expr).expect("should serialize");

        let decoded = from_bytes(&bytes).expect("should deserialize");

        let orig = expr.eval(&HashMap::new()).expect("should eval");
        let dec = decoded.eval(&HashMap::new()).expect("should eval");

        assert!((orig - dec).abs() < 1e-10);
    }

    #[test]
    fn test_serialize_many() {
        let exprs = vec![
            Expression::symbol("x"),
            Expression::symbol("y"),
            Expression::int(42),
        ];

        let bytes = to_bytes_many(&exprs).expect("should serialize");
        let decoded = from_bytes_many(&bytes).expect("should deserialize");

        assert_eq!(decoded.len(), 3);
    }

    #[test]
    fn test_serialize_matrix() {
        let matrix = SymbolicMatrix::identity(2);
        let bytes = matrix_to_bytes(&matrix).expect("should serialize");

        let decoded = matrix_from_bytes(&bytes).expect("should deserialize");

        assert_eq!(decoded.nrows(), 2);
        assert_eq!(decoded.ncols(), 2);
        assert!(decoded.get(0, 0).is_one());
        assert!(decoded.get(0, 1).is_zero());
    }

    #[test]
    fn test_json_serialize() {
        let expr = Expression::symbol("x");
        let json = to_json(&expr);

        assert!(json.contains("\"expr\":"));
        assert!(json.contains("\"x\""));

        let decoded = from_json(&json).expect("should parse");
        assert_eq!(decoded.as_symbol(), Some("x"));
    }

    #[test]
    fn test_json_escape() {
        let s = "hello\"world\\test";
        let escaped = escape_json(s);
        let unescaped = unescape_json(&escaped);
        assert_eq!(s, unescaped);
    }
}
