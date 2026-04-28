//! Import OpenQASM 2.0 programs back to `QuantRS2` gate lists.
//!
//! This parser understands the subset of QASM 2.0 needed for round-tripping
//! circuits exported by `circuit_to_qasm`, plus common idioms produced by
//! Qiskit and other tools (e.g. `pi`-based angle expressions).

use super::error::QasmError;
use quantrs2_core::gate::GateOp;
use quantrs2_core::qubit::QubitId;
use std::collections::HashMap;

// ─── expression evaluator ───────────────────────────────────────────────────

/// Evaluate a simple QASM 2.0 angle expression to an `f64`.
///
/// Handles:
/// - Numeric literals (`1.5707963`, `0`)
/// - `pi`
/// - Binary `+`, `-`, `*`, `/` with `pi`
/// - Unary `-`
///
/// Anything more complex returns `ExpressionError`.
fn eval_expr(raw: &str) -> Result<f64, QasmError> {
    let s = raw.trim();

    // Direct float/int
    if let Ok(v) = s.parse::<f64>() {
        return Ok(v);
    }

    // Bare pi
    if s == "pi" {
        return Ok(std::f64::consts::PI);
    }

    // Unary minus: -expr
    if let Some(rest) = s.strip_prefix('-') {
        if !rest.contains('-') && !rest.contains('+') {
            return eval_expr(rest).map(|v| -v);
        }
    }

    // Binary operators — find the last +/- outside parentheses, then * and /
    // We handle a single binary op to cover common Qiskit outputs like `pi/2`, `2*pi`
    // Search right-to-left so that `-pi/2` splits as `(-pi) (/) (2)`.
    // Precedence: first try low-precedence +/- (rightmost), then */
    let chars: Vec<char> = s.chars().collect();
    let mut depth = 0i32;

    // Addition/subtraction (low precedence) — search right-to-left
    for i in (1..chars.len()).rev() {
        match chars[i] {
            ')' => depth += 1,
            '(' => depth -= 1,
            '+' | '-' if depth == 0 => {
                let lhs = &s[..i];
                let rhs = &s[i + 1..];
                let op = chars[i];
                if lhs.is_empty() {
                    // unary; handled above, should not be reached
                    continue;
                }
                let l = eval_expr(lhs)?;
                let r = eval_expr(rhs)?;
                return Ok(if op == '+' { l + r } else { l - r });
            }
            _ => {}
        }
    }

    // Multiplication/division — search right-to-left
    depth = 0;
    for i in (1..chars.len()).rev() {
        match chars[i] {
            ')' => depth += 1,
            '(' => depth -= 1,
            '*' | '/' if depth == 0 => {
                let lhs = &s[..i];
                let rhs = &s[i + 1..];
                let op = chars[i];
                let l = eval_expr(lhs)?;
                let r = eval_expr(rhs)?;
                if op == '/' {
                    if r == 0.0 {
                        return Err(QasmError::ExpressionError(
                            format!("division by zero in '{}'", s),
                        ));
                    }
                    return Ok(l / r);
                }
                return Ok(l * r);
            }
            _ => {}
        }
    }

    // Parenthesised expression
    if s.starts_with('(') && s.ends_with(')') {
        return eval_expr(&s[1..s.len() - 1]);
    }

    Err(QasmError::ExpressionError(s.to_string()))
}

// ─── tokenizer ──────────────────────────────────────────────────────────────

/// A single lexical token from a QASM 2.0 source line.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    Int(usize),
    Expr(String), // angle expression (may contain pi, operators)
    Arrow,        // ->
    Comma,
    Semicolon,
    LParen,
    RParen,
    LBracket,
    RBracket,
}

fn tokenize(line: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = line.char_indices().peekable();

    while let Some((i, ch)) = chars.peek().copied() {
        match ch {
            ' ' | '\t' | '\r' | '\n' => {
                chars.next();
            }
            '/' if line[i..].starts_with("//") => {
                // line comment — consume remainder
                break;
            }
            '-' if line[i..].starts_with("->") => {
                chars.next();
                chars.next();
                tokens.push(Token::Arrow);
            }
            ',' => {
                chars.next();
                tokens.push(Token::Comma);
            }
            ';' => {
                chars.next();
                tokens.push(Token::Semicolon);
            }
            '(' => {
                chars.next();
                tokens.push(Token::LParen);
            }
            ')' => {
                chars.next();
                tokens.push(Token::RParen);
            }
            '[' => {
                chars.next();
                tokens.push(Token::LBracket);
            }
            ']' => {
                chars.next();
                tokens.push(Token::RBracket);
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                // identifier
                let start = i;
                chars.next();
                while let Some((_, nc)) = chars.peek() {
                    if nc.is_ascii_alphanumeric() || *nc == '_' {
                        chars.next();
                    } else {
                        break;
                    }
                }
                // determine end index
                let end = chars.peek().map(|(idx, _)| *idx).unwrap_or(line.len());
                tokens.push(Token::Ident(line[start..end].to_string()));
            }
            c if c.is_ascii_digit() || c == '.' => {
                // numeric literal (integer or float)
                let start = i;
                chars.next();
                while let Some((_, nc)) = chars.peek() {
                    if nc.is_ascii_digit() || *nc == '.' || *nc == 'e' || *nc == 'E' || *nc == '+' || *nc == '-' {
                        chars.next();
                    } else {
                        break;
                    }
                }
                let end = chars.peek().map(|(idx, _)| *idx).unwrap_or(line.len());
                let s = &line[start..end];
                if let Ok(v) = s.parse::<usize>() {
                    tokens.push(Token::Int(v));
                } else {
                    tokens.push(Token::Expr(s.to_string()));
                }
            }
            '-' => {
                // Could be start of negative number or unary minus in expr
                // Consume it as part of an Expr token
                let start = i;
                chars.next();
                while let Some((_, nc)) = chars.peek() {
                    if nc.is_ascii_alphanumeric() || matches!(*nc, '.' | '_' | '+' | '-' | '*' | '/' | '(' | ')' | 'e' | 'E') {
                        chars.next();
                    } else {
                        break;
                    }
                }
                let end = chars.peek().map(|(idx, _)| *idx).unwrap_or(line.len());
                tokens.push(Token::Expr(line[start..end].to_string()));
            }
            _ => {
                chars.next(); // skip unknown
            }
        }
    }
    tokens
}

// ─── register table ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RegisterInfo {
    size: usize,
    is_classical: bool,
}

// ─── gate constructor ────────────────────────────────────────────────────────

fn make_gate(
    name: &str,
    params: &[f64],
    qubits: &[QubitId],
    line_no: usize,
) -> Result<Box<dyn GateOp>, QasmError> {
    use quantrs2_core::gate::multi::{CRX, CRY, CRZ, CNOT, CY, CZ, CH, Fredkin, SWAP, Toffoli};
    use quantrs2_core::gate::single::{
        Hadamard, Identity, PGate, PauliX, PauliY, PauliZ, Phase, PhaseDagger, RotationX,
        RotationY, RotationZ, SqrtX, SqrtXDagger, T, TDagger, UGate,
    };

    let q = |idx: usize| -> Result<QubitId, QasmError> {
        qubits.get(idx).copied().ok_or_else(|| QasmError::parse(
            line_no,
            format!("gate '{}' needs qubit index {}", name, idx),
        ))
    };

    let p = |idx: usize| -> Result<f64, QasmError> {
        params.get(idx).copied().ok_or_else(|| QasmError::WrongParameterCount {
            gate: name.to_string(),
            expected: idx + 1,
            actual: params.len(),
        })
    };

    match name {
        "id" => Ok(Box::new(Identity { target: q(0)? })),
        "x" => Ok(Box::new(PauliX { target: q(0)? })),
        "y" => Ok(Box::new(PauliY { target: q(0)? })),
        "z" => Ok(Box::new(PauliZ { target: q(0)? })),
        "h" => Ok(Box::new(Hadamard { target: q(0)? })),
        "s" => Ok(Box::new(Phase { target: q(0)? })),
        "sdg" => Ok(Box::new(PhaseDagger { target: q(0)? })),
        "t" => Ok(Box::new(T { target: q(0)? })),
        "tdg" => Ok(Box::new(TDagger { target: q(0)? })),
        "sx" => Ok(Box::new(SqrtX { target: q(0)? })),
        "sxdg" => Ok(Box::new(SqrtXDagger { target: q(0)? })),
        "rx" => Ok(Box::new(RotationX {
            target: q(0)?,
            theta: p(0)?,
        })),
        "ry" => Ok(Box::new(RotationY {
            target: q(0)?,
            theta: p(0)?,
        })),
        "rz" => Ok(Box::new(RotationZ {
            target: q(0)?,
            theta: p(0)?,
        })),
        // u1(λ) == P(λ)
        "u1" | "p" => Ok(Box::new(PGate {
            target: q(0)?,
            lambda: p(0)?,
        })),
        // u3(θ,φ,λ) == U(θ,φ,λ)
        "u3" | "u" => Ok(Box::new(UGate {
            target: q(0)?,
            theta: p(0)?,
            phi: p(1)?,
            lambda: p(2)?,
        })),
        // u2(φ,λ) ≡ U(π/2, φ, λ)
        "u2" => Ok(Box::new(UGate {
            target: q(0)?,
            theta: std::f64::consts::FRAC_PI_2,
            phi: p(0)?,
            lambda: p(1)?,
        })),
        "cx" => Ok(Box::new(CNOT {
            control: q(0)?,
            target: q(1)?,
        })),
        "cy" => Ok(Box::new(CY {
            control: q(0)?,
            target: q(1)?,
        })),
        "cz" => Ok(Box::new(CZ {
            control: q(0)?,
            target: q(1)?,
        })),
        "ch" => Ok(Box::new(CH {
            control: q(0)?,
            target: q(1)?,
        })),
        "crx" => Ok(Box::new(CRX {
            control: q(0)?,
            target: q(1)?,
            theta: p(0)?,
        })),
        "cry" => Ok(Box::new(CRY {
            control: q(0)?,
            target: q(1)?,
            theta: p(0)?,
        })),
        "crz" => Ok(Box::new(CRZ {
            control: q(0)?,
            target: q(1)?,
            theta: p(0)?,
        })),
        "swap" => Ok(Box::new(SWAP {
            qubit1: q(0)?,
            qubit2: q(1)?,
        })),
        "ccx" => Ok(Box::new(Toffoli {
            control1: q(0)?,
            control2: q(1)?,
            target: q(2)?,
        })),
        "cswap" => Ok(Box::new(Fredkin {
            control: q(0)?,
            target1: q(1)?,
            target2: q(2)?,
        })),
        _ => Err(QasmError::UnsupportedGate(name.to_string())),
    }
}

// ─── qubit reference parser ──────────────────────────────────────────────────

/// Parse a qubit reference like `q[0]` from a `[Ident, LBracket, Int, RBracket]` token slice.
/// Returns the `QubitId` and the number of tokens consumed.
fn parse_qubit_ref(
    tokens: &[Token],
    registers: &HashMap<String, RegisterInfo>,
    line_no: usize,
) -> Result<(QubitId, usize), QasmError> {
    if tokens.len() < 4 {
        return Err(QasmError::parse(line_no, "expected qubit reference like q[0]"));
    }
    let reg_name = match &tokens[0] {
        Token::Ident(s) => s.clone(),
        _ => return Err(QasmError::parse(line_no, "expected register name")),
    };
    if tokens[1] != Token::LBracket {
        return Err(QasmError::parse(line_no, "expected '[' after register name"));
    }
    let idx = match &tokens[2] {
        Token::Int(n) => *n,
        _ => return Err(QasmError::parse(line_no, "expected integer index")),
    };
    if tokens[3] != Token::RBracket {
        return Err(QasmError::parse(line_no, "expected ']' after index"));
    }

    let info = registers.get(&reg_name).ok_or_else(|| QasmError::UndefinedRegister(reg_name.clone()))?;
    if info.is_classical {
        return Err(QasmError::parse(line_no, format!("register '{}' is classical, expected quantum", reg_name)));
    }
    if idx >= info.size {
        return Err(QasmError::QubitIndexOutOfRange {
            register: reg_name,
            index: idx,
            size: info.size,
        });
    }

    Ok((QubitId(idx as u32), 4))
}

// ─── parser ──────────────────────────────────────────────────────────────────

/// Parse an OpenQASM 2.0 string.
///
/// Returns `(gates, num_qubits)` where `gates` is the ordered list of
/// quantum operations and `num_qubits` is the total size of the quantum
/// register(s) declared.
///
/// Only the first `qreg` declaration is used to determine `num_qubits`.
/// Multiple `qreg` declarations are not supported (use concatenated indices).
pub fn qasm_to_gates(qasm: &str) -> Result<(Vec<Box<dyn GateOp>>, usize), QasmError> {
    let mut registers: HashMap<String, RegisterInfo> = HashMap::new();
    let mut gates: Vec<Box<dyn GateOp>> = Vec::new();
    let mut num_qubits: usize = 0;

    for (line_idx, raw_line) in qasm.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = raw_line.trim();

        // Skip blank lines and comments
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        // Skip OPENQASM version header
        if line.starts_with("OPENQASM") {
            continue;
        }

        // Skip include directives
        if line.starts_with("include") {
            continue;
        }

        // Skip gate definitions (not needed for round-trip of standard gates)
        if line.starts_with("gate ") {
            // Consume until matching `}` — for now just skip the rest of the line
            // (full gate defs would need multi-line tracking)
            continue;
        }

        // Remove trailing semicolon (required for most statements)
        let stmt = line.trim_end_matches(';').trim();
        if stmt.is_empty() {
            continue;
        }

        let tokens = tokenize(stmt);
        if tokens.is_empty() {
            continue;
        }

        // qreg <name>[<size>]
        if matches!(&tokens[0], Token::Ident(s) if s == "qreg") {
            if tokens.len() < 4 {
                return Err(QasmError::parse(line_no, "malformed qreg declaration"));
            }
            let reg_name = match &tokens[1] {
                Token::Ident(s) => s.clone(),
                _ => return Err(QasmError::parse(line_no, "expected register name")),
            };
            let size = match &tokens[3] {
                Token::Int(n) => *n,
                _ => return Err(QasmError::parse(line_no, "expected register size")),
            };
            registers.insert(
                reg_name,
                RegisterInfo {
                    size,
                    is_classical: false,
                },
            );
            // Track total quantum register size
            num_qubits += size;
            continue;
        }

        // creg <name>[<size>]
        if matches!(&tokens[0], Token::Ident(s) if s == "creg") {
            if tokens.len() < 4 {
                return Err(QasmError::parse(line_no, "malformed creg declaration"));
            }
            let reg_name = match &tokens[1] {
                Token::Ident(s) => s.clone(),
                _ => return Err(QasmError::parse(line_no, "expected register name")),
            };
            let size = match &tokens[3] {
                Token::Int(n) => *n,
                _ => return Err(QasmError::parse(line_no, "expected register size")),
            };
            registers.insert(
                reg_name,
                RegisterInfo {
                    size,
                    is_classical: true,
                },
            );
            continue;
        }

        // barrier — skip (not a gate in our model)
        if matches!(&tokens[0], Token::Ident(s) if s == "barrier") {
            continue;
        }

        // measure <qreg>[i] -> <creg>[j]
        if matches!(&tokens[0], Token::Ident(s) if s == "measure") {
            // We skip measurements in the gate list (no Measure gate in GateOp for now)
            // but validate the qubit ref
            if tokens.len() >= 5 {
                let _ = parse_qubit_ref(&tokens[1..], &registers, line_no)?;
            }
            continue;
        }

        // reset <qreg>[i]
        if matches!(&tokens[0], Token::Ident(s) if s == "reset") {
            continue;
        }

        // Otherwise: gate invocation
        // Syntax: <name>[(<params>)] <qubit>[, <qubit>]*
        //
        // We use raw-string parsing for the angle params to correctly handle
        // expressions like `pi/2`, `2*pi`, etc., which contain operators that
        // the simple tokenizer doesn't handle.

        let gate_name = match &tokens[0] {
            Token::Ident(s) => s.to_lowercase(),
            _ => {
                return Err(QasmError::parse(line_no, format!("unexpected token: {:?}", tokens[0])));
            }
        };

        // Extract params and qubit-arguments portion from the raw `stmt` string.
        // Strategy: find the gate name in stmt, then check if '(' follows.
        //   If so, extract the raw content between '(' and matching ')',
        //   split by top-level commas, and eval_expr each segment.
        //   The qubit arguments start after the ')' (or after the gate name if no parens).

        let mut params: Vec<f64> = Vec::new();
        // `qubit_raw` is the slice of `stmt` starting at the qubit argument list
        let qubit_raw: &str;

        // Find where the gate name ends in stmt (skip leading whitespace already trimmed)
        let after_name = stmt[gate_name.len()..].trim_start();

        if after_name.starts_with('(') {
            // Find matching closing paren
            let mut depth = 0i32;
            let mut paren_end = None;
            for (ci, ch) in after_name.char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            paren_end = Some(ci);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let close_idx = paren_end.ok_or_else(|| {
                QasmError::parse(line_no, format!("unmatched '(' in gate '{}'", gate_name))
            })?;

            // The raw expression content is between '(' and ')'
            let expr_content = &after_name[1..close_idx];

            // Split by commas at depth 0
            let mut expr_start = 0;
            let mut d = 0i32;
            let mut expr_parts: Vec<&str> = Vec::new();
            for (ci, ch) in expr_content.char_indices() {
                match ch {
                    '(' => d += 1,
                    ')' => d -= 1,
                    ',' if d == 0 => {
                        expr_parts.push(&expr_content[expr_start..ci]);
                        expr_start = ci + 1;
                    }
                    _ => {}
                }
            }
            expr_parts.push(&expr_content[expr_start..]);

            for part in &expr_parts {
                let trimmed = part.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let val = eval_expr(trimmed).map_err(|e| QasmError::InvalidParameter {
                    gate: gate_name.clone(),
                    message: e.to_string(),
                })?;
                params.push(val);
            }

            qubit_raw = after_name[close_idx + 1..].trim();
        } else {
            qubit_raw = after_name;
        }

        // Parse qubit arguments from `qubit_raw` using the tokenizer
        let qubit_tokens = tokenize(qubit_raw);
        let mut qubits: Vec<QubitId> = Vec::new();
        let mut cursor = 0usize;

        while cursor < qubit_tokens.len() {
            // Skip commas between qubit args
            if qubit_tokens[cursor] == Token::Comma {
                cursor += 1;
                continue;
            }
            if !matches!(&qubit_tokens[cursor], Token::Ident(_)) {
                break;
            }
            let remaining = &qubit_tokens[cursor..];
            if remaining.len() < 4 {
                break;
            }
            let (qubit, consumed) = parse_qubit_ref(remaining, &registers, line_no)?;
            qubits.push(qubit);
            cursor += consumed;
        }

        if qubits.is_empty() {
            return Err(QasmError::parse(line_no, format!("gate '{}' has no qubit arguments", gate_name)));
        }

        let gate = make_gate(&gate_name, &params, &qubits, line_no)?;
        gates.push(gate);
    }

    Ok((gates, num_qubits))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_expr_numeric() {
        assert!((eval_expr("1.5707963").expect("eval") - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
        assert_eq!(eval_expr("0").expect("eval"), 0.0);
    }

    #[test]
    fn test_eval_expr_pi() {
        let pi = std::f64::consts::PI;
        assert!((eval_expr("pi").expect("eval") - pi).abs() < 1e-10);
        assert!((eval_expr("pi/2").expect("eval") - pi / 2.0).abs() < 1e-10);
        assert!((eval_expr("-pi/4").expect("eval") + pi / 4.0).abs() < 1e-10);
        assert!((eval_expr("2*pi").expect("eval") - 2.0 * pi).abs() < 1e-10);
        assert!((eval_expr("pi/4+pi/4").expect("eval") - pi / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_simple_circuit() {
        let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"#;
        let (gates, n) = qasm_to_gates(qasm).expect("parse");
        assert_eq!(n, 2);
        assert_eq!(gates.len(), 2);
        assert_eq!(gates[0].name(), "H");
        assert_eq!(gates[1].name(), "CNOT");
    }

    #[test]
    fn test_parse_rotation() {
        let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx(pi/2) q[0];
"#;
        let (gates, _) = qasm_to_gates(qasm).expect("parse");
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].name(), "RX");
    }

    #[test]
    fn test_parse_u3_gate() {
        let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
u3(1.0,2.0,3.0) q[0];
"#;
        let (gates, _) = qasm_to_gates(qasm).expect("parse");
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].name(), "U");
    }

    #[test]
    fn test_parse_toffoli() {
        let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
ccx q[0], q[1], q[2];
"#;
        let (gates, n) = qasm_to_gates(qasm).expect("parse");
        assert_eq!(n, 3);
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].name(), "Toffoli");
    }
}
