//! Validator for `OpenQASM` 3.0 programs

use super::ast::{
    BinaryOp, ClassicalRef, Condition, Declaration, Expression, Literal, Measurement, QasmGate,
    QasmProgram, QasmStatement, QubitRef, UnaryOp,
};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Validation error types
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Undefined register: {0}")]
    UndefinedRegister(String),

    #[error("Undefined gate: {0}")]
    UndefinedGate(String),

    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),

    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error(
        "Index out of bounds: register {register} has size {size}, but index {index} was used"
    )]
    IndexOutOfBounds {
        register: String,
        size: usize,
        index: usize,
    },

    #[error("Parameter count mismatch: gate {gate} expects {expected} parameters, but {found} were provided")]
    ParameterCountMismatch {
        gate: String,
        expected: usize,
        found: usize,
    },

    #[error(
        "Qubit count mismatch: gate {gate} expects {expected} qubits, but {found} were provided"
    )]
    QubitCountMismatch {
        gate: String,
        expected: usize,
        found: usize,
    },

    #[error("Invalid slice: start index {start} is greater than end index {end}")]
    InvalidSlice { start: usize, end: usize },

    #[error("Duplicate declaration: {0}")]
    DuplicateDeclaration(String),

    #[error("Invalid control: {0}")]
    InvalidControl(String),

    #[error("Semantic error: {0}")]
    SemanticError(String),
}

/// Symbol information in the validator
#[derive(Debug, Clone)]
enum Symbol {
    QuantumRegister {
        size: usize,
    },
    ClassicalRegister {
        size: usize,
    },
    Gate {
        params: Vec<String>,
        qubits: Vec<String>,
    },
    Variable {
        typ: ValueType,
    },
    Constant {
        typ: ValueType,
    },
}

/// Value types in QASM
#[derive(Debug, Clone, PartialEq)]
enum ValueType {
    Bool,
    Int,
    Float,
    Angle,
    Duration,
    Qubit,
    Bit,
    String,
}

/// QASM validator
pub struct QasmValidator {
    /// Symbol table
    symbols: HashMap<String, Symbol>,
    /// Standard gates (name -> (`param_count`, `qubit_count`))
    standard_gates: HashMap<String, (usize, usize)>,
    /// Current scope for nested blocks
    scope_stack: Vec<HashMap<String, Symbol>>,
}

impl Default for QasmValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl QasmValidator {
    /// Create a new validator
    #[must_use]
    pub fn new() -> Self {
        let mut standard_gates = HashMap::new();

        // Single-qubit gates
        standard_gates.insert("id".to_string(), (0, 1));
        standard_gates.insert("x".to_string(), (0, 1));
        standard_gates.insert("y".to_string(), (0, 1));
        standard_gates.insert("z".to_string(), (0, 1));
        standard_gates.insert("h".to_string(), (0, 1));
        standard_gates.insert("s".to_string(), (0, 1));
        standard_gates.insert("sdg".to_string(), (0, 1));
        standard_gates.insert("t".to_string(), (0, 1));
        standard_gates.insert("tdg".to_string(), (0, 1));
        standard_gates.insert("sx".to_string(), (0, 1));
        standard_gates.insert("sxdg".to_string(), (0, 1));

        // Parametric single-qubit gates
        standard_gates.insert("rx".to_string(), (1, 1));
        standard_gates.insert("ry".to_string(), (1, 1));
        standard_gates.insert("rz".to_string(), (1, 1));
        standard_gates.insert("p".to_string(), (1, 1));
        standard_gates.insert("u1".to_string(), (1, 1));
        standard_gates.insert("u2".to_string(), (2, 1));
        standard_gates.insert("u3".to_string(), (3, 1));
        standard_gates.insert("u".to_string(), (3, 1));

        // Two-qubit gates
        standard_gates.insert("cx".to_string(), (0, 2));
        standard_gates.insert("cy".to_string(), (0, 2));
        standard_gates.insert("cz".to_string(), (0, 2));
        standard_gates.insert("ch".to_string(), (0, 2));
        standard_gates.insert("swap".to_string(), (0, 2));
        standard_gates.insert("iswap".to_string(), (0, 2));
        standard_gates.insert("ecr".to_string(), (0, 2));
        standard_gates.insert("dcx".to_string(), (0, 2));

        // Parametric two-qubit gates
        standard_gates.insert("crx".to_string(), (1, 2));
        standard_gates.insert("cry".to_string(), (1, 2));
        standard_gates.insert("crz".to_string(), (1, 2));
        standard_gates.insert("cp".to_string(), (1, 2));
        standard_gates.insert("cu1".to_string(), (1, 2));
        standard_gates.insert("rxx".to_string(), (1, 2));
        standard_gates.insert("ryy".to_string(), (1, 2));
        standard_gates.insert("rzz".to_string(), (1, 2));
        standard_gates.insert("rzx".to_string(), (1, 2));

        // Three-qubit gates
        standard_gates.insert("ccx".to_string(), (0, 3));
        standard_gates.insert("cswap".to_string(), (0, 3));

        Self {
            symbols: HashMap::new(),
            standard_gates,
            scope_stack: vec![],
        }
    }

    /// Validate a QASM program
    pub fn validate(&mut self, program: &QasmProgram) -> Result<(), ValidationError> {
        // Clear previous state
        self.symbols.clear();
        self.scope_stack.clear();

        // Add built-in constants
        self.symbols.insert(
            "pi".to_string(),
            Symbol::Constant {
                typ: ValueType::Float,
            },
        );
        self.symbols.insert(
            "e".to_string(),
            Symbol::Constant {
                typ: ValueType::Float,
            },
        );
        self.symbols.insert(
            "tau".to_string(),
            Symbol::Constant {
                typ: ValueType::Float,
            },
        );

        // Validate declarations
        for decl in &program.declarations {
            self.validate_declaration(decl)?;
        }

        // Validate statements
        for stmt in &program.statements {
            self.validate_statement(stmt)?;
        }

        Ok(())
    }

    /// Validate a declaration
    fn validate_declaration(&mut self, decl: &Declaration) -> Result<(), ValidationError> {
        match decl {
            Declaration::QuantumRegister(reg) => {
                if self.symbols.contains_key(&reg.name) {
                    return Err(ValidationError::DuplicateDeclaration(reg.name.clone()));
                }

                if reg.size == 0 {
                    return Err(ValidationError::SemanticError(
                        "Register size must be greater than 0".to_string(),
                    ));
                }

                self.symbols
                    .insert(reg.name.clone(), Symbol::QuantumRegister { size: reg.size });
            }
            Declaration::ClassicalRegister(reg) => {
                if self.symbols.contains_key(&reg.name) {
                    return Err(ValidationError::DuplicateDeclaration(reg.name.clone()));
                }

                if reg.size == 0 {
                    return Err(ValidationError::SemanticError(
                        "Register size must be greater than 0".to_string(),
                    ));
                }

                self.symbols.insert(
                    reg.name.clone(),
                    Symbol::ClassicalRegister { size: reg.size },
                );
            }
            Declaration::GateDefinition(def) => {
                if self.symbols.contains_key(&def.name) {
                    return Err(ValidationError::DuplicateDeclaration(def.name.clone()));
                }

                // Create new scope for gate body
                self.push_scope();

                // Add parameters to scope
                for param in &def.params {
                    self.add_to_current_scope(
                        param.clone(),
                        Symbol::Variable {
                            typ: ValueType::Angle,
                        },
                    );
                }

                // Add qubit arguments to scope
                for qubit in &def.qubits {
                    self.add_to_current_scope(
                        qubit.clone(),
                        Symbol::Variable {
                            typ: ValueType::Qubit,
                        },
                    );
                }

                // Validate gate body
                for stmt in &def.body {
                    self.validate_statement(stmt)?;
                }

                // Pop scope
                self.pop_scope();

                // Add gate to symbols
                self.symbols.insert(
                    def.name.clone(),
                    Symbol::Gate {
                        params: def.params.clone(),
                        qubits: def.qubits.clone(),
                    },
                );
            }
            Declaration::Constant(name, expr) => {
                if self.symbols.contains_key(name) {
                    return Err(ValidationError::DuplicateDeclaration(name.clone()));
                }

                let typ = self.validate_expression(expr)?;

                self.symbols.insert(name.clone(), Symbol::Constant { typ });
            }
        }

        Ok(())
    }

    /// Validate a statement
    fn validate_statement(&mut self, stmt: &QasmStatement) -> Result<(), ValidationError> {
        match stmt {
            QasmStatement::Gate(gate) => self.validate_gate(gate),
            QasmStatement::Measure(meas) => self.validate_measure(meas),
            QasmStatement::Reset(qubits) => {
                for qubit in qubits {
                    self.validate_qubit_ref(qubit)?;
                }
                Ok(())
            }
            QasmStatement::Barrier(qubits) => {
                for qubit in qubits {
                    self.validate_qubit_ref(qubit)?;
                }
                Ok(())
            }
            QasmStatement::Assignment(var, expr) => {
                let typ = self.validate_expression(expr)?;

                // Check if variable exists
                if let Some(symbol) = self.lookup_symbol(var) {
                    match symbol {
                        Symbol::Variable { typ: var_typ } => {
                            if !self.types_compatible(var_typ, &typ) {
                                return Err(ValidationError::TypeMismatch {
                                    expected: format!("{var_typ:?}"),
                                    found: format!("{typ:?}"),
                                });
                            }
                        }
                        _ => {
                            return Err(ValidationError::SemanticError(format!(
                                "{var} is not a variable"
                            )))
                        }
                    }
                } else {
                    // Create new variable
                    self.add_to_current_scope(var.clone(), Symbol::Variable { typ });
                }

                Ok(())
            }
            QasmStatement::If(cond, stmt) => {
                self.validate_condition(cond)?;
                self.validate_statement(stmt)
            }
            QasmStatement::For(for_loop) => {
                self.push_scope();

                // Add loop variable
                self.add_to_current_scope(
                    for_loop.variable.clone(),
                    Symbol::Variable {
                        typ: ValueType::Int,
                    },
                );

                // Validate range
                let start_typ = self.validate_expression(&for_loop.start)?;
                let end_typ = self.validate_expression(&for_loop.end)?;

                if start_typ != ValueType::Int || end_typ != ValueType::Int {
                    return Err(ValidationError::TypeMismatch {
                        expected: "int".to_string(),
                        found: "non-int".to_string(),
                    });
                }

                if let Some(step) = &for_loop.step {
                    let step_typ = self.validate_expression(step)?;
                    if step_typ != ValueType::Int {
                        return Err(ValidationError::TypeMismatch {
                            expected: "int".to_string(),
                            found: format!("{step_typ:?}"),
                        });
                    }
                }

                // Validate body
                for stmt in &for_loop.body {
                    self.validate_statement(stmt)?;
                }

                self.pop_scope();
                Ok(())
            }
            QasmStatement::While(cond, body) => {
                self.validate_condition(cond)?;

                self.push_scope();
                for stmt in body {
                    self.validate_statement(stmt)?;
                }
                self.pop_scope();

                Ok(())
            }
            QasmStatement::Call(name, args) => {
                // For now, just check that arguments are valid expressions
                for arg in args {
                    self.validate_expression(arg)?;
                }
                Ok(())
            }
            QasmStatement::Delay(duration, qubits) => {
                let dur_typ = self.validate_expression(duration)?;
                if dur_typ != ValueType::Duration && dur_typ != ValueType::Float {
                    return Err(ValidationError::TypeMismatch {
                        expected: "duration".to_string(),
                        found: format!("{dur_typ:?}"),
                    });
                }

                for qubit in qubits {
                    self.validate_qubit_ref(qubit)?;
                }

                Ok(())
            }
        }
    }

    /// Validate a gate application
    fn validate_gate(&mut self, gate: &QasmGate) -> Result<(), ValidationError> {
        // Check if gate exists
        let (expected_params, expected_qubits) =
            if let Some(&(p, q)) = self.standard_gates.get(&gate.name) {
                (p, q)
            } else if let Some(symbol) = self.symbols.get(&gate.name) {
                match symbol {
                    Symbol::Gate { params, qubits } => (params.len(), qubits.len()),
                    _ => return Err(ValidationError::UndefinedGate(gate.name.clone())),
                }
            } else {
                return Err(ValidationError::UndefinedGate(gate.name.clone()));
            };

        // Check parameter count
        if gate.params.len() != expected_params {
            return Err(ValidationError::ParameterCountMismatch {
                gate: gate.name.clone(),
                expected: expected_params,
                found: gate.params.len(),
            });
        }

        // Validate parameters
        for param in &gate.params {
            let typ = self.validate_expression(param)?;
            if typ != ValueType::Float && typ != ValueType::Angle && typ != ValueType::Int {
                return Err(ValidationError::TypeMismatch {
                    expected: "numeric".to_string(),
                    found: format!("{typ:?}"),
                });
            }
        }

        // Check qubit count (accounting for control modifier)
        let actual_qubits = gate.qubits.len();
        let required_qubits = expected_qubits + gate.control.unwrap_or(0);

        if actual_qubits != required_qubits {
            return Err(ValidationError::QubitCountMismatch {
                gate: gate.name.clone(),
                expected: required_qubits,
                found: actual_qubits,
            });
        }

        // Validate qubits
        for qubit in &gate.qubits {
            self.validate_qubit_ref(qubit)?;
        }

        Ok(())
    }

    /// Validate a measurement
    fn validate_measure(&mut self, meas: &Measurement) -> Result<(), ValidationError> {
        if meas.qubits.len() != meas.targets.len() {
            return Err(ValidationError::SemanticError(
                "Measurement must have equal number of qubits and classical bits".to_string(),
            ));
        }

        for qubit in &meas.qubits {
            self.validate_qubit_ref(qubit)?;
        }

        for target in &meas.targets {
            self.validate_classical_ref(target)?;
        }

        Ok(())
    }

    /// Validate a qubit reference
    fn validate_qubit_ref(&self, qubit_ref: &QubitRef) -> Result<(), ValidationError> {
        match qubit_ref {
            QubitRef::Single { register, index } => {
                match self.lookup_symbol(register) {
                    Some(Symbol::QuantumRegister { size }) => {
                        if *index >= *size {
                            return Err(ValidationError::IndexOutOfBounds {
                                register: register.clone(),
                                size: *size,
                                index: *index,
                            });
                        }
                    }
                    Some(Symbol::Variable {
                        typ: ValueType::Qubit,
                    }) => {
                        // Single qubit variable
                    }
                    _ => return Err(ValidationError::UndefinedRegister(register.clone())),
                }
            }
            QubitRef::Slice {
                register,
                start,
                end,
            } => match self.lookup_symbol(register) {
                Some(Symbol::QuantumRegister { size }) => {
                    if *start >= *size || *end > *size {
                        return Err(ValidationError::IndexOutOfBounds {
                            register: register.clone(),
                            size: *size,
                            index: (*start).max(*end),
                        });
                    }
                    if *start >= *end {
                        return Err(ValidationError::InvalidSlice {
                            start: *start,
                            end: *end,
                        });
                    }
                }
                _ => return Err(ValidationError::UndefinedRegister(register.clone())),
            },
            QubitRef::Register(name) => match self.lookup_symbol(name) {
                Some(Symbol::QuantumRegister { .. }) => {}
                Some(Symbol::Variable {
                    typ: ValueType::Qubit,
                }) => {}
                _ => return Err(ValidationError::UndefinedRegister(name.clone())),
            },
        }

        Ok(())
    }

    /// Validate a classical reference
    fn validate_classical_ref(&self, classical_ref: &ClassicalRef) -> Result<(), ValidationError> {
        match classical_ref {
            ClassicalRef::Single { register, index } => {
                match self.lookup_symbol(register) {
                    Some(Symbol::ClassicalRegister { size }) => {
                        if *index >= *size {
                            return Err(ValidationError::IndexOutOfBounds {
                                register: register.clone(),
                                size: *size,
                                index: *index,
                            });
                        }
                    }
                    Some(Symbol::Variable {
                        typ: ValueType::Bit,
                    }) => {
                        // Single bit variable
                    }
                    _ => return Err(ValidationError::UndefinedRegister(register.clone())),
                }
            }
            ClassicalRef::Slice {
                register,
                start,
                end,
            } => match self.lookup_symbol(register) {
                Some(Symbol::ClassicalRegister { size }) => {
                    if *start >= *size || *end > *size {
                        return Err(ValidationError::IndexOutOfBounds {
                            register: register.clone(),
                            size: *size,
                            index: (*start).max(*end),
                        });
                    }
                    if *start >= *end {
                        return Err(ValidationError::InvalidSlice {
                            start: *start,
                            end: *end,
                        });
                    }
                }
                _ => return Err(ValidationError::UndefinedRegister(register.clone())),
            },
            ClassicalRef::Register(name) => match self.lookup_symbol(name) {
                Some(Symbol::ClassicalRegister { .. }) => {}
                Some(Symbol::Variable {
                    typ: ValueType::Bit,
                }) => {}
                _ => return Err(ValidationError::UndefinedRegister(name.clone())),
            },
        }

        Ok(())
    }

    /// Validate an expression and return its type
    fn validate_expression(&self, expr: &Expression) -> Result<ValueType, ValidationError> {
        match expr {
            Expression::Literal(lit) => Ok(match lit {
                Literal::Integer(_) => ValueType::Int,
                Literal::Float(_) | Literal::Pi | Literal::Euler | Literal::Tau => ValueType::Float,
                Literal::Bool(_) => ValueType::Bool,
                Literal::String(_) => ValueType::String,
            }),
            Expression::Variable(name) => match self.lookup_symbol(name) {
                Some(Symbol::Variable { typ }) => Ok(typ.clone()),
                Some(Symbol::Constant { typ }) => Ok(typ.clone()),
                _ => Err(ValidationError::UndefinedVariable(name.clone())),
            },
            Expression::Binary(op, left, right) => {
                let left_typ = self.validate_expression(left)?;
                let right_typ = self.validate_expression(right)?;

                // Type checking for binary operations
                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div => {
                        if (left_typ == ValueType::Int || left_typ == ValueType::Float)
                            && (right_typ == ValueType::Int || right_typ == ValueType::Float)
                        {
                            Ok(ValueType::Float)
                        } else {
                            Err(ValidationError::TypeMismatch {
                                expected: "numeric".to_string(),
                                found: format!("{left_typ:?} and {right_typ:?}"),
                            })
                        }
                    }
                    BinaryOp::Mod
                    | BinaryOp::BitAnd
                    | BinaryOp::BitOr
                    | BinaryOp::BitXor
                    | BinaryOp::Shl
                    | BinaryOp::Shr => {
                        if left_typ == ValueType::Int && right_typ == ValueType::Int {
                            Ok(ValueType::Int)
                        } else {
                            Err(ValidationError::TypeMismatch {
                                expected: "int".to_string(),
                                found: format!("{left_typ:?} and {right_typ:?}"),
                            })
                        }
                    }
                    BinaryOp::And | BinaryOp::Or | BinaryOp::Xor => {
                        if left_typ == ValueType::Bool && right_typ == ValueType::Bool {
                            Ok(ValueType::Bool)
                        } else {
                            Err(ValidationError::TypeMismatch {
                                expected: "bool".to_string(),
                                found: format!("{left_typ:?} and {right_typ:?}"),
                            })
                        }
                    }
                    BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge => Ok(ValueType::Bool),
                    BinaryOp::Pow => {
                        if (left_typ == ValueType::Int || left_typ == ValueType::Float)
                            && (right_typ == ValueType::Int || right_typ == ValueType::Float)
                        {
                            Ok(ValueType::Float)
                        } else {
                            Err(ValidationError::TypeMismatch {
                                expected: "numeric".to_string(),
                                found: format!("{left_typ:?} and {right_typ:?}"),
                            })
                        }
                    }
                }
            }
            Expression::Unary(op, expr) => {
                let typ = self.validate_expression(expr)?;

                match op {
                    UnaryOp::Neg => {
                        if typ == ValueType::Int || typ == ValueType::Float {
                            Ok(typ)
                        } else {
                            Err(ValidationError::TypeMismatch {
                                expected: "numeric".to_string(),
                                found: format!("{typ:?}"),
                            })
                        }
                    }
                    UnaryOp::Not => {
                        if typ == ValueType::Bool {
                            Ok(ValueType::Bool)
                        } else {
                            Err(ValidationError::TypeMismatch {
                                expected: "bool".to_string(),
                                found: format!("{typ:?}"),
                            })
                        }
                    }
                    UnaryOp::BitNot => {
                        if typ == ValueType::Int {
                            Ok(ValueType::Int)
                        } else {
                            Err(ValidationError::TypeMismatch {
                                expected: "int".to_string(),
                                found: format!("{typ:?}"),
                            })
                        }
                    }
                    UnaryOp::Sin
                    | UnaryOp::Cos
                    | UnaryOp::Tan
                    | UnaryOp::Asin
                    | UnaryOp::Acos
                    | UnaryOp::Atan
                    | UnaryOp::Exp
                    | UnaryOp::Ln
                    | UnaryOp::Sqrt => {
                        if typ == ValueType::Int || typ == ValueType::Float {
                            Ok(ValueType::Float)
                        } else {
                            Err(ValidationError::TypeMismatch {
                                expected: "numeric".to_string(),
                                found: format!("{typ:?}"),
                            })
                        }
                    }
                }
            }
            Expression::Function(name, args) => {
                // Validate arguments
                for arg in args {
                    self.validate_expression(arg)?;
                }

                // For now, assume functions return float
                Ok(ValueType::Float)
            }
            Expression::Index(name, index) => {
                // Validate index
                let idx_typ = self.validate_expression(index)?;
                if idx_typ != ValueType::Int {
                    return Err(ValidationError::TypeMismatch {
                        expected: "int".to_string(),
                        found: format!("{idx_typ:?}"),
                    });
                }

                // For now, assume indexing returns same type
                match self.lookup_symbol(name) {
                    Some(Symbol::Variable { typ }) => Ok(typ.clone()),
                    _ => Err(ValidationError::UndefinedVariable(name.clone())),
                }
            }
        }
    }

    /// Validate a condition
    fn validate_condition(&self, cond: &Condition) -> Result<(), ValidationError> {
        let left_typ = self.validate_expression(&cond.left)?;
        let right_typ = self.validate_expression(&cond.right)?;

        // For comparisons, types should be compatible
        if !self.types_compatible(&left_typ, &right_typ) {
            return Err(ValidationError::TypeMismatch {
                expected: format!("{left_typ:?}"),
                found: format!("{right_typ:?}"),
            });
        }

        Ok(())
    }

    /// Check if two types are compatible
    fn types_compatible(&self, typ1: &ValueType, typ2: &ValueType) -> bool {
        match (typ1, typ2) {
            (ValueType::Int, ValueType::Float) | (ValueType::Float, ValueType::Int) => true,
            (ValueType::Angle, ValueType::Float) | (ValueType::Float, ValueType::Angle) => true,
            (ValueType::Duration, ValueType::Float) | (ValueType::Float, ValueType::Duration) => {
                true
            }
            (t1, t2) => t1 == t2,
        }
    }

    /// Push a new scope
    fn push_scope(&mut self) {
        self.scope_stack.push(HashMap::new());
    }

    /// Pop the current scope
    fn pop_scope(&mut self) {
        self.scope_stack.pop();
    }

    /// Add symbol to current scope
    fn add_to_current_scope(&mut self, name: String, symbol: Symbol) {
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.insert(name, symbol);
        } else {
            self.symbols.insert(name, symbol);
        }
    }

    /// Look up a symbol in all scopes
    fn lookup_symbol(&self, name: &str) -> Option<&Symbol> {
        // Check scopes from innermost to outermost
        for scope in self.scope_stack.iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Some(symbol);
            }
        }

        // Check global symbols
        self.symbols.get(name)
    }
}

/// Validate a QASM 3.0 program
pub fn validate_qasm3(program: &QasmProgram) -> Result<(), ValidationError> {
    let mut validator = QasmValidator::new();
    validator.validate(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qasm::parser::parse_qasm3;

    #[test]
    fn test_validate_simple_circuit() {
        let input = r"
OPENQASM 3.0;

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
measure q -> c;
";

        let program = parse_qasm3(input).unwrap();
        let result = validate_qasm3(&program);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_undefined_register() {
        let input = r"
OPENQASM 3.0;

qubit[2] q;

h q[0];
cx q[0], r[1];  // r is undefined
";

        let program = parse_qasm3(input).unwrap();
        let result = validate_qasm3(&program);
        assert!(matches!(result, Err(ValidationError::UndefinedRegister(_))));
    }

    #[test]
    fn test_validate_index_out_of_bounds() {
        let input = r"
OPENQASM 3.0;

qubit[2] q;

h q[5];  // Index 5 is out of bounds
";

        let program = parse_qasm3(input).unwrap();
        let result = validate_qasm3(&program);
        assert!(matches!(
            result,
            Err(ValidationError::IndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_validate_gate_parameters() {
        let input = r"
OPENQASM 3.0;

qubit q;

rx(pi/2) q;  // Correct
rx q;        // Missing parameter
";

        let program = parse_qasm3(input).unwrap();
        let result = validate_qasm3(&program);
        assert!(matches!(
            result,
            Err(ValidationError::ParameterCountMismatch { .. })
        ));
    }
}
