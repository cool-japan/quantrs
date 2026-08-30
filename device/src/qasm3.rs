//! OpenQASM 3.0 support for QuantRS2.
//!
//! This module provides QASM 3.0 circuit representation and conversion,
//! compatible with IBM Quantum and other modern quantum systems.
//!
//! ## Features
//!
//! - Full OpenQASM 3.0 syntax support
//! - Classical control flow (if/else, while, for)
//! - Dynamic circuits with mid-circuit measurements
//! - Gate modifiers (ctrl, inv, pow)
//! - Type system (qubit, bit, int, float, angle)
//! - Subroutine definitions
//!
//! ## Example
//!
//! ```rust,ignore
//! use quantrs2_device::qasm3::{Qasm3Builder, Qasm3Circuit};
//!
//! let mut builder = Qasm3Builder::new(5);
//! builder.gate("h", &[0])?;
//! builder.gate("cx", &[0, 1])?;
//! builder.measure(0, "c", 0)?;
//! builder.if_statement("c[0] == 1", |b| {
//!     b.gate("x", &[1])
//! })?;
//!
//! let qasm3 = builder.build()?;
//! println!("{}", qasm3.to_string());
//! ```

use std::collections::HashMap;
use std::fmt;

use crate::{DeviceError, DeviceResult};

/// OpenQASM 3.0 version
pub const QASM3_VERSION: &str = "3.0";

/// QASM 3.0 data types
#[derive(Debug, Clone, PartialEq)]
pub enum Qasm3Type {
    /// Quantum bit
    Qubit,
    /// Array of quantum bits
    QubitArray(usize),
    /// Classical bit
    Bit,
    /// Array of classical bits
    BitArray(usize),
    /// Integer type
    Int(Option<usize>),
    /// Unsigned integer
    Uint(Option<usize>),
    /// Floating point
    Float(Option<usize>),
    /// Angle type
    Angle(Option<usize>),
    /// Boolean
    Bool,
    /// Duration type
    Duration,
    /// Stretch (timing constraint)
    Stretch,
    /// Complex number
    Complex(Option<usize>),
}

impl fmt::Display for Qasm3Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qubit => write!(f, "qubit"),
            Self::QubitArray(n) => write!(f, "qubit[{}]", n),
            Self::Bit => write!(f, "bit"),
            Self::BitArray(n) => write!(f, "bit[{}]", n),
            Self::Int(Some(n)) => write!(f, "int[{}]", n),
            Self::Int(None) => write!(f, "int"),
            Self::Uint(Some(n)) => write!(f, "uint[{}]", n),
            Self::Uint(None) => write!(f, "uint"),
            Self::Float(Some(n)) => write!(f, "float[{}]", n),
            Self::Float(None) => write!(f, "float"),
            Self::Angle(Some(n)) => write!(f, "angle[{}]", n),
            Self::Angle(None) => write!(f, "angle"),
            Self::Bool => write!(f, "bool"),
            Self::Duration => write!(f, "duration"),
            Self::Stretch => write!(f, "stretch"),
            Self::Complex(Some(n)) => write!(f, "complex[float[{}]]", n),
            Self::Complex(None) => write!(f, "complex[float]"),
        }
    }
}

/// QASM 3.0 gate modifier
#[derive(Debug, Clone)]
pub enum GateModifier {
    /// Control modifier: ctrl @ gate
    Ctrl(usize),
    /// Negated control: negctrl @ gate
    NegCtrl(usize),
    /// Inverse modifier: inv @ gate
    Inv,
    /// Power modifier: pow(n) @ gate
    Pow(f64),
}

impl fmt::Display for GateModifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ctrl(n) if *n == 1 => write!(f, "ctrl @"),
            Self::Ctrl(n) => write!(f, "ctrl({}) @", n),
            Self::NegCtrl(n) if *n == 1 => write!(f, "negctrl @"),
            Self::NegCtrl(n) => write!(f, "negctrl({}) @", n),
            Self::Inv => write!(f, "inv @"),
            Self::Pow(n) => write!(f, "pow({}) @", n),
        }
    }
}

/// QASM 3.0 statement
#[derive(Debug, Clone)]
pub enum Qasm3Statement {
    /// Include directive
    Include(String),
    /// Variable declaration
    Declaration {
        var_type: Qasm3Type,
        name: String,
        init_value: Option<String>,
    },
    /// Gate definition
    GateDef {
        name: String,
        params: Vec<String>,
        qubits: Vec<String>,
        body: Vec<Qasm3Statement>,
    },
    /// Gate application
    Gate {
        name: String,
        modifiers: Vec<GateModifier>,
        params: Vec<String>,
        qubits: Vec<String>,
    },
    /// Measurement
    Measure { qubit: String, classical: String },
    /// Reset
    Reset(String),
    /// Barrier
    Barrier(Vec<String>),
    /// If statement
    If {
        condition: String,
        then_body: Vec<Qasm3Statement>,
        else_body: Option<Vec<Qasm3Statement>>,
    },
    /// Switch statement (QASM 3.0 dynamic circuits)
    Switch {
        expression: String,
        cases: Vec<(Vec<i64>, Vec<Qasm3Statement>)>,
        default_case: Option<Vec<Qasm3Statement>>,
    },
    /// While loop
    While {
        condition: String,
        body: Vec<Qasm3Statement>,
    },
    /// For loop
    For {
        var_name: String,
        range: String,
        body: Vec<Qasm3Statement>,
    },
    /// Classical assignment
    Assignment { target: String, value: String },
    /// Delay
    Delay {
        duration: String,
        qubits: Vec<String>,
    },
    /// Box (timing block)
    Box { body: Vec<Qasm3Statement> },
    /// Subroutine definition
    DefCal {
        name: String,
        params: Vec<String>,
        qubits: Vec<String>,
        body: String,
    },
    /// Comment
    Comment(String),
    /// Pragma
    Pragma(String),
}

impl fmt::Display for Qasm3Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Include(file) => writeln!(f, "include \"{}\";", file),
            Self::Declaration {
                var_type,
                name,
                init_value,
            } => {
                if let Some(val) = init_value {
                    writeln!(f, "{} {} = {};", var_type, name, val)
                } else {
                    writeln!(f, "{} {};", var_type, name)
                }
            }
            Self::GateDef {
                name,
                params,
                qubits,
                body,
            } => {
                let params_str = if params.is_empty() {
                    String::new()
                } else {
                    format!("({})", params.join(", "))
                };
                writeln!(f, "gate {}{} {} {{", name, params_str, qubits.join(", "))?;
                for stmt in body {
                    write!(f, "  {}", stmt)?;
                }
                writeln!(f, "}}")
            }
            Self::Gate {
                name,
                modifiers,
                params,
                qubits,
            } => {
                let mod_str = if modifiers.is_empty() {
                    String::new()
                } else {
                    modifiers
                        .iter()
                        .map(|m| m.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                        + " "
                };
                let params_str = if params.is_empty() {
                    String::new()
                } else {
                    format!("({})", params.join(", "))
                };
                writeln!(
                    f,
                    "{}{}{} {};",
                    mod_str,
                    name,
                    params_str,
                    qubits.join(", ")
                )
            }
            Self::Measure { qubit, classical } => {
                writeln!(f, "{} = measure {};", classical, qubit)
            }
            Self::Reset(qubit) => writeln!(f, "reset {};", qubit),
            Self::Barrier(qubits) => writeln!(f, "barrier {};", qubits.join(", ")),
            Self::If {
                condition,
                then_body,
                else_body,
            } => {
                writeln!(f, "if ({}) {{", condition)?;
                for stmt in then_body {
                    write!(f, "  {}", stmt)?;
                }
                if let Some(else_stmts) = else_body {
                    writeln!(f, "}} else {{")?;
                    for stmt in else_stmts {
                        write!(f, "  {}", stmt)?;
                    }
                }
                writeln!(f, "}}")
            }
            Self::Switch {
                expression,
                cases,
                default_case,
            } => {
                writeln!(f, "switch ({}) {{", expression)?;
                for (values, body) in cases {
                    let values_str = values
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    writeln!(f, "  case {}: {{", values_str)?;
                    for stmt in body {
                        write!(f, "    {}", stmt)?;
                    }
                    writeln!(f, "  }}")?;
                }
                if let Some(default_body) = default_case {
                    writeln!(f, "  default: {{")?;
                    for stmt in default_body {
                        write!(f, "    {}", stmt)?;
                    }
                    writeln!(f, "  }}")?;
                }
                writeln!(f, "}}")
            }
            Self::While { condition, body } => {
                writeln!(f, "while ({}) {{", condition)?;
                for stmt in body {
                    write!(f, "  {}", stmt)?;
                }
                writeln!(f, "}}")
            }
            Self::For {
                var_name,
                range,
                body,
            } => {
                writeln!(f, "for {} in {} {{", var_name, range)?;
                for stmt in body {
                    write!(f, "  {}", stmt)?;
                }
                writeln!(f, "}}")
            }
            Self::Assignment { target, value } => {
                writeln!(f, "{} = {};", target, value)
            }
            Self::Delay { duration, qubits } => {
                writeln!(f, "delay[{}] {};", duration, qubits.join(", "))
            }
            Self::Box { body } => {
                writeln!(f, "box {{")?;
                for stmt in body {
                    write!(f, "  {}", stmt)?;
                }
                writeln!(f, "}}")
            }
            Self::DefCal {
                name,
                params,
                qubits,
                body,
            } => {
                let params_str = if params.is_empty() {
                    String::new()
                } else {
                    format!("({})", params.join(", "))
                };
                writeln!(f, "defcal {}{} {} {{", name, params_str, qubits.join(", "))?;
                writeln!(f, "  {}", body)?;
                writeln!(f, "}}")
            }
            Self::Comment(text) => writeln!(f, "// {}", text),
            Self::Pragma(text) => writeln!(f, "#pragma {}", text),
        }
    }
}

/// QASM 3.0 circuit representation
#[derive(Debug, Clone)]
pub struct Qasm3Circuit {
    /// Version (always "3.0")
    pub version: String,
    /// Statements in the circuit
    pub statements: Vec<Qasm3Statement>,
    /// Input parameters
    pub inputs: HashMap<String, Qasm3Type>,
    /// Output parameters
    pub outputs: HashMap<String, Qasm3Type>,
}

impl Qasm3Circuit {
    /// Create a new empty QASM 3.0 circuit
    pub fn new() -> Self {
        Self {
            version: QASM3_VERSION.to_string(),
            statements: Vec::new(),
            inputs: HashMap::new(),
            outputs: HashMap::new(),
        }
    }

    /// Add an input parameter
    pub fn add_input(&mut self, name: &str, var_type: Qasm3Type) {
        self.inputs.insert(name.to_string(), var_type);
    }

    /// Add an output parameter
    pub fn add_output(&mut self, name: &str, var_type: Qasm3Type) {
        self.outputs.insert(name.to_string(), var_type);
    }

    /// Add a statement
    pub fn add_statement(&mut self, stmt: Qasm3Statement) {
        self.statements.push(stmt);
    }

    /// Get the number of qubits used
    pub fn num_qubits(&self) -> usize {
        let mut max_qubit = 0;
        for stmt in &self.statements {
            if let Qasm3Statement::Declaration {
                var_type: Qasm3Type::QubitArray(n),
                ..
            } = stmt
            {
                max_qubit = max_qubit.max(*n);
            }
        }
        max_qubit
    }

    /// Convert to QASM 3.0 string
    pub fn to_qasm3(&self) -> String {
        self.to_string()
    }

    /// Convert to QASM 2.0 string (for backward compatibility)
    pub fn to_qasm2(&self) -> DeviceResult<String> {
        let mut qasm2 = String::from("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\n");

        // Find qubit and bit declarations
        let mut num_qubits = 0;
        let mut num_bits = 0;

        for stmt in &self.statements {
            match stmt {
                Qasm3Statement::Declaration {
                    var_type: Qasm3Type::QubitArray(n),
                    ..
                } => {
                    num_qubits = num_qubits.max(*n);
                }
                Qasm3Statement::Declaration {
                    var_type: Qasm3Type::BitArray(n),
                    ..
                } => {
                    num_bits = num_bits.max(*n);
                }
                _ => {}
            }
        }

        if num_qubits > 0 {
            qasm2.push_str(&format!("qreg q[{}];\n", num_qubits));
        }
        if num_bits > 0 {
            qasm2.push_str(&format!("creg c[{}];\n", num_bits));
        }
        qasm2.push('\n');

        // Convert statements
        for stmt in &self.statements {
            match stmt {
                Qasm3Statement::Gate {
                    name,
                    modifiers,
                    params,
                    qubits,
                } => {
                    if !modifiers.is_empty() {
                        return Err(DeviceError::QasmError(
                            "Gate modifiers not supported in QASM 2.0".to_string(),
                        ));
                    }
                    let params_str = if params.is_empty() {
                        String::new()
                    } else {
                        format!("({})", params.join(", "))
                    };
                    qasm2.push_str(&format!("{}{} {};\n", name, params_str, qubits.join(", ")));
                }
                Qasm3Statement::Measure { qubit, classical } => {
                    qasm2.push_str(&format!("measure {} -> {};\n", qubit, classical));
                }
                Qasm3Statement::Reset(qubit) => {
                    qasm2.push_str(&format!("reset {};\n", qubit));
                }
                Qasm3Statement::Barrier(qubits) => {
                    qasm2.push_str(&format!("barrier {};\n", qubits.join(", ")));
                }
                Qasm3Statement::If {
                    condition,
                    then_body,
                    else_body,
                } => {
                    if else_body.is_some() {
                        return Err(DeviceError::QasmError(
                            "If-else not supported in QASM 2.0".to_string(),
                        ));
                    }
                    // QASM 2.0 only supports simple if statements
                    qasm2.push_str(&format!("if ({}) ", condition));
                    if let Some(stmt) = then_body.first() {
                        if let Qasm3Statement::Gate {
                            name,
                            params,
                            qubits,
                            ..
                        } = stmt
                        {
                            let params_str = if params.is_empty() {
                                String::new()
                            } else {
                                format!("({})", params.join(", "))
                            };
                            qasm2.push_str(&format!(
                                "{}{} {};\n",
                                name,
                                params_str,
                                qubits.join(", ")
                            ));
                        }
                    }
                }
                Qasm3Statement::While { .. } | Qasm3Statement::For { .. } => {
                    return Err(DeviceError::QasmError(
                        "Loops not supported in QASM 2.0".to_string(),
                    ));
                }
                Qasm3Statement::Declaration { .. }
                | Qasm3Statement::Include(_)
                | Qasm3Statement::Comment(_) => {
                    // Skip declarations (handled above) and comments
                }
                _ => {}
            }
        }

        Ok(qasm2)
    }
}

impl Default for Qasm3Circuit {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Qasm3Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "OPENQASM {};", self.version)?;
        writeln!(f)?;

        // Include standard library
        writeln!(f, "include \"stdgates.inc\";")?;
        writeln!(f)?;

        // Input declarations
        for (name, var_type) in &self.inputs {
            writeln!(f, "input {} {};", var_type, name)?;
        }
        if !self.inputs.is_empty() {
            writeln!(f)?;
        }

        // Output declarations
        for (name, var_type) in &self.outputs {
            writeln!(f, "output {} {};", var_type, name)?;
        }
        if !self.outputs.is_empty() {
            writeln!(f)?;
        }

        // Statements
        for stmt in &self.statements {
            write!(f, "{}", stmt)?;
        }

        Ok(())
    }
}

/// Builder for QASM 3.0 circuits
pub struct Qasm3Builder {
    /// The circuit being built
    pub circuit: Qasm3Circuit,
    /// Number of qubits in the circuit
    pub num_qubits: usize,
    /// Number of classical bits in the circuit
    pub num_bits: usize,
}

impl Qasm3Builder {
    /// Create a new builder with specified number of qubits
    pub fn new(num_qubits: usize) -> Self {
        let mut builder = Self {
            circuit: Qasm3Circuit::new(),
            num_qubits,
            num_bits: num_qubits, // Default same number of classical bits
        };

        // Add qubit and bit declarations
        builder.circuit.add_statement(Qasm3Statement::Declaration {
            var_type: Qasm3Type::QubitArray(num_qubits),
            name: "q".to_string(),
            init_value: None,
        });
        builder.circuit.add_statement(Qasm3Statement::Declaration {
            var_type: Qasm3Type::BitArray(num_qubits),
            name: "c".to_string(),
            init_value: None,
        });

        builder
    }

    /// Set number of classical bits
    pub fn with_bits(mut self, num_bits: usize) -> Self {
        self.num_bits = num_bits;
        // Update the bit declaration
        for stmt in &mut self.circuit.statements {
            if let Qasm3Statement::Declaration {
                var_type: Qasm3Type::BitArray(n),
                name,
                ..
            } = stmt
            {
                if name == "c" {
                    *n = num_bits;
                }
            }
        }
        self
    }

    /// Add a gate operation
    pub fn gate(&mut self, name: &str, qubits: &[usize]) -> DeviceResult<&mut Self> {
        self.gate_with_params(name, &[], qubits)
    }

    /// Add a gate with parameters
    pub fn gate_with_params(
        &mut self,
        name: &str,
        params: &[f64],
        qubits: &[usize],
    ) -> DeviceResult<&mut Self> {
        for &q in qubits {
            if q >= self.num_qubits {
                return Err(DeviceError::InvalidInput(format!(
                    "Qubit {} out of range (max {})",
                    q,
                    self.num_qubits - 1
                )));
            }
        }

        let qubit_strs: Vec<String> = qubits.iter().map(|q| format!("q[{}]", q)).collect();
        let param_strs: Vec<String> = params.iter().map(|p| format!("{}", p)).collect();

        self.circuit.add_statement(Qasm3Statement::Gate {
            name: name.to_string(),
            modifiers: Vec::new(),
            params: param_strs,
            qubits: qubit_strs,
        });

        Ok(self)
    }

    /// Add a controlled gate
    pub fn ctrl_gate(
        &mut self,
        name: &str,
        control: usize,
        target: usize,
        params: &[f64],
    ) -> DeviceResult<&mut Self> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(DeviceError::InvalidInput("Qubit out of range".to_string()));
        }

        let param_strs: Vec<String> = params.iter().map(|p| format!("{}", p)).collect();

        self.circuit.add_statement(Qasm3Statement::Gate {
            name: name.to_string(),
            modifiers: vec![GateModifier::Ctrl(1)],
            params: param_strs,
            qubits: vec![format!("q[{}]", control), format!("q[{}]", target)],
        });

        Ok(self)
    }

    /// Add an inverse gate
    pub fn inv_gate(&mut self, name: &str, qubits: &[usize]) -> DeviceResult<&mut Self> {
        for &q in qubits {
            if q >= self.num_qubits {
                return Err(DeviceError::InvalidInput("Qubit out of range".to_string()));
            }
        }

        let qubit_strs: Vec<String> = qubits.iter().map(|q| format!("q[{}]", q)).collect();

        self.circuit.add_statement(Qasm3Statement::Gate {
            name: name.to_string(),
            modifiers: vec![GateModifier::Inv],
            params: Vec::new(),
            qubits: qubit_strs,
        });

        Ok(self)
    }

    /// Add measurement
    pub fn measure(&mut self, qubit: usize, bit: usize) -> DeviceResult<&mut Self> {
        if qubit >= self.num_qubits {
            return Err(DeviceError::InvalidInput("Qubit out of range".to_string()));
        }
        if bit >= self.num_bits {
            return Err(DeviceError::InvalidInput("Bit out of range".to_string()));
        }

        self.circuit.add_statement(Qasm3Statement::Measure {
            qubit: format!("q[{}]", qubit),
            classical: format!("c[{}]", bit),
        });

        Ok(self)
    }

    /// Measure all qubits
    pub fn measure_all(&mut self) -> DeviceResult<&mut Self> {
        for i in 0..self.num_qubits.min(self.num_bits) {
            self.measure(i, i)?;
        }
        Ok(self)
    }

    /// Add reset operation
    pub fn reset(&mut self, qubit: usize) -> DeviceResult<&mut Self> {
        if qubit >= self.num_qubits {
            return Err(DeviceError::InvalidInput("Qubit out of range".to_string()));
        }

        self.circuit
            .add_statement(Qasm3Statement::Reset(format!("q[{}]", qubit)));

        Ok(self)
    }

    /// Add barrier
    pub fn barrier(&mut self, qubits: &[usize]) -> DeviceResult<&mut Self> {
        for &q in qubits {
            if q >= self.num_qubits {
                return Err(DeviceError::InvalidInput("Qubit out of range".to_string()));
            }
        }

        let qubit_strs: Vec<String> = qubits.iter().map(|q| format!("q[{}]", q)).collect();
        self.circuit
            .add_statement(Qasm3Statement::Barrier(qubit_strs));

        Ok(self)
    }

    /// Add if statement (dynamic circuit)
    pub fn if_statement<F>(&mut self, condition: &str, then_block: F) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut Qasm3Builder) -> DeviceResult<()>,
    {
        let mut inner_builder = Qasm3Builder {
            circuit: Qasm3Circuit::new(),
            num_qubits: self.num_qubits,
            num_bits: self.num_bits,
        };
        inner_builder.circuit.statements.clear(); // Clear default declarations

        then_block(&mut inner_builder)?;

        self.circuit.add_statement(Qasm3Statement::If {
            condition: condition.to_string(),
            then_body: inner_builder.circuit.statements,
            else_body: None,
        });

        Ok(self)
    }

    /// Add if-else statement (dynamic circuit)
    pub fn if_else_statement<F, G>(
        &mut self,
        condition: &str,
        then_block: F,
        else_block: G,
    ) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut Qasm3Builder) -> DeviceResult<()>,
        G: FnOnce(&mut Qasm3Builder) -> DeviceResult<()>,
    {
        let mut then_builder = Qasm3Builder {
            circuit: Qasm3Circuit::new(),
            num_qubits: self.num_qubits,
            num_bits: self.num_bits,
        };
        then_builder.circuit.statements.clear();
        then_block(&mut then_builder)?;

        let mut else_builder = Qasm3Builder {
            circuit: Qasm3Circuit::new(),
            num_qubits: self.num_qubits,
            num_bits: self.num_bits,
        };
        else_builder.circuit.statements.clear();
        else_block(&mut else_builder)?;

        self.circuit.add_statement(Qasm3Statement::If {
            condition: condition.to_string(),
            then_body: then_builder.circuit.statements,
            else_body: Some(else_builder.circuit.statements),
        });

        Ok(self)
    }

    /// Add while loop
    pub fn while_loop<F>(&mut self, condition: &str, body: F) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut Qasm3Builder) -> DeviceResult<()>,
    {
        let mut inner_builder = Qasm3Builder {
            circuit: Qasm3Circuit::new(),
            num_qubits: self.num_qubits,
            num_bits: self.num_bits,
        };
        inner_builder.circuit.statements.clear();

        body(&mut inner_builder)?;

        self.circuit.add_statement(Qasm3Statement::While {
            condition: condition.to_string(),
            body: inner_builder.circuit.statements,
        });

        Ok(self)
    }

    /// Add for loop
    pub fn for_loop<F>(&mut self, var_name: &str, range: &str, body: F) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut Qasm3Builder) -> DeviceResult<()>,
    {
        let mut inner_builder = Qasm3Builder {
            circuit: Qasm3Circuit::new(),
            num_qubits: self.num_qubits,
            num_bits: self.num_bits,
        };
        inner_builder.circuit.statements.clear();

        body(&mut inner_builder)?;

        self.circuit.add_statement(Qasm3Statement::For {
            var_name: var_name.to_string(),
            range: range.to_string(),
            body: inner_builder.circuit.statements,
        });

        Ok(self)
    }

    /// Add switch statement (QASM 3.0 dynamic circuit feature)
    ///
    /// # Arguments
    /// * `expression` - The expression to switch on (e.g., "c")
    /// * `case_builder` - A closure that builds the switch cases
    ///
    /// # Example
    /// ```ignore
    /// builder.switch_statement("c", |sw| {
    ///     sw.case(&[0], |b| b.gate("x", &[0]))?;
    ///     sw.case(&[1], |b| b.gate("y", &[0]))?;
    ///     sw.default(|b| b.gate("z", &[0]))?;
    ///     Ok(())
    /// })?;
    /// ```
    pub fn switch_statement<F>(
        &mut self,
        expression: &str,
        case_builder: F,
    ) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut SwitchBuilder) -> DeviceResult<()>,
    {
        let mut switch_builder = SwitchBuilder::new(self.num_qubits, self.num_bits);
        case_builder(&mut switch_builder)?;

        self.circuit.add_statement(Qasm3Statement::Switch {
            expression: expression.to_string(),
            cases: switch_builder.cases,
            default_case: switch_builder.default_case,
        });

        Ok(self)
    }

    /// Add classical assignment
    ///
    /// # Arguments
    /// * `target` - Target variable (e.g., `"c[0]"`)
    /// * `value` - Value expression (e.g., `"c[0] + 1"`, `"c[0] & c[1]"`)
    pub fn assign(&mut self, target: &str, value: &str) -> &mut Self {
        self.circuit.add_statement(Qasm3Statement::Assignment {
            target: target.to_string(),
            value: value.to_string(),
        });
        self
    }

    /// Add comment
    pub fn comment(&mut self, text: &str) -> &mut Self {
        self.circuit
            .add_statement(Qasm3Statement::Comment(text.to_string()));
        self
    }

    /// Build the QASM 3.0 circuit
    pub fn build(self) -> DeviceResult<Qasm3Circuit> {
        Ok(self.circuit)
    }
}

/// Builder for switch statement cases
pub struct SwitchBuilder {
    num_qubits: usize,
    num_bits: usize,
    cases: Vec<(Vec<i64>, Vec<Qasm3Statement>)>,
    default_case: Option<Vec<Qasm3Statement>>,
}

impl SwitchBuilder {
    fn new(num_qubits: usize, num_bits: usize) -> Self {
        Self {
            num_qubits,
            num_bits,
            cases: Vec::new(),
            default_case: None,
        }
    }

    /// Add a case to the switch statement
    ///
    /// # Arguments
    /// * `values` - The case values (can be multiple for fallthrough)
    /// * `body` - A closure that builds the case body
    pub fn case<F>(&mut self, values: &[i64], body: F) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut Qasm3Builder) -> DeviceResult<()>,
    {
        let mut inner_builder = Qasm3Builder {
            circuit: Qasm3Circuit::new(),
            num_qubits: self.num_qubits,
            num_bits: self.num_bits,
        };
        inner_builder.circuit.statements.clear();

        body(&mut inner_builder)?;

        self.cases
            .push((values.to_vec(), inner_builder.circuit.statements));
        Ok(self)
    }

    /// Add a default case to the switch statement
    pub fn default<F>(&mut self, body: F) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut Qasm3Builder) -> DeviceResult<()>,
    {
        let mut inner_builder = Qasm3Builder {
            circuit: Qasm3Circuit::new(),
            num_qubits: self.num_qubits,
            num_bits: self.num_bits,
        };
        inner_builder.circuit.statements.clear();

        body(&mut inner_builder)?;

        self.default_case = Some(inner_builder.circuit.statements);
        Ok(self)
    }
}

/// Convert a `QuantRS2` circuit to QASM 3.0.
///
/// Iterates the circuit gates and emits one QASM 3.0 statement per gate using
/// the standard OpenQASM 3.0 gate names. Parameterized rotations are emitted
/// with their angle; gates with no QASM 3.0 standard-library equivalent produce
/// an honest [`DeviceError::CircuitConversion`] naming the offending gate rather
/// than being silently dropped.
pub fn circuit_to_qasm3<const N: usize>(
    circuit: &quantrs2_circuit::prelude::Circuit<N>,
) -> DeviceResult<Qasm3Circuit> {
    use quantrs2_core::gate::{multi::*, single::*, GateOp};

    let mut builder = Qasm3Builder::new(N);

    for gate in circuit.gates() {
        let any = gate.as_any();
        let qubits: Vec<usize> = gate.qubits().iter().map(|q| q.id() as usize).collect();

        // Parameterized single-qubit rotations.
        if let Some(g) = any.downcast_ref::<RotationX>() {
            builder.gate_with_params("rx", &[g.theta], &qubits)?;
            continue;
        }
        if let Some(g) = any.downcast_ref::<RotationY>() {
            builder.gate_with_params("ry", &[g.theta], &qubits)?;
            continue;
        }
        if let Some(g) = any.downcast_ref::<RotationZ>() {
            builder.gate_with_params("rz", &[g.theta], &qubits)?;
            continue;
        }
        if let Some(g) = any.downcast_ref::<PGate>() {
            builder.gate_with_params("p", &[g.lambda], &qubits)?;
            continue;
        }
        if let Some(g) = any.downcast_ref::<UGate>() {
            builder.gate_with_params("u", &[g.theta, g.phi, g.lambda], &qubits)?;
            continue;
        }
        // Parameterized two-qubit rotations.
        if let Some(g) = any.downcast_ref::<CRX>() {
            builder.gate_with_params("crx", &[g.theta], &qubits)?;
            continue;
        }
        if let Some(g) = any.downcast_ref::<CRY>() {
            builder.gate_with_params("cry", &[g.theta], &qubits)?;
            continue;
        }
        if let Some(g) = any.downcast_ref::<CRZ>() {
            builder.gate_with_params("crz", &[g.theta], &qubits)?;
            continue;
        }
        if let Some(g) = any.downcast_ref::<RXX>() {
            builder.gate_with_params("rxx", &[g.theta], &qubits)?;
            continue;
        }
        if let Some(g) = any.downcast_ref::<RYY>() {
            builder.gate_with_params("ryy", &[g.theta], &qubits)?;
            continue;
        }
        if let Some(g) = any.downcast_ref::<RZZ>() {
            builder.gate_with_params("rzz", &[g.theta], &qubits)?;
            continue;
        }

        // Non-parameterized gates: map QuantRS2 names to QASM 3.0 standard names.
        let qasm_name = match gate.name() {
            "H" => "h",
            "X" => "x",
            "Y" => "y",
            "Z" => "z",
            "S" => "s",
            "S†" => "sdg",
            "T" => "t",
            "T†" => "tdg",
            "√X" => "sx",
            "√X†" => "sxdg",
            "I" => "id",
            "CNOT" => "cx",
            "CZ" => "cz",
            "CY" => "cy",
            "CH" => "ch",
            "SWAP" => "swap",
            "Toffoli" => "ccx",
            "Fredkin" => "cswap",
            other => {
                return Err(DeviceError::CircuitConversion(format!(
                    "Gate '{other}' has no OpenQASM 3.0 standard-library equivalent"
                )));
            }
        };
        builder.gate(qasm_name, &qubits)?;
    }

    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qasm3_type_display() {
        assert_eq!(format!("{}", Qasm3Type::Qubit), "qubit");
        assert_eq!(format!("{}", Qasm3Type::QubitArray(5)), "qubit[5]");
        assert_eq!(format!("{}", Qasm3Type::BitArray(3)), "bit[3]");
        assert_eq!(format!("{}", Qasm3Type::Int(Some(32))), "int[32]");
        assert_eq!(format!("{}", Qasm3Type::Float(None)), "float");
    }

    #[test]
    fn test_gate_modifier_display() {
        assert_eq!(format!("{}", GateModifier::Ctrl(1)), "ctrl @");
        assert_eq!(format!("{}", GateModifier::Ctrl(2)), "ctrl(2) @");
        assert_eq!(format!("{}", GateModifier::Inv), "inv @");
        assert_eq!(format!("{}", GateModifier::Pow(2.0)), "pow(2) @");
    }

    #[test]
    fn test_qasm3_builder_basic() {
        let mut builder = Qasm3Builder::new(3);
        builder.gate("h", &[0]).unwrap();
        builder.gate("cx", &[0, 1]).unwrap();
        builder.measure(0, 0).unwrap();

        let circuit = builder.build().unwrap();
        let qasm = circuit.to_string();

        assert!(qasm.contains("OPENQASM 3.0"));
        assert!(qasm.contains("qubit[3]"));
        assert!(qasm.contains("h q[0]"));
        assert!(qasm.contains("cx q[0], q[1]"));
    }

    #[test]
    fn test_qasm3_builder_params() {
        let mut builder = Qasm3Builder::new(2);
        builder
            .gate_with_params("rx", &[std::f64::consts::PI / 2.0], &[0])
            .unwrap();

        let circuit = builder.build().unwrap();
        let qasm = circuit.to_string();

        assert!(qasm.contains("rx"));
    }

    #[test]
    fn test_qasm3_if_statement() {
        let mut builder = Qasm3Builder::new(2);
        builder.gate("h", &[0]).unwrap();
        builder.measure(0, 0).unwrap();
        builder
            .if_statement("c[0] == 1", |b| {
                b.gate("x", &[1])?;
                Ok(())
            })
            .unwrap();

        let circuit = builder.build().unwrap();
        let qasm = circuit.to_string();

        assert!(qasm.contains("if (c[0] == 1)"));
        assert!(qasm.contains("x q[1]"));
    }

    #[test]
    fn test_qasm3_to_qasm2() {
        let mut builder = Qasm3Builder::new(2);
        builder.gate("h", &[0]).unwrap();
        builder.gate("cx", &[0, 1]).unwrap();
        builder.measure_all().unwrap();

        let circuit = builder.build().unwrap();
        let qasm2 = circuit.to_qasm2().unwrap();

        assert!(qasm2.contains("OPENQASM 2.0"));
        assert!(qasm2.contains("qreg q[2]"));
        assert!(qasm2.contains("creg c[2]"));
    }

    #[test]
    fn test_qasm3_ctrl_gate() {
        let mut builder = Qasm3Builder::new(3);
        builder.ctrl_gate("x", 0, 1, &[]).unwrap();

        let circuit = builder.build().unwrap();
        let qasm = circuit.to_string();

        assert!(qasm.contains("ctrl @"));
    }

    #[test]
    fn test_qasm3_inv_gate() {
        let mut builder = Qasm3Builder::new(2);
        builder.inv_gate("s", &[0]).unwrap();

        let circuit = builder.build().unwrap();
        let qasm = circuit.to_string();

        assert!(qasm.contains("inv @"));
    }

    #[test]
    fn test_qasm3_for_loop() {
        let mut builder = Qasm3Builder::new(4);
        builder
            .for_loop("i", "[0:3]", |b| {
                b.gate("h", &[0])?;
                Ok(())
            })
            .unwrap();

        let circuit = builder.build().unwrap();
        let qasm = circuit.to_string();

        assert!(qasm.contains("for i in [0:3]"));
    }

    #[test]
    fn test_circuit_to_qasm3_emits_real_statements() {
        use quantrs2_circuit::prelude::Circuit;

        // H, CNOT, RZ(pi/2): 3 gates → expect exactly 3 gate statements.
        let mut circuit = Circuit::<2>::new();
        circuit.h(0).expect("add H");
        circuit.cnot(0, 1).expect("add CNOT");
        circuit.rz(1, std::f64::consts::PI / 2.0).expect("add RZ");

        let qasm3 = circuit_to_qasm3(&circuit).expect("conversion should succeed");
        let text = qasm3.to_string();

        // The real gate statements must be present, not silently dropped.
        assert!(text.contains("h q[0]"), "missing H: {text}");
        assert!(text.contains("cx q[0], q[1]"), "missing CNOT: {text}");
        assert!(text.contains("rz"), "missing RZ: {text}");

        // Statement count must match the gate count (3 gates → 3 statements).
        let gate_statements = qasm3
            .statements
            .iter()
            .filter(|s| matches!(s, Qasm3Statement::Gate { .. }))
            .count();
        assert_eq!(
            gate_statements,
            circuit.gates().len(),
            "QASM3 statement count must equal circuit gate count"
        );
    }

    #[test]
    fn test_circuit_to_qasm3_unsupported_gate_is_honest_error() {
        use quantrs2_circuit::prelude::Circuit;
        use quantrs2_core::gate::multi::XXPlusYY;

        // XXPlusYY has no QASM 3.0 standard-library mapping here.
        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(XXPlusYY {
                qubit1: quantrs2_core::qubit::QubitId(0),
                qubit2: quantrs2_core::qubit::QubitId(1),
                theta: 0.5,
                beta: 0.25,
            })
            .expect("add XXPlusYY");

        let result = circuit_to_qasm3(&circuit);
        assert!(
            matches!(result, Err(DeviceError::CircuitConversion(_))),
            "expected honest CircuitConversion error, got {result:?}"
        );
    }
}
