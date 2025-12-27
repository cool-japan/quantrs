//! Classical control flow support for quantum circuits
//!
//! This module provides support for classical registers, measurements,
//! and conditional execution of quantum gates based on classical values.

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use std::collections::HashMap;
use std::fmt;

/// A classical register that can store measurement results
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClassicalRegister {
    /// Name of the register
    pub name: String,
    /// Number of bits in the register
    pub size: usize,
}

impl ClassicalRegister {
    /// Create a new classical register
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self {
            name: name.into(),
            size,
        }
    }
}

/// A classical bit reference within a register
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClassicalBit {
    /// The register containing this bit
    pub register: String,
    /// Index within the register
    pub index: usize,
}

/// Classical values that can be used in conditions
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassicalValue {
    /// A single bit value
    Bit(bool),
    /// A multi-bit integer value
    Integer(u64),
    /// A reference to a classical register
    Register(String),
}

/// Comparison operators for classical conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
    /// Less than
    Less,
    /// Less than or equal
    LessEqual,
    /// Greater than
    Greater,
    /// Greater than or equal
    GreaterEqual,
}

/// A condition that gates execution based on classical values
#[derive(Debug, Clone)]
pub struct ClassicalCondition {
    /// Left-hand side of the comparison
    pub lhs: ClassicalValue,
    /// Comparison operator
    pub op: ComparisonOp,
    /// Right-hand side of the comparison
    pub rhs: ClassicalValue,
}

impl ClassicalCondition {
    /// Create a new equality condition
    #[must_use]
    pub const fn equals(lhs: ClassicalValue, rhs: ClassicalValue) -> Self {
        Self {
            lhs,
            op: ComparisonOp::Equal,
            rhs,
        }
    }

    /// Check if a register equals a specific value
    #[must_use]
    pub fn register_equals(register: &str, value: u64) -> Self {
        Self {
            lhs: ClassicalValue::Register(register.to_string()),
            op: ComparisonOp::Equal,
            rhs: ClassicalValue::Integer(value),
        }
    }
}

/// A measurement operation that stores the result in a classical register
#[derive(Debug, Clone)]
pub struct MeasureOp {
    /// Qubit to measure
    pub qubit: QubitId,
    /// Classical bit to store the result
    pub cbit: ClassicalBit,
}

impl MeasureOp {
    /// Create a new measurement operation
    #[must_use]
    pub fn new(qubit: QubitId, register: &str, bit_index: usize) -> Self {
        Self {
            qubit,
            cbit: ClassicalBit {
                register: register.to_string(),
                index: bit_index,
            },
        }
    }
}

/// A gate that executes conditionally based on classical values
pub struct ConditionalGate {
    /// The condition to check
    pub condition: ClassicalCondition,
    /// The gate to execute if the condition is true
    pub gate: Box<dyn GateOp>,
}

impl fmt::Debug for ConditionalGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConditionalGate")
            .field("condition", &self.condition)
            .field("gate", &self.gate.name())
            .finish()
    }
}

/// Classical control flow operations
#[derive(Debug)]
pub enum ClassicalOp {
    /// Measure a qubit into a classical bit
    Measure(MeasureOp),
    /// Reset a classical register
    Reset(String),
    /// Conditional gate execution
    Conditional(ConditionalGate),
    /// Classical computation (e.g., XOR, AND)
    Compute {
        /// Output register
        output: String,
        /// Operation type
        op: String,
        /// Input registers
        inputs: Vec<String>,
    },
}

/// A circuit with classical control flow support
pub struct ClassicalCircuit<const N: usize> {
    /// Classical registers
    pub classical_registers: HashMap<String, ClassicalRegister>,
    /// Operations (both quantum and classical)
    pub operations: Vec<CircuitOp>,
}

/// Operations that can appear in a classical circuit
pub enum CircuitOp {
    /// A quantum gate operation
    Quantum(Box<dyn GateOp>),
    /// A classical operation
    Classical(ClassicalOp),
}

impl<const N: usize> Default for ClassicalCircuit<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> ClassicalCircuit<N> {
    /// Create a new circuit with classical control
    #[must_use]
    pub fn new() -> Self {
        Self {
            classical_registers: HashMap::new(),
            operations: Vec::new(),
        }
    }

    /// Add a classical register
    pub fn add_classical_register(&mut self, name: &str, size: usize) -> QuantRS2Result<()> {
        if self.classical_registers.contains_key(name) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Classical register '{name}' already exists"
            )));
        }

        self.classical_registers
            .insert(name.to_string(), ClassicalRegister::new(name, size));
        Ok(())
    }

    /// Add a quantum gate
    pub fn add_gate<G: GateOp + 'static>(&mut self, gate: G) -> QuantRS2Result<()> {
        // Validate qubits
        for qubit in gate.qubits() {
            if qubit.id() as usize >= N {
                return Err(QuantRS2Error::InvalidQubitId(qubit.id()));
            }
        }

        self.operations.push(CircuitOp::Quantum(Box::new(gate)));
        Ok(())
    }

    /// Add a measurement
    pub fn measure(&mut self, qubit: QubitId, register: &str, bit: usize) -> QuantRS2Result<()> {
        // Validate qubit
        if qubit.id() as usize >= N {
            return Err(QuantRS2Error::InvalidQubitId(qubit.id()));
        }

        // Validate classical register
        let creg = self.classical_registers.get(register).ok_or_else(|| {
            QuantRS2Error::InvalidInput(format!("Classical register '{register}' not found"))
        })?;

        if bit >= creg.size {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Bit index {} out of range for register '{}' (size: {})",
                bit, register, creg.size
            )));
        }

        self.operations
            .push(CircuitOp::Classical(ClassicalOp::Measure(MeasureOp::new(
                qubit, register, bit,
            ))));
        Ok(())
    }

    /// Add a conditional gate
    ///
    /// Adds a gate that will only execute if the classical condition is satisfied.
    ///
    /// # Arguments
    /// * `condition` - The classical condition that must be true for the gate to execute
    /// * `gate` - The quantum gate to execute conditionally
    ///
    /// # Errors
    /// Returns an error if:
    /// - The gate acts on invalid qubits (>= N)
    /// - The condition references non-existent classical registers
    ///
    /// # Examples
    /// ```ignore
    /// circuit.add_classical_register("measurement", 1)?;
    /// circuit.measure(QubitId(0), "measurement", 0)?;
    /// let condition = ClassicalCondition::register_equals("measurement", 1);
    /// circuit.add_conditional(condition, PauliX { target: QubitId(1) })?;
    /// ```
    pub fn add_conditional<G: GateOp + 'static>(
        &mut self,
        condition: ClassicalCondition,
        gate: G,
    ) -> QuantRS2Result<()> {
        // Validate gate qubits
        for qubit in gate.qubits() {
            if qubit.id() as usize >= N {
                return Err(QuantRS2Error::InvalidQubitId(qubit.id()));
            }
        }

        // Validate condition references valid registers
        self.validate_classical_value(&condition.lhs)?;
        self.validate_classical_value(&condition.rhs)?;

        self.operations
            .push(CircuitOp::Classical(ClassicalOp::Conditional(
                ConditionalGate {
                    condition,
                    gate: Box::new(gate),
                },
            )));
        Ok(())
    }

    /// Validate that a classical value references valid registers
    fn validate_classical_value(&self, value: &ClassicalValue) -> QuantRS2Result<()> {
        match value {
            ClassicalValue::Register(register_name) => {
                if !self.classical_registers.contains_key(register_name) {
                    return Err(QuantRS2Error::InvalidInput(format!(
                        "Classical register '{register_name}' not found. Available registers: {:?}",
                        self.classical_registers.keys().collect::<Vec<_>>()
                    )));
                }
            }
            ClassicalValue::Bit(_) | ClassicalValue::Integer(_) => {
                // Literal values are always valid
            }
        }
        Ok(())
    }

    /// Reset a classical register to zero
    pub fn reset_classical(&mut self, register: &str) -> QuantRS2Result<()> {
        if !self.classical_registers.contains_key(register) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Classical register '{register}' not found"
            )));
        }

        self.operations
            .push(CircuitOp::Classical(ClassicalOp::Reset(
                register.to_string(),
            )));
        Ok(())
    }

    /// Get the number of operations
    #[must_use]
    pub fn num_operations(&self) -> usize {
        self.operations.len()
    }
}

/// Builder pattern for classical circuits
pub struct ClassicalCircuitBuilder<const N: usize> {
    circuit: ClassicalCircuit<N>,
}

impl<const N: usize> Default for ClassicalCircuitBuilder<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> ClassicalCircuitBuilder<N> {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            circuit: ClassicalCircuit::new(),
        }
    }

    /// Add a classical register
    pub fn classical_register(mut self, name: &str, size: usize) -> QuantRS2Result<Self> {
        self.circuit.add_classical_register(name, size)?;
        Ok(self)
    }

    /// Add a quantum gate
    pub fn gate<G: GateOp + 'static>(mut self, gate: G) -> QuantRS2Result<Self> {
        self.circuit.add_gate(gate)?;
        Ok(self)
    }

    /// Add a measurement
    pub fn measure(mut self, qubit: QubitId, register: &str, bit: usize) -> QuantRS2Result<Self> {
        self.circuit.measure(qubit, register, bit)?;
        Ok(self)
    }

    /// Add a conditional gate
    pub fn conditional<G: GateOp + 'static>(
        mut self,
        condition: ClassicalCondition,
        gate: G,
    ) -> QuantRS2Result<Self> {
        self.circuit.add_conditional(condition, gate)?;
        Ok(self)
    }

    /// Build the circuit
    #[must_use]
    pub fn build(self) -> ClassicalCircuit<N> {
        self.circuit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::single::PauliX;

    #[test]
    fn test_classical_register() {
        let reg = ClassicalRegister::new("c", 3);
        assert_eq!(reg.name, "c");
        assert_eq!(reg.size, 3);
    }

    #[test]
    fn test_classical_condition() {
        let cond = ClassicalCondition::register_equals("c", 1);
        assert_eq!(cond.op, ComparisonOp::Equal);

        match &cond.lhs {
            ClassicalValue::Register(name) => assert_eq!(name, "c"),
            _ => panic!("Expected Register variant"),
        }

        match &cond.rhs {
            ClassicalValue::Integer(val) => assert_eq!(*val, 1),
            _ => panic!("Expected Integer variant"),
        }
    }

    #[test]
    fn test_classical_circuit_builder() {
        let circuit = ClassicalCircuitBuilder::<2>::new()
            .classical_register("c", 2)
            .expect("Failed to add classical register")
            .gate(PauliX { target: QubitId(0) })
            .expect("Failed to add PauliX gate")
            .measure(QubitId(0), "c", 0)
            .expect("Failed to add measurement")
            .conditional(
                ClassicalCondition::register_equals("c", 1),
                PauliX { target: QubitId(1) },
            )
            .expect("Failed to add conditional gate")
            .build();

        assert_eq!(circuit.classical_registers.len(), 1);
        assert_eq!(circuit.num_operations(), 3);
    }

    #[test]
    fn test_conditional_validation_invalid_register() {
        // Test that using a non-existent register in a condition produces an error
        let mut circuit = ClassicalCircuit::<2>::new();
        circuit
            .add_classical_register("measurement", 1)
            .expect("Failed to add classical register");

        // Try to add a conditional gate with a condition referencing a non-existent register
        let condition = ClassicalCondition::register_equals("nonexistent", 1);
        let result = circuit.add_conditional(condition, PauliX { target: QubitId(0) });

        assert!(result.is_err());
        match result {
            Err(QuantRS2Error::InvalidInput(msg)) => {
                assert!(msg.contains("nonexistent"));
                assert!(msg.contains("not found"));
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_conditional_validation_valid_register() {
        // Test that using a valid register works correctly
        let mut circuit = ClassicalCircuit::<2>::new();
        circuit
            .add_classical_register("measurement", 1)
            .expect("Failed to add classical register");
        circuit
            .measure(QubitId(0), "measurement", 0)
            .expect("Failed to add measurement");

        // Add a conditional gate with a valid condition
        let condition = ClassicalCondition::register_equals("measurement", 1);
        let result = circuit.add_conditional(condition, PauliX { target: QubitId(1) });

        assert!(result.is_ok());
        assert_eq!(circuit.num_operations(), 2); // Measure + Conditional
    }
}
