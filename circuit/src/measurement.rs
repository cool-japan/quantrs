//! Mid-circuit measurement and feed-forward support
//!
//! This module provides functionality for performing measurements during circuit
//! execution and using the results to control subsequent quantum operations.

use crate::builder::Circuit;
use crate::classical::{ClassicalCondition, ClassicalRegister};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Measurement operation that can be performed mid-circuit
#[derive(Debug, Clone)]
pub struct Measurement {
    /// Qubit to measure
    pub qubit: QubitId,
    /// Classical register to store result
    pub target_bit: usize,
    /// Optional label for the measurement
    pub label: Option<String>,
}

impl Measurement {
    /// Create a new measurement operation
    #[must_use]
    pub const fn new(qubit: QubitId, target_bit: usize) -> Self {
        Self {
            qubit,
            target_bit,
            label: None,
        }
    }

    /// Add a label to the measurement
    #[must_use]
    pub fn with_label(mut self, label: String) -> Self {
        self.label = Some(label);
        self
    }
}

/// Feed-forward operation based on measurement results
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// Condition for applying the operation
    pub condition: ClassicalCondition,
    /// Gate to apply if condition is met
    pub gate: Box<dyn GateOp>,
    /// Optional else gate
    pub else_gate: Option<Box<dyn GateOp>>,
}

impl FeedForward {
    /// Create a new feed-forward operation
    #[must_use]
    pub fn new(condition: ClassicalCondition, gate: Box<dyn GateOp>) -> Self {
        Self {
            condition,
            gate,
            else_gate: None,
        }
    }

    /// Add an else gate to apply if condition is not met
    #[must_use]
    pub fn with_else(mut self, else_gate: Box<dyn GateOp>) -> Self {
        self.else_gate = Some(else_gate);
        self
    }
}

/// Circuit operation that can include measurements and feed-forward
#[derive(Debug, Clone)]
pub enum CircuitOp {
    /// Standard quantum gate
    Gate(Box<dyn GateOp>),
    /// Mid-circuit measurement
    Measure(Measurement),
    /// Feed-forward operation
    FeedForward(FeedForward),
    /// Barrier for synchronization
    Barrier(Vec<QubitId>),
    /// Reset qubit to |0⟩
    Reset(QubitId),
}

/// Enhanced circuit builder with measurement support
pub struct MeasurementCircuit<const N: usize> {
    /// Operations in the circuit
    operations: Vec<CircuitOp>,
    /// Classical registers for storing measurement results
    classical_registers: HashMap<String, ClassicalRegister>,
    /// Measurement count for tracking
    measurement_count: usize,
    /// Current classical bit allocation
    current_bit: usize,
}

impl<const N: usize> Default for MeasurementCircuit<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> MeasurementCircuit<N> {
    /// Create a new measurement-enabled circuit
    #[must_use]
    pub fn new() -> Self {
        let mut classical_registers = HashMap::new();
        classical_registers.insert(
            "default".to_string(),
            ClassicalRegister::new("default".to_string(), N),
        );

        Self {
            operations: Vec::new(),
            classical_registers,
            measurement_count: 0,
            current_bit: 0,
        }
    }

    /// Add a quantum gate
    pub fn add_gate(&mut self, gate: Box<dyn GateOp>) -> QuantRS2Result<()> {
        // Validate qubit indices
        for qubit in gate.qubits() {
            if qubit.id() >= N as u32 {
                return Err(QuantRS2Error::InvalidQubitId(qubit.id()));
            }
        }

        self.operations.push(CircuitOp::Gate(gate));
        Ok(())
    }

    /// Add a mid-circuit measurement
    pub fn measure(&mut self, qubit: QubitId) -> QuantRS2Result<usize> {
        if qubit.id() >= N as u32 {
            return Err(QuantRS2Error::InvalidQubitId(qubit.id()));
        }

        if self.current_bit >= N {
            return Err(QuantRS2Error::InvalidInput(
                "Not enough classical bits for measurement".to_string(),
            ));
        }

        let target_bit = self.current_bit;
        self.current_bit += 1;

        let measurement =
            Measurement::new(qubit, target_bit).with_label(format!("m{}", self.measurement_count));

        self.operations.push(CircuitOp::Measure(measurement));
        self.measurement_count += 1;

        Ok(target_bit)
    }

    /// Add a conditional gate based on measurement result
    pub fn add_conditional(
        &mut self,
        condition: ClassicalCondition,
        gate: Box<dyn GateOp>,
    ) -> QuantRS2Result<()> {
        // Validate condition - simplified validation
        // In a full implementation, this would validate the register references

        // Validate gate qubits
        for qubit in gate.qubits() {
            if qubit.id() >= N as u32 {
                return Err(QuantRS2Error::InvalidQubitId(qubit.id()));
            }
        }

        let feed_forward = FeedForward::new(condition, gate);
        self.operations.push(CircuitOp::FeedForward(feed_forward));
        Ok(())
    }

    /// Add a conditional gate with else branch
    pub fn add_if_else(
        &mut self,
        condition: ClassicalCondition,
        if_gate: Box<dyn GateOp>,
        else_gate: Box<dyn GateOp>,
    ) -> QuantRS2Result<()> {
        // Validate gates
        for qubit in if_gate.qubits().iter().chain(else_gate.qubits().iter()) {
            if qubit.id() >= N as u32 {
                return Err(QuantRS2Error::InvalidQubitId(qubit.id()));
            }
        }

        let feed_forward = FeedForward::new(condition, if_gate).with_else(else_gate);
        self.operations.push(CircuitOp::FeedForward(feed_forward));
        Ok(())
    }

    /// Add a barrier for synchronization
    pub fn barrier(&mut self, qubits: Vec<QubitId>) -> QuantRS2Result<()> {
        for qubit in &qubits {
            if qubit.id() >= N as u32 {
                return Err(QuantRS2Error::InvalidQubitId(qubit.id()));
            }
        }

        self.operations.push(CircuitOp::Barrier(qubits));
        Ok(())
    }

    /// Reset a qubit to |0⟩
    pub fn reset(&mut self, qubit: QubitId) -> QuantRS2Result<()> {
        if qubit.id() >= N as u32 {
            return Err(QuantRS2Error::InvalidQubitId(qubit.id()));
        }

        self.operations.push(CircuitOp::Reset(qubit));
        Ok(())
    }

    /// Get the number of operations
    #[must_use]
    pub fn num_operations(&self) -> usize {
        self.operations.len()
    }

    /// Get the number of measurements
    #[must_use]
    pub const fn num_measurements(&self) -> usize {
        self.measurement_count
    }

    /// Get all operations
    #[must_use]
    pub fn operations(&self) -> &[CircuitOp] {
        &self.operations
    }

    /// Convert to a standard circuit (without measurements)
    pub fn to_circuit(&self) -> QuantRS2Result<Circuit<N>> {
        let mut circuit = Circuit::<N>::new();

        for op in &self.operations {
            match op {
                CircuitOp::Gate(gate) => {
                    // Convert Box<dyn GateOp> to concrete gate type
                    // For now, just skip as we can't easily convert
                    // circuit.add_gate(gate.clone())?;
                }
                CircuitOp::Measure(_) => {
                    // Skip measurements in standard circuit
                }
                CircuitOp::FeedForward(ff) => {
                    // Add only the gate, not the condition
                    // circuit.add_gate(ff.gate.clone())?;
                }
                CircuitOp::Barrier(_) => {
                    // Skip barriers in standard circuit
                }
                CircuitOp::Reset(_) => {
                    // Skip resets in standard circuit
                }
            }
        }

        Ok(circuit)
    }

    /// Analyze the circuit for measurement dependencies
    #[must_use]
    pub fn analyze_dependencies(&self) -> MeasurementDependencies {
        let mut deps = MeasurementDependencies::new();
        let mut measurement_map = HashMap::new();

        // First pass: collect all measurements
        for (i, op) in self.operations.iter().enumerate() {
            if let CircuitOp::Measure(m) = op {
                measurement_map.insert(m.target_bit, i);
                deps.measurements.push((i, m.clone()));
            }
        }

        // Second pass: find feed-forward dependencies
        for (i, op) in self.operations.iter().enumerate() {
            if let CircuitOp::FeedForward(_ff) = op {
                // In a full implementation, this would properly track classical dependencies
                // For now, assume all feed-forward depends on previous measurements
                if !measurement_map.is_empty() {
                    let last_measurement = measurement_map.len() - 1;
                    deps.feed_forward_deps.push((last_measurement, i));
                }
            }
        }

        deps
    }
}

/// Analysis result for measurement dependencies
#[derive(Debug)]
pub struct MeasurementDependencies {
    /// List of (index, measurement) pairs
    pub measurements: Vec<(usize, Measurement)>,
    /// List of (`measurement_index`, `feedforward_index`) dependencies
    pub feed_forward_deps: Vec<(usize, usize)>,
}

impl MeasurementDependencies {
    const fn new() -> Self {
        Self {
            measurements: Vec::new(),
            feed_forward_deps: Vec::new(),
        }
    }

    /// Check if there are any feed-forward operations
    #[must_use]
    pub fn has_feed_forward(&self) -> bool {
        !self.feed_forward_deps.is_empty()
    }

    /// Get the number of measurements
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.measurements.len()
    }
}

/// Builder pattern for measurement circuits
pub struct MeasurementCircuitBuilder<const N: usize> {
    circuit: MeasurementCircuit<N>,
}

impl<const N: usize> Default for MeasurementCircuitBuilder<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> MeasurementCircuitBuilder<N> {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            circuit: MeasurementCircuit::new(),
        }
    }

    /// Add a gate
    pub fn gate(mut self, gate: Box<dyn GateOp>) -> QuantRS2Result<Self> {
        self.circuit.add_gate(gate)?;
        Ok(self)
    }

    /// Add a measurement
    pub fn measure(mut self, qubit: QubitId) -> QuantRS2Result<(Self, usize)> {
        let bit = self.circuit.measure(qubit)?;
        Ok((self, bit))
    }

    /// Add a conditional gate
    pub fn when(
        mut self,
        condition: ClassicalCondition,
        gate: Box<dyn GateOp>,
    ) -> QuantRS2Result<Self> {
        self.circuit.add_conditional(condition, gate)?;
        Ok(self)
    }

    /// Add a conditional gate with else
    pub fn if_else(
        mut self,
        condition: ClassicalCondition,
        if_gate: Box<dyn GateOp>,
        else_gate: Box<dyn GateOp>,
    ) -> QuantRS2Result<Self> {
        self.circuit.add_if_else(condition, if_gate, else_gate)?;
        Ok(self)
    }

    /// Add a barrier
    pub fn barrier(mut self, qubits: Vec<QubitId>) -> QuantRS2Result<Self> {
        self.circuit.barrier(qubits)?;
        Ok(self)
    }

    /// Reset a qubit
    pub fn reset(mut self, qubit: QubitId) -> QuantRS2Result<Self> {
        self.circuit.reset(qubit)?;
        Ok(self)
    }

    /// Build the circuit
    #[must_use]
    pub fn build(self) -> MeasurementCircuit<N> {
        self.circuit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::single::{Hadamard, PauliX};

    #[test]
    fn test_measurement_circuit() {
        let mut circuit = MeasurementCircuit::<3>::new();

        // Add Hadamard gate
        circuit
            .add_gate(Box::new(Hadamard { target: QubitId(0) }))
            .unwrap();

        // Measure qubit 0
        let bit0 = circuit.measure(QubitId(0)).unwrap();
        assert_eq!(bit0, 0);

        // Add conditional X gate
        let condition = ClassicalCondition::equals(
            crate::classical::ClassicalValue::Integer(bit0 as u64),
            crate::classical::ClassicalValue::Integer(1),
        );
        circuit
            .add_conditional(condition, Box::new(PauliX { target: QubitId(1) }))
            .unwrap();

        assert_eq!(circuit.num_operations(), 3);
        assert_eq!(circuit.num_measurements(), 1);
    }

    #[test]
    fn test_feed_forward() {
        let mut circuit = MeasurementCircuit::<2>::new();

        // Bell state preparation with measurement
        circuit
            .add_gate(Box::new(Hadamard { target: QubitId(0) }))
            .unwrap();
        circuit
            .add_gate(Box::new(quantrs2_core::gate::multi::CNOT {
                control: QubitId(0),
                target: QubitId(1),
            }))
            .unwrap();

        // Measure first qubit
        let bit = circuit.measure(QubitId(0)).unwrap();

        // Apply X to second qubit if first measured as 1
        let condition = ClassicalCondition::equals(
            crate::classical::ClassicalValue::Integer(bit as u64),
            crate::classical::ClassicalValue::Integer(1),
        );
        circuit
            .add_conditional(condition, Box::new(PauliX { target: QubitId(1) }))
            .unwrap();

        // Analyze dependencies
        let deps = circuit.analyze_dependencies();
        assert_eq!(deps.num_measurements(), 1);
        assert!(deps.has_feed_forward());
    }

    #[test]
    fn test_builder_pattern() {
        let (builder, bit) = MeasurementCircuitBuilder::<2>::new()
            .gate(Box::new(Hadamard { target: QubitId(0) }))
            .unwrap()
            .measure(QubitId(0))
            .unwrap();

        let circuit = builder
            .when(
                ClassicalCondition::equals(
                    crate::classical::ClassicalValue::Integer(bit as u64),
                    crate::classical::ClassicalValue::Integer(1),
                ),
                Box::new(PauliX { target: QubitId(1) }),
            )
            .unwrap()
            .build();

        assert_eq!(circuit.num_operations(), 3);
    }
}
