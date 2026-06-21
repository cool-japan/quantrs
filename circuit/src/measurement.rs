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
                CircuitOp::Gate(_)
                | CircuitOp::Measure(_)
                | CircuitOp::FeedForward(_)
                | CircuitOp::Barrier(_)
                | CircuitOp::Reset(_) => {
                    // Skip: gates can't be easily converted, measurements/barriers/resets not in standard circuit
                }
            }
        }

        Ok(circuit)
    }

    /// Analyze the circuit for measurement dependencies.
    ///
    /// Builds a map from each measured classical bit (`target_bit`) to the
    /// operation index that wrote it, then resolves each feed-forward operation
    /// to the measurement(s) it actually reads.
    ///
    /// Dependency resolution inspects the feed-forward's [`ClassicalCondition`]:
    /// any operand referencing a classical bit (by index or by integer value, as
    /// produced by the measurement builders) is matched against the known
    /// measured bits, and the feed-forward is linked to each matching
    /// measurement that occurs earlier in the circuit. Each entry in
    /// `feed_forward_deps` is `(measurement_op_index, feedforward_op_index)`.
    ///
    /// If a condition references no resolvable measured bit (the classical wiring
    /// is opaque to the circuit), the feed-forward is conservatively linked to
    /// every preceding measurement rather than guessing a single arbitrary one.
    #[must_use]
    pub fn analyze_dependencies(&self) -> MeasurementDependencies {
        let mut deps = MeasurementDependencies::new();
        // Map measured classical bit -> operation index that produced it.
        let mut measurement_map: HashMap<usize, usize> = HashMap::new();

        // First pass: collect all measurements.
        for (i, op) in self.operations.iter().enumerate() {
            if let CircuitOp::Measure(m) = op {
                measurement_map.insert(m.target_bit, i);
                deps.measurements.push((i, m.clone()));
            }
        }

        // Second pass: resolve feed-forward dependencies against measured bits.
        for (i, op) in self.operations.iter().enumerate() {
            if let CircuitOp::FeedForward(ff) = op {
                let referenced_bits = Self::condition_bit_references(&ff.condition);

                // Measurements (by op index) this feed-forward genuinely reads:
                // a referenced bit whose writing measurement precedes this op.
                let mut resolved: Vec<usize> = referenced_bits
                    .iter()
                    .filter_map(|bit| measurement_map.get(bit).copied())
                    .filter(|&meas_idx| meas_idx < i)
                    .collect();

                if resolved.is_empty() {
                    // No resolvable bit reference: fall back to a conservative
                    // (over-approximating) dependency on all prior measurements.
                    resolved = deps
                        .measurements
                        .iter()
                        .map(|(meas_idx, _)| *meas_idx)
                        .filter(|&meas_idx| meas_idx < i)
                        .collect();
                }

                resolved.sort_unstable();
                resolved.dedup();
                for meas_idx in resolved {
                    deps.feed_forward_deps.push((meas_idx, i));
                }
            }
        }

        deps
    }

    /// Extract the classical bit indices a condition reads.
    ///
    /// The measurement builders encode a measured bit either directly as a bit
    /// index or, by convention, as an [`ClassicalValue::Integer`] holding that
    /// index. Both operands of the comparison are inspected; register and raw
    /// boolean operands carry no recoverable bit index and are ignored here.
    fn condition_bit_references(condition: &ClassicalCondition) -> Vec<usize> {
        use crate::classical::ClassicalValue;

        let mut bits = Vec::new();
        for value in [&condition.lhs, &condition.rhs] {
            if let ClassicalValue::Integer(v) = value {
                if let Ok(bit) = usize::try_from(*v) {
                    bits.push(bit);
                }
            }
        }
        bits
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
            .expect("Failed to add Hadamard gate");

        // Measure qubit 0
        let bit0 = circuit
            .measure(QubitId(0))
            .expect("Failed to measure qubit 0");
        assert_eq!(bit0, 0);

        // Add conditional X gate
        let condition = ClassicalCondition::equals(
            crate::classical::ClassicalValue::Integer(bit0 as u64),
            crate::classical::ClassicalValue::Integer(1),
        );
        circuit
            .add_conditional(condition, Box::new(PauliX { target: QubitId(1) }))
            .expect("Failed to add conditional X gate");

        assert_eq!(circuit.num_operations(), 3);
        assert_eq!(circuit.num_measurements(), 1);
    }

    #[test]
    fn test_feed_forward() {
        let mut circuit = MeasurementCircuit::<2>::new();

        // Bell state preparation with measurement
        circuit
            .add_gate(Box::new(Hadamard { target: QubitId(0) }))
            .expect("Failed to add Hadamard gate");
        circuit
            .add_gate(Box::new(quantrs2_core::gate::multi::CNOT {
                control: QubitId(0),
                target: QubitId(1),
            }))
            .expect("Failed to add CNOT gate");

        // Measure first qubit
        let bit = circuit
            .measure(QubitId(0))
            .expect("Failed to measure qubit 0");

        // Apply X to second qubit if first measured as 1
        let condition = ClassicalCondition::equals(
            crate::classical::ClassicalValue::Integer(bit as u64),
            crate::classical::ClassicalValue::Integer(1),
        );
        circuit
            .add_conditional(condition, Box::new(PauliX { target: QubitId(1) }))
            .expect("Failed to add conditional X gate");

        // Analyze dependencies
        let deps = circuit.analyze_dependencies();
        assert_eq!(deps.num_measurements(), 1);
        assert!(deps.has_feed_forward());

        // The single feed-forward must depend on the single measurement, and
        // the recorded measurement op index must precede the feed-forward op
        // index.
        assert_eq!(deps.feed_forward_deps.len(), 1);
        let (meas_idx, ff_idx) = deps.feed_forward_deps[0];
        assert!(meas_idx < ff_idx);
        // The measurement op index recorded in the dependency must be a real
        // measurement listed in `deps.measurements`.
        assert!(deps.measurements.iter().any(|(idx, _)| *idx == meas_idx));
    }

    #[test]
    fn test_feed_forward_resolves_correct_measurement() {
        use crate::classical::ClassicalValue;

        let mut circuit = MeasurementCircuit::<3>::new();

        // Measure qubit 0 -> bit 0, qubit 1 -> bit 1.
        let bit0 = circuit.measure(QubitId(0)).expect("measure q0");
        let bit1 = circuit.measure(QubitId(1)).expect("measure q1");
        assert_eq!(bit0, 0);
        assert_eq!(bit1, 1);

        // Conditional reads bit 1 only.
        let condition = ClassicalCondition::equals(
            ClassicalValue::Integer(bit1 as u64),
            ClassicalValue::Integer(1),
        );
        circuit
            .add_conditional(condition, Box::new(PauliX { target: QubitId(2) }))
            .expect("add conditional");

        let deps = circuit.analyze_dependencies();
        assert_eq!(deps.num_measurements(), 2);

        // The feed-forward should depend specifically on the measurement that
        // wrote bit 1 (operation index 1), not bit 0 (operation index 0).
        // (RHS Integer(1) coincidentally equals bit index 1, so a dependency on
        // the bit-1 measurement is expected; the bit-0 measurement must NOT be
        // linked since it is not referenced.)
        assert!(deps
            .feed_forward_deps
            .iter()
            .any(|&(meas_idx, _)| meas_idx == 1));
        assert!(!deps
            .feed_forward_deps
            .iter()
            .any(|&(meas_idx, _)| meas_idx == 0));
    }

    #[test]
    fn test_builder_pattern() {
        let (builder, bit) = MeasurementCircuitBuilder::<2>::new()
            .gate(Box::new(Hadamard { target: QubitId(0) }))
            .expect("Failed to add gate")
            .measure(QubitId(0))
            .expect("Failed to measure qubit");

        let circuit = builder
            .when(
                ClassicalCondition::equals(
                    crate::classical::ClassicalValue::Integer(bit as u64),
                    crate::classical::ClassicalValue::Integer(1),
                ),
                Box::new(PauliX { target: QubitId(1) }),
            )
            .expect("Failed to add conditional gate")
            .build();

        assert_eq!(circuit.num_operations(), 3);
    }
}
