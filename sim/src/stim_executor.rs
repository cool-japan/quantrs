//! Stim circuit executor with detector and observable support
//!
//! This module provides execution capabilities for Stim circuits, including:
//! - Running circuits with the stabilizer simulator
//! - Tracking measurement records for detector evaluation
//! - Computing detector values from measurement parities
//! - Computing observable values for logical qubit tracking
//!
//! ## Detector Model
//!
//! Detectors check parity conditions on measurement results. A detector triggers
//! when the XOR of its referenced measurement results differs from the expected value.
//!
//! ## Observable Model
//!
//! Observables track logical qubit values through a circuit. They are computed
//! as the XOR of specified measurement results.

use crate::error::{Result, SimulatorError};
use crate::stabilizer::{StabilizerSimulator, StabilizerTableau};
use crate::stim_parser::{
    MeasurementBasis, PauliTarget, PauliType, SingleQubitGateType, StimCircuit, StimInstruction,
    TwoQubitGateType,
};
use scirs2_core::random::prelude::*;

/// Record of a detector definition
#[derive(Debug, Clone)]
pub struct DetectorRecord {
    /// Detector index
    pub index: usize,
    /// Coordinates for visualization/debugging
    pub coordinates: Vec<f64>,
    /// Measurement record indices this detector depends on (negative = relative)
    pub record_targets: Vec<i32>,
    /// Expected parity (usually false for no error)
    pub expected_parity: bool,
}

/// Record of an observable definition
#[derive(Debug, Clone)]
pub struct ObservableRecord {
    /// Observable index
    pub index: usize,
    /// Measurement record indices this observable depends on
    pub record_targets: Vec<i32>,
}

/// Stim circuit executor with full error correction support
#[derive(Debug, Clone)]
pub struct StimExecutor {
    /// Stabilizer simulator for circuit execution
    simulator: StabilizerSimulator,
    /// Measurement record (chronological order)
    measurement_record: Vec<bool>,
    /// Detector definitions
    detectors: Vec<DetectorRecord>,
    /// Observable definitions
    observables: Vec<ObservableRecord>,
    /// Error tracking state for E/ELSE_CORRELATED_ERROR chains
    last_error_triggered: bool,
    /// Number of qubits
    num_qubits: usize,
}

impl StimExecutor {
    /// Create a new Stim executor
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            simulator: StabilizerSimulator::new(num_qubits),
            measurement_record: Vec::new(),
            detectors: Vec::new(),
            observables: Vec::new(),
            last_error_triggered: false,
            num_qubits,
        }
    }

    /// Create from a Stim circuit (auto-determines qubit count)
    #[must_use]
    pub fn from_circuit(circuit: &StimCircuit) -> Self {
        Self::new(circuit.num_qubits)
    }

    /// Execute a full Stim circuit
    pub fn execute(&mut self, circuit: &StimCircuit) -> Result<ExecutionResult> {
        // Reset state for new execution
        self.measurement_record.clear();
        self.detectors.clear();
        self.observables.clear();
        self.last_error_triggered = false;

        // Execute each instruction
        for instruction in &circuit.instructions {
            self.execute_instruction(instruction)?;
        }

        // Compute detector and observable values
        let detector_values = self.compute_detector_values();
        let observable_values = self.compute_observable_values();

        Ok(ExecutionResult {
            measurement_record: self.measurement_record.clone(),
            detector_values,
            observable_values,
            num_measurements: self.measurement_record.len(),
            num_detectors: self.detectors.len(),
            num_observables: self.observables.len(),
        })
    }

    /// Execute a single instruction
    fn execute_instruction(&mut self, instruction: &StimInstruction) -> Result<()> {
        match instruction {
            // Gates
            StimInstruction::SingleQubitGate { gate_type, qubit } => {
                self.execute_single_qubit_gate(*gate_type, *qubit)?;
            }
            StimInstruction::TwoQubitGate {
                gate_type,
                control,
                target,
            } => {
                self.execute_two_qubit_gate(*gate_type, *control, *target)?;
            }

            // Measurements
            StimInstruction::Measure { basis, qubits } => {
                self.execute_measurement(*basis, qubits)?;
            }

            // Reset
            StimInstruction::Reset { qubits } => {
                self.execute_reset(qubits)?;
            }

            // Measure and reset
            StimInstruction::MeasureReset { basis, qubits } => {
                self.execute_measure_reset(*basis, qubits)?;
            }

            // Detectors and observables
            StimInstruction::Detector {
                coordinates,
                record_targets,
            } => {
                self.process_detector(coordinates, record_targets)?;
            }
            StimInstruction::ObservableInclude {
                observable_index,
                record_targets,
            } => {
                self.process_observable(*observable_index, record_targets)?;
            }

            // Noise instructions (stochastic)
            StimInstruction::XError {
                probability,
                qubits,
            } => {
                self.execute_x_error(*probability, qubits)?;
            }
            StimInstruction::YError {
                probability,
                qubits,
            } => {
                self.execute_y_error(*probability, qubits)?;
            }
            StimInstruction::ZError {
                probability,
                qubits,
            } => {
                self.execute_z_error(*probability, qubits)?;
            }
            StimInstruction::Depolarize1 {
                probability,
                qubits,
            } => {
                self.execute_depolarize1(*probability, qubits)?;
            }
            StimInstruction::Depolarize2 {
                probability,
                qubit_pairs,
            } => {
                self.execute_depolarize2(*probability, qubit_pairs)?;
            }
            StimInstruction::CorrelatedError {
                probability,
                targets,
            } => {
                self.execute_correlated_error(*probability, targets)?;
            }
            StimInstruction::ElseCorrelatedError {
                probability,
                targets,
            } => {
                self.execute_else_correlated_error(*probability, targets)?;
            }
            StimInstruction::PauliChannel1 { px, py, pz, qubits } => {
                self.execute_pauli_channel_1(*px, *py, *pz, qubits)?;
            }
            StimInstruction::PauliChannel2 {
                probabilities,
                qubit_pairs,
            } => {
                self.execute_pauli_channel_2(probabilities, qubit_pairs)?;
            }

            // Metadata instructions (no execution effect)
            StimInstruction::Comment(_)
            | StimInstruction::Tick
            | StimInstruction::ShiftCoords { .. }
            | StimInstruction::QubitCoords { .. } => {}

            // Repeat blocks
            StimInstruction::Repeat {
                count,
                instructions,
            } => {
                for _ in 0..*count {
                    for inst in instructions {
                        self.execute_instruction(inst)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Execute a single-qubit gate
    fn execute_single_qubit_gate(
        &mut self,
        gate_type: SingleQubitGateType,
        qubit: usize,
    ) -> Result<()> {
        let gate = gate_type.to_stabilizer_gate(qubit);
        self.simulator.apply_gate(gate).map_err(|e| {
            SimulatorError::InvalidOperation(format!("Gate execution failed: {:?}", e))
        })
    }

    /// Execute a two-qubit gate
    fn execute_two_qubit_gate(
        &mut self,
        gate_type: TwoQubitGateType,
        control: usize,
        target: usize,
    ) -> Result<()> {
        let gate = gate_type.to_stabilizer_gate(control, target);
        self.simulator.apply_gate(gate).map_err(|e| {
            SimulatorError::InvalidOperation(format!("Gate execution failed: {:?}", e))
        })
    }

    /// Execute measurements
    fn execute_measurement(&mut self, basis: MeasurementBasis, qubits: &[usize]) -> Result<()> {
        for &qubit in qubits {
            let outcome = match basis {
                MeasurementBasis::Z => self.simulator.measure(qubit),
                MeasurementBasis::X => self.simulator.tableau.measure_x(qubit),
                MeasurementBasis::Y => self.simulator.tableau.measure_y(qubit),
            }
            .map_err(|e| {
                SimulatorError::InvalidOperation(format!("Measurement failed: {:?}", e))
            })?;

            self.measurement_record.push(outcome);
        }
        Ok(())
    }

    /// Execute reset operations
    fn execute_reset(&mut self, qubits: &[usize]) -> Result<()> {
        for &qubit in qubits {
            self.simulator
                .tableau
                .reset(qubit)
                .map_err(|e| SimulatorError::InvalidOperation(format!("Reset failed: {:?}", e)))?;
        }
        Ok(())
    }

    /// Execute measure-and-reset operations
    fn execute_measure_reset(&mut self, basis: MeasurementBasis, qubits: &[usize]) -> Result<()> {
        self.execute_measurement(basis, qubits)?;
        self.execute_reset(qubits)
    }

    /// Process a DETECTOR instruction
    fn process_detector(&mut self, coordinates: &[f64], record_targets: &[i32]) -> Result<()> {
        let detector = DetectorRecord {
            index: self.detectors.len(),
            coordinates: coordinates.to_vec(),
            record_targets: record_targets.to_vec(),
            expected_parity: false, // Default expectation is no error (parity = 0)
        };
        self.detectors.push(detector);
        Ok(())
    }

    /// Process an OBSERVABLE_INCLUDE instruction
    fn process_observable(
        &mut self,
        observable_index: usize,
        record_targets: &[i32],
    ) -> Result<()> {
        // Find or create the observable record
        while self.observables.len() <= observable_index {
            self.observables.push(ObservableRecord {
                index: self.observables.len(),
                record_targets: Vec::new(),
            });
        }

        // Add the record targets to the observable
        self.observables[observable_index]
            .record_targets
            .extend_from_slice(record_targets);

        Ok(())
    }

    /// Compute detector values from measurement record
    pub fn compute_detector_values(&self) -> Vec<bool> {
        self.detectors
            .iter()
            .map(|detector| {
                let parity = self.compute_record_parity(&detector.record_targets);
                // Detector fires when parity differs from expected
                parity != detector.expected_parity
            })
            .collect()
    }

    /// Compute observable values from measurement record
    pub fn compute_observable_values(&self) -> Vec<bool> {
        self.observables
            .iter()
            .map(|observable| self.compute_record_parity(&observable.record_targets))
            .collect()
    }

    /// Compute XOR parity of measurement results at given record indices
    fn compute_record_parity(&self, record_targets: &[i32]) -> bool {
        let record_len = self.measurement_record.len() as i32;

        record_targets
            .iter()
            .filter_map(|&idx| {
                // Convert relative (negative) to absolute index
                let abs_idx = if idx < 0 {
                    (record_len + idx) as usize
                } else {
                    idx as usize
                };

                self.measurement_record.get(abs_idx).copied()
            })
            .fold(false, |acc, x| acc ^ x)
    }

    // ==================== Error/Noise Execution ====================

    /// Execute X errors with given probability
    fn execute_x_error(&mut self, probability: f64, qubits: &[usize]) -> Result<()> {
        let mut rng = thread_rng();
        for &qubit in qubits {
            if rng.gen_bool(probability) {
                self.simulator
                    .apply_gate(crate::stabilizer::StabilizerGate::X(qubit))
                    .map_err(|e| {
                        SimulatorError::InvalidOperation(format!("X error failed: {:?}", e))
                    })?;
            }
        }
        Ok(())
    }

    /// Execute Y errors with given probability
    fn execute_y_error(&mut self, probability: f64, qubits: &[usize]) -> Result<()> {
        let mut rng = thread_rng();
        for &qubit in qubits {
            if rng.gen_bool(probability) {
                self.simulator
                    .apply_gate(crate::stabilizer::StabilizerGate::Y(qubit))
                    .map_err(|e| {
                        SimulatorError::InvalidOperation(format!("Y error failed: {:?}", e))
                    })?;
            }
        }
        Ok(())
    }

    /// Execute Z errors with given probability
    fn execute_z_error(&mut self, probability: f64, qubits: &[usize]) -> Result<()> {
        let mut rng = thread_rng();
        for &qubit in qubits {
            if rng.gen_bool(probability) {
                self.simulator
                    .apply_gate(crate::stabilizer::StabilizerGate::Z(qubit))
                    .map_err(|e| {
                        SimulatorError::InvalidOperation(format!("Z error failed: {:?}", e))
                    })?;
            }
        }
        Ok(())
    }

    /// Execute single-qubit depolarizing noise
    fn execute_depolarize1(&mut self, probability: f64, qubits: &[usize]) -> Result<()> {
        let mut rng = thread_rng();
        // Depolarizing: with prob p, apply X, Y, or Z uniformly at random
        for &qubit in qubits {
            if rng.gen_bool(probability) {
                let error_type: u8 = rng.gen_range(0..3);
                let gate = match error_type {
                    0 => crate::stabilizer::StabilizerGate::X(qubit),
                    1 => crate::stabilizer::StabilizerGate::Y(qubit),
                    _ => crate::stabilizer::StabilizerGate::Z(qubit),
                };
                self.simulator.apply_gate(gate).map_err(|e| {
                    SimulatorError::InvalidOperation(format!("Depolarizing error failed: {:?}", e))
                })?;
            }
        }
        Ok(())
    }

    /// Execute two-qubit depolarizing noise
    fn execute_depolarize2(
        &mut self,
        probability: f64,
        qubit_pairs: &[(usize, usize)],
    ) -> Result<()> {
        let mut rng = thread_rng();
        // Two-qubit depolarizing: 15 non-identity Pauli pairs
        for &(q1, q2) in qubit_pairs {
            if rng.gen_bool(probability) {
                // Select one of 15 non-identity two-qubit Paulis
                let error_idx: u8 = rng.gen_range(0..15);
                let (pauli1, pauli2) = Self::two_qubit_pauli_from_index(error_idx);
                self.apply_pauli_to_qubit(pauli1, q1)?;
                self.apply_pauli_to_qubit(pauli2, q2)?;
            }
        }
        Ok(())
    }

    /// Execute a correlated error (E instruction)
    fn execute_correlated_error(
        &mut self,
        probability: f64,
        targets: &[PauliTarget],
    ) -> Result<()> {
        let mut rng = thread_rng();
        self.last_error_triggered = rng.gen_bool(probability);

        if self.last_error_triggered {
            for target in targets {
                self.apply_pauli_to_qubit(target.pauli, target.qubit)?;
            }
        }
        Ok(())
    }

    /// Execute an else-correlated error (ELSE_CORRELATED_ERROR instruction)
    fn execute_else_correlated_error(
        &mut self,
        probability: f64,
        targets: &[PauliTarget],
    ) -> Result<()> {
        // Only consider triggering if the previous E did NOT trigger
        if !self.last_error_triggered {
            let mut rng = thread_rng();
            self.last_error_triggered = rng.gen_bool(probability);

            if self.last_error_triggered {
                for target in targets {
                    self.apply_pauli_to_qubit(target.pauli, target.qubit)?;
                }
            }
        }
        // If previous error triggered, this one doesn't run (else branch not taken)
        Ok(())
    }

    /// Execute Pauli channel (single qubit)
    fn execute_pauli_channel_1(
        &mut self,
        px: f64,
        py: f64,
        pz: f64,
        qubits: &[usize],
    ) -> Result<()> {
        let mut rng = thread_rng();
        for &qubit in qubits {
            let r: f64 = rng.gen();
            if r < px {
                self.simulator
                    .apply_gate(crate::stabilizer::StabilizerGate::X(qubit))
                    .map_err(|e| {
                        SimulatorError::InvalidOperation(format!("Pauli channel failed: {:?}", e))
                    })?;
            } else if r < px + py {
                self.simulator
                    .apply_gate(crate::stabilizer::StabilizerGate::Y(qubit))
                    .map_err(|e| {
                        SimulatorError::InvalidOperation(format!("Pauli channel failed: {:?}", e))
                    })?;
            } else if r < px + py + pz {
                self.simulator
                    .apply_gate(crate::stabilizer::StabilizerGate::Z(qubit))
                    .map_err(|e| {
                        SimulatorError::InvalidOperation(format!("Pauli channel failed: {:?}", e))
                    })?;
            }
            // Otherwise, identity (no error)
        }
        Ok(())
    }

    /// Execute Pauli channel (two qubits)
    fn execute_pauli_channel_2(
        &mut self,
        probabilities: &[f64],
        qubit_pairs: &[(usize, usize)],
    ) -> Result<()> {
        if probabilities.len() != 15 {
            return Err(SimulatorError::InvalidOperation(
                "PAULI_CHANNEL_2 requires 15 probabilities".to_string(),
            ));
        }

        let mut rng = thread_rng();
        for &(q1, q2) in qubit_pairs {
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;

            for (i, &p) in probabilities.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    let (pauli1, pauli2) = Self::two_qubit_pauli_from_index(i as u8);
                    self.apply_pauli_to_qubit(pauli1, q1)?;
                    self.apply_pauli_to_qubit(pauli2, q2)?;
                    break;
                }
            }
        }
        Ok(())
    }

    /// Apply a Pauli operator to a qubit
    fn apply_pauli_to_qubit(&mut self, pauli: PauliType, qubit: usize) -> Result<()> {
        let gate = match pauli {
            PauliType::I => return Ok(()), // Identity, do nothing
            PauliType::X => crate::stabilizer::StabilizerGate::X(qubit),
            PauliType::Y => crate::stabilizer::StabilizerGate::Y(qubit),
            PauliType::Z => crate::stabilizer::StabilizerGate::Z(qubit),
        };
        self.simulator.apply_gate(gate).map_err(|e| {
            SimulatorError::InvalidOperation(format!("Pauli application failed: {:?}", e))
        })
    }

    /// Convert index (0-14) to two-qubit Pauli pair (excluding II)
    /// Order: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ
    fn two_qubit_pauli_from_index(idx: u8) -> (PauliType, PauliType) {
        // Mapping: 0=I, 1=X, 2=Y, 3=Z
        // Index maps to (p1, p2) excluding (0,0)
        let idx = idx + 1; // Skip II
        let p1 = idx / 4;
        let p2 = idx % 4;

        let to_pauli = |i| match i {
            0 => PauliType::I,
            1 => PauliType::X,
            2 => PauliType::Y,
            _ => PauliType::Z,
        };

        (to_pauli(p1), to_pauli(p2))
    }

    // ==================== Accessors ====================

    /// Get the measurement record
    #[must_use]
    pub fn measurement_record(&self) -> &[bool] {
        &self.measurement_record
    }

    /// Get detector definitions
    #[must_use]
    pub fn detectors(&self) -> &[DetectorRecord] {
        &self.detectors
    }

    /// Get observable definitions
    #[must_use]
    pub fn observables(&self) -> &[ObservableRecord] {
        &self.observables
    }

    /// Get the stabilizer simulator
    #[must_use]
    pub fn simulator(&self) -> &StabilizerSimulator {
        &self.simulator
    }

    /// Get the current stabilizers
    #[must_use]
    pub fn get_stabilizers(&self) -> Vec<String> {
        self.simulator.get_stabilizers()
    }

    /// Get the number of qubits
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// Result of executing a Stim circuit
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Full measurement record
    pub measurement_record: Vec<bool>,
    /// Detector values (true = detector fired)
    pub detector_values: Vec<bool>,
    /// Observable values
    pub observable_values: Vec<bool>,
    /// Total number of measurements
    pub num_measurements: usize,
    /// Total number of detectors
    pub num_detectors: usize,
    /// Total number of observables
    pub num_observables: usize,
}

impl ExecutionResult {
    /// Check if any detector fired (indicating an error was detected)
    #[must_use]
    pub fn any_detector_fired(&self) -> bool {
        self.detector_values.iter().any(|&x| x)
    }

    /// Count how many detectors fired
    #[must_use]
    pub fn detector_fire_count(&self) -> usize {
        self.detector_values.iter().filter(|&&x| x).count()
    }

    /// Get a bit-packed representation of measurements
    #[must_use]
    pub fn packed_measurements(&self) -> Vec<u8> {
        self.measurement_record
            .chunks(8)
            .map(|chunk| {
                let mut byte = 0u8;
                for (i, &bit) in chunk.iter().enumerate() {
                    if bit {
                        byte |= 1 << i;
                    }
                }
                byte
            })
            .collect()
    }

    /// Get a bit-packed representation of detector values
    #[must_use]
    pub fn packed_detectors(&self) -> Vec<u8> {
        self.detector_values
            .chunks(8)
            .map(|chunk| {
                let mut byte = 0u8;
                for (i, &bit) in chunk.iter().enumerate() {
                    if bit {
                        byte |= 1 << i;
                    }
                }
                byte
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_execution() {
        let circuit_str = r#"
            H 0
            CNOT 0 1
            M 0 1
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let mut executor = StimExecutor::from_circuit(&circuit);
        let result = executor.execute(&circuit).unwrap();

        assert_eq!(result.num_measurements, 2);
        // Bell state: measurements should be correlated
        assert_eq!(result.measurement_record[0], result.measurement_record[1]);
    }

    #[test]
    fn test_detector_execution() {
        // Deterministic test: prepare |00⟩ state, measure both
        // M0=0, M1=0, XOR=0, detector should not fire
        let circuit_str = r#"
            M 0 1
            DETECTOR rec[-1] rec[-2]
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let mut executor = StimExecutor::from_circuit(&circuit);
        let result = executor.execute(&circuit).unwrap();

        assert_eq!(result.num_detectors, 1);
        // |00⟩ state: M0=0, M1=0, XOR=0, detector should NOT fire
        assert_eq!(result.measurement_record[0], false);
        assert_eq!(result.measurement_record[1], false);
        assert!(!result.detector_values[0]);
    }

    #[test]
    fn test_detector_with_error() {
        // Test detector firing when error is introduced
        // Apply X to qubit 0, measure both: M0=1, M1=0, XOR=1
        let circuit_str = r#"
            X 0
            M 0 1
            DETECTOR rec[-1] rec[-2]
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let mut executor = StimExecutor::from_circuit(&circuit);
        let result = executor.execute(&circuit).unwrap();

        assert_eq!(result.num_detectors, 1);
        // |10⟩ state: M0=1, M1=0, XOR=1, detector SHOULD fire
        assert_eq!(result.measurement_record[0], true);
        assert_eq!(result.measurement_record[1], false);
        assert!(result.detector_values[0]);
    }

    #[test]
    fn test_observable_execution() {
        let circuit_str = r#"
            H 0
            CNOT 0 1
            M 0 1
            OBSERVABLE_INCLUDE(0) rec[-1]
            OBSERVABLE_INCLUDE(1) rec[-2]
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let mut executor = StimExecutor::from_circuit(&circuit);
        let result = executor.execute(&circuit).unwrap();

        assert_eq!(result.num_observables, 2);
    }

    #[test]
    fn test_measure_reset() {
        let circuit_str = r#"
            H 0
            MR 0
            M 0
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let mut executor = StimExecutor::from_circuit(&circuit);
        let result = executor.execute(&circuit).unwrap();

        assert_eq!(result.num_measurements, 2);
        // After reset, second measurement should be 0
        assert!(!result.measurement_record[1]);
    }

    #[test]
    fn test_noise_execution() {
        // Note: This test is probabilistic, but with p=1.0 it's deterministic
        let circuit_str = r#"
            X_ERROR(1.0) 0
            M 0
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let mut executor = StimExecutor::from_circuit(&circuit);
        let result = executor.execute(&circuit).unwrap();

        // X error on |0⟩ gives |1⟩, so measurement should be 1
        assert!(result.measurement_record[0]);
    }

    #[test]
    fn test_correlated_error() {
        // E with probability 1.0 should always trigger
        let circuit_str = r#"
            E(1.0) X0 X1
            M 0 1
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let mut executor = StimExecutor::from_circuit(&circuit);
        let result = executor.execute(&circuit).unwrap();

        // Both qubits should be flipped
        assert!(result.measurement_record[0]);
        assert!(result.measurement_record[1]);
    }

    #[test]
    fn test_else_correlated_error() {
        // E(1.0) always triggers, so ELSE should not
        let circuit_str = r#"
            E(1.0) X0
            ELSE_CORRELATED_ERROR(1.0) X1
            M 0 1
        "#;

        let circuit = StimCircuit::from_str(circuit_str).unwrap();
        let mut executor = StimExecutor::from_circuit(&circuit);
        let result = executor.execute(&circuit).unwrap();

        // E triggers, ELSE doesn't
        assert!(result.measurement_record[0]); // X0 applied
        assert!(!result.measurement_record[1]); // X1 NOT applied
    }

    #[test]
    fn test_packed_measurements() {
        let mut result = ExecutionResult {
            measurement_record: vec![true, false, true, true, false, false, true, false, true],
            detector_values: vec![],
            observable_values: vec![],
            num_measurements: 9,
            num_detectors: 0,
            num_observables: 0,
        };

        let packed = result.packed_measurements();
        // First 8 bits: 1,0,1,1,0,0,1,0 = 0b01001101 = 77
        // 9th bit: 1 = 0b00000001 = 1
        assert_eq!(packed[0], 0b01001101);
        assert_eq!(packed[1], 0b00000001);
    }

    #[test]
    fn test_detector_fire_count() {
        let result = ExecutionResult {
            measurement_record: vec![],
            detector_values: vec![true, false, true, true, false],
            observable_values: vec![],
            num_measurements: 0,
            num_detectors: 5,
            num_observables: 0,
        };

        assert!(result.any_detector_fired());
        assert_eq!(result.detector_fire_count(), 3);
    }
}
