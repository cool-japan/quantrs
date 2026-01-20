//! Implementation of specific quantum error correction codes
//!
//! This module contains the implementation of various quantum error correction codes,
//! including the bit-flip code, phase-flip code, Shor code, and 5-qubit perfect code.

use super::ErrorCorrection;
use crate::error::{Result, SimulatorError};
use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;

/// The 3-qubit bit flip code
///
/// This code can detect and correct single bit flip errors.
/// It encodes a single logical qubit into 3 physical qubits.
#[derive(Debug, Clone, Copy)]
pub struct BitFlipCode;

impl ErrorCorrection for BitFlipCode {
    fn physical_qubits(&self) -> usize {
        3
    }

    fn logical_qubits(&self) -> usize {
        1
    }

    fn distance(&self) -> usize {
        3
    }

    fn encode_circuit(
        &self,
        logical_qubits: &[QubitId],
        ancilla_qubits: &[QubitId],
    ) -> Result<Circuit<16>> {
        // We limit the circuit to 16 qubits maximum
        let mut circuit = Circuit::<16>::new();

        // Check if we have enough qubits
        if logical_qubits.is_empty() {
            return Err(SimulatorError::InvalidInput(
                "BitFlipCode requires at least 1 logical qubit".to_string(),
            ));
        }
        if ancilla_qubits.len() < 2 {
            return Err(SimulatorError::InvalidInput(
                "BitFlipCode requires at least 2 ancilla qubits".to_string(),
            ));
        }

        // Extract qubit IDs
        let q0 = logical_qubits[0];
        let q1 = ancilla_qubits[0];
        let q2 = ancilla_qubits[1];

        // Encode |ψ⟩ -> |ψψψ⟩
        // CNOT from logical qubit to each ancilla qubit
        circuit.cnot(q0, q1).expect(
            "Failed to apply CNOT from logical qubit to first ancilla in BitFlipCode encoding",
        );
        circuit.cnot(q0, q2).expect(
            "Failed to apply CNOT from logical qubit to second ancilla in BitFlipCode encoding",
        );

        Ok(circuit)
    }

    fn decode_circuit(
        &self,
        encoded_qubits: &[QubitId],
        syndrome_qubits: &[QubitId],
    ) -> Result<Circuit<16>> {
        let mut circuit = Circuit::<16>::new();

        // Check if we have enough qubits
        if encoded_qubits.len() < 3 {
            return Err(SimulatorError::InvalidInput(
                "BitFlipCode requires at least 3 encoded qubits".to_string(),
            ));
        }
        if syndrome_qubits.len() < 2 {
            return Err(SimulatorError::InvalidInput(
                "BitFlipCode requires at least 2 syndrome qubits".to_string(),
            ));
        }

        // Extract qubit IDs
        let q0 = encoded_qubits[0];
        let q1 = encoded_qubits[1];
        let q2 = encoded_qubits[2];
        let s0 = syndrome_qubits[0];
        let s1 = syndrome_qubits[1];

        // Syndrome extraction: CNOT from data qubits to syndrome qubits
        circuit
            .cnot(q0, s0)
            .expect("Failed to apply CNOT from q0 to s0 in BitFlipCode syndrome extraction");
        circuit
            .cnot(q1, s0)
            .expect("Failed to apply CNOT from q1 to s0 in BitFlipCode syndrome extraction");
        circuit
            .cnot(q1, s1)
            .expect("Failed to apply CNOT from q1 to s1 in BitFlipCode syndrome extraction");
        circuit
            .cnot(q2, s1)
            .expect("Failed to apply CNOT from q2 to s1 in BitFlipCode syndrome extraction");

        // Apply corrections based on syndrome
        // Syndrome 01 (s1=0, s0=1): bit flip on q0
        circuit
            .x(s1)
            .expect("Failed to apply X to s1 before syndrome 01 correction in BitFlipCode");
        circuit
            .cx(s0, q0)
            .expect("Failed to apply controlled-X for syndrome 01 correction in BitFlipCode");
        circuit
            .x(s1)
            .expect("Failed to apply X to s1 after syndrome 01 correction in BitFlipCode");

        // Syndrome 10 (s1=1, s0=0): bit flip on q1
        circuit
            .x(s0)
            .expect("Failed to apply X to s0 before syndrome 10 correction in BitFlipCode");
        circuit
            .cx(s1, q1)
            .expect("Failed to apply controlled-X for syndrome 10 correction in BitFlipCode");
        circuit
            .x(s0)
            .expect("Failed to apply X to s0 after syndrome 10 correction in BitFlipCode");

        // Syndrome 11 (s1=1, s0=1): bit flip on q2
        circuit
            .cx(s0, q2)
            .expect("Failed to apply first controlled-X for syndrome 11 correction in BitFlipCode");
        circuit.cx(s1, q2).expect(
            "Failed to apply second controlled-X for syndrome 11 correction in BitFlipCode",
        );

        Ok(circuit)
    }
}

/// The 3-qubit phase flip code
///
/// This code can detect and correct single phase flip errors.
/// It encodes a single logical qubit into 3 physical qubits.
#[derive(Debug, Clone, Copy)]
pub struct PhaseFlipCode;

impl ErrorCorrection for PhaseFlipCode {
    fn physical_qubits(&self) -> usize {
        3
    }

    fn logical_qubits(&self) -> usize {
        1
    }

    fn distance(&self) -> usize {
        3
    }

    fn encode_circuit(
        &self,
        logical_qubits: &[QubitId],
        ancilla_qubits: &[QubitId],
    ) -> Result<Circuit<16>> {
        // We limit the circuit to 16 qubits maximum
        let mut circuit = Circuit::<16>::new();

        // Check if we have enough qubits
        if logical_qubits.is_empty() {
            return Err(SimulatorError::InvalidInput(
                "PhaseFlipCode requires at least 1 logical qubit".to_string(),
            ));
        }
        if ancilla_qubits.len() < 2 {
            return Err(SimulatorError::InvalidInput(
                "PhaseFlipCode requires at least 2 ancilla qubits".to_string(),
            ));
        }

        // Extract qubit IDs
        let q0 = logical_qubits[0];
        let q1 = ancilla_qubits[0];
        let q2 = ancilla_qubits[1];

        // Apply Hadamard to all qubits
        circuit
            .h(q0)
            .expect("Failed to apply first Hadamard to q0 in PhaseFlipCode encoding");
        circuit
            .h(q1)
            .expect("Failed to apply first Hadamard to q1 in PhaseFlipCode encoding");
        circuit
            .h(q2)
            .expect("Failed to apply first Hadamard to q2 in PhaseFlipCode encoding");

        // Encode using bit flip code
        circuit
            .cnot(q0, q1)
            .expect("Failed to apply CNOT from q0 to q1 in PhaseFlipCode encoding");
        circuit
            .cnot(q0, q2)
            .expect("Failed to apply CNOT from q0 to q2 in PhaseFlipCode encoding");

        // Apply Hadamard to all qubits again
        circuit
            .h(q0)
            .expect("Failed to apply second Hadamard to q0 in PhaseFlipCode encoding");
        circuit
            .h(q1)
            .expect("Failed to apply second Hadamard to q1 in PhaseFlipCode encoding");
        circuit
            .h(q2)
            .expect("Failed to apply second Hadamard to q2 in PhaseFlipCode encoding");

        Ok(circuit)
    }

    fn decode_circuit(
        &self,
        encoded_qubits: &[QubitId],
        syndrome_qubits: &[QubitId],
    ) -> Result<Circuit<16>> {
        let mut circuit = Circuit::<16>::new();

        // Check if we have enough qubits
        if encoded_qubits.len() < 3 {
            return Err(SimulatorError::InvalidInput(
                "PhaseFlipCode requires at least 3 encoded qubits".to_string(),
            ));
        }
        if syndrome_qubits.len() < 2 {
            return Err(SimulatorError::InvalidInput(
                "PhaseFlipCode requires at least 2 syndrome qubits".to_string(),
            ));
        }

        // Extract qubit IDs
        let q0 = encoded_qubits[0];
        let q1 = encoded_qubits[1];
        let q2 = encoded_qubits[2];
        let s0 = syndrome_qubits[0];
        let s1 = syndrome_qubits[1];

        // Apply Hadamard to all encoded qubits
        circuit
            .h(q0)
            .expect("Failed to apply first Hadamard to q0 in PhaseFlipCode decoding");
        circuit
            .h(q1)
            .expect("Failed to apply first Hadamard to q1 in PhaseFlipCode decoding");
        circuit
            .h(q2)
            .expect("Failed to apply first Hadamard to q2 in PhaseFlipCode decoding");

        // Syndrome extraction: CNOT from data qubits to syndrome qubits
        circuit
            .cnot(q0, s0)
            .expect("Failed to apply CNOT from q0 to s0 in PhaseFlipCode syndrome extraction");
        circuit
            .cnot(q1, s0)
            .expect("Failed to apply CNOT from q1 to s0 in PhaseFlipCode syndrome extraction");
        circuit
            .cnot(q1, s1)
            .expect("Failed to apply CNOT from q1 to s1 in PhaseFlipCode syndrome extraction");
        circuit
            .cnot(q2, s1)
            .expect("Failed to apply CNOT from q2 to s1 in PhaseFlipCode syndrome extraction");

        // Apply corrections based on syndrome in X basis
        // Syndrome 01 (s1=0, s0=1): bit flip on q0
        circuit
            .x(s1)
            .expect("Failed to apply X to s1 before syndrome 01 correction in PhaseFlipCode");
        circuit
            .cx(s0, q0)
            .expect("Failed to apply controlled-X for syndrome 01 correction in PhaseFlipCode");
        circuit
            .x(s1)
            .expect("Failed to apply X to s1 after syndrome 01 correction in PhaseFlipCode");

        // Syndrome 10 (s1=1, s0=0): bit flip on q1
        circuit
            .x(s0)
            .expect("Failed to apply X to s0 before syndrome 10 correction in PhaseFlipCode");
        circuit
            .cx(s1, q1)
            .expect("Failed to apply controlled-X for syndrome 10 correction in PhaseFlipCode");
        circuit
            .x(s0)
            .expect("Failed to apply X to s0 after syndrome 10 correction in PhaseFlipCode");

        // Syndrome 11 (s1=1, s0=1): bit flip on q2
        circuit.cx(s0, q2).expect(
            "Failed to apply first controlled-X for syndrome 11 correction in PhaseFlipCode",
        );
        circuit.cx(s1, q2).expect(
            "Failed to apply second controlled-X for syndrome 11 correction in PhaseFlipCode",
        );

        // Apply Hadamard to all encoded qubits to go back to computational basis
        circuit
            .h(q0)
            .expect("Failed to apply second Hadamard to q0 in PhaseFlipCode decoding");
        circuit
            .h(q1)
            .expect("Failed to apply second Hadamard to q1 in PhaseFlipCode decoding");
        circuit
            .h(q2)
            .expect("Failed to apply second Hadamard to q2 in PhaseFlipCode decoding");

        Ok(circuit)
    }
}

/// The 9-qubit Shor code
///
/// This code can detect and correct arbitrary single-qubit errors
/// (bit flips, phase flips, or both). It encodes a single logical
/// qubit into 9 physical qubits.
#[derive(Debug, Clone, Copy)]
pub struct ShorCode;

impl ErrorCorrection for ShorCode {
    fn physical_qubits(&self) -> usize {
        9
    }

    fn logical_qubits(&self) -> usize {
        1
    }

    fn distance(&self) -> usize {
        3
    }

    fn encode_circuit(
        &self,
        logical_qubits: &[QubitId],
        ancilla_qubits: &[QubitId],
    ) -> Result<Circuit<16>> {
        let mut circuit = Circuit::<16>::new();

        // Check if we have enough qubits
        if logical_qubits.is_empty() {
            return Err(SimulatorError::InvalidInput(
                "ShorCode requires at least 1 logical qubit".to_string(),
            ));
        }
        if ancilla_qubits.len() < 8 {
            return Err(SimulatorError::InvalidInput(
                "ShorCode requires at least 8 ancilla qubits".to_string(),
            ));
        }

        // Extract qubit IDs for easier reading
        let q = logical_qubits[0]; // logical qubit
        let a = &ancilla_qubits[0..8]; // ancilla qubits

        // Step 1: First encode the qubit for phase-flip protection
        // This is done by applying Hadamard and creating a 3-qubit GHZ-like state
        circuit
            .h(q)
            .expect("Failed to apply Hadamard to logical qubit in ShorCode encoding");

        // Create 3 blocks with one qubit each
        circuit
            .cnot(q, a[0])
            .expect("Failed to apply CNOT to create Block 1 in ShorCode encoding"); // Block 1 - first qubit
        circuit
            .cnot(q, a[3])
            .expect("Failed to apply CNOT to create Block 2 in ShorCode encoding"); // Block 2 - first qubit

        // Step 2: Encode each of these 3 qubits against bit-flips
        // using the 3-qubit bit-flip code

        // Encode Block 1 (qubits q, a[0], a[1], a[2])
        circuit
            .cnot(q, a[1])
            .expect("Failed to apply first CNOT for Block 1 bit-flip encoding in ShorCode");
        circuit
            .cnot(q, a[2])
            .expect("Failed to apply second CNOT for Block 1 bit-flip encoding in ShorCode");

        // Encode Block 2 (qubits a[3], a[4], a[5])
        circuit
            .cnot(a[3], a[4])
            .expect("Failed to apply first CNOT for Block 2 bit-flip encoding in ShorCode");
        circuit
            .cnot(a[3], a[5])
            .expect("Failed to apply second CNOT for Block 2 bit-flip encoding in ShorCode");

        // Encode Block 3 (qubits a[6], a[7])
        // CNOT with logical qubit to create the third block
        circuit
            .cnot(q, a[6])
            .expect("Failed to apply CNOT to create Block 3 in ShorCode encoding");
        circuit
            .cnot(a[6], a[7])
            .expect("Failed to apply CNOT for Block 3 bit-flip encoding in ShorCode");

        // At this point, we have encoded our logical |0⟩ as:
        // (|000_000_000⟩ + |111_111_111⟩)/√2 and
        // logical |1⟩ as: (|000_000_000⟩ - |111_111_111⟩)/√2

        // Apply Hadamards again to transform into the final Shor code state
        // For the standard Shor code representation, we would apply Hadamards again
        // to all qubits. For this implementation we'll leave it in the current basis.

        Ok(circuit)
    }

    fn decode_circuit(
        &self,
        encoded_qubits: &[QubitId],
        syndrome_qubits: &[QubitId],
    ) -> Result<Circuit<16>> {
        let mut circuit = Circuit::<16>::new();

        // Check if we have enough qubits
        if encoded_qubits.len() < 9 {
            return Err(SimulatorError::InvalidInput(
                "ShorCode requires at least 9 encoded qubits".to_string(),
            ));
        }
        if syndrome_qubits.len() < 8 {
            return Err(SimulatorError::InvalidInput(
                "ShorCode requires at least 8 syndrome qubits".to_string(),
            ));
        }

        // Extract qubit IDs for more readable code
        let data = encoded_qubits;
        let synd = syndrome_qubits;

        // Step 1: Bit-flip error detection within each group

        // Group 1 (qubits 0,1,2) syndrome detection
        circuit.cnot(data[0], synd[0]).expect(
            "Failed to apply CNOT from data[0] to synd[0] in ShorCode Group 1 syndrome detection",
        );
        circuit.cnot(data[1], synd[0]).expect(
            "Failed to apply CNOT from data[1] to synd[0] in ShorCode Group 1 syndrome detection",
        );
        circuit.cnot(data[1], synd[1]).expect(
            "Failed to apply CNOT from data[1] to synd[1] in ShorCode Group 1 syndrome detection",
        );
        circuit.cnot(data[2], synd[1]).expect(
            "Failed to apply CNOT from data[2] to synd[1] in ShorCode Group 1 syndrome detection",
        );

        // Group 2 (qubits 3,4,5) syndrome detection
        circuit.cnot(data[3], synd[2]).expect(
            "Failed to apply CNOT from data[3] to synd[2] in ShorCode Group 2 syndrome detection",
        );
        circuit.cnot(data[4], synd[2]).expect(
            "Failed to apply CNOT from data[4] to synd[2] in ShorCode Group 2 syndrome detection",
        );
        circuit.cnot(data[4], synd[3]).expect(
            "Failed to apply CNOT from data[4] to synd[3] in ShorCode Group 2 syndrome detection",
        );
        circuit.cnot(data[5], synd[3]).expect(
            "Failed to apply CNOT from data[5] to synd[3] in ShorCode Group 2 syndrome detection",
        );

        // Group 3 (qubits 6,7,8) syndrome detection
        circuit.cnot(data[6], synd[4]).expect(
            "Failed to apply CNOT from data[6] to synd[4] in ShorCode Group 3 syndrome detection",
        );
        circuit.cnot(data[7], synd[4]).expect(
            "Failed to apply CNOT from data[7] to synd[4] in ShorCode Group 3 syndrome detection",
        );
        circuit.cnot(data[7], synd[5]).expect(
            "Failed to apply CNOT from data[7] to synd[5] in ShorCode Group 3 syndrome detection",
        );
        circuit.cnot(data[8], synd[5]).expect(
            "Failed to apply CNOT from data[8] to synd[5] in ShorCode Group 3 syndrome detection",
        );

        // Step 2: Apply bit-flip corrections based on syndromes

        // Group 1 corrections
        // Syndrome 01 (s1=0, s0=1): bit flip on q0
        circuit.x(synd[1]).expect(
            "Failed to apply X to synd[1] before Group 1 syndrome 01 correction in ShorCode",
        );
        circuit
            .cx(synd[0], data[0])
            .expect("Failed to apply controlled-X for Group 1 syndrome 01 correction in ShorCode");
        circuit.x(synd[1]).expect(
            "Failed to apply X to synd[1] after Group 1 syndrome 01 correction in ShorCode",
        );

        // Syndrome 10 (s1=1, s0=0): bit flip on q1
        circuit.x(synd[0]).expect(
            "Failed to apply X to synd[0] before Group 1 syndrome 10 correction in ShorCode",
        );
        circuit
            .cx(synd[1], data[1])
            .expect("Failed to apply controlled-X for Group 1 syndrome 10 correction in ShorCode");
        circuit.x(synd[0]).expect(
            "Failed to apply X to synd[0] after Group 1 syndrome 10 correction in ShorCode",
        );

        // Syndrome 11 (s1=1, s0=1): bit flip on q2
        circuit.cx(synd[0], data[2]).expect(
            "Failed to apply first controlled-X for Group 1 syndrome 11 correction in ShorCode",
        );
        circuit.cx(synd[1], data[2]).expect(
            "Failed to apply second controlled-X for Group 1 syndrome 11 correction in ShorCode",
        );

        // Group 2 corrections
        // Syndrome 01 (s3=0, s2=1): bit flip on q3
        circuit.x(synd[3]).expect(
            "Failed to apply X to synd[3] before Group 2 syndrome 01 correction in ShorCode",
        );
        circuit
            .cx(synd[2], data[3])
            .expect("Failed to apply controlled-X for Group 2 syndrome 01 correction in ShorCode");
        circuit.x(synd[3]).expect(
            "Failed to apply X to synd[3] after Group 2 syndrome 01 correction in ShorCode",
        );

        // Syndrome 10 (s3=1, s2=0): bit flip on q4
        circuit.x(synd[2]).expect(
            "Failed to apply X to synd[2] before Group 2 syndrome 10 correction in ShorCode",
        );
        circuit
            .cx(synd[3], data[4])
            .expect("Failed to apply controlled-X for Group 2 syndrome 10 correction in ShorCode");
        circuit.x(synd[2]).expect(
            "Failed to apply X to synd[2] after Group 2 syndrome 10 correction in ShorCode",
        );

        // Syndrome 11 (s3=1, s2=1): bit flip on q5
        circuit.cx(synd[2], data[5]).expect(
            "Failed to apply first controlled-X for Group 2 syndrome 11 correction in ShorCode",
        );
        circuit.cx(synd[3], data[5]).expect(
            "Failed to apply second controlled-X for Group 2 syndrome 11 correction in ShorCode",
        );

        // Group 3 corrections
        // Syndrome 01 (s5=0, s4=1): bit flip on q6
        circuit.x(synd[5]).expect(
            "Failed to apply X to synd[5] before Group 3 syndrome 01 correction in ShorCode",
        );
        circuit
            .cx(synd[4], data[6])
            .expect("Failed to apply controlled-X for Group 3 syndrome 01 correction in ShorCode");
        circuit.x(synd[5]).expect(
            "Failed to apply X to synd[5] after Group 3 syndrome 01 correction in ShorCode",
        );

        // Syndrome 10 (s5=1, s4=0): bit flip on q7
        circuit.x(synd[4]).expect(
            "Failed to apply X to synd[4] before Group 3 syndrome 10 correction in ShorCode",
        );
        circuit
            .cx(synd[5], data[7])
            .expect("Failed to apply controlled-X for Group 3 syndrome 10 correction in ShorCode");
        circuit.x(synd[4]).expect(
            "Failed to apply X to synd[4] after Group 3 syndrome 10 correction in ShorCode",
        );

        // Syndrome 11 (s5=1, s4=1): bit flip on q8
        circuit.cx(synd[4], data[8]).expect(
            "Failed to apply first controlled-X for Group 3 syndrome 11 correction in ShorCode",
        );
        circuit.cx(synd[5], data[8]).expect(
            "Failed to apply second controlled-X for Group 3 syndrome 11 correction in ShorCode",
        );

        // Step 3: Phase-flip error detection between groups

        // Apply Hadamard gates to convert phase errors to bit errors
        for &q in &[data[0], data[3], data[6]] {
            circuit
                .h(q)
                .expect("Failed to apply Hadamard for phase error detection in ShorCode");
        }

        // Detect phase errors by comparing the first qubit of each group
        circuit.cnot(data[0], synd[6]).expect(
            "Failed to apply CNOT from data[0] to synd[6] in ShorCode phase error detection",
        );
        circuit.cnot(data[3], synd[6]).expect(
            "Failed to apply CNOT from data[3] to synd[6] in ShorCode phase error detection",
        );
        circuit.cnot(data[3], synd[7]).expect(
            "Failed to apply CNOT from data[3] to synd[7] in ShorCode phase error detection",
        );
        circuit.cnot(data[6], synd[7]).expect(
            "Failed to apply CNOT from data[6] to synd[7] in ShorCode phase error detection",
        );

        // Step 4: Apply phase-flip corrections based on syndrome

        // Syndrome 01 (s7=0, s6=1): phase flip on group 1 (qubits 0,1,2)
        circuit
            .x(synd[7])
            .expect("Failed to apply X to synd[7] before phase syndrome 01 correction in ShorCode");
        for &q in &[data[0], data[1], data[2]] {
            circuit.cz(synd[6], q).expect(
                "Failed to apply controlled-Z for Group 1 phase syndrome 01 correction in ShorCode",
            );
        }
        circuit
            .x(synd[7])
            .expect("Failed to apply X to synd[7] after phase syndrome 01 correction in ShorCode");

        // Syndrome 10 (s7=1, s6=0): phase flip on group 2 (qubits 3,4,5)
        circuit
            .x(synd[6])
            .expect("Failed to apply X to synd[6] before phase syndrome 10 correction in ShorCode");
        for &q in &[data[3], data[4], data[5]] {
            circuit.cz(synd[7], q).expect(
                "Failed to apply controlled-Z for Group 2 phase syndrome 10 correction in ShorCode",
            );
        }
        circuit
            .x(synd[6])
            .expect("Failed to apply X to synd[6] after phase syndrome 10 correction in ShorCode");

        // Syndrome 11 (s7=1, s6=1): phase flip on group 3 (qubits 6,7,8)
        for &q in &[data[6], data[7], data[8]] {
            circuit
                .cz(synd[6], q)
                .expect("Failed to apply first controlled-Z for Group 3 phase syndrome 11 correction in ShorCode");
            circuit
                .cz(synd[7], q)
                .expect("Failed to apply second controlled-Z for Group 3 phase syndrome 11 correction in ShorCode");
        }

        // Step 5: Transform back from Hadamard basis
        for &q in &[data[0], data[3], data[6]] {
            circuit.h(q).expect(
                "Failed to apply Hadamard to transform back from phase error basis in ShorCode",
            );
        }

        Ok(circuit)
    }
}

/// The 5-qubit perfect code
///
/// This is the smallest code that can correct an arbitrary single-qubit error.
/// It encodes a single logical qubit into 5 physical qubits.
#[derive(Debug, Clone, Copy)]
pub struct FiveQubitCode;

impl ErrorCorrection for FiveQubitCode {
    fn physical_qubits(&self) -> usize {
        5
    }

    fn logical_qubits(&self) -> usize {
        1
    }

    fn distance(&self) -> usize {
        3
    }

    fn encode_circuit(
        &self,
        logical_qubits: &[QubitId],
        ancilla_qubits: &[QubitId],
    ) -> Result<Circuit<16>> {
        let mut circuit = Circuit::<16>::new();

        // Check if we have enough qubits
        if logical_qubits.is_empty() {
            return Err(SimulatorError::InvalidInput(
                "FiveQubitCode requires at least 1 logical qubit".to_string(),
            ));
        }
        if ancilla_qubits.len() < 4 {
            return Err(SimulatorError::InvalidInput(
                "FiveQubitCode requires at least 4 ancilla qubits".to_string(),
            ));
        }

        // Extract qubit IDs
        let q0 = logical_qubits[0];
        let ancs = ancilla_qubits;

        // The encoding circuit for the 5-qubit perfect code
        // This implements the circuit described in Nielsen & Chuang

        // Initialize all ancilla qubits to |0⟩ (they start in this state by default)

        // Step 1: Apply the initial gates to start creating the superposition
        circuit
            .h(ancs[0])
            .expect("Failed to apply Hadamard to ancs[0] in FiveQubitCode encoding initialization");
        circuit
            .h(ancs[1])
            .expect("Failed to apply Hadamard to ancs[1] in FiveQubitCode encoding initialization");
        circuit
            .h(ancs[2])
            .expect("Failed to apply Hadamard to ancs[2] in FiveQubitCode encoding initialization");
        circuit
            .h(ancs[3])
            .expect("Failed to apply Hadamard to ancs[3] in FiveQubitCode encoding initialization");

        // Step 2: Apply the controlled encoding operations
        // CNOT from data qubit to ancilla qubits
        circuit
            .cnot(q0, ancs[0])
            .expect("Failed to apply CNOT from q0 to ancs[0] in FiveQubitCode encoding");
        circuit
            .cnot(q0, ancs[1])
            .expect("Failed to apply CNOT from q0 to ancs[1] in FiveQubitCode encoding");
        circuit
            .cnot(q0, ancs[2])
            .expect("Failed to apply CNOT from q0 to ancs[2] in FiveQubitCode encoding");
        circuit
            .cnot(q0, ancs[3])
            .expect("Failed to apply CNOT from q0 to ancs[3] in FiveQubitCode encoding");

        // Step 3: Apply the stabilizer operations
        // These specific gates implement the [[5,1,3]] perfect code

        // X stabilizer operations
        circuit
            .h(q0)
            .expect("Failed to apply Hadamard to q0 for X stabilizer in FiveQubitCode encoding");
        circuit.h(ancs[1]).expect(
            "Failed to apply Hadamard to ancs[1] for X stabilizer in FiveQubitCode encoding",
        );
        circuit.h(ancs[3]).expect(
            "Failed to apply Hadamard to ancs[3] for X stabilizer in FiveQubitCode encoding",
        );

        circuit
            .cnot(q0, ancs[0])
            .expect("Failed to apply CNOT for X stabilizer step 1 in FiveQubitCode encoding");
        circuit
            .cnot(ancs[1], ancs[0])
            .expect("Failed to apply CNOT for X stabilizer step 2 in FiveQubitCode encoding");
        circuit
            .cnot(ancs[0], ancs[2])
            .expect("Failed to apply CNOT for X stabilizer step 3 in FiveQubitCode encoding");
        circuit
            .cnot(ancs[2], ancs[3])
            .expect("Failed to apply CNOT for X stabilizer step 4 in FiveQubitCode encoding");

        // Z stabilizer operations
        circuit.cz(q0, ancs[1]).expect(
            "Failed to apply controlled-Z for Z stabilizer step 1 in FiveQubitCode encoding",
        );
        circuit.cz(ancs[0], ancs[2]).expect(
            "Failed to apply controlled-Z for Z stabilizer step 2 in FiveQubitCode encoding",
        );
        circuit.cz(ancs[1], ancs[3]).expect(
            "Failed to apply controlled-Z for Z stabilizer step 3 in FiveQubitCode encoding",
        );

        circuit
            .h(ancs[0])
            .expect("Failed to apply final Hadamard to ancs[0] in FiveQubitCode encoding");
        circuit
            .h(ancs[2])
            .expect("Failed to apply final Hadamard to ancs[2] in FiveQubitCode encoding");

        // This encodes the logical qubit into a 5-qubit entangled state that can
        // detect and correct any single-qubit error

        Ok(circuit)
    }

    fn decode_circuit(
        &self,
        encoded_qubits: &[QubitId],
        syndrome_qubits: &[QubitId],
    ) -> Result<Circuit<16>> {
        let mut circuit = Circuit::<16>::new();

        // Check if we have enough qubits
        if encoded_qubits.len() < 5 {
            return Err(SimulatorError::InvalidInput(
                "FiveQubitCode requires at least 5 encoded qubits".to_string(),
            ));
        }
        if syndrome_qubits.len() < 4 {
            return Err(SimulatorError::InvalidInput(
                "FiveQubitCode requires at least 4 syndrome qubits".to_string(),
            ));
        }

        // Extract qubit IDs
        let data = encoded_qubits;
        let synd = syndrome_qubits;

        // The 5-qubit code uses 4 stabilizer generators to detect errors
        // We'll implement the syndrome extraction circuit that measures these stabilizers

        // Generator 1: XZZXI
        circuit
            .h(synd[0])
            .expect("Failed to apply Hadamard to synd[0] before Generator 1 in FiveQubitCode syndrome extraction");
        circuit
            .cnot(synd[0], data[0])
            .expect("Failed to apply CNOT for Generator 1 XZZXI at position 0 in FiveQubitCode");
        circuit.cz(synd[0], data[1]).expect(
            "Failed to apply controlled-Z for Generator 1 XZZXI at position 1 in FiveQubitCode",
        );
        circuit.cz(synd[0], data[2]).expect(
            "Failed to apply controlled-Z for Generator 1 XZZXI at position 2 in FiveQubitCode",
        );
        circuit
            .cnot(synd[0], data[3])
            .expect("Failed to apply CNOT for Generator 1 XZZXI at position 3 in FiveQubitCode");
        circuit
            .h(synd[0])
            .expect("Failed to apply Hadamard to synd[0] after Generator 1 in FiveQubitCode syndrome extraction");

        // Generator 2: IXZZX
        circuit
            .h(synd[1])
            .expect("Failed to apply Hadamard to synd[1] before Generator 2 in FiveQubitCode syndrome extraction");
        circuit
            .cnot(synd[1], data[1])
            .expect("Failed to apply CNOT for Generator 2 IXZZX at position 1 in FiveQubitCode");
        circuit.cz(synd[1], data[2]).expect(
            "Failed to apply controlled-Z for Generator 2 IXZZX at position 2 in FiveQubitCode",
        );
        circuit.cz(synd[1], data[3]).expect(
            "Failed to apply controlled-Z for Generator 2 IXZZX at position 3 in FiveQubitCode",
        );
        circuit
            .cnot(synd[1], data[4])
            .expect("Failed to apply CNOT for Generator 2 IXZZX at position 4 in FiveQubitCode");
        circuit
            .h(synd[1])
            .expect("Failed to apply Hadamard to synd[1] after Generator 2 in FiveQubitCode syndrome extraction");

        // Generator 3: XIXZZ
        circuit
            .h(synd[2])
            .expect("Failed to apply Hadamard to synd[2] before Generator 3 in FiveQubitCode syndrome extraction");
        circuit
            .cnot(synd[2], data[0])
            .expect("Failed to apply CNOT for Generator 3 XIXZZ at position 0 in FiveQubitCode");
        circuit
            .cnot(synd[2], data[2])
            .expect("Failed to apply CNOT for Generator 3 XIXZZ at position 2 in FiveQubitCode");
        circuit.cz(synd[2], data[3]).expect(
            "Failed to apply controlled-Z for Generator 3 XIXZZ at position 3 in FiveQubitCode",
        );
        circuit.cz(synd[2], data[4]).expect(
            "Failed to apply controlled-Z for Generator 3 XIXZZ at position 4 in FiveQubitCode",
        );
        circuit
            .h(synd[2])
            .expect("Failed to apply Hadamard to synd[2] after Generator 3 in FiveQubitCode syndrome extraction");

        // Generator 4: ZXIXZ
        circuit
            .h(synd[3])
            .expect("Failed to apply Hadamard to synd[3] before Generator 4 in FiveQubitCode syndrome extraction");
        circuit.cz(synd[3], data[0]).expect(
            "Failed to apply controlled-Z for Generator 4 ZXIXZ at position 0 in FiveQubitCode",
        );
        circuit
            .cnot(synd[3], data[1])
            .expect("Failed to apply CNOT for Generator 4 ZXIXZ at position 1 in FiveQubitCode");
        circuit
            .cnot(synd[3], data[3])
            .expect("Failed to apply CNOT for Generator 4 ZXIXZ at position 3 in FiveQubitCode");
        circuit.cz(synd[3], data[4]).expect(
            "Failed to apply controlled-Z for Generator 4 ZXIXZ at position 4 in FiveQubitCode",
        );
        circuit
            .h(synd[3])
            .expect("Failed to apply Hadamard to synd[3] after Generator 4 in FiveQubitCode syndrome extraction");

        // After measuring the syndrome, we would apply the appropriate correction
        // The 5-qubit code has a complex error correction table with 16 possible syndromes
        // We'll implement a simplified version that corrects the most common errors

        // First, we'll correct bit flips (X errors)
        // Syndrome 0001: X error on qubit 0
        let syndrome_0001 = [false, false, false, true];
        self.add_conditional_correction(&mut circuit, synd, syndrome_0001, data[0], 'X')?;

        // Syndrome 0010: X error on qubit 1
        let syndrome_0010 = [false, false, true, false];
        self.add_conditional_correction(&mut circuit, synd, syndrome_0010, data[1], 'X')?;

        // Syndrome 0100: X error on qubit 2
        let syndrome_0100 = [false, true, false, false];
        self.add_conditional_correction(&mut circuit, synd, syndrome_0100, data[2], 'X')?;

        // Syndrome 1000: X error on qubit 3
        let syndrome_1000 = [true, false, false, false];
        self.add_conditional_correction(&mut circuit, synd, syndrome_1000, data[3], 'X')?;

        // Now, we'll correct phase flips (Z errors)
        // Syndrome 0011: Z error on qubit 0
        let syndrome_0011 = [false, false, true, true];
        self.add_conditional_correction(&mut circuit, synd, syndrome_0011, data[0], 'Z')?;

        // Syndrome 0101: Z error on qubit 1
        let syndrome_0101 = [false, true, false, true];
        self.add_conditional_correction(&mut circuit, synd, syndrome_0101, data[1], 'Z')?;

        // Syndrome 1001: Z error on qubit 2
        let syndrome_1001 = [true, false, false, true];
        self.add_conditional_correction(&mut circuit, synd, syndrome_1001, data[2], 'Z')?;

        // Syndrome 1100: Z error on qubit 3
        let syndrome_1100 = [true, true, false, false];
        self.add_conditional_correction(&mut circuit, synd, syndrome_1100, data[3], 'Z')?;

        // And finally, Y errors (both bit and phase flips)
        // Syndrome 0111: Y error on qubit 0
        let syndrome_0111 = [false, true, true, true];
        self.add_conditional_correction(&mut circuit, synd, syndrome_0111, data[0], 'Y')?;

        // Syndrome 1011: Y error on qubit 1
        let syndrome_1011 = [true, false, true, true];
        self.add_conditional_correction(&mut circuit, synd, syndrome_1011, data[1], 'Y')?;

        // Syndrome 1101: Y error on qubit 2
        let syndrome_1101 = [true, true, false, true];
        self.add_conditional_correction(&mut circuit, synd, syndrome_1101, data[2], 'Y')?;

        // Syndrome 1110: Y error on qubit 3
        let syndrome_1110 = [true, true, true, false];
        self.add_conditional_correction(&mut circuit, synd, syndrome_1110, data[3], 'Y')?;

        Ok(circuit)
    }
}

impl FiveQubitCode {
    /// Helper function to add conditionally controlled gates based on syndrome measurement
    fn add_conditional_correction(
        &self,
        circuit: &mut Circuit<16>,
        syndrome_qubits: &[QubitId],
        syndrome: [bool; 4],
        target: QubitId,
        error_type: char,
    ) -> Result<()> {
        // In a real quantum circuit, this would involve classical control
        // For our simulator, we simulate classical control using quantum gates

        // For each syndrome bit, apply X gate to negate it if needed
        for (i, &should_be_one) in syndrome.iter().enumerate() {
            if !should_be_one {
                circuit
                    .x(syndrome_qubits[i])
                    .expect("Failed to apply X gate to negate syndrome bit in FiveQubitCode conditional correction");
            }
        }

        // Apply the correction controlled on all syndrome bits being 1
        // We need to control the correction based on all syndrome bits
        // For more accuracy, we'd use a multi-controlled gate, but for this simulation
        // we'll implement a simplified approach

        // First, combine all syndrome bits into one control qubit
        // We do this by applying a series of controlled-X gates
        for i in 1..syndrome_qubits.len() {
            circuit
                .cx(syndrome_qubits[i], syndrome_qubits[0])
                .expect("Failed to apply controlled-X to combine syndrome bits in FiveQubitCode conditional correction");
        }

        // Now apply the appropriate correction controlled by the first syndrome bit
        match error_type {
            'X' => {
                // Apply X correction (for bit flip)
                circuit.cx(syndrome_qubits[0], target).expect(
                    "Failed to apply controlled-X correction for bit flip in FiveQubitCode",
                );
            }
            'Z' => {
                // Apply Z correction (for phase flip)
                circuit.cz(syndrome_qubits[0], target).expect(
                    "Failed to apply controlled-Z correction for phase flip in FiveQubitCode",
                );
            }
            'Y' => {
                // Apply Y correction (for bit-phase flip)
                // We can implement Y as Z followed by X
                circuit
                    .cz(syndrome_qubits[0], target)
                    .expect("Failed to apply controlled-Z for Y correction in FiveQubitCode");
                circuit
                    .cx(syndrome_qubits[0], target)
                    .expect("Failed to apply controlled-X for Y correction in FiveQubitCode");
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(format!(
                    "Unsupported error type: {error_type}"
                )))
            }
        }

        // Undo the combination of syndrome bits
        for i in 1..syndrome_qubits.len() {
            circuit
                .cx(syndrome_qubits[i], syndrome_qubits[0])
                .expect("Failed to apply controlled-X to undo syndrome bit combination in FiveQubitCode conditional correction");
        }

        // Reset syndrome bits to their original states
        for (i, &should_be_one) in syndrome.iter().enumerate() {
            if !should_be_one {
                circuit
                    .x(syndrome_qubits[i])
                    .expect("Failed to apply X gate to reset syndrome bit in FiveQubitCode conditional correction");
            }
        }

        Ok(())
    }
}
