//! Quantum Error Correction Module
//!
//! This module provides quantum error correction capabilities for protecting
//! quantum information against noise and decoherence.
//!
//! # Overview
//!
//! Quantum error correction is essential for creating fault-tolerant quantum computers.
//! It allows us to encode quantum information in a way that we can detect and correct
//! errors that may occur during computation.
//!
//! This module implements various quantum error correction codes:
//!
//! * **Bit Flip Code**: Protects against X (bit flip) errors
//! * **Phase Flip Code**: Protects against Z (phase flip) errors
//! * **Shor Code**: Protects against arbitrary single-qubit errors
//! * **5-Qubit Perfect Code**: The smallest code that can correct arbitrary single-qubit errors
//!
//! # Usage
//!
//! To use quantum error correction in your quantum circuits:
//!
//! 1. Create an error correction code object
//! 2. Use the object to generate encoding and decoding circuits
//! 3. Incorporate these circuits into your quantum program
//!
//! ```rust,no_run
//! use quantrs2_circuit::builder::Circuit;
//! use quantrs2_core::qubit::QubitId;
//! use quantrs2_sim::error_correction::{BitFlipCode, ErrorCorrection};
//! use quantrs2_sim::statevector::StateVectorSimulator;
//!
//! // Create a bit flip code object
//! let code = BitFlipCode;
//!
//! // Define qubits for encoding
//! let logical_qubits = vec![QubitId::new(0)];
//! let ancilla_qubits = vec![QubitId::new(1), QubitId::new(2)];
//!
//! // Generate encoding circuit
//! let encode_circuit = code.encode_circuit(&logical_qubits, &ancilla_qubits);
//!
//! // Define qubits for syndrome extraction and correction
//! let encoded_qubits = vec![QubitId::new(0), QubitId::new(1), QubitId::new(2)];
//! let syndrome_qubits = vec![QubitId::new(3), QubitId::new(4)];
//!
//! // Generate correction circuit
//! let correction_circuit = code.decode_circuit(&encoded_qubits, &syndrome_qubits);
//!
//! // Create a main circuit and add the encoding and correction operations
//! let mut circuit = Circuit::<5>::new();
//! // ... add your operations here ...
//! ```

use crate::error::{Result, SimulatorError};
use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;

mod codes;

pub use codes::*;

/// Trait for quantum error correction codes
pub trait ErrorCorrection {
    /// Get the number of physical qubits required
    fn physical_qubits(&self) -> usize;

    /// Get the number of logical qubits encoded
    fn logical_qubits(&self) -> usize;

    /// Get the distance of the code (minimum number of errors it can detect)
    fn distance(&self) -> usize;

    /// Create a circuit to encode logical qubits into the error correction code
    ///
    /// # Arguments
    ///
    /// * `logical_qubits` - The qubits containing the logical information to encode
    /// * `ancilla_qubits` - Additional qubits used for the encoding
    ///
    /// # Returns
    ///
    /// A Result containing the circuit with encoding operations, or an error if insufficient qubits
    fn encode_circuit(
        &self,
        logical_qubits: &[QubitId],
        ancilla_qubits: &[QubitId],
    ) -> Result<Circuit<16>>;

    /// Create a circuit to decode and correct errors
    ///
    /// # Arguments
    ///
    /// * `encoded_qubits` - The qubits that contain the encoded information
    /// * `syndrome_qubits` - Additional qubits used for syndrome extraction and error correction
    ///
    /// # Returns
    ///
    /// A Result containing the circuit with error detection and correction operations, or an error if insufficient qubits
    fn decode_circuit(
        &self,
        encoded_qubits: &[QubitId],
        syndrome_qubits: &[QubitId],
    ) -> Result<Circuit<16>>;
}

/// Utility functions for error correction
pub mod utils {
    use super::*;
    use quantrs2_circuit::builder::Circuit;

    /// Creates a complete error-corrected circuit including encoding, noise, and correction
    ///
    /// # Arguments
    ///
    /// * `initial_circuit` - The initial circuit containing the quantum state to protect
    /// * `code` - The error correction code to use
    /// * `logical_qubits` - The qubits containing the logical information
    /// * `ancilla_qubits` - Additional qubits used for encoding
    /// * `syndrome_qubits` - Qubits used for syndrome extraction and correction
    ///
    /// # Returns
    ///
    /// A Result containing the complete circuit with error correction
    pub fn create_error_corrected_circuit<T: ErrorCorrection, const N: usize>(
        initial_circuit: &Circuit<N>,
        code: &T,
        logical_qubits: &[QubitId],
        ancilla_qubits: &[QubitId],
        syndrome_qubits: &[QubitId],
    ) -> Result<Circuit<N>> {
        let mut result = Circuit::<N>::new();

        // Copy gates from initial circuit
        for op in initial_circuit.gates() {
            if op.qubits().is_empty() {
                continue;
            }

            if op.name() == "H" && op.qubits().len() >= 1 {
                let _ = result.h(op.qubits()[0]);
            } else if op.name() == "X" && op.qubits().len() >= 1 {
                let _ = result.x(op.qubits()[0]);
            } else if op.name() == "Y" && op.qubits().len() >= 1 {
                let _ = result.y(op.qubits()[0]);
            } else if op.name() == "Z" && op.qubits().len() >= 1 {
                let _ = result.z(op.qubits()[0]);
            } else if op.name() == "S" && op.qubits().len() >= 1 {
                let _ = result.s(op.qubits()[0]);
            } else if op.name() == "T" && op.qubits().len() >= 1 {
                let _ = result.t(op.qubits()[0]);
            } else if op.name() == "CNOT" && op.qubits().len() >= 2 {
                let _ = result.cnot(op.qubits()[0], op.qubits()[1]);
            } else if op.name() == "CZ" && op.qubits().len() >= 2 {
                let _ = result.cz(op.qubits()[0], op.qubits()[1]);
            } else if op.name() == "CY" && op.qubits().len() >= 2 {
                let _ = result.cy(op.qubits()[0], op.qubits()[1]);
            } else if op.name() == "SWAP" && op.qubits().len() >= 2 {
                let _ = result.swap(op.qubits()[0], op.qubits()[1]);
            }
        }

        // Copy gates from encoding circuit
        let encoder = code.encode_circuit(logical_qubits, ancilla_qubits)?;
        for op in encoder.gates() {
            if op.qubits().is_empty() {
                continue;
            }

            if op.name() == "H" && op.qubits().len() >= 1 {
                let _ = result.h(op.qubits()[0]);
            } else if op.name() == "X" && op.qubits().len() >= 1 {
                let _ = result.x(op.qubits()[0]);
            } else if op.name() == "Y" && op.qubits().len() >= 1 {
                let _ = result.y(op.qubits()[0]);
            } else if op.name() == "Z" && op.qubits().len() >= 1 {
                let _ = result.z(op.qubits()[0]);
            } else if op.name() == "CNOT" && op.qubits().len() >= 2 {
                let _ = result.cnot(op.qubits()[0], op.qubits()[1]);
            } else if op.name() == "CZ" && op.qubits().len() >= 2 {
                let _ = result.cz(op.qubits()[0], op.qubits()[1]);
            }
        }

        // Compute encoded qubits (logical + ancilla)
        let mut encoded_qubits = logical_qubits.to_vec();
        encoded_qubits.extend_from_slice(ancilla_qubits);

        // Copy gates from correction circuit
        let correction = code.decode_circuit(&encoded_qubits, syndrome_qubits)?;
        for op in correction.gates() {
            if op.qubits().is_empty() {
                continue;
            }

            if op.name() == "H" && op.qubits().len() >= 1 {
                let _ = result.h(op.qubits()[0]);
            } else if op.name() == "X" && op.qubits().len() >= 1 {
                let _ = result.x(op.qubits()[0]);
            } else if op.name() == "Y" && op.qubits().len() >= 1 {
                let _ = result.y(op.qubits()[0]);
            } else if op.name() == "Z" && op.qubits().len() >= 1 {
                let _ = result.z(op.qubits()[0]);
            } else if op.name() == "CNOT" && op.qubits().len() >= 2 {
                let _ = result.cnot(op.qubits()[0], op.qubits()[1]);
            } else if op.name() == "CZ" && op.qubits().len() >= 2 {
                let _ = result.cz(op.qubits()[0], op.qubits()[1]);
            }
        }

        Ok(result)
    }

    /// Analyzes the quality of error correction by comparing states before and after correction
    ///
    /// # Arguments
    ///
    /// * `ideal_state` - The amplitudes of the ideal (noise-free) state
    /// * `noisy_state` - The amplitudes of the state with noise
    /// * `corrected_state` - The amplitudes of the state after error correction
    ///
    /// # Returns
    ///
    /// A Result containing a tuple with (fidelity before correction, fidelity after correction)
    pub fn analyze_correction_quality(
        ideal_state: &[scirs2_core::Complex64],
        noisy_state: &[scirs2_core::Complex64],
        corrected_state: &[scirs2_core::Complex64],
    ) -> Result<(f64, f64)> {
        let fidelity_before = calculate_fidelity(ideal_state, noisy_state)?;
        let fidelity_after = calculate_fidelity(ideal_state, corrected_state)?;

        Ok((fidelity_before, fidelity_after))
    }

    /// Calculates the fidelity between two quantum states
    ///
    /// Fidelity measures how close two quantum states are to each other.
    /// A value of 1.0 means the states are identical.
    ///
    /// # Arguments
    ///
    /// * `state1` - The first state's amplitudes
    /// * `state2` - The second state's amplitudes
    ///
    /// # Returns
    ///
    /// The fidelity between the states (0.0 to 1.0)
    pub fn calculate_fidelity(
        state1: &[scirs2_core::Complex64],
        state2: &[scirs2_core::Complex64],
    ) -> Result<f64> {
        use scirs2_core::Complex64;

        if state1.len() != state2.len() {
            return Err(SimulatorError::DimensionMismatch(format!(
                "States have different dimensions: {} vs {}",
                state1.len(),
                state2.len()
            )));
        }

        // Calculate inner product
        let mut inner_product = Complex64::new(0.0, 0.0);
        for (a1, a2) in state1.iter().zip(state2.iter()) {
            inner_product += a1.conj() * a2;
        }

        // Fidelity is the square of the absolute value of the inner product
        Ok(inner_product.norm_sqr())
    }
}
