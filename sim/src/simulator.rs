//! Common simulator interface and results for quantum circuit simulations.

use num_complex::Complex64;

/// Result of a quantum circuit simulation
#[derive(Debug, Clone)]
pub struct SimulatorResult<const N: usize> {
    /// State vector amplitudes (complex coefficients)
    pub amplitudes: Vec<Complex64>,
    /// Number of qubits
    pub num_qubits: usize,
}

impl<const N: usize> SimulatorResult<N> {
    /// Create a new simulator result with the given amplitudes
    pub fn new(amplitudes: Vec<Complex64>) -> Self {
        Self {
            amplitudes,
            num_qubits: N,
        }
    }

    /// Get the state vector amplitudes
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }

    /// Get the probabilities for each basis state
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|amp| amp.norm_sqr()).collect()
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// Common trait for all quantum circuit simulators
pub trait Simulator {
    /// Run a quantum circuit and return the simulation result
    fn run<const N: usize>(
        &mut self,
        circuit: &quantrs2_circuit::prelude::Circuit<N>,
    ) -> SimulatorResult<N>;
}
