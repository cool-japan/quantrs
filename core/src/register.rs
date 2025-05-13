use num_complex::Complex64;
use std::marker::PhantomData;

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::qubit::QubitId;

/// A quantum register that holds the state of N qubits
#[derive(Debug, Clone)]
pub struct Register<const N: usize> {
    /// Complex amplitudes for each basis state
    ///
    /// The index corresponds to the integer representation of a basis state.
    /// For example, for 2 qubits, amplitudes[0] = |00⟩, amplitudes[1] = |01⟩,
    /// amplitudes[2] = |10⟩, amplitudes[3] = |11⟩
    amplitudes: Vec<Complex64>,

    /// Marker to enforce the const generic parameter
    _phantom: PhantomData<[(); N]>,
}

impl<const N: usize> Register<N> {
    /// Create a new register with N qubits in the |0...0⟩ state
    pub fn new() -> Self {
        let dim = 1 << N;
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); dim];
        amplitudes[0] = Complex64::new(1.0, 0.0);

        Self {
            amplitudes,
            _phantom: PhantomData,
        }
    }

    /// Create a register with custom initial amplitudes
    ///
    /// # Errors
    ///
    /// Returns an error if the provided amplitudes vector doesn't have
    /// the correct dimension (2^N) or if the vector isn't properly normalized.
    pub fn with_amplitudes(amplitudes: Vec<Complex64>) -> QuantRS2Result<Self> {
        let expected_dim = 1 << N;
        if amplitudes.len() != expected_dim {
            return Err(QuantRS2Error::CircuitValidationFailed(format!(
                "Amplitudes vector has incorrect dimension. Expected {}, got {}",
                expected_dim,
                amplitudes.len()
            )));
        }

        // Check if the state is properly normalized (within a small epsilon)
        let norm_squared: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum();

        if (norm_squared - 1.0).abs() > 1e-10 {
            return Err(QuantRS2Error::CircuitValidationFailed(format!(
                "Amplitudes vector is not properly normalized. Norm^2 = {}",
                norm_squared
            )));
        }

        Ok(Self {
            amplitudes,
            _phantom: PhantomData,
        })
    }

    /// Get the number of qubits in this register
    #[inline]
    pub const fn num_qubits(&self) -> usize {
        N
    }

    /// Get the dimension of the state space (2^N)
    #[inline]
    pub const fn dimension(&self) -> usize {
        1 << N
    }

    /// Get access to the raw amplitudes vector
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }

    /// Get mutable access to the raw amplitudes vector
    pub fn amplitudes_mut(&mut self) -> &mut [Complex64] {
        &mut self.amplitudes
    }

    /// Get the amplitude for a specific basis state
    ///
    /// The bits parameter must be a slice of length N, where each element
    /// is either 0 or 1 representing the computational basis state.
    ///
    /// # Errors
    ///
    /// Returns an error if the bits slice has incorrect length or contains
    /// values other than 0 or 1.
    pub fn amplitude(&self, bits: &[u8]) -> QuantRS2Result<Complex64> {
        if bits.len() != N {
            return Err(QuantRS2Error::CircuitValidationFailed(format!(
                "Bits slice has incorrect length. Expected {}, got {}",
                N,
                bits.len()
            )));
        }

        for &bit in bits {
            if bit > 1 {
                return Err(QuantRS2Error::CircuitValidationFailed(format!(
                    "Invalid bit value {}. Must be 0 or 1",
                    bit
                )));
            }
        }

        let index = bits
            .iter()
            .fold(0usize, |acc, &bit| (acc << 1) | bit as usize);

        Ok(self.amplitudes[index])
    }

    /// Calculate the probability of measuring a specific basis state
    ///
    /// The bits parameter must be a slice of length N, where each element
    /// is either 0 or 1 representing the computational basis state.
    ///
    /// # Errors
    ///
    /// Returns an error if the bits slice has incorrect length or contains
    /// values other than 0 or 1.
    pub fn probability(&self, bits: &[u8]) -> QuantRS2Result<f64> {
        let amplitude = self.amplitude(bits)?;
        Ok(amplitude.norm_sqr())
    }

    /// Calculate the probabilities of measuring each basis state
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Calculate the expectation value of a single-qubit Pauli operator
    pub fn expectation_z(&self, qubit: impl Into<QubitId>) -> QuantRS2Result<f64> {
        let qubit_id = qubit.into();
        let q_idx = qubit_id.id() as usize;

        if q_idx >= N {
            return Err(QuantRS2Error::InvalidQubitId(qubit_id.id()));
        }

        let dim = 1 << N;
        let mut result = 0.0;

        for i in 0..dim {
            // Check if the qubit is 0 or 1 in this basis state
            let bit_val = (i >> q_idx) & 1;

            // For Z measurement, +1 if bit is 0, -1 if bit is 1
            let z_val = if bit_val == 0 { 1.0 } else { -1.0 };

            // Add contribution to expectation value
            result += z_val * self.amplitudes[i].norm_sqr();
        }

        Ok(result)
    }
}

impl<const N: usize> Default for Register<N> {
    fn default() -> Self {
        Self::new()
    }
}
