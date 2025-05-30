//! Memory-efficient quantum state storage using SciRS2
//!
//! This module provides memory-efficient storage for quantum states by leveraging
//! SciRS2's memory management utilities.

use crate::error::{QuantRS2Error, QuantRS2Result};
use num_complex::Complex64;

/// A memory-efficient storage for large quantum state vectors
///
/// This provides memory-efficient storage and operations for quantum states,
/// with support for chunk-based processing of large state vectors.
pub struct EfficientStateVector {
    /// Number of qubits
    num_qubits: usize,
    /// The actual state data
    data: Vec<Complex64>,
}

impl EfficientStateVector {
    /// Create a new efficient state vector for the given number of qubits
    pub fn new(num_qubits: usize) -> QuantRS2Result<Self> {
        let size = 1 << num_qubits;
        if size > 1 << 30 {
            // For very large states (>30 qubits), create large vector directly
            // Note: BufferPool would need different design for thread-safe access
            let mut data = vec![Complex64::new(0.0, 0.0); size];
            data[0] = Complex64::new(1.0, 0.0); // Initialize to |00...0⟩
            Ok(Self { num_qubits, data })
        } else {
            // For smaller states, use regular allocation
            let mut data = vec![Complex64::new(0.0, 0.0); size];
            data[0] = Complex64::new(1.0, 0.0); // Initialize to |00...0⟩
            Ok(Self { num_qubits, data })
        }
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the size of the state vector
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to the state data
    pub fn data(&self) -> &[Complex64] {
        &self.data
    }

    /// Get a mutable reference to the state data
    pub fn data_mut(&mut self) -> &mut [Complex64] {
        &mut self.data
    }

    /// Normalize the state vector
    pub fn normalize(&mut self) -> QuantRS2Result<()> {
        let norm_sqr: f64 = self.data.iter().map(|c| c.norm_sqr()).sum();
        if norm_sqr == 0.0 {
            return Err(QuantRS2Error::InvalidInput(
                "Cannot normalize zero vector".to_string(),
            ));
        }
        let norm = norm_sqr.sqrt();
        for amplitude in &mut self.data {
            *amplitude /= norm;
        }
        Ok(())
    }

    /// Calculate the probability of measuring a specific basis state
    pub fn get_probability(&self, basis_state: usize) -> QuantRS2Result<f64> {
        if basis_state >= self.data.len() {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Basis state {} out of range for {} qubits",
                basis_state, self.num_qubits
            )));
        }
        Ok(self.data[basis_state].norm_sqr())
    }

    /// Apply a function to chunks of the state vector
    ///
    /// This is useful for operations that can be parallelized or when
    /// working with states too large to fit in cache.
    pub fn process_chunks<F>(&mut self, chunk_size: usize, mut f: F) -> QuantRS2Result<()>
    where
        F: FnMut(&mut [Complex64], usize),
    {
        if chunk_size == 0 || chunk_size > self.data.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Invalid chunk size".to_string(),
            ));
        }

        for (chunk_idx, chunk) in self.data.chunks_mut(chunk_size).enumerate() {
            f(chunk, chunk_idx * chunk_size);
        }
        Ok(())
    }
}

/// Memory usage statistics for quantum states
pub struct StateMemoryStats {
    /// Number of complex numbers stored
    pub num_amplitudes: usize,
    /// Memory used in bytes
    pub memory_bytes: usize,
}

impl EfficientStateVector {
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> StateMemoryStats {
        StateMemoryStats {
            num_amplitudes: self.data.len(),
            memory_bytes: self.data.len() * std::mem::size_of::<Complex64>(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efficient_state_vector() {
        let state = EfficientStateVector::new(3).unwrap();
        assert_eq!(state.num_qubits(), 3);
        assert_eq!(state.size(), 8);

        // Check initial state is |000⟩
        assert_eq!(state.data()[0], Complex64::new(1.0, 0.0));
        for i in 1..8 {
            assert_eq!(state.data()[i], Complex64::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_normalization() {
        let mut state = EfficientStateVector::new(2).unwrap();
        state.data_mut()[0] = Complex64::new(1.0, 0.0);
        state.data_mut()[1] = Complex64::new(0.0, 1.0);
        state.data_mut()[2] = Complex64::new(1.0, 0.0);
        state.data_mut()[3] = Complex64::new(0.0, -1.0);

        state.normalize().unwrap();

        let norm_sqr: f64 = state.data().iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sqr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chunk_processing() {
        let mut state = EfficientStateVector::new(3).unwrap();

        // Process in chunks of 2
        state
            .process_chunks(2, |chunk, start_idx| {
                for (i, amp) in chunk.iter_mut().enumerate() {
                    *amp = Complex64::new((start_idx + i) as f64, 0.0);
                }
            })
            .unwrap();

        // Verify the result
        for i in 0..8 {
            assert_eq!(state.data()[i], Complex64::new(i as f64, 0.0));
        }
    }
}
