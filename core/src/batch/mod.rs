//! Batch operations for quantum circuits using SciRS2 parallel algorithms
//!
//! This module provides efficient batch processing for quantum operations,
//! leveraging SciRS2's parallel computing capabilities for performance.

pub mod execution;
pub mod measurement;
pub mod operations;
pub mod optimization;

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex64;
use std::sync::Arc;

/// Configuration for batch operations
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of parallel workers
    pub num_workers: Option<usize>,
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    /// Whether to use GPU acceleration if available
    pub use_gpu: bool,
    /// Memory limit in bytes
    pub memory_limit: Option<usize>,
    /// Enable cache for repeated operations
    pub enable_cache: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            num_workers: None, // Use system default
            max_batch_size: 1024,
            use_gpu: true,
            memory_limit: None,
            enable_cache: true,
        }
    }
}

/// Batch of quantum states for parallel processing
#[derive(Clone)]
pub struct BatchStateVector {
    /// The batch of state vectors (batch_size, 2^n_qubits)
    pub states: Array2<Complex64>,
    /// Number of qubits
    pub n_qubits: usize,
    /// Batch configuration
    pub config: BatchConfig,
}

impl BatchStateVector {
    /// Create a new batch of quantum states
    pub fn new(batch_size: usize, n_qubits: usize, config: BatchConfig) -> QuantRS2Result<Self> {
        let state_size = 1 << n_qubits;

        // Check memory constraints
        if let Some(limit) = config.memory_limit {
            let required_memory = batch_size * state_size * std::mem::size_of::<Complex64>();
            if required_memory > limit {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Batch requires {} bytes, limit is {}",
                    required_memory, limit
                )));
            }
        }

        // Initialize all states to |0...0>
        let mut states = Array2::zeros((batch_size, state_size));
        for i in 0..batch_size {
            states[[i, 0]] = Complex64::new(1.0, 0.0);
        }

        Ok(Self {
            states,
            n_qubits,
            config,
        })
    }

    /// Create from existing state vectors
    pub fn from_states(states: Array2<Complex64>, config: BatchConfig) -> QuantRS2Result<Self> {
        let (batch_size, state_size) = states.dim();

        // Determine number of qubits
        let n_qubits = (state_size as f64).log2().round() as usize;
        if 1 << n_qubits != state_size {
            return Err(QuantRS2Error::InvalidInput(
                "State size must be a power of 2".to_string(),
            ));
        }

        Ok(Self {
            states,
            n_qubits,
            config,
        })
    }

    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.states.nrows()
    }

    /// Get a specific state from the batch
    pub fn get_state(&self, index: usize) -> QuantRS2Result<Array1<Complex64>> {
        if index >= self.batch_size() {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Index {} out of bounds for batch size {}",
                index,
                self.batch_size()
            )));
        }

        Ok(self.states.row(index).to_owned())
    }

    /// Set a specific state in the batch
    pub fn set_state(&mut self, index: usize, state: &Array1<Complex64>) -> QuantRS2Result<()> {
        if index >= self.batch_size() {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Index {} out of bounds for batch size {}",
                index,
                self.batch_size()
            )));
        }

        if state.len() != self.states.ncols() {
            return Err(QuantRS2Error::InvalidInput(format!(
                "State size {} doesn't match expected {}",
                state.len(),
                self.states.ncols()
            )));
        }

        self.states.row_mut(index).assign(state);
        Ok(())
    }
}

/// Batch circuit execution result
#[derive(Debug, Clone)]
pub struct BatchExecutionResult {
    /// Final state vectors
    pub final_states: Array2<Complex64>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Number of gates applied
    pub gates_applied: usize,
    /// Whether GPU was used
    pub used_gpu: bool,
}

/// Batch measurement result
#[derive(Debug, Clone)]
pub struct BatchMeasurementResult {
    /// Measurement outcomes for each state in the batch
    /// Shape: (batch_size, num_measurements)
    pub outcomes: Array2<u8>,
    /// Probabilities for each outcome
    /// Shape: (batch_size, num_measurements)
    pub probabilities: Array2<f64>,
    /// Post-measurement states (if requested)
    pub post_measurement_states: Option<Array2<Complex64>>,
}

/// Trait for batch-optimized gates
pub trait BatchGateOp: GateOp {
    /// Apply this gate to a batch of states
    fn apply_batch(
        &self,
        batch: &mut BatchStateVector,
        target_qubits: &[QubitId],
    ) -> QuantRS2Result<()>;

    /// Check if this gate has batch optimization
    fn has_batch_optimization(&self) -> bool {
        true
    }
}

/// Helper to create batches from a collection of states
pub fn create_batch<I>(states: I, config: BatchConfig) -> QuantRS2Result<BatchStateVector>
where
    I: IntoIterator<Item = Array1<Complex64>>,
{
    let states_vec: Vec<_> = states.into_iter().collect();
    if states_vec.is_empty() {
        return Err(QuantRS2Error::InvalidInput(
            "Cannot create empty batch".to_string(),
        ));
    }

    let state_size = states_vec[0].len();
    let batch_size = states_vec.len();

    // Validate all states have same size
    for (i, state) in states_vec.iter().enumerate() {
        if state.len() != state_size {
            return Err(QuantRS2Error::InvalidInput(format!(
                "State {} has size {}, expected {}",
                i,
                state.len(),
                state_size
            )));
        }
    }

    // Create 2D array
    let mut batch_array = Array2::zeros((batch_size, state_size));
    for (i, state) in states_vec.iter().enumerate() {
        batch_array.row_mut(i).assign(state);
    }

    BatchStateVector::from_states(batch_array, config)
}

/// Helper to split a large batch into smaller chunks
pub fn split_batch(batch: &BatchStateVector, chunk_size: usize) -> Vec<BatchStateVector> {
    let mut chunks = Vec::new();
    let batch_size = batch.batch_size();

    for start in (0..batch_size).step_by(chunk_size) {
        let end = (start + chunk_size).min(batch_size);
        let chunk_states = batch.states.slice(ndarray::s![start..end, ..]).to_owned();

        if let Ok(chunk) = BatchStateVector::from_states(chunk_states, batch.config.clone()) {
            chunks.push(chunk);
        }
    }

    chunks
}

/// Merge multiple batches into one
pub fn merge_batches(
    batches: Vec<BatchStateVector>,
    config: BatchConfig,
) -> QuantRS2Result<BatchStateVector> {
    if batches.is_empty() {
        return Err(QuantRS2Error::InvalidInput(
            "Cannot merge empty batches".to_string(),
        ));
    }

    // Validate all batches have same n_qubits
    let n_qubits = batches[0].n_qubits;
    for (i, batch) in batches.iter().enumerate() {
        if batch.n_qubits != n_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Batch {} has {} qubits, expected {}",
                i, batch.n_qubits, n_qubits
            )));
        }
    }

    // Concatenate states
    let mut all_states = Vec::new();
    for batch in batches {
        for i in 0..batch.batch_size() {
            all_states.push(batch.states.row(i).to_owned());
        }
    }

    create_batch(all_states, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_creation() {
        let batch = BatchStateVector::new(10, 3, BatchConfig::default()).unwrap();
        assert_eq!(batch.batch_size(), 10);
        assert_eq!(batch.n_qubits, 3);
        assert_eq!(batch.states.ncols(), 8); // 2^3

        // Check initial state is |000>
        for i in 0..10 {
            let state = batch.get_state(i).unwrap();
            assert_eq!(state[0], Complex64::new(1.0, 0.0));
            for j in 1..8 {
                assert_eq!(state[j], Complex64::new(0.0, 0.0));
            }
        }
    }

    #[test]
    fn test_batch_from_states() {
        let mut states = Array2::zeros((5, 4));
        for i in 0..5 {
            states[[i, i % 4]] = Complex64::new(1.0, 0.0);
        }

        let batch = BatchStateVector::from_states(states, BatchConfig::default()).unwrap();
        assert_eq!(batch.batch_size(), 5);
        assert_eq!(batch.n_qubits, 2); // 2^2 = 4
    }

    #[test]
    fn test_create_batch_helper() {
        let states = vec![
            Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]),
            Array1::from_vec(vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]),
            Array1::from_vec(vec![Complex64::new(0.707, 0.0), Complex64::new(0.707, 0.0)]),
        ];

        let batch = create_batch(states, BatchConfig::default()).unwrap();
        assert_eq!(batch.batch_size(), 3);
        assert_eq!(batch.n_qubits, 1);
    }

    #[test]
    fn test_split_batch() {
        let batch = BatchStateVector::new(10, 2, BatchConfig::default()).unwrap();
        let chunks = split_batch(&batch, 3);

        assert_eq!(chunks.len(), 4); // 3, 3, 3, 1
        assert_eq!(chunks[0].batch_size(), 3);
        assert_eq!(chunks[1].batch_size(), 3);
        assert_eq!(chunks[2].batch_size(), 3);
        assert_eq!(chunks[3].batch_size(), 1);
    }

    #[test]
    fn test_merge_batches() {
        let batch1 = BatchStateVector::new(3, 2, BatchConfig::default()).unwrap();
        let batch2 = BatchStateVector::new(2, 2, BatchConfig::default()).unwrap();

        let merged = merge_batches(vec![batch1, batch2], BatchConfig::default()).unwrap();
        assert_eq!(merged.batch_size(), 5);
        assert_eq!(merged.n_qubits, 2);
    }
}
