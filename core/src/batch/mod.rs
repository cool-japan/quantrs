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
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;

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
                    "Batch requires {required_memory} bytes, limit is {limit}"
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
        let (_batch_size, state_size) = states.dim();

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
        let chunk_states = batch
            .states
            .slice(scirs2_core::ndarray::s![start..end, ..])
            .to_owned();

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
        let batch = BatchStateVector::new(10, 3, BatchConfig::default())
            .expect("Failed to create batch state vector");
        assert_eq!(batch.batch_size(), 10);
        assert_eq!(batch.n_qubits, 3);
        assert_eq!(batch.states.ncols(), 8); // 2^3

        // Check initial state is |000>
        for i in 0..10 {
            let state = batch.get_state(i).expect("Failed to get state from batch");
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

        let batch = BatchStateVector::from_states(states, BatchConfig::default())
            .expect("Failed to create batch from states");
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

        let batch = create_batch(states, BatchConfig::default())
            .expect("Failed to create batch from state collection");
        assert_eq!(batch.batch_size(), 3);
        assert_eq!(batch.n_qubits, 1);
    }

    #[test]
    fn test_split_batch() {
        let batch = BatchStateVector::new(10, 2, BatchConfig::default())
            .expect("Failed to create batch for split test");
        let chunks = split_batch(&batch, 3);

        assert_eq!(chunks.len(), 4); // 3, 3, 3, 1
        assert_eq!(chunks[0].batch_size(), 3);
        assert_eq!(chunks[1].batch_size(), 3);
        assert_eq!(chunks[2].batch_size(), 3);
        assert_eq!(chunks[3].batch_size(), 1);
    }

    #[test]
    fn test_merge_batches() {
        let batch1 = BatchStateVector::new(3, 2, BatchConfig::default())
            .expect("Failed to create first batch");
        let batch2 = BatchStateVector::new(2, 2, BatchConfig::default())
            .expect("Failed to create second batch");

        let merged = merge_batches(vec![batch1, batch2], BatchConfig::default())
            .expect("Failed to merge batches");
        assert_eq!(merged.batch_size(), 5);
        assert_eq!(merged.n_qubits, 2);
    }

    // === Comprehensive Batch Operation Tests ===

    #[test]
    fn test_batch_memory_limit_enforcement() {
        let mut config = BatchConfig::default();
        // Set a very small memory limit
        config.memory_limit = Some(100);

        // Try to create a batch that exceeds the limit
        let result = BatchStateVector::new(10, 5, config);
        assert!(result.is_err());

        // Verify error message
        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(msg.contains("bytes") || msg.contains("limit"));
        }
    }

    #[test]
    fn test_batch_state_normalization() {
        let batch = BatchStateVector::new(5, 2, BatchConfig::default())
            .expect("Failed to create batch for normalization test");

        // Check that all states are normalized
        for i in 0..batch.batch_size() {
            let state = batch
                .get_state(i)
                .expect("Failed to get state for normalization check");
            let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "State {} not normalized: {}",
                i,
                norm
            );
        }
    }

    #[test]
    fn test_batch_state_get_set_roundtrip() {
        let mut batch = BatchStateVector::new(3, 2, BatchConfig::default())
            .expect("Failed to create batch for get/set test");

        // Create a custom state
        let custom_state = Array1::from_vec(vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ]);

        // Set and get
        batch
            .set_state(1, &custom_state)
            .expect("Failed to set custom state");
        let retrieved = batch
            .get_state(1)
            .expect("Failed to retrieve state after set");

        // Verify
        for i in 0..4 {
            assert!((retrieved[i] - custom_state[i]).norm() < 1e-10);
        }
    }

    #[test]
    fn test_batch_out_of_bounds_access() {
        let batch = BatchStateVector::new(5, 2, BatchConfig::default())
            .expect("Failed to create batch for bounds test");

        // Get out of bounds
        assert!(batch.get_state(5).is_err());
        assert!(batch.get_state(100).is_err());
    }

    #[test]
    fn test_batch_set_wrong_size_state() {
        let mut batch = BatchStateVector::new(5, 2, BatchConfig::default())
            .expect("Failed to create batch for wrong size test");

        // Try to set state with wrong size
        let wrong_state =
            Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        assert!(batch.set_state(0, &wrong_state).is_err());
    }

    #[test]
    fn test_empty_batch_creation_fails() {
        let result = create_batch(Vec::<Array1<Complex64>>::new(), BatchConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_mismatched_state_sizes() {
        let states = vec![
            Array1::from_vec(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]),
            Array1::from_vec(vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ]),
        ];

        let result = create_batch(states, BatchConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_invalid_state_size() {
        // State size not a power of 2
        let states = Array2::zeros((5, 3));
        let result = BatchStateVector::from_states(states, BatchConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_split_batch_single_element() {
        let batch = BatchStateVector::new(1, 2, BatchConfig::default())
            .expect("Failed to create single element batch");
        let chunks = split_batch(&batch, 10);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].batch_size(), 1);
    }

    #[test]
    fn test_split_batch_exact_division() {
        let batch = BatchStateVector::new(9, 2, BatchConfig::default())
            .expect("Failed to create batch for exact division test");
        let chunks = split_batch(&batch, 3);

        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert_eq!(chunk.batch_size(), 3);
        }
    }

    #[test]
    fn test_merge_batches_empty() {
        let result = merge_batches(Vec::new(), BatchConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_batches_mismatched_qubits() {
        let batch1 = BatchStateVector::new(3, 2, BatchConfig::default())
            .expect("Failed to create first batch with 2 qubits");
        let batch2 = BatchStateVector::new(2, 3, BatchConfig::default())
            .expect("Failed to create second batch with 3 qubits");

        let result = merge_batches(vec![batch1, batch2], BatchConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_config_defaults() {
        let config = BatchConfig::default();
        assert!(config.num_workers.is_none());
        assert_eq!(config.max_batch_size, 1024);
        assert!(config.use_gpu);
        assert!(config.memory_limit.is_none());
        assert!(config.enable_cache);
    }

    #[test]
    fn test_large_batch_creation() {
        // Test with larger batch size
        let batch = BatchStateVector::new(100, 4, BatchConfig::default())
            .expect("Failed to create large batch");
        assert_eq!(batch.batch_size(), 100);
        assert_eq!(batch.n_qubits, 4);
        assert_eq!(batch.states.ncols(), 16); // 2^4
    }

    #[test]
    fn test_batch_state_modification_isolation() {
        let mut batch = BatchStateVector::new(3, 2, BatchConfig::default())
            .expect("Failed to create batch for isolation test");

        // Modify one state
        let modified = Array1::from_vec(vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        batch
            .set_state(1, &modified)
            .expect("Failed to set modified state");

        // Check that other states are unchanged
        let state0 = batch.get_state(0).expect("Failed to get state 0");
        let state2 = batch.get_state(2).expect("Failed to get state 2");

        assert_eq!(state0[0], Complex64::new(1.0, 0.0));
        assert_eq!(state2[0], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_split_merge_roundtrip() {
        let batch = BatchStateVector::new(10, 2, BatchConfig::default())
            .expect("Failed to create batch for roundtrip test");
        let original_states = batch.states.clone();

        // Split and merge
        let chunks = split_batch(&batch, 3);
        let merged = merge_batches(chunks, BatchConfig::default())
            .expect("Failed to merge chunks in roundtrip test");

        // Verify states are preserved
        assert_eq!(merged.batch_size(), 10);
        for i in 0..10 {
            for j in 0..4 {
                assert_eq!(merged.states[[i, j]], original_states[[i, j]]);
            }
        }
    }

    #[test]
    fn test_batch_execution_result_fields() {
        let result = BatchExecutionResult {
            final_states: Array2::zeros((5, 4)),
            execution_time_ms: 100.0,
            gates_applied: 50,
            used_gpu: false,
        };

        assert_eq!(result.execution_time_ms, 100.0);
        assert_eq!(result.gates_applied, 50);
        assert!(!result.used_gpu);
    }

    #[test]
    fn test_batch_measurement_result_fields() {
        use scirs2_core::ndarray::Array2;

        let result = BatchMeasurementResult {
            outcomes: Array2::zeros((5, 10)),
            probabilities: Array2::zeros((5, 10)),
            post_measurement_states: None,
        };

        assert_eq!(result.outcomes.dim(), (5, 10));
        assert_eq!(result.probabilities.dim(), (5, 10));
        assert!(result.post_measurement_states.is_none());
    }
}
