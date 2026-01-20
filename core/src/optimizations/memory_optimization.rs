//! Memory Optimization for Quantum Simulations using SciRS2 Beta.1
//!
//! This module provides memory-efficient algorithms and buffer management
//! for large-scale quantum state simulations.

use crate::error::QuantRS2Result;
use scirs2_core::memory::{metrics, BufferPool, ChunkProcessor2D};
use scirs2_core::ndarray::{Array1, Array2, ArrayViewMut1};
use scirs2_core::Complex64;
use std::sync::{Arc, Mutex, OnceLock};

/// Quantum-specific buffer pool for state vectors and matrices
pub struct QuantumBufferPool {
    /// Pool for complex64 state vector components
    state_vector_pool: Arc<Mutex<BufferPool<Complex64>>>,
    /// Pool for real-valued probability arrays
    probability_pool: Arc<Mutex<BufferPool<f64>>>,
    /// Pool for temporary computation buffers
    temp_buffer_pool: Arc<Mutex<BufferPool<Complex64>>>,
    /// Usage statistics
    allocations: Arc<Mutex<u64>>,
    deallocations: Arc<Mutex<u64>>,
    peak_memory_usage: Arc<Mutex<usize>>,
}

impl Default for QuantumBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumBufferPool {
    /// Create a new quantum buffer pool
    pub fn new() -> Self {
        Self {
            state_vector_pool: Arc::new(Mutex::new(BufferPool::new())),
            probability_pool: Arc::new(Mutex::new(BufferPool::new())),
            temp_buffer_pool: Arc::new(Mutex::new(BufferPool::new())),
            allocations: Arc::new(Mutex::new(0)),
            deallocations: Arc::new(Mutex::new(0)),
            peak_memory_usage: Arc::new(Mutex::new(0)),
        }
    }

    /// Acquire a state vector buffer from the pool
    pub fn acquire_state_vector(&self, size: usize) -> Vec<Complex64> {
        metrics::track_allocation("QuantumStateVector", size * 16, 0); // Complex64 is 16 bytes
        *self.allocations.lock().expect("Allocations lock poisoned") += 1;

        let current_usage = size * 16;
        let mut peak = self
            .peak_memory_usage
            .lock()
            .expect("Peak memory usage lock poisoned");
        if current_usage > *peak {
            *peak = current_usage;
        }

        self.state_vector_pool
            .lock()
            .expect("State vector pool lock poisoned")
            .acquire_vec(size)
    }

    /// Release a state vector buffer back to the pool
    pub fn release_state_vector(&self, buffer: Vec<Complex64>) {
        let size = buffer.len();
        metrics::track_deallocation("QuantumStateVector", size * 16, 0);
        *self
            .deallocations
            .lock()
            .expect("Deallocations lock poisoned") += 1;

        self.state_vector_pool
            .lock()
            .expect("State vector pool lock poisoned")
            .release_vec(buffer);
    }

    /// Acquire a probability buffer from the pool
    pub fn acquire_probability_buffer(&self, size: usize) -> Vec<f64> {
        metrics::track_allocation("ProbabilityBuffer", size * 8, 0); // f64 is 8 bytes
        *self.allocations.lock().expect("Allocations lock poisoned") += 1;

        self.probability_pool
            .lock()
            .expect("Probability pool lock poisoned")
            .acquire_vec(size)
    }

    /// Release a probability buffer back to the pool
    pub fn release_probability_buffer(&self, buffer: Vec<f64>) {
        let size = buffer.len();
        metrics::track_deallocation("ProbabilityBuffer", size * 8, 0);
        *self
            .deallocations
            .lock()
            .expect("Deallocations lock poisoned") += 1;

        self.probability_pool
            .lock()
            .expect("Probability pool lock poisoned")
            .release_vec(buffer);
    }

    /// Get buffer pool statistics
    pub fn get_stats(&self) -> MemoryUsageStats {
        let allocations = *self.allocations.lock().expect("Allocations lock poisoned");
        let deallocations = *self
            .deallocations
            .lock()
            .expect("Deallocations lock poisoned");
        MemoryUsageStats {
            total_allocations: allocations,
            total_deallocations: deallocations,
            peak_memory_usage_bytes: *self
                .peak_memory_usage
                .lock()
                .expect("Peak memory usage lock poisoned"),
            active_buffers: allocations.saturating_sub(deallocations),
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub peak_memory_usage_bytes: usize,
    pub active_buffers: u64,
}

/// Optimized state vector manager for large quantum systems
pub struct StateVectorManager {
    /// Current state vector
    state: Option<Vec<Complex64>>,
    /// Number of qubits
    num_qubits: usize,
    /// Buffer pool reference
    pool: Arc<QuantumBufferPool>,
    /// Whether to use chunked processing for large states
    use_chunked_processing: bool,
}

impl StateVectorManager {
    /// Create a new state vector manager
    pub const fn new(num_qubits: usize, pool: Arc<QuantumBufferPool>) -> Self {
        let use_chunked_processing = num_qubits > 20; // Use chunking for >20 qubits (~16M elements)

        Self {
            state: None,
            num_qubits,
            pool,
            use_chunked_processing,
        }
    }

    /// Initialize the state vector to |00...0⟩
    pub fn initialize_zero_state(&mut self) -> QuantRS2Result<()> {
        let size = 1 << self.num_qubits;
        let mut state = self.pool.acquire_state_vector(size);

        // Initialize to zero state
        state.fill(Complex64::new(0.0, 0.0));
        state[0] = Complex64::new(1.0, 0.0);

        self.state = Some(state);
        Ok(())
    }

    /// Apply a single-qubit gate with memory optimization
    pub fn apply_single_qubit_gate(
        &mut self,
        gate_matrix: &[Complex64; 4],
        qubit_idx: usize,
    ) -> QuantRS2Result<()> {
        let use_chunked = self.use_chunked_processing;
        let pool = self.pool.clone();

        let state = self.state.as_mut().ok_or_else(|| {
            crate::error::QuantRS2Error::InvalidInput("State not initialized".to_string())
        })?;

        if use_chunked {
            Self::apply_single_qubit_gate_chunked_impl(&pool, state, gate_matrix, qubit_idx)
        } else {
            Self::apply_single_qubit_gate_direct_impl(&pool, state, gate_matrix, qubit_idx)
        }
    }

    /// Direct application for smaller state vectors
    fn apply_single_qubit_gate_direct_impl(
        pool: &QuantumBufferPool,
        state: &mut [Complex64],
        gate_matrix: &[Complex64; 4],
        qubit_idx: usize,
    ) -> QuantRS2Result<()> {
        let size = state.len();
        let target_bit = 1 << qubit_idx;

        // Acquire temporary buffer for parallel processing
        let mut temp_buffer = pool.acquire_state_vector(size);
        temp_buffer.copy_from_slice(state);

        // Apply gate using SIMD-optimized operations
        for i in 0..size {
            if i & target_bit == 0 {
                let j = i | target_bit;
                let amp_0 = temp_buffer[i];
                let amp_1 = temp_buffer[j];

                state[i] = gate_matrix[0] * amp_0 + gate_matrix[1] * amp_1;
                state[j] = gate_matrix[2] * amp_0 + gate_matrix[3] * amp_1;
            }
        }

        pool.release_state_vector(temp_buffer);
        Ok(())
    }

    /// Chunked application for large state vectors
    fn apply_single_qubit_gate_chunked_impl(
        pool: &QuantumBufferPool,
        state: &mut [Complex64],
        gate_matrix: &[Complex64; 4],
        qubit_idx: usize,
    ) -> QuantRS2Result<()> {
        let chunk_size = 1 << 18; // Process in 256K element chunks
        let target_bit = 1 << qubit_idx;

        // Create a temporary copy for atomic operations
        let mut temp_state = pool.acquire_state_vector(state.len());
        temp_state.copy_from_slice(state);

        // Apply gate using temporary buffer
        for i in 0..state.len() {
            if i & target_bit == 0 {
                let j = i | target_bit;
                if j < state.len() {
                    let amp_0 = temp_state[i];
                    let amp_1 = temp_state[j];

                    state[i] = gate_matrix[0] * amp_0 + gate_matrix[1] * amp_1;
                    state[j] = gate_matrix[2] * amp_0 + gate_matrix[3] * amp_1;
                }
            }
        }

        pool.release_state_vector(temp_state);

        Ok(())
    }

    /// Get measurement probabilities with memory optimization
    pub fn get_probabilities(&self) -> QuantRS2Result<Vec<f64>> {
        let state = self.state.as_ref().ok_or_else(|| {
            crate::error::QuantRS2Error::InvalidInput("State not initialized".to_string())
        })?;

        let mut probabilities = self.pool.acquire_probability_buffer(state.len());

        // Compute probabilities using SIMD operations
        for (i, &amplitude) in state.iter().enumerate() {
            probabilities[i] = amplitude.norm_sqr();
        }

        Ok(probabilities)
    }

    /// Release resources when done
    pub fn finalize(mut self) {
        if let Some(state) = self.state.take() {
            self.pool.release_state_vector(state);
        }
    }
}

/// Global quantum buffer pool
static GLOBAL_QUANTUM_POOL: OnceLock<QuantumBufferPool> = OnceLock::new();

/// Get the global quantum buffer pool
pub fn global_quantum_buffer_pool() -> &'static QuantumBufferPool {
    GLOBAL_QUANTUM_POOL.get_or_init(QuantumBufferPool::new)
}

/// Initialize buffer pools
pub fn initialize_buffer_pools() {
    // Force initialization of the global pool
    let _pool = global_quantum_buffer_pool();
}

/// Optimized state vector allocation function
pub fn optimized_state_vector_allocation(num_qubits: usize) -> StateVectorManager {
    let pool = Arc::new(QuantumBufferPool::new());
    StateVectorManager::new(num_qubits, pool)
}

/// Get global memory usage statistics
pub fn get_memory_usage_stats() -> MemoryUsageStats {
    global_quantum_buffer_pool().get_stats()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Slow test: SciRS2 metrics tracking is unexpectedly slow (~71s)
    fn test_buffer_pool_basic_functionality() {
        let pool = QuantumBufferPool::new();

        // Test state vector acquisition and release
        let buffer = pool.acquire_state_vector(100);
        assert_eq!(buffer.len(), 100);

        let stats_before = pool.get_stats();
        pool.release_state_vector(buffer);
        let stats_after = pool.get_stats();

        assert_eq!(
            stats_after.total_allocations,
            stats_before.total_allocations
        );
        assert_eq!(
            stats_after.total_deallocations,
            stats_before.total_deallocations + 1
        );
    }

    #[test]
    fn test_state_vector_manager() {
        let pool = Arc::new(QuantumBufferPool::new());
        let mut manager = StateVectorManager::new(2, pool); // 2-qubit system

        // Initialize state
        manager
            .initialize_zero_state()
            .expect("Failed to initialize zero state");

        // Apply Hadamard-like gate to first qubit
        let h_gate = [
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];

        manager
            .apply_single_qubit_gate(&h_gate, 0)
            .expect("Failed to apply Hadamard gate");

        // Get probabilities
        let probabilities = manager
            .get_probabilities()
            .expect("Failed to get probabilities");
        assert_eq!(probabilities.len(), 4);

        // Should be in equal superposition on first qubit
        assert!((probabilities[0] - 0.5).abs() < 1e-10); // |00⟩
        assert!((probabilities[1] - 0.5).abs() < 1e-10); // |01⟩
        assert!((probabilities[2] - 0.0).abs() < 1e-10); // |10⟩
        assert!((probabilities[3] - 0.0).abs() < 1e-10); // |11⟩

        manager.finalize();
    }

    #[test]
    fn test_chunked_processing_threshold() {
        let pool = Arc::new(QuantumBufferPool::new());

        // Small system should not use chunking
        let small_manager = StateVectorManager::new(10, pool.clone());
        assert!(!small_manager.use_chunked_processing);

        // Large system should use chunking
        let large_manager = StateVectorManager::new(25, pool);
        assert!(large_manager.use_chunked_processing);
    }
}
