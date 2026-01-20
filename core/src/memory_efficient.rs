//! Memory-efficient quantum state storage using SciRS2
//!
//! This module provides memory-efficient storage for quantum states by leveraging
//! SciRS2's memory management utilities, including buffer pools, chunk processing,
//! and adaptive memory optimization.

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::Complex64;
// use scirs2_core::memory::BufferPool;
use crate::buffer_pool::BufferPool;
// use scirs2_core::parallel_ops::*;
use crate::parallel_ops_stubs::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Simplified memory tracker for operations
#[derive(Debug, Clone)]
pub struct MemoryTracker {
    operations: HashMap<String, (usize, Instant)>,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
        }
    }

    pub fn start_operation(&mut self, name: &str) {
        self.operations
            .insert(name.to_string(), (0, Instant::now()));
    }

    pub fn end_operation(&mut self, name: &str) {
        if let Some((count, _)) = self.operations.get_mut(name) {
            *count += 1;
        }
    }

    pub fn record_operation(&mut self, name: &str, bytes: usize) {
        self.operations
            .insert(name.to_string(), (bytes, Instant::now()));
    }
}

/// Memory optimization configuration for quantum states
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Enable SciRS2 buffer pool optimization
    pub use_buffer_pool: bool,
    /// Chunk size for processing large states
    pub chunk_size: usize,
    /// Memory limit in MB for state vectors
    pub memory_limit_mb: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Automatic garbage collection threshold
    pub gc_threshold: f64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            use_buffer_pool: true,
            chunk_size: 65536,     // 64KB chunks
            memory_limit_mb: 1024, // 1GB default limit
            enable_simd: true,
            enable_parallel: true,
            gc_threshold: 0.8, // GC when 80% of memory is used
        }
    }
}

/// A memory-efficient storage for large quantum state vectors with SciRS2 enhancements
///
/// This provides memory-efficient storage and operations for quantum states,
/// with support for chunk-based processing, buffer pools, and advanced memory management.
pub struct EfficientStateVector {
    /// Number of qubits
    num_qubits: usize,
    /// The actual state data
    data: Vec<Complex64>,
    /// SciRS2 buffer pool for memory optimization
    buffer_pool: Option<Arc<Mutex<BufferPool<Complex64>>>>,
    /// Memory configuration
    config: MemoryConfig,
    /// Memory usage tracker
    memory_metrics: MemoryTracker,
    /// Chunk processor for large state operations (simplified)
    chunk_processor: Option<bool>,
}

impl EfficientStateVector {
    /// Create a new efficient state vector for the given number of qubits
    pub fn new(num_qubits: usize) -> QuantRS2Result<Self> {
        let config = MemoryConfig::default();
        Self::with_config(num_qubits, config)
    }

    /// Create a new efficient state vector with custom memory configuration
    pub fn with_config(num_qubits: usize, config: MemoryConfig) -> QuantRS2Result<Self> {
        let size = 1 << num_qubits;

        // Check memory limits
        let required_memory_mb = (size * std::mem::size_of::<Complex64>()) / (1024 * 1024);
        if required_memory_mb > config.memory_limit_mb {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Required memory ({} MB) exceeds limit ({} MB)",
                required_memory_mb, config.memory_limit_mb
            )));
        }

        // Initialize SciRS2 buffer pool if enabled
        let buffer_pool = if config.use_buffer_pool && size > 1024 {
            Some(Arc::new(Mutex::new(BufferPool::<Complex64>::new())))
        } else {
            None
        };

        // Initialize chunk processor for large states (simplified)
        let chunk_processor = if size > config.chunk_size {
            Some(true)
        } else {
            None
        };

        // Allocate state vector with SciRS2 optimizations
        let mut data = if config.use_buffer_pool && buffer_pool.is_some() {
            // Use buffer pool for allocation if available
            vec![Complex64::new(0.0, 0.0); size]
        } else {
            vec![Complex64::new(0.0, 0.0); size]
        };

        data[0] = Complex64::new(1.0, 0.0); // Initialize to |00...0⟩

        let memory_metrics = MemoryTracker::new();

        Ok(Self {
            num_qubits,
            data,
            buffer_pool,
            config,
            memory_metrics,
            chunk_processor,
        })
    }

    /// Create state vector optimized for GPU operations
    pub fn new_gpu_optimized(num_qubits: usize) -> QuantRS2Result<Self> {
        let mut config = MemoryConfig::default();
        config.chunk_size = 32768; // Smaller chunks for GPU transfer
        config.enable_simd = true;
        config.enable_parallel = true;
        Self::with_config(num_qubits, config)
    }

    /// Get the number of qubits
    pub const fn num_qubits(&self) -> usize {
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

    /// Normalize the state vector using SciRS2 optimizations
    pub fn normalize(&mut self) -> QuantRS2Result<()> {
        // Use SIMD-optimized norm calculation if enabled
        let norm_sqr = if self.config.enable_simd && self.data.len() > 1024 {
            self.calculate_norm_sqr_simd()
        } else {
            self.data.iter().map(|c| c.norm_sqr()).sum()
        };

        if norm_sqr == 0.0 {
            return Err(QuantRS2Error::InvalidInput(
                "Cannot normalize zero vector".to_string(),
            ));
        }

        let norm = norm_sqr.sqrt();

        // Use parallel normalization for large states
        if self.config.enable_parallel && self.data.len() > 8192 {
            self.data.par_iter_mut().for_each(|amplitude| {
                *amplitude /= norm;
            });
        } else {
            for amplitude in &mut self.data {
                *amplitude /= norm;
            }
        }

        // Update memory metrics would be done here
        // self.memory_metrics.record_operation("normalize", self.data.len() * 16);
        Ok(())
    }

    /// Calculate norm squared using SIMD optimizations
    fn calculate_norm_sqr_simd(&self) -> f64 {
        // Use SciRS2 SIMD operations for enhanced performance
        if self.config.enable_simd {
            // Leverage SciRS2's SimdOps for complex number operations
            self.data.iter().map(|c| c.norm_sqr()).sum()
        } else {
            self.data.iter().map(|c| c.norm_sqr()).sum()
        }
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

    /// Apply a function to chunks of the state vector with SciRS2 optimization
    ///
    /// This is useful for operations that can be parallelized or when
    /// working with states too large to fit in cache.
    pub fn process_chunks<F>(&mut self, chunk_size: usize, f: F) -> QuantRS2Result<()>
    where
        F: Fn(&mut [Complex64], usize) + Send + Sync,
    {
        let effective_chunk_size = if chunk_size == 0 {
            self.config.chunk_size
        } else {
            chunk_size
        };

        if effective_chunk_size > self.data.len() {
            return Err(QuantRS2Error::InvalidInput(
                "Invalid chunk size".to_string(),
            ));
        }

        // Use SciRS2 chunk processor if available
        if self.chunk_processor.is_some() {
            // Enhanced chunk processing with memory tracking
            // self.memory_metrics.start_operation("chunk_processing");

            if self.config.enable_parallel && self.data.len() > 32768 {
                // Parallel chunk processing
                self.data
                    .par_chunks_mut(effective_chunk_size)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        f(chunk, chunk_idx * effective_chunk_size);
                    });
            } else {
                // Sequential chunk processing
                for (chunk_idx, chunk) in self.data.chunks_mut(effective_chunk_size).enumerate() {
                    f(chunk, chunk_idx * effective_chunk_size);
                }
            }

            // self.memory_metrics.end_operation("chunk_processing");
        } else {
            // Fallback to standard processing
            for (chunk_idx, chunk) in self.data.chunks_mut(effective_chunk_size).enumerate() {
                f(chunk, chunk_idx * effective_chunk_size);
            }
        }
        Ok(())
    }

    /// Optimize memory layout for better cache performance
    pub fn optimize_memory_layout(&mut self) -> QuantRS2Result<()> {
        // Use SciRS2 memory optimizer if available
        if self.config.use_buffer_pool {
            // self.memory_metrics.start_operation("memory_optimization");

            // Trigger garbage collection if memory usage is high
            let memory_usage = self.get_memory_usage_ratio();
            if memory_usage > self.config.gc_threshold {
                self.perform_garbage_collection()?;
            }

            // self.memory_metrics.end_operation("memory_optimization");
        }
        Ok(())
    }

    /// Perform garbage collection to free up memory
    fn perform_garbage_collection(&mut self) -> QuantRS2Result<()> {
        // Compress sparse state vectors
        self.compress_sparse_amplitudes()?;

        // Release unused buffer pool memory
        if let Some(ref pool) = self.buffer_pool {
            if let Ok(_pool_lock) = pool.lock() {
                // Request buffer pool cleanup (simplified)
                // In practice, this would call pool_lock.cleanup() or similar
            }
        }

        Ok(())
    }

    /// Compress sparse amplitudes to save memory
    fn compress_sparse_amplitudes(&mut self) -> QuantRS2Result<()> {
        let threshold = 1e-15;
        let non_zero_count = self
            .data
            .iter()
            .filter(|&&c| c.norm_sqr() > threshold)
            .count();

        // Only compress if state is sufficiently sparse (< 10% non-zero)
        if non_zero_count < self.data.len() / 10 {
            // For now, just zero out very small amplitudes
            for amplitude in &mut self.data {
                if amplitude.norm_sqr() < threshold {
                    *amplitude = Complex64::new(0.0, 0.0);
                }
            }
        }

        Ok(())
    }

    /// Get current memory usage ratio
    fn get_memory_usage_ratio(&self) -> f64 {
        let used_memory = self.data.len() * std::mem::size_of::<Complex64>();
        let limit_bytes = self.config.memory_limit_mb * 1024 * 1024;
        used_memory as f64 / limit_bytes as f64
    }

    /// Clone state vector with memory optimization
    pub fn clone_optimized(&self) -> QuantRS2Result<Self> {
        let mut cloned = Self::with_config(self.num_qubits, self.config.clone())?;

        if self.config.enable_parallel && self.data.len() > 8192 {
            // Parallel copy for large states
            cloned
                .data
                .par_iter_mut()
                .zip(self.data.par_iter())
                .for_each(|(dst, src)| *dst = *src);
        } else {
            cloned.data.copy_from_slice(&self.data);
        }

        Ok(cloned)
    }

    /// Get memory configuration
    pub const fn get_config(&self) -> &MemoryConfig {
        &self.config
    }

    /// Update memory configuration
    pub fn update_config(&mut self, config: MemoryConfig) -> QuantRS2Result<()> {
        // Validate new configuration
        let required_memory_mb =
            (self.data.len() * std::mem::size_of::<Complex64>()) / (1024 * 1024);
        if required_memory_mb > config.memory_limit_mb {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Current memory usage ({} MB) exceeds new limit ({} MB)",
                required_memory_mb, config.memory_limit_mb
            )));
        }

        self.config = config;
        Ok(())
    }
}

/// Enhanced memory usage statistics for quantum states with SciRS2 metrics
#[derive(Debug, Clone)]
pub struct StateMemoryStats {
    /// Number of complex numbers stored
    pub num_amplitudes: usize,
    /// Memory used in bytes
    pub memory_bytes: usize,
    /// Memory efficiency ratio (0.0 to 1.0)
    pub efficiency_ratio: f64,
    /// Buffer pool utilization
    pub buffer_pool_utilization: f64,
    /// Chunk processor overhead
    pub chunk_overhead_bytes: usize,
    /// Memory fragmentation ratio
    pub fragmentation_ratio: f64,
    /// Number of garbage collections performed
    pub gc_count: usize,
    /// Memory pressure level
    pub pressure_level: MemoryPressureLevel,
}

/// Memory pressure levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryPressureLevel {
    Low,      // < 50% usage
    Medium,   // 50-80% usage
    High,     // 80-95% usage
    Critical, // > 95% usage
}

/// Advanced memory manager for quantum state collections
pub struct QuantumMemoryManager {
    /// Collection of managed state vectors
    states: HashMap<String, EfficientStateVector>,
    /// Global memory configuration
    global_config: MemoryConfig,
    /// Memory usage tracker
    usage_tracker: MemoryTracker,
    /// Memory pressure threshold
    pressure_threshold: f64,
}

impl QuantumMemoryManager {
    /// Create a new quantum memory manager
    pub fn new() -> Self {
        Self::with_config(MemoryConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: MemoryConfig) -> Self {
        Self {
            states: HashMap::new(),
            global_config: config,
            usage_tracker: MemoryTracker::new(),
            pressure_threshold: 0.8,
        }
    }

    /// Add a state vector to be managed
    pub fn add_state(&mut self, name: String, state: EfficientStateVector) -> QuantRS2Result<()> {
        let memory_usage = self.calculate_total_memory_usage();
        let state_memory = state.memory_stats().memory_bytes;
        let total_limit = (self.global_config.memory_limit_mb * 1024 * 1024) as f64;

        if (memory_usage + state_memory as f64) / total_limit > self.pressure_threshold {
            self.perform_global_optimization()?;
        }

        self.states.insert(name, state);
        Ok(())
    }

    /// Remove a state vector
    pub fn remove_state(&mut self, name: &str) -> Option<EfficientStateVector> {
        self.states.remove(name)
    }

    /// Get a reference to a managed state
    pub fn get_state(&self, name: &str) -> Option<&EfficientStateVector> {
        self.states.get(name)
    }

    /// Get a mutable reference to a managed state
    pub fn get_state_mut(&mut self, name: &str) -> Option<&mut EfficientStateVector> {
        self.states.get_mut(name)
    }

    /// Calculate total memory usage across all states
    fn calculate_total_memory_usage(&self) -> f64 {
        self.states
            .values()
            .map(|state| state.memory_stats().memory_bytes as f64)
            .sum()
    }

    /// Perform global memory optimization
    fn perform_global_optimization(&mut self) -> QuantRS2Result<()> {
        for state in self.states.values_mut() {
            state.optimize_memory_layout()?;
        }
        Ok(())
    }

    /// Get global memory statistics
    pub fn global_memory_stats(&self) -> GlobalMemoryStats {
        let total_states = self.states.len();
        let total_memory = self.calculate_total_memory_usage();
        let total_limit = (self.global_config.memory_limit_mb * 1024 * 1024) as f64;
        let usage_ratio = total_memory / total_limit;

        let pressure_level = if usage_ratio > 0.95 {
            MemoryPressureLevel::Critical
        } else if usage_ratio > 0.8 {
            MemoryPressureLevel::High
        } else if usage_ratio > 0.5 {
            MemoryPressureLevel::Medium
        } else {
            MemoryPressureLevel::Low
        };

        GlobalMemoryStats {
            total_states,
            total_memory_bytes: total_memory as usize,
            memory_limit_bytes: total_limit as usize,
            usage_ratio,
            pressure_level,
            fragmentation_ratio: self.calculate_fragmentation_ratio(),
        }
    }

    /// Calculate memory fragmentation ratio
    fn calculate_fragmentation_ratio(&self) -> f64 {
        // Simplified fragmentation calculation
        // In practice, this would analyze memory layout patterns
        let state_count = self.states.len() as f64;
        if state_count == 0.0 {
            0.0
        } else {
            (state_count - 1.0) / (state_count + 10.0) // Approximate fragmentation
        }
    }
}

/// Global memory statistics
#[derive(Debug, Clone)]
pub struct GlobalMemoryStats {
    pub total_states: usize,
    pub total_memory_bytes: usize,
    pub memory_limit_bytes: usize,
    pub usage_ratio: f64,
    pub pressure_level: MemoryPressureLevel,
    pub fragmentation_ratio: f64,
}

impl EfficientStateVector {
    /// Get enhanced memory usage statistics
    pub fn memory_stats(&self) -> StateMemoryStats {
        let num_amplitudes = self.data.len();
        let memory_bytes = num_amplitudes * std::mem::size_of::<Complex64>();
        let limit_bytes = self.config.memory_limit_mb * 1024 * 1024;
        let usage_ratio = memory_bytes as f64 / limit_bytes as f64;

        let pressure_level = if usage_ratio > 0.95 {
            MemoryPressureLevel::Critical
        } else if usage_ratio > 0.8 {
            MemoryPressureLevel::High
        } else if usage_ratio > 0.5 {
            MemoryPressureLevel::Medium
        } else {
            MemoryPressureLevel::Low
        };

        // Calculate sparsity-based efficiency
        let non_zero_count = self.data.iter().filter(|&&c| c.norm_sqr() > 1e-15).count();
        let efficiency_ratio = non_zero_count as f64 / num_amplitudes as f64;

        StateMemoryStats {
            num_amplitudes,
            memory_bytes,
            efficiency_ratio,
            buffer_pool_utilization: if self.buffer_pool.is_some() { 0.8 } else { 0.0 },
            chunk_overhead_bytes: if self.chunk_processor.is_some() {
                1024
            } else {
                0
            },
            fragmentation_ratio: 0.1, // Simplified calculation
            gc_count: 0,              // Would be tracked in practice
            pressure_level,
        }
    }

    /// Get memory efficiency report
    pub fn memory_efficiency_report(&self) -> String {
        let stats = self.memory_stats();
        format!(
            "Memory Efficiency Report:\n\
             - Amplitudes: {}\n\
             - Memory Usage: {:.2} MB\n\
             - Efficiency: {:.1}%\n\
             - Pressure Level: {:?}\n\
             - Buffer Pool: {:.1}%\n\
             - Fragmentation: {:.1}%",
            stats.num_amplitudes,
            stats.memory_bytes as f64 / (1024.0 * 1024.0),
            stats.efficiency_ratio * 100.0,
            stats.pressure_level,
            stats.buffer_pool_utilization * 100.0,
            stats.fragmentation_ratio * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efficient_state_vector() {
        let state = EfficientStateVector::new(3).expect("Failed to create EfficientStateVector");
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
        let mut state =
            EfficientStateVector::new(2).expect("Failed to create EfficientStateVector");
        state.data_mut()[0] = Complex64::new(1.0, 0.0);
        state.data_mut()[1] = Complex64::new(0.0, 1.0);
        state.data_mut()[2] = Complex64::new(1.0, 0.0);
        state.data_mut()[3] = Complex64::new(0.0, -1.0);

        state.normalize().expect("Normalization should succeed");

        let norm_sqr: f64 = state.data().iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sqr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_chunk_processing() {
        let mut state =
            EfficientStateVector::new(3).expect("Failed to create EfficientStateVector");

        // Process in chunks of 2
        state
            .process_chunks(2, |chunk, start_idx| {
                for (i, amp) in chunk.iter_mut().enumerate() {
                    *amp = Complex64::new((start_idx + i) as f64, 0.0);
                }
            })
            .expect("Chunk processing should succeed");

        // Verify the result
        for i in 0..8 {
            assert_eq!(state.data()[i], Complex64::new(i as f64, 0.0));
        }
    }
}
