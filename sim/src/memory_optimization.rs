//! Advanced Memory Optimization for Quantum Simulation
//!
//! This module provides sophisticated memory management strategies to optimize
//! memory usage patterns for large quantum state vector simulations.

use num_complex::Complex64;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Advanced memory pool with intelligent allocation strategies
#[derive(Debug)]
pub struct AdvancedMemoryPool {
    /// Stratified buffers organized by size classes
    size_pools: RwLock<HashMap<usize, VecDeque<Vec<Complex64>>>>,
    /// Maximum number of buffers per size class
    max_buffers_per_size: usize,
    /// Memory usage statistics
    stats: Arc<Mutex<MemoryStats>>,
    /// Automatic cleanup threshold
    cleanup_threshold: Duration,
    /// Last cleanup time
    last_cleanup: Mutex<Instant>,
}

/// Memory usage statistics for optimization
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total allocations requested
    pub total_allocations: u64,
    /// Cache hits (buffer reused)
    pub cache_hits: u64,
    /// Cache misses (new allocation)
    pub cache_misses: u64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Current memory usage in bytes
    pub current_memory_bytes: u64,
    /// Total cleanup operations
    pub cleanup_operations: u64,
    /// Average allocation size
    pub average_allocation_size: f64,
    /// Buffer size distribution
    pub size_distribution: HashMap<usize, u64>,
}

impl MemoryStats {
    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_allocations as f64
        }
    }

    /// Update statistics for a new allocation
    pub fn record_allocation(&mut self, size: usize, cache_hit: bool) {
        self.total_allocations += 1;
        if cache_hit {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }

        // Update average allocation size
        let total_size =
            self.average_allocation_size * (self.total_allocations - 1) as f64 + size as f64;
        self.average_allocation_size = total_size / self.total_allocations as f64;

        // Update size distribution
        *self.size_distribution.entry(size).or_insert(0) += 1;

        // Update memory usage (approximation)
        let allocation_bytes = size * std::mem::size_of::<Complex64>();
        self.current_memory_bytes += allocation_bytes as u64;
        if self.current_memory_bytes > self.peak_memory_bytes {
            self.peak_memory_bytes = self.current_memory_bytes;
        }
    }

    /// Record memory deallocation
    pub fn record_deallocation(&mut self, size: usize) {
        let deallocation_bytes = size * std::mem::size_of::<Complex64>();
        self.current_memory_bytes = self
            .current_memory_bytes
            .saturating_sub(deallocation_bytes as u64);
    }
}

impl AdvancedMemoryPool {
    /// Create new advanced memory pool
    pub fn new(max_buffers_per_size: usize, cleanup_threshold: Duration) -> Self {
        Self {
            size_pools: RwLock::new(HashMap::new()),
            max_buffers_per_size,
            stats: Arc::new(Mutex::new(MemoryStats::default())),
            cleanup_threshold,
            last_cleanup: Mutex::new(Instant::now()),
        }
    }

    /// Get optimal size class for a requested size (power of 2 buckets)
    fn get_size_class(size: usize) -> usize {
        if size <= 64 {
            64
        } else if size <= 128 {
            128
        } else if size <= 256 {
            256
        } else if size <= 512 {
            512
        } else if size <= 1024 {
            1024
        } else if size <= 2048 {
            2048
        } else if size <= 4096 {
            4096
        } else if size <= 8192 {
            8192
        } else {
            // For large sizes, round up to next power of 2
            let mut power = 1;
            while power < size {
                power <<= 1;
            }
            power
        }
    }

    /// Get buffer from pool with intelligent allocation
    pub fn get_buffer(&self, size: usize) -> Vec<Complex64> {
        let size_class = Self::get_size_class(size);
        let mut cache_hit = false;

        // Try to get from appropriate size pool
        let buffer = {
            let pools = self.size_pools.read().unwrap();
            if let Some(pool) = pools.get(&size_class) {
                if !pool.is_empty() {
                    cache_hit = true;
                    // Need to get write lock to modify
                    drop(pools);
                    let mut pools_write = self.size_pools.write().unwrap();
                    pools_write
                        .get_mut(&size_class)
                        .and_then(|pool| pool.pop_front())
                } else {
                    None
                }
            } else {
                None
            }
        };

        let buffer = if let Some(mut buffer) = buffer {
            // Reuse existing buffer
            buffer.clear();
            buffer.resize(size, Complex64::new(0.0, 0.0));
            buffer
        } else {
            // Allocate new buffer with size class capacity
            let mut buffer = Vec::with_capacity(size_class);
            buffer.resize(size, Complex64::new(0.0, 0.0));
            buffer
        };

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.record_allocation(size, cache_hit);
        }

        // Trigger cleanup if needed
        self.maybe_cleanup();

        buffer
    }

    /// Return buffer to appropriate size pool
    pub fn return_buffer(&self, buffer: Vec<Complex64>) {
        let capacity = buffer.capacity();
        let size_class = Self::get_size_class(capacity);

        // Only cache if capacity matches size class to avoid memory waste
        if capacity == size_class {
            let mut pools = self.size_pools.write().unwrap();
            let pool = pools.entry(size_class).or_insert_with(VecDeque::new);

            if pool.len() < self.max_buffers_per_size {
                pool.push_back(buffer);
                return;
            }
        }

        // Update deallocation stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.record_deallocation(capacity);
        }

        // Buffer will be dropped here if not cached
    }

    /// Periodic cleanup of unused buffers
    fn maybe_cleanup(&self) {
        if let Ok(mut last_cleanup) = self.last_cleanup.try_lock() {
            if last_cleanup.elapsed() > self.cleanup_threshold {
                self.cleanup_unused_buffers();
                *last_cleanup = Instant::now();

                if let Ok(mut stats) = self.stats.lock() {
                    stats.cleanup_operations += 1;
                }
            }
        }
    }

    /// Clean up unused buffers to free memory
    pub fn cleanup_unused_buffers(&self) {
        let mut pools = self.size_pools.write().unwrap();
        let mut freed_memory = 0u64;

        for (size_class, pool) in pools.iter_mut() {
            // Keep only half the buffers in each pool during cleanup
            let target_size = pool.len() / 2;
            while pool.len() > target_size {
                if let Some(buffer) = pool.pop_back() {
                    freed_memory += (buffer.capacity() * std::mem::size_of::<Complex64>()) as u64;
                }
            }
        }

        // Update memory stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.current_memory_bytes = stats.current_memory_bytes.saturating_sub(freed_memory);
        }
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        self.stats.lock().unwrap().clone()
    }

    /// Clear all cached buffers
    pub fn clear(&self) {
        let mut pools = self.size_pools.write().unwrap();
        let mut freed_memory = 0u64;

        for (_, pool) in pools.iter() {
            for buffer in pool.iter() {
                freed_memory += (buffer.capacity() * std::mem::size_of::<Complex64>()) as u64;
            }
        }

        pools.clear();

        // Update memory stats
        if let Ok(mut stats) = self.stats.lock() {
            stats.current_memory_bytes = stats.current_memory_bytes.saturating_sub(freed_memory);
        }
    }
}

/// NUMA-aware memory optimization strategies
pub struct NumaAwareAllocator {
    /// Node-specific memory pools
    node_pools: Vec<AdvancedMemoryPool>,
    /// Current allocation node
    current_node: Mutex<usize>,
}

impl NumaAwareAllocator {
    /// Create NUMA-aware allocator
    pub fn new(num_nodes: usize, max_buffers_per_size: usize) -> Self {
        let node_pools = (0..num_nodes)
            .map(|_| AdvancedMemoryPool::new(max_buffers_per_size, Duration::from_secs(30)))
            .collect();

        Self {
            node_pools,
            current_node: Mutex::new(0),
        }
    }

    /// Get buffer from specific NUMA node
    pub fn get_buffer_from_node(&self, size: usize, node: usize) -> Option<Vec<Complex64>> {
        if node < self.node_pools.len() {
            Some(self.node_pools[node].get_buffer(size))
        } else {
            None
        }
    }

    /// Get buffer with automatic load balancing
    pub fn get_buffer(&self, size: usize) -> Vec<Complex64> {
        let mut current_node = self.current_node.lock().unwrap();
        let node = *current_node;
        *current_node = (*current_node + 1) % self.node_pools.len();
        drop(current_node);

        self.node_pools[node].get_buffer(size)
    }

    /// Return buffer to appropriate node
    pub fn return_buffer(&self, buffer: Vec<Complex64>, preferred_node: Option<usize>) {
        let node = preferred_node.unwrap_or(0).min(self.node_pools.len() - 1);
        self.node_pools[node].return_buffer(buffer);
    }

    /// Get combined statistics from all nodes
    pub fn get_combined_stats(&self) -> MemoryStats {
        let mut combined = MemoryStats::default();

        for pool in &self.node_pools {
            let stats = pool.get_stats();
            combined.total_allocations += stats.total_allocations;
            combined.cache_hits += stats.cache_hits;
            combined.cache_misses += stats.cache_misses;
            combined.current_memory_bytes += stats.current_memory_bytes;
            combined.peak_memory_bytes = combined.peak_memory_bytes.max(stats.peak_memory_bytes);
            combined.cleanup_operations += stats.cleanup_operations;

            // Merge size distributions
            for (size, count) in stats.size_distribution {
                *combined.size_distribution.entry(size).or_insert(0) += count;
            }
        }

        // Recalculate average allocation size
        if combined.total_allocations > 0 {
            let total_size: u64 = combined
                .size_distribution
                .iter()
                .map(|(size, count)| *size as u64 * count)
                .sum();
            combined.average_allocation_size =
                total_size as f64 / combined.total_allocations as f64;
        }

        combined
    }
}

/// Memory optimization utility functions
pub mod utils {
    use super::*;

    /// Estimate memory requirements for a given number of qubits
    pub fn estimate_memory_requirements(num_qubits: usize) -> u64 {
        let state_size = 1usize << num_qubits;
        let bytes_per_amplitude = std::mem::size_of::<Complex64>();
        let state_memory = state_size * bytes_per_amplitude;

        // Add overhead for temporary buffers (estimated 3x for gates)
        let overhead_factor = 3;
        (state_memory * overhead_factor) as u64
    }

    /// Check if system has sufficient memory for simulation
    pub fn check_memory_availability(num_qubits: usize) -> bool {
        let required_memory = estimate_memory_requirements(num_qubits);

        // Get available system memory (this is a simplified check)
        // In practice, you'd use system-specific APIs
        let available_memory = get_available_memory();

        available_memory > required_memory
    }

    /// Get available system memory (placeholder implementation)
    fn get_available_memory() -> u64 {
        // This would use platform-specific APIs in practice
        // For now, return a conservative estimate
        8 * 1024 * 1024 * 1024 // 8 GB
    }

    /// Optimize buffer size for cache efficiency
    pub fn optimize_buffer_size(target_size: usize) -> usize {
        // Align to cache line size (typically 64 bytes)
        let cache_line_size = 64;
        let element_size = std::mem::size_of::<Complex64>();
        let elements_per_cache_line = cache_line_size / element_size;

        // Round up to nearest multiple of cache line elements
        ((target_size + elements_per_cache_line - 1) / elements_per_cache_line)
            * elements_per_cache_line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_memory_pool() {
        let pool = AdvancedMemoryPool::new(4, Duration::from_secs(1));

        // Test buffer allocation and reuse
        let buffer1 = pool.get_buffer(100);
        assert_eq!(buffer1.len(), 100);

        pool.return_buffer(buffer1);

        let buffer2 = pool.get_buffer(100);
        assert_eq!(buffer2.len(), 100);

        // Check cache hit ratio
        let stats = pool.get_stats();
        assert!(stats.cache_hit_ratio() > 0.0);
    }

    #[test]
    fn test_size_class_allocation() {
        assert_eq!(AdvancedMemoryPool::get_size_class(50), 64);
        assert_eq!(AdvancedMemoryPool::get_size_class(100), 128);
        assert_eq!(AdvancedMemoryPool::get_size_class(1000), 1024);
        assert_eq!(AdvancedMemoryPool::get_size_class(5000), 8192);
    }

    #[test]
    fn test_numa_aware_allocator() {
        let allocator = NumaAwareAllocator::new(2, 4);

        let buffer1 = allocator.get_buffer(100);
        let buffer2 = allocator.get_buffer(200);

        allocator.return_buffer(buffer1, Some(0));
        allocator.return_buffer(buffer2, Some(1));

        let stats = allocator.get_combined_stats();
        assert_eq!(stats.total_allocations, 2);
    }

    #[test]
    fn test_memory_estimation() {
        let memory_4_qubits = utils::estimate_memory_requirements(4);
        let memory_8_qubits = utils::estimate_memory_requirements(8);

        // 8-qubit simulation should require much more memory than 4-qubit
        assert!(memory_8_qubits > memory_4_qubits * 10);
    }
}
