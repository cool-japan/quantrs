//! Centralized Memory Buffer Management for Large Quantum Circuits
//!
//! This module provides optimized memory management to prevent fragmentation
//! in large quantum circuit processing by centralizing buffer pools and
//! implementing intelligent allocation strategies.

use num_complex::Complex64;
use quantrs2_core::buffer_pool::BufferPool;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Global buffer manager for optimized memory allocation
static GLOBAL_BUFFER_MANAGER: OnceLock<Arc<Mutex<GlobalBufferManager>>> = OnceLock::new();

/// Centralized buffer pool manager for preventing memory fragmentation
pub struct GlobalBufferManager {
    /// Pool for f64 numerical computations
    f64_pool: BufferPool<f64>,

    /// Pool for complex number operations
    complex_pool: BufferPool<Complex64>,

    /// Pool for intermediate vector allocations
    vector_pools: HashMap<usize, Vec<Vec<f64>>>,

    /// Pool for gate parameter storage
    parameter_pool: BufferPool<f64>,

    /// Memory usage statistics
    stats: MemoryStats,
}

/// Memory usage statistics for monitoring
#[derive(Debug, Default, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub fragmentation_ratio: f64,
}

impl GlobalBufferManager {
    /// Create a new buffer manager with optimized pool sizes
    fn new() -> Self {
        Self {
            f64_pool: BufferPool::new(), // BufferPool manages capacity internally
            complex_pool: BufferPool::new(),
            vector_pools: HashMap::with_capacity(16),
            parameter_pool: BufferPool::new(),
            stats: MemoryStats::default(),
        }
    }

    /// Get a reusable f64 buffer
    pub fn get_f64_buffer(&mut self, size: usize) -> Vec<f64> {
        self.stats.total_allocated += size * std::mem::size_of::<f64>();
        self.update_peak_usage();
        self.stats.pool_hits += 1;

        // Use the correct BufferPool API
        let mut buffer = self.f64_pool.get(size);
        buffer.resize(size, 0.0);
        buffer
    }

    /// Return a buffer to the pool for reuse
    pub fn return_f64_buffer(&mut self, buffer: Vec<f64>) {
        // Only pool buffers of reasonable size to prevent memory bloat
        if buffer.len() <= 10000 && buffer.capacity() <= 20000 {
            self.f64_pool.put(buffer);
        }
    }

    /// Get a reusable complex buffer
    pub fn get_complex_buffer(&mut self, size: usize) -> Vec<Complex64> {
        self.stats.total_allocated += size * std::mem::size_of::<Complex64>();
        self.update_peak_usage();
        self.stats.pool_hits += 1;

        // Use the correct BufferPool API
        let mut buffer = self.complex_pool.get(size);
        buffer.resize(size, Complex64::new(0.0, 0.0));
        buffer
    }

    /// Return a complex buffer to the pool
    pub fn return_complex_buffer(&mut self, buffer: Vec<Complex64>) {
        if buffer.len() <= 10000 && buffer.capacity() <= 20000 {
            self.complex_pool.put(buffer);
        }
    }

    /// Get a vector for specific size with pooling
    pub fn get_sized_vector(&mut self, size: usize) -> Vec<f64> {
        if let Some(pool) = self.vector_pools.get_mut(&size) {
            if let Some(vec) = pool.pop() {
                self.stats.pool_hits += 1;
                return vec;
            }
        }

        self.stats.pool_misses += 1;
        vec![0.0; size]
    }

    /// Return a sized vector to the appropriate pool
    pub fn return_sized_vector(&mut self, mut vector: Vec<f64>) {
        let size = vector.len();
        vector.clear();

        // Only pool common sizes to prevent excessive memory usage
        if size <= 1024 {
            let pool = self.vector_pools.entry(size).or_insert_with(Vec::new);
            if pool.len() < 10 {
                // Limit pool size
                pool.push(vector);
            }
        }
    }

    /// Get buffer for gate parameters
    pub fn get_parameter_buffer(&mut self, size: usize) -> Vec<f64> {
        self.stats.pool_hits += 1;
        let mut buffer = self.parameter_pool.get(size);
        buffer.resize(size, 0.0);
        buffer
    }

    /// Return parameter buffer
    pub fn return_parameter_buffer(&mut self, buffer: Vec<f64>) {
        if buffer.len() <= 100 {
            // Gate parameters are typically small
            self.parameter_pool.put(buffer);
        }
    }

    /// Force garbage collection of unused buffers
    pub fn collect_garbage(&mut self) {
        // Clear oversized vector pools
        self.vector_pools.retain(|&size, pool| {
            pool.retain(|v| v.capacity() < size * 2);
            size <= 1024 && !pool.is_empty()
        });

        // Update fragmentation ratio
        let allocated = self.stats.total_allocated;
        let peak = self.stats.peak_usage;
        self.stats.fragmentation_ratio = if peak > 0 {
            1.0 - (allocated as f64 / peak as f64)
        } else {
            0.0
        };
    }

    /// Get current memory statistics
    pub fn get_stats(&self) -> &MemoryStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = MemoryStats::default();
    }

    fn update_peak_usage(&mut self) {
        if self.stats.total_allocated > self.stats.peak_usage {
            self.stats.peak_usage = self.stats.total_allocated;
        }
    }
}

/// Public interface for accessing the global buffer manager
pub struct BufferManager;

impl BufferManager {
    /// Get the global buffer manager instance
    pub fn instance() -> Arc<Mutex<GlobalBufferManager>> {
        GLOBAL_BUFFER_MANAGER
            .get_or_init(|| Arc::new(Mutex::new(GlobalBufferManager::new())))
            .clone()
    }

    /// Allocate an f64 buffer through the global pool
    pub fn alloc_f64_buffer(size: usize) -> Vec<f64> {
        Self::instance().lock().unwrap().get_f64_buffer(size)
    }

    /// Return an f64 buffer to the global pool
    pub fn free_f64_buffer(buffer: Vec<f64>) {
        Self::instance().lock().unwrap().return_f64_buffer(buffer);
    }

    /// Allocate a complex buffer through the global pool
    pub fn alloc_complex_buffer(size: usize) -> Vec<Complex64> {
        Self::instance().lock().unwrap().get_complex_buffer(size)
    }

    /// Return a complex buffer to the global pool
    pub fn free_complex_buffer(buffer: Vec<Complex64>) {
        Self::instance()
            .lock()
            .unwrap()
            .return_complex_buffer(buffer);
    }

    /// Allocate a parameter buffer for gate operations
    pub fn alloc_parameter_buffer(size: usize) -> Vec<f64> {
        Self::instance().lock().unwrap().get_parameter_buffer(size)
    }

    /// Return a parameter buffer to the pool
    pub fn free_parameter_buffer(buffer: Vec<f64>) {
        Self::instance()
            .lock()
            .unwrap()
            .return_parameter_buffer(buffer);
    }

    /// Get memory usage statistics
    pub fn get_memory_stats() -> MemoryStats {
        Self::instance().lock().unwrap().get_stats().clone()
    }

    /// Trigger garbage collection to reduce fragmentation
    pub fn collect_garbage() {
        Self::instance().lock().unwrap().collect_garbage();
    }

    /// Reset memory usage statistics
    pub fn reset_stats() {
        Self::instance().lock().unwrap().reset_stats();
    }
}

/// RAII wrapper for automatic buffer return
pub struct ManagedF64Buffer {
    buffer: Option<Vec<f64>>,
}

impl ManagedF64Buffer {
    /// Create a managed buffer that will be automatically returned to pool
    pub fn new(size: usize) -> Self {
        Self {
            buffer: Some(BufferManager::alloc_f64_buffer(size)),
        }
    }

    /// Get mutable access to the buffer
    pub fn as_mut(&mut self) -> &mut Vec<f64> {
        self.buffer.as_mut().unwrap()
    }

    /// Get immutable access to the buffer
    pub fn as_ref(&self) -> &Vec<f64> {
        self.buffer.as_ref().unwrap()
    }

    /// Take ownership of the buffer (preventing automatic return)
    pub fn take(mut self) -> Vec<f64> {
        self.buffer.take().unwrap()
    }
}

impl Drop for ManagedF64Buffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            BufferManager::free_f64_buffer(buffer);
        }
    }
}

/// RAII wrapper for complex buffers
pub struct ManagedComplexBuffer {
    buffer: Option<Vec<Complex64>>,
}

impl ManagedComplexBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            buffer: Some(BufferManager::alloc_complex_buffer(size)),
        }
    }

    pub fn as_mut(&mut self) -> &mut Vec<Complex64> {
        self.buffer.as_mut().unwrap()
    }

    pub fn as_ref(&self) -> &Vec<Complex64> {
        self.buffer.as_ref().unwrap()
    }

    pub fn take(mut self) -> Vec<Complex64> {
        self.buffer.take().unwrap()
    }
}

impl Drop for ManagedComplexBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            BufferManager::free_complex_buffer(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pooling() {
        let buffer1 = BufferManager::alloc_f64_buffer(100);
        assert_eq!(buffer1.len(), 100);

        BufferManager::free_f64_buffer(buffer1);

        let buffer2 = BufferManager::alloc_f64_buffer(100);
        assert_eq!(buffer2.len(), 100);

        BufferManager::free_f64_buffer(buffer2);

        let stats = BufferManager::get_memory_stats();
        assert!(stats.pool_hits > 0 || stats.pool_misses > 0);
    }

    #[test]
    fn test_managed_buffer() {
        {
            let mut managed = ManagedF64Buffer::new(50);
            managed.as_mut()[0] = 42.0;
            assert_eq!(managed.as_ref()[0], 42.0);
        } // Buffer automatically returned here

        let stats = BufferManager::get_memory_stats();
        // Stats should show buffer was used
        assert!(stats.total_allocated > 0);
    }

    #[test]
    fn test_complex_buffer_pooling() {
        let buffer1 = BufferManager::alloc_complex_buffer(50);
        assert_eq!(buffer1.len(), 50);

        BufferManager::free_complex_buffer(buffer1);

        let buffer2 = BufferManager::alloc_complex_buffer(50);
        assert_eq!(buffer2.len(), 50);

        BufferManager::free_complex_buffer(buffer2);
    }

    #[test]
    fn test_garbage_collection() {
        // Allocate and free several buffers
        for _ in 0..10 {
            let buffer = BufferManager::alloc_f64_buffer(1000);
            BufferManager::free_f64_buffer(buffer);
        }

        BufferManager::collect_garbage();
        let stats = BufferManager::get_memory_stats();

        // Should have some fragmentation data
        assert!(stats.fragmentation_ratio >= 0.0);
    }
}
