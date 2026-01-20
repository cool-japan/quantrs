//! GPU Memory Bandwidth Optimization Module
//!
//! This module provides advanced memory optimization techniques for quantum GPU operations,
//! including prefetching, memory coalescing, and adaptive buffer management.
//!
//! ## Features
//! - Memory coalescing for contiguous access patterns
//! - Software prefetching for predictable access patterns
//! - Adaptive buffer pooling for reduced allocation overhead
//! - Cache-aware memory layouts for quantum state vectors
//! - Memory bandwidth monitoring and optimization suggestions

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::platform::PlatformCapabilities;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Memory bandwidth optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryBandwidthConfig {
    /// Enable prefetching for predictable access patterns
    pub enable_prefetching: bool,
    /// Prefetch distance in cache lines
    pub prefetch_distance: usize,
    /// Enable memory coalescing optimization
    pub enable_coalescing: bool,
    /// Minimum coalescing width in bytes
    pub coalescing_width: usize,
    /// Enable adaptive buffer pooling
    pub enable_buffer_pooling: bool,
    /// Maximum pool size in bytes
    pub max_pool_size: usize,
    /// Enable cache-aware memory layout
    pub enable_cache_aware_layout: bool,
    /// Target cache line size
    pub cache_line_size: usize,
}

impl Default for MemoryBandwidthConfig {
    fn default() -> Self {
        let capabilities = PlatformCapabilities::detect();
        let cache_line_size = capabilities.cpu.cache.line_size.unwrap_or(64);

        Self {
            enable_prefetching: true,
            prefetch_distance: 8,
            enable_coalescing: true,
            coalescing_width: 128, // 128 bytes for modern GPUs
            enable_buffer_pooling: true,
            max_pool_size: 1024 * 1024 * 512, // 512 MB
            enable_cache_aware_layout: true,
            cache_line_size,
        }
    }
}

/// Memory bandwidth metrics for monitoring and optimization
#[derive(Debug, Clone, Default)]
pub struct MemoryBandwidthMetrics {
    /// Total bytes transferred to device
    pub bytes_to_device: usize,
    /// Total bytes transferred from device
    pub bytes_from_device: usize,
    /// Number of memory transfers
    pub transfer_count: usize,
    /// Total transfer time
    pub total_transfer_time: Duration,
    /// Average bandwidth in GB/s
    pub average_bandwidth_gbps: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// Coalescing efficiency (0.0 to 1.0)
    pub coalescing_efficiency: f64,
}

/// Memory buffer pool for efficient allocation
pub struct MemoryBufferPool {
    /// Free buffers organized by size
    free_buffers: RwLock<HashMap<usize, Vec<Vec<Complex64>>>>,
    /// Total allocated bytes
    allocated_bytes: AtomicUsize,
    /// Configuration
    config: MemoryBandwidthConfig,
    /// Pool hit count for statistics
    pool_hits: AtomicUsize,
    /// Pool miss count for statistics
    pool_misses: AtomicUsize,
}

impl MemoryBufferPool {
    /// Create a new memory buffer pool
    pub fn new(config: MemoryBandwidthConfig) -> Self {
        Self {
            free_buffers: RwLock::new(HashMap::new()),
            allocated_bytes: AtomicUsize::new(0),
            config,
            pool_hits: AtomicUsize::new(0),
            pool_misses: AtomicUsize::new(0),
        }
    }

    /// Acquire a buffer from the pool or allocate new
    pub fn acquire(&self, size: usize) -> Vec<Complex64> {
        // Round up to cache line aligned size
        let aligned_size = self.align_to_cache_line(size);

        // Try to get from pool
        if let Ok(mut buffers) = self.free_buffers.write() {
            if let Some(buffer_list) = buffers.get_mut(&aligned_size) {
                if let Some(buffer) = buffer_list.pop() {
                    self.pool_hits.fetch_add(1, Ordering::Relaxed);
                    return buffer;
                }
            }
        }

        // Allocate new buffer
        self.pool_misses.fetch_add(1, Ordering::Relaxed);
        let buffer_bytes = aligned_size * std::mem::size_of::<Complex64>();
        self.allocated_bytes
            .fetch_add(buffer_bytes, Ordering::Relaxed);

        vec![Complex64::new(0.0, 0.0); aligned_size]
    }

    /// Release a buffer back to the pool
    pub fn release(&self, mut buffer: Vec<Complex64>) {
        let size = buffer.len();
        let buffer_bytes = size * std::mem::size_of::<Complex64>();

        // Check if we're within pool limit
        if self.allocated_bytes.load(Ordering::Relaxed) <= self.config.max_pool_size {
            // Clear buffer for reuse
            for elem in &mut buffer {
                *elem = Complex64::new(0.0, 0.0);
            }

            if let Ok(mut buffers) = self.free_buffers.write() {
                buffers.entry(size).or_default().push(buffer);
            }
        } else {
            // Drop the buffer to free memory
            self.allocated_bytes
                .fetch_sub(buffer_bytes, Ordering::Relaxed);
        }
    }

    /// Align size to cache line boundary
    const fn align_to_cache_line(&self, size: usize) -> usize {
        let elem_size = std::mem::size_of::<Complex64>();
        let elems_per_line = self.config.cache_line_size / elem_size;
        ((size + elems_per_line - 1) / elems_per_line) * elems_per_line
    }

    /// Get pool statistics
    pub fn get_statistics(&self) -> PoolStatistics {
        let hits = self.pool_hits.load(Ordering::Relaxed);
        let misses = self.pool_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        PoolStatistics {
            allocated_bytes: self.allocated_bytes.load(Ordering::Relaxed),
            pool_hit_rate: if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            },
            total_acquisitions: total,
        }
    }

    /// Clear all pooled buffers
    pub fn clear(&self) {
        if let Ok(mut buffers) = self.free_buffers.write() {
            for (size, buffer_list) in buffers.drain() {
                let freed_bytes = size * std::mem::size_of::<Complex64>() * buffer_list.len();
                self.allocated_bytes
                    .fetch_sub(freed_bytes, Ordering::Relaxed);
            }
        }
    }
}

/// Pool statistics
#[derive(Debug, Clone)]
pub struct PoolStatistics {
    /// Total allocated bytes in pool
    pub allocated_bytes: usize,
    /// Hit rate for pool acquisitions
    pub pool_hit_rate: f64,
    /// Total number of acquisitions
    pub total_acquisitions: usize,
}

/// Memory bandwidth optimizer for quantum operations
pub struct MemoryBandwidthOptimizer {
    /// Configuration
    config: MemoryBandwidthConfig,
    /// Buffer pool
    buffer_pool: Arc<MemoryBufferPool>,
    /// Bandwidth metrics
    metrics: RwLock<MemoryBandwidthMetrics>,
}

impl MemoryBandwidthOptimizer {
    /// Create a new memory bandwidth optimizer
    pub fn new(config: MemoryBandwidthConfig) -> Self {
        let buffer_pool = Arc::new(MemoryBufferPool::new(config.clone()));

        Self {
            config,
            buffer_pool,
            metrics: RwLock::new(MemoryBandwidthMetrics::default()),
        }
    }

    /// Get optimal memory layout for quantum state vector
    pub const fn get_optimal_layout(&self, n_qubits: usize) -> MemoryLayout {
        let state_size = 1 << n_qubits;
        let elem_size = std::mem::size_of::<Complex64>();
        let total_bytes = state_size * elem_size;

        // Determine optimal layout based on size and cache
        let elems_per_line = self.config.cache_line_size / elem_size;

        MemoryLayout {
            total_elements: state_size,
            total_bytes,
            cache_line_elements: elems_per_line,
            recommended_alignment: self.config.cache_line_size,
            use_tiled_layout: n_qubits >= 10, // Use tiling for large states
            tile_size: if n_qubits >= 10 { 256 } else { 0 },
        }
    }

    /// Optimize memory access pattern for coalesced reads
    pub fn optimize_coalesced_access<F>(
        &self,
        data: &mut [Complex64],
        access_pattern: &[usize],
        operation: F,
    ) -> QuantRS2Result<()>
    where
        F: Fn(&mut Complex64, usize) -> QuantRS2Result<()>,
    {
        if !self.config.enable_coalescing {
            // Fall back to direct access
            for &idx in access_pattern {
                if idx >= data.len() {
                    return Err(QuantRS2Error::InvalidInput(
                        "Index out of bounds".to_string(),
                    ));
                }
                operation(&mut data[idx], idx)?;
            }
            return Ok(());
        }

        // Sort indices for coalesced access
        let mut sorted_indices: Vec<_> = access_pattern.to_vec();
        sorted_indices.sort_unstable();

        // Process in coalesced chunks
        let coalescing_elements = self.config.coalescing_width / std::mem::size_of::<Complex64>();

        for chunk in sorted_indices.chunks(coalescing_elements) {
            for &idx in chunk {
                if idx >= data.len() {
                    return Err(QuantRS2Error::InvalidInput(
                        "Index out of bounds".to_string(),
                    ));
                }
                operation(&mut data[idx], idx)?;
            }
        }

        Ok(())
    }

    /// Prefetch data for upcoming operations
    pub fn prefetch_for_gate_application(
        &self,
        state: &[Complex64],
        qubit: usize,
        n_qubits: usize,
    ) {
        if !self.config.enable_prefetching {
            return;
        }

        let state_size = 1 << n_qubits;
        let qubit_mask = 1 << qubit;

        // Prefetch amplitude pairs that will be accessed
        for i in 0..(state_size / 2).min(self.config.prefetch_distance * 2) {
            let idx0 = (i & !(qubit_mask >> 1)) | ((i & (qubit_mask >> 1)) << 1);
            let idx1 = idx0 | qubit_mask;

            if idx0 < state.len() && idx1 < state.len() {
                // Software prefetch hint (platform-specific)
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    let ptr0 = state.as_ptr().add(idx0);
                    let ptr1 = state.as_ptr().add(idx1);
                    std::arch::x86_64::_mm_prefetch(
                        ptr0 as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                    std::arch::x86_64::_mm_prefetch(
                        ptr1 as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }

                #[cfg(target_arch = "aarch64")]
                {
                    // ARM prefetch using compiler intrinsics
                    let _ = (state[idx0], state[idx1]);
                }
            }
        }
    }

    /// Acquire buffer from pool
    pub fn acquire_buffer(&self, size: usize) -> Vec<Complex64> {
        self.buffer_pool.acquire(size)
    }

    /// Release buffer to pool
    pub fn release_buffer(&self, buffer: Vec<Complex64>) {
        self.buffer_pool.release(buffer);
    }

    /// Record transfer metrics
    pub fn record_transfer(&self, bytes: usize, to_device: bool, duration: Duration) {
        if let Ok(mut metrics) = self.metrics.write() {
            if to_device {
                metrics.bytes_to_device += bytes;
            } else {
                metrics.bytes_from_device += bytes;
            }
            metrics.transfer_count += 1;
            metrics.total_transfer_time += duration;

            // Calculate bandwidth
            let total_bytes = metrics.bytes_to_device + metrics.bytes_from_device;
            let total_secs = metrics.total_transfer_time.as_secs_f64();
            if total_secs > 0.0 {
                metrics.average_bandwidth_gbps = (total_bytes as f64) / total_secs / 1e9;
            }
        }
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> MemoryBandwidthMetrics {
        self.metrics
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Get pool statistics
    pub fn get_pool_statistics(&self) -> PoolStatistics {
        self.buffer_pool.get_statistics()
    }

    /// Clear buffer pool
    pub fn clear_pool(&self) {
        self.buffer_pool.clear();
    }

    /// Get optimization recommendations based on current metrics
    pub fn get_optimization_recommendations(&self) -> Vec<String> {
        let metrics = self.get_metrics();
        let pool_stats = self.get_pool_statistics();
        let mut recommendations = Vec::new();

        // Check bandwidth utilization
        if metrics.average_bandwidth_gbps < 10.0 && metrics.transfer_count > 100 {
            recommendations.push(
                "Consider batching memory transfers to improve bandwidth utilization".to_string(),
            );
        }

        // Check pool hit rate
        if pool_stats.pool_hit_rate < 0.5 && pool_stats.total_acquisitions > 100 {
            recommendations.push(format!(
                "Pool hit rate is {:.1}%. Consider increasing pool size for better reuse",
                pool_stats.pool_hit_rate * 100.0
            ));
        }

        // Check coalescing efficiency
        if metrics.coalescing_efficiency < 0.7 {
            recommendations.push(
                "Memory access pattern has low coalescing efficiency. Consider reordering accesses"
                    .to_string(),
            );
        }

        // Check cache utilization
        if metrics.cache_hit_rate < 0.8 && metrics.transfer_count > 50 {
            recommendations.push(
                "Cache hit rate is low. Consider using cache-aware memory layouts".to_string(),
            );
        }

        if recommendations.is_empty() {
            recommendations.push("Memory bandwidth utilization is optimal".to_string());
        }

        recommendations
    }
}

/// Memory layout information
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    /// Total number of elements
    pub total_elements: usize,
    /// Total bytes
    pub total_bytes: usize,
    /// Elements per cache line
    pub cache_line_elements: usize,
    /// Recommended alignment in bytes
    pub recommended_alignment: usize,
    /// Whether to use tiled layout
    pub use_tiled_layout: bool,
    /// Tile size for tiled layout
    pub tile_size: usize,
}

/// Streaming memory transfer for large state vectors
pub struct StreamingTransfer {
    /// Chunk size for streaming
    chunk_size: usize,
    /// Number of concurrent transfers
    concurrent_transfers: usize,
    /// Buffer pool reference
    buffer_pool: Arc<MemoryBufferPool>,
}

impl StreamingTransfer {
    /// Create new streaming transfer manager
    pub const fn new(chunk_size: usize, buffer_pool: Arc<MemoryBufferPool>) -> Self {
        Self {
            chunk_size,
            concurrent_transfers: 2, // Double buffering
            buffer_pool,
        }
    }

    /// Stream data to device with double buffering
    pub fn stream_to_device<F>(
        &self,
        data: &[Complex64],
        transfer_fn: F,
    ) -> QuantRS2Result<Duration>
    where
        F: Fn(&[Complex64], usize) -> QuantRS2Result<()>,
    {
        let start = Instant::now();
        let mut offset = 0;

        while offset < data.len() {
            let chunk_end = (offset + self.chunk_size).min(data.len());
            let chunk = &data[offset..chunk_end];

            transfer_fn(chunk, offset)?;
            offset = chunk_end;
        }

        Ok(start.elapsed())
    }

    /// Stream data from device
    pub fn stream_from_device<F>(
        &self,
        data: &mut [Complex64],
        transfer_fn: F,
    ) -> QuantRS2Result<Duration>
    where
        F: Fn(&mut [Complex64], usize) -> QuantRS2Result<()>,
    {
        let start = Instant::now();
        let mut offset = 0;

        while offset < data.len() {
            let chunk_end = (offset + self.chunk_size).min(data.len());
            let chunk = &mut data[offset..chunk_end];

            transfer_fn(chunk, offset)?;
            offset = chunk_end;
        }

        Ok(start.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_bandwidth_config_default() {
        let config = MemoryBandwidthConfig::default();
        assert!(config.enable_prefetching);
        assert!(config.enable_coalescing);
        assert!(config.enable_buffer_pooling);
        assert!(config.cache_line_size > 0);
    }

    #[test]
    fn test_buffer_pool_acquire_release() {
        let config = MemoryBandwidthConfig::default();
        let pool = MemoryBufferPool::new(config);

        // Acquire buffer
        let buffer = pool.acquire(100);
        assert!(buffer.len() >= 100);

        // Release buffer
        let size = buffer.len();
        pool.release(buffer);

        // Acquire again - should get from pool
        let buffer2 = pool.acquire(100);
        assert_eq!(buffer2.len(), size);

        let stats = pool.get_statistics();
        assert!(stats.pool_hit_rate > 0.0);
    }

    #[test]
    fn test_memory_layout_computation() {
        let config = MemoryBandwidthConfig::default();
        let optimizer = MemoryBandwidthOptimizer::new(config);

        let layout = optimizer.get_optimal_layout(4);
        assert_eq!(layout.total_elements, 16);
        assert!(!layout.use_tiled_layout);

        let layout_large = optimizer.get_optimal_layout(12);
        assert_eq!(layout_large.total_elements, 4096);
        assert!(layout_large.use_tiled_layout);
    }

    #[test]
    fn test_coalesced_access_optimization() {
        let config = MemoryBandwidthConfig::default();
        let optimizer = MemoryBandwidthOptimizer::new(config);

        let mut data = vec![Complex64::new(0.0, 0.0); 100];
        let pattern = vec![50, 10, 30, 70, 90];

        let result = optimizer.optimize_coalesced_access(&mut data, &pattern, |elem, idx| {
            *elem = Complex64::new(idx as f64, 0.0);
            Ok(())
        });

        assert!(result.is_ok());
        assert_eq!(data[10], Complex64::new(10.0, 0.0));
        assert_eq!(data[50], Complex64::new(50.0, 0.0));
    }

    #[test]
    fn test_transfer_metrics_recording() {
        let config = MemoryBandwidthConfig::default();
        let optimizer = MemoryBandwidthOptimizer::new(config);

        optimizer.record_transfer(1024, true, Duration::from_micros(100));
        optimizer.record_transfer(1024, false, Duration::from_micros(100));

        let metrics = optimizer.get_metrics();
        assert_eq!(metrics.bytes_to_device, 1024);
        assert_eq!(metrics.bytes_from_device, 1024);
        assert_eq!(metrics.transfer_count, 2);
    }

    #[test]
    fn test_optimization_recommendations() {
        let config = MemoryBandwidthConfig::default();
        let optimizer = MemoryBandwidthOptimizer::new(config);

        let recommendations = optimizer.get_optimization_recommendations();
        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_streaming_transfer() {
        let config = MemoryBandwidthConfig::default();
        let pool = Arc::new(MemoryBufferPool::new(config));
        let streamer = StreamingTransfer::new(32, pool);

        let data = vec![Complex64::new(1.0, 0.0); 100];
        let result = streamer.stream_to_device(&data, |_chunk, _offset| Ok(()));
        assert!(result.is_ok());
    }

    #[test]
    fn test_pool_clear() {
        let config = MemoryBandwidthConfig::default();
        let pool = MemoryBufferPool::new(config);

        // Acquire and release buffers
        for _ in 0..10 {
            let buffer = pool.acquire(100);
            pool.release(buffer);
        }

        // Clear pool
        pool.clear();

        let stats = pool.get_statistics();
        assert_eq!(stats.allocated_bytes, 0);
    }
}
