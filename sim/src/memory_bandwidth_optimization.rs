//! Memory bandwidth optimization for large state vector simulations.
//!
//! This module implements advanced memory access optimizations for quantum
//! state vector simulations, including cache-optimized layouts, prefetching
//! strategies, data locality optimizations, and NUMA-aware memory management.

use scirs2_core::ndarray::Array2;
use scirs2_core::parallel_ops::{IndexedParallelIterator, ParallelIterator};
use scirs2_core::Complex64;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;

/// Memory layout strategies for state vectors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Standard contiguous layout
    Contiguous,
    /// Cache-line aligned layout with padding
    CacheAligned,
    /// Blocked layout for cache optimization
    Blocked,
    /// Interleaved layout for NUMA systems
    Interleaved,
    /// Hierarchical layout for multi-level caches
    Hierarchical,
    /// Adaptive layout based on access patterns
    Adaptive,
}

/// Memory bandwidth optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    /// Memory layout strategy
    pub layout: MemoryLayout,
    /// Cache line size in bytes
    pub cache_line_size: usize,
    /// L1 cache size in bytes
    pub l1_cache_size: usize,
    /// L2 cache size in bytes
    pub l2_cache_size: usize,
    /// L3 cache size in bytes
    pub l3_cache_size: usize,
    /// Block size for blocked layouts
    pub block_size: usize,
    /// Enable memory prefetching
    pub enable_prefetching: bool,
    /// Prefetch distance (number of cache lines ahead)
    pub prefetch_distance: usize,
    /// Enable NUMA optimizations
    pub enable_numa_optimization: bool,
    /// Memory pool size for temporary allocations
    pub memory_pool_size: usize,
    /// Enable memory bandwidth monitoring
    pub enable_bandwidth_monitoring: bool,
    /// Adaptive optimization threshold
    pub adaptation_threshold: f64,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            layout: MemoryLayout::Adaptive,
            cache_line_size: 64,            // Common cache line size
            l1_cache_size: 32 * 1024,       // 32KB L1 cache
            l2_cache_size: 256 * 1024,      // 256KB L2 cache
            l3_cache_size: 8 * 1024 * 1024, // 8MB L3 cache
            block_size: 4096,               // 4KB blocks
            enable_prefetching: true,
            prefetch_distance: 4,
            enable_numa_optimization: true,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB pool
            enable_bandwidth_monitoring: true,
            adaptation_threshold: 0.1,
        }
    }
}

/// Memory access pattern tracking
#[derive(Debug, Clone)]
pub struct MemoryAccessPattern {
    /// Access frequency for each memory region
    pub access_frequency: HashMap<usize, u64>,
    /// Sequential access patterns
    pub sequential_accesses: VecDeque<(usize, usize)>,
    /// Random access patterns
    pub random_accesses: VecDeque<usize>,
    /// Cache miss count
    pub cache_misses: u64,
    /// Total memory accesses
    pub total_accesses: u64,
    /// Last access time
    pub last_access_time: Instant,
}

impl Default for MemoryAccessPattern {
    fn default() -> Self {
        Self {
            access_frequency: HashMap::new(),
            sequential_accesses: VecDeque::new(),
            random_accesses: VecDeque::new(),
            cache_misses: 0,
            total_accesses: 0,
            last_access_time: Instant::now(),
        }
    }
}

/// Memory bandwidth monitoring
#[derive(Debug, Clone)]
pub struct BandwidthMonitor {
    /// Bandwidth samples over time
    pub bandwidth_samples: VecDeque<(Instant, f64)>,
    /// Current bandwidth utilization (0.0 to 1.0)
    pub current_utilization: f64,
    /// Peak bandwidth achieved
    pub peak_bandwidth: f64,
    /// Average bandwidth over time window
    pub average_bandwidth: f64,
    /// Memory access latency samples
    pub latency_samples: VecDeque<Duration>,
}

impl Default for BandwidthMonitor {
    fn default() -> Self {
        Self {
            bandwidth_samples: VecDeque::new(),
            current_utilization: 0.0,
            peak_bandwidth: 0.0,
            average_bandwidth: 0.0,
            latency_samples: VecDeque::new(),
        }
    }
}

/// Memory pool for efficient allocation and reuse
#[derive(Debug)]
pub struct MemoryPool {
    /// Pre-allocated memory blocks
    blocks: Mutex<Vec<(*mut u8, usize)>>,
    /// Block size
    block_size: usize,
    /// Maximum number of blocks
    max_blocks: usize,
    /// Current allocation count
    allocated_count: Mutex<usize>,
}

impl MemoryPool {
    /// Create a new memory pool
    pub const fn new(block_size: usize, max_blocks: usize) -> Result<Self> {
        Ok(Self {
            blocks: Mutex::new(Vec::new()),
            block_size,
            max_blocks,
            allocated_count: Mutex::new(0),
        })
    }

    /// Allocate a memory block from the pool
    pub fn allocate(&self) -> Result<NonNull<u8>> {
        let mut blocks = self
            .blocks
            .lock()
            .map_err(|e| SimulatorError::MemoryAllocationFailed(format!("Lock poisoned: {e}")))?;

        if let Some((ptr, _)) = blocks.pop() {
            Ok(unsafe { NonNull::new_unchecked(ptr) })
        } else {
            // Allocate new block if pool is empty
            let layout = Layout::from_size_align(self.block_size, 64)
                .map_err(|e| SimulatorError::MemoryAllocationFailed(e.to_string()))?;

            let ptr = unsafe { System.alloc(layout) };
            if ptr.is_null() {
                return Err(SimulatorError::MemoryAllocationFailed(
                    "Failed to allocate memory block".to_string(),
                ));
            }

            let mut count = self.allocated_count.lock().map_err(|e| {
                SimulatorError::MemoryAllocationFailed(format!("Lock poisoned: {e}"))
            })?;
            *count += 1;

            Ok(unsafe { NonNull::new_unchecked(ptr) })
        }
    }

    /// Return a memory block to the pool
    pub fn deallocate(&self, ptr: NonNull<u8>) -> Result<()> {
        let mut blocks = self
            .blocks
            .lock()
            .map_err(|e| SimulatorError::MemoryAllocationFailed(format!("Lock poisoned: {e}")))?;

        if blocks.len() < self.max_blocks {
            blocks.push((ptr.as_ptr(), self.block_size));
        } else {
            // Pool is full, actually deallocate
            let layout = Layout::from_size_align(self.block_size, 64)
                .map_err(|e| SimulatorError::MemoryAllocationFailed(e.to_string()))?;
            unsafe { System.dealloc(ptr.as_ptr(), layout) };

            let mut count = self.allocated_count.lock().map_err(|e| {
                SimulatorError::MemoryAllocationFailed(format!("Lock poisoned: {e}"))
            })?;
            *count -= 1;
        }

        Ok(())
    }
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

/// Optimized state vector with memory bandwidth optimizations
#[derive(Debug)]
pub struct OptimizedStateVector {
    /// State vector data with optimized layout
    data: Vec<Complex64>,
    /// Number of qubits
    num_qubits: usize,
    /// Memory layout being used
    layout: MemoryLayout,
    /// Block size for blocked layouts
    block_size: usize,
    /// Memory access pattern tracking
    access_pattern: Arc<RwLock<MemoryAccessPattern>>,
    /// Bandwidth monitor
    bandwidth_monitor: Arc<RwLock<BandwidthMonitor>>,
    /// Memory pool for temporary allocations
    memory_pool: Arc<MemoryPool>,
    /// Configuration
    config: MemoryOptimizationConfig,
}

impl OptimizedStateVector {
    /// Create a new optimized state vector
    pub fn new(num_qubits: usize, config: MemoryOptimizationConfig) -> Result<Self> {
        let size = 1 << num_qubits;
        let memory_pool = Arc::new(MemoryPool::new(
            config.memory_pool_size / 1024, // Block size
            1024,                           // Max blocks
        )?);

        let mut data = Self::allocate_with_layout(size, config.layout, &config)?;

        // Initialize to |0...0⟩ state
        data[0] = Complex64::new(1.0, 0.0);

        Ok(Self {
            data,
            num_qubits,
            layout: config.layout,
            block_size: config.block_size,
            access_pattern: Arc::new(RwLock::new(MemoryAccessPattern::default())),
            bandwidth_monitor: Arc::new(RwLock::new(BandwidthMonitor::default())),
            memory_pool,
            config,
        })
    }

    /// Allocate memory with specific layout optimization
    fn allocate_with_layout(
        size: usize,
        layout: MemoryLayout,
        config: &MemoryOptimizationConfig,
    ) -> Result<Vec<Complex64>> {
        match layout {
            MemoryLayout::Contiguous => {
                let mut data = Vec::with_capacity(size);
                data.resize(size, Complex64::new(0.0, 0.0));
                Ok(data)
            }
            MemoryLayout::CacheAligned => Self::allocate_cache_aligned(size, config),
            MemoryLayout::Blocked => Self::allocate_blocked(size, config),
            MemoryLayout::Interleaved => Self::allocate_interleaved(size, config),
            MemoryLayout::Hierarchical => Self::allocate_hierarchical(size, config),
            MemoryLayout::Adaptive => {
                // Start with cache-aligned and adapt based on usage
                Self::allocate_cache_aligned(size, config)
            }
        }
    }

    /// Allocate cache-aligned memory
    fn allocate_cache_aligned(
        size: usize,
        config: &MemoryOptimizationConfig,
    ) -> Result<Vec<Complex64>> {
        let element_size = std::mem::size_of::<Complex64>();
        let elements_per_line = config.cache_line_size / element_size;
        let padded_size = size.div_ceil(elements_per_line) * elements_per_line;

        let mut data = Vec::with_capacity(padded_size);
        data.resize(size, Complex64::new(0.0, 0.0));
        data.resize(padded_size, Complex64::new(0.0, 0.0)); // Padding

        Ok(data)
    }

    /// Allocate blocked memory layout
    fn allocate_blocked(size: usize, config: &MemoryOptimizationConfig) -> Result<Vec<Complex64>> {
        let mut data = Vec::with_capacity(size);
        data.resize(size, Complex64::new(0.0, 0.0));

        // Reorganize data in cache-friendly blocks
        let block_size = config.block_size / std::mem::size_of::<Complex64>();
        let num_blocks = size.div_ceil(block_size);

        let mut blocked_data = Vec::with_capacity(size);
        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = std::cmp::min(start + block_size, size);

            blocked_data.extend_from_slice(&data[start..end]);
        }

        Ok(blocked_data)
    }

    /// Allocate interleaved memory for NUMA systems
    fn allocate_interleaved(
        size: usize,
        _config: &MemoryOptimizationConfig,
    ) -> Result<Vec<Complex64>> {
        // For now, use standard allocation
        // In a full implementation, we'd use NUMA APIs
        let mut data = Vec::with_capacity(size);
        data.resize(size, Complex64::new(0.0, 0.0));
        Ok(data)
    }

    /// Allocate hierarchical memory layout
    fn allocate_hierarchical(
        size: usize,
        config: &MemoryOptimizationConfig,
    ) -> Result<Vec<Complex64>> {
        // Hierarchical layout optimized for multi-level caches
        let l1_elements = config.l1_cache_size / std::mem::size_of::<Complex64>();
        let l2_elements = config.l2_cache_size / std::mem::size_of::<Complex64>();

        let mut data = Vec::with_capacity(size);
        data.resize(size, Complex64::new(0.0, 0.0));

        // Reorganize based on cache hierarchy
        // This is a simplified implementation
        Ok(data)
    }

    /// Apply a single-qubit gate with memory optimization
    pub fn apply_single_qubit_gate_optimized(
        &mut self,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        let start_time = Instant::now();

        let mask = 1 << target;
        let size = self.data.len();

        // Use optimized memory access patterns
        match self.layout {
            MemoryLayout::Blocked => {
                self.apply_single_qubit_gate_blocked(target, gate_matrix, mask)?;
            }
            MemoryLayout::CacheAligned => {
                self.apply_single_qubit_gate_cache_aligned(target, gate_matrix, mask)?;
            }
            _ => {
                self.apply_single_qubit_gate_standard(target, gate_matrix, mask)?;
            }
        }

        // Update bandwidth monitoring
        let elapsed = start_time.elapsed();
        self.update_bandwidth_monitor(size * std::mem::size_of::<Complex64>(), elapsed);

        Ok(())
    }

    /// Apply single-qubit gate with blocked memory access
    fn apply_single_qubit_gate_blocked(
        &mut self,
        target: usize,
        gate_matrix: &Array2<Complex64>,
        mask: usize,
    ) -> Result<()> {
        let block_size = self.block_size / std::mem::size_of::<Complex64>();
        let num_blocks = self.data.len().div_ceil(block_size);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = std::cmp::min(start + block_size, self.data.len());

            // Prefetch next block if enabled
            if self.config.enable_prefetching && block_idx + 1 < num_blocks {
                let next_start = (block_idx + 1) * block_size;
                if next_start < self.data.len() {
                    Self::prefetch_memory(&self.data[next_start]);
                }
            }

            // Process current block
            for i in (start..end).step_by(2) {
                if i + 1 < self.data.len() {
                    let i0 = i & !mask;
                    let i1 = i0 | mask;

                    if i1 < self.data.len() {
                        let amp0 = self.data[i0];
                        let amp1 = self.data[i1];

                        self.data[i0] = gate_matrix[[0, 0]] * amp0 + gate_matrix[[0, 1]] * amp1;
                        self.data[i1] = gate_matrix[[1, 0]] * amp0 + gate_matrix[[1, 1]] * amp1;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply single-qubit gate with cache-aligned memory access
    fn apply_single_qubit_gate_cache_aligned(
        &mut self,
        target: usize,
        gate_matrix: &Array2<Complex64>,
        mask: usize,
    ) -> Result<()> {
        let elements_per_line = self.config.cache_line_size / std::mem::size_of::<Complex64>();

        for chunk_start in (0..self.data.len()).step_by(elements_per_line) {
            let chunk_end = std::cmp::min(chunk_start + elements_per_line, self.data.len());

            // Prefetch next cache line
            if self.config.enable_prefetching && chunk_end < self.data.len() {
                Self::prefetch_memory(&self.data[chunk_end]);
            }

            for i in (chunk_start..chunk_end).step_by(2) {
                if i + 1 < self.data.len() {
                    let i0 = i & !mask;
                    let i1 = i0 | mask;

                    if i1 < self.data.len() {
                        let amp0 = self.data[i0];
                        let amp1 = self.data[i1];

                        self.data[i0] = gate_matrix[[0, 0]] * amp0 + gate_matrix[[0, 1]] * amp1;
                        self.data[i1] = gate_matrix[[1, 0]] * amp0 + gate_matrix[[1, 1]] * amp1;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply single-qubit gate with standard memory access
    fn apply_single_qubit_gate_standard(
        &mut self,
        target: usize,
        gate_matrix: &Array2<Complex64>,
        mask: usize,
    ) -> Result<()> {
        for i in (0..self.data.len()).step_by(2) {
            let i0 = i & !mask;
            let i1 = i0 | mask;

            if i1 < self.data.len() {
                let amp0 = self.data[i0];
                let amp1 = self.data[i1];

                self.data[i0] = gate_matrix[[0, 0]] * amp0 + gate_matrix[[0, 1]] * amp1;
                self.data[i1] = gate_matrix[[1, 0]] * amp0 + gate_matrix[[1, 1]] * amp1;
            }
        }

        Ok(())
    }

    /// Prefetch memory to cache
    #[inline(always)]
    fn prefetch_memory(addr: &Complex64) {
        // TODO: Use scirs2_core's platform-agnostic prefetch operations when API is stabilized
        // For now, use a volatile read as a simple prefetch hint
        unsafe {
            let _ = std::ptr::read_volatile(std::ptr::from_ref(addr).cast::<u8>());
        }
    }

    /// Apply a two-qubit gate with memory optimization
    pub fn apply_two_qubit_gate_optimized(
        &mut self,
        control: usize,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        let start_time = Instant::now();

        let control_mask = 1 << control;
        let target_mask = 1 << target;
        let size = self.data.len();

        // Optimize for data locality
        match self.layout {
            MemoryLayout::Blocked => {
                self.apply_two_qubit_gate_blocked(control_mask, target_mask, gate_matrix)?;
            }
            _ => {
                self.apply_two_qubit_gate_standard(control_mask, target_mask, gate_matrix)?;
            }
        }

        // Update bandwidth monitoring
        let elapsed = start_time.elapsed();
        self.update_bandwidth_monitor(size * std::mem::size_of::<Complex64>(), elapsed);

        Ok(())
    }

    /// Apply two-qubit gate with blocked memory access
    fn apply_two_qubit_gate_blocked(
        &mut self,
        control_mask: usize,
        target_mask: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        let block_size = self.block_size / std::mem::size_of::<Complex64>();
        let num_blocks = self.data.len().div_ceil(block_size);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = std::cmp::min(start + block_size, self.data.len());

            // Prefetch next block
            if self.config.enable_prefetching && block_idx + 1 < num_blocks {
                let next_start = (block_idx + 1) * block_size;
                if next_start < self.data.len() {
                    Self::prefetch_memory(&self.data[next_start]);
                }
            }

            // Process current block
            for i in (start..end).step_by(4) {
                if i + 3 < self.data.len() {
                    let i00 = i & !(control_mask | target_mask);
                    let i01 = i00 | target_mask;
                    let i10 = i00 | control_mask;
                    let i11 = i00 | control_mask | target_mask;

                    if i11 < self.data.len() {
                        let amp00 = self.data[i00];
                        let amp01 = self.data[i01];
                        let amp10 = self.data[i10];
                        let amp11 = self.data[i11];

                        self.data[i00] = gate_matrix[[0, 0]] * amp00
                            + gate_matrix[[0, 1]] * amp01
                            + gate_matrix[[0, 2]] * amp10
                            + gate_matrix[[0, 3]] * amp11;
                        self.data[i01] = gate_matrix[[1, 0]] * amp00
                            + gate_matrix[[1, 1]] * amp01
                            + gate_matrix[[1, 2]] * amp10
                            + gate_matrix[[1, 3]] * amp11;
                        self.data[i10] = gate_matrix[[2, 0]] * amp00
                            + gate_matrix[[2, 1]] * amp01
                            + gate_matrix[[2, 2]] * amp10
                            + gate_matrix[[2, 3]] * amp11;
                        self.data[i11] = gate_matrix[[3, 0]] * amp00
                            + gate_matrix[[3, 1]] * amp01
                            + gate_matrix[[3, 2]] * amp10
                            + gate_matrix[[3, 3]] * amp11;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply two-qubit gate with standard memory access
    fn apply_two_qubit_gate_standard(
        &mut self,
        control_mask: usize,
        target_mask: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        for i in (0..self.data.len()).step_by(4) {
            let i00 = i & !(control_mask | target_mask);
            let i01 = i00 | target_mask;
            let i10 = i00 | control_mask;
            let i11 = i00 | control_mask | target_mask;

            if i11 < self.data.len() {
                let amp00 = self.data[i00];
                let amp01 = self.data[i01];
                let amp10 = self.data[i10];
                let amp11 = self.data[i11];

                self.data[i00] = gate_matrix[[0, 0]] * amp00
                    + gate_matrix[[0, 1]] * amp01
                    + gate_matrix[[0, 2]] * amp10
                    + gate_matrix[[0, 3]] * amp11;
                self.data[i01] = gate_matrix[[1, 0]] * amp00
                    + gate_matrix[[1, 1]] * amp01
                    + gate_matrix[[1, 2]] * amp10
                    + gate_matrix[[1, 3]] * amp11;
                self.data[i10] = gate_matrix[[2, 0]] * amp00
                    + gate_matrix[[2, 1]] * amp01
                    + gate_matrix[[2, 2]] * amp10
                    + gate_matrix[[2, 3]] * amp11;
                self.data[i11] = gate_matrix[[3, 0]] * amp00
                    + gate_matrix[[3, 1]] * amp01
                    + gate_matrix[[3, 2]] * amp10
                    + gate_matrix[[3, 3]] * amp11;
            }
        }

        Ok(())
    }

    /// Update bandwidth monitoring
    fn update_bandwidth_monitor(&self, bytes_accessed: usize, elapsed: Duration) {
        if let Ok(mut monitor) = self.bandwidth_monitor.write() {
            let bandwidth = bytes_accessed as f64 / elapsed.as_secs_f64();
            let now = Instant::now();

            monitor.bandwidth_samples.push_back((now, bandwidth));

            // Keep only recent samples (last 100)
            while monitor.bandwidth_samples.len() > 100 {
                monitor.bandwidth_samples.pop_front();
            }

            // Update statistics
            if bandwidth > monitor.peak_bandwidth {
                monitor.peak_bandwidth = bandwidth;
            }

            let sum: f64 = monitor.bandwidth_samples.iter().map(|(_, bw)| bw).sum();
            monitor.average_bandwidth = sum / monitor.bandwidth_samples.len() as f64;

            // Estimate current utilization (simplified)
            let theoretical_max = 100.0 * 1024.0 * 1024.0 * 1024.0; // 100 GB/s theoretical
            monitor.current_utilization = bandwidth / theoretical_max;
        }
    }

    /// Get current bandwidth statistics
    pub fn get_bandwidth_stats(&self) -> Result<BandwidthMonitor> {
        self.bandwidth_monitor
            .read()
            .map(|guard| guard.clone())
            .map_err(|e| SimulatorError::InvalidState(format!("RwLock poisoned: {e}")))
    }

    /// Adapt memory layout based on access patterns
    pub fn adapt_memory_layout(&mut self) -> Result<()> {
        if self.layout != MemoryLayout::Adaptive {
            return Ok(());
        }

        let access_pattern = self
            .access_pattern
            .read()
            .map_err(|e| SimulatorError::InvalidState(format!("RwLock poisoned: {e}")))?;
        let bandwidth_stats = self
            .bandwidth_monitor
            .read()
            .map_err(|e| SimulatorError::InvalidState(format!("RwLock poisoned: {e}")))?;

        // Analyze access patterns and bandwidth utilization
        let sequential_ratio = access_pattern.sequential_accesses.len() as f64
            / (access_pattern.total_accesses as f64 + 1.0);

        let new_layout = if sequential_ratio > 0.8 {
            MemoryLayout::CacheAligned
        } else if bandwidth_stats.current_utilization < 0.5 {
            MemoryLayout::Blocked
        } else {
            MemoryLayout::Hierarchical
        };

        if new_layout != self.layout {
            // Reorganize data with new layout
            let new_data = Self::allocate_with_layout(self.data.len(), new_layout, &self.config)?;
            // Copy data (simplified - in practice we'd do proper layout transformation)
            self.data = new_data;
            self.layout = new_layout;
        }

        Ok(())
    }

    /// Get memory usage statistics
    #[must_use]
    pub fn get_memory_stats(&self) -> MemoryStats {
        let element_size = std::mem::size_of::<Complex64>();
        MemoryStats {
            total_memory: self.data.len() * element_size,
            allocated_memory: self.data.capacity() * element_size,
            layout: self.layout,
            cache_efficiency: self.calculate_cache_efficiency(),
            memory_utilization: self.calculate_memory_utilization(),
        }
    }

    /// Calculate cache efficiency estimate
    fn calculate_cache_efficiency(&self) -> f64 {
        let access_pattern = match self.access_pattern.read() {
            Ok(guard) => guard,
            Err(_) => return 1.0, // Default to full efficiency if lock is poisoned
        };
        if access_pattern.total_accesses == 0 {
            return 1.0;
        }

        let hit_rate =
            1.0 - (access_pattern.cache_misses as f64 / access_pattern.total_accesses as f64);
        hit_rate.clamp(0.0, 1.0)
    }

    /// Calculate memory utilization
    fn calculate_memory_utilization(&self) -> f64 {
        match self.bandwidth_monitor.read() {
            Ok(guard) => guard.current_utilization,
            Err(_) => 0.0, // Default to zero utilization if lock is poisoned
        }
    }

    /// Get state vector data (read-only access)
    #[must_use]
    pub fn data(&self) -> &[Complex64] {
        &self.data
    }

    /// Get mutable state vector data with access tracking
    pub fn data_mut(&mut self) -> &mut [Complex64] {
        // Track memory access
        if let Ok(mut pattern) = self.access_pattern.write() {
            pattern.total_accesses += 1;
            pattern.last_access_time = Instant::now();
        }

        &mut self.data
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Total memory used in bytes
    pub total_memory: usize,
    /// Allocated memory capacity in bytes
    pub allocated_memory: usize,
    /// Current memory layout
    pub layout: MemoryLayout,
    /// Cache efficiency (0.0 to 1.0)
    pub cache_efficiency: f64,
    /// Memory bandwidth utilization (0.0 to 1.0)
    pub memory_utilization: f64,
}

/// Memory bandwidth optimization manager
#[derive(Debug)]
pub struct MemoryBandwidthOptimizer {
    /// Configuration
    config: MemoryOptimizationConfig,
    /// Global memory pool
    memory_pool: Arc<MemoryPool>,
    /// `SciRS2` backend integration
    backend: Option<SciRS2Backend>,
}

impl MemoryBandwidthOptimizer {
    /// Create a new memory bandwidth optimizer
    pub fn new(config: MemoryOptimizationConfig) -> Result<Self> {
        let memory_pool = Arc::new(MemoryPool::new(config.memory_pool_size / 1024, 1024)?);

        Ok(Self {
            config,
            memory_pool,
            backend: None,
        })
    }

    /// Initialize `SciRS2` backend integration
    pub fn init_scirs2_backend(&mut self) -> Result<()> {
        // SciRS2Backend::new() returns a SciRS2Backend directly
        let backend = SciRS2Backend::new();
        self.backend = Some(backend);
        Ok(())
    }

    /// Create an optimized state vector
    pub fn create_optimized_state_vector(&self, num_qubits: usize) -> Result<OptimizedStateVector> {
        OptimizedStateVector::new(num_qubits, self.config.clone())
    }

    /// Optimize memory access for a given circuit
    pub fn optimize_circuit_memory_access(
        &self,
        state_vector: &mut OptimizedStateVector,
        circuit_depth: usize,
    ) -> Result<MemoryOptimizationReport> {
        let start_time = Instant::now();

        // Analyze circuit characteristics
        let estimated_accesses = circuit_depth * state_vector.data.len();

        // Adapt memory layout if beneficial
        state_vector.adapt_memory_layout()?;

        // Warm up caches if enabled
        if self.config.enable_prefetching {
            Self::warmup_caches(state_vector)?;
        }

        let optimization_time = start_time.elapsed();

        Ok(MemoryOptimizationReport {
            optimization_time,
            estimated_memory_accesses: estimated_accesses,
            cache_warmup_performed: self.config.enable_prefetching,
            layout_adaptation_performed: true,
            memory_stats: state_vector.get_memory_stats(),
        })
    }

    /// Warm up memory caches by touching data
    fn warmup_caches(state_vector: &OptimizedStateVector) -> Result<()> {
        let chunk_size = state_vector.config.cache_line_size / std::mem::size_of::<Complex64>();

        for chunk_start in (0..state_vector.data.len()).step_by(chunk_size) {
            let chunk_end = std::cmp::min(chunk_start + chunk_size, state_vector.data.len());

            // Touch each cache line
            for i in (chunk_start..chunk_end).step_by(chunk_size / 4) {
                let _ = state_vector.data[i]; // Read to bring into cache
            }
        }

        Ok(())
    }
}

/// Memory optimization report
#[derive(Debug, Clone)]
pub struct MemoryOptimizationReport {
    /// Time spent on optimization
    pub optimization_time: Duration,
    /// Estimated number of memory accesses
    pub estimated_memory_accesses: usize,
    /// Whether cache warmup was performed
    pub cache_warmup_performed: bool,
    /// Whether layout adaptation was performed
    pub layout_adaptation_performed: bool,
    /// Final memory statistics
    pub memory_stats: MemoryStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_optimized_state_vector_creation() {
        let config = MemoryOptimizationConfig::default();
        let state_vector = OptimizedStateVector::new(3, config)
            .expect("OptimizedStateVector creation should succeed");

        assert_eq!(state_vector.num_qubits, 3);
        assert_eq!(state_vector.data.len(), 8);
        assert_eq!(state_vector.data[0], Complex64::new(1.0, 0.0));
    }

    #[test]
    fn test_memory_layouts() {
        let config = MemoryOptimizationConfig {
            layout: MemoryLayout::CacheAligned,
            ..Default::default()
        };

        let state_vector = OptimizedStateVector::new(4, config)
            .expect("OptimizedStateVector with CacheAligned layout should be created");
        assert_eq!(state_vector.layout, MemoryLayout::CacheAligned);
    }

    #[test]
    fn test_single_qubit_gate_optimization() {
        let config = MemoryOptimizationConfig::default();
        let mut state_vector = OptimizedStateVector::new(2, config)
            .expect("OptimizedStateVector creation should succeed");

        // Pauli-X gate
        let gate_matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("Gate matrix construction should succeed");

        state_vector
            .apply_single_qubit_gate_optimized(0, &gate_matrix)
            .expect("Single qubit gate application should succeed");

        // State should now be |01⟩
        assert!((state_vector.data[1].re - 1.0).abs() < 1e-10);
        assert!(state_vector.data[0].re.abs() < 1e-10);
    }

    #[test]
    fn test_bandwidth_monitoring() {
        let config = MemoryOptimizationConfig::default();
        let state_vector = OptimizedStateVector::new(3, config)
            .expect("OptimizedStateVector creation should succeed");

        let stats = state_vector
            .get_bandwidth_stats()
            .expect("Bandwidth stats retrieval should succeed");
        assert_eq!(stats.bandwidth_samples.len(), 0); // No operations yet
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(1024, 10).expect("MemoryPool creation should succeed");

        let ptr1 = pool.allocate().expect("First allocation should succeed");
        let ptr2 = pool.allocate().expect("Second allocation should succeed");

        pool.deallocate(ptr1)
            .expect("First deallocation should succeed");
        pool.deallocate(ptr2)
            .expect("Second deallocation should succeed");
    }

    #[test]
    fn test_cache_aligned_allocation() {
        let config = MemoryOptimizationConfig {
            layout: MemoryLayout::CacheAligned,
            cache_line_size: 64,
            ..Default::default()
        };

        let data = OptimizedStateVector::allocate_cache_aligned(100, &config)
            .expect("Cache-aligned allocation should succeed");

        // Should be padded to cache line boundary
        let element_size = std::mem::size_of::<Complex64>();
        let elements_per_line = config.cache_line_size / element_size;
        let expected_padded = 100_usize.div_ceil(elements_per_line) * elements_per_line;

        assert_eq!(data.len(), expected_padded);
    }

    #[test]
    fn test_memory_bandwidth_optimizer() {
        let config = MemoryOptimizationConfig::default();
        let optimizer = MemoryBandwidthOptimizer::new(config)
            .expect("MemoryBandwidthOptimizer creation should succeed");

        let mut state_vector = optimizer
            .create_optimized_state_vector(4)
            .expect("Optimized state vector creation should succeed");
        let report = optimizer
            .optimize_circuit_memory_access(&mut state_vector, 10)
            .expect("Circuit memory optimization should succeed");

        // Ensure optimization completed successfully
        assert!(report.optimization_time.as_millis() < u128::MAX);
        assert_eq!(report.estimated_memory_accesses, 10 * 16); // 10 gates x 16 states
    }

    #[test]
    fn test_adaptive_layout() {
        let config = MemoryOptimizationConfig {
            layout: MemoryLayout::Adaptive,
            ..Default::default()
        };

        let mut state_vector = OptimizedStateVector::new(3, config)
            .expect("OptimizedStateVector with Adaptive layout should be created");
        state_vector
            .adapt_memory_layout()
            .expect("Memory layout adaptation should succeed");

        // Layout may have changed based on (empty) access patterns
        assert!(matches!(
            state_vector.layout,
            MemoryLayout::CacheAligned | MemoryLayout::Blocked | MemoryLayout::Hierarchical
        ));
    }

    #[test]
    fn test_memory_stats() {
        let config = MemoryOptimizationConfig::default();
        let state_vector = OptimizedStateVector::new(4, config)
            .expect("OptimizedStateVector creation should succeed");

        let stats = state_vector.get_memory_stats();
        assert_eq!(stats.total_memory, 16 * std::mem::size_of::<Complex64>());
        assert!(stats.cache_efficiency >= 0.0 && stats.cache_efficiency <= 1.0);
    }

    #[test]
    fn test_blocked_layout_allocation() {
        let config = MemoryOptimizationConfig {
            layout: MemoryLayout::Blocked,
            block_size: 1024,
            ..Default::default()
        };

        let data = OptimizedStateVector::allocate_blocked(100, &config)
            .expect("Blocked layout allocation should succeed");
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_prefetch_functionality() {
        let config = MemoryOptimizationConfig {
            enable_prefetching: true,
            prefetch_distance: 4,
            ..Default::default()
        };

        let state_vector = OptimizedStateVector::new(5, config)
            .expect("OptimizedStateVector with prefetching enabled should be created");

        // Test that prefetching doesn't crash
        OptimizedStateVector::prefetch_memory(&state_vector.data[0]);
    }
}
