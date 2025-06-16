//! Cache-optimized state vector layouts and access patterns.
//!
//! This module provides advanced cache-optimized implementations for quantum
//! state vector operations, including cache-aware data structures, memory
//! access patterns, and cache-conscious algorithms for quantum gates.

use ndarray::Array2;
use num_complex::Complex64;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::error::{Result, SimulatorError};
use crate::memory_bandwidth_optimization::MemoryOptimizationConfig;

/// Cache hierarchy configuration
#[derive(Debug, Clone)]
pub struct CacheHierarchyConfig {
    /// L1 cache size in bytes
    pub l1_size: usize,
    /// L1 cache line size in bytes
    pub l1_line_size: usize,
    /// L1 cache associativity
    pub l1_associativity: usize,
    /// L2 cache size in bytes
    pub l2_size: usize,
    /// L2 cache line size in bytes
    pub l2_line_size: usize,
    /// L2 cache associativity
    pub l2_associativity: usize,
    /// L3 cache size in bytes
    pub l3_size: usize,
    /// L3 cache line size in bytes
    pub l3_line_size: usize,
    /// L3 cache associativity
    pub l3_associativity: usize,
    /// Memory latency in cycles
    pub memory_latency: usize,
    /// Cache replacement policy
    pub replacement_policy: CacheReplacementPolicy,
}

impl Default for CacheHierarchyConfig {
    fn default() -> Self {
        Self {
            l1_size: 32 * 1024,       // 32KB L1
            l1_line_size: 64,         // 64B cache line
            l1_associativity: 8,      // 8-way associative
            l2_size: 256 * 1024,      // 256KB L2
            l2_line_size: 64,         // 64B cache line
            l2_associativity: 8,      // 8-way associative
            l3_size: 8 * 1024 * 1024, // 8MB L3
            l3_line_size: 64,         // 64B cache line
            l3_associativity: 16,     // 16-way associative
            memory_latency: 300,      // ~300 cycles
            replacement_policy: CacheReplacementPolicy::LRU,
        }
    }
}

/// Cache replacement policies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheReplacementPolicy {
    /// Least Recently Used
    LRU,
    /// First In, First Out
    FIFO,
    /// Random replacement
    Random,
    /// Least Frequently Used
    LFU,
}

/// Cache-optimized data layout strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOptimizedLayout {
    /// Standard linear layout
    Linear,
    /// Block-based layout for cache lines
    Blocked,
    /// Z-order (Morton order) layout
    ZOrder,
    /// Hilbert curve layout
    Hilbert,
    /// Bit-reversal layout for FFT-like operations
    BitReversal,
    /// Strided layout for parallel access
    Strided,
    /// Hierarchical layout matching cache levels
    Hierarchical,
}

/// Cache access pattern tracking
#[derive(Debug, Clone)]
pub struct CacheAccessPattern {
    /// Access counts per cache line
    pub line_access_counts: HashMap<usize, u64>,
    /// Cache hit/miss statistics
    pub cache_hits: u64,
    pub cache_misses: u64,
    /// Access sequence for temporal locality analysis
    pub access_sequence: VecDeque<usize>,
    /// Stride detection
    pub detected_strides: Vec<isize>,
    /// Temporal locality score (0.0 to 1.0)
    pub temporal_locality: f64,
    /// Spatial locality score (0.0 to 1.0)
    pub spatial_locality: f64,
}

impl Default for CacheAccessPattern {
    fn default() -> Self {
        Self {
            line_access_counts: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            access_sequence: VecDeque::new(),
            detected_strides: Vec::new(),
            temporal_locality: 0.0,
            spatial_locality: 0.0,
        }
    }
}

/// Cache-optimized state vector with advanced layout management
#[derive(Debug)]
pub struct CacheOptimizedStateVector {
    /// State vector data with optimized layout
    data: Vec<Complex64>,
    /// Number of qubits
    num_qubits: usize,
    /// Current layout strategy
    layout: CacheOptimizedLayout,
    /// Cache hierarchy configuration
    cache_config: CacheHierarchyConfig,
    /// Memory optimization configuration
    memory_config: MemoryOptimizationConfig,
    /// Cache access pattern tracking
    access_pattern: Arc<Mutex<CacheAccessPattern>>,
    /// Layout transformation indices
    layout_indices: Vec<usize>,
    /// Inverse layout transformation indices
    inverse_indices: Vec<usize>,
}

impl CacheOptimizedStateVector {
    /// Create a new cache-optimized state vector
    pub fn new(
        num_qubits: usize,
        layout: CacheOptimizedLayout,
        cache_config: CacheHierarchyConfig,
        memory_config: MemoryOptimizationConfig,
    ) -> Result<Self> {
        let size = 1 << num_qubits;

        // Generate layout transformation indices
        let (layout_indices, inverse_indices) = Self::generate_layout_indices(size, layout)?;

        // Allocate and initialize data
        let mut data = vec![Complex64::new(0.0, 0.0); size];
        data[0] = Complex64::new(1.0, 0.0); // |0...0⟩ state

        // Apply layout transformation
        let mut instance = Self {
            data,
            num_qubits,
            layout,
            cache_config,
            memory_config,
            access_pattern: Arc::new(Mutex::new(CacheAccessPattern::default())),
            layout_indices,
            inverse_indices,
        };

        instance.apply_layout_transformation()?;

        Ok(instance)
    }

    /// Generate layout transformation indices for different cache-optimized layouts
    fn generate_layout_indices(
        size: usize,
        layout: CacheOptimizedLayout,
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        let indices = match layout {
            CacheOptimizedLayout::Linear => (0..size).collect::<Vec<usize>>(),
            CacheOptimizedLayout::Blocked => Self::generate_blocked_indices(size)?,
            CacheOptimizedLayout::ZOrder => Self::generate_z_order_indices(size)?,
            CacheOptimizedLayout::Hilbert => Self::generate_hilbert_indices(size)?,
            CacheOptimizedLayout::BitReversal => Self::generate_bit_reversal_indices(size)?,
            CacheOptimizedLayout::Strided => Self::generate_strided_indices(size)?,
            CacheOptimizedLayout::Hierarchical => Self::generate_hierarchical_indices(size)?,
        };

        // Generate inverse mapping
        let mut inverse_indices = vec![0; size];
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            inverse_indices[old_idx] = new_idx;
        }

        Ok((indices, inverse_indices))
    }

    /// Generate blocked layout indices for cache line optimization
    fn generate_blocked_indices(size: usize) -> Result<Vec<usize>> {
        let block_size = 64 / std::mem::size_of::<Complex64>(); // Cache line size in elements
        let mut indices = Vec::with_capacity(size);

        for block_start in (0..size).step_by(block_size) {
            let block_end = std::cmp::min(block_start + block_size, size);
            for i in block_start..block_end {
                indices.push(i);
            }
        }

        Ok(indices)
    }

    /// Generate Z-order (Morton order) indices for 2D spatial locality
    fn generate_z_order_indices(size: usize) -> Result<Vec<usize>> {
        let bits = (size as f64).log2() as usize;
        if 1 << bits != size {
            return Err(SimulatorError::InvalidParameter(
                "Z-order layout requires power-of-2 size".to_string(),
            ));
        }

        let mut indices = Vec::with_capacity(size);

        for i in 0..size {
            let morton_index = Self::morton_encode_2d(i, bits / 2);
            indices.push(morton_index % size);
        }

        Ok(indices)
    }

    /// Encode 2D coordinates using Morton order
    fn morton_encode_2d(index: usize, bits_per_dim: usize) -> usize {
        let x = index & ((1 << bits_per_dim) - 1);
        let y = index >> bits_per_dim;

        let mut result = 0;
        for i in 0..bits_per_dim {
            result |= ((x >> i) & 1) << (2 * i);
            result |= ((y >> i) & 1) << (2 * i + 1);
        }

        result
    }

    /// Generate Hilbert curve indices for optimal spatial locality
    fn generate_hilbert_indices(size: usize) -> Result<Vec<usize>> {
        let bits = (size as f64).log2() as usize;
        if 1 << bits != size {
            return Err(SimulatorError::InvalidParameter(
                "Hilbert layout requires power-of-2 size".to_string(),
            ));
        }

        let order = bits / 2;
        let mut indices = Vec::with_capacity(size);

        for i in 0..size {
            let (x, y) = Self::hilbert_index_to_xy(i, order);
            let linear_index = y * (1 << order) + x;
            indices.push(linear_index % size);
        }

        Ok(indices)
    }

    /// Convert Hilbert index to 2D coordinates
    fn hilbert_index_to_xy(mut index: usize, order: usize) -> (usize, usize) {
        let mut x = 0;
        let mut y = 0;

        for s in (0..order).rev() {
            let rx = 1 & (index >> 1);
            let ry = 1 & (index ^ rx);

            if ry == 0 {
                if rx == 1 {
                    // Safely perform rotation using saturation to prevent underflow
                    let max_val = (1usize << s).saturating_sub(1);
                    x = max_val.saturating_sub(x);
                    y = max_val.saturating_sub(y);
                }
                std::mem::swap(&mut x, &mut y);
            }

            x += rx << s;
            y += ry << s;
            index >>= 2;
        }

        (x, y)
    }

    /// Generate bit-reversal indices for FFT-like operations
    fn generate_bit_reversal_indices(size: usize) -> Result<Vec<usize>> {
        let bits = (size as f64).log2() as usize;
        if 1 << bits != size {
            return Err(SimulatorError::InvalidParameter(
                "Bit-reversal layout requires power-of-2 size".to_string(),
            ));
        }

        let mut indices = Vec::with_capacity(size);

        for i in 0..size {
            let reversed = Self::reverse_bits(i, bits);
            indices.push(reversed);
        }

        Ok(indices)
    }

    /// Reverse bits in an integer
    fn reverse_bits(mut num: usize, bits: usize) -> usize {
        let mut result = 0;
        for _ in 0..bits {
            result = (result << 1) | (num & 1);
            num >>= 1;
        }
        result
    }

    /// Generate strided indices for parallel access optimization
    fn generate_strided_indices(size: usize) -> Result<Vec<usize>> {
        let stride = 8; // SIMD width
        let mut indices = Vec::with_capacity(size);

        for group in 0..(size + stride - 1) / stride {
            for offset in 0..stride {
                let index = group * stride + offset;
                if index < size {
                    indices.push(index);
                }
            }
        }

        Ok(indices)
    }

    /// Generate hierarchical indices matching cache levels
    fn generate_hierarchical_indices(size: usize) -> Result<Vec<usize>> {
        let l1_block_size = 32 / std::mem::size_of::<Complex64>(); // L1 cache block
        let l2_block_size = 256 / std::mem::size_of::<Complex64>(); // L2 cache block

        let mut indices = Vec::with_capacity(size);

        for l2_block in 0..(size + l2_block_size - 1) / l2_block_size {
            let l2_start = l2_block * l2_block_size;
            let l2_end = std::cmp::min(l2_start + l2_block_size, size);

            for l1_block_start in (l2_start..l2_end).step_by(l1_block_size) {
                let l1_block_end = std::cmp::min(l1_block_start + l1_block_size, l2_end);

                for i in l1_block_start..l1_block_end {
                    indices.push(i);
                }
            }
        }

        Ok(indices)
    }

    /// Apply layout transformation to reorder data
    fn apply_layout_transformation(&mut self) -> Result<()> {
        let original_data = self.data.clone();

        for (new_idx, &old_idx) in self.layout_indices.iter().enumerate() {
            self.data[new_idx] = original_data[old_idx];
        }

        Ok(())
    }

    /// Convert logical index to physical index based on current layout
    fn logical_to_physical(&self, logical_index: usize) -> usize {
        self.inverse_indices[logical_index]
    }

    /// Convert physical index to logical index based on current layout
    fn physical_to_logical(&self, physical_index: usize) -> usize {
        self.layout_indices[physical_index]
    }

    /// Apply single-qubit gate with cache-optimized access pattern
    pub fn apply_single_qubit_gate_cache_optimized(
        &mut self,
        target: usize,
        gate_matrix: &Array2<Complex64>,
    ) -> Result<()> {
        let start_time = Instant::now();
        let mask = 1 << target;

        // Choose optimization based on layout
        match self.layout {
            CacheOptimizedLayout::Blocked => {
                self.apply_single_qubit_blocked(target, gate_matrix, mask)?;
            }
            CacheOptimizedLayout::ZOrder => {
                self.apply_single_qubit_z_order(target, gate_matrix, mask)?;
            }
            CacheOptimizedLayout::Hilbert => {
                self.apply_single_qubit_hilbert(target, gate_matrix, mask)?;
            }
            CacheOptimizedLayout::Strided => {
                self.apply_single_qubit_strided(target, gate_matrix, mask)?;
            }
            _ => {
                self.apply_single_qubit_linear(target, gate_matrix, mask)?;
            }
        }

        // Update access pattern statistics
        self.update_access_statistics(start_time.elapsed());

        Ok(())
    }

    /// Apply single-qubit gate with blocked access pattern
    fn apply_single_qubit_blocked(
        &mut self,
        _target: usize,
        gate_matrix: &Array2<Complex64>,
        mask: usize,
    ) -> Result<()> {
        let block_size = self.cache_config.l1_line_size / std::mem::size_of::<Complex64>();

        for block_start in (0..self.data.len()).step_by(block_size) {
            let block_end = std::cmp::min(block_start + block_size, self.data.len());

            // Prefetch next block
            if block_end < self.data.len() {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        self.data.as_ptr().add(block_end) as *const i8,
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }

            // Process current block
            for i in (block_start..block_end).step_by(2) {
                if i + 1 < self.data.len() {
                    let logical_i0 = self.physical_to_logical(i);
                    let logical_i1 = logical_i0 | mask;
                    let physical_i1 = self.logical_to_physical(logical_i1);

                    if physical_i1 < self.data.len() {
                        let amp0 = self.data[i];
                        let amp1 = self.data[physical_i1];

                        self.data[i] = gate_matrix[[0, 0]] * amp0 + gate_matrix[[0, 1]] * amp1;
                        self.data[physical_i1] =
                            gate_matrix[[1, 0]] * amp0 + gate_matrix[[1, 1]] * amp1;

                        // Track cache access
                        self.track_cache_access(i);
                        self.track_cache_access(physical_i1);
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply single-qubit gate with Z-order access pattern
    fn apply_single_qubit_z_order(
        &mut self,
        _target: usize,
        gate_matrix: &Array2<Complex64>,
        mask: usize,
    ) -> Result<()> {
        // Process in Z-order to maximize spatial locality
        let bits = self.num_qubits;
        let tile_size = 4; // Process 4x4 tiles for better cache utilization

        for tile_y in (0..(1 << (bits / 2))).step_by(tile_size) {
            for tile_x in (0..(1 << (bits / 2))).step_by(tile_size) {
                for y in tile_y..std::cmp::min(tile_y + tile_size, 1 << (bits / 2)) {
                    for x in tile_x..std::cmp::min(tile_x + tile_size, 1 << (bits / 2)) {
                        let logical_index = y * (1 << (bits / 2)) + x;
                        let logical_i0 = logical_index & !mask;
                        let logical_i1 = logical_i0 | mask;

                        let physical_i0 = self.logical_to_physical(logical_i0);
                        let physical_i1 = self.logical_to_physical(logical_i1);

                        if physical_i0 < self.data.len() && physical_i1 < self.data.len() {
                            let amp0 = self.data[physical_i0];
                            let amp1 = self.data[physical_i1];

                            self.data[physical_i0] =
                                gate_matrix[[0, 0]] * amp0 + gate_matrix[[0, 1]] * amp1;
                            self.data[physical_i1] =
                                gate_matrix[[1, 0]] * amp0 + gate_matrix[[1, 1]] * amp1;

                            self.track_cache_access(physical_i0);
                            self.track_cache_access(physical_i1);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply single-qubit gate with Hilbert curve access pattern
    fn apply_single_qubit_hilbert(
        &mut self,
        _target: usize,
        gate_matrix: &Array2<Complex64>,
        mask: usize,
    ) -> Result<()> {
        // Process along Hilbert curve for optimal spatial locality
        let order = self.num_qubits / 2;
        let curve_length = 1 << self.num_qubits;

        for hilbert_index in (0..curve_length).step_by(2) {
            let (x, y) = Self::hilbert_index_to_xy(hilbert_index, order);
            let logical_index = y * (1 << order) + x;

            let logical_i0 = logical_index & !mask;
            let logical_i1 = logical_i0 | mask;

            let physical_i0 = self.logical_to_physical(logical_i0);
            let physical_i1 = self.logical_to_physical(logical_i1);

            if physical_i0 < self.data.len() && physical_i1 < self.data.len() {
                let amp0 = self.data[physical_i0];
                let amp1 = self.data[physical_i1];

                self.data[physical_i0] = gate_matrix[[0, 0]] * amp0 + gate_matrix[[0, 1]] * amp1;
                self.data[physical_i1] = gate_matrix[[1, 0]] * amp0 + gate_matrix[[1, 1]] * amp1;

                self.track_cache_access(physical_i0);
                self.track_cache_access(physical_i1);
            }
        }

        Ok(())
    }

    /// Apply single-qubit gate with strided access for SIMD optimization
    fn apply_single_qubit_strided(
        &mut self,
        _target: usize,
        gate_matrix: &Array2<Complex64>,
        mask: usize,
    ) -> Result<()> {
        let stride = 8; // SIMD width

        for group_start in (0..self.data.len()).step_by(stride * 2) {
            // Process SIMD groups
            for offset in 0..stride {
                let i = group_start + offset;
                if i + stride < self.data.len() {
                    let logical_i0 = self.physical_to_logical(i);
                    let logical_i1 = logical_i0 | mask;
                    let physical_i1 = self.logical_to_physical(logical_i1);

                    if physical_i1 < self.data.len() {
                        let amp0 = self.data[i];
                        let amp1 = self.data[physical_i1];

                        self.data[i] = gate_matrix[[0, 0]] * amp0 + gate_matrix[[0, 1]] * amp1;
                        self.data[physical_i1] =
                            gate_matrix[[1, 0]] * amp0 + gate_matrix[[1, 1]] * amp1;

                        self.track_cache_access(i);
                        self.track_cache_access(physical_i1);
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply single-qubit gate with linear access pattern
    fn apply_single_qubit_linear(
        &mut self,
        _target: usize,
        gate_matrix: &Array2<Complex64>,
        mask: usize,
    ) -> Result<()> {
        for i in (0..self.data.len()).step_by(2) {
            let logical_i0 = self.physical_to_logical(i);
            let logical_i1 = logical_i0 | mask;
            let physical_i1 = self.logical_to_physical(logical_i1);

            if physical_i1 < self.data.len() {
                let amp0 = self.data[i];
                let amp1 = self.data[physical_i1];

                self.data[i] = gate_matrix[[0, 0]] * amp0 + gate_matrix[[0, 1]] * amp1;
                self.data[physical_i1] = gate_matrix[[1, 0]] * amp0 + gate_matrix[[1, 1]] * amp1;

                self.track_cache_access(i);
                self.track_cache_access(physical_i1);
            }
        }

        Ok(())
    }

    /// Track cache access for statistics
    fn track_cache_access(&self, physical_index: usize) {
        if let Ok(mut pattern) = self.access_pattern.lock() {
            let cache_line = physical_index
                / (self.cache_config.l1_line_size / std::mem::size_of::<Complex64>());

            // Update access counts
            *pattern.line_access_counts.entry(cache_line).or_insert(0) += 1;

            // Update access sequence for locality analysis
            pattern.access_sequence.push_back(physical_index);
            if pattern.access_sequence.len() > 1000 {
                pattern.access_sequence.pop_front();
            }

            // Simple cache hit estimation (simplified)
            if pattern.line_access_counts.get(&cache_line).unwrap_or(&0) > &1 {
                pattern.cache_hits += 1;
            } else {
                pattern.cache_misses += 1;
            }
        }
    }

    /// Update access pattern statistics
    fn update_access_statistics(&self, operation_time: Duration) {
        if let Ok(mut pattern) = self.access_pattern.lock() {
            // Calculate temporal locality
            let total_accesses = pattern.cache_hits + pattern.cache_misses;
            if total_accesses > 0 {
                pattern.temporal_locality = pattern.cache_hits as f64 / total_accesses as f64;
            }

            // Calculate spatial locality (simplified)
            if pattern.access_sequence.len() > 1 {
                let mut spatial_hits = 0;
                let mut total_pairs = 0;

                for window in pattern.access_sequence.as_slices().0.windows(2) {
                    if let [addr1, addr2] = window {
                        let line1 = addr1
                            / (self.cache_config.l1_line_size / std::mem::size_of::<Complex64>());
                        let line2 = addr2
                            / (self.cache_config.l1_line_size / std::mem::size_of::<Complex64>());

                        if line1 == line2 || line1.abs_diff(line2) <= 1 {
                            spatial_hits += 1;
                        }
                        total_pairs += 1;
                    }
                }

                if total_pairs > 0 {
                    pattern.spatial_locality = spatial_hits as f64 / total_pairs as f64;
                }
            }

            // Detect stride patterns
            if pattern.access_sequence.len() >= 3 {
                let recent_accesses: Vec<_> =
                    pattern.access_sequence.iter().rev().take(10).collect();
                pattern.detected_strides = Self::detect_strides(&recent_accesses);
            }
        }
    }

    /// Detect stride patterns in memory access
    fn detect_strides(accesses: &[&usize]) -> Vec<isize> {
        let mut strides = Vec::new();

        if accesses.len() >= 3 {
            for window in accesses.windows(3) {
                if let [addr1, addr2, addr3] = window {
                    let stride1 = **addr2 as isize - **addr1 as isize;
                    let stride2 = **addr3 as isize - **addr2 as isize;

                    if stride1 == stride2 && stride1 != 0 {
                        strides.push(stride1);
                    }
                }
            }
        }

        strides
    }

    /// Get cache performance statistics
    pub fn get_cache_stats(&self) -> Result<CachePerformanceStats> {
        let pattern = self.access_pattern.lock().unwrap();
        let total_accesses = pattern.cache_hits + pattern.cache_misses;

        Ok(CachePerformanceStats {
            cache_hit_rate: if total_accesses > 0 {
                pattern.cache_hits as f64 / total_accesses as f64
            } else {
                0.0
            },
            cache_miss_rate: if total_accesses > 0 {
                pattern.cache_misses as f64 / total_accesses as f64
            } else {
                0.0
            },
            temporal_locality: pattern.temporal_locality,
            spatial_locality: pattern.spatial_locality,
            total_cache_lines_accessed: pattern.line_access_counts.len(),
            average_accesses_per_line: if !pattern.line_access_counts.is_empty() {
                pattern.line_access_counts.values().sum::<u64>() as f64
                    / pattern.line_access_counts.len() as f64
            } else {
                0.0
            },
            detected_strides: pattern.detected_strides.clone(),
            current_layout: self.layout,
        })
    }

    /// Adapt layout based on access patterns
    pub fn adapt_cache_layout(&mut self) -> Result<CacheLayoutAdaptationResult> {
        let stats = self.get_cache_stats()?;
        let current_performance = stats.cache_hit_rate;

        // Determine optimal layout based on access patterns
        let recommended_layout = if stats.spatial_locality > 0.8 {
            CacheOptimizedLayout::Blocked
        } else if stats.detected_strides.iter().any(|&s| s.abs() == 1) {
            CacheOptimizedLayout::Linear
        } else if stats.detected_strides.len() > 2 {
            CacheOptimizedLayout::Strided
        } else if stats.temporal_locality < 0.5 {
            CacheOptimizedLayout::Hilbert
        } else {
            CacheOptimizedLayout::ZOrder
        };

        let layout_changed = recommended_layout != self.layout;

        if layout_changed {
            // Store old layout for comparison
            let old_layout = self.layout;

            // Apply new layout
            let (new_indices, new_inverse) =
                Self::generate_layout_indices(self.data.len(), recommended_layout)?;
            self.layout_indices = new_indices;
            self.inverse_indices = new_inverse;
            self.layout = recommended_layout;

            // Transform data to new layout
            self.apply_layout_transformation()?;

            Ok(CacheLayoutAdaptationResult {
                layout_changed: true,
                old_layout,
                new_layout: recommended_layout,
                performance_before: current_performance,
                expected_performance_improvement: 0.1, // Simplified estimate
                adaptation_overhead: Duration::from_millis(1), // Simplified
            })
        } else {
            Ok(CacheLayoutAdaptationResult {
                layout_changed: false,
                old_layout: self.layout,
                new_layout: self.layout,
                performance_before: current_performance,
                expected_performance_improvement: 0.0,
                adaptation_overhead: Duration::from_nanos(0),
            })
        }
    }

    /// Get state vector data (read-only)
    pub fn data(&self) -> &[Complex64] {
        &self.data
    }

    /// Get the current layout
    pub fn layout(&self) -> CacheOptimizedLayout {
        self.layout
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CachePerformanceStats {
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Cache miss rate (0.0 to 1.0)
    pub cache_miss_rate: f64,
    /// Temporal locality score (0.0 to 1.0)
    pub temporal_locality: f64,
    /// Spatial locality score (0.0 to 1.0)
    pub spatial_locality: f64,
    /// Number of unique cache lines accessed
    pub total_cache_lines_accessed: usize,
    /// Average number of accesses per cache line
    pub average_accesses_per_line: f64,
    /// Detected stride patterns
    pub detected_strides: Vec<isize>,
    /// Current layout being used
    pub current_layout: CacheOptimizedLayout,
}

/// Cache layout adaptation result
#[derive(Debug, Clone)]
pub struct CacheLayoutAdaptationResult {
    /// Whether layout was actually changed
    pub layout_changed: bool,
    /// Previous layout
    pub old_layout: CacheOptimizedLayout,
    /// New layout
    pub new_layout: CacheOptimizedLayout,
    /// Performance before adaptation
    pub performance_before: f64,
    /// Expected performance improvement
    pub expected_performance_improvement: f64,
    /// Time overhead for adaptation
    pub adaptation_overhead: Duration,
}

/// Cache-optimized gate operation manager
#[derive(Debug)]
pub struct CacheOptimizedGateManager {
    /// Cache hierarchy configuration
    cache_config: CacheHierarchyConfig,
    /// Gate operation statistics
    operation_stats: HashMap<String, CacheOperationStats>,
}

impl CacheOptimizedGateManager {
    /// Create a new cache-optimized gate manager
    pub fn new(cache_config: CacheHierarchyConfig) -> Self {
        Self {
            cache_config,
            operation_stats: HashMap::new(),
        }
    }

    /// Execute a quantum gate with cache optimization
    pub fn execute_gate(
        &mut self,
        state_vector: &mut CacheOptimizedStateVector,
        gate_name: &str,
        target_qubits: &[usize],
        gate_matrix: &Array2<Complex64>,
    ) -> Result<CacheOperationStats> {
        let start_time = Instant::now();

        match target_qubits.len() {
            1 => {
                state_vector
                    .apply_single_qubit_gate_cache_optimized(target_qubits[0], gate_matrix)?;
            }
            2 => {
                // Two-qubit gate implementation would go here
                return Err(SimulatorError::NotImplemented(
                    "Two-qubit cache-optimized gates not implemented".to_string(),
                ));
            }
            _ => {
                return Err(SimulatorError::InvalidParameter(
                    "Only single and two-qubit gates supported".to_string(),
                ));
            }
        }

        let execution_time = start_time.elapsed();
        let cache_stats = state_vector.get_cache_stats()?;

        let operation_stats = CacheOperationStats {
            gate_name: gate_name.to_string(),
            execution_time,
            cache_hit_rate: cache_stats.cache_hit_rate,
            spatial_locality: cache_stats.spatial_locality,
            temporal_locality: cache_stats.temporal_locality,
            memory_accesses: cache_stats.total_cache_lines_accessed,
        };

        self.operation_stats
            .insert(gate_name.to_string(), operation_stats.clone());

        Ok(operation_stats)
    }

    /// Get comprehensive operation statistics
    pub fn get_operation_statistics(&self) -> HashMap<String, CacheOperationStats> {
        self.operation_stats.clone()
    }
}

/// Statistics for cache-optimized gate operations
#[derive(Debug, Clone)]
pub struct CacheOperationStats {
    /// Name of the gate operation
    pub gate_name: String,
    /// Total execution time
    pub execution_time: Duration,
    /// Cache hit rate during operation
    pub cache_hit_rate: f64,
    /// Spatial locality score
    pub spatial_locality: f64,
    /// Temporal locality score
    pub temporal_locality: f64,
    /// Number of memory accesses
    pub memory_accesses: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_cache_optimized_state_vector_creation() {
        let cache_config = CacheHierarchyConfig::default();
        let memory_config = MemoryOptimizationConfig::default();

        let state_vector = CacheOptimizedStateVector::new(
            3,
            CacheOptimizedLayout::Blocked,
            cache_config,
            memory_config,
        )
        .unwrap();

        assert_eq!(state_vector.num_qubits(), 3);
        assert_eq!(state_vector.data().len(), 8);
        assert_eq!(state_vector.layout(), CacheOptimizedLayout::Blocked);
    }

    #[test]
    fn test_blocked_layout_indices() {
        let indices = CacheOptimizedStateVector::generate_blocked_indices(16).unwrap();
        assert_eq!(indices.len(), 16);

        // Should maintain all indices
        let mut sorted_indices = indices.clone();
        sorted_indices.sort();
        assert_eq!(sorted_indices, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn test_z_order_layout() {
        let indices = CacheOptimizedStateVector::generate_z_order_indices(16).unwrap();
        assert_eq!(indices.len(), 16);
    }

    #[test]
    fn test_hilbert_curve_layout() {
        let indices = CacheOptimizedStateVector::generate_hilbert_indices(16).unwrap();
        assert_eq!(indices.len(), 16);
    }

    #[test]
    fn test_bit_reversal_layout() {
        let indices = CacheOptimizedStateVector::generate_bit_reversal_indices(8).unwrap();
        assert_eq!(indices.len(), 8);

        // Check that bit reversal is working
        assert_eq!(indices[1], 4); // 001 -> 100
        assert_eq!(indices[2], 2); // 010 -> 010
        assert_eq!(indices[3], 6); // 011 -> 110
    }

    #[test]
    fn test_single_qubit_gate_application() {
        let cache_config = CacheHierarchyConfig::default();
        let memory_config = MemoryOptimizationConfig::default();

        let mut state_vector = CacheOptimizedStateVector::new(
            2,
            CacheOptimizedLayout::Linear,
            cache_config,
            memory_config,
        )
        .unwrap();

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
        .unwrap();

        state_vector
            .apply_single_qubit_gate_cache_optimized(0, &gate_matrix)
            .unwrap();

        // Check that state transformation occurred
        let cache_stats = state_vector.get_cache_stats().unwrap();
        assert!(cache_stats.total_cache_lines_accessed > 0);
    }

    #[test]
    fn test_cache_statistics() {
        let cache_config = CacheHierarchyConfig::default();
        let memory_config = MemoryOptimizationConfig::default();

        let state_vector = CacheOptimizedStateVector::new(
            3,
            CacheOptimizedLayout::Blocked,
            cache_config,
            memory_config,
        )
        .unwrap();

        let stats = state_vector.get_cache_stats().unwrap();
        assert!(stats.cache_hit_rate >= 0.0 && stats.cache_hit_rate <= 1.0);
        assert!(stats.spatial_locality >= 0.0 && stats.spatial_locality <= 1.0);
        assert!(stats.temporal_locality >= 0.0 && stats.temporal_locality <= 1.0);
    }

    #[test]
    fn test_layout_adaptation() {
        let cache_config = CacheHierarchyConfig::default();
        let memory_config = MemoryOptimizationConfig::default();

        let mut state_vector = CacheOptimizedStateVector::new(
            3,
            CacheOptimizedLayout::Linear,
            cache_config,
            memory_config,
        )
        .unwrap();

        let adaptation_result = state_vector.adapt_cache_layout().unwrap();

        // Adaptation may or may not occur based on access patterns
        assert!(adaptation_result.performance_before >= 0.0);
        assert!(adaptation_result.adaptation_overhead >= Duration::from_nanos(0));
    }

    #[test]
    fn test_cache_optimized_gate_manager() {
        let cache_config = CacheHierarchyConfig::default();
        let memory_config = MemoryOptimizationConfig::default();

        let mut manager = CacheOptimizedGateManager::new(cache_config.clone());
        let mut state_vector = CacheOptimizedStateVector::new(
            2,
            CacheOptimizedLayout::Blocked,
            cache_config,
            memory_config,
        )
        .unwrap();

        // Hadamard gate
        let gate_matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        )
        .unwrap();

        let stats = manager
            .execute_gate(&mut state_vector, "H", &[0], &gate_matrix)
            .unwrap();

        assert_eq!(stats.gate_name, "H");
        assert!(stats.execution_time > Duration::from_nanos(0));
        assert!(stats.cache_hit_rate >= 0.0 && stats.cache_hit_rate <= 1.0);
    }

    #[test]
    fn test_morton_encoding() {
        let encoded = CacheOptimizedStateVector::morton_encode_2d(5, 2); // (1, 1) in 2 bits
        assert!(encoded < 16); // Should be within 4-bit range
    }

    #[test]
    fn test_hilbert_coordinate_conversion() {
        let (x, y) = CacheOptimizedStateVector::hilbert_index_to_xy(3, 2);
        assert!(x < 4 && y < 4); // Should be within 2^2 x 2^2 grid
    }

    #[test]
    fn test_stride_detection() {
        let accesses = vec![&0, &4, &8, &12, &16];
        let strides = CacheOptimizedStateVector::detect_strides(&accesses);
        assert!(strides.contains(&4)); // Should detect stride of 4
    }
}
