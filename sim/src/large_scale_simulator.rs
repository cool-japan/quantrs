//! Large-Scale Quantum Simulator with Advanced Memory Optimization
//!
//! This module provides state-of-the-art memory optimization techniques to enable
//! simulation of 40+ qubit quantum circuits on standard hardware through sparse
//! representations, compression, memory mapping, and `SciRS2` high-performance computing.

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    buffer_pool::BufferPool,
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::parallel_ops::{IndexedParallelIterator, ParallelIterator}; // SciRS2 POLICY compliant
                                                                            // use quantrs2_core::platform::PlatformCapabilities;
                                                                            // use scirs2_core::memory::BufferPool as SciRS2BufferPool;
                                                                            // use scirs2_optimize::compression::{CompressionEngine, HuffmanEncoder, LZ4Encoder};
                                                                            // use scirs2_linalg::{Matrix, Vector, SVD, sparse::CSRMatrix};
                                                                            // flate2 replaced by oxiarc-deflate (COOLJAPAN Pure Rust Policy)
use memmap2::{MmapMut, MmapOptions};
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use uuid::Uuid;

/// Large-scale simulator configuration for 40+ qubit systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeScaleSimulatorConfig {
    /// Maximum number of qubits to simulate
    pub max_qubits: usize,

    /// Enable sparse state vector representation
    pub enable_sparse_representation: bool,

    /// Enable state compression
    pub enable_compression: bool,

    /// Enable memory mapping for very large states
    pub enable_memory_mapping: bool,

    /// Enable chunked processing
    pub enable_chunked_processing: bool,

    /// Chunk size for processing (in complex numbers)
    pub chunk_size: usize,

    /// Sparsity threshold (fraction of non-zero elements)
    pub sparsity_threshold: f64,

    /// Compression threshold (minimum size to compress)
    pub compression_threshold: usize,

    /// Memory mapping threshold (minimum size to use mmap)
    pub memory_mapping_threshold: usize,

    /// Working directory for temporary files
    pub working_directory: PathBuf,

    /// Enable `SciRS2` optimizations
    pub enable_scirs2_optimizations: bool,

    /// Memory budget in bytes
    pub memory_budget: usize,

    /// Enable adaptive precision
    pub enable_adaptive_precision: bool,

    /// Precision tolerance for adaptive precision
    pub precision_tolerance: f64,
}

impl Default for LargeScaleSimulatorConfig {
    fn default() -> Self {
        Self {
            max_qubits: 50,
            enable_sparse_representation: true,
            enable_compression: true,
            enable_memory_mapping: true,
            enable_chunked_processing: true,
            chunk_size: 1024 * 1024,                    // 1M complex numbers
            sparsity_threshold: 0.1,                    // Sparse if < 10% non-zero
            compression_threshold: 1024 * 1024 * 8,     // 8MB
            memory_mapping_threshold: 1024 * 1024 * 64, // 64MB
            working_directory: std::env::temp_dir().join("quantrs_large_scale"),
            enable_scirs2_optimizations: true,
            memory_budget: 8 * 1024 * 1024 * 1024, // 8GB
            enable_adaptive_precision: true,
            precision_tolerance: 1e-12,
        }
    }
}

/// Simple sparse matrix representation for quantum states
#[derive(Debug, Clone)]
pub struct SimpleSparseMatrix {
    /// Non-zero values
    values: Vec<Complex64>,
    /// Column indices for non-zero values
    col_indices: Vec<usize>,
    /// Row pointers (start index for each row)
    row_ptr: Vec<usize>,
    /// Matrix dimensions
    rows: usize,
    cols: usize,
}

impl SimpleSparseMatrix {
    #[must_use]
    pub fn from_dense(dense: &[Complex64], threshold: f64) -> Self {
        let rows = dense.len();
        let cols = 1; // For state vectors
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0];

        for (i, &val) in dense.iter().enumerate() {
            if val.norm() > threshold {
                values.push(val);
                col_indices.push(0); // State vector is column vector
            }
            row_ptr.push(values.len());
        }

        Self {
            values,
            col_indices,
            row_ptr,
            rows,
            cols,
        }
    }

    #[must_use]
    pub fn to_dense(&self) -> Vec<Complex64> {
        let mut dense = vec![Complex64::new(0.0, 0.0); self.rows];

        for (row, &val) in self.values.iter().enumerate() {
            if row < self.row_ptr.len() - 1 {
                let start = self.row_ptr[row];
                let end = self.row_ptr[row + 1];
                if start < end && start < self.col_indices.len() {
                    let col = self.col_indices[start];
                    if col < dense.len() {
                        dense[row] = val;
                    }
                }
            }
        }

        dense
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get amplitude at a specific row (basis state index)
    #[must_use]
    pub fn get_amplitude(&self, row: usize) -> Complex64 {
        if row >= self.rows || row + 1 >= self.row_ptr.len() {
            return Complex64::new(0.0, 0.0);
        }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        if start < end && start < self.values.len() {
            self.values[start]
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Build from a HashMap of non-zero amplitudes (sparse direct construction)
    #[must_use]
    pub fn from_sparse_map(
        amplitudes: &HashMap<usize, Complex64>,
        dimension: usize,
        threshold: f64,
    ) -> Self {
        let rows = dimension;
        let cols = 1;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0usize; rows + 1];

        // First pass: count non-zero entries per row
        for (&idx, &val) in amplitudes {
            if idx < rows && val.norm() > threshold {
                row_ptr[idx + 1] = 1;
            }
        }

        // Prefix sum to get actual row pointers
        for i in 1..=rows {
            row_ptr[i] += row_ptr[i - 1];
        }

        let nnz = row_ptr[rows];
        values.resize(nnz, Complex64::new(0.0, 0.0));
        col_indices.resize(nnz, 0usize);

        // Second pass: fill values
        let mut fill_pos = vec![0usize; rows];
        fill_pos[..rows].copy_from_slice(&row_ptr[..rows]);

        for (&idx, &val) in amplitudes {
            if idx < rows && val.norm() > threshold {
                let pos = fill_pos[idx];
                values[pos] = val;
                col_indices[pos] = 0;
                fill_pos[idx] += 1;
            }
        }

        Self {
            values,
            col_indices,
            row_ptr,
            rows,
            cols,
        }
    }

    /// Convert to a HashMap of non-zero amplitudes
    #[must_use]
    pub fn to_sparse_map(&self) -> HashMap<usize, Complex64> {
        let mut map = HashMap::new();
        for row in 0..self.rows {
            if row + 1 < self.row_ptr.len() {
                let start = self.row_ptr[row];
                let end = self.row_ptr[row + 1];
                if start < end && start < self.values.len() {
                    let val = self.values[start];
                    if val.norm() > 0.0 {
                        map.insert(row, val);
                    }
                }
            }
        }
        map
    }
}

/// Sparse quantum state representation using simple sparse matrices
#[derive(Debug)]
pub struct SparseQuantumState {
    /// Sparse representation of non-zero amplitudes
    sparse_amplitudes: SimpleSparseMatrix,

    /// Number of qubits
    num_qubits: usize,

    /// Total dimension (`2^num_qubits`)
    dimension: usize,

    /// Non-zero indices and their positions
    nonzero_indices: HashMap<usize, usize>,

    /// Sparsity ratio
    sparsity_ratio: f64,
}

impl SparseQuantumState {
    /// Create new sparse quantum state
    pub fn new(num_qubits: usize) -> QuantRS2Result<Self> {
        let dimension = 1usize << num_qubits;

        // Initialize in |0...0⟩ state (single non-zero element)
        let mut dense = vec![Complex64::new(0.0, 0.0); dimension];
        dense[0] = Complex64::new(1.0, 0.0);

        let sparse_amplitudes = SimpleSparseMatrix::from_dense(&dense, 1e-15);

        let mut nonzero_indices = HashMap::new();
        nonzero_indices.insert(0, 0);

        Ok(Self {
            sparse_amplitudes,
            num_qubits,
            dimension,
            nonzero_indices,
            sparsity_ratio: 1.0 / dimension as f64,
        })
    }

    /// Convert from dense state vector
    pub fn from_dense(amplitudes: &[Complex64], threshold: f64) -> QuantRS2Result<Self> {
        let num_qubits = (amplitudes.len() as f64).log2() as usize;
        let dimension = amplitudes.len();

        // Find non-zero elements
        let mut nonzero_indices = HashMap::new();
        let mut nonzero_count = 0;

        for (i, &amplitude) in amplitudes.iter().enumerate() {
            if amplitude.norm() > threshold {
                nonzero_indices.insert(i, nonzero_count);
                nonzero_count += 1;
            }
        }

        let sparse_amplitudes = SimpleSparseMatrix::from_dense(amplitudes, threshold);
        let sparsity_ratio = nonzero_count as f64 / dimension as f64;

        Ok(Self {
            sparse_amplitudes,
            num_qubits,
            dimension,
            nonzero_indices,
            sparsity_ratio,
        })
    }

    /// Convert to dense state vector
    pub fn to_dense(&self) -> QuantRS2Result<Vec<Complex64>> {
        Ok(self.sparse_amplitudes.to_dense())
    }

    /// Apply sparse gate operation using true sparse arithmetic.
    ///
    /// For single-qubit gates we iterate only over non-zero amplitude pairs
    /// (basis states that differ only in the target qubit bit), computing the
    /// two output amplitudes from the 2×2 gate matrix without ever constructing
    /// the full dense state vector.
    ///
    /// For two-qubit gates we enumerate all four combinations of the two qubit
    /// bits across non-zero amplitudes and compute the four output amplitudes.
    ///
    /// Amplitudes whose norm falls below 1e-12 are pruned to maintain sparsity.
    pub fn apply_sparse_gate(&mut self, gate: &dyn GateOp) -> QuantRS2Result<()> {
        let qubits = gate.qubits();
        match qubits.len() {
            1 => {
                let target = qubits[0].id() as usize;
                let matrix = gate.matrix()?;
                self.apply_single_qubit_sparse(&matrix, target)?;
            }
            2 => {
                let q0 = qubits[0].id() as usize;
                let q1 = qubits[1].id() as usize;
                let matrix = gate.matrix()?;
                self.apply_two_qubit_sparse(&matrix, q0, q1)?;
            }
            _ => {
                // Fallback to dense operation for 3+ qubit gates
                let matrix = gate.matrix()?;
                self.apply_dense_gate(&matrix, &qubits)?;
            }
        }

        Ok(())
    }

    /// Apply a single-qubit gate in true sparse representation.
    ///
    /// For each pair of basis states (i0, i1) that differ only in `target` bit:
    ///   new[i0] = m[0]*amp[i0] + m[1]*amp[i1]
    ///   new[i1] = m[2]*amp[i0] + m[3]*amp[i1]
    ///
    /// We only process pairs where at least one amplitude is non-zero.
    fn apply_single_qubit_sparse(
        &mut self,
        matrix: &[Complex64],
        target: usize,
    ) -> QuantRS2Result<()> {
        if matrix.len() < 4 {
            return Err(QuantRS2Error::InvalidInput(
                "Single-qubit gate matrix must have 4 elements".to_string(),
            ));
        }
        if target >= self.num_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Target qubit {} out of range (num_qubits={})",
                target, self.num_qubits
            )));
        }

        const THRESHOLD: f64 = 1e-12;

        // Collect current non-zero amplitudes into a HashMap for O(1) lookup
        let current: HashMap<usize, Complex64> = self.sparse_amplitudes.to_sparse_map();

        let target_mask = 1usize << target;
        // We need to process each (i0, i1) pair exactly once.
        // i0 has target bit = 0, i1 = i0 | target_mask
        let mut visited: HashSet<usize> = HashSet::new();
        let mut new_amplitudes: HashMap<usize, Complex64> = HashMap::new();

        for &idx in current.keys() {
            // Normalise to the i0 version (target bit = 0)
            let i0 = idx & !target_mask;
            if visited.contains(&i0) {
                continue;
            }
            visited.insert(i0);
            let i1 = i0 | target_mask;

            let a0 = current
                .get(&i0)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0));
            let a1 = current
                .get(&i1)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0));

            let new0 = matrix[0] * a0 + matrix[1] * a1;
            let new1 = matrix[2] * a0 + matrix[3] * a1;

            if new0.norm() > THRESHOLD {
                new_amplitudes.insert(i0, new0);
            }
            if new1.norm() > THRESHOLD {
                new_amplitudes.insert(i1, new1);
            }
        }

        // Rebuild sparse representation from new amplitudes
        self.nonzero_indices.clear();
        for (pos, (&idx, _)) in new_amplitudes.iter().enumerate() {
            self.nonzero_indices.insert(idx, pos);
        }
        self.sparse_amplitudes =
            SimpleSparseMatrix::from_sparse_map(&new_amplitudes, self.dimension, THRESHOLD);
        self.sparsity_ratio = new_amplitudes.len() as f64 / self.dimension as f64;

        Ok(())
    }

    /// Apply a two-qubit gate in true sparse representation.
    ///
    /// For each group of four basis states sharing all bits except q0/q1, we
    /// read the four amplitudes (|00⟩, |01⟩, |10⟩, |11⟩), apply the 4×4
    /// matrix, and write back only outputs above threshold.
    fn apply_two_qubit_sparse(
        &mut self,
        matrix: &[Complex64],
        q0: usize,
        q1: usize,
    ) -> QuantRS2Result<()> {
        if matrix.len() < 16 {
            return Err(QuantRS2Error::InvalidInput(
                "Two-qubit gate matrix must have 16 elements".to_string(),
            ));
        }
        if q0 >= self.num_qubits || q1 >= self.num_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Qubit indices {},{} out of range (num_qubits={})",
                q0, q1, self.num_qubits
            )));
        }

        const THRESHOLD: f64 = 1e-12;

        let current: HashMap<usize, Complex64> = self.sparse_amplitudes.to_sparse_map();
        let mask0 = 1usize << q0;
        let mask1 = 1usize << q1;
        let both_mask = mask0 | mask1;

        // Base index has both q0 and q1 bits = 0
        let mut visited: HashSet<usize> = HashSet::new();
        let mut new_amplitudes: HashMap<usize, Complex64> = HashMap::new();

        for &idx in current.keys() {
            let base = idx & !both_mask;
            if visited.contains(&base) {
                continue;
            }
            visited.insert(base);

            // Four basis states: |b_q0=0, b_q1=0⟩, |01⟩, |10⟩, |11⟩
            let i00 = base;
            let i01 = base | mask1;
            let i10 = base | mask0;
            let i11 = base | both_mask;

            let a00 = current
                .get(&i00)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0));
            let a01 = current
                .get(&i01)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0));
            let a10 = current
                .get(&i10)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0));
            let a11 = current
                .get(&i11)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0));

            let inputs = [a00, a01, a10, a11];
            let indices = [i00, i01, i10, i11];

            for (row, &out_idx) in indices.iter().enumerate() {
                let mut new_val = Complex64::new(0.0, 0.0);
                for (col, &inp) in inputs.iter().enumerate() {
                    new_val += matrix[row * 4 + col] * inp;
                }
                if new_val.norm() > THRESHOLD {
                    new_amplitudes.insert(out_idx, new_val);
                }
            }
        }

        self.nonzero_indices.clear();
        for (pos, (&idx, _)) in new_amplitudes.iter().enumerate() {
            self.nonzero_indices.insert(idx, pos);
        }
        self.sparse_amplitudes =
            SimpleSparseMatrix::from_sparse_map(&new_amplitudes, self.dimension, THRESHOLD);
        self.sparsity_ratio = new_amplitudes.len() as f64 / self.dimension as f64;

        Ok(())
    }

    /// Apply Pauli-X gate efficiently in sparse representation
    fn apply_pauli_x_sparse(&mut self, target: usize) -> QuantRS2Result<()> {
        // Pauli-X just flips the target bit in the indices
        let mut new_nonzero_indices = HashMap::new();
        let target_mask = 1usize << target;

        for (&old_idx, &pos) in &self.nonzero_indices {
            let new_idx = old_idx ^ target_mask;
            new_nonzero_indices.insert(new_idx, pos);
        }

        self.nonzero_indices = new_nonzero_indices;

        // Update sparse matrix indices
        self.update_sparse_matrix()?;

        Ok(())
    }

    /// Apply Hadamard gate in sparse representation
    fn apply_hadamard_sparse(&mut self, target: usize) -> QuantRS2Result<()> {
        // Hadamard creates superposition, potentially doubling non-zero elements
        // For true sparsity preservation, we'd need more sophisticated techniques
        // For now, use dense conversion
        let dense = self.to_dense()?;
        let mut new_dense = vec![Complex64::new(0.0, 0.0); self.dimension];

        let h_00 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let h_01 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let h_10 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let h_11 = Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0);

        let target_mask = 1usize << target;

        for i in 0..self.dimension {
            let paired_idx = i ^ target_mask;
            let bit_val = (i >> target) & 1;

            if bit_val == 0 {
                new_dense[i] = h_00 * dense[i] + h_01 * dense[paired_idx];
                new_dense[paired_idx] = h_10 * dense[i] + h_11 * dense[paired_idx];
            }
        }

        *self = Self::from_dense(&new_dense, 1e-15)?;

        Ok(())
    }

    /// Apply dense gate operation
    fn apply_dense_gate(&mut self, matrix: &[Complex64], qubits: &[QubitId]) -> QuantRS2Result<()> {
        // Convert to dense, apply gate, convert back if still sparse enough
        let mut dense = self.to_dense()?;

        // Apply gate operation (simplified single-qubit case)
        if qubits.len() == 1 {
            let target = qubits[0].id() as usize;
            let target_mask = 1usize << target;

            for i in 0..self.dimension {
                if (i & target_mask) == 0 {
                    let paired_idx = i | target_mask;
                    let old_0 = dense[i];
                    let old_1 = dense[paired_idx];

                    dense[i] = matrix[0] * old_0 + matrix[1] * old_1;
                    dense[paired_idx] = matrix[2] * old_0 + matrix[3] * old_1;
                }
            }
        }

        // Check if result is still sparse enough
        let nonzero_count = dense.iter().filter(|&&x| x.norm() > 1e-15).count();
        let new_sparsity = nonzero_count as f64 / self.dimension as f64;

        if new_sparsity < 0.5 {
            // If still reasonably sparse
            *self = Self::from_dense(&dense, 1e-15)?;
        } else {
            // Convert to dense representation if no longer sparse
            return Err(QuantRS2Error::ComputationError(
                "State no longer sparse".to_string(),
            ));
        }

        Ok(())
    }

    /// Update sparse matrix after index changes
    fn update_sparse_matrix(&mut self) -> QuantRS2Result<()> {
        // Rebuild sparse matrix from current nonzero_indices
        let mut dense = vec![Complex64::new(0.0, 0.0); self.dimension];

        for &idx in self.nonzero_indices.keys() {
            if idx < dense.len() {
                // Set normalized amplitude (simplified)
                dense[idx] = Complex64::new(1.0 / (self.nonzero_indices.len() as f64).sqrt(), 0.0);
            }
        }

        self.sparse_amplitudes = SimpleSparseMatrix::from_dense(&dense, 1e-15);
        self.sparsity_ratio = self.nonzero_indices.len() as f64 / self.dimension as f64;

        Ok(())
    }

    /// Get current sparsity ratio
    #[must_use]
    pub const fn sparsity_ratio(&self) -> f64 {
        self.sparsity_ratio
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.nonzero_indices.len()
            * (std::mem::size_of::<usize>() + std::mem::size_of::<Complex64>())
    }
}

/// Simple compression engine for quantum states
#[derive(Debug)]
pub struct SimpleCompressionEngine {
    /// Internal buffer for compression operations
    buffer: Vec<u8>,
}

impl Default for SimpleCompressionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleCompressionEngine {
    #[must_use]
    pub const fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Simple LZ-style compression (using oxiarc-deflate zlib)
    pub fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        oxiarc_deflate::zlib::zlib_compress(data, 6).map_err(|e| format!("Compression failed: {e}"))
    }

    /// Simple decompression (using oxiarc-deflate zlib)
    pub fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        oxiarc_deflate::zlib::zlib_decompress(data)
            .map_err(|e| format!("Decompression failed: {e}"))
    }

    /// Huffman compression placeholder
    pub fn compress_huffman(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // For now, just use the LZ4 method
        self.compress_lz4(data)
    }

    /// Huffman decompression placeholder
    pub fn decompress_huffman(&self, data: &[u8]) -> Result<Vec<u8>, String> {
        // For now, just use the LZ4 method
        self.decompress_lz4(data)
    }
}

/// Compressed quantum state using simple compression
#[derive(Debug)]
pub struct CompressedQuantumState {
    /// Compressed amplitude data
    compressed_data: Vec<u8>,

    /// Compression metadata
    compression_metadata: CompressionMetadata,

    /// Compression engine
    compression_engine: SimpleCompressionEngine,

    /// Number of qubits
    num_qubits: usize,

    /// Original size in bytes
    original_size: usize,
}

/// Compression metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetadata {
    /// Compression algorithm used
    pub algorithm: CompressionAlgorithm,

    /// Compression ratio achieved
    pub compression_ratio: f64,

    /// Original data size
    pub original_size: usize,

    /// Compressed data size
    pub compressed_size: usize,

    /// Checksum for integrity verification
    pub checksum: u64,
}

/// Supported compression algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// Huffman encoding for sparse data
    Huffman,
    /// LZ4 for general compression
    LZ4,
    /// Quantum-specific amplitude compression
    QuantumAmplitude,
    /// No compression
    None,
}

impl CompressedQuantumState {
    /// Create new compressed state from dense amplitudes
    pub fn from_dense(
        amplitudes: &[Complex64],
        algorithm: CompressionAlgorithm,
    ) -> QuantRS2Result<Self> {
        let num_qubits = (amplitudes.len() as f64).log2() as usize;
        let original_size = std::mem::size_of_val(amplitudes);

        // Convert to bytes
        let amplitude_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(amplitudes.as_ptr().cast::<u8>(), original_size) };

        // Initialize compression engine
        let compression_engine = SimpleCompressionEngine::new();

        let (compressed_data, metadata) = match algorithm {
            CompressionAlgorithm::Huffman => {
                let compressed = compression_engine
                    .compress_huffman(amplitude_bytes)
                    .map_err(|e| {
                        QuantRS2Error::ComputationError(format!("Huffman compression failed: {e}"))
                    })?;

                let metadata = CompressionMetadata {
                    algorithm: CompressionAlgorithm::Huffman,
                    compression_ratio: original_size as f64 / compressed.len() as f64,
                    original_size,
                    compressed_size: compressed.len(),
                    checksum: Self::calculate_checksum(amplitude_bytes),
                };

                (compressed, metadata)
            }
            CompressionAlgorithm::LZ4 => {
                let compressed = compression_engine
                    .compress_lz4(amplitude_bytes)
                    .map_err(|e| {
                        QuantRS2Error::ComputationError(format!("LZ4 compression failed: {e}"))
                    })?;

                let metadata = CompressionMetadata {
                    algorithm: CompressionAlgorithm::LZ4,
                    compression_ratio: original_size as f64 / compressed.len() as f64,
                    original_size,
                    compressed_size: compressed.len(),
                    checksum: Self::calculate_checksum(amplitude_bytes),
                };

                (compressed, metadata)
            }
            CompressionAlgorithm::QuantumAmplitude => {
                // Custom quantum amplitude compression
                let compressed = Self::compress_quantum_amplitudes(amplitudes)?;

                let metadata = CompressionMetadata {
                    algorithm: CompressionAlgorithm::QuantumAmplitude,
                    compression_ratio: original_size as f64 / compressed.len() as f64,
                    original_size,
                    compressed_size: compressed.len(),
                    checksum: Self::calculate_checksum(amplitude_bytes),
                };

                (compressed, metadata)
            }
            CompressionAlgorithm::None => {
                let metadata = CompressionMetadata {
                    algorithm: CompressionAlgorithm::None,
                    compression_ratio: 1.0,
                    original_size,
                    compressed_size: original_size,
                    checksum: Self::calculate_checksum(amplitude_bytes),
                };

                (amplitude_bytes.to_vec(), metadata)
            }
        };

        Ok(Self {
            compressed_data,
            compression_metadata: metadata,
            compression_engine,
            num_qubits,
            original_size,
        })
    }

    /// Decompress to dense amplitudes
    pub fn to_dense(&self) -> QuantRS2Result<Vec<Complex64>> {
        let decompressed_bytes = match self.compression_metadata.algorithm {
            CompressionAlgorithm::Huffman => self
                .compression_engine
                .decompress_huffman(&self.compressed_data)
                .map_err(|e| {
                    QuantRS2Error::ComputationError(format!("Huffman decompression failed: {e}"))
                })?,
            CompressionAlgorithm::LZ4 => self
                .compression_engine
                .decompress_lz4(&self.compressed_data)
                .map_err(|e| {
                    QuantRS2Error::ComputationError(format!("LZ4 decompression failed: {e}"))
                })?,
            CompressionAlgorithm::QuantumAmplitude => {
                Self::decompress_quantum_amplitudes(&self.compressed_data, self.num_qubits)?
            }
            CompressionAlgorithm::None => self.compressed_data.clone(),
        };

        // Verify checksum
        let checksum = Self::calculate_checksum(&decompressed_bytes);
        if checksum != self.compression_metadata.checksum {
            return Err(QuantRS2Error::ComputationError(
                "Checksum verification failed".to_string(),
            ));
        }

        // Convert bytes back to complex numbers
        let amplitudes = unsafe {
            std::slice::from_raw_parts(
                decompressed_bytes.as_ptr().cast::<Complex64>(),
                decompressed_bytes.len() / std::mem::size_of::<Complex64>(),
            )
        }
        .to_vec();

        Ok(amplitudes)
    }

    /// Custom quantum amplitude compression
    fn compress_quantum_amplitudes(amplitudes: &[Complex64]) -> QuantRS2Result<Vec<u8>> {
        // Quantum-specific compression using magnitude-phase representation
        let mut compressed = Vec::new();

        for &amplitude in amplitudes {
            let magnitude = amplitude.norm();
            let phase = amplitude.arg();

            // Quantize magnitude and phase for better compression
            let quantized_magnitude = (magnitude * 65_535.0) as u16;
            let quantized_phase =
                ((phase + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * 65_535.0) as u16;

            compressed.extend_from_slice(&quantized_magnitude.to_le_bytes());
            compressed.extend_from_slice(&quantized_phase.to_le_bytes());
        }

        Ok(compressed)
    }

    /// Custom quantum amplitude decompression
    fn decompress_quantum_amplitudes(data: &[u8], num_qubits: usize) -> QuantRS2Result<Vec<u8>> {
        let dimension = 1usize << num_qubits;
        let mut amplitudes = Vec::with_capacity(dimension);

        for i in 0..dimension {
            let offset = i * 4; // 2 bytes magnitude + 2 bytes phase
            if offset + 4 <= data.len() {
                let magnitude_bytes = [data[offset], data[offset + 1]];
                let phase_bytes = [data[offset + 2], data[offset + 3]];

                let quantized_magnitude = u16::from_le_bytes(magnitude_bytes);
                let quantized_phase = u16::from_le_bytes(phase_bytes);

                let magnitude = f64::from(quantized_magnitude) / 65_535.0;
                let phase = ((f64::from(quantized_phase) / 65_535.0) * 2.0)
                    .mul_add(std::f64::consts::PI, -std::f64::consts::PI);

                let amplitude = Complex64::new(magnitude * phase.cos(), magnitude * phase.sin());
                amplitudes.push(amplitude);
            }
        }

        // Convert back to bytes
        let amplitude_bytes = unsafe {
            std::slice::from_raw_parts(
                amplitudes.as_ptr().cast::<u8>(),
                amplitudes.len() * std::mem::size_of::<Complex64>(),
            )
        };

        Ok(amplitude_bytes.to_vec())
    }

    /// Calculate checksum for integrity verification
    fn calculate_checksum(data: &[u8]) -> u64 {
        // Simple checksum implementation
        data.iter()
            .enumerate()
            .map(|(i, &b)| (i as u64).wrapping_mul(u64::from(b)))
            .sum()
    }

    /// Get compression ratio
    #[must_use]
    pub const fn compression_ratio(&self) -> f64 {
        self.compression_metadata.compression_ratio
    }

    /// Get memory usage in bytes
    #[must_use]
    pub fn memory_usage(&self) -> usize {
        self.compressed_data.len()
    }
}

/// Memory-mapped quantum state for very large simulations
#[derive(Debug)]
pub struct MemoryMappedQuantumState {
    /// Memory-mapped file
    mmap: MmapMut,

    /// File path
    file_path: PathBuf,

    /// Number of qubits
    num_qubits: usize,

    /// State dimension
    dimension: usize,

    /// Chunk size for processing
    chunk_size: usize,
}

impl MemoryMappedQuantumState {
    /// Create new memory-mapped state
    pub fn new(num_qubits: usize, chunk_size: usize, working_dir: &Path) -> QuantRS2Result<Self> {
        let dimension = 1usize << num_qubits;
        let file_size = dimension * std::mem::size_of::<Complex64>();

        // Create temporary file
        std::fs::create_dir_all(working_dir).map_err(|e| {
            QuantRS2Error::InvalidInput(format!("Failed to create working directory: {e}"))
        })?;

        let file_path = working_dir.join(format!("quantum_state_{}.tmp", Uuid::new_v4()));

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&file_path)
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Failed to create temp file: {e}")))?;

        file.set_len(file_size as u64)
            .map_err(|e| QuantRS2Error::InvalidInput(format!("Failed to set file size: {e}")))?;

        let mmap = unsafe {
            MmapOptions::new().map_mut(&file).map_err(|e| {
                QuantRS2Error::InvalidInput(format!("Failed to create memory map: {e}"))
            })?
        };

        let mut state = Self {
            mmap,
            file_path,
            num_qubits,
            dimension,
            chunk_size,
        };

        // Initialize to |0...0⟩ state
        state.initialize_zero_state()?;

        Ok(state)
    }

    /// Initialize state to |0...0⟩
    fn initialize_zero_state(&mut self) -> QuantRS2Result<()> {
        let amplitudes = self.get_amplitudes_mut();

        // Clear all amplitudes
        for amplitude in amplitudes.iter_mut() {
            *amplitude = Complex64::new(0.0, 0.0);
        }

        // Set first amplitude to 1.0 (|0...0⟩ state)
        if !amplitudes.is_empty() {
            amplitudes[0] = Complex64::new(1.0, 0.0);
        }

        Ok(())
    }

    /// Get amplitudes as mutable slice
    fn get_amplitudes_mut(&mut self) -> &mut [Complex64] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.mmap.as_mut_ptr().cast::<Complex64>(),
                self.dimension,
            )
        }
    }

    /// Get amplitudes as slice
    fn get_amplitudes(&self) -> &[Complex64] {
        unsafe {
            std::slice::from_raw_parts(self.mmap.as_ptr().cast::<Complex64>(), self.dimension)
        }
    }

    /// Apply gate operation using chunked processing
    pub fn apply_gate_chunked(&mut self, gate: &dyn GateOp) -> QuantRS2Result<()> {
        let num_chunks = self.dimension.div_ceil(self.chunk_size);

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * self.chunk_size;
            let end = (start + self.chunk_size).min(self.dimension);

            self.apply_gate_to_chunk(gate, start, end)?;
        }

        Ok(())
    }

    /// Apply gate to specific chunk
    fn apply_gate_to_chunk(
        &mut self,
        gate: &dyn GateOp,
        start: usize,
        end: usize,
    ) -> QuantRS2Result<()> {
        // Cache dimension to avoid borrowing issues
        let dimension = self.dimension;
        let amplitudes = self.get_amplitudes_mut();

        match gate.name() {
            "X" => {
                if let Some(target) = gate.qubits().first() {
                    let target_idx = target.id() as usize;
                    let target_mask = 1usize << target_idx;

                    for i in start..end {
                        if (i & target_mask) == 0 {
                            let paired_idx = i | target_mask;
                            if paired_idx < dimension {
                                amplitudes.swap(i, paired_idx);
                            }
                        }
                    }
                }
            }
            "H" => {
                if let Some(target) = gate.qubits().first() {
                    let target_idx = target.id() as usize;
                    let target_mask = 1usize << target_idx;
                    let inv_sqrt2 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

                    // Create temporary buffer to avoid borrowing issues
                    let mut temp_buffer = vec![Complex64::new(0.0, 0.0); end - start];
                    for (i, &val) in amplitudes[start..end].iter().enumerate() {
                        temp_buffer[i] = val;
                    }

                    for i in start..end {
                        if (i & target_mask) == 0 {
                            let paired_idx = i | target_mask;
                            if paired_idx < dimension && paired_idx >= start && paired_idx < end {
                                let old_0 = temp_buffer[i - start];
                                let old_1 = temp_buffer[paired_idx - start];

                                amplitudes[i] = inv_sqrt2 * (old_0 + old_1);
                                amplitudes[paired_idx] = inv_sqrt2 * (old_0 - old_1);
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(QuantRS2Error::UnsupportedOperation(format!(
                    "Chunked operation not implemented for gate {}",
                    gate.name()
                )));
            }
        }

        Ok(())
    }

    /// Get memory usage (just metadata, actual state is in file)
    #[must_use]
    pub const fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    /// Get file size
    #[must_use]
    pub const fn file_size(&self) -> usize {
        self.dimension * std::mem::size_of::<Complex64>()
    }
}

impl Drop for MemoryMappedQuantumState {
    fn drop(&mut self) {
        // Clean up temporary file
        let _ = std::fs::remove_file(&self.file_path);
    }
}

/// Large-scale quantum simulator supporting 40+ qubits
#[derive(Debug)]
pub struct LargeScaleQuantumSimulator {
    /// Configuration
    config: LargeScaleSimulatorConfig,

    /// Current quantum state representation
    state: QuantumStateRepresentation,

    /// Buffer pool for optimizations
    buffer_pool: Arc<Mutex<Vec<Vec<Complex64>>>>,

    /// Memory usage statistics
    memory_stats: Arc<Mutex<MemoryStatistics>>,
}

/// Different quantum state representations
#[derive(Debug)]
pub enum QuantumStateRepresentation {
    /// Dense state vector (traditional)
    Dense(Vec<Complex64>),

    /// Sparse state representation
    Sparse(SparseQuantumState),

    /// Compressed state representation
    Compressed(CompressedQuantumState),

    /// Memory-mapped state for very large simulations
    MemoryMapped(MemoryMappedQuantumState),
}

/// Memory usage statistics
#[derive(Debug, Default, Clone)]
pub struct MemoryStatistics {
    /// Current memory usage in bytes
    pub current_usage: usize,

    /// Peak memory usage in bytes
    pub peak_usage: usize,

    /// Number of allocations
    pub allocations: u64,

    /// Number of deallocations
    pub deallocations: u64,

    /// Compression ratio achieved
    pub compression_ratio: f64,

    /// Sparsity ratio
    pub sparsity_ratio: f64,

    /// Time spent in memory operations (microseconds)
    pub memory_operation_time_us: u64,
}

impl LargeScaleQuantumSimulator {
    /// Create new large-scale simulator
    pub fn new(config: LargeScaleSimulatorConfig) -> QuantRS2Result<Self> {
        let buffer_pool = Arc::new(Mutex::new(Vec::new()));
        let memory_stats = Arc::new(Mutex::new(MemoryStatistics::default()));

        // Initialize with appropriate state representation
        let state = QuantumStateRepresentation::Dense(vec![Complex64::new(1.0, 0.0)]);

        Ok(Self {
            config,
            state,
            buffer_pool,
            memory_stats,
        })
    }

    /// Initialize quantum state for given number of qubits
    pub fn initialize_state(&mut self, num_qubits: usize) -> QuantRS2Result<()> {
        if num_qubits > self.config.max_qubits {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Number of qubits {} exceeds maximum {}",
                num_qubits, self.config.max_qubits
            )));
        }

        let dimension = 1usize << num_qubits;
        let memory_required = dimension * std::mem::size_of::<Complex64>();

        // Choose appropriate representation based on size and configuration
        self.state = if memory_required > self.config.memory_mapping_threshold {
            // Use memory mapping for very large states
            QuantumStateRepresentation::MemoryMapped(MemoryMappedQuantumState::new(
                num_qubits,
                self.config.chunk_size,
                &self.config.working_directory,
            )?)
        } else if memory_required > self.config.compression_threshold
            && self.config.enable_compression
        {
            // Use compression for medium-large states
            let amplitudes = vec![Complex64::new(0.0, 0.0); dimension];
            let mut amplitudes = amplitudes;
            amplitudes[0] = Complex64::new(1.0, 0.0); // |0...0⟩ state

            QuantumStateRepresentation::Compressed(CompressedQuantumState::from_dense(
                &amplitudes,
                CompressionAlgorithm::LZ4,
            )?)
        } else if self.config.enable_sparse_representation {
            // Use sparse representation
            QuantumStateRepresentation::Sparse(SparseQuantumState::new(num_qubits)?)
        } else {
            // Use traditional dense representation
            let mut amplitudes = vec![Complex64::new(0.0, 0.0); dimension];
            amplitudes[0] = Complex64::new(1.0, 0.0); // |0...0⟩ state
            QuantumStateRepresentation::Dense(amplitudes)
        };

        self.update_memory_stats()?;

        Ok(())
    }

    /// Apply quantum gate
    pub fn apply_gate(&mut self, gate: &dyn GateOp) -> QuantRS2Result<()> {
        let start_time = std::time::Instant::now();

        // Handle different state representations
        let mut needs_state_change = None;

        match &mut self.state {
            QuantumStateRepresentation::Dense(amplitudes) => {
                // Create a copy to avoid borrowing issues
                let mut amplitudes_copy = amplitudes.clone();
                Self::apply_gate_dense(&mut amplitudes_copy, gate, &self.config)?;
                *amplitudes = amplitudes_copy;
            }
            QuantumStateRepresentation::Sparse(sparse_state) => {
                sparse_state.apply_sparse_gate(gate)?;

                // Check if still sparse enough
                if sparse_state.sparsity_ratio() > self.config.sparsity_threshold {
                    // Convert to dense if no longer sparse
                    let dense = sparse_state.to_dense()?;
                    needs_state_change = Some(QuantumStateRepresentation::Dense(dense));
                }
            }
            QuantumStateRepresentation::Compressed(compressed_state) => {
                // Decompress, apply gate, recompress
                let mut dense = compressed_state.to_dense()?;
                Self::apply_gate_dense(&mut dense, gate, &self.config)?;

                // Recompress if beneficial
                let new_compressed =
                    CompressedQuantumState::from_dense(&dense, CompressionAlgorithm::LZ4)?;
                if new_compressed.compression_ratio() > 1.5 {
                    needs_state_change =
                        Some(QuantumStateRepresentation::Compressed(new_compressed));
                } else {
                    needs_state_change = Some(QuantumStateRepresentation::Dense(dense));
                }
            }
            QuantumStateRepresentation::MemoryMapped(mmap_state) => {
                mmap_state.apply_gate_chunked(gate)?;
            }
        }

        // Apply state change if needed
        if let Some(new_state) = needs_state_change {
            self.state = new_state;
        }

        let elapsed = start_time.elapsed();
        if let Ok(mut stats) = self.memory_stats.lock() {
            stats.memory_operation_time_us += elapsed.as_micros() as u64;
        }

        Ok(())
    }

    /// Apply gate to dense representation
    fn apply_gate_dense(
        amplitudes: &mut [Complex64],
        gate: &dyn GateOp,
        config: &LargeScaleSimulatorConfig,
    ) -> QuantRS2Result<()> {
        match gate.name() {
            "X" => {
                if let Some(target) = gate.qubits().first() {
                    let target_idx = target.id() as usize;
                    Self::apply_pauli_x_dense(amplitudes, target_idx)?;
                }
            }
            "H" => {
                if let Some(target) = gate.qubits().first() {
                    let target_idx = target.id() as usize;
                    Self::apply_hadamard_dense(amplitudes, target_idx, config)?;
                }
            }
            "CNOT" => {
                if gate.qubits().len() >= 2 {
                    let control_idx = gate.qubits()[0].id() as usize;
                    let target_idx = gate.qubits()[1].id() as usize;
                    Self::apply_cnot_dense(amplitudes, control_idx, target_idx)?;
                }
            }
            _ => {
                return Err(QuantRS2Error::UnsupportedOperation(format!(
                    "Gate {} not implemented in large-scale simulator",
                    gate.name()
                )));
            }
        }

        Ok(())
    }

    /// Apply Pauli-X gate to dense representation
    fn apply_pauli_x_dense(amplitudes: &mut [Complex64], target: usize) -> QuantRS2Result<()> {
        let target_mask = 1usize << target;

        // Sequential operation for now (parallel optimization can be added later)
        for i in 0..amplitudes.len() {
            if (i & target_mask) == 0 {
                let paired_idx = i | target_mask;
                if paired_idx < amplitudes.len() {
                    amplitudes.swap(i, paired_idx);
                }
            }
        }

        Ok(())
    }

    /// Apply Hadamard gate to dense representation
    fn apply_hadamard_dense(
        amplitudes: &mut [Complex64],
        target: usize,
        _config: &LargeScaleSimulatorConfig,
    ) -> QuantRS2Result<()> {
        let target_mask = 1usize << target;
        let inv_sqrt2 = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

        // Create temporary buffer
        let mut temp = vec![Complex64::new(0.0, 0.0); amplitudes.len()];
        temp.copy_from_slice(amplitudes);

        // Sequential operation for now (parallel optimization can be added later)
        for i in 0..amplitudes.len() {
            if (i & target_mask) == 0 {
                let paired_idx = i | target_mask;
                if paired_idx < amplitudes.len() {
                    let old_0 = temp[i];
                    let old_1 = temp[paired_idx];

                    amplitudes[i] = inv_sqrt2 * (old_0 + old_1);
                    amplitudes[paired_idx] = inv_sqrt2 * (old_0 - old_1);
                }
            }
        }

        Ok(())
    }

    /// Apply CNOT gate to dense representation
    fn apply_cnot_dense(
        amplitudes: &mut [Complex64],
        control: usize,
        target: usize,
    ) -> QuantRS2Result<()> {
        let control_mask = 1usize << control;
        let target_mask = 1usize << target;

        // Sequential operation for now (parallel optimization can be added later)
        for i in 0..amplitudes.len() {
            if (i & control_mask) != 0 && (i & target_mask) == 0 {
                let flipped_idx = i | target_mask;
                if flipped_idx < amplitudes.len() {
                    amplitudes.swap(i, flipped_idx);
                }
            }
        }

        Ok(())
    }

    /// Get current state as dense vector (for measurement)
    pub fn get_dense_state(&self) -> QuantRS2Result<Vec<Complex64>> {
        match &self.state {
            QuantumStateRepresentation::Dense(amplitudes) => Ok(amplitudes.clone()),
            QuantumStateRepresentation::Sparse(sparse_state) => sparse_state.to_dense(),
            QuantumStateRepresentation::Compressed(compressed_state) => compressed_state.to_dense(),
            QuantumStateRepresentation::MemoryMapped(mmap_state) => {
                Ok(mmap_state.get_amplitudes().to_vec())
            }
        }
    }

    /// Update memory usage statistics
    fn update_memory_stats(&self) -> QuantRS2Result<()> {
        if let Ok(mut stats) = self.memory_stats.lock() {
            let current_usage = match &self.state {
                QuantumStateRepresentation::Dense(amplitudes) => {
                    amplitudes.len() * std::mem::size_of::<Complex64>()
                }
                QuantumStateRepresentation::Sparse(sparse_state) => sparse_state.memory_usage(),
                QuantumStateRepresentation::Compressed(compressed_state) => {
                    compressed_state.memory_usage()
                }
                QuantumStateRepresentation::MemoryMapped(mmap_state) => mmap_state.memory_usage(),
            };

            stats.current_usage = current_usage;
            if current_usage > stats.peak_usage {
                stats.peak_usage = current_usage;
            }

            // Update compression and sparsity ratios
            match &self.state {
                QuantumStateRepresentation::Compressed(compressed_state) => {
                    stats.compression_ratio = compressed_state.compression_ratio();
                }
                QuantumStateRepresentation::Sparse(sparse_state) => {
                    stats.sparsity_ratio = sparse_state.sparsity_ratio();
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Get memory statistics
    #[must_use]
    pub fn get_memory_stats(&self) -> MemoryStatistics {
        self.memory_stats
            .lock()
            .map(|stats| stats.clone())
            .unwrap_or_default()
    }

    /// Get configuration
    #[must_use]
    pub const fn get_config(&self) -> &LargeScaleSimulatorConfig {
        &self.config
    }

    /// Check if current state can simulate given number of qubits
    #[must_use]
    pub const fn can_simulate(&self, num_qubits: usize) -> bool {
        if num_qubits > self.config.max_qubits {
            return false;
        }

        let dimension = 1usize << num_qubits;
        let memory_required = dimension * std::mem::size_of::<Complex64>();

        memory_required <= self.config.memory_budget
    }

    /// Estimate memory requirements for given number of qubits
    #[must_use]
    pub fn estimate_memory_requirements(&self, num_qubits: usize) -> usize {
        let dimension = 1usize << num_qubits;
        let base_memory = dimension * std::mem::size_of::<Complex64>();

        // Add overhead for operations and temporary buffers
        let overhead_factor = 1.5;
        (base_memory as f64 * overhead_factor) as usize
    }
}

impl<const N: usize> Simulator<N> for LargeScaleQuantumSimulator {
    fn run(&self, circuit: &Circuit<N>) -> QuantRS2Result<quantrs2_core::register::Register<N>> {
        let mut simulator = Self::new(self.config.clone())?;
        simulator.initialize_state(N)?;

        // Apply all gates in the circuit
        for gate in circuit.gates() {
            simulator.apply_gate(gate.as_ref())?;
        }

        // Get final state and create register
        let final_state = simulator.get_dense_state()?;
        quantrs2_core::register::Register::with_amplitudes(final_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX};
    use quantrs2_core::qubit::QubitId;

    #[test]
    fn test_sparse_quantum_state() {
        let mut sparse_state =
            SparseQuantumState::new(3).expect("Sparse state creation should succeed in test");
        assert_eq!(sparse_state.num_qubits, 3);
        assert_eq!(sparse_state.dimension, 8);
        assert!(sparse_state.sparsity_ratio() < 0.2);

        let dense = sparse_state
            .to_dense()
            .expect("Sparse to dense conversion should succeed in test");
        assert_eq!(dense.len(), 8);
        assert!((dense[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_compressed_quantum_state() {
        let amplitudes = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let compressed = CompressedQuantumState::from_dense(&amplitudes, CompressionAlgorithm::LZ4)
            .expect("Compression should succeed in test");
        let decompressed = compressed
            .to_dense()
            .expect("Decompression should succeed in test");

        assert_eq!(decompressed.len(), 4);
        assert!((decompressed[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_large_scale_simulator() {
        let config = LargeScaleSimulatorConfig::default();
        let mut simulator = LargeScaleQuantumSimulator::new(config)
            .expect("Simulator creation should succeed in test");

        // Test with 10 qubits (medium scale)
        simulator
            .initialize_state(10)
            .expect("State initialization should succeed in test");
        assert!(simulator.can_simulate(10));

        // Test basic gates
        let x_gate = PauliX { target: QubitId(0) };
        simulator
            .apply_gate(&x_gate)
            .expect("X gate application should succeed in test");

        let h_gate = Hadamard { target: QubitId(1) };
        simulator
            .apply_gate(&h_gate)
            .expect("H gate application should succeed in test");

        let final_state = simulator
            .get_dense_state()
            .expect("State retrieval should succeed in test");
        assert_eq!(final_state.len(), 1024); // 2^10
    }

    #[test]
    fn test_memory_stats() {
        let config = LargeScaleSimulatorConfig::default();
        let mut simulator = LargeScaleQuantumSimulator::new(config)
            .expect("Simulator creation should succeed in test");

        simulator
            .initialize_state(5)
            .expect("State initialization should succeed in test");
        let stats = simulator.get_memory_stats();

        assert!(stats.current_usage > 0);
        assert_eq!(stats.peak_usage, stats.current_usage);
    }
}
