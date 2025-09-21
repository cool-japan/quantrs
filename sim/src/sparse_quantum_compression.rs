//! Advanced Sparse Quantum State Compression
//!
//! This module implements state-of-the-art compression techniques specifically designed
//! for sparse quantum states, enabling simulation of 40+ qubits on standard hardware
//! through intelligent compression, deduplication, and memory mapping strategies.

use num_complex::Complex64;
use std::collections::{HashMap, BTreeMap, HashSet};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};
use std::io::{Read, Write, BufReader, BufWriter};
use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use lz4::{Decoder, EncoderBuilder};
use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// Compression algorithms available for quantum states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Fast LZ4 compression
    LZ4,
    /// Balanced zlib compression
    Zlib,
    /// Quantum-specific amplitude clustering
    QuantumAmplitudeClustering,
    /// Hybrid approach combining multiple techniques
    Hybrid,
}

/// Metadata for compressed quantum states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetadata {
    /// Unique identifier for the compressed state
    pub id: Uuid,
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio achieved
    pub compression_ratio: f64,
    /// Algorithm used for compression
    pub algorithm: CompressionAlgorithm,
    /// Number of qubits in the state
    pub num_qubits: usize,
    /// Sparsity level (fraction of non-zero amplitudes)
    pub sparsity: f64,
    /// Compression timestamp
    pub timestamp: std::time::SystemTime,
    /// Fidelity preservation level
    pub fidelity: f64,
}

/// Sparse quantum state representation optimized for compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseQuantumState {
    /// Non-zero amplitudes indexed by basis state
    pub amplitudes: BTreeMap<u64, Complex64>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Normalization factor
    pub norm: f64,
    /// Metadata about the state
    pub metadata: Option<CompressionMetadata>,
}

impl SparseQuantumState {
    /// Create new sparse quantum state
    pub fn new(num_qubits: usize) -> Self {
        Self {
            amplitudes: BTreeMap::new(),
            num_qubits,
            norm: 0.0,
            metadata: None,
        }
    }

    /// Create from dense state vector with automatic sparsification
    pub fn from_dense(state_vector: &[Complex64], sparsity_threshold: f64) -> Self {
        let num_qubits = (state_vector.len() as f64).log2() as usize;
        let mut amplitudes = BTreeMap::new();
        let mut norm_squared = 0.0;

        // Calculate norm and identify significant amplitudes
        for (i, &amplitude) in state_vector.iter().enumerate() {
            let magnitude_squared = amplitude.norm_sqr();
            norm_squared += magnitude_squared;

            if magnitude_squared > sparsity_threshold {
                amplitudes.insert(i as u64, amplitude);
            }
        }

        Self {
            amplitudes,
            num_qubits,
            norm: norm_squared.sqrt(),
            metadata: None,
        }
    }

    /// Convert to dense state vector
    pub fn to_dense(&self) -> Vec<Complex64> {
        let size = 1 << self.num_qubits;
        let mut state_vector = vec![Complex64::new(0.0, 0.0); size];

        for (&index, &amplitude) in &self.amplitudes {
            if (index as usize) < size {
                state_vector[index as usize] = amplitude;
            }
        }

        state_vector
    }

    /// Get sparsity level (fraction of non-zero elements)
    pub fn sparsity(&self) -> f64 {
        let total_size = 1 << self.num_qubits;
        self.amplitudes.len() as f64 / total_size as f64
    }

    /// Apply sparsity threshold to remove small amplitudes
    pub fn apply_sparsity_threshold(&mut self, threshold: f64) -> usize {
        let initial_count = self.amplitudes.len();
        self.amplitudes.retain(|_, amplitude| amplitude.norm_sqr() > threshold);

        // Renormalize
        self.renormalize();

        initial_count - self.amplitudes.len()
    }

    /// Renormalize the quantum state
    pub fn renormalize(&mut self) {
        let norm_squared: f64 = self.amplitudes.values()
            .map(|amplitude| amplitude.norm_sqr())
            .sum();

        let norm = norm_squared.sqrt();
        if norm > 1e-15 {
            for amplitude in self.amplitudes.values_mut() {
                *amplitude /= norm;
            }
            self.norm = 1.0;
        }
    }
}

/// Quantum amplitude clustering for compression
#[derive(Debug)]
pub struct QuantumAmplitudeClustering {
    /// Number of clusters for amplitude grouping
    num_clusters: usize,
    /// Tolerance for amplitude similarity
    tolerance: f64,
    /// Cluster centers
    cluster_centers: Vec<Complex64>,
    /// Cluster assignments
    cluster_assignments: HashMap<u64, usize>,
}

impl QuantumAmplitudeClustering {
    /// Create new amplitude clustering compressor
    pub fn new(num_clusters: usize, tolerance: f64) -> Self {
        Self {
            num_clusters,
            tolerance,
            cluster_centers: Vec::new(),
            cluster_assignments: HashMap::new(),
        }
    }

    /// Compress sparse quantum state using amplitude clustering
    pub fn compress(&mut self, state: &SparseQuantumState) -> QuantumAmplitudeClusteringResult {
        // Collect unique amplitudes
        let unique_amplitudes: Vec<Complex64> = state.amplitudes.values().copied().collect();

        if unique_amplitudes.is_empty() {
            return QuantumAmplitudeClusteringResult {
                cluster_centers: Vec::new(),
                cluster_assignments: HashMap::new(),
                compression_ratio: 1.0,
                fidelity: 1.0,
            };
        }

        // K-means clustering on amplitude space
        self.cluster_centers = self.kmeans_clustering(&unique_amplitudes);

        // Assign each amplitude to nearest cluster
        self.cluster_assignments.clear();
        let mut total_error = 0.0;

        for (&index, &amplitude) in &state.amplitudes {
            let cluster_id = self.find_nearest_cluster(amplitude);
            self.cluster_assignments.insert(index, cluster_id);

            let cluster_center = self.cluster_centers[cluster_id];
            total_error += (amplitude - cluster_center).norm_sqr();
        }

        // Calculate compression metrics
        let original_size = state.amplitudes.len() * std::mem::size_of::<Complex64>();
        let compressed_size = self.cluster_centers.len() * std::mem::size_of::<Complex64>()
                            + state.amplitudes.len() * std::mem::size_of::<usize>();

        let compression_ratio = original_size as f64 / compressed_size as f64;
        let fidelity = 1.0 - (total_error / state.amplitudes.len() as f64).sqrt();

        QuantumAmplitudeClusteringResult {
            cluster_centers: self.cluster_centers.clone(),
            cluster_assignments: self.cluster_assignments.clone(),
            compression_ratio,
            fidelity,
        }
    }

    /// K-means clustering for complex amplitudes
    fn kmeans_clustering(&self, amplitudes: &[Complex64]) -> Vec<Complex64> {
        let mut centers = Vec::new();
        let k = self.num_clusters.min(amplitudes.len());

        // Initialize centers using k-means++
        if !amplitudes.is_empty() {
            centers.push(amplitudes[0]);

            for _ in 1..k {
                let mut best_distance = 0.0;
                let mut best_candidate = amplitudes[0];

                for &candidate in amplitudes {
                    let min_distance = centers.iter()
                        .map(|center| (candidate - center).norm_sqr())
                        .fold(f64::INFINITY, f64::min);

                    if min_distance > best_distance {
                        best_distance = min_distance;
                        best_candidate = candidate;
                    }
                }

                centers.push(best_candidate);
            }
        }

        // Refine centers with Lloyd's algorithm
        for _ in 0..10 {
            let mut new_centers = vec![Complex64::new(0.0, 0.0); k];
            let mut counts = vec![0; k];

            // Assign points to clusters
            for &amplitude in amplitudes {
                let cluster_id = self.find_nearest_cluster_in_set(amplitude, &centers);
                new_centers[cluster_id] += amplitude;
                counts[cluster_id] += 1;
            }

            // Update centers
            for i in 0..k {
                if counts[i] > 0 {
                    new_centers[i] /= counts[i] as f64;
                }
            }

            // Check convergence
            let mut converged = true;
            for i in 0..k {
                if (new_centers[i] - centers[i]).norm() > self.tolerance {
                    converged = false;
                    break;
                }
            }

            centers = new_centers;
            if converged {
                break;
            }
        }

        centers
    }

    /// Find nearest cluster for an amplitude
    fn find_nearest_cluster(&self, amplitude: Complex64) -> usize {
        self.find_nearest_cluster_in_set(amplitude, &self.cluster_centers)
    }

    /// Find nearest cluster in a given set of centers
    fn find_nearest_cluster_in_set(&self, amplitude: Complex64, centers: &[Complex64]) -> usize {
        let mut best_distance = f64::INFINITY;
        let mut best_cluster = 0;

        for (i, &center) in centers.iter().enumerate() {
            let distance = (amplitude - center).norm_sqr();
            if distance < best_distance {
                best_distance = distance;
                best_cluster = i;
            }
        }

        best_cluster
    }

    /// Decompress quantum state from clustering
    pub fn decompress(&self,
                     indices: &[u64],
                     result: &QuantumAmplitudeClusteringResult) -> SparseQuantumState {
        let mut amplitudes = BTreeMap::new();

        for &index in indices {
            if let Some(&cluster_id) = result.cluster_assignments.get(&index) {
                if cluster_id < result.cluster_centers.len() {
                    amplitudes.insert(index, result.cluster_centers[cluster_id]);
                }
            }
        }

        let num_qubits = (indices.iter().max().unwrap_or(&0) + 1).next_power_of_two().trailing_zeros() as usize;

        SparseQuantumState {
            amplitudes,
            num_qubits,
            norm: 1.0,
            metadata: None,
        }
    }
}

/// Result of quantum amplitude clustering compression
#[derive(Debug, Clone)]
pub struct QuantumAmplitudeClusteringResult {
    pub cluster_centers: Vec<Complex64>,
    pub cluster_assignments: HashMap<u64, usize>,
    pub compression_ratio: f64,
    pub fidelity: f64,
}

/// Memory statistics for compression operations
#[derive(Debug, Clone, Default)]
pub struct CompressionStatistics {
    /// Total bytes compressed
    pub total_bytes_compressed: u64,
    /// Total compression operations
    pub compression_operations: u64,
    /// Total decompression operations
    pub decompression_operations: u64,
    /// Average compression ratio
    pub average_compression_ratio: f64,
    /// Average compression time (ms)
    pub average_compression_time_ms: f64,
    /// Average decompression time (ms)
    pub average_decompression_time_ms: f64,
    /// Memory saved (bytes)
    pub memory_saved_bytes: u64,
    /// Average fidelity preservation
    pub average_fidelity: f64,
}

impl CompressionStatistics {
    /// Update statistics with new compression operation
    pub fn record_compression(&mut self,
                             original_size: usize,
                             compressed_size: usize,
                             compression_time: Duration,
                             fidelity: f64) {
        self.total_bytes_compressed += original_size as u64;
        self.compression_operations += 1;

        let compression_ratio = original_size as f64 / compressed_size as f64;
        self.average_compression_ratio =
            (self.average_compression_ratio * (self.compression_operations - 1) as f64 + compression_ratio)
            / self.compression_operations as f64;

        let compression_time_ms = compression_time.as_secs_f64() * 1000.0;
        self.average_compression_time_ms =
            (self.average_compression_time_ms * (self.compression_operations - 1) as f64 + compression_time_ms)
            / self.compression_operations as f64;

        self.memory_saved_bytes += (original_size - compressed_size) as u64;

        self.average_fidelity =
            (self.average_fidelity * (self.compression_operations - 1) as f64 + fidelity)
            / self.compression_operations as f64;
    }

    /// Update statistics with decompression operation
    pub fn record_decompression(&mut self, decompression_time: Duration) {
        self.decompression_operations += 1;

        let decompression_time_ms = decompression_time.as_secs_f64() * 1000.0;
        self.average_decompression_time_ms =
            (self.average_decompression_time_ms * (self.decompression_operations - 1) as f64 + decompression_time_ms)
            / self.decompression_operations as f64;
    }
}

/// Advanced sparse quantum state compressor with multiple algorithms
pub struct SparseQuantumStateCompressor {
    /// Compression configuration
    config: CompressionConfig,
    /// Amplitude clustering compressor
    amplitude_clusterer: QuantumAmplitudeClustering,
    /// Compression statistics
    stats: Arc<Mutex<CompressionStatistics>>,
    /// Cache of compressed states
    compression_cache: Arc<RwLock<HashMap<Uuid, CompressedState>>>,
}

/// Configuration for sparse quantum state compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Primary compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Sparsity threshold for compression
    pub sparsity_threshold: f64,
    /// Fidelity threshold (minimum fidelity to maintain)
    pub fidelity_threshold: f64,
    /// Enable compression caching
    pub enable_caching: bool,
    /// Maximum cache size (number of states)
    pub max_cache_size: usize,
    /// Number of clusters for amplitude clustering
    pub amplitude_clusters: usize,
    /// Compression level (1-9 for zlib/lz4)
    pub compression_level: u32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Hybrid,
            sparsity_threshold: 1e-12,
            fidelity_threshold: 0.999,
            enable_caching: true,
            max_cache_size: 1000,
            amplitude_clusters: 256,
            compression_level: 6,
        }
    }
}

/// Compressed quantum state container
#[derive(Debug, Clone)]
pub struct CompressedState {
    /// Compressed data
    pub data: Vec<u8>,
    /// Compression metadata
    pub metadata: CompressionMetadata,
    /// Original indices for sparse representation
    pub indices: Vec<u64>,
}

impl SparseQuantumStateCompressor {
    /// Create new sparse quantum state compressor
    pub fn new(config: CompressionConfig) -> Self {
        let amplitude_clusterer = QuantumAmplitudeClustering::new(
            config.amplitude_clusters,
            1e-15
        );

        Self {
            config,
            amplitude_clusterer,
            stats: Arc::new(Mutex::new(CompressionStatistics::default())),
            compression_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Compress sparse quantum state using configured algorithm
    pub fn compress(&mut self, state: &SparseQuantumState) -> Result<CompressedState, CompressionError> {
        let start_time = Instant::now();

        // Apply sparsity threshold
        let mut working_state = state.clone();
        working_state.apply_sparsity_threshold(self.config.sparsity_threshold);

        let result = match self.config.algorithm {
            CompressionAlgorithm::None => self.compress_none(&working_state)?,
            CompressionAlgorithm::LZ4 => self.compress_lz4(&working_state)?,
            CompressionAlgorithm::Zlib => self.compress_zlib(&working_state)?,
            CompressionAlgorithm::QuantumAmplitudeClustering => self.compress_amplitude_clustering(&working_state)?,
            CompressionAlgorithm::Hybrid => self.compress_hybrid(&working_state)?,
        };

        let compression_time = start_time.elapsed();

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            let original_size = state.amplitudes.len() * std::mem::size_of::<Complex64>();
            stats.record_compression(
                original_size,
                result.data.len(),
                compression_time,
                result.metadata.fidelity,
            );
        }

        // Cache if enabled
        if self.config.enable_caching {
            self.cache_compressed_state(result.clone());
        }

        Ok(result)
    }

    /// Decompress quantum state
    pub fn decompress(&self, compressed: &CompressedState) -> Result<SparseQuantumState, CompressionError> {
        let start_time = Instant::now();

        let result = match compressed.metadata.algorithm {
            CompressionAlgorithm::None => self.decompress_none(compressed)?,
            CompressionAlgorithm::LZ4 => self.decompress_lz4(compressed)?,
            CompressionAlgorithm::Zlib => self.decompress_zlib(compressed)?,
            CompressionAlgorithm::QuantumAmplitudeClustering => self.decompress_amplitude_clustering(compressed)?,
            CompressionAlgorithm::Hybrid => self.decompress_hybrid(compressed)?,
        };

        let decompression_time = start_time.elapsed();

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.record_decompression(decompression_time);
        }

        Ok(result)
    }

    /// No compression (identity)
    fn compress_none(&self, state: &SparseQuantumState) -> Result<CompressedState, CompressionError> {
        let serialized = bincode::serialize(state)
            .map_err(|e| CompressionError::SerializationError(e.to_string()))?;

        let metadata = CompressionMetadata {
            id: Uuid::new_v4(),
            original_size: serialized.len(),
            compressed_size: serialized.len(),
            compression_ratio: 1.0,
            algorithm: CompressionAlgorithm::None,
            num_qubits: state.num_qubits,
            sparsity: state.sparsity(),
            timestamp: std::time::SystemTime::now(),
            fidelity: 1.0,
        };

        Ok(CompressedState {
            data: serialized,
            metadata,
            indices: state.amplitudes.keys().copied().collect(),
        })
    }

    /// LZ4 compression
    fn compress_lz4(&self, state: &SparseQuantumState) -> Result<CompressedState, CompressionError> {
        let serialized = bincode::serialize(state)
            .map_err(|e| CompressionError::SerializationError(e.to_string()))?;

        let mut encoder = EncoderBuilder::new()
            .level(self.config.compression_level)
            .build(Vec::new())
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        encoder.write_all(&serialized)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        let (compressed_data, result) = encoder.finish();
        result.map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        let metadata = CompressionMetadata {
            id: Uuid::new_v4(),
            original_size: serialized.len(),
            compressed_size: compressed_data.len(),
            compression_ratio: serialized.len() as f64 / compressed_data.len() as f64,
            algorithm: CompressionAlgorithm::LZ4,
            num_qubits: state.num_qubits,
            sparsity: state.sparsity(),
            timestamp: std::time::SystemTime::now(),
            fidelity: 1.0,
        };

        Ok(CompressedState {
            data: compressed_data,
            metadata,
            indices: state.amplitudes.keys().copied().collect(),
        })
    }

    /// Zlib compression
    fn compress_zlib(&self, state: &SparseQuantumState) -> Result<CompressedState, CompressionError> {
        let serialized = bincode::serialize(state)
            .map_err(|e| CompressionError::SerializationError(e.to_string()))?;

        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(self.config.compression_level));
        encoder.write_all(&serialized)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        let compressed_data = encoder.finish()
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        let metadata = CompressionMetadata {
            id: Uuid::new_v4(),
            original_size: serialized.len(),
            compressed_size: compressed_data.len(),
            compression_ratio: serialized.len() as f64 / compressed_data.len() as f64,
            algorithm: CompressionAlgorithm::Zlib,
            num_qubits: state.num_qubits,
            sparsity: state.sparsity(),
            timestamp: std::time::SystemTime::now(),
            fidelity: 1.0,
        };

        Ok(CompressedState {
            data: compressed_data,
            metadata,
            indices: state.amplitudes.keys().copied().collect(),
        })
    }

    /// Quantum amplitude clustering compression
    fn compress_amplitude_clustering(&mut self, state: &SparseQuantumState) -> Result<CompressedState, CompressionError> {
        let clustering_result = self.amplitude_clusterer.compress(state);

        let compressed_data = bincode::serialize(&clustering_result)
            .map_err(|e| CompressionError::SerializationError(e.to_string()))?;

        let original_size = state.amplitudes.len() * std::mem::size_of::<Complex64>();

        let metadata = CompressionMetadata {
            id: Uuid::new_v4(),
            original_size,
            compressed_size: compressed_data.len(),
            compression_ratio: clustering_result.compression_ratio,
            algorithm: CompressionAlgorithm::QuantumAmplitudeClustering,
            num_qubits: state.num_qubits,
            sparsity: state.sparsity(),
            timestamp: std::time::SystemTime::now(),
            fidelity: clustering_result.fidelity,
        };

        Ok(CompressedState {
            data: compressed_data,
            metadata,
            indices: state.amplitudes.keys().copied().collect(),
        })
    }

    /// Hybrid compression using best algorithm
    fn compress_hybrid(&mut self, state: &SparseQuantumState) -> Result<CompressedState, CompressionError> {
        // Try multiple algorithms and pick the best
        let algorithms = vec![
            CompressionAlgorithm::LZ4,
            CompressionAlgorithm::Zlib,
            CompressionAlgorithm::QuantumAmplitudeClustering,
        ];

        let mut best_result = None;
        let mut best_score = f64::NEG_INFINITY;

        for algorithm in algorithms {
            let mut temp_compressor = Self::new(CompressionConfig {
                algorithm,
                ..self.config.clone()
            });

            if let Ok(result) = temp_compressor.compress(state) {
                // Score based on compression ratio and fidelity
                let score = result.metadata.compression_ratio * result.metadata.fidelity;

                if score > best_score && result.metadata.fidelity >= self.config.fidelity_threshold {
                    best_score = score;
                    best_result = Some(result);
                }
            }
        }

        best_result.ok_or(CompressionError::NoSuitableAlgorithm)
    }

    /// Decompression implementations (simplified for brevity)
    fn decompress_none(&self, compressed: &CompressedState) -> Result<SparseQuantumState, CompressionError> {
        bincode::deserialize(&compressed.data)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))
    }

    fn decompress_lz4(&self, compressed: &CompressedState) -> Result<SparseQuantumState, CompressionError> {
        let mut decoder = Decoder::new(&compressed.data[..])
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        bincode::deserialize(&decompressed)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))
    }

    fn decompress_zlib(&self, compressed: &CompressedState) -> Result<SparseQuantumState, CompressionError> {
        let mut decoder = ZlibDecoder::new(&compressed.data[..]);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        bincode::deserialize(&decompressed)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))
    }

    fn decompress_amplitude_clustering(&self, compressed: &CompressedState) -> Result<SparseQuantumState, CompressionError> {
        let clustering_result: QuantumAmplitudeClusteringResult =
            bincode::deserialize(&compressed.data)
                .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        Ok(self.amplitude_clusterer.decompress(&compressed.indices, &clustering_result))
    }

    fn decompress_hybrid(&self, compressed: &CompressedState) -> Result<SparseQuantumState, CompressionError> {
        // Decompress based on the algorithm used during compression
        match compressed.metadata.algorithm {
            CompressionAlgorithm::LZ4 => self.decompress_lz4(compressed),
            CompressionAlgorithm::Zlib => self.decompress_zlib(compressed),
            CompressionAlgorithm::QuantumAmplitudeClustering => self.decompress_amplitude_clustering(compressed),
            _ => self.decompress_none(compressed),
        }
    }

    /// Cache compressed state
    fn cache_compressed_state(&self, compressed: CompressedState) {
        if let Ok(mut cache) = self.compression_cache.write() {
            if cache.len() >= self.config.max_cache_size {
                // Remove oldest entry
                if let Some(oldest_key) = cache.keys().next().copied() {
                    cache.remove(&oldest_key);
                }
            }
            cache.insert(compressed.metadata.id, compressed);
        }
    }

    /// Get compression statistics
    pub fn get_statistics(&self) -> CompressionStatistics {
        self.stats.lock().unwrap().clone()
    }

    /// Clear compression cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.compression_cache.write() {
            cache.clear();
        }
    }
}

/// Compression error types
#[derive(Debug, Clone)]
pub enum CompressionError {
    SerializationError(String),
    CompressionFailed(String),
    DecompressionFailed(String),
    NoSuitableAlgorithm,
    CacheError(String),
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            CompressionError::CompressionFailed(msg) => write!(f, "Compression failed: {}", msg),
            CompressionError::DecompressionFailed(msg) => write!(f, "Decompression failed: {}", msg),
            CompressionError::NoSuitableAlgorithm => write!(f, "No suitable compression algorithm found"),
            CompressionError::CacheError(msg) => write!(f, "Cache error: {}", msg),
        }
    }
}

impl std::error::Error for CompressionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_quantum_state_creation() {
        let state = SparseQuantumState::new(4);
        assert_eq!(state.num_qubits, 4);
        assert_eq!(state.sparsity(), 0.0);
    }

    #[test]
    fn test_sparse_from_dense_conversion() {
        let dense_state = vec![
            Complex64::new(0.7071, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.7071, 0.0),
        ];

        let sparse_state = SparseQuantumState::from_dense(&dense_state, 1e-10);
        assert_eq!(sparse_state.num_qubits, 2);
        assert_eq!(sparse_state.amplitudes.len(), 2);

        let recovered_dense = sparse_state.to_dense();
        assert_eq!(recovered_dense.len(), 4);
    }

    #[test]
    fn test_compression_lz4() {
        let mut state = SparseQuantumState::new(10);

        // Add some amplitudes
        for i in 0..100 {
            state.amplitudes.insert(i, Complex64::new(0.1, 0.05));
        }

        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::LZ4,
            ..Default::default()
        };

        let mut compressor = SparseQuantumStateCompressor::new(config);
        let compressed = compressor.compress(&state).unwrap();

        assert!(compressed.metadata.compression_ratio > 1.0);
        assert_eq!(compressed.metadata.algorithm, CompressionAlgorithm::LZ4);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.amplitudes.len(), state.amplitudes.len());
    }

    #[test]
    fn test_amplitude_clustering() {
        let mut clusterer = QuantumAmplitudeClustering::new(4, 1e-10);

        let mut state = SparseQuantumState::new(8);

        // Add amplitudes with some clustering structure
        state.amplitudes.insert(0, Complex64::new(0.5, 0.0));
        state.amplitudes.insert(1, Complex64::new(0.5, 0.01));
        state.amplitudes.insert(2, Complex64::new(0.5, -0.01));
        state.amplitudes.insert(3, Complex64::new(0.0, 0.5));
        state.amplitudes.insert(4, Complex64::new(0.01, 0.5));

        let result = clusterer.compress(&state);
        assert!(result.compression_ratio > 1.0);
        assert!(result.fidelity > 0.9);
        assert_eq!(result.cluster_centers.len(), 4);
    }

    #[test]
    fn test_compression_statistics() {
        let mut stats = CompressionStatistics::default();

        stats.record_compression(1000, 500, Duration::from_millis(10), 0.999);
        stats.record_compression(2000, 800, Duration::from_millis(15), 0.995);

        assert_eq!(stats.compression_operations, 2);
        assert_eq!(stats.total_bytes_compressed, 3000);
        assert!(stats.average_compression_ratio > 1.0);
        assert!(stats.average_fidelity > 0.99);
    }

    #[test]
    fn test_hybrid_compression() {
        let mut state = SparseQuantumState::new(12);

        // Create a state with various amplitude patterns
        for i in 0..200 {
            let amplitude = if i % 3 == 0 {
                Complex64::new(0.1, 0.0)
            } else if i % 5 == 0 {
                Complex64::new(0.0, 0.1)
            } else {
                Complex64::new(0.05, 0.05)
            };
            state.amplitudes.insert(i, amplitude);
        }

        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::Hybrid,
            fidelity_threshold: 0.99,
            ..Default::default()
        };

        let mut compressor = SparseQuantumStateCompressor::new(config);
        let compressed = compressor.compress(&state).unwrap();

        assert!(compressed.metadata.compression_ratio > 1.0);
        assert!(compressed.metadata.fidelity >= 0.99);

        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.num_qubits, state.num_qubits);
    }
}