//! Gate compilation caching with persistent storage
//!
//! This module provides a high-performance caching system for compiled quantum gates,
//! with support for persistent storage to disk, automatic cache management, and
//! concurrent access patterns.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    simd_ops,
};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    fs::{self, File},
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
// use sha2::{Sha256, Digest}; // Disabled for simplified implementation
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Compiled gate representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledGate {
    /// Unique identifier for the gate
    pub gate_id: String,
    /// Gate matrix elements (row-major order)
    pub matrix: Vec<Complex64>,
    /// Number of qubits the gate acts on
    pub num_qubits: usize,
    /// Optimized representations
    pub optimizations: GateOptimizations,
    /// Metadata about the compilation
    pub metadata: CompilationMetadata,
}

/// Optimized representations of a gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateOptimizations {
    /// Diagonal representation if applicable
    pub diagonal: Option<Vec<Complex64>>,
    /// Decomposition into simpler gates
    pub decomposition: Option<GateDecomposition>,
    /// SIMD-optimized matrix layout
    pub simd_layout: Option<SimdLayout>,
    /// GPU kernel identifier
    pub gpu_kernel_id: Option<String>,
    /// Tensor network representation
    pub tensor_network: Option<TensorNetworkRep>,
}

/// Gate decomposition into simpler gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecomposition {
    /// Sequence of gate identifiers
    pub gates: Vec<String>,
    /// Parameters for parametric gates
    pub parameters: Vec<Vec<f64>>,
    /// Target qubits for each gate
    pub targets: Vec<Vec<usize>>,
    /// Total gate count
    pub gate_count: usize,
    /// Decomposition error
    pub error: f64,
}

/// SIMD-optimized memory layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdLayout {
    /// Layout type (e.g., "avx2", "avx512", "neon")
    pub layout_type: String,
    /// Reordered matrix data for SIMD access
    pub data: Vec<Complex64>,
    /// Stride information
    pub stride: usize,
    /// Alignment requirement
    pub alignment: usize,
}

/// Tensor network representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorNetworkRep {
    /// Tensor indices
    pub tensors: Vec<TensorNode>,
    /// Contraction order
    pub contraction_order: Vec<(usize, usize)>,
    /// Bond dimensions
    pub bond_dims: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorNode {
    pub id: usize,
    pub shape: Vec<usize>,
    pub data: Vec<Complex64>,
}

/// Metadata about gate compilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationMetadata {
    /// Timestamp of compilation
    pub compiled_at: u64,
    /// Compilation time in microseconds
    pub compilation_time_us: u64,
    /// Compiler version
    pub compiler_version: String,
    /// Hardware target
    pub target_hardware: String,
    /// Optimization level
    pub optimization_level: u32,
    /// Cache hits for this gate
    pub cache_hits: u64,
    /// Last access time
    pub last_accessed: u64,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Total number of cache hits
    pub total_hits: u64,
    /// Total number of cache misses
    pub total_misses: u64,
    /// Total compilation time saved (microseconds)
    pub time_saved_us: u64,
    /// Number of entries in cache
    pub num_entries: usize,
    /// Total cache size in bytes
    pub total_size_bytes: usize,
    /// Cache creation time
    pub created_at: u64,
}

/// Configuration for the compilation cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries in memory
    pub max_memory_entries: usize,
    /// Maximum total size in bytes
    pub max_size_bytes: usize,
    /// Cache directory path
    pub cache_dir: PathBuf,
    /// Enable persistent storage
    pub enable_persistence: bool,
    /// Cache expiration time
    pub expiration_time: Duration,
    /// Compression level (0-9, 0 = no compression)
    pub compression_level: u32,
    /// Enable async writes
    pub async_writes: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_memory_entries: 10000,
            max_size_bytes: 1024 * 1024 * 1024, // 1GB
            cache_dir: PathBuf::from(".quantrs_cache"),
            enable_persistence: true,
            expiration_time: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            compression_level: 3,
            async_writes: true,
        }
    }
}

/// Gate compilation cache with persistent storage
pub struct CompilationCache {
    /// In-memory cache
    memory_cache: Arc<RwLock<MemoryCache>>,
    /// Configuration
    config: CacheConfig,
    /// Cache statistics
    statistics: Arc<RwLock<CacheStatistics>>,
    /// Background writer handle
    writer_handle: Option<std::thread::JoinHandle<()>>,
    /// Write queue for async persistence
    write_queue: Arc<RwLock<VecDeque<CompiledGate>>>,
}

/// In-memory cache structure
struct MemoryCache {
    /// Gate storage by ID
    gates: HashMap<String, CompiledGate>,
    /// LRU queue for eviction
    lru_queue: VecDeque<String>,
    /// Current total size
    current_size: usize,
}

impl CompilationCache {
    /// Create a new compilation cache
    pub fn new(config: CacheConfig) -> QuantRS2Result<Self> {
        // Create cache directory if needed
        if config.enable_persistence {
            fs::create_dir_all(&config.cache_dir)?;
        }

        let memory_cache = Arc::new(RwLock::new(MemoryCache {
            gates: HashMap::new(),
            lru_queue: VecDeque::new(),
            current_size: 0,
        }));

        let statistics = Arc::new(RwLock::new(CacheStatistics {
            total_hits: 0,
            total_misses: 0,
            time_saved_us: 0,
            num_entries: 0,
            total_size_bytes: 0,
            created_at: current_timestamp(),
        }));

        let write_queue = Arc::new(RwLock::new(VecDeque::new()));

        // Start background writer if async writes are enabled
        let writer_handle = if config.async_writes && config.enable_persistence {
            Some(Self::start_background_writer(
                config.cache_dir.clone(),
                Arc::clone(&write_queue),
            ))
        } else {
            None
        };

        Ok(Self {
            memory_cache,
            config,
            statistics,
            writer_handle,
            write_queue,
        })
    }

    /// Get or compile a gate
    pub fn get_or_compile<F>(
        &self,
        gate: &dyn GateOp,
        compile_fn: F,
    ) -> QuantRS2Result<CompiledGate>
    where
        F: FnOnce(&dyn GateOp) -> QuantRS2Result<CompiledGate>,
    {
        let gate_id = self.compute_gate_id(gate)?;

        // Try memory cache first
        if let Some(compiled) = self.get_from_memory(&gate_id)? {
            self.record_hit(&gate_id)?;
            return Ok(compiled);
        }

        // Try persistent cache
        if self.config.enable_persistence {
            if let Some(compiled) = self.get_from_disk(&gate_id)? {
                self.add_to_memory(compiled.clone())?;
                self.record_hit(&gate_id)?;
                return Ok(compiled);
            }
        }

        // Cache miss - compile the gate
        self.record_miss()?;
        let start_time = std::time::Instant::now();
        
        let mut compiled = compile_fn(gate)?;
        
        let compilation_time = start_time.elapsed().as_micros() as u64;
        compiled.metadata.compilation_time_us = compilation_time;
        compiled.gate_id = gate_id;

        // Add to cache
        self.add_to_memory(compiled.clone())?;
        
        if self.config.enable_persistence {
            if self.config.async_writes {
                self.queue_for_write(compiled.clone())?;
            } else {
                self.write_to_disk(&compiled)?;
            }
        }

        Ok(compiled)
    }

    /// Compute unique gate identifier
    fn compute_gate_id(&self, gate: &dyn GateOp) -> QuantRS2Result<String> {
        let mut hasher = DefaultHasher::new();
        
        // Hash gate name
        gate.name().hash(&mut hasher);
        
        // Hash gate matrix
        let matrix = gate.matrix()?;
        for elem in &matrix {
            elem.re.to_bits().hash(&mut hasher);
            elem.im.to_bits().hash(&mut hasher);
        }
        
        // Hash target qubits
        for qubit in gate.qubits() {
            qubit.0.hash(&mut hasher);
        }
        
        let result = hasher.finish();
        Ok(format!("{:x}", result))
    }

    /// Get from memory cache
    fn get_from_memory(&self, gate_id: &str) -> QuantRS2Result<Option<CompiledGate>> {
        let mut cache = self.memory_cache.write().unwrap();
        
        if let Some(compiled) = cache.gates.get(gate_id).cloned() {
            // Update LRU
            cache.lru_queue.retain(|id| id != gate_id);
            cache.lru_queue.push_front(gate_id.to_string());
            
            // Update last accessed time
            let mut updated_compiled = compiled.clone();
            updated_compiled.metadata.last_accessed = current_timestamp();
            cache.gates.insert(gate_id.to_string(), updated_compiled.clone());
            
            Ok(Some(updated_compiled))
        } else {
            Ok(None)
        }
    }

    /// Add to memory cache
    fn add_to_memory(&self, compiled: CompiledGate) -> QuantRS2Result<()> {
        let mut cache = self.memory_cache.write().unwrap();
        let gate_size = self.estimate_size(&compiled);
        
        // Evict entries if needed
        while cache.gates.len() >= self.config.max_memory_entries ||
              cache.current_size + gate_size > self.config.max_size_bytes {
            if let Some(evict_id) = cache.lru_queue.pop_back() {
                if let Some(evicted) = cache.gates.remove(&evict_id) {
                    cache.current_size -= self.estimate_size(&evicted);
                }
            } else {
                break;
            }
        }
        
        // Add new entry
        cache.gates.insert(compiled.gate_id.clone(), compiled.clone());
        cache.lru_queue.push_front(compiled.gate_id.clone());
        cache.current_size += gate_size;
        
        // Update statistics
        let mut stats = self.statistics.write().unwrap();
        stats.num_entries = cache.gates.len();
        stats.total_size_bytes = cache.current_size;
        
        Ok(())
    }

    /// Get from persistent storage
    fn get_from_disk(&self, gate_id: &str) -> QuantRS2Result<Option<CompiledGate>> {
        let file_path = self.cache_file_path(gate_id);
        
        if !file_path.exists() {
            return Ok(None);
        }
        
        // Check expiration
        let metadata = fs::metadata(&file_path)?;
        let modified = metadata.modified()?;
        let age = SystemTime::now().duration_since(modified).unwrap_or_default();
        
        if age > self.config.expiration_time {
            // Expired - remove file
            fs::remove_file(&file_path)?;
            return Ok(None);
        }
        
        // Read and deserialize
        let file = File::open(&file_path)?;
        let reader = BufReader::new(file);
        
        let compiled = bincode::deserialize_from(reader)?;
        
        Ok(Some(compiled))
    }

    /// Write to persistent storage
    fn write_to_disk(&self, compiled: &CompiledGate) -> QuantRS2Result<()> {
        let file_path = self.cache_file_path(&compiled.gate_id);
        
        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        let file = File::create(&file_path)?;
        let writer = BufWriter::new(file);
        
        bincode::serialize_into(writer, compiled)?;
        
        Ok(())
    }

    /// Queue gate for asynchronous write
    fn queue_for_write(&self, compiled: CompiledGate) -> QuantRS2Result<()> {
        let mut queue = self.write_queue.write().unwrap();
        queue.push_back(compiled);
        Ok(())
    }

    /// Start background writer thread
    fn start_background_writer(
        cache_dir: PathBuf,
        write_queue: Arc<RwLock<VecDeque<CompiledGate>>>,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_millis(100));
                
                let gates_to_write: Vec<CompiledGate> = {
                    let mut queue = write_queue.write().unwrap();
                    queue.drain(..).collect()
                };
                
                for compiled in gates_to_write {
                    let file_path = cache_dir.join(format!("{}.cache", &compiled.gate_id[..16]));
                    
                    if let Err(e) = Self::write_gate_to_file(&file_path, &compiled, 3) {
                        eprintln!("Failed to write gate to cache: {}", e);
                    }
                }
            }
        })
    }

    /// Write a single gate to file (static method for thread)
    fn write_gate_to_file(
        file_path: &Path,
        compiled: &CompiledGate,
        compression_level: i32,
    ) -> QuantRS2Result<()> {
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        let file = File::create(file_path)?;
        let writer = BufWriter::new(file);
        
        bincode::serialize_into(writer, compiled)?;
        
        Ok(())
    }

    /// Get cache file path for a gate
    fn cache_file_path(&self, gate_id: &str) -> PathBuf {
        // Use first 16 chars of hash for filename
        let filename = format!("{}.cache", &gate_id[..16.min(gate_id.len())]);
        self.config.cache_dir.join(filename)
    }

    /// Estimate size of compiled gate
    fn estimate_size(&self, compiled: &CompiledGate) -> usize {
        std::mem::size_of::<CompiledGate>() +
        compiled.matrix.len() * std::mem::size_of::<Complex64>() +
        compiled.gate_id.len() +
        // Rough estimate for nested structures
        1024
    }

    /// Record cache hit
    fn record_hit(&self, gate_id: &str) -> QuantRS2Result<()> {
        let mut stats = self.statistics.write().unwrap();
        stats.total_hits += 1;
        
        // Estimate time saved (average compilation time)
        if let Ok(cache) = self.memory_cache.read() {
            if let Some(compiled) = cache.gates.get(gate_id) {
                stats.time_saved_us += compiled.metadata.compilation_time_us;
            }
        }
        
        Ok(())
    }

    /// Record cache miss
    fn record_miss(&self) -> QuantRS2Result<()> {
        let mut stats = self.statistics.write().unwrap();
        stats.total_misses += 1;
        Ok(())
    }

    /// Clear the cache
    pub fn clear(&self) -> QuantRS2Result<()> {
        // Clear memory cache
        let mut cache = self.memory_cache.write().unwrap();
        cache.gates.clear();
        cache.lru_queue.clear();
        cache.current_size = 0;
        
        // Clear disk cache if enabled
        if self.config.enable_persistence {
            if self.config.cache_dir.exists() {
                for entry in fs::read_dir(&self.config.cache_dir)? {
                    let entry = entry?;
                    if entry.path().extension().and_then(|s| s.to_str()) == Some("cache") {
                        fs::remove_file(entry.path())?;
                    }
                }
            }
        }
        
        // Reset statistics
        let mut stats = self.statistics.write().unwrap();
        *stats = CacheStatistics {
            total_hits: 0,
            total_misses: 0,
            time_saved_us: 0,
            num_entries: 0,
            total_size_bytes: 0,
            created_at: current_timestamp(),
        };
        
        Ok(())
    }

    /// Get cache statistics
    pub fn statistics(&self) -> CacheStatistics {
        self.statistics.read().unwrap().clone()
    }

    /// Optimize cache by removing expired entries
    pub fn optimize(&self) -> QuantRS2Result<()> {
        if !self.config.enable_persistence {
            return Ok(());
        }
        
        let mut removed_count = 0;
        
        for entry in fs::read_dir(&self.config.cache_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("cache") {
                let metadata = fs::metadata(&path)?;
                let modified = metadata.modified()?;
                let age = SystemTime::now().duration_since(modified).unwrap_or_default();
                
                if age > self.config.expiration_time {
                    fs::remove_file(&path)?;
                    removed_count += 1;
                }
            }
        }
        
        println!("Cache optimization: removed {} expired entries", removed_count);
        Ok(())
    }

    /// Export cache statistics to file
    pub fn export_statistics(&self, path: &Path) -> QuantRS2Result<()> {
        let stats = self.statistics();
        let json = serde_json::to_string_pretty(&stats)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Precompile and cache common gates
    pub fn precompile_common_gates(&self) -> QuantRS2Result<()> {
        use crate::gate::{single::*, multi::*};
        
        // Single-qubit gates
        let single_qubit_gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard { target: crate::qubit::QubitId(0) }),
            Box::new(PauliX { target: crate::qubit::QubitId(0) }),
            Box::new(PauliY { target: crate::qubit::QubitId(0) }),
            Box::new(PauliZ { target: crate::qubit::QubitId(0) }),
            Box::new(Phase { target: crate::qubit::QubitId(0) }),
            Box::new(RotationZ { target: crate::qubit::QubitId(0), theta: std::f64::consts::PI / 4.0 }),
        ];
        
        for gate in single_qubit_gates {
            let _ = self.get_or_compile(gate.as_ref(), |g| {
                compile_single_qubit_gate(g)
            })?;
        }
        
        // Two-qubit gates
        let two_qubit_gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(CNOT { 
                control: crate::qubit::QubitId(0), 
                target: crate::qubit::QubitId(1) 
            }),
            Box::new(CZ { 
                control: crate::qubit::QubitId(0), 
                target: crate::qubit::QubitId(1) 
            }),
            Box::new(SWAP { 
                qubit1: crate::qubit::QubitId(0), 
                qubit2: crate::qubit::QubitId(1) 
            }),
        ];
        
        for gate in two_qubit_gates {
            let _ = self.get_or_compile(gate.as_ref(), |g| {
                compile_two_qubit_gate(g)
            })?;
        }
        
        Ok(())
    }
}

/// Get current timestamp in seconds since UNIX epoch
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Default gate compilation functions

fn compile_single_qubit_gate(gate: &dyn GateOp) -> QuantRS2Result<CompiledGate> {
    let matrix = gate.matrix()?;
    let gate_id = String::new(); // Will be set by cache
    
    // Check if gate is diagonal
    let is_diagonal = matrix[1].norm() < 1e-10 && matrix[2].norm() < 1e-10;
    let diagonal = if is_diagonal {
        Some(vec![matrix[0], matrix[3]])
    } else {
        None
    };
    
    // Create SIMD layout
    let simd_layout = if false { // Simplified - disable SIMD check
        Some(SimdLayout {
            layout_type: "avx2".to_string(),
            data: matrix.clone(),
            stride: 2,
            alignment: 32,
        })
    } else {
        None
    };
    
    Ok(CompiledGate {
        gate_id,
        matrix,
        num_qubits: 1,
        optimizations: GateOptimizations {
            diagonal,
            decomposition: None,
            simd_layout,
            gpu_kernel_id: None,
            tensor_network: None,
        },
        metadata: CompilationMetadata {
            compiled_at: current_timestamp(),
            compilation_time_us: 0, // Will be set by cache
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            target_hardware: "generic".to_string(),
            optimization_level: 2,
            cache_hits: 0,
            last_accessed: current_timestamp(),
        },
    })
}

fn compile_two_qubit_gate(gate: &dyn GateOp) -> QuantRS2Result<CompiledGate> {
    let matrix = gate.matrix()?;
    let gate_id = String::new(); // Will be set by cache
    
    // Check for decomposition opportunities
    let decomposition = if gate.name() == "CNOT" {
        Some(GateDecomposition {
            gates: vec!["H".to_string(), "CZ".to_string(), "H".to_string()],
            parameters: vec![vec![], vec![], vec![]],
            targets: vec![vec![1], vec![0, 1], vec![1]],
            gate_count: 3,
            error: 1e-15,
        })
    } else {
        None
    };
    
    Ok(CompiledGate {
        gate_id,
        matrix,
        num_qubits: 2,
        optimizations: GateOptimizations {
            diagonal: None,
            decomposition,
            simd_layout: None,
            gpu_kernel_id: Some(format!("{}_kernel", gate.name().to_lowercase())),
            tensor_network: None,
        },
        metadata: CompilationMetadata {
            compiled_at: current_timestamp(),
            compilation_time_us: 0,
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            target_hardware: "generic".to_string(),
            optimization_level: 2,
            cache_hits: 0,
            last_accessed: current_timestamp(),
        },
    })
}

/// Global compilation cache instance
static mut GLOBAL_CACHE: Option<Arc<CompilationCache>> = None;
static CACHE_INIT: std::sync::Once = std::sync::Once::new();

/// Initialize the global compilation cache
pub fn initialize_compilation_cache(config: CacheConfig) -> QuantRS2Result<()> {
    CACHE_INIT.call_once(|| {
        match CompilationCache::new(config) {
            Ok(cache) => unsafe {
                GLOBAL_CACHE = Some(Arc::new(cache));
            },
            Err(e) => {
                eprintln!("Failed to initialize compilation cache: {}", e);
            }
        }
    });
    
    Ok(())
}

/// Get the global compilation cache
pub fn get_compilation_cache() -> QuantRS2Result<Arc<CompilationCache>> {
    unsafe {
        GLOBAL_CACHE.as_ref()
            .map(Arc::clone)
            .ok_or_else(|| QuantRS2Error::RuntimeError(
                "Compilation cache not initialized".to_string()
            ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::single::{Hadamard, PauliX};
    use crate::qubit::QubitId;
    use std::fs;
    // use tempfile::TempDir;

    #[test]
    fn test_cache_creation() {
        let temp_dir = std::env::temp_dir().join("quantrs_test_cache");
        let config = CacheConfig {
            cache_dir: temp_dir,
            enable_persistence: false, // Disable persistence for tests
            ..Default::default()
        };
        
        let cache = CompilationCache::new(config).unwrap();
        let stats = cache.statistics();
        
        assert_eq!(stats.total_hits, 0);
        assert_eq!(stats.total_misses, 0);
        assert_eq!(stats.num_entries, 0);
    }

    #[test]
    fn test_gate_compilation_and_caching() {
        let temp_dir = std::env::temp_dir().join(format!("quantrs_test_{}", std::process::id()));
        let config = CacheConfig {
            cache_dir: temp_dir,
            enable_persistence: true,
            async_writes: false,
            ..Default::default()
        };
        
        let cache = CompilationCache::new(config).unwrap();
        let gate = Hadamard { target: QubitId(0) };
        
        // First access - should compile
        let compiled1 = cache.get_or_compile(&gate, compile_single_qubit_gate).unwrap();
        let stats1 = cache.statistics();
        assert_eq!(stats1.total_misses, 1);
        assert_eq!(stats1.total_hits, 0);
        
        // Second access - should hit cache
        let compiled2 = cache.get_or_compile(&gate, compile_single_qubit_gate).unwrap();
        let stats2 = cache.statistics();
        assert_eq!(stats2.total_misses, 1);
        assert_eq!(stats2.total_hits, 1);
        
        // Verify same gate
        assert_eq!(compiled1.gate_id, compiled2.gate_id);
        assert_eq!(compiled1.matrix, compiled2.matrix);
    }

    #[test]
    fn test_cache_eviction() {
        let temp_dir = std::env::temp_dir().join(format!("quantrs_test_{}", std::process::id()));
        let config = CacheConfig {
            cache_dir: temp_dir,
            max_memory_entries: 2,
            enable_persistence: false,
            ..Default::default()
        };
        
        let cache = CompilationCache::new(config).unwrap();
        
        // Add three gates to trigger eviction
        for i in 0..3 {
            let gate = PauliX { target: QubitId(i) };
            let _ = cache.get_or_compile(&gate, compile_single_qubit_gate).unwrap();
        }
        
        let stats = cache.statistics();
        assert_eq!(stats.num_entries, 2); // One should have been evicted
    }

    #[test]
    fn test_persistent_cache() {
        let temp_dir = std::env::temp_dir().join(format!("quantrs_test_{}", std::process::id()));
        let config = CacheConfig {
            cache_dir: temp_dir,
            enable_persistence: true,
            async_writes: false,
            ..Default::default()
        };
        
        let gate = Hadamard { target: QubitId(0) };
        let gate_id;
        
        // Create cache and compile gate
        {
            let cache = CompilationCache::new(config.clone()).unwrap();
            let compiled = cache.get_or_compile(&gate, compile_single_qubit_gate).unwrap();
            gate_id = compiled.gate_id.clone();
        }
        
        // Create new cache instance and verify persistence
        {
            let cache = CompilationCache::new(config).unwrap();
            let compiled = cache.get_or_compile(&gate, compile_single_qubit_gate).unwrap();
            
            assert_eq!(compiled.gate_id, gate_id);
            
            let stats = cache.statistics();
            assert_eq!(stats.total_hits, 1); // Should hit persistent cache
            assert_eq!(stats.total_misses, 0);
        }
    }

    #[test]
    fn test_cache_optimization() {
        let temp_dir = std::env::temp_dir().join(format!("quantrs_test_{}", std::process::id()));
        let config = CacheConfig {
            cache_dir: temp_dir,
            enable_persistence: true,
            expiration_time: Duration::from_secs(0), // Immediate expiration
            async_writes: false,
            ..Default::default()
        };
        
        let cache = CompilationCache::new(config).unwrap();
        let gate = Hadamard { target: QubitId(0) };
        
        // Compile and cache gate
        let _ = cache.get_or_compile(&gate, compile_single_qubit_gate).unwrap();
        
        // Wait a bit and optimize
        std::thread::sleep(Duration::from_millis(100));
        cache.optimize().unwrap();
        
        // Try to access again - should miss due to expiration
        cache.clear().unwrap(); // Clear memory cache
        let _ = cache.get_or_compile(&gate, compile_single_qubit_gate).unwrap();
        
        let stats = cache.statistics();
        assert_eq!(stats.total_misses, 1); // Should have missed
    }

    #[test]
    fn test_precompile_common_gates() {
        let temp_dir = std::env::temp_dir().join(format!("quantrs_test_{}", std::process::id()));
        let config = CacheConfig {
            cache_dir: temp_dir,
            ..Default::default()
        };
        
        let cache = CompilationCache::new(config).unwrap();
        cache.precompile_common_gates().unwrap();
        
        let stats = cache.statistics();
        assert!(stats.num_entries > 0);
        println!("Precompiled {} gates", stats.num_entries);
    }

    #[test]
    fn test_statistics_export() {
        let temp_dir = std::env::temp_dir().join(format!("quantrs_test_{}", std::process::id()));
        let config = CacheConfig {
            cache_dir: temp_dir.clone(),
            ..Default::default()
        };
        
        let cache = CompilationCache::new(config).unwrap();
        
        // Generate some statistics
        let gate = Hadamard { target: QubitId(0) };
        let _ = cache.get_or_compile(&gate, compile_single_qubit_gate).unwrap();
        let _ = cache.get_or_compile(&gate, compile_single_qubit_gate).unwrap();
        
        // Export statistics
        let stats_path = temp_dir.join("stats.json");
        cache.export_statistics(&stats_path).unwrap();
        
        // Verify file exists and contains valid JSON
        assert!(stats_path.exists());
        let contents = fs::read_to_string(&stats_path).unwrap();
        let parsed: CacheStatistics = serde_json::from_str(&contents).unwrap();
        
        assert_eq!(parsed.total_hits, 1);
        assert_eq!(parsed.total_misses, 1);
    }
}