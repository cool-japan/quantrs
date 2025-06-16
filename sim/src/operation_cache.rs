//! Advanced Caching Strategies for Quantum Operations
//!
//! This module provides sophisticated caching mechanisms for frequently computed
//! quantum operations, gate matrices, and intermediate results to optimize performance.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Cache key for quantum operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationKey {
    /// Operation type identifier
    pub operation_type: String,
    /// Operation parameters (angles, targets, etc.)
    pub parameters: Vec<f64>,
    /// Qubit indices involved
    pub qubits: Vec<usize>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Hash for OperationKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.operation_type.hash(state);
        // Hash parameters with precision consideration
        for param in &self.parameters {
            // Round to avoid floating point precision issues
            let rounded = (param * 1e12).round() as i64;
            rounded.hash(state);
        }
        self.qubits.hash(state);
        for (k, v) in &self.metadata {
            k.hash(state);
            v.hash(state);
        }
    }
}

impl Eq for OperationKey {}

/// Cached operation result
#[derive(Debug, Clone)]
pub struct CachedOperation {
    /// The computed gate matrix or result
    pub data: CachedData,
    /// When this entry was created
    pub created_at: Instant,
    /// How many times this entry has been accessed
    pub access_count: u64,
    /// Last access time
    pub last_accessed: Instant,
    /// Size in bytes (for memory management)
    pub size_bytes: usize,
}

/// Different types of cached data
#[derive(Debug, Clone)]
pub enum CachedData {
    /// 2x2 single-qubit gate matrix
    SingleQubitMatrix(Array2<Complex64>),
    /// 4x4 two-qubit gate matrix
    TwoQubitMatrix(Array2<Complex64>),
    /// Arbitrary size matrix
    Matrix(Array2<Complex64>),
    /// State vector
    StateVector(Array1<Complex64>),
    /// Expectation value result
    ExpectationValue(Complex64),
    /// Probability distribution
    Probabilities(Vec<f64>),
    /// Custom data with serialization
    Custom(Vec<u8>),
}

impl CachedData {
    /// Estimate memory usage of cached data
    pub fn estimate_size(&self) -> usize {
        match self {
            CachedData::SingleQubitMatrix(_) => 4 * std::mem::size_of::<Complex64>(),
            CachedData::TwoQubitMatrix(_) => 16 * std::mem::size_of::<Complex64>(),
            CachedData::Matrix(m) => m.len() * std::mem::size_of::<Complex64>(),
            CachedData::StateVector(v) => v.len() * std::mem::size_of::<Complex64>(),
            CachedData::ExpectationValue(_) => std::mem::size_of::<Complex64>(),
            CachedData::Probabilities(p) => p.len() * std::mem::size_of::<f64>(),
            CachedData::Custom(data) => data.len(),
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based expiration
    TTL,
    /// First In, First Out
    FIFO,
    /// Hybrid policy combining LRU and LFU
    Hybrid,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// TTL for time-based eviction
    pub ttl: Duration,
    /// Enable cache statistics
    pub enable_stats: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            max_memory_bytes: 256 * 1024 * 1024, // 256 MB
            eviction_policy: EvictionPolicy::LRU,
            ttl: Duration::from_secs(3600), // 1 hour
            enable_stats: true,
            cleanup_interval: Duration::from_secs(60), // 1 minute
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache requests
    pub total_requests: u64,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Current number of entries
    pub current_entries: usize,
    /// Current memory usage
    pub current_memory_bytes: usize,
    /// Peak memory usage
    pub peak_memory_bytes: usize,
    /// Average access time (nanoseconds)
    pub average_access_time_ns: f64,
    /// Cache efficiency by operation type
    pub efficiency_by_type: HashMap<String, f64>,
}

impl CacheStats {
    /// Calculate hit ratio
    pub fn hit_ratio(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_requests as f64
        }
    }

    /// Update statistics for a cache access
    pub fn record_access(&mut self, operation_type: &str, hit: bool, access_time_ns: u64) {
        self.total_requests += 1;
        if hit {
            self.hits += 1;
        } else {
            self.misses += 1;
        }

        // Update average access time
        let total_time =
            self.average_access_time_ns * (self.total_requests - 1) as f64 + access_time_ns as f64;
        self.average_access_time_ns = total_time / self.total_requests as f64;

        // Update efficiency by operation type
        let type_stats = self
            .efficiency_by_type
            .entry(operation_type.to_string())
            .or_insert(0.0);
        // Simplified efficiency calculation
        if hit {
            *type_stats = (*type_stats + 1.0) / 2.0;
        } else {
            *type_stats = *type_stats / 2.0;
        }
    }
}

/// Advanced quantum operation cache
#[derive(Debug)]
pub struct QuantumOperationCache {
    /// Main cache storage
    cache: RwLock<HashMap<OperationKey, CachedOperation>>,
    /// Access order for LRU
    access_order: Mutex<VecDeque<OperationKey>>,
    /// Configuration
    config: CacheConfig,
    /// Statistics
    stats: Arc<Mutex<CacheStats>>,
    /// Last cleanup time
    last_cleanup: Mutex<Instant>,
}

impl QuantumOperationCache {
    /// Create new quantum operation cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            access_order: Mutex::new(VecDeque::new()),
            config,
            stats: Arc::new(Mutex::new(CacheStats::default())),
            last_cleanup: Mutex::new(Instant::now()),
        }
    }

    /// Get cached operation result
    pub fn get(&self, key: &OperationKey) -> Option<CachedData> {
        let start_time = Instant::now();

        let result = {
            let entry_data = {
                let cache = self.cache.read().unwrap();
                cache.get(key).map(|entry| entry.data.clone())
            };

            if entry_data.is_some() {
                self.update_access_stats(key);
            }

            entry_data
        };

        let access_time = start_time.elapsed().as_nanos() as u64;

        // Update statistics
        if self.config.enable_stats {
            if let Ok(mut stats) = self.stats.lock() {
                stats.record_access(&key.operation_type, result.is_some(), access_time);
            }
        }

        result
    }

    /// Store operation result in cache
    pub fn put(&self, key: OperationKey, data: CachedData) {
        let size = data.estimate_size();
        let entry = CachedOperation {
            data,
            created_at: Instant::now(),
            access_count: 0,
            last_accessed: Instant::now(),
            size_bytes: size,
        };

        // Check if we need to evict entries first
        self.maybe_evict(&key, size);

        // Insert the new entry
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(key.clone(), entry);
        }

        // Update access order
        {
            let mut access_order = self.access_order.lock().unwrap();
            access_order.push_back(key);
        }

        // Update statistics
        if self.config.enable_stats {
            if let Ok(mut stats) = self.stats.lock() {
                stats.current_entries = self.cache.read().unwrap().len();
                stats.current_memory_bytes += size;
                if stats.current_memory_bytes > stats.peak_memory_bytes {
                    stats.peak_memory_bytes = stats.current_memory_bytes;
                }
            }
        }

        // Trigger cleanup if needed
        self.maybe_cleanup();
    }

    /// Update access statistics for an entry
    fn update_access_stats(&self, key: &OperationKey) {
        let mut cache = self.cache.write().unwrap();
        if let Some(entry) = cache.get_mut(key) {
            entry.access_count += 1;
            entry.last_accessed = Instant::now();
        }

        // Update access order for LRU
        if self.config.eviction_policy == EvictionPolicy::LRU {
            let mut access_order = self.access_order.lock().unwrap();
            // Remove key from current position and add to back
            if let Some(pos) = access_order.iter().position(|k| k == key) {
                access_order.remove(pos);
            }
            access_order.push_back(key.clone());
        }
    }

    /// Check if eviction is needed and perform it
    fn maybe_evict(&self, _new_key: &OperationKey, new_size: usize) {
        let (current_entries, current_memory) = {
            let cache = self.cache.read().unwrap();
            (cache.len(), self.get_current_memory_usage())
        };

        let needs_eviction = current_entries >= self.config.max_entries
            || current_memory + new_size > self.config.max_memory_bytes;

        if needs_eviction {
            self.evict_entries();
        }
    }

    /// Evict entries based on configured policy
    fn evict_entries(&self) {
        match self.config.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru(),
            EvictionPolicy::LFU => self.evict_lfu(),
            EvictionPolicy::TTL => self.evict_expired(),
            EvictionPolicy::FIFO => self.evict_fifo(),
            EvictionPolicy::Hybrid => self.evict_hybrid(),
        }
    }

    /// Evict least recently used entries
    fn evict_lru(&self) {
        let keys_to_evict: Vec<OperationKey> = {
            let mut access_order = self.access_order.lock().unwrap();
            let mut keys = Vec::new();

            // Evict 25% of entries or until memory constraint is satisfied
            let max_evictions = (self.config.max_entries / 4).max(1);

            for _ in 0..max_evictions {
                if let Some(key) = access_order.pop_front() {
                    keys.push(key);
                } else {
                    break;
                }
            }
            keys
        };

        self.remove_entries(keys_to_evict);
    }

    /// Evict least frequently used entries
    fn evict_lfu(&self) {
        let keys_to_evict: Vec<OperationKey> = {
            let cache = self.cache.read().unwrap();
            let mut entries: Vec<_> = cache.iter().collect();

            // Sort by access count (ascending)
            entries.sort_by_key(|(_, entry)| entry.access_count);

            // Take first 25% for eviction
            let max_evictions = (cache.len() / 4).max(1);
            entries
                .into_iter()
                .take(max_evictions)
                .map(|(key, _)| key.clone())
                .collect()
        };

        self.remove_entries(keys_to_evict);
    }

    /// Evict expired entries based on TTL
    fn evict_expired(&self) {
        let now = Instant::now();
        let keys_to_evict: Vec<OperationKey> = {
            let cache = self.cache.read().unwrap();
            cache
                .iter()
                .filter(|(_, entry)| now.duration_since(entry.created_at) > self.config.ttl)
                .map(|(key, _)| key.clone())
                .collect()
        };

        self.remove_entries(keys_to_evict);
    }

    /// Evict in FIFO order
    fn evict_fifo(&self) {
        // Similar to LRU but based on creation time
        let keys_to_evict: Vec<OperationKey> = {
            let cache = self.cache.read().unwrap();
            let mut entries: Vec<_> = cache.iter().collect();

            // Sort by creation time (ascending)
            entries.sort_by_key(|(_, entry)| entry.created_at);

            // Take first 25% for eviction
            let max_evictions = (cache.len() / 4).max(1);
            entries
                .into_iter()
                .take(max_evictions)
                .map(|(key, _)| key.clone())
                .collect()
        };

        self.remove_entries(keys_to_evict);
    }

    /// Hybrid eviction combining LRU and LFU
    fn evict_hybrid(&self) {
        let keys_to_evict: Vec<OperationKey> = {
            let cache = self.cache.read().unwrap();
            let mut entries: Vec<_> = cache.iter().collect();

            // Hybrid score: combine recency and frequency
            entries.sort_by_key(|(_, entry)| {
                let recency_score = entry.last_accessed.elapsed().as_secs();
                let frequency_score = 1000 / (entry.access_count + 1); // Inverse frequency
                recency_score + frequency_score
            });

            // Take first 25% for eviction
            let max_evictions = (cache.len() / 4).max(1);
            entries
                .into_iter()
                .take(max_evictions)
                .map(|(key, _)| key.clone())
                .collect()
        };

        self.remove_entries(keys_to_evict);
    }

    /// Remove specified entries from cache
    fn remove_entries(&self, keys: Vec<OperationKey>) {
        let mut total_freed_memory = 0usize;

        {
            let mut cache = self.cache.write().unwrap();
            for key in &keys {
                if let Some(entry) = cache.remove(key) {
                    total_freed_memory += entry.size_bytes;
                }
            }
        }

        // Update access order
        {
            let mut access_order = self.access_order.lock().unwrap();
            access_order.retain(|key| !keys.contains(key));
        }

        // Update statistics
        if self.config.enable_stats {
            if let Ok(mut stats) = self.stats.lock() {
                stats.evictions += keys.len() as u64;
                stats.current_entries = self.cache.read().unwrap().len();
                stats.current_memory_bytes = stats
                    .current_memory_bytes
                    .saturating_sub(total_freed_memory);
            }
        }
    }

    /// Get current memory usage
    fn get_current_memory_usage(&self) -> usize {
        let cache = self.cache.read().unwrap();
        cache.values().map(|entry| entry.size_bytes).sum()
    }

    /// Periodic cleanup of expired entries
    fn maybe_cleanup(&self) {
        if let Ok(mut last_cleanup) = self.last_cleanup.try_lock() {
            if last_cleanup.elapsed() > self.config.cleanup_interval {
                self.evict_expired();
                *last_cleanup = Instant::now();
            }
        }
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();

        let mut access_order = self.access_order.lock().unwrap();
        access_order.clear();

        if self.config.enable_stats {
            if let Ok(mut stats) = self.stats.lock() {
                stats.current_entries = 0;
                stats.current_memory_bytes = 0;
            }
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get cache configuration
    pub fn get_config(&self) -> &CacheConfig {
        &self.config
    }
}

/// Specialized cache for gate matrices
pub struct GateMatrixCache {
    /// Internal operation cache
    cache: QuantumOperationCache,
}

impl GateMatrixCache {
    /// Create new gate matrix cache
    pub fn new() -> Self {
        let config = CacheConfig {
            max_entries: 5000,
            max_memory_bytes: 64 * 1024 * 1024, // 64 MB
            eviction_policy: EvictionPolicy::LRU,
            ttl: Duration::from_secs(1800), // 30 minutes
            enable_stats: true,
            cleanup_interval: Duration::from_secs(120), // 2 minutes
        };

        Self {
            cache: QuantumOperationCache::new(config),
        }
    }

    /// Get single-qubit gate matrix
    pub fn get_single_qubit_gate(
        &self,
        gate_name: &str,
        parameters: &[f64],
    ) -> Option<Array2<Complex64>> {
        let key = OperationKey {
            operation_type: format!("single_qubit_{}", gate_name),
            parameters: parameters.to_vec(),
            qubits: vec![],
            metadata: HashMap::new(),
        };

        match self.cache.get(&key) {
            Some(CachedData::SingleQubitMatrix(matrix)) => Some(matrix),
            Some(CachedData::Matrix(matrix)) if matrix.shape() == [2, 2] => Some(matrix),
            _ => None,
        }
    }

    /// Cache single-qubit gate matrix
    pub fn put_single_qubit_gate(
        &self,
        gate_name: &str,
        parameters: &[f64],
        matrix: Array2<Complex64>,
    ) {
        let key = OperationKey {
            operation_type: format!("single_qubit_{}", gate_name),
            parameters: parameters.to_vec(),
            qubits: vec![],
            metadata: HashMap::new(),
        };

        self.cache.put(key, CachedData::SingleQubitMatrix(matrix));
    }

    /// Get two-qubit gate matrix
    pub fn get_two_qubit_gate(
        &self,
        gate_name: &str,
        parameters: &[f64],
    ) -> Option<Array2<Complex64>> {
        let key = OperationKey {
            operation_type: format!("two_qubit_{}", gate_name),
            parameters: parameters.to_vec(),
            qubits: vec![],
            metadata: HashMap::new(),
        };

        match self.cache.get(&key) {
            Some(CachedData::TwoQubitMatrix(matrix)) => Some(matrix),
            Some(CachedData::Matrix(matrix)) if matrix.shape() == [4, 4] => Some(matrix),
            _ => None,
        }
    }

    /// Cache two-qubit gate matrix
    pub fn put_two_qubit_gate(
        &self,
        gate_name: &str,
        parameters: &[f64],
        matrix: Array2<Complex64>,
    ) {
        let key = OperationKey {
            operation_type: format!("two_qubit_{}", gate_name),
            parameters: parameters.to_vec(),
            qubits: vec![],
            metadata: HashMap::new(),
        };

        self.cache.put(key, CachedData::TwoQubitMatrix(matrix));
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.cache.get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_operation_cache() {
        let config = CacheConfig::default();
        let cache = QuantumOperationCache::new(config);

        let key = OperationKey {
            operation_type: "test_op".to_string(),
            parameters: vec![0.5, 1.0],
            qubits: vec![0, 1],
            metadata: HashMap::new(),
        };

        let matrix = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
        ];

        // Test cache miss
        assert!(cache.get(&key).is_none());

        // Test cache put and hit
        cache.put(key.clone(), CachedData::Matrix(matrix.clone()));

        match cache.get(&key) {
            Some(CachedData::Matrix(cached_matrix)) => {
                assert_eq!(cached_matrix, matrix);
            }
            _ => panic!("Expected cached matrix"),
        }

        let stats = cache.get_stats();
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_gate_matrix_cache() {
        let cache = GateMatrixCache::new();

        let pauli_x = array![
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
        ];

        // Test cache miss
        assert!(cache.get_single_qubit_gate("X", &[]).is_none());

        // Test cache put and hit
        cache.put_single_qubit_gate("X", &[], pauli_x.clone());

        let cached_matrix = cache.get_single_qubit_gate("X", &[]).unwrap();
        assert_eq!(cached_matrix, pauli_x);
    }

    #[test]
    fn test_eviction_policies() {
        let mut config = CacheConfig::default();
        config.max_entries = 2;

        let cache = QuantumOperationCache::new(config);

        // Fill cache to capacity
        for i in 0..3 {
            let key = OperationKey {
                operation_type: format!("test_{}", i),
                parameters: vec![i as f64],
                qubits: vec![i],
                metadata: HashMap::new(),
            };
            cache.put(
                key,
                CachedData::ExpectationValue(Complex64::new(i as f64, 0.0)),
            );
        }

        // Should have evicted first entry
        let stats = cache.get_stats();
        assert_eq!(stats.current_entries, 2);
        assert!(stats.evictions > 0);
    }
}
