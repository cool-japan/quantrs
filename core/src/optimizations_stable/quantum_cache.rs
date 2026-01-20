//! Stable Quantum Computing Cache System
//!
//! High-performance caching for quantum gate matrices, decompositions, and results
//! using only stable Rust features and standard library components.

use crate::error::QuantRS2Result;
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, Instant};

/// Cache key for quantum computations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub operation: String,
    pub parameters: Vec<u64>, // Quantized parameters for consistent hashing
    pub qubit_count: usize,
}

impl CacheKey {
    /// Create a new cache key
    pub fn new(operation: &str, params: Vec<f64>, qubit_count: usize) -> Self {
        // Quantize floating point parameters for consistent hashing
        let quantized_params: Vec<u64> = params
            .into_iter()
            .map(|p| {
                // Quantize to avoid floating point precision issues
                (p * 1_000_000.0).round() as u64
            })
            .collect();

        Self {
            operation: operation.to_string(),
            parameters: quantized_params,
            qubit_count,
        }
    }
}

/// Cached quantum computation results
#[derive(Debug, Clone)]
pub enum CachedResult {
    Matrix(Vec<Complex64>),
    StateVector(Vec<Complex64>),
    Probability(Vec<f64>),
    Scalar(Complex64),
    Decomposition(Vec<String>),
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    result: CachedResult,
    created_at: Instant,
    access_count: u64,
    last_accessed: Instant,
    /// Monotonic sequence number for LRU tracking (higher = more recently accessed)
    access_sequence: u64,
}

/// High-performance quantum cache with LRU eviction
pub struct StableQuantumCache {
    entries: Arc<RwLock<HashMap<CacheKey, CacheEntry>>>,
    max_size: usize,
    max_age: Duration,
    stats: Arc<RwLock<CacheStatistics>>,
    /// Monotonic counter for access sequence tracking
    access_counter: AtomicU64,
}

/// Cache performance statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_size: usize,
    pub average_access_count: f64,
    pub oldest_entry_age: Duration,
}

impl StableQuantumCache {
    /// Create a new quantum cache
    pub fn new(max_size: usize, max_age_seconds: u64) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            max_age: Duration::from_secs(max_age_seconds),
            stats: Arc::new(RwLock::new(CacheStatistics::default())),
            access_counter: AtomicU64::new(0),
        }
    }

    /// Insert a result into the cache
    pub fn insert(&self, key: CacheKey, result: CachedResult) {
        let now = Instant::now();
        let seq = self.access_counter.fetch_add(1, Ordering::Relaxed);
        let entry = CacheEntry {
            result,
            created_at: now,
            access_count: 0,
            last_accessed: now,
            access_sequence: seq,
        };

        {
            let mut entries = self.entries.write().expect("Cache entries lock poisoned");
            entries.insert(key, entry);

            // Perform maintenance if needed
            if entries.len() > self.max_size {
                self.evict_lru(&mut entries);
            }
        }

        // Update statistics
        let mut stats = self.stats.write().expect("Cache stats lock poisoned");
        stats.total_size += 1;
    }

    /// Get a result from the cache
    pub fn get(&self, key: &CacheKey) -> Option<CachedResult> {
        let now = Instant::now();

        // Check if entry exists and is not expired
        let result = {
            let mut entries = self.entries.write().expect("Cache entries lock poisoned");
            if let Some(entry) = entries.get_mut(key) {
                // Check if entry is expired
                if now.duration_since(entry.created_at) > self.max_age {
                    entries.remove(key);
                    let mut stats = self.stats.write().expect("Cache stats lock poisoned");
                    stats.misses += 1;
                    stats.evictions += 1;
                    return None;
                }

                // Update access statistics
                entry.access_count += 1;
                entry.last_accessed = now;
                entry.access_sequence = self.access_counter.fetch_add(1, Ordering::Relaxed);

                let mut stats = self.stats.write().expect("Cache stats lock poisoned");
                stats.hits += 1;

                Some(entry.result.clone())
            } else {
                let mut stats = self.stats.write().expect("Cache stats lock poisoned");
                stats.misses += 1;
                None
            }
        };

        result
    }

    /// Check if a key exists in the cache
    pub fn contains(&self, key: &CacheKey) -> bool {
        let entries = self.entries.read().expect("Cache entries lock poisoned");
        entries.contains_key(key)
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        let mut entries = self.entries.write().expect("Cache entries lock poisoned");
        entries.clear();

        let mut stats = self.stats.write().expect("Cache stats lock poisoned");
        *stats = CacheStatistics::default();
    }

    /// Perform LRU eviction
    fn evict_lru(&self, entries: &mut HashMap<CacheKey, CacheEntry>) {
        // Find the least recently used entry using min_by_key on access_sequence
        // The entry with the smallest access_sequence is the least recently accessed
        if let Some(oldest_key) = entries
            .iter()
            .min_by_key(|(_, entry)| entry.access_sequence)
            .map(|(key, _)| key.clone())
        {
            entries.remove(&oldest_key);
            let mut stats = self.stats.write().expect("Cache stats lock poisoned");
            stats.evictions += 1;
        }
    }

    /// Remove expired entries
    pub fn cleanup_expired(&self) {
        let now = Instant::now();
        let mut entries = self.entries.write().expect("Cache entries lock poisoned");
        let mut expired_keys = Vec::new();

        for (key, entry) in entries.iter() {
            if now.duration_since(entry.created_at) > self.max_age {
                expired_keys.push(key.clone());
            }
        }

        let expired_count = expired_keys.len();
        for key in expired_keys {
            entries.remove(&key);
        }

        let mut stats = self.stats.write().expect("Cache stats lock poisoned");
        stats.evictions += expired_count as u64;
    }

    /// Get cache statistics
    pub fn get_statistics(&self) -> CacheStatistics {
        let entries = self.entries.read().expect("Cache entries lock poisoned");
        let mut stats = self
            .stats
            .read()
            .expect("Cache stats lock poisoned")
            .clone();

        // Update computed statistics
        stats.total_size = entries.len();

        if !entries.is_empty() {
            let total_accesses: u64 = entries.values().map(|e| e.access_count).sum();
            stats.average_access_count = total_accesses as f64 / entries.len() as f64;

            if let Some(oldest_entry) = entries.values().min_by_key(|e| e.created_at) {
                stats.oldest_entry_age = Instant::now().duration_since(oldest_entry.created_at);
            }
        }

        stats
    }

    /// Get cache hit ratio
    pub fn hit_ratio(&self) -> f64 {
        let stats = self.stats.read().expect("Cache stats lock poisoned");
        if stats.hits + stats.misses == 0 {
            0.0
        } else {
            stats.hits as f64 / (stats.hits + stats.misses) as f64
        }
    }

    /// Get memory usage estimate in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let entries = self.entries.read().expect("Cache entries lock poisoned");
        let mut total_size = 0;

        for (key, entry) in entries.iter() {
            // Estimate key size
            total_size += key.operation.len();
            total_size += key.parameters.len() * 8; // u64 = 8 bytes
            total_size += 8; // qubit_count

            // Estimate result size
            total_size += match &entry.result {
                CachedResult::Matrix(m) => m.len() * 16, // Complex64 = 16 bytes
                CachedResult::StateVector(s) => s.len() * 16,
                CachedResult::Probability(p) => p.len() * 8, // f64 = 8 bytes
                CachedResult::Scalar(_) => 16,
                CachedResult::Decomposition(d) => d.iter().map(|s| s.len()).sum(),
            };

            // Metadata size
            total_size += 32; // Approximate size of CacheEntry metadata
        }

        total_size
    }
}

/// Global quantum cache instance
static GLOBAL_CACHE: OnceLock<StableQuantumCache> = OnceLock::new();

/// Get the global quantum cache
pub fn get_global_cache() -> &'static StableQuantumCache {
    GLOBAL_CACHE.get_or_init(|| {
        StableQuantumCache::new(
            4096, // 4K entries
            3600, // 1 hour TTL
        )
    })
}

/// Macro for easy caching of quantum computations
#[macro_export]
macro_rules! cached_quantum_computation {
    ($operation:expr, $params:expr, $qubits:expr, $compute:expr) => {{
        let cache = $crate::optimizations_stable::quantum_cache::get_global_cache();
        let key = $crate::optimizations_stable::quantum_cache::CacheKey::new(
            $operation, $params, $qubits,
        );

        if let Some(result) = cache.get(&key) {
            result
        } else {
            let computed_result = $compute;
            cache.insert(key, computed_result.clone());
            computed_result
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let cache = StableQuantumCache::new(100, 60);

        let key = CacheKey::new("test_op", vec![1.0], 2);
        let result = CachedResult::Scalar(Complex64::new(1.0, 0.0));

        // Test insertion and retrieval
        cache.insert(key.clone(), result.clone());
        let retrieved = cache
            .get(&key)
            .expect("Cache should contain the inserted key");

        match (&result, &retrieved) {
            (CachedResult::Scalar(a), CachedResult::Scalar(b)) => {
                assert!((a - b).norm() < 1e-10);
            }
            _ => panic!("Wrong result type"),
        }

        // Test statistics
        let stats = cache.get_statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_key_quantization() {
        let key1 = CacheKey::new("rx", vec![std::f64::consts::PI], 1);
        let key2 = CacheKey::new("rx", vec![std::f64::consts::PI + 1e-10], 1);

        // Should be the same after quantization
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let cache = StableQuantumCache::new(2, 60); // Small cache for testing

        let key1 = CacheKey::new("op1", vec![], 1);
        let key2 = CacheKey::new("op2", vec![], 1);
        let key3 = CacheKey::new("op3", vec![], 1);

        let result = CachedResult::Scalar(Complex64::new(1.0, 0.0));

        // Fill cache to capacity
        cache.insert(key1.clone(), result.clone());
        cache.insert(key2.clone(), result.clone());

        // Access key1 to make it more recently used
        let _ = cache.get(&key1);

        // Insert key3, should evict key2 (least recently used)
        cache.insert(key3.clone(), result.clone());

        assert!(cache.contains(&key1)); // Should still exist
        assert!(!cache.contains(&key2)); // Should be evicted
        assert!(cache.contains(&key3)); // Should exist
    }

    #[test]
    fn test_memory_usage_estimation() {
        let cache = StableQuantumCache::new(100, 60);

        // Insert a matrix result
        let key = CacheKey::new("matrix_op", vec![1.0], 2);
        let matrix = vec![Complex64::new(1.0, 0.0); 16]; // 4x4 matrix
        let result = CachedResult::Matrix(matrix);

        cache.insert(key, result);

        let memory_usage = cache.estimated_memory_usage();
        assert!(memory_usage > 0);

        // Should include matrix data (16 * 16 bytes) plus metadata
        assert!(memory_usage >= 256);
    }

    #[test]
    fn test_hit_ratio_calculation() {
        let cache = StableQuantumCache::new(100, 60);

        let key1 = CacheKey::new("op1", vec![], 1);
        let key2 = CacheKey::new("op2", vec![], 1);
        let result = CachedResult::Scalar(Complex64::new(1.0, 0.0));

        // No operations yet - hit ratio should be 0
        assert_eq!(cache.hit_ratio(), 0.0);

        // Insert and hit once
        cache.insert(key1.clone(), result);
        let _ = cache.get(&key1); // Hit
        let _ = cache.get(&key2); // Miss

        // Should be 50% hit ratio
        assert!((cache.hit_ratio() - 0.5).abs() < 1e-10);
    }
}
