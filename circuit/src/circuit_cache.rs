//! Circuit caching for repeated execution
//!
//! This module provides sophisticated caching mechanisms for quantum circuits to avoid
//! redundant compilation, optimization, and execution overhead when circuits are reused
//! across multiple invocations.

use crate::builder::Circuit;
use crate::simulator_interface::{CompiledCircuit, ExecutionResult};
use crate::transpiler::TranspilationResult;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Cache entry for compiled circuits
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// Cached value
    pub value: T,
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Last access timestamp
    pub last_accessed: SystemTime,
    /// Access count
    pub access_count: usize,
    /// Cache key hash
    pub key_hash: u64,
    /// Size estimate in bytes
    pub size_bytes: usize,
}

/// Cache eviction policy
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time To Live
    TTL(Duration),
    /// First In First Out
    FIFO,
    /// Random eviction
    Random,
    /// Size-based eviction
    SizeBased,
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
    /// Enable compression for stored circuits
    pub enable_compression: bool,
    /// Cache hit statistics collection
    pub collect_stats: bool,
    /// Persistence to disk
    pub persist_to_disk: bool,
    /// Disk cache directory
    pub cache_directory: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            max_memory_bytes: 100 * 1024 * 1024, // 100 MB
            eviction_policy: EvictionPolicy::LRU,
            enable_compression: true,
            collect_stats: true,
            persist_to_disk: false,
            cache_directory: None,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: usize,
    /// Total cache misses
    pub misses: usize,
    /// Total evictions
    pub evictions: usize,
    /// Current memory usage
    pub memory_usage_bytes: usize,
    /// Average access time
    pub avg_access_time: Duration,
    /// Cache efficiency ratio
    pub hit_ratio: f64,
}

/// Circuit signature for cache key generation
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CircuitSignature {
    /// Number of qubits
    pub num_qubits: usize,
    /// Gate sequence hash
    pub gate_sequence_hash: u64,
    /// Parameter hash (for parametrized gates)
    pub parameter_hash: u64,
    /// Compiler options hash
    pub options_hash: u64,
}

/// Generic cache implementation with configurable policies
pub struct CircuitCache<T: Clone> {
    /// Cache entries
    entries: Arc<RwLock<HashMap<u64, CacheEntry<T>>>>,
    /// Access order for LRU
    access_order: Arc<Mutex<VecDeque<u64>>>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics
    stats: Arc<Mutex<CacheStats>>,
    /// Background cleanup thread handle
    cleanup_handle: Option<std::thread::JoinHandle<()>>,
}

impl<T: Clone + Send + Sync + 'static> CircuitCache<T> {
    /// Create a new circuit cache
    pub fn new(config: CacheConfig) -> Self {
        let cache = Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(Mutex::new(VecDeque::new())),
            config,
            stats: Arc::new(Mutex::new(CacheStats::default())),
            cleanup_handle: None,
        };

        cache
    }

    /// Create cache with default configuration
    pub fn with_default_config() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Get value from cache
    pub fn get(&self, key: &CircuitSignature) -> Option<T> {
        let start_time = Instant::now();
        let key_hash = self.hash_signature(key);

        let result = {
            let entries = self.entries.read().unwrap();
            entries.get(&key_hash).map(|entry| entry.value.clone())
        };

        if let Some(value) = &result {
            self.update_access(key_hash);
            if self.config.collect_stats {
                self.update_stats(true, start_time.elapsed());
            }
        } else if self.config.collect_stats {
            self.update_stats(false, start_time.elapsed());
        }

        result
    }

    /// Put value into cache
    pub fn put(&self, key: CircuitSignature, value: T) -> QuantRS2Result<()> {
        let key_hash = self.hash_signature(&key);
        let size_estimate = self.estimate_size(&value);

        // Check if we need to evict entries
        self.ensure_capacity(size_estimate)?;

        let entry = CacheEntry {
            value,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            access_count: 1,
            key_hash,
            size_bytes: size_estimate,
        };

        {
            let mut entries = self.entries.write().unwrap();
            entries.insert(key_hash, entry);
        }

        self.update_access(key_hash);
        self.update_memory_usage(size_estimate as isize);

        Ok(())
    }

    /// Remove entry from cache
    pub fn remove(&self, key: &CircuitSignature) -> Option<T> {
        let key_hash = self.hash_signature(key);

        let removed = {
            let mut entries = self.entries.write().unwrap();
            entries.remove(&key_hash)
        };

        if let Some(entry) = &removed {
            self.remove_from_access_order(key_hash);
            self.update_memory_usage(-(entry.size_bytes as isize));
        }

        removed.map(|entry| entry.value)
    }

    /// Clear all cache entries
    pub fn clear(&self) {
        {
            let mut entries = self.entries.write().unwrap();
            entries.clear();
        }

        {
            let mut access_order = self.access_order.lock().unwrap();
            access_order.clear();
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.memory_usage_bytes = 0;
        }
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        self.stats.lock().unwrap().clone()
    }

    /// Get current cache size
    pub fn size(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    /// Check if cache contains key
    pub fn contains_key(&self, key: &CircuitSignature) -> bool {
        let key_hash = self.hash_signature(key);
        self.entries.read().unwrap().contains_key(&key_hash)
    }

    /// Start background cleanup process
    pub fn start_cleanup(&mut self, interval: Duration) {
        let entries = Arc::clone(&self.entries);
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);

        let handle = std::thread::spawn(move || {
            loop {
                std::thread::sleep(interval);

                if let Ok(mut entries_guard) = entries.write() {
                    let now = SystemTime::now();
                    let mut to_remove = Vec::new();

                    // TTL-based cleanup
                    if let EvictionPolicy::TTL(ttl) = &config.eviction_policy {
                        for (key, entry) in entries_guard.iter() {
                            if let Ok(elapsed) = now.duration_since(entry.created_at) {
                                if elapsed > *ttl {
                                    to_remove.push(*key);
                                }
                            }
                        }

                        for key in to_remove {
                            if let Some(entry) = entries_guard.remove(&key) {
                                if let Ok(mut stats_guard) = stats.lock() {
                                    stats_guard.evictions += 1;
                                    stats_guard.memory_usage_bytes = stats_guard
                                        .memory_usage_bytes
                                        .saturating_sub(entry.size_bytes);
                                }
                            }
                        }
                    }
                }
            }
        });

        self.cleanup_handle = Some(handle);
    }

    /// Hash circuit signature
    fn hash_signature(&self, signature: &CircuitSignature) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        signature.hash(&mut hasher);
        hasher.finish()
    }

    /// Update access information
    fn update_access(&self, key_hash: u64) {
        // Update last accessed time
        if let Ok(mut entries) = self.entries.write() {
            if let Some(entry) = entries.get_mut(&key_hash) {
                entry.last_accessed = SystemTime::now();
                entry.access_count += 1;
            }
        }

        // Update access order for LRU
        if matches!(self.config.eviction_policy, EvictionPolicy::LRU) {
            let mut access_order = self.access_order.lock().unwrap();

            // Remove from current position
            access_order.retain(|&x| x != key_hash);

            // Add to front
            access_order.push_front(key_hash);
        }
    }

    /// Remove from access order tracking
    fn remove_from_access_order(&self, key_hash: u64) {
        let mut access_order = self.access_order.lock().unwrap();
        access_order.retain(|&x| x != key_hash);
    }

    /// Update cache statistics
    fn update_stats(&self, hit: bool, access_time: Duration) {
        if let Ok(mut stats) = self.stats.lock() {
            if hit {
                stats.hits += 1;
            } else {
                stats.misses += 1;
            }

            let total_accesses = stats.hits + stats.misses;
            stats.hit_ratio = stats.hits as f64 / total_accesses as f64;

            // Update average access time (simple moving average)
            let current_avg_nanos = stats.avg_access_time.as_nanos() as f64;
            let new_avg_nanos = (current_avg_nanos * (total_accesses - 1) as f64
                + access_time.as_nanos() as f64)
                / total_accesses as f64;
            stats.avg_access_time = Duration::from_nanos(new_avg_nanos as u64);
        }
    }

    /// Update memory usage statistics
    fn update_memory_usage(&self, delta: isize) {
        if let Ok(mut stats) = self.stats.lock() {
            if delta > 0 {
                stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_add(delta as usize);
            } else {
                stats.memory_usage_bytes =
                    stats.memory_usage_bytes.saturating_sub((-delta) as usize);
            }
        }
    }

    /// Ensure cache has capacity for new entry
    fn ensure_capacity(&self, new_entry_size: usize) -> QuantRS2Result<()> {
        let current_stats = self.get_stats();
        let would_exceed_memory =
            current_stats.memory_usage_bytes + new_entry_size > self.config.max_memory_bytes;
        let would_exceed_count = self.size() >= self.config.max_entries;

        if would_exceed_memory || would_exceed_count {
            self.evict_entries(new_entry_size)?;
        }

        Ok(())
    }

    /// Evict entries based on policy
    fn evict_entries(&self, needed_space: usize) -> QuantRS2Result<()> {
        match &self.config.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru(needed_space),
            EvictionPolicy::LFU => self.evict_lfu(needed_space),
            EvictionPolicy::FIFO => self.evict_fifo(needed_space),
            EvictionPolicy::Random => self.evict_random(needed_space),
            EvictionPolicy::SizeBased => self.evict_size_based(needed_space),
            EvictionPolicy::TTL(_) => Ok(()), // TTL is handled by background cleanup
        }
    }

    /// Evict using LRU policy
    fn evict_lru(&self, needed_space: usize) -> QuantRS2Result<()> {
        let mut freed_space = 0;
        let mut evicted_count = 0;

        while freed_space < needed_space && self.size() > 0 {
            let key_to_evict = {
                let access_order = self.access_order.lock().unwrap();
                access_order.back().copied()
            };

            if let Some(key) = key_to_evict {
                if let Ok(mut entries) = self.entries.write() {
                    if let Some(entry) = entries.remove(&key) {
                        freed_space += entry.size_bytes;
                        evicted_count += 1;
                        self.remove_from_access_order(key);
                    }
                }
            } else {
                break;
            }
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.evictions += evicted_count;
            stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(freed_space);
        }

        Ok(())
    }

    /// Evict using LFU policy
    fn evict_lfu(&self, needed_space: usize) -> QuantRS2Result<()> {
        let mut freed_space = 0;
        let mut evicted_count = 0;

        while freed_space < needed_space && self.size() > 0 {
            let key_to_evict = {
                let entries = self.entries.read().unwrap();
                entries
                    .iter()
                    .min_by_key(|(_, entry)| entry.access_count)
                    .map(|(key, _)| *key)
            };

            if let Some(key) = key_to_evict {
                if let Ok(mut entries) = self.entries.write() {
                    if let Some(entry) = entries.remove(&key) {
                        freed_space += entry.size_bytes;
                        evicted_count += 1;
                        self.remove_from_access_order(key);
                    }
                }
            } else {
                break;
            }
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.evictions += evicted_count;
            stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(freed_space);
        }

        Ok(())
    }

    /// Evict using FIFO policy
    fn evict_fifo(&self, needed_space: usize) -> QuantRS2Result<()> {
        let mut freed_space = 0;
        let mut evicted_count = 0;

        while freed_space < needed_space && self.size() > 0 {
            let key_to_evict = {
                let entries = self.entries.read().unwrap();
                entries
                    .iter()
                    .min_by_key(|(_, entry)| entry.created_at)
                    .map(|(key, _)| *key)
            };

            if let Some(key) = key_to_evict {
                if let Ok(mut entries) = self.entries.write() {
                    if let Some(entry) = entries.remove(&key) {
                        freed_space += entry.size_bytes;
                        evicted_count += 1;
                        self.remove_from_access_order(key);
                    }
                }
            } else {
                break;
            }
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.evictions += evicted_count;
            stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(freed_space);
        }

        Ok(())
    }

    /// Evict random entries
    fn evict_random(&self, needed_space: usize) -> QuantRS2Result<()> {
        use rand::seq::IteratorRandom;
        let mut rng = rand::thread_rng();
        let mut freed_space = 0;
        let mut evicted_count = 0;

        while freed_space < needed_space && self.size() > 0 {
            let key_to_evict = {
                let entries = self.entries.read().unwrap();
                entries.keys().choose(&mut rng).copied()
            };

            if let Some(key) = key_to_evict {
                if let Ok(mut entries) = self.entries.write() {
                    if let Some(entry) = entries.remove(&key) {
                        freed_space += entry.size_bytes;
                        evicted_count += 1;
                        self.remove_from_access_order(key);
                    }
                }
            } else {
                break;
            }
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.evictions += evicted_count;
            stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(freed_space);
        }

        Ok(())
    }

    /// Evict largest entries first
    fn evict_size_based(&self, needed_space: usize) -> QuantRS2Result<()> {
        let mut freed_space = 0;
        let mut evicted_count = 0;

        while freed_space < needed_space && self.size() > 0 {
            let key_to_evict = {
                let entries = self.entries.read().unwrap();
                entries
                    .iter()
                    .max_by_key(|(_, entry)| entry.size_bytes)
                    .map(|(key, _)| *key)
            };

            if let Some(key) = key_to_evict {
                if let Ok(mut entries) = self.entries.write() {
                    if let Some(entry) = entries.remove(&key) {
                        freed_space += entry.size_bytes;
                        evicted_count += 1;
                        self.remove_from_access_order(key);
                    }
                }
            } else {
                break;
            }
        }

        if let Ok(mut stats) = self.stats.lock() {
            stats.evictions += evicted_count;
            stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(freed_space);
        }

        Ok(())
    }

    /// Estimate size of cached value
    fn estimate_size(&self, _value: &T) -> usize {
        // This is a simplified size estimation
        // In practice, this would use serialization or memory introspection
        std::mem::size_of::<T>() + 1024 // Base overhead estimate
    }
}

/// Specialized cache for compiled circuits
pub type CompiledCircuitCache = CircuitCache<CompiledCircuit>;

/// Specialized cache for execution results
pub type ExecutionResultCache = CircuitCache<ExecutionResult>;

/// Specialized cache for transpilation results
pub type TranspilationCache<const N: usize> = CircuitCache<TranspilationResult<N>>;

/// Circuit signature generator for different circuit types
pub struct SignatureGenerator;

impl SignatureGenerator {
    /// Generate signature for a circuit
    pub fn generate_circuit_signature<const N: usize>(
        circuit: &Circuit<N>,
        options_hash: u64,
    ) -> CircuitSignature {
        use std::collections::hash_map::DefaultHasher;

        let mut gate_hasher = DefaultHasher::new();
        let mut param_hasher = DefaultHasher::new();

        // Hash gate sequence
        for gate in circuit.gates() {
            gate.name().hash(&mut gate_hasher);
            for qubit in gate.qubits() {
                qubit.id().hash(&mut gate_hasher);
            }

            // Hash parameters if any (simplified)
            // In practice, this would extract actual gate parameters
            0u64.hash(&mut param_hasher);
        }

        CircuitSignature {
            num_qubits: N,
            gate_sequence_hash: gate_hasher.finish(),
            parameter_hash: param_hasher.finish(),
            options_hash,
        }
    }

    /// Generate signature with compilation options
    pub fn generate_with_compilation_options<const N: usize>(
        circuit: &Circuit<N>,
        backend: &str,
        optimization_level: u32,
    ) -> CircuitSignature {
        use std::collections::hash_map::DefaultHasher;

        let mut options_hasher = DefaultHasher::new();
        backend.hash(&mut options_hasher);
        optimization_level.hash(&mut options_hasher);

        Self::generate_circuit_signature(circuit, options_hasher.finish())
    }

    /// Generate signature with transpilation options
    pub fn generate_with_transpilation_options<const N: usize>(
        circuit: &Circuit<N>,
        device: &str,
        strategy: &str,
    ) -> CircuitSignature {
        use std::collections::hash_map::DefaultHasher;

        let mut options_hasher = DefaultHasher::new();
        device.hash(&mut options_hasher);
        strategy.hash(&mut options_hasher);

        Self::generate_circuit_signature(circuit, options_hasher.finish())
    }
}

/// High-level cache manager that coordinates different cache types
pub struct CacheManager {
    /// Cache for compiled circuits
    pub compiled_cache: CompiledCircuitCache,
    /// Cache for execution results
    pub execution_cache: ExecutionResultCache,
    /// Global cache configuration
    config: CacheConfig,
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new(config: CacheConfig) -> Self {
        Self {
            compiled_cache: CompiledCircuitCache::new(config.clone()),
            execution_cache: ExecutionResultCache::new(config.clone()),
            config,
        }
    }

    /// Create with default configuration
    pub fn with_default_config() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Get aggregated cache statistics
    pub fn get_aggregated_stats(&self) -> HashMap<String, CacheStats> {
        let mut stats = HashMap::new();
        stats.insert("compiled".to_string(), self.compiled_cache.get_stats());
        stats.insert("execution".to_string(), self.execution_cache.get_stats());
        stats
    }

    /// Clear all caches
    pub fn clear_all(&self) {
        self.compiled_cache.clear();
        self.execution_cache.clear();
    }

    /// Start background cleanup for all caches
    pub fn start_background_cleanup(&mut self, interval: Duration) {
        self.compiled_cache.start_cleanup(interval);
        self.execution_cache.start_cleanup(interval);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::single::Hadamard;
    use quantrs2_core::qubit::QubitId;

    #[test]
    fn test_cache_creation() {
        let cache: CircuitCache<String> = CircuitCache::with_default_config();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_put_get() {
        let cache: CircuitCache<String> = CircuitCache::with_default_config();
        let signature = CircuitSignature {
            num_qubits: 2,
            gate_sequence_hash: 12345,
            parameter_hash: 67890,
            options_hash: 11111,
        };

        cache
            .put(signature.clone(), "test_value".to_string())
            .unwrap();
        let result = cache.get(&signature);
        assert_eq!(result, Some("test_value".to_string()));
    }

    #[test]
    fn test_cache_miss() {
        let cache: CircuitCache<String> = CircuitCache::with_default_config();
        let signature = CircuitSignature {
            num_qubits: 2,
            gate_sequence_hash: 12345,
            parameter_hash: 67890,
            options_hash: 11111,
        };

        let result = cache.get(&signature);
        assert_eq!(result, None);
    }

    #[test]
    fn test_signature_generation() {
        let mut circuit = Circuit::<2>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();

        let signature = SignatureGenerator::generate_circuit_signature(&circuit, 0);
        assert_eq!(signature.num_qubits, 2);
        assert_ne!(signature.gate_sequence_hash, 0);
    }

    #[test]
    fn test_cache_stats() {
        let cache: CircuitCache<String> = CircuitCache::with_default_config();
        let signature = CircuitSignature {
            num_qubits: 2,
            gate_sequence_hash: 12345,
            parameter_hash: 67890,
            options_hash: 11111,
        };

        // Miss
        cache.get(&signature);

        // Put and hit
        cache.put(signature.clone(), "test".to_string()).unwrap();
        cache.get(&signature);

        let stats = cache.get_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_ratio, 0.5);
    }

    #[test]
    fn test_cache_manager() {
        let manager = CacheManager::with_default_config();
        let stats = manager.get_aggregated_stats();
        assert!(stats.contains_key("compiled"));
        assert!(stats.contains_key("execution"));
    }
}
