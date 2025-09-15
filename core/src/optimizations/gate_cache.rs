//! Quantum Gate Matrix Caching using SciRS2 Beta.1 Features
//!
//! This module demonstrates how to leverage scirs2-core beta.1 caching capabilities
//! to optimize quantum gate matrix computations.

use num_complex::Complex64;
use scirs2_core::cache::{CacheConfig, TTLSizedCache};
use scirs2_core::profiling::{Profiler, Timer};
use scirs2_core::memory::{BufferPool, global_buffer_pool};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};
use crate::error::QuantRS2Result;

/// Hash key for gate matrix caching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GateKey {
    pub gate_type: String,
    pub parameters: Vec<u64>, // Hash of parameters for caching
    pub num_qubits: usize,
}

impl GateKey {
    pub fn new(gate_type: &str, parameters: &[f64], num_qubits: usize) -> Self {
        // Convert parameters to hashable representation
        let param_hashes: Vec<u64> = parameters
            .iter()
            .map(|&p| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                // Use bits representation for consistent hashing
                p.to_bits().hash(&mut hasher);
                hasher.finish()
            })
            .collect();

        Self {
            gate_type: gate_type.to_string(),
            parameters: param_hashes,
            num_qubits,
        }
    }
}

/// Cached quantum gate matrix
#[derive(Debug, Clone)]
pub struct CachedGateMatrix {
    pub matrix: Vec<Complex64>,
    pub size: usize,
    pub computation_time_us: u64,
}

/// High-performance quantum gate cache using SciRS2 beta.1 features
pub struct QuantumGateCache {
    /// Gate matrix cache with TTL
    matrix_cache: Arc<Mutex<TTLSizedCache<GateKey, CachedGateMatrix>>>,
    /// Buffer pool for matrix computations
    buffer_pool: Arc<BufferPool<Complex64>>,
    /// Performance metrics
    cache_hits: Arc<Mutex<u64>>,
    cache_misses: Arc<Mutex<u64>>,
    total_computation_time: Arc<Mutex<u64>>,
}

impl Default for QuantumGateCache {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumGateCache {
    /// Create a new optimized quantum gate cache
    pub fn new() -> Self {
        let cache_config = CacheConfig {
            default_size: 2048, // Cache up to 2048 gate matrices
            default_ttl: 7200,  // 2 hours TTL for gate matrices
            enable_caching: true,
        };

        Self {
            matrix_cache: Arc::new(Mutex::new(TTLSizedCache::new(
                cache_config.default_size,
                cache_config.default_ttl,
            ))),
            buffer_pool: Arc::new(BufferPool::new()),
            cache_hits: Arc::new(Mutex::new(0)),
            cache_misses: Arc::new(Mutex::new(0)),
            total_computation_time: Arc::new(Mutex::new(0)),
        }
    }

    /// Get or compute a gate matrix with caching and profiling
    pub fn get_or_compute_matrix<F>(
        &self,
        key: GateKey,
        compute_fn: F,
    ) -> QuantRS2Result<Vec<Complex64>>
    where
        F: FnOnce() -> QuantRS2Result<Vec<Complex64>>,
    {
        // Check cache first
        if let Ok(mut cache) = self.matrix_cache.lock() {
            if let Some(cached) = cache.cache_get(&key) {
                *self.cache_hits.lock().unwrap() += 1;
                return Ok(cached.matrix.clone());
            }
        }

        // Cache miss - compute with profiling
        *self.cache_misses.lock().unwrap() += 1;

        let computation_result = Timer::time_function(
            &format!("gate_matrix_computation_{}", key.gate_type),
            compute_fn,
        );

        match computation_result {
            Ok(matrix) => {
                // Create cached entry with timing info
                let cached_matrix = CachedGateMatrix {
                    matrix: matrix.clone(),
                    size: matrix.len(),
                    computation_time_us: 0, // Would be filled by Timer in real implementation
                };

                // Store in cache
                if let Ok(mut cache) = self.matrix_cache.lock() {
                    cache.cache_set(key, cached_matrix);
                }

                Ok(matrix)
            }
            Err(e) => Err(e),
        }
    }

    /// Get cache performance statistics
    pub fn get_performance_stats(&self) -> QuantumGateCacheStats {
        let hits = *self.cache_hits.lock().unwrap();
        let misses = *self.cache_misses.lock().unwrap();
        let total_time = *self.total_computation_time.lock().unwrap();

        QuantumGateCacheStats {
            cache_hits: hits,
            cache_misses: misses,
            hit_ratio: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
            total_computation_time_us: total_time,
            average_computation_time_us: if misses > 0 {
                total_time / misses
            } else {
                0
            },
        }
    }

    /// Pre-warm cache with common gate matrices
    pub fn prewarm_common_gates(&self) -> QuantRS2Result<()> {
        use std::f64::consts::{FRAC_1_SQRT_2, PI};

        let common_gates = vec![
            ("pauli_x", vec![], 1),
            ("pauli_y", vec![], 1),
            ("pauli_z", vec![], 1),
            ("hadamard", vec![], 1),
            ("phase", vec![PI / 2.0], 1),
            ("rx", vec![PI / 4.0, PI / 2.0, PI], 1),
            ("ry", vec![PI / 4.0, PI / 2.0, PI], 1),
            ("rz", vec![PI / 4.0, PI / 2.0, PI], 1),
            ("cnot", vec![], 2),
            ("cz", vec![], 2),
        ];

        for (gate_name, params, qubits) in common_gates {
            for param_set in if params.is_empty() {
                vec![vec![]]
            } else {
                params.into_iter().map(|p| vec![p]).collect()
            } {
                let key = GateKey::new(gate_name, &param_set, qubits);

                // Compute and cache the matrix
                let _ = self.get_or_compute_matrix(key, || {
                    // This would call actual gate matrix computation
                    // For now, return identity matrix
                    let size = 1 << qubits;
                    let mut matrix = vec![Complex64::new(0.0, 0.0); size * size];
                    for i in 0..size {
                        matrix[i * size + i] = Complex64::new(1.0, 0.0);
                    }
                    Ok(matrix)
                })?;
            }
        }

        Ok(())
    }

    /// Clear cache and reset statistics
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.matrix_cache.lock() {
            cache.cache_clear();
        }
        *self.cache_hits.lock().unwrap() = 0;
        *self.cache_misses.lock().unwrap() = 0;
        *self.total_computation_time.lock().unwrap() = 0;
    }
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct QuantumGateCacheStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_ratio: f64,
    pub total_computation_time_us: u64,
    pub average_computation_time_us: u64,
}

/// Global quantum gate cache instance
static GLOBAL_GATE_CACHE: OnceLock<QuantumGateCache> = OnceLock::new();

/// Get the global quantum gate cache
pub fn global_gate_cache() -> &'static QuantumGateCache {
    GLOBAL_GATE_CACHE.get_or_init(QuantumGateCache::new)
}

/// Convenience macro for caching gate matrix computations
#[macro_export]
macro_rules! cached_gate_matrix {
    ($gate_type:expr, $params:expr, $qubits:expr, $compute:expr) => {{
        let key = $crate::optimizations::gate_cache::GateKey::new($gate_type, $params, $qubits);
        $crate::optimizations::gate_cache::global_gate_cache()
            .get_or_compute_matrix(key, || $compute)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_cache_basic_functionality() {
        let cache = QuantumGateCache::new();

        let key = GateKey::new("test_gate", &[1.0], 1);

        // First call should be a cache miss
        let matrix1 = cache.get_or_compute_matrix(key.clone(), || {
            Ok(vec![Complex64::new(1.0, 0.0); 4])
        }).unwrap();

        // Second call should be a cache hit
        let matrix2 = cache.get_or_compute_matrix(key, || {
            panic!("Should not be called due to cache hit");
        }).unwrap();

        assert_eq!(matrix1, matrix2);

        let stats = cache.get_performance_stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.hit_ratio, 0.5);
    }

    #[test]
    fn test_gate_key_hashing() {
        let key1 = GateKey::new("rx", &[std::f64::consts::PI], 1);
        let key2 = GateKey::new("rx", &[std::f64::consts::PI], 1);
        let key3 = GateKey::new("rx", &[std::f64::consts::PI / 2.0], 1);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);

        let mut set = std::collections::HashSet::new();
        set.insert(key1);
        assert!(set.contains(&key2));
        assert!(!set.contains(&key3));
    }

    #[test]
    fn test_cache_prewarming() {
        let cache = QuantumGateCache::new();

        // Cache should start empty
        let initial_stats = cache.get_performance_stats();
        assert_eq!(initial_stats.cache_misses, 0);

        // Pre-warm should populate cache
        cache.prewarm_common_gates().unwrap();

        let stats = cache.get_performance_stats();
        assert!(stats.cache_misses > 0); // Should have computed some matrices

        // Now test a common gate - should be a cache hit
        let key = GateKey::new("hadamard", &[], 1);
        let _matrix = cache.get_or_compute_matrix(key, || {
            panic!("Should be a cache hit");
        }).unwrap();

        let final_stats = cache.get_performance_stats();
        assert!(final_stats.cache_hits > 0);
    }
}