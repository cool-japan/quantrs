//! Quantum Gate Matrix Caching using SciRS2 Beta.3 Features
//!
//! This module demonstrates how to leverage scirs2-core beta.3 caching capabilities
//! to optimize quantum gate matrix computations.

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::gate::functions::multi::{CNOT, CZ};
use crate::gate::functions::single::{
    Hadamard, PauliX, PauliY, PauliZ, Phase, RotationX, RotationY, RotationZ,
};
use crate::gate::GateOp;
use crate::qubit::QubitId;
use scirs2_core::cache::{CacheConfig, TTLSizedCache};
use scirs2_core::memory::{global_buffer_pool, BufferPool};
use scirs2_core::profiling::{Profiler, Timer};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

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

/// High-performance quantum gate cache using SciRS2 beta.3 features
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
            if let Some(cached) = cache.get(&key) {
                if let Ok(mut hits) = self.cache_hits.lock() {
                    *hits += 1;
                }
                return Ok(cached.matrix);
            }
        }

        // Cache miss - compute with profiling
        if let Ok(mut misses) = self.cache_misses.lock() {
            *misses += 1;
        }

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
                    cache.insert(key, cached_matrix);
                }

                Ok(matrix)
            }
            Err(e) => Err(e),
        }
    }

    /// Get cache performance statistics
    pub fn get_performance_stats(&self) -> QuantumGateCacheStats {
        let hits = self.cache_hits.lock().map(|g| *g).unwrap_or(0);
        let misses = self.cache_misses.lock().map(|g| *g).unwrap_or(0);
        let total_time = self.total_computation_time.lock().map(|g| *g).unwrap_or(0);

        QuantumGateCacheStats {
            cache_hits: hits,
            cache_misses: misses,
            hit_ratio: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
            total_computation_time_us: total_time,
            average_computation_time_us: total_time.checked_div(misses).unwrap_or(0),
        }
    }

    /// Pre-warm cache with common gate matrices
    pub fn prewarm_common_gates(&self) -> QuantRS2Result<()> {
        use std::f64::consts::PI;

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

                // Compute and cache the real gate matrix by constructing the
                // corresponding gate from `crate::gate::functions` and calling
                // its `matrix()` implementation.
                let _ =
                    self.get_or_compute_matrix(key, || compute_gate_matrix(gate_name, &param_set))?;
            }
        }

        Ok(())
    }

    /// Clear cache and reset statistics
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.matrix_cache.lock() {
            cache.clear();
        }
        if let Ok(mut hits) = self.cache_hits.lock() {
            *hits = 0;
        }
        if let Ok(mut misses) = self.cache_misses.lock() {
            *misses = 0;
        }
        if let Ok(mut time) = self.total_computation_time.lock() {
            *time = 0;
        }
    }
}

/// Compute the real matrix for a named gate using `crate::gate::functions`.
///
/// `gate_name` follows the convention used by [`QuantumGateCache::prewarm_common_gates`]
/// (e.g. `"pauli_x"`, `"hadamard"`, `"rx"`). Parameterized gates read their angle from
/// `params[0]`. Qubit identities (`QubitId(0)` / `QubitId(1)`) do not affect the returned
/// matrix, so fixed placeholders are used. Unknown gate names produce an honest error
/// rather than a fabricated identity matrix.
fn compute_gate_matrix(gate_name: &str, params: &[f64]) -> QuantRS2Result<Vec<Complex64>> {
    let q0 = QubitId(0);
    let q1 = QubitId(1);

    let angle = |gate: &str| -> QuantRS2Result<f64> {
        params.first().copied().ok_or_else(|| {
            QuantRS2Error::InvalidInput(format!("gate '{gate}' requires a rotation parameter"))
        })
    };

    match gate_name {
        "pauli_x" => PauliX { target: q0 }.matrix(),
        "pauli_y" => PauliY { target: q0 }.matrix(),
        "pauli_z" => PauliZ { target: q0 }.matrix(),
        "hadamard" => Hadamard { target: q0 }.matrix(),
        "phase" => Phase { target: q0 }.matrix(),
        "rx" => RotationX {
            target: q0,
            theta: angle("rx")?,
        }
        .matrix(),
        "ry" => RotationY {
            target: q0,
            theta: angle("ry")?,
        }
        .matrix(),
        "rz" => RotationZ {
            target: q0,
            theta: angle("rz")?,
        }
        .matrix(),
        "cnot" => CNOT {
            control: q0,
            target: q1,
        }
        .matrix(),
        "cz" => CZ {
            control: q0,
            target: q1,
        }
        .matrix(),
        other => Err(QuantRS2Error::UnsupportedOperation(format!(
            "unknown gate name '{other}' for matrix prewarming"
        ))),
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
        let matrix1 = cache
            .get_or_compute_matrix(key.clone(), || Ok(vec![Complex64::new(1.0, 0.0); 4]))
            .expect("matrix computation should succeed");

        // Second call should be a cache hit
        let matrix2 = cache
            .get_or_compute_matrix(key, || {
                panic!("Should not be called due to cache hit");
            })
            .expect("cache hit should succeed");

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
        cache
            .prewarm_common_gates()
            .expect("prewarming common gates should succeed");

        let stats = cache.get_performance_stats();
        assert!(stats.cache_misses > 0); // Should have computed some matrices

        // Now test a common gate - should be a cache hit
        let key = GateKey::new("hadamard", &[], 1);
        let _matrix = cache
            .get_or_compute_matrix(key, || {
                panic!("Should be a cache hit");
            })
            .expect("cache hit for hadamard gate should succeed");

        let final_stats = cache.get_performance_stats();
        assert!(final_stats.cache_hits > 0);
    }

    #[test]
    fn test_prewarmed_hadamard_is_real_not_identity() {
        use std::f64::consts::FRAC_1_SQRT_2;

        let cache = QuantumGateCache::new();
        cache
            .prewarm_common_gates()
            .expect("prewarming common gates should succeed");

        // Fetch the cached Hadamard matrix; this must be a cache hit so the
        // closure must NOT execute.
        let key = GateKey::new("hadamard", &[], 1);
        let matrix = cache
            .get_or_compute_matrix(key, || panic!("Hadamard should already be cached"))
            .expect("cache hit for hadamard gate should succeed");

        // Real Hadamard: 1/sqrt(2) * [[1, 1], [1, -1]] (NOT the identity that the
        // old fabricated implementation produced).
        assert_eq!(matrix.len(), 4);
        let tol = 1e-12;
        assert!((matrix[0].re - FRAC_1_SQRT_2).abs() < tol);
        assert!((matrix[1].re - FRAC_1_SQRT_2).abs() < tol);
        assert!((matrix[2].re - FRAC_1_SQRT_2).abs() < tol);
        assert!((matrix[3].re + FRAC_1_SQRT_2).abs() < tol);
        // Guard against the old identity-matrix fabrication: off-diagonals were 0.
        assert!(matrix[1].norm() > 0.5, "off-diagonal must be non-zero");
        assert!(matrix[2].norm() > 0.5, "off-diagonal must be non-zero");

        // Also verify a parameterized gate (RX(pi)) was cached with its real matrix.
        let rx_key = GateKey::new("rx", &[std::f64::consts::PI], 1);
        let rx = cache
            .get_or_compute_matrix(rx_key, || panic!("RX(pi) should already be cached"))
            .expect("cache hit for RX gate should succeed");
        // RX(pi) = [[0, -i], [-i, 0]]
        assert!(rx[0].norm() < tol);
        assert!((rx[1].im + 1.0).abs() < tol);
    }
}
