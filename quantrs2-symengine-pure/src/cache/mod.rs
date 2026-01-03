//! Expression caching and memoization.
//!
//! This module provides caching mechanisms for expensive operations
//! like evaluation, simplification, and complex number operations.
//!
//! ## Features
//!
//! - **`EvalCache`**: Thread-safe cache for real-valued evaluation results with LRU eviction
//! - **`ComplexEvalCache`**: Thread-safe cache for complex-valued evaluations (quantum amplitudes)
//! - **`SimplificationCache`**: Cache for expression simplification results
//! - **`BatchEvalCache`**: Optimized for VQE optimization loops with parameter sweeps
//! - **`CachedEvaluator`**: Convenient wrapper with all caching features integrated
//! - **`ExpressionCache`**: Hash consing for structural sharing of expressions
//!
//! ## Performance Benefits
//!
//! Expression caching is critical for quantum computing applications:
//!
//! - **VQE/QAOA loops**: Same expressions evaluated thousands of times with different parameters
//! - **Gradient computation**: Derivatives computed repeatedly during optimization
//! - **Circuit simulation**: Gate matrices cached after first computation
//!
//! ## Example
//!
//! ```ignore
//! use quantrs2_symengine_pure::cache::{CachedEvaluator, hash_params};
//! use quantrs2_symengine_pure::Expression;
//! use std::collections::HashMap;
//!
//! let evaluator = CachedEvaluator::new();
//! let expr = Expression::symbol("x").sin();
//!
//! // First evaluation computes the result
//! let mut params = HashMap::new();
//! params.insert("x".to_string(), 0.5);
//! let result1 = evaluator.eval(&expr, &params).unwrap();
//!
//! // Second evaluation retrieves from cache
//! let result2 = evaluator.eval(&expr, &params).unwrap();
//! assert!((result1 - result2).abs() < 1e-10);
//!
//! // Check hit rate
//! let stats = evaluator.stats();
//! println!("Cache hit rate: {:.1}%", stats.overall_hit_rate() * 100.0);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use rustc_hash::FxHasher;
use scirs2_core::Complex64;

use crate::error::SymEngineResult;
use crate::expr::Expression;

/// Default maximum cache size (entries)
pub const DEFAULT_MAX_CACHE_SIZE: usize = 10_000;

/// Thread-safe cache for expression evaluation results with LRU eviction.
///
/// Uses DashMap for concurrent access and FxHasher for fast hashing.
pub struct EvalCache {
    cache: DashMap<(u64, u64), CachedValue<f64>, std::hash::BuildHasherDefault<FxHasher>>,
    max_size: usize,
    access_counter: AtomicU64,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

/// A cached value with access tracking for LRU eviction
#[derive(Clone)]
struct CachedValue<T> {
    value: T,
    last_access: u64,
}

impl EvalCache {
    /// Create a new evaluation cache with default size
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_CACHE_SIZE)
    }

    /// Create a new evaluation cache with specified maximum size
    #[must_use]
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            cache: DashMap::with_hasher(std::hash::BuildHasherDefault::<FxHasher>::default()),
            max_size,
            access_counter: AtomicU64::new(0),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Get or compute an evaluation result
    pub fn get_or_compute<F>(&self, expr_hash: u64, params_hash: u64, compute: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        let key = (expr_hash, params_hash);
        let access_time = self.access_counter.fetch_add(1, Ordering::Relaxed);

        if let Some(mut entry) = self.cache.get_mut(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            entry.last_access = access_time;
            return entry.value;
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let result = compute();

        // Check if we need to evict
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        self.cache.insert(
            key,
            CachedValue {
                value: result,
                last_access: access_time,
            },
        );
        result
    }

    /// Try to get a cached value without computing
    #[must_use]
    pub fn get(&self, expr_hash: u64, params_hash: u64) -> Option<f64> {
        let key = (expr_hash, params_hash);
        let access_time = self.access_counter.fetch_add(1, Ordering::Relaxed);

        self.cache.get_mut(&key).map(|mut entry| {
            entry.last_access = access_time;
            entry.value
        })
    }

    /// Get or compute with Result return type
    pub fn get_or_try_compute<F, E>(
        &self,
        expr_hash: u64,
        params_hash: u64,
        compute: F,
    ) -> Result<f64, E>
    where
        F: FnOnce() -> Result<f64, E>,
    {
        let key = (expr_hash, params_hash);
        let access_time = self.access_counter.fetch_add(1, Ordering::Relaxed);

        if let Some(mut entry) = self.cache.get_mut(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            entry.last_access = access_time;
            return Ok(entry.value);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let result = compute()?;

        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        self.cache.insert(
            key,
            CachedValue {
                value: result,
                last_access: access_time,
            },
        );
        Ok(result)
    }

    /// Insert a value into the cache
    pub fn insert(&self, expr_hash: u64, params_hash: u64, value: f64) {
        let key = (expr_hash, params_hash);
        let access_time = self.access_counter.fetch_add(1, Ordering::Relaxed);

        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        self.cache.insert(
            key,
            CachedValue {
                value,
                last_access: access_time,
            },
        );
    }

    /// Evict the least recently used entries (removes ~10% of cache)
    fn evict_lru(&self) {
        let evict_count = self.max_size / 10;
        if evict_count == 0 {
            return;
        }

        // Collect entries sorted by access time
        let mut entries: Vec<_> = self
            .cache
            .iter()
            .map(|e| (*e.key(), e.value().last_access))
            .collect();
        entries.sort_by_key(|(_, access)| *access);

        // Remove oldest entries
        for (key, _) in entries.into_iter().take(evict_count) {
            self.cache.remove(&key);
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Get the number of cached entries
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        CacheStats {
            size: self.cache.len(),
            max_size: self.max_size,
            hits,
            misses,
            hit_rate: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of entries
    pub size: usize,
    /// Maximum allowed entries
    pub max_size: usize,
    /// Number of cache hits
    pub hits: usize,
    /// Number of cache misses
    pub misses: usize,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
}

impl Default for EvalCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash consing for structural sharing of expressions
pub struct ExpressionCache {
    cache: DashMap<u64, Arc<Expression>, std::hash::BuildHasherDefault<FxHasher>>,
}

impl ExpressionCache {
    /// Create a new expression cache
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: DashMap::with_hasher(std::hash::BuildHasherDefault::<FxHasher>::default()),
        }
    }

    /// Get or insert an expression, returning a shared reference
    pub fn get_or_insert(&self, expr: Expression) -> Arc<Expression> {
        let hash = compute_hash(&expr);
        self.cache
            .entry(hash)
            .or_insert_with(|| Arc::new(expr))
            .clone()
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
    }
}

impl Default for ExpressionCache {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Complex Evaluation Cache
// =========================================================================

/// Thread-safe cache for complex number evaluation results.
///
/// Used for caching quantum amplitude calculations and complex-valued
/// expression evaluations.
pub struct ComplexEvalCache {
    cache: DashMap<(u64, u64), CachedValue<Complex64>, std::hash::BuildHasherDefault<FxHasher>>,
    max_size: usize,
    access_counter: AtomicU64,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl ComplexEvalCache {
    /// Create a new complex evaluation cache with default size
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_CACHE_SIZE)
    }

    /// Create a new complex evaluation cache with specified maximum size
    #[must_use]
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            cache: DashMap::with_hasher(std::hash::BuildHasherDefault::<FxHasher>::default()),
            max_size,
            access_counter: AtomicU64::new(0),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Get or compute a complex evaluation result
    pub fn get_or_compute<F>(&self, expr_hash: u64, params_hash: u64, compute: F) -> Complex64
    where
        F: FnOnce() -> Complex64,
    {
        let key = (expr_hash, params_hash);
        let access_time = self.access_counter.fetch_add(1, Ordering::Relaxed);

        if let Some(mut entry) = self.cache.get_mut(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            entry.last_access = access_time;
            return entry.value;
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let result = compute();

        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        self.cache.insert(
            key,
            CachedValue {
                value: result,
                last_access: access_time,
            },
        );
        result
    }

    /// Get or compute with Result return type
    pub fn get_or_try_compute<F, E>(
        &self,
        expr_hash: u64,
        params_hash: u64,
        compute: F,
    ) -> Result<Complex64, E>
    where
        F: FnOnce() -> Result<Complex64, E>,
    {
        let key = (expr_hash, params_hash);
        let access_time = self.access_counter.fetch_add(1, Ordering::Relaxed);

        if let Some(mut entry) = self.cache.get_mut(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            entry.last_access = access_time;
            return Ok(entry.value);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let result = compute()?;

        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        self.cache.insert(
            key,
            CachedValue {
                value: result,
                last_access: access_time,
            },
        );
        Ok(result)
    }

    /// Evict the least recently used entries
    fn evict_lru(&self) {
        let evict_count = self.max_size / 10;
        if evict_count == 0 {
            return;
        }

        let mut entries: Vec<_> = self
            .cache
            .iter()
            .map(|e| (*e.key(), e.value().last_access))
            .collect();
        entries.sort_by_key(|(_, access)| *access);

        for (key, _) in entries.into_iter().take(evict_count) {
            self.cache.remove(&key);
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Get the number of cached entries
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        CacheStats {
            size: self.cache.len(),
            max_size: self.max_size,
            hits,
            misses,
            hit_rate: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for ComplexEvalCache {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Simplification Cache
// =========================================================================

/// Thread-safe cache for expression simplification results.
///
/// Caches the result of expensive simplification operations to avoid
/// re-running e-graph saturation for the same expressions.
pub struct SimplificationCache {
    cache: DashMap<u64, CachedValue<Expression>, std::hash::BuildHasherDefault<FxHasher>>,
    max_size: usize,
    access_counter: AtomicU64,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl SimplificationCache {
    /// Create a new simplification cache with default size
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_CACHE_SIZE)
    }

    /// Create a new simplification cache with specified maximum size
    #[must_use]
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            cache: DashMap::with_hasher(std::hash::BuildHasherDefault::<FxHasher>::default()),
            max_size,
            access_counter: AtomicU64::new(0),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Get or compute a simplified expression
    pub fn get_or_simplify<F>(&self, expr: &Expression, simplify: F) -> Expression
    where
        F: FnOnce() -> Expression,
    {
        let expr_hash = compute_hash(expr);
        let access_time = self.access_counter.fetch_add(1, Ordering::Relaxed);

        if let Some(mut entry) = self.cache.get_mut(&expr_hash) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            entry.last_access = access_time;
            return entry.value.clone();
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let result = simplify();

        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        self.cache.insert(
            expr_hash,
            CachedValue {
                value: result.clone(),
                last_access: access_time,
            },
        );
        result
    }

    /// Evict the least recently used entries
    fn evict_lru(&self) {
        let evict_count = self.max_size / 10;
        if evict_count == 0 {
            return;
        }

        let mut entries: Vec<_> = self
            .cache
            .iter()
            .map(|e| (*e.key(), e.value().last_access))
            .collect();
        entries.sort_by_key(|(_, access)| *access);

        for (key, _) in entries.into_iter().take(evict_count) {
            self.cache.remove(&key);
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Get the number of cached entries
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        CacheStats {
            size: self.cache.len(),
            max_size: self.max_size,
            hits,
            misses,
            hit_rate: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for SimplificationCache {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Batch Evaluation Cache
// =========================================================================

/// Cache for batch evaluation results in VQE optimization loops.
///
/// Optimized for scenarios where the same expression is evaluated many times
/// with slightly different parameter sets (e.g., parameter sweeps).
pub struct BatchEvalCache {
    /// Expression hash -> (params_hash -> result)
    cache: DashMap<
        u64,
        DashMap<u64, f64, std::hash::BuildHasherDefault<FxHasher>>,
        std::hash::BuildHasherDefault<FxHasher>,
    >,
    max_expressions: usize,
    max_params_per_expr: usize,
}

impl BatchEvalCache {
    /// Create a new batch evaluation cache
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(1000, 1000)
    }

    /// Create a new batch evaluation cache with specified capacities
    #[must_use]
    pub fn with_capacity(max_expressions: usize, max_params_per_expr: usize) -> Self {
        Self {
            cache: DashMap::with_hasher(std::hash::BuildHasherDefault::<FxHasher>::default()),
            max_expressions,
            max_params_per_expr,
        }
    }

    /// Get or compute a batch of evaluation results
    pub fn get_or_compute_batch<F>(
        &self,
        expr_hash: u64,
        param_hashes: &[u64],
        compute: F,
    ) -> Vec<f64>
    where
        F: FnOnce(&[usize]) -> Vec<f64>,
    {
        // Find which parameter sets we need to compute
        let expr_cache = self.cache.entry(expr_hash).or_insert_with(|| {
            DashMap::with_hasher(std::hash::BuildHasherDefault::<FxHasher>::default())
        });

        let mut results = vec![0.0; param_hashes.len()];
        let mut missing_indices = Vec::new();

        for (i, &ph) in param_hashes.iter().enumerate() {
            if let Some(val) = expr_cache.get(&ph) {
                results[i] = *val;
            } else {
                missing_indices.push(i);
            }
        }

        // Compute missing values
        if !missing_indices.is_empty() {
            let computed = compute(&missing_indices);

            for (j, &i) in missing_indices.iter().enumerate() {
                results[i] = computed[j];
                let ph = param_hashes[i];

                // Check if we need to evict from per-expression cache
                if expr_cache.len() >= self.max_params_per_expr {
                    // Simple random eviction (for speed)
                    // Extract key first to avoid holding the iterator lock
                    let first_key = expr_cache.iter().next().map(|e| *e.key());
                    if let Some(key) = first_key {
                        expr_cache.remove(&key);
                    }
                }

                expr_cache.insert(ph, computed[j]);
            }
        }

        results
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
    }

    /// Get the number of cached expressions
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get total number of cached parameter sets across all expressions
    #[must_use]
    pub fn total_params_cached(&self) -> usize {
        self.cache.iter().map(|e| e.value().len()).sum()
    }
}

impl Default for BatchEvalCache {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// Cached Expression Evaluator
// =========================================================================

/// An expression evaluator with integrated caching.
///
/// This provides a convenient interface for evaluating expressions with
/// automatic caching of results.
#[allow(clippy::struct_field_names)]
pub struct CachedEvaluator {
    eval_cache: EvalCache,
    complex_cache: ComplexEvalCache,
    simplification_cache: SimplificationCache,
}

impl CachedEvaluator {
    /// Create a new cached evaluator
    #[must_use]
    pub fn new() -> Self {
        Self {
            eval_cache: EvalCache::new(),
            complex_cache: ComplexEvalCache::new(),
            simplification_cache: SimplificationCache::new(),
        }
    }

    /// Create a new cached evaluator with specified cache sizes
    #[must_use]
    pub fn with_capacity(eval_size: usize, complex_size: usize, simplify_size: usize) -> Self {
        Self {
            eval_cache: EvalCache::with_capacity(eval_size),
            complex_cache: ComplexEvalCache::with_capacity(complex_size),
            simplification_cache: SimplificationCache::with_capacity(simplify_size),
        }
    }

    /// Evaluate an expression with caching
    pub fn eval(&self, expr: &Expression, values: &HashMap<String, f64>) -> SymEngineResult<f64> {
        let expr_hash = compute_hash(expr);
        let params_hash = hash_params(values);

        // Use get_or_try_compute to properly track hits/misses
        self.eval_cache
            .get_or_try_compute(expr_hash, params_hash, || expr.eval(values))
    }

    /// Evaluate an expression as complex with caching
    pub fn eval_complex(
        &self,
        expr: &Expression,
        values: &HashMap<String, f64>,
    ) -> SymEngineResult<Complex64> {
        let expr_hash = compute_hash(expr);
        let params_hash = hash_params(values);

        self.complex_cache
            .get_or_try_compute(expr_hash, params_hash, || expr.eval_complex(values))
    }

    /// Simplify an expression with caching
    pub fn simplify(&self, expr: &Expression) -> Expression {
        self.simplification_cache
            .get_or_simplify(expr, || expr.simplify())
    }

    /// Clear all caches
    pub fn clear(&self) {
        self.eval_cache.clear();
        self.complex_cache.clear();
        self.simplification_cache.clear();
    }

    /// Get combined cache statistics
    #[must_use]
    pub fn stats(&self) -> CombinedCacheStats {
        CombinedCacheStats {
            eval: self.eval_cache.stats(),
            complex: self.complex_cache.stats(),
            simplification: self.simplification_cache.stats(),
        }
    }
}

impl Default for CachedEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined statistics for all cache types
#[derive(Debug, Clone)]
pub struct CombinedCacheStats {
    /// Real evaluation cache stats
    pub eval: CacheStats,
    /// Complex evaluation cache stats
    pub complex: CacheStats,
    /// Simplification cache stats
    pub simplification: CacheStats,
}

impl CombinedCacheStats {
    /// Get the total number of cached entries
    #[must_use]
    pub const fn total_size(&self) -> usize {
        self.eval.size + self.complex.size + self.simplification.size
    }

    /// Get the total number of cache hits
    #[must_use]
    pub const fn total_hits(&self) -> usize {
        self.eval.hits + self.complex.hits + self.simplification.hits
    }

    /// Get the total number of cache misses
    #[must_use]
    pub const fn total_misses(&self) -> usize {
        self.eval.misses + self.complex.misses + self.simplification.misses
    }

    /// Get the overall hit rate
    #[must_use]
    pub fn overall_hit_rate(&self) -> f64 {
        let total = self.total_hits() + self.total_misses();
        if total > 0 {
            self.total_hits() as f64 / total as f64
        } else {
            0.0
        }
    }
}

// =========================================================================
// Hash Functions
// =========================================================================

/// Compute a hash for an expression
pub fn compute_hash(expr: &Expression) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = FxHasher::default();
    expr.to_string().hash(&mut hasher);
    hasher.finish()
}

/// Compute a hash for a set of real parameters
pub fn hash_params(params: &HashMap<String, f64>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = FxHasher::default();

    // Sort keys for consistent hashing
    let mut keys: Vec<_> = params.keys().collect();
    keys.sort();

    for key in keys {
        key.hash(&mut hasher);
        if let Some(value) = params.get(key) {
            value.to_bits().hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Compute a hash for complex parameters
pub fn hash_complex_params(params: &HashMap<String, Complex64>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = FxHasher::default();

    let mut keys: Vec<_> = params.keys().collect();
    keys.sort();

    for key in keys {
        key.hash(&mut hasher);
        if let Some(value) = params.get(key) {
            value.re.to_bits().hash(&mut hasher);
            value.im.to_bits().hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Compute a hash for a parameter array (for batch operations)
pub fn hash_param_array(params: &[f64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = FxHasher::default();

    for value in params {
        value.to_bits().hash(&mut hasher);
    }

    hasher.finish()
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_cache() {
        let cache = EvalCache::new();

        let result1 = cache.get_or_compute(1, 1, || 42.0);
        assert!((result1 - 42.0).abs() < 1e-10);

        // Should return cached value
        let result2 = cache.get_or_compute(1, 1, || 100.0);
        assert!((result2 - 42.0).abs() < 1e-10);

        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_eval_cache_stats() {
        let cache = EvalCache::new();

        // Miss then hit
        cache.get_or_compute(1, 1, || 42.0);
        cache.get_or_compute(1, 1, || 42.0);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_eval_cache_lru_eviction() {
        let cache = EvalCache::with_capacity(10);

        // Fill cache beyond capacity
        for i in 0..15u64 {
            cache.get_or_compute(i, 0, || i as f64);
        }

        // Should have evicted some entries
        assert!(cache.len() <= 10);
    }

    #[test]
    fn test_complex_eval_cache() {
        let cache = ComplexEvalCache::new();

        let result1 = cache.get_or_compute(1, 1, || Complex64::new(3.0, 4.0));
        assert!((result1.re - 3.0).abs() < 1e-10);
        assert!((result1.im - 4.0).abs() < 1e-10);

        // Should return cached value
        let result2 = cache.get_or_compute(1, 1, || Complex64::new(100.0, 200.0));
        assert!((result2.re - 3.0).abs() < 1e-10);
        assert!((result2.im - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_eval_cache_try_compute() {
        let cache = ComplexEvalCache::new();

        let result: Result<_, &str> =
            cache.get_or_try_compute(1, 1, || Ok(Complex64::new(1.0, 2.0)));
        assert!(result.is_ok());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);

        // Second call should hit cache
        let result2: Result<_, &str> =
            cache.get_or_try_compute(1, 1, || Err("should not be called"));
        assert!(result2.is_ok());

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_simplification_cache() {
        let cache = SimplificationCache::new();

        let expr = Expression::symbol("x") + Expression::symbol("x");
        let simplified = cache.get_or_simplify(&expr, || {
            // This simulates simplification
            Expression::int(2) * Expression::symbol("x")
        });

        // Should have cached
        assert_eq!(cache.len(), 1);

        // Second call should return cached
        let simplified2 = cache.get_or_simplify(&expr, || {
            // This should not be called
            Expression::symbol("should_not_appear")
        });

        // Both should be equivalent
        assert_eq!(simplified.to_string(), simplified2.to_string());

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_batch_eval_cache() {
        let cache = BatchEvalCache::new();

        let expr_hash = 12345u64;
        let param_hashes = vec![1, 2, 3, 4, 5];

        let mut compute_count = 0;
        let results = cache.get_or_compute_batch(expr_hash, &param_hashes, |missing| {
            compute_count = missing.len();
            missing.iter().map(|&i| i as f64 * 10.0).collect()
        });

        assert_eq!(compute_count, 5); // All were missing
        assert!((results[0] - 0.0).abs() < 1e-10);
        assert!((results[1] - 10.0).abs() < 1e-10);

        // Second call - all should be cached
        let mut compute_count2 = 0;
        let results2 = cache.get_or_compute_batch(expr_hash, &param_hashes, |missing| {
            compute_count2 = missing.len();
            missing.iter().map(|&i| i as f64 * 100.0).collect()
        });

        assert_eq!(compute_count2, 0); // All were cached
        assert!((results2[0] - 0.0).abs() < 1e-10);
        assert!((results2[1] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_eval_cache_partial_hit() {
        let cache = BatchEvalCache::new();

        let expr_hash = 12345u64;

        // First call with params 1, 2, 3
        cache.get_or_compute_batch(expr_hash, &[1, 2, 3], |missing| {
            missing.iter().map(|&i| i as f64).collect()
        });

        // Second call with params 2, 3, 4, 5 - 2 and 3 should be cached
        let mut computed_indices = Vec::new();
        cache.get_or_compute_batch(expr_hash, &[2, 3, 4, 5], |missing| {
            computed_indices = missing.to_vec();
            missing.iter().map(|&i| i as f64).collect()
        });

        // Only indices 2 and 3 (params 4 and 5) should be computed
        assert_eq!(computed_indices, vec![2, 3]);
    }

    #[test]
    fn test_cached_evaluator() {
        let evaluator = CachedEvaluator::new();

        let expr = Expression::symbol("x");
        let mut values = HashMap::new();
        values.insert("x".to_string(), 5.0);

        let result1 = evaluator.eval(&expr, &values).expect("should eval");
        assert!((result1 - 5.0).abs() < 1e-10);

        // Second call should use cache
        let result2 = evaluator.eval(&expr, &values).expect("should eval");
        assert!((result2 - 5.0).abs() < 1e-10);

        let stats = evaluator.stats();
        assert_eq!(stats.eval.misses, 1);
        assert_eq!(stats.eval.hits, 1);
    }

    #[test]
    fn test_cached_evaluator_complex() {
        let evaluator = CachedEvaluator::new();

        // Expression: 1 + I (imaginary unit)
        let expr = Expression::int(1) + Expression::symbol("I");
        let values = HashMap::new();

        let result = evaluator.eval_complex(&expr, &values).expect("should eval");
        assert!((result.re - 1.0).abs() < 1e-10);
        assert!((result.im - 1.0).abs() < 1e-10);

        let stats = evaluator.stats();
        assert_eq!(stats.complex.misses, 1);
    }

    #[test]
    fn test_cached_evaluator_simplify() {
        let evaluator = CachedEvaluator::new();

        let expr = Expression::symbol("x") + Expression::int(0);
        let simplified = evaluator.simplify(&expr);

        // x + 0 should simplify to just x
        assert!(simplified.is_symbol() || simplified.to_string().contains('x'));

        // Second call should use cache
        let simplified2 = evaluator.simplify(&expr);
        assert_eq!(simplified.to_string(), simplified2.to_string());

        let stats = evaluator.stats();
        assert_eq!(stats.simplification.misses, 1);
        assert_eq!(stats.simplification.hits, 1);
    }

    #[test]
    fn test_combined_cache_stats() {
        let evaluator = CachedEvaluator::new();

        // Generate some hits and misses
        let expr = Expression::symbol("x");
        let mut values = HashMap::new();
        values.insert("x".to_string(), 1.0);

        // Miss, hit, hit
        for _ in 0..3 {
            let _ = evaluator.eval(&expr, &values);
        }

        let stats = evaluator.stats();
        assert_eq!(stats.total_size(), 1);
        assert_eq!(stats.total_hits(), 2);
        assert_eq!(stats.total_misses(), 1);
        assert!((stats.overall_hit_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hash_params() {
        let mut params1 = HashMap::new();
        params1.insert("x".to_string(), 1.0);
        params1.insert("y".to_string(), 2.0);

        let mut params2 = HashMap::new();
        params2.insert("y".to_string(), 2.0);
        params2.insert("x".to_string(), 1.0);

        // Order shouldn't matter
        assert_eq!(hash_params(&params1), hash_params(&params2));
    }

    #[test]
    fn test_hash_complex_params() {
        let mut params1 = HashMap::new();
        params1.insert("a".to_string(), Complex64::new(1.0, 2.0));
        params1.insert("b".to_string(), Complex64::new(3.0, 4.0));

        let mut params2 = HashMap::new();
        params2.insert("b".to_string(), Complex64::new(3.0, 4.0));
        params2.insert("a".to_string(), Complex64::new(1.0, 2.0));

        // Order shouldn't matter
        assert_eq!(hash_complex_params(&params1), hash_complex_params(&params2));
    }

    #[test]
    fn test_hash_param_array() {
        let params1 = [1.0, 2.0, 3.0];
        let params2 = [1.0, 2.0, 3.0];
        let params3 = [1.0, 2.0, 4.0];

        assert_eq!(hash_param_array(&params1), hash_param_array(&params2));
        assert_ne!(hash_param_array(&params1), hash_param_array(&params3));
    }

    #[test]
    fn test_expression_cache() {
        let cache = ExpressionCache::new();

        let expr1 = Expression::symbol("x");
        let arc1 = cache.get_or_insert(expr1.clone());
        let arc2 = cache.get_or_insert(expr1);

        // Should be the same Arc
        assert!(Arc::ptr_eq(&arc1, &arc2));
    }

    #[test]
    fn test_cache_clear() {
        let cache = EvalCache::new();
        cache.get_or_compute(1, 1, || 42.0);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());

        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }
}
