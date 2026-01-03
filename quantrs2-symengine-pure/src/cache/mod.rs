//! Expression caching and memoization.
//!
//! This module provides caching mechanisms for expensive operations
//! like evaluation and simplification.

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use rustc_hash::FxHasher;

use crate::expr::Expression;

/// Thread-safe cache for expression evaluation results
pub struct EvalCache {
    cache: DashMap<(u64, u64), f64, std::hash::BuildHasherDefault<FxHasher>>,
}

impl EvalCache {
    /// Create a new evaluation cache
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: DashMap::with_hasher(std::hash::BuildHasherDefault::<FxHasher>::default()),
        }
    }

    /// Get or compute an evaluation result
    pub fn get_or_compute<F>(&self, expr_hash: u64, params_hash: u64, compute: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        let key = (expr_hash, params_hash);
        if let Some(value) = self.cache.get(&key) {
            return *value;
        }

        let result = compute();
        self.cache.insert(key, result);
        result
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
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

/// Compute a hash for an expression
fn compute_hash(expr: &Expression) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = FxHasher::default();
    expr.to_string().hash(&mut hasher);
    hasher.finish()
}

/// Compute a hash for a set of parameters
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

#[cfg(test)]
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
}
