//! Expression caching and memoization for performance optimization
//!
//! This module provides caching mechanisms to avoid redundant symbolic computations.

use crate::Expression;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

/// Cache key type for expression operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CacheKey {
    /// Key for expanded expressions
    Expand(String),
    /// Key for simplified expressions
    Simplify(String),
    /// Key for differentiation (expression, variable)
    Diff(String, String),
    /// Key for substitution (expression, old, new)
    Substitute(String, String, String),
}

/// Thread-safe cache for symbolic expression operations
///
/// This cache helps avoid redundant computations by storing results of
/// expensive operations like expansion, simplification, and differentiation.
#[derive(Debug, Clone)]
pub struct ExpressionCache {
    cache: Arc<RwLock<HashMap<CacheKey, Expression>>>,
    max_size: usize,
}

impl ExpressionCache {
    /// Create a new expression cache with specified maximum size
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of entries to store (0 = unlimited)
    ///
    /// # Example
    /// ```
    /// use quantrs2_symengine::cache::ExpressionCache;
    ///
    /// let cache = ExpressionCache::new(1000);
    /// ```
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
        }
    }

    /// Create a new unlimited cache
    #[must_use]
    pub fn unlimited() -> Self {
        Self::new(0)
    }

    /// Get a cached expanded expression
    pub fn get_expanded(&self, expr: &Expression) -> Option<Expression> {
        let key = CacheKey::Expand(expr.to_string());
        self.cache.read().ok()?.get(&key).cloned()
    }

    /// Cache an expanded expression
    pub fn put_expanded(&self, original: &Expression, expanded: Expression) {
        let key = CacheKey::Expand(original.to_string());
        self.put(key, expanded);
    }

    /// Get a cached simplified expression
    pub fn get_simplified(&self, expr: &Expression) -> Option<Expression> {
        let key = CacheKey::Simplify(expr.to_string());
        self.cache.read().ok()?.get(&key).cloned()
    }

    /// Cache a simplified expression
    pub fn put_simplified(&self, original: &Expression, simplified: Expression) {
        let key = CacheKey::Simplify(original.to_string());
        self.put(key, simplified);
    }

    /// Get a cached derivative
    pub fn get_diff(&self, expr: &Expression, variable: &Expression) -> Option<Expression> {
        let key = CacheKey::Diff(expr.to_string(), variable.to_string());
        self.cache.read().ok()?.get(&key).cloned()
    }

    /// Cache a derivative
    pub fn put_diff(&self, expr: &Expression, variable: &Expression, derivative: Expression) {
        let key = CacheKey::Diff(expr.to_string(), variable.to_string());
        self.put(key, derivative);
    }

    /// Get a cached substitution result
    pub fn get_substitute(
        &self,
        expr: &Expression,
        old: &Expression,
        new: &Expression,
    ) -> Option<Expression> {
        let key = CacheKey::Substitute(expr.to_string(), old.to_string(), new.to_string());
        self.cache.read().ok()?.get(&key).cloned()
    }

    /// Cache a substitution result
    pub fn put_substitute(
        &self,
        expr: &Expression,
        old: &Expression,
        new: &Expression,
        result: Expression,
    ) {
        let key = CacheKey::Substitute(expr.to_string(), old.to_string(), new.to_string());
        self.put(key, result);
    }

    /// Internal method to put an entry in the cache
    fn put(&self, key: CacheKey, value: Expression) {
        if let Ok(mut cache) = self.cache.write() {
            // Check size limit
            if self.max_size > 0 && cache.len() >= self.max_size {
                // Simple FIFO eviction - remove first entry
                if let Some(first_key) = cache.keys().next().cloned() {
                    cache.remove(&first_key);
                }
            }

            cache.insert(key, value);
        }
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }

    /// Get the number of cached entries
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache.read().map_or(0, |cache| cache.len())
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.len(),
            max_size: self.max_size,
        }
    }
}

impl Default for ExpressionCache {
    fn default() -> Self {
        Self::new(10000) // Default to 10k entries
    }
}

/// Cache statistics
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Current number of entries
    pub size: usize,
    /// Maximum size (0 = unlimited)
    pub max_size: usize,
}

impl CacheStats {
    /// Get the fill percentage (0.0 to 1.0)
    #[must_use]
    pub fn fill_percentage(&self) -> f64 {
        if self.max_size == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let result = self.size as f64 / self.max_size as f64;
            result
        }
    }

    /// Check if cache is near capacity (>90% full)
    #[must_use]
    pub fn is_near_capacity(&self) -> bool {
        self.fill_percentage() > 0.9
    }
}

/// Extension trait for cached operations on Expression
pub trait CachedOps {
    /// Expand with caching
    fn expand_cached(&self, cache: &ExpressionCache) -> Expression;

    /// Simplify with caching
    fn simplify_cached(&self, cache: &ExpressionCache) -> Expression;

    /// Differentiate with caching
    fn diff_cached(&self, variable: &Expression, cache: &ExpressionCache) -> Expression;

    /// Substitute with caching
    fn substitute_cached(
        &self,
        old: &Expression,
        new: &Expression,
        cache: &ExpressionCache,
    ) -> Expression;
}

impl CachedOps for Expression {
    fn expand_cached(&self, cache: &ExpressionCache) -> Expression {
        cache.get_expanded(self).unwrap_or_else(|| {
            let expanded = self.expand();
            cache.put_expanded(self, expanded.clone());
            expanded
        })
    }

    fn simplify_cached(&self, cache: &ExpressionCache) -> Expression {
        cache.get_simplified(self).unwrap_or_else(|| {
            let simplified = self.simplify();
            cache.put_simplified(self, simplified.clone());
            simplified
        })
    }

    fn diff_cached(&self, variable: &Expression, cache: &ExpressionCache) -> Expression {
        cache.get_diff(self, variable).unwrap_or_else(|| {
            let derivative = self.diff(variable);
            cache.put_diff(self, variable, derivative.clone());
            derivative
        })
    }

    fn substitute_cached(
        &self,
        old: &Expression,
        new: &Expression,
        cache: &ExpressionCache,
    ) -> Expression {
        cache.get_substitute(self, old, new).unwrap_or_else(|| {
            let result = self.substitute(old, new);
            cache.put_substitute(self, old, new, result.clone());
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let cache = ExpressionCache::new(100);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        let stats = cache.stats();
        assert_eq!(stats.size, 0);
        assert_eq!(stats.max_size, 100);
    }

    #[test]
    fn test_cache_expand() {
        let cache = ExpressionCache::new(10);
        let x = Expression::symbol("x");
        let expr = (x + Expression::from(1)).pow(&Expression::from(2));

        // First call - not cached
        let result1 = expr.expand_cached(&cache);
        assert_eq!(cache.len(), 1);

        // Second call - should use cache
        let result2 = expr.expand_cached(&cache);
        assert_eq!(cache.len(), 1);

        assert_eq!(result1.to_string(), result2.to_string());
    }

    #[test]
    fn test_cache_differentiation() {
        let cache = ExpressionCache::new(10);
        let x = Expression::symbol("x");
        let expr = x.pow(&Expression::from(3));

        // First diff - not cached
        let result1 = expr.diff_cached(&x, &cache);
        assert_eq!(cache.len(), 1);

        // Second diff - should use cache
        let result2 = expr.diff_cached(&x, &cache);
        assert_eq!(cache.len(), 1);

        assert_eq!(result1.to_string(), result2.to_string());
    }

    #[test]
    fn test_cache_size_limit() {
        let cache = ExpressionCache::new(3);

        for i in 0..5 {
            let x = Expression::symbol(format!("x{i}"));
            let expr = x.pow(&Expression::from(2));
            let _ = expr.expand_cached(&cache);
        }

        // Should not exceed max size
        assert!(cache.len() <= 3);
    }

    #[test]
    fn test_cache_clear() {
        let cache = ExpressionCache::new(10);
        let x = Expression::symbol("x");
        let expr = x.pow(&Expression::from(2));

        let _ = expr.expand_cached(&cache);
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_stats() {
        let cache = ExpressionCache::new(100);
        let stats = cache.stats();

        assert_eq!(stats.size, 0);
        assert_eq!(stats.max_size, 100);
        assert_eq!(stats.fill_percentage(), 0.0);
        assert!(!stats.is_near_capacity());
    }

    #[test]
    fn test_unlimited_cache() {
        let cache = ExpressionCache::unlimited();
        let stats = cache.stats();
        assert_eq!(stats.max_size, 0);
    }
}
