//! Memory Management Module for Scientific Performance Optimization
//!
//! This module provides hierarchical memory management with intelligent caching,
//! memory pools, and comprehensive statistics tracking for large-scale scientific
//! computing applications.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::config::{
    MemoryOptimizationConfig, MemoryPoolConfig, CompressionConfig,
    PoolGrowthStrategy, CompressionAlgorithm, GarbageCollectionStrategy,
};

/// Hierarchical memory manager
pub struct HierarchicalMemoryManager {
    /// Configuration
    pub config: MemoryOptimizationConfig,
    /// Memory pools
    pub memory_pools: HashMap<usize, MemoryPool>,
    /// Cache hierarchy
    pub cache_hierarchy: CacheHierarchy,
    /// Memory statistics
    pub memory_stats: MemoryStatistics,
}

impl HierarchicalMemoryManager {
    pub fn new(config: MemoryOptimizationConfig) -> Self {
        let mut manager = Self {
            memory_pools: HashMap::new(),
            cache_hierarchy: CacheHierarchy::new(),
            memory_stats: MemoryStatistics::default(),
            config,
        };

        // Initialize memory pools based on configuration
        manager.initialize_memory_pools();
        manager
    }

    fn initialize_memory_pools(&mut self) {
        for &block_size in &self.config.memory_pool_config.block_sizes {
            let pool = MemoryPool::new(
                format!("pool_{}", block_size),
                block_size,
                self.config.memory_pool_config.pool_size / self.config.memory_pool_config.block_sizes.len(),
            );
            self.memory_pools.insert(block_size, pool);
        }
    }

    pub fn allocate(&mut self, size: usize) -> Option<*mut u8> {
        // Find the appropriate pool
        let pool_size = self.find_suitable_pool_size(size);

        if let Some(pool) = self.memory_pools.get_mut(&pool_size) {
            let ptr = pool.allocate();
            if ptr.is_some() {
                self.memory_stats.allocation_count += 1;
                self.memory_stats.current_usage += size;
                self.memory_stats.total_allocated += size;

                if self.memory_stats.current_usage > self.memory_stats.peak_usage {
                    self.memory_stats.peak_usage = self.memory_stats.current_usage;
                }
            }
            ptr
        } else {
            None
        }
    }

    pub fn deallocate(&mut self, ptr: *mut u8, size: usize) {
        let pool_size = self.find_suitable_pool_size(size);

        if let Some(pool) = self.memory_pools.get_mut(&pool_size) {
            pool.deallocate(ptr);
            self.memory_stats.deallocation_count += 1;
            self.memory_stats.current_usage = self.memory_stats.current_usage.saturating_sub(size);
        }
    }

    fn find_suitable_pool_size(&self, size: usize) -> usize {
        self.config.memory_pool_config.block_sizes
            .iter()
            .find(|&&block_size| block_size >= size)
            .cloned()
            .unwrap_or_else(|| {
                *self.config.memory_pool_config.block_sizes.last().unwrap_or(&4096)
            })
    }

    pub fn get_memory_statistics(&self) -> &MemoryStatistics {
        &self.memory_stats
    }

    pub fn trigger_garbage_collection(&mut self) {
        match self.config.gc_strategy {
            GarbageCollectionStrategy::Manual => {
                // Manual GC - do nothing automatically
            },
            GarbageCollectionStrategy::Automatic => {
                self.perform_automatic_gc();
            },
            GarbageCollectionStrategy::Adaptive => {
                if self.should_trigger_adaptive_gc() {
                    self.perform_adaptive_gc();
                }
            },
            GarbageCollectionStrategy::Generational => {
                self.perform_generational_gc();
            },
        }
    }

    fn perform_automatic_gc(&mut self) {
        // Clear unused cache entries
        self.cache_hierarchy.cleanup_unused_entries();
    }

    fn should_trigger_adaptive_gc(&self) -> bool {
        let memory_pressure = self.memory_stats.current_usage as f64 / self.config.cache_size_limit as f64;
        memory_pressure > 0.8 // Trigger when 80% full
    }

    fn perform_adaptive_gc(&mut self) {
        // Adaptive GC based on usage patterns
        self.cache_hierarchy.adaptive_cleanup();
    }

    fn perform_generational_gc(&mut self) {
        // Generational GC - clean older cache entries first
        self.cache_hierarchy.generational_cleanup();
    }
}

/// Memory pool implementation
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool identifier
    pub id: String,
    /// Block size
    pub block_size: usize,
    /// Total capacity
    pub total_capacity: usize,
    /// Used capacity
    pub used_capacity: usize,
    /// Free blocks
    pub free_blocks: VecDeque<*mut u8>,
    /// Allocation statistics
    pub allocation_stats: AllocationStatistics,
}

impl MemoryPool {
    pub fn new(id: String, block_size: usize, capacity: usize) -> Self {
        let mut pool = Self {
            id,
            block_size,
            total_capacity: capacity,
            used_capacity: 0,
            free_blocks: VecDeque::new(),
            allocation_stats: AllocationStatistics::default(),
        };

        // Pre-allocate blocks if configured
        pool.preallocate_blocks();
        pool
    }

    fn preallocate_blocks(&mut self) {
        let num_blocks = self.total_capacity / self.block_size;
        for _ in 0..num_blocks {
            // In a real implementation, this would allocate actual memory
            let fake_ptr = Box::into_raw(vec![0u8; self.block_size].into_boxed_slice()) as *mut u8;
            self.free_blocks.push_back(fake_ptr);
        }
    }

    pub fn allocate(&mut self) -> Option<*mut u8> {
        if let Some(ptr) = self.free_blocks.pop_front() {
            self.used_capacity += self.block_size;
            self.allocation_stats.total_allocations += 1;
            self.update_utilization();
            Some(ptr)
        } else {
            self.allocation_stats.failed_allocations += 1;
            None
        }
    }

    pub fn deallocate(&mut self, ptr: *mut u8) {
        self.free_blocks.push_back(ptr);
        self.used_capacity = self.used_capacity.saturating_sub(self.block_size);
        self.update_utilization();
    }

    fn update_utilization(&mut self) {
        self.allocation_stats.utilization =
            self.used_capacity as f64 / self.total_capacity as f64;

        if self.allocation_stats.total_allocations > 0 {
            self.allocation_stats.avg_allocation_size =
                self.used_capacity as f64 / self.allocation_stats.total_allocations as f64;
        }
    }

    pub fn get_statistics(&self) -> &AllocationStatistics {
        &self.allocation_stats
    }
}

/// Cache hierarchy for multi-level caching
#[derive(Debug)]
pub struct CacheHierarchy {
    /// L1 cache (fastest, smallest)
    pub l1_cache: LRUCache<String, Vec<u8>>,
    /// L2 cache (medium speed/size)
    pub l2_cache: LRUCache<String, Vec<u8>>,
    /// L3 cache (slowest, largest)
    pub l3_cache: LRUCache<String, Vec<u8>>,
    /// Cache statistics
    pub cache_stats: CacheStatistics,
}

impl CacheHierarchy {
    pub fn new() -> Self {
        Self {
            l1_cache: LRUCache::new(1024),             // 1KB L1
            l2_cache: LRUCache::new(1024 * 1024),      // 1MB L2
            l3_cache: LRUCache::new(10 * 1024 * 1024), // 10MB L3
            cache_stats: CacheStatistics::default(),
        }
    }

    pub fn get(&mut self, key: &str) -> Option<&Vec<u8>> {
        let start_time = Instant::now();

        // Try L1 first
        if let Some(value) = self.l1_cache.get(key) {
            self.cache_stats.hits += 1;
            self.update_access_time(start_time);
            return Some(value);
        }

        // Try L2
        if let Some(value) = self.l2_cache.get(key) {
            // Promote to L1
            let value_clone = value.clone();
            self.l1_cache.put(key.to_string(), value_clone);
            self.cache_stats.hits += 1;
            self.update_access_time(start_time);
            return self.l1_cache.get(key);
        }

        // Try L3
        if let Some(value) = self.l3_cache.get(key) {
            // Promote to L2 and L1
            let value_clone = value.clone();
            self.l2_cache.put(key.to_string(), value_clone.clone());
            self.l1_cache.put(key.to_string(), value_clone);
            self.cache_stats.hits += 1;
            self.update_access_time(start_time);
            return self.l1_cache.get(key);
        }

        self.cache_stats.misses += 1;
        self.update_access_time(start_time);
        None
    }

    pub fn put(&mut self, key: String, value: Vec<u8>) {
        // Always start with L1
        self.l1_cache.put(key, value);
    }

    fn update_access_time(&mut self, start_time: Instant) {
        let access_time = start_time.elapsed();
        let total_accesses = self.cache_stats.hits + self.cache_stats.misses;

        if total_accesses > 0 {
            let total_time = self.cache_stats.avg_access_time * (total_accesses - 1) as u32 + access_time;
            self.cache_stats.avg_access_time = total_time / total_accesses as u32;
        } else {
            self.cache_stats.avg_access_time = access_time;
        }

        self.cache_stats.hit_rate = self.cache_stats.hits as f64 / total_accesses as f64;
    }

    pub fn cleanup_unused_entries(&mut self) {
        // Simple cleanup - remove 25% of LRU entries from each cache
        self.l1_cache.cleanup(0.25);
        self.l2_cache.cleanup(0.25);
        self.l3_cache.cleanup(0.25);
    }

    pub fn adaptive_cleanup(&mut self) {
        // More aggressive cleanup when memory pressure is high
        self.l1_cache.cleanup(0.5);
        self.l2_cache.cleanup(0.4);
        self.l3_cache.cleanup(0.3);
    }

    pub fn generational_cleanup(&mut self) {
        // Clean based on age - older entries first
        self.l3_cache.cleanup(0.6);
        self.l2_cache.cleanup(0.4);
        self.l1_cache.cleanup(0.2);
    }

    pub fn get_statistics(&self) -> &CacheStatistics {
        &self.cache_stats
    }
}

/// LRU Cache implementation
#[derive(Debug)]
pub struct LRUCache<K, V> {
    /// Cache capacity
    pub capacity: usize,
    /// Current size
    pub current_size: usize,
    /// Cache data
    pub data: HashMap<K, V>,
    /// Access order
    pub access_order: VecDeque<K>,
}

impl<K: Clone + std::hash::Hash + Eq, V> LRUCache<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            current_size: 0,
            data: HashMap::new(),
            access_order: VecDeque::new(),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.data.contains_key(key) {
            // Move to front
            self.access_order.retain(|k| k != key);
            self.access_order.push_front(key.clone());
            self.data.get(key)
        } else {
            None
        }
    }

    pub fn put(&mut self, key: K, value: V) {
        if self.data.contains_key(&key) {
            // Update existing entry
            self.data.insert(key.clone(), value);
            self.access_order.retain(|k| k != &key);
            self.access_order.push_front(key);
        } else {
            // New entry
            if self.current_size >= self.capacity {
                // Evict LRU entry
                if let Some(lru_key) = self.access_order.pop_back() {
                    self.data.remove(&lru_key);
                    self.current_size -= 1;
                }
            }

            self.data.insert(key.clone(), value);
            self.access_order.push_front(key);
            self.current_size += 1;
        }
    }

    pub fn cleanup(&mut self, fraction: f64) {
        let entries_to_remove = (self.current_size as f64 * fraction) as usize;

        for _ in 0..entries_to_remove {
            if let Some(lru_key) = self.access_order.pop_back() {
                self.data.remove(&lru_key);
                self.current_size -= 1;
            } else {
                break;
            }
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.access_order.clear();
        self.current_size = 0;
    }

    pub fn len(&self) -> usize {
        self.current_size
    }

    pub fn is_empty(&self) -> bool {
        self.current_size == 0
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStatistics {
    /// Total allocated memory
    pub total_allocated: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Current usage
    pub current_usage: usize,
    /// Allocation count
    pub allocation_count: u64,
    /// Deallocation count
    pub deallocation_count: u64,
    /// Memory efficiency
    pub memory_efficiency: f64,
}

impl Default for MemoryStatistics {
    fn default() -> Self {
        Self {
            total_allocated: 0,
            peak_usage: 0,
            current_usage: 0,
            allocation_count: 0,
            deallocation_count: 0,
            memory_efficiency: 1.0,
        }
    }
}

impl MemoryStatistics {
    pub fn update_efficiency(&mut self) {
        if self.allocation_count > 0 {
            self.memory_efficiency = self.deallocation_count as f64 / self.allocation_count as f64;
        }
    }

    pub fn get_fragmentation_ratio(&self) -> f64 {
        if self.peak_usage > 0 {
            self.current_usage as f64 / self.peak_usage as f64
        } else {
            1.0
        }
    }
}

/// Allocation statistics for memory pools
#[derive(Debug, Clone)]
pub struct AllocationStatistics {
    /// Total allocations
    pub total_allocations: u64,
    /// Failed allocations
    pub failed_allocations: u64,
    /// Average allocation size
    pub avg_allocation_size: f64,
    /// Pool utilization
    pub utilization: f64,
}

impl Default for AllocationStatistics {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            failed_allocations: 0,
            avg_allocation_size: 0.0,
            utilization: 0.0,
        }
    }
}

impl AllocationStatistics {
    pub fn get_success_rate(&self) -> f64 {
        if self.total_allocations + self.failed_allocations > 0 {
            self.total_allocations as f64 / (self.total_allocations + self.failed_allocations) as f64
        } else {
            1.0
        }
    }
}

/// Cache performance statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Hit rate
    pub hit_rate: f64,
    /// Average access time
    pub avg_access_time: Duration,
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self {
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
            avg_access_time: Duration::from_nanos(0),
        }
    }
}

impl CacheStatistics {
    pub fn reset(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.hit_rate = 0.0;
        self.avg_access_time = Duration::from_nanos(0);
    }

    pub fn total_accesses(&self) -> u64 {
        self.hits + self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scientific_performance_optimization::config::MemoryOptimizationConfig;

    #[test]
    fn test_hierarchical_memory_manager() {
        let config = MemoryOptimizationConfig::default();
        let manager = HierarchicalMemoryManager::new(config);

        assert!(!manager.memory_pools.is_empty());
        assert_eq!(manager.memory_stats.allocation_count, 0);
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new("test_pool".to_string(), 1024, 4096);
        assert_eq!(pool.block_size, 1024);
        assert_eq!(pool.total_capacity, 4096);
        assert_eq!(pool.used_capacity, 0);
    }

    #[test]
    fn test_lru_cache() {
        let mut cache = LRUCache::new(2);

        cache.put("key1", "value1");
        cache.put("key2", "value2");

        assert_eq!(cache.get(&"key1"), Some(&"value1"));

        // This should evict key2
        cache.put("key3", "value3");
        assert_eq!(cache.get(&"key2"), None);
        assert_eq!(cache.get(&"key1"), Some(&"value1"));
        assert_eq!(cache.get(&"key3"), Some(&"value3"));
    }

    #[test]
    fn test_cache_hierarchy() {
        let mut hierarchy = CacheHierarchy::new();

        hierarchy.put("test_key".to_string(), vec![1, 2, 3, 4]);

        let result = hierarchy.get("test_key");
        assert!(result.is_some());
        assert_eq!(result.unwrap(), &vec![1, 2, 3, 4]);

        // Test cache miss
        let missing = hierarchy.get("missing_key");
        assert!(missing.is_none());
    }

    #[test]
    fn test_memory_statistics() {
        let mut stats = MemoryStatistics::default();

        stats.allocation_count = 10;
        stats.deallocation_count = 8;
        stats.update_efficiency();

        assert_eq!(stats.memory_efficiency, 0.8);
    }
}