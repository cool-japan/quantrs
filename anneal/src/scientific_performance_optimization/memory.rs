//! Memory management types for scientific performance optimization.
//!
//! This module contains hierarchical memory management, memory pools,
//! cache hierarchies, and memory statistics.

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use super::config::MemoryOptimizationConfig;

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
    /// Create a new hierarchical memory manager
    #[must_use]
    pub fn new(config: MemoryOptimizationConfig) -> Self {
        Self {
            config,
            memory_pools: HashMap::new(),
            cache_hierarchy: CacheHierarchy::new(),
            memory_stats: MemoryStatistics::default(),
        }
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
    /// Create a new cache hierarchy
    #[must_use]
    pub fn new() -> Self {
        Self {
            l1_cache: LRUCache::new(1024),             // 1KB L1
            l2_cache: LRUCache::new(1024 * 1024),      // 1MB L2
            l3_cache: LRUCache::new(10 * 1024 * 1024), // 10MB L3
            cache_stats: CacheStatistics::default(),
        }
    }
}

impl Default for CacheHierarchy {
    fn default() -> Self {
        Self::new()
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
    /// Create a new LRU cache with given capacity
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            current_size: 0,
            data: HashMap::new(),
            access_order: VecDeque::new(),
        }
    }

    /// Get value from cache
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.data.contains_key(key) {
            // Move to front of access order
            self.access_order.retain(|k| k != key);
            self.access_order.push_front(key.clone());
            self.data.get(key)
        } else {
            None
        }
    }

    /// Insert value into cache
    pub fn insert(&mut self, key: K, value: V) {
        // Remove old entry if exists
        if self.data.contains_key(&key) {
            self.access_order.retain(|k| k != &key);
        } else if self.current_size >= self.capacity {
            // Evict least recently used
            if let Some(lru_key) = self.access_order.pop_back() {
                self.data.remove(&lru_key);
                self.current_size = self.current_size.saturating_sub(1);
            }
        }

        self.data.insert(key.clone(), value);
        self.access_order.push_front(key);
        self.current_size += 1;
    }

    /// Check if key exists in cache
    #[must_use]
    pub fn contains(&self, key: &K) -> bool {
        self.data.contains_key(key)
    }

    /// Get current size of cache
    #[must_use]
    pub fn len(&self) -> usize {
        self.current_size
    }

    /// Check if cache is empty
    #[must_use]
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
    /// Record an allocation
    pub fn record_allocation(&mut self, size: usize) {
        self.total_allocated += size;
        self.current_usage += size;
        self.allocation_count += 1;

        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }

        self.update_efficiency();
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
        self.deallocation_count += 1;
        self.update_efficiency();
    }

    /// Update memory efficiency
    fn update_efficiency(&mut self) {
        if self.peak_usage > 0 {
            self.memory_efficiency = self.current_usage as f64 / self.peak_usage as f64;
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
    /// Record a cache hit
    pub fn record_hit(&mut self) {
        self.hits += 1;
        self.update_hit_rate();
    }

    /// Record a cache miss
    pub fn record_miss(&mut self) {
        self.misses += 1;
        self.update_hit_rate();
    }

    /// Update hit rate
    fn update_hit_rate(&mut self) {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hit_rate = self.hits as f64 / total as f64;
        }
    }
}
