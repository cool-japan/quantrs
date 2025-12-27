//! GPU memory pooling for efficient allocation and reuse.
//!
//! This module provides memory pooling functionality to reduce allocation
//! overhead in GPU computations, particularly for iterative algorithms.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

#[cfg(feature = "scirs")]
use scirs2_core::gpu;

// Stub for missing GPU functionality
#[cfg(feature = "scirs")]
pub struct GpuContext;

#[cfg(feature = "scirs")]
impl GpuContext {
    pub fn new(_device_id: u32) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self)
    }
}

#[cfg(feature = "scirs")]
#[derive(Clone, Default)]
pub struct GpuMemory {
    id: usize,
    size: usize,
}

/// Memory block information
#[cfg(feature = "scirs")]
#[derive(Clone)]
struct MemoryBlock {
    /// Unique ID for this block
    id: usize,
    /// Size in bytes
    size: usize,
    /// Whether the block is currently in use
    in_use: bool,
    /// Last access time for LRU eviction
    last_access: std::time::Instant,
}

/// GPU memory pool for efficient allocation
pub struct GpuMemoryPool {
    /// GPU context
    #[cfg(feature = "scirs")]
    context: Arc<GpuContext>,
    /// Pool of memory blocks by size
    #[cfg(feature = "scirs")]
    blocks_by_size: HashMap<usize, VecDeque<MemoryBlock>>,
    /// All allocated blocks
    #[cfg(feature = "scirs")]
    all_blocks: Vec<MemoryBlock>,
    /// Maximum pool size in bytes
    max_size: usize,
    /// Current allocated size
    current_size: usize,
    /// Allocation statistics
    stats: AllocationStats,
    /// Mutex for thread safety
    mutex: Arc<Mutex<()>>,
    /// Next block ID
    next_block_id: usize,
}

/// Allocation statistics
#[derive(Default, Clone)]
pub struct AllocationStats {
    /// Total allocations
    pub total_allocations: usize,
    /// Cache hits (reused blocks)
    pub cache_hits: usize,
    /// Cache misses (new allocations)
    pub cache_misses: usize,
    /// Total bytes allocated
    pub total_bytes_allocated: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
    /// Number of evictions
    pub evictions: usize,
}

#[cfg(feature = "scirs")]
impl GpuMemoryPool {
    /// Create a new memory pool
    pub fn new(context: Arc<GpuContext>, max_size: usize) -> Self {
        Self {
            context,
            blocks_by_size: HashMap::new(),
            all_blocks: Vec::new(),
            max_size,
            current_size: 0,
            stats: AllocationStats::default(),
            mutex: Arc::new(Mutex::new(())),
            next_block_id: 0,
        }
    }

    /// Allocate memory from the pool
    #[cfg(feature = "scirs")]
    pub fn allocate(&mut self, size: usize) -> Result<GpuMemory, String> {
        let _lock = self
            .mutex
            .lock()
            .map_err(|e| format!("Failed to acquire lock in allocate: {e}"))?;

        self.stats.total_allocations += 1;

        // Round up to nearest power of 2 for better reuse
        let aligned_size = size.next_power_of_two();

        // Check if we have a free block of the right size
        if let Some(blocks) = self.blocks_by_size.get_mut(&aligned_size) {
            if let Some(mut block) = blocks.pop_front() {
                if !block.in_use {
                    block.in_use = true;
                    block.last_access = std::time::Instant::now();
                    self.stats.cache_hits += 1;

                    // Update the block in all_blocks
                    for b in &mut self.all_blocks {
                        if b.id == block.id {
                            b.in_use = true;
                            b.last_access = block.last_access;
                            break;
                        }
                    }

                    return Ok(GpuMemory {
                        id: block.id,
                        size: block.size,
                    });
                }
            }
        }

        // No suitable block found, allocate new
        self.stats.cache_misses += 1;

        // Check if we need to evict blocks
        if self.current_size + aligned_size > self.max_size {
            // Drop the lock before calling evict method
            drop(_lock);
            self.evict_lru_blocks(aligned_size)?;
            // Re-acquire lock
            let _lock = self
                .mutex
                .lock()
                .map_err(|e| format!("Failed to re-acquire lock after eviction: {e}"))?;
        }

        // Allocate new block
        let block_id = self.next_block_id;
        self.next_block_id += 1;

        let block = MemoryBlock {
            id: block_id,
            size: aligned_size,
            in_use: true,
            last_access: std::time::Instant::now(),
        };

        self.all_blocks.push(block);
        self.current_size += aligned_size;
        self.stats.total_bytes_allocated += aligned_size;

        if self.current_size > self.stats.peak_memory_usage {
            self.stats.peak_memory_usage = self.current_size;
        }

        Ok(GpuMemory {
            id: block_id,
            size: aligned_size,
        })
    }

    /// Release memory back to the pool
    #[cfg(feature = "scirs")]
    pub fn release(&mut self, memory: GpuMemory) {
        // Use if let to gracefully handle lock poisoning
        if let Ok(_lock) = self.mutex.lock() {
            // Find the block and mark it as free
            for block in &mut self.all_blocks {
                if block.id == memory.id {
                    block.in_use = false;
                    block.last_access = std::time::Instant::now();

                    // Add to the pool for reuse
                    self.blocks_by_size
                        .entry(block.size)
                        .or_default()
                        .push_back(block.clone());

                    break;
                }
            }
        }
        // If lock is poisoned, we silently skip releasing to avoid panic
    }

    /// Evict least recently used blocks to make space
    #[cfg(feature = "scirs")]
    fn evict_lru_blocks(&mut self, required_size: usize) -> Result<(), String> {
        let mut freed_size = 0;
        let mut blocks_to_evict = Vec::new();

        // Sort blocks by last access time
        let mut free_blocks: Vec<_> = self.all_blocks.iter().filter(|b| !b.in_use).collect();
        free_blocks.sort_by_key(|b| b.last_access);

        // Evict blocks until we have enough space
        for block in free_blocks {
            if freed_size >= required_size {
                break;
            }

            blocks_to_evict.push(block.id);
            freed_size += block.size;
            self.stats.evictions += 1;
        }

        if freed_size < required_size {
            return Err("Insufficient memory in pool even after eviction".to_string());
        }

        // Actually evict the blocks
        for block_id in blocks_to_evict {
            self.all_blocks.retain(|b| b.id != block_id);

            // Remove from size-based pools
            for blocks in self.blocks_by_size.values_mut() {
                blocks.retain(|b| b.id != block_id);
            }

            // Free GPU memory
            // TODO: Implement free_raw in GPU stub
            // unsafe {
            //     self.context
            //         .free_raw(ptr)
            //         .map_err(|e| format!("Failed to free GPU memory: {}", e))?;
            // }
        }

        self.current_size -= freed_size;

        Ok(())
    }

    /// Get allocation statistics
    pub fn stats(&self) -> AllocationStats {
        self.stats.clone()
    }

    /// Clear the entire pool
    #[cfg(feature = "scirs")]
    pub fn clear(&mut self) -> Result<(), String> {
        let _lock = self
            .mutex
            .lock()
            .map_err(|e| format!("Failed to acquire lock in clear: {e}"))?;

        // Clear all blocks (in a real implementation, this would free GPU memory)
        // For our stub implementation, we just clear the tracking structures

        self.blocks_by_size.clear();
        self.all_blocks.clear();
        self.current_size = 0;

        Ok(())
    }

    /// Defragment the pool to reduce fragmentation
    #[cfg(feature = "scirs")]
    pub fn defragment(&mut self) -> Result<(), String> {
        let _lock = self
            .mutex
            .lock()
            .map_err(|e| format!("Failed to acquire lock in defragment: {e}"))?;

        // This is a complex operation that would involve:
        // 1. Identifying fragmented regions
        // 2. Allocating new contiguous blocks
        // 3. Copying data
        // 4. Updating pointers
        // 5. Freeing old blocks

        // For now, we just compact the free block lists
        for blocks in self.blocks_by_size.values_mut() {
            blocks.retain(|b| !b.in_use);
        }

        Ok(())
    }
}

/// Scoped memory allocation that automatically returns to pool
pub struct ScopedGpuMemory {
    memory: Option<GpuMemory>,
    pool: Arc<Mutex<GpuMemoryPool>>,
}

impl ScopedGpuMemory {
    /// Create a new scoped allocation
    #[cfg(feature = "scirs")]
    pub fn new(pool: Arc<Mutex<GpuMemoryPool>>, size: usize) -> Result<Self, String> {
        let memory = pool
            .lock()
            .map_err(|e| format!("Failed to acquire pool lock: {e}"))?
            .allocate(size)?;
        Ok(Self {
            memory: Some(memory),
            pool,
        })
    }

    /// Get the underlying memory
    ///
    /// # Panics
    /// Panics if called after the memory has been released (should never happen in normal use)
    #[cfg(feature = "scirs")]
    pub fn memory(&self) -> &GpuMemory {
        self.memory
            .as_ref()
            .expect("ScopedGpuMemory::memory called after memory was released - this is a bug")
    }

    /// Get mutable access to memory
    ///
    /// # Panics
    /// Panics if called after the memory has been released (should never happen in normal use)
    #[cfg(feature = "scirs")]
    pub fn memory_mut(&mut self) -> &mut GpuMemory {
        self.memory
            .as_mut()
            .expect("ScopedGpuMemory::memory_mut called after memory was released - this is a bug")
    }
}

#[cfg(feature = "scirs")]
impl Drop for ScopedGpuMemory {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            // Use if let to gracefully handle lock poisoning during drop
            if let Ok(mut pool) = self.pool.lock() {
                pool.release(memory);
            }
            // If lock is poisoned, we silently skip releasing to avoid panic in Drop
        }
    }
}

/// Memory pool manager for multiple devices
pub struct MultiDeviceMemoryPool {
    /// Pools for each device
    device_pools: HashMap<usize, Arc<Mutex<GpuMemoryPool>>>,
}

impl Default for MultiDeviceMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiDeviceMemoryPool {
    /// Create a new multi-device pool
    pub fn new() -> Self {
        Self {
            device_pools: HashMap::new(),
        }
    }

    /// Add a device pool
    #[cfg(feature = "scirs")]
    pub fn add_device(&mut self, device_id: usize, context: Arc<GpuContext>, max_size: usize) {
        let pool = Arc::new(Mutex::new(GpuMemoryPool::new(context, max_size)));
        self.device_pools.insert(device_id, pool);
    }

    /// Get pool for a device
    pub fn get_pool(&self, device_id: usize) -> Option<Arc<Mutex<GpuMemoryPool>>> {
        self.device_pools.get(&device_id).cloned()
    }

    /// Allocate from a specific device
    #[cfg(feature = "scirs")]
    pub fn allocate(&self, device_id: usize, size: usize) -> Result<ScopedGpuMemory, String> {
        let pool = self
            .get_pool(device_id)
            .ok_or_else(|| format!("No pool for device {device_id}"))?;

        ScopedGpuMemory::new(pool, size)
    }

    /// Get combined statistics
    ///
    /// Note: Skips any device pools that cannot be locked (e.g., due to lock poisoning)
    pub fn combined_stats(&self) -> AllocationStats {
        let mut combined = AllocationStats::default();

        for pool in self.device_pools.values() {
            // Use if let to gracefully handle lock poisoning
            if let Ok(pool_guard) = pool.lock() {
                let stats = pool_guard.stats();
                combined.total_allocations += stats.total_allocations;
                combined.cache_hits += stats.cache_hits;
                combined.cache_misses += stats.cache_misses;
                combined.total_bytes_allocated += stats.total_bytes_allocated;
                combined.peak_memory_usage += stats.peak_memory_usage;
                combined.evictions += stats.evictions;
            }
            // Silently skip pools we can't lock to avoid panic
        }

        combined
    }
}

// Placeholder implementations when SciRS2 is not available
#[cfg(not(feature = "scirs"))]
pub struct GpuMemory;

#[cfg(not(feature = "scirs"))]
impl GpuMemoryPool {
    pub fn new(_max_size: usize) -> Self {
        Self {
            max_size: 0,
            current_size: 0,
            stats: AllocationStats::default(),
            mutex: Arc::new(Mutex::new(())),
            next_block_id: 0,
        }
    }

    pub fn allocate(&mut self, _size: usize) -> Result<GpuMemory, String> {
        Err("GPU memory pooling requires SciRS2 feature".to_string())
    }

    pub fn stats(&self) -> AllocationStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_stats() {
        let stats = AllocationStats {
            total_allocations: 100,
            cache_hits: 80,
            cache_misses: 20,
            total_bytes_allocated: 1024 * 1024,
            peak_memory_usage: 512 * 1024,
            evictions: 5,
        };

        assert_eq!(stats.total_allocations, 100);
        assert_eq!(stats.cache_hits, 80);

        let hit_rate = stats.cache_hits as f64 / stats.total_allocations as f64;
        assert!(hit_rate > 0.79 && hit_rate < 0.81);
    }

    #[test]
    #[cfg(feature = "scirs")]
    fn test_memory_pool_basic() {
        use crate::gpu_memory_pool::GpuContext;

        let context = Arc::new(GpuContext::new(0).expect("Failed to create GPU context for test"));
        let mut pool = GpuMemoryPool::new(context, 1024 * 1024); // 1MB pool

        // First allocation should be a cache miss
        let mem1 = pool
            .allocate(1024)
            .expect("First allocation should succeed");
        assert_eq!(pool.stats().cache_misses, 1);
        assert_eq!(pool.stats().cache_hits, 0);

        // Release and reallocate should be a cache hit
        pool.release(mem1);
        let _mem2 = pool
            .allocate(1024)
            .expect("Second allocation should succeed");
        assert_eq!(pool.stats().cache_misses, 1);
        assert_eq!(pool.stats().cache_hits, 1);
    }
}
