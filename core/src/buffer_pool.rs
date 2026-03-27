//! Temporary BufferPool implementation to replace scirs2_core::memory::BufferPool
//! TODO: Replace with scirs2_core when regex dependency issue is fixed

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Mutex;

/// A simple buffer pool implementation
pub struct BufferPool<T> {
    pool: Mutex<VecDeque<Vec<T>>>,
    _phantom: PhantomData<T>,
}

impl<T: Clone + Default> BufferPool<T> {
    /// Create a new buffer pool
    pub const fn new() -> Self {
        Self {
            pool: Mutex::new(VecDeque::new()),
            _phantom: PhantomData,
        }
    }

    /// Get a buffer from the pool
    pub fn get(&self, size: usize) -> Vec<T> {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        pool.pop_front().map_or_else(
            || vec![T::default(); size],
            |mut buffer| {
                buffer.resize(size, T::default());
                buffer
            },
        )
    }

    /// Return a buffer to the pool for reuse
    pub fn put(&self, buffer: Vec<T>) {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        pool.push_back(buffer);
    }

    /// Alias for `put` — return a buffer to the pool
    #[inline]
    pub fn return_buffer(&self, buffer: Vec<T>) {
        self.put(buffer);
    }

    /// Returns the current number of pooled buffers
    pub fn pool_size(&self) -> usize {
        self.pool.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Clear all buffers in the pool
    pub fn clear(&self) {
        let mut pool = self.pool.lock().unwrap_or_else(|e| e.into_inner());
        pool.clear();
    }
}

impl<T> Default for BufferPool<T> {
    fn default() -> Self {
        Self {
            pool: Mutex::new(VecDeque::new()),
            _phantom: PhantomData,
        }
    }
}
