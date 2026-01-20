//! Data loading utilities for PyTorch-like API

use crate::error::{MLError, Result};
use crate::scirs2_integration::SciRS2Array;

/// Data loader trait
pub trait DataLoader {
    /// Get next batch
    fn next_batch(&mut self) -> Result<Option<(SciRS2Array, SciRS2Array)>>;

    /// Reset to beginning
    fn reset(&mut self);

    /// Get batch size
    fn batch_size(&self) -> usize;
}

/// Simple in-memory data loader
pub struct MemoryDataLoader {
    /// Input data
    inputs: SciRS2Array,
    /// Target data
    targets: SciRS2Array,
    /// Batch size
    batch_size_val: usize,
    /// Current position
    current_pos: usize,
    /// Shuffle data
    shuffle: bool,
    /// Indices for shuffling
    indices: Vec<usize>,
}

impl MemoryDataLoader {
    /// Create new memory data loader
    pub fn new(
        inputs: SciRS2Array,
        targets: SciRS2Array,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<Self> {
        let num_samples = inputs.data.shape()[0];
        if targets.data.shape()[0] != num_samples {
            return Err(MLError::InvalidConfiguration(
                "Input and target batch sizes don't match".to_string(),
            ));
        }

        let indices: Vec<usize> = (0..num_samples).collect();

        Ok(Self {
            inputs,
            targets,
            batch_size_val: batch_size,
            current_pos: 0,
            shuffle,
            indices,
        })
    }

    /// Shuffle indices
    fn shuffle_indices(&mut self) {
        if self.shuffle {
            for i in (1..self.indices.len()).rev() {
                let j = fastrand::usize(0..=i);
                self.indices.swap(i, j);
            }
        }
    }
}

impl DataLoader for MemoryDataLoader {
    fn next_batch(&mut self) -> Result<Option<(SciRS2Array, SciRS2Array)>> {
        if self.current_pos >= self.indices.len() {
            return Ok(None);
        }

        let end_pos = (self.current_pos + self.batch_size_val).min(self.indices.len());
        let _batch_indices = &self.indices[self.current_pos..end_pos];

        // Extract batch data (simplified - would use proper indexing)
        let batch_inputs = self.inputs.clone();
        let batch_targets = self.targets.clone();

        self.current_pos = end_pos;

        Ok(Some((batch_inputs, batch_targets)))
    }

    fn reset(&mut self) {
        self.current_pos = 0;
        self.shuffle_indices();
    }

    fn batch_size(&self) -> usize {
        self.batch_size_val
    }
}
