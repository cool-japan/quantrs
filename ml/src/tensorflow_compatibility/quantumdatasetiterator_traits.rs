//! # QuantumDatasetIterator - Trait Implementations
//!
//! This module contains trait implementations for `QuantumDatasetIterator`.
//!
//! ## Implemented Traits
//!
//! - `Iterator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::simulator_backends::{DynamicCircuit, Observable, SimulationResult, SimulatorBackend};
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, ArrayD, Axis};

use super::types::QuantumDatasetIterator;

impl<'a> Iterator for QuantumDatasetIterator<'a> {
    type Item = (Vec<DynamicCircuit>, Array2<f64>, Array1<f64>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_batch >= self.total_batches {
            return None;
        }
        let start_idx = self.current_batch * self.dataset.batch_size;
        let end_idx =
            ((self.current_batch + 1) * self.dataset.batch_size).min(self.dataset.circuits.len());
        let batch_circuits = self.dataset.circuits[start_idx..end_idx].to_vec();
        let batch_parameters = self
            .dataset
            .parameters
            .slice(s![start_idx..end_idx, ..])
            .to_owned();
        let batch_labels = self.dataset.labels.slice(s![start_idx..end_idx]).to_owned();
        self.current_batch += 1;
        Some((batch_circuits, batch_parameters, batch_labels))
    }
}
