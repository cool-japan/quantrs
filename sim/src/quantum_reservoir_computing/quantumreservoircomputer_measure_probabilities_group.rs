//! # QuantumReservoirComputer - measure_probabilities_group Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Measure probability distribution
    pub(super) fn measure_probabilities(&self) -> Result<Array1<f64>> {
        let probabilities: Vec<f64> = self
            .reservoir_state
            .state_vector
            .iter()
            .map(scirs2_core::Complex::norm_sqr)
            .collect();
        let max_size = 1 << 10;
        if probabilities.len() > max_size {
            let mut sampled = Vec::with_capacity(max_size);
            for _ in 0..max_size {
                let idx = thread_rng().random_range(0..probabilities.len());
                sampled.push(probabilities[idx]);
            }
            Ok(Array1::from_vec(sampled))
        } else {
            Ok(Array1::from_vec(probabilities))
        }
    }
}
