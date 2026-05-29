//! # TensorNetworkSampler - Trait Implementations
//!
//! This module contains trait implementations for `TensorNetworkSampler`.
//!
//! ## Implemented Traits
//!
//! - `Sampler`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::sampler::{SampleResult, Sampler, SamplerError, SamplerResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayD};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;

use super::types::TensorNetworkSampler;

impl Sampler for TensorNetworkSampler {
    fn run_qubo(
        &self,
        qubo: &(
            scirs2_core::ndarray::Array2<f64>,
            std::collections::HashMap<String, usize>,
        ),
        num_reads: usize,
    ) -> SamplerResult<Vec<crate::sampler::SampleResult>> {
        let (matrix, var_map) = qubo;
        // Lift the 2-D QUBO matrix to a dynamic-rank tensor (ndim == 2)
        // so that run_hobo can handle it via the existing dispatch path.
        let tensor: ArrayD<f64> = matrix.clone().into_dyn();
        self.run_hobo(&(tensor, var_map.clone()), num_reads)
    }
    fn run_hobo(
        &self,
        problem: &(
            scirs2_core::ndarray::ArrayD<f64>,
            std::collections::HashMap<String, usize>,
        ),
        num_reads: usize,
    ) -> SamplerResult<Vec<crate::sampler::SampleResult>> {
        let (hamiltonian, _var_map) = problem;
        let mut sampler_copy = Self::new(self.config.clone());
        match sampler_copy.sample(hamiltonian, num_reads) {
            Ok(results) => Ok(results),
            Err(e) => Err(SamplerError::InvalidParameter(e.to_string())),
        }
    }
}
