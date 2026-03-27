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
        _qubo: &(
            scirs2_core::ndarray::Array2<f64>,
            std::collections::HashMap<String, usize>,
        ),
        _num_reads: usize,
    ) -> SamplerResult<Vec<crate::sampler::SampleResult>> {
        Err(SamplerError::NotImplemented(
            "Use run_hobo instead ".to_string(),
        ))
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
