//! # SamplingConfig - Trait Implementations
//!
//! This module contains trait implementations for `SamplingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{ProposalDistribution, SamplingAlgorithm, SamplingConfig, WaveFunctionConfig};

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            algorithm_type: SamplingAlgorithm::QuantumInspiredMCMC,
            num_samples: 10_000,
            burn_in: 1000,
            thinning: 10,
            proposal_distribution: ProposalDistribution::Gaussian,
            wave_function_config: WaveFunctionConfig::default(),
        }
    }
}
