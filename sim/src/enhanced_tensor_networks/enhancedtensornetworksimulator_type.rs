//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::scirs2_integration::SciRS2Backend;
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, HashSet};

#[cfg(feature = "advanced_math")]
use super::types::ContractionOptimizer;
use super::types::{
    EnhancedTensor, EnhancedTensorNetworkConfig, TensorNetwork, TensorNetworkStats,
};

/// Enhanced tensor network simulator
pub struct EnhancedTensorNetworkSimulator {
    /// Configuration
    pub(super) config: EnhancedTensorNetworkConfig,
    /// Current tensor network
    pub(super) network: TensorNetwork,
    /// `SciRS2` backend
    pub(super) backend: Option<SciRS2Backend>,
    /// Contraction optimizer
    #[cfg(feature = "advanced_math")]
    pub(super) optimizer: Option<ContractionOptimizer>,
    /// Tensor cache for reused patterns
    pub(super) tensor_cache: HashMap<String, EnhancedTensor>,
    /// Performance statistics
    pub(super) stats: TensorNetworkStats,
}
