//! # EnhancedTensorNetworkSimulator - accumulate_block_result_group Methods
//!
//! This module contains method implementations for `EnhancedTensorNetworkSimulator`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::ndarray::{Array, Array2, ArrayD, IxDyn};
use scirs2_core::random::prelude::*;
use scirs2_core::Complex64;

use super::enhancedtensornetworksimulator_type::EnhancedTensorNetworkSimulator;

impl EnhancedTensorNetworkSimulator {
    pub(super) const fn accumulate_block_result(
        _result: &ArrayD<Complex64>,
        _block_result: &ArrayD<Complex64>,
        _block_idx: usize,
    ) -> Result<()> {
        Ok(())
    }
}
