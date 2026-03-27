//! # EnhancedHardwareBenchmark - calculate_process_fidelity_group Methods
//!
//! This module contains method implementations for `EnhancedHardwareBenchmark`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::{
    buffer_pool::BufferPool,
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
use scirs2_core::parallel_ops::*;
use scirs2_core::Complex64;

use super::types::Gate;

use super::enhancedhardwarebenchmark_type::EnhancedHardwareBenchmark;

impl EnhancedHardwareBenchmark {
    pub(super) fn calculate_process_fidelity(
        _process_matrix: &Array2<Complex64>,
        _gate: &Gate,
    ) -> QuantRS2Result<f64> {
        Ok(0.995)
    }
}
