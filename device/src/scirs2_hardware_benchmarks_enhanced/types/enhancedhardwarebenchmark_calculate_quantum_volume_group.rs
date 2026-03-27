//! # EnhancedHardwareBenchmark - calculate_quantum_volume_group Methods
//!
//! This module contains method implementations for `EnhancedHardwareBenchmark`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::{
    buffer_pool::BufferPool,
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use scirs2_core::parallel_ops::*;

use super::types::BenchmarkSuiteResult;

use super::enhancedhardwarebenchmark_type::EnhancedHardwareBenchmark;

impl EnhancedHardwareBenchmark {
    pub(super) fn calculate_quantum_volume(result: &BenchmarkSuiteResult) -> QuantRS2Result<usize> {
        let mut max_qv = 1;
        for (n, measurements) in &result.measurements {
            let success_rates: Vec<f64> = measurements.iter().map(|m| m.success_rate).collect();
            let avg_success = success_rates.iter().sum::<f64>() / success_rates.len() as f64;
            if avg_success > 2.0 / 3.0 {
                max_qv = max_qv.max(1 << n);
            }
        }
        Ok(max_qv)
    }
}
