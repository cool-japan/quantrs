//! # MemoryAnalysisConfig - Trait Implementations
//!
//! This module contains trait implementations for `MemoryAnalysisConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{EntropyMeasure, IPCFunction, MemoryAnalysisConfig, MemoryTask};

impl Default for MemoryAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_capacity_estimation: true,
            capacity_tasks: vec![
                MemoryTask::DelayLine,
                MemoryTask::TemporalXOR,
                MemoryTask::Parity,
            ],
            enable_nonlinear: true,
            nonlinearity_orders: vec![2, 3, 4],
            enable_temporal_correlation: true,
            correlation_lags: (1..=20).collect(),
            enable_ipc: true,
            ipc_functions: vec![
                IPCFunction::Linear,
                IPCFunction::Quadratic,
                IPCFunction::Cubic,
            ],
            enable_entropy: true,
            entropy_measures: vec![
                EntropyMeasure::Shannon,
                EntropyMeasure::Renyi,
                EntropyMeasure::VonNeumann,
            ],
        }
    }
}
