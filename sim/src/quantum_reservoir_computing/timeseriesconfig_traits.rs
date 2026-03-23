//! # TimeSeriesConfig - Trait Implementations
//!
//! This module contains trait implementations for `TimeSeriesConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{MemoryKernel, TimeSeriesConfig, TrendDetectionMethod};

impl Default for TimeSeriesConfig {
    fn default() -> Self {
        Self {
            enable_arima: true,
            ar_order: 2,
            ma_order: 1,
            diff_order: 1,
            enable_nar: true,
            nar_order: 3,
            memory_kernel: MemoryKernel::Exponential,
            kernel_params: vec![0.9, 0.1],
            enable_seasonal: false,
            seasonal_period: 12,
            trend_method: TrendDetectionMethod::LinearRegression,
            enable_changepoint: false,
            anomaly_threshold: 2.0,
        }
    }
}
