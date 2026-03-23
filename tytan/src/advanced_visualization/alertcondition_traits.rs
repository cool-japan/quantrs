//! # AlertCondition - Trait Implementations
//!
//! This module contains trait implementations for `AlertCondition`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AlertCondition;

impl std::fmt::Debug for AlertCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ThresholdExceeded { metric, threshold } => f
                .debug_struct("ThresholdExceeded")
                .field("metric", metric)
                .field("threshold", threshold)
                .finish(),
            Self::TrendDetected { trend, duration } => f
                .debug_struct("TrendDetected")
                .field("trend", trend)
                .field("duration", duration)
                .finish(),
            Self::AnomalyDetected { anomaly_type } => f
                .debug_struct("AnomalyDetected")
                .field("anomaly_type", anomaly_type)
                .finish(),
            Self::Custom(_) => f
                .debug_struct("Custom")
                .field("function", &"<custom function>")
                .finish(),
        }
    }
}

impl Clone for AlertCondition {
    fn clone(&self) -> Self {
        match self {
            Self::ThresholdExceeded { metric, threshold } => Self::ThresholdExceeded {
                metric: metric.clone(),
                threshold: *threshold,
            },
            Self::TrendDetected { trend, duration } => Self::TrendDetected {
                trend: trend.clone(),
                duration: *duration,
            },
            Self::AnomalyDetected { anomaly_type } => Self::AnomalyDetected {
                anomaly_type: anomaly_type.clone(),
            },
            Self::Custom(_) => Self::Custom(Box::new(|_| false)),
        }
    }
}
