//! # UserAnalyticsConfig - Trait Implementations
//!
//! This module contains trait implementations for `UserAnalyticsConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::UserAnalyticsConfig;

impl Default for UserAnalyticsConfig {
    fn default() -> Self {
        Self {
            track_sessions: true,
            track_feature_usage: true,
            behavior_analysis: true,
            privacy_preserving: true,
        }
    }
}
