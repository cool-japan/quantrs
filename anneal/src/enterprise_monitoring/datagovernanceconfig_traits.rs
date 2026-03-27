//! # DataGovernanceConfig - Trait Implementations
//!
//! This module contains trait implementations for `DataGovernanceConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant, SystemTime};

use super::types::{DataGovernanceConfig, DataType, RetentionPolicy};

impl Default for DataGovernanceConfig {
    fn default() -> Self {
        Self {
            enable_data_governance: true,
            track_data_lineage: true,
            privacy_compliance: true,
            data_quality_monitoring: true,
            retention_policies: vec![
                RetentionPolicy {
                    data_type: DataType::Logs,
                    retention_period: Duration::from_secs(90 * 24 * 3600),
                    archive_after: Some(Duration::from_secs(30 * 24 * 3600)),
                },
                RetentionPolicy {
                    data_type: DataType::Metrics,
                    retention_period: Duration::from_secs(365 * 24 * 3600),
                    archive_after: Some(Duration::from_secs(90 * 24 * 3600)),
                },
                RetentionPolicy {
                    data_type: DataType::SecurityEvents,
                    retention_period: Duration::from_secs(2555 * 24 * 3600),
                    archive_after: Some(Duration::from_secs(365 * 24 * 3600)),
                },
            ],
        }
    }
}
