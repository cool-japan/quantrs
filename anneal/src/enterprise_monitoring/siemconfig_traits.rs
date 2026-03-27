//! # SiemConfig - Trait Implementations
//!
//! This module contains trait implementations for `SiemConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AuthenticationMethod, RetryPolicy, SecurityEventType, SiemConfig};

impl Default for SiemConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://siem.company.com/api/events".to_string(),
            authentication: AuthenticationMethod::ApiKey(String::new()),
            event_types: vec![
                SecurityEventType::Authentication,
                SecurityEventType::Authorization,
                SecurityEventType::DataAccess,
                SecurityEventType::ThreatDetection,
            ],
            batch_size: 100,
            retry_policy: RetryPolicy::default(),
        }
    }
}
