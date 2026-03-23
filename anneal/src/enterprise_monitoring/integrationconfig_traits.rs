//! # IntegrationConfig - Trait Implementations
//!
//! This module contains trait implementations for `IntegrationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CicdConfig, IntegrationConfig, ItsmConfig, SiemConfig};

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            siem_integration: Some(SiemConfig::default()),
            itsm_integration: Some(ItsmConfig::default()),
            cicd_integration: Some(CicdConfig::default()),
            external_tools: vec![],
        }
    }
}
