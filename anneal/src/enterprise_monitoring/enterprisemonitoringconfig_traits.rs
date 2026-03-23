//! # EnterpriseMonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnterpriseMonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    BusinessMetricsConfig, CostMonitoringConfig, DataGovernanceConfig, EnterpriseMonitoringConfig,
    IntegrationConfig, SecurityMonitoringConfig, SloConfig,
};

impl Default for EnterpriseMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_structured_logging: true,
            enable_distributed_tracing: true,
            slo_config: SloConfig::default(),
            security_config: SecurityMonitoringConfig::default(),
            business_metrics_config: BusinessMetricsConfig::default(),
            cost_monitoring_config: CostMonitoringConfig::default(),
            data_governance_config: DataGovernanceConfig::default(),
            integration_config: IntegrationConfig::default(),
        }
    }
}
