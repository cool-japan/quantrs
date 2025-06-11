//! Quantum Algorithm Marketplace Configuration Types

use serde::{Deserialize, Serialize};

/// Configuration for Quantum Algorithm Optimization Marketplace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAlgorithmMarketplaceConfig {
    /// Algorithm registry configuration
    pub registry_config: AlgorithmRegistryConfig,
    /// Optimization engine configuration
    pub optimization_config: MarketplaceOptimizationConfig,
    /// Benchmarking and evaluation configuration
    pub benchmarking_config: BenchmarkingConfig,
    /// Collaboration and sharing configuration
    pub collaboration_config: CollaborationConfig,
    /// Discovery and recommendation configuration
    pub discovery_config: AlgorithmDiscoveryConfig,
    /// Performance analytics configuration
    pub analytics_config: MarketplaceAnalyticsConfig,
    /// Security and access control configuration
    pub security_config: MarketplaceSecurityConfig,
    /// Machine learning and AI configuration
    pub ml_config: MarketplaceMLConfig,
    /// Integration configuration
    pub integration_config: MarketplaceIntegrationConfig,
    /// Economic and incentive configuration
    pub economic_config: MarketplaceEconomicConfig,
}

// Forward declarations for types that will be defined in other modules
use super::{
    analytics::MarketplaceAnalyticsConfig, benchmarking::BenchmarkingConfig,
    collaboration::CollaborationConfig, discovery::AlgorithmDiscoveryConfig,
    economic::MarketplaceEconomicConfig, integration::MarketplaceIntegrationConfig,
    ml_integration::MarketplaceMLConfig, optimization::MarketplaceOptimizationConfig,
    registry::AlgorithmRegistryConfig, security::MarketplaceSecurityConfig,
};
