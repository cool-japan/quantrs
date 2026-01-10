//! Configuration types for provider capability discovery.
//!
//! This module contains all configuration structs and enums for the
//! provider capability discovery system.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::types::{ProviderFeature, TopologyType};

/// Configuration for capability discovery system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enable automatic discovery
    pub enable_auto_discovery: bool,
    /// Discovery interval in seconds
    pub discovery_interval: u64,
    /// Enable capability caching
    pub enable_caching: bool,
    /// Cache expiration time
    pub cache_expiration: Duration,
    /// Enable real-time monitoring
    pub enable_monitoring: bool,
    /// Enable analytics
    pub enable_analytics: bool,
    /// Discovery strategies
    pub discovery_strategies: Vec<DiscoveryStrategy>,
    /// Capability verification settings
    pub verification_config: VerificationConfig,
    /// Provider filtering settings
    pub filtering_config: FilteringConfig,
    /// Analytics configuration
    pub analytics_config: CapabilityAnalyticsConfig,
    /// Monitoring configuration
    pub monitoring_config: CapabilityMonitoringConfig,
    /// Comparison configuration
    pub comparison_config: ComparisonConfig,
}

/// Discovery strategies for finding providers
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryStrategy {
    /// API-based discovery
    APIDiscovery,
    /// Registry-based discovery
    RegistryDiscovery,
    /// Network-based discovery
    NetworkDiscovery,
    /// Configuration-based discovery
    ConfigurationDiscovery,
    /// Machine learning-enhanced discovery
    MLEnhancedDiscovery,
    /// Hybrid multi-strategy discovery
    HybridDiscovery,
}

/// Verification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Enable capability verification
    pub enable_verification: bool,
    /// Verification timeout
    pub verification_timeout: Duration,
    /// Verification strategies
    pub verification_strategies: Vec<VerificationStrategy>,
    /// Required verification confidence
    pub min_verification_confidence: f64,
    /// Enable continuous verification
    pub enable_continuous_verification: bool,
    /// Verification frequency
    pub verification_frequency: Duration,
}

/// Verification strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStrategy {
    /// API endpoint testing
    EndpointTesting,
    /// Capability probing
    CapabilityProbing,
    /// Benchmark testing
    BenchmarkTesting,
    /// Historical analysis
    HistoricalAnalysis,
    /// Community validation
    CommunityValidation,
}

/// Provider filtering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilteringConfig {
    /// Enable provider filtering
    pub enable_filtering: bool,
    /// Minimum capability requirements
    pub min_requirements: CapabilityRequirements,
    /// Excluded providers
    pub excluded_providers: HashSet<String>,
    /// Preferred providers
    pub preferred_providers: Vec<String>,
    /// Quality thresholds
    pub quality_thresholds: QualityThresholds,
    /// Geographic restrictions
    pub geographic_restrictions: Option<GeographicRestrictions>,
}

/// Capability requirements for filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityRequirements {
    /// Minimum number of qubits
    pub min_qubits: Option<usize>,
    /// Maximum error rate
    pub max_error_rate: Option<f64>,
    /// Required gate types
    pub required_gates: HashSet<String>,
    /// Required connectivity
    pub required_connectivity: Option<ConnectivityRequirement>,
    /// Required features
    pub required_features: HashSet<ProviderFeature>,
    /// Performance requirements
    pub performance_requirements: PerformanceRequirements,
}

/// Connectivity requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityRequirement {
    /// Full connectivity required
    FullyConnected,
    /// Minimum connectivity degree
    MinimumDegree(usize),
    /// Specific topology required
    SpecificTopology(TopologyType),
    /// Custom connectivity pattern
    CustomPattern(Vec<(usize, usize)>),
}

/// Performance requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    /// Maximum execution time
    pub max_execution_time: Option<Duration>,
    /// Minimum throughput (circuits/hour)
    pub min_throughput: Option<f64>,
    /// Maximum queue time
    pub max_queue_time: Option<Duration>,
    /// Minimum availability
    pub min_availability: Option<f64>,
    /// Maximum cost per shot
    pub max_cost_per_shot: Option<f64>,
}

/// Quality thresholds for filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum fidelity
    pub min_fidelity: f64,
    /// Maximum error rate
    pub max_error_rate: f64,
    /// Minimum uptime
    pub min_uptime: f64,
    /// Minimum reliability score
    pub min_reliability: f64,
    /// Minimum performance score
    pub min_performance: f64,
}

/// Geographic restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicRestrictions {
    /// Allowed regions
    pub allowed_regions: HashSet<String>,
    /// Blocked regions
    pub blocked_regions: HashSet<String>,
    /// Data sovereignty requirements
    pub data_sovereignty: bool,
    /// Compliance requirements
    pub compliance_requirements: Vec<ComplianceStandard>,
}

/// Compliance standards
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStandard {
    GDPR,
    HIPAA,
    SOC2,
    ISO27001,
    FedRAMP,
    Custom(String),
}

/// Analytics configuration for capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityAnalyticsConfig {
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    /// Enable predictive analytics
    pub enable_predictive_analytics: bool,
    /// Enable comparative analysis
    pub enable_comparative_analysis: bool,
    /// Analysis depth
    pub analysis_depth: AnalysisDepth,
    /// Historical data retention
    pub retention_period: Duration,
    /// Statistical confidence level
    pub confidence_level: f64,
}

/// Analysis depth levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisDepth {
    Basic,
    Standard,
    Advanced,
    Comprehensive,
}

/// Monitoring configuration for capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityMonitoringConfig {
    /// Enable real-time monitoring
    pub enable_realtime_monitoring: bool,
    /// Monitoring frequency
    pub monitoring_frequency: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Anomaly sensitivity
    pub anomaly_sensitivity: f64,
}

/// Comparison configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonConfig {
    /// Enable automatic comparison
    pub enable_auto_comparison: bool,
    /// Comparison criteria
    pub comparison_criteria: Vec<ComparisonCriterion>,
    /// Ranking algorithms
    pub ranking_algorithms: Vec<RankingAlgorithm>,
    /// Weight distribution
    pub criterion_weights: HashMap<String, f64>,
    /// Enable multi-dimensional analysis
    pub enable_multidimensional_analysis: bool,
}

/// Comparison criteria
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonCriterion {
    Performance,
    Cost,
    Reliability,
    Availability,
    Features,
    Security,
    Compliance,
    Support,
    Innovation,
    Custom(String),
}

/// Ranking algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankingAlgorithm {
    WeightedSum,
    TOPSIS,
    AHP, // Analytic Hierarchy Process
    ELECTRE,
    PROMETHEE,
    MachineLearning,
    Custom(String),
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enable_auto_discovery: true,
            discovery_interval: 3600, // 1 hour
            enable_caching: true,
            cache_expiration: Duration::from_secs(86400), // 24 hours
            enable_monitoring: true,
            enable_analytics: true,
            discovery_strategies: vec![
                DiscoveryStrategy::APIDiscovery,
                DiscoveryStrategy::RegistryDiscovery,
            ],
            verification_config: VerificationConfig {
                enable_verification: true,
                verification_timeout: Duration::from_secs(300),
                verification_strategies: vec![
                    VerificationStrategy::EndpointTesting,
                    VerificationStrategy::CapabilityProbing,
                ],
                min_verification_confidence: 0.8,
                enable_continuous_verification: true,
                verification_frequency: Duration::from_secs(86400),
            },
            filtering_config: FilteringConfig {
                enable_filtering: true,
                min_requirements: CapabilityRequirements {
                    min_qubits: Some(2),
                    max_error_rate: Some(0.1),
                    required_gates: HashSet::new(),
                    required_connectivity: None,
                    required_features: HashSet::new(),
                    performance_requirements: PerformanceRequirements {
                        max_execution_time: None,
                        min_throughput: None,
                        max_queue_time: None,
                        min_availability: Some(0.9),
                        max_cost_per_shot: None,
                    },
                },
                excluded_providers: HashSet::new(),
                preferred_providers: Vec::new(),
                quality_thresholds: QualityThresholds {
                    min_fidelity: 0.8,
                    max_error_rate: 0.1,
                    min_uptime: 0.95,
                    min_reliability: 0.9,
                    min_performance: 0.7,
                },
                geographic_restrictions: None,
            },
            analytics_config: CapabilityAnalyticsConfig {
                enable_trend_analysis: true,
                enable_predictive_analytics: true,
                enable_comparative_analysis: true,
                analysis_depth: AnalysisDepth::Standard,
                retention_period: Duration::from_secs(30 * 86400), // 30 days
                confidence_level: 0.95,
            },
            monitoring_config: CapabilityMonitoringConfig {
                enable_realtime_monitoring: true,
                monitoring_frequency: Duration::from_secs(300), // 5 minutes
                health_check_interval: Duration::from_secs(600), // 10 minutes
                alert_thresholds: HashMap::new(),
                enable_anomaly_detection: true,
                anomaly_sensitivity: 0.8,
            },
            comparison_config: ComparisonConfig {
                enable_auto_comparison: true,
                comparison_criteria: vec![
                    ComparisonCriterion::Performance,
                    ComparisonCriterion::Cost,
                    ComparisonCriterion::Reliability,
                ],
                ranking_algorithms: vec![RankingAlgorithm::WeightedSum],
                criterion_weights: HashMap::new(),
                enable_multidimensional_analysis: true,
            },
        }
    }
}
