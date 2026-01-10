//! Advanced Provider Capability Discovery System
//!
//! This module provides comprehensive discovery, analysis, and management of quantum
//! computing provider capabilities. Features include real-time capability discovery,
//! comparative analysis, performance benchmarking, and intelligent provider selection
//! with SciRS2-powered analytics.

mod capabilities;
mod config;
mod engines;
mod events;
mod system;
mod types;

// Re-export all public types
pub use capabilities::*;
pub use config::*;
pub use engines::*;
pub use events::*;
pub use system::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_discovery_config_default() {
        let config = DiscoveryConfig::default();
        assert!(config.enable_auto_discovery);
        assert_eq!(config.discovery_interval, 3600);
        assert!(config.enable_caching);
        assert!(config.enable_monitoring);
        assert!(config.enable_analytics);
    }

    #[test]
    fn test_provider_info_creation() {
        let provider = ProviderInfo {
            provider_id: "test_provider".to_string(),
            name: "Test Provider".to_string(),
            description: "A test quantum provider".to_string(),
            provider_type: ProviderType::CloudProvider,
            contact_info: ContactInfo {
                support_email: Some("support@test.com".to_string()),
                support_phone: None,
                support_website: None,
                technical_contact: None,
                business_contact: None,
                emergency_contact: None,
            },
            endpoints: Vec::new(),
            supported_regions: vec!["us-east-1".to_string()],
            pricing_model: PricingModel {
                pricing_type: PricingType::PayPerUse,
                cost_per_shot: Some(0.01),
                cost_per_circuit: None,
                cost_per_hour: None,
                monthly_subscription: None,
                free_tier: None,
                currency: "USD".to_string(),
                billing_model: BillingModel::Postpaid,
            },
            terms_of_service: None,
            privacy_policy: None,
            compliance_certifications: Vec::new(),
            last_updated: std::time::SystemTime::now(),
        };

        assert_eq!(provider.provider_id, "test_provider");
        assert_eq!(provider.provider_type, ProviderType::CloudProvider);
    }

    #[test]
    fn test_capability_requirements() {
        let requirements = CapabilityRequirements {
            min_qubits: Some(5),
            max_error_rate: Some(0.01),
            required_gates: ["H", "CNOT"].iter().map(|s| s.to_string()).collect(),
            required_connectivity: Some(ConnectivityRequirement::FullyConnected),
            required_features: [ProviderFeature::QuantumComputing]
                .iter()
                .cloned()
                .collect(),
            performance_requirements: PerformanceRequirements {
                max_execution_time: Some(Duration::from_secs(300)),
                min_throughput: Some(100.0),
                max_queue_time: Some(Duration::from_secs(60)),
                min_availability: Some(0.99),
                max_cost_per_shot: Some(0.005),
            },
        };

        assert_eq!(requirements.min_qubits, Some(5));
        assert_eq!(requirements.max_error_rate, Some(0.01));
        assert!(requirements
            .required_features
            .contains(&ProviderFeature::QuantumComputing));
    }

    #[test]
    fn test_discovery_system_creation() {
        let config = DiscoveryConfig::default();
        let _system = ProviderCapabilityDiscoverySystem::new(config);
        // System should be created successfully
    }

    #[test]
    fn test_high_performance_config() {
        let config = create_high_performance_discovery_config();
        assert_eq!(config.discovery_interval, 1800);
        assert_eq!(
            config.analytics_config.analysis_depth,
            AnalysisDepth::Comprehensive
        );
        assert_eq!(config.verification_config.min_verification_confidence, 0.9);
    }

    #[tokio::test]
    async fn test_discovery_system_start() {
        let config = DiscoveryConfig::default();
        let system = ProviderCapabilityDiscoverySystem::new(config);

        let start_result = system.start().await;
        assert!(start_result.is_ok());
    }

    #[test]
    fn test_verification_config() {
        let config = VerificationConfig {
            enable_verification: true,
            verification_timeout: Duration::from_secs(300),
            verification_strategies: vec![
                VerificationStrategy::EndpointTesting,
                VerificationStrategy::CapabilityProbing,
            ],
            min_verification_confidence: 0.8,
            enable_continuous_verification: true,
            verification_frequency: Duration::from_secs(86400),
        };

        assert!(config.enable_verification);
        assert_eq!(config.verification_strategies.len(), 2);
    }

    #[test]
    fn test_quality_thresholds() {
        let thresholds = QualityThresholds {
            min_fidelity: 0.9,
            max_error_rate: 0.05,
            min_uptime: 0.98,
            min_reliability: 0.95,
            min_performance: 0.8,
        };

        assert!(thresholds.min_fidelity > 0.8);
        assert!(thresholds.max_error_rate < 0.1);
    }

    #[test]
    fn test_comparison_config() {
        let config = ComparisonConfig {
            enable_auto_comparison: true,
            comparison_criteria: vec![
                ComparisonCriterion::Performance,
                ComparisonCriterion::Cost,
                ComparisonCriterion::Reliability,
            ],
            ranking_algorithms: vec![RankingAlgorithm::WeightedSum, RankingAlgorithm::TOPSIS],
            criterion_weights: std::collections::HashMap::new(),
            enable_multidimensional_analysis: true,
        };

        assert!(config.enable_auto_comparison);
        assert_eq!(config.comparison_criteria.len(), 3);
        assert_eq!(config.ranking_algorithms.len(), 2);
    }

    #[test]
    fn test_provider_feature_equality() {
        let feature1 = ProviderFeature::QuantumComputing;
        let feature2 = ProviderFeature::QuantumComputing;
        let feature3 = ProviderFeature::NoiseModeling;

        assert_eq!(feature1, feature2);
        assert_ne!(feature1, feature3);
    }

    #[test]
    fn test_discovery_strategy_variants() {
        let strategies = vec![
            DiscoveryStrategy::APIDiscovery,
            DiscoveryStrategy::RegistryDiscovery,
            DiscoveryStrategy::NetworkDiscovery,
            DiscoveryStrategy::ConfigurationDiscovery,
            DiscoveryStrategy::MLEnhancedDiscovery,
            DiscoveryStrategy::HybridDiscovery,
        ];

        assert_eq!(strategies.len(), 6);
    }

    #[test]
    fn test_health_level_ordering() {
        // Test that health levels can be compared
        let excellent = HealthLevel::Excellent;
        let good = HealthLevel::Good;

        assert_ne!(excellent, good);
    }

    #[test]
    fn test_issue_severity() {
        let severities = vec![
            IssueSeverity::Low,
            IssueSeverity::Medium,
            IssueSeverity::High,
            IssueSeverity::Critical,
        ];

        assert_eq!(severities.len(), 4);
    }

    #[test]
    fn test_monitoring_target_type() {
        let types = vec![
            MonitoringTargetType::Provider,
            MonitoringTargetType::Endpoint,
            MonitoringTargetType::Service,
            MonitoringTargetType::Capability,
            MonitoringTargetType::Performance,
            MonitoringTargetType::Cost,
            MonitoringTargetType::Security,
        ];

        assert_eq!(types.len(), 7);
    }

    #[test]
    fn test_trend_model_type() {
        let model_types = vec![
            TrendModelType::Linear,
            TrendModelType::Exponential,
            TrendModelType::Polynomial,
            TrendModelType::Seasonal,
            TrendModelType::ARIMA,
            TrendModelType::MachineLearning,
        ];

        assert_eq!(model_types.len(), 6);
    }

    #[test]
    fn test_predictive_model_type() {
        let model_types = vec![
            PredictiveModelType::LinearRegression,
            PredictiveModelType::RandomForest,
            PredictiveModelType::NeuralNetwork,
            PredictiveModelType::SVM,
            PredictiveModelType::DecisionTree,
            PredictiveModelType::Ensemble,
        ];

        assert_eq!(model_types.len(), 6);
    }
}
