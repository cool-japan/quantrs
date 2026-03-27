//! # CostOptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `CostOptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use super::types::{
    AggregationStrategy, AlertAggregationConfig, AlertCondition, AlertSeverity, BudgetConfig,
    BudgetRolloverPolicy, CircuitFeature, ComparisonMetric, ConvergenceCriteria, CostAlertConfig,
    CostAlertRule, CostEstimationConfig, CostMonitoringConfig, CostOptimizationConfig,
    CostOptimizationStrategy, CostReportingConfig, DashboardConfig, DashboardWidget,
    FeatureEngineeringConfig, FeatureSelectionMethod, MonitoringMetric, MultiObjectiveConfig,
    NormalizationMethod, NotificationFrequency, OptimizationObjective, ParetoConfig,
    PredictiveModelType, PredictiveModelingConfig, ProviderComparisonConfig, ProviderFeature,
    ReportFormat, ReportType, ResourceOptimizationAlgorithm, ResourceOptimizationConfig,
    SolutionSelectionStrategy, TimeFeature, UsageFeature,
};

impl Default for CostOptimizationConfig {
    fn default() -> Self {
        Self {
            budget_config: BudgetConfig {
                total_budget: 10000.0,
                daily_budget: Some(100.0),
                monthly_budget: Some(3000.0),
                provider_budgets: HashMap::new(),
                circuit_type_budgets: HashMap::new(),
                auto_budget_management: true,
                rollover_policy: BudgetRolloverPolicy::PercentageRollover(0.2),
            },
            estimation_config: CostEstimationConfig {
                provider_models: HashMap::new(),
                include_queue_time: true,
                include_overhead_costs: true,
                accuracy_target: 0.9,
                model_update_frequency: Duration::from_secs(3600),
                enable_ml_estimation: true,
                data_retention_period: Duration::from_secs(30 * 24 * 3600),
            },
            optimization_strategy: CostOptimizationStrategy::MaximizeCostPerformance,
            provider_comparison: ProviderComparisonConfig {
                comparison_metrics: vec![
                    ComparisonMetric::TotalCost,
                    ComparisonMetric::QueueTime,
                    ComparisonMetric::Fidelity,
                ],
                normalization_method: NormalizationMethod::MinMax,
                metric_weights: HashMap::new(),
                real_time_comparison: true,
                update_frequency: Duration::from_secs(300),
                include_reliability: true,
            },
            predictive_modeling: PredictiveModelingConfig {
                enabled: true,
                prediction_horizon: Duration::from_secs(24 * 3600),
                model_types: vec![
                    PredictiveModelType::RandomForest,
                    PredictiveModelType::SciRS2Enhanced,
                ],
                feature_engineering: FeatureEngineeringConfig {
                    time_features: vec![TimeFeature::HourOfDay, TimeFeature::DayOfWeek],
                    circuit_features: vec![CircuitFeature::QubitCount, CircuitFeature::GateCount],
                    provider_features: vec![
                        ProviderFeature::QueueLength,
                        ProviderFeature::SystemLoad,
                    ],
                    usage_features: vec![
                        UsageFeature::HistoricalCosts,
                        UsageFeature::UsagePatterns,
                    ],
                    feature_selection: FeatureSelectionMethod::Correlation(0.1),
                },
                training_frequency: Duration::from_secs(24 * 3600),
                confidence_threshold: 0.8,
                enable_ensemble: true,
            },
            resource_optimization: ResourceOptimizationConfig {
                enabled: true,
                algorithms: vec![ResourceOptimizationAlgorithm::SciRS2Optimization],
                constraints: vec![],
                optimization_frequency: Duration::from_secs(3600),
                enable_parallel_optimization: true,
                multi_objective_config: MultiObjectiveConfig {
                    objectives: vec![
                        OptimizationObjective::MinimizeCost,
                        OptimizationObjective::MaximizeQuality,
                    ],
                    pareto_config: ParetoConfig {
                        max_solutions: 100,
                        convergence_criteria: ConvergenceCriteria {
                            max_iterations: 1000,
                            tolerance: 1e-6,
                            patience: 50,
                        },
                        diversity_preservation: true,
                    },
                    selection_strategy: SolutionSelectionStrategy::Weighted(
                        [
                            (OptimizationObjective::MinimizeCost, 0.6),
                            (OptimizationObjective::MaximizeQuality, 0.4),
                        ]
                        .iter()
                        .cloned()
                        .collect(),
                    ),
                },
            },
            monitoring_config: CostMonitoringConfig {
                real_time_monitoring: true,
                monitoring_frequency: Duration::from_secs(60),
                tracked_metrics: vec![
                    MonitoringMetric::TotalCost,
                    MonitoringMetric::BudgetUtilization,
                    MonitoringMetric::CostEfficiency,
                ],
                reporting_config: CostReportingConfig {
                    automated_reports: true,
                    report_frequency: Duration::from_secs(24 * 3600),
                    report_types: vec![ReportType::CostSummary, ReportType::BudgetAnalysis],
                    recipients: vec![],
                    format: ReportFormat::JSON,
                },
                dashboard_config: Some(DashboardConfig {
                    enabled: true,
                    update_frequency: Duration::from_secs(30),
                    widgets: vec![
                        DashboardWidget::CostGauge,
                        DashboardWidget::BudgetProgress,
                        DashboardWidget::ProviderComparison,
                    ],
                    custom_visualizations: vec![],
                }),
            },
            alert_config: CostAlertConfig {
                enabled: true,
                alert_rules: vec![CostAlertRule {
                    name: "Budget threshold".to_string(),
                    condition: AlertCondition::BudgetThreshold {
                        threshold: 80.0,
                        percentage: true,
                    },
                    severity: AlertSeverity::Warning,
                    frequency: NotificationFrequency::Immediate,
                    enabled: true,
                }],
                notification_channels: vec![],
                aggregation_config: AlertAggregationConfig {
                    enabled: true,
                    window: Duration::from_secs(300),
                    max_alerts_per_window: 5,
                    strategy: AggregationStrategy::SeverityBased,
                },
            },
        }
    }
}
