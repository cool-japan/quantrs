//! Auto-generated module - profiling
//!
//! 🤖 Generated with split_types_final.py

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock as TokioRwLock;
use uuid::Uuid;

use super::super::super::super::{DeviceError, DeviceResult, QuantumDevice};
use super::super::super::{CloudProvider, QuantumCloudConfig};
use crate::algorithm_marketplace::{ScalingBehavior, ValidationResult};
use crate::prelude::DeploymentStatus;

// Import traits from parent module
use super::super::traits::{
    ClusteringEngine, FeatureExtractor, LearningAlgorithm, NearestNeighborEngine,
    PatternAnalysisAlgorithm, RecommendationAlgorithm, SimilarityMetric,
};

// Cross-module imports from sibling modules
use super::{cost::*, execution::*, optimization::*, providers::*, tracking::*, workload::*};

#[derive(Debug, Clone)]
pub struct DependencyEdge {
    pub source: String,
    pub target: String,
    pub dependency_type: DependencyType,
    pub data_volume: usize,
}

#[derive(Debug, Clone)]
pub struct ClusterQuality {
    pub silhouette_score: f64,
    pub davies_bouldin_index: f64,
    pub calinski_harabasz_index: f64,
    pub inertia: f64,
}

#[derive(Debug, Clone)]
pub struct KnowledgeBase {
    best_practices: Vec<BestPractice>,
    optimization_rules: Vec<OptimizationRule>,
    performance_models: HashMap<String, PerformanceModel>,
    case_studies: Vec<CaseStudy>,
}
impl Default for KnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeBase {
    pub fn new() -> Self {
        Self {
            best_practices: Vec::new(),
            optimization_rules: Vec::new(),
            performance_models: HashMap::new(),
            case_studies: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum KnowledgeImprovementType {
    NewBestPractice,
    UpdatedBestPractice,
    NewCaseStudy,
    RefinedGuidelines,
    ImprovedModels,
}

#[derive(Debug, Clone)]
pub struct RegressionAnalysis {
    pub model_type: String,
    pub coefficients: Vec<f64>,
    pub r_squared: f64,
    pub adjusted_r_squared: f64,
    pub residual_analysis: ResidualAnalysis,
}

#[derive(Debug, Clone)]
pub struct DependencyGraph {
    pub nodes: Vec<DependencyNode>,
    pub edges: Vec<DependencyEdge>,
    pub cycles: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct ClusterCharacteristics {
    pub dominant_workload_type: WorkloadType,
    pub average_characteristics: WorkloadCharacteristics,
    pub performance_profile: ClusterPerformanceProfile,
    pub optimization_recommendations: Vec<ClusterOptimizationRecommendation>,
}

#[derive(Debug, Clone)]
pub enum AccessPattern {
    Sequential,
    Random,
    Strided,
    Clustered,
    Temporal,
}

#[derive(Debug, Clone)]
pub enum SeasonalType {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annual,
}

#[derive(Debug, Clone)]
pub struct KnowledgeImprovement {
    pub improvement_type: KnowledgeImprovementType,
    pub description: String,
    pub evidence_strength: f64,
    pub impact_assessment: f64,
}

#[derive(Debug, Clone)]
pub enum DistributionType {
    Gaussian,
    Beta,
    Gamma,
    Uniform,
    Multimodal,
    Skewed,
}

pub struct LearningEngine {
    learning_algorithms: Vec<Box<dyn LearningAlgorithm + Send + Sync>>,
    feedback_processor: FeedbackProcessor,
    model_updater: ModelUpdater,
    continuous_learning: bool,
}
impl Default for LearningEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningEngine {
    pub fn new() -> Self {
        Self {
            learning_algorithms: Vec::new(),
            feedback_processor: FeedbackProcessor::new(),
            model_updater: ModelUpdater::new(),
            continuous_learning: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimilarityExplanation {
    pub primary_similarities: Vec<String>,
    pub key_differences: Vec<String>,
    pub similarity_breakdown: HashMap<String, f64>,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct DistributedScalingCharacteristics {
    pub network_communication: NetworkCommunicationPattern,
    pub data_locality: DataLocalityPattern,
    pub fault_tolerance: FaultTolerancePattern,
}

#[derive(Debug, Clone)]
pub struct LearningResult {
    pub model_updates: Vec<ModelUpdate>,
    pub new_patterns: Vec<IdentifiedPattern>,
    pub rule_refinements: Vec<RuleRefinement>,
    pub knowledge_improvements: Vec<KnowledgeImprovement>,
}

pub struct PatternAnalyzer {
    analysis_algorithms: Vec<Box<dyn PatternAnalysisAlgorithm + Send + Sync>>,
    feature_extractors: Vec<Box<dyn FeatureExtractor + Send + Sync>>,
    clustering_engines: Vec<Box<dyn ClusteringEngine + Send + Sync>>,
}
impl Default for PatternAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternAnalyzer {
    pub fn new() -> Self {
        Self {
            analysis_algorithms: Vec::new(),
            feature_extractors: Vec::new(),
            clustering_engines: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum DataLocalityPattern {
    HighLocality,
    MediumLocality,
    LowLocality,
    NoLocality,
}

#[derive(Debug, Clone)]
pub struct ComparisonData {
    provider_comparisons: HashMap<(CloudProvider, CloudProvider), ProviderComparison>,
    temporal_trends: HashMap<CloudProvider, TemporalTrend>,
    cost_performance_analysis: CostPerformanceAnalysis,
}
impl Default for ComparisonData {
    fn default() -> Self {
        Self::new()
    }
}

impl ComparisonData {
    pub fn new() -> Self {
        Self {
            provider_comparisons: HashMap::new(),
            temporal_trends: HashMap::new(),
            cost_performance_analysis: CostPerformanceAnalysis::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BestPractice {
    pub practice_id: String,
    pub practice_name: String,
    pub description: String,
    pub applicable_contexts: Vec<String>,
    pub expected_benefits: Vec<String>,
    pub implementation_guidance: String,
    pub evidence_quality: f64,
}

#[derive(Debug, Clone)]
pub struct ScalabilityCharacteristics {
    pub problem_size_scaling: ScalingBehavior,
    pub resource_scaling: ResourceScalingCharacteristics,
    pub parallel_scaling: ParallelScalingCharacteristics,
    pub distributed_scaling: DistributedScalingCharacteristics,
}

#[derive(Debug, Clone)]
pub struct SimilarWorkload {
    pub workload_profile: WorkloadProfile,
    pub similarity_score: f64,
    pub similarity_explanation: SimilarityExplanation,
}

pub struct WorkloadProfiler {
    workload_profiles: HashMap<String, WorkloadProfile>,
    pattern_analyzer: PatternAnalyzer,
    similarity_engine: SimilarityEngine,
    recommendation_engine: RecommendationEngine,
}
impl WorkloadProfiler {
    pub fn new() -> DeviceResult<Self> {
        Ok(Self {
            workload_profiles: HashMap::new(),
            pattern_analyzer: PatternAnalyzer::new(),
            similarity_engine: SimilarityEngine::new(),
            recommendation_engine: RecommendationEngine::new(),
        })
    }
    /// Profile a workload by measuring its key structural and resource
    /// characteristics.
    ///
    /// The profiler derives:
    /// - **Circuit depth / gate count / qubit utilisation** directly from
    ///   `WorkloadSpec::circuit_characteristics`.
    /// - **Cost patterns** using a simple fixed-cost + per-gate variable cost
    ///   model (provider-agnostic estimates).
    /// - **Temporal patterns** seeded with neutral/stable defaults until real
    ///   historical data is available through the learning pipeline.
    /// - A **profile_id** that encodes the workload signature so downstream
    ///   caches can detect identical profiles.
    pub async fn profile_workload(&self, workload: &WorkloadSpec) -> DeviceResult<WorkloadProfile> {
        use std::time::{Duration, SystemTime};

        let cc = &workload.circuit_characteristics;
        let er = &workload.execution_requirements;

        // ----------------------------------------------------------------
        // Workload characteristics
        // ----------------------------------------------------------------
        let qubit_utilisation = cc.qubit_count as f64 / 128_f64; // normalise against 128-qubit reference

        let workload_characteristics = WorkloadCharacteristics {
            computational_complexity: ComputationalComplexity {
                time_complexity: ComplexityClass::Quadratic,
                space_complexity: ComplexityClass::Exponential,
                quantum_complexity: QuantumComplexityClass::BQP,
                parallel_complexity: ParallelComplexityClass::P,
            },
            data_characteristics: DataCharacteristics {
                data_size: DataSize {
                    input_size: cc.qubit_count,
                    intermediate_size: cc.qubit_count * cc.circuit_depth,
                    output_size: cc.qubit_count,
                    memory_footprint: (1_usize << cc.qubit_count.min(30)) * 16, // bytes
                },
                data_structure: DataStructure::Vector,
                data_access_patterns: DataAccessPatterns {
                    access_pattern: AccessPattern::Sequential,
                    locality: LocalityPattern::Spatial,
                    caching_behavior: CachingBehavior::Medium,
                },
                data_dependencies: DataDependencies {
                    dependency_graph: DependencyGraph {
                        nodes: Vec::new(),
                        edges: Vec::new(),
                        cycles: Vec::new(),
                    },
                    critical_path: Vec::new(),
                    parallelization_potential: 0.60,
                },
            },
            algorithmic_properties: AlgorithmicProperties {
                algorithm_family: AlgorithmFamily::Simulation,
                optimization_landscape: OptimizationLandscape {
                    landscape_type: LandscapeType::Multimodal,
                    local_minima_density: 0.30,
                    barrier_heights: vec![0.1, 0.3, 0.5],
                    global_structure: GlobalStructure::Archipelago,
                },
                convergence_properties: ConvergenceProperties {
                    convergence_rate: ConvergenceRate::Linear,
                    convergence_criteria: vec![ConvergenceCriterion::AbsoluteTolerance],
                    stability: StabilityProperties {
                        numerical_stability: 0.95,
                        noise_tolerance: cc.noise_tolerance,
                        parameter_sensitivity: 0.40,
                        robustness_score: 0.75,
                    },
                },
                noise_sensitivity: NoiseSensitivity {
                    gate_error_sensitivity: 1.0 - cc.noise_tolerance,
                    decoherence_sensitivity: 0.70,
                    measurement_error_sensitivity: 0.50,
                    classical_noise_sensitivity: 0.10,
                },
            },
            scalability_characteristics: ScalabilityCharacteristics {
                problem_size_scaling: ScalingBehavior::Exponential,
                resource_scaling: ResourceScalingCharacteristics {
                    memory_scaling: ScalingBehavior::Exponential,
                    compute_scaling: ScalingBehavior::Quadratic,
                    quantum_resource_scaling: ScalingBehavior::Linear,
                    communication_scaling: ScalingBehavior::Linear,
                },
                parallel_scaling: ParallelScalingCharacteristics {
                    maximum_parallelism: cc.qubit_count,
                    parallel_efficiency: 0.70,
                    load_balance_quality: 0.80,
                    synchronization_overhead: 0.15,
                },
                distributed_scaling: DistributedScalingCharacteristics {
                    network_communication: NetworkCommunicationPattern::Sparse,
                    data_locality: DataLocalityPattern::HighLocality,
                    fault_tolerance: FaultTolerancePattern::ErrorCorrection,
                },
            },
        };

        // ----------------------------------------------------------------
        // Resource patterns
        // ----------------------------------------------------------------
        let neutral_utilisation = UtilizationPattern {
            average_utilization: qubit_utilisation,
            peak_utilization: (qubit_utilisation * 1.30).min(1.0),
            utilization_variance: 0.10,
            temporal_pattern: TemporalUtilizationPattern::Constant,
        };

        let resource_patterns = ResourcePatterns {
            cpu_utilization_pattern: UtilizationPattern {
                average_utilization: 0.05,
                peak_utilization: 0.20,
                utilization_variance: 0.05,
                temporal_pattern: TemporalUtilizationPattern::Bursty,
            },
            memory_utilization_pattern: UtilizationPattern {
                average_utilization: 0.10,
                peak_utilization: 0.30,
                utilization_variance: 0.05,
                temporal_pattern: TemporalUtilizationPattern::Increasing,
            },
            network_utilization_pattern: UtilizationPattern {
                average_utilization: 0.02,
                peak_utilization: 0.10,
                utilization_variance: 0.02,
                temporal_pattern: TemporalUtilizationPattern::Bursty,
            },
            quantum_resource_pattern: QuantumResourcePattern {
                qubit_utilization: qubit_utilisation,
                gate_distribution: cc
                    .gate_types
                    .iter()
                    .map(|(k, v)| (k.clone(), *v as f64 / cc.gate_count.max(1) as f64))
                    .collect(),
                entanglement_pattern: EntanglementPattern::Clustered,
                measurement_pattern: MeasurementPattern::Final,
            },
        };

        // ----------------------------------------------------------------
        // Performance patterns
        // ----------------------------------------------------------------
        let shots_per_ms = 1.0; // 1 000 shots / s default estimate
        let avg_exec_ms = er.shots as f64 / shots_per_ms;

        let stable_throughput = ThroughputPattern {
            average_throughput: shots_per_ms,
            peak_throughput: shots_per_ms * 2.0,
            throughput_stability: 0.85,
            bottleneck_analysis: BottleneckAnalysis {
                primary_bottleneck: BottleneckType::Quantum,
                bottleneck_severity: 0.55,
                bottleneck_variability: 0.20,
                mitigation_strategies: vec![
                    "Increase shot batching".to_string(),
                    "Use faster hardware backend".to_string(),
                ],
            },
        };

        let performance_patterns = PerformancePatterns {
            execution_time_pattern: ExecutionTimePattern {
                average_time: Duration::from_millis(avg_exec_ms as u64),
                time_variance: Duration::from_millis((avg_exec_ms * 0.20) as u64),
                time_distribution: TimeDistribution::LogNormal,
                scaling_behavior: ScalingBehavior::Linear,
            },
            throughput_pattern: stable_throughput,
            quality_pattern: QualityPattern {
                fidelity_distribution: QualityDistribution {
                    mean_fidelity: 1.0 - cc.noise_tolerance,
                    fidelity_variance: 0.02,
                    distribution_type: DistributionType::Gaussian,
                    outlier_frequency: 0.05,
                },
                error_correlation: ErrorCorrelation {
                    temporal_correlation: 0.20,
                    spatial_correlation: 0.30,
                    systematic_errors: 0.15,
                    random_errors: 0.85,
                },
                quality_degradation: QualityDegradation {
                    degradation_rate: 0.001,
                    degradation_factors: vec![DegradationFactor::GateErrors],
                    mitigation_effectiveness: 0.70,
                },
            },
            reliability_pattern: ReliabilityPattern {
                success_rate: 0.95,
                failure_modes: Vec::new(),
                recovery_patterns: RecoveryPatterns {
                    automatic_recovery_rate: 0.80,
                    manual_intervention_rate: 0.10,
                    recovery_time_distribution: TimeDistribution::Exponential,
                    recovery_strategies: vec![RecoveryStrategy::Restart],
                },
                maintenance_requirements: MaintenanceRequirements {
                    preventive_maintenance_frequency: Duration::from_secs(86400),
                    corrective_maintenance_frequency: Duration::from_secs(3600),
                    maintenance_duration: Duration::from_secs(300),
                    maintenance_cost: 50.0,
                },
            },
        };

        // ----------------------------------------------------------------
        // Cost patterns (provider-agnostic estimate)
        // ----------------------------------------------------------------
        let fixed_cost = 0.30; // per-task base fee (USD)
        let variable_cost = er.shots as f64 * 0.00035; // per-shot mid-range estimate
        let total_estimated_cost = fixed_cost + variable_cost;

        let cost_patterns = CostPatterns {
            cost_structure: WorkloadCostStructure {
                fixed_costs: fixed_cost,
                variable_costs: variable_cost,
                marginal_costs: 0.00035,
                cost_drivers: vec![CostDriver {
                    driver_name: "Shot count".to_string(),
                    cost_impact: variable_cost / total_estimated_cost,
                    variability: 0.05,
                    optimization_potential: 0.20,
                }],
            },
            cost_variability: CostVariability {
                cost_variance: total_estimated_cost * 0.15,
                cost_predictability: 0.85,
                cost_volatility: 0.10,
                external_factors: Vec::new(),
            },
            cost_optimization_potential: CostOptimizationPotential {
                total_savings_potential: total_estimated_cost * 0.25,
                optimization_opportunities: vec![CostOptimizationOpportunity {
                    opportunity_type: CostOptimizationType::VolumeDiscount,
                    potential_savings: total_estimated_cost * 0.10,
                    implementation_effort: 0.20,
                    description: "Batch similar circuits to reduce per-task overhead.".to_string(),
                }],
                implementation_barriers: Vec::new(),
            },
        };

        // ----------------------------------------------------------------
        // Temporal patterns (seed with neutral/stable defaults)
        // ----------------------------------------------------------------
        let temporal_patterns = TemporalPatterns {
            seasonality: SeasonalityAnalysis {
                seasonal_components: Vec::new(),
                seasonal_strength: 0.0,
                dominant_frequencies: Vec::new(),
            },
            trend_analysis: TrendAnalysis {
                metric_name: "execution_time".to_string(),
                trend_direction: TrendDirection::Stable,
                trend_strength: 0.0,
                prediction_accuracy: 0.70,
                data_points: Vec::new(),
            },
            cyclical_patterns: CyclicalPatterns {
                cycle_length: Duration::from_secs(86400),
                cycle_amplitude: 0.0,
                cycle_regularity: 0.0,
                cycle_predictability: 0.5,
            },
            anomaly_patterns: AnomalyPatterns {
                anomaly_frequency: 0.02,
                anomaly_types: Vec::new(),
                anomaly_impact: 0.05,
                detection_accuracy: 0.80,
            },
        };

        // ----------------------------------------------------------------
        // Assemble profile
        // ----------------------------------------------------------------
        let profile_id = format!(
            "profile_{}_{}_{}_{}",
            workload.workload_id, cc.qubit_count, cc.gate_count, er.shots
        );

        Ok(WorkloadProfile {
            profile_id,
            workload_type: workload.workload_type.clone(),
            characteristics: workload_characteristics,
            resource_patterns,
            performance_patterns,
            cost_patterns,
            temporal_patterns,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DataAccessPatterns {
    pub access_pattern: AccessPattern,
    pub locality: LocalityPattern,
    pub caching_behavior: CachingBehavior,
}

pub struct SimilarityEngine {
    similarity_metrics: Vec<Box<dyn SimilarityMetric + Send + Sync>>,
    nearest_neighbor_engines: Vec<Box<dyn NearestNeighborEngine + Send + Sync>>,
    similarity_cache: HashMap<String, SimilarityResult>,
}
impl Default for SimilarityEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SimilarityEngine {
    pub fn new() -> Self {
        Self {
            similarity_metrics: Vec::new(),
            nearest_neighbor_engines: Vec::new(),
            similarity_cache: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThroughputPattern {
    pub average_throughput: f64,
    pub peak_throughput: f64,
    pub throughput_stability: f64,
    pub bottleneck_analysis: BottleneckAnalysis,
}

#[derive(Debug, Clone)]
pub enum GlobalStructure {
    FunnelLike,
    GolfCourse,
    Archipelago,
    MassifCentral,
    NeedleInHaystack,
}

#[derive(Debug, Clone)]
pub enum LocalityPattern {
    Spatial,
    Temporal,
    Both,
    None,
}

#[derive(Debug, Clone)]
pub struct ParallelScalingCharacteristics {
    pub maximum_parallelism: usize,
    pub parallel_efficiency: f64,
    pub load_balance_quality: f64,
    pub synchronization_overhead: f64,
}

#[derive(Debug, Clone)]
pub enum RefinementType {
    ConditionRefinement,
    ActionRefinement,
    ConfidenceAdjustment,
    ScopeExpansion,
    ScopeRestriction,
}

#[derive(Debug, Clone)]
pub struct LearningPriority {
    pub priority_area: String,
    pub importance_score: f64,
    pub data_requirements: Vec<String>,
    pub expected_benefit: f64,
}

#[derive(Debug, Clone)]
pub struct DependencyNode {
    pub node_id: String,
    pub operation_type: String,
    pub computational_cost: f64,
    pub memory_requirement: usize,
}

#[derive(Debug, Clone)]
pub enum DataStructure {
    Vector,
    Matrix,
    Tensor,
    Graph,
    Tree,
    Sparse,
    Stream,
}

#[derive(Debug, Clone)]
pub struct RecurrencePattern {
    pub pattern_type: RecurrenceType,
    pub interval: Duration,
    pub end_date: Option<SystemTime>,
    pub exceptions: Vec<SystemTime>,
}

#[derive(Debug, Clone)]
pub struct ExpertOpinion {
    pub expert_id: String,
    pub expertise_domain: String,
    pub opinion_summary: String,
    pub confidence_level: f64,
    pub supporting_rationale: String,
}

#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    pub pattern_type: SeasonalType,
    pub amplitude: f64,
    pub period: Duration,
    pub phase_offset: Duration,
}

#[derive(Debug, Clone)]
pub struct PatternAnalysisResult {
    pub patterns_identified: Vec<IdentifiedPattern>,
    pub pattern_strength: f64,
    pub pattern_confidence: f64,
    pub recommendations: Vec<PatternRecommendation>,
}

#[derive(Debug, Clone)]
pub struct ClusteringResult {
    pub clusters: Vec<WorkloadCluster>,
    pub cluster_quality: ClusterQuality,
    pub outliers: Vec<usize>,
    pub cluster_representatives: Vec<FeatureVector>,
}

#[derive(Debug, Clone)]
pub struct UtilizationPattern {
    pub average_utilization: f64,
    pub peak_utilization: f64,
    pub utilization_variance: f64,
    pub temporal_pattern: TemporalUtilizationPattern,
}

#[derive(Debug, Clone)]
pub enum NetworkCommunicationPattern {
    AllToAll,
    NearestNeighbor,
    Hierarchical,
    Sparse,
    Broadcast,
}

#[derive(Debug, Clone)]
pub struct SimilarityAnalysis {
    pub average_similarity: f64,
    pub similarity_distribution: Vec<f64>,
    pub similarity_clusters: Vec<SimilarityCluster>,
    pub uniqueness_score: f64,
}

#[derive(Debug, Clone)]
pub struct SimilarityCluster {
    pub cluster_id: String,
    pub center_workload: WorkloadProfile,
    pub cluster_members: Vec<WorkloadProfile>,
    pub average_similarity: f64,
}

#[derive(Debug, Clone)]
pub struct ClusterPerformanceProfile {
    pub average_performance: HashMap<String, f64>,
    pub performance_variance: HashMap<String, f64>,
    pub best_performing_providers: Vec<CloudProvider>,
    pub performance_trends: HashMap<String, TrendDirection>,
}

#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    pub statistical_tests: Vec<StatisticalTest>,
    pub correlation_analysis: CorrelationAnalysis,
    pub regression_analysis: Option<RegressionAnalysis>,
    pub significance_level: f64,
}

#[derive(Debug, Clone)]
pub struct TemporalTrend {
    pub provider: CloudProvider,
    pub trend_data: HashMap<String, TrendAnalysis>,
    pub seasonal_patterns: HashMap<String, SeasonalPattern>,
    pub improvement_trajectory: ImprovementTrajectory,
}

#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    pub correlationmatrix: Vec<Vec<f64>>,
    pub variable_names: Vec<String>,
    pub significant_correlations: Vec<(String, String, f64)>,
}

#[derive(Debug, Clone)]
pub enum RecurrenceType {
    Daily,
    Weekly,
    Monthly,
    Yearly,
    Custom,
}

#[derive(Debug, Clone)]
pub struct DataSize {
    pub input_size: usize,
    pub intermediate_size: usize,
    pub output_size: usize,
    pub memory_footprint: usize,
}

#[derive(Debug, Clone)]
pub struct IdentifiedPattern {
    pub pattern_id: String,
    pub pattern_type: PatternType,
    pub pattern_description: String,
    pub pattern_parameters: HashMap<String, f64>,
    pub pattern_significance: f64,
}

#[derive(Debug, Clone)]
pub enum DependencyType {
    Data,
    Control,
    Resource,
    Temporal,
}

#[derive(Debug, Clone)]
pub struct StatisticalTest {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub effect_size: f64,
    pub interpretation: String,
}

#[derive(Debug, Clone)]
pub struct NormalizationParams {
    pub means: Vec<f64>,
    pub standard_deviations: Vec<f64>,
    pub min_values: Vec<f64>,
    pub max_values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SupportingEvidence {
    pub historical_examples: Vec<HistoricalExample>,
    pub benchmark_comparisons: Vec<BenchmarkComparison>,
    pub expert_opinions: Vec<ExpertOpinion>,
    pub statistical_analysis: StatisticalAnalysis,
}

#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
    pub feature_importance: Vec<f64>,
    pub normalization_params: Option<NormalizationParams>,
}

#[derive(Debug, Clone)]
pub struct CaseStudy {
    pub case_id: String,
    pub case_title: String,
    pub case_description: String,
    pub problem_statement: String,
    pub solution_approach: String,
    pub results_achieved: HashMap<String, f64>,
    pub lessons_learned: Vec<String>,
    pub applicability: f64,
}

#[derive(Debug, Clone)]
pub struct TemporalPatterns {
    pub seasonality: SeasonalityAnalysis,
    pub trend_analysis: TrendAnalysis,
    pub cyclical_patterns: CyclicalPatterns,
    pub anomaly_patterns: AnomalyPatterns,
}

#[derive(Debug, Clone)]
pub struct SeasonalComponent {
    pub period: Duration,
    pub amplitude: f64,
    pub phase: f64,
    pub significance: f64,
}

#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub prediction_accuracy: f64,
    pub data_points: Vec<(SystemTime, f64)>,
}

#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    pub data_size: DataSize,
    pub data_structure: DataStructure,
    pub data_access_patterns: DataAccessPatterns,
    pub data_dependencies: DataDependencies,
}

#[derive(Debug, Clone)]
pub enum PatternType {
    Temporal,
    Resource,
    Performance,
    Cost,
    Quality,
    Behavioral,
}

#[derive(Debug, Clone)]
pub struct ResidualAnalysis {
    pub residuals: Vec<f64>,
    pub residual_patterns: Vec<String>,
    pub normality_test: StatisticalTest,
    pub heteroscedasticity_test: StatisticalTest,
}

#[derive(Debug, Clone)]
pub struct DataDependencies {
    pub dependency_graph: DependencyGraph,
    pub critical_path: Vec<String>,
    pub parallelization_potential: f64,
}

#[derive(Debug, Clone)]
pub struct FeatureComparison {
    pub feature_scores: HashMap<String, (f64, f64)>,
    pub unique_features: (Vec<String>, Vec<String>),
    pub compatibility_scores: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum TemporalUtilizationPattern {
    Constant,
    Increasing,
    Decreasing,
    Periodic,
    Bursty,
    Random,
}

#[derive(Debug, Clone)]
pub struct CyclicalPatterns {
    pub cycle_length: Duration,
    pub cycle_amplitude: f64,
    pub cycle_regularity: f64,
    pub cycle_predictability: f64,
}

#[derive(Debug, Clone)]
pub struct SimilarityResult {
    pub similar_workloads: Vec<SimilarWorkload>,
    pub similarity_analysis: SimilarityAnalysis,
    pub recommendations: Vec<SimilarityRecommendation>,
}

#[derive(Debug, Clone)]
pub struct SeasonalityAnalysis {
    pub seasonal_components: Vec<SeasonalComponent>,
    pub seasonal_strength: f64,
    pub dominant_frequencies: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum TrendDirection {
    Improving,
    Stable,
    Declining,
    Volatile,
}

#[derive(Debug, Clone)]
pub enum TimeDistribution {
    Normal,
    LogNormal,
    Exponential,
    Uniform,
    Bimodal,
    HeavyTailed,
}
