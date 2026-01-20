//! Auto-generated module - workload
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
    ClusteringEngine, FeatureExtractor, FeedbackAggregator, FeedbackAnalyzer, FeedbackValidator,
    LearningAlgorithm, NearestNeighborEngine, PatternAnalysisAlgorithm, ProviderOptimizer,
    RecommendationAlgorithm, SimilarityMetric, UpdateStrategy,
};

// Cross-module imports from sibling modules
use super::{cost::*, execution::*, optimization::*, profiling::*, providers::*, tracking::*};

#[derive(Debug, Clone)]
pub struct CircuitCharacteristics {
    pub qubit_count: usize,
    pub gate_count: usize,
    pub circuit_depth: usize,
    pub gate_types: HashMap<String, usize>,
    pub connectivity_requirements: ConnectivityRequirements,
    pub coherence_requirements: CoherenceRequirements,
    pub noise_tolerance: f64,
}

#[derive(Debug, Clone)]
pub enum QubitMappingStrategy {
    Trivial,
    NoiseAdaptive,
    TopologyAware,
    ConnectivityOptimized,
    FidelityOptimized,
    MlOptimized,
}

#[derive(Debug, Clone)]
pub struct CoherenceTimeRequirements {
    pub min_t1_us: f64,
    pub min_t2_us: f64,
    pub min_gate_time_ns: f64,
    pub thermal_requirements: f64,
}

#[derive(Debug, Clone)]
pub enum BarrierType {
    Technical,
    Economic,
    Organizational,
    Regulatory,
    Cultural,
}

#[derive(Debug, Clone)]
pub enum SpecializedProcessor {
    TPU,
    FPGA,
    ASIC,
    Neuromorphic,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct ImplementationDetails {
    pub implementation_steps: Vec<ImplementationStep>,
    pub required_resources: Vec<String>,
    pub estimated_duration: Duration,
    pub dependencies: Vec<String>,
    pub rollback_plan: Option<RollbackPlan>,
}

#[derive(Debug, Clone)]
pub struct RepeatabilityRequirements {
    pub required_consistency: f64,
    pub max_variance: f64,
    pub calibration_frequency: Duration,
    pub drift_tolerance: f64,
}

#[derive(Debug, Clone)]
pub enum ExampleComplexity {
    Beginner,
    Intermediate,
    Advanced,
}

#[derive(Debug, Clone)]
pub struct WorkloadSpec {
    pub workload_id: String,
    pub workload_type: WorkloadType,
    pub circuit_characteristics: CircuitCharacteristics,
    pub execution_requirements: ExecutionRequirements,
    pub resource_constraints: ResourceConstraints,
    pub priority: WorkloadPriority,
    pub deadline: Option<SystemTime>,
}

#[derive(Debug, Clone)]
pub enum QuantumComplexityClass {
    BQP,
    QMA,
    QPSPACE,
    BPP,
    NP,
    Unknown,
}

#[derive(Debug, Clone)]
pub enum MeasurementPattern {
    Final,
    Intermediate,
    Adaptive,
    Continuous,
    Conditional,
}

#[derive(Debug, Clone)]
pub struct AlgorithmRegistration {
    pub algorithm_name: String,
    pub algorithm_type: AlgorithmCategory,
    pub code: AlgorithmCode,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct WorkloadProfile {
    pub profile_id: String,
    pub workload_type: WorkloadType,
    pub characteristics: WorkloadCharacteristics,
    pub resource_patterns: ResourcePatterns,
    pub performance_patterns: PerformancePatterns,
    pub cost_patterns: CostPatterns,
    pub temporal_patterns: TemporalPatterns,
}

#[derive(Debug, Clone)]
pub struct NoiseSensitivity {
    pub gate_error_sensitivity: f64,
    pub decoherence_sensitivity: f64,
    pub measurement_error_sensitivity: f64,
    pub classical_noise_sensitivity: f64,
}

#[derive(Debug, Clone)]
pub enum RoutingOptimizationStrategy {
    ShortestPath,
    MinimumSwaps,
    FidelityAware,
    NoiseAware,
    CongestionAware,
    Adaptive,
}

#[derive(Debug, Clone)]
pub enum WorkloadPriority {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
}

#[derive(Debug, Clone)]
pub enum NoiseAdaptationStrategy {
    None,
    Statistical,
    ModelBased,
    MlBased,
    Hybrid,
}

#[derive(Debug, Clone)]
pub enum ComplexityClass {
    Constant,
    Logarithmic,
    Linear,
    Quadratic,
    Cubic,
    Exponential,
    Factorial,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct TestCircuit {
    pub circuit_id: String,
    pub circuit_type: String,
    pub qubit_count: usize,
    pub gate_count: usize,
    pub circuit_depth: usize,
    pub complexity_score: f64,
}

#[derive(Debug, Clone)]
pub enum TopologyType {
    Linear,
    Ring,
    Grid,
    Ladder,
    Star,
    Complete,
    HeavyHex,
    Falcon,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum EntanglementPattern {
    Local,
    GloballyEntangled,
    Clustered,
    Linear,
    Tree,
    Random,
}

#[derive(Debug, Clone)]
pub enum WorkloadType {
    Simulation,
    Optimization,
    MachineLearning,
    Cryptography,
    Chemistry,
    FinancialModeling,
    Research,
    Production,
    Custom(String),
}
impl WorkloadType {
    /// Convert WorkloadType to u8 for hashing/identification purposes
    pub const fn as_u8(&self) -> u8 {
        match self {
            Self::Simulation => 0,
            Self::Optimization => 1,
            Self::MachineLearning => 2,
            Self::Cryptography => 3,
            Self::Chemistry => 4,
            Self::FinancialModeling => 5,
            Self::Research => 6,
            Self::Production => 7,
            Self::Custom(_) => 8,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TranspilationLevel {
    None,
    Basic,
    Intermediate,
    Advanced,
    Aggressive,
}

#[derive(Debug, Clone)]
pub enum AlgorithmFamily {
    Optimization,
    Simulation,
    MachineLearning,
    Cryptography,
    Search,
    Factorization,
    LinearAlgebra,
}

#[derive(Debug, Clone)]
pub enum CalibrationOptimizationStrategy {
    Static,
    Dynamic,
    Predictive,
    RealTime,
    MlDriven,
}

#[derive(Debug, Clone)]
pub enum AlgorithmCategory {
    Optimization,
    Simulation,
    MachineLearning,
    Cryptography,
    Chemistry,
    FinancialModeling,
}

#[derive(Debug, Clone)]
pub struct CircuitOptimizationSettings {
    pub gate_fusion: bool,
    pub gate_cancellation: bool,
    pub circuit_compression: bool,
    pub transpilation_level: TranspilationLevel,
    pub error_mitigation: ErrorMitigationSettings,
}

#[derive(Debug, Clone)]
pub struct WorkloadData {
    pub execution_history: Vec<ExecutionRecord>,
    pub resource_usage_history: Vec<ResourceUsageRecord>,
    pub performance_history: Vec<PerformanceRecord>,
    pub cost_history: Vec<CostRecord>,
}

#[derive(Debug, Clone)]
pub struct AlgorithmCode {
    pub code: String,
    pub language: String,
    pub framework: String,
}

#[derive(Debug, Clone)]
pub struct AlgorithmicProperties {
    pub algorithm_family: AlgorithmFamily,
    pub optimization_landscape: OptimizationLandscape,
    pub convergence_properties: ConvergenceProperties,
    pub noise_sensitivity: NoiseSensitivity,
}

#[derive(Debug, Clone)]
pub struct ComputationalComplexity {
    pub time_complexity: ComplexityClass,
    pub space_complexity: ComplexityClass,
    pub quantum_complexity: QuantumComplexityClass,
    pub parallel_complexity: ParallelComplexityClass,
}

#[derive(Debug, Clone)]
pub enum ParallelComplexityClass {
    NC,
    P,
    RNC,
    AC,
    TC,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ConnectivityRequirements {
    pub topology_type: TopologyType,
    pub min_connectivity: f64,
    pub required_couplings: Vec<(usize, usize)>,
    pub coupling_strength_requirements: HashMap<(usize, usize), f64>,
}

#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    pub computational_complexity: ComputationalComplexity,
    pub data_characteristics: DataCharacteristics,
    pub algorithmic_properties: AlgorithmicProperties,
    pub scalability_characteristics: ScalabilityCharacteristics,
}

#[derive(Debug, Clone)]
pub struct CoherenceRequirements {
    pub min_t1_time: Duration,
    pub min_t2_time: Duration,
    pub min_gate_fidelity: f64,
    pub min_readout_fidelity: f64,
    pub thermal_requirements: ThermalRequirements,
}

#[derive(Debug, Clone)]
pub struct WorkloadCostStructure {
    pub fixed_costs: f64,
    pub variable_costs: f64,
    pub marginal_costs: f64,
    pub cost_drivers: Vec<CostDriver>,
}

#[derive(Debug, Clone)]
pub struct WorkloadCluster {
    pub cluster_id: String,
    pub cluster_center: FeatureVector,
    pub cluster_members: Vec<usize>,
    pub cluster_characteristics: ClusterCharacteristics,
    pub intra_cluster_similarity: f64,
}
