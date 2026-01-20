//! Quantum Advantage Analysis Suite
//!
//! This module provides tools to analyze when quantum optimization approaches
//! provide theoretical and practical advantages over classical methods.

#![allow(dead_code)]

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Quantum advantage analysis engine
pub struct QuantumAdvantageAnalyzer {
    /// Configuration for analysis
    config: AnalysisConfig,
    /// Classical complexity estimator
    classical_estimator: ClassicalComplexityEstimator,
    /// Quantum resource estimator
    quantum_estimator: QuantumResourceEstimator,
    /// Benchmarking suite
    benchmarker: QuantumSupremacyBenchmarker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Problem size range for analysis
    pub problem_size_range: (usize, usize),
    /// Number of samples for statistical analysis
    pub num_samples: usize,
    /// Confidence level for advantage detection
    pub confidence_level: f64,
    /// Classical algorithms to compare against
    pub classical_baselines: Vec<ClassicalAlgorithm>,
    /// Quantum algorithms to analyze
    pub quantum_algorithms: Vec<QuantumAlgorithm>,
    /// Hardware models to consider
    pub hardware_models: Vec<HardwareModel>,
    /// Noise models for realistic analysis
    pub noise_models: Vec<NoiseModel>,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            problem_size_range: (10, 1000),
            num_samples: 100,
            confidence_level: 0.95,
            classical_baselines: vec![
                ClassicalAlgorithm::SimulatedAnnealing,
                ClassicalAlgorithm::TabuSearch,
                ClassicalAlgorithm::GeneticAlgorithm,
                ClassicalAlgorithm::BranchAndBound,
            ],
            quantum_algorithms: vec![
                QuantumAlgorithm::QuantumAnnealing,
                QuantumAlgorithm::QAOA,
                QuantumAlgorithm::VQE,
                QuantumAlgorithm::QFAST,
            ],
            hardware_models: vec![
                HardwareModel::IdealQuantum,
                HardwareModel::NoisyNISQ { error_rate: 0.001 },
                HardwareModel::DigitalAnnealer,
                HardwareModel::PhotonicIsing,
            ],
            noise_models: vec![
                NoiseModel::None,
                NoiseModel::Depolarizing { rate: 0.001 },
                NoiseModel::AmplitudeDamping { rate: 0.01 },
                NoiseModel::Realistic {
                    decoherence_time: Duration::from_micros(100),
                },
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ClassicalAlgorithm {
    SimulatedAnnealing,
    TabuSearch,
    GeneticAlgorithm,
    BranchAndBound,
    ConstraintProgramming,
    LinearProgramming,
    SDP,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QuantumAlgorithm {
    QuantumAnnealing,
    QAOA,
    VQE,
    QFAST,
    QuantumWalk,
    AdiabaticQuantumComputing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareModel {
    IdealQuantum,
    NoisyNISQ { error_rate: f64 },
    DigitalAnnealer,
    PhotonicIsing,
    CoherentIsingMachine,
    SuperconductingAnnealer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseModel {
    None,
    Depolarizing { rate: f64 },
    AmplitudeDamping { rate: f64 },
    PhaseDamping { rate: f64 },
    Realistic { decoherence_time: Duration },
}

/// Results of quantum advantage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvantageAnalysisResult {
    /// Problem characteristics
    pub problem_info: ProblemCharacteristics,
    /// Classical performance estimates
    pub classical_performance: HashMap<ClassicalAlgorithm, PerformanceMetrics>,
    /// Quantum performance estimates
    pub quantum_performance: HashMap<QuantumAlgorithm, QuantumPerformanceMetrics>,
    /// Advantage analysis
    pub advantage_analysis: AdvantageAnalysis,
    /// Threshold analysis
    pub threshold_analysis: ThresholdAnalysis,
    /// Recommendations
    pub recommendations: Vec<AlgorithmRecommendation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemCharacteristics {
    /// Problem type
    pub problem_type: String,
    /// Number of variables
    pub num_variables: usize,
    /// Problem density (fraction of non-zero coefficients)
    pub density: f64,
    /// Connectivity structure
    pub connectivity: ConnectivityStructure,
    /// Problem hardness indicators
    pub hardness_indicators: HardnessIndicators,
    /// Symmetry properties
    pub symmetries: Vec<SymmetryType>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityStructure {
    FullyConnected,
    Sparse { avg_degree: f64 },
    Grid { dimensions: Vec<usize> },
    Tree,
    Planar,
    SmallWorld { clustering_coefficient: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardnessIndicators {
    /// Estimated classical complexity class
    pub complexity_class: ComplexityClass,
    /// Problem-specific difficulty metrics
    pub difficulty_metrics: HashMap<String, f64>,
    /// Approximation ratio bounds
    pub approximation_bounds: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityClass {
    P,
    NP,
    NPComplete,
    NPHard,
    PSpace,
    ExpTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymmetryType {
    Permutation,
    Reflection,
    Rotation,
    Translation,
    Scale,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Time complexity estimate
    pub time_complexity: TimeComplexity,
    /// Space complexity estimate
    pub space_complexity: SpaceComplexity,
    /// Solution quality estimate
    pub solution_quality: QualityMetrics,
    /// Success probability
    pub success_probability: f64,
    /// Scaling behavior
    pub scaling_behavior: ScalingBehavior,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPerformanceMetrics {
    /// Base performance metrics
    pub base_metrics: PerformanceMetrics,
    /// Quantum-specific metrics
    pub quantum_metrics: QuantumSpecificMetrics,
    /// Hardware requirements
    pub hardware_requirements: HardwareRequirements,
    /// Noise sensitivity
    pub noise_sensitivity: NoiseSensitivity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSpecificMetrics {
    /// Number of qubits required
    pub qubits_required: usize,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Number of quantum operations
    pub quantum_operations: usize,
    /// Entanglement measures
    pub entanglement_measures: EntanglementMeasures,
    /// Coherence time requirements
    pub coherence_time_required: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRequirements {
    /// Minimum qubit count
    pub min_qubits: usize,
    /// Required connectivity
    pub connectivity_requirements: ConnectivityRequirement,
    /// Gate fidelity requirements
    pub gate_fidelity_threshold: f64,
    /// Measurement fidelity requirements
    pub measurement_fidelity_threshold: f64,
    /// Operating temperature requirements
    pub temperature_requirements: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityRequirement {
    AllToAll,
    NearestNeighbor,
    Specific { required_edges: Vec<(usize, usize)> },
    MinDegree { min_degree: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseSensitivity {
    /// Threshold error rates for different noise types
    pub error_thresholds: HashMap<String, f64>,
    /// Degradation rates under noise
    pub degradation_rates: HashMap<String, f64>,
    /// Error mitigation effectiveness
    pub mitigation_effectiveness: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementMeasures {
    /// Average entanglement entropy
    pub avg_entanglement_entropy: f64,
    /// Maximum entanglement entropy
    pub max_entanglement_entropy: f64,
    /// Entanglement depth
    pub entanglement_depth: usize,
    /// Multipartite entanglement measures
    pub multipartite_measures: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeComplexity {
    /// Best case complexity
    pub best_case: ComplexityFunction,
    /// Average case complexity
    pub average_case: ComplexityFunction,
    /// Worst case complexity
    pub worst_case: ComplexityFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceComplexity {
    /// Memory requirements
    pub memory_complexity: ComplexityFunction,
    /// Storage requirements
    pub storage_complexity: ComplexityFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityFunction {
    Constant,
    Logarithmic,
    Linear,
    Linearithmic,
    Quadratic,
    Cubic,
    Polynomial { degree: f64 },
    Exponential { base: f64 },
    Factorial,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Expected approximation ratio
    pub approximation_ratio: f64,
    /// Solution variance
    pub solution_variance: f64,
    /// Probability of optimal solution
    pub optimal_probability: f64,
    /// Expected energy gap
    pub energy_gap: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingBehavior {
    /// Scaling exponent
    pub scaling_exponent: f64,
    /// Scaling constant
    pub scaling_constant: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// R-squared of fit
    pub fit_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvantageAnalysis {
    /// Overall advantage assessment
    pub has_quantum_advantage: bool,
    /// Advantage confidence level
    pub confidence: f64,
    /// Advantage factors by metric
    pub advantage_factors: HashMap<String, f64>,
    /// Conditional advantages
    pub conditional_advantages: Vec<ConditionalAdvantage>,
    /// Break-even analysis
    pub break_even_points: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalAdvantage {
    /// Conditions under which advantage exists
    pub conditions: Vec<String>,
    /// Advantage magnitude under these conditions
    pub advantage_magnitude: f64,
    /// Probability conditions are met
    pub condition_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdAnalysis {
    /// Problem size thresholds for quantum advantage
    pub size_thresholds: HashMap<String, usize>,
    /// Noise thresholds for quantum advantage
    pub noise_thresholds: HashMap<String, f64>,
    /// Hardware requirement thresholds
    pub hardware_thresholds: HashMap<String, f64>,
    /// Time-to-advantage predictions
    pub time_to_advantage: HashMap<String, Duration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmRecommendation {
    /// Recommended algorithm
    pub algorithm: String,
    /// Algorithm type (classical or quantum)
    pub algorithm_type: AlgorithmType,
    /// Confidence in recommendation
    pub confidence: f64,
    /// Expected performance
    pub expected_performance: PerformanceMetrics,
    /// Justification for recommendation
    pub justification: String,
    /// Alternative algorithms
    pub alternatives: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmType {
    Classical,
    Quantum,
    Hybrid,
}

/// Classical complexity estimator
pub struct ClassicalComplexityEstimator {
    /// Known complexity results
    complexity_database: HashMap<String, ComplexityInfo>,
    /// Heuristic estimators
    heuristic_estimators: Vec<Box<dyn ComplexityHeuristic>>,
}

pub trait ComplexityHeuristic {
    fn estimate_complexity(&self, problem: &ProblemCharacteristics) -> PerformanceMetrics;
    fn algorithm_name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct ComplexityInfo {
    pub time_complexity: TimeComplexity,
    pub space_complexity: SpaceComplexity,
    pub approximation_bounds: Option<(f64, f64)>,
    pub practical_scaling: ScalingBehavior,
}

/// Quantum resource estimator
pub struct QuantumResourceEstimator {
    /// Quantum algorithm database
    algorithm_database: HashMap<String, QuantumAlgorithmInfo>,
    /// Hardware model database
    hardware_database: HashMap<String, HardwareCharacteristics>,
}

#[derive(Debug, Clone)]
pub struct QuantumAlgorithmInfo {
    pub resource_requirements: HardwareRequirements,
    pub performance_model: QuantumPerformanceModel,
    pub noise_sensitivity: NoiseSensitivity,
    pub theoretical_advantage: Option<AdvantageEstimate>,
}

pub struct QuantumPerformanceModel {
    pub time_to_solution: Box<dyn Fn(usize) -> Duration>,
    pub success_probability: Box<dyn Fn(usize, f64) -> f64>,
    pub resource_scaling: ScalingBehavior,
}

impl std::fmt::Debug for QuantumPerformanceModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantumPerformanceModel")
            .field("time_to_solution", &"<function>")
            .field("success_probability", &"<function>")
            .field("resource_scaling", &self.resource_scaling)
            .finish()
    }
}

impl Clone for QuantumPerformanceModel {
    fn clone(&self) -> Self {
        Self {
            time_to_solution: Box::new(|_| Duration::from_secs(1)),
            success_probability: Box::new(|_, _| 0.5),
            resource_scaling: self.resource_scaling.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdvantageEstimate {
    pub speedup_factor: f64,
    pub advantage_type: AdvantageType,
    pub confidence_level: f64,
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum AdvantageType {
    Polynomial { degree_reduction: f64 },
    Exponential { base_improvement: f64 },
    Constant { factor: f64 },
    Conditional { conditions: Vec<String> },
}

#[derive(Debug, Clone)]
pub struct HardwareCharacteristics {
    pub qubit_count: usize,
    pub connectivity: ConnectivityStructure,
    pub gate_fidelities: HashMap<String, f64>,
    pub measurement_fidelity: f64,
    pub coherence_times: HashMap<String, Duration>,
    pub operating_parameters: HashMap<String, f64>,
}

/// Quantum supremacy benchmarker
pub struct QuantumSupremacyBenchmarker {
    /// Benchmark problem generators
    problem_generators: Vec<Box<dyn BenchmarkProblemGenerator>>,
    /// Performance trackers
    performance_trackers: HashMap<String, PerformanceTracker>,
}

pub trait BenchmarkProblemGenerator {
    fn generate_problem(&self, size: usize, seed: Option<u64>) -> BenchmarkProblem;
    fn problem_type(&self) -> &str;
    fn difficulty_level(&self) -> DifficultyLevel;
}

#[derive(Debug, Clone)]
pub struct BenchmarkProblem {
    pub qubo: Array2<f64>,
    pub metadata: ProblemMetadata,
    pub known_optimal: Option<f64>,
    pub theoretical_properties: TheoreticalProperties,
}

#[derive(Debug, Clone)]
pub struct ProblemMetadata {
    pub problem_type: String,
    pub size: usize,
    pub density: f64,
    pub generation_seed: Option<u64>,
    pub generation_time: Instant,
}

#[derive(Debug, Clone)]
pub struct TheoreticalProperties {
    pub complexity_class: ComplexityClass,
    pub approximation_hardness: Option<f64>,
    pub spectral_gap: Option<f64>,
    pub landscape_features: LandscapeFeatures,
}

#[derive(Debug, Clone)]
pub struct LandscapeFeatures {
    pub num_local_minima: Option<usize>,
    pub barrier_heights: Vec<f64>,
    pub correlation_length: f64,
    pub ruggedness_measure: f64,
}

#[derive(Debug, Clone)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    ExtremelyHard,
}

#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    pub algorithm_name: String,
    pub performance_history: Vec<PerformanceDataPoint>,
    pub scaling_model: Option<ScalingBehavior>,
}

#[derive(Debug, Clone)]
pub struct PerformanceDataPoint {
    pub problem_size: usize,
    pub execution_time: Duration,
    pub solution_quality: f64,
    pub success_rate: f64,
    pub timestamp: Instant,
}

impl QuantumAdvantageAnalyzer {
    /// Create new quantum advantage analyzer
    pub fn new(config: AnalysisConfig) -> Self {
        Self {
            config,
            classical_estimator: ClassicalComplexityEstimator::new(),
            quantum_estimator: QuantumResourceEstimator::new(),
            benchmarker: QuantumSupremacyBenchmarker::new(),
        }
    }

    /// Analyze quantum advantage for a given problem
    pub fn analyze_advantage(
        &self,
        qubo: &Array2<f64>,
        problem_metadata: Option<ProblemMetadata>,
    ) -> Result<AdvantageAnalysisResult, String> {
        // Characterize the problem
        let problem_chars = self.characterize_problem(qubo, problem_metadata)?;

        // Estimate classical performance
        let classical_performance = self
            .classical_estimator
            .estimate_classical_performance(&problem_chars, &self.config.classical_baselines)?;

        // Estimate quantum performance
        let quantum_performance = self
            .quantum_estimator
            .estimate_quantum_performance(&problem_chars, &self.config.quantum_algorithms)?;

        // Perform advantage analysis
        let advantage_analysis = self.analyze_quantum_advantage(
            &classical_performance,
            &quantum_performance,
            &problem_chars,
        )?;

        // Perform threshold analysis
        let threshold_analysis =
            self.analyze_thresholds(&classical_performance, &quantum_performance, &problem_chars)?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &classical_performance,
            &quantum_performance,
            &advantage_analysis,
            &problem_chars,
        )?;

        Ok(AdvantageAnalysisResult {
            problem_info: problem_chars,
            classical_performance,
            quantum_performance,
            advantage_analysis,
            threshold_analysis,
            recommendations,
        })
    }

    /// Characterize problem structure and properties
    fn characterize_problem(
        &self,
        qubo: &Array2<f64>,
        metadata: Option<ProblemMetadata>,
    ) -> Result<ProblemCharacteristics, String> {
        let num_variables = qubo.shape()[0];

        // Calculate density
        let non_zero_count = qubo.iter().filter(|&&x| x.abs() > 1e-10).count();
        let density = non_zero_count as f64 / (num_variables * num_variables) as f64;

        // Analyze connectivity
        let connectivity = self.analyze_connectivity(qubo)?;

        // Compute hardness indicators
        let hardness_indicators = self.compute_hardness_indicators(qubo)?;

        // Detect symmetries
        let symmetries = self.detect_symmetries(qubo)?;

        Ok(ProblemCharacteristics {
            problem_type: metadata
                .as_ref()
                .map_or_else(|| "Unknown".to_string(), |m| m.problem_type.clone()),
            num_variables,
            density,
            connectivity,
            hardness_indicators,
            symmetries,
        })
    }

    /// Analyze connectivity structure of QUBO
    fn analyze_connectivity(&self, qubo: &Array2<f64>) -> Result<ConnectivityStructure, String> {
        let n = qubo.shape()[0];
        let mut edge_count = 0;
        let mut degree_sum = 0;

        for i in 0..n {
            let mut degree = 0;
            for j in 0..n {
                if i != j && qubo[[i, j]].abs() > 1e-10 {
                    if i < j {
                        edge_count += 1;
                    }
                    degree += 1;
                }
            }
            degree_sum += degree;
        }

        let avg_degree = degree_sum as f64 / n as f64;

        // Determine connectivity type based on density and structure
        if edge_count == n * (n - 1) / 2 {
            Ok(ConnectivityStructure::FullyConnected)
        } else if avg_degree < n as f64 / 4.0 {
            Ok(ConnectivityStructure::Sparse { avg_degree })
        } else {
            // Check for grid-like structure (simplified heuristic)
            let expected_grid_degree = 4.0; // For 2D grid
            if (avg_degree - expected_grid_degree).abs() < 1.0 {
                Ok(ConnectivityStructure::Grid {
                    dimensions: vec![(n as f64).sqrt() as usize, (n as f64).sqrt() as usize],
                })
            } else {
                Ok(ConnectivityStructure::Sparse { avg_degree })
            }
        }
    }

    /// Compute hardness indicators for the problem
    fn compute_hardness_indicators(
        &self,
        qubo: &Array2<f64>,
    ) -> Result<HardnessIndicators, String> {
        let n = qubo.shape()[0];

        // Estimate complexity class based on structure
        let complexity_class = if n < 100 {
            ComplexityClass::P
        } else {
            ComplexityClass::NPComplete // Most QUBO problems are NP-complete
        };

        // Compute difficulty metrics
        let mut difficulty_metrics = HashMap::new();

        // Compute coefficient variance as a measure of heterogeneity
        let coeffs: Vec<f64> = qubo.iter().copied().collect();
        let mean = coeffs.iter().sum::<f64>() / coeffs.len() as f64;
        let variance = coeffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / coeffs.len() as f64;
        difficulty_metrics.insert("coefficient_variance".to_string(), variance);

        // Compute spectral gap estimate (simplified)
        let max_coeff = coeffs.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let min_coeff = coeffs
            .iter()
            .map(|x| x.abs())
            .filter(|&x| x > 1e-10)
            .fold(f64::INFINITY, f64::min);
        if min_coeff.is_finite() {
            difficulty_metrics.insert("dynamic_range".to_string(), max_coeff / min_coeff);
        }

        // Frustration measure (simplified)
        let mut frustration = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                if qubo[[i, j]] > 0.0 {
                    frustration += qubo[[i, j]];
                }
            }
        }
        difficulty_metrics.insert("frustration_measure".to_string(), frustration);

        Ok(HardnessIndicators {
            complexity_class,
            difficulty_metrics,
            approximation_bounds: None, // Would require more sophisticated analysis
        })
    }

    /// Detect symmetries in the QUBO
    fn detect_symmetries(&self, qubo: &Array2<f64>) -> Result<Vec<SymmetryType>, String> {
        let mut symmetries = Vec::new();
        let n = qubo.shape()[0];

        // Check for permutation symmetries (simplified check)
        let mut is_symmetric = true;
        for i in 0..n {
            for j in 0..n {
                if (qubo[[i, j]] - qubo[[j, i]]).abs() > 1e-10 {
                    is_symmetric = false;
                    break;
                }
            }
            if !is_symmetric {
                break;
            }
        }

        if is_symmetric {
            symmetries.push(SymmetryType::Permutation);
        }

        // Additional symmetry detection could be added here

        Ok(symmetries)
    }

    /// Analyze quantum advantage by comparing performance metrics
    fn analyze_quantum_advantage(
        &self,
        classical_perf: &HashMap<ClassicalAlgorithm, PerformanceMetrics>,
        quantum_perf: &HashMap<QuantumAlgorithm, QuantumPerformanceMetrics>,
        _problem_chars: &ProblemCharacteristics,
    ) -> Result<AdvantageAnalysis, String> {
        let mut advantage_factors = HashMap::new();
        let mut conditional_advantages = Vec::new();
        let break_even_points = HashMap::new();

        // Find best classical performance
        let best_classical = classical_perf.values().min_by(|a, b| {
            a.time_complexity
                .average_case
                .partial_cmp(&b.time_complexity.average_case)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Find best quantum performance
        let best_quantum = quantum_perf.values().min_by(|a, b| {
            a.base_metrics
                .time_complexity
                .average_case
                .partial_cmp(&b.base_metrics.time_complexity.average_case)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let has_quantum_advantage = if let (Some(classical), Some(quantum)) =
            (best_classical, best_quantum)
        {
            // Simplified advantage analysis based on success probability and scaling
            let classical_quality = classical.solution_quality.approximation_ratio;
            let quantum_quality = quantum.base_metrics.solution_quality.approximation_ratio;

            advantage_factors.insert("quality".to_string(), quantum_quality / classical_quality);

            let classical_success = classical.success_probability;
            let quantum_success = quantum.base_metrics.success_probability;

            advantage_factors.insert(
                "success_rate".to_string(),
                quantum_success / classical_success,
            );

            // Consider noise effects
            if quantum
                .noise_sensitivity
                .error_thresholds
                .values()
                .any(|&x| x < 0.01)
            {
                conditional_advantages.push(ConditionalAdvantage {
                    conditions: vec!["Low noise environment".to_string()],
                    advantage_magnitude: 2.0,
                    condition_probability: 0.3,
                });
            }

            quantum_quality > classical_quality || quantum_success > classical_success
        } else {
            false
        };

        // Compute overall confidence (simplified)
        let confidence = if has_quantum_advantage {
            0.7 // This would be computed based on statistical analysis
        } else {
            0.3
        };

        Ok(AdvantageAnalysis {
            has_quantum_advantage,
            confidence,
            advantage_factors,
            conditional_advantages,
            break_even_points,
        })
    }

    /// Analyze thresholds for quantum advantage
    fn analyze_thresholds(
        &self,
        _classical_perf: &HashMap<ClassicalAlgorithm, PerformanceMetrics>,
        quantum_perf: &HashMap<QuantumAlgorithm, QuantumPerformanceMetrics>,
        problem_chars: &ProblemCharacteristics,
    ) -> Result<ThresholdAnalysis, String> {
        let mut size_thresholds = HashMap::new();
        let mut noise_thresholds = HashMap::new();
        let mut hardware_thresholds = HashMap::new();
        let mut time_to_advantage = HashMap::new();

        // Estimate problem size threshold
        let estimated_threshold = if problem_chars.num_variables < 100 {
            500 // Classical methods likely better for small problems
        } else {
            100 // Quantum advantage may exist for larger problems
        };

        size_thresholds.insert("general".to_string(), estimated_threshold);

        // Noise thresholds
        for (alg, perf) in quantum_perf {
            if let Some(min_threshold) = perf
                .noise_sensitivity
                .error_thresholds
                .values()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                noise_thresholds.insert(format!("{alg:?}"), *min_threshold);
            }
        }

        // Hardware thresholds
        for (alg, perf) in quantum_perf {
            hardware_thresholds.insert(
                format!("{alg:?}_qubits"),
                perf.hardware_requirements.min_qubits as f64,
            );
            hardware_thresholds.insert(
                format!("{alg:?}_fidelity"),
                perf.hardware_requirements.gate_fidelity_threshold,
            );
        }

        // Time to advantage (simplified projection)
        time_to_advantage.insert(
            "conservative".to_string(),
            Duration::from_secs(365 * 24 * 3600 * 5),
        ); // 5 years
        time_to_advantage.insert(
            "optimistic".to_string(),
            Duration::from_secs(365 * 24 * 3600 * 2),
        ); // 2 years

        Ok(ThresholdAnalysis {
            size_thresholds,
            noise_thresholds,
            hardware_thresholds,
            time_to_advantage,
        })
    }

    /// Generate algorithm recommendations
    fn generate_recommendations(
        &self,
        classical_perf: &HashMap<ClassicalAlgorithm, PerformanceMetrics>,
        quantum_perf: &HashMap<QuantumAlgorithm, QuantumPerformanceMetrics>,
        advantage_analysis: &AdvantageAnalysis,
        problem_chars: &ProblemCharacteristics,
    ) -> Result<Vec<AlgorithmRecommendation>, String> {
        let mut recommendations = Vec::new();

        // Find best classical algorithm
        if let Some((best_classical_alg, best_classical_perf)) =
            classical_perf.iter().max_by(|a, b| {
                a.1.solution_quality
                    .approximation_ratio
                    .partial_cmp(&b.1.solution_quality.approximation_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            recommendations.push(AlgorithmRecommendation {
                algorithm: format!("{best_classical_alg:?}"),
                algorithm_type: AlgorithmType::Classical,
                confidence: 0.8,
                expected_performance: best_classical_perf.clone(),
                justification: "Best performing classical algorithm for this problem type"
                    .to_string(),
                alternatives: classical_perf.keys().map(|k| format!("{k:?}")).collect(),
            });
        }

        // Find best quantum algorithm if quantum advantage exists
        if advantage_analysis.has_quantum_advantage {
            if let Some((best_quantum_alg, best_quantum_perf)) =
                quantum_perf.iter().max_by(|a, b| {
                    a.1.base_metrics
                        .solution_quality
                        .approximation_ratio
                        .partial_cmp(&b.1.base_metrics.solution_quality.approximation_ratio)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                recommendations.push(AlgorithmRecommendation {
                    algorithm: format!("{best_quantum_alg:?}"),
                    algorithm_type: AlgorithmType::Quantum,
                    confidence: advantage_analysis.confidence,
                    expected_performance: best_quantum_perf.base_metrics.clone(),
                    justification: "Quantum advantage detected for this problem".to_string(),
                    alternatives: quantum_perf.keys().map(|k| format!("{k:?}")).collect(),
                });
            }
        }

        // Always recommend hybrid approach for large problems
        if problem_chars.num_variables > 1000 {
            recommendations.push(AlgorithmRecommendation {
                algorithm: "Hybrid Quantum-Classical".to_string(),
                algorithm_type: AlgorithmType::Hybrid,
                confidence: 0.6,
                expected_performance: PerformanceMetrics {
                    time_complexity: TimeComplexity {
                        best_case: ComplexityFunction::Linear,
                        average_case: ComplexityFunction::Quadratic,
                        worst_case: ComplexityFunction::Cubic,
                    },
                    space_complexity: SpaceComplexity {
                        memory_complexity: ComplexityFunction::Linear,
                        storage_complexity: ComplexityFunction::Linear,
                    },
                    solution_quality: QualityMetrics {
                        approximation_ratio: 0.9,
                        solution_variance: 0.1,
                        optimal_probability: 0.1,
                        energy_gap: 0.1,
                    },
                    success_probability: 0.8,
                    scaling_behavior: ScalingBehavior {
                        scaling_exponent: 1.5,
                        scaling_constant: 1.0,
                        confidence_interval: (1.2, 1.8),
                        fit_quality: 0.85,
                    },
                },
                justification: "Hybrid approaches often perform well for large-scale problems"
                    .to_string(),
                alternatives: vec!["Pure classical".to_string(), "Pure quantum".to_string()],
            });
        }

        Ok(recommendations)
    }

    /// Run comprehensive benchmark suite
    pub fn run_supremacy_benchmark(&mut self) -> Result<SupremacyBenchmarkResult, String> {
        self.benchmarker.run_comprehensive_benchmark(&self.config)
    }

    /// Export analysis results
    pub fn export_results(
        &self,
        results: &AdvantageAnalysisResult,
        format: ExportFormat,
    ) -> Result<String, String> {
        match format {
            ExportFormat::JSON => serde_json::to_string_pretty(results)
                .map_err(|e| format!("JSON export failed: {e}")),
            ExportFormat::Python => {
                // Generate Python code for visualization and further analysis
                Ok(self.generate_python_analysis_code(results))
            }
            ExportFormat::Rust => {
                // Generate Rust code for reproducible analysis
                Ok(self.generate_rust_analysis_code(results))
            }
            ExportFormat::QUBO => {
                Err("QUBO export not applicable for analysis results".to_string())
            }
        }
    }

    fn generate_python_analysis_code(&self, results: &AdvantageAnalysisResult) -> String {
        format!(
            r#"
# Quantum Advantage Analysis Results
# Generated by QuantRS2-Tytan

import numpy as np
import matplotlib.pyplot as plt

# Problem characteristics
problem_size = {}
problem_density = {:.3}
has_quantum_advantage = {}
confidence = {:.3}

# Visualization code
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title('Quantum vs Classical Performance')
# Add your performance comparison plots here

plt.subplot(2, 2, 2)
plt.title('Advantage Factors')
# Add advantage factor visualization here

plt.tight_layout()
plt.show()

print(f"Problem size: {{problem_size}}")
print(f"Quantum advantage detected: {{has_quantum_advantage}}")
print(f"Confidence level: {{confidence:.1%}}")
"#,
            results.problem_info.num_variables,
            results.problem_info.density,
            results.advantage_analysis.has_quantum_advantage,
            results.advantage_analysis.confidence,
        )
    }

    fn generate_rust_analysis_code(&self, results: &AdvantageAnalysisResult) -> String {
        format!(
            r#"
// Quantum Advantage Analysis Results
// Generated by QuantRS2-Tytan

use quantrs2_tytan::quantum_advantage_analysis::*;

fn reproduce_analysis() {{
    let problem_size = {};
    let has_advantage = {};
    let confidence = {:.3};

    println!("Problem size: {{}}", problem_size);
    println!("Quantum advantage: {{}}", has_advantage);
    println!("Confidence: {{:.1%}}", confidence);

    // Add your analysis reproduction code here
}}
"#,
            results.problem_info.num_variables,
            results.advantage_analysis.has_quantum_advantage,
            results.advantage_analysis.confidence,
        )
    }
}

impl Default for ClassicalComplexityEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassicalComplexityEstimator {
    pub fn new() -> Self {
        Self {
            complexity_database: Self::build_complexity_database(),
            heuristic_estimators: vec![],
        }
    }

    fn build_complexity_database() -> HashMap<String, ComplexityInfo> {
        let mut db = HashMap::new();

        // Add known complexity results for common problems
        db.insert(
            "max_cut".to_string(),
            ComplexityInfo {
                time_complexity: TimeComplexity {
                    best_case: ComplexityFunction::Exponential { base: 2.0 },
                    average_case: ComplexityFunction::Exponential { base: 2.0 },
                    worst_case: ComplexityFunction::Exponential { base: 2.0 },
                },
                space_complexity: SpaceComplexity {
                    memory_complexity: ComplexityFunction::Linear,
                    storage_complexity: ComplexityFunction::Linear,
                },
                approximation_bounds: Some((0.878, 1.0)),
                practical_scaling: ScalingBehavior {
                    scaling_exponent: 1.8,
                    scaling_constant: 10.0,
                    confidence_interval: (1.6, 2.0),
                    fit_quality: 0.9,
                },
            },
        );

        db
    }

    pub fn estimate_classical_performance(
        &self,
        problem: &ProblemCharacteristics,
        algorithms: &[ClassicalAlgorithm],
    ) -> Result<HashMap<ClassicalAlgorithm, PerformanceMetrics>, String> {
        let mut results = HashMap::new();

        for algorithm in algorithms {
            let performance = self.estimate_algorithm_performance(algorithm, problem)?;
            results.insert(algorithm.clone(), performance);
        }

        Ok(results)
    }

    fn estimate_algorithm_performance(
        &self,
        algorithm: &ClassicalAlgorithm,
        problem: &ProblemCharacteristics,
    ) -> Result<PerformanceMetrics, String> {
        // Simplified performance estimation based on algorithm type and problem characteristics
        let base_complexity = match algorithm {
            ClassicalAlgorithm::SimulatedAnnealing => {
                ComplexityFunction::Polynomial { degree: 2.0 }
            }
            ClassicalAlgorithm::TabuSearch => ComplexityFunction::Polynomial { degree: 2.5 },
            ClassicalAlgorithm::GeneticAlgorithm => ComplexityFunction::Polynomial { degree: 3.0 },
            ClassicalAlgorithm::BranchAndBound => ComplexityFunction::Exponential { base: 2.0 },
            ClassicalAlgorithm::ConstraintProgramming => {
                ComplexityFunction::Exponential { base: 1.5 }
            }
            ClassicalAlgorithm::LinearProgramming => ComplexityFunction::Polynomial { degree: 3.5 },
            ClassicalAlgorithm::SDP => ComplexityFunction::Polynomial { degree: 4.0 },
        };

        // Adjust for problem characteristics
        let scaling_factor = match problem.connectivity {
            ConnectivityStructure::FullyConnected => 1.5,
            ConnectivityStructure::Sparse { .. } => 0.8,
            ConnectivityStructure::Grid { .. } => 0.9,
            ConnectivityStructure::Tree => 0.6,
            ConnectivityStructure::Planar => 0.7,
            ConnectivityStructure::SmallWorld { .. } => 1.1,
        };

        let approximation_ratio = match algorithm {
            ClassicalAlgorithm::SimulatedAnnealing => 0.85,
            ClassicalAlgorithm::TabuSearch => 0.9,
            ClassicalAlgorithm::GeneticAlgorithm => 0.8,
            ClassicalAlgorithm::BranchAndBound => 1.0, // Exact for small instances
            ClassicalAlgorithm::ConstraintProgramming => 0.95,
            ClassicalAlgorithm::LinearProgramming => 0.99, // If problem is relaxable
            ClassicalAlgorithm::SDP => 0.878,              // For Max-Cut
        };

        Ok(PerformanceMetrics {
            time_complexity: TimeComplexity {
                best_case: base_complexity.clone(),
                average_case: base_complexity.clone(),
                worst_case: base_complexity,
            },
            space_complexity: SpaceComplexity {
                memory_complexity: ComplexityFunction::Linear,
                storage_complexity: ComplexityFunction::Linear,
            },
            solution_quality: QualityMetrics {
                approximation_ratio,
                solution_variance: 0.1,
                optimal_probability: if matches!(algorithm, ClassicalAlgorithm::BranchAndBound) {
                    1.0
                } else {
                    0.1
                },
                energy_gap: 0.05,
            },
            success_probability: 0.9,
            scaling_behavior: ScalingBehavior {
                scaling_exponent: scaling_factor,
                scaling_constant: 1.0,
                confidence_interval: (scaling_factor * 0.8, scaling_factor * 1.2),
                fit_quality: 0.8,
            },
        })
    }
}

impl Default for QuantumResourceEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumResourceEstimator {
    pub fn new() -> Self {
        Self {
            algorithm_database: Self::build_algorithm_database(),
            hardware_database: Self::build_hardware_database(),
        }
    }

    fn build_algorithm_database() -> HashMap<String, QuantumAlgorithmInfo> {
        let mut db = HashMap::new();

        // Add quantum algorithm information
        db.insert(
            "QAOA".to_string(),
            QuantumAlgorithmInfo {
                resource_requirements: HardwareRequirements {
                    min_qubits: 10,
                    connectivity_requirements: ConnectivityRequirement::AllToAll,
                    gate_fidelity_threshold: 0.99,
                    measurement_fidelity_threshold: 0.95,
                    temperature_requirements: Some(0.01), // Kelvin
                },
                performance_model: QuantumPerformanceModel {
                    time_to_solution: Box::new(|n| {
                        Duration::from_millis((n as f64 * n as f64 * 0.1) as u64)
                    }),
                    success_probability: Box::new(|n, error_rate| {
                        (1.0 - error_rate).powf(n as f64)
                    }),
                    resource_scaling: ScalingBehavior {
                        scaling_exponent: 1.0,
                        scaling_constant: 1.0,
                        confidence_interval: (0.8, 1.2),
                        fit_quality: 0.85,
                    },
                },
                noise_sensitivity: NoiseSensitivity {
                    error_thresholds: {
                        let mut map = HashMap::new();
                        map.insert("gate_error".to_string(), 0.001);
                        map.insert("measurement_error".to_string(), 0.01);
                        map
                    },
                    degradation_rates: HashMap::new(),
                    mitigation_effectiveness: HashMap::new(),
                },
                theoretical_advantage: Some(AdvantageEstimate {
                    speedup_factor: 1.5,
                    advantage_type: AdvantageType::Polynomial {
                        degree_reduction: 0.5,
                    },
                    confidence_level: 0.7,
                    conditions: vec![
                        "Low noise".to_string(),
                        "Sufficient circuit depth".to_string(),
                    ],
                }),
            },
        );

        db
    }

    fn build_hardware_database() -> HashMap<String, HardwareCharacteristics> {
        let mut db = HashMap::new();

        // Add hardware characteristics
        db.insert(
            "ideal_quantum".to_string(),
            HardwareCharacteristics {
                qubit_count: 10000,
                connectivity: ConnectivityStructure::FullyConnected,
                gate_fidelities: {
                    let mut map = HashMap::new();
                    map.insert("single_qubit".to_string(), 1.0);
                    map.insert("two_qubit".to_string(), 1.0);
                    map
                },
                measurement_fidelity: 1.0,
                coherence_times: {
                    let mut map = HashMap::new();
                    map.insert("T1".to_string(), Duration::from_secs(1000));
                    map.insert("T2".to_string(), Duration::from_secs(1000));
                    map
                },
                operating_parameters: HashMap::new(),
            },
        );

        db
    }

    pub fn estimate_quantum_performance(
        &self,
        problem: &ProblemCharacteristics,
        algorithms: &[QuantumAlgorithm],
    ) -> Result<HashMap<QuantumAlgorithm, QuantumPerformanceMetrics>, String> {
        let mut results = HashMap::new();

        for algorithm in algorithms {
            let performance = self.estimate_quantum_algorithm_performance(algorithm, problem)?;
            results.insert(algorithm.clone(), performance);
        }

        Ok(results)
    }

    fn estimate_quantum_algorithm_performance(
        &self,
        algorithm: &QuantumAlgorithm,
        problem: &ProblemCharacteristics,
    ) -> Result<QuantumPerformanceMetrics, String> {
        // Estimate quantum-specific requirements
        let qubits_required = match algorithm {
            QuantumAlgorithm::QuantumAnnealing => problem.num_variables,
            QuantumAlgorithm::QAOA => problem.num_variables,
            QuantumAlgorithm::VQE => problem.num_variables,
            QuantumAlgorithm::QFAST => problem.num_variables + 10, // Extra ancilla qubits
            QuantumAlgorithm::QuantumWalk => (problem.num_variables as f64).log2().ceil() as usize,
            QuantumAlgorithm::AdiabaticQuantumComputing => problem.num_variables,
        };

        let circuit_depth = match algorithm {
            QuantumAlgorithm::QuantumAnnealing => 1, // Continuous evolution
            QuantumAlgorithm::QAOA => 10,            // Typical QAOA depth
            QuantumAlgorithm::VQE => 50,
            QuantumAlgorithm::QFAST => 100,
            QuantumAlgorithm::QuantumWalk => 20,
            QuantumAlgorithm::AdiabaticQuantumComputing => 1,
        };

        // Base performance estimation
        let base_metrics = PerformanceMetrics {
            time_complexity: TimeComplexity {
                best_case: ComplexityFunction::Linear,
                average_case: ComplexityFunction::Linearithmic,
                worst_case: ComplexityFunction::Quadratic,
            },
            space_complexity: SpaceComplexity {
                memory_complexity: ComplexityFunction::Linear,
                storage_complexity: ComplexityFunction::Linear,
            },
            solution_quality: QualityMetrics {
                approximation_ratio: 0.95, // Generally better than classical heuristics
                solution_variance: 0.05,
                optimal_probability: 0.3,
                energy_gap: 0.1,
            },
            success_probability: 0.8,
            scaling_behavior: ScalingBehavior {
                scaling_exponent: 1.2,
                scaling_constant: 0.5,
                confidence_interval: (1.0, 1.4),
                fit_quality: 0.75,
            },
        };

        // Quantum-specific metrics
        let quantum_metrics = QuantumSpecificMetrics {
            qubits_required,
            circuit_depth,
            quantum_operations: circuit_depth * qubits_required,
            entanglement_measures: EntanglementMeasures {
                avg_entanglement_entropy: 2.0,
                max_entanglement_entropy: (qubits_required as f64).log2(),
                entanglement_depth: circuit_depth / 5,
                multipartite_measures: HashMap::new(),
            },
            coherence_time_required: Duration::from_micros(circuit_depth as u64 * 10),
        };

        // Hardware requirements
        let hardware_requirements = HardwareRequirements {
            min_qubits: qubits_required,
            connectivity_requirements: ConnectivityRequirement::AllToAll,
            gate_fidelity_threshold: 0.99,
            measurement_fidelity_threshold: 0.95,
            temperature_requirements: Some(0.01),
        };

        // Noise sensitivity
        let noise_sensitivity = NoiseSensitivity {
            error_thresholds: {
                let mut map = HashMap::new();
                map.insert("gate_error".to_string(), 1.0 / circuit_depth as f64);
                map.insert("measurement_error".to_string(), 0.01);
                map
            },
            degradation_rates: HashMap::new(),
            mitigation_effectiveness: HashMap::new(),
        };

        Ok(QuantumPerformanceMetrics {
            base_metrics,
            quantum_metrics,
            hardware_requirements,
            noise_sensitivity,
        })
    }
}

impl Default for QuantumSupremacyBenchmarker {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumSupremacyBenchmarker {
    pub fn new() -> Self {
        Self {
            problem_generators: vec![],
            performance_trackers: HashMap::new(),
        }
    }

    pub fn run_comprehensive_benchmark(
        &mut self,
        config: &AnalysisConfig,
    ) -> Result<SupremacyBenchmarkResult, String> {
        let mut results = Vec::new();

        for size in (config.problem_size_range.0..=config.problem_size_range.1).step_by(50) {
            let benchmark_result = self.run_size_benchmark(size, config)?;
            results.push(benchmark_result);
        }

        Ok(SupremacyBenchmarkResult {
            benchmark_results: results.clone(),
            summary: self.generate_summary(&results)?,
        })
    }

    fn run_size_benchmark(
        &self,
        size: usize,
        _config: &AnalysisConfig,
    ) -> Result<SizeBenchmarkResult, String> {
        // This would run actual benchmarks
        // For now, return simulated results
        Ok(SizeBenchmarkResult {
            problem_size: size,
            classical_times: HashMap::new(),
            quantum_times: HashMap::new(),
            solution_qualities: HashMap::new(),
            advantage_detected: size > 100, // Simplified threshold
        })
    }

    fn generate_summary(
        &self,
        results: &[SizeBenchmarkResult],
    ) -> Result<BenchmarkSummary, String> {
        let advantage_threshold = results
            .iter()
            .find(|r| r.advantage_detected)
            .map_or(usize::MAX, |r| r.problem_size);

        Ok(BenchmarkSummary {
            total_benchmarks: results.len(),
            advantage_threshold,
            confidence_level: 0.8,
            recommendations: vec![
                "Quantum advantage likely for problems > 100 variables".to_string()
            ],
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupremacyBenchmarkResult {
    pub benchmark_results: Vec<SizeBenchmarkResult>,
    pub summary: BenchmarkSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeBenchmarkResult {
    pub problem_size: usize,
    pub classical_times: HashMap<String, Duration>,
    pub quantum_times: HashMap<String, Duration>,
    pub solution_qualities: HashMap<String, f64>,
    pub advantage_detected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub total_benchmarks: usize,
    pub advantage_threshold: usize,
    pub confidence_level: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    JSON,
    Python,
    Rust,
    QUBO,
}

// Additional trait implementations for missing partial comparisons
impl PartialOrd for ComplexityFunction {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use ComplexityFunction::{
            Constant, Cubic, Exponential, Factorial, Linear, Linearithmic, Logarithmic, Polynomial,
            Quadratic,
        };
        match (self, other) {
            (Constant, Constant) => Some(std::cmp::Ordering::Equal),
            (Constant, _) => Some(std::cmp::Ordering::Less),
            (_, Constant) => Some(std::cmp::Ordering::Greater),
            (Logarithmic, Logarithmic) => Some(std::cmp::Ordering::Equal),
            (Logarithmic, _) => Some(std::cmp::Ordering::Less),
            (_, Logarithmic) => Some(std::cmp::Ordering::Greater),
            (Linear, Linear) => Some(std::cmp::Ordering::Equal),
            (Linear, _) => Some(std::cmp::Ordering::Less),
            (_, Linear) => Some(std::cmp::Ordering::Greater),
            (Linearithmic, Linearithmic) => Some(std::cmp::Ordering::Equal),
            (Linearithmic, _) => Some(std::cmp::Ordering::Less),
            (_, Linearithmic) => Some(std::cmp::Ordering::Greater),
            (Quadratic, Quadratic) => Some(std::cmp::Ordering::Equal),
            (Quadratic, _) => Some(std::cmp::Ordering::Less),
            (_, Quadratic) => Some(std::cmp::Ordering::Greater),
            (Cubic, Cubic) => Some(std::cmp::Ordering::Equal),
            (Cubic, _) => Some(std::cmp::Ordering::Less),
            (_, Cubic) => Some(std::cmp::Ordering::Greater),
            (Polynomial { degree: d1 }, Polynomial { degree: d2 }) => d1.partial_cmp(d2),
            (Polynomial { .. }, _) => Some(std::cmp::Ordering::Less),
            (_, Polynomial { .. }) => Some(std::cmp::Ordering::Greater),
            (Exponential { base: b1 }, Exponential { base: b2 }) => b1.partial_cmp(b2),
            (Exponential { .. }, Factorial) => Some(std::cmp::Ordering::Less),
            (Factorial, Exponential { .. }) => Some(std::cmp::Ordering::Greater),
            (Factorial, Factorial) => Some(std::cmp::Ordering::Equal),
        }
    }
}

impl PartialEq for ComplexityFunction {
    fn eq(&self, other: &Self) -> bool {
        matches!(self.partial_cmp(other), Some(std::cmp::Ordering::Equal))
    }
}
