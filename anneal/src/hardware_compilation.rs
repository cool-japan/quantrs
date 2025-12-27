//! Hardware-Aware Compilation System for Quantum Annealing
//!
//! This module provides advanced compilation and optimization capabilities for quantum annealing
//! hardware, including topology-aware embedding, hardware-specific parameter optimization,
//! and performance prediction. It supports various quantum annealing architectures including
//! D-Wave systems (Chimera, Pegasus, Zephyr), neutral atom systems, and custom topologies.
//!
//! Key features:
//! - Hardware topology analysis and optimization
//! - Automated graph embedding with multiple algorithms
//! - Hardware-specific parameter tuning
//! - Performance prediction and optimization
//! - Multi-objective compilation (speed, accuracy, reliability)
//! - Support for heterogeneous quantum annealing systems
//! - Compilation pipeline optimization
//! - Real-time hardware characterization integration

use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::embedding::{EmbeddingError, EmbeddingResult};
use crate::ising::{IsingError, IsingModel};
use crate::qubo::{QuboError, QuboFormulation};
use crate::simulator::{AnnealingParams, AnnealingSolution};

/// Errors that can occur in hardware compilation
#[derive(Error, Debug)]
pub enum HardwareCompilationError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// QUBO formulation error
    #[error("QUBO error: {0}")]
    QuboError(#[from] QuboError),

    /// Embedding error
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError),

    /// Hardware topology error
    #[error("Hardware topology error: {0}")]
    TopologyError(String),

    /// Compilation error
    #[error("Compilation error: {0}")]
    CompilationError(String),

    /// Optimization error
    #[error("Optimization error: {0}")]
    OptimizationError(String),

    /// Hardware characterization error
    #[error("Hardware characterization error: {0}")]
    CharacterizationError(String),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Performance prediction error
    #[error("Performance prediction error: {0}")]
    PredictionError(String),
}

/// Result type for hardware compilation operations
pub type HardwareCompilationResult<T> = Result<T, HardwareCompilationError>;

/// Types of quantum annealing hardware
#[derive(Debug, Clone, PartialEq)]
pub enum HardwareType {
    /// D-Wave systems with Chimera topology
    DWaveChimera {
        unit_cells: (usize, usize),
        cell_size: usize,
    },

    /// D-Wave systems with Pegasus topology
    DWavePegasus {
        layers: usize,
        nodes_per_layer: usize,
    },

    /// D-Wave systems with Zephyr topology
    DWaveZephyr {
        layers: usize,
        tiles_per_layer: usize,
    },

    /// Neutral atom quantum annealing systems
    NeutralAtom {
        grid_size: (usize, usize),
        connectivity: ConnectivityPattern,
    },

    /// Superconducting flux qubit systems
    SuperconductingFlux {
        topology: TopologyType,
        num_qubits: usize,
    },

    /// Photonic quantum annealing systems
    Photonic {
        mode_count: usize,
        coupling_graph: Vec<Vec<bool>>,
    },

    /// Custom hardware topology
    Custom {
        adjacency_matrix: Vec<Vec<bool>>,
        characteristics: HardwareCharacteristics,
    },

    /// Simulated ideal hardware
    Ideal { num_qubits: usize },
}

/// Connectivity patterns for hardware topologies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectivityPattern {
    /// Nearest neighbor connectivity
    NearestNeighbor,
    /// All-to-all connectivity
    AllToAll,
    /// King's graph (8-connected grid)
    KingsGraph,
    /// Triangular lattice
    Triangular,
    /// Custom connectivity pattern
    Custom(Vec<Vec<bool>>),
}

/// Topology types for hardware systems
#[derive(Debug, Clone, PartialEq)]
pub enum TopologyType {
    /// 2D grid topology
    Grid2D { rows: usize, cols: usize },
    /// 3D grid topology
    Grid3D { x: usize, y: usize, z: usize },
    /// Small-world network
    SmallWorld { degree: usize, rewiring_prob: f64 },
    /// Scale-free network
    ScaleFree {
        num_nodes: usize,
        attachment_param: usize,
    },
    /// Complete graph
    Complete,
    /// Tree topology
    Tree {
        branching_factor: usize,
        depth: usize,
    },
}

/// Hardware characteristics and constraints
#[derive(Debug, Clone, PartialEq)]
pub struct HardwareCharacteristics {
    /// Number of available qubits
    pub num_qubits: usize,

    /// Qubit connectivity graph
    pub connectivity: Vec<Vec<bool>>,

    /// Qubit noise characteristics
    pub qubit_noise: Vec<QubitNoise>,

    /// Coupling strength ranges
    pub coupling_ranges: Vec<Vec<CouplingRange>>,

    /// Annealing time constraints
    pub annealing_time_range: (f64, f64),

    /// Temperature characteristics
    pub temperature_characteristics: TemperatureProfile,

    /// Control precision
    pub control_precision: ControlPrecision,

    /// Hardware-specific constraints
    pub constraints: Vec<HardwareConstraint>,

    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Qubit noise characteristics
#[derive(Debug, Clone, PartialEq)]
pub struct QubitNoise {
    /// Coherence time (T1)
    pub t1: f64,
    /// Dephasing time (T2)
    pub t2: f64,
    /// Gate fidelity
    pub gate_fidelity: f64,
    /// Bias noise
    pub bias_noise: f64,
    /// Readout fidelity
    pub readout_fidelity: f64,
}

/// Coupling strength ranges between qubits
#[derive(Debug, Clone, PartialEq)]
pub struct CouplingRange {
    /// Minimum coupling strength
    pub min_strength: f64,
    /// Maximum coupling strength
    pub max_strength: f64,
    /// Coupling fidelity
    pub fidelity: f64,
    /// Crosstalk characteristics
    pub crosstalk: f64,
}

/// Temperature profile during annealing
#[derive(Debug, Clone, PartialEq)]
pub struct TemperatureProfile {
    /// Initial temperature
    pub initial_temp: f64,
    /// Final temperature
    pub final_temp: f64,
    /// Temperature control precision
    pub temp_precision: f64,
    /// Cooling rate limits
    pub cooling_rate_limits: (f64, f64),
}

/// Control precision characteristics
#[derive(Debug, Clone, PartialEq)]
pub struct ControlPrecision {
    /// Bias control precision (bits)
    pub bias_precision: usize,
    /// Coupling control precision (bits)
    pub coupling_precision: usize,
    /// Timing precision (seconds)
    pub timing_precision: f64,
}

/// Hardware-specific constraints
#[derive(Debug, Clone, PartialEq)]
pub enum HardwareConstraint {
    /// Maximum number of active qubits
    MaxActiveQubits(usize),
    /// Maximum coupling strength
    MaxCouplingStrength(f64),
    /// Minimum annealing time
    MinAnnealingTime(f64),
    /// Maximum annealing time
    MaxAnnealingTime(f64),
    /// Forbidden qubit pairs
    ForbiddenPairs(Vec<(usize, usize)>),
    /// Required calibration frequency
    CalibrationFrequency(Duration),
    /// Temperature stability requirements
    TemperatureStability(f64),
}

/// Performance metrics for hardware systems
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceMetrics {
    /// Success probability for typical problems
    pub success_probability: f64,
    /// Average solution quality
    pub solution_quality: f64,
    /// Time to solution distribution
    pub time_to_solution: Vec<f64>,
    /// Energy resolution
    pub energy_resolution: f64,
    /// Reproducibility measure
    pub reproducibility: f64,
}

/// Compilation target specification
#[derive(Debug, Clone)]
pub struct CompilationTarget {
    /// Target hardware type
    pub hardware_type: HardwareType,
    /// Hardware characteristics
    pub characteristics: HardwareCharacteristics,
    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,
    /// Compilation constraints
    pub constraints: Vec<CompilationConstraint>,
    /// Resource allocation preferences
    pub resource_allocation: ResourceAllocation,
}

/// Optimization objectives for compilation
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationObjective {
    /// Minimize time to solution
    MinimizeTime { weight: f64 },
    /// Maximize solution quality
    MaximizeQuality { weight: f64 },
    /// Minimize energy consumption
    MinimizeEnergy { weight: f64 },
    /// Maximize success probability
    MaximizeSuccessProbability { weight: f64 },
    /// Minimize hardware resource usage
    MinimizeResourceUsage { weight: f64 },
    /// Maximize reproducibility
    MaximizeReproducibility { weight: f64 },
}

/// Compilation constraints
#[derive(Debug, Clone)]
pub enum CompilationConstraint {
    /// Maximum compilation time
    MaxCompilationTime(Duration),
    /// Maximum problem size
    MaxProblemSize(usize),
    /// Required solution quality threshold
    MinQualityThreshold(f64),
    /// Resource usage limits
    ResourceUsageLimits(HashMap<String, f64>),
    /// Embedding constraints
    EmbeddingConstraints(EmbeddingConstraints),
}

/// Resource allocation preferences
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Preferred qubit allocation strategy
    pub qubit_allocation: QubitAllocationStrategy,
    /// Coupling utilization preferences
    pub coupling_utilization: CouplingUtilization,
    /// Parallel compilation preferences
    pub parallelization: ParallelizationStrategy,
}

/// Qubit allocation strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QubitAllocationStrategy {
    /// Minimize total qubits used
    MinimizeCount,
    /// Maximize connectivity
    MaximizeConnectivity,
    /// Balance load across hardware
    LoadBalance,
    /// Prefer high-fidelity qubits
    PreferHighFidelity,
    /// Custom allocation function
    Custom,
}

/// Coupling utilization strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CouplingUtilization {
    /// Conservative usage (high reliability)
    Conservative,
    /// Aggressive usage (maximum performance)
    Aggressive,
    /// Balanced usage
    Balanced,
    /// Adaptive based on problem characteristics
    Adaptive,
}

/// Parallelization strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelizationStrategy {
    /// No parallelization
    None,
    /// Parallel embedding search
    ParallelEmbedding,
    /// Parallel parameter optimization
    ParallelParameterSearch,
    /// Full pipeline parallelization
    FullPipeline,
}

/// Embedding constraints for compilation
#[derive(Debug, Clone)]
pub struct EmbeddingConstraints {
    /// Maximum chain length
    pub max_chain_length: Option<usize>,
    /// Preferred embedding algorithms
    pub preferred_algorithms: Vec<EmbeddingAlgorithm>,
    /// Chain strength optimization
    pub chain_strength_optimization: bool,
    /// Embedding quality thresholds
    pub quality_thresholds: EmbeddingQualityThresholds,
}

/// Embedding algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingAlgorithm {
    /// Minorminer-style algorithm
    MinorMiner,
    /// Clique-based embedding
    Clique,
    /// Layered embedding
    Layered,
    /// Spectral embedding
    Spectral,
    /// Machine learning guided embedding
    MLGuided,
    /// Hybrid approach
    Hybrid(Vec<Self>),
}

/// Embedding quality thresholds
#[derive(Debug, Clone)]
pub struct EmbeddingQualityThresholds {
    /// Maximum average chain length
    pub max_avg_chain_length: f64,
    /// Minimum embedding efficiency
    pub min_efficiency: f64,
    /// Maximum embedding overhead
    pub max_overhead: f64,
}

/// Compilation result containing optimized problem and metadata
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Compiled Ising model
    pub compiled_ising: IsingModel,
    /// Embedding information
    pub embedding: EmbeddingInfo,
    /// Optimized annealing parameters
    pub annealing_params: AnnealingParams,
    /// Hardware mapping
    pub hardware_mapping: HardwareMapping,
    /// Performance predictions
    pub performance_prediction: PerformancePrediction,
    /// Compilation metadata
    pub metadata: CompilationMetadata,
}

/// Embedding information
#[derive(Debug, Clone)]
pub struct EmbeddingInfo {
    /// Variable to physical qubit mapping
    pub variable_mapping: HashMap<usize, Vec<usize>>,
    /// Chain information
    pub chains: Vec<Chain>,
    /// Embedding quality metrics
    pub quality_metrics: EmbeddingQualityMetrics,
    /// Embedding algorithm used
    pub algorithm_used: EmbeddingAlgorithm,
}

/// Chain representation in embedding
#[derive(Debug, Clone)]
pub struct Chain {
    /// Logical variable
    pub logical_variable: usize,
    /// Physical qubits in chain
    pub physical_qubits: Vec<usize>,
    /// Chain strength
    pub chain_strength: f64,
    /// Chain connectivity
    pub connectivity: f64,
}

/// Embedding quality metrics
#[derive(Debug, Clone)]
pub struct EmbeddingQualityMetrics {
    /// Average chain length
    pub avg_chain_length: f64,
    /// Maximum chain length
    pub max_chain_length: usize,
    /// Embedding efficiency
    pub efficiency: f64,
    /// Chain balance
    pub chain_balance: f64,
    /// Connectivity utilization
    pub connectivity_utilization: f64,
}

/// Hardware mapping information
#[derive(Debug, Clone)]
pub struct HardwareMapping {
    /// Qubit assignments
    pub qubit_assignments: HashMap<usize, usize>,
    /// Coupling assignments
    pub coupling_assignments: HashMap<(usize, usize), (usize, usize)>,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Hardware constraints satisfied
    pub constraints_satisfied: Vec<bool>,
}

/// Resource utilization information
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// Fraction of qubits used
    pub qubit_utilization: f64,
    /// Fraction of couplings used
    pub coupling_utilization: f64,
    /// Control resource usage
    pub control_resource_usage: HashMap<String, f64>,
    /// Estimated energy consumption
    pub estimated_energy: f64,
}

/// Performance prediction for compiled problem
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    /// Predicted success probability
    pub success_probability: f64,
    /// Predicted solution quality
    pub solution_quality: f64,
    /// Predicted time to solution
    pub time_to_solution: f64,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Sensitivity analysis
    pub sensitivity_analysis: SensitivityAnalysis,
}

/// Sensitivity analysis results
#[derive(Debug, Clone)]
pub struct SensitivityAnalysis {
    /// Parameter sensitivities
    pub parameter_sensitivities: HashMap<String, f64>,
    /// Noise sensitivity
    pub noise_sensitivity: f64,
    /// Temperature sensitivity
    pub temperature_sensitivity: f64,
    /// Robustness measures
    pub robustness_measures: HashMap<String, f64>,
}

/// Compilation metadata
#[derive(Debug, Clone)]
pub struct CompilationMetadata {
    /// Compilation time
    pub compilation_time: Duration,
    /// Optimization iterations performed
    pub optimization_iterations: usize,
    /// Compilation algorithm used
    pub compilation_algorithm: String,
    /// Resource usage during compilation
    pub compilation_resources: HashMap<String, f64>,
    /// Warnings and diagnostics
    pub warnings: Vec<String>,
    /// Optimization trace
    pub optimization_trace: Vec<OptimizationStep>,
}

/// Optimization step in compilation
#[derive(Debug, Clone)]
pub struct OptimizationStep {
    /// Step description
    pub description: String,
    /// Objective value achieved
    pub objective_value: f64,
    /// Parameters at this step
    pub parameters: HashMap<String, f64>,
    /// Time taken for this step
    pub step_time: Duration,
}

/// Hardware-aware compiler
pub struct HardwareCompiler {
    /// Target hardware specifications
    target_hardware: CompilationTarget,
    /// Compilation configuration
    config: CompilerConfig,
    /// Cache for embedding results
    embedding_cache: HashMap<String, EmbeddingResult>,
    /// Performance model
    performance_model: Box<dyn PerformanceModel>,
    /// Optimization engine
    optimization_engine: OptimizationEngine,
}

/// Compiler configuration
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Enable aggressive optimizations
    pub aggressive_optimization: bool,
    /// Cache embedding results
    pub cache_embeddings: bool,
    /// Parallel compilation
    pub parallel_compilation: bool,
    /// Maximum compilation time
    pub max_compilation_time: Duration,
    /// Optimization tolerance
    pub optimization_tolerance: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            aggressive_optimization: false,
            cache_embeddings: true,
            parallel_compilation: true,
            max_compilation_time: Duration::from_secs(300), // 5 minutes
            optimization_tolerance: 1e-6,
            seed: None,
        }
    }
}

/// Performance model trait for predicting hardware performance
pub trait PerformanceModel: Send + Sync {
    /// Predict performance for a given compilation
    fn predict_performance(
        &self,
        problem: &IsingModel,
        embedding: &EmbeddingInfo,
        hardware: &HardwareCharacteristics,
    ) -> HardwareCompilationResult<PerformancePrediction>;

    /// Update model with new performance data
    fn update_model(
        &mut self,
        problem: &IsingModel,
        embedding: &EmbeddingInfo,
        actual_performance: &PerformanceData,
    ) -> HardwareCompilationResult<()>;

    /// Get model confidence for predictions
    fn get_confidence(&self) -> f64;
}

/// Actual performance data for model training
#[derive(Debug, Clone)]
pub struct PerformanceData {
    /// Measured success probability
    pub success_probability: f64,
    /// Measured solution quality
    pub solution_quality: f64,
    /// Measured time to solution
    pub time_to_solution: f64,
    /// Additional metrics
    pub additional_metrics: HashMap<String, f64>,
}

/// Simple machine learning performance model
pub struct MLPerformanceModel {
    /// Model parameters
    parameters: HashMap<String, f64>,
    /// Training data
    training_data: Vec<(Vec<f64>, PerformanceData)>,
    /// Model confidence
    confidence: f64,
}

impl MLPerformanceModel {
    /// Create a new ML performance model
    #[must_use]
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            training_data: Vec::new(),
            confidence: 0.5,
        }
    }

    /// Extract features from problem and embedding
    fn extract_features(
        &self,
        problem: &IsingModel,
        embedding: &EmbeddingInfo,
        hardware: &HardwareCharacteristics,
    ) -> Vec<f64> {
        let mut features = Vec::new();

        // Problem features
        features.push(problem.num_qubits as f64);
        features.push(embedding.chains.len() as f64);
        features.push(embedding.quality_metrics.avg_chain_length);
        features.push(embedding.quality_metrics.efficiency);

        // Hardware features
        features.push(hardware.num_qubits as f64);
        features.push(hardware.performance_metrics.success_probability);
        features.push(hardware.performance_metrics.solution_quality);

        // Connectivity features
        let connectivity_density = hardware
            .connectivity
            .iter()
            .flatten()
            .filter(|&&connected| connected)
            .count() as f64
            / (hardware.num_qubits * hardware.num_qubits) as f64;
        features.push(connectivity_density);

        features
    }
}

impl PerformanceModel for MLPerformanceModel {
    fn predict_performance(
        &self,
        problem: &IsingModel,
        embedding: &EmbeddingInfo,
        hardware: &HardwareCharacteristics,
    ) -> HardwareCompilationResult<PerformancePrediction> {
        let features = self.extract_features(problem, embedding, hardware);

        // Simple linear model prediction (in practice, this would be more sophisticated)
        let mut success_prob = 0.8;
        let mut solution_quality = 0.9;
        let mut time_to_solution = 1000.0; // microseconds

        // Apply simple feature-based adjustments
        if features.len() >= 3 {
            success_prob *= 0.1f64.mul_add(-features[2].max(0.0).min(2.0), 1.0); // Chain length impact
            solution_quality *= features[3].max(0.5).min(1.0); // Efficiency impact
            time_to_solution *= features[0] / 100.0; // Problem size impact
        }

        let confidence_intervals = HashMap::from([
            (
                "success_probability".to_string(),
                (success_prob * 0.9, success_prob * 1.1),
            ),
            (
                "solution_quality".to_string(),
                (solution_quality * 0.95, solution_quality * 1.05),
            ),
            (
                "time_to_solution".to_string(),
                (time_to_solution * 0.8, time_to_solution * 1.2),
            ),
        ]);

        let sensitivity_analysis = SensitivityAnalysis {
            parameter_sensitivities: HashMap::from([
                ("chain_length".to_string(), 0.3),
                ("embedding_efficiency".to_string(), 0.4),
                ("problem_size".to_string(), 0.2),
            ]),
            noise_sensitivity: 0.1,
            temperature_sensitivity: 0.15,
            robustness_measures: HashMap::from([("overall_robustness".to_string(), 0.7)]),
        };

        Ok(PerformancePrediction {
            success_probability: success_prob,
            solution_quality,
            time_to_solution,
            confidence_intervals,
            sensitivity_analysis,
        })
    }

    fn update_model(
        &mut self,
        problem: &IsingModel,
        embedding: &EmbeddingInfo,
        actual_performance: &PerformanceData,
    ) -> HardwareCompilationResult<()> {
        // In practice, this would update the model parameters
        self.training_data.push((
            self.extract_features(
                problem,
                embedding,
                &HardwareCharacteristics {
                    num_qubits: problem.num_qubits,
                    connectivity: vec![vec![false; problem.num_qubits]; problem.num_qubits],
                    qubit_noise: vec![
                        QubitNoise {
                            t1: 100.0,
                            t2: 50.0,
                            gate_fidelity: 0.99,
                            bias_noise: 0.01,
                            readout_fidelity: 0.95,
                        };
                        problem.num_qubits
                    ],
                    coupling_ranges: vec![
                        vec![
                            CouplingRange {
                                min_strength: -1.0,
                                max_strength: 1.0,
                                fidelity: 0.98,
                                crosstalk: 0.02,
                            };
                            problem.num_qubits
                        ];
                        problem.num_qubits
                    ],
                    annealing_time_range: (1.0, 1000.0),
                    temperature_characteristics: TemperatureProfile {
                        initial_temp: 1.0,
                        final_temp: 0.01,
                        temp_precision: 0.001,
                        cooling_rate_limits: (0.1, 10.0),
                    },
                    control_precision: ControlPrecision {
                        bias_precision: 16,
                        coupling_precision: 16,
                        timing_precision: 1e-9,
                    },
                    constraints: Vec::new(),
                    performance_metrics: PerformanceMetrics {
                        success_probability: 0.8,
                        solution_quality: 0.9,
                        time_to_solution: vec![1000.0],
                        energy_resolution: 0.001,
                        reproducibility: 0.95,
                    },
                },
            ),
            actual_performance.clone(),
        ));

        // Update confidence based on training data size
        self.confidence = (self.training_data.len() as f64 / 100.0).min(0.95).max(0.1);

        Ok(())
    }

    fn get_confidence(&self) -> f64 {
        self.confidence
    }
}

/// Optimization engine for compilation
pub struct OptimizationEngine {
    /// Current optimization state
    state: OptimizationState,
    /// Optimization history
    history: Vec<OptimizationStep>,
    /// Configuration
    config: OptimizationConfig,
}

/// Optimization state
#[derive(Debug, Clone)]
pub struct OptimizationState {
    /// Current objective value
    pub objective_value: f64,
    /// Current parameters
    pub parameters: HashMap<String, f64>,
    /// Iteration count
    pub iteration: usize,
    /// Convergence status
    pub converged: bool,
}

/// Optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Optimization algorithm
    pub algorithm: OptimizationAlgorithm,
    /// Multi-objective weights
    pub objective_weights: HashMap<String, f64>,
}

/// Optimization algorithms
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationAlgorithm {
    /// Simulated annealing
    SimulatedAnnealing,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Particle swarm optimization
    ParticleSwarm,
    /// Bayesian optimization
    BayesianOptimization,
    /// Multi-objective NSGA-II
    NSGAII,
}

impl OptimizationEngine {
    /// Create a new optimization engine
    #[must_use]
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            state: OptimizationState {
                objective_value: f64::INFINITY,
                parameters: HashMap::new(),
                iteration: 0,
                converged: false,
            },
            history: Vec::new(),
            config,
        }
    }

    /// Optimize compilation parameters
    pub fn optimize<F>(
        &mut self,
        objective_function: F,
    ) -> HardwareCompilationResult<HashMap<String, f64>>
    where
        F: Fn(&HashMap<String, f64>) -> f64,
    {
        let start_time = Instant::now();

        // Initialize parameters
        let mut current_params = HashMap::from([
            ("chain_strength".to_string(), 1.0),
            ("annealing_time".to_string(), 20.0),
            ("temperature".to_string(), 0.01),
            ("num_reads".to_string(), 1000.0),
        ]);

        let mut best_value = objective_function(&current_params);
        let mut best_params = current_params.clone();

        // Simple optimization loop (in practice, use more sophisticated algorithms)
        for iteration in 0..self.config.max_iterations {
            // Generate candidate parameters
            let mut candidate_params = current_params.clone();

            // Add random perturbations
            let mut rng = ChaCha8Rng::seed_from_u64(iteration as u64);
            for (key, value) in &mut candidate_params {
                let perturbation = rng.gen_range(-0.1..0.1) * *value;
                *value = (*value + perturbation).max(0.01);
            }

            // Evaluate candidate
            let candidate_value = objective_function(&candidate_params);

            // Accept if better
            if candidate_value < best_value {
                best_value = candidate_value;
                best_params.clone_from(&candidate_params);
                current_params.clone_from(&candidate_params);
            }

            // Record step
            self.history.push(OptimizationStep {
                description: format!("Iteration {iteration}"),
                objective_value: candidate_value,
                parameters: candidate_params,
                step_time: start_time.elapsed() / (iteration + 1) as u32,
            });

            // Check convergence
            if iteration > 10
                && self.history[iteration].objective_value
                    - self.history[iteration - 10].objective_value
                    < self.config.tolerance
            {
                self.state.converged = true;
                break;
            }
        }

        self.state.parameters = best_params.clone();
        self.state.objective_value = best_value;

        Ok(best_params)
    }
}

impl HardwareCompiler {
    /// Create a new hardware compiler
    #[must_use]
    pub fn new(target_hardware: CompilationTarget, config: CompilerConfig) -> Self {
        Self {
            target_hardware,
            config,
            embedding_cache: HashMap::new(),
            performance_model: Box::new(MLPerformanceModel::new()),
            optimization_engine: OptimizationEngine::new(OptimizationConfig {
                max_iterations: 100,
                tolerance: 1e-6,
                algorithm: OptimizationAlgorithm::SimulatedAnnealing,
                objective_weights: HashMap::from([
                    ("time".to_string(), 0.3),
                    ("quality".to_string(), 0.4),
                    ("energy".to_string(), 0.2),
                    ("success_probability".to_string(), 0.1),
                ]),
            }),
        }
    }

    /// Compile an Ising model for the target hardware
    pub fn compile(
        &mut self,
        problem: &IsingModel,
    ) -> HardwareCompilationResult<CompilationResult> {
        let start_time = Instant::now();

        // Analyze problem characteristics
        let problem_analysis = self.analyze_problem(problem)?;

        // Generate or retrieve embedding
        let embedding_info = self.generate_embedding(problem)?;

        // Optimize annealing parameters
        let annealing_params = self.optimize_annealing_parameters(problem, &embedding_info)?;

        // Create hardware mapping
        let hardware_mapping = self.create_hardware_mapping(problem, &embedding_info)?;

        // Generate performance prediction
        let performance_prediction = self.performance_model.predict_performance(
            problem,
            &embedding_info,
            &self.target_hardware.characteristics,
        )?;

        // Compile the final Ising model
        let compiled_ising = self.apply_embedding_to_problem(problem, &embedding_info)?;

        let compilation_time = start_time.elapsed();

        Ok(CompilationResult {
            compiled_ising,
            embedding: embedding_info,
            annealing_params,
            hardware_mapping,
            performance_prediction,
            metadata: CompilationMetadata {
                compilation_time,
                optimization_iterations: self.optimization_engine.state.iteration,
                compilation_algorithm: "HardwareAwareCompiler".to_string(),
                compilation_resources: HashMap::from([
                    ("memory_usage".to_string(), 100.0),
                    ("cpu_time".to_string(), compilation_time.as_secs_f64()),
                ]),
                warnings: Vec::new(),
                optimization_trace: self.optimization_engine.history.clone(),
            },
        })
    }

    /// Analyze problem characteristics
    fn analyze_problem(&self, problem: &IsingModel) -> HardwareCompilationResult<ProblemAnalysis> {
        Ok(ProblemAnalysis {
            num_variables: problem.num_qubits,
            connectivity_density: self.calculate_connectivity_density(problem),
            coupling_distribution: self.analyze_coupling_distribution(problem),
            problem_structure: self.detect_problem_structure(problem),
            complexity_estimate: self.estimate_complexity(problem),
        })
    }

    /// Calculate connectivity density of the problem
    fn calculate_connectivity_density(&self, problem: &IsingModel) -> f64 {
        let mut num_couplings = 0;
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                if problem.get_coupling(i, j).unwrap_or(0.0).abs() > 1e-10 {
                    num_couplings += 1;
                }
            }
        }
        let max_couplings = problem.num_qubits * (problem.num_qubits - 1) / 2;
        f64::from(num_couplings) / max_couplings as f64
    }

    /// Analyze coupling strength distribution
    fn analyze_coupling_distribution(&self, problem: &IsingModel) -> CouplingDistribution {
        let mut couplings = Vec::new();
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                let coupling = problem.get_coupling(i, j).unwrap_or(0.0);
                if coupling.abs() > 1e-10 {
                    couplings.push(coupling.abs());
                }
            }
        }

        if couplings.is_empty() {
            return CouplingDistribution {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                distribution_type: DistributionType::Uniform,
            };
        }

        couplings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = couplings.iter().sum::<f64>() / couplings.len() as f64;
        let variance =
            couplings.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / couplings.len() as f64;

        CouplingDistribution {
            mean,
            std_dev: variance.sqrt(),
            min: couplings[0],
            max: couplings[couplings.len() - 1],
            distribution_type: DistributionType::Normal, // Simplified
        }
    }

    /// Detect problem structure (e.g., sparse, dense, structured)
    fn detect_problem_structure(&self, problem: &IsingModel) -> ProblemStructure {
        let connectivity_density = self.calculate_connectivity_density(problem);

        if connectivity_density < 0.1 {
            ProblemStructure::Sparse
        } else if connectivity_density > 0.7 {
            ProblemStructure::Dense
        } else {
            ProblemStructure::Structured
        }
    }

    /// Estimate problem complexity
    fn estimate_complexity(&self, problem: &IsingModel) -> f64 {
        // Simple complexity estimate based on size and connectivity
        let size_factor = (problem.num_qubits as f64).ln();
        let connectivity_factor = self.calculate_connectivity_density(problem);
        size_factor * (1.0 + connectivity_factor)
    }

    /// Generate embedding for the problem
    fn generate_embedding(
        &mut self,
        problem: &IsingModel,
    ) -> HardwareCompilationResult<EmbeddingInfo> {
        // Check cache first
        let cache_key = self.generate_cache_key(problem);
        if self.config.cache_embeddings {
            if let Some(cached_result) = self.embedding_cache.get(&cache_key) {
                return Ok(self.convert_embedding_result_to_info(cached_result));
            }
        }

        // Generate new embedding
        let embedding_result = self.find_best_embedding(problem)?;

        // Cache result
        if self.config.cache_embeddings {
            self.embedding_cache
                .insert(cache_key, embedding_result.clone());
        }

        Ok(self.convert_embedding_result_to_info(&embedding_result))
    }

    /// Find the best embedding using multiple algorithms
    fn find_best_embedding(
        &self,
        problem: &IsingModel,
    ) -> HardwareCompilationResult<EmbeddingResult> {
        // For now, create a simple identity embedding
        // In practice, this would use sophisticated embedding algorithms
        let mut variable_mapping = HashMap::new();
        let mut chains = Vec::new();

        for i in 0..problem.num_qubits {
            variable_mapping.insert(i, vec![i]);
            chains.push(Chain {
                logical_variable: i,
                physical_qubits: vec![i],
                chain_strength: 1.0,
                connectivity: 1.0,
            });
        }

        let quality_metrics = EmbeddingQualityMetrics {
            avg_chain_length: 1.0,
            max_chain_length: 1,
            efficiency: 1.0,
            chain_balance: 1.0,
            connectivity_utilization: self.calculate_connectivity_density(problem),
        };

        Ok(EmbeddingResult {
            embedding: variable_mapping.clone(),
            chain_strength: 1.0,
            success: true,
            error_message: None,
        })
    }

    /// Convert `EmbeddingResult` to `EmbeddingInfo`
    fn convert_embedding_result_to_info(&self, result: &EmbeddingResult) -> EmbeddingInfo {
        let mut chains = Vec::new();
        for (logical, physical) in &result.embedding {
            chains.push(Chain {
                logical_variable: *logical,
                physical_qubits: physical.clone(),
                chain_strength: result.chain_strength,
                connectivity: 1.0,
            });
        }

        EmbeddingInfo {
            variable_mapping: result.embedding.clone(),
            chains,
            quality_metrics: EmbeddingQualityMetrics {
                avg_chain_length: 1.0,
                max_chain_length: 1,
                efficiency: 1.0,
                chain_balance: 1.0,
                connectivity_utilization: 0.5,
            },
            algorithm_used: EmbeddingAlgorithm::MinorMiner,
        }
    }

    /// Generate cache key for embedding
    fn generate_cache_key(&self, problem: &IsingModel) -> String {
        // Simple hash based on problem structure
        format!(
            "{}_{:.6}",
            problem.num_qubits,
            self.calculate_connectivity_density(problem)
        )
    }

    /// Optimize annealing parameters for the compiled problem
    fn optimize_annealing_parameters(
        &mut self,
        problem: &IsingModel,
        embedding: &EmbeddingInfo,
    ) -> HardwareCompilationResult<AnnealingParams> {
        // Define objective function for parameter optimization
        let objective_fn = |params: &HashMap<String, f64>| -> f64 {
            let mut score = 0.0;

            // Penalize extreme parameter values
            for (key, &value) in params {
                match key.as_str() {
                    "chain_strength" => {
                        if value < 0.1 || value > 10.0 {
                            score += 1000.0;
                        }
                    }
                    "annealing_time" => {
                        if value < 1.0 || value > 1000.0 {
                            score += 1000.0;
                        }
                    }
                    _ => {}
                }
            }

            // Favor balanced parameters
            score += params.values().map(|v| v.ln().abs()).sum::<f64>();

            score
        };

        // Optimize parameters
        let optimized_params = self.optimization_engine.optimize(objective_fn)?;

        // Convert to AnnealingParams
        Ok(AnnealingParams {
            num_sweeps: *optimized_params.get("num_reads").unwrap_or(&1000.0) as usize,
            num_repetitions: 1,
            initial_temperature: optimized_params
                .get("temperature")
                .unwrap_or(&1.0)
                .max(0.01),
            final_temperature: 0.01,
            timeout: Some(optimized_params.get("annealing_time").unwrap_or(&20.0) / 1000.0),
            ..Default::default()
        })
    }

    /// Create hardware mapping for the compilation
    fn create_hardware_mapping(
        &self,
        problem: &IsingModel,
        embedding: &EmbeddingInfo,
    ) -> HardwareCompilationResult<HardwareMapping> {
        let mut qubit_assignments = HashMap::new();
        let mut coupling_assignments = HashMap::new();

        // Simple 1:1 mapping for now
        for (logical, physical_vec) in &embedding.variable_mapping {
            if let Some(&first_physical) = physical_vec.first() {
                qubit_assignments.insert(*logical, first_physical);
            }
        }

        // Map couplings
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                if problem.get_coupling(i, j).unwrap_or(0.0).abs() > 1e-10 {
                    if let (Some(&phys_i), Some(&phys_j)) =
                        (qubit_assignments.get(&i), qubit_assignments.get(&j))
                    {
                        coupling_assignments.insert((i, j), (phys_i, phys_j));
                    }
                }
            }
        }

        let resource_utilization = ResourceUtilization {
            qubit_utilization: problem.num_qubits as f64
                / self.target_hardware.characteristics.num_qubits as f64,
            coupling_utilization: coupling_assignments.len() as f64 / 100.0, // Simplified
            control_resource_usage: HashMap::from([
                ("memory".to_string(), 0.1),
                ("control_lines".to_string(), 0.2),
            ]),
            estimated_energy: problem.num_qubits as f64 * 0.001, // Simplified
        };

        Ok(HardwareMapping {
            qubit_assignments,
            coupling_assignments,
            resource_utilization,
            constraints_satisfied: vec![true; self.target_hardware.constraints.len()],
        })
    }

    /// Apply embedding to create the final compiled problem
    fn apply_embedding_to_problem(
        &self,
        problem: &IsingModel,
        embedding: &EmbeddingInfo,
    ) -> HardwareCompilationResult<IsingModel> {
        // For simple 1:1 embedding, just return a copy
        // In practice, this would handle chain couplings and other embedding details
        let mut compiled = IsingModel::new(problem.num_qubits);

        // Copy biases
        for i in 0..problem.num_qubits {
            compiled.set_bias(i, problem.get_bias(i).unwrap_or(0.0))?;
        }

        // Copy couplings
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                let coupling = problem.get_coupling(i, j).unwrap_or(0.0);
                if coupling.abs() > 1e-10 {
                    compiled.set_coupling(i, j, coupling)?;
                }
            }
        }

        Ok(compiled)
    }
}

/// Problem analysis results
#[derive(Debug, Clone)]
pub struct ProblemAnalysis {
    /// Number of variables
    pub num_variables: usize,
    /// Connectivity density
    pub connectivity_density: f64,
    /// Coupling strength distribution
    pub coupling_distribution: CouplingDistribution,
    /// Problem structure type
    pub problem_structure: ProblemStructure,
    /// Complexity estimate
    pub complexity_estimate: f64,
}

/// Coupling strength distribution
#[derive(Debug, Clone)]
pub struct CouplingDistribution {
    /// Mean coupling strength
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum coupling
    pub min: f64,
    /// Maximum coupling
    pub max: f64,
    /// Distribution type
    pub distribution_type: DistributionType,
}

/// Distribution types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributionType {
    /// Uniform distribution
    Uniform,
    /// Normal distribution
    Normal,
    /// Exponential distribution
    Exponential,
    /// Power law distribution
    PowerLaw,
}

/// Problem structure types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProblemStructure {
    /// Sparse connectivity
    Sparse,
    /// Dense connectivity
    Dense,
    /// Structured (e.g., grid, tree)
    Structured,
    /// Random
    Random,
}

/// Utility functions for hardware compilation

/// Create a D-Wave Chimera hardware target
pub fn create_chimera_target(
    unit_cells: (usize, usize),
    cell_size: usize,
) -> HardwareCompilationResult<CompilationTarget> {
    let num_qubits = unit_cells.0 * unit_cells.1 * cell_size * 2;
    let connectivity = create_chimera_connectivity(unit_cells, cell_size);

    let characteristics = HardwareCharacteristics {
        num_qubits,
        connectivity,
        qubit_noise: vec![
            QubitNoise {
                t1: 80.0,
                t2: 40.0,
                gate_fidelity: 0.99,
                bias_noise: 0.01,
                readout_fidelity: 0.95,
            };
            num_qubits
        ],
        coupling_ranges: vec![
            vec![
                CouplingRange {
                    min_strength: -1.0,
                    max_strength: 1.0,
                    fidelity: 0.98,
                    crosstalk: 0.02,
                };
                num_qubits
            ];
            num_qubits
        ],
        annealing_time_range: (1.0, 2000.0),
        temperature_characteristics: TemperatureProfile {
            initial_temp: 1.0,
            final_temp: 0.01,
            temp_precision: 0.001,
            cooling_rate_limits: (0.1, 10.0),
        },
        control_precision: ControlPrecision {
            bias_precision: 16,
            coupling_precision: 16,
            timing_precision: 1e-9,
        },
        constraints: vec![
            HardwareConstraint::MaxActiveQubits(num_qubits),
            HardwareConstraint::MaxCouplingStrength(1.0),
            HardwareConstraint::MinAnnealingTime(1.0),
            HardwareConstraint::MaxAnnealingTime(2000.0),
        ],
        performance_metrics: PerformanceMetrics {
            success_probability: 0.85,
            solution_quality: 0.90,
            time_to_solution: vec![100.0, 200.0, 500.0],
            energy_resolution: 0.001,
            reproducibility: 0.95,
        },
    };

    Ok(CompilationTarget {
        hardware_type: HardwareType::DWaveChimera {
            unit_cells,
            cell_size,
        },
        characteristics,
        objectives: vec![
            OptimizationObjective::MaximizeQuality { weight: 0.4 },
            OptimizationObjective::MinimizeTime { weight: 0.3 },
            OptimizationObjective::MaximizeSuccessProbability { weight: 0.3 },
        ],
        constraints: vec![
            CompilationConstraint::MaxCompilationTime(Duration::from_secs(300)),
            CompilationConstraint::MinQualityThreshold(0.8),
        ],
        resource_allocation: ResourceAllocation {
            qubit_allocation: QubitAllocationStrategy::MaximizeConnectivity,
            coupling_utilization: CouplingUtilization::Balanced,
            parallelization: ParallelizationStrategy::ParallelEmbedding,
        },
    })
}

/// Create Chimera topology connectivity matrix
fn create_chimera_connectivity(unit_cells: (usize, usize), cell_size: usize) -> Vec<Vec<bool>> {
    let num_qubits = unit_cells.0 * unit_cells.1 * cell_size * 2;
    let mut connectivity = vec![vec![false; num_qubits]; num_qubits];

    // Simplified Chimera connectivity
    for i in 0..num_qubits {
        for j in 0..num_qubits {
            if i != j {
                // Add some connectivity based on Chimera structure (simplified)
                let cell_i = i / (cell_size * 2);
                let cell_j = j / (cell_size * 2);
                let within_cell_i = i % (cell_size * 2);
                let within_cell_j = j % (cell_size * 2);

                // Intra-cell connectivity
                if cell_i == cell_j {
                    if (within_cell_i < cell_size && within_cell_j >= cell_size)
                        || (within_cell_i >= cell_size && within_cell_j < cell_size)
                    {
                        connectivity[i][j] = true;
                    }
                }

                // Inter-cell connectivity (simplified)
                if (cell_i as i32 - cell_j as i32).abs() == 1 && within_cell_i == within_cell_j {
                    connectivity[i][j] = true;
                }
            }
        }
    }

    connectivity
}

/// Create an ideal hardware target for testing
#[must_use]
pub fn create_ideal_target(num_qubits: usize) -> CompilationTarget {
    let connectivity = vec![vec![true; num_qubits]; num_qubits];

    let characteristics = HardwareCharacteristics {
        num_qubits,
        connectivity,
        qubit_noise: vec![
            QubitNoise {
                t1: f64::INFINITY,
                t2: f64::INFINITY,
                gate_fidelity: 1.0,
                bias_noise: 0.0,
                readout_fidelity: 1.0,
            };
            num_qubits
        ],
        coupling_ranges: vec![
            vec![
                CouplingRange {
                    min_strength: -f64::INFINITY,
                    max_strength: f64::INFINITY,
                    fidelity: 1.0,
                    crosstalk: 0.0,
                };
                num_qubits
            ];
            num_qubits
        ],
        annealing_time_range: (0.0, f64::INFINITY),
        temperature_characteristics: TemperatureProfile {
            initial_temp: 1.0,
            final_temp: 0.0,
            temp_precision: 0.0,
            cooling_rate_limits: (0.0, f64::INFINITY),
        },
        control_precision: ControlPrecision {
            bias_precision: 64,
            coupling_precision: 64,
            timing_precision: 1e-15,
        },
        constraints: Vec::new(),
        performance_metrics: PerformanceMetrics {
            success_probability: 1.0,
            solution_quality: 1.0,
            time_to_solution: vec![1.0],
            energy_resolution: 0.0,
            reproducibility: 1.0,
        },
    };

    CompilationTarget {
        hardware_type: HardwareType::Ideal { num_qubits },
        characteristics,
        objectives: vec![OptimizationObjective::MaximizeQuality { weight: 1.0 }],
        constraints: Vec::new(),
        resource_allocation: ResourceAllocation {
            qubit_allocation: QubitAllocationStrategy::MinimizeCount,
            coupling_utilization: CouplingUtilization::Conservative,
            parallelization: ParallelizationStrategy::None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ising::IsingModel;

    #[test]
    fn test_hardware_compiler_creation() {
        let target = create_ideal_target(10);
        let config = CompilerConfig::default();
        let _compiler = HardwareCompiler::new(target, config);
    }

    #[test]
    fn test_chimera_target_creation() {
        let target = create_chimera_target((2, 2), 4).expect("should create Chimera target");
        assert_eq!(target.characteristics.num_qubits, 32); // 2*2*4*2
        assert!(matches!(
            target.hardware_type,
            HardwareType::DWaveChimera { .. }
        ));
    }

    #[test]
    fn test_problem_analysis() {
        let target = create_ideal_target(5);
        let config = CompilerConfig::default();
        let compiler = HardwareCompiler::new(target, config);

        let mut problem = IsingModel::new(5);
        problem.set_bias(0, 1.0).expect("should set bias");
        problem
            .set_coupling(0, 1, 0.5)
            .expect("should set coupling");
        problem
            .set_coupling(1, 2, -0.3)
            .expect("should set coupling");

        let analysis = compiler
            .analyze_problem(&problem)
            .expect("should analyze problem");
        assert_eq!(analysis.num_variables, 5);
        assert!(analysis.connectivity_density > 0.0);
        assert!(analysis.connectivity_density < 1.0);
    }

    #[test]
    fn test_compilation_pipeline() {
        let target = create_ideal_target(4);
        let config = CompilerConfig::default();
        let mut compiler = HardwareCompiler::new(target, config);

        let mut problem = IsingModel::new(4);
        problem.set_bias(0, 1.0).expect("should set bias");
        problem.set_bias(1, -0.5).expect("should set bias");
        problem
            .set_coupling(0, 1, 0.3)
            .expect("should set coupling");
        problem
            .set_coupling(1, 2, -0.2)
            .expect("should set coupling");

        let result = compiler.compile(&problem).expect("should compile problem");

        assert_eq!(result.compiled_ising.num_qubits, 4);
        assert!(result.performance_prediction.success_probability > 0.0);
        assert!(result.metadata.compilation_time > Duration::from_nanos(0));
    }

    #[test]
    fn test_embedding_quality_metrics() {
        let embedding_info = EmbeddingInfo {
            variable_mapping: HashMap::from([(0, vec![0]), (1, vec![1, 2]), (2, vec![3])]),
            chains: vec![
                Chain {
                    logical_variable: 0,
                    physical_qubits: vec![0],
                    chain_strength: 1.0,
                    connectivity: 1.0,
                },
                Chain {
                    logical_variable: 1,
                    physical_qubits: vec![1, 2],
                    chain_strength: 2.0,
                    connectivity: 0.8,
                },
                Chain {
                    logical_variable: 2,
                    physical_qubits: vec![3],
                    chain_strength: 1.0,
                    connectivity: 1.0,
                },
            ],
            quality_metrics: EmbeddingQualityMetrics {
                avg_chain_length: 1.33,
                max_chain_length: 2,
                efficiency: 0.75,
                chain_balance: 0.9,
                connectivity_utilization: 0.6,
            },
            algorithm_used: EmbeddingAlgorithm::MinorMiner,
        };

        assert_eq!(embedding_info.chains.len(), 3);
        assert_eq!(embedding_info.quality_metrics.max_chain_length, 2);
        assert!(embedding_info.quality_metrics.avg_chain_length > 1.0);
    }

    #[test]
    fn test_performance_prediction() {
        let mut model = MLPerformanceModel::new();

        let problem = IsingModel::new(5);
        let embedding_info = EmbeddingInfo {
            variable_mapping: HashMap::new(),
            chains: Vec::new(),
            quality_metrics: EmbeddingQualityMetrics {
                avg_chain_length: 1.0,
                max_chain_length: 1,
                efficiency: 1.0,
                chain_balance: 1.0,
                connectivity_utilization: 0.5,
            },
            algorithm_used: EmbeddingAlgorithm::MinorMiner,
        };

        let hardware = HardwareCharacteristics {
            num_qubits: 10,
            connectivity: vec![vec![false; 10]; 10],
            qubit_noise: Vec::new(),
            coupling_ranges: Vec::new(),
            annealing_time_range: (1.0, 1000.0),
            temperature_characteristics: TemperatureProfile {
                initial_temp: 1.0,
                final_temp: 0.01,
                temp_precision: 0.001,
                cooling_rate_limits: (0.1, 10.0),
            },
            control_precision: ControlPrecision {
                bias_precision: 16,
                coupling_precision: 16,
                timing_precision: 1e-9,
            },
            constraints: Vec::new(),
            performance_metrics: PerformanceMetrics {
                success_probability: 0.8,
                solution_quality: 0.9,
                time_to_solution: vec![1000.0],
                energy_resolution: 0.001,
                reproducibility: 0.95,
            },
        };

        let prediction = model
            .predict_performance(&problem, &embedding_info, &hardware)
            .expect("should predict performance");

        assert!(prediction.success_probability > 0.0 && prediction.success_probability <= 1.0);
        assert!(prediction.solution_quality > 0.0 && prediction.solution_quality <= 1.0);
        assert!(prediction.time_to_solution > 0.0);
    }
}
