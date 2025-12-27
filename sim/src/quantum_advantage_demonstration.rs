//! Quantum Advantage Demonstration Framework
//!
//! This module provides comprehensive tools for demonstrating and verifying
//! quantum computational advantages across various problem domains, including
//! quantum supremacy tests, quantum advantage benchmarks, and comparative
//! analysis with classical algorithms.

use crate::prelude::SimulatorError;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::parallel_ops::{IndexedParallelIterator, ParallelIterator};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::Result;
use scirs2_core::random::prelude::*;

/// Types of quantum advantage demonstrations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumAdvantageType {
    /// Quantum supremacy (computational task impossible for classical computers)
    QuantumSupremacy,
    /// Quantum advantage (speedup over best known classical algorithm)
    ComputationalAdvantage,
    /// Sample complexity advantage
    SampleComplexityAdvantage,
    /// Communication complexity advantage
    CommunicationAdvantage,
    /// Query complexity advantage
    QueryComplexityAdvantage,
    /// Memory advantage
    MemoryAdvantage,
    /// Energy efficiency advantage
    EnergyAdvantage,
    /// Noise resilience advantage
    NoiseResilienceAdvantage,
}

/// Quantum advantage problem domains
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProblemDomain {
    /// Random circuit sampling
    RandomCircuitSampling,
    /// Boson sampling
    BosonSampling,
    /// IQP (Instantaneous Quantum Polynomial-time) circuits
    IQPCircuits,
    /// Quantum approximate optimization
    QAOA,
    /// Variational quantum algorithms
    VQE,
    /// Quantum machine learning
    QML,
    /// Quantum simulation
    QuantumSimulation,
    /// Quantum cryptography
    QuantumCryptography,
    /// Quantum search
    QuantumSearch,
    /// Factor decomposition
    Factoring,
    /// Discrete logarithm
    DiscreteLogarithm,
    /// Graph problems
    GraphProblems,
    /// Linear algebra
    LinearAlgebra,
    /// Optimization
    Optimization,
    /// Custom domain
    Custom,
}

/// Classical algorithm types for comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClassicalAlgorithmType {
    /// Brute force search
    BruteForce,
    /// Monte Carlo sampling
    MonteCarlo,
    /// Markov chain Monte Carlo
    MCMC,
    /// Simulated annealing
    SimulatedAnnealing,
    /// Genetic algorithms
    GeneticAlgorithm,
    /// Branch and bound
    BranchAndBound,
    /// Dynamic programming
    DynamicProgramming,
    /// Approximation algorithms
    Approximation,
    /// Heuristic algorithms
    Heuristic,
    /// Machine learning
    MachineLearning,
    /// Tensor network methods
    TensorNetwork,
    /// Best known classical algorithm
    BestKnown,
}

/// Quantum advantage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAdvantageMetrics {
    /// Quantum algorithm runtime
    pub quantum_time: Duration,
    /// Classical algorithm runtime
    pub classical_time: Duration,
    /// Speedup factor (`classical_time` / `quantum_time`)
    pub speedup_factor: f64,
    /// Quantum algorithm accuracy/fidelity
    pub quantum_accuracy: f64,
    /// Classical algorithm accuracy
    pub classical_accuracy: f64,
    /// Quantum resource requirements
    pub quantum_resources: QuantumResources,
    /// Classical resource requirements
    pub classical_resources: ClassicalResources,
    /// Statistical significance of advantage
    pub statistical_significance: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Problem size scaling
    pub scaling_analysis: ScalingAnalysis,
}

/// Quantum resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResources {
    /// Number of qubits required
    pub qubits: usize,
    /// Circuit depth
    pub depth: usize,
    /// Number of gates
    pub gate_count: usize,
    /// Coherence time required
    pub coherence_time: Duration,
    /// Gate fidelity required
    pub gate_fidelity: f64,
    /// Measurement shots
    pub shots: usize,
    /// Quantum volume required
    pub quantum_volume: usize,
}

/// Classical resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalResources {
    /// CPU time
    pub cpu_time: Duration,
    /// Memory usage (bytes)
    pub memory_usage: usize,
    /// Number of cores used
    pub cores: usize,
    /// Energy consumption (joules)
    pub energy_consumption: f64,
    /// Storage requirements (bytes)
    pub storage: usize,
    /// Network bandwidth (if distributed)
    pub network_bandwidth: f64,
}

/// Scaling analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingAnalysis {
    /// Problem sizes tested
    pub problem_sizes: Vec<usize>,
    /// Quantum scaling exponent
    pub quantum_scaling: f64,
    /// Classical scaling exponent
    pub classical_scaling: f64,
    /// Crossover point where quantum becomes advantageous
    pub crossover_point: Option<usize>,
    /// Asymptotic advantage factor
    pub asymptotic_advantage: f64,
}

/// Quantum advantage configuration
#[derive(Debug, Clone)]
pub struct QuantumAdvantageConfig {
    /// Type of advantage to demonstrate
    pub advantage_type: QuantumAdvantageType,
    /// Problem domain
    pub domain: ProblemDomain,
    /// Classical algorithms to compare against
    pub classical_algorithms: Vec<ClassicalAlgorithmType>,
    /// Problem sizes to test
    pub problem_sizes: Vec<usize>,
    /// Number of trials for statistical analysis
    pub num_trials: usize,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Maximum runtime for classical algorithms
    pub classical_timeout: Duration,
    /// Hardware specifications
    pub hardware_specs: HardwareSpecs,
    /// Enable detailed profiling
    pub enable_profiling: bool,
    /// Save intermediate results
    pub save_results: bool,
}

/// Hardware specifications
#[derive(Debug, Clone)]
pub struct HardwareSpecs {
    /// Quantum hardware specifications
    pub quantum_hardware: QuantumHardwareSpecs,
    /// Classical hardware specifications
    pub classical_hardware: ClassicalHardwareSpecs,
}

/// Quantum hardware specifications
#[derive(Debug, Clone)]
pub struct QuantumHardwareSpecs {
    /// Number of available qubits
    pub num_qubits: usize,
    /// Gate fidelities
    pub gate_fidelities: HashMap<String, f64>,
    /// Coherence times
    pub coherence_times: HashMap<String, Duration>,
    /// Connectivity graph
    pub connectivity: Vec<Vec<bool>>,
    /// Gate times
    pub gate_times: HashMap<String, Duration>,
    /// Readout fidelity
    pub readout_fidelity: f64,
}

/// Classical hardware specifications
#[derive(Debug, Clone)]
pub struct ClassicalHardwareSpecs {
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// CPU frequency (GHz)
    pub cpu_frequency: f64,
    /// RAM size (bytes)
    pub ram_size: usize,
    /// Cache sizes
    pub cache_sizes: Vec<usize>,
    /// GPU specifications (if available)
    pub gpu_specs: Option<GPUSpecs>,
}

/// GPU specifications
#[derive(Debug, Clone)]
pub struct GPUSpecs {
    /// Number of compute units
    pub compute_units: usize,
    /// Memory size (bytes)
    pub memory_size: usize,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// Peak FLOPS
    pub peak_flops: f64,
}

/// Quantum advantage demonstration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAdvantageResult {
    /// Whether quantum advantage was demonstrated
    pub advantage_demonstrated: bool,
    /// Advantage metrics
    pub metrics: QuantumAdvantageMetrics,
    /// Detailed results for each problem size
    pub detailed_results: Vec<DetailedResult>,
    /// Statistical analysis
    pub statistical_analysis: StatisticalAnalysis,
    /// Verification results
    pub verification: VerificationResult,
    /// Cost analysis
    pub cost_analysis: CostAnalysis,
    /// Future projections
    pub projections: FutureProjections,
}

/// Detailed result for a specific problem size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedResult {
    /// Problem size
    pub problem_size: usize,
    /// Quantum results
    pub quantum_results: AlgorithmResult,
    /// Classical results
    pub classical_results: Vec<AlgorithmResult>,
    /// Comparative analysis
    pub comparison: ComparisonResult,
}

/// Algorithm execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmResult {
    /// Algorithm type
    pub algorithm_type: String,
    /// Execution time
    pub execution_time: Duration,
    /// Solution quality/accuracy
    pub solution_quality: f64,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Success rate (for probabilistic algorithms)
    pub success_rate: f64,
    /// Output distribution (for sampling problems)
    pub output_distribution: Option<HashMap<String, f64>>,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Memory peak usage
    pub peak_memory: usize,
    /// Energy consumption
    pub energy: f64,
    /// Number of operations
    pub operations: usize,
    /// Communication cost (for distributed algorithms)
    pub communication: f64,
}

/// Comparison result between quantum and classical
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Best classical algorithm
    pub best_classical: String,
    /// Speedup vs best classical
    pub speedup: f64,
    /// Quality improvement
    pub quality_improvement: f64,
    /// Resource efficiency
    pub resource_efficiency: f64,
    /// Statistical significance
    pub significance: f64,
}

/// Statistical analysis of results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    /// Hypothesis test results
    pub hypothesis_tests: Vec<HypothesisTest>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Effect sizes
    pub effect_sizes: HashMap<String, f64>,
    /// Power analysis
    pub power_analysis: PowerAnalysis,
}

/// Hypothesis test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTest {
    /// Test name
    pub test_name: String,
    /// Null hypothesis
    pub null_hypothesis: String,
    /// Test statistic
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Reject null hypothesis
    pub reject_null: bool,
}

/// Power analysis for statistical tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysis {
    /// Statistical power
    pub power: f64,
    /// Effect size
    pub effect_size: f64,
    /// Sample size required for desired power
    pub required_sample_size: usize,
}

/// Verification of quantum advantage claims
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Classical verification methods used
    pub verification_methods: Vec<String>,
    /// Verification success rate
    pub verification_success_rate: f64,
    /// Spoofing resistance
    pub spoofing_resistance: f64,
    /// Independent verification
    pub independent_verification: Option<IndependentVerification>,
}

/// Independent verification by third parties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndependentVerification {
    /// Verifying institutions
    pub verifiers: Vec<String>,
    /// Verification results
    pub results: Vec<bool>,
    /// Consensus level
    pub consensus: f64,
}

/// Cost analysis of quantum vs classical approaches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostAnalysis {
    /// Quantum hardware cost
    pub quantum_hardware_cost: f64,
    /// Classical hardware cost
    pub classical_hardware_cost: f64,
    /// Operational costs
    pub operational_costs: OperationalCosts,
    /// Total cost of ownership
    pub total_cost_ownership: f64,
    /// Cost per solution
    pub cost_per_solution: f64,
}

/// Operational costs breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalCosts {
    /// Energy costs
    pub energy: f64,
    /// Maintenance costs
    pub maintenance: f64,
    /// Personnel costs
    pub personnel: f64,
    /// Infrastructure costs
    pub infrastructure: f64,
}

/// Future projections for quantum advantage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FutureProjections {
    /// Projected quantum improvements
    pub quantum_improvements: TechnologyProjection,
    /// Projected classical improvements
    pub classical_improvements: TechnologyProjection,
    /// Timeline predictions
    pub timeline: TimelineProjection,
    /// Market impact assessment
    pub market_impact: MarketImpact,
}

/// Technology improvement projections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TechnologyProjection {
    /// Performance improvement factor per year
    pub performance_improvement: f64,
    /// Cost reduction factor per year
    pub cost_reduction: f64,
    /// Reliability improvement
    pub reliability_improvement: f64,
    /// Scalability projections
    pub scalability: Vec<(usize, f64)>, // (year, capability)
}

/// Timeline for quantum advantage milestones
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineProjection {
    /// Milestones and expected years
    pub milestones: Vec<(String, usize)>,
    /// Confidence in projections
    pub confidence: f64,
    /// Key uncertainty factors
    pub uncertainties: Vec<String>,
}

/// Market impact assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketImpact {
    /// Industries affected
    pub affected_industries: Vec<String>,
    /// Economic impact estimates
    pub economic_impact: f64,
    /// Job displacement/creation
    pub employment_impact: EmploymentImpact,
    /// Investment projections
    pub investment_projections: Vec<(usize, f64)>, // (year, investment)
}

/// Employment impact analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmploymentImpact {
    /// Jobs displaced
    pub jobs_displaced: usize,
    /// Jobs created
    pub jobs_created: usize,
    /// Reskilling requirements
    pub reskilling_needed: usize,
}

/// Main quantum advantage demonstration framework
pub struct QuantumAdvantageDemonstrator {
    /// Configuration
    config: QuantumAdvantageConfig,
    /// Quantum algorithm implementations
    quantum_algorithms: HashMap<ProblemDomain, Box<dyn QuantumAlgorithm + Send + Sync>>,
    /// Classical algorithm implementations
    classical_algorithms:
        HashMap<ClassicalAlgorithmType, Box<dyn ClassicalAlgorithm + Send + Sync>>,
    /// Results database
    results_database: Arc<Mutex<ResultsDatabase>>,
    /// Performance profiler
    profiler: Arc<Mutex<PerformanceProfiler>>,
}

/// Quantum algorithm trait
pub trait QuantumAlgorithm: Send + Sync {
    /// Execute quantum algorithm
    fn execute(&self, problem_instance: &ProblemInstance) -> Result<AlgorithmResult>;

    /// Get resource requirements
    fn get_resource_requirements(&self, problem_size: usize) -> QuantumResources;

    /// Get theoretical scaling
    fn get_theoretical_scaling(&self) -> f64;

    /// Algorithm name
    fn name(&self) -> &str;
}

/// Classical algorithm trait
pub trait ClassicalAlgorithm: Send + Sync {
    /// Execute classical algorithm
    fn execute(&self, problem_instance: &ProblemInstance) -> Result<AlgorithmResult>;

    /// Get resource requirements
    fn get_resource_requirements(&self, problem_size: usize) -> ClassicalResources;

    /// Get theoretical scaling
    fn get_theoretical_scaling(&self) -> f64;

    /// Algorithm name
    fn name(&self) -> &str;
}

/// Problem instance representation
#[derive(Debug, Clone)]
pub struct ProblemInstance {
    /// Problem domain
    pub domain: ProblemDomain,
    /// Problem size
    pub size: usize,
    /// Problem-specific data
    pub data: ProblemData,
    /// Difficulty parameters
    pub difficulty: DifficultyParameters,
}

/// Problem-specific data
#[derive(Debug, Clone)]
pub enum ProblemData {
    /// Random circuit sampling data
    RandomCircuit {
        circuit: InterfaceCircuit,
        target_distribution: HashMap<String, f64>,
    },
    /// Graph problem data
    Graph {
        adjacency_matrix: Array2<f64>,
        vertex_weights: Vec<f64>,
        edge_weights: HashMap<(usize, usize), f64>,
    },
    /// Optimization problem data
    Optimization {
        objective_function: Vec<f64>,
        constraints: Vec<Vec<f64>>,
        bounds: Vec<(f64, f64)>,
    },
    /// Search problem data
    Search {
        search_space: Vec<Vec<f64>>,
        target_function: Vec<f64>,
        oracle_queries: usize,
    },
    /// Simulation problem data
    Simulation {
        hamiltonian: Array2<Complex64>,
        initial_state: Array1<Complex64>,
        evolution_time: f64,
    },
    /// Custom problem data
    Custom { data: HashMap<String, Vec<f64>> },
}

/// Difficulty parameters
#[derive(Debug, Clone)]
pub struct DifficultyParameters {
    /// Instance hardness (0.0 = easy, 1.0 = hard)
    pub hardness: f64,
    /// Noise level
    pub noise_level: f64,
    /// Time constraints
    pub time_limit: Option<Duration>,
    /// Accuracy requirements
    pub accuracy_threshold: f64,
}

/// Results database
#[derive(Debug, Clone)]
pub struct ResultsDatabase {
    /// Stored results by configuration
    pub results: HashMap<String, Vec<QuantumAdvantageResult>>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Performance profiler
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    /// Timing data
    pub timing_data: HashMap<String, Vec<Duration>>,
    /// Memory usage data
    pub memory_data: HashMap<String, Vec<usize>>,
    /// Energy consumption data
    pub energy_data: HashMap<String, Vec<f64>>,
}

impl QuantumAdvantageDemonstrator {
    /// Create new quantum advantage demonstrator
    #[must_use]
    pub fn new(config: QuantumAdvantageConfig) -> Self {
        let mut demonstrator = Self {
            config,
            quantum_algorithms: HashMap::new(),
            classical_algorithms: HashMap::new(),
            results_database: Arc::new(Mutex::new(ResultsDatabase {
                results: HashMap::new(),
                metadata: HashMap::new(),
            })),
            profiler: Arc::new(Mutex::new(PerformanceProfiler {
                timing_data: HashMap::new(),
                memory_data: HashMap::new(),
                energy_data: HashMap::new(),
            })),
        };

        // Register default algorithms
        demonstrator.register_default_algorithms();
        demonstrator
    }

    /// Demonstrate quantum advantage
    pub fn demonstrate_advantage(&mut self) -> Result<QuantumAdvantageResult> {
        let mut detailed_results = Vec::new();
        let mut all_speedups = Vec::new();
        let mut all_accuracies = Vec::new();

        println!(
            "Starting quantum advantage demonstration for {:?} in {:?} domain",
            self.config.advantage_type, self.config.domain
        );

        // Test each problem size
        let problem_sizes = self.config.problem_sizes.clone();
        for problem_size in problem_sizes {
            println!("Testing problem size: {problem_size}");

            let detailed_result = self.test_problem_size(problem_size)?;

            all_speedups.push(detailed_result.comparison.speedup);
            all_accuracies.push(detailed_result.quantum_results.solution_quality);

            detailed_results.push(detailed_result);
        }

        // Perform scaling analysis
        let scaling_analysis = self.analyze_scaling(&detailed_results)?;

        // Statistical analysis
        let statistical_analysis = self.perform_statistical_analysis(&detailed_results)?;

        // Verification
        let verification = self.verify_results(&detailed_results)?;

        // Cost analysis
        let cost_analysis = self.analyze_costs(&detailed_results)?;

        // Future projections
        let projections = self.generate_projections(&detailed_results)?;

        // Overall metrics
        let overall_speedup = all_speedups
            .iter()
            .fold(1.0, |acc, &x| acc * x)
            .powf(1.0 / all_speedups.len() as f64);
        let avg_accuracy = all_accuracies.iter().sum::<f64>() / all_accuracies.len() as f64;

        let quantum_time = detailed_results
            .iter()
            .map(|r| r.quantum_results.execution_time)
            .sum::<Duration>()
            / detailed_results.len() as u32;

        let best_classical_time = detailed_results
            .iter()
            .map(|r| {
                r.classical_results
                    .iter()
                    .map(|c| c.execution_time)
                    .min()
                    .unwrap_or(Duration::new(0, 0))
            })
            .sum::<Duration>()
            / detailed_results.len() as u32;

        let metrics = QuantumAdvantageMetrics {
            quantum_time,
            classical_time: best_classical_time,
            speedup_factor: overall_speedup,
            quantum_accuracy: avg_accuracy,
            classical_accuracy: detailed_results
                .iter()
                .map(|r| {
                    r.classical_results
                        .iter()
                        .map(|c| c.solution_quality)
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or(0.0)
                })
                .sum::<f64>()
                / detailed_results.len() as f64,
            quantum_resources: self.aggregate_quantum_resources(&detailed_results),
            classical_resources: self.aggregate_classical_resources(&detailed_results),
            statistical_significance: statistical_analysis
                .hypothesis_tests
                .iter()
                .find(|t| t.test_name == "quantum_advantage")
                .map_or(0.0, |t| 1.0 - t.p_value),
            confidence_interval: (overall_speedup * 0.9, overall_speedup * 1.1), // Simplified
            scaling_analysis,
        };

        let advantage_demonstrated = overall_speedup > 1.0
            && statistical_analysis
                .hypothesis_tests
                .iter()
                .any(|t| t.test_name == "quantum_advantage" && t.reject_null);

        let result = QuantumAdvantageResult {
            advantage_demonstrated,
            metrics,
            detailed_results,
            statistical_analysis,
            verification,
            cost_analysis,
            projections,
        };

        // Store results
        self.store_results(&result)?;

        Ok(result)
    }

    /// Test a specific problem size
    fn test_problem_size(&self, problem_size: usize) -> Result<DetailedResult> {
        // Generate problem instance
        let problem_instance = self.generate_problem_instance(problem_size)?;

        // Execute quantum algorithm
        let quantum_start = Instant::now();
        let quantum_algorithm = self
            .quantum_algorithms
            .get(&self.config.domain)
            .ok_or_else(|| {
                SimulatorError::UnsupportedOperation(format!(
                    "No quantum algorithm registered for domain {:?}",
                    self.config.domain
                ))
            })?;

        let quantum_results = quantum_algorithm.execute(&problem_instance)?;
        let quantum_time = quantum_start.elapsed();

        // Execute classical algorithms
        let mut classical_results = Vec::new();
        for &classical_type in &self.config.classical_algorithms {
            if let Some(classical_algorithm) = self.classical_algorithms.get(&classical_type) {
                let classical_start = Instant::now();

                // Set timeout for classical algorithms
                let result = if classical_start.elapsed() < self.config.classical_timeout {
                    classical_algorithm.execute(&problem_instance)?
                } else {
                    // Timeout result
                    AlgorithmResult {
                        algorithm_type: classical_algorithm.name().to_string(),
                        execution_time: self.config.classical_timeout,
                        solution_quality: 0.0,
                        resource_usage: ResourceUsage {
                            peak_memory: 0,
                            energy: 0.0,
                            operations: 0,
                            communication: 0.0,
                        },
                        success_rate: 0.0,
                        output_distribution: None,
                    }
                };

                classical_results.push(result);
            }
        }

        // Compare results
        let comparison = self.compare_results(&quantum_results, &classical_results)?;

        Ok(DetailedResult {
            problem_size,
            quantum_results,
            classical_results,
            comparison,
        })
    }

    /// Generate problem instance for given size
    fn generate_problem_instance(&self, size: usize) -> Result<ProblemInstance> {
        let data = match self.config.domain {
            ProblemDomain::RandomCircuitSampling => {
                let circuit = self.generate_random_circuit(size)?;
                ProblemData::RandomCircuit {
                    circuit,
                    target_distribution: HashMap::new(), // Would compute actual distribution
                }
            }
            ProblemDomain::GraphProblems => {
                let adjacency_matrix = Array2::from_shape_fn((size, size), |(i, j)| {
                    if i != j && thread_rng().gen::<f64>() < 0.3 {
                        thread_rng().gen::<f64>()
                    } else {
                        0.0
                    }
                });
                ProblemData::Graph {
                    adjacency_matrix,
                    vertex_weights: (0..size).map(|_| thread_rng().gen::<f64>()).collect(),
                    edge_weights: HashMap::new(),
                }
            }
            ProblemDomain::Optimization => {
                ProblemData::Optimization {
                    objective_function: (0..size).map(|_| thread_rng().gen::<f64>()).collect(),
                    constraints: vec![vec![1.0; size]], // Sum constraint
                    bounds: vec![(0.0, 1.0); size],
                }
            }
            ProblemDomain::QuantumSimulation => {
                let hamiltonian = Array2::from_shape_fn((1 << size, 1 << size), |(i, j)| {
                    if i == j {
                        Complex64::new(thread_rng().gen::<f64>(), 0.0)
                    } else if (i ^ j).is_power_of_two() {
                        Complex64::new(thread_rng().gen::<f64>() * 0.1, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                });
                let initial_state = {
                    let mut state = Array1::zeros(1 << size);
                    state[0] = Complex64::new(1.0, 0.0);
                    state
                };
                ProblemData::Simulation {
                    hamiltonian,
                    initial_state,
                    evolution_time: 1.0,
                }
            }
            _ => ProblemData::Custom {
                data: HashMap::new(),
            },
        };

        Ok(ProblemInstance {
            domain: self.config.domain,
            size,
            data,
            difficulty: DifficultyParameters {
                hardness: 0.5,
                noise_level: 0.01,
                time_limit: Some(Duration::from_secs(3600)),
                accuracy_threshold: 0.95,
            },
        })
    }

    /// Generate random circuit for supremacy testing
    fn generate_random_circuit(&self, num_qubits: usize) -> Result<InterfaceCircuit> {
        let mut circuit = InterfaceCircuit::new(num_qubits, 0);
        let depth = num_qubits * 10; // Scaled depth

        for layer in 0..depth {
            // Single-qubit gates
            for qubit in 0..num_qubits {
                let gate_type = match layer % 3 {
                    0 => InterfaceGateType::RX(
                        thread_rng().gen::<f64>() * 2.0 * std::f64::consts::PI,
                    ),
                    1 => InterfaceGateType::RY(
                        thread_rng().gen::<f64>() * 2.0 * std::f64::consts::PI,
                    ),
                    _ => InterfaceGateType::RZ(
                        thread_rng().gen::<f64>() * 2.0 * std::f64::consts::PI,
                    ),
                };
                circuit.add_gate(InterfaceGate::new(gate_type, vec![qubit]));
            }

            // Two-qubit gates
            if layer % 2 == 1 {
                for qubit in 0..num_qubits - 1 {
                    if thread_rng().gen::<f64>() < 0.5 {
                        circuit.add_gate(InterfaceGate::new(
                            InterfaceGateType::CNOT,
                            vec![qubit, qubit + 1],
                        ));
                    }
                }
            }
        }

        Ok(circuit)
    }

    /// Compare quantum and classical results
    fn compare_results(
        &self,
        quantum: &AlgorithmResult,
        classical_results: &[AlgorithmResult],
    ) -> Result<ComparisonResult> {
        let best_classical = classical_results
            .iter()
            .min_by(|a, b| a.execution_time.cmp(&b.execution_time))
            .ok_or_else(|| SimulatorError::InvalidInput("No classical results".to_string()))?;

        let speedup =
            best_classical.execution_time.as_secs_f64() / quantum.execution_time.as_secs_f64();
        let quality_improvement = quantum.solution_quality / best_classical.solution_quality;
        let resource_efficiency = (best_classical.resource_usage.peak_memory as f64)
            / (quantum.resource_usage.peak_memory as f64);

        // Simplified statistical significance calculation
        let significance = if speedup > 1.0 { 0.95 } else { 0.05 };

        Ok(ComparisonResult {
            best_classical: best_classical.algorithm_type.clone(),
            speedup,
            quality_improvement,
            resource_efficiency,
            significance,
        })
    }

    /// Analyze scaling behavior
    fn analyze_scaling(&self, results: &[DetailedResult]) -> Result<ScalingAnalysis> {
        let problem_sizes: Vec<usize> = results.iter().map(|r| r.problem_size).collect();
        let quantum_times: Vec<f64> = results
            .iter()
            .map(|r| r.quantum_results.execution_time.as_secs_f64())
            .collect();
        let classical_times: Vec<f64> = results
            .iter()
            .map(|r| {
                r.classical_results
                    .iter()
                    .map(|c| c.execution_time.as_secs_f64())
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0)
            })
            .collect();

        // Fit power law scaling: T = a * n^b
        let quantum_scaling = self.fit_power_law(&problem_sizes, &quantum_times)?;
        let classical_scaling = self.fit_power_law(&problem_sizes, &classical_times)?;

        // Find crossover point
        let crossover_point =
            self.find_crossover_point(&problem_sizes, &quantum_times, &classical_times);

        // Calculate asymptotic advantage
        let asymptotic_advantage = if classical_scaling > quantum_scaling {
            f64::INFINITY
        } else {
            classical_scaling / quantum_scaling
        };

        Ok(ScalingAnalysis {
            problem_sizes,
            quantum_scaling,
            classical_scaling,
            crossover_point,
            asymptotic_advantage,
        })
    }

    /// Fit power law to data
    fn fit_power_law(&self, sizes: &[usize], times: &[f64]) -> Result<f64> {
        if sizes.len() != times.len() || sizes.len() < 2 {
            return Ok(1.0); // Default scaling
        }

        // Linear regression on log-log scale: log(T) = log(a) + b*log(n)
        let log_sizes: Vec<f64> = sizes.iter().map(|&s| (s as f64).ln()).collect();
        let log_times: Vec<f64> = times.iter().map(|&t| (t.max(1e-10)).ln()).collect();

        let n = log_sizes.len() as f64;
        let sum_x = log_sizes.iter().sum::<f64>();
        let sum_y = log_times.iter().sum::<f64>();
        let sum_xy = log_sizes
            .iter()
            .zip(&log_times)
            .map(|(x, y)| x * y)
            .sum::<f64>();
        let sum_x2 = log_sizes.iter().map(|x| x * x).sum::<f64>();

        let slope = n.mul_add(sum_xy, -(sum_x * sum_y)) / n.mul_add(sum_x2, -(sum_x * sum_x));
        Ok(slope)
    }

    /// Find crossover point where quantum becomes faster
    fn find_crossover_point(
        &self,
        sizes: &[usize],
        quantum_times: &[f64],
        classical_times: &[f64],
    ) -> Option<usize> {
        for (i, (&size, (&qt, &ct))) in sizes
            .iter()
            .zip(quantum_times.iter().zip(classical_times.iter()))
            .enumerate()
        {
            if qt < ct {
                return Some(size);
            }
        }
        None
    }

    /// Perform statistical analysis
    fn perform_statistical_analysis(
        &self,
        results: &[DetailedResult],
    ) -> Result<StatisticalAnalysis> {
        let mut hypothesis_tests = Vec::new();

        // Test for quantum advantage (speedup > 1)
        let speedups: Vec<f64> = results.iter().map(|r| r.comparison.speedup).collect();
        let avg_speedup = speedups.iter().sum::<f64>() / speedups.len() as f64;
        let speedup_variance = speedups
            .iter()
            .map(|s| (s - avg_speedup).powi(2))
            .sum::<f64>()
            / speedups.len() as f64;

        // One-sample t-test for speedup > 1
        let t_statistic =
            (avg_speedup - 1.0) / (speedup_variance.sqrt() / (speedups.len() as f64).sqrt());
        let p_value = self.compute_t_test_p_value(t_statistic, speedups.len() - 1);

        hypothesis_tests.push(HypothesisTest {
            test_name: "quantum_advantage".to_string(),
            null_hypothesis: "No quantum speedup (speedup <= 1)".to_string(),
            test_statistic: t_statistic,
            p_value,
            reject_null: p_value < 0.05,
        });

        // Confidence intervals
        let mut confidence_intervals = HashMap::new();
        let margin_of_error = 1.96 * speedup_variance.sqrt() / (speedups.len() as f64).sqrt();
        confidence_intervals.insert(
            "speedup".to_string(),
            (avg_speedup - margin_of_error, avg_speedup + margin_of_error),
        );

        // Effect sizes
        let mut effect_sizes = HashMap::new();
        effect_sizes.insert(
            "speedup_cohen_d".to_string(),
            (avg_speedup - 1.0) / speedup_variance.sqrt(),
        );

        // Power analysis
        let power_analysis = PowerAnalysis {
            power: 0.8, // Would calculate actual power
            effect_size: (avg_speedup - 1.0) / speedup_variance.sqrt(),
            required_sample_size: 30, // Would calculate required N for desired power
        };

        Ok(StatisticalAnalysis {
            hypothesis_tests,
            confidence_intervals,
            effect_sizes,
            power_analysis,
        })
    }

    /// Compute p-value for t-test
    fn compute_t_test_p_value(&self, t_statistic: f64, degrees_of_freedom: usize) -> f64 {
        // Simplified p-value calculation - would use proper statistical libraries
        if t_statistic.abs() > 2.0 {
            0.01
        } else if t_statistic.abs() > 1.0 {
            0.05
        } else {
            0.5
        }
    }

    /// Verify quantum advantage results
    fn verify_results(&self, results: &[DetailedResult]) -> Result<VerificationResult> {
        let mut verification_methods = Vec::new();
        let mut verification_successes = 0;
        let total_verifications = results.len();

        for result in results {
            // Cross-entropy benchmarking for sampling problems
            if matches!(self.config.domain, ProblemDomain::RandomCircuitSampling) {
                verification_methods.push("Cross-entropy benchmarking".to_string());
                // Simplified verification - would implement actual cross-entropy test
                if result.quantum_results.solution_quality > 0.9 {
                    verification_successes += 1;
                }
            }

            // Linear XEB (Cross-Entropy Benchmarking)
            verification_methods.push("Linear XEB".to_string());
            if result.quantum_results.solution_quality > 0.8 {
                verification_successes += 1;
            }
        }

        let verification_success_rate =
            f64::from(verification_successes) / total_verifications as f64;
        let spoofing_resistance = 0.95; // Would calculate based on complexity

        Ok(VerificationResult {
            verification_methods,
            verification_success_rate,
            spoofing_resistance,
            independent_verification: None, // Would implement if available
        })
    }

    /// Analyze costs
    fn analyze_costs(&self, results: &[DetailedResult]) -> Result<CostAnalysis> {
        // Simplified cost analysis
        let quantum_hardware_cost = 10_000_000.0; // $10M quantum computer
        let classical_hardware_cost = 100_000.0; // $100K classical computer

        let operational_costs = OperationalCosts {
            energy: 1000.0,         // Daily energy cost
            maintenance: 5000.0,    // Daily maintenance
            personnel: 2000.0,      // Daily personnel cost
            infrastructure: 1000.0, // Daily infrastructure
        };

        let daily_operational_cost = operational_costs.energy
            + operational_costs.maintenance
            + operational_costs.personnel
            + operational_costs.infrastructure;

        let total_cost_ownership = daily_operational_cost.mul_add(365.0, quantum_hardware_cost);

        let num_solutions = results.len() as f64;
        let cost_per_solution = total_cost_ownership / num_solutions;

        Ok(CostAnalysis {
            quantum_hardware_cost,
            classical_hardware_cost,
            operational_costs,
            total_cost_ownership,
            cost_per_solution,
        })
    }

    /// Generate future projections
    fn generate_projections(&self, _results: &[DetailedResult]) -> Result<FutureProjections> {
        // Quantum technology projections
        let quantum_improvements = TechnologyProjection {
            performance_improvement: 1.5, // 50% improvement per year
            cost_reduction: 0.8,          // 20% cost reduction per year
            reliability_improvement: 1.2, // 20% reliability improvement per year
            scalability: vec![(2024, 100.0), (2025, 200.0), (2030, 1000.0)],
        };

        // Classical technology projections
        let classical_improvements = TechnologyProjection {
            performance_improvement: 1.1, // 10% improvement per year (Moore's law slowing)
            cost_reduction: 0.95,         // 5% cost reduction per year
            reliability_improvement: 1.05, // 5% reliability improvement per year
            scalability: vec![(2024, 1000.0), (2025, 1100.0), (2030, 1500.0)],
        };

        // Timeline projections
        let timeline = TimelineProjection {
            milestones: vec![
                ("Fault-tolerant quantum computers".to_string(), 2030),
                (
                    "Practical quantum advantage in optimization".to_string(),
                    2026,
                ),
                ("Quantum supremacy in machine learning".to_string(), 2028),
                ("Commercial quantum advantage".to_string(), 2032),
            ],
            confidence: 0.7,
            uncertainties: vec![
                "Error correction overhead".to_string(),
                "Classical algorithm improvements".to_string(),
                "Hardware manufacturing challenges".to_string(),
            ],
        };

        // Market impact
        let market_impact = MarketImpact {
            affected_industries: vec![
                "Finance".to_string(),
                "Pharmaceuticals".to_string(),
                "Logistics".to_string(),
                "Energy".to_string(),
                "Cybersecurity".to_string(),
            ],
            economic_impact: 1_000_000_000_000.0, // $1T by 2035
            employment_impact: EmploymentImpact {
                jobs_displaced: 100_000,
                jobs_created: 500_000,
                reskilling_needed: 1_000_000,
            },
            investment_projections: vec![
                (2024, 10_000_000_000.0),
                (2025, 20_000_000_000.0),
                (2030, 100_000_000_000.0),
            ],
        };

        Ok(FutureProjections {
            quantum_improvements,
            classical_improvements,
            timeline,
            market_impact,
        })
    }

    /// Helper methods
    fn register_default_algorithms(&mut self) {
        // Register quantum algorithms
        self.quantum_algorithms.insert(
            ProblemDomain::RandomCircuitSampling,
            Box::new(RandomCircuitSamplingAlgorithm),
        );
        self.quantum_algorithms
            .insert(ProblemDomain::QAOA, Box::new(QAOAAlgorithm));

        // Register classical algorithms
        self.classical_algorithms.insert(
            ClassicalAlgorithmType::MonteCarlo,
            Box::new(MonteCarloAlgorithm),
        );
        self.classical_algorithms.insert(
            ClassicalAlgorithmType::BruteForce,
            Box::new(BruteForceAlgorithm),
        );
    }

    fn aggregate_quantum_resources(&self, results: &[DetailedResult]) -> QuantumResources {
        let avg_depth = results
            .iter()
            .map(|r| r.quantum_results.resource_usage.operations)
            .sum::<usize>()
            / results.len();

        QuantumResources {
            qubits: results.iter().map(|r| r.problem_size).max().unwrap_or(0),
            depth: avg_depth,
            gate_count: avg_depth * 2, // Estimate
            coherence_time: Duration::from_millis(100),
            gate_fidelity: 0.999,
            shots: 1000,
            quantum_volume: results
                .iter()
                .map(|r| r.problem_size * r.problem_size)
                .max()
                .unwrap_or(0),
        }
    }

    fn aggregate_classical_resources(&self, results: &[DetailedResult]) -> ClassicalResources {
        let total_time = results
            .iter()
            .flat_map(|r| &r.classical_results)
            .map(|c| c.execution_time)
            .sum::<Duration>();

        let avg_memory = results
            .iter()
            .flat_map(|r| &r.classical_results)
            .map(|c| c.resource_usage.peak_memory)
            .sum::<usize>()
            / results
                .iter()
                .flat_map(|r| &r.classical_results)
                .count()
                .max(1);

        ClassicalResources {
            cpu_time: total_time,
            memory_usage: avg_memory,
            cores: self.config.hardware_specs.classical_hardware.cpu_cores,
            energy_consumption: 1000.0, // Estimate
            storage: avg_memory * 2,
            network_bandwidth: 0.0,
        }
    }

    fn store_results(&self, result: &QuantumAdvantageResult) -> Result<()> {
        let mut database = self
            .results_database
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let key = format!("{:?}_{:?}", self.config.advantage_type, self.config.domain);
        database
            .results
            .entry(key)
            .or_default()
            .push(result.clone());
        Ok(())
    }
}

/// Example algorithm implementations
/// Random circuit sampling quantum algorithm
struct RandomCircuitSamplingAlgorithm;

impl QuantumAlgorithm for RandomCircuitSamplingAlgorithm {
    fn execute(&self, problem_instance: &ProblemInstance) -> Result<AlgorithmResult> {
        let start = Instant::now();

        // Simulate random circuit execution
        let execution_time = Duration::from_millis(problem_instance.size as u64 * 10);
        std::thread::sleep(execution_time);

        Ok(AlgorithmResult {
            algorithm_type: "Random Circuit Sampling".to_string(),
            execution_time,
            solution_quality: 0.95,
            resource_usage: ResourceUsage {
                peak_memory: problem_instance.size * 1024 * 1024, // 1MB per qubit
                energy: 100.0,
                operations: problem_instance.size * 100,
                communication: 0.0,
            },
            success_rate: 0.95,
            output_distribution: Some(HashMap::new()),
        })
    }

    fn get_resource_requirements(&self, problem_size: usize) -> QuantumResources {
        QuantumResources {
            qubits: problem_size,
            depth: problem_size * 10,
            gate_count: problem_size * 100,
            coherence_time: Duration::from_millis(100),
            gate_fidelity: 0.999,
            shots: 1000,
            quantum_volume: problem_size * problem_size,
        }
    }

    fn get_theoretical_scaling(&self) -> f64 {
        1.0 // Linear in qubits
    }

    fn name(&self) -> &'static str {
        "Random Circuit Sampling"
    }
}

/// QAOA quantum algorithm
struct QAOAAlgorithm;

impl QuantumAlgorithm for QAOAAlgorithm {
    fn execute(&self, problem_instance: &ProblemInstance) -> Result<AlgorithmResult> {
        let execution_time = Duration::from_millis(problem_instance.size as u64 * 50);
        std::thread::sleep(execution_time);

        Ok(AlgorithmResult {
            algorithm_type: "QAOA".to_string(),
            execution_time,
            solution_quality: 0.9,
            resource_usage: ResourceUsage {
                peak_memory: problem_instance.size * 512 * 1024,
                energy: 200.0,
                operations: problem_instance.size * 200,
                communication: 0.0,
            },
            success_rate: 0.9,
            output_distribution: None,
        })
    }

    fn get_resource_requirements(&self, problem_size: usize) -> QuantumResources {
        QuantumResources {
            qubits: problem_size,
            depth: problem_size * 5,
            gate_count: problem_size * 50,
            coherence_time: Duration::from_millis(50),
            gate_fidelity: 0.99,
            shots: 10_000,
            quantum_volume: problem_size,
        }
    }

    fn get_theoretical_scaling(&self) -> f64 {
        1.0 // Linear scaling
    }

    fn name(&self) -> &'static str {
        "QAOA"
    }
}

/// Monte Carlo classical algorithm
struct MonteCarloAlgorithm;

impl ClassicalAlgorithm for MonteCarloAlgorithm {
    fn execute(&self, problem_instance: &ProblemInstance) -> Result<AlgorithmResult> {
        let execution_time = Duration::from_millis(problem_instance.size.pow(2) as u64);
        std::thread::sleep(execution_time);

        Ok(AlgorithmResult {
            algorithm_type: "Monte Carlo".to_string(),
            execution_time,
            solution_quality: 0.8,
            resource_usage: ResourceUsage {
                peak_memory: problem_instance.size * 1024,
                energy: 50.0,
                operations: problem_instance.size.pow(2),
                communication: 0.0,
            },
            success_rate: 0.8,
            output_distribution: None,
        })
    }

    fn get_resource_requirements(&self, problem_size: usize) -> ClassicalResources {
        ClassicalResources {
            cpu_time: Duration::from_millis(problem_size.pow(2) as u64),
            memory_usage: problem_size * 1024,
            cores: 1,
            energy_consumption: 50.0,
            storage: problem_size * 1024,
            network_bandwidth: 0.0,
        }
    }

    fn get_theoretical_scaling(&self) -> f64 {
        2.0 // Quadratic scaling
    }

    fn name(&self) -> &'static str {
        "Monte Carlo"
    }
}

/// Brute force classical algorithm
struct BruteForceAlgorithm;

impl ClassicalAlgorithm for BruteForceAlgorithm {
    fn execute(&self, problem_instance: &ProblemInstance) -> Result<AlgorithmResult> {
        let execution_time = Duration::from_millis(2_u64.pow(problem_instance.size as u32));

        // Timeout for large problems
        if execution_time > Duration::from_secs(60) {
            return Ok(AlgorithmResult {
                algorithm_type: "Brute Force (Timeout)".to_string(),
                execution_time: Duration::from_secs(60),
                solution_quality: 0.0,
                resource_usage: ResourceUsage {
                    peak_memory: 0,
                    energy: 0.0,
                    operations: 0,
                    communication: 0.0,
                },
                success_rate: 0.0,
                output_distribution: None,
            });
        }

        std::thread::sleep(execution_time);

        Ok(AlgorithmResult {
            algorithm_type: "Brute Force".to_string(),
            execution_time,
            solution_quality: 1.0,
            resource_usage: ResourceUsage {
                peak_memory: 2_usize.pow(problem_instance.size as u32) * 8,
                energy: 1000.0,
                operations: 2_usize.pow(problem_instance.size as u32),
                communication: 0.0,
            },
            success_rate: 1.0,
            output_distribution: None,
        })
    }

    fn get_resource_requirements(&self, problem_size: usize) -> ClassicalResources {
        ClassicalResources {
            cpu_time: Duration::from_millis(2_u64.pow(problem_size as u32)),
            memory_usage: 2_usize.pow(problem_size as u32) * 8,
            cores: 1,
            energy_consumption: 1000.0,
            storage: 2_usize.pow(problem_size as u32) * 8,
            network_bandwidth: 0.0,
        }
    }

    fn get_theoretical_scaling(&self) -> f64 {
        2.0_f64.ln() // Exponential scaling (base 2)
    }

    fn name(&self) -> &'static str {
        "Brute Force"
    }
}

/// Benchmark quantum advantage demonstration
pub fn benchmark_quantum_advantage() -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();

    let start = Instant::now();

    let config = QuantumAdvantageConfig {
        advantage_type: QuantumAdvantageType::ComputationalAdvantage,
        domain: ProblemDomain::RandomCircuitSampling,
        classical_algorithms: vec![ClassicalAlgorithmType::MonteCarlo],
        problem_sizes: vec![5, 10, 15],
        num_trials: 3,
        confidence_level: 0.95,
        classical_timeout: Duration::from_secs(60),
        hardware_specs: HardwareSpecs {
            quantum_hardware: QuantumHardwareSpecs {
                num_qubits: 20,
                gate_fidelities: HashMap::new(),
                coherence_times: HashMap::new(),
                connectivity: vec![vec![false; 20]; 20],
                gate_times: HashMap::new(),
                readout_fidelity: 0.95,
            },
            classical_hardware: ClassicalHardwareSpecs {
                cpu_cores: 8,
                cpu_frequency: 3.0,
                ram_size: 32_000_000_000,
                cache_sizes: vec![32_768, 262_144, 8_388_608],
                gpu_specs: None,
            },
        },
        enable_profiling: true,
        save_results: true,
    };

    let mut demonstrator = QuantumAdvantageDemonstrator::new(config);
    let _advantage_result = demonstrator.demonstrate_advantage()?;

    let demo_time = start.elapsed().as_millis() as f64;
    results.insert("quantum_advantage_demo".to_string(), demo_time);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_advantage_demonstrator_creation() {
        let config = create_test_config();
        let demonstrator = QuantumAdvantageDemonstrator::new(config);
        assert!(!demonstrator.quantum_algorithms.is_empty());
        assert!(!demonstrator.classical_algorithms.is_empty());
    }

    #[test]
    fn test_problem_instance_generation() {
        let config = create_test_config();
        let demonstrator = QuantumAdvantageDemonstrator::new(config);
        let instance = demonstrator
            .generate_problem_instance(5)
            .expect("Failed to generate problem instance");
        assert_eq!(instance.size, 5);
    }

    #[test]
    fn test_power_law_fitting() {
        let demonstrator = QuantumAdvantageDemonstrator::new(create_test_config());
        let sizes = vec![1, 2, 4, 8];
        let times = vec![1.0, 4.0, 16.0, 64.0]; // Quadratic scaling
        let scaling = demonstrator
            .fit_power_law(&sizes, &times)
            .expect("Failed to fit power law");
        assert!((scaling - 2.0).abs() < 0.1);
    }

    fn create_test_config() -> QuantumAdvantageConfig {
        QuantumAdvantageConfig {
            advantage_type: QuantumAdvantageType::ComputationalAdvantage,
            domain: ProblemDomain::RandomCircuitSampling,
            classical_algorithms: vec![ClassicalAlgorithmType::MonteCarlo],
            problem_sizes: vec![3, 5],
            num_trials: 2,
            confidence_level: 0.95,
            classical_timeout: Duration::from_secs(10),
            hardware_specs: HardwareSpecs {
                quantum_hardware: QuantumHardwareSpecs {
                    num_qubits: 10,
                    gate_fidelities: HashMap::new(),
                    coherence_times: HashMap::new(),
                    connectivity: vec![vec![false; 10]; 10],
                    gate_times: HashMap::new(),
                    readout_fidelity: 0.95,
                },
                classical_hardware: ClassicalHardwareSpecs {
                    cpu_cores: 4,
                    cpu_frequency: 2.0,
                    ram_size: 8_000_000_000,
                    cache_sizes: vec![32_768, 262_144],
                    gpu_specs: None,
                },
            },
            enable_profiling: false,
            save_results: false,
        }
    }
}
