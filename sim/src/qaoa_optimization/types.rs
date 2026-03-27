//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Parameter transfer database
#[derive(Debug, Clone)]
pub struct ParameterDatabase {
    /// Stored parameter sets by problem characteristics
    pub parameters: HashMap<ProblemCharacteristics, Vec<(Vec<f64>, Vec<f64>, f64)>>,
}
/// QAOA problem types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QAOAProblemType {
    /// Maximum Cut problem
    MaxCut,
    /// Maximum Weight Independent Set
    MaxWeightIndependentSet,
    /// Minimum Vertex Cover
    MinVertexCover,
    /// Graph Coloring
    GraphColoring,
    /// Traveling Salesman Problem
    TSP,
    /// Portfolio Optimization
    PortfolioOptimization,
    /// Job Shop Scheduling
    JobShopScheduling,
    /// Boolean 3-SAT
    Boolean3SAT,
    /// Quadratic Unconstrained Binary Optimization
    QUBO,
    /// Maximum Clique
    MaxClique,
    /// Bin Packing
    BinPacking,
    /// Custom Problem
    Custom,
}
/// QAOA level configuration
#[derive(Debug, Clone)]
pub struct QAOALevel {
    /// Problem size at this level
    pub problem_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Optimization budget
    pub optimization_budget: usize,
    /// Level-specific mixer
    pub mixer_type: QAOAMixerType,
}
/// Solution quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolutionQuality {
    /// Feasibility (satisfies constraints)
    pub feasible: bool,
    /// Gap to optimal solution (if known)
    pub optimality_gap: Option<f64>,
    /// Solution variance across multiple runs
    pub solution_variance: f64,
    /// Confidence in solution
    pub confidence: f64,
    /// Number of constraint violations
    pub constraint_violations: usize,
}
/// QAOA statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAOAStats {
    /// Total optimization time
    pub total_time: Duration,
    /// Time per layer evaluation
    pub layer_times: Vec<Duration>,
    /// Circuit depth per layer
    pub circuit_depths: Vec<usize>,
    /// Parameter sensitivity analysis
    pub parameter_sensitivity: HashMap<String, f64>,
    /// Quantum advantage metrics
    pub quantum_advantage: QuantumAdvantageMetrics,
}
/// Multi-level QAOA configuration
#[derive(Debug, Clone)]
pub struct MultiLevelQAOAConfig {
    /// Hierarchical levels
    pub levels: Vec<QAOALevel>,
    /// Parameter sharing between levels
    pub parameter_sharing: bool,
    /// Level transition criteria
    pub transition_criteria: LevelTransitionCriteria,
}
/// Level transition criteria
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelTransitionCriteria {
    /// Fixed schedule
    FixedSchedule,
    /// Performance based
    PerformanceBased,
    /// Convergence based
    ConvergenceBased,
    /// Adaptive
    Adaptive,
}
/// QAOA result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAOAResult {
    /// Optimal gamma parameters
    pub optimal_gammas: Vec<f64>,
    /// Optimal beta parameters
    pub optimal_betas: Vec<f64>,
    /// Best cost value found
    pub best_cost: f64,
    /// Approximation ratio
    pub approximation_ratio: f64,
    /// Optimization history
    pub cost_history: Vec<f64>,
    /// Parameter evolution
    pub parameter_history: Vec<(Vec<f64>, Vec<f64>)>,
    /// Final probability distribution
    pub final_probabilities: HashMap<String, f64>,
    /// Best solution bitstring
    pub best_solution: String,
    /// Solution quality metrics
    pub solution_quality: SolutionQuality,
    /// Optimization time
    pub optimization_time: Duration,
    /// Number of function evaluations
    pub function_evaluations: usize,
    /// Convergence information
    pub converged: bool,
}
/// Problem characteristics for parameter transfer
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ProblemCharacteristics {
    pub problem_type: QAOAProblemType,
    pub num_vertices: usize,
    pub density: u32,
    pub regularity: u32,
}
/// QAOA optimization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QAOAOptimizationStrategy {
    /// Classical optimization of angles
    Classical,
    /// Quantum optimization using quantum gradients
    Quantum,
    /// Hybrid classical-quantum optimization
    Hybrid,
    /// Machine learning guided optimization
    MLGuided,
    /// Adaptive parameter optimization
    Adaptive,
    /// `OptiRS` optimization (Adam, SGD, `RMSprop`, etc.) - requires "optimize" feature
    #[cfg(feature = "optimize")]
    OptiRS,
}
/// Graph representation for QAOA problems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAOAGraph {
    /// Number of vertices
    pub num_vertices: usize,
    /// Adjacency matrix
    pub adjacency_matrix: Array2<f64>,
    /// Vertex weights
    pub vertex_weights: Vec<f64>,
    /// Edge weights
    pub edge_weights: HashMap<(usize, usize), f64>,
    /// Additional constraints
    pub constraints: Vec<QAOAConstraint>,
}
/// Quantum advantage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAdvantageMetrics {
    /// Classical algorithm comparison time
    pub classical_time: Duration,
    /// Quantum speedup factor
    pub speedup_factor: f64,
    /// Success probability
    pub success_probability: f64,
    /// Quantum volume required
    pub quantum_volume: usize,
}
/// QAOA mixer types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QAOAMixerType {
    /// Standard X mixer (unconstrained)
    Standard,
    /// XY mixer for number conservation
    XY,
    /// Ring mixer for cyclic structures
    Ring,
    /// Grover mixer for amplitude amplification
    Grover,
    /// Dicke state mixer for cardinality constraints
    Dicke,
    /// Custom mixer with specified structure
    Custom,
}
/// QAOA configuration
#[derive(Debug, Clone)]
pub struct QAOAConfig {
    /// Number of QAOA layers (p)
    pub num_layers: usize,
    /// Mixer type
    pub mixer_type: QAOAMixerType,
    /// Initialization strategy
    pub initialization: QAOAInitializationStrategy,
    /// Optimization strategy
    pub optimization_strategy: QAOAOptimizationStrategy,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub convergence_tolerance: f64,
    /// Learning rate for optimization
    pub learning_rate: f64,
    /// Enable multi-angle QAOA
    pub multi_angle: bool,
    /// Enable parameter transfer learning
    pub parameter_transfer: bool,
    /// Hardware-specific optimizations
    pub hardware_aware: bool,
    /// Shot noise for finite sampling
    pub shots: Option<usize>,
    /// Enable adaptive layer growth
    pub adaptive_layers: bool,
    /// Maximum adaptive layers
    pub max_adaptive_layers: usize,
}
/// QAOA constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QAOAConstraint {
    /// Cardinality constraint (exactly k vertices selected)
    Cardinality { target: usize },
    /// Upper bound on selected vertices
    UpperBound { max_vertices: usize },
    /// Lower bound on selected vertices
    LowerBound { min_vertices: usize },
    /// Parity constraint
    Parity { even: bool },
    /// Custom linear constraint
    LinearConstraint { coefficients: Vec<f64>, bound: f64 },
}
/// QAOA initialization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QAOAInitializationStrategy {
    /// Uniform superposition
    UniformSuperposition,
    /// Warm start from classical solution
    WarmStart,
    /// Adiabatic initialization
    AdiabaticStart,
    /// Random initialization
    Random,
    /// Problem-specific initialization
    ProblemSpecific,
}
