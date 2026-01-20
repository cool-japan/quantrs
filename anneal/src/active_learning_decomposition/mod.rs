//! Active Learning for Problem Decomposition
//!
//! This module implements active learning techniques for intelligent decomposition
//! of complex optimization problems into smaller, more manageable subproblems.
//! It uses machine learning to guide the decomposition process and adaptively
//! improve decomposition strategies based on performance feedback.
//!
//! Key features:
//! - Intelligent problem decomposition using graph analysis
//! - Active learning for decomposition strategy selection
//! - Hierarchical decomposition with adaptive granularity
//! - Performance-guided decomposition refinement
//! - Multi-objective decomposition optimization
//! - Transfer learning across problem domains

pub mod config;
pub mod core;
pub mod knowledge_base;
pub mod performance_evaluation;
pub mod problem_analysis;
pub mod strategy_learning;
pub mod subproblem_generation;
pub mod types;
pub mod utils;

pub use config::*;
pub use core::*;
pub use knowledge_base::*;
pub use performance_evaluation::*;
pub use problem_analysis::*;
pub use strategy_learning::*;
pub use subproblem_generation::*;
pub use types::*;
pub use utils::*;

use crate::ising::IsingModel;
use crate::simulator::AnnealingResult;
use scirs2_core::ndarray::Array1;
use std::time::{Duration, Instant};

/// Decomposition strategies
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DecompositionStrategy {
    /// Graph partitioning based
    GraphPartitioning,
    /// Community detection based
    CommunityDetection,
    /// Spectral clustering
    SpectralClustering,
    /// Hierarchical decomposition
    Hierarchical,
    /// Random decomposition
    Random,
    /// Greedy decomposition
    Greedy,
    /// No decomposition
    NoDecomposition,
    /// Custom strategy
    Custom(String),
}

/// Problem analysis result
#[derive(Debug, Clone)]
pub struct ProblemAnalysis {
    /// Graph metrics
    pub graph_metrics: GraphMetrics,
    /// Detected communities
    pub communities: Vec<DetectedCommunity>,
    /// Detected structures
    pub structures: Vec<DetectedStructure>,
    /// Complexity estimate
    pub complexity: ComplexityEstimate,
    /// Decomposability score
    pub decomposability: DecomposabilityScore,
    /// Extracted features
    pub problem_features: Array1<f64>,
}

/// Detected community
#[derive(Debug, Clone)]
pub struct DetectedCommunity {
    /// Community ID
    pub id: usize,
    /// Vertices in community
    pub vertices: Vec<usize>,
    /// Community modularity
    pub modularity: f64,
    /// Internal density
    pub internal_density: f64,
    /// External density
    pub external_density: f64,
}

/// Subproblem representation
#[derive(Debug, Clone)]
pub struct Subproblem {
    /// Subproblem ID
    pub id: usize,
    /// Ising model for subproblem
    pub model: IsingModel,
    /// Original vertex indices
    pub vertices: Vec<usize>,
    /// Boundary edges to other subproblems
    pub boundary_edges: Vec<BoundaryEdge>,
    /// Subproblem metadata
    pub metadata: SubproblemMetadata,
}

impl Subproblem {
    #[must_use]
    pub fn from_full_problem(problem: &IsingModel) -> Self {
        Self {
            id: 0,
            model: problem.clone(),
            vertices: (0..problem.num_qubits).collect(),
            boundary_edges: Vec::new(),
            metadata: SubproblemMetadata::new(),
        }
    }
}

/// Boundary edge connecting subproblems
#[derive(Debug, Clone)]
pub struct BoundaryEdge {
    /// Vertex index within this subproblem
    pub internal_vertex: usize,
    /// Vertex index in original problem
    pub external_vertex: usize,
    /// Coupling strength
    pub coupling_strength: f64,
}

/// Subproblem metadata
#[derive(Debug, Clone)]
pub struct SubproblemMetadata {
    /// Creation timestamp
    pub creation_time: Instant,
    /// Subproblem size
    pub size: usize,
    /// Complexity estimate
    pub complexity_estimate: f64,
    /// Expected solution time
    pub expected_solution_time: Duration,
}

impl SubproblemMetadata {
    #[must_use]
    pub fn new() -> Self {
        Self {
            creation_time: Instant::now(),
            size: 0,
            complexity_estimate: 0.0,
            expected_solution_time: Duration::from_secs(1),
        }
    }
}

/// Decomposition result
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Generated subproblems
    pub subproblems: Vec<Subproblem>,
    /// Strategy used
    pub strategy_used: DecompositionStrategy,
    /// Quality score
    pub quality_score: f64,
    /// Problem analysis
    pub analysis: ProblemAnalysis,
    /// Decomposition metadata
    pub metadata: DecompositionMetadata,
}

/// Decomposition metadata
#[derive(Debug, Clone)]
pub struct DecompositionMetadata {
    /// Time spent on decomposition
    pub decomposition_time: Duration,
    /// Time spent on strategy selection
    pub strategy_selection_time: Duration,
    /// Total number of subproblems
    pub total_subproblems: usize,
    /// Decomposition depth
    pub decomposition_depth: usize,
}

impl DecompositionMetadata {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            decomposition_time: Duration::from_secs(0),
            strategy_selection_time: Duration::from_secs(0),
            total_subproblems: 0,
            decomposition_depth: 1,
        }
    }
}
