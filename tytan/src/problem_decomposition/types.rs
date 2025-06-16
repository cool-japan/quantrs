//! Data structures and types for problem decomposition

use crate::sampler::SampleResult;
use ndarray::Array2;
use std::collections::{HashMap, HashSet};

/// Graph representation for partitioning
#[derive(Debug, Clone)]
pub struct Graph {
    pub num_nodes: usize,
    pub edges: Vec<Edge>,
    pub node_weights: Vec<f64>,
}

/// Graph edge
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

/// Partitioning algorithms
#[derive(Debug, Clone)]
pub enum PartitioningAlgorithm {
    /// Kernighan-Lin algorithm
    KernighanLin,
    /// Fiduccia-Mattheyses algorithm
    FiducciaMattheyses,
    /// Spectral partitioning
    Spectral,
    /// METIS-style multilevel
    Multilevel,
    /// Community detection
    CommunityDetection,
    /// Min-cut max-flow
    MinCutMaxFlow,
}

/// Result of partitioning
#[derive(Debug, Clone)]
pub struct Partitioning {
    pub partition_assignment: Vec<usize>,
    pub subproblems: Vec<Subproblem>,
    pub coupling_terms: Vec<CouplingTerm>,
    pub metrics: PartitionMetrics,
}

/// Individual subproblem
#[derive(Debug, Clone)]
pub struct Subproblem {
    pub id: usize,
    pub variables: Vec<String>,
    pub qubo: Array2<f64>,
    pub var_map: HashMap<String, usize>,
}

/// Coupling between subproblems
#[derive(Debug, Clone)]
pub struct CouplingTerm {
    pub var1: String,
    pub var2: String,
    pub subproblem1: usize,
    pub subproblem2: usize,
    pub weight: f64,
}

/// Partition quality metrics
#[derive(Debug, Clone)]
pub struct PartitionMetrics {
    pub edge_cut: f64,
    pub balance: f64,
    pub modularity: f64,
    pub conductance: f64,
}

/// Hierarchical solving strategies
#[derive(Debug, Clone)]
pub enum HierarchicalStrategy {
    /// Coarsen then solve
    CoarsenSolve,
    /// Multi-grid approach
    MultiGrid,
    /// V-cycle
    VCycle,
}

/// Coarsening strategies
#[derive(Debug, Clone)]
pub enum CoarseningStrategy {
    /// Cluster strongly connected variables
    VariableClustering,
    /// Edge collapsing
    EdgeCollapsing,
    /// Algebraic multigrid
    AlgebraicMultigrid,
}

/// Hierarchy of problems
#[derive(Debug, Clone)]
pub struct Hierarchy {
    pub levels: Vec<HierarchyLevel>,
    pub projections: Vec<Projection>,
}

/// Single level in hierarchy
#[derive(Debug, Clone)]
pub struct HierarchyLevel {
    pub level: usize,
    pub qubo: Array2<f64>,
    pub var_map: HashMap<String, usize>,
    pub size: usize,
}

/// Projection between levels
#[derive(Debug, Clone)]
pub struct Projection {
    pub fine_to_coarse: Vec<usize>,
    pub coarse_to_fine: Vec<Vec<usize>>,
}

/// Domain decomposition strategies
#[derive(Debug, Clone)]
pub enum DecompositionStrategy {
    /// Schwarz alternating method
    Schwarz,
    /// Block Jacobi
    BlockJacobi,
    /// Additive Schwarz
    AdditiveSchwarz,
    /// Multiplicative Schwarz
    MultiplicativeSchwarz,
}

/// Coordination strategies for parallel solving
#[derive(Debug, Clone)]
pub enum CoordinationStrategy {
    /// Alternating Direction Method of Multipliers
    ADMM { rho: f64 },
    /// Consensus approach
    Consensus,
    /// Lagrangian relaxation
    LagrangianRelaxation,
    /// Price coordination
    PriceCoordination,
}

/// Domain for decomposition
#[derive(Debug, Clone)]
pub struct Domain {
    pub id: usize,
    pub variables: Vec<String>,
    pub qubo: Array2<f64>,
    pub var_map: HashMap<String, usize>,
    pub boundary_vars: Vec<usize>,
    pub internal_vars: Vec<usize>,
}

/// Coordination state for parallel solving
#[derive(Debug, Clone)]
pub struct CoordinationState {
    pub iteration: usize,
    pub lagrange_multipliers: Option<HashMap<(usize, usize), f64>>,
    pub consensus_variables: Option<HashMap<usize, bool>>,
    pub convergence_tolerance: f64,
    pub max_iterations: usize,
}

/// Solution for a subdomain
#[derive(Debug, Clone)]
pub struct SubdomainSolution {
    pub domain_id: usize,
    pub results: SampleResult,
}

/// CSP decomposition strategies
#[derive(Debug, Clone)]
pub enum CSPDecompositionStrategy {
    /// Tree decomposition
    TreeDecomposition,
    /// Constraint clustering
    ConstraintClustering,
    /// Cycle cutset
    CycleCutset,
    /// Bucket elimination
    BucketElimination,
}

/// Variable ordering heuristics
#[derive(Debug, Clone)]
pub enum VariableOrderingHeuristic {
    /// Minimum width ordering
    MinWidth,
    /// Maximum cardinality
    MaxCardinality,
    /// Fill-in heuristic
    MinFillIn,
    /// Weighted min-fill
    WeightedMinFill,
}

/// Constraint propagation levels
#[derive(Debug, Clone)]
pub enum PropagationLevel {
    /// No propagation
    None,
    /// Arc consistency
    ArcConsistency,
    /// Path consistency
    PathConsistency,
    /// Full consistency
    FullConsistency,
}

/// CSP Problem representation
#[derive(Debug, Clone)]
pub struct CSPProblem {
    pub variables: HashMap<String, DomainCsp>,
    pub constraints: Vec<CSPConstraint>,
    pub constraint_graph: ConstraintGraph,
}

/// CSP variable domain
#[derive(Debug, Clone)]
pub struct DomainCsp {
    pub values: Vec<i32>,
}

/// CSP constraint
#[derive(Debug, Clone)]
pub struct CSPConstraint {
    pub id: usize,
    pub scope: Vec<String>,
    pub constraint_type: ConstraintType,
    pub tuples: Option<Vec<Vec<i32>>>,
}

/// Constraint types
#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// All variables must be different
    AllDifferent,
    /// Linear constraint
    Linear { coefficients: Vec<f64>, rhs: f64 },
    /// Table constraint
    Table,
    /// Global constraint
    Global { name: String },
}

/// Constraint graph for CSP
#[derive(Debug, Clone)]
pub struct ConstraintGraph {
    /// Adjacency list representation
    pub adjacency: HashMap<String, HashSet<String>>,
    /// Constraint hypergraph
    pub hyperedges: Vec<HashSet<String>>,
}

/// CSP decomposition result
#[derive(Debug, Clone)]
pub struct CSPDecomposition {
    pub clusters: Vec<CSPCluster>,
    pub cluster_tree: ClusterTree,
    pub separator_sets: Vec<HashSet<String>>,
    pub width: usize,
}

/// CSP cluster
#[derive(Debug, Clone)]
pub struct CSPCluster {
    pub id: usize,
    pub variables: HashSet<String>,
    pub constraints: Vec<usize>,
    pub subproblem: Option<CSPSubproblem>,
}

/// CSP subproblem
#[derive(Debug, Clone)]
pub struct CSPSubproblem {
    pub variables: HashMap<String, DomainCsp>,
    pub constraints: Vec<CSPConstraint>,
}

/// Cluster tree for CSP
#[derive(Debug, Clone)]
pub struct ClusterTree {
    pub nodes: Vec<TreeNode>,
    pub edges: Vec<(usize, usize)>,
    pub root: usize,
}

/// Tree node in cluster tree
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub id: usize,
    pub cluster_id: usize,
    pub separator: HashSet<String>,
    pub children: Vec<usize>,
    pub parent: Option<usize>,
}

/// Tree decomposition result
#[derive(Debug, Clone)]
pub struct TreeDecomposition {
    pub bags: Vec<HashSet<String>>,
    pub tree_edges: Vec<(usize, usize)>,
    pub width: usize,
}

/// Decomposition metrics
#[derive(Debug, Clone)]
pub struct DecompositionMetrics {
    pub width: usize,
    pub num_clusters: usize,
    pub balance_factor: f64,
    pub separator_size: f64,
    pub decomposition_time: std::time::Duration,
}

/// Solution integration strategies
#[derive(Debug, Clone)]
pub enum IntegrationStrategy {
    /// Weighted voting
    WeightedVoting,
    /// Consensus building
    Consensus,
    /// Best solution selection
    BestSelection,
    /// Majority voting
    MajorityVoting,
}

/// Integrated solution result
#[derive(Debug, Clone)]
pub struct IntegratedSolution {
    pub assignment: HashMap<String, bool>,
    pub energy: f64,
    pub confidence: f64,
    pub component_solutions: Vec<ComponentSolution>,
}

/// Component solution from subproblem
#[derive(Debug, Clone)]
pub struct ComponentSolution {
    pub subproblem_id: usize,
    pub assignment: HashMap<String, bool>,
    pub energy: f64,
    pub weight: f64,
}

/// Decomposition configuration
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    pub max_subproblem_size: usize,
    pub min_subproblem_size: usize,
    pub overlap_factor: f64,
    pub balance_tolerance: f64,
    pub quality_threshold: f64,
}

/// Parallel execution configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub num_threads: usize,
    pub thread_pool_size: usize,
    pub load_balancing: bool,
    pub dynamic_scheduling: bool,
}
