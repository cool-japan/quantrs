//! Result and analysis types for scientific performance optimization.
//!
//! This module contains optimization results, performance reports,
//! bottleneck analysis, and resource utilization types.

use std::time::Duration;

use crate::applications::{
    drug_discovery::DrugDiscoveryProblem, materials_science::MaterialsOptimizationProblem,
    protein_folding::ProteinFoldingProblem,
};

use super::config::{CacheEvictionPolicy, DecompositionStrategy};
use super::parallel::{CrystalStructure, DefectAnalysisResult};

/// Optimized protein folding result
#[derive(Debug, Clone)]
pub struct OptimizedProteinFoldingResult {
    /// Original problem
    pub original_problem: ProteinFoldingProblem,
    /// Optimized result
    pub optimized_result: ProteinFoldingOptimizationResult,
    /// Memory optimizations applied
    pub memory_optimizations: MemoryOptimizations,
    /// Parallel optimizations applied
    pub parallel_optimizations: ParallelOptimizations,
    /// Algorithm optimizations applied
    pub algorithm_optimizations: AlgorithmOptimizations,
    /// Performance metrics
    pub performance_metrics: OptimizationPerformanceMetrics,
}

/// Optimized materials science result
#[derive(Debug, Clone)]
pub struct OptimizedMaterialsScienceResult {
    /// Original problem
    pub original_problem: MaterialsOptimizationProblem,
    /// Optimized result
    pub optimized_result: MaterialsOptimizationResult,
    /// Decomposition strategy used
    pub decomposition_strategy: DecompositionStrategy,
    /// Parallel strategy used
    pub parallel_strategy: ParallelLatticeStrategy,
    /// Performance metrics
    pub performance_metrics: OptimizationPerformanceMetrics,
}

/// Optimized drug discovery result
#[derive(Debug, Clone)]
pub struct OptimizedDrugDiscoveryResult {
    /// Original problem
    pub original_problem: DrugDiscoveryProblem,
    /// Optimized result
    pub optimized_result: DrugDiscoveryOptimizationResult,
    /// Caching strategy used
    pub caching_strategy: MolecularCachingStrategy,
    /// Distributed strategy used
    pub distributed_strategy: DistributedScreeningStrategy,
    /// Performance metrics
    pub performance_metrics: OptimizationPerformanceMetrics,
}

/// Optimization performance metrics
#[derive(Debug, Clone)]
pub struct OptimizationPerformanceMetrics {
    /// Total time
    pub total_time: Duration,
    /// Memory usage reduction
    pub memory_usage_reduction: f64,
    /// Speedup factor
    pub speedup_factor: f64,
    /// Quality improvement
    pub quality_improvement: f64,
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct ComprehensivePerformanceReport {
    /// System metrics
    pub system_metrics: SystemPerformanceMetrics,
    /// Optimization recommendations
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    /// Bottleneck analysis
    pub bottleneck_analysis: BottleneckAnalysis,
    /// Resource utilization
    pub resource_utilization: ResourceUtilizationAnalysis,
}

/// System performance metrics
#[derive(Debug, Clone)]
pub struct SystemPerformanceMetrics {
    /// Overall performance score
    pub overall_performance_score: f64,
    /// Memory efficiency
    pub memory_efficiency: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Category
    pub category: OptimizationCategory,
    /// Recommendation text
    pub recommendation: String,
    /// Impact level
    pub impact: OptimizationImpact,
    /// Estimated improvement
    pub estimated_improvement: f64,
}

/// Optimization categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationCategory {
    Memory,
    Parallelization,
    Algorithm,
    Distributed,
    GPU,
}

/// Optimization impact levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationImpact {
    Low,
    Medium,
    High,
    Critical,
}

/// Bottleneck analysis
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    /// Primary bottleneck
    pub primary_bottleneck: BottleneckType,
    /// Secondary bottlenecks
    pub secondary_bottlenecks: Vec<BottleneckType>,
    /// Bottleneck impact
    pub bottleneck_impact: f64,
    /// Resolution suggestions
    pub resolution_suggestions: Vec<String>,
}

/// Bottleneck types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BottleneckType {
    CPUUtilization,
    MemoryBandwidth,
    DiskIO,
    NetworkLatency,
    GPUMemory,
    AlgorithmComplexity,
}

/// Resource utilization analysis
#[derive(Debug, Clone)]
pub struct ResourceUtilizationAnalysis {
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Disk utilization
    pub disk_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
    /// GPU utilization
    pub gpu_utilization: f64,
    /// Efficiency score
    pub efficiency_score: f64,
}

/// Problem analysis
#[derive(Debug, Clone)]
pub struct ProblemAnalysis {
    /// Problem type
    pub problem_type: ProblemType,
    /// Complexity score
    pub complexity_score: f64,
    /// Memory requirements
    pub memory_requirements: usize,
    /// Parallel potential
    pub parallel_potential: f64,
    /// Recommended optimizations
    pub recommended_optimizations: Vec<OptimizationType>,
}

/// Problem types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProblemType {
    ProteinFolding,
    MaterialsScience,
    DrugDiscovery,
    Generic,
}

/// Optimization types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationType {
    MemoryPooling,
    ParallelExecution,
    ResultCaching,
    Approximation,
    Decomposition,
}

/// Memory optimizations
#[derive(Debug, Clone)]
pub struct MemoryOptimizations {
    /// Memory pool enabled
    pub memory_pool_enabled: bool,
    /// Cache strategy
    pub cache_strategy: CacheStrategy,
    /// Compression enabled
    pub compression_enabled: bool,
    /// Memory mapping enabled
    pub memory_mapping_enabled: bool,
    /// Estimated savings
    pub estimated_savings: f64,
}

/// Cache strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheStrategy {
    Simple,
    Hierarchical,
    Adaptive,
}

/// Parallel optimizations
#[derive(Debug, Clone)]
pub struct ParallelOptimizations {
    /// Parallel strategy
    pub parallel_strategy: ParallelStrategy,
    /// Thread count
    pub thread_count: usize,
    /// Load balancing enabled
    pub load_balancing_enabled: bool,
    /// NUMA awareness enabled
    pub numa_awareness_enabled: bool,
    /// Estimated speedup
    pub estimated_speedup: f64,
}

/// Parallel strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelStrategy {
    DataParallelism,
    TaskParallelism,
    Pipeline,
    Hybrid,
}

/// Algorithm optimizations
#[derive(Debug, Clone)]
pub struct AlgorithmOptimizations {
    /// Decomposition enabled
    pub decomposition_enabled: bool,
    /// Approximation enabled
    pub approximation_enabled: bool,
    /// Caching enabled
    pub caching_enabled: bool,
    /// Streaming enabled
    pub streaming_enabled: bool,
    /// Estimated improvement
    pub estimated_improvement: f64,
}

/// Protein folding optimization result
#[derive(Debug, Clone)]
pub struct ProteinFoldingOptimizationResult {
    /// Optimized conformation
    pub optimized_conformation: Vec<i32>,
    /// Energy reduction
    pub energy_reduction: f64,
    /// Convergence improvement
    pub convergence_improvement: f64,
    /// Execution time
    pub execution_time: Duration,
}

/// Crystal structure analysis
#[derive(Debug, Clone)]
pub struct CrystalStructureAnalysis {
    /// Lattice complexity
    pub lattice_complexity: f64,
    /// Atom count
    pub atom_count: usize,
    /// Symmetry groups
    pub symmetry_groups: Vec<String>,
    /// Optimization potential
    pub optimization_potential: f64,
}

/// Parallel lattice strategy
#[derive(Debug, Clone)]
pub struct ParallelLatticeStrategy {
    /// Partitioning method
    pub partitioning_method: PartitioningMethod,
    /// Communication pattern
    pub communication_pattern: CommunicationPattern,
    /// Load balancing method
    pub load_balancing: LoadBalancingMethod,
}

/// Partitioning methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitioningMethod {
    Spatial,
    Spectral,
    RandomizedBisection,
}

/// Communication patterns
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommunicationPattern {
    AllToAll,
    NearestNeighbor,
    TreeBased,
}

/// Load balancing methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadBalancingMethod {
    Static,
    Dynamic,
    Adaptive,
}

/// Materials optimization result
#[derive(Debug, Clone, Default)]
pub struct MaterialsOptimizationResult {
    /// Optimized structure
    pub optimized_structure: CrystalStructure,
    /// Energy minimization
    pub energy_minimization: f64,
    /// Defect analysis
    pub defect_analysis: DefectAnalysisResult,
    /// Simulation time
    pub simulation_time: Duration,
}

/// Molecular complexity analysis
#[derive(Debug, Clone)]
pub struct MolecularComplexityAnalysis {
    /// Molecular weight
    pub molecular_weight: f64,
    /// Rotatable bonds
    pub rotatable_bonds: usize,
    /// Ring count
    pub ring_count: usize,
    /// Complexity score
    pub complexity_score: f64,
}

/// Molecular caching strategy
#[derive(Debug, Clone)]
pub struct MolecularCachingStrategy {
    /// Cache type
    pub cache_type: MolecularCacheType,
    /// Cache size
    pub cache_size: usize,
    /// Eviction policy
    pub eviction_policy: CacheEvictionPolicy,
    /// Hit rate target
    pub hit_rate_target: f64,
}

/// Molecular cache types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MolecularCacheType {
    StructureBased,
    PropertyBased,
    InteractionBased,
}

/// Distributed screening strategy
#[derive(Debug, Clone)]
pub struct DistributedScreeningStrategy {
    /// Screening method
    pub screening_method: ScreeningMethod,
    /// Cluster size
    pub cluster_size: usize,
    /// Task distribution method
    pub task_distribution: TaskDistributionMethod,
    /// Fault tolerance enabled
    pub fault_tolerance: bool,
}

/// Screening methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScreeningMethod {
    VirtualScreening,
    PhysicalScreening,
    HybridScreening,
}

/// Task distribution methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskDistributionMethod {
    RoundRobin,
    LoadBalanced,
    Priority,
}

/// Drug discovery optimization result
#[derive(Debug, Clone)]
pub struct DrugDiscoveryOptimizationResult {
    /// Optimized molecules
    pub optimized_molecules: Vec<String>,
    /// Screening efficiency
    pub screening_efficiency: f64,
    /// Hit rate improvement
    pub hit_rate_improvement: f64,
    /// Discovery time
    pub discovery_time: Duration,
}
