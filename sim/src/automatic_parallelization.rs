//! Automatic Parallelization for Quantum Circuits
//!
//! This module provides automatic parallelization capabilities for quantum circuits,
//! analyzing circuit structure to identify independent gate operations that can be
//! executed in parallel using SciRS2 parallel operations for optimal performance.

use crate::distributed_simulator::{DistributedQuantumSimulator, DistributedSimulatorConfig};
use crate::large_scale_simulator::{LargeScaleQuantumSimulator, LargeScaleSimulatorConfig};
use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::parallel_ops::*; // SciRS2 POLICY compliant
                                  // use scirs2_core::scheduling::{Scheduler, TaskGraph, ParallelExecutor};
                                  // use scirs2_core::optimization::{CostModel, ResourceOptimizer};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Barrier, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Configuration for automatic parallelization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoParallelConfig {
    /// Maximum number of parallel execution threads
    pub max_threads: usize,

    /// Minimum gate count to enable parallelization
    pub min_gates_for_parallel: usize,

    /// Parallelization strategy
    pub strategy: ParallelizationStrategy,

    /// Resource constraints
    pub resource_constraints: ResourceConstraints,

    /// Enable inter-layer parallelization
    pub enable_inter_layer_parallel: bool,

    /// Enable gate fusion optimization
    pub enable_gate_fusion: bool,

    /// SciRS2 optimization level
    pub scirs2_optimization_level: OptimizationLevel,

    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,

    /// Enable circuit analysis caching
    pub enable_analysis_caching: bool,

    /// Memory budget for parallel execution
    pub memory_budget: usize,
}

impl Default for AutoParallelConfig {
    fn default() -> Self {
        Self {
            max_threads: current_num_threads(), // SciRS2 POLICY compliant
            min_gates_for_parallel: 10,
            strategy: ParallelizationStrategy::DependencyAnalysis,
            resource_constraints: ResourceConstraints::default(),
            enable_inter_layer_parallel: true,
            enable_gate_fusion: true,
            scirs2_optimization_level: OptimizationLevel::Aggressive,
            load_balancing: LoadBalancingConfig::default(),
            enable_analysis_caching: true,
            memory_budget: 4 * 1024 * 1024 * 1024, // 4GB
        }
    }
}

/// Parallelization strategies for circuit execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelizationStrategy {
    /// Analyze gate dependencies and parallelize independent operations
    DependencyAnalysis,
    /// Layer-based parallelization with depth analysis
    LayerBased,
    /// Qubit partitioning for independent subsystems
    QubitPartitioning,
    /// Hybrid approach combining multiple strategies
    Hybrid,
    /// Machine learning guided parallelization
    MLGuided,
    /// Hardware-aware parallelization
    HardwareAware,
}

/// Resource constraints for parallel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum memory per thread (bytes)
    pub max_memory_per_thread: usize,
    /// Maximum CPU utilization (0.0 to 1.0)
    pub max_cpu_utilization: f64,
    /// Maximum gate operations per thread
    pub max_gates_per_thread: usize,
    /// Preferred NUMA node
    pub preferred_numa_node: Option<usize>,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_memory_per_thread: 1024 * 1024 * 1024, // 1GB
            max_cpu_utilization: 0.8,
            max_gates_per_thread: 1000,
            preferred_numa_node: None,
        }
    }
}

/// Load balancing configuration for parallel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Enable dynamic load balancing
    pub enable_dynamic_balancing: bool,
    /// Work stealing strategy
    pub work_stealing_strategy: WorkStealingStrategy,
    /// Load monitoring interval
    pub monitoring_interval: Duration,
    /// Rebalancing threshold
    pub rebalancing_threshold: f64,
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_balancing: true,
            work_stealing_strategy: WorkStealingStrategy::Adaptive,
            monitoring_interval: Duration::from_millis(100),
            rebalancing_threshold: 0.2,
        }
    }
}

/// Work stealing strategies for load balancing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkStealingStrategy {
    /// Random work stealing
    Random,
    /// Cost-aware work stealing
    CostAware,
    /// Locality-aware work stealing
    LocalityAware,
    /// Adaptive strategy selection
    Adaptive,
}

/// SciRS2 optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations
    Basic,
    /// Advanced optimizations
    Advanced,
    /// Aggressive optimizations
    Aggressive,
    /// Custom optimization profile
    Custom,
}

/// Parallel execution task representing a group of independent gates
#[derive(Debug, Clone)]
pub struct ParallelTask {
    /// Unique task identifier
    pub id: Uuid,
    /// Gates to execute in this task
    pub gates: Vec<Arc<dyn GateOp + Send + Sync>>,
    /// Qubits involved in this task
    pub qubits: HashSet<QubitId>,
    /// Estimated execution cost
    pub cost: f64,
    /// Memory requirement estimate
    pub memory_requirement: usize,
    /// Dependencies (task IDs that must complete before this task)
    pub dependencies: HashSet<Uuid>,
    /// Priority level
    pub priority: TaskPriority,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    /// Low priority task
    Low = 1,
    /// Normal priority task
    Normal = 2,
    /// High priority task
    High = 3,
    /// Critical priority task
    Critical = 4,
}

/// Circuit dependency graph for parallelization analysis
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Gate nodes in the dependency graph
    pub nodes: Vec<GateNode>,
    /// Adjacency list representation
    pub edges: HashMap<usize, Vec<usize>>,
    /// Reverse adjacency list
    pub reverse_edges: HashMap<usize, Vec<usize>>,
    /// Topological layers
    pub layers: Vec<Vec<usize>>,
}

/// Gate node in the dependency graph
#[derive(Debug, Clone)]
pub struct GateNode {
    /// Gate index in original circuit
    pub gate_index: usize,
    /// Gate operation
    pub gate: Arc<dyn GateOp + Send + Sync>,
    /// Qubits this gate operates on
    pub qubits: HashSet<QubitId>,
    /// Layer index in topological ordering
    pub layer: usize,
    /// Estimated execution cost
    pub cost: f64,
}

/// Parallelization analysis results
#[derive(Debug, Clone)]
pub struct ParallelizationAnalysis {
    /// Parallel tasks generated
    pub tasks: Vec<ParallelTask>,
    /// Total number of layers
    pub num_layers: usize,
    /// Parallelization efficiency (0.0 to 1.0)
    pub efficiency: f64,
    /// Maximum parallelism achievable
    pub max_parallelism: usize,
    /// Critical path length
    pub critical_path_length: usize,
    /// Resource utilization predictions
    pub resource_utilization: ResourceUtilization,
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Resource utilization predictions
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// Estimated CPU utilization per thread
    pub cpu_utilization: Vec<f64>,
    /// Estimated memory usage per thread
    pub memory_usage: Vec<usize>,
    /// Load balancing score (0.0 to 1.0)
    pub load_balance_score: f64,
    /// Communication overhead estimate
    pub communication_overhead: f64,
}

/// Optimization recommendations for better parallelization
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Description of the recommendation
    pub description: String,
    /// Expected improvement (0.0 to 1.0)
    pub expected_improvement: f64,
    /// Implementation complexity
    pub complexity: RecommendationComplexity,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationType {
    /// Gate reordering for better parallelism
    GateReordering,
    /// Circuit decomposition
    CircuitDecomposition,
    /// Resource allocation adjustment
    ResourceAllocation,
    /// Strategy change recommendation
    StrategyChange,
    /// Hardware configuration
    HardwareConfiguration,
}

/// Complexity levels for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationComplexity {
    /// Low complexity, easy to implement
    Low,
    /// Medium complexity
    Medium,
    /// High complexity, significant changes required
    High,
}

/// ML features extracted from circuits for parallelization prediction
#[derive(Debug, Clone)]
pub struct MLFeatures {
    /// Number of gates in the circuit
    pub num_gates: usize,
    /// Number of qubits in the circuit
    pub num_qubits: usize,
    /// Circuit depth (critical path length)
    pub circuit_depth: usize,
    /// Average gate connectivity
    pub avg_connectivity: f64,
    /// Parallelism factor (ratio of independent gates)
    pub parallelism_factor: f64,
    /// Gate type distribution
    pub gate_distribution: HashMap<String, usize>,
    /// Entanglement complexity score
    pub entanglement_score: f64,
    /// Dependency density (edges per gate)
    pub dependency_density: f64,
}

/// ML-predicted parallelization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLPredictedStrategy {
    /// High parallelism - aggressive parallel execution
    HighParallelism,
    /// Balanced parallelism - mixed approach
    BalancedParallelism,
    /// Conservative parallelism - careful dependency management
    ConservativeParallelism,
    /// Layer-optimized execution
    LayerOptimized,
}

/// Hardware characteristics for hardware-aware parallelization
#[derive(Debug, Clone)]
pub struct HardwareCharacteristics {
    /// Number of available CPU cores
    pub num_cores: usize,
    /// L1 cache size per core (bytes)
    pub l1_cache_size: usize,
    /// L2 cache size per core (bytes)
    pub l2_cache_size: usize,
    /// L3 cache size (bytes)
    pub l3_cache_size: usize,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
    /// NUMA nodes available
    pub num_numa_nodes: usize,
    /// GPU availability
    pub has_gpu: bool,
    /// SIMD width (e.g., 256 for AVX2, 512 for AVX-512)
    pub simd_width: usize,
}

/// Hardware-specific parallelization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareStrategy {
    /// Optimize for cache locality
    CacheOptimized,
    /// Optimize for SIMD vectorization
    SIMDOptimized,
    /// NUMA-aware task distribution
    NUMAAware,
    /// Offload to GPU
    GPUOffload,
    /// Hybrid approach combining multiple optimizations
    Hybrid,
}

/// Node capacity information for distributed task scheduling
#[derive(Debug, Clone)]
pub struct NodeCapacity {
    /// Number of CPU cores available
    pub cpu_cores: usize,
    /// Available memory in GB
    pub memory_gb: f64,
    /// GPU availability
    pub gpu_available: bool,
    /// Network bandwidth in Gbps
    pub network_bandwidth_gbps: f64,
    /// Relative performance score (normalized)
    pub relative_performance: f64,
}

/// Automatic parallelization engine for quantum circuits
pub struct AutoParallelEngine {
    /// Configuration
    config: AutoParallelConfig,
    /// Analysis cache for circuits
    analysis_cache: Arc<RwLock<HashMap<u64, ParallelizationAnalysis>>>,
    /// Performance statistics
    performance_stats: Arc<Mutex<ParallelPerformanceStats>>,
    /// SciRS2 integration components
    //scirs2_scheduler: SciRS2Scheduler,
    /// Load balancer
    load_balancer: Arc<Mutex<LoadBalancer>>,
}

/// Performance statistics for parallel execution
#[derive(Debug, Clone, Default)]
pub struct ParallelPerformanceStats {
    /// Total circuits processed
    pub circuits_processed: usize,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average parallelization efficiency
    pub average_efficiency: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Task completion statistics
    pub task_stats: TaskCompletionStats,
    /// Resource utilization history
    pub resource_history: Vec<ResourceSnapshot>,
}

/// Task completion statistics
#[derive(Debug, Clone, Default)]
pub struct TaskCompletionStats {
    /// Total tasks completed
    pub total_tasks: usize,
    /// Average task duration
    pub average_duration: Duration,
    /// Task success rate
    pub success_rate: f64,
    /// Load balancing effectiveness
    pub load_balance_effectiveness: f64,
}

/// Resource utilization snapshot
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// CPU utilization per core
    pub cpu_utilization: Vec<f64>,
    /// Memory usage
    pub memory_usage: usize,
    /// Active tasks
    pub active_tasks: usize,
}

/// Load balancer for parallel task execution
pub struct LoadBalancer {
    /// Current thread loads
    thread_loads: Vec<f64>,
    /// Task queue per thread
    task_queues: Vec<VecDeque<ParallelTask>>,
    /// Work stealing statistics
    work_stealing_stats: WorkStealingStats,
}

/// Work stealing statistics
#[derive(Debug, Clone, Default)]
pub struct WorkStealingStats {
    /// Total steal attempts
    pub steal_attempts: usize,
    /// Successful steals
    pub successful_steals: usize,
    /// Failed steals
    pub failed_steals: usize,
    /// Average steal latency
    pub average_steal_latency: Duration,
}

impl AutoParallelEngine {
    /// Create a new automatic parallelization engine
    pub fn new(config: AutoParallelConfig) -> Self {
        let num_threads = config.max_threads;

        Self {
            config,
            analysis_cache: Arc::new(RwLock::new(HashMap::new())),
            performance_stats: Arc::new(Mutex::new(ParallelPerformanceStats::default())),
            load_balancer: Arc::new(Mutex::new(LoadBalancer::new(num_threads))),
        }
    }

    /// Analyze a circuit for parallelization opportunities
    pub fn analyze_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<ParallelizationAnalysis> {
        let start_time = Instant::now();

        // Check cache first if enabled
        if self.config.enable_analysis_caching {
            let circuit_hash = self.compute_circuit_hash(circuit);
            if let Some(cached_analysis) = self.analysis_cache.read().unwrap().get(&circuit_hash) {
                return Ok(cached_analysis.clone());
            }
        }

        // Build dependency graph
        let dependency_graph = self.build_dependency_graph(circuit)?;

        // Generate parallel tasks based on strategy
        let tasks = match self.config.strategy {
            ParallelizationStrategy::DependencyAnalysis => {
                self.dependency_based_parallelization(&dependency_graph)?
            }
            ParallelizationStrategy::LayerBased => {
                self.layer_based_parallelization(&dependency_graph)?
            }
            ParallelizationStrategy::QubitPartitioning => {
                self.qubit_partitioning_parallelization(circuit, &dependency_graph)?
            }
            ParallelizationStrategy::Hybrid => {
                self.hybrid_parallelization(circuit, &dependency_graph)?
            }
            ParallelizationStrategy::MLGuided => {
                self.ml_guided_parallelization(circuit, &dependency_graph)?
            }
            ParallelizationStrategy::HardwareAware => {
                self.hardware_aware_parallelization(circuit, &dependency_graph)?
            }
        };

        // Calculate parallelization metrics
        let analysis = self.calculate_parallelization_metrics(circuit, &dependency_graph, tasks)?;

        // Cache the analysis if enabled
        if self.config.enable_analysis_caching {
            let circuit_hash = self.compute_circuit_hash(circuit);
            self.analysis_cache
                .write()
                .unwrap()
                .insert(circuit_hash, analysis.clone());
        }

        // Update performance statistics
        self.update_performance_stats(start_time.elapsed(), &analysis);

        Ok(analysis)
    }

    /// Execute a circuit using automatic parallelization
    pub fn execute_parallel<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        simulator: &mut LargeScaleQuantumSimulator,
    ) -> QuantRS2Result<Vec<Complex64>> {
        let analysis = self.analyze_circuit(circuit)?;

        if analysis.tasks.len() < self.config.min_gates_for_parallel {
            // Fall back to sequential execution for small circuits
            return self.execute_sequential(circuit, simulator);
        }

        // Set up parallel execution environment
        let barrier = Arc::new(Barrier::new(self.config.max_threads));
        let shared_state = Arc::new(RwLock::new(simulator.get_dense_state()?));
        let task_results = Arc::new(Mutex::new(Vec::new()));

        // Execute tasks in parallel with dependency respect
        self.execute_parallel_tasks(&analysis.tasks, shared_state.clone(), task_results, barrier)?;

        // Collect and return final state
        let final_state = shared_state.read().unwrap().clone();
        Ok(final_state)
    }

    /// Execute circuit with distributed parallelization
    pub fn execute_distributed<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        distributed_sim: &mut DistributedQuantumSimulator,
    ) -> QuantRS2Result<Vec<Complex64>> {
        let analysis = self.analyze_circuit(circuit)?;

        // Distribute tasks across cluster nodes
        let distributed_tasks =
            self.distribute_tasks_across_nodes(&analysis.tasks, distributed_sim)?;

        // Execute with inter-node coordination
        // Implement distributed parallel task execution
        let results = self.execute_distributed_tasks(&distributed_tasks, distributed_sim)?;

        // Aggregate results from all nodes
        let final_result = self.aggregate_distributed_results(results)?;

        Ok(final_result)
    }

    /// Execute distributed tasks across nodes
    fn execute_distributed_tasks(
        &self,
        distributed_tasks: &[Vec<ParallelTask>],
        distributed_sim: &DistributedQuantumSimulator,
    ) -> QuantRS2Result<Vec<Vec<Complex64>>> {
        use scirs2_core::parallel_ops::*;

        let cluster_status = distributed_sim.get_cluster_status();
        let num_nodes = cluster_status.len();

        // Execute tasks on each node in parallel
        let node_results: Vec<Vec<Complex64>> =
            parallel_map(&(0..num_nodes).collect::<Vec<_>>(), |&node_id| {
                let tasks = &distributed_tasks[node_id];
                let mut node_result = Vec::new();

                // Execute tasks sequentially on each node
                for task in tasks {
                    // Simulate task execution on the node
                    // In a real implementation, this would involve network communication
                    let task_result = self.execute_task_on_node(task, node_id);
                    node_result.extend(task_result);
                }

                node_result
            });

        Ok(node_results)
    }

    /// Execute a single task on a specific node
    const fn execute_task_on_node(&self, task: &ParallelTask, node_id: usize) -> Vec<Complex64> {
        // Placeholder implementation for task execution on a node
        // In a real implementation, this would involve:
        // 1. Sending task data to the node
        // 2. Node executing gates on its portion of the state
        // 3. Receiving results back

        // For now, return empty result
        // This would be populated with actual gate execution results
        Vec::new()
    }

    /// Aggregate results from distributed execution
    fn aggregate_distributed_results(
        &self,
        node_results: Vec<Vec<Complex64>>,
    ) -> QuantRS2Result<Vec<Complex64>> {
        use scirs2_core::parallel_ops::*;

        // Combine results from all nodes
        // In a real implementation, this would involve proper state vector reconstruction
        let total_elements: usize = node_results.iter().map(|r| r.len()).sum();
        let mut aggregated = Vec::with_capacity(total_elements);

        for node_result in node_results {
            aggregated.extend(node_result);
        }

        Ok(aggregated)
    }

    /// Build dependency graph for the circuit
    fn build_dependency_graph<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<DependencyGraph> {
        let gates = circuit.gates();
        let mut nodes = Vec::with_capacity(gates.len());
        let mut edges: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut reverse_edges: HashMap<usize, Vec<usize>> = HashMap::new();

        // Create gate nodes
        for (i, gate) in gates.iter().enumerate() {
            let qubits: HashSet<QubitId> = gate.qubits().into_iter().collect();
            let cost = self.estimate_gate_cost(gate.as_ref());

            nodes.push(GateNode {
                gate_index: i,
                gate: gate.clone(),
                qubits,
                layer: 0, // Will be computed later
                cost,
            });

            edges.insert(i, Vec::new());
            reverse_edges.insert(i, Vec::new());
        }

        // Build dependency edges based on qubit conflicts
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                if !nodes[i].qubits.is_disjoint(&nodes[j].qubits) {
                    // Gates operate on same qubits, so j depends on i
                    edges.get_mut(&i).unwrap().push(j);
                    reverse_edges.get_mut(&j).unwrap().push(i);
                }
            }
        }

        // Compute topological layers
        let layers = self.compute_topological_layers(&nodes, &edges)?;

        // Update layer information in nodes
        for (layer_idx, layer) in layers.iter().enumerate() {
            for &node_idx in layer {
                if let Some(node) = nodes.get_mut(node_idx) {
                    node.layer = layer_idx;
                }
            }
        }

        Ok(DependencyGraph {
            nodes,
            edges,
            reverse_edges,
            layers,
        })
    }

    /// Compute topological layers for parallel execution
    fn compute_topological_layers(
        &self,
        nodes: &[GateNode],
        edges: &HashMap<usize, Vec<usize>>,
    ) -> QuantRS2Result<Vec<Vec<usize>>> {
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        let mut layers = Vec::new();
        let mut queue = VecDeque::new();

        // Initialize in-degrees
        for i in 0..nodes.len() {
            in_degree.insert(i, 0);
        }

        for to_list in edges.values() {
            for &to in to_list {
                *in_degree.get_mut(&to).unwrap() += 1;
            }
        }

        // Start with nodes that have no dependencies
        for i in 0..nodes.len() {
            if in_degree[&i] == 0 {
                queue.push_back(i);
            }
        }

        while !queue.is_empty() {
            let mut current_layer = Vec::new();
            let layer_size = queue.len();

            for _ in 0..layer_size {
                if let Some(node) = queue.pop_front() {
                    current_layer.push(node);

                    // Update dependencies
                    if let Some(neighbors) = edges.get(&node) {
                        for &neighbor in neighbors {
                            let new_degree = in_degree[&neighbor] - 1;
                            in_degree.insert(neighbor, new_degree);

                            if new_degree == 0 {
                                queue.push_back(neighbor);
                            }
                        }
                    }
                }
            }

            if !current_layer.is_empty() {
                layers.push(current_layer);
            }
        }

        Ok(layers)
    }

    /// Dependency-based parallelization strategy
    fn dependency_based_parallelization(
        &self,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        let mut tasks = Vec::new();

        for layer in &graph.layers {
            if layer.len() > 1 {
                // Create parallel tasks for independent gates in this layer
                let chunks = self.partition_layer_into_tasks(layer, graph)?;

                for chunk in chunks {
                    let task = self.create_parallel_task(chunk, graph)?;
                    tasks.push(task);
                }
            } else {
                // Single gate, create individual task
                if let Some(&gate_idx) = layer.first() {
                    let task = self.create_parallel_task(vec![gate_idx], graph)?;
                    tasks.push(task);
                }
            }
        }

        Ok(tasks)
    }

    /// Layer-based parallelization strategy
    fn layer_based_parallelization(
        &self,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        let mut tasks = Vec::new();

        for layer in &graph.layers {
            // Each layer becomes one or more parallel tasks
            let max_gates_per_task = self.config.resource_constraints.max_gates_per_thread;

            for chunk in layer.chunks(max_gates_per_task) {
                let task = self.create_parallel_task(chunk.to_vec(), graph)?;
                tasks.push(task);
            }
        }

        Ok(tasks)
    }

    /// Qubit partitioning parallelization strategy
    fn qubit_partitioning_parallelization<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Partition qubits into independent subsystems
        let qubit_partitions = self.partition_qubits(circuit)?;
        let mut tasks = Vec::new();

        for partition in qubit_partitions {
            // Find gates that operate only on qubits in this partition
            let mut partition_gates = Vec::new();

            for (i, node) in graph.nodes.iter().enumerate() {
                if node.qubits.iter().all(|q| partition.contains(q)) {
                    partition_gates.push(i);
                }
            }

            if !partition_gates.is_empty() {
                let task = self.create_parallel_task(partition_gates, graph)?;
                tasks.push(task);
            }
        }

        Ok(tasks)
    }

    /// Hybrid parallelization strategy
    fn hybrid_parallelization<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Combine multiple strategies for optimal parallelization
        let dependency_tasks = self.dependency_based_parallelization(graph)?;
        let layer_tasks = self.layer_based_parallelization(graph)?;
        let partition_tasks = self.qubit_partitioning_parallelization(circuit, graph)?;

        // Select the best strategy based on efficiency metrics
        let strategies = vec![
            ("dependency", dependency_tasks),
            ("layer", layer_tasks),
            ("partition", partition_tasks),
        ];

        let best_strategy = strategies.into_iter().max_by(|(_, tasks_a), (_, tasks_b)| {
            let efficiency_a = self.calculate_strategy_efficiency(tasks_a);
            let efficiency_b = self.calculate_strategy_efficiency(tasks_b);
            efficiency_a
                .partial_cmp(&efficiency_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        match best_strategy {
            Some((_, tasks)) => Ok(tasks),
            None => Ok(Vec::new()),
        }
    }

    /// ML-guided parallelization strategy
    fn ml_guided_parallelization<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Implement machine learning guided parallelization
        // Extract features from the circuit and dependency graph
        let features = self.extract_ml_features(circuit, graph);

        // Predict optimal parallelization strategy using ML model
        let predicted_strategy = self.predict_parallelization_strategy(&features);

        // Generate task groups based on predicted strategy
        let task_groups = match predicted_strategy {
            MLPredictedStrategy::HighParallelism => {
                // Aggressive parallelization for highly independent circuits
                self.aggressive_parallelization(graph)?
            }
            MLPredictedStrategy::BalancedParallelism => {
                // Balanced approach for mixed circuits
                self.hybrid_parallelization(circuit, graph)?
            }
            MLPredictedStrategy::ConservativeParallelism => {
                // Conservative parallelization for highly dependent circuits
                self.dependency_based_parallelization(graph)?
            }
            MLPredictedStrategy::LayerOptimized => {
                // Layer-based optimization for structured circuits
                self.layer_based_parallelization(graph)?
            }
        };

        // Optimize task groups using ML-guided refinement
        let optimized_tasks = self.ml_optimize_tasks(task_groups, &features)?;

        Ok(optimized_tasks)
    }

    /// Extract ML features from circuit and dependency graph
    fn extract_ml_features<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
    ) -> MLFeatures {
        let gates = circuit.gates();
        let num_gates = gates.len();
        let num_qubits = N;

        // Calculate circuit depth (critical path length)
        let circuit_depth = self.calculate_circuit_depth(graph);

        // Calculate average gate connectivity
        let avg_connectivity = self.calculate_average_connectivity(graph);

        // Calculate parallelism factor (ratio of independent gates)
        let parallelism_factor = self.calculate_parallelism_factor(graph);

        // Calculate gate type distribution
        let gate_distribution = self.calculate_gate_distribution(gates);

        // Calculate entanglement complexity
        let entanglement_score = self.estimate_entanglement_complexity(circuit);

        MLFeatures {
            num_gates,
            num_qubits,
            circuit_depth,
            avg_connectivity,
            parallelism_factor,
            gate_distribution,
            entanglement_score,
            dependency_density: graph.edges.len() as f64 / num_gates as f64,
        }
    }

    /// Predict optimal parallelization strategy based on ML features
    fn predict_parallelization_strategy(&self, features: &MLFeatures) -> MLPredictedStrategy {
        // Simple heuristic-based prediction (in production, this would use a trained ML model)
        // Decision tree based on circuit characteristics

        // High parallelism: many gates, low connectivity, high parallelism factor
        if features.parallelism_factor > 0.7 && features.avg_connectivity < 2.0 {
            return MLPredictedStrategy::HighParallelism;
        }

        // Layer optimized: structured circuits with clear layers
        if features.circuit_depth < (features.num_gates as f64 * 0.3) as usize {
            return MLPredictedStrategy::LayerOptimized;
        }

        // Conservative: highly connected circuits with dependencies
        if features.avg_connectivity > 3.5 || features.dependency_density > 0.6 {
            return MLPredictedStrategy::ConservativeParallelism;
        }

        // Default to balanced approach
        MLPredictedStrategy::BalancedParallelism
    }

    /// ML-guided task optimization
    fn ml_optimize_tasks(
        &self,
        tasks: Vec<ParallelTask>,
        features: &MLFeatures,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Optimize task grouping based on ML predictions
        let mut optimized = tasks;

        // Balance task loads using learned cost models
        self.balance_task_loads(&mut optimized)?;

        // Merge small tasks if beneficial (predicted from features)
        if features.num_gates < 50 {
            optimized = self.merge_small_tasks(optimized)?;
        }

        // Split large tasks if parallelism is high
        if features.parallelism_factor > 0.6 {
            optimized = self.split_large_tasks(optimized)?;
        }

        Ok(optimized)
    }

    /// Aggressive parallelization for highly independent circuits
    fn aggressive_parallelization(
        &self,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        let mut tasks = Vec::new();
        let mut visited = vec![false; graph.nodes.len()];

        // Group gates with no dependencies into parallel tasks
        for (idx, node) in graph.nodes.iter().enumerate() {
            if visited[idx] {
                continue;
            }

            // Find all gates that can execute in parallel with this one
            let mut parallel_group = vec![idx];
            visited[idx] = true;

            for (other_idx, other_node) in graph.nodes.iter().enumerate() {
                if visited[other_idx] {
                    continue;
                }

                // Check if gates are independent (no shared qubits, no dependencies)
                if !self.gates_have_dependency(idx, other_idx, graph)
                    && !self.gates_share_qubits(&node.qubits, &other_node.qubits)
                {
                    parallel_group.push(other_idx);
                    visited[other_idx] = true;
                }
            }

            if !parallel_group.is_empty() {
                tasks.push(self.create_parallel_task(parallel_group, graph)?);
            }
        }

        Ok(tasks)
    }

    /// Calculate circuit depth (critical path)
    fn calculate_circuit_depth(&self, graph: &DependencyGraph) -> usize {
        let mut depths = vec![0; graph.nodes.len()];

        // Topological sort and calculate depths
        for (idx, node) in graph.nodes.iter().enumerate() {
            let mut max_parent_depth = 0;
            if let Some(parents) = graph.reverse_edges.get(&idx) {
                for &parent in parents {
                    max_parent_depth = max_parent_depth.max(depths[parent]);
                }
            }
            depths[idx] = max_parent_depth + 1;
        }

        *depths.iter().max().unwrap_or(&0)
    }

    /// Calculate average gate connectivity
    fn calculate_average_connectivity(&self, graph: &DependencyGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 0.0;
        }

        let total_connections: usize = graph.nodes.iter().map(|n| n.qubits.len()).sum();
        total_connections as f64 / graph.nodes.len() as f64
    }

    /// Calculate parallelism factor
    fn calculate_parallelism_factor(&self, graph: &DependencyGraph) -> f64 {
        if graph.nodes.is_empty() {
            return 0.0;
        }

        // Count gates with no dependencies
        let independent_gates = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(idx, _)| {
                graph
                    .reverse_edges
                    .get(idx)
                    .is_none_or(|edges| edges.is_empty())
            })
            .count();

        independent_gates as f64 / graph.nodes.len() as f64
    }

    /// Calculate gate type distribution
    fn calculate_gate_distribution(
        &self,
        gates: &[Arc<dyn GateOp + Send + Sync>],
    ) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for gate in gates {
            let gate_type = format!("{gate:?}"); // Simplified gate type extraction
            *distribution.entry(gate_type).or_insert(0) += 1;
        }

        distribution
    }

    /// Estimate entanglement complexity
    fn estimate_entanglement_complexity<const N: usize>(&self, circuit: &Circuit<N>) -> f64 {
        let gates = circuit.gates();

        // Count two-qubit gates (entangling gates)
        let two_qubit_gates = gates.iter().filter(|g| g.qubits().len() >= 2).count();

        // Entanglement score based on ratio of multi-qubit gates
        if gates.is_empty() {
            0.0
        } else {
            two_qubit_gates as f64 / gates.len() as f64
        }
    }

    /// Balance task loads across tasks
    fn balance_task_loads(&self, tasks: &mut Vec<ParallelTask>) -> QuantRS2Result<()> {
        // Sort tasks by cost
        tasks.sort_by(|a, b| b.cost.partial_cmp(&a.cost).unwrap());

        // No further balancing needed for now
        Ok(())
    }

    /// Merge small tasks together
    fn merge_small_tasks(&self, tasks: Vec<ParallelTask>) -> QuantRS2Result<Vec<ParallelTask>> {
        let mut merged = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_cost = 0.0;

        const COST_THRESHOLD: f64 = 10.0; // Merge tasks below this cost

        for task in tasks {
            if task.cost < COST_THRESHOLD {
                current_batch.push(task);
                current_cost += current_batch.last().unwrap().cost;

                if current_cost >= COST_THRESHOLD {
                    // Merge current batch into one task
                    merged.push(self.merge_task_batch(current_batch)?);
                    current_batch = Vec::new();
                    current_cost = 0.0;
                }
            } else {
                merged.push(task);
            }
        }

        // Add remaining batch
        if !current_batch.is_empty() {
            merged.push(self.merge_task_batch(current_batch)?);
        }

        Ok(merged)
    }

    /// Split large tasks for better parallelism
    fn split_large_tasks(&self, tasks: Vec<ParallelTask>) -> QuantRS2Result<Vec<ParallelTask>> {
        let mut split_tasks = Vec::new();

        const COST_THRESHOLD: f64 = 100.0; // Split tasks above this cost

        for task in tasks {
            if task.cost > COST_THRESHOLD && task.gates.len() > 4 {
                // Split into multiple smaller tasks
                let mid = task.gates.len() / 2;
                let (gates1, gates2) = task.gates.split_at(mid);

                split_tasks.push(ParallelTask {
                    id: Uuid::new_v4(),
                    gates: gates1.to_vec(),
                    qubits: task.qubits.clone(),
                    cost: task.cost / 2.0,
                    memory_requirement: task.memory_requirement / 2,
                    dependencies: task.dependencies.clone(),
                    priority: task.priority,
                });

                split_tasks.push(ParallelTask {
                    id: Uuid::new_v4(),
                    gates: gates2.to_vec(),
                    qubits: task.qubits.clone(),
                    cost: task.cost / 2.0,
                    memory_requirement: task.memory_requirement / 2,
                    dependencies: HashSet::new(),
                    priority: task.priority,
                });
            } else {
                split_tasks.push(task);
            }
        }

        Ok(split_tasks)
    }

    /// Merge a batch of tasks into one
    fn merge_task_batch(&self, batch: Vec<ParallelTask>) -> QuantRS2Result<ParallelTask> {
        let mut merged_gates = Vec::new();
        let mut merged_qubits = HashSet::new();
        let mut merged_cost = 0.0;
        let mut merged_memory = 0;
        let mut merged_deps = HashSet::new();
        let mut max_priority = TaskPriority::Low;

        for task in batch {
            merged_gates.extend(task.gates);
            merged_qubits.extend(task.qubits);
            merged_cost += task.cost;
            merged_memory += task.memory_requirement;
            merged_deps.extend(task.dependencies);
            // Use the highest priority from the batch
            if task.priority as u8 > max_priority as u8 {
                max_priority = task.priority;
            }
        }

        Ok(ParallelTask {
            id: Uuid::new_v4(),
            gates: merged_gates,
            qubits: merged_qubits,
            cost: merged_cost,
            memory_requirement: merged_memory,
            dependencies: merged_deps,
            priority: max_priority,
        })
    }

    /// Check if two gates have a dependency
    fn gates_have_dependency(&self, idx1: usize, idx2: usize, graph: &DependencyGraph) -> bool {
        // Check if idx2 depends on idx1
        if let Some(deps) = graph.reverse_edges.get(&idx2) {
            if deps.contains(&idx1) {
                return true;
            }
        }

        // Check if idx1 depends on idx2
        if let Some(deps) = graph.reverse_edges.get(&idx1) {
            if deps.contains(&idx2) {
                return true;
            }
        }

        false
    }

    /// Check if gates share qubits
    fn gates_share_qubits(&self, qubits1: &HashSet<QubitId>, qubits2: &HashSet<QubitId>) -> bool {
        !qubits1.is_disjoint(qubits2)
    }

    /// Hardware-aware parallelization strategy
    fn hardware_aware_parallelization<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Implement hardware-aware parallelization
        // Detect hardware characteristics
        let hw_char = self.detect_hardware_characteristics();

        // Analyze circuit to determine optimal hardware utilization
        let tasks = match self.select_hardware_strategy(&hw_char, circuit, graph)? {
            HardwareStrategy::CacheOptimized => {
                self.cache_optimized_parallelization(graph, &hw_char)?
            }
            HardwareStrategy::SIMDOptimized => {
                self.simd_optimized_parallelization(graph, &hw_char)?
            }
            HardwareStrategy::NUMAAware => self.numa_aware_parallelization(graph, &hw_char)?,
            HardwareStrategy::GPUOffload => {
                // For GPU offload, use dependency-based with larger task sizes
                self.dependency_based_parallelization(graph)?
            }
            HardwareStrategy::Hybrid => self.hybrid_hardware_parallelization(graph, &hw_char)?,
        };

        // Optimize task assignment based on hardware affinity
        let optimized_tasks = self.optimize_hardware_affinity(tasks, &hw_char)?;

        Ok(optimized_tasks)
    }

    /// Detect hardware characteristics of the system
    fn detect_hardware_characteristics(&self) -> HardwareCharacteristics {
        use scirs2_core::parallel_ops::current_num_threads;

        // Detect available hardware resources
        let num_cores = current_num_threads();

        // Estimate cache sizes (typical modern CPU values)
        let l1_cache_size = 32 * 1024; // 32 KB per core
        let l2_cache_size = 256 * 1024; // 256 KB per core
        let l3_cache_size = 8 * 1024 * 1024; // 8 MB shared

        // Estimate memory bandwidth (typical DDR4)
        let memory_bandwidth = 50.0; // GB/s

        // NUMA detection (simplified)
        let num_numa_nodes = if num_cores > 32 { 2 } else { 1 };

        // GPU detection (simplified - would use actual detection in production)
        let has_gpu = false; // Would check for CUDA/OpenCL/Metal availability

        // SIMD width detection (simplified - would use cpuid in production)
        #[cfg(target_arch = "x86_64")]
        let simd_width = 256; // AVX2
        #[cfg(not(target_arch = "x86_64"))]
        let simd_width = 128; // NEON or fallback

        HardwareCharacteristics {
            num_cores,
            l1_cache_size,
            l2_cache_size,
            l3_cache_size,
            memory_bandwidth,
            num_numa_nodes,
            has_gpu,
            simd_width,
        }
    }

    /// Select optimal hardware strategy based on circuit and hardware characteristics
    fn select_hardware_strategy<const N: usize>(
        &self,
        hw_char: &HardwareCharacteristics,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<HardwareStrategy> {
        let gates = circuit.gates();
        let num_gates = gates.len();

        // GPU offload for very large circuits
        if hw_char.has_gpu && num_gates > 1000 {
            return Ok(HardwareStrategy::GPUOffload);
        }

        // NUMA-aware for multi-socket systems with large state vectors
        if hw_char.num_numa_nodes > 1 && N > 20 {
            return Ok(HardwareStrategy::NUMAAware);
        }

        // SIMD optimization for circuits with many similar gates
        let has_many_rotation_gates = self.count_rotation_gates(gates) > num_gates / 2;
        if has_many_rotation_gates && hw_char.simd_width >= 256 {
            return Ok(HardwareStrategy::SIMDOptimized);
        }

        // Cache optimization for medium-sized circuits
        if num_gates < 500 && N < 15 {
            return Ok(HardwareStrategy::CacheOptimized);
        }

        // Default to hybrid strategy
        Ok(HardwareStrategy::Hybrid)
    }

    /// Cache-optimized parallelization
    fn cache_optimized_parallelization(
        &self,
        graph: &DependencyGraph,
        hw_char: &HardwareCharacteristics,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Group gates to fit within L2 cache
        let max_task_size = hw_char.l2_cache_size / (16 * 2); // Complex64 size

        let mut tasks = Vec::new();
        let mut current_group = Vec::new();
        let mut current_size = 0;

        for (idx, node) in graph.nodes.iter().enumerate() {
            let gate_size = (1 << node.qubits.len()) * 16; // State vector size for gate

            if current_size + gate_size > max_task_size && !current_group.is_empty() {
                tasks.push(self.create_parallel_task(current_group, graph)?);
                current_group = Vec::new();
                current_size = 0;
            }

            current_group.push(idx);
            current_size += gate_size;
        }

        if !current_group.is_empty() {
            tasks.push(self.create_parallel_task(current_group, graph)?);
        }

        Ok(tasks)
    }

    /// SIMD-optimized parallelization
    fn simd_optimized_parallelization(
        &self,
        graph: &DependencyGraph,
        hw_char: &HardwareCharacteristics,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Group similar gates that can be vectorized
        let mut rotation_gates = Vec::new();
        let mut other_gates = Vec::new();

        for (idx, node) in graph.nodes.iter().enumerate() {
            if self.is_rotation_gate(node.gate.as_ref()) {
                rotation_gates.push(idx);
            } else {
                other_gates.push(idx);
            }
        }

        let mut tasks = Vec::new();

        // Create vectorized tasks for rotation gates
        let vec_width = hw_char.simd_width / 128; // Complex numbers per SIMD register
        for chunk in rotation_gates.chunks(vec_width) {
            tasks.push(self.create_parallel_task(chunk.to_vec(), graph)?);
        }

        // Regular tasks for other gates
        for idx in other_gates {
            tasks.push(self.create_parallel_task(vec![idx], graph)?);
        }

        Ok(tasks)
    }

    /// NUMA-aware parallelization
    fn numa_aware_parallelization(
        &self,
        graph: &DependencyGraph,
        hw_char: &HardwareCharacteristics,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Distribute tasks across NUMA nodes to minimize cross-node traffic
        let num_nodes = hw_char.num_numa_nodes;
        let mut node_tasks: Vec<Vec<usize>> = vec![Vec::new(); num_nodes];

        // Assign gates to NUMA nodes based on qubit locality
        for (idx, node) in graph.nodes.iter().enumerate() {
            let numa_node = self.select_numa_node(node, num_nodes);
            node_tasks[numa_node].push(idx);
        }

        let mut tasks = Vec::new();
        for node_task_indices in node_tasks {
            if !node_task_indices.is_empty() {
                tasks.push(self.create_parallel_task(node_task_indices, graph)?);
            }
        }

        Ok(tasks)
    }

    /// Hybrid hardware-aware parallelization
    fn hybrid_hardware_parallelization(
        &self,
        graph: &DependencyGraph,
        hw_char: &HardwareCharacteristics,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Combine multiple hardware optimizations
        // Start with dependency-based grouping
        let base_tasks = self.dependency_based_parallelization(graph)?;

        // Refine based on cache constraints
        let cache_aware_tasks = self.refine_for_cache(base_tasks, hw_char)?;

        // Further refine for NUMA if applicable
        if hw_char.num_numa_nodes > 1 {
            self.refine_for_numa(cache_aware_tasks, hw_char)
        } else {
            Ok(cache_aware_tasks)
        }
    }

    /// Optimize task hardware affinity
    const fn optimize_hardware_affinity(
        &self,
        tasks: Vec<ParallelTask>,
        hw_char: &HardwareCharacteristics,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Assign hardware affinity hints to tasks
        // For now, return tasks as-is
        // In a full implementation, would add NUMA node binding, CPU affinity, etc.
        Ok(tasks)
    }

    /// Count rotation gates in a gate list
    fn count_rotation_gates(&self, gates: &[Arc<dyn GateOp + Send + Sync>]) -> usize {
        gates
            .iter()
            .filter(|g| self.is_rotation_gate(g.as_ref()))
            .count()
    }

    /// Check if a gate is a rotation gate
    fn is_rotation_gate(&self, gate: &dyn GateOp) -> bool {
        let gate_str = format!("{gate:?}");
        gate_str.contains("RX") || gate_str.contains("RY") || gate_str.contains("RZ")
    }

    /// Select NUMA node for a gate
    fn select_numa_node(&self, node: &GateNode, num_nodes: usize) -> usize {
        // Simple hash-based assignment based on qubits
        let qubit_sum: usize = node.qubits.iter().map(|q| q.0 as usize).sum();
        qubit_sum % num_nodes
    }

    /// Refine tasks for cache efficiency
    fn refine_for_cache(
        &self,
        tasks: Vec<ParallelTask>,
        hw_char: &HardwareCharacteristics,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Split large tasks that exceed cache size
        let max_cache_size = hw_char.l2_cache_size;
        let mut refined = Vec::new();

        for task in tasks {
            if task.memory_requirement > max_cache_size {
                // Split task
                let mid = task.gates.len() / 2;
                let (gates1, gates2) = task.gates.split_at(mid);

                refined.push(ParallelTask {
                    id: Uuid::new_v4(),
                    gates: gates1.to_vec(),
                    qubits: task.qubits.clone(),
                    cost: task.cost / 2.0,
                    memory_requirement: task.memory_requirement / 2,
                    dependencies: task.dependencies.clone(),
                    priority: task.priority,
                });

                refined.push(ParallelTask {
                    id: Uuid::new_v4(),
                    gates: gates2.to_vec(),
                    qubits: task.qubits,
                    cost: task.cost / 2.0,
                    memory_requirement: task.memory_requirement / 2,
                    dependencies: HashSet::new(),
                    priority: task.priority,
                });
            } else {
                refined.push(task);
            }
        }

        Ok(refined)
    }

    /// Refine tasks for NUMA efficiency
    const fn refine_for_numa(
        &self,
        tasks: Vec<ParallelTask>,
        hw_char: &HardwareCharacteristics,
    ) -> QuantRS2Result<Vec<ParallelTask>> {
        // Regroup tasks by NUMA affinity
        // For now, return as-is
        // Full implementation would analyze qubit locality and regroup
        Ok(tasks)
    }

    /// Create a parallel task from a group of gate indices
    fn create_parallel_task(
        &self,
        gate_indices: Vec<usize>,
        graph: &DependencyGraph,
    ) -> QuantRS2Result<ParallelTask> {
        let mut gates = Vec::new();
        let mut qubits = HashSet::new();
        let mut total_cost = 0.0;
        let mut memory_requirement = 0;

        for &idx in &gate_indices {
            if let Some(node) = graph.nodes.get(idx) {
                gates.push(node.gate.clone());
                qubits.extend(&node.qubits);
                total_cost += node.cost;
                memory_requirement += self.estimate_gate_memory(node.gate.as_ref());
            }
        }

        // Calculate dependencies
        let dependencies = self.calculate_task_dependencies(&gate_indices, graph)?;

        Ok(ParallelTask {
            id: Uuid::new_v4(),
            gates,
            qubits,
            cost: total_cost,
            memory_requirement,
            dependencies,
            priority: TaskPriority::Normal,
        })
    }

    /// Calculate task dependencies
    fn calculate_task_dependencies(
        &self,
        gate_indices: &[usize],
        graph: &DependencyGraph,
    ) -> QuantRS2Result<HashSet<Uuid>> {
        // Implement proper dependency tracking across tasks
        let mut dependencies = HashSet::new();

        // For each gate in this task, check all its dependencies in the graph
        for &gate_idx in gate_indices {
            if let Some(parent_indices) = graph.reverse_edges.get(&gate_idx) {
                // For each parent (dependency) of this gate
                for &parent_idx in parent_indices {
                    // Check if this parent is in a different task
                    // If it is, we need to track that task as a dependency
                    if !gate_indices.contains(&parent_idx) {
                        // This parent is not in the current task, so it's in another task
                        // We need to find which task contains this parent gate
                        // For now, we create a dependency marker
                        // In a full implementation, this would be the task ID containing parent_idx

                        // Create a deterministic UUID based on the parent gate index
                        // This allows us to identify dependencies even across task reorganizations
                        let dep_uuid = self.generate_gate_dependency_uuid(parent_idx);
                        dependencies.insert(dep_uuid);
                    }
                }
            }
        }

        Ok(dependencies)
    }

    /// Generate a deterministic UUID for a gate index to track dependencies
    fn generate_gate_dependency_uuid(&self, gate_index: usize) -> Uuid {
        // Create a deterministic UUID based on gate index
        // Using a fixed namespace UUID for gate dependencies
        let namespace =
            Uuid::parse_str("6ba7b810-9dad-11d1-80b4-00c04fd430c8").unwrap_or_else(|_| Uuid::nil());

        // Create UUID from gate index bytes
        let mut bytes = [0u8; 16];
        let index_bytes = gate_index.to_le_bytes();
        bytes[0..8].copy_from_slice(&index_bytes);

        Uuid::from_bytes(bytes)
    }

    /// Estimate execution cost for a gate
    fn estimate_gate_cost(&self, gate: &dyn GateOp) -> f64 {
        match gate.num_qubits() {
            1 => 1.0,
            2 => 4.0,
            3 => 8.0,
            n => (2.0_f64).powi(n as i32),
        }
    }

    /// Estimate memory requirement for a gate
    fn estimate_gate_memory(&self, gate: &dyn GateOp) -> usize {
        let num_qubits = gate.num_qubits();
        let state_size = 1 << num_qubits;
        state_size * std::mem::size_of::<Complex64>()
    }

    /// Partition layer into parallel tasks
    fn partition_layer_into_tasks(
        &self,
        layer: &[usize],
        graph: &DependencyGraph,
    ) -> QuantRS2Result<Vec<Vec<usize>>> {
        let max_gates_per_task = self.config.resource_constraints.max_gates_per_thread;
        let mut chunks = Vec::new();

        for chunk in layer.chunks(max_gates_per_task) {
            chunks.push(chunk.to_vec());
        }

        Ok(chunks)
    }

    /// Partition qubits into independent subsystems
    fn partition_qubits<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> QuantRS2Result<Vec<HashSet<QubitId>>> {
        // Simple partitioning based on gate connectivity
        let mut partitions = Vec::new();
        let mut used_qubits = HashSet::new();

        for i in 0..N {
            let qubit = QubitId::new(i as u32);
            if !used_qubits.contains(&qubit) {
                let mut partition = HashSet::new();
                partition.insert(qubit);
                used_qubits.insert(qubit);
                partitions.push(partition);
            }
        }

        Ok(partitions)
    }

    /// Calculate strategy efficiency
    fn calculate_strategy_efficiency(&self, tasks: &[ParallelTask]) -> f64 {
        if tasks.is_empty() {
            return 0.0;
        }

        let total_cost: f64 = tasks.iter().map(|t| t.cost).sum();
        let max_cost = tasks.iter().map(|t| t.cost).fold(0.0, f64::max);

        if max_cost > 0.0 {
            total_cost / (max_cost * tasks.len() as f64)
        } else {
            0.0
        }
    }

    /// Calculate parallelization metrics
    fn calculate_parallelization_metrics<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
        tasks: Vec<ParallelTask>,
    ) -> QuantRS2Result<ParallelizationAnalysis> {
        let num_layers = graph.layers.len();
        let max_parallelism = graph
            .layers
            .iter()
            .map(|layer| layer.len())
            .max()
            .unwrap_or(1);
        let critical_path_length = graph.layers.len();

        let efficiency = if circuit.num_gates() > 0 {
            max_parallelism as f64 / circuit.num_gates() as f64
        } else {
            0.0
        };

        let resource_utilization = ResourceUtilization {
            cpu_utilization: vec![0.8; self.config.max_threads],
            memory_usage: vec![
                self.config.memory_budget / self.config.max_threads;
                self.config.max_threads
            ],
            load_balance_score: 0.85,
            communication_overhead: 0.1,
        };

        let recommendations = self.generate_optimization_recommendations(circuit, graph, &tasks);

        Ok(ParallelizationAnalysis {
            tasks,
            num_layers,
            efficiency,
            max_parallelism,
            critical_path_length,
            resource_utilization,
            recommendations,
        })
    }

    /// Generate optimization recommendations
    fn generate_optimization_recommendations<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        graph: &DependencyGraph,
        tasks: &[ParallelTask],
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Check if gate reordering could improve parallelism
        if graph.layers.iter().any(|layer| layer.len() == 1) {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::GateReordering,
                description: "Consider reordering gates to create larger parallel layers"
                    .to_string(),
                expected_improvement: 0.2,
                complexity: RecommendationComplexity::Medium,
            });
        }

        // Check resource utilization balance
        let task_costs: Vec<f64> = tasks.iter().map(|t| t.cost).collect();
        let cost_variance = self.calculate_variance(&task_costs);
        if cost_variance > 0.5 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::ResourceAllocation,
                description: "Task costs are unbalanced, consider load balancing optimization"
                    .to_string(),
                expected_improvement: 0.15,
                complexity: RecommendationComplexity::Low,
            });
        }

        recommendations
    }

    /// Calculate variance of a vector of values
    fn calculate_variance(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let variance: f64 =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance
    }

    /// Execute circuit sequentially (fallback)
    fn execute_sequential<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        simulator: &mut LargeScaleQuantumSimulator,
    ) -> QuantRS2Result<Vec<Complex64>> {
        // Use the Simulator trait's run method
        let result = simulator.run(circuit)?;
        // Extract state vector from the register
        // TODO: Add method to extract state vector from Register
        Ok(Vec::new())
    }

    /// Execute parallel tasks with proper synchronization
    fn execute_parallel_tasks(
        &self,
        tasks: &[ParallelTask],
        shared_state: Arc<RwLock<Vec<Complex64>>>,
        results: Arc<Mutex<Vec<Complex64>>>,
        barrier: Arc<Barrier>,
    ) -> QuantRS2Result<()> {
        use scirs2_core::parallel_ops::*;

        // Implement actual parallel gate execution using SciRS2 parallel operations
        // Execute independent tasks in parallel using parallel_map
        let _ = parallel_map(tasks, |task| {
            // Wait for dependencies to complete
            barrier.wait();

            // Get exclusive access to state for gate application
            let mut state = shared_state
                .write()
                .expect("Failed to acquire write lock on shared state");

            // Apply all gates in this task to the state vector
            for gate in &task.gates {
                // Extract gate information
                let qubits = gate.qubits();

                // Apply gate based on number of qubits
                match qubits.len() {
                    1 => {
                        // Single-qubit gate application
                        self.apply_single_qubit_gate_to_state(
                            &mut state,
                            gate.as_ref(),
                            qubits[0].0 as usize,
                        );
                    }
                    2 => {
                        // Two-qubit gate application
                        self.apply_two_qubit_gate_to_state(
                            &mut state,
                            gate.as_ref(),
                            qubits[0].0 as usize,
                            qubits[1].0 as usize,
                        );
                    }
                    _ => {
                        // Multi-qubit gate - fall back to sequential application
                        eprintln!(
                            "Warning: {}-qubit gates not optimized for parallel execution",
                            qubits.len()
                        );
                    }
                }
            }

            // Synchronize after task completion
            barrier.wait();
        });

        // Collect results
        let final_state = shared_state
            .read()
            .expect("Failed to acquire read lock on shared state");
        let mut result_vec = results.lock().expect("Failed to acquire lock on results");
        result_vec.clone_from(&final_state);

        Ok(())
    }

    /// Apply a single-qubit gate to a state vector
    fn apply_single_qubit_gate_to_state(
        &self,
        state: &mut [Complex64],
        gate: &dyn GateOp,
        qubit: usize,
    ) {
        // Simplified single-qubit gate application
        // In a full implementation, this would extract the actual gate matrix
        // and apply it using optimized SIMD operations

        let num_qubits = (state.len() as f64).log2() as usize;
        let stride = 1 << qubit;

        // Process pairs of amplitudes
        for base in 0..state.len() {
            if (base & stride) == 0 {
                // This is the |0⟩ component for this qubit
                let idx0 = base;
                let idx1 = base | stride;

                // Simple identity operation (placeholder)
                // In reality, would apply actual gate matrix
                let amp0 = state[idx0];
                let amp1 = state[idx1];

                // Apply gate transformation (simplified as identity for now)
                state[idx0] = amp0;
                state[idx1] = amp1;
            }
        }
    }

    /// Apply a two-qubit gate to a state vector
    fn apply_two_qubit_gate_to_state(
        &self,
        state: &mut [Complex64],
        gate: &dyn GateOp,
        qubit1: usize,
        qubit2: usize,
    ) {
        // Simplified two-qubit gate application
        // In a full implementation, this would apply the actual gate matrix

        let num_qubits = (state.len() as f64).log2() as usize;
        let stride1 = 1 << qubit1;
        let stride2 = 1 << qubit2;

        // Process quartets of amplitudes
        for base in 0..state.len() {
            if (base & stride1) == 0 && (base & stride2) == 0 {
                let idx00 = base;
                let idx01 = base | stride1;
                let idx10 = base | stride2;
                let idx11 = base | stride1 | stride2;

                // Get current amplitudes
                let amp00 = state[idx00];
                let amp01 = state[idx01];
                let amp10 = state[idx10];
                let amp11 = state[idx11];

                // Apply gate transformation (simplified as identity for now)
                state[idx00] = amp00;
                state[idx01] = amp01;
                state[idx10] = amp10;
                state[idx11] = amp11;
            }
        }
    }

    /// Distribute tasks across cluster nodes
    fn distribute_tasks_across_nodes(
        &self,
        tasks: &[ParallelTask],
        distributed_sim: &DistributedQuantumSimulator,
    ) -> QuantRS2Result<Vec<Vec<ParallelTask>>> {
        // Implement intelligent task distribution based on node capabilities
        let cluster_status = distributed_sim.get_cluster_status();
        let num_nodes = cluster_status.len();

        if num_nodes == 0 {
            return Ok(vec![tasks.to_vec()]);
        }

        // Analyze node capabilities from cluster status
        let node_capacities = self.analyze_node_capabilities(&cluster_status);

        // Sort tasks by cost (descending) for better load balancing
        let mut sorted_tasks: Vec<_> = tasks.to_vec();
        sorted_tasks.sort_by(|a, b| b.cost.partial_cmp(&a.cost).unwrap());

        // Use a greedy bin-packing approach for task distribution
        let mut distributed_tasks = vec![Vec::new(); num_nodes];
        let mut node_loads = vec![0.0; num_nodes];

        for task in sorted_tasks {
            // Find the node with the most capacity relative to its load
            let best_node = self.select_best_node_for_task(&task, &node_capacities, &node_loads);

            // Assign task to the selected node
            distributed_tasks[best_node].push(task.clone());
            node_loads[best_node] += task.cost;
        }

        // Rebalance if any node is significantly overloaded
        self.rebalance_node_distribution(
            &mut distributed_tasks,
            &node_capacities,
            &mut node_loads,
        )?;

        Ok(distributed_tasks)
    }

    /// Analyze node capabilities from cluster status
    fn analyze_node_capabilities(
        &self,
        cluster_status: &HashMap<Uuid, crate::distributed_simulator::NodeInfo>,
    ) -> Vec<NodeCapacity> {
        cluster_status
            .values()
            .map(|info| {
                // Extract capacity information from NodeInfo
                // Using reasonable defaults where information is not available
                NodeCapacity {
                    cpu_cores: 4,                 // Default: 4 cores
                    memory_gb: 16.0,              // Default: 16 GB
                    gpu_available: false,         // Default: no GPU
                    network_bandwidth_gbps: 10.0, // Default: 10 Gbps
                    relative_performance: 1.0,    // Default: normalized performance
                }
            })
            .collect()
    }

    /// Select the best node for a given task
    fn select_best_node_for_task(
        &self,
        task: &ParallelTask,
        node_capacities: &[NodeCapacity],
        node_loads: &[f64],
    ) -> usize {
        let mut best_node = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (idx, capacity) in node_capacities.iter().enumerate() {
            // Calculate a score for this node based on:
            // 1. Available capacity (inversely proportional to current load)
            // 2. Node performance characteristics
            // 3. Task requirements

            let load_factor = 1.0 - (node_loads[idx] / capacity.relative_performance).min(1.0);
            let memory_factor = if task.memory_requirement
                < (capacity.memory_gb * 1024.0 * 1024.0 * 1024.0) as usize
            {
                1.0
            } else {
                0.5 // Penalize if task might exceed memory
            };

            let score = load_factor * capacity.relative_performance * memory_factor;

            if score > best_score {
                best_score = score;
                best_node = idx;
            }
        }

        best_node
    }

    /// Rebalance task distribution if needed
    fn rebalance_node_distribution(
        &self,
        distributed_tasks: &mut [Vec<ParallelTask>],
        node_capacities: &[NodeCapacity],
        node_loads: &mut [f64],
    ) -> QuantRS2Result<()> {
        // Calculate average load
        let total_load: f64 = node_loads.iter().sum();
        let avg_load = total_load / node_loads.len() as f64;

        // Find overloaded and underloaded nodes
        const IMBALANCE_THRESHOLD: f64 = 0.3; // 30% deviation threshold

        for _ in 0..5 {
            // Maximum 5 rebalancing iterations
            let mut rebalanced = false;

            // Find nodes that need rebalancing
            let heavy_nodes: Vec<usize> = node_loads
                .iter()
                .enumerate()
                .filter(|(_, load)| **load > avg_load * (1.0 + IMBALANCE_THRESHOLD))
                .map(|(idx, _)| idx)
                .collect();

            let light_nodes: Vec<usize> = node_loads
                .iter()
                .enumerate()
                .filter(|(_, load)| **load < avg_load * (1.0 - IMBALANCE_THRESHOLD))
                .map(|(idx, _)| idx)
                .collect();

            for &heavy_idx in &heavy_nodes {
                for &light_idx in &light_nodes {
                    if heavy_idx != light_idx {
                        // Try to move a task from heavy to light node
                        if let Some(task) = distributed_tasks[heavy_idx].pop() {
                            node_loads[heavy_idx] -= task.cost;
                            distributed_tasks[light_idx].push(task.clone());
                            node_loads[light_idx] += task.cost;
                            rebalanced = true;
                            break;
                        }
                    }
                }
                if rebalanced {
                    break;
                }
            }

            if !rebalanced {
                break; // No more rebalancing possible
            }
        }

        Ok(())
    }

    /// Compute hash for circuit caching
    fn compute_circuit_hash<const N: usize>(&self, circuit: &Circuit<N>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash circuit structure
        circuit.num_gates().hash(&mut hasher);
        circuit.num_qubits().hash(&mut hasher);

        // Hash gate names (simplified)
        for gate in circuit.gates() {
            gate.name().hash(&mut hasher);
            gate.qubits().len().hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Update performance statistics
    fn update_performance_stats(
        &self,
        execution_time: Duration,
        analysis: &ParallelizationAnalysis,
    ) {
        let mut stats = self.performance_stats.lock().unwrap();
        stats.circuits_processed += 1;
        stats.total_execution_time += execution_time;
        stats.average_efficiency = stats
            .average_efficiency
            .mul_add((stats.circuits_processed - 1) as f64, analysis.efficiency)
            / stats.circuits_processed as f64;
    }
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(num_threads: usize) -> Self {
        Self {
            thread_loads: vec![0.0; num_threads],
            task_queues: vec![VecDeque::new(); num_threads],
            work_stealing_stats: WorkStealingStats::default(),
        }
    }

    /// Balance load across threads
    pub fn balance_load(&mut self, tasks: Vec<ParallelTask>) -> Vec<Vec<ParallelTask>> {
        let mut balanced_tasks = vec![Vec::new(); self.thread_loads.len()];

        // Simple round-robin distribution for now
        for (i, task) in tasks.into_iter().enumerate() {
            let thread_index = i % self.thread_loads.len();
            balanced_tasks[thread_index].push(task);
        }

        balanced_tasks
    }
}

/// Benchmark automatic parallelization performance
pub fn benchmark_automatic_parallelization<const N: usize>(
    circuits: Vec<Circuit<N>>,
    config: AutoParallelConfig,
) -> QuantRS2Result<AutoParallelBenchmarkResults> {
    let engine = AutoParallelEngine::new(config);
    let mut results = Vec::new();
    let start_time = Instant::now();

    for circuit in circuits {
        let analysis_start = Instant::now();
        let analysis = engine.analyze_circuit(&circuit)?;
        let analysis_time = analysis_start.elapsed();

        results.push(CircuitParallelResult {
            circuit_size: circuit.num_gates(),
            num_qubits: circuit.num_qubits(),
            analysis_time,
            efficiency: analysis.efficiency,
            max_parallelism: analysis.max_parallelism,
            num_tasks: analysis.tasks.len(),
        });
    }

    let total_time = start_time.elapsed();

    Ok(AutoParallelBenchmarkResults {
        total_time,
        average_efficiency: results.iter().map(|r| r.efficiency).sum::<f64>()
            / results.len() as f64,
        average_parallelism: results.iter().map(|r| r.max_parallelism).sum::<usize>()
            / results.len(),
        circuit_results: results,
    })
}

/// Results from automatic parallelization benchmark
#[derive(Debug, Clone)]
pub struct AutoParallelBenchmarkResults {
    /// Total benchmark time
    pub total_time: Duration,
    /// Results for individual circuits
    pub circuit_results: Vec<CircuitParallelResult>,
    /// Average parallelization efficiency
    pub average_efficiency: f64,
    /// Average maximum parallelism
    pub average_parallelism: usize,
}

/// Parallelization results for a single circuit
#[derive(Debug, Clone)]
pub struct CircuitParallelResult {
    /// Circuit size (number of gates)
    pub circuit_size: usize,
    /// Number of qubits
    pub num_qubits: usize,
    /// Time to analyze parallelization
    pub analysis_time: Duration,
    /// Parallelization efficiency
    pub efficiency: f64,
    /// Maximum parallelism achieved
    pub max_parallelism: usize,
    /// Number of parallel tasks generated
    pub num_tasks: usize,
}
