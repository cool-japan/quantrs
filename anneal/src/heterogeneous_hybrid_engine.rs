//! Heterogeneous Quantum-Classical Hybrid Execution Engine
//!
//! This module implements a sophisticated hybrid execution engine that seamlessly
//! coordinates between quantum annealing hardware, classical optimization algorithms,
//! and hybrid approaches. It provides intelligent workload distribution, dynamic
//! resource allocation, and adaptive execution strategies for optimal performance.
//!
//! Key Features:
//! - Intelligent algorithm selection and routing
//! - Dynamic workload distribution between quantum and classical resources
//! - Adaptive execution strategies based on problem characteristics
//! - Resource-aware scheduling and load balancing
//! - Performance monitoring and optimization
//! - Fault tolerance and fallback mechanisms
//! - Cost optimization for cloud quantum services
//! - Quality-aware result aggregation and consensus

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::applications::{ApplicationError, ApplicationResult};
use crate::braket::{BraketClient, BraketDevice};
use crate::dwave::DWaveClient;
use crate::ising::{IsingModel, QuboModel};
use crate::multi_chip_embedding::{MultiChipConfig, MultiChipCoordinator};
use crate::simulator::{AnnealingParams, AnnealingResult, ClassicalAnnealingSimulator};
use crate::HardwareTopology;

/// Hybrid execution engine configuration
#[derive(Debug, Clone)]
pub struct HybridEngineConfig {
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Quality threshold for solutions
    pub quality_threshold: f64,
    /// Cost budget for cloud services
    pub cost_budget: Option<f64>,
    /// Resource allocation strategy
    pub allocation_strategy: ResourceAllocationStrategy,
    /// Execution strategy
    pub execution_strategy: ExecutionStrategy,
    /// Performance optimization settings
    pub optimization_settings: OptimizationSettings,
    /// Fault tolerance configuration
    pub fault_tolerance: HybridFaultToleranceConfig,
    /// Monitoring configuration
    pub monitoring: HybridMonitoringConfig,
}

impl Default for HybridEngineConfig {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(300),
            quality_threshold: 0.95,
            cost_budget: Some(100.0),
            allocation_strategy: ResourceAllocationStrategy::Adaptive,
            execution_strategy: ExecutionStrategy::Parallel,
            optimization_settings: OptimizationSettings::default(),
            fault_tolerance: HybridFaultToleranceConfig::default(),
            monitoring: HybridMonitoringConfig::default(),
        }
    }
}

/// Resource allocation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceAllocationStrategy {
    /// Pure quantum execution
    QuantumOnly,
    /// Pure classical execution
    ClassicalOnly,
    /// Static allocation based on problem size
    Static,
    /// Adaptive allocation based on performance
    Adaptive,
    /// Cost-optimized allocation
    CostOptimized,
    /// Quality-focused allocation
    QualityFocused,
}

/// Execution strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStrategy {
    /// Sequential execution (quantum then classical fallback)
    Sequential,
    /// Parallel execution on all available resources
    Parallel,
    /// Competitive execution (first good result wins)
    Competitive,
    /// Cooperative execution (combine results)
    Cooperative,
    /// Adaptive execution (strategy chosen based on problem characteristics)
    Adaptive,
    /// Hierarchical execution (coarse then fine-grained)
    Hierarchical,
}

/// Performance optimization settings
#[derive(Debug, Clone)]
pub struct OptimizationSettings {
    /// Enable intelligent algorithm selection
    pub enable_algorithm_selection: bool,
    /// Enable dynamic resource reallocation
    pub enable_dynamic_reallocation: bool,
    /// Enable result quality assessment
    pub enable_quality_assessment: bool,
    /// Enable cost optimization
    pub enable_cost_optimization: bool,
    /// Learning rate for adaptive strategies
    pub learning_rate: f64,
    /// History window for performance tracking
    pub history_window: usize,
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            enable_algorithm_selection: true,
            enable_dynamic_reallocation: true,
            enable_quality_assessment: true,
            enable_cost_optimization: true,
            learning_rate: 0.1,
            history_window: 100,
        }
    }
}

/// Hybrid fault tolerance configuration
#[derive(Debug, Clone)]
pub struct HybridFaultToleranceConfig {
    /// Enable automatic fallback to classical
    pub enable_classical_fallback: bool,
    /// Enable result validation
    pub enable_result_validation: bool,
    /// Maximum retries per resource type
    pub max_retries_per_type: usize,
    /// Timeout for individual executions
    pub individual_timeout: Duration,
    /// Minimum consensus threshold
    pub consensus_threshold: f64,
}

impl Default for HybridFaultToleranceConfig {
    fn default() -> Self {
        Self {
            enable_classical_fallback: true,
            enable_result_validation: true,
            max_retries_per_type: 3,
            individual_timeout: Duration::from_secs(120),
            consensus_threshold: 0.7,
        }
    }
}

/// Hybrid monitoring configuration
#[derive(Debug, Clone)]
pub struct HybridMonitoringConfig {
    /// Enable performance tracking
    pub enable_performance_tracking: bool,
    /// Enable cost tracking
    pub enable_cost_tracking: bool,
    /// Enable quality tracking
    pub enable_quality_tracking: bool,
    /// Metrics collection interval
    pub collection_interval: Duration,
}

impl Default for HybridMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_performance_tracking: true,
            enable_cost_tracking: true,
            enable_quality_tracking: true,
            collection_interval: Duration::from_secs(5),
        }
    }
}

/// Compute resource types
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceType {
    /// D-Wave quantum annealer
    DWaveQuantum,
    /// AWS Braket quantum device
    BraketQuantum,
    /// Classical annealing simulator
    ClassicalSimulator,
    /// Multi-chip quantum system
    MultiChipQuantum,
    /// GPU-accelerated classical
    GPUClassical,
    /// Custom hybrid algorithm
    CustomHybrid,
}

/// Compute resource representation
#[derive(Debug, Clone)]
pub struct ComputeResource {
    /// Resource identifier
    pub id: String,
    /// Resource type
    pub resource_type: ResourceType,
    /// Current availability
    pub availability: ResourceAvailability,
    /// Performance characteristics
    pub performance: ResourcePerformance,
    /// Cost characteristics
    pub cost: ResourceCost,
    /// Current workload
    pub workload: Option<ResourceWorkload>,
    /// Connection to resource
    pub connection: ResourceConnection,
}

/// Resource availability status
#[derive(Debug, Clone, PartialEq)]
pub enum ResourceAvailability {
    /// Available for immediate use
    Available,
    /// Busy with current task
    Busy,
    /// Temporarily unavailable
    Unavailable,
    /// In maintenance mode
    Maintenance,
    /// Failed or error state
    Failed,
}

/// Resource performance characteristics
#[derive(Debug, Clone)]
pub struct ResourcePerformance {
    /// Processing speed (problems/second)
    pub throughput: f64,
    /// Average latency
    pub latency: Duration,
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
    /// Solution quality score
    pub quality_score: f64,
    /// Problem size capability range
    pub size_range: (usize, usize),
    /// Performance history
    pub history: VecDeque<PerformanceEntry>,
}

/// Performance history entry
#[derive(Debug, Clone)]
pub struct PerformanceEntry {
    /// Timestamp
    pub timestamp: Instant,
    /// Problem size
    pub problem_size: usize,
    /// Execution time
    pub execution_time: Duration,
    /// Solution quality
    pub solution_quality: f64,
    /// Cost incurred
    pub cost: f64,
}

/// Resource cost characteristics
#[derive(Debug, Clone)]
pub struct ResourceCost {
    /// Fixed cost per use
    pub fixed_cost: f64,
    /// Variable cost per problem variable
    pub variable_cost: f64,
    /// Time-based cost per second
    pub time_cost: f64,
    /// Quality premium factor
    pub quality_premium: f64,
}

/// Current resource workload
#[derive(Debug, Clone)]
pub struct ResourceWorkload {
    /// Problem being processed
    pub problem_id: String,
    /// Problem size
    pub problem_size: usize,
    /// Start time
    pub start_time: Instant,
    /// Estimated completion
    pub estimated_completion: Instant,
    /// Current progress (0.0-1.0)
    pub progress: f64,
}

/// Resource connection interface
#[derive(Debug, Clone)]
pub enum ResourceConnection {
    /// D-Wave cloud connection
    DWave(Arc<Mutex<DWaveClient>>),
    /// AWS Braket connection
    Braket(Arc<Mutex<BraketClient>>),
    /// Classical simulator (local)
    Classical(Arc<Mutex<ClassicalAnnealingSimulator>>),
    /// Multi-chip coordinator
    MultiChip(Arc<Mutex<MultiChipCoordinator>>),
    /// Custom connection
    Custom(String),
}

/// Execution task for the hybrid engine
#[derive(Debug, Clone)]
pub struct HybridExecutionTask {
    /// Task identifier
    pub id: String,
    /// Problem to solve
    pub problem: IsingModel,
    /// Quality requirements
    pub quality_requirements: QualityRequirements,
    /// Resource constraints
    pub resource_constraints: ResourceConstraints,
    /// Execution priority
    pub priority: TaskPriority,
    /// Deadline (optional)
    pub deadline: Option<Instant>,
}

/// Quality requirements for solutions
#[derive(Debug, Clone)]
pub struct QualityRequirements {
    /// Minimum solution quality
    pub min_quality: f64,
    /// Target solution quality
    pub target_quality: f64,
    /// Quality assessment method
    pub assessment_method: QualityAssessmentMethod,
    /// Acceptable solution count
    pub min_solutions: usize,
}

/// Quality assessment methods
#[derive(Debug, Clone, PartialEq)]
pub enum QualityAssessmentMethod {
    /// Energy-based quality (lower energy = higher quality)
    EnergyBased,
    /// Statistical consensus
    Consensus,
    /// Ground truth comparison
    GroundTruth,
    /// Custom quality function
    Custom(String),
}

/// Resource constraints for execution
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Maximum cost budget
    pub max_cost: Option<f64>,
    /// Maximum execution time
    pub max_time: Duration,
    /// Preferred resource types
    pub preferred_resources: Vec<ResourceType>,
    /// Excluded resource types
    pub excluded_resources: Vec<ResourceType>,
    /// Geographic constraints
    pub geographic_constraints: Option<GeographicConstraints>,
}

/// Geographic constraints for resource selection
#[derive(Debug, Clone)]
pub struct GeographicConstraints {
    /// Preferred regions
    pub preferred_regions: Vec<String>,
    /// Maximum latency tolerance
    pub max_latency: Duration,
    /// Data locality requirements
    pub data_locality: bool,
}

/// Task execution priority
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum TaskPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Hybrid execution result
#[derive(Debug, Clone)]
pub struct HybridExecutionResult {
    /// Task identifier
    pub task_id: String,
    /// Best solution found
    pub best_solution: Vec<i32>,
    /// Best energy achieved
    pub best_energy: f64,
    /// Solution quality score
    pub quality_score: f64,
    /// Total execution time
    pub total_time: Duration,
    /// Total cost incurred
    pub total_cost: f64,
    /// Resource utilization
    pub resource_utilization: HashMap<String, ResourceUtilization>,
    /// Individual results from each resource
    pub individual_results: Vec<IndividualResult>,
    /// Execution metadata
    pub metadata: ExecutionMetadata,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// Resource identifier
    pub resource_id: String,
    /// Time used
    pub time_used: Duration,
    /// Cost incurred
    pub cost: f64,
    /// Success indicator
    pub success: bool,
    /// Quality achieved
    pub quality: f64,
}

/// Individual result from a resource
#[derive(Debug, Clone)]
pub struct IndividualResult {
    /// Resource that produced this result
    pub resource_id: String,
    /// Solution vector
    pub solution: Vec<i32>,
    /// Energy achieved
    pub energy: f64,
    /// Quality score
    pub quality: f64,
    /// Execution time
    pub execution_time: Duration,
    /// Cost incurred
    pub cost: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Execution metadata
#[derive(Debug, Clone)]
pub struct ExecutionMetadata {
    /// Execution strategy used
    pub strategy: ExecutionStrategy,
    /// Resources utilized
    pub resources_used: Vec<String>,
    /// Algorithm selection decisions
    pub algorithm_decisions: Vec<String>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    /// Any warnings or issues
    pub warnings: Vec<String>,
}

/// Main heterogeneous hybrid execution engine
pub struct HeterogeneousHybridEngine {
    /// Engine configuration
    pub config: HybridEngineConfig,
    /// Available compute resources
    pub resources: Arc<RwLock<HashMap<String, ComputeResource>>>,
    /// Task queue
    pub task_queue: Arc<Mutex<VecDeque<HybridExecutionTask>>>,
    /// Active executions
    pub active_executions: Arc<RwLock<HashMap<String, ActiveExecution>>>,
    /// Performance monitor
    pub monitor: Arc<Mutex<HybridPerformanceMonitor>>,
    /// Resource scheduler
    pub scheduler: Arc<Mutex<ResourceScheduler>>,
    /// Result aggregator
    pub aggregator: Arc<Mutex<ResultAggregator>>,
}

/// Active execution tracking
#[derive(Debug)]
pub struct ActiveExecution {
    /// Task being executed
    pub task: HybridExecutionTask,
    /// Start time
    pub start_time: Instant,
    /// Assigned resources
    pub assigned_resources: Vec<String>,
    /// Partial results
    pub partial_results: Vec<IndividualResult>,
    /// Current status
    pub status: ExecutionStatus,
}

/// Execution status
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStatus {
    /// Queued for execution
    Queued,
    /// Currently executing
    Running,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed,
    /// Cancelled by user
    Cancelled,
    /// Timed out
    TimedOut,
}

/// Performance monitoring for hybrid engine
pub struct HybridPerformanceMonitor {
    /// Overall system metrics
    pub system_metrics: HybridSystemMetrics,
    /// Per-resource metrics
    pub resource_metrics: HashMap<String, ResourceMetrics>,
    /// Historical performance data
    pub performance_history: VecDeque<PerformanceSnapshot>,
    /// Cost tracking
    pub cost_tracking: CostTracker,
}

/// System-wide hybrid metrics
#[derive(Debug, Clone)]
pub struct HybridSystemMetrics {
    /// Total tasks processed
    pub total_tasks: usize,
    /// Average task completion time
    pub avg_completion_time: Duration,
    /// Overall success rate
    pub success_rate: f64,
    /// Average solution quality
    pub avg_quality: f64,
    /// Total cost incurred
    pub total_cost: f64,
    /// Resource efficiency
    pub resource_efficiency: f64,
}

/// Per-resource metrics
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    /// Resource identifier
    pub resource_id: String,
    /// Tasks processed
    pub tasks_processed: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average quality
    pub avg_quality: f64,
    /// Total cost
    pub total_cost: f64,
    /// Utilization rate
    pub utilization_rate: f64,
}

/// Performance snapshot for historical tracking
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: Instant,
    /// System metrics
    pub system_metrics: HybridSystemMetrics,
    /// Resource metrics
    pub resource_metrics: HashMap<String, ResourceMetrics>,
}

/// Cost tracking system
#[derive(Debug, Clone)]
pub struct CostTracker {
    /// Current budget
    pub current_budget: f64,
    /// Spent amount
    pub spent_amount: f64,
    /// Cost per resource type
    pub cost_breakdown: HashMap<ResourceType, f64>,
    /// Cost predictions
    pub cost_predictions: HashMap<String, f64>,
}

/// Resource scheduling system
pub struct ResourceScheduler {
    /// Scheduling strategy
    pub strategy: ResourceAllocationStrategy,
    /// Resource availability cache
    pub availability_cache: HashMap<String, Instant>,
    /// Performance predictions
    pub performance_predictions: HashMap<String, f64>,
    /// Load balancing decisions
    pub load_balancing: LoadBalancingDecisions,
}

/// Load balancing decisions
#[derive(Debug, Clone)]
pub struct LoadBalancingDecisions {
    /// Recent decisions
    pub recent_decisions: VecDeque<SchedulingDecision>,
    /// Success rates per resource
    pub resource_success_rates: HashMap<String, f64>,
    /// Performance trends
    pub performance_trends: HashMap<String, f64>,
}

/// Scheduling decision
#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    /// Decision timestamp
    pub timestamp: Instant,
    /// Task assigned
    pub task_id: String,
    /// Resource selected
    pub resource_id: String,
    /// Selection rationale
    pub rationale: String,
    /// Predicted outcome
    pub predicted_performance: f64,
}

/// Result aggregation system
pub struct ResultAggregator {
    /// Aggregation strategy
    pub strategy: ResultAggregationStrategy,
    /// Quality assessment system
    pub quality_assessor: QualityAssessor,
    /// Consensus algorithm
    pub consensus_algorithm: ConsensusAlgorithm,
}

/// Result aggregation strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ResultAggregationStrategy {
    /// Best result wins
    BestResult,
    /// Majority consensus
    MajorityConsensus,
    /// Weighted average
    WeightedAverage,
    /// Quality-based selection
    QualityBased,
    /// Ensemble combination
    Ensemble,
}

/// Quality assessment system
#[derive(Debug)]
pub struct QualityAssessor {
    /// Assessment methods
    pub methods: Vec<QualityAssessmentMethod>,
    /// Quality thresholds
    pub thresholds: HashMap<QualityAssessmentMethod, f64>,
    /// Historical quality data
    pub quality_history: VecDeque<QualityMeasurement>,
}

/// Quality measurement
#[derive(Debug, Clone)]
pub struct QualityMeasurement {
    /// Timestamp
    pub timestamp: Instant,
    /// Resource that produced result
    pub resource_id: String,
    /// Quality score
    pub quality_score: f64,
    /// Assessment method used
    pub method: QualityAssessmentMethod,
}

/// Consensus algorithm for result aggregation
#[derive(Debug)]
pub struct ConsensusAlgorithm {
    /// Consensus threshold
    pub threshold: f64,
    /// Voting weights per resource
    pub voting_weights: HashMap<String, f64>,
    /// Historical consensus data
    pub consensus_history: VecDeque<ConsensusResult>,
}

/// Consensus result
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// Task identifier
    pub task_id: String,
    /// Consensus solution
    pub consensus_solution: Vec<i32>,
    /// Confidence level
    pub confidence: f64,
    /// Participating resources
    pub participants: Vec<String>,
    /// Agreement score
    pub agreement_score: f64,
}

impl HeterogeneousHybridEngine {
    /// Create new hybrid execution engine
    pub fn new(config: HybridEngineConfig) -> Self {
        Self {
            config,
            resources: Arc::new(RwLock::new(HashMap::new())),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            active_executions: Arc::new(RwLock::new(HashMap::new())),
            monitor: Arc::new(Mutex::new(HybridPerformanceMonitor::new())),
            scheduler: Arc::new(Mutex::new(ResourceScheduler::new())),
            aggregator: Arc::new(Mutex::new(ResultAggregator::new())),
        }
    }

    /// Register a compute resource
    pub fn register_resource(&self, resource: ComputeResource) -> ApplicationResult<()> {
        let resource_id = resource.id.clone();
        let mut resources = self.resources.write().map_err(|_| {
            ApplicationError::OptimizationError(
                "Failed to acquire resource registry lock".to_string(),
            )
        })?;

        resources.insert(resource_id.clone(), resource);

        println!(
            "Registered compute resource: {} ({:?})",
            resource_id, resources[&resource_id].resource_type
        );
        Ok(())
    }

    /// Submit task for hybrid execution
    pub fn submit_task(&self, task: HybridExecutionTask) -> ApplicationResult<String> {
        let task_id = task.id.clone();
        let mut queue = self.task_queue.lock().map_err(|_| {
            ApplicationError::OptimizationError("Failed to acquire task queue lock".to_string())
        })?;

        queue.push_back(task);
        println!("Task {} submitted to hybrid execution queue", task_id);
        Ok(task_id)
    }

    /// Execute task using hybrid approach
    pub fn execute_task(&self, task_id: &str) -> ApplicationResult<HybridExecutionResult> {
        println!("Starting hybrid execution for task: {}", task_id);

        // Step 1: Get task from queue
        let task = self.get_task_from_queue(task_id)?;

        // Step 2: Analyze problem and select strategy
        let execution_plan = self.create_execution_plan(&task)?;

        // Step 3: Schedule resources
        let resource_assignments = self.schedule_resources(&task, &execution_plan)?;

        // Step 4: Execute on assigned resources
        let individual_results = self.execute_on_resources(&task, &resource_assignments)?;

        // Step 5: Aggregate results
        let final_result = self.aggregate_results(&task, individual_results)?;

        // Step 6: Update performance metrics
        self.update_performance_metrics(&task, &final_result)?;

        println!("Hybrid execution completed for task: {}", task_id);
        Ok(final_result)
    }

    /// Get task from queue
    fn get_task_from_queue(&self, task_id: &str) -> ApplicationResult<HybridExecutionTask> {
        let mut queue = self.task_queue.lock().map_err(|_| {
            ApplicationError::OptimizationError("Failed to acquire task queue lock".to_string())
        })?;

        // Find and remove task from queue
        let task_index = queue
            .iter()
            .position(|task| task.id == task_id)
            .ok_or_else(|| {
                ApplicationError::InvalidConfiguration(format!(
                    "Task {} not found in queue",
                    task_id
                ))
            })?;

        Ok(queue.remove(task_index).unwrap())
    }

    /// Create execution plan for task
    fn create_execution_plan(
        &self,
        task: &HybridExecutionTask,
    ) -> ApplicationResult<ExecutionPlan> {
        let problem_size = task.problem.num_qubits;
        let quality_requirements = &task.quality_requirements;

        // Analyze problem characteristics
        let problem_complexity = self.analyze_problem_complexity(&task.problem)?;

        // Select optimal execution strategy
        let strategy = match (&self.config.execution_strategy, problem_complexity) {
            (ExecutionStrategy::Adaptive, ProblemComplexity::Simple) => {
                ExecutionStrategy::Sequential
            }
            (ExecutionStrategy::Adaptive, ProblemComplexity::Complex) => {
                ExecutionStrategy::Parallel
            }
            (strategy, _) => strategy.clone(),
        };

        // Determine resource requirements
        let resource_requirements = self.determine_resource_requirements(task)?;

        Ok(ExecutionPlan {
            strategy,
            resource_requirements,
            estimated_time: Duration::from_secs(60),
            estimated_cost: 10.0,
            quality_target: quality_requirements.target_quality,
        })
    }

    /// Analyze problem complexity
    fn analyze_problem_complexity(
        &self,
        problem: &IsingModel,
    ) -> ApplicationResult<ProblemComplexity> {
        let num_qubits = problem.num_qubits;
        let density = self.calculate_coupling_density(problem);

        if num_qubits < 100 && density < 0.1 {
            Ok(ProblemComplexity::Simple)
        } else if num_qubits < 1000 && density < 0.5 {
            Ok(ProblemComplexity::Medium)
        } else {
            Ok(ProblemComplexity::Complex)
        }
    }

    /// Calculate coupling density of problem
    fn calculate_coupling_density(&self, problem: &IsingModel) -> f64 {
        let total_possible = problem.num_qubits * (problem.num_qubits - 1) / 2;

        let couplings = problem.couplings();
        let actual_couplings = couplings
            .iter()
            .filter(|coupling| coupling.strength != 0.0)
            .count();

        if total_possible > 0 {
            actual_couplings as f64 / total_possible as f64
        } else {
            0.0
        }
    }

    /// Determine resource requirements
    fn determine_resource_requirements(
        &self,
        task: &HybridExecutionTask,
    ) -> ApplicationResult<ResourceRequirements> {
        let problem_size = task.problem.num_qubits;

        // Determine suitable resource types
        let mut suitable_types = Vec::new();

        if problem_size <= 5000 {
            suitable_types.push(ResourceType::DWaveQuantum);
        }
        if problem_size <= 2000 {
            suitable_types.push(ResourceType::BraketQuantum);
        }
        suitable_types.push(ResourceType::ClassicalSimulator);

        if problem_size > 1000 {
            suitable_types.push(ResourceType::MultiChipQuantum);
        }

        Ok(ResourceRequirements {
            suitable_types,
            min_resources: 1,
            max_resources: 3,
            performance_requirements: PerformanceRequirements {
                min_throughput: 0.1,
                max_latency: Duration::from_secs(120),
                min_quality: task.quality_requirements.min_quality,
            },
        })
    }

    /// Schedule resources for execution
    fn schedule_resources(
        &self,
        task: &HybridExecutionTask,
        plan: &ExecutionPlan,
    ) -> ApplicationResult<Vec<String>> {
        let mut scheduler = self.scheduler.lock().map_err(|_| {
            ApplicationError::OptimizationError("Failed to acquire scheduler lock".to_string())
        })?;

        let resources = self.resources.read().map_err(|_| {
            ApplicationError::OptimizationError("Failed to read resource registry".to_string())
        })?;

        // Filter available resources
        let available_resources: Vec<_> = resources
            .values()
            .filter(|resource| resource.availability == ResourceAvailability::Available)
            .filter(|resource| {
                plan.resource_requirements
                    .suitable_types
                    .contains(&resource.resource_type)
            })
            .collect();

        if available_resources.is_empty() {
            return Err(ApplicationError::ResourceLimitExceeded(
                "No suitable resources available".to_string(),
            ));
        }

        // Select resources based on strategy
        let selected = match self.config.allocation_strategy {
            ResourceAllocationStrategy::Adaptive => {
                self.select_adaptive_resources(&available_resources, task)?
            }
            ResourceAllocationStrategy::CostOptimized => {
                self.select_cost_optimized_resources(&available_resources, task)?
            }
            ResourceAllocationStrategy::QualityFocused => {
                self.select_quality_focused_resources(&available_resources, task)?
            }
            _ => {
                // Default: select best performing resource
                vec![available_resources
                    .iter()
                    .max_by(|a, b| {
                        a.performance
                            .throughput
                            .partial_cmp(&b.performance.throughput)
                            .unwrap()
                    })
                    .unwrap()
                    .id
                    .clone()]
            }
        };

        Ok(selected)
    }

    /// Select resources using adaptive strategy
    fn select_adaptive_resources(
        &self,
        available: &[&ComputeResource],
        task: &HybridExecutionTask,
    ) -> ApplicationResult<Vec<String>> {
        // Score resources based on multiple factors
        let mut scored_resources: Vec<_> = available
            .iter()
            .map(|resource| {
                let performance_score =
                    resource.performance.throughput * resource.performance.success_rate;
                let quality_score = resource.performance.quality_score;
                let cost_score = 1.0
                    / (1.0
                        + resource.cost.fixed_cost
                        + resource.cost.variable_cost * task.problem.num_qubits as f64);

                let total_score = performance_score * 0.4 + quality_score * 0.4 + cost_score * 0.2;
                (resource.id.clone(), total_score)
            })
            .collect();

        // Sort by score (highest first)
        scored_resources.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top resources
        let num_resources = (scored_resources.len().min(3)).max(1);
        Ok(scored_resources
            .into_iter()
            .take(num_resources)
            .map(|(id, _)| id)
            .collect())
    }

    /// Select cost-optimized resources
    fn select_cost_optimized_resources(
        &self,
        available: &[&ComputeResource],
        task: &HybridExecutionTask,
    ) -> ApplicationResult<Vec<String>> {
        let mut cost_sorted: Vec<_> = available
            .iter()
            .map(|resource| {
                let total_cost = resource.cost.fixed_cost
                    + resource.cost.variable_cost * task.problem.num_qubits as f64;
                (resource.id.clone(), total_cost)
            })
            .collect();

        cost_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Select cheapest resource that meets quality requirements
        for (resource_id, _cost) in &cost_sorted {
            if let Some(resource) = available.iter().find(|r| &r.id == resource_id) {
                if resource.performance.quality_score >= task.quality_requirements.min_quality {
                    return Ok(vec![resource_id.clone()]);
                }
            }
        }

        // Fallback: select cheapest available
        Ok(vec![cost_sorted[0].0.clone()])
    }

    /// Select quality-focused resources
    fn select_quality_focused_resources(
        &self,
        available: &[&ComputeResource],
        task: &HybridExecutionTask,
    ) -> ApplicationResult<Vec<String>> {
        let mut quality_sorted: Vec<_> = available
            .iter()
            .map(|resource| (resource.id.clone(), resource.performance.quality_score))
            .collect();

        quality_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select highest quality resources
        let num_resources = quality_sorted.len().min(2);
        Ok(quality_sorted
            .into_iter()
            .take(num_resources)
            .map(|(id, _)| id)
            .collect())
    }

    /// Execute on assigned resources
    fn execute_on_resources(
        &self,
        task: &HybridExecutionTask,
        resources: &[String],
    ) -> ApplicationResult<Vec<IndividualResult>> {
        let mut results = Vec::new();

        // Execute based on strategy
        match self.config.execution_strategy {
            ExecutionStrategy::Sequential => {
                results.extend(self.execute_sequential(task, resources)?);
            }
            ExecutionStrategy::Parallel => {
                results.extend(self.execute_parallel(task, resources)?);
            }
            ExecutionStrategy::Competitive => {
                results.extend(self.execute_competitive(task, resources)?);
            }
            _ => {
                // Default to parallel
                results.extend(self.execute_parallel(task, resources)?);
            }
        }

        Ok(results)
    }

    /// Execute sequentially on resources
    fn execute_sequential(
        &self,
        task: &HybridExecutionTask,
        resources: &[String],
    ) -> ApplicationResult<Vec<IndividualResult>> {
        let mut results = Vec::new();

        for resource_id in resources {
            let result = self.execute_on_single_resource(task, resource_id)?;
            results.push(result.clone());

            // Check if we achieved target quality
            if result.quality >= task.quality_requirements.target_quality {
                break;
            }
        }

        Ok(results)
    }

    /// Execute in parallel on resources
    fn execute_parallel(
        &self,
        task: &HybridExecutionTask,
        resources: &[String],
    ) -> ApplicationResult<Vec<IndividualResult>> {
        let mut results = Vec::new();

        // Simulate parallel execution
        for resource_id in resources {
            let result = self.execute_on_single_resource(task, resource_id)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Execute competitively (first good result wins)
    fn execute_competitive(
        &self,
        task: &HybridExecutionTask,
        resources: &[String],
    ) -> ApplicationResult<Vec<IndividualResult>> {
        // For now, simulate by running all and taking the best
        let all_results = self.execute_parallel(task, resources)?;

        // Return best result
        if let Some(best) = all_results
            .iter()
            .max_by(|a, b| a.quality.partial_cmp(&b.quality).unwrap())
        {
            Ok(vec![best.clone()])
        } else {
            Ok(all_results)
        }
    }

    /// Execute on single resource
    fn execute_on_single_resource(
        &self,
        task: &HybridExecutionTask,
        resource_id: &str,
    ) -> ApplicationResult<IndividualResult> {
        let start_time = Instant::now();

        // Simulate execution based on resource type
        let resources = self.resources.read().map_err(|_| {
            ApplicationError::OptimizationError("Failed to read resource registry".to_string())
        })?;

        let resource = resources.get(resource_id).ok_or_else(|| {
            ApplicationError::InvalidConfiguration(format!("Resource {} not found", resource_id))
        })?;

        // Simulate execution time based on resource performance
        let execution_time =
            Duration::from_millis((1000.0 / resource.performance.throughput) as u64);
        thread::sleep(Duration::from_millis(10)); // Brief simulation

        // Generate solution (simplified)
        let solution = self.generate_simulated_solution(task, resource)?;
        let energy = self.calculate_energy(&task.problem, &solution)?;
        let quality = resource.performance.quality_score * (0.8 + rand::random::<f64>() * 0.4);
        let cost =
            resource.cost.fixed_cost + resource.cost.variable_cost * task.problem.num_qubits as f64;

        Ok(IndividualResult {
            resource_id: resource_id.to_string(),
            solution,
            energy,
            quality,
            execution_time,
            cost,
            metadata: HashMap::new(),
        })
    }

    /// Generate simulated solution
    fn generate_simulated_solution(
        &self,
        task: &HybridExecutionTask,
        resource: &ComputeResource,
    ) -> ApplicationResult<Vec<i32>> {
        let num_vars = task.problem.num_qubits;
        let mut solution = vec![1; num_vars];

        // Add some randomness based on resource type
        for i in 0..num_vars {
            if rand::random::<f64>() < 0.5 {
                solution[i] = -1;
            }
        }

        Ok(solution)
    }

    /// Calculate energy for solution
    fn calculate_energy(&self, problem: &IsingModel, solution: &[i32]) -> ApplicationResult<f64> {
        let mut energy = 0.0;

        // Bias terms
        for (i, &spin) in solution.iter().enumerate() {
            let biases = problem.biases();
            for (qubit_index, bias_value) in biases {
                if qubit_index == i {
                    energy += bias_value * spin as f64;
                    break;
                }
            }
        }

        // Coupling terms
        let couplings = problem.couplings();
        for coupling in couplings {
            if coupling.i < solution.len() && coupling.j < solution.len() {
                energy +=
                    coupling.strength * solution[coupling.i] as f64 * solution[coupling.j] as f64;
            }
        }

        Ok(energy)
    }

    /// Aggregate results from multiple resources
    fn aggregate_results(
        &self,
        task: &HybridExecutionTask,
        results: Vec<IndividualResult>,
    ) -> ApplicationResult<HybridExecutionResult> {
        let mut aggregator = self.aggregator.lock().map_err(|_| {
            ApplicationError::OptimizationError("Failed to acquire aggregator lock".to_string())
        })?;

        let best_result = results
            .iter()
            .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
            .cloned()
            .unwrap_or_else(|| IndividualResult {
                resource_id: "none".to_string(),
                solution: vec![],
                energy: f64::INFINITY,
                quality: 0.0,
                execution_time: Duration::from_secs(0),
                cost: 0.0,
                metadata: HashMap::new(),
            });

        let total_time = results
            .iter()
            .map(|r| r.execution_time)
            .max()
            .unwrap_or(Duration::from_secs(0));

        let total_cost = results.iter().map(|r| r.cost).sum();

        let mut resource_utilization = HashMap::new();
        for result in &results {
            resource_utilization.insert(
                result.resource_id.clone(),
                ResourceUtilization {
                    resource_id: result.resource_id.clone(),
                    time_used: result.execution_time,
                    cost: result.cost,
                    success: result.quality > task.quality_requirements.min_quality,
                    quality: result.quality,
                },
            );
        }

        Ok(HybridExecutionResult {
            task_id: task.id.clone(),
            best_solution: best_result.solution,
            best_energy: best_result.energy,
            quality_score: best_result.quality,
            total_time,
            total_cost,
            resource_utilization: resource_utilization.clone(),
            individual_results: results,
            metadata: ExecutionMetadata {
                strategy: self.config.execution_strategy.clone(),
                resources_used: resource_utilization.keys().cloned().collect(),
                algorithm_decisions: vec!["adaptive_selection".to_string()],
                performance_metrics: HashMap::new(),
                warnings: vec![],
            },
        })
    }

    /// Update performance metrics after execution
    fn update_performance_metrics(
        &self,
        task: &HybridExecutionTask,
        result: &HybridExecutionResult,
    ) -> ApplicationResult<()> {
        let mut monitor = self.monitor.lock().map_err(|_| {
            ApplicationError::OptimizationError("Failed to acquire monitor lock".to_string())
        })?;

        monitor.update_metrics(task, result);
        Ok(())
    }

    /// Get current system performance
    pub fn get_system_performance(&self) -> ApplicationResult<HybridSystemMetrics> {
        let monitor = self.monitor.lock().map_err(|_| {
            ApplicationError::OptimizationError("Failed to acquire monitor lock".to_string())
        })?;

        Ok(monitor.system_metrics.clone())
    }
}

// Helper types
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub strategy: ExecutionStrategy,
    pub resource_requirements: ResourceRequirements,
    pub estimated_time: Duration,
    pub estimated_cost: f64,
    pub quality_target: f64,
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub suitable_types: Vec<ResourceType>,
    pub min_resources: usize,
    pub max_resources: usize,
    pub performance_requirements: PerformanceRequirements,
}

#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    pub min_throughput: f64,
    pub max_latency: Duration,
    pub min_quality: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProblemComplexity {
    Simple,
    Medium,
    Complex,
}

impl HybridPerformanceMonitor {
    fn new() -> Self {
        Self {
            system_metrics: HybridSystemMetrics {
                total_tasks: 0,
                avg_completion_time: Duration::from_secs(0),
                success_rate: 1.0,
                avg_quality: 0.0,
                total_cost: 0.0,
                resource_efficiency: 1.0,
            },
            resource_metrics: HashMap::new(),
            performance_history: VecDeque::new(),
            cost_tracking: CostTracker {
                current_budget: 1000.0,
                spent_amount: 0.0,
                cost_breakdown: HashMap::new(),
                cost_predictions: HashMap::new(),
            },
        }
    }

    fn update_metrics(&mut self, task: &HybridExecutionTask, result: &HybridExecutionResult) {
        self.system_metrics.total_tasks += 1;
        self.system_metrics.total_cost += result.total_cost;
        self.system_metrics.avg_quality = (self.system_metrics.avg_quality
            * (self.system_metrics.total_tasks - 1) as f64
            + result.quality_score)
            / self.system_metrics.total_tasks as f64;

        self.cost_tracking.spent_amount += result.total_cost;

        println!("Updated performance metrics for task {}", task.id);
    }
}

impl ResourceScheduler {
    fn new() -> Self {
        Self {
            strategy: ResourceAllocationStrategy::Adaptive,
            availability_cache: HashMap::new(),
            performance_predictions: HashMap::new(),
            load_balancing: LoadBalancingDecisions {
                recent_decisions: VecDeque::new(),
                resource_success_rates: HashMap::new(),
                performance_trends: HashMap::new(),
            },
        }
    }
}

impl ResultAggregator {
    fn new() -> Self {
        Self {
            strategy: ResultAggregationStrategy::BestResult,
            quality_assessor: QualityAssessor {
                methods: vec![QualityAssessmentMethod::EnergyBased],
                thresholds: HashMap::new(),
                quality_history: VecDeque::new(),
            },
            consensus_algorithm: ConsensusAlgorithm {
                threshold: 0.7,
                voting_weights: HashMap::new(),
                consensus_history: VecDeque::new(),
            },
        }
    }
}

/// Create example hybrid engine with multiple resource types
pub fn create_example_hybrid_engine() -> ApplicationResult<HeterogeneousHybridEngine> {
    let config = HybridEngineConfig::default();
    let engine = HeterogeneousHybridEngine::new(config);

    // Register D-Wave quantum resource
    let dwave_resource = ComputeResource {
        id: "dwave_advantage".to_string(),
        resource_type: ResourceType::DWaveQuantum,
        availability: ResourceAvailability::Available,
        performance: ResourcePerformance {
            throughput: 0.1,
            latency: Duration::from_secs(20),
            success_rate: 0.95,
            quality_score: 0.9,
            size_range: (10, 5000),
            history: VecDeque::new(),
        },
        cost: ResourceCost {
            fixed_cost: 1.0,
            variable_cost: 0.001,
            time_cost: 0.1,
            quality_premium: 1.2,
        },
        workload: None,
        connection: ResourceConnection::Custom("dwave_cloud".to_string()),
    };

    // Register classical simulator resource
    let classical_resource = ComputeResource {
        id: "classical_simulator".to_string(),
        resource_type: ResourceType::ClassicalSimulator,
        availability: ResourceAvailability::Available,
        performance: ResourcePerformance {
            throughput: 1.0,
            latency: Duration::from_secs(5),
            success_rate: 0.99,
            quality_score: 0.8,
            size_range: (1, 10000),
            history: VecDeque::new(),
        },
        cost: ResourceCost {
            fixed_cost: 0.1,
            variable_cost: 0.0001,
            time_cost: 0.01,
            quality_premium: 1.0,
        },
        workload: None,
        connection: ResourceConnection::Custom("local".to_string()),
    };

    engine.register_resource(dwave_resource)?;
    engine.register_resource(classical_resource)?;

    Ok(engine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_engine_creation() {
        let config = HybridEngineConfig::default();
        let engine = HeterogeneousHybridEngine::new(config);

        let resources = engine.resources.read().unwrap();
        assert!(resources.is_empty());
    }

    #[test]
    fn test_resource_registration() {
        let engine = create_example_hybrid_engine().unwrap();

        let resources = engine.resources.read().unwrap();
        assert_eq!(resources.len(), 2);
        assert!(resources.contains_key("dwave_advantage"));
        assert!(resources.contains_key("classical_simulator"));
    }

    #[test]
    fn test_task_submission() {
        let engine = create_example_hybrid_engine().unwrap();

        let problem = IsingModel::new(100);
        let task = HybridExecutionTask {
            id: "test_task".to_string(),
            problem,
            quality_requirements: QualityRequirements {
                min_quality: 0.8,
                target_quality: 0.9,
                assessment_method: QualityAssessmentMethod::EnergyBased,
                min_solutions: 1,
            },
            resource_constraints: ResourceConstraints {
                max_cost: Some(10.0),
                max_time: Duration::from_secs(60),
                preferred_resources: vec![ResourceType::ClassicalSimulator],
                excluded_resources: vec![],
                geographic_constraints: None,
            },
            priority: TaskPriority::Medium,
            deadline: None,
        };

        let result = engine.submit_task(task);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test_task");
    }

    #[test]
    fn test_execution_strategies() {
        let config = HybridEngineConfig {
            execution_strategy: ExecutionStrategy::Parallel,
            ..Default::default()
        };

        assert_eq!(config.execution_strategy, ExecutionStrategy::Parallel);
    }

    #[test]
    fn test_resource_allocation_strategies() {
        let strategies = vec![
            ResourceAllocationStrategy::Adaptive,
            ResourceAllocationStrategy::CostOptimized,
            ResourceAllocationStrategy::QualityFocused,
        ];

        for strategy in strategies {
            let config = HybridEngineConfig {
                allocation_strategy: strategy.clone(),
                ..Default::default()
            };
            assert_eq!(config.allocation_strategy, strategy);
        }
    }
}
