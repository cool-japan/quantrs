//! Quantum-Classical Co-optimization Framework
//!
//! This module provides tools for optimizing hybrid quantum-classical algorithms
//! where quantum circuits and classical processing are interleaved and optimized together.

use crate::builder::Circuit;
use num_complex::Complex64;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use std::collections::HashMap;

/// A hybrid quantum-classical optimization problem
///
/// This combines quantum circuits with classical processing steps,
/// allowing for co-optimization of both quantum parameters and classical algorithms.
#[derive(Debug, Clone)]
pub struct HybridOptimizationProblem<const N: usize> {
    /// Quantum circuit components
    pub quantum_circuits: Vec<ParameterizedQuantumComponent<N>>,
    /// Classical processing steps
    pub classical_steps: Vec<ClassicalProcessingStep>,
    /// Data flow between quantum and classical components
    pub data_flow: DataFlowGraph,
    /// Global optimization parameters
    pub global_parameters: Vec<f64>,
    /// Objective function for optimization
    pub objective: ObjectiveFunction,
}

/// A parameterized quantum circuit component
#[derive(Debug, Clone)]
pub struct ParameterizedQuantumComponent<const N: usize> {
    /// The quantum circuit
    pub circuit: Circuit<N>,
    /// Parameter indices in the global parameter vector
    pub parameter_indices: Vec<usize>,
    /// Input data from classical components
    pub classical_inputs: Vec<String>,
    /// Output measurements to classical components
    pub quantum_outputs: Vec<String>,
    /// Component identifier
    pub id: String,
}

/// A classical processing step in the hybrid algorithm
#[derive(Debug, Clone)]
pub struct ClassicalProcessingStep {
    /// Step identifier
    pub id: String,
    /// Type of classical processing
    pub step_type: ClassicalStepType,
    /// Input data sources
    pub inputs: Vec<String>,
    /// Output data destinations
    pub outputs: Vec<String>,
    /// Parameters for this processing step
    pub parameters: HashMap<String, f64>,
}

/// Types of classical processing steps
#[derive(Debug, Clone, PartialEq)]
pub enum ClassicalStepType {
    /// Linear algebra operations
    LinearAlgebra(LinearAlgebraOp),
    /// Machine learning model inference
    MachineLearning(MLModelType),
    /// Optimization subroutine
    Optimization(OptimizationMethod),
    /// Data preprocessing
    DataProcessing(DataProcessingOp),
    /// Control flow decision
    ControlFlow(ControlFlowType),
    /// Parameter update rule
    ParameterUpdate(UpdateRule),
    /// Custom processing function
    Custom(String),
}

/// Linear algebra operations
#[derive(Debug, Clone, PartialEq)]
pub enum LinearAlgebraOp {
    MatrixMultiplication,
    Eigendecomposition,
    SVD,
    LeastSquares,
    LinearSolve,
    TensorContraction,
}

/// Machine learning model types
#[derive(Debug, Clone, PartialEq)]
pub enum MLModelType {
    NeuralNetwork,
    SupportVectorMachine,
    RandomForest,
    GaussianProcess,
    LinearRegression,
    LogisticRegression,
}

/// Optimization methods for classical subroutines
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationMethod {
    GradientDescent,
    BFGS,
    NelderMead,
    SimulatedAnnealing,
    GeneticAlgorithm,
    BayesianOptimization,
}

/// Data preprocessing operations
#[derive(Debug, Clone, PartialEq)]
pub enum DataProcessingOp {
    Normalization,
    Standardization,
    PCA,
    FeatureSelection,
    DataAugmentation,
    OutlierRemoval,
}

/// Control flow types
#[derive(Debug, Clone, PartialEq)]
pub enum ControlFlowType {
    Conditional,
    Loop,
    Parallel,
    Adaptive,
}

/// Parameter update rules
#[derive(Debug, Clone, PartialEq)]
pub enum UpdateRule {
    GradientBased,
    MomentumBased,
    AdamOptimizer,
    AdaGrad,
    RMSProp,
    Custom(String),
}

/// Data flow graph representing connections between components
#[derive(Debug, Clone)]
pub struct DataFlowGraph {
    /// Nodes in the graph (component IDs)
    pub nodes: Vec<String>,
    /// Edges representing data flow (source, target, data_type)
    pub edges: Vec<(String, String, DataType)>,
    /// Execution order constraints
    pub execution_order: Vec<Vec<String>>,
}

/// Types of data flowing between components
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    /// Quantum measurement results
    Measurements(Vec<f64>),
    /// Probability distributions
    Probabilities(Vec<f64>),
    /// Classical vectors/matrices
    Matrix(Vec<Vec<f64>>),
    /// Scalar values
    Scalar(f64),
    /// Parameter vectors
    Parameters(Vec<f64>),
    /// Boolean control signals
    Control(bool),
    /// Custom data format
    Custom(String),
}

/// Objective function for hybrid optimization
#[derive(Debug, Clone)]
pub struct ObjectiveFunction {
    /// Function type
    pub function_type: ObjectiveFunctionType,
    /// Target value (for minimization/maximization)
    pub target: Option<f64>,
    /// Weights for multi-objective optimization
    pub weights: Vec<f64>,
    /// Regularization terms
    pub regularization: Vec<RegularizationTerm>,
}

/// Types of objective functions
#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveFunctionType {
    /// Minimize expectation value
    ExpectationValue,
    /// Maximize fidelity
    Fidelity,
    /// Minimize cost function
    CostFunction,
    /// Multi-objective optimization
    MultiObjective(Vec<ObjectiveFunctionType>),
    /// Custom objective
    Custom(String),
}

/// Regularization terms for the objective function
#[derive(Debug, Clone)]
pub struct RegularizationTerm {
    /// Type of regularization
    pub reg_type: RegularizationType,
    /// Regularization strength
    pub strength: f64,
    /// Parameters to regularize
    pub parameter_indices: Vec<usize>,
}

/// Types of regularization
#[derive(Debug, Clone, PartialEq)]
pub enum RegularizationType {
    L1,
    L2,
    ElasticNet,
    TotalVariation,
    Sparsity,
    Smoothness,
}

/// Hybrid optimization result
#[derive(Debug, Clone)]
pub struct HybridOptimizationResult {
    /// Optimal parameters
    pub optimal_parameters: Vec<f64>,
    /// Optimal objective value
    pub optimal_value: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Convergence status
    pub converged: bool,
    /// Execution history
    pub history: OptimizationHistory,
    /// Final quantum state information
    pub quantum_info: QuantumStateInfo,
}

/// Optimization history tracking
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    /// Objective values over iterations
    pub objective_values: Vec<f64>,
    /// Parameter values over iterations  
    pub parameter_history: Vec<Vec<f64>>,
    /// Gradient norms
    pub gradient_norms: Vec<f64>,
    /// Step sizes used
    pub step_sizes: Vec<f64>,
    /// Timing information
    pub execution_times: Vec<f64>,
}

/// Information about final quantum states
#[derive(Debug, Clone)]
pub struct QuantumStateInfo {
    /// Final quantum states for each circuit
    pub final_states: HashMap<String, Vec<Complex64>>,
    /// Measurement statistics
    pub measurement_stats: HashMap<String, MeasurementStatistics>,
    /// Entanglement measures
    pub entanglement_info: HashMap<String, EntanglementInfo>,
}

/// Statistics from quantum measurements
#[derive(Debug, Clone)]
pub struct MeasurementStatistics {
    /// Mean values
    pub means: Vec<f64>,
    /// Standard deviations
    pub std_devs: Vec<f64>,
    /// Correlations between measurements
    pub correlations: Vec<Vec<f64>>,
    /// Number of shots used
    pub num_shots: usize,
}

/// Entanglement information
#[derive(Debug, Clone)]
pub struct EntanglementInfo {
    /// Von Neumann entropy
    pub von_neumann_entropy: f64,
    /// Mutual information matrix
    pub mutual_information: Vec<Vec<f64>>,
    /// Entanglement spectrum
    pub entanglement_spectrum: Vec<f64>,
}

/// Hybrid optimizer for quantum-classical co-optimization
pub struct HybridOptimizer {
    /// Optimization algorithm
    pub algorithm: HybridOptimizationAlgorithm,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate schedule
    pub learning_rate_schedule: LearningRateSchedule,
    /// Parallelization settings
    pub parallelization: ParallelizationConfig,
}

/// Hybrid optimization algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum HybridOptimizationAlgorithm {
    /// Coordinate descent (alternate quantum and classical optimization)
    CoordinateDescent,
    /// Simultaneous optimization of all parameters
    SimultaneousOptimization,
    /// Hierarchical optimization (coarse-to-fine)
    HierarchicalOptimization,
    /// Adaptive algorithm selection
    AdaptiveOptimization,
    /// Custom algorithm
    Custom(String),
}

/// Learning rate schedules
#[derive(Debug, Clone)]
pub struct LearningRateSchedule {
    /// Initial learning rate
    pub initial_rate: f64,
    /// Schedule type
    pub schedule_type: ScheduleType,
    /// Schedule parameters
    pub parameters: HashMap<String, f64>,
}

/// Types of learning rate schedules
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleType {
    Constant,
    LinearDecay,
    ExponentialDecay,
    StepDecay,
    CosineAnnealing,
    Adaptive,
}

/// Parallelization configuration
#[derive(Debug, Clone)]
pub struct ParallelizationConfig {
    /// Number of parallel quantum circuit evaluations
    pub quantum_parallelism: usize,
    /// Number of parallel classical processing threads
    pub classical_parallelism: usize,
    /// Enable asynchronous execution
    pub asynchronous: bool,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WorkStealing,
    Dynamic,
    Static,
}

impl<const N: usize> HybridOptimizationProblem<N> {
    /// Create a new hybrid optimization problem
    pub fn new() -> Self {
        Self {
            quantum_circuits: Vec::new(),
            classical_steps: Vec::new(),
            data_flow: DataFlowGraph {
                nodes: Vec::new(),
                edges: Vec::new(),
                execution_order: Vec::new(),
            },
            global_parameters: Vec::new(),
            objective: ObjectiveFunction {
                function_type: ObjectiveFunctionType::ExpectationValue,
                target: None,
                weights: vec![1.0],
                regularization: Vec::new(),
            },
        }
    }

    /// Add a quantum circuit component
    pub fn add_quantum_component(
        &mut self,
        id: String,
        circuit: Circuit<N>,
        parameter_indices: Vec<usize>,
    ) -> QuantRS2Result<()> {
        // Validate parameter indices
        for &idx in &parameter_indices {
            if idx >= self.global_parameters.len() {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Parameter index {} out of range (total parameters: {})",
                    idx,
                    self.global_parameters.len()
                )));
            }
        }

        let component = ParameterizedQuantumComponent {
            circuit,
            parameter_indices,
            classical_inputs: Vec::new(),
            quantum_outputs: Vec::new(),
            id: id.clone(),
        };

        self.quantum_circuits.push(component);
        self.data_flow.nodes.push(id);
        Ok(())
    }

    /// Add a classical processing step
    pub fn add_classical_step(
        &mut self,
        id: String,
        step_type: ClassicalStepType,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> QuantRS2Result<()> {
        let step = ClassicalProcessingStep {
            id: id.clone(),
            step_type,
            inputs,
            outputs,
            parameters: HashMap::new(),
        };

        self.classical_steps.push(step);
        self.data_flow.nodes.push(id);
        Ok(())
    }

    /// Add data flow edge between components
    pub fn add_data_flow(
        &mut self,
        source: String,
        target: String,
        data_type: DataType,
    ) -> QuantRS2Result<()> {
        // Validate that source and target exist
        if !self.data_flow.nodes.contains(&source) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Source component '{}' not found",
                source
            )));
        }
        if !self.data_flow.nodes.contains(&target) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Target component '{}' not found",
                target
            )));
        }

        self.data_flow.edges.push((source, target, data_type));
        Ok(())
    }

    /// Set global parameters
    pub fn set_global_parameters(&mut self, parameters: Vec<f64>) {
        self.global_parameters = parameters;
    }

    /// Add regularization term
    pub fn add_regularization(
        &mut self,
        reg_type: RegularizationType,
        strength: f64,
        parameter_indices: Vec<usize>,
    ) -> QuantRS2Result<()> {
        // Validate parameter indices
        for &idx in &parameter_indices {
            if idx >= self.global_parameters.len() {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Parameter index {} out of range",
                    idx
                )));
            }
        }

        self.objective.regularization.push(RegularizationTerm {
            reg_type,
            strength,
            parameter_indices,
        });

        Ok(())
    }

    /// Validate the optimization problem
    pub fn validate(&self) -> QuantRS2Result<()> {
        // Check that all components are connected properly
        for edge in &self.data_flow.edges {
            let (source, target, _) = edge;
            if !self.data_flow.nodes.contains(source) {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Data flow edge references non-existent source '{}'",
                    source
                )));
            }
            if !self.data_flow.nodes.contains(target) {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Data flow edge references non-existent target '{}'",
                    target
                )));
            }
        }

        // Check for circular dependencies
        if self.has_circular_dependencies()? {
            return Err(QuantRS2Error::InvalidInput(
                "Circular dependencies detected in data flow graph".to_string(),
            ));
        }

        Ok(())
    }

    /// Check for circular dependencies in the data flow graph
    fn has_circular_dependencies(&self) -> QuantRS2Result<bool> {
        // Simplified cycle detection - a full implementation would use DFS
        // For now, just check if any node has a self-loop
        for (source, target, _) in &self.data_flow.edges {
            if source == target {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl Default for HybridOptimizationProblem<4> {
    fn default() -> Self {
        Self::new()
    }
}

impl HybridOptimizer {
    /// Create a new hybrid optimizer
    pub fn new(algorithm: HybridOptimizationAlgorithm) -> Self {
        Self {
            algorithm,
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate_schedule: LearningRateSchedule {
                initial_rate: 0.01,
                schedule_type: ScheduleType::Constant,
                parameters: HashMap::new(),
            },
            parallelization: ParallelizationConfig {
                quantum_parallelism: 1,
                classical_parallelism: 1,
                asynchronous: false,
                load_balancing: LoadBalancingStrategy::RoundRobin,
            },
        }
    }

    /// Optimize a hybrid quantum-classical problem
    pub fn optimize<const N: usize>(
        &self,
        problem: &mut HybridOptimizationProblem<N>,
    ) -> QuantRS2Result<HybridOptimizationResult> {
        // Validate the problem first
        problem.validate()?;

        // Initialize optimization history
        let mut history = OptimizationHistory {
            objective_values: Vec::new(),
            parameter_history: Vec::new(),
            gradient_norms: Vec::new(),
            step_sizes: Vec::new(),
            execution_times: Vec::new(),
        };

        let mut current_parameters = problem.global_parameters.clone();
        let mut best_parameters = current_parameters.clone();
        let mut best_value = f64::INFINITY;

        // Main optimization loop
        for iteration in 0..self.max_iterations {
            let start_time = std::time::Instant::now();

            // Evaluate objective function
            let current_value = self.evaluate_objective(problem, &current_parameters)?;

            if current_value < best_value {
                best_value = current_value;
                best_parameters = current_parameters.clone();
            }

            // Store history
            history.objective_values.push(current_value);
            history.parameter_history.push(current_parameters.clone());

            // Compute gradients (simplified)
            let gradients = self.compute_gradients(problem, &current_parameters)?;
            let gradient_norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
            history.gradient_norms.push(gradient_norm);

            // Check convergence
            if gradient_norm < self.tolerance {
                let execution_time = start_time.elapsed().as_secs_f64();
                history.execution_times.push(execution_time);

                return Ok(HybridOptimizationResult {
                    optimal_parameters: best_parameters,
                    optimal_value: best_value,
                    iterations: iteration + 1,
                    converged: true,
                    history,
                    quantum_info: self.extract_quantum_info(problem)?,
                });
            }

            // Update parameters
            let learning_rate = self.get_learning_rate(iteration);
            for (i, gradient) in gradients.iter().enumerate() {
                current_parameters[i] -= learning_rate * gradient;
            }

            let step_size = learning_rate * gradient_norm;
            history.step_sizes.push(step_size);

            let execution_time = start_time.elapsed().as_secs_f64();
            history.execution_times.push(execution_time);
        }

        // Maximum iterations reached
        Ok(HybridOptimizationResult {
            optimal_parameters: best_parameters,
            optimal_value: best_value,
            iterations: self.max_iterations,
            converged: false,
            history,
            quantum_info: self.extract_quantum_info(problem)?,
        })
    }

    /// Evaluate the objective function (simplified)
    fn evaluate_objective<const N: usize>(
        &self,
        _problem: &HybridOptimizationProblem<N>,
        _parameters: &[f64],
    ) -> QuantRS2Result<f64> {
        // This is a placeholder - real implementation would:
        // 1. Execute quantum circuits with current parameters
        // 2. Run classical processing steps
        // 3. Combine results according to objective function
        // 4. Apply regularization terms

        // For now, return a dummy value
        Ok(1.0)
    }

    /// Compute gradients (simplified)
    fn compute_gradients<const N: usize>(
        &self,
        problem: &HybridOptimizationProblem<N>,
        _parameters: &[f64],
    ) -> QuantRS2Result<Vec<f64>> {
        // This is a placeholder - real implementation would use:
        // 1. Parameter shift rule for quantum gradients
        // 2. Automatic differentiation for classical components
        // 3. Chain rule for hybrid components

        // For now, return dummy gradients
        Ok(vec![0.001; problem.global_parameters.len()])
    }

    /// Get learning rate for current iteration
    fn get_learning_rate(&self, iteration: usize) -> f64 {
        match self.learning_rate_schedule.schedule_type {
            ScheduleType::Constant => self.learning_rate_schedule.initial_rate,
            ScheduleType::LinearDecay => {
                let decay_rate = self
                    .learning_rate_schedule
                    .parameters
                    .get("decay_rate")
                    .unwrap_or(&0.001);
                self.learning_rate_schedule.initial_rate / (1.0 + decay_rate * iteration as f64)
            }
            ScheduleType::ExponentialDecay => {
                let decay_rate = self
                    .learning_rate_schedule
                    .parameters
                    .get("decay_rate")
                    .unwrap_or(&0.95);
                self.learning_rate_schedule.initial_rate * decay_rate.powi(iteration as i32)
            }
            _ => self.learning_rate_schedule.initial_rate, // Simplified
        }
    }

    /// Extract quantum state information
    fn extract_quantum_info<const N: usize>(
        &self,
        _problem: &HybridOptimizationProblem<N>,
    ) -> QuantRS2Result<QuantumStateInfo> {
        // This is a placeholder - real implementation would extract:
        // 1. Final quantum states from each circuit
        // 2. Measurement statistics
        // 3. Entanglement measures

        Ok(QuantumStateInfo {
            final_states: HashMap::new(),
            measurement_stats: HashMap::new(),
            entanglement_info: HashMap::new(),
        })
    }
}

impl Default for HybridOptimizer {
    fn default() -> Self {
        Self::new(HybridOptimizationAlgorithm::CoordinateDescent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_problem_creation() {
        let problem = HybridOptimizationProblem::<4>::new();
        assert_eq!(problem.quantum_circuits.len(), 0);
        assert_eq!(problem.classical_steps.len(), 0);
    }

    #[test]
    fn test_component_addition() {
        let mut problem = HybridOptimizationProblem::<2>::new();
        problem.set_global_parameters(vec![0.1, 0.2, 0.3]);

        let circuit = Circuit::<2>::new();
        problem
            .add_quantum_component("q1".to_string(), circuit, vec![0, 1])
            .unwrap();

        assert_eq!(problem.quantum_circuits.len(), 1);
        assert_eq!(problem.data_flow.nodes.len(), 1);
    }

    #[test]
    fn test_data_flow() {
        let mut problem = HybridOptimizationProblem::<2>::new();
        problem.set_global_parameters(vec![0.1, 0.2]);

        let circuit = Circuit::<2>::new();
        problem
            .add_quantum_component("q1".to_string(), circuit, vec![0])
            .unwrap();
        problem
            .add_classical_step(
                "c1".to_string(),
                ClassicalStepType::LinearAlgebra(LinearAlgebraOp::MatrixMultiplication),
                vec!["q1".to_string()],
                vec!["output".to_string()],
            )
            .unwrap();

        problem
            .add_data_flow(
                "q1".to_string(),
                "c1".to_string(),
                DataType::Measurements(vec![0.1, 0.2]),
            )
            .unwrap();

        assert_eq!(problem.data_flow.edges.len(), 1);
    }

    #[test]
    fn test_optimizer_creation() {
        let optimizer = HybridOptimizer::new(HybridOptimizationAlgorithm::SimultaneousOptimization);
        assert_eq!(
            optimizer.algorithm,
            HybridOptimizationAlgorithm::SimultaneousOptimization
        );
        assert_eq!(optimizer.max_iterations, 1000);
    }
}
