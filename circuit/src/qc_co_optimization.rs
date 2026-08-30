//! Quantum-Classical Co-optimization Framework
//!
//! This module provides tools for optimizing hybrid quantum-classical algorithms
//! where quantum circuits and classical processing are interleaved and optimized together.

use crate::builder::Circuit;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::single::{RotationX, RotationY, RotationZ},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::parallel_ops::{IntoParallelRefIterator, ParallelIterator};
use scirs2_core::Complex64;
use std::collections::{HashMap, HashSet};

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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearAlgebraOp {
    MatrixMultiplication,
    Eigendecomposition,
    SVD,
    LeastSquares,
    LinearSolve,
    TensorContraction,
}

/// Machine learning model types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MLModelType {
    NeuralNetwork,
    SupportVectorMachine,
    RandomForest,
    GaussianProcess,
    LinearRegression,
    LogisticRegression,
}

/// Optimization methods for classical subroutines
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationMethod {
    GradientDescent,
    BFGS,
    NelderMead,
    SimulatedAnnealing,
    GeneticAlgorithm,
    BayesianOptimization,
}

/// Data preprocessing operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataProcessingOp {
    Normalization,
    Standardization,
    PCA,
    FeatureSelection,
    DataAugmentation,
    OutlierRemoval,
}

/// Control flow types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlFlowType {
    Conditional,
    Loop,
    Parallel,
    Adaptive,
}

/// Parameter update rules
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Edges representing data flow (source, target, `data_type`)
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
    MultiObjective(Vec<Self>),
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WorkStealing,
    Dynamic,
    Static,
}

impl<const N: usize> HybridOptimizationProblem<N> {
    /// Create a new hybrid optimization problem
    #[must_use]
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
                "Source component '{source}' not found"
            )));
        }
        if !self.data_flow.nodes.contains(&target) {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Target component '{target}' not found"
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
                    "Parameter index {idx} out of range"
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
                    "Data flow edge references non-existent source '{source}'"
                )));
            }
            if !self.data_flow.nodes.contains(target) {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Data flow edge references non-existent target '{target}'"
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

    /// Check for circular dependencies in the data flow graph.
    ///
    /// Performs a standard three-colour (white/gray/black) depth-first search
    /// over `data_flow.nodes`/`data_flow.edges`: a node is *gray* while it is
    /// on the current DFS recursion stack and *black* once fully explored.
    /// Encountering an edge into a gray node means the recursion stack itself
    /// forms a cycle (e.g. `A -> B -> C -> A`), not merely a direct self-loop.
    fn has_circular_dependencies(&self) -> QuantRS2Result<bool> {
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Colour {
            White,
            Gray,
            Black,
        }

        // Adjacency list keyed by node name, built once up front.
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
        for node in &self.data_flow.nodes {
            adjacency.entry(node.as_str()).or_default();
        }
        for (source, target, _) in &self.data_flow.edges {
            adjacency
                .entry(source.as_str())
                .or_default()
                .push(target.as_str());
        }

        let mut colour: HashMap<&str, Colour> = self
            .data_flow
            .nodes
            .iter()
            .map(|n| (n.as_str(), Colour::White))
            .collect();

        // Iterative DFS (explicit stack) to avoid unbounded recursion depth on
        // large graphs; each stack frame tracks its outgoing-edge cursor.
        for start in &self.data_flow.nodes {
            if colour.get(start.as_str()).copied() != Some(Colour::White) {
                continue;
            }

            let mut stack: Vec<(&str, usize)> = vec![(start.as_str(), 0)];
            colour.insert(start.as_str(), Colour::Gray);

            while let Some((node, cursor)) = stack.pop() {
                let neighbours = adjacency.get(node).map(Vec::as_slice).unwrap_or(&[]);
                if cursor < neighbours.len() {
                    let next = neighbours[cursor];
                    // Resume this frame at the following neighbour once we
                    // return to it.
                    stack.push((node, cursor + 1));
                    match colour.get(next).copied() {
                        Some(Colour::Gray) => return Ok(true), // back-edge => cycle
                        Some(Colour::White) => {
                            colour.insert(next, Colour::Gray);
                            stack.push((next, 0));
                        }
                        Some(Colour::Black) | None => {}
                    }
                } else {
                    colour.insert(node, Colour::Black);
                }
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
    #[must_use]
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

    /// Optimize a hybrid quantum-classical problem.
    ///
    /// The concrete update rule performed each iteration depends on
    /// [`Self::algorithm`] — see [`Self::active_parameter_mask`] for exactly
    /// how each [`HybridOptimizationAlgorithm`] variant differs from plain
    /// full-batch gradient descent. [`HybridOptimizationAlgorithm::Custom`]
    /// names an algorithm this optimizer does not implement and is rejected
    /// with an honest [`QuantRS2Error::UnsupportedOperation`] rather than
    /// silently running as if it were [`HybridOptimizationAlgorithm::SimultaneousOptimization`].
    pub fn optimize<const N: usize>(
        &self,
        problem: &mut HybridOptimizationProblem<N>,
    ) -> QuantRS2Result<HybridOptimizationResult> {
        // Validate the problem first
        problem.validate()?;

        if let HybridOptimizationAlgorithm::Custom(name) = &self.algorithm {
            return Err(QuantRS2Error::UnsupportedOperation(format!(
                "custom hybrid optimization algorithm '{name}' is not implemented; use \
                 CoordinateDescent, SimultaneousOptimization, HierarchicalOptimization, or \
                 AdaptiveOptimization, each of which runs a genuinely distinct update rule"
            )));
        }

        // Global-parameter indices that drive at least one quantum gate
        // (via some component's `parameter_indices`) versus the remainder,
        // which only ever appear in classical regularization terms. This
        // partition is what `CoordinateDescent` and `AdaptiveOptimization`
        // alternate between.
        let quantum_indices = quantum_parameter_indices(problem);

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
        let num_params = current_parameters.len();

        // Main optimization loop
        for iteration in 0..self.max_iterations {
            let start_time = std::time::Instant::now();

            // Evaluate objective function
            let current_value = self.evaluate_objective(problem, &current_parameters)?;

            if current_value < best_value {
                best_value = current_value;
                best_parameters.clone_from(&current_parameters);
            }

            // Store history
            history.objective_values.push(current_value);
            history.parameter_history.push(current_parameters.clone());

            // Compute gradients (parameter-shift for the quantum part plus the
            // analytic regularization derivative).
            let gradients = self.compute_gradients(problem, &current_parameters)?;
            let gradient_norm = gradients.iter().map(|g| g * g).sum::<f64>().sqrt();
            history.gradient_norms.push(gradient_norm);

            // Check convergence
            if gradient_norm < self.tolerance {
                let execution_time = start_time.elapsed().as_secs_f64();
                history.execution_times.push(execution_time);

                // Make the problem (and hence the extracted quantum state)
                // reflect the best parameters found before reporting.
                problem.global_parameters.clone_from(&best_parameters);
                let quantum_info = self.extract_quantum_info(problem)?;
                return Ok(HybridOptimizationResult {
                    optimal_parameters: best_parameters,
                    optimal_value: best_value,
                    iterations: iteration + 1,
                    converged: true,
                    history,
                    quantum_info,
                });
            }

            // Which parameters this iteration actually updates, per
            // `self.algorithm`.
            let active = self.active_parameter_mask(
                &quantum_indices,
                iteration,
                num_params,
                &history.objective_values,
            );

            // Update parameters (only the active block), and track the real
            // step actually taken -- not the full unmasked gradient -- so
            // `history.step_sizes` honestly reflects block algorithms too.
            let learning_rate = self.get_learning_rate(iteration, &history.gradient_norms);
            let mut applied_grad_norm_sq = 0.0;
            for (i, gradient) in gradients.iter().enumerate() {
                if active[i] {
                    current_parameters[i] -= learning_rate * gradient;
                    applied_grad_norm_sq += gradient * gradient;
                }
            }

            let step_size = learning_rate * applied_grad_norm_sq.sqrt();
            history.step_sizes.push(step_size);

            let execution_time = start_time.elapsed().as_secs_f64();
            history.execution_times.push(execution_time);
        }

        // Maximum iterations reached: report the best parameters seen.
        problem.global_parameters.clone_from(&best_parameters);
        let quantum_info = self.extract_quantum_info(problem)?;
        Ok(HybridOptimizationResult {
            optimal_parameters: best_parameters,
            optimal_value: best_value,
            iterations: self.max_iterations,
            converged: false,
            history,
            quantum_info,
        })
    }

    /// Evaluate the hybrid objective function for a concrete parameter vector.
    ///
    /// This is a *real* evaluation, not a placeholder.  For every quantum
    /// component the provided `parameters` are bound into the component's
    /// parameterized rotation gates (the `k`-th entry of the component's
    /// `parameter_indices` drives the `k`-th parameterized gate, in gate order —
    /// the standard ansatz convention) and the resulting state `|ψ(θ)⟩` is
    /// produced by an exact dense state-vector simulation
    /// ([`statevector::simulate`]).  A scalar cost is then derived from that
    /// state according to [`ObjectiveFunctionType`]:
    ///
    /// * [`ObjectiveFunctionType::ExpectationValue`] /
    ///   [`ObjectiveFunctionType::CostFunction`] / [`ObjectiveFunctionType::Custom`]
    ///   minimize `⟨ψ| (Σ_q Z_q) |ψ⟩` (the canonical diagonal cost Hamiltonian
    ///   whose ground state is `|0…0⟩`).
    /// * [`ObjectiveFunctionType::Fidelity`] maximizes the overlap with `|0…0⟩`,
    ///   expressed as the minimization objective `1 − |⟨0…0|ψ⟩|²`.
    /// * [`ObjectiveFunctionType::MultiObjective`] combines its sub-objectives.
    ///
    /// Per-component contributions are combined with `objective.weights`
    /// (defaulting to weight `1`), and every regularization term is added on top
    /// via [`Self::regularization_value`].  These regularization terms make the
    /// objective depend on parameters that do not drive any gate, exactly as a
    /// real hybrid cost would.
    fn evaluate_objective<const N: usize>(
        &self,
        problem: &HybridOptimizationProblem<N>,
        parameters: &[f64],
    ) -> QuantRS2Result<f64> {
        let eval_component = |component_index: usize| -> QuantRS2Result<f64> {
            let component = &problem.quantum_circuits[component_index];
            let bound = bind_parameters(component, parameters)?;
            let state = statevector::simulate(&bound)?;
            let contribution =
                Self::objective_from_state(&state, N, &problem.objective.function_type)?;
            let weight = problem
                .objective
                .weights
                .get(component_index)
                .copied()
                .unwrap_or(1.0);
            Ok(weight * contribution)
        };

        // Each component's state-vector simulation is fully independent, so
        // `parallelization.quantum_parallelism` (the configured number of
        // parallel quantum circuit evaluations) genuinely drives whether this
        // runs across the SciRS2 parallel executor or sequentially in-order.
        let component_indices: Vec<usize> = (0..problem.quantum_circuits.len()).collect();
        let component_values: Vec<QuantRS2Result<f64>> =
            if self.parallelization.quantum_parallelism > 1 && component_indices.len() > 1 {
                component_indices
                    .par_iter()
                    .map(|&idx| eval_component(idx))
                    .collect()
            } else {
                component_indices
                    .iter()
                    .map(|&idx| eval_component(idx))
                    .collect()
            };

        let mut value = 0.0;
        for contribution in component_values {
            value += contribution?;
        }

        // Classical regularization terms operate directly on the parameter
        // vector and are genuine (parameter-dependent) contributions; they are
        // the "classical processing" this optimizer performs, so
        // `parallelization.classical_parallelism` drives their evaluation.
        let regularization_values: Vec<QuantRS2Result<f64>> =
            if self.parallelization.classical_parallelism > 1
                && problem.objective.regularization.len() > 1
            {
                problem
                    .objective
                    .regularization
                    .par_iter()
                    .map(|term| Self::regularization_value(term, parameters))
                    .collect()
            } else {
                problem
                    .objective
                    .regularization
                    .iter()
                    .map(|term| Self::regularization_value(term, parameters))
                    .collect()
            };
        for contribution in regularization_values {
            value += contribution?;
        }

        Ok(value)
    }

    /// Derive a scalar cost from a simulated state according to `function_type`.
    fn objective_from_state(
        state: &[Complex64],
        num_qubits: usize,
        function_type: &ObjectiveFunctionType,
    ) -> QuantRS2Result<f64> {
        match function_type {
            ObjectiveFunctionType::ExpectationValue
            | ObjectiveFunctionType::CostFunction
            | ObjectiveFunctionType::Custom(_) => {
                // ⟨Σ_q Z_q⟩ for the diagonal cost Hamiltonian.
                Ok(statevector::sum_z_expectation(state, num_qubits))
            }
            ObjectiveFunctionType::Fidelity => {
                // Maximize fidelity with |0…0⟩ ⇒ minimize 1 − |⟨0…0|ψ⟩|².
                let amplitude = state.first().copied().unwrap_or(Complex64::new(0.0, 0.0));
                Ok(1.0 - amplitude.norm_sqr())
            }
            ObjectiveFunctionType::MultiObjective(sub_objectives) => {
                let mut total = 0.0;
                for sub in sub_objectives {
                    total += Self::objective_from_state(state, num_qubits, sub)?;
                }
                Ok(total)
            }
        }
    }

    /// Value of a single regularization term for the given parameter vector.
    fn regularization_value(term: &RegularizationTerm, parameters: &[f64]) -> QuantRS2Result<f64> {
        let selected = collect_parameters(term, parameters)?;
        let penalty = match term.reg_type {
            RegularizationType::L1 | RegularizationType::Sparsity => {
                selected.iter().map(|p| p.abs()).sum::<f64>()
            }
            RegularizationType::L2 => selected.iter().map(|p| p * p).sum::<f64>(),
            RegularizationType::ElasticNet => {
                let l1 = selected.iter().map(|p| p.abs()).sum::<f64>();
                let l2 = selected.iter().map(|p| p * p).sum::<f64>();
                0.5 * l1 + 0.5 * l2
            }
            RegularizationType::TotalVariation | RegularizationType::Smoothness => {
                // Sum of squared consecutive differences (∝ discrete gradient
                // energy), the standard smoothness/total-variation penalty.
                selected
                    .windows(2)
                    .map(|w| {
                        let d = w[1] - w[0];
                        d * d
                    })
                    .sum::<f64>()
            }
        };
        Ok(term.strength * penalty)
    }

    /// Analytic gradient of a single regularization term w.r.t. every global
    /// parameter (zero for parameters the term does not touch).
    fn regularization_gradient(
        term: &RegularizationTerm,
        parameters: &[f64],
        gradient: &mut [f64],
    ) -> QuantRS2Result<()> {
        for &idx in &term.parameter_indices {
            if idx >= parameters.len() {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Regularization parameter index {idx} out of range (total parameters: {})",
                    parameters.len()
                )));
            }
        }

        match term.reg_type {
            RegularizationType::L1 | RegularizationType::Sparsity => {
                for &idx in &term.parameter_indices {
                    gradient[idx] += term.strength * parameters[idx].signum();
                }
            }
            RegularizationType::L2 => {
                for &idx in &term.parameter_indices {
                    gradient[idx] += term.strength * 2.0 * parameters[idx];
                }
            }
            RegularizationType::ElasticNet => {
                for &idx in &term.parameter_indices {
                    gradient[idx] +=
                        term.strength * (0.5 * parameters[idx].signum() + parameters[idx]);
                }
            }
            RegularizationType::TotalVariation | RegularizationType::Smoothness => {
                // d/dθ_k Σ_j (θ_{j+1} − θ_j)² for the ordered selected indices.
                let indices = &term.parameter_indices;
                for window in indices.windows(2) {
                    let (lo, hi) = (window[0], window[1]);
                    let diff = parameters[hi] - parameters[lo];
                    gradient[hi] += term.strength * 2.0 * diff;
                    gradient[lo] -= term.strength * 2.0 * diff;
                }
            }
        }

        Ok(())
    }

    /// Compute the objective gradient with respect to every global parameter.
    ///
    /// The quantum contribution of each parameter is obtained with the analytic
    /// **parameter-shift rule** — for rotation gates `U(θ) = exp(−i θ P / 2)`
    /// (the `RX`/`RY`/`RZ` gates this module binds), `∂⟨H⟩/∂θ = ½[E(θ + π/2) −
    /// E(θ − π/2)]` is exact.  The classical regularization terms contribute
    /// their exact analytic derivative.  Parameters that drive no gate still get
    /// their (regularization) gradient, so the result is a faithful gradient of
    /// the real objective evaluated by [`Self::evaluate_objective`].
    fn compute_gradients<const N: usize>(
        &self,
        problem: &HybridOptimizationProblem<N>,
        parameters: &[f64],
    ) -> QuantRS2Result<Vec<f64>> {
        let num_params = parameters.len();
        let shift = std::f64::consts::FRAC_PI_2;

        // Flatten every (component, parameterized-gate) parameter-shift
        // evaluation into an independent job `(component_index, global_index,
        // weight)`. Each job requires two full state-vector simulations and
        // is otherwise completely independent of every other job, which is
        // exactly what `parallelization.quantum_parallelism` ("number of
        // parallel quantum circuit evaluations") promises to parallelize.
        let mut jobs: Vec<(usize, usize, f64)> = Vec::new();
        for (component_index, component) in problem.quantum_circuits.iter().enumerate() {
            let num_param_gates = count_parameterized_gates(&component.circuit);
            let weight = problem
                .objective
                .weights
                .get(component_index)
                .copied()
                .unwrap_or(1.0);

            for slot in 0..num_param_gates.min(component.parameter_indices.len()) {
                let global_index = component.parameter_indices[slot];
                if global_index >= num_params {
                    return Err(QuantRS2Error::InvalidInput(format!(
                        "Component '{}' references parameter index {} but only {} parameters exist",
                        component.id, global_index, num_params
                    )));
                }
                jobs.push((component_index, global_index, weight));
            }
        }

        let eval_job = |&(component_index, global_index, weight): &(usize, usize, f64)| -> QuantRS2Result<(usize, f64)> {
            let component = &problem.quantum_circuits[component_index];

            let mut plus = parameters.to_vec();
            plus[global_index] += shift;
            let bound_plus = bind_parameters(component, &plus)?;
            let state_plus = statevector::simulate(&bound_plus)?;
            let energy_plus =
                Self::objective_from_state(&state_plus, N, &problem.objective.function_type)?;

            let mut minus = parameters.to_vec();
            minus[global_index] -= shift;
            let bound_minus = bind_parameters(component, &minus)?;
            let state_minus = statevector::simulate(&bound_minus)?;
            let energy_minus =
                Self::objective_from_state(&state_minus, N, &problem.objective.function_type)?;

            Ok((global_index, weight * 0.5 * (energy_plus - energy_minus)))
        };

        let contributions: Vec<QuantRS2Result<(usize, f64)>> =
            if self.parallelization.quantum_parallelism > 1 && jobs.len() > 1 {
                jobs.par_iter().map(eval_job).collect()
            } else {
                jobs.iter().map(eval_job).collect()
            };

        let mut gradients = vec![0.0; num_params];
        for contribution in contributions {
            let (global_index, value) = contribution?;
            gradients[global_index] += value;
        }

        // Every regularization term's analytic gradient is independent of
        // every other term, so `parallelization.classical_parallelism` (the
        // "classical processing" parallel worker count) drives whether these
        // run concurrently, each accumulating into its own gradient buffer
        // that is then summed sequentially.
        if self.parallelization.classical_parallelism > 1
            && problem.objective.regularization.len() > 1
        {
            let partials: Vec<QuantRS2Result<Vec<f64>>> = problem
                .objective
                .regularization
                .par_iter()
                .map(|term| {
                    let mut partial = vec![0.0; num_params];
                    Self::regularization_gradient(term, parameters, &mut partial)?;
                    Ok(partial)
                })
                .collect();
            for partial in partials {
                let partial = partial?;
                for (g, p) in gradients.iter_mut().zip(partial) {
                    *g += p;
                }
            }
        } else {
            for term in &problem.objective.regularization {
                Self::regularization_gradient(term, parameters, &mut gradients)?;
            }
        }

        Ok(gradients)
    }

    /// Get the learning rate for the current iteration under
    /// `self.learning_rate_schedule.schedule_type`.
    ///
    /// `gradient_norm_history` is `history.gradient_norms` as built up so far
    /// (including the value just recorded for `iteration`); it drives
    /// [`ScheduleType::Adaptive`], the only schedule whose rate depends on the
    /// optimization trajectory rather than purely on `iteration`.
    fn get_learning_rate(&self, iteration: usize, gradient_norm_history: &[f64]) -> f64 {
        let initial_rate = self.learning_rate_schedule.initial_rate;
        let params = &self.learning_rate_schedule.parameters;

        match self.learning_rate_schedule.schedule_type {
            ScheduleType::Constant => initial_rate,
            ScheduleType::LinearDecay => {
                let decay_rate = params.get("decay_rate").copied().unwrap_or(0.001);
                initial_rate / (1.0 + decay_rate * iteration as f64)
            }
            ScheduleType::ExponentialDecay => {
                let decay_rate = params.get("decay_rate").copied().unwrap_or(0.95);
                initial_rate * decay_rate.powi(iteration as i32)
            }
            ScheduleType::StepDecay => {
                // Piecewise-constant: multiply by `decay_factor` every
                // `step_size` iterations, e.g. rate, rate*f, rate*f^2, ...
                let step_size = params.get("step_size").copied().unwrap_or(100.0).max(1.0);
                let decay_factor = params.get("decay_factor").copied().unwrap_or(0.5);
                let num_steps = (iteration as f64 / step_size).floor();
                initial_rate * decay_factor.powf(num_steps)
            }
            ScheduleType::CosineAnnealing => {
                // Standard cosine annealing from `initial_rate` down to
                // `min_rate` over `max_iterations`.
                let min_rate = params.get("min_rate").copied().unwrap_or(0.0);
                let total = (self.max_iterations.max(1) - 1) as f64;
                let progress = if total > 0.0 {
                    (iteration as f64 / total).min(1.0)
                } else {
                    0.0
                };
                min_rate
                    + 0.5
                        * (initial_rate - min_rate)
                        * (1.0 + (std::f64::consts::PI * progress).cos())
            }
            ScheduleType::Adaptive => {
                // Scale the rate by the ratio of the previous to the current
                // gradient norm: a shrinking gradient (converging nicely)
                // grows the rate (up to `max_scale`); a growing gradient
                // (overshooting / diverging) shrinks it (down to
                // `min_scale`), a simple, bounded Rprop-style adaptation.
                let min_scale = params.get("min_scale").copied().unwrap_or(0.5);
                let max_scale = params.get("max_scale").copied().unwrap_or(2.0);
                let scale = match gradient_norm_history {
                    [.., previous, current] => {
                        let ratio = previous / current.max(1e-15);
                        ratio.clamp(min_scale, max_scale)
                    }
                    _ => 1.0,
                };
                initial_rate * scale
            }
        }
    }

    /// Which global-parameter indices are updated this iteration, per
    /// `self.algorithm`.
    ///
    /// * [`HybridOptimizationAlgorithm::SimultaneousOptimization`] updates
    ///   every parameter every iteration (plain full-batch gradient descent).
    /// * [`HybridOptimizationAlgorithm::CoordinateDescent`] alternates whole
    ///   blocks: on even iterations only `quantum_indices` (parameters that
    ///   drive at least one gate) update; on odd iterations only the
    ///   remaining classical/regularization-only parameters update. If either
    ///   block is empty the alternation is degenerate, so every parameter is
    ///   simply updated every iteration.
    /// * [`HybridOptimizationAlgorithm::HierarchicalOptimization`] anneals
    ///   from coarse to fine resolution: early iterations only update
    ///   parameters at indices that are multiples of a shrinking
    ///   power-of-two stride, converging to "every parameter" (stride `1`) by
    ///   the end of the run -- a coarse-to-fine parameter grouping.
    /// * [`HybridOptimizationAlgorithm::AdaptiveOptimization`] behaves like
    ///   `SimultaneousOptimization` while the objective keeps improving, and
    ///   falls back to the `CoordinateDescent` block schedule as soon as it
    ///   fails to improve on the previous iteration (an adaptive escape from
    ///   a stalled full-batch step).
    ///
    /// [`HybridOptimizationAlgorithm::Custom`] never reaches this method:
    /// `optimize` rejects it up front with an honest
    /// [`QuantRS2Error::UnsupportedOperation`].
    fn active_parameter_mask(
        &self,
        quantum_indices: &HashSet<usize>,
        iteration: usize,
        num_params: usize,
        recent_objectives: &[f64],
    ) -> Vec<bool> {
        match &self.algorithm {
            HybridOptimizationAlgorithm::SimultaneousOptimization => vec![true; num_params],
            HybridOptimizationAlgorithm::CoordinateDescent => {
                coordinate_descent_mask(quantum_indices, iteration, num_params)
            }
            HybridOptimizationAlgorithm::HierarchicalOptimization => {
                hierarchical_mask(iteration, self.max_iterations, num_params)
            }
            HybridOptimizationAlgorithm::AdaptiveOptimization => {
                let improving = match recent_objectives {
                    [.., previous, current] => *current < previous - 1e-12,
                    _ => true,
                };
                if improving {
                    vec![true; num_params]
                } else {
                    coordinate_descent_mask(quantum_indices, iteration, num_params)
                }
            }
            HybridOptimizationAlgorithm::Custom(_) => vec![true; num_params],
        }
    }

    /// Extract real quantum-state information for every component.
    ///
    /// Each component's circuit is bound with the problem's current
    /// `global_parameters` and simulated exactly; from the resulting amplitudes
    /// we record:
    ///
    /// * `final_states` — the full `2^N` state vector `|ψ(θ)⟩`.
    /// * `measurement_stats` — per-qubit `⟨Z⟩` means with their statistical
    ///   standard deviations `√(1 − ⟨Z⟩²)` (the exact spread of a `±1` Z
    ///   measurement) computed directly from the state.
    /// * `entanglement_info` — the von Neumann entropy of qubit 0's reduced
    ///   density matrix together with that single-qubit entanglement spectrum.
    ///
    /// Nothing here is fabricated: every number is derived from the simulated
    /// amplitudes.  A component with no parameterized gates simply yields the
    /// state produced by its fixed gates.
    fn extract_quantum_info<const N: usize>(
        &self,
        problem: &HybridOptimizationProblem<N>,
    ) -> QuantRS2Result<QuantumStateInfo> {
        let mut final_states = HashMap::new();
        let mut measurement_stats = HashMap::new();
        let mut entanglement_info = HashMap::new();

        for component in &problem.quantum_circuits {
            let bound = bind_parameters(component, &problem.global_parameters)?;
            let state = statevector::simulate(&bound)?;

            // Per-qubit ⟨Z⟩ means and their exact Z-measurement std-devs.
            let mut means = Vec::with_capacity(N);
            let mut std_devs = Vec::with_capacity(N);
            for qubit in 0..N {
                let z_expectation = statevector::single_z_expectation(&state, qubit);
                means.push(z_expectation);
                // Var(Z) = ⟨Z²⟩ − ⟨Z⟩² = 1 − ⟨Z⟩² for a ±1-valued observable.
                std_devs.push((1.0 - z_expectation * z_expectation).max(0.0).sqrt());
            }

            measurement_stats.insert(
                component.id.clone(),
                MeasurementStatistics {
                    means,
                    std_devs,
                    correlations: Vec::new(),
                    num_shots: 0,
                },
            );

            // Entanglement of qubit 0 with the rest of the register.
            if N >= 1 {
                let spectrum = statevector::single_qubit_eigenvalues(&state, 0);
                let entropy = von_neumann_entropy(&spectrum);
                entanglement_info.insert(
                    component.id.clone(),
                    EntanglementInfo {
                        von_neumann_entropy: entropy,
                        mutual_information: Vec::new(),
                        entanglement_spectrum: spectrum,
                    },
                );
            }

            final_states.insert(component.id.clone(), state);
        }

        Ok(QuantumStateInfo {
            final_states,
            measurement_stats,
            entanglement_info,
        })
    }
}

/// Collect the global-parameter indices that drive at least one parameterized
/// gate (RX/RY/RZ) of some quantum component, i.e. the indices that admit the
/// parameter-shift rule in [`HybridOptimizer::compute_gradients`].
///
/// Every other index only ever appears in classical regularization terms.
/// This partition is what [`HybridOptimizationAlgorithm::CoordinateDescent`]
/// and [`HybridOptimizationAlgorithm::AdaptiveOptimization`] alternate
/// between.
fn quantum_parameter_indices<const N: usize>(
    problem: &HybridOptimizationProblem<N>,
) -> HashSet<usize> {
    let mut indices = HashSet::new();
    for component in &problem.quantum_circuits {
        let num_param_gates = count_parameterized_gates(&component.circuit);
        for &idx in component.parameter_indices.iter().take(num_param_gates) {
            indices.insert(idx);
        }
    }
    indices
}

/// Block-coordinate-descent mask: alternates between the `quantum_indices`
/// block (even iterations) and the complementary classical block (odd
/// iterations). Degenerates to "update everything" when one of the two
/// blocks is empty, since there is then nothing to alternate with.
fn coordinate_descent_mask(
    quantum_indices: &HashSet<usize>,
    iteration: usize,
    num_params: usize,
) -> Vec<bool> {
    let classical_count = num_params.saturating_sub(quantum_indices.len());
    if quantum_indices.is_empty() || classical_count == 0 {
        return vec![true; num_params];
    }

    let update_quantum_this_round = iteration % 2 == 0;
    (0..num_params)
        .map(|i| quantum_indices.contains(&i) == update_quantum_this_round)
        .collect()
}

/// Coarse-to-fine hierarchical mask: at the coarsest level (`iteration ==
/// 0`) only every `2^max_level`-th parameter updates; the level shrinks by
/// one every `max_iterations / (max_level + 1)` iterations until it reaches
/// `0` (stride `1`, i.e. every parameter updates), refining the resolution
/// as optimization proceeds.
fn hierarchical_mask(iteration: usize, max_iterations: usize, num_params: usize) -> Vec<bool> {
    if num_params == 0 {
        return Vec::new();
    }

    let max_level = (num_params as f64).log2().floor() as u32;
    let num_phases = max_level + 1;
    let phase_len = ((max_iterations.max(1) as f64) / (num_phases as f64))
        .ceil()
        .max(1.0) as usize;
    let phase = (iteration / phase_len).min(max_level as usize) as u32;
    let level = max_level - phase;
    let stride = 1usize << level;

    (0..num_params).map(|i| i % stride == 0).collect()
}

/// Count the parameterized rotation gates (RX/RY/RZ) in a circuit, in gate order.
fn count_parameterized_gates<const N: usize>(circuit: &Circuit<N>) -> usize {
    circuit
        .gates()
        .iter()
        .filter(|gate| {
            let any = gate.as_any();
            any.is::<RotationX>() || any.is::<RotationY>() || any.is::<RotationZ>()
        })
        .count()
}

/// Bind `parameters` into a copy of `component`'s circuit.
///
/// The `k`-th entry of `component.parameter_indices` supplies the angle for the
/// `k`-th parameterized rotation gate (RX/RY/RZ) encountered in gate order; all
/// other gates are preserved verbatim.  If a component lists more parameter
/// indices than it has parameterized gates the surplus indices are ignored
/// (they may, for example, only feed classical regularization), and vice-versa.
fn bind_parameters<const N: usize>(
    component: &ParameterizedQuantumComponent<N>,
    parameters: &[f64],
) -> QuantRS2Result<Circuit<N>> {
    let old_gates = component.circuit.gates_as_boxes();
    let mut param_slot = 0usize;
    let mut new_gates: Vec<Box<dyn GateOp>> = Vec::with_capacity(old_gates.len());

    for gate in old_gates {
        let any = gate.as_any();
        // Resolve the global parameter for the current parameterized gate, if any.
        let resolve = |slot: usize| -> QuantRS2Result<Option<f64>> {
            match component.parameter_indices.get(slot) {
                Some(&global_index) => match parameters.get(global_index) {
                    Some(&value) => Ok(Some(value)),
                    None => Err(QuantRS2Error::InvalidInput(format!(
                        "Component '{}' references parameter index {} but only {} parameters exist",
                        component.id,
                        global_index,
                        parameters.len()
                    ))),
                },
                // No global parameter mapped to this gate: keep its existing angle.
                None => Ok(None),
            }
        };

        if let Some(rx) = any.downcast_ref::<RotationX>() {
            let theta = resolve(param_slot)?.unwrap_or(rx.theta);
            param_slot += 1;
            new_gates.push(Box::new(RotationX {
                target: rx.target,
                theta,
            }));
        } else if let Some(ry) = any.downcast_ref::<RotationY>() {
            let theta = resolve(param_slot)?.unwrap_or(ry.theta);
            param_slot += 1;
            new_gates.push(Box::new(RotationY {
                target: ry.target,
                theta,
            }));
        } else if let Some(rz) = any.downcast_ref::<RotationZ>() {
            let theta = resolve(param_slot)?.unwrap_or(rz.theta);
            param_slot += 1;
            new_gates.push(Box::new(RotationZ {
                target: rz.target,
                theta,
            }));
        } else {
            new_gates.push(gate);
        }
    }

    Circuit::<N>::from_gates(new_gates)
}

/// Collect the parameter values referenced by a regularization term, in order.
fn collect_parameters(term: &RegularizationTerm, parameters: &[f64]) -> QuantRS2Result<Vec<f64>> {
    let mut selected = Vec::with_capacity(term.parameter_indices.len());
    for &idx in &term.parameter_indices {
        let value = parameters.get(idx).copied().ok_or_else(|| {
            QuantRS2Error::InvalidInput(format!(
                "Regularization parameter index {idx} out of range (total parameters: {})",
                parameters.len()
            ))
        })?;
        selected.push(value);
    }
    Ok(selected)
}

/// Von Neumann entropy `S = −Σ_i λ_i log₂ λ_i` of a probability/eigenvalue spectrum.
fn von_neumann_entropy(eigenvalues: &[f64]) -> f64 {
    let mut entropy = 0.0;
    for &lambda in eigenvalues {
        if lambda > 1e-12 {
            entropy -= lambda * lambda.log2();
        }
    }
    entropy
}

/// Dense state-vector simulation utilities used by the hybrid objective.
///
/// `quantrs2-circuit` is a dependency of `quantrs2-sim`, so it cannot depend on
/// the simulator crate (that would be a dependency cycle).  These helpers
/// therefore provide a small, self-contained exact state-vector engine driven
/// purely by the generic [`GateOp::matrix`] / [`GateOp::qubits`] interface, so
/// they correctly handle *every* gate type a component circuit can contain.
mod statevector {
    use super::{Circuit, GateOp};
    use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
    use scirs2_core::Complex64;

    /// Simulate `circuit` on `2^N` amplitudes starting from `|0…0⟩`.
    pub fn simulate<const N: usize>(circuit: &Circuit<N>) -> QuantRS2Result<Vec<Complex64>> {
        let dim = 1usize << N;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);

        for gate in circuit.gates() {
            apply_gate(&mut state, N, gate.as_ref())?;
        }

        Ok(state)
    }

    /// Apply a single (possibly multi-qubit) gate to the state vector in place.
    ///
    /// The gate's `2^k × 2^k` unitary (row-major, `k = gate.num_qubits()`) is
    /// applied to the subspace spanned by the gate's qubits.  Qubit `q` is the
    /// bit at position `q` of the basis index (little-endian), matching the
    /// framework's `QubitId` convention.
    fn apply_gate(
        state: &mut [Complex64],
        num_qubits: usize,
        gate: &dyn GateOp,
    ) -> QuantRS2Result<()> {
        let targets: Vec<usize> = gate.qubits().iter().map(|q| q.id() as usize).collect();
        let k = targets.len();
        if k == 0 {
            return Ok(());
        }
        for &t in &targets {
            if t >= num_qubits {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Gate '{}' acts on qubit {} but circuit only has {} qubits",
                    gate.name(),
                    t,
                    num_qubits
                )));
            }
        }

        let matrix = gate.matrix()?;
        let side = 1usize << k;
        if matrix.len() != side * side {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Gate '{}' returned a {}-element matrix but {} qubits require {}",
                gate.name(),
                matrix.len(),
                k,
                side * side
            )));
        }

        // `bit_masks[b]` is the state-vector bit driven by bit `b` of the local
        // gate-block index.  The gate matrix follows the standard convention in
        // which the *first* qubit of `gate.qubits()` is the most-significant bit
        // of the block index, so we reverse the qubit order here: local bit 0
        // (LSB) ↔ the last qubit, local bit k−1 (MSB) ↔ the first qubit.
        let bit_masks: Vec<usize> = targets.iter().rev().map(|&t| 1usize << t).collect();
        let mut fixed_mask = 0usize;
        for &m in &bit_masks {
            fixed_mask |= m;
        }
        let dim = state.len();

        let mut visited = vec![false; dim];
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); side];
        let mut indices = vec![0usize; side];

        for base in 0..dim {
            if visited[base] || (base & fixed_mask) != 0 {
                continue;
            }

            for (local, slot) in indices.iter_mut().enumerate() {
                let mut idx = base;
                for (bit, &mask) in bit_masks.iter().enumerate() {
                    if (local >> bit) & 1 == 1 {
                        idx |= mask;
                    }
                }
                *slot = idx;
                amplitudes[local] = state[idx];
                visited[idx] = true;
            }

            for r in 0..side {
                let mut acc = Complex64::new(0.0, 0.0);
                let row = r * side;
                for (c, amp) in amplitudes.iter().enumerate() {
                    acc += matrix[row + c] * amp;
                }
                state[indices[r]] = acc;
            }
        }

        Ok(())
    }

    /// `⟨ψ| Z_qubit |ψ⟩` for a single qubit (eigenvalue `+1` on `|0⟩`, `−1` on `|1⟩`).
    pub fn single_z_expectation(state: &[Complex64], qubit: usize) -> f64 {
        let mask = 1usize << qubit;
        let mut expectation = 0.0;
        for (idx, amp) in state.iter().enumerate() {
            let sign = if idx & mask == 0 { 1.0 } else { -1.0 };
            expectation += sign * amp.norm_sqr();
        }
        expectation
    }

    /// `⟨ψ| (Σ_q Z_q) |ψ⟩` — the diagonal cost Hamiltonian energy.
    pub fn sum_z_expectation(state: &[Complex64], num_qubits: usize) -> f64 {
        (0..num_qubits)
            .map(|q| single_z_expectation(state, q))
            .sum()
    }

    /// Eigenvalues of qubit `qubit`'s reduced density matrix.
    ///
    /// Tracing out every other qubit yields a `2 × 2` Hermitian density matrix
    /// `ρ`; its two eigenvalues quantify the entanglement of that qubit with the
    /// remainder of the register (both `0.5` ⇒ maximal entanglement, `{1, 0}` ⇒
    /// product state).
    pub fn single_qubit_eigenvalues(state: &[Complex64], qubit: usize) -> Vec<f64> {
        let mask = 1usize << qubit;
        // ρ = [[r00, r01], [r10, r11]] with r10 = conj(r01).
        let mut r00 = 0.0;
        let mut r11 = 0.0;
        let mut r01 = Complex64::new(0.0, 0.0);
        for (idx, amp) in state.iter().enumerate() {
            if idx & mask == 0 {
                r00 += amp.norm_sqr();
                let partner = idx | mask;
                r01 += amp.conj() * state[partner];
            } else {
                r11 += amp.norm_sqr();
            }
        }

        // Eigenvalues of a 2×2 Hermitian matrix: (tr ± √(tr² − 4 det)) / 2.
        let trace = r00 + r11;
        let det = r00 * r11 - r01.norm_sqr();
        let discriminant = (trace * trace - 4.0 * det).max(0.0).sqrt();
        let lambda_plus = 0.5 * (trace + discriminant);
        let lambda_minus = 0.5 * (trace - discriminant);
        vec![lambda_plus.max(0.0), lambda_minus.max(0.0)]
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
            .expect("add_quantum_component should succeed");

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
            .expect("add_quantum_component should succeed");
        problem
            .add_classical_step(
                "c1".to_string(),
                ClassicalStepType::LinearAlgebra(LinearAlgebraOp::MatrixMultiplication),
                vec!["q1".to_string()],
                vec!["output".to_string()],
            )
            .expect("add_classical_step should succeed");

        problem
            .add_data_flow(
                "q1".to_string(),
                "c1".to_string(),
                DataType::Measurements(vec![0.1, 0.2]),
            )
            .expect("add_data_flow should succeed");

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

    /// Build a one-qubit problem whose single RY gate is driven by parameter 0,
    /// with an `ExpectationValue` objective (⟨Z⟩).
    fn single_ry_problem(theta: f64) -> HybridOptimizationProblem<1> {
        let mut problem = HybridOptimizationProblem::<1>::new();
        problem.set_global_parameters(vec![theta]);
        let mut circuit = Circuit::<1>::new();
        circuit
            .ry(QubitId(0), 0.0)
            .expect("add RY gate to test circuit");
        problem
            .add_quantum_component("q".to_string(), circuit, vec![0])
            .expect("add quantum component");
        problem
    }

    /// `⟨0|RY(θ)† Z RY(θ)|0⟩ = cos θ`.  The former fabrication returned a
    /// constant `1.0`; this pins the real objective to the analytic value.
    #[test]
    fn test_objective_matches_analytic_cos() {
        use std::f64::consts::PI;

        let optimizer = HybridOptimizer::default();
        for &theta in &[0.0, PI / 6.0, PI / 3.0, PI / 2.0, 2.0 * PI / 3.0, PI] {
            let problem = single_ry_problem(theta);
            let value = optimizer
                .evaluate_objective(&problem, &problem.global_parameters)
                .expect("evaluate objective");
            assert!(
                (value - theta.cos()).abs() < 1e-9,
                "objective for RY({theta}) was {value}, expected {}",
                theta.cos()
            );
        }
    }

    /// The objective must depend on the parameters — a constant `1.0`
    /// fabrication would make every value identical.
    #[test]
    fn test_objective_is_not_constant() {
        use std::f64::consts::PI;

        let optimizer = HybridOptimizer::default();
        let p0 = single_ry_problem(0.0);
        let p_pi = single_ry_problem(PI);
        let e0 = optimizer
            .evaluate_objective(&p0, &p0.global_parameters)
            .expect("e0");
        let e_pi = optimizer
            .evaluate_objective(&p_pi, &p_pi.global_parameters)
            .expect("e_pi");

        assert!((e0 - 1.0).abs() < 1e-9, "⟨Z⟩ at θ=0 should be +1, got {e0}");
        assert!(
            (e_pi + 1.0).abs() < 1e-9,
            "⟨Z⟩ at θ=π should be -1, got {e_pi}"
        );
        assert!(
            (e0 - e_pi).abs() > 1.0,
            "objective must vary with parameters (e0={e0}, e_pi={e_pi})"
        );
    }

    /// The fidelity objective rewards overlap with |0…0⟩: it is `0` for the
    /// untouched state and grows as the state rotates away.
    #[test]
    fn test_objective_fidelity_variant() {
        use std::f64::consts::PI;

        let optimizer = HybridOptimizer::default();
        let mut problem = single_ry_problem(0.0);
        problem.objective.function_type = ObjectiveFunctionType::Fidelity;

        // θ=0 ⇒ state is |0⟩ ⇒ fidelity 1 ⇒ objective 1 − 1 = 0.
        let e0 = optimizer
            .evaluate_objective(&problem, &[0.0])
            .expect("fidelity θ=0");
        assert!(
            (e0 - 0.0).abs() < 1e-9,
            "expected 0 fidelity-cost, got {e0}"
        );

        // θ=π ⇒ state is |1⟩ ⇒ fidelity 0 ⇒ objective 1.
        let e_pi = optimizer
            .evaluate_objective(&problem, &[PI])
            .expect("fidelity θ=π");
        assert!(
            (e_pi - 1.0).abs() < 1e-9,
            "expected 1 fidelity-cost, got {e_pi}"
        );
    }

    /// The analytic parameter-shift gradient must agree with a central finite
    /// difference of the *real* objective at a generic, non-symmetric point.
    #[test]
    fn test_parameter_shift_gradient_matches_finite_difference() {
        use std::f64::consts::PI;

        let optimizer = HybridOptimizer::default();

        // Two-qubit component: RY(q0), RZ(q1), CNOT, RY(q1) — three param gates.
        let mut problem = HybridOptimizationProblem::<2>::new();
        let base = vec![0.31, -0.52, 1.07 - PI / 4.0];
        problem.set_global_parameters(base.clone());

        let mut circuit = Circuit::<2>::new();
        circuit.ry(QubitId(0), 0.0).expect("ry0");
        circuit.rz(QubitId(1), 0.0).expect("rz1");
        circuit.cnot(QubitId(0), QubitId(1)).expect("cnot");
        circuit.ry(QubitId(1), 0.0).expect("ry1");
        problem
            .add_quantum_component("q".to_string(), circuit, vec![0, 1, 2])
            .expect("add component");

        // A non-trivial multi-term L2 regularization so the classical
        // derivative path is exercised alongside the quantum one.
        problem
            .add_regularization(RegularizationType::L2, 0.13, vec![0, 2])
            .expect("add reg");

        let analytic = optimizer
            .compute_gradients(&problem, &base)
            .expect("analytic gradient");
        assert_eq!(analytic.len(), base.len());

        let eps = 1e-6;
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let ep = optimizer.evaluate_objective(&problem, &plus).expect("e+");

            let mut minus = base.clone();
            minus[i] -= eps;
            let em = optimizer.evaluate_objective(&problem, &minus).expect("e-");

            let numeric = (ep - em) / (2.0 * eps);
            assert!(
                (analytic[i] - numeric).abs() < 1e-5,
                "param {i}: analytic {} vs finite-difference {}",
                analytic[i],
                numeric
            );
        }
    }

    /// The L2 regularization term genuinely contributes to the objective and is
    /// not silently ignored.
    #[test]
    fn test_regularization_contributes() {
        let optimizer = HybridOptimizer::default();

        // θ = π/2 ⇒ ⟨Z⟩ = cos(π/2) = 0, so any non-zero value comes from the
        // regularization term alone.
        let mut problem = single_ry_problem(std::f64::consts::FRAC_PI_2);
        let without = optimizer
            .evaluate_objective(&problem, &problem.global_parameters.clone())
            .expect("without reg");
        assert!(
            without.abs() < 1e-9,
            "quantum part should vanish, got {without}"
        );

        problem
            .add_regularization(RegularizationType::L2, 2.0, vec![0])
            .expect("add reg");
        let with = optimizer
            .evaluate_objective(&problem, &problem.global_parameters.clone())
            .expect("with reg");
        // L2 penalty = strength * θ² = 2.0 * (π/2)².
        let expected = 2.0 * (std::f64::consts::FRAC_PI_2).powi(2);
        assert!(
            (with - expected).abs() < 1e-9,
            "regularized objective {with}, expected {expected}"
        );
    }

    /// End-to-end: minimizing ⟨Z⟩ with an RY ansatz must drive the objective
    /// toward the ground-state value `-1` and populate real quantum info.
    #[test]
    fn test_optimize_reaches_z_ground_state() {
        let mut optimizer = HybridOptimizer::default();
        optimizer.learning_rate_schedule.initial_rate = 0.3;
        optimizer.max_iterations = 500;

        // Start away from the minimum (θ=π) and the maximum (θ=0).
        let mut problem = single_ry_problem(0.6);

        let result = optimizer.optimize(&mut problem).expect("optimize");
        assert!(
            (result.optimal_value + 1.0).abs() < 1e-3,
            "optimized objective {} should approach -1",
            result.optimal_value
        );

        // The extracted quantum info must be real, not empty.
        let stats = result
            .quantum_info
            .measurement_stats
            .get("q")
            .expect("measurement stats present");
        // At the minimum the state is |1⟩, so ⟨Z⟩ ≈ -1.
        assert!(
            (stats.means[0] + 1.0).abs() < 1e-2,
            "⟨Z⟩ at optimum should be ≈ -1, got {}",
            stats.means[0]
        );
        let state = result
            .quantum_info
            .final_states
            .get("q")
            .expect("final state present");
        assert_eq!(state.len(), 2, "1-qubit state must have 2 amplitudes");
    }

    /// `extract_quantum_info` produces real entanglement: a Bell-state circuit
    /// has maximally mixed single-qubit marginals (entropy ≈ 1 bit), whereas a
    /// product state has zero entanglement entropy.
    #[test]
    fn test_extract_quantum_info_entanglement() {
        let optimizer = HybridOptimizer::default();

        // Bell state: H(q0) then CNOT(q0,q1).
        let mut problem = HybridOptimizationProblem::<2>::new();
        let mut circuit = Circuit::<2>::new();
        circuit.h(QubitId(0)).expect("h");
        circuit.cnot(QubitId(0), QubitId(1)).expect("cnot");
        problem
            .add_quantum_component("bell".to_string(), circuit, Vec::new())
            .expect("add component");

        let info = optimizer
            .extract_quantum_info(&problem)
            .expect("extract info");
        let ent = info
            .entanglement_info
            .get("bell")
            .expect("entanglement info present");
        assert!(
            (ent.von_neumann_entropy - 1.0).abs() < 1e-9,
            "Bell state entropy should be 1 bit, got {}",
            ent.von_neumann_entropy
        );

        // Product state |00⟩ (empty circuit) has zero entanglement.
        let mut product = HybridOptimizationProblem::<2>::new();
        product
            .add_quantum_component("prod".to_string(), Circuit::<2>::new(), Vec::new())
            .expect("add product component");
        let product_info = optimizer
            .extract_quantum_info(&product)
            .expect("extract product info");
        let prod_ent = product_info
            .entanglement_info
            .get("prod")
            .expect("product entanglement info");
        assert!(
            prod_ent.von_neumann_entropy < 1e-9,
            "product state entropy should be 0, got {}",
            prod_ent.von_neumann_entropy
        );
    }

    /// `has_circular_dependencies` must catch a cycle that spans more than one
    /// edge (`A -> B -> C -> A`), not just a direct self-loop.
    #[test]
    fn test_multi_node_cycle_is_detected() {
        let mut problem = HybridOptimizationProblem::<1>::new();
        problem.data_flow.nodes = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        problem.data_flow.edges = vec![
            (
                "A".to_string(),
                "B".to_string(),
                DataType::Probabilities(vec![]),
            ),
            (
                "B".to_string(),
                "C".to_string(),
                DataType::Probabilities(vec![]),
            ),
            (
                "C".to_string(),
                "A".to_string(),
                DataType::Probabilities(vec![]),
            ),
        ];

        let result = problem.validate();
        assert!(
            result.is_err(),
            "A->B->C->A must be flagged as a circular dependency"
        );
    }

    /// A genuine DAG (no cycle at all, not even a self-loop) must validate.
    #[test]
    fn test_acyclic_data_flow_validates() {
        let mut problem = HybridOptimizationProblem::<1>::new();
        problem.data_flow.nodes = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        problem.data_flow.edges = vec![
            (
                "A".to_string(),
                "B".to_string(),
                DataType::Probabilities(vec![]),
            ),
            (
                "A".to_string(),
                "C".to_string(),
                DataType::Probabilities(vec![]),
            ),
            (
                "B".to_string(),
                "C".to_string(),
                DataType::Probabilities(vec![]),
            ),
        ];

        assert!(
            problem.validate().is_ok(),
            "acyclic data flow must not be rejected as circular"
        );
    }

    /// `ScheduleType::StepDecay` must be piecewise-constant, dropping by
    /// `decay_factor` every `step_size` iterations, not silently aliasing the
    /// constant initial rate.
    #[test]
    fn test_step_decay_learning_rate() {
        let mut optimizer =
            HybridOptimizer::new(HybridOptimizationAlgorithm::SimultaneousOptimization);
        optimizer.learning_rate_schedule.schedule_type = ScheduleType::StepDecay;
        optimizer.learning_rate_schedule.initial_rate = 0.1;
        optimizer
            .learning_rate_schedule
            .parameters
            .insert("step_size".to_string(), 10.0);
        optimizer
            .learning_rate_schedule
            .parameters
            .insert("decay_factor".to_string(), 0.5);

        let empty_history: Vec<f64> = Vec::new();
        assert!((optimizer.get_learning_rate(0, &empty_history) - 0.1).abs() < 1e-12);
        assert!((optimizer.get_learning_rate(9, &empty_history) - 0.1).abs() < 1e-12);
        assert!((optimizer.get_learning_rate(10, &empty_history) - 0.05).abs() < 1e-12);
        assert!((optimizer.get_learning_rate(20, &empty_history) - 0.025).abs() < 1e-12);
    }

    /// `ScheduleType::CosineAnnealing` must trace a cosine curve from
    /// `initial_rate` at iteration 0 down to `min_rate` at the final
    /// iteration, not silently alias the constant initial rate.
    #[test]
    fn test_cosine_annealing_learning_rate() {
        let mut optimizer =
            HybridOptimizer::new(HybridOptimizationAlgorithm::SimultaneousOptimization);
        optimizer.learning_rate_schedule.schedule_type = ScheduleType::CosineAnnealing;
        optimizer.learning_rate_schedule.initial_rate = 1.0;
        optimizer.max_iterations = 101; // iterations 0..=100
        optimizer
            .learning_rate_schedule
            .parameters
            .insert("min_rate".to_string(), 0.0);

        let empty_history: Vec<f64> = Vec::new();
        let start = optimizer.get_learning_rate(0, &empty_history);
        let mid = optimizer.get_learning_rate(50, &empty_history);
        let end = optimizer.get_learning_rate(100, &empty_history);

        assert!(
            (start - 1.0).abs() < 1e-9,
            "rate at iter 0 should be ~1.0, got {start}"
        );
        assert!(
            mid < start && mid > end,
            "rate should monotonically decay across the run"
        );
        assert!(
            end.abs() < 1e-9,
            "rate at final iter should be ~0.0, got {end}"
        );
    }

    /// `ScheduleType::Adaptive` must scale the rate by the gradient-norm
    /// trend: a shrinking gradient grows the rate, a growing gradient shrinks
    /// it, rather than silently aliasing the constant initial rate.
    #[test]
    fn test_adaptive_learning_rate_tracks_gradient_trend() {
        let mut optimizer =
            HybridOptimizer::new(HybridOptimizationAlgorithm::SimultaneousOptimization);
        optimizer.learning_rate_schedule.schedule_type = ScheduleType::Adaptive;
        optimizer.learning_rate_schedule.initial_rate = 0.1;

        // No history yet: falls back to the initial rate.
        let none: Vec<f64> = Vec::new();
        assert!((optimizer.get_learning_rate(0, &none) - 0.1).abs() < 1e-12);

        // Shrinking gradient norm (converging) => rate should grow.
        let shrinking = vec![1.0, 0.5];
        let grown = optimizer.get_learning_rate(1, &shrinking);
        assert!(
            grown > 0.1,
            "shrinking gradient norm should grow the rate, got {grown}"
        );

        // Growing gradient norm (diverging) => rate should shrink.
        let growing = vec![0.5, 1.0];
        let shrunk = optimizer.get_learning_rate(1, &growing);
        assert!(
            shrunk < 0.1,
            "growing gradient norm should shrink the rate, got {shrunk}"
        );
    }

    /// `quantum_parameter_indices` must contain exactly the global indices
    /// referenced by a component's parameterized gates, not indices that only
    /// feed classical regularization.
    #[test]
    fn test_quantum_parameter_indices_partition() {
        // 2 global parameters: index 0 drives the RY gate, index 1 is
        // regularization-only.
        let mut problem = single_ry_problem(0.3);
        problem.set_global_parameters(vec![0.3, 99.0]);
        problem
            .add_regularization(RegularizationType::L2, 1.0, vec![1])
            .expect("add reg");

        let indices = quantum_parameter_indices(&problem);
        assert!(indices.contains(&0), "index 0 drives the RY gate");
        assert!(
            !indices.contains(&1),
            "index 1 only feeds regularization, must not be 'quantum'"
        );
    }

    /// `CoordinateDescent` must alternate: quantum-only indices update on
    /// even iterations, classical-only indices update on odd iterations.
    #[test]
    fn test_coordinate_descent_mask_alternates() {
        let mut quantum = HashSet::new();
        quantum.insert(0);
        // num_params = 2: index 0 quantum, index 1 classical.
        let even = coordinate_descent_mask(&quantum, 0, 2);
        let odd = coordinate_descent_mask(&quantum, 1, 2);
        assert_eq!(
            even,
            vec![true, false],
            "even iteration updates the quantum block"
        );
        assert_eq!(
            odd,
            vec![false, true],
            "odd iteration updates the classical block"
        );
    }

    /// With no classical parameters at all, `CoordinateDescent` must degrade
    /// to updating every parameter every iteration rather than stalling.
    #[test]
    fn test_coordinate_descent_mask_degenerates_without_classical_block() {
        let mut quantum = HashSet::new();
        quantum.insert(0);
        quantum.insert(1);
        let mask = coordinate_descent_mask(&quantum, 1, 2);
        assert_eq!(mask, vec![true, true]);
    }

    /// `HierarchicalOptimization` must start coarse (few active parameters)
    /// and refine to "every parameter active" by the final iteration.
    #[test]
    fn test_hierarchical_mask_coarse_to_fine() {
        let num_params = 8;
        let max_iterations = 80;

        let coarse = hierarchical_mask(0, max_iterations, num_params);
        let coarse_active = coarse.iter().filter(|&&b| b).count();
        assert!(
            coarse_active < num_params,
            "iteration 0 should not yet update every parameter, got {coarse_active}/{num_params}"
        );
        assert!(coarse[0], "index 0 is always active at every level");

        let fine = hierarchical_mask(max_iterations - 1, max_iterations, num_params);
        assert!(
            fine.iter().all(|&b| b),
            "the final iteration must update every parameter"
        );
    }

    /// `HybridOptimizationAlgorithm::Custom` must be an honest error, not a
    /// silent alias for `SimultaneousOptimization`.
    #[test]
    fn test_custom_algorithm_is_honest_error() {
        let optimizer =
            HybridOptimizer::new(HybridOptimizationAlgorithm::Custom("my-algo".to_string()));
        let mut problem = single_ry_problem(0.3);
        let result = optimizer.optimize(&mut problem);
        assert!(
            matches!(result, Err(QuantRS2Error::UnsupportedOperation(_))),
            "Custom algorithm must error honestly, got {result:?}"
        );
    }

    /// End-to-end: `CoordinateDescent` on a problem with both a quantum and a
    /// classical parameter must still converge (alternating updates instead
    /// of a single full-batch step per iteration).
    #[test]
    fn test_coordinate_descent_end_to_end_converges() {
        let mut optimizer = HybridOptimizer::new(HybridOptimizationAlgorithm::CoordinateDescent);
        optimizer.learning_rate_schedule.initial_rate = 0.3;
        optimizer.max_iterations = 2000;

        // Index 0 drives the RY gate (quantum); index 1 only feeds an L2
        // regularization term (classical), so the two blocks alternate.
        let mut problem = single_ry_problem(0.6);
        problem.set_global_parameters(vec![0.6, 5.0]);
        problem
            .add_regularization(RegularizationType::L2, 0.5, vec![1])
            .expect("add reg");

        let result = optimizer.optimize(&mut problem).expect("optimize");
        assert!(
            (result.optimal_value + 1.0).abs() < 1e-2,
            "quantum part of the objective {} should approach -1",
            result.optimal_value
        );
        assert!(
            result.optimal_parameters[1].abs() < 1e-1,
            "classical parameter should be driven toward 0 by L2 regularization, got {}",
            result.optimal_parameters[1]
        );
    }

    /// Parallel evaluation (quantum/classical parallelism > 1) must produce
    /// the same objective and gradients as the sequential path -- the
    /// parallelism setting changes *how* the work is scheduled, not the
    /// answer.
    #[test]
    fn test_parallelization_matches_sequential_result() {
        let mut problem = single_ry_problem(0.4);
        problem.set_global_parameters(vec![0.4, 2.0]);
        problem
            .add_regularization(RegularizationType::L2, 1.0, vec![1])
            .expect("add reg");

        let sequential =
            HybridOptimizer::new(HybridOptimizationAlgorithm::SimultaneousOptimization);
        let mut parallel =
            HybridOptimizer::new(HybridOptimizationAlgorithm::SimultaneousOptimization);
        parallel.parallelization.quantum_parallelism = 8;
        parallel.parallelization.classical_parallelism = 8;

        let seq_value = sequential
            .evaluate_objective(&problem, &problem.global_parameters.clone())
            .expect("sequential objective");
        let par_value = parallel
            .evaluate_objective(&problem, &problem.global_parameters.clone())
            .expect("parallel objective");
        assert!((seq_value - par_value).abs() < 1e-12);

        let seq_grad = sequential
            .compute_gradients(&problem, &problem.global_parameters.clone())
            .expect("sequential gradients");
        let par_grad = parallel
            .compute_gradients(&problem, &problem.global_parameters.clone())
            .expect("parallel gradients");
        assert_eq!(seq_grad.len(), par_grad.len());
        for (s, p) in seq_grad.iter().zip(par_grad.iter()) {
            assert!((s - p).abs() < 1e-9, "sequential {s} vs parallel {p}");
        }
    }
}
