//! Integration with annealing module for QUBO problems in quantum ML
//!
//! This module provides seamless integration between quantum ML algorithms
//! and the QuantRS2 annealing module, enabling optimization of quantum ML
//! models using quantum annealing and classical optimization techniques.

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use quantrs2_anneal::{ising::IsingModel, qubo::QuboBuilder, simulator::*};
use std::collections::HashMap;

/// QUBO formulation of quantum ML optimization problems
pub struct QuantumMLQUBO {
    /// QUBO matrix
    qubo_matrix: Array2<f64>,
    /// Problem description
    description: String,
    /// Variable mapping
    variable_map: HashMap<String, usize>,
    /// Objective function offset
    offset: f64,
}

impl QuantumMLQUBO {
    /// Create new QUBO formulation
    pub fn new(size: usize, description: impl Into<String>) -> Self {
        Self {
            qubo_matrix: Array2::zeros((size, size)),
            description: description.into(),
            variable_map: HashMap::new(),
            offset: 0.0,
        }
    }

    /// Set QUBO coefficient
    pub fn set_coefficient(&mut self, i: usize, j: usize, value: f64) -> Result<()> {
        if i >= self.qubo_matrix.nrows() || j >= self.qubo_matrix.ncols() {
            return Err(MLError::InvalidConfiguration(
                "Index out of bounds".to_string(),
            ));
        }
        self.qubo_matrix[[i, j]] = value;
        Ok(())
    }

    /// Add variable mapping
    pub fn add_variable(&mut self, name: impl Into<String>, index: usize) {
        self.variable_map.insert(name.into(), index);
    }

    /// Get QUBO matrix
    pub fn qubo_matrix(&self) -> &Array2<f64> {
        &self.qubo_matrix
    }

    /// Convert to Ising problem
    pub fn to_ising(&self) -> IsingProblem {
        // Convert QUBO to Ising using standard transformation
        let n = self.qubo_matrix.nrows();
        let mut h = Array1::zeros(n);
        let mut j = Array2::zeros((n, n));
        let mut offset = self.offset;

        // Standard QUBO to Ising transformation
        for i in 0..n {
            h[i] = self.qubo_matrix[[i, i]];
            for k in 0..n {
                if k != i {
                    h[i] += 0.5 * self.qubo_matrix[[i, k]];
                }
            }
            offset += 0.5 * self.qubo_matrix[[i, i]];
        }

        for i in 0..n {
            for k in i + 1..n {
                j[[i, k]] = 0.25 * self.qubo_matrix[[i, k]];
            }
        }

        IsingProblem::new(h, j, offset)
    }
}

/// Ising problem representation
#[derive(Debug, Clone)]
pub struct IsingProblem {
    /// Local magnetic fields
    pub h: Array1<f64>,
    /// Coupling strengths
    pub j: Array2<f64>,
    /// Energy offset
    pub offset: f64,
}

impl IsingProblem {
    /// Create new Ising problem
    pub fn new(h: Array1<f64>, j: Array2<f64>, offset: f64) -> Self {
        Self { h, j, offset }
    }

    /// Compute energy of a configuration
    pub fn energy(&self, spins: &[i8]) -> f64 {
        let mut energy = self.offset;

        // Linear terms
        for (i, &spin) in spins.iter().enumerate() {
            energy -= self.h[i] * spin as f64;
        }

        // Quadratic terms
        for i in 0..spins.len() {
            for j in i + 1..spins.len() {
                energy -= self.j[[i, j]] * spins[i] as f64 * spins[j] as f64;
            }
        }

        energy
    }
}

/// Quantum ML annealing optimizer
pub struct QuantumMLAnnealer {
    /// Annealing parameters
    params: AnnealingParams,
    /// Problem embedding
    embedding: Option<Embedding>,
    /// Anneal client
    client: Option<Box<dyn AnnealingClient>>,
}

/// Annealing parameters
#[derive(Debug, Clone)]
pub struct AnnealingParams {
    /// Number of annealing sweeps
    pub num_sweeps: usize,
    /// Annealing schedule
    pub schedule: AnnealingSchedule,
    /// Temperature range
    pub temperature_range: (f64, f64),
    /// Number of chains
    pub num_chains: usize,
    /// Chain strength
    pub chain_strength: f64,
}

impl Default for AnnealingParams {
    fn default() -> Self {
        Self {
            num_sweeps: 1000,
            schedule: AnnealingSchedule::Linear,
            temperature_range: (0.01, 10.0),
            num_chains: 1,
            chain_strength: 1.0,
        }
    }
}

/// Annealing schedule types
#[derive(Debug, Clone)]
pub enum AnnealingSchedule {
    /// Linear schedule
    Linear,
    /// Exponential schedule
    Exponential,
    /// Custom schedule
    Custom(Vec<f64>),
}

/// Problem embedding for hardware
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Logical to physical qubit mapping
    pub logical_to_physical: HashMap<usize, Vec<usize>>,
    /// Physical to logical qubit mapping
    pub physical_to_logical: HashMap<usize, usize>,
}

impl Embedding {
    /// Create identity embedding
    pub fn identity(num_qubits: usize) -> Self {
        let logical_to_physical: HashMap<usize, Vec<usize>> =
            (0..num_qubits).map(|i| (i, vec![i])).collect();
        let physical_to_logical: HashMap<usize, usize> = (0..num_qubits).map(|i| (i, i)).collect();

        Self {
            logical_to_physical,
            physical_to_logical,
        }
    }

    /// Create embedding with chains
    pub fn with_chains(chains: HashMap<usize, Vec<usize>>) -> Self {
        let mut physical_to_logical = HashMap::new();
        for (logical, physical_qubits) in &chains {
            for &physical in physical_qubits {
                physical_to_logical.insert(physical, *logical);
            }
        }

        Self {
            logical_to_physical: chains,
            physical_to_logical,
        }
    }
}

/// Annealing client trait
pub trait AnnealingClient: Send + Sync {
    /// Solve QUBO problem
    fn solve_qubo(&self, qubo: &QuantumMLQUBO, params: &AnnealingParams)
        -> Result<AnnealingResult>;

    /// Solve Ising problem
    fn solve_ising(
        &self,
        ising: &IsingProblem,
        params: &AnnealingParams,
    ) -> Result<AnnealingResult>;

    /// Get client name
    fn name(&self) -> &str;

    /// Get hardware capabilities
    fn capabilities(&self) -> AnnealingCapabilities;
}

/// Annealing result
#[derive(Debug, Clone)]
pub struct AnnealingResult {
    /// Solution (variable assignments)
    pub solution: Array1<i8>,
    /// Energy of solution
    pub energy: f64,
    /// Number of evaluations
    pub num_evaluations: usize,
    /// Timing information
    pub timing: AnnealingTiming,
    /// Additional metadata
    pub metadata: HashMap<String, f64>,
}

/// Annealing timing information
#[derive(Debug, Clone)]
pub struct AnnealingTiming {
    /// Total solve time
    pub total_time: std::time::Duration,
    /// Queue time (for cloud services)
    pub queue_time: Option<std::time::Duration>,
    /// Annealing time
    pub anneal_time: Option<std::time::Duration>,
}

/// Annealing hardware capabilities
#[derive(Debug, Clone)]
pub struct AnnealingCapabilities {
    /// Maximum number of variables
    pub max_variables: usize,
    /// Maximum number of couplers
    pub max_couplers: usize,
    /// Connectivity graph
    pub connectivity: ConnectivityGraph,
    /// Supported problems
    pub supported_problems: Vec<ProblemType>,
}

/// Connectivity graph
#[derive(Debug, Clone)]
pub enum ConnectivityGraph {
    /// Complete graph
    Complete,
    /// Chimera graph
    Chimera {
        rows: usize,
        cols: usize,
        shore: usize,
    },
    /// Pegasus graph
    Pegasus { size: usize },
    /// Custom graph
    Custom(Array2<bool>),
}

/// Problem types
#[derive(Debug, Clone, Copy)]
pub enum ProblemType {
    /// QUBO problems
    QUBO,
    /// Ising problems
    Ising,
    /// Constrained optimization
    Constrained,
}

impl QuantumMLAnnealer {
    /// Create new annealer
    pub fn new() -> Self {
        Self {
            params: AnnealingParams::default(),
            embedding: None,
            client: None,
        }
    }

    /// Set annealing parameters
    pub fn with_params(mut self, params: AnnealingParams) -> Self {
        self.params = params;
        self
    }

    /// Set problem embedding
    pub fn with_embedding(mut self, embedding: Embedding) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set annealing client
    pub fn with_client(mut self, client: Box<dyn AnnealingClient>) -> Self {
        self.client = Some(client);
        self
    }

    /// Optimize quantum ML problem
    pub fn optimize(&self, problem: QuantumMLOptimizationProblem) -> Result<OptimizationResult> {
        // Convert ML problem to QUBO
        let qubo = self.convert_to_qubo(&problem)?;

        // Solve using annealing
        let anneal_result = if let Some(ref client) = self.client {
            client.solve_qubo(&qubo, &self.params)?
        } else {
            // Use classical simulated annealing
            self.simulated_annealing(&qubo)?
        };

        // Convert back to ML solution
        self.convert_to_ml_solution(&problem, &anneal_result)
    }

    /// Convert ML problem to QUBO formulation
    fn convert_to_qubo(&self, problem: &QuantumMLOptimizationProblem) -> Result<QuantumMLQUBO> {
        match problem {
            QuantumMLOptimizationProblem::FeatureSelection(fs_problem) => {
                self.feature_selection_to_qubo(fs_problem)
            }
            QuantumMLOptimizationProblem::HyperparameterOptimization(hp_problem) => {
                self.hyperparameter_to_qubo(hp_problem)
            }
            QuantumMLOptimizationProblem::CircuitOptimization(circuit_problem) => {
                self.circuit_optimization_to_qubo(circuit_problem)
            }
            QuantumMLOptimizationProblem::PortfolioOptimization(portfolio_problem) => {
                self.portfolio_to_qubo(portfolio_problem)
            }
        }
    }

    /// Convert feature selection to QUBO
    fn feature_selection_to_qubo(
        &self,
        problem: &FeatureSelectionProblem,
    ) -> Result<QuantumMLQUBO> {
        let num_features = problem.feature_importance.len();
        let mut qubo = QuantumMLQUBO::new(num_features, "Feature Selection");

        // Objective: maximize feature importance (minimize negative importance)
        for i in 0..num_features {
            qubo.set_coefficient(i, i, -problem.feature_importance[i])?;
            qubo.add_variable(format!("feature_{}", i), i);
        }

        // Constraint: limit number of selected features
        let penalty = problem.penalty_strength;
        for i in 0..num_features {
            for j in i + 1..num_features {
                qubo.set_coefficient(i, j, penalty)?;
            }
        }

        Ok(qubo)
    }

    /// Convert hyperparameter optimization to QUBO
    fn hyperparameter_to_qubo(&self, problem: &HyperparameterProblem) -> Result<QuantumMLQUBO> {
        let total_bits = problem
            .parameter_encodings
            .iter()
            .map(|encoding| encoding.num_bits)
            .sum();

        let mut qubo = QuantumMLQUBO::new(total_bits, "Hyperparameter Optimization");

        // Encode discrete hyperparameters as binary variables
        let mut bit_offset = 0;
        for (param_idx, encoding) in problem.parameter_encodings.iter().enumerate() {
            for bit in 0..encoding.num_bits {
                qubo.add_variable(format!("param_{}_bit_{}", param_idx, bit), bit_offset + bit);
            }
            bit_offset += encoding.num_bits;
        }

        // Objective would be based on cross-validation performance
        // This is a simplified placeholder
        for i in 0..total_bits {
            qubo.set_coefficient(i, i, fastrand::f64() - 0.5)?;
        }

        Ok(qubo)
    }

    /// Convert circuit optimization to QUBO
    fn circuit_optimization_to_qubo(
        &self,
        problem: &CircuitOptimizationProblem,
    ) -> Result<QuantumMLQUBO> {
        let num_positions = problem.gate_positions.len();
        let num_gate_types = problem.gate_types.len();
        let total_vars = num_positions * num_gate_types;

        let mut qubo = QuantumMLQUBO::new(total_vars, "Circuit Optimization");

        // Variables: x_{i,g} = 1 if gate type g is at position i
        for pos in 0..num_positions {
            for gate in 0..num_gate_types {
                let var_idx = pos * num_gate_types + gate;
                qubo.add_variable(format!("pos_{}_gate_{}", pos, gate), var_idx);

                // Objective: minimize circuit depth weighted by gate costs
                let cost = problem.gate_costs.get(&gate).copied().unwrap_or(1.0);
                qubo.set_coefficient(var_idx, var_idx, cost)?;
            }
        }

        // Constraint: exactly one gate type per position
        let penalty = 10.0;
        for pos in 0..num_positions {
            for g1 in 0..num_gate_types {
                for g2 in g1 + 1..num_gate_types {
                    let var1 = pos * num_gate_types + g1;
                    let var2 = pos * num_gate_types + g2;
                    qubo.set_coefficient(var1, var2, penalty)?;
                }
            }
        }

        Ok(qubo)
    }

    /// Convert portfolio optimization to QUBO
    fn portfolio_to_qubo(&self, problem: &PortfolioOptimizationProblem) -> Result<QuantumMLQUBO> {
        let num_assets = problem.expected_returns.len();
        let mut qubo = QuantumMLQUBO::new(num_assets, "Portfolio Optimization");

        // Objective: maximize return - risk penalty
        for i in 0..num_assets {
            // Return term
            qubo.set_coefficient(i, i, -problem.expected_returns[i])?;
            qubo.add_variable(format!("asset_{}", i), i);
        }

        // Risk term (covariance)
        for i in 0..num_assets {
            for j in i..num_assets {
                let risk_penalty = problem.risk_aversion * problem.covariance_matrix[[i, j]];
                qubo.set_coefficient(i, j, risk_penalty)?;
            }
        }

        Ok(qubo)
    }

    /// Simulated annealing fallback
    fn simulated_annealing(&self, qubo: &QuantumMLQUBO) -> Result<AnnealingResult> {
        let start_time = std::time::Instant::now();
        let n = qubo.qubo_matrix.nrows();
        let mut solution = Array1::from_vec(
            (0..n)
                .map(|_| if fastrand::bool() { 1 } else { -1 })
                .collect(),
        );
        let mut best_energy = self.compute_qubo_energy(qubo, &solution);

        let (t_start, t_end) = self.params.temperature_range;
        let cooling_rate = (t_end / t_start).powf(1.0 / self.params.num_sweeps as f64);
        let mut temperature = t_start;

        for _sweep in 0..self.params.num_sweeps {
            for i in 0..n {
                // Flip spin
                solution[i] *= -1;
                let new_energy = self.compute_qubo_energy(qubo, &solution);

                // Accept or reject move
                if new_energy < best_energy
                    || fastrand::f64() < ((best_energy - new_energy) / temperature).exp()
                {
                    best_energy = new_energy;
                } else {
                    // Reject move
                    solution[i] *= -1;
                }
            }
            temperature *= cooling_rate;
        }

        Ok(AnnealingResult {
            solution,
            energy: best_energy,
            num_evaluations: self.params.num_sweeps * n,
            timing: AnnealingTiming {
                total_time: start_time.elapsed(),
                queue_time: None,
                anneal_time: Some(start_time.elapsed()),
            },
            metadata: HashMap::new(),
        })
    }

    /// Compute QUBO energy
    fn compute_qubo_energy(&self, qubo: &QuantumMLQUBO, solution: &Array1<i8>) -> f64 {
        let mut energy = 0.0;
        let n = solution.len();

        for i in 0..n {
            for j in 0..n {
                energy += qubo.qubo_matrix[[i, j]] * (solution[i] as f64) * (solution[j] as f64);
            }
        }

        energy
    }

    /// Convert annealing result to ML solution
    fn convert_to_ml_solution(
        &self,
        problem: &QuantumMLOptimizationProblem,
        result: &AnnealingResult,
    ) -> Result<OptimizationResult> {
        match problem {
            QuantumMLOptimizationProblem::FeatureSelection(_) => {
                let selected_features: Vec<usize> = result
                    .solution
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &val)| if val > 0 { Some(i) } else { None })
                    .collect();

                Ok(OptimizationResult::FeatureSelection { selected_features })
            }
            QuantumMLOptimizationProblem::HyperparameterOptimization(hp_problem) => {
                let mut parameters = Vec::new();
                let mut bit_offset = 0;

                for encoding in &hp_problem.parameter_encodings {
                    let mut param_value = 0;
                    for bit in 0..encoding.num_bits {
                        if result.solution[bit_offset + bit] > 0 {
                            param_value |= 1 << bit;
                        }
                    }
                    parameters.push(param_value as f64);
                    bit_offset += encoding.num_bits;
                }

                Ok(OptimizationResult::Hyperparameters { parameters })
            }
            QuantumMLOptimizationProblem::CircuitOptimization(circuit_problem) => {
                let num_gate_types = circuit_problem.gate_types.len();
                let mut circuit_design = Vec::new();

                for pos in 0..circuit_problem.gate_positions.len() {
                    for gate in 0..num_gate_types {
                        let var_idx = pos * num_gate_types + gate;
                        if result.solution[var_idx] > 0 {
                            circuit_design.push(gate);
                            break;
                        }
                    }
                }

                Ok(OptimizationResult::CircuitDesign { circuit_design })
            }
            QuantumMLOptimizationProblem::PortfolioOptimization(_) => {
                let portfolio: Vec<f64> = result
                    .solution
                    .iter()
                    .map(|&val| if val > 0 { 1.0 } else { 0.0 })
                    .collect();

                Ok(OptimizationResult::Portfolio { weights: portfolio })
            }
        }
    }
}

/// Quantum ML optimization problems
#[derive(Debug, Clone)]
pub enum QuantumMLOptimizationProblem {
    /// Feature selection
    FeatureSelection(FeatureSelectionProblem),
    /// Hyperparameter optimization
    HyperparameterOptimization(HyperparameterProblem),
    /// Circuit optimization
    CircuitOptimization(CircuitOptimizationProblem),
    /// Portfolio optimization
    PortfolioOptimization(PortfolioOptimizationProblem),
}

/// Feature selection problem
#[derive(Debug, Clone)]
pub struct FeatureSelectionProblem {
    /// Feature importance scores
    pub feature_importance: Array1<f64>,
    /// Penalty strength for number of features
    pub penalty_strength: f64,
    /// Maximum number of features
    pub max_features: Option<usize>,
}

/// Hyperparameter optimization problem
#[derive(Debug, Clone)]
pub struct HyperparameterProblem {
    /// Parameter encodings
    pub parameter_encodings: Vec<ParameterEncoding>,
    /// Cross-validation function (placeholder)
    pub cv_function: String,
}

/// Parameter encoding for QUBO
#[derive(Debug, Clone)]
pub struct ParameterEncoding {
    /// Parameter name
    pub name: String,
    /// Number of bits for encoding
    pub num_bits: usize,
    /// Value range
    pub range: (f64, f64),
}

/// Circuit optimization problem
#[derive(Debug, Clone)]
pub struct CircuitOptimizationProblem {
    /// Available gate positions
    pub gate_positions: Vec<usize>,
    /// Available gate types
    pub gate_types: Vec<String>,
    /// Cost of each gate type
    pub gate_costs: HashMap<usize, f64>,
    /// Connectivity constraints
    pub connectivity: Array2<bool>,
}

/// Portfolio optimization problem
#[derive(Debug, Clone)]
pub struct PortfolioOptimizationProblem {
    /// Expected returns
    pub expected_returns: Array1<f64>,
    /// Covariance matrix
    pub covariance_matrix: Array2<f64>,
    /// Risk aversion parameter
    pub risk_aversion: f64,
    /// Budget constraint
    pub budget: f64,
}

/// Optimization result
#[derive(Debug, Clone)]
pub enum OptimizationResult {
    /// Selected features
    FeatureSelection { selected_features: Vec<usize> },
    /// Optimized hyperparameters
    Hyperparameters { parameters: Vec<f64> },
    /// Optimized circuit design
    CircuitDesign { circuit_design: Vec<usize> },
    /// Portfolio weights
    Portfolio { weights: Vec<f64> },
}

/// D-Wave quantum annealer client
pub struct DWaveClient {
    /// API token
    token: String,
    /// Solver name
    solver: String,
    /// Chain strength
    chain_strength: f64,
}

impl DWaveClient {
    /// Create new D-Wave client
    pub fn new(token: impl Into<String>, solver: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            solver: solver.into(),
            chain_strength: 1.0,
        }
    }

    /// Set chain strength
    pub fn with_chain_strength(mut self, strength: f64) -> Self {
        self.chain_strength = strength;
        self
    }
}

impl AnnealingClient for DWaveClient {
    fn solve_qubo(
        &self,
        qubo: &QuantumMLQUBO,
        params: &AnnealingParams,
    ) -> Result<AnnealingResult> {
        // Placeholder - would make actual D-Wave API call
        let start_time = std::time::Instant::now();
        let n = qubo.qubo_matrix.nrows();
        let solution = Array1::from_vec(
            (0..n)
                .map(|_| if fastrand::bool() { 1 } else { -1 })
                .collect(),
        );
        let energy = 0.0; // Would compute actual energy

        Ok(AnnealingResult {
            solution,
            energy,
            num_evaluations: params.num_sweeps,
            timing: AnnealingTiming {
                total_time: start_time.elapsed(),
                queue_time: Some(std::time::Duration::from_millis(100)),
                anneal_time: Some(std::time::Duration::from_millis(20)),
            },
            metadata: HashMap::new(),
        })
    }

    fn solve_ising(
        &self,
        ising: &IsingProblem,
        params: &AnnealingParams,
    ) -> Result<AnnealingResult> {
        // Convert to QUBO and solve
        let qubo = self.ising_to_qubo(ising);
        self.solve_qubo(&qubo, params)
    }

    fn name(&self) -> &str {
        "D-Wave"
    }

    fn capabilities(&self) -> AnnealingCapabilities {
        AnnealingCapabilities {
            max_variables: 5000,
            max_couplers: 40000,
            connectivity: ConnectivityGraph::Pegasus { size: 16 },
            supported_problems: vec![ProblemType::QUBO, ProblemType::Ising],
        }
    }
}

impl DWaveClient {
    fn ising_to_qubo(&self, ising: &IsingProblem) -> QuantumMLQUBO {
        let n = ising.h.len();
        let mut qubo = QuantumMLQUBO::new(n, "Ising to QUBO");

        // Standard Ising to QUBO transformation
        for i in 0..n {
            qubo.set_coefficient(i, i, -2.0 * ising.h[i]).unwrap();
        }

        for i in 0..n {
            for j in i + 1..n {
                qubo.set_coefficient(i, j, -4.0 * ising.j[[i, j]]).unwrap();
            }
        }

        qubo
    }
}

/// Utility functions for annealing integration
pub mod anneal_utils {
    use super::*;

    /// Create feature selection problem
    pub fn create_feature_selection_problem(
        num_features: usize,
        max_features: usize,
    ) -> FeatureSelectionProblem {
        let feature_importance =
            Array1::from_vec((0..num_features).map(|_| fastrand::f64()).collect());

        FeatureSelectionProblem {
            feature_importance,
            penalty_strength: 0.1,
            max_features: Some(max_features),
        }
    }

    /// Create hyperparameter optimization problem
    pub fn create_hyperparameter_problem(
        param_names: Vec<String>,
        param_ranges: Vec<(f64, f64)>,
        bits_per_param: usize,
    ) -> HyperparameterProblem {
        let parameter_encodings = param_names
            .into_iter()
            .zip(param_ranges.into_iter())
            .map(|(name, range)| ParameterEncoding {
                name,
                num_bits: bits_per_param,
                range,
            })
            .collect();

        HyperparameterProblem {
            parameter_encodings,
            cv_function: "accuracy".to_string(),
        }
    }

    /// Create portfolio optimization problem
    pub fn create_portfolio_problem(
        num_assets: usize,
        risk_aversion: f64,
    ) -> PortfolioOptimizationProblem {
        let expected_returns = Array1::from_vec(
            (0..num_assets)
                .map(|_| 0.05 + 0.1 * fastrand::f64())
                .collect(),
        );

        let mut covariance_matrix = Array2::zeros((num_assets, num_assets));
        for i in 0..num_assets {
            for j in 0..num_assets {
                let cov = if i == j {
                    0.01 + 0.02 * fastrand::f64()
                } else {
                    0.005 * fastrand::f64()
                };
                covariance_matrix[[i, j]] = cov;
            }
        }

        PortfolioOptimizationProblem {
            expected_returns,
            covariance_matrix,
            risk_aversion,
            budget: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubo_creation() {
        let mut qubo = QuantumMLQUBO::new(3, "Test QUBO");
        qubo.set_coefficient(0, 0, 1.0).unwrap();
        qubo.set_coefficient(0, 1, -2.0).unwrap();

        assert_eq!(qubo.qubo_matrix[[0, 0]], 1.0);
        assert_eq!(qubo.qubo_matrix[[0, 1]], -2.0);
    }

    #[test]
    fn test_ising_conversion() {
        let mut qubo = QuantumMLQUBO::new(2, "Test");
        qubo.set_coefficient(0, 0, 1.0).unwrap();
        qubo.set_coefficient(1, 1, -1.0).unwrap();
        qubo.set_coefficient(0, 1, 2.0).unwrap();

        let ising = qubo.to_ising();
        assert_eq!(ising.h.len(), 2);
        assert_eq!(ising.j.shape(), [2, 2]);
    }

    #[test]
    fn test_annealer_creation() {
        let annealer = QuantumMLAnnealer::new();
        assert_eq!(annealer.params.num_sweeps, 1000);
    }

    #[test]
    fn test_embedding() {
        let embedding = Embedding::identity(5);
        assert_eq!(embedding.logical_to_physical.len(), 5);
        assert_eq!(embedding.physical_to_logical.len(), 5);
    }

    #[test]
    fn test_feature_selection_problem() {
        let problem = anneal_utils::create_feature_selection_problem(10, 5);
        assert_eq!(problem.feature_importance.len(), 10);
        assert_eq!(problem.max_features, Some(5));
    }
}
