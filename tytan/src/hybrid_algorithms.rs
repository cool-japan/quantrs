//! Hybrid quantum-classical algorithms for optimization.
//!
//! This module provides implementations of hybrid algorithms that combine
//! quantum and classical computing approaches for solving optimization problems.

use crate::sampler::{Sampler, SampleResult, SamplerError, SamplerResult};
#[cfg(feature = "dwave")]
use crate::compile::CompiledModel;
use ndarray::{Array, Array1, Array2, IxDyn};
use std::collections::HashMap;
use std::f64::consts::PI;
use rand::thread_rng;

#[cfg(feature = "scirs")]
use crate::scirs_stub::{
    scirs2_optimization::gradient::LBFGS,
    scirs2_optimization::{Optimizer, ObjectiveFunction, Bounds},
};

/// Variational Quantum Eigensolver (VQE) interface
pub struct VQE {
    /// Number of qubits
    n_qubits: usize,
    /// Ansatz type
    ansatz: AnsatzType,
    /// Classical optimizer
    optimizer: ClassicalOptimizer,
    /// Number of optimization iterations
    max_iterations: usize,
    /// Convergence threshold
    convergence_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum AnsatzType {
    /// Hardware-efficient ansatz
    HardwareEfficient {
        depth: usize,
        entangling_gate: String,
    },
    /// Unitary Coupled Cluster (UCC) ansatz
    UCC {
        excitation_order: usize,
    },
    /// Problem-specific ansatz
    Custom {
        name: String,
        parameters: HashMap<String, f64>,
    },
}

#[derive(Debug, Clone)]
pub enum ClassicalOptimizer {
    /// COBYLA optimizer
    COBYLA {
        rhobeg: f64,
        rhoend: f64,
    },
    /// L-BFGS optimizer
    LBFGS {
        max_iterations: usize,
        tolerance: f64,
    },
    /// SPSA optimizer
    SPSA {
        a: f64,
        c: f64,
        alpha: f64,
        gamma: f64,
    },
    /// Gradient descent
    GradientDescent {
        learning_rate: f64,
        momentum: f64,
    },
}

impl VQE {
    /// Create new VQE solver
    pub fn new(
        n_qubits: usize,
        ansatz: AnsatzType,
        optimizer: ClassicalOptimizer,
    ) -> Self {
        Self {
            n_qubits,
            ansatz,
            optimizer,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
        }
    }
    
    /// Set maximum iterations
    pub fn with_max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }
    
    /// Set convergence threshold
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }
    
    /// Solve optimization problem
    pub fn solve(&mut self, hamiltonian: &Hamiltonian) -> Result<VQEResult, String> {
        // Initialize parameters
        let num_params = self.get_num_parameters();
        let mut params = self.initialize_parameters(num_params);
        
        let mut energy_history = Vec::new();
        let mut param_history = Vec::new();
        let mut converged = false;
        
        // Optimization loop
        for iteration in 0..self.max_iterations {
            // Evaluate energy
            let energy = self.evaluate_energy(&params, hamiltonian)?;
            energy_history.push(energy);
            param_history.push(params.clone());
            
            // Check convergence
            if iteration > 0 {
                let energy_change = (energy_history[iteration] - energy_history[iteration-1]).abs();
                if energy_change < self.convergence_threshold {
                    converged = true;
                    break;
                }
            }
            
            // Update parameters
            params = self.update_parameters(params, hamiltonian)?;
        }
        
        let iterations = energy_history.len();
        let ground_state_energy = *energy_history.last().unwrap();
        
        Ok(VQEResult {
            optimal_parameters: params,
            ground_state_energy,
            energy_history,
            parameter_history: param_history,
            converged,
            iterations,
        })
    }
    
    /// Get number of parameters for ansatz
    fn get_num_parameters(&self) -> usize {
        match &self.ansatz {
            AnsatzType::HardwareEfficient { depth, .. } => {
                // Single-qubit rotations + entangling gates
                self.n_qubits * 3 * depth + (self.n_qubits - 1) * depth
            }
            AnsatzType::UCC { excitation_order } => {
                // Simplified: depends on excitation order
                self.n_qubits * excitation_order
            }
            AnsatzType::Custom { parameters, .. } => {
                parameters.len()
            }
        }
    }
    
    /// Initialize parameters
    fn initialize_parameters(&self, num_params: usize) -> Vec<f64> {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        (0..num_params)
            .map(|_| rng.gen_range(-PI..PI))
            .collect()
    }
    
    /// Evaluate energy for given parameters
    fn evaluate_energy(
        &self,
        params: &[f64],
        hamiltonian: &Hamiltonian,
    ) -> Result<f64, String> {
        // This would involve:
        // 1. Prepare quantum state with ansatz
        // 2. Measure expectation value of Hamiltonian
        // 3. Return energy
        
        // Placeholder implementation
        let energy = hamiltonian.evaluate_classical(params)?;
        Ok(energy)
    }
    
    /// Update parameters using classical optimizer
    fn update_parameters(
        &self,
        current_params: Vec<f64>,
        hamiltonian: &Hamiltonian,
    ) -> Result<Vec<f64>, String> {
        match &self.optimizer {
            ClassicalOptimizer::GradientDescent { learning_rate, momentum } => {
                // Compute gradient
                let gradient = self.compute_gradient(&current_params, hamiltonian)?;
                
                // Update with momentum
                let mut new_params = current_params.clone();
                for i in 0..new_params.len() {
                    new_params[i] -= learning_rate * gradient[i];
                }
                
                Ok(new_params)
            }
            ClassicalOptimizer::SPSA { a, c, alpha, gamma } => {
                // SPSA update
                self.spsa_update(current_params, hamiltonian, *a, *c, *alpha, *gamma)
            }
            _ => {
                // Other optimizers would be implemented similarly
                Ok(current_params)
            }
        }
    }
    
    /// Compute gradient using parameter shift rule
    fn compute_gradient(
        &self,
        params: &[f64],
        hamiltonian: &Hamiltonian,
    ) -> Result<Vec<f64>, String> {
        let mut gradient = vec![0.0; params.len()];
        let shift = PI / 2.0;
        
        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            let mut params_minus = params.to_vec();
            
            params_plus[i] += shift;
            params_minus[i] -= shift;
            
            let energy_plus = self.evaluate_energy(&params_plus, hamiltonian)?;
            let energy_minus = self.evaluate_energy(&params_minus, hamiltonian)?;
            
            gradient[i] = (energy_plus - energy_minus) / (2.0 * shift);
        }
        
        Ok(gradient)
    }
    
    /// SPSA parameter update
    fn spsa_update(
        &self,
        params: Vec<f64>,
        hamiltonian: &Hamiltonian,
        a: f64,
        c: f64,
        _alpha: f64,
        _gamma: f64,
    ) -> Result<Vec<f64>, String> {
        use rand::prelude::*;
        let mut rng = thread_rng();
        
        // Generate random perturbation
        let delta: Vec<f64> = (0..params.len())
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 })
            .collect();
        
        // Evaluate at perturbed points
        let mut params_plus = params.clone();
        let mut params_minus = params.clone();
        
        for i in 0..params.len() {
            params_plus[i] += c * delta[i];
            params_minus[i] -= c * delta[i];
        }
        
        let energy_plus = self.evaluate_energy(&params_plus, hamiltonian)?;
        let energy_minus = self.evaluate_energy(&params_minus, hamiltonian)?;
        
        // Update parameters
        let mut new_params = params.clone();
        for i in 0..params.len() {
            new_params[i] -= a * (energy_plus - energy_minus) / (2.0 * c * delta[i]);
        }
        
        Ok(new_params)
    }
}

/// VQE result
#[derive(Debug, Clone)]
pub struct VQEResult {
    pub optimal_parameters: Vec<f64>,
    pub ground_state_energy: f64,
    pub energy_history: Vec<f64>,
    pub parameter_history: Vec<Vec<f64>>,
    pub converged: bool,
    pub iterations: usize,
}

/// Hamiltonian representation
#[derive(Debug, Clone)]
pub struct Hamiltonian {
    /// Pauli terms
    pub terms: Vec<PauliTerm>,
}

#[derive(Debug, Clone)]
pub struct PauliTerm {
    /// Coefficient
    pub coefficient: f64,
    /// Pauli string (I, X, Y, Z for each qubit)
    pub pauli_string: Vec<char>,
}

impl Hamiltonian {
    /// Create from QUBO
    pub fn from_qubo(qubo: &Array2<f64>) -> Self {
        let n = qubo.shape()[0];
        let mut terms = Vec::new();
        
        // Convert QUBO to Ising model
        // H = sum_i h_i Z_i + sum_{i<j} J_{ij} Z_i Z_j
        
        for i in 0..n {
            // Linear terms
            let h_i = qubo[[i, i]];
            if h_i.abs() > 1e-10 {
                let mut pauli_string = vec!['I'; n];
                pauli_string[i] = 'Z';
                terms.push(PauliTerm {
                    coefficient: h_i,
                    pauli_string,
                });
            }
            
            // Quadratic terms
            for j in i+1..n {
                let j_ij = qubo[[i, j]];
                if j_ij.abs() > 1e-10 {
                    let mut pauli_string = vec!['I'; n];
                    pauli_string[i] = 'Z';
                    pauli_string[j] = 'Z';
                    terms.push(PauliTerm {
                        coefficient: j_ij,
                        pauli_string,
                    });
                }
            }
        }
        
        Self { terms }
    }
    
    /// Evaluate classically (placeholder)
    fn evaluate_classical(&self, _params: &[f64]) -> Result<f64, String> {
        // This would evaluate the Hamiltonian expectation value
        // For now, return a simple function of parameters
        Ok(_params.iter().map(|p| p.sin()).sum())
    }
}

/// Quantum Approximate Optimization Algorithm (QAOA)
pub struct QAOA {
    /// Number of QAOA layers (p)
    p: usize,
    /// Classical optimizer
    optimizer: ClassicalOptimizer,
    /// Initial state preparation
    initial_state: InitialState,
}

#[derive(Debug, Clone)]
pub enum InitialState {
    /// Equal superposition
    EqualSuperposition,
    /// Custom state
    Custom(Vec<f64>),
    /// Warm start from classical solution
    WarmStart(Vec<bool>),
}

impl QAOA {
    /// Create new QAOA solver
    pub fn new(p: usize, optimizer: ClassicalOptimizer) -> Self {
        Self {
            p,
            optimizer,
            initial_state: InitialState::EqualSuperposition,
        }
    }
    
    /// Set initial state
    pub fn with_initial_state(mut self, state: InitialState) -> Self {
        self.initial_state = state;
        self
    }
    
    /// Solve QUBO problem
    pub fn solve_qubo(&mut self, qubo: &Array2<f64>) -> Result<QAOAResult, String> {
        let hamiltonian = Hamiltonian::from_qubo(qubo);
        
        // Initialize parameters (beta, gamma for each layer)
        let mut betas = vec![0.0; self.p];
        let mut gammas = vec![0.0; self.p];
        
        // Optimization loop
        let mut energy_history = Vec::new();
        let max_iterations = 100;
        
        for iteration in 0..max_iterations {
            // Evaluate energy
            let energy = self.evaluate_qaoa_energy(&betas, &gammas, &hamiltonian)?;
            energy_history.push(energy);
            
            // Update parameters
            let (new_betas, new_gammas) = self.update_qaoa_parameters(
                &betas,
                &gammas,
                &hamiltonian,
            )?;
            
            betas = new_betas;
            gammas = new_gammas;
            
            // Check convergence
            if iteration > 0 {
                let improvement = energy_history[iteration-1] - energy;
                if improvement < 1e-6 && improvement > 0.0 {
                    break;
                }
            }
        }
        
        // Sample final state
        let samples = self.sample_qaoa_state(&betas, &gammas, &hamiltonian, 1000)?;
        
        let best_energy = *energy_history.last().unwrap();
        
        Ok(QAOAResult {
            optimal_betas: betas,
            optimal_gammas: gammas,
            energy_history,
            samples,
            best_energy,
        })
    }
    
    /// Evaluate QAOA energy
    fn evaluate_qaoa_energy(
        &self,
        betas: &[f64],
        gammas: &[f64],
        hamiltonian: &Hamiltonian,
    ) -> Result<f64, String> {
        // This would:
        // 1. Prepare initial state
        // 2. Apply QAOA circuit with given parameters
        // 3. Measure expectation value
        
        // Placeholder
        let param_sum: f64 = betas.iter().sum::<f64>() + gammas.iter().sum::<f64>();
        Ok(param_sum.sin() * 10.0)
    }
    
    /// Update QAOA parameters
    fn update_qaoa_parameters(
        &self,
        betas: &[f64],
        gammas: &[f64],
        hamiltonian: &Hamiltonian,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        // Combine into single parameter vector
        let mut params = Vec::new();
        params.extend_from_slice(betas);
        params.extend_from_slice(gammas);
        
        // Update using optimizer
        match &self.optimizer {
            ClassicalOptimizer::GradientDescent { learning_rate, .. } => {
                // Compute gradients
                let gradient = self.compute_qaoa_gradient(&params, hamiltonian)?;
                
                // Update
                for i in 0..params.len() {
                    params[i] -= learning_rate * gradient[i];
                }
            }
            _ => {
                // Other optimizers
            }
        }
        
        // Split back into betas and gammas
        let new_betas = params[..self.p].to_vec();
        let new_gammas = params[self.p..].to_vec();
        
        Ok((new_betas, new_gammas))
    }
    
    /// Compute QAOA gradient
    fn compute_qaoa_gradient(
        &self,
        params: &[f64],
        _hamiltonian: &Hamiltonian,
    ) -> Result<Vec<f64>, String> {
        // Parameter shift rule for QAOA
        let mut gradient = vec![0.0; params.len()];
        
        // Placeholder gradient
        for i in 0..params.len() {
            gradient[i] = -params[i].cos();
        }
        
        Ok(gradient)
    }
    
    /// Sample from QAOA state
    fn sample_qaoa_state(
        &self,
        _betas: &[f64],
        _gammas: &[f64],
        _hamiltonian: &Hamiltonian,
        num_samples: usize,
    ) -> Result<Vec<Vec<bool>>, String> {
        // This would sample from the prepared quantum state
        // Placeholder: return random samples
        use rand::prelude::*;
        let mut rng = thread_rng();
        let n_qubits = 10; // Would get from hamiltonian
        
        let samples = (0..num_samples)
            .map(|_| {
                (0..n_qubits)
                    .map(|_| rng.gen_bool(0.5))
                    .collect()
            })
            .collect();
        
        Ok(samples)
    }
}

/// QAOA result
#[derive(Debug, Clone)]
pub struct QAOAResult {
    pub optimal_betas: Vec<f64>,
    pub optimal_gammas: Vec<f64>,
    pub energy_history: Vec<f64>,
    pub samples: Vec<Vec<bool>>,
    pub best_energy: f64,
}

/// Warm start strategies for hybrid algorithms
pub struct WarmStartStrategy {
    /// Classical pre-solver
    pre_solver: Box<dyn Sampler>,
    /// Number of classical iterations
    classical_iterations: usize,
    /// How to use classical solution
    usage: WarmStartUsage,
}

#[derive(Debug, Clone)]
pub enum WarmStartUsage {
    /// Use as initial state
    InitialState,
    /// Use to initialize parameters
    ParameterInitialization,
    /// Use as reference for relative optimization
    ReferencePoint,
    /// Hybrid: alternate between quantum and classical
    Alternating { switch_threshold: f64 },
}

impl WarmStartStrategy {
    /// Create new warm start strategy
    pub fn new(
        pre_solver: Box<dyn Sampler>,
        classical_iterations: usize,
        usage: WarmStartUsage,
    ) -> Self {
        Self {
            pre_solver,
            classical_iterations,
            usage,
        }
    }
    
    /// Generate warm start
    #[cfg(feature = "dwave")]
    pub fn generate_warm_start(
        &mut self,
        problem: &CompiledModel,
    ) -> Result<WarmStartResult, String> {
        // Run classical pre-solver
        let qubo = problem.to_qubo();
        let classical_results = self.pre_solver.run_qubo(&qubo, self.classical_iterations)?;
        
        // Extract best solution
        let best_solution = classical_results.first()
            .ok_or("No classical solution found")?;
        
        // Convert to initial state or parameters based on usage
        match &self.usage {
            WarmStartUsage::InitialState => {
                // Convert solution to quantum state
                let state_vector = self.solution_to_state_vector(&best_solution.assignments)?;
                Ok(WarmStartResult::StateVector(state_vector))
            }
            WarmStartUsage::ParameterInitialization => {
                // Use solution to initialize variational parameters
                let params = self.solution_to_parameters(&best_solution.assignments)?;
                Ok(WarmStartResult::Parameters(params))
            }
            _ => {
                // Other strategies
                Ok(WarmStartResult::Solution(best_solution.clone()))
            }
        }
    }
    
    /// Convert solution to state vector
    fn solution_to_state_vector(
        &self,
        assignments: &HashMap<String, bool>,
    ) -> Result<Vec<f64>, String> {
        let n = assignments.len();
        let dim = 1 << n; // 2^n
        let mut state = vec![0.0; dim];
        
        // Convert assignments to binary index
        let mut index = 0;
        let mut vars: Vec<_> = assignments.keys().collect();
        vars.sort();
        
        for (i, var) in vars.iter().enumerate() {
            if assignments[var.as_str()] {
                index |= 1 << i;
            }
        }
        
        state[index] = 1.0;
        Ok(state)
    }
    
    /// Convert solution to variational parameters
    fn solution_to_parameters(
        &self,
        assignments: &HashMap<String, bool>,
    ) -> Result<Vec<f64>, String> {
        // Heuristic: use solution to bias parameters
        let params: Vec<f64> = assignments.values()
            .map(|&b| if b { PI / 4.0 } else { -PI / 4.0 })
            .collect();
        
        Ok(params)
    }
}

#[derive(Debug, Clone)]
pub enum WarmStartResult {
    StateVector(Vec<f64>),
    Parameters(Vec<f64>),
    Solution(SampleResult),
}

/// Iterative refinement for hybrid algorithms
pub struct IterativeRefinement {
    /// Quantum solver
    quantum_solver: Box<dyn Sampler>,
    /// Classical post-processor
    classical_processor: ClassicalProcessor,
    /// Number of refinement iterations
    max_refinements: usize,
    /// Refinement threshold
    improvement_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum ClassicalProcessor {
    /// Local search around quantum solution
    LocalSearch { neighborhood_size: usize },
    /// Simulated annealing refinement
    SimulatedAnnealing { initial_temp: f64, cooling_rate: f64 },
    /// Tabu search
    TabuSearch { tabu_tenure: usize },
    /// Variable neighborhood search
    VariableNeighborhood { k_max: usize },
}

impl IterativeRefinement {
    /// Create new iterative refinement
    pub fn new(
        quantum_solver: Box<dyn Sampler>,
        classical_processor: ClassicalProcessor,
        max_refinements: usize,
    ) -> Self {
        Self {
            quantum_solver,
            classical_processor,
            max_refinements,
            improvement_threshold: 1e-6,
        }
    }
    
    /// Refine solution iteratively
    #[cfg(feature = "dwave")]
    pub fn refine(
        &mut self,
        problem: &CompiledModel,
        initial_shots: usize,
    ) -> Result<RefinementResult, String> {
        let qubo = problem.to_qubo();
        let mut history = Vec::new();
        let mut best_solution = None;
        let mut best_energy = f64::INFINITY;
        
        for iteration in 0..self.max_refinements {
            // Run quantum solver
            let quantum_results = self.quantum_solver.run_qubo(&qubo.clone(), initial_shots)?;
            
            // Apply classical refinement
            let refined_results = self.apply_classical_refinement(
                &quantum_results,
                &qubo.0,
                &qubo.1,
            )?;
            
            // Track best solution
            for result in &refined_results {
                if result.energy < best_energy {
                    best_energy = result.energy;
                    best_solution = Some(result.clone());
                }
            }
            
            history.push(IterationResult {
                quantum_energy: quantum_results.first()
                    .map(|r| r.energy)
                    .unwrap_or(f64::INFINITY),
                refined_energy: refined_results.first()
                    .map(|r| r.energy)
                    .unwrap_or(f64::INFINITY),
                improvement: 0.0, // Calculate actual improvement
            });
            
            // Check convergence
            if iteration > 0 {
                let improvement = history[iteration-1].refined_energy - history[iteration].refined_energy;
                if improvement < self.improvement_threshold {
                    break;
                }
            }
        }
        
        Ok(RefinementResult {
            best_solution: best_solution.ok_or("No solution found")?,
            refinement_history: history,
            total_iterations: history.len(),
        })
    }
    
    /// Apply classical refinement
    fn apply_classical_refinement(
        &self,
        quantum_results: &[SampleResult],
        qubo_matrix: &Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<Vec<SampleResult>, String> {
        match &self.classical_processor {
            ClassicalProcessor::LocalSearch { neighborhood_size } => {
                self.local_search_refinement(
                    quantum_results,
                    qubo_matrix,
                    var_map,
                    *neighborhood_size,
                )
            }
            _ => {
                // Other processors would be implemented
                Ok(quantum_results.to_vec())
            }
        }
    }
    
    /// Local search refinement
    fn local_search_refinement(
        &self,
        results: &[SampleResult],
        qubo_matrix: &Array2<f64>,
        var_map: &HashMap<String, usize>,
        neighborhood_size: usize,
    ) -> Result<Vec<SampleResult>, String> {
        let mut refined_results = Vec::new();
        
        for result in results.iter().take(10) {
            // Convert to binary vector
            let mut state: Vec<bool> = var_map.iter()
                .map(|(var, _)| result.assignments[var])
                .collect();
            
            let mut best_state = state.clone();
            let mut best_energy = result.energy;
            
            // Search neighborhood
            for _ in 0..neighborhood_size {
                // Flip random bits
                let mut neighbor = state.clone();
                use rand::prelude::*;
                let mut rng = thread_rng();
                let flip_idx = rng.gen_range(0..state.len());
                neighbor[flip_idx] = !neighbor[flip_idx];
                
                // Evaluate energy
                let energy = self.evaluate_qubo_energy(&neighbor, qubo_matrix);
                
                if energy < best_energy {
                    best_energy = energy;
                    best_state = neighbor.clone();
                }
                
                state = best_state.clone();
            }
            
            // Convert back to assignments
            let assignments: HashMap<String, bool> = var_map.iter()
                .zip(best_state.iter())
                .map(|((var, _), &val)| (var.clone(), val))
                .collect();
            
            refined_results.push(SampleResult {
                assignments,
                energy: best_energy,
                occurrences: result.occurrences,
            });
        }
        
        refined_results.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
        Ok(refined_results)
    }
    
    /// Evaluate QUBO energy
    fn evaluate_qubo_energy(&self, state: &[bool], matrix: &Array2<f64>) -> f64 {
        let n = state.len();
        let mut energy = 0.0;
        
        for i in 0..n {
            if state[i] {
                energy += matrix[[i, i]];
                for j in i+1..n {
                    if state[j] {
                        energy += matrix[[i, j]];
                    }
                }
            }
        }
        
        energy
    }
}

#[derive(Debug, Clone)]
pub struct RefinementResult {
    pub best_solution: SampleResult,
    pub refinement_history: Vec<IterationResult>,
    pub total_iterations: usize,
}

#[derive(Debug, Clone)]
pub struct IterationResult {
    pub quantum_energy: f64,
    pub refined_energy: f64,
    pub improvement: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vqe_initialization() {
        let vqe = VQE::new(
            4,
            AnsatzType::HardwareEfficient { depth: 2, entangling_gate: "CZ".to_string() },
            ClassicalOptimizer::GradientDescent { learning_rate: 0.1, momentum: 0.9 },
        );
        
        assert_eq!(vqe.n_qubits, 4);
        assert_eq!(vqe.get_num_parameters(), 4 * 3 * 2 + 3 * 2); // 30 parameters
    }
    
    #[test]
    fn test_hamiltonian_from_qubo() {
        let mut qubo = Array2::zeros((3, 3));
        qubo[[0, 0]] = 1.0;
        qubo[[0, 1]] = -2.0;
        qubo[[1, 0]] = -2.0;
        qubo[[1, 1]] = 1.0;
        
        let hamiltonian = Hamiltonian::from_qubo(&qubo);
        assert_eq!(hamiltonian.terms.len(), 3); // 2 linear + 1 quadratic
    }
    
    #[test]
    fn test_qaoa_initialization() {
        let qaoa = QAOA::new(
            3,
            ClassicalOptimizer::SPSA { a: 0.1, c: 0.1, alpha: 0.602, gamma: 0.101 },
        );
        
        assert_eq!(qaoa.p, 3);
    }
}