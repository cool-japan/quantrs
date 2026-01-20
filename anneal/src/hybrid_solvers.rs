//! Quantum-Classical Hybrid Solvers
//!
//! This module implements hybrid quantum-classical algorithms that leverage
//! both quantum annealing hardware and classical optimization techniques
//! for solving large-scale optimization problems.

use crate::embedding::Embedding;
use crate::ising::{IsingError, IsingModel, IsingResult, QuboModel};
use crate::partitioning::{Partition, SpectralPartitioner};
use crate::qubo::QuboFormulation;
use crate::simulator::{
    AnnealingParams, AnnealingSolution, ClassicalAnnealingSimulator, QuantumAnnealingSimulator,
};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Rng;
use std::collections::HashMap;

/// Configuration for hybrid quantum-classical solver
#[derive(Debug, Clone)]
pub struct HybridSolverConfig {
    /// Maximum size of subproblems for quantum solver
    pub max_quantum_size: usize,
    /// Number of hybrid iterations
    pub num_iterations: usize,
    /// Use classical solver for small subproblems
    pub classical_threshold: usize,
    /// Overlap between partitions
    pub partition_overlap: usize,
    /// Temperature for combining solutions
    pub combination_temperature: f64,
    /// Use adaptive partitioning
    pub adaptive_partitioning: bool,
    /// Learning rate for solution improvement
    pub learning_rate: f64,
}

impl Default for HybridSolverConfig {
    fn default() -> Self {
        Self {
            max_quantum_size: 100,
            num_iterations: 10,
            classical_threshold: 20,
            partition_overlap: 5,
            combination_temperature: 1.0,
            adaptive_partitioning: true,
            learning_rate: 0.1,
        }
    }
}

/// Results from hybrid quantum-classical solver
#[derive(Debug, Clone)]
pub struct HybridSolverResult {
    /// Best solution found
    pub best_solution: Vec<i8>,
    /// Best energy found
    pub best_energy: f64,
    /// Number of partitions used
    pub num_partitions: usize,
    /// Number of quantum calls
    pub quantum_calls: usize,
    /// Number of classical calls
    pub classical_calls: usize,
    /// Convergence history
    pub energy_history: Vec<f64>,
}

/// Hybrid quantum-classical solver
pub struct HybridQuantumClassicalSolver {
    config: HybridSolverConfig,
    quantum_solver: QuantumAnnealingSimulator,
    classical_solver: ClassicalAnnealingSimulator,
    partitioner: SpectralPartitioner,
}

impl HybridQuantumClassicalSolver {
    /// Create a new hybrid solver
    pub fn new(config: HybridSolverConfig) -> IsingResult<Self> {
        let quantum_params = AnnealingParams::default();
        let classical_params = AnnealingParams::default();

        Ok(Self {
            config,
            quantum_solver: QuantumAnnealingSimulator::new(quantum_params).map_err(|e| {
                IsingError::InvalidValue(format!("Failed to create quantum solver: {e}"))
            })?,
            classical_solver: ClassicalAnnealingSimulator::new(classical_params).map_err(|e| {
                IsingError::InvalidValue(format!("Failed to create classical solver: {e}"))
            })?,
            partitioner: SpectralPartitioner::default(),
        })
    }

    /// Solve an Ising model using hybrid quantum-classical approach
    pub fn solve_ising(&mut self, model: &IsingModel) -> IsingResult<HybridSolverResult> {
        let mut result = HybridSolverResult {
            best_solution: vec![1; model.num_qubits],
            best_energy: f64::INFINITY,
            num_partitions: 0,
            quantum_calls: 0,
            classical_calls: 0,
            energy_history: Vec::new(),
        };

        // Initialize with random solution
        for i in 0..model.num_qubits {
            result.best_solution[i] = if thread_rng().gen::<bool>() { 1 } else { -1 };
        }
        result.best_energy = model.energy(&result.best_solution)?;

        // Main hybrid loop
        for iteration in 0..self.config.num_iterations {
            // Partition the problem
            let partitions = self.partition_problem(model)?;
            result.num_partitions = partitions.len();

            // Solve each partition
            let mut partition_solutions = Vec::new();
            for partition in &partitions {
                let subproblem = self.extract_subproblem(model, partition)?;

                let subsolution = if subproblem.num_qubits <= self.config.classical_threshold {
                    // Use classical solver for small problems
                    result.classical_calls += 1;
                    self.classical_solver.solve(&subproblem).map_err(|e| {
                        IsingError::InvalidValue(format!("Classical solver failed: {e}"))
                    })?
                } else {
                    // Use quantum solver for larger problems
                    result.quantum_calls += 1;
                    self.quantum_solver.solve(&subproblem).map_err(|e| {
                        IsingError::InvalidValue(format!("Quantum solver failed: {e}"))
                    })?
                };

                partition_solutions.push((partition.clone(), subsolution));
            }

            // Combine partition solutions
            let combined_solution =
                self.combine_solutions(&partition_solutions, model.num_qubits)?;
            let combined_energy = model.energy(&combined_solution)?;

            // Local refinement
            let refined_solution = self.local_refinement(model, &combined_solution)?;
            let refined_energy = model.energy(&refined_solution)?;

            // Update best solution
            if refined_energy < result.best_energy {
                result.best_solution = refined_solution;
                result.best_energy = refined_energy;
            }

            result.energy_history.push(result.best_energy);

            // Adaptive partitioning based on solution quality
            if self.config.adaptive_partitioning && iteration < self.config.num_iterations - 1 {
                self.update_partitioning_strategy(&partition_solutions);
            }
        }

        Ok(result)
    }

    /// Solve a QUBO model using hybrid approach
    pub fn solve_qubo(&mut self, model: &QuboModel) -> IsingResult<HybridSolverResult> {
        let (ising, _offset) = model.to_ising();
        let mut result = self.solve_ising(&ising)?;

        // Convert solution from {-1,1} to {0,1}
        for val in &mut result.best_solution {
            *val = (*val + 1) / 2;
        }

        Ok(result)
    }

    /// Partition the problem into smaller subproblems
    fn partition_problem(&self, model: &IsingModel) -> IsingResult<Vec<Partition>> {
        // Create adjacency matrix from couplings
        let n = model.num_qubits;
        let mut edges = Vec::new();

        for coupling in model.couplings() {
            edges.push((coupling.i, coupling.j, coupling.strength));
        }

        // Use spectral partitioning
        let num_partitions = (n as f64 / self.config.max_quantum_size as f64).ceil() as usize;
        let partition = self
            .partitioner
            .partition_graph(n, &edges, num_partitions.max(2))?;

        // Convert single partition into vector of partitions (one per partition group)
        let mut partitions = Vec::new();
        for p in 0..partition.num_partitions {
            let mut part = Partition::new(1);
            for var in partition.get_partition(p) {
                part.assignment.insert(var, 0);
            }
            partitions.push(part);
        }

        Ok(partitions)
    }

    /// Extract subproblem for a partition
    fn extract_subproblem(
        &self,
        model: &IsingModel,
        partition: &Partition,
    ) -> IsingResult<IsingModel> {
        let indices: Vec<usize> = partition.assignment.keys().copied().collect();
        let mut submodel = IsingModel::new(indices.len());

        // Map original indices to subproblem indices
        let index_map: HashMap<usize, usize> = indices
            .iter()
            .enumerate()
            .map(|(new_idx, &old_idx)| (old_idx, new_idx))
            .collect();

        // Copy biases
        for (i, &orig_idx) in indices.iter().enumerate() {
            let bias = model.get_bias(orig_idx).unwrap_or(0.0);
            if bias != 0.0 {
                submodel.set_bias(i, bias)?;
            }
        }

        // Copy couplings within partition
        for coupling in model.couplings() {
            if let (Some(&new_i), Some(&new_j)) =
                (index_map.get(&coupling.i), index_map.get(&coupling.j))
            {
                submodel.set_coupling(new_i, new_j, coupling.strength)?;
            }
        }

        // Add boundary terms as biases (mean field approximation)
        if self.config.partition_overlap > 0 {
            self.add_boundary_terms(&mut submodel, model, partition, &index_map)?;
        }

        Ok(submodel)
    }

    /// Add boundary terms to account for connections outside partition
    fn add_boundary_terms(
        &self,
        submodel: &mut IsingModel,
        original_model: &IsingModel,
        partition: &Partition,
        index_map: &HashMap<usize, usize>,
    ) -> IsingResult<()> {
        let partition_nodes: std::collections::HashSet<usize> =
            partition.assignment.keys().copied().collect();

        // For each node in partition, check connections to nodes outside
        for &node in partition.assignment.keys() {
            if let Some(&sub_idx) = index_map.get(&node) {
                let mut boundary_field = 0.0;

                // Sum contributions from neighbors outside partition
                for coupling in original_model.couplings() {
                    if coupling.i == node && !partition_nodes.contains(&coupling.j) {
                        // Use mean field approximation: assume external spin = 0
                        boundary_field += coupling.strength * 0.0;
                    } else if coupling.j == node && !partition_nodes.contains(&coupling.i) {
                        boundary_field += coupling.strength * 0.0;
                    }
                }

                // Add to existing bias
                let current_bias = submodel.get_bias(sub_idx).unwrap_or(0.0);
                submodel.set_bias(sub_idx, current_bias + boundary_field)?;
            }
        }

        Ok(())
    }

    /// Combine solutions from different partitions
    fn combine_solutions(
        &self,
        partition_solutions: &[(Partition, AnnealingSolution)],
        num_qubits: usize,
    ) -> IsingResult<Vec<i8>> {
        let mut combined = vec![0i8; num_qubits];
        let mut vote_counts = vec![0i32; num_qubits];

        // Collect votes from each partition
        for (partition, result) in partition_solutions {
            let indices: Vec<usize> = partition.assignment.keys().copied().collect();

            for (sub_idx, &orig_idx) in indices.iter().enumerate() {
                if sub_idx < result.best_spins.len() {
                    combined[orig_idx] += result.best_spins[sub_idx];
                    vote_counts[orig_idx] += 1;
                }
            }
        }

        // Resolve by majority vote
        for i in 0..num_qubits {
            if vote_counts[i] > 0 {
                combined[i] = if combined[i] >= 0 { 1 } else { -1 };
            } else {
                // No vote, random assignment
                combined[i] = if thread_rng().gen::<bool>() { 1 } else { -1 };
            }
        }

        Ok(combined)
    }

    /// Local refinement using classical optimization
    fn local_refinement(&self, model: &IsingModel, solution: &[i8]) -> IsingResult<Vec<i8>> {
        let mut refined = solution.to_vec();
        let mut current_energy = model.energy(&refined)?;
        let mut improved = true;

        // Simple local search: single spin flips
        while improved {
            improved = false;

            for i in 0..refined.len() {
                // Try flipping spin i
                refined[i] *= -1;
                let new_energy = model.energy(&refined)?;

                if new_energy < current_energy {
                    current_energy = new_energy;
                    improved = true;
                } else {
                    // Revert flip
                    refined[i] *= -1;
                }
            }
        }

        Ok(refined)
    }

    /// Update partitioning strategy based on solution quality
    fn update_partitioning_strategy(
        &mut self,
        partition_solutions: &[(Partition, AnnealingSolution)],
    ) {
        // Analyze solution quality per partition
        let mut partition_qualities = Vec::new();

        for (partition, result) in partition_solutions {
            let quality = if result.best_energy.is_finite() {
                1.0 / (1.0 + result.best_energy.abs())
            } else {
                0.0
            };
            partition_qualities.push(quality);
        }

        // Adjust partition sizes based on quality
        // (In practice, this would modify the partitioner's parameters)
        let avg_quality =
            partition_qualities.iter().sum::<f64>() / partition_qualities.len() as f64;

        if avg_quality < 0.5 {
            // Poor quality, use smaller partitions
            self.config.max_quantum_size = (self.config.max_quantum_size as f64 * 0.9) as usize;
        } else if avg_quality > 0.8 {
            // Good quality, can use larger partitions
            self.config.max_quantum_size = (self.config.max_quantum_size as f64 * 1.1) as usize;
        }
    }
}

/// Variational Quantum-Classical Solver
/// Uses a variational approach similar to VQE but for optimization problems
pub struct VariationalHybridSolver {
    /// Number of variational parameters
    num_parameters: usize,
    /// Current parameter values
    parameters: Vec<f64>,
    /// Learning rate
    learning_rate: f64,
    /// Maximum iterations
    max_iterations: usize,
}

impl VariationalHybridSolver {
    /// Create a new variational hybrid solver
    #[must_use]
    pub fn new(num_parameters: usize, learning_rate: f64, max_iterations: usize) -> Self {
        Self {
            num_parameters,
            parameters: vec![0.5; num_parameters],
            learning_rate,
            max_iterations,
        }
    }

    /// Optimize parameters using gradient descent
    pub fn optimize(&mut self, model: &IsingModel) -> IsingResult<Vec<i8>> {
        for _ in 0..self.max_iterations {
            // Compute gradients
            let gradients = self.compute_gradients(model)?;

            // Update parameters
            for i in 0..self.num_parameters {
                self.parameters[i] -= self.learning_rate * gradients[i];
                // Keep parameters in valid range
                self.parameters[i] = self.parameters[i].max(0.0).min(1.0);
            }
        }

        // Generate final solution using optimized parameters
        self.generate_solution(model)
    }

    /// Compute gradients with respect to parameters
    fn compute_gradients(&self, model: &IsingModel) -> IsingResult<Vec<f64>> {
        let mut gradients = vec![0.0; self.num_parameters];
        let epsilon = 0.01;

        for i in 0..self.num_parameters {
            // Finite difference approximation
            let mut params_plus = self.parameters.clone();
            params_plus[i] += epsilon;
            let energy_plus = self.evaluate_parameters(model, &params_plus)?;

            let mut params_minus = self.parameters.clone();
            params_minus[i] -= epsilon;
            let energy_minus = self.evaluate_parameters(model, &params_minus)?;

            gradients[i] = (energy_plus - energy_minus) / (2.0 * epsilon);
        }

        Ok(gradients)
    }

    /// Evaluate energy for given parameters
    fn evaluate_parameters(&self, model: &IsingModel, parameters: &[f64]) -> IsingResult<f64> {
        // Generate solution based on parameters
        let solution = self.generate_solution_with_params(model, parameters)?;
        model.energy(&solution)
    }

    /// Generate solution based on current parameters
    fn generate_solution(&self, model: &IsingModel) -> IsingResult<Vec<i8>> {
        self.generate_solution_with_params(model, &self.parameters)
    }

    /// Generate solution with specific parameters
    fn generate_solution_with_params(
        &self,
        model: &IsingModel,
        parameters: &[f64],
    ) -> IsingResult<Vec<i8>> {
        let n = model.num_qubits;
        let mut solution = vec![1i8; n];

        // Use parameters to bias solution generation
        // Simple approach: use parameters as probabilities for spin values
        for i in 0..n {
            let param_idx = i % parameters.len();
            let prob = parameters[param_idx];
            solution[i] = if thread_rng().gen::<f64>() < prob {
                1
            } else {
                -1
            };
        }

        Ok(solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_solver_creation() {
        let config = HybridSolverConfig::default();
        let solver =
            HybridQuantumClassicalSolver::new(config).expect("Failed to create hybrid solver");
        assert!(solver.config.max_quantum_size > 0);
    }

    #[test]
    fn test_small_problem_solving() {
        let mut model = IsingModel::new(4);
        model.set_bias(0, 1.0).expect("Failed to set bias");
        model
            .set_coupling(0, 1, -2.0)
            .expect("Failed to set coupling");
        model
            .set_coupling(1, 2, -1.0)
            .expect("Failed to set coupling");

        let config = HybridSolverConfig {
            max_quantum_size: 2,
            ..Default::default()
        };

        let mut solver =
            HybridQuantumClassicalSolver::new(config).expect("Failed to create hybrid solver");
        let result = solver.solve_ising(&model).expect("Failed to solve Ising");

        assert_eq!(result.best_solution.len(), 4);
        assert!(result.best_energy.is_finite());
    }

    #[test]
    fn test_variational_solver() {
        let model = IsingModel::new(3);
        let mut solver = VariationalHybridSolver::new(3, 0.1, 10);
        let solution = solver.optimize(&model).expect("Failed to optimize");
        assert_eq!(solution.len(), 3);
    }
}
