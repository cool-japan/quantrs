//! Large-scale QUBO decomposition for efficient solving
//!
//! This module provides methods for decomposing large QUBO problems into
//! smaller sub-problems that can be solved more efficiently using various
//! decomposition strategies.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::ising::{IsingError, IsingModel, QuboModel};
use crate::partitioning::{Partition, SpectralPartitioner};
use crate::simulator::{AnnealingParams, AnnealingSolution, QuantumAnnealingSimulator};

/// Errors that can occur during QUBO decomposition
#[derive(Error, Debug)]
pub enum DecompositionError {
    /// Ising model error
    #[error("Ising error: {0}")]
    IsingError(#[from] IsingError),

    /// Invalid decomposition parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Decomposition failed
    #[error("Decomposition failed: {0}")]
    DecompositionFailed(String),

    /// Solving failed
    #[error("Solving failed: {0}")]
    SolvingFailed(String),

    /// Reconstruction failed
    #[error("Reconstruction failed: {0}")]
    ReconstructionFailed(String),
}

/// Result type for decomposition operations
pub type DecompositionResult<T> = Result<T, DecompositionError>;

/// Decomposition strategy for QUBO problems
#[derive(Debug, Clone)]
pub enum DecompositionStrategy {
    /// Spectral decomposition using eigenvector-based partitioning
    Spectral {
        num_partitions: usize,
        overlap_size: usize,
    },

    /// Block decomposition based on problem structure
    Block {
        block_size: usize,
        overlap_size: usize,
    },

    /// Hierarchical decomposition using recursive bisection
    Hierarchical {
        min_partition_size: usize,
        max_depth: usize,
        overlap_size: usize,
    },

    /// Variable clustering based on interaction strength
    Clustering {
        num_clusters: usize,
        strength_threshold: f64,
        overlap_size: usize,
    },
}

/// Configuration for QUBO decomposition
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Decomposition strategy
    pub strategy: DecompositionStrategy,

    /// Maximum sub-problem size
    pub max_subproblem_size: usize,

    /// Minimum sub-problem size
    pub min_subproblem_size: usize,

    /// Number of iterations for iterative refinement
    pub refinement_iterations: usize,

    /// Convergence tolerance for iterative methods
    pub convergence_tolerance: f64,

    /// Solver parameters for sub-problems
    pub solver_params: AnnealingParams,

    /// Enable parallel solving of sub-problems
    pub parallel_solving: bool,

    /// Maximum runtime for decomposition
    pub timeout: Option<Duration>,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            strategy: DecompositionStrategy::Spectral {
                num_partitions: 4,
                overlap_size: 2,
            },
            max_subproblem_size: 100,
            min_subproblem_size: 10,
            refinement_iterations: 5,
            convergence_tolerance: 1e-6,
            solver_params: AnnealingParams {
                num_sweeps: 1000,
                num_repetitions: 5,
                ..Default::default()
            },
            parallel_solving: true,
            timeout: Some(Duration::from_secs(3600)),
        }
    }
}

/// Sub-problem extracted from the original QUBO
#[derive(Debug, Clone)]
pub struct SubProblem {
    /// Variables in this sub-problem
    pub variables: Vec<usize>,

    /// Mapping from sub-problem indices to original indices
    pub variable_mapping: HashMap<usize, usize>,

    /// QUBO model for this sub-problem
    pub qubo: QuboModel,

    /// Fixed variables from other sub-problems
    pub fixed_variables: HashMap<usize, bool>,

    /// Sub-problem identifier
    pub id: usize,
}

/// Solution for a sub-problem
#[derive(Debug, Clone)]
pub struct SubSolution {
    /// Sub-problem identifier
    pub subproblem_id: usize,

    /// Solution values for variables in this sub-problem
    pub values: HashMap<usize, bool>,

    /// Objective value achieved
    pub objective_value: f64,

    /// Solver statistics
    pub solver_info: String,
}

/// Complete decomposition solution
#[derive(Debug, Clone)]
pub struct DecomposedSolution {
    /// Variable assignments for the complete problem
    pub variable_values: HashMap<usize, bool>,

    /// Objective value of the complete solution
    pub objective_value: f64,

    /// Sub-problem solutions
    pub sub_solutions: Vec<SubSolution>,

    /// Decomposition statistics
    pub stats: DecompositionStats,
}

/// Statistics about the decomposition process
#[derive(Debug, Clone)]
pub struct DecompositionStats {
    /// Number of sub-problems created
    pub num_subproblems: usize,

    /// Average sub-problem size
    pub avg_subproblem_size: f64,

    /// Maximum sub-problem size
    pub max_subproblem_size: usize,

    /// Minimum sub-problem size
    pub min_subproblem_size: usize,

    /// Total decomposition time
    pub decomposition_time: Duration,

    /// Total solving time
    pub solving_time: Duration,

    /// Number of refinement iterations performed
    pub refinement_iterations: usize,

    /// Convergence achieved
    pub converged: bool,
}

/// QUBO decomposition solver
pub struct QuboDecomposer {
    /// Configuration
    config: DecompositionConfig,
}

impl QuboDecomposer {
    /// Create a new QUBO decomposer
    #[must_use]
    pub const fn new(config: DecompositionConfig) -> Self {
        Self { config }
    }

    /// Solve a large QUBO problem using decomposition
    pub fn solve(&self, qubo: &QuboModel) -> DecompositionResult<DecomposedSolution> {
        let total_start = Instant::now();

        // Step 1: Decompose the problem
        let decompose_start = Instant::now();
        let sub_problems = self.decompose_problem(qubo)?;
        let decomposition_time = decompose_start.elapsed();

        println!(
            "Decomposed problem into {} sub-problems",
            sub_problems.len()
        );

        // Step 2: Solve sub-problems
        let solve_start = Instant::now();
        let mut sub_solutions = self.solve_subproblems(&sub_problems)?;
        let mut solving_time = solve_start.elapsed();

        // Step 3: Iterative refinement
        let mut converged = false;
        let mut refinement_iterations = 0;

        for iteration in 0..self.config.refinement_iterations {
            let refine_start = Instant::now();

            let (new_solutions, improvement) =
                self.refine_solutions(qubo, &sub_problems, &sub_solutions)?;

            solving_time += refine_start.elapsed();
            refinement_iterations += 1;

            if improvement < self.config.convergence_tolerance {
                converged = true;
                break;
            }

            sub_solutions = new_solutions;

            // Check timeout
            if let Some(timeout) = self.config.timeout {
                if total_start.elapsed() > timeout {
                    break;
                }
            }

            println!(
                "Refinement iteration {}: improvement = {:.6}",
                iteration + 1,
                improvement
            );
        }

        // Step 4: Reconstruct complete solution
        let (variable_values, objective_value) = self.reconstruct_solution(qubo, &sub_solutions)?;

        // Calculate statistics
        let subproblem_sizes: Vec<usize> =
            sub_problems.iter().map(|sp| sp.variables.len()).collect();

        let stats = DecompositionStats {
            num_subproblems: sub_problems.len(),
            avg_subproblem_size: subproblem_sizes.iter().sum::<usize>() as f64
                / subproblem_sizes.len() as f64,
            max_subproblem_size: *subproblem_sizes.iter().max().unwrap_or(&0),
            min_subproblem_size: *subproblem_sizes.iter().min().unwrap_or(&0),
            decomposition_time,
            solving_time,
            refinement_iterations,
            converged,
        };

        Ok(DecomposedSolution {
            variable_values,
            objective_value,
            sub_solutions,
            stats,
        })
    }

    /// Decompose the QUBO problem into sub-problems
    fn decompose_problem(&self, qubo: &QuboModel) -> DecompositionResult<Vec<SubProblem>> {
        match &self.config.strategy {
            DecompositionStrategy::Spectral {
                num_partitions,
                overlap_size,
            } => self.spectral_decomposition(qubo, *num_partitions, *overlap_size),
            DecompositionStrategy::Block {
                block_size,
                overlap_size,
            } => self.block_decomposition(qubo, *block_size, *overlap_size),
            DecompositionStrategy::Hierarchical {
                min_partition_size,
                max_depth,
                overlap_size,
            } => self.hierarchical_decomposition(
                qubo,
                *min_partition_size,
                *max_depth,
                *overlap_size,
            ),
            DecompositionStrategy::Clustering {
                num_clusters,
                strength_threshold,
                overlap_size,
            } => self.clustering_decomposition(
                qubo,
                *num_clusters,
                *strength_threshold,
                *overlap_size,
            ),
        }
    }

    /// Spectral decomposition using graph partitioning
    fn spectral_decomposition(
        &self,
        qubo: &QuboModel,
        num_partitions: usize,
        overlap_size: usize,
    ) -> DecompositionResult<Vec<SubProblem>> {
        // Convert QUBO to graph representation
        let (ising, _) = qubo.to_ising();

        // Use spectral partitioner
        let partitioner = SpectralPartitioner::new();
        let edges: Vec<(usize, usize, f64)> = ising
            .couplings()
            .into_iter()
            .map(|coupling| (coupling.i, coupling.j, coupling.strength))
            .collect();
        let partition = partitioner
            .partition_graph(ising.num_qubits, &edges, num_partitions)
            .map_err(|e| DecompositionError::DecompositionFailed(e.to_string()))?;

        // Create sub-problems with overlap
        let mut sub_problems = Vec::new();

        for id in 0..num_partitions {
            let nodes = partition.get_partition(id);
            let variables = self.add_overlap(&nodes, &ising, overlap_size);
            let sub_qubo = self.extract_subproblem(qubo, &variables)?;
            let variable_mapping = variables
                .iter()
                .enumerate()
                .map(|(i, &var)| (i, var))
                .collect();

            sub_problems.push(SubProblem {
                variables: (0..variables.len()).collect(),
                variable_mapping,
                qubo: sub_qubo,
                fixed_variables: HashMap::new(),
                id,
            });
        }

        Ok(sub_problems)
    }

    /// Block decomposition based on variable indices
    fn block_decomposition(
        &self,
        qubo: &QuboModel,
        block_size: usize,
        overlap_size: usize,
    ) -> DecompositionResult<Vec<SubProblem>> {
        let mut sub_problems = Vec::new();
        let mut id = 0;

        for start in (0..qubo.num_variables).step_by(block_size - overlap_size) {
            let end = std::cmp::min(start + block_size, qubo.num_variables);

            if end - start < self.config.min_subproblem_size {
                break;
            }

            let variables: Vec<usize> = (start..end).collect();
            let sub_qubo = self.extract_subproblem(qubo, &variables)?;
            let variable_mapping = variables
                .iter()
                .enumerate()
                .map(|(i, &var)| (i, var))
                .collect();

            sub_problems.push(SubProblem {
                variables: (0..variables.len()).collect(),
                variable_mapping,
                qubo: sub_qubo,
                fixed_variables: HashMap::new(),
                id,
            });

            id += 1;
        }

        Ok(sub_problems)
    }

    /// Hierarchical decomposition using recursive bisection
    fn hierarchical_decomposition(
        &self,
        qubo: &QuboModel,
        min_partition_size: usize,
        max_depth: usize,
        overlap_size: usize,
    ) -> DecompositionResult<Vec<SubProblem>> {
        let all_variables: Vec<usize> = (0..qubo.num_variables).collect();
        let mut sub_problems = Vec::new();
        let mut id = 0;

        self.recursive_bisection(
            qubo,
            all_variables,
            0,
            max_depth,
            min_partition_size,
            overlap_size,
            &mut sub_problems,
            &mut id,
        )?;

        Ok(sub_problems)
    }

    /// Recursive bisection helper
    fn recursive_bisection(
        &self,
        qubo: &QuboModel,
        variables: Vec<usize>,
        depth: usize,
        max_depth: usize,
        min_size: usize,
        overlap_size: usize,
        sub_problems: &mut Vec<SubProblem>,
        id: &mut usize,
    ) -> DecompositionResult<()> {
        if variables.len() <= min_size || depth >= max_depth {
            // Create leaf sub-problem
            let sub_qubo = self.extract_subproblem(qubo, &variables)?;
            let variable_mapping = variables
                .iter()
                .enumerate()
                .map(|(i, &var)| (i, var))
                .collect();

            sub_problems.push(SubProblem {
                variables: (0..variables.len()).collect(),
                variable_mapping,
                qubo: sub_qubo,
                fixed_variables: HashMap::new(),
                id: *id,
            });

            *id += 1;
            return Ok(());
        }

        // Bisect the variables
        let mid = variables.len() / 2;
        let left_vars = variables[..mid].to_vec();
        let right_vars = variables[mid..].to_vec();

        // Add overlap
        let left_with_overlap = self.add_overlap_to_vars(&left_vars, &variables, overlap_size);
        let right_with_overlap = self.add_overlap_to_vars(&right_vars, &variables, overlap_size);

        // Recurse
        self.recursive_bisection(
            qubo,
            left_with_overlap,
            depth + 1,
            max_depth,
            min_size,
            overlap_size,
            sub_problems,
            id,
        )?;

        self.recursive_bisection(
            qubo,
            right_with_overlap,
            depth + 1,
            max_depth,
            min_size,
            overlap_size,
            sub_problems,
            id,
        )?;

        Ok(())
    }

    /// Clustering decomposition based on interaction strength
    fn clustering_decomposition(
        &self,
        qubo: &QuboModel,
        num_clusters: usize,
        strength_threshold: f64,
        overlap_size: usize,
    ) -> DecompositionResult<Vec<SubProblem>> {
        // Build interaction graph
        let mut clusters = vec![0; qubo.num_variables];
        let mut current_cluster = 0;
        let mut visited = vec![false; qubo.num_variables];

        // Simple clustering based on strong interactions
        for i in 0..qubo.num_variables {
            if visited[i] {
                continue;
            }

            let mut cluster_vars = vec![i];
            let mut stack = vec![i];
            visited[i] = true;

            while let Some(var) = stack.pop() {
                for j in 0..qubo.num_variables {
                    if i != j && !visited[j] {
                        let interaction = qubo.get_quadratic(var, j).unwrap_or(0.0).abs();
                        if interaction > strength_threshold {
                            visited[j] = true;
                            clusters[j] = current_cluster;
                            cluster_vars.push(j);
                            stack.push(j);
                        }
                    }
                }
            }

            clusters[i] = current_cluster;
            current_cluster += 1;

            if current_cluster >= num_clusters {
                break;
            }
        }

        // Assign remaining variables to nearest clusters
        for i in 0..qubo.num_variables {
            if !visited[i] {
                clusters[i] = i % num_clusters;
            }
        }

        // Create sub-problems from clusters
        let mut cluster_vars: HashMap<usize, Vec<usize>> = HashMap::new();
        for (var, &cluster) in clusters.iter().enumerate() {
            cluster_vars.entry(cluster).or_default().push(var);
        }

        let mut sub_problems = Vec::new();
        for (id, mut variables) in cluster_vars {
            // Add overlap
            let (ising, _) = qubo.to_ising();
            variables = self.add_overlap(&variables, &ising, overlap_size);

            if variables.len() >= self.config.min_subproblem_size {
                let sub_qubo = self.extract_subproblem(qubo, &variables)?;
                let variable_mapping = variables
                    .iter()
                    .enumerate()
                    .map(|(i, &var)| (i, var))
                    .collect();

                sub_problems.push(SubProblem {
                    variables: (0..variables.len()).collect(),
                    variable_mapping,
                    qubo: sub_qubo,
                    fixed_variables: HashMap::new(),
                    id,
                });
            }
        }

        Ok(sub_problems)
    }

    /// Add overlap to a set of variables
    fn add_overlap(
        &self,
        variables: &[usize],
        ising: &IsingModel,
        overlap_size: usize,
    ) -> Vec<usize> {
        let mut result: HashSet<usize> = variables.iter().copied().collect();

        for &var in variables {
            let mut neighbors = Vec::new();

            // Find neighbors through couplings
            for coupling in ising.couplings() {
                if coupling.i == var && !result.contains(&coupling.j) {
                    neighbors.push(coupling.j);
                } else if coupling.j == var && !result.contains(&coupling.i) {
                    neighbors.push(coupling.i);
                }
            }

            // Add the strongest neighbors up to overlap_size
            neighbors.sort_by(|&a, &b| {
                let strength_a = ising.get_coupling(var, a).unwrap_or(0.0).abs();
                let strength_b = ising.get_coupling(var, b).unwrap_or(0.0).abs();
                strength_b
                    .partial_cmp(&strength_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for &neighbor in neighbors.iter().take(overlap_size) {
                result.insert(neighbor);
            }
        }

        let mut result_vec: Vec<usize> = result.into_iter().collect();
        result_vec.sort_unstable();
        result_vec
    }

    /// Add overlap to a list of variables
    fn add_overlap_to_vars(
        &self,
        variables: &[usize],
        all_vars: &[usize],
        overlap_size: usize,
    ) -> Vec<usize> {
        let mut result: HashSet<usize> = variables.iter().copied().collect();

        // Add some variables from the boundary
        let boundary_start = if variables.len() >= overlap_size {
            variables.len() - overlap_size
        } else {
            0
        };

        for i in boundary_start..std::cmp::min(variables.len() + overlap_size, all_vars.len()) {
            if i < all_vars.len() {
                result.insert(all_vars[i]);
            }
        }

        let mut result_vec: Vec<usize> = result.into_iter().collect();
        result_vec.sort_unstable();
        result_vec
    }

    /// Extract a sub-problem from the original QUBO
    fn extract_subproblem(
        &self,
        qubo: &QuboModel,
        variables: &[usize],
    ) -> DecompositionResult<QuboModel> {
        let mut sub_qubo = QuboModel::new(variables.len());

        // Create variable index mapping
        let var_map: HashMap<usize, usize> = variables
            .iter()
            .enumerate()
            .map(|(i, &var)| (var, i))
            .collect();

        // Copy linear terms
        for (i, &var) in variables.iter().enumerate() {
            let linear = qubo.get_linear(var)?;
            if linear != 0.0 {
                sub_qubo.set_linear(i, linear)?;
            }
        }

        // Copy quadratic terms
        for &var1 in variables {
            for &var2 in variables {
                if var1 < var2 {
                    let quadratic = qubo.get_quadratic(var1, var2)?;
                    if quadratic != 0.0 {
                        let sub_var1 = var_map[&var1];
                        let sub_var2 = var_map[&var2];
                        sub_qubo.set_quadratic(sub_var1, sub_var2, quadratic)?;
                    }
                }
            }
        }

        Ok(sub_qubo)
    }

    /// Solve all sub-problems
    fn solve_subproblems(
        &self,
        sub_problems: &[SubProblem],
    ) -> DecompositionResult<Vec<SubSolution>> {
        let mut sub_solutions = Vec::new();

        for sub_problem in sub_problems {
            let (ising, offset) = sub_problem.qubo.to_ising();

            let mut simulator = QuantumAnnealingSimulator::new(self.config.solver_params.clone())
                .map_err(|e| DecompositionError::SolvingFailed(e.to_string()))?;

            let result = simulator
                .solve(&ising)
                .map_err(|e| DecompositionError::SolvingFailed(e.to_string()))?;

            // Convert spin solution to binary
            let values: HashMap<usize, bool> = result
                .best_spins
                .iter()
                .enumerate()
                .map(|(i, &spin)| {
                    let original_var = sub_problem.variable_mapping[&i];
                    (original_var, spin > 0)
                })
                .collect();

            let objective_value = result.best_energy + offset;

            sub_solutions.push(SubSolution {
                subproblem_id: sub_problem.id,
                values,
                objective_value,
                solver_info: result.info,
            });
        }

        Ok(sub_solutions)
    }

    /// Refine solutions through iterative improvement
    fn refine_solutions(
        &self,
        _qubo: &QuboModel,
        _sub_problems: &[SubProblem],
        sub_solutions: &[SubSolution],
    ) -> DecompositionResult<(Vec<SubSolution>, f64)> {
        // For now, return the same solutions with zero improvement
        // In a real implementation, this would:
        // 1. Fix variables based on current solutions
        // 2. Re-solve sub-problems with fixed boundary conditions
        // 3. Calculate improvement in objective value

        Ok((sub_solutions.to_vec(), 0.0))
    }

    /// Reconstruct the complete solution from sub-solutions
    fn reconstruct_solution(
        &self,
        qubo: &QuboModel,
        sub_solutions: &[SubSolution],
    ) -> DecompositionResult<(HashMap<usize, bool>, f64)> {
        let mut variable_values = HashMap::new();
        let mut vote_counts: HashMap<usize, (usize, usize)> = HashMap::new(); // (true_votes, false_votes)

        // Collect votes from all sub-solutions
        for sub_solution in sub_solutions {
            for (&var, &value) in &sub_solution.values {
                let (true_votes, false_votes) = vote_counts.entry(var).or_insert((0, 0));
                if value {
                    *true_votes += 1;
                } else {
                    *false_votes += 1;
                }
            }
        }

        // Decide final values based on majority vote
        for (var, (true_votes, false_votes)) in vote_counts {
            variable_values.insert(var, true_votes > false_votes);
        }

        // Calculate objective value
        let binary_vars: Vec<bool> = (0..qubo.num_variables)
            .map(|i| variable_values.get(&i).copied().unwrap_or(false))
            .collect();

        let objective_value = qubo.objective(&binary_vars)?;

        Ok((variable_values, objective_value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qubo::QuboBuilder;

    #[test]
    fn test_decomposition_config() {
        let config = DecompositionConfig::default();
        assert!(config.max_subproblem_size > 0);
        assert!(config.refinement_iterations > 0);
    }

    #[test]
    fn test_block_decomposition() {
        let mut qubo = QuboModel::new(20); // Increase size to 20 variables
        for i in 0..20 {
            qubo.set_linear(i, 1.0).expect("set_linear should succeed");
        }

        let config = DecompositionConfig {
            strategy: DecompositionStrategy::Block {
                block_size: 4,
                overlap_size: 1,
            },
            min_subproblem_size: 3, // Reduce minimum size to allow small blocks
            ..Default::default()
        };

        let decomposer = QuboDecomposer::new(config);
        let sub_problems = decomposer
            .block_decomposition(&qubo, 4, 1)
            .expect("block_decomposition should succeed");

        assert!(!sub_problems.is_empty());
        for sub_problem in &sub_problems {
            assert!(sub_problem.variables.len() <= 4);
        }
    }

    #[test]
    fn test_small_qubo_decomposition() {
        let mut builder = QuboBuilder::new();

        // Create a small test problem
        let vars: Vec<_> = (0..6)
            .map(|i| {
                builder
                    .add_variable(format!("x{}", i))
                    .expect("add_variable should succeed")
            })
            .collect();

        for i in 0..6 {
            builder
                .set_linear_term(&vars[i], 1.0)
                .expect("set_linear_term should succeed");
        }

        for i in 0..5 {
            builder
                .set_quadratic_term(&vars[i], &vars[i + 1], -2.0)
                .expect("set_quadratic_term should succeed");
        }

        let qubo_formulation = builder.build();
        let qubo = qubo_formulation.to_qubo_model();

        let config = DecompositionConfig {
            strategy: DecompositionStrategy::Block {
                block_size: 3,
                overlap_size: 1,
            },
            max_subproblem_size: 5,
            min_subproblem_size: 2,
            refinement_iterations: 2,
            ..Default::default()
        };

        let decomposer = QuboDecomposer::new(config);
        let result = decomposer.solve(&qubo).expect("solve should succeed");

        assert_eq!(result.variable_values.len(), 6);
        assert!(result.stats.num_subproblems > 0);
    }
}
