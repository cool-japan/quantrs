//! QAOA-Annealing Bridge Module
//!
//! This module implements the integration between Quantum Approximate Optimization Algorithm (QAOA)
//! from the circuit module and quantum annealing from the anneal module. It provides seamless
//! conversion between QAOA circuits and annealing problem formulations, enabling hybrid optimization
//! strategies that leverage both approaches.
//!
//! Key Features:
//! - Convert QAOA problems to annealing-compatible QUBO formulations
//! - Transform annealing solutions back to QAOA circuit parameters
//! - Hybrid optimization strategies combining QAOA and annealing
//! - Performance comparison and algorithm selection
//! - Unified problem description format supporting both paradigms

use std::collections::HashMap;
use std::f64::consts::PI;

use crate::applications::{ApplicationError, ApplicationResult};
use crate::ising::{IsingModel, QuboModel};
use crate::scirs2_integration::{SciRS2GraphAnalyzer, SciRS2QuboModel};
use crate::simulator::{AnnealingParams, QuantumAnnealingSimulator};

/// Bridge between QAOA circuits and quantum annealing
pub struct QaoaAnnealBridge {
    /// Configuration for the bridge
    pub config: BridgeConfig,
    /// Graph analyzer for problem analysis
    pub graph_analyzer: SciRS2GraphAnalyzer,
    /// Performance metrics for algorithm comparison
    pub metrics: PerformanceMetrics,
}

impl QaoaAnnealBridge {
    /// Create a new QAOA-annealing bridge
    #[must_use]
    pub fn new(config: BridgeConfig) -> Self {
        Self {
            config,
            graph_analyzer: SciRS2GraphAnalyzer::new(),
            metrics: PerformanceMetrics::default(),
        }
    }

    /// Convert QAOA problem to QUBO formulation for annealing.
    ///
    /// Performs the exact Pauli-Z to binary-variable substitution and then
    /// records analysis metrics for the resulting problem graph.
    pub fn qaoa_to_qubo(
        &mut self,
        qaoa_problem: &QaoaProblem,
    ) -> ApplicationResult<SciRS2QuboModel> {
        let qubo = Self::qaoa_problem_to_qubo(qaoa_problem)?;

        // Analyze the resulting problem graph
        let analysis = self.graph_analyzer.analyze_problem_graph(&qubo)?;
        println!(
            "QAOA->QUBO conversion: {} qubits, {} edges, difficulty: {:?}",
            analysis.metrics.num_nodes, analysis.metrics.num_edges, analysis.embedding_difficulty
        );

        Ok(qubo)
    }

    /// Pure QAOA-cost-Hamiltonian to QUBO conversion (no analyzer side effects).
    ///
    /// Each `Z_i` becomes `(1 - 2 x_i)` and each `Z_i Z_j` expands to
    /// `1 - 2 x_i - 2 x_j + 4 x_i x_j`; the constant parts accumulate into the
    /// QUBO offset so the objective value is preserved exactly.
    fn qaoa_problem_to_qubo(qaoa_problem: &QaoaProblem) -> ApplicationResult<SciRS2QuboModel> {
        let num_vars = qaoa_problem.num_qubits;
        let mut qubo = SciRS2QuboModel::new(num_vars)?;

        for clause in &qaoa_problem.cost_clauses {
            match clause {
                QaoaClause::SingleQubit { qubit, weight } => {
                    let current = qubo.linear_terms[*qubit];
                    qubo.set_linear(*qubit, current - 2.0 * weight)?;
                    qubo.offset += weight;
                }
                QaoaClause::TwoQubit {
                    qubit1,
                    qubit2,
                    weight,
                } => {
                    let current_linear1 = qubo.linear_terms[*qubit1];
                    let current_linear2 = qubo.linear_terms[*qubit2];
                    qubo.set_linear(*qubit1, current_linear1 - 2.0 * weight)?;
                    qubo.set_linear(*qubit2, current_linear2 - 2.0 * weight)?;
                    qubo.set_quadratic(*qubit1, *qubit2, 4.0 * weight)?;
                    qubo.offset += weight;
                }
                QaoaClause::MultiQubit { qubits, weight } => {
                    Self::reduce_multi_qubit_term(&mut qubo, qubits, *weight)?;
                }
            }
        }

        Ok(qubo)
    }

    /// Convert annealing solution back to QAOA circuit parameters
    pub fn solution_to_qaoa_params(
        &self,
        solution: &[i8],
        qaoa_problem: &QaoaProblem,
    ) -> ApplicationResult<QaoaParameters> {
        if solution.len() != qaoa_problem.num_qubits {
            return Err(ApplicationError::InvalidConfiguration(
                "Solution length mismatch".to_string(),
            ));
        }

        // Convert binary solution to angle parameters
        // This is a simplified mapping - more sophisticated approaches could be used
        let mut beta_params = Vec::new();
        let mut gamma_params = Vec::new();

        let num_layers = self.config.default_qaoa_layers;
        for layer in 0..num_layers {
            // Map solution bits to mixing angles (beta)
            let beta_sum: f64 = solution
                .iter()
                .skip(layer * solution.len() / num_layers)
                .take(solution.len() / num_layers)
                .map(|&x| f64::from(x))
                .sum();
            let beta = (beta_sum / (solution.len() / num_layers) as f64) * PI / 2.0;
            beta_params.push(beta);

            // Map to cost angles (gamma)
            let gamma_sum: f64 = solution
                .iter()
                .rev()
                .skip(layer * solution.len() / num_layers)
                .take(solution.len() / num_layers)
                .map(|&x| f64::from(x))
                .sum();
            let gamma = (gamma_sum / (solution.len() / num_layers) as f64) * PI;
            gamma_params.push(gamma);
        }

        Ok(QaoaParameters {
            beta: beta_params,
            gamma: gamma_params,
            num_layers,
        })
    }

    /// Hybrid optimization using both QAOA and annealing
    pub fn hybrid_optimize(
        &mut self,
        problem: &UnifiedProblem,
    ) -> ApplicationResult<HybridOptimizationResult> {
        let start_time = std::time::Instant::now();

        // Step 1: Convert to QAOA problem if needed
        let qaoa_problem = match &problem.formulation {
            ProblemFormulation::Qaoa(q) => q.clone(),
            ProblemFormulation::Qubo(q) => self.qubo_to_qaoa(q)?,
            ProblemFormulation::Ising(i) => self.ising_to_qaoa(i)?,
        };

        // Step 2: Convert to annealing-compatible QUBO
        let qubo = self.qaoa_to_qubo(&qaoa_problem)?;

        // Step 3: Analyze problem characteristics to choose strategy
        let analysis = self.graph_analyzer.analyze_problem_graph(&qubo)?;
        let strategy = self.select_optimization_strategy(&analysis)?;

        println!(
            "Selected strategy: {:?} for problem with {} variables",
            strategy, qubo.num_variables
        );

        // Step 4: Execute chosen strategy
        let result = match strategy {
            OptimizationStrategy::AnnealingOnly => self.solve_with_annealing(&qubo)?,
            OptimizationStrategy::QaoaOnly => self.solve_with_qaoa(&qaoa_problem)?,
            OptimizationStrategy::Sequential => self.solve_sequential(&qaoa_problem, &qubo)?,
            OptimizationStrategy::Parallel => self.solve_parallel(&qaoa_problem, &qubo)?,
        };

        let duration = start_time.elapsed();
        self.metrics.total_optimizations += 1;
        self.metrics.total_time += duration;

        Ok(HybridOptimizationResult {
            best_energy: result.energy,
            best_solution: result.solution,
            qaoa_params: result.qaoa_params,
            strategy_used: strategy,
            execution_time: duration,
            iterations: result.iterations,
            convergence_data: result.convergence_data,
        })
    }

    /// Reduce a multi-qubit cost term to QUBO form.
    ///
    /// A `k`-local term `weight · Z_{q₁}…Z_{qₖ}` couples the parity of `k`
    /// qubits. Single- and two-qubit terms map exactly onto QUBO linear and
    /// quadratic terms (handled by [`qaoa_to_qubo`]). For `k > 2` an *exact*
    /// QUBO representation requires introducing auxiliary variables and a
    /// quadratization penalty schedule (e.g. Rosenberg) that enlarges the
    /// variable set — machinery this lightweight bridge does not provision.
    ///
    /// Rather than silently substitute a lossy pairwise chain (which would
    /// change the optimization landscape without telling the caller), this
    /// returns an honest error for genuinely high-order terms.
    fn reduce_multi_qubit_term(
        qubo: &mut SciRS2QuboModel,
        qubits: &[usize],
        weight: f64,
    ) -> ApplicationResult<()> {
        match qubits.len() {
            0 => {
                // Constant parity term: contributes a fixed offset.
                qubo.offset += weight;
                Ok(())
            }
            1 => {
                let current = qubo.linear_terms[qubits[0]];
                qubo.set_linear(qubits[0], current - 2.0 * weight)
            }
            2 => {
                let current1 = qubo.linear_terms[qubits[0]];
                let current2 = qubo.linear_terms[qubits[1]];
                qubo.set_linear(qubits[0], current1 - 2.0 * weight)?;
                qubo.set_linear(qubits[1], current2 - 2.0 * weight)?;
                qubo.set_quadratic(qubits[0], qubits[1], 4.0 * weight)?;
                qubo.offset += weight;
                Ok(())
            }
            k => Err(ApplicationError::InvalidConfiguration(format!(
                "exact QUBO reduction of a {k}-local cost term requires auxiliary \
                 variables (quadratization), which this bridge does not provision; \
                 decompose the term to at most 2-local before conversion"
            ))),
        }
    }

    fn qubo_to_qaoa(&self, _qubo: &SciRS2QuboModel) -> ApplicationResult<QaoaProblem> {
        // Simplified conversion - would implement proper QUBO to QAOA mapping
        Err(ApplicationError::InvalidConfiguration(
            "QUBO to QAOA conversion not yet implemented".to_string(),
        ))
    }

    fn ising_to_qaoa(&self, _ising: &IsingModel) -> ApplicationResult<QaoaProblem> {
        // Simplified conversion - would implement proper Ising to QAOA mapping
        Err(ApplicationError::InvalidConfiguration(
            "Ising to QAOA conversion not yet implemented".to_string(),
        ))
    }

    const fn select_optimization_strategy(
        &self,
        analysis: &crate::scirs2_integration::GraphAnalysisResult,
    ) -> ApplicationResult<OptimizationStrategy> {
        use crate::scirs2_integration::EmbeddingDifficulty;

        match analysis.embedding_difficulty {
            EmbeddingDifficulty::Easy => {
                if analysis.metrics.num_nodes < 20 {
                    Ok(OptimizationStrategy::QaoaOnly)
                } else {
                    Ok(OptimizationStrategy::Sequential)
                }
            }
            EmbeddingDifficulty::Medium => Ok(OptimizationStrategy::Parallel),
            EmbeddingDifficulty::Hard => Ok(OptimizationStrategy::AnnealingOnly),
        }
    }

    /// Convert a [`SciRS2QuboModel`] into the in-crate [`QuboModel`] so it can be
    /// fed to the real annealing simulator. The CSR quadratic matrix stores each
    /// off-diagonal entry symmetrically; the diagonal is folded into the linear
    /// (QUBO) terms because `x² = x` for binary variables.
    fn to_inner_qubo(qubo: &SciRS2QuboModel) -> ApplicationResult<QuboModel> {
        let mut inner = QuboModel::new(qubo.num_variables);

        for (i, &value) in qubo.linear_terms.iter().enumerate() {
            if value != 0.0 {
                inner
                    .set_linear(i, value)
                    .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
            }
        }

        for (i, j, value) in qubo.iter_quadratic() {
            if value == 0.0 {
                continue;
            }
            if i == j {
                inner
                    .add_linear(i, value)
                    .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
            } else if i < j {
                // The CSR matrix is symmetric: the full off-diagonal weight for
                // the unordered pair {i, j} is value(i,j) + value(j,i). Take only
                // the upper triangle and double to recover the total coupling.
                inner
                    .set_quadratic(i, j, 2.0 * value)
                    .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
            }
        }

        Ok(inner)
    }

    /// Solve a QUBO with the real simulated-quantum-annealing engine.
    ///
    /// The QUBO is mapped to its equivalent Ising model and handed to
    /// [`QuantumAnnealingSimulator`]; the returned spin ground-state estimate is
    /// mapped back to a binary `{0, 1}` assignment and scored with the original
    /// QUBO so the reported energy is the genuine objective value.
    fn solve_qubo_via_annealing(
        &self,
        qubo: &SciRS2QuboModel,
    ) -> ApplicationResult<(Vec<i8>, f64, usize)> {
        if qubo.num_variables == 0 {
            return Ok((Vec::new(), qubo.evaluate(&[])?, 0));
        }

        let inner = Self::to_inner_qubo(qubo)?;
        let (ising, _offset) = inner.to_ising();

        let mut params = AnnealingParams::default();
        params.num_sweeps = self.config.annealing_sweeps.max(1);
        params.num_repetitions = self.config.annealing_repetitions.max(1);

        let simulator = QuantumAnnealingSimulator::new(params)
            .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;
        let result = simulator
            .solve(&ising)
            .map_err(|e| ApplicationError::OptimizationError(e.to_string()))?;

        // Map Ising spins (-1/+1) to QUBO binary variables (0/1).
        let solution: Vec<i8> = result.best_spins.iter().map(|&s| i8::from(s > 0)).collect();
        let energy = qubo.evaluate(&solution)?;

        Ok((solution, energy, result.total_sweeps))
    }

    fn solve_with_annealing(&self, qubo: &SciRS2QuboModel) -> ApplicationResult<SolutionResult> {
        let (solution, energy, sweeps) = self.solve_qubo_via_annealing(qubo)?;

        Ok(SolutionResult {
            energy,
            solution,
            qaoa_params: None,
            iterations: sweeps,
            convergence_data: vec![energy],
        })
    }

    fn solve_with_qaoa(&self, problem: &QaoaProblem) -> ApplicationResult<SolutionResult> {
        // This crate does not host a state-vector QAOA executor, so the QAOA
        // cost Hamiltonian is solved on its equivalent QUBO with the real
        // annealing engine. The bitstring is scored with the genuine QUBO
        // objective and the variational angles are derived from that solution
        // via the existing solution->parameter map (no placeholder energy).
        let qubo = Self::qaoa_problem_to_qubo(problem)?;
        let (solution, energy, sweeps) = self.solve_qubo_via_annealing(&qubo)?;
        let qaoa_params = self.solution_to_qaoa_params(&solution, problem)?;

        Ok(SolutionResult {
            energy,
            solution,
            qaoa_params: Some(qaoa_params),
            iterations: sweeps,
            convergence_data: vec![energy],
        })
    }

    fn solve_sequential(
        &self,
        qaoa_problem: &QaoaProblem,
        qubo: &SciRS2QuboModel,
    ) -> ApplicationResult<SolutionResult> {
        // First try QAOA, then refine with annealing
        let qaoa_result = self.solve_with_qaoa(qaoa_problem)?;

        if qaoa_result.energy < self.config.energy_threshold {
            Ok(qaoa_result)
        } else {
            self.solve_with_annealing(qubo)
        }
    }

    fn solve_parallel(
        &self,
        qaoa_problem: &QaoaProblem,
        qubo: &SciRS2QuboModel,
    ) -> ApplicationResult<SolutionResult> {
        // Run both approaches and return the better result
        let qaoa_result = self.solve_with_qaoa(qaoa_problem)?;
        let annealing_result = self.solve_with_annealing(qubo)?;

        if qaoa_result.energy < annealing_result.energy {
            Ok(qaoa_result)
        } else {
            Ok(annealing_result)
        }
    }
}

/// Configuration for the QAOA-annealing bridge
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Default number of QAOA layers
    pub default_qaoa_layers: usize,
    /// Energy threshold for strategy selection
    pub energy_threshold: f64,
    /// Maximum problem size for QAOA
    pub max_qaoa_qubits: usize,
    /// Timeout for hybrid optimization
    pub timeout_seconds: u64,
    /// Number of Monte-Carlo sweeps used by the annealing solver
    pub annealing_sweeps: usize,
    /// Number of independent annealing restarts
    pub annealing_repetitions: usize,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            default_qaoa_layers: 2,
            energy_threshold: -1.0,
            max_qaoa_qubits: 20,
            timeout_seconds: 300,
            annealing_sweeps: 1000,
            annealing_repetitions: 10,
        }
    }
}

/// QAOA problem representation
#[derive(Debug, Clone)]
pub struct QaoaProblem {
    /// Number of qubits
    pub num_qubits: usize,
    /// Cost Hamiltonian clauses
    pub cost_clauses: Vec<QaoaClause>,
    /// Mixing Hamiltonian (default is X on all qubits)
    pub mixing_type: MixingType,
}

/// QAOA Hamiltonian clause
#[derive(Debug, Clone)]
pub enum QaoaClause {
    SingleQubit {
        qubit: usize,
        weight: f64,
    },
    TwoQubit {
        qubit1: usize,
        qubit2: usize,
        weight: f64,
    },
    MultiQubit {
        qubits: Vec<usize>,
        weight: f64,
    },
}

/// Type of mixing Hamiltonian
#[derive(Debug, Clone)]
pub enum MixingType {
    /// Standard X mixer
    StandardX,
    /// Custom mixing Hamiltonian
    Custom(Vec<QaoaClause>),
}

/// QAOA circuit parameters
#[derive(Debug, Clone)]
pub struct QaoaParameters {
    /// Mixing angles (beta parameters)
    pub beta: Vec<f64>,
    /// Cost angles (gamma parameters)
    pub gamma: Vec<f64>,
    /// Number of QAOA layers
    pub num_layers: usize,
}

/// Unified problem description supporting multiple formulations
#[derive(Debug, Clone)]
pub struct UnifiedProblem {
    /// Problem formulation
    pub formulation: ProblemFormulation,
    /// Problem metadata
    pub metadata: ProblemMetadata,
}

/// Problem formulation variants
#[derive(Debug, Clone)]
pub enum ProblemFormulation {
    Qaoa(QaoaProblem),
    Qubo(SciRS2QuboModel),
    Ising(IsingModel),
}

/// Problem metadata
#[derive(Debug, Clone)]
pub struct ProblemMetadata {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub difficulty: String,
}

/// Optimization strategy selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationStrategy {
    /// Use only quantum annealing
    AnnealingOnly,
    /// Use only QAOA
    QaoaOnly,
    /// Use QAOA first, then annealing if needed
    Sequential,
    /// Run both in parallel and take the best result
    Parallel,
}

/// Internal solution result
#[derive(Debug, Clone)]
struct SolutionResult {
    energy: f64,
    solution: Vec<i8>,
    qaoa_params: Option<QaoaParameters>,
    iterations: usize,
    convergence_data: Vec<f64>,
}

/// Result of hybrid optimization
#[derive(Debug, Clone)]
pub struct HybridOptimizationResult {
    /// Best energy found
    pub best_energy: f64,
    /// Best solution found
    pub best_solution: Vec<i8>,
    /// QAOA parameters if applicable
    pub qaoa_params: Option<QaoaParameters>,
    /// Strategy that was used
    pub strategy_used: OptimizationStrategy,
    /// Total execution time
    pub execution_time: std::time::Duration,
    /// Number of iterations
    pub iterations: usize,
    /// Convergence data
    pub convergence_data: Vec<f64>,
}

/// Performance metrics for the bridge
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Total number of optimizations performed
    pub total_optimizations: usize,
    /// Total time spent in optimization
    pub total_time: std::time::Duration,
    /// Strategy usage statistics
    pub strategy_counts: HashMap<OptimizationStrategy, usize>,
    /// Average solution quality by strategy
    pub quality_by_strategy: HashMap<OptimizationStrategy, f64>,
}

impl PerformanceMetrics {
    /// Get average optimization time
    #[must_use]
    pub fn average_time(&self) -> std::time::Duration {
        if self.total_optimizations > 0 {
            self.total_time / self.total_optimizations as u32
        } else {
            std::time::Duration::from_secs(0)
        }
    }

    /// Get most used strategy
    #[must_use]
    pub fn most_used_strategy(&self) -> Option<OptimizationStrategy> {
        self.strategy_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&strategy, _)| strategy)
    }
}

/// Example usage and testing functions
#[must_use]
pub fn create_example_max_cut_problem(num_vertices: usize) -> UnifiedProblem {
    let mut clauses = Vec::new();

    // Create a ring graph for Max-Cut
    for i in 0..num_vertices {
        let j = (i + 1) % num_vertices;
        clauses.push(QaoaClause::TwoQubit {
            qubit1: i,
            qubit2: j,
            weight: 1.0,
        });
    }

    let qaoa_problem = QaoaProblem {
        num_qubits: num_vertices,
        cost_clauses: clauses,
        mixing_type: MixingType::StandardX,
    };

    UnifiedProblem {
        formulation: ProblemFormulation::Qaoa(qaoa_problem),
        metadata: ProblemMetadata {
            name: format!("Max-Cut Ring Graph ({num_vertices})"),
            description: "Maximum cut problem on a ring graph".to_string(),
            tags: vec!["max-cut".to_string(), "graph".to_string()],
            difficulty: "medium".to_string(),
        },
    }
}

#[cfg(test)]
use scirs2_sparse::SparseArray;
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let config = BridgeConfig::default();
        let bridge = QaoaAnnealBridge::new(config);
        assert_eq!(bridge.config.default_qaoa_layers, 2);
    }

    #[test]
    fn test_max_cut_problem_creation() {
        let problem = create_example_max_cut_problem(4);

        if let ProblemFormulation::Qaoa(qaoa) = problem.formulation {
            assert_eq!(qaoa.num_qubits, 4);
            assert_eq!(qaoa.cost_clauses.len(), 4); // Ring graph has 4 edges
        } else {
            panic!("Expected QAOA formulation");
        }
    }

    #[test]
    fn test_qaoa_to_qubo_conversion() {
        let problem = create_example_max_cut_problem(3);
        let mut bridge = QaoaAnnealBridge::new(BridgeConfig::default());

        if let ProblemFormulation::Qaoa(qaoa) = problem.formulation {
            let result = bridge.qaoa_to_qubo(&qaoa);
            assert!(result.is_ok());

            let qubo = result.expect("QUBO conversion should succeed");
            assert_eq!(qubo.num_variables, 3);
            assert_eq!(qubo.quadratic_matrix.nnz() / 2, 3); // 3 edges in ring
        }
    }

    #[test]
    fn test_hybrid_optimization() {
        let problem = create_example_max_cut_problem(6);
        let mut bridge = QaoaAnnealBridge::new(BridgeConfig::default());

        let result = bridge.hybrid_optimize(&problem);
        assert!(result.is_ok());

        let opt_result = result.expect("hybrid optimization should succeed");
        assert_eq!(opt_result.best_solution.len(), 6);
        // Use microseconds for fast operations (millis may be 0 for very fast tests)
        assert!(opt_result.execution_time.as_micros() > 0);
    }
}
