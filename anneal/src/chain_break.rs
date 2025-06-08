//! Chain break resolution algorithms for quantum annealing
//!
//! When logical variables are embedded onto physical qubits using chains,
//! the physical qubits in a chain may disagree in the solution. This module
//! provides algorithms to resolve these chain breaks.

use std::collections::{HashMap, HashSet};
use crate::embedding::Embedding;
use crate::ising::{IsingError, IsingResult};

/// Represents a solution from quantum annealing hardware
#[derive(Debug, Clone)]
pub struct HardwareSolution {
    /// Values of physical qubits (spin values: +1 or -1)
    pub spins: Vec<i8>,
    /// Energy of this solution
    pub energy: f64,
    /// Number of occurrences (for multiple reads)
    pub occurrences: usize,
}

/// Resolved solution after chain break resolution
#[derive(Debug, Clone)]
pub struct ResolvedSolution {
    /// Values of logical variables
    pub logical_spins: Vec<i8>,
    /// Number of broken chains
    pub chain_breaks: usize,
    /// Energy after resolution
    pub energy: f64,
    /// Original hardware solution
    pub hardware_solution: HardwareSolution,
}

/// Chain break resolution method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionMethod {
    /// Take majority vote within each chain
    MajorityVote,
    /// Minimize energy of the logical problem
    EnergyMinimization,
    /// Use weighted majority based on coupling strengths
    WeightedMajority,
    /// Discard solutions with broken chains
    Discard,
}

/// Chain break resolver
pub struct ChainBreakResolver {
    /// Resolution method to use
    pub method: ResolutionMethod,
    /// Tie-breaking strategy for majority vote
    pub tie_break_random: bool,
    /// Random seed for tie-breaking
    pub seed: Option<u64>,
}

impl Default for ChainBreakResolver {
    fn default() -> Self {
        Self {
            method: ResolutionMethod::MajorityVote,
            tie_break_random: true,
            seed: None,
        }
    }
}

impl ChainBreakResolver {
    /// Resolve chain breaks in a single hardware solution
    pub fn resolve_solution(
        &self,
        hardware_solution: &HardwareSolution,
        embedding: &Embedding,
        logical_problem: Option<&LogicalProblem>,
    ) -> IsingResult<ResolvedSolution> {
        match self.method {
            ResolutionMethod::MajorityVote => {
                self.resolve_majority_vote(hardware_solution, embedding)
            }
            ResolutionMethod::WeightedMajority => {
                self.resolve_weighted_majority(hardware_solution, embedding)
            }
            ResolutionMethod::EnergyMinimization => {
                let problem = logical_problem.ok_or_else(|| {
                    IsingError::InvalidValue(
                        "Energy minimization requires logical problem".to_string()
                    )
                })?;
                self.resolve_energy_minimization(hardware_solution, embedding, problem)
            }
            ResolutionMethod::Discard => {
                self.resolve_discard(hardware_solution, embedding)
            }
        }
    }

    /// Resolve multiple hardware solutions
    pub fn resolve_solutions(
        &self,
        hardware_solutions: &[HardwareSolution],
        embedding: &Embedding,
        logical_problem: Option<&LogicalProblem>,
    ) -> IsingResult<Vec<ResolvedSolution>> {
        let mut resolved = Vec::new();

        for hw_solution in hardware_solutions {
            match self.resolve_solution(hw_solution, embedding, logical_problem) {
                Ok(solution) => resolved.push(solution),
                Err(_) if self.method == ResolutionMethod::Discard => {
                    // Skip broken solutions when using discard method
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        // Sort by energy
        resolved.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());

        Ok(resolved)
    }

    /// Resolve using majority vote
    fn resolve_majority_vote(
        &self,
        hardware_solution: &HardwareSolution,
        embedding: &Embedding,
    ) -> IsingResult<ResolvedSolution> {
        let mut logical_spins = Vec::new();
        let mut chain_breaks = 0;
        let num_vars = embedding.chains.len();

        for var in 0..num_vars {
            let chain = embedding.chains.get(&var)
                .ok_or_else(|| IsingError::InvalidQubit(var))?;

            // Count votes
            let mut plus_votes = 0;
            let mut minus_votes = 0;

            for &qubit in chain {
                if qubit >= hardware_solution.spins.len() {
                    return Err(IsingError::InvalidQubit(qubit));
                }

                match hardware_solution.spins[qubit] {
                    1 => plus_votes += 1,
                    -1 => minus_votes += 1,
                    _ => return Err(IsingError::InvalidValue(
                        "Invalid spin value".to_string()
                    )),
                }
            }

            // Determine logical value
            let logical_value = if plus_votes > minus_votes {
                1
            } else if minus_votes > plus_votes {
                -1
            } else {
                // Tie - use random or default to +1
                if self.tie_break_random {
                    // Simple deterministic tie-break based on variable index
                    if var % 2 == 0 { 1 } else { -1 }
                } else {
                    1
                }
            };

            // Check for chain breaks
            let unanimous = plus_votes == 0 || minus_votes == 0;
            if !unanimous {
                chain_breaks += 1;
            }

            logical_spins.push(logical_value);
        }

        Ok(ResolvedSolution {
            logical_spins,
            chain_breaks,
            energy: hardware_solution.energy, // Will be recalculated if needed
            hardware_solution: hardware_solution.clone(),
        })
    }

    /// Resolve using weighted majority based on coupling strengths
    fn resolve_weighted_majority(
        &self,
        hardware_solution: &HardwareSolution,
        embedding: &Embedding,
    ) -> IsingResult<ResolvedSolution> {
        // For now, use simple majority vote
        // TODO: Implement proper weighted majority using coupling strengths
        self.resolve_majority_vote(hardware_solution, embedding)
    }

    /// Resolve by minimizing energy of logical problem
    fn resolve_energy_minimization(
        &self,
        hardware_solution: &HardwareSolution,
        embedding: &Embedding,
        logical_problem: &LogicalProblem,
    ) -> IsingResult<ResolvedSolution> {
        let mut resolved = self.resolve_majority_vote(hardware_solution, embedding)?;

        // For each broken chain, try flipping the logical variable
        for var in 0..resolved.logical_spins.len() {
            if self.is_chain_broken(var, hardware_solution, embedding)? {
                // Calculate energy with current value
                let current_energy = logical_problem.calculate_energy(&resolved.logical_spins);

                // Flip and calculate energy
                resolved.logical_spins[var] *= -1;
                let flipped_energy = logical_problem.calculate_energy(&resolved.logical_spins);

                // Keep the flip if it lowers energy
                if flipped_energy >= current_energy {
                    resolved.logical_spins[var] *= -1; // Flip back
                }
            }
        }

        // Recalculate final energy
        resolved.energy = logical_problem.calculate_energy(&resolved.logical_spins);

        Ok(resolved)
    }

    /// Discard solutions with broken chains
    fn resolve_discard(
        &self,
        hardware_solution: &HardwareSolution,
        embedding: &Embedding,
    ) -> IsingResult<ResolvedSolution> {
        let resolved = self.resolve_majority_vote(hardware_solution, embedding)?;

        if resolved.chain_breaks > 0 {
            Err(IsingError::HardwareConstraint(
                format!("Solution has {} broken chains", resolved.chain_breaks)
            ))
        } else {
            Ok(resolved)
        }
    }

    /// Check if a chain is broken
    fn is_chain_broken(
        &self,
        var: usize,
        hardware_solution: &HardwareSolution,
        embedding: &Embedding,
    ) -> IsingResult<bool> {
        let chain = embedding.chains.get(&var)
            .ok_or_else(|| IsingError::InvalidQubit(var))?;

        if chain.is_empty() {
            return Ok(false);
        }

        let first_spin = hardware_solution.spins[chain[0]];
        
        for &qubit in &chain[1..] {
            if hardware_solution.spins[qubit] != first_spin {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

/// Represents a logical problem (QUBO or Ising)
#[derive(Debug, Clone)]
pub struct LogicalProblem {
    /// Linear coefficients (h_i in Ising, diagonal in QUBO)
    pub linear: Vec<f64>,
    /// Quadratic coefficients as adjacency list
    pub quadratic: HashMap<(usize, usize), f64>,
    /// Constant offset
    pub offset: f64,
}

impl LogicalProblem {
    /// Create a new logical problem
    pub fn new(num_vars: usize) -> Self {
        Self {
            linear: vec![0.0; num_vars],
            quadratic: HashMap::new(),
            offset: 0.0,
        }
    }

    /// Calculate energy for a given spin configuration
    pub fn calculate_energy(&self, spins: &[i8]) -> f64 {
        let mut energy = self.offset;

        // Linear terms
        for (i, &h) in self.linear.iter().enumerate() {
            if i < spins.len() {
                energy += h * spins[i] as f64;
            }
        }

        // Quadratic terms
        for (&(i, j), &J) in &self.quadratic {
            if i < spins.len() && j < spins.len() {
                energy += J * spins[i] as f64 * spins[j] as f64;
            }
        }

        energy
    }

    /// Convert from QUBO to Ising representation
    pub fn from_qubo(qubo_matrix: &[Vec<f64>], offset: f64) -> IsingResult<Self> {
        let n = qubo_matrix.len();
        let mut problem = Self::new(n);
        problem.offset = offset;

        // Convert QUBO Q_ij to Ising h_i and J_ij
        // x_i = (s_i + 1) / 2
        // Minimize x^T Q x becomes minimize sum_i h_i s_i + sum_{i<j} J_ij s_i s_j

        for i in 0..n {
            for j in i..n {
                let q_ij = qubo_matrix[i][j];
                if q_ij.abs() > 1e-10 {
                    if i == j {
                        // Diagonal term contributes to linear coefficient
                        problem.linear[i] += q_ij / 2.0;
                        problem.offset += q_ij / 4.0;
                    } else {
                        // Off-diagonal term
                        problem.quadratic.insert((i, j), q_ij / 4.0);
                        problem.linear[i] += q_ij / 4.0;
                        problem.linear[j] += q_ij / 4.0;
                        problem.offset += q_ij / 4.0;
                    }
                }
            }
        }

        Ok(problem)
    }
}

/// Chain strength optimizer
pub struct ChainStrengthOptimizer {
    /// Minimum chain strength
    pub min_strength: f64,
    /// Maximum chain strength
    pub max_strength: f64,
    /// Number of strength values to try
    pub num_tries: usize,
}

impl Default for ChainStrengthOptimizer {
    fn default() -> Self {
        Self {
            min_strength: 0.1,
            max_strength: 10.0,
            num_tries: 10,
        }
    }
}

impl ChainStrengthOptimizer {
    /// Find optimal chain strength by analyzing the problem
    pub fn find_optimal_strength(&self, logical_problem: &LogicalProblem) -> f64 {
        // Calculate statistics of the logical problem coefficients
        let mut all_coeffs = Vec::new();

        // Add linear coefficients
        for &h in &logical_problem.linear {
            if h.abs() > 1e-10 {
                all_coeffs.push(h.abs());
            }
        }

        // Add quadratic coefficients
        for &J in logical_problem.quadratic.values() {
            if J.abs() > 1e-10 {
                all_coeffs.push(J.abs());
            }
        }

        if all_coeffs.is_empty() {
            return 1.0; // Default strength
        }

        // Sort coefficients
        all_coeffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use median as base strength
        let median = if all_coeffs.len() % 2 == 0 {
            (all_coeffs[all_coeffs.len() / 2 - 1] + all_coeffs[all_coeffs.len() / 2]) / 2.0
        } else {
            all_coeffs[all_coeffs.len() / 2]
        };

        // Chain strength should be strong enough to keep chains together
        // but not so strong as to dominate the problem
        (median * 1.5).max(self.min_strength).min(self.max_strength)
    }

    /// Optimize chain strength through multiple runs
    pub fn optimize_strength(
        &self,
        logical_problem: &LogicalProblem,
        test_solutions: &[Vec<i8>],
    ) -> f64 {
        let mut best_strength = self.find_optimal_strength(logical_problem);
        let mut best_score = f64::INFINITY;

        // Try different strengths
        let step = (self.max_strength - self.min_strength) / (self.num_tries as f64);
        
        for i in 0..self.num_tries {
            let strength = self.min_strength + i as f64 * step;
            
            // Evaluate this strength
            let score = self.evaluate_strength(strength, logical_problem, test_solutions);
            
            if score < best_score {
                best_score = score;
                best_strength = strength;
            }
        }

        best_strength
    }

    /// Evaluate a chain strength
    fn evaluate_strength(
        &self,
        strength: f64,
        logical_problem: &LogicalProblem,
        test_solutions: &[Vec<i8>],
    ) -> f64 {
        // Simple evaluation: prefer strengths that maintain solution quality
        // In practice, this would run actual annealing with different strengths
        
        // For now, return a score based on the ratio to problem coefficients
        let avg_coeff = self.calculate_average_coefficient(logical_problem);
        
        // Penalty for being too different from problem scale
        (strength / avg_coeff - 1.5).abs()
    }

    /// Calculate average coefficient magnitude
    fn calculate_average_coefficient(&self, logical_problem: &LogicalProblem) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        for &h in &logical_problem.linear {
            if h.abs() > 1e-10 {
                sum += h.abs();
                count += 1;
            }
        }

        for &J in logical_problem.quadratic.values() {
            if J.abs() > 1e-10 {
                sum += J.abs();
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            1.0
        }
    }
}

/// Statistics about chain breaks
#[derive(Debug, Clone, Default)]
pub struct ChainBreakStats {
    /// Total number of chains
    pub total_chains: usize,
    /// Number of broken chains per solution
    pub broken_chains: Vec<usize>,
    /// Chain break rate
    pub break_rate: f64,
    /// Most frequently broken variables
    pub frequent_breaks: Vec<(usize, usize)>,
}

impl ChainBreakStats {
    /// Analyze chain breaks across multiple solutions
    pub fn analyze(
        hardware_solutions: &[HardwareSolution],
        embedding: &Embedding,
    ) -> IsingResult<Self> {
        let total_chains = embedding.chains.len();
        let mut broken_chains = Vec::new();
        let mut break_counts: HashMap<usize, usize> = HashMap::new();

        for hw_solution in hardware_solutions {
            let mut breaks_in_solution = 0;

            for (&var, chain) in &embedding.chains {
                if chain.len() > 1 {
                    let first_spin = hw_solution.spins[chain[0]];
                    let is_broken = chain[1..].iter()
                        .any(|&q| hw_solution.spins[q] != first_spin);

                    if is_broken {
                        breaks_in_solution += 1;
                        *break_counts.entry(var).or_insert(0) += 1;
                    }
                }
            }

            broken_chains.push(breaks_in_solution);
        }

        // Calculate statistics
        let total_breaks: usize = broken_chains.iter().sum();
        let break_rate = if hardware_solutions.is_empty() || total_chains == 0 {
            0.0
        } else {
            total_breaks as f64 / (hardware_solutions.len() * total_chains) as f64
        };

        // Find most frequently broken variables
        let mut frequent_breaks: Vec<(usize, usize)> = break_counts.into_iter().collect();
        frequent_breaks.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
        frequent_breaks.truncate(10); // Keep top 10

        Ok(Self {
            total_chains,
            broken_chains,
            break_rate,
            frequent_breaks,
        })
    }

    /// Get recommendations based on statistics
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.break_rate > 0.5 {
            recommendations.push(
                "High chain break rate detected. Consider increasing chain strength.".to_string()
            );
        }

        if self.break_rate > 0.2 {
            recommendations.push(
                "Moderate chain breaks. Try optimizing embedding or chain strength.".to_string()
            );
        }

        if !self.frequent_breaks.is_empty() {
            let vars: Vec<String> = self.frequent_breaks.iter()
                .take(3)
                .map(|(var, _)| var.to_string())
                .collect();
            recommendations.push(format!(
                "Variables {} frequently have broken chains. Check embedding quality.",
                vars.join(", ")
            ));
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_majority_vote_resolution() {
        let mut embedding = Embedding::new();
        embedding.add_chain(0, vec![0, 1, 2]).unwrap();
        embedding.add_chain(1, vec![3, 4, 5]).unwrap();

        let hw_solution = HardwareSolution {
            spins: vec![1, 1, -1, -1, -1, -1], // First chain: 2 vs 1, second: unanimous
            energy: -1.0,
            occurrences: 1,
        };

        let resolver = ChainBreakResolver::default();
        let resolved = resolver.resolve_solution(&hw_solution, &embedding, None).unwrap();

        assert_eq!(resolved.logical_spins, vec![1, -1]);
        assert_eq!(resolved.chain_breaks, 1); // First chain is broken
    }

    #[test]
    fn test_chain_strength_optimizer() {
        let mut problem = LogicalProblem::new(3);
        problem.linear = vec![1.0, -0.5, 0.0];
        problem.quadratic.insert((0, 1), -2.0);
        problem.quadratic.insert((1, 2), 1.5);

        let optimizer = ChainStrengthOptimizer::default();
        let strength = optimizer.find_optimal_strength(&problem);

        // Should be around the median of coefficients
        assert!(strength > 0.5 && strength < 5.0);
    }

    #[test]
    fn test_chain_break_stats() {
        let mut embedding = Embedding::new();
        embedding.add_chain(0, vec![0, 1]).unwrap();
        embedding.add_chain(1, vec![2, 3]).unwrap();

        let solutions = vec![
            HardwareSolution {
                spins: vec![1, 1, -1, -1], // No breaks
                energy: -1.0,
                occurrences: 1,
            },
            HardwareSolution {
                spins: vec![1, -1, -1, -1], // First chain broken
                energy: -0.5,
                occurrences: 1,
            },
        ];

        let stats = ChainBreakStats::analyze(&solutions, &embedding).unwrap();
        
        assert_eq!(stats.total_chains, 2);
        assert_eq!(stats.broken_chains, vec![0, 1]);
        assert_eq!(stats.break_rate, 0.25); // 1 break out of 4 chain instances
    }
}