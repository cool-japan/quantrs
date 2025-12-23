//! Domain decomposition methods for parallel optimization

use super::types::*;
use crate::sampler::{SampleResult, Sampler};
use scirs2_core::ndarray::Array2;
#[cfg(feature = "parallel")]
use scirs2_core::parallel_ops::*;
use std::collections::HashMap;

/// Domain decomposition solver
pub struct DomainDecomposer<S: Sampler> {
    /// Base sampler for subdomains
    base_sampler: S,
    /// Decomposition strategy
    strategy: DecompositionStrategy,
    /// Coordination method
    coordination: CoordinationStrategy,
    /// Maximum iterations for coordination
    max_coordination_iterations: usize,
    /// Convergence tolerance
    convergence_tolerance: f64,
    /// Overlap between domains
    overlap_size: usize,
}

impl<S: Sampler + Send + Sync + Clone> DomainDecomposer<S> {
    /// Create new domain decomposer
    pub const fn new(base_sampler: S) -> Self {
        Self {
            base_sampler,
            strategy: DecompositionStrategy::Schwarz,
            coordination: CoordinationStrategy::ADMM { rho: 1.0 },
            max_coordination_iterations: 50,
            convergence_tolerance: 1e-4,
            overlap_size: 2,
        }
    }

    /// Set decomposition strategy
    pub const fn with_strategy(mut self, strategy: DecompositionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set coordination strategy
    pub const fn with_coordination(mut self, coordination: CoordinationStrategy) -> Self {
        self.coordination = coordination;
        self
    }

    /// Set maximum coordination iterations
    pub const fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_coordination_iterations = max_iterations;
        self
    }

    /// Solve using domain decomposition
    pub fn solve(
        &mut self,
        qubo: &Array2<f64>,
        var_map: &HashMap<String, usize>,
        shots: usize,
    ) -> Result<SampleResult, String> {
        // Decompose problem into domains
        let domains = self.decompose_into_domains(qubo, var_map)?;

        match self.strategy {
            DecompositionStrategy::Schwarz => self.schwarz_solve(&domains, shots),
            DecompositionStrategy::BlockJacobi => self.block_jacobi_solve(&domains, shots),
            DecompositionStrategy::AdditiveSchwarz => self.additive_schwarz_solve(&domains, shots),
            DecompositionStrategy::MultiplicativeSchwarz => {
                self.multiplicative_schwarz_solve(&domains, shots)
            }
        }
    }

    /// Decompose QUBO into overlapping domains
    fn decompose_into_domains(
        &self,
        qubo: &Array2<f64>,
        var_map: &HashMap<String, usize>,
    ) -> Result<Vec<Domain>, String> {
        let n = qubo.shape()[0];
        let num_domains = (n as f64 / 10.0).ceil() as usize; // Target ~10 variables per domain
        let domain_size = n / num_domains + self.overlap_size;

        let mut domains = Vec::new();
        let reverse_var_map: HashMap<usize, String> =
            var_map.iter().map(|(k, v)| (*v, k.clone())).collect();

        for domain_id in 0..num_domains {
            let start_idx = domain_id * (domain_size - self.overlap_size);
            let end_idx =
                ((domain_id + 1) * (domain_size - self.overlap_size) + self.overlap_size).min(n);

            if start_idx >= n {
                break;
            }

            let domain_indices: Vec<usize> = (start_idx..end_idx).collect();
            let variables: Vec<String> = domain_indices
                .iter()
                .filter_map(|&i| reverse_var_map.get(&i))
                .cloned()
                .collect();

            // Extract domain QUBO
            let domain_size_actual = domain_indices.len();
            let mut domain_qubo = Array2::zeros((domain_size_actual, domain_size_actual));

            for (i, &idx_i) in domain_indices.iter().enumerate() {
                for (j, &idx_j) in domain_indices.iter().enumerate() {
                    domain_qubo[[i, j]] = qubo[[idx_i, idx_j]];
                }
            }

            // Build domain variable map
            let mut domain_var_map = HashMap::new();
            for (i, var) in variables.iter().enumerate() {
                domain_var_map.insert(var.clone(), i);
            }

            // Identify boundary and internal variables
            let mut boundary_vars = Vec::new();
            let mut internal_vars = Vec::new();

            for (i, &global_idx) in domain_indices.iter().enumerate() {
                let is_boundary = self.is_boundary_variable(qubo, global_idx, &domain_indices);
                if is_boundary {
                    boundary_vars.push(i);
                } else {
                    internal_vars.push(i);
                }
            }

            domains.push(Domain {
                id: domain_id,
                variables,
                qubo: domain_qubo,
                var_map: domain_var_map,
                boundary_vars,
                internal_vars,
            });
        }

        Ok(domains)
    }

    /// Check if variable is on domain boundary
    fn is_boundary_variable(
        &self,
        qubo: &Array2<f64>,
        var_idx: usize,
        domain_indices: &[usize],
    ) -> bool {
        let n = qubo.shape()[0];

        // Check if variable has connections outside the domain
        for j in 0..n {
            if !domain_indices.contains(&j) && qubo[[var_idx, j]].abs() > 1e-10 {
                return true;
            }
        }

        false
    }

    /// Schwarz alternating method
    fn schwarz_solve(&mut self, domains: &[Domain], shots: usize) -> Result<SampleResult, String> {
        let mut coordination_state = CoordinationState {
            iteration: 0,
            lagrange_multipliers: None,
            consensus_variables: None,
            convergence_tolerance: self.convergence_tolerance,
            max_iterations: self.max_coordination_iterations,
        };

        // Initialize coordination state
        self.initialize_coordination_state(&mut coordination_state, domains)?;

        for iteration in 0..self.max_coordination_iterations {
            coordination_state.iteration = iteration;

            // Solve all subdomains in parallel
            let subdomain_solutions =
                self.solve_subdomains_parallel(domains, &coordination_state, shots)?;

            // Update coordination variables
            let converged =
                self.update_coordination(&mut coordination_state, &subdomain_solutions, domains)?;

            if converged {
                break;
            }
        }

        // Integrate final solution
        self.integrate_solutions(domains, &coordination_state)
    }

    /// Block Jacobi method (parallel subdomain solving)
    fn block_jacobi_solve(
        &mut self,
        domains: &[Domain],
        shots: usize,
    ) -> Result<SampleResult, String> {
        let mut coordination_state = CoordinationState {
            iteration: 0,
            lagrange_multipliers: None,
            consensus_variables: None,
            convergence_tolerance: self.convergence_tolerance,
            max_iterations: self.max_coordination_iterations,
        };

        self.initialize_coordination_state(&mut coordination_state, domains)?;

        for iteration in 0..self.max_coordination_iterations {
            coordination_state.iteration = iteration;

            // Solve all subdomains simultaneously
            let subdomain_solutions =
                self.solve_subdomains_parallel(domains, &coordination_state, shots)?;

            // Update all coordination variables simultaneously
            let converged =
                self.update_coordination(&mut coordination_state, &subdomain_solutions, domains)?;

            if converged {
                break;
            }
        }

        self.integrate_solutions(domains, &coordination_state)
    }

    /// Additive Schwarz method
    fn additive_schwarz_solve(
        &mut self,
        domains: &[Domain],
        shots: usize,
    ) -> Result<SampleResult, String> {
        // Similar to block Jacobi but with additive updates
        self.block_jacobi_solve(domains, shots)
    }

    /// Multiplicative Schwarz method
    fn multiplicative_schwarz_solve(
        &mut self,
        domains: &[Domain],
        shots: usize,
    ) -> Result<SampleResult, String> {
        let mut coordination_state = CoordinationState {
            iteration: 0,
            lagrange_multipliers: None,
            consensus_variables: None,
            convergence_tolerance: self.convergence_tolerance,
            max_iterations: self.max_coordination_iterations,
        };

        self.initialize_coordination_state(&mut coordination_state, domains)?;

        for iteration in 0..self.max_coordination_iterations {
            coordination_state.iteration = iteration;

            // Solve subdomains sequentially with immediate updates
            for domain in domains {
                let solution = self.solve_single_subdomain(domain, &coordination_state, shots)?;

                // Update coordination immediately after each subdomain
                self.update_coordination_single(&mut coordination_state, &solution, domain)?;
            }

            // Check convergence
            if self.check_convergence(&coordination_state) {
                break;
            }
        }

        self.integrate_solutions(domains, &coordination_state)
    }

    /// Initialize coordination state
    fn initialize_coordination_state(
        &self,
        state: &mut CoordinationState,
        domains: &[Domain],
    ) -> Result<(), String> {
        if let CoordinationStrategy::ADMM { .. } = &self.coordination {
            let mut lagrange_multipliers = HashMap::new();
            let mut consensus_variables = HashMap::new();

            // Initialize Lagrange multipliers and consensus variables
            for domain in domains {
                for &boundary_var in &domain.boundary_vars {
                    lagrange_multipliers.insert((domain.id, boundary_var), 0.0);
                    consensus_variables.insert(boundary_var, false);
                }
            }

            state.lagrange_multipliers = Some(lagrange_multipliers);
            state.consensus_variables = Some(consensus_variables);
        } else {
            // Other coordination strategies would be initialized here
        }

        Ok(())
    }

    /// Solve subdomains in parallel
    fn solve_subdomains_parallel(
        &self,
        domains: &[Domain],
        coordination: &CoordinationState,
        shots: usize,
    ) -> Result<Vec<SubdomainSolution>, String> {
        let solutions: Vec<_> = {
            #[cfg(feature = "parallel")]
            {
                domains
                    .par_iter()
                    .map(|domain| self.solve_single_subdomain(domain, coordination, shots))
                    .collect::<Result<Vec<_>, _>>()?
            }
            #[cfg(not(feature = "parallel"))]
            {
                domains
                    .iter()
                    .map(|domain| self.solve_single_subdomain(domain, coordination, shots))
                    .collect::<Result<Vec<_>, _>>()?
            }
        };

        Ok(solutions)
    }

    /// Solve single subdomain
    fn solve_single_subdomain(
        &self,
        domain: &Domain,
        coordination: &CoordinationState,
        shots: usize,
    ) -> Result<SubdomainSolution, String> {
        // Modify QUBO with coordination terms
        let modified_qubo = self.add_coordination_terms(domain, coordination)?;

        // Solve
        let results_vec = self
            .base_sampler
            .run_qubo(&(modified_qubo, domain.var_map.clone()), shots)
            .map_err(|e| format!("Sampler error: {e:?}"))?;

        // Take the best result (first one, since they're sorted by energy)
        let results = results_vec
            .into_iter()
            .next()
            .ok_or_else(|| "No solutions found for subdomain".to_string())?;

        Ok(SubdomainSolution {
            domain_id: domain.id,
            results,
        })
    }

    /// Add coordination terms to subdomain QUBO
    fn add_coordination_terms(
        &self,
        domain: &Domain,
        coordination: &CoordinationState,
    ) -> Result<Array2<f64>, String> {
        let mut modified_qubo = domain.qubo.clone();

        if let CoordinationStrategy::ADMM { rho } = &self.coordination {
            if let (Some(lagrange), Some(consensus)) = (
                &coordination.lagrange_multipliers,
                &coordination.consensus_variables,
            ) {
                // Add augmented Lagrangian terms
                for &boundary_var in &domain.boundary_vars {
                    if let Some(local_idx) = domain.var_map.values().find(|&&v| v == boundary_var) {
                        let lambda = lagrange.get(&(domain.id, boundary_var)).unwrap_or(&0.0);
                        let z = if *consensus.get(&boundary_var).unwrap_or(&false) {
                            1.0
                        } else {
                            0.0
                        };

                        // Add (rho/2)||x - z||^2 + lambda^T(x - z)
                        modified_qubo[[*local_idx, *local_idx]] += rho + 2.0 * lambda * (1.0 - z);
                    }
                }
            }
        }

        Ok(modified_qubo)
    }

    /// Update coordination variables
    fn update_coordination(
        &self,
        state: &mut CoordinationState,
        solutions: &[SubdomainSolution],
        domains: &[Domain],
    ) -> Result<bool, String> {
        match &self.coordination {
            CoordinationStrategy::ADMM { rho } => {
                self.update_admm_coordination(state, solutions, domains, *rho)
            }
            _ => Ok(false),
        }
    }

    /// Update ADMM coordination
    fn update_admm_coordination(
        &self,
        state: &mut CoordinationState,
        solutions: &[SubdomainSolution],
        domains: &[Domain],
        rho: f64,
    ) -> Result<bool, String> {
        if let (Some(lagrange), Some(consensus)) = (
            &mut state.lagrange_multipliers,
            &mut state.consensus_variables,
        ) {
            let mut max_residual = 0.0_f64;

            // Update consensus variables (z-update)
            for domain in domains {
                for &boundary_var in &domain.boundary_vars {
                    let mut total_value = 0.0;
                    let mut count = 0;

                    // Collect values from all domains containing this variable
                    for solution in solutions {
                        if let Some(sample) = solution.results.best_sample() {
                            if let Some(var_name) = domain
                                .var_map
                                .iter()
                                .find(|(_, &idx)| idx == boundary_var)
                                .map(|(name, _)| name)
                            {
                                if let Some(&value) = sample.get(var_name) {
                                    total_value += if value { 1.0 } else { 0.0 };
                                    count += 1;
                                }
                            }
                        }
                    }

                    // Consensus as majority vote
                    let new_consensus = if count > 0 {
                        total_value / count as f64 > 0.5
                    } else {
                        false
                    };

                    let old_consensus = *consensus.get(&boundary_var).unwrap_or(&false);
                    consensus.insert(boundary_var, new_consensus);

                    // Track residual
                    let residual = if new_consensus == old_consensus {
                        0.0
                    } else {
                        1.0
                    };
                    max_residual = max_residual.max(residual);
                }
            }

            // Update Lagrange multipliers (dual update)
            for domain in domains {
                for &boundary_var in &domain.boundary_vars {
                    if let Some(solution) = solutions.iter().find(|s| s.domain_id == domain.id) {
                        if let Some(sample) = solution.results.best_sample() {
                            if let Some(var_name) = domain
                                .var_map
                                .iter()
                                .find(|(_, &idx)| idx == boundary_var)
                                .map(|(name, _)| name)
                            {
                                if let Some(&x_value) = sample.get(var_name) {
                                    let z_value = *consensus.get(&boundary_var).unwrap_or(&false);
                                    let x_val = if x_value { 1.0 } else { 0.0 };
                                    let z_val = if z_value { 1.0 } else { 0.0 };

                                    let lambda_key = (domain.id, boundary_var);
                                    let old_lambda = *lagrange.get(&lambda_key).unwrap_or(&0.0);
                                    let new_lambda = rho.mul_add(x_val - z_val, old_lambda);
                                    lagrange.insert(lambda_key, new_lambda);
                                }
                            }
                        }
                    }
                }
            }

            // Check convergence
            Ok(max_residual < state.convergence_tolerance)
        } else {
            Ok(false)
        }
    }

    /// Update coordination for single domain (multiplicative Schwarz)
    fn update_coordination_single(
        &self,
        state: &mut CoordinationState,
        solution: &SubdomainSolution,
        domain: &Domain,
    ) -> Result<(), String> {
        // Simplified single domain update
        if let Some(consensus) = &mut state.consensus_variables {
            for &boundary_var in &domain.boundary_vars {
                if let Some(sample) = solution.results.best_sample() {
                    if let Some(var_name) = domain
                        .var_map
                        .iter()
                        .find(|(_, &idx)| idx == boundary_var)
                        .map(|(name, _)| name)
                    {
                        if let Some(&value) = sample.get(var_name) {
                            consensus.insert(boundary_var, value);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Check coordination convergence
    const fn check_convergence(&self, state: &CoordinationState) -> bool {
        // Simplified convergence check
        state.iteration >= self.max_coordination_iterations / 2
    }

    /// Integrate solutions from all subdomains
    fn integrate_solutions(
        &self,
        domains: &[Domain],
        coordination_state: &CoordinationState,
    ) -> Result<SampleResult, String> {
        // Use consensus variables as final solution
        if let Some(consensus) = &coordination_state.consensus_variables {
            let mut final_sample = HashMap::new();

            // Collect all variables from all domains
            for domain in domains {
                for (var_name, &local_idx) in &domain.var_map {
                    if let Some(&value) = consensus.get(&local_idx) {
                        final_sample.insert(var_name.clone(), value);
                    }
                }
            }

            // Create SampleResult with integrated solution
            // This is simplified - in practice would compute proper energy
            Ok(SampleResult {
                assignments: final_sample,
                energy: 0.0,
                occurrences: 1,
            })
        } else {
            Err("No consensus variables available".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::simulated_annealing::SASampler;
    use scirs2_core::ndarray::Array2;
    use std::collections::HashMap;

    #[test]
    fn test_domain_decomposer_creation() {
        let base_sampler = SASampler::new(None);
        let decomposer = DomainDecomposer::new(base_sampler);

        assert_eq!(decomposer.max_coordination_iterations, 50);
        assert_eq!(decomposer.overlap_size, 2);
    }

    #[test]
    fn test_domain_decomposition() {
        let base_sampler = SASampler::new(None);
        let decomposer = DomainDecomposer::new(base_sampler);

        // Create test QUBO
        let qubo =
            Array2::from_shape_vec((6, 6), (0..36).map(|x| x as f64 * 0.1).collect()).unwrap();

        let mut var_map = HashMap::new();
        for i in 0..6 {
            var_map.insert(format!("x{}", i), i);
        }

        let domains = decomposer.decompose_into_domains(&qubo, &var_map);
        assert!(domains.is_ok());

        let domains = domains.unwrap();
        assert!(!domains.is_empty());

        // Check that domains cover all variables
        let mut all_vars = std::collections::HashSet::new();
        for domain in &domains {
            for var in &domain.variables {
                all_vars.insert(var.clone());
            }
        }
        assert_eq!(all_vars.len(), 6);
    }

    #[test]
    fn test_boundary_variable_detection() {
        let base_sampler = SASampler::new(None);
        let decomposer = DomainDecomposer::new(base_sampler);

        // Create QUBO with clear structure
        let mut qubo = Array2::zeros((4, 4));
        qubo[[0, 1]] = 1.0; // Connection within domain
        qubo[[2, 3]] = 1.0; // Connection within domain
        qubo[[1, 2]] = 1.0; // Cross-domain connection

        let mut domain_indices = vec![0, 1];

        // Variable 1 should be boundary (connects to variable 2 outside domain)
        assert!(decomposer.is_boundary_variable(&qubo, 1, &domain_indices));
        // Variable 0 should not be boundary
        assert!(!decomposer.is_boundary_variable(&qubo, 0, &domain_indices));
    }
}
