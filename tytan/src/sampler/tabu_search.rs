//! Tabu Search Sampler for QUBO/HOBO problems
//!
//! Implements a classical Tabu Search metaheuristic for binary optimization.
//! Tabu Search uses a short-term memory (tabu list) to avoid revisiting
//! recently explored solutions, allowing escape from local optima.
//!
//! # Algorithm
//!
//! 1. Start from a random binary assignment
//! 2. Maintain a FIFO ring tabu list of recently-flipped variable indices
//! 3. Each iteration: compute ΔE for all single-bit flips using O(n) incremental formula
//! 4. Pick best non-tabu flip (aspiration: allow tabu if better than best-ever)
//! 5. Track best-ever assignment
//! 6. After `restart_threshold` non-improving iterations, restart from best + perturbation
//!
//! # Reference
//!
//! Glover, F. (1989). Tabu Search—Part I. ORSA Journal on Computing, 1(3), 190-206.

use scirs2_core::ndarray::{Array, ArrayD, Ix2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use std::collections::HashMap;

use super::{SampleResult, Sampler, SamplerError, SamplerResult};

/// Parameters for the Tabu Search algorithm
#[derive(Debug, Clone)]
pub struct TabuParams {
    /// Tabu list length (how long a flip is forbidden)
    pub tenure: usize,
    /// Maximum iterations per run
    pub max_iter: usize,
    /// Number of non-improving iterations before restart
    pub restart_threshold: usize,
    /// Allow tabu moves if they improve best-ever solution (aspiration criterion)
    pub aspiration: bool,
}

impl Default for TabuParams {
    fn default() -> Self {
        Self {
            tenure: 10,
            max_iter: 1000,
            restart_threshold: 200,
            aspiration: true,
        }
    }
}

/// Tabu Search Sampler
///
/// Uses the Tabu Search metaheuristic with short-term memory (tabu list)
/// to escape local optima in QUBO/HOBO problems.
///
/// # Example
///
/// ```rust,no_run
/// use quantrs2_tytan::sampler::{TabuSampler, Sampler};
/// use std::collections::HashMap;
/// use scirs2_core::ndarray::Array;
///
/// let mut q = Array::<f64, _>::zeros((3, 3));
/// q[[0,0]] = -1.0; q[[1,1]] = -1.0; q[[2,2]] = -1.0;
/// q[[0,1]] = 2.0;  q[[0,2]] = 2.0;  q[[1,2]] = 2.0;
///
/// let mut var_map = HashMap::new();
/// var_map.insert("x0".to_string(), 0);
/// var_map.insert("x1".to_string(), 1);
/// var_map.insert("x2".to_string(), 2);
///
/// let sampler = TabuSampler::new().with_seed(42);
/// let results = sampler.run_qubo(&(q, var_map), 50).expect("Tabu search failed");
/// println!("Best energy: {}", results[0].energy);
/// ```
#[derive(Debug, Clone)]
pub struct TabuSampler {
    /// Random number generator seed
    pub seed: Option<u64>,
    /// Tabu search parameters
    pub params: TabuParams,
}

impl TabuSampler {
    /// Create a new Tabu Search sampler with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self {
            seed: None,
            params: TabuParams::default(),
        }
    }

    /// Set the random seed for reproducibility
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the tabu tenure (how many iterations a flip is forbidden)
    #[must_use]
    pub fn with_tenure(mut self, tenure: usize) -> Self {
        self.params.tenure = tenure;
        self
    }

    /// Set the maximum number of iterations
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.params.max_iter = max_iter;
        self
    }

    /// Set the restart threshold (non-improving iterations before restart)
    #[must_use]
    pub fn with_restart_threshold(mut self, restart_threshold: usize) -> Self {
        self.params.restart_threshold = restart_threshold;
        self
    }

    /// Enable or disable the aspiration criterion
    #[must_use]
    pub fn with_aspiration(mut self, aspiration: bool) -> Self {
        self.params.aspiration = aspiration;
        self
    }

    /// Compute QUBO energy: E(x) = sum_{i,j} Q[i,j] * x[i] * x[j]
    ///
    /// Uses the full matrix (symmetric or upper-triangular).
    fn compute_qubo_energy(q_matrix: &[f64], state: &[bool], n: usize) -> f64 {
        let mut energy = 0.0;
        for i in 0..n {
            if !state[i] {
                continue;
            }
            for j in 0..n {
                if state[j] {
                    energy += q_matrix[i * n + j];
                }
            }
        }
        energy
    }

    /// Initialize the influence vector g[i] = sum_j (Q[i,j] + Q[j,i]) * x[j] - Q[i,i] * x[i]
    ///
    /// g[i] represents the effective field on bit i:
    /// ΔE from flipping bit i = (1 - 2*x[i]) * g[i]
    /// where g[i] = Q[i,i] + sum_{j!=i} (Q[i,j] + Q[j,i]) * x[j]
    fn compute_influence_vector(q_matrix: &[f64], state: &[bool], n: usize) -> Vec<f64> {
        let mut g = vec![0.0f64; n];
        for i in 0..n {
            // Diagonal term
            g[i] += q_matrix[i * n + i];
            // Off-diagonal terms: sum over j != i of (Q[i,j] + Q[j,i]) * x[j]
            for j in 0..n {
                if j != i && state[j] {
                    g[i] += q_matrix[i * n + j] + q_matrix[j * n + i];
                }
            }
        }
        g
    }

    /// Update the influence vector after flipping bit k
    ///
    /// After flipping x[k]: x[k] changes by delta = new_x[k] - old_x[k] in {-1, +1}.
    /// g[i] += (Q[i,k] + Q[k,i]) * delta  for i != k
    /// g[k]: recompute fully or adjust: new g[k] remains same formula with new x[k] implicitly
    fn update_influence_vector(
        g: &mut [f64],
        q_matrix: &[f64],
        k: usize,
        flipped_to: bool,
        n: usize,
    ) {
        let delta = if flipped_to { 1.0 } else { -1.0 };
        for i in 0..n {
            if i != k {
                g[i] += (q_matrix[i * n + k] + q_matrix[k * n + i]) * delta;
            }
        }
        // g[k] itself: diagonal term is constant, off-diagonals depend on other bits (not x[k])
        // No change needed for g[k] from flipping k itself
    }

    /// Run a single tabu search on a QUBO problem encoded as flat matrix
    fn run_single_qubo(
        &self,
        q_matrix: &[f64],
        n_vars: usize,
        rng: &mut StdRng,
    ) -> (Vec<bool>, f64) {
        if n_vars == 0 {
            return (vec![], 0.0);
        }

        let tenure = self.params.tenure;
        let max_iter = self.params.max_iter;
        let restart_threshold = self.params.restart_threshold;
        let aspiration = self.params.aspiration;

        // Initialize random state
        let mut state: Vec<bool> = (0..n_vars).map(|_| rng.random_bool(0.5)).collect();
        let mut energy = Self::compute_qubo_energy(q_matrix, &state, n_vars);
        let mut g = Self::compute_influence_vector(q_matrix, &state, n_vars);

        // Best ever tracking
        let mut best_state = state.clone();
        let mut best_energy = energy;

        // Tabu list as a ring buffer: tabu_until[i] = iteration at which bit i becomes non-tabu
        let mut tabu_until = vec![0usize; n_vars];
        let mut non_improving_count = 0usize;
        let mut iter = 0usize;

        while iter < max_iter {
            // Find best non-tabu flip (or aspiration-permitted tabu flip)
            let mut best_flip_idx = None;
            let mut best_flip_delta = f64::INFINITY;

            for i in 0..n_vars {
                let delta_e = (1.0 - 2.0 * if state[i] { 1.0 } else { 0.0 }) * g[i];
                let is_tabu = tabu_until[i] > iter;

                if !is_tabu {
                    if delta_e < best_flip_delta {
                        best_flip_delta = delta_e;
                        best_flip_idx = Some(i);
                    }
                } else if aspiration && (energy + delta_e < best_energy) {
                    // Aspiration: allow tabu if it improves best-ever
                    if delta_e < best_flip_delta {
                        best_flip_delta = delta_e;
                        best_flip_idx = Some(i);
                    }
                }
            }

            let flip_idx = match best_flip_idx {
                Some(idx) => idx,
                None => {
                    // All moves tabu — force best tabu move
                    let mut forced_best = 0;
                    let mut forced_best_delta = f64::INFINITY;
                    for i in 0..n_vars {
                        let delta_e = (1.0 - 2.0 * if state[i] { 1.0 } else { 0.0 }) * g[i];
                        if delta_e < forced_best_delta {
                            forced_best_delta = delta_e;
                            forced_best = i;
                        }
                    }
                    forced_best
                }
            };

            // Apply the flip
            let new_x = !state[flip_idx];
            energy += best_flip_delta;
            Self::update_influence_vector(&mut g, q_matrix, flip_idx, new_x, n_vars);
            state[flip_idx] = new_x;

            // Update tabu list
            tabu_until[flip_idx] = iter + tenure + 1;

            // Track best
            if energy < best_energy - 1e-12 {
                best_energy = energy;
                best_state = state.clone();
                non_improving_count = 0;
            } else {
                non_improving_count += 1;
            }

            // Restart mechanism
            if non_improving_count >= restart_threshold {
                // Restart from best + random perturbation
                state = best_state.clone();
                energy = best_energy;

                let n_perturb = (n_vars as f64 / 10.0).ceil() as usize;
                for _ in 0..n_perturb {
                    let idx = rng.random_range(0..n_vars);
                    state[idx] = !state[idx];
                }
                energy = Self::compute_qubo_energy(q_matrix, &state, n_vars);
                g = Self::compute_influence_vector(q_matrix, &state, n_vars);
                non_improving_count = 0;
                // Clear tabu list on restart
                tabu_until.fill(0);
            }

            iter += 1;
        }

        (best_state, best_energy)
    }

    /// Evaluate energy for a higher-order binary optimization tensor
    fn evaluate_hobo_energy<D>(tensor: &Array<f64, D>, state: &[bool], n_vars: usize) -> f64
    where
        D: scirs2_core::ndarray::Dimension + 'static,
    {
        // Convert to dynamic dimensionality to allow slice-based indexing
        let dyn_tensor: ArrayD<f64> = tensor.to_owned().into_dyn();
        Self::evaluate_hobo_energy_dyn(&dyn_tensor, state, n_vars)
    }

    /// Evaluate energy for a dynamic-dimension tensor
    fn evaluate_hobo_energy_dyn(tensor: &ArrayD<f64>, state: &[bool], n_vars: usize) -> f64 {
        let mut energy = 0.0;
        let shape = tensor.shape();
        let ndim = shape.len();

        if ndim == 2 {
            let n0 = shape[0].min(n_vars);
            let n1 = shape[1].min(n_vars);
            for i in 0..n0 {
                if !state[i] {
                    continue;
                }
                for j in 0..n1 {
                    if state[j] {
                        if let Some(&v) = tensor.get([i, j].as_slice()) {
                            energy += v;
                        }
                    }
                }
            }
        } else if ndim == 3 {
            let n0 = shape[0].min(n_vars);
            let n1 = shape[1].min(n_vars);
            let n2 = shape[2].min(n_vars);
            for i in 0..n0 {
                if !state[i] {
                    continue;
                }
                for j in 0..n1 {
                    if !state[j] {
                        continue;
                    }
                    for k in 0..n2 {
                        if state[k] {
                            if let Some(&v) = tensor.get([i, j, k].as_slice()) {
                                energy += v;
                            }
                        }
                    }
                }
            }
        }
        // Higher order: fall through with 0 contribution
        energy
    }

    /// Run a single tabu search on a HOBO problem
    fn run_single_hobo<D>(
        &self,
        tensor: &Array<f64, D>,
        n_vars: usize,
        rng: &mut StdRng,
    ) -> (Vec<bool>, f64)
    where
        D: scirs2_core::ndarray::Dimension + 'static,
    {
        // Convert once to dynamic dimension for repeated indexing
        let dyn_tensor: ArrayD<f64> = tensor.to_owned().into_dyn();
        self.run_single_hobo_dyn(&dyn_tensor, n_vars, rng)
    }

    /// Run a single tabu search on a dynamic-dimension HOBO tensor
    fn run_single_hobo_dyn(
        &self,
        tensor: &ArrayD<f64>,
        n_vars: usize,
        rng: &mut StdRng,
    ) -> (Vec<bool>, f64) {
        if n_vars == 0 {
            return (vec![], 0.0);
        }

        let tenure = self.params.tenure;
        let max_iter = self.params.max_iter;
        let restart_threshold = self.params.restart_threshold;
        let aspiration = self.params.aspiration;

        let mut state: Vec<bool> = (0..n_vars).map(|_| rng.random_bool(0.5)).collect();
        let mut energy = Self::evaluate_hobo_energy_dyn(tensor, &state, n_vars);

        let mut best_state = state.clone();
        let mut best_energy = energy;

        let mut tabu_until = vec![0usize; n_vars];
        let mut non_improving_count = 0usize;

        for iter in 0..max_iter {
            let mut best_flip_idx = None;
            let mut best_flip_delta = f64::INFINITY;

            for i in 0..n_vars {
                let is_tabu = tabu_until[i] > iter;
                state[i] = !state[i];
                let new_energy = Self::evaluate_hobo_energy_dyn(tensor, &state, n_vars);
                let delta_e = new_energy - energy;
                state[i] = !state[i]; // Restore

                if (!is_tabu || (aspiration && (energy + delta_e < best_energy)))
                    && delta_e < best_flip_delta
                {
                    best_flip_delta = delta_e;
                    best_flip_idx = Some(i);
                }
            }

            let flip_idx = match best_flip_idx {
                Some(idx) => idx,
                None => {
                    // All tabu — pick best overall
                    let mut forced_best = 0;
                    let mut forced_best_delta = f64::INFINITY;
                    for i in 0..n_vars {
                        state[i] = !state[i];
                        let new_energy = Self::evaluate_hobo_energy_dyn(tensor, &state, n_vars);
                        let delta_e = new_energy - energy;
                        state[i] = !state[i];
                        if delta_e < forced_best_delta {
                            forced_best_delta = delta_e;
                            forced_best = i;
                        }
                    }
                    forced_best
                }
            };

            state[flip_idx] = !state[flip_idx];
            energy += best_flip_delta;
            tabu_until[flip_idx] = iter + tenure + 1;

            if energy < best_energy - 1e-12 {
                best_energy = energy;
                best_state = state.clone();
                non_improving_count = 0;
            } else {
                non_improving_count += 1;
            }

            if non_improving_count >= restart_threshold {
                state = best_state.clone();
                energy = best_energy;
                let n_perturb = (n_vars as f64 / 10.0).ceil() as usize;
                for _ in 0..n_perturb {
                    let idx = rng.random_range(0..n_vars);
                    state[idx] = !state[idx];
                }
                energy = Self::evaluate_hobo_energy_dyn(tensor, &state, n_vars);
                non_improving_count = 0;
                tabu_until.fill(0);
            }
        }

        (best_state, best_energy)
    }

    /// Run generic sampler on any array (dispatches by dimension)
    fn run_generic<D>(
        &self,
        matrix_or_tensor: &Array<f64, D>,
        var_map: &HashMap<String, usize>,
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>>
    where
        D: scirs2_core::ndarray::Dimension + 'static,
    {
        let shots = shots.max(1);
        let n_vars = var_map.len();
        if n_vars == 0 {
            return Err(SamplerError::InvalidParameter(
                "Variable map is empty".to_string(),
            ));
        }

        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

        let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

        if matrix_or_tensor.ndim() == 2 {
            // QUBO path: use O(n) incremental ΔE
            let q2 = matrix_or_tensor
                .to_owned()
                .into_dimensionality::<Ix2>()
                .map_err(|e| SamplerError::InvalidParameter(format!("Array cast error: {e}")))?;
            let n = q2.dim().0;
            if n != q2.dim().1 {
                return Err(SamplerError::InvalidParameter(
                    "QUBO matrix must be square".to_string(),
                ));
            }
            let q_flat: Vec<f64> = q2.as_slice().ok_or_else(|| {
                SamplerError::InvalidParameter("Non-contiguous QUBO matrix".to_string())
            })?.to_vec();

            for shot_idx in 0..shots {
                let seed = match self.seed {
                    Some(s) => s.wrapping_add(shot_idx as u64),
                    None => {
                        let mut rng_tmp = thread_rng();
                        rng_tmp.random()
                    }
                };
                let mut rng = StdRng::seed_from_u64(seed);
                let (best_state, best_energy) =
                    self.run_single_qubo(&q_flat, n, &mut rng);

                let entry = solution_counts
                    .entry(best_state)
                    .or_insert((best_energy, 0));
                entry.1 += 1;
            }
        } else {
            // HOBO path: brute-force ΔE evaluation
            for shot_idx in 0..shots {
                let seed = match self.seed {
                    Some(s) => s.wrapping_add(shot_idx as u64),
                    None => {
                        let mut rng_tmp = thread_rng();
                        rng_tmp.random()
                    }
                };
                let mut rng = StdRng::seed_from_u64(seed);
                let (best_state, best_energy) =
                    self.run_single_hobo(matrix_or_tensor, n_vars, &mut rng);

                let entry = solution_counts
                    .entry(best_state)
                    .or_insert((best_energy, 0));
                entry.1 += 1;
            }
        }

        // Convert to (state, SampleResult) pairs, sort by (energy, state) for determinism
        let mut pairs: Vec<(Vec<bool>, SampleResult)> = solution_counts
            .into_iter()
            .map(|(state, (energy, count))| {
                let assignments: HashMap<String, bool> = state
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &value)| {
                        idx_to_var.get(&idx).map(|name| (name.clone(), value))
                    })
                    .collect();
                let result = SampleResult {
                    assignments,
                    energy,
                    occurrences: count,
                };
                (state, result)
            })
            .collect();

        // Sort by energy ascending, then by state vector for deterministic tie-breaking
        pairs.sort_by(|(state_a, a), (state_b, b)| {
            a.energy
                .partial_cmp(&b.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| state_a.cmp(state_b))
        });

        let results: Vec<SampleResult> = pairs.into_iter().map(|(_, r)| r).collect();
        Ok(results)
    }
}

impl Sampler for TabuSampler {
    fn run_qubo(
        &self,
        qubo: &(Array<f64, Ix2>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        self.run_generic(&qubo.0, &qubo.1, shots)
    }

    fn run_hobo(
        &self,
        hobo: &(ArrayD<f64>, HashMap<String, usize>),
        shots: usize,
    ) -> SamplerResult<Vec<SampleResult>> {
        self.run_generic(&hobo.0, &hobo.1, shots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build K3 Max-Cut QUBO matrix.
    ///
    /// Max-Cut on K3 (triangle): maximize cut edges = maximize sum_{(i,j) in E} x_i(1-x_j) + x_j(1-x_i)
    /// = maximize sum x_i + x_j - 2*x_i*x_j
    /// QUBO minimization form: E = sum_{(i,j)} (2*x_i*x_j - x_i - x_j) + const
    /// With 3 edges: E = 2*(x0*x1 + x0*x2 + x1*x2) - 2*(x0 + x1 + x2)
    /// Optimal: one vertex on one side → 2 cut edges → E = -2
    fn build_k3_maxcut_qubo() -> (Array2<f64>, HashMap<String, usize>) {
        let mut q = Array2::<f64>::zeros((3, 3));
        // Linear terms: -2 for each variable
        q[[0, 0]] = -2.0;
        q[[1, 1]] = -2.0;
        q[[2, 2]] = -2.0;
        // Quadratic terms: +2 for each pair
        q[[0, 1]] = 2.0;
        q[[0, 2]] = 2.0;
        q[[1, 2]] = 2.0;
        // Upper triangular only (symmetric handled by energy function)

        let mut var_map = HashMap::new();
        var_map.insert("x0".to_string(), 0);
        var_map.insert("x1".to_string(), 1);
        var_map.insert("x2".to_string(), 2);

        (q, var_map)
    }

    #[test]
    fn test_tabu_3var_maxcut() {
        let (q, var_map) = build_k3_maxcut_qubo();
        let sampler = TabuSampler::new()
            .with_seed(42)
            .with_max_iter(200)
            .with_tenure(5);

        let results = sampler
            .run_qubo(&(q, var_map), 50)
            .expect("Tabu run_qubo failed");

        assert!(!results.is_empty(), "Expected non-empty results");
        // Optimal energy for K3 Max-Cut QUBO is -2 (2 edges cut)
        let best_energy = results[0].energy;
        assert!(
            best_energy <= -2.0 + 1e-9,
            "Expected optimal energy <= -2.0, got {best_energy}"
        );
    }

    #[test]
    fn test_tabu_determinism() {
        let (q, var_map) = build_k3_maxcut_qubo();

        let s1 = TabuSampler::new().with_seed(42);
        let s2 = TabuSampler::new().with_seed(42);

        let r1 = s1
            .run_qubo(&(q.clone(), var_map.clone()), 10)
            .expect("Run 1 failed");
        let r2 = s2
            .run_qubo(&(q, var_map), 10)
            .expect("Run 2 failed");

        assert_eq!(r1.len(), r2.len(), "Result lengths differ");
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert!(
                (a.energy - b.energy).abs() < 1e-12,
                "Energies differ: {} vs {}",
                a.energy,
                b.energy
            );
            assert_eq!(
                a.assignments, b.assignments,
                "Assignments differ for same seed"
            );
        }
    }

    #[test]
    fn test_tabu_hobo_smoke() {
        // Simple 2D HOBO (treated as QUBO)
        let mut q = Array2::<f64>::zeros((2, 2));
        q[[0, 0]] = -1.0;
        q[[1, 1]] = -1.0;

        let mut var_map = HashMap::new();
        var_map.insert("a".to_string(), 0);
        var_map.insert("b".to_string(), 1);

        let sampler = TabuSampler::new().with_seed(7);
        let q_dyn = q.into_dyn();
        let results = sampler
            .run_hobo(&(q_dyn, var_map), 10)
            .expect("HOBO run failed");
        assert!(!results.is_empty());
        assert!(results[0].energy <= -2.0 + 1e-9);
    }

    #[test]
    fn test_tabu_aspiration() {
        let (q, var_map) = build_k3_maxcut_qubo();
        // Test with and without aspiration — both should find optimum
        let with_aspiration = TabuSampler::new()
            .with_seed(100)
            .with_aspiration(true)
            .with_tenure(100) // Very high tenure forces aspiration
            .with_max_iter(500);
        let without_aspiration = TabuSampler::new()
            .with_seed(100)
            .with_aspiration(false)
            .with_max_iter(500);

        let r1 = with_aspiration
            .run_qubo(&(q.clone(), var_map.clone()), 20)
            .expect("With aspiration failed");
        let r2 = without_aspiration
            .run_qubo(&(q, var_map), 20)
            .expect("Without aspiration failed");

        assert!(!r1.is_empty());
        assert!(!r2.is_empty());
        // At least one should find optimum
        assert!(r1[0].energy <= -2.0 + 1e-9 || r2[0].energy <= -2.0 + 1e-9);
    }
}
