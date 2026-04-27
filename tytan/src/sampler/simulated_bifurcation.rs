//! Simulated Bifurcation (SB) Sampler for QUBO problems
//!
//! Implements Toshiba's Simulated Bifurcation (SB) algorithm
//! (Goto, Tatsumura, Dixon 2019) for solving QUBO/Ising problems.
//!
//! # Algorithm Variants
//!
//! - **Ballistic SB (bSB)**: Continuous oscillator network with soft clipping
//! - **Discrete SB (dSB)**: Uses sign(x) instead of x in the coupling term
//!
//! # QUBO → Ising Conversion
//!
//! Given QUBO matrix Q (upper-triangular or symmetric), the Ising parameters are:
//! - Q_sym[i,j] = (Q[i,j] + Q[j,i]) / 2   (for i ≠ j)
//! - J[i,j] = Q_sym[i,j] / 4                (off-diagonal coupling)
//! - h[i] = Q[i,i]/2 + Σ_{j≠i} Q_sym[i,j]/2  (local field)
//!
//! # SB Dynamics (Symplectic Euler)
//!
//! For each time step t:
//! - a(t) = a_init + (a_final - a_init) * t / T
//! - bSB: y_i ← y_i + (-a(t)*x_i - c0*Σ_j J_ij*x_j - h_i) * dt
//! - dSB: y_i ← y_i + (-a(t)*x_i - c0*Σ_j J_ij*sign(x_j) - h_i) * dt
//! - x_i ← x_i + y_i * dt
//! - Clip: if |x_i| > 1: x_i = sign(x_i), y_i = 0
//!
//! # Reference
//!
//! Goto, H., Tatsumura, K., & Dixon, A. R. (2019). Combinatorial optimization
//! by simulating adiabatic bifurcations in nonlinear Hamiltonian systems.
//! Science Advances, 5(4), eaav2372.

use scirs2_core::ndarray::{Array, ArrayD, Ix2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use std::collections::HashMap;

use super::{SampleResult, Sampler, SamplerError, SamplerResult};

/// Variant of Simulated Bifurcation algorithm
#[derive(Debug, Clone, PartialEq)]
pub enum SBVariant {
    /// Ballistic SB (bSB): uses x_i in the coupling term (continuous)
    Ballistic,
    /// Discrete SB (dSB): uses sign(x_i) in the coupling term
    Discrete,
}

/// Parameters for the Simulated Bifurcation algorithm
#[derive(Debug, Clone)]
pub struct SBParams {
    /// Algorithm variant (Ballistic or Discrete)
    pub variant: SBVariant,
    /// Symplectic Euler time step size
    pub dt: f64,
    /// Coupling scale factor (default: 0.5, adjusted internally if 0.0)
    pub c0: f64,
    /// Initial pump parameter
    pub a_init: f64,
    /// Final pump parameter
    pub a_final: f64,
    /// Number of integration time steps
    pub time_steps: usize,
}

impl Default for SBParams {
    fn default() -> Self {
        Self {
            variant: SBVariant::Discrete,
            dt: 0.5,
            c0: 0.5,
            a_init: 0.0,
            a_final: 1.0,
            time_steps: 1000,
        }
    }
}

/// Simulated Bifurcation Sampler
///
/// Solves QUBO/Ising problems using Toshiba's Simulated Bifurcation algorithm.
/// The algorithm simulates the adiabatic bifurcation of a nonlinear oscillator
/// network to find low-energy configurations.
///
/// # Example
///
/// ```rust,no_run
/// use quantrs2_tytan::sampler::{SBSampler, SBVariant, Sampler};
/// use std::collections::HashMap;
/// use scirs2_core::ndarray::Array;
///
/// // K4 Max-Cut QUBO
/// let mut q = Array::<f64, _>::zeros((4, 4));
/// for i in 0..4 {
///     q[[i,i]] = -3.0;
///     for j in (i+1)..4 {
///         q[[i,j]] = 2.0;
///     }
/// }
///
/// let mut var_map = HashMap::new();
/// for i in 0..4 {
///     var_map.insert(format!("x{i}"), i);
/// }
///
/// let sampler = SBSampler::new()
///     .with_seed(42)
///     .with_variant(SBVariant::Discrete)
///     .with_time_steps(1000);
///
/// let results = sampler.run_qubo(&(q, var_map), 20).expect("SB sampler failed");
/// println!("Best energy: {}", results[0].energy);
/// ```
#[derive(Debug, Clone)]
pub struct SBSampler {
    /// Random number generator seed
    pub seed: Option<u64>,
    /// SB algorithm parameters
    pub params: SBParams,
}

impl SBSampler {
    /// Create a new Simulated Bifurcation sampler with default parameters
    #[must_use]
    pub fn new() -> Self {
        Self {
            seed: None,
            params: SBParams::default(),
        }
    }

    /// Set the random seed for reproducibility
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the SB variant (Ballistic or Discrete)
    #[must_use]
    pub fn with_variant(mut self, variant: SBVariant) -> Self {
        self.params.variant = variant;
        self
    }

    /// Set the number of time steps
    #[must_use]
    pub fn with_time_steps(mut self, time_steps: usize) -> Self {
        self.params.time_steps = time_steps;
        self
    }

    /// Set the time step size
    #[must_use]
    pub fn with_dt(mut self, dt: f64) -> Self {
        self.params.dt = dt;
        self
    }

    /// Set the coupling scale factor
    #[must_use]
    pub fn with_c0(mut self, c0: f64) -> Self {
        self.params.c0 = c0;
        self
    }

    /// Set the pump parameter range
    #[must_use]
    pub fn with_pump_range(mut self, a_init: f64, a_final: f64) -> Self {
        self.params.a_init = a_init;
        self.params.a_final = a_final;
        self
    }

    /// Convert QUBO matrix to symmetrized Ising parameters (J, h)
    ///
    /// Given QUBO E(x) = x^T Q x, substituting x_i = (1 + s_i) / 2:
    ///
    /// E = const + Σ_i h[i] s_i + (1/4) Σ_{i≠j} Q_sym[i,j] s_i s_j
    ///
    /// where:
    /// - Q_sym[i,j] = (Q[i,j] + Q[j,i]) / 2   (symmetrized coupling)
    /// - h[i] = Q[i,i]/2 + Σ_{j≠i} Q_sym[i,j]/2   (linear field)
    ///
    /// For the SB dynamics, we use the coupling matrix J (off-diagonal part of
    /// the symmetrized Q, scaled for the gradient of the Ising Hamiltonian):
    /// - J[i,j] = Q_sym[i,j] / 4  for i ≠ j
    fn qubo_to_ising(q_matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut j = vec![0.0f64; n * n]; // off-diagonal coupling (J[i,i] = 0)
        let mut h = vec![0.0f64; n];

        // Linear field from diagonal Q[i,i]
        for i in 0..n {
            h[i] += q_matrix[i * n + i] / 2.0;
        }

        // Off-diagonal couplings and their contributions to the linear field
        for i in 0..n {
            for jj in 0..n {
                if i == jj {
                    continue;
                }
                let q_sym = (q_matrix[i * n + jj] + q_matrix[jj * n + i]) / 2.0;
                j[i * n + jj] = q_sym / 4.0;
                // Off-diagonal contribution to linear field: Q_sym[i,j] / 2
                h[i] += q_sym / 2.0;
            }
        }

        (j, h)
    }

    /// Compute QUBO energy: E = x^T Q x (using binary {0,1} variables)
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

    /// Run a single SB trajectory on a QUBO problem
    fn run_single_qubo(
        &self,
        q_flat: &[f64],
        n: usize,
        rng: &mut StdRng,
    ) -> (Vec<bool>, f64) {
        if n == 0 {
            return (vec![], 0.0);
        }

        let (j_mat, h_vec) = Self::qubo_to_ising(q_flat, n);

        let c0 = if self.params.c0 > 0.0 {
            self.params.c0
        } else {
            0.5 / (n as f64).sqrt()
        };
        let dt = self.params.dt;
        let a_init = self.params.a_init;
        let a_final = self.params.a_final;
        let time_steps = self.params.time_steps;

        // Initialize x and y with small random values
        let mut x_vec: Vec<f64> = (0..n)
            .map(|_| rng.random_range(-0.1f64..0.1f64))
            .collect();
        let mut y_vec: Vec<f64> = (0..n)
            .map(|_| rng.random_range(-0.1f64..0.1f64))
            .collect();

        for t in 0..time_steps {
            let a = a_init + (a_final - a_init) * t as f64 / time_steps as f64;

            match self.params.variant {
                SBVariant::Ballistic => {
                    // bSB: coupling uses x directly
                    for i in 0..n {
                        let mut coupling = 0.0;
                        for jj in 0..n {
                            if i != jj {
                                coupling += j_mat[i * n + jj] * x_vec[jj];
                            }
                        }
                        y_vec[i] += (-a * x_vec[i] - c0 * coupling - h_vec[i]) * dt;
                    }
                }
                SBVariant::Discrete => {
                    // dSB: coupling uses sign(x)
                    for i in 0..n {
                        let mut coupling = 0.0;
                        for jj in 0..n {
                            if i != jj {
                                coupling += j_mat[i * n + jj] * x_vec[jj].signum();
                            }
                        }
                        y_vec[i] += (-a * x_vec[i] - c0 * coupling - h_vec[i]) * dt;
                    }
                }
            }

            // Update positions and apply clipping
            for i in 0..n {
                x_vec[i] += y_vec[i] * dt;
                if x_vec[i] > 1.0 {
                    x_vec[i] = 1.0;
                    y_vec[i] = 0.0;
                } else if x_vec[i] < -1.0 {
                    x_vec[i] = -1.0;
                    y_vec[i] = 0.0;
                }
            }
        }

        // Readout: s_i = sign(x_i) → binary x_i = (s_i + 1) / 2
        let binary_state: Vec<bool> = x_vec.iter().map(|&xi| xi >= 0.0).collect();
        let energy = Self::compute_qubo_energy(q_flat, &binary_state, n);

        (binary_state, energy)
    }

    /// Run generic sampler
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

        if matrix_or_tensor.ndim() != 2 {
            return Err(SamplerError::UnsupportedOperation(
                "SBSampler only supports QUBO (2D matrix) problems. \
                 Convert HOBO to QUBO first."
                    .to_string(),
            ));
        }

        let idx_to_var: HashMap<usize, String> = var_map
            .iter()
            .map(|(var, &idx)| (idx, var.clone()))
            .collect();

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

        let mut solution_counts: HashMap<Vec<bool>, (f64, usize)> = HashMap::new();

        for shot_idx in 0..shots {
            let seed = match self.seed {
                Some(s) => s.wrapping_add(shot_idx as u64),
                None => {
                    let mut rng_tmp = thread_rng();
                    rng_tmp.random()
                }
            };
            let mut rng = StdRng::seed_from_u64(seed);
            let (state, energy) = self.run_single_qubo(&q_flat, n, &mut rng);

            let entry = solution_counts.entry(state).or_insert((energy, 0));
            entry.1 += 1;
        }

        // Sort by (energy, state) for deterministic ordering
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

impl Sampler for SBSampler {
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
    /// Optimal energy: -2.0 (2 edges cut)
    fn build_k3_maxcut_qubo() -> (Array2<f64>, HashMap<String, usize>) {
        let mut q = Array2::<f64>::zeros((3, 3));
        q[[0, 0]] = -2.0;
        q[[1, 1]] = -2.0;
        q[[2, 2]] = -2.0;
        q[[0, 1]] = 2.0;
        q[[0, 2]] = 2.0;
        q[[1, 2]] = 2.0;

        let mut var_map = HashMap::new();
        var_map.insert("x0".to_string(), 0);
        var_map.insert("x1".to_string(), 1);
        var_map.insert("x2".to_string(), 2);

        (q, var_map)
    }

    #[test]
    fn test_sb_3var_maxcut() {
        let (q, var_map) = build_k3_maxcut_qubo();
        let sampler = SBSampler::new()
            .with_seed(42)
            .with_variant(SBVariant::Discrete)
            .with_time_steps(1000);

        let results = sampler
            .run_qubo(&(q, var_map), 50)
            .expect("SB run_qubo failed");

        assert!(!results.is_empty(), "Expected non-empty results");
        let best_energy = results[0].energy;
        assert!(
            best_energy <= -2.0 + 1e-9,
            "Expected optimal energy <= -2.0, got {best_energy}"
        );
    }

    #[test]
    fn test_sb_ballistic_maxcut() {
        let (q, var_map) = build_k3_maxcut_qubo();
        let sampler = SBSampler::new()
            .with_seed(99)
            .with_variant(SBVariant::Ballistic)
            .with_time_steps(1000);

        let results = sampler
            .run_qubo(&(q, var_map), 50)
            .expect("SB ballistic failed");

        assert!(!results.is_empty());
        let best_energy = results[0].energy;
        assert!(
            best_energy <= -2.0 + 1e-9,
            "Ballistic SB: expected energy <= -2.0, got {best_energy}"
        );
    }

    #[test]
    fn test_sb_determinism() {
        let (q, var_map) = build_k3_maxcut_qubo();

        let s1 = SBSampler::new().with_seed(42).with_time_steps(500);
        let s2 = SBSampler::new().with_seed(42).with_time_steps(500);

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
            assert_eq!(a.assignments, b.assignments, "Assignments differ for same seed");
        }
    }

    #[test]
    fn test_sb_ising_chain() {
        // 1D Ising chain: J[i,i+1] = -1 (ferromagnetic), no field
        // Ground state: all spins aligned (all 0 or all 1 in QUBO)
        // QUBO: minimize E = Σ_{i} (2*x_i*x_{i+1} - x_i - x_{i+1}) + const
        // Which means Q[i,i] = -1, Q[i,i+1] = Q[i+1,i] = 1 (upper + lower)
        let n = 4;
        let mut q = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            q[[i, i]] = -1.0; // Linear: -x_i (partial)
        }
        // Pair terms
        for i in 0..(n - 1) {
            q[[i, i + 1]] = 2.0;
            // Note: lower triangular left as 0 (upper-triangular convention)
        }
        // Adjust diagonal to properly encode the chain
        // E = sum_{i=0}^{n-2} (x_i - x_{i+1})^2 - (n-1) = min at all-same
        // = sum x_i^2 - 2*x_i*x_{i+1} + x_{i+1}^2 - (n-1)
        // Since x^2 = x for binary: = sum x_i + x_{i+1} - 2*x_i*x_{i+1} - (n-1)
        // QUBO: Q[i,i] = 1 (from x_{i-1} chain) + 1 (from x_{i+1} chain), Q[i,i+1] = -2
        let mut q2 = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            q2[[i, i]] = 1.0; // Will be adjusted
        }
        // Pair interactions: minimize (x_i - x_{i+1})^2 = x_i + x_{i+1} - 2*x_i*x_{i+1}
        for i in 0..(n - 1) {
            q2[[i, i]] += 0.0; // Already 1 from initialization
            q2[[i + 1, i + 1]] += 0.0;
            q2[[i, i + 1]] = -2.0; // quadratic penalty
        }
        // Actually build proper chain QUBO:
        // E = sum_{i=0}^{n-2} (x_i + x_{i+1} - 2*x_i*x_{i+1})
        // Minimum = 0 (all-zero or all-one)
        let mut q_chain = Array2::<f64>::zeros((n, n));
        for i in 0..(n - 1) {
            q_chain[[i, i]] += 1.0;
            q_chain[[i + 1, i + 1]] += 1.0;
            q_chain[[i, i + 1]] = -2.0;
        }

        let mut var_map = HashMap::new();
        for i in 0..n {
            var_map.insert(format!("s{i}"), i);
        }

        let sampler = SBSampler::new()
            .with_seed(42)
            .with_variant(SBVariant::Discrete)
            .with_time_steps(1000);

        let results = sampler
            .run_qubo(&(q_chain, var_map), 30)
            .expect("Ising chain failed");

        assert!(!results.is_empty());
        // Ground state energy = 0 (all-same-spin)
        let best_energy = results[0].energy;
        assert!(
            best_energy <= 1e-9,
            "Expected energy <= 0.0 for ferromagnetic chain, got {best_energy}"
        );
    }

    #[test]
    fn test_sb_hobo_returns_error() {
        // SBSampler does not support HOBO (ndim > 2)
        use scirs2_core::ndarray::Array3;
        let tensor = Array3::<f64>::zeros((2, 2, 2));
        let mut var_map = HashMap::new();
        var_map.insert("a".to_string(), 0);
        var_map.insert("b".to_string(), 1);

        let sampler = SBSampler::new().with_seed(1);
        let result = sampler.run_hobo(&(tensor.into_dyn(), var_map), 10);
        assert!(result.is_err(), "Expected error for 3D HOBO in SBSampler");
    }
}
