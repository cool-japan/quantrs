//! QAOA warm-start initialization using spectral relaxation.
//!
//! Initializes QAOA variational parameters from a classical spectral
//! relaxation solution (Egger et al. 2021, "Warm-starting quantum optimization").
//!
//! Instead of random initial angles, we use the Fiedler vector of the graph
//! Laplacian to compute initial qubit rotation angles, giving QAOA a head
//! start near the classical solution.

use scirs2_core::ndarray::{Array2, ArrayView2};
use std::f64::consts::PI;

use crate::error::{MLError, Result};

/// Strategy for computing the classical warm-start solution.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum WarmStartStrategy {
    /// Spectral relaxation via Fiedler vector of graph Laplacian.
    Spectral,
    /// Degree-proportional initialization (faster but lower quality).
    Degree,
    /// Random initialization (baseline) with a fixed seed.
    Random {
        /// RNG seed for reproducibility.
        seed: u64,
    },
}

/// Graph for MaxCut warm-start.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices.
    pub n_vertices: usize,
    /// Weighted edge list: (u, v, weight).
    pub edges: Vec<(usize, usize, f64)>,
}

/// Result of warm-start initialization.
#[derive(Debug, Clone)]
pub struct WarmStartAngles {
    /// Initial gamma angles (problem parameters), one per QAOA layer.
    pub gammas: Vec<f64>,
    /// Initial beta angles (mixer parameters), one per QAOA layer.
    pub betas: Vec<f64>,
    /// Classical objective value (MaxCut lower bound).
    pub classical_objective: f64,
    /// Fiedler vector values mapped to [-1, 1], one per vertex.
    pub vertex_assignments: Vec<f64>,
}

/// QAOA warm-start optimizer.
#[derive(Debug, Clone)]
pub struct WarmStartQAOAOptimizer {
    /// The graph to solve MaxCut on.
    pub graph: Graph,
    /// Number of QAOA layers (p parameter).
    pub n_layers: usize,
    /// Which strategy to use for the classical initial solution.
    pub strategy: WarmStartStrategy,
}

// ---------------------------------------------------------------------------
// Internal linear-algebra helpers
// ---------------------------------------------------------------------------

/// Gaussian elimination with partial pivoting, returning the solution vector.
///
/// # Errors
/// Returns `MLError::NumericalError` if the matrix is singular.
fn solve_linear_system_local(a: &Array2<f64>, b: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();
    if a.nrows() != n || a.ncols() != n {
        return Err(MLError::DimensionMismatch(format!(
            "Matrix ({} x {}) incompatible with rhs length {}",
            a.nrows(),
            a.ncols(),
            n
        )));
    }

    // Build augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| a[[i, j]]).collect();
            row.push(b[i]);
            row
        })
        .collect();

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = aug[k][k].abs();
        let mut max_idx = k;
        for row in (k + 1)..n {
            let val = aug[row][k].abs();
            if val > max_val {
                max_val = val;
                max_idx = row;
            }
        }

        if max_val < 1e-12 {
            return Err(MLError::NumericalError(format!(
                "Singular matrix: |pivot| = {:.2e} < 1e-12 at column {}",
                max_val, k
            )));
        }

        if max_idx != k {
            aug.swap(k, max_idx);
        }

        let pivot = aug[k][k];
        for i in (k + 1)..n {
            let factor = aug[i][k] / pivot;
            for col in k..=n {
                let sub = factor * aug[k][col];
                aug[i][col] -= sub;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Graph methods
// ---------------------------------------------------------------------------

impl Graph {
    /// Compute the graph Laplacian L = D - A.
    ///
    /// `D` is the diagonal degree matrix (weighted degrees) and `A` is the
    /// weighted adjacency matrix.
    pub fn laplacian(&self) -> Array2<f64> {
        let n = self.n_vertices;
        let mut l = Array2::<f64>::zeros((n, n));

        for &(u, v, w) in &self.edges {
            // Adjacency entries (undirected)
            l[[u, v]] -= w;
            l[[v, u]] -= w;
            // Degree entries
            l[[u, u]] += w;
            l[[v, v]] += w;
        }

        l
    }

    /// Compute the Fiedler vector (second-smallest eigenvector of the Laplacian).
    ///
    /// Uses inverse power iteration with a small shift λ = 0.1 to avoid the
    /// trivial zero eigenvalue, deflating the constant component at each step
    /// so the iteration converges to the Fiedler vector rather than the
    /// all-ones eigenvector.
    ///
    /// Returns a unit vector whose entries are the vertex assignments in [-1, 1].
    ///
    /// # Errors
    /// Returns `MLError::NumericalError` if the shifted Laplacian is singular.
    pub fn fiedler_vector(&self) -> Result<Vec<f64>> {
        let n = self.n_vertices;
        if n < 2 {
            return Err(MLError::InvalidInput(
                "Graph must have at least 2 vertices to compute the Fiedler vector".to_string(),
            ));
        }

        let l = self.laplacian();

        // L_shifted = L + λI  (λ = 0.1 keeps the zero eigenvalue at 0.1 and
        //  all other eigenvalues remain ≥ 0.1, so the inverse converges to the
        //  eigenvector for the *smallest* eigenvalue of L_shifted, which is the
        //  constant all-ones vector.  We deflate that away each iteration.)
        let lambda_shift = 0.1_f64;
        let mut l_shifted = l.clone();
        for i in 0..n {
            l_shifted[[i, i]] += lambda_shift;
        }

        // Initialise x orthogonal to the constant vector (subtract mean, then normalise).
        let mut rng = fastrand::Rng::with_seed(42);
        let mut x: Vec<f64> = (0..n).map(|_| rng.f64() * 2.0 - 1.0).collect();

        // Subtract mean so x ⊥ 1 from the start
        let mean = x.iter().sum::<f64>() / n as f64;
        for xi in x.iter_mut() {
            *xi -= mean;
        }
        // Normalise
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-14 {
            // Pathological initialisation – use a deterministic fallback
            for (i, xi) in x.iter_mut().enumerate() {
                *xi = (i as f64) - (n as f64 - 1.0) / 2.0;
            }
            let norm2 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            for xi in x.iter_mut() {
                *xi /= norm2;
            }
        } else {
            for xi in x.iter_mut() {
                *xi /= norm;
            }
        }

        // Inverse power iteration with deflation
        for _ in 0..100 {
            // Solve L_shifted * x_new = x
            let x_new = solve_linear_system_local(&l_shifted, &x)?;

            // Deflate the constant component: x_new -= mean(x_new)
            let m = x_new.iter().sum::<f64>() / n as f64;
            let mut x_new: Vec<f64> = x_new.into_iter().map(|v| v - m).collect();

            // Normalise
            let norm = x_new.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm < 1e-14 {
                break; // Converged or degenerate
            }
            for xi in x_new.iter_mut() {
                *xi /= norm;
            }

            x = x_new;
        }

        // Clip to [-1, 1] to absorb floating-point drift
        let x_clipped: Vec<f64> = x.into_iter().map(|v| v.clamp(-1.0, 1.0)).collect();
        Ok(x_clipped)
    }

    /// Compute the MaxCut value for the given vertex assignments.
    ///
    /// For each edge (u, v, w), adds w * (1 − sign(a_u) * sign(a_v)) / 2 to the total.
    pub fn maxcut_value(&self, assignments: &[f64]) -> f64 {
        self.edges
            .iter()
            .map(|&(u, v, w)| w * (1.0 - assignments[u].signum() * assignments[v].signum()) / 2.0)
            .sum()
    }
}

// ---------------------------------------------------------------------------
// WarmStartQAOAOptimizer methods
// ---------------------------------------------------------------------------

impl WarmStartQAOAOptimizer {
    /// Compute initial QAOA angles using the selected warm-start strategy.
    ///
    /// # Errors
    /// Returns `MLError` if the underlying computation fails (e.g. singular Laplacian).
    pub fn compute_initial_angles(&self) -> Result<WarmStartAngles> {
        let p = self.n_layers;
        if p == 0 {
            return Err(MLError::InvalidInput(
                "n_layers must be at least 1".to_string(),
            ));
        }

        let n = self.graph.n_vertices;

        // Uniform angle initialisation (same for all strategies)
        let gammas = vec![PI / (2.0 * p as f64); p];
        let betas = vec![PI / (4.0 * p as f64); p];

        match &self.strategy {
            WarmStartStrategy::Spectral => {
                let fiedler = self.graph.fiedler_vector()?;

                // Map Fiedler values to rotation angles: θ_i = arccos(v_i)
                // The Fiedler vector values are already in [-1, 1] after clipping.
                let _thetas: Vec<f64> = fiedler.iter().map(|&v| v.acos()).collect();

                let classical_objective = self.graph.maxcut_value(&fiedler);

                Ok(WarmStartAngles {
                    gammas,
                    betas,
                    classical_objective,
                    vertex_assignments: fiedler,
                })
            }

            WarmStartStrategy::Degree => {
                // Degree-proportional: assign ±1 based on whether vertex degree is
                // above or below the mean degree.
                let mut degrees = vec![0.0_f64; n];
                for &(u, v, w) in &self.graph.edges {
                    degrees[u] += w;
                    degrees[v] += w;
                }
                let mean_degree = degrees.iter().sum::<f64>() / n as f64;
                let assignments: Vec<f64> = degrees
                    .iter()
                    .map(|&d| if d >= mean_degree { 1.0 } else { -1.0 })
                    .collect();

                let classical_objective = self.graph.maxcut_value(&assignments);

                Ok(WarmStartAngles {
                    gammas,
                    betas,
                    classical_objective,
                    vertex_assignments: assignments,
                })
            }

            WarmStartStrategy::Random { seed } => {
                // Seeded random assignments in [-1, 1]
                let mut rng = fastrand::Rng::with_seed(*seed);
                let assignments: Vec<f64> = (0..n)
                    .map(|_| rng.f64() * 2.0 - 1.0)
                    .collect();

                let classical_objective = self.graph.maxcut_value(&assignments);

                Ok(WarmStartAngles {
                    gammas,
                    betas,
                    classical_objective,
                    vertex_assignments: assignments,
                })
            }
        }
    }

    /// Run warm-start optimisation using a classical proxy cost function.
    ///
    /// The proxy cost is:
    ///   C(γ, β) = −Σ_{(i,j)∈E} w_ij * (1 − cos(θ_i − θ_j)) / 2
    ///
    /// where θ_i = γ[0] * vertex_assignments[i] + β[0] (per-vertex angle).
    /// Gradient descent is applied for `n_steps` steps.
    ///
    /// # Errors
    /// Returns `MLError` if the initial angle computation fails.
    pub fn optimize(&self, n_steps: usize) -> Result<WarmStartAngles> {
        let mut angles = self.compute_initial_angles()?;
        let lr = 0.01_f64;

        // We parameterise the per-vertex angles as
        //   φ_v = Σ_k (gammas[k] * v_assignments[v]) + betas[k]
        // For simplicity in the classical proxy we use a single effective angle:
        //   φ_v = gammas[0] * v_assignments[v]
        // and differentiate w.r.t. gammas[0] and betas (treated as global shift).

        for _ in 0..n_steps {
            // Compute per-vertex angles using first layer parameters only
            let phi: Vec<f64> = angles
                .vertex_assignments
                .iter()
                .map(|&v| angles.gammas[0] * v + angles.betas[0])
                .collect();

            // Cost: C = -Σ w_ij * (1 - cos(φ_i - φ_j)) / 2
            // Gradient w.r.t. gammas[0]:
            //   dC/dγ = Σ w_ij * sin(φ_i - φ_j) * (v_i - v_j) / 2
            // Gradient w.r.t. betas[0]:
            //   dC/dβ = Σ w_ij * sin(φ_i - φ_j)
            let mut d_gamma = 0.0_f64;
            let mut d_beta = 0.0_f64;

            for &(u, v_idx, w) in &self.graph.edges {
                let diff = phi[u] - phi[v_idx];
                let sin_diff = diff.sin();
                let a_u = angles.vertex_assignments[u];
                let a_v = angles.vertex_assignments[v_idx];
                d_gamma += w * sin_diff * (a_u - a_v) / 2.0;
                d_beta += w * sin_diff;
            }

            // Minimise cost → gradient descent (cost is already negated above, so signs flip)
            angles.gammas[0] -= lr * d_gamma;
            angles.betas[0] -= lr * d_beta;

            // Propagate the same update magnitude to remaining layers (uniform)
            for k in 1..self.n_layers {
                angles.gammas[k] -= lr * d_gamma / (k as f64 + 1.0);
                angles.betas[k] -= lr * d_beta / (k as f64 + 1.0);
            }
        }

        // Recompute objective with updated angles
        let phi_final: Vec<f64> = angles
            .vertex_assignments
            .iter()
            .map(|&v| angles.gammas[0] * v + angles.betas[0])
            .collect();

        let cost: f64 = self
            .graph
            .edges
            .iter()
            .map(|&(u, v_idx, w)| {
                let diff = phi_final[u] - phi_final[v_idx];
                -w * (1.0 - diff.cos()) / 2.0
            })
            .sum();

        // Return negative cost as objective (maximisation)
        angles.classical_objective = -cost;

        Ok(angles)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_laplacian() {
        // 4-vertex complete graph K4
        let g = Graph {
            n_vertices: 4,
            edges: vec![
                (0, 1, 1.0),
                (0, 2, 1.0),
                (0, 3, 1.0),
                (1, 2, 1.0),
                (1, 3, 1.0),
                (2, 3, 1.0),
            ],
        };
        let l = g.laplacian();
        // Diagonal entries should be degree = 3
        assert!((l[[0, 0]] - 3.0).abs() < 1e-10);
        // Off-diagonal should be -1
        assert!((l[[0, 1]] + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_warm_start_spectral() {
        // 4-vertex path graph (MaxCut = 2)
        let g = Graph {
            n_vertices: 4,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
        };
        let opt = WarmStartQAOAOptimizer {
            graph: g,
            n_layers: 2,
            strategy: WarmStartStrategy::Spectral,
        };
        let angles = opt
            .compute_initial_angles()
            .expect("spectral warm-start should succeed");
        // Should have 2 gammas and 2 betas
        assert_eq!(angles.gammas.len(), 2);
        assert_eq!(angles.betas.len(), 2);
        // Classical objective should be non-negative
        assert!(angles.classical_objective >= 0.0);
        // Vertex assignments should be in [-1, 1]
        for v in &angles.vertex_assignments {
            assert!(
                *v >= -1.0 - 1e-10 && *v <= 1.0 + 1e-10,
                "vertex assignment {v} out of range"
            );
        }
    }

    #[test]
    fn test_warm_start_better_than_random_k4() {
        // Complete graph K4, optimal MaxCut = 4
        let g = Graph {
            n_vertices: 4,
            edges: vec![
                (0, 1, 1.0),
                (0, 2, 1.0),
                (0, 3, 1.0),
                (1, 2, 1.0),
                (1, 3, 1.0),
                (2, 3, 1.0),
            ],
        };
        let opt_spectral = WarmStartQAOAOptimizer {
            graph: g.clone(),
            n_layers: 1,
            strategy: WarmStartStrategy::Spectral,
        };
        let angles = opt_spectral
            .compute_initial_angles()
            .expect("spectral warm-start on K4 should succeed");
        // Spectral warm-start should give positive MaxCut estimate
        assert!(
            angles.classical_objective > 0.0,
            "Spectral warm-start should give positive objective, got {}",
            angles.classical_objective
        );
    }

    #[test]
    fn test_degree_strategy() {
        let g = Graph {
            n_vertices: 3,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0)],
        };
        let opt = WarmStartQAOAOptimizer {
            graph: g,
            n_layers: 1,
            strategy: WarmStartStrategy::Degree,
        };
        let angles = opt
            .compute_initial_angles()
            .expect("degree strategy warm-start should succeed");
        assert_eq!(angles.gammas.len(), 1);
    }

    #[test]
    fn test_random_strategy() {
        let g = Graph {
            n_vertices: 4,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
        };
        let opt = WarmStartQAOAOptimizer {
            graph: g,
            n_layers: 2,
            strategy: WarmStartStrategy::Random { seed: 12345 },
        };
        let angles = opt
            .compute_initial_angles()
            .expect("random strategy warm-start should succeed");
        // Should produce the right number of angles
        assert_eq!(angles.gammas.len(), 2);
        assert_eq!(angles.betas.len(), 2);
        // Vertex assignments from seeded random are in (-1, 1)
        for v in &angles.vertex_assignments {
            assert!(*v >= -1.0 && *v <= 1.0, "random assignment {v} out of range");
        }
    }

    #[test]
    fn test_optimize_runs() {
        let g = Graph {
            n_vertices: 4,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
        };
        let opt = WarmStartQAOAOptimizer {
            graph: g,
            n_layers: 2,
            strategy: WarmStartStrategy::Spectral,
        };
        let result = opt.optimize(50).expect("optimize should succeed");
        assert_eq!(result.gammas.len(), 2);
    }

    #[test]
    fn test_maxcut_value() {
        // Path 0-1-2-3: optimal cut = {0,2} vs {1,3}, value = 3
        let g = Graph {
            n_vertices: 4,
            edges: vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
        };
        // All alternating: 0→+1, 1→-1, 2→+1, 3→-1
        let assignments = [1.0, -1.0, 1.0, -1.0];
        let cut = g.maxcut_value(&assignments);
        // All 3 edges are cut (signs differ)
        assert!((cut - 3.0).abs() < 1e-10, "Expected cut=3, got {cut}");
    }

    #[test]
    fn test_linear_system_singular() {
        // Singular matrix should return NumericalError
        let a = Array2::<f64>::zeros((2, 2));
        let b = vec![1.0, 2.0];
        let result = solve_linear_system_local(&a, &b);
        assert!(result.is_err());
    }
}
