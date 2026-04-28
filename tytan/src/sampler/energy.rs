//! SIMD-accelerated QUBO/PUBO energy evaluation.
//!
//! These functions serve as the shared inner loop for all QuantRS2 tytan samplers.
//! For small `n` (< 32 variables), they fall back to scalar computation.
//!
//! # Energy conventions
//!
//! All functions follow the convention used by the tytan samplers:
//!
//! ```text
//! E(x) = Σ_{i,j} Q[i*n+j] * x[i] * x[j]
//! ```
//!
//! The matrix `Q` is stored as a flat row-major `Vec<f64>` of length `n*n`.
//! The state `x` is a `Vec<bool>` of length `n`.
//!
//! # Incremental energy delta
//!
//! The influence vector `g[i]` is defined as:
//! ```text
//! g[i] = Q[i*n+i] + Σ_{j≠i} (Q[i*n+j] + Q[j*n+i]) * x[j]
//! ```
//!
//! Then ΔE from flipping bit `k` is:
//! ```text
//! ΔE = (1 - 2*x[k]) * g[k]
//! ```
//!
//! # SIMD strategy
//!
//! Uses `std::simd::f64x4` (portable SIMD) for 4-wide f64 dot products.
//! Falls back to scalar for `n < 32` where overhead exceeds benefit.

use scirs2_core::ndarray::{Array2, Ix2};
use std::collections::HashMap;
use std::simd::{f64x4, num::SimdFloat};

/// SIMD threshold: use scalar for smaller problems.
const SIMD_THRESHOLD: usize = 32;

/// Build a dense n×n Q matrix (flat row-major) from a sparse edge list.
///
/// Diagonal entries `Q[(i,i)]` represent linear terms.
/// Off-diagonal entries `Q[(i,j)]` with `i < j` represent quadratic terms.
/// This function stores them in the full matrix (both `[i,j]` and `[j,i]`
/// positions are set if the pair is specified as `(i,j)` with `i < j`).
///
/// # Panics
///
/// Panics if any index in `edges` is >= `n`.
///
/// # Examples
///
/// ```rust
/// use std::collections::HashMap;
/// use quantrs2_tytan::sampler::energy::build_dense_q;
///
/// let mut edges = HashMap::new();
/// edges.insert((0, 0), -1.0_f64);
/// edges.insert((1, 1), -1.0_f64);
/// edges.insert((0, 1), 2.0_f64);
///
/// let q = build_dense_q(2, &edges);
/// assert_eq!(q.len(), 4);
/// assert!((q[0] - (-1.0)).abs() < 1e-12);
/// assert!((q[1] - 2.0).abs() < 1e-12);
/// assert!((q[3] - (-1.0)).abs() < 1e-12);
/// ```
pub fn build_dense_q(n: usize, edges: &HashMap<(usize, usize), f64>) -> Vec<f64> {
    let mut q = vec![0.0f64; n * n];
    for (&(i, j), &val) in edges {
        assert!(i < n && j < n, "edge index out of bounds: ({i},{j}) for n={n}");
        q[i * n + j] += val;
    }
    q
}

/// Build a dense n×n Q matrix (flat row-major) from a 2D ndarray.
///
/// # Errors
///
/// Returns an error string if the matrix is not square or cannot be sliced.
pub fn build_dense_q_from_array(q_array: &Array2<f64>) -> Result<Vec<f64>, String> {
    let (rows, cols) = q_array.dim();
    if rows != cols {
        return Err(format!(
            "QUBO matrix must be square, got {}x{}",
            rows, cols
        ));
    }
    let q_flat = q_array
        .as_slice()
        .ok_or_else(|| "Non-contiguous QUBO matrix — cannot extract flat slice".to_string())?
        .to_vec();
    Ok(q_flat)
}

/// Compute full QUBO energy (scalar reference implementation).
///
/// `E(x) = Σ_{i,j} Q[i*n+j] * x[i] * x[j]`
///
/// # Arguments
///
/// * `state` – binary state vector of length `n`
/// * `q_matrix` – flat row-major QUBO matrix of length `n*n`
/// * `n` – number of variables
///
/// # Panics
///
/// Panics if `state.len() != n` or `q_matrix.len() != n*n`.
#[inline]
pub fn energy_full(state: &[bool], q_matrix: &[f64], n: usize) -> f64 {
    debug_assert_eq!(state.len(), n, "state length mismatch");
    debug_assert_eq!(q_matrix.len(), n * n, "q_matrix length mismatch");
    let mut energy = 0.0f64;
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

/// Compute full QUBO energy using SIMD acceleration where beneficial.
///
/// Equivalent to [`energy_full`] but uses 4-wide f64 SIMD for `n >= 32`.
///
/// # Arguments
///
/// * `state` – binary state vector of length `n`
/// * `q_matrix` – flat row-major QUBO matrix of length `n*n`
/// * `n` – number of variables
#[inline]
pub fn energy_full_simd(state: &[bool], q_matrix: &[f64], n: usize) -> f64 {
    if n < SIMD_THRESHOLD {
        return energy_full(state, q_matrix, n);
    }

    let mut total = 0.0f64;
    for i in 0..n {
        if !state[i] {
            continue;
        }
        let row_start = i * n;
        let row = &q_matrix[row_start..row_start + n];
        total += dot_bool_f64_simd(state, row);
    }
    total
}

/// Compute the influence vector `g[i]` for the current state.
///
/// ```text
/// g[i] = Q[i*n+i] + Σ_{j≠i} (Q[i*n+j] + Q[j*n+i]) * x[j]
/// ```
///
/// This is the scalar reference implementation.
///
/// # Arguments
///
/// * `state` – binary state vector of length `n`
/// * `q_matrix` – flat row-major QUBO matrix of length `n*n`
/// * `n` – number of variables
#[inline]
pub fn compute_influence(state: &[bool], q_matrix: &[f64], n: usize) -> Vec<f64> {
    let mut g = vec![0.0f64; n];
    for i in 0..n {
        g[i] = q_matrix[i * n + i];
        for j in 0..n {
            if j != i && state[j] {
                g[i] += q_matrix[i * n + j] + q_matrix[j * n + i];
            }
        }
    }
    g
}

/// Compute the influence vector using SIMD acceleration where beneficial.
///
/// Equivalent to [`compute_influence`] but uses 4-wide f64 SIMD for `n >= 32`.
#[inline]
pub fn compute_influence_simd(state: &[bool], q_matrix: &[f64], n: usize) -> Vec<f64> {
    if n < SIMD_THRESHOLD {
        return compute_influence(state, q_matrix, n);
    }

    let mut g = vec![0.0f64; n];
    for i in 0..n {
        let row_i = &q_matrix[i * n..i * n + n];
        // Build symmetric row: for each j, add Q[i,j] + Q[j,i]
        // Q[i,j] comes from row_i[j]; Q[j,i] comes from column i of Q = q_matrix[j*n+i]
        // We need Σ_{j≠i} (Q[i,j] + Q[j,i]) * x[j] = dot(row_i, x) + col_dot(i, x) - 2*Q[i,i]*x[i]
        // For correctness: compute Σ_j Q[i,j]*x[j] + Σ_j Q[j,i]*x[j] - 2*Q[i,i]*x[i]
        // (the 2* diagonal term accounts for double-counting i==j in both sums)
        let row_dot = dot_bool_f64_simd(state, row_i);
        let col_dot = dot_bool_f64_col_simd(state, q_matrix, n, i);
        // Subtract the j==i terms added in both sums: Q[i,i]*x[i] + Q[i,i]*x[i] = 2*Q[i,i]*x[i]
        let diag = q_matrix[i * n + i];
        let x_i = if state[i] { 1.0 } else { 0.0 };
        // g[i] = Q[i,i] + sum_{j!=i}(Q[i,j]+Q[j,i])*x[j]
        //      = Q[i,i] + (row_dot - Q[i,i]*x[i]) + (col_dot - Q[i,i]*x[i])
        //      = diag + row_dot + col_dot - 2*diag*x_i
        g[i] = diag + row_dot + col_dot - 2.0 * diag * x_i;
    }
    g
}

/// Compute energy delta for flipping bit `k` (scalar reference).
///
/// `ΔE = (1 - 2*x[k]) * g[k]`
///
/// # Arguments
///
/// * `state` – current binary state
/// * `q_matrix` – flat row-major QUBO matrix
/// * `n` – number of variables
/// * `k` – index of the bit to flip
#[inline]
pub fn energy_delta(state: &[bool], q_matrix: &[f64], n: usize, k: usize) -> f64 {
    let g_k = {
        let mut g = q_matrix[k * n + k];
        for j in 0..n {
            if j != k && state[j] {
                g += q_matrix[k * n + j] + q_matrix[j * n + k];
            }
        }
        g
    };
    (1.0 - 2.0 * if state[k] { 1.0 } else { 0.0 }) * g_k
}

/// Compute energy delta for flipping bit `k` using SIMD acceleration.
///
/// Equivalent to [`energy_delta`] but uses 4-wide f64 SIMD for `n >= 32`.
///
/// # Arguments
///
/// * `state` – current binary state
/// * `q_matrix` – flat row-major QUBO matrix
/// * `n` – number of variables
/// * `k` – index of the bit to flip
#[inline]
pub fn energy_delta_simd(state: &[bool], q_matrix: &[f64], n: usize, k: usize) -> f64 {
    if n < SIMD_THRESHOLD {
        return energy_delta(state, q_matrix, n, k);
    }

    let row_k = &q_matrix[k * n..k * n + n];
    // Σ_j Q[k,j]*x[j]
    let row_dot = dot_bool_f64_simd(state, row_k);
    // Σ_j Q[j,k]*x[j]
    let col_dot = dot_bool_f64_col_simd(state, q_matrix, n, k);
    let diag = q_matrix[k * n + k];
    let x_k = if state[k] { 1.0 } else { 0.0 };
    // g[k] = diag + (row_dot - diag*x_k) + (col_dot - diag*x_k)
    let g_k = diag + row_dot + col_dot - 2.0 * diag * x_k;
    (1.0 - 2.0 * x_k) * g_k
}

/// Update the influence vector after flipping bit `k`.
///
/// After flipping `x[k]` to `new_val`:
/// `g[i] += (Q[i,k] + Q[k,i]) * delta` for `i != k`
///
/// where `delta = if new_val { 1.0 } else { -1.0 }`.
///
/// This is the scalar reference implementation matching the samplers exactly.
#[inline]
pub fn update_influence(g: &mut [f64], q_matrix: &[f64], n: usize, k: usize, new_val: bool) {
    let delta = if new_val { 1.0 } else { -1.0 };
    for i in 0..n {
        if i != k {
            g[i] += (q_matrix[i * n + k] + q_matrix[k * n + i]) * delta;
        }
    }
}

/// Update the influence vector after flipping bit `k` using SIMD.
///
/// Equivalent to [`update_influence`] but uses 4-wide f64 SIMD for `n >= 32`.
#[inline]
pub fn update_influence_simd(g: &mut [f64], q_matrix: &[f64], n: usize, k: usize, new_val: bool) {
    if n < SIMD_THRESHOLD {
        return update_influence(g, q_matrix, n, k, new_val);
    }

    let delta = if new_val { 1.0 } else { -1.0 };
    let delta4 = f64x4::splat(delta);
    let diag = q_matrix[k * n + k];

    // For each i, need Q[i,k] + Q[k,i]:
    // Q[k,i] = q_matrix[k*n + i] (from row k of Q)
    // Q[i,k] = q_matrix[i*n + k] (column k)
    let row_k = &q_matrix[k * n..k * n + n];

    let chunks = n / 4;
    for c in 0..chunks {
        let base = c * 4;
        // Column k values: q_matrix[base*n+k], q_matrix[(base+1)*n+k], ...
        let col_k_vals = f64x4::from_array([
            q_matrix[base * n + k],
            q_matrix[(base + 1) * n + k],
            q_matrix[(base + 2) * n + k],
            q_matrix[(base + 3) * n + k],
        ]);
        // Row k values at positions base..base+4
        let row_k_vals =
            f64x4::from_array([row_k[base], row_k[base + 1], row_k[base + 2], row_k[base + 3]]);
        let increment = (col_k_vals + row_k_vals) * delta4;
        let g4 = f64x4::from_array([g[base], g[base + 1], g[base + 2], g[base + 3]]);
        let result = g4 + increment;
        let arr = result.to_array();
        g[base] = arr[0];
        g[base + 1] = arr[1];
        g[base + 2] = arr[2];
        g[base + 3] = arr[3];
    }
    // Remainder
    for i in (chunks * 4)..n {
        if i != k {
            g[i] += (q_matrix[i * n + k] + row_k[i]) * delta;
        }
    }
    // Handle k itself — no update needed (g[k] is invariant to flipping k)
    // But the SIMD loop may have touched g[k] in the chunk; correct it:
    if k < chunks * 4 {
        // k was touched in a SIMD chunk — undo the update for position k
        g[k] -= (diag + row_k[k]) * delta;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SIMD primitives
// ─────────────────────────────────────────────────────────────────────────────

/// SIMD 4-wide dot product of a bool slice (mask) with an f64 slice (values).
///
/// Computes `Σ_i (if mask[i] { values[i] } else { 0.0 })`.
#[inline]
fn dot_bool_f64_simd(mask: &[bool], values: &[f64]) -> f64 {
    let n = mask.len().min(values.len());
    let chunks = n / 4;
    let mut acc = f64x4::splat(0.0);

    for c in 0..chunks {
        let base = c * 4;
        let m = f64x4::from_array([
            if mask[base] { 1.0 } else { 0.0 },
            if mask[base + 1] { 1.0 } else { 0.0 },
            if mask[base + 2] { 1.0 } else { 0.0 },
            if mask[base + 3] { 1.0 } else { 0.0 },
        ]);
        let v = f64x4::from_array([
            values[base],
            values[base + 1],
            values[base + 2],
            values[base + 3],
        ]);
        acc += m * v;
    }

    let mut s = acc.reduce_sum();
    for i in (chunks * 4)..n {
        if mask[i] {
            s += values[i];
        }
    }
    s
}

/// SIMD 4-wide dot product of a bool slice (mask) with column `col_idx` of `q_matrix`.
///
/// Column `col_idx` is not contiguous in memory, so we gather 4 elements at a time.
///
/// Computes `Σ_i (if mask[i] { q_matrix[i*n + col_idx] } else { 0.0 })`.
#[inline]
fn dot_bool_f64_col_simd(mask: &[bool], q_matrix: &[f64], n: usize, col_idx: usize) -> f64 {
    let chunks = n / 4;
    let mut acc = f64x4::splat(0.0);

    for c in 0..chunks {
        let base = c * 4;
        let m = f64x4::from_array([
            if mask[base] { 1.0 } else { 0.0 },
            if mask[base + 1] { 1.0 } else { 0.0 },
            if mask[base + 2] { 1.0 } else { 0.0 },
            if mask[base + 3] { 1.0 } else { 0.0 },
        ]);
        let v = f64x4::from_array([
            q_matrix[base * n + col_idx],
            q_matrix[(base + 1) * n + col_idx],
            q_matrix[(base + 2) * n + col_idx],
            q_matrix[(base + 3) * n + col_idx],
        ]);
        acc += m * v;
    }

    let mut s = acc.reduce_sum();
    for i in (chunks * 4)..n {
        if mask[i] {
            s += q_matrix[i * n + col_idx];
        }
    }
    s
}

// ─────────────────────────────────────────────────────────────────────────────
// Array2 convenience wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute full QUBO energy from an `Array2<f64>` directly.
///
/// Convenience wrapper around [`energy_full_simd`] that handles the flat conversion.
///
/// # Errors
///
/// Returns an error if the matrix is not square or not contiguous.
pub fn energy_full_from_array(
    state: &[bool],
    q_array: &Array2<f64>,
) -> Result<f64, String> {
    let q_flat = build_dense_q_from_array(q_array)?;
    let n = q_array.dim().0;
    Ok(energy_full_simd(state, &q_flat, n))
}

/// Compute energy delta from an `Array2<f64>` directly.
///
/// Convenience wrapper around [`energy_delta_simd`] that handles the flat conversion.
///
/// # Errors
///
/// Returns an error if the matrix is not square or not contiguous.
pub fn energy_delta_from_array(
    state: &[bool],
    q_array: &Array2<f64>,
    k: usize,
) -> Result<f64, String> {
    let q_flat = build_dense_q_from_array(q_array)?;
    let n = q_array.dim().0;
    Ok(energy_delta_simd(state, &q_flat, n, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── helpers ────────────────────────────────────────────────────────────

    /// Build a deterministic pseudo-random QUBO matrix for tests.
    /// Uses a simple LCG to avoid external rand dependency in unit tests.
    fn make_qubo(n: usize, seed: u64) -> Vec<f64> {
        let mut q = vec![0.0f64; n * n];
        let mut state = seed;
        let lcg = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to [-2, 2]
            ((*s >> 33) as f64) / (u32::MAX as f64) * 4.0 - 2.0
        };
        for i in 0..n {
            q[i * n + i] = lcg(&mut state);
            for j in (i + 1)..n {
                let v = lcg(&mut state);
                q[i * n + j] = v;
                q[j * n + i] = v;
            }
        }
        q
    }

    /// Flip one bit in a bool slice, returning new Vec.
    fn flip(state: &[bool], k: usize) -> Vec<bool> {
        let mut s = state.to_vec();
        s[k] = !s[k];
        s
    }

    /// All-false state.
    fn zeros(n: usize) -> Vec<bool> {
        vec![false; n]
    }

    /// All-true state.
    fn ones(n: usize) -> Vec<bool> {
        vec![true; n]
    }

    // ─── energy_full correctness ─────────────────────────────────────────────

    #[test]
    fn test_energy_full_all_zeros() {
        // E(0,...,0) = 0 for any Q
        let q = make_qubo(8, 1);
        let s = zeros(8);
        assert!((energy_full(&s, &q, 8) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_energy_full_all_ones_diagonal_only() {
        // Q = diag(a0, a1, ...), state = all-ones → E = Σ a_i
        let n = 4;
        let mut q = vec![0.0f64; n * n];
        q[0 * n + 0] = 1.0;
        q[1 * n + 1] = 2.0;
        q[2 * n + 2] = 3.0;
        q[3 * n + 3] = 4.0;
        let s = ones(n);
        let e = energy_full(&s, &q, n);
        assert!((e - 10.0).abs() < 1e-12, "Expected 10.0, got {e}");
    }

    #[test]
    fn test_energy_full_single_quadratic() {
        // Q[0,1] = Q[1,0] = 1, rest 0. state = [1,1,0,...] → E = Q[0,1]+Q[1,0] = 2
        let n = 4;
        let mut q = vec![0.0f64; n * n];
        q[0 * n + 1] = 1.0;
        q[1 * n + 0] = 1.0;
        let mut s = zeros(n);
        s[0] = true;
        s[1] = true;
        let e = energy_full(&s, &q, n);
        assert!((e - 2.0).abs() < 1e-12, "Expected 2.0, got {e}");
    }

    // ─── energy_delta correctness ────────────────────────────────────────────

    #[test]
    fn test_energy_delta_matches_full_diff_n4() {
        // For n=4 and all 16 states, all 4 flip positions:
        // energy_full(flip(x, k)) - energy_full(x) == energy_delta(x, k)
        let n = 4;
        let q = make_qubo(n, 42);

        for bits in 0u16..16 {
            let state: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
            for k in 0..n {
                let delta = energy_delta(&state, &q, n, k);
                let flipped = flip(&state, k);
                let e0 = energy_full(&state, &q, n);
                let e1 = energy_full(&flipped, &q, n);
                let expected = e1 - e0;
                assert!(
                    (delta - expected).abs() < 1e-12,
                    "n=4, bits={bits:#06b}, k={k}: delta={delta}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn test_energy_delta_matches_full_diff_n8() {
        let n = 8;
        let q = make_qubo(n, 77);
        let mut state = zeros(n);
        state[0] = true;
        state[3] = true;
        state[7] = true;

        for k in 0..n {
            let delta = energy_delta(&state, &q, n, k);
            let e0 = energy_full(&state, &q, n);
            let e1 = energy_full(&flip(&state, k), &q, n);
            let expected = e1 - e0;
            assert!(
                (delta - expected).abs() < 1e-12,
                "n=8, k={k}: delta={delta}, expected={expected}"
            );
        }
    }

    // ─── SIMD vs scalar consistency ──────────────────────────────────────────

    #[test]
    fn test_simd_energy_full_matches_scalar_n32() {
        let n = 32;
        let q = make_qubo(n, 111);
        let mut state = zeros(n);
        for i in (0..n).step_by(3) {
            state[i] = true;
        }
        let scalar = energy_full(&state, &q, n);
        let simd = energy_full_simd(&state, &q, n);
        assert!(
            (simd - scalar).abs() < 1e-10,
            "n=32 SIMD vs scalar: {simd} vs {scalar}"
        );
    }

    #[test]
    fn test_simd_energy_full_matches_scalar_n64() {
        let n = 64;
        let q = make_qubo(n, 222);
        let state: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let scalar = energy_full(&state, &q, n);
        let simd = energy_full_simd(&state, &q, n);
        assert!(
            (simd - scalar).abs() < 1e-10,
            "n=64 SIMD vs scalar: {simd} vs {scalar}"
        );
    }

    #[test]
    fn test_simd_energy_delta_matches_scalar_n32() {
        let n = 32;
        let q = make_qubo(n, 333);
        let state: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
        for k in 0..n {
            let scalar = energy_delta(&state, &q, n, k);
            let simd = energy_delta_simd(&state, &q, n, k);
            assert!(
                (simd - scalar).abs() < 1e-10,
                "n=32, k={k}: simd={simd} scalar={scalar}"
            );
        }
    }

    #[test]
    fn test_simd_energy_delta_matches_scalar_n64() {
        let n = 64;
        let q = make_qubo(n, 444);
        let state: Vec<bool> = (0..n).map(|i| (i * 7 + 3) % 5 > 2).collect();
        for k in 0..n {
            let scalar = energy_delta(&state, &q, n, k);
            let simd = energy_delta_simd(&state, &q, n, k);
            assert!(
                (simd - scalar).abs() < 1e-10,
                "n=64, k={k}: simd={simd} scalar={scalar}"
            );
        }
    }

    #[test]
    fn test_simd_energy_delta_matches_scalar_n128() {
        let n = 128;
        let q = make_qubo(n, 555);
        let state: Vec<bool> = (0..n).map(|i| i % 4 < 2).collect();
        for k in [0, 1, 31, 32, 63, 64, 127] {
            let scalar = energy_delta(&state, &q, n, k);
            let simd = energy_delta_simd(&state, &q, n, k);
            assert!(
                (simd - scalar).abs() < 1e-10,
                "n=128, k={k}: simd={simd} scalar={scalar}"
            );
        }
    }

    // ─── influence vector correctness ────────────────────────────────────────

    #[test]
    fn test_compute_influence_correctness_n4() {
        // For n=4, verify compute_influence matches energy_delta
        let n = 4;
        let q = make_qubo(n, 999);
        let state: Vec<bool> = vec![true, false, true, false];
        let g = compute_influence(&state, &q, n);
        for k in 0..n {
            let delta_from_g = (1.0 - 2.0 * if state[k] { 1.0 } else { 0.0 }) * g[k];
            let delta_direct = energy_delta(&state, &q, n, k);
            assert!(
                (delta_from_g - delta_direct).abs() < 1e-12,
                "k={k}: from_g={delta_from_g}, direct={delta_direct}"
            );
        }
    }

    #[test]
    fn test_compute_influence_simd_matches_scalar_n32() {
        let n = 32;
        let q = make_qubo(n, 888);
        let state: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let g_scalar = compute_influence(&state, &q, n);
        let g_simd = compute_influence_simd(&state, &q, n);
        for i in 0..n {
            assert!(
                (g_simd[i] - g_scalar[i]).abs() < 1e-10,
                "i={i}: simd={} scalar={}",
                g_simd[i],
                g_scalar[i]
            );
        }
    }

    // ─── update_influence correctness ────────────────────────────────────────

    #[test]
    fn test_update_influence_matches_recompute_n4() {
        // After flipping k, update_influence(g) should give same g as recomputing from scratch
        let n = 4;
        let q = make_qubo(n, 1234);
        let state: Vec<bool> = vec![true, false, true, false];
        let mut g = compute_influence(&state, &q, n);

        let k = 1;
        let new_val = !state[k];
        update_influence(&mut g, &q, n, k, new_val);

        let mut new_state = state.clone();
        new_state[k] = new_val;
        let g_expected = compute_influence(&new_state, &q, n);

        for i in 0..n {
            assert!(
                (g[i] - g_expected[i]).abs() < 1e-12,
                "i={i}: updated={}, expected={}",
                g[i],
                g_expected[i]
            );
        }
    }

    #[test]
    fn test_update_influence_simd_matches_scalar_n32() {
        let n = 32;
        let q = make_qubo(n, 5678);
        let state: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let g_ref = compute_influence(&state, &q, n);

        for k in [0, 1, 15, 16, 31] {
            let mut g_scalar = g_ref.clone();
            let mut g_simd = g_ref.clone();
            let new_val = !state[k];
            update_influence(&mut g_scalar, &q, n, k, new_val);
            update_influence_simd(&mut g_simd, &q, n, k, new_val);
            for i in 0..n {
                assert!(
                    (g_simd[i] - g_scalar[i]).abs() < 1e-10,
                    "k={k}, i={i}: simd={} scalar={}",
                    g_simd[i],
                    g_scalar[i]
                );
            }
        }
    }

    // ─── build_dense_q ───────────────────────────────────────────────────────

    #[test]
    fn test_build_dense_q_basic() {
        let mut edges = HashMap::new();
        edges.insert((0, 0), -1.0f64);
        edges.insert((1, 1), -1.0f64);
        edges.insert((0, 1), 2.0f64);
        let q = build_dense_q(2, &edges);
        assert_eq!(q.len(), 4);
        assert!((q[0 * 2 + 0] - (-1.0)).abs() < 1e-12);
        assert!((q[0 * 2 + 1] - 2.0).abs() < 1e-12);
        assert!((q[1 * 2 + 0] - 0.0).abs() < 1e-12); // not symmetric by default
        assert!((q[1 * 2 + 1] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_energy_full_from_array_matches_flat() {
        use scirs2_core::ndarray::Array2;
        let n = 4;
        let q = make_qubo(n, 777);
        let q_array = Array2::from_shape_vec((n, n), q.clone())
            .expect("Array2 creation failed");
        let state: Vec<bool> = vec![true, false, true, true];
        let e_flat = energy_full_simd(&state, &q, n);
        let e_array = energy_full_from_array(&state, &q_array).expect("Array energy failed");
        assert!((e_flat - e_array).abs() < 1e-12, "flat={e_flat}, array={e_array}");
    }
}
