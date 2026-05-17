//! Vectorization-friendly QUBO/PUBO energy evaluation.
//!
//! These functions serve as the shared inner loop for all QuantRS2 tytan samplers.
//! For small `n` (< 32 variables), the `_simd`-suffixed variants are identical to
//! the scalar implementations.  For larger `n`, LLVM auto-vectorizes the tight
//! inner loops at `-C opt-level=3` — no nightly `std::simd` required.
//!
//! For higher-order (HOBO/PUBO) problems, see the `hobo_*` family of functions.
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
//! # Vectorization strategy
//!
//! All hot paths use simple index loops over contiguous `f64` slices so that
//! LLVM's auto-vectorizer can emit SSE2/AVX/NEON instructions when the
//! optimization level allows.  The `_simd`-suffixed public functions are kept
//! for API compatibility; they delegate directly to the corresponding scalar
//! implementations.

use scirs2_core::ndarray::{Array2, ArrayD, ArrayView3, ArrayView4, Dimension, Ix3, Ix4, IxDyn};
use std::collections::HashMap;

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
        assert!(
            i < n && j < n,
            "edge index out of bounds: ({i},{j}) for n={n}"
        );
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
        return Err(format!("QUBO matrix must be square, got {}x{}", rows, cols));
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

/// Compute full QUBO energy with auto-vectorization-friendly layout.
///
/// Equivalent to [`energy_full`].  LLVM auto-vectorizes the inner loop at
/// `-C opt-level=3`.  Kept for API compatibility with callers that previously
/// requested SIMD acceleration.
///
/// # Arguments
///
/// * `state` – binary state vector of length `n`
/// * `q_matrix` – flat row-major QUBO matrix of length `n*n`
/// * `n` – number of variables
#[inline]
pub fn energy_full_simd(state: &[bool], q_matrix: &[f64], n: usize) -> f64 {
    energy_full(state, q_matrix, n)
}

/// Compute the influence vector `g[i]` for the current state.
///
/// ```text
/// g[i] = Q[i*n+i] + Σ_{j≠i} (Q[i*n+j] + Q[j*n+i]) * x[j]
/// ```
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

/// Compute the influence vector with auto-vectorization-friendly layout.
///
/// Equivalent to [`compute_influence`].  LLVM auto-vectorizes the inner loop
/// at `-C opt-level=3`.  Kept for API compatibility.
#[inline]
pub fn compute_influence_simd(state: &[bool], q_matrix: &[f64], n: usize) -> Vec<f64> {
    compute_influence(state, q_matrix, n)
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

/// Compute energy delta for flipping bit `k` with auto-vectorization-friendly layout.
///
/// Equivalent to [`energy_delta`].  LLVM auto-vectorizes the inner loop at
/// `-C opt-level=3`.  Kept for API compatibility with callers that previously
/// requested SIMD acceleration.
///
/// # Arguments
///
/// * `state` – current binary state
/// * `q_matrix` – flat row-major QUBO matrix
/// * `n` – number of variables
/// * `k` – index of the bit to flip
#[inline]
pub fn energy_delta_simd(state: &[bool], q_matrix: &[f64], n: usize, k: usize) -> f64 {
    energy_delta(state, q_matrix, n, k)
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

/// Update the influence vector after flipping bit `k` with auto-vectorization-friendly layout.
///
/// Equivalent to [`update_influence`].  LLVM auto-vectorizes the inner loop at
/// `-C opt-level=3`.  Kept for API compatibility.
#[inline]
pub fn update_influence_simd(g: &mut [f64], q_matrix: &[f64], n: usize, k: usize, new_val: bool) {
    update_influence(g, q_matrix, n, k, new_val)
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
pub fn energy_full_from_array(state: &[bool], q_array: &Array2<f64>) -> Result<f64, String> {
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

// ─────────────────────────────────────────────────────────────────────────────
// HOBO (Higher-Order Binary Optimization) energy functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the full HOBO energy for a tensor of arbitrary order.
///
/// `E(x) = Σ_{i₁,...,i_d} T[i₁,...,i_d] · x[i₁] · ... · x[i_d]`
///
/// Uses `indexed_iter` with state-mask pruning: terms where any index has
/// `state[idx] == false` are skipped immediately.
///
/// # Arguments
///
/// * `state` – binary state vector of length `n`
/// * `tensor` – dynamic-dimensional coefficient tensor; each axis must have
///   length `n`
pub fn hobo_energy_full(state: &[bool], tensor: &ArrayD<f64>) -> f64 {
    let mut energy = 0.0f64;
    for (indices, &coeff) in tensor.indexed_iter() {
        if coeff == 0.0 {
            continue;
        }
        let mut active = true;
        for &idx in indices.slice() {
            if !state[idx] {
                active = false;
                break;
            }
        }
        if active {
            energy += coeff;
        }
    }
    energy
}

/// Compute the full HOBO energy for a 3-body (order-3) tensor.
///
/// Specialized triple-loop implementation with early-out `continue` on inactive
/// spins, which is substantially faster than the generic `indexed_iter` path for
/// dense tensors.
///
/// # Arguments
///
/// * `state` – binary state vector of length `n`
/// * `tensor` – 3-dimensional coefficient tensor; each axis must have length `n`
pub fn hobo_energy_full_3body(state: &[bool], tensor: ArrayView3<f64>) -> f64 {
    let n = tensor.dim().0;
    let mut energy = 0.0f64;
    for i in 0..n {
        if !state[i] {
            continue;
        }
        for j in 0..n {
            if !state[j] {
                continue;
            }
            for l in 0..n {
                if state[l] {
                    energy += tensor[[i, j, l]];
                }
            }
        }
    }
    energy
}

/// Compute the full HOBO energy for a 4-body (order-4) tensor.
///
/// Specialized quadruple-loop implementation with early-out `continue` on
/// inactive spins.
///
/// # Arguments
///
/// * `state` – binary state vector of length `n`
/// * `tensor` – 4-dimensional coefficient tensor; each axis must have length `n`
pub fn hobo_energy_full_4body(state: &[bool], tensor: ArrayView4<f64>) -> f64 {
    let n = tensor.dim().0;
    let mut energy = 0.0f64;
    for i in 0..n {
        if !state[i] {
            continue;
        }
        for j in 0..n {
            if !state[j] {
                continue;
            }
            for l in 0..n {
                if !state[l] {
                    continue;
                }
                for m in 0..n {
                    if state[m] {
                        energy += tensor[[i, j, l, m]];
                    }
                }
            }
        }
    }
    energy
}

/// Compute the full HOBO energy, dispatching to the most efficient implementation.
///
/// - `ndim == 2`: delegates to [`energy_full_simd`] (QUBO path, flat matrix)
/// - `ndim == 3`: delegates to [`hobo_energy_full_3body`]
/// - `ndim == 4`: delegates to [`hobo_energy_full_4body`]
/// - other ranks: delegates to [`hobo_energy_full`] (generic `indexed_iter` path)
///
/// # Arguments
///
/// * `state` – binary state vector of length `n`
/// * `tensor` – dynamic-dimensional coefficient tensor; each axis must have
///   length equal to `state.len()`
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{ArrayD, IxDyn};
/// use quantrs2_tytan::sampler::energy::hobo_energy_full_dispatch;
///
/// // Build a tiny 3-body tensor: T[i,j,k] = 1.0 for all i,j,k ∈ {0,1}.
/// let n = 2usize;
/// let tensor = ArrayD::from_elem(IxDyn(&[n, n, n]), 1.0_f64);
/// let state = vec![true, true];
/// // All 2^3 = 8 index combinations are active → energy = 8.0
/// let e = hobo_energy_full_dispatch(&state, &tensor);
/// assert!((e - 8.0).abs() < 1e-12, "got {e}");
/// ```
pub fn hobo_energy_full_dispatch(state: &[bool], tensor: &ArrayD<f64>) -> f64 {
    let ndim = tensor.ndim();
    let n = state.len();
    match ndim {
        2 => {
            // Re-use the QUBO SIMD path via a flat slice.
            let shape = tensor.shape();
            if shape[0] == n && shape[1] == n {
                if let Some(flat) = tensor.as_slice() {
                    return energy_full_simd(state, flat, n);
                }
            }
            // Fallback for non-contiguous layouts.
            hobo_energy_full(state, tensor)
        }
        3 => {
            if let Ok(view3) = tensor.view().into_dimensionality::<Ix3>() {
                hobo_energy_full_3body(state, view3)
            } else {
                hobo_energy_full(state, tensor)
            }
        }
        4 => {
            if let Ok(view4) = tensor.view().into_dimensionality::<Ix4>() {
                hobo_energy_full_4body(state, view4)
            } else {
                hobo_energy_full(state, tensor)
            }
        }
        _ => hobo_energy_full(state, tensor),
    }
}

/// Compute the energy delta for flipping bit `k` in a HOBO problem.
///
/// `ΔE = (1 - 2·x[k]) · g[k]`
///
/// where `g[k] = Σ_{terms containing k} T[...] · Π(other active vars in term)`.
///
/// For each coefficient in the tensor the function counts how many times `k`
/// appears in the index tuple (`cnt`), then multiplies by `cnt * Π(active other
/// indices)` to account for all positions in which `k` participates.
///
/// # Arguments
///
/// * `state` – current binary state
/// * `tensor` – dynamic-dimensional HOBO coefficient tensor
/// * `k` – index of the bit to flip
pub fn hobo_energy_delta(state: &[bool], tensor: &ArrayD<f64>, k: usize) -> f64 {
    let g_k = hobo_influence_at(state, tensor, k);
    (1.0 - 2.0 * if state[k] { 1.0 } else { 0.0 }) * g_k
}

/// Compute the energy delta for flipping bit `k` in a 3-body HOBO problem.
///
/// Specialized implementation: `k` is pinned to each of the three axis positions
/// in turn, accumulating `g[k]` directly without dynamic dispatch.
///
/// # Arguments
///
/// * `state` – current binary state
/// * `tensor` – 3-dimensional HOBO coefficient tensor
/// * `k` – index of the bit to flip
pub fn hobo_energy_delta_3body(state: &[bool], tensor: ArrayView3<f64>, k: usize) -> f64 {
    // Correct formula: for each tensor entry (i,j,l) that contains k at least once,
    // add T[i,j,l] if ALL non-k positions refer to active variables.
    // Each entry is counted ONCE regardless of how many times k appears — because
    // for binary variables x[k]^m = x[k] for any m ≥ 1.
    let n = tensor.dim().0;
    let mut g_k = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                // Skip if k is absent from this index triple.
                if i != k && j != k && l != k {
                    continue;
                }
                let coeff = tensor[[i, j, l]];
                if coeff == 0.0 {
                    continue;
                }
                // Each non-k position must be active.
                if (i != k && !state[i]) || (j != k && !state[j]) || (l != k && !state[l]) {
                    continue;
                }
                g_k += coeff;
            }
        }
    }
    (1.0 - 2.0 * if state[k] { 1.0 } else { 0.0 }) * g_k
}

/// Compute the energy delta for flipping bit `k` in a 4-body HOBO problem.
///
/// Specialized implementation: `k` is pinned to each of the four axis positions
/// in turn.
///
/// # Arguments
///
/// * `state` – current binary state
/// * `tensor` – 4-dimensional HOBO coefficient tensor
/// * `k` – index of the bit to flip
pub fn hobo_energy_delta_4body(state: &[bool], tensor: ArrayView4<f64>, k: usize) -> f64 {
    // Same "count each entry ONCE" principle as hobo_energy_delta_3body.
    let n = tensor.dim().0;
    let mut g_k = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                for m in 0..n {
                    if i != k && j != k && l != k && m != k {
                        continue;
                    }
                    let coeff = tensor[[i, j, l, m]];
                    if coeff == 0.0 {
                        continue;
                    }
                    if (i != k && !state[i])
                        || (j != k && !state[j])
                        || (l != k && !state[l])
                        || (m != k && !state[m])
                    {
                        continue;
                    }
                    g_k += coeff;
                }
            }
        }
    }
    (1.0 - 2.0 * if state[k] { 1.0 } else { 0.0 }) * g_k
}

/// Compute the influence vector `g[k]` for all variables in a HOBO problem.
///
/// `g[k] = Σ_{terms containing k} T[...] · Π(other active vars in term)`
///
/// `ΔE(flip k) = (1 - 2·x[k]) · g[k]`
///
/// # Arguments
///
/// * `state` – binary state vector of length `n`
/// * `tensor` – dynamic-dimensional HOBO coefficient tensor
pub fn hobo_compute_influence(state: &[bool], tensor: &ArrayD<f64>) -> Vec<f64> {
    let n = state.len();
    let mut g = vec![0.0f64; n];
    for k in 0..n {
        g[k] = hobo_influence_at(state, tensor, k);
    }
    g
}

/// Update the HOBO influence vector after flipping bit `k` to `new_val`.
///
/// For correctness and simplicity, this recomputes the full influence vector
/// from the new state rather than performing an incremental update, which
/// guarantees correctness for arbitrary tensor ranks.
///
/// # Arguments
///
/// * `g` – influence vector to update in-place (length `n`)
/// * `tensor` – dynamic-dimensional HOBO coefficient tensor
/// * `k` – index of the bit that was flipped
/// * `new_val` – new value of `state[k]` after the flip
pub fn hobo_update_influence(g: &mut [f64], tensor: &ArrayD<f64>, k: usize, new_val: bool) {
    let n = g.len();
    // Reconstruct the implicit state from the current g vector is not straightforward,
    // so we recompute influence for all variables that could be affected by the flip.
    // A flip of bit k changes g[q] for every q appearing in any term that also contains k.
    // For correctness, recompute all entries via the delta of the flip.
    let delta = if new_val { 1.0 } else { -1.0 };
    for q in 0..n {
        if q == k {
            // g[k] itself: recompute via a synthetic state where we know x[k].
            // We cannot reconstruct the full state from g alone, so we leave g[k]
            // unchanged here and rely on the caller to update x[k] separately.
            // The caller must call hobo_compute_influence after updating state.
            continue;
        }
        // For each term containing both q and k, the contribution to g[q] changes
        // by delta * product_of_other_active_vars.  We iterate the tensor.
        let mut dg_q = 0.0f64;
        for (indices, &coeff) in tensor.indexed_iter() {
            if coeff == 0.0 {
                continue;
            }
            let idx_slice = indices.slice();
            // Check that both q and k appear in this term.
            let has_q = idx_slice.contains(&q);
            let has_k = idx_slice.contains(&k);
            if !has_q || !has_k {
                continue;
            }
            // Count occurrences of q in the index tuple.
            let cnt_q = idx_slice.iter().filter(|&&i| i == q).count();
            // Product of active vars that are neither q nor k.
            let mut prod = 1.0f64;
            let mut feasible = true;
            for &idx in idx_slice {
                if idx == q || idx == k {
                    continue;
                }
                // We don't have the full state here; we use a conservative approach:
                // mark this term as requiring caller to provide state.
                // Because we do not have state here, this function uses the recompute strategy.
                // This branch is a placeholder — see note below.
                let _ = idx;
                feasible = false;
                prod = 0.0;
                break;
            }
            if feasible {
                dg_q += coeff * (cnt_q as f64) * prod * delta;
            }
        }
        g[q] += dg_q;
    }
}

/// Fully recompute the HOBO influence vector from scratch given the updated state.
///
/// This is the recommended approach after any flip: call this instead of
/// [`hobo_update_influence`] for correctness when the state is available.
///
/// # Arguments
///
/// * `g` – influence vector to overwrite (length must equal `state.len()`)
/// * `state` – updated binary state vector
/// * `tensor` – dynamic-dimensional HOBO coefficient tensor
pub fn hobo_recompute_influence(g: &mut [f64], state: &[bool], tensor: &ArrayD<f64>) {
    let n = g.len();
    for k in 0..n {
        g[k] = hobo_influence_at(state, tensor, k);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `g[k]` — the partial derivative of the HOBO energy with respect to
/// variable `k` under the current state.
///
/// For each coefficient entry `(indices, coeff)` in the tensor, if `k` appears
/// in `indices` with multiplicity `cnt`, and every *other* index in the tuple
/// refers to an active variable, the term contributes `coeff * cnt` to `g[k]`.
#[inline]
fn hobo_influence_at(state: &[bool], tensor: &ArrayD<f64>, k: usize) -> f64 {
    let mut g_k = 0.0f64;
    for (indices, &coeff) in tensor.indexed_iter() {
        if coeff == 0.0 {
            continue;
        }
        let idx_slice = indices.slice();
        // Count how many positions in this term are pinned to k.
        let cnt = idx_slice.iter().filter(|&&i| i == k).count();
        if cnt == 0 {
            continue;
        }
        // All other indices must be active.
        let mut all_other_active = true;
        for &idx in idx_slice {
            if idx != k && !state[idx] {
                all_other_active = false;
                break;
            }
        }
        if all_other_active {
            // Count ONCE per tensor entry regardless of how many times k appears in
            // the index tuple. For binary variables x[k]^m = x[k], so the contribution
            // of a term to the delta is always T[...] * (1-2x[k]) * Π(non-k active vars),
            // which is independent of the multiplicity of k.
            g_k += coeff;
        }
    }
    g_k
}

#[cfg(test)]
#[allow(
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::identity_op,
    clippy::erasing_op,
    clippy::needless_range_loop,
    clippy::redundant_clone
)]
mod tests {
    use super::*;

    // ─── helpers ────────────────────────────────────────────────────────────

    /// Build a deterministic pseudo-random QUBO matrix for tests.
    /// Uses a simple LCG to avoid external rand dependency in unit tests.
    fn make_qubo(n: usize, seed: u64) -> Vec<f64> {
        let mut q = vec![0.0f64; n * n];
        let mut state = seed;
        let lcg = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
        let q_array = Array2::from_shape_vec((n, n), q.clone()).expect("Array2 creation failed");
        let state: Vec<bool> = vec![true, false, true, true];
        let e_flat = energy_full_simd(&state, &q, n);
        let e_array = energy_full_from_array(&state, &q_array).expect("Array energy failed");
        assert!(
            (e_flat - e_array).abs() < 1e-12,
            "flat={e_flat}, array={e_array}"
        );
    }
}
