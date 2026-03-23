//! Eigenvalue solver helpers for quantum walk graph analysis.

use crate::error::QuantRS2Result;
use scirs2_core::ndarray::Array2;

// ─── Eigenvalue solver helpers ────────────────────────────────────────────────

/// Householder tridiagonalization of a real symmetric matrix A.
///
/// Returns `(diag, off_diag)` so that Q^T A Q = T where T is symmetric
/// tridiagonal with the returned diagonal and off-diagonal entries.
fn householder_tridiagonalize(a: &Array2<f64>) -> QuantRS2Result<(Vec<f64>, Vec<f64>)> {
    let n = a.nrows();
    let mut mat = a.to_owned();

    for k in 0..n.saturating_sub(2) {
        // Householder reflector for column k, rows k+1..n
        let col: Vec<f64> = (k + 1..n).map(|i| mat[[i, k]]).collect();
        let sigma: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        if sigma < 1e-14 {
            continue;
        }

        // Choose sign to maximize numerical stability
        let mut v = col.clone();
        v[0] += if col[0] >= 0.0 { sigma } else { -sigma };

        let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-28 {
            continue;
        }

        let m = v.len(); // = n - (k+1)

        // Apply H from the left: mat[k+1.., ..] -= (2/v^Tv) v (v^T mat[k+1.., ..])
        for j in k..n {
            let dot: f64 = (0..m).map(|i| v[i] * mat[[k + 1 + i, j]]).sum::<f64>();
            let scale = 2.0 * dot / v_norm_sq;
            for i in 0..m {
                mat[[k + 1 + i, j]] -= scale * v[i];
            }
        }

        // Apply H from the right: mat[.., k+1..] -= (2/v^Tv) (mat[.., k+1..] v) v^T
        for i in 0..n {
            let dot: f64 = (0..m).map(|j| mat[[i, k + 1 + j]] * v[j]).sum::<f64>();
            let scale = 2.0 * dot / v_norm_sq;
            for j in 0..m {
                mat[[i, k + 1 + j]] -= scale * v[j];
            }
        }
    }

    let mut diag = vec![0.0f64; n];
    let mut off = vec![0.0f64; n.saturating_sub(1)];
    for i in 0..n {
        diag[i] = mat[[i, i]];
    }
    for i in 0..n.saturating_sub(1) {
        // Average the symmetric pair to cancel floating-point asymmetry
        off[i] = (mat[[i, i + 1]] + mat[[i + 1, i]]) * 0.5;
    }
    Ok((diag, off))
}

/// Count the number of eigenvalues < λ for a symmetric tridiagonal matrix
/// using the Sturm sequence.
///
/// The Sturm sequence for T is {p₀=1, p₁=d[0]-λ, pₖ=(d[k-1]-λ)pₖ₋₁ - e[k-2]²pₖ₋₂}.
/// The number of sign changes in this sequence equals the number of eigenvalues < λ.
fn sturm_count(diag: &[f64], off_sq: &[f64], lambda: f64) -> usize {
    let n = diag.len();
    if n == 0 {
        return 0;
    }
    // Track the full Sturm sequence including p₀ = 1
    let mut count = 0usize;
    let mut p_prev = 1.0f64; // p₀ = 1
    let mut p_curr = diag[0] - lambda; // p₁ = d[0] - λ

    // Check sign change between p₀ and p₁
    if p_curr < 0.0 {
        count += 1;
    }

    for i in 1..n {
        // pₖ₊₁ = (d[i] - λ) * pₖ - e[i-1]² * pₖ₋₁
        let p_next = if p_curr.abs() < f64::MIN_POSITIVE {
            // Avoid division by zero / loss of precision: use a small perturbation.
            // The exact value doesn't matter for sign-change counting as long as
            // we handle the zero case: treat zero as having the SAME sign as prev.
            // Use the recurrence with a small positive perturbation.
            -(off_sq[i - 1] / f64::EPSILON)
        } else {
            (diag[i] - lambda) * p_curr - off_sq[i - 1] * p_prev
        };

        if (p_next < 0.0) != (p_curr < 0.0) && p_curr != 0.0 {
            count += 1;
        }

        p_prev = p_curr;
        p_curr = p_next;
    }
    count
}

/// Compute all eigenvalues of a real symmetric tridiagonal matrix using the
/// bisection method (Sturm sequence).  This is O(n² log(1/ε)) but
/// unconditionally stable and correct.  Returns eigenvalues in ascending order.
fn qr_iteration_tridiagonal(diag: &[f64], off: &[f64]) -> QuantRS2Result<Vec<f64>> {
    let n = diag.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![diag[0]]);
    }
    // Precompute squared off-diagonals (only the square appears in Sturm sequences)
    let off_sq: Vec<f64> = off.iter().map(|e| e * e).collect();

    // Compute a global eigenvalue bracket using Gershgorin circles
    let lo_bound = diag
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let r = (if i > 0 { off_sq[i - 1].sqrt() } else { 0.0 })
                + (if i < n - 1 { off_sq[i].sqrt() } else { 0.0 });
            d - r
        })
        .fold(f64::INFINITY, f64::min);
    let hi_bound = diag
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let r = (if i > 0 { off_sq[i - 1].sqrt() } else { 0.0 })
                + (if i < n - 1 { off_sq[i].sqrt() } else { 0.0 });
            d + r
        })
        .fold(f64::NEG_INFINITY, f64::max);

    // For each eigenvalue index k (0-based, ascending), bisect to find λ_k
    // such that exactly k eigenvalues lie below λ_k.
    let mut eigenvalues = Vec::with_capacity(n);
    let tol = 1e-11 * (hi_bound - lo_bound + 1.0);

    for k in 0..n {
        // Find bracket [a, b] such that count(a) = k and count(b) = k+1
        // Starting from the global bracket, narrow down
        let mut a = lo_bound - 1.0;
        let mut b = hi_bound + 1.0;

        // Refine a so that sturm_count(a) = k
        // and b so that sturm_count(b) >= k+1
        // Binary search within [lo-1, hi+1]:
        let count_a_target = k;
        // First ensure b gives count > k
        // then ensure a gives count <= k
        // Then bisect until |b-a| < tol
        for _ in 0..200 {
            let mid = (a + b) * 0.5;
            let cnt = sturm_count(diag, &off_sq, mid);
            if cnt <= count_a_target {
                a = mid;
            } else {
                b = mid;
            }
            if b - a < tol {
                break;
            }
        }
        eigenvalues.push((a + b) * 0.5);
    }

    Ok(eigenvalues)
}

/// Compute all eigenvalues of a real symmetric (Laplacian) matrix.
///
/// Internally uses Householder tridiagonalization followed by Golub-Reinsch
/// QR iteration.  The returned eigenvalues are sorted in ascending order and
/// clamped to be non-negative (Laplacian eigenvalues are always ≥ 0).
pub(crate) fn compute_laplacian_eigenvalues_impl(
    laplacian: &Array2<f64>,
) -> QuantRS2Result<Vec<f64>> {
    let n = laplacian.nrows();
    if n == 0 {
        return Ok(vec![]);
    }
    if n == 1 {
        return Ok(vec![laplacian[[0, 0]]]);
    }

    let (tri_diag, tri_off) = householder_tridiagonalize(laplacian)?;
    let mut eigenvalues = qr_iteration_tridiagonal(&tri_diag, &tri_off)?;

    // Clamp tiny negatives from floating-point rounding
    for ev in &mut eigenvalues {
        if *ev < 0.0 {
            *ev = 0.0;
        }
    }

    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(eigenvalues)
}

/// Estimate the Fiedler value (second smallest eigenvalue of the Laplacian) via
/// Rayleigh quotient power iteration restricted to the subspace orthogonal to
/// the all-ones vector.
pub(crate) fn estimate_fiedler_value_impl(laplacian: &Array2<f64>) -> f64 {
    let n = laplacian.nrows();
    if n <= 1 {
        return 0.0;
    }

    // Initial vector: alternating ±1/√n, naturally orthogonal to constant vector
    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let mut v: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { inv_sqrt_n } else { -inv_sqrt_n })
        .collect();

    // Explicit orthogonalization against all-ones (handles odd-length case)
    let mean: f64 = v.iter().sum::<f64>() / n as f64;
    for x in &mut v {
        *x -= mean;
    }
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm < 1e-14 {
        return 0.0;
    }
    for x in &mut v {
        *x /= norm;
    }

    let mut rayleigh = 0.0_f64;

    for _ in 0..200 {
        // w = L * v
        let mut w = vec![0.0f64; n];
        for i in 0..n {
            let mut acc = 0.0f64;
            for j in 0..n {
                acc += laplacian[[i, j]] * v[j];
            }
            w[i] = acc;
        }

        // Rayleigh quotient  λ ≈ (v^T w) / (v^T v)
        let vw: f64 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        let vv: f64 = v.iter().map(|x| x * x).sum();
        let new_rayleigh = if vv > 1e-28 { vw / vv } else { 0.0 };

        if (new_rayleigh - rayleigh).abs() < 1e-12 * (1.0 + new_rayleigh.abs()) {
            rayleigh = new_rayleigh;
            break;
        }
        rayleigh = new_rayleigh;

        // Normalize w for the next power-iteration step
        let w_norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if w_norm < 1e-14 {
            break;
        }
        v = w.iter().map(|x| x / w_norm).collect();

        // Re-orthogonalize against the null-space direction [1,1,...,1]
        let proj: f64 = v.iter().sum::<f64>() / n as f64;
        for x in &mut v {
            *x -= proj;
        }

        let norm2: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm2 < 1e-14 {
            break;
        }
        for x in &mut v {
            *x /= norm2;
        }
    }

    rayleigh.max(0.0)
}
