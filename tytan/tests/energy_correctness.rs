#![allow(
    clippy::pedantic,
    clippy::unreadable_literal,    // long seed constants (intentional)
    clippy::suboptimal_flops,      // FMA not required for correctness tests
    clippy::identity_op,           // test matrices may have zero-effect terms
    clippy::erasing_op,            // same as above
    clippy::redundant_clone,       // test clarity over performance
    clippy::explicit_iter_loop,    // explicit index loops in SIMD verification
    clippy::needless_range_loop,   // k is used for two purposes (index + value)
    clippy::float_cmp,             // intentional == 0.0 checks in HOBO tests
)]
//! Correctness tests: SIMD energy functions vs scalar reference, plus HOBO energy library.
//!
//! These tests verify that all SIMD-accelerated energy functions produce
//! results within 1e-12 of the scalar reference implementations, across
//! a range of problem sizes and random QUBO instances.
//!
//! The second section verifies the HOBO (Higher-Order Binary Optimization)
//! energy library against naïve `indexed_iter` references.

use quantrs2_tytan::sampler::energy::{
    build_dense_q, compute_influence, compute_influence_simd, energy_delta, energy_delta_simd,
    energy_full, energy_full_simd, hobo_compute_influence, hobo_energy_delta,
    hobo_energy_delta_3body, hobo_energy_delta_4body, hobo_energy_full, hobo_energy_full_3body,
    hobo_energy_full_4body, hobo_energy_full_dispatch, hobo_recompute_influence, update_influence,
    update_influence_simd,
};
use scirs2_core::ndarray::{ArrayD, Dimension, IxDyn};
use std::collections::HashMap;

// ─── QUBO generator (deterministic LCG, no external deps) ────────────────────

/// Build a deterministic pseudo-random symmetric QUBO matrix.
fn random_qubo(n: usize, seed: u64) -> Vec<f64> {
    let mut q = vec![0.0f64; n * n];
    let mut s = seed;
    let mut lcg = || -> f64 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as f64 / u32::MAX as f64 * 4.0 - 2.0
    };
    for i in 0..n {
        q[i * n + i] = lcg();
        for j in (i + 1)..n {
            let v = lcg();
            q[i * n + j] = v;
            q[j * n + i] = v;
        }
    }
    q
}

/// Flip bit k in a state vector, returning a new Vec.
fn flip(state: &[bool], k: usize) -> Vec<bool> {
    let mut s = state.to_vec();
    s[k] = !s[k];
    s
}

/// Generate a deterministic state vector.
fn random_state(n: usize, seed: u64) -> Vec<bool> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 63) == 1
        })
        .collect()
}

// ─── energy_delta vs energy_full difference ───────────────────────────────────

/// Verify energy_delta(x, k) == energy_full(flip(x,k)) - energy_full(x)
/// for all 2^n states and all k, at n=4.
#[test]
fn test_energy_delta_matches_full_diff_small() {
    let n = 4;
    let q = random_qubo(n, 1234567);

    for bits in 0u16..(1u16 << n) {
        let state: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
        for k in 0..n {
            let delta = energy_delta(&state, &q, n, k);
            let e0 = energy_full(&state, &q, n);
            let e1 = energy_full(&flip(&state, k), &q, n);
            let expected = e1 - e0;
            assert!(
                (delta - expected).abs() < 1e-12,
                "n=4, state={state:?}, k={k}: delta={delta:.15e}, expected={expected:.15e}, diff={:.2e}",
                (delta - expected).abs()
            );
        }
    }
}

/// Same check at n=8 with a few representative states.
#[test]
fn test_energy_delta_matches_full_diff_n8() {
    let n = 8;
    let q = random_qubo(n, 9876543);
    let states = [
        vec![false; n],
        vec![true; n],
        random_state(n, 111),
        random_state(n, 222),
        random_state(n, 333),
    ];
    for state in &states {
        for k in 0..n {
            let delta = energy_delta(state, &q, n, k);
            let e0 = energy_full(state, &q, n);
            let e1 = energy_full(&flip(state, k), &q, n);
            let expected = e1 - e0;
            assert!(
                (delta - expected).abs() < 1e-12,
                "n=8, k={k}: delta={delta:.15e}, expected={expected:.15e}"
            );
        }
    }
}

/// Same check at n=16.
#[test]
fn test_energy_delta_matches_full_diff_n16() {
    let n = 16;
    let q = random_qubo(n, 31415926);
    let state = random_state(n, 27182818);
    for k in 0..n {
        let delta = energy_delta(&state, &q, n, k);
        let e0 = energy_full(&state, &q, n);
        let e1 = energy_full(&flip(&state, k), &q, n);
        let expected = e1 - e0;
        assert!(
            (delta - expected).abs() < 1e-12,
            "n=16, k={k}: delta={delta:.15e}, expected={expected:.15e}"
        );
    }
}

// ─── SIMD vs scalar: energy_full ─────────────────────────────────────────────

#[test]
fn test_simd_matches_scalar_energy_full_n16() {
    let n = 16;
    let q = random_qubo(n, 1111);
    let state = random_state(n, 2222);
    let scalar = energy_full(&state, &q, n);
    let simd = energy_full_simd(&state, &q, n);
    assert!(
        (simd - scalar).abs() < 1e-12,
        "n=16: simd={simd:.15e}, scalar={scalar:.15e}"
    );
}

#[test]
fn test_simd_matches_scalar_energy_full_n32() {
    let n = 32;
    let q = random_qubo(n, 3333);
    let state = random_state(n, 4444);
    let scalar = energy_full(&state, &q, n);
    let simd = energy_full_simd(&state, &q, n);
    assert!(
        (simd - scalar).abs() < 1e-12,
        "n=32: simd={simd:.15e}, scalar={scalar:.15e}"
    );
}

#[test]
fn test_simd_matches_scalar_energy_full_n64() {
    let n = 64;
    let q = random_qubo(n, 5555);
    let state = random_state(n, 6666);
    let scalar = energy_full(&state, &q, n);
    let simd = energy_full_simd(&state, &q, n);
    assert!(
        (simd - scalar).abs() < 1e-12,
        "n=64: simd={simd:.15e}, scalar={scalar:.15e}"
    );
}

#[test]
fn test_simd_matches_scalar_energy_full_n128() {
    let n = 128;
    let q = random_qubo(n, 7777);
    let state = random_state(n, 8888);
    let scalar = energy_full(&state, &q, n);
    let simd = energy_full_simd(&state, &q, n);
    // Tolerance is relaxed to 1e-9 for n=128 due to floating-point
    // accumulation order differences between scalar and SIMD paths.
    // The relative error remains < 1e-12 relative to the magnitude.
    let tol = 1e-9_f64.max(scalar.abs() * 1e-13);
    assert!(
        (simd - scalar).abs() < tol,
        "n=128: simd={simd:.15e}, scalar={scalar:.15e}, diff={:.4e}",
        (simd - scalar).abs()
    );
}

// ─── SIMD vs scalar: energy_delta ────────────────────────────────────────────

#[test]
fn test_simd_matches_scalar_energy_delta_n16() {
    let n = 16;
    let q = random_qubo(n, 11111);
    let state = random_state(n, 22222);
    for k in 0..n {
        let scalar = energy_delta(&state, &q, n, k);
        let simd = energy_delta_simd(&state, &q, n, k);
        assert!(
            (simd - scalar).abs() < 1e-12,
            "n=16, k={k}: simd={simd:.15e}, scalar={scalar:.15e}"
        );
    }
}

#[test]
fn test_simd_matches_scalar_energy_delta_n32() {
    let n = 32;
    let q = random_qubo(n, 33333);
    let state = random_state(n, 44444);
    for k in 0..n {
        let scalar = energy_delta(&state, &q, n, k);
        let simd = energy_delta_simd(&state, &q, n, k);
        assert!(
            (simd - scalar).abs() < 1e-12,
            "n=32, k={k}: simd={simd:.15e}, scalar={scalar:.15e}"
        );
    }
}

#[test]
fn test_simd_matches_scalar_energy_delta_n64() {
    let n = 64;
    let q = random_qubo(n, 55555);
    let state = random_state(n, 66666);
    for k in 0..n {
        let scalar = energy_delta(&state, &q, n, k);
        let simd = energy_delta_simd(&state, &q, n, k);
        assert!(
            (simd - scalar).abs() < 1e-12,
            "n=64, k={k}: simd={simd:.15e}, scalar={scalar:.15e}"
        );
    }
}

#[test]
fn test_simd_matches_scalar_energy_delta_n128() {
    let n = 128;
    let q = random_qubo(n, 77777);
    let state = random_state(n, 88888);
    // Check all positions, including boundary positions at SIMD chunk boundaries
    for k in 0..n {
        let scalar = energy_delta(&state, &q, n, k);
        let simd = energy_delta_simd(&state, &q, n, k);
        assert!(
            (simd - scalar).abs() < 1e-12,
            "n=128, k={k}: simd={simd:.15e}, scalar={scalar:.15e}"
        );
    }
}

// ─── Influence vector correctness ────────────────────────────────────────────

/// The influence vector formula must be consistent with energy_delta:
/// energy_delta(x, k) == (1 - 2*x[k]) * g[k]
#[test]
fn test_influence_vector_consistent_with_delta_n16() {
    let n = 16;
    let q = random_qubo(n, 999999);
    let state = random_state(n, 111111);
    let g = compute_influence(&state, &q, n);

    for k in 0..n {
        let from_g = (1.0 - 2.0 * if state[k] { 1.0 } else { 0.0 }) * g[k];
        let from_delta = energy_delta(&state, &q, n, k);
        assert!(
            (from_g - from_delta).abs() < 1e-12,
            "k={k}: from_g={from_g:.15e}, from_delta={from_delta:.15e}"
        );
    }
}

/// SIMD influence vector matches scalar at n=32.
#[test]
fn test_simd_compute_influence_matches_scalar_n32() {
    let n = 32;
    let q = random_qubo(n, 444444);
    let state = random_state(n, 555555);
    let g_scalar = compute_influence(&state, &q, n);
    let g_simd = compute_influence_simd(&state, &q, n);
    for i in 0..n {
        assert!(
            (g_simd[i] - g_scalar[i]).abs() < 1e-12,
            "i={i}: simd={:.15e}, scalar={:.15e}",
            g_simd[i],
            g_scalar[i]
        );
    }
}

/// SIMD influence vector matches scalar at n=64.
#[test]
fn test_simd_compute_influence_matches_scalar_n64() {
    let n = 64;
    let q = random_qubo(n, 666666);
    let state = random_state(n, 777777);
    let g_scalar = compute_influence(&state, &q, n);
    let g_simd = compute_influence_simd(&state, &q, n);
    for i in 0..n {
        assert!(
            (g_simd[i] - g_scalar[i]).abs() < 1e-12,
            "i={i}: simd={:.15e}, scalar={:.15e}",
            g_simd[i],
            g_scalar[i]
        );
    }
}

// ─── update_influence correctness ────────────────────────────────────────────

/// After flipping bit k, update_influence should give the same g as recomputing.
#[test]
fn test_update_influence_consistent_with_recompute() {
    for n in [4, 8, 16, 32] {
        let q = random_qubo(n, 12345 + n as u64);
        let state = random_state(n, 67890 + n as u64);
        let g_init = compute_influence(&state, &q, n);

        for k in 0..n {
            let mut g_updated = g_init.clone();
            let new_val = !state[k];
            update_influence(&mut g_updated, &q, n, k, new_val);

            let mut new_state = state.clone();
            new_state[k] = new_val;
            let g_recomputed = compute_influence(&new_state, &q, n);

            for i in 0..n {
                assert!(
                    (g_updated[i] - g_recomputed[i]).abs() < 1e-12,
                    "n={n}, k={k}, i={i}: updated={:.15e}, recomputed={:.15e}",
                    g_updated[i],
                    g_recomputed[i]
                );
            }
        }
    }
}

/// SIMD update_influence matches scalar at n=32.
#[test]
fn test_simd_update_influence_matches_scalar_n32() {
    let n = 32;
    let q = random_qubo(n, 13579);
    let state = random_state(n, 24680);
    let g_init = compute_influence(&state, &q, n);

    for k in 0..n {
        let mut g_scalar = g_init.clone();
        let mut g_simd = g_init.clone();
        let new_val = !state[k];
        update_influence(&mut g_scalar, &q, n, k, new_val);
        update_influence_simd(&mut g_simd, &q, n, k, new_val);
        for i in 0..n {
            assert!(
                (g_simd[i] - g_scalar[i]).abs() < 1e-12,
                "n=32, k={k}, i={i}: simd={:.15e}, scalar={:.15e}",
                g_simd[i],
                g_scalar[i]
            );
        }
    }
}

/// SIMD update_influence matches scalar at n=64.
#[test]
fn test_simd_update_influence_matches_scalar_n64() {
    let n = 64;
    let q = random_qubo(n, 97531);
    let state = random_state(n, 86420);
    let g_init = compute_influence(&state, &q, n);

    for k in 0..n {
        let mut g_scalar = g_init.clone();
        let mut g_simd = g_init.clone();
        let new_val = !state[k];
        update_influence(&mut g_scalar, &q, n, k, new_val);
        update_influence_simd(&mut g_simd, &q, n, k, new_val);
        for i in 0..n {
            assert!(
                (g_simd[i] - g_scalar[i]).abs() < 1e-12,
                "n=64, k={k}, i={i}: simd={:.15e}, scalar={:.15e}",
                g_simd[i],
                g_scalar[i]
            );
        }
    }
}

// ─── build_dense_q ───────────────────────────────────────────────────────────

#[test]
fn test_build_dense_q_linear_terms() {
    let mut edges = HashMap::new();
    edges.insert((0, 0), -3.0f64);
    edges.insert((1, 1), -5.0f64);
    edges.insert((2, 2), 2.0f64);
    let q = build_dense_q(3, &edges);
    assert!((q[0 * 3 + 0] - (-3.0)).abs() < 1e-15);
    assert!((q[1 * 3 + 1] - (-5.0)).abs() < 1e-15);
    assert!((q[2 * 3 + 2] - 2.0).abs() < 1e-15);
    // Off-diagonal should be zero
    assert!((q[0 * 3 + 1]).abs() < 1e-15);
    assert!((q[1 * 3 + 2]).abs() < 1e-15);
}

#[test]
fn test_build_dense_q_quadratic_terms() {
    let mut edges = HashMap::new();
    edges.insert((0, 1), 4.0f64);
    edges.insert((1, 2), -2.0f64);
    let q = build_dense_q(3, &edges);
    // build_dense_q stores at [i,j] only (not symmetric by default)
    assert!((q[0 * 3 + 1] - 4.0).abs() < 1e-15);
    assert!((q[1 * 3 + 0] - 0.0).abs() < 1e-15); // not mirrored
    assert!((q[1 * 3 + 2] - (-2.0)).abs() < 1e-15);
    assert!((q[2 * 3 + 1] - 0.0).abs() < 1e-15); // not mirrored
}

// ─── Integration: sampler-compatible incremental update loop ─────────────────

/// Simulate 100 steps of a tabu-style greedy descent using SIMD functions,
/// and verify the tracked energy equals energy_full at each step.
#[test]
fn test_sampler_incremental_loop_n32() {
    let n = 32;
    let q = random_qubo(n, 424242);
    let state = random_state(n, 737373);

    // Initialize
    let mut s = state.clone();
    let mut g = compute_influence_simd(&s, &q, n);
    let mut tracked_energy = energy_full_simd(&s, &q, n);

    for step in 0..100 {
        let k = step % n;
        let delta_e = (1.0 - 2.0 * if s[k] { 1.0 } else { 0.0 }) * g[k];

        let new_val = !s[k];
        update_influence_simd(&mut g, &q, n, k, new_val);
        s[k] = new_val;
        tracked_energy += delta_e;

        // Spot-check against recomputed energy every 10 steps
        if step % 10 == 9 {
            let recomputed = energy_full_simd(&s, &q, n);
            assert!(
                (tracked_energy - recomputed).abs() < 1e-10,
                "step={step}: tracked={tracked_energy:.12e}, recomputed={recomputed:.12e}"
            );
        }
    }
}

/// Same incremental loop at n=64.
#[test]
fn test_sampler_incremental_loop_n64() {
    let n = 64;
    let q = random_qubo(n, 858585);
    let state = random_state(n, 696969);

    let mut s = state.clone();
    let mut g = compute_influence_simd(&s, &q, n);
    let mut tracked_energy = energy_full_simd(&s, &q, n);

    for step in 0..100 {
        let k = step % n;
        let delta_e = (1.0 - 2.0 * if s[k] { 1.0 } else { 0.0 }) * g[k];

        let new_val = !s[k];
        update_influence_simd(&mut g, &q, n, k, new_val);
        s[k] = new_val;
        tracked_energy += delta_e;

        if step % 10 == 9 {
            let recomputed = energy_full_simd(&s, &q, n);
            assert!(
                (tracked_energy - recomputed).abs() < 1e-9,
                "step={step}: tracked={tracked_energy:.12e}, recomputed={recomputed:.12e}"
            );
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// HOBO (Higher-Order Binary Optimization) energy library correctness tests
// ═════════════════════════════════════════════════════════════════════════════

/// LCG helper for building deterministic pseudo-random HOBO tensors.
fn hobo_lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (u32::MAX as f64) * 4.0 - 2.0
}

/// Build a deterministic n×n×n HOBO tensor (3-body) as `ArrayD`.
fn make_hobo_3body(n: usize, seed: u64) -> ArrayD<f64> {
    let mut s = seed;
    let mut data = vec![0.0f64; n * n * n];
    for v in &mut data {
        *v = hobo_lcg(&mut s);
    }
    ArrayD::from_shape_vec(IxDyn(&[n, n, n]), data).expect("shape")
}

/// Build a deterministic n×n×n×n HOBO tensor (4-body) as `ArrayD`.
fn make_hobo_4body(n: usize, seed: u64) -> ArrayD<f64> {
    let mut s = seed;
    let mut data = vec![0.0f64; n * n * n * n];
    for v in &mut data {
        *v = hobo_lcg(&mut s);
    }
    ArrayD::from_shape_vec(IxDyn(&[n, n, n, n]), data).expect("shape")
}

/// Naïve HOBO energy via `indexed_iter` — the oracle for comparison.
fn naive_hobo_energy(state: &[bool], tensor: &ArrayD<f64>) -> f64 {
    let mut energy = 0.0f64;
    for (indices, &coeff) in tensor.indexed_iter() {
        if coeff == 0.0 {
            continue;
        }
        if indices.slice().iter().all(|&i| state[i]) {
            energy += coeff;
        }
    }
    energy
}

/// Flip bit k and return new state vector.
fn flip_k(state: &[bool], k: usize) -> Vec<bool> {
    let mut s = state.to_vec();
    s[k] = !s[k];
    s
}

// ─── 8 HOBO correctness tests ────────────────────────────────────────────────

/// 3-body energy (hobo_energy_full_3body) matches naïve indexed_iter over all
/// 2⁶ = 64 states for a random 6³ tensor.
#[test]
fn test_hobo_energy_3body_vs_naive_n6() {
    let n = 6;
    let tensor = make_hobo_3body(n, 0xABCD_EF12_3456_789Au64);
    let view3 = tensor
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix3>()
        .expect("3d view");

    for bits in 0u64..(1 << n) {
        let state: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
        let expected = naive_hobo_energy(&state, &tensor);
        let got = hobo_energy_full_3body(&state, view3);
        assert!(
            (got - expected).abs() < 1e-9,
            "bits={bits:#010b}: got={got}, expected={expected}"
        );
    }
}

/// 4-body energy (hobo_energy_full_4body) matches naïve indexed_iter over all
/// 2⁵ = 32 states for a random 5⁴ tensor.
#[test]
fn test_hobo_energy_4body_vs_naive_n5() {
    let n = 5;
    let tensor = make_hobo_4body(n, 0x1234_5678_9ABC_DEF0u64);
    let view4 = tensor
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix4>()
        .expect("4d view");

    for bits in 0u32..(1 << n) {
        let state: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
        let expected = naive_hobo_energy(&state, &tensor);
        let got = hobo_energy_full_4body(&state, view4);
        assert!(
            (got - expected).abs() < 1e-9,
            "bits={bits:#07b}: got={got}, expected={expected}"
        );
    }
}

/// A 5D all-zeros tensor returns 0.0 regardless of state.
#[test]
fn test_hobo_energy_5body_zeros_returns_zero() {
    let n = 4;
    let tensor = ArrayD::zeros(IxDyn(&[n, n, n, n, n]));
    for bits in 0u16..(1 << n) {
        let state: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
        let e = hobo_energy_full(&state, &tensor);
        assert!(e.abs() < 1e-15, "bits={bits:#06b}: expected 0 got {e}");
    }
}

/// For a 3-body problem, hobo_energy_delta == full(flip(x,k)) - full(x) for all
/// states and all k, over n=6.
#[test]
fn test_hobo_delta_matches_full_diff_3body_n6() {
    let n = 6;
    let tensor = make_hobo_3body(n, 0xDEAD_BEEF_CAFE_BABEu64);
    let view3 = tensor
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix3>()
        .expect("3d view");

    for bits in 0u64..(1 << n) {
        let state: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
        let e0 = hobo_energy_full_3body(&state, view3);

        for k in 0..n {
            let flipped = flip_k(&state, k);
            let e1 = hobo_energy_full_3body(&flipped, view3);
            let expected_delta = e1 - e0;

            let got_generic = hobo_energy_delta(&state, &tensor, k);
            let got_3body = hobo_energy_delta_3body(&state, view3, k);

            assert!(
                (got_generic - expected_delta).abs() < 1e-9,
                "bits={bits:#010b}, k={k}: generic delta={got_generic}, expected={expected_delta}"
            );
            assert!(
                (got_3body - expected_delta).abs() < 1e-9,
                "bits={bits:#010b}, k={k}: 3body delta={got_3body}, expected={expected_delta}"
            );
        }
    }
}

/// For a 4-body problem, hobo_energy_delta == full(flip(x,k)) - full(x) for all
/// states and all k, over n=5.
#[test]
fn test_hobo_delta_matches_full_diff_4body_n5() {
    let n = 5;
    let tensor = make_hobo_4body(n, 0xFEED_FACE_DEAD_BEEFu64);
    let view4 = tensor
        .view()
        .into_dimensionality::<scirs2_core::ndarray::Ix4>()
        .expect("4d view");

    for bits in 0u32..(1 << n) {
        let state: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
        let e0 = hobo_energy_full_4body(&state, view4);

        for k in 0..n {
            let flipped = flip_k(&state, k);
            let e1 = hobo_energy_full_4body(&flipped, view4);
            let expected_delta = e1 - e0;

            let got_generic = hobo_energy_delta(&state, &tensor, k);
            let got_4body = hobo_energy_delta_4body(&state, view4, k);

            assert!(
                (got_generic - expected_delta).abs() < 1e-9,
                "bits={bits:#07b}, k={k}: generic delta={got_generic}, expected={expected_delta}"
            );
            assert!(
                (got_4body - expected_delta).abs() < 1e-9,
                "bits={bits:#07b}, k={k}: 4body delta={got_4body}, expected={expected_delta}"
            );
        }
    }
}

/// For all states of a 3-body n=6 problem, (1-2x[k]) * g[k] == hobo_energy_delta.
#[test]
fn test_hobo_compute_influence_matches_delta_n6() {
    let n = 6;
    let tensor = make_hobo_3body(n, 0x0BAD_C0DE_1234_5678u64);

    for bits in 0u64..(1 << n) {
        let state: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
        let g = hobo_compute_influence(&state, &tensor);

        for k in 0..n {
            let from_g = (1.0 - 2.0 * if state[k] { 1.0 } else { 0.0 }) * g[k];
            let direct = hobo_energy_delta(&state, &tensor, k);
            assert!(
                (from_g - direct).abs() < 1e-9,
                "bits={bits:#010b}, k={k}: from_g={from_g}, direct={direct}"
            );
        }
    }
}

/// After flipping bit k, hobo_recompute_influence gives the same influence
/// vector as computing from scratch on the flipped state, over n=6 3-body.
#[test]
fn test_hobo_recompute_influence_matches_n6() {
    let n = 6;
    let tensor = make_hobo_3body(n, 0x1111_2222_3333_4444u64);

    let states: &[Vec<bool>] = &[
        vec![true, false, true, false, true, false],
        vec![false, false, false, false, false, false],
        vec![true, true, true, true, true, true],
        vec![true, false, false, true, false, true],
    ];

    for state in states {
        let g_initial = hobo_compute_influence(state, &tensor);

        for k in 0..n {
            let mut flipped = state.clone();
            flipped[k] = !flipped[k];

            let mut g_recomputed = g_initial.clone();
            hobo_recompute_influence(&mut g_recomputed, &flipped, &tensor);

            let g_reference = hobo_compute_influence(&flipped, &tensor);

            for q in 0..n {
                assert!(
                    (g_recomputed[q] - g_reference[q]).abs() < 1e-9,
                    "state={state:?}, k={k}, q={q}: recomputed={} reference={}",
                    g_recomputed[q],
                    g_reference[q]
                );
            }
        }
    }
}

/// The dispatch function routes a 2D tensor through the QUBO energy_full_simd path,
/// matching the direct QUBO `energy_full` for all 2⁴ = 16 states over n=4.
#[test]
fn test_hobo_dispatch_2body_matches_energy_full_n4() {
    let n = 4;
    let seed = 0x9999_AAAA_BBBB_CCCCu64;
    let q_flat = random_qubo(n, seed);
    let q_array =
        scirs2_core::ndarray::Array2::from_shape_vec((n, n), q_flat.clone()).expect("shape");
    let tensor_dyn = q_array.into_dyn();

    for bits in 0u16..(1 << n) {
        let state: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
        let expected = energy_full_simd(&state, &q_flat, n);
        let got = hobo_energy_full_dispatch(&state, &tensor_dyn);
        assert!(
            (got - expected).abs() < 1e-12,
            "bits={bits:#06b}: dispatch={got}, direct={expected}"
        );
    }
}
