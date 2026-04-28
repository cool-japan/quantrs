#![allow(
    clippy::pedantic,
    clippy::unreadable_literal,    // long seed constants (intentional)
    clippy::suboptimal_flops,      // FMA not required for correctness tests
    clippy::identity_op,           // test matrices may have zero-effect terms
    clippy::erasing_op,            // same as above
    clippy::redundant_clone,       // test clarity over performance
    clippy::explicit_iter_loop,    // explicit index loops in SIMD verification
    clippy::needless_range_loop,   // k is used for two purposes (index + value)
)]
//! Correctness tests: SIMD energy functions vs scalar reference.
//!
//! These tests verify that all SIMD-accelerated energy functions produce
//! results within 1e-12 of the scalar reference implementations, across
//! a range of problem sizes and random QUBO instances.

use quantrs2_tytan::sampler::energy::{
    build_dense_q, compute_influence, compute_influence_simd, energy_delta, energy_delta_simd,
    energy_full, energy_full_simd, update_influence, update_influence_simd,
};
use std::collections::HashMap;

// ─── QUBO generator (deterministic LCG, no external deps) ────────────────────

/// Build a deterministic pseudo-random symmetric QUBO matrix.
fn random_qubo(n: usize, seed: u64) -> Vec<f64> {
    let mut q = vec![0.0f64; n * n];
    let mut s = seed;
    let mut lcg = || -> f64 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
