#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::field_reassign_with_default
)]
//! Sampler scalability smoke tests for QuantRS2 v0.2.0
//!
//! Verifies that the SA, Tabu, and SB samplers can handle moderately large
//! QUBO problems (50–200 variables) within reasonable time bounds and return
//! non-empty, sorted sample vectors.
//!
//! Tests marked `#[ignore]` may take > 10 s on minimal CI runners — run with:
//!   cargo nextest run -p quantrs2-tytan -- --ignored

use std::collections::HashMap;

use quantrs2_tytan::sampler::{SASampler, SBSampler, SBVariant, Sampler, TabuSampler};

// ---------------------------------------------------------------------------
// Local helper: reproducible random QUBO generator
//
// Produces an upper-triangular n×n matrix with entries in [-2, 2] using a
// simple LCG (no external rand dependency beyond what's already in scope).
// ---------------------------------------------------------------------------
fn generate_random_qubo(n: usize, seed: u64) -> scirs2_core::ndarray::Array2<f64> {
    // LCG parameters (Numerical Recipes)
    const A: u64 = 1_664_525;
    const C: u64 = 1_013_904_223;
    let mut lcg_state = seed;

    let mut lcg_next = || -> f64 {
        lcg_state = lcg_state.wrapping_mul(A).wrapping_add(C);
        (lcg_state as f64 / u64::MAX as f64) * 4.0 - 2.0
    };

    let mut q = scirs2_core::ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        q[[i, i]] = lcg_next(); // diagonal (linear term)
        for j in (i + 1)..n {
            q[[i, j]] = lcg_next(); // upper-triangular (quadratic term)
        }
    }
    q
}

/// Build a variable name map for `n` variables: x0, x1, ..., x(n-1).
fn build_var_map(n: usize) -> HashMap<String, usize> {
    (0..n).map(|i| (format!("x{i}"), i)).collect()
}

/// Assert that a result vector is non-empty and sorted by energy (ascending).
fn assert_valid_results(
    results: &[quantrs2_tytan::sampler::SampleResult],
    label: &str,
    n_vars: usize,
) {
    assert!(
        !results.is_empty(),
        "{label}: sampler returned 0 results — expected at least one"
    );

    // Each result must contain all `n_vars` variables
    for (idx, res) in results.iter().enumerate() {
        assert_eq!(
            res.assignments.len(),
            n_vars,
            "{label}: result[{idx}] has {} assignments, expected {n_vars}",
            res.assignments.len()
        );
        for k in 0..n_vars {
            let key = format!("x{k}");
            assert!(
                res.assignments.contains_key(&key),
                "{label}: result[{idx}] missing variable '{key}'"
            );
        }
        assert!(
            res.occurrences > 0,
            "{label}: result[{idx}] has occurrences=0 — must be ≥ 1"
        );
    }

    // Results must be sorted by energy ascending (best first)
    for window in results.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        assert!(
            a.energy <= b.energy + 1e-9,
            "{label}: results not sorted — result energy {:.6} > next energy {:.6}",
            a.energy,
            b.energy
        );
    }
}

// ---------------------------------------------------------------------------
// Test 1: SASampler on a random 50-variable QUBO (5 shots)
//
// 50 variables → 50×50 QUBO matrix.  The SA sampler should handle this in
// well under 1 s and return 5 non-empty, energy-sorted results.
// ---------------------------------------------------------------------------
#[test]
fn test_sa_50_var_random_qubo() {
    const N: usize = 50;
    const SHOTS: usize = 5;

    let q = generate_random_qubo(N, 12345);
    let var_map = build_var_map(N);

    let sampler = SASampler::new(Some(99));
    let results = sampler
        .run_qubo(&(q, var_map), SHOTS)
        .expect("SASampler failed on 50-var QUBO");

    assert_valid_results(&results, "SA-50var", N);

    // Heuristic: for 5 shots on 50 variables the best energy must be finite
    let best_energy = results[0].energy;
    assert!(
        best_energy.is_finite(),
        "SA-50var: best energy is not finite: {best_energy}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: TabuSampler on a random 100-variable QUBO (3 shots)
//
// 100 variables → 100×100 QUBO.  Tabu search should solve each shot in < 2 s.
// ---------------------------------------------------------------------------
#[test]
fn test_tabu_100_var_random_qubo() {
    const N: usize = 100;
    const SHOTS: usize = 3;

    let q = generate_random_qubo(N, 54321);
    let var_map = build_var_map(N);

    let sampler = TabuSampler::new().with_seed(42);
    let results = sampler
        .run_qubo(&(q, var_map), SHOTS)
        .expect("TabuSampler failed on 100-var QUBO");

    assert_valid_results(&results, "Tabu-100var", N);

    let best_energy = results[0].energy;
    assert!(
        best_energy.is_finite(),
        "Tabu-100var: best energy is not finite: {best_energy}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: SBSampler (Ballistic) on a random 200-variable QUBO (2 shots)
//
// 200 variables → 200×200 QUBO.  Simulated Bifurcation runs O(n²) per step
// and may be slow at n=200.  Marked `#[ignore]` for minimal CI runners;
// run with:  cargo nextest run -p quantrs2-tytan -- --ignored
// ---------------------------------------------------------------------------
#[test]
#[ignore] // cargo nextest run -- --ignored   (may take > 10 s at n=200)
fn test_sb_200_var_random_qubo() {
    const N: usize = 200;
    const SHOTS: usize = 2;

    let q = generate_random_qubo(N, 99999);
    let var_map = build_var_map(N);

    let sampler = SBSampler::new()
        .with_seed(7)
        .with_variant(SBVariant::Ballistic);
    let results = sampler
        .run_qubo(&(q, var_map), SHOTS)
        .expect("SBSampler (Ballistic) failed on 200-var QUBO");

    assert_valid_results(&results, "SB-Ballistic-200var", N);

    let best_energy = results[0].energy;
    assert!(
        best_energy.is_finite(),
        "SB-Ballistic-200var: best energy is not finite: {best_energy}"
    );
}

// ---------------------------------------------------------------------------
// Test 4: SBSampler (Discrete) on a random 50-variable QUBO (5 shots)
//
// The Discrete variant is generally faster than Ballistic; this test
// exercises the alternative code path without the `#[ignore]` overhead.
// ---------------------------------------------------------------------------
#[test]
fn test_sb_discrete_50_var_random_qubo() {
    const N: usize = 50;
    const SHOTS: usize = 5;

    let q = generate_random_qubo(N, 77777);
    let var_map = build_var_map(N);

    let sampler = SBSampler::new()
        .with_seed(13)
        .with_variant(SBVariant::Discrete);
    let results = sampler
        .run_qubo(&(q, var_map), SHOTS)
        .expect("SBSampler (Discrete) failed on 50-var QUBO");

    assert_valid_results(&results, "SB-Discrete-50var", N);

    let best_energy = results[0].energy;
    assert!(
        best_energy.is_finite(),
        "SB-Discrete-50var: best energy is not finite: {best_energy}"
    );
}
