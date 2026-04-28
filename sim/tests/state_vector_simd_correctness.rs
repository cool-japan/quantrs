//! Correctness tests: SIMD vs scalar gate application on random states.
//!
//! For each gate and each combination of (n_qubits, target) we apply:
//! 1. The SIMD-accelerated function from `state_vector_simd`.
//! 2. The scalar reference via `apply_gate_2x2_scalar`.
//!
//! We then assert that the maximum amplitude-wise L2 distance is below 1e-12.

use quantrs2_sim::state_vector_simd::{
    apply_gate_2x2_scalar, apply_gate_2x2_simd, apply_h_simd, apply_rx_simd, apply_ry_simd,
    apply_rz_simd, apply_s_simd, apply_t_simd, apply_x_simd, apply_y_simd, apply_z_simd,
};
use scirs2_core::Complex64;

// ============================================================================
// Helpers
// ============================================================================

/// Simple LCG RNG — produces reproducible "random" states without `rand`.
fn lcg_random_state(n_qubits: usize, seed: u64) -> Vec<Complex64> {
    let mut rng = seed;
    let mut state: Vec<Complex64> = (0..(1usize << n_qubits))
        .map(|_| {
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let re = (rng as f64) / (u64::MAX as f64) * 2.0 - 1.0;
            rng = rng
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let im = (rng as f64) / (u64::MAX as f64) * 2.0 - 1.0;
            Complex64::new(re, im)
        })
        .collect();

    let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    state.iter_mut().for_each(|c| *c /= norm);
    state
}

fn max_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0_f64, f64::max)
}

// ============================================================================
// H gate
// ============================================================================

#[test]
fn test_h_gate_correctness_all_targets() {
    use std::f64::consts::FRAC_1_SQRT_2;
    let h_mat = [
        [
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
        ],
        [
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(-FRAC_1_SQRT_2, 0.0),
        ],
    ];

    for &n in &[3usize, 5, 7] {
        let base = lcg_random_state(n, 1001 + n as u64);
        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_h_simd(&mut simd_state, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, h_mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "H gate SIMD vs scalar mismatch: n={}, target={}, diff={}",
                n,
                target,
                diff
            );
        }
    }
}

// ============================================================================
// X gate
// ============================================================================

#[test]
fn test_x_gate_correctness_all_targets() {
    let x_mat = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];

    for &n in &[3usize, 5, 7] {
        let base = lcg_random_state(n, 2001 + n as u64);
        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_x_simd(&mut simd_state, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, x_mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "X gate SIMD vs scalar mismatch: n={}, target={}, diff={}",
                n,
                target,
                diff
            );
        }
    }
}

// ============================================================================
// Y gate
// ============================================================================

#[test]
fn test_y_gate_correctness_all_targets() {
    let y_mat = [
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
    ];

    for &n in &[3usize, 5, 7] {
        let base = lcg_random_state(n, 3001 + n as u64);
        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_y_simd(&mut simd_state, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, y_mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "Y gate SIMD vs scalar mismatch: n={}, target={}, diff={}",
                n,
                target,
                diff
            );
        }
    }
}

// ============================================================================
// Z gate
// ============================================================================

#[test]
fn test_z_gate_correctness_all_targets() {
    let z_mat = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
    ];

    for &n in &[3usize, 5, 7] {
        let base = lcg_random_state(n, 4001 + n as u64);
        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_z_simd(&mut simd_state, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, z_mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "Z gate SIMD vs scalar mismatch: n={}, target={}, diff={}",
                n,
                target,
                diff
            );
        }
    }
}

// ============================================================================
// S gate
// ============================================================================

#[test]
fn test_s_gate_correctness_all_targets() {
    let s_mat = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
    ];

    for &n in &[3usize, 5, 7] {
        let base = lcg_random_state(n, 5001 + n as u64);
        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_s_simd(&mut simd_state, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, s_mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "S gate SIMD vs scalar mismatch: n={}, target={}, diff={}",
                n,
                target,
                diff
            );
        }
    }
}

// ============================================================================
// T gate
// ============================================================================

#[test]
fn test_t_gate_correctness_all_targets() {
    use std::f64::consts::FRAC_PI_4;
    let t_mat = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(FRAC_PI_4.cos(), FRAC_PI_4.sin()),
        ],
    ];

    for &n in &[3usize, 5, 7] {
        let base = lcg_random_state(n, 6001 + n as u64);
        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_t_simd(&mut simd_state, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, t_mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "T gate SIMD vs scalar mismatch: n={}, target={}, diff={}",
                n,
                target,
                diff
            );
        }
    }
}

// ============================================================================
// RX gate
// ============================================================================

#[test]
fn test_rx_gate_correctness_all_targets() {
    let theta = std::f64::consts::PI / 3.0;
    let h = theta / 2.0;
    let rx_mat = [
        [Complex64::new(h.cos(), 0.0), Complex64::new(0.0, -h.sin())],
        [Complex64::new(0.0, -h.sin()), Complex64::new(h.cos(), 0.0)],
    ];

    for &n in &[3usize, 5, 7] {
        let base = lcg_random_state(n, 7001 + n as u64);
        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_rx_simd(&mut simd_state, theta, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, rx_mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "RX gate SIMD vs scalar mismatch: n={}, target={}, diff={}",
                n,
                target,
                diff
            );
        }
    }
}

// ============================================================================
// RY gate
// ============================================================================

#[test]
fn test_ry_gate_correctness_all_targets() {
    let theta = std::f64::consts::PI / 5.0;
    let h = theta / 2.0;
    let ry_mat = [
        [Complex64::new(h.cos(), 0.0), Complex64::new(-h.sin(), 0.0)],
        [Complex64::new(h.sin(), 0.0), Complex64::new(h.cos(), 0.0)],
    ];

    for &n in &[3usize, 5, 7] {
        let base = lcg_random_state(n, 8001 + n as u64);
        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_ry_simd(&mut simd_state, theta, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, ry_mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "RY gate SIMD vs scalar mismatch: n={}, target={}, diff={}",
                n,
                target,
                diff
            );
        }
    }
}

// ============================================================================
// RZ gate
// ============================================================================

#[test]
fn test_rz_gate_correctness_all_targets() {
    let theta = std::f64::consts::PI / 7.0;
    let h = theta / 2.0;
    let rz_mat = [
        [Complex64::new(h.cos(), -h.sin()), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(h.cos(), h.sin())],
    ];

    for &n in &[3usize, 5, 7] {
        let base = lcg_random_state(n, 9001 + n as u64);
        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_rz_simd(&mut simd_state, theta, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, rz_mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "RZ gate SIMD vs scalar mismatch: n={}, target={}, diff={}",
                n,
                target,
                diff
            );
        }
    }
}

// ============================================================================
// Generic apply_gate_2x2_simd
// ============================================================================

#[test]
fn test_generic_gate_2x2_simd_correctness() {
    // Use a random unitary-ish matrix (not necessarily unitary, but enough to
    // test the arithmetic path).
    let mat = [
        [
            Complex64::new(0.6, 0.2),
            Complex64::new(-0.3, 0.5),
        ],
        [
            Complex64::new(0.7, -0.1),
            Complex64::new(0.4, 0.3),
        ],
    ];

    for &n in &[3usize, 5, 7] {
        let base = lcg_random_state(n, 10001 + n as u64);
        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_gate_2x2_simd(&mut simd_state, mat, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "Generic 2x2 gate SIMD vs scalar mismatch: n={}, target={}, diff={}",
                n,
                target,
                diff
            );
        }
    }
}

// ============================================================================
// Normalization preservation smoke test
// ============================================================================

#[test]
fn test_unitary_gates_preserve_norm() {
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    let n = 5usize;
    let mut state = lcg_random_state(n, 77777);

    // Chain several unitary gates
    apply_h_simd(&mut state, 0, n);
    apply_x_simd(&mut state, 1, n);
    apply_y_simd(&mut state, 2, n);
    apply_z_simd(&mut state, 3, n);
    apply_s_simd(&mut state, 4, n);
    apply_t_simd(&mut state, 0, n);
    apply_rx_simd(&mut state, PI / 4.0, 1, n);
    apply_ry_simd(&mut state, PI / 3.0, 2, n);
    apply_rz_simd(&mut state, PI / 6.0, 3, n);

    let norm_sq: f64 = state.iter().map(|c| c.norm_sqr()).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-10,
        "Norm not preserved after gate chain: norm_sq={}",
        norm_sq
    );
}
