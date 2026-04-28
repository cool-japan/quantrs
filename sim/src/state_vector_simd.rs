//! SIMD-accelerated single-qubit gate kernels for state vector simulation.
//!
//! This module provides a high-level public API for applying individual quantum gates
//! directly to `Vec<Complex64>` state vectors. It owns the gather/scatter logic
//! internally so callers do not need to split amplitudes into separate `in_amps0` /
//! `in_amps1` buffers themselves.
//!
//! Internally the implementations delegate to the existing
//! `crate::optimized_simd` SIMD primitives (backed by
//! `scirs2_core::simd_ops::SimdUnifiedOps`) for all vector arithmetic, with
//! a scalar fallback for small state vectors (< 256 amplitudes, i.e. < 8 qubits).
//!
//! ## Usage example
//!
//! ```no_run
//! use quantrs2_sim::state_vector_simd::{apply_h_simd, apply_x_simd};
//! use scirs2_core::Complex64;
//!
//! let n_qubits = 4usize;
//! let mut state = vec![Complex64::new(0.0, 0.0); 1 << n_qubits];
//! state[0] = Complex64::new(1.0, 0.0); // |0000⟩
//!
//! apply_h_simd(&mut state, 0, n_qubits);   // qubit 0 → |+⟩
//! apply_x_simd(&mut state, 1, n_qubits);   // qubit 1 → |1⟩ (CNOT-like)
//! ```
//!
//! These SIMD kernels are provided as a standalone module and can be called
//! directly. Integration into the main `StateVectorSimulator` dispatch already
//! exists in `statevector.rs` via `apply_single_qubit_gate_simd`; this module
//! adds the named, standalone API surface (see also TODO.md).

use scirs2_core::Complex64;

use crate::optimized_simd::{
    apply_h_gate_simd, apply_rx_gate_simd, apply_ry_gate_simd, apply_rz_gate_simd,
    apply_s_gate_simd, apply_single_qubit_gate_optimized, apply_t_gate_simd, apply_x_gate_simd,
    apply_y_gate_simd, apply_z_gate_simd,
};

// ============================================================================
// Internal gather / scatter helpers
// ============================================================================

/// Gather amplitudes for a given target qubit into `out0` (bit=0) and `out1` (bit=1)
/// buffers.  Returns the number of pairs gathered (= `n_states / 2`).
fn gather_pairs(
    state: &[Complex64],
    target: usize,
    n_qubits: usize,
    out0: &mut Vec<Complex64>,
    out1: &mut Vec<Complex64>,
) -> usize {
    let n_states = 1usize << n_qubits;
    let stride = 1usize << target;
    let total_pairs = n_states / 2;

    out0.clear();
    out1.clear();
    out0.reserve(total_pairs);
    out1.reserve(total_pairs);

    let mut i = 0usize;
    while i < n_states {
        for j in i..(i + stride) {
            out0.push(state[j]);
            out1.push(state[j + stride]);
        }
        i += 2 * stride;
    }

    total_pairs
}

/// Scatter computed amplitude pairs back into `state`.
fn scatter_pairs(
    state: &mut [Complex64],
    target: usize,
    n_qubits: usize,
    src0: &[Complex64],
    src1: &[Complex64],
) {
    let n_states = 1usize << n_qubits;
    let stride = 1usize << target;

    let mut pair_idx = 0usize;
    let mut i = 0usize;
    while i < n_states {
        for j in i..(i + stride) {
            state[j] = src0[pair_idx];
            state[j + stride] = src1[pair_idx];
            pair_idx += 1;
        }
        i += 2 * stride;
    }
}

// ============================================================================
// Threshold
// ============================================================================

/// Minimum number of amplitudes for which the SIMD path is used.
/// Below this threshold the scalar fallback is cheaper.
const SIMD_THRESHOLD: usize = 256; // 8 qubits

// ============================================================================
// Scalar fallback for 2×2 unitary gate
// ============================================================================

/// Pure-scalar application of a 2×2 unitary `matrix` to qubit `target`.
///
/// Uses the stride-based pair traversal; always correct regardless of
/// `n_qubits` or `target`.
pub fn apply_gate_2x2_scalar(
    state: &mut [Complex64],
    matrix: [[Complex64; 2]; 2],
    target: usize,
    n_qubits: usize,
) {
    let stride = 1usize << target;
    let n_states = 1usize << n_qubits;
    let [[a, b], [c, d]] = matrix;

    let mut i = 0usize;
    while i < n_states {
        for j in i..(i + stride) {
            let zero = state[j];
            let one = state[j + stride];
            state[j] = a * zero + b * one;
            state[j + stride] = c * zero + d * one;
        }
        i += 2 * stride;
    }
}

// ============================================================================
// Public API — generic dispatcher
// ============================================================================

/// Apply a generic 2×2 unitary gate to qubit `target` using SIMD acceleration.
///
/// `matrix` is given as `[[a, b], [c, d]]` where the new amplitudes are:
///   - new_|0⟩ = a·old_|0⟩ + b·old_|1⟩
///   - new_|1⟩ = c·old_|0⟩ + d·old_|1⟩
///
/// Falls back to scalar arithmetic for `state.len() < 256`.
pub fn apply_gate_2x2_simd(
    state: &mut Vec<Complex64>,
    matrix: [[Complex64; 2]; 2],
    target: usize,
    n_qubits: usize,
) {
    if state.len() < SIMD_THRESHOLD {
        apply_gate_2x2_scalar(state, matrix, target, n_qubits);
        return;
    }

    let flat = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]];

    let mut amps0 = Vec::with_capacity(state.len() / 2);
    let mut amps1 = Vec::with_capacity(state.len() / 2);
    let n_pairs = gather_pairs(state, target, n_qubits, &mut amps0, &mut amps1);

    let mut out0 = vec![Complex64::new(0.0, 0.0); n_pairs];
    let mut out1 = vec![Complex64::new(0.0, 0.0); n_pairs];

    apply_single_qubit_gate_optimized(&flat, &amps0, &amps1, &mut out0, &mut out1);
    scatter_pairs(state, target, n_qubits, &out0, &out1);
}

// ============================================================================
// Named single-qubit gates
// ============================================================================

/// Apply the Hadamard gate H to qubit `target` using SIMD.
///
/// H = (1/√2) [[1, 1], [1, -1]]
pub fn apply_h_simd(state: &mut Vec<Complex64>, target: usize, n_qubits: usize) {
    if state.len() < SIMD_THRESHOLD {
        use std::f64::consts::FRAC_1_SQRT_2;
        let h = [
            [
                Complex64::new(FRAC_1_SQRT_2, 0.0),
                Complex64::new(FRAC_1_SQRT_2, 0.0),
            ],
            [
                Complex64::new(FRAC_1_SQRT_2, 0.0),
                Complex64::new(-FRAC_1_SQRT_2, 0.0),
            ],
        ];
        apply_gate_2x2_scalar(state, h, target, n_qubits);
        return;
    }

    let mut amps0 = Vec::with_capacity(state.len() / 2);
    let mut amps1 = Vec::with_capacity(state.len() / 2);
    let n_pairs = gather_pairs(state, target, n_qubits, &mut amps0, &mut amps1);

    let mut out0 = vec![Complex64::new(0.0, 0.0); n_pairs];
    let mut out1 = vec![Complex64::new(0.0, 0.0); n_pairs];

    apply_h_gate_simd(&amps0, &amps1, &mut out0, &mut out1);
    scatter_pairs(state, target, n_qubits, &out0, &out1);
}

/// Apply the Pauli-X (NOT) gate to qubit `target` using SIMD.
///
/// X = [[0, 1], [1, 0]]
pub fn apply_x_simd(state: &mut Vec<Complex64>, target: usize, n_qubits: usize) {
    if state.len() < SIMD_THRESHOLD {
        apply_gate_2x2_scalar(
            state,
            [
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ],
            target,
            n_qubits,
        );
        return;
    }

    let mut amps0 = Vec::with_capacity(state.len() / 2);
    let mut amps1 = Vec::with_capacity(state.len() / 2);
    let n_pairs = gather_pairs(state, target, n_qubits, &mut amps0, &mut amps1);

    let mut out0 = vec![Complex64::new(0.0, 0.0); n_pairs];
    let mut out1 = vec![Complex64::new(0.0, 0.0); n_pairs];

    apply_x_gate_simd(&amps0, &amps1, &mut out0, &mut out1);
    scatter_pairs(state, target, n_qubits, &out0, &out1);
}

/// Apply the Pauli-Y gate to qubit `target` using SIMD.
///
/// Y = [[0, -i], [i, 0]]
pub fn apply_y_simd(state: &mut Vec<Complex64>, target: usize, n_qubits: usize) {
    if state.len() < SIMD_THRESHOLD {
        apply_gate_2x2_scalar(
            state,
            [
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
                [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
            ],
            target,
            n_qubits,
        );
        return;
    }

    let mut amps0 = Vec::with_capacity(state.len() / 2);
    let mut amps1 = Vec::with_capacity(state.len() / 2);
    let n_pairs = gather_pairs(state, target, n_qubits, &mut amps0, &mut amps1);

    let mut out0 = vec![Complex64::new(0.0, 0.0); n_pairs];
    let mut out1 = vec![Complex64::new(0.0, 0.0); n_pairs];

    apply_y_gate_simd(&amps0, &amps1, &mut out0, &mut out1);
    scatter_pairs(state, target, n_qubits, &out0, &out1);
}

/// Apply the Pauli-Z gate to qubit `target` using SIMD.
///
/// Z = [[1, 0], [0, -1]]
pub fn apply_z_simd(state: &mut Vec<Complex64>, target: usize, n_qubits: usize) {
    if state.len() < SIMD_THRESHOLD {
        apply_gate_2x2_scalar(
            state,
            [
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
            ],
            target,
            n_qubits,
        );
        return;
    }

    let mut amps0 = Vec::with_capacity(state.len() / 2);
    let mut amps1 = Vec::with_capacity(state.len() / 2);
    let n_pairs = gather_pairs(state, target, n_qubits, &mut amps0, &mut amps1);

    let mut out0 = vec![Complex64::new(0.0, 0.0); n_pairs];
    let mut out1 = vec![Complex64::new(0.0, 0.0); n_pairs];

    apply_z_gate_simd(&amps0, &amps1, &mut out0, &mut out1);
    scatter_pairs(state, target, n_qubits, &out0, &out1);
}

/// Apply the S (phase) gate to qubit `target` using SIMD.
///
/// S = [[1, 0], [0, i]]
pub fn apply_s_simd(state: &mut Vec<Complex64>, target: usize, n_qubits: usize) {
    if state.len() < SIMD_THRESHOLD {
        apply_gate_2x2_scalar(
            state,
            [
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
            ],
            target,
            n_qubits,
        );
        return;
    }

    let mut amps0 = Vec::with_capacity(state.len() / 2);
    let mut amps1 = Vec::with_capacity(state.len() / 2);
    let n_pairs = gather_pairs(state, target, n_qubits, &mut amps0, &mut amps1);

    let mut out0 = vec![Complex64::new(0.0, 0.0); n_pairs];
    let mut out1 = vec![Complex64::new(0.0, 0.0); n_pairs];

    apply_s_gate_simd(&amps0, &amps1, &mut out0, &mut out1);
    scatter_pairs(state, target, n_qubits, &out0, &out1);
}

/// Apply the T gate to qubit `target` using SIMD.
///
/// T = [[1, 0], [0, exp(iπ/4)]]
pub fn apply_t_simd(state: &mut Vec<Complex64>, target: usize, n_qubits: usize) {
    if state.len() < SIMD_THRESHOLD {
        use std::f64::consts::FRAC_PI_4;
        apply_gate_2x2_scalar(
            state,
            [
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [
                    Complex64::new(0.0, 0.0),
                    Complex64::new(FRAC_PI_4.cos(), FRAC_PI_4.sin()),
                ],
            ],
            target,
            n_qubits,
        );
        return;
    }

    let mut amps0 = Vec::with_capacity(state.len() / 2);
    let mut amps1 = Vec::with_capacity(state.len() / 2);
    let n_pairs = gather_pairs(state, target, n_qubits, &mut amps0, &mut amps1);

    let mut out0 = vec![Complex64::new(0.0, 0.0); n_pairs];
    let mut out1 = vec![Complex64::new(0.0, 0.0); n_pairs];

    apply_t_gate_simd(&amps0, &amps1, &mut out0, &mut out1);
    scatter_pairs(state, target, n_qubits, &out0, &out1);
}

/// Apply the RX(theta) rotation gate to qubit `target` using SIMD.
///
/// RX(θ) = [[cos(θ/2), −i·sin(θ/2)], [−i·sin(θ/2), cos(θ/2)]]
pub fn apply_rx_simd(state: &mut Vec<Complex64>, theta: f64, target: usize, n_qubits: usize) {
    if state.len() < SIMD_THRESHOLD {
        let h = theta / 2.0;
        apply_gate_2x2_scalar(
            state,
            [
                [Complex64::new(h.cos(), 0.0), Complex64::new(0.0, -h.sin())],
                [Complex64::new(0.0, -h.sin()), Complex64::new(h.cos(), 0.0)],
            ],
            target,
            n_qubits,
        );
        return;
    }

    let mut amps0 = Vec::with_capacity(state.len() / 2);
    let mut amps1 = Vec::with_capacity(state.len() / 2);
    let n_pairs = gather_pairs(state, target, n_qubits, &mut amps0, &mut amps1);

    let mut out0 = vec![Complex64::new(0.0, 0.0); n_pairs];
    let mut out1 = vec![Complex64::new(0.0, 0.0); n_pairs];

    apply_rx_gate_simd(theta, &amps0, &amps1, &mut out0, &mut out1);
    scatter_pairs(state, target, n_qubits, &out0, &out1);
}

/// Apply the RY(theta) rotation gate to qubit `target` using SIMD.
///
/// RY(θ) = [[cos(θ/2), −sin(θ/2)], [sin(θ/2), cos(θ/2)]]
pub fn apply_ry_simd(state: &mut Vec<Complex64>, theta: f64, target: usize, n_qubits: usize) {
    if state.len() < SIMD_THRESHOLD {
        let h = theta / 2.0;
        apply_gate_2x2_scalar(
            state,
            [
                [Complex64::new(h.cos(), 0.0), Complex64::new(-h.sin(), 0.0)],
                [Complex64::new(h.sin(), 0.0), Complex64::new(h.cos(), 0.0)],
            ],
            target,
            n_qubits,
        );
        return;
    }

    let mut amps0 = Vec::with_capacity(state.len() / 2);
    let mut amps1 = Vec::with_capacity(state.len() / 2);
    let n_pairs = gather_pairs(state, target, n_qubits, &mut amps0, &mut amps1);

    let mut out0 = vec![Complex64::new(0.0, 0.0); n_pairs];
    let mut out1 = vec![Complex64::new(0.0, 0.0); n_pairs];

    apply_ry_gate_simd(theta, &amps0, &amps1, &mut out0, &mut out1);
    scatter_pairs(state, target, n_qubits, &out0, &out1);
}

/// Apply the RZ(theta) rotation gate to qubit `target` using SIMD.
///
/// RZ(θ) = [[exp(−iθ/2), 0], [0, exp(iθ/2)]]
pub fn apply_rz_simd(state: &mut Vec<Complex64>, theta: f64, target: usize, n_qubits: usize) {
    if state.len() < SIMD_THRESHOLD {
        let h = theta / 2.0;
        apply_gate_2x2_scalar(
            state,
            [
                [Complex64::new(h.cos(), -h.sin()), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(h.cos(), h.sin())],
            ],
            target,
            n_qubits,
        );
        return;
    }

    let mut amps0 = Vec::with_capacity(state.len() / 2);
    let mut amps1 = Vec::with_capacity(state.len() / 2);
    let n_pairs = gather_pairs(state, target, n_qubits, &mut amps0, &mut amps1);

    let mut out0 = vec![Complex64::new(0.0, 0.0); n_pairs];
    let mut out1 = vec![Complex64::new(0.0, 0.0); n_pairs];

    apply_rz_gate_simd(theta, &amps0, &amps1, &mut out0, &mut out1);
    scatter_pairs(state, target, n_qubits, &out0, &out1);
}

// ============================================================================
// Runtime SIMD capability detection
// ============================================================================

/// Returns `true` when SIMD acceleration is available at runtime on this CPU.
///
/// On x86_64: requires AVX2.  On aarch64: NEON is always available.
/// On other architectures: returns `false`.
pub fn simd_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

// ============================================================================
// In-file unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    /// Build an n-qubit |0...0⟩ state.
    fn zero_state(n: usize) -> Vec<Complex64> {
        let mut s = vec![Complex64::new(0.0, 0.0); 1 << n];
        s[0] = Complex64::new(1.0, 0.0);
        s
    }

    /// Maximum L2 distance between two state vectors.
    fn max_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0_f64, f64::max)
    }

    // -----------------------------------------------------------------------
    // Scalar fallback tests (small state, n < 8)
    // -----------------------------------------------------------------------

    #[test]
    fn test_h_gate_zero_state() {
        // H|0⟩ = (|0⟩ + |1⟩)/√2
        let mut state = zero_state(1);
        apply_h_simd(&mut state, 0, 1);

        assert!(
            (state[0] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-12,
            "H|0> amplitude of |0> wrong: {:?}",
            state[0]
        );
        assert!(
            (state[1] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-12,
            "H|0> amplitude of |1> wrong: {:?}",
            state[1]
        );
    }

    #[test]
    fn test_x_gate() {
        // X|0⟩ = |1⟩
        let mut state = zero_state(2);
        apply_x_simd(&mut state, 0, 2);

        // Qubit 0 flipped → index 1 is |1⟩
        assert!(
            (state[0] - Complex64::new(0.0, 0.0)).norm() < 1e-12,
            "X|0>: state[0] should be 0"
        );
        assert!(
            (state[1] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "X|0>: state[1] should be 1"
        );
    }

    #[test]
    fn test_z_gate_on_plus_state() {
        // Prepare |+⟩ then apply Z → |−⟩
        // |+⟩ = (|0⟩ + |1⟩)/√2  after H on |0⟩
        let mut state = zero_state(1);
        apply_h_simd(&mut state, 0, 1);
        apply_z_simd(&mut state, 0, 1);

        // Z|+⟩ = |−⟩ = (|0⟩ - |1⟩)/√2
        assert!(
            (state[0] - Complex64::new(FRAC_1_SQRT_2, 0.0)).norm() < 1e-12,
            "Z|+>: state[0] wrong"
        );
        assert!(
            (state[1] - Complex64::new(-FRAC_1_SQRT_2, 0.0)).norm() < 1e-12,
            "Z|+>: state[1] wrong"
        );
    }

    #[test]
    fn test_rx_half_pi() {
        // RX(π/2)|0⟩ = cos(π/4)|0⟩ − i·sin(π/4)|1⟩
        let theta = PI / 2.0;
        let mut state = zero_state(1);
        apply_rx_simd(&mut state, theta, 0, 1);

        let expected0 = Complex64::new((theta / 2.0).cos(), 0.0);
        let expected1 = Complex64::new(0.0, -(theta / 2.0).sin());

        assert!(
            (state[0] - expected0).norm() < 1e-12,
            "RX(π/2)|0>: state[0] wrong: {:?}",
            state[0]
        );
        assert!(
            (state[1] - expected1).norm() < 1e-12,
            "RX(π/2)|0>: state[1] wrong: {:?}",
            state[1]
        );
    }

    #[test]
    fn test_ry_pi() {
        // RY(π)|0⟩ ≈ |1⟩  (up to global phase: sin(π/2)=1, cos(π/2)=0)
        let mut state = zero_state(1);
        apply_ry_simd(&mut state, PI, 0, 1);

        assert!(
            state[0].norm() < 1e-12,
            "RY(π)|0>: state[0] should be ~0, got {:?}",
            state[0]
        );
        assert!(
            (state[1] - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "RY(π)|0>: state[1] should be ~1, got {:?}",
            state[1]
        );
    }

    #[test]
    fn test_s_gate() {
        // S|1⟩ = i|1⟩
        let mut state = zero_state(1);
        apply_x_simd(&mut state, 0, 1); // |1⟩
        apply_s_simd(&mut state, 0, 1);

        assert!(
            state[0].norm() < 1e-12,
            "S|1>: state[0] should be 0"
        );
        assert!(
            (state[1] - Complex64::new(0.0, 1.0)).norm() < 1e-12,
            "S|1>: state[1] should be i"
        );
    }

    #[test]
    fn test_t_gate() {
        // T|1⟩ = exp(iπ/4)|1⟩
        use std::f64::consts::FRAC_PI_4;
        let mut state = zero_state(1);
        apply_x_simd(&mut state, 0, 1); // |1⟩
        apply_t_simd(&mut state, 0, 1);

        let expected = Complex64::new(FRAC_PI_4.cos(), FRAC_PI_4.sin());
        assert!(
            state[0].norm() < 1e-12,
            "T|1>: state[0] should be 0"
        );
        assert!(
            (state[1] - expected).norm() < 1e-12,
            "T|1>: state[1] wrong"
        );
    }

    // -----------------------------------------------------------------------
    // SIMD vs scalar consistency — 6-qubit random state (uses SIMD path)
    // -----------------------------------------------------------------------

    /// Simple LCG for reproducible random states without external rand dep.
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

        // Normalize
        let norm: f64 = state.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        state.iter_mut().for_each(|c| *c /= norm);
        state
    }

    #[test]
    fn test_simd_vs_scalar_h() {
        let n = 6usize;
        let base = lcg_random_state(n, 42);

        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_h_simd(&mut simd_state, target, n);

            let h = [
                [
                    Complex64::new(FRAC_1_SQRT_2, 0.0),
                    Complex64::new(FRAC_1_SQRT_2, 0.0),
                ],
                [
                    Complex64::new(FRAC_1_SQRT_2, 0.0),
                    Complex64::new(-FRAC_1_SQRT_2, 0.0),
                ],
            ];
            apply_gate_2x2_scalar(&mut scalar_state, h, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "SIMD vs scalar H mismatch at target={}: max_diff={}",
                target,
                diff
            );
        }
    }

    #[test]
    fn test_simd_vs_scalar_x() {
        let n = 6usize;
        let base = lcg_random_state(n, 123);
        let x_mat = [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ];

        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_x_simd(&mut simd_state, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, x_mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "SIMD vs scalar X mismatch at target={}: max_diff={}",
                target,
                diff
            );
        }
    }

    #[test]
    fn test_simd_vs_scalar_rz() {
        let n = 6usize;
        let base = lcg_random_state(n, 999);
        let theta = 1.23456_f64;
        let h = theta / 2.0;
        let rz_mat = [
            [Complex64::new(h.cos(), -h.sin()), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(h.cos(), h.sin())],
        ];

        for target in 0..n {
            let mut simd_state = base.clone();
            let mut scalar_state = base.clone();

            apply_rz_simd(&mut simd_state, theta, target, n);
            apply_gate_2x2_scalar(&mut scalar_state, rz_mat, target, n);

            let diff = max_diff(&simd_state, &scalar_state);
            assert!(
                diff < 1e-12,
                "SIMD vs scalar RZ mismatch at target={}: max_diff={}",
                target,
                diff
            );
        }
    }

    #[test]
    fn test_gate_2x2_simd_identity() {
        // Applying identity should leave state unchanged.
        let n = 4usize;
        let mut state = lcg_random_state(n, 7);
        let original = state.clone();
        let id = [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ];
        apply_gate_2x2_scalar(&mut state, id, 0, n);
        let diff = max_diff(&state, &original);
        assert!(diff < 1e-15, "Identity gate altered state: max_diff={}", diff);
    }

    #[test]
    fn test_y_gate_eigenvalue() {
        // Y|+y⟩ = |+y⟩ where |+y⟩ = (|0⟩ + i|1⟩)/√2
        let mut state = vec![
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(0.0, FRAC_1_SQRT_2),
        ];
        let original = state.clone();
        apply_y_simd(&mut state, 0, 1);

        // Y|+y⟩ = |+y⟩  (eigenvalue +1)
        let diff = max_diff(&state, &original);
        assert!(
            diff < 1e-12,
            "Y eigenstate property failed: max_diff={}",
            diff
        );
    }
}
