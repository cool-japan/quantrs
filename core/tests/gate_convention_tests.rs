//! Regression tests for the Rz-family sign convention.
//!
//! Rz(θ) must equal exp(-i * θ/2 * Z) = diag(e^{-iθ/2}, e^{+iθ/2}), matching
//! the OpenQASM 3 `stdgates.inc` / Qiskit convention, this crate's own
//! RotationX/RotationY/RZZ/RZX sign, and the specialized RZ paths in
//! `quantrs2-sim` (`RZSpecialized`, adaptive gate fusion, circuit interfaces).
//!
//! These tests are deliberately phase-sensitive: magnitude-only assertions
//! are satisfied by any diagonal unitary and cannot detect a reversed sign.
//! See <https://github.com/cool-japan/quantrs/issues/32>.

use quantrs2_core::gate::multi::CRZ;
use quantrs2_core::gate::single::{Phase, RotationZ};
use quantrs2_core::gate::GateOp;
use quantrs2_core::parametric::ParametricRotationZ;
use quantrs2_core::qubit::QubitId;
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

const EPS: f64 = 1e-12;

fn assert_entry(re: f64, im: f64, expected_re: f64, expected_im: f64, what: &str) {
    assert!(
        (re - expected_re).abs() < EPS && (im - expected_im).abs() < EPS,
        "{what}: got {re}{im:+}i, expected {expected_re}{expected_im:+}i"
    );
}

#[test]
fn rotation_z_matches_openqasm3_standard() {
    let rz = RotationZ {
        target: QubitId(0),
        theta: FRAC_PI_2,
    };
    let m = rz.matrix().expect("RZ matrix should be available");
    let c = FRAC_PI_4.cos();
    // Rz(π/2) = diag(e^{-iπ/4}, e^{+iπ/4})
    assert_entry(m[0].re, m[0].im, c, -c, "Rz(π/2)[0][0]");
    assert_entry(m[3].re, m[3].im, c, c, "Rz(π/2)[1][1]");
    assert_entry(m[1].re, m[1].im, 0.0, 0.0, "Rz(π/2)[0][1]");
    assert_entry(m[2].re, m[2].im, 0.0, 0.0, "Rz(π/2)[1][0]");
}

#[test]
fn rotation_z_half_pi_is_s_up_to_global_phase() {
    // Within this gate set, Rz(π/2) must equal the S gate up to global phase
    // (their diagonal ratios m11/m00 must both be +i).
    let rz = RotationZ {
        target: QubitId(0),
        theta: FRAC_PI_2,
    };
    let s = Phase { target: QubitId(0) };
    let m = rz.matrix().expect("RZ matrix should be available");
    let ms = s.matrix().expect("S matrix should be available");
    let ratio = m[3] / m[0];
    let s_ratio = ms[3] / ms[0];
    assert_entry(
        ratio.re,
        ratio.im,
        s_ratio.re,
        s_ratio.im,
        "Rz(π/2) diagonal ratio vs S",
    );
    assert_entry(ratio.re, ratio.im, 0.0, 1.0, "Rz(π/2) diagonal ratio");
}

#[test]
fn parametric_rotation_z_matches_gate_rotation_z() {
    let theta = 0.7312;
    let gate = RotationZ {
        target: QubitId(0),
        theta,
    };
    let par = ParametricRotationZ::new(QubitId(0), theta);
    let mg = gate.matrix().expect("RZ matrix should be available");
    let mp = par
        .matrix()
        .expect("parametric RZ matrix should be available");
    for (i, (a, b)) in mg.iter().zip(mp.iter()).enumerate() {
        assert_entry(
            a.re,
            a.im,
            b.re,
            b.im,
            &format!("RotationZ vs ParametricRotationZ entry {i}"),
        );
    }
}

#[test]
fn crz_target_block_matches_openqasm3_standard() {
    let crz = CRZ {
        control: QubitId(0),
        target: QubitId(1),
        theta: FRAC_PI_2,
    };
    let m = crz.matrix().expect("CRZ matrix should be available");
    let c = FRAC_PI_4.cos();
    // The control=|1⟩ block applies Rz(π/2) = diag(e^{-iπ/4}, e^{+iπ/4});
    // in the row-major 4x4 layout these are entries [10] and [15].
    assert_entry(m[10].re, m[10].im, c, -c, "CRZ(π/2)[2][2]");
    assert_entry(m[15].re, m[15].im, c, c, "CRZ(π/2)[3][3]");
}
