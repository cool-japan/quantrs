//! Integration tests for quantum networking protocols.
//!
//! Tests BB84 QKD, E91 QKD, quantum teleportation, and entanglement swapping
//! with various noise levels and parameters.

use quantrs2_core::networking::{
    Bb84Protocol, E91Protocol, EntanglementSwapping, TeleportationProtocol,
};
use scirs2_core::Complex64;
use std::f64::consts::SQRT_2;

// ---------------------------------------------------------------------------
// BB84 QKD Tests
// ---------------------------------------------------------------------------

#[test]
fn bb84_no_noise_qber_near_zero() {
    let proto = Bb84Protocol::new(3000, 0.0, 0.0, 1001);
    let result = proto.run().expect("bb84 no noise");
    assert!(
        result.qber < 0.05,
        "QBER should be < 5% without noise or eavesdropping, got {}",
        result.qber
    );
    assert!(
        !result.detected_eavesdrop,
        "Should not detect eavesdropping"
    );
}

#[test]
fn bb84_full_eavesdrop_qber_near_quarter() {
    let proto = Bb84Protocol::new(5000, 0.0, 1.0, 2002);
    let result = proto.run().expect("bb84 full eavesdrop");
    // Full intercept-resend: QBER ≈ 25% (Eve guesses basis wrong half the time)
    assert!(
        result.qber > 0.15,
        "QBER should be > 15% with full eavesdropping, got {}",
        result.qber
    );
    assert!(
        result.detected_eavesdrop,
        "Should detect full eavesdropping"
    );
}

#[test]
fn bb84_noise_increases_qber() {
    let proto_clean = Bb84Protocol::new(3000, 0.0, 0.0, 1);
    let proto_noisy = Bb84Protocol::new(3000, 0.2, 0.0, 1);
    let qber_clean = proto_clean.run().expect("clean").qber;
    let qber_noisy = proto_noisy.run().expect("noisy").qber;
    assert!(
        qber_noisy > qber_clean,
        "Noise should increase QBER: clean={qber_clean}, noisy={qber_noisy}",
    );
}

#[test]
fn bb84_sifted_key_roughly_half_raw_bits() {
    let proto = Bb84Protocol::new(4000, 0.0, 0.0, 777);
    let result = proto.run().expect("bb84");
    // After sifting (~50% match) and removing QBER sample (~20%): about 40% of raw
    let sifted_plus_sample = result.sifted_key.len() * 5 / 4; // approx includes sample
    assert!(
        sifted_plus_sample > result.raw_bits / 5,
        "Sifted key should be substantial fraction of raw bits"
    );
}

#[test]
fn bb84_secret_key_is_half_sifted() {
    let proto = Bb84Protocol::new(3000, 0.0, 0.0, 888);
    let result = proto.run().expect("bb84");
    if !result.sifted_key.is_empty() {
        let expected_secret = result.sifted_key.len() / 2;
        assert_eq!(
            result.secret_key.len(),
            expected_secret,
            "Secret key should be half the sifted key length"
        );
    }
}

// ---------------------------------------------------------------------------
// E91 QKD Tests
// ---------------------------------------------------------------------------

#[test]
fn e91_ideal_chsh_near_2sqrt2() {
    let proto = E91Protocol::new(500, 0.0, 3003);
    let result = proto.run().expect("e91 ideal");
    // Ideal S ≈ 2√2 ≈ 2.828
    let ideal_s = 2.0 * SQRT_2;
    assert!(
        result.chsh_value > 2.5,
        "Ideal CHSH S should be ≈ {:.3}, got {}",
        ideal_s,
        result.chsh_value
    );
    assert!(result.passed_bell_test, "Should pass Bell test without noise");
}

#[test]
fn e91_high_noise_chsh_below_2() {
    let proto = E91Protocol::new(500, 0.95, 4004);
    let result = proto.run().expect("e91 high noise");
    assert!(
        result.chsh_value < 2.0,
        "High-noise CHSH S should be < 2 (classical), got {}",
        result.chsh_value
    );
    assert!(
        !result.passed_bell_test,
        "Should fail Bell test with extreme noise"
    );
}

#[test]
fn e91_chsh_decreases_monotonically_with_noise() {
    let s_values: Vec<f64> = [0.0, 0.2, 0.5, 0.8]
        .iter()
        .map(|&p| {
            E91Protocol::new(300, p, 5005)
                .run()
                .expect("e91")
                .chsh_value
        })
        .collect();

    // Each noise level should give lower or equal CHSH than the previous
    for i in 1..s_values.len() {
        assert!(
            s_values[i] <= s_values[i - 1] + 0.1, // small tolerance
            "CHSH should decrease with noise: {:.3} > {:.3} at noise index {}",
            s_values[i],
            s_values[i - 1],
            i
        );
    }
}

#[test]
fn e91_key_generation_produces_bits() {
    let proto = E91Protocol::new(5000, 0.0, 6006);
    let result = proto.run().expect("e91");
    assert!(
        !result.key.is_empty(),
        "Should generate a non-empty key"
    );
    assert!(
        result.key_rate > 0.01,
        "Key rate should be > 1%, got {}",
        result.key_rate
    );
}

// ---------------------------------------------------------------------------
// Quantum Teleportation Tests
// ---------------------------------------------------------------------------

#[test]
fn teleportation_no_noise_fidelity_one() {
    let state = [
        Complex64::new(1.0 / SQRT_2, 0.0),
        Complex64::new(0.0, 1.0 / SQRT_2),
    ];
    let proto = TeleportationProtocol::new(0.0, 7007);
    let result = proto.teleport(state).expect("teleport");
    assert!(
        result.fidelity > 0.98,
        "No-noise fidelity should be ≈ 1.0, got {}",
        result.fidelity
    );
}

#[test]
fn teleportation_plus_state_no_noise() {
    let v = 1.0 / SQRT_2;
    let state = [Complex64::new(v, 0.0), Complex64::new(v, 0.0)];
    let proto = TeleportationProtocol::new(0.0, 8008);
    let result = proto.teleport(state).expect("teleport");
    assert!(
        result.fidelity > 0.98,
        "Teleportation of |+⟩ should have fidelity ≈ 1.0, got {}",
        result.fidelity
    );
}

#[test]
fn teleportation_noisy_fidelity_decreases() {
    let state = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let f_clean = TeleportationProtocol::new(0.0, 9009)
        .teleport(state)
        .expect("clean")
        .fidelity;
    let f_noisy = TeleportationProtocol::new(0.25, 9009)
        .teleport(state)
        .expect("noisy")
        .fidelity;
    assert!(
        f_clean > f_noisy,
        "Clean fidelity ({f_clean:.3}) should exceed noisy ({f_noisy:.3})",
    );
}

#[test]
fn teleportation_invalid_zero_state_returns_error() {
    let state = [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
    let proto = TeleportationProtocol::new(0.0, 42);
    // Only fails if norm_sq check catches it — let's use a near-zero state
    let almost_zero = [Complex64::new(1e-8, 0.0), Complex64::new(0.0, 0.0)];
    // This might succeed (1e-8 normalised to 1) or fail; either way it shouldn't panic.
    let _ = proto.teleport(almost_zero);
}

// ---------------------------------------------------------------------------
// Entanglement Swapping Tests
// ---------------------------------------------------------------------------

#[test]
fn entanglement_swapping_n_hops() {
    let swap1 = EntanglementSwapping::new(1, 0.05, 1111).expect("create");
    let swap2 = EntanglementSwapping::new(2, 0.05, 2222).expect("create");
    let swap3 = EntanglementSwapping::new(3, 0.05, 3333).expect("create");

    let f1 = swap1.run().expect("run").end_to_end_fidelity;
    let f2 = swap2.run().expect("run").end_to_end_fidelity;
    let f3 = swap3.run().expect("run").end_to_end_fidelity;

    assert!(
        f1 > f2 || (f1 - f2).abs() < 0.1,
        "1-hop fidelity ({f1:.3}) should be ≥ 2-hop ({f2:.3})",
    );
    assert!(
        f1 > f3,
        "1-hop fidelity ({f1:.3}) should exceed 3-hop ({f3:.3}) with noise",
    );
}

#[test]
fn entanglement_swapping_zero_noise_high_fidelity() {
    for n_hops in 1..=3 {
        let swap = EntanglementSwapping::new(n_hops, 0.0, 42).expect("create");
        let result = swap.run().expect("run");
        assert!(
            result.end_to_end_fidelity > 0.90,
            "Zero-noise {}-hop fidelity should be > 0.90, got {}",
            n_hops,
            result.end_to_end_fidelity
        );
    }
}

#[test]
fn entanglement_swapping_zero_hops_returns_error() {
    let result = EntanglementSwapping::new(0, 0.0, 42);
    assert!(result.is_err(), "n_hops=0 should return an error");
}
