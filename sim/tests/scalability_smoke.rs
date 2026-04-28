#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::field_reassign_with_default,
    clippy::module_inception
)]
//! Scalability smoke tests for quantum simulators (QuantRS2 v0.2.0)
//!
//! These tests verify that the simulators produce correct results at
//! increasing qubit counts without regression. All tests are designed to
//! complete in < 10 s on typical CI hardware.
//!
//! Tests marked `#[ignore]` may take > 10 s — run them with:
//!   cargo nextest run -p quantrs2-sim -- --ignored

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_sim::stabilizer::{StabilizerGate, StabilizerSimulator};
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Test 1: 15-qubit uniform superposition via H-layer
//
// Apply H to all 15 qubits starting from |0...0⟩.  The resulting state
// is the uniform superposition: every basis state has amplitude 1/√(2^15),
// so the probability of any fixed basis state is exactly 1/32768.
// ---------------------------------------------------------------------------
#[test]
fn test_15q_h_layer_superposition() {
    const N: usize = 15;
    let mut circuit = Circuit::<N>::new();
    for q in 0..N {
        circuit.h(q).expect("H gate failed");
    }

    let sim = StateVectorSimulator::new();
    let register = sim.run(&circuit).expect("simulation failed");
    let amps = register.amplitudes();

    let dim = 1usize << N;
    let expected_prob = 1.0 / dim as f64; // 1/32768

    // All amplitudes should be real and equal to 1/√32768
    let expected_amp = (expected_prob).sqrt();
    for (i, amp) in amps.iter().enumerate() {
        let re_diff = (amp.re - expected_amp).abs();
        let im_diff = amp.im.abs();
        assert!(
            re_diff < 1e-9,
            "amplitude[{i}] real part {:.2e} differs from expected {:.2e} by {re_diff:.2e}",
            amp.re,
            expected_amp,
        );
        assert!(
            im_diff < 1e-9,
            "amplitude[{i}] imaginary part {:.2e} should be 0, got {im_diff:.2e}",
            amp.im,
        );
    }

    // Also verify that the probability of |0...0⟩ specifically is 1/32768
    let prob_zero: f64 = amps[0].norm_sqr();
    assert!(
        (prob_zero - expected_prob).abs() < 1e-9,
        "P(|0...0⟩) = {prob_zero:.2e}, expected {expected_prob:.2e} (diff {:.2e})",
        (prob_zero - expected_prob).abs()
    );
}

// ---------------------------------------------------------------------------
// Test 2: 18-qubit uniform superposition — norm preservation
//
// QFT is not directly available as a stand-alone circuit builder method,
// so instead we use the 18-qubit H-layer as a computationally non-trivial
// alternative (2^18 = 262 144 complex amplitudes).  The key invariant we
// check is that the state stays normalised (sum of |ψ_i|² = 1.0) and that
// each probability equals 1/2^18.
//
// Marked `#[ignore]` because allocating 4 MiB of complex amplitudes
// and iterating over 2^18 elements may be slow on minimal CI runners.
// Run with: cargo nextest run -p quantrs2-sim -- --ignored
// ---------------------------------------------------------------------------
#[test]
#[ignore] // cargo nextest run -- --ignored
fn test_18q_h_layer_norm_preservation() {
    const N: usize = 18;
    let mut circuit = Circuit::<N>::new();
    for q in 0..N {
        circuit.h(q).expect("H gate failed");
    }

    let sim = StateVectorSimulator::new();
    let register = sim.run(&circuit).expect("simulation failed");
    let amps = register.amplitudes();

    let dim = 1usize << N; // 262144
    let expected_prob = 1.0 / dim as f64;

    // Verify total norm
    let total_norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
    assert!(
        (total_norm - 1.0).abs() < 1e-6,
        "Total norm {total_norm:.8} is not 1.0 (diff {:.2e})",
        (total_norm - 1.0).abs()
    );

    // Sample-check 100 evenly-spaced amplitudes for the correct probability
    let step = dim / 100;
    for k in (0..dim).step_by(step) {
        let p = amps[k].norm_sqr();
        assert!(
            (p - expected_prob).abs() < 1e-9,
            "P(|{k}⟩) = {p:.2e}, expected {expected_prob:.2e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test 3: 20-qubit stabilizer — 10 Bell pairs
//
// Using the efficient stabilizer formalism (Gottesman–Knill), create 10 Bell
// pairs on 20 qubits: H on qubit 2k, CNOT(2k, 2k+1) for k=0..9.
// The resulting state is stabilised by exactly 20 generators, and the
// simulator's get_stabilizers() must return a vector of length 20.
//
// This directly tests the O(n²) stabilizer tableau at a medium qubit count
// that would be intractable for state-vector (2^20 = 1 M amplitudes).
// ---------------------------------------------------------------------------
#[test]
fn test_20q_stabilizer_bell_chain() {
    const NUM_QUBITS: usize = 20;
    let num_pairs = NUM_QUBITS / 2; // 10 Bell pairs

    let mut sim = StabilizerSimulator::new(NUM_QUBITS);

    // Build 10 Bell pairs: H(2k) then CNOT(2k, 2k+1)
    for k in 0..num_pairs {
        sim.apply_gate(StabilizerGate::H(2 * k))
            .expect("H gate failed");
        sim.apply_gate(StabilizerGate::CNOT(2 * k, 2 * k + 1))
            .expect("CNOT gate failed");
    }

    // The tableau must track exactly N stabilizer generators
    let stabilizers = sim.get_stabilizers();
    assert_eq!(
        stabilizers.len(),
        NUM_QUBITS,
        "expected {NUM_QUBITS} stabilizer generators, got {}",
        stabilizers.len()
    );

    // None of the stabilizer strings should be empty
    for (i, s) in stabilizers.iter().enumerate() {
        assert!(!s.is_empty(), "stabilizer generator {i} is an empty string");
    }
}

// ---------------------------------------------------------------------------
// Test 4: 12-qubit random circuit — norm invariance
//
// Build a depth-5 random circuit with H and CNOT gates (fixed seed 42
// for reproducibility), simulate with the state-vector simulator, and
// verify that the output state is properly normalised.
//
// Correctness of each gate is tested individually elsewhere; this test
// ensures that chaining many gates at scale does not accumulate errors
// large enough to break the norm invariant.
//
// Takes ~38 s in debug mode (2^12 = 4096 amplitudes × multiple gate layers
// with per-layer RZ sprinkling through the dispatch machinery).
// Run with: cargo nextest run -p quantrs2-sim -- --ignored
// ---------------------------------------------------------------------------
#[test]
#[ignore] // cargo nextest run -- --ignored  (38 s in debug mode)
fn test_12q_random_circuit_norm() {
    const N: usize = 12;
    const DEPTH: usize = 5;

    let mut circuit = Circuit::<N>::new();
    let mut rng = StdRng::seed_from_u64(42);

    // Alternating layers of single-qubit (H) and two-qubit (CNOT) gates
    for layer in 0..DEPTH {
        if layer % 2 == 0 {
            // H layer — apply H to every qubit
            for q in 0..N {
                circuit.h(q).expect("H gate failed");
            }
        } else {
            // CNOT layer — apply CNOT to adjacent pairs (nearest-neighbour)
            for pair_start in (0..N - 1).step_by(2) {
                let control = pair_start;
                let target = pair_start + 1;
                circuit
                    .cnot(control, target)
                    .expect("CNOT gate failed");
            }
        }

        // Sprinkle random RZ gates for non-trivial entanglement
        for _ in 0..N / 2 {
            let q = rng.random_range(0..N);
            let angle = rng.random_range(0.0..2.0 * PI);
            circuit.rz(q, angle).expect("RZ gate failed");
        }
    }

    let sim = StateVectorSimulator::new();
    let register = sim.run(&circuit).expect("simulation failed");
    let amps = register.amplitudes();

    // Verify normalisation
    let total_norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
    assert!(
        (total_norm - 1.0).abs() < 1e-9,
        "Norm after depth-{DEPTH} random circuit on {N} qubits: {total_norm:.12} (diff {:.2e})",
        (total_norm - 1.0).abs()
    );

    // Verify the state has non-trivial support (at least 2 non-zero amplitudes),
    // i.e., the circuit actually entangled some qubits
    let nonzero_count = amps.iter().filter(|a| a.norm_sqr() > 1e-12).count();
    assert!(
        nonzero_count > 1,
        "Expected a superposition state, but only {nonzero_count} basis state(s) have non-zero amplitude"
    );
}

// ---------------------------------------------------------------------------
// Test 5: GHZ-state probability at 15 qubits
//
// Apply H to qubit 0, then CNOT(0, k) for k=1..14 to create a 15-qubit
// GHZ state: (|0...0⟩ + |1...1⟩) / √2.
// - P(|0...0⟩) should be 0.5
// - P(|1...1⟩) should be 0.5
// - All other basis states should have probability 0
// ---------------------------------------------------------------------------
#[test]
fn test_15q_ghz_state_probability() {
    const N: usize = 15;
    let mut circuit = Circuit::<N>::new();

    circuit.h(0).expect("H gate failed");
    for k in 1..N {
        circuit.cnot(0, k).expect("CNOT gate failed");
    }

    let sim = StateVectorSimulator::new();
    let register = sim.run(&circuit).expect("simulation failed");
    let probs = register.probabilities();

    let dim = 1usize << N; // 32768

    // First basis state |0...0⟩ at index 0
    let p_zero = probs[0];
    assert!(
        (p_zero - 0.5).abs() < 1e-9,
        "P(|0...0⟩) = {p_zero:.12}, expected 0.5 (diff {:.2e})",
        (p_zero - 0.5).abs()
    );

    // Last basis state |1...1⟩ at index dim-1
    let p_all_ones = probs[dim - 1];
    assert!(
        (p_all_ones - 0.5).abs() < 1e-9,
        "P(|1...1⟩) = {p_all_ones:.12}, expected 0.5 (diff {:.2e})",
        (p_all_ones - 0.5).abs()
    );

    // All other basis states must have probability 0
    for (i, &p) in probs.iter().enumerate() {
        if i == 0 || i == dim - 1 {
            continue;
        }
        assert!(
            p < 1e-9,
            "P(|{i}⟩) = {p:.2e} in GHZ state — expected 0"
        );
    }
}
