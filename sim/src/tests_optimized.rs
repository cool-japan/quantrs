//! Comprehensive tests for optimized quantum simulators
//!
//! This module provides a test suite for the optimized simulator implementations
//! to verify correctness and benchmark performance.

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::f64::consts::FRAC_1_SQRT_2;

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::{qubit::QubitId, register::Register};

use crate::benchmark::generate_benchmark_circuit;
use crate::optimized_simulator::OptimizedSimulator;
use crate::optimized_simulator_chunked::OptimizedSimulatorChunked;
use crate::optimized_simulator_simple::OptimizedSimulatorSimple;
use crate::statevector::StateVectorSimulator;

/// Test the Bell state circuit with different simulator implementations
#[test]
fn test_bell_state_all_simulators() {
    // Create a Bell state circuit
    let mut circuit = Circuit::<2>::new();
    circuit
        .h(QubitId::new(0))
        .unwrap()
        .cnot(QubitId::new(0), QubitId::new(1))
        .unwrap();

    // Run with standard simulator
    let standard_sim = StateVectorSimulator::new();
    let standard_result = standard_sim.run(&circuit).unwrap();

    // Run with optimized simulators
    let simple_opt_sim = OptimizedSimulatorSimple::new();
    let simple_opt_result = simple_opt_sim.run(&circuit).unwrap();

    let chunked_sim = OptimizedSimulatorChunked::new();
    let chunked_opt_result = chunked_sim.run(&circuit).unwrap();

    let full_opt_sim = OptimizedSimulator::new();
    let full_opt_result = full_opt_sim.run(&circuit).unwrap();

    // Expected amplitudes for the Bell state
    let expected_amplitudes = vec![
        Complex64::new(FRAC_1_SQRT_2, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, 0.0),
    ];

    // Check that all simulators produce the expected result
    assert_state_vector_close(&standard_result.amplitudes(), &expected_amplitudes, 1e-10);
    assert_state_vector_close(&simple_opt_result.amplitudes(), &expected_amplitudes, 1e-10);
    assert_state_vector_close(
        &chunked_opt_result.amplitudes(),
        &expected_amplitudes,
        1e-10,
    );
    assert_state_vector_close(&full_opt_result.amplitudes(), &expected_amplitudes, 1e-10);
}

/// Test the GHZ state circuit with different simulator implementations
#[test]
fn test_ghz_state_all_simulators() {
    // Create a GHZ state circuit for 3 qubits (|000> + |111>)/sqrt(2)
    let mut circuit = Circuit::<3>::new();
    circuit
        .h(QubitId::new(0))
        .unwrap()
        .cnot(QubitId::new(0), QubitId::new(1))
        .unwrap()
        .cnot(QubitId::new(1), QubitId::new(2))
        .unwrap();

    // Run with standard simulator
    let standard_sim = StateVectorSimulator::new();
    let standard_result = standard_sim.run(&circuit).unwrap();

    // Run with optimized simulators
    let simple_opt_sim = OptimizedSimulatorSimple::new();
    let simple_opt_result = simple_opt_sim.run(&circuit).unwrap();

    let chunked_sim = OptimizedSimulatorChunked::new();
    let chunked_opt_result = chunked_sim.run(&circuit).unwrap();

    let full_opt_sim = OptimizedSimulator::new();
    let full_opt_result = full_opt_sim.run(&circuit).unwrap();

    // Expected amplitudes for the GHZ state
    let mut expected_amplitudes = vec![Complex64::new(0.0, 0.0); 8];
    expected_amplitudes[0] = Complex64::new(FRAC_1_SQRT_2, 0.0);
    expected_amplitudes[7] = Complex64::new(FRAC_1_SQRT_2, 0.0);

    // Check that all simulators produce the expected result
    assert_state_vector_close(&standard_result.amplitudes(), &expected_amplitudes, 1e-10);
    assert_state_vector_close(&simple_opt_result.amplitudes(), &expected_amplitudes, 1e-10);
    assert_state_vector_close(
        &chunked_opt_result.amplitudes(),
        &expected_amplitudes,
        1e-10,
    );
    assert_state_vector_close(&full_opt_result.amplitudes(), &expected_amplitudes, 1e-10);
}

/// Test a quantum fourier transform-like circuit
#[test]
fn test_qft_like_circuit() {
    // Create a QFT-like circuit for 3 qubits
    let circuit = create_simple_qft_circuit::<3>();

    // Run with standard simulator
    let standard_sim = StateVectorSimulator::new();
    let standard_result = standard_sim.run(&circuit).unwrap();

    // Run with optimized simulators
    let simple_opt_sim = OptimizedSimulatorSimple::new();
    let simple_opt_result = simple_opt_sim.run(&circuit).unwrap();

    let chunked_sim = OptimizedSimulatorChunked::new();
    let chunked_opt_result = chunked_sim.run(&circuit).unwrap();

    let full_opt_sim = OptimizedSimulator::new();
    let full_opt_result = full_opt_sim.run(&circuit).unwrap();

    // Check that all simulators produce equivalent results
    assert_state_vector_close(
        &standard_result.amplitudes(),
        &simple_opt_result.amplitudes(),
        1e-10,
    );
    assert_state_vector_close(
        &standard_result.amplitudes(),
        &chunked_opt_result.amplitudes(),
        1e-10,
    );
    assert_state_vector_close(
        &standard_result.amplitudes(),
        &full_opt_result.amplitudes(),
        1e-10,
    );
}

/// Test consistency between simulators on random circuits
#[test]
fn test_random_circuit_consistency() {
    const QUBITS: usize = 5;
    const NUM_GATES: usize = 20;
    const TWO_QUBIT_RATIO: f64 = 0.3;

    // Generate a random circuit
    let circuit = generate_benchmark_circuit::<QUBITS>(NUM_GATES, TWO_QUBIT_RATIO);

    // Run with standard simulator
    let standard_sim = StateVectorSimulator::new();
    let standard_result = standard_sim.run(&circuit).unwrap();

    // Run with optimized simulators
    let simple_opt_sim = OptimizedSimulatorSimple::new();
    let simple_opt_result = simple_opt_sim.run(&circuit).unwrap();

    let chunked_sim = OptimizedSimulatorChunked::new();
    let chunked_opt_result = chunked_sim.run(&circuit).unwrap();

    let full_opt_sim = OptimizedSimulator::new();
    let full_opt_result = full_opt_sim.run(&circuit).unwrap();

    // Check that all simulators produce equivalent results
    assert_state_vector_close(
        &standard_result.amplitudes(),
        &simple_opt_result.amplitudes(),
        1e-10,
    );
    assert_state_vector_close(
        &standard_result.amplitudes(),
        &chunked_opt_result.amplitudes(),
        1e-10,
    );
    assert_state_vector_close(
        &standard_result.amplitudes(),
        &full_opt_result.amplitudes(),
        1e-10,
    );
}

/// Test a larger circuit with more qubits
#[test]
fn test_larger_circuit() {
    const QUBITS: usize = 10;

    // Create a circuit with random gates
    let mut circuit = Circuit::<QUBITS>::new();

    // Apply Hadamard to all qubits
    for i in 0..QUBITS {
        circuit.h(QubitId::new(i as u32)).unwrap();
    }

    // Apply some controlled gates
    for i in 0..(QUBITS - 1) {
        circuit
            .cnot(QubitId::new(i as u32), QubitId::new((i + 1) as u32))
            .unwrap();
    }

    // Apply some rotations
    for i in 0..QUBITS {
        circuit
            .rz(
                QubitId::new(i as u32),
                std::f64::consts::PI / (i + 1) as f64,
            )
            .unwrap();
    }

    // Run with optimized simulators
    let simple_opt_sim = OptimizedSimulatorSimple::new();
    let simple_opt_result = simple_opt_sim.run(&circuit).unwrap();

    let full_opt_sim = OptimizedSimulator::new();
    let full_opt_result = full_opt_sim.run(&circuit).unwrap();

    // For a larger circuit, we focus on consistency between optimized simulators
    assert_state_vector_close(
        &simple_opt_result.amplitudes(),
        &full_opt_result.amplitudes(),
        1e-10,
    );
}

/// Create a simplified Quantum Fourier Transform-like circuit
fn create_simple_qft_circuit<const N: usize>() -> Circuit<N> {
    let mut circuit = Circuit::<N>::new();

    // Apply H to all qubits
    for i in 0..N {
        circuit.h(QubitId::new(i as u32)).unwrap();
    }

    // Apply controlled-Z gates between adjacent qubits
    for i in 0..(N - 1) {
        circuit
            .cz(QubitId::new(i as u32), QubitId::new((i + 1) as u32))
            .unwrap();
    }

    // Apply some Z rotations
    for i in 0..N {
        circuit
            .rz(QubitId::new(i as u32), std::f64::consts::PI / 4.0)
            .unwrap();
    }

    // Apply SWAP operations at the end
    for i in 0..(N / 2) {
        circuit
            .swap(QubitId::new(i as u32), QubitId::new((N - i - 1) as u32))
            .unwrap();
    }

    circuit
}

/// Helper function to check if two state vectors are approximately equal
fn assert_state_vector_close(actual: &[Complex64], expected: &[Complex64], epsilon: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "State vectors must have the same length"
    );

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).norm() < epsilon,
            "Amplitude at index {} differs: actual={}, expected={}, diff={}",
            i,
            a,
            e,
            (a - e).norm()
        );
    }
}
