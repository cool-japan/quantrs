//! Tests for quantum simulators
//!
//! This module provides tests for the various simulator implementations
//! to ensure correctness and compatibility.

use std::f64::consts::FRAC_1_SQRT_2;
use num_complex::Complex64;

use quantrs_circuit::builder::{Circuit, Simulator};
use quantrs_core::register::Register;

use crate::statevector::StateVectorSimulator;
use crate::optimized_simulator::OptimizedSimulator;

/// Create a bell state circuit
fn create_bell_circuit<const N: usize>() -> Circuit<N> {
    let mut circuit = Circuit::new();
    
    // Apply Hadamard to qubit 0
    circuit.h(0).unwrap();
    
    // Apply CNOT with qubit 0 as control and qubit 1 as target
    circuit.cnot(0, 1).unwrap();
    
    circuit
}

/// Create a GHZ state circuit for N qubits
fn create_ghz_circuit<const N: usize>() -> Circuit<N> {
    let mut circuit = Circuit::new();
    
    // Apply Hadamard to qubit 0
    circuit.h(0).unwrap();
    
    // Apply CNOT gates to entangle all qubits
    for i in 1..N {
        circuit.cnot(0, i).unwrap();
    }
    
    circuit
}

/// Create a random circuit with the specified number of gates
fn create_random_circuit<const N: usize>(num_gates: usize) -> Circuit<N> {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;
    use std::f64::consts::PI;
    
    let mut circuit = Circuit::new();
    let mut rng = StdRng::seed_from_u64(42); // Use fixed seed for reproducibility
    
    for _ in 0..num_gates {
        let gate_type = rng.gen_range(0..5);
        
        match gate_type {
            0 => {
                // Hadamard gate
                let target = rng.gen_range(0..N);
                circuit.h(target).unwrap();
            },
            1 => {
                // Pauli-X gate
                let target = rng.gen_range(0..N);
                circuit.x(target).unwrap();
            },
            2 => {
                // Rotation-Z gate
                let target = rng.gen_range(0..N);
                let angle = rng.gen_range(0.0..2.0 * PI);
                circuit.rz(target, angle).unwrap();
            },
            3 => {
                // CNOT gate
                let control = rng.gen_range(0..N);
                let mut target = rng.gen_range(0..N);
                while target == control {
                    target = rng.gen_range(0..N);
                }
                circuit.cnot(control, target).unwrap();
            },
            4 => {
                // CZ gate
                let control = rng.gen_range(0..N);
                let mut target = rng.gen_range(0..N);
                while target == control {
                    target = rng.gen_range(0..N);
                }
                circuit.cz(control, target).unwrap();
            },
            _ => unreachable!(),
        }
    }
    
    circuit
}

/// Compare results between standard and optimized simulators
fn compare_simulators<const N: usize>(circuit: &Circuit<N>, epsilon: f64) -> bool {
    let standard_sim = StateVectorSimulator::new();
    let optimized_sim = OptimizedSimulator::new();
    
    let standard_result = standard_sim.run(circuit).unwrap();
    let optimized_result = optimized_sim.run(circuit).unwrap();
    
    let standard_state = standard_result.amplitudes();
    let optimized_state = optimized_result.amplitudes();
    
    // Check that the dimensions match
    if standard_state.len() != optimized_state.len() {
        println!("State vector dimensions don't match");
        return false;
    }
    
    // Check each amplitude
    for (i, (std_amp, opt_amp)) in standard_state.iter().zip(optimized_state.iter()).enumerate() {
        let diff = (std_amp - opt_amp).norm();
        if diff > epsilon {
            println!("Amplitude {} differs: standard={}, optimized={}, diff={}", 
                i, std_amp, opt_amp, diff);
            return false;
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bell_state() {
        const N: usize = 2;
        let circuit = create_bell_circuit::<N>();
        
        // Run the circuit with both simulators
        let standard_sim = StateVectorSimulator::new();
        let optimized_sim = OptimizedSimulator::new();
        
        let standard_result = standard_sim.run(&circuit).unwrap();
        let optimized_result = optimized_sim.run(&circuit).unwrap();
        
        // Expected result: (|00> + |11>) / sqrt(2)
        let expected_amplitudes = vec![
            Complex64::new(FRAC_1_SQRT_2, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(FRAC_1_SQRT_2, 0.0),
        ];
        
        // Check standard simulator
        let standard_state = standard_result.amplitudes();
        for (i, (actual, expected)) in standard_state.iter().zip(expected_amplitudes.iter()).enumerate() {
            let diff = (actual - expected).norm();
            assert!(diff < 1e-10, "Standard simulator: state[{}] differs by {}", i, diff);
        }
        
        // Check optimized simulator
        let optimized_state = optimized_result.amplitudes();
        for (i, (actual, expected)) in optimized_state.iter().zip(expected_amplitudes.iter()).enumerate() {
            let diff = (actual - expected).norm();
            assert!(diff < 1e-10, "Optimized simulator: state[{}] differs by {}", i, diff);
        }
    }
    
    #[test]
    fn test_ghz_state() {
        const N: usize = 3;
        let circuit = create_ghz_circuit::<N>();
        
        // Run the circuit with both simulators
        let standard_sim = StateVectorSimulator::new();
        let optimized_sim = OptimizedSimulator::new();
        
        let standard_result = standard_sim.run(&circuit).unwrap();
        let optimized_result = optimized_sim.run(&circuit).unwrap();
        
        // Expected result: (|000> + |111>) / sqrt(2)
        let mut expected_amplitudes = vec![Complex64::new(0.0, 0.0); 1 << N];
        expected_amplitudes[0] = Complex64::new(FRAC_1_SQRT_2, 0.0);
        expected_amplitudes[7] = Complex64::new(FRAC_1_SQRT_2, 0.0);
        
        // Check standard simulator
        let standard_state = standard_result.amplitudes();
        for (i, (actual, expected)) in standard_state.iter().zip(expected_amplitudes.iter()).enumerate() {
            let diff = (actual - expected).norm();
            assert!(diff < 1e-10, "Standard simulator: state[{}] differs by {}", i, diff);
        }
        
        // Check optimized simulator
        let optimized_state = optimized_result.amplitudes();
        for (i, (actual, expected)) in optimized_state.iter().zip(expected_amplitudes.iter()).enumerate() {
            let diff = (actual - expected).norm();
            assert!(diff < 1e-10, "Optimized simulator: state[{}] differs by {}", i, diff);
        }
    }
    
    #[test]
    fn test_random_circuit_4qubits() {
        const N: usize = 4;
        let circuit = create_random_circuit::<N>(20);
        
        assert!(compare_simulators(&circuit, 1e-10), 
                "4-qubit random circuit: simulators disagree");
    }
    
    #[test]
    fn test_random_circuit_8qubits() {
        const N: usize = 8;
        let circuit = create_random_circuit::<N>(20);
        
        assert!(compare_simulators(&circuit, 1e-10), 
                "8-qubit random circuit: simulators disagree");
    }
    
    #[test]
    #[ignore] // This test is resource-intensive, so we'll ignore it by default
    fn test_random_circuit_16qubits() {
        const N: usize = 16;
        let circuit = create_random_circuit::<N>(10);
        
        assert!(compare_simulators(&circuit, 1e-10), 
                "16-qubit random circuit: simulators disagree");
    }
    
    #[test]
    #[ignore] // This test is even more resource-intensive
    fn test_random_circuit_20qubits() {
        const N: usize = 20;
        let circuit = create_random_circuit::<N>(5);
        
        assert!(compare_simulators(&circuit, 1e-10), 
                "20-qubit random circuit: simulators disagree");
    }
}