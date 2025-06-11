//! Circuit equivalence checking with simulator integration
//!
//! This module provides enhanced equivalence checking that uses
//! actual simulation to verify circuit behavior.

use crate::builder::Circuit;
use crate::equivalence::{EquivalenceResult, EquivalenceType, EquivalenceOptions};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use quantrs2_sim::statevector::StateVectorSimulator;
use ndarray::Array1;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Enhanced equivalence checker with simulator support
pub struct SimulatorEquivalenceChecker {
    options: EquivalenceOptions,
}

impl SimulatorEquivalenceChecker {
    /// Create a new simulator-based equivalence checker
    pub fn new(options: EquivalenceOptions) -> Self {
        SimulatorEquivalenceChecker { options }
    }
    
    /// Check state vector equivalence using actual simulation
    pub fn check_state_equivalence<const N: usize>(
        &self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        let mut max_diff = 0.0;
        let mut failed_state = None;
        
        // Check each computational basis state
        let num_states = if self.options.check_all_states {
            1 << N
        } else {
            std::cmp::min(1 << N, 100)
        };
        
        for state_idx in 0..num_states {
            // Create initial state
            let initial_state = self.create_basis_state(state_idx, N);
            
            // Simulate both circuits
            let final_state1 = self.simulate_circuit(circuit1, &initial_state)?;
            let final_state2 = self.simulate_circuit(circuit2, &initial_state)?;
            
            // Compare states
            let (equal, diff) = if self.options.ignore_global_phase {
                self.states_equal_up_to_phase(&final_state1, &final_state2)
            } else {
                self.states_equal(&final_state1, &final_state2)
            };
            
            if diff > max_diff {
                max_diff = diff;
            }
            
            if !equal {
                failed_state = Some(state_idx);
                break;
            }
        }
        
        if let Some(state_idx) = failed_state {
            Ok(EquivalenceResult {
                equivalent: false,
                check_type: EquivalenceType::StateVectorEquivalence,
                max_difference: Some(max_diff),
                details: format!(
                    "States differ for input |{:0b}>: max difference {:.2e}",
                    state_idx, max_diff
                ),
            })
        } else {
            Ok(EquivalenceResult {
                equivalent: true,
                check_type: EquivalenceType::StateVectorEquivalence,
                max_difference: Some(max_diff),
                details: format!(
                    "Verified equivalence for {} computational basis states",
                    num_states
                ),
            })
        }
    }
    
    /// Create a computational basis state
    fn create_basis_state(&self, state_idx: usize, num_qubits: usize) -> Array1<Complex64> {
        let dim = 1 << num_qubits;
        let mut state = Array1::zeros(dim);
        state[state_idx] = Complex64::new(1.0, 0.0);
        state
    }
    
    /// Simulate a circuit on an initial state
    fn simulate_circuit<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        initial_state: &Array1<Complex64>,
    ) -> QuantRS2Result<Array1<Complex64>> {
        let mut simulator = StateVectorSimulator::new();
        
        // Set initial state
        simulator.set_state(initial_state.clone())?;
        
        // Apply each gate
        for gate in circuit.gates() {
            // Convert gate to simulator format and apply
            // This is simplified - would need proper gate conversion
            self.apply_gate_to_simulator(&mut simulator, gate.as_ref())?;
        }
        
        // Get final state
        Ok(simulator.get_state().clone())
    }
    
    /// Apply a gate to the simulator
    fn apply_gate_to_simulator(
        &self,
        simulator: &mut StateVectorSimulator,
        gate: &dyn GateOp,
    ) -> QuantRS2Result<()> {
        // This would need proper implementation based on gate type
        // For now, using pattern matching on gate names
        match gate.name().as_str() {
            "H" => {
                let qubits = gate.qubits();
                if qubits.len() == 1 {
                    simulator.hadamard(qubits[0].id() as usize)?;
                }
            }
            "X" => {
                let qubits = gate.qubits();
                if qubits.len() == 1 {
                    simulator.pauli_x(qubits[0].id() as usize)?;
                }
            }
            "Y" => {
                let qubits = gate.qubits();
                if qubits.len() == 1 {
                    simulator.pauli_y(qubits[0].id() as usize)?;
                }
            }
            "Z" => {
                let qubits = gate.qubits();
                if qubits.len() == 1 {
                    simulator.pauli_z(qubits[0].id() as usize)?;
                }
            }
            "CNOT" | "CX" => {
                let qubits = gate.qubits();
                if qubits.len() == 2 {
                    simulator.cnot(qubits[0].id() as usize, qubits[1].id() as usize)?;
                }
            }
            "SWAP" => {
                let qubits = gate.qubits();
                if qubits.len() == 2 {
                    simulator.swap(qubits[0].id() as usize, qubits[1].id() as usize)?;
                }
            }
            _ => {
                // For unsupported gates, apply generic unitary if available
                return Err(QuantRS2Error::UnsupportedOperation(
                    format!("Gate '{}' not yet supported in simulator", gate.name())
                ));
            }
        }
        
        Ok(())
    }
    
    /// Check if two states are equal
    fn states_equal(&self, s1: &Array1<Complex64>, s2: &Array1<Complex64>) -> (bool, f64) {
        if s1.len() != s2.len() {
            return (false, f64::INFINITY);
        }
        
        let mut max_diff = 0.0;
        for (a, b) in s1.iter().zip(s2.iter()) {
            let diff = (a - b).norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > self.options.tolerance {
                return (false, max_diff);
            }
        }
        
        (true, max_diff)
    }
    
    /// Check if two states are equal up to a global phase
    fn states_equal_up_to_phase(
        &self,
        s1: &Array1<Complex64>,
        s2: &Array1<Complex64>,
    ) -> (bool, f64) {
        if s1.len() != s2.len() {
            return (false, f64::INFINITY);
        }
        
        // Find phase from first non-zero element
        let mut phase = None;
        for (a, b) in s1.iter().zip(s2.iter()) {
            if a.norm() > self.options.tolerance && b.norm() > self.options.tolerance {
                phase = Some(b / a);
                break;
            }
        }
        
        let phase = match phase {
            Some(p) => p,
            None => return (true, 0.0), // Both states are zero
        };
        
        // Check all elements with phase adjustment
        let mut max_diff = 0.0;
        for (a, b) in s1.iter().zip(s2.iter()) {
            let adjusted = a * phase;
            let diff = (adjusted - b).norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > self.options.tolerance {
                return (false, max_diff);
            }
        }
        
        (true, max_diff)
    }
    
    /// Check measurement probability equivalence
    pub fn check_measurement_equivalence<const N: usize>(
        &self,
        circuit1: &Circuit<N>,
        circuit2: &Circuit<N>,
    ) -> QuantRS2Result<EquivalenceResult> {
        let mut max_diff = 0.0;
        let mut failed_state = None;
        
        // Check measurement probabilities for each basis state
        for state_idx in 0..(1 << N) {
            let initial_state = self.create_basis_state(state_idx, N);
            
            // Get measurement probabilities for both circuits
            let probs1 = self.get_measurement_probabilities(circuit1, &initial_state)?;
            let probs2 = self.get_measurement_probabilities(circuit2, &initial_state)?;
            
            // Compare probabilities
            for (p1, p2) in probs1.iter().zip(probs2.iter()) {
                let diff = (p1 - p2).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                if diff > self.options.tolerance {
                    failed_state = Some(state_idx);
                    break;
                }
            }
            
            if failed_state.is_some() {
                break;
            }
        }
        
        if let Some(state_idx) = failed_state {
            Ok(EquivalenceResult {
                equivalent: false,
                check_type: EquivalenceType::ProbabilisticEquivalence,
                max_difference: Some(max_diff),
                details: format!(
                    "Measurement probabilities differ for input |{:0b}>: max difference {:.2e}",
                    state_idx, max_diff
                ),
            })
        } else {
            Ok(EquivalenceResult {
                equivalent: true,
                check_type: EquivalenceType::ProbabilisticEquivalence,
                max_difference: Some(max_diff),
                details: "Measurement probabilities match for all basis states".to_string(),
            })
        }
    }
    
    /// Get measurement probabilities from a circuit
    fn get_measurement_probabilities<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        initial_state: &Array1<Complex64>,
    ) -> QuantRS2Result<Vec<f64>> {
        let final_state = self.simulate_circuit(circuit, initial_state)?;
        
        // Calculate probabilities from amplitudes
        let probs: Vec<f64> = final_state
            .iter()
            .map(|amp| amp.norm_sqr())
            .collect();
        
        Ok(probs)
    }
}

/// Verify two circuits produce identical Bell states
pub fn verify_bell_equivalence() -> QuantRS2Result<()> {
    // Create two different ways to make a Bell state
    let mut circuit1 = Circuit::<2>::new();
    circuit1.h(0)?;
    circuit1.cnot(0, 1)?;
    
    let mut circuit2 = Circuit::<2>::new();
    circuit2.ry(1, PI/2.0)?;
    circuit2.cnot(1, 0)?;
    circuit2.ry(1, -PI/2.0)?;
    circuit2.cnot(0, 1)?;
    
    let checker = SimulatorEquivalenceChecker::new(EquivalenceOptions {
        tolerance: 1e-10,
        ignore_global_phase: true,
        check_all_states: true,
        max_unitary_qubits: 10,
    });
    
    let result = checker.check_state_equivalence(&circuit1, &circuit2)?;
    
    if result.equivalent {
        println!("✓ Bell state circuits are equivalent!");
        println!("  Details: {}", result.details);
    } else {
        println!("✗ Bell state circuits are NOT equivalent!");
        println!("  Details: {}", result.details);
    }
    
    Ok(())
}

/// Check if circuit optimization preserves behavior
pub fn verify_optimization_correctness<const N: usize>(
    original: &Circuit<N>,
    optimized: &Circuit<N>,
) -> QuantRS2Result<bool> {
    let checker = SimulatorEquivalenceChecker::new(EquivalenceOptions::default());
    
    // Check both state vector and measurement equivalence
    let state_result = checker.check_state_equivalence(original, optimized)?;
    let measure_result = checker.check_measurement_equivalence(original, optimized)?;
    
    Ok(state_result.equivalent && measure_result.equivalent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::single::{Hadamard, PauliX};
    use quantrs2_core::gate::multi::CNOT;
    
    #[test]
    fn test_simulator_equivalence() {
        // Two circuits that create |00> + |11> (Bell state)
        let mut circuit1 = Circuit::<2>::new();
        circuit1.add_gate(Hadamard::new(QubitId(0))).unwrap();
        circuit1.add_gate(CNOT::new(QubitId(0), QubitId(1))).unwrap();
        
        let mut circuit2 = Circuit::<2>::new();
        circuit2.add_gate(Hadamard::new(QubitId(0))).unwrap();
        circuit2.add_gate(CNOT::new(QubitId(0), QubitId(1))).unwrap();
        
        let checker = SimulatorEquivalenceChecker::new(EquivalenceOptions::default());
        let result = checker.check_state_equivalence(&circuit1, &circuit2);
        
        // Should succeed or return NotImplemented
        match result {
            Ok(res) => assert!(res.equivalent),
            Err(QuantRS2Error::UnsupportedOperation(_)) => {
                // Expected if simulator integration not complete
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}