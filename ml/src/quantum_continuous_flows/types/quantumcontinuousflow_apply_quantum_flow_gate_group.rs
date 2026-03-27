//! # QuantumContinuousFlow - apply_quantum_flow_gate_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};

use super::types::{QuantumFlowGate, QuantumFlowGateType, QuantumFlowState};

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Apply quantum flow gate
    pub(super) fn apply_quantum_flow_gate(
        &self,
        gate: &QuantumFlowGate,
        state: &QuantumFlowState,
    ) -> Result<QuantumFlowState> {
        let mut new_state = state.clone();
        match &gate.gate_type {
            QuantumFlowGateType::ParameterizedRotation { axis } => {
                let angle = gate.parameters[0];
                for &target_qubit in &gate.target_qubits {
                    if target_qubit < new_state.amplitudes.len() {
                        let rotation_factor = Complex64::from_polar(1.0, angle);
                        new_state.amplitudes[target_qubit] *= rotation_factor;
                        new_state.phases[target_qubit] *= rotation_factor;
                    }
                }
            }
            QuantumFlowGateType::EntanglementGate { entanglement_type } => {
                if gate.target_qubits.len() >= 2 {
                    let control = gate.control_qubits[0];
                    let target = gate.target_qubits[0];
                    if control < new_state.amplitudes.len() && target < new_state.amplitudes.len() {
                        let entanglement_factor = 0.1;
                        let control_amplitude = new_state.amplitudes[control];
                        new_state.amplitudes[target] += entanglement_factor * control_amplitude;
                        new_state.entanglement_measure =
                            (new_state.entanglement_measure + 0.1).min(1.0);
                    }
                }
            }
            _ => {
                new_state.fidelity *= 0.99;
            }
        }
        Ok(new_state)
    }
}
