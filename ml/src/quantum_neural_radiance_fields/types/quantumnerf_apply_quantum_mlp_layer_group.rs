//! # QuantumNeRF - apply_quantum_mlp_layer_group Methods
//!
//! This module contains method implementations for `QuantumNeRF`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};

use super::types::{
    MLPLayerOutput, QuantumActivationType, QuantumMLPGate, QuantumMLPGateType, QuantumMLPLayer,
    QuantumMLPState, QuantumNormalizationType,
};

use super::quantumnerf_type::QuantumNeRF;

impl QuantumNeRF {
    /// Apply quantum MLP layer
    pub(super) fn apply_quantum_mlp_layer(
        &self,
        layer: &QuantumMLPLayer,
        input: &Array1<f64>,
        quantum_state: &QuantumMLPState,
    ) -> Result<MLPLayerOutput> {
        let linear_output = if input.len() == layer.input_dim {
            Array1::ones(layer.output_dim) * input.sum() / input.len() as f64
        } else {
            Array1::ones(layer.output_dim) * 0.5
        };
        let mut updated_quantum_state = quantum_state.clone();
        for gate in &layer.quantum_gates {
            updated_quantum_state = self.apply_quantum_mlp_gate(gate, &updated_quantum_state)?;
        }
        let activated_output = match layer.activation {
            QuantumActivationType::QuantumReLU => linear_output.mapv(|x: f64| x.max(0.0)),
            QuantumActivationType::QuantumSigmoid => {
                linear_output.mapv(|x| 1.0 / (1.0 + (-x).exp()))
            }
            QuantumActivationType::QuantumSoftplus => {
                linear_output.mapv(|x: f64| (1.0f64 + x.exp()).ln())
            }
            QuantumActivationType::QuantumEntanglementActivation => {
                let entanglement_factor = updated_quantum_state.entanglement_measure;
                linear_output.mapv(|x| x * (1.0 + entanglement_factor))
            }
            _ => linear_output,
        };
        let normalized_output = if let Some(ref norm_type) = layer.normalization {
            self.apply_quantum_normalization(&activated_output, norm_type)?
        } else {
            activated_output
        };
        Ok(MLPLayerOutput {
            features: normalized_output,
            quantum_state: updated_quantum_state,
        })
    }
    /// Apply quantum MLP gate
    pub(super) fn apply_quantum_mlp_gate(
        &self,
        gate: &QuantumMLPGate,
        quantum_state: &QuantumMLPState,
    ) -> Result<QuantumMLPState> {
        let mut new_state = quantum_state.clone();
        match &gate.gate_type {
            QuantumMLPGateType::ParameterizedRotation { axis } => {
                let angle = gate.parameters[0];
                for &target_qubit in &gate.target_qubits {
                    if target_qubit < new_state.quantum_amplitudes.len() {
                        let rotation_factor = Complex64::from_polar(1.0, angle);
                        new_state.quantum_amplitudes[target_qubit] *= rotation_factor;
                    }
                }
            }
            QuantumMLPGateType::EntanglementGate { gate_name } => {
                if gate_name == "CNOT"
                    && gate.control_qubits.len() > 0
                    && gate.target_qubits.len() > 0
                {
                    let control = gate.control_qubits[0];
                    let target = gate.target_qubits[0];
                    if control < new_state.quantum_amplitudes.len()
                        && target < new_state.quantum_amplitudes.len()
                    {
                        let entanglement_factor = 0.1;
                        let control_amplitude = new_state.quantum_amplitudes[control];
                        new_state.quantum_amplitudes[target] +=
                            entanglement_factor * control_amplitude;
                        new_state.entanglement_measure =
                            (new_state.entanglement_measure + 0.1).min(1.0);
                    }
                }
            }
            _ => {
                new_state.quantum_fidelity *= 0.99;
            }
        }
        Ok(new_state)
    }
    /// Apply quantum normalization
    pub(super) fn apply_quantum_normalization(
        &self,
        input: &Array1<f64>,
        norm_type: &QuantumNormalizationType,
    ) -> Result<Array1<f64>> {
        match norm_type {
            QuantumNormalizationType::QuantumLayerNorm => {
                let mean = input.sum() / input.len() as f64;
                let variance =
                    input.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / input.len() as f64;
                let std_dev = (variance + 1e-8).sqrt();
                Ok(input.mapv(|x| (x - mean) / std_dev))
            }
            QuantumNormalizationType::EntanglementNorm => {
                let quantum_norm =
                    input.dot(input).sqrt() * (1.0 + self.config.quantum_enhancement_level);
                if quantum_norm > 1e-10 {
                    Ok(input / quantum_norm)
                } else {
                    Ok(input.clone())
                }
            }
            _ => Ok(input.clone()),
        }
    }
}
