//! # QuantumContinuousFlow - apply_quantum_coupling_network_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};

use super::types::{
    ClassicalFlowLayer, ClassicalFlowLayerType, CouplingNetworkOutput, FlowActivation, LayerOutput,
    MeasurementOutput, MeasurementStrategy, QuantumCouplingNetwork, QuantumCouplingType,
    QuantumFlowLayer, QuantumFlowNetworkLayer, QuantumFlowState, QuantumLayerState,
};

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Apply quantum coupling network
    pub(super) fn apply_quantum_coupling_network(
        &self,
        network: &QuantumCouplingNetwork,
        x: &Array1<f64>,
    ) -> Result<CouplingNetworkOutput> {
        let mut quantum_state = self.classical_to_quantum_encoding(x)?;
        for layer in &network.quantum_layers {
            quantum_state = self.apply_quantum_flow_layer(layer, &quantum_state)?;
        }
        let measurement_results = self.measure_quantum_state(&quantum_state)?;
        let mut classical_output = x.clone();
        for layer in &network.classical_layers {
            classical_output = self.apply_classical_flow_layer(layer, &classical_output)?;
        }
        let scale_params = &measurement_results.expectation_values * 0.5 + &classical_output * 0.5;
        let translation_params =
            &measurement_results.variance_measures * 0.3 + &classical_output * 0.7;
        Ok(CouplingNetworkOutput {
            scale_params,
            translation_params,
            entanglement_factor: measurement_results.entanglement_measure,
            quantum_phase: measurement_results.average_phase,
            quantum_state: QuantumLayerState {
                quantum_fidelity: quantum_state.fidelity,
                entanglement_measure: measurement_results.entanglement_measure,
                coherence_time: quantum_state.coherence_time,
                quantum_volume: self.config.num_qubits as f64,
            },
        })
    }
    /// Apply quantum flow layer to quantum state
    pub(super) fn apply_quantum_flow_layer(
        &self,
        layer: &QuantumFlowNetworkLayer,
        state: &QuantumFlowState,
    ) -> Result<QuantumFlowState> {
        let mut new_state = state.clone();
        for gate in &layer.quantum_gates {
            new_state = self.apply_quantum_flow_gate(gate, &new_state)?;
        }
        match &layer.measurement_strategy {
            MeasurementStrategy::ExpectationValue { observables } => {
                for observable in observables {
                    let expectation = self.compute_expectation_value(observable, &new_state)?;
                    new_state.fidelity *= (1.0 + expectation * 0.1);
                }
            }
            _ => {
                new_state.fidelity *= 0.99;
            }
        }
        Ok(new_state)
    }
    /// Measure quantum state
    fn measure_quantum_state(&self, state: &QuantumFlowState) -> Result<MeasurementOutput> {
        let expectation_values = state.amplitudes.mapv(|amp| amp.norm_sqr());
        let variance_measures = state
            .amplitudes
            .mapv(|amp| amp.norm_sqr() * (1.0 - amp.norm_sqr()));
        let average_phase = state.phases.iter().sum::<Complex64>() / state.phases.len() as f64;
        Ok(MeasurementOutput {
            expectation_values,
            variance_measures,
            entanglement_measure: state.entanglement_measure,
            average_phase,
        })
    }
    /// Apply classical flow layer
    pub(super) fn apply_classical_flow_layer(
        &self,
        layer: &ClassicalFlowLayer,
        x: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        match &layer.layer_type {
            ClassicalFlowLayerType::Dense {
                input_dim,
                output_dim,
            } => {
                if x.len() != *input_dim {
                    return Err(MLError::ModelCreationError(format!(
                        "Input dimension mismatch: expected {}, got {}",
                        input_dim,
                        x.len()
                    )));
                }
                let output = layer.parameters.dot(x);
                let activated_output = match layer.activation {
                    FlowActivation::ReLU => output.mapv(|x| x.max(0.0)),
                    FlowActivation::Swish => output.mapv(|x| x / (1.0 + (-x).exp())),
                    FlowActivation::GELU => output.mapv(|x| {
                        0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
                    }),
                    FlowActivation::Tanh => output.mapv(|x| x.tanh()),
                    _ => output,
                };
                Ok(activated_output)
            }
            _ => Ok(x.clone()),
        }
    }
    /// True inverse of `apply_quantum_coupling_layer`:
    /// uses the *same* `coupling_network` that the forward pass used.
    pub(super) fn apply_inverse_coupling_layer(
        &self,
        layer: &QuantumFlowLayer,
        z: &Array1<f64>,
        split_dim: usize,
        coupling_type: &QuantumCouplingType,
    ) -> Result<LayerOutput> {
        let z1 = z.slice(scirs2_core::ndarray::s![..split_dim]).to_owned();
        let z2 = z.slice(scirs2_core::ndarray::s![split_dim..]).to_owned();
        let coupling_output = self.apply_quantum_coupling_network(&layer.coupling_network, &z1)?;
        let (x2, log_jacobian) = match coupling_type {
            QuantumCouplingType::AffineCoupling => {
                let scale = &coupling_output.scale_params;
                let translation = &coupling_output.translation_params;
                let safe_scale = scale.mapv(|s| {
                    if s.abs() < 1e-8 {
                        1e-8_f64.copysign(s)
                    } else {
                        s
                    }
                });
                let x2 = (&z2 - translation) / &safe_scale;
                let log_jac = -safe_scale.mapv(|s| s.ln()).sum();
                (x2, log_jac)
            }
            QuantumCouplingType::QuantumEntangledCoupling => {
                let entanglement_factor = coupling_output.entanglement_factor;
                let quantum_phase = coupling_output.quantum_phase;
                let safe_ef = if entanglement_factor.abs() < 1e-8 {
                    1e-8_f64.copysign(entanglement_factor)
                } else {
                    entanglement_factor
                };
                let mut x2 = z2.clone();
                for i in 0..x2.len() {
                    x2[i] = (x2[i] - quantum_phase.re * 0.1) / safe_ef;
                }
                let log_jac = -(x2.len() as f64 * safe_ef.ln());
                (x2, log_jac)
            }
            _ => (z2.clone(), 0.0),
        };
        let mut x = Array1::zeros(z.len());
        x.slice_mut(scirs2_core::ndarray::s![..split_dim])
            .assign(&z1);
        x.slice_mut(scirs2_core::ndarray::s![split_dim..])
            .assign(&x2);
        Ok(LayerOutput {
            transformed_data: x,
            log_jacobian_det: log_jacobian,
            quantum_state: coupling_output.quantum_state,
            entanglement_measure: coupling_output.entanglement_factor,
        })
    }
}
