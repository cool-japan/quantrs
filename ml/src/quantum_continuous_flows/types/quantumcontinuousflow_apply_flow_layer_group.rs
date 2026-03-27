//! # QuantumContinuousFlow - apply_flow_layer_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{
    FlowLayerType, LayerOutput, QuantumCouplingType, QuantumFlowLayer, QuantumLayerState,
    QuantumNetwork, QuantumODEFunction,
};

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Apply single flow layer
    pub(super) fn apply_flow_layer(
        &self,
        layer: &QuantumFlowLayer,
        x: &Array1<f64>,
        layer_idx: usize,
    ) -> Result<LayerOutput> {
        match &layer.layer_type {
            FlowLayerType::QuantumCouplingLayer {
                coupling_type,
                split_dimension,
            } => self.apply_quantum_coupling_layer(layer, x, *split_dimension, coupling_type),
            FlowLayerType::QuantumNeuralODE {
                ode_func,
                integration_time,
            } => self.apply_quantum_neural_ode_layer(layer, x, ode_func, *integration_time),
            FlowLayerType::QuantumAffineCoupling {
                scale_network,
                translation_network,
            } => self.apply_quantum_affine_coupling(layer, x, scale_network, translation_network),
            _ => Ok(LayerOutput {
                transformed_data: x.clone(),
                log_jacobian_det: 0.0,
                quantum_state: QuantumLayerState::default(),
                entanglement_measure: 0.5,
            }),
        }
    }
    /// Apply quantum coupling layer
    fn apply_quantum_coupling_layer(
        &self,
        layer: &QuantumFlowLayer,
        x: &Array1<f64>,
        split_dim: usize,
        coupling_type: &QuantumCouplingType,
    ) -> Result<LayerOutput> {
        let x1 = x.slice(scirs2_core::ndarray::s![..split_dim]).to_owned();
        let x2 = x.slice(scirs2_core::ndarray::s![split_dim..]).to_owned();
        let coupling_output = self.apply_quantum_coupling_network(&layer.coupling_network, &x1)?;
        let (z2, log_jacobian) = match coupling_type {
            QuantumCouplingType::AffineCoupling => {
                let scale = &coupling_output.scale_params;
                let translation = &coupling_output.translation_params;
                let z2 = &x2 * scale + translation;
                let log_jac = scale.mapv(|s| s.ln()).sum();
                (z2, log_jac)
            }
            QuantumCouplingType::QuantumEntangledCoupling => {
                let entanglement_factor = coupling_output.entanglement_factor;
                let quantum_phase = coupling_output.quantum_phase;
                let mut z2 = x2.clone();
                for i in 0..z2.len() {
                    z2[i] = z2[i] * entanglement_factor + quantum_phase.re * 0.1;
                }
                let log_jac = z2.len() as f64 * entanglement_factor.ln();
                (z2, log_jac)
            }
            _ => (x2.clone(), 0.0),
        };
        let mut z = Array1::zeros(x.len());
        z.slice_mut(scirs2_core::ndarray::s![..split_dim])
            .assign(&x1);
        z.slice_mut(scirs2_core::ndarray::s![split_dim..])
            .assign(&z2);
        Ok(LayerOutput {
            transformed_data: z,
            log_jacobian_det: log_jacobian,
            quantum_state: coupling_output.quantum_state,
            entanglement_measure: coupling_output.entanglement_factor,
        })
    }
    /// Apply quantum Neural ODE layer
    fn apply_quantum_neural_ode_layer(
        &self,
        layer: &QuantumFlowLayer,
        x: &Array1<f64>,
        ode_func: &QuantumODEFunction,
        integration_time: f64,
    ) -> Result<LayerOutput> {
        let mut quantum_state = self.classical_to_quantum_encoding(x)?;
        let integrated_state =
            self.integrate_quantum_ode(&quantum_state, ode_func, integration_time)?;
        let output_data = integrated_state.amplitudes.mapv(|amp| amp.re);
        let log_jacobian_det =
            self.compute_quantum_ode_jacobian(&integrated_state, integration_time)?;
        Ok(LayerOutput {
            transformed_data: output_data,
            log_jacobian_det,
            quantum_state: QuantumLayerState {
                quantum_fidelity: integrated_state.fidelity,
                entanglement_measure: integrated_state.entanglement_measure,
                coherence_time: integrated_state.coherence_time,
                quantum_volume: self.config.num_qubits as f64,
            },
            entanglement_measure: integrated_state.entanglement_measure,
        })
    }
    /// Apply quantum affine coupling
    fn apply_quantum_affine_coupling(
        &self,
        layer: &QuantumFlowLayer,
        x: &Array1<f64>,
        scale_network: &QuantumNetwork,
        translation_network: &QuantumNetwork,
    ) -> Result<LayerOutput> {
        let split_dim = x.len() / 2;
        let x1 = x.slice(scirs2_core::ndarray::s![..split_dim]).to_owned();
        let x2 = x.slice(scirs2_core::ndarray::s![split_dim..]).to_owned();
        let scale_output = self.apply_quantum_network(scale_network, &x1)?;
        let translation_output = self.apply_quantum_network(translation_network, &x1)?;
        let z2 = &x2 * &scale_output.output + &translation_output.output;
        let log_jacobian = scale_output.output.mapv(|s| s.ln()).sum();
        let mut z = Array1::zeros(x.len());
        z.slice_mut(scirs2_core::ndarray::s![..split_dim])
            .assign(&x1);
        z.slice_mut(scirs2_core::ndarray::s![split_dim..])
            .assign(&z2);
        Ok(LayerOutput {
            transformed_data: z,
            log_jacobian_det: log_jacobian,
            quantum_state: scale_output.quantum_state,
            entanglement_measure: scale_output.entanglement_measure,
        })
    }
}
