//! # QuantumContinuousFlow - apply_quantum_network_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{
    CouplingFunction, LayerOutput, QuantumFlowLayer, QuantumLayerState, QuantumNetwork,
    QuantumNetworkOutput,
};

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Apply quantum network
    pub(super) fn apply_quantum_network(
        &self,
        network: &QuantumNetwork,
        x: &Array1<f64>,
    ) -> Result<QuantumNetworkOutput> {
        let quantum_state = self.classical_to_quantum_encoding(x)?;
        let mut processed_state = quantum_state;
        for layer in &network.layers {
            processed_state = self.apply_quantum_flow_layer(layer, &processed_state)?;
        }
        let full_output = processed_state
            .amplitudes
            .mapv(|amp| amp.re * network.quantum_enhancement);
        let output = if full_output.len() > x.len() {
            full_output
                .slice(scirs2_core::ndarray::s![..x.len()])
                .to_owned()
        } else {
            full_output
        };
        Ok(QuantumNetworkOutput {
            output,
            quantum_state: QuantumLayerState {
                quantum_fidelity: processed_state.fidelity,
                entanglement_measure: processed_state.entanglement_measure,
                coherence_time: processed_state.coherence_time,
                quantum_volume: network.layers.len() as f64,
            },
            entanglement_measure: processed_state.entanglement_measure,
        })
    }
    /// Apply inverse quantum coupling via the stored invertible_component
    /// (legacy path, used as fallback for non-standard layer types)
    pub(super) fn apply_inverse_quantum_coupling(
        &self,
        _layer: &QuantumFlowLayer,
        z: &Array1<f64>,
        coupling_function: &CouplingFunction,
        mask: &Array1<bool>,
    ) -> Result<LayerOutput> {
        let split_dim = mask.iter().filter(|&&m| m).count();
        let z1 = z.slice(scirs2_core::ndarray::s![..split_dim]).to_owned();
        let z2 = z.slice(scirs2_core::ndarray::s![split_dim..]).to_owned();
        let scale_output = self.apply_quantum_network(&coupling_function.scale_function, &z1)?;
        let translation_output =
            self.apply_quantum_network(&coupling_function.translation_function, &z1)?;
        let safe_scale = scale_output.output.mapv(|s| {
            if s.abs() < 1e-8 {
                1e-8_f64.copysign(s)
            } else {
                s
            }
        });
        let x2 = (&z2 - &translation_output.output) / &safe_scale;
        let log_jacobian = -safe_scale.mapv(|s| s.ln()).sum();
        let mut x = Array1::zeros(z.len());
        x.slice_mut(scirs2_core::ndarray::s![..split_dim])
            .assign(&z1);
        x.slice_mut(scirs2_core::ndarray::s![split_dim..])
            .assign(&x2);
        Ok(LayerOutput {
            transformed_data: x,
            log_jacobian_det: log_jacobian,
            quantum_state: scale_output.quantum_state,
            entanglement_measure: scale_output.entanglement_measure,
        })
    }
}
