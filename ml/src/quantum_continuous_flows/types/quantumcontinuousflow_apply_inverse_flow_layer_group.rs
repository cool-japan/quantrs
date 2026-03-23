//! # QuantumContinuousFlow - apply_inverse_flow_layer_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{
    FlowLayerType, InvertibleTransform, LayerOutput, QuantumFlowLayer, QuantumLayerState,
};

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Apply inverse flow layer
    ///
    /// Dispatches based on `layer.layer_type` (the same discriminant used by
    /// the forward pass) so that the inverse truly undoes the forward
    /// transformation rather than running a structurally-different network.
    pub(super) fn apply_inverse_flow_layer(
        &self,
        layer: &QuantumFlowLayer,
        z: &Array1<f64>,
    ) -> Result<LayerOutput> {
        match &layer.layer_type {
            FlowLayerType::QuantumCouplingLayer {
                coupling_type,
                split_dimension,
            } => self.apply_inverse_coupling_layer(layer, z, *split_dimension, coupling_type),
            FlowLayerType::QuantumNeuralODE { .. }
            | FlowLayerType::QuantumAffineCoupling { .. } => Ok(LayerOutput {
                transformed_data: z.clone(),
                log_jacobian_det: 0.0,
                quantum_state: QuantumLayerState::default(),
                entanglement_measure: 0.5,
            }),
            _ => match &layer.invertible_component.inverse_transform {
                InvertibleTransform::QuantumCouplingTransform {
                    coupling_function,
                    mask,
                } => self.apply_inverse_quantum_coupling(layer, z, coupling_function, mask),
                _ => Ok(LayerOutput {
                    transformed_data: z.clone(),
                    log_jacobian_det: 0.0,
                    quantum_state: QuantumLayerState::default(),
                    entanglement_measure: 0.5,
                }),
            },
        }
    }
}
