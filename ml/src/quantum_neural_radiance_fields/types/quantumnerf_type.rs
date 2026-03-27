//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{
    NeRFOptimizationState, NeRFTrainingMetrics, QuantumLightField, QuantumMLP, QuantumNeRFConfig,
    QuantumPositionalEncoder, QuantumRayMarcher, QuantumRenderingMetrics, QuantumScaleAttention,
    QuantumSceneRepresentation, QuantumSpatialAttention, QuantumViewAttention, QuantumViewEncoder,
    QuantumVolumeRenderer,
};

/// Main Quantum Neural Radiance Field model
pub struct QuantumNeRF {
    pub(super) config: QuantumNeRFConfig,
    pub(super) quantum_mlp_coarse: QuantumMLP,
    pub(super) quantum_mlp_fine: QuantumMLP,
    pub(super) quantum_positional_encoder: QuantumPositionalEncoder,
    pub(super) quantum_view_encoder: QuantumViewEncoder,
    pub(super) spatial_attention: QuantumSpatialAttention,
    pub(super) view_attention: QuantumViewAttention,
    pub(super) scale_attention: QuantumScaleAttention,
    pub(super) quantum_volume_renderer: QuantumVolumeRenderer,
    pub(super) quantum_ray_marcher: QuantumRayMarcher,
    pub(super) training_history: Vec<NeRFTrainingMetrics>,
    pub(super) quantum_rendering_metrics: QuantumRenderingMetrics,
    pub(super) optimization_state: NeRFOptimizationState,
    pub(super) quantum_scene_representation: QuantumSceneRepresentation,
    pub(super) quantum_light_field: QuantumLightField,
}
