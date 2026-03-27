//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{
    EntanglementCoupling, FlowOptimizationState, FlowTrainingMetrics, InvertibilityTracker,
    QuantumBaseDistribution, QuantumContinuousFlowConfig, QuantumFlowLayer, QuantumFlowMetrics,
    QuantumTransformation,
};

/// Main Quantum Continuous Normalization Flow model
pub struct QuantumContinuousFlow {
    pub(super) config: QuantumContinuousFlowConfig,
    pub(super) flow_layers: Vec<QuantumFlowLayer>,
    pub(super) base_distribution: QuantumBaseDistribution,
    pub(super) quantum_transformations: Vec<QuantumTransformation>,
    pub(super) entanglement_couplings: Vec<EntanglementCoupling>,
    pub(super) training_history: Vec<FlowTrainingMetrics>,
    pub(super) quantum_flow_metrics: QuantumFlowMetrics,
    pub(super) optimization_state: FlowOptimizationState,
    pub(super) invertibility_tracker: InvertibilityTracker,
}
