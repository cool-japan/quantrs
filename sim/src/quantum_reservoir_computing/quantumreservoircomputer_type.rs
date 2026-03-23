//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{
    CircuitInterface, InterfaceCircuit, InterfaceGate, InterfaceGateType,
};
use crate::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, VecDeque};

use super::types::{
    QuantumReservoirConfig, QuantumReservoirState, ReservoirMetrics, TrainingExample,
};

/// Quantum reservoir computing system
pub struct QuantumReservoirComputer {
    /// Configuration
    pub(super) config: QuantumReservoirConfig,
    /// Current reservoir state
    pub(super) reservoir_state: QuantumReservoirState,
    /// Reservoir circuit
    pub(super) reservoir_circuit: InterfaceCircuit,
    /// Input coupling circuit
    pub(super) input_coupling_circuit: InterfaceCircuit,
    /// Output weights (trainable)
    pub(super) output_weights: Array2<f64>,
    /// State vector simulator
    pub(super) simulator: StateVectorSimulator,
    /// Circuit interface
    pub(super) circuit_interface: CircuitInterface,
    /// Performance metrics
    pub(super) metrics: ReservoirMetrics,
    /// Training history
    pub(super) training_history: VecDeque<TrainingExample>,
}
