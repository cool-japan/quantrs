//! QML Types and Training State
//!
//! Supporting types for quantum machine learning framework.

use super::config::RotationGate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Training state for QML framework
#[derive(Debug, Clone)]
pub struct QMLTrainingState {
    pub current_epoch: usize,
    pub current_learning_rate: f64,
    pub best_validation_loss: f64,
    pub patience_counter: usize,
    pub training_loss_history: Vec<f64>,
    pub validation_loss_history: Vec<f64>,
}

impl Default for QMLTrainingState {
    fn default() -> Self {
        Self::new()
    }
}

impl QMLTrainingState {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            current_epoch: 0,
            current_learning_rate: 0.01,
            best_validation_loss: f64::INFINITY,
            patience_counter: 0,
            training_loss_history: Vec::new(),
            validation_loss_history: Vec::new(),
        }
    }
}

/// Training result for QML framework
#[derive(Debug, Clone)]
pub struct QMLTrainingResult {
    pub final_training_loss: f64,
    pub final_validation_loss: f64,
    pub best_validation_loss: f64,
    pub epochs_trained: usize,
    pub total_training_time: std::time::Duration,
    pub training_metrics: Vec<QMLEpochMetrics>,
    pub quantum_advantage_metrics: QuantumAdvantageMetrics,
}

/// Training metrics for a single epoch
#[derive(Debug, Clone)]
pub struct QMLEpochMetrics {
    pub epoch: usize,
    pub training_loss: f64,
    pub validation_loss: f64,
    pub epoch_time: std::time::Duration,
    pub learning_rate: f64,
}

/// Quantum advantage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAdvantageMetrics {
    pub quantum_volume: f64,
    pub classical_simulation_cost: f64,
    pub quantum_speedup_factor: f64,
    pub circuit_depth: usize,
    pub gate_count: usize,
    pub entanglement_measure: f64,
}

/// QML framework statistics
#[derive(Debug, Clone)]
pub struct QMLStats {
    pub forward_passes: usize,
    pub backward_passes: usize,
    pub total_training_time: std::time::Duration,
    pub average_epoch_time: std::time::Duration,
    pub peak_memory_usage: usize,
    pub num_parameters: usize,
}

impl Default for QMLStats {
    fn default() -> Self {
        Self::new()
    }
}

impl QMLStats {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            forward_passes: 0,
            backward_passes: 0,
            total_training_time: std::time::Duration::from_secs(0),
            average_epoch_time: std::time::Duration::from_secs(0),
            peak_memory_usage: 0,
            num_parameters: 0,
        }
    }
}

/// Parameterized quantum circuit gate
#[derive(Debug, Clone)]
pub struct PQCGate {
    pub gate_type: PQCGateType,
    pub qubits: Vec<usize>,
    pub parameter_index: Option<usize>,
}

/// Types of PQC gates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PQCGateType {
    SingleQubit(RotationGate),
    TwoQubit(TwoQubitGate),
}

/// Two-qubit gates
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoQubitGate {
    CNOT,
    CZ,
    SWAP,
    CPhase,
}

/// Convolutional filter structure
#[derive(Debug, Clone)]
pub struct ConvolutionalFilter {
    pub qubits: Vec<usize>,
    pub parameter_indices: Vec<usize>,
}

/// Dense layer connection
#[derive(Debug, Clone)]
pub struct DenseConnection {
    pub qubit1: usize,
    pub qubit2: usize,
    pub parameter_index: usize,
}

/// LSTM gate structure
#[derive(Debug, Clone)]
pub struct LSTMGate {
    pub gate_type: LSTMGateType,
    pub parameter_start: usize,
    pub parameter_count: usize,
}

/// LSTM gate types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LSTMGateType {
    Forget,
    Input,
    Output,
    Candidate,
}

/// Attention head structure
#[derive(Debug, Clone)]
pub struct AttentionHead {
    pub head_id: usize,
    pub parameter_start: usize,
    pub parameter_count: usize,
    pub query_qubits: Vec<usize>,
    pub key_qubits: Vec<usize>,
}

/// QML benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QMLBenchmarkResults {
    pub training_times: HashMap<String, std::time::Duration>,
    pub final_accuracies: HashMap<String, f64>,
    pub convergence_rates: HashMap<String, f64>,
    pub memory_usage: HashMap<String, usize>,
    pub quantum_advantage: HashMap<String, QuantumAdvantageMetrics>,
    pub parameter_counts: HashMap<String, usize>,
    pub circuit_depths: HashMap<String, usize>,
    pub gate_counts: HashMap<String, usize>,
}
