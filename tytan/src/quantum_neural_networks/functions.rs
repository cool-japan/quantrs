//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::sampler::SamplerError;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;

use super::types::{
    EarlyStoppingConfig, EntanglementPattern, GradientEstimationMethod, LossFunction,
    MeasurementScheme, OptimizerType, PauliBasis, PostprocessingScheme, QNNArchitecture,
    QNNTrainingConfig, QuantumNeuralNetwork, QuantumNoiseConfig, RegularizationConfig,
};

/// Create a default QNN for binary classification
pub fn create_binary_classification_qnn(
    num_qubits: usize,
) -> Result<QuantumNeuralNetwork, SamplerError> {
    let architecture = QNNArchitecture {
        input_dim: num_qubits,
        output_dim: 1,
        num_qubits,
        circuit_depth: 3,
        entanglement_pattern: EntanglementPattern::Linear,
        measurement_scheme: MeasurementScheme::Computational,
        postprocessing: PostprocessingScheme::Linear,
    };
    let training_config = QNNTrainingConfig {
        learning_rate: 0.01,
        batch_size: 32,
        num_epochs: 100,
        optimizer: OptimizerType::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        loss_function: LossFunction::MeanSquaredError,
        regularization: RegularizationConfig {
            l1_strength: 0.0,
            l2_strength: 0.001,
            dropout_prob: 0.0,
            parameter_noise: 0.0,
            quantum_noise: QuantumNoiseConfig {
                enable_noise: false,
                depolarizing_strength: 0.0,
                amplitude_damping: 0.0,
                phase_damping: 0.0,
                gate_error_rates: HashMap::new(),
            },
        },
        early_stopping: EarlyStoppingConfig {
            enabled: true,
            patience: 10,
            min_improvement: 1e-4,
            monitor_metric: "validation_loss".to_string(),
        },
        gradient_estimation: GradientEstimationMethod::ParameterShift,
    };
    QuantumNeuralNetwork::new(architecture, training_config)
}
/// Create a QNN for optimization problems
pub fn create_optimization_qnn(problem_size: usize) -> Result<QuantumNeuralNetwork, SamplerError> {
    let num_qubits = (problem_size as f64).log2().ceil() as usize;
    let architecture = QNNArchitecture {
        input_dim: problem_size,
        output_dim: problem_size,
        num_qubits,
        circuit_depth: 5,
        entanglement_pattern: EntanglementPattern::HardwareEfficient,
        measurement_scheme: MeasurementScheme::Pauli {
            bases: vec![PauliBasis::Z, PauliBasis::X, PauliBasis::Y],
        },
        postprocessing: PostprocessingScheme::NonlinearNN {
            hidden_dims: vec![64, 32],
        },
    };
    let training_config = QNNTrainingConfig {
        learning_rate: 0.005,
        batch_size: 16,
        num_epochs: 200,
        optimizer: OptimizerType::QuantumNaturalGradient,
        loss_function: LossFunction::ExpectationValueLoss,
        regularization: RegularizationConfig {
            l1_strength: 0.001,
            l2_strength: 0.01,
            dropout_prob: 0.1,
            parameter_noise: 0.01,
            quantum_noise: QuantumNoiseConfig {
                enable_noise: true,
                depolarizing_strength: 0.01,
                amplitude_damping: 0.001,
                phase_damping: 0.001,
                gate_error_rates: HashMap::new(),
            },
        },
        early_stopping: EarlyStoppingConfig {
            enabled: true,
            patience: 20,
            min_improvement: 1e-5,
            monitor_metric: "validation_loss".to_string(),
        },
        gradient_estimation: GradientEstimationMethod::QuantumFisherInformation,
    };
    QuantumNeuralNetwork::new(architecture, training_config)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_qnn_creation() {
        let qnn = create_binary_classification_qnn(4)
            .expect("Failed to create binary classification QNN with 4 qubits");
        assert_eq!(qnn.architecture.num_qubits, 4);
        assert_eq!(qnn.architecture.circuit_depth, 3);
        assert_eq!(qnn.layers.len(), 3);
    }
    #[test]
    fn test_qnn_forward_pass() {
        let qnn = create_binary_classification_qnn(2)
            .expect("Failed to create binary classification QNN with 2 qubits");
        let input = Array1::from_vec(vec![0.5, 0.7]);
        let output = qnn.forward(&input);
        assert!(output.is_ok());
    }
    #[test]
    fn test_optimization_qnn_creation() {
        let qnn = create_optimization_qnn(8)
            .expect("Failed to create optimization QNN with problem size 8");
        assert_eq!(qnn.architecture.input_dim, 8);
        assert_eq!(qnn.architecture.output_dim, 8);
        assert!(qnn.architecture.num_qubits >= 3);
    }
    #[test]
    fn test_parameter_initialization() {
        let mut qnn = create_binary_classification_qnn(3)
            .expect("Failed to create binary classification QNN with 3 qubits");
        qnn.initialize_parameters()
            .expect("Failed to initialize QNN parameters");
        for &param in &qnn.parameters.quantum_params {
            assert!((-PI..=PI).contains(&param));
        }
    }
    #[test]
    fn test_quantum_gate_application() {
        let qnn = create_binary_classification_qnn(2)
            .expect("Failed to create binary classification QNN with 2 qubits");
        let state = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let new_state = qnn
            .apply_rx_gate(&state, 0, PI / 2.0)
            .expect("Failed to apply RX gate");
        assert!((new_state[0] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
        assert!((new_state[2] - 1.0 / 2.0_f64.sqrt()).abs() < 1e-10);
    }
}
