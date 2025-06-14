//! Comprehensive tests for Quantum Neural Networks module

#[cfg(test)]
mod tests {
    use quantrs2_tytan::quantum_neural_networks::*;
    use quantrs2_tytan::sampler_framework::{SampleResult, SamplerError, SamplerResult};
    use quantrs2_tytan::compile::QuboFormulation;
    use quantrs2_tytan::symbol::Symbol;
    use ndarray::{Array1, Array2, Array3};
    use std::collections::HashMap;

    /// Test basic QNN architecture creation
    #[test]
    fn test_qnn_architecture_creation() {
        let architecture = QNNArchitecture {
            input_dim: 4,
            output_dim: 2,
            num_qubits: 8,
            circuit_depth: 3,
            entanglement_pattern: EntanglementPattern::Linear,
            measurement_scheme: MeasurementScheme::Computational,
            postprocessing: PostprocessingScheme::Linear,
        };

        assert_eq!(architecture.input_dim, 4);
        assert_eq!(architecture.output_dim, 2);
        assert_eq!(architecture.num_qubits, 8);
        assert_eq!(architecture.circuit_depth, 3);
        assert_eq!(architecture.entanglement_pattern, EntanglementPattern::Linear);
    }

    /// Test entanglement patterns
    #[test]
    fn test_entanglement_patterns() {
        let patterns = vec![
            EntanglementPattern::Linear,
            EntanglementPattern::Full,
            EntanglementPattern::Circular,
            EntanglementPattern::Random { connectivity: 0.5 },
            EntanglementPattern::HardwareEfficient,
            EntanglementPattern::ProblemAdapted,
        ];

        for pattern in patterns {
            match pattern {
                EntanglementPattern::Linear => assert!(true),
                EntanglementPattern::Full => assert!(true),
                EntanglementPattern::Circular => assert!(true),
                EntanglementPattern::Random { connectivity } => {
                    assert!(connectivity >= 0.0 && connectivity <= 1.0);
                }
                EntanglementPattern::HardwareEfficient => assert!(true),
                EntanglementPattern::ProblemAdapted => assert!(true),
            }
        }
    }

    /// Test measurement schemes
    #[test]
    fn test_measurement_schemes() {
        let computational = MeasurementScheme::Computational;
        let pauli = MeasurementScheme::Pauli {
            bases: vec![PauliBasis::X, PauliBasis::Y, PauliBasis::Z],
        };
        let shadow = MeasurementScheme::ShadowTomography { num_shadows: 100 };

        assert_eq!(computational, MeasurementScheme::Computational);
        
        if let MeasurementScheme::Pauli { bases } = pauli {
            assert_eq!(bases.len(), 3);
            assert!(bases.contains(&PauliBasis::X));
        }

        if let MeasurementScheme::ShadowTomography { num_shadows } = shadow {
            assert_eq!(num_shadows, 100);
        }
    }

    /// Test quantum layer creation
    #[test]
    fn test_quantum_layer_creation() {
        let layer = QuantumLayer {
            layer_id: 0,
            num_qubits: 4,
            gates: vec![
                QuantumGate::RX { qubit: 0, angle: std::f64::consts::PI / 4.0 },
                QuantumGate::CNOT { control: 0, target: 1 },
            ],
            parametrized_gates: vec![],
            layer_type: QuantumLayerType::Variational,
            skip_connections: vec![],
        };

        assert_eq!(layer.layer_id, 0);
        assert_eq!(layer.num_qubits, 4);
        assert_eq!(layer.gates.len(), 2);
        assert_eq!(layer.layer_type, QuantumLayerType::Variational);
    }

    /// Test quantum gates
    #[test]
    fn test_quantum_gates() {
        let rx_gate = QuantumGate::RX { qubit: 0, angle: std::f64::consts::PI / 2.0 };
        let cnot_gate = QuantumGate::CNOT { control: 0, target: 1 };
        let toffoli_gate = QuantumGate::Toffoli { controls: vec![0, 1], target: 2 };

        match rx_gate {
            QuantumGate::RX { qubit, angle } => {
                assert_eq!(qubit, 0);
                assert!((angle - std::f64::consts::PI / 2.0).abs() < 1e-10);
            }
            _ => panic!("Wrong gate type"),
        }

        match cnot_gate {
            QuantumGate::CNOT { control, target } => {
                assert_eq!(control, 0);
                assert_eq!(target, 1);
            }
            _ => panic!("Wrong gate type"),
        }

        match toffoli_gate {
            QuantumGate::Toffoli { controls, target } => {
                assert_eq!(controls, vec![0, 1]);
                assert_eq!(target, 2);
            }
            _ => panic!("Wrong gate type"),
        }
    }

    /// Test parametrized gates
    #[test]
    fn test_parametrized_gates() {
        let param_gate = ParametrizedGate {
            gate_type: ParametrizedGateType::Rotation { axis: RotationAxis::X },
            qubits: vec![0],
            parameter_indices: vec![0],
            gate_function: GateFunction::StandardRotation,
        };

        assert_eq!(param_gate.qubits, vec![0]);
        assert_eq!(param_gate.parameter_indices, vec![0]);
        
        match param_gate.gate_type {
            ParametrizedGateType::Rotation { axis } => {
                assert_eq!(axis, RotationAxis::X);
            }
            _ => panic!("Wrong parametrized gate type"),
        }
    }

    /// Test classical layer
    #[test]
    fn test_classical_layer() {
        let weights = Array2::ones((4, 2));
        let biases = Array1::zeros(2);

        let layer = ClassicalLayer {
            layer_type: ClassicalLayerType::Dense,
            input_dim: 4,
            output_dim: 2,
            weights,
            biases,
            activation: ActivationFunction::ReLU,
        };

        assert_eq!(layer.input_dim, 4);
        assert_eq!(layer.output_dim, 2);
        assert_eq!(layer.layer_type, ClassicalLayerType::Dense);
        assert_eq!(layer.activation, ActivationFunction::ReLU);
        assert_eq!(layer.weights.shape(), &[4, 2]);
        assert_eq!(layer.biases.len(), 2);
    }

    /// Test training configuration
    #[test]
    fn test_training_config() {
        let config = QNNTrainingConfig {
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
                dropout_prob: 0.1,
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

        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.num_epochs, 100);
        assert_eq!(config.loss_function, LossFunction::MeanSquaredError);
        assert_eq!(config.gradient_estimation, GradientEstimationMethod::ParameterShift);
    }

    /// Test optimizer types
    #[test]
    fn test_optimizer_types() {
        let adam = OptimizerType::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };

        let sgd = OptimizerType::SGD { momentum: 0.9 };

        match adam {
            OptimizerType::Adam { beta1, beta2, epsilon } => {
                assert_eq!(beta1, 0.9);
                assert_eq!(beta2, 0.999);
                assert_eq!(epsilon, 1e-8);
            }
            _ => panic!("Wrong optimizer type"),
        }

        match sgd {
            OptimizerType::SGD { momentum } => {
                assert_eq!(momentum, 0.9);
            }
            _ => panic!("Wrong optimizer type"),
        }
    }

    /// Test gradient estimation methods
    #[test]
    fn test_gradient_estimation() {
        let param_shift = GradientEstimationMethod::ParameterShift;
        let finite_diff = GradientEstimationMethod::FiniteDifferences { epsilon: 1e-6 };
        let natural_grad = GradientEstimationMethod::NaturalGradient;

        assert_eq!(param_shift, GradientEstimationMethod::ParameterShift);
        assert_eq!(natural_grad, GradientEstimationMethod::NaturalGradient);

        match finite_diff {
            GradientEstimationMethod::FiniteDifferences { epsilon } => {
                assert_eq!(epsilon, 1e-6);
            }
            _ => panic!("Wrong gradient estimation method"),
        }
    }

    /// Test parameter initialization schemes
    #[test]
    fn test_parameter_initialization() {
        let uniform = ParameterInitializationScheme::RandomUniform { min: -1.0, max: 1.0 };
        let normal = ParameterInitializationScheme::RandomNormal { mean: 0.0, std: 0.1 };
        let xavier = ParameterInitializationScheme::Xavier;

        match uniform {
            ParameterInitializationScheme::RandomUniform { min, max } => {
                assert_eq!(min, -1.0);
                assert_eq!(max, 1.0);
            }
            _ => panic!("Wrong initialization scheme"),
        }

        match normal {
            ParameterInitializationScheme::RandomNormal { mean, std } => {
                assert_eq!(mean, 0.0);
                assert_eq!(std, 0.1);
            }
            _ => panic!("Wrong initialization scheme"),
        }

        assert_eq!(xavier, ParameterInitializationScheme::Xavier);
    }

    /// Test QNN parameters structure
    #[test]
    fn test_qnn_parameters() {
        let quantum_params = Array1::zeros(10);
        let classical_params = vec![Array2::ones((4, 2)), Array2::ones((2, 1))];
        let bias_params = vec![Array1::zeros(2), Array1::zeros(1)];
        let parameter_bounds = vec![(-1.0, 1.0); 10];

        let params = QNNParameters {
            quantum_params,
            classical_params,
            bias_params,
            parameter_bounds,
            initialization_scheme: ParameterInitializationScheme::Xavier,
        };

        assert_eq!(params.quantum_params.len(), 10);
        assert_eq!(params.classical_params.len(), 2);
        assert_eq!(params.bias_params.len(), 2);
        assert_eq!(params.parameter_bounds.len(), 10);
        assert_eq!(params.initialization_scheme, ParameterInitializationScheme::Xavier);
    }

    /// Test training epoch data
    #[test]
    fn test_training_epoch() {
        let parameter_stats = ParameterStatistics {
            mean_values: Array1::zeros(10),
            std_values: Array1::ones(10),
            ranges: vec![(-1.0, 1.0); 10],
            correlations: Array2::eye(10),
        };

        let epoch = TrainingEpoch {
            epoch: 1,
            training_loss: 0.5,
            validation_loss: Some(0.6),
            training_accuracy: 0.8,
            validation_accuracy: Some(0.75),
            learning_rate: 0.01,
            training_time: 10.5,
            gradient_norms: vec![0.1, 0.2, 0.15],
            parameter_stats,
        };

        assert_eq!(epoch.epoch, 1);
        assert_eq!(epoch.training_loss, 0.5);
        assert_eq!(epoch.validation_loss, Some(0.6));
        assert_eq!(epoch.training_accuracy, 0.8);
        assert_eq!(epoch.validation_accuracy, Some(0.75));
        assert_eq!(epoch.learning_rate, 0.01);
        assert_eq!(epoch.gradient_norms.len(), 3);
    }

    /// Test quantum noise configuration
    #[test]
    fn test_quantum_noise_config() {
        let mut gate_errors = HashMap::new();
        gate_errors.insert("RX".to_string(), 0.001);
        gate_errors.insert("CNOT".to_string(), 0.005);

        let noise_config = QuantumNoiseConfig {
            enable_noise: true,
            depolarizing_strength: 0.01,
            amplitude_damping: 0.02,
            phase_damping: 0.015,
            gate_error_rates: gate_errors,
        };

        assert!(noise_config.enable_noise);
        assert_eq!(noise_config.depolarizing_strength, 0.01);
        assert_eq!(noise_config.amplitude_damping, 0.02);
        assert_eq!(noise_config.phase_damping, 0.015);
        assert_eq!(noise_config.gate_error_rates.len(), 2);
        assert_eq!(noise_config.gate_error_rates["RX"], 0.001);
        assert_eq!(noise_config.gate_error_rates["CNOT"], 0.005);
    }

    /// Test metrics structures
    #[test]
    fn test_qnn_metrics() {
        let training_metrics = TrainingMetrics {
            final_training_loss: 0.1,
            convergence_rate: 0.95,
            epochs_to_convergence: 50,
            training_stability: 0.9,
            overfitting_measure: 0.05,
        };

        let validation_metrics = ValidationMetrics {
            best_validation_loss: 0.15,
            validation_accuracy: 0.85,
            generalization_gap: 0.05,
            cv_scores: vec![0.8, 0.82, 0.85, 0.83, 0.84],
            confidence_intervals: vec![(0.75, 0.9), (0.78, 0.92)],
        };

        let quantum_metrics = QuantumMetrics {
            quantum_volume: 64.0,
            entanglement_measures: vec![0.5, 0.6, 0.7],
            quantum_advantage: 1.2,
            fidelity_measures: vec![0.95, 0.94, 0.96],
            coherence_utilization: 0.8,
        };

        let computational_metrics = ComputationalMetrics {
            training_time_per_epoch: 120.5,
            inference_time: 0.05,
            memory_usage: 1024.0,
            quantum_execution_time: 10.0,
            classical_computation_time: 110.5,
        };

        let metrics = QNNMetrics {
            training_metrics,
            validation_metrics,
            quantum_metrics,
            computational_metrics,
        };

        assert_eq!(metrics.training_metrics.final_training_loss, 0.1);
        assert_eq!(metrics.validation_metrics.best_validation_loss, 0.15);
        assert_eq!(metrics.quantum_metrics.quantum_volume, 64.0);
        assert_eq!(metrics.computational_metrics.training_time_per_epoch, 120.5);
    }

    /// Test activation functions
    #[test]
    fn test_activation_functions() {
        let activations = vec![
            ActivationFunction::ReLU,
            ActivationFunction::Sigmoid,
            ActivationFunction::Tanh,
            ActivationFunction::Linear,
            ActivationFunction::Softmax,
            ActivationFunction::Swish,
            ActivationFunction::GELU,
        ];

        for activation in activations {
            match activation {
                ActivationFunction::ReLU => assert!(true),
                ActivationFunction::Sigmoid => assert!(true),
                ActivationFunction::Tanh => assert!(true),
                ActivationFunction::Linear => assert!(true),
                ActivationFunction::Softmax => assert!(true),
                ActivationFunction::Swish => assert!(true),
                ActivationFunction::GELU => assert!(true),
            }
        }
    }

    /// Test postprocessing schemes
    #[test]
    fn test_postprocessing_schemes() {
        let schemes = vec![
            PostprocessingScheme::None,
            PostprocessingScheme::Linear,
            PostprocessingScheme::NonlinearNN { hidden_dims: vec![64, 32] },
            PostprocessingScheme::Attention,
            PostprocessingScheme::GraphNN,
        ];

        for scheme in schemes {
            match scheme {
                PostprocessingScheme::None => assert!(true),
                PostprocessingScheme::Linear => assert!(true),
                PostprocessingScheme::NonlinearNN { hidden_dims } => {
                    assert_eq!(hidden_dims, vec![64, 32]);
                }
                PostprocessingScheme::Attention => assert!(true),
                PostprocessingScheme::GraphNN => assert!(true),
            }
        }
    }

    /// Test early stopping configuration
    #[test]
    fn test_early_stopping_config() {
        let config = EarlyStoppingConfig {
            enabled: true,
            patience: 15,
            min_improvement: 1e-5,
            monitor_metric: "val_loss".to_string(),
        };

        assert!(config.enabled);
        assert_eq!(config.patience, 15);
        assert_eq!(config.min_improvement, 1e-5);
        assert_eq!(config.monitor_metric, "val_loss");
    }

    /// Test regularization configuration
    #[test]
    fn test_regularization_config() {
        let quantum_noise = QuantumNoiseConfig {
            enable_noise: true,
            depolarizing_strength: 0.01,
            amplitude_damping: 0.02,
            phase_damping: 0.015,
            gate_error_rates: HashMap::new(),
        };

        let reg_config = RegularizationConfig {
            l1_strength: 0.001,
            l2_strength: 0.01,
            dropout_prob: 0.2,
            parameter_noise: 0.05,
            quantum_noise,
        };

        assert_eq!(reg_config.l1_strength, 0.001);
        assert_eq!(reg_config.l2_strength, 0.01);
        assert_eq!(reg_config.dropout_prob, 0.2);
        assert_eq!(reg_config.parameter_noise, 0.05);
        assert!(reg_config.quantum_noise.enable_noise);
    }

    /// Test loss functions
    #[test]
    fn test_loss_functions() {
        let mse = LossFunction::MeanSquaredError;
        let cross_entropy = LossFunction::CrossEntropy;
        let huber = LossFunction::HuberLoss { delta: 1.0 };
        let qfi = LossFunction::QuantumFisherInformation;
        let custom = LossFunction::Custom { name: "CustomLoss".to_string() };

        assert_eq!(mse, LossFunction::MeanSquaredError);
        assert_eq!(cross_entropy, LossFunction::CrossEntropy);
        assert_eq!(qfi, LossFunction::QuantumFisherInformation);

        match huber {
            LossFunction::HuberLoss { delta } => assert_eq!(delta, 1.0),
            _ => panic!("Wrong loss function type"),
        }

        match custom {
            LossFunction::Custom { name } => assert_eq!(name, "CustomLoss"),
            _ => panic!("Wrong loss function type"),
        }
    }
}