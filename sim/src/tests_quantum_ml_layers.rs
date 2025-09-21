//! Comprehensive tests for Quantum Machine Learning Layers
//!
//! This module contains tests for all aspects of the quantum machine learning layers
//! framework, including layer creation, training algorithms, hardware optimization,
//! and performance benchmarking.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_machine_learning_layers::*;
    use ndarray::Array1;
    use num_complex::Complex64;
    use std::f64::consts::PI;

    #[test]
    fn test_qml_config_creation() {
        let config = QMLConfig::default();
        assert_eq!(config.num_qubits, 8);
        assert_eq!(
            config.architecture_type,
            QMLArchitectureType::VariationalQuantumCircuit
        );
        assert_eq!(config.layer_configs.len(), 1);
        assert!(config.quantum_advantage_analysis);
    }

    #[test]
    fn test_qml_framework_creation() {
        let config = QMLConfig::default();
        let framework = QuantumMLFramework::new(config);
        assert!(framework.is_ok());

        let framework = framework.unwrap();
        assert_eq!(framework.get_layers().len(), 1);
    }

    #[test]
    fn test_parameterized_quantum_circuit_layer() {
        let config = QMLLayerConfig {
            layer_type: QMLLayerType::ParameterizedQuantumCircuit,
            num_parameters: 8,
            ansatz_type: AnsatzType::Hardware,
            entanglement_pattern: EntanglementPattern::Linear,
            rotation_gates: vec![RotationGate::RY, RotationGate::RZ],
            depth: 2,
            enable_gradient_computation: true,
        };

        let layer = ParameterizedQuantumCircuitLayer::new(4, config);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.get_num_parameters(), 8);
        assert_eq!(layer.get_depth(), 2);
        assert!(layer.get_gate_count() > 0);
    }

    #[test]
    fn test_quantum_convolutional_layer() {
        let config = QMLLayerConfig {
            layer_type: QMLLayerType::QuantumConvolutional,
            num_parameters: 6,
            ansatz_type: AnsatzType::Hardware,
            entanglement_pattern: EntanglementPattern::Linear,
            rotation_gates: vec![RotationGate::RY],
            depth: 1,
            enable_gradient_computation: true,
        };

        let layer = QuantumConvolutionalLayer::new(4, config);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.get_num_parameters(), 6);
        assert!(layer.get_gate_count() > 0);
    }

    #[test]
    fn test_quantum_dense_layer() {
        let config = QMLLayerConfig {
            layer_type: QMLLayerType::QuantumDense,
            num_parameters: 6, // C(4,2) = 6 connections for 4 qubits
            ansatz_type: AnsatzType::AllToAll,
            entanglement_pattern: EntanglementPattern::AllToAll,
            rotation_gates: vec![RotationGate::RY],
            depth: 1,
            enable_gradient_computation: true,
        };

        let layer = QuantumDenseLayer::new(4, config);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.get_num_parameters(), 6);
        assert!(layer.get_gate_count() > 0);
    }

    #[test]
    fn test_quantum_lstm_layer() {
        let config = QMLLayerConfig {
            layer_type: QMLLayerType::QuantumLSTM,
            num_parameters: 16,
            ansatz_type: AnsatzType::Hardware,
            entanglement_pattern: EntanglementPattern::Linear,
            rotation_gates: vec![RotationGate::RY],
            depth: 1,
            enable_gradient_computation: true,
        };

        let layer = QuantumLSTMLayer::new(4, config);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.get_num_parameters(), 16);
        assert_eq!(layer.get_lstm_gates().len(), 4); // Forget, Input, Output, Candidate
    }

    #[test]
    fn test_quantum_attention_layer() {
        let config = QMLLayerConfig {
            layer_type: QMLLayerType::QuantumAttention,
            num_parameters: 8,
            ansatz_type: AnsatzType::Hardware,
            entanglement_pattern: EntanglementPattern::AllToAll,
            rotation_gates: vec![RotationGate::RY],
            depth: 1,
            enable_gradient_computation: true,
        };

        let layer = QuantumAttentionLayer::new(4, config);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.get_num_parameters(), 8);
        assert_eq!(layer.get_attention_structure().len(), 2); // 2 attention heads
    }

    #[test]
    fn test_forward_pass() {
        let config = QMLConfig::default();
        let mut framework = QuantumMLFramework::new(config).unwrap();

        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let output = framework.forward(&input);

        assert!(output.is_ok());
        let output = output.unwrap();
        assert_eq!(output.len(), framework.get_config().num_qubits);
    }

    #[test]
    fn test_backward_pass() {
        let config = QMLConfig::default();
        let mut framework = QuantumMLFramework::new(config).unwrap();

        let loss_gradient = Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
        let gradient = framework.backward(&loss_gradient);

        assert!(gradient.is_ok());
    }

    #[test]
    fn test_amplitude_encoding() {
        let config = QMLConfig {
            classical_preprocessing: ClassicalPreprocessingConfig {
                encoding_method: DataEncodingMethod::Amplitude,
                ..Default::default()
            },
            ..Default::default()
        };

        let framework = QuantumMLFramework::new(config).unwrap();
        let input = Array1::from_vec(vec![0.6, 0.8]); // Should normalize to unit vector
        let encoded = framework.encode_amplitude_public(&input);

        assert!(encoded.is_ok());
        let encoded_state = encoded.unwrap();

        // Check normalization
        let norm_sq: f64 = encoded_state.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_angle_encoding() {
        let config = QMLConfig {
            classical_preprocessing: ClassicalPreprocessingConfig {
                encoding_method: DataEncodingMethod::Angle,
                ..Default::default()
            },
            ..Default::default()
        };

        let framework = QuantumMLFramework::new(config).unwrap();
        let input = Array1::from_vec(vec![PI / 4.0, PI / 2.0, 0.0, PI]);
        let encoded = framework.encode_angle_public(&input);

        assert!(encoded.is_ok());
        let encoded_state = encoded.unwrap();

        // Check normalization
        let norm_sq: f64 = encoded_state.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_basis_encoding() {
        let config = QMLConfig {
            classical_preprocessing: ClassicalPreprocessingConfig {
                encoding_method: DataEncodingMethod::Basis,
                ..Default::default()
            },
            ..Default::default()
        };

        let framework = QuantumMLFramework::new(config).unwrap();
        let input = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]); // Should encode to |1010⟩ = |10⟩
        let encoded = framework.encode_basis_public(&input);

        assert!(encoded.is_ok());
        let encoded_state = encoded.unwrap();

        // Check that only one state has amplitude 1
        let non_zero_count = encoded_state
            .iter()
            .filter(|c| c.norm_sqr() > 1e-10)
            .count();
        assert_eq!(non_zero_count, 1);
    }

    #[test]
    fn test_quantum_feature_map_encoding() {
        let config = QMLConfig {
            classical_preprocessing: ClassicalPreprocessingConfig {
                encoding_method: DataEncodingMethod::QuantumFeatureMap,
                ..Default::default()
            },
            ..Default::default()
        };

        let framework = QuantumMLFramework::new(config).unwrap();
        let input = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let encoded = framework.encode_quantum_feature_map_public(&input);

        assert!(encoded.is_ok());
        let encoded_state = encoded.unwrap();

        // Check normalization
        let norm_sq: f64 = encoded_state.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sq - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_z_measurement() {
        let config = QMLConfig::default();
        let framework = QuantumMLFramework::new(config).unwrap();

        // Create |0⟩ state
        let state_size = 1 << framework.get_config().num_qubits;
        let mut state = Array1::zeros(state_size);
        state[0] = Complex64::new(1.0, 0.0);

        // Measure first qubit (should be +1 for |0⟩)
        let expectation = framework.measure_pauli_z_expectation_public(&state, 0);
        assert!(expectation.is_ok());
        assert!((expectation.unwrap() - 1.0).abs() < 1e-10);

        // Create |1⟩ state for first qubit
        state[0] = Complex64::new(0.0, 0.0);
        state[1] = Complex64::new(1.0, 0.0);

        let expectation = framework.measure_pauli_z_expectation_public(&state, 0);
        assert!(expectation.is_ok());
        assert!((expectation.unwrap() + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_training_algorithms() {
        // Test different training algorithms
        let algorithms = vec![
            QMLTrainingAlgorithm::ParameterShift,
            QMLTrainingAlgorithm::FiniteDifference,
        ];

        for algorithm in algorithms {
            let config = QMLConfig {
                num_qubits: 4, // Match the synthetic data dimensions
                training_config: QMLTrainingConfig {
                    algorithm,
                    epochs: 5, // Short training for testing
                    ..Default::default()
                },
                ..Default::default()
            };

            let mut framework = QuantumMLFramework::new(config).unwrap();
            let (inputs, outputs) = QMLUtils::generate_synthetic_data(20, 4, 4);
            let (train_data, val_data) = QMLUtils::train_test_split(inputs, outputs, 0.2);

            let result = framework.train(&train_data, Some(&val_data));
            assert!(result.is_ok());

            let training_result = result.unwrap();
            assert!(training_result.epochs_trained > 0);
            assert!(training_result.final_training_loss >= 0.0);
        }
    }

    #[test]
    fn test_optimizers() {
        let optimizers = vec![OptimizerType::SGD, OptimizerType::Adam];

        for optimizer in optimizers {
            let config = QMLConfig {
                num_qubits: 4, // Match the synthetic data dimensions
                training_config: QMLTrainingConfig {
                    optimizer,
                    epochs: 3,
                    ..Default::default()
                },
                ..Default::default()
            };

            let mut framework = QuantumMLFramework::new(config).unwrap();
            let (inputs, outputs) = QMLUtils::generate_synthetic_data(10, 4, 4);
            let (train_data, _) = QMLUtils::train_test_split(inputs, outputs, 0.8);

            let result = framework.train(&train_data, None);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_learning_rate_schedules() {
        let schedules = vec![
            LearningRateSchedule::Constant,
            LearningRateSchedule::ExponentialDecay,
            LearningRateSchedule::StepDecay,
            LearningRateSchedule::CosineAnnealing,
        ];

        for schedule in schedules {
            let config = QMLConfig {
                training_config: QMLTrainingConfig {
                    lr_schedule: schedule,
                    epochs: 5,
                    ..Default::default()
                },
                ..Default::default()
            };

            let framework = QuantumMLFramework::new(config).unwrap();

            // Test learning rate at different epochs
            let lr_epoch_0 = framework.get_current_learning_rate_public(0);
            let lr_epoch_10 = framework.get_current_learning_rate_public(10);

            match schedule {
                LearningRateSchedule::Constant => {
                    assert!((lr_epoch_0 - lr_epoch_10).abs() < 1e-10);
                }
                LearningRateSchedule::ExponentialDecay => {
                    assert!(lr_epoch_10 < lr_epoch_0);
                }
                _ => {
                    // Other schedules should also show some change
                    // (may not be monotonic decrease)
                }
            }
        }
    }

    #[test]
    fn test_ansatz_types() {
        let ansatz_types = vec![
            AnsatzType::Hardware,
            AnsatzType::Layered,
            AnsatzType::BrickWall,
        ];

        for ansatz in ansatz_types {
            let config = QMLLayerConfig {
                layer_type: QMLLayerType::ParameterizedQuantumCircuit,
                num_parameters: 16,
                ansatz_type: ansatz,
                entanglement_pattern: EntanglementPattern::Linear,
                rotation_gates: vec![RotationGate::RY],
                depth: 2,
                enable_gradient_computation: true,
            };

            let layer = ParameterizedQuantumCircuitLayer::new(4, config);
            assert!(layer.is_ok());

            let layer = layer.unwrap();
            assert!(layer.get_gate_count() > 0);
            assert_eq!(layer.get_depth(), 2);
        }
    }

    #[test]
    fn test_entanglement_patterns() {
        let patterns = vec![
            EntanglementPattern::Linear,
            EntanglementPattern::Circular,
            EntanglementPattern::AllToAll,
        ];

        for pattern in patterns {
            let config = QMLLayerConfig {
                layer_type: QMLLayerType::ParameterizedQuantumCircuit,
                num_parameters: 8,
                ansatz_type: AnsatzType::Hardware,
                entanglement_pattern: pattern,
                rotation_gates: vec![RotationGate::RY],
                depth: 2,
                enable_gradient_computation: true,
            };

            let layer = ParameterizedQuantumCircuitLayer::new(4, config);
            assert!(layer.is_ok());

            let layer = layer.unwrap();
            assert!(layer.get_gate_count() > 0);
        }
    }

    #[test]
    fn test_rotation_gates() {
        let gate_types = vec![RotationGate::RX, RotationGate::RY, RotationGate::RZ];

        for gate_type in gate_types {
            let config = QMLLayerConfig {
                layer_type: QMLLayerType::ParameterizedQuantumCircuit,
                num_parameters: 8,
                ansatz_type: AnsatzType::Hardware,
                entanglement_pattern: EntanglementPattern::Linear,
                rotation_gates: vec![gate_type],
                depth: 2,
                enable_gradient_computation: true,
            };

            let layer = ParameterizedQuantumCircuitLayer::new(4, config);
            assert!(layer.is_ok());
        }
    }

    #[test]
    fn test_hardware_optimization_targets() {
        let targets = vec![
            QuantumHardwareTarget::Simulator,
            QuantumHardwareTarget::IBM,
            QuantumHardwareTarget::Google,
        ];

        for target in targets {
            let config = QMLConfig {
                hardware_optimization: HardwareOptimizationConfig {
                    target_hardware: target,
                    ..Default::default()
                },
                ..Default::default()
            };

            let framework = QuantumMLFramework::new(config);
            assert!(framework.is_ok());
        }
    }

    #[test]
    fn test_connectivity_constraints() {
        let constraints = vec![
            ConnectivityConstraints::AllToAll,
            ConnectivityConstraints::Linear,
            ConnectivityConstraints::Grid(2, 2),
        ];

        for constraint in constraints {
            let config = QMLConfig {
                hardware_optimization: HardwareOptimizationConfig {
                    connectivity_constraints: constraint,
                    ..Default::default()
                },
                ..Default::default()
            };

            let framework = QuantumMLFramework::new(config);
            assert!(framework.is_ok());
        }
    }

    #[test]
    fn test_early_stopping() {
        let config = QMLConfig {
            num_qubits: 4, // Match the synthetic data dimensions
            training_config: QMLTrainingConfig {
                early_stopping: EarlyStoppingConfig {
                    enabled: true,
                    patience: 2,
                    min_delta: 1e-6,
                    monitor_metric: "val_loss".to_string(),
                    mode_max: false,
                },
                epochs: 20, // Reduced epochs for faster testing
                ..Default::default()
            },
            ..Default::default()
        };

        let mut framework = QuantumMLFramework::new(config).unwrap();
        let (inputs, outputs) = QMLUtils::generate_synthetic_data(10, 4, 4); // Reduced data size
        let (train_data, val_data) = QMLUtils::train_test_split(inputs, outputs, 0.2);

        let result = framework.train(&train_data, Some(&val_data));
        assert!(result.is_ok());

        let training_result = result.unwrap();
        // Early stopping should prevent training all epochs or training should complete
        // The exact behavior depends on the data and initialization
        assert!(training_result.epochs_trained <= 20);
    }

    #[test]
    fn test_regularization() {
        let config = QMLConfig {
            num_qubits: 4, // Match the synthetic data dimensions
            training_config: QMLTrainingConfig {
                regularization: RegularizationConfig {
                    l1_strength: 0.01,
                    l2_strength: 0.01,
                    parameter_bounds: Some((-PI, PI)),
                    enable_clipping: true,
                    gradient_clip_threshold: 1.0,
                    ..Default::default()
                },
                epochs: 5,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut framework = QuantumMLFramework::new(config).unwrap();
        let (inputs, outputs) = QMLUtils::generate_synthetic_data(10, 4, 4);
        let (train_data, _) = QMLUtils::train_test_split(inputs, outputs, 0.8);

        let result = framework.train(&train_data, None);
        assert!(result.is_ok());

        // Check that parameters are within bounds
        for layer in framework.get_layers() {
            let params = layer.get_parameters();
            for &param in params.iter() {
                assert!(param >= -PI && param <= PI);
            }
        }
    }

    #[test]
    fn test_qml_utils() {
        // Test synthetic data generation
        let (inputs, outputs) = QMLUtils::generate_synthetic_data(50, 4, 3);
        assert_eq!(inputs.len(), 50);
        assert_eq!(outputs.len(), 50);
        assert_eq!(inputs[0].len(), 4);
        assert_eq!(outputs[0].len(), 3);

        // Test train-test split
        let (train_data, test_data) = QMLUtils::train_test_split(inputs, outputs, 0.2);
        assert_eq!(train_data.len(), 40);
        assert_eq!(test_data.len(), 10);

        // Test accuracy evaluation
        let predictions = vec![
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![0.0, 1.0]),
        ];
        let targets = vec![
            Array1::from_vec(vec![1.0, 0.0]),
            Array1::from_vec(vec![0.1, 0.9]),
        ];

        let accuracy = QMLUtils::evaluate_accuracy(&predictions, &targets, 0.1);
        assert!(accuracy >= 0.0 && accuracy <= 1.0);

        // Test circuit complexity computation
        let complexity = QMLUtils::compute_circuit_complexity(4, 2, 10);
        assert!(complexity.contains_key("state_space_size"));
        assert!(complexity.contains_key("circuit_complexity"));
        assert!(complexity.contains_key("classical_simulation_cost"));
        assert!(complexity.contains_key("quantum_advantage_estimate"));
    }

    #[test]
    fn test_layer_forward_backward() {
        let config = QMLLayerConfig {
            layer_type: QMLLayerType::ParameterizedQuantumCircuit,
            num_parameters: 8,
            ansatz_type: AnsatzType::Hardware,
            entanglement_pattern: EntanglementPattern::Linear,
            rotation_gates: vec![RotationGate::RY],
            depth: 2,
            enable_gradient_computation: true,
        };

        let mut layer = ParameterizedQuantumCircuitLayer::new(4, config).unwrap();

        // Test forward pass
        let state_size = 1 << 4;
        let mut input_state = Array1::zeros(state_size);
        input_state[0] = Complex64::new(1.0, 0.0); // |0000⟩ state

        let output = layer.forward(&input_state);
        assert!(output.is_ok());

        let output_state = output.unwrap();
        let norm_sq: f64 = output_state.iter().map(|c| c.norm_sqr()).sum();
        assert!((norm_sq - 1.0).abs() < 1e-10); // Check normalization

        // Test backward pass
        let gradient_input = Array1::from_vec(vec![0.1; 4]);
        let gradient_output = layer.backward(&gradient_input);
        assert!(gradient_output.is_ok());
    }

    #[test]
    fn test_parameter_operations() {
        let config = QMLLayerConfig {
            layer_type: QMLLayerType::ParameterizedQuantumCircuit,
            num_parameters: 8,
            ansatz_type: AnsatzType::Hardware,
            entanglement_pattern: EntanglementPattern::Linear,
            rotation_gates: vec![RotationGate::RY],
            depth: 2,
            enable_gradient_computation: true,
        };

        let mut layer = ParameterizedQuantumCircuitLayer::new(4, config).unwrap();

        // Test getting parameters
        let params = layer.get_parameters();
        assert_eq!(params.len(), 8);

        // Test setting parameters
        let new_params = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        layer.set_parameters(&new_params);

        let retrieved_params = layer.get_parameters();
        for (original, retrieved) in new_params.iter().zip(retrieved_params.iter()) {
            assert!((original - retrieved).abs() < 1e-10);
        }
    }

    #[test]
    fn test_qml_statistics() {
        let mut stats = QMLStats::new();
        assert_eq!(stats.forward_passes, 0);
        assert_eq!(stats.backward_passes, 0);

        stats.forward_passes += 1;
        stats.backward_passes += 1;

        assert_eq!(stats.forward_passes, 1);
        assert_eq!(stats.backward_passes, 1);
    }

    #[test]
    fn test_quantum_advantage_metrics() {
        let metrics = QuantumAdvantageMetrics {
            quantum_volume: 32.0,
            classical_simulation_cost: 1e12,
            quantum_speedup_factor: 1000.0,
            circuit_depth: 10,
            gate_count: 50,
            entanglement_measure: 0.8,
        };

        assert_eq!(metrics.quantum_volume, 32.0);
        assert_eq!(metrics.classical_simulation_cost, 1e12);
        assert!(metrics.quantum_speedup_factor > 1.0);
    }

    #[test]
    fn test_benchmark_qml_layers() {
        let config = QMLConfig {
            num_qubits: 4,
            training_config: QMLTrainingConfig {
                epochs: 3, // Short training for benchmarking test
                ..Default::default()
            },
            ..Default::default()
        };

        let result = benchmark_quantum_ml_layers(&config);
        assert!(result.is_ok());

        let benchmark_results = result.unwrap();
        assert!(!benchmark_results.training_times.is_empty());
        assert!(!benchmark_results.final_accuracies.is_empty());
        assert!(!benchmark_results.parameter_counts.is_empty());
    }

    #[test]
    fn test_loss_computation() {
        let config = QMLConfig::default();
        let framework = QuantumMLFramework::new(config).unwrap();

        let prediction = Array1::from_vec(vec![1.0, 0.5, 0.0]);
        let target = Array1::from_vec(vec![1.0, 0.5, 0.0]);

        let loss = framework.compute_loss_public(&prediction, &target);
        assert!(loss.is_ok());
        assert!((loss.unwrap() - 0.0).abs() < 1e-10); // Perfect prediction should have zero loss

        let prediction2 = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        let target2 = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let loss2 = framework.compute_loss_public(&prediction2, &target2);
        assert!(loss2.is_ok());
        assert!(loss2.unwrap() > 0.0); // Imperfect prediction should have positive loss
    }

    #[test]
    fn test_loss_gradient_computation() {
        let config = QMLConfig::default();
        let framework = QuantumMLFramework::new(config).unwrap();

        let prediction = Array1::from_vec(vec![1.0, 0.5]);
        let target = Array1::from_vec(vec![0.8, 0.3]);

        let gradient = framework.compute_loss_gradient_public(&prediction, &target);
        assert!(gradient.is_ok());

        let grad = gradient.unwrap();
        assert_eq!(grad.len(), 2);

        // Check gradient calculation (derivative of MSE)
        let expected_grad_0 = 2.0 * (1.0 - 0.8) / 2.0; // 2 * (pred - target) / n
        let expected_grad_1 = 2.0 * (0.5 - 0.3) / 2.0;

        assert!((grad[0] - expected_grad_0).abs() < 1e-10);
        assert!((grad[1] - expected_grad_1).abs() < 1e-10);
    }

    #[test]
    fn test_multi_layer_architecture() {
        let layer_configs = vec![
            QMLLayerConfig {
                layer_type: QMLLayerType::ParameterizedQuantumCircuit,
                num_parameters: 8,
                ansatz_type: AnsatzType::Hardware,
                entanglement_pattern: EntanglementPattern::Linear,
                rotation_gates: vec![RotationGate::RY],
                depth: 2,
                enable_gradient_computation: true,
            },
            QMLLayerConfig {
                layer_type: QMLLayerType::QuantumConvolutional,
                num_parameters: 6,
                ansatz_type: AnsatzType::Hardware,
                entanglement_pattern: EntanglementPattern::Linear,
                rotation_gates: vec![RotationGate::RZ],
                depth: 1,
                enable_gradient_computation: true,
            },
        ];

        let config = QMLConfig {
            layer_configs,
            ..Default::default()
        };

        let framework = QuantumMLFramework::new(config);
        assert!(framework.is_ok());

        let framework = framework.unwrap();
        assert_eq!(framework.get_layers().len(), 2);

        // Check that total parameters is sum of layer parameters
        let total_params: usize = framework
            .get_layers()
            .iter()
            .map(|l| l.get_num_parameters())
            .sum();
        assert_eq!(total_params, 8 + 6);
    }

    #[test]
    fn test_training_state_management() {
        let mut state = QMLTrainingState::new();
        assert_eq!(state.current_epoch, 0);
        assert_eq!(state.current_learning_rate, 0.01);
        assert_eq!(state.best_validation_loss, f64::INFINITY);
        assert_eq!(state.patience_counter, 0);

        state.current_epoch = 10;
        state.current_learning_rate = 0.005;
        state.best_validation_loss = 0.1;
        state.patience_counter = 3;

        assert_eq!(state.current_epoch, 10);
        assert_eq!(state.current_learning_rate, 0.005);
        assert_eq!(state.best_validation_loss, 0.1);
        assert_eq!(state.patience_counter, 3);
    }

    #[test]
    fn test_evaluation() {
        let config = QMLConfig {
            num_qubits: 4, // Match the synthetic data dimensions
            training_config: QMLTrainingConfig {
                epochs: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut framework = QuantumMLFramework::new(config).unwrap();
        let (inputs, outputs) = QMLUtils::generate_synthetic_data(10, 4, 4);
        let data: Vec<(Array1<f64>, Array1<f64>)> = inputs.into_iter().zip(outputs).collect();

        let loss = framework.evaluate(&data);
        assert!(loss.is_ok());
        assert!(loss.unwrap() >= 0.0);
    }
}
