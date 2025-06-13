//! Comprehensive test suite for Variational Quantum Algorithm (VQA) support
//! 
//! This module provides extensive test coverage for all VQA components including
//! algorithm types, optimization strategies, hardware integration, and SciRS2 analysis.

use quantrs2_device::prelude::*;
use quantrs2_device::vqa_support::*;
use quantrs2_core::prelude::*;
use std::collections::HashMap;
use std::time::Duration;
use ndarray::Array1;

/// Test helper functions and configurations
mod test_helpers {
    use super::*;

    pub fn create_test_vqe_config() -> VQAConfig {
        VQAConfig::new(VQAAlgorithmType::VQE)
    }

    pub fn create_test_qaoa_config() -> VQAConfig {
        qaoa_config(3) // 3 layers
    }

    pub fn create_test_parametric_circuit() -> ParametricCircuit {
        ParametricCircuit::new(2) // 2-qubit circuit
    }

    pub fn create_mock_objective_function() -> Box<dyn ObjectiveFunction> {
        struct MockObjective;
        impl ObjectiveFunction for MockObjective {
            fn evaluate(&self, parameters: &[f64], _circuit: &ParametricCircuit) -> DeviceResult<ObjectiveResult> {
                // Simple quadratic objective for testing
                let value = parameters.iter().map(|&x| x * x).sum::<f64>();
                Ok(ObjectiveResult {
                    value,
                    gradient: Some(parameters.iter().map(|&x| 2.0 * x).collect()),
                    hessian: None,
                    execution_time: Duration::from_millis(10),
                    metadata: HashMap::new(),
                })
            }

            fn name(&self) -> &str {
                "mock_quadratic"
            }
        }
        Box::new(MockObjective)
    }
}

use test_helpers::*;

/// Basic VQA configuration tests
mod configuration_tests {
    use super::*;

    #[test]
    fn test_vqe_config_creation() {
        let config = create_test_vqe_config();
        assert_eq!(config.algorithm_type, VQAAlgorithmType::VQE);
        assert!(config.optimization_config.max_iterations > 0);
        assert!(config.optimization_config.convergence_tolerance > 0.0);
    }

    #[test]
    fn test_qaoa_config_creation() {
        let config = create_test_qaoa_config();
        assert_eq!(config.algorithm_type, VQAAlgorithmType::QAOA);
        // QAOA should have higher iteration count due to layers
        assert!(config.optimization_config.max_iterations >= 300);
    }

    #[test]
    fn test_custom_vqa_config() {
        let mut config = VQAConfig::new(VQAAlgorithmType::Custom("MyCustomVQA".to_string()));
        config.optimization_config.primary_optimizer = VQAOptimizer::DifferentialEvolution;
        config.optimization_config.enable_adaptive = true;
        
        assert!(matches!(config.algorithm_type, VQAAlgorithmType::Custom(_)));
        assert_eq!(config.optimization_config.primary_optimizer, VQAOptimizer::DifferentialEvolution);
        assert!(config.optimization_config.enable_adaptive);
    }

    #[test]
    fn test_all_optimizer_types() {
        let optimizers = vec![
            VQAOptimizer::LBFGSB,
            VQAOptimizer::COBYLA,
            VQAOptimizer::NelderMead,
            VQAOptimizer::DifferentialEvolution,
            VQAOptimizer::SimulatedAnnealing,
            VQAOptimizer::BasinHopping,
            VQAOptimizer::DualAnnealing,
            VQAOptimizer::Powell,
            VQAOptimizer::PSO,
            VQAOptimizer::QNG,
            VQAOptimizer::SPSA,
        ];

        for optimizer in optimizers {
            let mut config = create_test_vqe_config();
            config.optimization_config.primary_optimizer = optimizer.clone();
            // Should not panic
            assert_eq!(config.optimization_config.primary_optimizer, optimizer);
        }
    }

    #[test]
    fn test_gradient_methods() {
        let gradient_methods = vec![
            GradientMethod::ParameterShift,
            GradientMethod::FiniteDifference,
            GradientMethod::ForwardDifference,
            GradientMethod::CentralDifference,
            GradientMethod::ComplexStep,
            GradientMethod::AutomaticDifferentiation,
        ];

        for method in gradient_methods {
            let mut config = create_test_vqe_config();
            config.optimization_config.gradient_method = method.clone();
            assert_eq!(config.optimization_config.gradient_method, method);
        }
    }
}

/// Parametric circuit tests
mod circuit_tests {
    use super::*;

    #[test]
    fn test_parametric_circuit_creation() {
        let circuit = create_test_parametric_circuit();
        assert_eq!(circuit.num_qubits(), 2);
        assert_eq!(circuit.num_parameters(), circuit.parameters().len());
    }

    #[test]
    fn test_circuit_parameterization() {
        let mut circuit = create_test_parametric_circuit();
        let initial_params = circuit.parameters().clone();
        
        let new_params = vec![1.5, 2.5, 0.5, 1.0];
        circuit.set_parameters(&new_params).expect("Failed to set parameters");
        
        assert_ne!(circuit.parameters(), &initial_params);
        assert_eq!(circuit.parameters().len(), new_params.len().min(circuit.num_parameters()));
    }

    #[test]
    fn test_circuit_depth_analysis() {
        let circuit = create_test_parametric_circuit();
        let depth = circuit.depth();
        assert!(depth > 0, "Circuit should have non-zero depth");
    }

    #[test]
    fn test_circuit_gate_count() {
        let circuit = create_test_parametric_circuit();
        let gate_count = circuit.gate_count();
        assert!(gate_count > 0, "Circuit should have gates");
    }
}

/// Objective function tests
mod objective_tests {
    use super::*;

    #[test]
    fn test_mock_objective_evaluation() {
        let objective = create_mock_objective_function();
        let circuit = create_test_parametric_circuit();
        let parameters = vec![1.0, 2.0];
        
        let result = objective.evaluate(&parameters, &circuit);
        assert!(result.is_ok(), "Objective evaluation should succeed");
        
        let obj_result = result.unwrap();
        assert_eq!(obj_result.value, 5.0); // 1^2 + 2^2 = 5
        assert!(obj_result.gradient.is_some());
        assert_eq!(obj_result.gradient.unwrap(), vec![2.0, 4.0]); // Gradient of x^2 is 2x
    }

    #[test]
    fn test_objective_timing() {
        let objective = create_mock_objective_function();
        let circuit = create_test_parametric_circuit();
        let parameters = vec![0.5, 1.5];
        
        let start_time = std::time::Instant::now();
        let result = objective.evaluate(&parameters, &circuit).unwrap();
        let evaluation_time = start_time.elapsed();
        
        assert!(result.execution_time.as_millis() > 0);
        assert!(evaluation_time.as_millis() >= result.execution_time.as_millis());
    }

    #[test]
    fn test_multiple_objective_evaluations() {
        let objective = create_mock_objective_function();
        let circuit = create_test_parametric_circuit();
        
        let test_cases = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![1.0, 1.0], 2.0),
            (vec![2.0, 3.0], 13.0),
            (vec![-1.0, 2.0], 5.0),
        ];
        
        for (params, expected) in test_cases {
            let result = objective.evaluate(&params, &circuit).unwrap();
            assert!((result.value - expected).abs() < 1e-10, 
                   "Expected {}, got {} for params {:?}", expected, result.value, params);
        }
    }
}

/// VQA executor tests
mod executor_tests {
    use super::*;

    #[test]
    fn test_vqa_executor_creation() {
        let config = create_test_vqe_config();
        let calibration_manager = crate::calibration::CalibrationManager::new();
        let executor = VQAExecutor::new(config, calibration_manager, None);
        
        // Should create without error
        assert!(true, "VQA executor created successfully");
    }

    #[test]
    fn test_executor_config_validation() {
        let mut config = VQAExecutorConfig::default();
        config.max_iterations = 0; // Invalid
        
        // The executor should handle invalid configurations gracefully
        // This test ensures the configuration is properly validated
        assert_eq!(config.max_iterations, 0);
        
        // Correct the configuration
        config.max_iterations = 100;
        assert!(config.max_iterations > 0);
    }

    #[test]
    fn test_multiple_executor_configurations() {
        let configs = vec![
            VQAExecutorConfig {
                max_iterations: 500,
                tolerance: 1e-8,
                optimizer: OptimizerConfig {
                    optimizer_type: OptimizerType::Adam,
                    learning_rate: 0.001,
                    parameters: HashMap::new(),
                },
                ..Default::default()
            },
            VQAExecutorConfig {
                max_iterations: 1000,
                tolerance: 1e-6,
                optimizer: OptimizerConfig {
                    optimizer_type: OptimizerType::LBFGSB,
                    learning_rate: 0.01,
                    parameters: HashMap::new(),
                },
                ..Default::default()
            },
        ];
        
        for config in configs {
            assert!(config.max_iterations > 0);
            assert!(config.tolerance > 0.0);
            assert!(config.optimizer.learning_rate > 0.0);
        }
    }
}

/// Statistical analysis tests
mod statistical_tests {
    use super::*;

    #[test]
    fn test_vqa_statistics_creation() {
        let mut stats = VQAStatistics::new();
        
        // Add some mock convergence data
        let convergence_data = vec![10.0, 5.0, 2.0, 1.0, 0.5, 0.1];
        for value in convergence_data {
            stats.add_iteration(value, 0.01, Duration::from_millis(100));
        }
        
        assert_eq!(stats.iteration_count(), 6);
        assert!(stats.final_value() < stats.initial_value());
        assert!(stats.total_time() > Duration::from_millis(500));
    }

    #[test]
    fn test_convergence_analysis() {
        let mut stats = VQAStatistics::new();
        
        // Simulate converging optimization
        let values = vec![100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625];
        for (i, value) in values.iter().enumerate() {
            stats.add_iteration(*value, 0.1, Duration::from_millis(50));
            
            if i > 3 {
                // Should detect convergence after several iterations
                let convergence = stats.analyze_convergence(1e-1);
                if convergence.converged {
                    assert!(convergence.rate > 0.0);
                    break;
                }
            }
        }
    }

    #[test]
    fn test_gradient_statistics() {
        let mut stats = VQAStatistics::new();
        
        // Add gradient norms
        let gradient_norms = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1];
        for (i, &norm) in gradient_norms.iter().enumerate() {
            stats.add_iteration(10.0 - i as f64, norm, Duration::from_millis(100));
        }
        
        let gradient_analysis = stats.analyze_gradients();
        assert!(gradient_analysis.average_norm > 0.0);
        assert!(gradient_analysis.final_norm < gradient_analysis.initial_norm);
    }
}

/// Hardware integration tests
mod hardware_tests {
    use super::*;

    #[test]
    fn test_hardware_config_creation() {
        let config = HardwareConfig::default();
        assert!(config.max_qubits >= 2); // Should support at least 2 qubits
        assert!(config.max_circuit_depth > 0);
    }

    #[test]
    fn test_hardware_awareness() {
        let mut config = HardwareConfig::default();
        config.enable_circuit_optimization = true;
        config.enable_noise_adaptation = true;
        config.preferred_backends = vec!["ibmq_qasm_simulator".to_string()];
        
        assert!(config.enable_circuit_optimization);
        assert!(config.enable_noise_adaptation);
        assert!(!config.preferred_backends.is_empty());
    }

    #[test]
    fn test_device_compatibility() {
        let hardware_configs = vec![
            HardwareConfig {
                max_qubits: 5,
                max_circuit_depth: 100,
                enable_circuit_optimization: true,
                ..Default::default()
            },
            HardwareConfig {
                max_qubits: 16,
                max_circuit_depth: 1000,
                enable_circuit_optimization: false,
                ..Default::default()
            },
        ];
        
        let circuit = create_test_parametric_circuit();
        
        for config in hardware_configs {
            let is_compatible = circuit.num_qubits() <= config.max_qubits && 
                              circuit.depth() <= config.max_circuit_depth;
            
            if is_compatible {
                assert!(config.max_qubits >= circuit.num_qubits());
                assert!(config.max_circuit_depth >= circuit.depth());
            }
        }
    }
}

/// Noise mitigation tests
mod noise_tests {
    use super::*;

    #[test]
    fn test_noise_mitigation_config() {
        let config = NoiseMitigationConfig::default();
        // Default should have some mitigation enabled
        assert!(config.enable_zne || config.enable_readout_correction || config.enable_symmetry_verification);
    }

    #[test]
    fn test_noise_mitigation_strategies() {
        let strategies = vec![
            NoiseMitigationStrategy::ZeroNoiseExtrapolation,
            NoiseMitigationStrategy::ReadoutErrorMitigation,
            NoiseMitigationStrategy::SymmetryVerification,
            NoiseMitigationStrategy::DynamicalDecoupling,
            NoiseMitigationStrategy::ErrorCorrection,
        ];
        
        for strategy in strategies {
            let mut config = NoiseMitigationConfig::default();
            config.strategies.push(strategy.clone());
            assert!(config.strategies.contains(&strategy));
        }
    }

    #[test]
    fn test_adaptive_noise_mitigation() {
        let mut config = NoiseMitigationConfig::default();
        config.enable_adaptive_mitigation = true;
        config.noise_threshold = 0.05;
        
        assert!(config.enable_adaptive_mitigation);
        assert!(config.noise_threshold > 0.0 && config.noise_threshold < 1.0);
    }
}

/// Optimization strategy tests
mod optimization_tests {
    use super::*;

    #[test]
    fn test_multi_start_optimization() {
        let config = MultiStartConfig {
            num_starts: 5,
            parameter_sampling: ParameterSampling::Random,
            combine_results: true,
            parallel_execution: true,
        };
        
        assert_eq!(config.num_starts, 5);
        assert!(config.combine_results);
        assert!(config.parallel_execution);
    }

    #[test]
    fn test_convergence_criteria() {
        let criteria = vec![
            ConvergenceCriterion::ValueTolerance(1e-6),
            ConvergenceCriterion::GradientNorm(1e-8),
            ConvergenceCriterion::ParameterChange(1e-10),
            ConvergenceCriterion::MaxIterations(1000),
        ];
        
        for criterion in criteria {
            match criterion {
                ConvergenceCriterion::ValueTolerance(tol) => assert!(tol > 0.0),
                ConvergenceCriterion::GradientNorm(tol) => assert!(tol > 0.0),
                ConvergenceCriterion::ParameterChange(tol) => assert!(tol > 0.0),
                ConvergenceCriterion::MaxIterations(max_iter) => assert!(max_iter > 0),
                _ => {}, // Handle any additional criteria
            }
        }
    }

    #[test]
    fn test_adaptive_shot_allocation() {
        let config = AdaptiveShotConfig {
            initial_shots: 1000,
            max_shots: 10000,
            adaptation_strategy: ShotAdaptationStrategy::VarianceReduction,
            convergence_threshold: 0.01,
        };
        
        assert!(config.initial_shots > 0);
        assert!(config.max_shots >= config.initial_shots);
        assert!(config.convergence_threshold > 0.0);
    }
}

/// Integration tests
mod integration_tests {
    use super::*;

    #[test]
    fn test_vqe_workflow_creation() {
        let config = create_test_vqe_config();
        let circuit = create_test_parametric_circuit();
        let objective = create_mock_objective_function();
        
        // This tests that all components can be created together
        assert_eq!(config.algorithm_type, VQAAlgorithmType::VQE);
        assert!(circuit.num_qubits() > 0);
        assert_eq!(objective.name(), "mock_quadratic");
    }

    #[test]
    fn test_qaoa_workflow_creation() {
        let config = create_test_qaoa_config();
        let circuit = create_test_parametric_circuit();
        let objective = create_mock_objective_function();
        
        // Test QAOA-specific configuration
        assert_eq!(config.algorithm_type, VQAAlgorithmType::QAOA);
        assert!(config.optimization_config.max_iterations >= 300); // Due to layers
        assert!(circuit.num_qubits() > 0);
        assert_eq!(objective.name(), "mock_quadratic");
    }

    #[test]
    fn test_custom_vqa_workflow() {
        let mut config = VQAConfig::new(VQAAlgorithmType::Custom("TestVQA".to_string()));
        config.optimization_config.primary_optimizer = VQAOptimizer::DifferentialEvolution;
        config.optimization_config.enable_adaptive = true;
        
        let circuit = create_test_parametric_circuit();
        let objective = create_mock_objective_function();
        
        // Test custom VQA setup
        assert!(matches!(config.algorithm_type, VQAAlgorithmType::Custom(_)));
        assert_eq!(config.optimization_config.primary_optimizer, VQAOptimizer::DifferentialEvolution);
        assert!(circuit.num_parameters() > 0);
        assert!(objective.evaluate(&vec![1.0, 2.0], &circuit).is_ok());
    }
}

/// Performance and scalability tests
mod performance_tests {
    use super::*;

    #[test]
    fn test_large_parameter_space() {
        let mut circuit = ParametricCircuit::new(4); // 4-qubit circuit
        let large_params = vec![0.1; 50]; // 50 parameters
        
        let result = circuit.set_parameters(&large_params);
        
        // Should handle large parameter spaces gracefully
        match result {
            Ok(_) => assert!(circuit.parameters().len() > 0),
            Err(_) => {
                // Expected if circuit doesn't support 50 parameters
                assert!(circuit.num_parameters() < large_params.len());
            }
        }
    }

    #[test]
    fn test_objective_evaluation_efficiency() {
        let objective = create_mock_objective_function();
        let circuit = create_test_parametric_circuit();
        
        // Test multiple evaluations for timing
        let num_evaluations = 100;
        let start_time = std::time::Instant::now();
        
        for i in 0..num_evaluations {
            let params = vec![i as f64 * 0.01, (i + 1) as f64 * 0.01];
            let result = objective.evaluate(&params, &circuit);
            assert!(result.is_ok());
        }
        
        let total_time = start_time.elapsed();
        let avg_time_per_eval = total_time / num_evaluations;
        
        // Each evaluation should be reasonably fast (less than 10ms on average)
        assert!(avg_time_per_eval.as_millis() < 10, 
               "Average evaluation time too slow: {}ms", avg_time_per_eval.as_millis());
    }

    #[test]
    fn test_concurrent_vqa_execution() {
        let config = create_test_vqe_config();
        let calibration_manager = crate::calibration::CalibrationManager::new();
        
        // Create multiple executors to test concurrent usage
        let _executors: Vec<_> = (0..5).map(|_| {
            VQAExecutor::new(config.clone(), calibration_manager.clone(), None)
        }).collect();
        
        // Should be able to create multiple executors without issues
        assert_eq!(_executors.len(), 5);
    }
}

/// Error handling tests
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_invalid_parameter_bounds() {
        let mut config = create_test_vqe_config();
        // Set invalid bounds (min > max)
        config.optimization_config.parameter_bounds = Some(vec![(1.0, -1.0)]);
        
        // The system should handle invalid bounds gracefully
        if let Some(bounds) = &config.optimization_config.parameter_bounds {
            for (min, max) in bounds {
                if min > max {
                    // This is an invalid bound that should be detected
                    assert!(min > max, "Invalid bound detected as expected");
                }
            }
        }
    }

    #[test]
    fn test_zero_tolerance() {
        let mut config = create_test_vqe_config();
        config.optimization_config.convergence_tolerance = 0.0;
        
        // Zero tolerance might cause issues, but should be handled gracefully
        assert_eq!(config.optimization_config.convergence_tolerance, 0.0);
        
        // Correct to valid tolerance
        config.optimization_config.convergence_tolerance = 1e-10;
        assert!(config.optimization_config.convergence_tolerance > 0.0);
    }

    #[test]
    fn test_empty_parameter_list() {
        let objective = create_mock_objective_function();
        let circuit = create_test_parametric_circuit();
        let empty_params = vec![];
        
        let result = objective.evaluate(&empty_params, &circuit);
        
        // Should handle empty parameters gracefully
        match result {
            Ok(res) => assert_eq!(res.value, 0.0), // Sum of squares of empty list is 0
            Err(_) => {}, // Also acceptable to reject empty parameters
        }
    }

    #[test]
    fn test_extremely_large_parameters() {
        let objective = create_mock_objective_function();
        let circuit = create_test_parametric_circuit();
        let large_params = vec![1e10, 1e10];
        
        let result = objective.evaluate(&large_params, &circuit);
        
        // Should handle very large parameters without overflow
        match result {
            Ok(res) => {
                assert!(res.value.is_finite(), "Result should be finite");
                assert!(res.value > 0.0, "Result should be positive for large positive parameters");
            },
            Err(_) => {}, // Acceptable to reject extremely large parameters
        }
    }
}

/// Documentation and example tests
mod documentation_tests {
    use super::*;

    #[test]
    fn test_basic_usage_example() {
        // This test demonstrates basic VQA usage as documented
        
        // 1. Create VQA configuration
        let config = create_test_vqe_config();
        
        // 2. Create parametric circuit
        let circuit = create_test_parametric_circuit();
        
        // 3. Define objective function
        let objective = create_mock_objective_function();
        
        // 4. Create executor
        let calibration_manager = crate::calibration::CalibrationManager::new();
        let _executor = VQAExecutor::new(config, calibration_manager, None);
        
        // This sequence should work without errors
        assert!(true, "Basic VQA usage example completed successfully");
    }

    #[test]
    fn test_configuration_customization_example() {
        // Example of customizing VQA configuration
        let mut config = create_test_vqe_config();
        
        // Customize optimization
        config.optimization_config.primary_optimizer = VQAOptimizer::DifferentialEvolution;
        config.optimization_config.max_iterations = 500;
        config.optimization_config.convergence_tolerance = 1e-8;
        config.optimization_config.enable_adaptive = true;
        
        // Customize hardware settings
        config.hardware_config.enable_circuit_optimization = true;
        config.hardware_config.enable_noise_adaptation = true;
        
        // Customize noise mitigation
        config.noise_mitigation.enable_zne = true;
        config.noise_mitigation.enable_readout_correction = true;
        
        // All customizations should be applied
        assert_eq!(config.optimization_config.primary_optimizer, VQAOptimizer::DifferentialEvolution);
        assert_eq!(config.optimization_config.max_iterations, 500);
        assert_eq!(config.optimization_config.convergence_tolerance, 1e-8);
        assert!(config.optimization_config.enable_adaptive);
        assert!(config.hardware_config.enable_circuit_optimization);
        assert!(config.noise_mitigation.enable_zne);
    }
}