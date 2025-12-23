//! Comprehensive tests for Quantum-Inspired Classical Algorithms
//!
//! This module contains tests for all aspects of the quantum-inspired classical algorithms
//! framework, including optimization algorithms, machine learning, sampling, linear algebra,
//! graph algorithms, and performance benchmarking.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_inspired_classical::*;
    use scirs2_core::ndarray::{Array1, Array2};
    use scirs2_core::Complex64;
    use std::collections::HashMap;
    use std::f64::consts::PI;

    #[test]
    fn test_quantum_inspired_config_creation() {
        let config = QuantumInspiredConfig::default();
        assert_eq!(config.num_variables, 16);
        assert_eq!(config.algorithm_category, AlgorithmCategory::Optimization);
        assert!(config.enable_quantum_heuristics);
        assert_eq!(config.precision, 1e-8);
    }

    #[test]
    fn test_quantum_inspired_config_custom() {
        let config = QuantumInspiredConfig {
            num_variables: 32,
            algorithm_category: AlgorithmCategory::MachineLearning,
            enable_quantum_heuristics: false,
            precision: 1e-10,
            random_seed: Some(42),
            ..Default::default()
        };

        assert_eq!(config.num_variables, 32);
        assert_eq!(
            config.algorithm_category,
            AlgorithmCategory::MachineLearning
        );
        assert!(!config.enable_quantum_heuristics);
        assert_eq!(config.precision, 1e-10);
        assert_eq!(config.random_seed, Some(42));
    }

    #[test]
    fn test_framework_creation() {
        let config = QuantumInspiredConfig::default();
        let framework = QuantumInspiredFramework::new(config);
        assert!(framework.is_ok());

        let framework = framework.unwrap();
        assert_eq!(framework.get_state().variables.len(), 16);
        assert_eq!(framework.get_state().iteration, 0);
        assert_eq!(framework.get_state().best_objective, f64::INFINITY);
    }

    #[test]
    fn test_framework_creation_with_seed() {
        let config = QuantumInspiredConfig {
            random_seed: Some(123),
            ..Default::default()
        };

        let framework = QuantumInspiredFramework::new(config);
        assert!(framework.is_ok());
    }

    #[test]
    fn test_algorithm_config() {
        let config = AlgorithmConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.tolerance, 1e-6);
        assert_eq!(config.population_size, 100);
        assert_eq!(config.elite_ratio, 0.1);
        assert_eq!(config.mutation_rate, 0.1);
        assert_eq!(config.crossover_rate, 0.8);
        assert_eq!(
            config.temperature_schedule,
            TemperatureSchedule::Exponential
        );
    }

    #[test]
    fn test_quantum_parameters() {
        let params = QuantumParameters::default();
        assert!(params.superposition_strength > 0.0);
        assert!(params.entanglement_strength > 0.0);
        assert!(params.interference_strength > 0.0);
        assert!(params.tunneling_probability > 0.0);
        assert!(params.decoherence_rate > 0.0);
        assert!(params.measurement_probability > 0.0);
    }

    #[test]
    fn test_quantum_walk_params() {
        let params = QuantumWalkParams::default();
        assert_eq!(params.coin_bias, 0.5);
        assert_eq!(params.step_size, 1.0);
        assert_eq!(params.num_steps, 100);
        assert_eq!(params.dimension, 1);
    }

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::default();
        assert_eq!(
            config.algorithm_type,
            OptimizationAlgorithm::QuantumGeneticAlgorithm
        );
        assert_eq!(config.objective_function, ObjectiveFunction::Quadratic);
        assert_eq!(config.bounds.len(), 16);
        assert_eq!(config.constraint_method, ConstraintMethod::PenaltyFunction);
        assert!(!config.multi_objective);
        assert!(config.parallel_evaluation);
    }

    #[test]
    fn test_ml_config() {
        let config = MLConfig::default();
        assert_eq!(
            config.algorithm_type,
            MLAlgorithm::QuantumInspiredNeuralNetwork
        );
        assert_eq!(config.architecture.input_dim, 16);
        assert_eq!(config.architecture.output_dim, 8);
        assert!(config.architecture.quantum_connections);
    }

    #[test]
    fn test_network_architecture() {
        let arch = NetworkArchitecture::default();
        assert_eq!(arch.input_dim, 16);
        assert_eq!(arch.hidden_layers, vec![32, 16]);
        assert_eq!(arch.output_dim, 8);
        assert_eq!(arch.activation, ActivationFunction::QuantumInspiredTanh);
    }

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.optimizer, OptimizerType::QuantumInspiredAdam);
        assert_eq!(config.regularization, 0.001);
    }

    #[test]
    fn test_tensor_network_config() {
        let config = TensorNetworkConfig::default();
        assert_eq!(config.bond_dimension, 64);
        assert_eq!(config.topology, TensorTopology::MPS);
        assert_eq!(
            config.contraction_method,
            ContractionMethod::OptimalContraction
        );
        assert_eq!(config.truncation_threshold, 1e-12);
    }

    #[test]
    fn test_sampling_config() {
        let config = SamplingConfig::default();
        assert_eq!(
            config.algorithm_type,
            SamplingAlgorithm::QuantumInspiredMCMC
        );
        assert_eq!(config.num_samples, 10000);
        assert_eq!(config.burn_in, 1000);
        assert_eq!(config.thinning, 10);
        assert_eq!(config.proposal_distribution, ProposalDistribution::Gaussian);
    }

    #[test]
    fn test_wave_function_config() {
        let config = WaveFunctionConfig::default();
        assert_eq!(config.wave_function_type, WaveFunctionType::SlaterJastrow);
        assert_eq!(config.num_parameters, 32);
        assert_eq!(config.jastrow_strength, 1.0);
        assert!(!config.backflow_enabled);
    }

    #[test]
    fn test_linalg_config() {
        let config = LinalgConfig::default();
        assert_eq!(
            config.algorithm_type,
            LinalgAlgorithm::QuantumInspiredLinearSolver
        );
        assert_eq!(config.matrix_dimension, 1024);
        assert_eq!(config.precision, 1e-8);
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.krylov_dimension, 50);
    }

    #[test]
    fn test_graph_config() {
        let config = GraphConfig::default();
        assert_eq!(
            config.algorithm_type,
            GraphAlgorithm::QuantumInspiredRandomWalk
        );
        assert_eq!(config.num_vertices, 100);
        assert_eq!(config.connectivity, 0.1);
    }

    #[test]
    fn test_community_detection_params() {
        let params = CommunityDetectionParams::default();
        assert_eq!(params.resolution, 1.0);
        assert_eq!(params.num_iterations, 100);
        assert_eq!(params.modularity_threshold, 0.01);
    }

    #[test]
    fn test_benchmarking_config() {
        let config = BenchmarkingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.num_runs, 10);
        assert!(config.compare_classical);
        assert!(config.detailed_metrics);
    }

    #[test]
    fn test_performance_analysis_config() {
        let config = PerformanceAnalysisConfig::default();
        assert!(config.analyze_convergence);
        assert!(config.analyze_scalability);
        assert!(config.analyze_quantum_advantage);
        assert!(config.record_memory_usage);
    }

    #[test]
    fn test_runtime_stats() {
        let stats = RuntimeStats::default();
        assert_eq!(stats.function_evaluations, 0);
        assert_eq!(stats.gradient_evaluations, 0);
        assert_eq!(stats.cpu_time, 0.0);
        assert_eq!(stats.memory_usage, 0);
        assert_eq!(stats.quantum_operations, 0);
    }

    #[test]
    fn test_execution_stats() {
        let stats = ExecutionStats::default();
        assert_eq!(stats.total_runtime, 0.0);
        assert_eq!(stats.avg_runtime_per_iteration, 0.0);
        assert_eq!(stats.peak_memory_usage, 0);
        assert_eq!(stats.successful_runs, 0);
        assert_eq!(stats.failed_runs, 0);
    }

    #[test]
    fn test_comparison_stats() {
        let stats = ComparisonStats::default();
        assert_eq!(stats.quantum_inspired_performance, 0.0);
        assert_eq!(stats.classical_performance, 0.0);
        assert_eq!(stats.speedup_factor, 1.0);
        assert_eq!(stats.solution_quality_ratio, 1.0);
        assert_eq!(stats.convergence_speed_ratio, 1.0);
    }

    #[test]
    fn test_convergence_analysis() {
        let analysis = ConvergenceAnalysis::default();
        assert_eq!(analysis.convergence_rate, 0.0);
        assert_eq!(analysis.iterations_to_convergence, 0);
        assert_eq!(analysis.final_gradient_norm, f64::INFINITY);
        assert!(!analysis.converged);
        assert_eq!(analysis.convergence_criterion, "tolerance");
    }

    #[test]
    fn test_quantum_advantage_metrics() {
        let metrics = QuantumAdvantageMetrics::default();
        assert_eq!(metrics.theoretical_speedup, 1.0);
        assert_eq!(metrics.practical_advantage, 1.0);
        assert_eq!(metrics.complexity_class, "NP");
        assert_eq!(metrics.quantum_resource_requirements, 0);
        assert_eq!(metrics.classical_resource_requirements, 0);
    }

    #[test]
    fn test_objective_function_quadratic() {
        let config = QuantumInspiredConfig {
            optimization_config: OptimizationConfig {
                objective_function: ObjectiveFunction::Quadratic,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let solution = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = framework.evaluate_objective_public(&solution).unwrap();

        // x^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
        assert!((result - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_objective_function_sphere() {
        let config = QuantumInspiredConfig {
            optimization_config: OptimizationConfig {
                objective_function: ObjectiveFunction::Sphere,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let solution = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let result = framework.evaluate_objective_public(&solution).unwrap();

        assert!((result - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_objective_function_rosenbrock() {
        let config = QuantumInspiredConfig {
            optimization_config: OptimizationConfig {
                objective_function: ObjectiveFunction::Rosenbrock,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let solution = Array1::from_vec(vec![1.0, 1.0]); // Global minimum
        let result = framework.evaluate_objective_public(&solution).unwrap();

        assert!(result < 1e-10); // Should be close to 0 at the global minimum
    }

    #[test]
    fn test_objective_function_rastrigin() {
        let config = QuantumInspiredConfig {
            optimization_config: OptimizationConfig {
                objective_function: ObjectiveFunction::Rastrigin,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let solution = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]); // Global minimum
        let result = framework.evaluate_objective_public(&solution).unwrap();

        assert!(result < 1e-10); // Should be close to 0 at the global minimum
    }

    #[test]
    fn test_objective_function_ackley() {
        let config = QuantumInspiredConfig {
            optimization_config: OptimizationConfig {
                objective_function: ObjectiveFunction::Ackley,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let solution = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]); // Global minimum
        let result = framework.evaluate_objective_public(&solution).unwrap();

        assert!(result < 1e-10); // Should be close to 0 at the global minimum
    }

    #[test]
    fn test_quantum_genetic_algorithm() {
        let mut config = QuantumInspiredConfig::default();
        config.optimization_config.algorithm_type = OptimizationAlgorithm::QuantumGeneticAlgorithm;
        config.algorithm_config.max_iterations = 10; // Short test
        config.algorithm_config.population_size = 20;
        config.num_variables = 4;

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let result = framework.optimize();
        assert!(result.is_ok());

        let opt_result = result.unwrap();
        assert!(opt_result.iterations <= 10);
        assert!(opt_result.objective_value.is_finite());
        assert_eq!(opt_result.solution.len(), 4);
        assert!(opt_result.runtime_stats.function_evaluations > 0);
    }

    #[test]
    fn test_quantum_particle_swarm_optimization() {
        let mut config = QuantumInspiredConfig::default();
        config.optimization_config.algorithm_type = OptimizationAlgorithm::QuantumParticleSwarm;
        config.algorithm_config.max_iterations = 10;
        config.algorithm_config.population_size = 20;
        config.num_variables = 4;

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let result = framework.optimize();
        assert!(result.is_ok());

        let opt_result = result.unwrap();
        assert!(opt_result.iterations <= 10);
        assert!(opt_result.objective_value.is_finite());
        assert_eq!(opt_result.solution.len(), 4);
    }

    #[test]
    fn test_quantum_simulated_annealing() {
        let mut config = QuantumInspiredConfig::default();
        config.optimization_config.algorithm_type =
            OptimizationAlgorithm::QuantumSimulatedAnnealing;
        config.algorithm_config.max_iterations = 10;
        config.num_variables = 4;

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let result = framework.optimize();
        assert!(result.is_ok());

        let opt_result = result.unwrap();
        assert!(opt_result.iterations <= 10);
        assert!(opt_result.objective_value.is_finite());
        assert_eq!(opt_result.solution.len(), 4);
    }

    #[test]
    fn test_temperature_schedules() {
        let schedules = vec![
            TemperatureSchedule::Exponential,
            TemperatureSchedule::Linear,
            TemperatureSchedule::Logarithmic,
            TemperatureSchedule::QuantumAdiabatic,
            TemperatureSchedule::Custom,
        ];

        for schedule in schedules {
            let mut config = QuantumInspiredConfig::default();
            config.optimization_config.algorithm_type =
                OptimizationAlgorithm::QuantumSimulatedAnnealing;
            config.algorithm_config.temperature_schedule = schedule;
            config.algorithm_config.max_iterations = 5;
            config.num_variables = 2;

            let mut framework = QuantumInspiredFramework::new(config).unwrap();
            let result = framework.optimize();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_quantum_parameters_effects() {
        // Test with high quantum effects
        let mut config = QuantumInspiredConfig::default();
        config.algorithm_config.quantum_parameters = QuantumParameters {
            superposition_strength: 1.0,
            entanglement_strength: 1.0,
            interference_strength: 1.0,
            tunneling_probability: 0.5,
            decoherence_rate: 0.1,
            measurement_probability: 0.5,
            quantum_walk_params: QuantumWalkParams::default(),
        };
        config.algorithm_config.max_iterations = 5;
        config.num_variables = 4;

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let result = framework.optimize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_optimization_bounds() {
        let mut config = QuantumInspiredConfig::default();
        config.optimization_config.bounds =
            vec![(-5.0, 5.0), (-10.0, 10.0), (-1.0, 1.0), (-2.0, 2.0)];
        config.algorithm_config.max_iterations = 5;
        config.num_variables = 4;

        let bounds = config.optimization_config.bounds.clone();
        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let result = framework.optimize();
        assert!(result.is_ok());

        let opt_result = result.unwrap();
        // Check that solution respects bounds
        for (i, &val) in opt_result.solution.iter().enumerate() {
            let (min_bound, max_bound) = bounds[i];
            assert!(val >= min_bound - 1e-10);
            assert!(val <= max_bound + 1e-10);
        }
    }

    #[test]
    fn test_convergence_detection() {
        let mut config = QuantumInspiredConfig::default();
        config.algorithm_config.tolerance = 1e-3;
        config.algorithm_config.max_iterations = 5;
        config.num_variables = 2;

        let mut framework = QuantumInspiredFramework::new(config).unwrap();

        // Manually set convergence history to test convergence detection
        framework.get_state_mut().convergence_history =
            vec![10.0, 9.0, 8.0, 7.0, 6.999999, 6.999999];
        let converged = framework.check_convergence_public().unwrap();
        assert!(converged);
    }

    #[test]
    fn test_framework_reset() {
        let config = QuantumInspiredConfig::default();
        let mut framework = QuantumInspiredFramework::new(config).unwrap();

        // Modify state
        framework.get_state_mut().iteration = 10;
        framework.get_state_mut().best_objective = 5.0;
        framework.get_state_mut().convergence_history.push(1.0);
        framework.get_state_mut().runtime_stats.function_evaluations = 100;

        // Reset
        framework.reset();

        // Check reset state
        assert_eq!(framework.get_state().iteration, 0);
        assert_eq!(framework.get_state().best_objective, f64::INFINITY);
        assert_eq!(framework.get_state().convergence_history.len(), 0);
        assert_eq!(framework.get_state().runtime_stats.function_evaluations, 0);
    }

    #[test]
    fn test_optimization_result_structure() {
        let mut config = QuantumInspiredConfig::default();
        config.algorithm_config.max_iterations = 3;
        config.num_variables = 2;

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let result = framework.optimize().unwrap();

        assert_eq!(result.solution.len(), 2);
        assert!(result.objective_value.is_finite());
        assert!(result.iterations <= 3);
        assert!(result.runtime_stats.function_evaluations > 0);
    }

    #[test]
    fn test_different_objective_functions() {
        let objectives = vec![
            ObjectiveFunction::Quadratic,
            ObjectiveFunction::Sphere,
            ObjectiveFunction::Rosenbrock,
            ObjectiveFunction::Rastrigin,
            ObjectiveFunction::Ackley,
            ObjectiveFunction::Griewank,
            ObjectiveFunction::Custom,
        ];

        for objective in objectives {
            let mut config = QuantumInspiredConfig::default();
            config.optimization_config.objective_function = objective;
            config.algorithm_config.max_iterations = 3;
            config.num_variables = 4;

            let mut framework = QuantumInspiredFramework::new(config).unwrap();
            let result = framework.optimize();
            assert!(result.is_ok(), "Failed for objective: {objective:?}");
        }
    }

    #[test]
    fn test_ml_training_placeholder() {
        let config = QuantumInspiredConfig {
            algorithm_category: AlgorithmCategory::MachineLearning,
            ..Default::default()
        };

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let training_data = vec![(Array1::zeros(4), Array1::zeros(2))];

        let result = framework.train_ml_model(&training_data);
        // Should return not implemented error for now
        assert!(result.is_err());
    }

    #[test]
    fn test_sampling_placeholder() {
        let config = QuantumInspiredConfig {
            algorithm_category: AlgorithmCategory::Sampling,
            ..Default::default()
        };

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let result = framework.sample();
        // Should return not implemented error for now
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_algebra_placeholder() {
        let config = QuantumInspiredConfig {
            algorithm_category: AlgorithmCategory::LinearAlgebra,
            ..Default::default()
        };

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let matrix = Array2::eye(4);
        let rhs = Array1::from_vec(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]);

        let result = framework.solve_linear_algebra(&matrix, &rhs);
        // Should return not implemented error for now
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_algorithms_placeholder() {
        let config = QuantumInspiredConfig {
            algorithm_category: AlgorithmCategory::GraphAlgorithms,
            ..Default::default()
        };

        let mut framework = QuantumInspiredFramework::new(config).unwrap();
        let adjacency_matrix = Array2::eye(4);

        let result = framework.solve_graph_problem(&adjacency_matrix);
        // Should return not implemented error for now
        assert!(result.is_err());
    }

    #[test]
    fn test_quantum_inspired_utils_optimization_problem() {
        let (obj_func, bounds, optimal) = QuantumInspiredUtils::generate_optimization_problem(
            ObjectiveFunction::Quadratic,
            8,
            (-5.0, 5.0),
        );

        assert_eq!(obj_func, ObjectiveFunction::Quadratic);
        assert_eq!(bounds.len(), 8);
        assert_eq!(optimal.len(), 8);

        for (min_bound, max_bound) in bounds {
            assert_eq!(min_bound, -5.0);
            assert_eq!(max_bound, 5.0);
        }
    }

    #[test]
    fn test_quantum_inspired_utils_convergence_analysis() {
        let convergence_history = vec![100.0, 90.0, 80.0, 70.0, 65.0, 64.9, 64.8, 64.8, 64.8];
        let analysis = QuantumInspiredUtils::analyze_convergence(&convergence_history);

        assert!(analysis.convergence_rate > 0.0);
        assert!(analysis.iterations_to_convergence > 0);
        assert_eq!(analysis.convergence_criterion, "variance");
        assert!(analysis.converged);
    }

    #[test]
    fn test_quantum_inspired_utils_convergence_analysis_no_convergence() {
        let convergence_history = vec![100.0, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0];
        let analysis = QuantumInspiredUtils::analyze_convergence(&convergence_history);

        assert!(analysis.convergence_rate > 0.0);
        assert!(!analysis.converged);
    }

    #[test]
    fn test_quantum_inspired_utils_convergence_analysis_empty() {
        let convergence_history = vec![];
        let analysis = QuantumInspiredUtils::analyze_convergence(&convergence_history);

        assert_eq!(analysis.convergence_rate, 0.0);
        assert_eq!(analysis.iterations_to_convergence, 0);
        assert!(!analysis.converged);
    }

    #[test]
    fn test_quantum_inspired_utils_algorithm_comparison() {
        let results1 = vec![
            OptimizationResult {
                solution: Array1::zeros(4),
                objective_value: 5.0,
                iterations: 10,
                converged: true,
                runtime_stats: RuntimeStats::default(),
                metadata: HashMap::new(),
            },
            OptimizationResult {
                solution: Array1::zeros(4),
                objective_value: 6.0,
                iterations: 12,
                converged: true,
                runtime_stats: RuntimeStats::default(),
                metadata: HashMap::new(),
            },
        ];

        let results2 = vec![
            OptimizationResult {
                solution: Array1::zeros(4),
                objective_value: 10.0,
                iterations: 20,
                converged: true,
                runtime_stats: RuntimeStats::default(),
                metadata: HashMap::new(),
            },
            OptimizationResult {
                solution: Array1::zeros(4),
                objective_value: 12.0,
                iterations: 22,
                converged: true,
                runtime_stats: RuntimeStats::default(),
                metadata: HashMap::new(),
            },
        ];

        let comparison = QuantumInspiredUtils::compare_algorithms(&results1, &results2);

        assert_eq!(comparison.quantum_inspired_performance, 5.5); // (5+6)/2
        assert_eq!(comparison.classical_performance, 11.0); // (10+12)/2
        assert_eq!(comparison.speedup_factor, 2.0); // 11/5.5
    }

    #[test]
    fn test_quantum_inspired_utils_quantum_advantage_estimation() {
        let metrics = QuantumInspiredUtils::estimate_quantum_advantage(
            16,
            OptimizationAlgorithm::QuantumGeneticAlgorithm,
        );

        assert_eq!(metrics.theoretical_speedup, 4.0); // sqrt(16)
        assert_eq!(metrics.practical_advantage, 2.0); // 4.0 * 0.5
        assert_eq!(metrics.complexity_class, "BQP");
        assert_eq!(metrics.quantum_resource_requirements, 160); // 16 * 10
        assert_eq!(metrics.classical_resource_requirements, 256); // 16^2
    }

    #[test]
    fn test_quantum_advantage_estimation_different_algorithms() {
        let algorithms = vec![
            (OptimizationAlgorithm::QuantumGeneticAlgorithm, 4.0), // sqrt(16)
            (OptimizationAlgorithm::QuantumParticleSwarm, 4.0),    // log2(16)
            (OptimizationAlgorithm::QuantumSimulatedAnnealing, 1.0), // default
            (OptimizationAlgorithm::ClassicalQAOA, 256.0),         // 2^(16/2)
        ];

        for (algorithm, expected_speedup) in algorithms {
            let metrics = QuantumInspiredUtils::estimate_quantum_advantage(16, algorithm);
            assert_eq!(metrics.theoretical_speedup, expected_speedup);
        }
    }

    #[test]
    fn test_benchmarking_framework() {
        let mut config = QuantumInspiredConfig::default();
        config.algorithm_config.max_iterations = 3;
        config.benchmarking_config.num_runs = 3;
        config.num_variables = 4;

        let result = benchmark_quantum_inspired_algorithms(&config);
        assert!(result.is_ok());

        let benchmark = result.unwrap();
        assert_eq!(benchmark.execution_times.len(), 3);
        assert_eq!(benchmark.solution_qualities.len(), 3);
        assert_eq!(benchmark.convergence_rates.len(), 3);
        assert_eq!(benchmark.memory_usage.len(), 3);

        // Check statistical analysis
        assert!(benchmark.statistical_analysis.mean_performance.is_finite());
        assert!(benchmark.statistical_analysis.std_deviation >= 0.0);
        assert!(benchmark.statistical_analysis.effect_size.is_finite());
    }

    #[test]
    fn test_benchmarking_different_algorithms() {
        let algorithms = vec![
            OptimizationAlgorithm::QuantumGeneticAlgorithm,
            OptimizationAlgorithm::QuantumParticleSwarm,
            OptimizationAlgorithm::QuantumSimulatedAnnealing,
        ];

        for algorithm in algorithms {
            let mut config = QuantumInspiredConfig::default();
            config.optimization_config.algorithm_type = algorithm;
            config.algorithm_config.max_iterations = 2;
            config.benchmarking_config.num_runs = 2;
            config.num_variables = 2;

            let result = benchmark_quantum_inspired_algorithms(&config);
            assert!(result.is_ok(), "Failed for algorithm: {algorithm:?}");
        }
    }

    #[test]
    fn test_statistical_analysis() {
        let analysis = StatisticalAnalysis {
            mean_performance: 10.0,
            std_deviation: 2.0,
            confidence_intervals: (6.08, 13.92), // 10 ± 1.96*2
            p_value: 0.05,
            effect_size: 5.0, // 10/2
        };

        assert_eq!(analysis.mean_performance, 10.0);
        assert_eq!(analysis.std_deviation, 2.0);
        assert_eq!(analysis.effect_size, 5.0);
    }

    #[test]
    fn test_error_handling_unimplemented_algorithms() {
        let unimplemented_algorithms = vec![
            OptimizationAlgorithm::QuantumDifferentialEvolution,
            OptimizationAlgorithm::ClassicalQAOA,
            OptimizationAlgorithm::ClassicalVQE,
            OptimizationAlgorithm::QuantumAntColony,
            OptimizationAlgorithm::QuantumHarmonySearch,
        ];

        for algorithm in unimplemented_algorithms {
            let mut config = QuantumInspiredConfig::default();
            config.optimization_config.algorithm_type = algorithm;
            config.num_variables = 4;

            let mut framework = QuantumInspiredFramework::new(config).unwrap();
            let result = framework.optimize();

            assert!(
                result.is_err(),
                "Expected error for unimplemented algorithm: {algorithm:?}"
            );
            if let Err(e) = result {
                match e {
                    crate::error::SimulatorError::NotImplemented(_) => {
                        // Expected error type
                    }
                    _ => panic!("Unexpected error type for unimplemented algorithm"),
                }
            }
        }
    }

    #[test]
    fn test_framework_statistics() {
        let config = QuantumInspiredConfig::default();
        let framework = QuantumInspiredFramework::new(config).unwrap();

        let stats = framework.get_stats();
        assert_eq!(stats.execution_stats.total_runtime, 0.0);
        assert_eq!(stats.comparison_stats.speedup_factor, 1.0);
        assert!(!stats.convergence_analysis.converged);
        assert_eq!(stats.quantum_advantage_metrics.theoretical_speedup, 1.0);
    }

    #[test]
    fn test_different_algorithm_categories() {
        let categories = vec![
            AlgorithmCategory::Optimization,
            AlgorithmCategory::MachineLearning,
            AlgorithmCategory::Sampling,
            AlgorithmCategory::LinearAlgebra,
            AlgorithmCategory::GraphAlgorithms,
            AlgorithmCategory::HybridQuantumClassical,
        ];

        for category in categories {
            let config = QuantumInspiredConfig {
                algorithm_category: category,
                ..Default::default()
            };

            let framework = QuantumInspiredFramework::new(config);
            assert!(framework.is_ok(), "Failed for category: {category:?}");
        }
    }

    #[test]
    fn test_comprehensive_configuration() {
        let config = QuantumInspiredConfig {
            num_variables: 32,
            algorithm_category: AlgorithmCategory::Optimization,
            algorithm_config: AlgorithmConfig {
                max_iterations: 500,
                tolerance: 1e-8,
                population_size: 50,
                elite_ratio: 0.2,
                mutation_rate: 0.05,
                crossover_rate: 0.9,
                temperature_schedule: TemperatureSchedule::QuantumAdiabatic,
                quantum_parameters: QuantumParameters {
                    superposition_strength: 0.8,
                    entanglement_strength: 0.6,
                    interference_strength: 0.4,
                    tunneling_probability: 0.2,
                    decoherence_rate: 0.05,
                    measurement_probability: 0.3,
                    quantum_walk_params: QuantumWalkParams {
                        coin_bias: 0.7,
                        step_size: 1.5,
                        num_steps: 200,
                        dimension: 2,
                    },
                },
            },
            optimization_config: OptimizationConfig {
                algorithm_type: OptimizationAlgorithm::QuantumGeneticAlgorithm,
                objective_function: ObjectiveFunction::Rastrigin,
                bounds: vec![(-10.0, 10.0); 32],
                constraint_method: ConstraintMethod::BarrierFunction,
                multi_objective: true,
                parallel_evaluation: true,
            },
            ml_config: Some(MLConfig {
                algorithm_type: MLAlgorithm::TensorNetworkML,
                architecture: NetworkArchitecture {
                    input_dim: 32,
                    hidden_layers: vec![64, 32, 16],
                    output_dim: 8,
                    activation: ActivationFunction::QuantumPhase,
                    quantum_connections: true,
                },
                training_config: TrainingConfig {
                    learning_rate: 0.001,
                    epochs: 200,
                    batch_size: 64,
                    optimizer: OptimizerType::QuantumNaturalGradient,
                    regularization: 0.01,
                },
                tensor_network_config: TensorNetworkConfig {
                    bond_dimension: 128,
                    topology: TensorTopology::PEPS,
                    contraction_method: ContractionMethod::BranchAndBound,
                    truncation_threshold: 1e-14,
                },
            }),
            sampling_config: SamplingConfig {
                algorithm_type: SamplingAlgorithm::QuantumInspiredVMC,
                num_samples: 50000,
                burn_in: 5000,
                thinning: 5,
                proposal_distribution: ProposalDistribution::QuantumInspired,
                wave_function_config: WaveFunctionConfig {
                    wave_function_type: WaveFunctionType::QuantumNeuralNetwork,
                    num_parameters: 128,
                    jastrow_strength: 2.0,
                    backflow_enabled: true,
                },
            },
            linalg_config: LinalgConfig {
                algorithm_type: LinalgAlgorithm::QuantumInspiredSVD,
                matrix_dimension: 2048,
                precision: 1e-12,
                max_iterations: 2000,
                krylov_dimension: 100,
            },
            graph_config: GraphConfig {
                algorithm_type: GraphAlgorithm::QuantumInspiredCommunityDetection,
                num_vertices: 500,
                connectivity: 0.05,
                walk_params: QuantumWalkParams {
                    coin_bias: 0.6,
                    step_size: 2.0,
                    num_steps: 1000,
                    dimension: 3,
                },
                community_params: CommunityDetectionParams {
                    resolution: 1.5,
                    num_iterations: 500,
                    modularity_threshold: 0.001,
                },
            },
            benchmarking_config: BenchmarkingConfig {
                enabled: true,
                num_runs: 20,
                compare_classical: true,
                detailed_metrics: true,
                performance_analysis: PerformanceAnalysisConfig {
                    analyze_convergence: true,
                    analyze_scalability: true,
                    analyze_quantum_advantage: true,
                    record_memory_usage: true,
                },
            },
            enable_quantum_heuristics: true,
            precision: 1e-12,
            random_seed: Some(12345),
        };

        let framework = QuantumInspiredFramework::new(config);
        assert!(framework.is_ok());

        let framework = framework.unwrap();
        assert_eq!(framework.get_state().variables.len(), 32);
    }
}
