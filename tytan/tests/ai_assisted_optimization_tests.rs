//! Comprehensive tests for AI-assisted quantum optimization.

use ndarray::{Array1, Array2};
use quantrs2_tytan::ai_assisted_optimization::*;
use std::collections::HashMap;
use std::time::Duration;

#[test]
fn test_ai_optimizer_creation() {
    let mut config = AIOptimizerConfig::default();
    let mut optimizer = AIAssistedOptimizer::new(config);

    // Test that optimizer is created successfully
    assert!(true);
}

#[test]
fn test_ai_optimization_small_problem() {
    let config = AIOptimizerConfig {
        parameter_optimization_enabled: true,
        reinforcement_learning_enabled: false, // Disabled for faster testing
        auto_algorithm_selection_enabled: true,
        structure_recognition_enabled: true,
        quality_prediction_enabled: true,
        learning_rate: 0.01,
        batch_size: 16,
        max_training_iterations: 100,
        convergence_threshold: 1e-4,
        replay_buffer_size: 1000,
    };

    let mut optimizer = AIAssistedOptimizer::new(config);

    // Create a small test QUBO problem
    let mut qubo = Array2::zeros((6, 6));

    // Create a structured problem with two blocks
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                qubo[[i, j]] = -1.0; // Strong coupling within first block
            }
        }
    }
    for i in 3..6 {
        for j in 3..6 {
            if i != j {
                qubo[[i, j]] = -1.0; // Strong coupling within second block
            }
        }
    }

    // Weak coupling between blocks
    qubo[[2, 3]] = 0.1;
    qubo[[3, 2]] = 0.1;

    let result = optimizer.optimize(&qubo, Some(0.9), Some(Duration::from_secs(30)));
    assert!(result.is_ok());

    let optimization_result = result.unwrap();

    // Verify basic result structure
    assert_eq!(optimization_result.problem_info.size, 6);
    assert!(!optimization_result.recommended_algorithm.is_empty());
    assert!(optimization_result.confidence >= 0.0);
    assert!(optimization_result.confidence <= 1.0);

    // Verify structure recognition worked
    assert!(!optimization_result
        .problem_info
        .structure_patterns
        .is_empty());
    let has_block_pattern = optimization_result
        .problem_info
        .structure_patterns
        .iter()
        .any(|p| matches!(p, StructurePattern::Block { .. }));
    assert!(has_block_pattern);

    // Verify quality prediction
    assert!(optimization_result.predicted_quality.expected_quality >= 0.0);
    assert!(optimization_result.predicted_quality.expected_quality <= 1.0);
    assert!(
        optimization_result.predicted_quality.confidence_interval.0
            <= optimization_result.predicted_quality.expected_quality
    );
    assert!(
        optimization_result.predicted_quality.expected_quality
            <= optimization_result.predicted_quality.confidence_interval.1
    );

    // Verify alternatives are provided
    assert!(!optimization_result.alternatives.is_empty());

    println!("Small problem AI optimization completed successfully");
    println!(
        "Recommended algorithm: {}",
        optimization_result.recommended_algorithm
    );
    println!(
        "Predicted quality: {:.3}",
        optimization_result.predicted_quality.expected_quality
    );
    println!("Confidence: {:.3}", optimization_result.confidence);
}

#[test]
fn test_problem_feature_extraction() {
    let mut config = AIOptimizerConfig::default();
    let mut optimizer = AIAssistedOptimizer::new(config);

    // Test different problem types

    // 1. Dense problem
    let mut dense_qubo = Array2::ones((5, 5));
    let dense_features = optimizer.extract_problem_features(&dense_qubo).unwrap();
    assert_eq!(dense_features[0], 5.0); // Size
    assert!(dense_features[4] > 0.9); // High density

    // 2. Sparse problem
    let mut sparse_qubo = Array2::zeros((8, 8));
    sparse_qubo[[0, 1]] = 1.0;
    sparse_qubo[[1, 0]] = 1.0;
    sparse_qubo[[2, 3]] = 1.0;
    sparse_qubo[[3, 2]] = 1.0;

    let sparse_features = optimizer.extract_problem_features(&sparse_qubo).unwrap();
    assert_eq!(sparse_features[0], 8.0); // Size
    assert!(sparse_features[4] < 0.5); // Low density

    // 3. Symmetric problem
    let mut symmetric_qubo = Array2::zeros((4, 4));
    symmetric_qubo[[0, 1]] = 2.0;
    symmetric_qubo[[1, 0]] = 2.0;
    symmetric_qubo[[2, 3]] = -1.0;
    symmetric_qubo[[3, 2]] = -1.0;

    let symmetric_features = optimizer.extract_problem_features(&symmetric_qubo).unwrap();
    assert_eq!(symmetric_features[8], 1.0); // Symmetric flag should be 1.0

    println!("Feature extraction tests passed");
}

#[test]
fn test_structure_recognition() {
    let mut recognizer = ProblemStructureRecognizer::new();

    // Test grid structure recognition
    let mut grid_qubo = Array2::zeros((9, 9)); // 3x3 grid

    // Add grid edges
    for i in 0..3 {
        for j in 0..3 {
            let idx = i * 3 + j;

            // Right neighbor
            if j < 2 {
                let right_idx = i * 3 + (j + 1);
                grid_qubo[[idx, right_idx]] = -1.0;
                grid_qubo[[right_idx, idx]] = -1.0;
            }

            // Bottom neighbor
            if i < 2 {
                let bottom_idx = (i + 1) * 3 + j;
                grid_qubo[[idx, bottom_idx]] = -1.0;
                grid_qubo[[bottom_idx, idx]] = -1.0;
            }
        }
    }

    let grid_patterns = recognizer.recognize_structure(&grid_qubo).unwrap();
    let has_grid = grid_patterns
        .iter()
        .any(|p| matches!(p, StructurePattern::Grid { .. }));
    assert!(has_grid);

    // Test block structure recognition
    let mut block_qubo = Array2::zeros((6, 6));

    // Two blocks of size 3
    for block in 0..2 {
        let start = block * 3;
        let end = (block + 1) * 3;

        for i in start..end {
            for j in start..end {
                if i != j {
                    block_qubo[[i, j]] = -2.0;
                }
            }
        }
    }

    let block_patterns = recognizer.recognize_structure(&block_qubo).unwrap();
    let has_block = block_patterns
        .iter()
        .any(|p| matches!(p, StructurePattern::Block { .. }));
    assert!(has_block);

    // Test chain structure recognition
    let mut chain_qubo = Array2::zeros((5, 5));
    for i in 0..4 {
        chain_qubo[[i, i + 1]] = -1.0;
        chain_qubo[[i + 1, i]] = -1.0;
    }

    let chain_patterns = recognizer.recognize_structure(&chain_qubo).unwrap();
    let has_chain = chain_patterns
        .iter()
        .any(|p| matches!(p, StructurePattern::Chain { .. }));
    assert!(has_chain);

    println!("Structure recognition tests passed");
}

#[test]
fn test_algorithm_selection() {
    let mut config = AIOptimizerConfig::default();
    let mut selector = AutomatedAlgorithmSelector::new(&config);

    // Test selection for different problem characteristics

    // Small problem
    let mut small_features = Array1::from(vec![50.0, 0.0, 1.0, 1.0, 0.3, 5.0, 1.0, 3.0, 1.0, 0.5]);
    let small_patterns = vec![];
    let small_alg = selector
        .select_algorithm(&small_features, &small_patterns)
        .unwrap();
    assert_eq!(small_alg, "BranchAndBound"); // Should prefer exact methods for small problems

    // Dense problem
    let mut dense_features = Array1::from(vec![200.0, 0.0, 1.0, 1.0, 0.9, 5.0, 1.0, 8.0, 1.0, 2.0]);
    let dense_patterns = vec![];
    let dense_alg = selector
        .select_algorithm(&dense_features, &dense_patterns)
        .unwrap();
    assert_eq!(dense_alg, "SimulatedAnnealing"); // Should prefer SA for dense problems

    // Tree-structured problem
    let mut tree_features = Array1::from(vec![100.0, 0.0, 1.0, 1.0, 0.2, 3.0, 1.0, 2.5, 1.0, 1.0]);
    let tree_patterns = vec![StructurePattern::Tree {
        depth: 5,
        branching_factor: 2.0,
    }];
    let tree_alg = selector
        .select_algorithm(&tree_features, &tree_patterns)
        .unwrap();
    assert_eq!(tree_alg, "DynamicProgramming"); // Should prefer DP for tree structures

    // Large general problem
    let mut large_features =
        Array1::from(vec![1000.0, 0.0, 2.0, 1.5, 0.4, 8.0, 2.0, 6.0, 1.0, 3.0]);
    let large_patterns = vec![];
    let large_alg = selector
        .select_algorithm(&large_features, &large_patterns)
        .unwrap();
    assert_eq!(large_alg, "GeneticAlgorithm"); // Should prefer GA for large problems

    println!("Algorithm selection tests passed");
}

#[test]
fn test_parameter_optimization() {
    let mut config = AIOptimizerConfig::default();
    let mut optimizer_net = ParameterOptimizationNetwork::new(&config);

    // Test parameter optimization for different algorithms
    let mut features = Array1::from(vec![100.0, 0.0, 1.0, 1.0, 0.5, 5.0, 1.0, 4.0, 1.0, 1.5]);

    // Test Simulated Annealing parameters
    let sa_params = optimizer_net
        .optimize_parameters(&features, "SimulatedAnnealing", Some(0.9))
        .unwrap();
    assert!(sa_params.contains_key("initial_temperature"));
    assert!(sa_params.contains_key("cooling_rate"));
    assert!(sa_params.contains_key("min_temperature"));

    let initial_temp = sa_params["initial_temperature"];
    let cooling_rate = sa_params["cooling_rate"];
    let min_temp = sa_params["min_temperature"];

    assert!(initial_temp > 0.0);
    assert!(cooling_rate > 0.0 && cooling_rate < 1.0);
    assert!(min_temp > 0.0 && min_temp < initial_temp);

    // Test Genetic Algorithm parameters
    let ga_params = optimizer_net
        .optimize_parameters(&features, "GeneticAlgorithm", Some(0.85))
        .unwrap();
    assert!(ga_params.contains_key("population_size"));
    assert!(ga_params.contains_key("mutation_rate"));
    assert!(ga_params.contains_key("crossover_rate"));

    let pop_size = ga_params["population_size"];
    let mutation_rate = ga_params["mutation_rate"];
    let crossover_rate = ga_params["crossover_rate"];

    assert!(pop_size > 0.0);
    assert!(mutation_rate >= 0.0 && mutation_rate <= 1.0);
    assert!(crossover_rate >= 0.0 && crossover_rate <= 1.0);

    println!("Parameter optimization tests passed");
}

#[test]
fn test_quality_prediction() {
    let mut config = AIOptimizerConfig::default();
    let mut predictor = SolutionQualityPredictor::new(&config);

    // Test quality prediction for different scenarios
    let mut features = Array1::from(vec![50.0, 0.0, 1.0, 1.0, 0.3, 3.0, 1.0, 2.5, 1.0, 1.0]);
    let mut params = HashMap::new();

    // Test predictions for different algorithms
    let algorithms = vec!["SimulatedAnnealing", "GeneticAlgorithm", "TabuSearch"];

    for algorithm in algorithms {
        let prediction = predictor
            .predict_quality(&features, algorithm, &params)
            .unwrap();

        // Verify prediction structure
        assert!(prediction.expected_quality >= 0.0);
        assert!(prediction.expected_quality <= 1.0);
        assert!(prediction.confidence_interval.0 <= prediction.expected_quality);
        assert!(prediction.expected_quality <= prediction.confidence_interval.1);
        assert!(prediction.optimal_probability >= 0.0);
        assert!(prediction.optimal_probability <= 1.0);
        assert!(prediction.expected_convergence_time > Duration::ZERO);

        println!(
            "Algorithm {}: quality = {:.3}, confidence interval = [{:.3}, {:.3}]",
            algorithm,
            prediction.expected_quality,
            prediction.confidence_interval.0,
            prediction.confidence_interval.1
        );
    }

    println!("Quality prediction tests passed");
}

#[test]
fn test_difficulty_assessment() {
    let mut config = AIOptimizerConfig::default();
    let mut optimizer = AIAssistedOptimizer::new(config);

    // Test different difficulty levels

    // Easy problem (small, sparse)
    let mut easy_qubo = Array2::zeros((10, 10));
    easy_qubo[[0, 1]] = 1.0;
    easy_qubo[[1, 0]] = 1.0;

    let easy_features = optimizer.extract_problem_features(&easy_qubo).unwrap();
    let easy_patterns = vec![];
    let easy_assessment = optimizer
        .assess_difficulty(&easy_qubo, &easy_features, &easy_patterns)
        .unwrap();

    assert!(easy_assessment.difficulty_score < 0.5);
    assert!(!easy_assessment.recommended_resources.gpu_recommended);
    assert!(
        !easy_assessment
            .recommended_resources
            .distributed_recommended
    );

    // Hard problem (large, dense, high frustration)
    let size = 100;
    let mut hard_qubo = Array2::ones((size, size));
    for i in 0..size {
        hard_qubo[[i, i]] = 0.0; // Remove self-loops
    }

    let hard_features = optimizer.extract_problem_features(&hard_qubo).unwrap();
    let hard_patterns = vec![];
    let hard_assessment = optimizer
        .assess_difficulty(&hard_qubo, &hard_features, &hard_patterns)
        .unwrap();

    assert!(hard_assessment.difficulty_score > 0.5);
    assert!(hard_assessment.recommended_resources.cpu_cores >= 4);
    assert!(hard_assessment.expected_solution_time > Duration::from_secs(1));

    println!("Difficulty assessment tests passed");
    println!(
        "Easy problem difficulty: {:.3}",
        easy_assessment.difficulty_score
    );
    println!(
        "Hard problem difficulty: {:.3}",
        hard_assessment.difficulty_score
    );
}

#[test]
fn test_training_components() {
    let config = AIOptimizerConfig {
        max_training_iterations: 10, // Reduced for testing
        batch_size: 8,
        ..Default::default()
    };
    let mut optimizer = AIAssistedOptimizer::new(config);

    // Create mock training data
    let mut training_data = Vec::new();

    for i in 0..20 {
        let size = 10 + i * 5;
        let mut features = Array1::from(vec![
            size as f64,
            0.0,
            1.0,
            1.0,
            0.3,
            3.0,
            1.0,
            2.5,
            1.0,
            1.0,
        ]);

        let mut algorithm_scores = HashMap::new();
        algorithm_scores.insert("SimulatedAnnealing".to_string(), 0.8);
        algorithm_scores.insert("GeneticAlgorithm".to_string(), 0.75);
        algorithm_scores.insert("TabuSearch".to_string(), 0.85);

        training_data.push(TrainingExample {
            features,
            optimal_algorithm: "TabuSearch".to_string(),
            algorithm_scores,
            metadata: ProblemMetadata {
                problem_type: "Test".to_string(),
                size,
                density: 0.3,
                source: "Generated".to_string(),
                difficulty_level: DifficultyLevel::Medium,
            },
        });
    }

    // Test training
    let training_results = optimizer.train(&training_data, 0.2);
    assert!(training_results.is_ok());

    let results = training_results.unwrap();

    // Verify training results structure
    if let Some(param_results) = results.parameter_optimizer_results {
        assert!(param_results.final_loss >= 0.0);
        assert!(param_results.training_time > Duration::ZERO);
    }

    if let Some(selector_results) = results.algorithm_selector_results {
        assert!(selector_results.accuracy >= 0.0);
        assert!(selector_results.accuracy <= 1.0);
        assert!(!selector_results.cross_validation_scores.is_empty());
    }

    if let Some(predictor_results) = results.quality_predictor_results {
        assert!(predictor_results.r2_score >= -1.0); // R² can be negative for bad models
        assert!(predictor_results.mae >= 0.0);
        assert!(predictor_results.rmse >= 0.0);
    }

    println!("Training components tests passed");
}

#[test]
fn test_comprehensive_optimization_workflow() {
    let config = AIOptimizerConfig {
        parameter_optimization_enabled: true,
        reinforcement_learning_enabled: false, // Skip RL for faster testing
        auto_algorithm_selection_enabled: true,
        structure_recognition_enabled: true,
        quality_prediction_enabled: true,
        learning_rate: 0.01,
        batch_size: 16,
        max_training_iterations: 50,
        convergence_threshold: 1e-4,
        replay_buffer_size: 1000,
    };

    let mut optimizer = AIAssistedOptimizer::new(config);

    // Test workflow on different problem types
    let test_cases = vec![
        ("Small Dense", create_small_dense_qubo()),
        ("Medium Sparse", create_medium_sparse_qubo()),
        ("Large Structured", create_large_structured_qubo()),
    ];

    for (name, qubo) in test_cases {
        println!("\nTesting workflow on: {}", name);

        let result = optimizer.optimize(&qubo, Some(0.8), Some(Duration::from_secs(60)));
        assert!(result.is_ok(), "Optimization failed for {}", name);

        let optimization_result = result.unwrap();

        // Verify comprehensive results
        assert!(!optimization_result.recommended_algorithm.is_empty());
        assert!(optimization_result.confidence > 0.0);
        assert!(optimization_result.predicted_quality.expected_quality > 0.0);
        assert!(!optimization_result.alternatives.is_empty());

        // Verify problem type inference
        assert!(!optimization_result.problem_info.problem_type.is_empty());

        // Verify difficulty assessment is reasonable
        let difficulty = &optimization_result.problem_info.difficulty_assessment;
        assert!(difficulty.difficulty_score >= 0.0);
        assert!(difficulty.difficulty_score <= 1.0);
        assert!(difficulty.expected_solution_time > Duration::ZERO);

        println!("  Algorithm: {}", optimization_result.recommended_algorithm);
        println!("  Confidence: {:.3}", optimization_result.confidence);
        println!(
            "  Predicted quality: {:.3}",
            optimization_result.predicted_quality.expected_quality
        );
        println!("  Difficulty: {:.3}", difficulty.difficulty_score);
        println!(
            "  Problem type: {}",
            optimization_result.problem_info.problem_type
        );
    }

    println!("\nComprehensive optimization workflow tests passed");
}

#[test]
fn test_reinforcement_learning_components() {
    let mut config = AIOptimizerConfig::default();
    let mut rl_agent = SamplingStrategyAgent::new(&config);

    // Test basic RL agent structure
    assert_eq!(rl_agent.q_network().state_encoder.embedding_dim, 64);
    assert_eq!(rl_agent.q_network().action_decoder.action_dim, 10);
    assert_eq!(rl_agent.replay_buffer().max_size, config.replay_buffer_size);
    assert_eq!(rl_agent.training_stats().episodes, 0);

    // Test training (simplified)
    let training_data = vec![]; // Empty for this test
    let rl_results = rl_agent.train(&training_data);
    assert!(rl_results.is_ok());

    let results = rl_results.unwrap();
    assert!(results.episodes > 0);
    assert!(results.total_steps > 0);
    assert!(!results.loss_history.is_empty());

    println!("Reinforcement learning components tests passed");
}

#[test]
fn test_activation_functions() {
    use ActivationFunction::*;

    // Test that activation functions can be created and used
    let activations = vec![
        ReLU,
        Tanh,
        Sigmoid,
        LeakyReLU { alpha: 0.01 },
        ELU { alpha: 1.0 },
        Swish,
    ];

    for activation in activations {
        // Just test that they can be created and cloned
        let _cloned = activation.clone();
        println!("Activation function: {:?}", activation);
    }

    println!("Activation functions tests passed");
}

#[test]
fn test_ensemble_methods() {
    use EnsembleMethod::*;

    // Test different ensemble methods
    let methods = vec![Voting, Bagging, Boosting, WeightedAverage, DynamicSelection];

    for method in methods {
        let _cloned = method.clone();
        println!("Ensemble method: {:?}", method);
    }

    // Test stacking with nested box
    let stacking = Stacking {
        meta_learner: Box::new(RegressionModel::LinearRegression),
    };
    let _cloned = stacking.clone();

    println!("Ensemble methods tests passed");
}

// Helper functions to create test problems

fn create_small_dense_qubo() -> Array2<f64> {
    let mut qubo = Array2::ones((8, 8));
    for i in 0..8 {
        qubo[[i, i]] = -2.0; // Encourage variables to be 1
    }
    qubo
}

fn create_medium_sparse_qubo() -> Array2<f64> {
    let mut qubo = Array2::zeros((20, 20));

    // Add sparse connections
    for i in 0..19 {
        qubo[[i, i + 1]] = -1.0;
        qubo[[i + 1, i]] = -1.0;
    }

    // Add some random connections
    qubo[[0, 10]] = 0.5;
    qubo[[10, 0]] = 0.5;
    qubo[[5, 15]] = -0.5;
    qubo[[15, 5]] = -0.5;

    qubo
}

fn create_large_structured_qubo() -> Array2<f64> {
    let size = 50;
    let mut qubo = Array2::zeros((size, size));

    // Create block structure with 5 blocks of 10 variables each
    for block in 0..5 {
        let start = block * 10;
        let end = (block + 1) * 10;

        for i in start..end {
            for j in start..end {
                if i != j {
                    qubo[[i, j]] = -1.0; // Strong intra-block coupling
                }
            }
        }
    }

    // Add weak inter-block coupling
    for i in 0..size - 1 {
        if i % 10 == 9 {
            // Connect last element of each block to first of next
            qubo[[i, i + 1]] = 0.1;
            qubo[[i + 1, i]] = 0.1;
        }
    }

    qubo
}
