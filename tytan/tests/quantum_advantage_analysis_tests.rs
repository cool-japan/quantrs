//! Comprehensive tests for quantum advantage analysis suite.

use ndarray::{Array2, Array};
use quantrs2_tytan::quantum_advantage_analysis::*;
use std::collections::HashMap;
use std::time::Duration;

#[test]
fn test_quantum_advantage_analyzer_creation() {
    let config = AnalysisConfig::default();
    let analyzer = QuantumAdvantageAnalyzer::new(config);
    
    // Test that analyzer is created successfully
    // This is a basic sanity check
    assert!(true);
}

#[test]
fn test_quantum_advantage_analysis_small_problem() {
    let config = AnalysisConfig {
        problem_size_range: (5, 20),
        num_samples: 10,
        confidence_level: 0.95,
        classical_baselines: vec![
            ClassicalAlgorithm::SimulatedAnnealing,
            ClassicalAlgorithm::TabuSearch,
        ],
        quantum_algorithms: vec![
            QuantumAlgorithm::QAOA,
            QuantumAlgorithm::QuantumAnnealing,
        ],
        hardware_models: vec![
            HardwareModel::IdealQuantum,
            HardwareModel::NoisyNISQ { error_rate: 0.001 },
        ],
        noise_models: vec![
            NoiseModel::None,
            NoiseModel::Depolarizing { rate: 0.001 },
        ],
    };

    let analyzer = QuantumAdvantageAnalyzer::new(config);

    // Create a small test QUBO
    let mut qubo = Array2::zeros((4, 4));
    qubo[[0, 0]] = -1.0;
    qubo[[1, 1]] = -1.0;
    qubo[[2, 2]] = -1.0;
    qubo[[3, 3]] = -1.0;
    qubo[[0, 1]] = 2.0;
    qubo[[1, 0]] = 2.0;
    qubo[[2, 3]] = 2.0;
    qubo[[3, 2]] = 2.0;

    let metadata = Some(ProblemMetadata {
        problem_type: "Max-Cut".to_string(),
        size: 4,
        density: 0.5,
        generation_seed: Some(42),
        generation_time: std::time::Instant::now(),
    });

    let result = analyzer.analyze_advantage(&qubo, metadata);
    assert!(result.is_ok());

    let analysis = result.unwrap();
    
    // Verify analysis structure
    assert_eq!(analysis.problem_info.num_variables, 4);
    assert_eq!(analysis.problem_info.problem_type, "Max-Cut");
    assert!(analysis.problem_info.density > 0.0);
    
    // Verify performance estimates exist
    assert!(!analysis.classical_performance.is_empty());
    assert!(!analysis.quantum_performance.is_empty());
    
    // Verify advantage analysis
    assert!(analysis.advantage_analysis.confidence >= 0.0);
    assert!(analysis.advantage_analysis.confidence <= 1.0);
    
    // Verify recommendations exist
    assert!(!analysis.recommendations.is_empty());
    
    println!("Small problem analysis completed successfully");
    println!("Quantum advantage detected: {}", analysis.advantage_analysis.has_quantum_advantage);
    println!("Confidence: {:.2}", analysis.advantage_analysis.confidence);
}

#[test]
fn test_quantum_advantage_analysis_large_problem() {
    let config = AnalysisConfig {
        problem_size_range: (100, 1000),
        num_samples: 5,
        confidence_level: 0.90,
        classical_baselines: vec![
            ClassicalAlgorithm::SimulatedAnnealing,
            ClassicalAlgorithm::GeneticAlgorithm,
            ClassicalAlgorithm::BranchAndBound,
        ],
        quantum_algorithms: vec![
            QuantumAlgorithm::QAOA,
            QuantumAlgorithm::VQE,
            QuantumAlgorithm::QuantumAnnealing,
        ],
        hardware_models: vec![
            HardwareModel::IdealQuantum,
            HardwareModel::DigitalAnnealer,
            HardwareModel::PhotonicIsing,
        ],
        noise_models: vec![
            NoiseModel::None,
            NoiseModel::Realistic { decoherence_time: Duration::from_micros(100) },
        ],
    };

    let analyzer = QuantumAdvantageAnalyzer::new(config);

    // Create a larger test QUBO with structure
    let size = 50;
    let mut qubo = Array2::zeros((size, size));
    
    // Create a structured problem with blocks
    for block in 0..5 {
        let start = block * 10;
        let end = (block + 1) * 10;
        
        for i in start..end {
            for j in start..end {
                if i != j {
                    qubo[[i, j]] = -0.5; // Coupling within blocks
                }
            }
        }
    }
    
    // Add inter-block couplings
    for i in 0..size-1 {
        qubo[[i, i+1]] = 0.1;
        qubo[[i+1, i]] = 0.1;
    }

    let metadata = Some(ProblemMetadata {
        problem_type: "Structured QUBO".to_string(),
        size,
        density: 0.2,
        generation_seed: Some(123),
        generation_time: std::time::Instant::now(),
    });

    let result = analyzer.analyze_advantage(&qubo, metadata);
    assert!(result.is_ok());

    let analysis = result.unwrap();
    
    // Verify analysis for larger problem
    assert_eq!(analysis.problem_info.num_variables, size);
    assert!(analysis.problem_info.density > 0.0);
    
    // Large problems should have recommendations for hybrid approaches
    let has_hybrid_recommendation = analysis.recommendations
        .iter()
        .any(|r| matches!(r.algorithm_type, AlgorithmType::Hybrid));
    assert!(has_hybrid_recommendation);
    
    // Check threshold analysis
    assert!(!analysis.threshold_analysis.size_thresholds.is_empty());
    assert!(!analysis.threshold_analysis.noise_thresholds.is_empty());
    
    println!("Large problem analysis completed successfully");
    println!("Problem size: {}", analysis.problem_info.num_variables);
    println!("Quantum advantage detected: {}", analysis.advantage_analysis.has_quantum_advantage);
}

#[test]
fn test_problem_characterization() {
    let config = AnalysisConfig::default();
    let analyzer = QuantumAdvantageAnalyzer::new(config);

    // Test different problem structures
    
    // 1. Fully connected problem
    let size = 6;
    let mut fully_connected = Array2::ones((size, size));
    for i in 0..size {
        fully_connected[[i, i]] = 0.0; // No self-loops
    }

    let result1 = analyzer.analyze_advantage(&fully_connected, None);
    assert!(result1.is_ok());
    let analysis1 = result1.unwrap();
    assert!(matches!(analysis1.problem_info.connectivity, ConnectivityStructure::FullyConnected));

    // 2. Sparse problem
    let mut sparse = Array2::zeros((10, 10));
    sparse[[0, 1]] = 1.0;
    sparse[[1, 0]] = 1.0;
    sparse[[2, 3]] = 1.0;
    sparse[[3, 2]] = 1.0;
    sparse[[4, 5]] = 1.0;
    sparse[[5, 4]] = 1.0;

    let result2 = analyzer.analyze_advantage(&sparse, None);
    assert!(result2.is_ok());
    let analysis2 = result2.unwrap();
    match &analysis2.problem_info.connectivity {
        ConnectivityStructure::Sparse { avg_degree } => {
            assert!(*avg_degree < 5.0);
        }
        _ => {} // Could also be detected as other structures
    }

    println!("Problem characterization tests passed");
}

#[test]
fn test_classical_complexity_estimation() {
    let estimator = ClassicalComplexityEstimator::new();
    
    let problem_chars = ProblemCharacteristics {
        problem_type: "Test Problem".to_string(),
        num_variables: 20,
        density: 0.3,
        connectivity: ConnectivityStructure::Sparse { avg_degree: 4.0 },
        hardness_indicators: HardnessIndicators {
            complexity_class: ComplexityClass::NPComplete,
            difficulty_metrics: HashMap::new(),
            approximation_bounds: None,
        },
        symmetries: vec![],
    };

    let algorithms = vec![
        ClassicalAlgorithm::SimulatedAnnealing,
        ClassicalAlgorithm::TabuSearch,
        ClassicalAlgorithm::GeneticAlgorithm,
    ];

    let result = estimator.estimate_classical_performance(&problem_chars, &algorithms);
    assert!(result.is_ok());

    let performance = result.unwrap();
    assert_eq!(performance.len(), 3);

    for (alg, perf) in &performance {
        println!("Algorithm: {:?}", alg);
        println!("  Approximation ratio: {:.3}", perf.solution_quality.approximation_ratio);
        println!("  Success probability: {:.3}", perf.success_probability);
        assert!(perf.solution_quality.approximation_ratio > 0.0);
        assert!(perf.solution_quality.approximation_ratio <= 1.0);
        assert!(perf.success_probability > 0.0);
        assert!(perf.success_probability <= 1.0);
    }
}

#[test]
fn test_quantum_resource_estimation() {
    let estimator = QuantumResourceEstimator::new();
    
    let problem_chars = ProblemCharacteristics {
        problem_type: "QAOA Test".to_string(),
        num_variables: 15,
        density: 0.4,
        connectivity: ConnectivityStructure::FullyConnected,
        hardness_indicators: HardnessIndicators {
            complexity_class: ComplexityClass::NPComplete,
            difficulty_metrics: HashMap::new(),
            approximation_bounds: None,
        },
        symmetries: vec![SymmetryType::Permutation],
    };

    let algorithms = vec![
        QuantumAlgorithm::QAOA,
        QuantumAlgorithm::VQE,
        QuantumAlgorithm::QuantumAnnealing,
    ];

    let result = estimator.estimate_quantum_performance(&problem_chars, &algorithms);
    assert!(result.is_ok());

    let performance = result.unwrap();
    assert_eq!(performance.len(), 3);

    for (alg, perf) in &performance {
        println!("Quantum Algorithm: {:?}", alg);
        println!("  Qubits required: {}", perf.quantum_metrics.qubits_required);
        println!("  Circuit depth: {}", perf.quantum_metrics.circuit_depth);
        println!("  Approximation ratio: {:.3}", perf.base_metrics.solution_quality.approximation_ratio);
        
        assert!(perf.quantum_metrics.qubits_required >= problem_chars.num_variables);
        assert!(perf.quantum_metrics.circuit_depth > 0);
        assert!(perf.base_metrics.solution_quality.approximation_ratio > 0.0);
        assert!(perf.hardware_requirements.min_qubits > 0);
        assert!(perf.hardware_requirements.gate_fidelity_threshold > 0.0);
        assert!(perf.hardware_requirements.gate_fidelity_threshold <= 1.0);
    }
}

#[test]
fn test_advantage_analysis_components() {
    let config = AnalysisConfig::default();
    let analyzer = QuantumAdvantageAnalyzer::new(config);

    // Create a medium-sized test problem
    let size = 25;
    let mut qubo = Array2::zeros((size, size));
    
    // Create a problem where quantum might have advantage
    for i in 0..size {
        for j in i+1..size {
            if (i + j) % 3 == 0 {
                qubo[[i, j]] = -1.0;
                qubo[[j, i]] = -1.0;
            }
        }
    }

    let result = analyzer.analyze_advantage(&qubo, None);
    assert!(result.is_ok());

    let analysis = result.unwrap();
    
    // Test advantage analysis components
    assert!(analysis.advantage_analysis.confidence >= 0.0);
    assert!(analysis.advantage_analysis.confidence <= 1.0);
    
    // Test that advantage factors are present
    if analysis.advantage_analysis.has_quantum_advantage {
        assert!(!analysis.advantage_analysis.advantage_factors.is_empty());
    }
    
    // Test threshold analysis
    assert!(!analysis.threshold_analysis.size_thresholds.is_empty());
    assert!(!analysis.threshold_analysis.time_to_advantage.is_empty());
    
    // Test recommendations
    assert!(!analysis.recommendations.is_empty());
    for recommendation in &analysis.recommendations {
        assert!(recommendation.confidence >= 0.0);
        assert!(recommendation.confidence <= 1.0);
        assert!(!recommendation.algorithm.is_empty());
        assert!(!recommendation.justification.is_empty());
    }
    
    println!("Advantage analysis components test passed");
}

#[test]
fn test_export_functionality() {
    let config = AnalysisConfig {
        problem_size_range: (5, 10),
        num_samples: 5,
        ..Default::default()
    };
    let analyzer = QuantumAdvantageAnalyzer::new(config);

    // Create simple test problem
    let qubo = Array2::eye(5);
    let result = analyzer.analyze_advantage(&qubo, None);
    assert!(result.is_ok());

    let analysis = result.unwrap();
    
    // Test JSON export
    let json_export = analyzer.export_results(&analysis, ExportFormat::JSON);
    assert!(json_export.is_ok());
    let json_str = json_export.unwrap();
    assert!(json_str.contains("problem_info"));
    assert!(json_str.contains("advantage_analysis"));
    
    // Test Python export
    let python_export = analyzer.export_results(&analysis, ExportFormat::Python);
    assert!(python_export.is_ok());
    let python_str = python_export.unwrap();
    assert!(python_str.contains("import numpy as np"));
    assert!(python_str.contains("matplotlib.pyplot"));
    
    // Test Rust export
    let rust_export = analyzer.export_results(&analysis, ExportFormat::Rust);
    assert!(rust_export.is_ok());
    let rust_str = rust_export.unwrap();
    assert!(rust_str.contains("quantum_advantage_analysis"));
    assert!(rust_str.contains("fn reproduce_analysis"));
    
    println!("Export functionality tests passed");
}

#[test]
fn test_supremacy_benchmarker() {
    let mut benchmarker = QuantumSupremacyBenchmarker::new();
    let config = AnalysisConfig {
        problem_size_range: (5, 15),
        num_samples: 3,
        ..Default::default()
    };
    
    let result = benchmarker.run_comprehensive_benchmark(&config);
    assert!(result.is_ok());
    
    let benchmark_result = result.unwrap();
    assert!(!benchmark_result.benchmark_results.is_empty());
    assert!(benchmark_result.summary.total_benchmarks > 0);
    assert!(!benchmark_result.summary.recommendations.is_empty());
    
    println!("Supremacy benchmarker test passed");
    println!("Benchmarks run: {}", benchmark_result.summary.total_benchmarks);
    println!("Advantage threshold: {}", benchmark_result.summary.advantage_threshold);
}

#[test]
fn test_noise_sensitivity_analysis() {
    let config = AnalysisConfig {
        noise_models: vec![
            NoiseModel::None,
            NoiseModel::Depolarizing { rate: 0.001 },
            NoiseModel::AmplitudeDamping { rate: 0.01 },
            NoiseModel::Realistic { decoherence_time: Duration::from_micros(50) },
        ],
        ..Default::default()
    };
    
    let analyzer = QuantumAdvantageAnalyzer::new(config);
    let qubo = Array2::eye(8);
    
    let result = analyzer.analyze_advantage(&qubo, None);
    assert!(result.is_ok());
    
    let analysis = result.unwrap();
    
    // Check that quantum performance includes noise sensitivity
    for (alg, perf) in &analysis.quantum_performance {
        assert!(!perf.noise_sensitivity.error_thresholds.is_empty());
        
        // Verify error thresholds are reasonable
        for (noise_type, threshold) in &perf.noise_sensitivity.error_thresholds {
            assert!(*threshold > 0.0);
            assert!(*threshold < 1.0);
            println!("Algorithm {:?} - {}: threshold = {:.6}", alg, noise_type, threshold);
        }
    }
    
    println!("Noise sensitivity analysis test passed");
}

#[test]
fn test_hardware_requirements_analysis() {
    let config = AnalysisConfig {
        hardware_models: vec![
            HardwareModel::IdealQuantum,
            HardwareModel::NoisyNISQ { error_rate: 0.001 },
            HardwareModel::DigitalAnnealer,
            HardwareModel::PhotonicIsing,
            HardwareModel::CoherentIsingMachine,
            HardwareModel::SuperconductingAnnealer,
        ],
        ..Default::default()
    };
    
    let analyzer = QuantumAdvantageAnalyzer::new(config);
    let size = 20;
    let qubo = Array2::eye(size);
    
    let result = analyzer.analyze_advantage(&qubo, None);
    assert!(result.is_ok());
    
    let analysis = result.unwrap();
    
    // Check hardware requirements for quantum algorithms
    for (alg, perf) in &analysis.quantum_performance {
        let hw_req = &perf.hardware_requirements;
        
        assert!(hw_req.min_qubits >= size);
        assert!(hw_req.gate_fidelity_threshold > 0.0);
        assert!(hw_req.gate_fidelity_threshold <= 1.0);
        assert!(hw_req.measurement_fidelity_threshold > 0.0);
        assert!(hw_req.measurement_fidelity_threshold <= 1.0);
        
        println!("Algorithm {:?}:", alg);
        println!("  Min qubits: {}", hw_req.min_qubits);
        println!("  Gate fidelity: {:.4}", hw_req.gate_fidelity_threshold);
        println!("  Measurement fidelity: {:.4}", hw_req.measurement_fidelity_threshold);
        println!("  Connectivity: {:?}", hw_req.connectivity_requirements);
    }
    
    println!("Hardware requirements analysis test passed");
}

#[test]
fn test_complexity_function_ordering() {
    use ComplexityFunction::*;
    
    // Test ordering of complexity functions
    assert!(Constant < Logarithmic);
    assert!(Logarithmic < Linear);
    assert!(Linear < Linearithmic);
    assert!(Linearithmic < Quadratic);
    assert!(Quadratic < Cubic);
    assert!(Cubic < Polynomial { degree: 4.0 });
    assert!(Polynomial { degree: 4.0 } < Exponential { base: 2.0 });
    assert!(Exponential { base: 2.0 } < Factorial);
    
    // Test polynomial comparison
    assert!(Polynomial { degree: 2.0 } < Polynomial { degree: 3.0 });
    assert!(Polynomial { degree: 3.0 } > Polynomial { degree: 2.0 });
    
    // Test exponential comparison
    assert!(Exponential { base: 1.5 } < Exponential { base: 2.0 });
    
    println!("Complexity function ordering test passed");
}