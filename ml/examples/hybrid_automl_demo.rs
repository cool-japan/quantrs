#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
//! Hybrid AutoML Engine Demonstration
//!
//! This example demonstrates the Quantum-Classical Hybrid AutoML Decision Engine
//! that intelligently recommends the optimal algorithm (quantum, classical, or hybrid)
//! based on problem characteristics, available resources, and performance requirements.
//!
//! Run with: `cargo run --example hybrid_automl_demo`

use quantrs2_ml::hybrid_automl_engine::{
    AlgorithmChoice, ClassicalCompute, DeviceAvailability, HybridAutoMLEngine,
    ProblemCharacteristics, ProblemDomain, QuantumDevice, ResourceConstraints, TaskType,
};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Rng};

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Quantum-Classical Hybrid AutoML Decision Engine        ║");
    println!("║  Intelligent Algorithm Selection & Configuration        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Create the AutoML engine
    let engine = HybridAutoMLEngine::new();

    // ========================================================================
    // Scenario 1: Small Dataset - Drug Discovery
    // ========================================================================

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Scenario 1: Small High-Dimensional Dataset (Drug Discovery)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let small_dataset = generate_synthetic_dataset(200, 50, 2);
    let mut chars1 = ProblemCharacteristics::from_dataset(&small_dataset.0, &small_dataset.1);
    chars1.domain = ProblemDomain::DrugDiscovery;

    println!("Problem Characteristics:");
    println!("  Samples: {}", chars1.n_samples);
    println!("  Features: {}", chars1.n_features);
    println!("  Classes: {}", chars1.n_classes);
    println!("  Dimensionality ratio: {:.3}", chars1.dimensionality_ratio);
    println!("  Sparsity: {:.2}%", chars1.sparsity * 100.0);
    println!("  Class imbalance: {:.2}", chars1.class_imbalance);
    println!("  Domain: {:?}\n", chars1.domain);

    let constraints1 = create_quantum_available_constraints();

    match engine.analyze_and_recommend(&chars1, &constraints1) {
        Ok(recommendation) => {
            print_recommendation(&recommendation, "Scenario 1");
        }
        Err(e) => println!("Error: {e}"),
    }

    // ========================================================================
    // Scenario 2: Large Dataset - Finance
    // ========================================================================

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Scenario 2: Large Low-Dimensional Dataset (Finance)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let large_dataset = generate_synthetic_dataset(10000, 15, 2);
    let mut chars2 = ProblemCharacteristics::from_dataset(&large_dataset.0, &large_dataset.1);
    chars2.domain = ProblemDomain::Finance;

    println!("Problem Characteristics:");
    println!("  Samples: {}", chars2.n_samples);
    println!("  Features: {}", chars2.n_features);
    println!("  Classes: {}", chars2.n_classes);
    println!("  Dimensionality ratio: {:.3}", chars2.dimensionality_ratio);
    println!("  Sparsity: {:.2}%", chars2.sparsity * 100.0);
    println!("  Class imbalance: {:.2}", chars2.class_imbalance);
    println!("  Domain: {:?}\n", chars2.domain);

    let constraints2 = create_quantum_available_constraints();

    match engine.analyze_and_recommend(&chars2, &constraints2) {
        Ok(recommendation) => {
            print_recommendation(&recommendation, "Scenario 2");
        }
        Err(e) => println!("Error: {e}"),
    }

    // ========================================================================
    // Scenario 3: Multi-class Classification - Computer Vision
    // ========================================================================

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Scenario 3: Multi-class Classification (Computer Vision)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let multiclass_dataset = generate_synthetic_dataset(5000, 784, 10);
    let mut chars3 =
        ProblemCharacteristics::from_dataset(&multiclass_dataset.0, &multiclass_dataset.1);
    chars3.domain = ProblemDomain::ComputerVision;

    println!("Problem Characteristics:");
    println!("  Samples: {}", chars3.n_samples);
    println!("  Features: {} (28×28 images)", chars3.n_features);
    println!("  Classes: {}", chars3.n_classes);
    println!("  Dimensionality ratio: {:.3}", chars3.dimensionality_ratio);
    println!("  Sparsity: {:.2}%", chars3.sparsity * 100.0);
    println!("  Class imbalance: {:.2}", chars3.class_imbalance);
    println!("  Domain: {:?}\n", chars3.domain);

    let constraints3 = create_classical_only_constraints();

    match engine.analyze_and_recommend(&chars3, &constraints3) {
        Ok(recommendation) => {
            print_recommendation(&recommendation, "Scenario 3");
        }
        Err(e) => println!("Error: {e}"),
    }

    // ========================================================================
    // Scenario 4: Constrained Resources - Edge Computing
    // ========================================================================

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Scenario 4: Resource-Constrained Environment (Edge Device)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let edge_dataset = generate_synthetic_dataset(1000, 30, 2);
    let mut chars4 = ProblemCharacteristics::from_dataset(&edge_dataset.0, &edge_dataset.1);
    chars4.domain = ProblemDomain::General;

    println!("Problem Characteristics:");
    println!("  Samples: {}", chars4.n_samples);
    println!("  Features: {}", chars4.n_features);
    println!("  Classes: {}", chars4.n_classes);
    println!("  Dimensionality ratio: {:.3}", chars4.dimensionality_ratio);
    println!("  Domain: {:?}\n", chars4.domain);

    let constraints4 = create_edge_device_constraints();

    println!("Resource Constraints:");
    println!("  Max latency: 10ms");
    println!("  Max cost per inference: $0.0001");
    println!("  Max training time: 60s");
    println!("  Power limit: 10W\n");

    match engine.analyze_and_recommend(&chars4, &constraints4) {
        Ok(recommendation) => {
            print_recommendation(&recommendation, "Scenario 4");
        }
        Err(e) => println!("Error: {e}"),
    }

    // ========================================================================
    // Comparison Summary
    // ========================================================================

    println!("\n\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Decision Summary Across Scenarios                      ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("┌─────────────┬────────────────────┬──────────────────────────┐");
    println!("│ Scenario    │ Recommended        │ Reasoning                │");
    println!("├─────────────┼────────────────────┼──────────────────────────┤");
    println!("│ 1. Drug     │ Quantum (QSVM)     │ High-dim, small sample  │");
    println!("│ Discovery   │                    │ Quantum kernel advantage │");
    println!("├─────────────┼────────────────────┼──────────────────────────┤");
    println!("│ 2. Finance  │ Classical (XGBoost)│ Large sample, low-dim    │");
    println!("│             │                    │ Classical more efficient │");
    println!("├─────────────┼────────────────────┼──────────────────────────┤");
    println!("│ 3. Vision   │ Classical (CNN)    │ Many samples, no quantum │");
    println!("│             │                    │ devices available        │");
    println!("├─────────────┼────────────────────┼──────────────────────────┤");
    println!("│ 4. Edge     │ Classical (RF)     │ Latency & power          │");
    println!("│ Device      │                    │ constraints critical     │");
    println!("└─────────────┴────────────────────┴──────────────────────────┘");

    println!("\n\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Key Insights                                            ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("1. 🔬 Quantum Advantages:");
    println!("   • High-dimensional feature spaces (features >> samples)");
    println!("   • Complex kernel functions (non-linear separability)");
    println!("   • Small to medium datasets (< 10,000 samples)");
    println!("   • Domain-specific problems (chemistry, materials)");

    println!("\n2. ⚡ Classical Advantages:");
    println!("   • Large datasets (> 10,000 samples)");
    println!("   • Low-dimensional feature spaces");
    println!("   • Strict latency requirements (< 10ms)");
    println!("   • Cost-sensitive applications");

    println!("\n3. 🤝 Hybrid Approaches:");
    println!("   • Quantum feature engineering + classical training");
    println!("   • Ensemble methods combining both paradigms");
    println!("   • Quantum-enhanced hyperparameter optimization");

    println!("\n4. 💡 Production Considerations:");
    println!("   • Always include probability calibration");
    println!("   • Monitor performance drift over time");
    println!("   • Have classical fallback for quantum unavailability");
    println!("   • Optimize batch sizes for throughput");
    println!("   • Enable caching for repeated inference patterns");

    println!("\n✨ Hybrid AutoML demonstration complete! ✨\n");
}

/// Generate synthetic dataset
fn generate_synthetic_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array1<usize>) {
    let mut rng = thread_rng();

    let x = Array2::from_shape_fn((n_samples, n_features), |(_, _)| {
        rng.random::<f64>().mul_add(2.0, -1.0)
    });

    let y = Array1::from_shape_fn(n_samples, |_| rng.random_range(0..n_classes));

    (x, y)
}

/// Create resource constraints with quantum devices available
fn create_quantum_available_constraints() -> ResourceConstraints {
    ResourceConstraints {
        quantum_devices: vec![QuantumDevice {
            name: "IBM Quantum Eagle".to_string(),
            n_qubits: 127,
            gate_error_rate: 0.001,
            measurement_error_rate: 0.01,
            decoherence_time_us: 100.0,
            cost_per_shot: 0.0001,
            availability: DeviceAvailability::Available,
        }],
        classical_compute: ClassicalCompute {
            n_cpu_cores: 16,
            has_gpu: true,
            gpu_memory_gb: 32.0,
            ram_gb: 128.0,
        },
        max_latency_ms: None,
        max_cost_per_inference: None,
        max_training_time: None,
        max_power_consumption: None,
    }
}

/// Create resource constraints with only classical compute
const fn create_classical_only_constraints() -> ResourceConstraints {
    ResourceConstraints {
        quantum_devices: vec![],
        classical_compute: ClassicalCompute {
            n_cpu_cores: 32,
            has_gpu: true,
            gpu_memory_gb: 80.0,
            ram_gb: 256.0,
        },
        max_latency_ms: Some(100.0),
        max_cost_per_inference: Some(0.01),
        max_training_time: Some(3600.0),
        max_power_consumption: None,
    }
}

/// Create resource constraints for edge device
const fn create_edge_device_constraints() -> ResourceConstraints {
    ResourceConstraints {
        quantum_devices: vec![],
        classical_compute: ClassicalCompute {
            n_cpu_cores: 4,
            has_gpu: false,
            gpu_memory_gb: 0.0,
            ram_gb: 8.0,
        },
        max_latency_ms: Some(10.0),
        max_cost_per_inference: Some(0.0001),
        max_training_time: Some(60.0),
        max_power_consumption: Some(10.0),
    }
}

/// Print recommendation details
fn print_recommendation(
    recommendation: &quantrs2_ml::hybrid_automl_engine::AlgorithmRecommendation,
    scenario: &str,
) {
    println!("🎯 Recommendation for {scenario}:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Algorithm choice
    print!("Algorithm: ");
    match &recommendation.algorithm_choice {
        AlgorithmChoice::QuantumOnly { algorithm, device } => {
            println!("{algorithm} (Quantum)");
            println!("  Device: {device}");
        }
        AlgorithmChoice::ClassicalOnly { algorithm, backend } => {
            println!("{algorithm} (Classical)");
            println!("  Backend: {backend}");
        }
        AlgorithmChoice::Hybrid {
            quantum_component,
            classical_component,
            splitting_strategy,
        } => {
            println!("Hybrid Approach");
            println!("  Quantum: {quantum_component}");
            println!("  Classical: {classical_component}");
            println!("  Strategy: {splitting_strategy}");
        }
    }

    // Quantum advantage metrics
    if recommendation.quantum_advantage.speedup > 1.0 {
        println!("\nQuantum Advantage Metrics:");
        println!(
            "  Speedup: {:.2}x",
            recommendation.quantum_advantage.speedup
        );
        println!(
            "  Accuracy improvement: {:.1}%",
            recommendation.quantum_advantage.accuracy_improvement * 100.0
        );
        println!(
            "  Sample efficiency: {:.2}x",
            recommendation.quantum_advantage.sample_efficiency
        );
        println!(
            "  Statistical significance: p={:.4}",
            recommendation.quantum_advantage.statistical_significance
        );
    }

    // Performance estimates
    println!("\nExpected Performance:");
    println!(
        "  Accuracy: {:.1}% (95% CI: [{:.1}%, {:.1}%])",
        recommendation.expected_performance.accuracy * 100.0,
        recommendation.expected_performance.accuracy_ci.0 * 100.0,
        recommendation.expected_performance.accuracy_ci.1 * 100.0
    );
    println!(
        "  Training time: {:.1}s",
        recommendation.expected_performance.training_time_s
    );
    println!(
        "  Inference latency: {:.2}ms",
        recommendation.expected_performance.inference_latency_ms
    );
    println!(
        "  Memory footprint: {:.1}MB",
        recommendation.expected_performance.memory_mb
    );

    // Cost analysis
    println!("\nCost Analysis:");
    println!(
        "  Training cost: ${:.2}",
        recommendation.cost_analysis.training_cost
    );
    println!(
        "  Inference cost: ${:.6}/sample",
        recommendation.cost_analysis.inference_cost_per_sample
    );
    println!(
        "  Estimated total: ${:.2}",
        recommendation.cost_analysis.total_cost
    );

    // Recommended hyperparameters
    if !recommendation.hyperparameters.is_empty() {
        println!("\nRecommended Hyperparameters:");
        for (name, value) in &recommendation.hyperparameters {
            println!("  {name}: {value}");
        }
    }

    // Calibration
    if let Some(ref calib_method) = recommendation.calibration_method {
        println!("\nCalibration: {calib_method}");
    }

    // Production configuration
    println!("\nProduction Configuration:");
    println!(
        "  Batch size: {}",
        recommendation.production_config.batch_size
    );
    println!("  Workers: {}", recommendation.production_config.n_workers);
    println!(
        "  Caching: {}",
        if recommendation.production_config.enable_caching {
            "Enabled"
        } else {
            "Disabled"
        }
    );

    println!("\n  Monitoring:");
    println!(
        "    Log interval: {} inferences",
        recommendation.production_config.monitoring.log_interval
    );
    println!(
        "    Tracked metrics: {}",
        recommendation
            .production_config
            .monitoring
            .tracked_metrics
            .len()
    );

    if recommendation.production_config.scaling.auto_scaling {
        println!("\n  Auto-scaling:");
        println!(
            "    Range: {}-{} instances",
            recommendation.production_config.scaling.min_instances,
            recommendation.production_config.scaling.max_instances
        );
        println!(
            "    Scale up at: {:.0}% CPU",
            recommendation.production_config.scaling.scale_up_threshold
        );
        println!(
            "    Scale down at: {:.0}% CPU",
            recommendation
                .production_config
                .scaling
                .scale_down_threshold
        );
    }

    println!(
        "\n  Confidence: {:.1}%\n",
        recommendation.confidence * 100.0
    );
}
