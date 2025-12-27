//! QEC Performance Benchmarking Demo
//!
//! This example demonstrates how to use the QEC benchmarking system to measure
//! and analyze the performance of different quantum error correction codes.

use quantrs2_device::qec::benchmarking::{QECBenchmarkConfig, QECBenchmarkSuite};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QuantRS2 QEC Performance Benchmarking Demo ===\n");

    // Create a comprehensive benchmark configuration
    let config = QECBenchmarkConfig {
        iterations: 50, // Number of iterations per benchmark
        shots_per_measurement: 500,
        error_rates: vec![0.001, 0.005, 0.01, 0.02],
        circuit_depths: vec![10, 20, 50],
        enable_detailed_stats: true,
        enable_profiling: true,
        max_duration: Duration::from_secs(120),
        confidence_level: 0.95,
    };

    println!("Benchmark Configuration:");
    println!("  - Iterations: {}", config.iterations);
    println!(
        "  - Shots per Measurement: {}",
        config.shots_per_measurement
    );
    println!("  - Error Rates: {:?}", config.error_rates);
    println!("  - Circuit Depths: {:?}", config.circuit_depths);
    println!(
        "  - Confidence Level: {:.2}%\n",
        config.confidence_level * 100.0
    );

    // Create the benchmark suite
    let suite = QECBenchmarkSuite::new(config);

    println!("Running comprehensive QEC benchmarks...");
    println!("This will benchmark:");
    println!("  1. Surface Code [[13,1,3]]");
    println!("  2. Steane Code [[7,1,3]]");
    println!("  3. Shor Code [[9,1,3]]");
    println!("  4. Toric Code 2x2 lattice");
    println!("  5. Syndrome detection methods");
    println!("  6. Error correction strategies");
    println!("  7. Adaptive QEC systems\n");

    // Run the comprehensive benchmark
    let start_time = std::time::Instant::now();
    let results = suite.run_comprehensive_benchmark()?;
    let elapsed = start_time.elapsed();

    println!(
        "\n✓ Benchmarks completed in {:.2}s\n",
        elapsed.as_secs_f64()
    );

    // Display summary results
    println!("=== Benchmark Results Summary ===\n");

    println!("QEC Code Performances:");
    println!("{:-<80}", "");
    for perf in &results.code_performances {
        println!("\n{}", perf.code_name);
        println!("  Data Qubits: {}", perf.num_data_qubits);
        println!("  Ancilla Qubits: {}", perf.num_ancilla_qubits);
        println!("  Code Distance: {}", perf.code_distance);
        println!(
            "  Encoding Time: {:.2} µs (± {:.2} µs)",
            perf.encoding_time.mean / 1000.0,
            perf.encoding_time.std_dev / 1000.0
        );
        println!(
            "  Syndrome Extraction: {:.2} µs (± {:.2} µs)",
            perf.syndrome_extraction_time.mean / 1000.0,
            perf.syndrome_extraction_time.std_dev / 1000.0
        );
        println!(
            "  Decoding Time: {:.2} µs (± {:.2} µs)",
            perf.decoding_time.mean / 1000.0,
            perf.decoding_time.std_dev / 1000.0
        );
        println!(
            "  Correction Time: {:.2} µs (± {:.2} µs)",
            perf.correction_time.mean / 1000.0,
            perf.correction_time.std_dev / 1000.0
        );
        println!("  Throughput: {:.2} ops/sec", perf.throughput);
        println!("  Memory Overhead: {:.2}x", perf.memory_overhead);

        if let Some(threshold) = perf.threshold_estimate {
            println!("  Estimated Threshold: {threshold:.4}");
        }

        println!("\n  Logical Error Rates:");
        for (error_rate, logical_rate) in &perf.logical_error_rates {
            println!("    {error_rate}: {logical_rate:.6}");
        }
    }

    println!("\n{:-<80}", "");
    println!("\nSyndrome Detection Performances:");
    println!("{:-<80}", "");
    for perf in &results.syndrome_detection_performances {
        println!("\n{}", perf.method_name);
        println!(
            "  Detection Time: {:.2} µs (± {:.2} µs)",
            perf.detection_time.mean / 1000.0,
            perf.detection_time.std_dev / 1000.0
        );
        println!("  Accuracy: {:.2}%", perf.accuracy * 100.0);
        println!("  Precision: {:.2}%", perf.precision * 100.0);
        println!("  Recall: {:.2}%", perf.recall * 100.0);
        println!("  F1 Score: {:.4}", perf.f1_score);
        if let Some(roc_auc) = perf.roc_auc {
            println!("  ROC AUC: {roc_auc:.4}");
        }
    }

    println!("\n{:-<80}", "");
    println!("\nError Correction Performances:");
    println!("{:-<80}", "");
    for perf in &results.error_correction_performances {
        println!("\n{}", perf.strategy_name);
        println!(
            "  Correction Time: {:.2} µs (± {:.2} µs)",
            perf.correction_time.mean / 1000.0,
            perf.correction_time.std_dev / 1000.0
        );
        println!("  Success Rate: {:.2}%", perf.success_rate * 100.0);
        println!(
            "  Avg Operations per Error: {:.2}",
            perf.avg_operations_per_error
        );
        println!("  Resource Overhead: {:.2}x", perf.resource_overhead);
        println!(
            "  Fidelity Improvement: {:.2}%",
            perf.fidelity_improvement * 100.0
        );
    }

    println!("\n{:-<80}", "");
    println!("\nAdaptive QEC Performances:");
    println!("{:-<80}", "");
    for perf in &results.adaptive_qec_performances {
        println!("\n{}", perf.system_id);
        println!(
            "  Convergence Time: {:.2}s",
            perf.convergence_time.as_secs_f64()
        );
        println!(
            "  Adaptation Overhead: {:.2}%",
            perf.adaptation_overhead * 100.0
        );
        println!(
            "  Improvement over Static: {:.2}%",
            perf.improvement_over_static * 100.0
        );
        if let Some(training_time) = &perf.ml_training_time {
            println!("  ML Training Time: {:.2}s", training_time.as_secs_f64());
        }
        if let Some(inference_time) = &perf.ml_inference_time {
            println!(
                "  ML Inference Time: {:.2} µs (± {:.2} µs)",
                inference_time.mean / 1000.0,
                inference_time.std_dev / 1000.0
            );
        }
    }

    println!("\n{:-<80}", "");
    println!("\nComparative Analysis:");
    println!("{:-<80}", "");
    println!("\nBest Performers by Metric:");
    for (metric, code) in &results.comparative_analysis.best_by_metric {
        println!("  {metric}: {code}");
    }

    println!("\nPerformance Rankings:");
    for (metric, ranking) in &results.comparative_analysis.rankings {
        println!("  {metric} (fastest to slowest):");
        for (i, code) in ranking.iter().enumerate() {
            println!("    {}. {}", i + 1, code);
        }
    }

    println!("\nStatistical Significance Tests:");
    for test in &results.comparative_analysis.significance_tests {
        println!(
            "  {} - {}: p={:.4} ({})",
            test.metric,
            test.comparison,
            test.p_value,
            if test.is_significant {
                "significant"
            } else {
                "not significant"
            }
        );
        println!("    Effect size: {:.3}", test.effect_size);
    }

    println!("\nRecommendations:");
    for (i, rec) in results
        .comparative_analysis
        .recommendations
        .iter()
        .enumerate()
    {
        println!("  {}. {}", i + 1, rec);
    }

    println!("\n{:-<80}", "");
    println!("\n=== Detailed Report ===\n");

    // Generate and display detailed report
    let report = suite.generate_report(&results);
    println!("{report}");

    println!("\n{:-<80}", "");
    println!("\nBenchmark completed successfully!");
    println!(
        "Total benchmark duration: {:.2}s",
        results.total_duration.as_secs_f64()
    );
    println!("Timestamp: {:?}", results.timestamp);

    Ok(())
}
