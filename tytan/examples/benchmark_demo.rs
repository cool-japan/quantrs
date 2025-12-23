//! Demonstration of hardware benchmarking suite with `SciRS2` analysis
//!
//! This example shows how to use the comprehensive benchmarking suite
//! to evaluate quantum annealing hardware and simulation backends.

use quantrs2_tytan::benchmark::prelude::*;
use quantrs2_tytan::benchmark::runner::SamplerConfig;
use std::collections::HashMap;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== QuantRS2 Hardware Benchmarking Suite ===\n");

    // Example 1: Quick benchmark
    quick_benchmark_demo()?;
    println!();

    // Example 2: Custom benchmark configuration
    custom_benchmark_demo()?;
    println!();

    // Example 3: Comparative analysis
    comparative_benchmark_demo()?;

    Ok(())
}

/// Quick benchmark with default settings
fn quick_benchmark_demo() -> Result<(), Box<dyn Error>> {
    println!("1. Quick Benchmark Demo");
    println!("   Running quick benchmark for 100-variable problem...");

    let metrics = quantrs2_tytan::benchmark::runner::quick_benchmark(100)?;

    println!("   Results:");
    println!("   - Total time: {:?}", metrics.timings.total_time);
    println!(
        "   - Time per sample: {:?}",
        metrics.timings.time_per_sample
    );
    println!("   - Best energy: {:.4}", metrics.quality.best_energy);
    println!(
        "   - Peak memory: {} MB",
        metrics.memory.peak_memory / 1_048_576
    );

    let efficiency = metrics.calculate_efficiency();
    println!(
        "   - Samples per second: {:.2}",
        efficiency.samples_per_second
    );
    println!(
        "   - Memory efficiency: {:.4}",
        efficiency.memory_efficiency
    );

    Ok(())
}

/// Custom benchmark configuration
fn custom_benchmark_demo() -> Result<(), Box<dyn Error>> {
    println!("2. Custom Benchmark Configuration");
    println!("   Setting up comprehensive benchmark suite...");

    // Configure benchmark
    let config = BenchmarkConfig {
        problem_sizes: vec![10, 50, 100, 250],
        problem_densities: vec![0.1, 0.5, 1.0],
        num_reads: 50,
        num_repetitions: 3,
        backends: vec!["cpu".to_string()],
        sampler_configs: vec![
            SamplerConfig {
                name: "SA".to_string(),
                params: HashMap::from([
                    ("T_0".to_string(), 10.0),
                    ("T_f".to_string(), 0.01),
                    ("steps".to_string(), 1000.0),
                    ("threads".to_string(), 4.0),
                ]),
            },
            SamplerConfig {
                name: "GA".to_string(),
                params: HashMap::from([
                    ("population_size".to_string(), 100.0),
                    ("max_generations".to_string(), 50.0),
                    ("mutation_rate".to_string(), 0.1),
                ]),
            },
        ],
        save_intermediate: true,
        output_dir: Some("benchmark_results".to_string()),
        timeout_seconds: 600,
    };

    // Run benchmarks
    let mut runner = BenchmarkRunner::new(config);
    let report = runner.run_complete_suite()?;

    // Display summary
    println!("\n   Benchmark Summary:");
    println!(
        "   - Total benchmarks: {}",
        report.metadata.total_benchmarks
    );
    println!("   - Duration: {:?}", report.metadata.total_duration);
    println!(
        "   - Best configuration: {}-{}",
        report.summary.most_efficient_backend, report.summary.most_efficient_sampler
    );

    // Show recommendations
    println!("\n   Recommendations:");
    for rec in &report.recommendations {
        println!(
            "   - [{}] {}",
            match rec.impact {
                quantrs2_tytan::benchmark::analysis::ImpactLevel::High => "HIGH",
                quantrs2_tytan::benchmark::analysis::ImpactLevel::Medium => "MEDIUM",
                quantrs2_tytan::benchmark::analysis::ImpactLevel::Low => "LOW",
            },
            rec.message
        );
    }

    Ok(())
}

/// Comparative benchmark analysis
fn comparative_benchmark_demo() -> Result<(), Box<dyn Error>> {
    use quantrs2_tytan::benchmark::visualization::BenchmarkVisualizer;

    println!("3. Comparative Benchmark Analysis");
    println!("   Running comparative analysis across problem types...");

    // Configure for comparative analysis
    let config = BenchmarkConfig {
        problem_sizes: vec![50, 100, 200],
        problem_densities: vec![0.1, 0.3, 0.5, 0.7, 1.0],
        num_reads: 25,
        num_repetitions: 2,
        backends: vec!["cpu".to_string()],
        sampler_configs: vec![
            SamplerConfig {
                name: "SA-fast".to_string(),
                params: HashMap::from([
                    ("T_0".to_string(), 5.0),
                    ("T_f".to_string(), 0.1),
                    ("steps".to_string(), 500.0),
                ]),
            },
            SamplerConfig {
                name: "SA-quality".to_string(),
                params: HashMap::from([
                    ("T_0".to_string(), 20.0),
                    ("T_f".to_string(), 0.001),
                    ("steps".to_string(), 5000.0),
                ]),
            },
        ],
        save_intermediate: false,
        output_dir: Some("comparative_results".to_string()),
        timeout_seconds: 300,
    };

    // Run analysis
    let mut runner = BenchmarkRunner::new(config);
    let report = runner.run_complete_suite()?;

    // Analyze scaling
    println!("\n   Scaling Analysis:");
    println!(
        "   - Time complexity: {} (R² = {:.3})",
        report.scaling_analysis.time_complexity.order,
        report.scaling_analysis.time_complexity.r_squared
    );
    println!(
        "   - Memory complexity: {} (R² = {:.3})",
        report.scaling_analysis.memory_complexity.order,
        report.scaling_analysis.memory_complexity.r_squared
    );
    println!(
        "   - Optimal sizes: {:?}",
        report.scaling_analysis.optimal_problem_sizes
    );

    // Show Pareto frontier
    println!("\n   Pareto-Optimal Configurations:");
    for point in &report.comparison.pareto_frontier {
        println!(
            "   - {}: quality={:.2}, performance={:.2}",
            point.configuration,
            -point.quality_score, // Negative because lower energy is better
            point.performance_score
        );
    }

    // Generate visualizations
    println!("\n   Generating visualizations...");
    let mut visualizer = BenchmarkVisualizer::new(report);
    visualizer.generate_all("comparative_results")?;
    println!("   Visualizations saved to comparative_results/");

    Ok(())
}

/// Example showing SciRS2 optimizations
#[cfg(feature = "scirs")]
fn scirs_optimization_demo() -> Result<(), Box<dyn Error>> {
    println!("4. SciRS2 Optimization Analysis");
    println!("   Comparing standard vs SciRS2-optimized implementations...");

    // This would compare backends with and without SciRS2 optimizations
    // The actual implementation would show performance differences

    println!("   SciRS2 optimizations enabled:");
    println!("   - SIMD operations for energy calculations");
    println!("   - Optimized sparse matrix operations");
    println!("   - Advanced memory management");
    println!("   - Parallel tensor contractions");

    Ok(())
}
