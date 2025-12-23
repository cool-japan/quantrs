//! Performance Benchmark Demo
//!
//! This example demonstrates the comprehensive performance benchmarking
//! capabilities of the `QuantRS2` simulation framework.

use quantrs2_core::error::QuantRS2Result;
use quantrs2_sim::performance_benchmark::{
    run_quick_benchmark, BenchmarkConfig, QuantumBenchmarkSuite,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 QuantRS2 Performance Benchmark Demo");
    println!("======================================\n");

    // Run a quick benchmark first
    println!("📊 Running Quick Benchmark...");
    println!("-----------------------------");
    run_quick_benchmark()?;

    println!("\n🔬 Running Custom Benchmark Suite...");
    println!("-----------------------------------");

    // Create a custom benchmark configuration for demonstration
    let config = BenchmarkConfig {
        qubit_range: 2..8, // Small range for demo
        iterations: 3,     // Few iterations for speed
        profile_memory: true,
        compare_optimizations: true,
        scalability_analysis: true,
        warmup_iterations: 1,
        max_circuit_depth: 10,
    };

    let mut suite = QuantumBenchmarkSuite::new(config);

    // Run selected benchmarks for demonstration
    suite.benchmark_basic_gates()?;
    suite.benchmark_circuit_execution()?;
    suite.benchmark_memory_efficiency()?;

    if suite.get_config().compare_optimizations {
        suite.benchmark_optimization_comparison()?;
    }

    // Generate and display results
    println!("\n📈 Benchmark Results Summary");
    println!("---------------------------");

    let results = suite.get_results();
    for result in results.iter().take(5) {
        // Show first 5 results
        println!(
            "  ✓ {}: {:.2}ms avg",
            result.name,
            result.timing.average_ns as f64 / 1_000_000.0
        );
    }

    // Export results to JSON (for demonstration)
    match suite.export_json() {
        Ok(json_data) => {
            println!("\n💾 Results exported to JSON ({} bytes)", json_data.len());
            // In a real application, you would save this to a file
        }
        Err(e) => println!("⚠️ Failed to export JSON: {e}"),
    }

    // Generate final report
    suite.generate_final_report();

    println!("✅ Benchmark Demo Complete!");

    Ok(())
}
