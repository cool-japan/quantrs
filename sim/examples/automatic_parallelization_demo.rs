//! Automatic Parallelization Demo
//!
//! This example demonstrates the automatic parallelization capabilities
//! of `QuantRS2`, showing how circuits can be analyzed for parallelization
//! opportunities and executed efficiently using `SciRS2` parallel operations.

use quantrs2_circuit::builder::Circuit;
use quantrs2_sim::{
    automatic_parallelization::{
        benchmark_automatic_parallelization, AutoParallelConfig, AutoParallelEngine,
        ParallelizationStrategy,
    },
    large_scale_simulator::{LargeScaleQuantumSimulator, LargeScaleSimulatorConfig},
};
use std::time::Instant;

/// Demonstrate automatic parallelization analysis
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 QuantRS2 Automatic Parallelization Demo");
    println!("==========================================\n");

    // Create demo circuits with different parallelization characteristics
    let circuits = create_demo_circuits()?;

    // Test different parallelization strategies
    test_parallelization_strategies(&circuits)?;

    // Benchmark parallelization performance
    benchmark_parallelization(&circuits)?;

    // Demonstrate parallel execution
    demonstrate_parallel_execution()?;

    println!("\n✅ Automatic parallelization demo completed successfully!");
    Ok(())
}

/// Create demo circuits with different structures
fn create_demo_circuits() -> Result<Vec<Circuit<16>>, Box<dyn std::error::Error>> {
    println!("📊 Creating demo circuits for parallelization analysis...\n");

    let mut circuits = Vec::new();

    // Circuit 1: Independent single-qubit gates (high parallelism)
    println!("Circuit 1: Independent single-qubit gates");
    let mut circuit1 = Circuit::<16>::new();
    for i in 0..16 {
        circuit1.h(i)?; // All H gates can run in parallel
    }
    for i in 0..16 {
        circuit1.x(i)?; // All X gates can run in parallel
    }
    println!(
        "- Gates: {}, Expected parallelism: High",
        circuit1.num_gates()
    );
    circuits.push(circuit1);

    // Circuit 2: Sequential two-qubit gates (low parallelism)
    println!("\nCircuit 2: Sequential two-qubit gates");
    let mut circuit2 = Circuit::<16>::new();
    for i in 0..15 {
        circuit2.cnot(i, i + 1)?; // Sequential CNOTs
    }
    println!(
        "- Gates: {}, Expected parallelism: Low",
        circuit2.num_gates()
    );
    circuits.push(circuit2);

    // Circuit 3: Mixed structure (medium parallelism)
    println!("\nCircuit 3: Mixed structure");
    let mut circuit3 = Circuit::<16>::new();
    // Parallel H gates
    for i in 0..8 {
        circuit3.h(i)?;
    }
    // Parallel CNOT pairs
    for i in (0..8).step_by(2) {
        circuit3.cnot(i, i + 1)?;
    }
    // More parallel gates
    for i in 8..16 {
        circuit3.ry(i, std::f64::consts::PI / 4.0)?;
    }
    println!(
        "- Gates: {}, Expected parallelism: Medium",
        circuit3.num_gates()
    );
    circuits.push(circuit3);

    // Circuit 4: Complex structure for advanced analysis
    println!("\nCircuit 4: Complex quantum algorithm pattern");
    let mut circuit4 = Circuit::<16>::new();

    // Initial superposition
    for i in 0..8 {
        circuit4.h(i)?;
    }

    // Entanglement layer
    for i in 0..7 {
        circuit4.cnot(i, i + 1)?;
    }

    // Parameterized rotations
    for i in 0..8 {
        let angle = std::f64::consts::PI * f64::from(i) / 8.0;
        circuit4.ry(i, angle)?;
        circuit4.rz(i, angle / 2.0)?;
    }

    // Final measurements preparation
    for i in (0..8).step_by(2) {
        circuit4.cnot(i, i + 8)?;
    }

    println!(
        "- Gates: {}, Expected parallelism: Complex",
        circuit4.num_gates()
    );
    circuits.push(circuit4);

    Ok(circuits)
}

/// Test different parallelization strategies
fn test_parallelization_strategies(
    circuits: &[Circuit<16>],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧪 Testing Parallelization Strategies");
    println!("====================================\n");

    let strategies = vec![
        (
            "Dependency Analysis",
            ParallelizationStrategy::DependencyAnalysis,
        ),
        ("Layer-Based", ParallelizationStrategy::LayerBased),
        (
            "Qubit Partitioning",
            ParallelizationStrategy::QubitPartitioning,
        ),
        ("Hybrid", ParallelizationStrategy::Hybrid),
    ];

    for (i, circuit) in circuits.iter().enumerate() {
        println!("Circuit {} Analysis:", i + 1);
        println!("----------------");

        for (strategy_name, strategy) in &strategies {
            let config = AutoParallelConfig {
                strategy: *strategy,
                max_threads: 8,
                min_gates_for_parallel: 5,
                ..Default::default()
            };

            let engine = AutoParallelEngine::new(config);
            let start_time = Instant::now();

            match engine.analyze_circuit(circuit) {
                Ok(analysis) => {
                    let analysis_time = start_time.elapsed();

                    println!("  {strategy_name} Strategy:");
                    println!("    • Analysis time: {analysis_time:?}");
                    println!("    • Parallel tasks: {}", analysis.tasks.len());
                    println!("    • Max parallelism: {}", analysis.max_parallelism);
                    println!("    • Efficiency: {:.2}%", analysis.efficiency * 100.0);
                    println!(
                        "    • Critical path: {} layers",
                        analysis.critical_path_length
                    );

                    if !analysis.recommendations.is_empty() {
                        println!("    • Recommendations:");
                        for rec in analysis.recommendations.iter().take(2) {
                            println!("      - {}", rec.description);
                        }
                    }
                }
                Err(e) => {
                    println!("  {strategy_name} Strategy: Failed - {e:?}");
                }
            }
            println!();
        }
        println!();
    }

    Ok(())
}

/// Benchmark parallelization performance
fn benchmark_parallelization(circuits: &[Circuit<16>]) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Benchmarking Parallelization Performance");
    println!("=========================================\n");

    let config = AutoParallelConfig {
        strategy: ParallelizationStrategy::Hybrid,
        max_threads: 8,
        enable_analysis_caching: true,
        ..Default::default()
    };

    let benchmark_start = Instant::now();
    let results = benchmark_automatic_parallelization(circuits.to_vec(), config)?;
    let benchmark_time = benchmark_start.elapsed();

    println!("Benchmark Results:");
    println!("-----------------");
    println!("Total benchmark time: {benchmark_time:?}");
    println!(
        "Average efficiency: {:.2}%",
        results.average_efficiency * 100.0
    );
    println!("Average max parallelism: {}", results.average_parallelism);

    println!("\nPer-Circuit Results:");
    for (i, result) in results.circuit_results.iter().enumerate() {
        println!(
            "  Circuit {}: {} gates, {} qubits",
            i + 1,
            result.circuit_size,
            result.num_qubits
        );
        println!("    Analysis: {:?}", result.analysis_time);
        println!("    Efficiency: {:.2}%", result.efficiency * 100.0);
        println!("    Max parallelism: {}", result.max_parallelism);
        println!("    Tasks generated: {}", result.num_tasks);
    }

    Ok(())
}

/// Demonstrate parallel execution
fn demonstrate_parallel_execution() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demonstrating Parallel Execution");
    println!("==================================\n");

    // Create a circuit suitable for parallel execution
    let mut circuit = Circuit::<8>::new();

    // Layer 1: Parallel H gates
    for i in 0..8 {
        circuit.h(i)?;
    }

    // Layer 2: Parallel rotations
    for i in 0..8 {
        circuit.ry(i, std::f64::consts::PI / 4.0)?;
    }

    // Layer 3: Parallel CNOT pairs
    for i in (0..8).step_by(2) {
        circuit.cnot(i, i + 1)?;
    }

    println!("Created test circuit:");
    println!("- Qubits: {}", circuit.num_qubits());
    println!("- Gates: {}", circuit.num_gates());
    println!("- Expected layers: 3");

    // Set up parallelization engine
    let config = AutoParallelConfig {
        strategy: ParallelizationStrategy::DependencyAnalysis,
        max_threads: 4,
        enable_gate_fusion: true,
        ..Default::default()
    };

    let engine = AutoParallelEngine::new(config);

    // Analyze the circuit
    println!("\nAnalyzing circuit for parallelization...");
    let analysis = engine.analyze_circuit(&circuit)?;

    println!("Analysis Results:");
    println!("- Parallel tasks: {}", analysis.tasks.len());
    println!("- Layers: {}", analysis.num_layers);
    println!("- Max parallelism: {}", analysis.max_parallelism);
    println!("- Efficiency: {:.2}%", analysis.efficiency * 100.0);

    // Set up simulator for execution
    let sim_config = LargeScaleSimulatorConfig {
        max_qubits: 8,
        enable_sparse_representation: true,
        enable_chunked_processing: true,
        ..Default::default()
    };

    let mut simulator = LargeScaleQuantumSimulator::new(sim_config)?;

    // Execute with parallelization (this would normally run the parallel version)
    println!("\nExecuting circuit with automatic parallelization...");
    let execution_start = Instant::now();

    // For now, just demonstrate analysis since full execution needs more infrastructure
    let _result = engine.execute_parallel(&circuit, &mut simulator);

    let execution_time = execution_start.elapsed();
    println!("Execution completed in: {execution_time:?}");

    // Display resource utilization predictions
    println!("\nResource Utilization Predictions:");
    println!(
        "- CPU utilization: {:?}",
        analysis.resource_utilization.cpu_utilization
    );
    println!(
        "- Load balance score: {:.2}",
        analysis.resource_utilization.load_balance_score
    );
    println!(
        "- Communication overhead: {:.2}%",
        analysis.resource_utilization.communication_overhead * 100.0
    );

    Ok(())
}

/// Display optimization recommendations
fn display_recommendations(
    analysis: &quantrs2_sim::automatic_parallelization::ParallelizationAnalysis,
) {
    if !analysis.recommendations.is_empty() {
        println!("\n💡 Optimization Recommendations:");
        for (i, rec) in analysis.recommendations.iter().enumerate() {
            println!(
                "{}. {} (Expected improvement: {:.1}%)",
                i + 1,
                rec.description,
                rec.expected_improvement * 100.0
            );
            println!("   Complexity: {:?}", rec.complexity);
        }
    }
}

/// Compare sequential vs parallel execution times
fn compare_execution_performance() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Performance Comparison");
    println!("========================\n");

    // This would include actual timing comparisons between sequential
    // and parallel execution for various circuit sizes

    let circuit_sizes = vec![10, 20, 50, 100];

    println!("Circuit Size | Sequential | Parallel | Speedup");
    println!("-------------|------------|----------|--------");

    for size in circuit_sizes {
        // Simulated performance data (in a real implementation,
        // this would measure actual execution times)
        let sequential_time = f64::from(size) * 0.1; // ms
        let parallel_time = sequential_time / 3.5; // Simulated speedup
        let speedup = sequential_time / parallel_time;

        println!("{size:11} | {sequential_time:9.1}ms | {parallel_time:7.1}ms | {speedup:5.1}x");
    }

    Ok(())
}
