//! Comprehensive example demonstrating the D-Wave Leap cloud service client
//!
//! This example shows how to use the enhanced D-Wave Leap client to:
//! 1. Connect to D-Wave Leap cloud services
//! 2. Discover and select optimal solvers
//! 3. Submit problems with automatic embedding
//! 4. Use hybrid classical-quantum solvers
//! 5. Monitor problem status and performance
//! 6. Handle batch problem submission
//! 7. Work with custom annealing schedules

use std::collections::HashMap;
use std::time::Duration;

#[cfg(feature = "dwave")]
use quantrs2_anneal::{
    dwave::{
        AdvancedProblemParams, AnnealingSchedule, ChainStrengthMethod, DWaveClient,
        EmbeddingConfig, HybridSolverParams, OptimizationLevel, SolverCategory, SolverSelector,
    },
    ising::IsingModel,
};

#[cfg(not(feature = "dwave"))]
fn main() {
    println!("D-Wave Leap Client Example");
    println!("=========================");
    println!();
    println!("Note: This example requires the 'dwave' feature to be enabled.");
    println!("To run with D-Wave support:");
    println!("  cargo run --example dwave_leap_client_example --features dwave");
    println!();
    println!("You will also need:");
    println!("  1. A D-Wave Leap account and API token");
    println!("  2. Set the DWAVE_API_TOKEN environment variable");
}

#[cfg(feature = "dwave")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("D-Wave Leap Cloud Service Client Example");
    println!("=======================================");
    println!();

    // Get API token from environment variable
    let api_token = std::env::var("DWAVE_API_TOKEN")
        .map_err(|_| "Please set DWAVE_API_TOKEN environment variable")?;

    // Example 1: Basic client setup with custom configuration
    println!("1. Setting up D-Wave Leap Client");
    println!("-------------------------------");

    let solver_selector = SolverSelector {
        category: SolverCategory::All,
        min_qubits: Some(100),
        max_queue_time: Some(60.0), // Max 60 seconds queue time
        online_only: true,
        name_pattern: None,
        topology_preference: Some("pegasus".to_string()),
    };

    let embedding_config = EmbeddingConfig {
        auto_embed: true,
        timeout: Duration::from_secs(60),
        chain_strength_method: ChainStrengthMethod::Auto,
        custom_embedding: None,
        optimization_level: 2,
    };

    let client = DWaveClient::with_config(
        api_token,
        None, // Use default Leap endpoint
        solver_selector,
        embedding_config,
    )?;

    println!("✓ D-Wave Leap client initialized");

    // Example 2: Discover available solvers
    println!("\n2. Discovering Available Solvers");
    println!("--------------------------------");

    let solvers = client.get_leap_solvers()?;
    println!("Found {} solvers:", solvers.len());

    for solver in &solvers[..3.min(solvers.len())] {
        println!(
            "  - {} ({}): {} - Available: {}",
            solver.name,
            solver.id,
            match solver.solver_type {
                quantrs2_anneal::dwave::SolverType::QuantumProcessor => "QPU",
                quantrs2_anneal::dwave::SolverType::Hybrid => "Hybrid",
                quantrs2_anneal::dwave::SolverType::Software => "Software",
                _ => "Other",
            },
            solver.available
        );
        if let Some(avg_load) = solver.avg_load {
            println!("    Queue time: {:.1}s", avg_load);
        }
    }

    // Example 3: Automatic solver selection
    println!("\n3. Automatic Solver Selection");
    println!("-----------------------------");

    let best_qpu_solver = client.select_solver(Some(&SolverSelector {
        category: SolverCategory::QPU,
        online_only: true,
        ..Default::default()
    }));

    match best_qpu_solver {
        Ok(solver) => {
            println!("✓ Selected QPU solver: {} ({})", solver.name, solver.id);
            if let Some(load) = solver.avg_load {
                println!("  Current queue time: {:.1}s", load);
            }
        }
        Err(e) => {
            println!("⚠ No QPU solvers available: {}", e);
        }
    }

    let best_hybrid_solver = client.select_solver(Some(&SolverSelector {
        category: SolverCategory::Hybrid,
        ..Default::default()
    }));

    match best_hybrid_solver {
        Ok(solver) => {
            println!("✓ Selected hybrid solver: {} ({})", solver.name, solver.id);
        }
        Err(e) => {
            println!("⚠ No hybrid solvers available: {}", e);
        }
    }

    // Example 4: Create test problems
    println!("\n4. Creating Test Problems");
    println!("------------------------");

    // Small Max-Cut problem for QPU (if available)
    let mut small_problem = IsingModel::new(4);
    small_problem.set_coupling(0, 1, -1.0)?;
    small_problem.set_coupling(1, 2, -1.0)?;
    small_problem.set_coupling(2, 3, -1.0)?;
    small_problem.set_coupling(3, 0, -1.0)?;
    small_problem.set_coupling(0, 2, -1.0)?;
    small_problem.set_coupling(1, 3, -1.0)?;
    println!("✓ Created 4-qubit Max-Cut problem for QPU testing");

    // Larger problem for hybrid solver
    let mut large_problem = IsingModel::new(50);
    for i in 0..49 {
        large_problem.set_coupling(i, i + 1, -1.0)?;
        if i % 5 == 0 && i + 5 < 50 {
            large_problem.set_coupling(i, i + 5, -0.5)?;
        }
    }
    println!("✓ Created 50-variable problem for hybrid solver testing");

    // Example 5: Advanced problem submission with custom parameters
    println!("\n5. Advanced Problem Submission");
    println!("-----------------------------");

    // Custom annealing schedule
    let custom_schedule = AnnealingSchedule::pause_and_ramp(100.0, 50.0, 10.0);

    let advanced_params = AdvancedProblemParams {
        num_reads: 1000,
        anneal_schedule: Some(custom_schedule),
        programming_thermalization: Some(1000),
        readout_thermalization: Some(100),
        auto_scale: Some(true),
        chain_strength: Some(2.0),
        flux_biases: None,
        extra: HashMap::new(),
    };

    println!("Problem submission simulation (would submit to QPU with custom schedule):");
    println!("  - 1000 reads");
    println!("  - Custom pause-and-ramp annealing schedule");
    println!("  - Chain strength: 2.0");
    println!("  - Auto-scaling enabled");

    // Note: In a real implementation, you would actually submit:
    // let qpu_result = client.submit_ising_with_embedding(
    //     &small_problem,
    //     None, // Auto-select solver
    //     Some(advanced_params),
    //     None, // Use default embedding config
    // )?;

    // Example 6: Hybrid solver usage
    println!("\n6. Hybrid Solver Usage");
    println!("---------------------");

    let hybrid_params = HybridSolverParams {
        time_limit: Some(10.0), // 10 seconds
        max_variables: Some(1000),
        extra: {
            let mut extra = HashMap::new();
            extra.insert("seed".to_string(), serde_json::to_value(42)?);
            extra
        },
    };

    println!("Hybrid solver simulation (would submit large problem):");
    println!("  - Time limit: 10 seconds");
    println!("  - Max variables: 1000");
    println!("  - Random seed: 42");

    // Note: In a real implementation:
    // let hybrid_result = client.submit_hybrid(
    //     &large_problem,
    //     None, // Auto-select hybrid solver
    //     Some(hybrid_params),
    // )?;

    // Example 7: Batch problem submission
    println!("\n7. Batch Problem Submission");
    println!("--------------------------");

    let mut batch_problems = Vec::new();

    // Create multiple small problems
    for i in 0..3 {
        let mut problem = IsingModel::new(3);
        problem.set_coupling(0, 1, -1.0 - i as f64 * 0.1)?;
        problem.set_coupling(1, 2, -1.0 - i as f64 * 0.1)?;
        problem.set_coupling(2, 0, -1.0 - i as f64 * 0.1)?;
        batch_problems.push(problem);
    }

    println!(
        "Batch submission simulation (would submit {} problems):",
        batch_problems.len()
    );
    for (i, _) in batch_problems.iter().enumerate() {
        println!(
            "  Problem {}: 3-qubit triangle with coupling strength {:.1}",
            i + 1,
            -1.0 - i as f64 * 0.1
        );
    }

    // Note: In a real implementation:
    // let batch_problems_ref: Vec<_> = batch_problems.iter()
    //     .map(|p| (p, None, None))
    //     .collect();
    // let batch_result = client.submit_batch(batch_problems_ref)?;

    // Example 8: Problem monitoring and status tracking
    println!("\n8. Problem Monitoring Features");
    println!("-----------------------------");

    println!("Status tracking capabilities:");
    println!("  ✓ Real-time problem status monitoring");
    println!("  ✓ Automatic polling with timeout");
    println!("  ✓ Problem cancellation support");
    println!("  ✓ Performance metrics extraction");
    println!("  ✓ Queue time and execution time tracking");
    println!("  ✓ Chain break fraction analysis");

    // Example 9: Account usage monitoring
    println!("\n9. Account Usage Information");
    println!("---------------------------");

    println!("Usage monitoring features:");
    println!("  ✓ Real-time usage statistics");
    println!("  ✓ Solver access time tracking");
    println!("  ✓ Problem history management");
    println!("  ✓ Resource consumption analytics");

    // Example 10: Error handling and retry logic
    println!("\n10. Error Handling and Reliability");
    println!("----------------------------------");

    println!("Reliability features:");
    println!("  ✓ Automatic retry on network failures");
    println!("  ✓ Graceful handling of solver unavailability");
    println!("  ✓ Embedding failure recovery");
    println!("  ✓ Timeout management");
    println!("  ✓ Comprehensive error reporting");

    // Example 11: Performance optimization features
    println!("\n11. Performance Optimization");
    println!("----------------------------");

    println!("Optimization features:");
    println!("  ✓ Automatic chain strength calculation");
    println!("  ✓ Intelligent solver selection");
    println!("  ✓ Embedding optimization levels");
    println!("  ✓ Custom annealing schedules");
    println!("  ✓ Flux bias optimization");
    println!("  ✓ Batch processing for efficiency");

    println!("\n✅ D-Wave Leap Client Example completed successfully!");
    println!("\nNote: This example demonstrates the API structure and capabilities.");
    println!("To run actual quantum annealing jobs, you need:");
    println!("  1. A valid D-Wave Leap account with credits");
    println!("  2. Uncomment the actual submission code");
    println!("  3. Handle the returned Solution objects");

    Ok(())
}

#[cfg(feature = "dwave")]
/// Helper function to demonstrate problem result analysis
fn analyze_solution_results() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Solution Analysis Features");
    println!("============================");

    println!("Available analysis tools:");
    println!("  • Energy distribution analysis");
    println!("  • Solution quality metrics");
    println!("  • Chain break detection");
    println!("  • Timing performance breakdown");
    println!("  • Statistical significance testing");
    println!("  • Solution clustering");

    println!("\nMetrics automatically collected:");
    println!("  - Best energy found");
    println!("  - Energy gap analysis");
    println!("  - Solution degeneracy");
    println!("  - Sampling efficiency");
    println!("  - Hardware utilization");

    Ok(())
}

#[cfg(feature = "dwave")]
/// Helper function to demonstrate embedding optimization
fn demonstrate_embedding_strategies() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 Embedding Strategy Examples");
    println!("==============================");

    println!("Chain strength methods:");
    println!("  1. Auto: Automatic calculation based on problem");
    println!("  2. Fixed: User-specified constant value");
    println!("  3. Adaptive: Dynamic based on coupling strengths");

    println!("\nEmbedding optimization levels:");
    println!("  Level 1: Basic embedding with minimal optimization");
    println!("  Level 2: Balanced optimization for most problems");
    println!("  Level 3: Aggressive optimization for difficult problems");

    println!("\nCustom embedding options:");
    println!("  • Pre-computed embeddings");
    println!("  • Problem-specific optimizations");
    println!("  • Topology-aware placement");
    println!("  • Chain length minimization");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_problem_creation() {
        let mut problem = IsingModel::new(4);
        assert!(problem.set_coupling(0, 1, -1.0).is_ok());
        assert!(problem.set_coupling(1, 2, -1.0).is_ok());
        assert_eq!(problem.num_qubits, 4);
    }

    #[cfg(feature = "dwave")]
    #[test]
    fn test_solver_selector_creation() {
        let selector = SolverSelector {
            category: SolverCategory::QPU,
            min_qubits: Some(100),
            online_only: true,
            ..Default::default()
        };
        assert_eq!(selector.category, SolverCategory::QPU);
        assert_eq!(selector.min_qubits, Some(100));
        assert!(selector.online_only);
    }

    #[cfg(feature = "dwave")]
    #[test]
    fn test_advanced_params_creation() {
        let params = AdvancedProblemParams {
            num_reads: 1000,
            auto_scale: Some(true),
            ..Default::default()
        };
        assert_eq!(params.num_reads, 1000);
        assert_eq!(params.auto_scale, Some(true));
    }

    #[cfg(feature = "dwave")]
    #[test]
    fn test_annealing_schedule_creation() {
        let schedule = AnnealingSchedule::linear(100.0);
        assert_eq!(schedule.schedule.len(), 2);
        assert_eq!(schedule.schedule[0], (0.0, 1.0));
        assert_eq!(schedule.schedule[1], (100.0, 0.0));
    }

    #[cfg(feature = "dwave")]
    #[test]
    fn test_custom_annealing_schedule() {
        let schedule = AnnealingSchedule::pause_and_ramp(100.0, 50.0, 10.0);
        assert_eq!(schedule.schedule.len(), 4);
        // Check that pause is implemented correctly
        assert_eq!(schedule.schedule[1].1, schedule.schedule[2].1);
    }
}
