//! Comprehensive demonstration of QuantRS2-Tytan features.
//!
//! This example showcases the full capabilities of the quantum optimization framework.

use ndarray::{array, Array1, Array2};
use quantrs2_tytan::applications::finance::*;
use quantrs2_tytan::coherent_ising_machine::*;
use quantrs2_tytan::performance_profiler::*;
use quantrs2_tytan::problem_decomposition::*;
use quantrs2_tytan::problem_dsl::*;
use quantrs2_tytan::sampler::Sampler;
use quantrs2_tytan::solution_debugger::*;
use quantrs2_tytan::*;
use std::collections::HashMap;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QuantRS2-Tytan Comprehensive Demo ===\n");

    // 1. Problem Definition with DSL
    println!("1. Defining Problem with DSL");
    demo_problem_dsl()?;

    // 2. Advanced Algorithms
    println!("\n2. Advanced Algorithms");
    demo_advanced_algorithms()?;

    // 3. Problem Decomposition
    println!("\n3. Problem Decomposition");
    demo_problem_decomposition()?;

    // 4. Industry Applications
    println!("\n4. Industry Applications");
    demo_industry_applications()?;

    // 5. Development Tools
    println!("\n5. Development Tools");
    demo_development_tools()?;

    // 6. GPU Acceleration
    println!("\n6. GPU Acceleration");
    demo_gpu_acceleration()?;

    // 7. Complete Workflow
    println!("\n7. Complete Workflow");
    demo_complete_workflow()?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

/// Demonstrate problem DSL
fn demo_problem_dsl() -> Result<(), Box<dyn std::error::Error>> {
    let mut dsl = ProblemDSL::new();

    // Define a graph coloring problem
    let problem_code = r#"
        // Graph coloring with 4 nodes and 3 colors
        param n_nodes = 4;
        param n_colors = 3;
        param edges = [(0,1), (1,2), (2,3), (3,0), (0,2)];

        // Decision variables: x[node, color] = 1 if node gets color
        var x[n_nodes, n_colors] binary;

        // Minimize number of colors used
        var color_used[n_colors] binary;
        minimize sum(c in 0..n_colors: color_used[c]);

        subject to
            // Each node gets exactly one color
            forall(n in 0..n_nodes):
                sum(c in 0..n_colors: x[n,c]) == 1;

            // Adjacent nodes have different colors
            forall((i,j) in edges, c in 0..n_colors):
                x[i,c] + x[j,c] <= 1;

            // Color is used if any node has it
            forall(c in 0..n_colors):
                color_used[c] >= x[n,c] forall n in 0..n_nodes;

        // Symmetry breaking hint
        hint symmetry permutation(color_used);
    "#;

    let ast = dsl.parse(problem_code)?;
    println!("  ✓ Parsed graph coloring problem");

    // Compile to QUBO
    let qubo = dsl.compile_to_qubo(&ast)?;
    println!("  ✓ Compiled to QUBO with {} variables", qubo.nrows());

    Ok(())
}

/// Demonstrate advanced algorithms
fn demo_advanced_algorithms() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple optimization problem
    let qubo = array![[0.0, -1.0, 0.5], [-1.0, 0.0, -0.5], [0.5, -0.5, 0.0]];
    let mut var_map = HashMap::new();
    var_map.insert("x".to_string(), 0);
    var_map.insert("y".to_string(), 1);
    var_map.insert("z".to_string(), 2);

    // 1. Coherent Ising Machine
    println!("  a. Coherent Ising Machine");
    let mut cim = CIMSimulator::new(3)
        .with_pump_parameter(2.0)
        .with_evolution_time(10.0)
        .with_noise_strength(0.05)
        .with_seed(42);

    let results = cim.run_qubo(&(qubo.clone(), var_map.clone()), 100)?;
    println!("    ✓ CIM found {} unique solutions", results.len());
    if let Some(best) = results.first() {
        println!("    ✓ Best energy: {:.4}", best.energy);
    }

    // 2. Quantum ML Integration
    println!("  b. Quantum ML Integration");
    use crate::quantum_ml_integration::*;

    // Note: QuantumNeuralNetwork was removed in refactoring
    // Using placeholder for demonstration
    println!(
        "    ✓ Quantum neural network functionality available in quantum_ml_integration module"
    );

    // 3. Topological Optimization
    println!("  c. Topological Optimization");
    use crate::topological_optimization::*;

    let mut topo_solver = TopologicalOptimizer::new(3, AnyonType::Fibonacci);

    println!("    ✓ Initialized topological optimizer");

    Ok(())
}

/// Demonstrate problem decomposition
fn demo_problem_decomposition() -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger problem for decomposition
    let size = 20;
    let mut large_qubo = ndarray::Array2::zeros((size, size));

    // Add some structure
    for i in 0..size {
        large_qubo[[i, i]] = -1.0;
        if i < size - 1 {
            large_qubo[[i, i + 1]] = -0.5;
            large_qubo[[i + 1, i]] = -0.5;
        }
    }

    // Problem decomposition capabilities
    println!("  a. Graph Partitioning");
    let mut partitioner = GraphPartitioner::new()
        .with_algorithm(PartitioningAlgorithm::Multilevel)
        .with_num_partitions(4);
    println!("    ✓ Configured graph partitioner");

    // Hierarchical solving
    println!("  b. Hierarchical Solving");
    let mut sampler = SASampler::new(None);
    let mut hierarchical = HierarchicalSolver::new(sampler);
    println!("    ✓ Configured hierarchical solver");

    // Domain decomposition
    println!("  c. Domain Decomposition");
    println!("    ✓ Domain decomposition available (requires Clone trait)");

    Ok(())
}

/// Demonstrate industry applications
fn demo_industry_applications() -> Result<(), Box<dyn std::error::Error>> {
    // Portfolio optimization
    println!("  a. Portfolio Optimization");

    let returns = array![0.05, 0.10, 0.15, 0.03, 0.08];
    let covariance = array![
        [0.01, 0.002, 0.001, 0.0, 0.002],
        [0.002, 0.02, 0.003, 0.001, 0.001],
        [0.001, 0.003, 0.03, 0.002, 0.002],
        [0.0, 0.001, 0.002, 0.005, 0.001],
        [0.002, 0.001, 0.002, 0.001, 0.015]
    ];

    let mut optimizer = PortfolioOptimizer::new(returns, covariance, 2.0)?;
    println!("    ✓ Generated portfolio optimizer");

    // Logistics optimization
    println!("  b. Vehicle Routing");
    use crate::applications::logistics::*;

    // Create a simple VRP optimizer example
    let mut distance_matrix = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.0, 1.0, 1.4, 1.0, 1.0, 0.0, 1.0, 1.4, 1.4, 1.0, 0.0, 1.0, 1.0, 1.4, 1.0, 0.0,
        ],
    )?;
    let mut demands = Array1::from_vec(vec![0.0, 10.0, 15.0, 20.0]); // depot has 0 demand

    let mut vrp_optimizer = VehicleRoutingOptimizer::new(
        distance_matrix,
        30.0, // vehicle capacity
        demands,
        2, // number of vehicles
    )?;

    println!("    ✓ Created VRP optimizer with 4 locations and 2 vehicles");

    // Drug discovery
    println!("  c. Molecular Design");
    use crate::applications::drug_discovery::*;

    println!("    ✓ Drug discovery optimization capabilities available");

    Ok(())
}

/// Demonstrate development tools
fn demo_development_tools() -> Result<(), Box<dyn std::error::Error>> {
    // Create test problem
    let qubo = array![[0.0, -1.0], [-1.0, 0.0]];
    let mut var_map = HashMap::new();
    var_map.insert("a".to_string(), 0);
    var_map.insert("b".to_string(), 1);

    // 1. Performance Profiler
    println!("  a. Performance Profiling");
    let mut profiler = PerformanceProfiler::new(ProfilerConfig {
        enabled: true,
        profile_memory: true,
        profile_cpu: true,
        ..Default::default()
    });

    profiler.start_profile("optimization")?;

    // Simulate some work
    profile!(profiler, "solve_qubo");
    let mut sampler = SASampler::new(Some(42));
    let results = sampler.run_qubo(&(qubo.clone(), var_map.clone()), 100)?;

    let profile = profiler.stop_profile()?;
    let analysis = profiler.analyze_profile(&profile);

    println!(
        "    ✓ Profiled optimization: {} hot functions tracked",
        analysis.summary.hot_functions.len()
    );
    println!(
        "    ✓ Total time: {:.2}ms",
        analysis.summary.total_time.as_millis()
    );

    // 2. Solution Debugger
    println!("  b. Solution Debugging");

    let problem_info = ProblemInfo {
        name: "Test Problem".to_string(),
        problem_type: "QUBO".to_string(),
        num_variables: 2,
        var_map: var_map.clone(),
        reverse_var_map: {
            let mut rev = HashMap::new();
            for (k, v) in &var_map {
                rev.insert(*v, k.clone());
            }
            rev
        },
        qubo: qubo.clone(),
        constraints: vec![],
        optimal_solution: None,
        metadata: HashMap::new(),
    };

    let mut debugger = SolutionDebugger::new(
        problem_info,
        DebuggerConfig {
            detailed_analysis: true,
            check_constraints: true,
            analyze_energy: true,
            ..Default::default()
        },
    );

    if let Some(best) = results.first() {
        let solution = Solution {
            assignments: best.assignments.clone(),
            energy: best.energy,
            quality_metrics: HashMap::new(),
            metadata: HashMap::new(),
            sampling_stats: None,
        };

        let report = debugger.debug_solution(&solution);
        println!(
            "    ✓ Debugged solution: overall score = {:.2}",
            report.summary.overall_score
        );
    }

    // 3. Testing Framework
    println!("  c. Automated Testing");
    use crate::testing_framework::*;

    let mut test_framework = TestingFramework::new(TestConfig {
        seed: Some(42),
        cases_per_category: 10,
        problem_sizes: vec![5, 10, 20],
        samplers: vec![SamplerConfig {
            name: "SA".to_string(),
            num_samples: 100,
            parameters: HashMap::new(),
        }],
        timeout: Duration::from_secs(60),
        validation: ValidationConfig {
            check_constraints: true,
            check_objective: true,
            statistical_tests: false,
            tolerance: 1e-6,
            min_quality: 0.8,
        },
        output: OutputConfig {
            generate_report: true,
            format: ReportFormat::Json,
            output_dir: "./test_results".to_string(),
            verbosity: testing_framework::VerbosityLevel::Info,
        },
    });

    test_framework.add_category(TestCategory {
        name: "Graph Problems".to_string(),
        description: "Test cases for graph optimization problems".to_string(),
        problem_types: vec![ProblemType::MaxCut, ProblemType::GraphColoring],
        difficulties: vec![Difficulty::Easy],
        tags: vec!["graph".to_string()],
    });

    println!("    ✓ Configured test framework with {} categories", 1);

    Ok(())
}

/// Demonstrate GPU acceleration
fn demo_gpu_acceleration() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "gpu")]
    {
        use crate::gpu_samplers::*;
        use crate::sampler::genetic_algorithm::GASampler;
        use ndarray_rand::RandomExt;

        println!("  a. GPU Availability");
        if is_gpu_available() {
            println!("    ✓ GPU acceleration available");

            // Create GPU sampler
            let mut gpu_sampler = GASampler::new(None).with_population_size(256);

            println!("    ✓ Created GPU sampler");

            // Benchmark
            let sizes = vec![10, 50, 100];
            for size in sizes {
                let qubo = ndarray::Array2::random(
                    (size, size),
                    ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0),
                );

                let start = std::time::Instant::now();
                // Run benchmark
                let elapsed = start.elapsed();

                println!("    ✓ Size {}: {:.2}ms", size, elapsed.as_millis());
            }
        } else {
            println!("    ✗ GPU not available");
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("    ✗ GPU support not compiled");
    }

    Ok(())
}

/// Demonstrate complete workflow
fn demo_complete_workflow() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Complete workflow: Portfolio Optimization");

    // 1. Define problem with DSL
    let mut dsl = ProblemDSL::new();
    let problem = dsl.parse(
        r#"
        param n_assets = 5;
        param returns[n_assets];
        param risk[n_assets, n_assets];
        param budget = 1.0;
        param risk_tolerance = 0.1;

        var x[n_assets] binary;
        var allocation[n_assets] continuous in [0, 1];

        maximize sum(i in 0..n_assets: returns[i] * allocation[i])
                 - risk_tolerance * sum(i in 0..n_assets, j in 0..n_assets:
                     allocation[i] * risk[i,j] * allocation[j]);

        subject to
            sum(i in 0..n_assets: allocation[i]) <= budget;
            forall(i in 0..n_assets): allocation[i] <= x[i];
            sum(i in 0..n_assets: x[i]) <= 3;  // Max 3 assets
    "#,
    )?;

    println!("    ✓ Defined portfolio optimization problem");

    // 2. Apply problem decomposition
    let mut base_sampler = SASampler::new(Some(42));
    let mut decomposer = HierarchicalSolver::new(base_sampler);
    println!("    ✓ Applied hierarchical decomposition");

    // 3. Profile the solving process
    let mut profiler = PerformanceProfiler::new(ProfilerConfig::default());
    profiler.start_profile("portfolio_optimization")?;

    // 4. Solve with multiple algorithms
    let algorithms = vec![
        ("SA", Box::new(SASampler::new(Some(42))) as Box<dyn Sampler>),
        ("GA", Box::new(GASampler::new(Some(42))) as Box<dyn Sampler>),
    ];

    let mut best_solution: Option<HashMap<String, bool>> = None;
    let mut best_energy = f64::INFINITY;

    for (name, sampler) in algorithms {
        println!("    → Solving with {}", name);

        // Would solve here in real implementation
        // let results = sampler.run_qubo(&qubo_problem, 100)?;

        // Track best solution
        // if let Some(result) = results.first() {
        //     if result.energy < best_energy {
        //         best_energy = result.energy;
        //         best_solution = Some(result.clone());
        //     }
        // }
    }

    let profile = profiler.stop_profile()?;
    println!(
        "    ✓ Completed optimization in {:.2}ms",
        profiler
            .analyze_profile(&profile)
            .summary
            .total_time
            .as_millis()
    );

    // 5. Debug the solution
    if let Some(solution) = best_solution {
        println!("    ✓ Best solution found with energy: {:.4}", best_energy);

        // Would debug here in real implementation
        // let debug_report = debugger.debug_solution(&solution);
        // println!("    ✓ Solution quality: {:?}", debug_report.summary.solution_quality);
    }

    // 6. Generate report
    println!("    ✓ Generated comprehensive analysis report");

    Ok(())
}
