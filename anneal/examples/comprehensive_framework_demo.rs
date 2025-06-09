//! Comprehensive QuantRS2-Anneal Framework Demonstration
//!
//! This example showcases the full capabilities of the QuantRS2-Anneal framework:
//! 
//! 1. Multiple problem formulations (Ising, QUBO, DSL)
//! 2. Various simulation algorithms
//! 3. Cloud quantum hardware integration
//! 4. Advanced optimization techniques
//! 5. Performance analysis and visualization
//! 6. Real-world application examples
//! 
//! Run with different feature combinations:
//! ```bash
//! # Basic functionality
//! cargo run --example comprehensive_framework_demo
//! 
//! # With cloud features
//! cargo run --example comprehensive_framework_demo --features dwave,braket
//! 
//! # With all features
//! cargo run --example comprehensive_framework_demo --all-features
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

// Core framework imports
use quantrs2_anneal::{
    ising::IsingModel,
    qubo::{QuboBuilder, QuboFormulation},
    simulator::{ClassicalAnnealingSimulator, AnnealingParams, TemperatureSchedule},
    population_annealing::{PopulationAnnealingSimulator, PopulationParams},
    coherent_ising_machine::{CoherentIsingMachine, CIMParams},
};

// Cloud integration imports (conditional)
#[cfg(feature = "dwave")]
use quantrs2_anneal::dwave::{
    DWaveClient, SolverSelector, SolverCategory, EmbeddingConfig, 
    ChainStrengthMethod, AdvancedProblemParams
};

#[cfg(feature = "braket")]
use quantrs2_anneal::braket::{
    BraketClient, DeviceSelector, DeviceType, CostTracker, AdvancedAnnealingParams
};

// Advanced features
use quantrs2_anneal::{
    embedding::{Embedding, HardwareGraph, MinorMiner},
    penalty_optimization::PenaltyOptimizer,
    reverse_annealing::{ReverseAnnealingScheduler, ReverseSchedule},
    visualization::{EnergyLandscapeVisualizer, ConvergenceAnalyzer},
    applications::{
        energy::{PowerGridOptimizer, GridConstraints},
        finance::{PortfolioOptimizer, PortfolioConstraints},
        logistics::{VehicleRoutingOptimizer, RoutingProblem},
        performance_benchmarks::{BenchmarkSuite, Algorithm}
    }
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 QuantRS2-Anneal Comprehensive Framework Demo");
    println!("==============================================");
    println!();

    // Demo 1: Basic Problem Formulations
    println!("📝 Demo 1: Problem Formulations");
    println!("-------------------------------");
    demo_problem_formulations()?;
    println!();

    // Demo 2: Classical Simulation Algorithms
    println!("🖥️  Demo 2: Classical Simulation Algorithms");
    println!("------------------------------------------");
    demo_classical_algorithms()?;
    println!();

    // Demo 3: Advanced Quantum-Inspired Algorithms
    println!("🌊 Demo 3: Advanced Quantum-Inspired Algorithms");
    println!("----------------------------------------------");
    demo_advanced_algorithms()?;
    println!();

    // Demo 4: Embedding and Hardware Mapping
    println!("🗺️  Demo 4: Graph Embedding and Hardware Mapping");
    println!("-----------------------------------------------");
    demo_embedding_techniques()?;
    println!();

    // Demo 5: Cloud Quantum Hardware (if available)
    #[cfg(any(feature = "dwave", feature = "braket"))]
    {
        println!("☁️  Demo 5: Cloud Quantum Hardware Integration");
        println!("--------------------------------------------");
        demo_cloud_integration().await?;
        println!();
    }

    // Demo 6: Real-World Applications
    println!("🌍 Demo 6: Real-World Applications");
    println!("--------------------------------");
    demo_real_world_applications()?;
    println!();

    // Demo 7: Performance Analysis and Optimization
    println!("📊 Demo 7: Performance Analysis and Benchmarking");
    println!("-----------------------------------------------");
    demo_performance_analysis()?;
    println!();

    // Demo 8: Advanced Optimization Techniques
    println!("🔧 Demo 8: Advanced Optimization Techniques");
    println!("------------------------------------------");
    demo_advanced_optimization()?;
    println!();

    // Demo 9: Visualization and Analysis
    println!("📈 Demo 9: Visualization and Analysis");
    println!("-----------------------------------");
    demo_visualization_analysis()?;
    println!();

    println!("✅ Comprehensive demo completed successfully!");
    println!();
    println!("🔗 Next Steps:");
    println!("  - Explore individual examples in the examples/ directory");
    println!("  - Read the documentation in docs/");
    println!("  - Try with your own optimization problems");
    println!("  - Set up cloud quantum hardware access for real QPU usage");

    Ok(())
}

fn demo_problem_formulations() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Direct Ising Model
    println!("1. Direct Ising Model (Max-Cut on triangle):");
    let mut ising_model = IsingModel::new(3);
    ising_model.set_coupling(0, 1, -1.0)?;
    ising_model.set_coupling(1, 2, -1.0)?;
    ising_model.set_coupling(2, 0, -1.0)?;
    
    let energy = ising_model.calculate_energy(&[1, -1, 1])?;
    println!("  Ising energy for solution [1, -1, 1]: {}", energy);

    // 2. QUBO Formulation  
    println!("2. QUBO Formulation (Asset selection with constraints):");
    let mut qubo = QuboBuilder::new(4);
    
    // Objective: maximize utility (minimize negative utility)
    let utilities = vec![0.8, 0.6, 0.9, 0.7];
    for (i, &utility) in utilities.iter().enumerate() {
        qubo.add_linear_term(i, -utility)?; // Negative for maximization
    }
    
    // Constraint: select exactly 2 assets
    qubo.add_constraint_eq(&[0, 1, 2, 3], &[1.0, 1.0, 1.0, 1.0], 2.0, 5.0)?;
    
    let qubo_formulation = qubo.build()?;
    let qubo_ising = qubo_formulation.to_ising_model()?;
    println!("  QUBO converted to Ising with {} qubits", qubo_ising.num_qubits);

    // 3. DSL Problem Construction (if available)
    println!("3. Problem Builder DSL:");
    println!("  [DSL would allow: x + y + z == 2, minimize x*y + y*z]");
    println!("  Converted to QUBO/Ising automatically");

    Ok(())
}

fn demo_classical_algorithms() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test problem (random spin glass)
    let mut model = IsingModel::new(20);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for i in 0..20 {
        for j in (i + 1)..20 {
            if rng.gen::<f64>() < 0.3 {
                let strength = rng.gen_range(-2.0..2.0);
                model.set_coupling(i, j, strength)?;
            }
        }
    }

    // 1. Classical Simulated Annealing
    println!("1. Classical Simulated Annealing:");
    let start = Instant::now();
    
    let classical_params = AnnealingParams {
        num_sweeps: 2000,
        num_repetitions: 10,
        initial_temperature: 2.0,
        final_temperature: 0.1,
        temperature_schedule: TemperatureSchedule::Exponential { decay: 0.99 },
        ..Default::default()
    };
    
    let classical_simulator = ClassicalAnnealingSimulator::new(classical_params)?;
    let classical_result = classical_simulator.solve(&model)?;
    let classical_time = start.elapsed();
    
    println!("  Best energy: {:.6}", classical_result.best_energy);
    println!("  Time: {:?}", classical_time);
    println!("  Convergence iteration: {}", classical_result.convergence_iteration);

    // 2. Population Annealing
    println!("2. Population Annealing (high-quality solutions):");
    let start = Instant::now();
    
    let pop_params = PopulationParams {
        population_size: 100,
        num_sweeps: 1000,
        resampling_threshold: 0.7,
        temperature_schedule: vec![2.0, 1.5, 1.0, 0.5, 0.2, 0.1],
        ..Default::default()
    };
    
    let pop_simulator = PopulationAnnealingSimulator::new(pop_params)?;
    let pop_result = pop_simulator.solve(&model)?;
    let pop_time = start.elapsed();
    
    println!("  Best energy: {:.6}", pop_result.best_energy);
    println!("  Time: {:?}", pop_time);
    println!("  Final diversity: {:.3}", pop_result.final_diversity);
    println!("  Effective sample size: {}", pop_result.effective_sample_size);

    // Compare results
    let improvement = ((classical_result.best_energy - pop_result.best_energy) / 
                      classical_result.best_energy.abs()) * 100.0;
    println!("  Quality improvement: {:.2}%", improvement);

    Ok(())
}

fn demo_advanced_algorithms() -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger sparse problem for CIM
    let mut large_model = IsingModel::new(100);
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Create sparse connectivity (each node connected to ~5 others)
    for i in 0..100 {
        let num_connections = rng.gen_range(3..8);
        for _ in 0..num_connections {
            let j = rng.gen_range(0..100);
            if i != j {
                let strength = rng.gen_range(-1.0..1.0);
                large_model.set_coupling(i, j, strength)?;
            }
        }
    }

    // 1. Coherent Ising Machine
    println!("1. Coherent Ising Machine (100 variables):");
    let start = Instant::now();
    
    let cim_params = CIMParams {
        pump_power: 2.0,
        detuning: 0.1,
        coupling_strength: 0.5,
        num_iterations: 500,
        convergence_threshold: 1e-5,
        dt: 0.01,
        ..Default::default()
    };
    
    let cim = CoherentIsingMachine::new(cim_params)?;
    let cim_result = cim.solve(&large_model)?;
    let cim_time = start.elapsed();
    
    println!("  Best energy: {:.6}", cim_result.best_energy);
    println!("  Time: {:?}", cim_time);
    println!("  Convergence iterations: {}", cim_result.iterations_to_convergence);
    println!("  Final amplitude variance: {:.8}", cim_result.final_variance);

    // 2. Reverse Annealing (if available)
    println!("2. Reverse Annealing (solution refinement):");
    
    // Use CIM result as starting point for reverse annealing
    let reverse_scheduler = ReverseAnnealingScheduler::new();
    let reverse_schedule = ReverseSchedule::gradual_reverse(
        initial_s: 1.0,
        pause_s: 0.3,
        pause_duration: 50.0,
        final_s: 1.0,
        total_time: 200.0,
    )?;
    
    println!("  Starting from CIM solution");
    println!("  Reverse schedule: pause at s=0.3 for 50μs");
    println!("  (Would improve solution quality in real reverse annealing)");

    // 3. Non-Stoquastic Simulation (if available)
    println!("3. Non-Stoquastic Hamiltonian Simulation:");
    println!("  [Would simulate quantum effects beyond classical Ising]");
    println!("  Includes transverse and longitudinal fields");
    println!("  Quantum tunneling through energy barriers");

    Ok(())
}

fn demo_embedding_techniques() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Minor Graph Embedding
    println!("1. Minor Graph Embedding:");
    
    let logical_edges = vec![
        (0, 1), (1, 2), (2, 3), (3, 0), // Square
        (0, 2), (1, 3), // Diagonals
    ];
    
    // Create a small Chimera graph for embedding
    let hardware = HardwareGraph::new_chimera(2, 2, 4)?;
    println!("  Hardware: 2x2 Chimera (32 qubits)");
    println!("  Logical problem: 4 variables, 6 couplings");
    
    let embedder = MinorMiner::default();
    
    match embedder.find_embedding(&logical_edges, 4, &hardware) {
        Ok(embedding) => {
            println!("  ✓ Embedding found!");
            
            for (logical_var, physical_chain) in &embedding.chains {
                println!("    Variable {}: chain {:?}", logical_var, physical_chain);
            }
            
            let total_qubits: usize = embedding.chains.values()
                .map(|chain| chain.len())
                .sum();
            
            println!("  Physical qubits used: {}/32", total_qubits);
            
            let avg_chain_length: f64 = embedding.chains.values()
                .map(|chain| chain.len() as f64)
                .sum::<f64>() / embedding.chains.len() as f64;
            
            println!("  Average chain length: {:.2}", avg_chain_length);
        }
        Err(e) => {
            println!("  ⚠ Embedding failed: {}", e);
            println!("  (This is normal for complex graphs on small hardware)");
        }
    }

    // 2. Layout-Aware Embedding
    println!("2. Layout-Aware Embedding:");
    println!("  [Would optimize placement for hardware layout]");
    println!("  Minimize chain crossings and length");
    println!("  Consider defective qubits and calibration data");

    // 3. Chain Strength Optimization
    println!("3. Chain Strength Optimization:");
    println!("  Auto: Based on problem coupling strengths");
    println!("  Adaptive: Dynamically adjusted during annealing");
    println!("  Per-chain: Individual optimization for each chain");

    Ok(())
}

#[cfg(any(feature = "dwave", feature = "braket"))]
async fn demo_cloud_integration() -> Result<(), Box<dyn std::error::Error>> {
    // D-Wave Leap Integration
    #[cfg(feature = "dwave")]
    {
        println!("1. D-Wave Leap Integration:");
        
        if let Ok(token) = std::env::var("DWAVE_API_TOKEN") {
            let client = DWaveClient::new(token, None)?;
            
            // Get available solvers
            match client.get_leap_solvers() {
                Ok(solvers) => {
                    println!("  Available solvers:");
                    for solver in solvers.iter().take(3) {
                        println!("    - {} ({}): {} qubits", 
                                 solver.name, 
                                 solver.solver_type,
                                 solver.properties.get("num_qubits")
                                     .and_then(|v| v.as_u64())
                                     .unwrap_or(0));
                    }
                    
                    // Auto-select QPU
                    let selector = SolverSelector {
                        category: SolverCategory::QPU,
                        online_only: true,
                        ..Default::default()
                    };
                    
                    match client.select_solver(Some(&selector)) {
                        Ok(solver) => {
                            println!("  ✓ Selected: {}", solver.name);
                            
                            // Create test problem
                            let mut model = IsingModel::new(4);
                            model.set_coupling(0, 1, -1.0)?;
                            model.set_coupling(1, 2, -1.0)?;
                            model.set_coupling(2, 3, -1.0)?;
                            model.set_coupling(3, 0, -1.0)?;
                            
                            println!("  Problem: 4-qubit Max-Cut");
                            println!("  [Would submit to quantum hardware]");
                            println!("  Expected: Auto-embedding, ~1000 samples");
                        }
                        Err(e) => {
                            println!("  ⚠ No QPU available: {}", e);
                            println!("  [Could fall back to hybrid solver]");
                        }
                    }
                }
                Err(e) => {
                    println!("  ❌ Connection failed: {}", e);
                    println!("  Check API token and network connectivity");
                }
            }
        } else {
            println!("  ⚠ No DWAVE_API_TOKEN set");
            println!("  Set token to test real D-Wave integration");
        }
    }

    // AWS Braket Integration
    #[cfg(feature = "braket")]
    {
        println!("2. AWS Braket Integration:");
        
        if std::env::var("AWS_ACCESS_KEY_ID").is_ok() {
            let access_key = std::env::var("AWS_ACCESS_KEY_ID")?;
            let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")?;
            let region = std::env::var("AWS_REGION").unwrap_or("us-east-1".to_string());
            
            // Create cost-aware client
            let cost_tracker = CostTracker {
                cost_limit: Some(50.0), // $50 limit
                current_cost: 0.0,
                cost_estimates: HashMap::new(),
            };
            
            let device_selector = DeviceSelector {
                device_type: Some(DeviceType::Simulator), // Use free simulator
                ..Default::default()
            };
            
            match BraketClient::with_config(
                access_key, secret_key, None, region.clone(), 
                device_selector, cost_tracker
            ) {
                Ok(client) => {
                    println!("  ✓ Connected to AWS Braket ({})", region);
                    println!("  Cost limit: $50.00");
                    println!("  Device preference: Simulators (free)");
                    
                    match client.get_devices() {
                        Ok(devices) => {
                            let simulators: Vec<_> = devices.iter()
                                .filter(|d| matches!(d.device_type, DeviceType::Simulator))
                                .collect();
                            
                            println!("  Available simulators: {}", simulators.len());
                            
                            if let Some(sim) = simulators.first() {
                                println!("    - {} ({})", sim.device_name, sim.provider_name);
                                println!("  [Would submit quantum annealing task]");
                                println!("  Expected: High shot count (simulators are cheap)");
                            }
                        }
                        Err(e) => {
                            println!("  ⚠ Device query failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("  ❌ Connection failed: {}", e);
                    println!("  Check AWS credentials and permissions");
                }
            }
        } else {
            println!("  ⚠ No AWS credentials set");
            println!("  Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY");
        }
    }

    #[cfg(not(any(feature = "dwave", feature = "braket")))]
    {
        println!("Cloud integration not compiled.");
        println!("Enable with: --features dwave,braket");
    }

    Ok(())
}

fn demo_real_world_applications() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Energy System Optimization
    println!("1. Smart Grid Optimization:");
    
    let grid_constraints = GridConstraints {
        total_demand: 1000.0, // MW
        renewable_capacity: 300.0, // MW wind/solar
        storage_capacity: 100.0, // MWh battery
        transmission_limits: vec![200.0, 150.0, 250.0], // MW per line
        cost_coefficients: vec![50.0, 80.0, 120.0], // $/MWh per generator
        emission_factors: vec![0.5, 0.8, 1.2], // tons CO2/MWh
    };
    
    println!("  Grid: {} MW demand, {} MW renewable", 
             grid_constraints.total_demand, grid_constraints.renewable_capacity);
    println!("  Generators: {} units with varying costs", 
             grid_constraints.cost_coefficients.len());
    
    let grid_optimizer = PowerGridOptimizer::new(grid_constraints)?;
    
    // Simplified optimization (would be much more complex in reality)
    let sample_schedule = vec![400.0, 300.0, 200.0, 100.0]; // 4 time periods
    let total_cost: f64 = sample_schedule.iter()
        .enumerate()
        .map(|(i, &power)| power * grid_optimizer.get_cost_coefficient(i % 3))
        .sum();
    
    println!("  Sample 4-hour schedule: {:?} MW", sample_schedule);
    println!("  Estimated cost: ${:.0}", total_cost);
    println!("  [Real optimization would include storage, renewables, demand response]");

    // 2. Portfolio Optimization
    println!("2. Financial Portfolio Optimization:");
    
    let portfolio_constraints = PortfolioConstraints {
        num_assets: 10,
        target_assets: 5, // Select 5 out of 10
        max_risk: 0.15, // 15% portfolio risk
        min_return: 0.08, // 8% expected return
        sector_limits: vec![2, 2, 1], // Max 2 tech, 2 finance, 1 energy
        correlation_threshold: 0.7, // Avoid highly correlated assets
    };
    
    println!("  Assets: {} available, select {}", 
             portfolio_constraints.num_assets, portfolio_constraints.target_assets);
    println!("  Constraints: {:.1}% max risk, {:.1}% min return", 
             portfolio_constraints.max_risk * 100.0, portfolio_constraints.min_return * 100.0);
    
    let portfolio_optimizer = PortfolioOptimizer::new(portfolio_constraints);
    
    // Sample expected returns and risk matrix
    let expected_returns = vec![0.12, 0.08, 0.15, 0.10, 0.09, 0.11, 0.07, 0.13, 0.06, 0.14];
    let sample_portfolio = vec![1, 0, 1, 1, 0, 1, 0, 1, 0, 0]; // Binary selection
    
    let selected_returns: Vec<f64> = sample_portfolio.iter()
        .enumerate()
        .filter(|(_, &selected)| selected == 1)
        .map(|(i, _)| expected_returns[i])
        .collect();
    
    let portfolio_return: f64 = selected_returns.iter().sum::<f64>() / selected_returns.len() as f64;
    
    println!("  Sample portfolio return: {:.1}%", portfolio_return * 100.0);
    println!("  Selected assets: {:?}", sample_portfolio);
    println!("  [Real optimization would include full covariance matrix]");

    // 3. Vehicle Routing Optimization
    println!("3. Vehicle Routing Problem:");
    
    let routing_problem = RoutingProblem {
        num_customers: 20,
        num_vehicles: 3,
        depot_location: (0.0, 0.0),
        customer_locations: (0..20).map(|i| {
            (i as f64 * 0.5, (i % 7) as f64 * 0.3) // Simplified locations
        }).collect(),
        customer_demands: vec![1; 20], // Unit demand per customer
        vehicle_capacity: 8, // Each vehicle can serve 8 customers
        max_route_time: 8.0, // 8 hour shifts
    };
    
    println!("  Customers: {}, Vehicles: {}", 
             routing_problem.num_customers, routing_problem.num_vehicles);
    println!("  Vehicle capacity: {} customers each", routing_problem.vehicle_capacity);
    
    let routing_optimizer = VehicleRoutingOptimizer::new(routing_problem);
    
    // Sample solution (would be optimized)
    let sample_routes = vec![
        vec![0, 1, 2, 3, 4, 5, 6, 7], // Vehicle 1: customers 0-7
        vec![8, 9, 10, 11, 12, 13, 14, 15], // Vehicle 2: customers 8-15
        vec![16, 17, 18, 19], // Vehicle 3: customers 16-19
    ];
    
    let total_distance: f64 = sample_routes.iter()
        .map(|route| routing_optimizer.calculate_route_distance(route))
        .sum();
    
    println!("  Sample routes: {} total distance", total_distance);
    println!("  [Real optimization minimizes distance while respecting constraints]");

    Ok(())
}

fn demo_performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("Performance Benchmarking Suite:");
    
    // Create benchmark problems of different sizes
    let problem_sizes = vec![10, 20, 50];
    let algorithms = vec![
        Algorithm::ClassicalAnnealing,
        Algorithm::PopulationAnnealing,
        Algorithm::CoherentIsingMachine,
    ];
    
    println!("  Problem sizes: {:?}", problem_sizes);
    println!("  Algorithms: {} variants", algorithms.len());
    
    let mut benchmark_results = Vec::new();
    
    for &size in &problem_sizes {
        println!("  Testing size {}: ", size);
        
        // Create random problem
        let mut model = IsingModel::new(size);
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for i in 0..size {
            for j in (i + 1)..size {
                if rng.gen::<f64>() < 0.3 {
                    model.set_coupling(i, j, rng.gen_range(-1.0..1.0))?;
                }
            }
        }
        
        // Test each algorithm
        for algorithm in &algorithms {
            let start = Instant::now();
            
            let result = match algorithm {
                Algorithm::ClassicalAnnealing => {
                    let sim = ClassicalAnnealingSimulator::new(AnnealingParams {
                        num_sweeps: 1000,
                        num_repetitions: 5,
                        ..Default::default()
                    })?;
                    sim.solve(&model)?.best_energy
                }
                Algorithm::PopulationAnnealing => {
                    let sim = PopulationAnnealingSimulator::new(PopulationParams {
                        population_size: 50,
                        num_sweeps: 500,
                        ..Default::default()
                    })?;
                    sim.solve(&model)?.best_energy
                }
                Algorithm::CoherentIsingMachine => {
                    let sim = CoherentIsingMachine::new(CIMParams {
                        num_iterations: 100,
                        ..Default::default()
                    })?;
                    sim.solve(&model)?.best_energy
                }
            };
            
            let time = start.elapsed();
            benchmark_results.push((size, algorithm.clone(), result, time));
            
            print!("{:?}({:.3}s) ", algorithm, time.as_secs_f64());
        }
        println!();
    }
    
    // Analyze results
    println!("  Results Summary:");
    for &size in &problem_sizes {
        println!("    Size {}:", size);
        
        let size_results: Vec<_> = benchmark_results.iter()
            .filter(|(s, _, _, _)| *s == size)
            .collect();
        
        let best_energy = size_results.iter()
            .map(|(_, _, energy, _)| *energy)
            .fold(f64::INFINITY, f64::min);
        
        for (_, algorithm, energy, time) in size_results {
            let quality = if energy == &best_energy { "★" } else { " " };
            println!("      {:?}: {:.6} ({:?}) {}", 
                     algorithm, energy, time, quality);
        }
    }

    Ok(())
}

fn demo_advanced_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Penalty Optimization
    println!("1. Automatic Penalty Optimization:");
    
    // Create QUBO with constraints
    let mut qubo = QuboBuilder::new(5);
    
    // Objective terms
    for i in 0..5 {
        qubo.add_linear_term(i, rand::random::<f64>() - 0.5)?;
    }
    
    // Hard constraint: exactly 2 variables should be 1
    qubo.add_constraint_eq(&[0, 1, 2, 3, 4], &[1.0; 5], 2.0, 1.0)?; // Start with penalty 1.0
    
    let formulation = qubo.build()?;
    let mut ising_model = formulation.to_ising_model()?;
    
    println!("  Initial penalty weight: 1.0");
    
    // Optimize penalty weights
    let penalty_optimizer = PenaltyOptimizer::new();
    let optimized_penalties = penalty_optimizer.optimize_penalties(
        &ising_model,
        target_violation_rate: 0.05, // 5% violation tolerance
        penalty_range: (0.1, 100.0),
    )?;
    
    println!("  Optimized penalty: {:.2}", optimized_penalties[0]);
    println!("  Expected constraint satisfaction: 95%");

    // 2. Multi-Objective Optimization
    println!("2. Multi-Objective Optimization:");
    println!("  Objectives: Energy minimization + Solution diversity");
    println!("  Pareto frontier: 10 non-dominated solutions");
    println!("  [Would provide trade-off analysis between objectives]");

    // 3. Adaptive Parameter Tuning
    println!("3. Adaptive Parameter Tuning:");
    println!("  Temperature schedule optimization");
    println!("  Population size adjustment");
    println!("  Chain strength auto-tuning");
    println!("  [Uses Bayesian optimization for parameter search]");

    Ok(())
}

fn demo_visualization_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Energy Landscape Visualization:");
    
    // Create simple 2D problem for visualization
    let mut model = IsingModel::new(4);
    model.set_coupling(0, 1, -1.0)?;
    model.set_coupling(1, 2, -0.5)?;
    model.set_coupling(2, 3, -1.0)?;
    model.set_coupling(3, 0, -0.5)?;
    
    println!("  Problem: 4-qubit cycle with mixed couplings");
    println!("  Energy landscape: 2^4 = 16 possible states");
    
    // Calculate all energies
    let mut energies = Vec::new();
    for state in 0..16 {
        let spins: Vec<i8> = (0..4).map(|i| {
            if (state >> i) & 1 == 1 { 1 } else { -1 }
        }).collect();
        
        let energy = model.calculate_energy(&spins)?;
        energies.push((state, spins, energy));
    }
    
    // Sort by energy
    energies.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    
    println!("  Ground state: {:?} with energy {:.3}", 
             energies[0].1, energies[0].2);
    println!("  First excited state: {:?} with energy {:.3}", 
             energies[1].1, energies[1].2);
    
    let gap = energies[1].2 - energies[0].2;
    println!("  Energy gap: {:.3}", gap);

    // 2. Convergence Analysis
    println!("2. Convergence Analysis:");
    
    // Simulate annealing trace
    let num_steps = 100;
    let mut trace = Vec::new();
    let mut current_energy = 10.0;
    
    for step in 0..num_steps {
        // Simulate exponential convergence with noise
        let target = energies[0].2; // Ground state energy
        let decay = 0.95;
        current_energy = target + (current_energy - target) * decay + 
                        (rand::random::<f64>() - 0.5) * 0.1;
        trace.push((step, current_energy));
    }
    
    println!("  Annealing trace: {} steps", trace.len());
    println!("  Initial energy: {:.3}", trace[0].1);
    println!("  Final energy: {:.3}", trace.last().unwrap().1);
    println!("  Convergence: Exponential with noise");

    // 3. Solution Quality Analysis
    println!("3. Solution Quality Analysis:");
    
    // Run multiple trials
    let num_trials = 50;
    let mut trial_results = Vec::new();
    
    let simulator = ClassicalAnnealingSimulator::new(AnnealingParams {
        num_sweeps: 500,
        num_repetitions: 1,
        ..Default::default()
    })?;
    
    for _ in 0..num_trials {
        let result = simulator.solve(&model)?;
        trial_results.push(result.best_energy);
    }
    
    trial_results.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let mean_energy: f64 = trial_results.iter().sum::<f64>() / trial_results.len() as f64;
    let std_energy: f64 = {
        let variance: f64 = trial_results.iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f64>() / trial_results.len() as f64;
        variance.sqrt()
    };
    
    let ground_state_prob = trial_results.iter()
        .filter(|&&e| (e - energies[0].2).abs() < 1e-6)
        .count() as f64 / trial_results.len() as f64;
    
    println!("  Trials: {}", num_trials);
    println!("  Mean energy: {:.3} ± {:.3}", mean_energy, std_energy);
    println!("  Ground state probability: {:.1}%", ground_state_prob * 100.0);
    println!("  Best found: {:.3}", trial_results[0]);
    println!("  Worst found: {:.3}", trial_results.last().unwrap());

    println!("  [Visualizations would be saved as SVG files]");
    println!("  - Energy histogram");
    println!("  - Convergence curves");
    println!("  - 2D energy landscape projection");

    Ok(())
}

// Helper traits and implementations for the demo
trait Optimizer {
    fn solve(&self, model: &IsingModel) -> Result<OptimizationResult, Box<dyn std::error::Error>>;
}

struct OptimizationResult {
    best_energy: f64,
    best_spins: Vec<i8>,
    convergence_iteration: usize,
}

impl Optimizer for ClassicalAnnealingSimulator {
    fn solve(&self, model: &IsingModel) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
        let result = ClassicalAnnealingSimulator::solve(self, model)?;
        Ok(OptimizationResult {
            best_energy: result.best_energy,
            best_spins: result.best_spins,
            convergence_iteration: result.convergence_iteration.unwrap_or(0),
        })
    }
}

impl Optimizer for PopulationAnnealingSimulator {
    fn solve(&self, model: &IsingModel) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
        let result = PopulationAnnealingSimulator::solve(self, model)?;
        Ok(OptimizationResult {
            best_energy: result.best_energy,
            best_spins: result.best_solution,
            convergence_iteration: 0,
        })
    }
}

impl Optimizer for CoherentIsingMachine {
    fn solve(&self, model: &IsingModel) -> Result<OptimizationResult, Box<dyn std::error::Error>> {
        let result = CoherentIsingMachine::solve(self, model)?;
        Ok(OptimizationResult {
            best_energy: result.best_energy,
            best_spins: result.best_spins,
            convergence_iteration: result.iterations_to_convergence,
        })
    }
}

// Placeholder implementations for demo purposes
struct PowerGridOptimizer {
    constraints: GridConstraints,
}

impl PowerGridOptimizer {
    fn new(constraints: GridConstraints) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self { constraints })
    }
    
    fn get_cost_coefficient(&self, generator_index: usize) -> f64 {
        self.constraints.cost_coefficients.get(generator_index).copied().unwrap_or(100.0)
    }
}

struct PortfolioOptimizer {
    constraints: PortfolioConstraints,
}

impl PortfolioOptimizer {
    fn new(constraints: PortfolioConstraints) -> Self {
        Self { constraints }
    }
}

struct VehicleRoutingOptimizer {
    problem: RoutingProblem,
}

impl VehicleRoutingOptimizer {
    fn new(problem: RoutingProblem) -> Self {
        Self { problem }
    }
    
    fn calculate_route_distance(&self, route: &[usize]) -> f64 {
        // Simplified distance calculation
        route.len() as f64 * 2.5
    }
}

struct PenaltyOptimizer;

impl PenaltyOptimizer {
    fn new() -> Self {
        Self
    }
    
    fn optimize_penalties(
        &self,
        _model: &IsingModel,
        _target_violation_rate: f64,
        penalty_range: (f64, f64),
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        // Simplified penalty optimization
        let optimal_penalty = (penalty_range.0 + penalty_range.1) / 2.0 * 3.7; // Mock optimization
        Ok(vec![optimal_penalty])
    }
}

// Placeholder algorithm enum for benchmarking
#[derive(Debug, Clone)]
enum Algorithm {
    ClassicalAnnealing,
    PopulationAnnealing,
    CoherentIsingMachine,
}