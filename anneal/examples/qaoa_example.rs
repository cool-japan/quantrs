//! Example demonstrating Quantum Approximate Optimization Algorithm (QAOA)
//!
//! This example shows how to:
//! 1. Configure different QAOA variants (standard, QAOA+, warm-start, recursive)
//! 2. Use various mixer strategies (X-mixer, XY-mixer, custom mixers)
//! 3. Apply different classical optimization algorithms
//! 4. Solve optimization problems with different problem encodings
//! 5. Analyze circuit performance and quantum state properties
//! 6. Compare QAOA performance across configurations
//! 7. Demonstrate parameter sensitivity and optimization landscapes

use quantrs2_anneal::{
    ising::IsingModel,
    qaoa::{
        QaoaOptimizer, QaoaConfig, QaoaVariant, MixerType, ProblemEncoding,
        QaoaClassicalOptimizer, ParameterInitialization,
        create_standard_qaoa_config, create_qaoa_plus_config, create_warm_start_qaoa_config,
        create_constrained_qaoa_config,
    },
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantum Approximate Optimization Algorithm (QAOA) Demo ===\n");

    // Example 1: Standard QAOA
    println!("Example 1: Standard QAOA for MaxCut Problem");
    standard_qaoa_example()?;
    
    // Example 2: QAOA+ with multi-angle mixers
    println!("\nExample 2: QAOA+ with Multi-Angle Mixers");
    qaoa_plus_example()?;
    
    // Example 3: Warm-start QAOA
    println!("\nExample 3: Warm-Start QAOA with Classical Initialization");
    warm_start_qaoa_example()?;
    
    // Example 4: Constrained optimization with XY mixer
    println!("\nExample 4: Constrained Optimization with XY Mixer");
    constrained_qaoa_example()?;
    
    // Example 5: Classical optimizer comparison
    println!("\nExample 5: Classical Optimizer Comparison");
    optimizer_comparison_example()?;
    
    // Example 6: Parameter depth study
    println!("\nExample 6: QAOA Parameter Depth Study");
    parameter_depth_study_example()?;
    
    // Example 7: Large-scale optimization
    println!("\nExample 7: Large-Scale Optimization Problem");
    large_scale_qaoa_example()?;

    Ok(())
}

fn standard_qaoa_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a MaxCut problem on a triangle graph
    let mut problem = IsingModel::new(3);
    problem.set_coupling(0, 1, -1.0)?;  // Negative couplings for MaxCut
    problem.set_coupling(1, 2, -1.0)?;
    problem.set_coupling(2, 0, -1.0)?;
    
    // Configure standard QAOA with 2 layers
    let config = create_standard_qaoa_config(2, 1000);
    
    let start = Instant::now();
    let mut qaoa = QaoaOptimizer::new(config)?;
    let results = qaoa.solve(&problem)?;
    let runtime = start.elapsed();
    
    println!("Standard QAOA Results:");
    println!("  Problem: MaxCut on triangle graph (3 vertices)");
    println!("  QAOA layers: 2");
    println!("  Shots: 1000");
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Approximation ratio: {:.3}", results.approximation_ratio);
    println!("  Converged: {}", results.converged);
    println!("  Function evaluations: {}", results.function_evaluations);
    println!("  Runtime: {:.2?}", runtime);
    
    // Parameter analysis
    println!("\n  Optimal QAOA parameters:");
    for (i, &param) in results.optimal_parameters.iter().enumerate() {
        let param_type = if i % 2 == 0 { "gamma" } else { "beta" };
        let layer = i / 2;
        println!("    Layer {} {}: {:.6}", layer, param_type, param);
    }
    
    // Circuit analysis
    println!("\n  Circuit statistics:");
    println!("    Total depth: {}", results.circuit_stats.total_depth);
    println!("    Single-qubit gates: {}", results.circuit_stats.single_qubit_gates);
    println!("    Two-qubit gates: {}", results.circuit_stats.two_qubit_gates);
    println!("    Estimated fidelity: {:.3}", results.circuit_stats.estimated_fidelity);
    
    // Quantum state analysis
    println!("\n  Quantum state statistics:");
    println!("    Optimal overlap: {:.3}", results.quantum_stats.optimal_overlap);
    println!("    Concentration ratio: {:.3}", results.quantum_stats.concentration_ratio);
    println!("    Expectation variance: {:.6}", results.quantum_stats.expectation_variance);
    
    Ok(())
}

fn qaoa_plus_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a frustrated system
    let mut problem = IsingModel::new(4);
    
    // Create a frustrated square with competing interactions
    problem.set_coupling(0, 1, 1.0)?;   // Ferromagnetic
    problem.set_coupling(1, 2, 1.0)?;
    problem.set_coupling(2, 3, 1.0)?;
    problem.set_coupling(3, 0, 1.0)?;
    problem.set_coupling(0, 2, -0.8)?;  // Antiferromagnetic diagonal
    problem.set_coupling(1, 3, -0.8)?;
    
    // Configure QAOA+ with multi-angle mixers
    let mut config = create_qaoa_plus_config(3, 1200);
    config.optimizer = QaoaClassicalOptimizer::GradientBased {
        learning_rate: 0.01,
        gradient_step: 0.01,
        max_iterations: 200,
    };
    config.parameter_init = ParameterInitialization::Linear {
        gamma_max: 0.5,
        beta_max: std::f64::consts::PI / 2.0,
    };
    
    let start = Instant::now();
    let mut qaoa = QaoaOptimizer::new(config)?;
    let results = qaoa.solve(&problem)?;
    let runtime = start.elapsed();
    
    println!("QAOA+ Results:");
    println!("  Problem: Frustrated 4-qubit square with diagonal couplings");
    println!("  QAOA variant: QAOA+ with multi-angle mixers");
    println!("  Layers: 3");
    println!("  Optimizer: Gradient-based (lr=0.01)");
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Approximation ratio: {:.3}", results.approximation_ratio);
    println!("  Converged: {}", results.converged);
    println!("  Function evaluations: {}", results.function_evaluations);
    println!("  Runtime: {:.2?}", runtime);
    
    // Parameter evolution analysis
    if results.parameter_history.len() > 1 {
        println!("\n  Parameter evolution:");
        let initial_params = &results.parameter_history[0];
        let final_params = &results.optimal_parameters;
        
        for i in 0..initial_params.len() {
            let change = (final_params[i] - initial_params[i]).abs();
            let param_type = if i % 2 == 0 { "gamma" } else { "beta" };
            println!("    {} {}: {:.6} → {:.6} (Δ = {:.6})",
                     param_type, i / 2, initial_params[i], final_params[i], change);
        }
    }
    
    // Energy convergence analysis
    if results.energy_history.len() > 10 {
        println!("\n  Energy convergence:");
        let initial_energy = results.energy_history[0];
        let final_energy = *results.energy_history.last().unwrap();
        let improvement = initial_energy - final_energy;
        
        println!("    Initial energy: {:.6}", initial_energy);
        println!("    Final energy: {:.6}", final_energy);
        println!("    Total improvement: {:.6}", improvement);
        
        // Show convergence pattern
        let quarter = results.energy_history.len() / 4;
        println!("    Energy at 25%: {:.6}", results.energy_history[quarter]);
        println!("    Energy at 50%: {:.6}", results.energy_history[2 * quarter]);
        println!("    Energy at 75%: {:.6}", results.energy_history[3 * quarter]);
    }
    
    Ok(())
}

fn warm_start_qaoa_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger random problem
    let mut problem = IsingModel::new(5);
    
    // Add random couplings and biases
    for i in 0..5 {
        problem.set_bias(i, (i as f64 - 2.0) * 0.2)?;
        for j in (i+1)..5 {
            if (i + j) % 3 == 0 {
                let coupling = if (i * j) % 2 == 0 { 0.6 } else { -0.4 };
                problem.set_coupling(i, j, coupling)?;
            }
        }
    }
    
    // Create a classical initial solution (greedy approach)
    let classical_solution = vec![1, -1, 1, -1, 1]; // Alternating pattern
    
    // Configure warm-start QAOA
    let mut config = create_warm_start_qaoa_config(2, classical_solution.clone(), 800);
    config.optimizer = QaoaClassicalOptimizer::NelderMead {
        initial_size: 0.3,
        tolerance: 1e-5,
        max_iterations: 150,
    };
    config.detailed_logging = true;
    
    let start = Instant::now();
    let mut qaoa = QaoaOptimizer::new(config)?;
    let results = qaoa.solve(&problem)?;
    let runtime = start.elapsed();
    
    println!("Warm-Start QAOA Results:");
    println!("  Problem: 5-qubit random sparse Ising model");
    println!("  Initial solution: {:?}", classical_solution);
    println!("  QAOA layers: 2");
    println!("  Optimizer: Nelder-Mead");
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Approximation ratio: {:.3}", results.approximation_ratio);
    println!("  Converged: {}", results.converged);
    println!("  Function evaluations: {}", results.function_evaluations);
    println!("  Runtime: {:.2?}", runtime);
    
    // Compare with classical solution
    let classical_energy = evaluate_ising_energy(&classical_solution, &problem)?;
    let quantum_energy = results.best_energy;
    let improvement = classical_energy - quantum_energy;
    
    println!("\n  Warm-start analysis:");
    println!("    Classical energy: {:.6}", classical_energy);
    println!("    QAOA energy: {:.6}", quantum_energy);
    println!("    Improvement: {:.6}", improvement);
    println!("    Improvement ratio: {:.1}%", 
             if classical_energy != 0.0 { improvement / classical_energy.abs() * 100.0 } else { 0.0 });
    
    // Solution similarity analysis
    let similarity = calculate_hamming_similarity(&classical_solution, &results.best_solution);
    println!("    Solution similarity: {:.1}%", similarity * 100.0);
    
    Ok(())
}

fn constrained_qaoa_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a constrained optimization problem (vertex cover)
    let mut problem = IsingModel::new(4);
    
    // Vertex cover on a square: minimize vertices while covering all edges
    // Transform to max problem: maximize -1 * (vertex penalty + edge penalty)
    
    // Vertex penalties (prefer fewer vertices)
    for i in 0..4 {
        problem.set_bias(i, 1.0)?; // Penalty for selecting vertex
    }
    
    // Edge coverage constraints (both endpoints or at least one)
    // Penalty for uncovered edges
    problem.set_coupling(0, 1, 3.0)?; // Edge (0,1) penalty
    problem.set_coupling(1, 2, 3.0)?; // Edge (1,2) penalty  
    problem.set_coupling(2, 3, 3.0)?; // Edge (2,3) penalty
    problem.set_coupling(3, 0, 3.0)?; // Edge (3,0) penalty
    
    // Configure QAOA with XY mixer for constraint preservation
    let mut config = create_constrained_qaoa_config(3, 1500);
    config.problem_encoding = ProblemEncoding::PenaltyMethod { penalty_weight: 10.0 };
    config.optimizer = QaoaClassicalOptimizer::Powell {
        tolerance: 1e-6,
        max_iterations: 100,
    };
    
    let start = Instant::now();
    let mut qaoa = QaoaOptimizer::new(config)?;
    let results = qaoa.solve(&problem)?;
    let runtime = start.elapsed();
    
    println!("Constrained QAOA Results:");
    println!("  Problem: Vertex cover on 4-vertex square graph");
    println!("  Mixer: XY mixer for constraint preservation");
    println!("  Encoding: Penalty method");
    println!("  QAOA layers: 3");
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Approximation ratio: {:.3}", results.approximation_ratio);
    println!("  Converged: {}", results.converged);
    println!("  Function evaluations: {}", results.function_evaluations);
    println!("  Runtime: {:.2?}", runtime);
    
    // Constraint satisfaction analysis
    let num_vertices = results.best_solution.iter().filter(|&&x| x == 1).count();
    let covered_edges = count_covered_edges(&results.best_solution);
    
    println!("\n  Constraint analysis:");
    println!("    Vertices selected: {}/4", num_vertices);
    println!("    Edges covered: {}/4", covered_edges);
    println!("    Constraint satisfaction: {:.1}%", covered_edges as f64 / 4.0 * 100.0);
    println!("    Solution feasibility: {}", if covered_edges == 4 { "✓" } else { "✗" });
    
    // Performance metrics
    println!("\n  Performance metrics:");
    println!("    Success probability: {:.1}%", results.performance_metrics.success_probability * 100.0);
    println!("    Relative energy: {:.3}", results.performance_metrics.relative_energy);
    println!("    Optimization efficiency: {:.6}", results.performance_metrics.optimization_efficiency);
    
    Ok(())
}

fn optimizer_comparison_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test problem
    let mut problem = IsingModel::new(4);
    problem.set_coupling(0, 1, -0.8)?;
    problem.set_coupling(1, 2, 0.6)?;
    problem.set_coupling(2, 3, -0.7)?;
    problem.set_coupling(0, 3, 0.5)?;
    problem.set_coupling(0, 2, -0.3)?;
    problem.set_coupling(1, 3, 0.4)?;
    
    let optimizers = vec![
        ("Nelder-Mead", QaoaClassicalOptimizer::NelderMead {
            initial_size: 0.5,
            tolerance: 1e-6,
            max_iterations: 200,
        }),
        ("Gradient-Based", QaoaClassicalOptimizer::GradientBased {
            learning_rate: 0.02,
            gradient_step: 0.01,
            max_iterations: 150,
        }),
        ("Powell", QaoaClassicalOptimizer::Powell {
            tolerance: 1e-6,
            max_iterations: 100,
        }),
        ("COBYLA", QaoaClassicalOptimizer::Cobyla {
            rhobeg: 0.5,
            rhoend: 1e-6,
            maxfun: 500,
        }),
    ];
    
    println!("Classical Optimizer Comparison:");
    println!("  Problem: 4-qubit mixed coupling Ising model");
    println!("  QAOA layers: 2");
    println!("  Shots: 800");
    
    for (optimizer_name, optimizer) in optimizers {
        let mut config = create_standard_qaoa_config(2, 800);
        config.optimizer = optimizer;
        config.detailed_logging = false;
        config.seed = Some(42); // Same seed for fair comparison
        
        let start = Instant::now();
        let mut qaoa = QaoaOptimizer::new(config)?;
        let results = qaoa.solve(&problem)?;
        let runtime = start.elapsed();
        
        println!("\n  {} Results:", optimizer_name);
        println!("    Best energy: {:.6}", results.best_energy);
        println!("    Approximation ratio: {:.3}", results.approximation_ratio);
        println!("    Converged: {}", results.converged);
        println!("    Function evaluations: {}", results.function_evaluations);
        println!("    Runtime: {:.2?}", runtime);
        
        // Convergence analysis
        if results.energy_history.len() > 1 {
            let initial_energy = results.energy_history[0];
            let final_energy = *results.energy_history.last().unwrap();
            let improvement = initial_energy - final_energy;
            let convergence_rate = improvement / results.function_evaluations as f64;
            
            println!("    Energy improvement: {:.6}", improvement);
            println!("    Convergence rate: {:.8}/eval", convergence_rate);
        }
        
        // Parameter statistics
        let param_variance = {
            let mean = results.optimal_parameters.iter().sum::<f64>() / results.optimal_parameters.len() as f64;
            results.optimal_parameters.iter()
                .map(|&p| (p - mean).powi(2))
                .sum::<f64>() / results.optimal_parameters.len() as f64
        };
        println!("    Parameter variance: {:.6}", param_variance);
    }
    
    println!("\n  Optimizer Selection Guide:");
    println!("    - Nelder-Mead: Good general-purpose, derivative-free");
    println!("    - Gradient-Based: Fast for smooth landscapes, requires gradients");
    println!("    - Powell: Good for low-dimensional problems");
    println!("    - COBYLA: Handles constraints well, robust");
    
    Ok(())
}

fn parameter_depth_study_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a test problem
    let mut problem = IsingModel::new(5);
    
    // Create a structured problem with multiple scales
    for i in 0..5 {
        problem.set_bias(i, (i as f64 - 2.0) * 0.15)?;
        let j = (i + 1) % 5;
        problem.set_coupling(i, j, 0.8)?; // Ring couplings
        
        if i < 3 {
            let k = i + 2;
            problem.set_coupling(i, k, -0.3)?; // Long-range interactions
        }
    }
    
    let depths = vec![1, 2, 3, 4, 5];
    
    println!("QAOA Parameter Depth Study:");
    println!("  Problem: 5-qubit ring with long-range interactions");
    println!("  Optimizer: Nelder-Mead");
    println!("  Shots: 1000");
    
    for depth in depths {
        let mut config = create_standard_qaoa_config(depth, 1000);
        config.optimizer = QaoaClassicalOptimizer::NelderMead {
            initial_size: 0.4,
            tolerance: 1e-6,
            max_iterations: 100,
        };
        config.seed = Some(123);
        
        let start = Instant::now();
        let mut qaoa = QaoaOptimizer::new(config)?;
        let results = qaoa.solve(&problem)?;
        let runtime = start.elapsed();
        
        println!("\n  Depth {} Results:", depth);
        println!("    Parameters: {}", results.optimal_parameters.len());
        println!("    Best energy: {:.6}", results.best_energy);
        println!("    Approximation ratio: {:.3}", results.approximation_ratio);
        println!("    Converged: {}", results.converged);
        println!("    Function evaluations: {}", results.function_evaluations);
        println!("    Runtime: {:.2?}", runtime);
        
        // Circuit complexity analysis
        println!("    Circuit depth: {}", results.circuit_stats.total_depth);
        println!("    Total gates: {}", 
                 results.circuit_stats.single_qubit_gates + results.circuit_stats.two_qubit_gates);
        
        // Parameter utilization
        let significant_params = results.optimal_parameters.iter()
            .filter(|&&p| p.abs() > 0.05)
            .count();
        let utilization = significant_params as f64 / results.optimal_parameters.len() as f64;
        println!("    Parameter utilization: {:.1}%", utilization * 100.0);
        
        // Energy landscape exploration
        if results.energy_history.len() > 5 {
            let energy_range = results.energy_history.iter().cloned().fold(f64::NEG_INFINITY, f64::max) -
                               results.energy_history.iter().cloned().fold(f64::INFINITY, f64::min);
            println!("    Energy range explored: {:.6}", energy_range);
        }
        
        // Optimal parameters pattern
        let gamma_params: Vec<f64> = results.optimal_parameters.iter()
            .step_by(2).cloned().collect();
        let beta_params: Vec<f64> = results.optimal_parameters.iter()
            .skip(1).step_by(2).cloned().collect();
        
        if gamma_params.len() > 1 {
            let gamma_trend = gamma_params.windows(2)
                .map(|w| w[1] - w[0])
                .collect::<Vec<f64>>();
            let avg_gamma_change = gamma_trend.iter().sum::<f64>() / gamma_trend.len() as f64;
            println!("    Gamma parameter trend: {:.6}/layer", avg_gamma_change);
        }
        
        if beta_params.len() > 1 {
            let beta_variance = {
                let mean = beta_params.iter().sum::<f64>() / beta_params.len() as f64;
                beta_params.iter()
                    .map(|&p| (p - mean).powi(2))
                    .sum::<f64>() / beta_params.len() as f64
            };
            println!("    Beta parameter variance: {:.6}", beta_variance);
        }
    }
    
    println!("\n  Depth Selection Guidelines:");
    println!("    - p=1: Good for simple problems, fast optimization");
    println!("    - p=2-3: Sweet spot for most problems");
    println!("    - p>3: Diminishing returns, harder optimization");
    println!("    - Consider circuit depth vs. approximation quality trade-off");
    
    Ok(())
}

fn large_scale_qaoa_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a larger optimization problem
    let problem_size = 8;
    let mut problem = IsingModel::new(problem_size);
    
    // Create a random 3-regular graph problem
    for i in 0..problem_size {
        // Add small random biases
        problem.set_bias(i, (i as f64 - 3.5) * 0.05)?;
        
        // Add couplings to form a roughly 3-regular graph
        for offset in 1..=3 {
            let j = (i + offset) % problem_size;
            if i != j {
                let coupling = if (i + j + offset) % 2 == 0 { -0.7 } else { 0.5 };
                problem.set_coupling(i, j, coupling)?;
            }
        }
    }
    
    // Configure QAOA for larger problem
    let mut config = create_standard_qaoa_config(3, 2000);
    config.optimizer = QaoaClassicalOptimizer::GradientBased {
        learning_rate: 0.01,
        gradient_step: 0.005,
        max_iterations: 300,
    };
    config.parameter_init = ParameterInitialization::ProblemAware;
    config.use_symmetry_reduction = true;
    config.detailed_logging = true;
    
    let start = Instant::now();
    let mut qaoa = QaoaOptimizer::new(config)?;
    let results = qaoa.solve(&problem)?;
    let runtime = start.elapsed();
    
    println!("Large-Scale QAOA Results:");
    println!("  Problem: {}-qubit 3-regular graph", problem_size);
    println!("  QAOA layers: 3");
    println!("  Shots: 2000");
    println!("  Initialization: Problem-aware");
    println!("  Best energy: {:.6}", results.best_energy);
    println!("  Best solution: {:?}", results.best_solution);
    println!("  Approximation ratio: {:.3}", results.approximation_ratio);
    println!("  Converged: {}", results.converged);
    println!("  Function evaluations: {}", results.function_evaluations);
    println!("  Runtime: {:.2?}", runtime);
    
    // Scaling analysis
    println!("\n  Scaling metrics:");
    println!("    Qubits: {}", problem_size);
    println!("    Parameters: {}", results.optimal_parameters.len());
    println!("    Circuit depth: {}", results.circuit_stats.total_depth);
    println!("    Gate count: {}", 
             results.circuit_stats.single_qubit_gates + results.circuit_stats.two_qubit_gates);
    println!("    Evaluations per qubit: {:.1}", 
             results.function_evaluations as f64 / problem_size as f64);
    println!("    Runtime per parameter: {:.2?}", 
             runtime / results.optimal_parameters.len() as u32);
    
    // Solution quality analysis
    let solution_entropy = calculate_solution_entropy(&results.best_solution);
    println!("\n  Solution analysis:");
    println!("    Solution entropy: {:.3}", solution_entropy);
    println!("    Spin up count: {}", results.best_solution.iter().filter(|&&s| s == 1).count());
    println!("    Spin down count: {}", results.best_solution.iter().filter(|&&s| s == -1).count());
    
    // Performance metrics
    println!("\n  Performance metrics:");
    println!("    Success probability: {:.1}%", results.performance_metrics.success_probability * 100.0);
    println!("    Relative energy: {:.3}", results.performance_metrics.relative_energy);
    println!("    Optimization efficiency: {:.6}", results.performance_metrics.optimization_efficiency);
    println!("    Quantum simulation time: {:.2?}", results.performance_metrics.quantum_simulation_time);
    
    // Convergence pattern analysis
    if results.energy_history.len() >= 20 {
        let early_phase = &results.energy_history[0..results.energy_history.len()/4];
        let late_phase = &results.energy_history[3*results.energy_history.len()/4..];
        
        let early_improvement = early_phase[0] - early_phase[early_phase.len()-1];
        let late_improvement = late_phase[0] - late_phase[late_phase.len()-1];
        
        println!("\n  Convergence analysis:");
        println!("    Early phase improvement: {:.6}", early_improvement);
        println!("    Late phase improvement: {:.6}", late_improvement);
        println!("    Convergence pattern: {}", 
                 if early_improvement > late_improvement * 3.0 { "Fast early" } else { "Steady" });
    }
    
    // Parameter sensitivity
    if !results.performance_metrics.parameter_sensitivity.is_empty() {
        let avg_sensitivity = results.performance_metrics.parameter_sensitivity.iter().sum::<f64>() 
                             / results.performance_metrics.parameter_sensitivity.len() as f64;
        let max_sensitivity = results.performance_metrics.parameter_sensitivity.iter()
                              .cloned().fold(f64::NEG_INFINITY, f64::max);
        
        println!("\n  Parameter sensitivity:");
        println!("    Average sensitivity: {:.6}", avg_sensitivity);
        println!("    Maximum sensitivity: {:.6}", max_sensitivity);
        println!("    Parameter robustness: {}", 
                 if max_sensitivity < 0.1 { "High" } else if max_sensitivity < 0.5 { "Medium" } else { "Low" });
    }
    
    Ok(())
}

/// Helper function to evaluate Ising energy
fn evaluate_ising_energy(spins: &[i8], problem: &IsingModel) -> Result<f64, Box<dyn std::error::Error>> {
    let mut energy = 0.0;
    
    // Bias terms
    for i in 0..spins.len() {
        energy += problem.get_bias(i)? * spins[i] as f64;
    }
    
    // Coupling terms
    for i in 0..spins.len() {
        for j in (i+1)..spins.len() {
            energy += problem.get_coupling(i, j)? * spins[i] as f64 * spins[j] as f64;
        }
    }
    
    Ok(energy)
}

/// Calculate Hamming similarity between two solutions
fn calculate_hamming_similarity(solution1: &[i8], solution2: &[i8]) -> f64 {
    if solution1.len() != solution2.len() {
        return 0.0;
    }
    
    let matches = solution1.iter()
        .zip(solution2.iter())
        .filter(|(&a, &b)| a == b)
        .count();
    
    matches as f64 / solution1.len() as f64
}

/// Count covered edges in vertex cover solution
fn count_covered_edges(solution: &[i8]) -> usize {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
    
    edges.iter()
        .filter(|&&(i, j)| solution[i] == 1 || solution[j] == 1)
        .count()
}

/// Calculate solution entropy
fn calculate_solution_entropy(solution: &[i8]) -> f64 {
    let up_count = solution.iter().filter(|&&s| s == 1).count();
    let down_count = solution.len() - up_count;
    
    if up_count == 0 || down_count == 0 {
        return 0.0;
    }
    
    let p_up = up_count as f64 / solution.len() as f64;
    let p_down = down_count as f64 / solution.len() as f64;
    
    -(p_up * p_up.ln() + p_down * p_down.ln())
}