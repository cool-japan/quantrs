# QuantRS2-Tytan Integration Guide

This guide demonstrates how to integrate all the components of QuantRS2-Tytan to solve complex optimization problems efficiently.

## Table of Contents

1. [Overview](#overview)
2. [Basic Workflow](#basic-workflow)
3. [Advanced Integration Patterns](#advanced-integration-patterns)
4. [Component Integration](#component-integration)
5. [Performance Optimization](#performance-optimization)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Overview

QuantRS2-Tytan provides a comprehensive framework for quantum optimization with:
- Problem definition using DSL
- Advanced algorithms (CIM, QAOA variants, VQF, etc.)
- Problem decomposition strategies
- Industry-specific applications
- Development and debugging tools
- Performance optimization features

## Basic Workflow

### 1. Define Your Problem

You can define problems in three ways:

#### Using the DSL
```rust
use quantrs2_tytan::problem_dsl::*;

let mut dsl = ProblemDSL::new();
let problem = dsl.parse(r#"
    param n = 5;
    var x[n] binary;
    
    minimize sum(i in 0..n: cost[i] * x[i]);
    
    subject to
        sum(i in 0..n: weight[i] * x[i]) <= capacity;
        sum(i in 0..n: x[i]) >= 1;
"#)?;

let (qubo, var_map) = dsl.compile_to_qubo(&problem)?;
```

#### Using Symbolic Math (with `dwave` feature)
```rust
use quantrs2_tytan::{symbols, Compile};

let x = symbols("x");
let y = symbols("y");
let h = (x + y - 1).pow(2);

let (qubo, offset) = Compile::new(&h).get_qubo()?;
```

#### Direct QUBO Construction
```rust
use ndarray::Array2;
use std::collections::HashMap;

let mut qubo = Array2::zeros((3, 3));
qubo[[0, 1]] = -2.0;
qubo[[1, 2]] = -1.0;

let mut var_map = HashMap::new();
var_map.insert("x0".to_string(), 0);
var_map.insert("x1".to_string(), 1);
var_map.insert("x2".to_string(), 2);
```

### 2. Choose and Configure Solver

Select from various solvers based on your problem:

```rust
use quantrs2_tytan::sampler::*;
use quantrs2_tytan::coherent_ising_machine::*;

// Simulated Annealing
let sa_sampler = SASampler::new(Some(42))
    .with_beta_range(0.1, 10.0)
    .with_sweeps(1000);

// Coherent Ising Machine
let cim = CIMSimulator::new(qubo.shape()[0])
    .with_pump_parameter(2.0)
    .with_evolution_time(20.0)
    .with_noise_strength(0.1);

// Advanced CIM with error correction
let advanced_cim = AdvancedCIM::new(qubo.shape()[0])
    .with_pulse_shape(PulseShape::Gaussian { width: 1.0, amplitude: 2.0 })
    .with_error_correction(ErrorCorrectionScheme::MajorityVoting { window_size: 3 })
    .with_bifurcation_control(BifurcationControl {
        initial_param: 0.0,
        final_param: 2.5,
        ramp_time: 15.0,
        ramp_type: RampType::Sigmoid,
    });
```

### 3. Apply Problem Decomposition (for large problems)

```rust
use quantrs2_tytan::problem_decomposition::*;

// Graph partitioning
let partitioner = GraphPartitioner::new()
    .with_num_partitions(4)
    .with_algorithm(PartitioningAlgorithm::Metis)
    .with_balance_factor(1.1);

let partitions = partitioner.partition(&qubo)?;

// Hierarchical decomposition
let hierarchical = HierarchicalSolver::new()
    .with_coarsening_strategy(CoarseningStrategy::HeavyEdgeMatching)
    .with_refinement_method(RefinementMethod::LocalSearch)
    .with_max_levels(3);

let hierarchy = hierarchical.create_hierarchy(&qubo)?;

// Domain decomposition with ADMM
let domain_decomp = DomainDecomposer::new()
    .with_method(DecompositionMethod::ADMM)
    .with_num_domains(4)
    .with_overlap(2)
    .with_admm_params(ADMMParams {
        rho: 1.0,
        max_iter: 100,
        tolerance: 1e-4,
    });

let domains = domain_decomp.decompose(&qubo)?;
```

### 4. Profile and Optimize

```rust
use quantrs2_tytan::performance_profiler::*;
use quantrs2_tytan::performance_optimization::*;

// Set up profiler
let mut profiler = PerformanceProfiler::new(ProfilerConfig {
    enabled: true,
    profile_memory: true,
    profile_cpu: true,
    profile_gpu: cfg!(feature = "gpu"),
    ..Default::default()
});

// Profile solving process
profiler.start_profile("optimization_run")?;

// Use optimized evaluator for large QUBOs
let opt_evaluator = OptimizedQUBOEvaluator::new(qubo.clone());

// Run optimization with profiling
profile!(profiler, "solve_qubo");
let results = sampler.run_qubo(&(qubo, var_map), 1000)?;

let profile = profiler.stop_profile()?;
let analysis = profiler.analyze_profile(&profile);

println!("Total time: {:.2}ms", analysis.total_time);
println!("Hottest function: {:?}", analysis.hot_paths.first());
```

### 5. Debug and Analyze Solutions

```rust
use quantrs2_tytan::solution_debugger::*;

// Create problem info
let problem_info = ProblemInfo {
    name: "My Optimization Problem".to_string(),
    problem_type: "QUBO".to_string(),
    num_variables: var_map.len(),
    var_map: var_map.clone(),
    qubo: qubo.clone(),
    constraints: vec![/* your constraints */],
    ..Default::default()
};

// Create debugger
let mut debugger = SolutionDebugger::new(problem_info, DebuggerConfig {
    detailed_analysis: true,
    check_constraints: true,
    analyze_energy: true,
    ..Default::default()
});

// Debug best solution
if let Some(best) = results.first() {
    let solution = Solution {
        assignments: best.assignments.clone(),
        objective_value: best.energy,
        timestamp: Some(std::time::SystemTime::now()),
        solver: Some("CIM".to_string()),
    };
    
    let report = debugger.debug_solution(&solution);
    println!("Solution quality: {:?}", report.summary.solution_quality);
    
    // Interactive debugging
    let mut interactive = InteractiveDebugger::new(problem_info);
    interactive.load_solution(solution);
    
    // Add watches and breakpoints
    interactive.add_watch("critical_var".to_string());
    interactive.add_breakpoint(Breakpoint::EnergyThreshold { threshold: -100.0 });
}
```

## Advanced Integration Patterns

### Pattern 1: Multi-Algorithm Ensemble

Run multiple algorithms and combine results:

```rust
use rayon::prelude::*;

fn ensemble_solve(
    qubo: &(Array2<f64>, HashMap<String, usize>),
    shots_per_algorithm: usize,
) -> Result<Vec<SampleResult>, Box<dyn std::error::Error>> {
    let algorithms: Vec<(&str, Box<dyn Sampler>)> = vec![
        ("SA", Box::new(SASampler::new(Some(42)))),
        ("GA", Box::new(GASampler::new(Some(43)))),
        ("CIM", Box::new(CIMSimulator::new(qubo.0.shape()[0]))),
    ];
    
    // Run algorithms in parallel
    let all_results: Vec<_> = algorithms
        .into_par_iter()
        .map(|(name, sampler)| {
            println!("Running {}", name);
            sampler.run_qubo(qubo, shots_per_algorithm)
                .map(|results| (name, results))
        })
        .collect::<Result<Vec<_>, _>>()?;
    
    // Merge and sort results
    let mut merged = Vec::new();
    for (_, results) in all_results {
        merged.extend(results);
    }
    
    merged.sort_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap());
    merged.dedup_by(|a, b| a.assignments == b.assignments);
    
    Ok(merged)
}
```

### Pattern 2: Adaptive Algorithm Selection

Choose algorithm based on problem characteristics:

```rust
fn select_algorithm(qubo: &Array2<f64>) -> Box<dyn Sampler> {
    let n = qubo.shape()[0];
    let density = qubo.iter().filter(|&&x| x.abs() > 1e-6).count() as f64 
        / (n * n) as f64;
    
    if n < 50 {
        // Small problems: use exact or near-exact methods
        Box::new(SASampler::new(None).with_sweeps(10000))
    } else if density < 0.1 {
        // Sparse problems: use graph-based methods
        Box::new(CIMSimulator::new(n).with_coupling_scale(2.0))
    } else if n < 200 {
        // Medium dense problems: use genetic algorithm
        Box::new(GASampler::new(None).with_population_size(100))
    } else {
        // Large problems: use decomposition + parallel solving
        // Would implement decomposition-based solver here
        Box::new(SASampler::new(None))
    }
}
```

### Pattern 3: Progressive Refinement

Start with coarse solution and progressively refine:

```rust
use quantrs2_tytan::testing_framework::*;

fn progressive_solve(
    qubo: &Array2<f64>,
    var_map: &HashMap<String, usize>,
) -> Result<Solution, Box<dyn std::error::Error>> {
    let n = qubo.shape()[0];
    
    // Phase 1: Quick exploration with high temperature SA
    let explorer = SASampler::new(Some(42))
        .with_beta_range(0.01, 0.1)
        .with_sweeps(100);
    
    let initial_results = explorer.run_qubo(&(qubo.clone(), var_map.clone()), 100)?;
    let best_initial = initial_results.first().unwrap();
    
    // Phase 2: Local refinement around best solution
    let mut current = best_initial.assignments.clone();
    let refiner = OptimizedSA::new(qubo.clone())
        .with_schedule(AnnealingSchedule::Geometric { t0: 0.1, alpha: 0.99 });
    
    // Convert to binary array
    let mut binary_sol = Array1::zeros(n);
    for (var, &idx) in var_map {
        binary_sol[idx] = if current[var] { 1 } else { 0 };
    }
    
    let (refined_binary, energy) = refiner.anneal(binary_sol, 1000, &mut rand::thread_rng());
    
    // Convert back to assignments
    let mut refined_assignments = HashMap::new();
    for (var, &idx) in var_map {
        refined_assignments.insert(var.clone(), refined_binary[idx] == 1);
    }
    
    Ok(Solution {
        assignments: refined_assignments,
        objective_value: energy,
        timestamp: Some(std::time::SystemTime::now()),
        solver: Some("Progressive".to_string()),
    })
}
```

### Pattern 4: Constraint-Guided Search

Use constraint information to guide the search:

```rust
fn constraint_guided_solve(
    problem_info: &ProblemInfo,
) -> Result<Solution, Box<dyn std::error::Error>> {
    // Analyze constraints to identify critical variables
    let mut critical_vars = HashSet::new();
    for constraint in &problem_info.constraints {
        if constraint.is_hard {
            critical_vars.extend(constraint.variables.clone());
        }
    }
    
    // Create custom sampler that respects constraints
    let sampler = SASampler::new(Some(42));
    
    // Add penalty terms for constraint violations
    let mut penalized_qubo = problem_info.qubo.clone();
    for constraint in &problem_info.constraints {
        // Add quadratic penalties for constraint violations
        match &constraint.constraint_type {
            ConstraintType::OneHot => {
                // Add penalties for one-hot constraints
                for i in 0..constraint.variables.len() {
                    for j in i+1..constraint.variables.len() {
                        let idx_i = problem_info.var_map[&constraint.variables[i]];
                        let idx_j = problem_info.var_map[&constraint.variables[j]];
                        penalized_qubo[[idx_i, idx_j]] += constraint.penalty;
                        penalized_qubo[[idx_j, idx_i]] += constraint.penalty;
                    }
                }
            }
            _ => {}
        }
    }
    
    // Solve with constraint-aware QUBO
    let results = sampler.run_qubo(
        &(penalized_qubo, problem_info.var_map.clone()),
        1000
    )?;
    
    // Verify constraints and select best valid solution
    for result in results {
        let solution = Solution {
            assignments: result.assignments,
            objective_value: result.energy,
            timestamp: Some(std::time::SystemTime::now()),
            solver: Some("Constraint-Guided".to_string()),
        };
        
        // Check if solution satisfies all hard constraints
        let violations = check_constraints(&solution, &problem_info.constraints);
        if violations.is_empty() {
            return Ok(solution);
        }
    }
    
    Err("No valid solution found".into())
}

fn check_constraints(
    solution: &Solution,
    constraints: &[ConstraintInfo],
) -> Vec<String> {
    let mut violations = Vec::new();
    
    for constraint in constraints {
        if constraint.is_hard {
            // Check constraint satisfaction
            match &constraint.constraint_type {
                ConstraintType::OneHot => {
                    let active_count = constraint.variables.iter()
                        .filter(|v| solution.assignments.get(*v).copied().unwrap_or(false))
                        .count();
                    
                    if active_count != 1 {
                        violations.push(format!(
                            "OneHot constraint '{}' violated: {} variables active",
                            constraint.name, active_count
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    
    violations
}
```

## Component Integration

### Integrating with GPU Acceleration

```rust
#[cfg(feature = "gpu")]
use quantrs2_tytan::gpu_samplers::*;

fn gpu_accelerated_solve(
    qubo: &Array2<f64>,
    var_map: &HashMap<String, usize>,
) -> Result<Vec<SampleResult>, Box<dyn std::error::Error>> {
    if !is_gpu_available() {
        return Err("GPU not available".into());
    }
    
    let gpu_sampler = GPUSampler::new()?
        .with_device(0)
        .with_workgroup_size(256)
        .with_algorithm(GPUAlgorithm::ParallelTempering);
    
    // Use memory pool for efficient GPU memory management
    let pool = GPUMemoryPool::new(1 << 20)?; // 1MB pool
    
    let results = gpu_sampler
        .with_memory_pool(pool)
        .run_qubo(&(qubo.clone(), var_map.clone()), 10000)?;
    
    Ok(results)
}
```

### Integrating Industry Applications

```rust
use quantrs2_tytan::applications::finance::*;
use quantrs2_tytan::applications::logistics::*;

fn integrated_portfolio_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // Define portfolio problem
    let returns = array![0.05, 0.10, 0.15, 0.03, 0.08];
    let covariance = array![
        [0.01, 0.002, 0.001, 0.0, 0.002],
        [0.002, 0.02, 0.003, 0.001, 0.001],
        [0.001, 0.003, 0.03, 0.002, 0.002],
        [0.0, 0.001, 0.002, 0.005, 0.001],
        [0.002, 0.001, 0.002, 0.001, 0.015]
    ];
    
    // Create optimizer with constraints
    let optimizer = PortfolioOptimizer::new(returns, covariance)
        .with_risk_aversion(2.0)
        .with_constraints(PortfolioConstraints {
            min_investment: 0.05,
            max_investment: 0.40,
            target_return: Some(0.08),
            max_assets: Some(3),
            sector_limits: {
                let mut limits = HashMap::new();
                limits.insert("tech".to_string(), 0.5);
                limits.insert("finance".to_string(), 0.3);
                limits
            },
        })
        .with_method(OptimizationMethod::Markowitz);
    
    // Convert to QUBO
    let (qubo, mapping) = optimizer.to_qubo()?;
    
    // Use problem decomposition for large portfolios
    if mapping.len() > 100 {
        let decomposer = DomainDecomposer::new()
            .with_method(DecompositionMethod::VariablePartitioning);
        
        let domains = decomposer.decompose(&qubo)?;
        
        // Solve each domain
        let coordinator = ParallelCoordinator::new()
            .with_coordination_method(CoordinationMethod::ConsensusADMM);
        
        let solution = coordinator.solve_domains(domains)?;
        
        // Decode solution
        let portfolio = optimizer.decode_solution(&solution)?;
        println!("Optimal portfolio: {:?}", portfolio);
    } else {
        // Direct solving for smaller problems
        let sampler = select_algorithm(&qubo);
        let results = sampler.run_qubo(&(qubo, mapping), 1000)?;
        
        if let Some(best) = results.first() {
            let portfolio = optimizer.decode_solution(&best.assignments)?;
            println!("Optimal portfolio: {:?}", portfolio);
        }
    }
    
    Ok(())
}
```

### Complete Integration Example

Here's a complete example integrating all components:

```rust
use quantrs2_tytan::*;
use std::time::Instant;

fn complete_optimization_pipeline(
    problem_description: &str,
) -> Result<OptimizationReport, Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    
    // 1. Parse problem with DSL
    let mut dsl = ProblemDSL::new();
    
    // Add optimization hints
    dsl.add_hint(OptimizationHint::SolverPreference("CIM".to_string()));
    dsl.add_hint(OptimizationHint::Decomposition(DecompositionHint {
        method: "hierarchical".to_string(),
        parameters: {
            let mut params = HashMap::new();
            params.insert("max_levels".to_string(), Value::Number(3.0));
            params
        },
    }));
    
    let ast = dsl.parse(problem_description)?;
    let (qubo, var_map) = dsl.compile_to_qubo(&ast)?;
    
    // 2. Set up profiling and debugging
    let mut profiler = PerformanceProfiler::new(ProfilerConfig {
        enabled: true,
        profile_memory: true,
        profile_cpu: true,
        ..Default::default()
    });
    
    profiler.start_profile("full_pipeline")?;
    
    // 3. Problem analysis and decomposition
    let problem_info = analyze_problem(&qubo, &var_map);
    
    let solving_strategy = if problem_info.size > 200 {
        SolvingStrategy::Decomposition
    } else if problem_info.density < 0.1 {
        SolvingStrategy::Sparse
    } else {
        SolvingStrategy::Direct
    };
    
    // 4. Solve based on strategy
    let solution = match solving_strategy {
        SolvingStrategy::Decomposition => {
            profile!(profiler, "decomposition");
            
            let decomposer = HierarchicalSolver::new()
                .with_coarsening_strategy(CoarseningStrategy::HeavyEdgeMatching)
                .with_refinement_method(RefinementMethod::V_Cycle);
            
            let hierarchy = decomposer.create_hierarchy(&qubo)?;
            let coarse_solution = solve_coarse_problem(&hierarchy)?;
            decomposer.refine_solution(coarse_solution, &hierarchy)?
        }
        
        SolvingStrategy::Sparse => {
            profile!(profiler, "sparse_solve");
            
            let cim = AdvancedCIM::new(qubo.shape()[0])
                .with_pulse_shape(PulseShape::Sech { width: 1.5, amplitude: 2.0 })
                .with_bifurcation_control(BifurcationControl {
                    initial_param: 0.0,
                    final_param: 2.0,
                    ramp_time: 20.0,
                    ramp_type: RampType::Adaptive,
                });
            
            let results = cim.run_qubo(&(qubo.clone(), var_map.clone()), 100)?;
            results.into_iter().next().unwrap()
        }
        
        SolvingStrategy::Direct => {
            profile!(profiler, "direct_solve");
            
            let results = ensemble_solve(&(qubo.clone(), var_map.clone()), 100)?;
            results.into_iter().next().unwrap()
        }
    };
    
    // 5. Solution debugging and validation
    profile!(profiler, "validation");
    
    let mut debugger = SolutionDebugger::new(
        create_problem_info(&qubo, &var_map),
        DebuggerConfig {
            detailed_analysis: true,
            check_constraints: true,
            analyze_energy: true,
            ..Default::default()
        }
    );
    
    let debug_solution = Solution {
        assignments: solution.assignments.clone(),
        objective_value: solution.energy,
        timestamp: Some(std::time::SystemTime::now()),
        solver: Some(format!("{:?}", solving_strategy)),
    };
    
    let debug_report = debugger.debug_solution(&debug_solution);
    
    // 6. Generate final report
    let profile = profiler.stop_profile()?;
    let perf_analysis = profiler.analyze_profile(&profile);
    
    let total_time = start_time.elapsed();
    
    Ok(OptimizationReport {
        problem_size: qubo.shape()[0],
        solving_strategy,
        solution: solution.assignments,
        objective_value: solution.energy,
        solution_quality: debug_report.summary.solution_quality,
        constraint_violations: debug_report.issues.len(),
        total_time: total_time.as_millis() as f64,
        performance_breakdown: perf_analysis,
        suggestions: debug_report.suggestions,
    })
}

#[derive(Debug)]
struct OptimizationReport {
    problem_size: usize,
    solving_strategy: SolvingStrategy,
    solution: HashMap<String, bool>,
    objective_value: f64,
    solution_quality: SolutionQuality,
    constraint_violations: usize,
    total_time: f64,
    performance_breakdown: PerformanceAnalysis,
    suggestions: Vec<Suggestion>,
}

#[derive(Debug)]
enum SolvingStrategy {
    Direct,
    Sparse,
    Decomposition,
}

fn analyze_problem(qubo: &Array2<f64>, var_map: &HashMap<String, usize>) -> ProblemAnalysis {
    let n = qubo.shape()[0];
    let non_zeros = qubo.iter().filter(|&&x| x.abs() > 1e-6).count();
    let density = non_zeros as f64 / (n * n) as f64;
    
    ProblemAnalysis {
        size: n,
        density,
        structure: identify_structure(qubo),
    }
}

#[derive(Debug)]
struct ProblemAnalysis {
    size: usize,
    density: f64,
    structure: ProblemStructure,
}

#[derive(Debug)]
enum ProblemStructure {
    Random,
    Sparse,
    Structured,
}

fn identify_structure(qubo: &Array2<f64>) -> ProblemStructure {
    // Simple heuristic - would be more sophisticated in practice
    let n = qubo.shape()[0];
    let diagonal_dominant = (0..n)
        .map(|i| qubo[[i, i]].abs())
        .sum::<f64>() > qubo.sum().abs() * 0.5;
    
    if diagonal_dominant {
        ProblemStructure::Structured
    } else {
        ProblemStructure::Random
    }
}
```

## Performance Optimization

### Memory-Efficient Large Problem Handling

```rust
use quantrs2_tytan::performance_optimization::*;

fn handle_large_problem(
    qubo: &Array2<f64>,
    block_size: usize,
) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
    let n = qubo.shape()[0];
    
    // Use memory pool for efficient allocation
    let mut pool: MemoryPool<f64> = MemoryPool::new(block_size * block_size, 10);
    
    // Process in blocks
    let mut solution = vec![false; n];
    
    for block_start in (0..n).step_by(block_size) {
        let block_end = (block_start + block_size).min(n);
        let block_size_actual = block_end - block_start;
        
        // Extract block
        let mut block_qubo = if let Some(buffer) = pool.get() {
            // Reuse buffer
            let mut block = Array2::from_shape_vec(
                (block_size_actual, block_size_actual),
                buffer[..block_size_actual * block_size_actual].to_vec()
            )?;
            
            for i in 0..block_size_actual {
                for j in 0..block_size_actual {
                    block[[i, j]] = qubo[[block_start + i, block_start + j]];
                }
            }
            
            block
        } else {
            // Allocate new
            qubo.slice(s![block_start..block_end, block_start..block_end])
                .to_owned()
        };
        
        // Solve block
        let block_solution = solve_block(&block_qubo)?;
        
        // Update solution
        for (i, &val) in block_solution.iter().enumerate() {
            solution[block_start + i] = val;
        }
    }
    
    Ok(solution)
}

fn solve_block(block: &Array2<f64>) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
    // Use optimized solver for block
    let opt_sa = OptimizedSA::new(block.clone())
        .with_schedule(AnnealingSchedule::Adaptive { 
            t0: 1.0, 
            target_rate: 0.3 
        });
    
    let initial = Array1::zeros(block.shape()[0]);
    let (solution, _) = opt_sa.anneal(initial, 1000, &mut rand::thread_rng());
    
    Ok(solution.iter().map(|&x| x == 1).collect())
}
```

### Parallel Pipeline Processing

```rust
use std::sync::mpsc;
use std::thread;

fn parallel_pipeline_solve(
    problems: Vec<(Array2<f64>, HashMap<String, usize>)>,
) -> Vec<SampleResult> {
    let num_workers = num_cpus::get();
    let (problem_tx, problem_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    
    // Spawn worker threads
    let workers: Vec<_> = (0..num_workers)
        .map(|id| {
            let problem_rx = problem_rx.clone();
            let result_tx = result_tx.clone();
            
            thread::spawn(move || {
                while let Ok((qubo, var_map)) = problem_rx.recv() {
                    // Select algorithm based on problem
                    let sampler = select_algorithm(&qubo);
                    
                    // Solve
                    if let Ok(results) = sampler.run_qubo(&(qubo, var_map), 100) {
                        if let Some(best) = results.into_iter().next() {
                            result_tx.send(best).ok();
                        }
                    }
                }
            })
        })
        .collect();
    
    // Send problems to workers
    thread::spawn(move || {
        for problem in problems {
            problem_tx.send(problem).ok();
        }
    });
    
    // Collect results
    let mut all_results = Vec::new();
    while let Ok(result) = result_rx.recv() {
        all_results.push(result);
    }
    
    all_results
}
```

## Best Practices

### 1. Problem Formulation
- Use the DSL for complex constraint problems
- Apply symmetry breaking to reduce search space
- Choose appropriate penalty weights for constraints
- Consider problem-specific decomposition

### 2. Algorithm Selection
- Small problems (< 50 variables): Use exact or high-quality heuristics
- Sparse problems: Use graph-based methods like CIM
- Dense problems: Use population-based methods like GA
- Large problems: Apply decomposition first

### 3. Performance Optimization
- Profile before optimizing
- Use parallel algorithms for problems > 100 variables
- Enable GPU acceleration for problems > 1000 variables
- Reuse memory allocations with pools

### 4. Solution Quality
- Always validate solutions against constraints
- Use ensemble methods for critical applications
- Apply progressive refinement for large problems
- Debug unexpected results with the solution debugger

### 5. Development Workflow
- Start with small test cases
- Use the testing framework for regression testing
- Profile performance regularly
- Document problem-specific optimizations

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Use problem decomposition
   - Enable memory-efficient modes
   - Process in blocks

2. **Poor Solution Quality**
   - Check constraint penalties
   - Try different algorithms
   - Increase number of shots
   - Use solution debugging

3. **Slow Performance**
   - Profile to identify bottlenecks
   - Enable parallelization
   - Use GPU acceleration
   - Apply problem decomposition

4. **Constraint Violations**
   - Increase penalty weights
   - Use hard constraint encoding
   - Apply constraint-guided search
   - Debug with interactive debugger

### Debugging Tips

```rust
// Enable detailed logging
std::env::set_var("RUST_LOG", "quantrs2_tytan=debug");
env_logger::init();

// Use interactive debugger
let mut debugger = InteractiveDebugger::new(problem_info);
debugger.start_recording();

// Common debugging commands
debugger.execute_command("analyze");
debugger.execute_command("constraints");
debugger.execute_command("watch critical_var");
debugger.execute_command("break energy -50.0");
debugger.execute_command("sensitivity");
```

## Conclusion

QuantRS2-Tytan provides a comprehensive framework for quantum optimization. By integrating its components effectively, you can:

- Define complex problems intuitively
- Apply state-of-the-art algorithms
- Handle large-scale problems through decomposition
- Optimize performance with profiling and GPU acceleration
- Debug and validate solutions systematically

The key to success is choosing the right combination of components for your specific problem and iterating based on profiling and debugging insights.