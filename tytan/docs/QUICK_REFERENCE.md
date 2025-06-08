# QuantRS2-Tytan Quick Reference

## Problem Definition

### DSL Syntax
```rust
// Variables
var x[n] binary;
var y[m] integer in [0, 10];
var z continuous in [0.0, 1.0];

// Parameters
param cost[n];
param capacity = 100;

// Objective
minimize sum(i in 0..n: cost[i] * x[i]);
maximize product(i in 0..n: profit[i] * x[i]);

// Constraints
subject to
    sum(i in 0..n: weight[i] * x[i]) <= capacity;
    forall(i in 0..n): x[i] + y[i] >= 1;
    exists(i in 0..n): x[i] == 1;

// Hints
hint symmetry permutation(x);
hint decomposition hierarchical;
```

### Symbolic Math (with `dwave` feature)
```rust
let x = symbols("x");
let y = symbols("y");
let h = 2*x*y - x - y + 1;
let (qubo, offset) = Compile::new(&h).get_qubo()?;
```

## Samplers

### Simulated Annealing
```rust
let sampler = SASampler::new(Some(42))
    .with_beta_range(0.1, 10.0)
    .with_sweeps(1000)
    .with_num_reads(100);
```

### Genetic Algorithm
```rust
let sampler = GASampler::new(Some(42))
    .with_population_size(100)
    .with_elite_fraction(0.1)
    .with_mutation_rate(0.01);
```

### Coherent Ising Machine
```rust
let cim = CIMSimulator::new(n_vars)
    .with_pump_parameter(2.0)
    .with_evolution_time(20.0)
    .with_noise_strength(0.1);

// Advanced CIM
let adv_cim = AdvancedCIM::new(n_vars)
    .with_pulse_shape(PulseShape::Gaussian { width: 1.0, amplitude: 2.0 })
    .with_error_correction(ErrorCorrectionScheme::MajorityVoting { window_size: 3 })
    .with_bifurcation_control(BifurcationControl {
        initial_param: 0.0,
        final_param: 2.5,
        ramp_time: 15.0,
        ramp_type: RampType::Sigmoid,
    });
```

## Problem Decomposition

### Graph Partitioning
```rust
let partitioner = GraphPartitioner::new()
    .with_num_partitions(4)
    .with_algorithm(PartitioningAlgorithm::Metis)
    .with_balance_factor(1.1);
```

### Hierarchical Decomposition
```rust
let solver = HierarchicalSolver::new()
    .with_coarsening_strategy(CoarseningStrategy::HeavyEdgeMatching)
    .with_refinement_method(RefinementMethod::V_Cycle)
    .with_max_levels(3);
```

### Domain Decomposition
```rust
let decomposer = DomainDecomposer::new()
    .with_method(DecompositionMethod::ADMM)
    .with_num_domains(4)
    .with_overlap(2)
    .with_admm_params(ADMMParams {
        rho: 1.0,
        max_iter: 100,
        tolerance: 1e-4,
    });
```

## Performance Profiling

### Basic Profiling
```rust
let mut profiler = PerformanceProfiler::new(ProfilerConfig::default());

profiler.start_profile("my_optimization")?;

// Use macros
profile!(profiler, "function_name");
time_it!(profiler, "operation", {
    // code to time
});

let profile = profiler.stop_profile()?;
let analysis = profiler.analyze_profile(&profile);
```

### Optimized Evaluation
```rust
let evaluator = OptimizedQUBOEvaluator::new(qubo);
let energy = evaluator.evaluate(&x_binary);
let delta = evaluator.delta_energy(&x_binary, bit_to_flip);
```

## Solution Debugging

### Basic Debugging
```rust
let debugger = SolutionDebugger::new(problem_info, config);
let report = debugger.debug_solution(&solution);
println!("Quality: {:?}", report.summary.solution_quality);
```

### Interactive Debugging
```rust
let mut interactive = InteractiveDebugger::new(problem_info);
interactive.load_solution(solution);

// Commands
interactive.execute_command("analyze");
interactive.execute_command("constraints");
interactive.execute_command("flip x_0");
interactive.execute_command("watch x_1");
interactive.execute_command("break energy -10.0");
interactive.execute_command("sensitivity");
interactive.execute_command("export json");
```

## Applications

### Portfolio Optimization
```rust
let optimizer = PortfolioOptimizer::new(returns, covariance)
    .with_risk_aversion(2.0)
    .with_constraints(PortfolioConstraints {
        min_investment: 0.05,
        max_investment: 0.40,
        target_return: Some(0.08),
        max_assets: Some(3),
        sector_limits: HashMap::new(),
    });
```

### Vehicle Routing
```rust
let vrp = VehicleRoutingProblem::new()
    .with_depot(Location { x: 0.0, y: 0.0 })
    .with_customers(customers)
    .with_vehicles(2)
    .with_capacity(30.0);
```

### Drug Discovery
```rust
let designer = DrugDesigner::new()
    .with_target_properties(TargetProperties {
        molecular_weight: (200.0, 500.0),
        logp: (1.0, 5.0),
        hbd: (0, 5),
        hba: (0, 10),
        tpsa: (0.0, 140.0),
    })
    .with_fragments(standard_fragments());
```

## Testing Framework

### Test Configuration
```rust
let mut framework = TestingFramework::new(TestConfig {
    test_sizes: vec![10, 50, 100],
    difficulties: vec![Difficulty::Easy, Difficulty::Medium],
    timeout: 60,
    parallel: true,
});

framework.add_category(TestCategory {
    name: "Graph Problems".to_string(),
    problem_types: vec![ProblemType::MaxCut, ProblemType::GraphColoring],
    difficulties: vec![Difficulty::Easy],
    tags: vec!["graph".to_string()],
});
```

## GPU Acceleration

### Check Availability
```rust
if is_gpu_available() {
    println!("GPU acceleration available");
}
```

### GPU Sampler
```rust
#[cfg(feature = "gpu")]
let gpu_sampler = GPUSampler::new()?
    .with_device(0)
    .with_workgroup_size(256)
    .with_algorithm(GPUAlgorithm::ParallelTempering);
```

## Common Patterns

### Ensemble Solving
```rust
let algorithms: Vec<Box<dyn Sampler>> = vec![
    Box::new(SASampler::new(Some(42))),
    Box::new(GASampler::new(Some(43))),
    Box::new(CIMSimulator::new(n)),
];

let results: Vec<_> = algorithms
    .into_par_iter()
    .flat_map(|sampler| sampler.run_qubo(&qubo, 100).unwrap_or_default())
    .collect();
```

### Progressive Refinement
```rust
// Phase 1: Exploration
let explorer = SASampler::new(None).with_beta_range(0.01, 0.1);
let initial = explorer.run_qubo(&qubo, 100)?;

// Phase 2: Refinement
let refiner = OptimizedSA::new(qubo)
    .with_schedule(AnnealingSchedule::Geometric { t0: 0.1, alpha: 0.99 });
let refined = refiner.anneal(initial_binary, 1000, &mut rng);
```

### Constraint-Aware Solving
```rust
// Add penalties for constraints
let mut penalized_qubo = qubo.clone();
for constraint in &constraints {
    add_constraint_penalty(&mut penalized_qubo, constraint);
}

// Solve and verify
let results = sampler.run_qubo(&(penalized_qubo, var_map), 1000)?;
let valid_solutions: Vec<_> = results
    .into_iter()
    .filter(|sol| check_constraints(sol, &constraints).is_empty())
    .collect();
```

## Error Handling

### Common Errors
```rust
match result {
    Err(SamplerError::InvalidParameter(msg)) => {
        eprintln!("Invalid parameter: {}", msg);
    }
    Err(SamplerError::Timeout) => {
        eprintln!("Solver timeout");
    }
    Err(SamplerError::NotImplemented(feature)) => {
        eprintln!("Feature not implemented: {}", feature);
    }
    Ok(solution) => {
        // Process solution
    }
}
```

### Debug Mode
```rust
// Enable debug logging
std::env::set_var("RUST_LOG", "quantrs2_tytan=debug");
env_logger::init();

// Use debug configuration
let config = DebuggerConfig {
    detailed_analysis: true,
    check_constraints: true,
    analyze_energy: true,
    verbosity: VerbosityLevel::Debug,
    ..Default::default()
};
```

## Performance Tips

1. **Problem Size < 50**: Use exact or high-iteration SA
2. **Sparse Problems**: Use CIM or graph-based methods
3. **Dense Problems**: Use GA or parallel tempering
4. **Large Problems (> 200)**: Apply decomposition first
5. **Critical Applications**: Use ensemble methods
6. **GPU Available**: Enable for problems > 1000 variables

## Complete Example

```rust
use quantrs2_tytan::*;

fn solve_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define problem
    let mut dsl = ProblemDSL::new();
    let problem = dsl.parse(r#"
        var x[10] binary;
        minimize sum(i in 0..10: cost[i] * x[i]);
        subject to sum(i in 0..10: x[i]) >= 3;
    "#)?;
    
    // 2. Compile to QUBO
    let (qubo, var_map) = dsl.compile_to_qubo(&problem)?;
    
    // 3. Profile solving
    let mut profiler = PerformanceProfiler::new(ProfilerConfig::default());
    profiler.start_profile("solve")?;
    
    // 4. Solve
    let sampler = SASampler::new(Some(42));
    let results = sampler.run_qubo(&(qubo, var_map), 100)?;
    
    // 5. Debug solution
    if let Some(best) = results.first() {
        let solution = Solution {
            assignments: best.assignments.clone(),
            objective_value: best.energy,
            timestamp: None,
            solver: Some("SA".to_string()),
        };
        
        let debugger = SolutionDebugger::new(problem_info, config);
        let report = debugger.debug_solution(&solution);
        
        println!("Best energy: {}", best.energy);
        println!("Quality: {:?}", report.summary.solution_quality);
    }
    
    let profile = profiler.stop_profile()?;
    println!("Time: {:.2}ms", profiler.analyze_profile(&profile).total_time);
    
    Ok(())
}
```