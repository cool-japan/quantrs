# Tytan Optimization Cookbook

This cookbook provides practical recipes for solving common optimization problems using the Tytan framework. Each recipe includes problem formulation, implementation, and optimization tips.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Optimization Problems](#basic-optimization-problems)
3. [Constraint Handling](#constraint-handling)
4. [GPU Acceleration](#gpu-acceleration)
5. [Hybrid Algorithms](#hybrid-algorithms)
6. [Advanced Techniques](#advanced-techniques)
7. [Performance Tuning](#performance-tuning)
8. [Real Hardware Usage](#real-hardware-usage)

## Getting Started

### Recipe: Your First QUBO Problem

```rust
use tytan::{QuboModel, Symbol, compile};

// Define decision variables
let x = Symbol::new("x", 3);  // 3 binary variables

// Create objective function: minimize x0 + 2*x1 + 3*x2
let objective = x[0] + 2.0 * x[1] + 3.0 * x[2];

// Add constraint: x0 + x1 + x2 = 1 (exactly one selected)
let constraint = (x[0] + x[1] + x[2] - 1.0).pow(2);

// Combine with penalty weight
let model = compile(objective + 10.0 * constraint);

// Solve
let solution = model.solve_cpu(1000);  // 1000 samples
println!("Best solution: {:?}", solution.best_sample());
```

### Recipe: Loading Problems from File

```rust
use tytan::{QuboModel, io};

// Load QUBO matrix from file
let qubo = io::load_qubo("problem.txt")?;

// Or load from standard formats
let model = QuboModel::from_dimacs("maxcut.cnf")?;
let model = QuboModel::from_json("config.json")?;

// Solve with default parameters
let solution = model.solve();
```

## Basic Optimization Problems

### Recipe: Maximum Cut Problem

```rust
use tytan::{Symbol, compile, Graph};

fn max_cut(edges: Vec<(usize, usize, f64)>, n_nodes: usize) -> QuboModel {
    let x = Symbol::new("node", n_nodes);
    
    let mut objective = 0.0;
    for (i, j, weight) in edges {
        // Maximize edges between different partitions
        objective += weight * (x[i] - x[j]).pow(2);
    }
    
    compile(-objective)  // Negative because we maximize
}

// Example usage
let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
let model = max_cut(edges, 3);
let solution = model.solve_gpu(10000);  // Use GPU for larger problems
```

### Recipe: Traveling Salesman Problem

```rust
use tytan::{Symbol, compile, one_hot};

fn tsp(distances: Vec<Vec<f64>>) -> QuboModel {
    let n = distances.len();
    // x[i][j] = 1 if city i is visited at position j
    let x = Symbol::matrix("route", n, n);
    
    // Objective: minimize total distance
    let mut objective = 0.0;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let next_k = (k + 1) % n;
                objective += distances[i][j] * x[i][k] * x[j][next_k];
            }
        }
    }
    
    // Constraints: each city visited once, one city per position
    let mut constraints = 0.0;
    for i in 0..n {
        constraints += one_hot(&x[i]).pow(2);  // Row constraint
        constraints += one_hot(&x.col(i)).pow(2);  // Column constraint
    }
    
    compile(objective + 100.0 * constraints)
}
```

### Recipe: Knapsack Problem

```rust
use tytan::{Symbol, compile};

fn knapsack(weights: Vec<f64>, values: Vec<f64>, capacity: f64) -> QuboModel {
    let n = weights.len();
    let x = Symbol::new("item", n);
    
    // Maximize value
    let objective: f64 = -(0..n).map(|i| values[i] * x[i]).sum();
    
    // Weight constraint using slack variables
    let slack_bits = (capacity.log2() as usize) + 1;
    let s = Symbol::new("slack", slack_bits);
    
    let total_weight: f64 = (0..n).map(|i| weights[i] * x[i]).sum();
    let slack_value: f64 = (0..slack_bits).map(|i| 2f64.powi(i as i32) * s[i]).sum();
    
    let constraint = (total_weight + slack_value - capacity).pow(2);
    
    compile(objective + 50.0 * constraint)
}
```

## Constraint Handling

### Recipe: Equality Constraints

```rust
use tytan::{Symbol, compile};

// Constraint: x + y + z = 2
fn equality_constraint() -> QuboModel {
    let vars = Symbol::new("v", 3);
    let constraint = (vars[0] + vars[1] + vars[2] - 2.0).pow(2);
    compile(100.0 * constraint)  // Large penalty weight
}
```

### Recipe: Inequality Constraints with Slack Variables

```rust
use tytan::{Symbol, compile};

// Constraint: 2x + 3y <= 5
fn inequality_constraint() -> QuboModel {
    let x = Symbol::new("x", 1);
    let y = Symbol::new("y", 1);
    let slack = Symbol::new("slack", 3);  // Binary encoding for slack
    
    // Slack can represent 0-7
    let slack_value = slack[0] + 2.0 * slack[1] + 4.0 * slack[2];
    
    // 2x + 3y + slack = 5
    let constraint = (2.0 * x[0] + 3.0 * y[0] + slack_value - 5.0).pow(2);
    
    compile(constraint)
}
```

### Recipe: Soft Constraints

```rust
use tytan::{Symbol, compile};

fn soft_constraints() -> QuboModel {
    let x = Symbol::new("x", 5);
    
    // Primary objective
    let objective = x[0] + x[1] + x[2];
    
    // Soft constraint with variable penalty
    let soft_constraint = (x[3] + x[4] - 1.0).pow(2);
    
    // Adaptive penalty weight
    let penalty_weight = 10.0;  // Start with moderate penalty
    
    compile(objective + penalty_weight * soft_constraint)
}
```

## GPU Acceleration

### Recipe: GPU-Accelerated Sampling

```rust
use tytan::{QuboModel, GpuSampler, SamplerConfig};

async fn gpu_sampling(model: QuboModel) {
    // Configure GPU sampler
    let config = SamplerConfig::default()
        .with_num_samples(100_000)
        .with_num_sweeps(100)
        .with_batch_size(1024)
        .with_temperature_schedule(vec![5.0, 2.0, 1.0, 0.5, 0.1]);
    
    // Create GPU sampler
    let sampler = GpuSampler::new(config).await?;
    
    // Run sampling
    let solution = sampler.sample(&model).await?;
    
    // Analyze results
    println!("Best energy: {}", solution.best_energy());
    println!("Unique solutions: {}", solution.unique_solutions());
}
```

### Recipe: Multi-GPU Optimization

```rust
use tytan::{QuboModel, MultiGpuSampler};

async fn multi_gpu_solve(model: QuboModel) {
    // Automatically use all available GPUs
    let sampler = MultiGpuSampler::new()
        .with_replicas_per_gpu(4)
        .with_exchange_interval(10);
    
    // Parallel tempering across GPUs
    let solution = sampler.parallel_tempering(&model).await?;
    
    println!("Speedup: {}x", solution.speedup_factor());
}
```

## Hybrid Algorithms

### Recipe: QAOA Implementation

```rust
use tytan::{Symbol, compile, qaoa};

fn qaoa_max_cut(edges: Vec<(usize, usize)>, n_nodes: usize) {
    // Create problem
    let x = Symbol::new("node", n_nodes);
    let mut hamiltonian = 0.0;
    
    for (i, j) in edges {
        hamiltonian += (x[i] - x[j]).pow(2);
    }
    
    let model = compile(-hamiltonian);
    
    // QAOA optimization
    let qaoa_config = qaoa::Config::default()
        .with_layers(3)
        .with_optimizer("COBYLA")
        .with_max_iterations(100);
    
    let (params, energy) = qaoa::optimize(model, qaoa_config)?;
    println!("Optimized parameters: {:?}", params);
    println!("Expected energy: {}", energy);
}
```

### Recipe: VQE for Chemistry Problems

```rust
use tytan::{vqe, molecular_hamiltonian};

fn vqe_h2_molecule(bond_length: f64) {
    // Create molecular Hamiltonian
    let h2 = molecular_hamiltonian::h2(bond_length);
    
    // Convert to QUBO (if possible)
    let qubo = h2.to_qubo()?;
    
    // Run VQE
    let vqe_config = vqe::Config::default()
        .with_ansatz("UCCSD")
        .with_initial_params("random");
    
    let ground_state = vqe::find_ground_state(qubo, vqe_config)?;
    println!("Ground state energy: {}", ground_state.energy);
}
```

## Advanced Techniques

### Recipe: Problem Decomposition

```rust
use tytan::{QuboModel, decompose, GraphPartitioner};

fn solve_large_problem(model: QuboModel) {
    // Automatic decomposition
    let partitioner = GraphPartitioner::new()
        .with_method("spectral")
        .with_max_partition_size(50);
    
    let subproblems = partitioner.decompose(&model)?;
    
    // Solve subproblems in parallel
    let solutions: Vec<_> = subproblems
        .par_iter()
        .map(|sub| sub.solve())
        .collect();
    
    // Merge solutions
    let final_solution = merge_solutions(solutions, &model);
}
```

### Recipe: Adaptive Sampling

```rust
use tytan::{AdaptiveSampler, QuboModel};

fn adaptive_optimization(model: QuboModel) {
    let sampler = AdaptiveSampler::new()
        .with_initial_samples(1000)
        .with_convergence_threshold(1e-6)
        .with_max_iterations(50);
    
    // Sampler automatically adjusts parameters
    let solution = sampler.solve(&model)?;
    
    println!("Samples used: {}", solution.total_samples());
    println!("Convergence achieved: {}", solution.converged());
}
```

### Recipe: Machine Learning Integration

```rust
use tytan::{QuboModel, ml::QMLOptimizer};

fn ml_enhanced_optimization(model: QuboModel, training_data: Vec<QuboModel>) {
    // Train ML model on similar problems
    let optimizer = QMLOptimizer::new()
        .train(&training_data)?
        .with_feature_extraction("graph_properties");
    
    // Use ML to guide optimization
    let solution = optimizer.solve(&model)?;
    
    // ML provides initial parameters and strategy
    println!("ML-suggested strategy: {:?}", solution.strategy());
}
```

## Performance Tuning

### Recipe: Profiling and Optimization

```rust
use tytan::{QuboModel, Profiler};

fn profile_optimization(model: QuboModel) {
    let profiler = Profiler::new();
    
    // Run with profiling
    let solution = profiler.profile(|| {
        model.solve_with_config(
            SolverConfig::default()
                .with_num_samples(10000)
                .with_num_threads(8)
        )
    })?;
    
    // Analyze performance
    println!("Time breakdown:");
    println!("  Initialization: {}ms", profiler.init_time());
    println!("  Sampling: {}ms", profiler.sampling_time());
    println!("  Post-processing: {}ms", profiler.post_time());
    
    // Get optimization suggestions
    let suggestions = profiler.suggest_improvements();
    for suggestion in suggestions {
        println!("Suggestion: {}", suggestion);
    }
}
```

### Recipe: Memory-Efficient Large Problems

```rust
use tytan::{SparseQuboModel, StreamingSampler};

fn solve_sparse_problem(n_vars: usize, interactions: Vec<(usize, usize, f64)>) {
    // Use sparse representation
    let model = SparseQuboModel::from_interactions(n_vars, interactions);
    
    // Streaming sampler for memory efficiency
    let sampler = StreamingSampler::new()
        .with_chunk_size(1000)
        .with_disk_cache("cache/");
    
    // Process results as they arrive
    sampler.sample_streaming(&model, |batch| {
        println!("Processing batch with {} samples", batch.len());
        // Process without storing all samples
    })?;
}
```

## Real Hardware Usage

### Recipe: D-Wave Quantum Annealer

```rust
use tytan::{QuboModel, DWaveSampler};

async fn dwave_optimization(model: QuboModel) {
    // Configure D-Wave sampler
    let sampler = DWaveSampler::new()
        .with_token(std::env::var("DWAVE_TOKEN")?)
        .with_solver("Advantage_system6.1")
        .with_embedding_method("minorminer");
    
    // Submit to quantum annealer
    let solution = sampler.sample(&model)
        .with_num_reads(1000)
        .with_annealing_time(20)  // microseconds
        .submit()
        .await?;
    
    println!("Quantum solution: {:?}", solution.best());
    println!("Chain break fraction: {}", solution.chain_break_fraction());
}
```

### Recipe: Hybrid Classical-Quantum

```rust
use tytan::{HybridSolver, QuboModel};

async fn hybrid_solve(model: QuboModel) {
    // Automatically decides classical vs quantum
    let solver = HybridSolver::new()
        .with_quantum_backend("dwave")
        .with_classical_backend("gurobi")
        .with_selection_strategy("problem_size");
    
    let solution = solver.solve(&model).await?;
    
    println!("Used backend: {}", solution.backend_used());
    println!("Solution quality: {}", solution.quality_score());
}
```

## Tips and Best Practices

### 1. Choosing Penalty Weights
- Start with penalty = 10 * |objective_range|
- Use binary search to find optimal penalty
- Consider adaptive penalty methods

### 2. Scaling Variables
- Keep coefficients in similar ranges
- Use integer coefficients when possible
- Scale to avoid numerical issues

### 3. GPU vs CPU
- Use GPU for >100 variables
- Use CPU for sparse problems
- Consider memory limitations

### 4. Convergence Checks
- Monitor energy variance
- Check solution diversity
- Use multiple random seeds

### 5. Debugging
- Visualize problem structure
- Check constraint satisfaction
- Validate QUBO formulation

## Common Pitfalls

1. **Insufficient Penalty Weights**: Constraints not satisfied
2. **Numerical Overflow**: Coefficients too large
3. **Poor Variable Encoding**: Inefficient binary representation
4. **Ignoring Problem Structure**: Missing optimization opportunities
5. **Over-constraining**: Making problem infeasible

## Further Resources

- [API Documentation](../api/index.html)
- [Examples Directory](../../examples/)
- [Performance Guide](performance.md)
- [Hardware Backends](hardware.md)
- [Community Forum](https://forum.tytan.dev)