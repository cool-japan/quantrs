# Quick Start Guide

Get up and running with QuantRS2-Anneal in minutes! This guide covers installation, basic usage, and your first optimization problems.

## Installation

### Basic Installation

Add QuantRS2-Anneal to your `Cargo.toml`:

```toml
[dependencies]
quantrs2-anneal = "0.1"
```

### With Cloud Features

For cloud quantum hardware access:

```toml
[dependencies]
quantrs2-anneal = { version = "0.1", features = ["dwave", "braket", "fujitsu"] }
```

### Build and Test

```bash
# Basic build
cargo build

# Build with all features
cargo build --all-features

# Run tests
cargo test

# Run examples
cargo run --example simple_penalty_optimization
```

## Your First Optimization Problem

### Hello World: Max-Cut Problem

Let's solve a simple Maximum Cut problem on a 4-node graph:

```rust
use quantrs2_anneal::{
    ising::IsingModel,
    simulator::{ClassicalAnnealingSimulator, AnnealingParams}
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 4-qubit Max-Cut problem (square graph)
    let mut model = IsingModel::new(4);
    
    // Add edges with negative couplings (encourages cutting)
    model.set_coupling(0, 1, -1.0)?;  // Edge 0-1
    model.set_coupling(1, 2, -1.0)?;  // Edge 1-2
    model.set_coupling(2, 3, -1.0)?;  // Edge 2-3
    model.set_coupling(3, 0, -1.0)?;  // Edge 3-0
    
    // Create and configure simulator
    let params = AnnealingParams {
        num_sweeps: 1000,
        num_repetitions: 10,
        initial_temperature: 2.0,
        final_temperature: 0.1,
        ..Default::default()
    };
    
    let simulator = ClassicalAnnealingSimulator::new(params)?;
    
    // Solve the problem
    let result = simulator.solve(&model)?;
    
    // Print results
    println!("Max-Cut Solution:");
    println!("  Best energy: {}", result.best_energy);
    println!("  Best solution: {:?}", result.best_spins);
    println!("  Cut edges: {}", count_cut_edges(&result.best_spins));
    
    Ok(())
}

fn count_cut_edges(spins: &[i8]) -> usize {
    let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
    edges.iter()
        .filter(|(i, j)| spins[*i] != spins[*j])
        .count()
}
```

### Output
```
Max-Cut Solution:
  Best energy: -4.0
  Best solution: [1, -1, 1, -1]
  Cut edges: 4
```

## Common Problem Types

### 1. QUBO Problem with Constraints

Portfolio selection: choose exactly 3 out of 5 assets to maximize return while minimizing risk.

```rust
use quantrs2_anneal::{
    qubo::{QuboBuilder, QuboFormulation},
    simulator::ClassicalAnnealingSimulator
};

fn portfolio_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // Expected returns for 5 assets
    let returns = vec![0.12, 0.08, 0.15, 0.10, 0.09];
    
    // Risk matrix (simplified)
    let risk_matrix = vec![
        vec![0.04, 0.01, 0.02, 0.01, 0.01],
        vec![0.01, 0.02, 0.01, 0.01, 0.01],
        vec![0.02, 0.01, 0.06, 0.02, 0.01],
        vec![0.01, 0.01, 0.02, 0.03, 0.01],
        vec![0.01, 0.01, 0.01, 0.01, 0.02],
    ];
    
    let mut qubo = QuboBuilder::new(5);
    
    // Objective: maximize return - risk_penalty * risk
    for i in 0..5 {
        qubo.add_linear_term(i, -returns[i])?; // Negative for maximization
        for j in 0..5 {
            qubo.add_quadratic_term(i, j, 0.5 * risk_matrix[i][j])?;
        }
    }
    
    // Constraint: select exactly 3 assets
    let selection_vars: Vec<usize> = (0..5).collect();
    let selection_coeffs = vec![1.0; 5];
    qubo.add_constraint_eq(&selection_vars, &selection_coeffs, 3.0, 10.0)?;
    
    // Solve
    let formulation = qubo.build()?;
    let ising_model = formulation.to_ising_model()?;
    
    let simulator = ClassicalAnnealingSimulator::new(Default::default())?;
    let result = simulator.solve(&ising_model)?;
    
    // Convert back to binary solution
    let portfolio = formulation.to_binary_solution(&result.best_spins)?;
    
    println!("Portfolio Optimization:");
    println!("  Selected assets: {:?}", portfolio);
    
    let selected_assets: Vec<usize> = portfolio.iter()
        .enumerate()
        .filter(|(_, &selected)| selected == 1)
        .map(|(i, _)| i)
        .collect();
    
    println!("  Asset indices: {:?}", selected_assets);
    
    let total_return: f64 = selected_assets.iter()
        .map(|&i| returns[i])
        .sum();
    
    println!("  Expected return: {:.1}%", total_return * 100.0);
    
    Ok(())
}
```

### 2. Graph Coloring Problem

Color a graph with minimum colors such that no adjacent nodes have the same color.

```rust
use quantrs2_anneal::{
    ising::IsingModel,
    applications::graph_problems::GraphColoringBuilder
};

fn graph_coloring() -> Result<(), Box<dyn std::error::Error>> {
    // Define a simple graph (triangle + one extra node)
    let edges = vec![(0, 1), (1, 2), (2, 0), (0, 3)];
    let num_nodes = 4;
    let num_colors = 3;
    
    let builder = GraphColoringBuilder::new(num_nodes, num_colors);
    let ising_model = builder.build_ising_model(&edges)?;
    
    let simulator = ClassicalAnnealingSimulator::new(Default::default())?;
    let result = simulator.solve(&ising_model)?;
    
    // Decode coloring
    let coloring = builder.decode_coloring(&result.best_spins)?;
    
    println!("Graph Coloring:");
    for (node, color) in coloring.iter().enumerate() {
        println!("  Node {}: Color {}", node, color);
    }
    
    // Verify no conflicts
    let conflicts = edges.iter()
        .filter(|(i, j)| coloring[*i] == coloring[*j])
        .count();
    
    println!("  Conflicts: {}", conflicts);
    println!("  Valid coloring: {}", conflicts == 0);
    
    Ok(())
}
```

## Cloud Quantum Computing

### D-Wave Leap (Requires API Token)

```rust
#[cfg(feature = "dwave")]
use quantrs2_anneal::dwave::{DWaveClient, SolverSelector, SolverCategory};

#[cfg(feature = "dwave")]
async fn dwave_example() -> Result<(), Box<dyn std::error::Error>> {
    // Set your D-Wave API token
    let api_token = std::env::var("DWAVE_API_TOKEN")
        .expect("Set DWAVE_API_TOKEN environment variable");
    
    // Create client
    let client = DWaveClient::new(api_token, None)?;
    
    // Auto-select a QPU solver
    let selector = SolverSelector {
        category: SolverCategory::QPU,
        online_only: true,
        ..Default::default()
    };
    
    match client.select_solver(Some(&selector)) {
        Ok(solver) => {
            println!("Selected solver: {}", solver.name);
            
            // Create a simple problem
            let mut model = IsingModel::new(4);
            model.set_coupling(0, 1, -1.0)?;
            model.set_coupling(1, 2, -1.0)?;
            model.set_coupling(2, 3, -1.0)?;
            model.set_coupling(3, 0, -1.0)?;
            
            // Submit with automatic embedding
            let solution = client.submit_ising_with_embedding(
                &model,
                None, // Auto-select solver
                None, // Default parameters
                None, // Auto-embedding
            )?;
            
            println!("D-Wave Solution:");
            println!("  Problem ID: {}", solution.problem_id);
            println!("  Best energy: {}", 
                     solution.energies.iter().fold(f64::INFINITY, |a, &b| a.min(b)));
        }
        Err(e) => {
            println!("No QPU available: {}. Using simulator instead.", e);
            // Fall back to classical simulation
        }
    }
    
    Ok(())
}

#[cfg(not(feature = "dwave"))]
fn dwave_example() {
    println!("D-Wave support not compiled. Enable with --features dwave");
}
```

### AWS Braket (Requires AWS Credentials)

```rust
#[cfg(feature = "braket")]
use quantrs2_anneal::braket::{BraketClient, DeviceSelector, DeviceType};

#[cfg(feature = "braket")]
async fn braket_example() -> Result<(), Box<dyn std::error::Error>> {
    // Set up AWS credentials (or use IAM roles)
    let access_key = std::env::var("AWS_ACCESS_KEY_ID")?;
    let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY")?;
    let region = std::env::var("AWS_REGION").unwrap_or("us-east-1".to_string());
    
    let client = BraketClient::new(access_key, secret_key, region)?;
    
    // Auto-select a simulator (free)
    let selector = DeviceSelector {
        device_type: Some(DeviceType::Simulator),
        ..Default::default()
    };
    
    match client.select_device(Some(&selector)) {
        Ok(device) => {
            println!("Selected device: {}", device.device_name);
            
            // Create problem
            let mut model = IsingModel::new(4);
            model.set_coupling(0, 1, -1.0)?;
            model.set_coupling(1, 2, -1.0)?;
            
            // Submit task
            let task_result = client.submit_ising(&model, None, None)?;
            
            println!("AWS Braket Task:");
            println!("  Task ARN: {}", task_result.task_arn);
            println!("  Status: {:?}", task_result.task_status);
        }
        Err(e) => {
            println!("No devices available: {}", e);
        }
    }
    
    Ok(())
}

#[cfg(not(feature = "braket"))]
fn braket_example() {
    println!("AWS Braket support not compiled. Enable with --features braket");
}
```

## Advanced Features

### Population Annealing (High Quality Solutions)

```rust
use quantrs2_anneal::population_annealing::{
    PopulationAnnealingSimulator, PopulationParams
};

fn high_quality_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // Create a more complex problem
    let mut model = IsingModel::new(10);
    
    // Add random couplings
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    for i in 0..10 {
        for j in (i + 1)..10 {
            if rng.gen::<f64>() < 0.3 { // 30% connectivity
                let strength = rng.gen_range(-2.0..2.0);
                model.set_coupling(i, j, strength)?;
            }
        }
    }
    
    // Use population annealing for high-quality solutions
    let pop_params = PopulationParams {
        population_size: 1000,
        num_sweeps: 2000,
        resampling_threshold: 0.5,
        temperature_schedule: vec![2.0, 1.5, 1.0, 0.5, 0.2, 0.1, 0.05],
        ..Default::default()
    };
    
    let pop_simulator = PopulationAnnealingSimulator::new(pop_params)?;
    let result = pop_simulator.solve(&model)?;
    
    println!("Population Annealing Result:");
    println!("  Best energy: {}", result.best_energy);
    println!("  Population diversity: {:.3}", result.final_diversity);
    println!("  Effective sample size: {}", result.effective_sample_size);
    
    Ok(())
}
```

### Coherent Ising Machine (Large Problems)

```rust
use quantrs2_anneal::coherent_ising_machine::{
    CoherentIsingMachine, CIMParams
};

fn large_scale_optimization() -> Result<(), Box<dyn std::error::Error>> {
    // Create a large sparse problem
    let mut model = IsingModel::new(1000);
    
    // Add sparse connectivity (ring + random long-range)
    for i in 0..1000 {
        // Ring connectivity
        model.set_coupling(i, (i + 1) % 1000, -1.0)?;
        
        // Random long-range connections
        if rand::random::<f64>() < 0.01 {
            let j = rand::random::<usize>() % 1000;
            if i != j {
                model.set_coupling(i, j, rand::random::<f64>() * 2.0 - 1.0)?;
            }
        }
    }
    
    // Use CIM for large-scale optimization
    let cim_params = CIMParams {
        pump_power: 2.0,
        detuning: 0.1,
        coupling_strength: 0.5,
        num_iterations: 1000,
        convergence_threshold: 1e-6,
        ..Default::default()
    };
    
    let cim = CoherentIsingMachine::new(cim_params)?;
    let result = cim.solve(&model)?;
    
    println!("Coherent Ising Machine Result:");
    println!("  Problem size: {} variables", model.num_qubits);
    println!("  Best energy: {}", result.best_energy);
    println!("  Convergence iterations: {}", result.iterations_to_convergence);
    println!("  Final amplitude variance: {:.6}", result.final_variance);
    
    Ok(())
}
```

## Problem Builder DSL

For complex problems, use the Domain-Specific Language:

```rust
use quantrs2_anneal::dsl::{ProblemBuilder, Variable, Constraint};

fn dsl_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ProblemBuilder::new();
    
    // Define variables
    let x = builder.binary_var("x")?;
    let y = builder.binary_var("y")?;
    let z = builder.binary_var("z")?;
    
    // Define objective: minimize x + y - 2*z + x*y
    builder.minimize(
        x + y - 2*z + x*y
    )?;
    
    // Add constraints
    builder.add_constraint(
        x + y + z == 2  // Exactly 2 variables should be 1
    )?;
    
    builder.add_constraint(
        x - y >= 0      // x should be >= y
    )?;
    
    // Build and solve
    let problem = builder.build()?;
    let ising_model = problem.to_ising_model()?;
    
    let simulator = ClassicalAnnealingSimulator::new(Default::default())?;
    let result = simulator.solve(&ising_model)?;
    
    // Extract solution
    let solution = problem.extract_solution(&result.best_spins)?;
    
    println!("DSL Problem Solution:");
    for (var_name, value) in solution {
        println!("  {}: {}", var_name, value);
    }
    
    Ok(())
}
```

## Performance Tips

### 1. Choose the Right Algorithm

```rust
fn choose_algorithm(problem_size: usize, quality_requirement: f64) -> Box<dyn Optimizer> {
    match (problem_size, quality_requirement) {
        // Small problems, any quality
        (size, _) if size < 50 => 
            Box::new(ClassicalAnnealingSimulator::default()),
        
        // Medium problems, high quality needed
        (size, quality) if size < 500 && quality > 0.95 => 
            Box::new(PopulationAnnealingSimulator::default()),
        
        // Large problems
        (size, _) if size > 1000 => 
            Box::new(CoherentIsingMachine::default()),
        
        // Default
        _ => Box::new(ClassicalAnnealingSimulator::default())
    }
}
```

### 2. Use Parallel Processing

```rust
use quantrs2_anneal::simulator::ParallelAnnealingSimulator;

fn parallel_optimization() -> Result<(), Box<dyn std::error::Error>> {
    let params = AnnealingParams::default();
    let parallel_simulator = ParallelAnnealingSimulator::new(
        params,
        num_threads: num_cpus::get(),
    )?;
    
    let result = parallel_simulator.solve(&model)?;
    
    println!("Parallel result: {}", result.best_energy);
    Ok(())
}
```

## Next Steps

### 1. Explore Examples

Check out the [`examples/`](../examples/) directory for more comprehensive examples:

```bash
# Run all examples
cargo run --example simple_penalty_optimization
cargo run --example advanced_embedding --features dwave
cargo run --example coherent_ising_machine_example
cargo run --example energy_landscape_visualization
```

### 2. Read Documentation

- [Embedding Guide](embedding_guide.md) - Graph embedding techniques
- [Performance Guide](performance.md) - Optimization strategies
- [D-Wave Client Guide](dwave_leap_client.md) - Cloud quantum computing
- [AWS Braket Guide](aws_braket_client.md) - Multi-vendor quantum access

### 3. Set Up Cloud Access

#### For D-Wave Leap:
1. Register at [D-Wave Leap](https://cloud.dwavesys.com/)
2. Get your API token
3. Set environment variable: `export DWAVE_API_TOKEN="your_token"`

#### For AWS Braket:
1. Set up AWS account with Braket access
2. Configure credentials:
   ```bash
   export AWS_ACCESS_KEY_ID="your_key"
   export AWS_SECRET_ACCESS_KEY="your_secret"
   export AWS_REGION="us-east-1"
   ```

### 4. Try Real Applications

```rust
// Energy optimization
use quantrs2_anneal::applications::energy::PowerGridOptimizer;

// Financial optimization  
use quantrs2_anneal::applications::finance::PortfolioOptimizer;

// Logistics optimization
use quantrs2_anneal::applications::logistics::VehicleRoutingOptimizer;
```

## Troubleshooting

### Common Issues

1. **Compilation Errors with Features**
   ```bash
   # Make sure you enable the right features
   cargo build --features dwave,braket
   ```

2. **Poor Solution Quality**
   ```rust
   // Increase annealing parameters
   let params = AnnealingParams {
       num_sweeps: 10000,        // More sweeps
       num_repetitions: 100,     // More runs
       ..Default::default()
   };
   ```

3. **Memory Issues with Large Problems**
   ```rust
   // Use sparse matrices
   let sparse_config = SparseMatrixConfig::default();
   let model = IsingModel::with_sparse_config(num_qubits, sparse_config)?;
   ```

### Getting Help

- [GitHub Issues](https://github.com/cool-japan/quantrs/issues)
- [API Documentation](https://docs.rs/quantrs2-anneal)
- [Examples Directory](../examples/)

---

**🎉 Congratulations! You're now ready to solve optimization problems with quantum annealing!**

For more advanced features and optimization techniques, check out the comprehensive guides in the [`docs/`](../docs/) directory.