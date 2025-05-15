# QuantRS2-Anneal: Quantum Annealing Framework

[![Crates.io](https://img.shields.io/crates/v/quantrs2-anneal.svg)](https://crates.io/crates/quantrs2-anneal)
[![Documentation](https://docs.rs/quantrs2-anneal/badge.svg)](https://docs.rs/quantrs2-anneal)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Anneal is part of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, providing support for quantum annealing and optimization problems.

## Features

- **Ising Model**: Complete implementation of Ising model with biases and couplings
- **QUBO Formulations**: Quadratic Unconstrained Binary Optimization problem support
- **Annealing Simulators**: Classical and quantum annealing simulation for local testing
- **D-Wave Integration**: Connect to D-Wave quantum annealing hardware (optional)
- **Optimization Problems**: Built-in formulations for common optimization problems
- **Flexible API**: Simple and intuitive interface for problem formulation and solving

## Usage

### Solving an Ising Model with Classical Annealing

```rust
use quantrs2_anneal::{
    ising::IsingModel,
    simulator::{ClassicalAnnealingSimulator, AnnealingParams}
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple 3-qubit Ising model
    let mut model = IsingModel::new(3);
    model.set_bias(0, 1.0)?;
    model.set_bias(1, -0.5)?;
    model.set_coupling(0, 1, -1.0)?;
    model.set_coupling(1, 2, 0.5)?;

    // Configure annealing parameters
    let mut params = AnnealingParams::new();
    params.num_sweeps = 1000;
    params.num_repetitions = 20;
    params.initial_temperature = 10.0;
    params.final_temperature = 0.1;

    // Create a classical annealing simulator and solve the model
    let simulator = ClassicalAnnealingSimulator::new(params)?;
    let result = simulator.solve(&model)?;

    println!("Best energy: {}", result.best_energy);
    println!("Best solution: {:?}", result.best_spins);
    
    Ok(())
}
```

### QUBO Problem Formulation

```rust
use quantrs2_anneal::{
    qubo::{QuboBuilder, QuboFormulation},
    simulator::{ClassicalAnnealingSimulator, AnnealingParams}
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a QUBO for a simple constraint problem
    // Example: Find binary variables x0, x1, x2 such that x0 + x1 + x2 = 2
    let mut qubo = QuboBuilder::new(3);
    
    // Add the constraint (x0 + x1 + x2 - 2)^2 which is minimized at 0 when x0 + x1 + x2 = 2
    qubo.add_constraint_eq(&[0, 1, 2], &[1.0, 1.0, 1.0], 2.0, 5.0)?;
    
    // Build the QUBO formulation
    let formulation = qubo.build()?;
    
    // Convert to Ising model
    let ising_model = formulation.to_ising_model()?;
    
    // Solve using simulated annealing
    let params = AnnealingParams::default();
    let simulator = ClassicalAnnealingSimulator::new(params)?;
    let result = simulator.solve(&ising_model)?;
    
    // Convert result back to binary variables
    let binary_result = formulation.to_binary_solution(&result.best_spins)?;
    println!("Solution: {:?}", binary_result);
    
    Ok(())
}
```

### Using D-Wave Hardware (with `dwave` feature)

```rust
#[cfg(feature = "dwave")]
use quantrs2_anneal::{
    DWaveClient,
    ising::IsingModel,
};

#[cfg(feature = "dwave")]
async fn run_on_dwave() -> Result<(), Box<dyn std::error::Error>> {
    // Check if D-Wave support is available
    if !quantrs2_anneal::is_hardware_available() {
        println!("D-Wave support not available");
        return Ok(());
    }
    
    // Create a simple Ising model
    let mut model = IsingModel::new(4);
    model.set_bias(0, 1.0)?;
    model.set_coupling(0, 1, -1.0)?;
    
    // Create a D-Wave client
    let token = std::env::var("DWAVE_API_TOKEN")?;
    let client = DWaveClient::new(&token).await?;
    
    // Solve the model on D-Wave hardware
    let result = client.solve_ising(&model, 100).await?;
    
    println!("Best energy: {}", result.best_energy);
    println!("Best solution: {:?}", result.best_spins);
    
    Ok(())
}
```

## Module Structure

- **ising.rs**: Ising model representation and operations
- **qubo.rs**: QUBO problem formulation and constraints
- **simulator.rs**: Classical and quantum annealing simulators
- **dwave.rs**: D-Wave hardware integration (with feature flag)

## Feature Flags

- **default**: Basic functionality without external dependencies
- **dwave**: Enables D-Wave hardware integration (requires network connectivity)

## Implementation Notes

- Ising model representation uses sparse matrices for efficiency
- Simulated annealing uses a configurable temperature schedule
- Quantum annealing simulation uses path integral Monte Carlo
- QUBO formulations automatically handle constraint penalty weights

## Future Plans

See [TODO.md](TODO.md) for planned improvements and features.

## Integration with Other QuantRS2 Modules

This module is designed to work seamlessly with:
- [quantrs2-core](../core/README.md): Uses core types and error handling
- [quantrs2-tytan](../tytan/README.md): Advanced symbolic QUBO formulation

## License

This project is licensed under either:

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.