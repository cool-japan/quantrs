# Quantrs-Tytan

Quantrs-Tytan is a high-level quantum annealing library for the Quantrs framework, inspired by the Python [Tytan](https://github.com/tytansdk/tytan) library. It provides easy-to-use interfaces for formulating and solving quantum annealing problems, with support for multiple backend solvers.

## Features

- **Symbolic Problem Construction**: Define QUBO problems using symbolic expressions
- **Higher-Order Binary Optimization (HOBO)**: Support for terms beyond quadratic (3rd order and higher)
- **Multiple Samplers**: Choose from various solvers including:
  - Simulated Annealing
  - Genetic Algorithm
  - GPU-accelerated Annealing
  - External quantum annealing hardware (via D-Wave)
- **Auto Result Processing**: Automatically convert solutions to multi-dimensional arrays
- **SciRS2 Integration**: Optional high-performance scientific computing via SciRS2:
  - Optimized matrix operations with scirs2-linalg
  - Advanced numerical routines with scirs2-core
  - Specialized optimization algorithms with scirs2-optimize
  - Solution clustering analysis (disabled due to dependency conflicts)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
quantrs-tytan = { git = "https://github.com/your-repo/quantrs", version = "0.1.0" }
```

## Example

```rust
use quantrs_tytan::{symbols, Compile, SASampler, Auto_array};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Define variables
    let x = symbols("x");
    let y = symbols("y");
    let z = symbols("z");
    
    // Define expression (3 variables, want exactly 2 to be 1)
    let h = (x + y + z - 2).pow(2);
    
    // Compile to QUBO
    let (qubo, offset) = Compile::new(&h).get_qubo()?;
    
    // Choose a sampler
    let solver = SASampler::new(None);
    
    // Sample
    let result = solver.run(&qubo, 100)?;
    
    // Display results
    for r in &result {
        println!("{:?}", r);
    }
    
    Ok(())
}
```

## Features

- `parallel`: Enable multi-threading for samplers (default)
- `gpu`: Enable GPU-accelerated samplers
- `scirs`: Enable high-performance computing with SciRS2 libraries (without clustering)
- `advanced_optimization`: Enable advanced optimization algorithms from SciRS2
- `gpu_accelerated`: Full GPU-acceleration with SciRS2 GPU primitives
- `clustering`: Enable basic solution clustering and analysis tools (without scirs2-cluster)

## Integration with Existing Quantrs Modules

Quantrs-Tytan is built on top of the core Quantrs annealing stack, extending it with higher-level interfaces. It's compatible with all existing Quantrs projects and can be used alongside other components.

## License

Same license as Quantrs core.