# QuantRS2-Tytan

[![Crates.io](https://img.shields.io/crates/v/quantrs2-tytan.svg)](https://crates.io/crates/quantrs2-tytan)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Tytan is a high-level quantum annealing library for the QuantRS2 framework, inspired by the Python [Tytan](https://github.com/tytansdk/tytan) library. It provides easy-to-use interfaces for formulating and solving quantum annealing problems, with support for multiple backend solvers.

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
quantrs2-tytan = { git = "https://github.com/cool-japan/quantrs", version = "0.1.0-alpha.3" }
```

## Examples

### Example with symbolic math (requires 'dwave' feature)

```rust
// This example requires the 'dwave' feature
#[cfg(feature = "dwave")]
fn symbolic_example() -> Result<(), Box<dyn std::error::Error>> {
    use quantrs2_tytan::{symbols, Compile, SASampler};

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
    let result = solver.run_qubo(&qubo, 100)?;

    // Display results
    for r in &result {
        println!("{:?}", r);
    }

    Ok(())
}
```

### Example without symbolic math

```rust
use quantrs2_tytan::sampler::{SASampler, Sampler};
use std::collections::HashMap;
use ndarray::Array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple QUBO matrix manually
    let mut matrix = Array::<f64, _>::zeros((2, 2));
    matrix[[0, 0]] = -1.0;  // Linear term for x
    matrix[[1, 1]] = -1.0;  // Linear term for y
    matrix[[0, 1]] = 2.0;   // Quadratic term for x*y
    matrix[[1, 0]] = 2.0;   // Symmetric

    // Create variable map
    let mut var_map = HashMap::new();
    var_map.insert("x".to_string(), 0);
    var_map.insert("y".to_string(), 1);

    // Choose a sampler
    let solver = SASampler::new(None);

    // Sample
    let result = solver.run_hobo(&(matrix.into_dyn(), var_map), 100)?;

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
- `dwave`: Enable symbolic math and D-Wave connectivity (requires SymEngine)
- `scirs`: Enable high-performance computing with SciRS2 libraries
- `advanced_optimization`: Enable advanced optimization algorithms from SciRS2
- `gpu_accelerated`: Full GPU-acceleration with SciRS2 GPU primitives
- `clustering`: Enable basic solution clustering and analysis tools
- `plotters`: Enable visualization of energy distributions and other plots

## Integration with Existing QuantRS2 Modules

QuantRS2-Tytan is built on top of the core QuantRS2 annealing stack, extending it with higher-level interfaces. It's compatible with all existing Quantrs projects and can be used alongside other components.

## Building with SymEngine (for the `dwave` feature)

To build with SymEngine support on macOS:

1. Install the required libraries:
   ```bash
   brew install symengine gmp mpfr
   ```

2. Set the required environment variables:
   ```bash
   export SYMENGINE_DIR=$(brew --prefix symengine)
   export GMP_DIR=$(brew --prefix gmp)
   export MPFR_DIR=$(brew --prefix mpfr)
   export BINDGEN_EXTRA_CLANG_ARGS="-I$(brew --prefix symengine)/include -I$(brew --prefix gmp)/include -I$(brew --prefix mpfr)/include"
   ```

3. Build with the `dwave` feature:
   ```bash
   cargo build --features dwave
   ```

For more detailed instructions, see the [TODO.md](../TODO.md) file in the main project directory.

## License

This project is licensed under either:

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.