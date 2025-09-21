# quantrs2

[![Crates.io](https://img.shields.io/crates/v/quantrs2.svg)](https://crates.io/crates/quantrs2)
[![Documentation](https://docs.rs/quantrs2/badge.svg)](https://docs.rs/quantrs2)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

**Facade crate for QuantRS2: unified entry point and documentation**

This crate provides a single, convenient entry point that re-exports the public APIs from all QuantRS2 subcrates. Instead of managing multiple dependencies, you can simply add `quantrs2` with the features you need.

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
# Enable all features
quantrs2 = { version = "0.1.0-beta.1", features = ["full"] }

# Or select specific features
quantrs2 = { version = "0.1.0-beta.1", features = ["circuit", "sim"] }
```

Then use it in your code:

```rust
// With the facade crate, all modules are available under quantrs2::
use quantrs2::core::prelude::*;
use quantrs2::circuit::builder::Circuit;
use quantrs2::sim::statevector::StateVectorSimulator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a quantum circuit
    let mut circuit = Circuit::new();
    circuit.h(QubitId::new(0));
    circuit.cx(QubitId::new(0), QubitId::new(1));

    // Simulate the circuit
    let mut simulator = StateVectorSimulator::new();
    let result = simulator.run(&circuit)?;

    println!("Measurement results: {:?}", result);
    Ok(())
}
```

## Available Features

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `circuit` | Quantum circuit construction and optimization | `quantrs2-circuit` |
| `sim` | Quantum simulators (state vector, stabilizer, etc.) | `quantrs2-sim`, `circuit` |
| `anneal` | Quantum annealing algorithms | `quantrs2-anneal`, `circuit` |
| `device` | Hardware backends and device interfaces | `quantrs2-device`, `circuit` |
| `ml` | Quantum machine learning algorithms | `quantrs2-ml`, `sim`, `anneal` |
| `tytan` | TYTAN quantum annealing integration | `quantrs2-tytan`, `anneal` |
| `symengine` | Symbolic computation with SymEngine | `quantrs2-symengine` |
| `full` | All features enabled | All of the above |

## Module Structure

When you enable features, the corresponding modules become available:

```rust
// Core is always available
use quantrs2::core;

// Available with "circuit" feature
use quantrs2::circuit;

// Available with "sim" feature
use quantrs2::sim;

// Available with "anneal" feature
use quantrs2::anneal;

// Available with "device" feature
use quantrs2::device;

// Available with "ml" feature
use quantrs2::ml;

// Available with "tytan" feature
use quantrs2::tytan;

// Available with "symengine" feature
use quantrs2::symengine;
```

## Feature Dependencies

Some features automatically enable others for convenience:

- `sim` → enables `circuit`
- `anneal` → enables `circuit`
- `device` → enables `circuit`
- `ml` → enables `sim` and `anneal` (which also enables `circuit`)
- `tytan` → enables `anneal` (which also enables `circuit`)

## When to Use This Crate

**Use `quantrs2` when:**
- You want a simple, unified dependency
- You're building applications that use multiple QuantRS2 components
- You prefer feature flags over managing multiple crate dependencies
- You want the convenience of a single import namespace

**Use individual crates when:**
- You only need specific functionality (e.g., just `quantrs2-core`)
- You want minimal compile times and dependencies
- You're building libraries that should have minimal dependencies
- You need fine-grained control over versions

## Example Configurations

### Minimal quantum programming:
```toml
quantrs2 = { version = "0.1.0-beta.1", features = ["circuit"] }
```

### Circuit simulation:
```toml
quantrs2 = { version = "0.1.0-beta.1", features = ["sim"] }
```

### Quantum machine learning:
```toml
quantrs2 = { version = "0.1.0-beta.1", features = ["ml"] }
```

### Hardware interaction:
```toml
quantrs2 = { version = "0.1.0-beta.1", features = ["device", "sim"] }
```

### Everything:
```toml
quantrs2 = { version = "0.1.0-beta.1", features = ["full"] }
```

## Alternative: Individual Crates

If you prefer to use individual crates instead of the facade:

```toml
[dependencies]
quantrs2-core = "0.1.0-beta.1"
quantrs2-circuit = "0.1.0-beta.1"
quantrs2-sim = "0.1.0-beta.1"
# etc.
```

## Documentation

- [Main QuantRS2 Documentation](https://docs.rs/quantrs2-core)
- [Examples and Tutorials](https://github.com/cool-japan/quantrs/tree/master/examples)
- [Migration Guide](https://github.com/cool-japan/quantrs/blob/master/MIGRATION_GUIDE_ALPHA_TO_BETA.md)

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.