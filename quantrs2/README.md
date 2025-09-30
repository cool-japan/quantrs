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
quantrs2 = { version = "0.1.0-beta.2", features = ["full"] }

# Or select specific features
quantrs2 = { version = "0.1.0-beta.2", features = ["circuit", "sim"] }
```

Then use it in your code:

```rust
// With the facade crate, modules are available under quantrs2::
use quantrs2::core::api::prelude::essentials::*; // QubitId, Register, QuantRS2Result
use quantrs2::circuit::builder::{Circuit, Simulator}; // Circuit and the Simulator trait
use quantrs2::sim::statevector::StateVectorSimulator; // Feature "sim"

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 2-qubit circuit (const-generic N)
    const N: usize = 2;
    let mut circuit = Circuit::<N>::new();
    circuit.h(QubitId::new(0))?;
    circuit.cx(QubitId::new(0), QubitId::new(1))?;

    // Simulate the circuit
    let mut simulator = StateVectorSimulator::new();
    let result: Register<N> = simulator.run(&circuit)?;

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
quantrs2 = { version = "0.1.0-beta.2", features = ["circuit"] }
```

### Circuit simulation:
```toml
quantrs2 = { version = "0.1.0-beta.2", features = ["sim"] }
```

### Quantum machine learning:
```toml
quantrs2 = { version = "0.1.0-beta.2", features = ["ml"] }
```

### Hardware interaction:
```toml
quantrs2 = { version = "0.1.0-beta.2", features = ["device", "sim"] }
```

### Everything:
```toml
quantrs2 = { version = "0.1.0-beta.2", features = ["full"] }
```

## Alternative: Individual Crates

If you prefer to use individual crates instead of the facade:

```toml
[dependencies]
quantrs2-core = "0.1.0-beta.2"
quantrs2-circuit = "0.1.0-beta.2"
quantrs2-sim = "0.1.0-beta.2"
# etc.
```

## End-to-End Example

This example builds a Bell state on 2 qubits and prints the probabilities. Enable features `circuit` and `sim`.

```rust
use quantrs2::core::api::prelude::essentials::*;
use quantrs2::circuit::builder::{Circuit, Simulator};
use quantrs2::sim::statevector::StateVectorSimulator;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 2;
    let mut circuit = Circuit::<N>::new();
    circuit.h(QubitId::new(0))?;                  // H on qubit 0
    circuit.cx(QubitId::new(0), QubitId::new(1))?; // CNOT 0->1

    let sim = StateVectorSimulator::new();
    let reg: Register<N> = sim.run(&circuit)?;

    let probs = reg.probabilities();
    println!(
        "|00>: {:.3}, |01>: {:.3}, |10>: {:.3}, |11>: {:.3}",
        probs[0], probs[1], probs[2], probs[3]
    );
    Ok(())
}
```

## Documentation

- [Main QuantRS2 Documentation](https://docs.rs/quantrs2)
- [Examples and Tutorials](https://github.com/cool-japan/quantrs/tree/master/examples)

## Subcrates

- `quantrs2-core`: Core types, math, error handling, and APIs — https://github.com/cool-japan/quantrs/tree/master/core
- `quantrs2-circuit`: Circuit builder, DSL, optimization — https://github.com/cool-japan/quantrs/tree/master/circuit
- `quantrs2-sim`: Simulators (statevector, stabilizer, MPS, etc.) — https://github.com/cool-japan/quantrs/tree/master/sim
- `quantrs2-anneal`: Quantum annealing algorithms and workflows — https://github.com/cool-japan/quantrs/tree/master/anneal
- `quantrs2-device`: Hardware/device connectors and scheduling — https://github.com/cool-japan/quantrs/tree/master/device
- `quantrs2-ml`: Quantum machine learning utilities — https://github.com/cool-japan/quantrs/tree/master/ml
- `quantrs2-tytan`: High-level annealing interface inspired by TYTAN — https://github.com/cool-japan/quantrs/tree/master/tytan
- `quantrs2-symengine`: Symbolic computation bindings — https://github.com/cool-japan/quantrs/tree/master/quantrs2-symengine

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.