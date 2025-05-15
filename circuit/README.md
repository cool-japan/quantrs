# QuantRS2-Circuit: Quantum Circuit Construction

[![Crates.io](https://img.shields.io/crates/v/quantrs2-circuit.svg)](https://crates.io/crates/quantrs2-circuit)
[![Documentation](https://docs.rs/quantrs2-circuit/badge.svg)](https://docs.rs/quantrs2-circuit)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Circuit is part of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, providing a flexible API for building and manipulating quantum circuits.

## Features

- **Fluent Circuit Builder API**: Create quantum circuits with an intuitive, chainable interface
- **Type-Safe Construction**: Compile-time checking of qubit counts and operations
- **Comprehensive Gate Support**: All standard gates plus controlled variants
- **DSL Macros**: Convenient macros for circuit creation and manipulation
- **Simulator Integration**: Seamless connection to various simulator backends

## Usage

### Creating a Bell State Circuit

```rust
use quantrs2_circuit::builder::Circuit;
use quantrs2_core::prelude::*;

fn main() -> QuantRS2Result<()> {
    // Create a circuit with 2 qubits
    let mut circuit = Circuit::<2>::new();
    
    // Build the Bell state circuit
    circuit.h(0)?
           .cnot(0, 1)?;
           
    // The circuit is now ready to run on a simulator
    Ok(())
}
```

### Using Circuit Macros

```rust
use quantrs2_circuit::{circuit, qubits};
use quantrs2_core::prelude::*;

fn main() -> QuantRS2Result<()> {
    // Create a circuit with 4 qubits
    let mut circuit = circuit!(4);
    
    // Add gates using the builder API
    circuit.h(0)?
           .cnot(0, 1)?
           .h(2)?
           .cnot(2, 3)?;
           
    // Define a set of qubits to operate on
    let qs = qubits![0, 1];
    
    // Operations can be applied to qubit sets
    // (this feature depends on implementation details)
    
    Ok(())
}
```

## Module Structure

- **builder.rs**: Main circuit builder implementation 
- **lib.rs**: Module exports and macro definitions

## API Overview

### Core Types

- `Circuit<N>`: Quantum circuit with const generic for qubit count
- `Simulator`: Trait for backends that can run quantum circuits

### Macros

- `circuit!`: Creates a new circuit with the specified number of qubits
- `qubits!`: Creates a set of qubits for operations
- `quantum!`: DSL for quantum circuit construction (in development)

## Implementation Notes

- The circuit builder uses a fluent API for method chaining
- Gate operations are type-checked at compile time where possible
- The implementation supports custom gates through the `GateOp` trait
- Circuit operations return `QuantRS2Result` for error handling

## Future Plans

See [TODO.md](TODO.md) for planned improvements and features.

## Integration with Other QuantRS2 Modules

This module is designed to work seamlessly with:
- [quantrs2-core](../core/README.md): Uses core types and gates for circuit construction
- [quantrs2-sim](../sim/README.md): Circuits can be executed on simulators
- [quantrs2-device](../device/README.md): Circuits can be transpiled and run on real hardware

## License

This project is licensed under either:

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.