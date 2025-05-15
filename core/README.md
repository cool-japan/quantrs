# QuantRS2-Core: Foundational Quantum Types

[![Crates.io](https://img.shields.io/crates/v/quantrs2-core.svg)](https://crates.io/crates/quantrs2-core)
[![Documentation](https://docs.rs/quantrs2-core/badge.svg)](https://docs.rs/quantrs2-core)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Core is part of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, providing the foundational types and traits that form the basis of the entire ecosystem.

## Features

- **Type-Safe Qubit Identifiers**: Zero-cost abstractions for qubit references
- **Comprehensive Gate Definitions**: Standard quantum gates with matrix representations
- **Register Abstractions**: Quantum register types with const generics for compile-time validation
- **Robust Error Handling**: Comprehensive error types and result wrappers

## Usage

### Basic Example

```rust
use quantrs2_core::prelude::*;

fn main() -> QuantRS2Result<()> {
    // Create a qubit identifier
    let qubit = QubitId::new(0);
    
    // Use a standard gate
    let x_gate = XGate::new();
    let matrix = x_gate.matrix();
    
    // Create a register (with 2 qubits)
    let register = Register::<2>::new();
    
    Ok(())
}
```

### Creating Custom Gates

```rust
use quantrs2_core::prelude::*;
use num_complex::Complex64;

struct MyCustomGate;

impl GateOp for MyCustomGate {
    fn matrix(&self) -> Vec<Complex64> {
        // Define your custom gate matrix here
        // Example: custom rotation gate
        vec![
            Complex64::new(0.5, 0.0), Complex64::new(0.5, 0.5),
            Complex64::new(0.5, -0.5), Complex64::new(0.5, 0.0),
        ]
    }
    
    fn n_qubits(&self) -> usize {
        1 // This is a single-qubit gate
    }
}
```

## Module Structure

- **error.rs**: Error types and result wrappers
- **gate.rs**: Gate trait definitions and standard gate implementations
- **qubit.rs**: Qubit identifier type 
- **register.rs**: Quantum register abstractions

## API Overview

### Core Types

- `QubitId`: Type-safe wrapper around a qubit identifier
- `Register<N>`: Fixed-size quantum register using const generics
- `QuantRS2Error`: Error enumeration for all possible errors
- `QuantRS2Result<T>`: Convenient result type alias

### Important Traits

- `GateOp`: Trait for quantum gates with matrix representation
- `ControlledOp`: Trait for controlled quantum operations
- `StateVector`: Trait for quantum state representations

## Implementation Notes

- The `QubitId` type is a zero-cost abstraction using `#[repr(transparent)]`
- Gates are represented by their matrix form in column-major order
- Registers use Rust's const generics feature for compile-time qubit count validation
- Error handling follows Rust's standard practices with a custom error type

## Future Plans

See [TODO.md](TODO.md) for planned improvements and features.

## Integration with Other QuantRS2 Modules

This module is designed to work seamlessly with:
- [quantrs2-circuit](../circuit/README.md): Provides core types used in circuit construction
- [quantrs2-sim](../sim/README.md): Provides matrix representations used in simulation

## License

This project is licensed under either:

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.