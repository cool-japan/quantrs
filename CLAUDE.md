# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantRS2 is a Rust-based quantum computing framework that provides a modular, high-performance toolkit for quantum simulation, algorithm development, and hardware interaction. The framework supports both gate-based quantum computing and quantum annealing paradigms.

## Commands

### Building

```bash
# Build all crates
cargo build

# Build with release optimizations
cargo build --release

# Build a specific crate
cargo build -p quantrs2-core
```

### Testing

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p quantrs2-core

# Run a specific test
cargo test -p quantrs2-core -- qubit_operations
```

### Code Checks

```bash
# Check if code compiles without building
cargo check

# Check a specific crate
cargo check -p quantrs2-circuit
```

## Architecture

The project is organized as a Rust workspace with multiple crates:

1. **quantrs2-core**: Contains fundamental types and traits:
   - `QubitId`: Type-safe wrapper around a numeric qubit identifier
   - `GateOp`: Trait for quantum gates with matrix representation
   - `Register<N>`: State representation with compile-time qubit count

2. **quantrs2-circuit**: Provides circuit building capabilities:
   - `Circuit<N>`: Main circuit type parameterized by qubit count
   - Builder methods for common gates (h, x, cnot, etc.)
   - Macros for easier circuit construction

3. **quantrs2-sim**: Simulator implementations:
   - `StateVectorSimulator`: Full state vector simulation
   - `TensorNetworkSimulator`: (Placeholder for future implementation)
   - Uses multi-threading via Rayon for performance

4. **quantrs2-anneal**: Quantum annealing support (planned)
   - Will integrate with D-Wave systems
   - QUBO/Ising model formulation

5. **quantrs2-device**: Hardware connections (planned)
   - Will provide interfaces to IBM Quantum, Azure Quantum, etc.

6. **quantrs2-py**: Python bindings (planned)
   - Will expose Rust functionality to Python

## Key Design Patterns

1. **Zero-cost abstractions**: Uses Rust's type system to provide compile-time safety without runtime overhead.

2. **Const generics**: Circuit and Register types use const generics for qubit count, enabling compile-time verification.

3. **Builder pattern**: Circuit construction uses a fluent API with method chaining.

4. **Trait-based architecture**: Core functionality is defined through traits like `GateOp` and `Simulator`.

5. **Type-safe operations**: Operations on invalid qubits result in compile-time or runtime errors rather than undefined behavior.

## Implementation Notes

- The `GateOp` trait requires gate implementations to provide matrix representations
- For performance reasons, the CNOT and SWAP gates have specialized implementations
- The statevector simulator has both parallel (using Rayon) and sequential execution modes
- The codebase is designed to scale to large qubit counts (30+) through performance optimizations

## Building on macOS

On macOS, there might be issues with building the BLAS/LAPACK dependencies. If you encounter problems, try:

```bash
# Use system BLAS (Accelerate framework on macOS)
OPENBLAS_SYSTEM=1 cargo build
```

Or set these environment variables in your .cargo/config.toml:

```toml
[env]
OPENBLAS_SYSTEM = "1"
OPENBLAS64_SYSTEM = "1"
```