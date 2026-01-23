# quantrs2-symengine-pure

Pure Rust symbolic mathematics library for quantum computing.

[![Crates.io](https://img.shields.io/crates/v/quantrs2-symengine-pure.svg)](https://crates.io/crates/quantrs2-symengine-pure)
[![Documentation](https://docs.rs/quantrs2-symengine-pure/badge.svg)](https://docs.rs/quantrs2-symengine-pure)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

## Overview

`quantrs2-symengine-pure` provides symbolic computation capabilities for the QuantRS2 quantum computing framework. Unlike C++-based alternatives, this crate is implemented entirely in Rust, ensuring portability and seamless integration with the Rust ecosystem.

This crate uses [egg](https://egraphs-good.github.io/) (e-graphs good) for advanced expression simplification via equality saturation.

## Features

- **Pure Rust**: No C/C++ dependencies, fully portable across all platforms
- **Symbolic Expressions**: Create and manipulate symbolic mathematical expressions
- **Automatic Differentiation**: Compute symbolic gradients and Hessians
- **E-Graph Optimization**: Advanced expression simplification via equality saturation
- **Quantum Computing**: Specialized support for quantum gates, operators, and states
- **SciRS2 Integration**: Seamless integration with the SciRS2 scientific computing ecosystem
- **Arbitrary Precision**: Support for rational and big integer arithmetic

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
quantrs2-symengine-pure = "0.1.2"
```

## Quick Start

```rust
use quantrs2_symengine_pure::Expression;

// Create symbolic expressions
let x = Expression::symbol("x");
let y = Expression::symbol("y");

// Perform operations
let expr = x.clone() * x.clone() + x.clone() * 2.0 * y.clone() + y.clone() * y.clone();
let expanded = expr.expand();

// Compute derivatives
let dx = expr.diff(&x);

println!("Expression: {}", expr);
println!("Derivative wrt x: {}", dx);
```

## Modules

| Module | Description |
|--------|-------------|
| `expr` | Core symbolic expression types |
| `diff` | Automatic differentiation |
| `eval` | Expression evaluation |
| `simplify` | Expression simplification |
| `optimization` | E-graph based optimization |
| `parser` | Parse strings to expressions |
| `matrix` | Symbolic matrix operations |
| `quantum` | Quantum-specific symbolic types |
| `cache` | Expression caching |
| `serialize` | Serialization support |

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `simd` | Yes | SIMD acceleration via scirs2-core |
| `parallel` | Yes | Parallel operations via scirs2-core |
| `serde` | No | Serialization support |

## Policy Compliance

This crate follows QuantRS2/COOLJAPAN policies:

- **Pure Rust Policy**: No C/C++/Fortran dependencies
- **SciRS2 Policy**: Uses `scirs2-core` for complex numbers, arrays, and random generation
- **COOLJAPAN Policy**: Uses `oxicode` for serialization (not bincode)
- **No unwrap Policy**: All fallible operations return Result types

## Part of QuantRS2

This crate is part of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework.

## License

Licensed under either of:

- MIT License
- Apache License, Version 2.0

at your option.

## Author

COOLJAPAN OU (Team Kitasan)
