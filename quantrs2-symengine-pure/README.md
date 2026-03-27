# quantrs2-symengine-pure

Pure Rust symbolic mathematics library for quantum computing. 333 public APIs, 0 stubs.

[![Crates.io](https://img.shields.io/crates/v/quantrs2-symengine-pure.svg)](https://crates.io/crates/quantrs2-symengine-pure)
[![Documentation](https://docs.rs/quantrs2-symengine-pure/badge.svg)](https://docs.rs/quantrs2-symengine-pure)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

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
quantrs2-symengine-pure = "0.1.3"
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
| `cache` | Expression caching |
| `diff` | Automatic differentiation |
| `eval` | Expression evaluation |
| `expr` | Core symbolic expression types |
| `matrix` | Symbolic matrix operations |
| `ops` | Arithmetic and algebraic operations |
| `optimization` | E-graph based optimization (via egg) |
| `parser` | Parse strings to expressions |
| `pattern` | Pattern matching on expressions |
| `quantum` | Quantum-specific symbolic types |
| `scirs2_bridge` | SciRS2 ecosystem integration |
| `serialize` | Serialization support |
| `simplify` | Expression simplification |

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

Licensed under the Apache License, Version 2.0.

## Author

COOLJAPAN OU (Team Kitasan)
