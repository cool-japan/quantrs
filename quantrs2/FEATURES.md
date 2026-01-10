# QuantRS2 Facade Crate - Feature Documentation

This document provides comprehensive documentation for the QuantRS2 facade crate features.

## Table of Contents

1. [Overview](#overview)
2. [Hierarchical Prelude System](#hierarchical-prelude-system)
3. [Configuration Management](#configuration-management)
4. [System Diagnostics](#system-diagnostics)
5. [Error Handling](#error-handling)
6. [Utility Functions](#utility-functions)
7. [Testing Helpers](#testing-helpers)
8. [Version Management](#version-management)
9. [Examples](#examples)

## Overview

The QuantRS2 facade crate provides a unified entry point to the QuantRS2 quantum computing framework. It offers:

- **Zero-cost abstractions**: Feature-gated re-exports with no runtime overhead
- **Hierarchical preludes**: Import exactly what you need
- **System management**: Configuration, diagnostics, and version checking
- **Developer utilities**: Memory estimation, testing helpers, error handling

## Hierarchical Prelude System

### Prelude Levels

The facade provides 7 levels of preludes, each building on the previous:

```rust
// Level 1: Essentials (always available)
use quantrs2::prelude::essentials::*;

// Level 2: Circuit construction
use quantrs2::prelude::circuits::*;

// Level 3: Quantum simulation
use quantrs2::prelude::simulation::*;

// Level 4: Algorithms and ML
use quantrs2::prelude::algorithms::*;

// Level 5: Hardware integration
use quantrs2::prelude::hardware::*;

// Level 6: Quantum annealing
use quantrs2::prelude::quantum_annealing::*;

// Level 7: Tytan DSL
use quantrs2::prelude::tytan::*;

// Full: Everything available
use quantrs2::prelude::full::*;
```

### Choosing the Right Prelude

| Prelude | Use When | Compile Time | Features Included |
|---------|----------|--------------|-------------------|
| `essentials` | Type definitions only | Fastest | QubitId, Error types, Version |
| `circuits` | Building circuits | Fast | + Circuit, Gates |
| `simulation` | Running simulations | Medium | + Simulators, Backends |
| `algorithms` | VQE, QAOA, QML | Slow | + ML algorithms, Optimization |
| `hardware` | Real devices | Medium | + IBM, Azure, AWS |
| `quantum_annealing` | QUBO/Ising | Medium | + Annealing, D-Wave |
| `tytan` | High-level DSL | Medium | + Tytan API |
| `full` | Everything | Slowest | All features |

## Configuration Management

### Global Configuration

```rust
use quantrs2::config::Config;

// Get global singleton
let cfg = Config::global();

// Configure settings
cfg.set_num_threads(8);
cfg.set_memory_limit_gb(16);
cfg.set_log_level(LogLevel::Info);
cfg.set_default_backend(DefaultBackend::Auto);
cfg.set_gpu_enabled(true);
cfg.set_simd_enabled(true);
```

### Builder Pattern

```rust
use quantrs2::config::Config;

Config::builder()
    .num_threads(8)
    .log_level(LogLevel::Debug)
    .memory_limit_gb(32)
    .default_backend(DefaultBackend::Gpu)
    .enable_gpu(true)
    .enable_simd(true)
    .apply();
```

### Environment Variables

Configuration can also be set via environment variables:

```bash
export QUANTRS2_NUM_THREADS=8
export QUANTRS2_LOG_LEVEL=info
export QUANTRS2_MEMORY_LIMIT_GB=16
export QUANTRS2_BACKEND=gpu
export QUANTRS2_ENABLE_GPU=true
export QUANTRS2_ENABLE_SIMD=true
```

## System Diagnostics

### Running Diagnostics

```rust
use quantrs2::diagnostics;

// Run comprehensive diagnostics
let report = diagnostics::run_diagnostics();

// Check if system is ready
if report.is_ready() {
    println!("System ready for quantum simulation!");
} else {
    for error in report.errors() {
        eprintln!("ERROR: {}", error);
    }
    for warning in report.warnings() {
        eprintln!("WARNING: {}", warning);
    }
}

// Print full report
println!("{}", report);
```

### Diagnostic Information

The diagnostic report includes:

- **Version Information**: QuantRS2, SciRS2, Rust compiler versions
- **System Capabilities**: CPU cores, memory, GPU, SIMD support
- **Configuration**: Current settings
- **Issues**: Errors and warnings with suggestions

### Convenience Functions

```rust
// Quick check
if diagnostics::is_ready() {
    // Proceed with quantum simulation
}

// Print issues to stderr
diagnostics::print_issues();

// Print full report to stdout
diagnostics::print_report();

// Validate and panic if not ready
diagnostics::validate_or_panic();
```

## Error Handling

### Error Categories

All errors are categorized for easier handling:

```rust
use quantrs2::error::{ErrorCategory, QuantRS2ErrorExt};

let error = QuantRS2Error::InvalidQubitId(5);
match error.category() {
    ErrorCategory::Core => { /* core quantum operations */ },
    ErrorCategory::Circuit => { /* circuit validation */ },
    ErrorCategory::Simulation => { /* simulation errors */ },
    ErrorCategory::Hardware => { /* device errors */ },
    ErrorCategory::Algorithm => { /* optimization */ },
    ErrorCategory::Runtime => { /* I/O, network */ },
    _ => { /* other */ },
}
```

### Error Properties

```rust
// Check error properties
if error.is_recoverable() {
    // Retry the operation
}

if error.is_invalid_input() {
    // Validate and fix user input
}

if error.is_resource_error() {
    // Reduce problem size or allocate more resources
}

// Get user-friendly message
println!("{}", error.user_message());
```

### Adding Context

```rust
use quantrs2::error::with_context;

let error = QuantRS2Error::OptimizationFailed("gradient too small".into());
let error = with_context(error, "in QAOA layer 3");
let error = with_context(error, "while solving MaxCut problem");
```

## Utility Functions

### Memory Estimation

```rust
use quantrs2::utils;

// Estimate memory for N qubits
let memory_bytes = utils::estimate_statevector_memory(30);
println!("30 qubits: {}", utils::format_memory(memory_bytes));

// Check if configuration is valid
if utils::is_valid_qubit_count(25, available_memory) {
    println!("25 qubits will fit in available memory");
}

// Find maximum qubits for available memory
let max_qubits = utils::max_qubits_for_memory(16 * 1024 * 1024 * 1024);
println!("Can simulate {} qubits with 16 GB", max_qubits);
```

### Formatting Utilities

```rust
use quantrs2::utils;
use std::time::Duration;

// Format memory
let mem_str = utils::format_memory(1024 * 1024 * 1024); // "1.00 GB"

// Format duration
let dur_str = utils::format_duration(Duration::from_millis(1500)); // "1.5s"
```

### Mathematical Utilities

```rust
use quantrs2::utils;

// Binomial coefficient
let c = utils::binomial(10, 5); // C(10, 5) = 252

// Factorial
let f = utils::factorial(5); // 5! = 120

// Range validation
let valid = utils::is_in_range(&5, &0, &10); // true
```

## Testing Helpers

### Floating-Point Assertions

```rust
use quantrs2::testing;

// Assert approximate equality
testing::assert_approx_eq(1.0, 1.0000001, 1e-6);

// Assert vector equality
let a = vec![1.0, 2.0, 3.0];
let b = vec![1.0000001, 2.0000001, 3.0000001];
testing::assert_vec_approx_eq(&a, &b, 1e-6);
```

### Measurement Assertions

```rust
use quantrs2::testing;
use std::collections::HashMap;

// For stochastic quantum algorithms
let mut actual = HashMap::new();
actual.insert("00".to_string(), 495);
actual.insert("11".to_string(), 505);

let mut expected = HashMap::new();
expected.insert("00".to_string(), 500);
expected.insert("11".to_string(), 500);

// Allow 5% deviation
testing::assert_measurement_counts_close(&actual, &expected, 0.05);
```

### Test Data Generation

```rust
use quantrs2::testing;

// Generate reproducible test data
let seed = testing::test_seed(); // Always 42
let data = testing::generate_random_test_data(100, seed);

// Temporary directories
let temp_dir = testing::create_temp_test_dir();
```

## Version Management

### Version Information

```rust
use quantrs2::version;

// Version constants
println!("QuantRS2: {}", version::QUANTRS2_VERSION);
println!("SciRS2: {}", version::SCIRS2_VERSION);
println!("Rust: {}", version::RUSTC_VERSION);
println!("Build: {}", version::BUILD_TIMESTAMP);

// Detailed version info
let info = version::VersionInfo::current();
println!("{}", info.detailed_version_string());
```

### Compatibility Checking

```rust
use quantrs2::version;

match version::check_compatibility() {
    Ok(()) => println!("All compatibility checks passed!"),
    Err(issues) => {
        for issue in issues {
            eprintln!("Compatibility issue: {}", issue);
        }
    }
}

// Validate and panic if incompatible
version::validate_environment();
```

## Examples

### Running Examples

The facade crate includes 8 comprehensive examples:

```bash
# Basic facade usage
cargo run --example basic_usage

# Configuration management
cargo run --example configuration

# System diagnostics
cargo run --example diagnostics

# Utility functions
cargo run --example utility_functions

# Prelude hierarchy
cargo run --example prelude_hierarchy

# Memory estimation and capacity planning
cargo run --example memory_estimation

# Error handling patterns
cargo run --example error_handling

# Testing helpers
cargo run --example testing_helpers
```

### Example Output

Each example provides detailed, formatted output demonstrating the feature's capabilities. Run them to see:

- Memory requirements for different qubit counts
- System capability detection
- Error handling and recovery patterns
- Configuration options
- Testing utilities in action

## Comprehensive Feature Flag Guide

### Feature Flags Overview

QuantRS2 provides granular control over functionality through Cargo feature flags:

| Feature | Description | Dependencies | Enables |
|---------|-------------|--------------|---------|
| `core` | Core quantum types and traits | *(always enabled)* | Basic quantum primitives |
| `circuit` | Circuit representation and DSL | `core` | Circuit builder, gates, optimization |
| `sim` | Quantum simulators | `circuit` | State-vector, stabilizer, tensor network |
| `device` | Hardware backends | `circuit` | IBM, Azure, AWS quantum computers |
| `anneal` | Quantum annealing | `circuit` | QUBO, Ising models, simulated annealing |
| `ml` | Quantum machine learning | `sim`, `anneal` | VQE, QAOA, QNNs, QGANs |
| `tytan` | Tytan high-level API | `anneal` | DSL for optimization problems |
| `symengine` | Symbolic computation | `core` | Parametric gates, symbolic optimization |
| `full` | All features | *all above* | Complete QuantRS2 functionality |

### Feature Dependencies Diagram

```
core (always enabled)
  ├─ circuit
  │   ├─ sim
  │   │   └─ ml (also requires anneal)
  │   ├─ device
  │   └─ anneal
  │       ├─ ml (also requires sim)
  │       └─ tytan
  └─ symengine
```

### Feature Selection Decision Tree

```
START: What do you need to do?

├─ Circuit construction only
│  └─ features = ["circuit"]
│     Compilation: ~8s
│     Use Case: Building and exporting circuits
│
├─ Quantum simulation
│  └─ features = ["sim"]  # auto-enables circuit
│     Compilation: ~15s
│     Use Case: Algorithm research, prototyping
│
├─ Quantum machine learning
│  └─ features = ["ml"]   # auto-enables sim, anneal, circuit
│     Compilation: ~30s
│     Use Case: VQE, QAOA, QNN training
│
├─ Quantum annealing optimization
│  └─ features = ["tytan"]  # auto-enables anneal, circuit
│     Compilation: ~12s
│     Use Case: QUBO, TSP, portfolio optimization
│
├─ Real quantum hardware
│  └─ features = ["device", "sim"]
│     Compilation: ~20s
│     Use Case: IBM/Azure/AWS quantum computers
│
├─ Symbolic computation
│  └─ features = ["circuit", "symengine"]
│     Compilation: ~10s
│     Use Case: Parametric gates, symbolic derivatives
│
└─ Production application
   └─ features = ["full"]
      Compilation: ~45s
      Use Case: Complete quantum computing platform
```

### Feature Combinations for Common Scenarios

#### 1. Research & Prototyping

```toml
# Minimal for quick iteration
[dependencies]
quantrs2 = { version = "0.1.0-rc.2", features = ["circuit"] }
```

**When to use:**
- Circuit design and visualization
- Quick algorithm sketching
- Educational purposes
- Fast compilation required

**Available APIs:**
- Circuit builder
- Gate operations
- Circuit optimization
- QASM export/import

#### 2. Algorithm Development & Benchmarking

```toml
# Simulation for testing
[dependencies]
quantrs2 = { version = "0.1.0-rc.2", features = ["sim"] }
```

**When to use:**
- Algorithm development
- Performance benchmarking
- Noise analysis
- State vector simulation

**Available APIs:**
- All circuit APIs
- State-vector simulator
- Stabilizer simulator
- Tensor network simulator
- Noise models

#### 3. Quantum Machine Learning

```toml
# Full ML stack
[dependencies]
quantrs2 = { version = "0.1.0-rc.2", features = ["ml"] }
```

**When to use:**
- VQE for chemistry
- QAOA for optimization
- Quantum neural networks
- Hybrid quantum-classical ML

**Available APIs:**
- All simulation APIs
- VQE algorithms
- QAOA optimizers
- Quantum neural networks
- Autodiff support (via SciRS2)

#### 4. Optimization Problems

```toml
# High-level annealing
[dependencies]
quantrs2 = { version = "0.1.0-rc.2", features = ["tytan"] }
```

**When to use:**
- QUBO formulations
- Ising model problems
- Combinatorial optimization
- Portfolio optimization

**Available APIs:**
- Tytan DSL
- QUBO/Ising converters
- Simulated annealing
- GPU-accelerated solvers
- D-Wave integration (when available)

#### 5. Hardware Integration

```toml
# Real quantum devices
[dependencies]
quantrs2 = { version = "0.1.0-rc.2", features = ["device", "sim"] }
```

**When to use:**
- Running on IBM Quantum
- Azure Quantum integration
- AWS Braket execution
- Hardware noise characterization

**Available APIs:**
- All simulation APIs (for testing)
- IBM backend
- Azure backend
- AWS backend
- Automatic transpilation
- Error mitigation

#### 6. Production Deployment

```toml
# Everything
[dependencies]
quantrs2 = { version = "0.1.0-rc.2", features = ["full"] }
```

**When to use:**
- Production quantum services
- Multi-tenant platforms
- Quantum cloud APIs
- Complete quantum SDK

**Available APIs:**
- All QuantRS2 features
- Comprehensive diagnostics
- Production error handling
- Performance monitoring

### Feature Flag Performance Implications

| Feature Set | Compilation | Binary Size | Runtime Overhead | Memory Overhead |
|-------------|-------------|-------------|------------------|-----------------|
| `circuit` | ~8s | ~2 MB | 0% | Minimal |
| `sim` | ~15s | ~5 MB | 0% | State-dependent |
| `ml` | ~30s | ~12 MB | 0% | Optimizer-dependent |
| `tytan` | ~12s | ~4 MB | 0% | Problem-dependent |
| `device` | ~10s | ~3 MB | Network I/O | Minimal |
| `full` | ~45s | ~20 MB | 0% | Feature-dependent |

**Note**: All overhead values are **0%** because QuantRS2 uses zero-cost abstractions. Unused features are optimized out at compile time.

### Advanced Feature Configurations

#### Minimal Build for CI/CD

```toml
# Fastest compilation for testing
[dependencies]
quantrs2 = { version = "0.1.0-rc.2", default-features = false }
```

**Compilation time**: ~3s
**Use case**: Type checking, quick validation

#### Selective Feature Enabling

```toml
# Only what you need
[dependencies]
quantrs2 = { version = "0.1.0-rc.2", features = ["circuit", "symengine"] }
```

**Use case**: Parametric circuit optimization with symbolic derivatives

#### Platform-Specific Features (Future)

```toml
# Example of conditional features (not yet implemented)
[target.'cfg(target_arch = "x86_64")'.dependencies]
quantrs2 = { version = "0.1.0-rc.2", features = ["sim", "ml"] }

[target.'cfg(target_os = "macos")'.dependencies]
quantrs2 = { version = "0.1.0-rc.2", features = ["device"] }
```

### Feature Flag Best Practices

#### ✅ DO:

```toml
# Enable features you need
quantrs2 = { features = ["sim"] }

# Use hierarchical approach
# Level 1: circuit
# Level 2: Add sim
# Level 3: Add ml if needed
```

```rust
// Use appropriate prelude for features
#[cfg(feature = "sim")]
use quantrs2::prelude::simulation::*;

#[cfg(feature = "ml")]
use quantrs2::prelude::algorithms::*;
```

#### ❌ DON'T:

```toml
# Don't enable full if you only need circuit
quantrs2 = { features = ["full"] }  # 45s compile for 8s functionality

# Don't manually specify transitive dependencies
quantrs2 = { features = ["circuit", "sim"] }  # sim already enables circuit
```

```rust
// Don't use full prelude unnecessarily
use quantrs2::prelude::full::*;  // Brings in everything

// Don't forget feature gates in library code
pub fn requires_ml() {
    // Will fail to compile without ml feature!
    use quantrs2::ml::VQE;
}
```

### Feature Testing Strategy

Test your crate with different feature combinations:

```bash
# Test minimal features
cargo test --no-default-features

# Test individual features
cargo test --features circuit
cargo test --features sim
cargo test --features ml

# Test feature combinations
cargo test --features "sim,device"
cargo test --features "ml,symengine"

# Test everything
cargo test --features full
```

### SciRS2 Feature Integration

All features leverage SciRS2 for scientific computing:

```rust
// Unified SciRS2 usage across features
use scirs2_core::{Complex64, Complex32};  // Complex numbers
use scirs2_core::ndarray::*;              // Arrays
use scirs2_core::random::prelude::*;      // RNG

// Feature-specific SciRS2 integration
#[cfg(feature = "sim")]
use scirs2_linalg::*;                     // Linear algebra for simulation

#[cfg(feature = "ml")]
use scirs2_autograd::*;                   // Autodiff for VQE/QAOA

#[cfg(feature = "anneal")]
use scirs2_optimize::*;                   // Optimization for annealing
```

## Best Practices

### 1. Choose Appropriate Prelude

```rust
// ✓ GOOD: Use specific prelude for your needs
use quantrs2::prelude::simulation::*;

// ✗ AVOID: Using full prelude when not needed
use quantrs2::prelude::full::*; // Slower compilation
```

### 2. Validate System Before Heavy Computation

```rust
use quantrs2::diagnostics;

// Check system readiness
if !diagnostics::is_ready() {
    diagnostics::print_issues();
    return Err("System not ready".into());
}

// Proceed with quantum computation
```

### 3. Configure for Your Workload

```rust
use quantrs2::config::Config;

let cfg = Config::global();

// For memory-intensive workloads
cfg.set_memory_limit_gb(32);
cfg.set_num_threads(8);

// For GPU acceleration
cfg.set_gpu_enabled(true);
cfg.set_default_backend(DefaultBackend::Gpu);
```

### 4. Handle Errors with Context

```rust
use quantrs2::error::with_context;

fn my_operation() -> QuantRS2Result<()> {
    some_operation()
        .map_err(|e| with_context(e, "in my_operation"))?;
    Ok(())
}
```

### 5. Use Testing Helpers for Quantum Tests

```rust
#[cfg(test)]
mod tests {
    use quantrs2::testing;

    #[test]
    fn test_vqe_convergence() {
        let expected = -1.137;
        let result = run_vqe();
        testing::assert_approx_eq(expected, result, 1e-5);
    }
}
```

## Performance Considerations

### Compilation Time

Preludes are ordered by compilation time:
- `essentials`: < 1 second
- `circuits`: ~2-3 seconds
- `simulation`: ~5-10 seconds
- `algorithms`: ~15-30 seconds
- `full`: ~30-60 seconds

### Memory Overhead

The facade crate adds **zero runtime overhead**. All re-exports are inlined.

### Feature Selection

Enable only the features you need in `Cargo.toml`:

```toml
[dependencies]
quantrs2 = { version = "0.1.0-rc.2", features = ["circuit", "sim"] }
```

## Troubleshooting

### Common Issues

1. **Compilation errors with features**
   - Ensure feature dependencies are met (e.g., `sim` requires `circuit`)
   - Check `version::check_compatibility()` for issues

2. **Memory errors during simulation**
   - Use `utils::max_qubits_for_memory()` to estimate capacity
   - Consider tensor network or stabilizer simulation for larger systems

3. **Performance issues**
   - Enable SIMD: `cfg.set_simd_enabled(true)`
   - Use GPU backend: `cfg.set_default_backend(DefaultBackend::Gpu)`
   - Check `diagnostics::run_diagnostics()` for system issues

### Getting Help

- Review examples in `examples/` directory
- Check CLAUDE.md for development guidelines
- See SCIRS2_INTEGRATION_POLICY.md for SciRS2 usage patterns

## Version: 0.1.0-rc.2

Last updated: 2025-11-18
