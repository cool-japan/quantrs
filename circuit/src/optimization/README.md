# Circuit Optimization Module

This module provides a comprehensive framework for optimizing quantum circuits using gate properties, cost models, and various optimization passes.

## Features

### Gate Properties System
- **Comprehensive gate information**: cost, error rates, decomposition rules, commutation relations
- **Hardware-aware properties**: native gate sets for different quantum backends
- **Extensible framework**: easy to add new gate types and properties

### Optimization Passes

1. **Gate Cancellation**: Removes redundant gates (e.g., X·X = I, H·H = I)
2. **Gate Commutation**: Reorders gates to enable other optimizations
3. **Gate Merging**: Combines adjacent compatible gates
4. **Rotation Merging**: Specifically merges rotation gates (RX, RY, RZ)
5. **Decomposition Optimization**: Chooses optimal gate decompositions for target hardware
6. **Cost-Based Optimization**: Minimizes various metrics (gate count, depth, error, time)
7. **Two-Qubit Optimization**: Specialized optimizations for two-qubit gates
8. **Template Matching**: Replaces known patterns with more efficient equivalents
9. **Circuit Rewriting**: Uses equivalence rules to transform circuits

### Cost Models

- **Abstract Cost Model**: Hardware-agnostic optimization
- **Hardware Cost Models**: Backend-specific optimization for IBM, Google, AWS
- **Customizable Weights**: Balance between gate count, execution time, error rate, and depth

### Pass Manager

- **Optimization Levels**: None, Light, Medium, Heavy, Custom
- **Hardware Presets**: Optimized pass sequences for specific backends
- **Configurable Pipeline**: Add, remove, and reorder passes
- **Iterative Optimization**: Applies passes until convergence

### Circuit Analysis

- **Comprehensive Metrics**: Gate count, depth, two-qubit gates, execution time, error
- **Improvement Tracking**: Before/after comparison with percentage improvements
- **Detailed Reports**: Gate type breakdown, parallelism analysis, critical path

## Usage Examples

### Basic Optimization
```rust
use quantrs2_circuit::prelude::*;

let circuit = Circuit::<4>::new();
// ... build circuit ...

let mut optimizer = CircuitOptimizer2::with_level(OptimizationLevel::Medium);
let report = optimizer.optimize(&circuit)?;
report.print_summary();
```

### Hardware-Specific Optimization
```rust
let mut optimizer = CircuitOptimizer2::for_hardware("ibm");
let report = optimizer.optimize(&circuit)?;
```

### Custom Pipeline
```rust
let mut optimizer = CircuitOptimizer2::new();
optimizer.add_pass(Box::new(GateCancellation::new(true)));
optimizer.add_pass(Box::new(RotationMerging::new(1e-10)));
optimizer.add_pass(Box::new(TemplateMatching::new()));

let report = optimizer.optimize(&circuit)?;
```

### Gate Properties
```rust
let props = GateProperties::single_qubit("H");
println!("Hadamard gate duration: {} ns", props.cost.duration_ns);
println!("Is native: {}", props.is_native);
println!("Error rate: {}", props.error.error_rate);
```

## Architecture

The optimization system is designed to be:
- **Modular**: Each pass is independent and can be used standalone
- **Extensible**: Easy to add new passes, cost models, and gate properties
- **Efficient**: Optimizations are applied iteratively with early termination
- **Flexible**: Support for both general and hardware-specific optimization

## Future Enhancements

- [ ] Circuit introspection API integration
- [ ] More sophisticated template patterns
- [ ] Machine learning-based optimization
- [ ] Parallel pass execution
- [ ] Incremental optimization
- [ ] Visual optimization reports