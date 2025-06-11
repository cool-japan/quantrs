# Variational Quantum Gates Implementation

## Overview

This document describes the implementation of variational quantum gates with automatic differentiation support in QuantRS2. This enables gradient-based optimization of quantum circuits for variational quantum algorithms like VQE, QAOA, and quantum machine learning.

## Architecture

### Core Components

#### 1. Variational Gate (`variational.rs`)

```rust
pub struct VariationalGate {
    pub name: String,
    pub qubits: Vec<QubitId>,
    pub params: Vec<String>,
    pub values: Vec<f64>,
    pub generator: Arc<dyn Fn(&[f64]) -> Array2<Complex<f64>> + Send + Sync>,
    pub diff_mode: DiffMode,
}
```

Key features:
- Parameter-aware quantum gates
- Multiple differentiation modes
- GateOp trait implementation
- Thread-safe gate generator functions

#### 2. Differentiation Modes

```rust
pub enum DiffMode {
    Forward,                    // Forward-mode automatic differentiation
    Reverse,                   // Reverse-mode (backpropagation)
    ParameterShift,           // Parameter shift rule (default)
    FiniteDiff { epsilon: f64 }, // Finite differences
}
```

The parameter shift rule is the default as it's exact for quantum circuits.

#### 3. Dual Numbers (Forward-mode)

```rust
pub struct Dual {
    pub real: f64,  // Value
    pub dual: f64,  // Derivative
}
```

Supports arithmetic operations and common functions (sin, cos, exp, sqrt).

#### 4. Computation Graph (Reverse-mode)

```rust
pub struct ComputationGraph {
    nodes: Vec<Node>,
    params: FxHashMap<String, usize>,
    next_id: usize,
}
```

Builds a DAG for automatic differentiation with backward pass support.

## Gradient Computation

### Parameter Shift Rule

The parameter shift rule is exact for quantum circuits:

```
∂f/∂θ = [f(θ + π/2) - f(θ - π/2)] / 2
```

This is implemented as:

```rust
fn parameter_shift_gradient(&self, loss_fn: impl Fn(&Array2<Complex<f64>>) -> f64) 
    -> QuantRS2Result<Vec<f64>>
```

### Finite Differences

Approximates gradients numerically:

```
∂f/∂θ ≈ [f(θ + ε) - f(θ)] / ε
```

## Variational Circuits

### Circuit Structure

```rust
pub struct VariationalCircuit {
    pub gates: Vec<VariationalGate>,
    pub param_map: FxHashMap<String, Vec<usize>>,
    pub num_qubits: usize,
}
```

Features:
- Parameter sharing across gates
- Batch gradient computation
- Parameter name tracking

### Optimization

```rust
pub struct VariationalOptimizer {
    pub learning_rate: f64,
    pub momentum: f64,
    velocities: FxHashMap<String, f64>,
}
```

Supports gradient descent with momentum.

## Usage Examples

### Creating Variational Gates

```rust
use quantrs2_core::prelude::*;

// Single-qubit rotation
let rx_gate = VariationalGate::rx(QubitId(0), "theta".to_string(), 0.5);

// Controlled rotation
let cry_gate = VariationalGate::cry(
    QubitId(0), 
    QubitId(1), 
    "phi".to_string(), 
    1.0
);
```

### Building Circuits

```rust
let mut circuit = VariationalCircuit::new(2);

circuit.add_gate(VariationalGate::rx(QubitId(0), "theta1".to_string(), 0.1));
circuit.add_gate(VariationalGate::ry(QubitId(1), "theta2".to_string(), 0.2));
circuit.add_gate(VariationalGate::cry(QubitId(0), QubitId(1), "theta3".to_string(), 0.3));
```

### Computing Gradients

```rust
// Define loss function
let loss_fn = |gates: &[VariationalGate]| -> f64 {
    // Compute expectation value or other metric
    let expectation = compute_expectation(gates);
    expectation
};

// Compute gradients for all parameters
let gradients = circuit.compute_gradients(loss_fn)?;

// Optimize parameters
let mut optimizer = VariationalOptimizer::new(0.1, 0.9);
optimizer.step(&mut circuit, &gradients)?;
```

### Custom Variational Gates

```rust
// Create custom variational gate
let generator = Arc::new(|params: &[f64]| {
    let alpha = params[0];
    let beta = params[1];
    
    // Build matrix based on parameters
    Array2::from_shape_vec((2, 2), vec![
        Complex::new(alpha.cos(), 0.0), Complex::new(0.0, -beta.sin()),
        Complex::new(0.0, beta.sin()), Complex::new(alpha.cos(), 0.0),
    ]).unwrap()
});

let custom_gate = VariationalGate {
    name: "CustomGate".to_string(),
    qubits: vec![QubitId(0)],
    params: vec!["alpha".to_string(), "beta".to_string()],
    values: vec![0.5, 1.0],
    generator,
    diff_mode: DiffMode::ParameterShift,
};
```

## Implementation Details

### Gate Generation

Gates are generated dynamically from parameters:
- RX(θ): Rotation around X-axis
- RY(θ): Rotation around Y-axis  
- RZ(θ): Rotation around Z-axis
- CRY(θ): Controlled Y rotation

### Gradient Validation

The implementation includes comprehensive tests:
- Dual number arithmetic
- Parameter shift rule correctness
- Circuit parameter updates
- Optimizer convergence

### Performance Considerations

1. **Lazy String Allocation**: Gate names use leaked strings for 'static lifetime
2. **Arc for Generators**: Thread-safe sharing of gate functions
3. **FxHashMap**: Fast hashing for parameter lookups
4. **Cloning for Borrow Checker**: Strategic cloning in computation graph

## Future Enhancements

1. **Higher-Order Derivatives**: Hessian computation
2. **Natural Gradient**: Fisher information matrix
3. **Quantum-Aware Optimizers**: QNG, SPSA
4. **Gate Compilation**: Compile variational gates to fixed gates
5. **Noise-Aware Gradients**: Gradient computation with noise
6. **Symbolic Differentiation**: Full symbolic autodiff

## Testing

Comprehensive test suite covers:
- Dual number operations
- Gradient computation accuracy
- Circuit parameter management
- Optimization steps
- Gate unitarity preservation

All 105 core module tests pass successfully.

## Code Quality

- Well-documented with inline comments
- Type-safe parameter management
- Clean error handling
- Modular design for extensions
- Integration with GateOp trait

## References

1. Schuld et al., "Evaluating analytic gradients on quantum hardware"
2. Mitarai et al., "Quantum circuit learning"
3. McClean et al., "The theory of variational hybrid quantum-classical algorithms"
4. Bergholm et al., "PennyLane: Automatic differentiation of quantum programs"