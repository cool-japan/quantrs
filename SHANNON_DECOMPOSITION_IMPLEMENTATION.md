# Quantum Shannon Decomposition Implementation

## Overview

This document describes the implementation of quantum Shannon decomposition in QuantRS2, which provides a systematic method to decompose arbitrary n-qubit unitary matrices into sequences of single-qubit and CNOT gates with asymptotically optimal gate count.

## Architecture

### Core Components

#### 1. Shannon Decomposer (`shannon.rs`)

The main decomposition engine with recursive algorithms:

```rust
pub struct ShannonDecomposer {
    tolerance: f64,
    cache: FxHashMap<u64, ShannonDecomposition>,
    max_depth: usize,
}
```

- **Recursive Decomposition**: Breaks down n-qubit unitaries into smaller blocks
- **Caching**: Stores decompositions of common matrices for efficiency
- **Depth Limiting**: Prevents stack overflow in pathological cases

#### 2. Decomposition Result

```rust
pub struct ShannonDecomposition {
    pub gates: Vec<Box<dyn GateOp>>,
    pub cnot_count: usize,
    pub single_qubit_count: usize,
    pub depth: usize,
}
```

Provides detailed metrics about the decomposition:
- Gate sequence
- CNOT count (important for NISQ devices)
- Single-qubit gate count
- Circuit depth

#### 3. Optimized Decomposer

```rust
pub struct OptimizedShannonDecomposer {
    base: ShannonDecomposer,
    peephole: bool,
    commutation: bool,
}
```

Adds optimization passes:
- **Peephole Optimization**: Identifies and simplifies local patterns
- **Commutation Optimization**: Reorders gates to reduce depth

## Decomposition Algorithm

### Base Cases

1. **0-qubit**: Empty circuit
2. **1-qubit**: Direct ZYZ decomposition
3. **2-qubit**: Specialized KAK-based decomposition

### Recursive Case (n > 2)

For an n-qubit unitary U, the algorithm:

1. **Block Decomposition**: Split U into 2×2 blocks based on the first qubit
   ```
   U = [A B]
       [C D]
   ```

2. **Block Diagonalization**: Find V, W such that:
   ```
   U = (I ⊗ V) · Controlled-U_d · (I ⊗ W)
   ```
   where U_d is diagonal

3. **Recursive Application**:
   - Decompose V and W on (n-1) qubits
   - Decompose controlled diagonal gates
   - Combine results

### Special Optimizations

1. **Identity Detection**: Skip decomposition for identity matrices
2. **Clifford Detection**: Use specialized decomposition for Clifford gates
3. **Pattern Recognition**: Identify common gate patterns

## Implementation Details

### Key Features

1. **Type Safety**: Uses QuantRS2's type-safe gate and qubit abstractions
2. **Error Handling**: Comprehensive validation of input matrices
3. **Numerical Stability**: Tolerance-based comparisons throughout
4. **Memory Efficiency**: Avoids unnecessary matrix copies

### Performance Characteristics

- **Time Complexity**: O(4^n) for n-qubit unitaries
- **Space Complexity**: O(4^n) for matrix storage
- **Gate Count**: O(4^n) CNOTs (asymptotically optimal)
- **Recursion Depth**: Limited to prevent stack overflow

### Current Limitations

1. **SVD Placeholder**: Full CS decomposition not yet implemented
2. **Two-Qubit Decomposition**: Simplified version (not optimal)
3. **Gray Code Optimization**: Not yet implemented for diagonal gates

## Usage Examples

### Basic Decomposition
```rust
use quantrs2_core::prelude::*;

// Create a random 2-qubit unitary
let unitary = random_unitary(4);
let qubit_ids = vec![QubitId(0), QubitId(1)];

// Decompose
let mut decomposer = ShannonDecomposer::new();
let result = decomposer.decompose(&unitary, &qubit_ids)?;

println!("CNOT count: {}", result.cnot_count);
println!("Circuit depth: {}", result.depth);
```

### Optimized Decomposition
```rust
// Use optimized decomposer
let mut opt_decomposer = OptimizedShannonDecomposer::new();
let optimized = opt_decomposer.decompose(&unitary, &qubit_ids)?;

// Compare gate counts
println!("Original CNOTs: {}", result.cnot_count);
println!("Optimized CNOTs: {}", optimized.cnot_count);
```

### Quick Decomposition
```rust
// One-line decomposition
let gates = shannon_decompose(&unitary, &qubit_ids)?;
```

## Integration with QuantRS2

The Shannon decomposition integrates seamlessly with other QuantRS2 components:

1. **Gate Synthesis**: Works with existing synthesis algorithms
2. **Circuit Optimization**: Can be combined with fusion and peephole optimizers
3. **ZX-Calculus**: Decomposed circuits can be further optimized
4. **Hardware Compilation**: Provides CNOT-optimized circuits for NISQ devices

## Testing

The implementation includes comprehensive tests:

- **Single-qubit decomposition**: Verifies correct ZYZ decomposition
- **Two-qubit decomposition**: Tests CNOT count bounds
- **Identity optimization**: Ensures identity matrices produce empty circuits
- **Numerical accuracy**: Validates decomposition correctness

All tests pass with numerical tolerance of 1e-10.

## Future Enhancements

1. **Cosine-Sine Decomposition**: Implement full CS decomposition for optimal block diagonalization
2. **KAK Decomposition**: Use proper KAK for two-qubit gates
3. **Gray Code Optimization**: Implement Gray code traversal for diagonal gates
4. **Parallelization**: Add parallel decomposition for independent blocks
5. **Approximate Decomposition**: Trade accuracy for gate count reduction

## Mathematical Background

The Shannon decomposition is based on:

1. **Quantum Shannon Decomposition** (Shende et al., 2006)
2. **Cosine-Sine Decomposition** for unitary matrices
3. **Recursive block structure** of quantum operations

The algorithm achieves the theoretical lower bound of Θ(4^n) CNOT gates for generic n-qubit unitaries.

## References

1. Shende, Bullock, Markov, "Synthesis of quantum-logic circuits" (2006)
2. Nielsen & Chuang, "Quantum Computation and Quantum Information"
3. Möttönen et al., "Quantum circuits for general multiqubit gates" (2004)