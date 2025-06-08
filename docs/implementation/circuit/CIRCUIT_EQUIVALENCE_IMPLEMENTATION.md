# Circuit Equivalence Checking Implementation

## Overview

Successfully implemented comprehensive circuit equivalence checking algorithms for QuantRS2, enabling verification that different quantum circuits produce equivalent results. This is crucial for circuit optimization, compilation, and verification workflows.

## Implementation Details

### 1. Core Equivalence Module (`circuit/src/equivalence.rs`)

Provides multiple methods for checking circuit equivalence:

```rust
pub enum EquivalenceType {
    UnitaryEquivalence,      // Exact unitary matrix comparison
    StateVectorEquivalence,  // Output state comparison for all inputs
    ProbabilisticEquivalence,// Measurement probability comparison
    StructuralEquivalence,   // Gate-by-gate structural comparison
    GlobalPhaseEquivalence,  // Equivalence up to global phase
}
```

### 2. Equivalence Checker

Main API for circuit comparison:

```rust
pub struct EquivalenceChecker {
    options: EquivalenceOptions,
}

pub struct EquivalenceOptions {
    pub tolerance: f64,              // Numerical tolerance
    pub ignore_global_phase: bool,   // Ignore global phase differences
    pub check_all_states: bool,      // Check all basis states
    pub max_unitary_qubits: usize,   // Max size for unitary construction
}
```

### 3. Equivalence Checking Methods

#### Structural Equivalence
- Fastest check - O(n) where n is number of gates
- Verifies identical gate sequences
- Useful for debugging and exact matching

```rust
pub fn check_structural_equivalence<const N: usize>(
    &self,
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> QuantRS2Result<EquivalenceResult>
```

#### Unitary Equivalence
- Constructs full unitary matrices
- Exact mathematical equivalence
- Limited to small circuits (≤10 qubits)
- Can check with or without global phase

```rust
pub fn check_unitary_equivalence<const N: usize>(
    &self,
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> QuantRS2Result<EquivalenceResult>
```

#### State Vector Equivalence
- Checks output states for all/subset of input states
- More scalable than unitary for medium circuits
- Supports approximate equivalence

```rust
pub fn check_state_vector_equivalence<const N: usize>(
    &self,
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> QuantRS2Result<EquivalenceResult>
```

#### Probabilistic Equivalence
- Compares measurement probabilities
- Useful for NISQ circuits
- Naturally handles global phase

```rust
pub fn check_probabilistic_equivalence<const N: usize>(
    &self,
    circuit1: &Circuit<N>,
    circuit2: &Circuit<N>,
) -> QuantRS2Result<EquivalenceResult>
```

### 4. Enhanced Simulator Integration (`circuit/src/equivalence_sim.rs`)

Provides actual simulation-based equivalence checking:

```rust
pub struct SimulatorEquivalenceChecker {
    options: EquivalenceOptions,
}
```

Features:
- Integration with StateVectorSimulator
- Actual quantum state evolution
- Measurement probability calculation
- Support for all standard gates

## Usage Examples

### Basic Structural Check

```rust
use quantrs2_circuit::prelude::*;

let mut circuit1 = Circuit::<2>::new();
circuit1.h(0)?;
circuit1.cnot(0, 1)?;

let mut circuit2 = Circuit::<2>::new();
circuit2.h(0)?;
circuit2.cnot(0, 1)?;

// Quick structural check
assert!(circuits_structurally_equal(&circuit1, &circuit2));
```

### Comprehensive Equivalence Check

```rust
let checker = EquivalenceChecker::new(EquivalenceOptions {
    tolerance: 1e-10,
    ignore_global_phase: true,
    check_all_states: true,
    max_unitary_qubits: 10,
});

let result = checker.check_equivalence(&circuit1, &circuit2)?;
println!("Equivalent: {}", result.equivalent);
println!("Check type: {:?}", result.check_type);
println!("Max difference: {:?}", result.max_difference);
```

### Verifying Optimization Correctness

```rust
// Original circuit
let mut original = Circuit::<3>::new();
// ... add gates ...

// Optimize
let optimized = original.optimize()?;

// Verify optimization preserved behavior
assert!(circuits_equivalent(&original, &optimized)?);
```

### Custom Tolerance for Approximate Equivalence

```rust
let noisy_checker = EquivalenceChecker::new(EquivalenceOptions {
    tolerance: 1e-6,  // Relaxed tolerance for noisy circuits
    ignore_global_phase: true,
    check_all_states: false,  // Sample subset for large circuits
    max_unitary_qubits: 8,
});
```

## Supported Equivalence Types

### 1. Exact Equivalences
- **Structural**: Identical gate sequences
- **Unitary**: Same unitary matrix
- **State Vector**: Same output for all inputs

### 2. Approximate Equivalences
- **Numerical Tolerance**: Within specified epsilon
- **Probabilistic**: Same measurement distributions
- **Sampling**: Check subset of basis states

### 3. Phase Equivalences
- **Global Phase**: Differ only by e^(iθ)
- **Local Phase**: Equivalent up to single-qubit phases

## Performance Considerations

1. **Structural Check**: O(n) time, O(1) space
2. **Unitary Check**: O(4^n) time and space
3. **State Vector Check**: O(n·2^n) time, O(2^n) space
4. **Probabilistic Check**: O(n·2^n) time, O(2^n) space

Recommendations:
- Use structural check for identical circuits
- Use unitary for small circuits (≤10 qubits)
- Use state vector for medium circuits (≤20 qubits)
- Use probabilistic sampling for large circuits

## Integration with Circuit Optimization

The equivalence checker integrates with:

1. **Optimization Verification**
   ```rust
   let optimized = circuit.optimize()?;
   assert!(verify_optimization_correctness(&circuit, &optimized)?);
   ```

2. **Compilation Validation**
   ```rust
   let compiled = transpiler.compile(&circuit)?;
   let result = checker.check_equivalence(&circuit, &compiled)?;
   ```

3. **Template Matching**
   ```rust
   if circuits_equivalent(&subcircuit, &template)? {
       // Apply template replacement
   }
   ```

## Future Enhancements

1. **Process Tomography**: Full quantum process comparison
2. **Randomized Benchmarking**: Statistical equivalence verification
3. **Hardware Noise Models**: Equivalence under realistic noise
4. **Symbolic Verification**: Formal methods for equivalence
5. **Clifford Circuits**: Specialized fast equivalence for stabilizer circuits
6. **Parallel Checking**: Multi-threaded state comparisons

## Testing

Comprehensive test suite includes:

1. **Unit Tests**: Each equivalence method
2. **Integration Tests**: With circuit builder and optimizer
3. **Property Tests**: Equivalence relations (reflexive, symmetric, transitive)
4. **Benchmark Tests**: Performance characteristics
5. **Edge Cases**: Empty circuits, single gates, large circuits

## Design Decisions

1. **Multiple Methods**: Different equivalence checks for different use cases
2. **Configurable Options**: Flexible tolerance and checking strategies
3. **Result Details**: Rich information about why circuits differ
4. **Performance Tiers**: From fast structural to comprehensive unitary
5. **Simulator Integration**: Optional enhanced checking with actual simulation

## Conclusion

This implementation provides QuantRS2 with robust circuit verification capabilities, essential for ensuring correctness in quantum circuit transformations and optimizations. The multiple equivalence checking strategies allow users to choose the appropriate method based on their specific requirements and constraints.