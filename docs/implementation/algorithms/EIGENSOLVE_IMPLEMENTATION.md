# Gate Characterization with Eigenvalue Decomposition

## Overview

This document describes the implementation of gate characterization using eigenvalue decomposition in QuantRS2. The implementation provides comprehensive tools for analyzing quantum gates through their eigenstructure, which is crucial for gate synthesis, optimization, and verification.

## Key Components

### 1. Eigenvalue Decomposition Module (`eigensolve.rs`)

The eigensolve module provides optimized eigenvalue decomposition specifically for unitary matrices (quantum gates):

- **QR Algorithm with Shifts**: Implements the QR algorithm with Wilkinson shifts for efficient convergence
- **Hessenberg Reduction**: Uses Householder reflections to reduce matrices to Hessenberg form
- **Givens Rotations**: Employs Givens rotations for numerically stable QR decomposition
- **Inverse Iteration**: Refines eigenvectors using inverse iteration for improved accuracy
- **Analytical Solutions**: Provides exact solutions for 1×1 and 2×2 matrices

Key features:
```rust
pub fn eigen_decompose_unitary(
    matrix: &Array2<Complex>,
    tolerance: f64,
) -> QuantRS2Result<EigenDecomposition>
```

### 2. Gate Characterization Module (`characterization.rs`)

The characterization module leverages eigenvalue decomposition to analyze quantum gates:

#### Gate Eigenstructure
```rust
pub struct GateEigenstructure {
    pub eigenvalues: Vec<Complex>,
    pub eigenvectors: Array2<Complex>,
    pub matrix: Array2<Complex>,
}
```

#### Gate Identification
- **Single-qubit gates**: Identifies Pauli gates, Hadamard, rotations, and phase gates
- **Two-qubit gates**: Recognizes CNOT, controlled phase, SWAP, and variants
- **General gates**: Classifies arbitrary n-qubit gates

#### Analysis Tools
- **Rotation extraction**: Determines rotation angle and axis for single-qubit gates
- **Phase analysis**: Extracts eigenphases and global phase
- **Gate distance**: Computes Frobenius norm distance between gates
- **Clifford approximation**: Finds the closest Clifford gate to a given gate
- **Decomposition**: Breaks down gates into elementary rotations

### 3. Integration with QuantRS2

The implementation integrates seamlessly with the existing QuantRS2 infrastructure:

- Works with the `GateOp` trait for any quantum gate
- Uses `ndarray` for efficient matrix operations
- Provides `QuantRS2Result` error handling
- Exports through the prelude for easy access

## Technical Details

### Eigenvalue Algorithm

1. **Hessenberg Reduction**: Reduces computational complexity from O(n³) to O(n²) per iteration
2. **QR Iteration**: 
   - Uses Wilkinson shift for quadratic convergence near eigenvalues
   - Employs Givens rotations for numerical stability
3. **Eigenvector Refinement**: Uses inverse iteration to improve eigenvector accuracy

### Gate Type Identification

The system identifies gates based on their eigenvalue patterns:

- **Pauli gates**: Eigenvalues ±1 or ±i
- **Rotation gates**: Eigenvalues e^(±iθ/2)
- **Phase gates**: All eigenvalues have same magnitude
- **CNOT**: All eigenvalues ±1 (4×4 matrix)
- **SWAP**: Eigenvalues {1, 1, 1, -1}

### Global Phase Calculation

For a unitary matrix U = e^(iφ)V where V is special unitary:
```
φ = arg(det(U)) / n
```
where n is the matrix dimension.

## Usage Examples

### Basic Gate Analysis
```rust
use quantrs2_core::prelude::*;

let characterizer = GateCharacterizer::new(1e-10);
let gate = PauliX { target: QubitId(0) };

// Get eigenstructure
let eigen = characterizer.eigenstructure(&gate)?;
println!("Eigenvalues: {:?}", eigen.eigenvalues);

// Identify gate type
let gate_type = characterizer.identify_gate_type(&gate)?;
assert_eq!(gate_type, GateType::PauliX);
```

### Rotation Analysis
```rust
let rx = RotationX { target: QubitId(0), theta: PI / 4.0 };
let eigen = characterizer.eigenstructure(&rx)?;

// Extract rotation angle
if let Some(angle) = eigen.rotation_angle() {
    println!("Rotation angle: {}", angle);
}

// Decompose into elementary rotations
let decomposition = characterizer.decompose_to_rotations(&rx)?;
```

### Gate Approximation
```rust
// Find closest Clifford gate
let arbitrary_gate = RotationZ { target: QubitId(0), theta: PI / 8.0 };
let closest_clifford = characterizer.find_closest_clifford(&arbitrary_gate)?;

// Compute distance
let distance = characterizer.gate_distance(&arbitrary_gate, closest_clifford.as_ref())?;
```

## Performance Considerations

1. **Matrix Size**: The QR algorithm scales as O(n³) for n×n matrices
2. **Convergence**: Typically converges in 2-3 iterations per eigenvalue with shifts
3. **Memory**: Uses in-place operations where possible to minimize allocations
4. **Accuracy**: Achieves machine precision for well-conditioned matrices

## Future Enhancements

1. **Parallel eigensolvers**: Implement parallel algorithms for large matrices
2. **Sparse matrix support**: Optimize for sparse quantum gates
3. **GPU acceleration**: Leverage GPU for batch eigendecomposition
4. **Symbolic computation**: Support for parametric gates with symbolic eigenvalues
5. **Advanced decompositions**: Implement Cartan and canonical decompositions

## Testing

The implementation includes comprehensive tests:
- Analytical verification for known gates (Pauli, Hadamard)
- Numerical accuracy tests for eigenvalues and eigenvectors
- Orthogonality and normalization checks
- Gate identification for standard quantum gates
- Edge cases and error handling

All tests pass with numerical tolerance of 1e-10.