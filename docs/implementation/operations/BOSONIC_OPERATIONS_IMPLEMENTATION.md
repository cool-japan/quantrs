# Bosonic Operations Implementation

## Overview

The bosonic operations module in QuantRS2 provides comprehensive support for continuous variable quantum computing and quantum optics simulations. This implementation includes bosonic operators, Gaussian states, and Hamiltonians for various quantum optical systems.

## Key Features

### 1. Bosonic Operators

The module implements all fundamental bosonic operators:

- **Creation operator (a†)**: Creates a quantum of excitation
- **Annihilation operator (a)**: Destroys a quantum of excitation  
- **Number operator (n = a†a)**: Counts the number of excitations
- **Position operator (x = (a + a†)/√2)**: Quadrature operator
- **Momentum operator (p = i(a† - a)/√2)**: Conjugate quadrature
- **Displacement operator D(α)**: Coherent state generation
- **Squeeze operator S(z)**: Quadrature squeezing

### 2. Matrix Representations

All operators are represented as dense complex matrices with truncated Fock space:

```rust
let a = BosonOperator::annihilation(mode, truncation);
let matrix = a.to_matrix()?; // Array2<Complex64>
```

The implementation includes:
- Efficient matrix construction for sparse operators
- Custom matrix exponential via Padé approximation for displacement/squeeze operators
- Proper normalization for canonical commutation relations

### 3. Bosonic Terms and Hamiltonians

Support for multi-mode systems with arbitrary operator products:

```rust
let mut ham = BosonHamiltonian::new(n_modes, truncation);
ham.add_harmonic_oscillator(0, omega);
ham.add_beam_splitter(0, 1, coupling);
ham.add_kerr_nonlinearity(0, chi);
ham.add_two_mode_squeezing(0, 1, xi);
```

### 4. Gaussian States

Efficient representation of Gaussian quantum states:

```rust
let vacuum = GaussianState::vacuum(n_modes);
let coherent = GaussianState::coherent(n_modes, mode, alpha);
let squeezed = GaussianState::squeezed(n_modes, mode, r, phi);
```

Features:
- Displacement vector and covariance matrix representation
- Symplectic transformations
- State purity calculations

### 5. Helper Utilities

- **Kronecker products** for multi-mode operators
- **Normal ordering** for operator expressions
- **Sparse representations** for memory efficiency

## Implementation Details

### Matrix Exponential

Since SciRS2's matrix exponential doesn't support complex numbers, we implemented a custom Padé approximation:

```rust
fn matrix_exponential_complex(a: &Array2<Complex64>) -> QuantRS2Result<Array2<Complex64>> {
    // Scale matrix to reduce norm
    // Apply Padé(6,6) approximation
    // Square result to undo scaling
}
```

### Commutation Relations

The implementation correctly handles canonical commutation relations:
- [a, a†] = 1
- [x, p] = i

Note: Boundary effects occur at maximum Fock number due to truncation.

### Multi-mode Systems

Operators on different modes are combined using Kronecker products:

```rust
fn to_matrix(&self, n_modes: usize) -> QuantRS2Result<Array2<Complex64>> {
    // Build full operator using mode-wise Kronecker products
}
```

## Usage Examples

### Basic Operators

```rust
use quantrs2_core::prelude::*;

// Create operators
let a = BosonOperator::annihilation(0, 10);
let a_dag = BosonOperator::creation(0, 10);

// Get matrix representations
let a_matrix = a.to_matrix()?;
let n_matrix = BosonOperator::number(0, 10).to_matrix()?;
```

### Quantum Harmonic Oscillator

```rust
let mut ham = BosonHamiltonian::new(1, 20);
ham.add_harmonic_oscillator(0, 1.0); // ω = 1

let h_matrix = ham.to_matrix()?;
```

### Two-Mode Entanglement

```rust
let mut ham = BosonHamiltonian::new(2, 10);
ham.add_beam_splitter(0, 1, 0.5); // 50:50 beam splitter
```

### Gaussian State Evolution

```rust
let mut state = GaussianState::coherent(1, 0, Complex64::new(2.0, 0.0));

// Apply squeezing
let s_matrix = /* symplectic squeeze matrix */;
state.apply_symplectic(&s_matrix)?;
```

## Technical Notes

1. **Truncation**: All operators use finite Fock space truncation. Choose truncation based on expected photon numbers.

2. **Performance**: Dense matrix representation is used for simplicity. For large truncations, consider sparse representations.

3. **Numerical Stability**: Matrix exponential uses scaling and squaring for numerical stability.

4. **Conventions**: We use the physics convention where [x, p] = i (ℏ = 1).

## Future Enhancements

1. **Sparse Matrix Support**: Integrate with SciRS2 sparse matrices when complex number support is added
2. **GPU Acceleration**: Utilize GPU for large matrix operations
3. **Qubit Encoding**: Complete implementation of bosonic-to-qubit encodings (binary, unary, Gray code)
4. **Advanced Operations**: Add support for more exotic states (cat states, GKP states, etc.)
5. **Optimization**: Implement efficient algorithms for specific operator combinations

## References

- Weedbrook et al., "Gaussian quantum information", Rev. Mod. Phys. 84, 621 (2012)
- Serafini, "Quantum Continuous Variables", CRC Press (2017)
- Braunstein & van Loock, "Quantum information with continuous variables", Rev. Mod. Phys. 77, 513 (2005)