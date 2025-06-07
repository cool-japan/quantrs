# Fermionic Operations Implementation

## Overview

This document describes the implementation of fermionic operators and the Jordan-Wigner transformation in QuantRS2. This enables quantum simulation of fermionic systems such as molecules, condensed matter systems, and quantum chemistry applications.

## Architecture

### Core Components

#### 1. Fermionic Operators (`fermionic.rs`)

```rust
pub struct FermionOperator {
    pub op_type: FermionOperatorType,
    pub mode: usize,
    pub coefficient: Complex64,
}
```

Operator types:
- **Creation**: a† - Creates a fermion at mode i
- **Annihilation**: a - Destroys a fermion at mode i  
- **Number**: n = a†a - Counts fermions at mode i
- **Identity**: I - Identity operator

#### 2. Fermionic Terms

```rust
pub struct FermionTerm {
    pub operators: Vec<FermionOperator>,
    pub coefficient: Complex64,
}
```

Features:
- Products of fermionic operators
- Normal ordering with anticommutation
- Hermitian conjugation support

#### 3. Fermionic Hamiltonians

```rust
pub struct FermionHamiltonian {
    pub terms: Vec<FermionTerm>,
    pub n_modes: usize,
}
```

Common Hamiltonian terms:
- **One-body**: h_ij a†_i a_j (hopping, on-site energy)
- **Two-body**: g_ijkl a†_i a†_j a_k a_l (interactions)
- **Chemical potential**: μ n_i

## Jordan-Wigner Transformation

### Theory

The Jordan-Wigner transformation maps fermionic operators to qubit operators:

- a†_j = σ⁻_j ⊗ Z_{<j} = (X_j - iY_j)/2 ⊗ ∏_{k<j} Z_k
- a_j = σ⁺_j ⊗ Z_{<j} = (X_j + iY_j)/2 ⊗ ∏_{k<j} Z_k
- n_j = a†_j a_j = (I - Z_j)/2

Where the Z string maintains fermionic anticommutation relations.

### Implementation

```rust
pub struct JordanWigner {
    n_modes: usize,
}

impl JordanWigner {
    pub fn transform_operator(&self, op: &FermionOperator) 
        -> QuantRS2Result<Vec<QubitOperator>>
    
    pub fn transform_hamiltonian(&self, hamiltonian: &FermionHamiltonian) 
        -> QuantRS2Result<QubitOperator>
}
```

Key features:
1. **Automatic Z-string generation**: Correctly handles phase factors
2. **Operator composition**: Combines multiple fermionic operators
3. **Term simplification**: Merges like terms in qubit representation

## Qubit Operator Representation

### Pauli Operators

```rust
pub enum PauliOperator {
    I,      // Identity
    X,      // Pauli X
    Y,      // Pauli Y
    Z,      // Pauli Z
    Plus,   // (X + iY)/2
    Minus,  // (X - iY)/2
}
```

### Qubit Terms and Operators

```rust
pub struct QubitTerm {
    pub operators: Vec<(usize, PauliOperator)>,
    pub coefficient: Complex64,
}

pub struct QubitOperator {
    pub terms: Vec<QubitTerm>,
    pub n_qubits: usize,
}
```

Operations:
- Addition of operators
- Multiplication with Pauli algebra
- Simplification by combining terms

## Usage Examples

### Basic Fermionic Operations

```rust
use quantrs2_core::prelude::*;

// Create fermionic operators
let c_dag = FermionOperator::creation(0);
let c = FermionOperator::annihilation(1);

// Build a hopping term: -t c†_0 c_1
let hopping = FermionTerm::new(
    vec![c_dag, c],
    Complex64::new(-1.0, 0.0)  // t = 1.0
);
```

### Building Hamiltonians

```rust
// Create a Hubbard model Hamiltonian
let mut ham = FermionHamiltonian::new(4);

// Hopping terms
for i in 0..3 {
    ham.add_one_body(i, i+1, Complex64::new(-1.0, 0.0));
    ham.add_one_body(i+1, i, Complex64::new(-1.0, 0.0));
}

// On-site interaction
for i in 0..4 {
    ham.add_two_body(i, i, i, i, Complex64::new(2.0, 0.0));
}
```

### Jordan-Wigner Transformation

```rust
// Transform to qubit operators
let jw = JordanWigner::new(4);
let qubit_ham = jw.transform_hamiltonian(&ham)?;

// Convert to quantum gates
let gates = qubit_operator_to_gates(&qubit_ham)?;
```

### Molecular Hamiltonians

```rust
// H2 molecule in minimal basis
let mut h2 = FermionHamiltonian::new(4); // 2 spatial orbitals × 2 spins

// One-electron integrals
h2.add_one_body(0, 0, Complex64::new(-1.2563, 0.0));
h2.add_one_body(1, 1, Complex64::new(-1.2563, 0.0));
h2.add_one_body(2, 2, Complex64::new(-0.4719, 0.0));
h2.add_one_body(3, 3, Complex64::new(-0.4719, 0.0));

// Two-electron integrals
h2.add_two_body(0, 1, 1, 0, Complex64::new(0.6757, 0.0));
// ... more terms
```

## Implementation Details

### Anticommutation Relations

Fermionic operators satisfy:
- {a_i, a_j†} = δ_ij
- {a_i, a_j} = 0
- {a_i†, a_j†} = 0

The implementation handles these through:
1. Normal ordering algorithms
2. Sign tracking during operator reordering
3. Automatic simplification

### Performance Optimizations

1. **Sparse representation**: Only non-zero Pauli strings stored
2. **Hash maps**: Fast term lookup and combination
3. **Lazy evaluation**: Transform only when needed
4. **Caching**: Reuse common transformations

### Error Handling

Comprehensive validation:
- Mode index bounds checking
- Operator ordering validation
- Coefficient normalization
- Hermiticity verification

## Future Enhancements

1. **Bravyi-Kitaev Transform**: More efficient for some circuits
2. **Parity Transform**: Alternative mapping
3. **Symmetry Reduction**: Exploit particle number, spin
4. **Trotterization**: Time evolution operators
5. **Active Space**: Freeze core orbitals
6. **Sparse Hamiltonians**: For large systems
7. **GPU Acceleration**: Parallel transformations

## Testing

Comprehensive test suite:
- Operator creation and manipulation
- Hermitian conjugation
- Jordan-Wigner correctness
- Hamiltonian construction
- Qubit operator algebra
- 6 new tests, 115 total passing

## Applications

The fermionic operations enable:
- **Quantum Chemistry**: Molecular simulations
- **Condensed Matter**: Hubbard models, superconductivity
- **Nuclear Physics**: Pairing models
- **Quantum Biology**: Electron transport
- **Material Science**: Band structure calculations

## Code Quality

- Type-safe operator representations
- Clear mathematical mappings
- Comprehensive documentation
- Modular transformation framework
- Integration with existing gates

## References

1. Jordan & Wigner, "Über das Paulische Äquivalenzverbot" (1928)
2. Seeley et al., "The Bravyi-Kitaev transformation" (2012)
3. McClean et al., "OpenFermion: The electronic structure package" (2020)
4. Tranter et al., "The Bravyi-Kitaev transformation for quantum computation" (2018)