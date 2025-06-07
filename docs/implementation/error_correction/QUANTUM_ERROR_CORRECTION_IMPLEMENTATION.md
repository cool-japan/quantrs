# Quantum Error Correction Implementation

## Overview

This document describes the implementation of quantum error correction codes in QuantRS2, including stabilizer codes, surface codes, color codes, and syndrome decoders.

## Implementation Details

### Core Components

1. **Pauli Operators**
   - Single-qubit Pauli operators (I, X, Y, Z)
   - Multi-qubit Pauli strings with phase tracking
   - Pauli multiplication and commutation relations

2. **Stabilizer Codes**
   - General stabilizer code framework
   - Built-in codes: repetition, 5-qubit, Steane
   - Syndrome calculation and logical operators

3. **Surface Codes**
   - Lattice-based surface code construction
   - X and Z stabilizer plaquettes
   - Configurable lattice dimensions

4. **Color Codes**
   - Triangular color code lattice
   - Three-colorable face structure
   - Conversion to stabilizer formalism

5. **Syndrome Decoders**
   - Lookup table decoder for small codes
   - Minimum Weight Perfect Matching (MWPM) decoder
   - Extensible decoder interface

### Key Features

1. **Pauli String Operations**
   ```rust
   let ps1 = PauliString::new(vec![Pauli::X, Pauli::Y, Pauli::Z]);
   let ps2 = PauliString::new(vec![Pauli::Z, Pauli::I, Pauli::X]);
   let product = ps1.multiply(&ps2)?;
   let commutes = ps1.commutes_with(&ps2)?;
   ```

2. **Stabilizer Code Construction**
   ```rust
   // Built-in codes
   let rep_code = StabilizerCode::repetition_code();
   let steane = StabilizerCode::steane_code();
   
   // Custom code
   let code = StabilizerCode::new(n, k, d, stabilizers, logical_x, logical_z)?;
   ```

3. **Surface Code Lattice**
   ```rust
   let surface = SurfaceCode::new(5, 5); // 5x5 lattice
   let distance = surface.distance();     // Code distance
   let stab_code = surface.to_stabilizer_code();
   ```

4. **Syndrome Decoding**
   ```rust
   let code = StabilizerCode::steane_code();
   let decoder = LookupDecoder::new(&code)?;
   
   let error = PauliString::new(vec![Pauli::X, ...]);
   let syndrome = code.syndrome(&error)?;
   let correction = decoder.decode(&syndrome)?;
   ```

### Mathematical Foundation

1. **Stabilizer Formalism**
   - Codes defined by commuting Pauli operators
   - Logical operators commute with stabilizers
   - Error syndromes from anti-commutation

2. **Surface Code Properties**
   - Planar code with local stabilizers
   - Distance scales with lattice size
   - Topological error correction

3. **Decoding Algorithms**
   - Lookup table: precomputed syndrome-to-error mapping
   - MWPM: graph-based minimum weight matching
   - Maximum likelihood decoding approximation

### Implementation Choices

1. **Dense Matrix Representation**
   - Pauli matrices use dense arrays for simplicity
   - Complex phase tracking for Pauli strings
   - Efficient commutation checking

2. **Flexible Code Framework**
   - Abstract `StabilizerCode` for any CSS/non-CSS code
   - Specialized types for surface and color codes
   - Extensible decoder trait interface

3. **Error Handling**
   - Validation of code parameters (n, k, d)
   - Commutation checks for stabilizers
   - Syndrome decoding failure handling

## Usage Examples

### Basic Stabilizer Code
```rust
use quantrs2_core::prelude::*;

// Create a 3-qubit repetition code
let code = StabilizerCode::repetition_code();

// Apply an error
let error = PauliString::new(vec![Pauli::X, Pauli::I, Pauli::I]);

// Calculate syndrome
let syndrome = code.syndrome(&error)?;
println!("Syndrome: {:?}", syndrome); // [true, false]

// Decode the error
let decoder = LookupDecoder::new(&code)?;
let correction = decoder.decode(&syndrome)?;
```

### Surface Code
```rust
// Create a 5x5 surface code
let surface = SurfaceCode::new(5, 5);

// Convert to stabilizer representation
let code = surface.to_stabilizer_code();
println!("Qubits: {}, Distance: {}", code.n, code.d);

// Use MWPM decoder for surface codes
let decoder = MWPMDecoder::new(surface);
```

### Color Code
```rust
// Create triangular color code
let color = ColorCode::triangular(4);

// Access face information
for (qubits, color) in &color.faces {
    println!("Face {:?} on qubits {:?}", color, qubits);
}

// Convert to stabilizer code
let code = color.to_stabilizer_code();
```

## Testing

The implementation includes comprehensive tests:
- Pauli operator multiplication and commutation
- Stabilizer code validation and properties
- Surface code lattice construction
- Syndrome calculation correctness
- Decoder functionality

All tests pass with proper error correction behavior verified.

## Future Enhancements

1. **Additional Codes**
   - Bacon-Shor codes
   - Concatenated codes
   - Subsystem codes

2. **Advanced Decoders**
   - Full blossom algorithm for MWPM
   - Belief propagation decoders
   - Machine learning decoders

3. **Fault-Tolerant Operations**
   - Transversal gate implementations
   - Magic state distillation
   - Lattice surgery operations

4. **Performance Optimization**
   - Sparse syndrome representation
   - Parallel decoding algorithms
   - GPU-accelerated syndrome extraction