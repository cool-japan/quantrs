# Specialized Gate Implementations for Sim Module

## Overview

This document describes the specialized gate implementations added to the QuantRS2 simulation module. These implementations provide significant performance improvements over generic matrix multiplication by taking advantage of the specific structure of common quantum gates.

## Architecture

### Key Components

1. **SpecializedGate Trait** (`sim/src/specialized_gates.rs`)
   - Core trait for specialized gate implementations
   - Direct state vector manipulation methods
   - Gate fusion capabilities

2. **Specialized Gate Types**
   - Single-qubit: Hadamard, Pauli (X,Y,Z), Phase, S, T, Rotations (RX, RY, RZ)
   - Two-qubit: CNOT, CZ, SWAP, Controlled Phase
   - Multi-qubit: Toffoli (CCX), Fredkin (CSWAP)

3. **SpecializedStateVectorSimulator** (`sim/src/specialized_simulator.rs`)
   - Automatically uses specialized implementations when available
   - Fallback to generic matrix multiplication for unsupported gates
   - Performance tracking and statistics

## Features

### 1. Direct State Manipulation

Instead of matrix multiplication, specialized gates directly manipulate state vector amplitudes:

```rust
// Example: Specialized Hadamard gate
// Instead of matrix multiplication, directly compute:
// |0⟩ → (|0⟩ + |1⟩)/√2
// |1⟩ → (|0⟩ - |1⟩)/√2
```

### 2. Parallel Execution

All specialized gates support both sequential and parallel execution:
- Parallel execution using Rayon for large state vectors
- Configurable threshold for switching to parallel mode
- Optimal work distribution across CPU cores

### 3. Gate Fusion

Some gates can be fused for better performance:
- Two identical CNOTs cancel out
- Sequential rotations on same axis combine
- Hadamard pairs cancel out

### 4. Memory Efficiency

Specialized implementations minimize memory allocations:
- In-place amplitude updates where possible
- Reduced temporary vector allocations
- Cache-friendly access patterns

## Performance Characteristics

### Speedup Factors

Typical speedups over generic matrix multiplication:

| Gate Type | Speedup | Reason |
|-----------|---------|---------|
| Hadamard | 2-3x | Simple arithmetic operations |
| Pauli Gates | 3-4x | Direct swaps/sign flips |
| Phase Gates | 4-5x | Single amplitude modifications |
| CNOT | 2-3x | Conditional swaps only |
| Rotations | 1.5-2x | Trigonometric optimizations |

### Scaling

Performance scales well with:
- Number of qubits (up to memory limits)
- Circuit depth (through fusion and caching)
- Parallel execution for 10+ qubits

## Usage Examples

### Basic Usage

```rust
use quantrs2_sim::prelude::*;
use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;

// Create specialized simulator
let mut sim = SpecializedStateVectorSimulator::new(Default::default());

// Run circuit
let mut circuit = Circuit::new(2)?;
circuit.h(QubitId(0));
circuit.cnot(QubitId(0), QubitId(1));

let state = sim.run(&circuit)?;

// Check statistics
println!("Stats: {:?}", sim.get_stats());
```

### Configuration Options

```rust
let config = SpecializedSimulatorConfig {
    parallel: true,              // Use parallel execution
    enable_fusion: true,         // Enable gate fusion
    enable_reordering: true,     // Reorder gates for locality
    cache_conversions: true,     // Cache gate specializations
    parallel_threshold: 10,      // Min qubits for parallel
};

let mut sim = SpecializedStateVectorSimulator::new(config);
```

### Direct Gate Usage

```rust
use quantrs2_sim::specialized_gates::*;

// Create specialized gate
let gate = HadamardSpecialized { target: QubitId(0) };

// Apply to state vector
let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
gate.apply_specialized(&mut state, 1, false)?;
```

## Implementation Details

### Single-Qubit Gates

For a single-qubit gate on qubit `k` in an n-qubit system:

1. **Hadamard**: 
   ```
   For each basis state |x⟩:
   If bit k = 0: |x⟩ → (|x⟩ + |x⊕2^k⟩)/√2
   If bit k = 1: |x⟩ → (|x⊕2^k⟩ - |x⟩)/√2
   ```

2. **Pauli-X**: Swap amplitudes where bit k differs
3. **Pauli-Z**: Flip sign when bit k = 1
4. **Phase**: Apply phase when bit k = 1

### Two-Qubit Gates

For two-qubit gates on qubits `j` and `k`:

1. **CNOT**: If bit j = 1, flip bit k
2. **CZ**: If both bits = 1, flip sign
3. **SWAP**: Exchange bits j and k

### Optimization Strategies

1. **Bit Manipulation**: Use bitwise operations for index calculations
2. **Memory Access**: Process amplitudes in cache-friendly order
3. **Vectorization**: Structure loops for compiler auto-vectorization
4. **Work Stealing**: Use Rayon's work-stealing for load balancing

## Benchmarking

Run benchmarks with:

```bash
cargo run --release --example specialized_gates_demo
```

Typical results (on modern CPU):
- 4 qubits, 100 gates: 2-3x speedup
- 16 qubits, 500 gates: 3-4x speedup
- 20 qubits, 1000 gates: 4-5x speedup

## Future Enhancements

1. **Additional Gates**:
   - Controlled rotations
   - Arbitrary controlled gates
   - Custom user-defined gates

2. **Advanced Optimizations**:
   - SIMD intrinsics for specific gates
   - GPU kernel implementations
   - Circuit-level optimizations

3. **Integration**:
   - Automatic circuit optimization
   - Hardware-specific tuning
   - Noise model integration

## Testing

Comprehensive tests ensure correctness:

```bash
cargo test -p quantrs2-sim specialized
```

Tests verify:
- Correctness against matrix multiplication
- Parallel vs sequential consistency
- Edge cases (single qubit, max qubits)
- Performance characteristics