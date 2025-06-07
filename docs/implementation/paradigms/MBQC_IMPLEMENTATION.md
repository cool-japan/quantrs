# Measurement-Based Quantum Computing Implementation

## Overview

This document describes the implementation of measurement-based quantum computing (MBQC) in QuantRS2, including cluster states, graph states, measurement patterns, and the one-way quantum computing model.

## Implementation Details

### Core Components

1. **Measurement Bases**
   - Computational (Z), X, Y bases
   - Parameterized measurements in XY, XZ, YZ planes
   - Measurement operator representations

2. **Graph Structures**
   - General graph representation with adjacency lists
   - Pre-built patterns: linear, rectangular, complete, star
   - Edge addition and neighbor queries

3. **Cluster States**
   - Generation from arbitrary graphs
   - CZ gate application for entanglement
   - State vector representation

4. **Measurement Patterns**
   - Measurement basis specification
   - Measurement ordering for adaptivity
   - X/Z correction dependencies
   - Input/output qubit designation

5. **MBQC Computation**
   - Step-by-step measurement execution
   - Adaptive corrections based on outcomes
   - Final output state extraction

### Mathematical Foundation

1. **Cluster State Preparation**
   - Initialize all qubits in |+⟩ = (|0⟩ + |1⟩)/√2
   - Apply CZ gates for each edge in graph
   - Result: highly entangled resource state

2. **Universal Computation**
   - Single-qubit rotations via measurements
   - Two-qubit gates through entanglement structure
   - Measurement angles determine gate operations

3. **Measurement Calculus**
   - Pauli corrections propagate through cluster
   - Adaptive measurements based on prior outcomes
   - Deterministic computation despite randomness

### Key Features

1. **Graph State Creation**
   ```rust
   // Create linear cluster
   let graph = Graph::linear_cluster(5);
   
   // Create 2D cluster
   let graph = Graph::rectangular_cluster(4, 4);
   
   // Create cluster state
   let cluster = ClusterState::from_graph(graph)?;
   ```

2. **Measurement Patterns**
   ```rust
   let mut pattern = MeasurementPattern::new();
   
   // Add measurements
   pattern.add_measurement(0, MeasurementBasis::X);
   pattern.add_measurement(1, MeasurementBasis::XY(PI/4.0));
   
   // Add corrections
   pattern.add_x_correction(2, 0, true);
   pattern.add_z_correction(2, 1, true);
   ```

3. **MBQC Execution**
   ```rust
   let mut computation = MBQCComputation::new(graph, pattern)?;
   
   // Run all measurements
   let outcomes = computation.run()?;
   
   // Get output state
   let output = computation.output_state()?;
   ```

4. **Circuit Conversion**
   ```rust
   let converter = CircuitToMBQC::new();
   
   // Convert gates to MBQC
   let (graph, pattern) = converter.convert_single_qubit_gate(0, PI/2.0);
   let (graph, pattern) = converter.convert_cnot(0, 1);
   ```

### Implementation Choices

1. **Dense State Vector**
   - Full state representation for accuracy
   - Suitable for small to medium clusters
   - Direct measurement simulation

2. **Graph Flexibility**
   - Support arbitrary graph topologies
   - Pre-built common patterns
   - Extensible graph construction

3. **Adaptive Measurements**
   - Real-time correction application
   - Measurement outcome tracking
   - Deterministic final results

## Usage Examples

### Basic Cluster State
```rust
use quantrs2_core::prelude::*;

// Create a 3x3 cluster state
let graph = MBQCGraph::rectangular_cluster(3, 3);
let mut cluster = ClusterState::from_graph(graph)?;

// Measure corner qubit in X basis
let outcome = cluster.measure(0, MeasurementBasis::X)?;
println!("Measurement outcome: {}", outcome);
```

### Single-Qubit Rotation
```rust
// Create pattern for R_z(θ) rotation
let pattern = MeasurementPattern::single_qubit_rotation(PI/4.0);

// Create linear cluster for the pattern
let graph = MBQCGraph::linear_cluster(3);

// Execute computation
let mut mbqc = MBQCComputation::new(graph, pattern)?;
let outcomes = mbqc.run()?;

// Get rotated output state
let output = mbqc.output_state()?;
```

### Universal Gate Set
```rust
// CNOT pattern
let cnot_pattern = MeasurementPattern::cnot();

// Create appropriate cluster
let graph = MBQCGraph::rectangular_cluster(5, 3);

// Set input states
cnot_pattern.set_inputs(vec![0, 1]);
cnot_pattern.set_outputs(vec![13, 14]);

// Execute CNOT via measurements
let mut computation = MBQCComputation::new(graph, cnot_pattern)?;
computation.run()?;
```

### Custom Measurement Pattern
```rust
let mut pattern = MeasurementPattern::new();

// Design custom computation
for i in 0..5 {
    pattern.add_measurement(i, MeasurementBasis::XY(i as f64 * PI / 5.0));
}

// Add correction flow
for i in 1..5 {
    pattern.add_x_correction(i, i-1, true);
}

pattern.set_inputs(vec![0]);
pattern.set_outputs(vec![4]);
```

## Testing

The implementation includes comprehensive tests:
- Graph construction and properties
- Cluster state generation and normalization
- Measurement outcomes and projections
- Pattern execution and corrections
- Circuit-to-MBQC conversion

All tests pass with correct MBQC behavior verified.

## Future Enhancements

1. **Optimization**
   - Graph state compression techniques
   - Measurement pattern optimization
   - Resource count minimization

2. **Advanced Features**
   - Fault-tolerant MBQC
   - Blind quantum computing protocols
   - Verification of quantum computations

3. **Performance**
   - Sparse state representation
   - GPU acceleration for large clusters
   - Distributed MBQC simulation

4. **Applications**
   - Quantum algorithms in MBQC
   - Error correction via topological clusters
   - Photonic quantum computing