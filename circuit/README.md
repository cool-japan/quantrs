# QuantRS2-Circuit: Advanced Quantum Circuit Framework

[![Crates.io](https://img.shields.io/crates/v/quantrs2-circuit.svg)](https://crates.io/crates/quantrs2-circuit)
[![Documentation](https://docs.rs/quantrs2-circuit/badge.svg)](https://docs.rs/quantrs2-circuit)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/cool-japan/quantrs)

QuantRS2-Circuit is the comprehensive quantum circuit construction and optimization engine of the [QuantRS2](https://github.com/cool-japan/quantrs) quantum computing framework, providing advanced circuit representation, analysis, optimization, and compilation capabilities for quantum computing applications.

## Version 0.1.2

This release leverages [SciRS2](https://github.com/cool-japan/scirs2) v0.1.2 integration with refined patterns for enhanced performance in:
- Graph-based circuit optimization algorithms
- Parallel circuit analysis and transformation
- Memory-efficient circuit representations
- Hardware-aware compilation optimizations

## Core Features

### Circuit Construction & Representation
- **Fluent Builder API**: Intuitive, chainable interface for quantum circuit construction
- **Type-Safe Operations**: Compile-time verification of qubit counts and gate operations
- **DAG Representation**: Directed acyclic graph representation for advanced circuit analysis
- **Classical Control**: Support for classical control flow, measurements, and feed-forward
- **Mid-Circuit Measurements**: Real-time measurement with conditional operations

### Advanced Optimization Framework
- **Multi-Pass Optimization**: Configurable optimization pipeline with cost-aware scheduling
- **Hardware-Aware Compilation**: Device-specific optimization with noise models and constraints
- **Gate Fusion**: Intelligent gate merging for reduced circuit depth and gate count
- **Template Matching**: Pattern recognition for efficient gate sequence replacement
- **Noise-Aware Optimization**: Circuit optimization considering realistic device characteristics

### Quantum Circuit Analysis
- **Commutation Analysis**: Automatic detection of commuting operations for reordering
- **Circuit Equivalence**: Verification of circuit equivalence using multiple methods
- **Topological Analysis**: Circuit connectivity and entanglement structure analysis
- **Resource Estimation**: Accurate resource counting and complexity analysis
- **Performance Profiling**: Detailed metrics for circuit optimization and execution

### Hardware Integration
- **Device Routing**: SABRE and lookahead algorithms for qubit mapping and SWAP insertion
- **Pulse-Level Control**: Translation to pulse sequences for direct hardware control
- **Crosstalk Mitigation**: Scheduling algorithms to minimize gate crosstalk effects
- **Calibration Integration**: Real-time device calibration data incorporation
- **Multi-Architecture Support**: Superconducting, trapped ion, photonic, and topological systems

## Usage Examples

### Basic Circuit Construction

```rust
use quantrs2_circuit::prelude::*;

fn basic_circuit_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a circuit with compile-time qubit count verification
    let mut circuit = Circuit::<4>::new();
    
    // Build a quantum Fourier transform circuit
    circuit.h(0)?
           .controlled_phase(0, 1, std::f64::consts::PI / 2.0)?
           .controlled_phase(0, 2, std::f64::consts::PI / 4.0)?
           .controlled_phase(0, 3, std::f64::consts::PI / 8.0)?
           .h(1)?
           .controlled_phase(1, 2, std::f64::consts::PI / 2.0)?
           .controlled_phase(1, 3, std::f64::consts::PI / 4.0)?
           .h(2)?
           .controlled_phase(2, 3, std::f64::consts::PI / 2.0)?
           .h(3)?;
    
    // Convert to DAG for analysis
    let dag = circuit_to_dag(&circuit);
    println!("Circuit depth: {}, Gate count: {}", dag.depth(), dag.gate_count());
    
    Ok(())
}
```

### Advanced Optimization Pipeline

```rust
use quantrs2_circuit::prelude::*;

fn optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::<8>::new();
    
    // Create a complex circuit with optimization opportunities
    for i in 0..8 {
        circuit.h(i)?;
    }
    for i in 0..7 {
        circuit.cnot(i, i + 1)?;
    }
    for i in 0..8 {
        circuit.rz(i, 0.1)?;
    }
    
    // Configure optimization pipeline
    let mut pass_manager = PassManager::new();
    pass_manager.add_pass(OptimizationPass::GateCancellation)?
                .add_pass(OptimizationPass::RotationMerging)?
                .add_pass(OptimizationPass::CommutationOptimization)?
                .add_pass(OptimizationPass::TemplateMatching)?;
    
    // Apply optimizations
    let optimized_circuit = pass_manager.run(&circuit)?;
    let report = pass_manager.get_report();
    
    println!("Original gates: {}, Optimized gates: {}, Reduction: {:.1}%",
             report.original_gates, report.final_gates, report.reduction_percentage());
    
    Ok(())
}
```

### Hardware-Aware Compilation

```rust
use quantrs2_circuit::prelude::*;

fn hardware_compilation_example() -> Result<(), Box<dyn std::error::Error>> {
    // Define device topology (e.g., Google Sycamore)
    let coupling_map = CouplingMap::from_adjacency_list(&[
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8), (8, 9),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    ]);
    
    // Create logical circuit
    let mut logical_circuit = Circuit::<10>::new();
    logical_circuit.h(0)?
                   .cnot(0, 5)?  // This requires routing
                   .cnot(5, 9)?
                   .cnot(9, 4)?;
    
    // Configure router with device constraints
    let router_config = SabreConfig {
        max_iterations: 100,
        decay: 0.001,
        lookahead_depth: 3,
    };
    
    let mut router = SabreRouter::new(coupling_map, router_config);
    let routed_result = router.route(&logical_circuit)?;
    
    println!("Added {} SWAP gates for routing", routed_result.swap_count);
    println!("Final circuit depth: {}", routed_result.circuit.depth());
    
    Ok(())
}
```

### Circuit Analysis and Verification

```rust
use quantrs2_circuit::prelude::*;

fn circuit_analysis_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create two allegedly equivalent circuits
    let mut circuit1 = Circuit::<2>::new();
    circuit1.h(0)?.cnot(0, 1)?;
    
    let mut circuit2 = Circuit::<2>::new();
    circuit2.cnot(0, 1)?.h(0)?.cnot(0, 1)?.x(1)?.cnot(0, 1)?;
    
    // Verify equivalence using multiple methods
    let equivalence_checker = EquivalenceChecker::new();
    let result = equivalence_checker.check_equivalence(&circuit1, &circuit2)?;
    
    match result.equivalence_type {
        EquivalenceType::Unitary => println!("Circuits are unitarily equivalent"),
        EquivalenceType::Structural => println!("Circuits are structurally identical"),
        EquivalenceType::None => println!("Circuits are not equivalent"),
    }
    
    // Perform detailed circuit analysis
    let analyzer = CircuitAnalyzer::new();
    let metrics = analyzer.analyze(&circuit1)?;
    
    println!("Circuit metrics:");
    println!("  Depth: {}", metrics.depth);
    println!("  Gate count: {}", metrics.gate_count);
    println!("  Two-qubit gates: {}", metrics.two_qubit_gate_count);
    println!("  Entanglement measure: {:.3}", metrics.entanglement_measure);
    
    Ok(())
}
```

### Classical Control and Measurement

```rust
use quantrs2_circuit::prelude::*;

fn classical_control_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create measurement circuit with classical control
    let mut measurement_circuit = MeasurementCircuitBuilder::new(3, 3);
    
    // Prepare a superposition state
    measurement_circuit.h(0)?
                       .cnot(0, 1)?
                       .cnot(1, 2)?;
    
    // Mid-circuit measurement with feed-forward
    measurement_circuit.measure(0, 0)?;  // Measure qubit 0 to classical bit 0
    
    // Conditional operations based on measurement result
    let condition = ClassicalCondition::equals(0, ClassicalValue::One);
    measurement_circuit.conditional_x(1, condition)?;
    
    // Final measurements
    measurement_circuit.measure(1, 1)?
                       .measure(2, 2)?;
    
    // Execute with conditional logic
    let execution_result = measurement_circuit.execute_with_simulation(1000)?;
    println!("Classical outcomes: {:?}", execution_result.classical_outcomes);
    
    Ok(())
}
```

### Fault-Tolerant Circuit Compilation

```rust
use quantrs2_circuit::prelude::*;

fn fault_tolerant_example() -> Result<(), Box<dyn std::error::Error>> {
    // Define a surface code for error correction
    let qec_code = QECCode::surface_code(3, 3)?;  // 3x3 surface code
    
    // Create logical circuit
    let mut logical_circuit = Circuit::<1>::new();
    logical_circuit.h(0)?
                   .s(0)?
                   .h(0)?;
    
    // Compile to fault-tolerant version
    let ft_compiler = FaultTolerantCompiler::new(qec_code);
    let ft_circuit = ft_compiler.compile(&logical_circuit)?;
    
    println!("Logical circuit gates: {}", logical_circuit.gate_count());
    println!("Fault-tolerant gates: {}", ft_circuit.physical_circuit.gate_count());
    println!("Resource overhead: {}x", ft_circuit.resource_overhead.total_overhead());
    
    // Analyze magic state requirements
    println!("Magic states required: {}", ft_circuit.resource_overhead.magic_states);
    
    Ok(())
}
```

### QASM Import/Export

```rust
use quantrs2_circuit::prelude::*;

fn qasm_interoperability_example() -> Result<(), Box<dyn std::error::Error>> {
    // Parse QASM circuit
    let qasm_code = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        
        qubit[2] q;
        bit[2] c;
        
        h q[0];
        cnot q[0], q[1];
        c = measure q;
    "#;
    
    let qasm_parser = QasmParser::new();
    let parsed_program = qasm_parser.parse(qasm_code)?;
    
    // Convert to QuantRS2 circuit
    let circuit = parsed_program.to_quantrs2_circuit()?;
    
    // Optimize the circuit
    let optimizer = CircuitOptimizer::new();
    let optimized_circuit = optimizer.optimize(&circuit, OptimizationLevel::Aggressive)?;
    
    // Export back to QASM
    let export_options = ExportOptions {
        include_measurements: true,
        use_custom_gates: false,
        optimization_level: Some(OptimizationLevel::Basic),
    };
    
    let qasm_exporter = QasmExporter::new();
    let exported_qasm = qasm_exporter.export(&optimized_circuit, &export_options)?;
    
    println!("Exported QASM:\n{}", exported_qasm);
    
    Ok(())
}
```

## Comprehensive Module Structure

### Core Circuit Framework
- **builder.rs**: Fluent circuit builder API with type-safe operations
- **dag.rs**: Directed acyclic graph representation for circuit analysis
- **classical.rs**: Classical control flow, conditional operations, and measurements
- **measurement.rs**: Mid-circuit measurements and feed-forward control

### Optimization & Analysis
- **optimization/**: Comprehensive optimization framework
  - **pass_manager.rs**: Configurable optimization pipeline management
  - **passes.rs**: Individual optimization passes (gate cancellation, fusion, etc.)
  - **cost_model.rs**: Hardware-aware cost models for optimization decisions
  - **analysis.rs**: Circuit analysis and metric computation
  - **gate_properties.rs**: Gate property analysis for optimization
- **optimizer.rs**: High-level circuit optimization interface
- **graph_optimizer.rs**: Graph-based optimization algorithms
- **commutation.rs**: Gate commutation analysis and reordering
- **equivalence.rs**: Circuit equivalence verification using multiple methods

### Hardware Integration & Compilation
- **routing/**: Quantum circuit routing and layout algorithms
  - **sabre.rs**: SABRE routing algorithm
  - **lookahead.rs**: Lookahead routing for enhanced performance
  - **coupling_map.rs**: Device topology representation and analysis
  - **swap_network.rs**: SWAP gate insertion and optimization
- **pulse.rs**: Pulse-level control and waveform generation
- **crosstalk.rs**: Crosstalk analysis and mitigation scheduling
- **topology.rs**: Circuit topology analysis and optimization

### Advanced Circuit Types
- **fault_tolerant.rs**: Fault-tolerant circuit compilation with error correction
- **photonic.rs**: Photonic quantum circuit representation and operations
- **topological.rs**: Topological quantum circuit compilation with anyons

### Circuit Transformation & Synthesis
- **synthesis.rs**: Unitary synthesis and gate decomposition
- **tensor_network.rs**: Tensor network representation and compression
- **zx_calculus.rs**: ZX-diagram optimization and circuit extraction
- **slicing.rs**: Circuit slicing for parallel execution

### Import/Export & Interoperability
- **qasm/**: QASM 2.0/3.0 support
  - **parser.rs**: QASM parsing with full language support
  - **exporter.rs**: Circuit export to QASM format
  - **ast.rs**: Abstract syntax tree representation
  - **validator.rs**: QASM validation and error checking
- **simulator_interface.rs**: Integration with various simulator backends

### Machine Learning & Advanced Algorithms
- **ml_optimization.rs**: Machine learning-based circuit optimization
- **scirs2_integration.rs**: SciRS2 integration for graph algorithms and optimization
  - Leverages `scirs2_core::parallel_ops` for parallel circuit transformations
  - Uses SciRS2 graph algorithms for optimized circuit routing
  - Integrates SciRS2 memory management for large circuit handling

## API Overview

### Core Types

- `Circuit<N>`: Quantum circuit with const generic for qubit count
- `Simulator`: Trait for backends that can run quantum circuits

### Macros

- `circuit!`: Creates a new circuit with the specified number of qubits
- `qubits!`: Creates a set of qubits for operations
- `quantum!`: DSL for quantum circuit construction (in development)

## Technical Details

- The circuit builder uses a fluent API for method chaining
- Gate operations are type-checked at compile time where possible
- The framework supports custom gates through the `GateOp` trait
- Circuit operations return `QuantRS2Result` for error handling

## Future Plans

See [TODO.md](TODO.md) for planned features.

## Integration with Other QuantRS2 Modules

This module is designed to work seamlessly with:
- [quantrs2-core](../core/README.md): Uses core types and gates for circuit construction
- [quantrs2-sim](../sim/README.md): Circuits can be executed on simulators
- [quantrs2-device](../device/README.md): Circuits can be transpiled and run on real hardware

## License

This project is licensed under either:

- [Apache License, Version 2.0](../LICENSE-APACHE)
- [MIT License](../LICENSE-MIT)

at your option.