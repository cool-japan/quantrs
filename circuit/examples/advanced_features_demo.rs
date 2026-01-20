//! Advanced Features Demonstration
//!
//! This example demonstrates the new advanced features implemented in the circuit module:
//! - ZX-calculus optimization
//! - Photonic quantum circuits
//! - ML-based optimization
//! - Fault-tolerant compilation
//! - Topological quantum circuits
//! - `SciRS2` graph analysis
//! - Circuit-to-simulator interfaces

use quantrs2_circuit::prelude::*;
use quantrs2_circuit::simulator_interface::CompilationTarget;
use quantrs2_core::gate::multi::CNOT;
use quantrs2_core::gate::single::{Hadamard, PauliX};
use quantrs2_core::qubit::QubitId;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QuantRS2 Circuit Advanced Features Demo ===\n");

    // 1. Basic circuit creation
    println!("1. Creating a basic quantum circuit:");
    let mut circuit = Circuit::<3>::new();
    circuit.add_gate(Hadamard { target: QubitId(0) })?;
    circuit.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;
    circuit.add_gate(CNOT {
        control: QubitId(1),
        target: QubitId(2),
    })?;
    println!("   Circuit with {} gates created", circuit.gates().len());

    // 2. ZX-calculus optimization
    println!("\n2. ZX-calculus optimization:");
    let zx_optimizer = ZXOptimizer::new();
    let zx_diagram = zx_optimizer.circuit_to_zx(&circuit)?;
    println!(
        "   ZX diagram with {} nodes created",
        zx_diagram.nodes.len()
    );

    // 3. SciRS2 graph analysis
    println!("\n3. SciRS2 graph analysis:");
    let mut analyzer = SciRS2CircuitAnalyzer::new();
    let analysis = analyzer.analyze_circuit(&circuit)?;
    println!(
        "   Graph metrics: {} nodes, {} edges",
        analysis.metrics.num_nodes, analysis.metrics.num_edges
    );
    println!("   Communities detected: {}", analysis.communities.len());

    // 4. ML-based optimization
    println!("\n4. ML-based circuit optimization:");
    let ml_strategy = MLStrategy::NeuralNetwork {
        architecture: vec![32, 16, 8],
        learning_rate: 0.001,
        epochs: 10,
        batch_size: 16,
    };
    let mut ml_optimizer = MLCircuitOptimizer::new(ml_strategy);
    let ml_result = ml_optimizer.optimize(&circuit)?;
    println!(
        "   ML optimization completed in {:?}",
        ml_result.optimization_time
    );

    // 5. Fault-tolerant compilation
    println!("\n5. Fault-tolerant compilation:");
    let code = QECCode::SteaneCode;
    let ft_compiler = FaultTolerantCompiler::new(code);
    let ft_circuit = ft_compiler.compile(&circuit)?;
    println!(
        "   Fault-tolerant circuit: {} physical qubits, {} magic states",
        ft_circuit.physical_qubit_count, ft_circuit.magic_state_requirements
    );

    // 6. Photonic circuit conversion
    println!("\n6. Photonic quantum computation:");
    let photonic_circuit = PhotonicConverter::quantum_to_photonic(&circuit)?;
    println!(
        "   Photonic circuit: {} modes, {} gates",
        photonic_circuit.num_modes,
        photonic_circuit.gates.len()
    );

    // 7. Topological quantum computation
    println!("\n7. Topological quantum computation:");
    let anyon_model = AnyonModel::fibonacci();
    let topo_compiler = TopologicalCompiler::new(anyon_model);
    let topo_circuit = topo_compiler.compile_quantum_circuit(&circuit)?;
    println!(
        "   Topological circuit: {} anyons, {} braiding operations",
        topo_circuit.anyon_count(),
        topo_circuit.total_braiding_operations()
    );

    // 8. Circuit-to-simulator interface
    println!("\n8. Circuit compilation for simulators:");
    let mut compiler = CircuitCompiler::new();
    compiler.add_target(CompilationTarget {
        backend: SimulatorBackend::StateVector {
            max_qubits: 20,
            use_gpu: false,
            memory_optimization: MemoryOptimization::Basic,
        },
        optimization_level: SimulatorOptimizationLevel::Basic,
        instruction_set: InstructionSet::Universal,
        parallel_execution: true,
        batch_size: Some(10),
    });

    let compiled = compiler.compile(&circuit)?;
    println!(
        "   Compiled circuit: {} instructions, estimated memory: {} bytes",
        compiled.instructions.len(),
        compiled.resources.memory_bytes
    );

    // 9. Mid-circuit measurements and feed-forward
    println!("\n9. Mid-circuit measurements and feed-forward:");
    let mut measurement_circuit = MeasurementCircuit::<2>::new();
    measurement_circuit.add_gate(Box::new(Hadamard { target: QubitId(0) }))?;
    let bit = measurement_circuit.measure(QubitId(0))?;

    let condition = ClassicalCondition::equals(
        ClassicalValue::Integer(bit as u64),
        ClassicalValue::Integer(1),
    );
    measurement_circuit.add_conditional(condition, Box::new(PauliX { target: QubitId(1) }))?;
    println!(
        "   Measurement circuit with {} operations created",
        measurement_circuit.num_operations()
    );

    // 10. Cross-talk aware scheduling
    println!("\n10. Cross-talk aware scheduling:");
    let crosstalk_model = CrosstalkModel::uniform(3, 0.05);
    let scheduler = CrosstalkScheduler::new(crosstalk_model);
    let schedule = scheduler.schedule(&circuit)?;
    println!(
        "   Scheduled into {} time slices, total crosstalk: {:.3}",
        schedule.time_slices.len(),
        schedule.total_crosstalk
    );

    println!("\n=== Demo completed successfully! ===");
    Ok(())
}
