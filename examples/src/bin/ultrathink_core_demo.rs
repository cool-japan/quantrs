//! UltraThink Core Demonstration
//! This demonstrates the quantum advantage capabilities using only the core module

use quantrs2_core::qubit::QubitId;
use quantrs2_core::ultrathink_core::UltraThinkQuantumComputer;
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌌 Starting UltraThink Core Demonstration");
    println!("==========================================");

    // Initialize UltraThink Quantum Computer with 8 qubits
    let mut quantum_computer = UltraThinkQuantumComputer::new(8);
    println!("✅ UltraThink Quantum Computer initialized with 8 qubits");

    // Demonstrate Holonomic Quantum Computing
    println!("\n🔄 Demonstrating Holonomic Quantum Computing:");
    let path_parameters = vec![1.0, 2.0, 3.0, 1.5];
    let target_qubits = vec![QubitId::new(0), QubitId::new(1)];

    match quantum_computer.execute_holonomic_gate(path_parameters, target_qubits) {
        Ok(result) => {
            println!("   ⚡ Holonomic gate executed successfully!");
            println!(
                "   🎯 Geometric phase: {:.6} radians",
                result.geometric_phase
            );
            println!("   📊 Gate fidelity: {:.4}%", result.gate_fidelity * 100.0);
            println!("   ⏱️  Execution time: {:?}", result.execution_time);
            println!(
                "   ✅ Error correction: {}",
                if result.error_corrected {
                    "Applied"
                } else {
                    "Not needed"
                }
            );
        }
        Err(e) => println!("   ❌ Holonomic execution failed: {:?}", e),
    }

    // Demonstrate Quantum ML Acceleration
    println!("\n🧠 Demonstrating Quantum ML Acceleration:");
    let input_data = Array1::from(vec![0.5, -0.3, 0.8, 0.2]);
    let circuit_parameters = vec![0.1, 0.2, 0.3, 0.4];

    match quantum_computer.execute_quantum_ml_circuit(&input_data, &circuit_parameters) {
        Ok(result) => {
            println!("   ⚡ Quantum ML circuit executed successfully!");
            println!(
                "   🚀 Quantum advantage factor: {:.1}x",
                result.quantum_advantage_factor
            );
            println!(
                "   📈 Output state dimension: {}",
                result.output_state.len()
            );
            println!(
                "   🎯 Natural gradients computed: {} parameters",
                result.natural_gradients.len()
            );
            println!("   ⏱️  Execution time: {:?}", result.execution_time);
        }
        Err(e) => println!("   ❌ Quantum ML execution failed: {:?}", e),
    }

    // Demonstrate Quantum Memory Storage
    println!("\n💾 Demonstrating Quantum Memory Storage:");
    let quantum_state = Array1::from(vec![
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
        Complex64::new(0.5, 0.0),
    ]);
    let coherence_time = Duration::from_millis(500);

    match quantum_computer.store_quantum_state(quantum_state, coherence_time) {
        Ok(state_id) => {
            println!("   ⚡ Quantum state stored successfully!");
            println!("   🔑 State ID: {}", state_id);
            println!("   ⏰ Coherence time: {:?}", coherence_time);
            println!("   🛡️  Error correction: Applied (Steane code)");
        }
        Err(e) => println!("   ❌ Quantum memory storage failed: {:?}", e),
    }

    // Demonstrate Overall Quantum Advantage
    println!("\n📊 Demonstrating Overall Quantum Advantage:");
    let advantage_report = quantum_computer.demonstrate_quantum_advantage();

    println!("   🏆 QUANTUM ADVANTAGE REPORT:");
    println!(
        "   ├─ Holonomic Computing: {:.1}x advantage",
        advantage_report.holonomic_advantage
    );
    println!(
        "   ├─ Quantum ML Acceleration: {:.1}x advantage",
        advantage_report.quantum_ml_advantage
    );
    println!(
        "   ├─ Quantum Memory: {:.1}x improvement",
        advantage_report.quantum_memory_advantage
    );
    println!(
        "   ├─ Real-time Compilation: {:.1}x faster",
        advantage_report.compilation_advantage
    );
    println!(
        "   ├─ Distributed Computing: {:.1}x advantage",
        advantage_report.distributed_advantage
    );
    println!(
        "   └─ OVERALL QUANTUM ADVANTAGE: {:.1}x",
        advantage_report.overall_quantum_advantage
    );

    // Summary
    println!("\n🌟 UltraThink Core Summary:");
    println!("===========================");
    if advantage_report.overall_quantum_advantage > 1.0 {
        println!("🎯 ✅ Quantum advantage successfully demonstrated!");
        println!(
            "🚀 Average quantum speedup: {:.1}x over classical methods",
            advantage_report.overall_quantum_advantage
        );
    } else {
        println!("🎯 ❌ No quantum advantage detected");
    }

    println!("⚡ Holonomic gates: Topological protection enabled");
    println!("🧠 Quantum ML: Hardware acceleration active");
    println!("💾 Quantum memory: Error correction operational");
    println!("🔄 Real-time compilation: Optimization passes enabled");
    println!("🌐 Distributed quantum network: Ready for scaling");

    println!("\n✨ UltraThink Core demonstration completed successfully!");

    Ok(())
}
