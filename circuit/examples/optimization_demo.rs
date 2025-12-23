//! Demonstration of the circuit optimization system
//!
//! This example shows how to use the various optimization passes and features.

use quantrs2_circuit::optimization::passes::OptimizationPassExt;
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::{multi, single, GateOp};
use quantrs2_core::qubit::QubitId;
use std::time::Instant;

fn main() {
    println!("=== QuantRS2 Circuit Optimization Demo ===\n");

    // Create a sample circuit
    let circuit = create_sample_circuit();

    // Demo 1: Basic optimization with different levels
    demo_optimization_levels(&circuit);

    // Demo 2: Hardware-specific optimization
    demo_hardware_optimization(&circuit);

    // Demo 3: Custom optimization pipeline
    demo_custom_optimization(&circuit);

    // Demo 4: Gate property analysis
    demo_gate_properties();

    // Demo 5: Benchmarking optimization passes
    benchmark_optimization_passes(&circuit);
}

fn create_sample_circuit() -> Circuit<4> {
    // Create a circuit that has optimization opportunities
    let mut circuit = Circuit::<4>::new();

    // Add some gates with optimization opportunities
    // Pattern 1: Redundant gates (H-H = I, X-X = I)
    let _ = circuit.h(0);
    let _ = circuit.h(0);

    // Pattern 2: Commutable gates
    let _ = circuit.x(1);
    let _ = circuit.z(0);
    let _ = circuit.cnot(0, 1);

    // Pattern 3: Mergeable rotations
    let _ = circuit.rz(2, std::f64::consts::PI / 4.0);
    let _ = circuit.rz(2, std::f64::consts::PI / 4.0);

    // Pattern 4: Known patterns (H-X-H = Z)
    let _ = circuit.h(3);
    let _ = circuit.x(3);
    let _ = circuit.h(3);

    // Pattern 5: Decomposable gates
    let _ = circuit.toffoli(0, 1, 2);

    circuit
}

fn demo_optimization_levels(circuit: &Circuit<4>) {
    println!("Demo 1: Optimization Levels");
    println!("----------------------------");

    let levels = [
        OptimizationLevel::None,
        OptimizationLevel::Light,
        OptimizationLevel::Medium,
        OptimizationLevel::Heavy,
    ];

    for level in levels {
        println!("\nOptimization Level: {level:?}");

        let start = Instant::now();
        let mut optimizer = CircuitOptimizer2::<4>::with_level(level);

        match optimizer.optimize(circuit) {
            Ok(report) => {
                let duration = start.elapsed();
                println!("  Optimization time: {duration:?}");
                println!("  Initial gates: {}", report.initial_metrics.gate_count);
                println!("  Final gates: {}", report.final_metrics.gate_count);
                println!("  Gate reduction: {:.1}%", report.improvement().gate_count);
                println!("  Applied passes: {:?}", report.applied_passes);
            }
            Err(e) => {
                println!("  Optimization failed: {e:?}");
            }
        }
    }
    println!();
}

fn demo_hardware_optimization(circuit: &Circuit<4>) {
    println!("\nDemo 2: Hardware-Specific Optimization");
    println!("--------------------------------------");

    let backends = ["ibm", "google", "aws"];

    for backend in backends {
        println!("\nBackend: {backend}");

        let mut optimizer = CircuitOptimizer2::<4>::for_hardware(backend);

        match optimizer.optimize(circuit) {
            Ok(report) => {
                println!(
                    "  Initial cost: {:.2}",
                    report.initial_metrics.execution_time
                );
                println!("  Final cost: {:.2}", report.final_metrics.execution_time);
                println!(
                    "  Time reduction: {:.1}%",
                    report.improvement().execution_time
                );
                println!(
                    "  Error reduction: {:.1}%",
                    report.improvement().total_error
                );
            }
            Err(e) => {
                println!("  Optimization failed: {e:?}");
            }
        }
    }
    println!();
}

fn demo_custom_optimization(circuit: &Circuit<4>) {
    println!("\nDemo 3: Custom Optimization Pipeline");
    println!("------------------------------------");

    // Create a custom pass manager
    let mut pass_manager = PassManager::new();

    // Configure with specific passes
    let config = PassConfig {
        max_iterations: 5,
        aggressive: true,
        level: OptimizationLevel::Custom,
        ..Default::default()
    };
    pass_manager.configure(config.clone());

    // Add specific passes in a custom order
    pass_manager.add_pass(Box::new(GateCancellation::new(true)));
    pass_manager.add_pass(Box::new(RotationMerging::new(1e-10)));
    pass_manager.add_pass(Box::new(GateCommutation::new(10)));
    pass_manager.add_pass(Box::new(TemplateMatching::new()));

    // Create optimizer with custom configuration
    let mut optimizer = CircuitOptimizer2::<4>::new();
    optimizer.configure(config);

    match optimizer.optimize(circuit) {
        Ok(report) => {
            println!("Custom optimization results:");
            report.print_summary();
        }
        Err(e) => {
            println!("Custom optimization failed: {e:?}");
        }
    }
}

fn demo_gate_properties() {
    println!("\nDemo 4: Gate Properties");
    println!("-----------------------");

    // Show properties of various gates
    let gates = vec!["H", "X", "CNOT", "Toffoli"];

    for gate_name in gates {
        let props = match gate_name {
            "H" | "X" => GateProperties::single_qubit(gate_name),
            "CNOT" => GateProperties::two_qubit(gate_name),
            "Toffoli" => GateProperties::multi_qubit(gate_name, 3),
            _ => continue,
        };

        println!("\n{gate_name} Gate Properties:");
        println!("  Native: {}", props.is_native);
        println!("  Duration: {:.1} ns", props.cost.duration_ns);
        println!("  Error rate: {:.6}", props.error.error_rate);
        println!("  Self-inverse: {}", props.is_self_inverse);
        println!("  Diagonal: {}", props.is_diagonal);
        println!("  Decompositions: {}", props.decompositions.len());
    }

    // Show commutation relations
    println!("\nCommutation Relations:");
    let comm_table = CommutationTable::new();

    let gate_pairs = vec![
        ("X", "Y"),
        ("X", "Z"),
        ("Z", "Z"),
        ("Z", "RZ"),
        ("CNOT", "CNOT"),
    ];

    for (g1, g2) in gate_pairs {
        println!(
            "  {} ↔ {}: {}",
            g1,
            g2,
            if comm_table.commutes(g1, g2) {
                "✓"
            } else {
                "✗"
            }
        );
    }
    println!();
}

fn benchmark_optimization_passes(circuit: &Circuit<4>) {
    println!("\nDemo 5: Pass Benchmarking");
    println!("-------------------------");

    // Benchmark individual passes
    let passes: Vec<(&str, Box<dyn OptimizationPass>)> = vec![
        ("Gate Cancellation", Box::new(GateCancellation::new(false))),
        ("Rotation Merging", Box::new(RotationMerging::new(1e-10))),
        ("Gate Commutation", Box::new(GateCommutation::new(5))),
        ("Template Matching", Box::new(TemplateMatching::new())),
        (
            "Two-Qubit Opt",
            Box::new(TwoQubitOptimization::new(false, true)),
        ),
    ];

    let cost_model = AbstractCostModel::default();

    for (name, pass) in passes {
        let start = Instant::now();

        match pass.apply(circuit, &cost_model) {
            Ok(_) => {
                let duration = start.elapsed();
                println!("  {name}: {duration:?}");
            }
            Err(e) => {
                println!("  {name} failed: {e:?}");
            }
        }
    }

    // Benchmark full optimization
    println!("\nFull Optimization Benchmark:");
    let start = Instant::now();
    let mut optimizer = CircuitOptimizer2::<4>::with_level(OptimizationLevel::Heavy);

    match optimizer.optimize(circuit) {
        Ok(report) => {
            let duration = start.elapsed();
            println!("  Total time: {duration:?}");
            println!(
                "  Gates reduced: {} → {}",
                report.initial_metrics.gate_count, report.final_metrics.gate_count
            );

            // Show detailed report
            println!("\nDetailed Report:");
            println!("{}", report.detailed_report());
        }
        Err(e) => {
            println!("  Benchmark failed: {e:?}");
        }
    }
}

// Example of creating a custom optimization pass
struct MyCustomPass;

impl OptimizationPass for MyCustomPass {
    fn name(&self) -> &'static str {
        "My Custom Pass"
    }

    fn apply_to_gates(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        _cost_model: &dyn CostModel,
    ) -> quantrs2_core::error::QuantRS2Result<Vec<Box<dyn GateOp>>> {
        // Custom optimization logic here
        Ok(gates)
    }

    fn should_apply(&self) -> bool {
        // Custom conditions for when to apply this pass
        true
    }
}
