//! Demonstration of noise-aware circuit optimization
//!
//! This example shows how to use noise models and noise-aware optimization
//! to improve quantum circuit fidelity on noisy devices.

use quantrs2_circuit::optimization::passes::OptimizationPassExt;
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::{multi::CNOT, single::Hadamard};
use quantrs2_core::qubit::QubitId;

fn main() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("=== Noise-Aware Circuit Optimization Demo ===\n");

    // Create a test circuit
    let mut circuit = Circuit::<4>::new();
    circuit.add_gate(Hadamard { target: QubitId(0) })?;
    circuit.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;
    circuit.add_gate(Hadamard { target: QubitId(2) })?;
    circuit.add_gate(CNOT {
        control: QubitId(2),
        target: QubitId(3),
    })?;
    circuit.add_gate(CNOT {
        control: QubitId(1),
        target: QubitId(2),
    })?;

    println!("Original circuit:");
    println!("H(0), CNOT(0,1), H(2), CNOT(2,3), CNOT(1,2)");
    println!("Total gates: {}\n", circuit.num_gates());

    // Test different noise models
    demo_uniform_noise(&circuit)?;
    demo_ibm_noise(&circuit)?;
    demo_noise_aware_cost_model(&circuit);
    demo_noise_optimization_passes(&circuit)?;

    Ok(())
}

fn demo_uniform_noise(circuit: &Circuit<4>) -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Uniform Noise Model ---");

    let noise_model = NoiseModel::uniform(4);
    let optimizer = NoiseAwareOptimizer::new(noise_model.clone());

    println!("Noise characteristics:");
    println!(
        "  Single-qubit error rate: {:.2e}",
        noise_model.single_qubit_error(0)
    );
    println!(
        "  Two-qubit error rate: {:.2e}",
        noise_model.two_qubit_error(0, 1)
    );
    println!("  T1 time: {:.1} μs", noise_model.t1_time(0));
    println!("  T2 time: {:.1} μs", noise_model.t2_time(0));
    println!("  CNOT gate time: {:.1} ns", noise_model.gate_time("CNOT"));

    let original_fidelity = optimizer.estimate_fidelity(circuit);
    println!("\nOriginal circuit fidelity: {original_fidelity:.4}");

    let optimized = optimizer.optimize(circuit)?;
    let optimized_fidelity = optimizer.estimate_fidelity(&optimized);
    println!("Optimized circuit fidelity: {optimized_fidelity:.4}");

    if optimized_fidelity > original_fidelity {
        println!(
            "✓ Fidelity improved by {:.4}",
            optimized_fidelity - original_fidelity
        );
    } else {
        println!("→ No fidelity improvement (circuit already optimal)");
    }

    println!();
    Ok(())
}

fn demo_ibm_noise(circuit: &Circuit<4>) -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- IBM-like Noise Model ---");

    let noise_model = NoiseModel::ibm_like(4);
    let coupling_map = CouplingMap::linear(4);
    let optimizer = NoiseAwareOptimizer::new(noise_model.clone()).with_coupling_map(coupling_map);

    println!("IBM-like noise characteristics:");
    println!(
        "  Single-qubit error rate: {:.2e}",
        noise_model.single_qubit_error(0)
    );
    println!(
        "  Two-qubit error rate (adjacent): {:.2e}",
        noise_model.two_qubit_error(0, 1)
    );
    println!("  Hadamard gate time: {:.1} ns", noise_model.gate_time("H"));
    println!("  CNOT gate time: {:.1} ns", noise_model.gate_time("CNOT"));

    let original_fidelity = optimizer.estimate_fidelity(circuit);
    println!("\nOriginal circuit fidelity: {original_fidelity:.4}");

    let optimized = optimizer.optimize(circuit)?;
    let optimized_fidelity = optimizer.estimate_fidelity(&optimized);
    println!("Optimized circuit fidelity: {optimized_fidelity:.4}");

    println!("Available optimization passes:");
    for pass in optimizer.get_passes() {
        println!("  - {}", pass.name());
    }

    println!();
    Ok(())
}

fn demo_noise_aware_cost_model(circuit: &Circuit<4>) {
    println!("--- Noise-Aware Cost Analysis ---");

    let uniform_noise = NoiseModel::uniform(4);
    let ibm_noise = NoiseModel::ibm_like(4);

    let uniform_cost_model = NoiseAwareCostModel::new(uniform_noise);
    let ibm_cost_model = NoiseAwareCostModel::new(ibm_noise);

    let uniform_cost = uniform_cost_model.circuit_cost(circuit);
    let ibm_cost = ibm_cost_model.circuit_cost(circuit);

    println!("Circuit costs with different noise models:");
    println!("  Uniform noise model: {uniform_cost:.2}");
    println!("  IBM-like noise model: {ibm_cost:.2}");

    // Analyze individual gate costs
    println!("\nGate-by-gate cost analysis (IBM model):");
    for (i, gate) in circuit.gates().iter().enumerate() {
        let gate_cost = ibm_cost_model.gate_cost(gate.as_ref());
        println!("  Gate {}: {} - Cost: {:.2}", i, gate.name(), gate_cost);
    }

    println!();
}

fn demo_noise_optimization_passes(
    circuit: &Circuit<4>,
) -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Individual Optimization Passes ---");

    let noise_model = NoiseModel::ibm_like(4);
    let coupling_map = CouplingMap::linear(4);

    // Test coherence optimization
    let coherence_opt = CoherenceOptimization::new(noise_model.clone());
    let cost_model = NoiseAwareCostModel::new(noise_model.clone());

    if coherence_opt.should_apply() {
        let coherence_result = coherence_opt.apply(circuit, &cost_model)?;
        println!("✓ Coherence optimization applied");
        println!("  Original gates: {}", circuit.num_gates());
        println!("  After coherence opt: {}", coherence_result.num_gates());
    }

    // Test noise-aware mapping
    let mapping_opt = NoiseAwareMapping::new(noise_model.clone(), coupling_map);
    if mapping_opt.should_apply() {
        let mapping_result = mapping_opt.apply(circuit, &cost_model)?;
        println!("✓ Noise-aware mapping applied");
        println!("  Original gates: {}", circuit.num_gates());
        println!("  After mapping opt: {}", mapping_result.num_gates());
    }

    // Test dynamical decoupling
    let dd_opt = DynamicalDecoupling::new(noise_model);
    if dd_opt.should_apply() {
        let dd_result = dd_opt.apply(circuit, &cost_model)?;
        println!("✓ Dynamical decoupling applied");
        println!("  Original gates: {}", circuit.num_gates());
        println!("  After DD insertion: {}", dd_result.num_gates());
    }

    println!();
    Ok(())
}

fn demo_fidelity_comparison() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Fidelity Comparison Across Noise Models ---");

    // Create circuits of different complexity
    let circuits = [
        create_simple_circuit()?,
        create_medium_circuit()?,
        create_complex_circuit()?,
    ];

    let noise_models = vec![
        ("Ideal", NoiseModel::new()),
        ("Uniform", NoiseModel::uniform(4)),
        ("IBM-like", NoiseModel::ibm_like(4)),
    ];

    println!(
        "{:<12} {:<8} {:<8} {:<8}",
        "Circuit", "Ideal", "Uniform", "IBM-like"
    );
    println!("{:-<40}", "");

    for (i, circuit) in circuits.iter().enumerate() {
        let circuit_name = match i {
            0 => "Simple",
            1 => "Medium",
            2 => "Complex",
            _ => "Unknown",
        };

        print!("{circuit_name:<12}");

        for (_, noise_model) in &noise_models {
            let optimizer = NoiseAwareOptimizer::new(noise_model.clone());
            let fidelity = optimizer.estimate_fidelity(circuit);
            print!(" {fidelity:<8.4}");
        }
        println!();
    }

    println!();
    Ok(())
}

fn create_simple_circuit() -> quantrs2_core::error::QuantRS2Result<Circuit<4>> {
    let mut circuit = Circuit::<4>::new();
    circuit.add_gate(Hadamard { target: QubitId(0) })?;
    circuit.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;
    Ok(circuit)
}

fn create_medium_circuit() -> quantrs2_core::error::QuantRS2Result<Circuit<4>> {
    let mut circuit = Circuit::<4>::new();
    circuit.add_gate(Hadamard { target: QubitId(0) })?;
    circuit.add_gate(Hadamard { target: QubitId(1) })?;
    circuit.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;
    circuit.add_gate(CNOT {
        control: QubitId(1),
        target: QubitId(2),
    })?;
    circuit.add_gate(CNOT {
        control: QubitId(2),
        target: QubitId(3),
    })?;
    Ok(circuit)
}

fn create_complex_circuit() -> quantrs2_core::error::QuantRS2Result<Circuit<4>> {
    let mut circuit = Circuit::<4>::new();
    for i in 0..4 {
        circuit.add_gate(Hadamard { target: QubitId(i) })?;
    }
    for i in 0..3 {
        circuit.add_gate(CNOT {
            control: QubitId(i),
            target: QubitId(i + 1),
        })?;
    }
    for i in 0..4 {
        circuit.add_gate(Hadamard { target: QubitId(i) })?;
    }
    for i in 0..3 {
        circuit.add_gate(CNOT {
            control: QubitId(i + 1),
            target: QubitId(i),
        })?;
    }
    Ok(circuit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_optimization_demo() {
        let result = main();
        assert!(result.is_ok());
    }

    #[test]
    fn test_circuit_creation() {
        let simple = create_simple_circuit();
        let medium = create_medium_circuit();
        let complex = create_complex_circuit();

        assert!(simple.is_ok());
        assert!(medium.is_ok());
        assert!(complex.is_ok());

        assert!(complex.unwrap().num_gates() > medium.unwrap().num_gates());
    }
}
