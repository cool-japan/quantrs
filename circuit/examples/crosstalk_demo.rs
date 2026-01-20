#![allow(clippy::unnecessary_wraps, clippy::pedantic)]
//! Demonstration of cross-talk aware scheduling
//!
//! This example shows how to optimize circuit execution by considering
//! unwanted interactions between qubits during parallel gate execution.

use quantrs2_circuit::crosstalk::CrosstalkCharacterization;
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::multi::{CNOT, CZ};
use quantrs2_core::gate::single::{Hadamard, PauliX, RotationZ};
use quantrs2_core::qubit::QubitId;

fn main() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("=== Cross-talk Aware Scheduling Demo ===\n");

    demo_crosstalk_model()?;
    demo_scheduling_strategies()?;
    demo_crosstalk_analysis()?;
    demo_device_specific_scheduling()?;

    Ok(())
}

fn demo_crosstalk_model() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Cross-talk Model ---");

    // Create a uniform crosstalk model
    let model = CrosstalkModel::uniform(5, 0.05);

    println!("Created crosstalk model for 5-qubit device");
    println!("Base crosstalk rate: 0.05");
    println!("Threshold for significant crosstalk: {}", model.threshold);

    // Test crosstalk between different gate pairs
    let h0 = Hadamard { target: QubitId(0) };
    let h1 = Hadamard { target: QubitId(1) };
    let h2 = Hadamard { target: QubitId(2) };

    let cnot01 = CNOT {
        control: QubitId(0),
        target: QubitId(1),
    };
    let cnot23 = CNOT {
        control: QubitId(2),
        target: QubitId(3),
    };

    println!("\nCrosstalk values:");
    println!("  H(0) || H(1): {:.4}", model.get_crosstalk(&h0, &h1));
    println!("  H(0) || H(2): {:.4}", model.get_crosstalk(&h0, &h2));
    println!(
        "  CNOT(0,1) || CNOT(2,3): {:.4}",
        model.get_crosstalk(&cnot01, &cnot23)
    );
    println!(
        "  H(0) || H(0): {:.4} (same qubit)",
        model.get_crosstalk(&h0, &h0)
    );

    // Check parallelization decisions
    println!("\nCan parallelize:");
    println!("  H(0) || H(2): {}", model.can_parallelize(&h0, &h2));
    println!(
        "  CNOT(0,1) || CNOT(2,3): {}",
        model.can_parallelize(&cnot01, &cnot23)
    );

    println!();
    Ok(())
}

fn demo_scheduling_strategies() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Scheduling Strategies ---");

    // Create a test circuit
    let mut circuit = Circuit::<5>::new();

    // Layer 1: Single-qubit gates
    for i in 0..5 {
        circuit.add_gate(Hadamard { target: QubitId(i) })?;
    }

    // Layer 2: Two-qubit gates
    circuit.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;
    circuit.add_gate(CNOT {
        control: QubitId(2),
        target: QubitId(3),
    })?;
    circuit.add_gate(CZ {
        control: QubitId(1),
        target: QubitId(2),
    })?;

    // Layer 3: More single-qubit gates
    for i in 0..5 {
        circuit.add_gate(RotationZ {
            target: QubitId(i),
            theta: std::f64::consts::PI / 4.0,
        })?;
    }

    println!("Test circuit: {} gates on 5 qubits", circuit.num_gates());

    let model = CrosstalkModel::uniform(5, 0.05);

    // Strategy 1: Minimize crosstalk
    let scheduler1 =
        CrosstalkScheduler::new(model.clone()).with_strategy(SchedulingStrategy::MinimizeCrosstalk);

    let schedule1 = scheduler1.schedule(&circuit)?;

    println!("\nMinimize Crosstalk Strategy:");
    println!("  Time slices: {}", schedule1.time_slices.len());
    println!("  Total crosstalk: {:.4}", schedule1.total_crosstalk);
    println!("  Execution time: {:.1} ns", schedule1.execution_time);

    // Strategy 2: Minimize time with constraint
    let scheduler2 =
        CrosstalkScheduler::new(model.clone()).with_strategy(SchedulingStrategy::MinimizeTime {
            max_crosstalk: 0.02,
        });

    let schedule2 = scheduler2.schedule(&circuit)?;

    println!("\nMinimize Time Strategy (max crosstalk = 0.02):");
    println!("  Time slices: {}", schedule2.time_slices.len());
    println!("  Total crosstalk: {:.4}", schedule2.total_crosstalk);
    println!("  Execution time: {:.1} ns", schedule2.execution_time);

    // Strategy 3: Balanced
    let scheduler3 = CrosstalkScheduler::new(model).with_strategy(SchedulingStrategy::Balanced {
        time_weight: 0.5,
        crosstalk_weight: 0.5,
    });

    let schedule3 = scheduler3.schedule(&circuit)?;

    println!("\nBalanced Strategy:");
    println!("  Time slices: {}", schedule3.time_slices.len());
    println!("  Total crosstalk: {:.4}", schedule3.total_crosstalk);
    println!("  Execution time: {:.1} ns", schedule3.execution_time);

    // Show time slice details for first schedule
    println!("\nTime slice details (MinimizeCrosstalk):");
    for (i, slice) in schedule1.time_slices.iter().enumerate() {
        println!(
            "  Slice {}: {} gates, max crosstalk = {:.4}, duration = {:.1} ns",
            i,
            slice.gates.len(),
            slice.max_crosstalk,
            slice.duration
        );
    }

    println!();
    Ok(())
}

fn demo_crosstalk_analysis() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Cross-talk Analysis ---");

    // Create a circuit with potential crosstalk issues
    let mut circuit = Circuit::<6>::new();

    // Parallel two-qubit gates on neighboring qubits
    circuit.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;
    circuit.add_gate(CNOT {
        control: QubitId(2),
        target: QubitId(3),
    })?;
    circuit.add_gate(CNOT {
        control: QubitId(4),
        target: QubitId(5),
    })?;

    // More gates that might have crosstalk
    circuit.add_gate(CZ {
        control: QubitId(1),
        target: QubitId(2),
    })?;
    circuit.add_gate(CZ {
        control: QubitId(3),
        target: QubitId(4),
    })?;

    let model = CrosstalkModel::uniform(6, 0.05);
    let analyzer = CrosstalkAnalyzer::new(model);

    let analysis = analyzer.analyze(&circuit);

    println!("Circuit analysis:");
    println!("  Total gates: {}", analysis.total_gates);
    println!("  Maximum crosstalk: {:.4}", analysis.max_crosstalk);
    println!("  Problematic pairs: {}", analysis.problematic_pairs.len());

    println!("\nTop problematic gate pairs:");
    for (i, (g1, g2, crosstalk)) in analysis.problematic_pairs.iter().take(3).enumerate() {
        println!(
            "  {}. Gates {} and {}: crosstalk = {:.4}",
            i + 1,
            g1,
            g2,
            crosstalk
        );
    }

    // Get reordering suggestions
    let suggestions = analyzer.suggest_reordering(&circuit)?;

    println!("\nSuggested reorderings:");
    for suggestion in suggestions.iter().take(3) {
        println!(
            "  Move gates {} and {} apart: {}",
            suggestion.gate1, suggestion.gate2, suggestion.reason
        );
        println!(
            "    Expected improvement: {:.4}",
            suggestion.expected_improvement
        );
    }

    println!();
    Ok(())
}

fn demo_device_specific_scheduling() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Device-Specific Scheduling ---");

    // Create a realistic device characterization
    let mut characterization = CrosstalkCharacterization {
        measured_crosstalk: std::collections::HashMap::new(),
        single_qubit_measurements: std::collections::HashMap::new(),
        significance_threshold: 0.01,
        device_coupling: vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4), // Linear connectivity
            (0, 5),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5), // Star center at qubit 5
        ],
    };

    // Add some measured crosstalk values
    characterization
        .single_qubit_measurements
        .insert((0, 1), 0.02);
    characterization
        .single_qubit_measurements
        .insert((1, 2), 0.02);
    characterization
        .single_qubit_measurements
        .insert((0, 5), 0.03);
    characterization
        .single_qubit_measurements
        .insert((1, 5), 0.03);

    characterization
        .measured_crosstalk
        .insert(((0, 1), (2, 3)), 0.05);
    characterization
        .measured_crosstalk
        .insert(((0, 1), (3, 4)), 0.02);
    characterization
        .measured_crosstalk
        .insert(((0, 5), (1, 5)), 0.08);

    let model = CrosstalkModel::from_characterization(&characterization);
    let scheduler = CrosstalkScheduler::new(model);

    // Create circuit that uses the device topology
    let mut circuit = Circuit::<6>::new();

    // Operations on linear chain
    circuit.add_gate(Hadamard { target: QubitId(0) })?;
    circuit.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;
    circuit.add_gate(CNOT {
        control: QubitId(1),
        target: QubitId(2),
    })?;

    // Operations using star center
    circuit.add_gate(Hadamard { target: QubitId(5) })?;
    circuit.add_gate(CZ {
        control: QubitId(5),
        target: QubitId(0),
    })?;
    circuit.add_gate(CZ {
        control: QubitId(5),
        target: QubitId(3),
    })?;

    let schedule = scheduler.schedule(&circuit)?;

    println!("Device-specific schedule:");
    println!("  Device topology: Linear chain + star center");
    println!("  Time slices: {}", schedule.time_slices.len());
    println!("  Total crosstalk: {:.4}", schedule.total_crosstalk);

    println!("\nSchedule details:");
    for (i, slice) in schedule.time_slices.iter().enumerate() {
        let gates = &circuit.gates();
        print!("  Slice {i}: ");
        for &gate_idx in &slice.gates {
            print!("{} ", gates[gate_idx].name());
        }
        println!("(crosstalk: {:.4})", slice.max_crosstalk);
    }

    println!();
    Ok(())
}

fn demo_adaptive_scheduling() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Adaptive Scheduling ---");

    // Create a circuit with varying gate densities
    let mut circuit = Circuit::<8>::new();

    // Dense region: many gates on qubits 0-3
    for _ in 0..3 {
        for i in 0..3 {
            circuit.add_gate(Hadamard { target: QubitId(i) })?;
            circuit.add_gate(RotationZ {
                target: QubitId(i),
                theta: 0.1,
            })?;
        }
        circuit.add_gate(CNOT {
            control: QubitId(0),
            target: QubitId(1),
        })?;
        circuit.add_gate(CNOT {
            control: QubitId(2),
            target: QubitId(3),
        })?;
    }

    // Sparse region: few gates on qubits 4-7
    circuit.add_gate(Hadamard { target: QubitId(4) })?;
    circuit.add_gate(CNOT {
        control: QubitId(4),
        target: QubitId(5),
    })?;
    circuit.add_gate(PauliX { target: QubitId(7) })?;

    println!("Circuit with dense (qubits 0-3) and sparse (qubits 4-7) regions");
    println!("Total gates: {}", circuit.num_gates());

    let model = CrosstalkModel::uniform(8, 0.05);
    let analyzer = CrosstalkAnalyzer::new(model.clone());

    // Analyze before scheduling
    let analysis = analyzer.analyze(&circuit);
    println!("\nCrosstalk analysis:");
    println!(
        "  Problematic pairs in dense region: {}",
        analysis
            .problematic_pairs
            .iter()
            .filter(|(g1, g2, _)| {
                let gates = circuit.gates();
                let q1 = gates[*g1].qubits()[0].id();
                let q2 = gates[*g2].qubits()[0].id();
                q1 < 4 && q2 < 4
            })
            .count()
    );

    // Schedule with adaptive strategy
    let scheduler = CrosstalkScheduler::new(model);
    let schedule = scheduler.schedule(&circuit)?;

    println!("\nAdaptive scheduling results:");
    println!("  Time slices: {}", schedule.time_slices.len());
    println!(
        "  Average gates per slice: {:.1}",
        circuit.num_gates() as f64 / schedule.time_slices.len() as f64
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crosstalk_demo() {
        assert!(main().is_ok());
    }
}
