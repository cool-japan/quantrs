// Extended gates example
//
// This example demonstrates the use of the extended gate set implemented
// in the quantrs framework, including:
// - S-dagger (S†) and T-dagger (T†) gates
// - Square Root of X (√X) gate
// - Controlled gates (CY, CH, CS, CRX, CRY, CRZ)

use quantrs_circuit::builder::Circuit;
use quantrs_sim::statevector::StateVectorSimulator;

fn main() {
    println!("Extended Quantum Gates Example");
    println!("==============================\n");

    // Create a simple circuit to demonstrate sqrt(X) gate
    println!("EXAMPLE 1: Square Root of X Gate");
    println!("--------------------------------");
    let mut sx_circuit = Circuit::<1>::new();
    sx_circuit.sx(0).unwrap();

    // Run the circuit
    let simulator = StateVectorSimulator::new();
    let result = sx_circuit.run(simulator).unwrap();

    // Print the resulting state vector
    println!("After applying √X to |0⟩:");
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        println!(
            "|{}⟩: {} (magnitude: {:.6})",
            i,
            amplitude,
            amplitude.norm()
        );
    }

    // This should give a superposition with a phase shift
    println!("\nNotice this is different from Hadamard - it includes an imaginary component.");

    // Create a circuit to demonstrate the dagger gates
    println!("\nEXAMPLE 2: Dagger Gates (S† and T†)");
    println!("----------------------------------");

    // S gate followed by S-dagger should give identity
    let mut sdg_circuit = Circuit::<1>::new();
    sdg_circuit.s(0).unwrap().sdg(0).unwrap();

    // Run the circuit
    let simulator = StateVectorSimulator::new();
    let result = sdg_circuit.run(simulator).unwrap();

    // Print the resulting state vector
    println!("After applying S then S† to |0⟩:");
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        println!(
            "|{}⟩: {} (magnitude: {:.6})",
            i,
            amplitude,
            amplitude.norm()
        );
    }
    println!("Notice this returns to the initial state (identity operation).");

    // Create a circuit to demonstrate controlled gates
    println!("\nEXAMPLE 3: Controlled Gates");
    println!("---------------------------");

    // Create a superposition on the control qubit and apply CH
    let mut controlled_circuit = Circuit::<2>::new();
    controlled_circuit.h(0).unwrap().ch(0, 1).unwrap();

    // Run the circuit
    let simulator = StateVectorSimulator::sequential(); // Use sequential for predictable output order
    let result = controlled_circuit.run(simulator).unwrap();

    // Print the resulting state vector
    println!("After applying H to q0, then CH(q0, q1):");
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        // Format the state in binary
        let bits = format!("{:02b}", i);
        println!(
            "|{}⟩: {} (magnitude: {:.6})",
            bits,
            amplitude,
            amplitude.norm()
        );
    }

    println!("\nNotice that the CH gate only applies the H operation to q1 when q0 is |1⟩.");

    println!("\nEXAMPLE 4: Controlled Rotation Gates");
    println!("-----------------------------------");

    // Create a circuit with CRZ gate
    let mut crz_circuit = Circuit::<2>::new();
    crz_circuit
        .x(0)
        .unwrap()
        .crz(0, 1, std::f64::consts::PI / 2.0)
        .unwrap();

    // Run the circuit
    let simulator = StateVectorSimulator::sequential();
    let result = crz_circuit.run(simulator).unwrap();

    // Print the resulting state vector
    println!("After applying X to q0, then CRZ(q0, q1, π/2):");
    for (i, amplitude) in result.amplitudes().iter().enumerate() {
        // Format the state in binary
        let bits = format!("{:02b}", i);
        println!(
            "|{}⟩: {} (magnitude: {:.6})",
            bits,
            amplitude,
            amplitude.norm()
        );
    }
    println!("\nNotice the phase applied to the |11⟩ state.");

    println!("\nAll these gates are useful for implementing quantum algorithms like QAOA, VQE, and QSVM.");
}
