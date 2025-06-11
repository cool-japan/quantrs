//! Demonstration of tensor network compression for quantum circuits
//!
//! This example shows how to use tensor network methods to compress
//! and analyze quantum circuits efficiently.

use nalgebra::Complex;
use quantrs2_circuit::prelude::*;
use quantrs2_core::gate::multi::CNOT;
use quantrs2_core::gate::single::{Hadamard, RotationZ, T};
use quantrs2_core::qubit::QubitId;

type C64 = Complex<f64>;

fn main() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("=== Tensor Network Compression Demo ===\n");

    demo_basic_tensor_network()?;
    demo_circuit_compression()?;
    demo_mps_representation()?;
    demo_compression_methods()?;

    Ok(())
}

fn demo_basic_tensor_network() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Basic Tensor Network Construction ---");

    // Create simple tensors
    let identity = Tensor::identity(2, "in".to_string(), "out".to_string());
    println!(
        "Created identity tensor: rank={}, size={}",
        identity.rank(),
        identity.size()
    );

    // Create Hadamard tensor
    let h_data = vec![
        C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        C64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let h_tensor = Tensor::new(
        h_data,
        vec![2, 2],
        vec!["h_in".to_string(), "h_out".to_string()],
    );

    // Build tensor network
    let mut tn = TensorNetwork::new();
    let id_idx = tn.add_tensor(identity);
    let h_idx = tn.add_tensor(h_tensor);

    // Connect tensors
    tn.add_bond(id_idx, "out".to_string(), h_idx, "h_in".to_string())?;

    println!("Built tensor network with {} tensors and {} bonds", 2, 1);
    println!();

    Ok(())
}

fn demo_circuit_compression() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Circuit Compression ---");

    // Create a circuit with repetitive structure
    let mut circuit = Circuit::<4>::new();

    // Add many gates
    for i in 0..3 {
        circuit.add_gate(Hadamard { target: QubitId(i) })?;
    }

    for i in 0..3 {
        circuit.add_gate(CNOT {
            control: QubitId(i),
            target: QubitId(i + 1),
        })?;
    }

    for i in 0..4 {
        circuit.add_gate(T { target: QubitId(i) })?;
    }

    for i in (1..4).rev() {
        circuit.add_gate(CNOT {
            control: QubitId(i - 1),
            target: QubitId(i),
        })?;
    }

    println!("Original circuit: {} gates", circuit.num_gates());

    // Compress using tensor networks
    let compressor = TensorNetworkCompressor::new(16); // max bond dimension
    let compressed = compressor.compress(&circuit)?;

    println!(
        "Compression ratio: {:.2}%",
        compressed.compression_ratio() * 100.0
    );

    // Check fidelity
    let fidelity = compressed.fidelity(&circuit)?;
    println!("Fidelity with original: {:.6}", fidelity);

    println!();
    Ok(())
}

fn demo_mps_representation() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Matrix Product State Representation ---");

    // Create a circuit that generates an interesting entangled state
    let mut circuit = Circuit::<6>::new();

    // Create W state: (|100000⟩ + |010000⟩ + |001000⟩ + |000100⟩ + |000010⟩ + |000001⟩)/√6
    circuit.add_gate(Hadamard { target: QubitId(0) })?;
    circuit.add_gate(RotationZ {
        target: QubitId(0),
        theta: std::f64::consts::PI / 3.0,
    })?;

    for i in 0..5 {
        circuit.add_gate(CNOT {
            control: QubitId(i),
            target: QubitId(i + 1),
        })?;
    }

    println!("Created circuit for W state preparation");

    // Convert to MPS
    let mps = MatrixProductState::from_circuit(&circuit)?;
    println!("Converted to MPS representation");

    // Compress with different bond dimensions
    let bond_dims = vec![2, 4, 8, 16];

    for &max_bond in &bond_dims {
        let mut mps_copy = MatrixProductState::from_circuit(&circuit)?;
        mps_copy.compress(max_bond, 1e-10)?;

        // In a real implementation, would calculate actual compression metrics
        println!("Max bond dimension {}: compression successful", max_bond);
    }

    println!();
    Ok(())
}

fn demo_compression_methods() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Different Compression Methods ---");

    let mut circuit = Circuit::<5>::new();

    // Build a deep circuit
    for _ in 0..5 {
        for i in 0..5 {
            circuit.add_gate(Hadamard { target: QubitId(i) })?;
        }
        for i in 0..4 {
            circuit.add_gate(CNOT {
                control: QubitId(i),
                target: QubitId(i + 1),
            })?;
        }
    }

    println!("Built deep circuit with {} gates", circuit.num_gates());

    // Test different compression methods
    let methods = vec![
        CompressionMethod::SVD,
        CompressionMethod::DMRG,
        CompressionMethod::TEBD,
    ];

    for method in methods {
        let compressor = TensorNetworkCompressor::new(32).with_method(method.clone());

        let compressed = compressor.compress(&circuit)?;

        println!("\n{:?} compression:", method);
        println!(
            "  Compression ratio: {:.2}%",
            compressed.compression_ratio() * 100.0
        );

        // Try to decompress
        let decompressed = compressed.decompress()?;
        println!("  Decompressed to {} gates", decompressed.num_gates());
    }

    println!();
    Ok(())
}

fn demo_tensor_contraction() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Tensor Contraction Optimization ---");

    // Create a circuit with specific structure
    let mut circuit = Circuit::<4>::new();

    // Layer 1: Single-qubit gates
    for i in 0..4 {
        circuit.add_gate(Hadamard { target: QubitId(i) })?;
    }

    // Layer 2: Entangling gates
    circuit.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;
    circuit.add_gate(CNOT {
        control: QubitId(2),
        target: QubitId(3),
    })?;

    // Layer 3: Cross entangling
    circuit.add_gate(CNOT {
        control: QubitId(1),
        target: QubitId(2),
    })?;

    // Convert to tensor network
    let converter = CircuitToTensorNetwork::<4>::new()
        .with_max_bond_dim(8)
        .with_tolerance(1e-12);

    let tn = converter.convert(&circuit)?;

    println!("Converted circuit to tensor network");
    println!("Network has {} tensors", circuit.num_gates());

    // Contract the network
    let result = tn.contract_all()?;
    println!("Contracted to single tensor of rank {}", result.rank());

    println!();
    Ok(())
}

fn demo_circuit_analysis() -> quantrs2_core::error::QuantRS2Result<()> {
    println!("--- Circuit Analysis via Tensor Networks ---");

    // Create circuits to compare
    let mut circuit1 = Circuit::<3>::new();
    circuit1.add_gate(Hadamard { target: QubitId(0) })?;
    circuit1.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;
    circuit1.add_gate(CNOT {
        control: QubitId(1),
        target: QubitId(2),
    })?;

    let mut circuit2 = Circuit::<3>::new();
    circuit2.add_gate(Hadamard { target: QubitId(0) })?;
    circuit2.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(2),
    })?;
    circuit2.add_gate(CNOT {
        control: QubitId(0),
        target: QubitId(1),
    })?;

    // Convert to MPS for efficient comparison
    let mps1 = MatrixProductState::from_circuit(&circuit1)?;
    let mps2 = MatrixProductState::from_circuit(&circuit2)?;

    // Calculate overlap (would indicate circuit similarity)
    let overlap = mps1.overlap(&mps2)?;
    println!("Circuit overlap: |⟨ψ₁|ψ₂⟩| = {:.6}", overlap.norm());

    // Compress both circuits
    let compressor = TensorNetworkCompressor::new(16);
    let comp1 = compressor.compress(&circuit1)?;
    let comp2 = compressor.compress(&circuit2)?;

    println!(
        "Circuit 1 compression: {:.2}%",
        comp1.compression_ratio() * 100.0
    );
    println!(
        "Circuit 2 compression: {:.2}%",
        comp2.compression_ratio() * 100.0
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_network_demo() {
        assert!(main().is_ok());
    }
}
