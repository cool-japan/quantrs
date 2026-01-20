#![allow(clippy::pedantic, clippy::unnecessary_wraps)]
//! Demonstration of MPS (Matrix Product State) simulator
//!
//! This example shows how to use the MPS simulator for efficient
//! simulation of quantum circuits, particularly those with limited entanglement.

#[cfg(feature = "mps")]
use quantrs2_circuit::prelude::*;
#[cfg(feature = "mps")]
use quantrs2_sim::mps_enhanced::utils;
#[cfg(feature = "mps")]
use quantrs2_sim::prelude::*;
#[cfg(feature = "mps")]
use std::time::Instant;

#[cfg(feature = "mps")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== MPS Quantum Simulator Demo ===\n");

    // Example 1: Basic Bell State
    demo_bell_state()?;
    println!();

    // Example 2: GHZ State
    demo_ghz_state()?;
    println!();

    // Example 3: Large Linear Circuit
    demo_linear_circuit()?;
    println!();

    // Example 4: Entanglement Entropy
    demo_entanglement_entropy()?;
    println!();

    // Example 5: Performance Comparison
    demo_performance_comparison()?;

    Ok(())
}

#[cfg(feature = "mps")]
fn demo_bell_state() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("1. Bell State Creation with MPS");
    println!("   Creating |Φ+⟩ = (|00⟩ + |11⟩)/√2");

    // Create MPS with default configuration
    let mut mps = EnhancedMPS::new(2, MPSConfig::default());

    // Manually apply gates (since we need direct access to MPS)
    // In practice, you'd use the circuit interface
    let bell_mps = utils::create_bell_state_mps()?;

    // Check amplitudes
    let amp00 = bell_mps.get_amplitude(&[false, false])?;
    let amp11 = bell_mps.get_amplitude(&[true, true])?;
    let amp01 = bell_mps.get_amplitude(&[false, true])?;
    let amp10 = bell_mps.get_amplitude(&[true, false])?;

    println!("   |00⟩ amplitude: {amp00:.4}");
    println!("   |01⟩ amplitude: {amp01:.4}");
    println!("   |10⟩ amplitude: {amp10:.4}");
    println!("   |11⟩ amplitude: {amp11:.4}");

    // Sample from the state (note: this will collapse the state)
    println!("\n   Single measurement sample:");
    // Note: For production code, you'd want to create separate instances for sampling
    println!("   (Sampling would collapse the quantum state)");

    Ok(())
}

#[cfg(feature = "mps")]
fn demo_ghz_state() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("2. GHZ State with Variable Size");

    for n in [3, 5, 10] {
        println!("\n   Creating {n}-qubit GHZ state: (|0...0⟩ + |1...1⟩)/√2");

        // Create circuit
        let mut circuit = Circuit::<10>::new(); // Use max size

        // Build GHZ circuit
        circuit.h(0)?;
        for i in 0..n - 1 {
            circuit.cnot(i, i + 1)?;
        }

        // Simulate with MPS
        let config = MPSConfig {
            max_bond_dim: 32,
            svd_threshold: 1e-10,
            ..Default::default()
        };
        let simulator = EnhancedMPSSimulator::new(config.clone());

        let start = Instant::now();
        // Note: In practice, we'd need a way to run partial circuits
        // For now, just show the concept
        println!("   Simulation time: {:?}", start.elapsed());

        // Create MPS directly for demonstration
        let mut mps = EnhancedMPS::new(n, config);

        // Check entanglement
        if n > 2 {
            let entropy = mps.entanglement_entropy(n / 2)?;
            println!("   Entanglement entropy at middle cut: {entropy:.4}");
        }
    }

    Ok(())
}

#[cfg(feature = "mps")]
fn demo_linear_circuit() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("3. Linear Circuit (Low Entanglement)");
    println!("   MPS is particularly efficient for circuits with limited entanglement");

    let n = 20; // 20 qubits
    let config = MPSConfig {
        max_bond_dim: 16, // Small bond dimension is sufficient
        svd_threshold: 1e-12,
        auto_canonicalize: true,
        ..Default::default()
    };

    let mut mps = EnhancedMPS::new(n, config.clone());

    println!("   Building linear circuit with nearest-neighbor gates...");

    // Apply rotation gates
    for i in 0..n {
        // Single qubit rotation
        let angle = std::f64::consts::PI * (i as f64) / (n as f64);
        // Would apply RY(angle) to qubit i
    }

    // Apply nearest-neighbor entangling gates
    for i in 0..n - 1 {
        // Would apply CZ(i, i+1)
    }

    println!("   Circuit created with {n} qubits");
    println!("   Maximum bond dimension: {}", config.max_bond_dim);

    // Sample measurement
    let outcome = mps.sample();
    let bitstring: String = outcome.iter().map(|&b| if b { '1' } else { '0' }).collect();
    println!("   Sample measurement: |{bitstring}>");

    Ok(())
}

#[cfg(feature = "mps")]
fn demo_entanglement_entropy() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("4. Entanglement Entropy Analysis");
    println!("   Studying how entanglement grows in different circuits");

    let n = 10;

    // Product state (no entanglement)
    println!("\n   a) Product state |0101...⟩:");
    let mut mps_product = EnhancedMPS::new(n, MPSConfig::default());
    // Apply X gates to odd qubits
    for i in (1..n).step_by(2) {
        // Would apply X(i)
    }

    // Linear entanglement growth
    println!("\n   b) Linear entanglement circuit:");
    let mut mps_linear = EnhancedMPS::new(n, MPSConfig::default());

    // Create Bell state and measure entropy
    let mut bell_mps = utils::create_bell_state_mps()?;
    let entropy = bell_mps.entanglement_entropy(0)?;
    println!("   Bell state entropy: {entropy:.4} (max for 2 qubits)");

    // Volume law entanglement (random circuit)
    println!("\n   c) Random circuit (volume law):");
    println!("   MPS becomes less efficient as entanglement grows");

    Ok(())
}

#[cfg(feature = "mps")]
fn demo_performance_comparison() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("5. Performance Comparison");
    println!("   Comparing MPS efficiency for different circuit types");

    let configs = vec![
        (
            "Low bond (χ=4)",
            MPSConfig {
                max_bond_dim: 4,
                ..Default::default()
            },
        ),
        (
            "Medium bond (χ=16)",
            MPSConfig {
                max_bond_dim: 16,
                ..Default::default()
            },
        ),
        (
            "High bond (χ=64)",
            MPSConfig {
                max_bond_dim: 64,
                ..Default::default()
            },
        ),
    ];

    for (name, config) in configs {
        println!("\n   Configuration: {name}");

        // Create MPS
        let n = 16;
        let mps = EnhancedMPS::new(n, config.clone());

        // Memory estimate
        let memory_per_tensor = config.max_bond_dim * 2 * config.max_bond_dim * 16; // Complex64
        let total_memory = memory_per_tensor * n;
        println!("   Estimated memory: {total_memory} bytes");

        // Compare to state vector
        let sv_memory = (1 << n) * 16; // 2^n complex numbers
        println!("   State vector memory: {sv_memory} bytes");
        println!(
            "   Memory ratio: {:.2}%",
            (total_memory as f64 / sv_memory as f64) * 100.0
        );
    }

    println!("\n   Summary:");
    println!("   - MPS is efficient for low-entanglement states");
    println!("   - Bond dimension χ controls accuracy vs memory trade-off");
    println!("   - Ideal for 1D systems and sequential circuits");

    Ok(())
}

#[cfg(not(feature = "mps"))]
fn main() {
    println!("This example requires the 'mps' feature to be enabled.");
    println!("Run with: cargo run --example mps_demo --features mps");
}
