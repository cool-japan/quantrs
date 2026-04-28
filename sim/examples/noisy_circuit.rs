//! Noisy Circuit Simulation — Bell State with Depolarizing Noise
//!
//! Demonstrates how depolarizing noise degrades a Bell state:
//!   |Φ+⟩ = (|00⟩ + |11⟩)/√2
//!
//! The `DepolarizingChannel` in quantrs2-sim uses a stochastic (Monte Carlo)
//! approach: with probability p one of X, Y, Z is applied to the target qubit.
//! A single shot may look ideal; averaging many shots reveals the degradation.
//!
//! With increasing noise, the average P(|00⟩) + P(|11⟩) decreases from 1 to ~0.5
//! (maximum mixed state over 4 basis states).
//!
//! Run with:
//!   cargo run --example noisy_circuit -p quantrs2-sim --all-features

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::{
    noise::{DepolarizingChannel, NoiseModel},
    statevector::StateVectorSimulator,
};

/// Run `shots` noisy simulations and return average probabilities for |00⟩ and |11⟩
fn noisy_bell_fidelity(noise_p: f64, shots: usize) -> Result<f64, Box<dyn std::error::Error>> {
    // Build a 2-qubit Bell circuit (reused across shots)
    let mut circuit = Circuit::<2>::new();
    circuit.h(0)?;
    circuit.cnot(0, 1)?;

    let mut sum_fidelity = 0.0f64;

    for _ in 0..shots {
        // Build a fresh noise model for each shot (fastrand inside makes it stochastic)
        let mut noise = NoiseModel::new(true); // per_gate = true
        noise.add_depolarizing(DepolarizingChannel {
            target: QubitId(0),
            probability: noise_p,
        });
        noise.add_depolarizing(DepolarizingChannel {
            target: QubitId(1),
            probability: noise_p,
        });

        let sim = StateVectorSimulator::with_noise(noise);
        let reg = sim.run(&circuit)?;
        let p = reg.probabilities();
        // Bell fidelity proxy: P(|00⟩) + P(|11⟩)  (= 1 for ideal Bell state)
        sum_fidelity += p[0] + p[3];
    }

    Ok(sum_fidelity / shots as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Noisy Bell State Simulation — Depolarizing Channel ===");
    println!("(stochastic noise model; averaging over shots)\n");

    let shots = 200usize;
    println!("Shots per noise level: {shots}\n");
    println!("  {:>8}  {:>18}", "noise_p", "avg[P(00)+P(11)]");
    println!("  {}", "-".repeat(30));

    let noise_levels = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50];
    let mut fidelities = Vec::new();
    for &p in &noise_levels {
        let f = noisy_bell_fidelity(p, shots)?;
        fidelities.push(f);
        let bar_len = (f * 30.0).round() as usize;
        let bar: String = "#".repeat(bar_len);
        println!("  {p:>8.2}  {f:>8.4}  {bar}");
    }

    println!();

    // ---- Assertions ----
    // At p=0: fidelity should be 1
    let f0 = fidelities[0];
    assert!(
        (f0 - 1.0).abs() < 1e-10,
        "Noiseless fidelity should be 1.0, got {f0}"
    );
    println!("Noiseless fidelity: {f0:.6}  — OK");

    // At p=0.5: strong noise should push fidelity below 0.75
    let f_high = *fidelities.last().expect("non-empty");
    assert!(
        f_high < 0.80,
        "At p=0.5, fidelity should degrade significantly (got {f_high})"
    );
    println!("High-noise fidelity (p=0.5): {f_high:.4}  — OK (degraded as expected)");

    // Monotonically non-increasing (allow small statistical fluctuations)
    for i in 0..fidelities.len() - 1 {
        assert!(
            fidelities[i] >= fidelities[i + 1] - 0.05,
            "Fidelity should not increase with noise: f[{i}]={:.4} < f[{}]={:.4}",
            fidelities[i],
            i + 1,
            fidelities[i + 1]
        );
    }
    println!("Fidelity decreases monotonically with noise — OK");

    // ---- 4-qubit noisy register: two Bell pairs ----
    println!("\n--- 4-qubit noisy register (two Bell pairs, p=0.05, 100 shots) ---");
    let shots4 = 100usize;
    let mut avg = [0.0f64; 16];

    let mut circuit4 = Circuit::<4>::new();
    circuit4.h(0)?;
    circuit4.cnot(0, 1)?;
    circuit4.h(2)?;
    circuit4.cnot(2, 3)?;

    let p4 = 0.05f64;
    for _ in 0..shots4 {
        let mut noise4 = NoiseModel::new(true);
        for q in 0..4u32 {
            noise4.add_depolarizing(DepolarizingChannel {
                target: QubitId(q),
                probability: p4,
            });
        }
        let reg = StateVectorSimulator::with_noise(noise4).run(&circuit4)?;
        let probs = reg.probabilities();
        for (k, &prob) in probs.iter().enumerate() {
            avg[k] += prob;
        }
    }
    for v in &mut avg {
        *v /= shots4 as f64;
    }

    // Expected dominant states: |0000⟩, |0011⟩, |1100⟩, |1111⟩ (each ~0.25)
    let dominant = avg[0b0000] + avg[0b0011] + avg[0b1100] + avg[0b1111];
    println!(
        "avg P(|0000⟩) = {:.4}, avg P(|0011⟩) = {:.4}, avg P(|1100⟩) = {:.4}, avg P(|1111⟩) = {:.4}",
        avg[0b0000], avg[0b0011], avg[0b1100], avg[0b1111]
    );
    println!("Sum of dominant states: {dominant:.4}  (noiseless=1.0)");

    assert!(
        dominant > 0.70,
        "Dominant states should sum to >0.70 at p=0.05; got {dominant:.4}"
    );

    println!("\nOK — depolarizing noise degrades Bell fidelity as expected.");
    Ok(())
}
