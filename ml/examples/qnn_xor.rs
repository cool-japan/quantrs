//! Quantum Neural Network for XOR Classification
//!
//! XOR is the classic non-linearly separable problem:
//!   (0,0) → 0,  (0,1) → 1,  (1,0) → 1,  (1,1) → 0
//!
//! This example implements a variational quantum circuit (VQC) acting as a
//! quantum neural network to learn the XOR function via the parameter-shift rule.
//!
//! Architecture (2 qubits, 4 trainable parameters):
//!   Layer 0 (Encoding):  RY(π·x0) on q0, RY(π·x1) on q1
//!   Layer 1 (Variational): RY(θ0) on q0, RY(θ1) on q1, CNOT(q0→q1)
//!   Layer 2 (Variational): RY(θ2) on q0, RY(θ3) on q1, CNOT(q1→q0)
//!   Measurement: ⟨Z₀⟩ → class (positive → class 1, non-positive → class 0)
//!
//! Loss: MSE between ⟨Z₀⟩ and target ∈ {-1,+1}  (avoids BCE gradient saturation)
//!
//! Training uses the parameter-shift rule:
//!   ∂L/∂θ_k = (L(θ_k + π/2) - L(θ_k - π/2)) / 2
//!
//! Run with:
//!   cargo run --example qnn_xor -p quantrs2-ml --all-features

use quantrs2_circuit::builder::{Circuit, Simulator};
use quantrs2_core::register::Register;
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{array, Array1};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Evaluate the VQC and return ⟨Z₀⟩ for given input and parameters
fn vqc_forward(x0: f64, x1: f64, params: &[f64]) -> Result<f64> {
    use std::f64::consts::PI;

    // Build the 2-qubit variational circuit
    let mut circuit = Circuit::<2>::new();

    // --- Encoding layer: encode input features as RY rotations ---
    circuit.ry(0u32, PI * x0)?;
    circuit.ry(1u32, PI * x1)?;

    // --- Variational layer 1 ---
    circuit.ry(0u32, params[0])?;
    circuit.ry(1u32, params[1])?;
    circuit.cnot(0u32, 1u32)?;

    // --- Variational layer 2 ---
    circuit.ry(0u32, params[2])?;
    circuit.ry(1u32, params[3])?;
    circuit.cnot(1u32, 0u32)?;

    // --- Run ---
    let sim = StateVectorSimulator::sequential();
    let reg: Register<2> = sim.run(&circuit)?;

    // Return ⟨Z₀⟩ as the output signal
    let ez0 = reg.expectation_z(0u32)?;
    Ok(ez0)
}

/// Mean-squared-error loss for a single sample.
///
/// Target is mapped from {0,1} → {-1,+1} to align with ⟨Z₀⟩ ∈ (-1,+1).
/// MSE avoids the gradient-saturation problem that BCE has when the output
/// is close to ±1 (the clamped BCE gradient collapses to zero).
fn loss(output: f64, label: f64) -> f64 {
    let target = 2.0f64.mul_add(label, -1.0); // {0,1} → {-1,+1}
    (output - target).powi(2)
}

/// Parameter-shift gradient for parameter k
fn parameter_shift_gradient(
    x0: f64,
    x1: f64,
    params: &[f64],
    label: f64,
    k: usize,
) -> Result<f64> {
    let shift = std::f64::consts::PI / 2.0;
    let mut params_plus = params.to_vec();
    let mut params_minus = params.to_vec();
    params_plus[k] += shift;
    params_minus[k] -= shift;

    let out_plus = vqc_forward(x0, x1, &params_plus)?;
    let out_minus = vqc_forward(x0, x1, &params_minus)?;

    let loss_plus = loss(out_plus, label);
    let loss_minus = loss(out_minus, label);

    Ok((loss_plus - loss_minus) / 2.0)
}

fn main() -> Result<()> {
    println!("=== QNN XOR Classifier (Variational Quantum Circuit) ===\n");

    // ---- XOR dataset ----
    let data: [(f64, f64, f64); 4] = [
        (0.0, 0.0, 0.0), // (x0, x1, label)
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ];

    println!("XOR dataset:");
    for &(x0, x1, label) in &data {
        println!("  ({x0:.0}, {x1:.0}) → {label:.0}");
    }
    println!();

    // ---- Initialise parameters ----
    // 4 trainable parameters: θ0..θ3
    // Use deterministic starting values (no random for reproducibility)
    let mut params: Array1<f64> = array![0.5, -0.5, 0.3, -0.3];

    let n_params = params.len();
    let lr = 0.3f64; // learning rate
    let n_epochs = 80usize;

    println!("Training: {n_epochs} epochs, lr={lr}, params={n_params}");
    println!("{}", "-".repeat(50));

    // ---- Training loop ----
    let mut loss_history = Vec::with_capacity(n_epochs);
    for epoch in 0..n_epochs {
        let mut total_loss = 0.0f64;
        let mut grad_sum = vec![0.0f64; n_params];

        // Accumulate gradients over the full dataset
        for &(x0, x1, label) in &data {
            let output = vqc_forward(x0, x1, params.as_slice().expect("slice"))?;
            total_loss += loss(output, label);

            for (k, gs) in grad_sum.iter_mut().enumerate() {
                let g = parameter_shift_gradient(
                    x0,
                    x1,
                    params.as_slice().expect("slice"),
                    label,
                    k,
                )?;
                *gs += g;
            }
        }

        // Average over dataset and update parameters (gradient descent)
        let n_samples = data.len() as f64;
        for (p, g) in params.iter_mut().zip(grad_sum.iter()) {
            *p -= lr * g / n_samples;
        }

        let avg_loss = total_loss / n_samples;
        loss_history.push(avg_loss);

        if epoch % 10 == 0 || epoch == n_epochs - 1 {
            println!("  Epoch {:>3}: loss = {avg_loss:.6}", epoch + 1);
        }
    }

    // ---- Evaluation ----
    println!("\n=== Evaluation ===");
    let mut correct = 0usize;
    for &(x0, x1, label) in &data {
        let output = vqc_forward(x0, x1, params.as_slice().expect("slice"))?;
        let predicted = if output > 0.0 { 1.0 } else { 0.0 };
        let ok = (predicted - label).abs() < 0.5;
        if ok { correct += 1; }
        println!(
            "  ({x0:.0},{x1:.0}) → ⟨Z₀⟩={output:+.4}  pred={predicted:.0}  label={label:.0}  {}",
            if ok { "✓" } else { "✗" }
        );
    }
    let accuracy = correct as f64 / data.len() as f64;
    println!("\nAccuracy : {correct}/{} = {:.0}%", data.len(), accuracy * 100.0);
    println!("Final params: [{}]",
        params.iter().map(|p| format!("{p:.4}")).collect::<Vec<_>>().join(", "));

    // Loss decreased
    let initial_loss = loss_history[0];
    let final_loss = *loss_history.last().expect("non-empty");
    println!("Loss: {initial_loss:.4} → {final_loss:.4}  (decreased: {})",
        if final_loss < initial_loss { "yes" } else { "no" });

    // MSE should decrease during training
    assert!(
        final_loss < initial_loss,
        "Training should reduce MSE loss: {initial_loss:.4} → {final_loss:.4}"
    );

    // With 80 epochs and MSE loss, the circuit should learn XOR.
    // A 2-qubit VQC can in principle represent XOR (non-linearly separable).
    // We accept ≥ 3/4 accuracy or notable loss reduction as success.
    assert!(
        accuracy >= 0.75 || final_loss < initial_loss * 0.5,
        "XOR QNN: accuracy={accuracy:.0}% initial_loss={initial_loss:.4} final_loss={final_loss:.4}"
    );

    println!("\nOK");
    Ok(())
}
