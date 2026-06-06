//! Example: 3D quantum state visualizations for a 2-qubit Bell state.
//!
//! Demonstrates that all five visualization renderers produce valid
//! Plotly JSON for the Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2.

use quantrs2_core::state_visualization_3d::{bloch, density_bars, husimi, qsphere, wigner};
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;

fn main() {
    // Build Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2
    let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
    let bell: Array1<Complex64> = Array1::from(vec![
        Complex64::new(inv_sqrt2, 0.0), // |00⟩
        Complex64::new(0.0, 0.0),       // |01⟩
        Complex64::new(0.0, 0.0),       // |10⟩
        Complex64::new(inv_sqrt2, 0.0), // |11⟩
    ]);
    let n_qubits = 2usize;

    // Bloch sphere array
    let bloch_json =
        bloch::bloch_array_plotly_json(&bell, n_qubits).expect("Bloch array JSON failed");
    let _: serde_json::Value = serde_json::from_str(&bloch_json).expect("Bloch JSON not valid");
    println!("  [OK] Bloch array — {} bytes", bloch_json.len());

    // Q-sphere
    let qsphere_json = qsphere::qsphere_plotly_json(&bell, n_qubits).expect("Q-sphere JSON failed");
    let _: serde_json::Value =
        serde_json::from_str(&qsphere_json).expect("Q-sphere JSON not valid");
    println!("  [OK] Q-sphere — {} bytes", qsphere_json.len());

    // Wigner (n=2)
    let wigner_json = wigner::wigner_plotly_json(&bell, n_qubits).expect("Wigner JSON failed");
    let _: serde_json::Value = serde_json::from_str(&wigner_json).expect("Wigner JSON not valid");
    println!("  [OK] Wigner n=2 — {} bytes", wigner_json.len());

    // Husimi
    let husimi_json = husimi::husimi_plotly_json(&bell, n_qubits).expect("Husimi JSON failed");
    let _: serde_json::Value = serde_json::from_str(&husimi_json).expect("Husimi JSON not valid");
    println!("  [OK] Husimi — {} bytes", husimi_json.len());

    // Density matrix bars
    let bars_json = density_bars::density_matrix_bars_plotly_json(&bell, n_qubits)
        .expect("Density bars JSON failed");
    let _: serde_json::Value =
        serde_json::from_str(&bars_json).expect("Density bars JSON not valid");
    println!("  [OK] Density bars — {} bytes", bars_json.len());

    println!("State visualization smoke test passed");
}
