use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use quantrs2_sim::holographic_quantum_error_correction::*;

fn main() {
    // Create a simple configuration
    let config = HolographicQECConfig {
        boundary_qubits: 2,
        bulk_qubits: 3,
        ads_radius: 1.0,
        central_charge: 12.0,
        error_correction_code: HolographicCodeType::AdSRindler,
        ..Default::default()
    };

    let simulator = HolographicQECSimulator::new(config);

    // Test each encoding method
    let boundary_dim = 1 << 2; // 4
    let bulk_dim = 1 << 3; // 8

    println!("Testing holographic encoding matrix creation...");
    println!(
        "Boundary dimension: {}, Bulk dimension: {}",
        boundary_dim, bulk_dim
    );

    match simulator.create_holographic_encoding_matrix(boundary_dim, bulk_dim) {
        Ok(matrix) => {
            println!(
                "Matrix created successfully with dimensions: {:?}",
                matrix.dim()
            );

            // Check if matrix is all zeros
            let mut zero_count = 0;
            let mut non_zero_count = 0;
            let mut max_magnitude = 0.0;

            for element in matrix.iter() {
                let magnitude = element.norm();
                if magnitude < 1e-10 {
                    zero_count += 1;
                } else {
                    non_zero_count += 1;
                    if magnitude > max_magnitude {
                        max_magnitude = magnitude;
                    }
                }
            }

            println!("Matrix statistics:");
            println!("  Zero elements: {}", zero_count);
            println!("  Non-zero elements: {}", non_zero_count);
            println!("  Max magnitude: {}", max_magnitude);
            println!("  Total elements: {}", matrix.len());

            // Print first few elements
            println!("\nFirst few matrix elements:");
            for i in 0..std::cmp::min(8, matrix.dim().0) {
                for j in 0..std::cmp::min(4, matrix.dim().1) {
                    print!("{:.6} ", matrix[[i, j]].norm());
                }
                println!();
            }

            if zero_count == matrix.len() {
                println!("\n❌ PROBLEM: Matrix is all zeros!");
            } else {
                println!("\n✅ Matrix contains non-zero elements");
            }
        }
        Err(e) => {
            println!("❌ Error creating matrix: {:?}", e);
        }
    }

    // Test individual encoding methods
    println!("\n--- Testing individual encoding methods ---");

    let mut test_matrix = Array2::zeros((bulk_dim, boundary_dim));

    // Test AdS-Rindler encoding
    println!("\nTesting AdS-Rindler encoding...");
    match simulator.create_ads_rindler_encoding(&mut test_matrix) {
        Ok(_) => {
            let norm: f64 = test_matrix.iter().map(|x| x.norm_sqr()).sum();
            println!("AdS-Rindler encoding matrix norm: {}", norm.sqrt());
            if norm < 1e-10 {
                println!("❌ AdS-Rindler matrix is effectively zero");
            } else {
                println!("✅ AdS-Rindler matrix has non-zero elements");
            }
        }
        Err(e) => {
            println!("❌ Error in AdS-Rindler encoding: {:?}", e);
        }
    }

    // Test individual factor calculations
    println!("\n--- Testing factor calculations ---");
    let rindler_factor = simulator.calculate_rindler_factor(1, 1);
    let entanglement_factor = simulator.calculate_entanglement_factor(1, 1);

    println!("Rindler factor (1,1): {}", rindler_factor);
    println!("Entanglement factor (1,1): {}", entanglement_factor);

    if rindler_factor.is_nan() || rindler_factor.is_infinite() {
        println!("❌ Rindler factor is NaN or infinite");
    }
    if entanglement_factor.is_nan() || entanglement_factor.is_infinite() {
        println!("❌ Entanglement factor is NaN or infinite");
    }

    // Test multiple indices
    println!("\nTesting multiple indices:");
    for i in 0..bulk_dim {
        for j in 0..boundary_dim {
            let rf = simulator.calculate_rindler_factor(i, j);
            let ef = simulator.calculate_entanglement_factor(i, j);
            println!(
                "  ({}, {}): Rindler={:.6}, Entanglement={:.6}",
                i, j, rf, ef
            );
        }
    }
}
