//! Benchmarking for holographic quantum error correction.
//!
//! This module contains benchmark functions for evaluating the performance
//! of the holographic quantum error correction framework.

use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

use crate::error::Result;

use super::config::HolographicQECConfig;
use super::simulator::HolographicQECSimulator;

/// Benchmark holographic quantum error correction
pub fn benchmark_holographic_qec(
    config: HolographicQECConfig,
    num_trials: usize,
    error_rates: &[f64],
) -> Result<HolographicQECBenchmarkResults> {
    let mut results = HolographicQECBenchmarkResults::default();
    let start_time = std::time::Instant::now();

    for &error_rate in error_rates {
        let mut trial_results = Vec::new();

        for _trial in 0..num_trials {
            let mut simulator = HolographicQECSimulator::new(config.clone());
            simulator.initialize()?;

            // Introduce random errors
            let num_errors = ((config.boundary_qubits as f64) * error_rate) as usize;
            let mut rng = thread_rng();
            let error_locations: Vec<usize> = (0..num_errors)
                .map(|_| rng.gen_range(0..config.boundary_qubits))
                .collect();

            // Perform error correction
            let correction_result = simulator.perform_error_correction(&error_locations)?;
            trial_results.push(correction_result);
        }

        // Calculate statistics for this error rate
        let success_rate = trial_results
            .iter()
            .map(|r| if r.correction_successful { 1.0 } else { 0.0 })
            .sum::<f64>()
            / num_trials as f64;

        let average_correction_time = trial_results
            .iter()
            .map(|r| r.correction_time.as_secs_f64())
            .sum::<f64>()
            / num_trials as f64;

        let average_entanglement_entropy = trial_results
            .iter()
            .map(|r| r.entanglement_entropy)
            .sum::<f64>()
            / num_trials as f64;

        results.error_rates.push(error_rate);
        results.success_rates.push(success_rate);
        results
            .average_correction_times
            .push(average_correction_time);
        results
            .average_entanglement_entropies
            .push(average_entanglement_entropy);
    }

    results.total_benchmark_time = start_time.elapsed();
    Ok(results)
}

/// Holographic QEC benchmark results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HolographicQECBenchmarkResults {
    /// Error rates tested
    pub error_rates: Vec<f64>,
    /// Success rates for each error rate
    pub success_rates: Vec<f64>,
    /// Average correction times
    pub average_correction_times: Vec<f64>,
    /// Average entanglement entropies
    pub average_entanglement_entropies: Vec<f64>,
    /// Total benchmark time
    pub total_benchmark_time: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holographic_quantum_error_correction::{
        BulkReconstructionMethod, HolographicCodeType,
    };
    use scirs2_core::ndarray::Array2;
    use scirs2_core::Complex64;

    #[test]
    #[ignore]
    fn test_holographic_qec_initialization() {
        let config = HolographicQECConfig::default();
        let mut simulator = HolographicQECSimulator::new(config);

        assert!(simulator.initialize().is_ok());
        assert!(simulator.boundary_state.is_some());
        assert!(simulator.bulk_state.is_some());
    }

    #[test]
    #[ignore]
    fn test_holographic_encoding_matrix() {
        let config = HolographicQECConfig {
            boundary_qubits: 3,
            bulk_qubits: 6,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let boundary_dim = 1 << 3;
        let bulk_dim = 1 << 6;
        let encoding_matrix = simulator.create_holographic_encoding_matrix(boundary_dim, bulk_dim);

        assert!(encoding_matrix.is_ok());
        let matrix = encoding_matrix.expect("encoding matrix creation should succeed");
        assert_eq!(matrix.dim(), (bulk_dim, boundary_dim));
    }

    #[test]
    #[ignore]
    fn test_ads_rindler_encoding() {
        let config = HolographicQECConfig {
            error_correction_code: HolographicCodeType::AdSRindler,
            boundary_qubits: 2,
            bulk_qubits: 4,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let mut encoding_matrix = Array2::zeros((16, 4));
        assert!(simulator
            .create_ads_rindler_encoding(&mut encoding_matrix)
            .is_ok());

        // Check that matrix is not all zeros
        let matrix_norm: f64 = encoding_matrix.iter().map(|x| x.norm_sqr()).sum();
        assert!(matrix_norm > 0.0);
    }

    #[test]
    #[ignore]
    fn test_holographic_stabilizer_encoding() {
        let config = HolographicQECConfig {
            error_correction_code: HolographicCodeType::HolographicStabilizer,
            boundary_qubits: 2,
            bulk_qubits: 4,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let mut encoding_matrix = Array2::zeros((16, 4));
        assert!(simulator
            .create_holographic_stabilizer_encoding(&mut encoding_matrix)
            .is_ok());

        // Check that matrix is not all zeros
        let matrix_norm: f64 = encoding_matrix.iter().map(|x| x.norm_sqr()).sum();
        assert!(matrix_norm > 0.0);
    }

    #[test]
    #[ignore]
    fn test_bulk_geometry_encoding() {
        let config = HolographicQECConfig {
            error_correction_code: HolographicCodeType::BulkGeometry,
            boundary_qubits: 2,
            bulk_qubits: 4,
            ads_radius: 1.0,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let mut encoding_matrix = Array2::zeros((16, 4));
        assert!(simulator
            .create_bulk_geometry_encoding(&mut encoding_matrix)
            .is_ok());

        // Check that matrix is not all zeros
        let matrix_norm: f64 = encoding_matrix.iter().map(|x| x.norm_sqr()).sum();
        assert!(matrix_norm > 0.0);
    }

    #[test]
    #[ignore]
    fn test_tensor_network_encoding() {
        let config = HolographicQECConfig {
            error_correction_code: HolographicCodeType::TensorNetwork,
            boundary_qubits: 2,
            bulk_qubits: 4,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let mut encoding_matrix = Array2::zeros((16, 4));
        assert!(simulator
            .create_tensor_network_encoding(&mut encoding_matrix)
            .is_ok());

        // Check that matrix is not all zeros
        let matrix_norm: f64 = encoding_matrix.iter().map(|x| x.norm_sqr()).sum();
        assert!(matrix_norm > 0.0);
    }

    #[test]
    #[ignore]
    fn test_holographic_surface_encoding() {
        let config = HolographicQECConfig {
            error_correction_code: HolographicCodeType::HolographicSurface,
            boundary_qubits: 2,
            bulk_qubits: 4,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let mut encoding_matrix = Array2::zeros((16, 4));
        assert!(simulator
            .create_holographic_surface_encoding(&mut encoding_matrix)
            .is_ok());

        // Check that matrix is not all zeros
        let matrix_norm: f64 = encoding_matrix.iter().map(|x| x.norm_sqr()).sum();
        assert!(matrix_norm > 0.0);
    }

    #[test]
    #[ignore]
    fn test_perfect_tensor_encoding() {
        let config = HolographicQECConfig {
            error_correction_code: HolographicCodeType::PerfectTensor,
            boundary_qubits: 2,
            bulk_qubits: 4,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let mut encoding_matrix = Array2::zeros((16, 4));
        assert!(simulator
            .create_perfect_tensor_encoding(&mut encoding_matrix)
            .is_ok());

        // Check that matrix is not all zeros
        let matrix_norm: f64 = encoding_matrix.iter().map(|x| x.norm_sqr()).sum();
        assert!(matrix_norm > 0.0);
    }

    #[test]
    #[ignore]
    fn test_entanglement_entropy_encoding() {
        let config = HolographicQECConfig {
            error_correction_code: HolographicCodeType::EntanglementEntropy,
            boundary_qubits: 2,
            bulk_qubits: 4,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let mut encoding_matrix = Array2::zeros((16, 4));
        assert!(simulator
            .create_entanglement_entropy_encoding(&mut encoding_matrix)
            .is_ok());

        // Check that matrix is not all zeros
        let matrix_norm: f64 = encoding_matrix.iter().map(|x| x.norm_sqr()).sum();
        assert!(matrix_norm > 0.0);
    }

    #[test]
    #[ignore]
    fn test_ads_cft_encoding() {
        let config = HolographicQECConfig {
            error_correction_code: HolographicCodeType::AdSCFTCode,
            boundary_qubits: 2,
            bulk_qubits: 4,
            ads_radius: 1.0,
            central_charge: 12.0,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let mut encoding_matrix = Array2::zeros((16, 4));
        assert!(simulator
            .create_ads_cft_encoding(&mut encoding_matrix)
            .is_ok());

        // Check that matrix is not all zeros
        let matrix_norm: f64 = encoding_matrix.iter().map(|x| x.norm_sqr()).sum();
        assert!(matrix_norm > 0.0);
    }

    #[test]
    #[ignore]
    fn test_rindler_factor_calculation() {
        let config = HolographicQECConfig {
            ads_radius: 1.0,
            boundary_qubits: 2,
            bulk_qubits: 4,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let factor = simulator.calculate_rindler_factor(5, 2);
        assert!(factor.is_finite());
        assert!(factor >= 0.0);
    }

    #[test]
    #[ignore]
    fn test_entanglement_factor_calculation() {
        let config = HolographicQECConfig::default();
        let simulator = HolographicQECSimulator::new(config);

        let factor = simulator.calculate_entanglement_factor(5, 2);
        assert!(factor.is_finite());
        assert!(factor >= 0.0);
    }

    #[test]
    #[ignore]
    fn test_mutual_information_calculation() {
        let config = HolographicQECConfig::default();
        let simulator = HolographicQECSimulator::new(config);

        let mi = simulator.calculate_mutual_information(5, 2);
        assert!(mi.is_finite());
    }

    #[test]
    #[ignore]
    fn test_rt_surface_area_calculation() {
        let config = HolographicQECConfig {
            ads_radius: 1.0,
            central_charge: 12.0,
            boundary_qubits: 3,
            bulk_qubits: 6,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let area = simulator.calculate_rt_surface_area(10, 3);
        assert!(area.is_finite());
        assert!(area >= 0.0);
    }

    #[test]
    #[ignore]
    fn test_geodesic_length_calculation() {
        let config = HolographicQECConfig {
            ads_radius: 1.0,
            boundary_qubits: 3,
            bulk_qubits: 6,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let length = simulator.calculate_geodesic_length(10, 3);
        assert!(length.is_finite());
        assert!(length >= 0.0);
    }

    #[test]
    #[ignore]
    fn test_stabilizer_generators_setup() {
        let config = HolographicQECConfig {
            boundary_qubits: 3,
            ..Default::default()
        };
        let mut simulator = HolographicQECSimulator::new(config.clone());

        assert!(simulator.setup_stabilizer_generators().is_ok());
        assert_eq!(
            simulator.stabilizer_generators.len(),
            config.boundary_qubits
        );
    }

    #[test]
    #[ignore]
    fn test_error_correction_operators_initialization() {
        let config = HolographicQECConfig {
            boundary_qubits: 2,
            ..Default::default()
        };
        let mut simulator = HolographicQECSimulator::new(config);

        assert!(simulator.initialize_error_correction_operators().is_ok());
        assert!(simulator.error_correction_operators.contains_key("PauliX"));
        assert!(simulator.error_correction_operators.contains_key("PauliZ"));
        assert!(simulator
            .error_correction_operators
            .contains_key("Holographic"));
    }

    #[test]
    #[ignore]
    fn test_syndrome_measurement() {
        let config = HolographicQECConfig {
            boundary_qubits: 2,
            ..Default::default()
        };
        let mut simulator = HolographicQECSimulator::new(config);

        assert!(simulator.initialize().is_ok());

        let syndromes = simulator.measure_syndromes();
        assert!(syndromes.is_ok());

        let syndrome_values = syndromes.expect("syndrome measurement should succeed");
        assert_eq!(syndrome_values.len(), simulator.config.boundary_qubits);

        for syndrome in syndrome_values {
            assert!(syndrome.is_finite());
        }
    }

    #[test]
    #[ignore]
    fn test_error_correction_performance() {
        let config = HolographicQECConfig {
            boundary_qubits: 3,
            bulk_qubits: 6,
            error_threshold: 0.1,
            ..Default::default()
        };
        let mut simulator = HolographicQECSimulator::new(config);

        assert!(simulator.initialize().is_ok());

        // Introduce single error
        let error_locations = vec![0];
        let result = simulator.perform_error_correction(&error_locations);

        assert!(result.is_ok());
        let correction_result = result.expect("error correction should succeed");
        assert!(!correction_result.syndromes.is_empty());
        assert!(correction_result.correction_time.as_nanos() > 0);
    }

    #[test]
    #[ignore]
    fn test_bulk_reconstruction() {
        let config = HolographicQECConfig {
            boundary_qubits: 2,
            bulk_qubits: 4,
            reconstruction_method: BulkReconstructionMethod::HKLL,
            ..Default::default()
        };
        let mut simulator = HolographicQECSimulator::new(config);

        assert!(simulator.initialize().is_ok());

        // Create boundary data
        let boundary_data = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.5, 0.5),
            Complex64::new(0.0, 0.0),
        ];

        let result = simulator.perform_bulk_reconstruction(&boundary_data);
        assert!(result.is_ok());

        let reconstruction_result = result.expect("bulk reconstruction should succeed");
        assert_eq!(reconstruction_result.reconstructed_bulk.len(), 1 << 4);
        assert!(reconstruction_result.reconstruction_fidelity >= 0.0);
        assert!(reconstruction_result.reconstruction_fidelity <= 1.0);
    }

    #[test]
    #[ignore]
    fn test_hkll_reconstruction() {
        let config = HolographicQECConfig {
            boundary_qubits: 2,
            bulk_qubits: 4,
            reconstruction_method: BulkReconstructionMethod::HKLL,
            ads_radius: 1.0,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let syndromes = vec![0.1, -0.05];
        let errors = simulator.decode_hkll_errors(&syndromes);

        assert!(errors.is_ok());
        let error_locations = errors.expect("HKLL error decoding should succeed");
        assert!(!error_locations.is_empty());
    }

    #[test]
    #[ignore]
    fn test_entanglement_wedge_reconstruction() {
        let config = HolographicQECConfig {
            boundary_qubits: 2,
            bulk_qubits: 4,
            reconstruction_method: BulkReconstructionMethod::EntanglementWedge,
            ..Default::default()
        };
        let simulator = HolographicQECSimulator::new(config);

        let syndromes = vec![0.1, -0.05];
        let errors = simulator.decode_entanglement_wedge_errors(&syndromes);

        assert!(errors.is_ok());
        let error_locations = errors.expect("entanglement wedge error decoding should succeed");
        assert!(!error_locations.is_empty());
    }

    #[test]
    #[ignore]
    fn test_holographic_qec_utils() {
        use super::super::utils::HolographicQECUtils;

        let threshold = HolographicQECUtils::calculate_error_threshold(1.0, 12.0, 8);
        assert!(threshold > 0.0);
        assert!(threshold < 1.0);

        let bulk_qubits = HolographicQECUtils::estimate_bulk_qubits(8, 2.0);
        assert_eq!(bulk_qubits, 16);

        let ads_radius = HolographicQECUtils::calculate_optimal_ads_radius(8, 0.01, 12.0);
        assert!(ads_radius > 0.0);

        let config = HolographicQECConfig::default();
        assert!(HolographicQECUtils::verify_code_parameters(&config).is_ok());
    }

    #[test]
    #[ignore]
    fn test_holographic_qec_benchmark() {
        let config = HolographicQECConfig {
            boundary_qubits: 2,
            bulk_qubits: 4,
            error_threshold: 0.1,
            ..Default::default()
        };

        let error_rates = vec![0.01, 0.05];
        let num_trials = 2;

        let benchmark_result = benchmark_holographic_qec(config, num_trials, &error_rates);
        assert!(benchmark_result.is_ok());

        let results = benchmark_result.expect("holographic QEC benchmark should succeed");
        assert_eq!(results.error_rates.len(), 2);
        assert_eq!(results.success_rates.len(), 2);
        assert!(results.total_benchmark_time.as_nanos() > 0);
    }

    #[test]
    #[ignore]
    fn test_all_holographic_code_types() {
        let code_types = vec![
            HolographicCodeType::AdSRindler,
            HolographicCodeType::HolographicStabilizer,
            HolographicCodeType::BulkGeometry,
            HolographicCodeType::TensorNetwork,
            HolographicCodeType::HolographicSurface,
            HolographicCodeType::PerfectTensor,
            HolographicCodeType::EntanglementEntropy,
            HolographicCodeType::AdSCFTCode,
        ];

        for code_type in code_types {
            let config = HolographicQECConfig {
                error_correction_code: code_type,
                boundary_qubits: 2,
                bulk_qubits: 4,
                ..Default::default()
            };

            let mut simulator = HolographicQECSimulator::new(config);
            assert!(simulator.initialize().is_ok());

            // Test encoding matrix creation
            let encoding_result = simulator.create_holographic_encoding_matrix(16, 4);
            assert!(encoding_result.is_ok());

            let encoding_matrix = encoding_result
                .expect("encoding matrix creation should succeed for all code types");
            let matrix_norm: f64 = encoding_matrix.iter().map(|x| x.norm_sqr()).sum();
            assert!(matrix_norm > 0.0);
        }
    }

    #[test]
    #[ignore]
    fn test_all_bulk_reconstruction_methods() {
        let reconstruction_methods = vec![
            BulkReconstructionMethod::HKLL,
            BulkReconstructionMethod::EntanglementWedge,
            BulkReconstructionMethod::QECReconstruction,
            BulkReconstructionMethod::TensorNetwork,
            BulkReconstructionMethod::HolographicTensorNetwork,
            BulkReconstructionMethod::BulkBoundaryDictionary,
            BulkReconstructionMethod::MinimalSurface,
        ];

        for method in reconstruction_methods {
            let config = HolographicQECConfig {
                reconstruction_method: method,
                boundary_qubits: 2,
                bulk_qubits: 4,
                error_threshold: 0.1,
                ..Default::default()
            };

            let simulator = HolographicQECSimulator::new(config);

            // Test error decoding
            let syndromes = vec![0.15, -0.12];
            let errors = simulator.decode_holographic_errors(&syndromes);
            assert!(errors.is_ok());

            let error_locations =
                errors.expect("holographic error decoding should succeed for all methods");
            assert!(!error_locations.is_empty());
        }
    }

    #[test]
    #[ignore]
    fn test_holographic_qec_statistics() {
        let config = HolographicQECConfig {
            boundary_qubits: 2,
            bulk_qubits: 4,
            ..Default::default()
        };
        let mut simulator = HolographicQECSimulator::new(config);

        assert!(simulator.initialize().is_ok());

        // Perform several error corrections
        for i in 0..3 {
            let error_locations = vec![i % 2];
            let _ = simulator.perform_error_correction(&error_locations);
        }

        let stats = simulator.get_stats();
        assert_eq!(stats.total_corrections, 3);
        assert!(stats.correction_time.as_nanos() > 0);
    }

    #[test]
    #[ignore]
    fn debug_holographic_encoding_matrix() {
        use std::f64::consts::PI;

        // Create a simple configuration for debugging
        let config = HolographicQECConfig {
            boundary_qubits: 2,
            bulk_qubits: 3,
            ads_radius: 1.0,
            central_charge: 12.0,
            error_correction_code: HolographicCodeType::AdSRindler,
            ..Default::default()
        };

        let simulator = HolographicQECSimulator::new(config);

        let boundary_dim = 1 << 2; // 4
        let bulk_dim = 1 << 3; // 8

        println!("Testing holographic encoding matrix creation...");
        println!("Boundary dimension: {boundary_dim}, Bulk dimension: {bulk_dim}");

        // Test matrix creation
        let matrix_result = simulator.create_holographic_encoding_matrix(boundary_dim, bulk_dim);
        assert!(matrix_result.is_ok());

        let matrix = matrix_result.expect("encoding matrix creation should succeed in debug test");
        println!(
            "Matrix created successfully with dimensions: {:?}",
            matrix.dim()
        );

        // Analyze matrix content
        let mut zero_count = 0;
        let mut non_zero_count = 0;
        let mut max_magnitude = 0.0;

        for element in &matrix {
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
        println!("  Zero elements: {zero_count}");
        println!("  Non-zero elements: {non_zero_count}");
        println!("  Max magnitude: {max_magnitude}");
        println!("  Total elements: {}", matrix.len());

        // Print sample elements
        println!("\nSample matrix elements:");
        for i in 0..std::cmp::min(4, matrix.dim().0) {
            for j in 0..std::cmp::min(4, matrix.dim().1) {
                print!("{:.6} ", matrix[[i, j]].norm());
            }
            println!();
        }

        // Test individual factor calculations
        println!("\n--- Testing factor calculations ---");
        let rindler_factor = simulator.calculate_rindler_factor(1, 1);
        let entanglement_factor = simulator.calculate_entanglement_factor(1, 1);

        println!("Rindler factor (1,1): {rindler_factor}");
        println!("Entanglement factor (1,1): {entanglement_factor}");

        // Check for problematic values
        assert!(!rindler_factor.is_nan(), "Rindler factor should not be NaN");
        assert!(
            !rindler_factor.is_infinite(),
            "Rindler factor should not be infinite"
        );
        assert!(
            !entanglement_factor.is_nan(),
            "Entanglement factor should not be NaN"
        );
        assert!(
            !entanglement_factor.is_infinite(),
            "Entanglement factor should not be infinite"
        );

        // Test AdS-Rindler encoding specifically
        println!("\n--- Testing AdS-Rindler encoding directly ---");
        let mut test_matrix = Array2::zeros((bulk_dim, boundary_dim));
        let ads_result = simulator.create_ads_rindler_encoding(&mut test_matrix);
        assert!(ads_result.is_ok());

        let ads_norm: f64 = test_matrix.iter().map(|x| x.norm_sqr()).sum();
        println!("AdS-Rindler encoding matrix norm: {}", ads_norm.sqrt());

        if ads_norm < 1e-10 {
            println!("WARNING: AdS-Rindler matrix is effectively zero!");
            // Let's investigate why
            println!("Investigating zero matrix cause...");

            for i in 0..bulk_dim {
                for j in 0..boundary_dim {
                    let rf = simulator.calculate_rindler_factor(i, j);
                    let ef = simulator.calculate_entanglement_factor(i, j);
                    let product = rf * ef;
                    if i < 2 && j < 2 {
                        println!(
                            "  ({i}, {j}): Rindler={rf:.6}, Entanglement={ef:.6}, Product={product:.6}"
                        );
                    }
                }
            }
        } else {
            println!("AdS-Rindler matrix has non-zero elements");
        }

        // Investigate the boundary position issue further
        println!("\n--- Analyzing boundary position cos values ---");
        for j in 0..boundary_dim {
            let boundary_position = (j as f64) / (1 << simulator.config.boundary_qubits) as f64;
            let cos_value = (2.0 * PI * boundary_position).cos();
            println!(
                "  boundary_index {j}: position={boundary_position:.3}, cos(2π*pos)={cos_value:.6}"
            );
        }

        println!("\n--- Analyzing bulk position cosh values ---");
        for i in 0..bulk_dim {
            let bulk_position = (i as f64) / (1 << simulator.config.bulk_qubits) as f64;
            let cosh_value = (simulator.config.ads_radius * bulk_position).cosh();
            println!(
                "  bulk_index {i}: position={bulk_position:.3}, cosh(ads_radius*pos)={cosh_value:.6}"
            );
        }

        // The matrix should not be all zeros
        assert!(
            non_zero_count > 0,
            "Holographic encoding matrix should not be all zeros"
        );
        assert!(
            max_magnitude > 1e-10,
            "Matrix should have meaningful magnitudes"
        );
    }
}
