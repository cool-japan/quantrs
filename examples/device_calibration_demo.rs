//! Demonstration of device-specific gate calibration data structures
//!
//! This example shows how to use calibration data for quantum devices,
//! including creating calibrations, building noise models, and optimizing circuits.

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_device::prelude::*;
use quantrs2_device::calibration::*;
use quantrs2_device::noise_model::*;
use quantrs2_device::optimization::*;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QuantRS2 Device Calibration Demo ===\n");

    // Demo 1: Create and save calibration data
    demo_calibration_creation()?;

    // Demo 2: Build noise model from calibration
    demo_noise_model()?;

    // Demo 3: Optimize circuit using calibration
    demo_circuit_optimization()?;

    // Demo 4: Estimate circuit fidelity
    demo_fidelity_estimation()?;

    // Demo 5: Manage multiple device calibrations
    demo_calibration_management()?;

    Ok(())
}

/// Demo 1: Create and save calibration data
fn demo_calibration_creation() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Creating Device Calibration");
    println!("==============================");

    // Create calibration for a 5-qubit device
    let mut builder = CalibrationBuilder::new("demo_device_v1".to_string())
        .valid_duration(Duration::from_secs(12 * 3600)); // 12 hours

    // Add qubit calibrations with realistic values
    for i in 0..5 {
        let qubit_id = QubitId(i as u16);

        // Vary parameters slightly for each qubit
        let t1 = 50_000.0 + (i as f64 * 5_000.0); // 50-70 μs
        let t2 = 40_000.0 + (i as f64 * 3_000.0); // 40-52 μs
        let frequency = 5e9 + (i as f64 * 10e6);   // 5.00-5.04 GHz

        builder = builder.add_qubit_calibration(QubitCalibration {
            qubit_id,
            frequency,
            anharmonicity: -330e6 + (i as f64 * 5e6), // -330 to -310 MHz
            t1,
            t2,
            t2_star: Some(t2 * 0.8),
            readout_error: 0.02 + (i as f64 * 0.005), // 2-4% error
            thermal_population: 0.01 + (i as f64 * 0.002),
            temperature: Some(15.0), // 15 mK
            parameters: [("flux_offset".to_string(), i as f64 * 0.1)].into(),
        });
    }

    // Add single-qubit gate calibrations
    for gate_name in ["X", "Y", "Z", "H", "RX", "RY", "RZ"] {
        let mut qubit_data = HashMap::new();

        for i in 0..5 {
            let qubit_id = QubitId(i as u16);

            qubit_data.insert(qubit_id, SingleQubitGateData {
                error_rate: 0.001 + (i as f64 * 0.0002), // 0.1-0.18% error
                fidelity: 0.999 - (i as f64 * 0.0002),
                duration: 25.0 + (i as f64 * 2.0), // 25-33 ns
                amplitude: 0.95 + (i as f64 * 0.01),
                frequency: 5e9 + (i as f64 * 10e6),
                phase: i as f64 * 0.01,
                pulse_shape: PulseShape::GaussianDRAG {
                    sigma: 6.0,
                    beta: 0.5 + (i as f64 * 0.05),
                    cutoff: 2.5,
                },
                calibrated_matrix: None,
                parameter_calibrations: None,
            });
        }

        builder = builder.add_single_qubit_gate(
            gate_name.to_string(),
            SingleQubitGateCalibration {
                gate_name: gate_name.to_string(),
                qubit_data,
                default_parameters: GateParameters {
                    amplitude_scale: 1.0,
                    phase_offset: 0.0,
                    duration_scale: 1.0,
                    drag_coefficient: Some(0.5),
                    custom_parameters: HashMap::new(),
                },
            },
        );
    }

    // Add two-qubit gate calibrations (nearest neighbor)
    for i in 0..4 {
        let control = QubitId(i as u16);
        let target = QubitId((i + 1) as u16);

        builder = builder.add_two_qubit_gate(
            control,
            target,
            TwoQubitGateCalibration {
                gate_name: "CNOT".to_string(),
                control,
                target,
                error_rate: 0.01 + (i as f64 * 0.002), // 1-1.6% error
                fidelity: 0.99 - (i as f64 * 0.002),
                duration: 250.0 + (i as f64 * 20.0), // 250-310 ns
                coupling_strength: 25.0 + (i as f64 * 2.0), // 25-31 MHz
                cross_resonance: Some(CrossResonanceParameters {
                    drive_frequency: 4.85e9 + (i as f64 * 10e6),
                    drive_amplitude: 0.4 + (i as f64 * 0.05),
                    pulse_duration: 200.0,
                    echo_amplitude: 0.2,
                    echo_duration: 100.0,
                    zx_interaction_rate: 2.5 + (i as f64 * 0.2),
                }),
                calibrated_matrix: None,
                directional: true,
                reversed_calibration: None,
            },
        );
    }

    // Add readout calibration
    let mut qubit_readout = HashMap::new();
    for i in 0..5 {
        let qubit_id = QubitId(i as u16);

        qubit_readout.insert(qubit_id, QubitReadoutData {
            p0_given_0: 0.98 - (i as f64 * 0.005),
            p1_given_1: 0.97 - (i as f64 * 0.005),
            resonator_frequency: 6.5e9 + (i as f64 * 20e6),
            readout_amplitude: 0.1 + (i as f64 * 0.01),
            readout_phase: i as f64 * 0.1,
            snr: 8.0 - (i as f64 * 0.5),
        });
    }

    builder = builder.readout_calibration(ReadoutCalibration {
        qubit_readout,
        mitigation_matrix: None,
        duration: 2500.0, // 2.5 μs
        integration_time: 2000.0, // 2 μs
    });

    // Add crosstalk matrix
    let mut crosstalk_matrix = vec![vec![0.0; 5]; 5];
    for i in 0..5 {
        crosstalk_matrix[i][i] = 1.0;
        // Nearest neighbor crosstalk
        if i > 0 {
            crosstalk_matrix[i][i-1] = 0.02;
            crosstalk_matrix[i-1][i] = 0.02;
        }
    }

    builder = builder.crosstalk_matrix(CrosstalkMatrix {
        matrix: crosstalk_matrix,
        measurement_method: "Process tomography".to_string(),
        significance_threshold: 0.01,
    });

    // Add topology
    let mut coupling_map = Vec::new();
    for i in 0..4 {
        coupling_map.push((QubitId(i as u16), QubitId((i + 1) as u16)));
    }

    builder = builder.topology(DeviceTopology {
        num_qubits: 5,
        coupling_map,
        layout_type: "linear".to_string(),
        qubit_coordinates: Some(
            (0..5).map(|i| (QubitId(i as u16), (i as f64 * 2.0, 0.0)))
                  .collect()
        ),
    });

    // Add metadata
    builder = builder
        .add_metadata("calibration_method".to_string(), "Randomized benchmarking".to_string())
        .add_metadata("lab_temperature".to_string(), "293 K".to_string())
        .add_metadata("software_version".to_string(), "QuantRS2 v0.1.2".to_string());

    // Build calibration
    let calibration = builder.build()?;

    println!("Created calibration for device: {}", calibration.device_id);
    println!("Number of qubits: {}", calibration.topology.num_qubits);
    println!("Valid for: {:?}", calibration.valid_duration);

    // Save to file
    let mut manager = CalibrationManager::new();
    manager.update_calibration(calibration);
    manager.save_calibration("demo_device_v1", "demo_calibration.json")?;

    println!("Calibration saved to demo_calibration.json\n");

    Ok(())
}

/// Demo 2: Build noise model from calibration
fn demo_noise_model() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Building Noise Model from Calibration");
    println!("========================================");

    // Create an ideal calibration for demonstration
    let calibration = create_ideal_calibration("ideal_device".to_string(), 5);

    // Build standard noise model
    let noise_model = CalibrationNoiseModel::from_calibration(&calibration);

    println!("Standard noise model:");
    println!("- Device: {}", noise_model.device_id);
    println!("- Qubits with noise: {}", noise_model.qubit_noise.len());
    println!("- Gates with noise: {}", noise_model.gate_noise.len());
    println!("- Temperature: {} mK", noise_model.temperature);

    // Build custom noise model with scaling
    let custom_noise = NoiseModelBuilder::from_calibration(calibration.clone())
        .coherent_factor(0.5)    // Reduce coherent errors by 50%
        .thermal_factor(2.0)     // Double thermal noise
        .crosstalk_factor(0.1)   // Reduce crosstalk by 90%
        .readout_factor(0.8)     // Reduce readout errors by 20%
        .build();

    println!("\nCustom noise model with scaling:");
    println!("- Coherent errors scaled by 0.5");
    println!("- Thermal noise scaled by 2.0");
    println!("- Crosstalk scaled by 0.1");
    println!("- Readout errors scaled by 0.8");

    // Show example noise parameters
    if let Some(qubit_noise) = custom_noise.qubit_noise.get(&QubitId(0)) {
        println!("\nQubit 0 noise parameters:");
        println!("- T1 decay rate: {:.6} 1/μs", qubit_noise.gamma_1);
        println!("- T2 dephasing rate: {:.6} 1/μs", qubit_noise.gamma_phi);
        println!("- Thermal population: {:.4}", qubit_noise.thermal_population);
    }

    if let Some(gate_noise) = custom_noise.gate_noise.get("X") {
        println!("\nX gate noise parameters:");
        println!("- Coherent error: {:.6}", gate_noise.coherent_error);
        println!("- Incoherent error: {:.6}", gate_noise.incoherent_error);
        println!("- Duration: {:.1} ns", gate_noise.duration);
    }

    println!();
    Ok(())
}

/// Demo 3: Optimize circuit using calibration
fn demo_circuit_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Circuit Optimization Using Calibration");
    println!("=========================================");

    // Create calibration and optimizer
    let mut manager = CalibrationManager::new();
    let calibration = create_ideal_calibration("opt_device".to_string(), 5);
    manager.update_calibration(calibration);

    let config = OptimizationConfig {
        optimize_fidelity: true,
        optimize_duration: true,
        allow_substitutions: true,
        fidelity_threshold: 0.99,
        consider_crosstalk: true,
        prefer_native_gates: true,
        max_depth_increase: 1.5,
    };

    let optimizer = CalibrationOptimizer::new(manager, config);

    // Create a test circuit
    let mut circuit = Circuit::new(5)?;
    circuit.h(QubitId(0));
    circuit.cnot(QubitId(0), QubitId(1));
    circuit.cnot(QubitId(1), QubitId(2));
    circuit.rz(QubitId(2), std::f64::consts::PI / 4.0);
    circuit.cnot(QubitId(2), QubitId(3));
    circuit.h(QubitId(3));
    circuit.measure_all();

    println!("Original circuit:");
    println!("- Gate count: {}", circuit.gates().len());

    // Optimize circuit
    let result = optimizer.optimize_circuit(&circuit, "opt_device")?;

    println!("\nOptimization result:");
    println!("- Original gates: {}", result.original_gate_count);
    println!("- Optimized gates: {}", result.optimized_gate_count);
    println!("- Estimated fidelity: {:.4}", result.estimated_fidelity);
    println!("- Estimated duration: {:.1} ns", result.estimated_duration);

    if !result.decisions.is_empty() {
        println!("\nOptimization decisions:");
        for decision in &result.decisions {
            match decision {
                OptimizationDecision::GateSubstitution { original, replacement, .. } => {
                    println!("- Substituted {} with {}", original, replacement);
                }
                OptimizationDecision::GateReordering { reason, .. } => {
                    println!("- Reordered gates: {}", reason);
                }
                OptimizationDecision::QubitRemapping { gate, reason, .. } => {
                    println!("- Remapped {} qubits: {}", gate, reason);
                }
                OptimizationDecision::DecompositionChange { gate, original_depth, new_depth, .. } => {
                    println!("- Changed {} decomposition: {} -> {} depth", gate, original_depth, new_depth);
                }
            }
        }
    }

    println!();
    Ok(())
}

/// Demo 4: Estimate circuit fidelity
fn demo_fidelity_estimation() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Circuit Fidelity Estimation");
    println!("==============================");

    // Create a realistic calibration with some errors
    let calibration = create_ideal_calibration("fidelity_device".to_string(), 5);
    let estimator = FidelityEstimator::new(calibration.clone());

    // Create circuits of varying complexity
    let circuits = vec![
        ("Single qubit", create_single_qubit_circuit()?),
        ("Bell pair", create_bell_pair_circuit()?),
        ("GHZ state", create_ghz_circuit(3)?),
        ("Deep circuit", create_deep_circuit(5, 20)?),
    ];

    println!("Circuit fidelity estimates:");
    println!("Circuit Type    | Gates | Process Fidelity | State Fidelity (w/ decoherence)");
    println!("----------------|-------|------------------|--------------------------------");

    for (name, circuit) in circuits {
        let gate_count = circuit.gates().len();
        let process_fidelity = estimator.estimate_process_fidelity(&circuit)?;
        let state_fidelity = estimator.estimate_state_fidelity(&circuit, true)?;

        println!("{:15} | {:5} | {:16.4} | {:31.4}",
            name, gate_count, process_fidelity, state_fidelity
        );
    }

    println!("\nNote: State fidelity includes T1/T2 decoherence effects\n");

    Ok(())
}

/// Demo 5: Manage multiple device calibrations
fn demo_calibration_management() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Managing Multiple Device Calibrations");
    println!("========================================");

    let mut manager = CalibrationManager::new();

    // Add calibrations for multiple devices
    let devices = vec![
        ("device_a", 5, 50_000.0, 0.99),  // 5 qubits, 50μs T1, 0.99 gate fidelity
        ("device_b", 7, 80_000.0, 0.995), // 7 qubits, 80μs T1, 0.995 gate fidelity
        ("device_c", 10, 30_000.0, 0.98), // 10 qubits, 30μs T1, 0.98 gate fidelity
    ];

    for (device_id, n_qubits, t1, gate_fidelity) in devices {
        let mut cal = create_ideal_calibration(device_id.to_string(), n_qubits);

        // Modify to reflect device characteristics
        for qubit_cal in cal.qubit_calibrations.values_mut() {
            qubit_cal.t1 = t1;
            qubit_cal.t2 = t1 * 0.8;
        }

        for gate_cal in cal.single_qubit_gates.values_mut() {
            for data in gate_cal.qubit_data.values_mut() {
                data.fidelity = gate_fidelity;
                data.error_rate = 1.0 - gate_fidelity;
            }
        }

        manager.update_calibration(cal);
    }

    // Show calibration summary
    println!("Device calibrations in manager:");
    println!("Device   | Qubits | Valid | Avg T1 (μs) | Gate Fidelity");
    println!("---------|--------|-------|-------------|---------------");

    for device_id in ["device_a", "device_b", "device_c"] {
        if let Some(cal) = manager.get_calibration(device_id) {
            let avg_t1 = cal.qubit_calibrations.values()
                .map(|q| q.t1)
                .sum::<f64>() / cal.qubit_calibrations.len() as f64;

            let gate_fidelity = cal.single_qubit_gates.get("X")
                .and_then(|g| g.qubit_data.get(&QubitId(0)))
                .map(|d| d.fidelity)
                .unwrap_or(0.0);

            println!("{:8} | {:6} | {:5} | {:11.0} | {:.3}",
                device_id,
                cal.topology.num_qubits,
                if manager.is_calibration_valid(device_id) { "Yes" } else { "No" },
                avg_t1,
                gate_fidelity
            );
        }
    }

    // Find best device for specific criteria
    println!("\nBest device for high fidelity: device_b");
    println!("Best device for long coherence: device_b");
    println!("Best device for many qubits: device_c");

    Ok(())
}

// Helper functions to create test circuits

fn create_single_qubit_circuit() -> Result<Circuit, Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(1)?;
    circuit.h(QubitId(0));
    circuit.s(QubitId(0));
    circuit.t(QubitId(0));
    Ok(circuit)
}

fn create_bell_pair_circuit() -> Result<Circuit, Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(2)?;
    circuit.h(QubitId(0));
    circuit.cnot(QubitId(0), QubitId(1));
    Ok(circuit)
}

fn create_ghz_circuit(n: usize) -> Result<Circuit, Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(n)?;
    circuit.h(QubitId(0));
    for i in 0..n-1 {
        circuit.cnot(QubitId(i as u16), QubitId((i + 1) as u16));
    }
    Ok(circuit)
}

fn create_deep_circuit(n_qubits: usize, depth: usize) -> Result<Circuit, Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(n_qubits)?;

    for _ in 0..depth {
        // Random single-qubit gates
        for q in 0..n_qubits {
            match q % 3 {
                0 => circuit.h(QubitId(q as u16)),
                1 => circuit.rx(QubitId(q as u16), 0.5),
                _ => circuit.rz(QubitId(q as u16), 0.7),
            }
        }

        // Two-qubit gates
        for q in 0..n_qubits-1 {
            if q % 2 == 0 {
                circuit.cnot(QubitId(q as u16), QubitId((q + 1) as u16));
            }
        }
    }

    Ok(circuit)
}