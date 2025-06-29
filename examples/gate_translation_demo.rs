//! Demonstration of gate translation for different hardware backends
//!
//! This example shows how to translate quantum circuits between different
//! hardware native gate sets, enabling portability across quantum platforms.

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::qubit::QubitId;
use quantrs2_device::translation::*;
use quantrs2_device::backend_traits::*;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== QuantRS2 Gate Translation Demo ===\n");

    // Demo 1: Basic gate translation
    demo_basic_translation()?;

    // Demo 2: Circuit translation
    demo_circuit_translation()?;

    // Demo 3: Backend comparison
    demo_backend_comparison()?;

    // Demo 4: Translation optimization
    demo_translation_optimization()?;

    // Demo 5: Hardware-specific gates
    demo_hardware_specific_gates()?;

    // Demo 6: Backend capabilities
    demo_backend_capabilities()?;

    Ok(())
}

/// Demo 1: Basic gate translation
fn demo_basic_translation() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Basic Gate Translation");
    println!("=========================");

    let mut translator = GateTranslator::new();

    // Create some gates to translate
    use quantrs2_core::gate::{single::*, multi::*};

    let gates = vec![
        ("Hadamard", Box::new(Hadamard { target: QubitId(0) }) as Box<dyn GateOp>),
        ("Pauli-Y", Box::new(PauliY { target: QubitId(0) }) as Box<dyn GateOp>),
        ("S gate", Box::new(SGate { target: QubitId(0) }) as Box<dyn GateOp>),
        ("T gate", Box::new(TGate { target: QubitId(0) }) as Box<dyn GateOp>),
        ("RX(π/4)", Box::new(RotationX { target: QubitId(0), theta: std::f64::consts::PI / 4.0 }) as Box<dyn GateOp>),
    ];

    let backends = vec![
        ("IBM Quantum", HardwareBackend::IBMQuantum),
        ("Google Sycamore", HardwareBackend::GoogleSycamore),
        ("IonQ", HardwareBackend::IonQ),
    ];

    for (gate_name, gate) in &gates {
        println!("\n{} gate translation:", gate_name);
        println!("----------------------------------------");

        for (backend_name, backend) in &backends {
            let decomposed = translator.translate_gate(gate.as_ref(), *backend)?;

            print!("{:15} -> ", backend_name);
            for (i, dec_gate) in decomposed.iter().enumerate() {
                if i > 0 { print!(", "); }
                print!("{}", dec_gate.native_gate);
                if !dec_gate.parameters.is_empty() {
                    print!("(");
                    for (j, param) in dec_gate.parameters.iter().enumerate() {
                        if j > 0 { print!(", "); }
                        print!("{:.3}", param);
                    }
                    print!(")");
                }
            }
            println!();
        }
    }

    println!();
    Ok(())
}

/// Demo 2: Circuit translation
fn demo_circuit_translation() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Circuit Translation");
    println!("======================");

    // Create a quantum algorithm circuit
    let mut circuit = Circuit::new(3)?;

    // Quantum Fourier Transform on 3 qubits
    circuit.h(QubitId(0));
    circuit.cp(QubitId(0), QubitId(1), std::f64::consts::PI / 2.0);
    circuit.cp(QubitId(0), QubitId(2), std::f64::consts::PI / 4.0);
    circuit.h(QubitId(1));
    circuit.cp(QubitId(1), QubitId(2), std::f64::consts::PI / 2.0);
    circuit.h(QubitId(2));
    circuit.swap(QubitId(0), QubitId(2));

    println!("Original circuit (QFT-3):");
    println!("- Gates: {}", circuit.gates().len());
    let gate_counts = count_gates(&circuit);
    for (gate, count) in &gate_counts {
        println!("  - {}: {}", gate, count);
    }

    let mut translator = GateTranslator::new();

    // Translate to different backends
    let backends = vec![
        ("IBM Quantum", HardwareBackend::IBMQuantum),
        ("IonQ", HardwareBackend::IonQ),
        ("Rigetti", HardwareBackend::Rigetti),
    ];

    println!("\nTranslated circuits:");
    for (backend_name, backend) in backends {
        let translated = translator.translate_circuit(&circuit, backend)?;
        let stats = TranslationStats::calculate(&circuit, &translated, backend);

        println!("\n{}:", backend_name);
        println!("- Native gates: {}", stats.native_gates);
        println!("- Expansion factor: {:.2}x", stats.native_gates as f64 / stats.original_gates as f64);
        println!("- Gate breakdown:");

        for (gate, count) in stats.gate_counts.iter() {
            println!("  - {}: {}", gate, count);
        }
    }

    println!();
    Ok(())
}

/// Demo 3: Backend comparison
fn demo_backend_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Backend Comparison");
    println!("=====================");

    let translator = GateTranslator::new();

    // Get native gate sets for each backend
    let backends = vec![
        HardwareBackend::IBMQuantum,
        HardwareBackend::GoogleSycamore,
        HardwareBackend::IonQ,
        HardwareBackend::Rigetti,
        HardwareBackend::Honeywell,
    ];

    println!("Native Gate Sets:");
    println!("-----------------");

    for backend in &backends {
        if let Some(gate_set) = translator.get_native_gates(*backend) {
            println!("\n{:?}:", backend);
            println!("  Single-qubit gates: {:?}", gate_set.single_qubit_gates);
            println!("  Two-qubit gates: {:?}", gate_set.two_qubit_gates);
            if !gate_set.multi_qubit_gates.is_empty() {
                println!("  Multi-qubit gates: {:?}", gate_set.multi_qubit_gates);
            }
            println!("  Arbitrary rotations: {}", gate_set.arbitrary_single_qubit);
            println!("  Virtual Z: {}", gate_set.constraints.virtual_z);
        }
    }

    // Compare CNOT implementations
    println!("\n\nCNOT Gate Implementations:");
    println!("--------------------------");

    use quantrs2_core::gate::multi::CNOT;
    let cnot = CNOT { control: QubitId(0), target: QubitId(1) };
    let mut translator = GateTranslator::new();

    for backend in &backends {
        let decomposed = translator.translate_gate(&cnot, *backend)?;
        println!("\n{:?}:", backend);

        if decomposed.len() == 1 && decomposed[0].native_gate == "cnot" || decomposed[0].native_gate == "cx" {
            println!("  Native CNOT implementation");
        } else {
            println!("  Decomposed into {} gates:", decomposed.len());
            for gate in &decomposed {
                println!("    - {}", gate.native_gate);
            }
        }
    }

    println!();
    Ok(())
}

/// Demo 4: Translation optimization
fn demo_translation_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Translation Optimization");
    println!("===========================");

    // Create optimizers with different strategies
    let strategies = vec![
        ("Minimize Gates", OptimizationStrategy::MinimizeGateCount),
        ("Minimize Error", OptimizationStrategy::MinimizeError),
        ("Minimize Depth", OptimizationStrategy::MinimizeDepth),
        ("Balanced", OptimizationStrategy::Balanced { weight: 0.5 }),
    ];

    // Test gate: Controlled-Z
    use quantrs2_core::gate::multi::ControlledZ;
    let cz = ControlledZ { control: QubitId(0), target: QubitId(1) };

    println!("Optimizing CZ gate translation for IBM backend:\n");

    for (strategy_name, strategy) in strategies {
        let mut optimizer = TranslationOptimizer::new(strategy);
        let optimized = optimizer.optimize_translation(&cz, HardwareBackend::IBMQuantum)?;

        println!("{:15} -> {} gates", strategy_name, optimized.len());
    }

    // Compare a complex circuit
    let mut circuit = Circuit::new(4)?;

    // Random circuit with various gates
    circuit.h(QubitId(0));
    circuit.cnot(QubitId(0), QubitId(1));
    circuit.ry(QubitId(1), 0.7);
    circuit.cz(QubitId(1), QubitId(2));
    circuit.ccx(QubitId(0), QubitId(1), QubitId(3));
    circuit.swap(QubitId(2), QubitId(3));

    println!("\n\nOptimizing complex circuit:");
    println!("Original gates: {}", circuit.gates().len());

    let mut translator = GateTranslator::new();
    let backends = vec![
        HardwareBackend::IBMQuantum,
        HardwareBackend::IonQ,
    ];

    for backend in backends {
        let translated = translator.translate_circuit(&circuit, backend)?;
        println!("\n{:?}:", backend);
        println!("  Translated gates: {}", translated.gates().len());

        // Validate translation
        let valid = validate_native_circuit(&translated, backend)?;
        println!("  Uses only native gates: {}", valid);
    }

    println!();
    Ok(())
}

/// Demo 5: Hardware-specific gates
fn demo_hardware_specific_gates() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Hardware-Specific Gates");
    println!("==========================");

    // IBM SX gate
    println!("\nIBM SX Gate (√X):");
    let sx = ibm_gates::SXGate { target: QubitId(0) };
    println!("- Name: {}", sx.name());
    println!("- Backend: {:?}", sx.backend());
    let metadata = sx.metadata();
    println!("- Duration: {} ns", metadata.get("duration_ns").unwrap_or(&"N/A".to_string()));

    // Google Sycamore gate
    println!("\nGoogle Sycamore Gate:");
    let syc = google_gates::SycamoreGate {
        qubit1: QubitId(0),
        qubit2: QubitId(1),
    };
    println!("- Name: {}", syc.name());
    println!("- Backend: {:?}", syc.backend());
    let metadata = syc.metadata();
    println!("- Duration: {} ns", metadata.get("duration_ns").unwrap_or(&"N/A".to_string()));
    println!("- Fidelity: {}", metadata.get("fidelity").unwrap_or(&"N/A".to_string()));

    // IonQ XX gate
    println!("\nIonQ XX Gate (Mølmer-Sørensen):");
    let xx = ionq_gates::XXGate {
        qubit1: QubitId(0),
        qubit2: QubitId(1),
        angle: std::f64::consts::PI / 2.0,
    };
    println!("- Name: {}", xx.name());
    println!("- Backend: {:?}", xx.backend());
    println!("- Angle: π/2");
    println!("- Calibration params: {:?}", xx.calibration_params());

    // Honeywell ZZ gate
    println!("\nHoneywell ZZ Gate:");
    let zz = honeywell_gates::ZZGate {
        qubit1: QubitId(0),
        qubit2: QubitId(1),
        angle: 0.5,
    };
    println!("- Name: {}", zz.name());
    println!("- Backend: {:?}", zz.backend());
    let metadata = zz.metadata();
    println!("- Gate type: {}", metadata.get("gate_type").unwrap_or(&"N/A".to_string()));
    println!("- Fidelity: {}", metadata.get("fidelity").unwrap_or(&"N/A".to_string()));

    println!();
    Ok(())
}

/// Demo 6: Backend capabilities
fn demo_backend_capabilities() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Backend Capabilities");
    println!("=======================");

    let backends = vec![
        HardwareBackend::IBMQuantum,
        HardwareBackend::GoogleSycamore,
        HardwareBackend::IonQ,
        HardwareBackend::Honeywell,
    ];

    println!("Backend Feature Comparison:");
    println!("---------------------------");
    println!("{:15} | {:8} | {:8} | {:10} | {:8} | {:8}",
        "Backend", "Max Qubits", "Mid-Meas", "Conditional", "Pulse", "All-to-All"
    );
    println!("{:-<15}-+-{:-<8}-+-{:-<8}-+-{:-<10}-+-{:-<8}-+-{:-<8}",
        "", "", "", "", "", ""
    );

    for backend in &backends {
        let caps = query_backend_capabilities(*backend);

        let all_to_all = caps.native_gates.constraints.coupling_map.is_none();

        println!("{:15} | {:8} | {:8} | {:10} | {:8} | {:8}",
            format!("{:?}", backend).replace("HardwareBackend::", ""),
            caps.features.max_qubits,
            if caps.features.mid_circuit_measurement { "Yes" } else { "No" },
            if caps.features.conditional_gates { "Yes" } else { "No" },
            if caps.features.pulse_control { "Yes" } else { "No" },
            if all_to_all { "Yes" } else { "No" }
        );
    }

    println!("\n\nBackend Performance Characteristics:");
    println!("------------------------------------");
    println!("{:15} | {:12} | {:12} | {:8} | {:8} | {:12}",
        "Backend", "1Q Time (ns)", "2Q Time (ns)", "T1 (μs)", "T2 (μs)", "2Q Fidelity"
    );
    println!("{:-<15}-+-{:-<12}-+-{:-<12}-+-{:-<8}-+-{:-<8}-+-{:-<12}",
        "", "", "", "", "", ""
    );

    for backend in &backends {
        let caps = query_backend_capabilities(*backend);
        let perf = &caps.performance;

        println!("{:15} | {:12.1} | {:12.1} | {:8.1} | {:8.1} | {:12.3}",
            format!("{:?}", backend).replace("HardwareBackend::", ""),
            perf.single_qubit_gate_time,
            perf.two_qubit_gate_time,
            perf.t1_time,
            perf.t2_time,
            perf.two_qubit_fidelity
        );
    }

    println!();
    Ok(())
}

// Helper function to count gates in a circuit
fn count_gates(circuit: &Circuit) -> HashMap<String, usize> {
    let mut counts = HashMap::new();

    for gate in circuit.gates() {
        *counts.entry(gate.name().to_string()).or_insert(0) += 1;
    }

    counts
}