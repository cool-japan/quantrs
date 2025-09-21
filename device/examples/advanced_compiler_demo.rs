//! Advanced Compiler Demo
//!
//! This example demonstrates the comprehensive hardware-specific compiler passes
//! with SciRS2 integration, showing multi-platform compilation, advanced optimization,
//! and performance analysis capabilities.

use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use quantrs2_device::{
    backend_traits::BackendCapabilities,
    calibration::create_ideal_calibration,
    compiler_passes::{
        AnalysisDepth, AzureProvider, BraketProvider, CompilationResult, CompilationTarget,
        CompilerConfig, HardwareCompiler, OptimizationObjective, SciRS2Config,
        SciRS2OptimizationMethod,
    },
    prelude::OptimizationLevel,
    topology_analysis::create_standard_topology,
};
use std::collections::HashMap;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Advanced Quantum Circuit Compiler Demo with SciRS2 Integration");
    println!("================================================================");

    // Demo 1: Multi-Platform Compilation
    demo_multi_platform_compilation().await?;

    // Demo 2: Advanced Circuit Optimization
    demo_advanced_optimization().await?;

    // Demo 3: SciRS2 Integration
    demo_scirs2_integration().await?;

    // Demo 4: Performance Analysis
    demo_performance_analysis().await?;

    // Demo 5: Adaptive Compilation
    demo_adaptive_compilation().await?;

    println!("\n✅ All demos completed successfully!");
    Ok(())
}

/// Demonstrate multi-platform compilation targeting different quantum platforms
async fn demo_multi_platform_compilation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📱 Demo 1: Multi-Platform Compilation");
    println!("-------------------------------------");

    // Create test circuit
    let mut circuit = Circuit::<4>::new();
    let _ = circuit.h(QubitId(0));
    let _ = circuit.cnot(QubitId(0), QubitId(1));
    let _ = circuit.cnot(QubitId(1), QubitId(2));
    let _ = circuit.cnot(QubitId(2), QubitId(3));
    let _ = circuit.h(QubitId(3));

    // IBM Quantum compilation
    println!("🔬 Compiling for IBM Quantum...");
    let ibm_config = create_ibm_config();
    let ibm_result = compile_for_platform(circuit.clone(), ibm_config).await?;
    print_compilation_summary("IBM Quantum", &ibm_result);

    // AWS Braket compilation
    println!("☁️ Compiling for AWS Braket...");
    let aws_config = create_aws_config();
    let aws_result = compile_for_platform(circuit.clone(), aws_config).await?;
    print_compilation_summary("AWS Braket", &aws_result);

    // Azure Quantum compilation
    println!("🌐 Compiling for Azure Quantum...");
    let azure_config = create_azure_config();
    let azure_result = compile_for_platform(circuit.clone(), azure_config).await?;
    print_compilation_summary("Azure Quantum", &azure_result);

    Ok(())
}

/// Demonstrate advanced circuit optimization capabilities
async fn demo_advanced_optimization() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n⚡ Demo 2: Advanced Circuit Optimization");
    println!("---------------------------------------");

    // Create a more complex circuit for optimization
    let mut circuit = Circuit::<8>::new();

    // Create entangled state with redundancies
    for i in 0..7 {
        let _ = circuit.h(QubitId(i));
        let _ = circuit.cnot(QubitId(i), QubitId(i + 1));
    }

    // Add some redundant operations
    let _ = circuit.z(QubitId(0));
    let _ = circuit.z(QubitId(0)); // Redundant - should be optimized away
    let _ = circuit.x(QubitId(1));
    let _ = circuit.x(QubitId(1)); // Redundant - should be optimized away

    println!(
        "📊 Original circuit: {} gates, estimated depth: {}",
        circuit.gates().len(),
        estimate_circuit_depth(&circuit)
    );

    // Create optimization-focused config
    let mut config = CompilerConfig::default();
    config.objectives = vec![
        OptimizationObjective::MinimizeGateCount,
        OptimizationObjective::MinimizeDepth,
        OptimizationObjective::MaximizeFidelity,
    ];
    config.scirs2_config.enable_advanced_optimization = true;

    let result = compile_for_platform(circuit, config).await?;

    println!("📈 Optimization Results:");
    println!(
        "  - Gate count: {} → {} ({}% reduction)",
        result.optimization_stats.original_gate_count,
        result.optimization_stats.optimized_gate_count,
        ((result.optimization_stats.original_gate_count
            - result.optimization_stats.optimized_gate_count) as f64
            / result.optimization_stats.original_gate_count as f64
            * 100.0)
    );

    println!(
        "  - Circuit depth: {} → {} ({}% reduction)",
        result.optimization_stats.original_depth,
        result.optimization_stats.optimized_depth,
        ((result.optimization_stats.original_depth - result.optimization_stats.optimized_depth)
            as f64
            / result.optimization_stats.original_depth as f64
            * 100.0)
    );

    Ok(())
}

/// Demonstrate SciRS2 integration for advanced algorithms
async fn demo_scirs2_integration() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🧮 Demo 3: SciRS2 Integration");
    println!("-----------------------------");

    // Create circuit with complex connectivity requirements
    let mut circuit = Circuit::<6>::new();

    // Create a pattern that benefits from graph optimization
    let _ = circuit.h(QubitId(0));
    let _ = circuit.cnot(QubitId(0), QubitId(2));
    let _ = circuit.cnot(QubitId(2), QubitId(4));
    let _ = circuit.cnot(QubitId(1), QubitId(3));
    let _ = circuit.cnot(QubitId(3), QubitId(5));
    let _ = circuit.cnot(QubitId(0), QubitId(5));

    // Enable comprehensive SciRS2 features
    let mut config = CompilerConfig::default();
    config.scirs2_config = SciRS2Config {
        enable_graph_optimization: true,
        enable_statistical_analysis: true,
        enable_advanced_optimization: true,
        enable_linalg_optimization: true,
        optimization_method: SciRS2OptimizationMethod::GeneticAlgorithm,
        significance_threshold: 0.01,
    };

    let result = compile_for_platform(circuit, config).await?;

    println!("🔬 SciRS2 Analysis Results:");
    println!(
        "  - Complexity score: {:.3}",
        result.advanced_metrics.complexity_score
    );
    println!(
        "  - Resource efficiency: {:.2}%",
        result.advanced_metrics.resource_efficiency * 100.0
    );
    println!(
        "  - Error resilience: {:.3}",
        result.advanced_metrics.error_resilience
    );
    println!(
        "  - Quantum volume: {}",
        result.advanced_metrics.quantum_volume
    );

    println!(
        "  - Statistical analysis: {} passes applied",
        result
            .applied_passes
            .iter()
            .filter(|p| p.name.contains("Statistical"))
            .count()
    );

    Ok(())
}

/// Demonstrate performance analysis and monitoring
async fn demo_performance_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📈 Demo 4: Performance Analysis");
    println!("-------------------------------");

    // Create a moderately complex circuit
    let mut circuit = Circuit::<5>::new();

    // Quantum Fourier Transform-like pattern
    for i in 0..5 {
        let _ = circuit.h(QubitId(i));
        for j in (i + 1)..5 {
            // Controlled phase rotations (simplified)
            let _ = circuit.cnot(QubitId(i), QubitId(j));
        }
    }

    // Enable performance monitoring
    let mut config = CompilerConfig::default();
    config.performance_monitoring = true;
    config.analysis_depth = AnalysisDepth::Comprehensive;

    let result = compile_for_platform(circuit, config).await?;

    println!("⏱️ Performance Metrics:");
    println!(
        "  - Total compilation time: {:.2}ms",
        result.compilation_time.as_millis()
    );

    println!("  - Pass execution breakdown:");
    for pass in &result.applied_passes {
        println!(
            "    • {}: {:.2}ms ({:.1}% improvement)",
            pass.name,
            pass.execution_time.as_millis(),
            pass.improvement * 100.0
        );
    }

    println!(
        "  - Compatibility score: {:.3}",
        result.advanced_metrics.compatibility_score
    );

    println!(
        "  - Expressivity: {:.3}",
        result.advanced_metrics.expressivity
    );

    Ok(())
}

/// Demonstrate adaptive compilation strategies
async fn demo_adaptive_compilation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🤖 Demo 5: Adaptive Compilation");
    println!("-------------------------------");

    // Create circuits of different complexities
    let circuits = vec![
        create_simple_circuit(),
        create_medium_circuit(),
        create_complex_circuit(),
    ];

    let names = vec!["Simple", "Medium", "Complex"];

    for (i, circuit) in circuits.into_iter().enumerate() {
        println!("🔄 Adaptive compilation for {} circuit...", names[i]);

        // Use adaptive configuration
        let mut config = CompilerConfig::default();
        config.analysis_depth = match i {
            0 => AnalysisDepth::Basic,
            1 => AnalysisDepth::Intermediate,
            2 => AnalysisDepth::Comprehensive,
            _ => AnalysisDepth::Advanced,
        };

        let result = compile_for_platform(circuit, config).await?;

        println!(
            "  - Optimization iterations: {}",
            result.optimization_history.len()
        );
        println!("  - Passes applied: {}", result.applied_passes.len());
        println!(
            "  - Final fidelity estimate: {:.4}",
            result.predicted_performance.fidelity
        );
    }

    Ok(())
}

// Helper functions for creating different platform configurations

fn create_ibm_config() -> CompilerConfig {
    let mut config = CompilerConfig::default();
    config.target = CompilationTarget::IBMQuantum {
        backend_name: "ibmq_qasm_simulator".to_string(),
        coupling_map: vec![(0, 1), (1, 2), (2, 3), (1, 4)],
        native_gates: ["rz", "sx", "cx"].iter().map(|s| s.to_string()).collect(),
        basis_gates: vec!["rz".to_string(), "sx".to_string(), "cx".to_string()],
        max_shots: 8192,
        simulator: true,
    };
    config
}

fn create_aws_config() -> CompilerConfig {
    let mut config = CompilerConfig::default();
    config.target = CompilationTarget::AWSBraket {
        device_arn: "arn:aws:braket:::device/quantum-simulator/amazon/sv1".to_string(),
        provider: BraketProvider::IonQ,
        supported_gates: ["x", "y", "z", "h", "cnot", "swap"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        max_shots: 1000,
        cost_per_shot: 0.00075,
    };
    config
}

fn create_azure_config() -> CompilerConfig {
    let mut config = CompilerConfig::default();
    config.target = CompilationTarget::AzureQuantum {
        workspace: "quantum-workspace-1".to_string(),
        target: "ionq.simulator".to_string(),
        provider: AzureProvider::IonQ,
        supported_operations: ["x", "y", "z", "h", "cnot", "swap"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        resource_estimation: true,
    };
    config
}

// Helper function for compilation
async fn compile_for_platform<const N: usize>(
    circuit: Circuit<N>,
    config: CompilerConfig,
) -> Result<CompilationResult, Box<dyn std::error::Error>> {
    let topology = create_standard_topology("linear", N)?;
    let calibration = create_ideal_calibration("demo".to_string(), N);
    let backend_capabilities = BackendCapabilities::default();

    let compiler =
        HardwareCompiler::new(config, topology, calibration, None, backend_capabilities)?;

    let result = compiler.compile_circuit(&circuit).await?;
    Ok(result)
}

fn print_compilation_summary(platform: &str, result: &CompilationResult) {
    println!(
        "  ✓ {} compilation completed in {:.2}ms",
        platform,
        result.compilation_time.as_millis()
    );
    println!("    - Passes applied: {}", result.applied_passes.len());
    println!(
        "    - Expected fidelity: {:.4}",
        result.predicted_performance.fidelity
    );
    println!(
        "    - Success probability: {:.1}%",
        result.predicted_performance.success_probability * 100.0
    );
}

fn estimate_circuit_depth<const N: usize>(circuit: &Circuit<N>) -> usize {
    // Simplified depth estimation
    circuit.gates().len() / 2
}

// Helper functions for creating test circuits

fn create_simple_circuit() -> Circuit<3> {
    let mut circuit = Circuit::<3>::new();
    let _ = circuit.h(QubitId(0));
    let _ = circuit.cnot(QubitId(0), QubitId(1));
    let _ = circuit.cnot(QubitId(1), QubitId(2));
    circuit
}

fn create_medium_circuit() -> Circuit<3> {
    let mut circuit = Circuit::<3>::new();
    for i in 0..3 {
        let _ = circuit.h(QubitId(i));
    }
    for i in 0..2 {
        let _ = circuit.cnot(QubitId(i), QubitId(i + 1));
    }
    let _ = circuit.cnot(QubitId(2), QubitId(0)); // Add cycle
    circuit
}

fn create_complex_circuit() -> Circuit<3> {
    let mut circuit = Circuit::<3>::new();

    // Create complex entanglement pattern
    for i in 0..3 {
        let _ = circuit.h(QubitId(i));
    }

    // Create multiple CNOT layers
    for layer in 0..2 {
        for i in 0..2 {
            let target = (i + layer + 1) % 3;
            let _ = circuit.cnot(QubitId(i), QubitId(target));
        }
    }

    // Add some single-qubit rotations
    for i in 0..3 {
        let _ = circuit.z(QubitId(i));
    }

    circuit
}
