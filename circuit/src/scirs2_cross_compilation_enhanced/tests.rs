//! Tests for cross-compilation module

use super::*;
use std::collections::HashMap;

#[test]
fn test_cross_compiler_creation() {
    let config = EnhancedCrossCompilationConfig::default();
    let compiler = EnhancedCrossCompiler::new(config);

    // Basic test to ensure creation works
    assert!(compiler.config.enable_ml_optimization);
}

#[test]
fn test_source_circuit() {
    let source = SourceCircuit {
        framework: QuantumFramework::QuantRS2,
        code: "// Quantum circuit".to_string(),
        metadata: HashMap::new(),
    };

    assert_eq!(source.framework, QuantumFramework::QuantRS2);
}

#[test]
fn test_config_defaults() {
    let config = EnhancedCrossCompilationConfig::default();

    assert!(config.enable_ml_optimization);
    assert!(config.enable_multistage_compilation);
    assert!(config.enable_target_optimization);
    assert!(config.enable_realtime_monitoring);
    assert!(config.enable_comprehensive_validation);
    assert!(config.enable_visual_flow);
    assert!(!config.source_frameworks.is_empty());
    assert!(!config.target_platforms.is_empty());
}

#[test]
fn test_quantum_ir_creation() {
    let mut ir = QuantumIR::new();
    ir.num_qubits = 5;
    ir.num_classical_bits = 5;

    assert_eq!(ir.num_qubits, 5);
    assert_eq!(ir.num_classical_bits, 5);
    assert!(ir.operations.is_empty());
}

#[test]
fn test_target_code_creation() {
    let code = TargetCode::new(TargetPlatform::IBMQuantum);

    assert_eq!(code.platform, TargetPlatform::IBMQuantum);
    assert!(code.code.is_empty());
}

#[test]
fn test_validation_result_creation() {
    let result = ValidationResult::new();

    assert!(result.is_valid);
    assert!(result.errors.is_empty());
    assert!(result.warnings.is_empty());
}

#[test]
fn test_compilation_report_creation() {
    let report = CompilationReport::new();

    assert!(report.stage_analyses.is_empty());
    assert!(report.recommendations.is_empty());
}

#[test]
fn test_visual_flow_creation() {
    let mut flow = VisualCompilationFlow::new();

    flow.add_node(FlowNode {
        id: 0,
        name: "Test".to_string(),
        node_type: NodeType::CompilationStage,
        metrics: HashMap::new(),
    });

    assert_eq!(flow.nodes.len(), 1);
}

#[test]
fn test_batch_result_creation() {
    let batch = BatchCompilationResult::new();

    assert!(batch.successful_compilations.is_empty());
    assert!(batch.failed_compilations.is_empty());
}
