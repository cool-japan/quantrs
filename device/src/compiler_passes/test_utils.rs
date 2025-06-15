//! Test utilities and helper functions

use std::collections::HashSet;

use super::config::*;
use super::types::*;
use crate::backend_traits::BackendCapabilities;

/// Utility functions for creating test configurations
pub fn create_test_ibm_target() -> CompilationTarget {
    CompilationTarget::IBMQuantum {
        backend_name: "test_backend".to_string(),
        coupling_map: vec![(0, 1), (1, 2), (2, 3)],
        native_gates: ["rz", "sx", "cx"].iter().map(|s| s.to_string()).collect(),
        basis_gates: vec!["rz".to_string(), "sx".to_string(), "cx".to_string()],
        max_shots: 1024,
        simulator: true,
    }
}

pub fn create_test_scirs2_config() -> SciRS2Config {
    SciRS2Config {
        enable_graph_optimization: true,
        enable_statistical_analysis: true,
        enable_advanced_optimization: false, // Disable for testing
        enable_linalg_optimization: true,
        optimization_method: SciRS2OptimizationMethod::NelderMead,
        significance_threshold: 0.01,
    }
}

pub fn create_test_compiler_config() -> CompilerConfig {
    CompilerConfig {
        enable_gate_synthesis: true,
        enable_error_optimization: true,
        enable_timing_optimization: true,
        enable_crosstalk_mitigation: false, // Disable for simpler testing
        enable_resource_optimization: true,
        max_iterations: 100,
        tolerance: 1e-6,
        target: create_test_ibm_target(),
        objectives: vec![OptimizationObjective::MinimizeError],
        constraints: HardwareConstraints {
            max_depth: Some(100),
            max_gates: Some(1000),
            max_execution_time: Some(10000.0),
            min_fidelity_threshold: 0.95,
            max_error_rate: 0.05,
            forbidden_pairs: HashSet::new(),
            min_idle_time: 50.0,
        },
        scirs2_config: create_test_scirs2_config(),
        parallel_config: ParallelConfig {
            enable_parallel_passes: false, // Disable for deterministic testing
            num_threads: 1,
            chunk_size: 10,
            enable_simd: false,
        },
        adaptive_config: None,
        performance_monitoring: true,
        analysis_depth: AnalysisDepth::Basic,
    }
}

/// Helper function to create test grid topology
pub fn create_test_grid_topology() -> GridTopology {
    GridTopology {
        rows: 2,
        cols: 2,
        connectivity: ConnectivityPattern::Square,
    }
}

/// Helper function to create test rigetti lattice
pub fn create_test_rigetti_lattice() -> RigettiLattice {
    RigettiLattice::Custom(vec![(0, 1), (1, 2), (2, 3)])
}

/// Helper function to create test performance prediction
pub fn create_test_performance_prediction() -> PerformancePrediction {
    PerformancePrediction {
        execution_time: std::time::Duration::from_micros(500),
        fidelity: 0.92,
        error_rate: 0.08,
        success_probability: 0.85,
        confidence_interval: (0.80, 0.90),
        model: "Test Model".to_string(),
    }
}

/// Helper function to create test advanced metrics
pub fn create_test_advanced_metrics() -> AdvancedMetrics {
    AdvancedMetrics {
        quantum_volume: 32,
        expressivity: 0.75,
        entanglement_entropy: 1.5,
        complexity_score: 0.65,
        resource_efficiency: 0.85,
        error_resilience: 0.88,
        compatibility_score: 0.92,
    }
}

/// Helper function to create test optimization stats
pub fn create_test_optimization_stats() -> OptimizationStats {
    OptimizationStats {
        original_gate_count: 20,
        optimized_gate_count: 15,
        original_depth: 10,
        optimized_depth: 8,
        error_improvement: 0.12,
        fidelity_improvement: 0.08,
        efficiency_gain: 0.25,
        overall_improvement: 0.15,
    }
}

/// Helper function to create test hardware allocation
pub fn create_test_hardware_allocation() -> HardwareAllocation {
    let mut qubit_mapping = std::collections::HashMap::new();
    qubit_mapping.insert(0, 0);
    qubit_mapping.insert(1, 1);
    qubit_mapping.insert(2, 2);
    qubit_mapping.insert(3, 3);

    HardwareAllocation {
        qubit_mapping,
        allocated_qubits: vec![0, 1, 2, 3],
        resource_utilization: 0.75,
        strategy: AllocationStrategy::OptimalMapping,
        quality_score: 0.90,
    }
}

/// Helper function to create test verification results
pub fn create_test_verification_results() -> VerificationResults {
    VerificationResults {
        equivalence_verified: true,
        constraints_satisfied: true,
        semantic_correctness: true,
        verification_time: std::time::Duration::from_millis(50),
        verification_report: "All verifications passed".to_string(),
    }
}

/// Helper function to create test pass info
pub fn create_test_pass_info(name: &str) -> PassInfo {
    let mut metrics = std::collections::HashMap::new();
    metrics.insert("improvement".to_string(), 0.1);
    metrics.insert("efficiency".to_string(), 0.85);

    PassInfo {
        name: name.to_string(),
        execution_time: std::time::Duration::from_millis(100),
        gates_modified: 5,
        improvement: 0.1,
        metrics,
        success: true,
        error_message: None,
    }
}

/// Helper function to create test compilation result
pub fn create_test_compilation_result() -> CompilationResult {
    CompilationResult {
        original_circuit: "Original Circuit".to_string(),
        optimized_circuit: "Optimized Circuit".to_string(),
        optimization_stats: create_test_optimization_stats(),
        applied_passes: vec![
            create_test_pass_info("gate_synthesis"),
            create_test_pass_info("error_optimization"),
        ],
        hardware_allocation: create_test_hardware_allocation(),
        predicted_performance: create_test_performance_prediction(),
        compilation_time: std::time::Duration::from_millis(500),
        advanced_metrics: create_test_advanced_metrics(),
        optimization_history: vec![],
        platform_specific: PlatformSpecificResults {
            platform: "Test Platform".to_string(),
            metrics: std::collections::HashMap::new(),
            transformations: vec!["Test Transformation".to_string()],
        },
        verification_results: create_test_verification_results(),
    }
}

/// Helper trait for creating mock results
pub trait MockResults {
    /// Create mock optimization result
    fn create_mock_optimization_result() -> AdvancedOptimizationResult {
        AdvancedOptimizationResult {
            method: "Mock".to_string(),
            converged: true,
            objective_value: 0.95,
            iterations: 50,
            parameter_evolution: vec![ndarray::Array1::zeros(4)],
            success: true,
            x: ndarray::Array1::zeros(4),
            improvement: 0.15,
        }
    }

    /// Create mock linalg result
    fn create_mock_linalg_result() -> LinalgOptimizationResult {
        LinalgOptimizationResult {
            decomposition_improvements: std::collections::HashMap::new(),
            stability_metrics: NumericalStabilityMetrics {
                condition_number: 5.0,
                numerical_rank: 4,
                spectral_radius: 1.1,
            },
            eigenvalue_analysis: EigenvalueAnalysis {
                eigenvalue_distribution: vec![],
                spectral_gap: 0.2,
                entanglement_spectrum: vec![],
            },
        }
    }
}

/// Test data generator
pub struct TestDataGenerator;

impl TestDataGenerator {
    /// Generate test circuit metrics
    pub fn generate_circuit_metrics() -> std::collections::HashMap<String, f64> {
        let mut metrics = std::collections::HashMap::new();
        metrics.insert("gate_count".to_string(), 15.0);
        metrics.insert("depth".to_string(), 8.0);
        metrics.insert("fidelity".to_string(), 0.92);
        metrics.insert("error_rate".to_string(), 0.08);
        metrics
    }

    /// Generate test objective values
    pub fn generate_objective_values() -> Vec<f64> {
        vec![0.95, 0.88, 0.91, 0.87]
    }

    /// Generate test performance data
    pub fn generate_performance_data() -> Vec<f64> {
        vec![0.92, 0.89, 0.95, 0.87, 0.93, 0.91, 0.88, 0.94]
    }
}

/// Configuration builder for tests
pub struct TestConfigBuilder {
    config: CompilerConfig,
}

impl TestConfigBuilder {
    /// Create new test config builder
    pub fn new() -> Self {
        Self {
            config: create_test_compiler_config(),
        }
    }

    /// Enable specific optimization
    pub fn enable_optimization(mut self, optimization: &str, enabled: bool) -> Self {
        match optimization {
            "gate_synthesis" => self.config.enable_gate_synthesis = enabled,
            "error_optimization" => self.config.enable_error_optimization = enabled,
            "timing_optimization" => self.config.enable_timing_optimization = enabled,
            "crosstalk_mitigation" => self.config.enable_crosstalk_mitigation = enabled,
            "resource_optimization" => self.config.enable_resource_optimization = enabled,
            _ => {}
        }
        self
    }

    /// Set target platform
    pub fn with_target(mut self, target: CompilationTarget) -> Self {
        self.config.target = target;
        self
    }

    /// Set analysis depth
    pub fn with_analysis_depth(mut self, depth: AnalysisDepth) -> Self {
        self.config.analysis_depth = depth;
        self
    }

    /// Build the configuration
    pub fn build(self) -> CompilerConfig {
        self.config
    }
}

impl Default for TestConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
