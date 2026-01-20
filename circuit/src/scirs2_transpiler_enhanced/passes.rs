//! Transpilation passes and strategies

use serde::{Deserialize, Serialize};

/// Performance constraints for transpilation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConstraints {
    /// Maximum circuit depth
    pub max_depth: Option<usize>,

    /// Maximum gate count
    pub max_gates: Option<usize>,

    /// Maximum execution time (seconds)
    pub max_execution_time: Option<f64>,

    /// Minimum fidelity requirement
    pub min_fidelity: Option<f64>,

    /// Maximum transpilation time (seconds)
    pub max_transpilation_time: Option<f64>,
}

impl Default for PerformanceConstraints {
    fn default() -> Self {
        Self {
            max_depth: None,
            max_gates: None,
            max_execution_time: None,
            min_fidelity: Some(0.95),
            max_transpilation_time: Some(60.0),
        }
    }
}

/// Export formats for transpiled circuits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExportFormat {
    QASM3,
    OpenQASM,
    Cirq,
    Qiskit,
    PyQuil,
    Braket,
    QSharp,
    Custom,
}

/// Transpilation pass types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TranspilationPass {
    /// Decompose gates to native gate set
    Decomposition(DecompositionStrategy),

    /// Route qubits based on connectivity
    Routing(RoutingStrategy),

    /// Optimize gate sequences
    Optimization(OptimizationStrategy),

    /// Apply error mitigation
    ErrorMitigation(MitigationStrategy),

    /// Custom pass with function pointer
    Custom(String),
}

/// Decomposition strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// Use KAK decomposition
    KAK,
    /// Use Euler decomposition
    Euler,
    /// Use optimal decomposition
    Optimal,
    /// Hardware-specific decomposition
    HardwareOptimized,
}

/// Routing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// SABRE routing algorithm
    SABRE,
    /// Stochastic routing
    Stochastic,
    /// Look-ahead routing
    LookAhead,
    /// ML-based routing
    MachineLearning,
    /// Hybrid approach
    Hybrid,
}

/// Optimization strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Gate cancellation
    GateCancellation,
    /// Gate fusion
    GateFusion,
    /// Commutation analysis
    Commutation,
    /// Template matching
    TemplateMatching,
    /// Peephole optimization
    Peephole,
    /// All optimizations
    All,
}

/// Error mitigation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MitigationStrategy {
    /// Zero noise extrapolation
    ZNE,
    /// Probabilistic error cancellation
    PEC,
    /// Symmetry verification
    SymmetryVerification,
    /// Virtual distillation
    VirtualDistillation,
    /// Dynamical decoupling
    DynamicalDecoupling,
}
