extern crate proc_macro;

/// Quantum circuit representation and DSL for the QuantRS2 framework.
///
/// This crate provides types for constructing and manipulating
/// quantum circuits with a fluent API.
pub mod builder;
pub mod graph_optimizer;
pub mod optimizer;
pub mod optimization;
pub mod dag;
pub mod commutation;
pub mod qasm;
pub mod slicing;
pub mod topology;
pub mod equivalence;
pub mod classical;

// Re-exports of commonly used types and traits
pub mod prelude {
    pub use crate::builder::*;
    // Convenience re-export
    pub use quantrs2_core::qubit::QubitId as Qubit;
    pub use crate::graph_optimizer::{CircuitDAG, GraphGate, GraphOptimizer, OptimizationStats};
    pub use crate::optimizer::{
        CircuitOptimizer, HardwareOptimizer, OptimizationPassType, OptimizationResult,
        RedundantGateElimination, SingleQubitGateFusion,
    };
    pub use crate::optimization::{
        CircuitOptimizer2, PassManager, OptimizationLevel, PassConfig,
        GateProperties, GateCost, GateError, CommutationTable,
        OptimizationPass, GateCancellation, GateCommutation, GateMerging,
        DecompositionOptimization, CostBasedOptimization, RotationMerging,
        TwoQubitOptimization, TemplateMatching, CircuitRewriting,
        CostModel, HardwareCostModel, AbstractCostModel,
        CircuitAnalyzer, CircuitMetrics, OptimizationReport,
    };
    pub use crate::dag::{CircuitDag, DagNode, DagEdge, EdgeType, circuit_to_dag};
    pub use crate::commutation::{
        CommutationAnalyzer, CommutationRules, CommutationResult, GateType,
        CommutationOptimization,
    };
    pub use crate::qasm::{
        parse_qasm3, export_qasm3, QasmParser, QasmExporter, ExportOptions,
        QasmProgram, QasmStatement, QasmGate, QasmRegister,
        validate_qasm3, ParseError, ValidationError,
    };
    pub use crate::qasm::exporter::ExportError;
    pub use crate::slicing::{
        CircuitSlicer, CircuitSlice, SlicingStrategy, SlicingResult,
    };
    pub use crate::topology::{
        TopologicalAnalyzer, TopologicalAnalysis, TopologicalStrategy,
    };
    pub use crate::equivalence::{
        EquivalenceChecker, EquivalenceResult, EquivalenceType, EquivalenceOptions,
        circuits_equivalent, circuits_structurally_equal,
    };
    pub use crate::classical::{
        ClassicalCircuit, ClassicalCircuitBuilder, ClassicalRegister, ClassicalBit,
        ClassicalValue, ClassicalCondition, ComparisonOp, MeasureOp, ConditionalGate,
        ClassicalOp, CircuitOp,
    };
}

// The following should be proc macros, but we'll implement them later
// for now they're just stubs

/// Creates a qubit set for quantum operations
///
/// # Example
///
/// ```ignore
/// let qs = qubits![0, 1, 2];
/// ```
#[macro_export]
macro_rules! qubits {
    ($($id:expr),* $(,)?) => {
        {
            use quantrs2_core::qubit::QubitSet;

            let mut qs = QubitSet::new();
            $(qs.add($id);)*
            qs
        }
    };
}

/// Constructs a quantum circuit with a fixed number of qubits
///
/// # Example
///
/// ```ignore
/// let circuit = circuit![4; // 4 qubits
///     h(0),
///     cnot(0, 1),
///     h(2),
///     cnot(2, 3)
/// ];
/// ```
#[macro_export]
macro_rules! circuit {
    ($n:expr) => {
        quantrs2_circuit::builder::Circuit::<$n>::new()
    };
}

/// Provides a DSL for constructing quantum circuits
///
/// # Example
///
/// ```ignore
/// use quantrs2_circuit::quantum;
///
/// quantum! {
///     let qc = circuit(4);  // 4 qubits
///     qc.h(0);
///     qc.cnot(0, 1);
///     qc.measure_all();
/// }
/// ```
#[macro_export]
macro_rules! quantum {
    ($($tokens:tt)*) => {
        compile_error!("quantum! macro not fully implemented yet");
    };
}
