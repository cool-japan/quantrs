//! Type definitions for JIT compilation
//!
//! This module provides types and enums for the JIT compilation system.

use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use crate::circuit_interfaces::{InterfaceGate, InterfaceGateType};

/// JIT compilation configuration
#[derive(Debug, Clone)]
pub struct JITConfig {
    /// Minimum frequency threshold for compilation
    pub compilation_threshold: usize,
    /// Maximum number of gates in a compilable sequence
    pub max_sequence_length: usize,
    /// Enable pattern analysis and optimization
    pub enable_pattern_analysis: bool,
    /// Enable adaptive compilation thresholds
    pub enable_adaptive_thresholds: bool,
    /// Maximum cache size for compiled sequences
    pub max_cache_size: usize,
    /// Enable runtime profiling for optimization
    pub enable_runtime_profiling: bool,
    /// JIT compilation optimization level
    pub optimization_level: JITOptimizationLevel,
    /// Enable parallel compilation
    pub enable_parallel_compilation: bool,
}

impl Default for JITConfig {
    fn default() -> Self {
        Self {
            compilation_threshold: 10,
            max_sequence_length: 20,
            enable_pattern_analysis: true,
            enable_adaptive_thresholds: true,
            max_cache_size: 1000,
            enable_runtime_profiling: true,
            optimization_level: JITOptimizationLevel::Aggressive,
            enable_parallel_compilation: true,
        }
    }
}

/// JIT optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JITOptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations (constant folding, dead code elimination)
    Basic,
    /// Advanced optimizations (loop unrolling, vectorization)
    Advanced,
    /// Aggressive optimizations (inline expansion, specialized paths)
    Aggressive,
}

/// Gate sequence pattern for compilation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateSequencePattern {
    /// Gate types in the sequence
    pub gate_types: Vec<InterfaceGateType>,
    /// Target qubits for each gate
    pub target_qubits: Vec<Vec<usize>>,
    /// Sequence hash for fast lookup
    pub hash: u64,
    /// Usage frequency
    pub frequency: usize,
    /// Last used timestamp
    pub last_used: Instant,
    /// Compilation status
    pub compilation_status: CompilationStatus,
}

impl Hash for GateSequencePattern {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.gate_types.hash(state);
        self.target_qubits.hash(state);
    }
}

/// Compilation status for gate sequences
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationStatus {
    /// Not yet compiled
    NotCompiled,
    /// Currently being compiled
    Compiling,
    /// Successfully compiled
    Compiled,
    /// Compilation failed
    Failed,
    /// Compiled and optimized
    Optimized,
}

/// Compiled gate sequence
#[derive(Debug, Clone)]
pub struct CompiledGateSequence {
    /// Original pattern
    pub pattern: GateSequencePattern,
    /// Compiled function pointer (simulation only)
    pub compiled_function: CompiledFunction,
    /// Compilation time
    pub compilation_time: Duration,
    /// Runtime performance statistics
    pub performance_stats: JITPerformanceStats,
    /// Memory usage
    pub memory_usage: usize,
    /// Optimization flags applied
    pub optimizations: Vec<JITOptimization>,
}

/// Compiled function representation
#[derive(Debug, Clone)]
pub enum CompiledFunction {
    /// Native machine code (placeholder for actual implementation)
    NativeCode {
        code_size: usize,
        entry_point: usize,
    },
    /// Optimized interpreter bytecode
    Bytecode {
        instructions: Vec<BytecodeInstruction>,
    },
    /// Specialized matrix operations
    MatrixOps { operations: Vec<MatrixOperation> },
    /// SIMD-optimized operations
    SIMDOps {
        vectorized_ops: Vec<VectorizedOperation>,
    },
}

/// JIT bytecode instructions
#[derive(Debug, Clone)]
pub enum BytecodeInstruction {
    /// Apply single-qubit gate
    ApplySingleQubit {
        gate_type: InterfaceGateType,
        target: usize,
    },
    /// Apply two-qubit gate
    ApplyTwoQubit {
        gate_type: InterfaceGateType,
        control: usize,
        target: usize,
    },
    /// Apply multi-qubit gate
    ApplyMultiQubit {
        gate_type: InterfaceGateType,
        targets: Vec<usize>,
    },
    /// Fused operation
    FusedOperation { operation: FusedGateOperation },
    /// Memory prefetch hint
    Prefetch { address_pattern: PrefetchPattern },
    /// Barrier/synchronization
    Barrier,
}

/// Matrix operation for compilation
#[derive(Debug, Clone)]
pub struct MatrixOperation {
    /// Operation type
    pub op_type: MatrixOpType,
    /// Target qubits
    pub targets: Vec<usize>,
    /// Matrix elements (if small enough to inline)
    pub matrix: Option<Array2<Complex64>>,
    /// Matrix computation function
    pub compute_matrix: MatrixComputeFunction,
}

/// Matrix operation types
#[derive(Debug, Clone)]
pub enum MatrixOpType {
    /// Direct matrix multiplication
    DirectMult,
    /// Kronecker product
    KroneckerProduct,
    /// Tensor contraction
    TensorContraction,
    /// Sparse matrix operation
    SparseOperation,
}

/// Matrix computation function
#[derive(Debug, Clone)]
pub enum MatrixComputeFunction {
    /// Precomputed matrix
    Precomputed(Array2<Complex64>),
    /// Runtime computation
    Runtime(String), // Function identifier
    /// Parameterized computation
    Parameterized {
        template: Array2<Complex64>,
        param_indices: Vec<usize>,
    },
}

/// Vectorized operation for SIMD
#[derive(Debug, Clone)]
pub struct VectorizedOperation {
    /// SIMD instruction type
    pub instruction: SIMDInstruction,
    /// Data layout requirements
    pub layout: SIMDLayout,
    /// Vector length
    pub vector_length: usize,
    /// Parallelization factor
    pub parallel_factor: usize,
}

/// SIMD instruction types
#[derive(Debug, Clone)]
pub enum SIMDInstruction {
    /// Vectorized complex multiplication
    ComplexMul,
    /// Vectorized complex addition
    ComplexAdd,
    /// Vectorized rotation
    Rotation,
    /// Vectorized tensor product
    TensorProduct,
    /// Vectorized gate application
    GateApplication,
}

/// SIMD data layout
#[derive(Debug, Clone)]
pub enum SIMDLayout {
    /// Array of structures (`AoS`)
    ArrayOfStructures,
    /// Structure of arrays (`SoA`)
    StructureOfArrays,
    /// Interleaved real/imaginary
    Interleaved,
    /// Separate real/imaginary arrays
    Separate,
}

/// Fused gate operation
#[derive(Debug, Clone)]
pub struct FusedGateOperation {
    /// Component gates
    pub gates: Vec<InterfaceGate>,
    /// Fused matrix
    pub fused_matrix: Array2<Complex64>,
    /// Target qubits for the fused operation
    pub targets: Vec<usize>,
    /// Optimization level applied
    pub optimization_level: JITOptimizationLevel,
}

/// Prefetch pattern for memory optimization
#[derive(Debug, Clone)]
pub enum PrefetchPattern {
    /// Sequential access
    Sequential { stride: usize },
    /// Strided access
    Strided { stride: usize, count: usize },
    /// Sparse access
    Sparse { indices: Vec<usize> },
    /// Block access
    Block { base: usize, size: usize },
}

/// JIT performance statistics
#[derive(Debug, Clone)]
pub struct JITPerformanceStats {
    /// Execution count
    pub execution_count: usize,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Average execution time
    pub average_execution_time: Duration,
    /// Best execution time
    pub best_execution_time: Duration,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// Memory bandwidth utilization
    pub memory_bandwidth: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Performance improvement over interpreted
    pub speedup_factor: f64,
}

impl Default for JITPerformanceStats {
    fn default() -> Self {
        Self {
            execution_count: 0,
            total_execution_time: Duration::new(0, 0),
            average_execution_time: Duration::new(0, 0),
            best_execution_time: Duration::from_secs(u64::MAX),
            cache_hit_ratio: 0.0,
            memory_bandwidth: 0.0,
            cpu_utilization: 0.0,
            speedup_factor: 1.0,
        }
    }
}

/// JIT optimization techniques applied
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JITOptimization {
    /// Constant folding
    ConstantFolding,
    /// Dead code elimination
    DeadCodeElimination,
    /// Loop unrolling
    LoopUnrolling,
    /// Vectorization
    Vectorization,
    /// Inline expansion
    InlineExpansion,
    /// Gate fusion
    GateFusion,
    /// Memory layout optimization
    MemoryLayoutOptimization,
    /// Instruction scheduling
    InstructionScheduling,
    /// Register allocation
    RegisterAllocation,
    /// Strength reduction
    StrengthReduction,
}

/// Optimization suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationSuggestion {
    /// Gate fusion optimization
    GateFusion,
    /// Vectorization optimization
    Vectorization,
    /// Constant folding optimization
    ConstantFolding,
    /// Loop unrolling optimization
    LoopUnrolling,
    /// Memory layout optimization
    MemoryLayoutOptimization,
    /// Instruction scheduling optimization
    InstructionScheduling,
}

/// Compilation priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// JIT benchmark results
#[derive(Debug, Clone)]
pub struct JITBenchmarkResults {
    /// Total number of gate sequences tested
    pub total_sequences: usize,
    /// Number of sequences that were compiled
    pub compiled_sequences: usize,
    /// Number of sequences that were interpreted
    pub interpreted_sequences: usize,
    /// Average compilation time
    pub average_compilation_time: Duration,
    /// Average execution time for compiled sequences
    pub average_execution_time_compiled: Duration,
    /// Average execution time for interpreted sequences
    pub average_execution_time_interpreted: Duration,
    /// Speedup factor (interpreted / compiled)
    pub speedup_factor: f64,
    /// Compilation success rate
    pub compilation_success_rate: f64,
    /// Memory usage reduction
    pub memory_usage_reduction: f64,
}

impl fmt::Display for JITBenchmarkResults {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "JIT Compilation Benchmark Results:")?;
        writeln!(f, "  Total sequences: {}", self.total_sequences)?;
        writeln!(f, "  Compiled sequences: {}", self.compiled_sequences)?;
        writeln!(f, "  Interpreted sequences: {}", self.interpreted_sequences)?;
        writeln!(
            f,
            "  Average compilation time: {:?}",
            self.average_compilation_time
        )?;
        writeln!(
            f,
            "  Average execution time (compiled): {:?}",
            self.average_execution_time_compiled
        )?;
        writeln!(
            f,
            "  Average execution time (interpreted): {:?}",
            self.average_execution_time_interpreted
        )?;
        writeln!(f, "  Speedup factor: {:.2}x", self.speedup_factor)?;
        writeln!(
            f,
            "  Compilation success rate: {:.1}%",
            self.compilation_success_rate * 100.0
        )?;
        write!(
            f,
            "  Memory usage reduction: {:.1}%",
            self.memory_usage_reduction * 100.0
        )
    }
}
