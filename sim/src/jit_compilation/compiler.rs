//! JIT Compiler implementation
//!
//! This module provides the main JIT compilation engine for quantum circuits.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use std::collections::HashMap;
use std::hash::Hasher;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::circuit_interfaces::{InterfaceGate, InterfaceGateType};
use crate::error::{Result, SimulatorError};

use super::analyzer::PatternAnalyzer;
use super::profiler::{JITCompilerStats, RuntimeProfiler};
use super::types::{
    BytecodeInstruction, CompilationStatus, CompiledFunction, CompiledGateSequence,
    FusedGateOperation, GateSequencePattern, JITConfig, JITOptimization, JITOptimizationLevel,
    JITPerformanceStats, MatrixComputeFunction, MatrixOpType, MatrixOperation, SIMDInstruction,
    SIMDLayout, VectorizedOperation,
};

/// JIT compilation engine
pub struct JITCompiler {
    /// Configuration
    pub(crate) config: JITConfig,
    /// Pattern database
    pub(crate) patterns: Arc<RwLock<HashMap<u64, GateSequencePattern>>>,
    /// Compiled sequence cache
    pub(crate) compiled_cache: Arc<RwLock<HashMap<u64, CompiledGateSequence>>>,
    /// Pattern analyzer
    pub(crate) pattern_analyzer: Arc<Mutex<PatternAnalyzer>>,
    /// Runtime profiler
    pub(crate) profiler: Arc<Mutex<RuntimeProfiler>>,
    /// Compilation statistics
    pub(crate) stats: Arc<RwLock<JITCompilerStats>>,
}

impl JITCompiler {
    /// Create a new JIT compiler
    #[must_use]
    pub fn new(config: JITConfig) -> Self {
        Self {
            config,
            patterns: Arc::new(RwLock::new(HashMap::new())),
            compiled_cache: Arc::new(RwLock::new(HashMap::new())),
            pattern_analyzer: Arc::new(Mutex::new(PatternAnalyzer::new())),
            profiler: Arc::new(Mutex::new(RuntimeProfiler::new())),
            stats: Arc::new(RwLock::new(JITCompilerStats::default())),
        }
    }

    /// Analyze gate sequence and potentially compile
    pub fn analyze_sequence(&self, gates: &[InterfaceGate]) -> Result<Option<u64>> {
        if gates.len() > self.config.max_sequence_length {
            return Ok(None);
        }

        // Update patterns_analyzed counter
        {
            let mut stats = self
                .stats
                .write()
                .expect("JIT stats lock should not be poisoned");
            stats.patterns_analyzed += 1;
        }

        let pattern = Self::extract_pattern(gates)?;
        let pattern_hash = pattern.hash;

        // Update pattern frequency
        {
            let mut patterns = self
                .patterns
                .write()
                .expect("JIT patterns lock should not be poisoned");
            if let Some(existing_pattern) = patterns.get_mut(&pattern_hash) {
                existing_pattern.frequency += 1;
                existing_pattern.last_used = Instant::now();
            } else {
                patterns.insert(pattern_hash, pattern);
            }
        }

        // Check if compilation threshold is met (compile after threshold is exceeded)
        let should_compile = {
            let patterns = self
                .patterns
                .read()
                .expect("JIT patterns lock should not be poisoned");
            if let Some(pattern) = patterns.get(&pattern_hash) {
                pattern.frequency > self.config.compilation_threshold
                    && pattern.compilation_status == CompilationStatus::NotCompiled
            } else {
                false
            }
        };

        if should_compile {
            self.compile_sequence(pattern_hash)?;
        }

        Ok(Some(pattern_hash))
    }

    /// Extract pattern from gate sequence
    pub fn extract_pattern(gates: &[InterfaceGate]) -> Result<GateSequencePattern> {
        let mut gate_types = Vec::new();
        let mut target_qubits = Vec::new();

        for gate in gates {
            gate_types.push(gate.gate_type.clone());
            target_qubits.push(gate.qubits.clone());
        }

        let mut pattern = GateSequencePattern {
            gate_types,
            target_qubits,
            hash: 0,
            frequency: 1,
            last_used: Instant::now(),
            compilation_status: CompilationStatus::NotCompiled,
        };

        // Calculate hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;
        let mut hasher = DefaultHasher::new();
        pattern.hash(&mut hasher);
        pattern.hash = hasher.finish();

        Ok(pattern)
    }

    /// Compile a gate sequence pattern
    fn compile_sequence(&self, pattern_hash: u64) -> Result<()> {
        // Mark as compiling
        {
            let mut patterns = self
                .patterns
                .write()
                .expect("JIT patterns lock should not be poisoned");
            if let Some(pattern) = patterns.get_mut(&pattern_hash) {
                pattern.compilation_status = CompilationStatus::Compiling;
            }
        }

        let compilation_start = Instant::now();

        // Get pattern for compilation
        let pattern = {
            let patterns = self
                .patterns
                .read()
                .expect("JIT patterns lock should not be poisoned");
            patterns
                .get(&pattern_hash)
                .cloned()
                .ok_or_else(|| SimulatorError::InvalidParameter("Pattern not found".to_string()))?
        };

        // Perform compilation
        let compiled_function = self.perform_compilation(&pattern)?;
        let compilation_time = compilation_start.elapsed();

        // Create compiled sequence
        let compiled_sequence = CompiledGateSequence {
            pattern: pattern.clone(),
            compiled_function,
            compilation_time,
            performance_stats: JITPerformanceStats::default(),
            memory_usage: Self::estimate_memory_usage(&pattern),
            optimizations: self.apply_optimizations(&pattern)?,
        };

        // Store compiled sequence
        {
            let mut cache = self
                .compiled_cache
                .write()
                .expect("JIT cache lock should not be poisoned");
            cache.insert(pattern_hash, compiled_sequence);
        }

        // Update pattern status
        {
            let mut patterns = self
                .patterns
                .write()
                .expect("JIT patterns lock should not be poisoned");
            if let Some(pattern) = patterns.get_mut(&pattern_hash) {
                pattern.compilation_status = CompilationStatus::Compiled;
            }
        }

        // Update statistics
        {
            let mut stats = self
                .stats
                .write()
                .expect("JIT stats lock should not be poisoned");
            stats.total_compilations += 1;
            stats.total_compilation_time += compilation_time;
        }

        Ok(())
    }

    /// Perform the actual compilation
    fn perform_compilation(&self, pattern: &GateSequencePattern) -> Result<CompiledFunction> {
        match self.config.optimization_level {
            JITOptimizationLevel::None => Self::compile_basic(pattern),
            JITOptimizationLevel::Basic => self.compile_with_basic_optimizations(pattern),
            JITOptimizationLevel::Advanced => self.compile_with_advanced_optimizations(pattern),
            JITOptimizationLevel::Aggressive => self.compile_with_aggressive_optimizations(pattern),
        }
    }

    /// Basic compilation (bytecode generation)
    fn compile_basic(pattern: &GateSequencePattern) -> Result<CompiledFunction> {
        let mut instructions = Vec::new();

        for (i, gate_type) in pattern.gate_types.iter().enumerate() {
            let targets = &pattern.target_qubits[i];

            let instruction = match targets.len() {
                1 => BytecodeInstruction::ApplySingleQubit {
                    gate_type: gate_type.clone(),
                    target: targets[0],
                },
                2 => BytecodeInstruction::ApplyTwoQubit {
                    gate_type: gate_type.clone(),
                    control: targets[0],
                    target: targets[1],
                },
                _ => BytecodeInstruction::ApplyMultiQubit {
                    gate_type: gate_type.clone(),
                    targets: targets.clone(),
                },
            };

            instructions.push(instruction);
        }

        Ok(CompiledFunction::Bytecode { instructions })
    }

    /// Compilation with basic optimizations
    fn compile_with_basic_optimizations(
        &self,
        pattern: &GateSequencePattern,
    ) -> Result<CompiledFunction> {
        let mut bytecode = Self::compile_basic(pattern)?;

        if let CompiledFunction::Bytecode { instructions } = &mut bytecode {
            // Apply constant folding
            Self::apply_constant_folding(instructions)?;

            // Apply dead code elimination
            Self::apply_dead_code_elimination(instructions)?;
        }

        Ok(bytecode)
    }

    /// Compilation with advanced optimizations
    fn compile_with_advanced_optimizations(
        &self,
        pattern: &GateSequencePattern,
    ) -> Result<CompiledFunction> {
        let mut bytecode = self.compile_with_basic_optimizations(pattern)?;

        if let CompiledFunction::Bytecode { instructions } = &mut bytecode {
            // Apply loop unrolling
            self.apply_loop_unrolling(instructions)?;

            // Apply vectorization
            return Self::apply_vectorization(instructions);
        }

        Ok(bytecode)
    }

    /// Compilation with aggressive optimizations
    fn compile_with_aggressive_optimizations(
        &self,
        pattern: &GateSequencePattern,
    ) -> Result<CompiledFunction> {
        // First try advanced optimizations
        let advanced_result = self.compile_with_advanced_optimizations(pattern)?;

        // Apply aggressive optimizations
        match advanced_result {
            CompiledFunction::Bytecode { instructions } => {
                // Try to convert to optimized matrix operations
                if let Ok(matrix_ops) = self.convert_to_matrix_operations(&instructions) {
                    return Ok(CompiledFunction::MatrixOps {
                        operations: matrix_ops,
                    });
                }

                // Apply gate fusion
                if let Ok(fused_ops) = self.apply_gate_fusion(&instructions) {
                    return Ok(CompiledFunction::Bytecode {
                        instructions: fused_ops,
                    });
                }

                Ok(CompiledFunction::Bytecode { instructions })
            }
            other => Ok(other),
        }
    }

    /// Apply constant folding optimization
    pub fn apply_constant_folding(instructions: &mut [BytecodeInstruction]) -> Result<()> {
        for instruction in instructions.iter_mut() {
            match instruction {
                BytecodeInstruction::ApplySingleQubit { gate_type, .. }
                | BytecodeInstruction::ApplyTwoQubit { gate_type, .. }
                | BytecodeInstruction::ApplyMultiQubit { gate_type, .. } => {
                    // Fold zero rotations to identity
                    match gate_type {
                        InterfaceGateType::RX(angle)
                        | InterfaceGateType::RY(angle)
                        | InterfaceGateType::RZ(angle)
                            if angle.abs() < f64::EPSILON =>
                        {
                            *gate_type = InterfaceGateType::Identity;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Apply dead code elimination
    pub fn apply_dead_code_elimination(instructions: &mut Vec<BytecodeInstruction>) -> Result<()> {
        instructions.retain(|instruction| {
            match instruction {
                BytecodeInstruction::ApplySingleQubit { gate_type, .. } => {
                    // Remove identity operations
                    !matches!(gate_type, InterfaceGateType::Identity)
                }
                _ => true,
            }
        });
        Ok(())
    }

    /// Apply loop unrolling optimization
    fn apply_loop_unrolling(&self, instructions: &mut Vec<BytecodeInstruction>) -> Result<()> {
        let mut unrolled = Vec::new();
        let mut i = 0;

        while i < instructions.len() {
            if let Some(repeat_count) = Self::find_repeated_sequence(&instructions[i..]) {
                for _ in 0..repeat_count {
                    unrolled.push(instructions[i].clone());
                }
                i += repeat_count;
            } else {
                unrolled.push(instructions[i].clone());
                i += 1;
            }
        }

        *instructions = unrolled;
        Ok(())
    }

    /// Find repeated instruction sequences
    fn find_repeated_sequence(instructions: &[BytecodeInstruction]) -> Option<usize> {
        if instructions.len() < 2 {
            return None;
        }

        if instructions.len() >= 2
            && std::mem::discriminant(&instructions[0]) == std::mem::discriminant(&instructions[1])
        {
            return Some(2);
        }

        None
    }

    /// Apply vectorization optimization
    fn apply_vectorization(instructions: &[BytecodeInstruction]) -> Result<CompiledFunction> {
        let mut vectorized_ops = Vec::new();

        for instruction in instructions {
            match instruction {
                BytecodeInstruction::ApplySingleQubit { gate_type, .. } => {
                    let simd_instruction = match gate_type {
                        InterfaceGateType::PauliX
                        | InterfaceGateType::X
                        | InterfaceGateType::PauliY
                        | InterfaceGateType::PauliZ => SIMDInstruction::GateApplication,
                        InterfaceGateType::RX(_)
                        | InterfaceGateType::RY(_)
                        | InterfaceGateType::RZ(_) => SIMDInstruction::Rotation,
                        _ => SIMDInstruction::GateApplication,
                    };

                    vectorized_ops.push(VectorizedOperation {
                        instruction: simd_instruction,
                        layout: SIMDLayout::StructureOfArrays,
                        vector_length: 8,
                        parallel_factor: 1,
                    });
                }
                _ => {
                    vectorized_ops.push(VectorizedOperation {
                        instruction: SIMDInstruction::GateApplication,
                        layout: SIMDLayout::Interleaved,
                        vector_length: 4,
                        parallel_factor: 1,
                    });
                }
            }
        }

        Ok(CompiledFunction::SIMDOps { vectorized_ops })
    }

    /// Convert bytecode to matrix operations
    fn convert_to_matrix_operations(
        &self,
        instructions: &[BytecodeInstruction],
    ) -> Result<Vec<MatrixOperation>> {
        let mut operations = Vec::new();

        for instruction in instructions {
            match instruction {
                BytecodeInstruction::ApplySingleQubit { gate_type, target } => {
                    let matrix = Self::get_gate_matrix(gate_type)?;
                    operations.push(MatrixOperation {
                        op_type: MatrixOpType::DirectMult,
                        targets: vec![*target],
                        matrix: Some(matrix),
                        compute_matrix: MatrixComputeFunction::Precomputed(Self::get_gate_matrix(
                            gate_type,
                        )?),
                    });
                }
                BytecodeInstruction::ApplyTwoQubit {
                    gate_type,
                    control,
                    target,
                } => {
                    let matrix = Self::get_two_qubit_gate_matrix(gate_type)?;
                    operations.push(MatrixOperation {
                        op_type: MatrixOpType::KroneckerProduct,
                        targets: vec![*control, *target],
                        matrix: Some(matrix),
                        compute_matrix: MatrixComputeFunction::Precomputed(
                            Self::get_two_qubit_gate_matrix(gate_type)?,
                        ),
                    });
                }
                _ => {
                    operations.push(MatrixOperation {
                        op_type: MatrixOpType::TensorContraction,
                        targets: vec![0],
                        matrix: None,
                        compute_matrix: MatrixComputeFunction::Runtime("default".to_string()),
                    });
                }
            }
        }

        Ok(operations)
    }

    /// Apply gate fusion optimization
    fn apply_gate_fusion(
        &self,
        instructions: &[BytecodeInstruction],
    ) -> Result<Vec<BytecodeInstruction>> {
        let mut fused_instructions = Vec::new();
        let mut i = 0;

        while i < instructions.len() {
            if let Some(fused_length) = Self::find_fusable_sequence(&instructions[i..]) {
                let gates =
                    Self::extract_gates_from_instructions(&instructions[i..i + fused_length])?;
                let fused_matrix = Self::compute_fused_matrix(&gates)?;
                let targets =
                    Self::extract_targets_from_instructions(&instructions[i..i + fused_length]);

                let fused_op = FusedGateOperation {
                    gates,
                    fused_matrix,
                    targets,
                    optimization_level: self.config.optimization_level,
                };

                fused_instructions.push(BytecodeInstruction::FusedOperation {
                    operation: fused_op,
                });

                i += fused_length;
            } else {
                fused_instructions.push(instructions[i].clone());
                i += 1;
            }
        }

        Ok(fused_instructions)
    }

    /// Find fusable gate sequences
    fn find_fusable_sequence(instructions: &[BytecodeInstruction]) -> Option<usize> {
        if instructions.len() < 2 {
            return None;
        }

        if let (
            BytecodeInstruction::ApplySingleQubit {
                target: target1, ..
            },
            BytecodeInstruction::ApplySingleQubit {
                target: target2, ..
            },
        ) = (&instructions[0], &instructions[1])
        {
            if target1 == target2 {
                return Some(2);
            }
        }

        None
    }

    /// Extract gates from bytecode instructions
    fn extract_gates_from_instructions(
        instructions: &[BytecodeInstruction],
    ) -> Result<Vec<InterfaceGate>> {
        let mut gates = Vec::new();

        for instruction in instructions {
            match instruction {
                BytecodeInstruction::ApplySingleQubit { gate_type, target } => {
                    gates.push(InterfaceGate::new(gate_type.clone(), vec![*target]));
                }
                BytecodeInstruction::ApplyTwoQubit {
                    gate_type,
                    control,
                    target,
                } => {
                    gates.push(InterfaceGate::new(
                        gate_type.clone(),
                        vec![*control, *target],
                    ));
                }
                BytecodeInstruction::ApplyMultiQubit { gate_type, targets } => {
                    gates.push(InterfaceGate::new(gate_type.clone(), targets.clone()));
                }
                _ => {
                    return Err(SimulatorError::NotImplemented(
                        "Complex gate extraction".to_string(),
                    ));
                }
            }
        }

        Ok(gates)
    }

    /// Extract target qubits from instructions
    fn extract_targets_from_instructions(instructions: &[BytecodeInstruction]) -> Vec<usize> {
        let mut targets = std::collections::HashSet::new();

        for instruction in instructions {
            match instruction {
                BytecodeInstruction::ApplySingleQubit { target, .. } => {
                    targets.insert(*target);
                }
                BytecodeInstruction::ApplyTwoQubit {
                    control, target, ..
                } => {
                    targets.insert(*control);
                    targets.insert(*target);
                }
                BytecodeInstruction::ApplyMultiQubit {
                    targets: multi_targets,
                    ..
                } => {
                    for &target in multi_targets {
                        targets.insert(target);
                    }
                }
                _ => {}
            }
        }

        targets.into_iter().collect()
    }

    /// Compute fused matrix for gate sequence
    fn compute_fused_matrix(gates: &[InterfaceGate]) -> Result<Array2<Complex64>> {
        if gates.is_empty() {
            return Err(SimulatorError::InvalidParameter(
                "Empty gate sequence".to_string(),
            ));
        }

        let mut result = Self::get_gate_matrix(&gates[0].gate_type)?;

        for gate in &gates[1..] {
            let gate_matrix = Self::get_gate_matrix(&gate.gate_type)?;
            result = result.dot(&gate_matrix);
        }

        Ok(result)
    }

    /// Get matrix representation of a gate
    pub fn get_gate_matrix(gate_type: &InterfaceGateType) -> Result<Array2<Complex64>> {
        let matrix = match gate_type {
            InterfaceGateType::Identity => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .expect("2x2 matrix shape should be valid"),
            InterfaceGateType::PauliX | InterfaceGateType::X => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .expect("2x2 matrix shape should be valid"),
            InterfaceGateType::PauliY => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -1.0),
                    Complex64::new(0.0, 1.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .expect("2x2 matrix shape should be valid"),
            InterfaceGateType::PauliZ => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )
            .expect("2x2 matrix shape should be valid"),
            InterfaceGateType::Hadamard | InterfaceGateType::H => {
                let sqrt2_inv = 1.0 / (2.0_f64).sqrt();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(sqrt2_inv, 0.0),
                        Complex64::new(sqrt2_inv, 0.0),
                        Complex64::new(sqrt2_inv, 0.0),
                        Complex64::new(-sqrt2_inv, 0.0),
                    ],
                )
                .expect("2x2 matrix shape should be valid")
            }
            InterfaceGateType::S => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 1.0),
                ],
            )
            .expect("2x2 matrix shape should be valid"),
            InterfaceGateType::T => {
                let phase = Complex64::new(0.0, std::f64::consts::PI / 4.0).exp();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        phase,
                    ],
                )
                .expect("2x2 matrix shape should be valid")
            }
            InterfaceGateType::RX(angle) => {
                let cos_half = (angle / 2.0).cos();
                let sin_half = (angle / 2.0).sin();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(cos_half, 0.0),
                        Complex64::new(0.0, -sin_half),
                        Complex64::new(0.0, -sin_half),
                        Complex64::new(cos_half, 0.0),
                    ],
                )
                .expect("2x2 matrix shape should be valid")
            }
            InterfaceGateType::RY(angle) => {
                let cos_half = (angle / 2.0).cos();
                let sin_half = (angle / 2.0).sin();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(cos_half, 0.0),
                        Complex64::new(-sin_half, 0.0),
                        Complex64::new(sin_half, 0.0),
                        Complex64::new(cos_half, 0.0),
                    ],
                )
                .expect("2x2 matrix shape should be valid")
            }
            InterfaceGateType::RZ(angle) => {
                let exp_neg = Complex64::new(0.0, -angle / 2.0).exp();
                let exp_pos = Complex64::new(0.0, angle / 2.0).exp();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        exp_neg,
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        exp_pos,
                    ],
                )
                .expect("2x2 matrix shape should be valid")
            }
            InterfaceGateType::Phase(angle) => {
                let phase = Complex64::new(0.0, *angle).exp();
                Array2::from_shape_vec(
                    (2, 2),
                    vec![
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(0.0, 0.0),
                        phase,
                    ],
                )
                .expect("2x2 matrix shape should be valid")
            }
            _ => Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .expect("2x2 matrix shape should be valid"),
        };

        Ok(matrix)
    }

    /// Get matrix representation of a two-qubit gate
    pub fn get_two_qubit_gate_matrix(gate_type: &InterfaceGateType) -> Result<Array2<Complex64>> {
        let matrix = match gate_type {
            InterfaceGateType::CNOT => Array2::from_shape_vec(
                (4, 4),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .expect("4x4 matrix shape should be valid"),
            InterfaceGateType::CZ => Array2::from_shape_vec(
                (4, 4),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(-1.0, 0.0),
                ],
            )
            .expect("4x4 matrix shape should be valid"),
            InterfaceGateType::SWAP => Array2::from_shape_vec(
                (4, 4),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .expect("4x4 matrix shape should be valid"),
            _ => Array2::from_shape_vec(
                (4, 4),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .expect("4x4 matrix shape should be valid"),
        };

        Ok(matrix)
    }

    /// Estimate memory usage for a pattern
    fn estimate_memory_usage(pattern: &GateSequencePattern) -> usize {
        let base_size = std::mem::size_of::<CompiledGateSequence>();
        let pattern_size = pattern.gate_types.len() * 64;
        let matrix_size = pattern.gate_types.len() * 32 * std::mem::size_of::<Complex64>();

        base_size + pattern_size + matrix_size
    }

    /// Apply optimizations to a pattern
    fn apply_optimizations(&self, _pattern: &GateSequencePattern) -> Result<Vec<JITOptimization>> {
        let mut optimizations = vec![
            JITOptimization::ConstantFolding,
            JITOptimization::DeadCodeElimination,
        ];

        match self.config.optimization_level {
            JITOptimizationLevel::Basic => {}
            JITOptimizationLevel::Advanced => {
                optimizations.extend_from_slice(&[
                    JITOptimization::LoopUnrolling,
                    JITOptimization::Vectorization,
                ]);
            }
            JITOptimizationLevel::Aggressive => {
                optimizations.extend_from_slice(&[
                    JITOptimization::LoopUnrolling,
                    JITOptimization::Vectorization,
                    JITOptimization::GateFusion,
                    JITOptimization::InlineExpansion,
                    JITOptimization::MemoryLayoutOptimization,
                ]);
            }
            JITOptimizationLevel::None => {
                optimizations.clear();
            }
        }

        Ok(optimizations)
    }

    /// Execute a compiled sequence
    pub fn execute_compiled(
        &self,
        pattern_hash: u64,
        state: &mut Array1<Complex64>,
    ) -> Result<Duration> {
        let execution_start = Instant::now();

        let compiled_sequence = {
            let cache = self
                .compiled_cache
                .read()
                .expect("JIT cache lock should not be poisoned");
            cache.get(&pattern_hash).cloned().ok_or_else(|| {
                SimulatorError::InvalidParameter("Compiled sequence not found".to_string())
            })?
        };

        match &compiled_sequence.compiled_function {
            CompiledFunction::Bytecode { instructions } => {
                self.execute_bytecode(instructions, state)?;
            }
            CompiledFunction::MatrixOps { operations } => {
                self.execute_matrix_operations(operations, state)?;
            }
            CompiledFunction::SIMDOps { vectorized_ops } => {
                self.execute_simd_operations(vectorized_ops, state)?;
            }
            CompiledFunction::NativeCode { .. } => {
                return Err(SimulatorError::NotImplemented(
                    "Native code execution".to_string(),
                ));
            }
        }

        let execution_time = execution_start.elapsed();

        {
            let mut cache = self
                .compiled_cache
                .write()
                .expect("JIT cache lock should not be poisoned");
            if let Some(sequence) = cache.get_mut(&pattern_hash) {
                let stats = &mut sequence.performance_stats;
                stats.execution_count += 1;
                stats.total_execution_time += execution_time;
                stats.average_execution_time =
                    stats.total_execution_time / stats.execution_count as u32;
                if execution_time < stats.best_execution_time {
                    stats.best_execution_time = execution_time;
                }
            }
        }

        Ok(execution_time)
    }

    /// Execute bytecode instructions
    fn execute_bytecode(
        &self,
        instructions: &[BytecodeInstruction],
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        for instruction in instructions {
            match instruction {
                BytecodeInstruction::ApplySingleQubit { gate_type, target } => {
                    Self::apply_single_qubit_gate(gate_type, *target, state)?;
                }
                BytecodeInstruction::ApplyTwoQubit {
                    gate_type,
                    control,
                    target,
                } => {
                    Self::apply_two_qubit_gate(gate_type, *control, *target, state)?;
                }
                BytecodeInstruction::ApplyMultiQubit { gate_type, targets } => {
                    Self::apply_multi_qubit_gate(gate_type, targets, state)?;
                }
                BytecodeInstruction::FusedOperation { operation } => {
                    Self::apply_fused_operation(operation, state)?;
                }
                BytecodeInstruction::Prefetch { .. } => {}
                BytecodeInstruction::Barrier => {
                    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                }
            }
        }
        Ok(())
    }

    /// Apply single-qubit gate to state
    fn apply_single_qubit_gate(
        gate_type: &InterfaceGateType,
        target: usize,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let num_qubits = (state.len() as f64).log2() as usize;
        if target >= num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Target qubit out of range".to_string(),
            ));
        }

        let matrix = Self::get_gate_matrix(gate_type)?;

        for i in 0..(1 << num_qubits) {
            if (i >> target) & 1 == 0 {
                let j = i | (1 << target);
                let amp0 = state[i];
                let amp1 = state[j];

                state[i] = matrix[(0, 0)] * amp0 + matrix[(0, 1)] * amp1;
                state[j] = matrix[(1, 0)] * amp0 + matrix[(1, 1)] * amp1;
            }
        }

        Ok(())
    }

    /// Apply two-qubit gate to state
    fn apply_two_qubit_gate(
        gate_type: &InterfaceGateType,
        control: usize,
        target: usize,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let num_qubits = (state.len() as f64).log2() as usize;
        if control >= num_qubits || target >= num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Qubit index out of range".to_string(),
            ));
        }

        match gate_type {
            InterfaceGateType::CNOT => {
                for i in 0..(1 << num_qubits) {
                    if (i >> control) & 1 == 1 {
                        let j = i ^ (1 << target);
                        if i < j {
                            let temp = state[i];
                            state[i] = state[j];
                            state[j] = temp;
                        }
                    }
                }
            }
            InterfaceGateType::CZ => {
                for i in 0..(1 << num_qubits) {
                    if (i >> control) & 1 == 1 && (i >> target) & 1 == 1 {
                        state[i] = -state[i];
                    }
                }
            }
            InterfaceGateType::SWAP => {
                for i in 0..(1 << num_qubits) {
                    let bit_control = (i >> control) & 1;
                    let bit_target = (i >> target) & 1;
                    if bit_control != bit_target {
                        let j = i ^ (1 << control) ^ (1 << target);
                        if i < j {
                            let temp = state[i];
                            state[i] = state[j];
                            state[j] = temp;
                        }
                    }
                }
            }
            _ => {
                let matrix = Self::get_two_qubit_gate_matrix(gate_type)?;
                Self::apply_two_qubit_matrix(&matrix, control, target, state)?;
            }
        }

        Ok(())
    }

    /// Apply multi-qubit gate to state
    fn apply_multi_qubit_gate(
        _gate_type: &InterfaceGateType,
        _targets: &[usize],
        _state: &mut Array1<Complex64>,
    ) -> Result<()> {
        Err(SimulatorError::NotImplemented(
            "Multi-qubit gate execution".to_string(),
        ))
    }

    /// Apply fused operation to state
    fn apply_fused_operation(
        operation: &FusedGateOperation,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        if operation.targets.len() == 1 {
            let target = operation.targets[0];
            let num_qubits = (state.len() as f64).log2() as usize;

            for i in 0..(1 << num_qubits) {
                if (i >> target) & 1 == 0 {
                    let j = i | (1 << target);
                    let amp0 = state[i];
                    let amp1 = state[j];

                    state[i] = operation.fused_matrix[(0, 0)] * amp0
                        + operation.fused_matrix[(0, 1)] * amp1;
                    state[j] = operation.fused_matrix[(1, 0)] * amp0
                        + operation.fused_matrix[(1, 1)] * amp1;
                }
            }
        }

        Ok(())
    }

    /// Execute matrix operations
    fn execute_matrix_operations(
        &self,
        operations: &[MatrixOperation],
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        for operation in operations {
            match &operation.op_type {
                MatrixOpType::DirectMult => {
                    if let Some(matrix) = &operation.matrix {
                        for &target in &operation.targets {
                            Self::apply_matrix_to_target(matrix, target, state)?;
                        }
                    }
                }
                MatrixOpType::KroneckerProduct => {
                    if operation.targets.len() == 2 {
                        if let Some(matrix) = operation.matrix.as_ref() {
                            let control = operation.targets[0];
                            let target = operation.targets[1];
                            Self::apply_two_qubit_matrix(matrix, control, target, state)?;
                        }
                    }
                }
                _ => {
                    return Err(SimulatorError::NotImplemented(
                        "Matrix operation type".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Apply matrix to specific target qubit
    fn apply_matrix_to_target(
        matrix: &Array2<Complex64>,
        target: usize,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let num_qubits = (state.len() as f64).log2() as usize;
        if target >= num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Target qubit out of range".to_string(),
            ));
        }

        for i in 0..(1 << num_qubits) {
            if (i >> target) & 1 == 0 {
                let j = i | (1 << target);
                let amp0 = state[i];
                let amp1 = state[j];

                state[i] = matrix[(0, 0)] * amp0 + matrix[(0, 1)] * amp1;
                state[j] = matrix[(1, 0)] * amp0 + matrix[(1, 1)] * amp1;
            }
        }

        Ok(())
    }

    /// Apply two-qubit matrix
    fn apply_two_qubit_matrix(
        matrix: &Array2<Complex64>,
        control: usize,
        target: usize,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let num_qubits = (state.len() as f64).log2() as usize;
        if control >= num_qubits || target >= num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Qubit index out of range".to_string(),
            ));
        }

        for i in 0..(1 << num_qubits) {
            let control_bit = (i >> control) & 1;
            let target_bit = (i >> target) & 1;
            let basis_state = control_bit * 2 + target_bit;

            if basis_state == 0 {
                let i00 = i;
                let i01 = i ^ (1 << target);
                let i10 = i ^ (1 << control);
                let i11 = i ^ (1 << control) ^ (1 << target);

                let amp00 = state[i00];
                let amp01 = state[i01];
                let amp10 = state[i10];
                let amp11 = state[i11];

                state[i00] = matrix[(0, 0)] * amp00
                    + matrix[(0, 1)] * amp01
                    + matrix[(0, 2)] * amp10
                    + matrix[(0, 3)] * amp11;
                state[i01] = matrix[(1, 0)] * amp00
                    + matrix[(1, 1)] * amp01
                    + matrix[(1, 2)] * amp10
                    + matrix[(1, 3)] * amp11;
                state[i10] = matrix[(2, 0)] * amp00
                    + matrix[(2, 1)] * amp01
                    + matrix[(2, 2)] * amp10
                    + matrix[(2, 3)] * amp11;
                state[i11] = matrix[(3, 0)] * amp00
                    + matrix[(3, 1)] * amp01
                    + matrix[(3, 2)] * amp10
                    + matrix[(3, 3)] * amp11;
            }
        }

        Ok(())
    }

    /// Execute SIMD operations
    fn execute_simd_operations(
        &self,
        operations: &[VectorizedOperation],
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        for operation in operations {
            match operation.instruction {
                SIMDInstruction::ComplexMul => {
                    Self::execute_simd_complex_mul(operation, state)?;
                }
                SIMDInstruction::ComplexAdd => {
                    Self::execute_simd_complex_add(operation, state)?;
                }
                SIMDInstruction::Rotation => {
                    Self::execute_simd_rotation(operation, state)?;
                }
                SIMDInstruction::GateApplication => {
                    Self::execute_simd_gate_application(operation, state)?;
                }
                SIMDInstruction::TensorProduct => {
                    return Err(SimulatorError::NotImplemented(
                        "SIMD instruction".to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn execute_simd_complex_mul(
        _operation: &VectorizedOperation,
        _state: &mut Array1<Complex64>,
    ) -> Result<()> {
        Ok(())
    }

    fn execute_simd_complex_add(
        _operation: &VectorizedOperation,
        _state: &mut Array1<Complex64>,
    ) -> Result<()> {
        Ok(())
    }

    fn execute_simd_rotation(
        _operation: &VectorizedOperation,
        _state: &mut Array1<Complex64>,
    ) -> Result<()> {
        Ok(())
    }

    fn execute_simd_gate_application(
        _operation: &VectorizedOperation,
        _state: &mut Array1<Complex64>,
    ) -> Result<()> {
        Ok(())
    }

    /// Get compilation statistics
    #[must_use]
    pub fn get_stats(&self) -> JITCompilerStats {
        self.stats
            .read()
            .expect("JIT stats lock should not be poisoned")
            .clone()
    }

    /// Clear compiled cache
    pub fn clear_cache(&self) {
        let mut cache = self
            .compiled_cache
            .write()
            .expect("JIT cache lock should not be poisoned");
        cache.clear();

        let mut stats = self
            .stats
            .write()
            .expect("JIT stats lock should not be poisoned");
        stats.cache_clears += 1;
    }
}
