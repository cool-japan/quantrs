//! `SciRS2` Intermediate Representation Tools
//!
//! This module provides the IR tools integration for SciRS2-enhanced cross-compilation,
//! implementing intermediate representation, optimization passes, and code generation
//! for multi-platform quantum computing deployment.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use uuid::Uuid;

/// `SciRS2` Intermediate Representation for Quantum Circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntermediateRepresentation {
    /// IR module identifier
    pub id: Uuid,
    /// Module name
    pub name: String,
    /// Version information
    pub version: String,
    /// IR instructions
    pub instructions: Vec<IRInstruction>,
    /// Symbol table
    pub symbols: SymbolTable,
    /// Metadata
    pub metadata: IRMetadata,
    /// Control flow graph
    pub control_flow: ControlFlowGraph,
    /// Data dependencies
    pub data_dependencies: DependencyGraph,
}

/// IR instruction types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum IRInstruction {
    /// Quantum gate operation
    Gate {
        opcode: GateOpcode,
        operands: Vec<Operand>,
        metadata: InstructionMetadata,
    },
    /// Memory operation
    Memory {
        operation: MemoryOperation,
        address: Operand,
        value: Option<Operand>,
        metadata: InstructionMetadata,
    },
    /// Control flow
    Control {
        operation: ControlOperation,
        condition: Option<Operand>,
        target: Option<String>,
        metadata: InstructionMetadata,
    },
    /// Function call
    Call {
        function: String,
        arguments: Vec<Operand>,
        return_value: Option<Operand>,
        metadata: InstructionMetadata,
    },
    /// Parallel region
    Parallel {
        instructions: Vec<Self>,
        synchronization: SynchronizationType,
        metadata: InstructionMetadata,
    },
}

/// Gate opcodes in IR
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateOpcode {
    // Single-qubit gates
    I,
    X,
    Y,
    Z,
    H,
    S,
    T,
    RX,
    RY,
    RZ,
    U1,
    U2,
    U3,
    // Two-qubit gates
    CX,
    CY,
    CZ,
    CH,
    SWAP,
    ISWAP,
    RXX,
    RYY,
    RZZ,
    // Multi-qubit gates
    CCX,
    CSWAP,
    MCX,
    MCY,
    MCZ,
    // Custom gates
    Custom(String),
}

/// IR operands
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operand {
    /// Quantum register
    QuantumRegister(String, usize),
    /// Classical register
    ClassicalRegister(String, usize),
    /// Immediate value
    Immediate(ImmediateValue),
    /// Memory reference
    Memory(String, usize),
    /// Symbolic reference
    Symbol(String),
}

/// Immediate values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImmediateValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    Complex(f64, f64),
}

/// Memory operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryOperation {
    Load,
    Store,
    Alloc,
    Free,
    Barrier,
}

/// Control operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlOperation {
    Branch,
    Jump,
    Call,
    Return,
    Loop,
    Break,
    Continue,
}

/// Synchronization types for parallel regions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SynchronizationType {
    None,
    Barrier,
    Critical,
    Atomic,
    Reduction,
}

/// Instruction metadata
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct InstructionMetadata {
    /// Instruction ID
    pub id: Option<Uuid>,
    /// Source location
    pub source_location: Option<SourceLocation>,
    /// Optimization hints
    pub optimization_hints: OptimizationHints,
    /// Performance annotations
    pub performance_annotations: Vec<PerformanceAnnotation>,
    /// Target-specific data
    pub target_data: HashMap<String, String>,
}

/// Source location information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

/// Optimization hints
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OptimizationHints {
    /// Can be parallelized
    pub parallelizable: bool,
    /// Memory access pattern
    pub memory_pattern: String,
    /// Expected frequency
    pub frequency: Option<f64>,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Performance annotations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceAnnotation {
    pub annotation_type: String,
    pub value: String,
    pub confidence: f64,
}

/// Symbol table for IR
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolTable {
    /// Symbols map
    pub symbols: HashMap<String, Symbol>,
    /// Scopes
    pub scopes: Vec<Scope>,
}

/// Symbol definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    /// Symbol name
    pub name: String,
    /// Symbol type
    pub symbol_type: SymbolType,
    /// Storage location
    pub storage: StorageLocation,
    /// Scope level
    pub scope: usize,
    /// Attributes
    pub attributes: HashMap<String, String>,
}

/// Symbol types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolType {
    QuantumRegister(usize),
    ClassicalRegister(usize),
    Function(FunctionSignature),
    Constant(ImmediateValue),
    Label,
}

/// Function signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSignature {
    pub parameters: Vec<ParameterType>,
    pub return_type: Option<ParameterType>,
}

/// Parameter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Qubit,
    Classical,
    Real,
    Integer,
    Boolean,
    Array(Box<Self>, usize),
}

/// Storage locations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageLocation {
    Register(usize),
    Memory(usize),
    Stack(isize),
    Global(String),
}

/// Scope information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scope {
    pub level: usize,
    pub parent: Option<usize>,
    pub symbols: HashSet<String>,
}

/// IR metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRMetadata {
    /// Creation timestamp
    pub created: std::time::SystemTime,
    /// Compilation flags
    pub compilation_flags: Vec<String>,
    /// Target platforms
    pub targets: HashSet<String>,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Debug information
    pub debug_info: DebugInfo,
}

impl Default for IRMetadata {
    fn default() -> Self {
        Self {
            created: std::time::SystemTime::now(),
            compilation_flags: Vec::new(),
            targets: HashSet::new(),
            optimization_level: OptimizationLevel::default(),
            debug_info: DebugInfo::default(),
        }
    }
}

/// Optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OptimizationLevel {
    None,
    Debug,
    #[default]
    Release,
    Aggressive,
    Size,
}

/// Debug information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DebugInfo {
    pub source_files: Vec<String>,
    pub line_info: HashMap<Uuid, u32>,
    pub variable_info: HashMap<String, VariableInfo>,
}

/// Variable debug information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableInfo {
    pub name: String,
    pub var_type: String,
    pub scope_start: u32,
    pub scope_end: u32,
}

/// Control flow graph
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ControlFlowGraph {
    /// Basic blocks
    pub blocks: HashMap<String, BasicBlock>,
    /// Entry block
    pub entry: Option<String>,
    /// Exit blocks
    pub exits: HashSet<String>,
    /// Edges between blocks
    pub edges: HashMap<String, Vec<String>>,
}

/// Basic block in control flow graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlock {
    /// Block label
    pub label: String,
    /// Instructions in block
    pub instructions: Vec<usize>, // Indices into IR instructions
    /// Predecessors
    pub predecessors: HashSet<String>,
    /// Successors
    pub successors: HashSet<String>,
}

/// Data dependency graph
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DependencyGraph {
    /// Dependencies between instructions
    pub dependencies: HashMap<usize, HashSet<usize>>,
    /// Reverse dependencies
    pub reverse_dependencies: HashMap<usize, HashSet<usize>>,
    /// Critical path
    pub critical_path: Vec<usize>,
}

/// IR Builder for constructing IR
pub struct IRBuilder {
    /// Current IR being built
    ir: IntermediateRepresentation,
    /// Current instruction index
    current_instruction: usize,
    /// Current scope
    current_scope: usize,
    /// Label counter
    label_counter: usize,
}

impl IRBuilder {
    /// Create new IR builder
    #[must_use]
    pub fn new(name: String) -> Self {
        let mut symbol_table = SymbolTable::default();
        symbol_table.scopes.push(Scope {
            level: 0,
            parent: None,
            symbols: HashSet::new(),
        });

        Self {
            ir: IntermediateRepresentation {
                id: Uuid::new_v4(),
                name,
                version: "1.0.0".to_string(),
                instructions: Vec::new(),
                symbols: symbol_table,
                metadata: IRMetadata::default(),
                control_flow: ControlFlowGraph::default(),
                data_dependencies: DependencyGraph::default(),
            },
            current_instruction: 0,
            current_scope: 0,
            label_counter: 0,
        }
    }

    /// Add gate instruction
    pub fn add_gate(&mut self, opcode: GateOpcode, operands: Vec<Operand>) -> usize {
        let instruction = IRInstruction::Gate {
            opcode,
            operands,
            metadata: InstructionMetadata {
                id: Some(Uuid::new_v4()),
                ..Default::default()
            },
        };

        self.ir.instructions.push(instruction);
        let index = self.ir.instructions.len() - 1;
        self.current_instruction = index;
        index
    }

    /// Add memory instruction
    pub fn add_memory(
        &mut self,
        operation: MemoryOperation,
        address: Operand,
        value: Option<Operand>,
    ) -> usize {
        let instruction = IRInstruction::Memory {
            operation,
            address,
            value,
            metadata: InstructionMetadata {
                id: Some(Uuid::new_v4()),
                ..Default::default()
            },
        };

        self.ir.instructions.push(instruction);
        let index = self.ir.instructions.len() - 1;
        self.current_instruction = index;
        index
    }

    /// Add control instruction
    pub fn add_control(
        &mut self,
        operation: ControlOperation,
        condition: Option<Operand>,
        target: Option<String>,
    ) -> usize {
        let instruction = IRInstruction::Control {
            operation,
            condition,
            target,
            metadata: InstructionMetadata {
                id: Some(Uuid::new_v4()),
                ..Default::default()
            },
        };

        self.ir.instructions.push(instruction);
        let index = self.ir.instructions.len() - 1;
        self.current_instruction = index;
        index
    }

    /// Add parallel region
    pub fn add_parallel(
        &mut self,
        instructions: Vec<IRInstruction>,
        sync: SynchronizationType,
    ) -> usize {
        let instruction = IRInstruction::Parallel {
            instructions,
            synchronization: sync,
            metadata: InstructionMetadata {
                id: Some(Uuid::new_v4()),
                ..Default::default()
            },
        };

        self.ir.instructions.push(instruction);
        let index = self.ir.instructions.len() - 1;
        self.current_instruction = index;
        index
    }

    /// Define symbol
    pub fn define_symbol(
        &mut self,
        name: String,
        symbol_type: SymbolType,
        storage: StorageLocation,
    ) {
        let symbol = Symbol {
            name: name.clone(),
            symbol_type,
            storage,
            scope: self.current_scope,
            attributes: HashMap::new(),
        };

        self.ir.symbols.symbols.insert(name.clone(), symbol);
        if let Some(scope) = self.ir.symbols.scopes.get_mut(self.current_scope) {
            scope.symbols.insert(name);
        }
    }

    /// Generate unique label
    pub fn generate_label(&mut self) -> String {
        let label = format!("L{}", self.label_counter);
        self.label_counter += 1;
        label
    }

    /// Enter new scope
    pub fn enter_scope(&mut self) -> usize {
        let new_level = self.ir.symbols.scopes.len();
        let scope = Scope {
            level: new_level,
            parent: Some(self.current_scope),
            symbols: HashSet::new(),
        };

        self.ir.symbols.scopes.push(scope);
        self.current_scope = new_level;
        new_level
    }

    /// Exit current scope
    pub fn exit_scope(&mut self) {
        if let Some(scope) = self.ir.symbols.scopes.get(self.current_scope) {
            if let Some(parent) = scope.parent {
                self.current_scope = parent;
            }
        }
    }

    /// Build final IR
    #[must_use]
    pub fn build(mut self) -> IntermediateRepresentation {
        self.analyze_control_flow();
        self.analyze_dependencies();
        self.ir
    }

    /// Analyze control flow
    fn analyze_control_flow(&mut self) {
        // Build control flow graph
        let mut current_block = "entry".to_string();
        let mut block_instructions = Vec::new();

        for (i, instruction) in self.ir.instructions.iter().enumerate() {
            match instruction {
                IRInstruction::Control {
                    operation, target, ..
                } => {
                    // Finalize current block
                    if !block_instructions.is_empty() {
                        let block = BasicBlock {
                            label: current_block.clone(),
                            instructions: block_instructions.clone(),
                            predecessors: HashSet::new(),
                            successors: HashSet::new(),
                        };
                        self.ir
                            .control_flow
                            .blocks
                            .insert(current_block.clone(), block);
                        block_instructions.clear();
                    }

                    // Handle control flow
                    match operation {
                        ControlOperation::Branch | ControlOperation::Jump => {
                            if let Some(target_label) = target {
                                // Add edge
                                self.ir
                                    .control_flow
                                    .edges
                                    .entry(current_block.clone())
                                    .or_default()
                                    .push(target_label.clone());

                                current_block = target_label.clone();
                            }
                        }
                        _ => {}
                    }
                }
                _ => {
                    block_instructions.push(i);
                }
            }
        }

        // Finalize last block
        if !block_instructions.is_empty() {
            let block = BasicBlock {
                label: current_block.clone(),
                instructions: block_instructions,
                predecessors: HashSet::new(),
                successors: HashSet::new(),
            };
            self.ir.control_flow.blocks.insert(current_block, block);
        }

        // Set entry block
        if !self.ir.control_flow.blocks.is_empty() {
            self.ir.control_flow.entry = Some("entry".to_string());
        }
    }

    /// Analyze data dependencies
    fn analyze_dependencies(&mut self) {
        // Simple dependency analysis
        for (i, instruction) in self.ir.instructions.iter().enumerate() {
            // Analyze instruction dependencies based on operands
            if let IRInstruction::Gate { operands, .. } = instruction {
                for operand in operands {
                    // Find instructions that define this operand
                    for (j, other_instruction) in self.ir.instructions.iter().enumerate().take(i) {
                        if self.instruction_defines_operand(other_instruction, operand) {
                            self.ir
                                .data_dependencies
                                .dependencies
                                .entry(i)
                                .or_default()
                                .insert(j);

                            self.ir
                                .data_dependencies
                                .reverse_dependencies
                                .entry(j)
                                .or_default()
                                .insert(i);
                        }
                    }
                }
            }
        }
    }

    /// Check if instruction defines operand
    fn instruction_defines_operand(&self, instruction: &IRInstruction, operand: &Operand) -> bool {
        // Simplified check - would be more sophisticated in practice
        match (instruction, operand) {
            (
                IRInstruction::Memory {
                    operation: MemoryOperation::Store,
                    address,
                    ..
                },
                target,
            ) => address == target,
            _ => false,
        }
    }
}

/// IR Optimizer for optimization passes
pub struct IROptimizer {
    /// Available optimization passes
    passes: Vec<Box<dyn IRTransform>>,
    /// Optimization statistics
    stats: OptimizationStats,
}

/// Optimization statistics
#[derive(Debug, Clone, Default)]
pub struct OptimizationStats {
    pub passes_applied: u32,
    pub instructions_eliminated: u32,
    pub instructions_modified: u32,
    pub optimization_time: std::time::Duration,
}

impl IROptimizer {
    /// Create new optimizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            stats: OptimizationStats::default(),
        }
    }

    /// Add optimization pass
    pub fn add_pass(&mut self, pass: Box<dyn IRTransform>) {
        self.passes.push(pass);
    }

    /// Run optimization passes
    pub fn optimize(
        &mut self,
        ir: &mut IntermediateRepresentation,
        level: OptimizationLevel,
    ) -> Result<(), IRError> {
        let start_time = std::time::Instant::now();
        let initial_instruction_count = ir.instructions.len();

        for pass in &mut self.passes {
            if pass.should_run(level) {
                pass.transform(ir)?;
                self.stats.passes_applied += 1;
            }
        }

        let final_instruction_count = ir.instructions.len();
        self.stats.instructions_eliminated +=
            (initial_instruction_count as i32 - final_instruction_count as i32).max(0) as u32;
        self.stats.optimization_time = start_time.elapsed();

        Ok(())
    }

    /// Get optimization statistics
    #[must_use]
    pub const fn get_stats(&self) -> &OptimizationStats {
        &self.stats
    }
}

impl Default for IROptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// IR transformation trait
pub trait IRTransform: Send + Sync {
    /// Apply transformation to IR
    fn transform(&mut self, ir: &mut IntermediateRepresentation) -> Result<(), IRError>;

    /// Check if pass should run at given optimization level
    fn should_run(&self, level: OptimizationLevel) -> bool;

    /// Get pass name
    fn name(&self) -> &str;
}

/// IR validation
pub struct IRValidator {
    /// Validation rules
    rules: Vec<Box<dyn ValidationRule>>,
}

impl IRValidator {
    /// Create new validator
    #[must_use]
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add validation rule
    pub fn add_rule(&mut self, rule: Box<dyn ValidationRule>) {
        self.rules.push(rule);
    }

    /// Validate IR
    pub fn validate(&self, ir: &IntermediateRepresentation) -> Result<ValidationReport, IRError> {
        let mut report = ValidationReport::default();

        for rule in &self.rules {
            let rule_result = rule.validate(ir);
            report.merge(rule_result);
        }

        Ok(report)
    }
}

impl Default for IRValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation rule trait
pub trait ValidationRule: Send + Sync {
    /// Validate IR against this rule
    fn validate(&self, ir: &IntermediateRepresentation) -> ValidationReport;

    /// Get rule name
    fn name(&self) -> &str;
}

/// Validation report
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub passed: bool,
}

impl ValidationReport {
    /// Merge another report
    pub fn merge(&mut self, other: Self) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        self.passed = self.passed && other.passed && self.errors.is_empty();
    }
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub message: String,
    pub location: Option<Uuid>,
    pub severity: ErrorSeverity,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub message: String,
    pub location: Option<Uuid>,
}

/// Error severity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Compilation pass trait
pub trait CompilationPass: Send + Sync {
    /// Run compilation pass
    fn run(&mut self, ir: &mut IntermediateRepresentation) -> Result<(), IRError>;

    /// Get pass dependencies
    fn dependencies(&self) -> Vec<String>;

    /// Get pass name
    fn name(&self) -> &str;
}

/// Target code generator
pub trait TargetGenerator: Send + Sync {
    /// Generate code for target
    fn generate(&self, ir: &IntermediateRepresentation) -> Result<GeneratedCode, IRError>;

    /// Get target name
    fn target_name(&self) -> &str;

    /// Get supported features
    fn supported_features(&self) -> Vec<String>;
}

/// Generated code
#[derive(Debug, Clone)]
pub struct GeneratedCode {
    pub language: String,
    pub code: String,
    pub metadata: CodeGenerationMetadata,
}

/// Code generation metadata
#[derive(Debug, Clone)]
pub struct CodeGenerationMetadata {
    pub generated_at: std::time::SystemTime,
    pub generator_version: String,
    pub target_features: Vec<String>,
    pub optimization_level: OptimizationLevel,
}

impl Default for CodeGenerationMetadata {
    fn default() -> Self {
        Self {
            generated_at: std::time::SystemTime::now(),
            generator_version: String::new(),
            target_features: Vec::new(),
            optimization_level: OptimizationLevel::default(),
        }
    }
}

/// Code emitter for output
pub trait CodeEmitter: Send + Sync {
    /// Emit generated code
    fn emit(&self, code: &GeneratedCode, output_path: &str) -> Result<(), IRError>;

    /// Get supported output formats
    fn supported_formats(&self) -> Vec<String>;
}

/// IR error types
#[derive(Debug, Clone)]
pub enum IRError {
    InvalidInstruction(String),
    UndefinedSymbol(String),
    InvalidOperand(String),
    OptimizationFailed(String),
    ValidationFailed(String),
    CodeGenerationFailed(String),
    InternalError(String),
}

impl fmt::Display for IRError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInstruction(msg) => write!(f, "Invalid instruction: {msg}"),
            Self::UndefinedSymbol(symbol) => write!(f, "Undefined symbol: {symbol}"),
            Self::InvalidOperand(msg) => write!(f, "Invalid operand: {msg}"),
            Self::OptimizationFailed(msg) => write!(f, "Optimization failed: {msg}"),
            Self::ValidationFailed(msg) => write!(f, "Validation failed: {msg}"),
            Self::CodeGenerationFailed(msg) => write!(f, "Code generation failed: {msg}"),
            Self::InternalError(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl std::error::Error for IRError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_builder() {
        let mut builder = IRBuilder::new("test_module".to_string());

        // Add some instructions
        builder.add_gate(
            GateOpcode::H,
            vec![Operand::QuantumRegister("q".to_string(), 0)],
        );

        builder.add_gate(
            GateOpcode::CX,
            vec![
                Operand::QuantumRegister("q".to_string(), 0),
                Operand::QuantumRegister("q".to_string(), 1),
            ],
        );

        let ir = builder.build();
        assert_eq!(ir.instructions.len(), 2);
        assert_eq!(ir.name, "test_module");
    }

    #[test]
    fn test_symbol_table() {
        let mut builder = IRBuilder::new("test".to_string());

        builder.define_symbol(
            "q".to_string(),
            SymbolType::QuantumRegister(2),
            StorageLocation::Register(0),
        );

        let ir = builder.build();
        assert!(ir.symbols.symbols.contains_key("q"));
    }

    #[test]
    fn test_ir_optimizer() {
        let mut optimizer = IROptimizer::new();
        let mut ir = IRBuilder::new("test".to_string()).build();

        let result = optimizer.optimize(&mut ir, OptimizationLevel::Release);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ir_validator() {
        let validator = IRValidator::new();
        let ir = IRBuilder::new("test".to_string()).build();

        let result = validator.validate(&ir);
        assert!(result.is_ok());
    }
}
