//! # EnhancedQASMCompiler - validation Methods
//!
//! This module contains method implementations for `EnhancedQASMCompiler`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_core::buffer_pool::BufferPool;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
    register::Register,
};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use super::types::{
    ASTNode, CodeGenerator, CompilationCache, CompilationResult, CompilationWarning,
    EnhancedQASMConfig, ErrorRecovery, ErrorType, ExportFormat, HardwareConstraints, MLOptimizer,
    QASMOptimizer, QASMParser, QASMVersion, SemanticAnalyzer, TypeChecker, TypeCheckingLevel,
    TypeError, ValidationError, ValidationResult, VersionConverter, AST,
};

use super::enhancedqasmcompiler_type::EnhancedQASMCompiler;

impl EnhancedQASMCompiler {
    /// Create a new enhanced QASM compiler
    #[must_use]
    pub fn new(config: EnhancedQASMConfig) -> Self {
        let parser = Arc::new(QASMParser::new(config.base_config.qasm_version));
        let semantic_analyzer = Arc::new(SemanticAnalyzer::new());
        let optimizer = Arc::new(QASMOptimizer::new(config.optimization_level));
        let code_generator = Arc::new(CodeGenerator::new());
        let ml_optimizer = if config.enable_ml_optimization {
            Some(Arc::new(MLOptimizer::new()))
        } else {
            None
        };
        let error_recovery = Arc::new(ErrorRecovery::new());
        let buffer_pool = BufferPool::new();
        let cache = Arc::new(Mutex::new(CompilationCache::new()));
        Self {
            config,
            parser,
            semantic_analyzer,
            optimizer,
            code_generator,
            ml_optimizer,
            error_recovery,
            buffer_pool,
            cache,
        }
    }
    /// Compile QASM code to target format
    pub fn compile(&self, source: &str) -> QuantRS2Result<CompilationResult> {
        let start_time = std::time::Instant::now();
        if let Some(cached) = self.check_cache(source)? {
            return Ok(cached);
        }
        let tokens = Self::lexical_analysis(source)?;
        let ast = self.parse_with_recovery(&tokens)?;
        let semantic_ast = if self.config.enable_semantic_analysis {
            SemanticAnalyzer::analyze(ast)?
        } else {
            ast
        };
        let optimized_ast = self.optimize_ast(semantic_ast)?;
        let mut generated_code = HashMap::new();
        for target in &self.config.compilation_targets {
            let code = CodeGenerator::generate(&optimized_ast, *target)?;
            generated_code.insert(*target, code);
        }
        let exports = self.export_to_formats(&optimized_ast)?;
        let visualizations = if self.config.enable_visual_ast {
            Some(Self::generate_visualizations(&optimized_ast)?)
        } else {
            None
        };
        let compilation_time = start_time.elapsed();
        let result = CompilationResult {
            ast: optimized_ast,
            generated_code,
            exports,
            visualizations,
            compilation_time,
            statistics: Self::calculate_statistics(&tokens)?,
            warnings: Self::collect_warnings()?,
            optimizations_applied: self.optimizer.get_applied_optimizations(),
        };
        self.cache_result(source, &result)?;
        Ok(result)
    }
    /// Validate QASM code
    pub fn validate(&self, source: &str) -> QuantRS2Result<ValidationResult> {
        let tokens = Self::lexical_analysis(source)?;
        let ast = match QASMParser::parse(&tokens) {
            Ok(ast) => ast,
            Err(e) => {
                return Ok(ValidationResult {
                    is_valid: false,
                    errors: vec![ValidationError {
                        error_type: ErrorType::SyntaxError,
                        message: e.to_string(),
                        location: None,
                        suggestion: Some("Check QASM syntax".to_string()),
                    }],
                    warnings: Vec::new(),
                    info: Vec::new(),
                });
            }
        };
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        if self.config.enable_semantic_analysis {
            let semantic_result = SemanticAnalyzer::validate(&ast)?;
            errors.extend(semantic_result.errors);
            warnings.extend(semantic_result.warnings);
        }
        if self.config.analysis_options.type_checking != TypeCheckingLevel::None {
            let type_errors = Self::type_check(&ast)?;
            errors.extend(type_errors);
        }
        if let Some(ref constraints) = self.config.base_config.hardware_constraints {
            let hw_errors = Self::validate_hardware_constraints(&ast, constraints)?;
            errors.extend(hw_errors);
        }
        Ok(ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            info: Self::collect_info(&ast)?,
        })
    }
    /// Export to various formats
    pub(super) fn export_to_formats(
        &self,
        ast: &AST,
    ) -> QuantRS2Result<HashMap<ExportFormat, Vec<u8>>> {
        let mut exports = HashMap::new();
        for format in &self.config.export_formats {
            let data = match format {
                ExportFormat::QuantRS2Native => Self::export_quantrs2_native(ast)?,
                ExportFormat::QASM2 => Self::export_qasm2(ast)?,
                ExportFormat::QASM3 => Self::export_qasm3(ast)?,
                ExportFormat::OpenQASM => Self::export_openqasm(ast)?,
                ExportFormat::Qiskit => Self::export_qiskit(ast)?,
                ExportFormat::Cirq => Self::export_cirq(ast)?,
                ExportFormat::JSON => Self::export_json(ast)?,
                ExportFormat::Binary => Self::export_binary(ast)?,
            };
            exports.insert(*format, data);
        }
        Ok(exports)
    }
    /// Type checking
    pub(super) fn type_check(ast: &AST) -> QuantRS2Result<Vec<ValidationError>> {
        let mut errors = Vec::new();
        let type_checker = TypeChecker::new(TypeCheckingLevel::Strict);
        for node in ast.nodes() {
            if let Err(e) = TypeChecker::check_node(node) {
                errors.push(ValidationError {
                    error_type: ErrorType::TypeError,
                    message: e.to_string(),
                    location: Some(ASTNode::location()),
                    suggestion: Some(TypeChecker::suggest_fix(&e)),
                });
            }
        }
        Ok(errors)
    }
    /// Validate hardware constraints
    pub(super) fn validate_hardware_constraints(
        ast: &AST,
        constraints: &HardwareConstraints,
    ) -> QuantRS2Result<Vec<ValidationError>> {
        let mut errors = Vec::new();
        let used_qubits = Self::extract_used_qubits(ast)?;
        if used_qubits.len() > constraints.max_qubits {
            errors.push(ValidationError {
                error_type: ErrorType::HardwareConstraint,
                message: format!(
                    "Circuit uses {} qubits, but hardware supports only {}",
                    used_qubits.len(),
                    constraints.max_qubits
                ),
                location: None,
                suggestion: Some("Consider using fewer qubits or different hardware".to_string()),
            });
        }
        let two_qubit_gates = Self::extract_two_qubit_gates(ast)?;
        for (q1, q2) in two_qubit_gates {
            if !constraints.connectivity.contains(&(q1, q2))
                && !constraints.connectivity.contains(&(q2, q1))
            {
                errors.push(ValidationError {
                    error_type: ErrorType::HardwareConstraint,
                    message: format!("No connection between qubits {q1} and {q2}"),
                    location: None,
                    suggestion: Some("Add SWAP gates or use different qubits".to_string()),
                });
            }
        }
        let used_gates = Self::extract_used_gates(ast)?;
        for gate in used_gates {
            if !constraints.native_gates.contains(&gate) {
                errors.push(ValidationError {
                    error_type: ErrorType::HardwareConstraint,
                    message: format!("Gate '{gate}' is not native to the hardware"),
                    location: None,
                    suggestion: Some("Decompose to native gates".to_string()),
                });
            }
        }
        Ok(errors)
    }
    /// Convert AST between versions
    pub(super) fn convert_ast_version(ast: AST, target: QASMVersion) -> QuantRS2Result<AST> {
        let converter = VersionConverter::new(Self::detect_ast_version(&ast)?, target);
        VersionConverter::convert(ast)
    }
    pub(super) fn extract_metadata(source: &str) -> QuantRS2Result<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        for line in source.lines() {
            if let Some(rest) = line.strip_prefix("// @") {
                if let Some((key, value)) = rest.split_once(':') {
                    metadata.insert(key.trim().to_string(), value.trim().to_string());
                }
            }
        }
        Ok(metadata)
    }
    pub(super) fn extract_includes(source: &str) -> QuantRS2Result<Vec<String>> {
        let mut includes = Vec::new();
        for line in source.lines() {
            if line.trim().starts_with("include") {
                if let Some(file) = line.split('"').nth(1) {
                    includes.push(file.to_string());
                }
            }
        }
        Ok(includes)
    }
    pub(super) fn collect_warnings() -> QuantRS2Result<Vec<CompilationWarning>> {
        Ok(Vec::new())
    }
    pub(super) fn extract_used_qubits(_ast: &AST) -> QuantRS2Result<HashSet<usize>> {
        Ok(HashSet::new())
    }
    pub(super) fn extract_two_qubit_gates(_ast: &AST) -> QuantRS2Result<Vec<(usize, usize)>> {
        Ok(Vec::new())
    }
    pub(super) fn extract_used_gates(_ast: &AST) -> QuantRS2Result<HashSet<String>> {
        Ok(HashSet::new())
    }
    pub(super) fn ast_to_circuit<const N: usize>(
        _ast: &AST,
    ) -> QuantRS2Result<crate::builder::Circuit<N>> {
        Ok(crate::builder::Circuit::<N>::new())
    }
}
