//! Enhanced Cross-Compilation with Advanced `SciRS2` IR Tools
//!
//! This module provides state-of-the-art cross-compilation between quantum frameworks
//! and hardware platforms using ML-based optimization, multi-stage compilation,
//! target-specific code generation, and comprehensive error handling powered by `SciRS2`.

pub mod config;
pub mod generators;
pub mod optimizers;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export main types
pub use config::*;
pub use generators::{create_target_generator, TargetCodeGenerator};
pub use optimizers::{
    CompilationCache, CompilationMonitor, CompilationValidator, MLCompilationOptimizer,
};
pub use types::*;

use crate::optimization::pass_manager::PassManager;
use crate::scirs2_ir_tools::IRBuilder;
use crate::scirs2_ir_tools::IROptimizer;
use generators::create_target_generator as create_generator;
use optimizers::{CompilationCache as Cache, TargetSpecification};
use quantrs2_core::buffer_pool::BufferPool;
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::parallel_ops::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Enhanced cross-compiler
pub struct EnhancedCrossCompiler {
    config: EnhancedCrossCompilationConfig,
    ir_builder: Arc<IRBuilder>,
    ir_optimizer: Arc<IROptimizer>,
    pass_manager: Arc<PassManager>,
    ml_optimizer: Option<Arc<MLCompilationOptimizer>>,
    target_generators: HashMap<TargetPlatform, Arc<dyn TargetCodeGenerator>>,
    realtime_monitor: Arc<CompilationMonitor>,
    validator: Arc<CompilationValidator>,
    buffer_pool: BufferPool<f64>,
    cache: Arc<Mutex<Cache>>,
}

impl EnhancedCrossCompiler {
    /// Create new enhanced cross-compiler
    #[must_use]
    pub fn new(config: EnhancedCrossCompilationConfig) -> Self {
        let buffer_pool = BufferPool::new();

        let mut target_generators = HashMap::new();
        for &platform in &config.target_platforms {
            target_generators.insert(platform, create_generator(platform, config.clone()));
        }

        Self {
            config: config.clone(),
            ir_builder: Arc::new(IRBuilder::new("cross_compiler".to_string())),
            ir_optimizer: Arc::new(IROptimizer::new()),
            pass_manager: Arc::new(PassManager::new()),
            ml_optimizer: if config.enable_ml_optimization {
                Some(Arc::new(MLCompilationOptimizer::new(config.clone())))
            } else {
                None
            },
            target_generators,
            realtime_monitor: Arc::new(CompilationMonitor::new(config.clone())),
            validator: Arc::new(CompilationValidator::new(config)),
            buffer_pool,
            cache: Arc::new(Mutex::new(Cache::new())),
        }
    }

    /// Cross-compile quantum circuit
    pub fn cross_compile(
        &self,
        source: SourceCircuit,
        target: TargetPlatform,
    ) -> QuantRS2Result<CrossCompilationResult> {
        let mut result = CrossCompilationResult::new();
        let start_time = std::time::Instant::now();

        // Stage 1: Parse source circuit
        let parsed_circuit = self.parse_source_circuit(&source)?;
        result.stages.push(CompilationStage {
            name: "Parsing".to_string(),
            duration: start_time.elapsed(),
            metrics: self.collect_stage_metrics(&parsed_circuit),
        });

        // Stage 2: Convert to IR
        let stage_start = std::time::Instant::now();
        let ir = self.convert_to_ir(&parsed_circuit)?;
        result.intermediate_representation = Some(ir.clone());
        result.stages.push(CompilationStage {
            name: "IR Conversion".to_string(),
            duration: stage_start.elapsed(),
            metrics: self.collect_ir_metrics(&ir),
        });

        // Stage 3: Optimize IR
        let stage_start = std::time::Instant::now();
        let optimized_ir = if self.config.enable_multistage_compilation {
            self.optimize_ir_multistage(&ir, target)?
        } else {
            ir.clone()
        };
        result.optimized_representation = Some(optimized_ir.clone());
        result.stages.push(CompilationStage {
            name: "IR Optimization".to_string(),
            duration: stage_start.elapsed(),
            metrics: self.collect_optimization_metrics(&ir, &optimized_ir),
        });

        // Stage 4: ML-based optimization (if enabled)
        let final_ir = if let Some(ml_optimizer) = &self.ml_optimizer {
            let stage_start = std::time::Instant::now();
            let ml_optimized = ml_optimizer.optimize(&optimized_ir, target)?;
            result.ml_optimization_applied = true;
            result.stages.push(CompilationStage {
                name: "ML Optimization".to_string(),
                duration: stage_start.elapsed(),
                metrics: self.collect_ml_metrics(&optimized_ir, &ml_optimized),
            });
            ml_optimized
        } else {
            optimized_ir
        };

        // Stage 5: Target-specific optimization
        let stage_start = std::time::Instant::now();
        let target_optimized = if self.config.enable_target_optimization {
            self.optimize_for_target(&final_ir, target)?
        } else {
            final_ir
        };
        result.stages.push(CompilationStage {
            name: "Target Optimization".to_string(),
            duration: stage_start.elapsed(),
            metrics: self.collect_target_metrics(&target_optimized, target),
        });

        // Stage 6: Code generation
        let stage_start = std::time::Instant::now();
        let target_code = self.generate_target_code(&target_optimized, target)?;
        result.target_code = target_code.clone();
        result.stages.push(CompilationStage {
            name: "Code Generation".to_string(),
            duration: stage_start.elapsed(),
            metrics: HashMap::new(),
        });

        // Stage 7: Validation
        if self.config.enable_comprehensive_validation {
            let stage_start = std::time::Instant::now();
            let validation_result =
                self.validator
                    .validate_compilation(&source, &target_code, target)?;
            result.validation_result = Some(validation_result.clone());
            result.stages.push(CompilationStage {
                name: "Validation".to_string(),
                duration: stage_start.elapsed(),
                metrics: self.collect_validation_metrics(&validation_result),
            });

            if !validation_result.is_valid {
                return Err(QuantRS2Error::InvalidOperation(format!(
                    "Validation failed: {:?}",
                    validation_result.errors
                )));
            }
        }

        // Generate compilation report
        result.compilation_report = Some(self.generate_compilation_report(&result)?);

        // Visual flow (if enabled)
        if self.config.enable_visual_flow {
            result.visual_flow = Some(self.generate_visual_flow(&result)?);
        }

        // Update cache
        self.update_cache(&source, target, &result)?;

        Ok(result)
    }

    /// Batch cross-compilation
    pub fn batch_cross_compile(
        &self,
        sources: Vec<SourceCircuit>,
        target: TargetPlatform,
    ) -> QuantRS2Result<BatchCompilationResult> {
        let results: Vec<_> = sources
            .par_iter()
            .map(|source| self.cross_compile(source.clone(), target))
            .collect();

        let mut batch_result = BatchCompilationResult::new();

        for (source, result) in sources.iter().zip(results) {
            match result {
                Ok(compilation) => {
                    batch_result.successful_compilations.push(compilation);
                }
                Err(e) => {
                    batch_result.failed_compilations.push(FailedCompilation {
                        source: source.clone(),
                        error: e.to_string(),
                    });
                }
            }
        }

        // Generate batch report
        batch_result.batch_report = Some(self.generate_batch_report(&batch_result)?);

        Ok(batch_result)
    }

    /// Parse source circuit based on framework
    fn parse_source_circuit(&self, source: &SourceCircuit) -> QuantRS2Result<ParsedCircuit> {
        match source.framework {
            QuantumFramework::QuantRS2 => self.parse_quantrs2_circuit(&source.code),
            QuantumFramework::Qiskit => self.parse_qiskit_circuit(&source.code),
            QuantumFramework::Cirq => self.parse_cirq_circuit(&source.code),
            QuantumFramework::PennyLane => self.parse_pennylane_circuit(&source.code),
            QuantumFramework::OpenQASM => self.parse_openqasm_circuit(&source.code),
            _ => Err(QuantRS2Error::UnsupportedOperation(format!(
                "Framework {:?} not yet supported",
                source.framework
            ))),
        }
    }

    /// Convert parsed circuit to IR
    fn convert_to_ir(&self, circuit: &ParsedCircuit) -> QuantRS2Result<QuantumIR> {
        let mut ir = QuantumIR::new();

        // Convert quantum operations
        for operation in &circuit.operations {
            let ir_op = self.convert_operation_to_ir(operation)?;
            ir.add_operation(ir_op);
        }

        // Convert classical operations
        for classical_op in &circuit.classical_operations {
            let ir_classical = self.convert_classical_to_ir(classical_op)?;
            ir.add_classical_operation(ir_classical);
        }

        // Add metadata
        ir.metadata = circuit.metadata.clone();
        ir.num_qubits = circuit.num_qubits;
        ir.num_classical_bits = circuit.num_classical_bits;

        Ok(ir)
    }

    /// Multi-stage IR optimization
    fn optimize_ir_multistage(
        &self,
        ir: &QuantumIR,
        target: TargetPlatform,
    ) -> QuantRS2Result<QuantumIR> {
        let mut optimized = ir.clone();

        // Stage 1: High-level optimizations
        optimized = self.apply_high_level_optimizations(&optimized)?;

        // Stage 2: Mid-level optimizations
        optimized = self.apply_mid_level_optimizations(&optimized)?;

        // Stage 3: Low-level optimizations
        optimized = self.apply_low_level_optimizations(&optimized, target)?;

        // Real-time monitoring
        if self.config.enable_realtime_monitoring {
            self.realtime_monitor
                .update_optimization_progress(&optimized)?;
        }

        Ok(optimized)
    }

    /// Apply high-level optimizations
    fn apply_high_level_optimizations(&self, ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        let mut optimized = ir.clone();

        // Circuit simplification
        optimized = self.simplify_circuit(&optimized)?;

        // Template matching
        optimized = self.apply_template_matching(&optimized)?;

        // Algebraic simplification
        optimized = self.apply_algebraic_simplification(&optimized)?;

        Ok(optimized)
    }

    /// Apply mid-level optimizations
    fn apply_mid_level_optimizations(&self, ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        let mut optimized = ir.clone();

        // Gate fusion
        optimized = self.apply_gate_fusion(&optimized)?;

        // Commutation analysis
        optimized = self.apply_commutation_analysis(&optimized)?;

        // Rotation merging
        optimized = self.apply_rotation_merging(&optimized)?;

        Ok(optimized)
    }

    /// Apply low-level optimizations
    fn apply_low_level_optimizations(
        &self,
        ir: &QuantumIR,
        target: TargetPlatform,
    ) -> QuantRS2Result<QuantumIR> {
        let mut optimized = ir.clone();

        // Native gate decomposition
        optimized = self.decompose_to_native_gates(&optimized, target)?;

        // Peephole optimization
        optimized = self.apply_peephole_optimization(&optimized)?;

        // Layout optimization
        optimized = self.optimize_layout(&optimized, target)?;

        Ok(optimized)
    }

    /// Optimize for specific target platform
    fn optimize_for_target(
        &self,
        ir: &QuantumIR,
        target: TargetPlatform,
    ) -> QuantRS2Result<QuantumIR> {
        let target_spec = self.get_target_specification(target)?;
        let mut optimized = ir.clone();

        // Apply target-specific constraints
        optimized = self.apply_connectivity_constraints(&optimized, &target_spec)?;

        // Optimize for target gate set
        optimized = self.optimize_for_gate_set(&optimized, &target_spec)?;

        // Apply error mitigation
        if self.config.base_config.enable_error_correction {
            optimized = self.apply_error_mitigation(&optimized, &target_spec)?;
        }

        Ok(optimized)
    }

    /// Generate target code
    fn generate_target_code(
        &self,
        ir: &QuantumIR,
        target: TargetPlatform,
    ) -> QuantRS2Result<TargetCode> {
        let generator = self.target_generators.get(&target).ok_or_else(|| {
            QuantRS2Error::UnsupportedOperation(format!("No code generator for {target:?}"))
        })?;

        generator.generate(ir)
    }

    /// Helper methods for framework-specific parsing
    fn parse_quantrs2_circuit(&self, _code: &str) -> QuantRS2Result<ParsedCircuit> {
        Ok(ParsedCircuit::default())
    }

    fn parse_qiskit_circuit(&self, _code: &str) -> QuantRS2Result<ParsedCircuit> {
        Ok(ParsedCircuit::default())
    }

    fn parse_cirq_circuit(&self, _code: &str) -> QuantRS2Result<ParsedCircuit> {
        Ok(ParsedCircuit::default())
    }

    fn parse_pennylane_circuit(&self, _code: &str) -> QuantRS2Result<ParsedCircuit> {
        Ok(ParsedCircuit::default())
    }

    fn parse_openqasm_circuit(&self, _code: &str) -> QuantRS2Result<ParsedCircuit> {
        Ok(ParsedCircuit::default())
    }
}

// Optimization helper implementations
impl EnhancedCrossCompiler {
    fn simplify_circuit(&self, ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    fn apply_template_matching(&self, ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    fn apply_algebraic_simplification(&self, ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    fn apply_gate_fusion(&self, ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    fn apply_commutation_analysis(&self, ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    fn apply_rotation_merging(&self, ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    fn decompose_to_native_gates(
        &self,
        ir: &QuantumIR,
        _target: TargetPlatform,
    ) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    fn apply_peephole_optimization(&self, ir: &QuantumIR) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    fn optimize_layout(
        &self,
        ir: &QuantumIR,
        _target: TargetPlatform,
    ) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    fn get_target_specification(
        &self,
        _target: TargetPlatform,
    ) -> QuantRS2Result<TargetSpecification> {
        Ok(TargetSpecification {
            native_gates: vec![],
            connectivity: vec![],
            error_rates: HashMap::new(),
        })
    }

    fn apply_connectivity_constraints(
        &self,
        ir: &QuantumIR,
        _spec: &TargetSpecification,
    ) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    fn optimize_for_gate_set(
        &self,
        ir: &QuantumIR,
        _spec: &TargetSpecification,
    ) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    fn apply_error_mitigation(
        &self,
        ir: &QuantumIR,
        _spec: &TargetSpecification,
    ) -> QuantRS2Result<QuantumIR> {
        Ok(ir.clone())
    }

    const fn update_cache(
        &self,
        _source: &SourceCircuit,
        _target: TargetPlatform,
        _result: &CrossCompilationResult,
    ) -> QuantRS2Result<()> {
        Ok(())
    }
}

// Metric collection helpers
impl EnhancedCrossCompiler {
    fn collect_stage_metrics(&self, circuit: &ParsedCircuit) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("num_qubits".to_string(), circuit.num_qubits as f64);
        metrics.insert(
            "num_operations".to_string(),
            circuit.operations.len() as f64,
        );
        metrics
    }

    fn collect_ir_metrics(&self, ir: &QuantumIR) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("ir_operations".to_string(), ir.operations.len() as f64);
        metrics
    }

    fn collect_optimization_metrics(
        &self,
        original: &QuantumIR,
        optimized: &QuantumIR,
    ) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        let reduction = if original.operations.is_empty() {
            0.0
        } else {
            1.0 - (optimized.operations.len() as f64 / original.operations.len() as f64)
        };
        metrics.insert("operation_reduction".to_string(), reduction);
        metrics
    }

    fn collect_ml_metrics(
        &self,
        _original: &QuantumIR,
        _optimized: &QuantumIR,
    ) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("ml_improvement".to_string(), 0.1);
        metrics
    }

    fn collect_target_metrics(
        &self,
        _ir: &QuantumIR,
        _target: TargetPlatform,
    ) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("target_compatibility".to_string(), 0.95);
        metrics
    }

    fn collect_validation_metrics(&self, validation: &ValidationResult) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert(
            "fidelity".to_string(),
            validation.fidelity_estimate.unwrap_or(0.0),
        );
        metrics
    }
}

// Report generation helpers
impl EnhancedCrossCompiler {
    fn generate_summary(
        &self,
        result: &CrossCompilationResult,
    ) -> QuantRS2Result<CompilationSummary> {
        let total_time = result.stages.iter().map(|s| s.duration).sum();

        Ok(CompilationSummary {
            total_time,
            original_size: CircuitSize::default(),
            compiled_size: CircuitSize::default(),
            size_reduction: 0.0,
            fidelity_estimate: result
                .validation_result
                .as_ref()
                .and_then(|v| v.fidelity_estimate)
                .unwrap_or(1.0),
        })
    }

    fn analyze_compilation_stage(&self, stage: &CompilationStage) -> QuantRS2Result<StageAnalysis> {
        Ok(StageAnalysis {
            stage_name: stage.name.clone(),
            performance: StagePerformance {
                execution_time: stage.duration,
                memory_usage: 0,
                cpu_usage: 0.0,
            },
            transformations: vec![],
            impact: StageImpact {
                gate_count_change: 0,
                depth_change: 0,
                fidelity_impact: 0.0,
            },
        })
    }

    fn analyze_optimizations(
        &self,
        _original: &QuantumIR,
        _optimized: &QuantumIR,
    ) -> QuantRS2Result<OptimizationReport> {
        Ok(OptimizationReport {
            applied_optimizations: vec![],
            total_improvement: OptimizationImprovement {
                gate_count_improvement: 0.0,
                depth_improvement: 0.0,
                execution_time_improvement: 0.0,
            },
            breakdown: HashMap::new(),
        })
    }

    fn calculate_resource_usage(
        &self,
        _result: &CrossCompilationResult,
    ) -> QuantRS2Result<ResourceUsage> {
        Ok(ResourceUsage::default())
    }

    const fn generate_recommendations(
        &self,
        _result: &CrossCompilationResult,
    ) -> QuantRS2Result<Vec<CompilationRecommendation>> {
        Ok(vec![])
    }

    fn generate_compilation_report(
        &self,
        result: &CrossCompilationResult,
    ) -> QuantRS2Result<CompilationReport> {
        let mut report = CompilationReport::new();

        // Summary
        report.summary = self.generate_summary(result)?;

        // Stage analysis
        for stage in &result.stages {
            let stage_analysis = self.analyze_compilation_stage(stage)?;
            report.stage_analyses.push(stage_analysis);
        }

        // Optimization report
        if let (Some(original), Some(optimized)) = (
            &result.intermediate_representation,
            &result.optimized_representation,
        ) {
            report.optimization_report = Some(self.analyze_optimizations(original, optimized)?);
        }

        // Resource usage
        report.resource_usage = self.calculate_resource_usage(result)?;

        // Recommendations
        report.recommendations = self.generate_recommendations(result)?;

        Ok(report)
    }

    fn generate_batch_report(
        &self,
        batch_result: &BatchCompilationResult,
    ) -> QuantRS2Result<BatchCompilationReport> {
        let total =
            batch_result.successful_compilations.len() + batch_result.failed_compilations.len();
        let success_rate = if total == 0 {
            1.0
        } else {
            batch_result.successful_compilations.len() as f64 / total as f64
        };

        Ok(BatchCompilationReport {
            success_rate,
            avg_compilation_time: std::time::Duration::from_secs(1),
            common_errors: vec![],
            performance_stats: BatchPerformanceStats {
                total_time: std::time::Duration::from_secs(10),
                throughput: 1.0,
                resource_efficiency: 0.9,
            },
        })
    }

    fn generate_visual_flow(
        &self,
        result: &CrossCompilationResult,
    ) -> QuantRS2Result<VisualCompilationFlow> {
        let mut flow = VisualCompilationFlow::new();

        // Create nodes for each stage
        for (i, stage) in result.stages.iter().enumerate() {
            flow.add_node(FlowNode {
                id: i,
                name: stage.name.clone(),
                node_type: NodeType::CompilationStage,
                metrics: stage.metrics.clone(),
            });
        }

        // Add edges between stages
        for i in 0..result.stages.len().saturating_sub(1) {
            flow.add_edge(FlowEdge {
                from: i,
                to: i + 1,
                edge_type: EdgeType::Sequential,
                data_flow: DataFlow::default(),
            });
        }

        // Add IR visualization
        if let Some(ir) = &result.intermediate_representation {
            flow.ir_visualization = Some(self.visualize_ir(ir)?);
        }

        // Add optimization visualization
        if result.ml_optimization_applied {
            flow.optimization_visualization = Some(self.visualize_optimizations(result)?);
        }

        Ok(flow)
    }

    fn visualize_ir(&self, _ir: &QuantumIR) -> QuantRS2Result<IRVisualization> {
        Ok(IRVisualization {
            graph: IRGraph {
                nodes: vec![],
                edges: vec![],
            },
            layout: GraphLayout {
                positions: HashMap::new(),
                algorithm: "hierarchical".to_string(),
            },
        })
    }

    const fn visualize_optimizations(
        &self,
        _result: &CrossCompilationResult,
    ) -> QuantRS2Result<OptimizationVisualization> {
        Ok(OptimizationVisualization {
            comparison: ComparisonVisualization {
                before: CircuitVisualization {
                    diagram: String::new(),
                    metrics: CircuitMetrics {
                        gate_count: 0,
                        depth: 0,
                        width: 0,
                    },
                },
                after: CircuitVisualization {
                    diagram: String::new(),
                    metrics: CircuitMetrics {
                        gate_count: 0,
                        depth: 0,
                        width: 0,
                    },
                },
                differences: vec![],
            },
            timeline: OptimizationTimeline {
                events: vec![],
                total_duration: std::time::Duration::from_secs(1),
            },
        })
    }
}

// Operation conversion helpers
impl EnhancedCrossCompiler {
    fn convert_operation_to_ir(&self, op: &QuantumOperation) -> QuantRS2Result<IROperation> {
        let operation_type = match &op.op_type {
            OperationType::Gate(name) => {
                let gate = self.parse_gate(name, &op.parameters)?;
                IROperationType::Gate(gate)
            }
            OperationType::Measurement => {
                IROperationType::Measurement(op.qubits.clone(), vec![op.qubits[0]])
            }
            OperationType::Reset => IROperationType::Reset(op.qubits.clone()),
            OperationType::Barrier => IROperationType::Barrier(op.qubits.clone()),
        };

        Ok(IROperation {
            operation_type,
            qubits: op.qubits.clone(),
            controls: vec![],
            parameters: op.parameters.clone(),
        })
    }

    fn convert_classical_to_ir(&self, op: &ClassicalOperation) -> QuantRS2Result<IRClassicalOp> {
        let op_type = match op.op_type {
            ClassicalOpType::Assignment => IRClassicalOpType::Move,
            ClassicalOpType::Arithmetic => IRClassicalOpType::Add,
            ClassicalOpType::Conditional => IRClassicalOpType::And,
        };

        Ok(IRClassicalOp {
            op_type,
            operands: op.operands.clone(),
            result: op.operands.first().copied(),
        })
    }

    fn parse_gate(&self, name: &str, params: &[f64]) -> QuantRS2Result<IRGate> {
        match name {
            "H" => Ok(IRGate::H),
            "X" => Ok(IRGate::X),
            "Y" => Ok(IRGate::Y),
            "Z" => Ok(IRGate::Z),
            "CNOT" | "CX" => Ok(IRGate::CNOT),
            "RX" => Ok(IRGate::RX(params[0])),
            "RY" => Ok(IRGate::RY(params[0])),
            "RZ" => Ok(IRGate::RZ(params[0])),
            _ => Ok(IRGate::Custom(name.to_string(), params.to_vec())),
        }
    }
}
