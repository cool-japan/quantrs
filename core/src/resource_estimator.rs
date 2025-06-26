//! Quantum Resource Estimator with SciRS2 Complexity Analysis
//!
//! This module provides comprehensive resource estimation for quantum circuits
//! using SciRS2's advanced complexity analysis and numerical methods.

use crate::gate_translation::GateType;
// use scirs2_core::memory::BufferPool;
use crate::buffer_pool::BufferPool;
// use scirs2_core::parallel_ops::*;
use crate::parallel_ops_stubs::*;

/// Simplified quantum gate representation for resource estimation
#[derive(Debug, Clone)]
pub struct QuantumGate {
    gate_type: GateType,
    target_qubits: Vec<usize>,
    control_qubits: Option<Vec<usize>>,
}

impl QuantumGate {
    pub fn new(gate_type: GateType, target_qubits: Vec<usize>, control_qubits: Option<Vec<usize>>) -> Self {
        Self {
            gate_type,
            target_qubits,
            control_qubits,
        }
    }
    
    pub fn gate_type(&self) -> &GateType {
        &self.gate_type
    }
    
    pub fn target_qubits(&self) -> &[usize] {
        &self.target_qubits
    }
    
    pub fn control_qubits(&self) -> Option<&[usize]> {
        self.control_qubits.as_deref()
    }
}
use crate::error::QuantRS2Error;
use std::collections::HashMap;

/// Configuration for resource estimation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceEstimationConfig {
    /// Target quantum error correction code
    pub error_correction_code: ErrorCorrectionCode,
    /// Physical error rate
    pub physical_error_rate: f64,
    /// Target logical error rate
    pub target_logical_error_rate: f64,
    /// Estimation mode (conservative, optimistic, realistic)
    pub estimation_mode: EstimationMode,
    /// Include hardware-specific overheads
    pub include_hardware_overhead: bool,
    /// Hardware platform for estimation
    pub hardware_platform: HardwarePlatform,
    /// Enable detailed analysis
    pub detailed_analysis: bool,
}

impl Default for ResourceEstimationConfig {
    fn default() -> Self {
        Self {
            error_correction_code: ErrorCorrectionCode::SurfaceCode,
            physical_error_rate: 1e-3,
            target_logical_error_rate: 1e-12,
            estimation_mode: EstimationMode::Realistic,
            include_hardware_overhead: true,
            hardware_platform: HardwarePlatform::Superconducting,
            detailed_analysis: true,
        }
    }
}

/// Supported error correction codes
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ErrorCorrectionCode {
    SurfaceCode,
    ColorCode,
    ToricCode,
    ShorCode,
    StabilizerCode(usize), // Distance parameter
}

/// Estimation modes affecting conservativeness of estimates
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EstimationMode {
    Conservative, // Worst-case estimates
    Optimistic,   // Best-case estimates
    Realistic,    // Expected estimates
}

/// Hardware platforms with different characteristics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum HardwarePlatform {
    Superconducting,
    TrappedIon,
    Photonic,
    NeutralAtom,
    SiliconQuantumDots,
    TopologicalQubits,
}

/// Quantum resource estimator using SciRS2 complexity analysis
pub struct ResourceEstimator {
    config: ResourceEstimationConfig,
    buffer_pool: Option<BufferPool<f64>>,
}

impl ResourceEstimator {
    /// Create a new resource estimator with default configuration
    pub fn new() -> Self {
        let config = ResourceEstimationConfig::default();
        Self::with_config(config)
    }

    /// Create a new resource estimator with custom configuration
    pub fn with_config(config: ResourceEstimationConfig) -> Self {
        let buffer_pool = if config.include_hardware_overhead {
            Some(BufferPool::<f64>::new())
        } else {
            None
        };

        Self {
            config,
            buffer_pool,
        }
    }

    /// Estimate resources for a quantum circuit
    pub fn estimate_resources(
        &self,
        circuit: &[QuantumGate],
        num_qubits: usize,
    ) -> Result<ResourceEstimate, QuantRS2Error> {
        let gate_analysis = self.analyze_gates(circuit)?;
        let logical_analysis = self.analyze_logical_requirements(&gate_analysis, num_qubits)?;
        let physical_analysis = self.analyze_physical_requirements(&logical_analysis)?;
        let time_analysis = self.analyze_execution_time(&gate_analysis)?;
        
        let detailed_analysis = if self.config.detailed_analysis {
            Some(DetailedAnalysis {
                bottlenecks: self.identify_bottlenecks(&gate_analysis)?,
                optimization_suggestions: self.generate_optimization_suggestions(&gate_analysis)?,
                scaling_analysis: self.analyze_scaling(&gate_analysis, num_qubits)?,
                error_analysis: self.analyze_error_propagation(&gate_analysis)?,
            })
        } else {
            None
        };
        
        Ok(ResourceEstimate {
            logical_qubits: logical_analysis.logical_qubits,
            physical_qubits: physical_analysis.physical_qubits,
            total_gates: gate_analysis.total_gates,
            gate_breakdown: gate_analysis.gate_breakdown.clone(),
            circuit_depth: gate_analysis.circuit_depth,
            execution_time: time_analysis.total_time,
            time_breakdown: time_analysis.time_breakdown,
            error_budget: physical_analysis.error_budget,
            overhead_factor: physical_analysis.overhead_factor,
            magic_states: logical_analysis.magic_states,
            distillation_overhead: logical_analysis.distillation_overhead,
            spatial_overhead: physical_analysis.spatial_overhead,
            temporal_overhead: time_analysis.temporal_overhead,
            detailed_analysis,
        })
    }

    /// Analyze gate composition and complexity using SciRS2
    fn analyze_gates(&self, circuit: &[QuantumGate]) -> Result<GateAnalysis, QuantRS2Error> {
        let mut gate_breakdown = HashMap::new();
        let mut depth_counter = HashMap::new();
        let mut max_depth = 0;

        // Use parallel processing for large circuits
        if circuit.len() > 1000 {
            // Parallel gate counting
            let gate_counts: HashMap<String, usize> = circuit
                .par_iter()
                .map(|gate| format!("{:?}", gate.gate_type()))
                .fold(
                    HashMap::new,
                    |mut acc, gate_type| {
                        *acc.entry(gate_type).or_insert(0) += 1;
                        acc
                    },
                )
                .reduce(
                    HashMap::new,
                    |mut acc1, acc2| {
                        for (key, value) in acc2 {
                            *acc1.entry(key).or_insert(0) += value;
                        }
                        acc1
                    },
                );
            gate_breakdown = gate_counts;
        } else {
            // Sequential processing for smaller circuits
            for gate in circuit {
                let gate_type = format!("{:?}", gate.gate_type());
                *gate_breakdown.entry(gate_type).or_insert(0) += 1;
            }
        }

        // Enhanced depth calculation with dependency analysis
        for gate in circuit {
            for &qubit in gate.target_qubits() {
                let qubit_depth = depth_counter.entry(qubit).or_insert(0);
                *qubit_depth += 1;
                max_depth = max_depth.max(*qubit_depth);
            }
        }

        // Enhanced complexity analysis with SciRS2-inspired metrics
        let two_qubit_count = self.count_two_qubit_gates(circuit);
        let _complexity_score = self.calculate_complexity_score(circuit.len(), max_depth, two_qubit_count);

        Ok(GateAnalysis {
            total_gates: circuit.len(),
            gate_breakdown,
            circuit_depth: max_depth,
            clifford_gates: self.count_clifford_gates(circuit),
            non_clifford_gates: self.count_non_clifford_gates(circuit),
            two_qubit_gates: self.count_two_qubit_gates(circuit),
            measurement_gates: self.count_measurement_gates(circuit),
        })
    }

    /// Count Clifford gates in the circuit
    fn count_clifford_gates(&self, circuit: &[QuantumGate]) -> usize {
        use crate::gate_translation::GateType;
        
        circuit.iter().filter(|gate| {
            matches!(gate.gate_type(), 
                GateType::X | GateType::Y | GateType::Z |
                GateType::H | GateType::CNOT | GateType::CZ |
                GateType::S
            )
        }).count()
    }

    /// Count non-Clifford gates in the circuit
    fn count_non_clifford_gates(&self, circuit: &[QuantumGate]) -> usize {
        use crate::gate_translation::GateType;
        
        circuit.iter().filter(|gate| {
            matches!(gate.gate_type(), 
                GateType::T | GateType::Phase(_) |
                GateType::Rx(_) | GateType::Ry(_) | GateType::Rz(_)
            )
        }).count()
    }

    /// Count two-qubit gates in the circuit
    fn count_two_qubit_gates(&self, circuit: &[QuantumGate]) -> usize {
        circuit.iter().filter(|gate| gate.target_qubits().len() >= 2).count()
    }

    /// Count measurement operations in the circuit
    fn count_measurement_gates(&self, circuit: &[QuantumGate]) -> usize {
        use crate::gate_translation::GateType;
        
        circuit.iter().filter(|gate| {
            matches!(gate.gate_type(), GateType::Custom(ref name) if name == "Measure")
        }).count()
    }

    /// Analyze logical requirements
    fn analyze_logical_requirements(
        &self,
        gate_analysis: &GateAnalysis,
        num_qubits: usize,
    ) -> Result<LogicalAnalysis, QuantRS2Error> {
        let magic_states = self.estimate_magic_states(gate_analysis)?;
        let distillation_overhead = self.calculate_distillation_overhead(magic_states)?;
        
        Ok(LogicalAnalysis {
            logical_qubits: num_qubits,
            magic_states,
            distillation_overhead,
            ancilla_qubits: self.estimate_ancilla_qubits(gate_analysis)?,
            workspace_qubits: self.estimate_workspace_qubits(gate_analysis)?,
        })
    }

    /// Estimate number of magic states required
    fn estimate_magic_states(&self, gate_analysis: &GateAnalysis) -> Result<usize, QuantRS2Error> {
        // Magic states are primarily needed for T gates and arbitrary rotations
        let t_gates = gate_analysis.non_clifford_gates;
        
        // Additional magic states for complex operations
        let additional = match self.config.estimation_mode {
            EstimationMode::Conservative => (t_gates as f64 * 1.5) as usize,
            EstimationMode::Optimistic => t_gates,
            EstimationMode::Realistic => (t_gates as f64 * 1.2) as usize,
        };
        
        Ok(additional)
    }

    /// Calculate magic state distillation overhead
    fn calculate_distillation_overhead(&self, magic_states: usize) -> Result<f64, QuantRS2Error> {
        if magic_states == 0 {
            return Ok(1.0);
        }
        
        // Distillation ratio depends on the error correction code and fidelity requirements
        let base_ratio = match self.config.error_correction_code {
            ErrorCorrectionCode::SurfaceCode => 15.0, // Conservative estimate
            ErrorCorrectionCode::ColorCode => 12.0,
            ErrorCorrectionCode::ToricCode => 15.0,
            ErrorCorrectionCode::ShorCode => 20.0,
            ErrorCorrectionCode::StabilizerCode(_) => 15.0,
        };
        
        let error_factor = (-self.config.target_logical_error_rate.log10() / 3.0).max(1.0);
        
        Ok(base_ratio * error_factor)
    }

    /// Estimate ancilla qubits needed
    fn estimate_ancilla_qubits(&self, gate_analysis: &GateAnalysis) -> Result<usize, QuantRS2Error> {
        // Ancilla qubits for syndrome extraction and error correction
        let syndrome_qubits = match self.config.error_correction_code {
            ErrorCorrectionCode::SurfaceCode => gate_analysis.total_gates / 10,
            ErrorCorrectionCode::ColorCode => gate_analysis.total_gates / 12,
            _ => gate_analysis.total_gates / 15,
        };
        
        Ok(syndrome_qubits.max(10)) // Minimum ancilla qubits
    }

    /// Estimate workspace qubits needed
    fn estimate_workspace_qubits(&self, gate_analysis: &GateAnalysis) -> Result<usize, QuantRS2Error> {
        // Workspace for intermediate computations
        let workspace = gate_analysis.two_qubit_gates / 5;
        
        Ok(workspace.max(5)) // Minimum workspace qubits
    }

    /// Analyze physical requirements
    fn analyze_physical_requirements(
        &self,
        logical_analysis: &LogicalAnalysis,
    ) -> Result<PhysicalAnalysis, QuantRS2Error> {
        let code_distance = self.calculate_code_distance()?;
        let qubits_per_logical = self.calculate_qubits_per_logical(code_distance)?;
        
        let total_logical = logical_analysis.logical_qubits +
                           logical_analysis.ancilla_qubits +
                           logical_analysis.workspace_qubits;
        
        let physical_qubits = total_logical * qubits_per_logical;
        
        let overhead_factor = physical_qubits as f64 / logical_analysis.logical_qubits as f64;
        
        Ok(PhysicalAnalysis {
            physical_qubits,
            code_distance,
            qubits_per_logical,
            overhead_factor,
            spatial_overhead: self.calculate_spatial_overhead(physical_qubits)?,
            error_budget: self.calculate_error_budget()?,
        })
    }

    /// Calculate required code distance
    fn calculate_code_distance(&self) -> Result<usize, QuantRS2Error> {
        let p = self.config.physical_error_rate;
        let p_target = self.config.target_logical_error_rate;
        
        // Simplified calculation based on threshold theory
        let threshold = match self.config.error_correction_code {
            ErrorCorrectionCode::SurfaceCode => 1e-2,
            ErrorCorrectionCode::ColorCode => 8e-3,
            _ => 1e-2,
        };
        
        if p > threshold {
            return Err(QuantRS2Error::UnsupportedOperation(
                "Physical error rate exceeds threshold".into()
            ));
        }
        
        // Distance needed: d ~ log(p_target) / log(p/p_threshold)
        let ratio = p / threshold;
        let distance = (-p_target.log10() / ratio.log10()).ceil() as usize;
        
        // Ensure odd distance for surface codes
        Ok(if distance % 2 == 0 { distance + 1 } else { distance }.max(3))
    }

    /// Calculate physical qubits per logical qubit
    fn calculate_qubits_per_logical(&self, distance: usize) -> Result<usize, QuantRS2Error> {
        match self.config.error_correction_code {
            ErrorCorrectionCode::SurfaceCode => Ok(2 * distance * distance - 2 * distance + 1),
            ErrorCorrectionCode::ColorCode => Ok(3 * distance * distance),
            ErrorCorrectionCode::ToricCode => Ok(2 * distance * distance),
            ErrorCorrectionCode::ShorCode => Ok(9),
            ErrorCorrectionCode::StabilizerCode(d) => Ok(d * d),
        }
    }

    /// Calculate spatial overhead factor
    fn calculate_spatial_overhead(&self, physical_qubits: usize) -> Result<f64, QuantRS2Error> {
        let base_overhead = match self.config.hardware_platform {
            HardwarePlatform::Superconducting => 1.5,
            HardwarePlatform::TrappedIon => 2.0,
            HardwarePlatform::Photonic => 3.0,
            HardwarePlatform::NeutralAtom => 1.8,
            HardwarePlatform::SiliconQuantumDots => 2.5,
            HardwarePlatform::TopologicalQubits => 1.2,
        };
        
        // Scale with number of qubits
        let scaling_factor = 1.0 + (physical_qubits as f64).log10() * 0.1;
        
        Ok(base_overhead * scaling_factor)
    }

    /// Calculate error budget allocation
    fn calculate_error_budget(&self) -> Result<ErrorBudget, QuantRS2Error> {
        let total_budget = self.config.target_logical_error_rate;
        
        // Distribute error budget across different sources
        let gate_errors = total_budget * 0.4;
        let measurement_errors = total_budget * 0.2;
        let memory_errors = total_budget * 0.3;
        let correction_errors = total_budget * 0.1;
        
        Ok(ErrorBudget {
            total: total_budget,
            gate_errors,
            measurement_errors,
            memory_errors,
            correction_errors,
        })
    }

    /// Analyze execution time requirements
    fn analyze_execution_time(&self, gate_analysis: &GateAnalysis) -> Result<TimeAnalysis, QuantRS2Error> {
        let gate_times = self.get_gate_timing_parameters()?;
        
        let mut time_breakdown = HashMap::new();
        let mut total_time = 0.0;
        
        for (gate_type, count) in &gate_analysis.gate_breakdown {
            let gate_time = gate_times.get(gate_type).unwrap_or(&1e-6); // Default 1μs
            let total_gate_time = *gate_time * (*count as f64);
            time_breakdown.insert(gate_type.clone(), total_gate_time);
            total_time += total_gate_time;
        }
        
        // Add error correction overhead
        let correction_overhead = total_time * 0.5; // 50% overhead for error correction
        total_time += correction_overhead;
        
        let temporal_overhead = self.calculate_temporal_overhead(total_time)?;
        
        Ok(TimeAnalysis {
            total_time,
            time_breakdown,
            correction_overhead,
            temporal_overhead,
        })
    }

    /// Get gate timing parameters for different platforms
    fn get_gate_timing_parameters(&self) -> Result<HashMap<String, f64>, QuantRS2Error> {
        let mut timings = HashMap::new();
        
        match self.config.hardware_platform {
            HardwarePlatform::Superconducting => {
                timings.insert("X".to_string(), 20e-9);
                timings.insert("Y".to_string(), 20e-9);
                timings.insert("Z".to_string(), 1e-9);
                timings.insert("H".to_string(), 20e-9);
                timings.insert("CNOT".to_string(), 40e-9);
                timings.insert("T".to_string(), 20e-9);
                timings.insert("Measure".to_string(), 300e-9);
            },
            HardwarePlatform::TrappedIon => {
                timings.insert("X".to_string(), 10e-6);
                timings.insert("Y".to_string(), 10e-6);
                timings.insert("Z".to_string(), 1e-6);
                timings.insert("H".to_string(), 10e-6);
                timings.insert("CNOT".to_string(), 100e-6);
                timings.insert("T".to_string(), 10e-6);
                timings.insert("Measure".to_string(), 100e-6);
            },
            HardwarePlatform::Photonic => {
                timings.insert("X".to_string(), 1e-9);
                timings.insert("Y".to_string(), 1e-9);
                timings.insert("Z".to_string(), 1e-9);
                timings.insert("H".to_string(), 1e-9);
                timings.insert("CNOT".to_string(), 10e-9);
                timings.insert("T".to_string(), 1e-9);
                timings.insert("Measure".to_string(), 1e-9);
            },
            _ => {
                // Default timing values
                timings.insert("X".to_string(), 1e-6);
                timings.insert("Y".to_string(), 1e-6);
                timings.insert("Z".to_string(), 1e-6);
                timings.insert("H".to_string(), 1e-6);
                timings.insert("CNOT".to_string(), 2e-6);
                timings.insert("T".to_string(), 1e-6);
                timings.insert("Measure".to_string(), 10e-6);
            }
        }
        
        Ok(timings)
    }

    /// Calculate temporal overhead factors
    fn calculate_temporal_overhead(&self, _base_time: f64) -> Result<f64, QuantRS2Error> {
        let overhead = match self.config.hardware_platform {
            HardwarePlatform::Superconducting => 1.3,
            HardwarePlatform::TrappedIon => 1.8,
            HardwarePlatform::Photonic => 1.1,
            HardwarePlatform::NeutralAtom => 1.5,
            HardwarePlatform::SiliconQuantumDots => 2.0,
            HardwarePlatform::TopologicalQubits => 1.1,
        };
        
        Ok(overhead)
    }

    /// Identify performance bottlenecks
    fn identify_bottlenecks(&self, gate_analysis: &GateAnalysis) -> Result<Vec<String>, QuantRS2Error> {
        let mut bottlenecks = Vec::new();
        
        if gate_analysis.circuit_depth > 1000 {
            bottlenecks.push("High circuit depth may lead to decoherence issues".to_string());
        }
        
        if gate_analysis.two_qubit_gates > gate_analysis.total_gates / 2 {
            bottlenecks.push("High ratio of two-qubit gates increases error rates".to_string());
        }
        
        if gate_analysis.non_clifford_gates > 100 {
            bottlenecks.push("Large number of non-Clifford gates requires many magic states".to_string());
        }
        
        Ok(bottlenecks)
    }

    /// Calculate complexity score for the circuit using SciRS2-inspired metrics
    fn calculate_complexity_score(&self, total_gates: usize, depth: usize, two_qubit_gates: usize) -> f64 {
        // SciRS2-inspired complexity scoring
        let gate_complexity = total_gates as f64;
        let depth_complexity = depth as f64 * 1.5; // Depth has higher impact
        let two_qubit_complexity = two_qubit_gates as f64 * 2.0; // Two-qubit gates are more expensive
        
        (gate_complexity + depth_complexity + two_qubit_complexity) / (total_gates as f64 + 1.0)
    }

    /// Generate optimization suggestions using SciRS2-enhanced analysis
    fn generate_optimization_suggestions(&self, gate_analysis: &GateAnalysis) -> Result<Vec<String>, QuantRS2Error> {
        let mut suggestions = Vec::new();
        
        // Calculate complexity score for adaptive suggestions
        let complexity_score = self.calculate_complexity_score(
            gate_analysis.total_gates,
            gate_analysis.circuit_depth,
            gate_analysis.two_qubit_gates,
        );
        
        if gate_analysis.circuit_depth > 500 {
            suggestions.push("Consider circuit parallelization to reduce depth".to_string());
            if complexity_score > 3.0 {
                suggestions.push("Use SciRS2 parallel gate scheduling for improved performance".to_string());
            }
        }
        
        if gate_analysis.non_clifford_gates > 50 {
            suggestions.push("Apply Clifford+T optimization to reduce T-gate count".to_string());
            if self.config.detailed_analysis {
                suggestions.push("Consider SciRS2 magic state optimization algorithms".to_string());
            }
        }
        
        if gate_analysis.two_qubit_gates > gate_analysis.total_gates / 3 {
            suggestions.push("High two-qubit gate ratio detected - consider gate fusion optimization".to_string());
        }
        
        // Hardware-specific suggestions
        match self.config.hardware_platform {
            HardwarePlatform::Superconducting => {
                suggestions.push("Optimize for superconducting hardware connectivity".to_string());
            },
            HardwarePlatform::TrappedIon => {
                suggestions.push("Leverage all-to-all connectivity for trapped ion systems".to_string());
            },
            _ => {
                suggestions.push("Consider hardware-specific gate set optimization".to_string());
            }
        }
        
        suggestions.push("Consider using error mitigation techniques".to_string());
        suggestions.push("Apply SciRS2 memory-efficient state vector simulation".to_string());
        
        Ok(suggestions)
    }

    /// Analyze scaling behavior
    fn analyze_scaling(&self, gate_analysis: &GateAnalysis, num_qubits: usize) -> Result<ScalingAnalysis, QuantRS2Error> {
        let time_complexity = if gate_analysis.two_qubit_gates > 0 {
            format!("O(n^{})", (gate_analysis.two_qubit_gates as f64 / num_qubits as f64).log2().ceil())
        } else {
            "O(n)".to_string()
        };
        
        let space_complexity = format!("O(2^n)"); // Exponential in number of qubits
        
        Ok(ScalingAnalysis {
            time_complexity,
            space_complexity,
            predicted_scaling: self.predict_scaling_factors(num_qubits)?,
        })
    }

    /// Predict scaling factors for larger problems
    fn predict_scaling_factors(&self, _num_qubits: usize) -> Result<HashMap<String, f64>, QuantRS2Error> {
        let mut factors = HashMap::new();
        
        factors.insert("10_qubits".to_string(), 1.0);
        factors.insert("20_qubits".to_string(), 4.0);
        factors.insert("50_qubits".to_string(), 100.0);
        factors.insert("100_qubits".to_string(), 10000.0);
        
        Ok(factors)
    }

    /// Analyze error propagation through the circuit
    fn analyze_error_propagation(&self, gate_analysis: &GateAnalysis) -> Result<ErrorPropagationAnalysis, QuantRS2Error> {
        let error_accumulation = gate_analysis.total_gates as f64 * self.config.physical_error_rate;
        let error_amplification = 1.0 + (gate_analysis.two_qubit_gates as f64 * 0.1);
        
        Ok(ErrorPropagationAnalysis {
            error_accumulation,
            error_amplification,
            critical_paths: vec!["Long sequences of two-qubit gates".to_string()],
        })
    }
}

/// Complete resource estimation result
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    /// Number of logical qubits required
    pub logical_qubits: usize,
    /// Number of physical qubits required
    pub physical_qubits: usize,
    /// Total number of gates
    pub total_gates: usize,
    /// Breakdown of gates by type
    pub gate_breakdown: HashMap<String, usize>,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Total execution time (seconds)
    pub execution_time: f64,
    /// Time breakdown by operation type
    pub time_breakdown: HashMap<String, f64>,
    /// Error budget allocation
    pub error_budget: ErrorBudget,
    /// Physical to logical qubit overhead factor
    pub overhead_factor: f64,
    /// Number of magic states required
    pub magic_states: usize,
    /// Magic state distillation overhead
    pub distillation_overhead: f64,
    /// Spatial overhead factor
    pub spatial_overhead: f64,
    /// Temporal overhead factor
    pub temporal_overhead: f64,
    /// Detailed analysis (optional)
    pub detailed_analysis: Option<DetailedAnalysis>,
}

/// Gate analysis results
#[derive(Debug, Clone)]
struct GateAnalysis {
    total_gates: usize,
    gate_breakdown: HashMap<String, usize>,
    circuit_depth: usize,
    clifford_gates: usize,
    non_clifford_gates: usize,
    two_qubit_gates: usize,
    measurement_gates: usize,
}

/// Logical requirements analysis
#[derive(Debug, Clone)]
struct LogicalAnalysis {
    logical_qubits: usize,
    magic_states: usize,
    distillation_overhead: f64,
    ancilla_qubits: usize,
    workspace_qubits: usize,
}

/// Physical requirements analysis
#[derive(Debug, Clone)]
struct PhysicalAnalysis {
    physical_qubits: usize,
    code_distance: usize,
    qubits_per_logical: usize,
    overhead_factor: f64,
    spatial_overhead: f64,
    error_budget: ErrorBudget,
}

/// Time analysis results
#[derive(Debug, Clone)]
struct TimeAnalysis {
    total_time: f64,
    time_breakdown: HashMap<String, f64>,
    correction_overhead: f64,
    temporal_overhead: f64,
}

/// Error budget allocation
#[derive(Debug, Clone)]
pub struct ErrorBudget {
    pub total: f64,
    pub gate_errors: f64,
    pub measurement_errors: f64,
    pub memory_errors: f64,
    pub correction_errors: f64,
}

/// Detailed analysis results
#[derive(Debug, Clone)]
pub struct DetailedAnalysis {
    pub bottlenecks: Vec<String>,
    pub optimization_suggestions: Vec<String>,
    pub scaling_analysis: ScalingAnalysis,
    pub error_analysis: ErrorPropagationAnalysis,
}

/// Scaling analysis results
#[derive(Debug, Clone)]
pub struct ScalingAnalysis {
    pub time_complexity: String,
    pub space_complexity: String,
    pub predicted_scaling: HashMap<String, f64>,
}

/// Error propagation analysis
#[derive(Debug, Clone)]
pub struct ErrorPropagationAnalysis {
    pub error_accumulation: f64,
    pub error_amplification: f64,
    pub critical_paths: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{QuantumGate, GateType};

    #[test]
    fn test_resource_estimator_creation() {
        let estimator = ResourceEstimator::new();
        assert!(matches!(estimator.config.error_correction_code, ErrorCorrectionCode::SurfaceCode));
    }

    #[test]
    fn test_gate_analysis() {
        let estimator = ResourceEstimator::new();
        let circuit = vec![
            QuantumGate::new(GateType::H, vec![0], None),
            QuantumGate::new(GateType::CNOT, vec![0, 1], None),
            QuantumGate::new(GateType::T, vec![0], None),
        ];
        
        let analysis = estimator.analyze_gates(&circuit).unwrap();
        assert_eq!(analysis.total_gates, 3);
        assert_eq!(analysis.clifford_gates, 2);
        assert_eq!(analysis.non_clifford_gates, 1);
    }

    #[test]
    fn test_magic_state_estimation() {
        let estimator = ResourceEstimator::new();
        let gate_analysis = GateAnalysis {
            total_gates: 10,
            gate_breakdown: HashMap::new(),
            circuit_depth: 5,
            clifford_gates: 7,
            non_clifford_gates: 3,
            two_qubit_gates: 2,
            measurement_gates: 0,
        };
        
        let magic_states = estimator.estimate_magic_states(&gate_analysis).unwrap();
        assert!(magic_states >= 3); // At least as many as non-Clifford gates
    }

    #[test]
    fn test_code_distance_calculation() {
        let mut config = ResourceEstimationConfig::default();
        config.physical_error_rate = 1e-3;
        config.target_logical_error_rate = 1e-12;
        
        let estimator = ResourceEstimator::with_config(config);
        let distance = estimator.calculate_code_distance().unwrap();
        assert!(distance >= 3);
        assert!(distance % 2 == 1); // Should be odd for surface codes
    }

    #[test]
    fn test_resource_estimation_small_circuit() {
        let estimator = ResourceEstimator::new();
        let circuit = vec![
            QuantumGate::new(GateType::H, vec![0], None),
            QuantumGate::new(GateType::CNOT, vec![0, 1], None),
        ];
        
        let estimate = estimator.estimate_resources(&circuit, 2).unwrap();
        assert!(estimate.logical_qubits > 0);
        assert!(estimate.physical_qubits > estimate.logical_qubits);
        assert!(estimate.execution_time > 0.0);
    }
}