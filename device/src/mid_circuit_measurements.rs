//! Mid-circuit measurement support for quantum hardware backends
//!
//! This module provides comprehensive support for executing circuits with mid-circuit
//! measurements on various quantum hardware platforms, including validation,
//! optimization, and hardware-specific adaptations.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use quantrs2_circuit::{
    classical::{
        ClassicalBit, ClassicalCircuit, ClassicalCondition, ClassicalOp, ClassicalRegister,
        ClassicalValue, ComparisonOp, MeasureOp,
    },
    measurement::{CircuitOp, FeedForward, Measurement, MeasurementCircuit},
    prelude::*,
};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

// SciRS2 dependencies (feature-gated for availability)
#[cfg(feature = "scirs2")]
use scirs2_graph::{betweenness_centrality, shortest_path, Graph};
#[cfg(feature = "scirs2")]
use scirs2_linalg::{eig, matrix_norm, LinalgResult};
#[cfg(feature = "scirs2")]
use scirs2_optimize::{minimize, OptimizeResult};
#[cfg(feature = "scirs2")]
use scirs2_stats::{mean, pearsonr, std, ttest_1samp, Alternative};

// Fallback implementations when SciRS2 is not available
#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2 {
    pub fn mean(_data: &ndarray::ArrayView1<f64>) -> Result<f64, String> {
        Ok(0.0) // Placeholder
    }
    pub fn std(_data: &ndarray::ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
        Ok(1.0) // Placeholder
    }
    pub fn pearsonr(
        _x: &ndarray::ArrayView1<f64>,
        _y: &ndarray::ArrayView1<f64>,
        _alternative: &str,
    ) -> Result<(f64, f64), String> {
        Ok((0.0, 0.5)) // Placeholder
    }
}

#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::*;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::{
    backend_traits::{query_backend_capabilities, BackendCapabilities, BackendFeatures},
    calibration::{CalibrationManager, DeviceCalibration},
    noise_model::CalibrationNoiseModel,
    topology::HardwareTopology,
    translation::{GateTranslator, HardwareBackend},
    CircuitResult, DeviceError, DeviceResult,
};

/// Mid-circuit measurement execution configuration
#[derive(Debug, Clone)]
pub struct MidCircuitConfig {
    /// Maximum allowed measurement latency (microseconds)
    pub max_measurement_latency: f64,
    /// Enable real-time classical processing
    pub enable_realtime_processing: bool,
    /// Buffer size for measurement results
    pub measurement_buffer_size: usize,
    /// Timeout for classical condition evaluation (microseconds)
    pub classical_timeout: f64,
    /// Enable measurement error mitigation
    pub enable_measurement_mitigation: bool,
    /// Parallel measurement execution
    pub enable_parallel_measurements: bool,
    /// Hardware-specific optimizations
    pub hardware_optimizations: HardwareOptimizations,
    /// Validation settings
    pub validation_config: ValidationConfig,
}

/// Hardware-specific optimizations for mid-circuit measurements
#[derive(Debug, Clone)]
pub struct HardwareOptimizations {
    /// Batch measurement operations when possible
    pub batch_measurements: bool,
    /// Optimize measurement scheduling
    pub optimize_scheduling: bool,
    /// Use hardware-native measurement protocols
    pub use_native_protocols: bool,
    /// Enable measurement compression
    pub measurement_compression: bool,
    /// Pre-compile classical conditions
    pub precompile_conditions: bool,
}

/// Validation configuration for mid-circuit measurements
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Validate backend measurement capabilities
    pub validate_capabilities: bool,
    /// Check measurement timing constraints
    pub check_timing_constraints: bool,
    /// Validate classical register sizes
    pub validate_register_sizes: bool,
    /// Check for measurement conflicts
    pub check_measurement_conflicts: bool,
    /// Validate feed-forward operations
    pub validate_feedforward: bool,
}

impl Default for MidCircuitConfig {
    fn default() -> Self {
        Self {
            max_measurement_latency: 1000.0, // 1ms
            enable_realtime_processing: true,
            measurement_buffer_size: 1024,
            classical_timeout: 100.0, // 100μs
            enable_measurement_mitigation: true,
            enable_parallel_measurements: true,
            hardware_optimizations: HardwareOptimizations {
                batch_measurements: true,
                optimize_scheduling: true,
                use_native_protocols: true,
                measurement_compression: false,
                precompile_conditions: true,
            },
            validation_config: ValidationConfig {
                validate_capabilities: true,
                check_timing_constraints: true,
                validate_register_sizes: true,
                check_measurement_conflicts: true,
                validate_feedforward: true,
            },
        }
    }
}

/// Mid-circuit measurement execution result
#[derive(Debug, Clone)]
pub struct MidCircuitExecutionResult {
    /// Final quantum measurement results
    pub final_measurements: HashMap<String, usize>,
    /// Classical register states
    pub classical_registers: HashMap<String, Vec<u8>>,
    /// Mid-circuit measurement history
    pub measurement_history: Vec<MeasurementEvent>,
    /// Execution statistics
    pub execution_stats: ExecutionStats,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Error analysis
    pub error_analysis: Option<ErrorAnalysis>,
}

/// Individual measurement event during execution
#[derive(Debug, Clone)]
pub struct MeasurementEvent {
    /// Timestamp (microseconds from start)
    pub timestamp: f64,
    /// Measured qubit
    pub qubit: QubitId,
    /// Measurement result (0 or 1)
    pub result: u8,
    /// Classical bit/register where result was stored
    pub storage_location: StorageLocation,
    /// Measurement latency (microseconds)
    pub latency: f64,
    /// Confidence/fidelity of measurement
    pub confidence: f64,
}

/// Location where measurement result is stored
#[derive(Debug, Clone)]
pub enum StorageLocation {
    /// Classical bit index
    ClassicalBit(usize),
    /// Classical register and bit index
    ClassicalRegister(String, usize),
    /// Temporary buffer
    Buffer(usize),
}

/// Execution statistics for mid-circuit measurement
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    /// Total execution time
    pub total_execution_time: Duration,
    /// Time spent on quantum operations
    pub quantum_time: Duration,
    /// Time spent on measurements
    pub measurement_time: Duration,
    /// Time spent on classical processing
    pub classical_time: Duration,
    /// Number of mid-circuit measurements
    pub num_measurements: usize,
    /// Number of conditional operations
    pub num_conditional_ops: usize,
    /// Average measurement latency
    pub avg_measurement_latency: f64,
    /// Maximum measurement latency
    pub max_measurement_latency: f64,
}

/// Performance metrics for mid-circuit measurements
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Measurement success rate
    pub measurement_success_rate: f64,
    /// Classical processing efficiency
    pub classical_efficiency: f64,
    /// Overall circuit fidelity
    pub circuit_fidelity: f64,
    /// Measurement error rate
    pub measurement_error_rate: f64,
    /// Timing overhead compared to no measurements
    pub timing_overhead: f64,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// Quantum resource usage (0-1)
    pub quantum_utilization: f64,
    /// Classical resource usage (0-1)
    pub classical_utilization: f64,
    /// Memory usage for classical data
    pub memory_usage: usize,
    /// Communication overhead
    pub communication_overhead: f64,
}

/// Error analysis for mid-circuit measurements
#[derive(Debug, Clone)]
pub struct ErrorAnalysis {
    /// Measurement errors by qubit
    pub measurement_errors: HashMap<QubitId, MeasurementErrorStats>,
    /// Classical processing errors
    pub classical_errors: Vec<ClassicalError>,
    /// Timing violations
    pub timing_violations: Vec<TimingViolation>,
    /// Correlation analysis
    pub error_correlations: Array2<f64>,
}

/// Measurement error statistics
#[derive(Debug, Clone)]
pub struct MeasurementErrorStats {
    /// Readout error rate
    pub readout_error_rate: f64,
    /// State preparation and measurement (SPAM) error
    pub spam_error: f64,
    /// Thermal relaxation during measurement
    pub thermal_relaxation: f64,
    /// Dephasing during measurement
    pub dephasing: f64,
}

/// Classical processing error
#[derive(Debug, Clone)]
pub struct ClassicalError {
    /// Error type
    pub error_type: ClassicalErrorType,
    /// Timestamp when error occurred
    pub timestamp: f64,
    /// Error description
    pub description: String,
    /// Affected operations
    pub affected_operations: Vec<usize>,
}

/// Types of classical errors
#[derive(Debug, Clone, PartialEq)]
pub enum ClassicalErrorType {
    /// Timeout in classical condition evaluation
    Timeout,
    /// Invalid register access
    InvalidRegisterAccess,
    /// Condition evaluation error
    ConditionEvaluationError,
    /// Buffer overflow
    BufferOverflow,
    /// Communication error
    CommunicationError,
}

/// Timing constraint violation
#[derive(Debug, Clone)]
pub struct TimingViolation {
    /// Operation that violated timing
    pub operation_index: usize,
    /// Expected timing (microseconds)
    pub expected_timing: f64,
    /// Actual timing (microseconds)
    pub actual_timing: f64,
    /// Violation severity (0-1)
    pub severity: f64,
}

/// Device capabilities for mid-circuit measurements
#[derive(Debug, Clone)]
pub struct MidCircuitCapabilities {
    /// Maximum number of mid-circuit measurements
    pub max_measurements: Option<usize>,
    /// Supported measurement types
    pub supported_measurement_types: Vec<MeasurementType>,
    /// Classical register capacity
    pub classical_register_capacity: usize,
    /// Maximum classical processing time
    pub max_classical_processing_time: f64,
    /// Real-time feedback support
    pub realtime_feedback: bool,
    /// Parallel measurement support
    pub parallel_measurements: bool,
    /// Native measurement protocols
    pub native_protocols: Vec<String>,
    /// Timing constraints
    pub timing_constraints: TimingConstraints,
}

/// Supported measurement types
#[derive(Debug, Clone, PartialEq)]
pub enum MeasurementType {
    /// Standard Z-basis measurement
    ZBasis,
    /// X-basis measurement
    XBasis,
    /// Y-basis measurement
    YBasis,
    /// Custom Pauli measurement
    Pauli(String),
    /// Joint measurement of multiple qubits
    Joint,
    /// Non-destructive measurement
    NonDestructive,
}

/// Timing constraints for mid-circuit measurements
#[derive(Debug, Clone)]
pub struct TimingConstraints {
    /// Minimum time between measurements (nanoseconds)
    pub min_measurement_spacing: f64,
    /// Maximum measurement duration (nanoseconds)
    pub max_measurement_duration: f64,
    /// Classical processing deadline (nanoseconds)
    pub classical_deadline: f64,
    /// Coherence time limits
    pub coherence_limits: HashMap<QubitId, f64>,
}

/// Mid-circuit measurement executor
pub struct MidCircuitExecutor {
    config: MidCircuitConfig,
    calibration_manager: CalibrationManager,
    capabilities: Option<MidCircuitCapabilities>,
    gate_translator: GateTranslator,
}

impl MidCircuitExecutor {
    /// Create a new mid-circuit measurement executor
    pub fn new(config: MidCircuitConfig, calibration_manager: CalibrationManager) -> Self {
        Self {
            config,
            calibration_manager,
            capabilities: None,
            gate_translator: GateTranslator::new(),
        }
    }

    /// Query and cache device capabilities for mid-circuit measurements
    pub fn query_capabilities(
        &mut self,
        backend: HardwareBackend,
        device_id: &str,
    ) -> DeviceResult<&MidCircuitCapabilities> {
        let backend_caps = query_backend_capabilities(backend);

        let capabilities = MidCircuitCapabilities {
            max_measurements: backend_caps.features.max_mid_circuit_measurements,
            supported_measurement_types: self.get_supported_measurement_types(backend)?,
            classical_register_capacity: backend_caps.features.classical_register_size,
            max_classical_processing_time: 1000.0, // 1ms default
            realtime_feedback: backend_caps.features.supports_real_time_feedback,
            parallel_measurements: backend_caps.features.supports_parallel_execution,
            native_protocols: self.get_native_protocols(backend),
            timing_constraints: self.get_timing_constraints(backend, device_id)?,
        };

        self.capabilities = Some(capabilities);
        Ok(self.capabilities.as_ref().unwrap())
    }

    /// Validate a measurement circuit against device capabilities
    pub fn validate_circuit<const N: usize>(
        &self,
        circuit: &MeasurementCircuit<N>,
        device_id: &str,
    ) -> DeviceResult<ValidationResult> {
        let mut validation_result = ValidationResult {
            is_valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
            recommendations: Vec::new(),
        };

        if !self.config.validation_config.validate_capabilities {
            return Ok(validation_result);
        }

        let capabilities = self
            .capabilities
            .as_ref()
            .ok_or_else(|| DeviceError::APIError("Capabilities not queried".into()))?;

        // Check measurement count limits
        let measurement_count = circuit
            .operations()
            .iter()
            .filter(|op| matches!(op, CircuitOp::Measure(_)))
            .count();

        if let Some(max_measurements) = capabilities.max_measurements {
            if measurement_count > max_measurements {
                validation_result.errors.push(format!(
                    "Circuit requires {} measurements but device supports maximum {}",
                    measurement_count, max_measurements
                ));
                validation_result.is_valid = false;
            }
        }

        // Validate classical register usage
        if self.config.validation_config.validate_register_sizes {
            self.validate_classical_registers(circuit, capabilities, &mut validation_result)?;
        }

        // Check timing constraints
        if self.config.validation_config.check_timing_constraints {
            self.validate_timing_constraints(circuit, capabilities, &mut validation_result)?;
        }

        // Validate feed-forward operations
        if self.config.validation_config.validate_feedforward {
            self.validate_feedforward_operations(circuit, capabilities, &mut validation_result)?;
        }

        // Check for measurement conflicts
        if self.config.validation_config.check_measurement_conflicts {
            self.check_measurement_conflicts(circuit, &mut validation_result)?;
        }

        Ok(validation_result)
    }

    /// Execute a circuit with mid-circuit measurements
    pub async fn execute_circuit<const N: usize>(
        &self,
        circuit: &MeasurementCircuit<N>,
        device_executor: &dyn MidCircuitDeviceExecutor,
        shots: usize,
    ) -> DeviceResult<MidCircuitExecutionResult> {
        let start_time = Instant::now();

        // Validate circuit before execution
        let validation = self.validate_circuit(circuit, device_executor.device_id())?;
        if !validation.is_valid {
            return Err(DeviceError::APIError(format!(
                "Circuit validation failed: {:?}",
                validation.errors
            )));
        }

        // Optimize circuit for hardware
        let optimized_circuit = self.optimize_for_hardware(circuit, device_executor).await?;

        // Execute with measurement tracking
        let mut measurement_history = Vec::new();
        let mut classical_registers = HashMap::new();
        let mut execution_stats = ExecutionStats {
            total_execution_time: Duration::from_millis(0),
            quantum_time: Duration::from_millis(0),
            measurement_time: Duration::from_millis(0),
            classical_time: Duration::from_millis(0),
            num_measurements: 0,
            num_conditional_ops: 0,
            avg_measurement_latency: 0.0,
            max_measurement_latency: 0.0,
        };

        // Execute the optimized circuit
        let final_measurements = self
            .execute_with_tracking(
                optimized_circuit,
                device_executor,
                shots,
                &mut measurement_history,
                &mut classical_registers,
                &mut execution_stats,
            )
            .await?;

        // Calculate performance metrics
        let performance_metrics =
            self.calculate_performance_metrics(&measurement_history, &execution_stats)?;

        // Perform error analysis
        let error_analysis = if self.config.enable_measurement_mitigation {
            Some(self.analyze_measurement_errors(&measurement_history, circuit)?)
        } else {
            None
        };

        execution_stats.total_execution_time = start_time.elapsed();

        Ok(MidCircuitExecutionResult {
            final_measurements,
            classical_registers,
            measurement_history,
            execution_stats,
            performance_metrics,
            error_analysis,
        })
    }

    /// Optimize circuit for specific hardware backend
    async fn optimize_for_hardware<'a, const N: usize>(
        &self,
        circuit: &'a MeasurementCircuit<N>,
        device_executor: &dyn MidCircuitDeviceExecutor,
    ) -> DeviceResult<&'a MeasurementCircuit<N>> {
        // Since optimization methods are currently placeholders that don't modify the circuit,
        // we can just return the input reference for now
        // TODO: Implement actual optimization that creates new circuits

        if self.config.hardware_optimizations.batch_measurements {
            // self.batch_measurements(circuit)?;
        }

        if self.config.hardware_optimizations.optimize_scheduling {
            // self.optimize_measurement_scheduling(circuit)?;
        }

        if self.config.hardware_optimizations.precompile_conditions {
            // self.precompile_classical_conditions(circuit)?;
        }

        Ok(circuit)
    }

    /// Execute circuit with detailed tracking
    async fn execute_with_tracking<const N: usize>(
        &self,
        circuit: &MeasurementCircuit<N>,
        device_executor: &dyn MidCircuitDeviceExecutor,
        shots: usize,
        measurement_history: &mut Vec<MeasurementEvent>,
        classical_registers: &mut HashMap<String, Vec<u8>>,
        execution_stats: &mut ExecutionStats,
    ) -> DeviceResult<HashMap<String, usize>> {
        let mut final_measurements = HashMap::new();
        let execution_start = Instant::now();

        // Process each shot
        for shot in 0..shots {
            let shot_start = Instant::now();

            // Reset classical registers for this shot
            classical_registers.clear();

            // Execute operations sequentially
            for (op_index, operation) in circuit.operations().iter().enumerate() {
                match operation {
                    CircuitOp::Gate(gate) => {
                        let gate_start = Instant::now();
                        device_executor.execute_gate(gate.as_ref()).await?;
                        execution_stats.quantum_time += gate_start.elapsed();
                    }
                    CircuitOp::Measure(measurement) => {
                        let measurement_start = Instant::now();
                        let result = self
                            .execute_measurement(
                                measurement,
                                device_executor,
                                measurement_history,
                                execution_start.elapsed().as_micros() as f64,
                            )
                            .await?;

                        // Store result in classical register
                        self.store_measurement_result(measurement, result, classical_registers)?;

                        execution_stats.num_measurements += 1;
                        let latency = measurement_start.elapsed().as_micros() as f64;
                        execution_stats.measurement_time += measurement_start.elapsed();

                        if latency > execution_stats.max_measurement_latency {
                            execution_stats.max_measurement_latency = latency;
                        }
                    }
                    CircuitOp::FeedForward(feedforward) => {
                        let classical_start = Instant::now();

                        // Evaluate condition
                        let condition_met = self.evaluate_classical_condition(
                            &feedforward.condition,
                            classical_registers,
                        )?;

                        if condition_met {
                            device_executor.execute_gate(&*feedforward.gate).await?;
                            execution_stats.num_conditional_ops += 1;
                        }

                        execution_stats.classical_time += classical_start.elapsed();
                    }
                    CircuitOp::Barrier(_) => {
                        // Synchronization point - ensure all previous operations complete
                        device_executor.synchronize().await?;
                    }
                    CircuitOp::Reset(qubit) => {
                        device_executor.reset_qubit(*qubit).await?;
                    }
                }
            }

            // Final measurements for this shot
            let final_result = device_executor.measure_all().await?;
            for (qubit_str, result) in final_result {
                *final_measurements.entry(qubit_str).or_insert(0) += result;
            }
        }

        // Calculate average measurement latency
        if execution_stats.num_measurements > 0 {
            execution_stats.avg_measurement_latency = execution_stats.measurement_time.as_micros()
                as f64
                / execution_stats.num_measurements as f64;
        }

        Ok(final_measurements)
    }

    /// Execute a single measurement with tracking
    async fn execute_measurement(
        &self,
        measurement: &Measurement,
        device_executor: &dyn MidCircuitDeviceExecutor,
        measurement_history: &mut Vec<MeasurementEvent>,
        timestamp: f64,
    ) -> DeviceResult<u8> {
        let measurement_start = Instant::now();

        let result = device_executor.measure_qubit(measurement.qubit).await?;

        let latency = measurement_start.elapsed().as_micros() as f64;

        // Calculate measurement confidence based on calibration data
        let confidence = self.calculate_measurement_confidence(measurement.qubit)?;

        measurement_history.push(MeasurementEvent {
            timestamp,
            qubit: measurement.qubit,
            result,
            storage_location: StorageLocation::ClassicalBit(measurement.target_bit),
            latency,
            confidence,
        });

        Ok(result)
    }

    /// Store measurement result in classical registers
    fn store_measurement_result(
        &self,
        measurement: &Measurement,
        result: u8,
        classical_registers: &mut HashMap<String, Vec<u8>>,
    ) -> DeviceResult<()> {
        // For now, store in a default register
        let register = classical_registers
            .entry("measurements".to_string())
            .or_insert_with(|| vec![0; 64]); // 64-bit default register

        if measurement.target_bit < register.len() {
            register[measurement.target_bit] = result;
        }

        Ok(())
    }

    /// Evaluate classical condition
    fn evaluate_classical_condition(
        &self,
        condition: &ClassicalCondition,
        classical_registers: &HashMap<String, Vec<u8>>,
    ) -> DeviceResult<bool> {
        // Evaluate the classical condition using the struct fields
        match (&condition.lhs, &condition.rhs) {
            (ClassicalValue::Bit(lhs_bit), ClassicalValue::Bit(rhs_bit)) => {
                Ok(match condition.op {
                    ComparisonOp::Equal => lhs_bit == rhs_bit,
                    ComparisonOp::NotEqual => lhs_bit != rhs_bit,
                    _ => false, // Other comparisons not meaningful for bits
                })
            }
            (ClassicalValue::Register(reg_name), ClassicalValue::Integer(expected)) => {
                if let Some(register) = classical_registers.get(reg_name) {
                    // Compare first few bits with expected value
                    let actual_value = register.iter()
                        .take(8) // Take first 8 bits
                        .enumerate()
                        .fold(0u8, |acc, (i, &bit)| acc | (bit << i));
                    Ok(actual_value == *expected as u8)
                } else {
                    Ok(false)
                }
            }
            // Add other condition types as needed
            _ => Ok(false),
        }
    }

    /// Calculate measurement confidence based on calibration
    fn calculate_measurement_confidence(&self, qubit: QubitId) -> DeviceResult<f64> {
        // Use calibration data to estimate measurement fidelity
        // This is a simplified implementation
        Ok(0.99) // 99% confidence default
    }

    /// Calculate performance metrics
    fn calculate_performance_metrics(
        &self,
        measurement_history: &[MeasurementEvent],
        execution_stats: &ExecutionStats,
    ) -> DeviceResult<PerformanceMetrics> {
        let total_measurements = measurement_history.len() as f64;

        // Calculate measurement success rate (simplified)
        let high_confidence_measurements = measurement_history
            .iter()
            .filter(|event| event.confidence > 0.95)
            .count() as f64;

        let measurement_success_rate = if total_measurements > 0.0 {
            high_confidence_measurements / total_measurements
        } else {
            1.0
        };

        // Calculate timing efficiency
        let total_time = execution_stats.total_execution_time.as_micros() as f64;
        let useful_time = execution_stats.quantum_time.as_micros() as f64;
        let timing_overhead = if useful_time > 0.0 {
            (total_time - useful_time) / useful_time
        } else {
            0.0
        };

        // Resource utilization
        let resource_utilization = ResourceUtilization {
            quantum_utilization: if total_time > 0.0 {
                useful_time / total_time
            } else {
                0.0
            },
            classical_utilization: if total_time > 0.0 {
                execution_stats.classical_time.as_micros() as f64 / total_time
            } else {
                0.0
            },
            memory_usage: total_measurements as usize * 32, // Estimate 32 bytes per measurement
            communication_overhead: execution_stats.measurement_time.as_micros() as f64
                / total_time,
        };

        Ok(PerformanceMetrics {
            measurement_success_rate,
            classical_efficiency: 0.95, // Placeholder
            circuit_fidelity: measurement_success_rate * 0.98, // Estimate
            measurement_error_rate: 1.0 - measurement_success_rate,
            timing_overhead,
            resource_utilization,
        })
    }

    /// Analyze measurement errors
    fn analyze_measurement_errors<const N: usize>(
        &self,
        measurement_history: &[MeasurementEvent],
        circuit: &MeasurementCircuit<N>,
    ) -> DeviceResult<ErrorAnalysis> {
        let mut measurement_errors = HashMap::new();

        // Calculate error statistics for each qubit
        for event in measurement_history {
            let error_stats =
                measurement_errors
                    .entry(event.qubit)
                    .or_insert(MeasurementErrorStats {
                        readout_error_rate: 0.01,
                        spam_error: 0.005,
                        thermal_relaxation: 0.002,
                        dephasing: 0.003,
                    });

            // Update error statistics based on confidence
            if event.confidence < 0.95 {
                error_stats.readout_error_rate += 0.001;
            }
        }

        // Placeholder for correlation analysis
        let n_qubits = measurement_errors.len();
        let error_correlations = Array2::eye(n_qubits);

        Ok(ErrorAnalysis {
            measurement_errors,
            classical_errors: Vec::new(),
            timing_violations: Vec::new(),
            error_correlations,
        })
    }

    // Helper methods for validation and optimization

    fn get_supported_measurement_types(
        &self,
        backend: HardwareBackend,
    ) -> DeviceResult<Vec<MeasurementType>> {
        match backend {
            HardwareBackend::IBMQuantum => Ok(vec![
                MeasurementType::ZBasis,
                MeasurementType::XBasis,
                MeasurementType::YBasis,
            ]),
            HardwareBackend::IonQ => Ok(vec![
                MeasurementType::ZBasis,
                MeasurementType::XBasis,
                MeasurementType::YBasis,
                MeasurementType::Joint,
            ]),
            HardwareBackend::Rigetti => Ok(vec![
                MeasurementType::ZBasis,
                MeasurementType::NonDestructive,
            ]),
            _ => Ok(vec![MeasurementType::ZBasis]),
        }
    }

    fn get_native_protocols(&self, backend: HardwareBackend) -> Vec<String> {
        match backend {
            HardwareBackend::IBMQuantum => vec!["qiskit_measurement".to_string()],
            HardwareBackend::IonQ => vec!["native_measurement".to_string()],
            HardwareBackend::Rigetti => vec!["quil_measurement".to_string()],
            _ => vec!["standard_measurement".to_string()],
        }
    }

    fn get_timing_constraints(
        &self,
        backend: HardwareBackend,
        device_id: &str,
    ) -> DeviceResult<TimingConstraints> {
        // Get timing constraints from calibration data or defaults
        Ok(TimingConstraints {
            min_measurement_spacing: 1000.0,   // 1μs
            max_measurement_duration: 10000.0, // 10μs
            classical_deadline: 100000.0,      // 100μs
            coherence_limits: HashMap::new(),
        })
    }

    fn validate_classical_registers<const N: usize>(
        &self,
        circuit: &MeasurementCircuit<N>,
        capabilities: &MidCircuitCapabilities,
        validation_result: &mut ValidationResult,
    ) -> DeviceResult<()> {
        // Count required classical bits
        let required_bits = circuit
            .operations()
            .iter()
            .filter_map(|op| match op {
                CircuitOp::Measure(m) => Some(m.target_bit),
                _ => None,
            })
            .max()
            .unwrap_or(0)
            + 1;

        if required_bits > capabilities.classical_register_capacity {
            validation_result.errors.push(format!(
                "Circuit requires {} classical bits but device supports {}",
                required_bits, capabilities.classical_register_capacity
            ));
            validation_result.is_valid = false;
        }

        Ok(())
    }

    fn validate_timing_constraints<const N: usize>(
        &self,
        circuit: &MeasurementCircuit<N>,
        capabilities: &MidCircuitCapabilities,
        validation_result: &mut ValidationResult,
    ) -> DeviceResult<()> {
        // Check measurement spacing and timing requirements
        let mut measurement_times = Vec::new();
        let mut current_time = 0.0;

        for operation in circuit.operations() {
            match operation {
                CircuitOp::Gate(_) => {
                    current_time += 100.0; // Estimate 100ns per gate
                }
                CircuitOp::Measure(_) => {
                    measurement_times.push(current_time);
                    current_time += capabilities.timing_constraints.max_measurement_duration;
                }
                _ => {
                    current_time += 10.0; // Small overhead for other operations
                }
            }
        }

        // Check spacing between measurements
        for window in measurement_times.windows(2) {
            let spacing = window[1] - window[0];
            if spacing < capabilities.timing_constraints.min_measurement_spacing {
                validation_result.warnings.push(format!(
                    "Measurement spacing {:.1}ns is less than minimum {:.1}ns",
                    spacing, capabilities.timing_constraints.min_measurement_spacing
                ));
            }
        }

        Ok(())
    }

    fn validate_feedforward_operations<const N: usize>(
        &self,
        circuit: &MeasurementCircuit<N>,
        capabilities: &MidCircuitCapabilities,
        validation_result: &mut ValidationResult,
    ) -> DeviceResult<()> {
        if !capabilities.realtime_feedback {
            let feedforward_count = circuit
                .operations()
                .iter()
                .filter(|op| matches!(op, CircuitOp::FeedForward(_)))
                .count();

            if feedforward_count > 0 {
                validation_result.errors.push(
                    "Circuit contains feed-forward operations but device doesn't support real-time feedback".to_string()
                );
                validation_result.is_valid = false;
            }
        }

        Ok(())
    }

    fn check_measurement_conflicts<const N: usize>(
        &self,
        circuit: &MeasurementCircuit<N>,
        validation_result: &mut ValidationResult,
    ) -> DeviceResult<()> {
        let mut measured_qubits = HashSet::new();

        for operation in circuit.operations() {
            if let CircuitOp::Measure(measurement) = operation {
                if measured_qubits.contains(&measurement.qubit) {
                    validation_result.warnings.push(format!(
                        "Qubit {:?} is measured multiple times",
                        measurement.qubit
                    ));
                }
                measured_qubits.insert(measurement.qubit);
            }
        }

        Ok(())
    }

    fn batch_measurements<const N: usize>(
        &self,
        circuit: MeasurementCircuit<N>,
    ) -> DeviceResult<MeasurementCircuit<N>> {
        // Implementation for batching measurements
        // This is a complex optimization that would group measurements when possible
        Ok(circuit)
    }

    fn optimize_measurement_scheduling<const N: usize>(
        &self,
        circuit: MeasurementCircuit<N>,
    ) -> DeviceResult<MeasurementCircuit<N>> {
        // Implementation for optimizing measurement scheduling
        // This would reorder operations to minimize measurement latency
        Ok(circuit)
    }

    fn precompile_classical_conditions<const N: usize>(
        &self,
        circuit: MeasurementCircuit<N>,
    ) -> DeviceResult<MeasurementCircuit<N>> {
        // Implementation for precompiling classical conditions
        // This would optimize condition evaluation
        Ok(circuit)
    }
}

/// Validation result for mid-circuit measurement circuits
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Trait for device executors that support mid-circuit measurements
#[async_trait::async_trait]
pub trait MidCircuitDeviceExecutor {
    /// Get device identifier
    fn device_id(&self) -> &str;

    /// Execute a quantum gate
    async fn execute_gate(&self, gate: &dyn GateOp) -> DeviceResult<()>;

    /// Measure a single qubit
    async fn measure_qubit(&self, qubit: QubitId) -> DeviceResult<u8>;

    /// Measure all qubits and return final results
    async fn measure_all(&self) -> DeviceResult<HashMap<String, usize>>;

    /// Reset a qubit to |0⟩ state
    async fn reset_qubit(&self, qubit: QubitId) -> DeviceResult<()>;

    /// Synchronization barrier
    async fn synchronize(&self) -> DeviceResult<()>;

    /// Get measurement capabilities
    fn get_measurement_capabilities(&self) -> MidCircuitCapabilities;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::create_ideal_calibration;
    use quantrs2_circuit::measurement::MeasurementCircuitBuilder;

    #[test]
    fn test_mid_circuit_config_default() {
        let config = MidCircuitConfig::default();
        assert!(config.enable_realtime_processing);
        assert!(config.enable_measurement_mitigation);
        assert_eq!(config.max_measurement_latency, 1000.0);
    }

    #[test]
    fn test_capabilities_validation() {
        let mut executor =
            MidCircuitExecutor::new(MidCircuitConfig::default(), CalibrationManager::new());

        // Test capability querying
        let capabilities = executor
            .query_capabilities(HardwareBackend::IBMQuantum, "test_device")
            .unwrap();

        assert!(!capabilities.supported_measurement_types.is_empty());
    }

    #[test]
    fn test_circuit_validation() {
        let mut executor =
            MidCircuitExecutor::new(MidCircuitConfig::default(), CalibrationManager::new());

        executor
            .query_capabilities(HardwareBackend::IBMQuantum, "test")
            .unwrap();

        // Create test circuit with mid-circuit measurements
        let circuit = MeasurementCircuit::<2>::new();

        let validation = executor.validate_circuit(&circuit, "test").unwrap();
        assert!(validation.is_valid);
    }

    #[test]
    fn test_measurement_event_creation() {
        let event = MeasurementEvent {
            timestamp: 1000.0,
            qubit: QubitId(0),
            result: 1,
            storage_location: StorageLocation::ClassicalBit(0),
            latency: 50.0,
            confidence: 0.99,
        };

        assert_eq!(event.result, 1);
        assert_eq!(event.confidence, 0.99);
        assert!(matches!(
            event.storage_location,
            StorageLocation::ClassicalBit(0)
        ));
    }

    #[test]
    fn test_performance_metrics_calculation() {
        let executor =
            MidCircuitExecutor::new(MidCircuitConfig::default(), CalibrationManager::new());

        let measurement_history = vec![
            MeasurementEvent {
                timestamp: 100.0,
                qubit: QubitId(0),
                result: 1,
                storage_location: StorageLocation::ClassicalBit(0),
                latency: 50.0,
                confidence: 0.99,
            },
            MeasurementEvent {
                timestamp: 200.0,
                qubit: QubitId(1),
                result: 0,
                storage_location: StorageLocation::ClassicalBit(1),
                latency: 45.0,
                confidence: 0.98,
            },
        ];

        let execution_stats = ExecutionStats {
            total_execution_time: Duration::from_millis(10),
            quantum_time: Duration::from_millis(8),
            measurement_time: Duration::from_millis(1),
            classical_time: Duration::from_millis(1),
            num_measurements: 2,
            num_conditional_ops: 1,
            avg_measurement_latency: 47.5,
            max_measurement_latency: 50.0,
        };

        let metrics = executor
            .calculate_performance_metrics(&measurement_history, &execution_stats)
            .unwrap();

        assert!(metrics.measurement_success_rate > 0.9);
        assert!(metrics.circuit_fidelity > 0.9);
    }
}
