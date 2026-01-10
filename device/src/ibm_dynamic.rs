//! IBM Dynamic Circuits Executor
//!
//! This module provides execution capabilities for dynamic circuits on IBM backends:
//! - Mid-circuit measurement with classical feedback
//! - Switch-case statements based on measurement results
//! - Classical arithmetic and logical operations
//! - Timing validation and capability checking
//!
//! ## Example
//!
//! ```rust,ignore
//! use quantrs2_device::ibm_dynamic::{IBMDynamicExecutor, DynamicCircuitBuilder};
//!
//! // Build a dynamic circuit with mid-circuit measurement
//! let mut builder = DynamicCircuitBuilder::new(2);
//! builder.h(0);
//! builder.measure(0, 0);
//! builder.if_then("c[0] == 1", |b| {
//!     b.x(1);
//!     Ok(())
//! })?;
//! builder.measure(1, 1);
//!
//! // Execute on IBM backend
//! let executor = IBMDynamicExecutor::new(client, "ibm_brisbane")?;
//! let result = executor.submit_dynamic_circuit(&builder.build()?).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::ibm::IBMQuantumClient;
use crate::qasm3::{Qasm3Builder, Qasm3Circuit, Qasm3Statement};
use crate::{DeviceError, DeviceResult};

/// Dynamic circuit capabilities of an IBM backend
#[derive(Debug, Clone)]
pub struct DynamicCapabilities {
    /// Backend supports classical feedback
    pub supports_classical_feedback: bool,
    /// Maximum classical computation latency in microseconds
    pub max_classical_latency_us: u64,
    /// Backend supports switch-case statements
    pub supports_switch_case: bool,
    /// Maximum depth for dynamic circuits
    pub max_dynamic_depth: usize,
    /// Supported classical operations
    pub supported_operations: Vec<ClassicalOperation>,
    /// Maximum number of mid-circuit measurements
    pub max_mid_circuit_measurements: usize,
    /// Supports real-time classical computation
    pub supports_realtime_classical: bool,
}

impl Default for DynamicCapabilities {
    fn default() -> Self {
        Self {
            supports_classical_feedback: true,
            max_classical_latency_us: 1000, // 1ms
            supports_switch_case: true,
            max_dynamic_depth: 100,
            supported_operations: vec![
                ClassicalOperation::And,
                ClassicalOperation::Or,
                ClassicalOperation::Xor,
                ClassicalOperation::Not,
                ClassicalOperation::Equal,
                ClassicalOperation::NotEqual,
            ],
            max_mid_circuit_measurements: 50,
            supports_realtime_classical: false,
        }
    }
}

/// Supported classical operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassicalOperation {
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
    /// Bitwise NOT
    Not,
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Modulo
    Mod,
    /// Equality comparison
    Equal,
    /// Inequality comparison
    NotEqual,
    /// Less than
    LessThan,
    /// Greater than
    GreaterThan,
    /// Less than or equal
    LessEqual,
    /// Greater than or equal
    GreaterEqual,
}

/// Configuration for dynamic circuit execution
#[derive(Debug, Clone)]
pub struct DynamicExecutionConfig {
    /// Number of shots
    pub shots: usize,
    /// Enable timing validation
    pub validate_timing: bool,
    /// Maximum execution time in seconds
    pub max_execution_time: u64,
    /// Enable optimization for dynamic circuits
    pub optimize_dynamic: bool,
    /// Classical computation timeout in microseconds
    pub classical_timeout_us: u64,
}

impl Default for DynamicExecutionConfig {
    fn default() -> Self {
        Self {
            shots: 4096,
            validate_timing: true,
            max_execution_time: 300,
            optimize_dynamic: true,
            classical_timeout_us: 1000,
        }
    }
}

/// Timing validation result
#[derive(Debug, Clone)]
pub struct TimingValidation {
    /// Whether timing constraints are satisfied
    pub is_valid: bool,
    /// Total circuit depth
    pub total_depth: usize,
    /// Number of mid-circuit measurements
    pub mid_circuit_measurements: usize,
    /// Estimated classical processing time in microseconds
    pub estimated_classical_time_us: u64,
    /// Warnings about timing
    pub warnings: Vec<String>,
    /// Errors that would prevent execution
    pub errors: Vec<String>,
}

/// Result from dynamic circuit execution
#[derive(Debug, Clone)]
pub struct DynamicExecutionResult {
    /// Measurement counts
    pub counts: HashMap<String, usize>,
    /// Per-shot measurement results (if available)
    pub memory: Option<Vec<String>>,
    /// Execution metadata
    pub metadata: DynamicExecutionMetadata,
}

/// Metadata from dynamic circuit execution
#[derive(Debug, Clone)]
pub struct DynamicExecutionMetadata {
    /// Job ID
    pub job_id: String,
    /// Backend used
    pub backend: String,
    /// Number of shots executed
    pub shots: usize,
    /// Execution time in seconds
    pub execution_time: f64,
    /// Number of mid-circuit measurements
    pub mid_circuit_measurements: usize,
    /// Number of classical operations
    pub classical_operations: usize,
}

/// IBM Dynamic Circuit Executor
#[cfg(feature = "ibm")]
pub struct IBMDynamicExecutor {
    /// IBM Quantum client
    client: Arc<IBMQuantumClient>,
    /// Backend name
    backend: String,
    /// Execution configuration
    config: DynamicExecutionConfig,
    /// Cached capabilities
    capabilities: Option<DynamicCapabilities>,
}

#[cfg(not(feature = "ibm"))]
pub struct IBMDynamicExecutor {
    /// Backend name
    backend: String,
    /// Execution configuration
    config: DynamicExecutionConfig,
}

#[cfg(feature = "ibm")]
impl IBMDynamicExecutor {
    /// Create a new dynamic circuit executor
    pub fn new(client: IBMQuantumClient, backend: &str) -> DeviceResult<Self> {
        Ok(Self {
            client: Arc::new(client),
            backend: backend.to_string(),
            config: DynamicExecutionConfig::default(),
            capabilities: None,
        })
    }

    /// Create with custom configuration
    pub fn with_config(
        client: IBMQuantumClient,
        backend: &str,
        config: DynamicExecutionConfig,
    ) -> DeviceResult<Self> {
        Ok(Self {
            client: Arc::new(client),
            backend: backend.to_string(),
            config,
            capabilities: None,
        })
    }

    /// Get dynamic circuit capabilities for the backend
    pub async fn get_capabilities(&mut self) -> DeviceResult<&DynamicCapabilities> {
        if self.capabilities.is_none() {
            // In a real implementation, this would query the backend
            // For now, return default capabilities
            self.capabilities = Some(DynamicCapabilities::default());
        }
        Ok(self
            .capabilities
            .as_ref()
            .expect("capabilities should be set"))
    }

    /// Validate timing constraints for a dynamic circuit
    pub fn validate_timing(&self, circuit: &Qasm3Circuit) -> TimingValidation {
        let mut mid_circuit_measurements = 0;
        let mut classical_operations = 0;
        let mut warnings = Vec::new();
        let mut errors = Vec::new();

        // Count dynamic operations
        for stmt in &circuit.statements {
            self.count_dynamic_ops(
                stmt,
                &mut mid_circuit_measurements,
                &mut classical_operations,
            );
        }

        // Check against capabilities
        let default_caps = DynamicCapabilities::default();
        let capabilities = self.capabilities.as_ref().unwrap_or(&default_caps);

        if mid_circuit_measurements > capabilities.max_mid_circuit_measurements {
            errors.push(format!(
                "Too many mid-circuit measurements: {} > {}",
                mid_circuit_measurements, capabilities.max_mid_circuit_measurements
            ));
        }

        let estimated_classical_time_us = (classical_operations as u64) * 100; // ~100us per operation

        if estimated_classical_time_us > capabilities.max_classical_latency_us {
            warnings.push(format!(
                "Estimated classical processing time ({} us) exceeds recommended limit ({} us)",
                estimated_classical_time_us, capabilities.max_classical_latency_us
            ));
        }

        TimingValidation {
            is_valid: errors.is_empty(),
            total_depth: circuit.statements.len(),
            mid_circuit_measurements,
            estimated_classical_time_us,
            warnings,
            errors,
        }
    }

    /// Count dynamic operations in a statement
    fn count_dynamic_ops(
        &self,
        stmt: &Qasm3Statement,
        measurements: &mut usize,
        classical_ops: &mut usize,
    ) {
        match stmt {
            Qasm3Statement::Measure { .. } => {
                *measurements += 1;
            }
            Qasm3Statement::If {
                then_body,
                else_body,
                ..
            } => {
                *classical_ops += 1;
                for s in then_body {
                    self.count_dynamic_ops(s, measurements, classical_ops);
                }
                if let Some(else_stmts) = else_body {
                    for s in else_stmts {
                        self.count_dynamic_ops(s, measurements, classical_ops);
                    }
                }
            }
            Qasm3Statement::Switch {
                cases,
                default_case,
                ..
            } => {
                *classical_ops += 1;
                for (_, body) in cases {
                    for s in body {
                        self.count_dynamic_ops(s, measurements, classical_ops);
                    }
                }
                if let Some(default_body) = default_case {
                    for s in default_body {
                        self.count_dynamic_ops(s, measurements, classical_ops);
                    }
                }
            }
            Qasm3Statement::Assignment { .. } => {
                *classical_ops += 1;
            }
            Qasm3Statement::While { body, .. } | Qasm3Statement::For { body, .. } => {
                *classical_ops += 1;
                for s in body {
                    self.count_dynamic_ops(s, measurements, classical_ops);
                }
            }
            _ => {}
        }
    }

    /// Submit a dynamic circuit for execution
    pub async fn submit_dynamic_circuit(
        &self,
        circuit: &Qasm3Circuit,
    ) -> DeviceResult<DynamicExecutionResult> {
        // Validate timing if enabled
        if self.config.validate_timing {
            let validation = self.validate_timing(circuit);
            if !validation.is_valid {
                return Err(DeviceError::InvalidInput(format!(
                    "Dynamic circuit validation failed: {:?}",
                    validation.errors
                )));
            }
        }

        // Convert circuit to QASM string
        let qasm = circuit.to_string();

        // Submit to IBM Runtime
        let config = crate::ibm::IBMCircuitConfig {
            name: "dynamic_circuit".to_string(),
            qasm,
            shots: self.config.shots,
            optimization_level: Some(1),
            initial_layout: None,
        };

        let job_id = self.client.submit_circuit(&self.backend, config).await?;
        let result = self
            .client
            .wait_for_job(&job_id, Some(self.config.max_execution_time))
            .await?;

        // Count operations for metadata
        let mut mid_circuit_measurements = 0;
        let mut classical_operations = 0;
        for stmt in &circuit.statements {
            self.count_dynamic_ops(
                stmt,
                &mut mid_circuit_measurements,
                &mut classical_operations,
            );
        }

        Ok(DynamicExecutionResult {
            counts: result.counts,
            memory: None, // Would be populated from actual result
            metadata: DynamicExecutionMetadata {
                job_id,
                backend: self.backend.clone(),
                shots: self.config.shots,
                execution_time: 0.0, // Would be from actual result
                mid_circuit_measurements,
                classical_operations,
            },
        })
    }

    /// Execute a pre-built QASM 3.0 string
    pub async fn execute_qasm(
        &self,
        qasm: &str,
        shots: usize,
    ) -> DeviceResult<DynamicExecutionResult> {
        let config = crate::ibm::IBMCircuitConfig {
            name: "dynamic_qasm".to_string(),
            qasm: qasm.to_string(),
            shots,
            optimization_level: Some(1),
            initial_layout: None,
        };

        let job_id = self.client.submit_circuit(&self.backend, config).await?;
        let result = self
            .client
            .wait_for_job(&job_id, Some(self.config.max_execution_time))
            .await?;

        Ok(DynamicExecutionResult {
            counts: result.counts,
            memory: None,
            metadata: DynamicExecutionMetadata {
                job_id,
                backend: self.backend.clone(),
                shots,
                execution_time: 0.0,
                mid_circuit_measurements: 0, // Unknown without parsing
                classical_operations: 0,
            },
        })
    }
}

#[cfg(not(feature = "ibm"))]
impl IBMDynamicExecutor {
    pub fn new(_client: IBMQuantumClient, backend: &str) -> DeviceResult<Self> {
        Ok(Self {
            backend: backend.to_string(),
            config: DynamicExecutionConfig::default(),
        })
    }

    pub async fn submit_dynamic_circuit(
        &self,
        _circuit: &Qasm3Circuit,
    ) -> DeviceResult<DynamicExecutionResult> {
        Err(DeviceError::UnsupportedDevice(
            "IBM Runtime support not enabled".to_string(),
        ))
    }

    pub fn validate_timing(&self, _circuit: &Qasm3Circuit) -> TimingValidation {
        TimingValidation {
            is_valid: false,
            total_depth: 0,
            mid_circuit_measurements: 0,
            estimated_classical_time_us: 0,
            warnings: vec![],
            errors: vec!["IBM support not enabled".to_string()],
        }
    }
}

/// Builder for dynamic circuits with mid-circuit measurement
pub struct DynamicCircuitBuilder {
    inner: Qasm3Builder,
    mid_circuit_measurements: usize,
}

impl DynamicCircuitBuilder {
    /// Create a new dynamic circuit builder
    pub fn new(num_qubits: usize) -> Self {
        Self {
            inner: Qasm3Builder::new(num_qubits),
            mid_circuit_measurements: 0,
        }
    }

    /// Create with specified number of classical bits
    pub fn with_bits(mut self, num_bits: usize) -> Self {
        self.inner = self.inner.with_bits(num_bits);
        self
    }

    /// Add a Hadamard gate
    pub fn h(&mut self, qubit: usize) -> DeviceResult<&mut Self> {
        self.inner.gate("h", &[qubit])?;
        Ok(self)
    }

    /// Add a Pauli-X gate
    pub fn x(&mut self, qubit: usize) -> DeviceResult<&mut Self> {
        self.inner.gate("x", &[qubit])?;
        Ok(self)
    }

    /// Add a Pauli-Y gate
    pub fn y(&mut self, qubit: usize) -> DeviceResult<&mut Self> {
        self.inner.gate("y", &[qubit])?;
        Ok(self)
    }

    /// Add a Pauli-Z gate
    pub fn z(&mut self, qubit: usize) -> DeviceResult<&mut Self> {
        self.inner.gate("z", &[qubit])?;
        Ok(self)
    }

    /// Add a CNOT gate
    pub fn cx(&mut self, control: usize, target: usize) -> DeviceResult<&mut Self> {
        self.inner.ctrl_gate("x", control, target, &[])?;
        Ok(self)
    }

    /// Add a rotation around X axis
    pub fn rx(&mut self, qubit: usize, angle: f64) -> DeviceResult<&mut Self> {
        self.inner.gate_with_params("rx", &[angle], &[qubit])?;
        Ok(self)
    }

    /// Add a rotation around Y axis
    pub fn ry(&mut self, qubit: usize, angle: f64) -> DeviceResult<&mut Self> {
        self.inner.gate_with_params("ry", &[angle], &[qubit])?;
        Ok(self)
    }

    /// Add a rotation around Z axis
    pub fn rz(&mut self, qubit: usize, angle: f64) -> DeviceResult<&mut Self> {
        self.inner.gate_with_params("rz", &[angle], &[qubit])?;
        Ok(self)
    }

    /// Add a measurement (mid-circuit or final)
    pub fn measure(&mut self, qubit: usize, bit: usize) -> DeviceResult<&mut Self> {
        self.inner.measure(qubit, bit)?;
        self.mid_circuit_measurements += 1;
        Ok(self)
    }

    /// Add measurements for all qubits
    pub fn measure_all(&mut self) -> DeviceResult<&mut Self> {
        self.inner.measure_all()?;
        Ok(self)
    }

    /// Add a reset operation
    pub fn reset(&mut self, qubit: usize) -> DeviceResult<&mut Self> {
        self.inner.reset(qubit)?;
        Ok(self)
    }

    /// Add a barrier
    pub fn barrier(&mut self, qubits: &[usize]) -> DeviceResult<&mut Self> {
        self.inner.barrier(qubits)?;
        Ok(self)
    }

    /// Add a conditional operation based on classical bit value
    ///
    /// # Example
    /// ```ignore
    /// builder.if_then("c[0] == 1", |b| {
    ///     b.x(1);
    ///     Ok(())
    /// })?;
    /// ```
    pub fn if_then<F>(&mut self, condition: &str, body: F) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut DynamicCircuitBuilder) -> DeviceResult<()>,
    {
        // Create a temporary builder for the body
        let mut temp_builder = DynamicCircuitBuilder {
            inner: Qasm3Builder {
                circuit: crate::qasm3::Qasm3Circuit::new(),
                num_qubits: self.inner.num_qubits,
                num_bits: self.inner.num_bits,
            },
            mid_circuit_measurements: 0,
        };
        temp_builder.inner.circuit.statements.clear();

        body(&mut temp_builder)?;

        self.inner.circuit.add_statement(Qasm3Statement::If {
            condition: condition.to_string(),
            then_body: temp_builder.inner.circuit.statements,
            else_body: None,
        });

        self.mid_circuit_measurements += temp_builder.mid_circuit_measurements;

        Ok(self)
    }

    /// Add a conditional operation with else branch
    pub fn if_then_else<F, G>(
        &mut self,
        condition: &str,
        then_body: F,
        else_body: G,
    ) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut DynamicCircuitBuilder) -> DeviceResult<()>,
        G: FnOnce(&mut DynamicCircuitBuilder) -> DeviceResult<()>,
    {
        let mut then_builder = DynamicCircuitBuilder {
            inner: Qasm3Builder {
                circuit: crate::qasm3::Qasm3Circuit::new(),
                num_qubits: self.inner.num_qubits,
                num_bits: self.inner.num_bits,
            },
            mid_circuit_measurements: 0,
        };
        then_builder.inner.circuit.statements.clear();
        then_body(&mut then_builder)?;

        let mut else_builder = DynamicCircuitBuilder {
            inner: Qasm3Builder {
                circuit: crate::qasm3::Qasm3Circuit::new(),
                num_qubits: self.inner.num_qubits,
                num_bits: self.inner.num_bits,
            },
            mid_circuit_measurements: 0,
        };
        else_builder.inner.circuit.statements.clear();
        else_body(&mut else_builder)?;

        self.inner.circuit.add_statement(Qasm3Statement::If {
            condition: condition.to_string(),
            then_body: then_builder.inner.circuit.statements,
            else_body: Some(else_builder.inner.circuit.statements),
        });

        self.mid_circuit_measurements += then_builder.mid_circuit_measurements;
        self.mid_circuit_measurements += else_builder.mid_circuit_measurements;

        Ok(self)
    }

    /// Add a switch-case statement
    ///
    /// # Example
    /// ```ignore
    /// builder.switch("c", |sw| {
    ///     sw.case(&[0], |b| { b.x(0)?; Ok(()) })?;
    ///     sw.case(&[1], |b| { b.y(0)?; Ok(()) })?;
    ///     sw.default(|b| { b.z(0)?; Ok(()) })?;
    ///     Ok(())
    /// })?;
    /// ```
    pub fn switch<F>(&mut self, expression: &str, case_builder: F) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut DynamicSwitchBuilder) -> DeviceResult<()>,
    {
        let mut switch_builder =
            DynamicSwitchBuilder::new(self.inner.num_qubits, self.inner.num_bits);
        case_builder(&mut switch_builder)?;

        self.inner.circuit.add_statement(Qasm3Statement::Switch {
            expression: expression.to_string(),
            cases: switch_builder.cases,
            default_case: switch_builder.default_case,
        });

        self.mid_circuit_measurements += switch_builder.mid_circuit_measurements;

        Ok(self)
    }

    /// Add a classical assignment
    pub fn assign(&mut self, target: &str, value: &str) -> &mut Self {
        self.inner.assign(target, value);
        self
    }

    /// Add a comment
    pub fn comment(&mut self, text: &str) -> &mut Self {
        self.inner.comment(text);
        self
    }

    /// Get number of mid-circuit measurements
    pub fn num_mid_circuit_measurements(&self) -> usize {
        self.mid_circuit_measurements
    }

    /// Build the dynamic circuit
    pub fn build(self) -> DeviceResult<Qasm3Circuit> {
        self.inner.build()
    }
}

/// Builder for switch-case statements in dynamic circuits
pub struct DynamicSwitchBuilder {
    num_qubits: usize,
    num_bits: usize,
    cases: Vec<(Vec<i64>, Vec<Qasm3Statement>)>,
    default_case: Option<Vec<Qasm3Statement>>,
    mid_circuit_measurements: usize,
}

impl DynamicSwitchBuilder {
    fn new(num_qubits: usize, num_bits: usize) -> Self {
        Self {
            num_qubits,
            num_bits,
            cases: Vec::new(),
            default_case: None,
            mid_circuit_measurements: 0,
        }
    }

    /// Add a case to the switch statement
    pub fn case<F>(&mut self, values: &[i64], body: F) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut DynamicCircuitBuilder) -> DeviceResult<()>,
    {
        let mut case_builder = DynamicCircuitBuilder {
            inner: Qasm3Builder {
                circuit: crate::qasm3::Qasm3Circuit::new(),
                num_qubits: self.num_qubits,
                num_bits: self.num_bits,
            },
            mid_circuit_measurements: 0,
        };
        case_builder.inner.circuit.statements.clear();

        body(&mut case_builder)?;

        self.cases
            .push((values.to_vec(), case_builder.inner.circuit.statements));
        self.mid_circuit_measurements += case_builder.mid_circuit_measurements;

        Ok(self)
    }

    /// Add a default case
    pub fn default<F>(&mut self, body: F) -> DeviceResult<&mut Self>
    where
        F: FnOnce(&mut DynamicCircuitBuilder) -> DeviceResult<()>,
    {
        let mut default_builder = DynamicCircuitBuilder {
            inner: Qasm3Builder {
                circuit: crate::qasm3::Qasm3Circuit::new(),
                num_qubits: self.num_qubits,
                num_bits: self.num_bits,
            },
            mid_circuit_measurements: 0,
        };
        default_builder.inner.circuit.statements.clear();

        body(&mut default_builder)?;

        self.default_case = Some(default_builder.inner.circuit.statements);
        self.mid_circuit_measurements += default_builder.mid_circuit_measurements;

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_capabilities_default() {
        let caps = DynamicCapabilities::default();
        assert!(caps.supports_classical_feedback);
        assert!(caps.supports_switch_case);
        assert_eq!(caps.max_mid_circuit_measurements, 50);
    }

    #[test]
    fn test_dynamic_execution_config_default() {
        let config = DynamicExecutionConfig::default();
        assert_eq!(config.shots, 4096);
        assert!(config.validate_timing);
    }

    #[test]
    fn test_dynamic_circuit_builder() {
        let mut builder = DynamicCircuitBuilder::new(2);
        builder.h(0).unwrap();
        builder.cx(0, 1).unwrap();
        builder.measure(0, 0).unwrap();
        builder.measure(1, 1).unwrap();

        let circuit = builder.build().unwrap();
        let qasm = circuit.to_string();

        assert!(qasm.contains("h q[0]"));
        assert!(qasm.contains("ctrl @"));
    }

    #[test]
    fn test_dynamic_circuit_with_if() {
        let mut builder = DynamicCircuitBuilder::new(2);
        builder.h(0).unwrap();
        builder.measure(0, 0).unwrap();
        builder
            .if_then("c[0] == 1", |b| {
                b.x(1)?;
                Ok(())
            })
            .unwrap();

        let circuit = builder.build().unwrap();
        let qasm = circuit.to_string();

        assert!(qasm.contains("if (c[0] == 1)"));
        assert!(qasm.contains("x q[1]"));
    }

    #[test]
    fn test_dynamic_circuit_with_switch() {
        let mut builder = DynamicCircuitBuilder::new(2);
        builder.h(0).unwrap();
        builder.measure(0, 0).unwrap();
        builder
            .switch("c[0]", |sw| {
                sw.case(&[0], |b| {
                    b.x(1)?;
                    Ok(())
                })?;
                sw.case(&[1], |b| {
                    b.y(1)?;
                    Ok(())
                })?;
                Ok(())
            })
            .unwrap();

        let circuit = builder.build().unwrap();
        let qasm = circuit.to_string();

        assert!(qasm.contains("switch (c[0])"));
        assert!(qasm.contains("case 0"));
        assert!(qasm.contains("case 1"));
    }

    #[test]
    fn test_classical_operations() {
        assert_eq!(ClassicalOperation::And, ClassicalOperation::And);
        assert_ne!(ClassicalOperation::And, ClassicalOperation::Or);
    }

    #[test]
    fn test_timing_validation_structure() {
        let validation = TimingValidation {
            is_valid: true,
            total_depth: 10,
            mid_circuit_measurements: 2,
            estimated_classical_time_us: 200,
            warnings: vec![],
            errors: vec![],
        };

        assert!(validation.is_valid);
        assert_eq!(validation.mid_circuit_measurements, 2);
    }
}
