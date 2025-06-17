//! Circuit validation for different quantum backends
//!
//! This module provides comprehensive validation capabilities to ensure quantum circuits
//! are compatible with specific backend requirements, constraints, and capabilities.

use crate::builder::Circuit;
use crate::noise_models::NoiseModel;
use crate::routing::CouplingMap;
use crate::transpiler::{HardwareSpec, NativeGateSet};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Validation rules for a quantum backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRules {
    /// Backend identifier
    pub backend_name: String,
    /// Maximum number of qubits
    pub max_qubits: usize,
    /// Qubit connectivity constraints
    pub connectivity: ConnectivityConstraints,
    /// Gate set restrictions
    pub gate_restrictions: GateRestrictions,
    /// Circuit depth limits
    pub depth_limits: DepthLimits,
    /// Measurement constraints
    pub measurement_constraints: MeasurementConstraints,
    /// Classical control flow constraints
    pub classical_constraints: ClassicalConstraints,
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

/// Connectivity constraints for qubits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityConstraints {
    /// Coupling map defining allowed connections
    pub coupling_map: Option<CouplingMap>,
    /// Whether all-to-all connectivity is allowed
    pub all_to_all: bool,
    /// Maximum distance for multi-qubit operations
    pub max_distance: Option<usize>,
    /// Restricted qubit pairs (forbidden connections)
    pub forbidden_pairs: HashSet<(usize, usize)>,
}

/// Gate set restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateRestrictions {
    /// Allowed native gates
    pub native_gates: NativeGateSet,
    /// Whether decomposition to native gates is required
    pub require_native: bool,
    /// Maximum gate parameters per gate
    pub max_parameters: usize,
    /// Forbidden gate combinations
    pub forbidden_sequences: Vec<Vec<String>>,
    /// Gate-specific qubit restrictions
    pub gate_qubit_restrictions: HashMap<String, HashSet<usize>>,
}

/// Circuit depth and timing limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthLimits {
    /// Maximum circuit depth
    pub max_depth: Option<usize>,
    /// Maximum execution time in microseconds
    pub max_execution_time: Option<f64>,
    /// Maximum number of gates
    pub max_gates: Option<usize>,
    /// Depth limits by gate type
    pub gate_type_limits: HashMap<String, usize>,
}

/// Measurement operation constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementConstraints {
    /// Whether mid-circuit measurements are allowed
    pub allow_mid_circuit: bool,
    /// Maximum number of measurements
    pub max_measurements: Option<usize>,
    /// Whether measurements can be conditional
    pub allow_conditional: bool,
    /// Required measurement basis
    pub required_basis: Option<String>,
    /// Qubits that cannot be measured
    pub non_measurable_qubits: HashSet<usize>,
}

/// Classical control flow constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalConstraints {
    /// Whether classical control is supported
    pub allow_classical_control: bool,
    /// Maximum classical registers
    pub max_classical_registers: Option<usize>,
    /// Whether feedback is allowed
    pub allow_feedback: bool,
    /// Maximum conditional depth
    pub max_conditional_depth: Option<usize>,
}

/// Resource usage limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: Option<usize>,
    /// Maximum execution shots
    pub max_shots: Option<usize>,
    /// Maximum job runtime in seconds
    pub max_runtime_seconds: Option<usize>,
    /// Priority constraints
    pub priority_constraints: Option<PriorityConstraints>,
}

/// Job priority constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityConstraints {
    /// Minimum user priority level
    pub min_priority: u32,
    /// Queue position limits
    pub max_queue_position: Option<usize>,
    /// Time-based restrictions
    pub time_restrictions: Option<TimeRestrictions>,
}

/// Time-based execution restrictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRestrictions {
    /// Allowed execution hours (0-23)
    pub allowed_hours: HashSet<u8>,
    /// Maintenance windows (UTC)
    pub maintenance_windows: Vec<(String, String)>,
    /// Maximum job duration by time of day
    pub duration_limits: HashMap<u8, usize>,
}

/// Circuit validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the circuit is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Validation statistics
    pub stats: ValidationStats,
    /// Suggested fixes
    pub suggestions: Vec<ValidationSuggestion>,
}

/// Validation error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    /// Too many qubits for backend
    ExceedsQubitLimit { required: usize, available: usize },
    /// Gate not supported by backend
    UnsupportedGate { gate_name: String, position: usize },
    /// Qubit connectivity violation
    ConnectivityViolation {
        gate_name: String,
        qubits: Vec<usize>,
        position: usize,
    },
    /// Circuit depth exceeds limit
    DepthLimitExceeded {
        actual_depth: usize,
        max_depth: usize,
    },
    /// Too many gates
    GateCountExceeded {
        actual_count: usize,
        max_count: usize,
    },
    /// Measurement constraint violation
    MeasurementViolation {
        violation_type: String,
        details: String,
    },
    /// Classical control violation
    ClassicalControlViolation {
        violation_type: String,
        details: String,
    },
    /// Resource limit exceeded
    ResourceLimitExceeded {
        resource_type: String,
        required: usize,
        available: usize,
    },
    /// Invalid gate sequence
    InvalidGateSequence {
        sequence: Vec<String>,
        position: usize,
    },
}

/// Validation warning types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationWarning {
    /// Suboptimal gate usage
    SuboptimalGateUsage {
        gate_name: String,
        suggested_alternative: String,
        positions: Vec<usize>,
    },
    /// High error rate expected
    HighErrorRate {
        estimated_error: f64,
        threshold: f64,
    },
    /// Long execution time
    LongExecutionTime {
        estimated_time: f64,
        recommended_max: f64,
    },
    /// Resource usage warning
    ResourceUsageWarning {
        resource_type: String,
        usage_percentage: f64,
    },
}

/// Validation statistics
#[derive(Debug, Clone)]
pub struct ValidationStats {
    /// Total validation time
    pub validation_time: std::time::Duration,
    /// Number of gates checked
    pub gates_checked: usize,
    /// Number of constraints evaluated
    pub constraints_evaluated: usize,
    /// Estimated circuit fidelity
    pub estimated_fidelity: Option<f64>,
    /// Estimated execution time
    pub estimated_execution_time: Option<f64>,
}

/// Validation suggestion for fixing errors
#[derive(Debug, Clone)]
pub enum ValidationSuggestion {
    /// Use transpilation to fix connectivity
    UseTranspilation { suggested_router: String },
    /// Decompose gates to native set
    DecomposeGates { gates_to_decompose: Vec<String> },
    /// Reduce circuit depth
    ReduceDepth { suggested_passes: Vec<String> },
    /// Split circuit into subcircuits
    SplitCircuit { suggested_split_points: Vec<usize> },
    /// Use different backend
    SwitchBackend { recommended_backends: Vec<String> },
}

/// Circuit validator for different backends
pub struct CircuitValidator {
    /// Validation rules by backend
    backend_rules: HashMap<String, ValidationRules>,
    /// Cached validation results
    validation_cache: HashMap<String, ValidationResult>,
}

impl CircuitValidator {
    /// Create a new circuit validator
    pub fn new() -> Self {
        let mut validator = Self {
            backend_rules: HashMap::new(),
            validation_cache: HashMap::new(),
        };

        // Load standard backend validation rules
        validator.load_standard_backends();
        validator
    }

    /// Add validation rules for a backend
    pub fn add_backend_rules(&mut self, rules: ValidationRules) {
        self.backend_rules.insert(rules.backend_name.clone(), rules);
    }

    /// Get available backends for validation
    pub fn available_backends(&self) -> Vec<String> {
        self.backend_rules.keys().cloned().collect()
    }

    /// Validate a circuit against backend requirements
    pub fn validate<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        backend: &str,
        noise_model: Option<&NoiseModel>,
    ) -> QuantRS2Result<ValidationResult> {
        let start_time = std::time::Instant::now();

        // Check if backend is supported
        let rules = self
            .backend_rules
            .get(backend)
            .ok_or_else(|| QuantRS2Error::InvalidInput(format!("Unknown backend: {}", backend)))?;

        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut suggestions = Vec::new();

        // Validate qubit count
        if N > rules.max_qubits {
            errors.push(ValidationError::ExceedsQubitLimit {
                required: N,
                available: rules.max_qubits,
            });
            suggestions.push(ValidationSuggestion::SwitchBackend {
                recommended_backends: self.find_backends_with_qubits(N),
            });
        }

        // Validate gate set
        self.validate_gate_set(circuit, rules, &mut errors, &mut warnings, &mut suggestions)?;

        // Validate connectivity
        self.validate_connectivity(circuit, rules, &mut errors, &mut suggestions)?;

        // Validate circuit depth and limits
        self.validate_depth_limits(circuit, rules, &mut errors, &mut warnings, &mut suggestions)?;

        // Validate measurements
        self.validate_measurements(circuit, rules, &mut errors, &mut warnings)?;

        // Validate resource requirements
        self.validate_resources(circuit, rules, &mut errors, &mut warnings)?;

        // Estimate fidelity if noise model provided
        let estimated_fidelity = if let Some(noise) = noise_model {
            Some(self.estimate_fidelity(circuit, noise)?)
        } else {
            None
        };

        let validation_time = start_time.elapsed();
        let is_valid = errors.is_empty();

        let result = ValidationResult {
            is_valid,
            errors,
            warnings,
            stats: ValidationStats {
                validation_time,
                gates_checked: circuit.gates().len(),
                constraints_evaluated: self.count_constraints(rules),
                estimated_fidelity,
                estimated_execution_time: Some(self.estimate_execution_time(circuit, rules)),
            },
            suggestions,
        };

        Ok(result)
    }

    /// Validate gate set compliance
    fn validate_gate_set<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        rules: &ValidationRules,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
        suggestions: &mut Vec<ValidationSuggestion>,
    ) -> QuantRS2Result<()> {
        let mut non_native_gates = Vec::new();
        let mut invalid_sequences: Vec<String> = Vec::new();

        for (i, gate) in circuit.gates().iter().enumerate() {
            let gate_name = gate.name();
            let qubit_count = gate.qubits().len();

            // Check if gate is native
            let is_native = match qubit_count {
                1 => rules
                    .gate_restrictions
                    .native_gates
                    .single_qubit
                    .contains(gate_name),
                2 => rules
                    .gate_restrictions
                    .native_gates
                    .two_qubit
                    .contains(gate_name),
                _ => rules
                    .gate_restrictions
                    .native_gates
                    .multi_qubit
                    .contains(gate_name),
            };

            if !is_native {
                if rules.gate_restrictions.require_native {
                    errors.push(ValidationError::UnsupportedGate {
                        gate_name: gate_name.to_string(),
                        position: i,
                    });
                } else {
                    non_native_gates.push(gate_name.to_string());
                }
            }

            // Check gate-specific qubit restrictions
            if let Some(allowed_qubits) = rules
                .gate_restrictions
                .gate_qubit_restrictions
                .get(gate_name)
            {
                for qubit in gate.qubits() {
                    let qubit_id = qubit.id() as usize;
                    if !allowed_qubits.contains(&qubit_id) {
                        errors.push(ValidationError::ConnectivityViolation {
                            gate_name: gate_name.to_string(),
                            qubits: vec![qubit_id],
                            position: i,
                        });
                    }
                }
            }
        }

        // Add suggestions for non-native gates
        if !non_native_gates.is_empty() {
            suggestions.push(ValidationSuggestion::DecomposeGates {
                gates_to_decompose: non_native_gates,
            });
        }

        Ok(())
    }

    /// Validate qubit connectivity constraints
    fn validate_connectivity<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        rules: &ValidationRules,
        errors: &mut Vec<ValidationError>,
        suggestions: &mut Vec<ValidationSuggestion>,
    ) -> QuantRS2Result<()> {
        if rules.connectivity.all_to_all {
            return Ok(()); // No connectivity constraints
        }

        let coupling_map = rules.connectivity.coupling_map.as_ref();
        let mut connectivity_violations = false;

        for (i, gate) in circuit.gates().iter().enumerate() {
            if gate.qubits().len() >= 2 {
                let qubits: Vec<usize> = gate.qubits().iter().map(|q| q.id() as usize).collect();

                // Check two-qubit connectivity
                if gate.qubits().len() == 2 {
                    let q1 = qubits[0];
                    let q2 = qubits[1];

                    if rules.connectivity.forbidden_pairs.contains(&(q1, q2))
                        || rules.connectivity.forbidden_pairs.contains(&(q2, q1))
                    {
                        errors.push(ValidationError::ConnectivityViolation {
                            gate_name: gate.name().to_string(),
                            qubits: vec![q1, q2],
                            position: i,
                        });
                        connectivity_violations = true;
                    }

                    if let Some(coupling) = coupling_map {
                        if !coupling.are_connected(q1, q2) {
                            errors.push(ValidationError::ConnectivityViolation {
                                gate_name: gate.name().to_string(),
                                qubits: vec![q1, q2],
                                position: i,
                            });
                            connectivity_violations = true;
                        }
                    }
                }

                // Check multi-qubit distance constraints
                if let Some(max_dist) = rules.connectivity.max_distance {
                    if let Some(coupling) = coupling_map {
                        for i in 0..qubits.len() {
                            for j in i + 1..qubits.len() {
                                let distance = coupling.distance(qubits[i], qubits[j]);
                                if distance > max_dist {
                                    errors.push(ValidationError::ConnectivityViolation {
                                        gate_name: gate.name().to_string(),
                                        qubits: vec![qubits[i], qubits[j]],
                                        position: i,
                                    });
                                    connectivity_violations = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        if connectivity_violations {
            suggestions.push(ValidationSuggestion::UseTranspilation {
                suggested_router: "SABRE".to_string(),
            });
        }

        Ok(())
    }

    /// Validate circuit depth and timing limits
    fn validate_depth_limits<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        rules: &ValidationRules,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
        suggestions: &mut Vec<ValidationSuggestion>,
    ) -> QuantRS2Result<()> {
        let circuit_depth = self.calculate_circuit_depth(circuit);
        let gate_count = circuit.gates().len();

        // Check maximum depth
        if let Some(max_depth) = rules.depth_limits.max_depth {
            if circuit_depth > max_depth {
                errors.push(ValidationError::DepthLimitExceeded {
                    actual_depth: circuit_depth,
                    max_depth,
                });
                suggestions.push(ValidationSuggestion::ReduceDepth {
                    suggested_passes: vec![
                        "GateCommutation".to_string(),
                        "GateCancellation".to_string(),
                    ],
                });
            }
        }

        // Check maximum gate count
        if let Some(max_gates) = rules.depth_limits.max_gates {
            if gate_count > max_gates {
                errors.push(ValidationError::GateCountExceeded {
                    actual_count: gate_count,
                    max_count: max_gates,
                });
                suggestions.push(ValidationSuggestion::SplitCircuit {
                    suggested_split_points: vec![max_gates / 2],
                });
            }
        }

        // Check execution time limits
        if let Some(max_time) = rules.depth_limits.max_execution_time {
            let estimated_time = self.estimate_execution_time(circuit, rules);
            if estimated_time > max_time {
                warnings.push(ValidationWarning::LongExecutionTime {
                    estimated_time,
                    recommended_max: max_time,
                });
            }
        }

        Ok(())
    }

    /// Validate measurement constraints
    fn validate_measurements<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        rules: &ValidationRules,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) -> QuantRS2Result<()> {
        // For now, basic measurement validation
        // In a full implementation, this would analyze measurement operations in the circuit
        Ok(())
    }

    /// Validate resource requirements
    fn validate_resources<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        rules: &ValidationRules,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) -> QuantRS2Result<()> {
        // Estimate memory usage
        let estimated_memory = self.estimate_memory_usage(circuit);

        if let Some(max_memory) = rules.resource_limits.max_memory_mb {
            let estimated_memory_mb = estimated_memory / (1024 * 1024);
            if estimated_memory_mb > max_memory {
                errors.push(ValidationError::ResourceLimitExceeded {
                    resource_type: "memory".to_string(),
                    required: estimated_memory_mb,
                    available: max_memory,
                });
            } else if estimated_memory_mb as f64 > max_memory as f64 * 0.8 {
                warnings.push(ValidationWarning::ResourceUsageWarning {
                    resource_type: "memory".to_string(),
                    usage_percentage: (estimated_memory_mb as f64 / max_memory as f64) * 100.0,
                });
            }
        }

        Ok(())
    }

    /// Calculate circuit depth
    fn calculate_circuit_depth<const N: usize>(&self, circuit: &Circuit<N>) -> usize {
        // Simplified depth calculation - in practice, this would consider gate parallelization
        circuit.gates().len()
    }

    /// Estimate execution time
    fn estimate_execution_time<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        rules: &ValidationRules,
    ) -> f64 {
        // Simplified estimation based on gate count
        circuit.gates().len() as f64 * 0.1 // 0.1 microseconds per gate average
    }

    /// Estimate memory usage
    fn estimate_memory_usage<const N: usize>(&self, circuit: &Circuit<N>) -> usize {
        // Estimate based on state vector simulation
        if N <= 30 {
            (1usize << N) * 16 // 16 bytes per complex amplitude
        } else {
            usize::MAX // Too large for classical simulation
        }
    }

    /// Estimate circuit fidelity
    fn estimate_fidelity<const N: usize>(
        &self,
        circuit: &Circuit<N>,
        noise_model: &NoiseModel,
    ) -> QuantRS2Result<f64> {
        // Simplified fidelity estimation
        let mut total_error = 0.0;

        for gate in circuit.gates() {
            let gate_name = gate.name();
            let error = match gate.qubits().len() {
                1 => noise_model
                    .single_qubit_errors
                    .get(gate_name)
                    .map(|e| e.depolarizing + e.amplitude_damping + e.phase_damping)
                    .unwrap_or(0.001),
                2 => noise_model
                    .two_qubit_errors
                    .get(gate_name)
                    .map(|e| e.depolarizing)
                    .unwrap_or(0.01),
                _ => 0.05, // Multi-qubit gates
            };
            total_error += error;
        }

        Ok((1.0 - total_error).max(0.0))
    }

    /// Count validation constraints
    fn count_constraints(&self, rules: &ValidationRules) -> usize {
        let mut count = 0;

        count += 1; // Qubit count
        count += rules.gate_restrictions.native_gates.single_qubit.len();
        count += rules.gate_restrictions.native_gates.two_qubit.len();
        count += rules.gate_restrictions.native_gates.multi_qubit.len();

        if rules.depth_limits.max_depth.is_some() {
            count += 1;
        }
        if rules.depth_limits.max_gates.is_some() {
            count += 1;
        }
        if rules.depth_limits.max_execution_time.is_some() {
            count += 1;
        }

        count
    }

    /// Find backends that support the required number of qubits
    fn find_backends_with_qubits(&self, required_qubits: usize) -> Vec<String> {
        self.backend_rules
            .iter()
            .filter(|(_, rules)| rules.max_qubits >= required_qubits)
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Load standard backend validation rules
    fn load_standard_backends(&mut self) {
        // IBM Quantum validation rules
        self.add_backend_rules(ValidationRules::ibm_quantum());

        // Google Quantum AI validation rules
        self.add_backend_rules(ValidationRules::google_quantum());

        // AWS Braket validation rules
        self.add_backend_rules(ValidationRules::aws_braket());

        // Simulator validation rules
        self.add_backend_rules(ValidationRules::simulator());
    }
}

impl ValidationRules {
    /// IBM Quantum validation rules
    pub fn ibm_quantum() -> Self {
        let native_gates = NativeGateSet {
            single_qubit: ["X", "Y", "Z", "H", "S", "T", "RZ", "RX", "RY"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            two_qubit: ["CNOT", "CZ"].iter().map(|s| s.to_string()).collect(),
            multi_qubit: HashSet::new(),
            parameterized: [("RZ", 1), ("RX", 1), ("RY", 1)]
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
        };

        Self {
            backend_name: "ibm_quantum".to_string(),
            max_qubits: 127,
            connectivity: ConnectivityConstraints {
                coupling_map: Some(CouplingMap::grid(11, 12)), // Roughly sqrt(127) grid
                all_to_all: false,
                max_distance: Some(10),
                forbidden_pairs: HashSet::new(),
            },
            gate_restrictions: GateRestrictions {
                native_gates,
                require_native: true,
                max_parameters: 3,
                forbidden_sequences: Vec::new(),
                gate_qubit_restrictions: HashMap::new(),
            },
            depth_limits: DepthLimits {
                max_depth: Some(10000),
                max_execution_time: Some(100000.0), // 100ms
                max_gates: Some(50000),
                gate_type_limits: HashMap::new(),
            },
            measurement_constraints: MeasurementConstraints {
                allow_mid_circuit: true,
                max_measurements: Some(1000),
                allow_conditional: true,
                required_basis: None,
                non_measurable_qubits: HashSet::new(),
            },
            classical_constraints: ClassicalConstraints {
                allow_classical_control: true,
                max_classical_registers: Some(100),
                allow_feedback: true,
                max_conditional_depth: Some(100),
            },
            resource_limits: ResourceLimits {
                max_memory_mb: Some(8192),
                max_shots: Some(100000),
                max_runtime_seconds: Some(3600),
                priority_constraints: None,
            },
        }
    }

    /// Google Quantum AI validation rules
    pub fn google_quantum() -> Self {
        let native_gates = NativeGateSet {
            single_qubit: ["X", "Y", "Z", "H", "RZ", "SQRT_X"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            two_qubit: ["CZ", "ISWAP"].iter().map(|s| s.to_string()).collect(),
            multi_qubit: HashSet::new(),
            parameterized: [("RZ", 1)]
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
        };

        Self {
            backend_name: "google_quantum".to_string(),
            max_qubits: 70,
            connectivity: ConnectivityConstraints {
                coupling_map: Some(CouplingMap::grid(8, 9)),
                all_to_all: false,
                max_distance: Some(5),
                forbidden_pairs: HashSet::new(),
            },
            gate_restrictions: GateRestrictions {
                native_gates,
                require_native: true,
                max_parameters: 1,
                forbidden_sequences: Vec::new(),
                gate_qubit_restrictions: HashMap::new(),
            },
            depth_limits: DepthLimits {
                max_depth: Some(5000),
                max_execution_time: Some(50000.0),
                max_gates: Some(20000),
                gate_type_limits: HashMap::new(),
            },
            measurement_constraints: MeasurementConstraints {
                allow_mid_circuit: false,
                max_measurements: Some(70),
                allow_conditional: false,
                required_basis: Some("Z".to_string()),
                non_measurable_qubits: HashSet::new(),
            },
            classical_constraints: ClassicalConstraints {
                allow_classical_control: false,
                max_classical_registers: Some(10),
                allow_feedback: false,
                max_conditional_depth: None,
            },
            resource_limits: ResourceLimits {
                max_memory_mb: Some(4096),
                max_shots: Some(50000),
                max_runtime_seconds: Some(1800),
                priority_constraints: None,
            },
        }
    }

    /// AWS Braket validation rules
    pub fn aws_braket() -> Self {
        let native_gates = NativeGateSet {
            single_qubit: ["X", "Y", "Z", "H", "RZ", "RX", "RY"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            two_qubit: ["CNOT", "CZ", "ISWAP"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            multi_qubit: HashSet::new(),
            parameterized: [("RZ", 1), ("RX", 1), ("RY", 1)]
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
        };

        Self {
            backend_name: "aws_braket".to_string(),
            max_qubits: 100,
            connectivity: ConnectivityConstraints {
                coupling_map: None, // Varies by device
                all_to_all: true,
                max_distance: None,
                forbidden_pairs: HashSet::new(),
            },
            gate_restrictions: GateRestrictions {
                native_gates,
                require_native: false,
                max_parameters: 5,
                forbidden_sequences: Vec::new(),
                gate_qubit_restrictions: HashMap::new(),
            },
            depth_limits: DepthLimits {
                max_depth: None,
                max_execution_time: Some(200000.0),
                max_gates: None,
                gate_type_limits: HashMap::new(),
            },
            measurement_constraints: MeasurementConstraints {
                allow_mid_circuit: true,
                max_measurements: None,
                allow_conditional: true,
                required_basis: None,
                non_measurable_qubits: HashSet::new(),
            },
            classical_constraints: ClassicalConstraints {
                allow_classical_control: true,
                max_classical_registers: None,
                allow_feedback: true,
                max_conditional_depth: None,
            },
            resource_limits: ResourceLimits {
                max_memory_mb: Some(16384),
                max_shots: Some(1000000),
                max_runtime_seconds: Some(7200),
                priority_constraints: None,
            },
        }
    }

    /// Simulator validation rules
    pub fn simulator() -> Self {
        let native_gates = NativeGateSet {
            single_qubit: ["X", "Y", "Z", "H", "S", "T", "RZ", "RX", "RY", "U"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            two_qubit: ["CNOT", "CZ", "ISWAP", "SWAP", "CX"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            multi_qubit: ["Toffoli", "Fredkin"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            parameterized: [("RZ", 1), ("RX", 1), ("RY", 1), ("U", 3)]
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
        };

        Self {
            backend_name: "simulator".to_string(),
            max_qubits: 30, // Practical limit for state vector simulation
            connectivity: ConnectivityConstraints {
                coupling_map: None,
                all_to_all: true,
                max_distance: None,
                forbidden_pairs: HashSet::new(),
            },
            gate_restrictions: GateRestrictions {
                native_gates,
                require_native: false,
                max_parameters: 10,
                forbidden_sequences: Vec::new(),
                gate_qubit_restrictions: HashMap::new(),
            },
            depth_limits: DepthLimits {
                max_depth: None,
                max_execution_time: None,
                max_gates: None,
                gate_type_limits: HashMap::new(),
            },
            measurement_constraints: MeasurementConstraints {
                allow_mid_circuit: true,
                max_measurements: None,
                allow_conditional: true,
                required_basis: None,
                non_measurable_qubits: HashSet::new(),
            },
            classical_constraints: ClassicalConstraints {
                allow_classical_control: true,
                max_classical_registers: None,
                allow_feedback: true,
                max_conditional_depth: None,
            },
            resource_limits: ResourceLimits {
                max_memory_mb: None,
                max_shots: None,
                max_runtime_seconds: None,
                priority_constraints: None,
            },
        }
    }
}

impl Default for CircuitValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::Hadamard;

    #[test]
    fn test_validator_creation() {
        let validator = CircuitValidator::new();
        assert!(!validator.available_backends().is_empty());
    }

    #[test]
    fn test_validation_rules_creation() {
        let rules = ValidationRules::ibm_quantum();
        assert_eq!(rules.backend_name, "ibm_quantum");
        assert_eq!(rules.max_qubits, 127);
    }

    #[test]
    fn test_simple_circuit_validation() {
        let mut validator = CircuitValidator::new();
        let mut circuit = Circuit::<2>::new();
        circuit.add_gate(Hadamard { target: QubitId(0) }).unwrap();

        let result = validator.validate(&circuit, "simulator", None).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_qubit_limit_validation() {
        let mut validator = CircuitValidator::new();
        let circuit = Circuit::<200>::new(); // Too many qubits

        let result = validator.validate(&circuit, "ibm_quantum", None).unwrap();
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_connectivity_validation() {
        let mut validator = CircuitValidator::new();
        let mut circuit = Circuit::<3>::new();
        // This would require checking actual connectivity constraints
        circuit
            .add_gate(CNOT {
                control: QubitId(0),
                target: QubitId(1),
            })
            .unwrap();

        let result = validator.validate(&circuit, "ibm_quantum", None).unwrap();
        // Result depends on the specific connectivity of the coupling map
    }
}
