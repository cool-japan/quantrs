//! Gate translation for different hardware backends
//!
//! This module provides comprehensive gate translation capabilities, converting
//! gates between different hardware native gate sets and decomposing complex
//! gates into hardware-supported operations.

use ndarray::Array2;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

use quantrs2_circuit::builder::Circuit;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::{multi::*, single::*, GateOp},
    qubit::QubitId,
    synthesis::{decompose_single_qubit_zyz, decompose_two_qubit_kak},
};

/// Hardware backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HardwareBackend {
    /// IBM Quantum devices
    IBMQuantum,
    /// Google Quantum devices
    GoogleSycamore,
    /// IonQ trapped ion devices
    IonQ,
    /// Rigetti quantum devices
    Rigetti,
    /// Amazon Braket devices
    AmazonBraket,
    /// Azure Quantum devices
    AzureQuantum,
    /// Honeywell/Quantinuum devices
    Honeywell,
    /// Generic/Custom backend
    Custom(u32), // Custom ID
}

/// Native gate set for a hardware backend
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeGateSet {
    /// Backend type
    pub backend: HardwareBackend,
    /// Single-qubit gates supported
    pub single_qubit_gates: Vec<String>,
    /// Two-qubit gates supported
    pub two_qubit_gates: Vec<String>,
    /// Multi-qubit gates supported (if any)
    pub multi_qubit_gates: Vec<String>,
    /// Whether arbitrary single-qubit rotations are supported
    pub arbitrary_single_qubit: bool,
    /// Supported rotation axes for parameterized gates
    pub rotation_axes: Vec<RotationAxis>,
    /// Additional backend-specific constraints
    pub constraints: BackendConstraints,
}

impl Default for NativeGateSet {
    fn default() -> Self {
        Self {
            backend: HardwareBackend::Custom(0),
            single_qubit_gates: vec![
                "H".to_string(),
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
            ],
            two_qubit_gates: vec!["CNOT".to_string(), "CZ".to_string()],
            multi_qubit_gates: Vec::new(),
            arbitrary_single_qubit: true,
            rotation_axes: vec![RotationAxis::X, RotationAxis::Y, RotationAxis::Z],
            constraints: BackendConstraints::default(),
        }
    }
}

/// Rotation axes supported by hardware
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RotationAxis {
    X,
    Y,
    Z,
    XY(f64),               // Rotation in X-Y plane at angle
    Custom(f64, f64, f64), // Arbitrary axis (x, y, z)
}

/// Backend-specific constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendConstraints {
    /// Maximum circuit depth
    pub max_depth: Option<usize>,
    /// Supported angles for rotation gates (None = continuous)
    pub discrete_angles: Option<Vec<f64>>,
    /// Whether phase gates are virtual (frame changes)
    pub virtual_z: bool,
    /// Coupling constraints (for 2-qubit gates)
    pub coupling_map: Option<Vec<(usize, usize)>>,
    /// Gate timing constraints
    pub timing_constraints: Option<TimingConstraints>,
}

impl Default for BackendConstraints {
    fn default() -> Self {
        Self {
            max_depth: None,
            discrete_angles: None,
            virtual_z: false,
            coupling_map: None,
            timing_constraints: None,
        }
    }
}

/// Timing constraints for hardware
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingConstraints {
    /// Minimum time between gates on same qubit (ns)
    pub min_gate_spacing: f64,
    /// Required alignment for gate start times
    pub time_alignment: f64,
    /// Maximum circuit duration (ns)
    pub max_duration: Option<f64>,
}

impl Default for TimingConstraints {
    fn default() -> Self {
        Self {
            min_gate_spacing: 0.0,
            time_alignment: 1.0,
            max_duration: None,
        }
    }
}

/// Gate translator for converting between gate sets
pub struct GateTranslator {
    /// Native gate sets for each backend
    native_gates: HashMap<HardwareBackend, NativeGateSet>,
    /// Translation rules
    translation_rules: HashMap<(HardwareBackend, String), TranslationRule>,
    /// Decomposition cache
    decomposition_cache: HashMap<String, Vec<DecomposedGate>>,
}

/// Translation rule for a specific gate
#[derive(Debug, Clone)]
pub struct TranslationRule {
    /// Gate name to translate
    pub gate_name: String,
    /// Translation method
    pub method: TranslationMethod,
    /// Expected fidelity after translation
    pub fidelity: f64,
    /// Number of native gates after translation
    pub gate_count: usize,
}

/// Methods for translating gates
pub enum TranslationMethod {
    /// Direct mapping to native gate
    Direct(String),
    /// Fixed decomposition
    FixedDecomposition(Vec<DecomposedGate>),
    /// Parameterized decomposition
    ParameterizedDecomposition(Box<dyn Fn(Vec<f64>) -> Vec<DecomposedGate> + Send + Sync>),
    /// Use synthesis algorithm
    Synthesis(SynthesisMethod),
    /// Custom translation function
    Custom(Box<dyn Fn(&dyn GateOp) -> QuantRS2Result<Vec<DecomposedGate>> + Send + Sync>),
}

impl std::fmt::Debug for TranslationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Direct(s) => write!(f, "Direct({})", s),
            Self::FixedDecomposition(gates) => write!(f, "FixedDecomposition({:?})", gates),
            Self::ParameterizedDecomposition(_) => {
                write!(f, "ParameterizedDecomposition(<function>)")
            }
            Self::Synthesis(method) => write!(f, "Synthesis({:?})", method),
            Self::Custom(_) => write!(f, "Custom(<function>)"),
        }
    }
}

impl Clone for TranslationMethod {
    fn clone(&self) -> Self {
        match self {
            Self::Direct(s) => Self::Direct(s.clone()),
            Self::FixedDecomposition(gates) => Self::FixedDecomposition(gates.clone()),
            Self::ParameterizedDecomposition(_) => {
                panic!(
                    "Cannot clone ParameterizedDecomposition - use Arc<TranslationMethod> instead"
                )
            }
            Self::Synthesis(method) => Self::Synthesis(method.clone()),
            Self::Custom(_) => {
                panic!("Cannot clone Custom - use Arc<TranslationMethod> instead")
            }
        }
    }
}

/// Decomposed gate representation
#[derive(Debug, Clone)]
pub struct DecomposedGate {
    /// Gate name in native set
    pub native_gate: String,
    /// Target qubits
    pub qubits: Vec<QubitId>,
    /// Parameters (if any)
    pub parameters: Vec<f64>,
    /// Optional global phase
    pub global_phase: Option<f64>,
}

/// Synthesis methods for gate decomposition
#[derive(Debug, Clone, Copy)]
pub enum SynthesisMethod {
    /// Single-qubit synthesis (ZYZ, XYX, etc.)
    SingleQubitZYZ,
    SingleQubitXYX,
    SingleQubitU3,
    /// Two-qubit synthesis
    KAKDecomposition,
    /// Clifford+T synthesis
    CliffordT,
    /// Solovay-Kitaev
    SolovayKitaev,
}

impl GateTranslator {
    /// Create a new gate translator
    pub fn new() -> Self {
        let mut translator = Self {
            native_gates: HashMap::new(),
            translation_rules: HashMap::new(),
            decomposition_cache: HashMap::new(),
        };

        // Initialize standard backend gate sets
        translator.init_ibm_gates();
        translator.init_google_gates();
        translator.init_ionq_gates();
        translator.init_rigetti_gates();
        translator.init_amazon_gates();
        translator.init_azure_gates();
        translator.init_honeywell_gates();

        translator
    }

    /// Initialize IBM Quantum native gates
    fn init_ibm_gates(&mut self) {
        let gate_set = NativeGateSet {
            backend: HardwareBackend::IBMQuantum,
            single_qubit_gates: vec![
                "id".to_string(),
                "rz".to_string(),
                "sx".to_string(),
                "x".to_string(),
            ],
            two_qubit_gates: vec!["cx".to_string()],
            multi_qubit_gates: vec![],
            arbitrary_single_qubit: false,
            rotation_axes: vec![RotationAxis::Z],
            constraints: BackendConstraints {
                max_depth: None,
                discrete_angles: None,
                virtual_z: true,
                coupling_map: None,
                timing_constraints: None,
            },
        };

        self.native_gates
            .insert(HardwareBackend::IBMQuantum, gate_set);

        // Add translation rules
        self.add_ibm_translation_rules();
    }

    /// Add IBM-specific translation rules
    fn add_ibm_translation_rules(&mut self) {
        let backend = HardwareBackend::IBMQuantum;

        // H = RZ(π/2) SX RZ(π/2)
        self.translation_rules.insert(
            (backend, "H".to_string()),
            TranslationRule {
                gate_name: "H".to_string(),
                method: TranslationMethod::FixedDecomposition(vec![
                    DecomposedGate {
                        native_gate: "rz".to_string(),
                        qubits: vec![QubitId(0)],
                        parameters: vec![PI / 2.0],
                        global_phase: None,
                    },
                    DecomposedGate {
                        native_gate: "sx".to_string(),
                        qubits: vec![QubitId(0)],
                        parameters: vec![],
                        global_phase: None,
                    },
                    DecomposedGate {
                        native_gate: "rz".to_string(),
                        qubits: vec![QubitId(0)],
                        parameters: vec![PI / 2.0],
                        global_phase: None,
                    },
                ]),
                fidelity: 0.9999,
                gate_count: 3,
            },
        );

        // Y = RZ(π) X
        self.translation_rules.insert(
            (backend, "Y".to_string()),
            TranslationRule {
                gate_name: "Y".to_string(),
                method: TranslationMethod::FixedDecomposition(vec![
                    DecomposedGate {
                        native_gate: "rz".to_string(),
                        qubits: vec![QubitId(0)],
                        parameters: vec![PI],
                        global_phase: None,
                    },
                    DecomposedGate {
                        native_gate: "x".to_string(),
                        qubits: vec![QubitId(0)],
                        parameters: vec![],
                        global_phase: None,
                    },
                ]),
                fidelity: 0.9999,
                gate_count: 2,
            },
        );

        // S = RZ(π/2)
        self.translation_rules.insert(
            (backend, "S".to_string()),
            TranslationRule {
                gate_name: "S".to_string(),
                method: TranslationMethod::Direct("rz".to_string()),
                fidelity: 1.0,
                gate_count: 1,
            },
        );

        // T = RZ(π/4)
        self.translation_rules.insert(
            (backend, "T".to_string()),
            TranslationRule {
                gate_name: "T".to_string(),
                method: TranslationMethod::Direct("rz".to_string()),
                fidelity: 1.0,
                gate_count: 1,
            },
        );

        // RX using RZ and X gates
        self.translation_rules.insert(
            (backend, "RX".to_string()),
            TranslationRule {
                gate_name: "RX".to_string(),
                method: TranslationMethod::ParameterizedDecomposition(Box::new(|params| {
                    let theta = params[0];
                    vec![
                        DecomposedGate {
                            native_gate: "rz".to_string(),
                            qubits: vec![QubitId(0)],
                            parameters: vec![PI / 2.0],
                            global_phase: None,
                        },
                        DecomposedGate {
                            native_gate: "sx".to_string(),
                            qubits: vec![QubitId(0)],
                            parameters: vec![],
                            global_phase: None,
                        },
                        DecomposedGate {
                            native_gate: "rz".to_string(),
                            qubits: vec![QubitId(0)],
                            parameters: vec![theta - PI / 2.0],
                            global_phase: None,
                        },
                        DecomposedGate {
                            native_gate: "sx".to_string(),
                            qubits: vec![QubitId(0)],
                            parameters: vec![],
                            global_phase: None,
                        },
                        DecomposedGate {
                            native_gate: "rz".to_string(),
                            qubits: vec![QubitId(0)],
                            parameters: vec![-PI / 2.0],
                            global_phase: None,
                        },
                    ]
                })),
                fidelity: 0.9998,
                gate_count: 5,
            },
        );

        // CNOT is called CX in IBM
        self.translation_rules.insert(
            (backend, "CNOT".to_string()),
            TranslationRule {
                gate_name: "CNOT".to_string(),
                method: TranslationMethod::Direct("cx".to_string()),
                fidelity: 1.0,
                gate_count: 1,
            },
        );
    }

    /// Initialize Google Sycamore native gates
    fn init_google_gates(&mut self) {
        let gate_set = NativeGateSet {
            backend: HardwareBackend::GoogleSycamore,
            single_qubit_gates: vec![
                "ph".to_string(),    // Phase gate
                "x_pow".to_string(), // X^t gate
                "y_pow".to_string(), // Y^t gate
                "z_pow".to_string(), // Z^t gate
            ],
            two_qubit_gates: vec![
                "syc".to_string(), // Sycamore gate
                "sqrt_iswap".to_string(),
            ],
            multi_qubit_gates: vec![],
            arbitrary_single_qubit: true,
            rotation_axes: vec![RotationAxis::X, RotationAxis::Y, RotationAxis::Z],
            constraints: BackendConstraints {
                max_depth: None,
                discrete_angles: None,
                virtual_z: false,
                coupling_map: None,
                timing_constraints: None,
            },
        };

        self.native_gates
            .insert(HardwareBackend::GoogleSycamore, gate_set);
        self.add_google_translation_rules();
    }

    /// Add Google-specific translation rules
    fn add_google_translation_rules(&mut self) {
        let backend = HardwareBackend::GoogleSycamore;

        // Direct mappings for powered gates
        self.translation_rules.insert(
            (backend, "X".to_string()),
            TranslationRule {
                gate_name: "X".to_string(),
                method: TranslationMethod::FixedDecomposition(vec![DecomposedGate {
                    native_gate: "x_pow".to_string(),
                    qubits: vec![QubitId(0)],
                    parameters: vec![1.0],
                    global_phase: None,
                }]),
                fidelity: 1.0,
                gate_count: 1,
            },
        );

        // CNOT using Sycamore gates
        self.translation_rules.insert(
            (backend, "CNOT".to_string()),
            TranslationRule {
                gate_name: "CNOT".to_string(),
                method: TranslationMethod::FixedDecomposition(vec![
                    // This is a simplified version - actual decomposition is more complex
                    DecomposedGate {
                        native_gate: "syc".to_string(),
                        qubits: vec![QubitId(0), QubitId(1)],
                        parameters: vec![],
                        global_phase: None,
                    },
                    // Additional single-qubit corrections would go here
                ]),
                fidelity: 0.998,
                gate_count: 3,
            },
        );
    }

    /// Initialize IonQ native gates
    fn init_ionq_gates(&mut self) {
        let gate_set = NativeGateSet {
            backend: HardwareBackend::IonQ,
            single_qubit_gates: vec!["rx".to_string(), "ry".to_string(), "rz".to_string()],
            two_qubit_gates: vec![
                "xx".to_string(), // Mølmer-Sørensen gate
            ],
            multi_qubit_gates: vec![],
            arbitrary_single_qubit: true,
            rotation_axes: vec![RotationAxis::X, RotationAxis::Y, RotationAxis::Z],
            constraints: BackendConstraints {
                max_depth: None,
                discrete_angles: None,
                virtual_z: false,
                coupling_map: None, // All-to-all connectivity
                timing_constraints: None,
            },
        };

        self.native_gates.insert(HardwareBackend::IonQ, gate_set);
        self.add_ionq_translation_rules();
    }

    /// Add IonQ-specific translation rules
    fn add_ionq_translation_rules(&mut self) {
        let backend = HardwareBackend::IonQ;

        // CNOT using XX gate
        self.translation_rules.insert(
            (backend, "CNOT".to_string()),
            TranslationRule {
                gate_name: "CNOT".to_string(),
                method: TranslationMethod::FixedDecomposition(vec![
                    DecomposedGate {
                        native_gate: "ry".to_string(),
                        qubits: vec![QubitId(0)],
                        parameters: vec![PI / 2.0],
                        global_phase: None,
                    },
                    DecomposedGate {
                        native_gate: "xx".to_string(),
                        qubits: vec![QubitId(0), QubitId(1)],
                        parameters: vec![PI / 2.0],
                        global_phase: None,
                    },
                    DecomposedGate {
                        native_gate: "rx".to_string(),
                        qubits: vec![QubitId(0)],
                        parameters: vec![-PI / 2.0],
                        global_phase: None,
                    },
                    DecomposedGate {
                        native_gate: "rx".to_string(),
                        qubits: vec![QubitId(1)],
                        parameters: vec![-PI / 2.0],
                        global_phase: None,
                    },
                    DecomposedGate {
                        native_gate: "ry".to_string(),
                        qubits: vec![QubitId(0)],
                        parameters: vec![-PI / 2.0],
                        global_phase: None,
                    },
                ]),
                fidelity: 0.999,
                gate_count: 5,
            },
        );
    }

    /// Initialize Rigetti native gates
    fn init_rigetti_gates(&mut self) {
        let gate_set = NativeGateSet {
            backend: HardwareBackend::Rigetti,
            single_qubit_gates: vec!["rx".to_string(), "rz".to_string()],
            two_qubit_gates: vec![
                "cz".to_string(),
                "xy".to_string(), // Parametrized XY gate
            ],
            multi_qubit_gates: vec![],
            arbitrary_single_qubit: true,
            rotation_axes: vec![RotationAxis::X, RotationAxis::Z],
            constraints: BackendConstraints {
                max_depth: None,
                discrete_angles: None,
                virtual_z: false,
                coupling_map: None,
                timing_constraints: None,
            },
        };

        self.native_gates.insert(HardwareBackend::Rigetti, gate_set);
    }

    /// Initialize Amazon Braket native gates
    fn init_amazon_gates(&mut self) {
        // Amazon supports multiple backends, this is a generic set
        let gate_set = NativeGateSet {
            backend: HardwareBackend::AmazonBraket,
            single_qubit_gates: vec![
                "h".to_string(),
                "x".to_string(),
                "y".to_string(),
                "z".to_string(),
                "rx".to_string(),
                "ry".to_string(),
                "rz".to_string(),
            ],
            two_qubit_gates: vec!["cnot".to_string(), "cz".to_string(), "swap".to_string()],
            multi_qubit_gates: vec!["ccnot".to_string()],
            arbitrary_single_qubit: true,
            rotation_axes: vec![RotationAxis::X, RotationAxis::Y, RotationAxis::Z],
            constraints: BackendConstraints {
                max_depth: None,
                discrete_angles: None,
                virtual_z: false,
                coupling_map: None,
                timing_constraints: None,
            },
        };

        self.native_gates
            .insert(HardwareBackend::AmazonBraket, gate_set);
    }

    /// Initialize Azure Quantum native gates
    fn init_azure_gates(&mut self) {
        // Azure supports multiple backends, this is a generic set
        let gate_set = NativeGateSet {
            backend: HardwareBackend::AzureQuantum,
            single_qubit_gates: vec![
                "h".to_string(),
                "x".to_string(),
                "y".to_string(),
                "z".to_string(),
                "s".to_string(),
                "t".to_string(),
                "rx".to_string(),
                "ry".to_string(),
                "rz".to_string(),
            ],
            two_qubit_gates: vec!["cnot".to_string(), "cz".to_string()],
            multi_qubit_gates: vec![],
            arbitrary_single_qubit: true,
            rotation_axes: vec![RotationAxis::X, RotationAxis::Y, RotationAxis::Z],
            constraints: BackendConstraints {
                max_depth: None,
                discrete_angles: None,
                virtual_z: false,
                coupling_map: None,
                timing_constraints: None,
            },
        };

        self.native_gates
            .insert(HardwareBackend::AzureQuantum, gate_set);
    }

    /// Initialize Honeywell/Quantinuum native gates
    fn init_honeywell_gates(&mut self) {
        let gate_set = NativeGateSet {
            backend: HardwareBackend::Honeywell,
            single_qubit_gates: vec![
                "u1".to_string(), // Phase gate
                "u2".to_string(), // Single-qubit gate
                "u3".to_string(), // General single-qubit gate
            ],
            two_qubit_gates: vec![
                "zz".to_string(), // Native ZZ interaction
            ],
            multi_qubit_gates: vec![],
            arbitrary_single_qubit: true,
            rotation_axes: vec![RotationAxis::Custom(1.0, 0.0, 0.0)],
            constraints: BackendConstraints {
                max_depth: None,
                discrete_angles: None,
                virtual_z: false,
                coupling_map: None, // All-to-all connectivity
                timing_constraints: None,
            },
        };

        self.native_gates
            .insert(HardwareBackend::Honeywell, gate_set);
    }

    /// Get native gate set for a backend
    pub fn get_native_gates(&self, backend: HardwareBackend) -> Option<&NativeGateSet> {
        self.native_gates.get(&backend)
    }

    /// Check if a gate is native to a backend
    pub fn is_native_gate(&self, backend: HardwareBackend, gate_name: &str) -> bool {
        if let Some(gate_set) = self.native_gates.get(&backend) {
            gate_set.single_qubit_gates.contains(&gate_name.to_string())
                || gate_set.two_qubit_gates.contains(&gate_name.to_string())
                || gate_set.multi_qubit_gates.contains(&gate_name.to_string())
        } else {
            false
        }
    }

    /// Translate a gate to native gate set
    pub fn translate_gate(
        &mut self,
        gate: &dyn GateOp,
        backend: HardwareBackend,
    ) -> QuantRS2Result<Vec<DecomposedGate>> {
        let gate_name = gate.name();

        // Check if already native
        if self.is_native_gate(backend, gate_name) {
            return Ok(vec![DecomposedGate {
                native_gate: gate_name.to_string(),
                qubits: gate.qubits(),
                parameters: self.extract_parameters(gate)?,
                global_phase: None,
            }]);
        }

        // Check cache
        let cache_key = format!("{}_{:?}", gate_name, backend);
        if let Some(cached) = self.decomposition_cache.get(&cache_key) {
            return Ok(self.remap_qubits(cached.clone(), &gate.qubits()));
        }

        // Look for translation rule
        if let Some(rule) = self
            .translation_rules
            .get(&(backend, gate_name.to_string()))
        {
            let decomposition = match &rule.method {
                TranslationMethod::Direct(native_name) => {
                    vec![DecomposedGate {
                        native_gate: native_name.clone(),
                        qubits: gate.qubits(),
                        parameters: self.extract_parameters(gate)?,
                        global_phase: None,
                    }]
                }
                TranslationMethod::FixedDecomposition(gates) => {
                    self.remap_qubits(gates.clone(), &gate.qubits())
                }
                TranslationMethod::ParameterizedDecomposition(func) => {
                    let params = self.extract_parameters(gate)?;
                    let gates = func(params);
                    self.remap_qubits(gates, &gate.qubits())
                }
                TranslationMethod::Synthesis(method) => {
                    self.synthesize_gate(gate, *method, backend)?
                }
                TranslationMethod::Custom(func) => func(gate)?,
            };

            // Cache the result
            self.decomposition_cache
                .insert(cache_key, decomposition.clone());
            return Ok(decomposition);
        }

        // Fall back to synthesis based on gate type
        match gate.num_qubits() {
            1 => self.synthesize_gate(gate, SynthesisMethod::SingleQubitZYZ, backend),
            2 => self.synthesize_gate(gate, SynthesisMethod::KAKDecomposition, backend),
            _ => Err(QuantRS2Error::InvalidInput(format!(
                "No translation available for {}-qubit gate {}",
                gate.num_qubits(),
                gate_name
            ))),
        }
    }

    /// Translate an entire circuit
    pub fn translate_circuit<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        backend: HardwareBackend,
    ) -> QuantRS2Result<Circuit<N>> {
        let mut translated = Circuit::<N>::new();

        for gate in circuit.gates() {
            let decomposed = self.translate_gate(gate.as_ref(), backend)?;

            for dec_gate in decomposed {
                self.add_decomposed_gate(&mut translated, &dec_gate)?;
            }
        }

        Ok(translated)
    }

    /// Extract parameters from a gate
    fn extract_parameters(&self, gate: &dyn GateOp) -> QuantRS2Result<Vec<f64>> {
        // This would need to be implemented based on gate type
        // For now, return empty vector
        Ok(vec![])
    }

    /// Remap qubits in decomposition
    fn remap_qubits(
        &self,
        mut gates: Vec<DecomposedGate>,
        actual_qubits: &[QubitId],
    ) -> Vec<DecomposedGate> {
        for gate in &mut gates {
            for qubit in &mut gate.qubits {
                if qubit.id() < actual_qubits.len() as u32 {
                    *qubit = actual_qubits[qubit.id() as usize];
                }
            }
        }
        gates
    }

    /// Synthesize a gate using specified method
    fn synthesize_gate(
        &self,
        gate: &dyn GateOp,
        method: SynthesisMethod,
        backend: HardwareBackend,
    ) -> QuantRS2Result<Vec<DecomposedGate>> {
        match method {
            SynthesisMethod::SingleQubitZYZ => {
                // Use ZYZ decomposition for single-qubit gates
                if gate.num_qubits() != 1 {
                    return Err(QuantRS2Error::InvalidInput(
                        "ZYZ synthesis only works for single-qubit gates".into(),
                    ));
                }

                let matrix_vec = gate.matrix()?;
                let matrix = Array2::from_shape_vec((2, 2), matrix_vec)
                    .map_err(|_| QuantRS2Error::InvalidInput("Invalid matrix shape".into()))?;
                let decomp = decompose_single_qubit_zyz(&matrix.view())?;

                Ok(vec![
                    DecomposedGate {
                        native_gate: "rz".to_string(),
                        qubits: gate.qubits(),
                        parameters: vec![decomp.theta1],
                        global_phase: None,
                    },
                    DecomposedGate {
                        native_gate: "ry".to_string(),
                        qubits: gate.qubits(),
                        parameters: vec![decomp.phi],
                        global_phase: None,
                    },
                    DecomposedGate {
                        native_gate: "rz".to_string(),
                        qubits: gate.qubits(),
                        parameters: vec![decomp.theta2],
                        global_phase: Some(decomp.global_phase),
                    },
                ])
            }
            SynthesisMethod::KAKDecomposition => {
                // Use KAK decomposition for two-qubit gates
                if gate.num_qubits() != 2 {
                    return Err(QuantRS2Error::InvalidInput(
                        "KAK decomposition only works for two-qubit gates".into(),
                    ));
                }

                // This would use the KAK decomposition from core
                // For now, return a placeholder
                Ok(vec![])
            }
            _ => Err(QuantRS2Error::InvalidInput(format!(
                "Synthesis method {:?} not yet implemented",
                method
            ))),
        }
    }

    /// Add a decomposed gate to circuit
    fn add_decomposed_gate<const N: usize>(
        &self,
        circuit: &mut Circuit<N>,
        gate: &DecomposedGate,
    ) -> QuantRS2Result<()> {
        match gate.native_gate.as_str() {
            // IBM gates
            "id" => {} // Identity, do nothing
            "x" => {
                circuit.x(gate.qubits[0]);
            }
            "sx" => {
                circuit.sx(gate.qubits[0]);
            }
            "rz" => {
                circuit.rz(gate.qubits[0], gate.parameters[0]);
            }
            "cx" => {
                circuit.cnot(gate.qubits[0], gate.qubits[1]);
            }

            // IonQ gates
            "rx" => {
                circuit.rx(gate.qubits[0], gate.parameters[0]);
            }
            "ry" => {
                circuit.ry(gate.qubits[0], gate.parameters[0]);
            }
            "xx" => {
                // XX gate would need to be added to circuit builder
                // For now, decompose to CNOT
                circuit.cnot(gate.qubits[0], gate.qubits[1]);
            }

            // Common gates
            "h" => {
                circuit.h(gate.qubits[0]);
            }
            "y" => {
                circuit.y(gate.qubits[0]);
            }
            "z" => {
                circuit.z(gate.qubits[0]);
            }
            "s" => {
                circuit.s(gate.qubits[0]);
            }
            "t" => {
                circuit.t(gate.qubits[0]);
            }
            "cnot" => {
                circuit.cnot(gate.qubits[0], gate.qubits[1]);
            }
            "cz" => {
                circuit.cz(gate.qubits[0], gate.qubits[1]);
            }
            "swap" => {
                circuit.swap(gate.qubits[0], gate.qubits[1]);
            }
            "ccnot" => {
                circuit.toffoli(gate.qubits[0], gate.qubits[1], gate.qubits[2]);
            }

            _ => {
                return Err(QuantRS2Error::InvalidInput(format!(
                    "Unknown native gate: {}",
                    gate.native_gate
                )));
            }
        }

        Ok(())
    }
}

/// Translation optimizer that minimizes gate count or error
pub struct TranslationOptimizer {
    /// Gate translator
    translator: GateTranslator,
    /// Optimization strategy
    strategy: OptimizationStrategy,
}

/// Optimization strategies for translation
#[derive(Debug, Clone, Copy)]
pub enum OptimizationStrategy {
    /// Minimize gate count
    MinimizeGateCount,
    /// Minimize error (maximize fidelity)
    MinimizeError,
    /// Minimize circuit depth
    MinimizeDepth,
    /// Balance between gate count and error
    Balanced { weight: f64 },
}

impl TranslationOptimizer {
    /// Create a new translation optimizer
    pub fn new(strategy: OptimizationStrategy) -> Self {
        Self {
            translator: GateTranslator::new(),
            strategy,
        }
    }

    /// Find optimal translation for a gate
    pub fn optimize_translation(
        &mut self,
        gate: &dyn GateOp,
        backend: HardwareBackend,
    ) -> QuantRS2Result<Vec<DecomposedGate>> {
        // Try multiple translation methods and pick the best
        let candidates = vec![
            self.translator.translate_gate(gate, backend)?,
            // Could try alternative decompositions here
        ];

        // Select best based on strategy
        let best = match self.strategy {
            OptimizationStrategy::MinimizeGateCount => candidates
                .into_iter()
                .min_by_key(|c| c.len())
                .unwrap_or_default(),
            OptimizationStrategy::MinimizeError => {
                // Would need fidelity information
                candidates.into_iter().next().unwrap_or_default()
            }
            OptimizationStrategy::MinimizeDepth => {
                // Would need to calculate depth
                candidates.into_iter().next().unwrap_or_default()
            }
            OptimizationStrategy::Balanced { weight } => {
                // Weighted optimization
                candidates.into_iter().next().unwrap_or_default()
            }
        };

        Ok(best)
    }
}

/// Validate that a circuit uses only native gates
pub fn validate_native_circuit<const N: usize>(
    circuit: &Circuit<N>,
    backend: HardwareBackend,
) -> QuantRS2Result<bool> {
    let translator = GateTranslator::new();

    for gate in circuit.gates() {
        if !translator.is_native_gate(backend, gate.name()) {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Get translation statistics for a circuit
pub struct TranslationStats {
    /// Original gate count
    pub original_gates: usize,
    /// Native gate count after translation
    pub native_gates: usize,
    /// Gate count by type
    pub gate_counts: HashMap<String, usize>,
    /// Estimated fidelity loss
    pub fidelity_loss: f64,
    /// Circuit depth change
    pub depth_ratio: f64,
}

impl TranslationStats {
    /// Calculate statistics for a translation
    pub fn calculate<const N: usize>(
        original: &Circuit<N>,
        translated: &Circuit<N>,
        backend: HardwareBackend,
    ) -> Self {
        let mut gate_counts = HashMap::new();

        for gate in translated.gates() {
            *gate_counts.entry(gate.name().to_string()).or_insert(0) += 1;
        }

        Self {
            original_gates: original.gates().len(),
            native_gates: translated.gates().len(),
            gate_counts,
            fidelity_loss: 0.0, // Would need to calculate
            depth_ratio: 1.0,   // Would need to calculate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_translator_creation() {
        let translator = GateTranslator::new();

        // Check that standard backends are initialized
        assert!(translator
            .get_native_gates(HardwareBackend::IBMQuantum)
            .is_some());
        assert!(translator
            .get_native_gates(HardwareBackend::GoogleSycamore)
            .is_some());
        assert!(translator.get_native_gates(HardwareBackend::IonQ).is_some());
    }

    #[test]
    fn test_native_gate_check() {
        let translator = GateTranslator::new();

        // IBM native gates
        assert!(translator.is_native_gate(HardwareBackend::IBMQuantum, "rz"));
        assert!(translator.is_native_gate(HardwareBackend::IBMQuantum, "sx"));
        assert!(translator.is_native_gate(HardwareBackend::IBMQuantum, "cx"));
        assert!(!translator.is_native_gate(HardwareBackend::IBMQuantum, "h"));

        // IonQ native gates
        assert!(translator.is_native_gate(HardwareBackend::IonQ, "rx"));
        assert!(translator.is_native_gate(HardwareBackend::IonQ, "xx"));
        assert!(!translator.is_native_gate(HardwareBackend::IonQ, "cnot"));
    }

    #[test]
    fn test_hadamard_translation_ibm() {
        let mut translator = GateTranslator::new();
        let h_gate = Hadamard { target: QubitId(0) };

        let decomposed = translator
            .translate_gate(&h_gate, HardwareBackend::IBMQuantum)
            .unwrap();

        // H = RZ(π/2) SX RZ(π/2)
        assert_eq!(decomposed.len(), 3);
        assert_eq!(decomposed[0].native_gate, "rz");
        assert_eq!(decomposed[1].native_gate, "sx");
        assert_eq!(decomposed[2].native_gate, "rz");
    }

    #[test]
    fn test_circuit_translation() {
        let mut translator = GateTranslator::new();

        let mut circuit = Circuit::<2>::new();
        circuit.h(QubitId(0));
        circuit.cnot(QubitId(0), QubitId(1));

        let translated = translator
            .translate_circuit(&circuit, HardwareBackend::IBMQuantum)
            .unwrap();

        // Original: H, CNOT
        // Translated: RZ, SX, RZ, CX
        assert!(translated.gates().len() >= 4);
    }
}
