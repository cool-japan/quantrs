//! Advanced Quantum Circuit Linter with Enhanced SciRS2 Pattern Matching
//!
//! This module provides state-of-the-art quantum circuit linting with sophisticated
//! pattern matching, optimization detection, anti-pattern identification, and
//! comprehensive code quality analysis powered by SciRS2.

use crate::error::QuantRS2Error;
use crate::gate_translation::GateType;
use crate::scirs2_quantum_linter::{LintSeverity, LintingConfig, QuantumGate};
use scirs2_core::Complex64;
// use scirs2_core::parallel_ops::*;
use crate::parallel_ops_stubs::*;
// use scirs2_core::memory::BufferPool;
use crate::buffer_pool::BufferPool;
use crate::platform::PlatformCapabilities;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

/// Enhanced linting configuration with advanced pattern matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedLintingConfig {
    /// Base linting configuration
    pub base_config: LintingConfig,

    /// Enable machine learning-based pattern detection
    pub enable_ml_pattern_detection: bool,

    /// Enable quantum algorithm recognition
    pub enable_algorithm_recognition: bool,

    /// Enable circuit complexity analysis
    pub enable_complexity_analysis: bool,

    /// Enable noise resilience checking
    pub enable_noise_resilience_check: bool,

    /// Enable topological optimization suggestions
    pub enable_topological_optimization: bool,

    /// Enable quantum error correction pattern detection
    pub enable_qec_pattern_detection: bool,

    /// Enable cross-compilation optimization
    pub enable_cross_compilation_check: bool,

    /// Enable hardware-specific linting
    pub enable_hardware_specific_linting: bool,

    /// Target hardware architectures
    pub target_architectures: Vec<HardwareArchitecture>,

    /// Pattern database version
    pub pattern_database_version: String,

    /// Maximum circuit depth for analysis
    pub max_analysis_depth: usize,

    /// Enable incremental linting
    pub enable_incremental_linting: bool,

    /// Custom lint rules
    pub custom_rules: Vec<CustomLintRule>,

    /// Report format options
    pub report_format: ReportFormat,
}

impl Default for EnhancedLintingConfig {
    fn default() -> Self {
        Self {
            base_config: LintingConfig::default(),
            enable_ml_pattern_detection: true,
            enable_algorithm_recognition: true,
            enable_complexity_analysis: true,
            enable_noise_resilience_check: true,
            enable_topological_optimization: true,
            enable_qec_pattern_detection: true,
            enable_cross_compilation_check: true,
            enable_hardware_specific_linting: true,
            target_architectures: vec![
                HardwareArchitecture::IBMQ,
                HardwareArchitecture::IonQ,
                HardwareArchitecture::Simulator,
            ],
            pattern_database_version: "1.0.0".to_string(),
            max_analysis_depth: 1000,
            enable_incremental_linting: true,
            custom_rules: Vec::new(),
            report_format: ReportFormat::Detailed,
        }
    }
}

/// Hardware architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareArchitecture {
    IBMQ,
    IonQ,
    Rigetti,
    GoogleSycamore,
    Honeywell,
    AWSBraket,
    Simulator,
    Custom,
}

/// Report format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    Summary,
    Detailed,
    JSON,
    SARIF,
    HTML,
    Markdown,
}

/// Custom lint rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomLintRule {
    pub name: String,
    pub description: String,
    pub pattern: LintPattern,
    pub severity: LintSeverity,
    pub fix_suggestion: Option<String>,
}

/// Lint pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LintPattern {
    /// Sequence of gates matching a pattern
    GateSequence(Vec<GatePatternMatcher>),
    /// Circuit structure pattern
    StructuralPattern(StructuralMatcher),
    /// Resource usage pattern
    ResourcePattern(ResourceMatcher),
    /// Custom regex-like pattern
    CustomPattern(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatePatternMatcher {
    pub gate_type: Option<GateType>,
    pub qubit_count: Option<usize>,
    pub is_controlled: Option<bool>,
    pub is_parameterized: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralMatcher {
    pub min_depth: Option<usize>,
    pub max_depth: Option<usize>,
    pub has_loops: Option<bool>,
    pub has_measurements: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMatcher {
    pub min_qubits: Option<usize>,
    pub max_qubits: Option<usize>,
    pub min_gates: Option<usize>,
    pub max_gates: Option<usize>,
}

/// Enhanced lint finding with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedLintFinding {
    pub finding_type: LintFindingType,
    pub severity: LintSeverity,
    pub location: CircuitLocation,
    pub message: String,
    pub explanation: String,
    pub impact: ImpactAnalysis,
    pub fix_suggestions: Vec<FixSuggestion>,
    pub related_findings: Vec<usize>,
    pub confidence: f64,
    pub references: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitLocation {
    pub gate_indices: Vec<usize>,
    pub qubit_indices: Vec<usize>,
    pub layer: Option<usize>,
    pub subcircuit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    pub performance_impact: PerformanceImpact,
    pub error_impact: f64,
    pub resource_impact: ResourceImpact,
    pub hardware_compatibility: Vec<(HardwareArchitecture, Compatibility)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceImpact {
    Negligible,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceImpact {
    pub additional_gates: i32,
    pub additional_qubits: i32,
    pub depth_increase: i32,
    pub memory_overhead: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Compatibility {
    FullyCompatible,
    PartiallyCompatible,
    RequiresTranspilation,
    Incompatible,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixSuggestion {
    pub description: String,
    pub automatic: bool,
    pub code_changes: Vec<CodeChange>,
    pub estimated_improvement: f64,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChange {
    pub operation: ChangeOperation,
    pub location: CircuitLocation,
    pub new_gates: Option<Vec<QuantumGate>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeOperation {
    Replace,
    Insert,
    Delete,
    Reorder,
    Merge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Safe,
    Low,
    Medium,
    High,
}

/// Lint finding types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LintFindingType {
    // Performance issues
    RedundantGates,
    InefficientDecomposition,
    MissedFusionOpportunity,
    SuboptimalGateOrder,
    ExcessiveCircuitDepth,

    // Anti-patterns
    QuantumAntiPattern,
    UnnecessaryMeasurement,
    EntanglementLeak,
    CoherenceViolation,

    // Best practices
    MissingErrorMitigation,
    PoorQubitAllocation,
    InadequateParameterization,
    LackOfModularity,

    // Hardware compatibility
    UnsupportedGateSet,
    ConnectivityViolation,
    ExceedsCoherenceTime,
    CalibrationMismatch,

    // Algorithmic issues
    IncorrectAlgorithmImplementation,
    SuboptimalAlgorithmChoice,
    MissingAncillaQubits,

    // Resource usage
    ExcessiveQubitUsage,
    MemoryInefficiency,
    ParallelizationOpportunity,

    // Numerical issues
    NumericalInstability,
    PrecisionLoss,
    PhaseAccumulation,

    // Custom
    CustomRule(String),
}

impl fmt::Display for LintFindingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RedundantGates => write!(f, "Redundant Gates"),
            Self::InefficientDecomposition => write!(f, "Inefficient Decomposition"),
            Self::MissedFusionOpportunity => write!(f, "Missed Fusion Opportunity"),
            Self::SuboptimalGateOrder => write!(f, "Suboptimal Gate Order"),
            Self::ExcessiveCircuitDepth => write!(f, "Excessive Circuit Depth"),
            Self::QuantumAntiPattern => write!(f, "Quantum Anti-Pattern"),
            Self::UnnecessaryMeasurement => write!(f, "Unnecessary Measurement"),
            Self::EntanglementLeak => write!(f, "Entanglement Leak"),
            Self::CoherenceViolation => write!(f, "Coherence Violation"),
            Self::MissingErrorMitigation => write!(f, "Missing Error Mitigation"),
            Self::PoorQubitAllocation => write!(f, "Poor Qubit Allocation"),
            Self::InadequateParameterization => write!(f, "Inadequate Parameterization"),
            Self::LackOfModularity => write!(f, "Lack of Modularity"),
            Self::UnsupportedGateSet => write!(f, "Unsupported Gate Set"),
            Self::ConnectivityViolation => write!(f, "Connectivity Violation"),
            Self::ExceedsCoherenceTime => write!(f, "Exceeds Coherence Time"),
            Self::CalibrationMismatch => write!(f, "Calibration Mismatch"),
            Self::IncorrectAlgorithmImplementation => {
                write!(f, "Incorrect Algorithm Implementation")
            }
            Self::SuboptimalAlgorithmChoice => write!(f, "Suboptimal Algorithm Choice"),
            Self::MissingAncillaQubits => write!(f, "Missing Ancilla Qubits"),
            Self::ExcessiveQubitUsage => write!(f, "Excessive Qubit Usage"),
            Self::MemoryInefficiency => write!(f, "Memory Inefficiency"),
            Self::ParallelizationOpportunity => write!(f, "Parallelization Opportunity"),
            Self::NumericalInstability => write!(f, "Numerical Instability"),
            Self::PrecisionLoss => write!(f, "Precision Loss"),
            Self::PhaseAccumulation => write!(f, "Phase Accumulation"),
            Self::CustomRule(name) => write!(f, "Custom Rule: {name}"),
        }
    }
}

/// Pattern database for efficient pattern matching
struct PatternDatabase {
    gate_patterns: HashMap<String, CompiledPattern>,
    algorithm_signatures: HashMap<String, AlgorithmSignature>,
    antipattern_library: Vec<AntiPattern>,
    optimization_rules: Vec<OptimizationRule>,
}

struct CompiledPattern {
    pattern_id: String,
    matcher: Box<dyn Fn(&[QuantumGate]) -> bool + Send + Sync>,
    min_length: usize,
    max_length: usize,
}

#[derive(Debug, Clone)]
struct AlgorithmSignature {
    name: String,
    gate_sequence: Vec<GateType>,
    variations: Vec<Vec<GateType>>,
    required_qubits: usize,
}

#[derive(Debug, Clone)]
struct AntiPattern {
    name: String,
    description: String,
    pattern: Vec<GatePatternMatcher>,
    severity: LintSeverity,
    fix_strategy: String,
}

struct OptimizationRule {
    name: String,
    condition: Box<dyn Fn(&[QuantumGate]) -> bool + Send + Sync>,
    optimization: Box<dyn Fn(&[QuantumGate]) -> Vec<QuantumGate> + Send + Sync>,
    improvement_estimate: f64,
}

/// Machine learning-based pattern detector
struct MLPatternDetector {
    model_type: MLModelType,
    feature_extractor: FeatureExtractor,
    pattern_classifier: PatternClassifier,
    confidence_threshold: f64,
}

#[derive(Debug, Clone, Copy)]
enum MLModelType {
    NeuralNetwork,
    DecisionTree,
    SVM,
    Ensemble,
}

struct FeatureExtractor {
    feature_functions: Vec<Box<dyn Fn(&[QuantumGate]) -> Vec<f64> + Send + Sync>>,
}

struct PatternClassifier {
    classify: Box<dyn Fn(&[f64]) -> (LintFindingType, f64) + Send + Sync>,
}

/// Enhanced quantum circuit linter
pub struct EnhancedQuantumLinter {
    config: EnhancedLintingConfig,
    platform_caps: PlatformCapabilities,
    buffer_pool: Arc<BufferPool<Complex64>>,
    pattern_database: Arc<PatternDatabase>,
    ml_detector: Option<MLPatternDetector>,
    cache: Arc<Mutex<LintCache>>,
    statistics: Arc<Mutex<LintStatistics>>,
}

struct LintCache {
    findings: HashMap<String, Vec<EnhancedLintFinding>>,
    analyzed_patterns: HashSet<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct LintStatistics {
    total_circuits_analyzed: usize,
    total_findings: usize,
    findings_by_type: HashMap<LintFindingType, usize>,
    findings_by_severity: HashMap<LintSeverity, usize>,
    average_analysis_time: std::time::Duration,
    patterns_detected: HashMap<String, usize>,
}

impl EnhancedQuantumLinter {
    /// Create a new enhanced linter with default configuration
    pub fn new() -> Self {
        Self::with_config(EnhancedLintingConfig::default())
    }

    /// Create a new enhanced linter with custom configuration
    pub fn with_config(config: EnhancedLintingConfig) -> Self {
        let platform_caps = PlatformCapabilities::detect();
        let buffer_pool = Arc::new(BufferPool::new());
        let pattern_database = Arc::new(Self::build_pattern_database());

        let ml_detector = if config.enable_ml_pattern_detection {
            Some(Self::build_ml_detector())
        } else {
            None
        };

        Self {
            config,
            platform_caps,
            buffer_pool,
            pattern_database,
            ml_detector,
            cache: Arc::new(Mutex::new(LintCache {
                findings: HashMap::new(),
                analyzed_patterns: HashSet::new(),
            })),
            statistics: Arc::new(Mutex::new(LintStatistics::default())),
        }
    }

    /// Build pattern database
    fn build_pattern_database() -> PatternDatabase {
        let mut gate_patterns = HashMap::new();
        let mut algorithm_signatures = HashMap::new();
        let mut antipattern_library = Vec::new();
        let mut optimization_rules = Vec::new();

        // Add common gate patterns
        gate_patterns.insert(
            "double_hadamard".to_string(),
            CompiledPattern {
                pattern_id: "double_hadamard".to_string(),
                matcher: Box::new(|gates| {
                    gates.windows(2).any(|window| {
                        matches!(window[0].gate_type(), GateType::H)
                            && matches!(window[1].gate_type(), GateType::H)
                            && window[0].target_qubits() == window[1].target_qubits()
                    })
                }),
                min_length: 2,
                max_length: 2,
            },
        );

        // Add algorithm signatures
        algorithm_signatures.insert(
            "qft".to_string(),
            AlgorithmSignature {
                name: "Quantum Fourier Transform".to_string(),
                gate_sequence: vec![GateType::H, GateType::Rz("0.0".to_string()), GateType::CNOT],
                variations: vec![],
                required_qubits: 2,
            },
        );

        algorithm_signatures.insert(
            "grover".to_string(),
            AlgorithmSignature {
                name: "Grover's Algorithm".to_string(),
                gate_sequence: vec![GateType::H, GateType::X, GateType::Z, GateType::H],
                variations: vec![],
                required_qubits: 1,
            },
        );

        // Add anti-patterns
        antipattern_library.push(AntiPattern {
            name: "Redundant Identity".to_string(),
            description: "Gates that cancel each other out".to_string(),
            pattern: vec![
                GatePatternMatcher {
                    gate_type: Some(GateType::X),
                    qubit_count: None,
                    is_controlled: Some(false),
                    is_parameterized: Some(false),
                },
                GatePatternMatcher {
                    gate_type: Some(GateType::X),
                    qubit_count: None,
                    is_controlled: Some(false),
                    is_parameterized: Some(false),
                },
            ],
            severity: LintSeverity::Warning,
            fix_strategy: "Remove both gates".to_string(),
        });

        PatternDatabase {
            gate_patterns,
            algorithm_signatures,
            antipattern_library,
            optimization_rules,
        }
    }

    /// Build ML pattern detector
    fn build_ml_detector() -> MLPatternDetector {
        let feature_extractor = FeatureExtractor {
            feature_functions: vec![
                Box::new(|gates| vec![gates.len() as f64]),
                Box::new(|gates| {
                    let depth = Self::calculate_circuit_depth(gates);
                    vec![depth as f64]
                }),
                Box::new(|gates| {
                    let cnot_count = gates
                        .iter()
                        .filter(|g| matches!(g.gate_type(), GateType::CNOT))
                        .count();
                    vec![cnot_count as f64]
                }),
            ],
        };

        let pattern_classifier = PatternClassifier {
            classify: Box::new(|features| {
                // Simplified classifier
                if features[0] > 100.0 {
                    (LintFindingType::ExcessiveCircuitDepth, 0.8)
                } else {
                    (LintFindingType::CustomRule("Unknown".to_string()), 0.5)
                }
            }),
        };

        MLPatternDetector {
            model_type: MLModelType::Ensemble,
            feature_extractor,
            pattern_classifier,
            confidence_threshold: 0.7,
        }
    }

    /// Lint a quantum circuit
    pub fn lint_circuit(
        &self,
        circuit: &[QuantumGate],
        circuit_metadata: Option<CircuitMetadata>,
    ) -> Result<EnhancedLintingReport, QuantRS2Error> {
        let start_time = std::time::Instant::now();

        // Update statistics
        {
            let mut stats = self
                .statistics
                .lock()
                .map_err(|e| QuantRS2Error::LockPoisoned(format!("Statistics lock: {e}")))?;
            stats.total_circuits_analyzed += 1;
        }

        // Check cache if incremental linting is enabled
        let cache_key = self.compute_cache_key(circuit);
        if self.config.enable_incremental_linting {
            if let Some(cached_findings) = self.check_cache(&cache_key) {
                return Ok(self.create_report(cached_findings, start_time.elapsed()));
            }
        }

        let mut all_findings = Vec::new();

        // Run different linting passes
        if self.config.base_config.detect_performance_issues {
            all_findings.extend(self.detect_performance_issues(circuit)?);
        }

        if self.config.base_config.analyze_gate_patterns {
            all_findings.extend(self.analyze_gate_patterns(circuit)?);
        }

        if self.config.base_config.detect_quantum_antipatterns {
            all_findings.extend(self.detect_antipatterns(circuit)?);
        }

        if self.config.enable_algorithm_recognition {
            all_findings.extend(self.recognize_algorithms(circuit)?);
        }

        if self.config.enable_complexity_analysis {
            all_findings.extend(self.analyze_complexity(circuit)?);
        }

        if self.config.enable_noise_resilience_check {
            all_findings.extend(self.check_noise_resilience(circuit)?);
        }

        if self.config.enable_hardware_specific_linting {
            all_findings.extend(self.check_hardware_compatibility(circuit)?);
        }

        // Apply custom rules
        for custom_rule in &self.config.custom_rules {
            all_findings.extend(self.apply_custom_rule(circuit, custom_rule)?);
        }

        // Run ML-based detection if enabled
        if let Some(ref ml_detector) = self.ml_detector {
            all_findings.extend(self.ml_pattern_detection(circuit, ml_detector)?);
        }

        // Sort findings by severity and location
        all_findings.sort_by(|a, b| {
            b.severity.cmp(&a.severity).then_with(|| {
                let a_idx = a.location.gate_indices.first().copied().unwrap_or(0);
                let b_idx = b.location.gate_indices.first().copied().unwrap_or(0);
                a_idx.cmp(&b_idx)
            })
        });

        // Filter by severity threshold
        let filtered_findings: Vec<_> = all_findings
            .into_iter()
            .filter(|f| f.severity >= self.config.base_config.severity_threshold)
            .collect();

        // Update cache
        if self.config.enable_incremental_linting {
            self.update_cache(cache_key, filtered_findings.clone());
        }

        // Update statistics
        self.update_statistics(&filtered_findings);

        // Create report
        Ok(self.create_report(filtered_findings, start_time.elapsed()))
    }

    /// Detect performance issues
    fn detect_performance_issues(
        &self,
        circuit: &[QuantumGate],
    ) -> Result<Vec<EnhancedLintFinding>, QuantRS2Error> {
        let mut findings = Vec::new();

        // Check for redundant gates
        for (i, window) in circuit.windows(2).enumerate() {
            if Self::are_inverse_gates(&window[0], &window[1]) {
                findings.push(EnhancedLintFinding {
                    finding_type: LintFindingType::RedundantGates,
                    severity: LintSeverity::Warning,
                    location: CircuitLocation {
                        gate_indices: vec![i, i + 1],
                        qubit_indices: window[0].target_qubits().to_vec(),
                        layer: None,
                        subcircuit: None,
                    },
                    message: "Redundant gate pair detected".to_string(),
                    explanation: format!(
                        "Gates {:?} and {:?} cancel each other out",
                        window[0].gate_type(),
                        window[1].gate_type()
                    ),
                    impact: ImpactAnalysis {
                        performance_impact: PerformanceImpact::Medium,
                        error_impact: 0.0,
                        resource_impact: ResourceImpact {
                            additional_gates: -2,
                            additional_qubits: 0,
                            depth_increase: -2,
                            memory_overhead: 0.0,
                        },
                        hardware_compatibility: vec![],
                    },
                    fix_suggestions: vec![FixSuggestion {
                        description: "Remove both gates".to_string(),
                        automatic: true,
                        code_changes: vec![CodeChange {
                            operation: ChangeOperation::Delete,
                            location: CircuitLocation {
                                gate_indices: vec![i, i + 1],
                                qubit_indices: vec![],
                                layer: None,
                                subcircuit: None,
                            },
                            new_gates: None,
                        }],
                        estimated_improvement: 0.1,
                        risk_level: RiskLevel::Safe,
                    }],
                    related_findings: vec![],
                    confidence: 1.0,
                    references: vec!["Quantum Circuit Optimization Guide".to_string()],
                });
            }
        }

        // Check for gate fusion opportunities
        findings.extend(self.find_fusion_opportunities(circuit)?);

        // Check circuit depth
        let depth = Self::calculate_circuit_depth(circuit);
        if depth > 100 {
            findings.push(EnhancedLintFinding {
                finding_type: LintFindingType::ExcessiveCircuitDepth,
                severity: LintSeverity::Warning,
                location: CircuitLocation {
                    gate_indices: (0..circuit.len()).collect(),
                    qubit_indices: vec![],
                    layer: None,
                    subcircuit: None,
                },
                message: format!("Circuit depth {depth} exceeds recommended threshold"),
                explanation: "Deep circuits are more susceptible to decoherence".to_string(),
                impact: ImpactAnalysis {
                    performance_impact: PerformanceImpact::High,
                    error_impact: depth as f64 * 0.001,
                    resource_impact: ResourceImpact {
                        additional_gates: 0,
                        additional_qubits: 0,
                        depth_increase: 0,
                        memory_overhead: 0.0,
                    },
                    hardware_compatibility: vec![
                        (
                            HardwareArchitecture::IBMQ,
                            Compatibility::PartiallyCompatible,
                        ),
                        (
                            HardwareArchitecture::IonQ,
                            Compatibility::PartiallyCompatible,
                        ),
                    ],
                },
                fix_suggestions: vec![FixSuggestion {
                    description: "Consider circuit parallelization".to_string(),
                    automatic: false,
                    code_changes: vec![],
                    estimated_improvement: 0.3,
                    risk_level: RiskLevel::Medium,
                }],
                related_findings: vec![],
                confidence: 0.9,
                references: vec!["NISQ Algorithm Design".to_string()],
            });
        }

        Ok(findings)
    }

    /// Analyze gate patterns
    fn analyze_gate_patterns(
        &self,
        circuit: &[QuantumGate],
    ) -> Result<Vec<EnhancedLintFinding>, QuantRS2Error> {
        let mut findings = Vec::new();

        // Check against pattern database
        for (pattern_id, compiled_pattern) in &self.pattern_database.gate_patterns {
            if circuit.len() >= compiled_pattern.min_length {
                for i in 0..=circuit.len().saturating_sub(compiled_pattern.min_length) {
                    let slice = &circuit[i..];
                    if (compiled_pattern.matcher)(slice) {
                        findings.push(self.create_pattern_finding(pattern_id, i, slice)?);
                    }
                }
            }
        }

        Ok(findings)
    }

    /// Detect anti-patterns
    fn detect_antipatterns(
        &self,
        circuit: &[QuantumGate],
    ) -> Result<Vec<EnhancedLintFinding>, QuantRS2Error> {
        let mut findings = Vec::new();

        for antipattern in &self.pattern_database.antipattern_library {
            let pattern_len = antipattern.pattern.len();
            if circuit.len() >= pattern_len {
                for i in 0..=circuit.len() - pattern_len {
                    if Self::matches_pattern(&circuit[i..i + pattern_len], &antipattern.pattern) {
                        findings.push(EnhancedLintFinding {
                            finding_type: LintFindingType::QuantumAntiPattern,
                            severity: antipattern.severity.clone(),
                            location: CircuitLocation {
                                gate_indices: (i..i + pattern_len).collect(),
                                qubit_indices: vec![],
                                layer: None,
                                subcircuit: None,
                            },
                            message: format!("Anti-pattern detected: {}", antipattern.name),
                            explanation: antipattern.description.clone(),
                            impact: ImpactAnalysis {
                                performance_impact: PerformanceImpact::Medium,
                                error_impact: 0.05,
                                resource_impact: ResourceImpact {
                                    additional_gates: 0,
                                    additional_qubits: 0,
                                    depth_increase: 0,
                                    memory_overhead: 0.0,
                                },
                                hardware_compatibility: vec![],
                            },
                            fix_suggestions: vec![FixSuggestion {
                                description: antipattern.fix_strategy.clone(),
                                automatic: false,
                                code_changes: vec![],
                                estimated_improvement: 0.2,
                                risk_level: RiskLevel::Low,
                            }],
                            related_findings: vec![],
                            confidence: 0.95,
                            references: vec!["Quantum Programming Best Practices".to_string()],
                        });
                    }
                }
            }
        }

        Ok(findings)
    }

    /// Recognize quantum algorithms
    fn recognize_algorithms(
        &self,
        circuit: &[QuantumGate],
    ) -> Result<Vec<EnhancedLintFinding>, QuantRS2Error> {
        let mut findings = Vec::new();

        for (algo_name, signature) in &self.pattern_database.algorithm_signatures {
            if let Some(match_location) = Self::find_algorithm_signature(circuit, signature) {
                findings.push(EnhancedLintFinding {
                    finding_type: LintFindingType::CustomRule(format!("Algorithm: {algo_name}")),
                    severity: LintSeverity::Info,
                    location: CircuitLocation {
                        gate_indices: match_location,
                        qubit_indices: vec![],
                        layer: None,
                        subcircuit: None,
                    },
                    message: format!("Recognized algorithm: {}", signature.name),
                    explanation: "Standard quantum algorithm pattern detected".to_string(),
                    impact: ImpactAnalysis {
                        performance_impact: PerformanceImpact::Negligible,
                        error_impact: 0.0,
                        resource_impact: ResourceImpact {
                            additional_gates: 0,
                            additional_qubits: 0,
                            depth_increase: 0,
                            memory_overhead: 0.0,
                        },
                        hardware_compatibility: vec![],
                    },
                    fix_suggestions: vec![],
                    related_findings: vec![],
                    confidence: 0.8,
                    references: vec![format!("{} Implementation Guide", signature.name)],
                });
            }
        }

        Ok(findings)
    }

    /// Analyze circuit complexity
    fn analyze_complexity(
        &self,
        circuit: &[QuantumGate],
    ) -> Result<Vec<EnhancedLintFinding>, QuantRS2Error> {
        let mut findings = Vec::new();

        // Calculate various complexity metrics
        let gate_count = circuit.len();
        let depth = Self::calculate_circuit_depth(circuit);
        let qubit_count = Self::count_qubits(circuit);
        let cnot_count = circuit
            .iter()
            .filter(|g| matches!(g.gate_type(), GateType::CNOT))
            .count();

        // T-count (important for fault-tolerant computing)
        let t_count = circuit
            .iter()
            .filter(|g| matches!(g.gate_type(), GateType::T))
            .count();

        // Check complexity thresholds
        if t_count > 50 {
            findings.push(EnhancedLintFinding {
                finding_type: LintFindingType::CustomRule("High T-count".to_string()),
                severity: LintSeverity::Warning,
                location: CircuitLocation {
                    gate_indices: vec![],
                    qubit_indices: vec![],
                    layer: None,
                    subcircuit: None,
                },
                message: format!("High T-gate count: {t_count}"),
                explanation: "T-gates are expensive in fault-tolerant quantum computing"
                    .to_string(),
                impact: ImpactAnalysis {
                    performance_impact: PerformanceImpact::High,
                    error_impact: t_count as f64 * 0.001,
                    resource_impact: ResourceImpact {
                        additional_gates: 0,
                        additional_qubits: 0,
                        depth_increase: 0,
                        memory_overhead: 0.0,
                    },
                    hardware_compatibility: vec![],
                },
                fix_suggestions: vec![FixSuggestion {
                    description: "Consider T-gate optimization techniques".to_string(),
                    automatic: false,
                    code_changes: vec![],
                    estimated_improvement: 0.3,
                    risk_level: RiskLevel::Medium,
                }],
                related_findings: vec![],
                confidence: 1.0,
                references: vec!["T-gate Optimization in Quantum Circuits".to_string()],
            });
        }

        Ok(findings)
    }

    /// Check noise resilience
    fn check_noise_resilience(
        &self,
        circuit: &[QuantumGate],
    ) -> Result<Vec<EnhancedLintFinding>, QuantRS2Error> {
        let mut findings = Vec::new();

        // Check for long sequences without error mitigation
        let mut consecutive_gates = 0;
        let mut last_check_index = 0;

        for (i, gate) in circuit.iter().enumerate() {
            consecutive_gates += 1;

            // Simple heuristic: look for measurement or reset as error mitigation
            // Skip measurement and reset operations (not in GateType enum)
            if false {
                consecutive_gates = 0;
                last_check_index = i;
            }

            if consecutive_gates > 20 {
                findings.push(EnhancedLintFinding {
                    finding_type: LintFindingType::MissingErrorMitigation,
                    severity: LintSeverity::Warning,
                    location: CircuitLocation {
                        gate_indices: (last_check_index..i).collect(),
                        qubit_indices: vec![],
                        layer: None,
                        subcircuit: None,
                    },
                    message: "Long gate sequence without error mitigation".to_string(),
                    explanation: "Consider adding error mitigation techniques".to_string(),
                    impact: ImpactAnalysis {
                        performance_impact: PerformanceImpact::Low,
                        error_impact: 0.1,
                        resource_impact: ResourceImpact {
                            additional_gates: 5,
                            additional_qubits: 0,
                            depth_increase: 5,
                            memory_overhead: 0.0,
                        },
                        hardware_compatibility: vec![],
                    },
                    fix_suggestions: vec![FixSuggestion {
                        description: "Insert dynamical decoupling sequence".to_string(),
                        automatic: false,
                        code_changes: vec![],
                        estimated_improvement: 0.2,
                        risk_level: RiskLevel::Low,
                    }],
                    related_findings: vec![],
                    confidence: 0.7,
                    references: vec!["Quantum Error Mitigation Techniques".to_string()],
                });
                consecutive_gates = 0;
                last_check_index = i;
            }
        }

        Ok(findings)
    }

    /// Check hardware compatibility
    fn check_hardware_compatibility(
        &self,
        circuit: &[QuantumGate],
    ) -> Result<Vec<EnhancedLintFinding>, QuantRS2Error> {
        let mut findings = Vec::new();

        for architecture in &self.config.target_architectures {
            let compatibility_issues =
                self.check_architecture_compatibility(circuit, architecture)?;
            findings.extend(compatibility_issues);
        }

        Ok(findings)
    }

    /// Check compatibility with specific architecture
    fn check_architecture_compatibility(
        &self,
        circuit: &[QuantumGate],
        architecture: &HardwareArchitecture,
    ) -> Result<Vec<EnhancedLintFinding>, QuantRS2Error> {
        let mut findings = Vec::new();

        // Check gate set compatibility
        let supported_gates = Self::get_supported_gates(architecture);

        for (i, gate) in circuit.iter().enumerate() {
            if !Self::is_gate_supported(gate.gate_type(), &supported_gates) {
                findings.push(EnhancedLintFinding {
                    finding_type: LintFindingType::UnsupportedGateSet,
                    severity: LintSeverity::Error,
                    location: CircuitLocation {
                        gate_indices: vec![i],
                        qubit_indices: gate.target_qubits().to_vec(),
                        layer: None,
                        subcircuit: None,
                    },
                    message: format!(
                        "Gate {:?} not supported on {:?}",
                        gate.gate_type(),
                        architecture
                    ),
                    explanation: "This gate requires decomposition for the target hardware"
                        .to_string(),
                    impact: ImpactAnalysis {
                        performance_impact: PerformanceImpact::Medium,
                        error_impact: 0.05,
                        resource_impact: ResourceImpact {
                            additional_gates: 3,
                            additional_qubits: 0,
                            depth_increase: 3,
                            memory_overhead: 0.0,
                        },
                        hardware_compatibility: vec![(
                            *architecture,
                            Compatibility::RequiresTranspilation,
                        )],
                    },
                    fix_suggestions: vec![FixSuggestion {
                        description: "Decompose gate into native gate set".to_string(),
                        automatic: true,
                        code_changes: vec![],
                        estimated_improvement: 0.0,
                        risk_level: RiskLevel::Low,
                    }],
                    related_findings: vec![],
                    confidence: 1.0,
                    references: vec![format!("{:?} Native Gate Set", architecture)],
                });
            }
        }

        Ok(findings)
    }

    /// Apply custom lint rule
    fn apply_custom_rule(
        &self,
        circuit: &[QuantumGate],
        rule: &CustomLintRule,
    ) -> Result<Vec<EnhancedLintFinding>, QuantRS2Error> {
        let mut findings = Vec::new();

        // Apply rule pattern matching
        match &rule.pattern {
            LintPattern::GateSequence(matchers) => {
                for i in 0..=circuit.len().saturating_sub(matchers.len()) {
                    if Self::matches_pattern(&circuit[i..i + matchers.len()], matchers) {
                        findings.push(EnhancedLintFinding {
                            finding_type: LintFindingType::CustomRule(rule.name.clone()),
                            severity: rule.severity.clone(),
                            location: CircuitLocation {
                                gate_indices: (i..i + matchers.len()).collect(),
                                qubit_indices: vec![],
                                layer: None,
                                subcircuit: None,
                            },
                            message: rule.name.clone(),
                            explanation: rule.description.clone(),
                            impact: ImpactAnalysis {
                                performance_impact: PerformanceImpact::Medium,
                                error_impact: 0.0,
                                resource_impact: ResourceImpact {
                                    additional_gates: 0,
                                    additional_qubits: 0,
                                    depth_increase: 0,
                                    memory_overhead: 0.0,
                                },
                                hardware_compatibility: vec![],
                            },
                            fix_suggestions: if let Some(fix) = &rule.fix_suggestion {
                                vec![FixSuggestion {
                                    description: fix.clone(),
                                    automatic: false,
                                    code_changes: vec![],
                                    estimated_improvement: 0.1,
                                    risk_level: RiskLevel::Low,
                                }]
                            } else {
                                vec![]
                            },
                            related_findings: vec![],
                            confidence: 0.9,
                            references: vec![],
                        });
                    }
                }
            }
            _ => {} // Other pattern types not implemented in this example
        }

        Ok(findings)
    }

    /// ML-based pattern detection
    fn ml_pattern_detection(
        &self,
        circuit: &[QuantumGate],
        ml_detector: &MLPatternDetector,
    ) -> Result<Vec<EnhancedLintFinding>, QuantRS2Error> {
        let mut findings = Vec::new();

        // Extract features
        let features = ml_detector
            .feature_extractor
            .feature_functions
            .iter()
            .flat_map(|f| f(circuit))
            .collect::<Vec<_>>();

        // Classify
        let (finding_type, confidence) = (ml_detector.pattern_classifier.classify)(&features);

        if confidence >= ml_detector.confidence_threshold {
            findings.push(EnhancedLintFinding {
                finding_type,
                severity: LintSeverity::Info,
                location: CircuitLocation {
                    gate_indices: (0..circuit.len()).collect(),
                    qubit_indices: vec![],
                    layer: None,
                    subcircuit: None,
                },
                message: "ML-detected pattern".to_string(),
                explanation: "Pattern detected by machine learning model".to_string(),
                impact: ImpactAnalysis {
                    performance_impact: PerformanceImpact::Low,
                    error_impact: 0.0,
                    resource_impact: ResourceImpact {
                        additional_gates: 0,
                        additional_qubits: 0,
                        depth_increase: 0,
                        memory_overhead: 0.0,
                    },
                    hardware_compatibility: vec![],
                },
                fix_suggestions: vec![],
                related_findings: vec![],
                confidence,
                references: vec![],
            });
        }

        Ok(findings)
    }

    /// Find gate fusion opportunities
    fn find_fusion_opportunities(
        &self,
        circuit: &[QuantumGate],
    ) -> Result<Vec<EnhancedLintFinding>, QuantRS2Error> {
        let mut findings = Vec::new();

        // Look for consecutive single-qubit gates on the same qubit
        let mut i = 0;
        while i < circuit.len() {
            if Self::is_single_qubit_gate(&circuit[i]) {
                let target_qubit = circuit[i].target_qubits()[0];
                let mut j = i + 1;

                while j < circuit.len()
                    && Self::is_single_qubit_gate(&circuit[j])
                    && circuit[j].target_qubits()[0] == target_qubit
                {
                    j += 1;
                }

                if j - i > 2 {
                    findings.push(EnhancedLintFinding {
                        finding_type: LintFindingType::MissedFusionOpportunity,
                        severity: LintSeverity::Info,
                        location: CircuitLocation {
                            gate_indices: (i..j).collect(),
                            qubit_indices: vec![target_qubit],
                            layer: None,
                            subcircuit: None,
                        },
                        message: format!("{} consecutive single-qubit gates can be fused", j - i),
                        explanation: "Multiple single-qubit gates can be combined into one"
                            .to_string(),
                        impact: ImpactAnalysis {
                            performance_impact: PerformanceImpact::Low,
                            error_impact: -0.01,
                            resource_impact: ResourceImpact {
                                additional_gates: -(j as i32 - i as i32 - 1),
                                additional_qubits: 0,
                                depth_increase: -(j as i32 - i as i32 - 1),
                                memory_overhead: 0.0,
                            },
                            hardware_compatibility: vec![],
                        },
                        fix_suggestions: vec![FixSuggestion {
                            description: "Fuse gates into single unitary".to_string(),
                            automatic: true,
                            code_changes: vec![],
                            estimated_improvement: 0.15,
                            risk_level: RiskLevel::Safe,
                        }],
                        related_findings: vec![],
                        confidence: 0.95,
                        references: vec!["Gate Fusion Optimization".to_string()],
                    });
                }

                i = j;
            } else {
                i += 1;
            }
        }

        Ok(findings)
    }

    /// Helper methods
    fn are_inverse_gates(gate1: &QuantumGate, gate2: &QuantumGate) -> bool {
        use GateType::{H, S, T, X, Y, Z};

        if gate1.target_qubits() != gate2.target_qubits() {
            return false;
        }

        matches!(
            (gate1.gate_type(), gate2.gate_type()),
            (X, X) | (Y, Y) | (Z, Z) | (H, H) | (S, S) | (T, T)
        )
    }

    fn calculate_circuit_depth(circuit: &[QuantumGate]) -> usize {
        if circuit.is_empty() {
            return 0;
        }

        let max_qubit = circuit
            .iter()
            .flat_map(|g| g.target_qubits())
            .max()
            .copied()
            .unwrap_or(0);

        let mut qubit_depths = vec![0; max_qubit + 1];

        for gate in circuit {
            let max_depth = gate
                .target_qubits()
                .iter()
                .map(|&q| qubit_depths[q])
                .max()
                .unwrap_or(0);

            for &qubit in gate.target_qubits() {
                qubit_depths[qubit] = max_depth + 1;
            }

            if let Some(control_qubits) = gate.control_qubits() {
                for &qubit in control_qubits {
                    qubit_depths[qubit] = max_depth + 1;
                }
            }
        }

        *qubit_depths.iter().max().unwrap_or(&0)
    }

    fn count_qubits(circuit: &[QuantumGate]) -> usize {
        circuit
            .iter()
            .flat_map(|g| {
                let mut qubits = g.target_qubits().to_vec();
                if let Some(controls) = g.control_qubits() {
                    qubits.extend(controls);
                }
                qubits
            })
            .collect::<HashSet<_>>()
            .len()
    }

    fn matches_pattern(gates: &[QuantumGate], pattern: &[GatePatternMatcher]) -> bool {
        if gates.len() != pattern.len() {
            return false;
        }

        gates.iter().zip(pattern.iter()).all(|(gate, matcher)| {
            if let Some(expected_type) = &matcher.gate_type {
                if gate.gate_type() != expected_type {
                    return false;
                }
            }

            if let Some(expected_count) = matcher.qubit_count {
                if gate.target_qubits().len() != expected_count {
                    return false;
                }
            }

            if let Some(expected_controlled) = matcher.is_controlled {
                if gate.control_qubits().is_some() != expected_controlled {
                    return false;
                }
            }

            true
        })
    }

    fn find_algorithm_signature(
        circuit: &[QuantumGate],
        signature: &AlgorithmSignature,
    ) -> Option<Vec<usize>> {
        let sig_len = signature.gate_sequence.len();

        if sig_len > circuit.len() {
            return None;
        }

        for i in 0..=circuit.len() - sig_len {
            let matches = circuit[i..i + sig_len]
                .iter()
                .zip(&signature.gate_sequence)
                .all(|(gate, expected)| gate.gate_type() == expected);

            if matches {
                return Some((i..i + sig_len).collect());
            }
        }

        None
    }

    fn is_single_qubit_gate(gate: &QuantumGate) -> bool {
        gate.target_qubits().len() == 1 && gate.control_qubits().is_none()
    }

    fn get_supported_gates(architecture: &HardwareArchitecture) -> HashSet<GateType> {
        use GateType::{Rx, Ry, Rz, CNOT, CZ, H, S, T, X, Y, Z};

        match architecture {
            HardwareArchitecture::IBMQ => vec![
                X,
                Y,
                Z,
                H,
                S,
                T,
                Rx("0.0".to_string()),
                Ry("0.0".to_string()),
                Rz("0.0".to_string()),
                CNOT,
                CZ,
            ]
            .into_iter()
            .collect(),
            HardwareArchitecture::IonQ => vec![
                X,
                Y,
                Z,
                H,
                Rx("0.0".to_string()),
                Ry("0.0".to_string()),
                Rz("0.0".to_string()),
                CNOT,
            ]
            .into_iter()
            .collect(),
            _ => {
                // Full gate set for simulators
                vec![
                    X,
                    Y,
                    Z,
                    H,
                    S,
                    T,
                    Rx("0.0".to_string()),
                    Ry("0.0".to_string()),
                    Rz("0.0".to_string()),
                    CNOT,
                    CZ,
                ]
                .into_iter()
                .collect()
            }
        }
    }

    fn is_gate_supported(gate_type: &GateType, supported: &HashSet<GateType>) -> bool {
        use GateType::{Rx, Ry, Rz};

        match gate_type {
            Rx(_) => supported.contains(&Rx("0.0".to_string())),
            Ry(_) => supported.contains(&Ry("0.0".to_string())),
            Rz(_) => supported.contains(&Rz("0.0".to_string())),
            other => supported.contains(other),
        }
    }

    fn create_pattern_finding(
        &self,
        pattern_id: &str,
        location: usize,
        gates: &[QuantumGate],
    ) -> Result<EnhancedLintFinding, QuantRS2Error> {
        let finding_type = match pattern_id {
            "double_hadamard" => LintFindingType::RedundantGates,
            _ => LintFindingType::CustomRule(pattern_id.to_string()),
        };

        Ok(EnhancedLintFinding {
            finding_type,
            severity: LintSeverity::Warning,
            location: CircuitLocation {
                gate_indices: (location..location + 2).collect(),
                qubit_indices: gates[0].target_qubits().to_vec(),
                layer: None,
                subcircuit: None,
            },
            message: format!("Pattern '{pattern_id}' detected"),
            explanation: "This pattern can be optimized".to_string(),
            impact: ImpactAnalysis {
                performance_impact: PerformanceImpact::Low,
                error_impact: 0.0,
                resource_impact: ResourceImpact {
                    additional_gates: -1,
                    additional_qubits: 0,
                    depth_increase: -1,
                    memory_overhead: 0.0,
                },
                hardware_compatibility: vec![],
            },
            fix_suggestions: vec![],
            related_findings: vec![],
            confidence: 0.9,
            references: vec![],
        })
    }

    fn compute_cache_key(&self, circuit: &[QuantumGate]) -> String {
        format!("{circuit:?}")
    }

    fn check_cache(&self, key: &str) -> Option<Vec<EnhancedLintFinding>> {
        self.cache
            .lock()
            .ok()
            .and_then(|guard| guard.findings.get(key).cloned())
    }

    fn update_cache(&self, key: String, findings: Vec<EnhancedLintFinding>) {
        if let Ok(mut guard) = self.cache.lock() {
            guard.findings.insert(key, findings);
        }
    }

    fn update_statistics(&self, findings: &[EnhancedLintFinding]) {
        let Ok(mut stats) = self.statistics.lock() else {
            return;
        };

        stats.total_findings += findings.len();

        for finding in findings {
            *stats
                .findings_by_type
                .entry(finding.finding_type.clone())
                .or_insert(0) += 1;
            *stats
                .findings_by_severity
                .entry(finding.severity.clone())
                .or_insert(0) += 1;
        }
    }

    fn create_report(
        &self,
        findings: Vec<EnhancedLintFinding>,
        analysis_time: std::time::Duration,
    ) -> EnhancedLintingReport {
        let stats = self
            .statistics
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default();

        let summary = LintingSummary {
            total_findings: findings.len(),
            findings_by_severity: Self::count_by_severity(&findings),
            findings_by_type: Self::count_by_type(&findings),
            analysis_time,
            circuits_analyzed: stats.total_circuits_analyzed,
        };

        let metrics = QualityMetrics {
            circuit_quality_score: Self::calculate_quality_score(&findings),
            performance_score: Self::calculate_performance_score(&findings),
            hardware_readiness_score: Self::calculate_hardware_score(&findings),
            maintainability_score: Self::calculate_maintainability_score(&findings),
        };

        EnhancedLintingReport {
            summary,
            findings: findings.clone(),
            metrics,
            recommendations: self.generate_recommendations(&findings),
            statistics: stats.clone(),
        }
    }

    fn count_by_severity(findings: &[EnhancedLintFinding]) -> HashMap<LintSeverity, usize> {
        let mut counts = HashMap::new();
        for finding in findings {
            *counts.entry(finding.severity.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn count_by_type(findings: &[EnhancedLintFinding]) -> HashMap<LintFindingType, usize> {
        let mut counts = HashMap::new();
        for finding in findings {
            *counts.entry(finding.finding_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    fn calculate_quality_score(findings: &[EnhancedLintFinding]) -> f64 {
        let base_score = 100.0;
        let deductions: f64 = findings
            .iter()
            .map(|f| match f.severity {
                LintSeverity::Critical => 10.0,
                LintSeverity::Error => 5.0,
                LintSeverity::Warning => 2.0,
                LintSeverity::Info => 0.5,
            })
            .sum();

        (base_score - deductions).max(0.0)
    }

    fn calculate_performance_score(findings: &[EnhancedLintFinding]) -> f64 {
        let performance_findings = findings
            .iter()
            .filter(|f| {
                matches!(
                    f.finding_type,
                    LintFindingType::RedundantGates
                        | LintFindingType::InefficientDecomposition
                        | LintFindingType::MissedFusionOpportunity
                        | LintFindingType::SuboptimalGateOrder
                )
            })
            .count();

        (performance_findings as f64).mul_add(-5.0, 100.0)
    }

    fn calculate_hardware_score(findings: &[EnhancedLintFinding]) -> f64 {
        let hardware_findings = findings
            .iter()
            .filter(|f| {
                matches!(
                    f.finding_type,
                    LintFindingType::UnsupportedGateSet
                        | LintFindingType::ConnectivityViolation
                        | LintFindingType::ExceedsCoherenceTime
                )
            })
            .count();

        (hardware_findings as f64).mul_add(-10.0, 100.0)
    }

    fn calculate_maintainability_score(findings: &[EnhancedLintFinding]) -> f64 {
        let maintainability_findings = findings
            .iter()
            .filter(|f| {
                matches!(
                    f.finding_type,
                    LintFindingType::LackOfModularity | LintFindingType::InadequateParameterization
                )
            })
            .count();

        (maintainability_findings as f64).mul_add(-8.0, 100.0)
    }

    fn generate_recommendations(&self, findings: &[EnhancedLintFinding]) -> Vec<String> {
        let mut recommendations = Vec::new();

        let critical_count = findings
            .iter()
            .filter(|f| f.severity == LintSeverity::Critical)
            .count();

        if critical_count > 0 {
            recommendations.push(format!(
                "Address {critical_count} critical issues before deployment"
            ));
        }

        let performance_issues = findings
            .iter()
            .filter(|f| {
                matches!(
                    f.impact.performance_impact,
                    PerformanceImpact::High | PerformanceImpact::Critical
                )
            })
            .count();

        if performance_issues > 0 {
            recommendations
                .push("Consider circuit optimization to improve performance".to_string());
        }

        recommendations
    }
}

/// Circuit metadata for context-aware linting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitMetadata {
    pub name: String,
    pub version: String,
    pub target_hardware: Option<HardwareArchitecture>,
    pub algorithm_type: Option<String>,
    pub expected_depth: Option<usize>,
    pub expected_gate_count: Option<usize>,
}

/// Enhanced linting report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedLintingReport {
    pub summary: LintingSummary,
    pub findings: Vec<EnhancedLintFinding>,
    pub metrics: QualityMetrics,
    pub recommendations: Vec<String>,
    pub statistics: LintStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintingSummary {
    pub total_findings: usize,
    pub findings_by_severity: HashMap<LintSeverity, usize>,
    pub findings_by_type: HashMap<LintFindingType, usize>,
    pub analysis_time: std::time::Duration,
    pub circuits_analyzed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub circuit_quality_score: f64,
    pub performance_score: f64,
    pub hardware_readiness_score: f64,
    pub maintainability_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_linter_creation() {
        let linter = EnhancedQuantumLinter::new();
        assert!(linter.config.enable_ml_pattern_detection);
    }

    #[test]
    fn test_redundant_gates_detection() {
        let linter = EnhancedQuantumLinter::new();
        let gates = vec![
            QuantumGate::new(GateType::X, vec![0], None),
            QuantumGate::new(GateType::X, vec![0], None), // X^2 = I
        ];

        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(report
            .findings
            .iter()
            .any(|f| f.finding_type == LintFindingType::RedundantGates));
    }

    #[test]
    fn test_pattern_matching() {
        let linter = EnhancedQuantumLinter::new();
        let gates = vec![
            QuantumGate::new(GateType::H, vec![0], None),
            QuantumGate::new(GateType::H, vec![0], None), // Double Hadamard
        ];

        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(!report.findings.is_empty());
    }

    #[test]
    fn test_complexity_analysis() {
        let config = EnhancedLintingConfig {
            enable_complexity_analysis: true,
            ..Default::default()
        };
        let linter = EnhancedQuantumLinter::with_config(config);

        let mut gates = Vec::new();
        for i in 0..60 {
            gates.push(QuantumGate::new(GateType::T, vec![i % 5], None));
        }

        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(report.findings.iter().any(|f|
            matches!(f.finding_type, LintFindingType::CustomRule(ref s) if s.contains("T-count"))
        ));
    }

    #[test]
    fn test_hardware_compatibility() {
        let config = EnhancedLintingConfig {
            enable_hardware_specific_linting: true,
            target_architectures: vec![HardwareArchitecture::IBMQ],
            ..Default::default()
        };
        let linter = EnhancedQuantumLinter::with_config(config);

        let gates = vec![
            // Create a Toffoli gate using Controlled(Controlled(X))
            QuantumGate::new(
                GateType::Controlled(Box::new(GateType::Controlled(Box::new(GateType::X)))),
                vec![2],          // target qubit
                Some(vec![0, 1]), // control qubits
            ),
        ];

        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(report
            .findings
            .iter()
            .any(|f| f.finding_type == LintFindingType::UnsupportedGateSet));
    }

    #[test]
    fn test_custom_rules() {
        let custom_rule = CustomLintRule {
            name: "No X after Z".to_string(),
            description: "X gate should not follow Z gate".to_string(),
            pattern: LintPattern::GateSequence(vec![
                GatePatternMatcher {
                    gate_type: Some(GateType::Z),
                    qubit_count: None,
                    is_controlled: None,
                    is_parameterized: None,
                },
                GatePatternMatcher {
                    gate_type: Some(GateType::X),
                    qubit_count: None,
                    is_controlled: None,
                    is_parameterized: None,
                },
            ]),
            severity: LintSeverity::Warning,
            fix_suggestion: Some("Reorder gates".to_string()),
        };

        let config = EnhancedLintingConfig {
            custom_rules: vec![custom_rule],
            ..Default::default()
        };
        let linter = EnhancedQuantumLinter::with_config(config);

        let gates = vec![
            QuantumGate::new(GateType::Z, vec![0], None),
            QuantumGate::new(GateType::X, vec![0], None),
        ];

        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(report.findings.iter().any(|f|
            matches!(f.finding_type, LintFindingType::CustomRule(ref s) if s.contains("No X after Z"))
        ));
    }

    #[test]
    fn test_quality_metrics() {
        let linter = EnhancedQuantumLinter::new();
        let gates = vec![
            QuantumGate::new(GateType::H, vec![0], None),
            QuantumGate::new(GateType::CNOT, vec![0, 1], None),
        ];

        let report = linter
            .lint_circuit(&gates, None)
            .expect("Failed to lint circuit");
        assert!(report.metrics.circuit_quality_score > 0.0);
        assert!(report.metrics.performance_score > 0.0);
        assert!(report.metrics.hardware_readiness_score > 0.0);
        assert!(report.metrics.maintainability_score > 0.0);
    }
}
