//! Common types and result structures for verification

use super::config::VerifierConfig;
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Verification status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    /// Circuit is verified correct
    Verified,
    /// Circuit has verification failures
    Failed,
    /// Verification incomplete due to timeout or resource limits
    Incomplete,
    /// Verification couldn't be performed
    Unknown,
    /// Verification in progress
    InProgress,
}

/// Verification outcome
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationOutcome {
    /// Property holds
    Satisfied,
    /// Property violated
    Violated,
    /// Cannot determine (insufficient evidence)
    Unknown,
    /// Verification timeout
    Timeout,
    /// Verification error
    Error { message: String },
}

/// Types of numerical evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Matrix norm measurement
    MatrixNorm,
    /// Eigenvalue analysis
    Eigenvalue,
    /// Fidelity measurement
    Fidelity,
    /// Entanglement measure
    Entanglement,
    /// Purity measurement
    Purity,
    /// Trace distance
    TraceDistance,
    /// Custom measurement
    Custom { name: String },
}

/// Numerical evidence for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalEvidence {
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Measured value
    pub measured_value: f64,
    /// Expected value
    pub expected_value: f64,
    /// Deviation from expected
    pub deviation: f64,
    /// Statistical p-value if applicable
    pub p_value: Option<f64>,
}

/// Error bounds for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBounds {
    /// Lower bound
    pub lower_bound: f64,
    /// Upper bound
    pub upper_bound: f64,
    /// Confidence interval
    pub confidence_interval: f64,
    /// Standard deviation
    pub standard_deviation: f64,
}

/// Violation severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Minor violation (within acceptable bounds)
    Minor,
    /// Moderate violation
    Moderate,
    /// Major violation
    Major,
    /// High severity violation
    High,
    /// Critical violation (circuit likely incorrect)
    Critical,
}

/// Proof status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofStatus {
    /// Theorem proved
    Proved,
    /// Theorem disproved
    Disproved,
    /// Proof incomplete
    Incomplete,
    /// Proof timeout
    Timeout,
    /// Proof error
    Error { message: String },
}

/// Proof steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStep {
    /// Step description
    pub description: String,
    /// Rule or axiom used
    pub rule: String,
    /// Mathematical justification
    pub justification: String,
    /// Confidence in this step
    pub confidence: f64,
}

/// Formal proof representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormalProof {
    /// Proof tree
    pub proof_tree: ProofTree,
    /// Proof steps
    pub steps: Vec<ProofStep>,
    /// Axioms used
    pub axioms_used: Vec<String>,
    /// Proof confidence
    pub confidence: f64,
    /// Verification checksum
    pub checksum: String,
}

/// Proof tree structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofTree {
    /// Root goal
    pub root: ProofNode,
    /// Proof branches
    pub branches: Vec<Self>,
}

/// Proof tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofNode {
    /// Goal statement
    pub goal: String,
    /// Applied rule
    pub rule: Option<String>,
    /// Subgoals
    pub subgoals: Vec<String>,
    /// Proof status
    pub status: ProofStatus,
}

/// Counterexample for failed proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counterexample {
    /// Input values that cause failure
    pub inputs: HashMap<String, f64>,
    /// Expected vs actual output
    pub expected_output: String,
    /// `actual_output`: String,
    pub actual_output: String,
    /// Minimal counterexample flag
    pub is_minimal: bool,
}

/// Proof complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofComplexityMetrics {
    /// Number of proof steps
    pub step_count: usize,
    /// Proof depth
    pub proof_depth: usize,
    /// Number of axioms used
    pub axiom_count: usize,
    /// Memory usage for proof
    pub memory_usage: usize,
    /// Proof verification time
    pub verification_time: Duration,
}

/// Execution trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Sequence of states
    pub states: Vec<QuantumState>,
    /// Sequence of transitions
    pub transitions: Vec<StateTransition>,
    /// Trace length
    pub length: usize,
    /// Trace properties
    pub properties: HashMap<String, f64>,
}

/// Quantum state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    /// State vector
    pub state_vector: Vec<Complex64>,
    /// State properties
    pub properties: HashMap<String, f64>,
    /// Time stamp
    pub timestamp: u64,
    /// State metadata
    pub metadata: HashMap<String, String>,
}

/// State transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Source state
    pub source: usize,
    /// Target state
    pub target: usize,
    /// Transition operation
    pub operation: String,
    /// Transition probability
    pub probability: f64,
    /// Transition time
    pub time: f64,
}

/// State space statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSpaceStatistics {
    /// Total number of states
    pub total_states: usize,
    /// Number of transitions
    pub total_transitions: usize,
    /// Maximum path length
    pub max_path_length: usize,
    /// Average path length
    pub avg_path_length: f64,
    /// State space diameter
    pub diameter: usize,
    /// Memory usage
    pub memory_usage: usize,
}

/// Test outcome
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestOutcome {
    /// Test passed
    Pass,
    /// Test failed
    Fail,
    /// Test skipped
    Skip,
    /// Test error
    Error { message: String },
}

/// Verification statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStatistics {
    /// Total verification time
    pub total_time: Duration,
    /// Number of properties verified
    pub properties_verified: usize,
    /// Number of invariants checked
    pub invariants_checked: usize,
    /// Number of theorems proved
    pub theorems_proved: usize,
    /// Success rate
    pub success_rate: f64,
    /// Memory usage
    pub memory_usage: usize,
    /// Confidence statistics
    pub confidence_stats: ConfidenceStatistics,
}

/// Confidence statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceStatistics {
    /// Average confidence
    pub average_confidence: f64,
    /// Minimum confidence
    pub min_confidence: f64,
    /// Maximum confidence
    pub max_confidence: f64,
    /// Confidence standard deviation
    pub confidence_std_dev: f64,
}

/// Verification issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationIssue {
    /// Issue type
    pub issue_type: IssueType,
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Location in circuit
    pub location: Option<CircuitLocation>,
    /// Suggested fix
    pub suggested_fix: Option<String>,
    /// Related evidence
    pub evidence: Vec<NumericalEvidence>,
}

/// Issue types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueType {
    /// Property violation
    PropertyViolation,
    /// Invariant violation
    InvariantViolation,
    /// Theorem proof failure
    TheoremFailure,
    /// Model checking failure
    ModelCheckFailure,
    /// Symbolic execution error
    SymbolicExecutionError,
    /// Numerical instability
    NumericalInstability,
    /// Performance issue
    PerformanceIssue,
}

/// Issue severity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Low severity - informational
    Low,
    /// Medium severity - potential problem
    Medium,
    /// High severity - likely problem
    High,
    /// Critical severity - definite problem
    Critical,
}

/// Circuit location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitLocation {
    /// Gate index
    pub gate_index: usize,
    /// Qubit indices
    pub qubit_indices: Vec<usize>,
    /// Circuit depth
    pub depth: usize,
}

/// Verification metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationMetadata {
    /// Verification timestamp
    pub timestamp: SystemTime,
    /// Verifier version
    pub verifier_version: String,
    /// `SciRS2` version
    pub scirs2_version: String,
    /// Verification configuration
    pub config: VerifierConfig,
    /// Hardware information
    pub hardware_info: HashMap<String, String>,
}

/// Proof strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofStrategy {
    /// Direct proof
    Direct,
    /// Proof by contradiction
    Contradiction,
    /// Proof by induction
    Induction,
    /// Case analysis
    CaseAnalysis,
    /// Symbolic computation
    SymbolicComputation,
    /// Numerical verification
    NumericalVerification,
    /// Statistical testing
    StatisticalTesting,
}

/// Error models for quantum error correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorModel {
    /// Depolarizing noise
    Depolarizing { probability: f64 },
    /// Bit flip errors
    BitFlip { probability: f64 },
    /// Phase flip errors
    PhaseFlip { probability: f64 },
    /// Amplitude damping
    AmplitudeDamping { gamma: f64 },
    /// Custom error model
    Custom {
        description: String,
        parameters: HashMap<String, f64>,
    },
}

/// Expected output for algorithm verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpectedOutput {
    /// Classical bit string
    ClassicalBits { bits: Vec<bool> },
    /// Quantum state
    QuantumState { state: Vec<Complex64> },
    /// Probability distribution
    ProbabilityDistribution { probabilities: Vec<f64> },
    /// Measurement statistics
    MeasurementStats { mean: f64, variance: f64 },
}

/// Proof obligations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofObligation {
    /// Obligation name
    pub name: String,
    /// Preconditions
    pub preconditions: Vec<String>,
    /// Postconditions
    pub postconditions: Vec<String>,
    /// Proof steps
    pub proof_steps: Vec<ProofStep>,
}

/// Test case for functional correctness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    /// Input state
    pub input: Vec<Complex64>,
    /// Expected output
    pub expected_output: Vec<Complex64>,
    /// Test description
    pub description: String,
    /// Test weight
    pub weight: f64,
}

/// Complexity class for scalability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityClass {
    /// Constant O(1)
    Constant,
    /// Logarithmic O(log n)
    Logarithmic,
    /// Linear O(n)
    Linear,
    /// Polynomial O(n^k)
    Polynomial { degree: f64 },
    /// Exponential O(2^n)
    Exponential,
}
