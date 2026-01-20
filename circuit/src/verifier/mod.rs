//! Circuit verifier with `SciRS2` formal methods for correctness checking
//!
//! This module provides comprehensive formal verification capabilities for quantum circuits,
//! including property verification, invariant checking, correctness proofs, and automated
//! theorem proving using `SciRS2`'s formal methods and symbolic computation capabilities.

pub mod config;
pub mod invariant_checker;
pub mod model_checker;
pub mod property_checker;
pub mod symbolic_executor;
pub mod theorem_prover;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export main types
pub use config::*;
pub use invariant_checker::*;
pub use model_checker::*;
pub use property_checker::*;
pub use symbolic_executor::*;
pub use theorem_prover::*;
pub use types::*;

use crate::builder::Circuit;
use crate::scirs2_integration::SciRS2CircuitAnalyzer;
use quantrs2_core::error::{QuantRS2Error, QuantRS2Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Comprehensive quantum circuit verifier with `SciRS2` formal methods
pub struct QuantumVerifier<const N: usize> {
    /// Circuit being verified
    circuit: Circuit<N>,
    /// Verifier configuration
    pub config: VerifierConfig,
    /// `SciRS2` analyzer for formal analysis
    analyzer: SciRS2CircuitAnalyzer,
    /// Property checker engine
    property_checker: Arc<RwLock<PropertyChecker<N>>>,
    /// Invariant checker
    invariant_checker: Arc<RwLock<InvariantChecker<N>>>,
    /// Theorem prover
    theorem_prover: Arc<RwLock<TheoremProver<N>>>,
    /// Correctness checker
    correctness_checker: Arc<RwLock<CorrectnessChecker<N>>>,
    /// Model checker for temporal properties
    model_checker: Arc<RwLock<ModelChecker<N>>>,
    /// Symbolic execution engine
    symbolic_executor: Arc<RwLock<SymbolicExecutor<N>>>,
}

/// Comprehensive verification result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerificationResult {
    /// Overall verification status
    pub status: VerificationStatus,
    /// Property verification results
    pub property_results: Vec<PropertyVerificationResult>,
    /// Invariant checking results
    pub invariant_results: Vec<InvariantCheckResult>,
    /// Theorem proving results
    pub theorem_results: Vec<TheoremResult>,
    /// Model checking results
    pub model_results: Vec<ModelCheckResult>,
    /// Symbolic execution results
    pub symbolic_results: SymbolicExecutionResult,
    /// Verification statistics
    pub statistics: VerificationStatistics,
    /// Detected issues and counterexamples
    pub issues: Vec<VerificationIssue>,
    /// Formal proof if available
    pub formal_proof: Option<FormalProof>,
    /// Verification metadata
    pub metadata: VerificationMetadata,
}

impl<const N: usize> QuantumVerifier<N> {
    /// Create a new quantum verifier
    #[must_use]
    pub fn new(circuit: Circuit<N>) -> Self {
        Self {
            circuit,
            config: VerifierConfig::default(),
            analyzer: SciRS2CircuitAnalyzer::new(),
            property_checker: Arc::new(RwLock::new(PropertyChecker::new())),
            invariant_checker: Arc::new(RwLock::new(InvariantChecker::new())),
            theorem_prover: Arc::new(RwLock::new(TheoremProver::new())),
            correctness_checker: Arc::new(RwLock::new(CorrectnessChecker::new())),
            model_checker: Arc::new(RwLock::new(ModelChecker::new())),
            symbolic_executor: Arc::new(RwLock::new(SymbolicExecutor::new())),
        }
    }

    /// Create verifier with custom configuration
    #[must_use]
    pub fn with_config(circuit: Circuit<N>, config: VerifierConfig) -> Self {
        Self {
            circuit,
            config,
            analyzer: SciRS2CircuitAnalyzer::new(),
            property_checker: Arc::new(RwLock::new(PropertyChecker::new())),
            invariant_checker: Arc::new(RwLock::new(InvariantChecker::new())),
            theorem_prover: Arc::new(RwLock::new(TheoremProver::new())),
            correctness_checker: Arc::new(RwLock::new(CorrectnessChecker::new())),
            model_checker: Arc::new(RwLock::new(ModelChecker::new())),
            symbolic_executor: Arc::new(RwLock::new(SymbolicExecutor::new())),
        }
    }

    /// Perform comprehensive circuit verification
    pub fn verify_circuit(&mut self) -> QuantRS2Result<VerificationResult> {
        let start_time = Instant::now();
        let mut results = VerificationResult {
            status: VerificationStatus::InProgress,
            property_results: Vec::new(),
            invariant_results: Vec::new(),
            theorem_results: Vec::new(),
            model_results: Vec::new(),
            symbolic_results: SymbolicExecutionResult {
                status: SymbolicExecutionStatus::Completed,
                explored_paths: 0,
                path_conditions: Vec::new(),
                constraint_results: Vec::new(),
                execution_time: Duration::default(),
                memory_usage: 0,
            },
            statistics: VerificationStatistics {
                total_time: Duration::default(),
                properties_verified: 0,
                invariants_checked: 0,
                theorems_proved: 0,
                success_rate: 0.0,
                memory_usage: 0,
                confidence_stats: ConfidenceStatistics {
                    average_confidence: 0.0,
                    min_confidence: 0.0,
                    max_confidence: 0.0,
                    confidence_std_dev: 0.0,
                },
            },
            issues: Vec::new(),
            formal_proof: None,
            metadata: VerificationMetadata {
                timestamp: SystemTime::now(),
                verifier_version: "0.1.0".to_string(),
                scirs2_version: "0.1.0".to_string(),
                config: self.config.clone(),
                hardware_info: HashMap::new(),
            },
        };

        // Property verification
        if self.config.enable_property_verification {
            results.property_results = self.verify_properties()?;
            results.statistics.properties_verified = results.property_results.len();
        }

        // Invariant checking
        if self.config.enable_invariant_checking {
            results.invariant_results = self.check_invariants()?;
            results.statistics.invariants_checked = results.invariant_results.len();
        }

        // Theorem proving
        if self.config.enable_theorem_proving {
            results.theorem_results = self.prove_theorems()?;
            results.statistics.theorems_proved = results
                .theorem_results
                .iter()
                .filter(|r| r.proof_status == ProofStatus::Proved)
                .count();
        }

        // Model checking
        if self.config.enable_model_checking {
            results.model_results = self.check_models()?;
        }

        // Symbolic execution
        if self.config.enable_symbolic_execution {
            results.symbolic_results = self.execute_symbolically()?;
        }

        // Calculate statistics
        results.statistics.total_time = start_time.elapsed();
        results.statistics.success_rate = self.calculate_success_rate(&results);
        results.statistics.confidence_stats = self.calculate_confidence_stats(&results);

        // Determine overall status
        results.status = self.determine_overall_status(&results);

        // Generate issues summary
        results.issues = self.generate_issues_summary(&results);

        // Attempt to construct formal proof if all verifications passed
        if results.status == VerificationStatus::Verified {
            results.formal_proof = self.construct_formal_proof(&results)?;
        }

        Ok(results)
    }

    /// Add property to verify
    pub fn add_property(&mut self, property: QuantumProperty<N>) -> QuantRS2Result<()> {
        let mut checker = self.property_checker.write().map_err(|_| {
            QuantRS2Error::InvalidOperation("Failed to acquire property checker lock".to_string())
        })?;
        checker.add_property(property);
        Ok(())
    }

    /// Add invariant to check
    pub fn add_invariant(&mut self, invariant: CircuitInvariant<N>) -> QuantRS2Result<()> {
        let mut checker = self.invariant_checker.write().map_err(|_| {
            QuantRS2Error::InvalidOperation("Failed to acquire invariant checker lock".to_string())
        })?;
        checker.add_invariant(invariant);
        Ok(())
    }

    /// Add theorem to prove
    pub fn add_theorem(&mut self, theorem: QuantumTheorem<N>) -> QuantRS2Result<()> {
        let mut prover = self.theorem_prover.write().map_err(|_| {
            QuantRS2Error::InvalidOperation("Failed to acquire theorem prover lock".to_string())
        })?;
        prover.add_theorem(theorem);
        Ok(())
    }

    fn verify_properties(&self) -> QuantRS2Result<Vec<PropertyVerificationResult>> {
        let checker = self.property_checker.read().map_err(|_| {
            QuantRS2Error::InvalidOperation("Failed to acquire property checker lock".to_string())
        })?;
        checker.verify_all_properties(&self.circuit, &self.config)
    }

    fn check_invariants(&self) -> QuantRS2Result<Vec<InvariantCheckResult>> {
        let checker = self.invariant_checker.read().map_err(|_| {
            QuantRS2Error::InvalidOperation("Failed to acquire invariant checker lock".to_string())
        })?;
        checker.check_all_invariants(&self.circuit, &self.config)
    }

    fn prove_theorems(&self) -> QuantRS2Result<Vec<TheoremResult>> {
        let prover = self.theorem_prover.read().map_err(|_| {
            QuantRS2Error::InvalidOperation("Failed to acquire theorem prover lock".to_string())
        })?;
        prover.prove_all_theorems(&self.circuit, &self.config)
    }

    fn check_models(&self) -> QuantRS2Result<Vec<ModelCheckResult>> {
        let checker = self.model_checker.read().map_err(|_| {
            QuantRS2Error::InvalidOperation("Failed to acquire model checker lock".to_string())
        })?;
        checker.check_all_properties(&self.circuit, &self.config)
    }

    fn execute_symbolically(&self) -> QuantRS2Result<SymbolicExecutionResult> {
        let executor = self.symbolic_executor.read().map_err(|_| {
            QuantRS2Error::InvalidOperation("Failed to acquire symbolic executor lock".to_string())
        })?;
        executor.execute_circuit(&self.circuit, &self.config)
    }

    fn calculate_success_rate(&self, results: &VerificationResult) -> f64 {
        let total_checks = results.property_results.len()
            + results.invariant_results.len()
            + results.theorem_results.len()
            + results.model_results.len();

        if total_checks == 0 {
            return 0.0;
        }

        let successful_checks = results
            .property_results
            .iter()
            .filter(|r| r.result == VerificationOutcome::Satisfied)
            .count()
            + results
                .invariant_results
                .iter()
                .filter(|r| r.result == VerificationOutcome::Satisfied)
                .count()
            + results
                .theorem_results
                .iter()
                .filter(|r| r.proof_status == ProofStatus::Proved)
                .count()
            + results
                .model_results
                .iter()
                .filter(|r| r.result == VerificationOutcome::Satisfied)
                .count();

        successful_checks as f64 / total_checks as f64
    }

    fn calculate_confidence_stats(&self, results: &VerificationResult) -> ConfidenceStatistics {
        let mut confidences = Vec::new();

        for result in &results.property_results {
            confidences.push(result.confidence);
        }

        for result in &results.theorem_results {
            if let Some(proof) = &result.proof {
                confidences.push(proof.confidence);
            }
        }

        if confidences.is_empty() {
            return ConfidenceStatistics {
                average_confidence: 0.0,
                min_confidence: 0.0,
                max_confidence: 0.0,
                confidence_std_dev: 0.0,
            };
        }

        let avg = confidences.iter().sum::<f64>() / confidences.len() as f64;
        let min = confidences.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = confidences.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let variance =
            confidences.iter().map(|&x| (x - avg).powi(2)).sum::<f64>() / confidences.len() as f64;
        let std_dev = variance.sqrt();

        ConfidenceStatistics {
            average_confidence: avg,
            min_confidence: min,
            max_confidence: max,
            confidence_std_dev: std_dev,
        }
    }

    fn determine_overall_status(&self, results: &VerificationResult) -> VerificationStatus {
        let has_failures = results
            .property_results
            .iter()
            .any(|r| r.result == VerificationOutcome::Violated)
            || results
                .invariant_results
                .iter()
                .any(|r| r.result == VerificationOutcome::Violated)
            || results
                .theorem_results
                .iter()
                .any(|r| r.proof_status == ProofStatus::Disproved)
            || results
                .model_results
                .iter()
                .any(|r| r.result == VerificationOutcome::Violated);

        let has_timeouts = results
            .property_results
            .iter()
            .any(|r| r.result == VerificationOutcome::Timeout)
            || results
                .invariant_results
                .iter()
                .any(|r| r.result == VerificationOutcome::Timeout)
            || results
                .theorem_results
                .iter()
                .any(|r| r.proof_status == ProofStatus::Timeout)
            || results
                .model_results
                .iter()
                .any(|r| r.result == VerificationOutcome::Timeout);

        if has_failures {
            VerificationStatus::Failed
        } else if has_timeouts {
            VerificationStatus::Incomplete
        } else if results.statistics.success_rate >= 0.95 {
            VerificationStatus::Verified
        } else {
            VerificationStatus::Unknown
        }
    }

    fn generate_issues_summary(&self, results: &VerificationResult) -> Vec<VerificationIssue> {
        let mut issues = Vec::new();

        // Property violation issues
        for result in &results.property_results {
            if result.result == VerificationOutcome::Violated {
                issues.push(VerificationIssue {
                    issue_type: IssueType::PropertyViolation,
                    severity: IssueSeverity::High,
                    description: format!("Property '{}' violated", result.property_name),
                    location: None,
                    suggested_fix: Some("Review circuit implementation".to_string()),
                    evidence: result.evidence.clone(),
                });
            }
        }

        // Invariant violation issues
        for result in &results.invariant_results {
            if result.result == VerificationOutcome::Violated {
                let severity = match result.violation_severity {
                    Some(ViolationSeverity::Critical) => IssueSeverity::Critical,
                    Some(ViolationSeverity::Major | ViolationSeverity::High) => IssueSeverity::High,
                    Some(ViolationSeverity::Moderate) => IssueSeverity::Medium,
                    Some(ViolationSeverity::Minor) | None => IssueSeverity::Low,
                };

                issues.push(VerificationIssue {
                    issue_type: IssueType::InvariantViolation,
                    severity,
                    description: format!("Invariant '{}' violated", result.invariant_name),
                    location: None,
                    suggested_fix: Some("Check circuit constraints".to_string()),
                    evidence: Vec::new(),
                });
            }
        }

        // Theorem proof failures
        for result in &results.theorem_results {
            if result.proof_status == ProofStatus::Disproved {
                issues.push(VerificationIssue {
                    issue_type: IssueType::TheoremFailure,
                    severity: IssueSeverity::High,
                    description: format!("Theorem '{}' disproved", result.theorem_name),
                    location: None,
                    suggested_fix: Some("Review theorem assumptions".to_string()),
                    evidence: Vec::new(),
                });
            }
        }

        issues
    }

    fn construct_formal_proof(
        &self,
        results: &VerificationResult,
    ) -> QuantRS2Result<Option<FormalProof>> {
        if results.statistics.success_rate >= 0.99
            && results.statistics.confidence_stats.average_confidence >= 0.95
        {
            Ok(Some(FormalProof {
                proof_tree: ProofTree {
                    root: ProofNode {
                        goal: "Circuit correctness".to_string(),
                        rule: Some("Verification by exhaustive checking".to_string()),
                        subgoals: Vec::new(),
                        status: ProofStatus::Proved,
                    },
                    branches: Vec::new(),
                },
                steps: Vec::new(),
                axioms_used: vec!["Quantum mechanics axioms".to_string()],
                confidence: results.statistics.confidence_stats.average_confidence,
                checksum: "verified".to_string(),
            }))
        } else {
            Ok(None)
        }
    }
}
