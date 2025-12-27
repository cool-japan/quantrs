//! Topological Quantum Computing Support
//!
//! This module implements support for topological quantum computers, including
//! anyonic qubits, braiding operations, fusion rules, and topological protection.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::Duration;
use thiserror::Error;

use crate::{CircuitExecutor, DeviceError, DeviceResult, QuantumDevice};
use scirs2_core::random::prelude::*;

pub mod anyons;
pub mod braiding;
pub mod device;
pub mod error_correction;
pub mod fusion;
pub mod topological_codes;

/// Errors specific to topological quantum computing
#[derive(Error, Debug)]
pub enum TopologicalError {
    #[error("Anyon creation failed: {0}")]
    AnyonCreationFailed(String),
    #[error("Invalid braiding operation: {0}")]
    InvalidBraiding(String),
    #[error("Fusion operation failed: {0}")]
    FusionFailed(String),
    #[error("Topological charge mismatch: expected {expected}, got {actual}")]
    TopologicalChargeMismatch { expected: String, actual: String },
    #[error("Insufficient anyons: needed {needed}, available {available}")]
    InsufficientAnyons { needed: usize, available: usize },
    #[error("Invalid worldline configuration: {0}")]
    InvalidWorldline(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type TopologicalResult<T> = Result<T, TopologicalError>;

/// Types of topological quantum systems
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TopologicalSystemType {
    /// Majorana fermion systems
    Majorana {
        wire_count: usize,
        coupling_strength: f64,
    },
    /// Abelian anyons (e.g., Laughlin states)
    Abelian {
        filling_factor: f64,
        braiding_group: String,
    },
    /// Non-Abelian anyons (e.g., Fibonacci, Ising)
    NonAbelian {
        anyon_type: NonAbelianAnyonType,
        fusion_rules: FusionRuleSet,
    },
    /// Parafermion systems
    Parafermion {
        order: usize,
        symmetry_group: String,
    },
}

/// Types of non-Abelian anyons
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NonAbelianAnyonType {
    /// Fibonacci anyons
    Fibonacci,
    /// Ising anyons (Majorana fermions)
    Ising,
    /// SU(2) level k anyons
    SU2 { level: usize },
    /// Metaplectic anyons
    Metaplectic,
    /// Jones-Kauffman anyons
    JonesKauffman,
}

/// Topological charge (quantum number) of an anyon
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologicalCharge {
    /// Charge label (e.g., "I", "σ", "τ" for Fibonacci)
    pub label: String,
    /// Quantum dimension
    pub quantum_dimension: String, // String to handle irrational values like φ
    /// Scaling dimension
    pub scaling_dimension: f64,
}

impl Eq for TopologicalCharge {}

impl std::hash::Hash for TopologicalCharge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.label.hash(state);
        self.quantum_dimension.hash(state);
        // For f64, we use the bit representation for hashing
        self.scaling_dimension.to_bits().hash(state);
    }
}

impl TopologicalCharge {
    /// Identity charge
    pub fn identity() -> Self {
        Self {
            label: "I".to_string(),
            quantum_dimension: "1".to_string(),
            scaling_dimension: 0.0,
        }
    }

    /// Fibonacci τ charge
    pub fn fibonacci_tau() -> Self {
        Self {
            label: "τ".to_string(),
            quantum_dimension: "φ".to_string(), // Golden ratio
            scaling_dimension: 2.0 / 5.0,
        }
    }

    /// Ising σ charge (Majorana fermion)
    pub fn ising_sigma() -> Self {
        Self {
            label: "σ".to_string(),
            quantum_dimension: "√2".to_string(),
            scaling_dimension: 1.0 / 16.0,
        }
    }

    /// Ising ψ charge (fermion)
    pub fn ising_psi() -> Self {
        Self {
            label: "ψ".to_string(),
            quantum_dimension: "1".to_string(),
            scaling_dimension: 1.0 / 2.0,
        }
    }
}

/// Individual anyon in a topological quantum system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anyon {
    /// Unique identifier
    pub anyon_id: usize,
    /// Topological charge
    pub charge: TopologicalCharge,
    /// Position in space (x, y)
    pub position: (f64, f64),
    /// Whether this anyon is part of a topological qubit
    pub is_qubit_part: bool,
    /// Associated topological qubit ID (if applicable)
    pub qubit_id: Option<usize>,
    /// Creation time (for tracking worldlines)
    pub creation_time: f64,
}

/// Topological qubit implemented using anyons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalQubit {
    /// Qubit identifier
    pub qubit_id: usize,
    /// Anyons that make up this qubit
    pub anyons: Vec<usize>, // Anyon IDs
    /// Current quantum state
    pub state: TopologicalQubitState,
    /// Fusion channel (for non-Abelian anyons)
    pub fusion_channel: Option<String>,
    /// Braiding history
    pub braiding_history: Vec<BraidingOperation>,
}

/// Quantum state of a topological qubit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalQubitState {
    /// Amplitude for |0⟩ state
    pub amplitude_0: f64,
    /// Amplitude for |1⟩ state
    pub amplitude_1: f64,
    /// Relative phase
    pub phase: f64,
    /// Topological protection factor
    pub protection_factor: f64,
}

impl TopologicalQubitState {
    /// Create |0⟩ state
    pub const fn zero() -> Self {
        Self {
            amplitude_0: 1.0,
            amplitude_1: 0.0,
            phase: 0.0,
            protection_factor: 1.0,
        }
    }

    /// Create |1⟩ state
    pub const fn one() -> Self {
        Self {
            amplitude_0: 0.0,
            amplitude_1: 1.0,
            phase: 0.0,
            protection_factor: 1.0,
        }
    }

    /// Create |+⟩ state
    pub fn plus() -> Self {
        Self {
            amplitude_0: 1.0 / (2.0_f64).sqrt(),
            amplitude_1: 1.0 / (2.0_f64).sqrt(),
            phase: 0.0,
            protection_factor: 1.0,
        }
    }

    /// Get probability of measuring |0⟩
    pub fn prob_zero(&self) -> f64 {
        self.amplitude_0 * self.amplitude_0
    }

    /// Get probability of measuring |1⟩
    pub fn prob_one(&self) -> f64 {
        self.amplitude_1 * self.amplitude_1
    }
}

/// Braiding operation between anyons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BraidingOperation {
    /// Unique operation identifier
    pub operation_id: usize,
    /// First anyon being braided
    pub anyon1: usize,
    /// Second anyon being braided
    pub anyon2: usize,
    /// Direction of braiding (clockwise or counterclockwise)
    pub direction: BraidingDirection,
    /// Number of full braids
    pub braid_count: usize,
    /// Resulting phase or fusion channel
    pub result: BraidingResult,
    /// Time when operation was performed
    pub timestamp: f64,
}

/// Direction of braiding operation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BraidingDirection {
    Clockwise,
    Counterclockwise,
}

/// Result of a braiding operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BraidingResult {
    /// Phase acquired (for Abelian anyons)
    Phase(f64),
    /// Fusion channel change (for non-Abelian anyons)
    FusionChannel(String),
    /// Unitary matrix (for complex systems)
    UnitaryMatrix(Vec<Vec<f64>>),
}

/// Fusion rules for anyon systems
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusionRuleSet {
    /// Anyon type this rule set applies to
    pub anyon_type: NonAbelianAnyonType,
    /// Fusion rules: (charge1, charge2) -> \[possible_products\]
    pub rules: HashMap<(String, String), Vec<String>>,
    /// F-symbols (associativity constraints)
    pub f_symbols: HashMap<String, f64>,
    /// R-symbols (braiding matrices)
    pub r_symbols: HashMap<String, Vec<Vec<f64>>>,
}

impl FusionRuleSet {
    /// Create Fibonacci fusion rules
    pub fn fibonacci() -> Self {
        let mut rules = HashMap::new();
        rules.insert(("I".to_string(), "I".to_string()), vec!["I".to_string()]);
        rules.insert(("I".to_string(), "τ".to_string()), vec!["τ".to_string()]);
        rules.insert(("τ".to_string(), "I".to_string()), vec!["τ".to_string()]);
        rules.insert(
            ("τ".to_string(), "τ".to_string()),
            vec!["I".to_string(), "τ".to_string()],
        );

        Self {
            anyon_type: NonAbelianAnyonType::Fibonacci,
            rules,
            f_symbols: HashMap::new(), // Would be populated with actual F-symbols
            r_symbols: HashMap::new(), // Would be populated with actual R-symbols
        }
    }

    /// Create Ising fusion rules
    pub fn ising() -> Self {
        let mut rules = HashMap::new();
        rules.insert(("I".to_string(), "I".to_string()), vec!["I".to_string()]);
        rules.insert(("I".to_string(), "σ".to_string()), vec!["σ".to_string()]);
        rules.insert(("I".to_string(), "ψ".to_string()), vec!["ψ".to_string()]);
        rules.insert(("σ".to_string(), "I".to_string()), vec!["σ".to_string()]);
        rules.insert(
            ("σ".to_string(), "σ".to_string()),
            vec!["I".to_string(), "ψ".to_string()],
        );
        rules.insert(("σ".to_string(), "ψ".to_string()), vec!["σ".to_string()]);
        rules.insert(("ψ".to_string(), "I".to_string()), vec!["ψ".to_string()]);
        rules.insert(("ψ".to_string(), "σ".to_string()), vec!["σ".to_string()]);
        rules.insert(("ψ".to_string(), "ψ".to_string()), vec!["I".to_string()]);

        Self {
            anyon_type: NonAbelianAnyonType::Ising,
            rules,
            f_symbols: HashMap::new(),
            r_symbols: HashMap::new(),
        }
    }
}

/// Topological quantum device implementation
#[derive(Debug)]
pub struct TopologicalDevice {
    /// System configuration
    pub system_type: TopologicalSystemType,
    /// Fusion rule set
    pub fusion_rules: FusionRuleSet,
    /// Current anyons in the system
    pub anyons: HashMap<usize, Anyon>,
    /// Current topological qubits
    pub qubits: HashMap<usize, TopologicalQubit>,
    /// System capabilities
    pub capabilities: TopologicalCapabilities,
    /// Next available IDs
    pub next_anyon_id: usize,
    pub next_qubit_id: usize,
    /// Current time (for worldline tracking)
    pub current_time: f64,
}

/// Capabilities of a topological quantum computer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalCapabilities {
    /// Maximum number of anyons that can be created
    pub max_anyons: usize,
    /// Maximum number of qubits
    pub max_qubits: usize,
    /// Supported anyon types
    pub supported_anyons: Vec<TopologicalCharge>,
    /// Available operations
    pub available_operations: Vec<TopologicalOperation>,
    /// Braiding fidelity
    pub braiding_fidelity: f64,
    /// Fusion fidelity
    pub fusion_fidelity: f64,
    /// Topological gap (energy scale)
    pub topological_gap: f64,
    /// Coherence length
    pub coherence_length: f64,
}

/// Types of topological operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologicalOperation {
    /// Create anyon pair
    AnyonCreation { charge_type: String },
    /// Braid two anyons
    Braiding { direction: BraidingDirection },
    /// Fuse two anyons
    Fusion,
    /// Measurement in computational basis
    Measurement,
    /// Move anyon to new position
    AnyonTransport,
}

impl TopologicalDevice {
    /// Create a new topological quantum device
    pub fn new(
        system_type: TopologicalSystemType,
        fusion_rules: FusionRuleSet,
        capabilities: TopologicalCapabilities,
    ) -> Self {
        Self {
            system_type,
            fusion_rules,
            anyons: HashMap::new(),
            qubits: HashMap::new(),
            capabilities,
            next_anyon_id: 0,
            next_qubit_id: 0,
            current_time: 0.0,
        }
    }

    /// Create a pair of anyons with opposite charges
    pub fn create_anyon_pair(
        &mut self,
        charge: TopologicalCharge,
        positions: [(f64, f64); 2],
    ) -> TopologicalResult<(usize, usize)> {
        if self.anyons.len() + 2 > self.capabilities.max_anyons {
            return Err(TopologicalError::InsufficientAnyons {
                needed: 2,
                available: self.capabilities.max_anyons - self.anyons.len(),
            });
        }

        let anyon1_id = self.next_anyon_id;
        self.next_anyon_id += 1;
        let anyon2_id = self.next_anyon_id;
        self.next_anyon_id += 1;

        let anyon1 = Anyon {
            anyon_id: anyon1_id,
            charge: charge.clone(),
            position: positions[0],
            is_qubit_part: false,
            qubit_id: None,
            creation_time: self.current_time,
        };

        // Create antiparticle (for simplicity, use same charge structure)
        let anyon2 = Anyon {
            anyon_id: anyon2_id,
            charge,
            position: positions[1],
            is_qubit_part: false,
            qubit_id: None,
            creation_time: self.current_time,
        };

        self.anyons.insert(anyon1_id, anyon1);
        self.anyons.insert(anyon2_id, anyon2);

        Ok((anyon1_id, anyon2_id))
    }

    /// Create a topological qubit from anyons
    pub fn create_topological_qubit(&mut self, anyon_ids: Vec<usize>) -> TopologicalResult<usize> {
        if self.qubits.len() >= self.capabilities.max_qubits {
            return Err(TopologicalError::InsufficientAnyons {
                needed: 1,
                available: self.capabilities.max_qubits - self.qubits.len(),
            });
        }

        // Verify all anyons exist and are not already part of a qubit
        for &anyon_id in &anyon_ids {
            if let Some(anyon) = self.anyons.get(&anyon_id) {
                if anyon.is_qubit_part {
                    return Err(TopologicalError::AnyonCreationFailed(format!(
                        "Anyon {anyon_id} is already part of a qubit"
                    )));
                }
            } else {
                return Err(TopologicalError::AnyonCreationFailed(format!(
                    "Anyon {anyon_id} not found"
                )));
            }
        }

        let qubit_id = self.next_qubit_id;
        self.next_qubit_id += 1;

        // Mark anyons as part of this qubit
        for &anyon_id in &anyon_ids {
            if let Some(anyon) = self.anyons.get_mut(&anyon_id) {
                anyon.is_qubit_part = true;
                anyon.qubit_id = Some(qubit_id);
            }
        }

        let qubit = TopologicalQubit {
            qubit_id,
            anyons: anyon_ids,
            state: TopologicalQubitState::zero(),
            fusion_channel: None,
            braiding_history: Vec::new(),
        };

        self.qubits.insert(qubit_id, qubit);
        Ok(qubit_id)
    }

    /// Perform braiding operation between two anyons
    pub fn braid_anyons(
        &mut self,
        anyon1_id: usize,
        anyon2_id: usize,
        direction: BraidingDirection,
        braid_count: usize,
    ) -> TopologicalResult<BraidingResult> {
        // Check if anyons exist
        let anyon1 = self.anyons.get(&anyon1_id).ok_or_else(|| {
            TopologicalError::InvalidBraiding(format!("Anyon {anyon1_id} not found"))
        })?;
        let anyon2 = self.anyons.get(&anyon2_id).ok_or_else(|| {
            TopologicalError::InvalidBraiding(format!("Anyon {anyon2_id} not found"))
        })?;

        // Calculate braiding result based on anyon types
        let result = self.calculate_braiding_result(
            &anyon1.charge,
            &anyon2.charge,
            &direction,
            braid_count,
        )?;

        // Create braiding operation record
        let operation = BraidingOperation {
            operation_id: self.anyons.len(), // Simple ID generation
            anyon1: anyon1_id,
            anyon2: anyon2_id,
            direction,
            braid_count,
            result: result.clone(),
            timestamp: self.current_time,
        };

        // Update qubit states if anyons are part of qubits
        if let (Some(qubit1_id), Some(qubit2_id)) = (anyon1.qubit_id, anyon2.qubit_id) {
            if qubit1_id == qubit2_id {
                // Braiding within the same qubit
                if let Some(qubit) = self.qubits.get_mut(&qubit1_id) {
                    qubit.braiding_history.push(operation);
                    Self::apply_braiding_to_qubit_state_static(qubit, &result)?;
                }
            }
        }

        self.current_time += 1.0;
        Ok(result)
    }

    /// Calculate the result of braiding two charges
    fn calculate_braiding_result(
        &self,
        charge1: &TopologicalCharge,
        charge2: &TopologicalCharge,
        direction: &BraidingDirection,
        braid_count: usize,
    ) -> TopologicalResult<BraidingResult> {
        match self.fusion_rules.anyon_type {
            NonAbelianAnyonType::Fibonacci => {
                // Fibonacci braiding statistics
                if charge1.label == "τ" && charge2.label == "τ" {
                    let phase = match direction {
                        BraidingDirection::Clockwise => -4.0 * PI / 5.0,
                        BraidingDirection::Counterclockwise => 4.0 * PI / 5.0,
                    } * braid_count as f64;
                    Ok(BraidingResult::Phase(phase))
                } else {
                    Ok(BraidingResult::Phase(0.0))
                }
            }
            NonAbelianAnyonType::Ising => {
                // Ising anyon braiding
                if charge1.label == "σ" && charge2.label == "σ" {
                    let phase = match direction {
                        BraidingDirection::Clockwise => PI / 8.0,
                        BraidingDirection::Counterclockwise => -PI / 8.0,
                    } * braid_count as f64;
                    Ok(BraidingResult::Phase(phase))
                } else {
                    Ok(BraidingResult::Phase(0.0))
                }
            }
            _ => {
                // Generic case
                Ok(BraidingResult::Phase(0.0))
            }
        }
    }

    /// Apply braiding result to qubit state
    fn apply_braiding_to_qubit_state(
        &self,
        qubit: &mut TopologicalQubit,
        result: &BraidingResult,
    ) -> TopologicalResult<()> {
        match result {
            BraidingResult::Phase(phase) => {
                qubit.state.phase += phase;
                // Apply small decoherence (topological protection is very strong)
                qubit.state.protection_factor *= 0.9999;
            }
            BraidingResult::FusionChannel(channel) => {
                qubit.fusion_channel = Some(channel.clone());
            }
            BraidingResult::UnitaryMatrix(matrix) => {
                // Apply unitary transformation to qubit state
                // This would involve matrix multiplication in practice
                qubit.state.protection_factor *= 0.9999;
            }
        }
        Ok(())
    }

    /// Static version of apply_braiding_to_qubit_state
    fn apply_braiding_to_qubit_state_static(
        qubit: &mut TopologicalQubit,
        result: &BraidingResult,
    ) -> TopologicalResult<()> {
        match result {
            BraidingResult::Phase(phase) => {
                qubit.state.phase += phase;
                // Apply small decoherence (topological protection is very strong)
                qubit.state.protection_factor *= 0.9999;
            }
            BraidingResult::FusionChannel(channel) => {
                qubit.fusion_channel = Some(channel.clone());
            }
            BraidingResult::UnitaryMatrix(matrix) => {
                // Apply unitary transformation to qubit state
                // This would involve matrix multiplication in practice
                qubit.state.protection_factor *= 0.9999;
            }
        }
        Ok(())
    }

    /// Fuse two anyons
    pub fn fuse_anyons(
        &mut self,
        anyon1_id: usize,
        anyon2_id: usize,
    ) -> TopologicalResult<Vec<String>> {
        let anyon1 = self
            .anyons
            .get(&anyon1_id)
            .ok_or_else(|| TopologicalError::FusionFailed(format!("Anyon {anyon1_id} not found")))?
            .clone();
        let anyon2 = self
            .anyons
            .get(&anyon2_id)
            .ok_or_else(|| TopologicalError::FusionFailed(format!("Anyon {anyon2_id} not found")))?
            .clone();

        // Look up fusion rules
        let fusion_key = (anyon1.charge.label.clone(), anyon2.charge.label.clone());
        let fusion_products = self.fusion_rules.rules.get(&fusion_key).ok_or_else(|| {
            TopologicalError::FusionFailed(format!("No fusion rule found for {fusion_key:?}"))
        })?;

        // For simplicity, always pick the first fusion product
        // In practice, this would be probabilistic or determined by measurement
        if let Some(product_charge) = fusion_products.first() {
            // Remove the two input anyons
            self.anyons.remove(&anyon1_id);
            self.anyons.remove(&anyon2_id);

            // Create new anyon with product charge
            let new_position = (
                f64::midpoint(anyon1.position.0, anyon2.position.0),
                f64::midpoint(anyon1.position.1, anyon2.position.1),
            );

            let product_anyon = Anyon {
                anyon_id: self.next_anyon_id,
                charge: TopologicalCharge {
                    label: product_charge.clone(),
                    quantum_dimension: "1".to_string(), // Simplified
                    scaling_dimension: 0.0,
                },
                position: new_position,
                is_qubit_part: false,
                qubit_id: None,
                creation_time: self.current_time,
            };

            self.anyons.insert(self.next_anyon_id, product_anyon);
            self.next_anyon_id += 1;
        }

        Ok(fusion_products.clone())
    }

    /// Measure a topological qubit
    pub fn measure_qubit(&mut self, qubit_id: usize) -> TopologicalResult<bool> {
        let qubit = self.qubits.get_mut(&qubit_id).ok_or_else(|| {
            TopologicalError::InvalidBraiding(format!("Qubit {qubit_id} not found"))
        })?;

        let prob_zero = qubit.state.prob_zero();

        // For deterministic cases (prob very close to 0 or 1), return deterministic result
        let measured_zero = if prob_zero >= 0.9999 {
            true // Definitely |0⟩
        } else if prob_zero <= 0.0001 {
            false // Definitely |1⟩
        } else {
            thread_rng().gen::<f64>() < prob_zero
        };

        // Apply measurement backaction
        if measured_zero {
            qubit.state = TopologicalQubitState::zero();
        } else {
            qubit.state = TopologicalQubitState::one();
        }

        // For deterministic cases, skip fidelity noise
        let final_result = if prob_zero >= 0.9999 || prob_zero <= 0.0001 {
            measured_zero // Deterministic for clear cases
        } else {
            // Measurement in topological systems has very high fidelity
            let measurement_fidelity = 0.999;
            if thread_rng().gen::<f64>() < measurement_fidelity {
                measured_zero
            } else {
                !measured_zero
            }
        };

        Ok(!final_result) // Convert to measurement convention: false=|0⟩, true=|1⟩
    }

    /// Get system status
    pub fn get_system_status(&self) -> TopologicalSystemStatus {
        TopologicalSystemStatus {
            total_anyons: self.anyons.len(),
            total_qubits: self.qubits.len(),
            system_type: self.system_type.clone(),
            topological_gap: self.capabilities.topological_gap,
            average_protection: self.calculate_average_protection(),
            current_time: self.current_time,
        }
    }

    /// Calculate average topological protection
    fn calculate_average_protection(&self) -> f64 {
        if self.qubits.is_empty() {
            return 1.0;
        }

        let total_protection: f64 = self
            .qubits
            .values()
            .map(|q| q.state.protection_factor)
            .sum();

        total_protection / self.qubits.len() as f64
    }

    /// Simulate topological evolution
    pub fn evolve(&mut self, time_step: f64) -> TopologicalResult<()> {
        self.current_time += time_step;

        // Topological systems are remarkably stable
        // Apply minimal decoherence due to finite size effects
        for qubit in self.qubits.values_mut() {
            let gap_protection = (-time_step / self.capabilities.topological_gap).exp();
            qubit.state.protection_factor *= gap_protection;
        }

        Ok(())
    }
}

/// System status for topological quantum computer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalSystemStatus {
    pub total_anyons: usize,
    pub total_qubits: usize,
    pub system_type: TopologicalSystemType,
    pub topological_gap: f64,
    pub average_protection: f64,
    pub current_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_charges() {
        let identity = TopologicalCharge::identity();
        assert_eq!(identity.label, "I");
        assert_eq!(identity.quantum_dimension, "1");

        let fibonacci_tau = TopologicalCharge::fibonacci_tau();
        assert_eq!(fibonacci_tau.label, "τ");
        assert_eq!(fibonacci_tau.quantum_dimension, "φ");
    }

    #[test]
    fn test_qubit_state_creation() {
        let zero = TopologicalQubitState::zero();
        assert_eq!(zero.prob_zero(), 1.0);
        assert_eq!(zero.prob_one(), 0.0);

        let one = TopologicalQubitState::one();
        assert_eq!(one.prob_zero(), 0.0);
        assert_eq!(one.prob_one(), 1.0);

        let plus = TopologicalQubitState::plus();
        assert!((plus.prob_zero() - 0.5).abs() < 1e-10);
        assert!((plus.prob_one() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fusion_rules() {
        let fibonacci_rules = FusionRuleSet::fibonacci();

        let tau_tau_fusion = fibonacci_rules
            .rules
            .get(&("τ".to_string(), "τ".to_string()))
            .expect("Fibonacci fusion rules should contain tau-tau fusion");
        assert!(tau_tau_fusion.contains(&"I".to_string()));
        assert!(tau_tau_fusion.contains(&"τ".to_string()));

        let ising_rules = FusionRuleSet::ising();
        let sigma_sigma_fusion = ising_rules
            .rules
            .get(&("σ".to_string(), "σ".to_string()))
            .expect("Ising fusion rules should contain sigma-sigma fusion");
        assert!(sigma_sigma_fusion.contains(&"I".to_string()));
        assert!(sigma_sigma_fusion.contains(&"ψ".to_string()));
    }

    #[test]
    fn test_topological_device_creation() {
        let system_type = TopologicalSystemType::NonAbelian {
            anyon_type: NonAbelianAnyonType::Fibonacci,
            fusion_rules: FusionRuleSet::fibonacci(),
        };

        let capabilities = TopologicalCapabilities {
            max_anyons: 100,
            max_qubits: 10,
            supported_anyons: vec![
                TopologicalCharge::identity(),
                TopologicalCharge::fibonacci_tau(),
            ],
            available_operations: vec![
                TopologicalOperation::AnyonCreation {
                    charge_type: "τ".to_string(),
                },
                TopologicalOperation::Braiding {
                    direction: BraidingDirection::Clockwise,
                },
            ],
            braiding_fidelity: 0.9999,
            fusion_fidelity: 0.999,
            topological_gap: 1.0,
            coherence_length: 100.0,
        };

        let device = TopologicalDevice::new(system_type, FusionRuleSet::fibonacci(), capabilities);

        assert_eq!(device.anyons.len(), 0);
        assert_eq!(device.qubits.len(), 0);
        assert_eq!(device.next_anyon_id, 0);
    }

    #[test]
    fn test_anyon_creation() {
        let system_type = TopologicalSystemType::NonAbelian {
            anyon_type: NonAbelianAnyonType::Fibonacci,
            fusion_rules: FusionRuleSet::fibonacci(),
        };

        let capabilities = TopologicalCapabilities {
            max_anyons: 10,
            max_qubits: 5,
            supported_anyons: vec![TopologicalCharge::fibonacci_tau()],
            available_operations: vec![],
            braiding_fidelity: 0.9999,
            fusion_fidelity: 0.999,
            topological_gap: 1.0,
            coherence_length: 100.0,
        };

        let mut device =
            TopologicalDevice::new(system_type, FusionRuleSet::fibonacci(), capabilities);

        let (anyon1_id, anyon2_id) = device
            .create_anyon_pair(TopologicalCharge::fibonacci_tau(), [(0.0, 0.0), (1.0, 0.0)])
            .expect("Anyon pair creation should succeed");

        assert_eq!(device.anyons.len(), 2);
        assert_eq!(anyon1_id, 0);
        assert_eq!(anyon2_id, 1);
    }

    #[test]
    fn test_topological_qubit_creation() {
        let system_type = TopologicalSystemType::NonAbelian {
            anyon_type: NonAbelianAnyonType::Fibonacci,
            fusion_rules: FusionRuleSet::fibonacci(),
        };

        let capabilities = TopologicalCapabilities {
            max_anyons: 10,
            max_qubits: 5,
            supported_anyons: vec![TopologicalCharge::fibonacci_tau()],
            available_operations: vec![],
            braiding_fidelity: 0.9999,
            fusion_fidelity: 0.999,
            topological_gap: 1.0,
            coherence_length: 100.0,
        };

        let mut device =
            TopologicalDevice::new(system_type, FusionRuleSet::fibonacci(), capabilities);

        // Create anyons
        let (anyon1_id, anyon2_id) = device
            .create_anyon_pair(TopologicalCharge::fibonacci_tau(), [(0.0, 0.0), (1.0, 0.0)])
            .expect("First anyon pair creation should succeed");

        let (anyon3_id, anyon4_id) = device
            .create_anyon_pair(TopologicalCharge::fibonacci_tau(), [(2.0, 0.0), (3.0, 0.0)])
            .expect("Second anyon pair creation should succeed");

        // Create topological qubit
        let qubit_id = device
            .create_topological_qubit(vec![anyon1_id, anyon2_id, anyon3_id, anyon4_id])
            .expect("Topological qubit creation should succeed");

        assert_eq!(device.qubits.len(), 1);
        assert_eq!(qubit_id, 0);

        let qubit = device
            .qubits
            .get(&qubit_id)
            .expect("Qubit should exist after creation");
        assert_eq!(qubit.anyons.len(), 4);
        assert_eq!(qubit.state.prob_zero(), 1.0);
    }

    #[test]
    fn test_braiding_operation() {
        let system_type = TopologicalSystemType::NonAbelian {
            anyon_type: NonAbelianAnyonType::Fibonacci,
            fusion_rules: FusionRuleSet::fibonacci(),
        };

        let capabilities = TopologicalCapabilities {
            max_anyons: 10,
            max_qubits: 5,
            supported_anyons: vec![TopologicalCharge::fibonacci_tau()],
            available_operations: vec![],
            braiding_fidelity: 0.9999,
            fusion_fidelity: 0.999,
            topological_gap: 1.0,
            coherence_length: 100.0,
        };

        let mut device =
            TopologicalDevice::new(system_type, FusionRuleSet::fibonacci(), capabilities);

        // Create anyons
        let (anyon1_id, anyon2_id) = device
            .create_anyon_pair(TopologicalCharge::fibonacci_tau(), [(0.0, 0.0), (1.0, 0.0)])
            .expect("Anyon pair creation should succeed");

        // Perform braiding
        let result = device
            .braid_anyons(anyon1_id, anyon2_id, BraidingDirection::Clockwise, 1)
            .expect("Braiding operation should succeed");

        match result {
            BraidingResult::Phase(phase) => {
                assert!((phase - (-4.0 * PI / 5.0)).abs() < 1e-10);
            }
            _ => panic!("Expected phase result for Fibonacci braiding"),
        }
    }

    #[test]
    fn test_measurement() {
        let system_type = TopologicalSystemType::NonAbelian {
            anyon_type: NonAbelianAnyonType::Fibonacci,
            fusion_rules: FusionRuleSet::fibonacci(),
        };

        let capabilities = TopologicalCapabilities {
            max_anyons: 10,
            max_qubits: 5,
            supported_anyons: vec![TopologicalCharge::fibonacci_tau()],
            available_operations: vec![],
            braiding_fidelity: 0.9999,
            fusion_fidelity: 0.999,
            topological_gap: 1.0,
            coherence_length: 100.0,
        };

        let mut device =
            TopologicalDevice::new(system_type, FusionRuleSet::fibonacci(), capabilities);

        // Create anyons and qubit
        let (anyon1_id, anyon2_id) = device
            .create_anyon_pair(TopologicalCharge::fibonacci_tau(), [(0.0, 0.0), (1.0, 0.0)])
            .expect("Anyon pair creation should succeed");

        let qubit_id = device
            .create_topological_qubit(vec![anyon1_id, anyon2_id])
            .expect("Topological qubit creation should succeed");

        // Measure the qubit
        let result = device
            .measure_qubit(qubit_id)
            .expect("Qubit measurement should succeed");

        // Should be false (0) since we started in |0⟩ state
        assert_eq!(result, false);

        // After measurement, state should be collapsed
        let qubit = device
            .qubits
            .get(&qubit_id)
            .expect("Qubit should exist after measurement");
        assert_eq!(qubit.state.prob_zero(), 1.0);
    }
}
