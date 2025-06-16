//! Anyon implementations for topological quantum computing
//!
//! This module provides detailed implementations of anyons, including their
//! creation, manipulation, and tracking within topological quantum systems.

use super::{
    Anyon, FusionRuleSet, NonAbelianAnyonType, TopologicalCharge, TopologicalError,
    TopologicalResult,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Factory for creating anyons with specific properties
pub struct AnyonFactory {
    anyon_type: NonAbelianAnyonType,
    fusion_rules: FusionRuleSet,
    next_id: usize,
}

impl AnyonFactory {
    /// Create a new anyon factory
    pub fn new(anyon_type: NonAbelianAnyonType, fusion_rules: FusionRuleSet) -> Self {
        Self {
            anyon_type,
            fusion_rules,
            next_id: 0,
        }
    }

    /// Create a new anyon with specified charge
    pub fn create_anyon(
        &mut self,
        charge: TopologicalCharge,
        position: (f64, f64),
        creation_time: f64,
    ) -> Anyon {
        let anyon = Anyon {
            anyon_id: self.next_id,
            charge,
            position,
            is_qubit_part: false,
            qubit_id: None,
            creation_time,
        };
        self.next_id += 1;
        anyon
    }

    /// Create an anyon pair for a specific system
    pub fn create_anyon_pair(
        &mut self,
        charge: TopologicalCharge,
        positions: [(f64, f64); 2],
        creation_time: f64,
    ) -> TopologicalResult<(Anyon, Anyon)> {
        let anyon1 = self.create_anyon(charge.clone(), positions[0], creation_time);
        let anyon2 = self.create_anyon(charge, positions[1], creation_time);
        Ok((anyon1, anyon2))
    }
}

/// Anyon worldline tracker for monitoring anyon evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnyonWorldline {
    pub anyon_id: usize,
    pub positions: Vec<(f64, f64, f64)>, // (x, y, time)
    pub charge_history: Vec<TopologicalCharge>,
    pub interaction_history: Vec<AnyonInteraction>,
}

/// Record of interactions between anyons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnyonInteraction {
    pub interaction_id: usize,
    pub anyon1_id: usize,
    pub anyon2_id: usize,
    pub interaction_type: AnyonInteractionType,
    pub timestamp: f64,
    pub result: InteractionResult,
}

/// Types of anyon interactions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnyonInteractionType {
    /// Braiding interaction
    Braiding,
    /// Fusion interaction
    Fusion,
    /// Long-range correlation
    Correlation,
    /// Measurement interaction
    Measurement,
}

/// Result of anyon interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionResult {
    /// Phase change
    PhaseChange(f64),
    /// Fusion to new charge
    FusionProduct(TopologicalCharge),
    /// Correlation value
    Correlation(f64),
    /// Measurement outcome
    MeasurementOutcome(bool),
}

/// Anyon tracker for managing anyon lifecycles
pub struct AnyonTracker {
    worldlines: HashMap<usize, AnyonWorldline>,
    interactions: Vec<AnyonInteraction>,
    next_interaction_id: usize,
}

impl AnyonTracker {
    /// Create a new anyon tracker
    pub fn new() -> Self {
        Self {
            worldlines: HashMap::new(),
            interactions: Vec::new(),
            next_interaction_id: 0,
        }
    }

    /// Start tracking an anyon
    pub fn track_anyon(&mut self, anyon: &Anyon) {
        let worldline = AnyonWorldline {
            anyon_id: anyon.anyon_id,
            positions: vec![(anyon.position.0, anyon.position.1, anyon.creation_time)],
            charge_history: vec![anyon.charge.clone()],
            interaction_history: Vec::new(),
        };
        self.worldlines.insert(anyon.anyon_id, worldline);
    }

    /// Update anyon position
    pub fn update_position(
        &mut self,
        anyon_id: usize,
        new_position: (f64, f64),
        timestamp: f64,
    ) -> TopologicalResult<()> {
        let worldline =
            self.worldlines
                .get_mut(&anyon_id)
                .ok_or(TopologicalError::AnyonCreationFailed(format!(
                    "Anyon {} not found for position update",
                    anyon_id
                )))?;
        worldline
            .positions
            .push((new_position.0, new_position.1, timestamp));
        Ok(())
    }

    /// Record an interaction between anyons
    pub fn record_interaction(
        &mut self,
        anyon1_id: usize,
        anyon2_id: usize,
        interaction_type: AnyonInteractionType,
        result: InteractionResult,
        timestamp: f64,
    ) -> TopologicalResult<()> {
        let interaction = AnyonInteraction {
            interaction_id: self.next_interaction_id,
            anyon1_id,
            anyon2_id,
            interaction_type,
            timestamp,
            result,
        };

        // Add to global interaction list
        self.interactions.push(interaction.clone());

        // Add to individual worldlines
        if let Some(worldline1) = self.worldlines.get_mut(&anyon1_id) {
            worldline1.interaction_history.push(interaction.clone());
        }
        if let Some(worldline2) = self.worldlines.get_mut(&anyon2_id) {
            worldline2.interaction_history.push(interaction);
        }

        self.next_interaction_id += 1;
        Ok(())
    }

    /// Get worldline for specific anyon
    pub fn get_worldline(&self, anyon_id: usize) -> Option<&AnyonWorldline> {
        self.worldlines.get(&anyon_id)
    }

    /// Calculate total braiding phase for an anyon
    pub fn calculate_total_phase(&self, anyon_id: usize) -> f64 {
        if let Some(worldline) = self.worldlines.get(&anyon_id) {
            worldline
                .interaction_history
                .iter()
                .filter(|i| i.interaction_type == AnyonInteractionType::Braiding)
                .map(|i| match &i.result {
                    InteractionResult::PhaseChange(phase) => *phase,
                    _ => 0.0,
                })
                .sum()
        } else {
            0.0
        }
    }
}

/// Charge algebra operations for anyons
pub struct ChargeAlgebra {
    anyon_type: NonAbelianAnyonType,
    fusion_rules: FusionRuleSet,
}

impl ChargeAlgebra {
    /// Create a new charge algebra handler
    pub fn new(anyon_type: NonAbelianAnyonType, fusion_rules: FusionRuleSet) -> Self {
        Self {
            anyon_type,
            fusion_rules,
        }
    }

    /// Check if a charge is valid for the system
    pub fn is_valid_charge(&self, charge: &TopologicalCharge) -> bool {
        match self.anyon_type {
            NonAbelianAnyonType::Fibonacci => {
                matches!(charge.label.as_str(), "I" | "τ")
            }
            NonAbelianAnyonType::Ising => {
                matches!(charge.label.as_str(), "I" | "σ" | "ψ")
            }
            _ => true, // For other types, assume all charges are valid
        }
    }

    /// Get fusion products for two charges
    pub fn fusion_products(
        &self,
        charge1: &TopologicalCharge,
        charge2: &TopologicalCharge,
    ) -> TopologicalResult<Vec<String>> {
        let fusion_key = (charge1.label.clone(), charge2.label.clone());
        self.fusion_rules
            .rules
            .get(&fusion_key)
            .cloned()
            .ok_or(TopologicalError::FusionFailed(format!(
                "No fusion rule found for {:?}",
                fusion_key
            )))
    }

    /// Calculate quantum dimension of a charge
    pub fn quantum_dimension(&self, charge: &TopologicalCharge) -> f64 {
        match (&self.anyon_type, charge.label.as_str()) {
            (&NonAbelianAnyonType::Fibonacci, "I") => 1.0,
            (&NonAbelianAnyonType::Fibonacci, "τ") => (1.0 + 5.0_f64.sqrt()) / 2.0, // Golden ratio
            (&NonAbelianAnyonType::Ising, "I") => 1.0,
            (&NonAbelianAnyonType::Ising, "σ") => 2.0_f64.sqrt(),
            (&NonAbelianAnyonType::Ising, "ψ") => 1.0,
            _ => 1.0, // Default for unknown charges
        }
    }

    /// Calculate total quantum dimension for multiple charges
    pub fn total_quantum_dimension(&self, charges: &[TopologicalCharge]) -> f64 {
        charges
            .iter()
            .map(|charge| self.quantum_dimension(charge))
            .product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anyon_factory() {
        let fusion_rules = FusionRuleSet::fibonacci();
        let mut factory = AnyonFactory::new(NonAbelianAnyonType::Fibonacci, fusion_rules);

        let charge = TopologicalCharge::fibonacci_tau();
        let anyon = factory.create_anyon(charge, (0.0, 0.0), 0.0);

        assert_eq!(anyon.anyon_id, 0);
        assert_eq!(anyon.charge.label, "τ");
        assert_eq!(anyon.position, (0.0, 0.0));
    }

    #[test]
    fn test_anyon_tracker() {
        let mut tracker = AnyonTracker::new();
        let charge = TopologicalCharge::fibonacci_tau();
        let anyon = Anyon {
            anyon_id: 0,
            charge,
            position: (0.0, 0.0),
            is_qubit_part: false,
            qubit_id: None,
            creation_time: 0.0,
        };

        tracker.track_anyon(&anyon);
        assert!(tracker.get_worldline(0).is_some());

        tracker.update_position(0, (1.0, 1.0), 1.0).unwrap();
        let worldline = tracker.get_worldline(0).unwrap();
        assert_eq!(worldline.positions.len(), 2);
    }

    #[test]
    fn test_charge_algebra() {
        let fusion_rules = FusionRuleSet::fibonacci();
        let algebra = ChargeAlgebra::new(NonAbelianAnyonType::Fibonacci, fusion_rules);

        let tau_charge = TopologicalCharge::fibonacci_tau();
        assert!(algebra.is_valid_charge(&tau_charge));

        let quantum_dim = algebra.quantum_dimension(&tau_charge);
        assert!((quantum_dim - (1.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_interaction_recording() {
        let mut tracker = AnyonTracker::new();

        tracker
            .record_interaction(
                0,
                1,
                AnyonInteractionType::Braiding,
                InteractionResult::PhaseChange(std::f64::consts::PI / 4.0),
                1.0,
            )
            .unwrap();

        assert_eq!(tracker.interactions.len(), 1);
        assert_eq!(tracker.calculate_total_phase(0), std::f64::consts::PI / 4.0);
    }
}
