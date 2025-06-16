//! Fusion operations and rules for topological quantum computing
//!
//! This module implements fusion operations, F-symbols, and fusion category
//! theory for topological quantum computation with anyons.

use super::{
    Anyon, FusionRuleSet, NonAbelianAnyonType, TopologicalCharge, TopologicalError,
    TopologicalResult,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Fusion tree node representing the fusion hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionTreeNode {
    /// Input charges
    pub input_charges: Vec<TopologicalCharge>,
    /// Output charge
    pub output_charge: TopologicalCharge,
    /// Fusion channel label
    pub channel: String,
    /// Child nodes in the fusion tree
    pub children: Vec<FusionTreeNode>,
}

/// Complete fusion tree for multi-anyon fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionTree {
    /// Root node of the tree
    pub root: FusionTreeNode,
    /// Total number of anyons being fused
    pub anyon_count: usize,
    /// Final fusion result
    pub final_charge: TopologicalCharge,
}

impl FusionTree {
    /// Create a new fusion tree for a specific set of charges
    pub fn new(
        input_charges: Vec<TopologicalCharge>,
        fusion_rules: &FusionRuleSet,
    ) -> TopologicalResult<Self> {
        if input_charges.is_empty() {
            return Err(TopologicalError::FusionFailed(
                "Cannot create fusion tree with no input charges".to_string(),
            ));
        }

        let anyon_count = input_charges.len();
        let root = Self::build_fusion_tree_recursive(input_charges, fusion_rules)?;
        let final_charge = root.output_charge.clone();

        Ok(Self {
            root,
            anyon_count,
            final_charge,
        })
    }

    /// Recursively build fusion tree
    fn build_fusion_tree_recursive(
        charges: Vec<TopologicalCharge>,
        fusion_rules: &FusionRuleSet,
    ) -> TopologicalResult<FusionTreeNode> {
        if charges.len() == 1 {
            // Base case: single charge
            return Ok(FusionTreeNode {
                input_charges: charges.clone(),
                output_charge: charges[0].clone(),
                channel: "identity".to_string(),
                children: Vec::new(),
            });
        }

        if charges.len() == 2 {
            // Base case: two charges
            let fusion_key = (charges[0].label.clone(), charges[1].label.clone());
            let fusion_products =
                fusion_rules
                    .rules
                    .get(&fusion_key)
                    .ok_or(TopologicalError::FusionFailed(format!(
                        "No fusion rule found for {:?}",
                        fusion_key
                    )))?;

            // For simplicity, take the first fusion product
            let output_charge = TopologicalCharge {
                label: fusion_products[0].clone(),
                quantum_dimension: "1".to_string(), // Simplified
                scaling_dimension: 0.0,
            };

            return Ok(FusionTreeNode {
                input_charges: charges,
                output_charge,
                channel: fusion_products[0].clone(),
                children: Vec::new(),
            });
        }

        // Recursive case: more than two charges
        // Fuse first two charges, then recursively fuse the result with remaining charges
        let first_two = vec![charges[0].clone(), charges[1].clone()];
        let remaining = charges[2..].to_vec();

        let left_child = Self::build_fusion_tree_recursive(first_two, fusion_rules)?;
        let intermediate_charge = left_child.output_charge.clone();

        let mut new_charges = vec![intermediate_charge];
        new_charges.extend(remaining);

        let right_child = Self::build_fusion_tree_recursive(new_charges, fusion_rules)?;
        let final_charge = right_child.output_charge.clone();

        Ok(FusionTreeNode {
            input_charges: charges,
            output_charge: final_charge,
            channel: "composite".to_string(),
            children: vec![left_child, right_child],
        })
    }

    /// Calculate the fusion multiplicity for this tree
    pub fn fusion_multiplicity(&self) -> usize {
        self.calculate_multiplicity_recursive(&self.root)
    }

    /// Recursively calculate fusion multiplicities
    fn calculate_multiplicity_recursive(&self, node: &FusionTreeNode) -> usize {
        if node.children.is_empty() {
            return 1;
        }

        node.children
            .iter()
            .map(|child| self.calculate_multiplicity_recursive(child))
            .product()
    }
}

/// F-symbol calculator for associativity constraints
pub struct FSymbolCalculator {
    anyon_type: NonAbelianAnyonType,
    fusion_rules: FusionRuleSet,
}

impl FSymbolCalculator {
    /// Create a new F-symbol calculator
    pub fn new(anyon_type: NonAbelianAnyonType, fusion_rules: FusionRuleSet) -> Self {
        Self {
            anyon_type,
            fusion_rules,
        }
    }

    /// Calculate F-symbol for four-anyon fusion
    /// F^{abc}_{def} represents the change of basis between different fusion orders
    pub fn calculate_f_symbol(
        &self,
        a: &TopologicalCharge,
        b: &TopologicalCharge,
        c: &TopologicalCharge,
        d: &TopologicalCharge,
        e: &TopologicalCharge,
        f: &TopologicalCharge,
    ) -> TopologicalResult<f64> {
        match self.anyon_type {
            NonAbelianAnyonType::Fibonacci => self.fibonacci_f_symbol(a, b, c, d, e, f),
            NonAbelianAnyonType::Ising => self.ising_f_symbol(a, b, c, d, e, f),
            _ => {
                // Default to 1.0 for unknown types
                Ok(1.0)
            }
        }
    }

    /// Calculate Fibonacci F-symbols
    fn fibonacci_f_symbol(
        &self,
        a: &TopologicalCharge,
        b: &TopologicalCharge,
        c: &TopologicalCharge,
        d: &TopologicalCharge,
        e: &TopologicalCharge,
        f: &TopologicalCharge,
    ) -> TopologicalResult<f64> {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0; // Golden ratio
        let phi_inv = 1.0 / phi;

        // Fibonacci F-symbols (simplified subset)
        match (
            a.label.as_str(),
            b.label.as_str(),
            c.label.as_str(),
            d.label.as_str(),
            e.label.as_str(),
            f.label.as_str(),
        ) {
            ("τ", "τ", "τ", "τ", "τ", "τ") => Ok(phi_inv),
            ("τ", "τ", "τ", "I", "τ", "τ") => Ok(-phi_inv),
            ("τ", "τ", "τ", "τ", "I", "I") => Ok(phi_inv.sqrt()),
            // Add more F-symbols as needed
            _ => Ok(1.0), // Default identity
        }
    }

    /// Calculate Ising F-symbols
    fn ising_f_symbol(
        &self,
        a: &TopologicalCharge,
        b: &TopologicalCharge,
        c: &TopologicalCharge,
        d: &TopologicalCharge,
        e: &TopologicalCharge,
        f: &TopologicalCharge,
    ) -> TopologicalResult<f64> {
        // Ising F-symbols (simplified subset)
        match (
            a.label.as_str(),
            b.label.as_str(),
            c.label.as_str(),
            d.label.as_str(),
            e.label.as_str(),
            f.label.as_str(),
        ) {
            ("σ", "σ", "σ", "σ", "I", "I") => Ok(1.0 / 2.0_f64.sqrt()),
            ("σ", "σ", "σ", "σ", "ψ", "ψ") => Ok(1.0 / 2.0_f64.sqrt()),
            ("σ", "σ", "ψ", "I", "σ", "σ") => Ok(-1.0),
            // Add more F-symbols as needed
            _ => Ok(1.0), // Default identity
        }
    }
}

/// Fusion space calculator for computing fusion vector spaces
pub struct FusionSpaceCalculator {
    anyon_type: NonAbelianAnyonType,
    fusion_rules: FusionRuleSet,
}

impl FusionSpaceCalculator {
    /// Create a new fusion space calculator
    pub fn new(anyon_type: NonAbelianAnyonType, fusion_rules: FusionRuleSet) -> Self {
        Self {
            anyon_type,
            fusion_rules,
        }
    }

    /// Calculate the dimension of the fusion space
    pub fn fusion_space_dimension(
        &self,
        input_charges: &[TopologicalCharge],
        output_charge: &TopologicalCharge,
    ) -> TopologicalResult<usize> {
        if input_charges.is_empty() {
            return Ok(0);
        }

        if input_charges.len() == 1 {
            // Single anyon case
            return Ok(if input_charges[0].label == output_charge.label {
                1
            } else {
                0
            });
        }

        // For multiple anyons, build all possible fusion trees
        let trees = self.enumerate_fusion_trees(input_charges, output_charge)?;
        Ok(trees.len())
    }

    /// Enumerate all possible fusion trees for given input and output
    fn enumerate_fusion_trees(
        &self,
        input_charges: &[TopologicalCharge],
        output_charge: &TopologicalCharge,
    ) -> TopologicalResult<Vec<FusionTree>> {
        let mut trees = Vec::new();

        // For two charges, check all possible fusion outcomes
        if input_charges.len() == 2 {
            let fusion_key = (
                input_charges[0].label.clone(),
                input_charges[1].label.clone(),
            );
            if let Some(fusion_products) = self.fusion_rules.rules.get(&fusion_key) {
                for product in fusion_products {
                    if product == &output_charge.label {
                        // Create a valid fusion tree for this outcome
                        let tree_result =
                            self.create_specific_fusion_tree(input_charges, output_charge);
                        if let Ok(tree) = tree_result {
                            trees.push(tree);
                        }
                    }
                }
            }
        } else {
            // For other cases, use the simplified implementation
            if let Ok(tree) = FusionTree::new(input_charges.to_vec(), &self.fusion_rules) {
                if tree.final_charge.label == output_charge.label {
                    trees.push(tree);
                }
            }
        }

        Ok(trees)
    }

    /// Create a fusion tree with specific output charge
    fn create_specific_fusion_tree(
        &self,
        input_charges: &[TopologicalCharge],
        output_charge: &TopologicalCharge,
    ) -> TopologicalResult<FusionTree> {
        if input_charges.len() != 2 {
            return Err(TopologicalError::FusionFailed(
                "create_specific_fusion_tree only supports two input charges".to_string(),
            ));
        }

        let root = FusionTreeNode {
            input_charges: input_charges.to_vec(),
            output_charge: output_charge.clone(),
            channel: output_charge.label.clone(),
            children: Vec::new(),
        };

        Ok(FusionTree {
            root,
            anyon_count: 2,
            final_charge: output_charge.clone(),
        })
    }

    /// Calculate fusion coefficients (N-symbols)
    pub fn fusion_coefficient(
        &self,
        charge1: &TopologicalCharge,
        charge2: &TopologicalCharge,
        product: &TopologicalCharge,
    ) -> usize {
        let fusion_key = (charge1.label.clone(), charge2.label.clone());
        if let Some(products) = self.fusion_rules.rules.get(&fusion_key) {
            products.iter().filter(|&p| p == &product.label).count()
        } else {
            0
        }
    }
}

/// Fusion operation executor for performing actual fusion operations
pub struct FusionOperationExecutor {
    anyon_type: NonAbelianAnyonType,
    fusion_rules: FusionRuleSet,
    f_calculator: FSymbolCalculator,
    operation_history: Vec<FusionOperation>,
}

/// Record of a fusion operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionOperation {
    /// Operation ID
    pub operation_id: usize,
    /// Input anyon IDs
    pub input_anyons: Vec<usize>,
    /// Output anyon ID
    pub output_anyon: usize,
    /// Fusion channel used
    pub fusion_channel: String,
    /// Timestamp of operation
    pub timestamp: f64,
    /// Associated F-symbol value
    pub f_symbol: f64,
}

impl FusionOperationExecutor {
    /// Create a new fusion operation executor
    pub fn new(anyon_type: NonAbelianAnyonType, fusion_rules: FusionRuleSet) -> Self {
        let f_calculator = FSymbolCalculator::new(anyon_type.clone(), fusion_rules.clone());

        Self {
            anyon_type,
            fusion_rules,
            f_calculator,
            operation_history: Vec::new(),
        }
    }

    /// Execute a fusion operation between two anyons
    pub fn execute_binary_fusion(
        &mut self,
        anyon1: &Anyon,
        anyon2: &Anyon,
        preferred_channel: Option<&str>,
    ) -> TopologicalResult<Anyon> {
        // Get possible fusion products
        let fusion_key = (anyon1.charge.label.clone(), anyon2.charge.label.clone());
        let fusion_products =
            self.fusion_rules
                .rules
                .get(&fusion_key)
                .ok_or(TopologicalError::FusionFailed(format!(
                    "No fusion rule found for {:?}",
                    fusion_key
                )))?;

        // Select fusion channel
        let selected_channel = if let Some(channel) = preferred_channel {
            if fusion_products.contains(&channel.to_string()) {
                channel.to_string()
            } else {
                return Err(TopologicalError::FusionFailed(format!(
                    "Requested fusion channel {} not available",
                    channel
                )));
            }
        } else {
            // Take the first available channel
            fusion_products[0].clone()
        };

        // Create the fusion product anyon
        let output_charge = TopologicalCharge {
            label: selected_channel.clone(),
            quantum_dimension: "1".to_string(), // Simplified
            scaling_dimension: 0.0,             // Would be computed properly
        };

        let output_anyon = Anyon {
            anyon_id: anyon1.anyon_id.max(anyon2.anyon_id) + 1, // Simple ID assignment
            charge: output_charge,
            position: (
                (anyon1.position.0 + anyon2.position.0) / 2.0,
                (anyon1.position.1 + anyon2.position.1) / 2.0,
            ),
            is_qubit_part: false,
            qubit_id: None,
            creation_time: anyon1.creation_time.max(anyon2.creation_time),
        };

        // Calculate associated F-symbol (simplified)
        let f_symbol = 1.0; // Would be computed using F-symbol calculator

        // Record the operation
        let operation = FusionOperation {
            operation_id: self.operation_history.len(),
            input_anyons: vec![anyon1.anyon_id, anyon2.anyon_id],
            output_anyon: output_anyon.anyon_id,
            fusion_channel: selected_channel,
            timestamp: 0.0, // Would be set to current time
            f_symbol,
        };

        self.operation_history.push(operation);
        Ok(output_anyon)
    }

    /// Execute multi-anyon fusion using fusion trees
    pub fn execute_multi_fusion(
        &mut self,
        anyons: &[Anyon],
        fusion_tree: &FusionTree,
    ) -> TopologicalResult<Anyon> {
        if anyons.len() != fusion_tree.anyon_count {
            return Err(TopologicalError::FusionFailed(
                "Number of anyons doesn't match fusion tree".to_string(),
            ));
        }

        // Execute fusion according to the tree structure
        self.execute_fusion_node(&fusion_tree.root, anyons)
    }

    /// Recursively execute fusion according to tree node
    fn execute_fusion_node(
        &mut self,
        node: &FusionTreeNode,
        anyons: &[Anyon],
    ) -> TopologicalResult<Anyon> {
        if node.children.is_empty() {
            // Leaf node - return the single anyon
            if anyons.len() == 1 {
                Ok(anyons[0].clone())
            } else {
                Err(TopologicalError::FusionFailed(
                    "Leaf node with multiple anyons".to_string(),
                ))
            }
        } else if node.children.len() == 2 {
            // Binary fusion node
            let left_result = self.execute_fusion_node(&node.children[0], &anyons[0..1])?;
            let right_result = self.execute_fusion_node(&node.children[1], &anyons[1..])?;

            self.execute_binary_fusion(&left_result, &right_result, Some(&node.channel))
        } else {
            Err(TopologicalError::FusionFailed(
                "Invalid fusion tree structure".to_string(),
            ))
        }
    }

    /// Get fusion operation history
    pub fn get_operation_history(&self) -> &[FusionOperation] {
        &self.operation_history
    }

    /// Calculate total fusion probability for a set of operations
    pub fn calculate_fusion_probability(&self, operations: &[usize]) -> f64 {
        operations
            .iter()
            .filter_map(|&op_id| self.operation_history.get(op_id))
            .map(|op| op.f_symbol.abs().powi(2))
            .product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_tree_creation() {
        let charges = vec![
            TopologicalCharge::fibonacci_tau(),
            TopologicalCharge::fibonacci_tau(),
        ];
        let fusion_rules = FusionRuleSet::fibonacci();

        let tree = FusionTree::new(charges, &fusion_rules).unwrap();
        assert_eq!(tree.anyon_count, 2);
    }

    #[test]
    fn test_f_symbol_calculator() {
        let fusion_rules = FusionRuleSet::fibonacci();
        let calculator = FSymbolCalculator::new(NonAbelianAnyonType::Fibonacci, fusion_rules);

        let tau = TopologicalCharge::fibonacci_tau();
        let identity = TopologicalCharge::identity();

        let f_symbol = calculator
            .calculate_f_symbol(&tau, &tau, &tau, &tau, &tau, &tau)
            .unwrap();

        assert!(f_symbol > 0.0);
    }

    #[test]
    fn test_fusion_space_dimension() {
        let fusion_rules = FusionRuleSet::fibonacci();
        let calculator = FusionSpaceCalculator::new(NonAbelianAnyonType::Fibonacci, fusion_rules);

        let tau = TopologicalCharge::fibonacci_tau();
        let charges = vec![tau.clone(), tau.clone()];

        let dimension = calculator.fusion_space_dimension(&charges, &tau).unwrap();
        assert!(dimension > 0);
    }

    #[test]
    fn test_fusion_operation_executor() {
        let fusion_rules = FusionRuleSet::fibonacci();
        let mut executor =
            FusionOperationExecutor::new(NonAbelianAnyonType::Fibonacci, fusion_rules);

        let anyon1 = Anyon {
            anyon_id: 0,
            charge: TopologicalCharge::fibonacci_tau(),
            position: (0.0, 0.0),
            is_qubit_part: false,
            qubit_id: None,
            creation_time: 0.0,
        };

        let anyon2 = Anyon {
            anyon_id: 1,
            charge: TopologicalCharge::fibonacci_tau(),
            position: (1.0, 0.0),
            is_qubit_part: false,
            qubit_id: None,
            creation_time: 0.0,
        };

        let result = executor.execute_binary_fusion(&anyon1, &anyon2, None);
        assert!(result.is_ok());
        assert_eq!(executor.get_operation_history().len(), 1);
    }
}
