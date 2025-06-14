//! Topological quantum computing primitives
//!
//! This module provides implementations of topological quantum computing concepts
//! including anyons, braiding operations, fusion rules, and topological gates.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;

/// Type alias for fusion coefficients
type FusionCoeff = Complex64;

/// Anyon type label
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AnyonType {
    /// Unique identifier for the anyon type
    pub id: u32,
    /// String label (e.g., "1", "σ", "ψ")
    pub label: &'static str,
}

impl AnyonType {
    /// Create a new anyon type
    pub const fn new(id: u32, label: &'static str) -> Self {
        Self { id, label }
    }

    /// Vacuum (identity) anyon
    pub const VACUUM: Self = Self::new(0, "1");
}

impl fmt::Display for AnyonType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label)
    }
}

/// Anyon model definition
pub trait AnyonModel: Send + Sync {
    /// Get all anyon types in this model
    fn anyon_types(&self) -> &[AnyonType];

    /// Get quantum dimension of an anyon
    fn quantum_dimension(&self, anyon: AnyonType) -> f64;

    /// Get topological spin of an anyon
    fn topological_spin(&self, anyon: AnyonType) -> Complex64;

    /// Check if two anyons can fuse into a third
    fn can_fuse(&self, a: AnyonType, b: AnyonType, c: AnyonType) -> bool;

    /// Get fusion rules N^c_{ab}
    fn fusion_multiplicity(&self, a: AnyonType, b: AnyonType, c: AnyonType) -> u32;

    /// Get F-symbols F^{abc}_d
    fn f_symbol(
        &self,
        a: AnyonType,
        b: AnyonType,
        c: AnyonType,
        d: AnyonType,
        e: AnyonType,
        f: AnyonType,
    ) -> FusionCoeff;

    /// Get R-symbols (braiding matrices) R^{ab}_c
    fn r_symbol(&self, a: AnyonType, b: AnyonType, c: AnyonType) -> FusionCoeff;

    /// Get the name of this anyon model
    fn name(&self) -> &str;

    /// Check if the model is modular (all anyons have non-zero quantum dimension)
    fn is_modular(&self) -> bool {
        self.anyon_types()
            .iter()
            .all(|&a| self.quantum_dimension(a) > 0.0)
    }

    /// Get total quantum dimension
    fn total_quantum_dimension(&self) -> f64 {
        self.anyon_types()
            .iter()
            .map(|&a| self.quantum_dimension(a).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// Fibonacci anyon model (simplest universal model)
pub struct FibonacciModel {
    anyons: Vec<AnyonType>,
    phi: f64, // Golden ratio
}

impl FibonacciModel {
    /// Create a new Fibonacci anyon model
    pub fn new() -> Self {
        let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let anyons = vec![
            AnyonType::new(0, "1"), // Vacuum
            AnyonType::new(1, "τ"), // Fibonacci anyon
        ];

        Self { anyons, phi }
    }
}

impl Default for FibonacciModel {
    fn default() -> Self {
        Self::new()
    }
}

impl AnyonModel for FibonacciModel {
    fn anyon_types(&self) -> &[AnyonType] {
        &self.anyons
    }

    fn quantum_dimension(&self, anyon: AnyonType) -> f64 {
        match anyon.id {
            0 => 1.0,      // Vacuum
            1 => self.phi, // τ anyon
            _ => 0.0,
        }
    }

    fn topological_spin(&self, anyon: AnyonType) -> Complex64 {
        match anyon.id {
            0 => Complex64::new(1.0, 0.0),                   // Vacuum
            1 => Complex64::from_polar(1.0, 4.0 * PI / 5.0), // τ anyon
            _ => Complex64::new(0.0, 0.0),
        }
    }

    fn can_fuse(&self, a: AnyonType, b: AnyonType, c: AnyonType) -> bool {
        self.fusion_multiplicity(a, b, c) > 0
    }

    fn fusion_multiplicity(&self, a: AnyonType, b: AnyonType, c: AnyonType) -> u32 {
        match (a.id, b.id, c.id) {
            (0, x, y) | (x, 0, y) if x == y => 1, // 1 × a = a
            (1, 1, 0) => 1,                       // τ × τ = 1
            (1, 1, 1) => 1,                       // τ × τ = τ
            _ => 0,
        }
    }

    fn f_symbol(
        &self,
        a: AnyonType,
        b: AnyonType,
        c: AnyonType,
        d: AnyonType,
        e: AnyonType,
        f: AnyonType,
    ) -> FusionCoeff {
        // Simplified F-symbols for Fibonacci anyons
        // Only non-trivial case is F^{τττ}_τ
        if a.id == 1 && b.id == 1 && c.id == 1 && d.id == 1 {
            if e.id == 1 && f.id == 1 {
                // F^{τττ}_τ[τ,τ] = φ^{-1}
                Complex64::new(1.0 / self.phi, 0.0)
            } else if e.id == 1 && f.id == 0 {
                // F^{τττ}_τ[τ,1] = φ^{-1/2}
                Complex64::new(1.0 / self.phi.sqrt(), 0.0)
            } else if e.id == 0 && f.id == 1 {
                // F^{τττ}_τ[1,τ] = φ^{-1/2}
                Complex64::new(1.0 / self.phi.sqrt(), 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        } else {
            // Most F-symbols are trivial (0 or 1)
            if self.is_valid_fusion_tree(a, b, c, d, e, f) {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        }
    }

    fn r_symbol(&self, a: AnyonType, b: AnyonType, c: AnyonType) -> FusionCoeff {
        // R^{ab}_c = θ_c / (θ_a θ_b)
        if self.can_fuse(a, b, c) {
            let theta_a = self.topological_spin(a);
            let theta_b = self.topological_spin(b);
            let theta_c = self.topological_spin(c);
            let r = theta_c / (theta_a * theta_b);
            // Ensure R-symbol has unit magnitude for unitary braiding
            Complex64::from_polar(1.0, r.arg())
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    fn name(&self) -> &str {
        "Fibonacci"
    }
}

impl FibonacciModel {
    /// Check if a fusion tree is valid
    fn is_valid_fusion_tree(
        &self,
        a: AnyonType,
        b: AnyonType,
        c: AnyonType,
        d: AnyonType,
        e: AnyonType,
        f: AnyonType,
    ) -> bool {
        self.can_fuse(a, b, e)
            && self.can_fuse(e, c, d)
            && self.can_fuse(b, c, f)
            && self.can_fuse(a, f, d)
    }
}

/// Ising anyon model (used in some proposals for topological quantum computing)
pub struct IsingModel {
    anyons: Vec<AnyonType>,
}

impl IsingModel {
    /// Create a new Ising anyon model
    pub fn new() -> Self {
        let anyons = vec![
            AnyonType::new(0, "1"), // Vacuum
            AnyonType::new(1, "σ"), // Ising anyon
            AnyonType::new(2, "ψ"), // Fermion
        ];

        Self { anyons }
    }
}

impl Default for IsingModel {
    fn default() -> Self {
        Self::new()
    }
}

impl AnyonModel for IsingModel {
    fn anyon_types(&self) -> &[AnyonType] {
        &self.anyons
    }

    fn quantum_dimension(&self, anyon: AnyonType) -> f64 {
        match anyon.id {
            0 => 1.0,            // Vacuum
            1 => 2.0_f64.sqrt(), // σ anyon
            2 => 1.0,            // ψ fermion
            _ => 0.0,
        }
    }

    fn topological_spin(&self, anyon: AnyonType) -> Complex64 {
        match anyon.id {
            0 => Complex64::new(1.0, 0.0),             // Vacuum
            1 => Complex64::from_polar(1.0, PI / 8.0), // σ anyon
            2 => Complex64::new(-1.0, 0.0),            // ψ fermion
            _ => Complex64::new(0.0, 0.0),
        }
    }

    fn can_fuse(&self, a: AnyonType, b: AnyonType, c: AnyonType) -> bool {
        self.fusion_multiplicity(a, b, c) > 0
    }

    fn fusion_multiplicity(&self, a: AnyonType, b: AnyonType, c: AnyonType) -> u32 {
        match (a.id, b.id, c.id) {
            // Vacuum fusion rules
            (0, x, y) | (x, 0, y) if x == y => 1,
            // σ × σ = 1 + ψ
            (1, 1, 0) | (1, 1, 2) => 1,
            // σ × ψ = σ
            (1, 2, 1) | (2, 1, 1) => 1,
            // ψ × ψ = 1
            (2, 2, 0) => 1,
            _ => 0,
        }
    }

    fn f_symbol(
        &self,
        a: AnyonType,
        b: AnyonType,
        c: AnyonType,
        d: AnyonType,
        e: AnyonType,
        f: AnyonType,
    ) -> FusionCoeff {
        // Ising model F-symbols
        // Most non-trivial case is F^{σσσ}_σ
        if a.id == 1 && b.id == 1 && c.id == 1 && d.id == 1 {
            match (e.id, f.id) {
                (0, 0) | (2, 2) => Complex64::new(0.5, 0.0),
                (0, 2) | (2, 0) => Complex64::new(0.5, 0.0),
                _ => Complex64::new(0.0, 0.0),
            }
        } else if self.is_valid_fusion_tree(a, b, c, d, e, f) {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    fn r_symbol(&self, a: AnyonType, b: AnyonType, c: AnyonType) -> FusionCoeff {
        // Special cases for Ising model
        match (a.id, b.id, c.id) {
            // R^{σσ}_ψ = -1
            (1, 1, 2) => Complex64::new(-1.0, 0.0),
            // R^{ψψ}_1 = -1
            (2, 2, 0) => Complex64::new(-1.0, 0.0),
            // General case
            _ => {
                if self.can_fuse(a, b, c) {
                    let theta_a = self.topological_spin(a);
                    let theta_b = self.topological_spin(b);
                    let theta_c = self.topological_spin(c);
                    theta_c / (theta_a * theta_b)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            }
        }
    }

    fn name(&self) -> &str {
        "Ising"
    }
}

impl IsingModel {
    /// Check if a fusion tree is valid
    fn is_valid_fusion_tree(
        &self,
        a: AnyonType,
        b: AnyonType,
        c: AnyonType,
        d: AnyonType,
        e: AnyonType,
        f: AnyonType,
    ) -> bool {
        self.can_fuse(a, b, e)
            && self.can_fuse(e, c, d)
            && self.can_fuse(b, c, f)
            && self.can_fuse(a, f, d)
    }
}

/// Anyon worldline in spacetime
#[derive(Debug, Clone)]
pub struct AnyonWorldline {
    /// Anyon type
    pub anyon_type: AnyonType,
    /// Start position (x, y, t)
    pub start: (f64, f64, f64),
    /// End position (x, y, t)
    pub end: (f64, f64, f64),
    /// Intermediate points for braiding
    pub path: Vec<(f64, f64, f64)>,
}

/// Braiding operation between two anyons
#[derive(Debug, Clone)]
pub struct BraidingOperation {
    /// First anyon being braided
    pub anyon1: usize,
    /// Second anyon being braided
    pub anyon2: usize,
    /// Direction of braiding (true = over, false = under)
    pub over: bool,
}

/// Fusion tree representation
#[derive(Debug, Clone)]
pub struct FusionTree {
    /// External anyons (leaves)
    pub external: Vec<AnyonType>,
    /// Internal fusion channels
    pub internal: Vec<AnyonType>,
    /// Tree structure (pairs of indices to fuse)
    pub structure: Vec<(usize, usize)>,
}

impl FusionTree {
    /// Create a new fusion tree
    pub fn new(external: Vec<AnyonType>) -> Self {
        let n = external.len();
        let internal = if n > 2 {
            vec![AnyonType::VACUUM; n - 2]
        } else {
            vec![]
        };
        let structure = if n > 1 {
            (0..n - 1).map(|i| (i, i + 1)).collect()
        } else {
            vec![]
        };

        Self {
            external,
            internal,
            structure,
        }
    }

    /// Get the total charge (root of the tree)
    pub fn total_charge(&self) -> AnyonType {
        if self.internal.is_empty() {
            if self.external.is_empty() {
                AnyonType::VACUUM
            } else if self.external.len() == 1 {
                self.external[0]
            } else {
                // For 2 external anyons with no internal, this should be set explicitly
                AnyonType::VACUUM
            }
        } else {
            *self.internal.last().unwrap()
        }
    }

    /// Set the total charge for a 2-anyon tree
    pub fn set_total_charge(&mut self, charge: AnyonType) {
        if self.external.len() == 2 && self.internal.is_empty() {
            // Store the charge as metadata (we'll use a hack for now)
            // In a real implementation, we'd have a separate field
            self.structure = vec![(charge.id as usize, charge.id as usize)];
        }
    }

    /// Get the total charge for a 2-anyon tree
    pub fn get_fusion_outcome(&self) -> Option<AnyonType> {
        if self.external.len() == 2 && self.internal.is_empty() && !self.structure.is_empty() {
            let charge_id = self.structure[0].0 as u32;
            Some(AnyonType::new(
                charge_id,
                match charge_id {
                    0 => "1",
                    1 => "σ",
                    2 => "ψ",
                    _ => "τ",
                },
            ))
        } else {
            None
        }
    }
}

/// Topological quantum computer state
pub struct TopologicalQC {
    /// Anyon model being used
    model: Box<dyn AnyonModel>,
    /// Current fusion tree basis
    fusion_trees: Vec<FusionTree>,
    /// Amplitudes for each fusion tree
    amplitudes: Array1<Complex64>,
}

impl TopologicalQC {
    /// Create a new topological quantum computer
    pub fn new(model: Box<dyn AnyonModel>, anyons: Vec<AnyonType>) -> QuantRS2Result<Self> {
        // Generate all possible fusion trees
        let fusion_trees = Self::generate_fusion_trees(&*model, anyons)?;
        let n = fusion_trees.len();

        if n == 0 {
            return Err(QuantRS2Error::InvalidInput(
                "No valid fusion trees for given anyons".to_string(),
            ));
        }

        // Initialize in equal superposition
        let amplitudes = Array1::from_elem(n, Complex64::new(1.0 / (n as f64).sqrt(), 0.0));

        Ok(Self {
            model,
            fusion_trees,
            amplitudes,
        })
    }

    /// Generate all valid fusion trees for given anyons
    fn generate_fusion_trees(
        model: &dyn AnyonModel,
        anyons: Vec<AnyonType>,
    ) -> QuantRS2Result<Vec<FusionTree>> {
        if anyons.len() < 2 {
            return Ok(vec![FusionTree::new(anyons)]);
        }

        let mut trees = Vec::new();

        // For two anyons, enumerate all possible fusion channels
        if anyons.len() == 2 {
            let a = anyons[0];
            let b = anyons[1];

            // Find all possible fusion outcomes
            for c in model.anyon_types() {
                if model.can_fuse(a, b, *c) {
                    let mut tree = FusionTree::new(anyons.clone());
                    tree.set_total_charge(*c);
                    trees.push(tree);
                }
            }
        } else {
            // For simplicity, just create one tree for more than 2 anyons
            trees.push(FusionTree::new(anyons.clone()));
        }

        if trees.is_empty() {
            // If no valid fusion trees, create default
            trees.push(FusionTree::new(anyons));
        }

        Ok(trees)
    }

    /// Apply a braiding operation
    pub fn braid(&mut self, op: &BraidingOperation) -> QuantRS2Result<()> {
        // Get braiding matrix in fusion tree basis
        let braid_matrix = self.compute_braiding_matrix(op)?;

        // Apply to state
        self.amplitudes = braid_matrix.dot(&self.amplitudes);

        Ok(())
    }

    /// Compute braiding matrix in fusion tree basis
    fn compute_braiding_matrix(&self, op: &BraidingOperation) -> QuantRS2Result<Array2<Complex64>> {
        let n = self.fusion_trees.len();
        let mut matrix = Array2::zeros((n, n));

        // Simplified: diagonal R-matrix action
        for (i, tree) in self.fusion_trees.iter().enumerate() {
            if op.anyon1 < tree.external.len() && op.anyon2 < tree.external.len() {
                let a = tree.external[op.anyon1];
                let b = tree.external[op.anyon2];

                // Find fusion channel
                let c = if let Some(charge) = tree.get_fusion_outcome() {
                    charge
                } else if tree.internal.is_empty() {
                    tree.total_charge()
                } else {
                    tree.internal[0]
                };

                let r_symbol = if op.over {
                    self.model.r_symbol(a, b, c)
                } else {
                    self.model.r_symbol(a, b, c).conj()
                };

                matrix[(i, i)] = r_symbol;
            } else {
                // If indices are out of bounds, set diagonal to 1
                matrix[(i, i)] = Complex64::new(1.0, 0.0);
            }
        }

        Ok(matrix)
    }

    /// Measure topological charge
    pub fn measure_charge(&self) -> (AnyonType, f64) {
        // Find most probable total charge
        let mut charge_probs: HashMap<u32, f64> = HashMap::new();

        for (tree, &amp) in self.fusion_trees.iter().zip(&self.amplitudes) {
            let charge = if let Some(c) = tree.get_fusion_outcome() {
                c
            } else {
                tree.total_charge()
            };
            *charge_probs.entry(charge.id).or_insert(0.0) += amp.norm_sqr();
        }

        let (charge_id, prob) = charge_probs
            .into_iter()
            .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
            .unwrap_or((0, 0.0));

        let charge = self
            .model
            .anyon_types()
            .iter()
            .find(|a| a.id == charge_id)
            .copied()
            .unwrap_or(AnyonType::VACUUM);

        (charge, prob)
    }
}

/// Topological gate using anyon braiding
#[derive(Debug, Clone)]
pub struct TopologicalGate {
    /// Sequence of braiding operations
    pub braids: Vec<BraidingOperation>,
    /// Target computational basis dimension
    pub comp_dim: usize,
}

impl TopologicalGate {
    /// Create a new topological gate
    pub fn new(braids: Vec<BraidingOperation>, comp_dim: usize) -> Self {
        Self { braids, comp_dim }
    }

    /// Create a topological CNOT gate (using Ising anyons)
    pub fn cnot() -> Self {
        // Simplified braiding sequence for CNOT
        let braids = vec![
            BraidingOperation {
                anyon1: 0,
                anyon2: 1,
                over: true,
            },
            BraidingOperation {
                anyon1: 2,
                anyon2: 3,
                over: true,
            },
            BraidingOperation {
                anyon1: 1,
                anyon2: 2,
                over: false,
            },
        ];

        Self::new(braids, 4)
    }

    /// Get the unitary matrix representation
    pub fn to_matrix(&self, model: &dyn AnyonModel) -> QuantRS2Result<Array2<Complex64>> {
        // This would compute the full braiding matrix
        // For now, return identity
        Ok(Array2::eye(self.comp_dim))
    }
}

/// Kitaev toric code model
pub struct ToricCode {
    /// Lattice size (L × L)
    pub size: usize,
    /// Vertex operators A_v
    pub vertex_ops: Vec<Vec<usize>>,
    /// Plaquette operators B_p  
    pub plaquette_ops: Vec<Vec<usize>>,
}

impl ToricCode {
    /// Create a new toric code on L × L lattice
    pub fn new(size: usize) -> Self {
        let mut vertex_ops = Vec::new();
        let mut plaquette_ops = Vec::new();

        // Create vertex and plaquette operators
        // (Simplified for demonstration)
        for i in 0..size {
            for j in 0..size {
                // Vertex operator: X on all edges meeting vertex
                let v_op = vec![
                    2 * (i * size + j),     // Horizontal edge
                    2 * (i * size + j) + 1, // Vertical edge
                ];
                vertex_ops.push(v_op);

                // Plaquette operator: Z on all edges around plaquette
                let p_op = vec![
                    2 * (i * size + j),
                    2 * (i * size + (j + 1) % size),
                    2 * (((i + 1) % size) * size + j),
                    2 * (i * size + j) + 1,
                ];
                plaquette_ops.push(p_op);
            }
        }

        Self {
            size,
            vertex_ops,
            plaquette_ops,
        }
    }

    /// Get the number of physical qubits
    pub fn num_qubits(&self) -> usize {
        2 * self.size * self.size
    }

    /// Get the number of logical qubits
    pub fn num_logical_qubits(&self) -> usize {
        2 // Toric code encodes 2 logical qubits
    }

    /// Create anyonic excitations
    pub fn create_anyons(&self, vertices: &[usize], plaquettes: &[usize]) -> Vec<AnyonType> {
        let mut anyons = Vec::new();

        // e anyons (vertex violations)
        for _ in vertices {
            anyons.push(AnyonType::new(1, "e"));
        }

        // m anyons (plaquette violations)
        for _ in plaquettes {
            anyons.push(AnyonType::new(2, "m"));
        }

        anyons
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_model() {
        let model = FibonacciModel::new();

        // Test quantum dimensions
        assert_eq!(model.quantum_dimension(AnyonType::VACUUM), 1.0);
        assert!((model.quantum_dimension(AnyonType::new(1, "τ")) - 1.618).abs() < 0.001);

        // Test fusion rules
        assert_eq!(
            model.fusion_multiplicity(
                AnyonType::VACUUM,
                AnyonType::new(1, "τ"),
                AnyonType::new(1, "τ")
            ),
            1
        );

        // Test total quantum dimension
        // For Fibonacci anyons: D = sqrt(1^2 + φ^2) ≈ 2.058
        let expected_dim = (1.0 + model.phi.powi(2)).sqrt();
        assert!((model.total_quantum_dimension() - expected_dim).abs() < 0.001);
    }

    #[test]
    fn test_ising_model() {
        let model = IsingModel::new();

        // Test quantum dimensions
        assert_eq!(model.quantum_dimension(AnyonType::VACUUM), 1.0);
        assert!((model.quantum_dimension(AnyonType::new(1, "σ")) - 1.414).abs() < 0.001);
        assert_eq!(model.quantum_dimension(AnyonType::new(2, "ψ")), 1.0);

        // Test fusion rules: σ × σ = 1 + ψ
        assert_eq!(
            model.fusion_multiplicity(
                AnyonType::new(1, "σ"),
                AnyonType::new(1, "σ"),
                AnyonType::VACUUM
            ),
            1
        );
        assert_eq!(
            model.fusion_multiplicity(
                AnyonType::new(1, "σ"),
                AnyonType::new(1, "σ"),
                AnyonType::new(2, "ψ")
            ),
            1
        );
    }

    #[test]
    fn test_fusion_tree() {
        let anyons = vec![
            AnyonType::new(1, "τ"),
            AnyonType::new(1, "τ"),
            AnyonType::new(1, "τ"),
        ];

        let tree = FusionTree::new(anyons);
        assert_eq!(tree.external.len(), 3);
        assert_eq!(tree.internal.len(), 1);
    }

    #[test]
    fn test_topological_qc() {
        let model = Box::new(FibonacciModel::new());
        let anyons = vec![AnyonType::new(1, "τ"), AnyonType::new(1, "τ")];

        let qc = TopologicalQC::new(model, anyons).unwrap();
        // τ × τ = 1 + τ, so we should have 2 fusion trees
        assert_eq!(qc.fusion_trees.len(), 2);

        // Test charge measurement
        let (charge, _prob) = qc.measure_charge();
        assert!(charge.id == 0 || charge.id == 1); // Can be 1 or τ
    }

    #[test]
    fn test_toric_code() {
        let toric = ToricCode::new(4);

        assert_eq!(toric.num_qubits(), 32); // 2 * 4 * 4
        assert_eq!(toric.num_logical_qubits(), 2);

        // Test anyon creation
        let anyons = toric.create_anyons(&[0, 1], &[2]);
        assert_eq!(anyons.len(), 3);
    }

    #[test]
    fn test_braiding_operation() {
        let model = Box::new(IsingModel::new());
        let anyons = vec![AnyonType::new(1, "σ"), AnyonType::new(1, "σ")];

        let mut qc = TopologicalQC::new(model, anyons).unwrap();

        // Check initial normalization
        let initial_norm: f64 = qc.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (initial_norm - 1.0).abs() < 1e-10,
            "Initial state not normalized: {}",
            initial_norm
        );

        // Apply braiding
        let braid = BraidingOperation {
            anyon1: 0,
            anyon2: 1,
            over: true,
        };

        qc.braid(&braid).unwrap();

        // State should be normalized
        let norm: f64 = qc.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Final state not normalized: {}",
            norm
        );
    }
}
