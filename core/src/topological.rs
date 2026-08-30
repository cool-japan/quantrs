//! Topological quantum computing primitives
//!
//! This module provides implementations of topological quantum computing concepts
//! including anyons, braiding operations, fusion rules, and topological gates.

use crate::error::{QuantRS2Error, QuantRS2Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
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
        let phi = f64::midpoint(1.0, 5.0_f64.sqrt());
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
            (1, 1, 0 | 1) => 1,                   // τ × τ = 1 or τ
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

    fn name(&self) -> &'static str {
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
            0 | 2 => 1.0,        // Vacuum and ψ fermion
            1 => 2.0_f64.sqrt(), // σ anyon
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
            // σ × σ = 1 + ψ, σ × ψ = σ, ψ × ψ = 1
            (1, 1, 0 | 2) | (1, 2, 1) | (2, 1, 1) | (2, 2, 0) => 1,
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
                (0 | 2, 0 | 2) => Complex64::new(0.5, 0.0),
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
            // R^{σσ}_ψ = -1, R^{ψψ}_1 = -1
            (1, 1, 2) | (2, 2, 0) => Complex64::new(-1.0, 0.0),
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

    fn name(&self) -> &'static str {
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
            // internal is not empty in this branch, but handle gracefully
            self.internal.last().copied().unwrap_or(AnyonType::VACUUM)
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

    /// Generate all valid fusion trees for the given anyons.
    ///
    /// For two anyons every allowed fusion channel `c` (with `N^c_{ab} > 0`) yields one
    /// basis state. For `n > 2` anyons we enumerate the standard left-linear fusion-tree
    /// basis: intermediate charges `e₁ ∈ a₀×a₁`, `e₂ ∈ e₁×a₂`, …, `e_{n-1} ∈ e_{n-2}×a_{n-1}`,
    /// keeping every consistent assignment. Each assignment becomes a [`FusionTree`]
    /// whose `internal` vector holds `(e₁, …, e_{n-1})` (the last entry is the total
    /// charge / root).
    fn generate_fusion_trees(
        model: &dyn AnyonModel,
        anyons: Vec<AnyonType>,
    ) -> QuantRS2Result<Vec<FusionTree>> {
        if anyons.len() < 2 {
            return Ok(vec![FusionTree::new(anyons)]);
        }

        let mut trees = Vec::new();

        if anyons.len() == 2 {
            // Two anyons: enumerate all possible fusion channels (preserved exactly).
            let a = anyons[0];
            let b = anyons[1];
            for c in model.anyon_types() {
                if model.can_fuse(a, b, *c) {
                    let mut tree = FusionTree::new(anyons.clone());
                    tree.set_total_charge(*c);
                    trees.push(tree);
                }
            }
        } else {
            // n > 2: enumerate the left-linear fusion-tree basis by sequentially
            // fusing in each anyon and branching over every allowed intermediate
            // charge. `partial` accumulates the chain of intermediate charges.
            fn enumerate(
                model: &dyn AnyonModel,
                anyons: &[AnyonType],
                running: AnyonType,
                next: usize,
                partial: &mut Vec<AnyonType>,
                out: &mut Vec<Vec<AnyonType>>,
            ) {
                if next == anyons.len() {
                    out.push(partial.clone());
                    return;
                }
                let b = anyons[next];
                for c in model.anyon_types() {
                    if model.can_fuse(running, b, *c) {
                        partial.push(*c);
                        enumerate(model, anyons, *c, next + 1, partial, out);
                        partial.pop();
                    }
                }
            }

            let mut internals: Vec<Vec<AnyonType>> = Vec::new();
            let mut partial = Vec::new();
            // The first fusion is a₀ × a₁; branch over its outcomes, then recurse.
            for c in model.anyon_types() {
                if model.can_fuse(anyons[0], anyons[1], *c) {
                    partial.push(*c);
                    enumerate(model, &anyons, *c, 2, &mut partial, &mut internals);
                    partial.pop();
                }
            }

            for internal in internals {
                let n = anyons.len();
                let structure = (0..n - 1).map(|i| (i, i + 1)).collect();
                trees.push(FusionTree {
                    external: anyons.clone(),
                    internal,
                    structure,
                });
            }
        }

        if trees.is_empty() {
            // No allowed fusion outcome: fall back to a single default tree so callers
            // always have a basis to work with.
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
            .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap_or(std::cmp::Ordering::Equal))
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
    pub const fn new(braids: Vec<BraidingOperation>, comp_dim: usize) -> Self {
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

    /// Compute the unitary matrix representation of this braiding sequence in the
    /// fusion-tree basis of `model`.
    ///
    /// Each [`BraidingOperation`] is a generator `σ_i` (or its inverse when
    /// `over == false`) of the braid group. Its matrix in the fusion-tree basis is the
    /// diagonal action of the model's R-symbols `R^{ab}_c` on the fusion channel `c`
    /// of the braided pair `(a, b)`. The full sequence is the ordered product
    /// `B = B(braid_{k-1}) ··· B(braid_0)` (later braids act last, i.e. on the left).
    ///
    /// The anyons are materialised as `n = (max index used) + 1` copies of the model's
    /// primary non-vacuum anyon (the smallest-id anyon with quantum dimension `> 1`,
    /// falling back to the first non-vacuum type), which is the standard setting for
    /// braiding-based gates (e.g. σ anyons for the Ising model).
    ///
    /// Returns an [`QuantRS2Error::UnsupportedOperation`] if the model exposes no
    /// non-vacuum anyon to braid (so no generator matrices can be built) — never a
    /// silent identity.
    pub fn to_matrix(&self, model: &dyn AnyonModel) -> QuantRS2Result<Array2<Complex64>> {
        // Determine how many anyons the braid indices reference.
        let n_anyons = self
            .braids
            .iter()
            .map(|b| b.anyon1.max(b.anyon2) + 1)
            .max()
            .unwrap_or(0)
            .max(2);

        // Pick the anyon species to braid: prefer a non-Abelian anyon (d > 1),
        // otherwise the first non-vacuum type.
        let species = model
            .anyon_types()
            .iter()
            .filter(|a| a.id != AnyonType::VACUUM.id)
            .min_by(|a, b| {
                let da = model.quantum_dimension(**a);
                let db = model.quantum_dimension(**b);
                // Prefer larger quantum dimension (non-Abelian), then smaller id.
                db.partial_cmp(&da)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.id.cmp(&b.id))
            })
            .copied()
            .ok_or_else(|| {
                QuantRS2Error::UnsupportedOperation(
                    "anyon model exposes no non-vacuum anyon; cannot build braiding matrix"
                        .to_string(),
                )
            })?;

        let anyons = vec![species; n_anyons];

        // Build the fusion-tree basis for these anyons.
        let trees = TopologicalQC::generate_fusion_trees(model, anyons)?;
        let dim = trees.len();
        if dim == 0 {
            return Err(QuantRS2Error::UnsupportedOperation(
                "no valid fusion trees for the requested anyons; cannot build braiding matrix"
                    .to_string(),
            ));
        }

        // Start from the identity in the fusion-tree basis and apply each braid.
        let mut result = Array2::<Complex64>::eye(dim);
        for braid in &self.braids {
            let b_mat = Self::braiding_generator_matrix(model, &trees, braid);
            // Later braids act last (on the left of the accumulated product).
            result = b_mat.dot(&result);
        }

        Ok(result)
    }

    /// Build the matrix of a single braid generator in the given fusion-tree basis.
    ///
    /// The action is diagonal in this basis: braiding anyons `(a, b)` that fuse to
    /// channel `c` multiplies the amplitude by `R^{ab}_c` (or its conjugate for an
    /// under-crossing). Trees whose indices fall outside the braided pair are left
    /// invariant (diagonal entry `1`).
    fn braiding_generator_matrix(
        model: &dyn AnyonModel,
        trees: &[FusionTree],
        op: &BraidingOperation,
    ) -> Array2<Complex64> {
        let dim = trees.len();
        let mut matrix = Array2::<Complex64>::zeros((dim, dim));

        for (i, tree) in trees.iter().enumerate() {
            if op.anyon1 < tree.external.len() && op.anyon2 < tree.external.len() {
                let a = tree.external[op.anyon1];
                let b = tree.external[op.anyon2];

                let c = if let Some(charge) = tree.get_fusion_outcome() {
                    charge
                } else if tree.internal.is_empty() {
                    tree.total_charge()
                } else {
                    tree.internal[0]
                };

                let r_symbol = if op.over {
                    model.r_symbol(a, b, c)
                } else {
                    model.r_symbol(a, b, c).conj()
                };

                // If this pair cannot fuse to c the R-symbol is zero; fall back to a
                // trivial (identity) action so the generator stays unitary on the
                // physical subspace rather than annihilating the amplitude.
                matrix[(i, i)] = if r_symbol.norm() > 1e-12 {
                    r_symbol
                } else {
                    Complex64::new(1.0, 0.0)
                };
            } else {
                matrix[(i, i)] = Complex64::new(1.0, 0.0);
            }
        }

        matrix
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
    pub const fn num_qubits(&self) -> usize {
        2 * self.size * self.size
    }

    /// Get the number of logical qubits
    pub const fn num_logical_qubits(&self) -> usize {
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

        let qc = TopologicalQC::new(model, anyons).expect("Failed to create TopologicalQC");
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
    fn test_topological_gate_to_matrix_is_real() {
        // Site-5 proof: the braiding matrix of the cnot() sequence must be a genuine
        // unitary that is NOT the identity (the old implementation returned eye()).
        let model = IsingModel::new();
        let gate = TopologicalGate::cnot();

        let m = gate
            .to_matrix(&model)
            .expect("braiding matrix should be computable for the Ising model");

        let dim = m.nrows();
        assert!(dim >= 2, "fusion-tree space must be non-trivial, got {dim}");

        // Unitarity: M† M ≈ I.
        let mdag = m.mapv(|z| z.conj()).t().to_owned();
        let prod = mdag.dot(&m);
        let mut max_dev = 0.0_f64;
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                max_dev = max_dev.max((prod[(i, j)] - Complex64::new(expected, 0.0)).norm());
            }
        }
        assert!(
            max_dev < 1e-10,
            "braiding matrix is not unitary, max deviation = {max_dev}"
        );

        // Not the identity: at least one off-diagonal or non-unit-phase diagonal entry.
        let identity = Array2::<Complex64>::eye(dim);
        let diff: f64 = m
            .iter()
            .zip(identity.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            diff > 1e-6,
            "braiding matrix collapsed to the identity (fabrication regression)"
        );
    }

    #[test]
    fn test_topological_gate_to_matrix_inverse_braid() {
        // Over- and under-crossings must be inverses: σ_i · σ_i^{-1} = I.
        let model = IsingModel::new();
        let over = TopologicalGate::new(
            vec![BraidingOperation {
                anyon1: 0,
                anyon2: 1,
                over: true,
            }],
            2,
        );
        let under = TopologicalGate::new(
            vec![BraidingOperation {
                anyon1: 0,
                anyon2: 1,
                over: false,
            }],
            2,
        );
        let m_over = over.to_matrix(&model).expect("over braid");
        let m_under = under.to_matrix(&model).expect("under braid");
        let prod = m_under.dot(&m_over);
        let dim = prod.nrows();
        let mut dev = 0.0_f64;
        for i in 0..dim {
            for j in 0..dim {
                let exp = if i == j { 1.0 } else { 0.0 };
                dev = dev.max((prod[(i, j)] - Complex64::new(exp, 0.0)).norm());
            }
        }
        assert!(dev < 1e-10, "σ·σ⁻¹ should be identity, deviation {dev}");
    }

    #[test]
    fn test_braiding_operation() {
        let model = Box::new(IsingModel::new());
        let anyons = vec![AnyonType::new(1, "σ"), AnyonType::new(1, "σ")];

        let mut qc = TopologicalQC::new(model, anyons).expect("Failed to create TopologicalQC");

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

        qc.braid(&braid)
            .expect("Failed to apply braiding operation");

        // State should be normalized
        let norm: f64 = qc.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Final state not normalized: {}",
            norm
        );
    }
}
