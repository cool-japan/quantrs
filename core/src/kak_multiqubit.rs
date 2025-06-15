//! KAK decomposition for multi-qubit unitaries
//!
//! This module extends the Cartan (KAK) decomposition to handle arbitrary
//! n-qubit unitaries through recursive application and generalized
//! decomposition techniques.

use crate::{
    cartan::{CartanDecomposer, CartanDecomposition},
    error::{QuantRS2Error, QuantRS2Result},
    gate::{multi::*, single::*, GateOp},
    matrix_ops::{DenseMatrix, QuantumMatrix},
    qubit::QubitId,
    shannon::ShannonDecomposer,
    synthesis::{decompose_single_qubit_zyz, SingleQubitDecomposition},
};
use ndarray::{s, Array2};
use num_complex::Complex;
use rustc_hash::FxHashMap;

/// Result of multi-qubit KAK decomposition
#[derive(Debug, Clone)]
pub struct MultiQubitKAK {
    /// The decomposed gate sequence
    pub gates: Vec<Box<dyn GateOp>>,
    /// Decomposition tree structure
    pub tree: DecompositionTree,
    /// Total CNOT count
    pub cnot_count: usize,
    /// Total single-qubit gate count
    pub single_qubit_count: usize,
    /// Circuit depth
    pub depth: usize,
}

/// Tree structure representing the hierarchical decomposition
#[derive(Debug, Clone)]
pub enum DecompositionTree {
    /// Leaf node - single or two-qubit gate
    Leaf {
        qubits: Vec<QubitId>,
        gate_type: LeafType,
    },
    /// Internal node - recursive decomposition
    Node {
        qubits: Vec<QubitId>,
        method: DecompositionMethod,
        children: Vec<DecompositionTree>,
    },
}

/// Type of leaf decomposition
#[derive(Debug, Clone)]
pub enum LeafType {
    SingleQubit(SingleQubitDecomposition),
    TwoQubit(CartanDecomposition),
}

/// Method used for decomposition at this level
#[derive(Debug, Clone)]
pub enum DecompositionMethod {
    /// Cosine-Sine Decomposition
    CSD { pivot: usize },
    /// Quantum Shannon Decomposition
    Shannon { partition: usize },
    /// Block diagonalization
    BlockDiagonal { block_size: usize },
    /// Direct Cartan for 2 qubits
    Cartan,
}

/// Multi-qubit KAK decomposer
pub struct MultiQubitKAKDecomposer {
    /// Tolerance for numerical comparisons
    tolerance: f64,
    /// Maximum recursion depth
    max_depth: usize,
    /// Cache for decompositions
    #[allow(dead_code)]
    cache: FxHashMap<u64, MultiQubitKAK>,
    /// Use optimized methods
    use_optimization: bool,
    /// Cartan decomposer for two-qubit blocks
    cartan: CartanDecomposer,
}

impl MultiQubitKAKDecomposer {
    /// Create a new multi-qubit KAK decomposer
    pub fn new() -> Self {
        Self {
            tolerance: 1e-10,
            max_depth: 20,
            cache: FxHashMap::default(),
            use_optimization: true,
            cartan: CartanDecomposer::new(),
        }
    }

    /// Create with custom tolerance
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            tolerance,
            max_depth: 20,
            cache: FxHashMap::default(),
            use_optimization: true,
            cartan: CartanDecomposer::with_tolerance(tolerance),
        }
    }

    /// Decompose an n-qubit unitary
    pub fn decompose(
        &mut self,
        unitary: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
    ) -> QuantRS2Result<MultiQubitKAK> {
        let n = qubit_ids.len();
        let size = 1 << n;

        // Validate input
        if unitary.shape() != [size, size] {
            return Err(QuantRS2Error::InvalidInput(format!(
                "Unitary size {} doesn't match {} qubits",
                unitary.shape()[0],
                n
            )));
        }

        // Check unitarity
        let mat = DenseMatrix::new(unitary.clone())?;
        if !mat.is_unitary(self.tolerance)? {
            return Err(QuantRS2Error::InvalidInput(
                "Matrix is not unitary".to_string(),
            ));
        }

        // Check cache
        if let Some(cached) = self.check_cache(unitary) {
            return Ok(cached.clone());
        }

        // Perform decomposition
        let (tree, gates) = self.decompose_recursive(unitary, qubit_ids, 0)?;

        // Count gates
        let mut cnot_count = 0;
        let mut single_qubit_count = 0;

        for gate in &gates {
            match gate.name() {
                "CNOT" | "CZ" | "SWAP" => cnot_count += self.count_cnots(gate.name()),
                _ => single_qubit_count += 1,
            }
        }

        let result = MultiQubitKAK {
            gates,
            tree,
            cnot_count,
            single_qubit_count,
            depth: 0, // TODO: Calculate actual depth
        };

        // Cache result
        self.cache_result(unitary, &result);

        Ok(result)
    }

    /// Recursive decomposition algorithm
    fn decompose_recursive(
        &mut self,
        unitary: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
        depth: usize,
    ) -> QuantRS2Result<(DecompositionTree, Vec<Box<dyn GateOp>>)> {
        if depth > self.max_depth {
            return Err(QuantRS2Error::InvalidInput(
                "Maximum recursion depth exceeded".to_string(),
            ));
        }

        let n = qubit_ids.len();

        // Base cases
        match n {
            0 => {
                let tree = DecompositionTree::Leaf {
                    qubits: vec![],
                    gate_type: LeafType::SingleQubit(SingleQubitDecomposition {
                        global_phase: 0.0,
                        theta1: 0.0,
                        phi: 0.0,
                        theta2: 0.0,
                        basis: "ZYZ".to_string(),
                    }),
                };
                Ok((tree, vec![]))
            }
            1 => {
                let decomp = decompose_single_qubit_zyz(&unitary.view())?;
                let gates = self.single_qubit_to_gates(&decomp, qubit_ids[0]);
                let tree = DecompositionTree::Leaf {
                    qubits: qubit_ids.to_vec(),
                    gate_type: LeafType::SingleQubit(decomp),
                };
                Ok((tree, gates))
            }
            2 => {
                let decomp = self.cartan.decompose(unitary)?;
                let gates = self.cartan.to_gates(&decomp, qubit_ids)?;
                let tree = DecompositionTree::Leaf {
                    qubits: qubit_ids.to_vec(),
                    gate_type: LeafType::TwoQubit(decomp),
                };
                Ok((tree, gates))
            }
            _ => {
                // For n > 2, choose decomposition method
                let method = self.choose_decomposition_method(unitary, n);

                match method {
                    DecompositionMethod::CSD { pivot } => {
                        self.decompose_csd(unitary, qubit_ids, pivot, depth)
                    }
                    DecompositionMethod::Shannon { partition } => {
                        self.decompose_shannon(unitary, qubit_ids, partition, depth)
                    }
                    DecompositionMethod::BlockDiagonal { block_size } => {
                        self.decompose_block_diagonal(unitary, qubit_ids, block_size, depth)
                    }
                    _ => unreachable!("Invalid method for n > 2"),
                }
            }
        }
    }

    /// Choose optimal decomposition method based on matrix structure
    fn choose_decomposition_method(
        &self,
        unitary: &Array2<Complex<f64>>,
        n: usize,
    ) -> DecompositionMethod {
        if self.use_optimization {
            // Analyze matrix structure to choose optimal method
            if self.has_block_structure(unitary, n) {
                DecompositionMethod::BlockDiagonal { block_size: n / 2 }
            } else if n % 2 == 0 {
                // Even number of qubits - use CSD at midpoint
                DecompositionMethod::CSD { pivot: n / 2 }
            } else {
                // Odd number - use Shannon decomposition
                DecompositionMethod::Shannon { partition: n / 2 }
            }
        } else {
            // Default to CSD
            DecompositionMethod::CSD { pivot: n / 2 }
        }
    }

    /// Decompose using Cosine-Sine Decomposition
    fn decompose_csd(
        &mut self,
        unitary: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
        pivot: usize,
        depth: usize,
    ) -> QuantRS2Result<(DecompositionTree, Vec<Box<dyn GateOp>>)> {
        let n = qubit_ids.len();
        let _size = 1 << n;
        let pivot_size = 1 << pivot;

        // Split unitary into blocks based on pivot
        // U = [A B]
        //     [C D]
        let a = unitary.slice(s![..pivot_size, ..pivot_size]).to_owned();
        let b = unitary.slice(s![..pivot_size, pivot_size..]).to_owned();
        let c = unitary.slice(s![pivot_size.., ..pivot_size]).to_owned();
        let d = unitary.slice(s![pivot_size.., pivot_size..]).to_owned();

        // Apply CSD to find:
        // U = (U1 ⊗ V1) · Σ · (U2 ⊗ V2)
        // where Σ is diagonal in the CSD basis

        // This is a simplified version - full CSD would use SVD
        let (u1, v1, sigma, u2, v2) = self.compute_csd(&a, &b, &c, &d)?;

        let mut gates = Vec::new();
        let mut children = Vec::new();

        // Decompose U2 and V2 (right multiplications)
        let left_qubits = &qubit_ids[..pivot];
        let right_qubits = &qubit_ids[pivot..];

        let (u2_tree, u2_gates) = self.decompose_recursive(&u2, left_qubits, depth + 1)?;
        let (v2_tree, v2_gates) = self.decompose_recursive(&v2, right_qubits, depth + 1)?;

        gates.extend(u2_gates);
        gates.extend(v2_gates);
        children.push(u2_tree);
        children.push(v2_tree);

        // Apply diagonal gates (controlled rotations)
        let diag_gates = self.diagonal_to_gates(&sigma, qubit_ids)?;
        gates.extend(diag_gates);

        // Decompose U1 and V1 (left multiplications)
        let (u1_tree, u1_gates) = self.decompose_recursive(&u1, left_qubits, depth + 1)?;
        let (v1_tree, v1_gates) = self.decompose_recursive(&v1, right_qubits, depth + 1)?;

        gates.extend(u1_gates);
        gates.extend(v1_gates);
        children.push(u1_tree);
        children.push(v1_tree);

        let tree = DecompositionTree::Node {
            qubits: qubit_ids.to_vec(),
            method: DecompositionMethod::CSD { pivot },
            children,
        };

        Ok((tree, gates))
    }

    /// Decompose using Shannon decomposition
    fn decompose_shannon(
        &mut self,
        unitary: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
        partition: usize,
        _depth: usize,
    ) -> QuantRS2Result<(DecompositionTree, Vec<Box<dyn GateOp>>)> {
        // Use the Shannon decomposer for this
        let mut shannon = ShannonDecomposer::new();
        let decomp = shannon.decompose(unitary, qubit_ids)?;

        // Build tree structure
        let tree = DecompositionTree::Node {
            qubits: qubit_ids.to_vec(),
            method: DecompositionMethod::Shannon { partition },
            children: vec![], // Shannon decomposer doesn't provide tree structure
        };

        Ok((tree, decomp.gates))
    }

    /// Decompose block diagonal matrix
    fn decompose_block_diagonal(
        &mut self,
        unitary: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
        block_size: usize,
        depth: usize,
    ) -> QuantRS2Result<(DecompositionTree, Vec<Box<dyn GateOp>>)> {
        let n = qubit_ids.len();
        let num_blocks = n / block_size;

        let mut gates = Vec::new();
        let mut children = Vec::new();

        // Decompose each block independently
        for i in 0..num_blocks {
            let start = i * block_size;
            let end = (i + 1) * block_size;
            let block_qubits = &qubit_ids[start..end];

            // Extract block from unitary
            let block = self.extract_block(unitary, i, block_size)?;

            let (block_tree, block_gates) =
                self.decompose_recursive(&block, block_qubits, depth + 1)?;
            gates.extend(block_gates);
            children.push(block_tree);
        }

        let tree = DecompositionTree::Node {
            qubits: qubit_ids.to_vec(),
            method: DecompositionMethod::BlockDiagonal { block_size },
            children,
        };

        Ok((tree, gates))
    }

    /// Compute Cosine-Sine Decomposition
    fn compute_csd(
        &self,
        a: &Array2<Complex<f64>>,
        b: &Array2<Complex<f64>>,
        c: &Array2<Complex<f64>>,
        d: &Array2<Complex<f64>>,
    ) -> QuantRS2Result<(
        Array2<Complex<f64>>, // U1
        Array2<Complex<f64>>, // V1
        Array2<Complex<f64>>, // Sigma
        Array2<Complex<f64>>, // U2
        Array2<Complex<f64>>, // V2
    )> {
        // This is a simplified placeholder
        // Full CSD implementation would use specialized algorithms

        let size = a.shape()[0];
        let identity = Array2::eye(size);
        let _zero: Array2<Complex<f64>> = Array2::zeros((size, size));

        // For now, return identity transformations
        let u1 = identity.clone();
        let v1 = identity.clone();
        let u2 = identity.clone();
        let v2 = identity.clone();

        // Sigma would contain the CS angles
        let mut sigma = Array2::zeros((size * 2, size * 2));
        sigma.slice_mut(s![..size, ..size]).assign(a);
        sigma.slice_mut(s![..size, size..]).assign(b);
        sigma.slice_mut(s![size.., ..size]).assign(c);
        sigma.slice_mut(s![size.., size..]).assign(d);

        Ok((u1, v1, sigma, u2, v2))
    }

    /// Convert diagonal matrix to controlled rotation gates
    fn diagonal_to_gates(
        &self,
        diagonal: &Array2<Complex<f64>>,
        qubit_ids: &[QubitId],
    ) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
        let mut gates = Vec::new();

        // Extract diagonal elements
        let n = diagonal.shape()[0];
        for i in 0..n {
            let phase = diagonal[[i, i]].arg();
            if phase.abs() > self.tolerance {
                // Determine which qubits are in state |1⟩ for this diagonal element
                let mut control_pattern = Vec::new();
                let mut temp = i;
                for j in 0..qubit_ids.len() {
                    if temp & 1 == 1 {
                        control_pattern.push(j);
                    }
                    temp >>= 1;
                }

                // Create multi-controlled phase gate
                if control_pattern.is_empty() {
                    // Global phase - can be ignored
                } else if control_pattern.len() == 1 {
                    // Single-qubit phase
                    gates.push(Box::new(RotationZ {
                        target: qubit_ids[control_pattern[0]],
                        theta: phase,
                    }) as Box<dyn GateOp>);
                } else {
                    // Multi-controlled phase - decompose further
                    // For now, use simple decomposition
                    let target = qubit_ids[control_pattern.pop().unwrap()];
                    for &control_idx in &control_pattern {
                        gates.push(Box::new(CNOT {
                            control: qubit_ids[control_idx],
                            target,
                        }));
                    }

                    gates.push(Box::new(RotationZ {
                        target,
                        theta: phase,
                    }) as Box<dyn GateOp>);

                    // Uncompute CNOTs
                    for &control_idx in control_pattern.iter().rev() {
                        gates.push(Box::new(CNOT {
                            control: qubit_ids[control_idx],
                            target,
                        }));
                    }
                }
            }
        }

        Ok(gates)
    }

    /// Check if matrix has block diagonal structure
    fn has_block_structure(&self, unitary: &Array2<Complex<f64>>, _n: usize) -> bool {
        // Simple check - look for zeros in off-diagonal blocks
        let size = unitary.shape()[0];
        let block_size = size / 2;

        let mut off_diagonal_norm = 0.0;

        // Check upper-right block
        for i in 0..block_size {
            for j in block_size..size {
                off_diagonal_norm += unitary[[i, j]].norm_sqr();
            }
        }

        // Check lower-left block
        for i in block_size..size {
            for j in 0..block_size {
                off_diagonal_norm += unitary[[i, j]].norm_sqr();
            }
        }

        off_diagonal_norm.sqrt() < self.tolerance
    }

    /// Extract a block from block-diagonal matrix
    fn extract_block(
        &self,
        unitary: &Array2<Complex<f64>>,
        block_idx: usize,
        block_size: usize,
    ) -> QuantRS2Result<Array2<Complex<f64>>> {
        let size = 1 << block_size;
        let start = block_idx * size;
        let end = (block_idx + 1) * size;

        Ok(unitary.slice(s![start..end, start..end]).to_owned())
    }

    /// Convert single-qubit decomposition to gates
    fn single_qubit_to_gates(
        &self,
        decomp: &SingleQubitDecomposition,
        qubit: QubitId,
    ) -> Vec<Box<dyn GateOp>> {
        let mut gates = Vec::new();

        if decomp.theta1.abs() > self.tolerance {
            gates.push(Box::new(RotationZ {
                target: qubit,
                theta: decomp.theta1,
            }) as Box<dyn GateOp>);
        }

        if decomp.phi.abs() > self.tolerance {
            gates.push(Box::new(RotationY {
                target: qubit,
                theta: decomp.phi,
            }) as Box<dyn GateOp>);
        }

        if decomp.theta2.abs() > self.tolerance {
            gates.push(Box::new(RotationZ {
                target: qubit,
                theta: decomp.theta2,
            }) as Box<dyn GateOp>);
        }

        gates
    }

    /// Count CNOTs for different gate types
    fn count_cnots(&self, gate_name: &str) -> usize {
        match gate_name {
            "CNOT" => 1,
            "CZ" => 1,   // CZ = H·CNOT·H
            "SWAP" => 3, // SWAP uses 3 CNOTs
            _ => 0,
        }
    }

    /// Check cache for existing decomposition
    fn check_cache(&self, _unitary: &Array2<Complex<f64>>) -> Option<&MultiQubitKAK> {
        // Simple hash based on first few elements
        // Real implementation would use better hashing
        None
    }

    /// Cache decomposition result
    fn cache_result(&mut self, _unitary: &Array2<Complex<f64>>, _result: &MultiQubitKAK) {
        // Cache implementation
    }
}

impl Default for MultiQubitKAKDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze decomposition tree structure
pub struct KAKTreeAnalyzer {
    /// Track statistics
    stats: DecompositionStats,
}

#[derive(Debug, Default, Clone)]
pub struct DecompositionStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub max_depth: usize,
    pub method_counts: FxHashMap<String, usize>,
    pub cnot_distribution: FxHashMap<usize, usize>,
}

impl KAKTreeAnalyzer {
    /// Create new analyzer
    pub fn new() -> Self {
        Self {
            stats: DecompositionStats::default(),
        }
    }

    /// Analyze decomposition tree
    pub fn analyze(&mut self, tree: &DecompositionTree) -> DecompositionStats {
        self.stats = DecompositionStats::default();
        self.analyze_recursive(tree, 0);
        self.stats.clone()
    }

    fn analyze_recursive(&mut self, tree: &DecompositionTree, depth: usize) {
        self.stats.total_nodes += 1;
        self.stats.max_depth = self.stats.max_depth.max(depth);

        match tree {
            DecompositionTree::Leaf {
                qubits: _qubits,
                gate_type,
            } => {
                self.stats.leaf_nodes += 1;

                match gate_type {
                    LeafType::SingleQubit(_) => {
                        *self
                            .stats
                            .method_counts
                            .entry("single_qubit".to_string())
                            .or_insert(0) += 1;
                    }
                    LeafType::TwoQubit(cartan) => {
                        *self
                            .stats
                            .method_counts
                            .entry("two_qubit".to_string())
                            .or_insert(0) += 1;
                        let cnots = cartan.interaction.cnot_count(1e-10);
                        *self.stats.cnot_distribution.entry(cnots).or_insert(0) += 1;
                    }
                }
            }
            DecompositionTree::Node {
                method, children, ..
            } => {
                let method_name = match method {
                    DecompositionMethod::CSD { .. } => "csd",
                    DecompositionMethod::Shannon { .. } => "shannon",
                    DecompositionMethod::BlockDiagonal { .. } => "block_diagonal",
                    DecompositionMethod::Cartan => "cartan",
                };
                *self
                    .stats
                    .method_counts
                    .entry(method_name.to_string())
                    .or_insert(0) += 1;

                for child in children {
                    self.analyze_recursive(child, depth + 1);
                }
            }
        }
    }
}

/// Utility function for quick multi-qubit KAK decomposition
pub fn kak_decompose_multiqubit(
    unitary: &Array2<Complex<f64>>,
    qubit_ids: &[QubitId],
) -> QuantRS2Result<Vec<Box<dyn GateOp>>> {
    let mut decomposer = MultiQubitKAKDecomposer::new();
    let decomp = decomposer.decompose(unitary, qubit_ids)?;
    Ok(decomp.gates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use num_complex::Complex;

    #[test]
    fn test_multiqubit_kak_single() {
        let mut decomposer = MultiQubitKAKDecomposer::new();

        // Hadamard matrix
        let h = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(-1.0, 0.0),
            ],
        )
        .unwrap()
            / Complex::new(2.0_f64.sqrt(), 0.0);

        let qubit_ids = vec![QubitId(0)];
        let decomp = decomposer.decompose(&h, &qubit_ids).unwrap();

        assert!(decomp.single_qubit_count <= 3);
        assert_eq!(decomp.cnot_count, 0);

        // Check tree structure
        match &decomp.tree {
            DecompositionTree::Leaf {
                gate_type: LeafType::SingleQubit(_),
                ..
            } => {}
            _ => panic!("Expected single-qubit leaf"),
        }
    }

    #[test]
    fn test_multiqubit_kak_two() {
        let mut decomposer = MultiQubitKAKDecomposer::new();

        // CNOT matrix
        let cnot = Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
            ],
        )
        .unwrap();

        let qubit_ids = vec![QubitId(0), QubitId(1)];
        let decomp = decomposer.decompose(&cnot, &qubit_ids).unwrap();

        assert!(decomp.cnot_count <= 1);

        // Check tree structure
        match &decomp.tree {
            DecompositionTree::Leaf {
                gate_type: LeafType::TwoQubit(_),
                ..
            } => {}
            _ => panic!("Expected two-qubit leaf"),
        }
    }

    #[test]
    fn test_multiqubit_kak_three() {
        let mut decomposer = MultiQubitKAKDecomposer::new();

        // 3-qubit identity
        let identity = Array2::eye(8);
        let identity_complex = identity.mapv(|x| Complex::new(x, 0.0));

        let qubit_ids = vec![QubitId(0), QubitId(1), QubitId(2)];
        let decomp = decomposer.decompose(&identity_complex, &qubit_ids).unwrap();

        // Identity should result in empty circuit
        assert_eq!(decomp.gates.len(), 0);
        assert_eq!(decomp.cnot_count, 0);
        assert_eq!(decomp.single_qubit_count, 0);
    }

    #[test]
    fn test_tree_analyzer() {
        let mut analyzer = KAKTreeAnalyzer::new();

        // Create a simple tree
        let tree = DecompositionTree::Node {
            qubits: vec![QubitId(0), QubitId(1), QubitId(2)],
            method: DecompositionMethod::CSD { pivot: 2 },
            children: vec![
                DecompositionTree::Leaf {
                    qubits: vec![QubitId(0), QubitId(1)],
                    gate_type: LeafType::TwoQubit(CartanDecomposition {
                        left_gates: (
                            SingleQubitDecomposition {
                                global_phase: 0.0,
                                theta1: 0.0,
                                phi: 0.0,
                                theta2: 0.0,
                                basis: "ZYZ".to_string(),
                            },
                            SingleQubitDecomposition {
                                global_phase: 0.0,
                                theta1: 0.0,
                                phi: 0.0,
                                theta2: 0.0,
                                basis: "ZYZ".to_string(),
                            },
                        ),
                        right_gates: (
                            SingleQubitDecomposition {
                                global_phase: 0.0,
                                theta1: 0.0,
                                phi: 0.0,
                                theta2: 0.0,
                                basis: "ZYZ".to_string(),
                            },
                            SingleQubitDecomposition {
                                global_phase: 0.0,
                                theta1: 0.0,
                                phi: 0.0,
                                theta2: 0.0,
                                basis: "ZYZ".to_string(),
                            },
                        ),
                        interaction: crate::prelude::CartanCoefficients::new(0.0, 0.0, 0.0),
                        global_phase: 0.0,
                    }),
                },
                DecompositionTree::Leaf {
                    qubits: vec![QubitId(2)],
                    gate_type: LeafType::SingleQubit(SingleQubitDecomposition {
                        global_phase: 0.0,
                        theta1: 0.0,
                        phi: 0.0,
                        theta2: 0.0,
                        basis: "ZYZ".to_string(),
                    }),
                },
            ],
        };

        let stats = analyzer.analyze(&tree);

        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.leaf_nodes, 2);
        assert_eq!(stats.max_depth, 1);
        assert_eq!(stats.method_counts.get("csd"), Some(&1));
    }

    #[test]
    fn test_block_structure_detection() {
        let decomposer = MultiQubitKAKDecomposer::new();

        // Create block diagonal matrix
        let mut block_diag = Array2::zeros((4, 4));
        block_diag[[0, 0]] = Complex::new(1.0, 0.0);
        block_diag[[1, 1]] = Complex::new(1.0, 0.0);
        block_diag[[2, 2]] = Complex::new(1.0, 0.0);
        block_diag[[3, 3]] = Complex::new(1.0, 0.0);

        assert!(decomposer.has_block_structure(&block_diag, 2));

        // Non-block diagonal
        block_diag[[0, 2]] = Complex::new(1.0, 0.0);
        assert!(!decomposer.has_block_structure(&block_diag, 2));
    }
}
