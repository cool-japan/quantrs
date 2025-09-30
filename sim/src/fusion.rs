//! Gate fusion optimization for quantum circuit simulation.
//!
//! This module implements gate fusion techniques to optimize quantum circuits
//! by combining consecutive gates that act on the same qubits into single
//! multi-qubit gates, reducing the number of matrix multiplications needed.

use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use std::collections::{HashMap, HashSet};

use crate::error::{Result, SimulatorError};
use crate::sparse::{CSRMatrix, SparseMatrixBuilder};
use quantrs2_core::gate::GateOp;
use quantrs2_core::qubit::QubitId;

// SciRS2 stub types (would be replaced with actual SciRS2 imports)
#[derive(Debug)]
struct SciRS2MatrixMultiplier;

impl SciRS2MatrixMultiplier {
    fn multiply_sparse(a: &CSRMatrix, b: &CSRMatrix) -> Result<CSRMatrix> {
        // Stub implementation for SciRS2 sparse matrix multiplication
        if a.num_cols != b.num_rows {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Cannot multiply {}x{} with {}x{}",
                a.num_rows, a.num_cols, b.num_rows, b.num_cols
            )));
        }

        let mut builder = SparseMatrixBuilder::new(a.num_rows, b.num_cols);

        // Simple sparse matrix multiplication
        for i in 0..a.num_rows {
            for k in a.row_ptr[i]..a.row_ptr[i + 1] {
                let a_val = a.values[k];
                let a_col = a.col_indices[k];

                for j_idx in b.row_ptr[a_col]..b.row_ptr[a_col + 1] {
                    let b_val = b.values[j_idx];
                    let b_col = b.col_indices[j_idx];

                    builder.add(i, b_col, a_val * b_val);
                }
            }
        }

        Ok(builder.build())
    }

    fn multiply_dense(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Result<Array2<Complex64>> {
        // Stub implementation for SciRS2 dense matrix multiplication
        if a.ncols() != b.nrows() {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Cannot multiply {}x{} with {}x{}",
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols()
            )));
        }

        Ok(a.dot(b))
    }
}

/// Gate fusion strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionStrategy {
    /// Fuse all consecutive gates on same qubits
    Aggressive,
    /// Only fuse if it reduces gate count
    Conservative,
    /// Fuse based on gate depth reduction
    DepthOptimized,
    /// Custom fusion with cost function
    Custom,
}

/// Fusable gate group
#[derive(Debug, Clone)]
pub struct GateGroup {
    /// Indices of gates in this group
    pub gate_indices: Vec<usize>,
    /// Qubits this group acts on
    pub qubits: Vec<QubitId>,
    /// Whether this group can be fused
    pub fusable: bool,
    /// Estimated cost of fusion
    pub fusion_cost: f64,
}

/// Gate fusion optimizer
pub struct GateFusion {
    /// Fusion strategy
    strategy: FusionStrategy,
    /// Maximum qubits to fuse (to limit matrix size)
    max_fusion_qubits: usize,
    /// Minimum gates to consider fusion
    min_fusion_gates: usize,
    /// Cost threshold for fusion
    cost_threshold: f64,
}

impl GateFusion {
    /// Create a new gate fusion optimizer
    pub fn new(strategy: FusionStrategy) -> Self {
        Self {
            strategy,
            max_fusion_qubits: 4,
            min_fusion_gates: 2,
            cost_threshold: 0.8,
        }
    }

    /// Configure fusion parameters
    pub fn with_params(mut self, max_qubits: usize, min_gates: usize, threshold: f64) -> Self {
        self.max_fusion_qubits = max_qubits;
        self.min_fusion_gates = min_gates;
        self.cost_threshold = threshold;
        self
    }

    /// Analyze circuit for fusion opportunities
    pub fn analyze_circuit(&self, gates: &[Box<dyn GateOp>]) -> Result<Vec<GateGroup>> {
        let mut groups = Vec::new();
        let mut processed = vec![false; gates.len()];

        for i in 0..gates.len() {
            if processed[i] {
                continue;
            }

            // Start a new group
            let mut group = GateGroup {
                gate_indices: vec![i],
                qubits: gates[i].qubits().to_vec(),
                fusable: false,
                fusion_cost: 0.0,
            };

            // Find consecutive gates that can be fused
            for j in i + 1..gates.len() {
                if processed[j] {
                    continue;
                }

                // Check if gate j can be added to the group
                if self.can_fuse_with_group(&group, gates[j].as_ref()) {
                    group.gate_indices.push(j);

                    // Update qubit set
                    for qubit in gates[j].qubits() {
                        if !group.qubits.contains(&qubit) {
                            group.qubits.push(qubit);
                        }
                    }

                    // Check if we've reached the limit
                    if group.qubits.len() > self.max_fusion_qubits {
                        group.gate_indices.pop();
                        break;
                    }
                } else if self.blocks_fusion(&group, gates[j].as_ref()) {
                    // This gate blocks further fusion
                    break;
                }
            }

            // Evaluate if this group should be fused
            if group.gate_indices.len() >= self.min_fusion_gates {
                group.fusion_cost = self.compute_fusion_cost(&group, gates)?;
                group.fusable = self.should_fuse(&group);

                // Mark gates as processed if we're fusing
                if group.fusable {
                    for &idx in &group.gate_indices {
                        processed[idx] = true;
                    }
                }
            }

            groups.push(group);
        }

        Ok(groups)
    }

    /// Check if a gate can be fused with a group
    fn can_fuse_with_group(&self, group: &GateGroup, gate: &dyn GateOp) -> bool {
        // Gate must share at least one qubit with the group
        let gate_qubits: HashSet<_> = gate.qubits().iter().cloned().collect();
        let group_qubits: HashSet<_> = group.qubits.iter().cloned().collect();

        match self.strategy {
            FusionStrategy::Aggressive => {
                // Fuse if there's any qubit overlap
                !gate_qubits.is_disjoint(&group_qubits)
            }
            FusionStrategy::Conservative => {
                // Only fuse if all qubits are in the group
                gate_qubits.is_subset(&group_qubits) || group_qubits.is_subset(&gate_qubits)
            }
            FusionStrategy::DepthOptimized => {
                // Fuse if it doesn't increase qubit count too much
                let combined_qubits: HashSet<_> =
                    gate_qubits.union(&group_qubits).cloned().collect();
                combined_qubits.len() <= self.max_fusion_qubits
            }
            FusionStrategy::Custom => {
                // Custom logic (simplified here)
                !gate_qubits.is_disjoint(&group_qubits)
            }
        }
    }

    /// Check if a gate blocks fusion
    fn blocks_fusion(&self, group: &GateGroup, gate: &dyn GateOp) -> bool {
        // A gate blocks fusion if it acts on some but not all qubits of the group
        let gate_qubits: HashSet<_> = gate.qubits().iter().cloned().collect();
        let group_qubits: HashSet<_> = group.qubits.iter().cloned().collect();

        let intersection = gate_qubits.intersection(&group_qubits).count();
        intersection > 0 && intersection < group_qubits.len()
    }

    /// Compute the cost of fusing a group
    fn compute_fusion_cost(&self, group: &GateGroup, gates: &[Box<dyn GateOp>]) -> Result<f64> {
        let num_qubits = group.qubits.len();
        let num_gates = group.gate_indices.len();

        // Cost factors:
        // 1. Matrix size (2^n x 2^n for n qubits)
        let matrix_size_cost = (1 << num_qubits) as f64;

        // 2. Number of operations saved
        let ops_saved = (num_gates - 1) as f64;

        // 3. Memory requirements
        let memory_cost = matrix_size_cost * matrix_size_cost * 16.0; // Complex64 size

        // Combined cost (lower is better)
        let cost = matrix_size_cost / (ops_saved + 1.0) + memory_cost / 1e9;

        Ok(cost)
    }

    /// Decide if a group should be fused
    fn should_fuse(&self, group: &GateGroup) -> bool {
        match self.strategy {
            FusionStrategy::Aggressive => true,
            FusionStrategy::Conservative => group.fusion_cost < self.cost_threshold,
            FusionStrategy::DepthOptimized => group.gate_indices.len() > 2,
            FusionStrategy::Custom => group.fusion_cost < self.cost_threshold,
        }
    }

    /// Fuse a group of gates into a single gate
    pub fn fuse_group(
        &self,
        group: &GateGroup,
        gates: &[Box<dyn GateOp>],
        num_qubits: usize,
    ) -> Result<FusedGate> {
        let group_qubits = &group.qubits;
        let group_size = group_qubits.len();

        // Create identity matrix for the fused gate
        let dim = 1 << group_size;
        let mut fused_matrix = Array2::eye(dim);

        // Apply each gate in sequence
        for &gate_idx in &group.gate_indices {
            let gate = &gates[gate_idx];
            let gate_matrix = self.get_gate_matrix(gate.as_ref())?;

            // Map gate qubits to group qubits
            let gate_qubits = gate.qubits();
            let qubit_map: HashMap<QubitId, usize> = group_qubits
                .iter()
                .enumerate()
                .map(|(i, &q)| (q, i))
                .collect();

            // Expand gate matrix to group dimension
            let expanded =
                self.expand_gate_matrix(&gate_matrix, &gate_qubits, &qubit_map, group_size)?;

            // Multiply using SciRS2
            fused_matrix = SciRS2MatrixMultiplier::multiply_dense(&expanded, &fused_matrix)?;
        }

        Ok(FusedGate {
            matrix: fused_matrix,
            qubits: group_qubits.clone(),
            original_gates: group.gate_indices.clone(),
        })
    }

    /// Get matrix representation of a gate
    fn get_gate_matrix(&self, gate: &dyn GateOp) -> Result<Array2<Complex64>> {
        // This would use the gate's matrix() method in a real implementation
        // For now, return a placeholder based on gate type
        match gate.name() {
            "Hadamard" => Ok(Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                    Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                    Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
                ],
            )
            .map_err(|_| SimulatorError::InvalidInput("Shape error".to_string()))?),
            "PauliX" => Ok(Array2::from_shape_vec(
                (2, 2),
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .map_err(|_| SimulatorError::InvalidInput("Shape error".to_string()))?),
            "CNOT" => Ok(Array2::from_shape_vec(
                (4, 4),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            )
            .map_err(|_| SimulatorError::InvalidInput("Shape error".to_string()))?),
            _ => {
                // Default to identity
                let n = gate.qubits().len();
                let dim = 1 << n;
                Ok(Array2::eye(dim))
            }
        }
    }

    /// Expand a gate matrix to act on a larger qubit space
    fn expand_gate_matrix(
        &self,
        gate_matrix: &Array2<Complex64>,
        gate_qubits: &[QubitId],
        qubit_map: &HashMap<QubitId, usize>,
        total_qubits: usize,
    ) -> Result<Array2<Complex64>> {
        let dim = 1 << total_qubits;
        let mut expanded = Array2::zeros((dim, dim));

        // Map gate qubits to their positions in the expanded space
        let gate_positions: Vec<usize> = gate_qubits
            .iter()
            .map(|q| qubit_map.get(q).copied().unwrap_or(0))
            .collect();

        // Fill the expanded matrix
        for i in 0..dim {
            for j in 0..dim {
                // Extract the relevant bits for the gate qubits
                let mut gate_i = 0;
                let mut gate_j = 0;
                let mut other_bits_match = true;

                for (k, &pos) in gate_positions.iter().enumerate() {
                    if (i >> pos) & 1 == 1 {
                        gate_i |= 1 << k;
                    }
                    if (j >> pos) & 1 == 1 {
                        gate_j |= 1 << k;
                    }
                }

                // Check that non-gate qubits match
                for k in 0..total_qubits {
                    if !gate_positions.contains(&k) && ((i >> k) & 1) != ((j >> k) & 1) {
                        other_bits_match = false;
                        break;
                    }
                }

                if other_bits_match {
                    expanded[[i, j]] = gate_matrix[[gate_i, gate_j]];
                }
            }
        }

        Ok(expanded)
    }

    /// Apply fusion to a circuit
    pub fn optimize_circuit(
        &self,
        gates: Vec<Box<dyn GateOp>>,
        num_qubits: usize,
    ) -> Result<OptimizedCircuit> {
        let groups = self.analyze_circuit(&gates)?;
        let mut optimized_gates = Vec::new();
        let mut fusion_map = HashMap::new();

        let mut processed = vec![false; gates.len()];

        for group in &groups {
            if group.fusable && group.gate_indices.len() > 1 {
                // Fuse this group
                let fused = self.fuse_group(&group, &gates, num_qubits)?;
                let fused_idx = optimized_gates.len();
                optimized_gates.push(OptimizedGate::Fused(fused));

                // Record fusion mapping
                for &gate_idx in &group.gate_indices {
                    fusion_map.insert(gate_idx, fused_idx);
                    processed[gate_idx] = true;
                }
            } else {
                // Keep gates unfused
                for &gate_idx in &group.gate_indices {
                    if !processed[gate_idx] {
                        optimized_gates.push(OptimizedGate::Original(gate_idx));
                        processed[gate_idx] = true;
                    }
                }
            }
        }

        // Add any remaining unfused gates
        for (i, &p) in processed.iter().enumerate() {
            if !p {
                optimized_gates.push(OptimizedGate::Original(i));
            }
        }

        Ok(OptimizedCircuit {
            gates: optimized_gates,
            original_gates: gates,
            fusion_map,
            stats: self.compute_stats(&groups),
        })
    }

    /// Compute fusion statistics
    fn compute_stats(&self, groups: &[GateGroup]) -> FusionStats {
        let total_groups = groups.len();
        let fused_groups = groups.iter().filter(|g| g.fusable).count();
        let total_gates: usize = groups.iter().map(|g| g.gate_indices.len()).sum();
        let fused_gates: usize = groups
            .iter()
            .filter(|g| g.fusable)
            .map(|g| g.gate_indices.len())
            .sum();

        FusionStats {
            total_gates,
            fused_gates,
            fusion_ratio: fused_gates as f64 / total_gates.max(1) as f64,
            groups_analyzed: total_groups,
            groups_fused: fused_groups,
        }
    }
}

/// A fused gate combining multiple gates
#[derive(Debug)]
pub struct FusedGate {
    /// Combined matrix representation
    pub matrix: Array2<Complex64>,
    /// Qubits this gate acts on
    pub qubits: Vec<QubitId>,
    /// Original gate indices that were fused
    pub original_gates: Vec<usize>,
}

impl FusedGate {
    /// Convert to sparse representation
    pub fn to_sparse(&self) -> Result<CSRMatrix> {
        let mut builder = SparseMatrixBuilder::new(self.matrix.nrows(), self.matrix.ncols());

        for ((i, j), &val) in self.matrix.indexed_iter() {
            if val.norm() > 1e-12 {
                builder.set_value(i, j, val);
            }
        }

        Ok(builder.build())
    }

    /// Get the dimension of the gate
    pub fn dimension(&self) -> usize {
        self.matrix.nrows()
    }
}

/// Optimized gate representation
#[derive(Debug)]
pub enum OptimizedGate {
    /// Original unfused gate (index into original gates)
    Original(usize),
    /// Fused gate combining multiple gates
    Fused(FusedGate),
}

/// Optimized circuit after fusion
#[derive(Debug)]
pub struct OptimizedCircuit {
    /// Optimized gate sequence
    pub gates: Vec<OptimizedGate>,
    /// Original gates (for reference)
    pub original_gates: Vec<Box<dyn GateOp>>,
    /// Mapping from original gate index to fused gate index
    pub fusion_map: HashMap<usize, usize>,
    /// Fusion statistics
    pub stats: FusionStats,
}

impl OptimizedCircuit {
    /// Get the effective gate count after fusion
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// Get memory usage estimate
    pub fn memory_usage(&self) -> usize {
        self.gates
            .iter()
            .map(|g| match g {
                OptimizedGate::Original(_) => 64, // Approximate
                OptimizedGate::Fused(f) => f.dimension() * f.dimension() * 16,
            })
            .sum()
    }
}

/// Fusion statistics
#[derive(Debug)]
pub struct FusionStats {
    /// Total number of gates before fusion
    pub total_gates: usize,
    /// Number of gates that were fused
    pub fused_gates: usize,
    /// Ratio of fused gates
    pub fusion_ratio: f64,
    /// Number of groups analyzed
    pub groups_analyzed: usize,
    /// Number of groups that were fused
    pub groups_fused: usize,
}

/// Benchmark different fusion strategies
pub fn benchmark_fusion_strategies(gates: Vec<Box<dyn GateOp>>, num_qubits: usize) -> Result<()> {
    println!("\nGate Fusion Benchmark");
    println!("Original circuit: {} gates", gates.len());
    println!("{:-<60}", "");

    for strategy in [
        FusionStrategy::Conservative,
        FusionStrategy::Aggressive,
        FusionStrategy::DepthOptimized,
    ] {
        let fusion = GateFusion::new(strategy);
        let start = std::time::Instant::now();

        let optimized = fusion.optimize_circuit(gates.clone(), num_qubits)?;
        let elapsed = start.elapsed();

        println!("\n{:?} Strategy:", strategy);
        println!("  Gates after fusion: {}", optimized.gate_count());
        println!(
            "  Fusion ratio: {:.2}%",
            optimized.stats.fusion_ratio * 100.0
        );
        println!(
            "  Groups fused: {}/{}",
            optimized.stats.groups_fused, optimized.stats.groups_analyzed
        );
        println!(
            "  Memory usage: {:.2} MB",
            optimized.memory_usage() as f64 / 1e6
        );
        println!("  Optimization time: {:?}", elapsed);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::multi::CNOT;
    use quantrs2_core::gate::single::{Hadamard, PauliX};

    #[test]
    fn test_gate_group_creation() {
        let group = GateGroup {
            gate_indices: vec![0, 1, 2],
            qubits: vec![QubitId::new(0), QubitId::new(1)],
            fusable: true,
            fusion_cost: 0.5,
        };

        assert_eq!(group.gate_indices.len(), 3);
        assert_eq!(group.qubits.len(), 2);
    }

    #[test]
    fn test_fusion_strategy() {
        let fusion = GateFusion::new(FusionStrategy::Conservative);
        assert_eq!(fusion.max_fusion_qubits, 4);
        assert_eq!(fusion.min_fusion_gates, 2);
    }

    #[test]
    fn test_sparse_matrix_multiplication() {
        let mut builder1 = SparseMatrixBuilder::new(2, 2);
        builder1.set_value(0, 0, Complex64::new(1.0, 0.0));
        builder1.set_value(1, 1, Complex64::new(1.0, 0.0));
        let m1 = builder1.build();

        let mut builder2 = SparseMatrixBuilder::new(2, 2);
        builder2.set_value(0, 1, Complex64::new(1.0, 0.0));
        builder2.set_value(1, 0, Complex64::new(1.0, 0.0));
        let m2 = builder2.build();

        let result = SciRS2MatrixMultiplier::multiply_sparse(&m1, &m2).unwrap();
        assert_eq!(result.num_rows, 2);
        assert_eq!(result.num_cols, 2);
    }

    #[test]
    fn test_fused_gate() {
        let matrix = Array2::eye(4);
        let fused = FusedGate {
            matrix,
            qubits: vec![QubitId::new(0), QubitId::new(1)],
            original_gates: vec![0, 1],
        };

        assert_eq!(fused.dimension(), 4);
        let sparse = fused.to_sparse().unwrap();
        assert_eq!(sparse.num_rows, 4);
    }

    #[test]
    fn test_fusion_cost() {
        let fusion = GateFusion::new(FusionStrategy::Conservative);
        let group = GateGroup {
            gate_indices: vec![0, 1],
            qubits: vec![QubitId::new(0), QubitId::new(1)],
            fusable: false,
            fusion_cost: 0.0,
        };

        let gates: Vec<Box<dyn GateOp>> = vec![
            Box::new(Hadamard {
                target: QubitId::new(0),
            }),
            Box::new(CNOT {
                control: QubitId::new(0),
                target: QubitId::new(1),
            }),
        ];

        let cost = fusion.compute_fusion_cost(&group, &gates).unwrap();
        assert!(cost > 0.0);
    }
}
