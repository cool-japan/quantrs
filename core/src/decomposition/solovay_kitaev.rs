//! Solovay-Kitaev algorithm for approximating arbitrary quantum gates
//!
//! The Solovay-Kitaev algorithm provides an efficient method to approximate
//! any single-qubit gate to arbitrary precision using a finite gate set
//! (typically Clifford+T gates).

use crate::error::{QuantRS2Error, QuantRS2Result};
use crate::gate::{single::*, GateOp};
use crate::matrix_ops::{matrices_approx_equal, DenseMatrix, QuantumMatrix};
use crate::qubit::QubitId;
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use rustc_hash::FxHashMap;
use scirs2_linalg::matrix_norm;
use smallvec::SmallVec;
use std::f64::consts::PI;

/// Configuration for the Solovay-Kitaev algorithm
#[derive(Debug, Clone)]
pub struct SolovayKitaevConfig {
    /// Maximum recursion depth
    pub max_depth: usize,
    /// Precision goal (epsilon)
    pub epsilon: f64,
    /// Base gate set to use
    pub base_set: BaseGateSet,
    /// Cache size limit (number of sequences to store)
    pub cache_limit: usize,
}

impl Default for SolovayKitaevConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            epsilon: 1e-3,
            base_set: BaseGateSet::CliffordT,
            cache_limit: 10000,
        }
    }
}

/// Available base gate sets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseGateSet {
    /// Clifford+T gate set (H, S, T)
    CliffordT,
    /// V basis (H, T, X)
    VBasis,
    /// Custom gate set
    Custom,
}

/// A sequence of gates
pub type GateSequence = SmallVec<[Box<dyn GateOp>; 8]>;

/// Gate sequence with its unitary matrix
#[derive(Debug)]
pub struct GateSequenceWithMatrix {
    /// The gate sequence
    pub sequence: GateSequence,
    /// The unitary matrix of the sequence
    pub matrix: Array2<Complex64>,
    /// The cost (e.g., T-count for Clifford+T)
    pub cost: usize,
}

/// Solovay-Kitaev approximator
pub struct SolovayKitaev {
    config: SolovayKitaevConfig,
    /// Cache of gate sequences by recursion level
    sequence_cache: Vec<Vec<GateSequenceWithMatrix>>,
    /// Lookup table for finding closest sequences
    lookup_table: FxHashMap<u64, Vec<usize>>,
}

impl SolovayKitaev {
    /// Create a new Solovay-Kitaev approximator
    pub fn new(config: SolovayKitaevConfig) -> Self {
        let max_depth = config.max_depth;
        let mut sequence_cache = Vec::with_capacity(max_depth + 1);
        for _ in 0..=max_depth {
            sequence_cache.push(Vec::new());
        }
        
        let mut sk = Self {
            config,
            sequence_cache,
            lookup_table: FxHashMap::default(),
        };
        sk.initialize_base_sequences();
        sk
    }
    
    /// Initialize the base gate sequences
    fn initialize_base_sequences(&mut self) {
        let qubit = QubitId(0);
        let base_gates = match self.config.base_set {
            BaseGateSet::CliffordT => {
                vec![
                    // Basic gates
                    self.create_sequence_with_matrix(vec![Box::new(Hadamard { target: qubit })]),
                    self.create_sequence_with_matrix(vec![Box::new(Phase { target: qubit })]),
                    self.create_sequence_with_matrix(vec![Box::new(T { target: qubit })]),
                    // Combinations
                    self.create_sequence_with_matrix(vec![
                        Box::new(Hadamard { target: qubit }),
                        Box::new(Phase { target: qubit }),
                    ]),
                    self.create_sequence_with_matrix(vec![
                        Box::new(Phase { target: qubit }),
                        Box::new(Hadamard { target: qubit }),
                    ]),
                ]
            }
            BaseGateSet::VBasis => {
                vec![
                    self.create_sequence_with_matrix(vec![Box::new(Hadamard { target: qubit })]),
                    self.create_sequence_with_matrix(vec![Box::new(T { target: qubit })]),
                    self.create_sequence_with_matrix(vec![Box::new(PauliX { target: qubit })]),
                ]
            }
            BaseGateSet::Custom => {
                // User should add custom gates
                vec![]
            }
        };
        
        // Add to level 0 cache
        for seq in base_gates {
            if let Ok(seq) = seq {
                self.sequence_cache[0].push(seq);
            }
        }
        
        // Build sequences for higher levels
        for level in 1..=self.config.max_depth.min(3) {
            self.build_sequences_at_level(level);
        }
    }
    
    /// Create a sequence with its matrix
    fn create_sequence_with_matrix(&self, gates: Vec<Box<dyn GateOp>>) -> QuantRS2Result<GateSequenceWithMatrix> {
        let mut matrix = Array2::eye(2);
        let mut cost = 0;
        
        for gate in &gates {
            let gate_matrix = gate.matrix()?;
            let gate_array = Array2::from_shape_vec((2, 2), gate_matrix)
                .map_err(|e| QuantRS2Error::InvalidInput(e.to_string()))?;
            matrix = matrix.dot(&gate_array);
            
            // Count T gates for cost
            if gate.name() == "T" || gate.name() == "T†" {
                cost += 1;
            }
        }
        
        Ok(GateSequenceWithMatrix {
            sequence: SmallVec::from_vec(gates),
            matrix,
            cost,
        })
    }
    
    /// Build sequences at a given recursion level
    fn build_sequences_at_level(&mut self, level: usize) {
        if level == 0 || level > self.config.max_depth {
            return;
        }
        
        let mut new_sequences = Vec::new();
        
        // Combine sequences from lower levels
        for i in 0..level {
            let j = level - 1 - i;
            let seq1_count = self.sequence_cache[i].len();
            let seq2_count = self.sequence_cache[j].len();
            
            for idx1 in 0..seq1_count {
                for idx2 in 0..seq2_count {
                    let seq1 = &self.sequence_cache[i][idx1];
                    let seq2 = &self.sequence_cache[j][idx2];
                    
                    // Combine sequences
                    let mut combined = SmallVec::new();
                    combined.extend(seq1.sequence.iter().map(|g| g.clone()));
                    combined.extend(seq2.sequence.iter().map(|g| g.clone()));
                    
                    let matrix = seq1.matrix.dot(&seq2.matrix);
                    let cost = seq1.cost + seq2.cost;
                    
                    let new_seq = GateSequenceWithMatrix {
                        sequence: combined,
                        matrix,
                        cost,
                    };
                    
                    // Check if this is a new/better sequence
                    if self.should_add_sequence(&new_seq, &new_sequences) {
                        new_sequences.push(new_seq);
                    }
                    
                    // Limit cache size
                    if new_sequences.len() >= self.config.cache_limit / 10 {
                        break;
                    }
                }
                if new_sequences.len() >= self.config.cache_limit / 10 {
                    break;
                }
            }
        }
        
        self.sequence_cache[level] = new_sequences;
    }
    
    /// Check if a sequence should be added to the cache
    fn should_add_sequence(&self, new_seq: &GateSequenceWithMatrix, existing: &[GateSequenceWithMatrix]) -> bool {
        // Check if we already have a similar but better sequence
        for seq in existing {
            if matrices_approx_equal(&new_seq.matrix.view(), &seq.matrix.view(), 1e-10) 
                && seq.cost <= new_seq.cost {
                return false;
            }
        }
        true
    }
    
    /// Approximate a unitary matrix using the Solovay-Kitaev algorithm
    pub fn approximate(&mut self, target: &ArrayView2<Complex64>) -> QuantRS2Result<GateSequence> {
        if target.shape() != &[2, 2] {
            return Err(QuantRS2Error::InvalidInput(
                "Target must be a 2x2 unitary matrix".to_string(),
            ));
        }
        
        // Check if target is unitary
        let target_dense = DenseMatrix::new(target.to_owned())?;
        if !target_dense.is_unitary(1e-10)? {
            return Err(QuantRS2Error::InvalidInput(
                "Target matrix is not unitary".to_string(),
            ));
        }
        
        // Start the recursive approximation
        let depth = self.calculate_required_depth();
        self.approximate_recursive(target, depth)
    }
    
    /// Calculate the required recursion depth based on epsilon
    fn calculate_required_depth(&self) -> usize {
        // The Solovay-Kitaev theorem guarantees error O(log^c(1/ε))
        // We use a heuristic based on the desired precision
        let log_inv_eps = (1.0 / self.config.epsilon).ln();
        let depth = (log_inv_eps * 2.0) as usize;
        depth.min(self.config.max_depth)
    }
    
    /// Recursive approximation algorithm
    fn approximate_recursive(&mut self, target: &ArrayView2<Complex64>, depth: usize) -> QuantRS2Result<GateSequence> {
        // Base case: find the closest sequence in our cache
        if depth == 0 {
            return self.find_closest_base_sequence(target);
        }
        
        // Find initial approximation at lower depth
        let u_n_minus_1 = self.approximate_recursive(target, depth - 1)?;
        let u_n_minus_1_matrix = self.compute_sequence_matrix(&u_n_minus_1)?;
        
        // Compute the error
        let error = target.to_owned() - &u_n_minus_1_matrix;
        
        // Compute Frobenius norm manually for complex matrices
        let mut error_norm = 0.0;
        for val in error.iter() {
            error_norm += val.norm_sqr();
        }
        let error_norm = error_norm.sqrt();
        
        // If error is small enough, return the approximation
        if error_norm < self.config.epsilon {
            return Ok(u_n_minus_1);
        }
        
        // Otherwise, apply group commutator method
        self.group_commutator_correction(target, &u_n_minus_1, &u_n_minus_1_matrix, depth)
    }
    
    /// Find the closest base sequence to a target unitary
    fn find_closest_base_sequence(&self, target: &ArrayView2<Complex64>) -> QuantRS2Result<GateSequence> {
        let mut best_sequence = None;
        let mut best_distance = f64::INFINITY;
        
        // Search through all cached sequences
        for sequences in &self.sequence_cache {
            for seq in sequences {
                let diff = target.to_owned() - &seq.matrix;
                
                // Compute Frobenius norm manually
                let mut distance = 0.0;
                for val in diff.iter() {
                    distance += val.norm_sqr();
                }
                let distance = distance.sqrt();
                
                if distance < best_distance {
                    best_distance = distance;
                    best_sequence = Some(seq.sequence.iter().map(|g| g.clone()).collect());
                }
            }
        }
        
        best_sequence.ok_or_else(|| {
            QuantRS2Error::ComputationError("No base sequences available".to_string())
        })
    }
    
    /// Apply group commutator correction
    fn group_commutator_correction(
        &mut self,
        target: &ArrayView2<Complex64>,
        base_seq: &GateSequence,
        base_matrix: &Array2<Complex64>,
        depth: usize,
    ) -> QuantRS2Result<GateSequence> {
        // Compute rotation angle of the error
        let error = target.to_owned() - base_matrix;
        let trace = error[[0, 0]] + error[[1, 1]];
        let angle = (trace.re / 2.0).acos();
        
        // Find sequences V and W such that VWV†W† approximates the error rotation
        let (v_seq, w_seq) = self.find_commutator_sequences(angle, depth - 1)?;
        
        // Construct the corrected sequence: U_n = VWV†W†U_{n-1}
        let mut result = SmallVec::new();
        result.extend(v_seq.iter().map(|g| g.clone()));
        result.extend(w_seq.iter().map(|g| g.clone()));
        result.extend(self.compute_inverse_sequence(&v_seq)?);
        result.extend(self.compute_inverse_sequence(&w_seq)?);
        result.extend(base_seq.iter().map(|g| g.clone()));
        
        Ok(result)
    }
    
    /// Find sequences V and W for group commutator
    fn find_commutator_sequences(&mut self, angle: f64, depth: usize) -> QuantRS2Result<(GateSequence, GateSequence)> {
        // This is a simplified implementation
        // In practice, this would involve finding specific sequences that produce the desired rotation
        let qubit = QubitId(0);
        
        // Use small rotations that combine to approximate the angle
        let small_angle = angle / 4.0;
        
        let v = vec![Box::new(RotationZ {
            target: qubit,
            theta: small_angle,
        }) as Box<dyn GateOp>];
        
        let w = vec![Box::new(RotationY {
            target: qubit,
            theta: small_angle,
        }) as Box<dyn GateOp>];
        
        Ok((SmallVec::from_vec(v), SmallVec::from_vec(w)))
    }
    
    /// Compute the matrix for a gate sequence
    fn compute_sequence_matrix(&self, sequence: &GateSequence) -> QuantRS2Result<Array2<Complex64>> {
        let mut matrix = Array2::eye(2);
        
        for gate in sequence {
            let gate_matrix = gate.matrix()?;
            let gate_array = Array2::from_shape_vec((2, 2), gate_matrix)
                .map_err(|e| QuantRS2Error::InvalidInput(e.to_string()))?;
            matrix = matrix.dot(&gate_array);
        }
        
        Ok(matrix)
    }
    
    /// Compute the inverse of a gate sequence
    fn compute_inverse_sequence(&self, sequence: &GateSequence) -> QuantRS2Result<GateSequence> {
        let mut inverse = SmallVec::new();
        
        // Reverse the sequence and invert each gate
        for gate in sequence.iter().rev() {
            inverse.push(self.invert_gate(gate.as_ref())?);
        }
        
        Ok(inverse)
    }
    
    /// Invert a single gate
    fn invert_gate(&self, gate: &dyn GateOp) -> QuantRS2Result<Box<dyn GateOp>> {
        let qubit = gate.qubits()[0]; // Assuming single-qubit gates
        
        match gate.name() {
            "H" => Ok(Box::new(Hadamard { target: qubit })), // H is self-inverse
            "X" => Ok(Box::new(PauliX { target: qubit })),   // X is self-inverse
            "Y" => Ok(Box::new(PauliY { target: qubit })),   // Y is self-inverse
            "Z" => Ok(Box::new(PauliZ { target: qubit })),   // Z is self-inverse
            "S" => Ok(Box::new(PhaseDagger { target: qubit })),
            "S†" => Ok(Box::new(Phase { target: qubit })),
            "T" => Ok(Box::new(TDagger { target: qubit })),
            "T†" => Ok(Box::new(T { target: qubit })),
            _ => {
                // For rotation gates, negate the angle
                if gate.is_parameterized() {
                    // This is a simplified approach - would need proper handling
                    Err(QuantRS2Error::UnsupportedOperation(
                        format!("Cannot invert parameterized gate {}", gate.name()),
                    ))
                } else {
                    Err(QuantRS2Error::UnsupportedOperation(
                        format!("Cannot invert gate {}", gate.name()),
                    ))
                }
            }
        }
    }
}

/// Count the number of T gates in a sequence
pub fn count_t_gates(sequence: &GateSequence) -> usize {
    sequence.iter().filter(|g| g.name() == "T" || g.name() == "T†").count()
}

/// Optimize a gate sequence by combining adjacent gates
pub fn optimize_sequence(sequence: GateSequence) -> GateSequence {
    let mut optimized = SmallVec::new();
    let mut i = 0;
    
    while i < sequence.len() {
        if i + 1 < sequence.len() {
            let gate1 = &sequence[i];
            let gate2 = &sequence[i + 1];
            
            // Check for cancellations
            if gate1.qubits() == gate2.qubits() {
                let combined = match (gate1.name(), gate2.name()) {
                    ("S", "S") => Some("Z"),
                    ("S†", "S†") => Some("Z"),
                    ("S", "S†") | ("S†", "S") => None, // Identity
                    ("T", "T†") | ("T†", "T") => None, // Identity
                    ("H", "H") => None, // Identity
                    _ => Some(""), // No combination
                };
                
                match combined {
                    None => {
                        // Skip both gates (they cancel)
                        i += 2;
                        continue;
                    }
                    Some("Z") => {
                        optimized.push(Box::new(PauliZ { target: gate1.qubits()[0] }) as Box<dyn GateOp>);
                        i += 2;
                        continue;
                    }
                    _ => {
                        // No optimization, keep first gate
                        optimized.push(sequence[i].clone());
                        i += 1;
                    }
                }
            } else {
                optimized.push(sequence[i].clone());
                i += 1;
            }
        } else {
            optimized.push(sequence[i].clone());
            i += 1;
        }
    }
    
    optimized
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_solovay_kitaev_initialization() {
        let config = SolovayKitaevConfig::default();
        let sk = SolovayKitaev::new(config);
        
        // Check that base sequences are initialized
        assert!(!sk.sequence_cache[0].is_empty());
    }
    
    #[test]
    fn test_t_gate_counting() {
        let qubit = QubitId(0);
        let sequence: GateSequence = SmallVec::from_vec(vec![
            Box::new(T { target: qubit }) as Box<dyn GateOp>,
            Box::new(Hadamard { target: qubit }),
            Box::new(TDagger { target: qubit }),
            Box::new(Phase { target: qubit }),
            Box::new(T { target: qubit }),
        ]);
        
        assert_eq!(count_t_gates(&sequence), 3);
    }
    
    #[test]
    fn test_sequence_optimization() {
        let qubit = QubitId(0);
        let sequence: GateSequence = SmallVec::from_vec(vec![
            Box::new(Hadamard { target: qubit }) as Box<dyn GateOp>,
            Box::new(Hadamard { target: qubit }),
            Box::new(Phase { target: qubit }),
            Box::new(PhaseDagger { target: qubit }),
        ]);
        
        let optimized = optimize_sequence(sequence);
        assert_eq!(optimized.len(), 0); // All gates should cancel
    }
}