//! QUBO matrix compression algorithms
//!
//! This module provides compression techniques for QUBO matrices to reduce
//! memory usage and improve computational efficiency for large problems.

use crate::ising::{IsingError, IsingResult};
use std::collections::{HashMap, HashSet};

/// Simple sparse vector implementation
#[derive(Debug, Clone)]
pub struct SparseVector<T> {
    data: HashMap<usize, T>,
    size: usize,
}

impl<T: Clone + Default + PartialEq> SparseVector<T> {
    pub fn new(size: usize) -> Self {
        Self {
            data: HashMap::new(),
            size,
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(&index)
    }

    pub fn set(&mut self, index: usize, value: T) {
        if index < self.size {
            if value == T::default() {
                self.data.remove(&index);
            } else {
                self.data.insert(index, value);
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.data.iter().map(|(&k, v)| (k, v))
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        self.data.remove(&index)
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn filter_inplace<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.data.retain(|_, v| predicate(v));
    }
}

/// Simple COO sparse matrix implementation
#[derive(Debug, Clone)]
pub struct CooMatrix<T> {
    data: HashMap<(usize, usize), T>,
    rows: usize,
    cols: usize,
}

impl<T: Clone + Default + PartialEq> CooMatrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: HashMap::new(),
            rows,
            cols,
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(&(row, col))
    }

    pub fn set(&mut self, row: usize, col: usize, value: T) {
        if row < self.rows && col < self.cols {
            if value == T::default() {
                self.data.remove(&(row, col));
            } else {
                self.data.insert((row, col), value);
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        self.data.iter().map(|(&(i, j), v)| (i, j, v))
    }

    pub fn retain<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&(usize, usize), &mut T) -> bool,
    {
        self.data.retain(predicate);
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn filter_inplace<F>(&mut self, mut predicate: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.data.retain(|_, v| predicate(v));
    }
}

/// Compressed QUBO representation using various sparse formats
#[derive(Debug, Clone)]
pub struct CompressedQubo {
    /// Number of variables
    pub num_vars: usize,
    /// Linear terms (diagonal elements) as sparse vector
    pub linear_terms: SparseVector<f64>,
    /// Quadratic terms (off-diagonal elements) as COO sparse matrix
    /// Stored as upper triangular: (i, j) where i < j
    pub quadratic_terms: CooMatrix<f64>,
    /// Constant offset
    pub offset: f64,
    /// Compression statistics
    pub stats: CompressionStats,
}

/// Statistics about compression
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Original number of non-zero elements
    pub original_nnz: usize,
    /// Compressed number of non-zero elements
    pub compressed_nnz: usize,
    /// Number of merged variables
    pub merged_vars: usize,
    /// Number of eliminated variables
    pub eliminated_vars: usize,
    /// Compression ratio
    pub compression_ratio: f64,
}

impl CompressedQubo {
    /// Create a new compressed QUBO
    pub fn new(num_vars: usize) -> Self {
        Self {
            num_vars,
            linear_terms: SparseVector::new(num_vars),
            quadratic_terms: CooMatrix::new(num_vars, num_vars),
            offset: 0.0,
            stats: CompressionStats::default(),
        }
    }

    /// Add or update a linear term
    pub fn add_linear(&mut self, var: usize, coefficient: f64) {
        if coefficient.abs() > 1e-10 {
            let current_value = *self.linear_terms.get(var).unwrap_or(&0.0);
            self.linear_terms.set(var, current_value + coefficient);
        }
    }

    /// Add or update a quadratic term
    pub fn add_quadratic(&mut self, i: usize, j: usize, coefficient: f64) {
        if i == j {
            self.add_linear(i, coefficient);
        } else if coefficient.abs() > 1e-10 {
            let (i, j) = if i < j { (i, j) } else { (j, i) };
            let current_value = *self.quadratic_terms.get(i, j).unwrap_or(&0.0);
            self.quadratic_terms.set(i, j, current_value + coefficient);
        }
    }

    /// Get the energy for a given solution
    pub fn evaluate(&self, solution: &[bool]) -> f64 {
        let mut energy = self.offset;

        // Linear terms
        for (var, coeff) in self.linear_terms.iter() {
            if var < solution.len() && solution[var] {
                energy += coeff;
            }
        }

        // Quadratic terms
        for (i, j, coeff) in self.quadratic_terms.iter() {
            if i < solution.len() && j < solution.len() && solution[i] && solution[j] {
                energy += coeff;
            }
        }

        energy
    }

    /// Get total number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.linear_terms.nnz() + self.quadratic_terms.nnz()
    }

    /// Apply threshold to remove small coefficients
    pub fn apply_threshold(&mut self, threshold: f64) {
        self.linear_terms.filter_inplace(|&v| v.abs() > threshold);
        self.quadratic_terms
            .filter_inplace(|&v| v.abs() > threshold);
    }
}

/// QUBO compression using Coordinate (COO) format
pub struct CooCompressor {
    /// Threshold for considering coefficients as zero
    pub zero_threshold: f64,
    /// Whether to sort by magnitude for better compression
    pub sort_by_magnitude: bool,
}

impl Default for CooCompressor {
    fn default() -> Self {
        Self {
            zero_threshold: 1e-10,
            sort_by_magnitude: true,
        }
    }
}

impl CooCompressor {
    /// Compress a QUBO matrix from dense format
    pub fn compress_dense(&self, matrix: &[Vec<f64>], offset: f64) -> IsingResult<CompressedQubo> {
        let n = matrix.len();
        let mut compressed = CompressedQubo::new(n);
        compressed.offset = offset;

        let mut original_nnz = 0;

        // Extract non-zero elements
        for i in 0..n {
            for j in i..n {
                let value = matrix[i][j];
                if value.abs() > self.zero_threshold {
                    original_nnz += 1;
                    if i == j {
                        compressed.add_linear(i, value);
                    } else {
                        compressed.add_quadratic(i, j, value);
                    }
                }
            }
        }

        // Update statistics
        compressed.stats.original_nnz = original_nnz;
        compressed.stats.compressed_nnz = compressed.nnz();
        compressed.stats.compression_ratio = if original_nnz > 0 {
            compressed.stats.compressed_nnz as f64 / original_nnz as f64
        } else {
            1.0
        };

        Ok(compressed)
    }

    /// Compress from edge list format
    pub fn compress_edges(
        &self,
        num_vars: usize,
        edges: &[(usize, usize, f64)],
        offset: f64,
    ) -> IsingResult<CompressedQubo> {
        let mut compressed = CompressedQubo::new(num_vars);
        compressed.offset = offset;

        for &(i, j, weight) in edges {
            if weight.abs() > self.zero_threshold {
                compressed.add_quadratic(i, j, weight);
            }
        }

        compressed.stats.original_nnz = edges.len();
        compressed.stats.compressed_nnz = compressed.nnz();
        compressed.stats.compression_ratio = if edges.len() > 0 {
            compressed.stats.compressed_nnz as f64 / edges.len() as f64
        } else {
            1.0
        };

        Ok(compressed)
    }
}

/// Variable reduction techniques for QUBO compression
pub struct VariableReducer {
    /// Threshold for considering variables as fixed
    pub fixing_threshold: f64,
    /// Enable variable merging
    pub enable_merging: bool,
    /// Enable variable elimination
    pub enable_elimination: bool,
}

impl Default for VariableReducer {
    fn default() -> Self {
        Self {
            fixing_threshold: 1e6,
            enable_merging: true,
            enable_elimination: true,
        }
    }
}

impl VariableReducer {
    /// Reduce QUBO by fixing, merging, and eliminating variables
    pub fn reduce(&self, qubo: &mut CompressedQubo) -> IsingResult<ReductionMapping> {
        let mut mapping = ReductionMapping::new(qubo.num_vars);

        // Fix variables with very large linear coefficients
        if self.enable_elimination {
            self.fix_variables(qubo, &mut mapping)?;
        }

        // Merge equivalent variables
        if self.enable_merging {
            self.merge_equivalent_variables(qubo, &mut mapping)?;
        }

        // Eliminate variables with degree 1
        if self.enable_elimination {
            self.eliminate_degree_one_variables(qubo, &mut mapping)?;
        }

        Ok(mapping)
    }

    /// Fix variables with large coefficients
    fn fix_variables(
        &self,
        qubo: &mut CompressedQubo,
        mapping: &mut ReductionMapping,
    ) -> IsingResult<()> {
        let mut to_fix = Vec::new();

        // Find variables to fix
        for (var, coeff) in qubo.linear_terms.iter() {
            if coeff.abs() > self.fixing_threshold {
                // Fix to 0 if coefficient is positive, 1 if negative
                let value = *coeff < 0.0;
                to_fix.push((var, value));
            }
        }

        // Apply fixings
        for (var, value) in to_fix {
            self.fix_variable(qubo, mapping, var, value)?;
            qubo.stats.eliminated_vars += 1;
        }

        Ok(())
    }

    /// Fix a single variable and update the QUBO
    fn fix_variable(
        &self,
        qubo: &mut CompressedQubo,
        mapping: &mut ReductionMapping,
        var: usize,
        value: bool,
    ) -> IsingResult<()> {
        mapping.fix_variable(var, value);

        // Update offset if variable is set to 1
        if value {
            let linear_coeff = *qubo.linear_terms.get(var).unwrap_or(&0.0);
            qubo.offset += linear_coeff;

            // Update other linear terms
            let mut updates = Vec::new();
            for (i, j, coeff) in qubo.quadratic_terms.iter() {
                if i == var {
                    updates.push((j, *coeff));
                } else if j == var {
                    updates.push((i, *coeff));
                }
            }

            for (other_var, coeff) in updates {
                qubo.add_linear(other_var, coeff);
            }
        }

        // Remove variable from QUBO
        qubo.linear_terms.remove(var);
        qubo.quadratic_terms.filter_inplace(|_| true); // Remove terms involving var
                                                       // Note: We need a more specific filter for removing terms involving var
        let mut new_quadratic = CooMatrix::new(qubo.num_vars, qubo.num_vars);
        for (i, j, coeff) in qubo.quadratic_terms.iter() {
            if i != var && j != var {
                new_quadratic.set(i, j, *coeff);
            }
        }
        qubo.quadratic_terms = new_quadratic;

        Ok(())
    }

    /// Merge variables that appear with the same coefficients
    fn merge_equivalent_variables(
        &self,
        qubo: &mut CompressedQubo,
        mapping: &mut ReductionMapping,
    ) -> IsingResult<()> {
        // Build variable signature based on coefficients
        let signatures = self.compute_variable_signatures(qubo);

        // Group variables by signature
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (var, sig) in signatures {
            groups.entry(sig).or_insert_with(Vec::new).push(var);
        }

        // Merge equivalent variables
        for (_, vars) in groups {
            if vars.len() > 1 {
                let representative = vars[0];
                for &var in &vars[1..] {
                    self.merge_variables(qubo, mapping, var, representative)?;
                    qubo.stats.merged_vars += 1;
                }
            }
        }

        Ok(())
    }

    /// Compute signature for each variable based on its coefficients
    fn compute_variable_signatures(&self, qubo: &CompressedQubo) -> HashMap<usize, String> {
        let mut signatures = HashMap::new();

        for var in 0..qubo.num_vars {
            let mut sig_parts = Vec::new();

            // Linear coefficient
            let linear_coeff = *qubo.linear_terms.get(var).unwrap_or(&0.0);
            if linear_coeff.abs() > 1e-10 {
                sig_parts.push(format!("L:{:.6}", linear_coeff));
            }

            // Quadratic coefficients
            let mut quad_coeffs = Vec::new();
            for (i, j, coeff) in qubo.quadratic_terms.iter() {
                if i == var || j == var {
                    let other = if i == var { j } else { i };
                    quad_coeffs.push((other, coeff));
                }
            }
            quad_coeffs.sort_by_key(|&(v, _)| v);

            for (other, coeff) in quad_coeffs {
                sig_parts.push(format!("Q{}:{:.6}", other, coeff));
            }

            signatures.insert(var, sig_parts.join("|"));
        }

        signatures
    }

    /// Merge two variables
    fn merge_variables(
        &self,
        qubo: &mut CompressedQubo,
        mapping: &mut ReductionMapping,
        from_var: usize,
        to_var: usize,
    ) -> IsingResult<()> {
        mapping.merge_variables(from_var, to_var);

        // Merge linear terms
        let from_coeff = *qubo.linear_terms.get(from_var).unwrap_or(&0.0);
        qubo.linear_terms.remove(from_var);
        if from_coeff.abs() > 1e-10 {
            qubo.add_linear(to_var, from_coeff);
        }

        // Update quadratic terms
        let mut updates = Vec::new();
        let mut new_quadratic = CooMatrix::new(qubo.num_vars, qubo.num_vars);

        for (i, j, coeff) in qubo.quadratic_terms.iter() {
            if i == from_var || j == from_var {
                let new_i = if i == from_var { to_var } else { i };
                let new_j = if j == from_var { to_var } else { j };
                updates.push((new_i, new_j, *coeff));
            } else {
                new_quadratic.set(i, j, *coeff);
            }
        }

        // Set the new quadratic matrix
        qubo.quadratic_terms = new_quadratic;

        // Add updated terms
        for (i, j, coeff) in updates {
            qubo.add_quadratic(i, j, coeff);
        }

        Ok(())
    }

    /// Eliminate variables with degree 1
    fn eliminate_degree_one_variables(
        &self,
        qubo: &mut CompressedQubo,
        mapping: &mut ReductionMapping,
    ) -> IsingResult<()> {
        loop {
            let mut eliminated = false;

            // Find degree-1 variables
            let degrees = self.compute_variable_degrees(qubo);

            for (var, degree) in degrees {
                if degree == 1 && !mapping.is_reduced(var) {
                    // Find the connected variable
                    let mut connected_var = None;
                    let mut connection_coeff = 0.0;

                    for (i, j, coeff) in qubo.quadratic_terms.iter() {
                        if i == var {
                            connected_var = Some(j);
                            connection_coeff = *coeff;
                            break;
                        } else if j == var {
                            connected_var = Some(i);
                            connection_coeff = *coeff;
                            break;
                        }
                    }

                    if let Some(other) = connected_var {
                        // Determine optimal value based on coefficients
                        let linear_coeff = *qubo.linear_terms.get(var).unwrap_or(&0.0);

                        // If connection is positive, variables should have opposite values
                        // If connection is negative, variables should have same value
                        if connection_coeff > 0.0 {
                            // Variables should be opposite
                            mapping.merge_variables_negated(var, other);
                        } else {
                            // Variables should be same
                            mapping.merge_variables(var, other);
                        }

                        // Update QUBO
                        qubo.linear_terms.remove(var);
                        let mut new_quadratic = CooMatrix::new(qubo.num_vars, qubo.num_vars);
                        for (i, j, coeff) in qubo.quadratic_terms.iter() {
                            if i != var && j != var {
                                new_quadratic.set(i, j, *coeff);
                            }
                        }
                        qubo.quadratic_terms = new_quadratic;

                        // Update linear term of connected variable
                        if linear_coeff.abs() > 1e-10 {
                            if connection_coeff > 0.0 {
                                // Opposite values: adjust offset
                                qubo.offset += linear_coeff.min(0.0);
                                qubo.add_linear(
                                    other,
                                    -(linear_coeff.abs() as f64).min(connection_coeff.abs() as f64),
                                );
                            } else {
                                // Same values: just add linear coefficient
                                qubo.add_linear(other, linear_coeff);
                            }
                        }

                        eliminated = true;
                        qubo.stats.eliminated_vars += 1;
                        break;
                    }
                }
            }

            if !eliminated {
                break;
            }
        }

        Ok(())
    }

    /// Compute degree of each variable
    fn compute_variable_degrees(&self, qubo: &CompressedQubo) -> HashMap<usize, usize> {
        let mut degrees = HashMap::new();

        for (var, _) in qubo.linear_terms.iter() {
            degrees.insert(var, 0);
        }

        for (i, j, _) in qubo.quadratic_terms.iter() {
            *degrees.entry(i).or_insert(0) += 1;
            *degrees.entry(j).or_insert(0) += 1;
        }

        degrees
    }
}

/// Mapping from original to reduced variables
#[derive(Debug, Clone)]
pub struct ReductionMapping {
    /// Maps original variable to (representative, negated)
    /// If negated is true, the variable equals NOT(representative)
    pub variable_map: HashMap<usize, (usize, bool)>,
    /// Fixed variables
    pub fixed_vars: HashMap<usize, bool>,
    /// Original number of variables
    pub original_vars: usize,
}

impl ReductionMapping {
    /// Create a new identity mapping
    pub fn new(num_vars: usize) -> Self {
        let mut variable_map = HashMap::new();
        for i in 0..num_vars {
            variable_map.insert(i, (i, false));
        }

        Self {
            variable_map,
            fixed_vars: HashMap::new(),
            original_vars: num_vars,
        }
    }

    /// Fix a variable to a specific value
    pub fn fix_variable(&mut self, var: usize, value: bool) {
        self.fixed_vars.insert(var, value);
        self.variable_map.remove(&var);
    }

    /// Merge two variables (they take the same value)
    pub fn merge_variables(&mut self, from: usize, to: usize) {
        if let Some((repr, negated)) = self.variable_map.get(&to).copied() {
            self.variable_map.insert(from, (repr, negated));
        } else {
            self.variable_map.insert(from, (to, false));
        }
    }

    /// Merge two variables with negation (from = NOT(to))
    pub fn merge_variables_negated(&mut self, from: usize, to: usize) {
        if let Some((repr, negated)) = self.variable_map.get(&to).copied() {
            self.variable_map.insert(from, (repr, !negated));
        } else {
            self.variable_map.insert(from, (to, true));
        }
    }

    /// Check if a variable has been reduced
    pub fn is_reduced(&self, var: usize) -> bool {
        self.fixed_vars.contains_key(&var)
            || self
                .variable_map
                .get(&var)
                .map_or(false, |&(repr, _)| repr != var)
    }

    /// Map solution from reduced to original variables
    pub fn expand_solution(&self, reduced_solution: &[bool]) -> Vec<bool> {
        let mut solution = vec![false; self.original_vars];

        // Set fixed variables
        for (&var, &value) in &self.fixed_vars {
            if var < solution.len() {
                solution[var] = value;
            }
        }

        // Set mapped variables
        for (&var, &(repr, negated)) in &self.variable_map {
            if var < solution.len() && repr < reduced_solution.len() {
                solution[var] = if negated {
                    !reduced_solution[repr]
                } else {
                    reduced_solution[repr]
                };
            }
        }

        solution
    }
}

/// Block structure detection for QUBO matrices
pub struct BlockDetector {
    /// Minimum block size to consider
    pub min_block_size: usize,
    /// Threshold for considering blocks as independent
    pub independence_threshold: f64,
}

impl Default for BlockDetector {
    fn default() -> Self {
        Self {
            min_block_size: 3,
            independence_threshold: 0.01,
        }
    }
}

impl BlockDetector {
    /// Detect block structure in QUBO
    pub fn detect_blocks(&self, qubo: &CompressedQubo) -> Vec<Vec<usize>> {
        let mut blocks = Vec::new();
        let mut unassigned: HashSet<usize> = (0..qubo.num_vars).collect();

        while !unassigned.is_empty() {
            // Start a new block with an arbitrary unassigned variable
            let start = *unassigned.iter().next().unwrap();
            let mut block = vec![start];
            let mut to_process = vec![start];
            unassigned.remove(&start);

            // Grow block using BFS
            while let Some(var) = to_process.pop() {
                // Find connected variables
                for (i, j, coeff) in qubo.quadratic_terms.iter() {
                    if coeff.abs() > self.independence_threshold {
                        let connected = if i == var && unassigned.contains(&j) {
                            Some(j)
                        } else if j == var && unassigned.contains(&i) {
                            Some(i)
                        } else {
                            None
                        };

                        if let Some(connected_var) = connected {
                            block.push(connected_var);
                            to_process.push(connected_var);
                            unassigned.remove(&connected_var);
                        }
                    }
                }
            }

            if block.len() >= self.min_block_size {
                blocks.push(block);
            } else {
                // Merge small blocks with the last block if possible
                if let Some(last_block) = blocks.last_mut() {
                    last_block.extend(block);
                } else {
                    blocks.push(block);
                }
            }
        }

        blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_compression() {
        let matrix = vec![
            vec![1.0, 0.0, -0.5],
            vec![0.0, 2.0, 0.0],
            vec![-0.5, 0.0, 3.0],
        ];

        let compressor = CooCompressor::default();
        let compressed = compressor.compress_dense(&matrix, 0.5).unwrap();

        assert_eq!(compressed.num_vars, 3);
        assert_eq!(compressed.offset, 0.5);
        assert_eq!(*compressed.linear_terms.get(0).unwrap_or(&0.0), 1.0);
        assert_eq!(*compressed.linear_terms.get(1).unwrap_or(&0.0), 2.0);
        assert_eq!(*compressed.linear_terms.get(2).unwrap_or(&0.0), 3.0);
        assert_eq!(*compressed.quadratic_terms.get(0, 2).unwrap_or(&0.0), -0.5);
    }

    #[test]
    fn test_variable_reduction() {
        let mut qubo = CompressedQubo::new(4);
        qubo.add_linear(0, 1e7); // Should be fixed to 0 (large positive)
        qubo.add_linear(1, -1e7); // Should be fixed to 1 (large negative)
        qubo.add_quadratic(2, 3, -1.0);

        let reducer = VariableReducer::default();
        let mapping = reducer.reduce(&mut qubo).unwrap();

        println!("Fixed vars: {:?}", mapping.fixed_vars);
        println!("Fixing threshold: {}", reducer.fixing_threshold);

        assert!(mapping.fixed_vars.contains_key(&0));
        assert!(!mapping.fixed_vars[&0]); // Fixed to 0
        assert!(mapping.fixed_vars.contains_key(&1));
        assert!(mapping.fixed_vars[&1]); // Fixed to 1
    }

    #[test]
    fn test_block_detection() {
        let mut qubo = CompressedQubo::new(6);
        // Block 1: 0-1-2
        qubo.add_quadratic(0, 1, 1.0);
        qubo.add_quadratic(1, 2, 1.0);
        // Block 2: 3-4-5
        qubo.add_quadratic(3, 4, 1.0);
        qubo.add_quadratic(4, 5, 1.0);

        let detector = BlockDetector::default();
        let blocks = detector.detect_blocks(&qubo);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].len(), 3);
        assert_eq!(blocks[1].len(), 3);
    }
}
