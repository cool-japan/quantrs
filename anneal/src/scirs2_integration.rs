//! `SciRS2` Integration Module
//!
//! This module provides comprehensive integration between QuantRS2-Anneal and the `SciRS2` scientific
//! computing framework. It implements full `SciRS2` integration including:
//! - Sparse matrix operations using scirs2-sparse for QUBO models
//! - Graph algorithms using scirs2-graph for embedding and partitioning  
//! - Statistical analysis using scirs2-stats for solution quality evaluation
//! - Advanced analytics and performance monitoring
//!
//! This integration enhances the quantum annealing framework with high-performance
//! scientific computing capabilities from the `SciRS2` ecosystem.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use crate::applications::{ApplicationError, ApplicationResult};
use crate::ising::{IsingModel, QuboModel};

// SciRS2 imports - FULL SciRS2 POLICY COMPLIANCE
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Zero;
use scirs2_graph::Graph;
use scirs2_sparse::csr_array::CsrArray;
use scirs2_sparse::SparseArray;
use scirs2_stats::{mean, std};

/// SciRS2-enhanced sparse QUBO model using proper sparse matrices
#[derive(Debug, Clone)]
pub struct SciRS2QuboModel {
    /// Number of variables
    pub num_variables: usize,
    /// Linear terms as dense array (typically sparse in practice)
    pub linear_terms: Array1<f64>,
    /// Quadratic terms using `SciRS2` sparse CSR matrix
    pub quadratic_matrix: CsrArray<f64>,
    /// Constant offset
    pub offset: f64,
}

impl SciRS2QuboModel {
    /// Create a new SciRS2-enhanced QUBO model
    pub fn new(num_variables: usize) -> ApplicationResult<Self> {
        let linear_terms = Array1::zeros(num_variables);

        // Create empty sparse matrix using scirs2-sparse from triplets
        let quadratic_matrix =
            CsrArray::from_triplets(&[], &[], &[], (num_variables, num_variables), false).map_err(
                |e| {
                    ApplicationError::InvalidConfiguration(format!(
                        "Failed to create empty sparse matrix: {e:?}"
                    ))
                },
            )?;

        Ok(Self {
            num_variables,
            linear_terms,
            quadratic_matrix,
            offset: 0.0,
        })
    }

    /// Create from existing QUBO model with proper sparse matrix construction
    pub fn from_qubo(qubo: &QuboModel) -> ApplicationResult<Self> {
        let num_vars = qubo.num_variables;
        let mut linear_terms = Array1::zeros(num_vars);

        // Extract linear terms
        for i in 0..num_vars {
            if let Ok(value) = qubo.get_linear(i) {
                linear_terms[i] = value;
            }
        }

        // Build sparse quadratic matrix from QUBO using triplet format (COO)
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..num_vars {
            for j in i..num_vars {
                if let Ok(value) = qubo.get_quadratic(i, j) {
                    if value.abs() > 1e-12 {
                        row_indices.push(i);
                        col_indices.push(j);
                        values.push(value);

                        // Add symmetric entry for i != j
                        if i != j {
                            row_indices.push(j);
                            col_indices.push(i);
                            values.push(value);
                        }
                    }
                }
            }
        }

        // Create sparse matrix from triplets using scirs2-sparse
        let quadratic_matrix = CsrArray::from_triplets(
            &row_indices,
            &col_indices,
            &values,
            (num_vars, num_vars),
            false, // not sorted yet
        )
        .map_err(|e| {
            ApplicationError::InvalidConfiguration(format!("Failed to create sparse matrix: {e:?}"))
        })?;

        Ok(Self {
            num_variables: num_vars,
            linear_terms,
            quadratic_matrix,
            offset: qubo.offset,
        })
    }

    /// Set linear coefficient using `SciRS2` array operations
    pub fn set_linear(&mut self, var: usize, value: f64) -> ApplicationResult<()> {
        if var >= self.num_variables {
            return Err(ApplicationError::InvalidConfiguration(format!(
                "Variable index {var} out of range"
            )));
        }
        self.linear_terms[var] = value;
        Ok(())
    }

    /// Set quadratic coefficient by updating sparse matrix
    pub fn set_quadratic(&mut self, var1: usize, var2: usize, value: f64) -> ApplicationResult<()> {
        if var1 >= self.num_variables || var2 >= self.num_variables {
            return Err(ApplicationError::InvalidConfiguration(
                "Variable index out of range".to_string(),
            ));
        }

        // Extract existing entries using CSR structure
        let mut data: HashMap<(usize, usize), f64> = HashMap::new();

        let indices = self.quadratic_matrix.get_indices();
        let indptr = self.quadratic_matrix.get_indptr();
        let values_arr = self.quadratic_matrix.get_data();

        for row in 0..self.num_variables {
            let start = indptr[row];
            let end = indptr[row + 1];
            for idx in start..end {
                let col = indices[idx];
                data.insert((row, col), values_arr[idx]);
            }
        }

        // Update the specified entry
        if value.abs() < 1e-12 {
            data.remove(&(var1, var2));
            data.remove(&(var2, var1));
        } else {
            data.insert((var1, var2), value);
            if var1 != var2 {
                data.insert((var2, var1), value);
            }
        }

        // Rebuild sparse matrix
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();

        for ((i, j), v) in &data {
            rows.push(*i);
            cols.push(*j);
            vals.push(*v);
        }

        self.quadratic_matrix = CsrArray::from_triplets(
            &rows,
            &cols,
            &vals,
            (self.num_variables, self.num_variables),
            false,
        )
        .map_err(|e| {
            ApplicationError::InvalidConfiguration(format!("Failed to update sparse matrix: {e:?}"))
        })?;

        Ok(())
    }

    /// Evaluate QUBO objective using `SciRS2` sparse matrix operations
    pub fn evaluate(&self, solution: &[i8]) -> ApplicationResult<f64> {
        if solution.len() != self.num_variables {
            return Err(ApplicationError::InvalidConfiguration(
                "Solution length mismatch".to_string(),
            ));
        }

        let mut energy = self.offset;

        // Linear terms using SciRS2 array operations
        for (i, &value) in self.linear_terms.iter().enumerate() {
            energy += value * f64::from(solution[i]);
        }

        // Quadratic terms using CSR structure directly
        let sol_array: Vec<f64> = solution.iter().map(|&x| f64::from(x)).collect();
        let indices = self.quadratic_matrix.get_indices();
        let indptr = self.quadratic_matrix.get_indptr();
        let values = self.quadratic_matrix.get_data();

        for i in 0..self.num_variables {
            let start = indptr[i];
            let end = indptr[i + 1];
            for idx in start..end {
                let j = indices[idx];
                let q_ij = values[idx];
                if i == j {
                    energy += q_ij * sol_array[i] * sol_array[j];
                } else {
                    // Count each off-diagonal element once (it's stored symmetrically)
                    energy += 0.5 * q_ij * sol_array[i] * sol_array[j];
                }
            }
        }

        Ok(energy)
    }

    /// Get problem statistics using `SciRS2` operations
    #[must_use]
    pub fn get_statistics(&self) -> QuboStatistics {
        let num_linear_terms = self
            .linear_terms
            .iter()
            .filter(|&&x| x.abs() > 1e-12)
            .count();
        let nnz = self.quadratic_matrix.nnz();

        // Count diagonal elements
        let indices = self.quadratic_matrix.get_indices();
        let indptr = self.quadratic_matrix.get_indptr();
        let mut diag_count = 0;
        for i in 0..self.num_variables {
            let start = indptr[i];
            let end = indptr[i + 1];
            for idx in start..end {
                if indices[idx] == i {
                    diag_count += 1;
                }
            }
        }

        // For symmetric matrix, count unique entries (upper triangle)
        let num_quadratic_terms = usize::midpoint(nnz, diag_count);
        let total_terms = num_linear_terms + num_quadratic_terms;

        let density = if self.num_variables > 0 {
            total_terms as f64 / (self.num_variables * (self.num_variables + 1) / 2) as f64
        } else {
            0.0
        };

        QuboStatistics {
            num_variables: self.num_variables,
            num_linear_terms,
            num_quadratic_terms,
            total_terms,
            density,
            memory_usage: self.estimate_memory_usage(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        // Sparse matrix memory + linear terms
        let linear_mem = self.linear_terms.len() * std::mem::size_of::<f64>();
        let sparse_mem = self.quadratic_matrix.nnz()
            * (std::mem::size_of::<f64>() + std::mem::size_of::<usize>())
            + self.num_variables * std::mem::size_of::<usize>(); // row pointers
        linear_mem + sparse_mem
    }

    /// Iterate over the stored (non-zero) quadratic entries as
    /// `(row, col, value)` triples. The matrix is stored symmetrically, so each
    /// off-diagonal pair appears twice (once as `(i, j)` and once as `(j, i)`).
    pub fn iter_quadratic(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.iter_nonzeros()
    }

    /// Helper to iterate over non-zero elements in COO format
    fn iter_nonzeros(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        let indices = self.quadratic_matrix.get_indices();
        let indptr = self.quadratic_matrix.get_indptr();
        let values = self.quadratic_matrix.get_data();

        (0..self.num_variables).flat_map(move |i| {
            let start = indptr[i];
            let end = indptr[i + 1];
            (start..end).map(move |idx| (i, indices[idx], values[idx]))
        })
    }
}

/// Statistics for QUBO problems computed with `SciRS2`
#[derive(Debug, Clone)]
pub struct QuboStatistics {
    pub num_variables: usize,
    pub num_linear_terms: usize,
    pub num_quadratic_terms: usize,
    pub total_terms: usize,
    pub density: f64,
    pub memory_usage: usize,
}

/// SciRS2-enhanced graph analyzer using scirs2-graph algorithms
pub struct SciRS2GraphAnalyzer {
    /// Graph representation using scirs2-graph
    pub graph: Option<Graph<usize, f64, usize>>,
    /// Computed metrics
    pub metrics: GraphMetrics,
}

impl SciRS2GraphAnalyzer {
    /// Create new graph analyzer
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: None,
            metrics: GraphMetrics::default(),
        }
    }

    /// Analyze problem graph for embedding using scirs2-graph algorithms
    pub fn analyze_problem_graph(
        &mut self,
        qubo: &SciRS2QuboModel,
    ) -> ApplicationResult<GraphAnalysisResult> {
        // Build graph from quadratic terms using scirs2-graph
        let mut graph = Graph::new();

        // Add nodes
        for i in 0..qubo.num_variables {
            graph.add_node(i);
        }

        // Add edges from non-zero quadratic terms, tracking real adjacency
        // (used below for a real clustering-coefficient computation).
        let mut num_edges = 0;
        let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (i, j, weight) in qubo.iter_nonzeros() {
            if i < j && weight.abs() > 1e-12 {
                graph.add_edge(i, j, weight);
                num_edges += 1;
                adjacency.entry(i).or_default().insert(j);
                adjacency.entry(j).or_default().insert(i);
            }
        }

        let num_nodes = qubo.num_variables;

        // Compute basic metrics
        let avg_degree = if num_nodes > 0 {
            2.0 * num_edges as f64 / num_nodes as f64
        } else {
            0.0
        };

        // Real global clustering coefficient (transitivity), computed from
        // the problem graph's own adjacency rather than a fixed constant.
        let clustering_coef = Self::estimate_clustering(&adjacency);

        let connectivity = self.compute_connectivity(num_nodes, num_edges);

        self.metrics = GraphMetrics {
            num_nodes,
            num_edges,
            avg_degree,
            connectivity,
            clustering_coefficient: clustering_coef,
        };

        self.graph = Some(graph);

        Ok(GraphAnalysisResult {
            metrics: self.metrics.clone(),
            embedding_difficulty: self.assess_embedding_difficulty(),
            recommended_chain_strength: self.recommend_chain_strength(),
        })
    }

    fn compute_connectivity(&self, num_nodes: usize, num_edges: usize) -> f64 {
        if num_nodes <= 1 {
            return 0.0;
        }
        let max_edges = num_nodes * (num_nodes - 1) / 2;
        if max_edges == 0 {
            return 0.0;
        }
        num_edges as f64 / max_edges as f64
    }

    /// Compute the real global clustering coefficient (transitivity) of the
    /// problem graph: `3 * (# triangles) / (# connected triples)`, evaluated
    /// directly over `adjacency` rather than returning a fixed constant.
    ///
    /// For each node, we count how many of its neighbor pairs are themselves
    /// connected (closed triples) versus the total number of neighbor pairs
    /// (open + closed triples, i.e. `C(deg(v), 2)`). Summed over all nodes,
    /// `closed / total` is exactly the standard transitivity ratio (both
    /// numerator and denominator count each triangle/triple the same way,
    /// once per constituent vertex, so the usual factor-of-3 cancels).
    fn estimate_clustering(adjacency: &HashMap<usize, HashSet<usize>>) -> f64 {
        let mut closed_triples: u64 = 0;
        let mut total_triples: u64 = 0;

        for neighbors in adjacency.values() {
            let neighbor_list: Vec<usize> = neighbors.iter().copied().collect();
            let degree = neighbor_list.len();
            if degree < 2 {
                continue;
            }

            for a in 0..neighbor_list.len() {
                for b in (a + 1)..neighbor_list.len() {
                    total_triples += 1;
                    let (u, v) = (neighbor_list[a], neighbor_list[b]);
                    let connected = adjacency.get(&u).is_some_and(|n| n.contains(&v));
                    if connected {
                        closed_triples += 1;
                    }
                }
            }
        }

        if total_triples == 0 {
            0.0
        } else {
            closed_triples as f64 / total_triples as f64
        }
    }

    fn assess_embedding_difficulty(&self) -> EmbeddingDifficulty {
        // Use graph metrics to assess difficulty
        let avg_deg = self.metrics.avg_degree;
        let clustering = self.metrics.clustering_coefficient;

        // High degree and high clustering indicate hard embedding
        if avg_deg > 6.0 || clustering > 0.6 {
            EmbeddingDifficulty::Hard
        } else if avg_deg > 3.0 || clustering > 0.3 {
            EmbeddingDifficulty::Medium
        } else {
            EmbeddingDifficulty::Easy
        }
    }

    fn recommend_chain_strength(&self) -> f64 {
        // Base chain strength on graph properties
        // Higher connectivity and clustering require stronger chains
        let base_strength = 1.0;
        let connectivity_factor = 2.0 * self.metrics.connectivity;
        let clustering_factor = 1.5 * self.metrics.clustering_coefficient;

        base_strength + connectivity_factor + clustering_factor
    }
}

impl Default for SciRS2GraphAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Graph metrics computed with scirs2-graph
#[derive(Debug, Clone, Default)]
pub struct GraphMetrics {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f64,
    pub connectivity: f64,
    pub clustering_coefficient: f64,
}

/// Result of graph analysis for embedding
#[derive(Debug, Clone)]
pub struct GraphAnalysisResult {
    pub metrics: GraphMetrics,
    pub embedding_difficulty: EmbeddingDifficulty,
    pub recommended_chain_strength: f64,
}

/// Difficulty assessment for embedding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingDifficulty {
    Easy,
    Medium,
    Hard,
}

/// SciRS2-enhanced solution analyzer using scirs2-stats
pub struct SciRS2SolutionAnalyzer {
    /// Statistical metrics
    pub stats: SolutionStatistics,
}

impl SciRS2SolutionAnalyzer {
    /// Create new solution analyzer
    #[must_use]
    pub fn new() -> Self {
        Self {
            stats: SolutionStatistics::default(),
        }
    }

    /// Analyze solution quality using scirs2-stats
    pub fn analyze_solutions(
        &mut self,
        solutions: &[Vec<i8>],
        energies: &[f64],
    ) -> ApplicationResult<SolutionAnalysisResult> {
        if solutions.is_empty() || energies.is_empty() || solutions.len() != energies.len() {
            return Err(ApplicationError::DataValidationError(
                "Invalid solution data".to_string(),
            ));
        }

        // Convert to Array1 for scirs2-stats
        let energy_array = Array1::from_vec(energies.to_vec());

        // Compute statistical metrics using scirs2-stats
        let min_energy = energies.iter().copied().fold(f64::INFINITY, f64::min);
        let max_energy = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mean_energy = mean(&energy_array.view()).map_err(|e| {
            ApplicationError::OptimizationError(format!("Mean calculation error: {e:?}"))
        })?;

        let std_energy = std(&energy_array.view(), 1, None).map_err(|e| {
            ApplicationError::OptimizationError(format!("Std dev calculation error: {e:?}"))
        })?;

        // Solution diversity analysis
        let diversity = self.compute_solution_diversity(solutions);

        self.stats = SolutionStatistics {
            num_solutions: solutions.len(),
            min_energy,
            max_energy,
            mean_energy,
            std_energy,
            diversity,
        };

        Ok(SolutionAnalysisResult {
            statistics: self.stats.clone(),
            quality_assessment: self.assess_quality(),
            recommendations: self.generate_recommendations(),
        })
    }

    fn compute_solution_diversity(&self, solutions: &[Vec<i8>]) -> f64 {
        if solutions.len() <= 1 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        for i in 0..solutions.len() {
            for j in (i + 1)..solutions.len() {
                let hamming_distance = solutions[i]
                    .iter()
                    .zip(solutions[j].iter())
                    .filter(|(&a, &b)| a != b)
                    .count();
                total_distance += hamming_distance as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_distance / f64::from(count)
        } else {
            0.0
        }
    }

    fn assess_quality(&self) -> QualityAssessment {
        let energy_range = self.stats.max_energy - self.stats.min_energy;
        let relative_std = if self.stats.mean_energy.abs() > 1e-12 {
            self.stats.std_energy.abs() / self.stats.mean_energy.abs()
        } else {
            self.stats.std_energy
        };

        if energy_range < 0.01 && relative_std < 0.1 {
            QualityAssessment::Excellent
        } else if energy_range < 0.1 && relative_std < 0.3 {
            QualityAssessment::Good
        } else if energy_range < 1.0 && relative_std < 0.5 {
            QualityAssessment::Fair
        } else {
            QualityAssessment::Poor
        }
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.stats.diversity < 1.0 {
            recommendations.push(
                "Increase temperature or annealing time to improve solution diversity".to_string(),
            );
        }

        if self.stats.std_energy > 1.0 {
            recommendations.push(
                "Solutions show high energy variance - consider parameter tuning".to_string(),
            );
        }

        if self.stats.num_solutions < 100 {
            recommendations
                .push("Collect more samples for better statistical analysis".to_string());
        }

        if matches!(self.assess_quality(), QualityAssessment::Poor) {
            recommendations.push(
                "Solution quality is poor - consider using different annealing parameters or algorithm".to_string()
            );
        }

        recommendations
    }
}

impl Default for SciRS2SolutionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Solution statistics computed with scirs2-stats
#[derive(Debug, Clone, Default)]
pub struct SolutionStatistics {
    pub num_solutions: usize,
    pub min_energy: f64,
    pub max_energy: f64,
    pub mean_energy: f64,
    pub std_energy: f64,
    pub diversity: f64,
}

/// Result of solution analysis
#[derive(Debug, Clone)]
pub struct SolutionAnalysisResult {
    pub statistics: SolutionStatistics,
    pub quality_assessment: QualityAssessment,
    pub recommendations: Vec<String>,
}

/// Quality assessment for solutions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityAssessment {
    Excellent,
    Good,
    Fair,
    Poor,
}

/// SciRS2-enhanced energy landscape plotter
pub struct SciRS2EnergyPlotter {
    /// Plotting configuration
    pub config: PlottingConfig,
}

impl SciRS2EnergyPlotter {
    /// Create new energy landscape plotter
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: PlottingConfig::default(),
        }
    }

    /// Write a real energy-landscape summary to `output_path`.
    ///
    /// No plotting backend (e.g. scirs2-plot) is wired into this crate, so
    /// this does not render an image; instead it writes the real computed
    /// energy statistics and a per-solution energy table as a plain-text
    /// report to the exact path requested, rather than silently discarding
    /// `output_path` after only printing to stdout. Returns a real I/O error
    /// if the file cannot be written.
    pub fn plot_energy_landscape(
        &self,
        qubo: &SciRS2QuboModel,
        solutions: &[Vec<i8>],
        energies: &[f64],
        output_path: &Path,
    ) -> ApplicationResult<()> {
        let mut report = String::new();
        let _ = writeln!(report, "=== Energy Landscape Summary ===");
        let _ = writeln!(report, "Problem size: {} variables", qubo.num_variables);
        let _ = writeln!(report, "Number of solutions: {}", solutions.len());

        let stats = qubo.get_statistics();
        let _ = writeln!(report, "Matrix density: {:.4}", stats.density);
        let _ = writeln!(
            report,
            "Non-zero quadratic terms: {}",
            stats.num_quadratic_terms
        );

        if !energies.is_empty() {
            let energy_arr = Array1::from_vec(energies.to_vec());
            let min_e = energies.iter().copied().fold(f64::INFINITY, f64::min);
            let max_e = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mean_e = mean(&energy_arr.view()).unwrap_or(0.0);
            let std_e = std(&energy_arr.view(), 1, None).unwrap_or(0.0);

            let _ = writeln!(report, "Energy statistics:");
            let _ = writeln!(report, "  Min:  {min_e:.6}");
            let _ = writeln!(report, "  Max:  {max_e:.6}");
            let _ = writeln!(report, "  Mean: {mean_e:.6}");
            let _ = writeln!(report, "  Std:  {std_e:.6}");

            let _ = writeln!(report, "index,energy");
            for (idx, energy) in energies.iter().enumerate() {
                let _ = writeln!(report, "{idx},{energy:.6}");
            }
        }
        let _ = writeln!(report, "================================");

        println!("{report}");
        std::fs::write(output_path, &report).map_err(|e| {
            ApplicationError::OptimizationError(format!(
                "failed to write energy landscape summary to {output_path:?}: {e}"
            ))
        })
    }

    /// Write a real solution-quality histogram summary to `output_path`.
    ///
    /// As with [`Self::plot_energy_landscape`], no plotting backend is
    /// available: this writes the real computed distribution statistics and
    /// binned counts as a plain-text report to the requested path instead of
    /// silently ignoring it. Returns a real I/O error if the file cannot be
    /// written.
    pub fn plot_solution_histogram(
        &self,
        energies: &[f64],
        output_path: &Path,
    ) -> ApplicationResult<()> {
        let mut report = String::new();
        let _ = writeln!(report, "=== Solution Histogram ===");
        let _ = writeln!(report, "Number of samples: {}", energies.len());

        if !energies.is_empty() {
            let energy_arr = Array1::from_vec(energies.to_vec());
            let mean_e = mean(&energy_arr.view()).unwrap_or(0.0);
            let std_e = std(&energy_arr.view(), 1, None).unwrap_or(0.0);
            let _ = writeln!(report, "Distribution: mean={mean_e:.4}, std={std_e:.4}");

            // Real fixed-bin-count histogram over the real energy samples.
            const NUM_BINS: usize = 10;
            let min_e = energies.iter().copied().fold(f64::INFINITY, f64::min);
            let max_e = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let span = (max_e - min_e).max(1e-12);
            let mut bins = vec![0usize; NUM_BINS];
            for &e in energies {
                let idx = (((e - min_e) / span) * NUM_BINS as f64).floor() as usize;
                bins[idx.min(NUM_BINS - 1)] += 1;
            }
            let _ = writeln!(report, "bin_start,bin_end,count");
            for (i, count) in bins.iter().enumerate() {
                let lo = min_e + span * i as f64 / NUM_BINS as f64;
                let hi = min_e + span * (i + 1) as f64 / NUM_BINS as f64;
                let _ = writeln!(report, "{lo:.6},{hi:.6},{count}");
            }
        }
        let _ = writeln!(report, "==========================");

        println!("{report}");
        std::fs::write(output_path, &report).map_err(|e| {
            ApplicationError::OptimizationError(format!(
                "failed to write solution histogram to {output_path:?}: {e}"
            ))
        })
    }
}

impl Default for SciRS2EnergyPlotter {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for plotting
#[derive(Debug, Clone)]
pub struct PlottingConfig {
    pub resolution: (usize, usize),
    pub color_scheme: String,
    pub show_grid: bool,
}

impl Default for PlottingConfig {
    fn default() -> Self {
        Self {
            resolution: (800, 600),
            color_scheme: "viridis".to_string(),
            show_grid: true,
        }
    }
}

/// Integration tests for SciRS2 functionality
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scirs2_qubo_creation() {
        let qubo = SciRS2QuboModel::new(4).expect("Failed to create QUBO model");
        assert_eq!(qubo.num_variables, 4);
        assert_eq!(qubo.linear_terms.len(), 4);
        assert_eq!(qubo.quadratic_matrix.nnz(), 0);
    }

    #[test]
    fn test_scirs2_qubo_operations() {
        let mut qubo = SciRS2QuboModel::new(3).expect("Failed to create QUBO model");

        qubo.set_linear(0, 1.5).expect("Failed to set linear term");
        assert_eq!(qubo.linear_terms[0], 1.5);

        qubo.set_quadratic(0, 1, 2.0)
            .expect("Failed to set quadratic term");
        // Check that the sparse matrix has the entry
        let mut found = false;
        for (i, j, val) in qubo.iter_nonzeros() {
            if (i == 0 && j == 1) || (i == 1 && j == 0) {
                assert!((val - 2.0).abs() < 1e-10);
                found = true;
            }
        }
        assert!(found, "Quadratic term not found in sparse matrix");

        let solution = vec![1, 0, 1];
        let energy = qubo
            .evaluate(&solution)
            .expect("Failed to evaluate solution");
        assert!((energy - 1.5).abs() < 1e-10); // 1.5 * 1 + 0 = 1.5
    }

    #[test]
    fn test_graph_analysis() {
        let mut qubo = SciRS2QuboModel::new(4).expect("Failed to create QUBO model");
        qubo.set_quadratic(0, 1, 1.0)
            .expect("Failed to set quadratic term");
        qubo.set_quadratic(1, 2, 1.0)
            .expect("Failed to set quadratic term");
        qubo.set_quadratic(2, 3, 1.0)
            .expect("Failed to set quadratic term");

        let mut analyzer = SciRS2GraphAnalyzer::new();
        let result = analyzer
            .analyze_problem_graph(&qubo)
            .expect("Failed to analyze graph");

        assert_eq!(result.metrics.num_nodes, 4);
        assert_eq!(result.metrics.num_edges, 3);
        assert!(result.metrics.clustering_coefficient >= 0.0);
        assert!(result.metrics.avg_degree >= 0.0);
    }

    #[test]
    fn clustering_coefficient_is_real_not_a_fixed_constant() {
        // A path graph (0-1-2-3) has no triangles at all: the real global
        // clustering coefficient must be exactly 0.0, not the old fabricated
        // 0.3 that was returned unconditionally for every input.
        let mut path_qubo = SciRS2QuboModel::new(4).expect("Failed to create QUBO model");
        path_qubo
            .set_quadratic(0, 1, 1.0)
            .expect("Failed to set quadratic term");
        path_qubo
            .set_quadratic(1, 2, 1.0)
            .expect("Failed to set quadratic term");
        path_qubo
            .set_quadratic(2, 3, 1.0)
            .expect("Failed to set quadratic term");

        let mut path_analyzer = SciRS2GraphAnalyzer::new();
        let path_result = path_analyzer
            .analyze_problem_graph(&path_qubo)
            .expect("Failed to analyze graph");
        assert!(
            path_result.metrics.clustering_coefficient.abs() < 1e-12,
            "a triangle-free path graph must have clustering coefficient 0.0, got {}",
            path_result.metrics.clustering_coefficient
        );

        // A triangle (0-1-2-0) is fully transitive: every connected triple is
        // also a closed triangle, so the real coefficient must be exactly 1.0.
        let mut triangle_qubo = SciRS2QuboModel::new(3).expect("Failed to create QUBO model");
        triangle_qubo
            .set_quadratic(0, 1, 1.0)
            .expect("Failed to set quadratic term");
        triangle_qubo
            .set_quadratic(1, 2, 1.0)
            .expect("Failed to set quadratic term");
        triangle_qubo
            .set_quadratic(0, 2, 1.0)
            .expect("Failed to set quadratic term");

        let mut triangle_analyzer = SciRS2GraphAnalyzer::new();
        let triangle_result = triangle_analyzer
            .analyze_problem_graph(&triangle_qubo)
            .expect("Failed to analyze graph");
        assert!(
            (triangle_result.metrics.clustering_coefficient - 1.0).abs() < 1e-12,
            "a triangle graph must have clustering coefficient 1.0, got {}",
            triangle_result.metrics.clustering_coefficient
        );
    }

    #[test]
    fn test_solution_analysis() {
        let solutions = vec![vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0]];
        let energies = vec![-1.0, -0.5, -0.8];

        let mut analyzer = SciRS2SolutionAnalyzer::new();
        let result = analyzer
            .analyze_solutions(&solutions, &energies)
            .expect("Failed to analyze solutions");

        assert_eq!(result.statistics.num_solutions, 3);
        assert_eq!(result.statistics.min_energy, -1.0);
        assert_eq!(result.statistics.max_energy, -0.5);
        // scirs2-stats provides accurate mean
        assert!((result.statistics.mean_energy - (-0.7_666_666_666_666_667)).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_matrix_efficiency() {
        // Test that sparse representation is actually efficient
        let mut qubo = SciRS2QuboModel::new(1000).expect("Failed to create QUBO model");

        // Add only a few non-zero terms
        qubo.set_quadratic(0, 1, 1.0)
            .expect("Failed to set quadratic term");
        qubo.set_quadratic(5, 10, 2.0)
            .expect("Failed to set quadratic term");
        qubo.set_quadratic(100, 200, 3.0)
            .expect("Failed to set quadratic term");

        let stats = qubo.get_statistics();

        // Should have only 3 quadratic terms (stored symmetrically)
        assert_eq!(stats.num_quadratic_terms, 3);

        // Memory usage should be much less than dense matrix
        let dense_memory = 1000 * 1000 * std::mem::size_of::<f64>();
        assert!(stats.memory_usage < dense_memory / 100); // At least 100x smaller
    }

    #[test]
    fn plot_energy_landscape_actually_writes_the_requested_file() {
        let qubo = SciRS2QuboModel::new(2).expect("Failed to create QUBO model");
        let plotter = SciRS2EnergyPlotter::new();
        let solutions = vec![vec![1, -1], vec![-1, 1]];
        let energies = vec![-2.0, -1.0];

        let mut path = std::env::temp_dir();
        path.push(format!(
            "quantrs2_anneal_energy_landscape_test_{}.txt",
            std::process::id()
        ));

        plotter
            .plot_energy_landscape(&qubo, &solutions, &energies, &path)
            .expect("plotting should succeed");

        let written = std::fs::read_to_string(&path).expect("output file should exist");
        assert!(written.contains("Energy statistics"));
        assert!(written.contains("Min:"));
        assert!(written.contains("-2.0"));

        std::fs::remove_file(&path).expect("cleanup should succeed");
    }

    #[test]
    fn plot_solution_histogram_actually_writes_the_requested_file() {
        let plotter = SciRS2EnergyPlotter::new();
        let energies = vec![-3.0, -2.0, -1.0, 0.0, 1.0];

        let mut path = std::env::temp_dir();
        path.push(format!(
            "quantrs2_anneal_solution_histogram_test_{}.txt",
            std::process::id()
        ));

        plotter
            .plot_solution_histogram(&energies, &path)
            .expect("plotting should succeed");

        let written = std::fs::read_to_string(&path).expect("output file should exist");
        assert!(written.contains("Distribution: mean="));
        assert!(written.contains("bin_start,bin_end,count"));

        std::fs::remove_file(&path).expect("cleanup should succeed");
    }
}
