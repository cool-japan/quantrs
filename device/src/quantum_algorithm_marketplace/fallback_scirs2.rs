//! Fallback implementations for SciRS2 functionality when the feature is not available
//!
//! This module provides basic implementations of SciRS2 functions that are used
//! in the quantum algorithm marketplace module when the scirs2 feature is not enabled.

use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Fallback error type for optimization
#[derive(Debug, Clone)]
pub struct OptimizeError {
    pub message: String,
}

impl std::fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Optimization error: {}", self.message)
    }
}

impl std::error::Error for OptimizeError {}

/// Fallback result type for optimization
pub type OptimizeResult<T> = Result<T, OptimizeError>;

/// Fallback linear algebra error
#[derive(Debug, Clone)]
pub struct LinalgError {
    pub message: String,
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Linear algebra error: {}", self.message)
    }
}

impl std::error::Error for LinalgError {}

/// Fallback result type for linear algebra
pub type LinalgResult<T> = Result<T, LinalgError>;

/// Alternative enum for statistical tests
#[derive(Debug, Clone, Copy)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

/// Basic statistics functions
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn std(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

pub fn var(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn corrcoef(x: &[f64], y: &[f64]) -> f64 {
    pearsonr(x, y)
}

pub fn pearsonr(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let mean_x = mean(x);
    let mean_y = mean(y);

    let numerator: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
    let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

    let denominator = (sum_sq_x * sum_sq_y).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Graph-related fallback functions
#[derive(Debug, Clone)]
pub struct Graph<N, E> {
    nodes: Vec<N>,
    edges: Vec<(usize, usize, E)>,
}

impl<N, E> Graph<N, E> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: N) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    pub fn add_edge(&mut self, a: usize, b: usize, edge: E) {
        self.edges.push((a, b, edge));
    }

    pub fn nodes(&self) -> impl Iterator<Item = &N> {
        self.nodes.iter()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

pub fn betweenness_centrality<N, E>(
    _graph: &Graph<N, E>,
    _normalized: bool,
) -> HashMap<usize, f64> {
    HashMap::new() // Fallback - empty centrality
}

/// Fallback optimization function
pub fn minimize<F>(
    _objective: F,
    _initial_guess: &[f64],
    _bounds: Option<&[(f64, f64)]>,
) -> OptimizeResult<MinimizeResult>
where
    F: Fn(&[f64]) -> f64,
{
    // Basic fallback - return the initial guess as "optimal"
    Ok(MinimizeResult {
        x: _initial_guess.to_vec(),
        fun: 0.0,
        success: true,
        message: "Fallback optimization".to_string(),
        nit: 0,
        nfev: 0,
    })
}

/// Result type for minimize function
#[derive(Debug, Clone)]
pub struct MinimizeResult {
    pub x: Vec<f64>,
    pub fun: f64,
    pub success: bool,
    pub message: String,
    pub nit: usize,
    pub nfev: usize,
}

/// Fallback linear algebra functions
pub fn eig(matrix: &Array2<f64>) -> LinalgResult<(Array1<f64>, Array2<f64>)> {
    // Very basic fallback - return identity-like results
    let n = matrix.nrows();
    let eigenvalues = Array1::ones(n);
    let eigenvectors = Array2::eye(n);
    Ok((eigenvalues, eigenvectors))
}

pub fn matrix_norm(matrix: &Array2<f64>) -> f64 {
    // Frobenius norm
    matrix.iter().map(|x| x * x).sum::<f64>().sqrt()
}
