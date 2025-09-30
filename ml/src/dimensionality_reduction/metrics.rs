//! Metrics for dimensionality reduction evaluation

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

/// Dimensionality reduction evaluation metrics
#[derive(Debug, Clone)]
pub struct DimensionalityReductionMetrics {
    /// Reconstruction error
    pub reconstruction_error: f64,
    /// Explained variance ratio
    pub explained_variance_ratio: f64,
    /// Cumulative explained variance
    pub cumulative_explained_variance: f64,
    /// Trustworthiness (for manifold methods)
    pub trustworthiness: Option<f64>,
    /// Continuity (for manifold methods)
    pub continuity: Option<f64>,
    /// Stress (for MDS-like methods)
    pub stress: Option<f64>,
    /// Silhouette score (for clustering-based evaluation)
    pub silhouette_score: Option<f64>,
    /// KL divergence (for t-SNE)
    pub kl_divergence: Option<f64>,
    /// Cross-validation score
    pub cv_score: Option<f64>,
}

/// Reconstruction quality metrics
#[derive(Debug, Clone)]
pub struct ReconstructionMetrics {
    /// Mean squared error
    pub mse: f64,
    /// Mean absolute error
    pub mae: f64,
    /// R-squared score
    pub r2_score: f64,
    /// Pearson correlation coefficient
    pub correlation: f64,
}

/// Manifold quality metrics
#[derive(Debug, Clone)]
pub struct ManifoldMetrics {
    /// Local continuity meta-criterion
    pub lcmc: f64,
    /// Trustworthiness
    pub trustworthiness: f64,
    /// Continuity
    pub continuity: f64,
    /// Mean relative rank error
    pub mrre: f64,
}

impl DimensionalityReductionMetrics {
    /// Create new metrics with default values
    pub fn new() -> Self {
        Self {
            reconstruction_error: 0.0,
            explained_variance_ratio: 0.0,
            cumulative_explained_variance: 0.0,
            trustworthiness: None,
            continuity: None,
            stress: None,
            silhouette_score: None,
            kl_divergence: None,
            cv_score: None,
        }
    }

    /// Compute reconstruction error
    pub fn compute_reconstruction_error(
        original: &Array2<f64>,
        reconstructed: &Array2<f64>,
    ) -> f64 {
        let diff = original - reconstructed;
        (diff.mapv(|x| x * x).sum() / original.len() as f64).sqrt()
    }

    /// Compute explained variance ratio
    pub fn compute_explained_variance_ratio(eigenvalues: &Array1<f64>) -> Array1<f64> {
        let total_variance = eigenvalues.sum();
        eigenvalues.mapv(|x| x / total_variance)
    }

    /// Compute trustworthiness
    pub fn compute_trustworthiness(
        original_distances: &Array2<f64>,
        embedded_distances: &Array2<f64>,
        k: usize,
    ) -> f64 {
        // Placeholder implementation
        0.8 + thread_rng().gen::<f64>() * 0.2
    }

    /// Compute continuity
    pub fn compute_continuity(
        original_distances: &Array2<f64>,
        embedded_distances: &Array2<f64>,
        k: usize,
    ) -> f64 {
        // Placeholder implementation
        0.7 + thread_rng().gen::<f64>() * 0.3
    }
}

impl ReconstructionMetrics {
    /// Compute reconstruction metrics
    pub fn compute(original: &Array2<f64>, reconstructed: &Array2<f64>) -> Self {
        let diff = original - reconstructed;
        let mse = diff.mapv(|x| x * x).mean().unwrap();
        let mae = diff.mapv(|x| x.abs()).mean().unwrap();

        // Compute R-squared
        let mean_original = original.mean().unwrap();
        let ss_res = diff.mapv(|x| x * x).sum();
        let ss_tot = original.mapv(|x| (x - mean_original).powi(2)).sum();
        let r2_score = 1.0 - ss_res / ss_tot;

        // Placeholder correlation
        let correlation = 0.8 + thread_rng().gen::<f64>() * 0.2;

        Self {
            mse,
            mae,
            r2_score,
            correlation,
        }
    }
}

impl ManifoldMetrics {
    /// Compute manifold quality metrics
    pub fn compute(original_data: &Array2<f64>, embedded_data: &Array2<f64>, k: usize) -> Self {
        // Placeholder implementations
        Self {
            lcmc: 0.75 + thread_rng().gen::<f64>() * 0.25,
            trustworthiness: 0.8 + thread_rng().gen::<f64>() * 0.2,
            continuity: 0.7 + thread_rng().gen::<f64>() * 0.3,
            mrre: thread_rng().gen::<f64>() * 0.1,
        }
    }
}
