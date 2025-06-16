//! Analysis utilities for quantum annealing results.
//!
//! This module provides tools for analyzing and interpreting
//! results from quantum annealing, including clustering and visualization.

#[cfg(feature = "clustering")]
use ndarray::Array2;
use std::collections::HashMap;
use thiserror::Error;

use crate::sampler::SampleResult;

// Re-export visualization module
pub mod visualization;
pub use visualization::*;

// Graph utilities
pub mod graph;

/// Errors that can occur during analysis
#[derive(Error, Debug)]
pub enum AnalysisError {
    /// Error in clustering algorithm
    #[error("Clustering error: {0}")]
    ClusteringError(String),

    /// Error in visualization
    #[error("Visualization error: {0}")]
    VisualizationError(String),

    /// Error in data processing
    #[error("Data processing error: {0}")]
    DataProcessingError(String),
}

/// Result type for analysis operations
pub type AnalysisResult<T> = Result<T, AnalysisError>;

/// Cluster similar solutions to identify patterns
#[cfg(feature = "clustering")]
pub fn cluster_solutions(
    results: &[SampleResult],
    max_clusters: usize,
) -> AnalysisResult<Vec<(Vec<usize>, f64)>> {
    use scirs2_cluster::vq::kmeans::{KMeans, KMeansParams};

    if results.is_empty() {
        return Err(AnalysisError::DataProcessingError(
            "Empty results list".to_string(),
        ));
    }

    // Extract all variable names
    let variable_names: Vec<String> = results[0].assignments.keys().cloned().collect();

    // Convert solutions to binary vectors
    let n_vars = variable_names.len();
    let n_samples = results.len();

    let data = Array2::<f64>::zeros((n_samples, n_vars));

    for (i, result) in results.iter().enumerate() {
        for (j, var_name) in variable_names.iter().enumerate() {
            if let Some(&value) = result.assignments.get(var_name) {
                data[[i, j]] = if value { 1.0 } else { 0.0 };
            }
        }
    }

    // Determine optimal number of clusters
    let actual_max_clusters = std::cmp::min(max_clusters, n_samples / 2);
    let actual_max_clusters = std::cmp::max(actual_max_clusters, 2); // At least 2 clusters

    // Setup K-means parameters
    let mut params = KMeansParams::default();
    params.n_clusters = actual_max_clusters;
    params.max_iter = 100;
    params.tol = 1e-4;

    // Run K-means clustering
    let kmeans = KMeans::new(params);
    let mut result = kmeans
        .fit(&data)
        .map_err(|e| AnalysisError::ClusteringError(e.to_string()))?;

    // Get cluster labels
    let labels = result.labels();

    // Group results by cluster
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut cluster_energies: HashMap<usize, Vec<f64>> = HashMap::new();

    for (i, &label) in labels.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
        cluster_energies
            .entry(label)
            .or_default()
            .push(results[i].energy);
    }

    // Calculate average energy for each cluster
    let mut cluster_results = Vec::new();
    for (label, indices) in clusters {
        let avg_energy: f64 = cluster_energies[&label].iter().sum::<f64>() / indices.len() as f64;
        cluster_results.push((indices, avg_energy));
    }

    // Sort clusters by average energy
    cluster_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    Ok(cluster_results)
}

/// Fallback clustering implementation
#[cfg(not(feature = "clustering"))]
pub fn cluster_solutions(
    results: &[SampleResult],
    _max_clusters: usize,
) -> AnalysisResult<Vec<(Vec<usize>, f64)>> {
    // Simple implementation: just group identical solutions
    if results.is_empty() {
        return Err(AnalysisError::DataProcessingError(
            "Empty results list".to_string(),
        ));
    }

    // Group solutions by their binary representation
    let mut groups: HashMap<Vec<bool>, Vec<usize>> = HashMap::new();
    let mut group_energies: HashMap<Vec<bool>, Vec<f64>> = HashMap::new();

    // Extract all variable names in sorted order
    let mut variable_names: Vec<String> = results[0].assignments.keys().cloned().collect();
    variable_names.sort();

    for (i, result) in results.iter().enumerate() {
        // Convert to sorted binary vector for consistent comparison
        let binary: Vec<bool> = variable_names
            .iter()
            .map(|name| *result.assignments.get(name).unwrap_or(&false))
            .collect();

        groups.entry(binary.clone()).or_default().push(i);
        group_energies
            .entry(binary)
            .or_default()
            .push(result.energy);
    }

    // Calculate average energy for each group
    let mut group_results = Vec::new();
    for (binary, indices) in groups {
        let avg_energy: f64 = group_energies[&binary].iter().sum::<f64>() / indices.len() as f64;
        group_results.push((indices, avg_energy));
    }

    // Sort groups by average energy
    group_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    Ok(group_results)
}

/// Calculate diversity metrics for a set of solutions
pub fn calculate_diversity(results: &[SampleResult]) -> AnalysisResult<HashMap<String, f64>> {
    if results.is_empty() {
        return Err(AnalysisError::DataProcessingError(
            "Empty results list".to_string(),
        ));
    }

    // Extract all variable names
    let variable_names: Vec<String> = results[0].assignments.keys().cloned().collect();

    let n_vars = variable_names.len();

    // Calculate diversity metrics
    let mut metrics = HashMap::new();

    // 1. Hamming distance statistics
    let mut distances = Vec::new();

    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let mut distance = 0;

            for var_name in &variable_names {
                let val_i = results[i].assignments.get(var_name).unwrap_or(&false);
                let val_j = results[j].assignments.get(var_name).unwrap_or(&false);

                if val_i != val_j {
                    distance += 1;
                }
            }

            distances.push(distance as f64 / n_vars as f64);
        }
    }

    if !distances.is_empty() {
        // Calculate statistics
        let avg_distance: f64 = distances.iter().sum::<f64>() / distances.len() as f64;

        // Sort for percentiles
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_distance = distances.first().cloned().unwrap_or(0.0);
        let max_distance = distances.last().cloned().unwrap_or(0.0);

        let median_idx = distances.len() / 2;
        let median_distance = if distances.len() % 2 == 0 {
            (distances[median_idx - 1] + distances[median_idx]) / 2.0
        } else {
            distances[median_idx]
        };

        metrics.insert("avg_distance".to_string(), avg_distance);
        metrics.insert("min_distance".to_string(), min_distance);
        metrics.insert("max_distance".to_string(), max_distance);
        metrics.insert("median_distance".to_string(), median_distance);
    }

    // 2. Energy spread
    let energies: Vec<f64> = results.iter().map(|r| r.energy).collect();

    if !energies.is_empty() {
        let min_energy = *energies
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);
        let max_energy = *energies
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);
        let energy_range = max_energy - min_energy;

        metrics.insert("energy_range".to_string(), energy_range);
        metrics.insert("min_energy".to_string(), min_energy);
        metrics.insert("max_energy".to_string(), max_energy);
    }

    // 3. Variable bias - how often each variable is 1
    for var_name in &variable_names {
        let var_count = results
            .iter()
            .filter(|r| *r.assignments.get(var_name).unwrap_or(&false))
            .count() as f64
            / results.len() as f64;

        metrics.insert(format!("var_bias_{}", var_name), var_count);
    }

    Ok(metrics)
}

/// Generate visualizations for solution distributions
#[cfg(feature = "plotters")]
pub fn visualize_energy_distribution(
    results: &[SampleResult],
    file_path: &str,
) -> AnalysisResult<()> {
    use plotters::prelude::*;

    if results.is_empty() {
        return Err(AnalysisError::DataProcessingError(
            "Empty results list".to_string(),
        ));
    }

    // Extract energies
    let energies: Vec<f64> = results.iter().map(|r| r.energy).collect();

    // Create energy histogram
    let root = BitMapBackend::new(file_path, (800, 600)).into_drawing_area();

    root.fill(&WHITE)
        .map_err(|e| AnalysisError::VisualizationError(e.to_string()))?;

    let min_energy = *energies
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max_energy = *energies
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    // Add some padding
    let energy_range = max_energy - min_energy;
    let padding = energy_range * 0.1;
    let y_min = min_energy - padding;
    let y_max = max_energy + padding;

    let mut chart = ChartBuilder::on(&root)
        .caption("Energy Distribution", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..results.len(), y_min..y_max)
        .map_err(|e| AnalysisError::VisualizationError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc("Solution Index")
        .y_desc("Energy")
        .draw()
        .map_err(|e| AnalysisError::VisualizationError(e.to_string()))?;

    // Sort energies for this plot
    let mut sorted_energies = energies.clone();
    sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    chart
        .draw_series(LineSeries::new(
            sorted_energies.iter().enumerate().map(|(i, &e)| (i, e)),
            &RED,
        ))
        .map_err(|e| AnalysisError::VisualizationError(e.to_string()))?
        .label("Energy")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .map_err(|e| AnalysisError::VisualizationError(e.to_string()))?;

    root.present()
        .map_err(|e| AnalysisError::VisualizationError(e.to_string()))?;

    Ok(())
}

/// Fallback visualization (empty implementation)
#[cfg(not(feature = "plotters"))]
pub fn visualize_energy_distribution(
    _results: &[SampleResult],
    _file_path: &str,
) -> AnalysisResult<()> {
    Err(AnalysisError::VisualizationError(
        "Visualization requires the 'plotters' feature".to_string(),
    ))
}
