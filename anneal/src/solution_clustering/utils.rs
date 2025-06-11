//! Utility functions for solution clustering

use super::error::{ClusteringError, ClusteringResult};
use super::types::*;
use crate::simulator::AnnealingSolution;

/// Analyze solution diversity
pub fn analyze_solution_diversity(solutions: &[AnnealingSolution]) -> ClusteringResult<f64> {
    if solutions.len() < 2 {
        return Ok(0.0);
    }

    let mut total_distance = 0.0;
    let mut count = 0;

    for i in 0..solutions.len() {
        for j in (i + 1)..solutions.len() {
            let distance = hamming_distance(&solutions[i].best_spins, &solutions[j].best_spins);
            total_distance += distance as f64;
            count += 1;
        }
    }

    Ok(total_distance / count as f64)
}

/// Calculate Hamming distance between two spin configurations
fn hamming_distance(spins1: &[i8], spins2: &[i8]) -> usize {
    spins1
        .iter()
        .zip(spins2.iter())
        .filter(|(a, b)| a != b)
        .count()
}

/// Find the most representative solution in a cluster
pub fn find_representative_solution(cluster: &SolutionCluster) -> Option<&SolutionPoint> {
    if cluster.solutions.is_empty() {
        return None;
    }

    let mut min_distance = f64::INFINITY;
    let mut representative_idx = 0;

    for (i, solution) in cluster.solutions.iter().enumerate() {
        if let Some(features) = solution.features.as_ref() {
            let mut total_distance = 0.0;
            let mut count = 0;

            for other_solution in &cluster.solutions {
                if let Some(other_features) = other_solution.features.as_ref() {
                    if let Ok(distance) = euclidean_distance(features, other_features) {
                        total_distance += distance;
                        count += 1;
                    }
                }
            }

            let avg_distance = if count > 0 {
                total_distance / count as f64
            } else {
                f64::INFINITY
            };

            if avg_distance < min_distance {
                min_distance = avg_distance;
                representative_idx = i;
            }
        }
    }

    cluster.solutions.get(representative_idx)
}

/// Calculate Euclidean distance
fn euclidean_distance(vec1: &[f64], vec2: &[f64]) -> ClusteringResult<f64> {
    if vec1.len() != vec2.len() {
        return Err(ClusteringError::DimensionMismatch {
            expected: vec1.len(),
            actual: vec2.len(),
        });
    }

    let sum_sq: f64 = vec1
        .iter()
        .zip(vec2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    Ok(sum_sq.sqrt())
}
