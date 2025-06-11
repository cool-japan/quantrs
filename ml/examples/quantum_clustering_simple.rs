//! Simple Quantum Clustering Example
//!
//! This example demonstrates basic quantum clustering functionality
//! with the working APIs in the quantum ML module.

use ndarray::{array, Array2};
use quantrs2_ml::clustering::*;
use quantrs2_ml::dimensionality_reduction::{QuantumDistanceMetric, QuantumEnhancementLevel};
use quantrs2_ml::prelude::*;

fn main() -> Result<()> {
    println!("🌀 Simple Quantum Clustering Demo");
    println!("=================================\n");

    // Create simple sample data
    let data = array![
        [1.0, 1.0],
        [1.1, 1.1],
        [0.9, 0.9],
        [1.2, 0.8],
        [5.0, 5.0],
        [5.1, 5.1],
        [4.9, 4.9],
        [5.2, 4.8],
    ];

    println!("Sample data:");
    for (i, row) in data.rows().into_iter().enumerate() {
        println!("  Point {}: [{:.1}, {:.1}]", i, row[0], row[1]);
    }

    // Demo 1: Quantum K-means Clustering
    demo_quantum_kmeans(&data)?;

    // Demo 2: Quantum DBSCAN
    demo_quantum_dbscan(&data)?;

    // Demo 3: Quantum Spectral Clustering
    demo_quantum_spectral(&data)?;

    println!("\n✅ Simple quantum clustering demos completed successfully!");

    Ok(())
}

/// Demo quantum K-means clustering
fn demo_quantum_kmeans(data: &Array2<f64>) -> Result<()> {
    println!("\n🎯 Demo 1: Quantum K-means Clustering");
    println!("-------------------------------------");

    // Create K-means configuration
    let config = QuantumKMeansConfig {
        n_clusters: 2,
        max_iterations: 100,
        tolerance: 1e-4,
        distance_metric: QuantumDistanceMetric::QuantumEuclidean,
        quantum_reps: 2,
        enhancement_level: QuantumEnhancementLevel::Moderate,
        seed: Some(42),
    };

    // Create and train clusterer
    let mut clusterer = QuantumClusterer::kmeans(config);
    let result = clusterer.fit(data)?;

    println!("   Clusters found: {}", result.n_clusters);
    println!("   Labels: {:?}", result.labels);
    println!("   Inertia: {:.4}", result.inertia.unwrap_or(0.0));

    // Show cluster centers if available
    if let Some(centers) = &result.cluster_centers {
        println!("   Cluster centers:");
        for (i, center) in centers.rows().into_iter().enumerate() {
            println!("     Cluster {}: [{:.3}, {:.3}]", i, center[0], center[1]);
        }
    }

    // Test prediction on new data
    let new_data = array![[1.5, 1.5], [4.5, 4.5]];
    let predictions = clusterer.predict(&new_data)?;
    println!("   Predictions for new data: {:?}", predictions);

    Ok(())
}

/// Demo quantum DBSCAN clustering
fn demo_quantum_dbscan(data: &Array2<f64>) -> Result<()> {
    println!("\n🎯 Demo 2: Quantum DBSCAN Clustering");
    println!("------------------------------------");

    // Create DBSCAN configuration
    let config = QuantumDBSCANConfig {
        eps: 1.0,
        min_samples: 3,
        distance_metric: QuantumDistanceMetric::QuantumEuclidean,
        enhancement_level: QuantumEnhancementLevel::Moderate,
        seed: None,
    };

    // Create and train clusterer
    let mut clusterer = QuantumClusterer::dbscan(config);
    let result = clusterer.fit(data)?;

    println!("   Clusters found: {}", result.n_clusters);
    println!("   Labels: {:?}", result.labels);

    // Count noise points (using MAX as noise label)
    let noise_count = result.labels.iter().filter(|&&x| x == usize::MAX).count();
    println!("   Noise points: {}", noise_count);

    Ok(())
}

/// Demo quantum spectral clustering
fn demo_quantum_spectral(data: &Array2<f64>) -> Result<()> {
    println!("\n🎯 Demo 3: Quantum Spectral Clustering");
    println!("--------------------------------------");

    // Create spectral configuration
    let config = QuantumSpectralConfig {
        n_clusters: 2,
        affinity: AffinityType::RBF,
        gamma: 1.0,
        enhancement_level: QuantumEnhancementLevel::Light,
        seed: None,
    };

    // Create and train clusterer
    let mut clusterer = QuantumClusterer::spectral(config);
    let result = clusterer.fit(data)?;

    println!("   Clusters found: {}", result.n_clusters);
    println!("   Labels: {:?}", result.labels);

    Ok(())
}
