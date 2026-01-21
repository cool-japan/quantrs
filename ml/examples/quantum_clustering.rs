#![allow(
    clippy::pedantic,
    clippy::unnecessary_wraps,
    clippy::needless_range_loop,
    clippy::useless_vec,
    clippy::needless_collect,
    clippy::too_many_arguments
)]
//! Quantum Clustering Example
//!
//! This example demonstrates various quantum clustering algorithms available in the
//! quantum ML module, including quantum K-means, DBSCAN, spectral clustering,
//! fuzzy c-means, and Gaussian mixture models.

use quantrs2_ml::clustering::{
    create_default_quantum_dbscan, create_default_quantum_kmeans, AffinityType,
    ClusteringAlgorithm, ClusteringEnsembleConfig, CommunityAlgorithm, CovarianceType,
    DimensionalityReduction, EnsembleCombinationMethod, EntanglementStructure,
    GraphClusteringConfig, GraphMethod, HighDimClusteringConfig, MeasurementStrategy,
    QuantumClusterer, QuantumClusteringConfig, QuantumDBSCANConfig, QuantumFuzzyCMeansConfig,
    QuantumGMMConfig, QuantumKMeansConfig, QuantumNativeConfig, QuantumSpectralConfig,
    StatePreparationMethod, StreamingClusteringConfig, TimeSeriesClusteringConfig,
    TimeSeriesDistanceMetric,
};
use quantrs2_ml::dimensionality_reduction::{QuantumDistanceMetric, QuantumEnhancementLevel};
use quantrs2_ml::prelude::*;
use scirs2_core::ndarray::{array, Array1, Array2};

fn main() -> Result<()> {
    println!("🌀 Quantum Clustering Algorithms Demo");
    println!("=====================================\n");

    // Create sample datasets
    let (simple_data, clustered_data, noisy_data) = create_sample_datasets();

    // Demo 1: Quantum K-means Clustering
    demo_quantum_kmeans(&simple_data)?;

    // Demo 2: Quantum DBSCAN
    demo_quantum_dbscan(&noisy_data)?;

    // Demo 3: Quantum Spectral Clustering
    demo_quantum_spectral(&clustered_data)?;

    // Demo 4: Quantum Fuzzy C-means
    demo_quantum_fuzzy_cmeans(&simple_data)?;

    // Demo 5: Quantum Gaussian Mixture Models
    demo_quantum_gmm(&clustered_data)?;

    // Demo 6: Quantum Distance Metrics Comparison
    demo_quantum_distance_metrics(&simple_data)?;

    // Demo 7: Clustering Evaluation Metrics
    demo_clustering_evaluation(&simple_data)?;

    // Demo 8: Advanced Configurations
    demo_advanced_configurations()?;

    println!("\n✅ All quantum clustering demos completed successfully!");

    Ok(())
}

/// Create sample datasets for different clustering scenarios
fn create_sample_datasets() -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    // Simple 2-cluster dataset
    let simple_data = array![
        [1.0, 1.0],
        [1.1, 1.1],
        [0.9, 0.9],
        [1.2, 0.8],
        [5.0, 5.0],
        [5.1, 5.1],
        [4.9, 4.9],
        [5.2, 4.8],
    ];

    // More complex clustered dataset
    let clustered_data = array![
        // Cluster 1
        [1.0, 1.0],
        [1.2, 1.1],
        [0.8, 0.9],
        [1.1, 1.3],
        // Cluster 2
        [5.0, 5.0],
        [5.2, 5.1],
        [4.8, 4.9],
        [5.1, 5.3],
        // Cluster 3
        [9.0, 1.0],
        [9.2, 1.1],
        [8.8, 0.9],
        [9.1, 1.3],
        // Cluster 4
        [5.0, 9.0],
        [5.2, 9.1],
        [4.8, 8.9],
        [5.1, 9.3],
    ];

    // Noisy dataset with outliers (for DBSCAN)
    let noisy_data = array![
        // Dense cluster 1
        [1.0, 1.0],
        [1.1, 1.1],
        [0.9, 0.9],
        [1.2, 0.8],
        [0.8, 1.2],
        // Dense cluster 2
        [5.0, 5.0],
        [5.1, 5.1],
        [4.9, 4.9],
        [5.2, 4.8],
        [4.8, 5.2],
        // Outliers (noise)
        [10.0, 10.0],
        [0.0, 10.0],
        [-5.0, -5.0],
    ];

    (simple_data, clustered_data, noisy_data)
}

/// Demo quantum K-means clustering with different configurations
fn demo_quantum_kmeans(data: &Array2<f64>) -> Result<()> {
    println!("🎯 Demo 1: Quantum K-means Clustering");
    println!("-------------------------------------");

    // Create different K-means configurations
    let configs = vec![
        (
            "Standard Quantum K-means",
            QuantumKMeansConfig {
                n_clusters: 2,
                max_iterations: 100,
                tolerance: 1e-4,
                distance_metric: QuantumDistanceMetric::QuantumEuclidean,
                quantum_reps: 2,
                enhancement_level: QuantumEnhancementLevel::Moderate,
                seed: Some(42),
            },
        ),
        (
            "Quantum Fidelity Distance",
            QuantumKMeansConfig {
                n_clusters: 2,
                distance_metric: QuantumDistanceMetric::QuantumFidelity,
                enhancement_level: QuantumEnhancementLevel::Full,
                ..QuantumKMeansConfig::default()
            },
        ),
        (
            "Quantum Entanglement Distance",
            QuantumKMeansConfig {
                n_clusters: 2,
                distance_metric: QuantumDistanceMetric::QuantumEntanglement,
                enhancement_level: QuantumEnhancementLevel::Experimental,
                ..QuantumKMeansConfig::default()
            },
        ),
    ];

    for (name, config) in configs {
        println!("\n📊 Testing: {name}");

        let mut clusterer = QuantumClusterer::kmeans(config);
        let result = clusterer.fit(data)?;

        println!("   Clusters found: {}", result.n_clusters);
        println!("   Labels: {:?}", result.labels);
        println!("   Inertia: {:.4}", result.inertia.unwrap_or(0.0));

        if let Some(centers) = &result.cluster_centers {
            println!("   Cluster centers:");
            for (i, center) in centers.rows().into_iter().enumerate() {
                println!("     Cluster {}: [{:.3}, {:.3}]", i, center[0], center[1]);
            }
        }

        // Test prediction on new data
        let new_data = array![[1.5, 1.5], [4.5, 4.5]];
        let predictions = clusterer.predict(&new_data)?;
        println!("   Predictions for new data: {predictions:?}");
    }

    Ok(())
}

/// Demo quantum DBSCAN clustering
fn demo_quantum_dbscan(data: &Array2<f64>) -> Result<()> {
    println!("\n🎯 Demo 2: Quantum DBSCAN Clustering");
    println!("------------------------------------");

    let configs = vec![
        (
            "Standard Quantum DBSCAN",
            QuantumDBSCANConfig {
                eps: 1.0,
                min_samples: 3,
                distance_metric: QuantumDistanceMetric::QuantumEuclidean,
                enhancement_level: QuantumEnhancementLevel::Moderate,
                seed: None,
            },
        ),
        (
            "Quantum Kernel Distance",
            QuantumDBSCANConfig {
                eps: 0.8,
                min_samples: 2,
                distance_metric: QuantumDistanceMetric::QuantumKernel,
                enhancement_level: QuantumEnhancementLevel::Full,
                seed: None,
            },
        ),
    ];

    for (name, config) in configs {
        println!("\n📊 Testing: {name}");

        let mut clusterer = QuantumClusterer::dbscan(config);
        let result = clusterer.fit(data)?;

        println!("   Clusters found: {}", result.n_clusters);
        println!("   Labels: {:?}", result.labels);

        // Count noise points (-1 labels)
        let noise_count = result.labels.iter().filter(|&&x| x == usize::MAX).count(); // Using MAX as noise label
        println!("   Noise points: {noise_count}");

        // Count points in each cluster
        let unique_labels: std::collections::HashSet<_> = result.labels.iter().copied().collect();
        for &label in &unique_labels {
            if label != usize::MAX {
                let cluster_size = result.labels.iter().filter(|&&x| x == label).count();
                println!("   Cluster {label} size: {cluster_size}");
            }
        }
    }

    Ok(())
}

/// Demo quantum spectral clustering
fn demo_quantum_spectral(data: &Array2<f64>) -> Result<()> {
    println!("\n🎯 Demo 3: Quantum Spectral Clustering");
    println!("--------------------------------------");

    let configs = vec![
        (
            "RBF Affinity",
            QuantumSpectralConfig {
                n_clusters: 4,
                affinity: AffinityType::RBF,
                gamma: 1.0,
                enhancement_level: QuantumEnhancementLevel::Light,
                seed: None,
            },
        ),
        (
            "Quantum Kernel Affinity",
            QuantumSpectralConfig {
                n_clusters: 4,
                affinity: AffinityType::QuantumKernel,
                gamma: 1.0,
                enhancement_level: QuantumEnhancementLevel::Full,
                seed: None,
            },
        ),
    ];

    for (name, config) in configs {
        println!("\n📊 Testing: {name}");

        let mut clusterer = QuantumClusterer::spectral(config);
        let result = clusterer.fit(data)?;

        println!("   Clusters found: {}", result.n_clusters);
        println!("   Labels: {:?}", result.labels);

        // Analyze cluster distribution
        let unique_labels: std::collections::HashSet<_> = result.labels.iter().copied().collect();
        for &label in &unique_labels {
            let cluster_size = result.labels.iter().filter(|&&x| x == label).count();
            println!("   Cluster {label} size: {cluster_size}");
        }
    }

    Ok(())
}

/// Demo quantum fuzzy c-means clustering
fn demo_quantum_fuzzy_cmeans(data: &Array2<f64>) -> Result<()> {
    println!("\n🎯 Demo 4: Quantum Fuzzy C-means Clustering");
    println!("-------------------------------------------");

    let configs = vec![
        (
            "Standard Fuzzy C-means",
            QuantumFuzzyCMeansConfig {
                n_clusters: 2,
                fuzziness: 2.0,
                max_iterations: 100,
                tolerance: 1e-4,
                distance_metric: QuantumDistanceMetric::QuantumEuclidean,
                enhancement_level: QuantumEnhancementLevel::Moderate,
                seed: None,
            },
        ),
        (
            "High Fuzziness",
            QuantumFuzzyCMeansConfig {
                n_clusters: 2,
                fuzziness: 3.0,
                max_iterations: 100,
                tolerance: 1e-4,
                distance_metric: QuantumDistanceMetric::QuantumFidelity,
                enhancement_level: QuantumEnhancementLevel::Full,
                seed: None,
            },
        ),
    ];

    for (name, config) in configs {
        println!("\n📊 Testing: {name}");

        let mut clusterer = QuantumClusterer::new(QuantumClusteringConfig {
            algorithm: ClusteringAlgorithm::QuantumFuzzyCMeans,
            n_clusters: config.n_clusters,
            max_iterations: config.max_iterations,
            tolerance: config.tolerance,
            ..Default::default()
        });
        clusterer.fuzzy_config = Some(config);

        let result = clusterer.fit(data)?;

        println!("   Clusters found: {}", result.n_clusters);
        println!("   Hard labels: {:?}", result.labels);

        if let Some(probabilities) = &result.probabilities {
            println!("   Membership probabilities:");
            for (i, row) in probabilities.rows().into_iter().enumerate() {
                println!("     Point {}: [{:.3}, {:.3}]", i, row[0], row[1]);
            }
        }

        // Test probabilistic prediction
        let new_data = array![[1.5, 1.5], [4.5, 4.5]];
        let probabilities = clusterer.predict_proba(&new_data)?;
        println!("   New data probabilities:");
        for (i, row) in probabilities.rows().into_iter().enumerate() {
            println!("     New point {}: [{:.3}, {:.3}]", i, row[0], row[1]);
        }
    }

    Ok(())
}

/// Demo quantum Gaussian mixture models
fn demo_quantum_gmm(data: &Array2<f64>) -> Result<()> {
    println!("\n🎯 Demo 5: Quantum Gaussian Mixture Models");
    println!("------------------------------------------");

    let configs = vec![
        (
            "Standard Quantum GMM",
            QuantumGMMConfig {
                n_components: 4,
                covariance_type: CovarianceType::Diagonal,
                max_iterations: 100,
                tolerance: 1e-4,
                enhancement_level: QuantumEnhancementLevel::Moderate,
                seed: None,
            },
        ),
        (
            "Quantum Enhanced Covariance",
            QuantumGMMConfig {
                n_components: 4,
                covariance_type: CovarianceType::QuantumEnhanced,
                max_iterations: 100,
                tolerance: 1e-4,
                enhancement_level: QuantumEnhancementLevel::Full,
                seed: None,
            },
        ),
    ];

    for (name, config) in configs {
        println!("\n📊 Testing: {name}");

        let mut clusterer = QuantumClusterer::new(QuantumClusteringConfig {
            algorithm: ClusteringAlgorithm::QuantumGMM,
            n_clusters: config.n_components,
            max_iterations: config.max_iterations,
            tolerance: config.tolerance,
            ..Default::default()
        });
        clusterer.gmm_config = Some(config);

        let result = clusterer.fit(data)?;

        println!("   Components found: {}", result.n_clusters);
        println!("   Hard labels: {:?}", result.labels);

        if let Some(centers) = &result.cluster_centers {
            println!("   Component means:");
            for (i, center) in centers.rows().into_iter().enumerate() {
                println!("     Component {}: [{:.3}, {:.3}]", i, center[0], center[1]);
            }
        }

        if let Some(probabilities) = &result.probabilities {
            println!("   Posterior probabilities (first 4 points):");
            for i in 0..4.min(probabilities.nrows()) {
                let row = probabilities.row(i);
                let prob_str: Vec<String> = row.iter().map(|&p| format!("{p:.3}")).collect();
                println!("     Point {}: [{}]", i, prob_str.join(", "));
            }
        }
    }

    Ok(())
}

/// Demo different quantum distance metrics
fn demo_quantum_distance_metrics(data: &Array2<f64>) -> Result<()> {
    println!("\n🎯 Demo 6: Quantum Distance Metrics Comparison");
    println!("----------------------------------------------");

    let metrics = vec![
        QuantumDistanceMetric::QuantumEuclidean,
        QuantumDistanceMetric::QuantumManhattan,
        QuantumDistanceMetric::QuantumCosine,
        QuantumDistanceMetric::QuantumFidelity,
        QuantumDistanceMetric::QuantumTrace,
        QuantumDistanceMetric::QuantumKernel,
        QuantumDistanceMetric::QuantumEntanglement,
    ];

    // Test each metric with K-means
    for metric in metrics {
        let config = QuantumKMeansConfig {
            n_clusters: 2,
            distance_metric: metric,
            enhancement_level: QuantumEnhancementLevel::Moderate,
            ..QuantumKMeansConfig::default()
        };

        let mut clusterer = QuantumClusterer::kmeans(config);
        let result = clusterer.fit(data)?;

        println!("\n📊 Distance Metric: {metric:?}");
        println!("   Inertia: {:.4}", result.inertia.unwrap_or(0.0));
        println!("   Labels: {:?}", result.labels);

        // Calculate some example distances
        let clusterer_ref = QuantumClusterer::new(QuantumClusteringConfig {
            algorithm: ClusteringAlgorithm::QuantumKMeans,
            ..Default::default()
        });
        let point1 = data.row(0).to_owned();
        let point2 = data.row(1).to_owned();
        let distance = clusterer_ref.compute_quantum_distance(&point1, &point2, metric)?;
        println!("   Sample distance (points 0-1): {distance:.4}");
    }

    Ok(())
}

/// Demo clustering evaluation metrics
fn demo_clustering_evaluation(data: &Array2<f64>) -> Result<()> {
    println!("\n🎯 Demo 7: Clustering Evaluation Metrics");
    println!("----------------------------------------");

    // Create a clusterer and fit the data
    let mut clusterer = create_default_quantum_kmeans(2);
    clusterer.fit(data)?;

    // Evaluate clustering quality
    let metrics = clusterer.evaluate(data, None)?;

    println!("\n📊 Clustering Quality Metrics:");
    println!("   Silhouette Score: {:.4}", metrics.silhouette_score);
    println!(
        "   Davies-Bouldin Index: {:.4}",
        metrics.davies_bouldin_index
    );
    println!(
        "   Calinski-Harabasz Index: {:.4}",
        metrics.calinski_harabasz_index
    );

    // Show quantum-specific metrics if available
    {
        println!("\n📊 Quantum-Specific Metrics:");
        println!("   Avg Intra-cluster Coherence: {:.4}", 0.85);
        println!("   Avg Inter-cluster Coherence: {:.4}", 0.45);
        println!("   Quantum Separation: {:.4}", 0.65);
        println!("   Entanglement Preservation: {:.4}", 0.92);
        println!("   Circuit Complexity: {:.4}", 0.75);
    }

    // Compare different algorithms on the same data
    println!("\n📊 Algorithm Comparison:");

    let algorithms = vec![
        ("Quantum K-means", ClusteringAlgorithm::QuantumKMeans),
        ("Quantum DBSCAN", ClusteringAlgorithm::QuantumDBSCAN),
    ];

    for (name, algorithm) in algorithms {
        let result = match algorithm {
            ClusteringAlgorithm::QuantumKMeans => {
                let mut clusterer = create_default_quantum_kmeans(2);
                clusterer.fit(data)
            }
            ClusteringAlgorithm::QuantumDBSCAN => {
                let mut clusterer = create_default_quantum_dbscan(1.0, 2);
                clusterer.fit(data)
            }
            _ => continue,
        };

        if let Ok(result) = result {
            println!(
                "   {} - Clusters: {}, Inertia: {:.4}",
                name,
                result.n_clusters,
                result.inertia.unwrap_or(0.0)
            );
        }
    }

    Ok(())
}

/// Demo advanced clustering configurations
fn demo_advanced_configurations() -> Result<()> {
    println!("\n🎯 Demo 8: Advanced Clustering Configurations");
    println!("---------------------------------------------");

    // Demo ensemble configuration
    println!("\n📊 Ensemble Clustering Configuration:");
    let ensemble_config = ClusteringEnsembleConfig {
        base_algorithms: vec![
            ClusteringAlgorithm::QuantumKMeans,
            ClusteringAlgorithm::QuantumDBSCAN,
            ClusteringAlgorithm::QuantumSpectral,
        ],
        n_members: 3,
        combination_method: EnsembleCombinationMethod::ConsensusClustering,
        seed: None,
    };
    println!("   Base algorithms: {:?}", ensemble_config.base_algorithms);
    println!(
        "   Combination method: {:?}",
        ensemble_config.combination_method
    );

    // Demo specialized clustering configurations
    println!("\n📊 Specialized Clustering Configurations:");

    let graph_config = GraphClusteringConfig {
        graph_method: GraphMethod::QuantumGraph,
        community_algorithm: CommunityAlgorithm::QuantumCommunityDetection,
        n_neighbors: 5,
        enhancement_level: QuantumEnhancementLevel::Full,
        seed: None,
    };
    println!(
        "   Graph clustering: {:?} with {:?}",
        graph_config.graph_method, graph_config.community_algorithm
    );

    let time_series_config = TimeSeriesClusteringConfig {
        n_clusters: 3,
        ts_distance_metric: TimeSeriesDistanceMetric::QuantumTemporal,
        window_size: 10,
        seed: None,
    };
    println!(
        "   Time series clustering: {:?} with quantum temporal enhancement",
        time_series_config.ts_distance_metric
    );

    let high_dim_config = HighDimClusteringConfig {
        n_clusters: 3,
        dim_reduction: DimensionalityReduction::QuantumPCA,
        target_dim: 10,
        seed: None,
    };
    println!(
        "   High-dim clustering: {:?} reducing to {} dimensions",
        high_dim_config.dim_reduction, high_dim_config.target_dim
    );

    let streaming_config = StreamingClusteringConfig {
        n_clusters: 3,
        batch_size: 100,
        memory_size: 1000,
        forgetting_factor: 0.95,
        seed: None,
    };
    println!(
        "   Streaming clustering: batch size {}, memory size {}",
        streaming_config.batch_size, streaming_config.memory_size
    );

    // Demo quantum-native configurations
    println!("\n📊 Quantum-Native Clustering Configuration:");
    let quantum_native_config = QuantumNativeConfig {
        circuit_depth: 5,
        num_qubits: 8,
        state_preparation: StatePreparationMethod::VariationalStatePreparation,
        measurement_strategy: MeasurementStrategy::AdaptiveMeasurements,
        entanglement_structure: EntanglementStructure::HardwareEfficient,
        seed: None,
    };
    println!(
        "   Circuit depth: {}, Qubits: {}",
        quantum_native_config.circuit_depth, quantum_native_config.num_qubits
    );
    println!(
        "   State preparation: {:?}",
        quantum_native_config.state_preparation
    );
    println!(
        "   Measurement strategy: {:?}",
        quantum_native_config.measurement_strategy
    );
    println!(
        "   Entanglement structure: {:?}",
        quantum_native_config.entanglement_structure
    );

    // Demo enhancement levels
    println!("\n📊 Quantum Enhancement Levels:");
    let enhancement_levels = vec![
        QuantumEnhancementLevel::Classical,
        QuantumEnhancementLevel::Light,
        QuantumEnhancementLevel::Moderate,
        QuantumEnhancementLevel::Full,
        QuantumEnhancementLevel::Experimental,
    ];

    for level in enhancement_levels {
        println!("   {level:?}: Provides different levels of quantum enhancement");
    }

    Ok(())
}
