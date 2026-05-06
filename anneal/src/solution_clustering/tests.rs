//! Tests for solution clustering

use super::*;
use crate::simulator::AnnealingSolution;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[test]
fn test_clustering_analyzer_creation() {
    let config = create_basic_clustering_config();
    let _analyzer = SolutionClusteringAnalyzer::new(config);
}

#[test]
fn test_solution_conversion() {
    let config = create_basic_clustering_config();
    let analyzer = SolutionClusteringAnalyzer::new(config);

    let solutions = vec![
        AnnealingSolution {
            best_spins: vec![1, -1, 1, -1],
            best_energy: -2.0,
            repetitions: 10,
            total_sweeps: 1000,
            runtime: Duration::from_millis(100),
            info: "Test solution 1".to_string(),
        },
        AnnealingSolution {
            best_spins: vec![-1, 1, -1, 1],
            best_energy: -1.5,
            repetitions: 12,
            total_sweeps: 1200,
            runtime: Duration::from_millis(120),
            info: "Test solution 2".to_string(),
        },
    ];

    let solution_points = analyzer
        .convert_solutions(&solutions)
        .expect("Failed to convert solutions");
    assert_eq!(solution_points.len(), 2);
    assert_eq!(solution_points[0].solution, vec![1, -1, 1, -1]);
    assert_eq!(solution_points[1].energy, -1.5);
}

#[test]
fn test_feature_extraction() {
    let config = ClusteringConfig {
        feature_extraction: FeatureExtractionMethod::Structural,
        ..create_basic_clustering_config()
    };
    let analyzer = SolutionClusteringAnalyzer::new(config);

    let solution_point = SolutionPoint {
        solution: vec![1, 1, -1, -1, 1],
        energy: -1.0,
        metrics: HashMap::new(),
        metadata: SolutionMetadata {
            id: 0,
            source: "test".to_string(),
            timestamp: Instant::now(),
            iterations: 100,
            quality_rank: None,
            is_feasible: true,
        },
        features: None,
    };

    let structural_features = analyzer.extract_structural_features(&solution_point.solution);
    assert_eq!(structural_features.len(), 6); // num_ones, num_neg_ones, fraction, max_consecutive_ones, max_consecutive_neg_ones, transitions
    assert_eq!(structural_features[0], 3.0); // num_ones
    assert_eq!(structural_features[1], 2.0); // num_neg_ones
}

#[test]
fn test_distance_calculations() {
    let config = create_basic_clustering_config();
    let analyzer = SolutionClusteringAnalyzer::new(config);

    let features1 = vec![1.0, 2.0, 3.0];
    let features2 = vec![4.0, 5.0, 6.0];

    let euclidean_dist = analyzer
        .calculate_distance(&features1, &features2)
        .expect("Failed to calculate distance");
    assert!((euclidean_dist - 5.196_152_422_706_632).abs() < 1e-10);
}

#[test]
fn test_kmeans_clustering() {
    let config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans {
            k: 2,
            max_iterations: 10,
        },
        seed: Some(42),
        ..create_basic_clustering_config()
    };
    let analyzer = SolutionClusteringAnalyzer::new(config);

    let solution_points = vec![
        SolutionPoint {
            solution: vec![1, 1, 1],
            energy: -3.0,
            metrics: HashMap::new(),
            metadata: SolutionMetadata {
                id: 0,
                source: "test".to_string(),
                timestamp: Instant::now(),
                iterations: 100,
                quality_rank: None,
                is_feasible: true,
            },
            features: Some(vec![1.0, 1.0, 1.0]),
        },
        SolutionPoint {
            solution: vec![-1, -1, -1],
            energy: 3.0,
            metrics: HashMap::new(),
            metadata: SolutionMetadata {
                id: 1,
                source: "test".to_string(),
                timestamp: Instant::now(),
                iterations: 100,
                quality_rank: None,
                is_feasible: true,
            },
            features: Some(vec![-1.0, -1.0, -1.0]),
        },
        SolutionPoint {
            solution: vec![1, 1, -1],
            energy: -1.0,
            metrics: HashMap::new(),
            metadata: SolutionMetadata {
                id: 2,
                source: "test".to_string(),
                timestamp: Instant::now(),
                iterations: 100,
                quality_rank: None,
                is_feasible: true,
            },
            features: Some(vec![1.0, 1.0, -1.0]),
        },
    ];

    let clusters = analyzer
        .kmeans_clustering(&solution_points, 2, 10)
        .expect("K-means clustering failed");
    assert!(clusters.len() <= 2);

    for cluster in &clusters {
        assert!(!cluster.solutions.is_empty());
        assert_eq!(cluster.centroid.len(), 3);
    }
}

#[test]
fn test_energy_statistics() {
    let config = create_basic_clustering_config();
    let analyzer = SolutionClusteringAnalyzer::new(config);

    let solution_points = vec![
        SolutionPoint {
            solution: vec![1, -1],
            energy: -2.0,
            metrics: HashMap::new(),
            metadata: SolutionMetadata {
                id: 0,
                source: "test".to_string(),
                timestamp: Instant::now(),
                iterations: 100,
                quality_rank: None,
                is_feasible: true,
            },
            features: None,
        },
        SolutionPoint {
            solution: vec![-1, 1],
            energy: -1.0,
            metrics: HashMap::new(),
            metadata: SolutionMetadata {
                id: 1,
                source: "test".to_string(),
                timestamp: Instant::now(),
                iterations: 100,
                quality_rank: None,
                is_feasible: true,
            },
            features: None,
        },
        SolutionPoint {
            solution: vec![1, 1],
            energy: 0.0,
            metrics: HashMap::new(),
            metadata: SolutionMetadata {
                id: 2,
                source: "test".to_string(),
                timestamp: Instant::now(),
                iterations: 100,
                quality_rank: None,
                is_feasible: true,
            },
            features: None,
        },
    ];

    let stats = analyzer.calculate_energy_statistics(&solution_points);
    assert_eq!(stats.min, -2.0);
    assert_eq!(stats.max, 0.0);
    assert!((stats.mean - (-1.0)).abs() < 1e-10);
    assert_eq!(stats.num_distinct_energies, 3);
}

#[test]
fn test_solution_diversity() {
    let solutions = vec![
        AnnealingSolution {
            best_spins: vec![1, -1, 1, -1],
            best_energy: -2.0,
            repetitions: 10,
            total_sweeps: 1000,
            runtime: Duration::from_millis(100),
            info: "Test solution 1".to_string(),
        },
        AnnealingSolution {
            best_spins: vec![-1, 1, -1, 1],
            best_energy: -1.5,
            repetitions: 12,
            total_sweeps: 1200,
            runtime: Duration::from_millis(120),
            info: "Test solution 2".to_string(),
        },
        AnnealingSolution {
            best_spins: vec![1, 1, 1, 1],
            best_energy: -1.0,
            repetitions: 8,
            total_sweeps: 800,
            runtime: Duration::from_millis(80),
            info: "Test solution 3".to_string(),
        },
    ];

    let diversity =
        analyze_solution_diversity(&solutions).expect("Failed to analyze solution diversity");
    assert!(diversity > 0.0);
    assert!(diversity <= 4.0); // Maximum Hamming distance for 4-bit strings
}

#[test]
fn test_comprehensive_config() {
    let config = create_comprehensive_clustering_config();
    assert!(matches!(
        config.algorithm,
        ClusteringAlgorithm::DBSCAN { .. }
    ));
    assert_eq!(config.analysis_depth, AnalysisDepth::Comprehensive);
    assert_eq!(
        config.feature_extraction,
        FeatureExtractionMethod::Structural
    );
}

// --- Helpers for the foundational-method tests below ------------------------

fn make_point(id: usize, solution: Vec<i8>, features: Vec<f64>, energy: f64) -> SolutionPoint {
    SolutionPoint {
        solution,
        energy,
        metrics: HashMap::new(),
        metadata: SolutionMetadata {
            id,
            source: "test".to_string(),
            timestamp: Instant::now(),
            iterations: 1,
            quality_rank: None,
            is_feasible: true,
        },
        features: Some(features),
    }
}

fn hamming_config() -> ClusteringConfig {
    ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans {
            k: 2,
            max_iterations: 50,
        },
        distance_metric: DistanceMetric::Hamming,
        feature_extraction: FeatureExtractionMethod::Raw,
        seed: Some(7),
        ..create_basic_clustering_config()
    }
}

fn euclidean_config_seeded() -> ClusteringConfig {
    ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans {
            k: 2,
            max_iterations: 100,
        },
        distance_metric: DistanceMetric::Euclidean,
        feature_extraction: FeatureExtractionMethod::Raw,
        seed: Some(42),
        ..create_basic_clustering_config()
    }
}

#[test]
fn test_hamming_distance_correct_on_known_pairs() {
    let analyzer = SolutionClusteringAnalyzer::new(hamming_config());

    // Identical vectors: distance 0.
    let zero = analyzer
        .calculate_distance(&[1.0, -1.0, 1.0, -1.0], &[1.0, -1.0, 1.0, -1.0])
        .expect("distance call must succeed");
    assert!((zero - 0.0).abs() < 1e-12);

    // All four positions differ: distance 4.
    let four = analyzer
        .calculate_distance(&[1.0, 1.0, 1.0, 1.0], &[-1.0, -1.0, -1.0, -1.0])
        .expect("distance call must succeed");
    assert!((four - 4.0).abs() < 1e-12);

    // Two of three differ.
    let two = analyzer
        .calculate_distance(&[1.0, 1.0, 1.0], &[-1.0, 1.0, -1.0])
        .expect("distance call must succeed");
    assert!((two - 2.0).abs() < 1e-12);

    // Symmetry.
    let a = analyzer
        .calculate_distance(&[1.0, 0.0, 1.0], &[0.0, 1.0, 1.0])
        .expect("distance");
    let b = analyzer
        .calculate_distance(&[0.0, 1.0, 1.0], &[1.0, 0.0, 1.0])
        .expect("distance");
    assert!((a - b).abs() < 1e-12);
}

#[test]
fn test_silhouette_positive_for_well_separated_clusters() {
    // Two tight, well-separated clusters in 2D feature space:
    //   cluster A: points around (10, 10)
    //   cluster B: points around (-10, -10)
    // The intra-cluster distances are tiny, the inter-cluster distance is huge,
    // so per-point silhouettes should approach 1.
    let analyzer = SolutionClusteringAnalyzer::new(euclidean_config_seeded());

    let mut clusters = vec![
        SolutionCluster {
            id: 0,
            solutions: vec![
                make_point(0, vec![1, 1], vec![10.0, 10.0], 0.0),
                make_point(1, vec![1, 1], vec![10.1, 10.0], 0.0),
                make_point(2, vec![1, 1], vec![10.0, 10.1], 0.0),
            ],
            centroid: vec![10.033, 10.033],
            representative: None,
            statistics: ClusterStatistics {
                size: 3,
                mean_energy: 0.0,
                energy_std: 0.0,
                min_energy: 0.0,
                max_energy: 0.0,
                intra_cluster_distance: 0.0,
                diameter: 0.0,
                density: 0.0,
            },
            quality_metrics: ClusterQualityMetrics {
                silhouette_coefficient: 0.0,
                inertia: 0.0,
                calinski_harabasz_index: 0.0,
                davies_bouldin_index: 0.0,
                stability: 0.0,
            },
        },
        SolutionCluster {
            id: 1,
            solutions: vec![
                make_point(3, vec![-1, -1], vec![-10.0, -10.0], 0.0),
                make_point(4, vec![-1, -1], vec![-10.1, -10.0], 0.0),
                make_point(5, vec![-1, -1], vec![-10.0, -10.1], 0.0),
            ],
            centroid: vec![-10.033, -10.033],
            representative: None,
            statistics: ClusterStatistics {
                size: 3,
                mean_energy: 0.0,
                energy_std: 0.0,
                min_energy: 0.0,
                max_energy: 0.0,
                intra_cluster_distance: 0.0,
                diameter: 0.0,
                density: 0.0,
            },
            quality_metrics: ClusterQualityMetrics {
                silhouette_coefficient: 0.0,
                inertia: 0.0,
                calinski_harabasz_index: 0.0,
                davies_bouldin_index: 0.0,
                stability: 0.0,
            },
        },
    ];

    analyzer
        .update_global_quality_metrics(&mut clusters)
        .expect("post-pass quality computation must succeed");

    for c in &clusters {
        assert!(
            c.quality_metrics.silhouette_coefficient > 0.9,
            "well-separated cluster {} should have silhouette ~1, got {}",
            c.id,
            c.quality_metrics.silhouette_coefficient
        );
    }
}

#[test]
fn test_davies_bouldin_lower_for_better_clustering() {
    // Build two cluster sets over the same six 2D points:
    //  GOOD: split by y-sign (tight, well separated).
    //  BAD: split by index parity (interleaved — high intra/centroid-distance ratio).
    let analyzer = SolutionClusteringAnalyzer::new(euclidean_config_seeded());

    let pa = make_point(0, vec![1, 1], vec![5.0, 5.0], 0.0);
    let pb = make_point(1, vec![1, 1], vec![5.1, 4.9], 0.0);
    let pc = make_point(2, vec![1, 1], vec![4.9, 5.1], 0.0);
    let pd = make_point(3, vec![-1, -1], vec![-5.0, -5.0], 0.0);
    let pe = make_point(4, vec![-1, -1], vec![-5.1, -4.9], 0.0);
    let pf = make_point(5, vec![-1, -1], vec![-4.9, -5.1], 0.0);

    let mut good = vec![
        SolutionCluster {
            id: 0,
            solutions: vec![pa.clone(), pb.clone(), pc.clone()],
            centroid: vec![5.0, 5.0],
            representative: None,
            statistics: ClusterStatistics {
                size: 3,
                mean_energy: 0.0,
                energy_std: 0.0,
                min_energy: 0.0,
                max_energy: 0.0,
                intra_cluster_distance: 0.0,
                diameter: 0.0,
                density: 0.0,
            },
            quality_metrics: ClusterQualityMetrics {
                silhouette_coefficient: 0.0,
                inertia: 0.0,
                calinski_harabasz_index: 0.0,
                davies_bouldin_index: 0.0,
                stability: 0.0,
            },
        },
        SolutionCluster {
            id: 1,
            solutions: vec![pd.clone(), pe.clone(), pf.clone()],
            centroid: vec![-5.0, -5.0],
            representative: None,
            statistics: ClusterStatistics {
                size: 3,
                mean_energy: 0.0,
                energy_std: 0.0,
                min_energy: 0.0,
                max_energy: 0.0,
                intra_cluster_distance: 0.0,
                diameter: 0.0,
                density: 0.0,
            },
            quality_metrics: ClusterQualityMetrics {
                silhouette_coefficient: 0.0,
                inertia: 0.0,
                calinski_harabasz_index: 0.0,
                davies_bouldin_index: 0.0,
                stability: 0.0,
            },
        },
    ];

    let mut bad = vec![
        SolutionCluster {
            id: 0,
            solutions: vec![pa.clone(), pd.clone(), pc.clone()],
            centroid: vec![1.633, 1.7],
            representative: None,
            statistics: ClusterStatistics {
                size: 3,
                mean_energy: 0.0,
                energy_std: 0.0,
                min_energy: 0.0,
                max_energy: 0.0,
                intra_cluster_distance: 0.0,
                diameter: 0.0,
                density: 0.0,
            },
            quality_metrics: ClusterQualityMetrics {
                silhouette_coefficient: 0.0,
                inertia: 0.0,
                calinski_harabasz_index: 0.0,
                davies_bouldin_index: 0.0,
                stability: 0.0,
            },
        },
        SolutionCluster {
            id: 1,
            solutions: vec![pb.clone(), pe.clone(), pf.clone()],
            centroid: vec![-1.633, -1.633],
            representative: None,
            statistics: ClusterStatistics {
                size: 3,
                mean_energy: 0.0,
                energy_std: 0.0,
                min_energy: 0.0,
                max_energy: 0.0,
                intra_cluster_distance: 0.0,
                diameter: 0.0,
                density: 0.0,
            },
            quality_metrics: ClusterQualityMetrics {
                silhouette_coefficient: 0.0,
                inertia: 0.0,
                calinski_harabasz_index: 0.0,
                davies_bouldin_index: 0.0,
                stability: 0.0,
            },
        },
    ];

    analyzer
        .update_global_quality_metrics(&mut good)
        .expect("good clustering quality update must succeed");
    analyzer
        .update_global_quality_metrics(&mut bad)
        .expect("bad clustering quality update must succeed");

    // For two clusters DB is symmetric: both clusters share the same DB value.
    let good_db = good[0].quality_metrics.davies_bouldin_index;
    let bad_db = bad[0].quality_metrics.davies_bouldin_index;
    assert!(
        good_db < bad_db,
        "good clustering should have lower DB ({}) than bad clustering ({})",
        good_db,
        bad_db
    );
    // Sanity: lower DB == better, so the good clustering should be quite small.
    assert!(good_db < 0.1, "good DB unexpectedly large: {}", good_db);
}

#[test]
fn test_centroid_of_singleton_equals_singleton() {
    // Indirect test via k-means with k=1: the unique cluster's centroid must
    // equal the only point's feature vector exactly.
    let config = ClusteringConfig {
        algorithm: ClusteringAlgorithm::KMeans {
            k: 1,
            max_iterations: 20,
        },
        distance_metric: DistanceMetric::Euclidean,
        feature_extraction: FeatureExtractionMethod::Raw,
        seed: Some(13),
        ..create_basic_clustering_config()
    };
    let analyzer = SolutionClusteringAnalyzer::new(config);

    let point = make_point(0, vec![1, -1, 1], vec![1.0, -1.0, 1.0], 0.0);
    let clusters = analyzer
        .kmeans_clustering(&[point], 1, 20)
        .expect("k=1 must produce one cluster");

    assert_eq!(clusters.len(), 1);
    let c = &clusters[0];
    assert_eq!(c.solutions.len(), 1);
    assert_eq!(c.centroid.len(), 3);
    for (i, &v) in c.centroid.iter().enumerate() {
        let expected = [1.0, -1.0, 1.0][i];
        assert!(
            (v - expected).abs() < 1e-12,
            "singleton centroid[{}]={} != expected {}",
            i,
            v,
            expected
        );
    }
}

#[test]
fn test_intra_cluster_variance_zero_for_identical_points() {
    // Three identical points form one tight cluster; intra_cluster_distance
    // and inertia must both be exactly zero, and silhouette is 0 by convention.
    let analyzer = SolutionClusteringAnalyzer::new(euclidean_config_seeded());

    let p0 = make_point(0, vec![1, -1], vec![2.0, 3.0], 0.0);
    let p1 = make_point(1, vec![1, -1], vec![2.0, 3.0], 0.0);
    let p2 = make_point(2, vec![1, -1], vec![2.0, 3.0], 0.0);

    let clusters = analyzer
        .kmeans_clustering(&[p0, p1, p2], 1, 50)
        .expect("kmeans on identical points must succeed");
    assert_eq!(clusters.len(), 1);
    let c = &clusters[0];

    assert!(
        c.statistics.intra_cluster_distance.abs() < 1e-10,
        "identical points must have zero intra-cluster distance, got {}",
        c.statistics.intra_cluster_distance
    );
    assert!(
        c.quality_metrics.inertia.abs() < 1e-10,
        "identical points must have zero inertia, got {}",
        c.quality_metrics.inertia
    );
}

#[test]
fn test_calinski_harabasz_higher_for_better_clustering() {
    // CH should be larger for tightly-grouped, well-separated clusters than
    // for an interleaved partition of the same points.
    let analyzer = SolutionClusteringAnalyzer::new(euclidean_config_seeded());

    let pa = make_point(0, vec![1, 1], vec![5.0, 5.0], 0.0);
    let pb = make_point(1, vec![1, 1], vec![5.1, 4.9], 0.0);
    let pc = make_point(2, vec![1, 1], vec![4.9, 5.1], 0.0);
    let pd = make_point(3, vec![-1, -1], vec![-5.0, -5.0], 0.0);
    let pe = make_point(4, vec![-1, -1], vec![-5.1, -4.9], 0.0);
    let pf = make_point(5, vec![-1, -1], vec![-4.9, -5.1], 0.0);

    fn make_cluster(id: usize, solutions: Vec<SolutionPoint>, centroid: Vec<f64>) -> SolutionCluster {
        // Pre-compute per-cluster inertia so update_global_quality_metrics can
        // compute CH (which depends on inertia).
        let inertia: f64 = solutions
            .iter()
            .map(|p| {
                p.features
                    .as_ref()
                    .map(|f| {
                        f.iter()
                            .zip(centroid.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                    })
                    .unwrap_or(0.0)
            })
            .sum();
        SolutionCluster {
            id,
            solutions,
            centroid,
            representative: None,
            statistics: ClusterStatistics {
                size: 3,
                mean_energy: 0.0,
                energy_std: 0.0,
                min_energy: 0.0,
                max_energy: 0.0,
                intra_cluster_distance: 0.0,
                diameter: 0.0,
                density: 0.0,
            },
            quality_metrics: ClusterQualityMetrics {
                silhouette_coefficient: 0.0,
                inertia,
                calinski_harabasz_index: 0.0,
                davies_bouldin_index: 0.0,
                stability: 0.0,
            },
        }
    }

    let mut good = vec![
        make_cluster(0, vec![pa.clone(), pb.clone(), pc.clone()], vec![5.0, 5.0]),
        make_cluster(1, vec![pd.clone(), pe.clone(), pf.clone()], vec![-5.0, -5.0]),
    ];
    let mut bad = vec![
        make_cluster(0, vec![pa.clone(), pd.clone(), pc.clone()], vec![1.633, 1.7]),
        make_cluster(1, vec![pb.clone(), pe.clone(), pf.clone()], vec![-1.633, -1.633]),
    ];

    analyzer
        .update_global_quality_metrics(&mut good)
        .expect("good update must succeed");
    analyzer
        .update_global_quality_metrics(&mut bad)
        .expect("bad update must succeed");

    let good_ch = good[0].quality_metrics.calinski_harabasz_index;
    let bad_ch = bad[0].quality_metrics.calinski_harabasz_index;
    assert!(
        good_ch > bad_ch,
        "good clustering should have higher CH ({}) than bad ({})",
        good_ch,
        bad_ch
    );
}

#[test]
fn test_connectivity_components_via_hamming_neighbours() {
    // Two well-separated solution clouds in spin space; the union-find pass
    // should detect exactly two connected components.
    let config = create_basic_clustering_config();
    let analyzer = SolutionClusteringAnalyzer::new(config);

    let solutions = vec![
        // Component A: three solutions within Hamming-1 of each other.
        make_point(0, vec![1, 1, 1, 1], vec![1.0, 1.0, 1.0, 1.0], 0.0),
        make_point(1, vec![1, 1, 1, -1], vec![1.0, 1.0, 1.0, -1.0], 0.0),
        make_point(2, vec![1, 1, -1, -1], vec![1.0, 1.0, -1.0, -1.0], 0.0),
        // Component B: completely opposite, far from A in Hamming distance.
        make_point(3, vec![-1, -1, -1, -1], vec![-1.0, -1.0, -1.0, -1.0], 0.0),
        make_point(4, vec![-1, -1, -1, 1], vec![-1.0, -1.0, -1.0, 1.0], 0.0),
    ];

    let connectivity = analyzer.analyze_connectivity(&solutions);
    assert_eq!(
        connectivity.num_components, 2,
        "expected two connected components"
    );
    assert!(
        connectivity.largest_component_size >= 2,
        "largest component too small: {}",
        connectivity.largest_component_size
    );
}

#[test]
fn test_autocorrelation_high_for_smooth_energy_walk() {
    // Monotone energies => lag-1 autocorrelation should be strongly positive,
    // so ruggedness_coefficient = 1 - rho1 should be small.
    let config = create_basic_clustering_config();
    let analyzer = SolutionClusteringAnalyzer::new(config);

    let mut solution_points = Vec::new();
    for i in 0..10usize {
        solution_points.push(make_point(
            i,
            vec![1, -1],
            vec![1.0, -1.0],
            i as f64,
        ));
    }

    let metrics = analyzer.calculate_ruggedness_metrics(&solution_points);
    assert!(!metrics.autocorrelation.is_empty());
    let rho1 = metrics.autocorrelation[0];
    assert!(
        rho1 > 0.5,
        "monotone energies should have high lag-1 autocorrelation, got {}",
        rho1
    );
    assert!(
        metrics.ruggedness_coefficient < 0.5,
        "smooth landscape should have small ruggedness, got {}",
        metrics.ruggedness_coefficient
    );
}

#[test]
fn test_multi_modality_detects_two_modes() {
    // Two energy clusters separated by an empty middle should yield two modes.
    let config = create_basic_clustering_config();
    let analyzer = SolutionClusteringAnalyzer::new(config);

    let mut solution_points = Vec::new();
    // Cluster around energy = -10
    for i in 0..6 {
        solution_points.push(make_point(
            i,
            vec![1, 1],
            vec![1.0, 1.0],
            -10.0 + (i as f64) * 0.05,
        ));
    }
    // Cluster around energy = +10
    for i in 0..6 {
        solution_points.push(make_point(
            6 + i,
            vec![-1, -1],
            vec![-1.0, -1.0],
            10.0 + (i as f64) * 0.05,
        ));
    }

    let mm = analyzer.analyze_multi_modality(&solution_points);
    assert!(
        mm.num_modes >= 2,
        "expected at least two modes for bimodal energy distribution, got {}",
        mm.num_modes
    );
    assert_eq!(mm.mode_strengths.len(), mm.num_modes);
    assert_eq!(mm.inter_mode_distances.len(), mm.num_modes);
    let total_strength: f64 = mm.mode_strengths.iter().sum();
    assert!(
        total_strength > 0.0 && total_strength <= 1.0 + 1e-9,
        "mode strengths must sum to a value in (0, 1], got {}",
        total_strength
    );
}

#[test]
fn test_basin_depth_zero_for_global_minimum() {
    // The cluster containing the lowest-energy point should have depth ~0.
    let config = create_basic_clustering_config();
    let analyzer = SolutionClusteringAnalyzer::new(config);

    let solution_points = vec![
        make_point(0, vec![1, 1], vec![1.0, 1.0], -5.0),
        make_point(1, vec![1, -1], vec![1.0, -1.0], -1.0),
    ];

    let clusters = vec![
        SolutionCluster {
            id: 0,
            solutions: vec![solution_points[0].clone()],
            centroid: vec![1.0, 1.0],
            representative: None,
            statistics: ClusterStatistics {
                size: 1,
                mean_energy: -5.0,
                energy_std: 0.0,
                min_energy: -5.0,
                max_energy: -5.0,
                intra_cluster_distance: 0.0,
                diameter: 0.0,
                density: 0.0,
            },
            quality_metrics: ClusterQualityMetrics {
                silhouette_coefficient: 0.0,
                inertia: 0.0,
                calinski_harabasz_index: 0.0,
                davies_bouldin_index: 0.0,
                stability: 0.0,
            },
        },
        SolutionCluster {
            id: 1,
            solutions: vec![solution_points[1].clone()],
            centroid: vec![1.0, -1.0],
            representative: None,
            statistics: ClusterStatistics {
                size: 1,
                mean_energy: -1.0,
                energy_std: 0.0,
                min_energy: -1.0,
                max_energy: -1.0,
                intra_cluster_distance: 0.0,
                diameter: 0.0,
                density: 0.0,
            },
            quality_metrics: ClusterQualityMetrics {
                silhouette_coefficient: 0.0,
                inertia: 0.0,
                calinski_harabasz_index: 0.0,
                davies_bouldin_index: 0.0,
                stability: 0.0,
            },
        },
    ];

    let basins = analyzer.detect_energy_basins(&solution_points, &clusters);
    assert_eq!(basins.len(), 2);
    // First basin is the global minimum, depth must be 0.
    assert!(basins[0].depth.abs() < 1e-12, "global-min basin depth should be 0, got {}", basins[0].depth);
    // Second basin sits 4.0 above the global minimum.
    assert!((basins[1].depth - 4.0).abs() < 1e-12, "depth of shallow basin should be 4.0, got {}", basins[1].depth);
}
