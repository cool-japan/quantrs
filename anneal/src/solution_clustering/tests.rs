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
