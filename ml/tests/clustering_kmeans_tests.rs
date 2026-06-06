//! Integration tests for k-means clustering in QuantumClusterer

use quantrs2_ml::clustering::{
    create_default_quantum_dbscan, create_default_quantum_kmeans, ClusteringAlgorithm,
    QuantumClusterer, QuantumClusteringConfig,
};
use scirs2_core::ndarray::array;

// ---------------------------------------------------------------------------
// Basic k-means: two clearly separated clusters
// ---------------------------------------------------------------------------

#[test]
fn test_kmeans_two_cluster_separable() {
    // Blob A near (0,0), Blob B near (10,10) — well separated
    let data = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1]
    ];
    let mut cl = create_default_quantum_kmeans(2);
    let result = cl.fit(&data).expect("fit should succeed");

    assert_eq!(result.n_clusters, 2);
    let labels = &result.labels;

    // First three points must share one label
    assert_eq!(labels[0], labels[1], "points in blob A must share a label");
    assert_eq!(labels[0], labels[2], "points in blob A must share a label");

    // Last three points must share the other label
    assert_eq!(labels[3], labels[4], "points in blob B must share a label");
    assert_eq!(labels[3], labels[5], "points in blob B must share a label");

    // The two blobs must have different labels
    assert_ne!(labels[0], labels[3], "blobs must be in different clusters");
}

// ---------------------------------------------------------------------------
// Inertia is non-negative and strictly positive for non-trivial data
// ---------------------------------------------------------------------------

#[test]
fn test_kmeans_inertia_positive() {
    let data = array![[0.0_f64, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]];
    let mut cl = create_default_quantum_kmeans(2);
    let result = cl.fit(&data).expect("fit should succeed");

    let inertia = result.inertia.expect("inertia must be present");
    assert!(
        inertia >= 0.0,
        "inertia must be non-negative (got {inertia})"
    );
    // With two clusters and spread data the inertia must be > 0
    assert!(
        inertia > 1e-10,
        "inertia must be positive for non-trivial data (got {inertia})"
    );
}

// ---------------------------------------------------------------------------
// predict() agrees with fit() labels on training data
// ---------------------------------------------------------------------------

#[test]
fn test_predict_consistency() {
    let data = array![[0.0_f64, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]];
    let mut cl = create_default_quantum_kmeans(2);
    let fit_result = cl.fit(&data).expect("fit should succeed");
    let pred = cl.predict(&data).expect("predict should succeed");

    assert_eq!(pred.len(), 4, "predict must return one label per sample");

    // predict() must agree with fit() labels
    for (i, (&fit_label, &pred_label)) in fit_result.labels.iter().zip(pred.iter()).enumerate() {
        assert_eq!(
            fit_label, pred_label,
            "predict label mismatch at index {i}: fit={fit_label} predict={pred_label}"
        );
    }

    // Structural check: first two points share a label, last two share a different one
    assert_eq!(pred[0], pred[1], "first blob must share a label");
    assert_eq!(pred[2], pred[3], "second blob must share a label");
    assert_ne!(pred[0], pred[2], "blobs must differ");
}

// ---------------------------------------------------------------------------
// predict() before fit() must return an error
// ---------------------------------------------------------------------------

#[test]
fn test_predict_before_fit_errors() {
    let data = array![[0.0_f64, 0.0], [1.0, 1.0]];
    let config = QuantumClusteringConfig {
        algorithm: ClusteringAlgorithm::QuantumKMeans,
        n_clusters: 2,
        max_iterations: 100,
        tolerance: 1e-4,
        num_qubits: 4,
        random_state: None,
    };
    let cl = QuantumClusterer::new(config);
    assert!(cl.predict(&data).is_err(), "predict before fit must error");
}

// ---------------------------------------------------------------------------
// Single-cluster case: k=1 should assign all points to label 0
// ---------------------------------------------------------------------------

#[test]
fn test_kmeans_single_cluster() {
    let data = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let mut cl = create_default_quantum_kmeans(1);
    let result = cl.fit(&data).expect("fit should succeed");

    assert_eq!(result.n_clusters, 1);
    for label in &result.labels {
        assert_eq!(*label, 0, "all points must map to cluster 0");
    }
    assert!(result.inertia.unwrap() >= 0.0);
}

// ---------------------------------------------------------------------------
// k > n_samples: k must be clamped to n_samples
// ---------------------------------------------------------------------------

#[test]
fn test_kmeans_k_clamped_to_n_samples() {
    // 3 points but k=10 requested — should not panic and must return ≤ 3 clusters
    let data = array![[0.0_f64, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let mut cl = create_default_quantum_kmeans(10);
    let result = cl
        .fit(&data)
        .expect("fit should succeed even when k > n_samples");
    assert!(result.n_clusters <= 3, "n_clusters must be ≤ n_samples");
}

// ---------------------------------------------------------------------------
// fit_predict convenience wrapper
// ---------------------------------------------------------------------------

#[test]
fn test_fit_predict_wrapper() {
    let data = array![[0.0_f64, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]];
    let mut cl = create_default_quantum_kmeans(2);
    let labels = cl.fit_predict(&data).expect("fit_predict should succeed");
    assert_eq!(labels.len(), 4);
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[2], labels[3]);
    assert_ne!(labels[0], labels[2]);
}

// ---------------------------------------------------------------------------
// cluster_centers() accessor returns correct shape after fit
// ---------------------------------------------------------------------------

#[test]
fn test_cluster_centers_shape() {
    let data = array![
        [0.0_f64, 0.0, 1.0],
        [0.1, 0.0, 1.0],
        [10.0, 10.0, 1.0],
        [10.1, 10.0, 1.0]
    ];
    let mut cl = create_default_quantum_kmeans(2);
    cl.fit(&data).expect("fit should succeed");

    let centers = cl
        .cluster_centers()
        .expect("centers must be present after fit");
    assert_eq!(centers.nrows(), 2, "must have 2 cluster centers");
    assert_eq!(
        centers.ncols(),
        3,
        "centers must have same #features as data"
    );
}

// ---------------------------------------------------------------------------
// DBSCAN: two well-separated blobs should resolve to 2 clusters
// ---------------------------------------------------------------------------

#[test]
fn test_dbscan_two_blobs() {
    // eps=0.5 captures intra-blob neighbours; no cross-blob links possible (distance ≈ 14)
    let data = array![
        [0.0_f64, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1]
    ];
    let mut cl = create_default_quantum_dbscan(0.5, 1);
    let result = cl.fit(&data).expect("dbscan fit should succeed");

    // Both blobs should map to two internally consistent clusters
    let labels = &result.labels;
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[0], labels[2]);
    assert_eq!(labels[3], labels[4]);
    assert_eq!(labels[3], labels[5]);
    assert_ne!(labels[0], labels[3]);
}

// ---------------------------------------------------------------------------
// Inertia consistency: more clusters → inertia never increases
// ---------------------------------------------------------------------------

#[test]
fn test_inertia_decreases_with_more_clusters() {
    let data = array![
        [0.0_f64, 0.0],
        [1.0, 0.0],
        [5.0, 0.0],
        [6.0, 0.0],
        [10.0, 0.0],
        [11.0, 0.0]
    ];

    let mut cl2 = create_default_quantum_kmeans(2);
    let res2 = cl2.fit(&data).expect("k=2 fit");
    let inertia2 = res2.inertia.unwrap();

    let mut cl3 = create_default_quantum_kmeans(3);
    let res3 = cl3.fit(&data).expect("k=3 fit");
    let inertia3 = res3.inertia.unwrap();

    assert!(
        inertia3 <= inertia2 + 1e-9,
        "inertia with k=3 ({inertia3}) should be ≤ k=2 ({inertia2})"
    );
}
