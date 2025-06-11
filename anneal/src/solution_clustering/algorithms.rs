//! Clustering algorithms and distance metrics

use std::collections::HashMap;

/// Solution clustering algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum ClusteringAlgorithm {
    /// K-means clustering
    KMeans { k: usize, max_iterations: usize },

    /// Hierarchical clustering
    Hierarchical {
        linkage: LinkageType,
        distance_threshold: f64,
    },

    /// DBSCAN density-based clustering
    DBSCAN { eps: f64, min_samples: usize },

    /// Spectral clustering
    Spectral { k: usize, sigma: f64 },

    /// Gaussian Mixture Models
    GaussianMixture {
        components: usize,
        max_iterations: usize,
    },

    /// Mean-shift clustering
    MeanShift { bandwidth: f64 },

    /// Affinity propagation
    AffinityPropagation { damping: f64, max_iterations: usize },

    /// Custom clustering algorithm
    Custom {
        name: String,
        parameters: HashMap<String, f64>,
    },
}

/// Linkage types for hierarchical clustering
#[derive(Debug, Clone, PartialEq)]
pub enum LinkageType {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage
    Average,
    /// Ward linkage (minimize variance)
    Ward,
}

/// Distance metrics for clustering
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Hamming distance for binary vectors
    Hamming,
    /// Cosine distance
    Cosine,
    /// Jaccard distance
    Jaccard,
    /// Custom distance function
    Custom { name: String },
}
