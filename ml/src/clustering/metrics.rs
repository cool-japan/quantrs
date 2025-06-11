//! Clustering evaluation metrics

/// Clustering evaluation metrics
#[derive(Debug, Clone)]
pub struct ClusteringMetrics {
    pub silhouette_score: f64,
    pub calinski_harabasz_score: f64,
    pub davies_bouldin_score: f64,
    pub adjusted_rand_index: f64,
    pub adjusted_mutual_info: f64,
}

impl ClusteringMetrics {
    pub fn new() -> Self {
        Self {
            silhouette_score: 0.0,
            calinski_harabasz_score: 0.0,
            davies_bouldin_score: 0.0,
            adjusted_rand_index: 0.0,
            adjusted_mutual_info: 0.0,
        }
    }
}
