//! Graph analysis components and result types

use super::*;

/// Comprehensive mapping result with SciRS2 analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SciRS2MappingResult {
    /// Initial logical-to-physical mapping
    pub initial_mapping: HashMap<usize, usize>,
    /// Final mapping after optimization
    pub final_mapping: HashMap<usize, usize>,
    /// Sequence of swap operations
    pub swap_operations: Vec<SwapOperation>,
    /// Graph analysis results
    pub graph_analysis: GraphAnalysisResult,
    /// Spectral analysis results
    pub spectral_analysis: Option<SpectralAnalysisResult>,
    /// Community structure analysis
    pub community_analysis: CommunityAnalysisResult,
    /// Centrality analysis
    pub centrality_analysis: CentralityAnalysisResult,
    /// Optimization metrics
    pub optimization_metrics: OptimizationMetrics,
    /// Performance predictions
    pub performance_predictions: Option<PerformancePredictions>,
    /// Real-time analytics results
    pub realtime_analytics: RealtimeAnalyticsResult,
    /// ML model performance
    pub ml_performance: Option<MLPerformanceResult>,
    /// Adaptive learning insights
    pub adaptive_insights: AdaptiveMappingInsights,
    /// Optimization recommendations
    pub optimization_recommendations: OptimizationRecommendations,
}

/// Graph analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAnalysisResult {
    /// Graph density
    pub density: f64,
    /// Average clustering coefficient
    pub clustering_coefficient: f64,
    /// Graph diameter
    pub diameter: usize,
    /// Graph radius
    pub radius: usize,
    /// Average path length
    pub average_path_length: f64,
    /// Connectivity statistics
    pub connectivity_stats: ConnectivityStats,
    /// Topological properties
    pub topological_properties: TopologicalProperties,
}

/// Spectral analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysisResult {
    /// Eigenvalues of the Laplacian matrix
    pub laplacian_eigenvalues: Array1<f64>,
    /// Eigenvectors for embedding
    pub embedding_vectors: Array2<f64>,
    /// Spectral radius
    pub spectral_radius: f64,
    /// Algebraic connectivity
    pub algebraic_connectivity: f64,
    /// Spectral gap
    pub spectral_gap: f64,
    /// Embedding quality metrics
    pub embedding_quality: EmbeddingQuality,
}

/// Community analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityAnalysisResult {
    /// Community assignments
    pub communities: HashMap<usize, usize>,
    /// Modularity score
    pub modularity: f64,
    /// Number of communities
    pub num_communities: usize,
    /// Community sizes
    pub community_sizes: Vec<usize>,
    /// Inter-community connections
    pub inter_community_edges: usize,
    /// Community quality metrics
    pub quality_metrics: CommunityQualityMetrics,
}

/// Centrality analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityAnalysisResult {
    /// Betweenness centrality for each node
    pub betweenness_centrality: HashMap<usize, f64>,
    /// Closeness centrality for each node
    pub closeness_centrality: HashMap<usize, f64>,
    /// Eigenvector centrality for each node
    pub eigenvector_centrality: HashMap<usize, f64>,
    /// PageRank centrality for each node
    pub pagerank_centrality: HashMap<usize, f64>,
    /// Correlation matrix between centrality measures
    pub centrality_correlations: Array2<f64>,
    /// Statistical summary of centrality measures
    pub centrality_statistics: CentralityStatistics,
}

/// Connectivity statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityStats {
    /// Edge connectivity
    pub edge_connectivity: usize,
    /// Vertex connectivity
    pub vertex_connectivity: usize,
    /// Algebraic connectivity
    pub algebraic_connectivity: f64,
    /// Whether graph is connected
    pub is_connected: bool,
    /// Number of connected components
    pub num_components: usize,
    /// Size of largest component
    pub largest_component_size: usize,
}

/// Topological properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalProperties {
    /// Whether graph is planar
    pub is_planar: bool,
    /// Whether graph is bipartite
    pub is_bipartite: bool,
    /// Whether graph is a tree
    pub is_tree: bool,
    /// Whether graph is a forest
    pub is_forest: bool,
    /// Whether graph has cycles
    pub has_cycles: bool,
    /// Girth (shortest cycle length)
    pub girth: usize,
    /// Chromatic number
    pub chromatic_number: usize,
    /// Independence number
    pub independence_number: usize,
}

/// Embedding quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingQuality {
    /// Stress value (lower is better)
    pub stress: f64,
    /// Distortion measure
    pub distortion: f64,
    /// Distance preservation ratio
    pub preservation_ratio: f64,
    /// Dimensionality of embedding
    pub embedding_dimension: usize,
}

/// Community quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityQualityMetrics {
    /// Silhouette score
    pub silhouette_score: f64,
    /// Conductance measure
    pub conductance: f64,
    /// Coverage ratio
    pub coverage: f64,
    /// Overall performance score
    pub performance: f64,
}

/// Statistical summary of centrality measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityStatistics {
    /// Maximum betweenness centrality
    pub max_betweenness: f64,
    /// Maximum closeness centrality
    pub max_closeness: f64,
    /// Maximum eigenvector centrality
    pub max_eigenvector: f64,
    /// Maximum PageRank centrality
    pub max_pagerank: f64,
    /// Mean betweenness centrality
    pub mean_betweenness: f64,
    /// Mean closeness centrality
    pub mean_closeness: f64,
    /// Mean eigenvector centrality
    pub mean_eigenvector: f64,
    /// Mean PageRank centrality
    pub mean_pagerank: f64,
}

/// Path analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathAnalysisResult {
    /// Shortest path distances between all pairs
    pub distance_matrix: Array2<f64>,
    /// Path efficiency measures
    pub efficiency_measures: EfficiencyMeasures,
    /// Routing quality metrics
    pub routing_quality: RoutingQualityMetrics,
}

/// Efficiency measures for graph paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMeasures {
    /// Global efficiency
    pub global_efficiency: f64,
    /// Local efficiency
    pub local_efficiency: f64,
    /// Average efficiency
    pub average_efficiency: f64,
    /// Efficiency distribution
    pub efficiency_distribution: Vec<f64>,
}

/// Routing quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingQualityMetrics {
    /// Average path length
    pub average_path_length: f64,
    /// Path length distribution
    pub path_length_distribution: HashMap<usize, usize>,
    /// Routing overhead
    pub routing_overhead: f64,
    /// Load balancing index
    pub load_balancing_index: f64,
}

/// Network flow analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowAnalysisResult {
    /// Maximum flow between critical pairs
    pub max_flows: HashMap<(usize, usize), f64>,
    /// Minimum cut analysis
    pub min_cuts: Vec<MinCutResult>,
    /// Bottleneck identification
    pub bottlenecks: Vec<BottleneckInfo>,
    /// Flow distribution metrics
    pub flow_metrics: FlowMetrics,
}

/// Minimum cut result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinCutResult {
    /// Source and sink nodes
    pub source_sink: (usize, usize),
    /// Minimum cut value
    pub cut_value: f64,
    /// Edges in the cut
    pub cut_edges: Vec<(usize, usize)>,
    /// Partition of nodes
    pub partition: (Vec<usize>, Vec<usize>),
}

/// Bottleneck information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckInfo {
    /// Bottleneck edge or node
    pub location: BottleneckLocation,
    /// Severity of bottleneck
    pub severity: f64,
    /// Impact on flow
    pub flow_impact: f64,
    /// Suggested improvements
    pub improvement_suggestions: Vec<String>,
}

/// Location of a bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckLocation {
    Edge(usize, usize),
    Node(usize),
    Region(Vec<usize>),
}

/// Flow metrics for the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMetrics {
    /// Total flow capacity
    pub total_capacity: f64,
    /// Utilized capacity
    pub utilized_capacity: f64,
    /// Utilization efficiency
    pub utilization_efficiency: f64,
    /// Flow distribution entropy
    pub flow_entropy: f64,
}
