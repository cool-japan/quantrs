//! Feature Extraction for Meta-Learning Optimization
//!
//! This module contains all feature extraction types and implementations used
//! by the meta-learning optimization system.

use std::collections::HashMap;
use std::time::Instant;
use crate::applications::ApplicationResult;
use crate::ising::IsingModel;
use super::config::{FeatureExtractionConfig, FeatureSelectionMethod, DimensionalityReduction, FeatureNormalization};

/// Problem feature representation
#[derive(Debug, Clone)]
pub struct ProblemFeatures {
    /// Problem size
    pub size: usize,
    /// Problem density
    pub density: f64,
    /// Graph-based features
    pub graph_features: GraphFeatures,
    /// Statistical features
    pub statistical_features: StatisticalFeatures,
    /// Spectral features
    pub spectral_features: SpectralFeatures,
    /// Domain-specific features
    pub domain_features: HashMap<String, f64>,
}

/// Graph-based features
#[derive(Debug, Clone)]
pub struct GraphFeatures {
    /// Number of vertices
    pub num_vertices: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Average degree
    pub avg_degree: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Path length statistics
    pub path_length_stats: PathLengthStats,
    /// Centrality measures
    pub centrality_measures: CentralityMeasures,
}

/// Path length statistics
#[derive(Debug, Clone)]
pub struct PathLengthStats {
    /// Average shortest path length
    pub avg_shortest_path: f64,
    /// Diameter
    pub diameter: usize,
    /// Radius
    pub radius: usize,
    /// Eccentricity distribution
    pub eccentricity_stats: DistributionStats,
}

/// Centrality measures
#[derive(Debug, Clone)]
pub struct CentralityMeasures {
    /// Degree centrality stats
    pub degree_centrality: DistributionStats,
    /// Betweenness centrality stats
    pub betweenness_centrality: DistributionStats,
    /// Closeness centrality stats
    pub closeness_centrality: DistributionStats,
    /// Eigenvector centrality stats
    pub eigenvector_centrality: DistributionStats,
}

/// Distribution statistics
#[derive(Debug, Clone)]
pub struct DistributionStats {
    /// Mean
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum
    pub min: f64,
    /// Maximum
    pub max: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
}

/// Statistical features
#[derive(Debug, Clone)]
pub struct StatisticalFeatures {
    /// Bias statistics
    pub bias_stats: DistributionStats,
    /// Coupling statistics
    pub coupling_stats: DistributionStats,
    /// Energy landscape features
    pub energy_landscape: EnergyLandscapeFeatures,
    /// Correlation features
    pub correlation_features: CorrelationFeatures,
}

/// Energy landscape features
#[derive(Debug, Clone)]
pub struct EnergyLandscapeFeatures {
    /// Number of local minima estimate
    pub local_minima_estimate: usize,
    /// Energy barrier estimates
    pub energy_barriers: Vec<f64>,
    /// Landscape ruggedness
    pub ruggedness: f64,
    /// Basin size distribution
    pub basin_sizes: DistributionStats,
}

/// Correlation features
#[derive(Debug, Clone)]
pub struct CorrelationFeatures {
    /// Autocorrelation function
    pub autocorrelation: Vec<f64>,
    /// Cross-correlation features
    pub cross_correlation: HashMap<String, f64>,
    /// Mutual information
    pub mutual_information: f64,
}

/// Spectral features
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Eigenvalue statistics
    pub eigenvalue_stats: DistributionStats,
    /// Spectral gap
    pub spectral_gap: f64,
    /// Spectral radius
    pub spectral_radius: f64,
    /// Trace
    pub trace: f64,
    /// Condition number
    pub condition_number: f64,
}

/// Feature extraction system
#[derive(Debug)]
pub struct FeatureExtractor {
    /// Configuration
    pub config: FeatureExtractionConfig,
    /// Feature transformers
    pub transformers: Vec<FeatureTransformer>,
    /// Feature selectors
    pub selectors: Vec<FeatureSelector>,
    /// Dimensionality reducers
    pub reducers: Vec<DimensionalityReducer>,
}

/// Feature transformer
#[derive(Debug)]
pub struct FeatureTransformer {
    /// Transformer type
    pub transformer_type: TransformerType,
    /// Parameters
    pub parameters: HashMap<String, f64>,
    /// Fitted state
    pub is_fitted: bool,
}

/// Transformer types
#[derive(Debug, Clone, PartialEq)]
pub enum TransformerType {
    /// Polynomial features
    Polynomial,
    /// Interaction features
    Interaction,
    /// Logarithmic transform
    Logarithmic,
    /// Box-Cox transform
    BoxCox,
    /// Custom transform
    Custom(String),
}

/// Feature selector
#[derive(Debug)]
pub struct FeatureSelector {
    /// Selection method
    pub method: FeatureSelectionMethod,
    /// Selected features
    pub selected_features: Vec<usize>,
    /// Feature importance scores
    pub importance_scores: Vec<f64>,
}

/// Dimensionality reducer
#[derive(Debug)]
pub struct DimensionalityReducer {
    /// Reduction method
    pub method: DimensionalityReduction,
    /// Target dimensions
    pub target_dims: usize,
    /// Transformation matrix
    pub transformation_matrix: Option<Vec<Vec<f64>>>,
    /// Explained variance
    pub explained_variance: Vec<f64>,
}

impl FeatureExtractor {
    pub fn new(config: FeatureExtractionConfig) -> Self {
        Self {
            config,
            transformers: Vec::new(),
            selectors: Vec::new(),
            reducers: Vec::new(),
        }
    }
    
    pub fn extract_features(&mut self, problem: &IsingModel) -> ApplicationResult<ProblemFeatures> {
        let graph_features = if self.config.enable_graph_features {
            self.extract_graph_features(problem)?
        } else {
            GraphFeatures::default()
        };
        
        let statistical_features = if self.config.enable_statistical_features {
            self.extract_statistical_features(problem)?
        } else {
            StatisticalFeatures::default()
        };
        
        let spectral_features = if self.config.enable_spectral_features {
            self.extract_spectral_features(problem)?
        } else {
            SpectralFeatures::default()
        };
        
        Ok(ProblemFeatures {
            size: problem.num_qubits,
            density: self.calculate_density(problem),
            graph_features,
            statistical_features,
            spectral_features,
            domain_features: HashMap::new(),
        })
    }
    
    fn extract_graph_features(&self, problem: &IsingModel) -> ApplicationResult<GraphFeatures> {
        let num_vertices = problem.num_qubits;
        let mut num_edges = 0;
        
        // Count edges (non-zero couplings)
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                if problem.get_coupling(i, j).unwrap_or(0.0).abs() > 1e-10 {
                    num_edges += 1;
                }
            }
        }
        
        let avg_degree = if num_vertices > 0 {
            2.0 * num_edges as f64 / num_vertices as f64
        } else {
            0.0
        };
        
        Ok(GraphFeatures {
            num_vertices,
            num_edges,
            avg_degree,
            clustering_coefficient: 0.1, // Simplified
            path_length_stats: PathLengthStats {
                avg_shortest_path: avg_degree.ln().max(1.0),
                diameter: num_vertices / 2,
                radius: num_vertices / 4,
                eccentricity_stats: DistributionStats::default(),
            },
            centrality_measures: CentralityMeasures {
                degree_centrality: DistributionStats::default(),
                betweenness_centrality: DistributionStats::default(),
                closeness_centrality: DistributionStats::default(),
                eigenvector_centrality: DistributionStats::default(),
            },
        })
    }
    
    fn extract_statistical_features(&self, problem: &IsingModel) -> ApplicationResult<StatisticalFeatures> {
        let mut bias_values = Vec::new();
        let mut coupling_values = Vec::new();
        
        // Collect bias values
        for i in 0..problem.num_qubits {
            bias_values.push(problem.get_bias(i).unwrap_or(0.0));
        }
        
        // Collect coupling values
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                let coupling = problem.get_coupling(i, j).unwrap_or(0.0);
                if coupling.abs() > 1e-10 {
                    coupling_values.push(coupling);
                }
            }
        }
        
        Ok(StatisticalFeatures {
            bias_stats: self.calculate_distribution_stats(&bias_values),
            coupling_stats: self.calculate_distribution_stats(&coupling_values),
            energy_landscape: EnergyLandscapeFeatures {
                local_minima_estimate: (problem.num_qubits as f64).sqrt() as usize,
                energy_barriers: vec![1.0, 2.0, 3.0],
                ruggedness: 0.5,
                basin_sizes: DistributionStats::default(),
            },
            correlation_features: CorrelationFeatures {
                autocorrelation: vec![1.0, 0.8, 0.6, 0.4, 0.2],
                cross_correlation: HashMap::new(),
                mutual_information: 0.3,
            },
        })
    }
    
    fn extract_spectral_features(&self, problem: &IsingModel) -> ApplicationResult<SpectralFeatures> {
        // Simplified spectral analysis
        let n = problem.num_qubits as f64;
        let spectral_gap_estimate = 1.0 / n.sqrt();
        
        Ok(SpectralFeatures {
            eigenvalue_stats: DistributionStats {
                mean: 0.0,
                std_dev: 1.0,
                min: -n,
                max: n,
                skewness: 0.0,
                kurtosis: 3.0,
            },
            spectral_gap: spectral_gap_estimate,
            spectral_radius: n,
            trace: 0.0,
            condition_number: n,
        })
    }
    
    fn calculate_density(&self, problem: &IsingModel) -> f64 {
        let mut num_edges = 0;
        let max_edges = problem.num_qubits * (problem.num_qubits - 1) / 2;
        
        for i in 0..problem.num_qubits {
            for j in (i + 1)..problem.num_qubits {
                if problem.get_coupling(i, j).unwrap_or(0.0).abs() > 1e-10 {
                    num_edges += 1;
                }
            }
        }
        
        if max_edges > 0 {
            num_edges as f64 / max_edges as f64
        } else {
            0.0
        }
    }
    
    fn calculate_distribution_stats(&self, values: &[f64]) -> DistributionStats {
        if values.is_empty() {
            return DistributionStats::default();
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        DistributionStats {
            mean,
            std_dev,
            min,
            max,
            skewness: 0.0, // Simplified
            kurtosis: 3.0, // Simplified
        }
    }
}

impl Default for DistributionStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
            min: 0.0,
            max: 1.0,
            skewness: 0.0,
            kurtosis: 3.0,
        }
    }
}

impl Default for GraphFeatures {
    fn default() -> Self {
        Self {
            num_vertices: 0,
            num_edges: 0,
            avg_degree: 0.0,
            clustering_coefficient: 0.0,
            path_length_stats: PathLengthStats {
                avg_shortest_path: 0.0,
                diameter: 0,
                radius: 0,
                eccentricity_stats: DistributionStats::default(),
            },
            centrality_measures: CentralityMeasures {
                degree_centrality: DistributionStats::default(),
                betweenness_centrality: DistributionStats::default(),
                closeness_centrality: DistributionStats::default(),
                eigenvector_centrality: DistributionStats::default(),
            },
        }
    }
}

impl Default for StatisticalFeatures {
    fn default() -> Self {
        Self {
            bias_stats: DistributionStats::default(),
            coupling_stats: DistributionStats::default(),
            energy_landscape: EnergyLandscapeFeatures {
                local_minima_estimate: 0,
                energy_barriers: Vec::new(),
                ruggedness: 0.0,
                basin_sizes: DistributionStats::default(),
            },
            correlation_features: CorrelationFeatures {
                autocorrelation: Vec::new(),
                cross_correlation: HashMap::new(),
                mutual_information: 0.0,
            },
        }
    }
}

impl Default for SpectralFeatures {
    fn default() -> Self {
        Self {
            eigenvalue_stats: DistributionStats::default(),
            spectral_gap: 0.0,
            spectral_radius: 0.0,
            trace: 0.0,
            condition_number: 1.0,
        }
    }
}