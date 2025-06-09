//! Quantum Recommender Systems
//!
//! This module implements quantum-enhanced recommender systems using quantum
//! computing principles for improved recommendations, including collaborative
//! filtering, content-based filtering, and hybrid approaches.

use crate::error::{MLError, Result};
use crate::qnn::{QNNLayerType, QuantumNeuralNetwork};
use crate::optimization::OptimizationMethod;
use ndarray::{Array1, Array2, Array3, Axis, s};
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

/// Quantum recommender system configuration
#[derive(Debug, Clone)]
pub struct QuantumRecommenderConfig {
    /// Number of qubits for quantum processing
    pub num_qubits: usize,
    
    /// Recommendation algorithm type
    pub algorithm: RecommendationAlgorithm,
    
    /// Number of latent factors
    pub num_factors: usize,
    
    /// Regularization strength
    pub regularization: f64,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Quantum enhancement level
    pub quantum_enhancement: QuantumEnhancementLevel,
    
    /// Similarity measure
    pub similarity_measure: SimilarityMeasure,
}

/// Recommendation algorithm types
#[derive(Debug, Clone)]
pub enum RecommendationAlgorithm {
    /// Quantum collaborative filtering
    QuantumCollaborativeFiltering {
        neighborhood_size: usize,
        min_common_items: usize,
    },
    
    /// Quantum matrix factorization
    QuantumMatrixFactorization {
        optimization_method: OptimizationMethod,
        num_iterations: usize,
    },
    
    /// Quantum content-based filtering
    QuantumContentBased {
        feature_extraction: FeatureExtractionMethod,
        profile_learning: ProfileLearningMethod,
    },
    
    /// Hybrid quantum recommender
    HybridQuantum {
        cf_weight: f64,
        cb_weight: f64,
        knowledge_weight: f64,
    },
    
    /// Quantum neural collaborative filtering
    QuantumNeuralCF {
        embedding_dim: usize,
        hidden_layers: Vec<usize>,
    },
    
    /// Quantum graph-based recommendations
    QuantumGraphRecommender {
        walk_length: usize,
        num_walks: usize,
        teleportation_prob: f64,
    },
}

/// Feature extraction methods for content-based filtering
#[derive(Debug, Clone)]
pub enum FeatureExtractionMethod {
    /// TF-IDF with quantum enhancement
    QuantumTFIDF,
    
    /// Quantum embeddings
    QuantumEmbeddings {
        embedding_dim: usize,
    },
    
    /// Quantum autoencoders
    QuantumAutoencoder {
        latent_dim: usize,
    },
    
    /// Deep quantum features
    DeepQuantumFeatures {
        layer_dims: Vec<usize>,
    },
}

/// Profile learning methods
#[derive(Debug, Clone)]
pub enum ProfileLearningMethod {
    /// Weighted average
    WeightedAverage,
    
    /// Quantum superposition
    QuantumSuperposition,
    
    /// Adaptive quantum learning
    AdaptiveQuantum {
        learning_rate: f64,
    },
}

/// Quantum enhancement levels
#[derive(Debug, Clone)]
pub enum QuantumEnhancementLevel {
    /// Minimal quantum processing
    Low,
    
    /// Balanced quantum-classical
    Medium,
    
    /// Maximum quantum advantage
    High,
    
    /// Custom quantum configuration
    Custom {
        entanglement_strength: f64,
        coherence_time: f64,
        circuit_depth: usize,
    },
}

/// Similarity measures
#[derive(Debug, Clone)]
pub enum SimilarityMeasure {
    /// Cosine similarity
    Cosine,
    
    /// Pearson correlation
    Pearson,
    
    /// Quantum state fidelity
    QuantumFidelity,
    
    /// Entanglement-based similarity
    EntanglementSimilarity,
    
    /// Hybrid similarity
    Hybrid {
        classical_weight: f64,
        quantum_weight: f64,
    },
}

/// Main quantum recommender system
#[derive(Debug, Clone)]
pub struct QuantumRecommender {
    /// Configuration
    config: QuantumRecommenderConfig,
    
    /// User-item interaction matrix
    interaction_matrix: InteractionMatrix,
    
    /// Quantum processor
    quantum_processor: QuantumProcessor,
    
    /// Recommendation engine
    engine: Box<dyn RecommendationEngine>,
    
    /// User profiles
    user_profiles: HashMap<usize, UserProfile>,
    
    /// Item features
    item_features: HashMap<usize, ItemFeatures>,
    
    /// Model parameters
    parameters: ModelParameters,
    
    /// Performance metrics
    metrics: RecommenderMetrics,
}

/// Interaction matrix for user-item data
#[derive(Debug, Clone)]
pub struct InteractionMatrix {
    /// Sparse user-item ratings
    ratings: HashMap<(usize, usize), f64>,
    
    /// User indices
    user_ids: HashSet<usize>,
    
    /// Item indices
    item_ids: HashSet<usize>,
    
    /// Implicit feedback
    implicit_feedback: Option<HashMap<(usize, usize), ImplicitFeedback>>,
    
    /// Temporal information
    timestamps: Option<HashMap<(usize, usize), f64>>,
}

/// Implicit feedback data
#[derive(Debug, Clone)]
pub struct ImplicitFeedback {
    /// View count
    pub views: usize,
    
    /// Interaction duration
    pub duration: f64,
    
    /// Click-through rate
    pub ctr: f64,
    
    /// Conversion indicator
    pub converted: bool,
}

/// Quantum processor for recommendation computations
#[derive(Debug, Clone)]
pub struct QuantumProcessor {
    /// Number of qubits
    num_qubits: usize,
    
    /// Quantum circuits for similarity computation
    similarity_circuits: Vec<Vec<f64>>,
    
    /// Quantum neural network for embeddings
    embedding_network: QuantumNeuralNetwork,
    
    /// Entanglement generator
    entanglement_generator: EntanglementGenerator,
}

/// Trait for recommendation engines
pub trait RecommendationEngine: std::fmt::Debug {
    /// Generate recommendations for a user
    fn recommend(&self, user_id: usize, n_items: usize, exclude_seen: bool) -> Result<Vec<Recommendation>>;
    
    /// Update model with new interaction
    fn update(&mut self, user_id: usize, item_id: usize, rating: f64) -> Result<()>;
    
    /// Compute similarity between users or items
    fn compute_similarity(&self, id1: usize, id2: usize, similarity_type: SimilarityType) -> Result<f64>;
    
    /// Get model parameters
    fn parameters(&self) -> &Array1<f64>;
    
    /// Clone the engine
    fn clone_box(&self) -> Box<dyn RecommendationEngine>;
}

impl Clone for Box<dyn RecommendationEngine> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Similarity computation type
#[derive(Debug, Clone)]
pub enum SimilarityType {
    UserToUser,
    ItemToItem,
    UserToItem,
}

/// Recommendation result
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// Item ID
    pub item_id: usize,
    
    /// Predicted score
    pub score: f64,
    
    /// Confidence interval
    pub confidence: (f64, f64),
    
    /// Explanation
    pub explanation: Option<RecommendationExplanation>,
    
    /// Quantum contribution
    pub quantum_contribution: f64,
}

/// Recommendation explanation
#[derive(Debug, Clone)]
pub struct RecommendationExplanation {
    /// Similar users who liked this item
    pub similar_users: Vec<(usize, f64)>,
    
    /// Similar items to user's history
    pub similar_items: Vec<(usize, f64)>,
    
    /// Feature-based reasons
    pub feature_reasons: Vec<String>,
    
    /// Quantum state information
    pub quantum_state: Option<QuantumStateInfo>,
}

/// Quantum state information for explanations
#[derive(Debug, Clone)]
pub struct QuantumStateInfo {
    /// Entanglement measure
    pub entanglement: f64,
    
    /// Superposition weights
    pub superposition_weights: Vec<f64>,
    
    /// Quantum phase
    pub phase: f64,
}

/// User profile
#[derive(Debug, Clone)]
pub struct UserProfile {
    /// User ID
    pub user_id: usize,
    
    /// Feature vector
    pub features: Array1<f64>,
    
    /// Preference history
    pub preferences: PreferenceHistory,
    
    /// Quantum state representation
    pub quantum_state: Array1<f64>,
    
    /// Profile metadata
    pub metadata: ProfileMetadata,
}

/// Preference history
#[derive(Debug, Clone)]
pub struct PreferenceHistory {
    /// Rated items with scores
    pub rated_items: Vec<(usize, f64)>,
    
    /// Item categories
    pub preferred_categories: HashMap<String, f64>,
    
    /// Temporal preferences
    pub temporal_patterns: TemporalPatterns,
    
    /// Interaction context
    pub contexts: Vec<InteractionContext>,
}

/// Temporal preference patterns
#[derive(Debug, Clone)]
pub struct TemporalPatterns {
    /// Time of day preferences
    pub hourly_distribution: Array1<f64>,
    
    /// Day of week preferences
    pub weekly_distribution: Array1<f64>,
    
    /// Seasonal preferences
    pub seasonal_factors: Array1<f64>,
    
    /// Trend indicators
    pub trends: Vec<TrendIndicator>,
}

/// Trend indicator
#[derive(Debug, Clone)]
pub struct TrendIndicator {
    /// Category or feature
    pub feature: String,
    
    /// Trend direction (-1 to 1)
    pub direction: f64,
    
    /// Trend strength
    pub strength: f64,
    
    /// Time window
    pub window: f64,
}

/// Interaction context
#[derive(Debug, Clone)]
pub struct InteractionContext {
    /// Device type
    pub device: String,
    
    /// Location (anonymized)
    pub location_cluster: usize,
    
    /// Session length
    pub session_duration: f64,
    
    /// Previous actions
    pub action_sequence: Vec<String>,
}

/// Profile metadata
#[derive(Debug, Clone)]
pub struct ProfileMetadata {
    /// Profile creation time
    pub created_at: f64,
    
    /// Last update time
    pub updated_at: f64,
    
    /// Number of interactions
    pub num_interactions: usize,
    
    /// Profile completeness
    pub completeness: f64,
}

/// Item features
#[derive(Debug, Clone)]
pub struct ItemFeatures {
    /// Item ID
    pub item_id: usize,
    
    /// Feature vector
    pub features: Array1<f64>,
    
    /// Categories
    pub categories: Vec<String>,
    
    /// Attributes
    pub attributes: HashMap<String, AttributeValue>,
    
    /// Quantum representation
    pub quantum_features: Array1<f64>,
}

/// Attribute value types
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Numeric(f64),
    Categorical(String),
    Binary(bool),
    Vector(Vec<f64>),
}

/// Model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters {
    /// User embeddings
    pub user_embeddings: Array2<f64>,
    
    /// Item embeddings
    pub item_embeddings: Array2<f64>,
    
    /// Quantum circuit parameters
    pub quantum_params: Vec<f64>,
    
    /// Bias terms
    pub user_bias: Array1<f64>,
    pub item_bias: Array1<f64>,
    pub global_bias: f64,
}

/// Entanglement generator for quantum similarity
#[derive(Debug, Clone)]
pub struct EntanglementGenerator {
    /// Entanglement patterns
    patterns: Vec<EntanglementPattern>,
    
    /// Circuit parameters
    circuit_params: Vec<f64>,
}

/// Entanglement pattern
#[derive(Debug, Clone)]
pub struct EntanglementPattern {
    /// Qubit pairs
    pub qubit_pairs: Vec<(usize, usize)>,
    
    /// Entanglement strength
    pub strength: f64,
    
    /// Pattern type
    pub pattern_type: PatternType,
}

/// Entanglement pattern types
#[derive(Debug, Clone)]
pub enum PatternType {
    /// Bell states
    Bell,
    
    /// GHZ states
    GHZ,
    
    /// Cluster states
    Cluster,
    
    /// Custom pattern
    Custom(Vec<f64>),
}

/// Recommender performance metrics
#[derive(Debug, Clone)]
pub struct RecommenderMetrics {
    /// Recommendation accuracy metrics
    pub accuracy_metrics: AccuracyMetrics,
    
    /// Diversity metrics
    pub diversity_metrics: DiversityMetrics,
    
    /// Coverage metrics
    pub coverage_metrics: CoverageMetrics,
    
    /// Quantum metrics
    pub quantum_metrics: QuantumMetrics,
}

/// Accuracy metrics
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    /// Mean Absolute Error
    pub mae: f64,
    
    /// Root Mean Square Error
    pub rmse: f64,
    
    /// Precision at K
    pub precision_at_k: HashMap<usize, f64>,
    
    /// Recall at K
    pub recall_at_k: HashMap<usize, f64>,
    
    /// NDCG at K
    pub ndcg_at_k: HashMap<usize, f64>,
}

/// Diversity metrics
#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    /// Intra-list diversity
    pub intra_list_diversity: f64,
    
    /// Inter-list diversity
    pub inter_list_diversity: f64,
    
    /// Category coverage
    pub category_coverage: f64,
    
    /// Novelty score
    pub novelty: f64,
}

/// Coverage metrics
#[derive(Debug, Clone)]
pub struct CoverageMetrics {
    /// Item coverage
    pub item_coverage: f64,
    
    /// User coverage
    pub user_coverage: f64,
    
    /// Cold start performance
    pub cold_start_performance: f64,
}

/// Quantum-specific metrics
#[derive(Debug, Clone)]
pub struct QuantumMetrics {
    /// Quantum advantage ratio
    pub quantum_advantage: f64,
    
    /// Entanglement utilization
    pub entanglement_utilization: f64,
    
    /// Coherence preservation
    pub coherence_preservation: f64,
    
    /// Circuit efficiency
    pub circuit_efficiency: f64,
}

impl QuantumRecommenderConfig {
    /// Create default configuration
    pub fn default() -> Self {
        Self {
            num_qubits: 10,
            algorithm: RecommendationAlgorithm::QuantumMatrixFactorization {
                optimization_method: OptimizationMethod::Adam,
                num_iterations: 100,
            },
            num_factors: 50,
            regularization: 0.01,
            learning_rate: 0.001,
            quantum_enhancement: QuantumEnhancementLevel::Medium,
            similarity_measure: SimilarityMeasure::QuantumFidelity,
        }
    }
    
    /// Configuration for collaborative filtering
    pub fn collaborative_filtering() -> Self {
        Self {
            num_qubits: 12,
            algorithm: RecommendationAlgorithm::QuantumCollaborativeFiltering {
                neighborhood_size: 50,
                min_common_items: 3,
            },
            num_factors: 100,
            regularization: 0.001,
            learning_rate: 0.01,
            quantum_enhancement: QuantumEnhancementLevel::High,
            similarity_measure: SimilarityMeasure::EntanglementSimilarity,
        }
    }
    
    /// Configuration for content-based filtering
    pub fn content_based() -> Self {
        Self {
            num_qubits: 10,
            algorithm: RecommendationAlgorithm::QuantumContentBased {
                feature_extraction: FeatureExtractionMethod::QuantumEmbeddings {
                    embedding_dim: 128,
                },
                profile_learning: ProfileLearningMethod::QuantumSuperposition,
            },
            num_factors: 64,
            regularization: 0.005,
            learning_rate: 0.005,
            quantum_enhancement: QuantumEnhancementLevel::Medium,
            similarity_measure: SimilarityMeasure::Cosine,
        }
    }
    
    /// Configuration for hybrid recommender
    pub fn hybrid() -> Self {
        Self {
            num_qubits: 14,
            algorithm: RecommendationAlgorithm::HybridQuantum {
                cf_weight: 0.6,
                cb_weight: 0.3,
                knowledge_weight: 0.1,
            },
            num_factors: 80,
            regularization: 0.01,
            learning_rate: 0.001,
            quantum_enhancement: QuantumEnhancementLevel::High,
            similarity_measure: SimilarityMeasure::Hybrid {
                classical_weight: 0.4,
                quantum_weight: 0.6,
            },
        }
    }
}

impl QuantumRecommender {
    /// Create new quantum recommender system
    pub fn new(config: QuantumRecommenderConfig) -> Result<Self> {
        // Initialize interaction matrix
        let interaction_matrix = InteractionMatrix::new();
        
        // Create quantum processor
        let quantum_processor = QuantumProcessor::new(config.num_qubits)?;
        
        // Create recommendation engine based on algorithm
        let engine: Box<dyn RecommendationEngine> = match &config.algorithm {
            RecommendationAlgorithm::QuantumCollaborativeFiltering { .. } => {
                Box::new(QuantumCFEngine::new(&config)?)
            }
            RecommendationAlgorithm::QuantumMatrixFactorization { .. } => {
                Box::new(QuantumMFEngine::new(&config)?)
            }
            RecommendationAlgorithm::QuantumContentBased { .. } => {
                Box::new(QuantumCBEngine::new(&config)?)
            }
            RecommendationAlgorithm::HybridQuantum { .. } => {
                Box::new(HybridQuantumEngine::new(&config)?)
            }
            RecommendationAlgorithm::QuantumNeuralCF { .. } => {
                Box::new(QuantumNCFEngine::new(&config)?)
            }
            RecommendationAlgorithm::QuantumGraphRecommender { .. } => {
                Box::new(QuantumGraphEngine::new(&config)?)
            }
        };
        
        // Initialize model parameters
        let parameters = ModelParameters::new(1000, 1000, config.num_factors);
        
        Ok(Self {
            config,
            interaction_matrix,
            quantum_processor,
            engine,
            user_profiles: HashMap::new(),
            item_features: HashMap::new(),
            parameters,
            metrics: RecommenderMetrics::new(),
        })
    }
    
    /// Add user-item interaction
    pub fn add_interaction(
        &mut self,
        user_id: usize,
        item_id: usize,
        rating: f64,
        context: Option<InteractionContext>,
    ) -> Result<()> {
        // Update interaction matrix
        self.interaction_matrix.add_rating(user_id, item_id, rating);
        
        // Update user profile
        if let Some(profile) = self.user_profiles.get_mut(&user_id) {
            profile.update_with_interaction(item_id, rating, context);
        } else {
            let mut profile = UserProfile::new(user_id);
            profile.update_with_interaction(item_id, rating, context);
            self.user_profiles.insert(user_id, profile);
        }
        
        // Update recommendation engine
        self.engine.update(user_id, item_id, rating)?;
        
        Ok(())
    }
    
    /// Get recommendations for a user
    pub fn recommend(
        &self,
        user_id: usize,
        n_items: usize,
        options: RecommendationOptions,
    ) -> Result<Vec<Recommendation>> {
        let mut recommendations = self.engine.recommend(user_id, n_items, options.exclude_seen)?;
        
        // Apply post-processing
        if options.diversify {
            recommendations = self.diversify_recommendations(recommendations, options.diversity_weight)?;
        }
        
        // Add explanations if requested
        if options.explain {
            for rec in &mut recommendations {
                rec.explanation = Some(self.generate_explanation(user_id, rec.item_id)?);
            }
        }
        
        // Apply business rules
        if let Some(rules) = options.business_rules {
            recommendations = self.apply_business_rules(recommendations, rules)?;
        }
        
        Ok(recommendations)
    }
    
    /// Train the recommender system
    pub fn train(
        &mut self,
        train_data: &[(usize, usize, f64)],
        val_data: Option<&[(usize, usize, f64)]>,
        epochs: usize,
    ) -> Result<TrainingHistory> {
        let mut history = TrainingHistory::new();
        
        for epoch in 0..epochs {
            // Training phase
            let mut train_loss = 0.0;
            
            for &(user_id, item_id, rating) in train_data {
                // Forward pass
                let prediction = self.predict(user_id, item_id)?;
                let loss = (prediction - rating).powi(2);
                train_loss += loss;
                
                // Update model
                self.update_model(user_id, item_id, rating, prediction)?;
            }
            
            train_loss /= train_data.len() as f64;
            
            // Validation phase
            let val_metrics = if let Some(val) = val_data {
                self.evaluate(val)?
            } else {
                EvaluationMetrics::default()
            };
            
            history.add_epoch(epoch, train_loss, val_metrics);
            
            // Early stopping check
            if history.should_stop_early() {
                break;
            }
        }
        
        Ok(history)
    }
    
    /// Predict rating for user-item pair
    pub fn predict(&self, user_id: usize, item_id: usize) -> Result<f64> {
        // Get embeddings
        let user_embedding = self.parameters.get_user_embedding(user_id)?;
        let item_embedding = self.parameters.get_item_embedding(item_id)?;
        
        // Quantum-enhanced prediction
        let quantum_similarity = self.quantum_processor.compute_similarity(
            &user_embedding,
            &item_embedding,
            &self.config.similarity_measure,
        )?;
        
        // Combine with biases
        let prediction = self.parameters.global_bias
            + self.parameters.user_bias[user_id]
            + self.parameters.item_bias[item_id]
            + quantum_similarity;
        
        Ok(prediction.max(1.0).min(5.0)) // Clip to rating range
    }
    
    /// Update model parameters
    fn update_model(
        &mut self,
        user_id: usize,
        item_id: usize,
        true_rating: f64,
        predicted_rating: f64,
    ) -> Result<()> {
        let error = true_rating - predicted_rating;
        let lr = self.config.learning_rate;
        let reg = self.config.regularization;
        
        // Update biases
        self.parameters.user_bias[user_id] += lr * (error - reg * self.parameters.user_bias[user_id]);
        self.parameters.item_bias[item_id] += lr * (error - reg * self.parameters.item_bias[item_id]);
        
        // Update embeddings with quantum gradient
        let quantum_gradient = self.quantum_processor.compute_gradient(
            user_id,
            item_id,
            error,
        )?;
        
        self.parameters.update_embeddings(user_id, item_id, &quantum_gradient, lr, reg)?;
        
        Ok(())
    }
    
    /// Diversify recommendations
    fn diversify_recommendations(
        &self,
        mut recommendations: Vec<Recommendation>,
        diversity_weight: f64,
    ) -> Result<Vec<Recommendation>> {
        let mut diversified = Vec::new();
        let mut selected_items = HashSet::new();
        
        while !recommendations.is_empty() && diversified.len() < recommendations.len() {
            let mut best_score = f64::NEG_INFINITY;
            let mut best_idx = 0;
            
            for (idx, rec) in recommendations.iter().enumerate() {
                let relevance_score = rec.score;
                let diversity_score = self.compute_diversity_score(rec.item_id, &selected_items)?;
                
                let combined_score = (1.0 - diversity_weight) * relevance_score
                    + diversity_weight * diversity_score;
                
                if combined_score > best_score {
                    best_score = combined_score;
                    best_idx = idx;
                }
            }
            
            let selected = recommendations.remove(best_idx);
            selected_items.insert(selected.item_id);
            diversified.push(selected);
        }
        
        Ok(diversified)
    }
    
    /// Compute diversity score
    fn compute_diversity_score(
        &self,
        item_id: usize,
        selected_items: &HashSet<usize>,
    ) -> Result<f64> {
        if selected_items.is_empty() {
            return Ok(1.0);
        }
        
        let mut min_similarity: f64 = 1.0;
        
        for &selected_id in selected_items {
            let similarity = self.engine.compute_similarity(
                item_id,
                selected_id,
                SimilarityType::ItemToItem,
            )?;
            min_similarity = min_similarity.min(similarity);
        }
        
        Ok(1.0 - min_similarity)
    }
    
    /// Generate explanation for recommendation
    fn generate_explanation(
        &self,
        user_id: usize,
        item_id: usize,
    ) -> Result<RecommendationExplanation> {
        // Find similar users who liked this item
        let similar_users = self.find_similar_users_for_item(user_id, item_id, 5)?;
        
        // Find similar items to user's history
        let similar_items = self.find_similar_items_to_history(user_id, item_id, 5)?;
        
        // Extract feature-based reasons
        let feature_reasons = self.extract_feature_reasons(user_id, item_id)?;
        
        // Get quantum state information
        let quantum_state = Some(self.quantum_processor.get_state_info(user_id, item_id)?);
        
        Ok(RecommendationExplanation {
            similar_users,
            similar_items,
            feature_reasons,
            quantum_state,
        })
    }
    
    /// Find similar users who liked an item
    fn find_similar_users_for_item(
        &self,
        user_id: usize,
        item_id: usize,
        n: usize,
    ) -> Result<Vec<(usize, f64)>> {
        let mut similar_users = Vec::new();
        
        // Get users who rated the item
        let item_users = self.interaction_matrix.get_item_users(item_id)?;
        
        for &other_user in &item_users {
            if other_user != user_id {
                let similarity = self.engine.compute_similarity(
                    user_id,
                    other_user,
                    SimilarityType::UserToUser,
                )?;
                similar_users.push((other_user, similarity));
            }
        }
        
        // Sort by similarity and take top n
        similar_users.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similar_users.truncate(n);
        
        Ok(similar_users)
    }
    
    /// Find similar items to user's history
    fn find_similar_items_to_history(
        &self,
        user_id: usize,
        target_item: usize,
        n: usize,
    ) -> Result<Vec<(usize, f64)>> {
        let mut similar_items = Vec::new();
        
        // Get user's rated items
        if let Some(profile) = self.user_profiles.get(&user_id) {
            for &(item_id, _) in &profile.preferences.rated_items {
                if item_id != target_item {
                    let similarity = self.engine.compute_similarity(
                        target_item,
                        item_id,
                        SimilarityType::ItemToItem,
                    )?;
                    similar_items.push((item_id, similarity));
                }
            }
        }
        
        // Sort by similarity and take top n
        similar_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similar_items.truncate(n);
        
        Ok(similar_items)
    }
    
    /// Extract feature-based reasons
    fn extract_feature_reasons(
        &self,
        user_id: usize,
        item_id: usize,
    ) -> Result<Vec<String>> {
        let mut reasons = Vec::new();
        
        // Get user preferences and item features
        if let (Some(user_profile), Some(item_features)) = 
            (self.user_profiles.get(&user_id), self.item_features.get(&item_id))
        {
            // Category match
            for category in &item_features.categories {
                if let Some(&pref_score) = user_profile.preferences.preferred_categories.get(category) {
                    if pref_score > 0.7 {
                        reasons.push(format!("Matches your interest in {}", category));
                    }
                }
            }
            
            // Attribute match
            for (attr_name, attr_value) in &item_features.attributes {
                match attr_value {
                    AttributeValue::Categorical(val) => {
                        reasons.push(format!("Features {}: {}", attr_name, val));
                    }
                    AttributeValue::Numeric(val) => {
                        if *val > 0.8 {
                            reasons.push(format!("High {} score", attr_name));
                        }
                    }
                    _ => {}
                }
            }
        }
        
        Ok(reasons)
    }
    
    /// Apply business rules
    fn apply_business_rules(
        &self,
        recommendations: Vec<Recommendation>,
        rules: BusinessRules,
    ) -> Result<Vec<Recommendation>> {
        let mut filtered = recommendations;
        
        // Apply filters
        if let Some(categories) = rules.required_categories {
            filtered = self.filter_by_categories(filtered, categories)?;
        }
        
        if let Some(boost_new) = rules.boost_new_items {
            filtered = self.boost_new_items(filtered, boost_new)?;
        }
        
        if let Some(max_price) = rules.max_price {
            filtered = self.filter_by_price(filtered, max_price)?;
        }
        
        Ok(filtered)
    }
    
    /// Filter recommendations by categories
    fn filter_by_categories(
        &self,
        recommendations: Vec<Recommendation>,
        categories: HashSet<String>,
    ) -> Result<Vec<Recommendation>> {
        Ok(recommendations.into_iter()
            .filter(|rec| {
                if let Some(features) = self.item_features.get(&rec.item_id) {
                    features.categories.iter().any(|cat| categories.contains(cat))
                } else {
                    false
                }
            })
            .collect())
    }
    
    /// Boost new items
    fn boost_new_items(
        &self,
        mut recommendations: Vec<Recommendation>,
        boost_factor: f64,
    ) -> Result<Vec<Recommendation>> {
        for rec in &mut recommendations {
            if self.is_new_item(rec.item_id)? {
                rec.score *= boost_factor;
            }
        }
        
        // Re-sort by score
        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        Ok(recommendations)
    }
    
    /// Check if item is new
    fn is_new_item(&self, item_id: usize) -> Result<bool> {
        // Simplified: check if item has few ratings
        let num_ratings = self.interaction_matrix.get_item_rating_count(item_id)?;
        Ok(num_ratings < 10)
    }
    
    /// Filter by price
    fn filter_by_price(
        &self,
        recommendations: Vec<Recommendation>,
        max_price: f64,
    ) -> Result<Vec<Recommendation>> {
        Ok(recommendations.into_iter()
            .filter(|rec| {
                if let Some(features) = self.item_features.get(&rec.item_id) {
                    if let Some(AttributeValue::Numeric(price)) = features.attributes.get("price") {
                        *price <= max_price
                    } else {
                        true
                    }
                } else {
                    true
                }
            })
            .collect())
    }
    
    /// Evaluate on test data
    pub fn evaluate(&self, test_data: &[(usize, usize, f64)]) -> Result<EvaluationMetrics> {
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();
        
        for &(user_id, item_id, rating) in test_data {
            let prediction = self.predict(user_id, item_id)?;
            predictions.push(prediction);
            actuals.push(rating);
        }
        
        Ok(EvaluationMetrics::compute(&predictions, &actuals))
    }
    
    /// Get performance metrics
    pub fn metrics(&self) -> &RecommenderMetrics {
        &self.metrics
    }
}

impl InteractionMatrix {
    /// Create new interaction matrix
    pub fn new() -> Self {
        Self {
            ratings: HashMap::new(),
            user_ids: HashSet::new(),
            item_ids: HashSet::new(),
            implicit_feedback: None,
            timestamps: None,
        }
    }
    
    /// Add rating
    pub fn add_rating(&mut self, user_id: usize, item_id: usize, rating: f64) {
        self.ratings.insert((user_id, item_id), rating);
        self.user_ids.insert(user_id);
        self.item_ids.insert(item_id);
    }
    
    /// Get users who rated an item
    pub fn get_item_users(&self, item_id: usize) -> Result<Vec<usize>> {
        Ok(self.ratings.keys()
            .filter_map(|(user, item)| {
                if *item == item_id { Some(*user) } else { None }
            })
            .collect())
    }
    
    /// Get item rating count
    pub fn get_item_rating_count(&self, item_id: usize) -> Result<usize> {
        Ok(self.ratings.keys()
            .filter(|(_, item)| *item == item_id)
            .count())
    }
}

impl QuantumProcessor {
    /// Create new quantum processor
    pub fn new(num_qubits: usize) -> Result<Self> {
        // Create quantum embedding network
        let layers = vec![
            QNNLayerType::EncodingLayer { num_features: 128 },
            QNNLayerType::VariationalLayer { num_params: 64 },
            QNNLayerType::EntanglementLayer { connectivity: "circular".to_string() },
            QNNLayerType::MeasurementLayer { measurement_basis: "computational".to_string() },
        ];
        
        let embedding_network = QuantumNeuralNetwork::new(layers, num_qubits, 128, 64)?;
        
        // Create entanglement generator
        let entanglement_generator = EntanglementGenerator::new(num_qubits);
        
        Ok(Self {
            num_qubits,
            similarity_circuits: Vec::new(),
            embedding_network,
            entanglement_generator,
        })
    }
    
    /// Compute quantum similarity
    pub fn compute_similarity(
        &self,
        vec1: &Array1<f64>,
        vec2: &Array1<f64>,
        measure: &SimilarityMeasure,
    ) -> Result<f64> {
        match measure {
            SimilarityMeasure::Cosine => {
                let dot = vec1.dot(vec2);
                let norm1 = vec1.dot(vec1).sqrt();
                let norm2 = vec2.dot(vec2).sqrt();
                Ok(dot / (norm1 * norm2 + 1e-10))
            }
            SimilarityMeasure::QuantumFidelity => {
                // Encode vectors as quantum states
                let state1 = self.encode_as_quantum_state(vec1)?;
                let state2 = self.encode_as_quantum_state(vec2)?;
                
                // Compute fidelity
                let fidelity = state1.dot(&state2).abs();
                Ok(fidelity * fidelity)
            }
            SimilarityMeasure::EntanglementSimilarity => {
                // Create entangled state from both vectors
                let entangled = self.entanglement_generator.create_entangled_state(vec1, vec2)?;
                
                // Measure entanglement
                let entanglement = self.measure_entanglement(&entangled)?;
                Ok(entanglement)
            }
            _ => Ok(0.5), // Default similarity
        }
    }
    
    /// Encode vector as quantum state
    fn encode_as_quantum_state(&self, vec: &Array1<f64>) -> Result<Array1<f64>> {
        // Normalize vector
        let norm = vec.dot(vec).sqrt();
        let normalized = vec / (norm + 1e-10);
        
        // Pad or truncate to quantum dimension
        let quantum_dim = 2_usize.pow(self.num_qubits as u32);
        let mut quantum_state = Array1::zeros(quantum_dim);
        
        for i in 0..normalized.len().min(quantum_dim) {
            quantum_state[i] = normalized[i];
        }
        
        Ok(quantum_state)
    }
    
    /// Measure entanglement
    fn measure_entanglement(&self, state: &Array1<f64>) -> Result<f64> {
        // Simplified entanglement measure
        let entropy = -state.iter()
            .filter(|&&x| x.abs() > 1e-10)
            .map(|&x| {
                let p = x * x;
                p * p.ln()
            })
            .sum::<f64>();
        
        Ok((entropy / (self.num_qubits as f64).ln()).min(1.0))
    }
    
    /// Compute quantum gradient
    pub fn compute_gradient(
        &self,
        user_id: usize,
        item_id: usize,
        error: f64,
    ) -> Result<Array1<f64>> {
        // Simplified quantum gradient computation
        let gradient_dim = 64;
        let mut gradient = Array1::zeros(gradient_dim);
        
        for i in 0..gradient_dim {
            gradient[i] = error * (0.1 * (i as f64 * 0.1 + user_id as f64 * 0.01 + item_id as f64 * 0.001).sin());
        }
        
        Ok(gradient)
    }
    
    /// Get quantum state information
    pub fn get_state_info(&self, user_id: usize, item_id: usize) -> Result<QuantumStateInfo> {
        Ok(QuantumStateInfo {
            entanglement: 0.7 + 0.3 * (user_id as f64 * 0.1).sin(),
            superposition_weights: vec![0.5, 0.3, 0.2],
            phase: PI * (item_id as f64 * 0.01).sin(),
        })
    }
}

impl EntanglementGenerator {
    /// Create new entanglement generator
    pub fn new(num_qubits: usize) -> Self {
        let patterns = vec![
            EntanglementPattern {
                qubit_pairs: (0..num_qubits-1).map(|i| (i, i+1)).collect(),
                strength: 0.8,
                pattern_type: PatternType::Bell,
            },
            EntanglementPattern {
                qubit_pairs: vec![(0, num_qubits-1)],
                strength: 0.5,
                pattern_type: PatternType::GHZ,
            },
        ];
        
        Self {
            patterns,
            circuit_params: vec![0.0; num_qubits * 3],
        }
    }
    
    /// Create entangled state
    pub fn create_entangled_state(
        &self,
        vec1: &Array1<f64>,
        vec2: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let combined = Array1::from_iter(
            vec1.iter().chain(vec2.iter()).cloned()
        );
        
        // Apply entanglement transformation
        let mut entangled = combined.clone();
        
        for pattern in &self.patterns {
            for &(q1, q2) in &pattern.qubit_pairs {
                if q1 < entangled.len() && q2 < entangled.len() {
                    let v1 = entangled[q1];
                    let v2 = entangled[q2];
                    
                    entangled[q1] = v1 * pattern.strength.cos() - v2 * pattern.strength.sin();
                    entangled[q2] = v1 * pattern.strength.sin() + v2 * pattern.strength.cos();
                }
            }
        }
        
        // Normalize
        let norm = entangled.dot(&entangled).sqrt();
        Ok(entangled / (norm + 1e-10))
    }
}

impl UserProfile {
    /// Create new user profile
    pub fn new(user_id: usize) -> Self {
        Self {
            user_id,
            features: Array1::zeros(128),
            preferences: PreferenceHistory::new(),
            quantum_state: Array1::zeros(64),
            metadata: ProfileMetadata {
                created_at: 0.0,
                updated_at: 0.0,
                num_interactions: 0,
                completeness: 0.0,
            },
        }
    }
    
    /// Update profile with new interaction
    pub fn update_with_interaction(
        &mut self,
        item_id: usize,
        rating: f64,
        context: Option<InteractionContext>,
    ) {
        self.preferences.rated_items.push((item_id, rating));
        
        if let Some(ctx) = context {
            self.preferences.contexts.push(ctx);
        }
        
        self.metadata.num_interactions += 1;
        self.metadata.updated_at = self.metadata.num_interactions as f64; // Simplified timestamp
    }
}

impl PreferenceHistory {
    /// Create new preference history
    pub fn new() -> Self {
        Self {
            rated_items: Vec::new(),
            preferred_categories: HashMap::new(),
            temporal_patterns: TemporalPatterns {
                hourly_distribution: Array1::zeros(24),
                weekly_distribution: Array1::zeros(7),
                seasonal_factors: Array1::zeros(4),
                trends: Vec::new(),
            },
            contexts: Vec::new(),
        }
    }
}

impl ModelParameters {
    /// Create new model parameters
    pub fn new(num_users: usize, num_items: usize, num_factors: usize) -> Self {
        Self {
            user_embeddings: Array2::from_shape_fn((num_users, num_factors), |(_, _)| {
                0.01 * (fastrand::f64() - 0.5)
            }),
            item_embeddings: Array2::from_shape_fn((num_items, num_factors), |(_, _)| {
                0.01 * (fastrand::f64() - 0.5)
            }),
            quantum_params: vec![0.0; num_factors * 10],
            user_bias: Array1::zeros(num_users),
            item_bias: Array1::zeros(num_items),
            global_bias: 3.5, // Typical average rating
        }
    }
    
    /// Get user embedding
    pub fn get_user_embedding(&self, user_id: usize) -> Result<Array1<f64>> {
        if user_id < self.user_embeddings.nrows() {
            Ok(self.user_embeddings.row(user_id).to_owned())
        } else {
            Ok(Array1::zeros(self.user_embeddings.ncols()))
        }
    }
    
    /// Get item embedding
    pub fn get_item_embedding(&self, item_id: usize) -> Result<Array1<f64>> {
        if item_id < self.item_embeddings.nrows() {
            Ok(self.item_embeddings.row(item_id).to_owned())
        } else {
            Ok(Array1::zeros(self.item_embeddings.ncols()))
        }
    }
    
    /// Update embeddings
    pub fn update_embeddings(
        &mut self,
        user_id: usize,
        item_id: usize,
        gradient: &Array1<f64>,
        lr: f64,
        reg: f64,
    ) -> Result<()> {
        if user_id < self.user_embeddings.nrows() && item_id < self.item_embeddings.nrows() {
            let user_emb = self.user_embeddings.row(user_id).to_owned();
            let item_emb = self.item_embeddings.row(item_id).to_owned();
            
            // Update user embedding
            self.user_embeddings.row_mut(user_id)
                .zip_mut_with(&user_emb, |param, &old| {
                    *param = old + lr * (gradient[0] - reg * old);
                });
            
            // Update item embedding
            self.item_embeddings.row_mut(item_id)
                .zip_mut_with(&item_emb, |param, &old| {
                    *param = old + lr * (gradient[0] - reg * old);
                });
        }
        
        Ok(())
    }
}

impl RecommenderMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self {
            accuracy_metrics: AccuracyMetrics {
                mae: 0.0,
                rmse: 0.0,
                precision_at_k: HashMap::new(),
                recall_at_k: HashMap::new(),
                ndcg_at_k: HashMap::new(),
            },
            diversity_metrics: DiversityMetrics {
                intra_list_diversity: 0.0,
                inter_list_diversity: 0.0,
                category_coverage: 0.0,
                novelty: 0.0,
            },
            coverage_metrics: CoverageMetrics {
                item_coverage: 0.0,
                user_coverage: 0.0,
                cold_start_performance: 0.0,
            },
            quantum_metrics: QuantumMetrics {
                quantum_advantage: 1.0,
                entanglement_utilization: 0.0,
                coherence_preservation: 0.0,
                circuit_efficiency: 0.0,
            },
        }
    }
}

/// Recommendation options
#[derive(Debug, Clone)]
pub struct RecommendationOptions {
    /// Exclude already seen items
    pub exclude_seen: bool,
    
    /// Diversify recommendations
    pub diversify: bool,
    
    /// Diversity weight (0-1)
    pub diversity_weight: f64,
    
    /// Include explanations
    pub explain: bool,
    
    /// Business rules to apply
    pub business_rules: Option<BusinessRules>,
}

impl Default for RecommendationOptions {
    fn default() -> Self {
        Self {
            exclude_seen: true,
            diversify: false,
            diversity_weight: 0.3,
            explain: false,
            business_rules: None,
        }
    }
}

/// Business rules for recommendations
#[derive(Debug, Clone)]
pub struct BusinessRules {
    /// Required categories
    pub required_categories: Option<HashSet<String>>,
    
    /// Boost new items
    pub boost_new_items: Option<f64>,
    
    /// Maximum price filter
    pub max_price: Option<f64>,
}

/// Training history
#[derive(Debug, Clone)]
pub struct TrainingHistory {
    pub epochs: Vec<usize>,
    pub train_losses: Vec<f64>,
    pub val_metrics: Vec<EvaluationMetrics>,
}

impl TrainingHistory {
    fn new() -> Self {
        Self {
            epochs: Vec::new(),
            train_losses: Vec::new(),
            val_metrics: Vec::new(),
        }
    }
    
    fn add_epoch(&mut self, epoch: usize, train_loss: f64, val_metrics: EvaluationMetrics) {
        self.epochs.push(epoch);
        self.train_losses.push(train_loss);
        self.val_metrics.push(val_metrics);
    }
    
    fn should_stop_early(&self) -> bool {
        // Simple early stopping: check if val loss increased for 3 epochs
        if self.val_metrics.len() < 4 {
            return false;
        }
        
        let recent = &self.val_metrics[self.val_metrics.len()-3..];
        recent[0].rmse < recent[1].rmse && recent[1].rmse < recent[2].rmse
    }
}

/// Evaluation metrics
#[derive(Debug, Clone, Default)]
pub struct EvaluationMetrics {
    pub mae: f64,
    pub rmse: f64,
}

impl EvaluationMetrics {
    fn compute(predictions: &[f64], actuals: &[f64]) -> Self {
        let n = predictions.len() as f64;
        
        let mae = predictions.iter()
            .zip(actuals)
            .map(|(p, a)| (p - a).abs())
            .sum::<f64>() / n;
        
        let rmse = (predictions.iter()
            .zip(actuals)
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>() / n)
            .sqrt();
        
        Self { mae, rmse }
    }
}

// Recommendation engine implementations

/// Quantum collaborative filtering engine
#[derive(Debug, Clone)]
struct QuantumCFEngine {
    config: QuantumRecommenderConfig,
    similarity_cache: HashMap<(usize, usize), f64>,
    parameters: Array1<f64>,
}

impl QuantumCFEngine {
    fn new(config: &QuantumRecommenderConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            similarity_cache: HashMap::new(),
            parameters: Array1::zeros(100),
        })
    }
}

impl RecommendationEngine for QuantumCFEngine {
    fn recommend(&self, _user_id: usize, n_items: usize, _exclude_seen: bool) -> Result<Vec<Recommendation>> {
        // Simplified implementation
        let mut recommendations = Vec::new();
        
        for i in 0..n_items {
            recommendations.push(Recommendation {
                item_id: i,
                score: 4.0 - 0.1 * i as f64,
                confidence: (3.5, 4.5),
                explanation: None,
                quantum_contribution: 0.3,
            });
        }
        
        Ok(recommendations)
    }
    
    fn update(&mut self, _user_id: usize, _item_id: usize, _rating: f64) -> Result<()> {
        Ok(())
    }
    
    fn compute_similarity(&self, _id1: usize, _id2: usize, _similarity_type: SimilarityType) -> Result<f64> {
        Ok(0.8)
    }
    
    fn parameters(&self) -> &Array1<f64> {
        &self.parameters
    }
    
    fn clone_box(&self) -> Box<dyn RecommendationEngine> {
        Box::new(self.clone())
    }
}

/// Quantum matrix factorization engine
#[derive(Debug, Clone)]
struct QuantumMFEngine {
    config: QuantumRecommenderConfig,
    parameters: Array1<f64>,
}

impl QuantumMFEngine {
    fn new(config: &QuantumRecommenderConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            parameters: Array1::zeros(100),
        })
    }
}

impl RecommendationEngine for QuantumMFEngine {
    fn recommend(&self, _user_id: usize, n_items: usize, _exclude_seen: bool) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();
        
        for i in 0..n_items {
            recommendations.push(Recommendation {
                item_id: i * 2,
                score: 4.2 - 0.05 * i as f64,
                confidence: (3.8, 4.6),
                explanation: None,
                quantum_contribution: 0.4,
            });
        }
        
        Ok(recommendations)
    }
    
    fn update(&mut self, _user_id: usize, _item_id: usize, _rating: f64) -> Result<()> {
        Ok(())
    }
    
    fn compute_similarity(&self, _id1: usize, _id2: usize, _similarity_type: SimilarityType) -> Result<f64> {
        Ok(0.75)
    }
    
    fn parameters(&self) -> &Array1<f64> {
        &self.parameters
    }
    
    fn clone_box(&self) -> Box<dyn RecommendationEngine> {
        Box::new(self.clone())
    }
}

/// Quantum content-based engine
#[derive(Debug, Clone)]
struct QuantumCBEngine {
    config: QuantumRecommenderConfig,
    parameters: Array1<f64>,
}

impl QuantumCBEngine {
    fn new(config: &QuantumRecommenderConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            parameters: Array1::zeros(100),
        })
    }
}

impl RecommendationEngine for QuantumCBEngine {
    fn recommend(&self, _user_id: usize, n_items: usize, _exclude_seen: bool) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();
        
        for i in 0..n_items {
            recommendations.push(Recommendation {
                item_id: i * 3,
                score: 3.9 - 0.08 * i as f64,
                confidence: (3.4, 4.4),
                explanation: None,
                quantum_contribution: 0.35,
            });
        }
        
        Ok(recommendations)
    }
    
    fn update(&mut self, _user_id: usize, _item_id: usize, _rating: f64) -> Result<()> {
        Ok(())
    }
    
    fn compute_similarity(&self, _id1: usize, _id2: usize, _similarity_type: SimilarityType) -> Result<f64> {
        Ok(0.7)
    }
    
    fn parameters(&self) -> &Array1<f64> {
        &self.parameters
    }
    
    fn clone_box(&self) -> Box<dyn RecommendationEngine> {
        Box::new(self.clone())
    }
}

/// Hybrid quantum engine
#[derive(Debug, Clone)]
struct HybridQuantumEngine {
    config: QuantumRecommenderConfig,
    parameters: Array1<f64>,
}

impl HybridQuantumEngine {
    fn new(config: &QuantumRecommenderConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            parameters: Array1::zeros(100),
        })
    }
}

impl RecommendationEngine for HybridQuantumEngine {
    fn recommend(&self, _user_id: usize, n_items: usize, _exclude_seen: bool) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();
        
        for i in 0..n_items {
            recommendations.push(Recommendation {
                item_id: i,
                score: 4.3 - 0.06 * i as f64,
                confidence: (3.9, 4.7),
                explanation: None,
                quantum_contribution: 0.5,
            });
        }
        
        Ok(recommendations)
    }
    
    fn update(&mut self, _user_id: usize, _item_id: usize, _rating: f64) -> Result<()> {
        Ok(())
    }
    
    fn compute_similarity(&self, _id1: usize, _id2: usize, _similarity_type: SimilarityType) -> Result<f64> {
        Ok(0.85)
    }
    
    fn parameters(&self) -> &Array1<f64> {
        &self.parameters
    }
    
    fn clone_box(&self) -> Box<dyn RecommendationEngine> {
        Box::new(self.clone())
    }
}

/// Quantum neural collaborative filtering engine
#[derive(Debug, Clone)]
struct QuantumNCFEngine {
    config: QuantumRecommenderConfig,
    parameters: Array1<f64>,
}

impl QuantumNCFEngine {
    fn new(config: &QuantumRecommenderConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            parameters: Array1::zeros(100),
        })
    }
}

impl RecommendationEngine for QuantumNCFEngine {
    fn recommend(&self, _user_id: usize, n_items: usize, _exclude_seen: bool) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();
        
        for i in 0..n_items {
            recommendations.push(Recommendation {
                item_id: i * 4,
                score: 4.1 - 0.07 * i as f64,
                confidence: (3.7, 4.5),
                explanation: None,
                quantum_contribution: 0.45,
            });
        }
        
        Ok(recommendations)
    }
    
    fn update(&mut self, _user_id: usize, _item_id: usize, _rating: f64) -> Result<()> {
        Ok(())
    }
    
    fn compute_similarity(&self, _id1: usize, _id2: usize, _similarity_type: SimilarityType) -> Result<f64> {
        Ok(0.82)
    }
    
    fn parameters(&self) -> &Array1<f64> {
        &self.parameters
    }
    
    fn clone_box(&self) -> Box<dyn RecommendationEngine> {
        Box::new(self.clone())
    }
}

/// Quantum graph-based engine
#[derive(Debug, Clone)]
struct QuantumGraphEngine {
    config: QuantumRecommenderConfig,
    parameters: Array1<f64>,
}

impl QuantumGraphEngine {
    fn new(config: &QuantumRecommenderConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            parameters: Array1::zeros(100),
        })
    }
}

impl RecommendationEngine for QuantumGraphEngine {
    fn recommend(&self, _user_id: usize, n_items: usize, _exclude_seen: bool) -> Result<Vec<Recommendation>> {
        let mut recommendations = Vec::new();
        
        for i in 0..n_items {
            recommendations.push(Recommendation {
                item_id: i * 5,
                score: 4.0 - 0.09 * i as f64,
                confidence: (3.5, 4.5),
                explanation: None,
                quantum_contribution: 0.42,
            });
        }
        
        Ok(recommendations)
    }
    
    fn update(&mut self, _user_id: usize, _item_id: usize, _rating: f64) -> Result<()> {
        Ok(())
    }
    
    fn compute_similarity(&self, _id1: usize, _id2: usize, _similarity_type: SimilarityType) -> Result<f64> {
        Ok(0.78)
    }
    
    fn parameters(&self) -> &Array1<f64> {
        &self.parameters
    }
    
    fn clone_box(&self) -> Box<dyn RecommendationEngine> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_recommender_creation() {
        let config = QuantumRecommenderConfig::default();
        let recommender = QuantumRecommender::new(config).unwrap();
        assert!(recommender.user_profiles.is_empty());
    }
    
    #[test]
    fn test_add_interaction() {
        let config = QuantumRecommenderConfig::default();
        let mut recommender = QuantumRecommender::new(config).unwrap();
        
        recommender.add_interaction(1, 10, 4.5, None).unwrap();
        assert_eq!(recommender.user_profiles.len(), 1);
    }
    
    #[test]
    fn test_recommendations() {
        let config = QuantumRecommenderConfig::default();
        let mut recommender = QuantumRecommender::new(config).unwrap();
        
        // Add some interactions
        recommender.add_interaction(1, 10, 4.5, None).unwrap();
        recommender.add_interaction(1, 20, 3.5, None).unwrap();
        
        // Get recommendations
        let options = RecommendationOptions::default();
        let recommendations = recommender.recommend(1, 5, options).unwrap();
        
        assert_eq!(recommendations.len(), 5);
        assert!(recommendations[0].score >= recommendations[1].score);
    }
    
    #[test]
    fn test_similarity_measures() {
        let processor = QuantumProcessor::new(8).unwrap();
        let vec1 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let vec2 = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
        
        let cosine_sim = processor.compute_similarity(
            &vec1,
            &vec2,
            &SimilarityMeasure::Cosine,
        ).unwrap();
        
        assert!(cosine_sim.abs() < 1e-10); // Orthogonal vectors
    }
}