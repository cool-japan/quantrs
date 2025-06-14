//! Transfer learning system for meta-learning optimization

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::config::*;
use super::feature_extraction::*;

/// Transfer learning system
pub struct TransferLearner {
    /// Source domains
    pub source_domains: Vec<SourceDomain>,
    /// Transfer strategies
    pub strategies: Vec<TransferStrategy>,
    /// Domain similarity analyzer
    pub similarity_analyzer: DomainSimilarityAnalyzer,
    /// Transfer history
    pub transfer_history: VecDeque<TransferRecord>,
    /// Adaptation mechanisms
    pub adaptation_mechanisms: Vec<AdaptationMechanism>,
}

impl TransferLearner {
    pub fn new() -> Self {
        Self {
            source_domains: Vec::new(),
            strategies: vec![
                TransferStrategy::InstanceTransfer,
                TransferStrategy::FeatureTransfer,
                TransferStrategy::ParameterTransfer,
            ],
            similarity_analyzer: DomainSimilarityAnalyzer::new(),
            transfer_history: VecDeque::new(),
            adaptation_mechanisms: vec![
                AdaptationMechanism::FineTuning,
                AdaptationMechanism::DomainAdaptation,
            ],
        }
    }

    pub fn add_source_domain(&mut self, domain: SourceDomain) {
        self.source_domains.push(domain);
        
        // Limit number of source domains
        while self.source_domains.len() > 50 {
            self.source_domains.remove(0);
        }
    }

    pub fn transfer_knowledge(
        &mut self,
        target_features: &ProblemFeatures,
        target_domain: &ProblemDomain,
    ) -> Result<Vec<TransferableModel>, String> {
        // Find most similar source domains
        let similar_domains = self.find_similar_domains(target_features, target_domain)?;
        
        if similar_domains.is_empty() {
            return Ok(Vec::new());
        }

        let mut transferred_models = Vec::new();

        for (domain, similarity) in similar_domains.iter().take(3) { // Top 3 similar domains
            for strategy in &self.strategies.clone() {
                if let Ok(model) = self.apply_transfer_strategy(domain, target_features, strategy, *similarity) {
                    transferred_models.push(model);
                }
            }
        }

        // Record transfer attempt
        let transfer_record = TransferRecord {
            timestamp: Instant::now(),
            source_domains: similar_domains.iter().map(|(d, _)| d.characteristics.clone()).collect(),
            target_features: target_features.clone(),
            strategies_used: self.strategies.clone(),
            success_rate: if transferred_models.is_empty() { 0.0 } else { 1.0 },
            models_transferred: transferred_models.len(),
        };

        self.transfer_history.push_back(transfer_record);

        // Limit transfer history
        while self.transfer_history.len() > 1000 {
            self.transfer_history.pop_front();
        }

        Ok(transferred_models)
    }

    fn find_similar_domains(
        &self,
        target_features: &ProblemFeatures,
        target_domain: &ProblemDomain,
    ) -> Result<Vec<(SourceDomain, f64)>, String> {
        let mut similarities = Vec::new();

        for source_domain in &self.source_domains {
            // Check domain compatibility
            if !self.is_domain_compatible(&source_domain.characteristics.domain, target_domain) {
                continue;
            }

            let similarity = self.similarity_analyzer.calculate_similarity(
                &source_domain.characteristics,
                target_features,
                target_domain,
            )?;

            if similarity > 0.3 { // Minimum similarity threshold
                similarities.push((source_domain.clone(), similarity));
            }
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(similarities)
    }

    fn is_domain_compatible(&self, source_domain: &ProblemDomain, target_domain: &ProblemDomain) -> bool {
        // Allow transfer within same domain or to/from general domains
        source_domain == target_domain || 
        matches!(source_domain, ProblemDomain::Combinatorial) ||
        matches!(target_domain, ProblemDomain::Combinatorial)
    }

    fn apply_transfer_strategy(
        &self,
        source_domain: &SourceDomain,
        target_features: &ProblemFeatures,
        strategy: &TransferStrategy,
        similarity: f64,
    ) -> Result<TransferableModel, String> {
        match strategy {
            TransferStrategy::InstanceTransfer => {
                self.instance_transfer(source_domain, target_features, similarity)
            },
            TransferStrategy::FeatureTransfer => {
                self.feature_transfer(source_domain, target_features, similarity)
            },
            TransferStrategy::ParameterTransfer => {
                self.parameter_transfer(source_domain, target_features, similarity)
            },
            TransferStrategy::RelationTransfer => {
                self.relation_transfer(source_domain, target_features, similarity)
            },
            TransferStrategy::ModelTransfer => {
                self.model_transfer(source_domain, target_features, similarity)
            },
        }
    }

    fn instance_transfer(
        &self,
        source_domain: &SourceDomain,
        target_features: &ProblemFeatures,
        similarity: f64,
    ) -> Result<TransferableModel, String> {
        // Transfer relevant instances based on similarity
        let relevant_experiences: Vec<_> = source_domain.experiences.iter()
            .filter(|exp| {
                let exp_similarity = self.calculate_experience_similarity(&exp.problem_features, target_features);
                exp_similarity > 0.5
            })
            .cloned()
            .collect();

        Ok(TransferableModel {
            model_type: ModelType::InstanceBased,
            source_domain_id: source_domain.id.clone(),
            parameters: HashMap::new(),
            knowledge: Knowledge::Instances(relevant_experiences),
            confidence: similarity * 0.8,
            adaptation_required: true,
        })
    }

    fn feature_transfer(
        &self,
        source_domain: &SourceDomain,
        target_features: &ProblemFeatures,
        similarity: f64,
    ) -> Result<TransferableModel, String> {
        // Transfer feature representations and transformations
        let mut feature_mappings = HashMap::new();
        
        // Simple feature mapping based on similarity
        feature_mappings.insert("size_scaling".to_string(), target_features.size as f64 / source_domain.characteristics.avg_problem_size);
        feature_mappings.insert("density_scaling".to_string(), target_features.density / source_domain.characteristics.avg_density);

        Ok(TransferableModel {
            model_type: ModelType::FeatureBased,
            source_domain_id: source_domain.id.clone(),
            parameters: feature_mappings,
            knowledge: Knowledge::FeatureMapping(HashMap::new()),
            confidence: similarity * 0.9,
            adaptation_required: false,
        })
    }

    fn parameter_transfer(
        &self,
        source_domain: &SourceDomain,
        target_features: &ProblemFeatures,
        similarity: f64,
    ) -> Result<TransferableModel, String> {
        // Transfer learned parameters from source domain
        let mut transferred_params = HashMap::new();

        // Get average hyperparameters from source domain experiences
        if !source_domain.experiences.is_empty() {
            let mut param_sums: HashMap<String, f64> = HashMap::new();
            let mut param_counts: HashMap<String, usize> = HashMap::new();

            for experience in &source_domain.experiences {
                for (param_name, param_value) in &experience.configuration.hyperparameters {
                    *param_sums.entry(param_name.clone()).or_insert(0.0) += param_value;
                    *param_counts.entry(param_name.clone()).or_insert(0) += 1;
                }
            }

            // Calculate averages and adapt to target domain
            for (param_name, total) in param_sums {
                if let Some(&count) = param_counts.get(&param_name) {
                    let avg_value = total / count as f64;
                    let adapted_value = self.adapt_parameter(&param_name, avg_value, target_features, similarity);
                    transferred_params.insert(param_name, adapted_value);
                }
            }
        }

        Ok(TransferableModel {
            model_type: ModelType::ParameterBased,
            source_domain_id: source_domain.id.clone(),
            parameters: transferred_params,
            knowledge: Knowledge::Parameters(HashMap::new()),
            confidence: similarity * 0.7,
            adaptation_required: true,
        })
    }

    fn relation_transfer(
        &self,
        source_domain: &SourceDomain,
        target_features: &ProblemFeatures,
        similarity: f64,
    ) -> Result<TransferableModel, String> {
        // Transfer relational knowledge between features and performance
        let mut relations = HashMap::new();
        
        // Simplified relation extraction
        relations.insert("size_performance_relation".to_string(), 0.8);
        relations.insert("density_performance_relation".to_string(), 0.6);

        Ok(TransferableModel {
            model_type: ModelType::RelationBased,
            source_domain_id: source_domain.id.clone(),
            parameters: relations,
            knowledge: Knowledge::Relations(HashMap::new()),
            confidence: similarity * 0.6,
            adaptation_required: true,
        })
    }

    fn model_transfer(
        &self,
        source_domain: &SourceDomain,
        target_features: &ProblemFeatures,
        similarity: f64,
    ) -> Result<TransferableModel, String> {
        // Transfer entire learned models
        Ok(TransferableModel {
            model_type: ModelType::CompleteBased,
            source_domain_id: source_domain.id.clone(),
            parameters: HashMap::new(),
            knowledge: Knowledge::CompleteModel(Vec::new()),
            confidence: similarity * 0.5,
            adaptation_required: true,
        })
    }

    fn adapt_parameter(&self, param_name: &str, value: f64, target_features: &ProblemFeatures, similarity: f64) -> f64 {
        // Simple parameter adaptation based on problem characteristics
        match param_name {
            "initial_temperature" => {
                // Scale initial temperature based on problem size
                let size_factor = (target_features.size as f64 / 100.0).sqrt();
                value * size_factor * similarity
            },
            "cooling_rate" => {
                // Adapt cooling rate based on problem density
                let density_factor = 1.0 + target_features.density * 0.2;
                value * density_factor
            },
            "num_sweeps" | "max_iterations" => {
                // Scale iterations based on problem size
                let size_factor = (target_features.size as f64 / 100.0).ln().max(1.0);
                value * size_factor
            },
            _ => value * similarity, // Default adaptation
        }
    }

    fn calculate_experience_similarity(&self, exp_features: &ProblemFeatures, target_features: &ProblemFeatures) -> f64 {
        // Simple similarity calculation
        let size_similarity = 1.0 - (exp_features.size as f64 - target_features.size as f64).abs() / 
                               exp_features.size.max(target_features.size) as f64;
        let density_similarity = 1.0 - (exp_features.density - target_features.density).abs();
        
        (size_similarity + density_similarity) / 2.0
    }

    pub fn adapt_model(&self, model: &mut TransferableModel, target_features: &ProblemFeatures) -> Result<(), String> {
        if !model.adaptation_required {
            return Ok(());
        }

        match model.model_type {
            ModelType::ParameterBased => {
                // Fine-tune parameters for target domain
                for (param_name, param_value) in model.parameters.iter_mut() {
                    *param_value = self.adapt_parameter(param_name, *param_value, target_features, model.confidence);
                }
            },
            ModelType::InstanceBased => {
                // Filter instances based on target domain relevance
                if let Knowledge::Instances(ref mut instances) = model.knowledge {
                    instances.retain(|exp| {
                        self.calculate_experience_similarity(&exp.problem_features, target_features) > 0.4
                    });
                }
            },
            _ => {
                // Other adaptation mechanisms would be implemented here
            }
        }

        model.adaptation_required = false;
        Ok(())
    }

    pub fn evaluate_transfer_success(&self) -> f64 {
        if self.transfer_history.is_empty() {
            return 0.0;
        }

        let total_success: f64 = self.transfer_history.iter()
            .map(|record| record.success_rate)
            .sum();

        total_success / self.transfer_history.len() as f64
    }
}

/// Source domain for transfer learning
#[derive(Debug, Clone)]
pub struct SourceDomain {
    /// Domain identifier
    pub id: String,
    /// Domain characteristics
    pub characteristics: DomainCharacteristics,
    /// Experiences from this domain
    pub experiences: Vec<OptimizationExperience>,
    /// Performance models
    pub models: Vec<TransferableModel>,
    /// Last updated
    pub last_updated: Instant,
}

/// Domain characteristics
#[derive(Debug, Clone)]
pub struct DomainCharacteristics {
    /// Problem domain type
    pub domain: ProblemDomain,
    /// Average problem size
    pub avg_problem_size: f64,
    /// Average problem density
    pub avg_density: f64,
    /// Typical algorithms used
    pub typical_algorithms: Vec<AlgorithmType>,
    /// Performance distribution
    pub performance_distribution: DistributionStats,
    /// Feature importance
    pub feature_importance: HashMap<String, f64>,
}

/// Transferable model
#[derive(Debug, Clone)]
pub struct TransferableModel {
    /// Model type
    pub model_type: ModelType,
    /// Source domain identifier
    pub source_domain_id: String,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Encoded knowledge
    pub knowledge: Knowledge,
    /// Transfer confidence
    pub confidence: f64,
    /// Whether adaptation is required
    pub adaptation_required: bool,
}

/// Types of transferable models
#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    /// Instance-based transfer
    InstanceBased,
    /// Feature-based transfer
    FeatureBased,
    /// Parameter-based transfer
    ParameterBased,
    /// Relation-based transfer
    RelationBased,
    /// Complete model transfer
    CompleteBased,
}

/// Knowledge representation
#[derive(Debug, Clone)]
pub enum Knowledge {
    /// Instance knowledge
    Instances(Vec<OptimizationExperience>),
    /// Feature mapping knowledge
    FeatureMapping(HashMap<String, Vec<f64>>),
    /// Parameter knowledge
    Parameters(HashMap<String, DistributionStats>),
    /// Relational knowledge
    Relations(HashMap<String, f64>),
    /// Complete model knowledge
    CompleteModel(Vec<u8>),
}

/// Transfer record
#[derive(Debug, Clone)]
pub struct TransferRecord {
    /// Transfer timestamp
    pub timestamp: Instant,
    /// Source domain characteristics
    pub source_domains: Vec<DomainCharacteristics>,
    /// Target features
    pub target_features: ProblemFeatures,
    /// Strategies used
    pub strategies_used: Vec<TransferStrategy>,
    /// Transfer success rate
    pub success_rate: f64,
    /// Number of models transferred
    pub models_transferred: usize,
}

/// Domain similarity analyzer
#[derive(Debug)]
pub struct DomainSimilarityAnalyzer {
    /// Similarity metrics
    pub metrics: Vec<SimilarityMetric>,
    /// Similarity methods
    pub methods: Vec<SimilarityMethod>,
    /// Cached similarities
    pub similarity_cache: HashMap<String, f64>,
}

impl DomainSimilarityAnalyzer {
    pub fn new() -> Self {
        Self {
            metrics: vec![
                SimilarityMetric::FeatureSimilarity,
                SimilarityMetric::StatisticalSimilarity,
                SimilarityMetric::PerformanceSimilarity,
            ],
            methods: vec![
                SimilarityMethod::EuclideanDistance,
                SimilarityMethod::CosineSimilarity,
                SimilarityMethod::KLDivergence,
            ],
            similarity_cache: HashMap::new(),
        }
    }

    pub fn calculate_similarity(
        &mut self,
        source_characteristics: &DomainCharacteristics,
        target_features: &ProblemFeatures,
        target_domain: &ProblemDomain,
    ) -> Result<f64, String> {
        // Create cache key
        let cache_key = format!("{}_{}_{}_{}", 
            source_characteristics.domain as u8,
            source_characteristics.avg_problem_size as u32,
            target_features.size,
            target_domain as u8
        );

        // Check cache first
        if let Some(&cached_similarity) = self.similarity_cache.get(&cache_key) {
            return Ok(cached_similarity);
        }

        let mut similarity_scores = Vec::new();

        // Domain type similarity
        let domain_similarity = if source_characteristics.domain == *target_domain {
            1.0
        } else if matches!(source_characteristics.domain, ProblemDomain::Combinatorial) || 
                  matches!(target_domain, ProblemDomain::Combinatorial) {
            0.7 // Combinatorial problems are somewhat similar to others
        } else {
            0.3
        };
        similarity_scores.push(domain_similarity);

        // Size similarity
        let size_ratio = source_characteristics.avg_problem_size / target_features.size as f64;
        let size_similarity = 1.0 - (size_ratio.ln().abs() / 3.0).min(1.0); // Logarithmic similarity
        similarity_scores.push(size_similarity);

        // Density similarity
        let density_similarity = 1.0 - (source_characteristics.avg_density - target_features.density).abs();
        similarity_scores.push(density_similarity);

        // Calculate overall similarity as weighted average
        let weights = vec![0.4, 0.3, 0.3]; // Domain type is most important
        let overall_similarity: f64 = similarity_scores.iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum();

        // Cache the result
        self.similarity_cache.insert(cache_key, overall_similarity);

        // Limit cache size
        if self.similarity_cache.len() > 10000 {
            self.similarity_cache.clear();
        }

        Ok(overall_similarity)
    }
}

/// Similarity metrics
#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityMetric {
    /// Feature-based similarity
    FeatureSimilarity,
    /// Statistical similarity
    StatisticalSimilarity,
    /// Performance similarity
    PerformanceSimilarity,
    /// Domain similarity
    DomainSimilarity,
    /// Task similarity
    TaskSimilarity,
}

/// Similarity measurement methods
#[derive(Debug, Clone, PartialEq)]
pub enum SimilarityMethod {
    /// Euclidean distance
    EuclideanDistance,
    /// Cosine similarity
    CosineSimilarity,
    /// Pearson correlation
    PearsonCorrelation,
    /// Kullback-Leibler divergence
    KLDivergence,
    /// Maximum Mean Discrepancy
    MaximumMeanDiscrepancy,
}

/// Transfer strategies
#[derive(Debug, Clone, PartialEq)]
pub enum TransferStrategy {
    /// Instance transfer
    InstanceTransfer,
    /// Feature transfer
    FeatureTransfer,
    /// Parameter transfer
    ParameterTransfer,
    /// Relation transfer
    RelationTransfer,
    /// Model transfer
    ModelTransfer,
}

/// Adaptation mechanisms
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationMechanism {
    /// Fine-tuning
    FineTuning,
    /// Domain adaptation
    DomainAdaptation,
    /// Multi-task learning
    MultiTaskLearning,
    /// Progressive transfer
    ProgressiveTransfer,
    /// Adversarial adaptation
    AdversarialAdaptation,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_learner_creation() {
        let transfer_learner = TransferLearner::new();
        assert_eq!(transfer_learner.source_domains.len(), 0);
        assert_eq!(transfer_learner.strategies.len(), 3);
    }

    #[test]
    fn test_domain_similarity_analyzer() {
        let mut analyzer = DomainSimilarityAnalyzer::new();
        
        let source_characteristics = DomainCharacteristics {
            domain: ProblemDomain::Combinatorial,
            avg_problem_size: 100.0,
            avg_density: 0.5,
            typical_algorithms: vec![AlgorithmType::SimulatedAnnealing],
            performance_distribution: DistributionStats::default(),
            feature_importance: HashMap::new(),
        };
        
        let target_features = ProblemFeatures {
            size: 120,
            density: 0.6,
            graph_features: GraphFeatures::default(),
            statistical_features: StatisticalFeatures::default(),
            spectral_features: SpectralFeatures::default(),
            domain_features: HashMap::new(),
        };
        
        let target_domain = ProblemDomain::Combinatorial;
        
        let similarity = analyzer.calculate_similarity(&source_characteristics, &target_features, &target_domain);
        assert!(similarity.is_ok());
        
        let sim_value = similarity.unwrap();
        assert!(sim_value >= 0.0 && sim_value <= 1.0);
    }

    #[test]
    fn test_transferable_model() {
        let model = TransferableModel {
            model_type: ModelType::ParameterBased,
            source_domain_id: "test_domain".to_string(),
            parameters: HashMap::new(),
            knowledge: Knowledge::Parameters(HashMap::new()),
            confidence: 0.8,
            adaptation_required: true,
        };
        
        assert_eq!(model.model_type, ModelType::ParameterBased);
        assert!(model.adaptation_required);
        assert_eq!(model.confidence, 0.8);
    }
}