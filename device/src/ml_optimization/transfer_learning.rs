//! Transfer Learning Configuration Types

/// Transfer learning configuration
#[derive(Debug, Clone)]
pub struct TransferLearningConfig {
    /// Enable transfer learning
    pub enable_transfer_learning: bool,
    /// Source domains
    pub source_domains: Vec<String>,
    /// Transfer methods
    pub transfer_methods: Vec<TransferMethod>,
    /// Domain adaptation
    pub domain_adaptation: DomainAdaptationConfig,
    /// Meta-learning
    pub meta_learning: MetaLearningConfig,
}

/// Transfer learning methods
#[derive(Debug, Clone, PartialEq)]
pub enum TransferMethod {
    FeatureTransfer,
    ParameterTransfer,
    InstanceTransfer,
    RelationalTransfer,
    DomainAdversarial,
    FewShotLearning,
}

/// Domain adaptation configuration
#[derive(Debug, Clone)]
pub struct DomainAdaptationConfig {
    /// Enable domain adaptation
    pub enable_adaptation: bool,
    /// Adaptation methods
    pub adaptation_methods: Vec<DomainAdaptationMethod>,
    /// Source-target similarity threshold
    pub similarity_threshold: f64,
    /// Maximum domain gap
    pub max_domain_gap: f64,
}

/// Domain adaptation methods
#[derive(Debug, Clone, PartialEq)]
pub enum DomainAdaptationMethod {
    CausalInference,
    DistributionMatching,
    AdversarialTraining,
    GradientReversal,
    CORAL,
}

/// Meta-learning configuration
#[derive(Debug, Clone)]
pub struct MetaLearningConfig {
    /// Enable meta-learning
    pub enable_meta_learning: bool,
    /// Meta-learning algorithms
    pub meta_algorithms: Vec<MetaLearningAlgorithm>,
    /// Inner loop iterations
    pub inner_loop_iterations: usize,
    /// Outer loop iterations
    pub outer_loop_iterations: usize,
}

/// Meta-learning algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum MetaLearningAlgorithm {
    MAML,
    Reptile,
    ProtoNet,
    RelationNet,
    MatchingNet,
}
