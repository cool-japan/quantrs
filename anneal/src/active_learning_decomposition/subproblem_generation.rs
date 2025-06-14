//! Subproblem generation components

use std::collections::HashMap;

use super::*;

/// Subproblem generator
#[derive(Debug, Clone)]
pub struct SubproblemGenerator {
    /// Generation strategies
    pub generation_strategies: Vec<GenerationStrategy>,
    /// Overlap manager
    pub overlap_manager: OverlapManager,
    /// Size controller
    pub size_controller: SizeController,
    /// Quality validator
    pub quality_validator: QualityValidator,
}

impl SubproblemGenerator {
    pub fn new() -> Result<Self, String> {
        Ok(Self {
            generation_strategies: Vec::new(),
            overlap_manager: OverlapManager::new(),
            size_controller: SizeController::new(),
            quality_validator: QualityValidator::new(),
        })
    }
}

/// Generation strategy
#[derive(Debug, Clone)]
pub struct GenerationStrategy {
    /// Strategy type
    pub strategy_type: GenerationStrategyType,
    /// Strategy parameters
    pub parameters: GenerationParameters,
    /// Success rate
    pub success_rate: f64,
    /// Average quality
    pub average_quality: f64,
}

/// Generation parameters
#[derive(Debug, Clone)]
pub struct GenerationParameters {
    /// Target number of subproblems
    pub target_num_subproblems: usize,
    /// Size balance tolerance
    pub size_balance_tolerance: f64,
    /// Quality threshold
    pub quality_threshold: f64,
    /// Maximum iterations
    pub max_iterations: usize,
}

/// Overlap manager
#[derive(Debug, Clone)]
pub struct OverlapManager {
    /// Overlap strategy
    pub overlap_strategy: OverlapStrategy,
    /// Overlap size
    pub overlap_size: usize,
    /// Overlap resolution method
    pub resolution_method: OverlapResolutionMethod,
}

impl OverlapManager {
    pub fn new() -> Self {
        Self {
            overlap_strategy: OverlapStrategy::NoOverlap,
            overlap_size: 0,
            resolution_method: OverlapResolutionMethod::Voting,
        }
    }
}

/// Size controller
#[derive(Debug, Clone)]
pub struct SizeController {
    /// Size constraints
    pub size_constraints: SizeConstraints,
    /// Size balancing strategy
    pub balancing_strategy: SizeBalancingStrategy,
    /// Adaptive sizing enabled
    pub adaptive_sizing: bool,
}

impl SizeController {
    pub fn new() -> Self {
        Self {
            size_constraints: SizeConstraints::default(),
            balancing_strategy: SizeBalancingStrategy::Flexible,
            adaptive_sizing: true,
        }
    }
}

/// Quality validator
#[derive(Debug, Clone)]
pub struct QualityValidator {
    /// Validation criteria
    pub validation_criteria: Vec<ValidationCriterion>,
    /// Validation threshold
    pub validation_threshold: f64,
    /// Strict validation enabled
    pub strict_validation: bool,
}

impl QualityValidator {
    pub fn new() -> Self {
        Self {
            validation_criteria: vec![
                ValidationCriterion {
                    criterion_type: ValidationCriterionType::ConnectivityPreservation,
                    weight: 0.3,
                    threshold: 0.8,
                },
                ValidationCriterion {
                    criterion_type: ValidationCriterionType::SizeBalance,
                    weight: 0.2,
                    threshold: 0.7,
                },
                ValidationCriterion {
                    criterion_type: ValidationCriterionType::CutQuality,
                    weight: 0.5,
                    threshold: 0.6,
                },
            ],
            validation_threshold: 0.7,
            strict_validation: false,
        }
    }
}

/// Validation criterion
#[derive(Debug, Clone)]
pub struct ValidationCriterion {
    /// Criterion type
    pub criterion_type: ValidationCriterionType,
    /// Weight in overall validation
    pub weight: f64,
    /// Threshold for this criterion
    pub threshold: f64,
}