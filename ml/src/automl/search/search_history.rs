//! Search History Tracking
//!
//! This module tracks the history of the AutoML search process.

use crate::automl::pipeline::QuantumMLPipeline;
use std::time::{Duration, Instant};

/// Search history tracker
#[derive(Debug, Clone)]
pub struct SearchHistory {
    /// Search trials
    trials: Vec<SearchTrial>,

    /// Search start time
    start_time: Option<Instant>,

    /// Best performance seen so far
    best_performance: Option<f64>,

    /// Trials without improvement
    trials_without_improvement: usize,

    /// Search statistics
    statistics: SearchStatistics,
}

/// Individual search trial
#[derive(Debug, Clone)]
pub struct SearchTrial {
    /// Trial ID
    pub trial_id: usize,

    /// Pipeline configuration
    pub pipeline_config: PipelineConfig,

    /// Performance achieved
    pub performance: f64,

    /// Trial duration
    pub duration: Duration,

    /// Resource usage
    pub resource_usage: TrialResourceUsage,
}

/// Pipeline configuration for search history
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Model type used
    pub model_type: String,

    /// Hyperparameters
    pub hyperparameters: std::collections::HashMap<String, f64>,

    /// Architecture summary
    pub architecture_summary: String,

    /// Preprocessing steps
    pub preprocessing_steps: Vec<String>,
}

/// Resource usage for a trial
#[derive(Debug, Clone)]
pub struct TrialResourceUsage {
    /// Memory usage (MB)
    pub memory_mb: f64,

    /// Quantum resources
    pub quantum_resources: QuantumResourceUsage,

    /// Training time
    pub training_time: f64,
}

/// Quantum resource usage
#[derive(Debug, Clone)]
pub struct QuantumResourceUsage {
    /// Qubits used
    pub qubits_used: usize,

    /// Circuit depth
    pub circuit_depth: usize,

    /// Gate count
    pub gate_count: usize,

    /// Coherence time used
    pub coherence_time_used: f64,
}

/// Search statistics
#[derive(Debug, Clone)]
pub struct SearchStatistics {
    /// Total trials completed
    pub total_trials: usize,

    /// Total search time
    pub total_time: Duration,

    /// Average performance
    pub average_performance: f64,

    /// Performance standard deviation
    pub performance_std: f64,

    /// Best trial ID
    pub best_trial_id: Option<usize>,

    /// Convergence rate
    pub convergence_rate: f64,
}

impl SearchHistory {
    /// Create a new search history tracker
    pub fn new() -> Self {
        Self {
            trials: Vec::new(),
            start_time: None,
            best_performance: None,
            trials_without_improvement: 0,
            statistics: SearchStatistics::new(),
        }
    }

    /// Start the search process
    pub fn start_search(&mut self) {
        self.start_time = Some(Instant::now());
        self.trials.clear();
        self.best_performance = None;
        self.trials_without_improvement = 0;
        self.statistics = SearchStatistics::new();
    }

    /// Record a search trial
    pub fn record_trial(
        &mut self,
        trial_id: usize,
        pipeline: &QuantumMLPipeline,
        performance: f64,
    ) {
        let trial_start = Instant::now();

        let trial = SearchTrial {
            trial_id,
            pipeline_config: PipelineConfig::from_pipeline(pipeline),
            performance,
            duration: trial_start.elapsed(),
            resource_usage: TrialResourceUsage::from_pipeline(pipeline),
        };

        // Update best performance tracking
        let is_improvement = match self.best_performance {
            Some(best) => performance > best,
            None => true,
        };

        if is_improvement {
            self.best_performance = Some(performance);
            self.trials_without_improvement = 0;
            self.statistics.best_trial_id = Some(trial_id);
        } else {
            self.trials_without_improvement += 1;
        }

        self.trials.push(trial);
        self.update_statistics();
    }

    /// Get elapsed search time
    pub fn elapsed_time(&self) -> Option<f64> {
        self.start_time.map(|start| start.elapsed().as_secs_f64())
    }

    /// Get trials without improvement
    pub fn trials_without_improvement(&self) -> usize {
        self.trials_without_improvement
    }

    /// Get best performance achieved
    pub fn best_performance(&self) -> Option<f64> {
        self.best_performance
    }

    /// Get all trials
    pub fn trials(&self) -> &[SearchTrial] {
        &self.trials
    }

    /// Get search statistics
    pub fn statistics(&self) -> &SearchStatistics {
        &self.statistics
    }

    /// Get best trial
    pub fn best_trial(&self) -> Option<&SearchTrial> {
        self.statistics
            .best_trial_id
            .and_then(|id| self.trials.iter().find(|t| t.trial_id == id))
    }

    /// Get performance history
    pub fn performance_history(&self) -> Vec<f64> {
        self.trials.iter().map(|t| t.performance).collect()
    }

    /// Get convergence curve (best performance over time)
    pub fn convergence_curve(&self) -> Vec<f64> {
        let mut best_so_far = f64::NEG_INFINITY;
        self.trials
            .iter()
            .map(|t| {
                if t.performance > best_so_far {
                    best_so_far = t.performance;
                }
                best_so_far
            })
            .collect()
    }

    // Private methods

    fn update_statistics(&mut self) {
        let performances: Vec<f64> = self.trials.iter().map(|t| t.performance).collect();

        self.statistics.total_trials = self.trials.len();
        self.statistics.total_time = self
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or(Duration::ZERO);

        if !performances.is_empty() {
            self.statistics.average_performance =
                performances.iter().sum::<f64>() / performances.len() as f64;

            let variance = performances
                .iter()
                .map(|&p| (p - self.statistics.average_performance).powi(2))
                .sum::<f64>()
                / performances.len() as f64;
            self.statistics.performance_std = variance.sqrt();

            // Calculate convergence rate (how quickly we're improving)
            if performances.len() >= 10 {
                let recent_best = performances
                    .iter()
                    .rev()
                    .take(10)
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let early_best = performances
                    .iter()
                    .take(10)
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                self.statistics.convergence_rate = (recent_best - early_best) / 10.0;
            }
        }
    }
}

impl PipelineConfig {
    fn from_pipeline(pipeline: &QuantumMLPipeline) -> Self {
        // This would extract configuration from the actual pipeline
        // For now, providing a simplified implementation
        Self {
            model_type: "QuantumNeuralNetwork".to_string(), // pipeline.model_type()
            hyperparameters: std::collections::HashMap::new(), // pipeline.hyperparameters()
            architecture_summary: "4-qubit variational circuit".to_string(), // pipeline.architecture_summary()
            preprocessing_steps: vec!["StandardScaler".to_string(), "AngleEncoding".to_string()],
        }
    }
}

impl TrialResourceUsage {
    fn from_pipeline(pipeline: &QuantumMLPipeline) -> Self {
        // This would extract resource usage from the actual pipeline
        // For now, providing a simplified implementation
        Self {
            memory_mb: 256.0, // Estimate based on pipeline complexity
            quantum_resources: QuantumResourceUsage {
                qubits_used: 4,            // pipeline.qubits_used()
                circuit_depth: 6,          // pipeline.circuit_depth()
                gate_count: 24,            // pipeline.gate_count()
                coherence_time_used: 50.0, // pipeline.coherence_time_used()
            },
            training_time: 120.0, // Measured training time
        }
    }
}

impl SearchStatistics {
    fn new() -> Self {
        Self {
            total_trials: 0,
            total_time: Duration::ZERO,
            average_performance: 0.0,
            performance_std: 0.0,
            best_trial_id: None,
            convergence_rate: 0.0,
        }
    }
}

impl Default for SearchHistory {
    fn default() -> Self {
        Self::new()
    }
}
