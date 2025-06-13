//! Adaptive dynamical decoupling system with real-time optimization

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use quantrs2_core::qubit::QubitId;

use crate::{DeviceResult, DeviceError};
use super::{
    config::{
        AdaptiveDDConfig, AdaptationCriteria, MonitoringConfig, MonitoringMetric, 
        FeedbackControlConfig, LearningConfig, LearningAlgorithm, ControlAlgorithm,
        DDSequenceType, ExplorationStrategy
    },
    sequences::{DDSequence, DDSequenceGenerator},
    performance::DDPerformanceAnalysis,
    noise::DDNoiseAnalysis,
    DDCircuitExecutor,
};

/// Adaptive DD system state
#[derive(Debug, Clone)]
pub struct AdaptiveDDState {
    /// Current sequence being used
    pub current_sequence: DDSequence,
    /// Performance history
    pub performance_history: Vec<PerformanceRecord>,
    /// Noise characteristics history
    pub noise_history: Vec<NoiseRecord>,
    /// Adaptation history
    pub adaptation_history: Vec<AdaptationRecord>,
    /// Current performance metrics
    pub current_metrics: PerformanceMetrics,
    /// System health status
    pub system_health: SystemHealth,
}

/// Performance record for historical tracking
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Timestamp
    pub timestamp: Instant,
    /// Coherence time
    pub coherence_time: f64,
    /// Process fidelity
    pub fidelity: f64,
    /// Gate overhead
    pub gate_overhead: f64,
    /// Success rate
    pub success_rate: f64,
    /// Resource utilization
    pub resource_utilization: f64,
}

/// Noise record for historical tracking
#[derive(Debug, Clone)]
pub struct NoiseRecord {
    /// Timestamp
    pub timestamp: Instant,
    /// Noise level by type
    pub noise_levels: HashMap<String, f64>,
    /// Dominant noise sources
    pub dominant_sources: Vec<String>,
    /// Environmental conditions
    pub environmental_conditions: EnvironmentalConditions,
}

/// Environmental conditions
#[derive(Debug, Clone)]
pub struct EnvironmentalConditions {
    /// Temperature
    pub temperature: f64,
    /// Magnetic field strength
    pub magnetic_field: f64,
    /// Electromagnetic interference level
    pub emi_level: f64,
    /// Vibration level
    pub vibration_level: f64,
}

/// Adaptation record
#[derive(Debug, Clone)]
pub struct AdaptationRecord {
    /// Timestamp
    pub timestamp: Instant,
    /// Previous sequence
    pub previous_sequence: DDSequenceType,
    /// New sequence
    pub new_sequence: DDSequenceType,
    /// Reason for adaptation
    pub adaptation_reason: AdaptationReason,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Actual improvement (if measured)
    pub actual_improvement: Option<f64>,
}

/// Reasons for adaptation
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptationReason {
    /// Performance degradation
    PerformanceDegradation,
    /// Noise change
    NoiseChange,
    /// Environmental change
    EnvironmentalChange,
    /// Scheduled optimization
    ScheduledOptimization,
    /// Learning-based improvement
    LearningImprovement,
    /// Emergency intervention
    Emergency,
}

/// Current performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Real-time coherence time
    pub coherence_time: f64,
    /// Real-time fidelity
    pub fidelity: f64,
    /// Error rates
    pub error_rates: HashMap<String, f64>,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Throughput
    pub throughput: f64,
    /// Latency
    pub latency: Duration,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory usage
    pub memory_usage: usize,
    /// Network bandwidth usage
    pub network_usage: f64,
    /// Power consumption
    pub power_consumption: f64,
}

/// System health status
#[derive(Debug, Clone)]
pub struct SystemHealth {
    /// Overall health score (0.0 to 1.0)
    pub health_score: f64,
    /// Component health
    pub component_health: HashMap<String, f64>,
    /// Active alerts
    pub active_alerts: Vec<Alert>,
    /// Predicted issues
    pub predicted_issues: Vec<PredictedIssue>,
}

/// System alert
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert type
    pub alert_type: AlertType,
    /// Severity level
    pub severity: AlertSeverity,
    /// Message
    pub message: String,
    /// Timestamp
    pub timestamp: Instant,
    /// Suggested actions
    pub suggested_actions: Vec<String>,
}

/// Alert types
#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    PerformanceDegradation,
    NoiseIncrease,
    HardwareFailure,
    TemperatureAnomaly,
    ResourceExhaustion,
    CommunicationLoss,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Predicted issue
#[derive(Debug, Clone)]
pub struct PredictedIssue {
    /// Issue type
    pub issue_type: String,
    /// Probability
    pub probability: f64,
    /// Time to occurrence
    pub time_to_occurrence: Duration,
    /// Potential impact
    pub potential_impact: f64,
    /// Prevention strategies
    pub prevention_strategies: Vec<String>,
}

/// Adaptive DD system
pub struct AdaptiveDDSystem {
    /// Configuration
    config: AdaptiveDDConfig,
    /// Current state
    state: Arc<Mutex<AdaptiveDDState>>,
    /// Available sequence types
    available_sequences: Vec<DDSequenceType>,
    /// Learning agent
    learning_agent: Option<LearningAgent>,
    /// Feedback controller
    feedback_controller: FeedbackController,
    /// Real-time monitor
    monitor: RealTimeMonitor,
    /// Sequence cache
    sequence_cache: HashMap<String, DDSequence>,
}

/// Learning agent for adaptive DD
struct LearningAgent {
    /// Q-table for Q-learning (simplified)
    q_table: HashMap<(String, String), f64>,
    /// Experience replay buffer
    replay_buffer: Vec<Experience>,
    /// Exploration strategy
    exploration_strategy: ExplorationStrategy,
    /// Learning statistics
    learning_stats: LearningStatistics,
}

/// Experience for learning
#[derive(Debug, Clone)]
struct Experience {
    /// State representation
    state: String,
    /// Action taken
    action: String,
    /// Reward received
    reward: f64,
    /// Next state
    next_state: String,
    /// Done flag
    done: bool,
}

/// Learning statistics
#[derive(Debug, Clone)]
struct LearningStatistics {
    /// Total episodes
    total_episodes: usize,
    /// Average reward
    average_reward: f64,
    /// Exploration rate
    exploration_rate: f64,
    /// Learning rate
    learning_rate: f64,
}

/// Feedback controller
struct FeedbackController {
    /// Control algorithm
    algorithm: ControlAlgorithm,
    /// PID state
    pid_state: PIDState,
    /// Control history
    control_history: Vec<ControlAction>,
}

/// PID controller state
#[derive(Debug, Clone)]
struct PIDState {
    /// Previous error
    previous_error: f64,
    /// Integral accumulator
    integral: f64,
    /// Last update time
    last_update: Instant,
}

/// Control action
#[derive(Debug, Clone)]
struct ControlAction {
    /// Timestamp
    timestamp: Instant,
    /// Control output
    output: f64,
    /// Target value
    target: f64,
    /// Current value
    current: f64,
    /// Error
    error: f64,
}

/// Real-time monitor
struct RealTimeMonitor {
    /// Monitoring thread handle
    monitoring_enabled: bool,
    /// Metric collectors
    metric_collectors: HashMap<MonitoringMetric, MetricCollector>,
    /// Alert manager
    alert_manager: AlertManager,
}

/// Metric collector
struct MetricCollector {
    /// Last collected value
    last_value: f64,
    /// Collection history
    history: Vec<(Instant, f64)>,
    /// Moving average
    moving_average: f64,
    /// Threshold values
    thresholds: (f64, f64), // (warning, critical)
}

/// Alert manager
struct AlertManager {
    /// Active alerts
    active_alerts: Vec<Alert>,
    /// Alert history
    alert_history: Vec<Alert>,
    /// Alert rules
    alert_rules: Vec<AlertRule>,
}

/// Alert rule
struct AlertRule {
    /// Metric to monitor
    metric: MonitoringMetric,
    /// Condition
    condition: AlertCondition,
    /// Threshold value
    threshold: f64,
    /// Severity
    severity: AlertSeverity,
    /// Message template
    message_template: String,
}

/// Alert conditions
#[derive(Debug, Clone, PartialEq)]
enum AlertCondition {
    GreaterThan,
    LessThan,
    Equal,
    RateOfChange,
    Anomaly,
}

impl AdaptiveDDSystem {
    /// Create new adaptive DD system
    pub fn new(
        config: AdaptiveDDConfig,
        initial_sequence: DDSequence,
        available_sequences: Vec<DDSequenceType>,
    ) -> Self {
        let initial_state = AdaptiveDDState {
            current_sequence: initial_sequence,
            performance_history: Vec::new(),
            noise_history: Vec::new(),
            adaptation_history: Vec::new(),
            current_metrics: PerformanceMetrics {
                coherence_time: 0.0,
                fidelity: 0.0,
                error_rates: HashMap::new(),
                resource_usage: ResourceUsage {
                    cpu_utilization: 0.0,
                    memory_usage: 0,
                    network_usage: 0.0,
                    power_consumption: 0.0,
                },
                throughput: 0.0,
                latency: Duration::from_nanos(0),
            },
            system_health: SystemHealth {
                health_score: 1.0,
                component_health: HashMap::new(),
                active_alerts: Vec::new(),
                predicted_issues: Vec::new(),
            },
        };

        let learning_agent = if config.learning_config.learning_algorithm != LearningAlgorithm::QLearning {
            None
        } else {
            Some(LearningAgent {
                q_table: HashMap::new(),
                replay_buffer: Vec::new(),
                exploration_strategy: ExplorationStrategy::EpsilonGreedy(0.1),
                learning_stats: LearningStatistics {
                    total_episodes: 0,
                    average_reward: 0.0,
                    exploration_rate: 0.1,
                    learning_rate: config.learning_config.learning_rate,
                },
            })
        };

        Self {
            config,
            state: Arc::new(Mutex::new(initial_state)),
            available_sequences,
            learning_agent,
            feedback_controller: FeedbackController {
                algorithm: ControlAlgorithm::PID,
                pid_state: PIDState {
                    previous_error: 0.0,
                    integral: 0.0,
                    last_update: Instant::now(),
                },
                control_history: Vec::new(),
            },
            monitor: RealTimeMonitor {
                monitoring_enabled: true,
                metric_collectors: HashMap::new(),
                alert_manager: AlertManager {
                    active_alerts: Vec::new(),
                    alert_history: Vec::new(),
                    alert_rules: Vec::new(),
                },
            },
            sequence_cache: HashMap::new(),
        }
    }

    /// Start adaptive DD system
    pub fn start(&mut self, executor: &dyn DDCircuitExecutor) -> DeviceResult<()> {
        println!("Starting adaptive DD system");
        
        // Initialize monitoring
        self.initialize_monitoring()?;
        
        // Start feedback control loop
        self.start_control_loop(executor)?;
        
        Ok(())
    }

    /// Update system with new performance data
    pub fn update_performance(&mut self, 
        performance_analysis: &DDPerformanceAnalysis,
        noise_analysis: &DDNoiseAnalysis,
    ) -> DeviceResult<()> {
        let mut state = self.state.lock().unwrap();
        
        // Record performance
        let performance_record = PerformanceRecord {
            timestamp: Instant::now(),
            coherence_time: performance_analysis.coherence_preservation.effective_t2,
            fidelity: performance_analysis.fidelity_metrics.average_gate_fidelity,
            gate_overhead: performance_analysis.overhead_analysis.gate_count_overhead,
            success_rate: performance_analysis.success_metrics.overall_success_rate,
            resource_utilization: performance_analysis.resource_analysis.resource_efficiency,
        };
        state.performance_history.push(performance_record);

        // Record noise characteristics
        let noise_record = NoiseRecord {
            timestamp: Instant::now(),
            noise_levels: noise_analysis.noise_characterization.noise_types
                .iter()
                .map(|(noise_type, characteristics)| {
                    (format!("{:?}", noise_type), characteristics.strength)
                })
                .collect(),
            dominant_sources: noise_analysis.noise_characterization.dominant_sources
                .iter()
                .map(|source| format!("{:?}", source.source_type))
                .collect(),
            environmental_conditions: EnvironmentalConditions {
                temperature: 20.0, // Would be read from sensors
                magnetic_field: 0.1,
                emi_level: 0.05,
                vibration_level: 0.02,
            },
        };
        state.noise_history.push(noise_record);

        // Update current metrics
        state.current_metrics.coherence_time = performance_analysis.coherence_preservation.effective_t2;
        state.current_metrics.fidelity = performance_analysis.fidelity_metrics.average_gate_fidelity;

        // Check for adaptation triggers
        if self.should_adapt(&state)? {
            drop(state); // Release lock before adaptation
            self.trigger_adaptation()?;
        }

        Ok(())
    }

    /// Check if adaptation should be triggered
    fn should_adapt(&self, state: &AdaptiveDDState) -> DeviceResult<bool> {
        let criteria = &self.config.adaptation_criteria;
        
        // Check coherence time degradation
        if state.current_metrics.coherence_time < criteria.coherence_threshold {
            return Ok(true);
        }
        
        // Check fidelity degradation
        if state.current_metrics.fidelity < criteria.fidelity_threshold {
            return Ok(true);
        }
        
        // Check noise level increase
        let average_noise: f64 = state.current_metrics.error_rates.values().sum::<f64>() 
                               / state.current_metrics.error_rates.len().max(1) as f64;
        if average_noise > criteria.noise_threshold {
            return Ok(true);
        }
        
        Ok(false)
    }

    /// Trigger adaptation process
    fn trigger_adaptation(&mut self) -> DeviceResult<()> {
        println!("Triggering DD adaptation");
        
        // Analyze current performance
        let current_state = self.analyze_current_state()?;
        
        // Select new sequence
        let new_sequence_type = self.select_optimal_sequence(&current_state)?;
        
        // Generate new sequence
        let new_sequence = self.generate_sequence(&new_sequence_type)?;
        
        // Record adaptation
        let adaptation_record = AdaptationRecord {
            timestamp: Instant::now(),
            previous_sequence: self.state.lock().unwrap().current_sequence.sequence_type.clone(),
            new_sequence: new_sequence_type,
            adaptation_reason: AdaptationReason::PerformanceDegradation,
            expected_improvement: 0.1, // Estimated improvement
            actual_improvement: None,
        };
        
        // Update state
        {
            let mut state = self.state.lock().unwrap();
            state.current_sequence = new_sequence;
            state.adaptation_history.push(adaptation_record);
        }
        
        println!("DD adaptation completed");
        Ok(())
    }

    /// Analyze current system state
    fn analyze_current_state(&self) -> DeviceResult<String> {
        let state = self.state.lock().unwrap();
        
        // Create state representation for learning
        let coherence_level = if state.current_metrics.coherence_time > 50e-6 {
            "high"
        } else if state.current_metrics.coherence_time > 20e-6 {
            "medium"
        } else {
            "low"
        };
        
        let fidelity_level = if state.current_metrics.fidelity > 0.99 {
            "high"
        } else if state.current_metrics.fidelity > 0.95 {
            "medium"
        } else {
            "low"
        };
        
        let noise_level = {
            let avg_noise: f64 = state.current_metrics.error_rates.values().sum::<f64>() 
                               / state.current_metrics.error_rates.len().max(1) as f64;
            if avg_noise < 0.01 {
                "low"
            } else if avg_noise < 0.05 {
                "medium"
            } else {
                "high"
            }
        };
        
        Ok(format!("coherence_{}_fidelity_{}_noise_{}", coherence_level, fidelity_level, noise_level))
    }

    /// Select optimal sequence based on current conditions
    fn select_optimal_sequence(&mut self, current_state: &str) -> DeviceResult<DDSequenceType> {
        // Use learning agent if available
        if let Some(ref mut agent) = self.learning_agent {
            return self.select_sequence_with_learning(agent, current_state);
        }
        
        // Fallback to rule-based selection
        self.select_sequence_rule_based(current_state)
    }

    /// Select sequence using learning agent
    fn select_sequence_with_learning(&self, agent: &mut LearningAgent, current_state: &str) -> DeviceResult<DDSequenceType> {
        // Get Q-values for all available actions
        let mut best_sequence = DDSequenceType::CPMG;
        let mut best_q_value = f64::NEG_INFINITY;
        
        for sequence_type in &self.available_sequences {
            let action = format!("{:?}", sequence_type);
            let q_value = agent.q_table.get(&(current_state.to_string(), action))
                               .copied()
                               .unwrap_or(0.0);
            
            if q_value > best_q_value {
                best_q_value = q_value;
                best_sequence = sequence_type.clone();
            }
        }
        
        // Apply exploration
        match agent.exploration_strategy {
            ExplorationStrategy::EpsilonGreedy(epsilon) => {
                if rand::random::<f64>() < epsilon {
                    // Explore: select random sequence
                    let random_idx = rand::random::<usize>() % self.available_sequences.len();
                    best_sequence = self.available_sequences[random_idx].clone();
                }
            },
            _ => {
                // Use greedy selection
            }
        }
        
        Ok(best_sequence)
    }

    /// Select sequence using rule-based approach
    fn select_sequence_rule_based(&self, current_state: &str) -> DeviceResult<DDSequenceType> {
        // Simple rule-based selection
        match current_state {
            s if s.contains("noise_high") => Ok(DDSequenceType::XY8),
            s if s.contains("coherence_low") => Ok(DDSequenceType::UDD),
            s if s.contains("fidelity_low") => Ok(DDSequenceType::XY4),
            _ => Ok(DDSequenceType::CPMG),
        }
    }

    /// Generate sequence of specified type
    fn generate_sequence(&mut self, sequence_type: &DDSequenceType) -> DeviceResult<DDSequence> {
        // Check cache first
        let cache_key = format!("{:?}", sequence_type);
        if let Some(cached_sequence) = self.sequence_cache.get(&cache_key) {
            return Ok(cached_sequence.clone());
        }
        
        // Generate new sequence
        let target_qubits = vec![quantrs2_core::qubit::QubitId(0), quantrs2_core::qubit::QubitId(1)];
        let duration = 100e-6; // 100 microseconds
        
        let sequence = DDSequenceGenerator::generate_base_sequence(
            sequence_type,
            &target_qubits,
            duration,
        )?;
        
        // Cache the sequence
        self.sequence_cache.insert(cache_key, sequence.clone());
        
        Ok(sequence)
    }

    /// Initialize monitoring system
    fn initialize_monitoring(&mut self) -> DeviceResult<()> {
        println!("Initializing DD monitoring system");
        
        // Initialize metric collectors
        for metric in &self.config.monitoring_config.metrics {
            let collector = MetricCollector {
                last_value: 0.0,
                history: Vec::new(),
                moving_average: 0.0,
                thresholds: (0.8, 0.6), // Warning and critical thresholds
            };
            self.monitor.metric_collectors.insert(metric.clone(), collector);
        }
        
        // Initialize alert rules
        self.monitor.alert_manager.alert_rules.push(AlertRule {
            metric: MonitoringMetric::CoherenceTime,
            condition: AlertCondition::LessThan,
            threshold: 20e-6, // 20 microseconds
            severity: AlertSeverity::Warning,
            message_template: "Coherence time degraded below threshold".to_string(),
        });
        
        self.monitor.alert_manager.alert_rules.push(AlertRule {
            metric: MonitoringMetric::Fidelity,
            condition: AlertCondition::LessThan,
            threshold: 0.95,
            severity: AlertSeverity::Error,
            message_template: "Process fidelity degraded below threshold".to_string(),
        });
        
        Ok(())
    }

    /// Start control loop
    fn start_control_loop(&mut self, _executor: &dyn DDCircuitExecutor) -> DeviceResult<()> {
        println!("Starting DD control loop");
        
        // In a real implementation, this would start a background thread
        // for continuous monitoring and control
        
        Ok(())
    }

    /// Get current system state
    pub fn get_current_state(&self) -> AdaptiveDDState {
        self.state.lock().unwrap().clone()
    }

    /// Get performance history
    pub fn get_performance_history(&self) -> Vec<PerformanceRecord> {
        self.state.lock().unwrap().performance_history.clone()
    }

    /// Get adaptation statistics
    pub fn get_adaptation_statistics(&self) -> AdaptationStatistics {
        let state = self.state.lock().unwrap();
        
        let total_adaptations = state.adaptation_history.len();
        let successful_adaptations = state.adaptation_history
            .iter()
            .filter(|record| record.actual_improvement.unwrap_or(0.0) > 0.0)
            .count();
        
        let success_rate = if total_adaptations > 0 {
            successful_adaptations as f64 / total_adaptations as f64
        } else {
            0.0
        };
        
        let average_improvement = state.adaptation_history
            .iter()
            .filter_map(|record| record.actual_improvement)
            .sum::<f64>() / state.adaptation_history.len().max(1) as f64;
        
        AdaptationStatistics {
            total_adaptations,
            successful_adaptations,
            success_rate,
            average_improvement,
            most_used_sequence: self.get_most_used_sequence(&state),
            adaptation_frequency: self.calculate_adaptation_frequency(&state),
        }
    }

    /// Get most used sequence type
    fn get_most_used_sequence(&self, state: &AdaptiveDDState) -> DDSequenceType {
        let mut sequence_counts: HashMap<DDSequenceType, usize> = HashMap::new();
        
        for record in &state.adaptation_history {
            *sequence_counts.entry(record.new_sequence.clone()).or_insert(0) += 1;
        }
        
        sequence_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(sequence, _)| sequence)
            .unwrap_or(DDSequenceType::CPMG)
    }

    /// Calculate adaptation frequency
    fn calculate_adaptation_frequency(&self, state: &AdaptiveDDState) -> f64 {
        if state.adaptation_history.len() < 2 {
            return 0.0;
        }
        
        let first_adaptation = state.adaptation_history.first().unwrap().timestamp;
        let last_adaptation = state.adaptation_history.last().unwrap().timestamp;
        let time_span = last_adaptation.duration_since(first_adaptation).as_secs_f64();
        
        if time_span > 0.0 {
            state.adaptation_history.len() as f64 / time_span
        } else {
            0.0
        }
    }
}

/// Adaptation statistics
#[derive(Debug, Clone)]
pub struct AdaptationStatistics {
    /// Total number of adaptations
    pub total_adaptations: usize,
    /// Number of successful adaptations
    pub successful_adaptations: usize,
    /// Success rate
    pub success_rate: f64,
    /// Average improvement per adaptation
    pub average_improvement: f64,
    /// Most frequently used sequence
    pub most_used_sequence: DDSequenceType,
    /// Adaptation frequency (adaptations per second)
    pub adaptation_frequency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamical_decoupling::{
        config::DDSequenceType,
        sequences::{DDSequence, DDSequenceProperties, SequenceSymmetry, ResourceRequirements},
    };

    fn create_test_sequence() -> DDSequence {
        DDSequence {
            sequence_type: DDSequenceType::CPMG,
            target_qubits: vec![quantrs2_core::qubit::QubitId(0)],
            duration: 100e-6,
            circuit: quantrs2_circuit::Circuit::<32>::new(),
            pulse_timings: vec![50e-6],
            pulse_phases: vec![std::f64::consts::PI],
            properties: DDSequenceProperties {
                pulse_count: 1,
                sequence_order: 1,
                periodicity: 1,
                symmetry: SequenceSymmetry {
                    time_reversal: true,
                    phase_symmetry: true,
                    rotational_symmetry: false,
                    inversion_symmetry: true,
                },
                noise_suppression: std::collections::HashMap::new(),
                resource_requirements: ResourceRequirements {
                    gate_count: 1,
                    circuit_depth: 1,
                    required_connectivity: Vec::new(),
                    execution_time: 100e-6,
                    memory_requirements: 8,
                },
            },
        }
    }

    #[test]
    fn test_adaptive_dd_system_creation() {
        let config = AdaptiveDDConfig::default();
        let initial_sequence = create_test_sequence();
        let available_sequences = vec![DDSequenceType::CPMG, DDSequenceType::XY4, DDSequenceType::XY8];
        
        let system = AdaptiveDDSystem::new(config, initial_sequence, available_sequences);
        let state = system.get_current_state();
        
        assert_eq!(state.current_sequence.sequence_type, DDSequenceType::CPMG);
        assert_eq!(state.system_health.health_score, 1.0);
    }

    #[test]
    fn test_rule_based_sequence_selection() {
        let config = AdaptiveDDConfig::default();
        let initial_sequence = create_test_sequence();
        let available_sequences = vec![DDSequenceType::CPMG, DDSequenceType::XY4, DDSequenceType::XY8];
        
        let system = AdaptiveDDSystem::new(config, initial_sequence, available_sequences);
        
        let high_noise_state = "coherence_medium_fidelity_medium_noise_high";
        let low_coherence_state = "coherence_low_fidelity_medium_noise_medium";
        let low_fidelity_state = "coherence_medium_fidelity_low_noise_medium";
        
        assert_eq!(system.select_sequence_rule_based(high_noise_state).unwrap(), DDSequenceType::XY8);
        assert_eq!(system.select_sequence_rule_based(low_coherence_state).unwrap(), DDSequenceType::UDD);
        assert_eq!(system.select_sequence_rule_based(low_fidelity_state).unwrap(), DDSequenceType::XY4);
    }

    #[test]
    fn test_adaptation_statistics() {
        let config = AdaptiveDDConfig::default();
        let initial_sequence = create_test_sequence();
        let available_sequences = vec![DDSequenceType::CPMG, DDSequenceType::XY4];
        
        let system = AdaptiveDDSystem::new(config, initial_sequence, available_sequences);
        let stats = system.get_adaptation_statistics();
        
        assert_eq!(stats.total_adaptations, 0);
        assert_eq!(stats.success_rate, 0.0);
        assert_eq!(stats.most_used_sequence, DDSequenceType::CPMG);
    }
}