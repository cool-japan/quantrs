//! Multi-level mitigation coordination

use std::collections::HashMap;
use std::time::Duration;

use super::*;
use crate::DeviceResult;

impl AlertGenerator {
    pub fn new(config: &AlertConfig) -> Self {
        Self {
            thresholds: config.thresholds.clone(),
            alert_history: VecDeque::with_capacity(1000),
            escalation_manager: EscalationManager::new(&config.escalation),
        }
    }

    pub fn check_for_alerts(&mut self, characterization: &CrosstalkCharacterization) -> DeviceResult<Vec<AlertEvent>> {
        let mut alerts = Vec::new();
        let current_time = SystemTime::now();

        // Check crosstalk threshold
        let max_crosstalk = characterization.crosstalk_matrix.mapv(|x| x.abs()).max().unwrap_or(0.0);
        if max_crosstalk > self.thresholds.crosstalk_threshold {
            let alert = AlertEvent {
                timestamp: current_time,
                level: AlertLevel::Warning,
                alert_type: "High Crosstalk".to_string(),
                message: format!("Crosstalk level {:.3} exceeds threshold {:.3}",
                               max_crosstalk, self.thresholds.crosstalk_threshold),
                affected_qubits: self.identify_affected_qubits(characterization),
                recommended_actions: vec!["Apply compensation".to_string(), "Recalibrate".to_string()],
            };
            alerts.push(alert.clone());
            self.alert_history.push_back(alert);
        }

        // Check for instability
        let instability_metric = self.calculate_instability_metric(characterization)?;
        if instability_metric > self.thresholds.instability_threshold {
            let alert = AlertEvent {
                timestamp: current_time,
                level: AlertLevel::Critical,
                alert_type: "System Instability".to_string(),
                message: format!("Instability metric {:.3} exceeds threshold {:.3}",
                               instability_metric, self.thresholds.instability_threshold),
                affected_qubits: vec![], // System-wide issue
                recommended_actions: vec!["Emergency stop".to_string(), "System diagnosis".to_string()],
            };
            alerts.push(alert.clone());
            self.alert_history.push_back(alert);
        }

        // Manage escalation
        self.escalation_manager.update_escalation(&alerts)?;

        // Keep alert history bounded
        while self.alert_history.len() > 1000 {
            self.alert_history.pop_front();
        }

        Ok(alerts)
    }

    fn identify_affected_qubits(&self, characterization: &CrosstalkCharacterization) -> Vec<usize> {
        let mut affected_qubits = Vec::new();
        let threshold = self.thresholds.crosstalk_threshold;

        for i in 0..characterization.crosstalk_matrix.nrows() {
            for j in 0..characterization.crosstalk_matrix.ncols() {
                if characterization.crosstalk_matrix[[i, j]].abs() > threshold {
                    if !affected_qubits.contains(&i) {
                        affected_qubits.push(i);
                    }
                    if !affected_qubits.contains(&j) {
                        affected_qubits.push(j);
                    }
                }
            }
        }

        affected_qubits
    }

    fn calculate_instability_metric(&self, characterization: &CrosstalkCharacterization) -> DeviceResult<f64> {
        // Calculate a metric that indicates system instability
        let matrix = &characterization.crosstalk_matrix;

        // Use matrix norm as instability indicator
        let frobenius_norm = matrix.mapv(|x| x * x).sum().sqrt();

        // Normalize by matrix size
        let normalized_norm = frobenius_norm / (matrix.nrows() as f64 * matrix.ncols() as f64).sqrt();

        Ok(normalized_norm)
    }

    pub fn get_alert_statistics(&self) -> AlertStatistics {
        if self.alert_history.is_empty() {
            return AlertStatistics::default();
        }

        let mut level_counts = HashMap::new();
        let mut type_counts = HashMap::new();

        for alert in &self.alert_history {
            *level_counts.entry(alert.level.clone()).or_insert(0) += 1;
            *type_counts.entry(alert.alert_type.clone()).or_insert(0) += 1;
        }

        let total_alerts = self.alert_history.len();
        let recent_alerts = self.alert_history.iter()
            .rev()
            .take(10)
            .count();

        AlertStatistics {
            total_alerts,
            recent_alerts,
            level_distribution: level_counts,
            type_distribution: type_counts,
        }
    }
}

impl AlertSystem {
    pub fn new(config: &AlertConfig) -> Self {
        Self {
            notification_channels: config.notification_channels.clone(),
            alert_queue: VecDeque::with_capacity(1000),
            notification_history: Vec::new(),
        }
    }

    pub fn send_alert(&mut self, alert: AlertEvent) -> DeviceResult<()> {
        // Add alert to queue
        self.alert_queue.push_back(alert.clone());

        // Send notifications through configured channels
        for channel in &self.notification_channels {
            self.send_notification(channel, &alert)?;
        }

        // Keep queue bounded
        if self.alert_queue.len() > 1000 {
            self.alert_queue.pop_front();
        }

        Ok(())
    }

    fn send_notification(&mut self, channel: &NotificationChannel, alert: &AlertEvent) -> DeviceResult<()> {
        let notification_message = format!(
            "[{}] {}: {} at {:?}",
            alert.level as u8,
            alert.alert_type,
            alert.message,
            alert.timestamp
        );

        match channel {
            NotificationChannel::Log { level } => {
                // Log the alert
                println!("[{}] {}", level, notification_message);
                self.notification_history.push(format!("LOG: {}", notification_message));
            },
            NotificationChannel::Email { recipients } => {
                // Send email (simplified)
                for recipient in recipients {
                    println!("EMAIL to {}: {}", recipient, notification_message);
                    self.notification_history.push(format!("EMAIL to {}: {}", recipient, notification_message));
                }
            },
            NotificationChannel::Slack { webhook_url, channel: slack_channel } => {
                // Send Slack notification (simplified)
                println!("SLACK #{}: {}", slack_channel, notification_message);
                self.notification_history.push(format!("SLACK #{}: {}", slack_channel, notification_message));
            },
            NotificationChannel::Database { table } => {
                // Store in database (simplified)
                println!("DB {}: {}", table, notification_message);
                self.notification_history.push(format!("DB {}: {}", table, notification_message));
            },
            NotificationChannel::WebSocket { endpoint } => {
                // Send via WebSocket (simplified)
                println!("WS {}: {}", endpoint, notification_message);
                self.notification_history.push(format!("WS {}: {}", endpoint, notification_message));
            },
        }

        Ok(())
    }

    pub fn get_pending_alerts(&self) -> &VecDeque<AlertEvent> {
        &self.alert_queue
    }

    pub fn clear_processed_alerts(&mut self, count: usize) {
        for _ in 0..count.min(self.alert_queue.len()) {
            self.alert_queue.pop_front();
        }
    }

    pub fn get_notification_history(&self) -> &Vec<String> {
        &self.notification_history
    }
}

impl EscalationManager {
    pub fn new(config: &AlertEscalation) -> Self {
        Self {
            escalation_levels: config.escalation_levels.clone(),
            current_level: 0,
            escalation_timer: None,
        }
    }

    pub fn update_escalation(&mut self, alerts: &[AlertEvent]) -> DeviceResult<()> {
        if alerts.is_empty() {
            // No alerts, reset escalation
            self.reset_escalation();
            return Ok(());
        }

        // Check if escalation is needed
        let highest_severity = alerts.iter()
            .map(|alert| self.get_alert_severity(&alert.level))
            .max()
            .unwrap_or(0);

        if highest_severity > self.current_level {
            self.escalate_to_level(highest_severity)?;
        }

        // Check escalation timer
        if let Some(timer_start) = self.escalation_timer {
            let escalation_time = Duration::from_secs(30); // Default escalation time
            if timer_start.elapsed().unwrap_or(Duration::ZERO) > escalation_time {
                self.escalate_next_level()?;
            }
        }

        Ok(())
    }

    fn get_alert_severity(&self, level: &AlertLevel) -> usize {
        match level {
            AlertLevel::Info => 0,
            AlertLevel::Warning => 1,
            AlertLevel::Error => 2,
            AlertLevel::Critical => 3,
            AlertLevel::Emergency => 4,
        }
    }

    fn escalate_to_level(&mut self, level: usize) -> DeviceResult<()> {
        if level < self.escalation_levels.len() {
            self.current_level = level;
            self.escalation_timer = Some(SystemTime::now());

            // Execute escalation actions
            let escalation_level = &self.escalation_levels[level];
            self.execute_escalation_actions(escalation_level)?;
        }

        Ok(())
    }

    fn escalate_next_level(&mut self) -> DeviceResult<()> {
        let next_level = self.current_level + 1;
        if next_level < self.escalation_levels.len() {
            self.escalate_to_level(next_level)?;
        }

        Ok(())
    }

    fn execute_escalation_actions(&self, level: &EscalationLevel) -> DeviceResult<()> {
        println!("Executing escalation actions for level: {}", level.level);

        for action in &level.actions {
            match action.as_str() {
                "alert" => {
                    println!("Sending escalated alert");
                },
                "compensate" => {
                    println!("Applying emergency compensation");
                },
                "shutdown" => {
                    println!("Initiating emergency shutdown");
                },
                _ => {
                    println!("Executing action: {}", action);
                },
            }
        }

        Ok(())
    }

    fn reset_escalation(&mut self) {
        self.current_level = 0;
        self.escalation_timer = None;
    }

    pub fn get_current_level(&self) -> usize {
        self.current_level
    }

    pub fn get_escalation_level_name(&self) -> Option<String> {
        if self.current_level < self.escalation_levels.len() {
            Some(self.escalation_levels[self.current_level].level.clone())
        } else {
            None
        }
    }
}

impl MitigationCoordinator {
    pub fn new(config: &MultilevelMitigationConfig) -> Self {
        Self {
            config: config.clone(),
            active_levels: HashMap::new(),
            coordination_strategy: config.coordination_strategy.clone(),
            resource_manager: ResourceManager::new(),
        }
    }

    pub async fn coordinate_mitigation(&mut self, characterization: &CrosstalkCharacterization) -> DeviceResult<MultilevelMitigationResult> {
        // Determine which levels should be active
        let required_levels = self.determine_required_levels(characterization)?;

        // Update active levels
        self.update_active_levels(&required_levels)?;

        // Coordinate between levels based on strategy
        let coordination_effectiveness = self.coordinate_levels(characterization).await?;

        // Calculate level performance
        let level_performance = self.evaluate_level_performance(characterization)?;

        // Calculate resource utilization
        let resource_utilization = self.resource_manager.calculate_utilization(&self.active_levels)?;

        // Calculate overall effectiveness
        let overall_effectiveness = self.calculate_overall_effectiveness(&level_performance);

        Ok(MultilevelMitigationResult {
            active_levels: required_levels,
            level_performance,
            coordination_effectiveness,
            resource_utilization,
            overall_effectiveness,
        })
    }

    fn determine_required_levels(&self, characterization: &CrosstalkCharacterization) -> DeviceResult<Vec<String>> {
        let mut required_levels = Vec::new();

        let crosstalk_strength = characterization.crosstalk_matrix.mapv(|x| x.abs()).mean().unwrap_or(0.0);
        let max_crosstalk = characterization.crosstalk_matrix.mapv(|x| x.abs()).max().unwrap_or(0.0);

        // Determine levels based on crosstalk severity
        for level in &self.config.mitigation_levels {
            let should_activate = match level.name.as_str() {
                "Level1_Fast" => max_crosstalk > 0.05, // Always active for any crosstalk
                "Level2_Accurate" => max_crosstalk > 0.1, // Activate for moderate crosstalk
                "Level3_Comprehensive" => max_crosstalk > 0.2, // Activate for severe crosstalk
                _ => crosstalk_strength > 0.1, // Default threshold
            };

            if should_activate {
                required_levels.push(level.name.clone());
            }
        }

        // Apply level selection strategy
        self.apply_level_selection_strategy(&mut required_levels)?;

        Ok(required_levels)
    }

    fn apply_level_selection_strategy(&self, levels: &mut Vec<String>) -> DeviceResult<()> {
        match &self.config.level_selection {
            LevelSelectionStrategy::Priority => {
                // Sort by priority (lower number = higher priority)
                levels.sort_by_key(|name| {
                    self.config.mitigation_levels.iter()
                        .find(|level| &level.name == name)
                        .map(|level| level.priority)
                        .unwrap_or(999)
                });
            },
            LevelSelectionStrategy::Dynamic { criteria } => {
                // Dynamic selection based on criteria
                self.apply_dynamic_selection(levels, criteria)?;
            },
            LevelSelectionStrategy::Adaptive { selection_criteria } => {
                // Adaptive selection based on performance history
                self.apply_adaptive_selection(levels, selection_criteria)?;
            },
            LevelSelectionStrategy::RoundRobin => {
                // Simple round-robin selection (simplified)
                if levels.len() > 1 {
                    levels.truncate(1); // Use only one level at a time
                }
            },
            LevelSelectionStrategy::LoadBalanced => {
                // Load-balanced selection based on resource availability
                self.apply_load_balanced_selection(levels)?;
            },
        }

        Ok(())
    }

    fn apply_dynamic_selection(&self, levels: &mut Vec<String>, criteria: &[String]) -> DeviceResult<()> {
        // Dynamic selection based on specified criteria
        for criterion in criteria {
            match criterion.as_str() {
                "latency" => {
                    // Prefer lower latency levels
                    levels.sort_by_key(|name| {
                        self.get_level_latency(name).unwrap_or(Duration::MAX)
                    });
                },
                "accuracy" => {
                    // Prefer higher accuracy levels
                    levels.sort_by_key(|name| {
                        -(self.get_level_accuracy(name).unwrap_or(0.0) * 1000.0) as i64
                    });
                },
                "efficiency" => {
                    // Prefer more efficient levels
                    levels.sort_by_key(|name| {
                        -(self.get_level_efficiency(name).unwrap_or(0.0) * 1000.0) as i64
                    });
                },
                _ => {
                    // Unknown criterion, use default ordering
                }
            }
        }

        Ok(())
    }

    fn apply_adaptive_selection(&self, levels: &mut Vec<String>, criteria: &[String]) -> DeviceResult<()> {
        // Adaptive selection based on historical performance
        // Simplified implementation
        self.apply_dynamic_selection(levels, criteria)
    }

    fn apply_load_balanced_selection(&self, levels: &mut Vec<String>) -> DeviceResult<()> {
        // Select levels based on available resources
        levels.retain(|name| {
            self.resource_manager.can_allocate_for_level(name).unwrap_or(false)
        });

        Ok(())
    }

    fn get_level_latency(&self, level_name: &str) -> Option<Duration> {
        self.config.mitigation_levels.iter()
            .find(|level| level.name == level_name)
            .map(|level| level.performance_targets.max_latency)
    }

    fn get_level_accuracy(&self, level_name: &str) -> Option<f64> {
        self.config.mitigation_levels.iter()
            .find(|level| level.name == level_name)
            .map(|level| level.performance_targets.crosstalk_reduction)
    }

    fn get_level_efficiency(&self, level_name: &str) -> Option<f64> {
        // Calculate efficiency as accuracy / resource_cost
        let level = self.config.mitigation_levels.iter()
            .find(|level| level.name == level_name)?;

        let accuracy = level.performance_targets.crosstalk_reduction;
        let cost = level.resource_requirements.computational_complexity;

        if cost > 0.0 {
            Some(accuracy / cost)
        } else {
            Some(accuracy)
        }
    }

    fn update_active_levels(&mut self, required_levels: &[String]) -> DeviceResult<()> {
        // Clear current active levels
        self.active_levels.clear();

        // Activate required levels
        for level_name in required_levels {
            self.active_levels.insert(level_name.clone(), true);
        }

        Ok(())
    }

    async fn coordinate_levels(&mut self, characterization: &CrosstalkCharacterization) -> DeviceResult<f64> {
        match &self.coordination_strategy {
            CoordinationStrategy::Sequential => {
                self.coordinate_sequential(characterization).await
            },
            CoordinationStrategy::Parallel => {
                self.coordinate_parallel(characterization).await
            },
            CoordinationStrategy::Hierarchical { control_hierarchy } => {
                self.coordinate_hierarchical(characterization, control_hierarchy).await
            },
            CoordinationStrategy::Adaptive { coordination_algorithm } => {
                self.coordinate_adaptive(characterization, coordination_algorithm).await
            },
        }
    }

    async fn coordinate_sequential(&mut self, characterization: &CrosstalkCharacterization) -> DeviceResult<f64> {
        // Sequential coordination: apply levels one after another
        let mut effectiveness = 0.0;
        let mut remaining_crosstalk = characterization.crosstalk_matrix.clone();

        for (level_name, &is_active) in &self.active_levels {
            if is_active {
                let level_effectiveness = self.apply_level_mitigation(level_name, &remaining_crosstalk)?;
                effectiveness += level_effectiveness;

                // Update remaining crosstalk (simplified)
                remaining_crosstalk = &remaining_crosstalk * (1.0 - level_effectiveness);
            }
        }

        Ok(effectiveness.min(1.0))
    }

    async fn coordinate_parallel(&mut self, characterization: &CrosstalkCharacterization) -> DeviceResult<f64> {
        // Parallel coordination: apply all levels simultaneously
        let mut total_effectiveness = 0.0;
        let mut level_count = 0;

        for (level_name, &is_active) in &self.active_levels {
            if is_active {
                let level_effectiveness = self.apply_level_mitigation(level_name, &characterization.crosstalk_matrix)?;
                total_effectiveness += level_effectiveness;
                level_count += 1;
            }
        }

        // Average effectiveness of parallel levels
        let coordination_effectiveness = if level_count > 0 {
            total_effectiveness / level_count as f64
        } else {
            0.0
        };

        Ok(coordination_effectiveness)
    }

    async fn coordinate_hierarchical(&mut self, characterization: &CrosstalkCharacterization, hierarchy: &[String]) -> DeviceResult<f64> {
        // Hierarchical coordination: levels in order of hierarchy
        let mut effectiveness = 0.0;
        let mut remaining_crosstalk = characterization.crosstalk_matrix.clone();

        for level_name in hierarchy {
            if self.active_levels.get(level_name).unwrap_or(&false) {
                let level_effectiveness = self.apply_level_mitigation(level_name, &remaining_crosstalk)?;
                effectiveness += level_effectiveness * 0.9_f64.powi(effectiveness as i32); // Diminishing returns

                // Update remaining crosstalk
                remaining_crosstalk = &remaining_crosstalk * (1.0 - level_effectiveness);
            }
        }

        Ok(effectiveness.min(1.0))
    }

    async fn coordinate_adaptive(&mut self, characterization: &CrosstalkCharacterization, algorithm: &str) -> DeviceResult<f64> {
        // Adaptive coordination based on specified algorithm
        match algorithm {
            "reinforcement_learning" => {
                // Simplified RL-based coordination
                self.coordinate_parallel(characterization).await
            },
            "genetic_algorithm" => {
                // Simplified GA-based coordination
                self.coordinate_hierarchical(characterization, &self.get_default_hierarchy()).await
            },
            _ => {
                // Default to sequential coordination
                self.coordinate_sequential(characterization).await
            }
        }
    }

    fn get_default_hierarchy(&self) -> Vec<String> {
        let mut levels: Vec<_> = self.active_levels.keys().cloned().collect();
        levels.sort();
        levels
    }

    fn apply_level_mitigation(&self, level_name: &str, crosstalk_matrix: &scirs2_core::ndarray::Array2<f64>) -> DeviceResult<f64> {
        // Apply mitigation for a specific level
        let level = self.config.mitigation_levels.iter()
            .find(|level| level.name == level_name);

        if let Some(level) = level {
            // Calculate effectiveness based on level's performance targets
            let target_reduction = level.performance_targets.crosstalk_reduction;
            let current_crosstalk = crosstalk_matrix.mapv(|x| x.abs()).mean().unwrap_or(0.0);

            // Simplified effectiveness calculation
            let effectiveness = (target_reduction * (1.0 - current_crosstalk)).min(target_reduction);
            Ok(effectiveness)
        } else {
            Ok(0.0)
        }
    }

    fn evaluate_level_performance(&self, characterization: &CrosstalkCharacterization) -> DeviceResult<HashMap<String, LevelPerformance>> {
        let mut performance_map = HashMap::new();

        for (level_name, &is_active) in &self.active_levels {
            if is_active {
                let level = self.config.mitigation_levels.iter()
                    .find(|level| level.name == level_name);

                if let Some(level) = level {
                    let effectiveness = self.apply_level_mitigation(level_name, &characterization.crosstalk_matrix)?;
                    let resource_usage = self.calculate_level_resource_usage(level);
                    let response_time = level.performance_targets.max_latency;
                    let stability = level.performance_targets.reliability;

                    performance_map.insert(level_name.clone(), LevelPerformance {
                        effectiveness,
                        resource_usage,
                        response_time,
                        stability,
                    });
                }
            }
        }

        Ok(performance_map)
    }

    fn calculate_level_resource_usage(&self, level: &MitigationLevel) -> f64 {
        // Calculate resource usage as fraction of available resources
        let computational_usage = level.resource_requirements.computational_complexity / 10.0; // Normalize
        let memory_usage = level.resource_requirements.memory_mb as f64 / 1024.0; // Normalize to GB

        (computational_usage + memory_usage) / 2.0
    }

    fn calculate_overall_effectiveness(&self, level_performance: &HashMap<String, LevelPerformance>) -> f64 {
        if level_performance.is_empty() {
            return 0.0;
        }

        // Calculate weighted average effectiveness
        let total_effectiveness: f64 = level_performance.values()
            .map(|perf| perf.effectiveness * perf.stability)
            .sum();

        total_effectiveness / level_performance.len() as f64
    }

    pub fn get_active_levels(&self) -> Vec<String> {
        self.active_levels.iter()
            .filter(|(_, &is_active)| is_active)
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn get_coordination_status(&self) -> CoordinationStatus {
        let active_count = self.active_levels.values().filter(|&&is_active| is_active).count();

        CoordinationStatus {
            active_level_count: active_count,
            coordination_strategy: self.coordination_strategy.clone(),
            resource_utilization: self.resource_manager.get_current_utilization(),
            last_update: SystemTime::now(),
        }
    }
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            available_resources: ResourceRequirements {
                computational_complexity: 10.0,
                memory_mb: 1024,
                realtime_constraints: Duration::from_millis(1),
                hardware_requirements: vec!["CPU".to_string(), "GPU".to_string()],
            },
            allocated_resources: HashMap::new(),
            optimization_targets: PerformanceTargets {
                crosstalk_reduction: 0.8,
                fidelity_improvement: 0.2,
                max_latency: Duration::from_millis(10),
                reliability: 0.95,
            },
        }
    }

    pub fn calculate_utilization(&mut self, active_levels: &HashMap<String, bool>) -> DeviceResult<ResourceUtilizationResult> {
        let mut cpu_utilization = 0.0;
        let mut memory_utilization = 0.0;
        let mut computation_time = Duration::ZERO;
        let mut hardware_utilization = HashMap::new();

        // Calculate utilization based on active levels
        for (level_name, &is_active) in active_levels {
            if is_active {
                // Simplified resource calculation
                cpu_utilization += 0.1; // Each level uses 10% CPU
                memory_utilization += 0.05; // Each level uses 5% memory
                computation_time += Duration::from_millis(5); // Each level adds 5ms

                hardware_utilization.insert(level_name.clone(), 0.1);
            }
        }

        Ok(ResourceUtilizationResult {
            cpu_utilization: cpu_utilization.min(1.0),
            memory_utilization: memory_utilization.min(1.0),
            computation_time,
            hardware_utilization,
        })
    }

    pub fn can_allocate_for_level(&self, level_name: &str) -> Option<bool> {
        // Check if resources are available for the specified level
        let current_cpu = self.get_current_cpu_usage();
        let current_memory = self.get_current_memory_usage();

        // Simplified resource check
        Some(current_cpu < 0.8 && current_memory < 0.8)
    }

    fn get_current_cpu_usage(&self) -> f64 {
        // Simplified: calculate current CPU usage
        self.allocated_resources.len() as f64 * 0.1
    }

    fn get_current_memory_usage(&self) -> f64 {
        // Simplified: calculate current memory usage
        self.allocated_resources.len() as f64 * 0.05
    }

    pub fn get_current_utilization(&self) -> f64 {
        // Return overall resource utilization
        let cpu_usage = self.get_current_cpu_usage();
        let memory_usage = self.get_current_memory_usage();

        (cpu_usage + memory_usage) / 2.0
    }

    pub fn allocate_resources(&mut self, level_name: String, requirements: ResourceRequirements) -> DeviceResult<()> {
        // Allocate resources for a level
        if self.can_allocate_for_level(&level_name).unwrap_or(false) {
            self.allocated_resources.insert(level_name, requirements);
            Ok(())
        } else {
            Err(crate::DeviceError::InsufficientResources)
        }
    }

    pub fn deallocate_resources(&mut self, level_name: &str) -> DeviceResult<()> {
        // Deallocate resources for a level
        self.allocated_resources.remove(level_name);
        Ok(())
    }

    pub fn optimize_allocation(&mut self) -> DeviceResult<()> {
        // Optimize resource allocation based on performance targets
        // Simplified implementation

        // Sort levels by efficiency
        let mut levels: Vec<_> = self.allocated_resources.keys().cloned().collect();
        levels.sort_by_key(|name| {
            // Sort by inverse complexity (prefer simpler levels)
            self.allocated_resources.get(name)
                .map(|req| -(req.computational_complexity * 1000.0) as i64)
                .unwrap_or(0)
        });

        // Rebuild allocation in optimized order
        let old_allocation = self.allocated_resources.clone();
        self.allocated_resources.clear();

        for level_name in levels {
            if let Some(requirements) = old_allocation.get(&level_name) {
                self.allocated_resources.insert(level_name, requirements.clone());
            }
        }

        Ok(())
    }
}

// Helper structs for multi-level coordination
#[derive(Debug, Clone)]
pub struct AlertStatistics {
    pub total_alerts: usize,
    pub recent_alerts: usize,
    pub level_distribution: HashMap<AlertLevel, usize>,
    pub type_distribution: HashMap<String, usize>,
}

impl Default for AlertStatistics {
    fn default() -> Self {
        Self {
            total_alerts: 0,
            recent_alerts: 0,
            level_distribution: HashMap::new(),
            type_distribution: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoordinationStatus {
    pub active_level_count: usize,
    pub coordination_strategy: CoordinationStrategy,
    pub resource_utilization: f64,
    pub last_update: SystemTime,
}