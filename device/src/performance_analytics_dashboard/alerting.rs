//! Alerting System for Performance Dashboard
//!
//! This module handles alert management, notification dispatch, and escalation
//! for the performance analytics dashboard.

use super::analytics::Anomaly;
use super::config::{
    AlertConfig, AlertSeverity, AlertThreshold, ChannelType, EscalationPolicy, NotificationChannel,
    SuppressionRule, ThresholdType,
};
use crate::DeviceResult;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;

/// Alert manager for handling all alerting functionality
pub struct AlertManager {
    config: AlertConfig,
    active_alerts: HashMap<String, ActiveAlert>,
    alert_history: VecDeque<ResolvedAlert>,
    notification_dispatcher: NotificationDispatcher,
    suppression_engine: SuppressionEngine,
    escalation_engine: EscalationEngine,
}

/// Active alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    pub alert_id: String,
    pub timestamp: SystemTime,
    pub metric_name: String,
    pub threshold: AlertThreshold,
    pub current_value: f64,
    pub severity: AlertSeverity,
    pub status: AlertStatus,
    pub acknowledgement: Option<AlertAcknowledgement>,
    pub escalation_level: usize,
    pub notification_count: usize,
    pub tags: HashMap<String, String>,
}

/// Alert status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertStatus {
    Triggered,
    Acknowledged,
    Escalated,
    Resolved,
    Suppressed,
}

/// Alert acknowledgement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertAcknowledgement {
    pub acknowledged_by: String,
    pub acknowledgement_time: SystemTime,
    pub acknowledgement_note: Option<String>,
    pub estimated_resolution_time: Option<SystemTime>,
}

/// Resolved alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedAlert {
    pub alert: ActiveAlert,
    pub resolution_timestamp: SystemTime,
    pub resolution_method: ResolutionMethod,
    pub duration: Duration,
    pub resolution_note: Option<String>,
    pub resolution_effectiveness: Option<f64>,
}

/// Resolution methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResolutionMethod {
    Automatic,
    Manual,
    SelfHealing,
    UserIntervention,
    SystemRestart,
    ConfigurationChange,
    Timeout,
}

/// Alert statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStatistics {
    pub total_alerts: usize,
    pub alerts_by_severity: HashMap<AlertSeverity, usize>,
    pub alerts_by_metric: HashMap<String, usize>,
    pub average_resolution_time: Duration,
    pub false_positive_rate: f64,
    pub escalation_rate: f64,
    pub acknowledgement_rate: f64,
}

/// Alert trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertTrends {
    pub alert_frequency_trend: TrendDirection,
    pub severity_trends: HashMap<AlertSeverity, TrendDirection>,
    pub resolution_time_trend: TrendDirection,
    pub false_positive_trend: TrendDirection,
}

/// Trend direction (placeholder from analytics)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Notification dispatcher for sending alerts
#[derive(Debug)]
pub struct NotificationDispatcher {
    channels: Vec<NotificationChannel>,
    notification_queue: VecDeque<NotificationTask>,
    delivery_history: VecDeque<DeliveryRecord>,
    rate_limiter: RateLimiter,
}

/// Notification task
#[derive(Debug, Clone)]
pub struct NotificationTask {
    pub task_id: String,
    pub alert_id: String,
    pub channel_type: ChannelType,
    pub message: NotificationMessage,
    pub priority: NotificationPriority,
    pub retry_count: usize,
    pub max_retries: usize,
    pub created_at: SystemTime,
}

/// Notification message
#[derive(Debug, Clone)]
pub struct NotificationMessage {
    pub subject: String,
    pub body: String,
    pub format: MessageFormat,
    pub attachments: Vec<NotificationAttachment>,
    pub metadata: HashMap<String, String>,
}

/// Message formats
#[derive(Debug, Clone, PartialEq)]
pub enum MessageFormat {
    PlainText,
    HTML,
    Markdown,
    JSON,
    Custom(String),
}

/// Notification attachment
#[derive(Debug, Clone)]
pub struct NotificationAttachment {
    pub name: String,
    pub content_type: String,
    pub data: Vec<u8>,
}

/// Notification priority
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum NotificationPriority {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
}

/// Delivery record
#[derive(Debug, Clone)]
pub struct DeliveryRecord {
    pub task_id: String,
    pub channel_type: ChannelType,
    pub delivery_time: SystemTime,
    pub delivery_status: DeliveryStatus,
    pub response_time: Duration,
    pub error_message: Option<String>,
}

/// Delivery status
#[derive(Debug, Clone, PartialEq)]
pub enum DeliveryStatus {
    Sent,
    Delivered,
    Failed,
    Retry,
    Suppressed,
}

/// Rate limiter for notifications
#[derive(Debug)]
pub struct RateLimiter {
    limits: HashMap<ChannelType, RateLimit>,
    usage_tracking: HashMap<ChannelType, VecDeque<SystemTime>>,
}

/// Rate limit configuration
#[derive(Debug, Clone)]
pub struct RateLimit {
    pub max_notifications: usize,
    pub time_window: Duration,
    pub burst_limit: Option<usize>,
}

/// Suppression engine for reducing noise
pub struct SuppressionEngine {
    rules: Vec<SuppressionRule>,
    suppressed_alerts: HashMap<String, SuppressionRecord>,
    suppression_history: VecDeque<SuppressionEvent>,
}

/// Suppression record
#[derive(Debug, Clone)]
pub struct SuppressionRecord {
    pub alert_id: String,
    pub rule_id: String,
    pub suppression_start: SystemTime,
    pub suppression_end: SystemTime,
    pub suppression_count: usize,
}

/// Suppression event
#[derive(Debug, Clone)]
pub struct SuppressionEvent {
    pub timestamp: SystemTime,
    pub event_type: SuppressionEventType,
    pub alert_id: String,
    pub rule_id: String,
    pub details: HashMap<String, String>,
}

/// Suppression event types
#[derive(Debug, Clone, PartialEq)]
pub enum SuppressionEventType {
    AlertSuppressed,
    AlertUnsuppressed,
    RuleActivated,
    RuleDeactivated,
    SuppressionExpired,
}

/// Escalation engine for managing alert escalations
pub struct EscalationEngine {
    policies: Vec<EscalationPolicy>,
    active_escalations: HashMap<String, EscalationState>,
    escalation_history: VecDeque<EscalationEvent>,
}

/// Escalation state
#[derive(Debug, Clone)]
pub struct EscalationState {
    pub alert_id: String,
    pub policy_id: String,
    pub current_step: usize,
    pub escalation_start: SystemTime,
    pub next_escalation: Option<SystemTime>,
    pub escalation_attempts: usize,
}

/// Escalation event
#[derive(Debug, Clone)]
pub struct EscalationEvent {
    pub timestamp: SystemTime,
    pub alert_id: String,
    pub policy_id: String,
    pub step_number: usize,
    pub escalation_type: EscalationType,
    pub success: bool,
    pub details: HashMap<String, String>,
}

/// Escalation types
#[derive(Debug, Clone, PartialEq)]
pub enum EscalationType {
    Notification,
    AutoRemediation,
    TicketCreation,
    OnCallEscalation,
    Custom(String),
}

/// Notification filter
#[derive(Debug, Clone)]
pub struct NotificationFilter {
    pub filter_type: String,
    pub condition: String,
    pub value: String,
}

impl AlertManager {
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config: config.clone(),
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
            notification_dispatcher: NotificationDispatcher::new(
                config.notification_channels.clone(),
            ),
            suppression_engine: SuppressionEngine::new(config.suppression_rules.clone()),
            escalation_engine: EscalationEngine::new(config.escalation_policies.clone()),
        }
    }

    pub async fn start_monitoring(&mut self) -> DeviceResult<()> {
        // Initialize monitoring components
        self.notification_dispatcher.start().await?;
        self.suppression_engine.start().await?;
        self.escalation_engine.start().await?;

        Ok(())
    }

    pub async fn stop_monitoring(&mut self) -> DeviceResult<()> {
        // Stop monitoring components
        self.notification_dispatcher.stop().await?;
        self.suppression_engine.stop().await?;
        self.escalation_engine.stop().await?;

        Ok(())
    }

    pub async fn process_metric_value(
        &mut self,
        metric_name: &str,
        value: f64,
    ) -> DeviceResult<()> {
        let threshold = self.config.alert_thresholds.get(metric_name).cloned();
        if let Some(threshold) = threshold {
            if self.should_trigger_alert(&threshold, value) {
                self.trigger_alert(metric_name, &threshold, value).await?;
            } else {
                let alert_id = self
                    .active_alerts
                    .get(metric_name)
                    .map(|a| a.alert_id.clone());
                if let Some(alert_id) = alert_id {
                    if self.should_resolve_alert(&threshold, value) {
                        self.resolve_alert(&alert_id, ResolutionMethod::Automatic)
                            .await?;
                    }
                }
            }
        }

        Ok(())
    }

    pub async fn process_anomaly(&mut self, anomaly: &Anomaly) -> DeviceResult<()> {
        // Create alert from anomaly
        let alert_threshold = AlertThreshold {
            metric_name: anomaly.metric_name.clone(),
            threshold_type: ThresholdType::StandardDeviation,
            value: anomaly.expected_value,
            severity: self.map_anomaly_severity(&anomaly.severity),
            duration: Duration::from_secs(60),
            enabled: true,
        };

        self.trigger_alert(
            &anomaly.metric_name,
            &alert_threshold,
            anomaly.current_value,
        )
        .await?;

        Ok(())
    }

    pub async fn acknowledge_alert(
        &mut self,
        alert_id: &str,
        user_id: &str,
        note: Option<String>,
    ) -> DeviceResult<()> {
        if let Some(alert) = self.active_alerts.get_mut(alert_id) {
            alert.acknowledgement = Some(AlertAcknowledgement {
                acknowledged_by: user_id.to_string(),
                acknowledgement_time: SystemTime::now(),
                acknowledgement_note: note,
                estimated_resolution_time: None,
            });
            alert.status = AlertStatus::Acknowledged;

            // Stop escalation for acknowledged alerts
            self.escalation_engine.stop_escalation(alert_id).await?;
        }

        Ok(())
    }

    pub async fn resolve_alert(
        &mut self,
        alert_id: &str,
        resolution_method: ResolutionMethod,
    ) -> DeviceResult<()> {
        if let Some(alert) = self.active_alerts.remove(alert_id) {
            let duration = SystemTime::now()
                .duration_since(alert.timestamp)
                .unwrap_or(Duration::from_secs(0));

            let resolved_alert = ResolvedAlert {
                alert: alert.clone(),
                resolution_timestamp: SystemTime::now(),
                resolution_method,
                duration,
                resolution_note: None,
                resolution_effectiveness: None,
            };

            self.alert_history.push_back(resolved_alert);

            // Keep only last 10000 resolved alerts
            if self.alert_history.len() > 10000 {
                self.alert_history.pop_front();
            }

            // Stop escalation
            self.escalation_engine.stop_escalation(alert_id).await?;
        }

        Ok(())
    }

    pub fn get_alert_statistics(&self) -> AlertStatistics {
        let mut stats = AlertStatistics {
            total_alerts: self.active_alerts.len() + self.alert_history.len(),
            alerts_by_severity: HashMap::new(),
            alerts_by_metric: HashMap::new(),
            average_resolution_time: Duration::from_secs(0),
            false_positive_rate: 0.0,
            escalation_rate: 0.0,
            acknowledgement_rate: 0.0,
        };

        // Calculate statistics from active and resolved alerts
        for alert in self.active_alerts.values() {
            *stats
                .alerts_by_severity
                .entry(alert.severity.clone())
                .or_insert(0) += 1;
            *stats
                .alerts_by_metric
                .entry(alert.metric_name.clone())
                .or_insert(0) += 1;
        }

        for resolved in &self.alert_history {
            *stats
                .alerts_by_severity
                .entry(resolved.alert.severity.clone())
                .or_insert(0) += 1;
            *stats
                .alerts_by_metric
                .entry(resolved.alert.metric_name.clone())
                .or_insert(0) += 1;
        }

        // Calculate average resolution time
        if !self.alert_history.is_empty() {
            let total_duration: Duration = self.alert_history.iter().map(|r| r.duration).sum();
            stats.average_resolution_time = total_duration / self.alert_history.len() as u32;
        }

        stats
    }

    async fn trigger_alert(
        &mut self,
        metric_name: &str,
        threshold: &AlertThreshold,
        value: f64,
    ) -> DeviceResult<()> {
        let alert_id = format!(
            "{}-{}",
            metric_name,
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );

        let alert = ActiveAlert {
            alert_id: alert_id.clone(),
            timestamp: SystemTime::now(),
            metric_name: metric_name.to_string(),
            threshold: threshold.clone(),
            current_value: value,
            severity: threshold.severity.clone(),
            status: AlertStatus::Triggered,
            acknowledgement: None,
            escalation_level: 0,
            notification_count: 0,
            tags: HashMap::new(),
        };

        // Check suppression
        if !self.suppression_engine.should_suppress(&alert).await? {
            // Send notification
            self.notification_dispatcher
                .send_alert_notification(&alert)
                .await?;

            // Start escalation if configured
            self.escalation_engine.start_escalation(&alert).await?;

            self.active_alerts.insert(alert_id, alert);
        }

        Ok(())
    }

    fn should_trigger_alert(&self, threshold: &AlertThreshold, value: f64) -> bool {
        if !threshold.enabled {
            return false;
        }

        match threshold.threshold_type {
            ThresholdType::GreaterThan => value > threshold.value,
            ThresholdType::LessThan => value < threshold.value,
            ThresholdType::Equal => (value - threshold.value).abs() < f64::EPSILON,
            ThresholdType::NotEqual => (value - threshold.value).abs() > f64::EPSILON,
            ThresholdType::PercentageChange => {
                // Implementation would need historical data
                false
            }
            ThresholdType::StandardDeviation => {
                // Implementation would need baseline statistics
                false
            }
            ThresholdType::Custom(_) => {
                // Custom threshold logic
                false
            }
        }
    }

    fn should_resolve_alert(&self, threshold: &AlertThreshold, value: f64) -> bool {
        // Resolve if the condition is no longer met
        !self.should_trigger_alert(threshold, value)
    }

    fn map_anomaly_severity(
        &self,
        anomaly_severity: &super::analytics::AnomalySeverity,
    ) -> AlertSeverity {
        match anomaly_severity {
            super::analytics::AnomalySeverity::Low => AlertSeverity::Info,
            super::analytics::AnomalySeverity::Medium => AlertSeverity::Warning,
            super::analytics::AnomalySeverity::High => AlertSeverity::Error,
            super::analytics::AnomalySeverity::Critical => AlertSeverity::Critical,
        }
    }
}

impl NotificationDispatcher {
    pub fn new(channels: Vec<NotificationChannel>) -> Self {
        Self {
            channels,
            notification_queue: VecDeque::new(),
            delivery_history: VecDeque::new(),
            rate_limiter: RateLimiter::new(),
        }
    }

    pub async fn start(&mut self) -> DeviceResult<()> {
        // Start notification processing
        Ok(())
    }

    pub async fn stop(&mut self) -> DeviceResult<()> {
        // Stop notification processing
        Ok(())
    }

    pub async fn send_alert_notification(&mut self, alert: &ActiveAlert) -> DeviceResult<()> {
        for channel in &self.channels {
            if !channel.enabled {
                continue;
            }

            if self.rate_limiter.should_rate_limit(&channel.channel_type) {
                continue;
            }

            let message = self.format_alert_message(alert, &channel.channel_type);
            let task = NotificationTask {
                task_id: format!("{}-{:?}", alert.alert_id, channel.channel_type),
                alert_id: alert.alert_id.clone(),
                channel_type: channel.channel_type.clone(),
                message,
                priority: self.map_severity_to_priority(&alert.severity),
                retry_count: 0,
                max_retries: 3,
                created_at: SystemTime::now(),
            };

            self.notification_queue.push_back(task);
        }

        self.process_notification_queue().await?;
        Ok(())
    }

    async fn process_notification_queue(&mut self) -> DeviceResult<()> {
        while let Some(task) = self.notification_queue.pop_front() {
            let delivery_status = self.deliver_notification(&task).await?;

            if delivery_status == DeliveryStatus::Failed && task.retry_count < task.max_retries {
                let mut retry_task = task.clone();
                retry_task.retry_count += 1;
                self.notification_queue.push_back(retry_task);
            }

            self.record_delivery(&task, delivery_status);
        }

        Ok(())
    }

    async fn deliver_notification(&self, task: &NotificationTask) -> DeviceResult<DeliveryStatus> {
        // Simplified delivery implementation
        match task.channel_type {
            ChannelType::Email => {
                // Email delivery logic
                Ok(DeliveryStatus::Sent)
            }
            ChannelType::Slack => {
                // Slack delivery logic
                Ok(DeliveryStatus::Sent)
            }
            ChannelType::Webhook => {
                // Webhook delivery logic
                Ok(DeliveryStatus::Sent)
            }
            _ => Ok(DeliveryStatus::Failed),
        }
    }

    fn format_alert_message(
        &self,
        alert: &ActiveAlert,
        channel_type: &ChannelType,
    ) -> NotificationMessage {
        let subject = format!(
            "[{}] Alert: {} threshold exceeded",
            alert.severity as i32, alert.metric_name
        );
        let body = format!(
            "Metric: {}\nCurrent Value: {:.4}\nThreshold: {:.4}\nSeverity: {:?}\nTime: {:?}",
            alert.metric_name,
            alert.current_value,
            alert.threshold.value,
            alert.severity,
            alert.timestamp
        );

        let format = match channel_type {
            ChannelType::Email => MessageFormat::HTML,
            ChannelType::Slack => MessageFormat::Markdown,
            _ => MessageFormat::PlainText,
        };

        NotificationMessage {
            subject,
            body,
            format,
            attachments: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    fn map_severity_to_priority(&self, severity: &AlertSeverity) -> NotificationPriority {
        match severity {
            AlertSeverity::Info => NotificationPriority::Low,
            AlertSeverity::Warning => NotificationPriority::Normal,
            AlertSeverity::Error => NotificationPriority::High,
            AlertSeverity::Critical => NotificationPriority::Critical,
        }
    }

    fn record_delivery(&mut self, task: &NotificationTask, status: DeliveryStatus) {
        let record = DeliveryRecord {
            task_id: task.task_id.clone(),
            channel_type: task.channel_type.clone(),
            delivery_time: SystemTime::now(),
            delivery_status: status,
            response_time: Duration::from_millis(100), // Simplified
            error_message: None,
        };

        self.delivery_history.push_back(record);

        // Keep only last 10000 records
        if self.delivery_history.len() > 10000 {
            self.delivery_history.pop_front();
        }
    }
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            limits: HashMap::new(),
            usage_tracking: HashMap::new(),
        }
    }

    pub fn should_rate_limit(&mut self, channel_type: &ChannelType) -> bool {
        let limit = self.limits.get(channel_type).cloned().unwrap_or(RateLimit {
            max_notifications: 100,
            time_window: Duration::from_secs(3600),
            burst_limit: Some(10),
        });

        let now = SystemTime::now();
        let usage = self
            .usage_tracking
            .entry(channel_type.clone())
            .or_insert_with(VecDeque::new);

        // Remove old entries outside the time window
        while let Some(front) = usage.front() {
            if now.duration_since(*front).unwrap_or(Duration::from_secs(0)) > limit.time_window {
                usage.pop_front();
            } else {
                break;
            }
        }

        // Check if we're within limits
        if usage.len() >= limit.max_notifications {
            return true; // Rate limited
        }

        // Record this usage
        usage.push_back(now);
        false // Not rate limited
    }
}

impl SuppressionEngine {
    pub fn new(rules: Vec<SuppressionRule>) -> Self {
        Self {
            rules,
            suppressed_alerts: HashMap::new(),
            suppression_history: VecDeque::new(),
        }
    }

    pub async fn start(&mut self) -> DeviceResult<()> {
        // Start suppression processing
        Ok(())
    }

    pub async fn stop(&mut self) -> DeviceResult<()> {
        // Stop suppression processing
        Ok(())
    }

    pub async fn should_suppress(&mut self, alert: &ActiveAlert) -> DeviceResult<bool> {
        for rule in self.rules.clone() {
            if !rule.enabled {
                continue;
            }

            if self.matches_suppression_rule(alert, &rule) {
                self.apply_suppression(alert, &rule).await?;
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn matches_suppression_rule(&self, _alert: &ActiveAlert, _rule: &SuppressionRule) -> bool {
        // Simplified rule matching logic
        false
    }

    async fn apply_suppression(
        &mut self,
        alert: &ActiveAlert,
        rule: &SuppressionRule,
    ) -> DeviceResult<()> {
        let record = SuppressionRecord {
            alert_id: alert.alert_id.clone(),
            rule_id: rule.rule_name.clone(),
            suppression_start: SystemTime::now(),
            suppression_end: SystemTime::now() + rule.duration,
            suppression_count: 1,
        };

        self.suppressed_alerts
            .insert(alert.alert_id.clone(), record);

        let event = SuppressionEvent {
            timestamp: SystemTime::now(),
            event_type: SuppressionEventType::AlertSuppressed,
            alert_id: alert.alert_id.clone(),
            rule_id: rule.rule_name.clone(),
            details: HashMap::new(),
        };

        self.suppression_history.push_back(event);

        Ok(())
    }
}

impl EscalationEngine {
    pub fn new(policies: Vec<EscalationPolicy>) -> Self {
        Self {
            policies,
            active_escalations: HashMap::new(),
            escalation_history: VecDeque::new(),
        }
    }

    pub async fn start(&mut self) -> DeviceResult<()> {
        // Initialize escalation engine
        Ok(())
    }

    pub async fn stop(&mut self) -> DeviceResult<()> {
        // Stop escalation engine and clear active escalations
        self.active_escalations.clear();
        Ok(())
    }

    pub async fn start_escalation(&mut self, alert: &ActiveAlert) -> DeviceResult<()> {
        // Find applicable escalation policy
        if let Some(policy) = self.find_escalation_policy(alert) {
            let state = EscalationState {
                alert_id: alert.alert_id.clone(),
                policy_id: policy.policy_name.clone(),
                current_step: 0,
                escalation_start: SystemTime::now(),
                next_escalation: Some(SystemTime::now() + Duration::from_secs(15 * 60)), // First escalation in 15 minutes
                escalation_attempts: 0,
            };

            self.active_escalations
                .insert(alert.alert_id.clone(), state);
        }

        Ok(())
    }

    pub async fn stop_escalation(&mut self, alert_id: &str) -> DeviceResult<()> {
        if let Some(_state) = self.active_escalations.remove(alert_id) {
            // Record escalation stop event
            let event = EscalationEvent {
                timestamp: SystemTime::now(),
                alert_id: alert_id.to_string(),
                policy_id: String::new(),
                step_number: 0,
                escalation_type: EscalationType::Notification,
                success: true,
                details: HashMap::new(),
            };

            self.escalation_history.push_back(event);
        }

        Ok(())
    }

    fn find_escalation_policy(&self, _alert: &ActiveAlert) -> Option<&EscalationPolicy> {
        // Simplified policy selection
        self.policies.first()
    }
}

// Default implementations and helper functions
impl Default for AlertStatistics {
    fn default() -> Self {
        Self {
            total_alerts: 0,
            alerts_by_severity: HashMap::new(),
            alerts_by_metric: HashMap::new(),
            average_resolution_time: Duration::from_secs(0),
            false_positive_rate: 0.0,
            escalation_rate: 0.0,
            acknowledgement_rate: 0.0,
        }
    }
}

impl Default for AlertTrends {
    fn default() -> Self {
        Self {
            alert_frequency_trend: TrendDirection::Stable,
            severity_trends: HashMap::new(),
            resolution_time_trend: TrendDirection::Stable,
            false_positive_trend: TrendDirection::Stable,
        }
    }
}
