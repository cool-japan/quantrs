//! Profiling session management
//!
//! This module provides session management capabilities for profiling
//! including session lifecycle, data storage, analytics, and persistence.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

// Import types from sibling modules
use super::analyzers::*;
use super::collectors::*;
use super::metrics::*;

pub struct SessionManager {
    /// Active sessions
    pub active_sessions: HashMap<String, ProfilingSession>,
    /// Session configuration
    pub session_config: SessionConfig,
    /// Session storage
    pub session_storage: SessionStorage,
    /// Session analytics
    pub session_analytics: SessionAnalytics,
}

/// Individual profiling session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingSession {
    /// Session ID
    pub id: String,
    /// Session start time
    pub start_time: SystemTime,
    /// Session end time
    pub end_time: Option<SystemTime>,
    /// Session status
    pub status: SessionStatus,
    /// Collected data
    pub collected_data: SessionData,
    /// Session metadata
    pub metadata: HashMap<String, String>,
}

/// Session status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionStatus {
    /// Session starting
    Starting,
    /// Session running
    Running,
    /// Session paused
    Paused,
    /// Session stopping
    Stopping,
    /// Session completed
    Completed,
    /// Session failed
    Failed { error: String },
}

/// Session data collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    /// Performance metrics
    pub metrics: Vec<PerformanceMetric>,
    /// Gate profiles
    pub gate_profiles: HashMap<String, GateProfile>,
    /// Memory snapshots
    pub memory_snapshots: Vec<MemorySnapshot>,
    /// Resource usage data
    pub resource_data: Vec<ResourceUsage>,
}

/// Session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Default session duration
    pub default_duration: Duration,
    /// Data collection interval
    pub collection_interval: Duration,
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: usize,
    /// Session timeout
    pub session_timeout: Duration,
}

/// Session storage configuration
#[derive(Debug, Clone)]
pub struct SessionStorage {
    /// Storage backend
    pub backend: StorageBackend,
    /// Storage configuration
    pub config: StorageConfig,
    /// Data serialization
    pub serialization: SerializationConfig,
}

/// Storage backend types
#[derive(Debug, Clone)]
pub enum StorageBackend {
    /// In-memory storage
    InMemory,
    /// File system storage
    FileSystem { path: String },
    /// Database storage
    Database { connection_string: String },
    /// Cloud storage
    Cloud {
        provider: String,
        config: HashMap<String, String>,
    },
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Enable compression
    pub enable_compression: bool,
    /// Enable encryption
    pub enable_encryption: bool,
    /// Retention policy
    pub retention_policy: DataRetentionPolicy,
    /// Backup configuration
    pub backup_config: Option<BackupConfig>,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Backup frequency
    pub frequency: Duration,
    /// Backup location
    pub location: String,
    /// Backup retention
    pub retention: Duration,
    /// Enable incremental backups
    pub incremental: bool,
}

/// Serialization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationConfig {
    /// Serialization format
    pub format: SerializationFormat,
    /// Enable schema validation
    pub schema_validation: bool,
    /// Version compatibility
    pub version_compatibility: bool,
}

/// Serialization formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// JSON format
    JSON,
    /// Binary format
    Binary,
    /// Protocol buffers
    ProtocolBuffers,
    /// `MessagePack`
    MessagePack,
}

/// Session analytics
#[derive(Debug, Clone)]
pub struct SessionAnalytics {
    /// Analytics configuration
    pub config: AnalyticsConfig,
    /// Session statistics
    pub statistics: SessionStatistics,
    /// Performance insights
    pub insights: Vec<PerformanceInsight>,
    /// Trend analysis
    pub trend_analysis: SessionTrendAnalysis,
}

/// Analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable real-time analytics
    pub enable_realtime: bool,
    /// Analytics depth
    pub depth: AnalysisDepth,
    /// Reporting frequency
    pub reporting_frequency: Duration,
    /// Custom metrics
    pub custom_metrics: Vec<String>,
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    /// Total sessions
    pub total_sessions: usize,
    /// Average session duration
    pub avg_duration: Duration,
    /// Session success rate
    pub success_rate: f64,
    /// Data collection efficiency
    pub collection_efficiency: f64,
}

/// Performance insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInsight {
    /// Insight type
    pub insight_type: InsightType,
    /// Insight description
    pub description: String,
    /// Confidence score
    pub confidence: f64,
    /// Impact assessment
    pub impact: f64,
    /// Recommended actions
    pub actions: Vec<String>,
}

/// Types of performance insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    /// Performance optimization opportunity
    OptimizationOpportunity,
    /// Resource utilization insight
    ResourceUtilization,
    /// Scaling recommendation
    ScalingRecommendation,
    /// Configuration improvement
    ConfigurationImprovement,
    /// Architecture recommendation
    ArchitectureRecommendation,
}

/// Session trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTrendAnalysis {
    /// Performance trends
    pub performance_trends: HashMap<String, TrendDirection>,
    /// Resource trends
    pub resource_trends: HashMap<String, TrendDirection>,
    /// Quality trends
    pub quality_trends: HashMap<String, TrendDirection>,
    /// Prediction accuracy trends
    pub prediction_trends: HashMap<String, f64>,
}
