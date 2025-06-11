//! Session management configurations

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Session management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionManagementConfig {
    /// Session affinity
    pub affinity: SessionAffinityConfig,
    /// Session persistence
    pub persistence: SessionPersistenceConfig,
    /// Session replication
    pub replication: SessionReplicationConfig,
}

/// Session affinity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAffinityConfig {
    /// Enable session affinity
    pub enabled: bool,
    /// Affinity method
    pub method: AffinityMethod,
    /// Timeout
    pub timeout: Duration,
}

/// Affinity methods
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AffinityMethod {
    CookieBased,
    IPBased,
    SessionID,
    Custom(String),
}

/// Session persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionPersistenceConfig {
    /// Persistence type
    pub persistence_type: PersistenceType,
    /// Storage backend
    pub storage: SessionStorageConfig,
    /// Encryption
    pub encryption: SessionEncryptionConfig,
}

/// Persistence types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PersistenceType {
    InMemory,
    Database,
    Redis,
    File,
    Custom(String),
}

/// Session storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStorageConfig {
    /// Storage type
    pub storage_type: String,
    /// Connection string
    pub connection_string: String,
    /// Pool size
    pub pool_size: usize,
}

/// Session encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEncryptionConfig {
    /// Enable encryption
    pub enabled: bool,
    /// Encryption algorithm
    pub algorithm: String,
    /// Key management
    pub key_management: KeyManagementConfig,
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    /// Key source
    pub key_source: KeySource,
    /// Key rotation
    pub rotation: KeyRotationConfig,
    /// Key backup
    pub backup: KeyBackupConfig,
}

/// Key sources
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KeySource {
    Static,
    Environment,
    KeyVault,
    HSM,
    Custom(String),
}

/// Key rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyRotationConfig {
    /// Enable rotation
    pub enabled: bool,
    /// Rotation interval
    pub interval: Duration,
    /// Overlap period
    pub overlap_period: Duration,
}

/// Key backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyBackupConfig {
    /// Enable backup
    pub enabled: bool,
    /// Backup frequency
    pub frequency: Duration,
    /// Backup location
    pub location: String,
}

/// Session replication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionReplicationConfig {
    /// Enable replication
    pub enabled: bool,
    /// Replication strategy
    pub strategy: ReplicationStrategy,
    /// Replication factor
    pub factor: usize,
    /// Consistency level
    pub consistency: ConsistencyLevel,
}

/// Replication strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReplicationStrategy {
    Synchronous,
    Asynchronous,
    SemiSynchronous,
    EventualConsistency,
}

/// Consistency levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    Strong,
    Eventual,
    Session,
    Bounded,
}
