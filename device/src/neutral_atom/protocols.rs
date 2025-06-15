//! Communication protocols for neutral atom quantum devices
//!
//! This module provides protocol implementations for communicating with
//! neutral atom quantum computers, including device discovery, command
//! execution, and status monitoring.

use crate::{DeviceError, DeviceResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Communication protocol types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtocolType {
    /// HTTP/REST protocol
    Http,
    /// WebSocket protocol for real-time communication
    WebSocket,
    /// TCP/IP direct connection
    Tcp,
    /// USB serial connection
    Serial,
    /// Custom protocol
    Custom(String),
}

/// Protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolConfig {
    /// Protocol type
    pub protocol_type: ProtocolType,
    /// Connection parameters
    pub connection_params: ConnectionParams,
    /// Authentication settings
    pub authentication: AuthenticationConfig,
    /// Timeout settings
    pub timeouts: TimeoutConfig,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Connection parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionParams {
    /// Host address
    pub host: String,
    /// Port number
    pub port: u16,
    /// Use secure connection (TLS/SSL)
    pub secure: bool,
    /// Connection endpoint path
    pub endpoint: String,
    /// Additional connection parameters
    pub parameters: HashMap<String, String>,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Authentication type
    pub auth_type: AuthenticationType,
    /// Username
    pub username: Option<String>,
    /// Password or token
    pub credentials: Option<String>,
    /// API key
    pub api_key: Option<String>,
    /// Certificate path for client certificates
    pub certificate_path: Option<String>,
}

/// Authentication types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthenticationType {
    /// No authentication
    None,
    /// Basic authentication (username/password)
    Basic,
    /// Bearer token authentication
    Bearer,
    /// API key authentication
    ApiKey,
    /// Client certificate authentication
    ClientCertificate,
    /// Custom authentication method
    Custom(String),
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Response timeout
    pub response_timeout: Duration,
    /// Keep-alive timeout
    pub keepalive_timeout: Duration,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: usize,
    /// Initial retry delay
    pub initial_delay: Duration,
    /// Maximum retry delay
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter factor
    pub jitter_factor: f64,
}

/// Protocol message types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageType {
    /// Device discovery request
    Discovery,
    /// Device status query
    Status,
    /// Circuit execution command
    Execute,
    /// Configuration update
    Configure,
    /// Calibration command
    Calibrate,
    /// Emergency stop
    EmergencyStop,
    /// Health check
    HealthCheck,
    /// Custom message
    Custom(String),
}

/// Protocol message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolMessage {
    /// Message ID
    pub message_id: String,
    /// Message type
    pub message_type: MessageType,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Sender information
    pub sender: String,
    /// Recipient information
    pub recipient: String,
    /// Message payload
    pub payload: HashMap<String, serde_json::Value>,
    /// Priority level
    pub priority: MessagePriority,
}

/// Message priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Protocol response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolResponse {
    /// Response ID (matches request message ID)
    pub response_id: String,
    /// Response timestamp
    pub timestamp: SystemTime,
    /// Success status
    pub success: bool,
    /// Status code
    pub status_code: u16,
    /// Status message
    pub status_message: String,
    /// Response data
    pub data: HashMap<String, serde_json::Value>,
    /// Execution time
    pub execution_time: Duration,
}

/// Device discovery protocol
pub struct DeviceDiscoveryProtocol {
    config: ProtocolConfig,
    discovered_devices: Vec<DiscoveredDevice>,
}

/// Discovered device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredDevice {
    /// Device ID
    pub device_id: String,
    /// Device name
    pub device_name: String,
    /// Device type
    pub device_type: String,
    /// Network address
    pub address: String,
    /// Port number
    pub port: u16,
    /// Device capabilities
    pub capabilities: Vec<String>,
    /// Device status
    pub status: DeviceStatus,
    /// Last seen timestamp
    pub last_seen: SystemTime,
}

/// Device status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceStatus {
    /// Device is online and available
    Online,
    /// Device is online but busy
    Busy,
    /// Device is offline
    Offline,
    /// Device is in maintenance mode
    Maintenance,
    /// Device is in error state
    Error,
    /// Unknown status
    Unknown,
}

impl DeviceDiscoveryProtocol {
    /// Create a new device discovery protocol
    pub fn new(config: ProtocolConfig) -> Self {
        Self {
            config,
            discovered_devices: Vec::new(),
        }
    }

    /// Discover available devices
    pub async fn discover_devices(&mut self) -> DeviceResult<Vec<DiscoveredDevice>> {
        // Implementation would depend on the specific protocol
        // For now, return mock devices
        let mock_device = DiscoveredDevice {
            device_id: "neutral_atom_1".to_string(),
            device_name: "Neutral Atom Device 1".to_string(),
            device_type: "NeutralAtom".to_string(),
            address: "192.168.1.100".to_string(),
            port: 8080,
            capabilities: vec![
                "rydberg_gates".to_string(),
                "optical_tweezers".to_string(),
                "hyperfine_manipulation".to_string(),
            ],
            status: DeviceStatus::Online,
            last_seen: SystemTime::now(),
        };

        self.discovered_devices = vec![mock_device.clone()];
        Ok(vec![mock_device])
    }

    /// Get discovered devices
    pub fn get_discovered_devices(&self) -> &[DiscoveredDevice] {
        &self.discovered_devices
    }

    /// Refresh device status
    pub async fn refresh_device_status(&mut self, device_id: &str) -> DeviceResult<DeviceStatus> {
        // Find the device and update its status
        for device in &mut self.discovered_devices {
            if device.device_id == device_id {
                // Mock status update
                device.last_seen = SystemTime::now();
                return Ok(device.status.clone());
            }
        }

        Err(DeviceError::DeviceNotFound(format!(
            "Device {} not found",
            device_id
        )))
    }
}

/// Command execution protocol
pub struct CommandExecutionProtocol {
    config: ProtocolConfig,
    pending_commands: HashMap<String, PendingCommand>,
}

/// Pending command information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingCommand {
    /// Command ID
    pub command_id: String,
    /// Command type
    pub command_type: String,
    /// Target device
    pub target_device: String,
    /// Command parameters
    pub parameters: HashMap<String, serde_json::Value>,
    /// Submission timestamp
    pub submitted_at: SystemTime,
    /// Expected completion time
    pub expected_completion: Option<SystemTime>,
    /// Current status
    pub status: CommandStatus,
}

/// Command execution status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandStatus {
    /// Command is queued
    Queued,
    /// Command is executing
    Executing,
    /// Command completed successfully
    Completed,
    /// Command failed
    Failed,
    /// Command was cancelled
    Cancelled,
    /// Command timed out
    TimedOut,
}

impl CommandExecutionProtocol {
    /// Create a new command execution protocol
    pub fn new(config: ProtocolConfig) -> Self {
        Self {
            config,
            pending_commands: HashMap::new(),
        }
    }

    /// Submit a command for execution
    pub async fn submit_command(
        &mut self,
        command_type: &str,
        target_device: &str,
        parameters: HashMap<String, serde_json::Value>,
    ) -> DeviceResult<String> {
        let command_id = format!("cmd_{}", uuid::Uuid::new_v4());
        
        let pending_command = PendingCommand {
            command_id: command_id.clone(),
            command_type: command_type.to_string(),
            target_device: target_device.to_string(),
            parameters,
            submitted_at: SystemTime::now(),
            expected_completion: None,
            status: CommandStatus::Queued,
        };

        self.pending_commands.insert(command_id.clone(), pending_command);
        Ok(command_id)
    }

    /// Get command status
    pub fn get_command_status(&self, command_id: &str) -> DeviceResult<CommandStatus> {
        self.pending_commands
            .get(command_id)
            .map(|cmd| cmd.status.clone())
            .ok_or_else(|| {
                DeviceError::InvalidInput(format!("Command {} not found", command_id))
            })
    }

    /// Cancel a command
    pub async fn cancel_command(&mut self, command_id: &str) -> DeviceResult<()> {
        if let Some(command) = self.pending_commands.get_mut(command_id) {
            command.status = CommandStatus::Cancelled;
            Ok(())
        } else {
            Err(DeviceError::InvalidInput(format!(
                "Command {} not found",
                command_id
            )))
        }
    }

    /// Get all pending commands
    pub fn get_pending_commands(&self) -> Vec<&PendingCommand> {
        self.pending_commands.values().collect()
    }
}

/// Status monitoring protocol
pub struct StatusMonitoringProtocol {
    config: ProtocolConfig,
    monitored_devices: HashMap<String, DeviceMonitoringInfo>,
}

/// Device monitoring information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMonitoringInfo {
    /// Device ID
    pub device_id: String,
    /// Last status update
    pub last_update: SystemTime,
    /// Current status
    pub current_status: DeviceStatus,
    /// Status history
    pub status_history: Vec<StatusHistoryEntry>,
    /// Performance metrics
    pub metrics: HashMap<String, f64>,
}

/// Status history entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusHistoryEntry {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Status
    pub status: DeviceStatus,
    /// Additional information
    pub info: Option<String>,
}

impl StatusMonitoringProtocol {
    /// Create a new status monitoring protocol
    pub fn new(config: ProtocolConfig) -> Self {
        Self {
            config,
            monitored_devices: HashMap::new(),
        }
    }

    /// Start monitoring a device
    pub fn start_monitoring(&mut self, device_id: &str) -> DeviceResult<()> {
        let monitoring_info = DeviceMonitoringInfo {
            device_id: device_id.to_string(),
            last_update: SystemTime::now(),
            current_status: DeviceStatus::Unknown,
            status_history: Vec::new(),
            metrics: HashMap::new(),
        };

        self.monitored_devices.insert(device_id.to_string(), monitoring_info);
        Ok(())
    }

    /// Stop monitoring a device
    pub fn stop_monitoring(&mut self, device_id: &str) -> DeviceResult<()> {
        self.monitored_devices.remove(device_id);
        Ok(())
    }

    /// Update device status
    pub fn update_device_status(
        &mut self,
        device_id: &str,
        status: DeviceStatus,
        info: Option<String>,
    ) -> DeviceResult<()> {
        if let Some(monitoring_info) = self.monitored_devices.get_mut(device_id) {
            let history_entry = StatusHistoryEntry {
                timestamp: SystemTime::now(),
                status: status.clone(),
                info,
            };

            monitoring_info.current_status = status;
            monitoring_info.last_update = SystemTime::now();
            monitoring_info.status_history.push(history_entry);

            // Keep only recent history (last 1000 entries)
            if monitoring_info.status_history.len() > 1000 {
                monitoring_info.status_history.drain(0..500);
            }

            Ok(())
        } else {
            Err(DeviceError::DeviceNotFound(format!(
                "Device {} not found in monitoring list",
                device_id
            )))
        }
    }

    /// Get device status
    pub fn get_device_status(&self, device_id: &str) -> DeviceResult<&DeviceMonitoringInfo> {
        self.monitored_devices.get(device_id).ok_or_else(|| {
            DeviceError::DeviceNotFound(format!(
                "Device {} not found in monitoring list",
                device_id
            ))
        })
    }

    /// Get all monitored devices
    pub fn get_monitored_devices(&self) -> Vec<&DeviceMonitoringInfo> {
        self.monitored_devices.values().collect()
    }
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            protocol_type: ProtocolType::Http,
            connection_params: ConnectionParams::default(),
            authentication: AuthenticationConfig::default(),
            timeouts: TimeoutConfig::default(),
            retry_config: RetryConfig::default(),
        }
    }
}

impl Default for ConnectionParams {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 8080,
            secure: false,
            endpoint: "/api/v1".to_string(),
            parameters: HashMap::new(),
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            auth_type: AuthenticationType::None,
            username: None,
            credentials: None,
            api_key: None,
            certificate_path: None,
        }
    }
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            connection_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(60),
            response_timeout: Duration::from_secs(120),
            keepalive_timeout: Duration::from_secs(300),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

/// Create a protocol message
pub fn create_protocol_message(
    message_type: MessageType,
    sender: &str,
    recipient: &str,
    payload: HashMap<String, serde_json::Value>,
    priority: MessagePriority,
) -> ProtocolMessage {
    ProtocolMessage {
        message_id: uuid::Uuid::new_v4().to_string(),
        message_type,
        timestamp: SystemTime::now(),
        sender: sender.to_string(),
        recipient: recipient.to_string(),
        payload,
        priority,
    }
}

/// Create a protocol response
pub fn create_protocol_response(
    request_id: &str,
    success: bool,
    status_code: u16,
    status_message: &str,
    data: HashMap<String, serde_json::Value>,
    execution_time: Duration,
) -> ProtocolResponse {
    ProtocolResponse {
        response_id: request_id.to_string(),
        timestamp: SystemTime::now(),
        success,
        status_code,
        status_message: status_message.to_string(),
        data,
        execution_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_config_creation() {
        let config = ProtocolConfig::default();
        assert_eq!(config.protocol_type, ProtocolType::Http);
        assert_eq!(config.connection_params.host, "localhost");
        assert_eq!(config.connection_params.port, 8080);
    }

    #[test]
    fn test_protocol_message_creation() {
        let payload = HashMap::new();
        let message = create_protocol_message(
            MessageType::Status,
            "client",
            "device_1",
            payload,
            MessagePriority::Normal,
        );

        assert_eq!(message.message_type, MessageType::Status);
        assert_eq!(message.sender, "client");
        assert_eq!(message.recipient, "device_1");
        assert_eq!(message.priority, MessagePriority::Normal);
    }

    #[test]
    fn test_device_discovery() {
        let config = ProtocolConfig::default();
        let discovery = DeviceDiscoveryProtocol::new(config);
        assert_eq!(discovery.get_discovered_devices().len(), 0);
    }
}

// Add uuid dependency if not already present
mod uuid {
    use std::fmt;

    pub struct Uuid([u8; 16]);

    impl Uuid {
        pub fn new_v4() -> Self {
            // Simple mock implementation - in real code would use proper UUID generation
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            use std::time::SystemTime;

            let mut hasher = DefaultHasher::new();
            SystemTime::now().hash(&mut hasher);
            let hash = hasher.finish();
            
            let mut bytes = [0u8; 16];
            bytes[0..8].copy_from_slice(&hash.to_le_bytes());
            bytes[8..16].copy_from_slice(&hash.to_be_bytes());
            
            Self(bytes)
        }
    }

    impl fmt::Display for Uuid {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                self.0[0], self.0[1], self.0[2], self.0[3],
                self.0[4], self.0[5],
                self.0[6], self.0[7],
                self.0[8], self.0[9],
                self.0[10], self.0[11], self.0[12], self.0[13], self.0[14], self.0[15])
        }
    }
}