//! Network optimization configurations

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Network optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOptimizationConfig {
    /// CDN configuration
    pub cdn: CDNConfig,
    /// Connection optimization
    pub connection: ConnectionOptimizationConfig,
    /// Protocol optimization
    pub protocol: ProtocolOptimizationConfig,
}

/// CDN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CDNConfig {
    /// Enable CDN
    pub enabled: bool,
    /// CDN providers
    pub providers: Vec<String>,
    /// Cache policies
    pub cache_policies: Vec<CachePolicy>,
}

/// Cache policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePolicy {
    /// Resource type
    pub resource_type: String,
    /// TTL
    pub ttl: Duration,
    /// Cache key strategy
    pub key_strategy: CacheKeyStrategy,
}

/// Cache key strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheKeyStrategy {
    Simple,
    Hierarchical,
    Hashed,
    Custom(String),
}

/// Connection optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionOptimizationConfig {
    /// Keep-alive settings
    pub keep_alive: KeepAliveConfig,
    /// Connection pooling
    pub pooling: ConnectionPoolingConfig,
    /// Compression
    pub compression: CompressionConfig,
}

/// Keep-alive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeepAliveConfig {
    /// Enable keep-alive
    pub enabled: bool,
    /// Timeout
    pub timeout: Duration,
    /// Max requests per connection
    pub max_requests: usize,
}

/// Connection pooling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolingConfig {
    /// Pool size
    pub pool_size: usize,
    /// Max connections per host
    pub max_per_host: usize,
    /// Connection timeout
    pub connection_timeout: Duration,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithms
    pub algorithms: Vec<CompressionAlgorithm>,
    /// Compression level
    pub level: u8,
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Deflate,
    Brotli,
    LZ4,
}

/// Protocol optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolOptimizationConfig {
    /// HTTP/2 configuration
    pub http2: HTTP2Config,
    /// QUIC configuration
    pub quic: QUICConfig,
    /// WebSocket configuration
    pub websocket: WebSocketConfig,
}

/// HTTP/2 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HTTP2Config {
    /// Enable HTTP/2
    pub enabled: bool,
    /// Server push
    pub server_push: bool,
    /// Stream multiplexing
    pub multiplexing: bool,
}

/// QUIC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QUICConfig {
    /// Enable QUIC
    pub enabled: bool,
    /// Connection migration
    pub connection_migration: bool,
    /// 0-RTT
    pub zero_rtt: bool,
}

/// WebSocket configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    /// Enable WebSocket
    pub enabled: bool,
    /// Ping interval
    pub ping_interval: Duration,
    /// Max frame size
    pub max_frame_size: usize,
}
