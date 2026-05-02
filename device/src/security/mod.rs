//! Enterprise security for quantum cloud operations.
//!
//! Provides structured audit logging, credential management, and token-bucket
//! rate limiting for production quantum computing workloads, as well as a
//! comprehensive Quantum System Security Framework.
//!
//! ## Modules
//!
//! - [`audit`] — Structured JSON audit log trail (file, in-memory, null sinks)
//! - [`credentials`] — Credential abstraction with env-var, file, and composite providers
//! - [`rate_limit`] — Per-provider token-bucket rate limiter
//! - [`config`] — Security configuration types for the full framework
//! - [`engines`] — Security engine and result types
//! - [`framework`] — Main `QuantumSystemSecurityFramework` implementation
//! - [`types`] — Security enums and shared types
//!
//! ## Quick start
//!
//! ```rust
//! use quantrs2_device::security::{
//!     InMemoryAuditLogger, AuditLogger, AuditEvent, OperationType,
//!     EnvVarCredentialProvider, CredentialProvider,
//!     RateLimiter,
//! };
//!
//! // Audit logging
//! let logger = InMemoryAuditLogger::new();
//! let event = AuditEvent::new(OperationType::CircuitSubmit, true)
//!     .with_backend("ibm_nairobi")
//!     .with_duration_ms(120);
//! logger.log(event).expect("audit log failed");
//!
//! // Rate limiting
//! let mut limiter = RateLimiter::with_cloud_defaults();
//! if limiter.try_consume("ibm") {
//!     // proceed with API call
//! }
//! ```

pub mod audit;
pub mod config;
pub mod credentials;
pub mod engines;
pub mod framework;
pub mod rate_limit;
pub mod types;

pub use audit::{
    circuit_hash, AuditError, AuditEvent, AuditLogger, FileAuditLogger, InMemoryAuditLogger,
    NullAuditLogger, OperationType,
};
pub use config::*;
pub use credentials::{
    CompositeCredentialProvider, CredentialError, CredentialProvider, EnvVarCredentialProvider,
    FileCredentialProvider, SecretString,
};
pub use engines::*;
pub use framework::*;
pub use rate_limit::{RateLimiter, TokenBucket};
pub use types::*;
