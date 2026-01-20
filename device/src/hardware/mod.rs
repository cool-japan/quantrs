//! Hardware-specific client implementations for quantum computing providers
//!
//! This module contains client implementations for various quantum computing platforms
//! and cloud services, providing unified access through the QuantRS hardware abstraction layer.

pub mod rigetti;
pub mod honeywell;

pub use rigetti::{RigettiBackend, RigettiClient, RigettiConfig};
pub use honeywell::{HoneywellBackend, HoneywellClient, HoneywellConfig};