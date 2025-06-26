//! Unified platform detection and capabilities module
//!
//! This module provides a centralized platform detection system that consolidates
//! all hardware capability detection across the QuantRS2 framework.

use std::sync::OnceLock;

pub mod capabilities;
pub mod detector;

pub use capabilities::*;
pub use detector::*;

static PLATFORM_CAPS: OnceLock<PlatformCapabilities> = OnceLock::new();

/// Get the cached platform capabilities
pub fn get_platform_capabilities() -> &'static PlatformCapabilities {
    PLATFORM_CAPS.get_or_init(|| detect_platform_capabilities())
}

/// Initialize platform detection (can be called multiple times safely)
pub fn initialize_platform_detection() {
    let _ = get_platform_capabilities();
}