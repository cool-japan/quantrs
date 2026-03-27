//! # MaintenancePriority - Trait Implementations
//!
//! This module contains trait implementations for `MaintenancePriority`.
//!
//! ## Implemented Traits
//!
//! - `PartialOrd`
//! - `Ord`
//! - `PartialEq`
//! - `Eq`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MaintenancePriority;

impl std::cmp::PartialOrd for MaintenancePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for MaintenancePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (MaintenancePriority::Emergency, MaintenancePriority::Emergency) => {
                std::cmp::Ordering::Equal
            }
            (MaintenancePriority::Emergency, _) => std::cmp::Ordering::Less,
            (_, MaintenancePriority::Emergency) => std::cmp::Ordering::Greater,
            (MaintenancePriority::High, MaintenancePriority::High) => {
                std::cmp::Ordering::Equal
            }
            (MaintenancePriority::High, _) => std::cmp::Ordering::Less,
            (_, MaintenancePriority::High) => std::cmp::Ordering::Greater,
            (MaintenancePriority::Medium, MaintenancePriority::Medium) => {
                std::cmp::Ordering::Equal
            }
            (MaintenancePriority::Medium, MaintenancePriority::Low) => {
                std::cmp::Ordering::Less
            }
            (MaintenancePriority::Low, MaintenancePriority::Medium) => {
                std::cmp::Ordering::Greater
            }
            (MaintenancePriority::Low, MaintenancePriority::Low) => {
                std::cmp::Ordering::Equal
            }
        }
    }
}

impl std::cmp::PartialEq for MaintenancePriority {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MaintenancePriority::Critical, MaintenancePriority::Critical) => true,
            (MaintenancePriority::High, MaintenancePriority::High) => true,
            (MaintenancePriority::Medium, MaintenancePriority::Medium) => true,
            (MaintenancePriority::Low, MaintenancePriority::Low) => true,
            _ => false,
        }
    }
}

impl std::cmp::Eq for MaintenancePriority {}

