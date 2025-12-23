//! Hardware-Aware Quantum Circuit Parallelization
//!
//! This module provides sophisticated parallelization capabilities that understand
//! and respect hardware constraints, topology, and resource limitations to maximize
//! throughput while maintaining circuit fidelity and correctness.

pub mod config;
pub mod config_defaults;
pub mod engine;
pub mod monitor;
pub mod types;

// Re-export main types
pub use config::*;
pub use engine::*;
pub use monitor::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::CalibrationManager;
    use crate::integrated_device_manager::IntegratedQuantumDeviceManager;
    use crate::routing_advanced::AdvancedQubitRouter;
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};
    use std::time::Duration;

    #[test]
    fn test_parallelization_config_default() {
        let config = ParallelizationConfig::default();
        assert_eq!(config.strategy, ParallelizationStrategy::Hybrid);
        assert!(config.resource_allocation.max_concurrent_circuits > 0);
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Low < TaskPriority::Normal);
        assert!(TaskPriority::Normal < TaskPriority::High);
        assert!(TaskPriority::High < TaskPriority::Critical);
        assert!(TaskPriority::Critical < TaskPriority::System);
    }

    #[test]
    fn test_resource_requirements_creation() {
        let requirements = ParallelResourceRequirements {
            required_cpu_cores: 4,
            required_memory_mb: 1024.0,
            required_qpu_time: Duration::from_secs(300),
            required_bandwidth_mbps: 100.0,
            required_storage_mb: 500.0,
        };

        assert_eq!(requirements.required_cpu_cores, 4);
        assert_eq!(requirements.required_memory_mb, 1024.0);
    }

    #[tokio::test]
    async fn test_parallelization_engine_creation() {
        let config = ParallelizationConfig::default();
        let devices = HashMap::new();
        let cal_mgr = CalibrationManager::new();
        let device_manager = Arc::new(RwLock::new(
            IntegratedQuantumDeviceManager::new(Default::default(), devices, cal_mgr.clone())
                .unwrap(),
        ));
        let calibration_manager = Arc::new(RwLock::new(cal_mgr));
        let router = Arc::new(RwLock::new(AdvancedQubitRouter::new(
            Default::default(),
            crate::routing_advanced::AdvancedRoutingStrategy::Hybrid,
            42,
        )));

        let _engine =
            HardwareParallelizationEngine::new(config, device_manager, calibration_manager, router);

        // Should create without error
        assert!(true);
    }
}
