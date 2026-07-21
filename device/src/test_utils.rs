//! Test utilities for quantum device testing

use crate::{CircuitExecutor, CircuitResult, DeviceResult, QuantumDevice};
use async_trait::async_trait;
use quantrs2_circuit::prelude::Circuit;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Mock quantum device for testing
pub struct MockQuantumDevice {
    pub qubit_count: usize,
    pub is_available: bool,
    pub is_simulator: bool,
}

impl MockQuantumDevice {
    pub fn new(qubit_count: usize) -> Self {
        Self {
            qubit_count,
            is_available: true,
            is_simulator: true,
        }
    }
}

#[cfg(feature = "_async_device")]
#[async_trait::async_trait]
impl QuantumDevice for MockQuantumDevice {
    async fn is_available(&self) -> DeviceResult<bool> {
        Ok(self.is_available)
    }

    async fn qubit_count(&self) -> DeviceResult<usize> {
        Ok(self.qubit_count)
    }

    async fn properties(&self) -> DeviceResult<HashMap<String, String>> {
        let mut props = HashMap::new();
        props.insert("device_type".to_string(), "mock".to_string());
        props.insert("qubit_count".to_string(), self.qubit_count.to_string());
        Ok(props)
    }

    async fn is_simulator(&self) -> DeviceResult<bool> {
        Ok(self.is_simulator)
    }
}

#[cfg(not(feature = "_async_device"))]
impl QuantumDevice for MockQuantumDevice {
    fn is_available(&self) -> DeviceResult<bool> {
        Ok(self.is_available)
    }

    fn qubit_count(&self) -> DeviceResult<usize> {
        Ok(self.qubit_count)
    }

    fn properties(&self) -> DeviceResult<HashMap<String, String>> {
        let mut props = HashMap::new();
        props.insert("device_type".to_string(), "mock".to_string());
        props.insert("qubit_count".to_string(), self.qubit_count.to_string());
        Ok(props)
    }

    fn is_simulator(&self) -> DeviceResult<bool> {
        Ok(self.is_simulator)
    }
}

#[cfg(feature = "_async_device")]
#[async_trait]
impl CircuitExecutor for MockQuantumDevice {
    async fn execute_circuit<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
        shots: usize,
    ) -> DeviceResult<CircuitResult> {
        let mut counts = HashMap::new();
        let all_zeros = "0".repeat(N);
        counts.insert(all_zeros, shots);

        let mut metadata = HashMap::new();
        metadata.insert("device_type".to_string(), "mock".to_string());

        Ok(CircuitResult {
            counts,
            shots,
            metadata,
        })
    }

    async fn execute_circuits<const N: usize>(
        &self,
        circuits: Vec<&Circuit<N>>,
        shots: usize,
    ) -> DeviceResult<Vec<CircuitResult>> {
        let mut results = Vec::new();
        for circuit in circuits {
            let result = self.execute_circuit(circuit, shots).await?;
            results.push(result);
        }
        Ok(results)
    }

    async fn can_execute_circuit<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
    ) -> DeviceResult<bool> {
        Ok(N <= self.qubit_count)
    }

    async fn estimated_queue_time<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
    ) -> DeviceResult<std::time::Duration> {
        Ok(std::time::Duration::from_secs(1))
    }
}

#[cfg(not(feature = "_async_device"))]
impl CircuitExecutor for MockQuantumDevice {
    fn execute_circuit<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
        shots: usize,
    ) -> DeviceResult<CircuitResult> {
        let mut counts = HashMap::new();
        let all_zeros = "0".repeat(N);
        counts.insert(all_zeros, shots);

        let mut metadata = HashMap::new();
        metadata.insert("device_type".to_string(), "mock".to_string());

        Ok(CircuitResult {
            counts,
            shots,
            metadata,
        })
    }

    fn execute_circuits<const N: usize>(
        &self,
        circuits: Vec<&Circuit<N>>,
        shots: usize,
    ) -> DeviceResult<Vec<CircuitResult>> {
        let mut results = Vec::new();
        for circuit in circuits {
            let result = self.execute_circuit(circuit, shots)?;
            results.push(result);
        }
        Ok(results)
    }

    fn can_execute_circuit<const N: usize>(&self, _circuit: &Circuit<N>) -> DeviceResult<bool> {
        Ok(N <= self.qubit_count)
    }

    fn estimated_queue_time<const N: usize>(
        &self,
        _circuit: &Circuit<N>,
    ) -> DeviceResult<std::time::Duration> {
        Ok(std::time::Duration::from_secs(1))
    }
}

/// Create a mock quantum device for testing
pub fn create_mock_quantum_device() -> Arc<RwLock<dyn QuantumDevice + Send + Sync>> {
    Arc::new(RwLock::new(MockQuantumDevice::new(20)))
}

// Regression tests for the `_async_device` cfg unification (see
// `device/Cargo.toml` and `device/src/lib.rs`): `QuantumDevice`/`CircuitExecutor`
// must expose an `async_trait` variant whenever `_async_device` is enabled
// (transitively, by any of the ibm/azure/aws/neutral_atom/photonic features)
// and a plain sync variant otherwise. Whichever variant is active must round
// trip correctly through `MockQuantumDevice`.
#[cfg(all(test, feature = "_async_device"))]
mod async_device_regression_tests {
    use super::*;

    #[tokio::test]
    async fn mock_device_async_trait_round_trip() {
        let device = MockQuantumDevice::new(5);

        assert!(device
            .is_available()
            .await
            .expect("is_available should succeed"));
        assert_eq!(
            device
                .qubit_count()
                .await
                .expect("qubit_count should succeed"),
            5
        );

        let props = device
            .properties()
            .await
            .expect("properties should succeed");
        assert_eq!(props.get("qubit_count").map(String::as_str), Some("5"));
        assert!(device
            .is_simulator()
            .await
            .expect("is_simulator should succeed"));

        let results = device
            .execute_circuits::<2>(Vec::new(), 100)
            .await
            .expect("execute_circuits should succeed");
        assert!(results.is_empty());
    }
}

#[cfg(all(test, not(feature = "_async_device")))]
mod sync_device_regression_tests {
    use super::*;

    #[test]
    fn mock_device_sync_trait_round_trip() {
        let device = MockQuantumDevice::new(5);

        assert!(device.is_available().expect("is_available should succeed"));
        assert_eq!(device.qubit_count().expect("qubit_count should succeed"), 5);

        let props = device.properties().expect("properties should succeed");
        assert_eq!(props.get("qubit_count").map(String::as_str), Some("5"));
        assert!(device.is_simulator().expect("is_simulator should succeed"));

        let results = device
            .execute_circuits::<2>(Vec::new(), 100)
            .expect("execute_circuits should succeed");
        assert!(results.is_empty());
    }
}
