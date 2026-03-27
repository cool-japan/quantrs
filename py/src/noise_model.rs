//! `PyRealisticNoiseModel` — Python wrapper for realistic quantum hardware noise models.

use pyo3::prelude::*;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::noise::{BitFlipChannel, DepolarizingChannel, NoiseChannelType};
use quantrs2_sim::noise_advanced::{AdvancedNoiseModel, RealisticNoiseModelBuilder};
use std::time::Duration;

/// Python wrapper for realistic noise models
#[pyclass]
pub struct PyRealisticNoiseModel {
    /// The internal Rust noise model
    pub(crate) noise_model: AdvancedNoiseModel,
}

/// Implementation of the `PyRealisticNoiseModel` class
#[pymethods]
impl PyRealisticNoiseModel {
    /// Create a new realistic noise model for IBM quantum devices
    ///
    /// Args:
    ///     `device_name` (str): The name of the IBM quantum device (e.g., "`ibmq_lima`", "`ibm_cairo`")
    ///
    /// Returns:
    ///     `PyRealisticNoiseModel`: A noise model configured with the specified device parameters
    #[staticmethod]
    fn ibm_device(device_name: &str) -> Self {
        // Convert device name to lowercase
        let device_name = device_name.to_lowercase();

        // Create a list of qubits from 0 to 31 (max 32 qubits support)
        let qubits: Vec<QubitId> = (0..32).map(QubitId::new).collect();

        // Create IBM device noise model
        let noise_model = RealisticNoiseModelBuilder::new(true)
            .with_ibm_device_noise(&qubits, &device_name)
            .build();

        Self { noise_model }
    }

    /// Create a new realistic noise model for Rigetti quantum devices
    ///
    /// Args:
    ///     `device_name` (str): The name of the Rigetti quantum device (e.g., "Aspen-M-2")
    ///
    /// Returns:
    ///     `PyRealisticNoiseModel`: A noise model configured with the specified device parameters
    #[staticmethod]
    fn rigetti_device(device_name: &str) -> Self {
        // Create a list of qubits from 0 to 31 (max 32 qubits support)
        let qubits: Vec<QubitId> = (0..32).map(QubitId::new).collect();

        // Create Rigetti device noise model
        let noise_model = RealisticNoiseModelBuilder::new(true)
            .with_rigetti_device_noise(&qubits, device_name)
            .build();

        Self { noise_model }
    }

    /// Create a new realistic noise model with custom parameters
    ///
    /// Args:
    ///     `t1_us` (float): T1 relaxation time in microseconds
    ///     `t2_us` (float): T2 dephasing time in microseconds
    ///     `gate_time_ns` (float): Gate time in nanoseconds
    ///     `gate_error_1q` (float): Single-qubit gate error rate (0.0 to 1.0)
    ///     `gate_error_2q` (float): Two-qubit gate error rate (0.0 to 1.0)
    ///     `readout_error` (float): Readout error rate (0.0 to 1.0)
    ///
    /// Returns:
    ///     `PyRealisticNoiseModel`: A custom noise model with the specified parameters
    #[staticmethod]
    #[pyo3(signature = (t1_us=100.0, t2_us=50.0, gate_time_ns=40.0, gate_error_1q=0.001, gate_error_2q=0.01, readout_error=0.02))]
    fn custom(
        t1_us: f64,
        t2_us: f64,
        gate_time_ns: f64,
        gate_error_1q: f64,
        gate_error_2q: f64,
        readout_error: f64,
    ) -> Self {
        // Create a list of qubits from 0 to 31 (max 32 qubits support)
        let qubits: Vec<QubitId> = (0..32).map(QubitId::new).collect();

        // Create pairs of adjacent qubits for two-qubit noise
        let qubit_pairs: Vec<(QubitId, QubitId)> = (0..31)
            .map(|i| (QubitId::new(i), QubitId::new(i + 1)))
            .collect();

        // Create custom noise model
        let noise_model = RealisticNoiseModelBuilder::new(true)
            .with_custom_thermal_relaxation(
                &qubits,
                Duration::from_secs_f64(t1_us * 1e-6),
                Duration::from_secs_f64(t2_us * 1e-6),
                Duration::from_secs_f64(gate_time_ns * 1e-9),
            )
            .with_custom_two_qubit_noise(&qubit_pairs, gate_error_2q)
            .build();

        // Add depolarizing noise for single-qubit gates and readout errors
        let mut result = Self { noise_model };

        for &qubit in &qubits {
            result
                .noise_model
                .add_base_channel(NoiseChannelType::Depolarizing(DepolarizingChannel {
                    target: qubit,
                    probability: gate_error_1q,
                }));

            result
                .noise_model
                .add_base_channel(NoiseChannelType::BitFlip(BitFlipChannel {
                    target: qubit,
                    probability: readout_error,
                }));
        }

        result
    }

    /// Get the number of noise channels in this model
    #[getter]
    fn num_channels(&self) -> usize {
        self.noise_model.num_channels()
    }
}
