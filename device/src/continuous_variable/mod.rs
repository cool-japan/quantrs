//! Continuous Variable Quantum Computing
//!
//! This module implements continuous variable (CV) quantum computing systems,
//! which operate on continuous degrees of freedom like position and momentum
//! rather than discrete qubits.

use crate::{CircuitExecutor, CircuitResult, DeviceError, DeviceResult, QuantumDevice};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::time::Duration;

pub mod cluster_states;
pub mod cv_gates;
pub mod error_correction;
pub mod gaussian_states;
pub mod heterodyne;
pub mod homodyne;
pub mod measurements;

pub use cluster_states::*;
pub use cv_gates::*;
pub use error_correction::*;
pub use gaussian_states::*;
pub use heterodyne::*;
pub use homodyne::*;
pub use measurements::*;

/// Complex number representation for CV quantum computing
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn zero() -> Self {
        Self {
            real: 0.0,
            imag: 0.0,
        }
    }

    pub fn one() -> Self {
        Self {
            real: 1.0,
            imag: 0.0,
        }
    }

    pub fn i() -> Self {
        Self {
            real: 0.0,
            imag: 1.0,
        }
    }

    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self {
            real: self.real * scalar,
            imag: self.imag * scalar,
        }
    }
}

/// Types of continuous variable quantum systems
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CVSystemType {
    /// Gaussian states with squeezed light
    GaussianStates,
    /// Cluster state quantum computing
    ClusterState,
    /// Measurement-based quantum computing
    MeasurementBased,
    /// Hybrid discrete-continuous
    HybridDvCv,
}

/// Continuous variable quantum device
pub struct CVQuantumDevice {
    /// System type
    pub system_type: CVSystemType,
    /// Number of modes
    pub num_modes: usize,
    /// Mode frequencies (Hz)
    pub mode_frequencies: Vec<f64>,
    /// Current Gaussian state
    pub gaussian_state: GaussianState,
    /// Device configuration
    pub config: CVDeviceConfig,
    /// Connection status
    pub is_connected: bool,
    /// Measurement results history
    pub measurement_history: Vec<CVMeasurementResult>,
}

/// Configuration for CV quantum devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVDeviceConfig {
    /// Maximum squeezing parameter (dB)
    pub max_squeezing_db: f64,
    /// Optical power (mW)
    pub optical_power_mw: f64,
    /// Detection efficiency
    pub detection_efficiency: f64,
    /// Electronic noise (dB)
    pub electronic_noise_db: f64,
    /// Homodyne detection bandwidth (Hz)
    pub homodyne_bandwidth_hz: f64,
    /// Phase noise (rad/√Hz)
    pub phase_noise: f64,
    /// Temperature (K)
    pub temperature_k: f64,
    /// Enable error correction
    pub enable_error_correction: bool,
}

impl Default for CVDeviceConfig {
    fn default() -> Self {
        Self {
            max_squeezing_db: 15.0,
            optical_power_mw: 1.0,
            detection_efficiency: 0.95,
            electronic_noise_db: -90.0,
            homodyne_bandwidth_hz: 10e6,
            phase_noise: 1e-6,
            temperature_k: 0.1,
            enable_error_correction: true,
        }
    }
}

impl CVQuantumDevice {
    /// Create a new CV quantum device
    pub fn new(
        system_type: CVSystemType,
        num_modes: usize,
        config: CVDeviceConfig,
    ) -> DeviceResult<Self> {
        let mode_frequencies = (0..num_modes)
            .map(|i| 1e14 + i as f64 * 1e12) // Base frequency 100 THz + mode spacing
            .collect();

        let gaussian_state = GaussianState::vacuum_state(num_modes);

        Ok(Self {
            system_type,
            num_modes,
            mode_frequencies,
            gaussian_state,
            config,
            is_connected: false,
            measurement_history: Vec::new(),
        })
    }

    /// Connect to the CV quantum hardware
    pub async fn connect(&mut self) -> DeviceResult<()> {
        // Simulate hardware connection and initialization
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Initialize optical components
        self.initialize_optical_system().await?;

        self.is_connected = true;
        Ok(())
    }

    /// Initialize optical system
    async fn initialize_optical_system(&mut self) -> DeviceResult<()> {
        // Initialize laser sources
        for (i, &freq) in self.mode_frequencies.iter().enumerate() {
            println!("Initializing mode {} at frequency {:.2e} Hz", i, freq);
        }

        // Initialize homodyne detectors
        println!("Initializing homodyne detection system");

        // Initialize squeezers
        println!("Initializing squeezing apparatus");

        Ok(())
    }

    /// Disconnect from hardware
    pub async fn disconnect(&mut self) -> DeviceResult<()> {
        self.is_connected = false;
        Ok(())
    }

    /// Apply displacement operation to a mode
    pub async fn displacement(
        &mut self,
        mode: usize,
        amplitude: f64,
        phase: f64,
    ) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes {}",
                mode, self.num_modes
            )));
        }

        let displacement = Complex::new(amplitude * phase.cos(), amplitude * phase.sin());

        self.gaussian_state.apply_displacement(mode, displacement)?;
        Ok(())
    }

    /// Apply squeezing operation to a mode
    pub async fn squeezing(
        &mut self,
        mode: usize,
        squeezing_param: f64,
        phase: f64,
    ) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes {}",
                mode, self.num_modes
            )));
        }

        if squeezing_param.abs() > self.config.max_squeezing_db / 8.686 {
            return Err(DeviceError::InvalidInput(format!(
                "Squeezing parameter {} exceeds maximum",
                squeezing_param
            )));
        }

        self.gaussian_state
            .apply_squeezing(mode, squeezing_param, phase)?;
        Ok(())
    }

    /// Apply two-mode squeezing
    pub async fn two_mode_squeezing(
        &mut self,
        mode1: usize,
        mode2: usize,
        squeezing_param: f64,
        phase: f64,
    ) -> DeviceResult<()> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(DeviceError::InvalidInput(
                "One or both modes exceed available modes".to_string(),
            ));
        }

        self.gaussian_state
            .apply_two_mode_squeezing(mode1, mode2, squeezing_param, phase)?;
        Ok(())
    }

    /// Apply beamsplitter operation
    pub async fn beamsplitter(
        &mut self,
        mode1: usize,
        mode2: usize,
        transmittance: f64,
        phase: f64,
    ) -> DeviceResult<()> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(DeviceError::InvalidInput(
                "One or both modes exceed available modes".to_string(),
            ));
        }

        if transmittance < 0.0 || transmittance > 1.0 {
            return Err(DeviceError::InvalidInput(
                "Transmittance must be between 0 and 1".to_string(),
            ));
        }

        self.gaussian_state
            .apply_beamsplitter(mode1, mode2, transmittance, phase)?;
        Ok(())
    }

    /// Apply phase rotation
    pub async fn phase_rotation(&mut self, mode: usize, phase: f64) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes {}",
                mode, self.num_modes
            )));
        }

        self.gaussian_state.apply_phase_rotation(mode, phase)?;
        Ok(())
    }

    /// Perform homodyne measurement
    pub async fn homodyne_measurement(&mut self, mode: usize, phase: f64) -> DeviceResult<f64> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes {}",
                mode, self.num_modes
            )));
        }

        let result = self
            .gaussian_state
            .homodyne_measurement(mode, phase, &self.config)?;

        self.measurement_history.push(CVMeasurementResult {
            mode,
            measurement_type: CVMeasurementType::Homodyne { phase },
            result: CVMeasurementOutcome::Real(result),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        });

        Ok(result)
    }

    /// Perform heterodyne measurement
    pub async fn heterodyne_measurement(&mut self, mode: usize) -> DeviceResult<Complex> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes {}",
                mode, self.num_modes
            )));
        }

        let result = self
            .gaussian_state
            .heterodyne_measurement(mode, &self.config)?;

        self.measurement_history.push(CVMeasurementResult {
            mode,
            measurement_type: CVMeasurementType::Heterodyne,
            result: CVMeasurementOutcome::Complex(result),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        });

        Ok(result)
    }

    /// Reset a mode to vacuum state
    pub async fn reset_mode(&mut self, mode: usize) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes {}",
                mode, self.num_modes
            )));
        }

        self.gaussian_state.reset_mode_to_vacuum(mode)?;
        Ok(())
    }

    /// Get current mode state information
    pub fn get_mode_state(&self, mode: usize) -> DeviceResult<CVModeState> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds available modes {}",
                mode, self.num_modes
            )));
        }

        Ok(self.gaussian_state.get_mode_state(mode)?)
    }

    /// Get system entanglement
    pub fn get_entanglement_measures(&self) -> CVEntanglementMeasures {
        self.gaussian_state.calculate_entanglement_measures()
    }

    /// Get device diagnostics
    pub async fn get_diagnostics(&self) -> CVDeviceDiagnostics {
        CVDeviceDiagnostics {
            is_connected: self.is_connected,
            num_modes: self.num_modes,
            total_measurements: self.measurement_history.len(),
            average_squeezing: self.gaussian_state.calculate_average_squeezing(),
            system_purity: self.gaussian_state.calculate_purity(),
            entanglement_entropy: self.gaussian_state.calculate_entanglement_entropy(),
            optical_power_mw: self.config.optical_power_mw,
            detection_efficiency: self.config.detection_efficiency,
        }
    }
}

/// CV measurement types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CVMeasurementType {
    /// Homodyne measurement at specific phase
    Homodyne { phase: f64 },
    /// Heterodyne measurement
    Heterodyne,
    /// Photon number measurement
    PhotonNumber,
    /// Parity measurement
    Parity,
}

/// CV measurement outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CVMeasurementOutcome {
    Real(f64),
    Complex(Complex),
    Integer(i32),
    Boolean(bool),
}

/// CV measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVMeasurementResult {
    pub mode: usize,
    pub measurement_type: CVMeasurementType,
    pub result: CVMeasurementOutcome,
    pub timestamp: f64,
}

/// State information for a CV mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVModeState {
    /// Mean field amplitude
    pub mean_amplitude: Complex,
    /// Quadrature variances (x, p)
    pub quadrature_variances: (f64, f64),
    /// Squeezing parameter
    pub squeezing_parameter: f64,
    /// Squeezing phase
    pub squeezing_phase: f64,
    /// Mode purity
    pub purity: f64,
}

/// Entanglement measures for CV systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVEntanglementMeasures {
    /// Logarithmic negativity
    pub logarithmic_negativity: f64,
    /// Entanglement of formation
    pub entanglement_of_formation: f64,
    /// Mutual information
    pub mutual_information: f64,
    /// EPR correlation
    pub epr_correlation: f64,
}

/// Diagnostics for CV devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVDeviceDiagnostics {
    pub is_connected: bool,
    pub num_modes: usize,
    pub total_measurements: usize,
    pub average_squeezing: f64,
    pub system_purity: f64,
    pub entanglement_entropy: f64,
    pub optical_power_mw: f64,
    pub detection_efficiency: f64,
}

#[async_trait::async_trait]
impl QuantumDevice for CVQuantumDevice {
    async fn is_available(&self) -> DeviceResult<bool> {
        Ok(self.is_connected)
    }

    async fn qubit_count(&self) -> DeviceResult<usize> {
        // CV systems don't have discrete qubits, return equivalent capacity
        Ok(self.num_modes)
    }

    async fn properties(&self) -> DeviceResult<HashMap<String, String>> {
        let mut props = HashMap::new();
        props.insert("device_type".to_string(), "continuous_variable".to_string());
        props.insert("system_type".to_string(), format!("{:?}", self.system_type));
        props.insert("num_modes".to_string(), self.num_modes.to_string());
        props.insert(
            "max_squeezing_db".to_string(),
            self.config.max_squeezing_db.to_string(),
        );
        props.insert(
            "detection_efficiency".to_string(),
            self.config.detection_efficiency.to_string(),
        );
        props.insert(
            "optical_power_mw".to_string(),
            self.config.optical_power_mw.to_string(),
        );
        Ok(props)
    }

    async fn is_simulator(&self) -> DeviceResult<bool> {
        Ok(true) // This implementation is a simulator
    }
}

/// Create a Gaussian CV device
pub fn create_gaussian_cv_device(
    num_modes: usize,
    config: Option<CVDeviceConfig>,
) -> DeviceResult<CVQuantumDevice> {
    let config = config.unwrap_or_default();
    CVQuantumDevice::new(CVSystemType::GaussianStates, num_modes, config)
}

/// Create a cluster state CV device
pub fn create_cluster_state_cv_device(
    num_modes: usize,
    config: Option<CVDeviceConfig>,
) -> DeviceResult<CVQuantumDevice> {
    let config = config.unwrap_or_default();
    CVQuantumDevice::new(CVSystemType::ClusterState, num_modes, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cv_device_creation() {
        let device = create_gaussian_cv_device(4, None).unwrap();
        assert_eq!(device.num_modes, 4);
        assert_eq!(device.system_type, CVSystemType::GaussianStates);
    }

    #[tokio::test]
    async fn test_cv_device_connection() {
        let mut device = create_gaussian_cv_device(2, None).unwrap();
        assert!(!device.is_connected);

        device.connect().await.unwrap();
        assert!(device.is_connected);

        device.disconnect().await.unwrap();
        assert!(!device.is_connected);
    }

    #[tokio::test]
    async fn test_displacement_operation() {
        let mut device = create_gaussian_cv_device(2, None).unwrap();
        device.connect().await.unwrap();

        device.displacement(0, 1.0, PI / 4.0).await.unwrap();

        let state = device.get_mode_state(0).unwrap();
        assert!(state.mean_amplitude.magnitude() > 0.0);
    }

    #[tokio::test]
    async fn test_squeezing_operation() {
        let mut device = create_gaussian_cv_device(2, None).unwrap();
        device.connect().await.unwrap();

        device.squeezing(0, 1.0, 0.0).await.unwrap();

        let state = device.get_mode_state(0).unwrap();
        assert!(state.squeezing_parameter > 0.0);
    }

    #[tokio::test]
    async fn test_homodyne_measurement() {
        let mut device = create_gaussian_cv_device(2, None).unwrap();
        device.connect().await.unwrap();

        // Displace the mode first
        device.displacement(0, 2.0, 0.0).await.unwrap();

        let result = device.homodyne_measurement(0, 0.0).await.unwrap();
        assert!(result.is_finite());

        assert_eq!(device.measurement_history.len(), 1);
    }

    #[test]
    fn test_complex_number_operations() {
        let z1 = Complex::new(1.0, 2.0);
        let z2 = Complex::new(3.0, 4.0);

        let sum = z1 + z2;
        assert_eq!(sum.real, 4.0);
        assert_eq!(sum.imag, 6.0);

        let product = z1 * z2;
        assert_eq!(product.real, -5.0);
        assert_eq!(product.imag, 10.0);

        assert!((z1.magnitude() - (5.0_f64).sqrt()).abs() < 1e-10);
    }
}
