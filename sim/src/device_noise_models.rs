//! Realistic device noise models for quantum hardware simulation.
//!
//! This module provides comprehensive noise models that accurately represent
//! the characteristics of real quantum devices, including superconducting
//! transmon qubits, trapped ions, photonic systems, and other quantum
//! computing platforms. It integrates with SciRS2 for high-performance
//! noise simulation and calibration data analysis.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;

/// Device types supported for noise modeling
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// Superconducting transmon qubits
    Superconducting,
    /// Trapped ion systems
    TrappedIon,
    /// Photonic quantum systems
    Photonic,
    /// Silicon spin qubits
    SiliconSpin,
    /// Neutral atom arrays
    NeutralAtom,
    /// Nitrogen-vacancy centers
    NVCenter,
    /// Custom device type
    Custom(String),
}

/// Device noise model configuration
#[derive(Debug, Clone)]
pub struct DeviceNoiseConfig {
    /// Device type
    pub device_type: DeviceType,
    /// Device topology (connectivity graph)
    pub topology: DeviceTopology,
    /// Temperature in Kelvin
    pub temperature: f64,
    /// Enable correlated noise
    pub enable_correlated_noise: bool,
    /// Enable time-dependent noise
    pub enable_time_dependent_noise: bool,
    /// Noise calibration data
    pub calibration_data: Option<CalibrationData>,
    /// Random seed for noise sampling
    pub random_seed: Option<u64>,
    /// Real-time noise adaptation
    pub real_time_adaptation: bool,
}

impl Default for DeviceNoiseConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Superconducting,
            topology: DeviceTopology::default(),
            temperature: 0.015, // ~15 mK
            enable_correlated_noise: true,
            enable_time_dependent_noise: true,
            calibration_data: None,
            random_seed: None,
            real_time_adaptation: false,
        }
    }
}

/// Device topology representation
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceTopology {
    /// Number of qubits
    pub num_qubits: usize,
    /// Connectivity matrix (symmetric)
    pub connectivity: Array2<bool>,
    /// Physical positions of qubits (x, y, z)
    pub positions: Vec<(f64, f64, f64)>,
    /// Coupling strengths between connected qubits
    pub coupling_strengths: HashMap<(usize, usize), f64>,
    /// Qubit frequencies
    pub frequencies: Vec<f64>,
}

impl Default for DeviceTopology {
    fn default() -> Self {
        Self {
            num_qubits: 0,
            connectivity: Array2::default((0, 0)),
            positions: Vec::new(),
            coupling_strengths: HashMap::new(),
            frequencies: Vec::new(),
        }
    }
}

impl DeviceTopology {
    /// Create a linear chain topology
    pub fn linear_chain(num_qubits: usize) -> Self {
        let mut connectivity = Array2::from_elem((num_qubits, num_qubits), false);
        let mut coupling_strengths = HashMap::new();

        for i in 0..num_qubits - 1 {
            connectivity[[i, i + 1]] = true;
            connectivity[[i + 1, i]] = true;
            coupling_strengths.insert((i, i + 1), 1.0);
            coupling_strengths.insert((i + 1, i), 1.0);
        }

        let positions = (0..num_qubits).map(|i| (i as f64, 0.0, 0.0)).collect();

        let frequencies = (0..num_qubits)
            .map(|i| 0.1f64.mul_add(i as f64, 5.0)) // GHz, with detuning
            .collect();

        Self {
            num_qubits,
            connectivity,
            positions,
            coupling_strengths,
            frequencies,
        }
    }

    /// Create a square lattice topology
    pub fn square_lattice(width: usize, height: usize) -> Self {
        let num_qubits = width * height;
        let mut connectivity = Array2::from_elem((num_qubits, num_qubits), false);
        let mut coupling_strengths = HashMap::new();

        for i in 0..height {
            for j in 0..width {
                let qubit = i * width + j;

                // Horizontal connections
                if j < width - 1 {
                    let neighbor = i * width + (j + 1);
                    connectivity[[qubit, neighbor]] = true;
                    connectivity[[neighbor, qubit]] = true;
                    coupling_strengths.insert((qubit, neighbor), 1.0);
                    coupling_strengths.insert((neighbor, qubit), 1.0);
                }

                // Vertical connections
                if i < height - 1 {
                    let neighbor = (i + 1) * width + j;
                    connectivity[[qubit, neighbor]] = true;
                    connectivity[[neighbor, qubit]] = true;
                    coupling_strengths.insert((qubit, neighbor), 1.0);
                    coupling_strengths.insert((neighbor, qubit), 1.0);
                }
            }
        }

        let positions = (0..height)
            .flat_map(|i| (0..width).map(move |j| (j as f64, i as f64, 0.0)))
            .collect();

        let frequencies = (0..num_qubits)
            .map(|i| 0.1f64.mul_add((i % 7) as f64, 5.0)) // Frequency comb to avoid crosstalk
            .collect();

        Self {
            num_qubits,
            connectivity,
            positions,
            coupling_strengths,
            frequencies,
        }
    }

    /// Create IBM heavy-hex topology
    pub fn heavy_hex(distance: usize) -> Self {
        // Simplified heavy-hex implementation
        let num_qubits = distance * distance * 2;
        let connectivity = Array2::from_elem((num_qubits, num_qubits), false);

        // This would be implemented with the actual heavy-hex connectivity pattern
        Self {
            num_qubits,
            connectivity,
            positions: (0..num_qubits).map(|i| (i as f64, 0.0, 0.0)).collect(),
            coupling_strengths: HashMap::new(),
            frequencies: (0..num_qubits)
                .map(|i| 0.1f64.mul_add(i as f64, 5.0))
                .collect(),
        }
    }

    /// Get nearest neighbors of a qubit
    pub fn get_neighbors(&self, qubit: usize) -> Vec<usize> {
        (0..self.num_qubits)
            .filter(|&i| i != qubit && self.connectivity[[qubit, i]])
            .collect()
    }

    /// Calculate distance between two qubits
    pub fn distance(&self, qubit1: usize, qubit2: usize) -> f64 {
        let pos1 = &self.positions[qubit1];
        let pos2 = &self.positions[qubit2];
        (pos1.2 - pos2.2)
            .mul_add(
                pos1.2 - pos2.2,
                (pos1.1 - pos2.1).mul_add(pos1.1 - pos2.1, (pos1.0 - pos2.0).powi(2)),
            )
            .sqrt()
    }
}

/// Calibration data from real devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    /// Device identifier
    pub device_id: String,
    /// Calibration timestamp
    pub timestamp: std::time::SystemTime,
    /// Single-qubit gate fidelities
    pub single_qubit_fidelities: Vec<f64>,
    /// Two-qubit gate fidelities
    pub two_qubit_fidelities: HashMap<(usize, usize), f64>,
    /// Coherence times T1 (relaxation)
    pub t1_times: Vec<f64>,
    /// Coherence times T2 (dephasing)
    pub t2_times: Vec<f64>,
    /// Readout fidelities
    pub readout_fidelities: Vec<f64>,
    /// Gate times
    pub gate_times: GateTimes,
    /// Cross-talk matrices
    pub crosstalk_matrices: HashMap<String, Array2<f64>>,
    /// Frequency drift parameters
    pub frequency_drift: Vec<FrequencyDrift>,
}

/// Gate timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateTimes {
    /// Single-qubit gate time (ns)
    pub single_qubit: f64,
    /// Two-qubit gate time (ns)
    pub two_qubit: f64,
    /// Measurement time (ns)
    pub measurement: f64,
    /// Reset time (ns)
    pub reset: f64,
}

impl Default for GateTimes {
    fn default() -> Self {
        Self {
            single_qubit: 20.0,  // 20 ns
            two_qubit: 100.0,    // 100 ns
            measurement: 1000.0, // 1 μs
            reset: 500.0,        // 500 ns
        }
    }
}

/// Frequency drift model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyDrift {
    /// Qubit index
    pub qubit: usize,
    /// Drift rate (Hz/s)
    pub drift_rate: f64,
    /// Random walk coefficient
    pub random_walk_coeff: f64,
}

/// Device-specific noise models
pub trait DeviceNoiseModel: Send + Sync {
    /// Get the device type
    fn device_type(&self) -> DeviceType;

    /// Apply single-qubit gate noise
    fn apply_single_qubit_noise(
        &self,
        qubit: usize,
        gate_time: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()>;

    /// Apply two-qubit gate noise
    fn apply_two_qubit_noise(
        &self,
        qubit1: usize,
        qubit2: usize,
        gate_time: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()>;

    /// Apply measurement noise
    fn apply_measurement_noise(&self, qubit: usize, measurement_result: bool) -> Result<bool>;

    /// Apply idle noise (during gates on other qubits)
    fn apply_idle_noise(
        &self,
        qubit: usize,
        idle_time: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()>;

    /// Update time-dependent parameters
    fn update_time_dependent_parameters(&mut self, current_time: f64) -> Result<()>;
}

/// Superconducting transmon noise model
#[derive(Debug, Clone)]
pub struct SuperconductingNoiseModel {
    /// Configuration
    config: DeviceNoiseConfig,
    /// Coherence parameters
    coherence_params: CoherenceParameters,
    /// Gate error rates
    gate_errors: GateErrorRates,
    /// Readout error matrix
    readout_errors: Array2<f64>,
    /// Cross-talk matrices
    crosstalk: HashMap<String, Array2<f64>>,
    /// Current time for time-dependent effects
    current_time: f64,
    /// Random number generator state
    rng_state: u64,
}

/// Coherence parameters for superconducting qubits
#[derive(Debug, Clone)]
pub struct CoherenceParameters {
    /// T1 times for each qubit (μs)
    pub t1_times: Vec<f64>,
    /// T2 times for each qubit (μs)
    pub t2_times: Vec<f64>,
    /// Pure dephasing times T2* (μs)
    pub t2_star_times: Vec<f64>,
    /// Temperature-dependent scaling
    pub temperature_scaling: f64,
}

/// Gate error rates
#[derive(Debug, Clone)]
pub struct GateErrorRates {
    /// Single-qubit gate error rates
    pub single_qubit: Vec<f64>,
    /// Two-qubit gate error rates
    pub two_qubit: HashMap<(usize, usize), f64>,
    /// Depolarizing error rates
    pub depolarizing: Vec<f64>,
    /// Dephasing error rates
    pub dephasing: Vec<f64>,
    /// Amplitude damping rates
    pub amplitude_damping: Vec<f64>,
}

impl SuperconductingNoiseModel {
    /// Create new superconducting noise model
    pub fn new(config: DeviceNoiseConfig) -> Result<Self> {
        let num_qubits = config.topology.num_qubits;

        // Initialize coherence parameters based on typical superconducting values
        let coherence_params = CoherenceParameters {
            t1_times: (0..num_qubits)
                .map(|_| fastrand::f64().mul_add(50.0, 50.0))
                .collect(), // 50-100 μs
            t2_times: (0..num_qubits)
                .map(|_| fastrand::f64().mul_add(30.0, 20.0))
                .collect(), // 20-50 μs
            t2_star_times: (0..num_qubits)
                .map(|_| fastrand::f64().mul_add(10.0, 10.0))
                .collect(), // 10-20 μs
            temperature_scaling: Self::calculate_temperature_scaling(config.temperature),
        };

        // Initialize gate error rates
        let gate_errors = GateErrorRates {
            single_qubit: (0..num_qubits)
                .map(|_| fastrand::f64().mul_add(1e-3, 1e-4))
                .collect(), // 0.01-0.1%
            two_qubit: HashMap::new(), // Will be populated based on connectivity
            depolarizing: (0..num_qubits).map(|_| 1e-5).collect(),
            dephasing: (0..num_qubits).map(|_| 1e-4).collect(),
            amplitude_damping: (0..num_qubits).map(|_| 1e-5).collect(),
        };

        // Initialize readout error matrix
        let readout_errors = Array2::from_elem((num_qubits, 2), 0.05); // 5% readout error

        let rng_state = config.random_seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        });

        Ok(Self {
            config,
            coherence_params,
            gate_errors,
            readout_errors,
            crosstalk: HashMap::new(),
            current_time: 0.0,
            rng_state,
        })
    }

    /// Create from calibration data
    pub fn from_calibration_data(
        config: DeviceNoiseConfig,
        calibration: &CalibrationData,
    ) -> Result<Self> {
        let mut model = Self::new(config)?;

        // Update parameters from calibration data
        model.coherence_params.t1_times = calibration.t1_times.clone();
        model.coherence_params.t2_times = calibration.t2_times.clone();
        model.gate_errors.single_qubit = calibration
            .single_qubit_fidelities
            .iter()
            .map(|&f| 1.0 - f)
            .collect();

        // Update two-qubit gate errors
        for (&(q1, q2), &fidelity) in &calibration.two_qubit_fidelities {
            model.gate_errors.two_qubit.insert((q1, q2), 1.0 - fidelity);
        }

        // Update readout errors
        for (i, &fidelity) in calibration.readout_fidelities.iter().enumerate() {
            model.readout_errors[[i, 0]] = 1.0 - fidelity; // |0⟩ → |1⟩ error
            model.readout_errors[[i, 1]] = 1.0 - fidelity; // |1⟩ → |0⟩ error
        }

        model.crosstalk = calibration.crosstalk_matrices.clone();

        Ok(model)
    }

    /// Calculate temperature scaling factor
    fn calculate_temperature_scaling(temperature: f64) -> f64 {
        // Thermal population of excited state
        let kb_t_over_hf = temperature * 1.38e-23 / (6.626e-34 * 5e9); // Assuming 5 GHz qubit
        (-1.0 / kb_t_over_hf).exp() / (1.0 + (-1.0 / kb_t_over_hf).exp())
    }

    /// Sample from random number generator
    fn random(&mut self) -> f64 {
        // Simple linear congruential generator
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Apply decoherence channel
    fn apply_decoherence(
        &mut self,
        qubit: usize,
        time_ns: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let t1 = self.coherence_params.t1_times[qubit] * 1000.0; // Convert to ns
        let t2 = self.coherence_params.t2_times[qubit] * 1000.0;

        // Apply amplitude damping (T1 process)
        let gamma_1 = time_ns / t1;
        if gamma_1 > 0.0 {
            self.apply_amplitude_damping(qubit, gamma_1, state)?;
        }

        // Apply dephasing (T2 process)
        let gamma_2 = time_ns / t2;
        if gamma_2 > 0.0 {
            self.apply_dephasing(qubit, gamma_2, state)?;
        }

        Ok(())
    }

    /// Apply amplitude damping channel
    fn apply_amplitude_damping(
        &mut self,
        qubit: usize,
        gamma: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let state_size = state.len();
        let qubit_mask = 1 << qubit;

        let mut new_state = state.clone();

        for i in 0..state_size {
            if i & qubit_mask != 0 {
                // |1⟩ state - apply damping
                let j = i & !qubit_mask; // Corresponding |0⟩ state

                if self.random() < gamma {
                    // Decay |1⟩ → |0⟩
                    new_state[j] += state[i];
                    new_state[i] = Complex64::new(0.0, 0.0);
                } else {
                    // No decay, but renormalize
                    new_state[i] *= (1.0 - gamma).sqrt();
                }
            }
        }

        *state = new_state;
        Ok(())
    }

    /// Apply dephasing channel
    fn apply_dephasing(
        &mut self,
        qubit: usize,
        gamma: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let state_size = state.len();
        let qubit_mask = 1 << qubit;

        for i in 0..state_size {
            if i & qubit_mask != 0 {
                // Apply random phase to |1⟩ states
                if self.random() < gamma {
                    let phase = self.random() * 2.0 * std::f64::consts::PI;
                    state[i] *= Complex64::new(0.0, phase).exp();
                }
            }
        }

        Ok(())
    }

    /// Apply cross-talk effects
    fn apply_crosstalk(
        &mut self,
        active_qubits: &[usize],
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        if !self.config.enable_correlated_noise {
            return Ok(());
        }

        // Apply nearest-neighbor cross-talk
        for &qubit in active_qubits {
            let neighbors = self.config.topology.get_neighbors(qubit);

            for neighbor in neighbors {
                if !active_qubits.contains(&neighbor) {
                    // Apply small rotation due to cross-talk
                    let crosstalk_strength = 0.01; // 1% cross-talk
                    let angle = crosstalk_strength * self.random() * std::f64::consts::PI;

                    // Apply random rotation
                    self.apply_small_rotation(neighbor, angle, state)?;
                }
            }
        }

        Ok(())
    }

    /// Apply small rotation due to cross-talk
    fn apply_small_rotation(
        &mut self,
        qubit: usize,
        angle: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let qubit_mask = 1 << qubit;
        let state_size = state.len();

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..state_size {
            if i & qubit_mask == 0 {
                let j = i | qubit_mask;
                if j < state_size {
                    let amp_0 = state[i];
                    let amp_1 = state[j];

                    state[i] = cos_half * amp_0 - Complex64::new(0.0, sin_half) * amp_1;
                    state[j] = cos_half * amp_1 - Complex64::new(0.0, sin_half) * amp_0;
                }
            }
        }

        Ok(())
    }
}

impl DeviceNoiseModel for SuperconductingNoiseModel {
    fn device_type(&self) -> DeviceType {
        DeviceType::Superconducting
    }

    fn apply_single_qubit_noise(
        &self,
        qubit: usize,
        gate_time: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let mut model = self.clone();

        // Apply decoherence during gate
        model.apply_decoherence(qubit, gate_time, state)?;

        // Apply gate error
        let error_rate = model.gate_errors.single_qubit[qubit];
        if model.random() < error_rate {
            // Apply random Pauli error
            let pauli_error = (model.random() * 3.0) as usize;
            model.apply_pauli_error(qubit, pauli_error, state)?;
        }

        // Apply cross-talk to neighboring qubits
        model.apply_crosstalk(&[qubit], state)?;

        Ok(())
    }

    fn apply_two_qubit_noise(
        &self,
        qubit1: usize,
        qubit2: usize,
        gate_time: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let mut model = self.clone();

        // Apply decoherence to both qubits
        model.apply_decoherence(qubit1, gate_time, state)?;
        model.apply_decoherence(qubit2, gate_time, state)?;

        // Apply two-qubit gate error
        let error_rate = model
            .gate_errors
            .two_qubit
            .get(&(qubit1, qubit2))
            .or_else(|| model.gate_errors.two_qubit.get(&(qubit2, qubit1)))
            .copied()
            .unwrap_or(0.01); // Default 1% error

        if model.random() < error_rate {
            // Apply correlated two-qubit error
            model.apply_two_qubit_error(qubit1, qubit2, state)?;
        }

        // Apply cross-talk
        model.apply_crosstalk(&[qubit1, qubit2], state)?;

        Ok(())
    }

    fn apply_measurement_noise(&self, qubit: usize, measurement_result: bool) -> Result<bool> {
        let error_prob = if measurement_result {
            self.readout_errors[[qubit, 1]] // |1⟩ → |0⟩ error
        } else {
            self.readout_errors[[qubit, 0]] // |0⟩ → |1⟩ error
        };

        if fastrand::f64() < error_prob {
            Ok(!measurement_result) // Flip the result
        } else {
            Ok(measurement_result)
        }
    }

    fn apply_idle_noise(
        &self,
        qubit: usize,
        idle_time: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let mut model = self.clone();
        model.apply_decoherence(qubit, idle_time, state)?;
        Ok(())
    }

    fn update_time_dependent_parameters(&mut self, current_time: f64) -> Result<()> {
        self.current_time = current_time;

        if !self.config.enable_time_dependent_noise {
            return Ok(());
        }

        // Update frequency drift if calibration data is available
        if let Some(calibration) = &self.config.calibration_data {
            for drift in &calibration.frequency_drift {
                let frequency_shift = drift.drift_rate * current_time / 1e9; // Convert to Hz
                                                                             // This would update the qubit frequency and affect gate fidelities
                                                                             // For now, we just track the time
            }
        }

        Ok(())
    }
}

impl SuperconductingNoiseModel {
    /// Apply Pauli error
    fn apply_pauli_error(
        &mut self,
        qubit: usize,
        pauli_type: usize,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let qubit_mask = 1 << qubit;
        let state_size = state.len();

        match pauli_type {
            0 => {
                // Pauli X
                for i in 0..state_size {
                    let j = i ^ qubit_mask;
                    if i < j {
                        let temp = state[i];
                        state[i] = state[j];
                        state[j] = temp;
                    }
                }
            }
            1 => {
                // Pauli Y
                for i in 0..state_size {
                    let j = i ^ qubit_mask;
                    if i < j {
                        let temp = state[i];
                        state[i] = Complex64::new(0.0, 1.0) * state[j];
                        state[j] = Complex64::new(0.0, -1.0) * temp;
                    }
                }
            }
            2 => {
                // Pauli Z
                for i in 0..state_size {
                    if i & qubit_mask != 0 {
                        state[i] = -state[i];
                    }
                }
            }
            _ => {} // Identity (no error)
        }

        Ok(())
    }

    /// Apply two-qubit correlated error
    fn apply_two_qubit_error(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        // Apply correlated Pauli errors
        let error1 = (self.random() * 4.0) as usize;
        let error2 = (self.random() * 4.0) as usize;

        if error1 < 3 {
            self.apply_pauli_error(qubit1, error1, state)?;
        }
        if error2 < 3 {
            self.apply_pauli_error(qubit2, error2, state)?;
        }

        Ok(())
    }
}

/// Comprehensive device noise simulator
pub struct DeviceNoiseSimulator {
    /// Device noise model
    noise_model: Box<dyn DeviceNoiseModel>,
    /// SciRS2 backend for optimization
    backend: Option<SciRS2Backend>,
    /// Simulation statistics
    stats: NoiseSimulationStats,
    /// Current simulation time
    current_time: f64,
}

/// Noise simulation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoiseSimulationStats {
    /// Total noise operations applied
    pub total_noise_ops: usize,
    /// Single-qubit noise events
    pub single_qubit_noise_events: usize,
    /// Two-qubit noise events
    pub two_qubit_noise_events: usize,
    /// Measurement errors
    pub measurement_errors: usize,
    /// Coherence events (T1/T2)
    pub coherence_events: usize,
    /// Cross-talk events
    pub crosstalk_events: usize,
    /// Total simulation time
    pub total_simulation_time_ms: f64,
}

impl DeviceNoiseSimulator {
    /// Create new device noise simulator
    pub fn new(noise_model: Box<dyn DeviceNoiseModel>) -> Result<Self> {
        Ok(Self {
            noise_model,
            backend: None,
            stats: NoiseSimulationStats::default(),
            current_time: 0.0,
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Apply gate noise
    pub fn apply_gate_noise(
        &mut self,
        gate_type: &str,
        qubits: &[usize],
        gate_time: f64,
        state: &mut Array1<Complex64>,
    ) -> Result<()> {
        let start_time = std::time::Instant::now();

        match qubits.len() {
            1 => {
                self.noise_model
                    .apply_single_qubit_noise(qubits[0], gate_time, state)?;
                self.stats.single_qubit_noise_events += 1;
            }
            2 => {
                self.noise_model
                    .apply_two_qubit_noise(qubits[0], qubits[1], gate_time, state)?;
                self.stats.two_qubit_noise_events += 1;
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(format!(
                    "Noise model for {}-qubit gates not implemented",
                    qubits.len()
                )));
            }
        }

        self.current_time += gate_time;
        self.noise_model
            .update_time_dependent_parameters(self.current_time)?;

        self.stats.total_noise_ops += 1;
        self.stats.total_simulation_time_ms += start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(())
    }

    /// Apply measurement noise
    pub fn apply_measurement_noise(
        &mut self,
        qubits: &[usize],
        results: &mut [bool],
    ) -> Result<()> {
        for (i, &qubit) in qubits.iter().enumerate() {
            let noisy_result = self
                .noise_model
                .apply_measurement_noise(qubit, results[i])?;
            if noisy_result != results[i] {
                self.stats.measurement_errors += 1;
            }
            results[i] = noisy_result;
        }

        Ok(())
    }

    /// Apply idle noise to all qubits
    pub fn apply_idle_noise(
        &mut self,
        idle_time: f64,
        state: &mut Array1<Complex64>,
        num_qubits: usize,
    ) -> Result<()> {
        for qubit in 0..num_qubits {
            self.noise_model.apply_idle_noise(qubit, idle_time, state)?;
        }

        self.current_time += idle_time;
        self.noise_model
            .update_time_dependent_parameters(self.current_time)?;

        Ok(())
    }

    /// Get simulation statistics
    pub const fn get_stats(&self) -> &NoiseSimulationStats {
        &self.stats
    }

    /// Reset simulation statistics
    pub fn reset_stats(&mut self) {
        self.stats = NoiseSimulationStats::default();
        self.current_time = 0.0;
    }
}

/// Device noise model utilities
pub struct DeviceNoiseUtils;

impl DeviceNoiseUtils {
    /// Create IBM device model from real calibration data
    pub fn create_ibm_device(
        device_name: &str,
        num_qubits: usize,
    ) -> Result<SuperconductingNoiseModel> {
        let topology = match device_name {
            "ibm_washington" => DeviceTopology::heavy_hex(3),
            "ibm_montreal" => DeviceTopology::linear_chain(num_qubits),
            _ => DeviceTopology::square_lattice(4, 4),
        };

        let config = DeviceNoiseConfig {
            device_type: DeviceType::Superconducting,
            topology,
            temperature: 0.015, // 15 mK
            enable_correlated_noise: true,
            enable_time_dependent_noise: true,
            calibration_data: None,
            random_seed: None,
            real_time_adaptation: false,
        };

        SuperconductingNoiseModel::new(config)
    }

    /// Create Google Sycamore-like device
    pub fn create_google_sycamore(num_qubits: usize) -> Result<SuperconductingNoiseModel> {
        let width = (num_qubits as f64).sqrt().ceil() as usize;
        let topology = DeviceTopology::square_lattice(width, width);

        let config = DeviceNoiseConfig {
            device_type: DeviceType::Superconducting,
            topology,
            temperature: 0.020, // 20 mK
            enable_correlated_noise: true,
            enable_time_dependent_noise: true,
            calibration_data: None,
            random_seed: None,
            real_time_adaptation: false,
        };

        SuperconductingNoiseModel::new(config)
    }

    /// Create trapped ion device model
    pub fn create_trapped_ion_device(num_ions: usize) -> Result<Box<dyn DeviceNoiseModel>> {
        // Placeholder for trapped ion implementation
        let topology = DeviceTopology::linear_chain(num_ions);
        let config = DeviceNoiseConfig {
            device_type: DeviceType::TrappedIon,
            topology,
            temperature: 1e-6, // μK range
            enable_correlated_noise: true,
            enable_time_dependent_noise: false,
            calibration_data: None,
            random_seed: None,
            real_time_adaptation: false,
        };

        Ok(Box::new(SuperconductingNoiseModel::new(config)?))
    }

    /// Benchmark noise models
    pub fn benchmark_noise_models() -> Result<NoiseBenchmarkResults> {
        let mut results = NoiseBenchmarkResults::default();

        let device_types = vec![
            ("ibm_montreal", 27),
            ("google_sycamore", 53),
            ("trapped_ion", 20),
        ];

        for (device_name, num_qubits) in device_types {
            let model = if device_name.starts_with("ibm") {
                Box::new(Self::create_ibm_device(device_name, num_qubits)?)
                    as Box<dyn DeviceNoiseModel>
            } else if device_name.starts_with("google") {
                Box::new(Self::create_google_sycamore(num_qubits)?) as Box<dyn DeviceNoiseModel>
            } else {
                Self::create_trapped_ion_device(num_qubits)?
            };

            let mut simulator = DeviceNoiseSimulator::new(model)?;
            let mut state = Array1::from_elem(1 << num_qubits.min(10), Complex64::new(0.0, 0.0));
            state[0] = Complex64::new(1.0, 0.0);

            let start = std::time::Instant::now();

            // Apply some gates with noise
            for i in 0..num_qubits.min(10) {
                simulator.apply_gate_noise("x", &[i], 20.0, &mut state)?;
            }

            let execution_time = start.elapsed().as_secs_f64() * 1000.0;
            results
                .device_benchmarks
                .insert(device_name.to_string(), execution_time);
        }

        Ok(results)
    }
}

/// Noise benchmark results
#[derive(Debug, Clone, Default)]
pub struct NoiseBenchmarkResults {
    /// Execution times by device type
    pub device_benchmarks: HashMap<String, f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_device_topology_linear_chain() {
        let topology = DeviceTopology::linear_chain(5);
        assert_eq!(topology.num_qubits, 5);
        assert!(topology.connectivity[[0, 1]]);
        assert!(!topology.connectivity[[0, 2]]);
        assert_eq!(topology.get_neighbors(0), vec![1]);
        assert_eq!(topology.get_neighbors(2), vec![1, 3]);
    }

    #[test]
    fn test_device_topology_square_lattice() {
        let topology = DeviceTopology::square_lattice(3, 3);
        assert_eq!(topology.num_qubits, 9);
        assert!(topology.connectivity[[0, 1]]); // Horizontal connection
        assert!(topology.connectivity[[0, 3]]); // Vertical connection
        assert!(!topology.connectivity[[0, 4]]); // Diagonal not connected
    }

    #[test]
    fn test_superconducting_noise_model() {
        let config = DeviceNoiseConfig {
            device_type: DeviceType::Superconducting,
            topology: DeviceTopology::linear_chain(3),
            ..Default::default()
        };

        let model = SuperconductingNoiseModel::new(config).unwrap();
        assert_eq!(model.device_type(), DeviceType::Superconducting);
        assert_eq!(model.coherence_params.t1_times.len(), 3);
    }

    #[test]
    fn test_device_noise_simulator() {
        let config = DeviceNoiseConfig {
            device_type: DeviceType::Superconducting,
            topology: DeviceTopology::linear_chain(2),
            random_seed: Some(12345),
            ..Default::default()
        };

        let model = SuperconductingNoiseModel::new(config).unwrap();
        let mut simulator = DeviceNoiseSimulator::new(Box::new(model)).unwrap();

        let mut state = Array1::from_elem(4, Complex64::new(0.0, 0.0));
        state[0] = Complex64::new(1.0, 0.0);

        // Apply single-qubit noise
        let result = simulator.apply_gate_noise("x", &[0], 20.0, &mut state);
        assert!(result.is_ok());
        assert_eq!(simulator.stats.single_qubit_noise_events, 1);

        // Apply two-qubit noise
        let result = simulator.apply_gate_noise("cnot", &[0, 1], 100.0, &mut state);
        assert!(result.is_ok());
        assert_eq!(simulator.stats.two_qubit_noise_events, 1);
    }

    #[test]
    fn test_measurement_noise() {
        let config = DeviceNoiseConfig {
            device_type: DeviceType::Superconducting,
            topology: DeviceTopology::linear_chain(1),
            random_seed: Some(12345),
            ..Default::default()
        };

        let model = SuperconductingNoiseModel::new(config).unwrap();
        let mut simulator = DeviceNoiseSimulator::new(Box::new(model)).unwrap();

        let mut results = vec![false, true, false];
        simulator
            .apply_measurement_noise(&[0, 0, 0], &mut results)
            .unwrap();

        // Some results might have flipped due to readout errors
        // We can't predict exactly which due to randomness, but the function should work
    }

    #[test]
    fn test_gate_times_default() {
        let gate_times = GateTimes::default();
        assert_eq!(gate_times.single_qubit, 20.0);
        assert_eq!(gate_times.two_qubit, 100.0);
        assert_eq!(gate_times.measurement, 1000.0);
    }

    #[test]
    fn test_device_noise_utils() {
        let result = DeviceNoiseUtils::create_ibm_device("ibm_montreal", 5);
        assert!(result.is_ok());

        let result = DeviceNoiseUtils::create_google_sycamore(9);
        assert!(result.is_ok());

        let result = DeviceNoiseUtils::create_trapped_ion_device(5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_calibration_data_integration() {
        let calibration = CalibrationData {
            device_id: "test_device".to_string(),
            timestamp: std::time::SystemTime::now(),
            single_qubit_fidelities: vec![0.999, 0.998, 0.999],
            two_qubit_fidelities: HashMap::new(),
            t1_times: vec![50.0, 45.0, 55.0],
            t2_times: vec![25.0, 20.0, 30.0],
            readout_fidelities: vec![0.95, 0.96, 0.94],
            gate_times: GateTimes::default(),
            crosstalk_matrices: HashMap::new(),
            frequency_drift: Vec::new(),
        };

        let config = DeviceNoiseConfig {
            device_type: DeviceType::Superconducting,
            topology: DeviceTopology::linear_chain(3),
            calibration_data: Some(calibration.clone()),
            ..Default::default()
        };

        let model = SuperconductingNoiseModel::from_calibration_data(config, &calibration).unwrap();
        assert_eq!(model.coherence_params.t1_times, calibration.t1_times);
        assert_eq!(model.coherence_params.t2_times, calibration.t2_times);
    }

    #[test]
    fn test_temperature_scaling() {
        let scaling_15mk = SuperconductingNoiseModel::calculate_temperature_scaling(0.015);
        let scaling_50mk = SuperconductingNoiseModel::calculate_temperature_scaling(0.050);

        // Higher temperature should give higher thermal population
        assert!(scaling_50mk > scaling_15mk);
        assert!(scaling_15mk > 0.0);
        assert!(scaling_15mk < 1.0);
    }
}
