//! Pulse-level gate compilation for superconducting qubits
//!
//! This module provides comprehensive pulse-level control for superconducting quantum devices,
//! including gate-to-pulse compilation, calibration management, and hardware-specific optimizations.

use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use rustc_hash::FxHashMap;
use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use std::f64::consts::PI;

/// Pulse envelope function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PulseEnvelope {
    /// Gaussian envelope with specified width
    Gaussian { sigma: f64 },
    /// Derivative of Gaussian (DRAG pulse)
    DRAG { sigma: f64, beta: f64 },
    /// Square/rectangular pulse
    Square,
    /// Raised cosine (cosine squared) envelope
    RaisedCosine,
    /// Hyperbolic secant envelope
    HyperbolicSecant { width: f64 },
    /// Hermite Gaussian envelope
    HermiteGaussian { n: usize, sigma: f64 },
}

impl PulseEnvelope {
    /// Evaluate the envelope at time t (normalized to pulse duration)
    pub fn evaluate(&self, t: f64) -> f64 {
        match self {
            PulseEnvelope::Gaussian { sigma } => {
                let t_norm = (t - 0.5) / sigma;
                (-0.5 * t_norm * t_norm).exp()
            }
            PulseEnvelope::DRAG { sigma, beta: _ } => {
                let t_norm = (t - 0.5) / sigma;
                (-0.5 * t_norm * t_norm).exp()
            }
            PulseEnvelope::Square => {
                if t >= 0.0 && t <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            PulseEnvelope::RaisedCosine => {
                if t >= 0.0 && t <= 1.0 {
                    let phase = 2.0 * PI * t;
                    0.5 * (1.0 - phase.cos())
                } else {
                    0.0
                }
            }
            PulseEnvelope::HyperbolicSecant { width } => {
                let t_scaled = (t - 0.5) / width;
                1.0 / t_scaled.cosh()
            }
            PulseEnvelope::HermiteGaussian { n, sigma } => {
                let t_norm = (t - 0.5) / sigma;
                let gaussian = (-0.5 * t_norm * t_norm).exp();
                let hermite = self.hermite_polynomial(*n, t_norm);
                gaussian * hermite
            }
        }
    }

    /// Calculate Hermite polynomial value
    fn hermite_polynomial(&self, n: usize, x: f64) -> f64 {
        match n {
            0 => 1.0,
            1 => 2.0 * x,
            _ => {
                let mut h_prev_prev = 1.0;
                let mut h_prev = 2.0 * x;
                for i in 2..=n {
                    let h_curr = 2.0 * x * h_prev - 2.0 * (i - 1) as f64 * h_prev_prev;
                    h_prev_prev = h_prev;
                    h_prev = h_curr;
                }
                h_prev
            }
        }
    }

    /// Get DRAG derivative component for DRAG pulses
    pub fn drag_derivative(&self, t: f64) -> f64 {
        match self {
            PulseEnvelope::DRAG { sigma, beta } => {
                let t_norm = (t - 0.5) / sigma;
                let gaussian = (-0.5 * t_norm * t_norm).exp();
                let derivative = -t_norm / sigma * gaussian;
                beta * derivative
            }
            _ => 0.0,
        }
    }
}

/// Pulse waveform for control signals
pub struct Pulse {
    /// Duration of the pulse in nanoseconds
    pub duration: f64,
    /// Amplitude (0.0 to 1.0, scaled by hardware limits)
    pub amplitude: f64,
    /// Frequency in GHz
    pub frequency: f64,
    /// Phase in radians
    pub phase: f64,
    /// Pulse envelope shape
    pub envelope: PulseEnvelope,
    /// Sampling rate in GSa/s
    pub sample_rate: f64,
    /// Additional phase modulation function
    pub phase_modulation: Option<Box<dyn Fn(f64) -> f64 + Send + Sync>>,
}

impl std::fmt::Debug for Pulse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pulse")
            .field("duration", &self.duration)
            .field("amplitude", &self.amplitude)
            .field("frequency", &self.frequency)
            .field("phase", &self.phase)
            .field("envelope", &self.envelope)
            .field("sample_rate", &self.sample_rate)
            .field("phase_modulation", &self.phase_modulation.is_some())
            .finish()
    }
}

impl Clone for Pulse {
    fn clone(&self) -> Self {
        Self {
            duration: self.duration,
            amplitude: self.amplitude,
            frequency: self.frequency,
            phase: self.phase,
            envelope: self.envelope.clone(),
            sample_rate: self.sample_rate,
            phase_modulation: None, // Cannot clone function objects, set to None
        }
    }
}

impl Pulse {
    /// Create a new pulse
    pub fn new(
        duration: f64,
        amplitude: f64,
        frequency: f64,
        phase: f64,
        envelope: PulseEnvelope,
        sample_rate: f64,
    ) -> Self {
        Self {
            duration,
            amplitude,
            frequency,
            phase,
            envelope,
            sample_rate,
            phase_modulation: None,
        }
    }

    /// Generate pulse waveform samples
    pub fn generate_waveform(&self) -> QuantRS2Result<Array1<Complex64>> {
        let num_samples = (self.duration * self.sample_rate).ceil() as usize;
        let dt = 1.0 / self.sample_rate;

        let mut waveform = Array1::zeros(num_samples);

        for i in 0..num_samples {
            let t = i as f64 * dt;
            let t_norm = t / self.duration;

            let envelope_value = self.envelope.evaluate(t_norm);
            let phase_mod = if let Some(ref phase_fn) = self.phase_modulation {
                phase_fn(t)
            } else {
                0.0
            };

            let total_phase = 2.0 * PI * self.frequency * t + self.phase + phase_mod;
            let complex_amplitude = Complex64::new(0.0, total_phase).exp();

            waveform[i] = self.amplitude * envelope_value * complex_amplitude;
        }

        Ok(waveform)
    }

    /// Generate DRAG pulse (for reduced leakage)
    pub fn generate_drag_waveform(&self) -> QuantRS2Result<(Array1<Complex64>, Array1<Complex64>)> {
        let num_samples = (self.duration * self.sample_rate).ceil() as usize;
        let dt = 1.0 / self.sample_rate;

        let mut i_component = Array1::zeros(num_samples);
        let mut q_component = Array1::zeros(num_samples);

        for i in 0..num_samples {
            let t = i as f64 * dt;
            let t_norm = t / self.duration;

            let envelope_value = self.envelope.evaluate(t_norm);
            let drag_derivative = self.envelope.drag_derivative(t_norm);

            let phase_mod = if let Some(ref phase_fn) = self.phase_modulation {
                phase_fn(t)
            } else {
                0.0
            };

            let total_phase = 2.0 * PI * self.frequency * t + self.phase + phase_mod;

            // I component: normal pulse
            i_component[i] =
                Complex64::new(self.amplitude * envelope_value * total_phase.cos(), 0.0);

            // Q component: DRAG correction
            q_component[i] = Complex64::new(
                self.amplitude * (envelope_value * total_phase.sin() + drag_derivative),
                0.0,
            );
        }

        Ok((i_component, q_component))
    }
}

/// Qubit control parameters for superconducting devices
#[derive(Debug, Clone)]
pub struct QubitControlParams {
    /// Drive frequency for X/Y gates (GHz)
    pub drive_frequency: f64,
    /// Anharmonicity (MHz)
    pub anharmonicity: f64,
    /// Rabi frequency for π pulse (MHz)
    pub rabi_frequency: f64,
    /// T1 relaxation time (μs)
    pub t1: f64,
    /// T2 dephasing time (μs)
    pub t2: f64,
    /// Gate time for single-qubit gates (ns)
    pub gate_time: f64,
    /// Calibrated π pulse amplitude
    pub pi_pulse_amplitude: f64,
    /// DRAG parameter for leakage suppression
    pub drag_parameter: f64,
}

impl Default for QubitControlParams {
    fn default() -> Self {
        Self {
            drive_frequency: 5.0,  // 5 GHz typical
            anharmonicity: -200.0, // -200 MHz typical
            rabi_frequency: 20.0,  // 20 MHz
            t1: 50.0,              // 50 μs
            t2: 30.0,              // 30 μs
            gate_time: 25.0,       // 25 ns
            pi_pulse_amplitude: 0.5,
            drag_parameter: 0.5,
        }
    }
}

/// Two-qubit coupling parameters
#[derive(Debug, Clone)]
pub struct CouplingParams {
    /// Coupling strength (MHz)
    pub coupling_strength: f64,
    /// Cross-talk coefficient
    pub crosstalk: f64,
    /// ZZ interaction strength (kHz)
    pub zz_coupling: f64,
}

impl Default for CouplingParams {
    fn default() -> Self {
        Self {
            coupling_strength: 10.0, // 10 MHz
            crosstalk: 0.02,         // 2% crosstalk
            zz_coupling: 50.0,       // 50 kHz
        }
    }
}

/// Hardware calibration data for superconducting quantum processor
#[derive(Debug, Clone)]
pub struct HardwareCalibration {
    /// Single-qubit control parameters
    pub qubit_params: FxHashMap<QubitId, QubitControlParams>,
    /// Two-qubit coupling parameters
    pub coupling_params: FxHashMap<(QubitId, QubitId), CouplingParams>,
    /// Flux control parameters for tunable couplers
    pub flux_params: FxHashMap<QubitId, f64>,
    /// Readout parameters
    pub readout_params: FxHashMap<QubitId, (f64, f64)>, // (frequency, amplitude)
    /// Global timing constraints
    pub timing_constraints: TimingConstraints,
}

/// Timing constraints for pulse sequences
#[derive(Debug, Clone)]
pub struct TimingConstraints {
    /// Minimum time between pulses (ns)
    pub min_pulse_separation: f64,
    /// Maximum pulse duration (ns)
    pub max_pulse_duration: f64,
    /// Sampling rate (GSa/s)
    pub sample_rate: f64,
    /// Clock resolution (ns)
    pub clock_resolution: f64,
}

impl Default for TimingConstraints {
    fn default() -> Self {
        Self {
            min_pulse_separation: 2.0,
            max_pulse_duration: 1000.0,
            sample_rate: 2.0,      // 2 GSa/s
            clock_resolution: 0.5, // 0.5 ns
        }
    }
}

impl Default for HardwareCalibration {
    fn default() -> Self {
        Self {
            qubit_params: FxHashMap::default(),
            coupling_params: FxHashMap::default(),
            flux_params: FxHashMap::default(),
            readout_params: FxHashMap::default(),
            timing_constraints: TimingConstraints::default(),
        }
    }
}

/// Pulse sequence for implementing quantum gates
#[derive(Debug, Clone)]
pub struct PulseSequence {
    /// List of pulses with timing information
    pub pulses: Vec<(f64, QubitId, Pulse)>, // (start_time, qubit, pulse)
    /// Total sequence duration (ns)
    pub duration: f64,
    /// Sequence name/identifier
    pub name: String,
}

impl PulseSequence {
    /// Create a new pulse sequence
    pub fn new(name: String) -> Self {
        Self {
            pulses: Vec::new(),
            duration: 0.0,
            name,
        }
    }

    /// Add a pulse to the sequence
    pub fn add_pulse(&mut self, start_time: f64, qubit: QubitId, pulse: Pulse) {
        let end_time = start_time + pulse.duration;
        if end_time > self.duration {
            self.duration = end_time;
        }
        self.pulses.push((start_time, qubit, pulse));
    }

    /// Get pulses for a specific qubit
    pub fn get_qubit_pulses(&self, qubit: QubitId) -> Vec<&(f64, QubitId, Pulse)> {
        self.pulses.iter().filter(|(_, q, _)| *q == qubit).collect()
    }

    /// Check for pulse overlaps on the same qubit
    pub fn check_overlaps(&self) -> QuantRS2Result<()> {
        let mut qubit_timings: FxHashMap<QubitId, Vec<(f64, f64)>> = FxHashMap::default();

        for (start_time, qubit, pulse) in &self.pulses {
            let end_time = start_time + pulse.duration;
            let timings = qubit_timings.entry(*qubit).or_default();

            for &(existing_start, existing_end) in timings.iter() {
                if start_time < &existing_end && end_time > existing_start {
                    return Err(QuantRS2Error::InvalidOperation(format!(
                        "Pulse overlap detected on qubit {:?}",
                        qubit
                    )));
                }
            }

            timings.push((*start_time, end_time));
        }

        Ok(())
    }
}

/// Gate-to-pulse compiler for superconducting qubits
pub struct PulseCompiler {
    /// Hardware calibration data
    pub calibration: HardwareCalibration,
    /// Pulse library for common gates
    pub pulse_library: FxHashMap<String, Box<dyn Fn(&QubitControlParams) -> Pulse + Send + Sync>>,
}

impl std::fmt::Debug for PulseCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PulseCompiler")
            .field("calibration", &self.calibration)
            .field(
                "pulse_library_keys",
                &self.pulse_library.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl PulseCompiler {
    /// Create a new pulse compiler
    pub fn new(calibration: HardwareCalibration) -> Self {
        let mut compiler = Self {
            calibration,
            pulse_library: FxHashMap::default(),
        };

        compiler.initialize_pulse_library();
        compiler
    }

    /// Initialize the standard pulse library
    fn initialize_pulse_library(&mut self) {
        // X gate (π rotation around X axis)
        self.pulse_library.insert(
            "X".to_string(),
            Box::new(|params| {
                Pulse::new(
                    params.gate_time,
                    params.pi_pulse_amplitude,
                    params.drive_frequency,
                    0.0, // X gate phase
                    PulseEnvelope::DRAG {
                        sigma: params.gate_time / 6.0,
                        beta: params.drag_parameter,
                    },
                    2.0, // 2 GSa/s
                )
            }),
        );

        // Y gate (π rotation around Y axis)
        self.pulse_library.insert(
            "Y".to_string(),
            Box::new(|params| {
                Pulse::new(
                    params.gate_time,
                    params.pi_pulse_amplitude,
                    params.drive_frequency,
                    PI / 2.0, // Y gate phase (90° phase shift from X)
                    PulseEnvelope::DRAG {
                        sigma: params.gate_time / 6.0,
                        beta: params.drag_parameter,
                    },
                    2.0,
                )
            }),
        );

        // Hadamard gate (composite pulse)
        self.pulse_library.insert(
            "H".to_string(),
            Box::new(|params| {
                // Simplified single-pulse implementation
                // In practice, this would be a composite sequence
                Pulse::new(
                    params.gate_time,
                    params.pi_pulse_amplitude * 0.707, // √2/2
                    params.drive_frequency,
                    PI / 4.0, // 45° phase
                    PulseEnvelope::DRAG {
                        sigma: params.gate_time / 6.0,
                        beta: params.drag_parameter,
                    },
                    2.0,
                )
            }),
        );

        // RZ gate (virtual Z rotation - phase tracking only)
        self.pulse_library.insert(
            "RZ".to_string(),
            Box::new(|_params| {
                // Virtual gate - no physical pulse needed
                Pulse::new(
                    0.0, // No duration
                    0.0, // No amplitude
                    0.0, // No frequency
                    0.0, // Phase handled virtually
                    PulseEnvelope::Square,
                    2.0,
                )
            }),
        );
    }

    /// Compile a gate to pulse sequence
    pub fn compile_gate(
        &self,
        gate: &dyn GateOp,
        qubits: &[QubitId],
    ) -> QuantRS2Result<PulseSequence> {
        let gate_name = gate.name();
        let mut sequence = PulseSequence::new(gate_name.to_string());

        match gate_name {
            "X" | "Y" | "H" => {
                if qubits.len() != 1 {
                    return Err(QuantRS2Error::InvalidOperation(
                        "Single-qubit gate requires exactly one qubit".to_string(),
                    ));
                }

                let qubit = qubits[0];
                let default_params = QubitControlParams::default();
                let params = self
                    .calibration
                    .qubit_params
                    .get(&qubit)
                    .unwrap_or(&default_params);

                if let Some(pulse_fn) = self.pulse_library.get(gate_name) {
                    let pulse = pulse_fn(params);
                    sequence.add_pulse(0.0, qubit, pulse);
                }
            }
            "CNOT" | "CX" => {
                if qubits.len() != 2 {
                    return Err(QuantRS2Error::InvalidOperation(
                        "CNOT gate requires exactly two qubits".to_string(),
                    ));
                }

                let control = qubits[0];
                let target = qubits[1];

                // Implement CNOT as a sequence of pulses
                // This is a simplified version - real implementation would use
                // cross-resonance or other two-qubit gate protocols
                let default_control_params = QubitControlParams::default();
                let default_target_params = QubitControlParams::default();
                let control_params = self
                    .calibration
                    .qubit_params
                    .get(&control)
                    .unwrap_or(&default_control_params);
                let target_params = self
                    .calibration
                    .qubit_params
                    .get(&target)
                    .unwrap_or(&default_target_params);

                // Cross-resonance pulse on control qubit at target frequency
                let cr_pulse = Pulse::new(
                    50.0, // 50 ns CR pulse
                    0.3 * control_params.pi_pulse_amplitude,
                    target_params.drive_frequency,
                    0.0,
                    PulseEnvelope::Square,
                    2.0,
                );

                sequence.add_pulse(0.0, control, cr_pulse);

                // Echo pulse on target to cancel unwanted rotations
                let echo_pulse = Pulse::new(
                    25.0,
                    target_params.pi_pulse_amplitude,
                    target_params.drive_frequency,
                    PI, // π phase for echo
                    PulseEnvelope::DRAG {
                        sigma: 25.0 / 6.0,
                        beta: target_params.drag_parameter,
                    },
                    2.0,
                );

                sequence.add_pulse(25.0, target, echo_pulse);
            }
            _ => {
                return Err(QuantRS2Error::InvalidOperation(format!(
                    "Gate '{}' not supported in pulse compiler",
                    gate_name
                )));
            }
        }

        sequence.check_overlaps()?;
        Ok(sequence)
    }

    /// Optimize pulse sequence for hardware constraints
    pub fn optimize_sequence(&self, sequence: &mut PulseSequence) -> QuantRS2Result<()> {
        // Apply timing constraints
        let constraints = &self.calibration.timing_constraints;

        // Round all timings to clock resolution
        for (start_time, _, _pulse) in &mut sequence.pulses {
            *start_time =
                (*start_time / constraints.clock_resolution).round() * constraints.clock_resolution;
        }

        // Ensure minimum pulse separation
        sequence
            .pulses
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for i in 1..sequence.pulses.len() {
            let prev_end = sequence.pulses[i - 1].0 + sequence.pulses[i - 1].2.duration;
            let curr_start = sequence.pulses[i].0;

            if curr_start - prev_end < constraints.min_pulse_separation {
                sequence.pulses[i].0 = prev_end + constraints.min_pulse_separation;
            }
        }

        // Update total duration
        if let Some((start_time, _, pulse)) = sequence.pulses.last() {
            sequence.duration = start_time + pulse.duration;
        }

        Ok(())
    }

    /// Add calibration data for a qubit
    pub fn add_qubit_calibration(&mut self, qubit: QubitId, params: QubitControlParams) {
        self.calibration.qubit_params.insert(qubit, params);
    }

    /// Add coupling calibration between qubits
    pub fn add_coupling_calibration(&mut self, q1: QubitId, q2: QubitId, params: CouplingParams) {
        self.calibration
            .coupling_params
            .insert((q1, q2), params.clone());
        self.calibration.coupling_params.insert((q2, q1), params); // Symmetric
    }

    /// Generate composite pulse for arbitrary rotation
    pub fn generate_arbitrary_rotation(
        &self,
        qubit: QubitId,
        theta: f64,
        phi: f64,
    ) -> QuantRS2Result<PulseSequence> {
        let default_params = QubitControlParams::default();
        let params = self
            .calibration
            .qubit_params
            .get(&qubit)
            .unwrap_or(&default_params);

        let amplitude = (theta / PI) * params.pi_pulse_amplitude;

        let pulse = Pulse::new(
            params.gate_time,
            amplitude,
            params.drive_frequency,
            phi,
            PulseEnvelope::DRAG {
                sigma: params.gate_time / 6.0,
                beta: params.drag_parameter,
            },
            2.0,
        );

        let mut sequence = PulseSequence::new(format!("R({:.3}, {:.3})", theta, phi));
        sequence.add_pulse(0.0, qubit, pulse);

        Ok(sequence)
    }
}

/// Pulse-level noise model for superconducting qubits
#[derive(Debug, Clone)]
pub struct PulseNoiseModel {
    /// Amplitude noise standard deviation
    pub amplitude_noise: f64,
    /// Phase noise standard deviation (radians)
    pub phase_noise: f64,
    /// Timing jitter standard deviation (ns)
    pub timing_jitter: f64,
    /// Flux noise for frequency fluctuations (MHz)
    pub flux_noise: f64,
}

impl Default for PulseNoiseModel {
    fn default() -> Self {
        Self {
            amplitude_noise: 0.01, // 1% amplitude noise
            phase_noise: 0.01,     // 10 mrad phase noise
            timing_jitter: 0.1,    // 100 ps timing jitter
            flux_noise: 0.1,       // 100 kHz flux noise
        }
    }
}

impl PulseNoiseModel {
    /// Apply noise to a pulse
    pub fn apply_noise(
        &self,
        pulse: &mut Pulse,
        rng: &mut dyn scirs2_core::random::RngCore,
    ) -> QuantRS2Result<()> {
        use scirs2_core::random::prelude::*;

        // Apply amplitude noise
        let amplitude_factor = 1.0 + rng.gen_range(0.0..1.0) * self.amplitude_noise;
        pulse.amplitude *= amplitude_factor;

        // Apply phase noise
        let phase_shift = rng.gen_range(0.0..1.0) * self.phase_noise;
        pulse.phase += phase_shift;

        // Apply frequency noise
        let freq_shift = rng.gen_range(0.0..1.0) * self.flux_noise / 1000.0; // Convert MHz to GHz
        pulse.frequency += freq_shift;

        Ok(())
    }

    /// Apply timing jitter to a pulse sequence
    pub fn apply_timing_jitter(
        &self,
        sequence: &mut PulseSequence,
        rng: &mut dyn scirs2_core::random::RngCore,
    ) -> QuantRS2Result<()> {
        use scirs2_core::random::prelude::*;

        for (start_time, _, _) in &mut sequence.pulses {
            let jitter = rng.gen_range(0.0..1.0) * self.timing_jitter;
            *start_time += jitter;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{multi::CNOT, single::PauliX};

    #[test]
    fn test_pulse_envelope_gaussian() {
        let envelope = PulseEnvelope::Gaussian { sigma: 0.1 };

        // Test at center
        let center_value = envelope.evaluate(0.5);
        assert!((center_value - 1.0).abs() < 1e-10);

        // Test symmetry
        let left_value = envelope.evaluate(0.3);
        let right_value = envelope.evaluate(0.7);
        assert!((left_value - right_value).abs() < 1e-10);
    }

    #[test]
    fn test_pulse_waveform_generation() {
        let pulse = Pulse::new(
            10.0, // 10 ns
            0.5,  // 50% amplitude
            5.0,  // 5 GHz
            0.0,  // 0 phase
            PulseEnvelope::Gaussian { sigma: 0.1 },
            2.0, // 2 GSa/s
        );

        let waveform = pulse.generate_waveform().unwrap();
        assert_eq!(waveform.len(), 20); // 10 ns * 2 GSa/s = 20 samples

        // Check that waveform is not all zeros
        let max_amplitude = waveform.iter().map(|x| x.norm()).fold(0.0, f64::max);
        assert!(max_amplitude > 0.0);
    }

    #[test]
    fn test_drag_pulse_generation() {
        let pulse = Pulse::new(
            20.0,
            1.0,
            5.0,
            0.0,
            PulseEnvelope::DRAG {
                sigma: 0.1,
                beta: 0.5,
            },
            2.0,
        );

        let (i_comp, q_comp) = pulse.generate_drag_waveform().unwrap();
        assert_eq!(i_comp.len(), 40);
        assert_eq!(q_comp.len(), 40);

        // Both components should have non-zero values
        let i_max = i_comp.iter().map(|x| x.norm()).fold(0.0, f64::max);
        let q_max = q_comp.iter().map(|x| x.norm()).fold(0.0, f64::max);
        assert!(i_max > 0.0);
        assert!(q_max > 0.0);
    }

    #[test]
    fn test_pulse_compiler_single_qubit() {
        let calibration = HardwareCalibration::default();
        let compiler = PulseCompiler::new(calibration);

        let gate = PauliX { target: QubitId(0) };
        let qubits = vec![QubitId(0)];

        let sequence = compiler.compile_gate(&gate, &qubits).unwrap();
        assert_eq!(sequence.pulses.len(), 1);
        assert_eq!(sequence.pulses[0].1, QubitId(0));
    }

    #[test]
    fn test_pulse_compiler_cnot() {
        let calibration = HardwareCalibration::default();
        let compiler = PulseCompiler::new(calibration);

        let gate = CNOT {
            control: QubitId(0),
            target: QubitId(1),
        };
        let qubits = vec![QubitId(0), QubitId(1)];

        let sequence = compiler.compile_gate(&gate, &qubits).unwrap();
        assert!(sequence.pulses.len() >= 2); // Should have multiple pulses for CNOT
    }

    #[test]
    fn test_pulse_sequence_overlap_detection() {
        let mut sequence = PulseSequence::new("test".to_string());

        let pulse1 = Pulse::new(10.0, 0.5, 5.0, 0.0, PulseEnvelope::Square, 2.0);
        let pulse2 = Pulse::new(10.0, 0.5, 5.0, 0.0, PulseEnvelope::Square, 2.0);

        sequence.add_pulse(0.0, QubitId(0), pulse1);
        sequence.add_pulse(5.0, QubitId(0), pulse2); // Overlaps with first pulse

        assert!(sequence.check_overlaps().is_err());
    }

    #[test]
    fn test_pulse_sequence_no_overlap() {
        let mut sequence = PulseSequence::new("test".to_string());

        let pulse1 = Pulse::new(10.0, 0.5, 5.0, 0.0, PulseEnvelope::Square, 2.0);
        let pulse2 = Pulse::new(10.0, 0.5, 5.0, 0.0, PulseEnvelope::Square, 2.0);

        sequence.add_pulse(0.0, QubitId(0), pulse1);
        sequence.add_pulse(15.0, QubitId(0), pulse2); // No overlap

        assert!(sequence.check_overlaps().is_ok());
    }

    #[test]
    fn test_arbitrary_rotation_compilation() {
        let calibration = HardwareCalibration::default();
        let compiler = PulseCompiler::new(calibration);

        let theta = PI / 4.0; // 45 degrees
        let phi = PI / 2.0; // 90 degrees

        let sequence = compiler
            .generate_arbitrary_rotation(QubitId(0), theta, phi)
            .unwrap();
        assert_eq!(sequence.pulses.len(), 1);

        let (_, _, pulse) = &sequence.pulses[0];
        assert!((pulse.phase - phi).abs() < 1e-10);
    }

    #[test]
    fn test_pulse_optimization() {
        let mut calibration = HardwareCalibration::default();
        calibration.timing_constraints.clock_resolution = 1.0; // 1 ns resolution
        calibration.timing_constraints.min_pulse_separation = 5.0; // 5 ns minimum

        let compiler = PulseCompiler::new(calibration);

        let mut sequence = PulseSequence::new("test".to_string());
        let pulse = Pulse::new(10.0, 0.5, 5.0, 0.0, PulseEnvelope::Square, 2.0);

        sequence.add_pulse(2.7, QubitId(0), pulse); // Non-aligned timing

        compiler.optimize_sequence(&mut sequence).unwrap();

        // Should be rounded to clock resolution
        assert!((sequence.pulses[0].0 - 3.0).abs() < 1e-10);
    }
}
