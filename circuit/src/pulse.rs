//! Pulse-level control for quantum circuits
//!
//! This module provides low-level pulse control capabilities for quantum operations,
//! allowing fine-grained optimization and hardware-specific calibration.

use crate::builder::Circuit;
// SciRS2 POLICY compliant - using scirs2_core::Complex64
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Complex amplitude type
type C64 = Complex64;

/// Time in nanoseconds
type Time = f64;

/// Frequency in GHz
type Frequency = f64;

/// Pulse waveform representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Waveform {
    /// Sample points
    pub samples: Vec<C64>,
    /// Sample rate in GS/s (gigasamples per second)
    pub sample_rate: f64,
    /// Total duration in nanoseconds
    pub duration: Time,
}

impl Waveform {
    /// Create a new waveform
    #[must_use]
    pub fn new(samples: Vec<C64>, sample_rate: f64) -> Self {
        let duration = (samples.len() as f64) / sample_rate;
        Self {
            samples,
            sample_rate,
            duration,
        }
    }

    /// Create a Gaussian pulse
    #[must_use]
    pub fn gaussian(amplitude: f64, sigma: f64, duration: Time, sample_rate: f64) -> Self {
        let n_samples = (duration * sample_rate) as usize;
        let center = duration / 2.0;
        let mut samples = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let t = (i as f64) / sample_rate;
            let envelope = amplitude * (-0.5 * ((t - center) / sigma).powi(2)).exp();
            samples.push(C64::new(envelope, 0.0));
        }

        Self::new(samples, sample_rate)
    }

    /// Create a DRAG (Derivative Removal by Adiabatic Gate) pulse
    #[must_use]
    pub fn drag(amplitude: f64, sigma: f64, beta: f64, duration: Time, sample_rate: f64) -> Self {
        let n_samples = (duration * sample_rate) as usize;
        let center = duration / 2.0;
        let mut samples = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let t = (i as f64) / sample_rate;
            let gaussian = amplitude * (-0.5 * ((t - center) / sigma).powi(2)).exp();
            let derivative = -((t - center) / sigma.powi(2)) * gaussian;
            let real_part = gaussian;
            let imag_part = beta * derivative;
            samples.push(C64::new(real_part, imag_part));
        }

        Self::new(samples, sample_rate)
    }

    /// Create a square pulse
    #[must_use]
    pub fn square(amplitude: f64, duration: Time, sample_rate: f64) -> Self {
        let n_samples = (duration * sample_rate) as usize;
        let samples = vec![C64::new(amplitude, 0.0); n_samples];
        Self::new(samples, sample_rate)
    }

    /// Apply frequency modulation
    pub fn modulate(&mut self, frequency: Frequency, phase: f64) {
        for (i, sample) in self.samples.iter_mut().enumerate() {
            let t = (i as f64) / self.sample_rate;
            let rotation = C64::from_polar(1.0, (2.0 * PI * frequency).mul_add(t, phase));
            *sample *= rotation;
        }
    }

    /// Scale amplitude
    pub fn scale(&mut self, factor: f64) {
        for sample in &mut self.samples {
            *sample *= factor;
        }
    }

    /// Get maximum amplitude
    pub fn max_amplitude(&self) -> f64 {
        self.samples.iter().map(|s| s.norm()).fold(0.0, f64::max)
    }
}

/// Pulse channel types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Channel {
    /// Drive channel for qubit control
    Drive(usize),
    /// Measurement channel
    Measure(usize),
    /// Control channel for two-qubit gates
    Control(usize, usize),
    /// Auxiliary channel
    Aux(String),
}

/// Pulse instruction
#[derive(Debug, Clone)]
pub enum PulseInstruction {
    /// Play a waveform on a channel
    Play {
        waveform: Waveform,
        channel: Channel,
        phase: f64,
    },
    /// Set channel frequency
    SetFrequency {
        channel: Channel,
        frequency: Frequency,
    },
    /// Set channel phase
    SetPhase { channel: Channel, phase: f64 },
    /// Shift channel phase
    ShiftPhase { channel: Channel, phase: f64 },
    /// Delay/wait
    Delay {
        duration: Time,
        channels: Vec<Channel>,
    },
    /// Barrier synchronization
    Barrier { channels: Vec<Channel> },
    /// Acquire measurement
    Acquire {
        duration: Time,
        channel: Channel,
        memory_slot: usize,
    },
}

/// Pulse schedule representing a quantum operation
#[derive(Debug, Clone)]
pub struct PulseSchedule {
    /// Instructions in the schedule
    pub instructions: Vec<(Time, PulseInstruction)>,
    /// Total duration
    pub duration: Time,
    /// Channels used
    pub channels: Vec<Channel>,
}

impl Default for PulseSchedule {
    fn default() -> Self {
        Self::new()
    }
}

impl PulseSchedule {
    /// Create a new empty schedule
    #[must_use]
    pub const fn new() -> Self {
        Self {
            instructions: Vec::new(),
            duration: 0.0,
            channels: Vec::new(),
        }
    }

    /// Add an instruction at a specific time
    pub fn add_instruction(&mut self, time: Time, instruction: PulseInstruction) {
        // Update duration
        let inst_duration = match &instruction {
            PulseInstruction::Play { waveform, .. } => waveform.duration,
            PulseInstruction::Delay { duration, .. } => *duration,
            PulseInstruction::Acquire { duration, .. } => *duration,
            _ => 0.0,
        };
        self.duration = self.duration.max(time + inst_duration);

        // Track channels
        match &instruction {
            PulseInstruction::Play { channel, .. }
            | PulseInstruction::SetFrequency { channel, .. }
            | PulseInstruction::SetPhase { channel, .. }
            | PulseInstruction::ShiftPhase { channel, .. } => {
                if !self.channels.contains(channel) {
                    self.channels.push(channel.clone());
                }
            }
            PulseInstruction::Delay { channels, .. } | PulseInstruction::Barrier { channels } => {
                for channel in channels {
                    if !self.channels.contains(channel) {
                        self.channels.push(channel.clone());
                    }
                }
            }
            PulseInstruction::Acquire { channel, .. } => {
                if !self.channels.contains(channel) {
                    self.channels.push(channel.clone());
                }
            }
        }

        self.instructions.push((time, instruction));
    }

    /// Merge with another schedule
    pub fn append(&mut self, other: &Self, time_offset: Time) {
        for (time, instruction) in &other.instructions {
            self.add_instruction(time + time_offset, instruction.clone());
        }
    }

    /// Align schedules on multiple channels
    pub fn align_parallel(&mut self, schedules: Vec<Self>) {
        let start_time = self.duration;
        let mut max_duration: f64 = 0.0;

        for schedule in schedules {
            self.append(&schedule, start_time);
            max_duration = max_duration.max(schedule.duration);
        }

        self.duration = start_time + max_duration;
    }
}

/// Pulse calibration for a specific gate
#[derive(Debug, Clone)]
pub struct PulseCalibration {
    /// Gate name
    pub gate_name: String,
    /// Qubits this calibration applies to
    pub qubits: Vec<QubitId>,
    /// Parameters (e.g., rotation angle)
    pub parameters: HashMap<String, f64>,
    /// The pulse schedule
    pub schedule: PulseSchedule,
}

/// Pulse-level compiler
pub struct PulseCompiler {
    /// Calibrations for gates
    calibrations: HashMap<String, Vec<PulseCalibration>>,
    /// Device configuration
    device_config: DeviceConfig,
}

/// Device configuration for pulse control
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Qubit frequencies (GHz)
    pub qubit_frequencies: HashMap<usize, Frequency>,
    /// Coupling strengths (MHz)
    pub coupling_strengths: HashMap<(usize, usize), f64>,
    /// Drive amplitudes
    pub drive_amplitudes: HashMap<usize, f64>,
    /// Measurement frequencies
    pub meas_frequencies: HashMap<usize, Frequency>,
    /// Sample rate (GS/s)
    pub sample_rate: f64,
}

impl DeviceConfig {
    /// Create a default configuration
    #[must_use]
    pub fn default_config(num_qubits: usize) -> Self {
        let mut config = Self {
            qubit_frequencies: HashMap::new(),
            coupling_strengths: HashMap::new(),
            drive_amplitudes: HashMap::new(),
            meas_frequencies: HashMap::new(),
            sample_rate: 1.0, // 1 GS/s
        };

        // Default frequencies around 5 GHz
        for i in 0..num_qubits {
            config
                .qubit_frequencies
                .insert(i, (i as f64).mul_add(0.1, 5.0));
            config
                .meas_frequencies
                .insert(i, (i as f64).mul_add(0.05, 6.5));
            config.drive_amplitudes.insert(i, 0.1);
        }

        // Default coupling for nearest neighbors
        for i in 0..(num_qubits - 1) {
            config.coupling_strengths.insert((i, i + 1), 0.01); // 10 MHz
        }

        config
    }
}

impl PulseCompiler {
    /// Create a new pulse compiler
    #[must_use]
    pub fn new(device_config: DeviceConfig) -> Self {
        let mut compiler = Self {
            calibrations: HashMap::new(),
            device_config,
        };

        // Add default calibrations
        compiler.add_default_calibrations();
        compiler
    }

    /// Add default gate calibrations
    fn add_default_calibrations(&mut self) {
        // Single-qubit gates
        self.add_single_qubit_calibrations();

        // Two-qubit gates
        self.add_two_qubit_calibrations();
    }

    /// Add single-qubit gate calibrations
    fn add_single_qubit_calibrations(&mut self) {
        let sample_rate = self.device_config.sample_rate;

        // X gate (pi pulse)
        let x_waveform = Waveform::gaussian(0.5, 10.0, 40.0, sample_rate);
        let mut x_schedule = PulseSchedule::new();
        x_schedule.add_instruction(
            0.0,
            PulseInstruction::Play {
                waveform: x_waveform,
                channel: Channel::Drive(0),
                phase: 0.0,
            },
        );

        self.calibrations.insert(
            "X".to_string(),
            vec![PulseCalibration {
                gate_name: "X".to_string(),
                qubits: vec![QubitId(0)],
                parameters: HashMap::new(),
                schedule: x_schedule,
            }],
        );

        // Y gate (pi pulse with phase)
        let y_waveform = Waveform::gaussian(0.5, 10.0, 40.0, sample_rate);
        let mut y_schedule = PulseSchedule::new();
        y_schedule.add_instruction(
            0.0,
            PulseInstruction::Play {
                waveform: y_waveform,
                channel: Channel::Drive(0),
                phase: PI / 2.0,
            },
        );

        self.calibrations.insert(
            "Y".to_string(),
            vec![PulseCalibration {
                gate_name: "Y".to_string(),
                qubits: vec![QubitId(0)],
                parameters: HashMap::new(),
                schedule: y_schedule,
            }],
        );
    }

    /// Add two-qubit gate calibrations
    fn add_two_qubit_calibrations(&mut self) {
        let sample_rate = self.device_config.sample_rate;

        // CNOT using cross-resonance
        let cr_waveform = Waveform::square(0.1, 200.0, sample_rate);
        let mut cnot_schedule = PulseSchedule::new();

        // Cross-resonance pulse
        cnot_schedule.add_instruction(
            0.0,
            PulseInstruction::Play {
                waveform: cr_waveform,
                channel: Channel::Control(0, 1),
                phase: 0.0,
            },
        );

        // Echo pulses
        let echo_waveform = Waveform::gaussian(0.25, 10.0, 40.0, sample_rate);
        cnot_schedule.add_instruction(
            100.0,
            PulseInstruction::Play {
                waveform: echo_waveform,
                channel: Channel::Drive(1),
                phase: PI,
            },
        );

        self.calibrations.insert(
            "CNOT".to_string(),
            vec![PulseCalibration {
                gate_name: "CNOT".to_string(),
                qubits: vec![QubitId(0), QubitId(1)],
                parameters: HashMap::new(),
                schedule: cnot_schedule,
            }],
        );
    }

    /// Compile a circuit to pulse schedule
    pub fn compile<const N: usize>(&self, circuit: &Circuit<N>) -> QuantRS2Result<PulseSchedule> {
        let mut schedule = PulseSchedule::new();
        let mut time = 0.0;

        for gate in circuit.gates() {
            let gate_schedule = self.compile_gate(gate.as_ref())?;
            schedule.append(&gate_schedule, time);
            time += gate_schedule.duration;
        }

        Ok(schedule)
    }

    /// Compile a single gate to pulse schedule
    fn compile_gate(&self, gate: &dyn GateOp) -> QuantRS2Result<PulseSchedule> {
        let gate_name = gate.name();
        let qubits = gate.qubits();

        // Look for matching calibration
        if let Some(calibrations) = self.calibrations.get(gate_name) {
            for calib in calibrations {
                if calib.qubits == qubits {
                    return Ok(self.instantiate_calibration(calib, qubits));
                }
            }
        }

        // No calibration found - use default
        self.default_gate_schedule(gate)
    }

    /// Instantiate a calibration for specific qubits
    fn instantiate_calibration(
        &self,
        calibration: &PulseCalibration,
        qubits: Vec<QubitId>,
    ) -> PulseSchedule {
        let mut schedule = calibration.schedule.clone();

        // Remap channels for actual qubits
        let mut remapped_instructions = Vec::new();
        for (time, instruction) in schedule.instructions {
            let remapped = match instruction {
                PulseInstruction::Play {
                    waveform,
                    channel,
                    phase,
                } => {
                    let new_channel = self.remap_channel(&channel, &qubits);
                    PulseInstruction::Play {
                        waveform,
                        channel: new_channel,
                        phase,
                    }
                }
                PulseInstruction::SetFrequency { channel, frequency } => {
                    let new_channel = self.remap_channel(&channel, &qubits);
                    PulseInstruction::SetFrequency {
                        channel: new_channel,
                        frequency,
                    }
                }
                // ... handle other instruction types
                _ => instruction,
            };
            remapped_instructions.push((time, remapped));
        }

        schedule.instructions = remapped_instructions;
        schedule
    }

    /// Remap channel for specific qubits
    fn remap_channel(&self, channel: &Channel, qubits: &[QubitId]) -> Channel {
        match channel {
            Channel::Drive(0) => Channel::Drive(qubits[0].id() as usize),
            Channel::Drive(1) if qubits.len() > 1 => Channel::Drive(qubits[1].id() as usize),
            Channel::Control(0, 1) if qubits.len() > 1 => {
                Channel::Control(qubits[0].id() as usize, qubits[1].id() as usize)
            }
            _ => channel.clone(),
        }
    }

    /// Generate default pulse schedule for a gate
    fn default_gate_schedule(&self, gate: &dyn GateOp) -> QuantRS2Result<PulseSchedule> {
        let mut schedule = PulseSchedule::new();
        let qubits = gate.qubits();

        // Simple default: Gaussian pulse
        let waveform = Waveform::gaussian(0.5, 10.0, 40.0, self.device_config.sample_rate);

        for qubit in qubits {
            schedule.add_instruction(
                0.0,
                PulseInstruction::Play {
                    waveform: waveform.clone(),
                    channel: Channel::Drive(qubit.id() as usize),
                    phase: 0.0,
                },
            );
        }

        Ok(schedule)
    }
}

/// Pulse optimization
pub struct PulseOptimizer {
    /// Target fidelity
    target_fidelity: f64,
    /// Maximum iterations
    max_iterations: usize,
}

impl Default for PulseOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PulseOptimizer {
    /// Create a new optimizer
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target_fidelity: 0.999,
            max_iterations: 100,
        }
    }

    /// Optimize a pulse schedule
    pub const fn optimize(&self, schedule: &mut PulseSchedule) -> QuantRS2Result<()> {
        // Placeholder for pulse optimization
        // Would implement gradient-based optimization, GRAPE, etc.
        Ok(())
    }

    /// Apply DRAG correction
    pub fn apply_drag_correction(&self, waveform: &mut Waveform, beta: f64) -> QuantRS2Result<()> {
        // Add derivative component for DRAG
        let mut derivative = vec![C64::new(0.0, 0.0); waveform.samples.len()];

        for i in 1..(waveform.samples.len() - 1) {
            let dt = 1.0 / waveform.sample_rate;
            derivative[i] = (waveform.samples[i + 1] - waveform.samples[i - 1]) / (2.0 * dt);
        }

        for (sample, deriv) in waveform.samples.iter_mut().zip(derivative.iter()) {
            *sample += C64::new(0.0, beta * deriv.re);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantrs2_core::gate::single::Hadamard;

    #[test]
    fn test_waveform_creation() {
        let gaussian = Waveform::gaussian(1.0, 10.0, 40.0, 1.0);
        assert_eq!(gaussian.samples.len(), 40);
        assert!(gaussian.max_amplitude() <= 1.0);

        let square = Waveform::square(0.5, 20.0, 1.0);
        assert_eq!(square.samples.len(), 20);
        assert_eq!(square.max_amplitude(), 0.5);
    }

    #[test]
    fn test_drag_pulse() {
        let drag = Waveform::drag(1.0, 10.0, 0.1, 40.0, 1.0);
        assert_eq!(drag.samples.len(), 40);

        // Check that DRAG has imaginary component
        let has_imag = drag.samples.iter().any(|s| s.im.abs() > 1e-10);
        assert!(has_imag);
    }

    #[test]
    fn test_pulse_schedule() {
        let mut schedule = PulseSchedule::new();
        let waveform = Waveform::gaussian(0.5, 10.0, 40.0, 1.0);

        schedule.add_instruction(
            0.0,
            PulseInstruction::Play {
                waveform: waveform.clone(),
                channel: Channel::Drive(0),
                phase: 0.0,
            },
        );

        schedule.add_instruction(
            50.0,
            PulseInstruction::Play {
                waveform,
                channel: Channel::Drive(1),
                phase: PI / 2.0,
            },
        );

        assert_eq!(schedule.instructions.len(), 2);
        assert_eq!(schedule.duration, 90.0); // 50 + 40
    }

    #[test]
    fn test_pulse_compiler() {
        let device_config = DeviceConfig::default_config(2);
        let compiler = PulseCompiler::new(device_config);

        let mut circuit = Circuit::<2>::new();
        circuit
            .add_gate(Hadamard { target: QubitId(0) })
            .expect("add H gate to circuit");

        let schedule = compiler
            .compile(&circuit)
            .expect("pulse compilation should succeed");
        assert!(schedule.duration > 0.0);
    }
}
