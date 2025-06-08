//! Pulse-level control interfaces for quantum hardware providers.
//!
//! This module provides low-level pulse control for quantum operations,
//! enabling fine-grained control over quantum gates and measurements.

use std::collections::HashMap;
use std::f64::consts::PI;
use num_complex::Complex64;
use crate::{DeviceError, DeviceResult};

/// Pulse shape types
#[derive(Debug, Clone, PartialEq)]
pub enum PulseShape {
    /// Gaussian pulse
    Gaussian {
        duration: f64,
        sigma: f64,
        amplitude: Complex64,
    },
    /// Gaussian with derivative removal (DRAG)
    GaussianDrag {
        duration: f64,
        sigma: f64,
        amplitude: Complex64,
        beta: f64,
    },
    /// Square/constant pulse
    Square {
        duration: f64,
        amplitude: Complex64,
    },
    /// Cosine-tapered pulse
    CosineTapered {
        duration: f64,
        amplitude: Complex64,
        rise_time: f64,
    },
    /// Arbitrary waveform
    Arbitrary {
        samples: Vec<Complex64>,
        sample_rate: f64,
    },
}

/// Channel types for pulse control
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChannelType {
    /// Drive channel for qubit control
    Drive(u32),
    /// Measurement channel
    Measure(u32),
    /// Control channel for two-qubit gates
    Control(u32, u32),
    /// Readout channel
    Readout(u32),
    /// Acquire channel for measurement
    Acquire(u32),
}

/// Pulse instruction
#[derive(Debug, Clone)]
pub struct PulseInstruction {
    /// Time when pulse starts (in dt units)
    pub t0: u64,
    /// Channel to play pulse on
    pub channel: ChannelType,
    /// Pulse shape and parameters
    pub pulse: PulseShape,
    /// Optional phase shift
    pub phase: Option<f64>,
    /// Optional frequency shift
    pub frequency: Option<f64>,
}

/// Pulse schedule (collection of instructions)
#[derive(Debug, Clone)]
pub struct PulseSchedule {
    /// Name of the schedule
    pub name: String,
    /// Duration in dt units
    pub duration: u64,
    /// List of pulse instructions
    pub instructions: Vec<PulseInstruction>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Calibration data for pulse operations
#[derive(Debug, Clone)]
pub struct PulseCalibration {
    /// Default pulse parameters for single-qubit gates
    pub single_qubit_defaults: HashMap<String, PulseShape>,
    /// Default pulse parameters for two-qubit gates
    pub two_qubit_defaults: HashMap<String, PulseShape>,
    /// Qubit frequencies (GHz)
    pub qubit_frequencies: Vec<f64>,
    /// Measurement frequencies (GHz)
    pub meas_frequencies: Vec<f64>,
    /// Drive power calibration
    pub drive_powers: Vec<f64>,
    /// Sample time (dt) in seconds
    pub dt: f64,
}

/// Pulse builder for creating schedules
pub struct PulseBuilder {
    schedule: PulseSchedule,
    current_time: u64,
    calibration: Option<PulseCalibration>,
}

impl PulseBuilder {
    /// Create a new pulse builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            schedule: PulseSchedule {
                name: name.into(),
                duration: 0,
                instructions: Vec::new(),
                metadata: HashMap::new(),
            },
            current_time: 0,
            calibration: None,
        }
    }

    /// Create with calibration data
    pub fn with_calibration(name: impl Into<String>, calibration: PulseCalibration) -> Self {
        let mut builder = Self::new(name);
        builder.calibration = Some(calibration);
        builder
    }

    /// Add a pulse instruction
    pub fn play(mut self, channel: ChannelType, pulse: PulseShape) -> Self {
        let duration = match &pulse {
            PulseShape::Gaussian { duration, .. } => *duration,
            PulseShape::GaussianDrag { duration, .. } => *duration,
            PulseShape::Square { duration, .. } => *duration,
            PulseShape::CosineTapered { duration, .. } => *duration,
            PulseShape::Arbitrary { samples, sample_rate } => {
                samples.len() as f64 / sample_rate
            }
        };

        let duration_dt = if let Some(cal) = &self.calibration {
            (duration / cal.dt) as u64
        } else {
            duration as u64
        };

        self.schedule.instructions.push(PulseInstruction {
            t0: self.current_time,
            channel,
            pulse,
            phase: None,
            frequency: None,
        });

        self.current_time += duration_dt;
        self.schedule.duration = self.schedule.duration.max(self.current_time);
        self
    }

    /// Add a delay
    pub fn delay(mut self, duration: u64, channel: ChannelType) -> Self {
        // Delays are implicit - just advance time
        self.current_time += duration;
        self.schedule.duration = self.schedule.duration.max(self.current_time);
        self
    }

    /// Set phase on a channel
    pub fn set_phase(mut self, channel: ChannelType, phase: f64) -> Self {
        self.schedule.instructions.push(PulseInstruction {
            t0: self.current_time,
            channel,
            pulse: PulseShape::Square {
                duration: 0.0,
                amplitude: Complex64::new(0.0, 0.0),
            },
            phase: Some(phase),
            frequency: None,
        });
        self
    }

    /// Set frequency on a channel
    pub fn set_frequency(mut self, channel: ChannelType, frequency: f64) -> Self {
        self.schedule.instructions.push(PulseInstruction {
            t0: self.current_time,
            channel,
            pulse: PulseShape::Square {
                duration: 0.0,
                amplitude: Complex64::new(0.0, 0.0),
            },
            phase: None,
            frequency: Some(frequency),
        });
        self
    }

    /// Barrier - synchronize channels
    pub fn barrier(mut self, channels: Vec<ChannelType>) -> Self {
        // Find latest time across channels
        let max_time = self.schedule.instructions
            .iter()
            .filter(|inst| channels.contains(&inst.channel))
            .map(|inst| {
                let duration = match &inst.pulse {
                    PulseShape::Gaussian { duration, .. } => *duration,
                    PulseShape::GaussianDrag { duration, .. } => *duration,
                    PulseShape::Square { duration, .. } => *duration,
                    PulseShape::CosineTapered { duration, .. } => *duration,
                    PulseShape::Arbitrary { samples, sample_rate } => {
                        samples.len() as f64 / sample_rate
                    }
                };
                let duration_dt = if let Some(cal) = &self.calibration {
                    (duration / cal.dt) as u64
                } else {
                    duration as u64
                };
                inst.t0 + duration_dt
            })
            .max()
            .unwrap_or(self.current_time);

        self.current_time = max_time;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.schedule.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the pulse schedule
    pub fn build(self) -> PulseSchedule {
        self.schedule
    }
}

/// Provider-specific pulse backend
pub trait PulseBackend {
    /// Execute a pulse schedule
    fn execute_pulse_schedule(
        &self,
        schedule: &PulseSchedule,
        shots: usize,
        meas_level: MeasLevel,
    ) -> DeviceResult<PulseResult>;

    /// Get pulse calibration data
    fn get_calibration(&self) -> DeviceResult<PulseCalibration>;

    /// Validate a pulse schedule
    fn validate_schedule(&self, schedule: &PulseSchedule) -> DeviceResult<()>;
}

/// Measurement level for pulse experiments
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasLevel {
    /// Raw ADC values
    Raw,
    /// IQ values after demodulation
    Kerneled,
    /// Discriminated qubit states
    Classified,
}

/// Result from pulse execution
#[derive(Debug, Clone)]
pub struct PulseResult {
    /// Measurement data
    pub measurements: Vec<MeasurementData>,
    /// Execution metadata
    pub metadata: HashMap<String, String>,
}

/// Measurement data from pulse execution
#[derive(Debug, Clone)]
pub enum MeasurementData {
    /// Raw ADC samples
    Raw(Vec<Vec<Complex64>>),
    /// IQ values
    IQ(Vec<Vec<Complex64>>),
    /// Classified states
    States(Vec<Vec<u8>>),
}

/// IBM Pulse backend implementation
#[cfg(feature = "ibm")]
pub struct IBMPulseBackend {
    backend_name: String,
    calibration: PulseCalibration,
}

#[cfg(feature = "ibm")]
impl IBMPulseBackend {
    pub fn new(backend_name: String) -> Self {
        // Default calibration - would be fetched from backend
        let calibration = PulseCalibration {
            single_qubit_defaults: HashMap::new(),
            two_qubit_defaults: HashMap::new(),
            qubit_frequencies: vec![5.0; 5], // GHz
            meas_frequencies: vec![6.5; 5],  // GHz
            drive_powers: vec![0.1; 5],
            dt: 2.2222e-10, // ~0.22 ns
        };

        Self {
            backend_name,
            calibration,
        }
    }
}

#[cfg(feature = "ibm")]
impl PulseBackend for IBMPulseBackend {
    fn execute_pulse_schedule(
        &self,
        schedule: &PulseSchedule,
        shots: usize,
        meas_level: MeasLevel,
    ) -> DeviceResult<PulseResult> {
        // Validate first
        self.validate_schedule(schedule)?;

        // Convert to Qiskit pulse format and execute
        // This is a placeholder - actual implementation would use Qiskit
        Ok(PulseResult {
            measurements: vec![],
            metadata: HashMap::new(),
        })
    }

    fn get_calibration(&self) -> DeviceResult<PulseCalibration> {
        Ok(self.calibration.clone())
    }

    fn validate_schedule(&self, schedule: &PulseSchedule) -> DeviceResult<()> {
        // Check duration limits
        if schedule.duration > 1_000_000 {
            return Err(DeviceError::APIError("Schedule too long".to_string()));
        }

        // Check channel availability
        for inst in &schedule.instructions {
            match &inst.channel {
                ChannelType::Drive(q) | ChannelType::Measure(q) => {
                    if *q >= self.calibration.qubit_frequencies.len() as u32 {
                        return Err(DeviceError::APIError(format!("Invalid qubit: {}", q)));
                    }
                }
                ChannelType::Control(q1, q2) => {
                    if *q1 >= self.calibration.qubit_frequencies.len() as u32
                        || *q2 >= self.calibration.qubit_frequencies.len() as u32
                    {
                        return Err(DeviceError::APIError("Invalid control channel".to_string()));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

/// Standard pulse library
pub struct PulseLibrary;

impl PulseLibrary {
    /// Create a Gaussian pulse
    pub fn gaussian(duration: f64, sigma: f64, amplitude: f64) -> PulseShape {
        PulseShape::Gaussian {
            duration,
            sigma,
            amplitude: Complex64::new(amplitude, 0.0),
        }
    }

    /// Create a DRAG pulse
    pub fn drag(duration: f64, sigma: f64, amplitude: f64, beta: f64) -> PulseShape {
        PulseShape::GaussianDrag {
            duration,
            sigma,
            amplitude: Complex64::new(amplitude, 0.0),
            beta,
        }
    }

    /// Create a square pulse
    pub fn square(duration: f64, amplitude: f64) -> PulseShape {
        PulseShape::Square {
            duration,
            amplitude: Complex64::new(amplitude, 0.0),
        }
    }

    /// Create a cosine-tapered pulse
    pub fn cosine_tapered(duration: f64, amplitude: f64, rise_time: f64) -> PulseShape {
        PulseShape::CosineTapered {
            duration,
            amplitude: Complex64::new(amplitude, 0.0),
            rise_time,
        }
    }

    /// Create X gate pulse
    pub fn x_pulse(calibration: &PulseCalibration, qubit: u32) -> PulseShape {
        // Default X pulse - π rotation
        Self::gaussian(160e-9, 40e-9, 0.5)
    }

    /// Create Y gate pulse
    pub fn y_pulse(calibration: &PulseCalibration, qubit: u32) -> PulseShape {
        // Y pulse with phase
        PulseShape::Gaussian {
            duration: 160e-9,
            sigma: 40e-9,
            amplitude: Complex64::new(0.0, 0.5), // 90 degree phase
        }
    }

    /// Create SX (√X) gate pulse
    pub fn sx_pulse(calibration: &PulseCalibration, qubit: u32) -> PulseShape {
        // π/2 rotation
        Self::gaussian(160e-9, 40e-9, 0.25)
    }

    /// Create RZ gate using phase shift
    pub fn rz_pulse(angle: f64) -> PulseInstruction {
        PulseInstruction {
            t0: 0,
            channel: ChannelType::Drive(0), // Will be updated
            pulse: PulseShape::Square {
                duration: 0.0,
                amplitude: Complex64::new(0.0, 0.0),
            },
            phase: Some(angle),
            frequency: None,
        }
    }

    /// Create measurement pulse
    pub fn measure_pulse(calibration: &PulseCalibration, qubit: u32) -> PulseShape {
        // Square measurement pulse
        Self::square(2e-6, 0.1)
    }
}

/// Pulse schedule templates for common operations
pub struct PulseTemplates;

impl PulseTemplates {
    /// Create a calibrated X gate schedule
    pub fn x_gate(qubit: u32, calibration: &PulseCalibration) -> PulseSchedule {
        PulseBuilder::with_calibration("x_gate", calibration.clone())
            .play(
                ChannelType::Drive(qubit),
                PulseLibrary::x_pulse(calibration, qubit),
            )
            .build()
    }

    /// Create a calibrated CNOT gate schedule
    pub fn cnot_gate(control: u32, target: u32, calibration: &PulseCalibration) -> PulseSchedule {
        // Simplified CNOT using cross-resonance
        let cr_amp = 0.3;
        let cr_duration = 560e-9;

        PulseBuilder::with_calibration("cnot_gate", calibration.clone())
            // Pre-rotation on control
            .play(
                ChannelType::Drive(control),
                PulseLibrary::sx_pulse(calibration, control),
            )
            // Cross-resonance pulse
            .play(
                ChannelType::Control(control, target),
                PulseLibrary::gaussian(cr_duration, cr_duration / 4.0, cr_amp),
            )
            // Simultaneous rotations
            .play(
                ChannelType::Drive(control),
                PulseLibrary::x_pulse(calibration, control),
            )
            .play(
                ChannelType::Drive(target),
                PulseLibrary::x_pulse(calibration, target),
            )
            .barrier(vec![ChannelType::Drive(control), ChannelType::Drive(target)])
            .build()
    }

    /// Create a measurement schedule
    pub fn measure(qubits: Vec<u32>, calibration: &PulseCalibration) -> PulseSchedule {
        let mut builder = PulseBuilder::with_calibration("measure", calibration.clone());

        // Play measurement pulses simultaneously
        for &qubit in &qubits {
            builder = builder.play(
                ChannelType::Measure(qubit),
                PulseLibrary::measure_pulse(calibration, qubit),
            );
        }

        // Acquire data
        for &qubit in &qubits {
            builder = builder.play(
                ChannelType::Acquire(qubit),
                PulseShape::Square {
                    duration: 2e-6,
                    amplitude: Complex64::new(1.0, 0.0),
                },
            );
        }

        builder.build()
    }

    /// Create a Rabi oscillation experiment
    pub fn rabi_experiment(
        qubit: u32,
        amplitudes: Vec<f64>,
        calibration: &PulseCalibration,
    ) -> Vec<PulseSchedule> {
        amplitudes
            .into_iter()
            .map(|amp| {
                PulseBuilder::with_calibration(format!("rabi_{}", amp), calibration.clone())
                    .play(
                        ChannelType::Drive(qubit),
                        PulseLibrary::gaussian(160e-9, 40e-9, amp),
                    )
                    .play(
                        ChannelType::Measure(qubit),
                        PulseLibrary::measure_pulse(calibration, qubit),
                    )
                    .build()
            })
            .collect()
    }

    /// Create a T1 relaxation experiment
    pub fn t1_experiment(
        qubit: u32,
        delays: Vec<u64>,
        calibration: &PulseCalibration,
    ) -> Vec<PulseSchedule> {
        delays
            .into_iter()
            .map(|delay| {
                PulseBuilder::with_calibration(format!("t1_{}", delay), calibration.clone())
                    .play(
                        ChannelType::Drive(qubit),
                        PulseLibrary::x_pulse(calibration, qubit),
                    )
                    .delay(delay, ChannelType::Drive(qubit))
                    .play(
                        ChannelType::Measure(qubit),
                        PulseLibrary::measure_pulse(calibration, qubit),
                    )
                    .build()
            })
            .collect()
    }

    /// Create a Ramsey experiment (T2)
    pub fn ramsey_experiment(
        qubit: u32,
        delays: Vec<u64>,
        detuning: f64,
        calibration: &PulseCalibration,
    ) -> Vec<PulseSchedule> {
        delays
            .into_iter()
            .map(|delay| {
                PulseBuilder::with_calibration(format!("ramsey_{}", delay), calibration.clone())
                    // First π/2 pulse
                    .play(
                        ChannelType::Drive(qubit),
                        PulseLibrary::sx_pulse(calibration, qubit),
                    )
                    // Evolution with detuning
                    .set_frequency(ChannelType::Drive(qubit), detuning)
                    .delay(delay, ChannelType::Drive(qubit))
                    .set_frequency(ChannelType::Drive(qubit), 0.0)
                    // Second π/2 pulse
                    .play(
                        ChannelType::Drive(qubit),
                        PulseLibrary::sx_pulse(calibration, qubit),
                    )
                    // Measure
                    .play(
                        ChannelType::Measure(qubit),
                        PulseLibrary::measure_pulse(calibration, qubit),
                    )
                    .build()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pulse_builder() {
        let schedule = PulseBuilder::new("test")
            .play(
                ChannelType::Drive(0),
                PulseLibrary::gaussian(100e-9, 25e-9, 0.5),
            )
            .delay(50, ChannelType::Drive(0))
            .play(
                ChannelType::Drive(0),
                PulseLibrary::gaussian(100e-9, 25e-9, 0.5),
            )
            .build();

        assert_eq!(schedule.name, "test");
        assert_eq!(schedule.instructions.len(), 2);
    }

    #[test]
    fn test_pulse_shapes() {
        let gaussian = PulseLibrary::gaussian(100e-9, 25e-9, 0.5);
        match gaussian {
            PulseShape::Gaussian { duration, sigma, amplitude } => {
                assert_eq!(duration, 100e-9);
                assert_eq!(sigma, 25e-9);
                assert_eq!(amplitude.re, 0.5);
            }
            _ => panic!("Wrong pulse type"),
        }

        let drag = PulseLibrary::drag(100e-9, 25e-9, 0.5, 0.1);
        match drag {
            PulseShape::GaussianDrag { beta, .. } => {
                assert_eq!(beta, 0.1);
            }
            _ => panic!("Wrong pulse type"),
        }
    }

    #[test]
    fn test_pulse_calibration() {
        let cal = PulseCalibration {
            single_qubit_defaults: HashMap::new(),
            two_qubit_defaults: HashMap::new(),
            qubit_frequencies: vec![5.0, 5.1, 5.2],
            meas_frequencies: vec![6.5, 6.6, 6.7],
            drive_powers: vec![0.1, 0.1, 0.1],
            dt: 2.2222e-10,
        };

        let schedule = PulseTemplates::x_gate(0, &cal);
        assert!(!schedule.instructions.is_empty());
    }

    #[test]
    fn test_experiments() {
        let cal = PulseCalibration {
            single_qubit_defaults: HashMap::new(),
            two_qubit_defaults: HashMap::new(),
            qubit_frequencies: vec![5.0],
            meas_frequencies: vec![6.5],
            drive_powers: vec![0.1],
            dt: 2.2222e-10,
        };

        // Rabi experiment
        let rabi_schedules = PulseTemplates::rabi_experiment(
            0,
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            &cal,
        );
        assert_eq!(rabi_schedules.len(), 5);

        // T1 experiment
        let t1_schedules = PulseTemplates::t1_experiment(
            0,
            vec![0, 100, 200, 500, 1000],
            &cal,
        );
        assert_eq!(t1_schedules.len(), 5);

        // Ramsey experiment
        let ramsey_schedules = PulseTemplates::ramsey_experiment(
            0,
            vec![0, 50, 100, 200],
            1e6, // 1 MHz detuning
            &cal,
        );
        assert_eq!(ramsey_schedules.len(), 4);
    }
}