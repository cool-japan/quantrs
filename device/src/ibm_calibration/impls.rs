//! Implementation blocks and additional types for IBM calibration.

use std::collections::HashMap;

use crate::{DeviceError, DeviceResult};

use super::types::*;

/// Manager for custom pulse calibrations
#[derive(Debug, Clone)]
pub struct CalibrationManager {
    /// Backend name
    pub backend_name: String,
    /// Custom calibrations
    pub custom_calibrations: Vec<CustomCalibration>,
    /// Backend constraints
    pub constraints: PulseBackendConstraints,
    /// Default calibrations from backend
    pub defaults: Option<CalibrationData>,
}

impl CalibrationManager {
    /// Create a new CalibrationManager for a backend
    pub fn new(backend_name: impl Into<String>) -> Self {
        Self {
            backend_name: backend_name.into(),
            custom_calibrations: Vec::new(),
            constraints: PulseBackendConstraints::default(),
            defaults: None,
        }
    }

    /// Create with calibration data from backend
    pub fn with_defaults(backend_name: impl Into<String>, defaults: CalibrationData) -> Self {
        let mut manager = Self::new(backend_name);
        manager.defaults = Some(defaults);
        manager
    }

    /// Set backend constraints
    pub fn with_constraints(mut self, constraints: PulseBackendConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Add a custom calibration
    pub fn add_calibration(&mut self, calibration: CustomCalibration) -> DeviceResult<()> {
        let validation = self.validate_calibration(&calibration)?;
        if !validation.is_valid {
            return Err(DeviceError::CalibrationError(format!(
                "Invalid calibration: {}",
                validation.errors.join(", ")
            )));
        }
        self.custom_calibrations.push(calibration);
        Ok(())
    }

    /// Remove a custom calibration by gate name and qubits
    pub fn remove_calibration(&mut self, gate_name: &str, qubits: &[usize]) -> bool {
        let initial_len = self.custom_calibrations.len();
        self.custom_calibrations
            .retain(|c| !(c.gate_name == gate_name && c.qubits == qubits));
        self.custom_calibrations.len() < initial_len
    }

    /// Get a custom calibration
    pub fn get_calibration(&self, gate_name: &str, qubits: &[usize]) -> Option<&CustomCalibration> {
        self.custom_calibrations
            .iter()
            .find(|c| c.gate_name == gate_name && c.qubits == qubits)
    }

    /// Validate a calibration against backend constraints
    pub fn validate_calibration(
        &self,
        calibration: &CustomCalibration,
    ) -> DeviceResult<CalibrationValidation> {
        let mut result = CalibrationValidation::valid();

        // Check qubits are within range
        if let Some(defaults) = &self.defaults {
            for &qubit in &calibration.qubits {
                if qubit >= defaults.qubits.len() {
                    result.add_error(format!(
                        "Qubit {} is out of range (max {})",
                        qubit,
                        defaults.qubits.len() - 1
                    ));
                }
            }
        }

        // Check pulse schedule
        let schedule = &calibration.pulse_schedule;

        // Validate duration is within limits
        if schedule.duration_dt < self.constraints.min_pulse_duration {
            result.add_error(format!(
                "Schedule duration {} dt is below minimum {} dt",
                schedule.duration_dt, self.constraints.min_pulse_duration
            ));
        }

        if schedule.duration_dt > self.constraints.max_pulse_duration {
            result.add_error(format!(
                "Schedule duration {} dt exceeds maximum {} dt",
                schedule.duration_dt, self.constraints.max_pulse_duration
            ));
        }

        // Validate each instruction
        for instruction in &schedule.instructions {
            self.validate_instruction(instruction, &mut result);
        }

        Ok(result)
    }

    /// Validate a single pulse instruction
    fn validate_instruction(
        &self,
        instruction: &PulseInstruction,
        result: &mut CalibrationValidation,
    ) {
        match instruction {
            PulseInstruction::Play { pulse, channel, .. } => {
                // Check amplitude
                let amp = match pulse {
                    PulseWaveform::Gaussian { amp, .. } => *amp,
                    PulseWaveform::GaussianSquare { amp, .. } => *amp,
                    PulseWaveform::Drag { amp, .. } => *amp,
                    PulseWaveform::Constant { amp, .. } => *amp,
                    PulseWaveform::Waveform { samples, .. } => {
                        // Check all samples
                        for (i, sample) in samples.iter().enumerate() {
                            let magnitude = (sample.0 * sample.0 + sample.1 * sample.1).sqrt();
                            if magnitude > self.constraints.max_amplitude {
                                result.add_error(format!(
                                    "Waveform sample {} has amplitude {:.4} exceeding max {:.4}",
                                    i, magnitude, self.constraints.max_amplitude
                                ));
                            }
                        }
                        (0.0, 0.0) // Already checked
                    }
                };

                let magnitude = (amp.0 * amp.0 + amp.1 * amp.1).sqrt();
                if magnitude > self.constraints.max_amplitude {
                    result.add_error(format!(
                        "Pulse amplitude {:.4} exceeds maximum {:.4}",
                        magnitude, self.constraints.max_amplitude
                    ));
                }

                // Validate channel exists
                self.validate_channel(channel, result);

                // Check pulse duration granularity
                let duration = pulse.duration();
                if duration % self.constraints.pulse_granularity != 0 {
                    result.add_warning(format!(
                        "Pulse duration {} dt is not a multiple of granularity {} dt",
                        duration, self.constraints.pulse_granularity
                    ));
                }
            }
            PulseInstruction::SetFrequency {
                frequency, channel, ..
            } => {
                let freq_ghz = frequency / 1e9;
                if freq_ghz < self.constraints.frequency_range.0
                    || freq_ghz > self.constraints.frequency_range.1
                {
                    result.add_error(format!(
                        "Frequency {:.3} GHz is outside allowed range ({:.3}, {:.3}) GHz",
                        freq_ghz,
                        self.constraints.frequency_range.0,
                        self.constraints.frequency_range.1
                    ));
                }
                self.validate_channel(channel, result);
            }
            PulseInstruction::ShiftFrequency { channel, .. } => {
                self.validate_channel(channel, result);
            }
            PulseInstruction::SetPhase { channel, .. } => {
                self.validate_channel(channel, result);
            }
            PulseInstruction::ShiftPhase { channel, .. } => {
                self.validate_channel(channel, result);
            }
            PulseInstruction::Delay {
                duration, channel, ..
            } => {
                if *duration > self.constraints.max_pulse_duration {
                    result.add_warning(format!("Delay duration {} dt may be too long", duration));
                }
                self.validate_channel(channel, result);
            }
            PulseInstruction::Acquire { qubit, .. } => {
                if let Some(defaults) = &self.defaults {
                    if *qubit >= defaults.qubits.len() {
                        result.add_error(format!("Acquire qubit {} is out of range", qubit));
                    }
                }
            }
            PulseInstruction::Barrier { channels, .. } => {
                for channel in channels {
                    self.validate_channel(channel, result);
                }
            }
        }
    }

    /// Validate a pulse channel
    fn validate_channel(&self, channel: &PulseChannel, result: &mut CalibrationValidation) {
        match channel {
            PulseChannel::Drive(idx) => {
                if !self.constraints.drive_channels.contains(idx) {
                    result.add_error(format!("Drive channel d{} is not available", idx));
                }
            }
            PulseChannel::Control(idx) => {
                if !self.constraints.control_channels.contains(idx) {
                    result.add_error(format!("Control channel u{} is not available", idx));
                }
            }
            PulseChannel::Measure(idx) => {
                if !self.constraints.measure_channels.contains(idx) {
                    result.add_error(format!("Measure channel m{} is not available", idx));
                }
            }
            PulseChannel::Acquire(_) => {
                // Acquire channels are usually same as measure
            }
        }
    }

    /// Generate QASM 3.0 defcal statements for custom calibrations
    pub fn generate_defcal_statements(&self) -> String {
        let mut output = String::new();

        // Header comment
        output.push_str("// Custom pulse calibrations\n");
        output.push_str("// Generated by QuantRS2 CalibrationManager\n\n");

        for cal in &self.custom_calibrations {
            output.push_str(&self.calibration_to_defcal(cal));
            output.push('\n');
        }

        output
    }

    /// Convert a single calibration to defcal statement
    fn calibration_to_defcal(&self, calibration: &CustomCalibration) -> String {
        let mut output = String::new();

        // Add description as comment
        if let Some(desc) = &calibration.description {
            output.push_str(&format!("// {}\n", desc));
        }

        // Build parameter list
        let params = if calibration.parameters.is_empty() {
            String::new()
        } else {
            format!("({})", calibration.parameters.join(", "))
        };

        // Build qubit list
        let qubits = calibration
            .qubits
            .iter()
            .map(|q| format!("$q{}", q))
            .collect::<Vec<_>>()
            .join(", ");

        // Start defcal block
        output.push_str(&format!(
            "defcal {}{} {} {{\n",
            calibration.gate_name, params, qubits
        ));

        // Add instructions
        for instruction in &calibration.pulse_schedule.instructions {
            output.push_str(&format!(
                "    {};\n",
                self.instruction_to_openpulse(instruction)
            ));
        }

        output.push_str("}\n");
        output
    }

    /// Convert a pulse instruction to OpenPulse statement
    fn instruction_to_openpulse(&self, instruction: &PulseInstruction) -> String {
        match instruction {
            PulseInstruction::Play {
                pulse,
                channel,
                t0,
                name,
            } => {
                let pulse_str = self.waveform_to_openpulse(pulse);
                let name_comment = name
                    .as_ref()
                    .map(|n| format!(" // {}", n))
                    .unwrap_or_default();
                format!("play({}, {}) @ {}{}", pulse_str, channel, t0, name_comment)
            }
            PulseInstruction::SetFrequency {
                frequency,
                channel,
                t0,
            } => {
                format!("set_frequency({}, {:.6e}) @ {}", channel, frequency, t0)
            }
            PulseInstruction::ShiftFrequency {
                frequency,
                channel,
                t0,
            } => {
                format!("shift_frequency({}, {:.6e}) @ {}", channel, frequency, t0)
            }
            PulseInstruction::SetPhase { phase, channel, t0 } => {
                format!("set_phase({}, {:.6}) @ {}", channel, phase, t0)
            }
            PulseInstruction::ShiftPhase { phase, channel, t0 } => {
                format!("shift_phase({}, {:.6}) @ {}", channel, phase, t0)
            }
            PulseInstruction::Delay {
                duration,
                channel,
                t0,
            } => {
                format!("delay({}, {}) @ {}", channel, duration, t0)
            }
            PulseInstruction::Acquire {
                duration,
                qubit,
                memory_slot,
                t0,
            } => {
                format!(
                    "acquire({}, {}, c{}) @ {}",
                    duration, qubit, memory_slot, t0
                )
            }
            PulseInstruction::Barrier { channels, t0 } => {
                let channels_str = channels
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("barrier({}) @ {}", channels_str, t0)
            }
        }
    }

    /// Convert a waveform to OpenPulse expression
    fn waveform_to_openpulse(&self, waveform: &PulseWaveform) -> String {
        match waveform {
            PulseWaveform::Gaussian {
                amp,
                duration,
                sigma,
                name,
            } => {
                let name_str = name
                    .as_ref()
                    .map(|n| format!(", name=\"{}\"", n))
                    .unwrap_or_default();
                format!(
                    "gaussian({}, {}, {:.2}, {:.2}{})",
                    duration, amp.0, amp.1, sigma, name_str
                )
            }
            PulseWaveform::GaussianSquare {
                amp,
                duration,
                sigma,
                width,
                risefall_shape,
                name,
            } => {
                let name_str = name
                    .as_ref()
                    .map(|n| format!(", name=\"{}\"", n))
                    .unwrap_or_default();
                format!(
                    "gaussian_square({}, {}, {:.2}, {:.2}, {}, \"{}\"{name_str})",
                    duration, amp.0, amp.1, sigma, width, risefall_shape
                )
            }
            PulseWaveform::Drag {
                amp,
                duration,
                sigma,
                beta,
                name,
            } => {
                let name_str = name
                    .as_ref()
                    .map(|n| format!(", name=\"{}\"", n))
                    .unwrap_or_default();
                format!(
                    "drag({}, {}, {:.2}, {:.2}, {:.4}{})",
                    duration, amp.0, amp.1, sigma, beta, name_str
                )
            }
            PulseWaveform::Constant {
                amp,
                duration,
                name,
            } => {
                let name_str = name
                    .as_ref()
                    .map(|n| format!(", name=\"{}\"", n))
                    .unwrap_or_default();
                format!(
                    "constant({}, {}, {:.2}{})",
                    duration, amp.0, amp.1, name_str
                )
            }
            PulseWaveform::Waveform { samples, name } => {
                let name_str = name.as_deref().unwrap_or("custom");
                format!("waveform(\"{}\", {} samples)", name_str, samples.len())
            }
        }
    }

    /// Convert to IBM-compatible JSON format for upload
    pub fn to_ibm_format(&self) -> DeviceResult<serde_json::Value> {
        let mut calibrations = Vec::new();

        for cal in &self.custom_calibrations {
            let schedule_json = self.schedule_to_ibm_json(&cal.pulse_schedule)?;
            calibrations.push(serde_json::json!({
                "gate_name": cal.gate_name,
                "qubits": cal.qubits,
                "schedule": schedule_json,
                "parameters": cal.parameters,
            }));
        }

        Ok(serde_json::json!({
            "backend": self.backend_name,
            "calibrations": calibrations,
        }))
    }

    /// Convert pulse schedule to IBM JSON format
    fn schedule_to_ibm_json(&self, schedule: &PulseSchedule) -> DeviceResult<serde_json::Value> {
        let mut instructions = Vec::new();

        for inst in &schedule.instructions {
            instructions.push(self.instruction_to_ibm_json(inst)?);
        }

        Ok(serde_json::json!({
            "name": schedule.name,
            "instructions": instructions,
            "duration": schedule.duration_dt,
            "dt": schedule.dt,
        }))
    }

    /// Convert instruction to IBM JSON format
    fn instruction_to_ibm_json(
        &self,
        instruction: &PulseInstruction,
    ) -> DeviceResult<serde_json::Value> {
        match instruction {
            PulseInstruction::Play {
                pulse,
                channel,
                t0,
                name,
            } => Ok(serde_json::json!({
                "name": "play",
                "t0": t0,
                "ch": channel.to_string(),
                "pulse": self.waveform_to_ibm_json(pulse)?,
                "label": name,
            })),
            PulseInstruction::SetFrequency {
                frequency,
                channel,
                t0,
            } => Ok(serde_json::json!({
                "name": "setf",
                "t0": t0,
                "ch": channel.to_string(),
                "frequency": frequency,
            })),
            PulseInstruction::ShiftFrequency {
                frequency,
                channel,
                t0,
            } => Ok(serde_json::json!({
                "name": "shiftf",
                "t0": t0,
                "ch": channel.to_string(),
                "frequency": frequency,
            })),
            PulseInstruction::SetPhase { phase, channel, t0 } => Ok(serde_json::json!({
                "name": "setp",
                "t0": t0,
                "ch": channel.to_string(),
                "phase": phase,
            })),
            PulseInstruction::ShiftPhase { phase, channel, t0 } => Ok(serde_json::json!({
                "name": "fc",
                "t0": t0,
                "ch": channel.to_string(),
                "phase": phase,
            })),
            PulseInstruction::Delay {
                duration,
                channel,
                t0,
            } => Ok(serde_json::json!({
                "name": "delay",
                "t0": t0,
                "ch": channel.to_string(),
                "duration": duration,
            })),
            PulseInstruction::Acquire {
                duration,
                qubit,
                memory_slot,
                t0,
            } => Ok(serde_json::json!({
                "name": "acquire",
                "t0": t0,
                "duration": duration,
                "qubits": [qubit],
                "memory_slot": [memory_slot],
            })),
            PulseInstruction::Barrier { channels, t0 } => {
                let ch_strs: Vec<String> = channels.iter().map(|c| c.to_string()).collect();
                Ok(serde_json::json!({
                    "name": "barrier",
                    "t0": t0,
                    "channels": ch_strs,
                }))
            }
        }
    }

    /// Convert waveform to IBM JSON format
    fn waveform_to_ibm_json(&self, waveform: &PulseWaveform) -> DeviceResult<serde_json::Value> {
        match waveform {
            PulseWaveform::Gaussian {
                amp,
                duration,
                sigma,
                name,
            } => Ok(serde_json::json!({
                "pulse_type": "Gaussian",
                "parameters": {
                    "amp": [amp.0, amp.1],
                    "duration": duration,
                    "sigma": sigma,
                },
                "name": name,
            })),
            PulseWaveform::GaussianSquare {
                amp,
                duration,
                sigma,
                width,
                risefall_shape,
                name,
            } => Ok(serde_json::json!({
                "pulse_type": "GaussianSquare",
                "parameters": {
                    "amp": [amp.0, amp.1],
                    "duration": duration,
                    "sigma": sigma,
                    "width": width,
                    "risefall_sigma_ratio": risefall_shape,
                },
                "name": name,
            })),
            PulseWaveform::Drag {
                amp,
                duration,
                sigma,
                beta,
                name,
            } => Ok(serde_json::json!({
                "pulse_type": "Drag",
                "parameters": {
                    "amp": [amp.0, amp.1],
                    "duration": duration,
                    "sigma": sigma,
                    "beta": beta,
                },
                "name": name,
            })),
            PulseWaveform::Constant {
                amp,
                duration,
                name,
            } => Ok(serde_json::json!({
                "pulse_type": "Constant",
                "parameters": {
                    "amp": [amp.0, amp.1],
                    "duration": duration,
                },
                "name": name,
            })),
            PulseWaveform::Waveform { samples, name } => {
                // Convert samples to lists
                let real: Vec<f64> = samples.iter().map(|(r, _)| *r).collect();
                let imag: Vec<f64> = samples.iter().map(|(_, i)| *i).collect();
                Ok(serde_json::json!({
                    "pulse_type": "Waveform",
                    "samples": {
                        "real": real,
                        "imag": imag,
                    },
                    "name": name,
                }))
            }
        }
    }

    /// Get the number of custom calibrations
    pub fn len(&self) -> usize {
        self.custom_calibrations.len()
    }

    /// Check if there are no custom calibrations
    pub fn is_empty(&self) -> bool {
        self.custom_calibrations.is_empty()
    }

    /// List all custom calibration gate names
    pub fn calibration_names(&self) -> Vec<(&str, &[usize])> {
        self.custom_calibrations
            .iter()
            .map(|c| (c.gate_name.as_str(), c.qubits.as_slice()))
            .collect()
    }
}

/// Builder for creating custom pulse calibrations
#[derive(Debug, Clone)]
pub struct CalibrationBuilder {
    gate_name: String,
    qubits: Vec<usize>,
    instructions: Vec<PulseInstruction>,
    parameters: Vec<String>,
    description: Option<String>,
    dt: f64,
}

impl CalibrationBuilder {
    /// Create a new calibration builder
    pub fn new(gate_name: impl Into<String>, qubits: Vec<usize>) -> Self {
        Self {
            gate_name: gate_name.into(),
            qubits,
            instructions: Vec::new(),
            parameters: Vec::new(),
            description: None,
            dt: 2.22e-10, // Default dt
        }
    }

    /// Set the device time unit
    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Add a parameter
    pub fn parameter(mut self, param: impl Into<String>) -> Self {
        self.parameters.push(param.into());
        self
    }

    /// Set description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a Gaussian pulse
    pub fn gaussian(
        mut self,
        channel: PulseChannel,
        t0: u64,
        duration: u64,
        amp: (f64, f64),
        sigma: f64,
    ) -> Self {
        self.instructions.push(PulseInstruction::Play {
            pulse: PulseWaveform::Gaussian {
                amp,
                duration,
                sigma,
                name: None,
            },
            channel,
            t0,
            name: None,
        });
        self
    }

    /// Add a DRAG pulse
    pub fn drag(
        mut self,
        channel: PulseChannel,
        t0: u64,
        duration: u64,
        amp: (f64, f64),
        sigma: f64,
        beta: f64,
    ) -> Self {
        self.instructions.push(PulseInstruction::Play {
            pulse: PulseWaveform::Drag {
                amp,
                duration,
                sigma,
                beta,
                name: None,
            },
            channel,
            t0,
            name: None,
        });
        self
    }

    /// Add a Gaussian square pulse
    pub fn gaussian_square(
        mut self,
        channel: PulseChannel,
        t0: u64,
        duration: u64,
        amp: (f64, f64),
        sigma: f64,
        width: u64,
    ) -> Self {
        self.instructions.push(PulseInstruction::Play {
            pulse: PulseWaveform::GaussianSquare {
                amp,
                duration,
                sigma,
                width,
                risefall_shape: "gaussian".to_string(),
                name: None,
            },
            channel,
            t0,
            name: None,
        });
        self
    }

    /// Add a constant pulse
    pub fn constant(
        mut self,
        channel: PulseChannel,
        t0: u64,
        duration: u64,
        amp: (f64, f64),
    ) -> Self {
        self.instructions.push(PulseInstruction::Play {
            pulse: PulseWaveform::Constant {
                amp,
                duration,
                name: None,
            },
            channel,
            t0,
            name: None,
        });
        self
    }

    /// Add a phase shift
    pub fn shift_phase(mut self, channel: PulseChannel, t0: u64, phase: f64) -> Self {
        self.instructions
            .push(PulseInstruction::ShiftPhase { phase, channel, t0 });
        self
    }

    /// Add a frequency shift
    pub fn shift_frequency(mut self, channel: PulseChannel, t0: u64, frequency: f64) -> Self {
        self.instructions.push(PulseInstruction::ShiftFrequency {
            frequency,
            channel,
            t0,
        });
        self
    }

    /// Add a delay
    pub fn delay(mut self, channel: PulseChannel, t0: u64, duration: u64) -> Self {
        self.instructions.push(PulseInstruction::Delay {
            duration,
            channel,
            t0,
        });
        self
    }

    /// Add a barrier
    pub fn barrier(mut self, channels: Vec<PulseChannel>, t0: u64) -> Self {
        self.instructions
            .push(PulseInstruction::Barrier { channels, t0 });
        self
    }

    /// Build the custom calibration
    pub fn build(self) -> CustomCalibration {
        // Calculate total duration
        let duration_dt = self
            .instructions
            .iter()
            .map(|i| match i {
                PulseInstruction::Play { pulse, t0, .. } => t0 + pulse.duration(),
                PulseInstruction::Delay { duration, t0, .. } => t0 + duration,
                PulseInstruction::Acquire { duration, t0, .. } => t0 + duration,
                _ => 0,
            })
            .max()
            .unwrap_or(0);

        CustomCalibration {
            gate_name: self.gate_name.clone(),
            qubits: self.qubits,
            pulse_schedule: PulseSchedule {
                name: self.gate_name,
                instructions: self.instructions,
                duration_dt,
                dt: self.dt,
            },
            parameters: self.parameters,
            description: self.description,
        }
    }
}

/// Instruction properties (Qiskit Target compatibility)
#[derive(Debug, Clone)]
pub struct InstructionProperties {
    /// Duration in seconds
    pub duration: Option<f64>,
    /// Error rate
    pub error: Option<f64>,
    /// Calibration data
    pub calibration: Option<String>,
}

impl Default for InstructionProperties {
    fn default() -> Self {
        Self {
            duration: None,
            error: None,
            calibration: None,
        }
    }
}

/// Target representation (Qiskit Target compatibility)
#[derive(Debug, Clone)]
pub struct Target {
    /// Number of qubits
    pub num_qubits: usize,
    /// Description
    pub description: String,
    /// Instruction properties map
    pub instruction_properties: HashMap<String, HashMap<Vec<usize>, InstructionProperties>>,
    /// Coupling map
    pub coupling_map: Vec<(usize, usize)>,
}

impl Target {
    /// Create a new Target from calibration data
    pub fn from_calibration(calibration: &CalibrationData) -> Self {
        let mut instruction_properties = HashMap::new();

        // Add gate properties
        for (gate_name, gates) in &calibration.gates {
            let mut props = HashMap::new();
            for gate in gates {
                props.insert(
                    gate.qubits.clone(),
                    InstructionProperties {
                        duration: Some(gate.gate_length.as_secs_f64()),
                        error: Some(gate.gate_error),
                        calibration: None,
                    },
                );
            }
            instruction_properties.insert(gate_name.clone(), props);
        }

        // Add measure properties
        let mut measure_props = HashMap::new();
        for qubit in &calibration.qubits {
            measure_props.insert(
                vec![qubit.qubit_id],
                InstructionProperties {
                    duration: Some(qubit.readout_length.as_secs_f64()),
                    error: Some(qubit.readout_error),
                    calibration: None,
                },
            );
        }
        instruction_properties.insert("measure".to_string(), measure_props);

        Self {
            num_qubits: calibration.qubits.len(),
            description: format!("Target for {}", calibration.backend_name),
            instruction_properties,
            coupling_map: calibration.general.coupling_map.clone(),
        }
    }

    /// Check if an instruction is supported on given qubits
    pub fn instruction_supported(&self, instruction: &str, qubits: &[usize]) -> bool {
        self.instruction_properties
            .get(instruction)
            .is_some_and(|props| props.contains_key(qubits))
    }

    /// Get instruction properties
    pub fn get_instruction_properties(
        &self,
        instruction: &str,
        qubits: &[usize],
    ) -> Option<&InstructionProperties> {
        self.instruction_properties
            .get(instruction)
            .and_then(|props| props.get(qubits))
    }
}

#[cfg(test)]
#[allow(clippy::pedantic, clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_qubit_calibration_quality_score() {
        let qubit = QubitCalibration {
            qubit_id: 0,
            t1: Duration::from_micros(100),
            t2: Duration::from_micros(75),
            frequency: 5.0,
            anharmonicity: -0.34,
            readout_error: 0.01,
            readout_length: Duration::from_nanos(500),
            prob_meas0_prep1: 0.02,
            prob_meas1_prep0: 0.01,
        };

        let score = qubit.quality_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_gate_calibration_fidelity() {
        let gate = GateCalibration {
            gate_name: "cx".to_string(),
            qubits: vec![0, 1],
            gate_error: 0.005,
            gate_length: Duration::from_nanos(300),
            parameters: HashMap::new(),
        };

        assert!((gate.fidelity() - 0.995).abs() < 1e-10);
    }

    #[test]
    fn test_processor_type_typical_values() {
        let eagle = ProcessorType::Eagle;
        assert!(eagle.typical_t1().as_micros() > 100);
        assert!(eagle.typical_cx_error() < 0.01);
    }

    #[test]
    fn test_target_instruction_supported() {
        let mut instruction_properties = HashMap::new();
        let mut cx_props = HashMap::new();
        cx_props.insert(
            vec![0, 1],
            InstructionProperties {
                duration: Some(3e-7),
                error: Some(0.005),
                calibration: None,
            },
        );
        instruction_properties.insert("cx".to_string(), cx_props);

        let target = Target {
            num_qubits: 5,
            description: "Test target".to_string(),
            instruction_properties,
            coupling_map: vec![(0, 1), (1, 2)],
        };

        assert!(target.instruction_supported("cx", &[0, 1]));
        assert!(!target.instruction_supported("cx", &[0, 2]));
    }

    // =========================================================================
    // CalibrationManager Tests
    // =========================================================================

    #[test]
    fn test_calibration_manager_new() {
        let manager = CalibrationManager::new("ibm_brisbane");
        assert_eq!(manager.backend_name, "ibm_brisbane");
        assert!(manager.is_empty());
    }

    #[test]
    fn test_calibration_builder_drag_pulse() {
        let cal = CalibrationBuilder::new("x", vec![0])
            .description("Custom X gate with DRAG pulse")
            .drag(PulseChannel::Drive(0), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();

        assert_eq!(cal.gate_name, "x");
        assert_eq!(cal.qubits, vec![0]);
        assert_eq!(cal.pulse_schedule.instructions.len(), 1);
        assert_eq!(cal.pulse_schedule.duration_dt, 160);
    }

    #[test]
    fn test_calibration_builder_gaussian_pulse() {
        let cal = CalibrationBuilder::new("sx", vec![0])
            .gaussian(PulseChannel::Drive(0), 0, 80, (0.25, 0.0), 20.0)
            .build();

        assert_eq!(cal.gate_name, "sx");
        assert_eq!(cal.pulse_schedule.duration_dt, 80);
    }

    #[test]
    fn test_calibration_builder_multi_instruction() {
        let cal = CalibrationBuilder::new("cx", vec![0, 1])
            .description("Cross-resonance CNOT gate")
            .shift_phase(PulseChannel::Drive(0), 0, std::f64::consts::PI / 2.0)
            .gaussian_square(PulseChannel::Control(0), 0, 1024, (0.8, 0.0), 64.0, 896)
            .shift_phase(PulseChannel::Drive(1), 1024, -std::f64::consts::PI / 2.0)
            .build();

        assert_eq!(cal.gate_name, "cx");
        assert_eq!(cal.qubits, vec![0, 1]);
        assert_eq!(cal.pulse_schedule.instructions.len(), 3);
        assert_eq!(cal.pulse_schedule.duration_dt, 1024);
    }

    #[test]
    fn test_calibration_manager_add_and_get() {
        let mut manager = CalibrationManager::new("test_backend");

        let cal = CalibrationBuilder::new("x", vec![0])
            .drag(PulseChannel::Drive(0), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();

        manager
            .add_calibration(cal)
            .expect("Should add calibration");
        assert_eq!(manager.len(), 1);

        let retrieved = manager.get_calibration("x", &[0]);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.map(|c| &c.gate_name), Some(&"x".to_string()));
    }

    #[test]
    fn test_calibration_manager_remove() {
        let mut manager = CalibrationManager::new("test_backend");

        let cal1 = CalibrationBuilder::new("x", vec![0])
            .drag(PulseChannel::Drive(0), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();
        let cal2 = CalibrationBuilder::new("x", vec![1])
            .drag(PulseChannel::Drive(1), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();

        manager.add_calibration(cal1).expect("add");
        manager.add_calibration(cal2).expect("add");
        assert_eq!(manager.len(), 2);

        assert!(manager.remove_calibration("x", &[0]));
        assert_eq!(manager.len(), 1);

        assert!(manager.get_calibration("x", &[0]).is_none());
        assert!(manager.get_calibration("x", &[1]).is_some());
    }

    #[test]
    fn test_calibration_validation_amplitude_error() {
        let manager = CalibrationManager::new("test_backend");

        // Create calibration with amplitude > 1.0
        let cal = CalibrationBuilder::new("x", vec![0])
            .drag(PulseChannel::Drive(0), 0, 160, (1.5, 0.0), 40.0, 0.1)
            .build();

        let result = manager.validate_calibration(&cal).expect("validation");
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_calibration_validation_duration_warning() {
        let mut constraints = PulseBackendConstraints::default();
        constraints.pulse_granularity = 16;

        let manager = CalibrationManager::new("test_backend").with_constraints(constraints);

        // Duration 100 is not multiple of 16
        let cal = CalibrationBuilder::new("x", vec![0])
            .gaussian(PulseChannel::Drive(0), 0, 100, (0.5, 0.0), 25.0)
            .build();

        let result = manager.validate_calibration(&cal).expect("validation");
        // This should produce a warning, but still be valid
        assert!(result.is_valid);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_generate_defcal_statements() {
        let mut manager = CalibrationManager::new("test_backend");

        let cal = CalibrationBuilder::new("x", vec![0])
            .description("Custom X gate")
            .drag(PulseChannel::Drive(0), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();

        manager.add_calibration(cal).expect("add");

        let defcal = manager.generate_defcal_statements();
        assert!(defcal.contains("defcal x $q0"));
        assert!(defcal.contains("drag("));
        assert!(defcal.contains("Custom X gate"));
    }

    #[test]
    fn test_to_ibm_format() {
        let mut manager = CalibrationManager::new("ibm_brisbane");

        let cal = CalibrationBuilder::new("sx", vec![0])
            .gaussian(PulseChannel::Drive(0), 0, 80, (0.25, 0.0), 20.0)
            .build();

        manager.add_calibration(cal).expect("add");

        let json = manager.to_ibm_format().expect("should serialize");
        let obj = json.as_object().expect("should be object");

        assert_eq!(
            obj.get("backend").and_then(|v| v.as_str()),
            Some("ibm_brisbane")
        );
        assert!(obj.get("calibrations").is_some());
    }

    #[test]
    fn test_pulse_waveform_duration() {
        let gaussian = PulseWaveform::Gaussian {
            amp: (0.5, 0.0),
            duration: 160,
            sigma: 40.0,
            name: None,
        };
        assert_eq!(gaussian.duration(), 160);

        let drag = PulseWaveform::Drag {
            amp: (0.5, 0.0),
            duration: 80,
            sigma: 20.0,
            beta: 0.1,
            name: None,
        };
        assert_eq!(drag.duration(), 80);

        let waveform = PulseWaveform::Waveform {
            samples: vec![(0.1, 0.0), (0.2, 0.0), (0.3, 0.0)],
            name: Some("custom".to_string()),
        };
        assert_eq!(waveform.duration(), 3);
    }

    #[test]
    fn test_pulse_channel_display() {
        assert_eq!(format!("{}", PulseChannel::Drive(0)), "d0");
        assert_eq!(format!("{}", PulseChannel::Control(5)), "u5");
        assert_eq!(format!("{}", PulseChannel::Measure(2)), "m2");
        assert_eq!(format!("{}", PulseChannel::Acquire(3)), "a3");
    }

    #[test]
    fn test_calibration_builder_with_parameters() {
        let cal = CalibrationBuilder::new("rz", vec![0])
            .parameter("theta")
            .shift_phase(PulseChannel::Drive(0), 0, 0.0)
            .build();

        assert_eq!(cal.parameters, vec!["theta"]);
    }

    #[test]
    fn test_calibration_names() {
        let mut manager = CalibrationManager::new("test_backend");

        let cal1 = CalibrationBuilder::new("x", vec![0])
            .drag(PulseChannel::Drive(0), 0, 160, (0.5, 0.0), 40.0, 0.1)
            .build();
        let cal2 = CalibrationBuilder::new("cx", vec![0, 1])
            .gaussian_square(PulseChannel::Control(0), 0, 1024, (0.8, 0.0), 64.0, 896)
            .build();

        manager.add_calibration(cal1).expect("add");
        manager.add_calibration(cal2).expect("add");

        let names = manager.calibration_names();
        assert_eq!(names.len(), 2);
        assert!(names
            .iter()
            .any(|(name, qubits)| *name == "x" && *qubits == [0]));
        assert!(names
            .iter()
            .any(|(name, qubits)| *name == "cx" && *qubits == [0, 1]));
    }
}
