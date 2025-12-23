//! Device-specific noise models based on calibration data
//!
//! This module creates realistic noise models from device calibration data,
//! enabling accurate simulation of quantum hardware behavior.

use scirs2_core::random::Rng;
use scirs2_core::Complex64;
use std::collections::HashMap;

use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use crate::calibration::{
    DeviceCalibration, QubitCalibration, QubitReadoutData, SingleQubitGateData,
    TwoQubitGateCalibration,
};
use scirs2_core::random::prelude::*;

/// Noise model derived from device calibration
#[derive(Debug, Clone)]
pub struct CalibrationNoiseModel {
    /// Device identifier
    pub device_id: String,
    /// Qubit noise parameters
    pub qubit_noise: HashMap<QubitId, QubitNoiseParams>,
    /// Gate noise parameters
    pub gate_noise: HashMap<String, GateNoiseParams>,
    /// Two-qubit gate noise
    pub two_qubit_noise: HashMap<(QubitId, QubitId), TwoQubitNoiseParams>,
    /// Readout noise parameters
    pub readout_noise: HashMap<QubitId, ReadoutNoiseParams>,
    /// Crosstalk noise
    pub crosstalk: CrosstalkNoise,
    /// Temperature (mK)
    pub temperature: f64,
}

/// Noise parameters for individual qubits
#[derive(Debug, Clone)]
pub struct QubitNoiseParams {
    /// T1 decay rate (1/μs)
    pub gamma_1: f64,
    /// T2 dephasing rate (1/μs)
    pub gamma_phi: f64,
    /// Thermal excitation probability
    pub thermal_population: f64,
    /// Frequency drift (Hz)
    pub frequency_drift: f64,
    /// Amplitude of 1/f noise
    pub flicker_noise: f64,
}

/// Gate-specific noise parameters
#[derive(Debug, Clone)]
pub struct GateNoiseParams {
    /// Coherent error (over/under rotation)
    pub coherent_error: f64,
    /// Incoherent error rate
    pub incoherent_error: f64,
    /// Gate duration (ns)
    pub duration: f64,
    /// Depolarizing rate during gate
    pub depolarizing_rate: f64,
    /// Amplitude noise
    pub amplitude_noise: f64,
    /// Phase noise
    pub phase_noise: f64,
}

/// Two-qubit gate noise parameters
#[derive(Debug, Clone)]
pub struct TwoQubitNoiseParams {
    /// Base gate noise
    pub gate_noise: GateNoiseParams,
    /// ZZ coupling strength (MHz)
    pub zz_coupling: f64,
    /// Crosstalk to spectator qubits
    pub spectator_crosstalk: HashMap<QubitId, f64>,
    /// Directional bias
    pub directional_error: f64,
}

/// Readout noise parameters
#[derive(Debug, Clone)]
pub struct ReadoutNoiseParams {
    /// Assignment error matrix [[p(0|0), p(0|1)], [p(1|0), p(1|1)]]
    pub assignment_matrix: [[f64; 2]; 2],
    /// Readout-induced excitation
    pub readout_excitation: f64,
    /// Readout-induced relaxation
    pub readout_relaxation: f64,
}

/// Crosstalk noise model
#[derive(Debug, Clone)]
pub struct CrosstalkNoise {
    /// Crosstalk matrix
    pub crosstalk_matrix: Vec<Vec<f64>>,
    /// Threshold for significant crosstalk
    pub threshold: f64,
    /// Crosstalk during single-qubit gates
    pub single_qubit_crosstalk: f64,
    /// Crosstalk during two-qubit gates
    pub two_qubit_crosstalk: f64,
}

impl CalibrationNoiseModel {
    /// Create noise model from device calibration
    pub fn from_calibration(calibration: &DeviceCalibration) -> Self {
        let mut qubit_noise = HashMap::new();
        let mut readout_noise = HashMap::new();

        // Extract qubit noise parameters
        for (qubit_id, qubit_cal) in &calibration.qubit_calibrations {
            qubit_noise.insert(
                *qubit_id,
                QubitNoiseParams {
                    gamma_1: 1.0 / qubit_cal.t1,
                    gamma_phi: 1.0 / qubit_cal.t2 - 0.5 / qubit_cal.t1,
                    thermal_population: qubit_cal.thermal_population,
                    frequency_drift: 1e3, // 1 kHz drift (typical)
                    flicker_noise: 1e-6,  // Typical 1/f noise amplitude
                },
            );
        }

        // Extract readout noise
        for (qubit_id, readout_data) in &calibration.readout_calibration.qubit_readout {
            let p00 = readout_data.p0_given_0;
            let p11 = readout_data.p1_given_1;

            readout_noise.insert(
                *qubit_id,
                ReadoutNoiseParams {
                    assignment_matrix: [[p00, 1.0 - p11], [1.0 - p00, p11]],
                    readout_excitation: 0.001, // Typical value
                    readout_relaxation: 0.002, // Typical value
                },
            );
        }

        // Extract gate noise
        let mut gate_noise = HashMap::new();
        for (gate_name, gate_cal) in &calibration.single_qubit_gates {
            // Average over all qubits
            let mut total_error = 0.0;
            let mut total_duration = 0.0;
            let mut count = 0;

            for gate_data in gate_cal.qubit_data.values() {
                total_error += gate_data.error_rate;
                total_duration += gate_data.duration;
                count += 1;
            }

            if count > 0 {
                let avg_error = total_error / count as f64;
                let avg_duration = total_duration / count as f64;

                gate_noise.insert(
                    gate_name.clone(),
                    GateNoiseParams {
                        coherent_error: avg_error * 0.3,   // 30% coherent
                        incoherent_error: avg_error * 0.7, // 70% incoherent
                        duration: avg_duration,
                        depolarizing_rate: avg_error / avg_duration,
                        amplitude_noise: 0.001, // Typical
                        phase_noise: 0.002,     // Typical
                    },
                );
            }
        }

        // Extract two-qubit gate noise
        let mut two_qubit_noise = HashMap::new();
        for ((control, target), gate_cal) in &calibration.two_qubit_gates {
            let base_noise = GateNoiseParams {
                coherent_error: gate_cal.error_rate * 0.4,
                incoherent_error: gate_cal.error_rate * 0.6,
                duration: gate_cal.duration,
                depolarizing_rate: gate_cal.error_rate / gate_cal.duration,
                amplitude_noise: 0.002,
                phase_noise: 0.003,
            };

            // Add to gate_noise map as well
            gate_noise.insert(gate_cal.gate_name.clone(), base_noise.clone());

            // Estimate spectator crosstalk
            let mut spectator_crosstalk = HashMap::new();

            // Add neighboring qubits
            for offset in [-1, 1] {
                let spectator_id = control.id() as i32 + offset;
                if spectator_id >= 0 && spectator_id < calibration.topology.num_qubits as i32 {
                    spectator_crosstalk.insert(QubitId(spectator_id as u32), 0.001);
                }

                let spectator_id = target.id() as i32 + offset;
                if spectator_id >= 0 && spectator_id < calibration.topology.num_qubits as i32 {
                    spectator_crosstalk.insert(QubitId(spectator_id as u32), 0.001);
                }
            }

            two_qubit_noise.insert(
                (*control, *target),
                TwoQubitNoiseParams {
                    gate_noise: base_noise,
                    zz_coupling: 0.1, // Typical residual coupling
                    spectator_crosstalk,
                    directional_error: if gate_cal.directional { 0.01 } else { 0.0 },
                },
            );
        }

        // Extract crosstalk
        let crosstalk = CrosstalkNoise {
            crosstalk_matrix: calibration.crosstalk_matrix.matrix.clone(),
            threshold: calibration.crosstalk_matrix.significance_threshold,
            single_qubit_crosstalk: 0.001,
            two_qubit_crosstalk: 0.01,
        };

        // Get temperature
        let temperature = calibration
            .qubit_calibrations
            .values()
            .filter_map(|q| q.temperature)
            .sum::<f64>()
            / calibration.qubit_calibrations.len() as f64;

        Self {
            device_id: calibration.device_id.clone(),
            qubit_noise,
            gate_noise,
            two_qubit_noise,
            readout_noise,
            crosstalk,
            temperature,
        }
    }

    /// Apply noise to a quantum state after a gate operation
    pub fn apply_gate_noise(
        &self,
        state: &mut Vec<Complex64>,
        gate: &dyn GateOp,
        qubits: &[QubitId],
        duration_ns: f64,
        rng: &mut impl Rng,
    ) -> QuantRS2Result<()> {
        let gate_name = gate.name();

        match qubits.len() {
            1 => self.apply_single_qubit_noise(state, gate_name, qubits[0], duration_ns, rng),
            2 => {
                self.apply_two_qubit_noise(state, gate_name, qubits[0], qubits[1], duration_ns, rng)
            }
            _ => {
                // For multi-qubit gates, apply as decomposed single/two-qubit noise
                for &qubit in qubits {
                    self.apply_single_qubit_noise(
                        state,
                        gate_name,
                        qubit,
                        duration_ns / qubits.len() as f64,
                        rng,
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Apply single-qubit gate noise
    fn apply_single_qubit_noise(
        &self,
        state: &mut Vec<Complex64>,
        gate_name: &str,
        qubit: QubitId,
        duration_ns: f64,
        rng: &mut impl Rng,
    ) -> QuantRS2Result<()> {
        // Get gate noise parameters
        let gate_params = self.gate_noise.get(gate_name);
        let qubit_params = self.qubit_noise.get(&qubit);

        if let Some(params) = gate_params {
            // Apply coherent error (over/under rotation)
            if params.coherent_error > 0.0 {
                let error_angle = rng.gen_range(-params.coherent_error..params.coherent_error);
                self.apply_rotation_error(state, qubit, error_angle)?;
            }

            // Apply amplitude noise
            if params.amplitude_noise > 0.0 {
                let amplitude_error =
                    1.0 + rng.gen_range(-params.amplitude_noise..params.amplitude_noise);
                self.apply_amplitude_scaling(state, qubit, amplitude_error)?;
            }

            // Apply phase noise
            if params.phase_noise > 0.0 {
                let phase_error = rng.gen_range(-params.phase_noise..params.phase_noise);
                self.apply_phase_error(state, qubit, phase_error)?;
            }
        }

        // Apply decoherence during gate
        if let Some(qubit_params) = qubit_params {
            let actual_duration = duration_ns / 1000.0; // Convert to μs

            // T1 decay
            let decay_prob = 1.0 - (-actual_duration * qubit_params.gamma_1).exp();
            if rng.gen::<f64>() < decay_prob {
                self.apply_amplitude_damping(state, qubit, decay_prob)?;
            }

            // T2 dephasing
            let dephase_prob = 1.0 - (-actual_duration * qubit_params.gamma_phi).exp();
            if rng.gen::<f64>() < dephase_prob {
                self.apply_phase_damping(state, qubit, dephase_prob)?;
            }
        }

        Ok(())
    }

    /// Apply two-qubit gate noise
    fn apply_two_qubit_noise(
        &self,
        state: &mut Vec<Complex64>,
        gate_name: &str,
        control: QubitId,
        target: QubitId,
        duration_ns: f64,
        rng: &mut impl Rng,
    ) -> QuantRS2Result<()> {
        if let Some(params) = self.two_qubit_noise.get(&(control, target)) {
            // Apply base gate noise
            let gate_params = &params.gate_noise;

            // Coherent errors
            if gate_params.coherent_error > 0.0 {
                let error = rng.gen_range(-gate_params.coherent_error..gate_params.coherent_error);
                self.apply_two_qubit_rotation_error(state, control, target, error)?;
            }

            // ZZ coupling during idle
            if params.zz_coupling > 0.0 {
                let zz_angle = params.zz_coupling * duration_ns / 1000.0; // MHz * μs = radians
                self.apply_zz_interaction(state, control, target, zz_angle)?;
            }

            // Spectator crosstalk
            for (&spectator, &coupling) in &params.spectator_crosstalk {
                if rng.gen::<f64>() < coupling {
                    self.apply_crosstalk_error(state, spectator, coupling * 0.1)?;
                }
            }

            // Apply decoherence to both qubits
            self.apply_single_qubit_noise(state, gate_name, control, duration_ns / 2.0, rng)?;
            self.apply_single_qubit_noise(state, gate_name, target, duration_ns / 2.0, rng)?;
        }

        Ok(())
    }

    /// Apply readout noise
    pub fn apply_readout_noise(&self, measurement: u8, qubit: QubitId, rng: &mut impl Rng) -> u8 {
        if let Some(params) = self.readout_noise.get(&qubit) {
            let prob = rng.gen::<f64>();

            if measurement == 0 {
                if prob > params.assignment_matrix[0][0] {
                    return 1;
                }
            } else {
                if prob > params.assignment_matrix[1][1] {
                    return 0;
                }
            }
        }

        measurement
    }

    // Helper methods for applying specific noise channels

    fn apply_rotation_error(
        &self,
        state: &mut Vec<Complex64>,
        qubit: QubitId,
        angle: f64,
    ) -> QuantRS2Result<()> {
        // Apply Z rotation error
        let n_qubits = (state.len() as f64).log2() as usize;
        let qubit_idx = qubit.id() as usize;

        for i in 0..state.len() {
            if (i >> qubit_idx) & 1 == 1 {
                state[i] *= Complex64::from_polar(1.0, angle);
            }
        }

        Ok(())
    }

    fn apply_amplitude_scaling(
        &self,
        state: &mut Vec<Complex64>,
        qubit: QubitId,
        scale: f64,
    ) -> QuantRS2Result<()> {
        // This is a simplified model - in reality would need more sophisticated approach
        for amp in state.iter_mut() {
            *amp *= scale;
        }

        // Renormalize
        let norm = state.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        for amp in state.iter_mut() {
            *amp /= norm;
        }

        Ok(())
    }

    fn apply_phase_error(
        &self,
        state: &mut Vec<Complex64>,
        qubit: QubitId,
        phase: f64,
    ) -> QuantRS2Result<()> {
        let qubit_idx = qubit.id() as usize;

        for i in 0..state.len() {
            if (i >> qubit_idx) & 1 == 1 {
                state[i] *= Complex64::from_polar(1.0, phase);
            }
        }

        Ok(())
    }

    fn apply_amplitude_damping(
        &self,
        state: &mut Vec<Complex64>,
        qubit: QubitId,
        gamma: f64,
    ) -> QuantRS2Result<()> {
        // Simplified amplitude damping
        let qubit_idx = qubit.id() as usize;
        let damping_factor = (1.0 - gamma).sqrt();

        for i in 0..state.len() {
            if (i >> qubit_idx) & 1 == 1 {
                state[i] *= damping_factor;
            }
        }

        Ok(())
    }

    fn apply_phase_damping(
        &self,
        state: &mut Vec<Complex64>,
        qubit: QubitId,
        gamma: f64,
    ) -> QuantRS2Result<()> {
        // Apply random phase to superposition states
        let qubit_idx = qubit.id() as usize;

        // This is simplified - proper implementation would track density matrix
        for i in 0..state.len() {
            if (i >> qubit_idx) & 1 == 1 {
                state[i] *= (1.0 - gamma).sqrt();
            }
        }

        Ok(())
    }

    fn apply_two_qubit_rotation_error(
        &self,
        state: &mut Vec<Complex64>,
        control: QubitId,
        target: QubitId,
        angle: f64,
    ) -> QuantRS2Result<()> {
        // Apply ZZ rotation error
        let control_idx = control.id() as usize;
        let target_idx = target.id() as usize;

        for i in 0..state.len() {
            let control_bit = (i >> control_idx) & 1;
            let target_bit = (i >> target_idx) & 1;

            if control_bit == 1 && target_bit == 1 {
                state[i] *= Complex64::from_polar(1.0, angle);
            }
        }

        Ok(())
    }

    fn apply_zz_interaction(
        &self,
        state: &mut Vec<Complex64>,
        qubit1: QubitId,
        qubit2: QubitId,
        angle: f64,
    ) -> QuantRS2Result<()> {
        let idx1 = qubit1.id() as usize;
        let idx2 = qubit2.id() as usize;

        for i in 0..state.len() {
            let bit1 = (i >> idx1) & 1;
            let bit2 = (i >> idx2) & 1;

            // ZZ interaction: |00⟩ and |11⟩ get +angle, |01⟩ and |10⟩ get -angle
            let phase = if bit1 == bit2 { angle } else { -angle };
            state[i] *= Complex64::from_polar(1.0, phase / 2.0);
        }

        Ok(())
    }

    fn apply_crosstalk_error(
        &self,
        state: &mut Vec<Complex64>,
        qubit: QubitId,
        strength: f64,
    ) -> QuantRS2Result<()> {
        // Apply small random rotation due to crosstalk
        self.apply_rotation_error(state, qubit, strength)
    }
}

/// Create a noise model from calibration with custom parameters
pub struct NoiseModelBuilder {
    calibration: DeviceCalibration,
    coherent_factor: f64,
    thermal_factor: f64,
    crosstalk_factor: f64,
    readout_factor: f64,
}

impl NoiseModelBuilder {
    /// Create builder from calibration
    pub fn from_calibration(calibration: DeviceCalibration) -> Self {
        Self {
            calibration,
            coherent_factor: 1.0,
            thermal_factor: 1.0,
            crosstalk_factor: 1.0,
            readout_factor: 1.0,
        }
    }

    /// Scale coherent errors
    pub fn coherent_factor(mut self, factor: f64) -> Self {
        self.coherent_factor = factor;
        self
    }

    /// Scale thermal noise
    pub fn thermal_factor(mut self, factor: f64) -> Self {
        self.thermal_factor = factor;
        self
    }

    /// Scale crosstalk
    pub fn crosstalk_factor(mut self, factor: f64) -> Self {
        self.crosstalk_factor = factor;
        self
    }

    /// Scale readout errors
    pub fn readout_factor(mut self, factor: f64) -> Self {
        self.readout_factor = factor;
        self
    }

    /// Build the noise model
    pub fn build(self) -> CalibrationNoiseModel {
        let mut model = CalibrationNoiseModel::from_calibration(&self.calibration);

        // Apply scaling factors
        for noise in model.gate_noise.values_mut() {
            noise.coherent_error *= self.coherent_factor;
        }

        for noise in model.qubit_noise.values_mut() {
            noise.thermal_population *= self.thermal_factor;
        }

        for row in model.crosstalk.crosstalk_matrix.iter_mut() {
            for val in row.iter_mut() {
                *val *= self.crosstalk_factor;
            }
        }

        for readout in model.readout_noise.values_mut() {
            // Scale off-diagonal elements (errors)
            readout.assignment_matrix[0][1] *= self.readout_factor;
            readout.assignment_matrix[1][0] *= self.readout_factor;
            // Adjust diagonal to maintain normalization
            readout.assignment_matrix[0][0] = 1.0 - readout.assignment_matrix[0][1];
            readout.assignment_matrix[1][1] = 1.0 - readout.assignment_matrix[1][0];
        }

        model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::create_ideal_calibration;

    #[test]
    fn test_noise_model_from_calibration() {
        let cal = create_ideal_calibration("test".to_string(), 5);
        let noise_model = CalibrationNoiseModel::from_calibration(&cal);

        assert_eq!(noise_model.device_id, "test");
        assert_eq!(noise_model.qubit_noise.len(), 5);
        assert!(noise_model.gate_noise.contains_key("X"));
        assert!(noise_model.gate_noise.contains_key("CNOT"));
    }

    #[test]
    fn test_noise_model_builder() {
        let cal = create_ideal_calibration("test".to_string(), 3);
        let noise_model = NoiseModelBuilder::from_calibration(cal)
            .coherent_factor(0.5)
            .thermal_factor(2.0)
            .crosstalk_factor(0.1)
            .readout_factor(0.5)
            .build();

        // Check that factors were applied
        let x_noise = noise_model.gate_noise.get("X").unwrap();
        assert!(x_noise.coherent_error < 0.001); // Should be reduced by factor
    }
}
