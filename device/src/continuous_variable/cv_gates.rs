//! Continuous variable quantum gates
//!
//! This module implements the standard gate set for continuous variable quantum computing,
//! including Gaussian operations and some non-Gaussian operations.

use super::{Complex, GaussianState};
use crate::{DeviceError, DeviceResult};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Types of CV gates
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CVGateType {
    /// Displacement gate D(α)
    Displacement { amplitude: Complex },
    /// Squeezing gate S(r, φ)
    Squeezing { parameter: f64, phase: f64 },
    /// Two-mode squeezing gate S₂(r, φ)
    TwoModeSqueezing { parameter: f64, phase: f64 },
    /// Beamsplitter gate BS(θ, φ)
    Beamsplitter { transmittance: f64, phase: f64 },
    /// Phase rotation gate R(φ)
    PhaseRotation { phase: f64 },
    /// Controlled displacement gate CD(α)
    ControlledDisplacement { amplitude: Complex },
    /// Controlled phase gate CP(s)
    ControlledPhase { parameter: f64 },
    /// Cross-Kerr gate CK(κ)
    CrossKerr { parameter: f64 },
    /// Cubic phase gate V(γ)
    CubicPhase { parameter: f64 },
    /// Position measurement M_x(s)
    PositionMeasurement { result: f64 },
    /// Momentum measurement M_p(s)
    MomentumMeasurement { result: f64 },
}

/// CV gate parameters for parameterized operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVGateParams {
    /// Real parameters (amplitudes, phases, etc.)
    pub real_params: Vec<f64>,
    /// Complex parameters (displacements, etc.)
    pub complex_params: Vec<Complex>,
    /// Target modes
    pub target_modes: Vec<usize>,
    /// Control modes (if applicable)
    pub control_modes: Vec<usize>,
}

/// Sequence of CV gates forming a quantum program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVGateSequence {
    /// Gates in the sequence
    pub gates: Vec<(CVGateType, CVGateParams)>,
    /// Total number of modes required
    pub num_modes: usize,
}

impl CVGateSequence {
    /// Create a new empty gate sequence
    pub fn new(num_modes: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_modes,
        }
    }

    /// Add a displacement gate
    pub fn displacement(&mut self, mode: usize, amplitude: Complex) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds sequence capacity",
                mode
            )));
        }

        self.gates.push((
            CVGateType::Displacement { amplitude },
            CVGateParams {
                real_params: vec![amplitude.real, amplitude.imag],
                complex_params: vec![amplitude],
                target_modes: vec![mode],
                control_modes: vec![],
            },
        ));

        Ok(())
    }

    /// Add a squeezing gate
    pub fn squeezing(&mut self, mode: usize, parameter: f64, phase: f64) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds sequence capacity",
                mode
            )));
        }

        self.gates.push((
            CVGateType::Squeezing { parameter, phase },
            CVGateParams {
                real_params: vec![parameter, phase],
                complex_params: vec![],
                target_modes: vec![mode],
                control_modes: vec![],
            },
        ));

        Ok(())
    }

    /// Add a two-mode squeezing gate
    pub fn two_mode_squeezing(
        &mut self,
        mode1: usize,
        mode2: usize,
        parameter: f64,
        phase: f64,
    ) -> DeviceResult<()> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(DeviceError::InvalidInput(
                "One or both modes exceed sequence capacity".to_string(),
            ));
        }

        self.gates.push((
            CVGateType::TwoModeSqueezing { parameter, phase },
            CVGateParams {
                real_params: vec![parameter, phase],
                complex_params: vec![],
                target_modes: vec![mode1, mode2],
                control_modes: vec![],
            },
        ));

        Ok(())
    }

    /// Add a beamsplitter gate
    pub fn beamsplitter(
        &mut self,
        mode1: usize,
        mode2: usize,
        transmittance: f64,
        phase: f64,
    ) -> DeviceResult<()> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(DeviceError::InvalidInput(
                "One or both modes exceed sequence capacity".to_string(),
            ));
        }

        if transmittance < 0.0 || transmittance > 1.0 {
            return Err(DeviceError::InvalidInput(
                "Transmittance must be between 0 and 1".to_string(),
            ));
        }

        self.gates.push((
            CVGateType::Beamsplitter {
                transmittance,
                phase,
            },
            CVGateParams {
                real_params: vec![transmittance, phase],
                complex_params: vec![],
                target_modes: vec![mode1, mode2],
                control_modes: vec![],
            },
        ));

        Ok(())
    }

    /// Add a phase rotation gate
    pub fn phase_rotation(&mut self, mode: usize, phase: f64) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds sequence capacity",
                mode
            )));
        }

        self.gates.push((
            CVGateType::PhaseRotation { phase },
            CVGateParams {
                real_params: vec![phase],
                complex_params: vec![],
                target_modes: vec![mode],
                control_modes: vec![],
            },
        ));

        Ok(())
    }

    /// Add a controlled displacement gate
    pub fn controlled_displacement(
        &mut self,
        control_mode: usize,
        target_mode: usize,
        amplitude: Complex,
    ) -> DeviceResult<()> {
        if control_mode >= self.num_modes || target_mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(
                "Control or target mode exceeds sequence capacity".to_string(),
            ));
        }

        self.gates.push((
            CVGateType::ControlledDisplacement { amplitude },
            CVGateParams {
                real_params: vec![amplitude.real, amplitude.imag],
                complex_params: vec![amplitude],
                target_modes: vec![target_mode],
                control_modes: vec![control_mode],
            },
        ));

        Ok(())
    }

    /// Add a controlled phase gate
    pub fn controlled_phase(
        &mut self,
        control_mode: usize,
        target_mode: usize,
        parameter: f64,
    ) -> DeviceResult<()> {
        if control_mode >= self.num_modes || target_mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(
                "Control or target mode exceeds sequence capacity".to_string(),
            ));
        }

        self.gates.push((
            CVGateType::ControlledPhase { parameter },
            CVGateParams {
                real_params: vec![parameter],
                complex_params: vec![],
                target_modes: vec![target_mode],
                control_modes: vec![control_mode],
            },
        ));

        Ok(())
    }

    /// Add a cross-Kerr gate
    pub fn cross_kerr(&mut self, mode1: usize, mode2: usize, parameter: f64) -> DeviceResult<()> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(DeviceError::InvalidInput(
                "One or both modes exceed sequence capacity".to_string(),
            ));
        }

        self.gates.push((
            CVGateType::CrossKerr { parameter },
            CVGateParams {
                real_params: vec![parameter],
                complex_params: vec![],
                target_modes: vec![mode1, mode2],
                control_modes: vec![],
            },
        ));

        Ok(())
    }

    /// Add a cubic phase gate (non-Gaussian)
    pub fn cubic_phase(&mut self, mode: usize, parameter: f64) -> DeviceResult<()> {
        if mode >= self.num_modes {
            return Err(DeviceError::InvalidInput(format!(
                "Mode {} exceeds sequence capacity",
                mode
            )));
        }

        self.gates.push((
            CVGateType::CubicPhase { parameter },
            CVGateParams {
                real_params: vec![parameter],
                complex_params: vec![],
                target_modes: vec![mode],
                control_modes: vec![],
            },
        ));

        Ok(())
    }

    /// Get the number of gates in the sequence
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    /// Get the depth of the sequence (maximum number of gates on any mode)
    pub fn depth(&self) -> usize {
        let mut mode_depths = vec![0; self.num_modes];

        for (_, params) in &self.gates {
            for &mode in &params.target_modes {
                mode_depths[mode] += 1;
            }
            for &mode in &params.control_modes {
                mode_depths[mode] += 1;
            }
        }

        *mode_depths.iter().max().unwrap_or(&0)
    }

    /// Check if the sequence contains only Gaussian operations
    pub fn is_gaussian(&self) -> bool {
        for (gate_type, _) in &self.gates {
            match gate_type {
                CVGateType::CubicPhase { .. } => return false,
                CVGateType::PositionMeasurement { .. } => return false,
                CVGateType::MomentumMeasurement { .. } => return false,
                _ => continue,
            }
        }
        true
    }

    /// Execute the gate sequence on a Gaussian state
    pub fn execute_on_state(&self, state: &mut GaussianState) -> DeviceResult<()> {
        if state.num_modes != self.num_modes {
            return Err(DeviceError::InvalidInput(
                "State mode count doesn't match sequence requirements".to_string(),
            ));
        }

        for (gate_type, params) in &self.gates {
            self.execute_single_gate(gate_type, params, state)?;
        }

        Ok(())
    }

    /// Execute a single gate on the state
    fn execute_single_gate(
        &self,
        gate_type: &CVGateType,
        params: &CVGateParams,
        state: &mut GaussianState,
    ) -> DeviceResult<()> {
        match gate_type {
            CVGateType::Displacement { amplitude } => {
                if params.target_modes.len() != 1 {
                    return Err(DeviceError::InvalidInput(
                        "Displacement gate requires exactly one target mode".to_string(),
                    ));
                }
                state.apply_displacement(params.target_modes[0], *amplitude)?;
            }

            CVGateType::Squeezing { parameter, phase } => {
                if params.target_modes.len() != 1 {
                    return Err(DeviceError::InvalidInput(
                        "Squeezing gate requires exactly one target mode".to_string(),
                    ));
                }
                state.apply_squeezing(params.target_modes[0], *parameter, *phase)?;
            }

            CVGateType::TwoModeSqueezing { parameter, phase } => {
                if params.target_modes.len() != 2 {
                    return Err(DeviceError::InvalidInput(
                        "Two-mode squeezing gate requires exactly two target modes".to_string(),
                    ));
                }
                state.apply_two_mode_squeezing(
                    params.target_modes[0],
                    params.target_modes[1],
                    *parameter,
                    *phase,
                )?;
            }

            CVGateType::Beamsplitter {
                transmittance,
                phase,
            } => {
                if params.target_modes.len() != 2 {
                    return Err(DeviceError::InvalidInput(
                        "Beamsplitter gate requires exactly two target modes".to_string(),
                    ));
                }
                state.apply_beamsplitter(
                    params.target_modes[0],
                    params.target_modes[1],
                    *transmittance,
                    *phase,
                )?;
            }

            CVGateType::PhaseRotation { phase } => {
                if params.target_modes.len() != 1 {
                    return Err(DeviceError::InvalidInput(
                        "Phase rotation gate requires exactly one target mode".to_string(),
                    ));
                }
                state.apply_phase_rotation(params.target_modes[0], *phase)?;
            }

            CVGateType::ControlledDisplacement { amplitude } => {
                if params.control_modes.len() != 1 || params.target_modes.len() != 1 {
                    return Err(DeviceError::InvalidInput(
                        "Controlled displacement requires one control and one target mode"
                            .to_string(),
                    ));
                }
                self.apply_controlled_displacement(
                    params.control_modes[0],
                    params.target_modes[0],
                    *amplitude,
                    state,
                )?;
            }

            CVGateType::ControlledPhase { parameter } => {
                if params.control_modes.len() != 1 || params.target_modes.len() != 1 {
                    return Err(DeviceError::InvalidInput(
                        "Controlled phase requires one control and one target mode".to_string(),
                    ));
                }
                self.apply_controlled_phase(
                    params.control_modes[0],
                    params.target_modes[0],
                    *parameter,
                    state,
                )?;
            }

            CVGateType::CrossKerr { parameter } => {
                if params.target_modes.len() != 2 {
                    return Err(DeviceError::InvalidInput(
                        "Cross-Kerr gate requires exactly two target modes".to_string(),
                    ));
                }
                self.apply_cross_kerr(
                    params.target_modes[0],
                    params.target_modes[1],
                    *parameter,
                    state,
                )?;
            }

            CVGateType::CubicPhase { parameter } => {
                return Err(DeviceError::UnsupportedOperation(
                    "Cubic phase gate is non-Gaussian and not supported for Gaussian states"
                        .to_string(),
                ));
            }

            CVGateType::PositionMeasurement { .. } | CVGateType::MomentumMeasurement { .. } => {
                return Err(DeviceError::UnsupportedOperation(
                    "Measurements should be performed separately from gate sequences".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Apply controlled displacement (simplified implementation)
    fn apply_controlled_displacement(
        &self,
        control_mode: usize,
        target_mode: usize,
        amplitude: Complex,
        state: &mut GaussianState,
    ) -> DeviceResult<()> {
        // Simplified implementation - would need full multimode transformation
        // For now, apply displacement scaled by control mode amplitude
        let control_amplitude = Complex::new(
            state.mean_vector[2 * control_mode] / (2.0_f64).sqrt(),
            state.mean_vector[2 * control_mode + 1] / (2.0_f64).sqrt(),
        );

        let scaled_amplitude = amplitude * control_amplitude.magnitude();
        state.apply_displacement(target_mode, scaled_amplitude)?;

        Ok(())
    }

    /// Apply controlled phase (simplified implementation)
    fn apply_controlled_phase(
        &self,
        control_mode: usize,
        target_mode: usize,
        parameter: f64,
        state: &mut GaussianState,
    ) -> DeviceResult<()> {
        // Simplified implementation
        let control_photon_number = self.estimate_photon_number(control_mode, state);
        let phase = parameter * control_photon_number;
        state.apply_phase_rotation(target_mode, phase)?;

        Ok(())
    }

    /// Apply cross-Kerr interaction
    fn apply_cross_kerr(
        &self,
        mode1: usize,
        mode2: usize,
        parameter: f64,
        state: &mut GaussianState,
    ) -> DeviceResult<()> {
        // Simplified cross-Kerr implementation
        // Cross-Kerr induces phase shifts proportional to photon numbers
        let n1 = self.estimate_photon_number(mode1, state);
        let n2 = self.estimate_photon_number(mode2, state);

        state.apply_phase_rotation(mode1, parameter * n2)?;
        state.apply_phase_rotation(mode2, parameter * n1)?;

        Ok(())
    }

    /// Estimate photon number for a mode (for simplified implementations)
    fn estimate_photon_number(&self, mode: usize, state: &GaussianState) -> f64 {
        let mean_x = state.mean_vector[2 * mode];
        let mean_p = state.mean_vector[2 * mode + 1];
        let var_x = state.covariance_matrix[2 * mode][2 * mode];
        let var_p = state.covariance_matrix[2 * mode + 1][2 * mode + 1];

        // Average photon number approximation
        0.5 * ((mean_x.powi(2) + mean_p.powi(2)) / 2.0 + (var_x + var_p) - 1.0)
    }
}

/// Common CV gate implementations
pub struct CVGateLibrary;

impl CVGateLibrary {
    /// Create a displacement gate
    pub fn displacement(amplitude: Complex) -> (CVGateType, CVGateParams) {
        (
            CVGateType::Displacement { amplitude },
            CVGateParams {
                real_params: vec![amplitude.real, amplitude.imag],
                complex_params: vec![amplitude],
                target_modes: vec![],
                control_modes: vec![],
            },
        )
    }

    /// Create a squeezing gate  
    pub fn squeezing(parameter: f64, phase: f64) -> (CVGateType, CVGateParams) {
        (
            CVGateType::Squeezing { parameter, phase },
            CVGateParams {
                real_params: vec![parameter, phase],
                complex_params: vec![],
                target_modes: vec![],
                control_modes: vec![],
            },
        )
    }

    /// Create a 50:50 beamsplitter
    pub fn balanced_beamsplitter() -> (CVGateType, CVGateParams) {
        (
            CVGateType::Beamsplitter {
                transmittance: 0.5,
                phase: 0.0,
            },
            CVGateParams {
                real_params: vec![0.5, 0.0],
                complex_params: vec![],
                target_modes: vec![],
                control_modes: vec![],
            },
        )
    }

    /// Create a Hadamard-like operation for CV (Fourier transform)
    pub fn fourier_transform() -> CVGateSequence {
        let mut sequence = CVGateSequence::new(1);
        sequence.phase_rotation(0, PI / 2.0).unwrap();
        sequence
    }

    /// Create a CNOT-like operation for CV
    pub fn cv_cnot() -> CVGateSequence {
        let mut sequence = CVGateSequence::new(2);
        // Simplified CV CNOT using beamsplitter and phase rotations
        sequence.beamsplitter(0, 1, 0.5, 0.0).unwrap();
        sequence.phase_rotation(1, PI).unwrap();
        sequence.beamsplitter(0, 1, 0.5, 0.0).unwrap();
        sequence
    }

    /// Create an EPR pair generation sequence
    pub fn epr_pair_generation(squeezing_param: f64) -> CVGateSequence {
        let mut sequence = CVGateSequence::new(2);
        sequence
            .two_mode_squeezing(0, 1, squeezing_param, 0.0)
            .unwrap();
        sequence
    }

    /// Create a GKP (Gottesman-Kitaev-Preskill) state preparation sequence
    pub fn gkp_state_preparation() -> CVGateSequence {
        let mut sequence = CVGateSequence::new(1);
        // Simplified GKP preparation using multiple squeezing operations
        for i in 0..10 {
            let phase = 2.0 * PI * i as f64 / 10.0;
            sequence.squeezing(0, 0.5, phase).unwrap();
        }
        sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_sequence_creation() {
        let mut sequence = CVGateSequence::new(3);
        assert_eq!(sequence.num_modes, 3);
        assert_eq!(sequence.gate_count(), 0);
    }

    #[test]
    fn test_displacement_gate_addition() {
        let mut sequence = CVGateSequence::new(2);
        let amplitude = Complex::new(1.0, 0.5);

        sequence.displacement(0, amplitude).unwrap();
        assert_eq!(sequence.gate_count(), 1);

        match &sequence.gates[0].0 {
            CVGateType::Displacement { amplitude: a } => {
                assert_eq!(*a, amplitude);
            }
            _ => panic!("Expected displacement gate"),
        }
    }

    #[test]
    fn test_beamsplitter_gate_addition() {
        let mut sequence = CVGateSequence::new(2);

        sequence.beamsplitter(0, 1, 0.7, PI / 4.0).unwrap();
        assert_eq!(sequence.gate_count(), 1);

        match &sequence.gates[0].0 {
            CVGateType::Beamsplitter {
                transmittance,
                phase,
            } => {
                assert_eq!(*transmittance, 0.7);
                assert_eq!(*phase, PI / 4.0);
            }
            _ => panic!("Expected beamsplitter gate"),
        }
    }

    #[test]
    fn test_gaussian_property() {
        let mut sequence = CVGateSequence::new(2);

        sequence.displacement(0, Complex::new(1.0, 0.0)).unwrap();
        sequence.squeezing(1, 0.5, 0.0).unwrap();
        sequence.beamsplitter(0, 1, 0.5, 0.0).unwrap();

        assert!(sequence.is_gaussian());

        sequence.cubic_phase(0, 0.1).unwrap();
        assert!(!sequence.is_gaussian());
    }

    #[test]
    fn test_sequence_depth_calculation() {
        let mut sequence = CVGateSequence::new(3);

        sequence.displacement(0, Complex::new(1.0, 0.0)).unwrap();
        sequence.displacement(0, Complex::new(0.5, 0.0)).unwrap();
        sequence.squeezing(1, 0.5, 0.0).unwrap();
        sequence.beamsplitter(0, 2, 0.5, 0.0).unwrap();

        assert_eq!(sequence.depth(), 3); // Mode 0 has 3 operations
    }

    #[test]
    fn test_gate_execution_on_state() {
        let mut sequence = CVGateSequence::new(2);
        sequence.displacement(0, Complex::new(1.0, 0.0)).unwrap();
        sequence.squeezing(1, 0.5, 0.0).unwrap();

        let mut state = GaussianState::vacuum_state(2);
        sequence.execute_on_state(&mut state).unwrap();

        // Check that the displacement was applied
        assert!(state.mean_vector[0] > 0.0);

        // Check that squeezing was applied
        assert!(state.covariance_matrix[2][2] < 0.5);
    }

    #[test]
    fn test_epr_pair_generation() {
        let sequence = CVGateLibrary::epr_pair_generation(1.0);
        assert_eq!(sequence.gate_count(), 1);

        let mut state = GaussianState::vacuum_state(2);
        sequence.execute_on_state(&mut state).unwrap();

        // EPR state should have correlations between modes
        let entanglement = state.calculate_entanglement_measures();
        assert!(entanglement.epr_correlation > 0.0);
    }

    #[test]
    fn test_balanced_beamsplitter() {
        let (gate_type, params) = CVGateLibrary::balanced_beamsplitter();

        match gate_type {
            CVGateType::Beamsplitter {
                transmittance,
                phase,
            } => {
                assert_eq!(transmittance, 0.5);
                assert_eq!(phase, 0.0);
            }
            _ => panic!("Expected balanced beamsplitter"),
        }
    }

    #[test]
    fn test_invalid_mode_error() {
        let mut sequence = CVGateSequence::new(2);

        let result = sequence.displacement(3, Complex::new(1.0, 0.0));
        assert!(result.is_err());

        let result = sequence.beamsplitter(0, 3, 0.5, 0.0);
        assert!(result.is_err());
    }
}
