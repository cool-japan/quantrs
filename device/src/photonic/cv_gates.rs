//! Continuous Variable Gate Operations
//!
//! This module implements specific gate operations for continuous variable quantum computing,
//! including displacement, squeezing, beamsplitters, and non-linear operations.

use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, SQRT_2};
use thiserror::Error;

use super::continuous_variable::{CVError, CVResult, Complex, GaussianState};

/// CV Gate operation errors
#[derive(Error, Debug)]
pub enum CVGateError {
    #[error("Invalid gate parameter: {0}")]
    InvalidParameter(String),
    #[error("Gate not supported: {0}")]
    UnsupportedGate(String),
    #[error("Mode index out of bounds: {0}")]
    ModeOutOfBounds(usize),
}

/// Types of CV gates
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CVGateType {
    /// Displacement gate D(α)
    Displacement,
    /// Single-mode squeezing S(r,φ)
    Squeezing,
    /// Two-mode squeezing S₂(r,φ)
    TwoModeSqueezing,
    /// Beamsplitter BS(θ,φ)
    Beamsplitter,
    /// Phase rotation R(φ)
    PhaseRotation,
    /// Kerr interaction K(κ)
    Kerr,
    /// Cross-Kerr interaction CK(κ)
    CrossKerr,
    /// Cubic phase gate V(γ)
    CubicPhase,
    /// Position measurement
    PositionMeasurement,
    /// Momentum measurement
    MomentumMeasurement,
    /// Homodyne measurement
    HomodyneMeasurement,
    /// Heterodyne measurement
    HeterodyneMeasurement,
}

/// CV gate parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVGateParams {
    /// Gate type
    pub gate_type: CVGateType,
    /// Target mode(s)
    pub modes: Vec<usize>,
    /// Gate parameters (interpretation depends on gate type)
    pub params: Vec<f64>,
    /// Complex parameters (for displacement, etc.)
    pub complex_params: Vec<Complex>,
    /// Gate duration (for time-dependent gates)
    pub duration: Option<f64>,
    /// Gate fidelity
    pub fidelity: Option<f64>,
}

/// Displacement gate implementation
pub struct DisplacementGate {
    /// Displacement amplitude
    pub alpha: Complex,
    /// Target mode
    pub mode: usize,
}

impl DisplacementGate {
    pub const fn new(alpha: Complex, mode: usize) -> Self {
        Self { alpha, mode }
    }

    /// Apply displacement to state
    pub fn apply(&self, state: &mut GaussianState) -> CVResult<()> {
        state.displace(self.alpha, self.mode)
    }

    /// Get gate matrix representation (for smaller Fock spaces)
    pub fn matrix(&self, cutoff: usize) -> Vec<Vec<Complex>> {
        let mut matrix = vec![vec![Complex::new(0.0, 0.0); cutoff]; cutoff];

        for n in 0..cutoff {
            for m in 0..cutoff {
                // Displacement matrix element <m|D(α)|n>
                let element = self.displacement_matrix_element(m, n, cutoff);
                matrix[m][n] = element;
            }
        }

        matrix
    }

    const fn displacement_matrix_element(&self, m: usize, n: usize, cutoff: usize) -> Complex {
        // Simplified calculation - full implementation would use Laguerre polynomials
        if m == n {
            Complex::new(1.0, 0.0)
        } else {
            Complex::new(0.0, 0.0)
        }
    }

    /// Get inverse displacement
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            alpha: Complex::new(-self.alpha.real, -self.alpha.imag),
            mode: self.mode,
        }
    }
}

/// Squeezing gate implementation
pub struct SqueezingGate {
    /// Squeezing parameter
    pub r: f64,
    /// Squeezing angle
    pub phi: f64,
    /// Target mode
    pub mode: usize,
}

impl SqueezingGate {
    pub fn new(r: f64, phi: f64, mode: usize) -> Result<Self, CVGateError> {
        if r.abs() > 20.0 {
            return Err(CVGateError::InvalidParameter(
                "Squeezing parameter too large".to_string(),
            ));
        }

        Ok(Self { r, phi, mode })
    }

    /// Apply squeezing to state
    pub fn apply(&self, state: &mut GaussianState) -> CVResult<()> {
        state.squeeze(self.r, self.phi, self.mode)
    }

    /// Get squeezing in dB
    pub fn squeezing_db(&self) -> f64 {
        10.0 * self.r.abs() / 2.0_f64.log(10.0_f64)
    }

    /// Create amplitude squeezing (φ=0)
    pub fn amplitude_squeezing(r: f64, mode: usize) -> Result<Self, CVGateError> {
        Self::new(r, 0.0, mode)
    }

    /// Create phase squeezing (φ=π/2)
    pub fn phase_squeezing(r: f64, mode: usize) -> Result<Self, CVGateError> {
        Self::new(r, PI / 2.0, mode)
    }
}

/// Two-mode squeezing gate implementation
pub struct TwoModeSqueezingGate {
    /// Squeezing parameter
    pub r: f64,
    /// Squeezing phase
    pub phi: f64,
    /// First mode
    pub mode1: usize,
    /// Second mode
    pub mode2: usize,
}

impl TwoModeSqueezingGate {
    pub fn new(r: f64, phi: f64, mode1: usize, mode2: usize) -> Result<Self, CVGateError> {
        if mode1 == mode2 {
            return Err(CVGateError::InvalidParameter(
                "Two-mode squeezing requires different modes".to_string(),
            ));
        }

        if r.abs() > 20.0 {
            return Err(CVGateError::InvalidParameter(
                "Squeezing parameter too large".to_string(),
            ));
        }

        Ok(Self {
            r,
            phi,
            mode1,
            mode2,
        })
    }

    /// Apply two-mode squeezing to state
    pub fn apply(&self, state: &mut GaussianState) -> CVResult<()> {
        state.two_mode_squeeze(self.r, self.phi, self.mode1, self.mode2)
    }

    /// Get entanglement strength
    pub fn entanglement_strength(&self) -> f64 {
        // Logarithmic negativity approximation
        2.0 * self.r.abs()
    }
}

/// Beamsplitter gate implementation
pub struct BeamsplitterGate {
    /// Transmission angle
    pub theta: f64,
    /// Phase
    pub phi: f64,
    /// First input mode
    pub mode1: usize,
    /// Second input mode
    pub mode2: usize,
}

impl BeamsplitterGate {
    pub fn new(theta: f64, phi: f64, mode1: usize, mode2: usize) -> Result<Self, CVGateError> {
        if mode1 == mode2 {
            return Err(CVGateError::InvalidParameter(
                "Beamsplitter requires different modes".to_string(),
            ));
        }

        Ok(Self {
            theta,
            phi,
            mode1,
            mode2,
        })
    }

    /// Apply beamsplitter to state
    pub fn apply(&self, state: &mut GaussianState) -> CVResult<()> {
        state.beamsplitter(self.theta, self.phi, self.mode1, self.mode2)
    }

    /// Create 50:50 beamsplitter
    pub fn fifty_fifty(mode1: usize, mode2: usize) -> Result<Self, CVGateError> {
        Self::new(PI / 4.0, 0.0, mode1, mode2)
    }

    /// Get transmission coefficient
    pub fn transmission(&self) -> f64 {
        self.theta.cos().powi(2)
    }

    /// Get reflection coefficient
    pub fn reflection(&self) -> f64 {
        self.theta.sin().powi(2)
    }

    /// Create variable beamsplitter with transmission T
    pub fn with_transmission(
        transmission: f64,
        phi: f64,
        mode1: usize,
        mode2: usize,
    ) -> Result<Self, CVGateError> {
        if !(0.0..=1.0).contains(&transmission) {
            return Err(CVGateError::InvalidParameter(
                "Transmission must be between 0 and 1".to_string(),
            ));
        }

        let theta = transmission.sqrt().acos();
        Self::new(theta, phi, mode1, mode2)
    }
}

/// Phase rotation gate implementation
pub struct PhaseRotationGate {
    /// Rotation angle
    pub phi: f64,
    /// Target mode
    pub mode: usize,
}

impl PhaseRotationGate {
    pub const fn new(phi: f64, mode: usize) -> Self {
        Self { phi, mode }
    }

    /// Apply phase rotation to state
    pub fn apply(&self, state: &mut GaussianState) -> CVResult<()> {
        state.phase_rotation(self.phi, self.mode)
    }

    /// Normalize phase to [0, 2π)
    pub fn normalized_phase(&self) -> f64 {
        self.phi.rem_euclid(2.0 * PI)
    }
}

/// Kerr gate implementation (non-linear)
pub struct KerrGate {
    /// Kerr parameter
    pub kappa: f64,
    /// Target mode
    pub mode: usize,
    /// Interaction time
    pub time: f64,
}

impl KerrGate {
    pub const fn new(kappa: f64, mode: usize, time: f64) -> Self {
        Self { kappa, mode, time }
    }

    /// Apply Kerr interaction (approximation for Gaussian states)
    pub fn apply(&self, state: &mut GaussianState) -> CVResult<()> {
        // Kerr interaction is non-Gaussian, so this is an approximation
        // In practice, this would require non-Gaussian state representation

        // Apply phase shift proportional to photon number
        let n_avg = state.average_photon_number(self.mode)?;
        let phase_shift = self.kappa * n_avg * self.time;

        state.phase_rotation(phase_shift, self.mode)
    }

    /// Get effective phase shift
    pub fn phase_shift(&self, photon_number: f64) -> f64 {
        self.kappa * photon_number * self.time
    }
}

/// Cross-Kerr gate implementation
pub struct CrossKerrGate {
    /// Cross-Kerr parameter
    pub kappa: f64,
    /// First mode
    pub mode1: usize,
    /// Second mode
    pub mode2: usize,
    /// Interaction time
    pub time: f64,
}

impl CrossKerrGate {
    pub fn new(kappa: f64, mode1: usize, mode2: usize, time: f64) -> Result<Self, CVGateError> {
        if mode1 == mode2 {
            return Err(CVGateError::InvalidParameter(
                "Cross-Kerr requires different modes".to_string(),
            ));
        }

        Ok(Self {
            kappa,
            mode1,
            mode2,
            time,
        })
    }

    /// Apply Cross-Kerr interaction (approximation)
    pub fn apply(&self, state: &mut GaussianState) -> CVResult<()> {
        // Cross-Kerr is non-Gaussian, approximation for Gaussian states
        let n1 = state.average_photon_number(self.mode1)?;
        let n2 = state.average_photon_number(self.mode2)?;

        // Apply conditional phase shifts
        let phase_shift1 = self.kappa * n2 * self.time;
        let phase_shift2 = self.kappa * n1 * self.time;

        state.phase_rotation(phase_shift1, self.mode1)?;
        state.phase_rotation(phase_shift2, self.mode2)?;

        Ok(())
    }

    /// Get conditional phase shift
    pub fn conditional_phase_shift(&self, control_photons: f64) -> f64 {
        self.kappa * control_photons * self.time
    }
}

/// Cubic phase gate implementation (non-Gaussian)
pub struct CubicPhaseGate {
    /// Cubic parameter
    pub gamma: f64,
    /// Target mode
    pub mode: usize,
}

impl CubicPhaseGate {
    pub const fn new(gamma: f64, mode: usize) -> Self {
        Self { gamma, mode }
    }

    /// Apply cubic phase (this breaks Gaussianity)
    pub fn apply(&self, _state: &mut GaussianState) -> CVResult<()> {
        // Cubic phase gate makes the state non-Gaussian
        // This would require a more general state representation
        Err(CVError::IncompatibleOperation(
            "Cubic phase gate is non-Gaussian and not supported for Gaussian states".to_string(),
        ))
    }
}

/// Gate sequence builder for CV operations
pub struct CVGateSequence {
    /// Sequence of gates
    pub gates: Vec<CVGateParams>,
    /// Number of modes
    pub num_modes: usize,
}

impl CVGateSequence {
    pub const fn new(num_modes: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_modes,
        }
    }

    /// Add displacement gate
    pub fn displacement(&mut self, alpha: Complex, mode: usize) -> Result<&mut Self, CVGateError> {
        if mode >= self.num_modes {
            return Err(CVGateError::ModeOutOfBounds(mode));
        }

        self.gates.push(CVGateParams {
            gate_type: CVGateType::Displacement,
            modes: vec![mode],
            params: vec![],
            complex_params: vec![alpha],
            duration: None,
            fidelity: Some(0.99),
        });

        Ok(self)
    }

    /// Add squeezing gate
    pub fn squeezing(&mut self, r: f64, phi: f64, mode: usize) -> Result<&mut Self, CVGateError> {
        if mode >= self.num_modes {
            return Err(CVGateError::ModeOutOfBounds(mode));
        }

        self.gates.push(CVGateParams {
            gate_type: CVGateType::Squeezing,
            modes: vec![mode],
            params: vec![r, phi],
            complex_params: vec![],
            duration: None,
            fidelity: Some(0.98),
        });

        Ok(self)
    }

    /// Add two-mode squeezing gate
    pub fn two_mode_squeezing(
        &mut self,
        r: f64,
        phi: f64,
        mode1: usize,
        mode2: usize,
    ) -> Result<&mut Self, CVGateError> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(CVGateError::ModeOutOfBounds(mode1.max(mode2)));
        }

        self.gates.push(CVGateParams {
            gate_type: CVGateType::TwoModeSqueezing,
            modes: vec![mode1, mode2],
            params: vec![r, phi],
            complex_params: vec![],
            duration: None,
            fidelity: Some(0.97),
        });

        Ok(self)
    }

    /// Add beamsplitter gate
    pub fn beamsplitter(
        &mut self,
        theta: f64,
        phi: f64,
        mode1: usize,
        mode2: usize,
    ) -> Result<&mut Self, CVGateError> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(CVGateError::ModeOutOfBounds(mode1.max(mode2)));
        }

        self.gates.push(CVGateParams {
            gate_type: CVGateType::Beamsplitter,
            modes: vec![mode1, mode2],
            params: vec![theta, phi],
            complex_params: vec![],
            duration: None,
            fidelity: Some(0.995),
        });

        Ok(self)
    }

    /// Add phase rotation gate
    pub fn phase_rotation(&mut self, phi: f64, mode: usize) -> Result<&mut Self, CVGateError> {
        if mode >= self.num_modes {
            return Err(CVGateError::ModeOutOfBounds(mode));
        }

        self.gates.push(CVGateParams {
            gate_type: CVGateType::PhaseRotation,
            modes: vec![mode],
            params: vec![phi],
            complex_params: vec![],
            duration: None,
            fidelity: Some(0.999),
        });

        Ok(self)
    }

    /// Apply entire sequence to a state
    pub fn apply(&self, state: &mut GaussianState) -> CVResult<()> {
        for gate in &self.gates {
            match gate.gate_type {
                CVGateType::Displacement => {
                    if let (Some(alpha), Some(&mode)) =
                        (gate.complex_params.first(), gate.modes.first())
                    {
                        state.displace(*alpha, mode)?;
                    }
                }
                CVGateType::Squeezing => {
                    if let (Some(&r), Some(&phi), Some(&mode)) =
                        (gate.params.first(), gate.params.get(1), gate.modes.first())
                    {
                        state.squeeze(r, phi, mode)?;
                    }
                }
                CVGateType::TwoModeSqueezing => {
                    if let (Some(&r), Some(&phi), Some(&mode1), Some(&mode2)) = (
                        gate.params.first(),
                        gate.params.get(1),
                        gate.modes.first(),
                        gate.modes.get(1),
                    ) {
                        state.two_mode_squeeze(r, phi, mode1, mode2)?;
                    }
                }
                CVGateType::Beamsplitter => {
                    if let (Some(&theta), Some(&phi), Some(&mode1), Some(&mode2)) = (
                        gate.params.first(),
                        gate.params.get(1),
                        gate.modes.first(),
                        gate.modes.get(1),
                    ) {
                        state.beamsplitter(theta, phi, mode1, mode2)?;
                    }
                }
                CVGateType::PhaseRotation => {
                    if let (Some(&phi), Some(&mode)) = (gate.params.first(), gate.modes.first()) {
                        state.phase_rotation(phi, mode)?;
                    }
                }
                _ => {
                    return Err(CVError::IncompatibleOperation(format!(
                        "Gate type {:?} not yet implemented",
                        gate.gate_type
                    )));
                }
            }
        }

        Ok(())
    }

    /// Get total sequence fidelity
    pub fn total_fidelity(&self) -> f64 {
        self.gates
            .iter()
            .map(|gate| gate.fidelity.unwrap_or(1.0))
            .product()
    }

    /// Get gate count by type
    pub fn gate_count(&self, gate_type: CVGateType) -> usize {
        self.gates
            .iter()
            .filter(|gate| gate.gate_type == gate_type)
            .count()
    }

    /// Optimize sequence (basic optimization)
    pub fn optimize(&mut self) -> Result<(), CVGateError> {
        // Remove redundant identity operations
        self.gates.retain(|gate| match gate.gate_type {
            CVGateType::PhaseRotation => {
                if let Some(&phi) = gate.params.first() {
                    (phi % (2.0 * PI)).abs() > 1e-10
                } else {
                    true
                }
            }
            CVGateType::Displacement => gate
                .complex_params
                .first()
                .map_or(true, |alpha| alpha.magnitude() > 1e-10),
            _ => true,
        });

        // Combine consecutive phase rotations on same mode
        // TODO: Implement more sophisticated optimizations

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::photonic::continuous_variable::GaussianState;

    #[test]
    fn test_displacement_gate() {
        let alpha = Complex::new(1.0, 0.5);
        let gate = DisplacementGate::new(alpha, 0);
        let mut state = GaussianState::vacuum(1);

        gate.apply(&mut state)
            .expect("Displacement gate application should succeed");

        assert!((state.mean[0] - alpha.real * SQRT_2).abs() < 1e-10);
        assert!((state.mean[1] - alpha.imag * SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_squeezing_gate() {
        let gate = SqueezingGate::new(1.0, 0.0, 0).expect("Squeezing gate creation should succeed");
        let mut state = GaussianState::vacuum(1);

        gate.apply(&mut state)
            .expect("Squeezing gate application should succeed");

        // Check that X quadrature is squeezed
        assert!(state.covariance[0][0] < 0.5);
        assert!(state.covariance[1][1] > 0.5);
    }

    #[test]
    fn test_beamsplitter_gate() {
        let gate = BeamsplitterGate::fifty_fifty(0, 1)
            .expect("50:50 beamsplitter creation should succeed");
        let mut state = GaussianState::coherent(Complex::new(2.0, 0.0), 0, 2)
            .expect("Coherent state creation should succeed");

        gate.apply(&mut state)
            .expect("Beamsplitter gate application should succeed");

        // Check that amplitude is distributed
        assert!(state.mean[0].abs() > 0.0);
        assert!(state.mean[2].abs() > 0.0);
        assert!((state.mean[0].abs() - state.mean[2].abs()).abs() < 1e-10);
    }

    #[test]
    fn test_gate_sequence() {
        let mut sequence = CVGateSequence::new(2);

        sequence
            .displacement(Complex::new(1.0, 0.0), 0)
            .expect("Adding displacement gate should succeed")
            .squeezing(0.5, 0.0, 0)
            .expect("Adding squeezing gate should succeed")
            .beamsplitter(PI / 4.0, 0.0, 0, 1)
            .expect("Adding beamsplitter gate should succeed");

        assert_eq!(sequence.gates.len(), 3);
        assert_eq!(sequence.gate_count(CVGateType::Displacement), 1);
        assert_eq!(sequence.gate_count(CVGateType::Squeezing), 1);
        assert_eq!(sequence.gate_count(CVGateType::Beamsplitter), 1);
    }

    #[test]
    fn test_sequence_application() {
        let mut sequence = CVGateSequence::new(1);
        sequence
            .displacement(Complex::new(1.0, 0.0), 0)
            .expect("Adding displacement gate should succeed")
            .phase_rotation(PI / 2.0, 0)
            .expect("Adding phase rotation gate should succeed");

        let mut state = GaussianState::vacuum(1);
        sequence
            .apply(&mut state)
            .expect("Gate sequence application should succeed");

        // State should be displaced and rotated
        assert!(state.mean[0].abs() > 0.0 || state.mean[1].abs() > 0.0);
    }

    #[test]
    fn test_sequence_optimization() {
        let mut sequence = CVGateSequence::new(1);
        sequence
            .displacement(Complex::new(0.0, 0.0), 0)
            .expect("Adding zero displacement should succeed") // Zero displacement
            .phase_rotation(0.0, 0)
            .expect("Adding zero phase rotation should succeed"); // Zero rotation

        sequence
            .optimize()
            .expect("Gate sequence optimization should succeed");

        // Should remove identity operations
        assert_eq!(sequence.gates.len(), 0);
    }
}
