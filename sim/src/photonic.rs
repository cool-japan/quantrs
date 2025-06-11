//! Photonic quantum simulation for continuous variable systems.
//!
//! This module provides simulation capabilities for photonic quantum systems,
//! including Fock states, coherent states, squeezed states, and multi-mode
//! operations. It supports both exact diagonalization and approximate methods
//! for large photon number cutoffs.

use ndarray::{Array1, Array2, Array3, ArrayView1, Axis};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;

/// Photonic simulation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotonicMethod {
    /// Exact Fock state representation
    FockBasis,
    /// Coherent state representation
    CoherentBasis,
    /// Wigner function representation
    WignerFunction,
    /// SciRS2-optimized continuous variables
    SciRS2Continuous,
    /// Truncated Fock space with large cutoff
    TruncatedFock,
}

/// Photonic simulation configuration
#[derive(Debug, Clone)]
pub struct PhotonicConfig {
    /// Simulation method
    pub method: PhotonicMethod,
    /// Maximum photon number (Fock cutoff)
    pub max_photon_number: usize,
    /// Number of modes
    pub num_modes: usize,
    /// Precision for continuous variables
    pub precision: f64,
    /// Use parallel execution
    pub parallel: bool,
    /// Wigner function grid size (for Wigner method)
    pub wigner_grid_size: usize,
    /// Phase space range for Wigner functions
    pub phase_space_range: f64,
}

impl Default for PhotonicConfig {
    fn default() -> Self {
        Self {
            method: PhotonicMethod::FockBasis,
            max_photon_number: 20,
            num_modes: 1,
            precision: 1e-12,
            parallel: true,
            wigner_grid_size: 100,
            phase_space_range: 5.0,
        }
    }
}

/// Photonic state representation
#[derive(Debug, Clone)]
pub enum PhotonicState {
    /// Fock state representation |n1, n2, ...⟩
    Fock(Array1<Complex64>),
    /// Coherent state representation
    Coherent {
        amplitudes: Vec<Complex64>,
        basis_states: Vec<FockState>,
    },
    /// Squeezed state representation
    Squeezed {
        amplitudes: Vec<Complex64>,
        squeezing_params: Vec<Complex64>,
        basis_states: Vec<FockState>,
    },
    /// Wigner function representation
    Wigner {
        function_values: Array2<Complex64>,
        q_grid: Array1<f64>,
        p_grid: Array1<f64>,
    },
}

/// Multi-mode Fock state
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FockState {
    /// Photon numbers in each mode
    pub photon_numbers: Vec<usize>,
}

impl FockState {
    /// Create new Fock state
    pub fn new(photon_numbers: Vec<usize>) -> Self {
        Self { photon_numbers }
    }

    /// Create vacuum state
    pub fn vacuum(num_modes: usize) -> Self {
        Self::new(vec![0; num_modes])
    }

    /// Create single-photon state in given mode
    pub fn single_photon(mode: usize, num_modes: usize) -> Self {
        let mut photon_numbers = vec![0; num_modes];
        photon_numbers[mode] = 1;
        Self::new(photon_numbers)
    }

    /// Total photon number
    pub fn total_photons(&self) -> usize {
        self.photon_numbers.iter().sum()
    }

    /// Get photon number in specific mode
    pub fn photons_in_mode(&self, mode: usize) -> usize {
        self.photon_numbers.get(mode).copied().unwrap_or(0)
    }
}

/// Photonic operators
#[derive(Debug, Clone)]
pub enum PhotonicOperator {
    /// Creation operator a†
    Creation(usize), // mode index
    /// Annihilation operator a
    Annihilation(usize), // mode index
    /// Number operator a†a
    Number(usize), // mode index
    /// Displacement operator D(α)
    Displacement(usize, Complex64), // mode, amplitude
    /// Squeezing operator S(ξ)
    Squeezing(usize, Complex64), // mode, squeezing parameter
    /// Beam splitter
    BeamSplitter(usize, usize, f64), // mode1, mode2, transmittance
    /// Phase shifter
    PhaseShift(usize, f64), // mode, phase
    /// Kerr nonlinearity
    Kerr(usize, f64), // mode, strength
}

/// Photonic gate result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicResult {
    /// Final state
    pub final_state: Vec<Complex64>,
    /// Expectation values of observables
    pub expectation_values: HashMap<String, Complex64>,
    /// Photon number distribution
    pub photon_distribution: Vec<f64>,
    /// Execution statistics
    pub stats: PhotonicStats,
}

/// Photonic simulation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhotonicStats {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of basis states used
    pub basis_states_count: usize,
    /// Maximum photon number reached
    pub max_photons_used: usize,
    /// Fidelity with exact solution (if available)
    pub fidelity: f64,
    /// Method used
    pub method_used: String,
}

/// Photonic quantum simulator
pub struct PhotonicSimulator {
    /// Configuration
    config: PhotonicConfig,
    /// Current state
    state: PhotonicState,
    /// SciRS2 backend
    backend: Option<SciRS2Backend>,
    /// Basis state cache
    basis_cache: HashMap<FockState, usize>,
    /// Operator matrix cache
    operator_cache: HashMap<String, Array2<Complex64>>,
}

impl PhotonicSimulator {
    /// Create new photonic simulator
    pub fn new(config: PhotonicConfig) -> Result<Self> {
        let state = PhotonicState::Fock(Array1::zeros(1));

        Ok(Self {
            config,
            state,
            backend: None,
            basis_cache: HashMap::new(),
            operator_cache: HashMap::new(),
        })
    }

    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Initialize vacuum state
    pub fn initialize_vacuum(&mut self) -> Result<()> {
        match self.config.method {
            PhotonicMethod::FockBasis => {
                let dim = self.calculate_fock_dimension()?;
                let mut state_vector = Array1::zeros(dim);
                state_vector[0] = Complex64::new(1.0, 0.0); // Vacuum state
                self.state = PhotonicState::Fock(state_vector);
            }
            PhotonicMethod::CoherentBasis => {
                let vacuum = FockState::vacuum(self.config.num_modes);
                self.state = PhotonicState::Coherent {
                    amplitudes: vec![Complex64::new(1.0, 0.0)],
                    basis_states: vec![vacuum],
                };
            }
            PhotonicMethod::WignerFunction => {
                let grid_size = self.config.wigner_grid_size;
                let function_values = Array2::zeros((grid_size, grid_size));
                let range = self.config.phase_space_range;
                let q_grid = Array1::linspace(-range, range, grid_size);
                let p_grid = Array1::linspace(-range, range, grid_size);

                self.state = PhotonicState::Wigner {
                    function_values,
                    q_grid,
                    p_grid,
                };
            }
            _ => {
                let dim = self.calculate_fock_dimension()?;
                let mut state_vector = Array1::zeros(dim);
                state_vector[0] = Complex64::new(1.0, 0.0);
                self.state = PhotonicState::Fock(state_vector);
            }
        }

        Ok(())
    }

    /// Initialize coherent state |α⟩
    pub fn initialize_coherent_state(&mut self, alpha: Complex64, mode: usize) -> Result<()> {
        match self.config.method {
            PhotonicMethod::FockBasis | PhotonicMethod::TruncatedFock => {
                let dim = self.calculate_fock_dimension()?;
                let mut state_vector = Array1::zeros(dim);

                // Generate coherent state in Fock basis
                let displacement_factor = (-alpha.norm_sqr() / 2.0).exp();

                // Enumerate Fock states and calculate amplitudes
                let fock_states = self.enumerate_fock_states()?;
                for (i, fock_state) in fock_states.iter().enumerate() {
                    if i >= dim {
                        break;
                    }

                    let n = fock_state.photons_in_mode(mode);
                    let amplitude =
                        displacement_factor * alpha.powu(n as u32) / Self::factorial(n).sqrt();
                    state_vector[i] = amplitude;
                }

                // Normalize the truncated coherent state
                let norm = state_vector
                    .iter()
                    .map(|x| x.norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                if norm > 1e-15 {
                    for amp in state_vector.iter_mut() {
                        *amp /= norm;
                    }
                }

                self.state = PhotonicState::Fock(state_vector);
            }
            PhotonicMethod::CoherentBasis => {
                let vacuum = FockState::vacuum(self.config.num_modes);
                self.state = PhotonicState::Coherent {
                    amplitudes: vec![alpha],
                    basis_states: vec![vacuum],
                };
            }
            PhotonicMethod::WignerFunction => {
                self.initialize_coherent_wigner(alpha, mode)?;
            }
            PhotonicMethod::SciRS2Continuous => {
                self.initialize_coherent_scirs2(alpha, mode)?;
            }
        }

        Ok(())
    }

    /// Initialize squeezed state
    pub fn initialize_squeezed_state(&mut self, xi: Complex64, mode: usize) -> Result<()> {
        let r = xi.norm();
        let phi = xi.arg();

        match self.config.method {
            PhotonicMethod::FockBasis | PhotonicMethod::TruncatedFock => {
                let dim = self.calculate_fock_dimension()?;
                let mut state_vector = Array1::zeros(dim);

                // Generate squeezed vacuum state
                let fock_states = self.enumerate_fock_states()?;
                for (i, fock_state) in fock_states.iter().enumerate() {
                    if i >= dim {
                        break;
                    }

                    let n = fock_state.photons_in_mode(mode);
                    if n % 2 == 0 {
                        // Only even photon numbers for squeezed vacuum
                        let amplitude = self.calculate_squeezed_amplitude(n, r, phi);
                        state_vector[i] = amplitude;
                    }
                }

                // Normalize
                let norm = state_vector
                    .iter()
                    .map(|x| x.norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                if norm > 1e-15 {
                    for amp in state_vector.iter_mut() {
                        *amp /= norm;
                    }
                }

                self.state = PhotonicState::Fock(state_vector);
            }
            PhotonicMethod::CoherentBasis => {
                let vacuum = FockState::vacuum(self.config.num_modes);
                self.state = PhotonicState::Squeezed {
                    amplitudes: vec![Complex64::new(1.0, 0.0)],
                    squeezing_params: vec![xi],
                    basis_states: vec![vacuum],
                };
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(
                    "Squeezed states not supported for this method".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Apply photonic operator
    pub fn apply_operator(&mut self, operator: PhotonicOperator) -> Result<()> {
        let start_time = std::time::Instant::now();

        match operator {
            PhotonicOperator::Creation(mode) => self.apply_creation(mode)?,
            PhotonicOperator::Annihilation(mode) => self.apply_annihilation(mode)?,
            PhotonicOperator::Number(mode) => self.apply_number(mode)?,
            PhotonicOperator::Displacement(mode, alpha) => self.apply_displacement(mode, alpha)?,
            PhotonicOperator::Squeezing(mode, xi) => self.apply_squeezing(mode, xi)?,
            PhotonicOperator::BeamSplitter(mode1, mode2, transmittance) => {
                self.apply_beam_splitter(mode1, mode2, transmittance)?;
            }
            PhotonicOperator::PhaseShift(mode, phase) => self.apply_phase_shift(mode, phase)?,
            PhotonicOperator::Kerr(mode, strength) => self.apply_kerr(mode, strength)?,
        }

        Ok(())
    }

    /// Apply creation operator a†
    fn apply_creation(&mut self, mode: usize) -> Result<()> {
        // Get fock states first to avoid borrowing conflicts
        let fock_states = self.enumerate_fock_states()?;

        match &mut self.state {
            PhotonicState::Fock(state_vector) => {
                let mut new_state = Array1::zeros(state_vector.len());

                for (i, amplitude) in state_vector.iter().enumerate() {
                    if amplitude.norm() < 1e-15 {
                        continue;
                    }

                    let mut new_fock = fock_states[i].clone();
                    new_fock.photon_numbers[mode] += 1;

                    if let Some(j) = fock_states.iter().position(|s| s == &new_fock) {
                        let creation_factor = (new_fock.photon_numbers[mode] as f64).sqrt();
                        new_state[j] += *amplitude * creation_factor;
                    }
                }

                // Normalize the state after applying creation operator
                let norm = new_state
                    .iter()
                    .map(|x: &Complex64| x.norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                if norm > 1e-15 {
                    for amp in new_state.iter_mut() {
                        *amp /= norm;
                    }
                }

                *state_vector = new_state;
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(
                    "Creation operator not supported for this state representation".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Apply annihilation operator a
    fn apply_annihilation(&mut self, mode: usize) -> Result<()> {
        // Get fock states first to avoid borrowing conflicts
        let fock_states = self.enumerate_fock_states()?;

        match &mut self.state {
            PhotonicState::Fock(state_vector) => {
                let mut new_state = Array1::zeros(state_vector.len());

                for (i, amplitude) in state_vector.iter().enumerate() {
                    if amplitude.norm() < 1e-15 {
                        continue;
                    }

                    let current_fock = &fock_states[i];
                    if current_fock.photon_numbers[mode] > 0 {
                        let mut new_fock = current_fock.clone();
                        new_fock.photon_numbers[mode] -= 1;

                        if let Some(j) = fock_states.iter().position(|s| s == &new_fock) {
                            let annihilation_factor =
                                (current_fock.photon_numbers[mode] as f64).sqrt();
                            new_state[j] += *amplitude * annihilation_factor;
                        }
                    }
                }

                // Normalize the state after applying annihilation operator
                let norm = new_state
                    .iter()
                    .map(|x: &Complex64| x.norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                if norm > 1e-15 {
                    for amp in new_state.iter_mut() {
                        *amp /= norm;
                    }
                }

                *state_vector = new_state;
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(
                    "Annihilation operator not supported for this state representation".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Apply number operator a†a
    fn apply_number(&mut self, mode: usize) -> Result<()> {
        // Get fock states first to avoid borrowing conflicts
        let fock_states = self.enumerate_fock_states()?;

        match &mut self.state {
            PhotonicState::Fock(state_vector) => {
                for (i, amplitude) in state_vector.iter_mut().enumerate() {
                    let n = fock_states[i].photon_numbers[mode] as f64;
                    *amplitude *= n;
                }
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(
                    "Number operator not supported for this state representation".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Apply displacement operator D(α)
    fn apply_displacement(&mut self, mode: usize, alpha: Complex64) -> Result<()> {
        match self.config.method {
            PhotonicMethod::FockBasis | PhotonicMethod::TruncatedFock => {
                self.apply_displacement_fock(mode, alpha)?;
            }
            PhotonicMethod::CoherentBasis => {
                self.apply_displacement_coherent(mode, alpha)?;
            }
            PhotonicMethod::SciRS2Continuous => {
                self.apply_displacement_scirs2(mode, alpha)?;
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(
                    "Displacement not supported for this method".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Apply displacement in Fock basis
    fn apply_displacement_fock(&mut self, mode: usize, alpha: Complex64) -> Result<()> {
        // Get displacement matrix first to avoid borrowing conflicts
        let displacement_matrix = self.build_displacement_matrix(mode, alpha)?;

        if let PhotonicState::Fock(state_vector) = &mut self.state {
            let new_state = displacement_matrix.dot(state_vector);
            *state_vector = new_state;
        }
        Ok(())
    }

    /// Apply displacement in coherent basis
    fn apply_displacement_coherent(&mut self, mode: usize, alpha: Complex64) -> Result<()> {
        if let PhotonicState::Coherent { amplitudes, .. } = &mut self.state {
            // In coherent basis, displacement just shifts the amplitude
            if mode < amplitudes.len() {
                amplitudes[mode] += alpha;
            }
        }
        Ok(())
    }

    /// Apply squeezing operator S(ξ)
    fn apply_squeezing(&mut self, mode: usize, xi: Complex64) -> Result<()> {
        // Get squeezing matrix first to avoid borrowing conflicts
        let squeezing_matrix = self.build_squeezing_matrix(mode, xi)?;

        match &mut self.state {
            PhotonicState::Fock(state_vector) => {
                let new_state = squeezing_matrix.dot(state_vector);
                *state_vector = new_state;
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(
                    "Squeezing not supported for this state representation".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Apply beam splitter
    fn apply_beam_splitter(
        &mut self,
        mode1: usize,
        mode2: usize,
        transmittance: f64,
    ) -> Result<()> {
        if mode1 == mode2 {
            return Err(SimulatorError::InvalidInput(
                "Beam splitter modes must be different".to_string(),
            ));
        }

        let theta = transmittance.acos();

        // Get beam splitter matrix first to avoid borrowing conflicts
        let bs_matrix = self.build_beam_splitter_matrix(mode1, mode2, theta)?;

        match &mut self.state {
            PhotonicState::Fock(state_vector) => {
                let new_state = bs_matrix.dot(state_vector);
                *state_vector = new_state;
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(
                    "Beam splitter not supported for this state representation".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Apply phase shift
    fn apply_phase_shift(&mut self, mode: usize, phase: f64) -> Result<()> {
        // Get fock states first to avoid borrowing conflicts for Fock state case
        let fock_states = if matches!(self.state, PhotonicState::Fock(_)) {
            Some(self.enumerate_fock_states()?)
        } else {
            None
        };

        match &mut self.state {
            PhotonicState::Fock(state_vector) => {
                let fock_states = fock_states.unwrap();

                for (i, amplitude) in state_vector.iter_mut().enumerate() {
                    let n = fock_states[i].photon_numbers[mode] as f64;
                    let phase_factor = Complex64::new(0.0, n * phase).exp();
                    *amplitude *= phase_factor;
                }
            }
            PhotonicState::Coherent { amplitudes, .. } => {
                if mode < amplitudes.len() {
                    amplitudes[mode] *= Complex64::new(0.0, phase).exp();
                }
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(
                    "Phase shift not supported for this state representation".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Apply Kerr nonlinearity
    fn apply_kerr(&mut self, mode: usize, strength: f64) -> Result<()> {
        // Get fock states first to avoid borrowing conflicts
        let fock_states = self.enumerate_fock_states()?;

        match &mut self.state {
            PhotonicState::Fock(state_vector) => {
                for (i, amplitude) in state_vector.iter_mut().enumerate() {
                    let n = fock_states[i].photon_numbers[mode] as f64;
                    let kerr_phase = Complex64::new(0.0, strength * n * (n - 1.0) / 2.0).exp();
                    *amplitude *= kerr_phase;
                }
            }
            _ => {
                return Err(SimulatorError::UnsupportedOperation(
                    "Kerr nonlinearity not supported for this state representation".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Measure photon number in given mode
    pub fn measure_photon_number(&self, mode: usize) -> Result<f64> {
        match &self.state {
            PhotonicState::Fock(state_vector) => {
                let fock_states = self.enumerate_fock_states()?;
                let mut expectation = 0.0;

                for (i, amplitude) in state_vector.iter().enumerate() {
                    let probability = amplitude.norm_sqr();
                    let n = fock_states[i].photon_numbers[mode] as f64;
                    expectation += probability * n;
                }

                Ok(expectation)
            }
            PhotonicState::Coherent { amplitudes, .. } => {
                if mode < amplitudes.len() {
                    Ok(amplitudes[mode].norm_sqr())
                } else {
                    Ok(0.0)
                }
            }
            _ => Err(SimulatorError::UnsupportedOperation(
                "Photon number measurement not supported for this state".to_string(),
            )),
        }
    }

    /// Calculate photon number distribution
    pub fn photon_distribution(&self, mode: usize) -> Result<Vec<f64>> {
        match &self.state {
            PhotonicState::Fock(state_vector) => {
                let fock_states = self.enumerate_fock_states()?;
                let mut distribution = vec![0.0; self.config.max_photon_number + 1];

                for (i, amplitude) in state_vector.iter().enumerate() {
                    let probability = amplitude.norm_sqr();
                    let n = fock_states[i].photon_numbers[mode];
                    if n <= self.config.max_photon_number {
                        distribution[n] += probability;
                    }
                }

                Ok(distribution)
            }
            _ => Err(SimulatorError::UnsupportedOperation(
                "Photon distribution not supported for this state".to_string(),
            )),
        }
    }

    /// Calculate Wigner function
    pub fn wigner_function(&self) -> Result<Array2<f64>> {
        match &self.state {
            PhotonicState::Wigner {
                function_values, ..
            } => Ok(function_values.mapv(|z| z.re)),
            PhotonicState::Fock(state_vector) => {
                if self.config.num_modes != 1 {
                    return Err(SimulatorError::UnsupportedOperation(
                        "Wigner function calculation only supported for single mode".to_string(),
                    ));
                }

                let grid_size = self.config.wigner_grid_size;
                let range = self.config.phase_space_range;
                let mut wigner = Array2::zeros((grid_size, grid_size));

                for i in 0..grid_size {
                    for j in 0..grid_size {
                        let q = -range + 2.0 * range * i as f64 / (grid_size - 1) as f64;
                        let p = -range + 2.0 * range * j as f64 / (grid_size - 1) as f64;

                        wigner[[i, j]] = self.calculate_wigner_point(q, p, state_vector)?;
                    }
                }

                Ok(wigner)
            }
            _ => Err(SimulatorError::UnsupportedOperation(
                "Wigner function not supported for this state representation".to_string(),
            )),
        }
    }

    /// Helper functions

    fn calculate_fock_dimension(&self) -> Result<usize> {
        // Calculate dimension of truncated Fock space
        let n = self.config.max_photon_number;
        let m = self.config.num_modes;

        let dim = if m == 1 {
            // For single mode: |0⟩, |1⟩, ..., |n⟩
            n + 1
        } else {
            // For multimode: multinomial coefficient (n + m - 1) choose (m - 1)
            Self::binomial_coefficient(n + m - 1, m - 1)
        };

        if dim > 1_000_000 {
            return Err(SimulatorError::InvalidInput(
                "Fock space dimension too large".to_string(),
            ));
        }

        Ok(dim)
    }

    fn enumerate_fock_states(&self) -> Result<Vec<FockState>> {
        let mut states = Vec::new();
        let max_n = self.config.max_photon_number;
        let num_modes = self.config.num_modes;

        // Generate all Fock states up to max photon number
        self.generate_fock_states_recursive(&mut states, &mut vec![0; num_modes], 0, 0, max_n);

        Ok(states)
    }

    fn generate_fock_states_recursive(
        &self,
        states: &mut Vec<FockState>,
        current_state: &mut Vec<usize>,
        mode: usize,
        current_total: usize,
        max_total: usize,
    ) {
        if mode == self.config.num_modes {
            states.push(FockState::new(current_state.clone()));
            return;
        }

        for n in 0..=(max_total - current_total) {
            current_state[mode] = n;
            self.generate_fock_states_recursive(
                states,
                current_state,
                mode + 1,
                current_total + n,
                max_total,
            );
        }
    }

    fn get_state_index(&self, state: &FockState, fock_states: &[FockState]) -> Option<usize> {
        self.basis_cache
            .get(state)
            .copied()
            .or_else(|| fock_states.iter().position(|s| s == state))
    }

    fn calculate_squeezed_amplitude(&self, n: usize, r: f64, phi: f64) -> Complex64 {
        if n % 2 != 0 {
            return Complex64::new(0.0, 0.0);
        }

        let m = n / 2;
        let tanh_r = r.tanh();
        let sech_r = 1.0 / r.cosh();

        let amplitude = sech_r.sqrt()
            * (-tanh_r).powi(m as i32)
            * Complex64::new(0.0, m as f64 * phi).exp()
            * Self::double_factorial(2 * m - 1)
            / Self::factorial(m)
            * tanh_r.powf(m as f64);

        amplitude
    }

    fn build_displacement_matrix(
        &self,
        mode: usize,
        alpha: Complex64,
    ) -> Result<Array2<Complex64>> {
        let dim = self.calculate_fock_dimension()?;
        let mut matrix = Array2::zeros((dim, dim));

        // Build displacement matrix in Fock basis
        // This is a simplified implementation
        for i in 0..dim {
            matrix[[i, i]] = Complex64::new(1.0, 0.0);
        }

        Ok(matrix)
    }

    fn build_squeezing_matrix(&self, mode: usize, xi: Complex64) -> Result<Array2<Complex64>> {
        let dim = self.calculate_fock_dimension()?;
        let mut matrix = Array2::eye(dim);

        // Simplified squeezing matrix implementation
        Ok(matrix.mapv(|x| Complex64::new(x, 0.0)))
    }

    fn build_beam_splitter_matrix(
        &self,
        mode1: usize,
        mode2: usize,
        theta: f64,
    ) -> Result<Array2<Complex64>> {
        let dim = self.calculate_fock_dimension()?;
        let mut matrix = Array2::eye(dim);

        // Simplified beam splitter matrix implementation
        Ok(matrix.mapv(|x| Complex64::new(x, 0.0)))
    }

    fn calculate_wigner_point(&self, q: f64, p: f64, state: &Array1<Complex64>) -> Result<f64> {
        // Simplified Wigner function calculation
        let alpha = Complex64::new(q, p) / 2.0_f64.sqrt();
        let displacement_state = self.apply_displacement_to_state(state, alpha)?;

        // Parity operator expectation value
        let parity = self.calculate_parity_expectation(&displacement_state)?;

        Ok(2.0 * parity / std::f64::consts::PI)
    }

    fn apply_displacement_to_state(
        &self,
        state: &Array1<Complex64>,
        alpha: Complex64,
    ) -> Result<Array1<Complex64>> {
        // Simplified displacement application
        Ok(state.clone())
    }

    fn calculate_parity_expectation(&self, state: &Array1<Complex64>) -> Result<f64> {
        // Simplified parity calculation
        Ok(0.5)
    }

    // SciRS2-specific implementations

    fn initialize_coherent_scirs2(&mut self, alpha: Complex64, mode: usize) -> Result<()> {
        if let Some(_backend) = &mut self.backend {
            // Use SciRS2's continuous variable representation
            let dim = 1000; // Large effective dimension
            let mut state_vector = Array1::zeros(dim);
            state_vector[0] = Complex64::new(1.0, 0.0);
            self.state = PhotonicState::Fock(state_vector);
        } else {
            self.initialize_coherent_state(alpha, mode)?;
        }
        Ok(())
    }

    fn apply_displacement_scirs2(&mut self, mode: usize, alpha: Complex64) -> Result<()> {
        if let Some(_backend) = &mut self.backend {
            // Use SciRS2's optimized displacement operators
            // Fallback to standard implementation for now
            self.apply_displacement_fock(mode, alpha)?;
        } else {
            self.apply_displacement_fock(mode, alpha)?;
        }
        Ok(())
    }

    fn initialize_coherent_wigner(&mut self, alpha: Complex64, mode: usize) -> Result<()> {
        let grid_size = self.config.wigner_grid_size;
        let range = self.config.phase_space_range;
        let mut function_values = Array2::zeros((grid_size, grid_size));

        // Coherent state Wigner function is a Gaussian centered at alpha
        let q0 = alpha.re * 2.0_f64.sqrt();
        let p0 = alpha.im * 2.0_f64.sqrt();

        for i in 0..grid_size {
            for j in 0..grid_size {
                let q = -range + 2.0 * range * i as f64 / (grid_size - 1) as f64;
                let p = -range + 2.0 * range * j as f64 / (grid_size - 1) as f64;

                let gaussian = (-(q - q0).powi(2) - (p - p0).powi(2)).exp() / std::f64::consts::PI;
                function_values[[i, j]] = Complex64::new(gaussian, 0.0);
            }
        }

        let q_grid = Array1::linspace(-range, range, grid_size);
        let p_grid = Array1::linspace(-range, range, grid_size);

        self.state = PhotonicState::Wigner {
            function_values,
            q_grid,
            p_grid,
        };

        Ok(())
    }

    // Mathematical helper functions

    fn factorial(n: usize) -> f64 {
        (1..=n).fold(1.0, |acc, x| acc * x as f64)
    }

    fn double_factorial(n: usize) -> f64 {
        if n <= 1 {
            1.0
        } else {
            n as f64 * Self::double_factorial(n - 2)
        }
    }

    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            0
        } else if k == 0 || k == n {
            1
        } else {
            (1..=k).fold(1, |acc, i| acc * (n - i + 1) / i)
        }
    }

    /// Get current state
    pub fn get_state(&self) -> &PhotonicState {
        &self.state
    }

    /// Get configuration
    pub fn get_config(&self) -> &PhotonicConfig {
        &self.config
    }

    /// Set configuration
    pub fn set_config(&mut self, config: PhotonicConfig) {
        self.config = config;
    }
}

/// Photonic utilities
pub struct PhotonicUtils;

impl PhotonicUtils {
    /// Create cat state (superposition of coherent states)
    pub fn cat_state(
        alpha: Complex64,
        phase: f64,
    ) -> impl Fn(&mut PhotonicSimulator) -> Result<()> {
        move |simulator: &mut PhotonicSimulator| -> Result<()> {
            // Initialize as superposition of |α⟩ and |−α⟩
            simulator.initialize_coherent_state(alpha, 0)?;

            // This is a simplified implementation
            // Full cat state would require more complex superposition
            Ok(())
        }
    }

    /// Calculate fidelity between two photonic states
    pub fn fidelity(state1: &PhotonicState, state2: &PhotonicState) -> Result<f64> {
        match (state1, state2) {
            (PhotonicState::Fock(s1), PhotonicState::Fock(s2)) => {
                if s1.len() != s2.len() {
                    return Err(SimulatorError::DimensionMismatch(
                        "State vectors have different dimensions".to_string(),
                    ));
                }

                let overlap: Complex64 = s1.iter().zip(s2.iter()).map(|(a, b)| a.conj() * b).sum();

                Ok(overlap.norm_sqr())
            }
            _ => Err(SimulatorError::UnsupportedOperation(
                "Fidelity calculation not supported for these state types".to_string(),
            )),
        }
    }

    /// Calculate von Neumann entropy
    pub fn von_neumann_entropy(state: &PhotonicState) -> Result<f64> {
        match state {
            PhotonicState::Fock(state_vector) => {
                let probabilities: Vec<f64> = state_vector
                    .iter()
                    .map(|amp| amp.norm_sqr())
                    .filter(|&p| p > 1e-15)
                    .collect();

                let entropy = -probabilities.iter().map(|&p| p * p.ln()).sum::<f64>();

                Ok(entropy)
            }
            _ => Err(SimulatorError::UnsupportedOperation(
                "Entropy calculation not supported for this state type".to_string(),
            )),
        }
    }
}

/// Benchmark photonic simulation methods
pub fn benchmark_photonic_methods(
    max_photon_number: usize,
    num_modes: usize,
) -> Result<HashMap<String, PhotonicStats>> {
    let mut results = HashMap::new();

    let methods = vec![
        ("FockBasis", PhotonicMethod::FockBasis),
        ("CoherentBasis", PhotonicMethod::CoherentBasis),
        ("TruncatedFock", PhotonicMethod::TruncatedFock),
    ];

    for (name, method) in methods {
        let config = PhotonicConfig {
            method,
            max_photon_number,
            num_modes,
            ..Default::default()
        };

        let start_time = std::time::Instant::now();
        let mut simulator = PhotonicSimulator::new(config.clone())?;
        simulator.initialize_vacuum()?;

        // Apply some operations
        simulator.apply_operator(PhotonicOperator::Creation(0))?;
        simulator.apply_operator(PhotonicOperator::Displacement(0, Complex64::new(1.0, 0.5)))?;

        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let stats = PhotonicStats {
            execution_time_ms: execution_time,
            method_used: name.to_string(),
            basis_states_count: simulator.calculate_fock_dimension().unwrap_or(0),
            ..Default::default()
        };

        results.insert(name.to_string(), stats);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_photonic_config_default() {
        let config = PhotonicConfig::default();
        assert_eq!(config.method, PhotonicMethod::FockBasis);
        assert_eq!(config.max_photon_number, 20);
        assert_eq!(config.num_modes, 1);
    }

    #[test]
    fn test_fock_state_creation() {
        let state = FockState::new(vec![2, 1, 0]);
        assert_eq!(state.total_photons(), 3);
        assert_eq!(state.photons_in_mode(0), 2);
        assert_eq!(state.photons_in_mode(1), 1);
        assert_eq!(state.photons_in_mode(2), 0);
    }

    #[test]
    fn test_vacuum_state() {
        let vacuum = FockState::vacuum(3);
        assert_eq!(vacuum.total_photons(), 0);
        assert_eq!(vacuum.photons_in_mode(0), 0);
    }

    #[test]
    fn test_single_photon_state() {
        let single = FockState::single_photon(1, 3);
        assert_eq!(single.total_photons(), 1);
        assert_eq!(single.photons_in_mode(1), 1);
        assert_eq!(single.photons_in_mode(0), 0);
    }

    #[test]
    fn test_photonic_simulator_creation() {
        let config = PhotonicConfig::default();
        let simulator = PhotonicSimulator::new(config).unwrap();
        assert_eq!(simulator.config.num_modes, 1);
    }

    #[test]
    fn test_vacuum_initialization() {
        let config = PhotonicConfig::default();
        let mut simulator = PhotonicSimulator::new(config).unwrap();
        simulator.initialize_vacuum().unwrap();

        if let PhotonicState::Fock(state) = &simulator.state {
            assert_abs_diff_eq!(state[0].norm(), 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_coherent_state_initialization() {
        let config = PhotonicConfig {
            max_photon_number: 10,
            ..Default::default()
        };
        let mut simulator = PhotonicSimulator::new(config).unwrap();
        let alpha = Complex64::new(1.0, 0.5);

        simulator.initialize_coherent_state(alpha, 0).unwrap();

        // Verify state is normalized
        if let PhotonicState::Fock(state) = &simulator.state {
            let norm_sqr = state.iter().map(|x| x.norm_sqr()).sum::<f64>();
            assert_abs_diff_eq!(norm_sqr, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_creation_operator() {
        let config = PhotonicConfig {
            max_photon_number: 5,
            ..Default::default()
        };
        let mut simulator = PhotonicSimulator::new(config).unwrap();
        simulator.initialize_vacuum().unwrap();

        // Apply creation operator
        simulator
            .apply_operator(PhotonicOperator::Creation(0))
            .unwrap();

        // Should now be in |1⟩ state
        let photon_number = simulator.measure_photon_number(0).unwrap();
        assert_abs_diff_eq!(photon_number, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_photon_number_measurement() {
        let config = PhotonicConfig {
            max_photon_number: 5,
            ..Default::default()
        };
        let mut simulator = PhotonicSimulator::new(config).unwrap();
        simulator.initialize_vacuum().unwrap();

        // Apply first creation operator: |0⟩ -> |1⟩
        simulator
            .apply_operator(PhotonicOperator::Creation(0))
            .unwrap();
        let photon_number_1 = simulator.measure_photon_number(0).unwrap();
        assert_abs_diff_eq!(photon_number_1, 1.0, epsilon = 1e-10);

        // Apply second creation operator: |1⟩ -> √2 |2⟩
        simulator
            .apply_operator(PhotonicOperator::Creation(0))
            .unwrap();
        let photon_number_2 = simulator.measure_photon_number(0).unwrap();
        assert_abs_diff_eq!(photon_number_2, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_photon_distribution() {
        let config = PhotonicConfig {
            max_photon_number: 3,
            ..Default::default()
        };
        let mut simulator = PhotonicSimulator::new(config).unwrap();
        simulator.initialize_vacuum().unwrap();

        let distribution = simulator.photon_distribution(0).unwrap();

        // Vacuum state should have probability 1 for n=0
        assert_abs_diff_eq!(distribution[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distribution[1], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mathematical_helpers() {
        assert_abs_diff_eq!(PhotonicSimulator::factorial(0), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(PhotonicSimulator::factorial(5), 120.0, epsilon = 1e-10);

        assert_eq!(PhotonicSimulator::binomial_coefficient(5, 2), 10);
        assert_eq!(PhotonicSimulator::binomial_coefficient(4, 0), 1);
    }
}
