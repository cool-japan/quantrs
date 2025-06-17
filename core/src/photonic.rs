//! Photonic Quantum Computing
//!
//! This module implements photonic quantum computing operations including
//! linear optical quantum gates, photon-based qubits, and measurement-based
//! quantum computation in the photonic platform.

use crate::error::{QuantRS2Error, QuantRS2Result};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::collections::HashMap;

/// Photonic quantum gate types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhotonicGateType {
    /// Beam splitter with reflectivity parameter
    BeamSplitter { reflectivity: f64 },
    /// Phase shifter with phase parameter
    PhaseShifter { phase: f64 },
    /// Mach-Zehnder interferometer
    MachZehnder { phase1: f64, phase2: f64 },
    /// Controlled-Z gate using linear optics
    ControlledZ,
    /// Controlled-NOT gate using linear optics
    ControlledNot,
    /// Hong-Ou-Mandel interference
    HongOuMandel,
    /// Photon number measurement
    PhotonNumber,
    /// Homodyne measurement
    Homodyne { phase: f64 },
    /// Heterodyne measurement
    Heterodyne,
}

/// Photonic qubit encoding
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhotonicEncoding {
    /// Dual-rail encoding (path encoding)
    DualRail,
    /// Polarization encoding
    Polarization,
    /// Time-bin encoding
    TimeBin,
    /// Frequency encoding
    Frequency,
    /// Continuous variable encoding
    ContinuousVariable,
}

/// Optical mode representation
#[derive(Debug, Clone)]
pub struct OpticalMode {
    /// Mode index
    pub mode_id: usize,
    /// Photon number state amplitudes
    pub fock_state: Array1<Complex64>,
    /// Maximum photon number considered
    pub max_photons: usize,
    /// Mode frequency (optional)
    pub frequency: Option<f64>,
    /// Mode polarization (optional)
    pub polarization: Option<String>,
}

impl OpticalMode {
    /// Create vacuum state
    pub fn vacuum(mode_id: usize, max_photons: usize) -> Self {
        let mut fock_state = Array1::zeros(max_photons + 1);
        fock_state[0] = Complex64::new(1.0, 0.0); // |0⟩ state

        Self {
            mode_id,
            fock_state,
            max_photons,
            frequency: None,
            polarization: None,
        }
    }

    /// Create single photon state
    pub fn single_photon(mode_id: usize, max_photons: usize) -> Self {
        let mut fock_state = Array1::zeros(max_photons + 1);
        fock_state[1] = Complex64::new(1.0, 0.0); // |1⟩ state

        Self {
            mode_id,
            fock_state,
            max_photons,
            frequency: None,
            polarization: None,
        }
    }

    /// Create coherent state |α⟩
    pub fn coherent(mode_id: usize, max_photons: usize, alpha: Complex64) -> Self {
        let mut fock_state = Array1::zeros(max_photons + 1);

        // |α⟩ = e^{-|α|²/2} Σ_n α^n/√(n!) |n⟩
        let alpha_norm_sq = alpha.norm_sqr();
        let prefactor = (-alpha_norm_sq / 2.0).exp();

        let mut factorial = 1.0;
        for n in 0..=max_photons {
            if n > 0 {
                factorial *= n as f64;
            }

            let coefficient = prefactor * alpha.powf(n as f64) / factorial.sqrt();
            fock_state[n] = coefficient;
        }

        Self {
            mode_id,
            fock_state,
            max_photons,
            frequency: None,
            polarization: None,
        }
    }

    /// Get photon number expectation value
    pub fn photon_number_expectation(&self) -> f64 {
        let mut expectation = 0.0;
        for n in 0..=self.max_photons {
            expectation += (n as f64) * self.fock_state[n].norm_sqr();
        }
        expectation
    }

    /// Get probability of measuring n photons
    pub fn photon_probability(&self, n: usize) -> f64 {
        if n > self.max_photons {
            0.0
        } else {
            self.fock_state[n].norm_sqr()
        }
    }
}

/// Photonic quantum system
#[derive(Debug, Clone)]
pub struct PhotonicSystem {
    /// Number of optical modes
    pub num_modes: usize,
    /// Optical modes
    pub modes: Vec<OpticalMode>,
    /// System state in multi-mode Fock basis
    pub state: Option<Array1<Complex64>>,
    /// Maximum photons per mode
    pub max_photons_per_mode: usize,
    /// Total maximum photons in system
    pub max_total_photons: usize,
}

impl PhotonicSystem {
    /// Create new photonic system
    pub fn new(num_modes: usize, max_photons_per_mode: usize) -> Self {
        let modes = (0..num_modes)
            .map(|i| OpticalMode::vacuum(i, max_photons_per_mode))
            .collect();

        let max_total_photons = num_modes * max_photons_per_mode;

        Self {
            num_modes,
            modes,
            state: None,
            max_photons_per_mode,
            max_total_photons,
        }
    }

    /// Initialize system state from individual mode states
    pub fn initialize_from_modes(&mut self) -> QuantRS2Result<()> {
        let total_dim = (self.max_photons_per_mode + 1).pow(self.num_modes as u32);
        let mut state = Array1::zeros(total_dim);

        // Compute tensor product of all mode states
        state[0] = Complex64::new(1.0, 0.0);

        for (mode_idx, mode) in self.modes.iter().enumerate() {
            let mut new_state = Array1::zeros(total_dim);
            let mode_dim = self.max_photons_per_mode + 1;

            for (state_idx, &amplitude) in state.iter().enumerate() {
                if amplitude.norm() > 1e-12 {
                    for (n, &mode_amp) in mode.fock_state.iter().enumerate() {
                        if mode_amp.norm() > 1e-12 {
                            let new_idx = state_idx + n * mode_dim.pow(mode_idx as u32);
                            if new_idx < total_dim {
                                new_state[new_idx] += amplitude * mode_amp;
                            }
                        }
                    }
                }
            }
            state = new_state;
        }

        self.state = Some(state);
        Ok(())
    }

    /// Apply beam splitter between two modes
    pub fn apply_beam_splitter(
        &mut self,
        mode1: usize,
        mode2: usize,
        reflectivity: f64,
    ) -> QuantRS2Result<()> {
        if mode1 >= self.num_modes || mode2 >= self.num_modes {
            return Err(QuantRS2Error::InvalidInput(
                "Mode index out of bounds".to_string(),
            ));
        }

        if mode1 == mode2 {
            return Err(QuantRS2Error::InvalidInput(
                "Cannot apply beam splitter to same mode".to_string(),
            ));
        }

        // Beam splitter transformation:
        // a₁' = √r a₁ + √(1-r) e^{iφ} a₂
        // a₂' = √(1-r) a₁ - √r e^{iφ} a₂
        let r = reflectivity;
        let t = (1.0 - r).sqrt();
        let r_sqrt = r.sqrt();

        // Apply transformation to individual modes first
        let mode1_state = self.modes[mode1].fock_state.clone();
        let mode2_state = self.modes[mode2].fock_state.clone();

        let mut new_mode1_state = Array1::zeros(self.max_photons_per_mode + 1);
        let mut new_mode2_state = Array1::zeros(self.max_photons_per_mode + 1);

        // This is a simplified implementation - full beam splitter requires
        // careful handling of multi-photon states
        for n in 0..=self.max_photons_per_mode {
            new_mode1_state[n] = r_sqrt * mode1_state[n] + t * mode2_state[n];
            new_mode2_state[n] = t * mode1_state[n] - r_sqrt * mode2_state[n];
        }

        self.modes[mode1].fock_state = new_mode1_state;
        self.modes[mode2].fock_state = new_mode2_state;

        // Update system state
        self.initialize_from_modes()?;

        Ok(())
    }

    /// Apply phase shifter to a mode
    pub fn apply_phase_shifter(&mut self, mode: usize, phase: f64) -> QuantRS2Result<()> {
        if mode >= self.num_modes {
            return Err(QuantRS2Error::InvalidInput(
                "Mode index out of bounds".to_string(),
            ));
        }

        // Phase shifter: |n⟩ → e^{inφ} |n⟩
        for n in 0..=self.max_photons_per_mode {
            let phase_factor = Complex64::new(0.0, (n as f64) * phase).exp();
            self.modes[mode].fock_state[n] *= phase_factor;
        }

        // Update system state
        self.initialize_from_modes()?;

        Ok(())
    }

    /// Measure photon number in a mode
    pub fn measure_photon_number(&mut self, mode: usize) -> QuantRS2Result<usize> {
        if mode >= self.num_modes {
            return Err(QuantRS2Error::InvalidInput(
                "Mode index out of bounds".to_string(),
            ));
        }

        // Get probabilities for each photon number
        let probabilities: Vec<f64> = (0..=self.max_photons_per_mode)
            .map(|n| self.modes[mode].photon_probability(n))
            .collect();

        // Sample outcome
        let mut rng = rand::rng();
        use rand::Rng;
        let random_value: f64 = rng.random();
        let mut cumulative = 0.0;

        for (n, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                // Collapse state to definite photon number
                let mut new_state = Array1::zeros(self.max_photons_per_mode + 1);
                new_state[n] = Complex64::new(1.0, 0.0);
                self.modes[mode].fock_state = new_state;

                self.initialize_from_modes()?;
                return Ok(n);
            }
        }

        // Fallback
        Ok(self.max_photons_per_mode)
    }

    /// Perform homodyne measurement
    pub fn homodyne_measurement(&mut self, mode: usize, phase: f64) -> QuantRS2Result<f64> {
        if mode >= self.num_modes {
            return Err(QuantRS2Error::InvalidInput(
                "Mode index out of bounds".to_string(),
            ));
        }

        // Homodyne measurement of X_φ = a e^{-iφ} + a† e^{iφ}
        // This is a simplified implementation
        let expectation = self.quadrature_expectation(mode, phase)?;

        // Add measurement noise (simplified)
        let mut rng = rand::rng();
        use rand::Rng;
        let noise = rng.random_range(-0.1..0.1); // Simplified noise model

        Ok(expectation + noise)
    }

    /// Calculate quadrature expectation value
    fn quadrature_expectation(&self, mode: usize, phase: f64) -> QuantRS2Result<f64> {
        // Simplified calculation for demonstration
        let photon_expectation = self.modes[mode].photon_number_expectation();
        Ok(photon_expectation.sqrt() * phase.cos())
    }
}

/// Photonic quantum gate
#[derive(Debug, Clone)]
pub struct PhotonicGate {
    /// Gate type
    pub gate_type: PhotonicGateType,
    /// Target modes
    pub target_modes: Vec<usize>,
    /// Gate parameters
    pub parameters: HashMap<String, f64>,
}

impl PhotonicGate {
    /// Create beam splitter gate
    pub fn beam_splitter(mode1: usize, mode2: usize, reflectivity: f64) -> Self {
        Self {
            gate_type: PhotonicGateType::BeamSplitter { reflectivity },
            target_modes: vec![mode1, mode2],
            parameters: {
                let mut params = HashMap::new();
                params.insert("reflectivity".to_string(), reflectivity);
                params
            },
        }
    }

    /// Create phase shifter gate
    pub fn phase_shifter(mode: usize, phase: f64) -> Self {
        Self {
            gate_type: PhotonicGateType::PhaseShifter { phase },
            target_modes: vec![mode],
            parameters: {
                let mut params = HashMap::new();
                params.insert("phase".to_string(), phase);
                params
            },
        }
    }

    /// Create Mach-Zehnder interferometer
    pub fn mach_zehnder(mode1: usize, mode2: usize, phase1: f64, phase2: f64) -> Self {
        Self {
            gate_type: PhotonicGateType::MachZehnder { phase1, phase2 },
            target_modes: vec![mode1, mode2],
            parameters: {
                let mut params = HashMap::new();
                params.insert("phase1".to_string(), phase1);
                params.insert("phase2".to_string(), phase2);
                params
            },
        }
    }

    /// Apply gate to photonic system
    pub fn apply(&self, system: &mut PhotonicSystem) -> QuantRS2Result<()> {
        match self.gate_type {
            PhotonicGateType::BeamSplitter { reflectivity } => {
                if self.target_modes.len() != 2 {
                    return Err(QuantRS2Error::InvalidInput(
                        "Beam splitter requires exactly 2 modes".to_string(),
                    ));
                }
                system.apply_beam_splitter(self.target_modes[0], self.target_modes[1], reflectivity)
            }
            PhotonicGateType::PhaseShifter { phase } => {
                if self.target_modes.len() != 1 {
                    return Err(QuantRS2Error::InvalidInput(
                        "Phase shifter requires exactly 1 mode".to_string(),
                    ));
                }
                system.apply_phase_shifter(self.target_modes[0], phase)
            }
            PhotonicGateType::MachZehnder { phase1, phase2 } => {
                if self.target_modes.len() != 2 {
                    return Err(QuantRS2Error::InvalidInput(
                        "Mach-Zehnder requires exactly 2 modes".to_string(),
                    ));
                }
                // Mach-Zehnder = BS + PS + BS + PS
                system.apply_beam_splitter(self.target_modes[0], self.target_modes[1], 0.5)?;
                system.apply_phase_shifter(self.target_modes[0], phase1)?;
                system.apply_phase_shifter(self.target_modes[1], phase2)?;
                system.apply_beam_splitter(self.target_modes[0], self.target_modes[1], 0.5)
            }
            PhotonicGateType::ControlledZ => {
                // Simplified CZ implementation using interference
                if self.target_modes.len() != 2 {
                    return Err(QuantRS2Error::InvalidInput(
                        "Controlled-Z requires exactly 2 modes".to_string(),
                    ));
                }
                // This is a placeholder - full implementation requires complex interferometry
                system.apply_phase_shifter(self.target_modes[1], std::f64::consts::PI)
            }
            PhotonicGateType::ControlledNot => {
                // Simplified CNOT implementation
                if self.target_modes.len() != 2 {
                    return Err(QuantRS2Error::InvalidInput(
                        "Controlled-NOT requires exactly 2 modes".to_string(),
                    ));
                }
                // Placeholder implementation
                system.apply_beam_splitter(self.target_modes[0], self.target_modes[1], 0.5)
            }
            _ => Err(QuantRS2Error::InvalidInput(
                "Gate type not yet implemented".to_string(),
            )),
        }
    }
}

/// Photonic quantum circuit
#[derive(Debug, Clone)]
pub struct PhotonicCircuit {
    /// Number of modes
    pub num_modes: usize,
    /// Sequence of photonic gates
    pub gates: Vec<PhotonicGate>,
    /// Maximum photons per mode
    pub max_photons_per_mode: usize,
}

impl PhotonicCircuit {
    /// Create new photonic circuit
    pub fn new(num_modes: usize, max_photons_per_mode: usize) -> Self {
        Self {
            num_modes,
            gates: Vec::new(),
            max_photons_per_mode,
        }
    }

    /// Add gate to circuit
    pub fn add_gate(&mut self, gate: PhotonicGate) -> QuantRS2Result<()> {
        // Validate gate targets
        for &mode in &gate.target_modes {
            if mode >= self.num_modes {
                return Err(QuantRS2Error::InvalidInput(
                    "Gate target mode out of bounds".to_string(),
                ));
            }
        }

        self.gates.push(gate);
        Ok(())
    }

    /// Execute circuit on photonic system
    pub fn execute(&self, system: &mut PhotonicSystem) -> QuantRS2Result<()> {
        if system.num_modes != self.num_modes {
            return Err(QuantRS2Error::InvalidInput(
                "System and circuit mode count mismatch".to_string(),
            ));
        }

        for gate in &self.gates {
            gate.apply(system)?;
        }

        Ok(())
    }

    /// Optimize circuit for linear optics
    pub fn optimize_linear_optics(&mut self) -> QuantRS2Result<()> {
        // Combine consecutive phase shifters on same mode
        let mut optimized_gates = Vec::new();
        let mut i = 0;

        while i < self.gates.len() {
            let current_gate = &self.gates[i];

            if let PhotonicGateType::PhaseShifter { phase } = current_gate.gate_type {
                let mode = current_gate.target_modes[0];
                let mut total_phase = phase;
                let mut j = i + 1;

                // Look for consecutive phase shifters on same mode
                while j < self.gates.len() {
                    if let PhotonicGateType::PhaseShifter { phase: next_phase } =
                        self.gates[j].gate_type
                    {
                        if self.gates[j].target_modes[0] == mode {
                            total_phase += next_phase;
                            j += 1;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                // Add optimized phase shifter
                if (total_phase % (2.0 * std::f64::consts::PI)).abs() > 1e-10 {
                    optimized_gates.push(PhotonicGate::phase_shifter(mode, total_phase));
                }

                i = j;
            } else {
                optimized_gates.push(current_gate.clone());
                i += 1;
            }
        }

        self.gates = optimized_gates;
        Ok(())
    }

    /// Convert to matrix representation (for small systems)
    pub fn to_matrix(&self) -> QuantRS2Result<Array2<Complex64>> {
        let dim = (self.max_photons_per_mode + 1).pow(self.num_modes as u32);
        let mut matrix = Array2::eye(dim);

        for gate in &self.gates {
            let gate_matrix = self.gate_to_matrix(gate)?;
            matrix = gate_matrix.dot(&matrix);
        }

        Ok(matrix)
    }

    /// Convert single gate to matrix (simplified)
    fn gate_to_matrix(&self, gate: &PhotonicGate) -> QuantRS2Result<Array2<Complex64>> {
        let dim = (self.max_photons_per_mode + 1).pow(self.num_modes as u32);

        match gate.gate_type {
            PhotonicGateType::PhaseShifter { phase } => {
                let mut matrix = Array2::eye(dim);
                let mode = gate.target_modes[0];

                // Apply phase to each Fock state
                for i in 0..dim {
                    let photon_count = self.extract_photon_count(i, mode);
                    let phase_factor = Complex64::new(0.0, (photon_count as f64) * phase).exp();
                    matrix[[i, i]] = phase_factor;
                }

                Ok(matrix)
            }
            _ => {
                // Simplified - return identity for complex gates
                Ok(Array2::eye(dim))
            }
        }
    }

    /// Extract photon count for specific mode from state index
    fn extract_photon_count(&self, state_index: usize, mode: usize) -> usize {
        let mode_dim = self.max_photons_per_mode + 1;
        (state_index / mode_dim.pow(mode as u32)) % mode_dim
    }
}

/// Photonic error correction
#[derive(Debug, Clone)]
pub struct PhotonicErrorCorrection {
    /// Number of physical modes per logical qubit
    pub physical_modes_per_logical: usize,
    /// Error correction code type
    pub code_type: String,
    /// Syndrome measurement circuits
    pub syndrome_circuits: Vec<PhotonicCircuit>,
}

impl PhotonicErrorCorrection {
    /// Create simple repetition code for photonic qubits
    pub fn repetition_code(num_repetitions: usize) -> Self {
        Self {
            physical_modes_per_logical: num_repetitions,
            code_type: "Repetition".to_string(),
            syndrome_circuits: Vec::new(),
        }
    }

    /// Encode logical qubit into physical modes
    pub fn encode_logical_qubit(
        &self,
        logical_state: &[Complex64; 2],
        system: &mut PhotonicSystem,
    ) -> QuantRS2Result<()> {
        // Simplified encoding - replicate logical state across physical modes
        for i in 0..self.physical_modes_per_logical {
            system.modes[i].fock_state[0] = logical_state[0];
            system.modes[i].fock_state[1] = logical_state[1];
        }

        system.initialize_from_modes()?;
        Ok(())
    }

    /// Perform error correction
    pub fn correct_errors(&self, system: &mut PhotonicSystem) -> QuantRS2Result<Vec<usize>> {
        // Simplified error correction - majority vote
        let mut corrections = Vec::new();

        // This is a placeholder for actual syndrome measurement and correction
        for i in 0..self.physical_modes_per_logical {
            let photon_prob = system.modes[i].photon_probability(1);
            if photon_prob > 0.5 {
                corrections.push(i);
            }
        }

        Ok(corrections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optical_mode_creation() {
        let vacuum = OpticalMode::vacuum(0, 5);
        assert_eq!(vacuum.mode_id, 0);
        assert_eq!(vacuum.max_photons, 5);
        assert!((vacuum.fock_state[0] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!(vacuum.photon_number_expectation() < 1e-10);

        let single_photon = OpticalMode::single_photon(1, 5);
        assert!((single_photon.photon_number_expectation() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coherent_state() {
        let alpha = Complex64::new(1.0, 0.0);
        let coherent = OpticalMode::coherent(0, 10, alpha);

        // For |α⟩ with α = 1, expectation value should be |α|² = 1
        assert!((coherent.photon_number_expectation() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_photonic_system_initialization() {
        let mut system = PhotonicSystem::new(3, 2);
        assert_eq!(system.num_modes, 3);
        assert_eq!(system.max_photons_per_mode, 2);

        system.initialize_from_modes().unwrap();
        assert!(system.state.is_some());
    }

    #[test]
    fn test_beam_splitter() {
        let mut system = PhotonicSystem::new(2, 3);

        // Put single photon in mode 0
        system.modes[0] = OpticalMode::single_photon(0, 3);
        system.initialize_from_modes().unwrap();

        // Apply 50-50 beam splitter
        system.apply_beam_splitter(0, 1, 0.5).unwrap();

        // Both modes should have some probability of containing the photon
        let prob0 = system.modes[0].photon_probability(1);
        let prob1 = system.modes[1].photon_probability(1);
        assert!(prob0 > 0.0);
        assert!(prob1 > 0.0);
    }

    #[test]
    fn test_phase_shifter() {
        let mut system = PhotonicSystem::new(1, 3);

        // Start with single photon
        system.modes[0] = OpticalMode::single_photon(0, 3);
        system.initialize_from_modes().unwrap();

        // Apply phase shift
        let phase = std::f64::consts::PI / 2.0;
        system.apply_phase_shifter(0, phase).unwrap();

        // Photon number should be unchanged
        assert!((system.modes[0].photon_number_expectation() - 1.0).abs() < 1e-10);

        // Phase should be applied
        let expected_phase = Complex64::new(0.0, phase).exp();
        assert!((system.modes[0].fock_state[1] - expected_phase).norm() < 1e-10);
    }

    #[test]
    fn test_photonic_gate_creation() {
        let bs_gate = PhotonicGate::beam_splitter(0, 1, 0.3);
        assert_eq!(bs_gate.target_modes, vec![0, 1]);
        assert!((bs_gate.parameters["reflectivity"] - 0.3).abs() < 1e-10);

        let ps_gate = PhotonicGate::phase_shifter(0, std::f64::consts::PI);
        assert_eq!(ps_gate.target_modes, vec![0]);
        assert!((ps_gate.parameters["phase"] - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_photonic_circuit() {
        let mut circuit = PhotonicCircuit::new(2, 2);

        let bs_gate = PhotonicGate::beam_splitter(0, 1, 0.5);
        let ps_gate = PhotonicGate::phase_shifter(0, std::f64::consts::PI / 4.0);

        circuit.add_gate(bs_gate).unwrap();
        circuit.add_gate(ps_gate).unwrap();

        assert_eq!(circuit.gates.len(), 2);

        let mut system = PhotonicSystem::new(2, 2);
        circuit.execute(&mut system).unwrap();
    }

    #[test]
    fn test_mach_zehnder_interferometer() {
        let mut system = PhotonicSystem::new(2, 2);

        // Single photon input
        system.modes[0] = OpticalMode::single_photon(0, 2);
        system.initialize_from_modes().unwrap();

        // Create Mach-Zehnder
        let mz_gate = PhotonicGate::mach_zehnder(0, 1, 0.0, std::f64::consts::PI);
        mz_gate.apply(&mut system).unwrap();

        // Check interference occurred
        let prob0 = system.modes[0].photon_probability(1);
        let prob1 = system.modes[1].photon_probability(1);
        assert!((prob0 + prob1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_photon_number_measurement() {
        let mut system = PhotonicSystem::new(1, 3);

        // Single photon state
        system.modes[0] = OpticalMode::single_photon(0, 3);
        system.initialize_from_modes().unwrap();

        let measurement_result = system.measure_photon_number(0).unwrap();

        // Should measure 1 photon (deterministic for pure state)
        assert_eq!(measurement_result, 1);

        // State should collapse to |1⟩
        assert!((system.modes[0].photon_probability(1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_homodyne_measurement() {
        let mut system = PhotonicSystem::new(1, 3);

        // Coherent state
        let alpha = Complex64::new(2.0, 0.0);
        system.modes[0] = OpticalMode::coherent(0, 3, alpha);
        system.initialize_from_modes().unwrap();

        let measurement_result = system.homodyne_measurement(0, 0.0).unwrap();

        // Should get real measurement result
        assert!(measurement_result.is_finite());
    }

    #[test]
    fn test_circuit_optimization() {
        let mut circuit = PhotonicCircuit::new(2, 2);

        // Add consecutive phase shifters
        circuit
            .add_gate(PhotonicGate::phase_shifter(0, std::f64::consts::PI / 4.0))
            .unwrap();
        circuit
            .add_gate(PhotonicGate::phase_shifter(0, std::f64::consts::PI / 4.0))
            .unwrap();
        circuit
            .add_gate(PhotonicGate::phase_shifter(0, std::f64::consts::PI / 2.0))
            .unwrap();

        assert_eq!(circuit.gates.len(), 3);

        circuit.optimize_linear_optics().unwrap();

        // Should be optimized to single phase shifter
        assert_eq!(circuit.gates.len(), 1);
        if let PhotonicGateType::PhaseShifter { phase } = circuit.gates[0].gate_type {
            assert!((phase - std::f64::consts::PI).abs() < 1e-10);
        }
    }

    #[test]
    fn test_photonic_error_correction() {
        let error_correction = PhotonicErrorCorrection::repetition_code(3);
        assert_eq!(error_correction.physical_modes_per_logical, 3);
        assert_eq!(error_correction.code_type, "Repetition");

        let mut system = PhotonicSystem::new(3, 2);
        let logical_state = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]; // |0⟩

        error_correction
            .encode_logical_qubit(&logical_state, &mut system)
            .unwrap();

        let corrections = error_correction.correct_errors(&mut system).unwrap();
        assert!(corrections.len() <= 3);
    }
}
