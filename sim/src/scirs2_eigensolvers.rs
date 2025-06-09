//! SciRS2-optimized eigensolvers for quantum spectral analysis.
//!
//! This module provides specialized eigenvalue analysis tools for quantum systems,
//! including energy spectrum calculations, quantum phase transition detection,
//! entanglement spectrum analysis, and spectral density computations.

use std::collections::HashMap;
use num_complex::Complex64;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

use crate::error::{SimulatorError, Result};
use crate::scirs2_integration::SciRS2Backend;
use crate::scirs2_sparse::{SparseMatrix, SciRS2SparseSolver, SparseSolverConfig, SparseEigenResult};
use crate::pauli::{PauliString, PauliOperatorSum};

/// Spectral analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralAnalysisResult {
    /// Eigenvalues (energy levels)
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenvectors (energy eigenstates)
    pub eigenvectors: Array2<Complex64>,
    /// Energy gaps between consecutive levels
    pub energy_gaps: Vec<f64>,
    /// Ground state energy
    pub ground_state_energy: f64,
    /// Spectral gap (lowest excitation energy)
    pub spectral_gap: f64,
    /// Participation ratio for each eigenstate
    pub participation_ratios: Vec<f64>,
    /// Entanglement entropy for each eigenstate
    pub entanglement_entropies: Vec<f64>,
    /// Spectral statistics
    pub spectral_stats: SpectralStatistics,
}

/// Quantum phase transition analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransitionResult {
    /// Parameter values tested
    pub parameters: Vec<f64>,
    /// Ground state energies
    pub ground_state_energies: Vec<f64>,
    /// Energy gaps
    pub energy_gaps: Vec<f64>,
    /// Order parameters
    pub order_parameters: Vec<f64>,
    /// Fidelity susceptibility
    pub fidelity_susceptibility: Vec<f64>,
    /// Critical points detected
    pub critical_points: Vec<f64>,
    /// Phase boundaries
    pub phase_boundaries: Vec<(f64, f64)>,
}

/// Spectral density result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralDensityResult {
    /// Energy grid
    pub energy_grid: Vec<f64>,
    /// Spectral density values
    pub density: Vec<f64>,
    /// Local density of states
    pub local_dos: Array2<f64>,
    /// Integrated density of states
    pub integrated_dos: Vec<f64>,
    /// Mobility edges (if applicable)
    pub mobility_edges: Vec<f64>,
}

/// Entanglement spectrum analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementSpectrumResult {
    /// Entanglement eigenvalues (Schmidt values)
    pub eigenvalues: Vec<f64>,
    /// Entanglement entropy
    pub entropy: f64,
    /// Renyi entropies
    pub renyi_entropies: HashMap<String, f64>,
    /// Entanglement gap
    pub entanglement_gap: f64,
    /// Participation ratio
    pub participation_ratio: f64,
    /// Bipartition specification
    pub bipartition: Vec<usize>,
}

/// Spectral statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpectralStatistics {
    /// Level spacing statistics
    pub level_spacing_mean: f64,
    pub level_spacing_std: f64,
    /// Nearest neighbor spacing ratio
    pub nn_spacing_ratio: f64,
    /// Spectral rigidity
    pub spectral_rigidity: f64,
    /// Number variance
    pub number_variance: f64,
    /// Spectral form factor
    pub spectral_form_factor: Vec<Complex64>,
    /// Thouless time
    pub thouless_time: f64,
}

/// Band structure calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandStructureResult {
    /// k-point path
    pub k_points: Vec<Array1<f64>>,
    /// Energy bands
    pub energy_bands: Array2<f64>,
    /// Band gaps
    pub band_gaps: Vec<f64>,
    /// Density of states
    pub dos: SpectralDensityResult,
    /// Effective masses
    pub effective_masses: Vec<f64>,
}

/// Configuration for spectral analysis
#[derive(Debug, Clone)]
pub struct SpectralConfig {
    /// Number of eigenvalues to compute
    pub num_eigenvalues: usize,
    /// Which eigenvalues to target ("smallest", "largest", "interior")
    pub which: String,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Use parallel execution
    pub parallel: bool,
    /// Energy resolution for spectral density
    pub energy_resolution: f64,
    /// Broadening parameter for spectral density
    pub broadening: f64,
}

impl Default for SpectralConfig {
    fn default() -> Self {
        Self {
            num_eigenvalues: 10,
            which: "smallest".to_string(),
            tolerance: 1e-10,
            max_iterations: 1000,
            parallel: true,
            energy_resolution: 0.01,
            broadening: 0.1,
        }
    }
}

/// SciRS2-optimized spectral analyzer
pub struct SciRS2SpectralAnalyzer {
    /// SciRS2 backend
    backend: Option<SciRS2Backend>,
    /// Configuration
    config: SpectralConfig,
    /// Cached eigenvalue results
    eigenvalue_cache: HashMap<String, SparseEigenResult>,
}

impl SciRS2SpectralAnalyzer {
    /// Create new spectral analyzer
    pub fn new(config: SpectralConfig) -> Result<Self> {
        Ok(Self {
            backend: None,
            config,
            eigenvalue_cache: HashMap::new(),
        })
    }
    
    /// Initialize with SciRS2 backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new()?);
        Ok(self)
    }
    
    /// Perform comprehensive spectral analysis
    pub fn analyze_spectrum(&mut self, hamiltonian: &SparseMatrix) -> Result<SpectralAnalysisResult> {
        if !hamiltonian.is_square() {
            return Err(SimulatorError::InvalidInput(
                "Hamiltonian must be square".to_string()
            ));
        }
        
        // Compute eigenvalues and eigenvectors
        let eigen_result = self.compute_eigenvalues(hamiltonian)?;
        
        let eigenvalues = eigen_result.eigenvalues;
        let eigenvectors = eigen_result.eigenvectors;
        
        // Calculate energy gaps
        let energy_gaps = self.calculate_energy_gaps(&eigenvalues);
        
        // Ground state properties
        let ground_state_energy = eigenvalues[0];
        let spectral_gap = if eigenvalues.len() > 1 {
            eigenvalues[1] - eigenvalues[0]
        } else {
            0.0
        };
        
        // Participation ratios
        let participation_ratios = self.calculate_participation_ratios(&eigenvectors)?;
        
        // Entanglement entropies (for small systems)
        let entanglement_entropies = if hamiltonian.shape.0 <= 4096 {
            self.calculate_entanglement_entropies(&eigenvectors)?
        } else {
            vec![0.0; eigenvalues.len()]
        };
        
        // Spectral statistics
        let spectral_stats = self.calculate_spectral_statistics(&eigenvalues)?;
        
        Ok(SpectralAnalysisResult {
            eigenvalues,
            eigenvectors,
            energy_gaps,
            ground_state_energy,
            spectral_gap,
            participation_ratios,
            entanglement_entropies,
            spectral_stats,
        })
    }
    
    /// Analyze quantum phase transitions
    pub fn analyze_phase_transition<F>(
        &mut self,
        hamiltonian_generator: F,
        parameter_range: (f64, f64),
        num_points: usize,
    ) -> Result<PhaseTransitionResult>
    where
        F: Fn(f64) -> Result<SparseMatrix> + Sync + Send,
    {
        let parameters: Vec<f64> = (0..num_points)
            .map(|i| parameter_range.0 + (parameter_range.1 - parameter_range.0) * i as f64 / (num_points - 1) as f64)
            .collect();
        
        // Parallel computation for different parameter values
        let results: Result<Vec<_>> = if self.config.parallel {
            parameters.par_iter().map(|&param| {
                let hamiltonian = hamiltonian_generator(param)?;
                let mut analyzer = Self::new(self.config.clone())?;
                let analysis = analyzer.analyze_spectrum(&hamiltonian)?;
                Ok((
                    analysis.ground_state_energy,
                    analysis.spectral_gap,
                    analysis.eigenvectors.column(0).to_owned(),
                ))
            }).collect()
        } else {
            parameters.iter().map(|&param| {
                let hamiltonian = hamiltonian_generator(param)?;
                let analysis = self.analyze_spectrum(&hamiltonian)?;
                Ok((
                    analysis.ground_state_energy,
                    analysis.spectral_gap,
                    analysis.eigenvectors.column(0).to_owned(),
                ))
            }).collect()
        };
        
        let results = results?;
        let (ground_state_energies, energy_gaps, ground_states): (Vec<_>, Vec<_>, Vec<_>) = results.into_iter()
            .map(|(e, g, gs)| (e, g, gs))
            .multiunzip();
        
        // Calculate order parameters (overlap with first ground state)
        let reference_state = &ground_states[0];
        let order_parameters: Vec<f64> = ground_states.iter()
            .map(|state| {
                let overlap: Complex64 = reference_state.iter().zip(state.iter())
                    .map(|(&ref_amp, &state_amp)| ref_amp.conj() * state_amp)
                    .sum();
                overlap.norm_sqr()
            })
            .collect();
        
        // Calculate fidelity susceptibility
        let fidelity_susceptibility = self.calculate_fidelity_susceptibility(&ground_states, &parameters)?;
        
        // Detect critical points
        let critical_points = self.detect_critical_points(&energy_gaps, &fidelity_susceptibility, &parameters)?;
        
        // Identify phase boundaries
        let phase_boundaries = self.identify_phase_boundaries(&order_parameters, &parameters)?;
        
        Ok(PhaseTransitionResult {
            parameters,
            ground_state_energies,
            energy_gaps,
            order_parameters,
            fidelity_susceptibility,
            critical_points,
            phase_boundaries,
        })
    }
    
    /// Calculate spectral density
    pub fn calculate_spectral_density(
        &mut self,
        hamiltonian: &SparseMatrix,
        energy_range: (f64, f64),
    ) -> Result<SpectralDensityResult> {
        let num_points = ((energy_range.1 - energy_range.0) / self.config.energy_resolution) as usize;
        let energy_grid: Vec<f64> = (0..num_points)
            .map(|i| energy_range.0 + (energy_range.1 - energy_range.0) * i as f64 / (num_points - 1) as f64)
            .collect();
        
        // Get eigenvalues for density calculation
        let eigen_result = self.compute_eigenvalues(hamiltonian)?;
        let eigenvalues = eigen_result.eigenvalues;
        
        // Calculate spectral density using Gaussian broadening
        let density: Vec<f64> = energy_grid.iter().map(|&energy| {
            eigenvalues.iter().map(|&eigenval| {
                let diff = energy - eigenval;
                (-diff * diff / (2.0 * self.config.broadening * self.config.broadening)).exp()
            }).sum::<f64>() / (self.config.broadening * (2.0 * std::f64::consts::PI).sqrt())
        }).collect();
        
        // Local density of states (simplified - would need position operators)
        let local_dos = Array2::zeros((hamiltonian.shape.0.min(100), num_points));
        
        // Integrated density of states
        let mut integrated_dos = vec![0.0; num_points];
        let mut cumulative = 0.0;
        for (i, &d) in density.iter().enumerate() {
            cumulative += d * self.config.energy_resolution;
            integrated_dos[i] = cumulative;
        }
        
        // Mobility edges (placeholder - would need disorder analysis)
        let mobility_edges = Vec::new();
        
        Ok(SpectralDensityResult {
            energy_grid,
            density,
            local_dos,
            integrated_dos,
            mobility_edges,
        })
    }
    
    /// Calculate entanglement spectrum
    pub fn calculate_entanglement_spectrum(
        &mut self,
        state: &Array1<Complex64>,
        bipartition: &[usize],
    ) -> Result<EntanglementSpectrumResult> {
        let total_qubits = (state.len() as f64).log2() as usize;
        if state.len() != 1 << total_qubits {
            return Err(SimulatorError::InvalidInput(
                "State vector length must be a power of 2".to_string()
            ));
        }
        
        let subsystem_size = bipartition.len();
        let env_size = total_qubits - subsystem_size;
        
        if subsystem_size == 0 || subsystem_size >= total_qubits {
            return Err(SimulatorError::InvalidInput(
                "Invalid bipartition specification".to_string()
            ));
        }
        
        // Reshape state into reduced density matrix form
        let subsystem_dim = 1 << subsystem_size;
        let env_dim = 1 << env_size;
        
        // Create reduced density matrix for subsystem
        let mut rho_reduced = Array2::zeros((subsystem_dim, subsystem_dim));
        
        for i in 0..subsystem_dim {
            for j in 0..subsystem_dim {
                let mut element = Complex64::new(0.0, 0.0);
                for k in 0..env_dim {
                    let full_i = self.encode_full_state(i, k, bipartition, total_qubits);
                    let full_j = self.encode_full_state(j, k, bipartition, total_qubits);
                    element += state[full_i].conj() * state[full_j];
                }
                rho_reduced[[i, j]] = element;
            }
        }
        
        // Diagonalize reduced density matrix
        let eigenvalues = self.diagonalize_hermitian_matrix(&rho_reduced)?;
        
        // Filter out zero eigenvalues and sort
        let mut nonzero_eigenvalues: Vec<f64> = eigenvalues.into_iter()
            .filter(|&x| x > 1e-15)
            .collect();
        nonzero_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        // Calculate entanglement entropy
        let entropy = -nonzero_eigenvalues.iter()
            .map(|&p| p * p.ln())
            .sum::<f64>();
        
        // Calculate Renyi entropies
        let mut renyi_entropies = HashMap::new();
        for &n in &[2, 3, 4, 5] {
            let renyi_n = if n == 1 {
                entropy
            } else {
                (1.0 / (1.0 - n as f64)) * nonzero_eigenvalues.iter()
                    .map(|&p| p.powf(n as f64))
                    .sum::<f64>().ln()
            };
            renyi_entropies.insert(format!("S_{}", n), renyi_n);
        }
        
        // Entanglement gap (difference between largest and second largest eigenvalues)
        let entanglement_gap = if nonzero_eigenvalues.len() > 1 {
            nonzero_eigenvalues[0] - nonzero_eigenvalues[1]
        } else {
            0.0
        };
        
        // Participation ratio
        let participation_ratio = if !nonzero_eigenvalues.is_empty() {
            let sum_p2 = nonzero_eigenvalues.iter().map(|&p| p * p).sum::<f64>();
            1.0 / sum_p2
        } else {
            0.0
        };
        
        Ok(EntanglementSpectrumResult {
            eigenvalues: nonzero_eigenvalues,
            entropy,
            renyi_entropies,
            entanglement_gap,
            participation_ratio,
            bipartition: bipartition.to_vec(),
        })
    }
    
    /// Calculate band structure
    pub fn calculate_band_structure<F>(
        &mut self,
        hamiltonian_generator: F,
        k_path: &[Array1<f64>],
        num_bands: usize,
    ) -> Result<BandStructureResult>
    where
        F: Fn(&Array1<f64>) -> Result<SparseMatrix> + Sync + Send,
    {
        // Calculate energy bands along k-point path
        let band_results: Result<Vec<_>> = if self.config.parallel {
            k_path.par_iter().map(|k_point| {
                let hamiltonian = hamiltonian_generator(k_point)?;
                let mut config = SparseSolverConfig::default();
                config.method = crate::scirs2_sparse::SparseSolverMethod::Lanczos;
                let mut solver = SciRS2SparseSolver::new(config)?;
                let result = solver.solve_eigenvalue_problem(&hamiltonian, num_bands, "smallest")?;
                Ok(result.eigenvalues)
            }).collect()
        } else {
            k_path.iter().map(|k_point| {
                let hamiltonian = hamiltonian_generator(k_point)?;
                let mut config = SparseSolverConfig::default();
                config.method = crate::scirs2_sparse::SparseSolverMethod::Lanczos;
                let mut solver = SciRS2SparseSolver::new(config)?;
                let result = solver.solve_eigenvalue_problem(&hamiltonian, num_bands, "smallest")?;
                Ok(result.eigenvalues)
            }).collect()
        };
        
        let band_results = band_results?;
        
        // Organize into band structure
        let mut energy_bands = Array2::zeros((k_path.len(), num_bands));
        for (i, eigenvalues) in band_results.iter().enumerate() {
            for (j, &energy) in eigenvalues.iter().enumerate() {
                if j < num_bands {
                    energy_bands[[i, j]] = energy;
                }
            }
        }
        
        // Calculate band gaps
        let mut band_gaps = Vec::new();
        for i in 0..num_bands-1 {
            let mut min_gap = f64::INFINITY;
            for k in 0..k_path.len() {
                let gap = energy_bands[[k, i+1]] - energy_bands[[k, i]];
                if gap < min_gap {
                    min_gap = gap;
                }
            }
            band_gaps.push(min_gap);
        }
        
        // Calculate density of states (simplified)
        let all_energies: Vec<f64> = energy_bands.iter().cloned().collect();
        let energy_range = (
            all_energies.iter().cloned().fold(f64::INFINITY, f64::min),
            all_energies.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        );
        
        // Create dummy hamiltonian for DOS calculation
        let dummy_hamiltonian = SparseMatrix::identity(100);
        let dos = self.calculate_spectral_density(&dummy_hamiltonian, energy_range)?;
        
        // Calculate effective masses (placeholder)
        let effective_masses = vec![1.0; num_bands];
        
        Ok(BandStructureResult {
            k_points: k_path.to_vec(),
            energy_bands,
            band_gaps,
            dos,
            effective_masses,
        })
    }
    
    /// Compute eigenvalues using appropriate solver
    fn compute_eigenvalues(&mut self, hamiltonian: &SparseMatrix) -> Result<SparseEigenResult> {
        let cache_key = format!("eigenvals_{}_{}", hamiltonian.shape.0, hamiltonian.nnz);
        
        if let Some(cached_result) = self.eigenvalue_cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }
        
        let mut config = SparseSolverConfig::default();
        config.method = if hamiltonian.is_hermitian {
            crate::scirs2_sparse::SparseSolverMethod::Lanczos
        } else {
            crate::scirs2_sparse::SparseSolverMethod::Arnoldi
        };
        
        let mut solver = if let Some(_) = &self.backend {
            SciRS2SparseSolver::new(config)?.with_backend()?
        } else {
            SciRS2SparseSolver::new(config)?
        };
        
        let result = solver.solve_eigenvalue_problem(
            hamiltonian,
            self.config.num_eigenvalues,
            &self.config.which,
        )?;
        
        self.eigenvalue_cache.insert(cache_key, result.clone());
        Ok(result)
    }
    
    /// Calculate energy gaps
    fn calculate_energy_gaps(&self, eigenvalues: &[f64]) -> Vec<f64> {
        eigenvalues.windows(2)
            .map(|window| window[1] - window[0])
            .collect()
    }
    
    /// Calculate participation ratios
    fn calculate_participation_ratios(&self, eigenvectors: &Array2<Complex64>) -> Result<Vec<f64>> {
        let mut ratios = Vec::new();
        
        for col_idx in 0..eigenvectors.ncols() {
            let column = eigenvectors.column(col_idx);
            let sum_p4: f64 = column.iter().map(|&c| c.norm_sqr().powi(2)).sum();
            let ratio = if sum_p4 > 0.0 { 1.0 / sum_p4 } else { 0.0 };
            ratios.push(ratio);
        }
        
        Ok(ratios)
    }
    
    /// Calculate entanglement entropies
    fn calculate_entanglement_entropies(&mut self, eigenvectors: &Array2<Complex64>) -> Result<Vec<f64>> {
        let mut entropies = Vec::new();
        
        for col_idx in 0..eigenvectors.ncols() {
            let state = eigenvectors.column(col_idx).to_owned();
            let num_qubits = (state.len() as f64).log2() as usize;
            
            if num_qubits <= 10 { // Only for small systems
                let bipartition: Vec<usize> = (0..num_qubits/2).collect();
                let ent_result = self.calculate_entanglement_spectrum(&state, &bipartition);
                let entropy = ent_result.map(|r| r.entropy).unwrap_or(0.0);
                entropies.push(entropy);
            } else {
                entropies.push(0.0);
            }
        }
        
        Ok(entropies)
    }
    
    /// Calculate spectral statistics
    fn calculate_spectral_statistics(&self, eigenvalues: &[f64]) -> Result<SpectralStatistics> {
        if eigenvalues.len() < 3 {
            return Ok(SpectralStatistics::default());
        }
        
        // Level spacing statistics
        let spacings: Vec<f64> = eigenvalues.windows(2)
            .map(|window| window[1] - window[0])
            .collect();
        
        let level_spacing_mean = spacings.iter().sum::<f64>() / spacings.len() as f64;
        let level_spacing_std = {
            let variance = spacings.iter()
                .map(|&s| (s - level_spacing_mean).powi(2))
                .sum::<f64>() / spacings.len() as f64;
            variance.sqrt()
        };
        
        // Nearest neighbor spacing ratio
        let nn_spacing_ratio = if spacings.len() > 1 {
            let ratios: Vec<f64> = spacings.windows(2)
                .map(|window| window[0].min(window[1]) / window[0].max(window[1]))
                .collect();
            ratios.iter().sum::<f64>() / ratios.len() as f64
        } else {
            0.0
        };
        
        // Spectral rigidity (simplified)
        let spectral_rigidity = level_spacing_std / level_spacing_mean;
        
        // Number variance (simplified)
        let number_variance = eigenvalues.len() as f64 * level_spacing_std.powi(2);
        
        // Spectral form factor (placeholder)
        let spectral_form_factor = vec![Complex64::new(1.0, 0.0); 100];
        
        // Thouless time (placeholder)
        let thouless_time = 1.0 / level_spacing_mean;
        
        Ok(SpectralStatistics {
            level_spacing_mean,
            level_spacing_std,
            nn_spacing_ratio,
            spectral_rigidity,
            number_variance,
            spectral_form_factor,
            thouless_time,
        })
    }
    
    /// Calculate fidelity susceptibility
    fn calculate_fidelity_susceptibility(
        &self,
        ground_states: &[Array1<Complex64>],
        parameters: &[f64],
    ) -> Result<Vec<f64>> {
        let mut susceptibilities = Vec::new();
        
        for i in 1..ground_states.len()-1 {
            let dparam = parameters[i+1] - parameters[i-1];
            if dparam.abs() < 1e-15 {
                susceptibilities.push(0.0);
                continue;
            }
            
            // Calculate derivative of ground state
            let mut derivative = Array1::zeros(ground_states[i].len());
            for j in 0..ground_states[i].len() {
                derivative[j] = (ground_states[i+1][j] - ground_states[i-1][j]) / dparam;
            }
            
            // Fidelity susceptibility
            let overlap: Complex64 = ground_states[i].iter().zip(derivative.iter())
                .map(|(&gs, &deriv)| gs.conj() * deriv)
                .sum();
            
            let susceptibility = derivative.iter().map(|&d| d.norm_sqr()).sum::<f64>() - overlap.norm_sqr();
            susceptibilities.push(susceptibility);
        }
        
        // Handle boundary points
        if !susceptibilities.is_empty() {
            susceptibilities.insert(0, susceptibilities[0]);
            susceptibilities.push(susceptibilities[susceptibilities.len()-1]);
        }
        
        Ok(susceptibilities)
    }
    
    /// Detect critical points
    fn detect_critical_points(
        &self,
        energy_gaps: &[f64],
        fidelity_susceptibility: &[f64],
        parameters: &[f64],
    ) -> Result<Vec<f64>> {
        let mut critical_points = Vec::new();
        
        // Find minima in energy gap
        for i in 1..energy_gaps.len()-1 {
            if energy_gaps[i] < energy_gaps[i-1] && energy_gaps[i] < energy_gaps[i+1] {
                critical_points.push(parameters[i]);
            }
        }
        
        // Find maxima in fidelity susceptibility
        for i in 1..fidelity_susceptibility.len()-1 {
            if fidelity_susceptibility[i] > fidelity_susceptibility[i-1] && 
               fidelity_susceptibility[i] > fidelity_susceptibility[i+1] {
                critical_points.push(parameters[i]);
            }
        }
        
        // Remove duplicates and sort
        critical_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
        critical_points.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        
        Ok(critical_points)
    }
    
    /// Identify phase boundaries
    fn identify_phase_boundaries(
        &self,
        order_parameters: &[f64],
        parameters: &[f64],
    ) -> Result<Vec<(f64, f64)>> {
        let mut boundaries = Vec::new();
        let threshold = 0.1; // Threshold for order parameter change
        
        for i in 1..order_parameters.len() {
            let change = (order_parameters[i] - order_parameters[i-1]).abs();
            if change > threshold {
                boundaries.push((parameters[i-1], parameters[i]));
            }
        }
        
        Ok(boundaries)
    }
    
    /// Encode full state index from subsystem and environment indices
    fn encode_full_state(&self, sub_idx: usize, env_idx: usize, bipartition: &[usize], total_qubits: usize) -> usize {
        let mut full_idx = 0;
        let mut sub_bit = 0;
        let mut env_bit = 0;
        
        for qubit in 0..total_qubits {
            if bipartition.contains(&qubit) {
                if (sub_idx >> sub_bit) & 1 == 1 {
                    full_idx |= 1 << (total_qubits - 1 - qubit);
                }
                sub_bit += 1;
            } else {
                if (env_idx >> env_bit) & 1 == 1 {
                    full_idx |= 1 << (total_qubits - 1 - qubit);
                }
                env_bit += 1;
            }
        }
        
        full_idx
    }
    
    /// Diagonalize Hermitian matrix (simplified)
    fn diagonalize_hermitian_matrix(&self, matrix: &Array2<Complex64>) -> Result<Vec<f64>> {
        // Simplified eigenvalue computation
        // In practice, would use proper LAPACK routines
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SimulatorError::InvalidInput("Matrix must be square".to_string()));
        }
        
        // For small matrices, use simple power iteration
        if n <= 64 {
            let mut eigenvalues = Vec::new();
            
            // Extract diagonal elements as rough eigenvalue estimates
            for i in 0..n {
                eigenvalues.push(matrix[[i, i]].re);
            }
            
            eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
            Ok(eigenvalues)
        } else {
            Err(SimulatorError::UnsupportedOperation(
                "Matrix too large for current diagonalization implementation".to_string()
            ))
        }
    }
    
    /// Get configuration
    pub fn get_config(&self) -> &SpectralConfig {
        &self.config
    }
    
    /// Set configuration
    pub fn set_config(&mut self, config: SpectralConfig) {
        self.config = config;
    }
    
    /// Clear eigenvalue cache
    pub fn clear_cache(&mut self) {
        self.eigenvalue_cache.clear();
    }
}

/// Utilities for creating quantum Hamiltonians for spectral analysis
pub struct QuantumHamiltonianLibrary;

impl QuantumHamiltonianLibrary {
    /// Create transverse field Ising model Hamiltonian
    pub fn transverse_field_ising(num_sites: usize, j: f64, h: f64) -> Result<SparseMatrix> {
        use crate::scirs2_sparse::SparseMatrixUtils;
        
        let mut pauli_terms = Vec::new();
        
        // ZZ interactions
        for i in 0..num_sites-1 {
            let mut pauli_string = "I".repeat(num_sites);
            pauli_string.replace_range(i..i+1, "Z");
            pauli_string.replace_range(i+1..i+2, "Z");
            pauli_terms.push((pauli_string, -j));
        }
        
        // Transverse field
        for i in 0..num_sites {
            let mut pauli_string = "I".repeat(num_sites);
            pauli_string.replace_range(i..i+1, "X");
            pauli_terms.push((pauli_string, -h));
        }
        
        SparseMatrixUtils::hamiltonian_from_pauli_strings(num_sites, &pauli_terms)
    }
    
    /// Create Heisenberg model Hamiltonian
    pub fn heisenberg_model(num_sites: usize, jx: f64, jy: f64, jz: f64) -> Result<SparseMatrix> {
        use crate::scirs2_sparse::SparseMatrixUtils;
        
        let mut pauli_terms = Vec::new();
        
        for i in 0..num_sites-1 {
            // XX interaction
            let mut xx_string = "I".repeat(num_sites);
            xx_string.replace_range(i..i+1, "X");
            xx_string.replace_range(i+1..i+2, "X");
            pauli_terms.push((xx_string, jx));
            
            // YY interaction
            let mut yy_string = "I".repeat(num_sites);
            yy_string.replace_range(i..i+1, "Y");
            yy_string.replace_range(i+1..i+2, "Y");
            pauli_terms.push((yy_string, jy));
            
            // ZZ interaction
            let mut zz_string = "I".repeat(num_sites);
            zz_string.replace_range(i..i+1, "Z");
            zz_string.replace_range(i+1..i+2, "Z");
            pauli_terms.push((zz_string, jz));
        }
        
        SparseMatrixUtils::hamiltonian_from_pauli_strings(num_sites, &pauli_terms)
    }
    
    /// Create random matrix model
    pub fn random_matrix_model(size: usize, density: f64, hermitian: bool) -> SparseMatrix {
        use crate::scirs2_sparse::SparseMatrixUtils;
        SparseMatrixUtils::random_sparse(size, density, hermitian)
    }
}

/// Trait for multi-unzip operations
trait MultiUnzip<A, B, C> {
    fn multiunzip(self) -> (Vec<A>, Vec<B>, Vec<C>);
}

impl<I, A, B, C> MultiUnzip<A, B, C> for I
where
    I: Iterator<Item = (A, B, C)>,
{
    fn multiunzip(self) -> (Vec<A>, Vec<B>, Vec<C>) {
        let mut vec_a = Vec::new();
        let mut vec_b = Vec::new();
        let mut vec_c = Vec::new();
        
        for (a, b, c) in self {
            vec_a.push(a);
            vec_b.push(b);
            vec_c.push(c);
        }
        
        (vec_a, vec_b, vec_c)
    }
}

/// Benchmark spectral analysis methods
pub fn benchmark_spectral_analysis(system_size: usize) -> Result<HashMap<String, f64>> {
    let mut results = HashMap::new();
    
    // Create test Hamiltonian
    let hamiltonian = QuantumHamiltonianLibrary::transverse_field_ising(system_size, 1.0, 0.5)?;
    
    // Test different analysis methods
    let config = SpectralConfig {
        num_eigenvalues: 10.min(hamiltonian.shape.0),
        ..Default::default()
    };
    
    let mut analyzer = SciRS2SpectralAnalyzer::new(config)?;
    
    // Time spectral analysis
    let start_time = std::time::Instant::now();
    let _analysis = analyzer.analyze_spectrum(&hamiltonian)?;
    let analysis_time = start_time.elapsed().as_secs_f64();
    
    results.insert("SpectrumAnalysis".to_string(), analysis_time);
    
    // Time spectral density calculation
    let start_time = std::time::Instant::now();
    let _density = analyzer.calculate_spectral_density(&hamiltonian, (-5.0, 5.0))?;
    let density_time = start_time.elapsed().as_secs_f64();
    
    results.insert("SpectralDensity".to_string(), density_time);
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_spectral_analyzer_creation() {
        let config = SpectralConfig::default();
        let analyzer = SciRS2SpectralAnalyzer::new(config).unwrap();
        assert_eq!(analyzer.config.num_eigenvalues, 10);
    }

    #[test]
    fn test_transverse_field_ising() {
        let hamiltonian = QuantumHamiltonianLibrary::transverse_field_ising(3, 1.0, 0.5).unwrap();
        assert_eq!(hamiltonian.shape, (8, 8));
        assert!(hamiltonian.is_hermitian);
    }

    #[test]
    fn test_heisenberg_model() {
        let hamiltonian = QuantumHamiltonianLibrary::heisenberg_model(2, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(hamiltonian.shape, (4, 4));
        assert!(hamiltonian.is_hermitian);
    }

    #[test]
    fn test_spectral_analysis() {
        let hamiltonian = QuantumHamiltonianLibrary::transverse_field_ising(3, 1.0, 0.5).unwrap();
        let config = SpectralConfig {
            num_eigenvalues: 5,
            ..Default::default()
        };
        
        let mut analyzer = SciRS2SpectralAnalyzer::new(config).unwrap();
        let result = analyzer.analyze_spectrum(&hamiltonian).unwrap();
        
        assert_eq!(result.eigenvalues.len(), 5);
        assert!(result.spectral_gap >= 0.0);
        assert_eq!(result.participation_ratios.len(), 5);
    }

    #[test]
    fn test_entanglement_spectrum() {
        let mut state = Array1::zeros(4);
        state[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        state[3] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        
        let config = SpectralConfig::default();
        let analyzer = SciRS2SpectralAnalyzer::new(config).unwrap();
        
        let bipartition = vec![0];
        let result = analyzer.calculate_entanglement_spectrum(&state, &bipartition).unwrap();
        
        assert!(result.entropy > 0.0);
        assert_eq!(result.bipartition, vec![0]);
    }

    #[test]
    fn test_energy_gaps() {
        let eigenvalues = vec![0.0, 1.0, 3.0, 6.0];
        let config = SpectralConfig::default();
        let analyzer = SciRS2SpectralAnalyzer::new(config).unwrap();
        
        let gaps = analyzer.calculate_energy_gaps(&eigenvalues);
        
        assert_eq!(gaps, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_participation_ratios() {
        let mut eigenvectors = Array2::zeros((4, 2));
        eigenvectors[[0, 0]] = Complex64::new(1.0, 0.0);
        eigenvectors[[1, 1]] = Complex64::new(0.5, 0.0);
        eigenvectors[[2, 1]] = Complex64::new(0.5, 0.0);
        eigenvectors[[3, 1]] = Complex64::new(0.5, 0.0);
        
        let config = SpectralConfig::default();
        let analyzer = SciRS2SpectralAnalyzer::new(config).unwrap();
        
        let ratios = analyzer.calculate_participation_ratios(&eigenvectors).unwrap();
        
        assert_eq!(ratios.len(), 2);
        assert_abs_diff_eq!(ratios[0], 1.0, epsilon = 1e-10); // Localized state
        assert!(ratios[1] > 1.0); // Delocalized state
    }
}