//! Quantum Field Theory Simulation Framework
//!
//! This module provides a comprehensive implementation of quantum field theory simulations,
//! including field operators, path integrals, lattice gauge theories, renormalization
//! group flows, and scattering process calculations. This framework enables simulation
//! of relativistic quantum field dynamics and many-body quantum systems.

use scirs2_core::ndarray::{Array1, Array4};
use scirs2_core::random::{thread_rng, Rng};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use scirs2_core::random::prelude::*;

/// Quantum field theory simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QFTConfig {
    /// Spacetime dimensions
    pub spacetime_dimensions: usize,
    /// Lattice size in each dimension
    pub lattice_size: Vec<usize>,
    /// Lattice spacing
    pub lattice_spacing: f64,
    /// Field theory type
    pub field_theory: FieldTheoryType,
    /// Boundary conditions
    pub boundary_conditions: QFTBoundaryConditions,
    /// Temperature for thermal field theory
    pub temperature: f64,
    /// Chemical potential
    pub chemical_potential: f64,
    /// Coupling constants
    pub coupling_constants: HashMap<String, f64>,
    /// Enable gauge invariance
    pub gauge_invariant: bool,
    /// Renormalization scheme
    pub renormalization_scheme: RenormalizationScheme,
    /// Path integral Monte Carlo steps
    pub mc_steps: usize,
}

impl Default for QFTConfig {
    fn default() -> Self {
        let mut couplings = HashMap::new();
        couplings.insert("g".to_string(), 0.1);
        couplings.insert("lambda".to_string(), 0.01);

        Self {
            spacetime_dimensions: 4,
            lattice_size: vec![16, 16, 16, 32],
            lattice_spacing: 1.0,
            field_theory: FieldTheoryType::ScalarPhi4,
            boundary_conditions: QFTBoundaryConditions::Periodic,
            temperature: 0.0,
            chemical_potential: 0.0,
            coupling_constants: couplings,
            gauge_invariant: true,
            renormalization_scheme: RenormalizationScheme::DimensionalRegularization,
            mc_steps: 10000,
        }
    }
}

/// Types of quantum field theories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldTheoryType {
    /// Scalar φ⁴ theory
    ScalarPhi4,
    /// Quantum Electrodynamics (QED)
    QED,
    /// Yang-Mills gauge theory
    YangMills,
    /// Quantum Chromodynamics (QCD)
    QCD,
    /// Chiral fermions
    ChiralFermions,
    /// Higgs field theory
    Higgs,
    /// Standard Model
    StandardModel,
    /// Non-linear sigma model
    NonLinearSigma,
    /// Gross-Neveu model
    GrossNeveu,
    /// Sine-Gordon model
    SineGordon,
    /// Custom field theory
    Custom,
}

/// Boundary conditions for field theory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QFTBoundaryConditions {
    /// Periodic boundary conditions
    Periodic,
    /// Antiperiodic boundary conditions (for fermions)
    Antiperiodic,
    /// Open boundary conditions
    Open,
    /// Twisted boundary conditions
    Twisted,
    /// Mixed boundary conditions
    Mixed,
}

/// Renormalization schemes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenormalizationScheme {
    /// Dimensional regularization
    DimensionalRegularization,
    /// Pauli-Villars regularization
    PauliVillars,
    /// Momentum cutoff
    MomentumCutoff,
    /// Zeta function regularization
    ZetaFunction,
    /// MS-bar scheme
    MSBar,
    /// On-shell scheme
    OnShell,
}

/// Field operator types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldOperatorType {
    /// Scalar field
    Scalar,
    /// Spinor field (fermion)
    Spinor,
    /// Vector field (gauge field)
    Vector,
    /// Tensor field
    Tensor,
    /// Creation operator
    Creation,
    /// Annihilation operator
    Annihilation,
}

/// Quantum field operator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldOperator {
    /// Field type
    pub field_type: FieldOperatorType,
    /// Spacetime position
    pub position: Vec<f64>,
    /// Momentum representation
    pub momentum: Option<Vec<f64>>,
    /// Field component index (for multi-component fields)
    pub component: usize,
    /// Time ordering
    pub time_ordering: TimeOrdering,
    /// Normal ordering coefficient
    pub normal_ordering: bool,
}

/// Time ordering for field operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeOrdering {
    /// Time-ordered product
    TimeOrdered,
    /// Normal-ordered product
    NormalOrdered,
    /// Anti-time-ordered product
    AntiTimeOrdered,
    /// Causal ordering
    Causal,
}

/// Path integral configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathIntegralConfig {
    /// Number of time slices
    pub time_slices: usize,
    /// Time step size
    pub time_step: f64,
    /// Action type
    pub action_type: ActionType,
    /// Monte Carlo algorithm
    pub mc_algorithm: MonteCarloAlgorithm,
    /// Importance sampling
    pub importance_sampling: bool,
    /// Metropolis acceptance rate target
    pub target_acceptance_rate: f64,
}

/// Action types for path integrals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    /// Euclidean action
    Euclidean,
    /// Minkowski action
    Minkowski,
    /// Wick-rotated action
    WickRotated,
    /// Effective action
    Effective,
    /// Wilson action (lattice)
    Wilson,
    /// Improved action
    Improved,
}

/// Monte Carlo algorithms for path integrals
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MonteCarloAlgorithm {
    /// Metropolis-Hastings
    Metropolis,
    /// Langevin dynamics
    Langevin,
    /// Hybrid Monte Carlo
    HybridMonteCarlo,
    /// Cluster algorithms
    Cluster,
    /// Worm algorithms
    Worm,
    /// Multi-canonical sampling
    Multicanonical,
}

/// Gauge field configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeFieldConfig {
    /// Gauge group (SU(N), U(1), etc.)
    pub gauge_group: GaugeGroup,
    /// Number of colors (for non-Abelian groups)
    pub num_colors: usize,
    /// Gauge coupling constant
    pub gauge_coupling: f64,
    /// Gauge fixing condition
    pub gauge_fixing: GaugeFixing,
    /// Wilson loop configurations
    pub wilson_loops: Vec<WilsonLoop>,
}

/// Gauge groups
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GaugeGroup {
    /// U(1) Abelian gauge group
    U1,
    /// SU(N) special unitary group
    SU(usize),
    /// SO(N) orthogonal group
    SO(usize),
    /// Sp(N) symplectic group
    Sp(usize),
}

/// Gauge fixing conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GaugeFixing {
    /// Coulomb gauge
    Coulomb,
    /// Landau gauge
    Landau,
    /// Axial gauge
    Axial,
    /// Temporal gauge
    Temporal,
    /// No gauge fixing
    None,
}

/// Wilson loop configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilsonLoop {
    /// Loop path on lattice
    pub path: Vec<(usize, usize)>,
    /// Loop size (spatial × temporal)
    pub size: (usize, usize),
    /// Expected vacuum expectation value
    pub vev: Option<Complex64>,
}

/// Renormalization group flow data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RGFlow {
    /// Energy scale
    pub scale: f64,
    /// Beta functions for coupling constants
    pub beta_functions: HashMap<String, f64>,
    /// Anomalous dimensions
    pub anomalous_dimensions: HashMap<String, f64>,
    /// Running coupling constants
    pub running_couplings: HashMap<String, f64>,
    /// Fixed points
    pub fixed_points: Vec<FixedPoint>,
}

/// Fixed point in RG flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPoint {
    /// Fixed point couplings
    pub couplings: HashMap<String, f64>,
    /// Stability (eigenvalues of linearized flow)
    pub eigenvalues: Vec<Complex64>,
    /// Fixed point type
    pub fp_type: FixedPointType,
}

/// Types of RG fixed points
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixedPointType {
    /// Infrared stable (attractive)
    IRStable,
    /// Ultraviolet stable (repulsive)
    UVStable,
    /// Saddle point
    Saddle,
    /// Gaussian fixed point
    Gaussian,
    /// Non-trivial interacting fixed point
    Interacting,
}

/// Correlation function data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationFunction {
    /// Operator types
    pub operators: Vec<FieldOperator>,
    /// Spacetime separations
    pub separations: Vec<Vec<f64>>,
    /// Correlation values
    pub values: Array1<Complex64>,
    /// Statistical errors
    pub errors: Array1<f64>,
    /// Connected vs. disconnected parts
    pub connected: bool,
}

/// Scattering process configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatteringProcess {
    /// Initial state particles
    pub initial_state: Vec<ParticleState>,
    /// Final state particles
    pub final_state: Vec<ParticleState>,
    /// Center-of-mass energy
    pub cms_energy: f64,
    /// Scattering angle
    pub scattering_angle: f64,
    /// Cross section
    pub cross_section: Option<f64>,
    /// S-matrix element
    pub s_matrix_element: Option<Complex64>,
}

/// Particle state in scattering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleState {
    /// Particle type/mass
    pub particle_type: String,
    /// Four-momentum
    pub momentum: [f64; 4],
    /// Spin/polarization state
    pub spin_state: Vec<Complex64>,
    /// Charge/quantum numbers
    pub quantum_numbers: HashMap<String, i32>,
}

/// Main quantum field theory simulator
#[derive(Debug)]
pub struct QuantumFieldTheorySimulator {
    /// Configuration
    config: QFTConfig,
    /// Field configurations on lattice
    field_configs: HashMap<String, Array4<Complex64>>,
    /// Gauge field configurations
    gauge_configs: Option<GaugeFieldConfig>,
    /// Path integral sampler
    path_integral: Option<PathIntegralSampler>,
    /// Renormalization group flow tracker
    rg_flow: Option<RGFlow>,
    /// Correlation function cache
    correlations: HashMap<String, CorrelationFunction>,
    /// SciRS2 backend for numerical computations
    backend: Option<SciRS2Backend>,
    /// Simulation statistics
    stats: QFTStats,
}

/// Path integral Monte Carlo sampler
#[derive(Debug)]
pub struct PathIntegralSampler {
    /// Configuration
    config: PathIntegralConfig,
    /// Current field configuration
    current_config: Array4<Complex64>,
    /// Action evaluator
    action_evaluator: ActionEvaluator,
    /// Monte Carlo state
    mc_state: MonteCarloState,
    /// Sample history
    sample_history: VecDeque<Array4<Complex64>>,
}

/// Action evaluator for different field theories
#[derive(Debug)]
pub struct ActionEvaluator {
    /// Field theory type
    theory_type: FieldTheoryType,
    /// Coupling constants
    couplings: HashMap<String, f64>,
    /// Lattice parameters
    lattice_params: LatticeParameters,
}

/// Lattice parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeParameters {
    /// Lattice spacing
    pub spacing: f64,
    /// Bare mass
    pub bare_mass: f64,
    /// Hopping parameter
    pub hopping_parameter: f64,
    /// Plaquette size
    pub plaquette_size: usize,
}

/// Monte Carlo sampling state
#[derive(Debug)]
pub struct MonteCarloState {
    /// Current step number
    pub step: usize,
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// Total accepted moves
    pub accepted_moves: usize,
    /// Total proposed moves
    pub proposed_moves: usize,
    /// Current energy/action
    pub current_action: f64,
    /// Auto-correlation time
    pub autocorr_time: Option<f64>,
}

/// QFT simulation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QFTStats {
    /// Total simulation time
    pub simulation_time: f64,
    /// Number of field evaluations
    pub field_evaluations: usize,
    /// Path integral samples
    pub pi_samples: usize,
    /// Correlation function calculations
    pub correlation_calculations: usize,
    /// RG flow steps
    pub rg_steps: usize,
    /// Average plaquette value
    pub avg_plaquette: Option<f64>,
    /// Topological charge
    pub topological_charge: Option<f64>,
}

impl QuantumFieldTheorySimulator {
    /// Create a new QFT simulator
    pub fn new(config: QFTConfig) -> Result<Self> {
        let mut field_configs = HashMap::new();

        // Initialize field configurations based on theory type
        let field_shape = (
            config.lattice_size[0],
            config.lattice_size[1],
            config.lattice_size.get(2).copied().unwrap_or(1),
            config.lattice_size.get(3).copied().unwrap_or(1),
        );

        match config.field_theory {
            FieldTheoryType::ScalarPhi4 => {
                field_configs.insert("phi".to_string(), Array4::zeros(field_shape));
            }
            FieldTheoryType::QED => {
                // Fermion field
                field_configs.insert("psi".to_string(), Array4::zeros(field_shape));
                // Gauge field A_μ
                for mu in 0..config.spacetime_dimensions {
                    field_configs.insert(format!("A_{mu}"), Array4::zeros(field_shape));
                }
            }
            FieldTheoryType::YangMills | FieldTheoryType::QCD => {
                // Non-Abelian gauge fields
                let num_colors: usize = match config.field_theory {
                    FieldTheoryType::QCD => 3,
                    _ => 2,
                };

                for mu in 0..config.spacetime_dimensions {
                    for a in 0..num_colors.pow(2) - 1 {
                        // SU(N) generators
                        field_configs.insert(format!("A_{mu}_{a}"), Array4::zeros(field_shape));
                    }
                }

                if config.field_theory == FieldTheoryType::QCD {
                    // Quark fields
                    for flavor in 0..6 {
                        // 6 quark flavors
                        field_configs.insert(format!("q_{flavor}"), Array4::zeros(field_shape));
                    }
                }
            }
            _ => {
                field_configs.insert("default".to_string(), Array4::zeros(field_shape));
            }
        }

        Ok(Self {
            config,
            field_configs,
            gauge_configs: None,
            path_integral: None,
            rg_flow: None,
            correlations: HashMap::new(),
            backend: None,
            stats: QFTStats::default(),
        })
    }

    /// Initialize path integral sampler
    pub fn initialize_path_integral(&mut self, pi_config: PathIntegralConfig) -> Result<()> {
        let field_shape = (
            self.config.lattice_size[0],
            self.config.lattice_size[1],
            self.config.lattice_size.get(2).copied().unwrap_or(1),
            pi_config.time_slices,
        );

        let lattice_params = LatticeParameters {
            spacing: self.config.lattice_spacing,
            bare_mass: self
                .config
                .coupling_constants
                .get("m0")
                .copied()
                .unwrap_or(1.0),
            hopping_parameter: 1.0
                / 2.0f64.mul_add(
                    self.config
                        .coupling_constants
                        .get("m0")
                        .copied()
                        .unwrap_or(1.0),
                    8.0,
                ),
            plaquette_size: 1,
        };

        let action_evaluator = ActionEvaluator {
            theory_type: self.config.field_theory,
            couplings: self.config.coupling_constants.clone(),
            lattice_params,
        };

        let mc_state = MonteCarloState {
            step: 0,
            acceptance_rate: 0.0,
            accepted_moves: 0,
            proposed_moves: 0,
            current_action: 0.0,
            autocorr_time: None,
        };

        self.path_integral = Some(PathIntegralSampler {
            config: pi_config,
            current_config: Array4::zeros(field_shape),
            action_evaluator,
            mc_state,
            sample_history: VecDeque::new(),
        });

        Ok(())
    }

    /// Set up gauge field configuration
    pub fn setup_gauge_fields(&mut self, gauge_config: GaugeFieldConfig) -> Result<()> {
        // Initialize Wilson loops for the gauge group
        let mut wilson_loops = Vec::new();

        // Create fundamental Wilson loops
        for r in 1..=4 {
            for t in 1..=4 {
                wilson_loops.push(WilsonLoop {
                    path: self.generate_wilson_loop_path(r, t)?,
                    size: (r, t),
                    vev: None,
                });
            }
        }

        let mut gauge_conf = gauge_config;
        gauge_conf.wilson_loops = wilson_loops;

        // Initialize gauge field matrices
        self.initialize_gauge_field_matrices(&gauge_conf)?;

        self.gauge_configs = Some(gauge_conf);
        Ok(())
    }

    /// Generate Wilson loop path on lattice
    fn generate_wilson_loop_path(
        &self,
        spatial_size: usize,
        temporal_size: usize,
    ) -> Result<Vec<(usize, usize)>> {
        let mut path = Vec::new();

        // Start at origin
        let start_x = 0;
        let start_t = 0;

        // Go right (spatial direction) - spatial_size links
        for i in 0..spatial_size {
            path.push((start_x + i, start_t));
        }

        // Go up (temporal direction) - temporal_size links
        for i in 0..temporal_size {
            path.push((start_x + spatial_size - 1, start_t + i));
        }

        // Go left - spatial_size links
        for i in 0..spatial_size {
            path.push((start_x + spatial_size - 1 - i, start_t + temporal_size - 1));
        }

        // Go down - temporal_size links to close the loop
        for i in 0..temporal_size {
            path.push((start_x, start_t + temporal_size - 1 - i));
        }

        Ok(path)
    }

    /// Initialize gauge field matrices on lattice
    fn initialize_gauge_field_matrices(&mut self, gauge_config: &GaugeFieldConfig) -> Result<()> {
        let field_shape = (
            self.config.lattice_size[0],
            self.config.lattice_size[1],
            self.config.lattice_size.get(2).copied().unwrap_or(1),
            self.config.lattice_size.get(3).copied().unwrap_or(1),
        );

        match gauge_config.gauge_group {
            GaugeGroup::U1 => {
                // U(1) gauge field - single complex phase
                for mu in 0..self.config.spacetime_dimensions {
                    self.field_configs.insert(
                        format!("U_{mu}"),
                        Array4::from_elem(field_shape, Complex64::new(1.0, 0.0)),
                    );
                }
            }
            GaugeGroup::SU(n) => {
                // SU(N) gauge field - N×N unitary matrices
                for mu in 0..self.config.spacetime_dimensions {
                    for i in 0..n {
                        for j in 0..n {
                            let initial_value = if i == j {
                                Complex64::new(1.0, 0.0)
                            } else {
                                Complex64::new(0.0, 0.0)
                            };

                            self.field_configs.insert(
                                format!("U_{mu}_{i}{j}"),
                                Array4::from_elem(field_shape, initial_value),
                            );
                        }
                    }
                }
            }
            _ => {
                return Err(SimulatorError::InvalidConfiguration(
                    "Unsupported gauge group".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Run path integral Monte Carlo simulation
    pub fn run_path_integral_mc(&mut self, num_steps: usize) -> Result<Vec<f64>> {
        let pi_sampler = self.path_integral.as_mut().ok_or_else(|| {
            SimulatorError::InvalidConfiguration("Path integral not initialized".to_string())
        })?;

        let mut action_history = Vec::with_capacity(num_steps);

        for step in 0..num_steps {
            // Propose field configuration update
            let proposed_config = Self::propose_field_update(&pi_sampler.current_config)?;

            // Calculate action difference
            let current_action = pi_sampler
                .action_evaluator
                .evaluate_action(&pi_sampler.current_config)?;
            let proposed_action = pi_sampler
                .action_evaluator
                .evaluate_action(&proposed_config)?;

            let delta_action = proposed_action - current_action;

            // Metropolis acceptance criterion
            let accept_prob = (-delta_action).exp().min(1.0);
            let rand_val: f64 = thread_rng().gen();

            pi_sampler.mc_state.proposed_moves += 1;

            if rand_val < accept_prob {
                // Accept the move
                pi_sampler.current_config = proposed_config;
                pi_sampler.mc_state.current_action = proposed_action;
                pi_sampler.mc_state.accepted_moves += 1;

                // Store in history
                if pi_sampler.sample_history.len() >= 1000 {
                    pi_sampler.sample_history.pop_front();
                }
                pi_sampler
                    .sample_history
                    .push_back(pi_sampler.current_config.clone());
            }

            // Update acceptance rate
            pi_sampler.mc_state.acceptance_rate = pi_sampler.mc_state.accepted_moves as f64
                / pi_sampler.mc_state.proposed_moves as f64;

            action_history.push(pi_sampler.mc_state.current_action);
            pi_sampler.mc_state.step = step;

            // Update statistics
            self.stats.pi_samples += 1;
        }

        Ok(action_history)
    }

    /// Propose a field configuration update
    fn propose_field_update(current_config: &Array4<Complex64>) -> Result<Array4<Complex64>> {
        let mut proposed = current_config.clone();
        let update_fraction = 0.1; // Update 10% of sites

        let total_sites = proposed.len();
        let sites_to_update = (total_sites as f64 * update_fraction) as usize;

        let mut rng = thread_rng();
        for _ in 0..sites_to_update {
            let i = rng.gen_range(0..proposed.shape()[0]);
            let j = rng.gen_range(0..proposed.shape()[1]);
            let k = rng.gen_range(0..proposed.shape()[2]);
            let l = rng.gen_range(0..proposed.shape()[3]);

            // Gaussian random update
            let delta_real: f64 = rng.gen::<f64>() - 0.5;
            let delta_imag: f64 = rng.gen::<f64>() - 0.5;
            let delta = Complex64::new(delta_real * 0.1, delta_imag * 0.1);

            proposed[[i, j, k, l]] += delta;
        }

        Ok(proposed)
    }

    /// Calculate correlation functions
    pub fn calculate_correlation_function(
        &mut self,
        operators: &[FieldOperator],
        max_separation: usize,
    ) -> Result<CorrelationFunction> {
        let separations: Vec<Vec<f64>> = (0..=max_separation)
            .map(|r| vec![r as f64, 0.0, 0.0, 0.0])
            .collect();

        let mut values = Array1::zeros(separations.len());
        let mut errors = Array1::zeros(separations.len());

        // Use path integral samples to calculate correlations
        if let Some(pi_sampler) = &self.path_integral {
            if !pi_sampler.sample_history.is_empty() {
                for (sep_idx, separation) in separations.iter().enumerate() {
                    let mut correlator_samples = Vec::new();

                    for config in &pi_sampler.sample_history {
                        let corr_value =
                            self.evaluate_correlator_on_config(operators, separation, config)?;
                        correlator_samples.push(corr_value);
                    }

                    // Calculate mean and standard error
                    let mean = correlator_samples.iter().sum::<Complex64>()
                        / correlator_samples.len() as f64;
                    let variance = correlator_samples
                        .iter()
                        .map(|x| (x - mean).norm_sqr())
                        .sum::<f64>()
                        / correlator_samples.len() as f64;
                    let std_error = (variance / correlator_samples.len() as f64).sqrt();

                    values[sep_idx] = mean;
                    errors[sep_idx] = std_error;
                }
            }
        }

        let correlation_fn = CorrelationFunction {
            operators: operators.to_vec(),
            separations,
            values,
            errors,
            connected: true,
        };

        self.stats.correlation_calculations += 1;
        Ok(correlation_fn)
    }

    /// Evaluate correlator on a specific field configuration
    fn evaluate_correlator_on_config(
        &self,
        operators: &[FieldOperator],
        separation: &[f64],
        config: &Array4<Complex64>,
    ) -> Result<Complex64> {
        // For simplicity, calculate 2-point correlator ⟨φ(0)φ(r)⟩
        if operators.len() != 2 {
            return Err(SimulatorError::InvalidConfiguration(
                "Only 2-point correlators currently supported".to_string(),
            ));
        }

        let r = separation[0] as usize;
        let lattice_size = config.shape()[0];

        if r >= lattice_size {
            return Ok(Complex64::new(0.0, 0.0));
        }

        let mut correlator = Complex64::new(0.0, 0.0);
        let mut norm = 0.0;

        // Average over all possible source positions
        for x0 in 0..lattice_size {
            for y0 in 0..config.shape()[1] {
                for z0 in 0..config.shape()[2] {
                    for t0 in 0..config.shape()[3] {
                        let x1 = (x0 + r) % lattice_size;

                        let field_0 = config[[x0, y0, z0, t0]];
                        let field_r = config[[x1, y0, z0, t0]];

                        correlator += field_0.conj() * field_r;
                        norm += 1.0;
                    }
                }
            }
        }

        Ok(correlator / norm)
    }

    /// Calculate Wilson loops for gauge theories
    pub fn calculate_wilson_loops(&mut self) -> Result<HashMap<String, Complex64>> {
        let gauge_config = self.gauge_configs.as_ref().ok_or_else(|| {
            SimulatorError::InvalidConfiguration("Gauge fields not initialized".to_string())
        })?;

        let mut wilson_values = HashMap::new();

        for (loop_idx, wilson_loop) in gauge_config.wilson_loops.iter().enumerate() {
            let loop_value = self.evaluate_wilson_loop(wilson_loop)?;
            wilson_values.insert(
                format!("WL_{}x{}", wilson_loop.size.0, wilson_loop.size.1),
                loop_value,
            );
        }

        Ok(wilson_values)
    }

    /// Evaluate a single Wilson loop
    fn evaluate_wilson_loop(&self, wilson_loop: &WilsonLoop) -> Result<Complex64> {
        let mut loop_value = Complex64::new(1.0, 0.0);

        // For U(1) gauge theory, Wilson loop is product of gauge links
        for (i, &(x, t)) in wilson_loop.path.iter().enumerate() {
            let next_site = wilson_loop.path.get(i + 1).unwrap_or(&wilson_loop.path[0]);

            // Determine direction of link
            let mu = if next_site.0 == x { 3 } else { 0 }; // spatial or temporal

            if let Some(gauge_field) = self.field_configs.get(&format!("U_{mu}")) {
                if x < gauge_field.shape()[0] && t < gauge_field.shape()[3] {
                    let link_value = gauge_field[[x, 0, 0, t]];
                    loop_value *= link_value;
                }
            }
        }

        Ok(loop_value)
    }

    /// Run renormalization group flow analysis
    pub fn analyze_rg_flow(&mut self, energy_scales: &[f64]) -> Result<RGFlow> {
        let mut beta_functions = HashMap::new();
        let mut anomalous_dimensions = HashMap::new();
        let mut running_couplings = HashMap::new();

        // Initialize with bare couplings
        for (coupling_name, &coupling_value) in &self.config.coupling_constants {
            running_couplings.insert(coupling_name.clone(), coupling_value);
        }

        let mut rg_flow = RGFlow {
            scale: energy_scales[0],
            beta_functions,
            anomalous_dimensions,
            running_couplings: running_couplings.clone(),
            fixed_points: Vec::new(),
        };

        // Calculate beta functions for each scale
        for &scale in energy_scales {
            let beta_g = self.calculate_beta_function("g", scale, &running_couplings)?;
            let beta_lambda = self.calculate_beta_function("lambda", scale, &running_couplings)?;

            rg_flow.beta_functions.insert("g".to_string(), beta_g);
            rg_flow
                .beta_functions
                .insert("lambda".to_string(), beta_lambda);

            // Update running couplings using RG equations
            let dt = 0.01; // Small step in log(scale)
            if let Some(g) = running_couplings.get_mut("g") {
                *g += beta_g * dt;
            }
            if let Some(lambda) = running_couplings.get_mut("lambda") {
                *lambda += beta_lambda * dt;
            }

            rg_flow.scale = scale;
            rg_flow.running_couplings = running_couplings.clone();
            self.stats.rg_steps += 1;
        }

        // Find fixed points
        rg_flow.fixed_points = self.find_rg_fixed_points(&rg_flow.beta_functions)?;

        self.rg_flow = Some(rg_flow.clone());
        Ok(rg_flow)
    }

    /// Calculate beta function for a coupling
    fn calculate_beta_function(
        &self,
        coupling_name: &str,
        scale: f64,
        couplings: &HashMap<String, f64>,
    ) -> Result<f64> {
        match self.config.field_theory {
            FieldTheoryType::ScalarPhi4 => {
                match coupling_name {
                    "lambda" => {
                        // β_λ = 3λ²/(4π)² in φ⁴ theory (one-loop)
                        let lambda = couplings.get("lambda").copied().unwrap_or(0.0);
                        Ok(3.0 * lambda.powi(2) / (16.0 * PI.powi(2)))
                    }
                    "g" => {
                        // No gauge coupling in scalar theory
                        Ok(0.0)
                    }
                    _ => Ok(0.0),
                }
            }
            FieldTheoryType::QED => {
                match coupling_name {
                    "e" | "g" => {
                        // β_e = e³/(12π²) in QED (one-loop)
                        let e = couplings
                            .get("e")
                            .or_else(|| couplings.get("g"))
                            .copied()
                            .unwrap_or(0.0);
                        Ok(e.powi(3) / (12.0 * PI.powi(2)))
                    }
                    _ => Ok(0.0),
                }
            }
            FieldTheoryType::YangMills => {
                match coupling_name {
                    "g" => {
                        // β_g = -b₀g³ where b₀ = 11N/(12π) for SU(N)
                        let g = couplings.get("g").copied().unwrap_or(0.0);
                        let n_colors = 3.0; // Default to SU(3)
                        let b0 = 11.0 * n_colors / (12.0 * PI);
                        Ok(-b0 * g.powi(3))
                    }
                    _ => Ok(0.0),
                }
            }
            _ => Ok(0.0),
        }
    }

    /// Find fixed points in RG flow
    fn find_rg_fixed_points(
        &self,
        beta_functions: &HashMap<String, f64>,
    ) -> Result<Vec<FixedPoint>> {
        let mut fixed_points = Vec::new();

        // Gaussian fixed point (all couplings = 0)
        let mut gaussian_couplings = HashMap::new();
        for coupling_name in beta_functions.keys() {
            gaussian_couplings.insert(coupling_name.clone(), 0.0);
        }

        fixed_points.push(FixedPoint {
            couplings: gaussian_couplings,
            eigenvalues: vec![Complex64::new(-1.0, 0.0)], // IR attractive
            fp_type: FixedPointType::Gaussian,
        });

        // Look for non-trivial fixed points numerically
        // (This is a simplified approach - real analysis would be more sophisticated)
        for lambda_star in [0.1, 0.5, 1.0, 2.0] {
            let mut test_couplings = HashMap::new();
            test_couplings.insert("lambda".to_string(), lambda_star);
            test_couplings.insert("g".to_string(), 0.0);

            let beta_lambda = self.calculate_beta_function("lambda", 1.0, &test_couplings)?;

            if beta_lambda.abs() < 1e-6 {
                // Found a fixed point
                fixed_points.push(FixedPoint {
                    couplings: test_couplings,
                    eigenvalues: vec![Complex64::new(1.0, 0.0)], // UV repulsive
                    fp_type: FixedPointType::Interacting,
                });
            }
        }

        Ok(fixed_points)
    }

    /// Calculate scattering cross section
    pub fn calculate_scattering_cross_section(
        &mut self,
        process: &ScatteringProcess,
    ) -> Result<f64> {
        // This is a simplified calculation - real implementation would involve
        // Feynman diagram evaluation, loop integrals, etc.

        let cms_energy = process.cms_energy;
        let num_initial = process.initial_state.len();
        let num_final = process.final_state.len();

        // Simple dimensional analysis estimate
        let coupling = self
            .config
            .coupling_constants
            .get("g")
            .copied()
            .unwrap_or(0.1);

        let cross_section = match self.config.field_theory {
            FieldTheoryType::ScalarPhi4 => {
                // σ ~ λ²/s for 2→2 scattering
                if num_initial == 2 && num_final == 2 {
                    let lambda = self
                        .config
                        .coupling_constants
                        .get("lambda")
                        .copied()
                        .unwrap_or(0.01);
                    lambda.powi(2) / cms_energy.powi(2)
                } else {
                    0.0
                }
            }
            FieldTheoryType::QED => {
                // σ ~ α²/s for e⁺e⁻ → μ⁺μ⁻
                let alpha = coupling.powi(2) / (4.0 * PI);
                alpha.powi(2) / cms_energy.powi(2)
            }
            _ => coupling.powi(2) / cms_energy.powi(2),
        };

        Ok(cross_section)
    }

    /// Get simulation statistics
    pub const fn get_statistics(&self) -> &QFTStats {
        &self.stats
    }

    /// Export field configurations for visualization
    pub fn export_field_configuration(&self, field_name: &str) -> Result<Array4<Complex64>> {
        self.field_configs.get(field_name).cloned().ok_or_else(|| {
            SimulatorError::InvalidConfiguration(format!("Field '{field_name}' not found"))
        })
    }
}

impl ActionEvaluator {
    /// Evaluate the action for a given field configuration
    pub fn evaluate_action(&self, config: &Array4<Complex64>) -> Result<f64> {
        match self.theory_type {
            FieldTheoryType::ScalarPhi4 => self.evaluate_phi4_action(config),
            FieldTheoryType::QED => self.evaluate_qed_action(config),
            FieldTheoryType::YangMills => self.evaluate_yang_mills_action(config),
            _ => self.evaluate_generic_action(config),
        }
    }

    /// Evaluate φ⁴ scalar field theory action
    fn evaluate_phi4_action(&self, config: &Array4<Complex64>) -> Result<f64> {
        let mut action = 0.0;
        let lattice_spacing = self.lattice_params.spacing;
        let mass_sq = self.lattice_params.bare_mass.powi(2);
        let lambda = self.couplings.get("lambda").copied().unwrap_or(0.01);

        let shape = config.shape();
        let (nx, ny, nz, nt) = (shape[0], shape[1], shape[2], shape[3]);

        // Kinetic term: -φ∇²φ
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    for t in 0..nt {
                        let phi = config[[x, y, z, t]];

                        // Laplacian in 4D
                        let mut laplacian = Complex64::new(0.0, 0.0);

                        // x-direction
                        let phi_xp = config[[(x + 1) % nx, y, z, t]];
                        let phi_xm = config[[(x + nx - 1) % nx, y, z, t]];
                        laplacian += (phi_xp + phi_xm - 2.0 * phi) / lattice_spacing.powi(2);

                        // y-direction
                        let phi_yp = config[[x, (y + 1) % ny, z, t]];
                        let phi_ym = config[[x, (y + ny - 1) % ny, z, t]];
                        laplacian += (phi_yp + phi_ym - 2.0 * phi) / lattice_spacing.powi(2);

                        // z-direction if 3D
                        if nz > 1 {
                            let phi_zp = config[[x, y, (z + 1) % nz, t]];
                            let phi_zm = config[[x, y, (z + nz - 1) % nz, t]];
                            laplacian += (phi_zp + phi_zm - 2.0 * phi) / lattice_spacing.powi(2);
                        }

                        // t-direction if 4D
                        if nt > 1 {
                            let phi_tp = config[[x, y, z, (t + 1) % nt]];
                            let phi_tm = config[[x, y, z, (t + nt - 1) % nt]];
                            laplacian += (phi_tp + phi_tm - 2.0 * phi) / lattice_spacing.powi(2);
                        }

                        // Kinetic term
                        let kinetic = -phi.conj() * laplacian;

                        // Mass term
                        let mass_term = mass_sq * phi.norm_sqr();

                        // Interaction term φ⁴
                        let interaction = lambda * phi.norm_sqr().powi(2);

                        action += (kinetic.re + mass_term + interaction) * lattice_spacing.powi(4);
                    }
                }
            }
        }

        Ok(action)
    }

    /// Evaluate QED action (simplified)
    fn evaluate_qed_action(&self, config: &Array4<Complex64>) -> Result<f64> {
        // Simplified QED action evaluation
        let mut action = 0.0;
        let lattice_spacing = self.lattice_params.spacing;
        let e = self.couplings.get("e").copied().unwrap_or(0.1);

        // This would normally involve gauge field kinetic terms,
        // fermion kinetic terms, and interaction terms
        action += config.iter().map(|phi| phi.norm_sqr()).sum::<f64>() * lattice_spacing.powi(4);

        Ok(action)
    }

    /// Evaluate Yang-Mills action (simplified)
    fn evaluate_yang_mills_action(&self, config: &Array4<Complex64>) -> Result<f64> {
        // Simplified Yang-Mills action with Wilson gauge action
        let mut action = 0.0;
        let lattice_spacing = self.lattice_params.spacing;
        let g = self.couplings.get("g").copied().unwrap_or(0.1);
        let beta = 2.0 / g.powi(2); // Lattice gauge theory beta parameter

        // Wilson plaquette action: S = β Σ Re[Tr(1 - U_p)]
        let shape = config.shape();
        let (nx, ny, nz, nt) = (shape[0], shape[1], shape[2], shape[3]);

        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    for t in 0..nt {
                        // Calculate plaquettes in all planes
                        let plaquette_xy = self.calculate_plaquette(config, x, y, z, t, 0, 1)?;
                        let plaquette_xt = self.calculate_plaquette(config, x, y, z, t, 0, 3)?;
                        let plaquette_yt = self.calculate_plaquette(config, x, y, z, t, 1, 3)?;

                        action +=
                            beta * (3.0 - plaquette_xy.re - plaquette_xt.re - plaquette_yt.re);
                    }
                }
            }
        }

        Ok(action)
    }

    /// Calculate Wilson plaquette for Yang-Mills action
    fn calculate_plaquette(
        &self,
        config: &Array4<Complex64>,
        x: usize,
        y: usize,
        z: usize,
        t: usize,
        mu: usize,
        nu: usize,
    ) -> Result<Complex64> {
        // For simplicity, assume U(1) gauge theory where links are phases
        let shape = config.shape();

        // Get gauge links around elementary plaquette
        let u_mu = config[[x, y, z, t]]; // U_μ(x)
        let u_nu_shifted = config[[
            (x + usize::from(mu == 0)) % shape[0],
            (y + usize::from(mu == 1)) % shape[1],
            z,
            t,
        ]]; // U_ν(x+μ)

        let u_mu_shifted = config[[
            (x + usize::from(nu == 0)) % shape[0],
            (y + usize::from(nu == 1)) % shape[1],
            z,
            t,
        ]]; // U_μ(x+ν)

        let u_nu = config[[x, y, z, t]]; // U_ν(x)

        // Wilson plaquette: U_μ(x) U_ν(x+μ) U_μ†(x+ν) U_ν†(x)
        let plaquette = u_mu * u_nu_shifted * u_mu_shifted.conj() * u_nu.conj();

        Ok(plaquette)
    }

    /// Evaluate generic action for other field theories
    fn evaluate_generic_action(&self, config: &Array4<Complex64>) -> Result<f64> {
        let lattice_spacing = self.lattice_params.spacing;

        // Simple kinetic + mass action
        let action = config.iter().map(|phi| phi.norm_sqr()).sum::<f64>() * lattice_spacing.powi(4);

        Ok(action)
    }
}

impl Default for QFTStats {
    fn default() -> Self {
        Self {
            simulation_time: 0.0,
            field_evaluations: 0,
            pi_samples: 0,
            correlation_calculations: 0,
            rg_steps: 0,
            avg_plaquette: None,
            topological_charge: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qft_simulator_creation() {
        let config = QFTConfig::default();
        let simulator = QuantumFieldTheorySimulator::new(config);
        assert!(simulator.is_ok());
    }

    #[test]
    fn test_scalar_phi4_configuration() {
        let mut config = QFTConfig::default();
        config.field_theory = FieldTheoryType::ScalarPhi4;
        config.lattice_size = vec![8, 8, 8, 16];

        let simulator = QuantumFieldTheorySimulator::new(config);
        assert!(simulator.is_ok());

        let sim = simulator.unwrap();
        assert!(sim.field_configs.contains_key("phi"));
    }

    #[test]
    fn test_qed_configuration() {
        let mut config = QFTConfig::default();
        config.field_theory = FieldTheoryType::QED;
        config.spacetime_dimensions = 4;

        let simulator = QuantumFieldTheorySimulator::new(config);
        assert!(simulator.is_ok());

        let sim = simulator.unwrap();
        assert!(sim.field_configs.contains_key("psi"));
        assert!(sim.field_configs.contains_key("A_0"));
        assert!(sim.field_configs.contains_key("A_1"));
        assert!(sim.field_configs.contains_key("A_2"));
        assert!(sim.field_configs.contains_key("A_3"));
    }

    #[test]
    fn test_path_integral_initialization() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let pi_config = PathIntegralConfig {
            time_slices: 32,
            time_step: 0.1,
            action_type: ActionType::Euclidean,
            mc_algorithm: MonteCarloAlgorithm::Metropolis,
            importance_sampling: true,
            target_acceptance_rate: 0.5,
        };

        let result = simulator.initialize_path_integral(pi_config);
        assert!(result.is_ok());
        assert!(simulator.path_integral.is_some());
    }

    #[test]
    fn test_gauge_field_setup_u1() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let gauge_config = GaugeFieldConfig {
            gauge_group: GaugeGroup::U1,
            num_colors: 1,
            gauge_coupling: 0.1,
            gauge_fixing: GaugeFixing::Landau,
            wilson_loops: Vec::new(),
        };

        let result = simulator.setup_gauge_fields(gauge_config);
        assert!(result.is_ok());
        assert!(simulator.gauge_configs.is_some());
    }

    #[test]
    fn test_gauge_field_setup_su3() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let gauge_config = GaugeFieldConfig {
            gauge_group: GaugeGroup::SU(3),
            num_colors: 3,
            gauge_coupling: 0.2,
            gauge_fixing: GaugeFixing::Coulomb,
            wilson_loops: Vec::new(),
        };

        let result = simulator.setup_gauge_fields(gauge_config);
        assert!(result.is_ok());
        assert!(simulator.gauge_configs.is_some());

        // Check that SU(3) gauge field matrices are created
        assert!(simulator.field_configs.contains_key("U_0_00"));
        assert!(simulator.field_configs.contains_key("U_0_11"));
        assert!(simulator.field_configs.contains_key("U_0_22"));
    }

    #[test]
    fn test_wilson_loop_generation() {
        let config = QFTConfig::default();
        let simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let path = simulator.generate_wilson_loop_path(2, 3).unwrap();
        assert_eq!(path.len(), 10); // 2+3+2+3 = 10 links for rectangular loop

        // Check that path forms a closed loop
        assert_eq!(path[0], path[path.len() - 1]);
    }

    #[test]
    fn test_action_evaluation_phi4() {
        let config = QFTConfig::default();
        let simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let lattice_params = LatticeParameters {
            spacing: 1.0,
            bare_mass: 1.0,
            hopping_parameter: 0.125,
            plaquette_size: 1,
        };

        let mut couplings = HashMap::new();
        couplings.insert("lambda".to_string(), 0.1);

        let evaluator = ActionEvaluator {
            theory_type: FieldTheoryType::ScalarPhi4,
            couplings,
            lattice_params,
        };

        let field_config = Array4::zeros((4, 4, 4, 4));
        let action = evaluator.evaluate_action(&field_config);
        assert!(action.is_ok());
        assert_eq!(action.unwrap(), 0.0); // Zero field gives zero action
    }

    #[test]
    fn test_beta_function_phi4() {
        let config = QFTConfig::default();
        let simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let mut couplings = HashMap::new();
        couplings.insert("lambda".to_string(), 0.1);

        let beta_lambda = simulator.calculate_beta_function("lambda", 1.0, &couplings);
        assert!(beta_lambda.is_ok());

        let beta_val = beta_lambda.unwrap();
        assert!(beta_val > 0.0); // Positive beta function in φ⁴ theory
    }

    #[test]
    fn test_beta_function_qed() {
        let mut config = QFTConfig::default();
        config.field_theory = FieldTheoryType::QED;
        let simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let mut couplings = HashMap::new();
        couplings.insert("e".to_string(), 0.3);

        let beta_e = simulator.calculate_beta_function("e", 1.0, &couplings);
        assert!(beta_e.is_ok());

        let beta_val = beta_e.unwrap();
        assert!(beta_val > 0.0); // QED has positive beta function (Landau pole)
    }

    #[test]
    fn test_beta_function_yang_mills() {
        let mut config = QFTConfig::default();
        config.field_theory = FieldTheoryType::YangMills;
        let simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let mut couplings = HashMap::new();
        couplings.insert("g".to_string(), 0.5);

        let beta_g = simulator.calculate_beta_function("g", 1.0, &couplings);
        assert!(beta_g.is_ok());

        let beta_val = beta_g.unwrap();
        assert!(beta_val < 0.0); // Yang-Mills has negative beta function (asymptotic freedom)
    }

    #[test]
    fn test_rg_flow_analysis() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let energy_scales = [0.1, 1.0, 10.0, 100.0];
        let rg_flow = simulator.analyze_rg_flow(&energy_scales);
        assert!(rg_flow.is_ok());

        let flow = rg_flow.unwrap();
        assert!(flow.beta_functions.contains_key("lambda"));
        assert!(!flow.fixed_points.is_empty());
        assert!(flow
            .fixed_points
            .iter()
            .any(|fp| fp.fp_type == FixedPointType::Gaussian));
    }

    #[test]
    fn test_scattering_cross_section_phi4() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let process = ScatteringProcess {
            initial_state: vec![
                ParticleState {
                    particle_type: "phi".to_string(),
                    momentum: [1.0, 0.0, 0.0, 0.0],
                    spin_state: vec![Complex64::new(1.0, 0.0)],
                    quantum_numbers: HashMap::new(),
                },
                ParticleState {
                    particle_type: "phi".to_string(),
                    momentum: [-1.0, 0.0, 0.0, 0.0],
                    spin_state: vec![Complex64::new(1.0, 0.0)],
                    quantum_numbers: HashMap::new(),
                },
            ],
            final_state: vec![
                ParticleState {
                    particle_type: "phi".to_string(),
                    momentum: [0.0, 1.0, 0.0, 0.0],
                    spin_state: vec![Complex64::new(1.0, 0.0)],
                    quantum_numbers: HashMap::new(),
                },
                ParticleState {
                    particle_type: "phi".to_string(),
                    momentum: [0.0, -1.0, 0.0, 0.0],
                    spin_state: vec![Complex64::new(1.0, 0.0)],
                    quantum_numbers: HashMap::new(),
                },
            ],
            cms_energy: 2.0,
            scattering_angle: PI / 2.0,
            cross_section: None,
            s_matrix_element: None,
        };

        let cross_section = simulator.calculate_scattering_cross_section(&process);
        assert!(cross_section.is_ok());

        let sigma = cross_section.unwrap();
        assert!(sigma > 0.0);
        assert!(sigma.is_finite());
    }

    #[test]
    fn test_field_operator_creation() {
        let field_op = FieldOperator {
            field_type: FieldOperatorType::Scalar,
            position: vec![0.0, 0.0, 0.0, 0.0],
            momentum: None,
            component: 0,
            time_ordering: TimeOrdering::TimeOrdered,
            normal_ordering: true,
        };

        assert_eq!(field_op.field_type, FieldOperatorType::Scalar);
        assert_eq!(field_op.position.len(), 4);
        assert!(field_op.normal_ordering);
    }

    #[test]
    #[ignore]
    fn test_path_integral_monte_carlo_short() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let pi_config = PathIntegralConfig {
            time_slices: 8,
            time_step: 0.1,
            action_type: ActionType::Euclidean,
            mc_algorithm: MonteCarloAlgorithm::Metropolis,
            importance_sampling: true,
            target_acceptance_rate: 0.5,
        };

        simulator.initialize_path_integral(pi_config).unwrap();

        // Run short Monte Carlo simulation
        let result = simulator.run_path_integral_mc(100);
        assert!(result.is_ok());

        let action_history = result.unwrap();
        assert_eq!(action_history.len(), 100);
        assert!(action_history.iter().all(|&a| a.is_finite()));
    }

    #[test]
    #[ignore]
    fn test_correlation_function_calculation() {
        let config = QFTConfig::default();
        let mut simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let pi_config = PathIntegralConfig {
            time_slices: 8,
            time_step: 0.1,
            action_type: ActionType::Euclidean,
            mc_algorithm: MonteCarloAlgorithm::Metropolis,
            importance_sampling: true,
            target_acceptance_rate: 0.5,
        };

        simulator.initialize_path_integral(pi_config).unwrap();

        // Generate some samples first
        simulator.run_path_integral_mc(50).unwrap();

        let operators = vec![
            FieldOperator {
                field_type: FieldOperatorType::Scalar,
                position: vec![0.0, 0.0, 0.0, 0.0],
                momentum: None,
                component: 0,
                time_ordering: TimeOrdering::TimeOrdered,
                normal_ordering: true,
            },
            FieldOperator {
                field_type: FieldOperatorType::Scalar,
                position: vec![1.0, 0.0, 0.0, 0.0],
                momentum: None,
                component: 0,
                time_ordering: TimeOrdering::TimeOrdered,
                normal_ordering: true,
            },
        ];

        let correlation = simulator.calculate_correlation_function(&operators, 4);
        assert!(correlation.is_ok());

        let corr_fn = correlation.unwrap();
        assert_eq!(corr_fn.separations.len(), 5); // 0, 1, 2, 3, 4
        assert_eq!(corr_fn.values.len(), 5);
        assert_eq!(corr_fn.errors.len(), 5);
    }

    #[test]
    fn test_export_field_configuration() {
        let config = QFTConfig::default();
        let simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let field_config = simulator.export_field_configuration("phi");
        assert!(field_config.is_ok());

        let config_array = field_config.unwrap();
        assert_eq!(config_array.ndim(), 4);

        // Test non-existent field
        let invalid_field = simulator.export_field_configuration("nonexistent");
        assert!(invalid_field.is_err());
    }

    #[test]
    fn test_statistics_tracking() {
        let config = QFTConfig::default();
        let simulator = QuantumFieldTheorySimulator::new(config).unwrap();

        let stats = simulator.get_statistics();
        assert_eq!(stats.field_evaluations, 0);
        assert_eq!(stats.pi_samples, 0);
        assert_eq!(stats.correlation_calculations, 0);
        assert_eq!(stats.rg_steps, 0);
    }
}
