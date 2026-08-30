//! Open quantum system simulation with Lindblad master equations.
//!
//! This module provides comprehensive simulation capabilities for open quantum systems,
//! including master equation evolution, Kraus operators, noise channels, and
//! process tomography for realistic quantum device modeling.

use crate::prelude::SimulatorError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::parallel_ops::{IndexedParallelIterator, ParallelIterator};
use scirs2_core::Complex64;
use std::collections::HashMap;

use crate::error::Result;
use crate::scirs2_integration::SciRS2Backend;

/// Lindblad master equation simulator for open quantum systems
pub struct LindladSimulator {
    /// Number of qubits
    num_qubits: usize,
    /// Current density matrix
    density_matrix: Array2<Complex64>,
    /// Lindblad operators
    lindblad_ops: Vec<LindladOperator>,
    /// System Hamiltonian
    hamiltonian: Option<Array2<Complex64>>,
    /// Time step for evolution
    time_step: f64,
    /// Integration method
    integration_method: IntegrationMethod,
    /// `SciRS2` backend for linear algebra
    backend: Option<SciRS2Backend>,
}

/// Lindblad operator with coefficient
#[derive(Debug, Clone)]
pub struct LindladOperator {
    /// Operator matrix
    pub operator: Array2<Complex64>,
    /// Collapse rate
    pub rate: f64,
    /// Optional label for the operator
    pub label: Option<String>,
}

/// Integration methods for master equation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationMethod {
    /// Explicit Euler method (fast, less accurate)
    Euler,
    /// 4th order Runge-Kutta (balanced)
    RungeKutta4,
    /// Adaptive Runge-Kutta with error control
    AdaptiveRK,
    /// Matrix exponential (exact for time-independent systems)
    MatrixExponential,
}

impl LindladSimulator {
    /// Create new Lindblad simulator
    pub fn new(num_qubits: usize) -> Result<Self> {
        let dim = 1 << num_qubits;
        let mut density_matrix = Array2::zeros((dim, dim));
        density_matrix[[0, 0]] = Complex64::new(1.0, 0.0); // |0⟩⟨0|

        Ok(Self {
            num_qubits,
            density_matrix,
            lindblad_ops: Vec::new(),
            hamiltonian: None,
            time_step: 0.01,
            integration_method: IntegrationMethod::RungeKutta4,
            backend: None,
        })
    }

    /// Initialize with `SciRS2` backend
    pub fn with_scirs2_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        Ok(self)
    }

    /// Set initial density matrix
    pub fn set_density_matrix(&mut self, rho: Array2<Complex64>) -> Result<()> {
        let dim = 1 << self.num_qubits;
        if rho.shape() != [dim, dim] {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Expected {dim}x{dim} density matrix"
            )));
        }

        // Verify trace normalization
        let trace: Complex64 = rho.diag().iter().sum();
        if (trace.re - 1.0).abs() > 1e-10 || trace.im.abs() > 1e-10 {
            return Err(SimulatorError::InvalidInput(format!(
                "Density matrix not normalized: trace = {trace}"
            )));
        }

        self.density_matrix = rho;
        Ok(())
    }

    /// Initialize from pure state vector
    pub fn from_state_vector(&mut self, psi: &ArrayView1<Complex64>) -> Result<()> {
        let dim = 1 << self.num_qubits;
        if psi.len() != dim {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Expected state vector of length {dim}"
            )));
        }

        // Create density matrix |ψ⟩⟨ψ|
        let mut rho = Array2::zeros((dim, dim));
        for i in 0..dim {
            for j in 0..dim {
                rho[[i, j]] = psi[i] * psi[j].conj();
            }
        }

        self.density_matrix = rho;
        Ok(())
    }

    /// Add Lindblad operator
    pub fn add_lindblad_operator(&mut self, operator: LindladOperator) -> Result<()> {
        let dim = 1 << self.num_qubits;
        if operator.operator.shape() != [dim, dim] {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Operator must be {dim}x{dim}"
            )));
        }

        self.lindblad_ops.push(operator);
        Ok(())
    }

    /// Set system Hamiltonian
    pub fn set_hamiltonian(&mut self, h: Array2<Complex64>) -> Result<()> {
        let dim = 1 << self.num_qubits;
        if h.shape() != [dim, dim] {
            return Err(SimulatorError::DimensionMismatch(format!(
                "Hamiltonian must be {dim}x{dim}"
            )));
        }

        self.hamiltonian = Some(h);
        Ok(())
    }

    /// Set time step for integration
    pub const fn set_time_step(&mut self, dt: f64) {
        self.time_step = dt;
    }

    /// Set integration method
    pub const fn set_integration_method(&mut self, method: IntegrationMethod) {
        self.integration_method = method;
    }

    /// Evolve system for given time
    pub fn evolve(&mut self, total_time: f64) -> Result<EvolutionResult> {
        let num_steps = (total_time / self.time_step).ceil() as usize;
        let actual_dt = total_time / num_steps as f64;

        let mut times = Vec::with_capacity(num_steps + 1);
        let mut densities = Vec::new();
        let mut purities = Vec::with_capacity(num_steps + 1);
        let mut traces = Vec::with_capacity(num_steps + 1);

        // Record initial state
        times.push(0.0);
        purities.push(self.purity());
        traces.push(self.trace().re);

        // Store initial density if requested (for small systems)
        if self.num_qubits <= 4 {
            densities.push(self.density_matrix.clone());
        }

        // Time evolution loop
        for step in 0..num_steps {
            match self.integration_method {
                IntegrationMethod::Euler => {
                    self.euler_step(actual_dt)?;
                }
                IntegrationMethod::RungeKutta4 => {
                    self.runge_kutta4_step(actual_dt)?;
                }
                IntegrationMethod::AdaptiveRK => {
                    self.adaptive_rk_step(actual_dt)?;
                }
                IntegrationMethod::MatrixExponential => {
                    self.matrix_exponential_step(actual_dt)?;
                }
            }

            let current_time = (step + 1) as f64 * actual_dt;
            times.push(current_time);
            purities.push(self.purity());
            traces.push(self.trace().re);

            if self.num_qubits <= 4 {
                densities.push(self.density_matrix.clone());
            }
        }

        Ok(EvolutionResult {
            times,
            densities,
            purities,
            traces,
            final_density: self.density_matrix.clone(),
        })
    }

    /// Single Euler integration step
    fn euler_step(&mut self, dt: f64) -> Result<()> {
        let derivative = self.compute_lindblad_derivative()?;

        // ρ(t + dt) = ρ(t) + dt * dρ/dt
        for ((i, j), drho_dt) in derivative.indexed_iter() {
            self.density_matrix[[i, j]] += dt * drho_dt;
        }

        // Renormalize to maintain trace = 1
        self.renormalize();

        Ok(())
    }

    /// Single 4th order Runge-Kutta step
    fn runge_kutta4_step(&mut self, dt: f64) -> Result<()> {
        let rho0 = self.density_matrix.clone();

        // k1 = f(t, y)
        let k1 = self.compute_lindblad_derivative()?;

        // k2 = f(t + dt/2, y + dt*k1/2)
        self.density_matrix = &rho0 + &(&k1 * (dt / 2.0));
        let k2 = self.compute_lindblad_derivative()?;

        // k3 = f(t + dt/2, y + dt*k2/2)
        self.density_matrix = &rho0 + &(&k2 * (dt / 2.0));
        let k3 = self.compute_lindblad_derivative()?;

        // k4 = f(t + dt, y + dt*k3)
        self.density_matrix = &rho0 + &(&k3 * dt);
        let k4 = self.compute_lindblad_derivative()?;

        // y(t+dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        let coeff = Complex64::new(dt / 6.0, 0.0);
        self.density_matrix = rho0
            + coeff
                * (&k1 + &(Complex64::new(2.0, 0.0) * k2) + &(Complex64::new(2.0, 0.0) * k3) + &k4);

        self.renormalize();
        Ok(())
    }

    /// Adaptive Runge-Kutta step with error control
    fn adaptive_rk_step(&mut self, dt: f64) -> Result<()> {
        // For now, use fixed RK4 - full adaptive implementation would require
        // error estimation and step size control
        self.runge_kutta4_step(dt)
    }

    /// Matrix exponential step (exact for time-independent Lindbladian)
    fn matrix_exponential_step(&mut self, dt: f64) -> Result<()> {
        if let Some(ref backend) = self.backend {
            // Use SciRS2 for matrix exponential if available
            self.matrix_exp_with_scirs2(dt)
        } else {
            // Fallback to series expansion
            self.matrix_exp_series(dt)
        }
    }

    /// Matrix exponential using `SciRS2`
    fn matrix_exp_with_scirs2(&mut self, dt: f64) -> Result<()> {
        // This would use SciRS2's matrix exponential routines
        // For now, fallback to series expansion
        self.matrix_exp_series(dt)
    }

    /// Matrix exponential using series expansion
    fn matrix_exp_series(&mut self, dt: f64) -> Result<()> {
        let lindbladian = self.construct_lindbladian_superoperator()?;

        // Compute matrix exponential using series: exp(L*dt) ≈ I + L*dt + (L*dt)²/2! + ...
        let dim_sq = lindbladian.nrows();
        let mut result = Array2::eye(dim_sq);
        let mut term = Array2::eye(dim_sq);
        let l_dt = &lindbladian * dt;

        // Series expansion up to reasonable order
        for n in 1..=20 {
            term = term.dot(&l_dt) / f64::from(n);
            result += &term;

            // Check convergence
            let term_norm: f64 = term.iter().map(|x| x.norm()).sum();
            if term_norm < 1e-12 {
                break;
            }
        }

        // Apply to vectorized density matrix
        let rho_vec = self.vectorize_density_matrix();
        let new_rho_vec = result.dot(&rho_vec);
        self.density_matrix = self.devectorize_density_matrix(&new_rho_vec);

        self.renormalize();
        Ok(())
    }

    /// Compute Lindblad master equation derivative
    fn compute_lindblad_derivative(&self) -> Result<Array2<Complex64>> {
        let dim = self.density_matrix.nrows();
        let mut derivative = Array2::zeros((dim, dim));

        // Hamiltonian evolution: -i[H, ρ]
        if let Some(ref h) = self.hamiltonian {
            let commutator = h.dot(&self.density_matrix) - self.density_matrix.dot(h);
            derivative += &(commutator * Complex64::new(0.0, -1.0));
        }

        // Lindblad dissipation terms
        for lindblad_op in &self.lindblad_ops {
            let l = &lindblad_op.operator;
            let l_dag = l.t().mapv(|x| x.conj());
            let rate = lindblad_op.rate;

            // Dissipation: γ(L ρ L† - (1/2){L†L, ρ})
            let dissipation = l.dot(&self.density_matrix).dot(&l_dag);
            let anticommutator =
                l_dag.dot(l).dot(&self.density_matrix) + self.density_matrix.dot(&l_dag.dot(l));
            let half = Complex64::new(0.5, 0.0);

            derivative += &((dissipation - &anticommutator * half) * rate);
        }

        Ok(derivative)
    }

    /// Construct Lindbladian superoperator matrix
    fn construct_lindbladian_superoperator(&self) -> Result<Array2<Complex64>> {
        let dim = 1 << self.num_qubits;
        let super_dim = dim * dim;
        let mut lindbladian = Array2::zeros((super_dim, super_dim));

        // Hamiltonian part: -i(H ⊗ I - I ⊗ H^T)
        if let Some(ref h) = self.hamiltonian {
            let eye: Array2<Complex64> = Array2::eye(dim);
            let h_left = kron(h, &eye);
            let h_t = h.t().to_owned();
            let h_right = kron(&eye, &h_t);
            lindbladian += &((h_left - h_right) * Complex64::new(0.0, -1.0));
        }

        // Lindblad terms
        for lindblad_op in &self.lindblad_ops {
            let l = &lindblad_op.operator;
            let l_dag = l.t().mapv(|x| x.conj());
            let rate = lindblad_op.rate;

            let eye: Array2<Complex64> = Array2::eye(dim);
            let l_dag_l = l_dag.dot(l);

            // L ⊗ L*
            let left_term = kron(l, &l.mapv(|x| x.conj()));

            // -(1/2)(L†L ⊗ I + I ⊗ (L†L)^T)
            let l_dag_l_t = l_dag_l.t().to_owned();
            let right_term = kron(&l_dag_l, &eye) + kron(&eye, &l_dag_l_t);
            let half = Complex64::new(0.5, 0.0);

            lindbladian += &((left_term - &right_term * half) * rate);
        }

        Ok(lindbladian)
    }

    /// Vectorize density matrix (column-major order)
    fn vectorize_density_matrix(&self) -> Array1<Complex64> {
        let dim = self.density_matrix.nrows();
        let mut vec = Array1::zeros(dim * dim);

        for (i, &val) in self.density_matrix.iter().enumerate() {
            vec[i] = val;
        }

        vec
    }

    /// Devectorize density matrix
    fn devectorize_density_matrix(&self, vec: &Array1<Complex64>) -> Array2<Complex64> {
        let dim = (vec.len() as f64).sqrt() as usize;
        Array2::from_shape_vec((dim, dim), vec.to_vec()).expect(
            "devectorize_density_matrix: shape mismatch should not occur for valid density matrix",
        )
    }

    /// Get current purity Tr(ρ²)
    #[must_use]
    pub fn purity(&self) -> f64 {
        let rho_squared = self.density_matrix.dot(&self.density_matrix);
        rho_squared.diag().iter().map(|x| x.re).sum()
    }

    /// Get current trace
    #[must_use]
    pub fn trace(&self) -> Complex64 {
        self.density_matrix.diag().iter().sum()
    }

    /// Renormalize density matrix to unit trace
    fn renormalize(&mut self) {
        let trace = self.trace();
        if trace.norm() > 1e-12 {
            self.density_matrix /= trace;
        }
    }

    /// Get current density matrix
    #[must_use]
    pub const fn get_density_matrix(&self) -> &Array2<Complex64> {
        &self.density_matrix
    }

    /// Compute expectation value of observable
    pub fn expectation_value(&self, observable: &Array2<Complex64>) -> Result<Complex64> {
        if observable.shape() != self.density_matrix.shape() {
            return Err(SimulatorError::DimensionMismatch(
                "Observable and density matrix dimensions must match".to_string(),
            ));
        }

        // Tr(ρ O)
        let product = self.density_matrix.dot(observable);
        Ok(product.diag().iter().sum())
    }
}

/// Result of time evolution
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    /// Time points
    pub times: Vec<f64>,
    /// Density matrices at each time (if stored)
    pub densities: Vec<Array2<Complex64>>,
    /// Purity at each time
    pub purities: Vec<f64>,
    /// Trace at each time
    pub traces: Vec<f64>,
    /// Final density matrix
    pub final_density: Array2<Complex64>,
}

/// Quantum channel represented by Kraus operators
#[derive(Debug, Clone)]
pub struct QuantumChannel {
    /// Kraus operators
    pub kraus_operators: Vec<Array2<Complex64>>,
    /// Channel name
    pub name: String,
}

impl QuantumChannel {
    /// Create depolarizing channel
    #[must_use]
    pub fn depolarizing(num_qubits: usize, probability: f64) -> Self {
        let dim = 1 << num_qubits;
        let mut kraus_ops = Vec::new();

        // Identity term
        let sqrt_p0 = (1.0 - probability).sqrt();
        let eye: Array2<Complex64> = Array2::eye(dim) * Complex64::new(sqrt_p0, 0.0);
        kraus_ops.push(eye);

        // Pauli terms
        if num_qubits == 1 {
            let sqrt_p = (probability / 3.0).sqrt();

            // Pauli X
            let mut pauli_x = Array2::zeros((2, 2));
            pauli_x[[0, 1]] = Complex64::new(sqrt_p, 0.0);
            pauli_x[[1, 0]] = Complex64::new(sqrt_p, 0.0);
            kraus_ops.push(pauli_x);

            // Pauli Y
            let mut pauli_y = Array2::zeros((2, 2));
            pauli_y[[0, 1]] = Complex64::new(0.0, -sqrt_p);
            pauli_y[[1, 0]] = Complex64::new(0.0, sqrt_p);
            kraus_ops.push(pauli_y);

            // Pauli Z
            let mut pauli_z = Array2::zeros((2, 2));
            pauli_z[[0, 0]] = Complex64::new(sqrt_p, 0.0);
            pauli_z[[1, 1]] = Complex64::new(-sqrt_p, 0.0);
            kraus_ops.push(pauli_z);
        }

        Self {
            kraus_operators: kraus_ops,
            name: format!("Depolarizing({probability:.3})"),
        }
    }

    /// Create amplitude damping channel
    #[must_use]
    pub fn amplitude_damping(gamma: f64) -> Self {
        let mut kraus_ops = Vec::new();

        // K0 = |0⟩⟨0| + √(1-γ)|1⟩⟨1|
        let mut k0 = Array2::zeros((2, 2));
        k0[[0, 0]] = Complex64::new(1.0, 0.0);
        k0[[1, 1]] = Complex64::new((1.0 - gamma).sqrt(), 0.0);
        kraus_ops.push(k0);

        // K1 = √γ|0⟩⟨1|
        let mut k1 = Array2::zeros((2, 2));
        k1[[0, 1]] = Complex64::new(gamma.sqrt(), 0.0);
        kraus_ops.push(k1);

        Self {
            kraus_operators: kraus_ops,
            name: format!("AmplitudeDamping({gamma:.3})"),
        }
    }

    /// Create phase damping channel
    #[must_use]
    pub fn phase_damping(gamma: f64) -> Self {
        let mut kraus_ops = Vec::new();

        // K0 = |0⟩⟨0| + √(1-γ)|1⟩⟨1|
        let mut k0 = Array2::zeros((2, 2));
        k0[[0, 0]] = Complex64::new(1.0, 0.0);
        k0[[1, 1]] = Complex64::new((1.0 - gamma).sqrt(), 0.0);
        kraus_ops.push(k0);

        // K1 = √γ|1⟩⟨1|
        let mut k1 = Array2::zeros((2, 2));
        k1[[1, 1]] = Complex64::new(gamma.sqrt(), 0.0);
        kraus_ops.push(k1);

        Self {
            kraus_operators: kraus_ops,
            name: format!("PhaseDamping({gamma:.3})"),
        }
    }

    /// Apply channel to density matrix
    #[must_use]
    pub fn apply(&self, rho: &Array2<Complex64>) -> Array2<Complex64> {
        let dim = rho.nrows();
        let mut result = Array2::zeros((dim, dim));

        for kraus_op in &self.kraus_operators {
            let k_dag = kraus_op.t().mapv(|x| x.conj());
            result += &kraus_op.dot(rho).dot(&k_dag);
        }

        result
    }

    /// Verify channel is trace-preserving
    #[must_use]
    pub fn is_trace_preserving(&self) -> bool {
        let dim = self.kraus_operators[0].nrows();
        let mut sum = Array2::zeros((dim, dim));

        for kraus_op in &self.kraus_operators {
            let k_dag = kraus_op.t().mapv(|x| x.conj());
            sum += &k_dag.dot(kraus_op);
        }

        // Check if sum ≈ I
        let eye: Array2<Complex64> = Array2::eye(dim);
        (&sum - &eye).iter().all(|&x| x.norm() < 1e-10)
    }
}

/// Process tomography for quantum channel characterization
pub struct ProcessTomography {
    /// Input basis states
    pub input_states: Vec<Array2<Complex64>>,
    /// Output measurements
    pub output_measurements: Vec<Array2<Complex64>>,
    /// Reconstructed process matrix
    pub process_matrix: Option<Array2<Complex64>>,
}

impl ProcessTomography {
    /// Create standard process tomography setup
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let mut input_states = Vec::new();

        // Prepare complete basis of density matrices
        if num_qubits == 1 {
            // Single qubit: |0⟩, |1⟩, |+⟩, |+i⟩
            let mut rho_0 = Array2::zeros((2, 2));
            rho_0[[0, 0]] = Complex64::new(1.0, 0.0);
            input_states.push(rho_0);

            let mut rho_1 = Array2::zeros((2, 2));
            rho_1[[1, 1]] = Complex64::new(1.0, 0.0);
            input_states.push(rho_1);

            let mut rho_plus = Array2::zeros((2, 2));
            rho_plus[[0, 0]] = Complex64::new(0.5, 0.0);
            rho_plus[[0, 1]] = Complex64::new(0.5, 0.0);
            rho_plus[[1, 0]] = Complex64::new(0.5, 0.0);
            rho_plus[[1, 1]] = Complex64::new(0.5, 0.0);
            input_states.push(rho_plus);

            let mut rho_plus_i = Array2::zeros((2, 2));
            rho_plus_i[[0, 0]] = Complex64::new(0.5, 0.0);
            rho_plus_i[[0, 1]] = Complex64::new(0.0, -0.5);
            rho_plus_i[[1, 0]] = Complex64::new(0.0, 0.5);
            rho_plus_i[[1, 1]] = Complex64::new(0.5, 0.0);
            input_states.push(rho_plus_i);
        }

        Self {
            input_states,
            output_measurements: Vec::new(),
            process_matrix: None,
        }
    }

    /// Perform tomography on quantum channel
    pub fn characterize_channel(&mut self, channel: &QuantumChannel) -> Result<()> {
        self.output_measurements.clear();

        for input_state in &self.input_states {
            let output = channel.apply(input_state);
            self.output_measurements.push(output);
        }

        self.reconstruct_process_matrix()?;
        Ok(())
    }

    /// Reconstruct the process (χ) matrix from the measured outputs via linear
    /// inversion.
    ///
    /// The channel is expanded in the `n`-qubit Pauli operator basis
    /// `{P_m}` (`m = 0..d²`, `d = 2^n`):
    ///
    /// `ε(ρ) = Σ_{m,n} χ_{mn} P_m ρ P_n†`.
    ///
    /// For each prepared input `ρ_j` and its measured output `σ_j`, every
    /// matrix entry gives one linear equation in the unknowns `χ_{mn}`. The
    /// stacked system `A · vec(χ) = b` is solved in the least-squares sense via
    /// the normal equations `(A† A) vec(χ) = A† b`, which is exact when the
    /// input set is tomographically complete. An input set that is not
    /// tomographically complete makes `A† A` singular and yields an honest
    /// error instead of a fabricated all-zero matrix.
    fn reconstruct_process_matrix(&mut self) -> Result<()> {
        if self.input_states.is_empty() {
            return Err(SimulatorError::InvalidInput(
                "Process tomography requires at least one prepared input state".to_string(),
            ));
        }
        if self.output_measurements.len() != self.input_states.len() {
            return Err(SimulatorError::InvalidInput(
                "Number of output measurements must match the number of input states".to_string(),
            ));
        }

        let dim = self.input_states[0].nrows();
        if dim == 0 || !dim.is_power_of_two() {
            return Err(SimulatorError::InvalidInput(format!(
                "Input state dimension {dim} must be a positive power of two"
            )));
        }
        let num_qubits = dim.trailing_zeros() as usize;
        let process_dim = dim * dim; // number of Pauli basis operators = 4^n
        let num_inputs = self.input_states.len();

        // Tomographic completeness requires at least d² linearly independent
        // input states (each contributes d² equations for d⁴ unknowns).
        if num_inputs < process_dim {
            return Err(SimulatorError::InvalidInput(format!(
                "Process tomography is under-determined: {num_inputs} input states provided, \
                 at least {process_dim} tomographically complete states are required"
            )));
        }

        let basis = pauli_operator_basis(num_qubits);
        let basis_dag: Vec<Array2<Complex64>> =
            basis.iter().map(|p| p.t().mapv(|z| z.conj())).collect();

        let unknowns = process_dim * process_dim;
        let equations = num_inputs * dim * dim;
        let mut a = Array2::<Complex64>::zeros((equations, unknowns));
        let mut b = Array1::<Complex64>::zeros(equations);

        for (j, (rho, sigma)) in self
            .input_states
            .iter()
            .zip(self.output_measurements.iter())
            .enumerate()
        {
            let eq_base = j * dim * dim;
            for m in 0..process_dim {
                let pm_rho = basis[m].dot(rho);
                for n in 0..process_dim {
                    let mmn = pm_rho.dot(&basis_dag[n]);
                    let u = m * process_dim + n;
                    for r in 0..dim {
                        for c in 0..dim {
                            a[[eq_base + r * dim + c, u]] = mmn[[r, c]];
                        }
                    }
                }
            }
            for r in 0..dim {
                for c in 0..dim {
                    b[eq_base + r * dim + c] = sigma[[r, c]];
                }
            }
        }

        // Normal equations: (A† A) x = A† b.
        let a_dag = a.t().mapv(|z| z.conj());
        let aha = a_dag.dot(&a);
        let ahb = a_dag.dot(&b);
        let aha_inv = scirs2_linalg::complex::complex_inverse(&aha.view()).map_err(|e| {
            SimulatorError::InvalidInput(format!(
                "Input state set is not tomographically complete (singular system): {e}"
            ))
        })?;
        let x = aha_inv.dot(&ahb);

        let mut chi = Array2::<Complex64>::zeros((process_dim, process_dim));
        for m in 0..process_dim {
            for n in 0..process_dim {
                chi[[m, n]] = x[m * process_dim + n];
            }
        }

        self.process_matrix = Some(chi);
        Ok(())
    }

    /// Get process fidelity
    pub fn process_fidelity(&self, ideal_channel: &QuantumChannel) -> Result<f64> {
        if self.output_measurements.is_empty() {
            return Err(SimulatorError::InvalidOperation(
                "No measurements available for fidelity calculation".to_string(),
            ));
        }

        let mut fidelity_sum = 0.0;

        for (i, input_state) in self.input_states.iter().enumerate() {
            let ideal_output = ideal_channel.apply(input_state);
            let measured_output = &self.output_measurements[i];

            // Simplified fidelity calculation
            let fidelity = quantum_fidelity(measured_output, &ideal_output);
            fidelity_sum += fidelity;
        }

        Ok(fidelity_sum / self.input_states.len() as f64)
    }
}

/// Build the `n`-qubit Pauli operator basis `{I, X, Y, Z}^{⊗n}`.
///
/// Returns `4^n` matrices of dimension `2^n × 2^n`, ordered so that the basis
/// index is read as a base-4 number whose most-significant digit selects the
/// Pauli acting on the first qubit. `basis[0]` is the identity. Uses the
/// module-level [`kron`] helper for the tensor products.
fn pauli_operator_basis(num_qubits: usize) -> Vec<Array2<Complex64>> {
    let i2: Array2<Complex64> = Array2::eye(2);
    let mut px: Array2<Complex64> = Array2::zeros((2, 2));
    px[[0, 1]] = Complex64::new(1.0, 0.0);
    px[[1, 0]] = Complex64::new(1.0, 0.0);
    let mut py: Array2<Complex64> = Array2::zeros((2, 2));
    py[[0, 1]] = Complex64::new(0.0, -1.0);
    py[[1, 0]] = Complex64::new(0.0, 1.0);
    let mut pz: Array2<Complex64> = Array2::zeros((2, 2));
    pz[[0, 0]] = Complex64::new(1.0, 0.0);
    pz[[1, 1]] = Complex64::new(-1.0, 0.0);
    let singles = [i2, px, py, pz];

    let mut basis: Vec<Array2<Complex64>> = vec![Array2::<Complex64>::eye(1)];
    for _ in 0..num_qubits {
        let mut next = Vec::with_capacity(basis.len() * 4);
        for op in &basis {
            for s in &singles {
                next.push(kron(op, s));
            }
        }
        basis = next;
    }
    basis
}

/// Compute quantum fidelity between two density matrices
#[must_use]
pub fn quantum_fidelity(rho1: &Array2<Complex64>, rho2: &Array2<Complex64>) -> f64 {
    // F(ρ₁, ρ₂) = Tr(√(√ρ₁ ρ₂ √ρ₁))²
    // Simplified calculation for small systems

    // For pure states: F = |⟨ψ₁|ψ₂⟩|²
    // For mixed states, approximate with trace distance
    let trace_distance = (rho1 - rho2).iter().map(|x| x.norm()).sum::<f64>();
    0.5f64.mul_add(-trace_distance, 1.0).max(0.0)
}

/// Kronecker product of two matrices
fn kron(a: &Array2<Complex64>, b: &Array2<Complex64>) -> Array2<Complex64> {
    let (m1, n1) = a.dim();
    let (m2, n2) = b.dim();
    let mut result = Array2::zeros((m1 * m2, n1 * n2));

    for i in 0..m1 {
        for j in 0..n1 {
            for k in 0..m2 {
                for l in 0..n2 {
                    result[[i * m2 + k, j * n2 + l]] = a[[i, j]] * b[[k, l]];
                }
            }
        }
    }

    result
}

/// Noise model builder for common quantum channels
pub struct NoiseModelBuilder {
    channels: HashMap<String, QuantumChannel>,
    application_order: Vec<String>,
}

impl Default for NoiseModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseModelBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            application_order: Vec::new(),
        }
    }

    /// Add depolarizing noise
    #[must_use]
    pub fn depolarizing(mut self, name: &str, probability: f64) -> Self {
        let channel = QuantumChannel::depolarizing(1, probability);
        self.channels.insert(name.to_string(), channel);
        self.application_order.push(name.to_string());
        self
    }

    /// Add amplitude damping
    #[must_use]
    pub fn amplitude_damping(mut self, name: &str, gamma: f64) -> Self {
        let channel = QuantumChannel::amplitude_damping(gamma);
        self.channels.insert(name.to_string(), channel);
        self.application_order.push(name.to_string());
        self
    }

    /// Add phase damping
    #[must_use]
    pub fn phase_damping(mut self, name: &str, gamma: f64) -> Self {
        let channel = QuantumChannel::phase_damping(gamma);
        self.channels.insert(name.to_string(), channel);
        self.application_order.push(name.to_string());
        self
    }

    /// Build composite noise model
    #[must_use]
    pub fn build(self) -> CompositeNoiseModel {
        CompositeNoiseModel {
            channels: self.channels,
            application_order: self.application_order,
        }
    }
}

/// Composite noise model with multiple channels
#[derive(Debug, Clone)]
pub struct CompositeNoiseModel {
    channels: HashMap<String, QuantumChannel>,
    application_order: Vec<String>,
}

impl CompositeNoiseModel {
    /// Apply all noise channels in order
    #[must_use]
    pub fn apply(&self, rho: &Array2<Complex64>) -> Array2<Complex64> {
        let mut result = rho.clone();

        for channel_name in &self.application_order {
            if let Some(channel) = self.channels.get(channel_name) {
                result = channel.apply(&result);
            }
        }

        result
    }

    /// Get channel by name
    #[must_use]
    pub fn get_channel(&self, name: &str) -> Option<&QuantumChannel> {
        self.channels.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lindblad_simulator_creation() {
        let sim = LindladSimulator::new(2).expect("should create Lindblad simulator with 2 qubits");
        assert_eq!(sim.num_qubits, 2);
        assert_eq!(sim.density_matrix.shape(), [4, 4]);
    }

    #[test]
    fn test_depolarizing_channel() {
        let channel = QuantumChannel::depolarizing(1, 0.1);
        assert!(channel.is_trace_preserving());
        assert_eq!(channel.kraus_operators.len(), 4);
    }

    #[test]
    fn test_amplitude_damping() {
        let channel = QuantumChannel::amplitude_damping(0.2);
        assert!(channel.is_trace_preserving());

        // Test on |1⟩ state
        let mut rho_1 = Array2::zeros((2, 2));
        rho_1[[1, 1]] = Complex64::new(1.0, 0.0);

        let result = channel.apply(&rho_1);

        // Should have some population in |0⟩
        assert!(result[[0, 0]].re > 0.0);
        assert!(result[[1, 1]].re < 1.0);
    }

    #[test]
    fn test_noise_model_builder() {
        let noise_model = NoiseModelBuilder::new()
            .depolarizing("depol", 0.01)
            .amplitude_damping("amp_damp", 0.02)
            .build();

        assert!(noise_model.get_channel("depol").is_some());
        assert!(noise_model.get_channel("amp_damp").is_some());
    }

    #[test]
    fn test_pauli_basis_is_orthonormal_under_hs_inner_product() {
        // Tr(P_m† P_n) = d · δ_{mn} for the unnormalised Pauli basis.
        let basis = pauli_operator_basis(1);
        assert_eq!(basis.len(), 4);
        for (m, pm) in basis.iter().enumerate() {
            for (n, pn) in basis.iter().enumerate() {
                let prod = pm.t().mapv(|z| z.conj()).dot(pn);
                let trace: Complex64 = (0..2).map(|i| prod[[i, i]]).sum();
                let expected = if m == n { 2.0 } else { 0.0 };
                assert!((trace.re - expected).abs() < 1e-12 && trace.im.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_process_tomography_reconstructs_identity_channel() {
        // Regression: reconstruct_process_matrix previously stored an all-zero
        // chi matrix. The identity channel must reconstruct to chi_00 = 1
        // (the I⊗I component) with everything else ~ 0.
        let mut tomography = ProcessTomography::new(1);
        let identity = QuantumChannel::depolarizing(1, 0.0); // only the identity Kraus survives
        tomography
            .characterize_channel(&identity)
            .expect("characterization should succeed");

        let chi = tomography
            .process_matrix
            .as_ref()
            .expect("process matrix must be populated");
        assert_eq!(chi.shape(), [4, 4]);
        assert!(
            (chi[[0, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-9,
            "chi_00 should be 1, got {}",
            chi[[0, 0]]
        );
        for m in 0..4 {
            for n in 0..4 {
                if (m, n) != (0, 0) {
                    assert!(
                        chi[[m, n]].norm() < 1e-9,
                        "chi[{m},{n}] should be ~0, got {}",
                        chi[[m, n]]
                    );
                }
            }
        }
    }

    #[test]
    fn test_process_tomography_reconstructs_bit_flip_channel() {
        // A pure X channel ε(ρ) = X ρ X must reconstruct to chi_11 = 1
        // (the X⊗X component), proving the linear inversion is real.
        let mut x_op: Array2<Complex64> = Array2::zeros((2, 2));
        x_op[[0, 1]] = Complex64::new(1.0, 0.0);
        x_op[[1, 0]] = Complex64::new(1.0, 0.0);
        let x_channel = QuantumChannel {
            kraus_operators: vec![x_op],
            name: "X".to_string(),
        };

        let mut tomography = ProcessTomography::new(1);
        tomography
            .characterize_channel(&x_channel)
            .expect("characterization should succeed");

        let chi = tomography
            .process_matrix
            .as_ref()
            .expect("process matrix must be populated");
        assert!(
            (chi[[1, 1]] - Complex64::new(1.0, 0.0)).norm() < 1e-9,
            "chi_11 should be 1 for the X channel, got {}",
            chi[[1, 1]]
        );
        assert!(
            chi[[0, 0]].norm() < 1e-9,
            "chi_00 should be ~0 for X channel"
        );
    }
}
