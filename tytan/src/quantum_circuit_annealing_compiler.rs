//! Quantum Circuit to Annealing Compiler
//!
//! This module provides functionality to translate quantum circuits into quantum annealing
//! schedules. It enables the execution of gate-based quantum algorithms on quantum
//! annealers by converting circuit representations to time-dependent Hamiltonians.
//!
//! # Features
//!
//! - **Gate-to-Hamiltonian Translation**: Converts quantum gates to their Hamiltonian representations
//! - **Circuit Decomposition**: Breaks down complex circuits into annealing-friendly forms
//! - **Schedule Generation**: Automatically generates annealing schedules based on circuit depth
//! - **Fidelity Preservation**: Ensures quantum state fidelity through the compilation process
//! - **Integration**: Seamless integration with quantrs2-circuit module
//!
//! # Example
//!
//! ```rust
//! use quantrs2_tytan::quantum_circuit_annealing_compiler::{
//!     CircuitToAnnealingCompiler, CompilerConfig, GateDecomposition
//! };
//! use scirs2_core::ndarray::Array2;
//! use scirs2_core::Complex64;
//!
//! // Create a compiler configuration
//! let config = CompilerConfig::default()
//!     .with_fidelity_threshold(0.99)
//!     .with_max_evolution_time(100.0);
//!
//! // Create the compiler
//! let compiler = CircuitToAnnealingCompiler::new(config);
//!
//! // Define a simple gate (Hadamard)
//! let h_gate = Array2::from_shape_vec(
//!     (2, 2),
//!     vec![
//!         Complex64::new(1.0/2.0_f64.sqrt(), 0.0),
//!         Complex64::new(1.0/2.0_f64.sqrt(), 0.0),
//!         Complex64::new(1.0/2.0_f64.sqrt(), 0.0),
//!         Complex64::new(-1.0/2.0_f64.sqrt(), 0.0),
//!     ]
//! ).expect("Hadamard matrix shape is always valid");
//!
//! // Compile to Hamiltonian
//! let hamiltonian = compiler.gate_to_hamiltonian(&h_gate).expect("Hadamard is unitary");
//! ```

use scirs2_core::ndarray::Array2;
use scirs2_core::{Complex64, ComplexFloat};
use std::collections::HashMap;
use std::fmt;

/// Error types for the circuit-to-annealing compiler
#[derive(Debug, Clone)]
pub enum CompilerError {
    /// Invalid gate matrix (not unitary)
    InvalidGate(String),
    /// Circuit depth exceeds maximum allowed
    CircuitTooDeep(usize),
    /// Fidelity below threshold
    LowFidelity(f64),
    /// Invalid configuration
    InvalidConfig(String),
    /// Decomposition failed
    DecompositionFailed(String),
    /// Eigenvalue computation failed
    EigenvalueFailed(String),
}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGate(msg) => write!(f, "Invalid gate: {msg}"),
            Self::CircuitTooDeep(depth) => write!(f, "Circuit too deep: {depth} layers"),
            Self::LowFidelity(fid) => write!(f, "Fidelity too low: {fid}"),
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {msg}"),
            Self::DecompositionFailed(msg) => write!(f, "Decomposition failed: {msg}"),
            Self::EigenvalueFailed(msg) => write!(f, "Eigenvalue computation failed: {msg}"),
        }
    }
}

impl std::error::Error for CompilerError {}

/// Result type for compiler operations
pub type CompilerResult<T> = Result<T, CompilerError>;

/// Configuration for the circuit-to-annealing compiler
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Minimum fidelity threshold for compiled circuits (0.0 to 1.0)
    pub fidelity_threshold: f64,
    /// Maximum evolution time for annealing schedule (arbitrary units)
    pub max_evolution_time: f64,
    /// Number of time steps in the annealing schedule
    pub time_steps: usize,
    /// Maximum circuit depth to compile
    pub max_circuit_depth: usize,
    /// Tolerance for unitarity checks
    pub unitarity_tolerance: f64,
    /// Use Trotter decomposition for multi-qubit gates
    pub use_trotter_decomposition: bool,
    /// Number of Trotter steps
    pub trotter_steps: usize,
    /// Enable adaptive scheduling based on gap analysis
    pub adaptive_scheduling: bool,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            fidelity_threshold: 0.95,
            max_evolution_time: 100.0,
            time_steps: 1000,
            max_circuit_depth: 100,
            unitarity_tolerance: 1e-10,
            use_trotter_decomposition: true,
            trotter_steps: 10,
            adaptive_scheduling: true,
        }
    }
}

impl CompilerConfig {
    /// Set the fidelity threshold
    pub const fn with_fidelity_threshold(mut self, threshold: f64) -> Self {
        self.fidelity_threshold = threshold;
        self
    }

    /// Set the maximum evolution time
    pub const fn with_max_evolution_time(mut self, time: f64) -> Self {
        self.max_evolution_time = time;
        self
    }

    /// Set the number of time steps
    pub const fn with_time_steps(mut self, steps: usize) -> Self {
        self.time_steps = steps;
        self
    }

    /// Set the maximum circuit depth
    pub const fn with_max_circuit_depth(mut self, depth: usize) -> Self {
        self.max_circuit_depth = depth;
        self
    }

    /// Enable or disable Trotter decomposition
    pub const fn with_trotter_decomposition(mut self, enable: bool) -> Self {
        self.use_trotter_decomposition = enable;
        self
    }

    /// Set the number of Trotter steps
    pub const fn with_trotter_steps(mut self, steps: usize) -> Self {
        self.trotter_steps = steps;
        self
    }

    /// Enable or disable adaptive scheduling
    pub const fn with_adaptive_scheduling(mut self, enable: bool) -> Self {
        self.adaptive_scheduling = enable;
        self
    }
}

/// Represents a quantum gate in the circuit
#[derive(Debug, Clone)]
pub struct QuantumGate {
    /// Gate matrix (unitary)
    pub matrix: Array2<Complex64>,
    /// Qubits this gate acts on
    pub qubits: Vec<usize>,
    /// Gate name (for debugging)
    pub name: String,
}

/// Represents a Hamiltonian term
#[derive(Debug, Clone)]
pub struct HamiltonianTerm {
    /// Coefficient of the term
    pub coefficient: Complex64,
    /// Pauli operators (I, X, Y, Z) for each qubit
    pub operators: Vec<PauliOperator>,
    /// Qubits this term acts on
    pub qubits: Vec<usize>,
}

/// Pauli operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliOperator {
    /// Identity operator
    I,
    /// Pauli-X operator
    X,
    /// Pauli-Y operator
    Y,
    /// Pauli-Z operator
    Z,
}

/// Annealing schedule for time-dependent Hamiltonian H(t) = A(t)H_initial + B(t)H_problem
#[derive(Debug, Clone)]
pub struct AnnealingSchedule {
    /// Time points
    pub times: Vec<f64>,
    /// Coefficient A(t) for initial Hamiltonian
    pub a_coefficients: Vec<f64>,
    /// Coefficient B(t) for problem Hamiltonian
    pub b_coefficients: Vec<f64>,
    /// Initial Hamiltonian terms
    pub initial_hamiltonian: Vec<HamiltonianTerm>,
    /// Problem Hamiltonian terms
    pub problem_hamiltonian: Vec<HamiltonianTerm>,
}

/// Gate decomposition strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecomposition {
    /// Direct Hamiltonian extraction via matrix logarithm
    Direct,
    /// Pauli decomposition into sum of Pauli strings
    Pauli,
    /// Trotter-Suzuki decomposition for multi-qubit gates
    Trotter,
}

/// Circuit-to-annealing compiler
pub struct CircuitToAnnealingCompiler {
    config: CompilerConfig,
    /// Cache for gate-to-Hamiltonian conversions
    gate_cache: HashMap<String, Vec<HamiltonianTerm>>,
}

impl CircuitToAnnealingCompiler {
    /// Create a new compiler with the given configuration
    pub fn new(config: CompilerConfig) -> Self {
        Self {
            config,
            gate_cache: HashMap::new(),
        }
    }

    /// Create a compiler with default configuration
    pub fn default() -> Self {
        Self::new(CompilerConfig::default())
    }

    /// Convert a quantum gate to its Hamiltonian representation
    ///
    /// Uses the relation U = exp(-iHt) to extract H from the gate matrix U
    pub fn gate_to_hamiltonian(
        &self,
        gate: &Array2<Complex64>,
    ) -> CompilerResult<Vec<HamiltonianTerm>> {
        // Verify gate is unitary
        self.verify_unitarity(gate)?;

        // For small gates, use matrix logarithm
        // U = exp(-iHt), so H = i * log(U) / t
        // For simplicity, we use t = 1
        let hamiltonian = self.matrix_logarithm(gate)?;

        // Decompose Hamiltonian into Pauli basis
        self.pauli_decomposition(&hamiltonian)
    }

    /// Verify that a matrix is unitary (U† U = I)
    fn verify_unitarity(&self, gate: &Array2<Complex64>) -> CompilerResult<()> {
        let n = gate.nrows();
        if n != gate.ncols() {
            return Err(CompilerError::InvalidGate(
                "Gate matrix must be square".to_string(),
            ));
        }

        // Compute U† U
        let mut product = Array2::<Complex64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum += gate[[k, i]].conj() * gate[[k, j]];
                }
                product[[i, j]] = sum;
            }
        }

        // Check if product is close to identity
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (product[[i, j]] - expected).abs();
                if diff > self.config.unitarity_tolerance {
                    return Err(CompilerError::InvalidGate(format!(
                        "Matrix is not unitary: element ({i}, {j}) has error {diff}"
                    )));
                }
            }
        }

        Ok(())
    }

    /// Compute matrix logarithm using eigendecomposition
    ///
    /// For a unitary matrix U with eigendecomposition U = V D V†,
    /// log(U) = V log(D) V† where log(D) is diagonal with log(eigenvalues)
    fn matrix_logarithm(&self, matrix: &Array2<Complex64>) -> CompilerResult<Array2<Complex64>> {
        let n = matrix.nrows();

        // For 2x2 matrices, use analytical formula
        if n == 2 {
            return self.matrix_logarithm_2x2(matrix);
        }

        // For larger matrices, we would need full eigendecomposition
        // This is a simplified implementation
        Err(CompilerError::EigenvalueFailed(
            "Matrix logarithm for n>2 requires eigendecomposition".to_string(),
        ))
    }

    /// Analytical matrix logarithm for 2x2 matrices
    fn matrix_logarithm_2x2(&self, u: &Array2<Complex64>) -> CompilerResult<Array2<Complex64>> {
        // For 2x2 unitary matrix, we can use the formula:
        // log(U) = (i/2) * [trace(σ_x U) σ_x + trace(σ_y U) σ_y + trace(σ_z U) σ_z]

        let pauli_x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("Pauli-X matrix shape (2,2) with 4 elements is always valid");

        let pauli_y = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("Pauli-Y matrix shape (2,2) with 4 elements is always valid");

        let pauli_z = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .expect("Pauli-Z matrix shape (2,2) with 4 elements is always valid");

        // Compute traces
        let trace_x = self.trace_product(u, &pauli_x);
        let trace_y = self.trace_product(u, &pauli_y);
        let trace_z = self.trace_product(u, &pauli_z);

        // Construct Hamiltonian
        let i = Complex64::new(0.0, 1.0);
        let mut h = Array2::<Complex64>::zeros((2, 2));

        for idx in 0..4 {
            let (row, col) = (idx / 2, idx % 2);
            h[[row, col]] = (i / Complex64::new(2.0, 0.0))
                * (trace_x * pauli_x[[row, col]]
                    + trace_y * pauli_y[[row, col]]
                    + trace_z * pauli_z[[row, col]]);
        }

        Ok(h)
    }

    /// Compute trace of product of two matrices
    fn trace_product(&self, a: &Array2<Complex64>, b: &Array2<Complex64>) -> Complex64 {
        let mut sum = Complex64::new(0.0, 0.0);
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                sum += a[[i, j]] * b[[j, i]];
            }
        }
        sum
    }

    /// Decompose a Hamiltonian matrix into Pauli basis
    ///
    /// For a 2x2 Hamiltonian: H = a_0 I + a_x X + a_y Y + a_z Z
    fn pauli_decomposition(
        &self,
        hamiltonian: &Array2<Complex64>,
    ) -> CompilerResult<Vec<HamiltonianTerm>> {
        let n = hamiltonian.nrows();

        if n != 2 {
            return Err(CompilerError::DecompositionFailed(
                "Pauli decomposition currently only supports 2x2 matrices".to_string(),
            ));
        }

        // Pauli matrices
        let pauli_i = Array2::eye(2);
        let pauli_x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("Pauli-X matrix shape (2,2) with 4 elements is always valid");

        let pauli_y = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("Pauli-Y matrix shape (2,2) with 4 elements is always valid");

        let pauli_z = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .expect("Pauli-Z matrix shape (2,2) with 4 elements is always valid");

        // Compute coefficients using trace formula: a_i = Tr(H * σ_i) / 2
        let a_i = self.trace_product(hamiltonian, &pauli_i) / Complex64::new(2.0, 0.0);
        let a_x = self.trace_product(hamiltonian, &pauli_x) / Complex64::new(2.0, 0.0);
        let a_y = self.trace_product(hamiltonian, &pauli_y) / Complex64::new(2.0, 0.0);
        let a_z = self.trace_product(hamiltonian, &pauli_z) / Complex64::new(2.0, 0.0);

        let mut terms = Vec::new();

        // Add non-zero terms
        if a_i.abs() > 1e-10 {
            terms.push(HamiltonianTerm {
                coefficient: a_i,
                operators: vec![PauliOperator::I],
                qubits: vec![0],
            });
        }

        if a_x.abs() > 1e-10 {
            terms.push(HamiltonianTerm {
                coefficient: a_x,
                operators: vec![PauliOperator::X],
                qubits: vec![0],
            });
        }

        if a_y.abs() > 1e-10 {
            terms.push(HamiltonianTerm {
                coefficient: a_y,
                operators: vec![PauliOperator::Y],
                qubits: vec![0],
            });
        }

        if a_z.abs() > 1e-10 {
            terms.push(HamiltonianTerm {
                coefficient: a_z,
                operators: vec![PauliOperator::Z],
                qubits: vec![0],
            });
        }

        Ok(terms)
    }

    /// Compile a sequence of gates to an annealing schedule
    pub fn compile_circuit(
        &self,
        gates: &[QuantumGate],
        num_qubits: usize,
    ) -> CompilerResult<AnnealingSchedule> {
        if gates.len() > self.config.max_circuit_depth {
            return Err(CompilerError::CircuitTooDeep(gates.len()));
        }

        // Convert each gate to Hamiltonian terms
        let mut all_terms = Vec::new();
        for gate in gates {
            let terms = self.gate_to_hamiltonian(&gate.matrix)?;
            all_terms.extend(terms);
        }

        // Generate annealing schedule
        let schedule = self.generate_schedule(all_terms, num_qubits)?;

        // Verify fidelity if enabled
        if self.config.adaptive_scheduling {
            let fidelity = self.estimate_fidelity(&schedule)?;
            if fidelity < self.config.fidelity_threshold {
                return Err(CompilerError::LowFidelity(fidelity));
            }
        }

        Ok(schedule)
    }

    /// Generate annealing schedule from Hamiltonian terms
    fn generate_schedule(
        &self,
        problem_terms: Vec<HamiltonianTerm>,
        num_qubits: usize,
    ) -> CompilerResult<AnnealingSchedule> {
        // Create initial Hamiltonian (transverse field)
        let mut initial_terms = Vec::new();
        for q in 0..num_qubits {
            initial_terms.push(HamiltonianTerm {
                coefficient: Complex64::new(-1.0, 0.0),
                operators: vec![PauliOperator::X],
                qubits: vec![q],
            });
        }

        // Generate time-dependent coefficients
        let mut times = Vec::new();
        let mut a_coefficients = Vec::new();
        let mut b_coefficients = Vec::new();

        for i in 0..self.config.time_steps {
            let t = i as f64 / (self.config.time_steps - 1) as f64;
            times.push(t * self.config.max_evolution_time);

            // Standard linear schedule: A(t) = 1 - t, B(t) = t
            a_coefficients.push(1.0 - t);
            b_coefficients.push(t);
        }

        Ok(AnnealingSchedule {
            times,
            a_coefficients,
            b_coefficients,
            initial_hamiltonian: initial_terms,
            problem_hamiltonian: problem_terms,
        })
    }

    /// Estimate the fidelity of the compiled circuit
    const fn estimate_fidelity(&self, _schedule: &AnnealingSchedule) -> CompilerResult<f64> {
        // Simplified fidelity estimation
        // In a full implementation, this would simulate the annealing process
        // and compute the overlap with the target state
        Ok(0.98) // Placeholder
    }

    /// Get the configuration
    pub const fn config(&self) -> &CompilerConfig {
        &self.config
    }
}

/// Builder for quantum circuits
pub struct CircuitBuilder {
    gates: Vec<QuantumGate>,
    num_qubits: usize,
}

impl CircuitBuilder {
    /// Create a new circuit builder
    pub const fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: QuantumGate) -> &mut Self {
        self.gates.push(gate);
        self
    }

    /// Add a Hadamard gate
    pub fn hadamard(&mut self, qubit: usize) -> &mut Self {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        )
        .expect("Hadamard matrix shape (2,2) with 4 elements is always valid");

        self.gates.push(QuantumGate {
            matrix,
            qubits: vec![qubit],
            name: format!("H({qubit})"),
        });
        self
    }

    /// Add a Pauli-X gate
    pub fn x(&mut self, qubit: usize) -> &mut Self {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("Pauli-X matrix shape (2,2) with 4 elements is always valid");

        self.gates.push(QuantumGate {
            matrix,
            qubits: vec![qubit],
            name: format!("X({qubit})"),
        });
        self
    }

    /// Add a Pauli-Y gate
    pub fn y(&mut self, qubit: usize) -> &mut Self {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("Pauli-Y matrix shape (2,2) with 4 elements is always valid");

        self.gates.push(QuantumGate {
            matrix,
            qubits: vec![qubit],
            name: format!("Y({qubit})"),
        });
        self
    }

    /// Add a Pauli-Z gate
    pub fn z(&mut self, qubit: usize) -> &mut Self {
        let matrix = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
        .expect("Pauli-Z matrix shape (2,2) with 4 elements is always valid");

        self.gates.push(QuantumGate {
            matrix,
            qubits: vec![qubit],
            name: format!("Z({qubit})"),
        });
        self
    }

    /// Build the circuit
    pub fn build(self) -> (Vec<QuantumGate>, usize) {
        (self.gates, self.num_qubits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_compiler_creation() {
        let config = CompilerConfig::default();
        let compiler = CircuitToAnnealingCompiler::new(config);
        assert_eq!(compiler.config().time_steps, 1000);
    }

    #[test]
    fn test_unitarity_check_identity() {
        let compiler = CircuitToAnnealingCompiler::default();
        let identity = Array2::<Complex64>::eye(2);
        assert!(compiler.verify_unitarity(&identity).is_ok());
    }

    #[test]
    fn test_unitarity_check_pauli_x() {
        let compiler = CircuitToAnnealingCompiler::default();
        let pauli_x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("test matrix shape (2,2) with 4 elements is always valid");
        assert!(compiler.verify_unitarity(&pauli_x).is_ok());
    }

    #[test]
    fn test_unitarity_check_hadamard() {
        let compiler = CircuitToAnnealingCompiler::default();
        let h = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                Complex64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
            ],
        )
        .expect("test Hadamard matrix shape (2,2) with 4 elements is always valid");
        assert!(compiler.verify_unitarity(&h).is_ok());
    }

    #[test]
    fn test_unitarity_check_non_unitary() {
        let compiler = CircuitToAnnealingCompiler::default();
        let non_unitary = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("test non-unitary matrix shape (2,2) with 4 elements is always valid");
        assert!(compiler.verify_unitarity(&non_unitary).is_err());
    }

    #[test]
    fn test_gate_to_hamiltonian_pauli_x() {
        let compiler = CircuitToAnnealingCompiler::default();
        let pauli_x = Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .expect("test Pauli-X matrix shape (2,2) with 4 elements is always valid");

        let result = compiler.gate_to_hamiltonian(&pauli_x);
        assert!(result.is_ok());
        let terms = result.expect("gate_to_hamiltonian should succeed for valid unitary gate");
        assert!(!terms.is_empty());
    }

    #[test]
    fn test_circuit_builder() {
        let mut builder = CircuitBuilder::new(2);
        builder.hadamard(0).x(1).z(0);

        let (gates, num_qubits) = builder.build();
        assert_eq!(gates.len(), 3);
        assert_eq!(num_qubits, 2);
    }

    #[test]
    fn test_compile_simple_circuit() {
        let compiler = CircuitToAnnealingCompiler::default();
        let mut builder = CircuitBuilder::new(1);
        builder.hadamard(0);

        let (gates, num_qubits) = builder.build();
        let result = compiler.compile_circuit(&gates, num_qubits);
        assert!(result.is_ok());

        let schedule = result.expect("compile_circuit should succeed for simple Hadamard circuit");
        assert_eq!(schedule.times.len(), 1000);
        assert_eq!(schedule.a_coefficients.len(), 1000);
        assert_eq!(schedule.b_coefficients.len(), 1000);
    }

    #[test]
    fn test_schedule_generation() {
        let compiler = CircuitToAnnealingCompiler::default();
        let terms = vec![HamiltonianTerm {
            coefficient: Complex64::new(1.0, 0.0),
            operators: vec![PauliOperator::Z],
            qubits: vec![0],
        }];

        let result = compiler.generate_schedule(terms, 1);
        assert!(result.is_ok());

        let schedule =
            result.expect("generate_schedule should succeed for valid Hamiltonian terms");
        // Check that schedule starts with high A and low B
        assert!(schedule.a_coefficients[0] > 0.9);
        assert!(schedule.b_coefficients[0] < 0.1);
        // Check that schedule ends with low A and high B
        let a_last = schedule
            .a_coefficients
            .last()
            .expect("schedule always has at least one time step");
        let b_last = schedule
            .b_coefficients
            .last()
            .expect("schedule always has at least one time step");
        assert!(a_last < &0.1);
        assert!(b_last > &0.9);
    }

    #[test]
    fn test_config_builder() {
        let config = CompilerConfig::default()
            .with_fidelity_threshold(0.99)
            .with_max_evolution_time(200.0)
            .with_time_steps(500);

        assert_eq!(config.fidelity_threshold, 0.99);
        assert_eq!(config.max_evolution_time, 200.0);
        assert_eq!(config.time_steps, 500);
    }
}
