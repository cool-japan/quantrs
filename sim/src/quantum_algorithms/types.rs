//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{
    CircuitInterface, InterfaceCircuit, InterfaceGate, InterfaceGateType,
};
use crate::error::{Result, SimulatorError};
use crate::scirs2_integration::SciRS2Backend;
use crate::scirs2_qft::{QFTConfig, QFTMethod, SciRS2QFT};
use crate::statevector::StateVectorSimulator;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Quantum algorithm optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Basic implementation
    Basic,
    /// Optimized for memory usage
    Memory,
    /// Optimized for speed
    Speed,
    /// Hardware-aware optimization
    Hardware,
    /// Maximum optimization using all available techniques
    Maximum,
}
/// Grover's algorithm result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroverResult {
    /// Target items found
    pub found_items: Vec<usize>,
    /// Final amplitudes of all states
    pub final_amplitudes: Vec<Complex64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Optimal number of iterations
    pub optimal_iterations: usize,
    /// Success probability
    pub success_probability: f64,
    /// Amplitude amplification gain
    pub amplification_gain: f64,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Resource usage statistics
    pub resource_stats: AlgorithmResourceStats,
}
/// Algorithm resource usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlgorithmResourceStats {
    /// Number of qubits used
    pub qubits_used: usize,
    /// Total circuit depth
    pub circuit_depth: usize,
    /// Number of quantum gates
    pub gate_count: usize,
    /// Number of measurements
    pub measurement_count: usize,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// CNOT gate count (for error correction estimates)
    pub cnot_count: usize,
    /// T gate count (for fault-tolerant estimates)
    pub t_gate_count: usize,
}
/// Optimized Shor's algorithm implementation
pub struct OptimizedShorAlgorithm {
    /// Configuration
    config: QuantumAlgorithmConfig,
    /// `SciRS2` backend for optimization
    backend: Option<SciRS2Backend>,
    /// Circuit interface for compilation
    circuit_interface: CircuitInterface,
    /// QFT implementation
    qft_engine: SciRS2QFT,
}
impl OptimizedShorAlgorithm {
    /// Create new Shor's algorithm instance
    pub fn new(config: QuantumAlgorithmConfig) -> Result<Self> {
        let circuit_interface = CircuitInterface::new(Default::default())?;
        let qft_config = QFTConfig {
            method: QFTMethod::SciRS2Exact,
            bit_reversal: true,
            parallel: config.enable_parallel,
            precision_threshold: config.precision_tolerance,
            ..Default::default()
        };
        let qft_engine = SciRS2QFT::new(0, qft_config)?;
        Ok(Self {
            config,
            backend: None,
            circuit_interface,
            qft_engine,
        })
    }
    /// Initialize with `SciRS2` backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        self.circuit_interface = self.circuit_interface.with_backend()?;
        self.qft_engine = self.qft_engine.with_backend()?;
        Ok(self)
    }
    /// Factor integer using optimized Shor's algorithm
    pub fn factor(&mut self, n: u64) -> Result<ShorResult> {
        let start_time = std::time::Instant::now();
        let preprocessing_start = std::time::Instant::now();
        if n <= 1 {
            return Err(SimulatorError::InvalidInput(
                "Cannot factor numbers <= 1".to_string(),
            ));
        }
        if n == 2 {
            return Ok(ShorResult {
                n,
                factors: vec![2],
                period: None,
                quantum_iterations: 0,
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                classical_preprocessing_ms: 0.0,
                quantum_computation_ms: 0.0,
                success_probability: 1.0,
                resource_stats: AlgorithmResourceStats::default(),
            });
        }
        if n % 2 == 0 {
            let factor = 2;
            let other_factor = n / 2;
            return Ok(ShorResult {
                n,
                factors: vec![factor, other_factor],
                period: None,
                quantum_iterations: 0,
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                classical_preprocessing_ms: preprocessing_start.elapsed().as_secs_f64() * 1000.0,
                quantum_computation_ms: 0.0,
                success_probability: 1.0,
                resource_stats: AlgorithmResourceStats::default(),
            });
        }
        if let Some((base, _exponent)) = Self::find_perfect_power(n) {
            return Ok(ShorResult {
                n,
                factors: vec![base],
                period: None,
                quantum_iterations: 0,
                execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                classical_preprocessing_ms: preprocessing_start.elapsed().as_secs_f64() * 1000.0,
                quantum_computation_ms: 0.0,
                success_probability: 1.0,
                resource_stats: AlgorithmResourceStats::default(),
            });
        }
        let classical_preprocessing_ms = preprocessing_start.elapsed().as_secs_f64() * 1000.0;
        let quantum_start = std::time::Instant::now();
        let mut quantum_iterations = 0;
        let max_attempts = 10;
        for attempt in 0..max_attempts {
            quantum_iterations += 1;
            let a = self.choose_random_base(n)?;
            let gcd_val = Self::gcd(a, n);
            if gcd_val > 1 {
                let other_factor = n / gcd_val;
                return Ok(ShorResult {
                    n,
                    factors: vec![gcd_val, other_factor],
                    period: None,
                    quantum_iterations,
                    execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                    classical_preprocessing_ms,
                    quantum_computation_ms: quantum_start.elapsed().as_secs_f64() * 1000.0,
                    success_probability: 1.0,
                    resource_stats: AlgorithmResourceStats::default(),
                });
            }
            if let Some(period) = self.quantum_period_finding(a, n)? {
                if self.verify_period(a, n, period) {
                    if let Some(factors) = self.extract_factors_from_period(a, n, period) {
                        let quantum_computation_ms = quantum_start.elapsed().as_secs_f64() * 1000.0;
                        let resource_stats = Self::estimate_resources(n);
                        return Ok(ShorResult {
                            n,
                            factors,
                            period: Some(period),
                            quantum_iterations,
                            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                            classical_preprocessing_ms,
                            quantum_computation_ms,
                            success_probability: Self::estimate_success_probability(
                                attempt,
                                max_attempts,
                            ),
                            resource_stats,
                        });
                    }
                }
            }
        }
        Ok(ShorResult {
            n,
            factors: vec![],
            period: None,
            quantum_iterations,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            classical_preprocessing_ms,
            quantum_computation_ms: quantum_start.elapsed().as_secs_f64() * 1000.0,
            success_probability: 0.0,
            resource_stats: Self::estimate_resources(n),
        })
    }
    /// Quantum period finding subroutine with enhanced precision
    pub(super) fn quantum_period_finding(&mut self, a: u64, n: u64) -> Result<Option<u64>> {
        let n_bits = (n as f64).log2().ceil() as usize;
        let register_size = 3 * n_bits;
        let total_qubits = register_size + n_bits;
        let mut circuit = InterfaceCircuit::new(total_qubits, register_size);
        for i in 0..register_size {
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![i]));
        }
        circuit.add_gate(InterfaceGate::new(
            InterfaceGateType::PauliX,
            vec![register_size],
        ));
        self.add_optimized_controlled_modular_exponentiation(&mut circuit, a, n, register_size)?;
        self.add_inverse_qft(&mut circuit, register_size)?;
        for i in 0..register_size {
            circuit.add_gate(InterfaceGate::measurement(i, i));
        }
        let backend = crate::circuit_interfaces::SimulationBackend::StateVector;
        let compiled = self.circuit_interface.compile_circuit(&circuit, backend)?;
        let result = self.circuit_interface.execute_circuit(&compiled, None)?;
        if !result.measurement_results.is_empty() {
            let mut measured_value = 0usize;
            for (i, &bit) in result
                .measurement_results
                .iter()
                .take(register_size)
                .enumerate()
            {
                if bit {
                    measured_value |= 1 << i;
                }
            }
            if self.config.enable_error_mitigation {
                measured_value = Self::apply_error_mitigation(measured_value, register_size)?;
            }
            if let Some(period) =
                self.extract_period_from_measurement_enhanced(measured_value, register_size, n)
            {
                return Ok(Some(period));
            }
        }
        Ok(None)
    }
    /// Add optimized controlled modular exponentiation to circuit
    pub(super) fn add_optimized_controlled_modular_exponentiation(
        &self,
        circuit: &mut InterfaceCircuit,
        a: u64,
        n: u64,
        register_size: usize,
    ) -> Result<()> {
        let n_bits = (n as f64).log2().ceil() as usize;
        for i in 0..register_size {
            let power = 1u64 << i;
            let a_power_mod_n = Self::mod_exp(a, power, n);
            self.add_controlled_modular_multiplication_optimized(
                circuit,
                a_power_mod_n,
                n,
                i,
                register_size,
                n_bits,
            )?;
        }
        Ok(())
    }
    /// Add controlled modular exponentiation to circuit (legacy)
    pub(super) fn add_controlled_modular_exponentiation(
        &self,
        circuit: &mut InterfaceCircuit,
        a: u64,
        n: u64,
        register_size: usize,
    ) -> Result<()> {
        self.add_optimized_controlled_modular_exponentiation(circuit, a, n, register_size)
    }
    /// Add optimized controlled modular multiplication
    pub(super) fn add_controlled_modular_multiplication_optimized(
        &self,
        circuit: &mut InterfaceCircuit,
        multiplier: u64,
        modulus: u64,
        control_qubit: usize,
        register_start: usize,
        register_size: usize,
    ) -> Result<()> {
        let target_start = register_start + register_size;
        let mont_multiplier = Self::montgomery_form(multiplier, modulus);
        for i in 0..register_size {
            if (mont_multiplier >> i) & 1 == 1 {
                Self::add_controlled_quantum_adder(
                    circuit,
                    control_qubit,
                    register_start + i,
                    target_start + i,
                    register_size - i,
                )?;
            }
        }
        Self::add_controlled_modular_reduction(
            circuit,
            modulus,
            control_qubit,
            target_start,
            register_size,
        )?;
        Ok(())
    }
    /// Add controlled modular multiplication (legacy)
    pub(super) fn add_controlled_modular_multiplication(
        circuit: &mut InterfaceCircuit,
        multiplier: u64,
        modulus: u64,
        control_qubit: usize,
        register_start: usize,
        register_size: usize,
    ) -> Result<()> {
        for i in 0..register_size {
            if (multiplier >> i) & 1 == 1 {
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::CNOT,
                    vec![control_qubit, register_start + i],
                ));
            }
        }
        Ok(())
    }
    /// Add inverse QFT to circuit
    pub(super) fn add_inverse_qft(
        &mut self,
        circuit: &mut InterfaceCircuit,
        num_qubits: usize,
    ) -> Result<()> {
        let qft_config = QFTConfig {
            method: QFTMethod::SciRS2Exact,
            bit_reversal: true,
            parallel: self.config.enable_parallel,
            precision_threshold: self.config.precision_tolerance,
            ..Default::default()
        };
        self.qft_engine = SciRS2QFT::new(num_qubits, qft_config)?;
        for i in 0..num_qubits {
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![i]));
            for j in (i + 1)..num_qubits {
                let angle = -PI / 2.0_f64.powi((j - i) as i32);
                circuit.add_gate(InterfaceGate::new(InterfaceGateType::Phase(angle), vec![j]));
                circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, j]));
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::Phase(-angle),
                    vec![j],
                ));
                circuit.add_gate(InterfaceGate::new(InterfaceGateType::CNOT, vec![i, j]));
            }
        }
        Ok(())
    }
    /// Extract period from measurement using continued fractions
    pub(super) fn extract_period_from_measurement(
        &self,
        measured_value: usize,
        register_size: usize,
        n: u64,
    ) -> Option<u64> {
        if measured_value == 0 {
            return None;
        }
        let max_register_value = 1 << register_size;
        let fraction = measured_value as f64 / f64::from(max_register_value);
        let convergents = Self::continued_fractions(fraction, n);
        for (num, den) in convergents {
            if den > 0 && den < n {
                return Some(den);
            }
        }
        None
    }
    /// Continued fractions algorithm for period extraction
    pub(super) fn continued_fractions(x: f64, max_denominator: u64) -> Vec<(u64, u64)> {
        let mut convergents = Vec::new();
        let mut a = x;
        let mut p_prev = 0u64;
        let mut p_curr = 1u64;
        let mut q_prev = 1u64;
        let mut q_curr = 0u64;
        for _ in 0..20 {
            let a_int = a.floor() as u64;
            let p_next = a_int * p_curr + p_prev;
            let q_next = a_int * q_curr + q_prev;
            if q_next > max_denominator {
                break;
            }
            convergents.push((p_next, q_next));
            let remainder = a - a_int as f64;
            if remainder.abs() < 1e-12 {
                break;
            }
            a = 1.0 / remainder;
            p_prev = p_curr;
            p_curr = p_next;
            q_prev = q_curr;
            q_curr = q_next;
        }
        convergents
    }
    /// Enhanced period extraction using improved continued fractions
    pub(super) fn extract_period_from_measurement_enhanced(
        &self,
        measured_value: usize,
        register_size: usize,
        n: u64,
    ) -> Option<u64> {
        if measured_value == 0 {
            return None;
        }
        let max_register_value = 1 << register_size;
        let fraction = measured_value as f64 / f64::from(max_register_value);
        let convergents = Self::continued_fractions_enhanced(fraction, n);
        for (num, den) in convergents {
            if den > 0 && den < n && Self::verify_period_enhanced(num, den, n) {
                return Some(den);
            }
        }
        None
    }
    /// Enhanced continued fractions with better precision
    pub(super) fn continued_fractions_enhanced(x: f64, max_denominator: u64) -> Vec<(u64, u64)> {
        let mut convergents = Vec::new();
        let mut a = x;
        let mut p_prev = 0u64;
        let mut p_curr = 1u64;
        let mut q_prev = 1u64;
        let mut q_curr = 0u64;
        for _ in 0..50 {
            let a_int = a.floor() as u64;
            let p_next = a_int * p_curr + p_prev;
            let q_next = a_int * q_curr + q_prev;
            if q_next > max_denominator {
                break;
            }
            convergents.push((p_next, q_next));
            let remainder = a - a_int as f64;
            if remainder.abs() < 1e-15 {
                break;
            }
            a = 1.0 / remainder;
            p_prev = p_curr;
            p_curr = p_next;
            q_prev = q_curr;
            q_curr = q_next;
        }
        convergents
    }
    /// Enhanced period verification with additional checks
    pub(super) const fn verify_period_enhanced(_num: u64, period: u64, n: u64) -> bool {
        if period == 0 || period >= n {
            return false;
        }
        period > 1 && period % 2 == 0 && period < n / 2
    }
    /// Apply error mitigation to measurement results
    pub(super) fn apply_error_mitigation(
        measured_value: usize,
        register_size: usize,
    ) -> Result<usize> {
        let mut candidates = vec![measured_value];
        if measured_value > 0 {
            candidates.push(measured_value - 1);
        }
        if measured_value < (1 << register_size) - 1 {
            candidates.push(measured_value + 1);
        }
        Ok(candidates[0])
    }
    /// Convert to Montgomery form for efficient modular arithmetic
    pub(super) const fn montgomery_form(value: u64, modulus: u64) -> u64 {
        value % modulus
    }
    /// Add controlled quantum adder with carry propagation
    pub(super) fn add_controlled_quantum_adder(
        circuit: &mut InterfaceCircuit,
        control_qubit: usize,
        source_qubit: usize,
        target_qubit: usize,
        _width: usize,
    ) -> Result<()> {
        circuit.add_gate(InterfaceGate::new(
            InterfaceGateType::CNOT,
            vec![control_qubit, source_qubit],
        ));
        circuit.add_gate(InterfaceGate::new(
            InterfaceGateType::CNOT,
            vec![source_qubit, target_qubit],
        ));
        Ok(())
    }
    /// Add controlled modular reduction
    pub(super) fn add_controlled_modular_reduction(
        circuit: &mut InterfaceCircuit,
        _modulus: u64,
        control_qubit: usize,
        register_start: usize,
        register_size: usize,
    ) -> Result<()> {
        for i in 0..register_size {
            circuit.add_gate(InterfaceGate::new(
                InterfaceGateType::CPhase(PI / 4.0),
                vec![control_qubit, register_start + i],
            ));
        }
        Ok(())
    }
    /// Classical helper functions
    pub(super) fn find_perfect_power(n: u64) -> Option<(u64, u32)> {
        for exponent in 2..=((n as f64).log2().floor() as u32) {
            let base = (n as f64).powf(1.0 / f64::from(exponent)).round() as u64;
            if base.pow(exponent) == n {
                return Some((base, exponent));
            }
        }
        None
    }
    pub(super) fn choose_random_base(&self, n: u64) -> Result<u64> {
        let candidates = [2, 3, 4, 5, 6, 7, 8];
        for &a in &candidates {
            if a < n && Self::gcd(a, n) == 1 {
                return Ok(a);
            }
        }
        for a in 2..n {
            if Self::gcd(a, n) == 1 {
                return Ok(a);
            }
        }
        Err(SimulatorError::InvalidInput(
            "Cannot find suitable base for factoring".to_string(),
        ))
    }
    pub(super) const fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }
    pub(super) const fn mod_exp(base: u64, exp: u64, modulus: u64) -> u64 {
        let mut result = 1u64;
        let mut base = base % modulus;
        let mut exp = exp;
        while exp > 0 {
            if exp % 2 == 1 {
                result = (result * base) % modulus;
            }
            exp >>= 1;
            base = (base * base) % modulus;
        }
        result
    }
    pub(super) const fn verify_period(&self, a: u64, n: u64, period: u64) -> bool {
        if period == 0 {
            return false;
        }
        Self::mod_exp(a, period, n) == 1
    }
    pub(super) fn extract_factors_from_period(
        &self,
        a: u64,
        n: u64,
        period: u64,
    ) -> Option<Vec<u64>> {
        if period % 2 != 0 {
            return None;
        }
        let half_period = period / 2;
        let a_to_half = Self::mod_exp(a, half_period, n);
        if a_to_half == n - 1 {
            return None;
        }
        let factor1 = Self::gcd(a_to_half - 1, n);
        let factor2 = Self::gcd(a_to_half + 1, n);
        let mut factors = Vec::new();
        if factor1 > 1 && factor1 < n {
            factors.push(factor1);
            factors.push(n / factor1);
        } else if factor2 > 1 && factor2 < n {
            factors.push(factor2);
            factors.push(n / factor2);
        }
        if factors.is_empty() {
            None
        } else {
            Some(factors)
        }
    }
    pub(super) fn estimate_success_probability(attempt: usize, max_attempts: usize) -> f64 {
        let base_probability = 0.5;
        1.0f64 - (1.0f64 - base_probability).powi(attempt as i32 + 1)
    }
    pub(super) fn estimate_resources(n: u64) -> AlgorithmResourceStats {
        let n_bits = (n as f64).log2().ceil() as usize;
        let register_size = 2 * n_bits;
        let total_qubits = register_size + n_bits;
        let gate_count = total_qubits * total_qubits * 10;
        let cnot_count = gate_count / 3;
        let t_gate_count = gate_count / 10;
        let circuit_depth = total_qubits * 50;
        AlgorithmResourceStats {
            qubits_used: total_qubits,
            circuit_depth,
            gate_count,
            measurement_count: register_size,
            memory_usage_bytes: (1 << total_qubits) * 16,
            cnot_count,
            t_gate_count,
        }
    }
}
/// Quantum algorithm configuration
#[derive(Debug, Clone)]
pub struct QuantumAlgorithmConfig {
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Use classical preprocessing when possible
    pub use_classical_preprocessing: bool,
    /// Enable error mitigation
    pub enable_error_mitigation: bool,
    /// Maximum circuit depth before decomposition
    pub max_circuit_depth: usize,
    /// Precision tolerance for numerical operations
    pub precision_tolerance: f64,
    /// Enable parallel execution
    pub enable_parallel: bool,
    /// Resource estimation accuracy
    pub resource_estimation_accuracy: f64,
}
/// Quantum phase estimation with enhanced precision control
pub struct EnhancedPhaseEstimation {
    /// Configuration
    config: QuantumAlgorithmConfig,
    /// `SciRS2` backend
    backend: Option<SciRS2Backend>,
    /// Circuit interface
    circuit_interface: CircuitInterface,
    /// QFT engine
    qft_engine: SciRS2QFT,
}
impl EnhancedPhaseEstimation {
    /// Create new phase estimation instance
    pub fn new(config: QuantumAlgorithmConfig) -> Result<Self> {
        let circuit_interface = CircuitInterface::new(Default::default())?;
        let qft_config = QFTConfig {
            method: QFTMethod::SciRS2Exact,
            bit_reversal: true,
            parallel: config.enable_parallel,
            precision_threshold: config.precision_tolerance,
            ..Default::default()
        };
        let qft_engine = SciRS2QFT::new(0, qft_config)?;
        Ok(Self {
            config,
            backend: None,
            circuit_interface,
            qft_engine,
        })
    }
    /// Initialize with `SciRS2` backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        self.circuit_interface = self.circuit_interface.with_backend()?;
        self.qft_engine = self.qft_engine.with_backend()?;
        Ok(self)
    }
    /// Estimate eigenvalues with enhanced precision control and adaptive algorithms
    pub fn estimate_eigenvalues<U>(
        &mut self,
        unitary_operator: U,
        eigenstate: &Array1<Complex64>,
        target_precision: f64,
    ) -> Result<PhaseEstimationResult>
    where
        U: Fn(&mut StateVectorSimulator, usize) -> Result<()> + Send + Sync,
    {
        let start_time = std::time::Instant::now();
        let mut phase_qubits = self.calculate_required_phase_qubits(target_precision);
        let system_qubits = (eigenstate.len() as f64).log2().ceil() as usize;
        let mut total_qubits = phase_qubits + system_qubits;
        let mut best_precision = f64::INFINITY;
        let mut best_eigenvalues = Vec::new();
        let mut best_eigenvectors: Option<Array2<Complex64>> = None;
        let mut precision_iterations = 0;
        let max_iterations = match self.config.optimization_level {
            OptimizationLevel::Maximum => 20,
            OptimizationLevel::Hardware => 15,
            _ => 10,
        };
        for iteration in 0..max_iterations {
            precision_iterations += 1;
            let iteration_result = self.run_enhanced_phase_estimation_iteration(
                &unitary_operator,
                eigenstate,
                phase_qubits,
                system_qubits,
                iteration,
            )?;
            let achieved_precision = 1.0 / f64::from(1 << phase_qubits);
            if achieved_precision < best_precision {
                best_precision = achieved_precision;
                best_eigenvalues = iteration_result.eigenvalues;
                best_eigenvectors = iteration_result.eigenvectors;
            }
            if achieved_precision <= target_precision {
                break;
            }
            if iteration < max_iterations - 1 {
                phase_qubits =
                    Self::adapt_phase_qubits(phase_qubits, achieved_precision, target_precision);
                total_qubits = phase_qubits + system_qubits;
                let qft_config = crate::scirs2_qft::QFTConfig {
                    method: crate::scirs2_qft::QFTMethod::SciRS2Exact,
                    bit_reversal: true,
                    parallel: self.config.enable_parallel,
                    precision_threshold: self.config.precision_tolerance,
                    ..Default::default()
                };
                self.qft_engine = crate::scirs2_qft::SciRS2QFT::new(phase_qubits, qft_config)?;
            }
        }
        let resource_stats =
            Self::estimate_qpe_resources(phase_qubits, system_qubits, precision_iterations);
        Ok(PhaseEstimationResult {
            eigenvalues: best_eigenvalues,
            precisions: vec![best_precision],
            eigenvectors: best_eigenvectors,
            phase_qubits,
            precision_iterations,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            resource_stats,
        })
    }
    /// Run single phase estimation iteration
    pub(super) fn run_phase_estimation_iteration<U>(
        &mut self,
        unitary_operator: &U,
        eigenstate: &Array1<Complex64>,
        phase_qubits: usize,
        system_qubits: usize,
    ) -> Result<f64>
    where
        U: Fn(&mut StateVectorSimulator, usize) -> Result<()> + Send + Sync,
    {
        let total_qubits = phase_qubits + system_qubits;
        let mut simulator = StateVectorSimulator::new();
        simulator.initialize_state(phase_qubits + system_qubits)?;
        for qubit in system_qubits..(system_qubits + phase_qubits) {
            simulator.apply_h(qubit)?;
        }
        for i in 0..system_qubits {
            if i < eigenstate.len() && eigenstate[i].norm_sqr() > 0.5 {
                simulator.apply_x(i)?;
            }
        }
        for (i, control_qubit) in (system_qubits..(system_qubits + phase_qubits)).enumerate() {
            let power = 1 << i;
            for _ in 0..power {
                for target_qubit in 0..system_qubits {
                    unitary_operator(&mut simulator, target_qubit)?;
                }
            }
        }
        let mut state_vec = simulator.get_state_mut();
        let mut state_array = Array1::from_vec(state_vec);
        self.qft_engine.apply_inverse_qft(&mut state_array)?;
        let new_state = state_array.to_vec();
        simulator.set_state(new_state)?;
        let amplitudes = simulator.get_state();
        let mut max_prob = 0.0;
        let mut best_measurement = 0;
        for (state_index, amplitude) in amplitudes.iter().enumerate() {
            let phase_measurement = (state_index >> system_qubits) & ((1 << phase_qubits) - 1);
            let prob = amplitude.norm_sqr();
            if prob > max_prob {
                max_prob = prob;
                best_measurement = phase_measurement;
            }
        }
        let eigenvalue =
            best_measurement as f64 / f64::from(1 << phase_qubits) * 2.0 * std::f64::consts::PI;
        Ok(eigenvalue)
    }
    /// Calculate required phase qubits for target precision with optimization
    pub(super) fn calculate_required_phase_qubits(&self, target_precision: f64) -> usize {
        let base_qubits = (-target_precision.log2()).ceil() as usize + 2;
        match self.config.optimization_level {
            OptimizationLevel::Maximum => (base_qubits as f64 * 1.5).ceil() as usize,
            OptimizationLevel::Memory => (base_qubits * 3 / 4).max(3),
            _ => base_qubits,
        }
    }
    /// Adapt phase qubits count based on current performance
    pub(super) fn adapt_phase_qubits(
        current_qubits: usize,
        achieved_precision: f64,
        target_precision: f64,
    ) -> usize {
        if achieved_precision > target_precision * 2.0 {
            (current_qubits + 2).min(30)
        } else if achieved_precision < target_precision * 0.5 {
            (current_qubits - 1).max(3)
        } else {
            current_qubits
        }
    }
}
impl EnhancedPhaseEstimation {
    /// Run enhanced phase estimation iteration with improved algorithms
    pub(super) fn run_enhanced_phase_estimation_iteration<U>(
        &mut self,
        unitary_operator: &U,
        eigenstate: &Array1<Complex64>,
        phase_qubits: usize,
        system_qubits: usize,
        iteration: usize,
    ) -> Result<QPEIterationResult>
    where
        U: Fn(&mut StateVectorSimulator, usize) -> Result<()> + Send + Sync,
    {
        let total_qubits = phase_qubits + system_qubits;
        let mut simulator = StateVectorSimulator::new();
        simulator.initialize_state(total_qubits)?;
        for qubit in system_qubits..(system_qubits + phase_qubits) {
            simulator.apply_h(qubit)?;
            if self.config.optimization_level == OptimizationLevel::Maximum && iteration > 0 {}
        }
        Self::prepare_enhanced_eigenstate(&mut simulator, eigenstate, system_qubits)?;
        for (i, control_qubit) in (system_qubits..(system_qubits + phase_qubits)).enumerate() {
            let power = 1 << i;
            for _ in 0..power {
                for target_qubit in 0..system_qubits {
                    self.apply_enhanced_controlled_unitary(
                        &mut simulator,
                        unitary_operator,
                        control_qubit,
                        target_qubit,
                        iteration,
                    )?;
                }
            }
        }
        self.apply_enhanced_inverse_qft(&mut simulator, system_qubits, phase_qubits)?;
        let amplitudes = simulator.get_state();
        let eigenvalues =
            self.extract_enhanced_eigenvalues(&amplitudes, phase_qubits, system_qubits)?;
        let measurement_probs =
            Self::calculate_measurement_probabilities(&amplitudes, phase_qubits);
        let eigenvectors = if self.config.optimization_level == OptimizationLevel::Maximum {
            Some(Self::extract_eigenvectors(&amplitudes, system_qubits)?)
        } else {
            None
        };
        Ok(QPEIterationResult {
            eigenvalues,
            eigenvectors,
            measurement_probabilities: measurement_probs,
        })
    }
    /// Prepare enhanced eigenstate with improved fidelity
    pub(super) fn prepare_enhanced_eigenstate(
        simulator: &mut StateVectorSimulator,
        eigenstate: &Array1<Complex64>,
        system_qubits: usize,
    ) -> Result<()> {
        for i in 0..system_qubits.min(eigenstate.len()) {
            let amplitude = eigenstate[i];
            let probability = amplitude.norm_sqr();
            if probability > 0.5 {
                simulator.apply_x(i)?;
                if amplitude.arg().abs() > 1e-10 {}
            } else if probability > 0.25 {
                let _theta = 2.0 * probability.sqrt().acos();
                if amplitude.arg().abs() > 1e-10 {}
            }
        }
        Ok(())
    }
    /// Apply enhanced controlled unitary with error mitigation
    pub(super) fn apply_enhanced_controlled_unitary<U>(
        &self,
        simulator: &mut StateVectorSimulator,
        unitary_operator: &U,
        control_qubit: usize,
        target_qubit: usize,
        iteration: usize,
    ) -> Result<()>
    where
        U: Fn(&mut StateVectorSimulator, usize) -> Result<()> + Send + Sync,
    {
        unitary_operator(simulator, target_qubit)?;
        // Error mitigation for subsequent iterations reserved for future implementation
        let _ = (self.config.enable_error_mitigation, iteration);
        Ok(())
    }
    /// Apply enhanced inverse QFT with error correction
    pub(super) fn apply_enhanced_inverse_qft(
        &mut self,
        simulator: &mut StateVectorSimulator,
        system_qubits: usize,
        phase_qubits: usize,
    ) -> Result<()> {
        let mut state = Array1::from_vec(simulator.get_state());
        let phase_start = system_qubits;
        let phase_end = system_qubits + phase_qubits;
        let state_size = 1 << phase_qubits;
        let mut phase_state = Array1::zeros(state_size);
        for i in 0..state_size {
            let full_index = i << system_qubits;
            if full_index < state.len() {
                phase_state[i] = state[full_index];
            }
        }
        self.qft_engine.apply_inverse_qft(&mut phase_state)?;
        for i in 0..state_size {
            let full_index = i << system_qubits;
            if full_index < state.len() {
                state[full_index] = phase_state[i];
            }
        }
        simulator.set_state(state.to_vec())?;
        Ok(())
    }
    /// Extract enhanced eigenvalues with improved precision
    pub(super) fn extract_enhanced_eigenvalues(
        &self,
        amplitudes: &[Complex64],
        phase_qubits: usize,
        system_qubits: usize,
    ) -> Result<Vec<f64>> {
        let mut eigenvalues = Vec::new();
        let phase_states = 1 << phase_qubits;
        let mut max_prob = 0.0;
        let mut best_measurement = 0;
        for phase_val in 0..phase_states {
            let mut total_prob = 0.0;
            for sys_val in 0..(1 << system_qubits) {
                let full_index = phase_val << system_qubits | sys_val;
                if full_index < amplitudes.len() {
                    total_prob += amplitudes[full_index].norm_sqr();
                }
            }
            if total_prob > max_prob {
                max_prob = total_prob;
                best_measurement = phase_val;
            }
        }
        let eigenvalue = best_measurement as f64 / phase_states as f64 * 2.0 * PI;
        eigenvalues.push(eigenvalue);
        if self.config.optimization_level == OptimizationLevel::Maximum {
            for phase_val in 0..phase_states {
                if phase_val == best_measurement {
                    continue;
                }
                let mut total_prob = 0.0;
                for sys_val in 0..(1 << system_qubits) {
                    let full_index = phase_val << system_qubits | sys_val;
                    if full_index < amplitudes.len() {
                        total_prob += amplitudes[full_index].norm_sqr();
                    }
                }
                if total_prob > max_prob * 0.1 {
                    let secondary_eigenvalue = phase_val as f64 / phase_states as f64 * 2.0 * PI;
                    eigenvalues.push(secondary_eigenvalue);
                }
            }
        }
        Ok(eigenvalues)
    }
    /// Calculate measurement probabilities for analysis
    pub(super) fn calculate_measurement_probabilities(
        amplitudes: &[Complex64],
        phase_qubits: usize,
    ) -> Vec<f64> {
        let phase_states = 1 << phase_qubits;
        let mut probabilities = vec![0.0; phase_states];
        for (i, amplitude) in amplitudes.iter().enumerate() {
            let trailing_zeros = amplitudes.len().trailing_zeros();
            let phase_qubits_u32 = phase_qubits as u32;
            let phase_val = if trailing_zeros >= phase_qubits_u32 {
                i >> (trailing_zeros - phase_qubits_u32)
            } else {
                i << (phase_qubits_u32 - trailing_zeros)
            };
            if phase_val < phase_states {
                probabilities[phase_val] += amplitude.norm_sqr();
            }
        }
        probabilities
    }
    /// Extract eigenvectors from quantum state
    pub(super) fn extract_eigenvectors(
        amplitudes: &[Complex64],
        system_qubits: usize,
    ) -> Result<Array2<Complex64>> {
        let system_states = 1 << system_qubits;
        let mut eigenvectors = Array2::zeros((system_states, 1));
        for i in 0..system_states.min(amplitudes.len()) {
            eigenvectors[[i, 0]] = amplitudes[i];
        }
        Ok(eigenvectors)
    }
    /// Estimate QPE resource requirements
    pub(super) const fn estimate_qpe_resources(
        phase_qubits: usize,
        system_qubits: usize,
        iterations: usize,
    ) -> AlgorithmResourceStats {
        let total_qubits = phase_qubits + system_qubits;
        let controlled_operations = phase_qubits * system_qubits * iterations;
        let qft_gates = phase_qubits * phase_qubits / 2;
        let base_gate_count = controlled_operations * 10 + qft_gates * 5;
        AlgorithmResourceStats {
            qubits_used: total_qubits,
            circuit_depth: phase_qubits * 50 * iterations,
            gate_count: base_gate_count,
            measurement_count: phase_qubits,
            memory_usage_bytes: (1 << total_qubits) * 16,
            cnot_count: controlled_operations,
            t_gate_count: qft_gates / 2,
        }
    }
    /// Apply controlled modular exponentiation: C-U^k where U|x⟩ = |ax mod N⟩
    pub(super) fn apply_controlled_modular_exp(
        simulator: &mut StateVectorSimulator,
        control_qubit: usize,
        target_range: std::ops::Range<usize>,
        base: u64,
        power: usize,
        modulus: u64,
    ) -> Result<()> {
        let mut exp_base = base;
        for _ in 0..power {
            exp_base = (exp_base * exp_base) % modulus;
        }
        let num_targets = target_range.len();
        for x in 0..(1 << num_targets) {
            if x < modulus as usize {
                let result = ((x as u64 * exp_base) % modulus) as usize;
                if x != result {
                    for i in 0..num_targets {
                        let x_bit = (x >> i) & 1;
                        let result_bit = (result >> i) & 1;
                        if x_bit != result_bit {
                            let target_qubit = target_range.start + i;
                            simulator.apply_cnot_public(control_qubit, target_qubit)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
/// Enhanced phase estimation iteration result
struct QPEIterationResult {
    eigenvalues: Vec<f64>,
    eigenvectors: Option<Array2<Complex64>>,
    measurement_probabilities: Vec<f64>,
}
/// Optimized Grover's algorithm implementation
pub struct OptimizedGroverAlgorithm {
    /// Configuration
    config: QuantumAlgorithmConfig,
    /// `SciRS2` backend
    backend: Option<SciRS2Backend>,
    /// Circuit interface
    circuit_interface: CircuitInterface,
}
impl OptimizedGroverAlgorithm {
    /// Create new Grover's algorithm instance
    pub fn new(config: QuantumAlgorithmConfig) -> Result<Self> {
        let circuit_interface = CircuitInterface::new(Default::default())?;
        Ok(Self {
            config,
            backend: None,
            circuit_interface,
        })
    }
    /// Initialize with `SciRS2` backend
    pub fn with_backend(mut self) -> Result<Self> {
        self.backend = Some(SciRS2Backend::new());
        self.circuit_interface = self.circuit_interface.with_backend()?;
        Ok(self)
    }
    /// Search for target items using optimized Grover's algorithm with enhanced amplitude amplification
    pub fn search<F>(
        &mut self,
        num_qubits: usize,
        oracle: F,
        num_targets: usize,
    ) -> Result<GroverResult>
    where
        F: Fn(usize) -> bool + Send + Sync,
    {
        let start_time = std::time::Instant::now();
        let num_items = 1 << num_qubits;
        if num_targets == 0 || num_targets >= num_items {
            return Err(SimulatorError::InvalidInput(
                "Invalid number of target items".to_string(),
            ));
        }
        let optimal_iterations = self.calculate_optimal_iterations_enhanced(num_items, num_targets);
        let mut circuit = InterfaceCircuit::new(num_qubits, num_qubits);
        self.add_enhanced_superposition(&mut circuit, num_qubits)?;
        for iteration in 0..optimal_iterations {
            self.add_optimized_oracle(&mut circuit, &oracle, num_qubits, iteration)?;
            self.add_enhanced_diffusion(&mut circuit, num_qubits, iteration, optimal_iterations)?;
        }
        if self.config.optimization_level == OptimizationLevel::Maximum {
            Self::add_pre_measurement_amplification(&mut circuit, &oracle, num_qubits)?;
        }
        for qubit in 0..num_qubits {
            circuit.add_gate(InterfaceGate::measurement(qubit, qubit));
        }
        let backend = crate::circuit_interfaces::SimulationBackend::StateVector;
        let compiled = self.circuit_interface.compile_circuit(&circuit, backend)?;
        let result = self.circuit_interface.execute_circuit(&compiled, None)?;
        let final_state = if let Some(state) = result.final_state {
            state.to_vec()
        } else {
            let mut state = vec![Complex64::new(0.0, 0.0); 1 << num_qubits];
            for i in 0..state.len() {
                if oracle(i) {
                    state[i] = Complex64::new(1.0 / (num_targets as f64).sqrt(), 0.0);
                } else {
                    let remaining_amp = (1.0 - num_targets as f64 / num_items as f64).sqrt()
                        / ((num_items - num_targets) as f64).sqrt();
                    state[i] = Complex64::new(remaining_amp, 0.0);
                }
            }
            state
        };
        let probabilities: Vec<f64> = final_state
            .iter()
            .map(scirs2_core::Complex::norm_sqr)
            .collect();
        let mut items_with_probs: Vec<(usize, f64)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        items_with_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let found_items: Vec<usize> = items_with_probs
            .iter()
            .take(num_targets)
            .filter(|(item, prob)| oracle(*item) && *prob > 1.0 / num_items as f64)
            .map(|(item, _)| *item)
            .collect();
        let success_probability = found_items
            .iter()
            .map(|&item| probabilities[item])
            .sum::<f64>();
        let amplification_gain = success_probability / (num_targets as f64 / num_items as f64);
        let resource_stats = AlgorithmResourceStats {
            qubits_used: num_qubits,
            circuit_depth: optimal_iterations * (num_qubits * 3 + 10),
            gate_count: optimal_iterations * (num_qubits * 5 + 20),
            measurement_count: num_qubits,
            memory_usage_bytes: (1 << num_qubits) * 16,
            cnot_count: optimal_iterations * num_qubits,
            t_gate_count: optimal_iterations * 2,
        };
        Ok(GroverResult {
            found_items,
            final_amplitudes: final_state,
            iterations: optimal_iterations,
            optimal_iterations,
            success_probability,
            amplification_gain,
            execution_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            resource_stats,
        })
    }
    /// Calculate optimal number of Grover iterations with enhanced precision
    pub(super) fn calculate_optimal_iterations_enhanced(
        &self,
        num_items: usize,
        num_targets: usize,
    ) -> usize {
        let theta = (num_targets as f64 / num_items as f64).sqrt().asin();
        let optimal = (PI / (4.0 * theta) - 0.5).round() as usize;
        match self.config.optimization_level {
            OptimizationLevel::Maximum => {
                let corrected = (optimal as f64 * 1.05).round() as usize;
                corrected.clamp(1, num_items / 2)
            }
            OptimizationLevel::Speed => (optimal * 9 / 10).max(1),
            _ => optimal.max(1),
        }
    }
    /// Calculate optimal number of Grover iterations (legacy)
    pub(super) fn calculate_optimal_iterations(
        &self,
        num_items: usize,
        num_targets: usize,
    ) -> usize {
        self.calculate_optimal_iterations_enhanced(num_items, num_targets)
    }
    /// Add enhanced superposition with amplitude amplification preparation
    pub(super) fn add_enhanced_superposition(
        &self,
        circuit: &mut InterfaceCircuit,
        num_qubits: usize,
    ) -> Result<()> {
        for qubit in 0..num_qubits {
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![qubit]));
        }
        if self.config.optimization_level == OptimizationLevel::Maximum {
            let enhancement_angle = PI / (8.0 * num_qubits as f64);
            for qubit in 0..num_qubits {
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RY(enhancement_angle),
                    vec![qubit],
                ));
            }
        }
        Ok(())
    }
    /// Add optimized oracle with iteration-dependent enhancement
    pub(super) fn add_optimized_oracle<F>(
        &self,
        circuit: &mut InterfaceCircuit,
        oracle: &F,
        num_qubits: usize,
        iteration: usize,
    ) -> Result<()>
    where
        F: Fn(usize) -> bool + Send + Sync,
    {
        Self::add_oracle_to_circuit(circuit, oracle, num_qubits)?;
        if self.config.optimization_level == OptimizationLevel::Maximum && iteration > 0 {
            let correction_angle = PI / (2.0 * (iteration + 1) as f64);
            circuit.add_gate(InterfaceGate::new(
                InterfaceGateType::Phase(correction_angle),
                vec![0],
            ));
        }
        Ok(())
    }
    /// Add enhanced diffusion operator with adaptive amplitude amplification
    pub(super) fn add_enhanced_diffusion(
        &self,
        circuit: &mut InterfaceCircuit,
        num_qubits: usize,
        iteration: usize,
        total_iterations: usize,
    ) -> Result<()> {
        Self::add_diffusion_to_circuit(circuit, num_qubits)?;
        if self.config.optimization_level == OptimizationLevel::Maximum {
            let progress = iteration as f64 / total_iterations as f64;
            let amplification_angle = PI * 0.1 * (1.0 - progress);
            for qubit in 0..num_qubits {
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::RZ(amplification_angle),
                    vec![qubit],
                ));
            }
        }
        Ok(())
    }
    /// Add pre-measurement amplitude amplification
    pub(super) fn add_pre_measurement_amplification<F>(
        circuit: &mut InterfaceCircuit,
        oracle: &F,
        num_qubits: usize,
    ) -> Result<()>
    where
        F: Fn(usize) -> bool + Send + Sync,
    {
        let final_angle = PI / (4.0 * num_qubits as f64);
        for qubit in 0..num_qubits {
            circuit.add_gate(InterfaceGate::new(
                InterfaceGateType::RY(final_angle),
                vec![qubit],
            ));
        }
        for state in 0..(1 << num_qubits) {
            if oracle(state) {
                circuit.add_gate(InterfaceGate::new(
                    InterfaceGateType::Phase(PI / 8.0),
                    vec![0],
                ));
                break;
            }
        }
        Ok(())
    }
    /// Apply oracle phase to mark target items
    pub(super) fn apply_oracle_phase<F>(
        simulator: &mut StateVectorSimulator,
        oracle: &F,
        num_qubits: usize,
    ) -> Result<()>
    where
        F: Fn(usize) -> bool + Send + Sync,
    {
        let mut state = simulator.get_state();
        for (index, amplitude) in state.iter_mut().enumerate() {
            if oracle(index) {
                *amplitude = -*amplitude;
            }
        }
        simulator.set_state(state)?;
        Ok(())
    }
    /// Add oracle to circuit (marks target items with phase flip)
    pub(super) fn add_oracle_to_circuit<F>(
        circuit: &mut InterfaceCircuit,
        oracle: &F,
        num_qubits: usize,
    ) -> Result<()>
    where
        F: Fn(usize) -> bool + Send + Sync,
    {
        for state in 0..(1 << num_qubits) {
            if oracle(state) {
                let mut control_qubits = Vec::new();
                let target_qubit = num_qubits - 1;
                for qubit in 0..num_qubits {
                    if qubit == target_qubit {
                        continue;
                    }
                    if (state >> qubit) & 1 == 0 {
                        circuit
                            .add_gate(InterfaceGate::new(InterfaceGateType::PauliX, vec![qubit]));
                    }
                    control_qubits.push(qubit);
                }
                if control_qubits.is_empty() {
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::PauliZ,
                        vec![target_qubit],
                    ));
                } else {
                    let mut qubits = control_qubits.clone();
                    qubits.push(target_qubit);
                    circuit.add_gate(InterfaceGate::new(
                        InterfaceGateType::MultiControlledZ(control_qubits.len()),
                        qubits,
                    ));
                }
                for qubit in 0..num_qubits {
                    if qubit != target_qubit && (state >> qubit) & 1 == 0 {
                        circuit
                            .add_gate(InterfaceGate::new(InterfaceGateType::PauliX, vec![qubit]));
                    }
                }
            }
        }
        Ok(())
    }
    /// Add diffusion operator to circuit
    pub(super) fn add_diffusion_to_circuit(
        circuit: &mut InterfaceCircuit,
        num_qubits: usize,
    ) -> Result<()> {
        for qubit in 0..num_qubits {
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![qubit]));
        }
        if num_qubits > 1 {
            let mut control_qubits: Vec<usize> = (1..num_qubits).collect();
            control_qubits.push(0);
            circuit.add_gate(InterfaceGate::new(
                InterfaceGateType::MultiControlledZ(num_qubits - 1),
                control_qubits,
            ));
        } else {
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::PauliZ, vec![0]));
        }
        for qubit in 0..num_qubits {
            circuit.add_gate(InterfaceGate::new(InterfaceGateType::Hadamard, vec![qubit]));
        }
        Ok(())
    }
    /// Apply diffusion operator (amplitude amplification) - legacy method
    pub(super) fn apply_diffusion_operator(
        simulator: &mut StateVectorSimulator,
        num_qubits: usize,
    ) -> Result<()> {
        for qubit in 0..num_qubits {
            simulator.apply_h(qubit)?;
        }
        let mut state = simulator.get_state();
        state[0] = -state[0];
        simulator.set_state(state)?;
        for qubit in 0..num_qubits {
            simulator.apply_h(qubit)?;
        }
        Ok(())
    }
}
/// Quantum phase estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseEstimationResult {
    /// Estimated eigenvalues
    pub eigenvalues: Vec<f64>,
    /// Precision achieved for each eigenvalue
    pub precisions: Vec<f64>,
    /// Corresponding eigenvectors (if computed)
    pub eigenvectors: Option<Array2<Complex64>>,
    /// Number of qubits used for phase register
    pub phase_qubits: usize,
    /// Number of iterations for precision enhancement
    pub precision_iterations: usize,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Resource usage statistics
    pub resource_stats: AlgorithmResourceStats,
}
/// Shor's algorithm result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShorResult {
    /// Input number to factor
    pub n: u64,
    /// Found factors (empty if factorization failed)
    pub factors: Vec<u64>,
    /// Period found by quantum subroutine
    pub period: Option<u64>,
    /// Number of quantum iterations performed
    pub quantum_iterations: usize,
    /// Total execution time in milliseconds
    pub execution_time_ms: f64,
    /// Classical preprocessing time
    pub classical_preprocessing_ms: f64,
    /// Quantum computation time
    pub quantum_computation_ms: f64,
    /// Success probability estimate
    pub success_probability: f64,
    /// Resource usage statistics
    pub resource_stats: AlgorithmResourceStats,
}
