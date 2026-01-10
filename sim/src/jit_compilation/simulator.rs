//! JIT-enabled quantum simulator
//!
//! This module provides a quantum simulator with JIT compilation support.

use scirs2_core::ndarray::Array1;
use scirs2_core::Complex64;
use std::time::{Duration, Instant};

use crate::circuit_interfaces::{InterfaceGate, InterfaceGateType};
use crate::error::{Result, SimulatorError};

use super::compiler::JITCompiler;
use super::profiler::{JITCompilerStats, JITSimulatorStats};
use super::types::{JITBenchmarkResults, JITConfig};

/// JIT-enabled quantum simulator
pub struct JITQuantumSimulator {
    /// State vector
    state: Array1<Complex64>,
    /// Number of qubits
    pub(crate) num_qubits: usize,
    /// JIT compiler
    pub(crate) compiler: JITCompiler,
    /// Execution statistics
    stats: JITSimulatorStats,
}

impl JITQuantumSimulator {
    /// Create new JIT-enabled simulator
    #[must_use]
    pub fn new(num_qubits: usize, config: JITConfig) -> Self {
        let state_size = 1 << num_qubits;
        let mut state = Array1::zeros(state_size);
        state[0] = Complex64::new(1.0, 0.0); // |0...0⟩ state

        Self {
            state,
            num_qubits,
            compiler: JITCompiler::new(config),
            stats: JITSimulatorStats::default(),
        }
    }

    /// Apply gate sequence with JIT optimization
    pub fn apply_gate_sequence(&mut self, gates: &[InterfaceGate]) -> Result<Duration> {
        let execution_start = Instant::now();

        // Analyze sequence for compilation opportunities
        if let Some(pattern_hash) = self.compiler.analyze_sequence(gates)? {
            // Check if compiled version exists
            if self.is_compiled(pattern_hash) {
                // Execute compiled version
                let exec_time = self
                    .compiler
                    .execute_compiled(pattern_hash, &mut self.state)?;
                self.stats.compiled_executions += 1;
                self.stats.total_compiled_time += exec_time;
                return Ok(exec_time);
            }
        }

        // Fall back to interpreted execution
        for gate in gates {
            self.apply_gate_interpreted(gate)?;
        }

        let execution_time = execution_start.elapsed();
        self.stats.interpreted_executions += 1;
        self.stats.total_interpreted_time += execution_time;

        Ok(execution_time)
    }

    /// Check if pattern is compiled
    fn is_compiled(&self, pattern_hash: u64) -> bool {
        let cache = self
            .compiler
            .compiled_cache
            .read()
            .expect("JIT cache lock should not be poisoned");
        cache.contains_key(&pattern_hash)
    }

    /// Apply single gate in interpreted mode
    pub fn apply_gate_interpreted(&mut self, gate: &InterfaceGate) -> Result<()> {
        match &gate.gate_type {
            InterfaceGateType::PauliX | InterfaceGateType::X => {
                if gate.qubits.len() != 1 {
                    return Err(SimulatorError::InvalidParameter(
                        "Pauli-X requires exactly one target".to_string(),
                    ));
                }
                self.apply_pauli_x(gate.qubits[0])
            }
            InterfaceGateType::PauliY => {
                if gate.qubits.len() != 1 {
                    return Err(SimulatorError::InvalidParameter(
                        "Pauli-Y requires exactly one target".to_string(),
                    ));
                }
                self.apply_pauli_y(gate.qubits[0])
            }
            InterfaceGateType::PauliZ => {
                if gate.qubits.len() != 1 {
                    return Err(SimulatorError::InvalidParameter(
                        "Pauli-Z requires exactly one target".to_string(),
                    ));
                }
                self.apply_pauli_z(gate.qubits[0])
            }
            InterfaceGateType::Hadamard | InterfaceGateType::H => {
                if gate.qubits.len() != 1 {
                    return Err(SimulatorError::InvalidParameter(
                        "Hadamard requires exactly one target".to_string(),
                    ));
                }
                self.apply_hadamard(gate.qubits[0])
            }
            InterfaceGateType::CNOT => {
                if gate.qubits.len() != 2 {
                    return Err(SimulatorError::InvalidParameter(
                        "CNOT requires exactly two targets".to_string(),
                    ));
                }
                self.apply_cnot(gate.qubits[0], gate.qubits[1])
            }
            InterfaceGateType::RX(angle) => {
                if gate.qubits.len() != 1 {
                    return Err(SimulatorError::InvalidParameter(
                        "RX requires one target".to_string(),
                    ));
                }
                self.apply_rx(gate.qubits[0], *angle)
            }
            InterfaceGateType::RY(angle) => {
                if gate.qubits.len() != 1 {
                    return Err(SimulatorError::InvalidParameter(
                        "RY requires one target".to_string(),
                    ));
                }
                self.apply_ry(gate.qubits[0], *angle)
            }
            InterfaceGateType::RZ(angle) => {
                if gate.qubits.len() != 1 {
                    return Err(SimulatorError::InvalidParameter(
                        "RZ requires one target".to_string(),
                    ));
                }
                self.apply_rz(gate.qubits[0], *angle)
            }
            _ => Err(SimulatorError::NotImplemented(format!(
                "Gate type {:?}",
                gate.gate_type
            ))),
        }
    }

    /// Apply Pauli-X gate
    fn apply_pauli_x(&mut self, target: usize) -> Result<()> {
        if target >= self.num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Target qubit out of range".to_string(),
            ));
        }

        for i in 0..(1 << self.num_qubits) {
            let j = i ^ (1 << target);
            if i < j {
                let temp = self.state[i];
                self.state[i] = self.state[j];
                self.state[j] = temp;
            }
        }

        Ok(())
    }

    /// Apply Pauli-Y gate
    fn apply_pauli_y(&mut self, target: usize) -> Result<()> {
        if target >= self.num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Target qubit out of range".to_string(),
            ));
        }

        for i in 0..(1 << self.num_qubits) {
            if (i >> target) & 1 == 0 {
                let j = i | (1 << target);
                let temp = self.state[i];
                self.state[i] = Complex64::new(0.0, 1.0) * self.state[j];
                self.state[j] = Complex64::new(0.0, -1.0) * temp;
            }
        }

        Ok(())
    }

    /// Apply Pauli-Z gate
    fn apply_pauli_z(&mut self, target: usize) -> Result<()> {
        if target >= self.num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Target qubit out of range".to_string(),
            ));
        }

        for i in 0..(1 << self.num_qubits) {
            if (i >> target) & 1 == 1 {
                self.state[i] = -self.state[i];
            }
        }

        Ok(())
    }

    /// Apply Hadamard gate
    fn apply_hadamard(&mut self, target: usize) -> Result<()> {
        if target >= self.num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Target qubit out of range".to_string(),
            ));
        }

        let sqrt2_inv = 1.0 / (2.0_f64).sqrt();

        for i in 0..(1 << self.num_qubits) {
            if (i >> target) & 1 == 0 {
                let j = i | (1 << target);
                let amp0 = self.state[i];
                let amp1 = self.state[j];

                self.state[i] = sqrt2_inv * (amp0 + amp1);
                self.state[j] = sqrt2_inv * (amp0 - amp1);
            }
        }

        Ok(())
    }

    /// Apply CNOT gate
    fn apply_cnot(&mut self, control: usize, target: usize) -> Result<()> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Qubit index out of range".to_string(),
            ));
        }

        for i in 0..(1 << self.num_qubits) {
            if (i >> control) & 1 == 1 {
                let j = i ^ (1 << target);
                if i < j {
                    let temp = self.state[i];
                    self.state[i] = self.state[j];
                    self.state[j] = temp;
                }
            }
        }

        Ok(())
    }

    /// Apply RX gate
    fn apply_rx(&mut self, target: usize, angle: f64) -> Result<()> {
        if target >= self.num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Target qubit out of range".to_string(),
            ));
        }

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..(1 << self.num_qubits) {
            if (i >> target) & 1 == 0 {
                let j = i | (1 << target);
                let amp0 = self.state[i];
                let amp1 = self.state[j];

                self.state[i] = cos_half * amp0 - Complex64::new(0.0, sin_half) * amp1;
                self.state[j] = -Complex64::new(0.0, sin_half) * amp0 + cos_half * amp1;
            }
        }

        Ok(())
    }

    /// Apply RY gate
    fn apply_ry(&mut self, target: usize, angle: f64) -> Result<()> {
        if target >= self.num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Target qubit out of range".to_string(),
            ));
        }

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();

        for i in 0..(1 << self.num_qubits) {
            if (i >> target) & 1 == 0 {
                let j = i | (1 << target);
                let amp0 = self.state[i];
                let amp1 = self.state[j];

                self.state[i] = cos_half * amp0 - sin_half * amp1;
                self.state[j] = sin_half * amp0 + cos_half * amp1;
            }
        }

        Ok(())
    }

    /// Apply RZ gate
    fn apply_rz(&mut self, target: usize, angle: f64) -> Result<()> {
        if target >= self.num_qubits {
            return Err(SimulatorError::InvalidParameter(
                "Target qubit out of range".to_string(),
            ));
        }

        let exp_neg = Complex64::new(0.0, -angle / 2.0).exp();
        let exp_pos = Complex64::new(0.0, angle / 2.0).exp();

        for i in 0..(1 << self.num_qubits) {
            if (i >> target) & 1 == 0 {
                self.state[i] *= exp_neg;
            } else {
                self.state[i] *= exp_pos;
            }
        }

        Ok(())
    }

    /// Get current state vector
    #[must_use]
    pub const fn get_state(&self) -> &Array1<Complex64> {
        &self.state
    }

    /// Get simulator statistics
    #[must_use]
    pub const fn get_stats(&self) -> &JITSimulatorStats {
        &self.stats
    }

    /// Get compiler statistics
    #[must_use]
    pub fn get_compiler_stats(&self) -> JITCompilerStats {
        self.compiler.get_stats()
    }
}

/// Benchmark JIT compilation system
pub fn benchmark_jit_compilation() -> Result<JITBenchmarkResults> {
    let num_qubits = 4;
    let config = JITConfig::default();
    let mut simulator = JITQuantumSimulator::new(num_qubits, config);

    // Create test gate sequences
    let gate_sequences = create_test_gate_sequences(num_qubits);

    let mut results = JITBenchmarkResults {
        total_sequences: gate_sequences.len(),
        compiled_sequences: 0,
        interpreted_sequences: 0,
        average_compilation_time: Duration::from_secs(0),
        average_execution_time_compiled: Duration::from_secs(0),
        average_execution_time_interpreted: Duration::from_secs(0),
        speedup_factor: 1.0,
        compilation_success_rate: 0.0,
        memory_usage_reduction: 0.0,
    };

    let mut total_execution_time_compiled = Duration::from_secs(0);
    let mut total_execution_time_interpreted = Duration::from_secs(0);

    // Run benchmarks
    for sequence in &gate_sequences {
        // First run (interpreted)
        let interpreted_time = simulator.apply_gate_sequence(sequence)?;
        total_execution_time_interpreted += interpreted_time;
        results.interpreted_sequences += 1;

        // Second run (potentially compiled)
        let execution_time = simulator.apply_gate_sequence(sequence)?;

        // Check if it was compiled
        if simulator.get_stats().compiled_executions > results.compiled_sequences {
            total_execution_time_compiled += execution_time;
            results.compiled_sequences += 1;
        }
    }

    // Calculate averages
    if results.compiled_sequences > 0 {
        results.average_execution_time_compiled =
            total_execution_time_compiled / results.compiled_sequences as u32;
    }

    if results.interpreted_sequences > 0 {
        results.average_execution_time_interpreted =
            total_execution_time_interpreted / results.interpreted_sequences as u32;
    }

    // Calculate speedup factor
    if results.average_execution_time_compiled.as_secs_f64() > 0.0 {
        results.speedup_factor = results.average_execution_time_interpreted.as_secs_f64()
            / results.average_execution_time_compiled.as_secs_f64();
    }

    // Calculate compilation success rate
    results.compilation_success_rate =
        results.compiled_sequences as f64 / results.total_sequences as f64;

    // Get compiler stats
    let compiler_stats = simulator.get_compiler_stats();
    if compiler_stats.total_compilations > 0 {
        results.average_compilation_time =
            compiler_stats.total_compilation_time / compiler_stats.total_compilations as u32;
    }

    Ok(results)
}

/// Create test gate sequences for benchmarking
pub fn create_test_gate_sequences(num_qubits: usize) -> Vec<Vec<InterfaceGate>> {
    let mut sequences = Vec::new();

    // Simple sequences
    for target in 0..num_qubits {
        sequences.push(vec![InterfaceGate::new(
            InterfaceGateType::PauliX,
            vec![target],
        )]);

        sequences.push(vec![InterfaceGate::new(
            InterfaceGateType::Hadamard,
            vec![target],
        )]);

        sequences.push(vec![InterfaceGate::new(
            InterfaceGateType::RX(std::f64::consts::PI / 4.0),
            vec![target],
        )]);
    }

    // Two-qubit sequences
    for control in 0..num_qubits {
        for target in 0..num_qubits {
            if control != target {
                sequences.push(vec![InterfaceGate::new(
                    InterfaceGateType::CNOT,
                    vec![control, target],
                )]);
            }
        }
    }

    // Longer sequences for compilation testing
    for target in 0..num_qubits {
        let sequence = vec![
            InterfaceGate::new(InterfaceGateType::Hadamard, vec![target]),
            InterfaceGate::new(
                InterfaceGateType::RZ(std::f64::consts::PI / 8.0),
                vec![target],
            ),
            InterfaceGate::new(InterfaceGateType::Hadamard, vec![target]),
        ];
        sequences.push(sequence);
    }

    // Repeat sequences multiple times to trigger compilation
    let mut repeated_sequences = Vec::new();
    for sequence in &sequences[0..5] {
        for _ in 0..15 {
            repeated_sequences.push(sequence.clone());
        }
    }

    sequences.extend(repeated_sequences);
    sequences
}
