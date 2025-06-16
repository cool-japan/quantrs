use crate::simulator::Simulator; // Local simulator trait
use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;
use quantrs2_circuit::builder::Circuit;
use quantrs2_circuit::builder::Simulator as CircuitSimulator; // Circuit simulator trait
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
};

// Unused imports
#[allow(unused_imports)]
use crate::simulator::SimulatorResult;
use crate::statevector::StateVectorSimulator;
#[allow(unused_imports)]
use quantrs2_core::qubit::QubitId;
#[allow(unused_imports)]
use std::collections::HashMap;

#[cfg(feature = "gpu")]
use crate::gpu::GpuStateVectorSimulator;

/// A dynamic circuit that encapsulates circuits of different qubit counts
pub enum DynamicCircuit {
    /// 2-qubit circuit
    Q2(Circuit<2>),
    /// 3-qubit circuit
    Q3(Circuit<3>),
    /// 4-qubit circuit
    Q4(Circuit<4>),
    /// 5-qubit circuit
    Q5(Circuit<5>),
    /// 6-qubit circuit
    Q6(Circuit<6>),
    /// 7-qubit circuit
    Q7(Circuit<7>),
    /// 8-qubit circuit
    Q8(Circuit<8>),
    /// 9-qubit circuit
    Q9(Circuit<9>),
    /// 10-qubit circuit
    Q10(Circuit<10>),
    /// 12-qubit circuit
    Q12(Circuit<12>),
    /// 16-qubit circuit
    Q16(Circuit<16>),
    /// 20-qubit circuit
    Q20(Circuit<20>),
    /// 24-qubit circuit
    Q24(Circuit<24>),
    /// 32-qubit circuit
    Q32(Circuit<32>),
}

impl DynamicCircuit {
    /// Create a new dynamic circuit with the specified number of qubits
    pub fn new(n_qubits: usize) -> QuantRS2Result<Self> {
        match n_qubits {
            2 => Ok(DynamicCircuit::Q2(Circuit::<2>::new())),
            3 => Ok(DynamicCircuit::Q3(Circuit::<3>::new())),
            4 => Ok(DynamicCircuit::Q4(Circuit::<4>::new())),
            5 => Ok(DynamicCircuit::Q5(Circuit::<5>::new())),
            6 => Ok(DynamicCircuit::Q6(Circuit::<6>::new())),
            7 => Ok(DynamicCircuit::Q7(Circuit::<7>::new())),
            8 => Ok(DynamicCircuit::Q8(Circuit::<8>::new())),
            9 => Ok(DynamicCircuit::Q9(Circuit::<9>::new())),
            10 => Ok(DynamicCircuit::Q10(Circuit::<10>::new())),
            12 => Ok(DynamicCircuit::Q12(Circuit::<12>::new())),
            16 => Ok(DynamicCircuit::Q16(Circuit::<16>::new())),
            20 => Ok(DynamicCircuit::Q20(Circuit::<20>::new())),
            24 => Ok(DynamicCircuit::Q24(Circuit::<24>::new())),
            32 => Ok(DynamicCircuit::Q32(Circuit::<32>::new())),
            _ => Err(QuantRS2Error::UnsupportedQubits(
                n_qubits,
                "Supported qubit counts are 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, and 32."
                    .to_string(),
            )),
        }
    }

    /// Get the list of gate names in the circuit
    pub fn gates(&self) -> Vec<String> {
        self.get_gate_names()
    }

    // This method is duplicated later in the file, removing it here

    // This method is duplicated later in the file, removing it here

    // This method is duplicated later in the file, removing it here

    // This method is duplicated later in the file, removing it here

    // This method is duplicated later in the file, removing it here

    /// Get the number of qubits in the circuit
    pub fn num_qubits(&self) -> usize {
        match self {
            DynamicCircuit::Q2(_) => 2,
            DynamicCircuit::Q3(_) => 3,
            DynamicCircuit::Q4(_) => 4,
            DynamicCircuit::Q5(_) => 5,
            DynamicCircuit::Q6(_) => 6,
            DynamicCircuit::Q7(_) => 7,
            DynamicCircuit::Q8(_) => 8,
            DynamicCircuit::Q9(_) => 9,
            DynamicCircuit::Q10(_) => 10,
            DynamicCircuit::Q12(_) => 12,
            DynamicCircuit::Q16(_) => 16,
            DynamicCircuit::Q20(_) => 20,
            DynamicCircuit::Q24(_) => 24,
            DynamicCircuit::Q32(_) => 32,
        }
    }

    /// Get the gate names in the circuit
    pub fn get_gate_names(&self) -> Vec<String> {
        match self {
            DynamicCircuit::Q2(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q3(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q4(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q5(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q6(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q7(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q8(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q9(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q10(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q12(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q16(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q20(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q24(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            DynamicCircuit::Q32(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
        }
    }

    /// Get the qubit for single-qubit gate
    pub fn get_single_qubit_for_gate(&self, gate_type: &str, index: usize) -> PyResult<u32> {
        // Placeholder for visualization - in a real implementation, we would track this information
        let gate_name = gate_type.to_string();
        let gates = self.get_gate_names();

        // Find the Nth occurrence of this gate type
        let mut count = 0;
        for (i, name) in gates.iter().enumerate() {
            if name == &gate_name {
                if count == index {
                    // Return a placeholder qubit ID - in a real implementation this would be accurate
                    match self {
                        DynamicCircuit::Q2(c) => {
                            if let Some(gate) = c.gates().get(i) {
                                if gate.qubits().len() == 1 {
                                    return Ok(gate.qubits()[0].id());
                                }
                            }
                        }
                        // Repeat for all other qubit counts
                        _ => return Ok(0),
                    }
                }
                count += 1;
            }
        }

        Err(PyValueError::new_err(format!(
            "Gate {} at index {} not found",
            gate_type, index
        )))
    }

    /// Get the parameters for a rotation gate
    pub fn get_rotation_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> PyResult<(u32, f64)> {
        // Placeholder for visualization - in a real implementation, we would track this information
        let gate_name = gate_type.to_string();
        let gates = self.get_gate_names();

        // Find the Nth occurrence of this gate type
        let mut count = 0;
        for name in gates.iter() {
            if name == &gate_name {
                if count == index {
                    // Return placeholder values - in a real implementation these would be accurate
                    return Ok((0, 0.0));
                }
                count += 1;
            }
        }

        Err(PyValueError::new_err(format!(
            "Gate {} at index {} not found",
            gate_type, index
        )))
    }

    /// Get the parameters for a two-qubit gate
    pub fn get_two_qubit_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> PyResult<(u32, u32)> {
        // Placeholder for visualization - in a real implementation, we would track this information
        let gate_name = gate_type.to_string();
        let gates = self.get_gate_names();

        // Find the Nth occurrence of this gate type
        let mut count = 0;
        for name in gates.iter() {
            if name == &gate_name {
                if count == index {
                    // Return placeholder values - in a real implementation these would be accurate
                    return Ok((0, 1));
                }
                count += 1;
            }
        }

        Err(PyValueError::new_err(format!(
            "Gate {} at index {} not found",
            gate_type, index
        )))
    }

    /// Get the parameters for a controlled rotation gate
    pub fn get_controlled_rotation_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> PyResult<(u32, u32, f64)> {
        // Placeholder for visualization - in a real implementation, we would track this information
        let gate_name = gate_type.to_string();
        let gates = self.get_gate_names();

        // Find the Nth occurrence of this gate type
        let mut count = 0;
        for name in gates.iter() {
            if name == &gate_name {
                if count == index {
                    // Return placeholder values - in a real implementation these would be accurate
                    return Ok((0, 1, 0.0));
                }
                count += 1;
            }
        }

        Err(PyValueError::new_err(format!(
            "Gate {} at index {} not found",
            gate_type, index
        )))
    }

    /// Get the parameters for a three-qubit gate
    pub fn get_three_qubit_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> PyResult<(u32, u32, u32)> {
        // Placeholder for visualization - in a real implementation, we would track this information
        let gate_name = gate_type.to_string();
        let gates = self.get_gate_names();

        // Find the Nth occurrence of this gate type
        let mut count = 0;
        for name in gates.iter() {
            if name == &gate_name {
                if count == index {
                    // Return placeholder values - in a real implementation these would be accurate
                    return Ok((0, 1, 2));
                }
                count += 1;
            }
        }

        Err(PyValueError::new_err(format!(
            "Gate {} at index {} not found",
            gate_type, index
        )))
    }

    /// Apply a gate to the circuit
    pub fn apply_gate<G: GateOp + Clone + Send + Sync + 'static>(
        &mut self,
        gate: G,
    ) -> QuantRS2Result<()> {
        match self {
            DynamicCircuit::Q2(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q3(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q4(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q5(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q6(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q7(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q8(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q9(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q10(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q12(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q16(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q20(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q24(c) => c.add_gate(gate).map(|_| ()),
            DynamicCircuit::Q32(c) => c.add_gate(gate).map(|_| ()),
        }
    }

    /// Run the circuit on a CPU simulator
    pub fn run(&self, simulator: &StateVectorSimulator) -> QuantRS2Result<DynamicResult> {
        match self {
            DynamicCircuit::Q2(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 2,
                })
            }
            DynamicCircuit::Q3(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 3,
                })
            }
            DynamicCircuit::Q4(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 4,
                })
            }
            DynamicCircuit::Q5(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 5,
                })
            }
            DynamicCircuit::Q6(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 6,
                })
            }
            DynamicCircuit::Q7(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 7,
                })
            }
            DynamicCircuit::Q8(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 8,
                })
            }
            DynamicCircuit::Q9(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 9,
                })
            }
            DynamicCircuit::Q10(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 10,
                })
            }
            DynamicCircuit::Q12(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 12,
                })
            }
            DynamicCircuit::Q16(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 16,
                })
            }
            DynamicCircuit::Q20(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 20,
                })
            }
            DynamicCircuit::Q24(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 24,
                })
            }
            DynamicCircuit::Q32(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 32,
                })
            }
        }
    }

    /// Check if GPU acceleration is available
    #[cfg(feature = "gpu")]
    pub fn is_gpu_available() -> bool {
        GpuStateVectorSimulator::is_available()
    }

    /// Run the circuit on a GPU simulator
    #[cfg(feature = "gpu")]
    pub fn run_gpu(&self) -> QuantRS2Result<DynamicResult> {
        // Try to create the GPU simulator
        let mut gpu_simulator = match GpuStateVectorSimulator::new_blocking() {
            Ok(sim) => sim,
            Err(e) => {
                return Err(QuantRS2Error::BackendExecutionFailed(format!(
                    "Failed to create GPU simulator: {}",
                    e
                )))
            }
        };

        // Run the circuit on the GPU
        match self {
            DynamicCircuit::Q2(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 2,
                })
            }
            DynamicCircuit::Q3(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 3,
                })
            }
            DynamicCircuit::Q4(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 4,
                })
            }
            DynamicCircuit::Q5(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 5,
                })
            }
            DynamicCircuit::Q6(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 6,
                })
            }
            DynamicCircuit::Q7(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 7,
                })
            }
            DynamicCircuit::Q8(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 8,
                })
            }
            DynamicCircuit::Q9(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 9,
                })
            }
            DynamicCircuit::Q10(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 10,
                })
            }
            DynamicCircuit::Q12(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 12,
                })
            }
            DynamicCircuit::Q16(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 16,
                })
            }
            DynamicCircuit::Q20(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 20,
                })
            }
            DynamicCircuit::Q24(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 24,
                })
            }
            DynamicCircuit::Q32(c) => {
                let result = gpu_simulator.run(c).map_err(|e| {
                    QuantRS2Error::BackendExecutionFailed(format!("GPU simulation failed: {}", e))
                })?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes.clone(),
                    num_qubits: 32,
                })
            }
        }
    }

    /// Run the circuit on the best available simulator (GPU if available, CPU otherwise)
    #[cfg(feature = "gpu")]
    pub fn run_best(&self) -> QuantRS2Result<DynamicResult> {
        if Self::is_gpu_available() && self.num_qubits() >= 4 {
            self.run_gpu()
        } else {
            let simulator = StateVectorSimulator::new();
            self.run(&simulator)
        }
    }

    /// Run the circuit on the best available simulator (CPU only if GPU feature is disabled)
    #[cfg(not(feature = "gpu"))]
    pub fn run_best(&self) -> QuantRS2Result<DynamicResult> {
        let simulator = StateVectorSimulator::new();
        self.run(&simulator)
    }
}

/// Dynamic simulation result that can handle any qubit count
pub struct DynamicResult {
    /// State vector amplitudes
    pub amplitudes: Vec<Complex64>,
    /// Number of qubits
    pub num_qubits: usize,
}

impl DynamicResult {
    /// Get the state vector amplitudes
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }

    /// Get the probabilities for each basis state
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|amp| amp.norm_sqr()).collect()
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}
