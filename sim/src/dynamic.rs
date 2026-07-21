//! Dynamic circuit dispatch for circuits of varying qubit counts.
//!
//! [`DynamicCircuit`] wraps concrete const-generic circuit types behind an
//! enum so circuits of different sizes can be handled uniformly at runtime
//! without heap allocation overhead.

use crate::simulator::Simulator; // Local simulator trait
#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::PyResult;
use quantrs2_circuit::builder::Circuit;
use quantrs2_circuit::builder::Simulator as CircuitSimulator; // Circuit simulator trait
#[cfg(feature = "python")]
use quantrs2_core::gate::multi::{CRX, CRY, CRZ};
#[cfg(feature = "python")]
use quantrs2_core::gate::single::{RotationX, RotationY, RotationZ};
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
};
use scirs2_core::Complex64;

// Unused imports
#[allow(unused_imports)]
use crate::simulator::SimulatorResult;
use crate::statevector::StateVectorSimulator;
#[allow(unused_imports)]
use quantrs2_core::qubit::QubitId;
#[allow(unused_imports)]
use std::collections::HashMap;

#[cfg(all(feature = "gpu", not(target_os = "macos")))]
use crate::gpu::GpuStateVectorSimulator;

/// A dynamic circuit that encapsulates circuits of different qubit counts
///
/// Wraps the const-generic [`quantrs2_circuit::builder::Circuit`] types behind an
/// enum so circuits of different sizes can be handled at runtime without heap
/// allocation overhead.
///
/// # Examples
///
/// ```rust
/// use quantrs2_sim::dynamic::DynamicCircuit;
/// use quantrs2_core::gate::functions::single::Hadamard;
/// use quantrs2_core::qubit::QubitId;
///
/// // Create a 2-qubit circuit dynamically
/// let mut dc = DynamicCircuit::new(2).expect("2 qubits supported");
/// assert_eq!(dc.num_qubits(), 2);
///
/// // Apply a Hadamard gate
/// let q0 = QubitId::new(0);
/// dc.apply_gate(Hadamard { target: q0 }).expect("H gate applied");
/// assert_eq!(dc.gates().len(), 1);
/// ```
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
            2 => Ok(Self::Q2(Circuit::<2>::new())),
            3 => Ok(Self::Q3(Circuit::<3>::new())),
            4 => Ok(Self::Q4(Circuit::<4>::new())),
            5 => Ok(Self::Q5(Circuit::<5>::new())),
            6 => Ok(Self::Q6(Circuit::<6>::new())),
            7 => Ok(Self::Q7(Circuit::<7>::new())),
            8 => Ok(Self::Q8(Circuit::<8>::new())),
            9 => Ok(Self::Q9(Circuit::<9>::new())),
            10 => Ok(Self::Q10(Circuit::<10>::new())),
            12 => Ok(Self::Q12(Circuit::<12>::new())),
            16 => Ok(Self::Q16(Circuit::<16>::new())),
            20 => Ok(Self::Q20(Circuit::<20>::new())),
            24 => Ok(Self::Q24(Circuit::<24>::new())),
            32 => Ok(Self::Q32(Circuit::<32>::new())),
            _ => Err(QuantRS2Error::UnsupportedQubits(
                n_qubits,
                "Supported qubit counts are 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, and 32."
                    .to_string(),
            )),
        }
    }

    /// Get the list of gate names in the circuit
    #[must_use]
    pub fn gates(&self) -> Vec<String> {
        self.get_gate_names()
    }

    // This method is duplicated later in the file, removing it here

    // This method is duplicated later in the file, removing it here

    // This method is duplicated later in the file, removing it here

    // This method is duplicated later in the file, removing it here

    // This method is duplicated later in the file, removing it here

    /// Get the number of qubits in the circuit
    #[must_use]
    pub const fn num_qubits(&self) -> usize {
        match self {
            Self::Q2(_) => 2,
            Self::Q3(_) => 3,
            Self::Q4(_) => 4,
            Self::Q5(_) => 5,
            Self::Q6(_) => 6,
            Self::Q7(_) => 7,
            Self::Q8(_) => 8,
            Self::Q9(_) => 9,
            Self::Q10(_) => 10,
            Self::Q12(_) => 12,
            Self::Q16(_) => 16,
            Self::Q20(_) => 20,
            Self::Q24(_) => 24,
            Self::Q32(_) => 32,
        }
    }

    /// Get the gate names in the circuit
    #[must_use]
    pub fn get_gate_names(&self) -> Vec<String> {
        match self {
            Self::Q2(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q3(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q4(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q5(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q6(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q7(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q8(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q9(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q10(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q12(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q16(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q20(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q24(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
            Self::Q32(c) => c
                .gates()
                .iter()
                .map(|gate| gate.name().to_string())
                .collect(),
        }
    }

    /// Get a reference to the `flat_index`-th gate in this circuit,
    /// regardless of which concrete qubit-count variant it is. This is
    /// what lets the introspection getters below give real (not
    /// hardcoded) answers for every circuit size, not just [`Self::Q2`].
    #[cfg(feature = "python")]
    fn get_gate_by_flat_index(&self, flat_index: usize) -> Option<&(dyn GateOp + Send + Sync)> {
        match self {
            Self::Q2(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q3(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q4(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q5(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q6(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q7(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q8(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q9(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q10(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q12(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q16(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q20(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q24(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
            Self::Q32(c) => c.gates().get(flat_index).map(|g| g.as_ref()),
        }
    }

    /// Find the `index`-th gate named `gate_type` and return a reference
    /// to it, or an honest `PyErr` if there is no such occurrence.
    #[cfg(feature = "python")]
    fn find_nth_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> PyResult<&(dyn GateOp + Send + Sync)> {
        let gates = self.get_gate_names();
        let mut count = 0;
        for (i, name) in gates.iter().enumerate() {
            if name == gate_type {
                if count == index {
                    return self.get_gate_by_flat_index(i).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "Gate {gate_type} at index {index} vanished while reading it"
                        ))
                    });
                }
                count += 1;
            }
        }
        Err(PyValueError::new_err(format!(
            "Gate {gate_type} at index {index} not found"
        )))
    }

    /// Get the qubit for a single-qubit gate. Works for every circuit
    /// size (previously only [`Self::Q2`] returned the real qubit; every
    /// other variant returned a hardcoded `0`).
    #[cfg(feature = "python")]
    pub fn get_single_qubit_for_gate(&self, gate_type: &str, index: usize) -> PyResult<u32> {
        let gate = self.find_nth_gate(gate_type, index)?;
        let qubits = gate.qubits();
        if qubits.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "Gate {gate_type} at index {index} acts on {} qubit(s), expected 1",
                qubits.len()
            )));
        }
        Ok(qubits[0].id())
    }

    /// Get the real `(qubit, angle)` parameters for a single-qubit
    /// rotation gate (RX/RY/RZ) by downcasting to the concrete gate
    /// struct and reading its actual `theta` field -- not a hardcoded
    /// `(0, 0.0)`.
    #[cfg(feature = "python")]
    pub fn get_rotation_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> PyResult<(u32, f64)> {
        let gate = self.find_nth_gate(gate_type, index)?;
        let qubits = gate.qubits();
        if qubits.is_empty() {
            return Err(PyValueError::new_err(format!(
                "Gate {gate_type} at index {index} has no qubits"
            )));
        }

        let theta = if let Some(g) = gate.as_any().downcast_ref::<RotationX>() {
            g.theta
        } else if let Some(g) = gate.as_any().downcast_ref::<RotationY>() {
            g.theta
        } else if let Some(g) = gate.as_any().downcast_ref::<RotationZ>() {
            g.theta
        } else {
            return Err(PyValueError::new_err(format!(
                "Gate {gate_type} at index {index} is not a recognized single-qubit rotation \
                 gate (RX/RY/RZ); cannot extract its rotation angle"
            )));
        };

        Ok((qubits[0].id(), theta))
    }

    /// Get the real `(qubit1, qubit2)` for a two-qubit gate (CNOT, CZ,
    /// SWAP, etc.), for every circuit size.
    #[cfg(feature = "python")]
    pub fn get_two_qubit_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> PyResult<(u32, u32)> {
        let gate = self.find_nth_gate(gate_type, index)?;
        let qubits = gate.qubits();
        if qubits.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Gate {gate_type} at index {index} acts on {} qubit(s), expected 2",
                qubits.len()
            )));
        }
        Ok((qubits[0].id(), qubits[1].id()))
    }

    /// Get the real `(control, target, angle)` for a controlled rotation
    /// gate (CRX/CRY/CRZ) by downcasting to the concrete gate struct and
    /// reading its actual `theta` field -- not a hardcoded
    /// `(0, 1, 0.0)`.
    #[cfg(feature = "python")]
    pub fn get_controlled_rotation_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> PyResult<(u32, u32, f64)> {
        let gate = self.find_nth_gate(gate_type, index)?;
        let qubits = gate.qubits();
        if qubits.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "Gate {gate_type} at index {index} acts on {} qubit(s), expected 2 (control, target)",
                qubits.len()
            )));
        }

        let theta = if let Some(g) = gate.as_any().downcast_ref::<CRX>() {
            g.theta
        } else if let Some(g) = gate.as_any().downcast_ref::<CRY>() {
            g.theta
        } else if let Some(g) = gate.as_any().downcast_ref::<CRZ>() {
            g.theta
        } else {
            return Err(PyValueError::new_err(format!(
                "Gate {gate_type} at index {index} is not a recognized controlled rotation gate \
                 (CRX/CRY/CRZ); cannot extract its rotation angle"
            )));
        };

        Ok((qubits[0].id(), qubits[1].id(), theta))
    }

    /// Get the real `(qubit1, qubit2, qubit3)` for a three-qubit gate
    /// (Toffoli, Fredkin, etc.), for every circuit size.
    #[cfg(feature = "python")]
    pub fn get_three_qubit_params_for_gate(
        &self,
        gate_type: &str,
        index: usize,
    ) -> PyResult<(u32, u32, u32)> {
        let gate = self.find_nth_gate(gate_type, index)?;
        let qubits = gate.qubits();
        if qubits.len() != 3 {
            return Err(PyValueError::new_err(format!(
                "Gate {gate_type} at index {index} acts on {} qubit(s), expected 3",
                qubits.len()
            )));
        }
        Ok((qubits[0].id(), qubits[1].id(), qubits[2].id()))
    }

    /// Apply a gate to the circuit
    pub fn apply_gate<G: GateOp + Clone + Send + Sync + 'static>(
        &mut self,
        gate: G,
    ) -> QuantRS2Result<()> {
        match self {
            Self::Q2(c) => c.add_gate(gate).map(|_| ()),
            Self::Q3(c) => c.add_gate(gate).map(|_| ()),
            Self::Q4(c) => c.add_gate(gate).map(|_| ()),
            Self::Q5(c) => c.add_gate(gate).map(|_| ()),
            Self::Q6(c) => c.add_gate(gate).map(|_| ()),
            Self::Q7(c) => c.add_gate(gate).map(|_| ()),
            Self::Q8(c) => c.add_gate(gate).map(|_| ()),
            Self::Q9(c) => c.add_gate(gate).map(|_| ()),
            Self::Q10(c) => c.add_gate(gate).map(|_| ()),
            Self::Q12(c) => c.add_gate(gate).map(|_| ()),
            Self::Q16(c) => c.add_gate(gate).map(|_| ()),
            Self::Q20(c) => c.add_gate(gate).map(|_| ()),
            Self::Q24(c) => c.add_gate(gate).map(|_| ()),
            Self::Q32(c) => c.add_gate(gate).map(|_| ()),
        }
    }

    /// Run the circuit on a CPU simulator
    pub fn run(&self, simulator: &StateVectorSimulator) -> QuantRS2Result<DynamicResult> {
        match self {
            Self::Q2(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 2,
                })
            }
            Self::Q3(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 3,
                })
            }
            Self::Q4(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 4,
                })
            }
            Self::Q5(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 5,
                })
            }
            Self::Q6(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 6,
                })
            }
            Self::Q7(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 7,
                })
            }
            Self::Q8(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 8,
                })
            }
            Self::Q9(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 9,
                })
            }
            Self::Q10(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 10,
                })
            }
            Self::Q12(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 12,
                })
            }
            Self::Q16(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 16,
                })
            }
            Self::Q20(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 20,
                })
            }
            Self::Q24(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 24,
                })
            }
            Self::Q32(c) => {
                let result = simulator.run(c)?;
                Ok(DynamicResult {
                    amplitudes: result.amplitudes().to_vec(),
                    num_qubits: 32,
                })
            }
        }
    }

    /// Check if GPU acceleration is available
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub fn is_gpu_available() -> bool {
        GpuStateVectorSimulator::is_available()
    }

    /// Run the circuit on a GPU simulator
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
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

    /// Check if GPU acceleration is available (stub for macOS)
    #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
    #[must_use]
    pub const fn is_gpu_available() -> bool {
        false
    }

    /// Run the circuit on a GPU simulator (stub for macOS)
    #[cfg(not(all(feature = "gpu", not(target_os = "macos"))))]
    pub fn run_gpu(&self) -> QuantRS2Result<DynamicResult> {
        Err(QuantRS2Error::BackendExecutionFailed(
            "GPU acceleration is not available on this platform".to_string(),
        ))
    }

    /// Run the circuit on the best available simulator (GPU if available, CPU otherwise)
    #[cfg(all(feature = "gpu", not(target_os = "macos")))]
    pub fn run_best(&self) -> QuantRS2Result<DynamicResult> {
        if Self::is_gpu_available() && self.num_qubits() >= 4 {
            self.run_gpu()
        } else {
            let simulator = StateVectorSimulator::new();
            self.run(&simulator)
        }
    }

    /// Run the circuit on the best available simulator (CPU only on macOS with GPU feature)
    #[cfg(all(feature = "gpu", target_os = "macos"))]
    pub fn run_best(&self) -> QuantRS2Result<DynamicResult> {
        let simulator = StateVectorSimulator::new();
        self.run(&simulator)
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
    #[must_use]
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }

    /// Get the probabilities for each basis state
    #[must_use]
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes
            .iter()
            .map(scirs2_core::Complex::norm_sqr)
            .collect()
    }

    /// Get the number of qubits
    #[must_use]
    pub const fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}
