//! WebAssembly Support for QuantRS2
//!
//! This module provides WebAssembly bindings for QuantRS2, enabling
//! quantum circuit simulation directly in web browsers.
//!
//! ## Features
//!
//! - **Browser-Based Quantum Computing**: Run quantum simulations in any modern web browser
//! - **JavaScript Integration**: Seamless integration with JavaScript/TypeScript
//! - **Interactive Visualization**: Real-time circuit visualization and state inspection
//! - **No Installation Required**: Works directly in the browser without Python/Rust installation
//! - **Educational Tools**: Perfect for quantum computing education and demonstrations
//!
//! ## SciRS2 Policy Compliance
//!
//! All numerical operations use SciRS2-Core abstractions:
//! - Complex numbers: `scirs2_core::Complex64`
//! - Arrays: `scirs2_core::ndarray::*`
//! - No direct usage of rand/ndarray/num-complex
//!
//! ## Example Usage (JavaScript)
//!
//! ```javascript
//! import init, { WasmCircuit } from './quantrs2_wasm.js';
//!
//! async function main() {
//!     await init();
//!
//!     const circuit = WasmCircuit.new(2);
//!     circuit.h(0);
//!     circuit.cnot(0, 1);
//!
//!     const result = circuit.run();
//!     console.log(result.probabilities());
//! }
//! ```

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use js_sys::{Array, Object, Reflect};

#[cfg(target_arch = "wasm32")]
use web_sys::console;

use scirs2_core::Complex64;
use scirs2_core::ndarray::Array1;
use quantrs2_sim::dynamic::DynamicCircuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use quantrs2_core::qubit::QubitId;
use std::convert::TryFrom;

/// WebAssembly wrapper for quantum circuits
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmCircuit {
    circuit: DynamicCircuit,
    n_qubits: usize,
}

/// WebAssembly wrapper for simulation results
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmResult {
    amplitudes: Vec<Complex64>,
    n_qubits: usize,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmCircuit {
    /// Create a new quantum circuit
    ///
    /// # Arguments
    ///
    /// * `n_qubits` - Number of qubits in the circuit (2-20 recommended for browser)
    ///
    /// # Example
    ///
    /// ```javascript
    /// const circuit = WasmCircuit.new(2);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(n_qubits: usize) -> Result<WasmCircuit, JsValue> {
        // Set panic hook for better error messages
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        if n_qubits < 2 {
            return Err(JsValue::from_str("Number of qubits must be at least 2"));
        }

        if n_qubits > 20 {
            console::warn_1(&JsValue::from_str(
                &format!("Warning: {} qubits may cause browser to slow down. Recommended max: 20", n_qubits)
            ));
        }

        let circuit = DynamicCircuit::new(n_qubits)
            .map_err(|e| JsValue::from_str(&format!("Failed to create circuit: {}", e)))?;

        Ok(WasmCircuit { circuit, n_qubits })
    }

    /// Get the number of qubits
    #[wasm_bindgen(getter)]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Apply Hadamard gate
    ///
    /// # Example
    ///
    /// ```javascript
    /// circuit.h(0);
    /// ```
    pub fn h(&mut self, qubit: usize) -> Result<(), JsValue> {
        self.apply_single_gate("H", qubit, |q| {
            Ok(quantrs2_core::gate::single::Hadamard { target: q })
        })
    }

    /// Apply Pauli-X gate
    pub fn x(&mut self, qubit: usize) -> Result<(), JsValue> {
        self.apply_single_gate("X", qubit, |q| {
            Ok(quantrs2_core::gate::single::PauliX { target: q })
        })
    }

    /// Apply Pauli-Y gate
    pub fn y(&mut self, qubit: usize) -> Result<(), JsValue> {
        self.apply_single_gate("Y", qubit, |q| {
            Ok(quantrs2_core::gate::single::PauliY { target: q })
        })
    }

    /// Apply Pauli-Z gate
    pub fn z(&mut self, qubit: usize) -> Result<(), JsValue> {
        self.apply_single_gate("Z", qubit, |q| {
            Ok(quantrs2_core::gate::single::PauliZ { target: q })
        })
    }

    /// Apply S gate (phase gate)
    pub fn s(&mut self, qubit: usize) -> Result<(), JsValue> {
        self.apply_single_gate("S", qubit, |q| {
            Ok(quantrs2_core::gate::single::Phase { target: q })
        })
    }

    /// Apply T gate
    pub fn t(&mut self, qubit: usize) -> Result<(), JsValue> {
        self.apply_single_gate("T", qubit, |q| {
            Ok(quantrs2_core::gate::single::T { target: q })
        })
    }

    /// Apply RX rotation gate
    ///
    /// # Example
    ///
    /// ```javascript
    /// circuit.rx(0, Math.PI / 2);
    /// ```
    pub fn rx(&mut self, qubit: usize, theta: f64) -> Result<(), JsValue> {
        let qubit_id = self.checked_qubit(qubit)?;
        self.circuit
            .apply_gate(quantrs2_core::gate::single::RotationX {
                target: qubit_id,
                theta,
            })
            .map_err(|e| JsValue::from_str(&format!("Error applying RX gate: {}", e)))
    }

    /// Apply RY rotation gate
    pub fn ry(&mut self, qubit: usize, theta: f64) -> Result<(), JsValue> {
        let qubit_id = self.checked_qubit(qubit)?;
        self.circuit
            .apply_gate(quantrs2_core::gate::single::RotationY {
                target: qubit_id,
                theta,
            })
            .map_err(|e| JsValue::from_str(&format!("Error applying RY gate: {}", e)))
    }

    /// Apply RZ rotation gate
    pub fn rz(&mut self, qubit: usize, theta: f64) -> Result<(), JsValue> {
        let qubit_id = self.checked_qubit(qubit)?;
        self.circuit
            .apply_gate(quantrs2_core::gate::single::RotationZ {
                target: qubit_id,
                theta,
            })
            .map_err(|e| JsValue::from_str(&format!("Error applying RZ gate: {}", e)))
    }

    /// Apply CNOT gate
    ///
    /// # Example
    ///
    /// ```javascript
    /// circuit.cnot(0, 1);
    /// ```
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<(), JsValue> {
        let control_id = self.checked_qubit(control)?;
        let target_id = self.checked_qubit(target)?;

        self.circuit
            .apply_gate(quantrs2_core::gate::multi::CNOT {
                control: control_id,
                target: target_id,
            })
            .map_err(|e| JsValue::from_str(&format!("Error applying CNOT gate: {}", e)))
    }

    /// Apply CZ gate
    pub fn cz(&mut self, control: usize, target: usize) -> Result<(), JsValue> {
        let control_id = self.checked_qubit(control)?;
        let target_id = self.checked_qubit(target)?;

        self.circuit
            .apply_gate(quantrs2_core::gate::multi::CZ {
                control: control_id,
                target: target_id,
            })
            .map_err(|e| JsValue::from_str(&format!("Error applying CZ gate: {}", e)))
    }

    /// Apply SWAP gate
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> Result<(), JsValue> {
        let q1 = self.checked_qubit(qubit1)?;
        let q2 = self.checked_qubit(qubit2)?;

        self.circuit
            .apply_gate(quantrs2_core::gate::multi::SWAP { qubit1: q1, qubit2: q2 })
            .map_err(|e| JsValue::from_str(&format!("Error applying SWAP gate: {}", e)))
    }

    /// Run the circuit simulation
    ///
    /// # Example
    ///
    /// ```javascript
    /// const result = circuit.run();
    /// const probs = result.probabilities();
    /// ```
    pub fn run(&self) -> Result<WasmResult, JsValue> {
        let simulator = StateVectorSimulator::new();
        let result = self.circuit
            .run(&simulator)
            .map_err(|e| JsValue::from_str(&format!("Simulation error: {}", e)))?;

        Ok(WasmResult {
            amplitudes: result.amplitudes().to_vec(),
            n_qubits: result.num_qubits(),
        })
    }

    /// Get circuit as QASM string
    ///
    /// # Example
    ///
    /// ```javascript
    /// const qasm = circuit.to_qasm();
    /// console.log(qasm);
    /// ```
    pub fn to_qasm(&self) -> String {
        // Simplified QASM export
        let mut qasm = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[{}];\n", self.n_qubits);

        // In a full implementation, we would iterate through gates and convert them
        // For now, just return the header
        qasm.push_str("// Circuit gates would be listed here\n");

        qasm
    }

    /// Get a JSON representation of the circuit
    pub fn to_json(&self) -> Result<JsValue, JsValue> {
        let obj = Object::new();

        Reflect::set(
            &obj,
            &JsValue::from_str("n_qubits"),
            &JsValue::from_f64(self.n_qubits as f64),
        )?;

        Reflect::set(
            &obj,
            &JsValue::from_str("gates"),
            &JsValue::from_str("[]"), // Simplified
        )?;

        Ok(obj.into())
    }

    // Helper methods
    fn checked_qubit(&self, qubit: usize) -> Result<QubitId, JsValue> {
        if qubit >= self.n_qubits {
            return Err(JsValue::from_str(&format!(
                "Qubit index {} out of range (circuit has {} qubits)",
                qubit, self.n_qubits
            )));
        }

        let id = u32::try_from(qubit).map_err(|_| {
            JsValue::from_str(&format!(
                "Qubit index {} exceeds maximum supported range",
                qubit
            ))
        })?;

        Ok(QubitId::new(id))
    }

    fn apply_single_gate<F, G>(&mut self, gate_name: &str, qubit: usize, gate_fn: F) -> Result<(), JsValue>
    where
        F: FnOnce(QubitId) -> Result<G, JsValue>,
        G: quantrs2_core::gate::QuantumGate,
    {
        let qubit_id = self.checked_qubit(qubit)?;
        let gate = gate_fn(qubit_id)?;

        self.circuit
            .apply_gate(gate)
            .map_err(|e| JsValue::from_str(&format!("Error applying {} gate: {}", gate_name, e)))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmResult {
    /// Get the number of qubits
    #[wasm_bindgen(getter)]
    pub fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get probabilities as a JavaScript array
    ///
    /// # Example
    ///
    /// ```javascript
    /// const probs = result.probabilities();
    /// console.log(probs);
    /// ```
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Get the amplitude for a specific basis state
    ///
    /// # Example
    ///
    /// ```javascript
    /// const amp = result.get_amplitude(0); // Get |00...0⟩ amplitude
    /// ```
    pub fn get_amplitude(&self, index: usize) -> Result<Array, JsValue> {
        if index >= self.amplitudes.len() {
            return Err(JsValue::from_str(&format!(
                "Index {} out of range (max: {})",
                index,
                self.amplitudes.len() - 1
            )));
        }

        let amp = self.amplitudes[index];
        let arr = Array::new();
        arr.push(&JsValue::from_f64(amp.re));
        arr.push(&JsValue::from_f64(amp.im));

        Ok(arr)
    }

    /// Get all amplitudes as a flat array [re0, im0, re1, im1, ...]
    pub fn amplitudes_flat(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.amplitudes.len() * 2);
        for amp in &self.amplitudes {
            result.push(amp.re);
            result.push(amp.im);
        }
        result
    }

    /// Get state probabilities as a JavaScript Map
    pub fn state_probabilities(&self) -> Result<Object, JsValue> {
        let obj = Object::new();

        for (i, amp) in self.amplitudes.iter().enumerate() {
            let prob = amp.norm_sqr();
            if prob > 1e-10 {
                let basis_state = format!("{:0width$b}", i, width = self.n_qubits);
                Reflect::set(
                    &obj,
                    &JsValue::from_str(&basis_state),
                    &JsValue::from_f64(prob),
                )?;
            }
        }

        Ok(obj)
    }

    /// Get the most probable state
    pub fn most_probable_state(&self) -> String {
        let (max_idx, _) = self
            .amplitudes
            .iter()
            .enumerate()
            .map(|(i, a)| (i, a.norm_sqr()))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, 0.0));

        format!("{:0width$b}", max_idx, width = self.n_qubits)
    }
}

/// Create a Bell state circuit (convenience function)
///
/// # Example
///
/// ```javascript
/// const circuit = create_bell_state();
/// const result = circuit.run();
/// ```
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn create_bell_state() -> Result<WasmCircuit, JsValue> {
    let mut circuit = WasmCircuit::new(2)?;
    circuit.h(0)?;
    circuit.cnot(0, 1)?;
    Ok(circuit)
}

/// Create a GHZ state circuit
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn create_ghz_state(n_qubits: usize) -> Result<WasmCircuit, JsValue> {
    let mut circuit = WasmCircuit::new(n_qubits)?;
    circuit.h(0)?;
    for i in 0..n_qubits - 1 {
        circuit.cnot(i, i + 1)?;
    }
    Ok(circuit)
}

/// Get version information
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Initialize the WASM module (must be called first)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn init_wasm() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    console::log_1(&JsValue::from_str(&format!(
        "QuantRS2 WASM v{} initialized",
        env!("CARGO_PKG_VERSION")
    )));
}

// Non-WASM stub implementations
#[cfg(not(target_arch = "wasm32"))]
pub struct WasmCircuit;

#[cfg(not(target_arch = "wasm32"))]
pub struct WasmResult;

#[cfg(not(target_arch = "wasm32"))]
pub fn create_bell_state() -> WasmCircuit {
    WasmCircuit
}

#[cfg(not(target_arch = "wasm32"))]
pub fn create_ghz_state(_n_qubits: usize) -> WasmCircuit {
    WasmCircuit
}

#[cfg(not(target_arch = "wasm32"))]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
