//! Parametric quantum circuits module.
//!
//! This module provides parametric gates for variational algorithms
//! (VQE/QAOA-style workflows). Unlike a purely bookkeeping representation,
//! `ParametricCircuit` records a real, ordered gate sequence and simulates it
//! on demand via `quantrs2-sim`'s statevector simulator (see
//! [`PyParametricCircuit::get_statevector`]); `CircuitOptimizer::step`
//! performs a real parameter-shift-rule gradient-descent update against a
//! caller-supplied observable.

// Allow unused_self for PyO3 method bindings and unnecessary_wraps for future error handling
#![allow(clippy::unused_self)]
#![allow(clippy::unnecessary_wraps)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use quantrs2_core::qubit::QubitId;
use quantrs2_sim::dynamic::DynamicCircuit;
use quantrs2_sim::statevector::StateVectorSimulator;
use scirs2_core::ndarray::Array2;
use scirs2_core::Complex64;
use scirs2_numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray2};
use std::collections::HashMap;

/// A single gate in a parametric circuit's real gate sequence. Fixed gates
/// carry no parameter; parametrized gates carry the name of the parameter
/// bound to their rotation angle (resolved at simulation time).
#[derive(Debug, Clone)]
enum ParamGate {
    H(usize),
    X(usize),
    Cnot(usize, usize),
    Rx(usize, String),
    Ry(usize, String),
    Rz(usize, String),
    Rxx(usize, usize, String),
    Ryy(usize, usize, String),
}

/// Python wrapper for parametric quantum circuits
#[pyclass(name = "ParametricCircuit")]
pub struct PyParametricCircuit {
    pub n_qubits: usize,
    pub parameters: HashMap<String, f64>,
    /// Parameter names in first-registration order. `get_statevector`'s
    /// `params` argument is positional against this order.
    param_order: Vec<String>,
    /// The real, ordered gate sequence applied by [`Self::simulate`].
    gates: Vec<ParamGate>,
}

impl PyParametricCircuit {
    /// Pure-Rust qubit-index validation, kept independent of `PyErr` so
    /// `simulate` is directly unit-testable without a Python interpreter
    /// (this crate builds `pyo3` with the `extension-module` feature, so a
    /// standalone test binary cannot resolve the CPython C-API symbols that
    /// `PyErr` construction pulls in -- see `Cargo.toml`'s `test = false`
    /// note on the `[lib]` target).
    fn checked_qubit(&self, qubit: usize) -> Result<QubitId, String> {
        if qubit >= self.n_qubits {
            return Err(format!(
                "Qubit index {qubit} out of range for a {}-qubit circuit",
                self.n_qubits
            ));
        }
        u32::try_from(qubit)
            .map(QubitId::new)
            .map_err(|_| format!("Qubit index {qubit} exceeds the maximum supported range"))
    }

    /// Register a parameter name (idempotent) and record its insertion order.
    fn register_parameter(&mut self, name: &str, value: f64) {
        if !self.parameters.contains_key(name) {
            self.param_order.push(name.to_string());
        }
        self.parameters.insert(name.to_string(), value);
    }

    /// Bind `bound_params` positionally against [`Self::param_order`] and
    /// simulate the real gate sequence on a fresh `DynamicCircuit`, returning
    /// the resulting statevector amplitudes. Pure Rust (`Result<_, String>`,
    /// no `PyErr`) so this is directly unit-testable; see [`Self::checked_qubit`].
    pub(crate) fn simulate(&self, bound_params: &[f64]) -> Result<Vec<Complex64>, String> {
        if bound_params.len() != self.param_order.len() {
            return Err(format!(
                "Expected {} parameter value(s) (one per registered parameter, in \
                 registration order {:?}), got {}",
                self.param_order.len(),
                self.param_order,
                bound_params.len()
            ));
        }
        let bound: HashMap<&str, f64> = self
            .param_order
            .iter()
            .map(String::as_str)
            .zip(bound_params.iter().copied())
            .collect();
        let resolve = |name: &str| -> Result<f64, String> {
            bound
                .get(name)
                .copied()
                .ok_or_else(|| format!("Unknown parameter '{name}'"))
        };

        let mut circuit = DynamicCircuit::new(self.n_qubits)
            .map_err(|e| format!("Cannot build a {}-qubit circuit: {e}", self.n_qubits))?;

        for gate in &self.gates {
            match gate {
                ParamGate::H(q) => {
                    let target = self.checked_qubit(*q)?;
                    circuit
                        .apply_gate(quantrs2_core::gate::single::Hadamard { target })
                        .map_err(|e| format!("Error applying gate: {e}"))?;
                }
                ParamGate::X(q) => {
                    let target = self.checked_qubit(*q)?;
                    circuit
                        .apply_gate(quantrs2_core::gate::single::PauliX { target })
                        .map_err(|e| format!("Error applying gate: {e}"))?;
                }
                ParamGate::Cnot(c, t) => {
                    let control = self.checked_qubit(*c)?;
                    let target = self.checked_qubit(*t)?;
                    circuit
                        .apply_gate(quantrs2_core::gate::multi::CNOT { control, target })
                        .map_err(|e| format!("Error applying gate: {e}"))?;
                }
                ParamGate::Rx(q, name) => {
                    let theta = resolve(name)?;
                    let target = self.checked_qubit(*q)?;
                    circuit
                        .apply_gate(quantrs2_core::gate::single::RotationX { target, theta })
                        .map_err(|e| format!("Error applying gate: {e}"))?;
                }
                ParamGate::Ry(q, name) => {
                    let theta = resolve(name)?;
                    let target = self.checked_qubit(*q)?;
                    circuit
                        .apply_gate(quantrs2_core::gate::single::RotationY { target, theta })
                        .map_err(|e| format!("Error applying gate: {e}"))?;
                }
                ParamGate::Rz(q, name) => {
                    let theta = resolve(name)?;
                    let target = self.checked_qubit(*q)?;
                    circuit
                        .apply_gate(quantrs2_core::gate::single::RotationZ { target, theta })
                        .map_err(|e| format!("Error applying gate: {e}"))?;
                }
                ParamGate::Rxx(q1, q2, name) => {
                    let theta = resolve(name)?;
                    let qubit1 = self.checked_qubit(*q1)?;
                    let qubit2 = self.checked_qubit(*q2)?;
                    circuit
                        .apply_gate(quantrs2_core::gate::multi::RXX {
                            qubit1,
                            qubit2,
                            theta,
                        })
                        .map_err(|e| format!("Error applying gate: {e}"))?;
                }
                ParamGate::Ryy(q1, q2, name) => {
                    let theta = resolve(name)?;
                    let qubit1 = self.checked_qubit(*q1)?;
                    let qubit2 = self.checked_qubit(*q2)?;
                    circuit
                        .apply_gate(quantrs2_core::gate::multi::RYY {
                            qubit1,
                            qubit2,
                            theta,
                        })
                        .map_err(|e| format!("Error applying gate: {e}"))?;
                }
            }
        }

        let simulator = StateVectorSimulator::new();
        let result = circuit
            .run(&simulator)
            .map_err(|e| format!("Error running simulation: {e}"))?;
        Ok(result.amplitudes().to_vec())
    }
}

#[pymethods]
impl PyParametricCircuit {
    #[new]
    #[pyo3(signature = (n_qubits, gradient_method="parameter_shift"))]
    pub fn new(n_qubits: usize, gradient_method: &str) -> PyResult<Self> {
        // Only the parameter-shift rule is implemented today (see
        // `PyCircuitOptimizer::step`); other gradient methods are not yet
        // wired up, so reject them honestly rather than silently ignoring
        // the choice.
        if gradient_method != "parameter_shift" {
            return Err(PyValueError::new_err(format!(
                "Unsupported gradient_method '{gradient_method}': only 'parameter_shift' is implemented"
            )));
        }
        // Fail fast if the underlying simulator cannot back this qubit count,
        // rather than discovering it lazily inside `get_statevector`.
        DynamicCircuit::new(n_qubits).map_err(|e| {
            PyValueError::new_err(format!("Cannot build a {n_qubits}-qubit circuit: {e}"))
        })?;

        Ok(Self {
            n_qubits,
            parameters: HashMap::new(),
            param_order: Vec::new(),
            gates: Vec::new(),
        })
    }

    /// Add a parameter
    pub fn add_parameter(&mut self, name: String, value: f64) -> PyResult<()> {
        self.register_parameter(&name, value);
        Ok(())
    }

    /// Get parameter value
    pub fn get_parameter(&self, name: &str) -> PyResult<f64> {
        self.parameters
            .get(name)
            .copied()
            .ok_or_else(|| PyValueError::new_err(format!("Parameter {name} not found")))
    }

    /// Set parameter value
    pub fn set_parameter(&mut self, name: &str, value: f64) -> PyResult<()> {
        if self.parameters.contains_key(name) {
            self.parameters.insert(name.to_string(), value);
            Ok(())
        } else {
            Err(PyValueError::new_err(format!("Parameter {name} not found")))
        }
    }

    /// Get all parameters
    pub fn get_parameters(&self, py: Python) -> Py<PyAny> {
        let dict = PyDict::new(py);
        for (name, value) in &self.parameters {
            let _ = dict.set_item(name, value);
        }
        dict.into()
    }

    /// Set parameters from dictionary
    pub fn set_parameters(&mut self, values: &Bound<'_, PyDict>) -> PyResult<()> {
        for (key, value) in values.iter() {
            let param_name: String = key.extract()?;
            let param_value: f64 = value.extract()?;
            self.set_parameter(&param_name, param_value)?;
        }
        Ok(())
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// Compute the real statevector produced by running this circuit's gate
    /// sequence with `params` bound positionally to the parameters in their
    /// registration order (see [`Self::get_parameters`] for the name->value
    /// mapping, or rely on registration order for VQE-style flat vectors).
    pub fn get_statevector(&self, py: Python, params: Vec<f64>) -> PyResult<Py<PyAny>> {
        let amplitudes = self.simulate(&params).map_err(PyValueError::new_err)?;
        let py_array = PyArray1::from_vec(py, amplitudes);
        Ok(py_array.into())
    }

    /// Apply a fixed (non-parametric) CNOT gate.
    pub fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.checked_qubit(control).map_err(PyValueError::new_err)?;
        self.checked_qubit(target).map_err(PyValueError::new_err)?;
        if control == target {
            return Err(PyValueError::new_err(
                "CNOT control and target qubits must differ",
            ));
        }
        self.gates.push(ParamGate::Cnot(control, target));
        Ok(())
    }

    /// Apply a fixed (non-parametric) Hadamard gate.
    pub fn h(&mut self, qubit: usize) -> PyResult<()> {
        self.checked_qubit(qubit).map_err(PyValueError::new_err)?;
        self.gates.push(ParamGate::H(qubit));
        Ok(())
    }

    /// Apply a fixed (non-parametric) Pauli-X gate.
    pub fn x(&mut self, qubit: usize) -> PyResult<()> {
        self.checked_qubit(qubit).map_err(PyValueError::new_err)?;
        self.gates.push(ParamGate::X(qubit));
        Ok(())
    }

    /// Apply a parametrized Rx rotation bound to `param_name`.
    pub fn rx(
        &mut self,
        qubit: usize,
        param_name: &str,
        initial_value: Option<f64>,
    ) -> PyResult<()> {
        self.checked_qubit(qubit).map_err(PyValueError::new_err)?;
        self.register_parameter(param_name, initial_value.unwrap_or(0.0));
        self.gates
            .push(ParamGate::Rx(qubit, param_name.to_string()));
        Ok(())
    }

    /// Apply a parametrized Ry rotation bound to `param_name`.
    pub fn ry(
        &mut self,
        qubit: usize,
        param_name: &str,
        initial_value: Option<f64>,
    ) -> PyResult<()> {
        self.checked_qubit(qubit).map_err(PyValueError::new_err)?;
        self.register_parameter(param_name, initial_value.unwrap_or(0.0));
        self.gates
            .push(ParamGate::Ry(qubit, param_name.to_string()));
        Ok(())
    }

    /// Apply a parametrized Rz rotation bound to `param_name`.
    pub fn rz(
        &mut self,
        qubit: usize,
        param_name: &str,
        initial_value: Option<f64>,
    ) -> PyResult<()> {
        self.checked_qubit(qubit).map_err(PyValueError::new_err)?;
        self.register_parameter(param_name, initial_value.unwrap_or(0.0));
        self.gates
            .push(ParamGate::Rz(qubit, param_name.to_string()));
        Ok(())
    }

    /// Apply a parametrized two-qubit XX rotation bound to `param_name`.
    pub fn rxx(&mut self, qubit1: usize, qubit2: usize, param_name: &str) -> PyResult<()> {
        self.checked_qubit(qubit1).map_err(PyValueError::new_err)?;
        self.checked_qubit(qubit2).map_err(PyValueError::new_err)?;
        if qubit1 == qubit2 {
            return Err(PyValueError::new_err("RXX qubit1 and qubit2 must differ"));
        }
        self.register_parameter(param_name, 0.0);
        self.gates
            .push(ParamGate::Rxx(qubit1, qubit2, param_name.to_string()));
        Ok(())
    }

    /// Apply a parametrized two-qubit YY rotation bound to `param_name`.
    pub fn ryy(&mut self, qubit1: usize, qubit2: usize, param_name: &str) -> PyResult<()> {
        self.checked_qubit(qubit1).map_err(PyValueError::new_err)?;
        self.checked_qubit(qubit2).map_err(PyValueError::new_err)?;
        if qubit1 == qubit2 {
            return Err(PyValueError::new_err("RYY qubit1 and qubit2 must differ"));
        }
        self.register_parameter(param_name, 0.0);
        self.gates
            .push(ParamGate::Ryy(qubit1, qubit2, param_name.to_string()));
        Ok(())
    }
}

/// Gradient-descent-with-momentum optimizer for `ParametricCircuit`s.
#[pyclass(name = "CircuitOptimizer")]
pub struct PyCircuitOptimizer {
    learning_rate: f64,
    momentum: f64,
    /// Momentum accumulator, one entry per optimized parameter; (re)sized on
    /// the first call to `step` for a given parameter vector length.
    velocity: Vec<f64>,
}

#[pymethods]
impl PyCircuitOptimizer {
    #[new]
    #[pyo3(signature = (learning_rate=0.01, momentum=0.0))]
    pub const fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: Vec::new(),
        }
    }

    /// Perform one gradient-descent-with-momentum optimization step.
    ///
    /// Computes the exact gradient of `<psi(params)|observable|psi(params)>`
    /// with respect to every registered parameter of `circuit` using the
    /// parameter-shift rule (shift = pi/2), which is exact for every gate
    /// `ParametricCircuit` can build (Rx/Ry/Rz/RXX/RYY all have generators
    /// whose eigenvalues are +-1), then applies a momentum update and
    /// returns the new parameter vector.
    #[pyo3(text_signature = "(circuit, observable, params, /)")]
    fn step(
        &mut self,
        circuit: &PyParametricCircuit,
        observable: PyReadonlyArray2<'_, Complex64>,
        params: Vec<f64>,
    ) -> PyResult<Vec<f64>> {
        let obs = observable.as_array().to_owned();
        parameter_shift_momentum_step(
            circuit,
            &obs,
            &params,
            self.learning_rate,
            self.momentum,
            &mut self.velocity,
        )
        .map_err(PyValueError::new_err)
    }
}

/// Pure-Rust core of [`PyCircuitOptimizer::step`]: parameter-shift gradient
/// of `<observable>` at `params`, followed by a momentum-based update.
/// Factored out from the `#[pymethods]` wrapper (`Result<_, String>` instead
/// of `PyResult`) so it is directly unit-testable without a Python
/// interpreter -- see the note on [`PyParametricCircuit::checked_qubit`].
fn parameter_shift_momentum_step(
    circuit: &PyParametricCircuit,
    observable: &Array2<Complex64>,
    params: &[f64],
    learning_rate: f64,
    momentum: f64,
    velocity: &mut Vec<f64>,
) -> Result<Vec<f64>, String> {
    let dim = 1usize << circuit.n_qubits;
    if observable.nrows() != dim || observable.ncols() != dim {
        return Err(format!(
            "Observable must be a {dim}x{dim} matrix for a {}-qubit circuit",
            circuit.n_qubits
        ));
    }
    if params.len() != circuit.param_order.len() {
        return Err(format!(
            "Expected {} parameter value(s) matching `circuit`'s registered parameters, got {}",
            circuit.param_order.len(),
            params.len()
        ));
    }

    let expectation = |p: &[f64]| -> Result<f64, String> {
        let amps = circuit.simulate(p)?;
        let mut acc = Complex64::new(0.0, 0.0);
        for i in 0..dim {
            let mut row_sum = Complex64::new(0.0, 0.0);
            for j in 0..dim {
                row_sum += observable[[i, j]] * amps[j];
            }
            acc += amps[i].conj() * row_sum;
        }
        Ok(acc.re)
    };

    let shift = std::f64::consts::FRAC_PI_2;
    let mut gradient = vec![0.0_f64; params.len()];
    for (i, grad_i) in gradient.iter_mut().enumerate() {
        let mut plus = params.to_vec();
        plus[i] += shift;
        let mut minus = params.to_vec();
        minus[i] -= shift;
        let e_plus = expectation(&plus)?;
        let e_minus = expectation(&minus)?;
        *grad_i = 0.5 * (e_plus - e_minus);
    }

    if velocity.len() != params.len() {
        *velocity = vec![0.0; params.len()];
    }
    let mut updated = params.to_vec();
    for i in 0..updated.len() {
        velocity[i] = momentum.mul_add(velocity[i], -learning_rate * gradient[i]);
        updated[i] += velocity[i];
    }
    Ok(updated)
}

/// Register the parametric module
pub fn register_parametric_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(m.py(), "parametric")?;

    submodule.add_class::<PyParametricCircuit>()?;
    submodule.add_class::<PyCircuitOptimizer>()?;

    m.add_submodule(&submodule)?;
    Ok(())
}

// Pure-Rust regression tests. These construct `PyParametricCircuit` via a
// direct struct literal and call only the plain (`Result<_, String>`)
// methods, deliberately never invoking any `#[pymethods]`-decorated function
// (`new`, `h`, `x`, `cnot`, `rx`, `ry`, `rz`, `get_statevector`, `step`, ...):
// this crate builds `pyo3` with the `extension-module` feature, so those
// functions' `PyErr`-construction paths reference CPython C-API symbols
// (`PyErr_Fetch`, `PyType_Type`, ...) that a standalone test binary cannot
// resolve, even along a branch the test never actually takes (see
// `Cargo.toml`'s `[lib] test = false` note, and `scirs2_bindings.rs`'s test
// module, which uses the same pure-core-vs-pymethods split for this reason).
#[cfg(test)]
mod tests {
    use super::*;

    fn build_bell_circuit() -> PyParametricCircuit {
        PyParametricCircuit {
            n_qubits: 2,
            parameters: HashMap::from([("theta".to_string(), 0.0)]),
            param_order: vec!["theta".to_string()],
            gates: vec![
                ParamGate::H(0),
                ParamGate::Rx(1, "theta".to_string()),
                ParamGate::Cnot(0, 1),
            ],
        }
    }

    #[test]
    fn fixed_gates_are_really_applied_h_then_cnot_makes_a_bell_state() {
        // H(0); RX(1, theta=0) [identity]; CNOT(0,1) starting from |00> should
        // produce the Bell state (|00> + |11>) / sqrt(2), *not* the |00>
        // ground state that the old stub always returned.
        let circuit = build_bell_circuit();
        let amplitudes = circuit.simulate(&[0.0]).expect("simulation should succeed");

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (amplitudes[0].re - inv_sqrt2).abs() < 1e-9,
            "{amplitudes:?}"
        );
        assert!(amplitudes[1].norm() < 1e-9, "{amplitudes:?}");
        assert!(amplitudes[2].norm() < 1e-9, "{amplitudes:?}");
        assert!(
            (amplitudes[3].re - inv_sqrt2).abs() < 1e-9,
            "{amplitudes:?}"
        );
    }

    #[test]
    fn get_statevector_responds_to_the_supplied_parameter_value() {
        // A lone RX(theta) on qubit 0 of |00>: at theta=pi the qubit should
        // fully flip.
        let circuit = PyParametricCircuit {
            n_qubits: 2,
            parameters: HashMap::from([("theta".to_string(), 0.0)]),
            param_order: vec!["theta".to_string()],
            gates: vec![ParamGate::Rx(0, "theta".to_string())],
        };

        let ground = circuit.simulate(&[0.0]).expect("simulate at theta=0");
        assert!((ground[0].norm() - 1.0).abs() < 1e-9);

        let flipped = circuit
            .simulate(&[std::f64::consts::PI])
            .expect("simulate at theta=pi");
        // RX(pi)|0> = -i|1> on the affected qubit, tensored with |0> on the
        // other qubit -> all amplitude moves off of index 0.
        assert!(ground[0].norm() > flipped[0].norm());
        let total_norm_sq: f64 = flipped.iter().map(scirs2_core::Complex64::norm_sqr).sum();
        assert!((total_norm_sq - 1.0).abs() < 1e-9, "{flipped:?}");
    }

    #[test]
    fn simulate_rejects_a_wrong_length_parameter_vector() {
        let circuit = build_bell_circuit();
        assert!(circuit.simulate(&[]).is_err());
        assert!(circuit.simulate(&[0.0, 0.0]).is_err());
    }

    #[test]
    fn simulate_rejects_an_out_of_range_qubit_index() {
        let circuit = PyParametricCircuit {
            n_qubits: 2,
            parameters: HashMap::new(),
            param_order: Vec::new(),
            gates: vec![ParamGate::H(5)],
        };
        assert!(circuit.simulate(&[]).is_err());
    }

    #[test]
    fn optimizer_step_moves_theta_downhill_on_a_cos_theta_objective() {
        // RY(theta) on qubit 0 of a 2-qubit register. The simulator's
        // amplitude index is little-endian in the qubit index (qubit 0 is
        // the least-significant bit, confirmed empirically: RY(theta) on
        // qubit 0 of |00> lands its amplitude on indices 0 and 1, not 0 and
        // 2), so the observable "Z on qubit 0, I on qubit 1" is
        // diag(1, -1, 1, -1), not diag(1, 1, -1, -1). Its expectation is
        // cos(theta) for this circuit, so gradient descent starting at
        // theta=0.2 (where d/dtheta cos(theta) = -sin(theta) < 0) must
        // increase theta on the first step.
        let circuit = PyParametricCircuit {
            n_qubits: 2,
            parameters: HashMap::from([("theta".to_string(), 0.2)]),
            param_order: vec!["theta".to_string()],
            gates: vec![ParamGate::Ry(0, "theta".to_string())],
        };

        let observable = Array2::from_diag(&scirs2_core::ndarray::array![
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ]);

        let mut velocity = Vec::new();
        let updated =
            parameter_shift_momentum_step(&circuit, &observable, &[0.2], 0.5, 0.0, &mut velocity)
                .expect("step should succeed");

        assert_eq!(updated.len(), 1);
        assert!(
            updated[0] > 0.2,
            "gradient descent on cos(theta) from theta=0.2 should increase theta, got {}",
            updated[0]
        );
        // Analytic gradient: d/dtheta cos(theta) = -sin(theta); at theta=0.2
        // that is about -0.1987, so with lr=0.5 the step should move theta by
        // about +0.0993.
        assert!(
            (updated[0] - (0.2 + 0.5 * 0.2_f64.sin())).abs() < 1e-4,
            "got {}",
            updated[0]
        );
    }

    #[test]
    fn optimizer_step_rejects_a_mismatched_observable_dimension() {
        let circuit = build_bell_circuit();
        let wrong_size_observable = Array2::<Complex64>::eye(2); // circuit has 2 qubits -> needs 4x4
        let mut velocity = Vec::new();
        assert!(parameter_shift_momentum_step(
            &circuit,
            &wrong_size_observable,
            &[0.0],
            0.1,
            0.0,
            &mut velocity
        )
        .is_err());
    }
}
