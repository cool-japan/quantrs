//! Python bindings for quantum gates
//!
//! This module provides comprehensive bindings for all gate operations from the core module,
//! including standard gates, parameterized gates, multi-qubit gates, and custom gate creation.

// Allow option_if_let_else for cleaner None handling in gate operations
#![allow(clippy::option_if_let_else)]

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyComplex, PyDict, PyList, PyTuple};
use quantrs2_core::gate::{multi, single, GateOp};
use quantrs2_core::parametric::{
    Parameter, ParametricCRX, ParametricGate, ParametricPhaseShift, ParametricRotationX,
    ParametricRotationY, ParametricRotationZ, ParametricU,
};
use quantrs2_core::qubit::QubitId;
use scirs2_core::Complex64;
use scirs2_numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use std::collections::HashMap;

/// Base class for all quantum gates
#[pyclass(subclass)]
#[derive(Clone)]
pub struct Gate {
    /// The internal Rust gate
    pub(crate) gate: Box<dyn GateOp>,
}

#[pymethods]
impl Gate {
    /// Get the name of the gate
    #[getter]
    fn name(&self) -> &str {
        self.gate.name()
    }

    /// Get the qubits this gate acts on
    #[getter]
    fn qubits(&self) -> Vec<usize> {
        self.gate.qubits().iter().map(|q| q.id() as usize).collect()
    }

    /// Get the number of qubits this gate acts on
    #[getter]
    fn num_qubits(&self) -> usize {
        self.gate.num_qubits()
    }

    /// Check if this gate is parameterized
    #[getter]
    fn is_parameterized(&self) -> bool {
        self.gate.is_parameterized()
    }

    /// Get the matrix representation of this gate as a numpy array
    fn matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let matrix_vec = self
            .gate
            .matrix()
            .map_err(|e| PyValueError::new_err(format!("Error getting gate matrix: {e}")))?;

        let size = (matrix_vec.len() as f64).sqrt() as usize;
        // Convert to ndarray first, then to PyArray
        let arr = scirs2_core::ndarray::Array2::from_shape_vec((size, size), matrix_vec)
            .map_err(|e| PyValueError::new_err(format!("Invalid matrix shape: {e}")))?;

        // scirs2-numpy handles scirs2_core::Complex64 natively
        Ok(arr.into_pyarray(py))
    }

    /// Get the matrix representation as a list of complex numbers
    fn matrix_list(&self, py: Python) -> PyResult<PyObject> {
        let matrix_vec = self
            .gate
            .matrix()
            .map_err(|e| PyValueError::new_err(format!("Error getting gate matrix: {e}")))?;

        let result = PyList::empty(py);
        for amp in matrix_vec {
            let complex = PyComplex::from_doubles(py, amp.re, amp.im);
            result.append(complex)?;
        }
        Ok(result.into())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Gate(name='{}', qubits={:?}, parameterized={})",
            self.name(),
            self.qubits(),
            self.is_parameterized()
        )
    }

    /// Check equality with another gate
    fn __eq__(&self, other: &Self) -> bool {
        self.gate.name() == other.gate.name() && self.gate.qubits() == other.gate.qubits()
    }
}

/// Single-qubit Hadamard gate
#[pyclass(extends=Gate)]
struct HadamardGate;

#[pymethods]
impl HadamardGate {
    #[new]
    fn new(qubit: usize) -> (Self, Gate) {
        let gate = Box::new(single::Hadamard {
            target: QubitId::new(qubit as u32),
        });
        (Self, Gate { gate })
    }
}

/// Single-qubit Pauli-X (NOT) gate
#[pyclass(extends=Gate)]
struct PauliXGate;

#[pymethods]
impl PauliXGate {
    #[new]
    fn new(qubit: usize) -> (Self, Gate) {
        let gate = Box::new(single::PauliX {
            target: QubitId::new(qubit as u32),
        });
        (Self, Gate { gate })
    }
}

/// Single-qubit Pauli-Y gate
#[pyclass(extends=Gate)]
struct PauliYGate;

#[pymethods]
impl PauliYGate {
    #[new]
    fn new(qubit: usize) -> (Self, Gate) {
        let gate = Box::new(single::PauliY {
            target: QubitId::new(qubit as u32),
        });
        (Self, Gate { gate })
    }
}

/// Single-qubit Pauli-Z gate
#[pyclass(extends=Gate)]
struct PauliZGate;

#[pymethods]
impl PauliZGate {
    #[new]
    fn new(qubit: usize) -> (Self, Gate) {
        let gate = Box::new(single::PauliZ {
            target: QubitId::new(qubit as u32),
        });
        (Self, Gate { gate })
    }
}

/// Single-qubit S (phase) gate
#[pyclass(extends=Gate)]
struct SGate;

#[pymethods]
impl SGate {
    #[new]
    fn new(qubit: usize) -> (Self, Gate) {
        let gate = Box::new(single::Phase {
            target: QubitId::new(qubit as u32),
        });
        (Self, Gate { gate })
    }
}

/// Single-qubit S-dagger gate
#[pyclass(extends=Gate)]
struct SDaggerGate;

#[pymethods]
impl SDaggerGate {
    #[new]
    fn new(qubit: usize) -> (Self, Gate) {
        let gate = Box::new(single::PhaseDagger {
            target: QubitId::new(qubit as u32),
        });
        (Self, Gate { gate })
    }
}

/// Single-qubit T gate
#[pyclass(extends=Gate)]
struct TGate;

#[pymethods]
impl TGate {
    #[new]
    fn new(qubit: usize) -> (Self, Gate) {
        let gate = Box::new(single::T {
            target: QubitId::new(qubit as u32),
        });
        (Self, Gate { gate })
    }
}

/// Single-qubit T-dagger gate
#[pyclass(extends=Gate)]
struct TDaggerGate;

#[pymethods]
impl TDaggerGate {
    #[new]
    fn new(qubit: usize) -> (Self, Gate) {
        let gate = Box::new(single::TDagger {
            target: QubitId::new(qubit as u32),
        });
        (Self, Gate { gate })
    }
}

/// Single-qubit square root of X gate
#[pyclass(extends=Gate)]
struct SqrtXGate;

#[pymethods]
impl SqrtXGate {
    #[new]
    fn new(qubit: usize) -> (Self, Gate) {
        let gate = Box::new(single::SqrtX {
            target: QubitId::new(qubit as u32),
        });
        (Self, Gate { gate })
    }
}

/// Single-qubit square root of X dagger gate
#[pyclass(extends=Gate)]
struct SqrtXDaggerGate;

#[pymethods]
impl SqrtXDaggerGate {
    #[new]
    fn new(qubit: usize) -> (Self, Gate) {
        let gate = Box::new(single::SqrtXDagger {
            target: QubitId::new(qubit as u32),
        });
        (Self, Gate { gate })
    }
}

/// Rotation around X-axis gate
#[pyclass(extends=Gate)]
struct RXGate;

#[pymethods]
impl RXGate {
    #[new]
    fn new(qubit: usize, theta: f64) -> (Self, Gate) {
        let gate = Box::new(single::RotationX {
            target: QubitId::new(qubit as u32),
            theta,
        });
        (Self, Gate { gate })
    }
}

/// Rotation around Y-axis gate
#[pyclass(extends=Gate)]
struct RYGate;

#[pymethods]
impl RYGate {
    #[new]
    fn new(qubit: usize, theta: f64) -> (Self, Gate) {
        let gate = Box::new(single::RotationY {
            target: QubitId::new(qubit as u32),
            theta,
        });
        (Self, Gate { gate })
    }
}

/// Rotation around Z-axis gate
#[pyclass(extends=Gate)]
struct RZGate;

#[pymethods]
impl RZGate {
    #[new]
    fn new(qubit: usize, theta: f64) -> (Self, Gate) {
        let gate = Box::new(single::RotationZ {
            target: QubitId::new(qubit as u32),
            theta,
        });
        (Self, Gate { gate })
    }
}

/// Two-qubit CNOT gate
#[pyclass(extends=Gate)]
struct CNOTGate;

#[pymethods]
impl CNOTGate {
    #[new]
    fn new(control: usize, target: usize) -> (Self, Gate) {
        let gate = Box::new(multi::CNOT {
            control: QubitId::new(control as u32),
            target: QubitId::new(target as u32),
        });
        (Self, Gate { gate })
    }
}

/// Two-qubit CY gate
#[pyclass(extends=Gate)]
struct CYGate;

#[pymethods]
impl CYGate {
    #[new]
    fn new(control: usize, target: usize) -> (Self, Gate) {
        let gate = Box::new(multi::CY {
            control: QubitId::new(control as u32),
            target: QubitId::new(target as u32),
        });
        (Self, Gate { gate })
    }
}

/// Two-qubit CZ gate
#[pyclass(extends=Gate)]
struct CZGate;

#[pymethods]
impl CZGate {
    #[new]
    fn new(control: usize, target: usize) -> (Self, Gate) {
        let gate = Box::new(multi::CZ {
            control: QubitId::new(control as u32),
            target: QubitId::new(target as u32),
        });
        (Self, Gate { gate })
    }
}

/// Two-qubit CH gate
#[pyclass(extends=Gate)]
struct CHGate;

#[pymethods]
impl CHGate {
    #[new]
    fn new(control: usize, target: usize) -> (Self, Gate) {
        let gate = Box::new(multi::CH {
            control: QubitId::new(control as u32),
            target: QubitId::new(target as u32),
        });
        (Self, Gate { gate })
    }
}

/// Two-qubit CS gate
#[pyclass(extends=Gate)]
struct CSGate;

#[pymethods]
impl CSGate {
    #[new]
    fn new(control: usize, target: usize) -> (Self, Gate) {
        let gate = Box::new(multi::CS {
            control: QubitId::new(control as u32),
            target: QubitId::new(target as u32),
        });
        (Self, Gate { gate })
    }
}

/// Two-qubit SWAP gate
#[pyclass(extends=Gate)]
struct SWAPGate;

#[pymethods]
impl SWAPGate {
    #[new]
    fn new(qubit1: usize, qubit2: usize) -> (Self, Gate) {
        let gate = Box::new(multi::SWAP {
            qubit1: QubitId::new(qubit1 as u32),
            qubit2: QubitId::new(qubit2 as u32),
        });
        (Self, Gate { gate })
    }
}

/// Controlled rotation around X-axis gate
#[pyclass(extends=Gate)]
struct CRXGate;

#[pymethods]
impl CRXGate {
    #[new]
    fn new(control: usize, target: usize, theta: f64) -> (Self, Gate) {
        let gate = Box::new(multi::CRX {
            control: QubitId::new(control as u32),
            target: QubitId::new(target as u32),
            theta,
        });
        (Self, Gate { gate })
    }
}

/// Controlled rotation around Y-axis gate
#[pyclass(extends=Gate)]
struct CRYGate;

#[pymethods]
impl CRYGate {
    #[new]
    fn new(control: usize, target: usize, theta: f64) -> (Self, Gate) {
        let gate = Box::new(multi::CRY {
            control: QubitId::new(control as u32),
            target: QubitId::new(target as u32),
            theta,
        });
        (Self, Gate { gate })
    }
}

/// Controlled rotation around Z-axis gate
#[pyclass(extends=Gate)]
struct CRZGate;

#[pymethods]
impl CRZGate {
    #[new]
    fn new(control: usize, target: usize, theta: f64) -> (Self, Gate) {
        let gate = Box::new(multi::CRZ {
            control: QubitId::new(control as u32),
            target: QubitId::new(target as u32),
            theta,
        });
        (Self, Gate { gate })
    }
}

/// Three-qubit Toffoli (CCNOT) gate
#[pyclass(extends=Gate)]
struct ToffoliGate;

#[pymethods]
impl ToffoliGate {
    #[new]
    fn new(control1: usize, control2: usize, target: usize) -> (Self, Gate) {
        let gate = Box::new(multi::Toffoli {
            control1: QubitId::new(control1 as u32),
            control2: QubitId::new(control2 as u32),
            target: QubitId::new(target as u32),
        });
        (Self, Gate { gate })
    }
}

/// Three-qubit Fredkin (CSWAP) gate
#[pyclass(extends=Gate)]
struct FredkinGate;

#[pymethods]
impl FredkinGate {
    #[new]
    fn new(control: usize, target1: usize, target2: usize) -> (Self, Gate) {
        let gate = Box::new(multi::Fredkin {
            control: QubitId::new(control as u32),
            target1: QubitId::new(target1 as u32),
            target2: QubitId::new(target2 as u32),
        });
        (Self, Gate { gate })
    }
}

/// Parametric gate parameter
#[pyclass]
#[derive(Clone)]
struct GateParameter {
    /// The internal parameter
    param: Parameter,
}

#[pymethods]
impl GateParameter {
    /// Create a constant parameter
    #[staticmethod]
    const fn constant(value: f64) -> Self {
        Self {
            param: Parameter::constant(value),
        }
    }

    /// Create a symbolic parameter
    #[staticmethod]
    fn symbol(name: &str) -> Self {
        Self {
            param: Parameter::symbol(name),
        }
    }

    /// Create a symbolic parameter with value
    #[staticmethod]
    fn symbol_with_value(name: &str, value: f64) -> Self {
        Self {
            param: Parameter::symbol_with_value(name, value),
        }
    }

    /// Get the value if available
    #[getter]
    fn value(&self) -> Option<f64> {
        self.param.value()
    }

    /// Check if parameter has a value
    #[getter]
    const fn has_value(&self) -> bool {
        self.param.has_value()
    }

    fn __repr__(&self) -> String {
        match &self.param {
            Parameter::Constant(v) => format!("GateParameter(constant={v})"),
            Parameter::ComplexConstant(v) => format!("GateParameter(complex_constant={v})"),
            Parameter::Symbol(s) => match s.value {
                Some(v) => format!("GateParameter(symbol='{}', value={})", s.name, v),
                None => format!("GateParameter(symbol='{}')", s.name),
            },
            Parameter::Symbolic(expr) => format!("GateParameter(symbolic='{expr}')"),
        }
    }
}

/// Base class for parametric gates
#[pyclass(subclass)]
struct ParametricGateBase {
    /// The internal parametric gate
    gate: Box<dyn ParametricGate>,
}

#[pymethods]
impl ParametricGateBase {
    /// Get the gate parameters
    fn parameters(&self) -> Vec<GateParameter> {
        self.gate
            .parameters()
            .into_iter()
            .map(|p| GateParameter { param: p })
            .collect()
    }

    /// Get parameter names
    fn parameter_names(&self) -> Vec<String> {
        self.gate.parameter_names()
    }

    /// Create a new gate with updated parameters
    fn with_parameters(&self, params: Vec<GateParameter>) -> PyResult<Self> {
        let rust_params: Vec<Parameter> = params.into_iter().map(|p| p.param).collect();
        let new_gate = self
            .gate
            .with_parameters(&rust_params)
            .map_err(|e| PyValueError::new_err(format!("Error updating parameters: {e}")))?;
        Ok(Self { gate: new_gate })
    }

    /// Assign values to symbolic parameters
    fn assign(&self, values: HashMap<String, f64>) -> PyResult<Self> {
        let values_vec: Vec<(String, f64)> = values.into_iter().collect();
        let new_gate = self
            .gate
            .assign(&values_vec)
            .map_err(|e| PyValueError::new_err(format!("Error assigning parameters: {e}")))?;
        Ok(Self { gate: new_gate })
    }

    /// Bind values to all parameters
    fn bind(&self, values: HashMap<String, f64>) -> PyResult<Self> {
        let values_vec: Vec<(String, f64)> = values.into_iter().collect();
        let new_gate = self
            .gate
            .bind(&values_vec)
            .map_err(|e| PyValueError::new_err(format!("Error binding parameters: {e}")))?;
        Ok(Self { gate: new_gate })
    }
}

/// Parametric rotation around X-axis
#[pyclass(extends=ParametricGateBase)]
struct ParametricRX;

#[pymethods]
impl ParametricRX {
    #[new]
    #[pyo3(signature = (qubit, theta=None))]
    fn new(qubit: usize, theta: Option<&Bound<'_, PyAny>>) -> PyResult<(Self, ParametricGateBase)> {
        let target = QubitId::new(qubit as u32);

        let gate: Box<dyn ParametricGate> = if let Some(theta_val) = theta {
            if let Ok(val) = theta_val.extract::<f64>() {
                Box::new(ParametricRotationX::new(target, val))
            } else if let Ok(name) = theta_val.extract::<String>() {
                Box::new(ParametricRotationX::new_symbolic(target, &name))
            } else {
                return Err(PyTypeError::new_err(
                    "theta must be a float or string (for symbolic parameter)",
                ));
            }
        } else {
            Box::new(ParametricRotationX::new_symbolic(target, "theta"))
        };

        Ok((Self, ParametricGateBase { gate }))
    }
}

/// Parametric rotation around Y-axis
#[pyclass(extends=ParametricGateBase)]
struct ParametricRY;

#[pymethods]
impl ParametricRY {
    #[new]
    #[pyo3(signature = (qubit, theta=None))]
    fn new(qubit: usize, theta: Option<&Bound<'_, PyAny>>) -> PyResult<(Self, ParametricGateBase)> {
        let target = QubitId::new(qubit as u32);

        let gate: Box<dyn ParametricGate> = if let Some(theta_val) = theta {
            if let Ok(val) = theta_val.extract::<f64>() {
                Box::new(ParametricRotationY::new(target, val))
            } else if let Ok(name) = theta_val.extract::<String>() {
                Box::new(ParametricRotationY::new_symbolic(target, &name))
            } else {
                return Err(PyTypeError::new_err(
                    "theta must be a float or string (for symbolic parameter)",
                ));
            }
        } else {
            Box::new(ParametricRotationY::new_symbolic(target, "theta"))
        };

        Ok((Self, ParametricGateBase { gate }))
    }
}

/// Parametric rotation around Z-axis
#[pyclass(extends=ParametricGateBase)]
struct ParametricRZ;

#[pymethods]
impl ParametricRZ {
    #[new]
    #[pyo3(signature = (qubit, theta=None))]
    fn new(qubit: usize, theta: Option<&Bound<'_, PyAny>>) -> PyResult<(Self, ParametricGateBase)> {
        let target = QubitId::new(qubit as u32);

        let gate: Box<dyn ParametricGate> = if let Some(theta_val) = theta {
            if let Ok(val) = theta_val.extract::<f64>() {
                Box::new(ParametricRotationZ::new(target, val))
            } else if let Ok(name) = theta_val.extract::<String>() {
                Box::new(ParametricRotationZ::new_symbolic(target, &name))
            } else {
                return Err(PyTypeError::new_err(
                    "theta must be a float or string (for symbolic parameter)",
                ));
            }
        } else {
            Box::new(ParametricRotationZ::new_symbolic(target, "theta"))
        };

        Ok((Self, ParametricGateBase { gate }))
    }
}

/// Parametric U gate (general single-qubit gate)
#[pyclass(extends=ParametricGateBase)]
struct ParametricUGate;

#[pymethods]
impl ParametricUGate {
    #[new]
    fn new(
        qubit: usize,
        theta: Option<&Bound<'_, PyAny>>,
        phi: Option<&Bound<'_, PyAny>>,
        lambda: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<(Self, ParametricGateBase)> {
        let target = QubitId::new(qubit as u32);

        // Helper to extract parameter
        let extract_param =
            |param: Option<&Bound<'_, PyAny>>, default_name: &str| -> PyResult<Parameter> {
                if let Some(p) = param {
                    if let Ok(val) = p.extract::<f64>() {
                        Ok(Parameter::constant(val))
                    } else if let Ok(name) = p.extract::<String>() {
                        Ok(Parameter::symbol(&name))
                    } else {
                        Err(PyTypeError::new_err("Parameter must be a float or string"))
                    }
                } else {
                    Ok(Parameter::symbol(default_name))
                }
            };

        let theta_param = extract_param(theta, "theta")?;
        let phi_param = extract_param(phi, "phi")?;
        let lambda_param = extract_param(lambda, "lambda")?;

        let gate = match (theta_param, phi_param, lambda_param) {
            (Parameter::Constant(t), Parameter::Constant(p), Parameter::Constant(l)) => {
                Box::new(ParametricU::new(target, t, p, l)) as Box<dyn ParametricGate>
            }
            _ => {
                // For now, create symbolic version
                Box::new(ParametricU::new_symbolic(target, "theta", "phi", "lambda"))
                    as Box<dyn ParametricGate>
            }
        };

        Ok((Self, ParametricGateBase { gate }))
    }
}

/// Custom gate from matrix
#[pyclass(extends=Gate)]
struct CustomGate {
    #[pyo3(get)]
    name: String,
    qubits: Vec<usize>,
    matrix: Vec<Complex64>,
}

#[pymethods]
impl CustomGate {
    #[new]
    fn new(
        name: String,
        qubits: Vec<usize>,
        matrix: PyReadonlyArray2<Complex64>,
    ) -> PyResult<(Self, Gate)> {
        // Convert Complex64 to scirs2_core::Complex64
        let matrix_slice = matrix.as_slice()?;
        let matrix_vec: Vec<Complex64> = matrix_slice
            .iter()
            .map(|c| Complex64::new(c.re, c.im))
            .collect();
        let expected_size = 1 << qubits.len();

        if matrix.shape()[0] != expected_size || matrix.shape()[1] != expected_size {
            return Err(PyValueError::new_err(format!(
                "Matrix size {}x{} doesn't match expected size {}x{} for {} qubits",
                matrix.shape()[0],
                matrix.shape()[1],
                expected_size,
                expected_size,
                qubits.len()
            )));
        }

        // Create a custom gate struct that implements GateOp
        #[derive(Debug)]
        struct CustomGateImpl {
            name: String,
            qubits: Vec<QubitId>,
            matrix: Vec<Complex64>,
        }

        impl GateOp for CustomGateImpl {
            fn name(&self) -> &'static str {
                Box::leak(self.name.clone().into_boxed_str())
            }

            fn qubits(&self) -> Vec<QubitId> {
                self.qubits.clone()
            }

            fn matrix(&self) -> quantrs2_core::error::QuantRS2Result<Vec<Complex64>> {
                Ok(self.matrix.clone())
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn clone_gate(&self) -> Box<dyn GateOp> {
                Box::new(Self {
                    name: self.name.clone(),
                    qubits: self.qubits.clone(),
                    matrix: self.matrix.clone(),
                })
            }
        }

        let gate_impl = CustomGateImpl {
            name: name.clone(),
            qubits: qubits.iter().map(|&q| QubitId::new(q as u32)).collect(),
            matrix: matrix_vec.clone(),
        };

        let custom = Self {
            name,
            qubits,
            matrix: matrix_vec,
        };

        Ok((
            custom,
            Gate {
                gate: Box::new(gate_impl),
            },
        ))
    }
}

/// Module function to create standard gates
#[pyfunction]
fn hadamard(qubit: usize) -> PyResult<Py<HadamardGate>> {
    Python::with_gil(|py| Py::new(py, HadamardGate::new(qubit)))
}

#[pyfunction]
fn pauli_x(qubit: usize) -> PyResult<Py<PauliXGate>> {
    Python::with_gil(|py| Py::new(py, PauliXGate::new(qubit)))
}

#[pyfunction]
fn pauli_y(qubit: usize) -> PyResult<Py<PauliYGate>> {
    Python::with_gil(|py| Py::new(py, PauliYGate::new(qubit)))
}

#[pyfunction]
fn pauli_z(qubit: usize) -> PyResult<Py<PauliZGate>> {
    Python::with_gil(|py| Py::new(py, PauliZGate::new(qubit)))
}

#[pyfunction]
fn cnot(control: usize, target: usize) -> PyResult<Py<CNOTGate>> {
    Python::with_gil(|py| Py::new(py, CNOTGate::new(control, target)))
}

#[pyfunction]
fn rx(qubit: usize, theta: f64) -> PyResult<Py<RXGate>> {
    Python::with_gil(|py| Py::new(py, RXGate::new(qubit, theta)))
}

#[pyfunction]
fn ry(qubit: usize, theta: f64) -> PyResult<Py<RYGate>> {
    Python::with_gil(|py| Py::new(py, RYGate::new(qubit, theta)))
}

#[pyfunction]
fn rz(qubit: usize, theta: f64) -> PyResult<Py<RZGate>> {
    Python::with_gil(|py| Py::new(py, RZGate::new(qubit, theta)))
}

/// Register the gates module with Python
pub fn register_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let gates_module = PyModule::new(parent_module.py(), "gates")?;

    // Register base classes
    gates_module.add_class::<Gate>()?;
    gates_module.add_class::<GateParameter>()?;
    gates_module.add_class::<ParametricGateBase>()?;

    // Register single-qubit gates
    gates_module.add_class::<HadamardGate>()?;
    gates_module.add_class::<PauliXGate>()?;
    gates_module.add_class::<PauliYGate>()?;
    gates_module.add_class::<PauliZGate>()?;
    gates_module.add_class::<SGate>()?;
    gates_module.add_class::<SDaggerGate>()?;
    gates_module.add_class::<TGate>()?;
    gates_module.add_class::<TDaggerGate>()?;
    gates_module.add_class::<SqrtXGate>()?;
    gates_module.add_class::<SqrtXDaggerGate>()?;
    gates_module.add_class::<RXGate>()?;
    gates_module.add_class::<RYGate>()?;
    gates_module.add_class::<RZGate>()?;

    // Register two-qubit gates
    gates_module.add_class::<CNOTGate>()?;
    gates_module.add_class::<CYGate>()?;
    gates_module.add_class::<CZGate>()?;
    gates_module.add_class::<CHGate>()?;
    gates_module.add_class::<CSGate>()?;
    gates_module.add_class::<SWAPGate>()?;
    gates_module.add_class::<CRXGate>()?;
    gates_module.add_class::<CRYGate>()?;
    gates_module.add_class::<CRZGate>()?;

    // Register three-qubit gates
    gates_module.add_class::<ToffoliGate>()?;
    gates_module.add_class::<FredkinGate>()?;

    // Register parametric gates
    gates_module.add_class::<ParametricRX>()?;
    gates_module.add_class::<ParametricRY>()?;
    gates_module.add_class::<ParametricRZ>()?;
    gates_module.add_class::<ParametricUGate>()?;

    // Register custom gate
    gates_module.add_class::<CustomGate>()?;

    // Register convenience functions
    gates_module.add_function(wrap_pyfunction!(hadamard, &gates_module)?)?;
    gates_module.add_function(wrap_pyfunction!(pauli_x, &gates_module)?)?;
    gates_module.add_function(wrap_pyfunction!(pauli_y, &gates_module)?)?;
    gates_module.add_function(wrap_pyfunction!(pauli_z, &gates_module)?)?;
    gates_module.add_function(wrap_pyfunction!(cnot, &gates_module)?)?;
    gates_module.add_function(wrap_pyfunction!(rx, &gates_module)?)?;
    gates_module.add_function(wrap_pyfunction!(ry, &gates_module)?)?;
    gates_module.add_function(wrap_pyfunction!(rz, &gates_module)?)?;

    parent_module.add_submodule(&gates_module)?;
    Ok(())
}
