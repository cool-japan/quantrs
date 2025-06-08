//! Support for custom gate definitions from Python.
//!
//! This module allows users to define custom quantum gates using:
//! - Numpy arrays for unitary matrices
//! - Python functions for parametric gates
//! - Symbolic expressions for gate decompositions

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::{PyValueError, PyTypeError};
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use ndarray::{Array2, ArrayView2};
use num_complex::Complex64;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Gate definition types
#[derive(Clone)]
pub enum GateDefinition {
    Matrix {
        unitary: Array2<Complex64>,
        num_qubits: usize,
    },
    Parametric {
        num_params: usize,
        num_qubits: usize,
        matrix_fn: Arc<dyn Fn(&[f64]) -> Array2<Complex64> + Send + Sync>,
    },
    Decomposition {
        num_qubits: usize,
        gates: Vec<(String, Vec<usize>, Option<Vec<f64>>)>,
    },
    ControlledGate {
        base_gate: String,
        num_controls: usize,
    },
}

/// Custom gate registry
pub struct GateRegistry {
    gates: Arc<Mutex<HashMap<String, GateDefinition>>>,
}

impl GateRegistry {
    fn new() -> Self {
        Self {
            gates: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    fn register(&self, name: String, definition: GateDefinition) -> PyResult<()> {
        let mut gates = self.gates.lock().unwrap();
        if gates.contains_key(&name) {
            return Err(PyValueError::new_err(format!("Gate {} already exists", name)));
        }
        gates.insert(name, definition);
        Ok(())
    }
    
    fn get(&self, name: &str) -> Option<GateDefinition> {
        let gates = self.gates.lock().unwrap();
        gates.get(name).cloned()
    }
    
    fn list_gates(&self) -> Vec<String> {
        let gates = self.gates.lock().unwrap();
        gates.keys().cloned().collect()
    }
}

/// Python wrapper for custom gate definitions
#[pyclass(name = "CustomGate")]
pub struct PyCustomGate {
    name: String,
    definition: GateDefinition,
}

#[pymethods]
impl PyCustomGate {
    /// Create a custom gate from a unitary matrix
    #[staticmethod]
    #[pyo3(signature = (name, matrix))]
    fn from_matrix(name: String, matrix: PyReadonlyArray2<Complex64>) -> PyResult<Self> {
        let arr = matrix.as_array();
        let (rows, cols) = arr.dim();
        
        // Check if matrix is square
        if rows != cols {
            return Err(PyValueError::new_err("Matrix must be square"));
        }
        
        // Check if dimension is power of 2
        if !rows.is_power_of_two() {
            return Err(PyValueError::new_err("Matrix dimension must be a power of 2"));
        }
        
        // Check if matrix is unitary (U†U = I)
        if !is_unitary(&arr, 1e-10) {
            return Err(PyValueError::new_err("Matrix must be unitary"));
        }
        
        let num_qubits = (rows as f64).log2() as usize;
        
        Ok(Self {
            name,
            definition: GateDefinition::Matrix {
                unitary: arr.to_owned(),
                num_qubits,
            },
        })
    }
    
    /// Create a parametric custom gate
    #[staticmethod]
    #[pyo3(signature = (name, num_qubits, num_params, matrix_fn))]
    fn from_function(
        name: String,
        num_qubits: usize,
        num_params: usize,
        matrix_fn: PyObject,
    ) -> PyResult<Self> {
        // Create wrapper that calls Python function
        let matrix_fn_wrapper = Arc::new(move |params: &[f64]| -> Array2<Complex64> {
            Python::with_gil(|py| {
                let params_py = params.iter().copied().collect::<Vec<f64>>().into_pyarray(py);
                let result = matrix_fn.call1(py, (params_py,)).unwrap();
                let matrix: PyReadonlyArray2<Complex64> = result.extract(py).unwrap();
                matrix.as_array().to_owned()
            })
        });
        
        Ok(Self {
            name,
            definition: GateDefinition::Parametric {
                num_params,
                num_qubits,
                matrix_fn: matrix_fn_wrapper,
            },
        })
    }
    
    /// Create a custom gate from decomposition
    #[staticmethod]
    #[pyo3(signature = (name, num_qubits, gates))]
    fn from_decomposition(
        name: String,
        num_qubits: usize,
        gates: &Bound<'_, PyList>,
    ) -> PyResult<Self> {
        let mut decomposition = Vec::new();
        
        for gate in gates {
            let gate_tuple: (String, Vec<usize>, Option<Vec<f64>>) = gate.extract()?;
            decomposition.push(gate_tuple);
        }
        
        Ok(Self {
            name,
            definition: GateDefinition::Decomposition {
                num_qubits,
                gates: decomposition,
            },
        })
    }
    
    /// Create a controlled version of an existing gate
    #[staticmethod]
    #[pyo3(signature = (name, base_gate, num_controls=1))]
    fn controlled(name: String, base_gate: String, num_controls: Option<usize>) -> PyResult<Self> {
        let num_controls = num_controls.unwrap_or(1);
        
        Ok(Self {
            name,
            definition: GateDefinition::ControlledGate {
                base_gate,
                num_controls,
            },
        })
    }
    
    /// Get the number of qubits this gate acts on
    #[getter]
    fn num_qubits(&self) -> usize {
        match &self.definition {
            GateDefinition::Matrix { num_qubits, .. } => *num_qubits,
            GateDefinition::Parametric { num_qubits, .. } => *num_qubits,
            GateDefinition::Decomposition { num_qubits, .. } => *num_qubits,
            GateDefinition::ControlledGate { base_gate, num_controls } => {
                // Would need to look up base gate
                *num_controls + 1
            }
        }
    }
    
    /// Get the number of parameters (0 for non-parametric gates)
    #[getter]
    fn num_params(&self) -> usize {
        match &self.definition {
            GateDefinition::Matrix { .. } => 0,
            GateDefinition::Parametric { num_params, .. } => *num_params,
            GateDefinition::Decomposition { .. } => 0,
            GateDefinition::ControlledGate { .. } => 0,
        }
    }
    
    /// Get the gate name
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the unitary matrix for given parameters
    fn get_matrix<'py>(&self, py: Python<'py>, params: Option<Vec<f64>>) -> PyResult<Py<PyArray2<Complex64>>> {
        match &self.definition {
            GateDefinition::Matrix { unitary, .. } => {
                Ok(unitary.clone().into_pyarray(py).into())
            }
            GateDefinition::Parametric { num_params, matrix_fn, .. } => {
                let params = params.ok_or_else(|| {
                    PyValueError::new_err(format!("Expected {} parameters", num_params))
                })?;
                
                if params.len() != *num_params {
                    return Err(PyValueError::new_err(format!(
                        "Expected {} parameters, got {}",
                        num_params,
                        params.len()
                    )));
                }
                
                let matrix = matrix_fn(&params);
                Ok(matrix.into_pyarray(py).into())
            }
            _ => Err(PyTypeError::new_err("Cannot get matrix for decomposition or controlled gates")),
        }
    }
}

/// Gate builder for creating custom gates with fluent API
#[pyclass(name = "GateBuilder")]
pub struct PyGateBuilder {
    name: Option<String>,
    matrix: Option<Array2<Complex64>>,
    num_qubits: Option<usize>,
    num_params: Option<usize>,
    decomposition: Option<Vec<(String, Vec<usize>, Option<Vec<f64>>)>>,
}

#[pymethods]
impl PyGateBuilder {
    #[new]
    fn new() -> Self {
        Self {
            name: None,
            matrix: None,
            num_qubits: None,
            num_params: None,
            decomposition: None,
        }
    }
    
    /// Set the gate name
    fn with_name(&mut self, name: String) -> PyResult<()> {
        self.name = Some(name);
        Ok(())
    }
    
    /// Set the unitary matrix
    fn with_matrix(&mut self, matrix: PyReadonlyArray2<Complex64>) -> PyResult<()> {
        let arr = matrix.as_array();
        if !is_unitary(&arr, 1e-10) {
            return Err(PyValueError::new_err("Matrix must be unitary"));
        }
        self.matrix = Some(arr.to_owned());
        self.num_qubits = Some((arr.nrows() as f64).log2() as usize);
        Ok(())
    }
    
    /// Add a gate to the decomposition
    fn add_gate(&mut self, gate_name: String, qubits: Vec<usize>, params: Option<Vec<f64>>) -> PyResult<()> {
        if self.decomposition.is_none() {
            self.decomposition = Some(Vec::new());
        }
        if let Some(ref mut decomp) = self.decomposition {
            decomp.push((gate_name, qubits, params));
        }
        Ok(())
    }
    
    /// Build the custom gate
    fn build(&self) -> PyResult<PyCustomGate> {
        let name = self.name.clone().ok_or_else(|| {
            PyValueError::new_err("Gate name not set")
        })?;
        
        if let Some(ref matrix) = self.matrix {
            Ok(PyCustomGate {
                name,
                definition: GateDefinition::Matrix {
                    unitary: matrix.clone(),
                    num_qubits: self.num_qubits.unwrap(),
                },
            })
        } else if let Some(ref decomp) = self.decomposition {
            let num_qubits = self.num_qubits.ok_or_else(|| {
                PyValueError::new_err("Number of qubits not set for decomposition")
            })?;
            
            Ok(PyCustomGate {
                name,
                definition: GateDefinition::Decomposition {
                    num_qubits,
                    gates: decomp.clone(),
                },
            })
        } else {
            Err(PyValueError::new_err("Must set either matrix or decomposition"))
        }
    }
}

/// Global gate registry
#[pyclass(name = "GateRegistry")]
pub struct PyGateRegistry {
    registry: GateRegistry,
}

#[pymethods]
impl PyGateRegistry {
    #[new]
    fn new() -> Self {
        Self {
            registry: GateRegistry::new(),
        }
    }
    
    /// Register a custom gate
    fn register(&self, gate: &PyCustomGate) -> PyResult<()> {
        self.registry.register(gate.name.clone(), gate.definition.clone())
    }
    
    /// Get a registered gate
    fn get(&self, name: &str) -> PyResult<Option<PyCustomGate>> {
        Ok(self.registry.get(name).map(|definition| {
            PyCustomGate {
                name: name.to_string(),
                definition,
            }
        }))
    }
    
    /// List all registered gates
    fn list_gates(&self) -> Vec<String> {
        self.registry.list_gates()
    }
    
    /// Clear all registered gates
    fn clear(&self) {
        let mut gates = self.registry.gates.lock().unwrap();
        gates.clear();
    }
    
    /// Check if a gate is registered
    fn contains(&self, name: &str) -> bool {
        let gates = self.registry.gates.lock().unwrap();
        gates.contains_key(name)
    }
}

// Helper function to check if a matrix is unitary
fn is_unitary(matrix: &ArrayView2<Complex64>, tolerance: f64) -> bool {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return false;
    }
    
    // Compute U†U
    let mut product = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..n {
                sum += matrix[[k, i]].conj() * matrix[[k, j]];
            }
            product[[i, j]] = sum;
        }
    }
    
    // Check if U†U ≈ I
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) };
            if (product[[i, j]] - expected).norm() > tolerance {
                return false;
            }
        }
    }
    
    true
}

/// Common gate templates
#[pyfunction]
fn create_phase_gate(phase: f64) -> PyResult<PyCustomGate> {
    let matrix = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(phase.cos(), phase.sin()),
        ],
    ).unwrap();
    
    Ok(PyCustomGate {
        name: format!("Phase({:.3})", phase),
        definition: GateDefinition::Matrix {
            unitary: matrix,
            num_qubits: 1,
        },
    })
}

/// Create a rotation gate around arbitrary axis
#[pyfunction]
fn create_rotation_gate(axis: Vec<f64>, angle: f64) -> PyResult<PyCustomGate> {
    if axis.len() != 3 {
        return Err(PyValueError::new_err("Axis must be a 3D vector"));
    }
    
    // Normalize axis
    let norm = (axis[0].powi(2) + axis[1].powi(2) + axis[2].powi(2)).sqrt();
    let nx = axis[0] / norm;
    let ny = axis[1] / norm;
    let nz = axis[2] / norm;
    
    // Compute rotation matrix using Rodrigues' formula
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();
    
    let matrix = Array2::from_shape_vec(
        (2, 2),
        vec![
            Complex64::new(cos_half, -sin_half * nz),
            Complex64::new(-sin_half * ny, -sin_half * nx),
            Complex64::new(sin_half * ny, -sin_half * nx),
            Complex64::new(cos_half, sin_half * nz),
        ],
    ).unwrap();
    
    Ok(PyCustomGate {
        name: format!("Rot({:.2},{:.2},{:.2},{:.3})", nx, ny, nz, angle),
        definition: GateDefinition::Matrix {
            unitary: matrix,
            num_qubits: 1,
        },
    })
}

/// Register the custom gates module
pub fn register_custom_gates_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(m.py(), "custom_gates")?;
    
    submodule.add_class::<PyCustomGate>()?;
    submodule.add_class::<PyGateBuilder>()?;
    submodule.add_class::<PyGateRegistry>()?;
    
    submodule.add_function(wrap_pyfunction!(create_phase_gate, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(create_rotation_gate, &submodule)?)?;
    
    m.add_submodule(&submodule)?;
    Ok(())
}