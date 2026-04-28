//! Python bindings for quantum networking protocols.
//!
//! Exposes BB84 QKD, E91 QKD, and quantum teleportation to Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use quantrs2_core::networking::{
    Bb84Protocol, E91Protocol, EntanglementSwapping, TeleportationProtocol,
};
use scirs2_core::Complex64;

// ---------------------------------------------------------------------------
// BB84 QKD
// ---------------------------------------------------------------------------

/// BB84 Quantum Key Distribution protocol (Python binding).
#[pyclass(name = "BB84Protocol")]
pub struct PyBb84Protocol {
    inner: Bb84Protocol,
}

#[pymethods]
impl PyBb84Protocol {
    /// Create a new BB84 protocol instance.
    ///
    /// Parameters
    /// ----------
    /// n_bits : int
    ///     Number of raw qubits Alice transmits.
    /// error_rate : float
    ///     Depolarizing noise probability per qubit [0, 1].
    /// eavesdrop_rate : float
    ///     Fraction of qubits Eve intercepts [0, 1].
    /// seed : int
    ///     Random seed for reproducibility.
    #[new]
    #[pyo3(signature = (n_bits, error_rate=0.0, eavesdrop_rate=0.0, seed=42))]
    fn new(n_bits: usize, error_rate: f64, eavesdrop_rate: f64, seed: u64) -> PyResult<Self> {
        if n_bits == 0 {
            return Err(PyValueError::new_err("n_bits must be > 0"));
        }
        Ok(Self {
            inner: Bb84Protocol::new(n_bits, error_rate, eavesdrop_rate, seed),
        })
    }

    /// Execute the BB84 protocol.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``raw_bits`` (int), ``sifted_key`` (list[bool]),
    ///     ``qber`` (float), ``secret_key`` (list[bool]),
    ///     ``detected_eavesdrop`` (bool).
    fn run(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .run()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("raw_bits", result.raw_bits)?;
        d.set_item("sifted_key", result.sifted_key)?;
        d.set_item("qber", result.qber)?;
        d.set_item("secret_key", result.secret_key)?;
        d.set_item("detected_eavesdrop", result.detected_eavesdrop)?;
        Ok(d.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "BB84Protocol(n_bits={}, error_rate={:.3}, eavesdrop_rate={:.3})",
            self.inner.n_bits, self.inner.error_rate, self.inner.eavesdrop_rate
        )
    }
}

// ---------------------------------------------------------------------------
// E91 QKD
// ---------------------------------------------------------------------------

/// E91 entanglement-based Quantum Key Distribution protocol (Python binding).
#[pyclass(name = "E91Protocol")]
pub struct PyE91Protocol {
    inner: E91Protocol,
}

#[pymethods]
impl PyE91Protocol {
    /// Create a new E91 protocol instance.
    ///
    /// Parameters
    /// ----------
    /// n_pairs : int
    ///     Number of entangled Bell pairs to generate.
    /// noise : float
    ///     Depolarizing noise probability per qubit [0, 1].
    /// seed : int
    ///     Random seed for reproducibility.
    #[new]
    #[pyo3(signature = (n_pairs, noise=0.0, seed=42))]
    fn new(n_pairs: usize, noise: f64, seed: u64) -> PyResult<Self> {
        if n_pairs == 0 {
            return Err(PyValueError::new_err("n_pairs must be > 0"));
        }
        Ok(Self {
            inner: E91Protocol::new(n_pairs, noise, seed),
        })
    }

    /// Execute the E91 protocol.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``key`` (list[bool]), ``chsh_value`` (float),
    ///     ``passed_bell_test`` (bool), ``key_rate`` (float).
    fn run(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let result = self
            .inner
            .run()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("key", result.key)?;
        d.set_item("chsh_value", result.chsh_value)?;
        d.set_item("passed_bell_test", result.passed_bell_test)?;
        d.set_item("key_rate", result.key_rate)?;
        Ok(d.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "E91Protocol(n_pairs={}, noise={:.3})",
            self.inner.n_pairs, self.inner.noise
        )
    }
}

// ---------------------------------------------------------------------------
// Quantum Teleportation
// ---------------------------------------------------------------------------

/// Quantum teleportation protocol (Python binding).
#[pyclass(name = "TeleportationProtocol")]
pub struct PyTeleportationProtocol {
    noise: f64,
    rng_seed: u64,
}

#[pymethods]
impl PyTeleportationProtocol {
    /// Create a new teleportation protocol.
    ///
    /// Parameters
    /// ----------
    /// noise : float
    ///     Depolarizing noise per qubit of the resource Bell pair [0, 1].
    /// seed : int
    ///     Random seed for reproducibility.
    #[new]
    #[pyo3(signature = (noise=0.0, seed=42))]
    const fn new(noise: f64, seed: u64) -> Self {
        Self {
            noise: noise.clamp(0.0, 1.0),
            rng_seed: seed,
        }
    }

    /// Teleport a qubit state.
    ///
    /// Parameters
    /// ----------
    /// alpha_re, alpha_im : float
    ///     Real and imaginary parts of the |0⟩ amplitude.
    /// beta_re, beta_im : float
    ///     Real and imaginary parts of the |1⟩ amplitude.
    ///
    /// The state need not be pre-normalised; it is normalised internally.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``fidelity`` (float), ``correction_bits`` (list[bool]).
    #[pyo3(signature = (alpha_re, alpha_im=0.0, beta_re=0.0, beta_im=0.0))]
    fn teleport(
        &self,
        py: Python<'_>,
        alpha_re: f64,
        alpha_im: f64,
        beta_re: f64,
        beta_im: f64,
    ) -> PyResult<Py<PyAny>> {
        let alpha = Complex64::new(alpha_re, alpha_im);
        let beta = Complex64::new(beta_re, beta_im);
        let norm = (alpha.norm_sqr() + beta.norm_sqr()).sqrt();
        if norm < 1e-10 {
            return Err(PyValueError::new_err("State vector must not be zero"));
        }
        let state = [alpha / norm, beta / norm];
        let proto = TeleportationProtocol::new(self.noise, self.rng_seed);
        let result = proto
            .teleport(state)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("fidelity", result.fidelity)?;
        d.set_item(
            "correction_bits",
            vec![result.correction_bits.0, result.correction_bits.1],
        )?;
        Ok(d.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "TeleportationProtocol(noise={:.3}, seed={})",
            self.noise, self.rng_seed
        )
    }
}

// ---------------------------------------------------------------------------
// Entanglement Swapping
// ---------------------------------------------------------------------------

/// Multi-hop entanglement swapping chain (Python binding).
#[pyclass(name = "EntanglementSwapping")]
pub struct PyEntanglementSwapping {
    n_hops: usize,
    noise_per_link: f64,
    rng_seed: u64,
}

#[pymethods]
impl PyEntanglementSwapping {
    /// Create a new entanglement swapping chain.
    ///
    /// Parameters
    /// ----------
    /// n_hops : int
    ///     Number of hops (Bell-pair links). Minimum 1.
    /// noise_per_link : float
    ///     Depolarizing noise per qubit per link [0, 1].
    /// seed : int
    ///     Random seed.
    #[new]
    #[pyo3(signature = (n_hops, noise_per_link=0.0, seed=42))]
    fn new(n_hops: usize, noise_per_link: f64, seed: u64) -> PyResult<Self> {
        if n_hops == 0 {
            return Err(PyValueError::new_err("n_hops must be >= 1"));
        }
        Ok(Self {
            n_hops,
            noise_per_link: noise_per_link.clamp(0.0, 1.0),
            rng_seed: seed,
        })
    }

    /// Run the entanglement swapping chain.
    ///
    /// Returns
    /// -------
    /// dict
    ///     ``end_to_end_fidelity`` (float).
    fn run(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let swapping = EntanglementSwapping::new(self.n_hops, self.noise_per_link, self.rng_seed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let result = swapping
            .run()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("end_to_end_fidelity", result.end_to_end_fidelity)?;
        Ok(d.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "EntanglementSwapping(n_hops={}, noise_per_link={:.3})",
            self.n_hops, self.noise_per_link
        )
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

/// Register the `networking` submodule into the parent `quantrs2` module.
pub fn register_networking_module(py: Python<'_>, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "networking")?;
    m.add_class::<PyBb84Protocol>()?;
    m.add_class::<PyE91Protocol>()?;
    m.add_class::<PyTeleportationProtocol>()?;
    m.add_class::<PyEntanglementSwapping>()?;
    parent.add_submodule(&m)?;
    Ok(())
}
