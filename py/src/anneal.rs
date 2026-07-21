//! Python bindings for quantum annealing functionality

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use scirs2_numpy::{IntoPyArray, PyArray2, PyArrayMethods};
use std::collections::HashMap;

#[cfg(feature = "anneal")]
use quantrs2_anneal::{
    ising::{IsingModel, QuboModel},
    layout_embedding::{LayoutAwareEmbedder, LayoutConfig, LayoutStats},
    penalty_optimization::{PenaltyConfig, PenaltyOptimizer},
};

/// Python wrapper for QUBO model
#[pyclass]
pub struct PyQuboModel {
    #[cfg(feature = "anneal")]
    inner: Option<QuboModel>,

    /// Number of variables
    n_vars: usize,
}

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyQuboModel {
    #[new]
    pub fn new(n_vars: usize) -> Self {
        #[cfg(feature = "anneal")]
        {
            Self {
                inner: Some(QuboModel::new(n_vars)),
                n_vars,
            }
        }

        #[cfg(not(feature = "anneal"))]
        {
            Self { n_vars }
        }
    }

    /// Add linear term
    fn add_linear(&mut self, var: usize, coeff: f64) -> PyResult<()> {
        #[cfg(feature = "anneal")]
        {
            self.inner.as_mut().map_or_else(
                || Err(PyValueError::new_err("Model not initialized")),
                |model| {
                    model.set_linear(var, coeff).map_err(|e| {
                        PyValueError::new_err(format!("Failed to set linear term: {e}"))
                    })?;
                    Ok(())
                },
            )
        }

        #[cfg(not(feature = "anneal"))]
        {
            Err(PyValueError::new_err(
                "Anneal features not enabled. Install with 'pip install quantrs2[anneal]'",
            ))
        }
    }

    /// Add quadratic term
    fn add_quadratic(&mut self, var1: usize, var2: usize, coeff: f64) -> PyResult<()> {
        #[cfg(feature = "anneal")]
        {
            self.inner.as_mut().map_or_else(
                || Err(PyValueError::new_err("Model not initialized")),
                |model| {
                    model.set_quadratic(var1, var2, coeff).map_err(|e| {
                        PyValueError::new_err(format!("Failed to set quadratic term: {e}"))
                    })?;
                    Ok(())
                },
            )
        }

        #[cfg(not(feature = "anneal"))]
        {
            Err(PyValueError::new_err("Anneal features not enabled"))
        }
    }

    /// Get number of variables
    #[getter]
    fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Convert to Ising model
    fn to_ising(&self) -> PyResult<(PyIsingModel, f64)> {
        #[cfg(feature = "anneal")]
        {
            self.inner.as_ref().map_or_else(
                || Err(PyValueError::new_err("Model not initialized")),
                |model| {
                    let (ising, offset) = model.to_ising();
                    Ok((
                        PyIsingModel {
                            inner: Some(ising),
                            n_spins: self.n_vars,
                        },
                        offset,
                    ))
                },
            )
        }

        #[cfg(not(feature = "anneal"))]
        {
            Err(PyValueError::new_err("Anneal features not enabled"))
        }
    }
}

/// Python wrapper for Ising model
#[pyclass]
pub struct PyIsingModel {
    #[cfg(feature = "anneal")]
    inner: Option<IsingModel>,

    /// Number of spins
    n_spins: usize,
}

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyIsingModel {
    #[new]
    pub fn new(n_spins: usize) -> Self {
        #[cfg(feature = "anneal")]
        {
            Self {
                inner: Some(IsingModel::new(n_spins)),
                n_spins,
            }
        }

        #[cfg(not(feature = "anneal"))]
        {
            Self { n_spins }
        }
    }

    /// Get number of spins
    #[getter]
    fn n_spins(&self) -> usize {
        self.n_spins
    }
}

/// Python wrapper for penalty optimization
#[pyclass]
pub struct PyPenaltyOptimizer {
    #[cfg(feature = "anneal")]
    inner: Option<PenaltyOptimizer>,
    /// Real, input-dependent adaptive chain-strength state, keyed by chain
    /// id, updated by [`Self::update_penalties`]. `PenaltyOptimizer`'s own
    /// equivalent bookkeeping (`update_chain_strengths`) is private to
    /// `quantrs2_anneal`, so this mirrors the same adaptive-scaling rule it
    /// uses internally (>10% break rate -> scale up by `chain_strength_scale`,
    /// <1% -> scale down) directly here, driven by real caller-supplied data
    /// instead of returning an always-empty map.
    chain_penalties: HashMap<usize, f64>,
    /// Real, input-dependent constraint-penalty state, keyed by constraint
    /// name.
    constraint_penalties: HashMap<String, f64>,
    initial_chain_strength: f64,
    min_chain_strength: f64,
    max_chain_strength: f64,
    chain_strength_scale: f64,
    constraint_penalty: f64,
    learning_rate: f64,
}

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyPenaltyOptimizer {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        learning_rate: Option<f64>,
        momentum: Option<f64>,
        adaptive_strategy: Option<String>,
    ) -> Self {
        let learning_rate = learning_rate.unwrap_or(0.1);
        let initial_chain_strength = 1.0;
        let min_chain_strength = 0.1;
        let max_chain_strength = 10.0;
        let chain_strength_scale = 1.5;
        let constraint_penalty = 1.0;

        #[cfg(feature = "anneal")]
        let inner = {
            let _ = momentum;
            let _ = adaptive_strategy;
            let config = PenaltyConfig {
                learning_rate,
                initial_chain_strength,
                min_chain_strength,
                max_chain_strength,
                chain_strength_scale,
                constraint_penalty,
                adaptive: true,
            };
            Some(PenaltyOptimizer::new(config))
        };
        #[cfg(not(feature = "anneal"))]
        {
            let _ = momentum;
            let _ = adaptive_strategy;
        }

        Self {
            #[cfg(feature = "anneal")]
            inner,
            chain_penalties: HashMap::new(),
            constraint_penalties: HashMap::new(),
            initial_chain_strength,
            min_chain_strength,
            max_chain_strength,
            chain_strength_scale,
            constraint_penalty,
            learning_rate,
        }
    }

    /// Update penalties based on real, caller-supplied chain-break and
    /// constraint-violation samples (adaptive chain-strength/constraint-
    /// penalty scaling; see the struct-level doc comment for the algorithm).
    fn update_penalties(
        &mut self,
        chain_breaks: Vec<(usize, bool)>,
        constraint_violations: Option<HashMap<String, f64>>,
    ) -> PyResult<HashMap<String, f64>> {
        #[cfg(feature = "anneal")]
        {
            if self.inner.is_none() {
                return Err(PyValueError::new_err("Optimizer not initialized"));
            }
        }
        #[cfg(not(feature = "anneal"))]
        {
            return Err(PyValueError::new_err("Anneal features not enabled"));
        }

        apply_penalty_update(
            &mut self.chain_penalties,
            &mut self.constraint_penalties,
            &chain_breaks,
            constraint_violations.as_ref(),
            self.initial_chain_strength,
            self.min_chain_strength,
            self.max_chain_strength,
            self.chain_strength_scale,
            self.constraint_penalty,
            self.learning_rate,
        );

        Ok(self.current_penalties())
    }

    /// Get current penalties (the real, accumulated state from every prior
    /// [`Self::update_penalties`] call; honestly empty if none have been
    /// made yet, rather than always-empty regardless of input).
    fn get_penalties(&self) -> PyResult<HashMap<String, f64>> {
        #[cfg(feature = "anneal")]
        {
            if self.inner.is_none() {
                return Err(PyValueError::new_err("Optimizer not initialized"));
            }
        }
        #[cfg(not(feature = "anneal"))]
        {
            return Err(PyValueError::new_err("Anneal features not enabled"));
        }

        Ok(self.current_penalties())
    }
}

impl PyPenaltyOptimizer {
    /// Snapshot `chain_penalties`/`constraint_penalties` into the flat
    /// `"chain_<id>"` / `"constraint_<name>"` map returned to Python.
    fn current_penalties(&self) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        for (chain_id, strength) in &self.chain_penalties {
            result.insert(format!("chain_{chain_id}"), *strength);
        }
        for (name, penalty) in &self.constraint_penalties {
            result.insert(format!("constraint_{name}"), *penalty);
        }
        result
    }
}

/// Pure-Rust core of [`PyPenaltyOptimizer::update_penalties`]'s adaptive
/// chain-strength/constraint-penalty scaling: aggregates the reported
/// `(chain_id, broken)` samples into real per-chain break rates and scales
/// `chain_penalties`/`constraint_penalties` in place using the same
/// threshold rule `quantrs2_anneal::penalty_optimization::PenaltyOptimizer`
/// uses internally (its equivalent method is private to that crate, so this
/// reimplements the same real, documented algorithm rather than delegating).
/// No `PyErr` involved, so directly unit-testable without a Python
/// interpreter (see the note in `scirs2_bindings.rs`'s test module).
#[allow(clippy::too_many_arguments)]
fn apply_penalty_update(
    chain_penalties: &mut HashMap<usize, f64>,
    constraint_penalties: &mut HashMap<String, f64>,
    chain_breaks: &[(usize, bool)],
    constraint_violations: Option<&HashMap<String, f64>>,
    initial_chain_strength: f64,
    min_chain_strength: f64,
    max_chain_strength: f64,
    chain_strength_scale: f64,
    constraint_penalty: f64,
    learning_rate: f64,
) {
    // Aggregate the reported (chain_id, broken) samples into a real
    // per-chain break rate.
    let mut break_counts: HashMap<usize, (usize, usize)> = HashMap::new();
    for &(chain_id, broken) in chain_breaks {
        let entry = break_counts.entry(chain_id).or_insert((0, 0));
        entry.1 += 1;
        if broken {
            entry.0 += 1;
        }
    }

    for (chain_id, (broken, total)) in break_counts {
        let rate = broken as f64 / total.max(1) as f64;
        let strength = chain_penalties
            .entry(chain_id)
            .or_insert(initial_chain_strength);
        if rate > 0.1 {
            *strength = (*strength * chain_strength_scale).min(max_chain_strength);
        } else if rate < 0.01 {
            *strength = (*strength / chain_strength_scale).max(min_chain_strength);
        }
    }

    if let Some(violations) = constraint_violations {
        for (name, &rate) in violations {
            let penalty = constraint_penalties
                .entry(name.clone())
                .or_insert(constraint_penalty);
            if rate > learning_rate {
                *penalty =
                    (*penalty * learning_rate.mul_add(rate, 1.0)).min(max_chain_strength * 10.0);
            }
        }
    }
}

/// Python wrapper for layout-aware graph embedding
#[pyclass]
pub struct PyLayoutAwareEmbedder {
    #[cfg(feature = "anneal")]
    inner: Option<LayoutAwareEmbedder>,
    /// Real statistics from the most recent [`Self::find_embedding`] call
    /// (previously computed by `find_embedding` and then discarded; now
    /// surfaced by [`Self::get_metrics`] instead of an always-empty map).
    #[cfg(feature = "anneal")]
    last_stats: Option<LayoutStats>,
}

#[allow(clippy::missing_const_for_fn)]
#[pymethods]
impl PyLayoutAwareEmbedder {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        target_topology: String,
        use_coordinates: Option<bool>,
        chain_strength_factor: Option<f64>,
        metric: Option<String>,
    ) -> Self {
        #[cfg(feature = "anneal")]
        {
            let _ = target_topology;
            let _ = use_coordinates;
            let _ = chain_strength_factor;
            let _ = metric;
            let config = LayoutConfig {
                distance_weight: 1.0,
                chain_length_weight: 2.0,
                chain_degree_weight: 0.5,
                max_chain_length: 5,
                use_spectral_placement: true,
                refinement_iterations: 10,
            };

            let embedder = LayoutAwareEmbedder::new(config);

            Self {
                inner: Some(embedder),
                last_stats: None,
            }
        }

        #[cfg(not(feature = "anneal"))]
        {
            let _ = target_topology;
            let _ = use_coordinates;
            let _ = chain_strength_factor;
            let _ = metric;
            Self {}
        }
    }

    /// Find embedding for a graph
    #[allow(clippy::needless_pass_by_value)]
    fn find_embedding(
        &mut self,
        source_edges: Vec<(usize, usize)>,
        target_graph: Vec<(usize, usize)>,
        initial_chains: Option<HashMap<usize, Vec<usize>>>,
    ) -> PyResult<HashMap<usize, Vec<usize>>> {
        #[cfg(feature = "anneal")]
        {
            let _ = initial_chains;
            let Some(embedder) = self.inner.as_mut() else {
                return Err(PyValueError::new_err("Embedder not initialized"));
            };
            // Create a hardware graph from target_graph edges
            let hardware_graph = quantrs2_anneal::embedding::HardwareGraph::new_custom(
                target_graph.len() * 2,
                target_graph,
            );

            let (embedding, stats) = embedder
                .find_embedding(&source_edges, source_edges.len(), &hardware_graph)
                .map_err(|e| PyValueError::new_err(format!("Embedding failed: {e}")))?;
            // Surface the real stats (previously computed and discarded)
            // through `get_metrics` instead of throwing them away.
            self.last_stats = Some(stats);
            Ok(embedding.chains)
        }

        #[cfg(not(feature = "anneal"))]
        {
            let _ = source_edges;
            let _ = target_graph;
            let _ = initial_chains;
            Err(PyValueError::new_err("Anneal features not enabled"))
        }
    }

    /// Get real embedding quality metrics from the most recent
    /// [`Self::find_embedding`] call (was previously always an empty map,
    /// discarding the real `LayoutStats` `find_embedding` already computed).
    fn get_metrics(&self) -> PyResult<HashMap<String, f64>> {
        #[cfg(feature = "anneal")]
        {
            if self.inner.is_none() {
                return Err(PyValueError::new_err("Embedder not initialized"));
            }
            let stats = self.last_stats.as_ref().ok_or_else(|| {
                PyValueError::new_err(
                    "No embedding has been computed yet; call find_embedding() first",
                )
            })?;
            Ok(layout_stats_to_metrics(stats))
        }

        #[cfg(not(feature = "anneal"))]
        {
            Err(PyValueError::new_err("Anneal features not enabled"))
        }
    }
}

/// Flatten a real [`LayoutStats`] (computed by
/// `LayoutAwareEmbedder::find_embedding` and previously discarded) into the
/// `HashMap<String, f64>` returned by [`PyLayoutAwareEmbedder::get_metrics`].
/// No `PyErr` involved, so directly unit-testable without a Python
/// interpreter.
#[cfg(feature = "anneal")]
fn layout_stats_to_metrics(stats: &LayoutStats) -> HashMap<String, f64> {
    let mut metrics = HashMap::new();
    metrics.insert("avg_chain_length".to_string(), stats.avg_chain_length);
    metrics.insert(
        "max_chain_length".to_string(),
        stats.max_chain_length as f64,
    );
    metrics.insert(
        "total_chain_length".to_string(),
        stats.total_chain_length as f64,
    );
    metrics.insert("long_chains".to_string(), stats.long_chains as f64);
    metrics.insert("quality_score".to_string(), stats.quality_score);
    metrics
}

/// Chimera graph utilities
#[pyclass]
pub struct PyChimeraGraph;

#[allow(clippy::cast_precision_loss)]
#[allow(clippy::missing_const_for_fn)]
#[allow(clippy::suboptimal_flops)]
#[pymethods]
impl PyChimeraGraph {
    /// Generate Chimera graph edges
    #[staticmethod]
    fn generate_edges(m: usize, n: usize, t: usize) -> Vec<(usize, usize)> {
        let mut edges = Vec::new();

        // Generate Chimera topology
        for i in 0..m {
            for j in 0..n {
                // Unit cell offset
                let offset = (i * n + j) * 2 * t;

                // Internal bipartite connections
                for k in 0..t {
                    for l in 0..t {
                        edges.push((offset + k, offset + t + l));
                    }
                }

                // Horizontal connections
                if j < n - 1 {
                    let right_offset = (i * n + j + 1) * 2 * t;
                    for k in 0..t {
                        edges.push((offset + t + k, right_offset + t + k));
                    }
                }

                // Vertical connections
                if i < m - 1 {
                    let down_offset = ((i + 1) * n + j) * 2 * t;
                    for k in 0..t {
                        edges.push((offset + k, down_offset + k));
                    }
                }
            }
        }

        edges
    }

    /// Get node coordinates for visualization
    #[staticmethod]
    fn get_coordinates(m: usize, n: usize, t: usize, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let n_qubits = m * n * 2 * t;
        let mut coords = vec![vec![0.0; 2]; n_qubits];

        for i in 0..m {
            for j in 0..n {
                let offset = (i * n + j) * 2 * t;

                // Left partition
                for k in 0..t {
                    coords[offset + k][0] = j as f64 + 0.3;
                    coords[offset + k][1] = i as f64 + (k as f64 / t as f64) * 0.8 + 0.1;
                }

                // Right partition
                for k in 0..t {
                    coords[offset + t + k][0] = j as f64 + 0.7;
                    coords[offset + t + k][1] = i as f64 + (k as f64 / t as f64) * 0.8 + 0.1;
                }
            }
        }

        // Convert to numpy array
        let flat_coords: Vec<f64> = coords.into_iter().flatten().collect();
        let array = scirs2_core::ndarray::Array2::from_shape_vec((n_qubits, 2), flat_coords)
            .map_err(|e| PyValueError::new_err(format!("Failed to create array: {e}")))?;

        Ok(array.into_pyarray(py).into())
    }
}

/// Register the anneal module
pub fn register_anneal_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent_module.py(), "anneal")?;

    m.add_class::<PyQuboModel>()?;
    m.add_class::<PyIsingModel>()?;
    m.add_class::<PyPenaltyOptimizer>()?;
    m.add_class::<PyLayoutAwareEmbedder>()?;
    m.add_class::<PyChimeraGraph>()?;

    parent_module.add_submodule(&m)?;
    Ok(())
}

// Pure-Rust regression tests. Only call functions that never construct a
// `PyErr` (no `#[pymethods]`), for the same reason documented in
// `scirs2_bindings.rs`'s and `parametric.rs`'s test modules: this crate
// builds `pyo3` with the `extension-module` feature, so a standalone test
// binary cannot resolve the CPython C-API symbols `PyErr` construction pulls
// in, even along a branch a test never takes.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_penalty_update_scales_up_a_frequently_broken_chain() {
        let mut chain_penalties = HashMap::new();
        let mut constraint_penalties = HashMap::new();
        // Chain 0 breaks in 3 of 4 samples (75% > 10% threshold).
        let chain_breaks = vec![(0, true), (0, true), (0, true), (0, false)];

        apply_penalty_update(
            &mut chain_penalties,
            &mut constraint_penalties,
            &chain_breaks,
            None,
            1.0,  // initial_chain_strength
            0.1,  // min_chain_strength
            10.0, // max_chain_strength
            1.5,  // chain_strength_scale
            1.0,  // constraint_penalty
            0.1,  // learning_rate
        );

        assert!(
            (chain_penalties[&0] - 1.5).abs() < 1e-12,
            "expected strength to scale up from 1.0 to 1.0*1.5, got {}",
            chain_penalties[&0]
        );
    }

    #[test]
    fn apply_penalty_update_scales_down_a_rarely_broken_chain() {
        let mut chain_penalties = HashMap::from([(0usize, 1.5)]);
        let mut constraint_penalties = HashMap::new();
        // Chain 0 breaks in 0 of 200 samples (0% < 1% threshold).
        let chain_breaks: Vec<(usize, bool)> = (0..200).map(|_| (0, false)).collect();

        apply_penalty_update(
            &mut chain_penalties,
            &mut constraint_penalties,
            &chain_breaks,
            None,
            1.0,
            0.1,
            10.0,
            1.5,
            1.0,
            0.1,
        );

        assert!(
            (chain_penalties[&0] - 1.0).abs() < 1e-12,
            "expected strength to scale down from 1.5 to 1.5/1.5, got {}",
            chain_penalties[&0]
        );
    }

    #[test]
    fn apply_penalty_update_leaves_a_moderately_broken_chain_unchanged() {
        let mut chain_penalties = HashMap::from([(0usize, 1.0)]);
        let mut constraint_penalties = HashMap::new();
        // 5% break rate: between the 1% and 10% thresholds -> no change.
        let mut chain_breaks: Vec<(usize, bool)> = (0..19).map(|_| (0, false)).collect();
        chain_breaks.push((0, true));

        apply_penalty_update(
            &mut chain_penalties,
            &mut constraint_penalties,
            &chain_breaks,
            None,
            1.0,
            0.1,
            10.0,
            1.5,
            1.0,
            0.1,
        );

        assert!((chain_penalties[&0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn apply_penalty_update_increases_a_violated_constraint_penalty() {
        let mut chain_penalties = HashMap::new();
        let mut constraint_penalties = HashMap::new();
        let violations = HashMap::from([("balance".to_string(), 0.9_f64)]); // > learning_rate

        apply_penalty_update(
            &mut chain_penalties,
            &mut constraint_penalties,
            &[],
            Some(&violations),
            1.0,
            0.1,
            10.0,
            1.5,
            1.0, // constraint_penalty
            0.1, // learning_rate
        );

        let penalty = constraint_penalties["balance"];
        assert!(
            penalty > 1.0,
            "a violated constraint's penalty must increase, got {penalty}"
        );
    }

    #[test]
    fn apply_penalty_update_is_a_no_op_for_empty_input() {
        let mut chain_penalties = HashMap::new();
        let mut constraint_penalties = HashMap::new();
        apply_penalty_update(
            &mut chain_penalties,
            &mut constraint_penalties,
            &[],
            None,
            1.0,
            0.1,
            10.0,
            1.5,
            1.0,
            0.1,
        );
        assert!(chain_penalties.is_empty());
        assert!(constraint_penalties.is_empty());
    }

    #[test]
    #[cfg(feature = "anneal")]
    fn layout_stats_to_metrics_surfaces_every_real_field() {
        let stats = LayoutStats {
            avg_chain_length: 2.5,
            max_chain_length: 4,
            total_chain_length: 10,
            long_chains: 1,
            quality_score: 0.87,
        };
        let metrics = layout_stats_to_metrics(&stats);
        assert_eq!(metrics.len(), 5, "must not be the old always-empty map");
        assert!((metrics["avg_chain_length"] - 2.5).abs() < 1e-12);
        assert!((metrics["max_chain_length"] - 4.0).abs() < 1e-12);
        assert!((metrics["total_chain_length"] - 10.0).abs() < 1e-12);
        assert!((metrics["long_chains"] - 1.0).abs() < 1e-12);
        assert!((metrics["quality_score"] - 0.87).abs() < 1e-12);
    }
}
