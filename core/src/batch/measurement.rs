//! Batch measurement operations using SciRS2 parallel algorithms

use super::{BatchMeasurementResult, BatchStateVector};
use crate::{
    error::{QuantRS2Error, QuantRS2Result},
    qubit::QubitId,
};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rand::prelude::*;
use rand::{thread_rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

/// Batch measurement configuration
#[derive(Debug, Clone)]
pub struct MeasurementConfig {
    /// Number of measurement shots per state
    pub shots: usize,
    /// Whether to return post-measurement states
    pub return_states: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Use parallel processing
    pub parallel: bool,
}

impl Default for MeasurementConfig {
    fn default() -> Self {
        Self {
            shots: 1024,
            return_states: false,
            seed: None,
            parallel: true,
        }
    }
}

/// Perform batch measurements on multiple quantum states
pub fn measure_batch(
    batch: &BatchStateVector,
    qubits_to_measure: &[QubitId],
    config: MeasurementConfig,
) -> QuantRS2Result<BatchMeasurementResult> {
    let batch_size = batch.batch_size();
    let n_qubits = batch.n_qubits;
    let num_measurements = qubits_to_measure.len();

    // Validate qubits
    for &qubit in qubits_to_measure {
        if qubit.0 as usize >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit.0));
        }
    }

    // Initialize results
    let mut outcomes = Array2::zeros((batch_size, num_measurements));
    let mut probabilities = Array2::zeros((batch_size, num_measurements));
    let post_measurement_states = if config.return_states {
        Some(batch.states.clone())
    } else {
        None
    };

    // Perform measurements
    if config.parallel && batch_size > 16 {
        // Parallel measurement
        let results: Vec<(Vec<u8>, Vec<f64>)> = (0..batch_size)
            .into_par_iter()
            .map(|i| {
                let state = batch.states.row(i);
                measure_single_state(&state.to_owned(), qubits_to_measure, &config)
            })
            .collect();

        // Collect results
        for (i, (outcome, probs)) in results.into_iter().enumerate() {
            for (j, &val) in outcome.iter().enumerate() {
                outcomes[[i, j]] = val;
            }
            for (j, &prob) in probs.iter().enumerate() {
                probabilities[[i, j]] = prob;
            }
        }
    } else {
        // Sequential measurement
        for i in 0..batch_size {
            let state = batch.states.row(i);
            let (outcome, probs) =
                measure_single_state(&state.to_owned(), qubits_to_measure, &config);

            for (j, &val) in outcome.iter().enumerate() {
                outcomes[[i, j]] = val;
            }
            for (j, &prob) in probs.iter().enumerate() {
                probabilities[[i, j]] = prob;
            }
        }
    }

    Ok(BatchMeasurementResult {
        outcomes,
        probabilities,
        post_measurement_states,
    })
}

/// Measure a single state
fn measure_single_state(
    state: &Array1<Complex64>,
    qubits_to_measure: &[QubitId],
    config: &MeasurementConfig,
) -> (Vec<u8>, Vec<f64>) {
    let mut rng = if let Some(seed) = config.seed {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_seed(thread_rng().gen())
    };

    let mut outcomes = Vec::with_capacity(qubits_to_measure.len());
    let mut probabilities = Vec::with_capacity(qubits_to_measure.len());

    for &qubit in qubits_to_measure {
        let (outcome, prob) = measure_qubit(state, qubit, &mut rng);
        outcomes.push(outcome);
        probabilities.push(prob);
    }

    (outcomes, probabilities)
}

/// Measure a single qubit
fn measure_qubit(state: &Array1<Complex64>, qubit: QubitId, rng: &mut StdRng) -> (u8, f64) {
    let qubit_idx = qubit.0 as usize;
    let state_size = state.len();
    let n_qubits = (state_size as f64).log2() as usize;

    // Calculate probability of measuring |0>
    let mut prob_zero = 0.0;
    let qubit_mask = 1 << qubit_idx;

    for i in 0..state_size {
        if i & qubit_mask == 0 {
            prob_zero += state[i].norm_sqr();
        }
    }

    // Perform measurement
    let outcome = if rng.gen::<f64>() < prob_zero { 0 } else { 1 };
    let probability = if outcome == 0 {
        prob_zero
    } else {
        1.0 - prob_zero
    };

    (outcome, probability)
}

/// Perform batch measurements with statistics
pub fn measure_batch_with_statistics(
    batch: &BatchStateVector,
    qubits_to_measure: &[QubitId],
    shots: usize,
) -> QuantRS2Result<BatchMeasurementStatistics> {
    let batch_size = batch.batch_size();
    let measurement_size = qubits_to_measure.len();

    // Collect measurement statistics for each state in parallel
    let statistics: Vec<_> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let state = batch.states.row(i);
            compute_measurement_statistics(&state.to_owned(), qubits_to_measure, shots)
        })
        .collect();

    Ok(BatchMeasurementStatistics {
        statistics,
        batch_size,
        measurement_size,
        shots,
    })
}

/// Measurement statistics for a batch
#[derive(Debug, Clone)]
pub struct BatchMeasurementStatistics {
    /// Statistics for each state in the batch
    pub statistics: Vec<MeasurementStatistics>,
    /// Batch size
    pub batch_size: usize,
    /// Number of qubits measured
    pub measurement_size: usize,
    /// Number of shots
    pub shots: usize,
}

/// Measurement statistics for a single state
#[derive(Debug, Clone)]
pub struct MeasurementStatistics {
    /// Count of each measurement outcome
    pub counts: HashMap<String, usize>,
    /// Probability of each outcome
    pub probabilities: HashMap<String, f64>,
    /// Most likely outcome
    pub most_likely: String,
    /// Entropy of the measurement distribution
    pub entropy: f64,
}

/// Compute measurement statistics for a single state
fn compute_measurement_statistics(
    state: &Array1<Complex64>,
    qubits_to_measure: &[QubitId],
    shots: usize,
) -> MeasurementStatistics {
    let mut rng = StdRng::from_seed(thread_rng().gen());
    let mut counts: HashMap<String, usize> = HashMap::new();

    // Perform measurements
    for _ in 0..shots {
        let mut outcome = String::new();
        for &qubit in qubits_to_measure {
            let (bit, _) = measure_qubit(state, qubit, &mut rng);
            outcome.push(if bit == 0 { '0' } else { '1' });
        }
        *counts.entry(outcome).or_insert(0) += 1;
    }

    // Compute probabilities
    let mut probabilities = HashMap::new();
    let mut most_likely = String::new();
    let mut max_count = 0;

    for (outcome, &count) in &counts {
        let prob = count as f64 / shots as f64;
        probabilities.insert(outcome.clone(), prob);

        if count > max_count {
            max_count = count;
            most_likely = outcome.clone();
        }
    }

    // Compute entropy
    let entropy = -probabilities
        .values()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.log2())
        .sum::<f64>();

    MeasurementStatistics {
        counts,
        probabilities,
        most_likely,
        entropy,
    }
}

/// Batch expectation value measurement
pub fn measure_expectation_batch(
    batch: &BatchStateVector,
    observable_qubits: &[(QubitId, Array2<Complex64>)],
) -> QuantRS2Result<Vec<f64>> {
    let batch_size = batch.batch_size();

    // Compute expectation values in parallel
    let expectations: Vec<_> = (0..batch_size)
        .into_par_iter()
        .map(|i| {
            let state = batch.states.row(i);
            compute_observable_expectation(&state.to_owned(), observable_qubits, batch.n_qubits)
        })
        .collect::<QuantRS2Result<Vec<_>>>()?;

    Ok(expectations)
}

/// Compute expectation value of an observable
fn compute_observable_expectation(
    state: &Array1<Complex64>,
    observable_qubits: &[(QubitId, Array2<Complex64>)],
    n_qubits: usize,
) -> QuantRS2Result<f64> {
    // For simplicity, compute single-qubit observable expectations
    // and multiply them (assuming they commute)
    let mut total_expectation = 1.0;

    for (qubit, observable) in observable_qubits {
        let qubit_idx = qubit.0 as usize;
        if qubit_idx >= n_qubits {
            return Err(QuantRS2Error::InvalidQubitId(qubit.0));
        }

        // Compute expectation for this qubit
        let exp = compute_single_qubit_expectation(state, *qubit, observable, n_qubits)?;
        total_expectation *= exp;
    }

    Ok(total_expectation)
}

/// Compute single-qubit expectation value
fn compute_single_qubit_expectation(
    state: &Array1<Complex64>,
    qubit: QubitId,
    observable: &Array2<Complex64>,
    n_qubits: usize,
) -> QuantRS2Result<f64> {
    if observable.shape() != &[2, 2] {
        return Err(QuantRS2Error::InvalidInput(
            "Observable must be a 2x2 matrix".to_string(),
        ));
    }

    let qubit_idx = qubit.0 as usize;
    let state_size = 1 << n_qubits;
    let qubit_mask = 1 << qubit_idx;

    let mut expectation = Complex64::new(0.0, 0.0);

    for i in 0..state_size {
        for j in 0..state_size {
            // Check if states differ only in the target qubit
            if (i ^ j) == qubit_mask {
                let qi = (i >> qubit_idx) & 1;
                let qj = (j >> qubit_idx) & 1;

                expectation += state[i].conj() * observable[[qi, qj]] * state[j];
            } else if i == j {
                let qi = (i >> qubit_idx) & 1;
                expectation += state[i].conj() * observable[[qi, qi]] * state[i];
            }
        }
    }

    Ok(expectation.re)
}

/// Batch tomography measurements
pub fn measure_tomography_batch(
    batch: &BatchStateVector,
    qubits: &[QubitId],
    basis: TomographyBasis,
) -> QuantRS2Result<BatchTomographyResult> {
    let measurements = match basis {
        TomographyBasis::Pauli => get_pauli_measurements(qubits),
        TomographyBasis::Computational => get_computational_measurements(qubits),
        TomographyBasis::Custom(ref bases) => bases.clone(),
    };

    let mut results = Vec::new();

    for (name, observable_qubits) in measurements {
        let expectations = measure_expectation_batch(batch, &observable_qubits)?;
        results.push((name, expectations));
    }

    Ok(BatchTomographyResult {
        measurements: results,
        basis,
        qubits: qubits.to_vec(),
    })
}

/// Tomography basis
#[derive(Debug, Clone)]
pub enum TomographyBasis {
    /// Pauli basis (X, Y, Z)
    Pauli,
    /// Computational basis (|0>, |1>)
    Computational,
    /// Custom measurement basis
    Custom(Vec<(String, Vec<(QubitId, Array2<Complex64>)>)>),
}

/// Batch tomography result
#[derive(Debug, Clone)]
pub struct BatchTomographyResult {
    /// Measurement results (name, expectations for each state)
    pub measurements: Vec<(String, Vec<f64>)>,
    /// Basis used
    pub basis: TomographyBasis,
    /// Qubits measured
    pub qubits: Vec<QubitId>,
}

/// Get Pauli basis measurements
fn get_pauli_measurements(qubits: &[QubitId]) -> Vec<(String, Vec<(QubitId, Array2<Complex64>)>)> {
    use ndarray::array;

    let pauli_x = array![
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    ];

    let pauli_y = array![
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
        [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)]
    ];

    let pauli_z = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
    ];

    let mut measurements = Vec::new();

    for &qubit in qubits {
        measurements.push((format!("X{}", qubit.0), vec![(qubit, pauli_x.clone())]));
        measurements.push((format!("Y{}", qubit.0), vec![(qubit, pauli_y.clone())]));
        measurements.push((format!("Z{}", qubit.0), vec![(qubit, pauli_z.clone())]));
    }

    measurements
}

/// Get computational basis measurements
fn get_computational_measurements(
    qubits: &[QubitId],
) -> Vec<(String, Vec<(QubitId, Array2<Complex64>)>)> {
    use ndarray::array;

    let proj_0 = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)]
    ];

    let proj_1 = array![
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
    ];

    let mut measurements = Vec::new();

    for &qubit in qubits {
        measurements.push((format!("|0⟩{}", qubit.0), vec![(qubit, proj_0.clone())]));
        measurements.push((format!("|1⟩{}", qubit.0), vec![(qubit, proj_1.clone())]));
    }

    measurements
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_batch_measurement() {
        let batch = BatchStateVector::new(5, 2, Default::default()).unwrap();
        let config = MeasurementConfig {
            shots: 100,
            return_states: false,
            seed: Some(42),
            parallel: false,
        };

        let result = measure_batch(&batch, &[QubitId(0), QubitId(1)], config).unwrap();

        assert_eq!(result.outcomes.shape(), &[5, 2]);
        assert_eq!(result.probabilities.shape(), &[5, 2]);

        // All states are |00>, so measurements should be 0
        for i in 0..5 {
            assert_eq!(result.outcomes[[i, 0]], 0);
            assert_eq!(result.outcomes[[i, 1]], 0);
            assert!((result.probabilities[[i, 0]] - 1.0).abs() < 1e-10);
            assert!((result.probabilities[[i, 1]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_measurement_statistics() {
        // Create a superposition state
        let mut states = Array2::zeros((1, 2));
        states[[0, 0]] = Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0);
        states[[0, 1]] = Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0);

        let batch = BatchStateVector::from_states(states, Default::default()).unwrap();

        let stats = measure_batch_with_statistics(&batch, &[QubitId(0)], 1000).unwrap();

        assert_eq!(stats.batch_size, 1);
        assert_eq!(stats.measurement_size, 1);

        let stat = &stats.statistics[0];
        // Should have roughly equal counts for "0" and "1"
        assert!(stat.counts.contains_key("0"));
        assert!(stat.counts.contains_key("1"));

        // Entropy should be close to 1 for equal superposition
        assert!((stat.entropy - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_expectation_measurement() {
        let batch = BatchStateVector::new(3, 1, Default::default()).unwrap();

        // Pauli Z observable
        let pauli_z = array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)]
        ];

        let expectations = measure_expectation_batch(&batch, &[(QubitId(0), pauli_z)]).unwrap();

        assert_eq!(expectations.len(), 3);
        // All states are |0>, so Z expectation is +1
        for exp in expectations {
            assert!((exp - 1.0).abs() < 1e-10);
        }
    }
}
