//! # QuantumReservoirComputer - encoding Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

use super::types::{InputEncoding, OutputMeasurement, ReservoirDynamics};

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Process input through quantum reservoir
    pub fn process_input(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        let start_time = std::time::Instant::now();
        self.encode_input(input)?;
        self.evolve_reservoir()?;
        let features = self.extract_features()?;
        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        self.update_processing_time(processing_time);
        Ok(features)
    }
    /// Encode input data into quantum state
    pub(super) fn encode_input(&mut self, input: &Array1<f64>) -> Result<()> {
        match self.config.input_encoding {
            InputEncoding::Amplitude => {
                self.encode_amplitude(input)?;
            }
            InputEncoding::Phase => {
                self.encode_phase(input)?;
            }
            InputEncoding::BasisState => {
                self.encode_basis_state(input)?;
            }
            _ => {
                self.encode_amplitude(input)?;
            }
        }
        Ok(())
    }
    /// Evolve quantum reservoir through dynamics
    pub(super) fn evolve_reservoir(&mut self) -> Result<()> {
        match self.config.dynamics {
            ReservoirDynamics::Unitary => {
                self.evolve_unitary()?;
            }
            ReservoirDynamics::Open => {
                self.evolve_open_system()?;
            }
            ReservoirDynamics::NISQ => {
                self.evolve_nisq()?;
            }
            ReservoirDynamics::Adiabatic => {
                self.evolve_adiabatic()?;
            }
            ReservoirDynamics::Floquet => {
                self.evolve_unitary()?;
            }
            ReservoirDynamics::QuantumWalk => {
                self.evolve_unitary()?;
            }
            ReservoirDynamics::ContinuousTime => {
                self.evolve_open_system()?;
            }
            ReservoirDynamics::DigitalQuantum => {
                self.evolve_unitary()?;
            }
            ReservoirDynamics::Variational => {
                self.evolve_unitary()?;
            }
            ReservoirDynamics::HamiltonianLearning => {
                self.evolve_unitary()?;
            }
            ReservoirDynamics::ManyBodyLocalized => {
                self.evolve_unitary()?;
            }
            ReservoirDynamics::QuantumChaotic => {
                self.evolve_unitary()?;
            }
        }
        Ok(())
    }
    /// Extract features from reservoir state
    pub(super) fn extract_features(&mut self) -> Result<Array1<f64>> {
        match self.config.output_measurement {
            OutputMeasurement::PauliExpectation => self.measure_pauli_expectations(),
            OutputMeasurement::Probability => self.measure_probabilities(),
            OutputMeasurement::Correlations => self.measure_correlations(),
            OutputMeasurement::Entanglement => self.measure_entanglement(),
            OutputMeasurement::Fidelity => self.measure_fidelity(),
            OutputMeasurement::QuantumFisherInformation => self.measure_pauli_expectations(),
            OutputMeasurement::Variance => self.measure_pauli_expectations(),
            OutputMeasurement::HigherOrderMoments => self.measure_pauli_expectations(),
            OutputMeasurement::SpectralProperties => self.measure_pauli_expectations(),
            OutputMeasurement::QuantumCoherence => self.measure_entanglement(),
            _ => self.measure_pauli_expectations(),
        }
    }
    /// Update processing time metrics
    pub(super) fn update_processing_time(&mut self, time_ms: f64) {
        let count = self.metrics.training_examples as f64;
        self.metrics.avg_processing_time_ms =
            self.metrics.avg_processing_time_ms.mul_add(count, time_ms) / (count + 1.0);
    }
}
