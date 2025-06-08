//! Python bindings for pulse-level control
//!
//! This module provides Python access to low-level pulse control for quantum operations.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::exceptions::PyValueError;
use std::collections::HashMap;
use num_complex::Complex64;

use quantrs2_device::pulse::{
    PulseShape, ChannelType, PulseInstruction, PulseSchedule, PulseCalibration,
    PulseBuilder, MeasLevel, PulseResult, MeasurementData, PulseLibrary, PulseTemplates,
};

/// Python wrapper for pulse shapes
#[pyclass(name = "PulseShape")]
#[derive(Clone)]
pub struct PyPulseShape {
    inner: PulseShape,
}

#[pymethods]
impl PyPulseShape {
    /// Create a Gaussian pulse
    #[staticmethod]
    #[pyo3(signature = (duration, sigma, amplitude, phase=0.0))]
    fn gaussian(duration: f64, sigma: f64, amplitude: f64, phase: f64) -> Self {
        let amp = Complex64::from_polar(amplitude, phase);
        Self {
            inner: PulseShape::Gaussian { duration, sigma, amplitude: amp },
        }
    }

    /// Create a Gaussian DRAG pulse
    #[staticmethod]
    #[pyo3(signature = (duration, sigma, amplitude, beta, phase=0.0))]
    fn gaussian_drag(duration: f64, sigma: f64, amplitude: f64, beta: f64, phase: f64) -> Self {
        let amp = Complex64::from_polar(amplitude, phase);
        Self {
            inner: PulseShape::GaussianDrag { duration, sigma, amplitude: amp, beta },
        }
    }

    /// Create a square pulse
    #[staticmethod]
    #[pyo3(signature = (duration, amplitude, phase=0.0))]
    fn square(duration: f64, amplitude: f64, phase: f64) -> Self {
        let amp = Complex64::from_polar(amplitude, phase);
        Self {
            inner: PulseShape::Square { duration, amplitude: amp },
        }
    }

    /// Create a cosine-tapered pulse
    #[staticmethod]
    #[pyo3(signature = (duration, amplitude, rise_time, phase=0.0))]
    fn cosine_tapered(duration: f64, amplitude: f64, rise_time: f64, phase: f64) -> Self {
        let amp = Complex64::from_polar(amplitude, phase);
        Self {
            inner: PulseShape::CosineTapered { duration, amplitude: amp, rise_time },
        }
    }

    /// Create an arbitrary waveform pulse
    #[staticmethod]
    fn arbitrary(samples: Vec<(f64, f64)>, sample_rate: f64) -> Self {
        let samples_complex: Vec<Complex64> = samples
            .into_iter()
            .map(|(re, im)| Complex64::new(re, im))
            .collect();
        Self {
            inner: PulseShape::Arbitrary { samples: samples_complex, sample_rate },
        }
    }

    /// Get pulse type as string
    #[getter]
    fn pulse_type(&self) -> &str {
        match &self.inner {
            PulseShape::Gaussian { .. } => "gaussian",
            PulseShape::GaussianDrag { .. } => "gaussian_drag",
            PulseShape::Square { .. } => "square",
            PulseShape::CosineTapered { .. } => "cosine_tapered",
            PulseShape::Arbitrary { .. } => "arbitrary",
        }
    }

    /// Get pulse duration
    #[getter]
    fn duration(&self) -> f64 {
        match &self.inner {
            PulseShape::Gaussian { duration, .. } |
            PulseShape::GaussianDrag { duration, .. } |
            PulseShape::Square { duration, .. } |
            PulseShape::CosineTapered { duration, .. } => *duration,
            PulseShape::Arbitrary { samples, sample_rate } => {
                samples.len() as f64 / sample_rate
            }
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            PulseShape::Gaussian { duration, sigma, amplitude } => {
                format!("PulseShape.gaussian(duration={:.2e}, sigma={:.2e}, amplitude={:.3})",
                    duration, sigma, amplitude.norm())
            }
            PulseShape::GaussianDrag { duration, sigma, amplitude, beta } => {
                format!("PulseShape.gaussian_drag(duration={:.2e}, sigma={:.2e}, amplitude={:.3}, beta={:.3})",
                    duration, sigma, amplitude.norm(), beta)
            }
            PulseShape::Square { duration, amplitude } => {
                format!("PulseShape.square(duration={:.2e}, amplitude={:.3})",
                    duration, amplitude.norm())
            }
            PulseShape::CosineTapered { duration, amplitude, rise_time } => {
                format!("PulseShape.cosine_tapered(duration={:.2e}, amplitude={:.3}, rise_time={:.2e})",
                    duration, amplitude.norm(), rise_time)
            }
            PulseShape::Arbitrary { samples, sample_rate } => {
                format!("PulseShape.arbitrary(num_samples={}, sample_rate={:.2e})",
                    samples.len(), sample_rate)
            }
        }
    }
}

/// Python wrapper for channel types
#[pyclass(name = "Channel")]
#[derive(Clone)]
pub struct PyChannel {
    inner: ChannelType,
}

#[pymethods]
impl PyChannel {
    /// Create a drive channel
    #[staticmethod]
    fn drive(qubit: u32) -> Self {
        Self { inner: ChannelType::Drive(qubit) }
    }

    /// Create a measure channel
    #[staticmethod]
    fn measure(qubit: u32) -> Self {
        Self { inner: ChannelType::Measure(qubit) }
    }

    /// Create a control channel
    #[staticmethod]
    fn control(control_qubit: u32, target_qubit: u32) -> Self {
        Self { inner: ChannelType::Control(control_qubit, target_qubit) }
    }

    /// Create a readout channel
    #[staticmethod]
    fn readout(qubit: u32) -> Self {
        Self { inner: ChannelType::Readout(qubit) }
    }

    /// Create an acquire channel
    #[staticmethod]
    fn acquire(qubit: u32) -> Self {
        Self { inner: ChannelType::Acquire(qubit) }
    }

    /// Get channel type as string
    #[getter]
    fn channel_type(&self) -> &str {
        match &self.inner {
            ChannelType::Drive(_) => "drive",
            ChannelType::Measure(_) => "measure",
            ChannelType::Control(_, _) => "control",
            ChannelType::Readout(_) => "readout",
            ChannelType::Acquire(_) => "acquire",
        }
    }

    /// Get associated qubit(s)
    #[getter]
    fn qubits(&self) -> Vec<u32> {
        match &self.inner {
            ChannelType::Drive(q) | ChannelType::Measure(q) |
            ChannelType::Readout(q) | ChannelType::Acquire(q) => vec![*q],
            ChannelType::Control(q1, q2) => vec![*q1, *q2],
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ChannelType::Drive(q) => format!("Channel.drive({})", q),
            ChannelType::Measure(q) => format!("Channel.measure({})", q),
            ChannelType::Control(q1, q2) => format!("Channel.control({}, {})", q1, q2),
            ChannelType::Readout(q) => format!("Channel.readout({})", q),
            ChannelType::Acquire(q) => format!("Channel.acquire({})", q),
        }
    }
}

/// Python wrapper for pulse schedules
#[pyclass(name = "PulseSchedule")]
#[derive(Clone)]
pub struct PyPulseSchedule {
    inner: PulseSchedule,
}

#[pymethods]
impl PyPulseSchedule {
    /// Get schedule name
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Get schedule duration
    #[getter]
    fn duration(&self) -> u64 {
        self.inner.duration
    }

    /// Get number of instructions
    #[getter]
    fn num_instructions(&self) -> usize {
        self.inner.instructions.len()
    }

    /// Get metadata as dict
    #[getter]
    fn metadata(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        for (key, value) in &self.inner.metadata {
            dict.set_item(key, value)?;
        }
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!("PulseSchedule(name='{}', duration={}, instructions={})",
            self.inner.name, self.inner.duration, self.inner.instructions.len())
    }
}

/// Python wrapper for pulse calibration
#[pyclass(name = "PulseCalibration")]
#[derive(Clone)]
pub struct PyPulseCalibration {
    inner: PulseCalibration,
}

#[pymethods]
impl PyPulseCalibration {
    /// Create a new calibration
    #[new]
    #[pyo3(signature = (num_qubits, dt=2.2222e-10))]
    fn new(num_qubits: usize, dt: f64) -> Self {
        let mut cal = PulseCalibration {
            single_qubit_defaults: HashMap::new(),
            two_qubit_defaults: HashMap::new(),
            qubit_frequencies: vec![5.0; num_qubits],
            meas_frequencies: vec![6.5; num_qubits],
            drive_powers: vec![0.1; num_qubits],
            dt,
        };

        // Add some default pulses
        cal.single_qubit_defaults.insert(
            "x".to_string(),
            PulseLibrary::gaussian(160e-9, 40e-9, 0.5),
        );
        cal.single_qubit_defaults.insert(
            "sx".to_string(),
            PulseLibrary::gaussian(160e-9, 40e-9, 0.25),
        );

        Self { inner: cal }
    }

    /// Get sampling time (dt)
    #[getter]
    fn dt(&self) -> f64 {
        self.inner.dt
    }

    /// Get qubit frequencies
    #[getter]
    fn qubit_frequencies(&self) -> Vec<f64> {
        self.inner.qubit_frequencies.clone()
    }

    /// Get measurement frequencies
    #[getter]
    fn meas_frequencies(&self) -> Vec<f64> {
        self.inner.meas_frequencies.clone()
    }

    /// Set qubit frequency
    fn set_qubit_frequency(&mut self, qubit: usize, frequency: f64) -> PyResult<()> {
        if qubit >= self.inner.qubit_frequencies.len() {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.inner.qubit_frequencies[qubit] = frequency;
        Ok(())
    }

    /// Set measurement frequency
    fn set_meas_frequency(&mut self, qubit: usize, frequency: f64) -> PyResult<()> {
        if qubit >= self.inner.meas_frequencies.len() {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.inner.meas_frequencies[qubit] = frequency;
        Ok(())
    }
}

/// Python wrapper for pulse builder
#[pyclass(name = "PulseBuilder")]
pub struct PyPulseBuilder {
    builder: Option<PulseBuilder>,
}

#[pymethods]
impl PyPulseBuilder {
    /// Create a new pulse builder
    #[new]
    #[pyo3(signature = (name, calibration=None))]
    fn new(name: String, calibration: Option<PyPulseCalibration>) -> Self {
        let builder = match calibration {
            Some(cal) => PulseBuilder::with_calibration(name, cal.inner),
            None => PulseBuilder::new(name),
        };
        Self { builder: Some(builder) }
    }

    /// Play a pulse on a channel
    fn play(&mut self, channel: &PyChannel, pulse: &PyPulseShape) -> PyResult<()> {
        if let Some(builder) = self.builder.take() {
            self.builder = Some(builder.play(channel.inner.clone(), pulse.inner.clone()));
            Ok(())
        } else {
            Err(PyValueError::new_err("Builder already consumed"))
        }
    }

    /// Add a delay
    fn delay(&mut self, duration: u64, channel: &PyChannel) -> PyResult<()> {
        if let Some(builder) = self.builder.take() {
            self.builder = Some(builder.delay(duration, channel.inner.clone()));
            Ok(())
        } else {
            Err(PyValueError::new_err("Builder already consumed"))
        }
    }

    /// Set phase on a channel
    fn set_phase(&mut self, channel: &PyChannel, phase: f64) -> PyResult<()> {
        if let Some(builder) = self.builder.take() {
            self.builder = Some(builder.set_phase(channel.inner.clone(), phase));
            Ok(())
        } else {
            Err(PyValueError::new_err("Builder already consumed"))
        }
    }

    /// Set frequency on a channel
    fn set_frequency(&mut self, channel: &PyChannel, frequency: f64) -> PyResult<()> {
        if let Some(builder) = self.builder.take() {
            self.builder = Some(builder.set_frequency(channel.inner.clone(), frequency));
            Ok(())
        } else {
            Err(PyValueError::new_err("Builder already consumed"))
        }
    }

    /// Add a barrier to synchronize channels
    fn barrier(&mut self, channels: Vec<PyChannel>) -> PyResult<()> {
        if let Some(builder) = self.builder.take() {
            let channel_types: Vec<ChannelType> = channels.into_iter()
                .map(|c| c.inner.clone())
                .collect();
            self.builder = Some(builder.barrier(channel_types));
            Ok(())
        } else {
            Err(PyValueError::new_err("Builder already consumed"))
        }
    }

    /// Add metadata
    fn with_metadata(&mut self, key: String, value: String) -> PyResult<()> {
        if let Some(builder) = self.builder.take() {
            self.builder = Some(builder.with_metadata(key, value));
            Ok(())
        } else {
            Err(PyValueError::new_err("Builder already consumed"))
        }
    }

    /// Build the pulse schedule
    fn build(&mut self) -> PyResult<PyPulseSchedule> {
        if let Some(builder) = self.builder.take() {
            Ok(PyPulseSchedule { inner: builder.build() })
        } else {
            Err(PyValueError::new_err("Builder already consumed"))
        }
    }
}

/// Python wrapper for measurement level
#[pyclass(name = "MeasLevel")]
#[derive(Clone, Copy)]
pub enum PyMeasLevel {
    /// Raw ADC values
    Raw,
    /// IQ values after demodulation
    Kerneled,
    /// Discriminated qubit states
    Classified,
}

impl From<PyMeasLevel> for MeasLevel {
    fn from(level: PyMeasLevel) -> Self {
        match level {
            PyMeasLevel::Raw => MeasLevel::Raw,
            PyMeasLevel::Kerneled => MeasLevel::Kerneled,
            PyMeasLevel::Classified => MeasLevel::Classified,
        }
    }
}

/// Python wrapper for pulse library
#[pyclass(name = "PulseLibrary")]
pub struct PyPulseLibrary;

#[pymethods]
impl PyPulseLibrary {
    /// Create a standard X gate pulse
    #[staticmethod]
    fn x_pulse(qubit: u32, calibration: Option<&PyPulseCalibration>) -> PyPulseShape {
        let pulse = if let Some(cal) = calibration {
            PulseLibrary::x_pulse(&cal.inner, qubit)
        } else {
            PulseLibrary::gaussian(160e-9, 40e-9, 0.5)
        };
        PyPulseShape { inner: pulse }
    }

    /// Create a standard Y gate pulse
    #[staticmethod]
    fn y_pulse(qubit: u32, calibration: Option<&PyPulseCalibration>) -> PyPulseShape {
        let pulse = if let Some(cal) = calibration {
            PulseLibrary::y_pulse(&cal.inner, qubit)
        } else {
            PulseShape::Gaussian {
                duration: 160e-9,
                sigma: 40e-9,
                amplitude: Complex64::new(0.0, 0.5),
            }
        };
        PyPulseShape { inner: pulse }
    }

    /// Create a standard SX (√X) gate pulse
    #[staticmethod]
    fn sx_pulse(qubit: u32, calibration: Option<&PyPulseCalibration>) -> PyPulseShape {
        let pulse = if let Some(cal) = calibration {
            PulseLibrary::sx_pulse(&cal.inner, qubit)
        } else {
            PulseLibrary::gaussian(160e-9, 40e-9, 0.25)
        };
        PyPulseShape { inner: pulse }
    }

    /// Create a measurement pulse
    #[staticmethod]
    fn measure_pulse(qubit: u32, calibration: Option<&PyPulseCalibration>) -> PyPulseShape {
        let pulse = if let Some(cal) = calibration {
            PulseLibrary::measure_pulse(&cal.inner, qubit)
        } else {
            PulseLibrary::square(2e-6, 0.1)
        };
        PyPulseShape { inner: pulse }
    }
}

/// Python wrapper for pulse templates
#[pyclass(name = "PulseTemplates")]
pub struct PyPulseTemplates;

#[pymethods]
impl PyPulseTemplates {
    /// Create an X gate schedule
    #[staticmethod]
    fn x_gate(qubit: u32, calibration: &PyPulseCalibration) -> PyPulseSchedule {
        PyPulseSchedule {
            inner: PulseTemplates::x_gate(qubit, &calibration.inner)
        }
    }

    /// Create a CNOT gate schedule
    #[staticmethod]
    fn cnot_gate(control: u32, target: u32, calibration: &PyPulseCalibration) -> PyPulseSchedule {
        PyPulseSchedule {
            inner: PulseTemplates::cnot_gate(control, target, &calibration.inner)
        }
    }

    /// Create a measurement schedule
    #[staticmethod]
    fn measure(qubits: Vec<u32>, calibration: &PyPulseCalibration) -> PyPulseSchedule {
        PyPulseSchedule {
            inner: PulseTemplates::measure(qubits, &calibration.inner)
        }
    }

    /// Create a Rabi experiment
    #[staticmethod]
    fn rabi_experiment(
        qubit: u32, 
        amplitudes: Vec<f64>, 
        calibration: &PyPulseCalibration
    ) -> Vec<PyPulseSchedule> {
        PulseTemplates::rabi_experiment(qubit, amplitudes, &calibration.inner)
            .into_iter()
            .map(|schedule| PyPulseSchedule { inner: schedule })
            .collect()
    }

    /// Create a T1 experiment
    #[staticmethod]
    fn t1_experiment(
        qubit: u32, 
        delays: Vec<u64>, 
        calibration: &PyPulseCalibration
    ) -> Vec<PyPulseSchedule> {
        PulseTemplates::t1_experiment(qubit, delays, &calibration.inner)
            .into_iter()
            .map(|schedule| PyPulseSchedule { inner: schedule })
            .collect()
    }

    /// Create a Ramsey (T2) experiment
    #[staticmethod]
    #[pyo3(signature = (qubit, delays, detuning, calibration))]
    fn ramsey_experiment(
        qubit: u32, 
        delays: Vec<u64>, 
        detuning: f64,
        calibration: &PyPulseCalibration
    ) -> Vec<PyPulseSchedule> {
        PulseTemplates::ramsey_experiment(qubit, delays, detuning, &calibration.inner)
            .into_iter()
            .map(|schedule| PyPulseSchedule { inner: schedule })
            .collect()
    }
}

/// Register the pulse module
pub fn register_pulse_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let pulse_module = PyModule::new(parent.py(), "pulse")?;
    
    // Add classes
    pulse_module.add_class::<PyPulseShape>()?;
    pulse_module.add_class::<PyChannel>()?;
    pulse_module.add_class::<PyPulseSchedule>()?;
    pulse_module.add_class::<PyPulseCalibration>()?;
    pulse_module.add_class::<PyPulseBuilder>()?;
    pulse_module.add_class::<PyMeasLevel>()?;
    pulse_module.add_class::<PyPulseLibrary>()?;
    pulse_module.add_class::<PyPulseTemplates>()?;
    
    // Add constants
    pulse_module.add("DEFAULT_DT", 2.2222e-10)?; // ~0.22 ns
    
    parent.add_submodule(&pulse_module)?;
    
    // Also add to sys.modules to support `from quantrs2.pulse import ...`
    parent.py().import("sys")?
        .getattr("modules")?
        .set_item("quantrs2.pulse", &pulse_module)?;
    
    Ok(())
}