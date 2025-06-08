"""Tests for pulse-level control functionality"""

import pytest
import quantrs2
from quantrs2.pulse import (
    PulseShape, Channel, PulseBuilder, PulseCalibration,
    PulseLibrary, PulseTemplates, MeasLevel
)


def test_pulse_shapes():
    """Test creation of different pulse shapes"""
    # Gaussian pulse
    gaussian = PulseShape.gaussian(duration=100e-9, sigma=25e-9, amplitude=0.5)
    assert gaussian.pulse_type == "gaussian"
    assert gaussian.duration == 100e-9
    
    # DRAG pulse
    drag = PulseShape.gaussian_drag(
        duration=100e-9, sigma=25e-9, amplitude=0.5, beta=0.1
    )
    assert drag.pulse_type == "gaussian_drag"
    
    # Square pulse
    square = PulseShape.square(duration=200e-9, amplitude=0.3)
    assert square.pulse_type == "square"
    assert square.duration == 200e-9
    
    # Cosine-tapered pulse
    tapered = PulseShape.cosine_tapered(
        duration=200e-9, amplitude=0.4, rise_time=20e-9
    )
    assert tapered.pulse_type == "cosine_tapered"
    
    # Arbitrary waveform
    samples = [(0.1, 0.0), (0.2, 0.1), (0.3, 0.0)]
    arbitrary = PulseShape.arbitrary(samples, sample_rate=1e9)
    assert arbitrary.pulse_type == "arbitrary"
    assert arbitrary.duration == 3e-9  # 3 samples at 1 GHz


def test_channels():
    """Test channel creation"""
    # Drive channel
    drive = Channel.drive(0)
    assert drive.channel_type == "drive"
    assert drive.qubits == [0]
    
    # Control channel
    control = Channel.control(0, 1)
    assert control.channel_type == "control"
    assert control.qubits == [0, 1]
    
    # Measure channel
    measure = Channel.measure(2)
    assert measure.channel_type == "measure"
    assert measure.qubits == [2]


def test_pulse_calibration():
    """Test calibration creation and modification"""
    cal = PulseCalibration(num_qubits=3)
    
    # Check defaults
    assert len(cal.qubit_frequencies) == 3
    assert len(cal.meas_frequencies) == 3
    assert cal.dt == 2.2222e-10
    
    # Modify frequencies
    cal.set_qubit_frequency(0, 4.5)
    cal.set_meas_frequency(0, 6.0)
    
    assert cal.qubit_frequencies[0] == 4.5
    assert cal.meas_frequencies[0] == 6.0
    
    # Check bounds
    with pytest.raises(ValueError):
        cal.set_qubit_frequency(10, 5.0)  # Out of range


def test_pulse_builder():
    """Test pulse schedule building"""
    cal = PulseCalibration(num_qubits=2)
    
    # Build a simple schedule
    builder = PulseBuilder("test_schedule", cal)
    x_pulse = PulseLibrary.x_pulse(0, cal)
    
    builder.play(Channel.drive(0), x_pulse)
    builder.delay(100, Channel.drive(0))
    builder.play(Channel.drive(1), PulseLibrary.sx_pulse(1, cal))
    builder.with_metadata("test", "value")
    schedule = builder.build()
    
    assert schedule.name == "test_schedule"
    assert schedule.num_instructions > 0
    assert schedule.duration > 0
    assert "test" in schedule.metadata


def test_pulse_library():
    """Test pulse library functions"""
    cal = PulseCalibration(num_qubits=2)
    
    # X pulse
    x_pulse = PulseLibrary.x_pulse(0, cal)
    assert x_pulse.pulse_type == "gaussian"
    
    # Y pulse
    y_pulse = PulseLibrary.y_pulse(0, cal)
    assert y_pulse.pulse_type == "gaussian"
    
    # SX pulse
    sx_pulse = PulseLibrary.sx_pulse(0, cal)
    assert sx_pulse.pulse_type == "gaussian"
    
    # Measure pulse
    meas_pulse = PulseLibrary.measure_pulse(0, cal)
    assert meas_pulse.pulse_type == "square"


def test_pulse_templates():
    """Test pre-built pulse templates"""
    cal = PulseCalibration(num_qubits=3)
    
    # X gate
    x_schedule = PulseTemplates.x_gate(0, cal)
    assert x_schedule.name == "x_gate"
    assert x_schedule.num_instructions > 0
    
    # CNOT gate
    cnot_schedule = PulseTemplates.cnot_gate(0, 1, cal)
    assert cnot_schedule.name == "cnot_gate"
    
    # Measurement
    meas_schedule = PulseTemplates.measure([0, 1], cal)
    assert meas_schedule.name == "measure"


def test_calibration_experiments():
    """Test calibration experiment generation"""
    cal = PulseCalibration(num_qubits=1)
    
    # Rabi experiment
    amplitudes = [0.1, 0.2, 0.3, 0.4, 0.5]
    rabi_schedules = PulseTemplates.rabi_experiment(0, amplitudes, cal)
    assert len(rabi_schedules) == 5
    
    # T1 experiment
    delays = [0, 100, 200, 500]
    t1_schedules = PulseTemplates.t1_experiment(0, delays, cal)
    assert len(t1_schedules) == 4
    
    # Ramsey experiment
    ramsey_schedules = PulseTemplates.ramsey_experiment(0, delays, 1e6, cal)
    assert len(ramsey_schedules) == 4


def test_builder_chaining():
    """Test builder method chaining"""
    cal = PulseCalibration(num_qubits=3)
    builder = PulseBuilder("chain_test", cal)
    
    # Test chaining multiple operations
    pulse1 = PulseShape.gaussian(100e-9, 25e-9, 0.5)
    pulse2 = PulseShape.square(200e-9, 0.3)
    
    builder.play(Channel.drive(0), pulse1)
    builder.play(Channel.drive(1), pulse2)
    builder.barrier([Channel.drive(0), Channel.drive(1)])
    builder.set_phase(Channel.drive(0), 1.57)
    builder.set_frequency(Channel.drive(1), 100e6)
    schedule = builder.build()
    
    assert schedule.num_instructions >= 4


def test_phase_and_frequency():
    """Test phase and frequency operations"""
    cal = PulseCalibration(num_qubits=1)
    builder = PulseBuilder("phase_freq_test", cal)
    
    builder.set_phase(Channel.drive(0), 3.14159)
    builder.set_frequency(Channel.drive(0), 50e6)
    builder.play(Channel.drive(0), PulseShape.gaussian(100e-9, 25e-9, 0.5))
    schedule = builder.build()
    
    assert schedule.num_instructions == 3


def test_repr_methods():
    """Test string representations"""
    # Test pulse shape repr
    pulse = PulseShape.gaussian(100e-9, 25e-9, 0.5)
    assert "gaussian" in repr(pulse).lower()
    assert "100" in repr(pulse) or "1.00e-07" in repr(pulse)
    
    # Test channel repr
    channel = Channel.drive(0)
    assert "drive(0)" in repr(channel)
    
    # Test schedule repr
    cal = PulseCalibration(num_qubits=1)
    schedule = PulseTemplates.x_gate(0, cal)
    assert "PulseSchedule" in repr(schedule)
    assert "x_gate" in repr(schedule)