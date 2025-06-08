# Pulse-Level Control in QuantRS2

The pulse control module provides low-level access to quantum hardware control pulses, enabling fine-grained control over quantum operations. This is essential for:

- Hardware calibration experiments
- Custom gate implementations
- Noise characterization
- Optimal control sequences

## Features

- **Multiple pulse shapes**: Gaussian, DRAG, square, cosine-tapered, and arbitrary waveforms
- **Channel abstraction**: Drive, measure, control, readout, and acquire channels
- **Pulse scheduling**: Build complex pulse sequences with precise timing
- **Calibration management**: Store and use hardware-specific calibration data
- **Pre-built templates**: Standard gates and calibration experiments
- **Hardware backends**: Support for IBM Pulse and other providers

## Basic Usage

### Creating Pulse Shapes

```python
from quantrs2.pulse import PulseShape

# Gaussian pulse
gaussian = PulseShape.gaussian(duration=100e-9, sigma=25e-9, amplitude=0.5)

# DRAG pulse (Derivative Removal by Adiabatic Gate)
drag = PulseShape.gaussian_drag(
    duration=100e-9, sigma=25e-9, amplitude=0.5, beta=0.1
)

# Square pulse
square = PulseShape.square(duration=200e-9, amplitude=0.3)

# Cosine-tapered pulse
tapered = PulseShape.cosine_tapered(
    duration=200e-9, amplitude=0.4, rise_time=20e-9
)

# Arbitrary waveform
import numpy as np
t = np.linspace(0, 100e-9, 1000)
samples = [(np.sin(2*np.pi*10e6*ti), 0) for ti in t]
arbitrary = PulseShape.arbitrary(samples, sample_rate=10e9)
```

### Working with Channels

```python
from quantrs2.pulse import Channel

# Drive channel for single-qubit gates
drive0 = Channel.drive(0)

# Control channel for two-qubit gates
control01 = Channel.control(0, 1)

# Measurement channels
measure0 = Channel.measure(0)
acquire0 = Channel.acquire(0)
```

### Building Pulse Schedules

```python
from quantrs2.pulse import PulseBuilder, PulseCalibration, PulseLibrary

# Create calibration
cal = PulseCalibration(num_qubits=2)

# Build a custom schedule
builder = PulseBuilder("my_gate", cal)

# Add pulses
builder.play(Channel.drive(0), PulseLibrary.x_pulse(0, cal))
builder.delay(50, Channel.drive(0))  # Delay in dt units
builder.play(Channel.drive(1), PulseLibrary.sx_pulse(1, cal))

# Synchronize channels
builder.barrier([Channel.drive(0), Channel.drive(1)])

# Add measurement
builder.play(Channel.measure(0), PulseLibrary.measure_pulse(0, cal))
builder.play(Channel.measure(1), PulseLibrary.measure_pulse(1, cal))

# Build the schedule
schedule = builder.build()
```

### Using Pre-built Templates

```python
from quantrs2.pulse import PulseTemplates

# Standard gates
x_schedule = PulseTemplates.x_gate(0, cal)
cnot_schedule = PulseTemplates.cnot_gate(0, 1, cal)

# Measurement
meas_schedule = PulseTemplates.measure([0, 1, 2], cal)
```

## Calibration Experiments

### Rabi Oscillation

Characterize the drive amplitude needed for a π rotation:

```python
import numpy as np

# Scan amplitudes from 0 to 1
amplitudes = np.linspace(0, 1, 21)
rabi_schedules = PulseTemplates.rabi_experiment(0, amplitudes.tolist(), cal)

# Execute schedules on hardware
for schedule in rabi_schedules:
    # backend.execute_pulse_schedule(schedule, shots=1024)
    pass
```

### T1 Relaxation

Measure qubit energy relaxation time:

```python
# Delays in dt units
delays = [0, 100, 200, 500, 1000, 2000, 5000]
t1_schedules = PulseTemplates.t1_experiment(0, delays, cal)
```

### Ramsey (T2) Experiment

Measure qubit dephasing time:

```python
delays = [0, 50, 100, 200, 400, 800]
detuning = 1e6  # 1 MHz off-resonance
ramsey_schedules = PulseTemplates.ramsey_experiment(0, delays, detuning, cal)
```

## Custom Calibration

```python
# Create calibration for specific hardware
cal = PulseCalibration(num_qubits=5)

# Set qubit frequencies (GHz)
cal.set_qubit_frequency(0, 4.8)
cal.set_qubit_frequency(1, 4.9)
cal.set_qubit_frequency(2, 5.0)

# Set measurement frequencies (GHz)
cal.set_meas_frequency(0, 6.3)
cal.set_meas_frequency(1, 6.4)
cal.set_meas_frequency(2, 6.5)

# Access calibration data
print(f"Sampling time: {cal.dt} seconds")
print(f"Qubit frequencies: {cal.qubit_frequencies}")
```

## Advanced Example: Bell State Preparation

```python
# Create Bell state using pulses
builder = PulseBuilder("bell_state", cal)

# Hadamard on qubit 0 (implemented as Rz(π) SX Rz(π))
builder.set_phase(Channel.drive(0), np.pi)
builder.play(Channel.drive(0), PulseLibrary.sx_pulse(0, cal))
builder.set_phase(Channel.drive(0), np.pi)

# CNOT using cross-resonance
cr_pulse = PulseShape.gaussian(560e-9, 140e-9, 0.3)
builder.play(Channel.control(0, 1), cr_pulse)

# Simultaneous single-qubit corrections
builder.play(Channel.drive(0), PulseLibrary.x_pulse(0, cal))
builder.play(Channel.drive(1), PulseLibrary.x_pulse(1, cal))

# Measure both qubits
builder.barrier([Channel.drive(0), Channel.drive(1)])
builder.play(Channel.measure(0), PulseLibrary.measure_pulse(0, cal))
builder.play(Channel.measure(1), PulseLibrary.measure_pulse(1, cal))

bell_schedule = builder.build()
```

## Hardware Backend Integration

Currently, the pulse control module provides the data structures and scheduling capabilities. Hardware execution requires integration with specific quantum providers:

- **IBM Quantum**: Use with Qiskit Pulse backend
- **AWS Braket**: Pulse control support varies by device
- **Other providers**: Check provider documentation for pulse-level access

Example integration pattern:

```python
# Convert QuantRS2 schedule to provider format
def to_qiskit_schedule(quantrs_schedule):
    # Implementation depends on provider
    pass

# Execute on hardware
# backend.run(to_qiskit_schedule(schedule), shots=1024)
```

## Best Practices

1. **Calibration**: Always use up-to-date calibration data from the hardware
2. **Timing**: Be aware of hardware timing constraints (minimum pulse duration, dt resolution)
3. **Power limits**: Respect amplitude constraints to avoid hardware damage
4. **Phase tracking**: Maintain consistent phase references across pulses
5. **Measurement**: Allow sufficient time for readout resonator ring-down

## Future Enhancements

- Direct hardware backend execution
- Pulse optimization algorithms
- Automated calibration routines
- Pulse visualization tools
- Integration with optimal control libraries

## API Reference

See the [API documentation](https://quantrs2.readthedocs.io/en/latest/api/pulse.html) for detailed class and method descriptions.