#!/usr/bin/env python3
"""
Demonstration of pulse-level control in QuantRS2

This example shows how to:
1. Create pulse shapes
2. Build pulse schedules
3. Use pre-built templates for gates and experiments
4. Create custom calibration data
"""

import quantrs2
from quantrs2.pulse import (
    PulseShape, Channel, PulseBuilder, PulseCalibration,
    PulseLibrary, PulseTemplates, MeasLevel
)
import matplotlib.pyplot as plt
import numpy as np


def demo_pulse_shapes():
    """Demonstrate different pulse shapes"""
    print("=== Pulse Shapes Demo ===")
    
    # Gaussian pulse
    gaussian = PulseShape.gaussian(duration=100e-9, sigma=25e-9, amplitude=0.5)
    print(f"Gaussian pulse: {gaussian}")
    
    # DRAG pulse (Gaussian with derivative removal)
    drag = PulseShape.gaussian_drag(
        duration=100e-9, sigma=25e-9, amplitude=0.5, beta=0.1
    )
    print(f"DRAG pulse: {drag}")
    
    # Square pulse
    square = PulseShape.square(duration=200e-9, amplitude=0.3)
    print(f"Square pulse: {square}")
    
    # Cosine-tapered pulse
    tapered = PulseShape.cosine_tapered(
        duration=200e-9, amplitude=0.4, rise_time=20e-9
    )
    print(f"Cosine-tapered pulse: {tapered}")
    
    # Arbitrary waveform
    t = np.linspace(0, 100e-9, 1000)
    samples = [(np.sin(2*np.pi*10e6*ti) * np.exp(-ti/50e-9), 0) for ti in t]
    arbitrary = PulseShape.arbitrary(samples, sample_rate=10e9)
    print(f"Arbitrary waveform: {arbitrary}")
    print()


def demo_channels():
    """Demonstrate channel types"""
    print("=== Channel Types Demo ===")
    
    # Drive channel for single-qubit gates
    drive0 = Channel.drive(0)
    print(f"Drive channel: {drive0}")
    
    # Measure channel
    measure0 = Channel.measure(0)
    print(f"Measure channel: {measure0}")
    
    # Control channel for two-qubit gates
    control01 = Channel.control(0, 1)
    print(f"Control channel: {control01}")
    
    # Readout and acquire channels
    readout0 = Channel.readout(0)
    acquire0 = Channel.acquire(0)
    print(f"Readout channel: {readout0}")
    print(f"Acquire channel: {acquire0}")
    print()


def demo_pulse_builder():
    """Demonstrate building custom pulse schedules"""
    print("=== Pulse Builder Demo ===")
    
    # Create calibration data
    cal = PulseCalibration(num_qubits=3, dt=2.2222e-10)
    
    # Build a custom gate sequence
    builder = PulseBuilder("custom_sequence", cal)
    
    # Add an X pulse on qubit 0
    x_pulse = PulseLibrary.x_pulse(0, cal)
    builder.play(Channel.drive(0), x_pulse)
    builder.delay(50, Channel.drive(0))  # 50 dt delay
    builder.play(Channel.drive(1), PulseLibrary.sx_pulse(1, cal))
    builder.barrier([Channel.drive(0), Channel.drive(1)])
    builder.play(Channel.measure(0), PulseLibrary.measure_pulse(0, cal))
    builder.play(Channel.measure(1), PulseLibrary.measure_pulse(1, cal))
    builder.with_metadata("experiment", "custom_demo")
    schedule = builder.build()
    
    print(f"Built schedule: {schedule}")
    print(f"Duration: {schedule.duration} dt")
    print(f"Number of instructions: {schedule.num_instructions}")
    print()


def demo_pulse_templates():
    """Demonstrate pre-built pulse templates"""
    print("=== Pulse Templates Demo ===")
    
    cal = PulseCalibration(num_qubits=5)
    
    # X gate
    x_schedule = PulseTemplates.x_gate(0, cal)
    print(f"X gate schedule: {x_schedule}")
    
    # CNOT gate
    cnot_schedule = PulseTemplates.cnot_gate(0, 1, cal)
    print(f"CNOT gate schedule: {cnot_schedule}")
    
    # Measurement
    meas_schedule = PulseTemplates.measure([0, 1, 2], cal)
    print(f"Measurement schedule: {meas_schedule}")
    print()


def demo_calibration_experiments():
    """Demonstrate calibration experiments"""
    print("=== Calibration Experiments Demo ===")
    
    cal = PulseCalibration(num_qubits=1)
    
    # Rabi experiment
    amplitudes = np.linspace(0, 1, 21)
    rabi_schedules = PulseTemplates.rabi_experiment(0, amplitudes.tolist(), cal)
    print(f"Rabi experiment: {len(rabi_schedules)} schedules")
    
    # T1 experiment
    delays = [0, 100, 200, 500, 1000, 2000, 5000]  # in dt units
    t1_schedules = PulseTemplates.t1_experiment(0, delays, cal)
    print(f"T1 experiment: {len(t1_schedules)} schedules")
    
    # Ramsey (T2) experiment
    delays = [0, 50, 100, 200, 400, 800]
    detuning = 1e6  # 1 MHz
    ramsey_schedules = PulseTemplates.ramsey_experiment(0, delays, detuning, cal)
    print(f"Ramsey experiment: {len(ramsey_schedules)} schedules")
    print()


def demo_custom_calibration():
    """Demonstrate custom calibration setup"""
    print("=== Custom Calibration Demo ===")
    
    # Create calibration for 3 qubits
    cal = PulseCalibration(num_qubits=3)
    
    # Customize qubit frequencies
    cal.set_qubit_frequency(0, 4.8)  # GHz
    cal.set_qubit_frequency(1, 4.9)
    cal.set_qubit_frequency(2, 5.0)
    
    # Customize measurement frequencies
    cal.set_meas_frequency(0, 6.3)  # GHz
    cal.set_meas_frequency(1, 6.4)
    cal.set_meas_frequency(2, 6.5)
    
    print(f"Sampling time (dt): {cal.dt} seconds")
    print(f"Qubit frequencies: {cal.qubit_frequencies} GHz")
    print(f"Measurement frequencies: {cal.meas_frequencies} GHz")
    print()


def demo_advanced_sequence():
    """Demonstrate an advanced pulse sequence"""
    print("=== Advanced Pulse Sequence Demo ===")
    
    cal = PulseCalibration(num_qubits=3)
    
    # Create a Bell state preparation sequence
    builder = PulseBuilder("bell_state_prep", cal)
    
    # Hadamard on qubit 0 (using two SX pulses and phase)
    sx_pulse = PulseLibrary.sx_pulse(0, cal)
    # H = Rz(π) SX Rz(π) = X SX
    builder.play(Channel.drive(0), PulseLibrary.x_pulse(0, cal))
    builder.play(Channel.drive(0), sx_pulse)
    # CNOT(0, 1)
    builder.play(Channel.control(0, 1), 
                 PulseShape.gaussian(560e-9, 140e-9, 0.3))
    builder.barrier([Channel.drive(0), Channel.drive(1)])
    # Measure both qubits
    builder.play(Channel.measure(0), PulseLibrary.measure_pulse(0, cal))
    builder.play(Channel.measure(1), PulseLibrary.measure_pulse(1, cal))
    builder.play(Channel.acquire(0), PulseShape.square(2e-6, 1.0))
    builder.play(Channel.acquire(1), PulseShape.square(2e-6, 1.0))
    schedule = builder.build()
    
    print(f"Bell state preparation: {schedule}")
    print()


def visualize_pulse_shape():
    """Visualize a pulse shape (requires matplotlib)"""
    print("=== Pulse Visualization ===")
    
    # Create time array
    t = np.linspace(0, 200e-9, 1000)
    
    # Gaussian pulse parameters
    duration = 200e-9
    sigma = 50e-9
    amplitude = 1.0
    
    # Calculate Gaussian envelope
    gauss = amplitude * np.exp(-0.5 * ((t - duration/2) / sigma)**2)
    
    # Calculate DRAG correction
    beta = 0.5
    drag_correction = -beta * (t - duration/2) / (sigma**2) * gauss
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t * 1e9, gauss, 'b-', label='Gaussian')
    plt.plot(t * 1e9, gauss + drag_correction, 'r--', label='DRAG')
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.title('Pulse Shapes')
    plt.legend()
    plt.grid(True)
    
    # Square pulse with cosine taper
    plt.subplot(2, 1, 2)
    square = np.ones_like(t) * 0.5
    rise_time = 20e-9
    rise_samples = int(rise_time / (t[1] - t[0]))
    # Apply cosine taper
    for i in range(rise_samples):
        square[i] *= 0.5 * (1 - np.cos(np.pi * i / rise_samples))
        square[-(i+1)] *= 0.5 * (1 - np.cos(np.pi * i / rise_samples))
    
    plt.plot(t * 1e9, square, 'g-', label='Cosine-tapered square')
    plt.xlabel('Time (ns)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pulse_shapes.png', dpi=150)
    print("Saved pulse visualization to 'pulse_shapes.png'")


def main():
    """Run all demonstrations"""
    print("QuantRS2 Pulse-Level Control Demonstration")
    print("==========================================\n")
    
    demo_pulse_shapes()
    demo_channels()
    demo_pulse_builder()
    demo_pulse_templates()
    demo_calibration_experiments()
    demo_custom_calibration()
    demo_advanced_sequence()
    
    try:
        visualize_pulse_shape()
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()