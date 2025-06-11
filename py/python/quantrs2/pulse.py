"""Pulse-level control for quantum operations.

This module provides low-level pulse control for quantum operations,
enabling fine-grained control over quantum gates and measurements.

Example:
    Basic pulse creation and scheduling::

        from quantrs2.pulse import PulseShape, Channel, PulseBuilder, PulseCalibration

        # Create calibration
        cal = PulseCalibration(num_qubits=2)
        
        # Build a pulse schedule
        builder = PulseBuilder("my_schedule", cal)
        builder.play(Channel.drive(0), PulseShape.gaussian(100e-9, 25e-9, 0.5))
        builder.play(Channel.drive(1), PulseShape.square(200e-9, 0.3))
        schedule = builder.build()

Classes:
    PulseShape: Different pulse waveform shapes
    Channel: Different channel types (drive, measure, control, etc.)
    PulseCalibration: Hardware calibration data
    PulseBuilder: Builder for creating pulse schedules
    PulseSchedule: Collection of pulse instructions
    PulseLibrary: Standard pulse shapes for common gates
    PulseTemplates: Pre-built schedules for gates and experiments
    MeasLevel: Measurement level (raw, kerneled, classified)
"""

try:
    from quantrs2._quantrs2.pulse import (
        PulseShape,
        Channel,
        PulseSchedule,
        PulseCalibration,
        PulseBuilder,
        MeasLevel,
        PulseLibrary,
        PulseTemplates,
        DEFAULT_DT,
    )
except ImportError:
    # Fallback implementations for testing
    from typing import Optional, Dict, Any, List
    
    DEFAULT_DT = 1e-9
    
    class PulseShape:
        @staticmethod
        def gaussian(duration: float, sigma: float, amplitude: float):
            return "gaussian"
            
        @staticmethod 
        def square(duration: float, amplitude: float):
            return "square"
            
    class Channel:
        @staticmethod
        def drive(qubit: int):
            return f"drive_{qubit}"
            
        @staticmethod
        def measure(qubit: int):
            return f"measure_{qubit}"
            
    class PulseSchedule:
        def __init__(self):
            self.pulses = []
            
    class PulseCalibration:
        def __init__(self, num_qubits: int):
            self.num_qubits = num_qubits
            
    class PulseBuilder:
        def __init__(self, name: str, calibration: PulseCalibration):
            self.name = name
            self.calibration = calibration
            self.schedule = PulseSchedule()
            
        def play(self, channel, pulse):
            self.schedule.pulses.append((channel, pulse))
            
        def build(self):
            return self.schedule
            
    class MeasLevel:
        RAW = 0
        KERNELED = 1
        CLASSIFIED = 2
        
    class PulseLibrary:
        pass
        
    class PulseTemplates:
        pass

__all__ = [
    "PulseShape",
    "Channel",
    "PulseSchedule",
    "PulseCalibration",
    "PulseBuilder",
    "MeasLevel",
    "PulseLibrary",
    "PulseTemplates",
    "DEFAULT_DT",
]