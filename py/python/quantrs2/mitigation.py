"""Quantum error mitigation techniques.

This module provides various error mitigation methods to reduce the impact
of noise in quantum computations:

- Zero-Noise Extrapolation (ZNE): Extrapolate to the zero-noise limit
- Probabilistic Error Cancellation (PEC): Cancel errors probabilistically
- Virtual Distillation: Purify quantum states virtually
- Symmetry Verification: Verify and enforce symmetries

Example:
    Basic ZNE usage::

        from quantrs2.mitigation import ZeroNoiseExtrapolation, ZNEConfig, Observable
        from quantrs2 import Circuit
        
        # Create circuit
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        
        # Configure ZNE
        config = ZNEConfig(
            scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
            scaling_method="global",
            extrapolation_method="richardson"
        )
        
        # Create ZNE executor
        zne = ZeroNoiseExtrapolation(config)
        
        # Define observable
        observable = Observable.z(0)
        
        # Run circuits at different noise scales and collect measurements
        measurements = []
        for scale in config.scale_factors:
            # In practice, fold circuit and execute on hardware
            folded_circuit = zne.fold_circuit(circuit, scale)
            # result = backend.execute(folded_circuit, shots=1024)
            # measurements.append((scale, result))
        
        # Extrapolate to zero noise
        # mitigated_result = zne.mitigate_observable(observable, measurements)

Classes:
    ZNEConfig: Configuration for Zero-Noise Extrapolation
    ZNEResult: Result from ZNE including mitigated value and error estimate
    Observable: Observable for expectation value calculation
    ZeroNoiseExtrapolation: Main ZNE executor
    CircuitFolding: Circuit folding utilities
    ExtrapolationFitting: Extrapolation fitting utilities
    ProbabilisticErrorCancellation: PEC implementation (placeholder)
    VirtualDistillation: Virtual distillation (placeholder)
    SymmetryVerification: Symmetry verification (placeholder)
"""

from quantrs2._quantrs2.mitigation import (
    ZNEConfig,
    ZNEResult,
    Observable,
    ZeroNoiseExtrapolation,
    CircuitFolding,
    ExtrapolationFitting,
    ProbabilisticErrorCancellation,
    VirtualDistillation,
    SymmetryVerification,
)

__all__ = [
    "ZNEConfig",
    "ZNEResult",
    "Observable",
    "ZeroNoiseExtrapolation",
    "CircuitFolding",
    "ExtrapolationFitting",
    "ProbabilisticErrorCancellation",
    "VirtualDistillation",
    "SymmetryVerification",
]