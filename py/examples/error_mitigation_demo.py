#!/usr/bin/env python3
"""Demonstration of quantum error mitigation techniques.

This example shows how to use Zero-Noise Extrapolation (ZNE) and other
error mitigation methods to improve the accuracy of quantum computations
in the presence of noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantrs2 import Circuit
from quantrs2.mitigation import (
    ZNEConfig, Observable, ZeroNoiseExtrapolation,
    CircuitFolding, ExtrapolationFitting
)
from quantrs2.measurement import MeasurementSampler


def create_test_circuit():
    """Create a simple test circuit that prepares a GHZ state."""
    circuit = Circuit(3)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)
    return circuit


def simulate_noisy_execution(circuit, noise_level, shots=1024):
    """Simulate noisy circuit execution (placeholder for real hardware)."""
    # In practice, this would execute on real hardware with natural noise
    # Here we simulate by adding depolarizing noise
    
    # Run circuit
    result = circuit.run()
    
    # Sample with simulated readout error
    sampler = MeasurementSampler()
    measurement = sampler.sample_with_error(result, shots, error_rate=0.01 * noise_level)
    
    return measurement


def demo_zero_noise_extrapolation():
    """Demonstrate Zero-Noise Extrapolation."""
    print("=" * 60)
    print("Zero-Noise Extrapolation (ZNE) Demo")
    print("=" * 60)
    
    # Create circuit
    circuit = create_test_circuit()
    print(f"\nCreated {circuit.num_qubits}-qubit GHZ state circuit")
    
    # Configure ZNE
    config = ZNEConfig(
        scale_factors=[1.0, 1.5, 2.0, 2.5, 3.0],
        scaling_method="global",
        extrapolation_method="richardson",
        bootstrap_samples=100,
        confidence_level=0.95
    )
    print(f"\nZNE Configuration:")
    print(f"  Scale factors: {config.scale_factors}")
    print(f"  Scaling method: {config.scaling_method}")
    print(f"  Extrapolation: {config.extrapolation_method}")
    
    # Create ZNE executor
    zne = ZeroNoiseExtrapolation(config)
    
    # Define observable (measure parity of all qubits)
    observable = Observable([(0, "Z"), (1, "Z"), (2, "Z")], coefficient=1.0)
    print(f"\nObservable: {observable}")
    
    # Collect measurements at different noise scales
    print("\nCollecting measurements at different noise scales...")
    measurements = []
    expectation_values = []
    
    for scale in config.scale_factors:
        # Apply circuit folding (placeholder - would fold in practice)
        folded_circuit = zne.fold_circuit(circuit, scale)
        
        # Simulate noisy execution
        result = simulate_noisy_execution(folded_circuit, scale, shots=4096)
        measurements.append((scale, result))
        
        # Calculate expectation value
        exp_val = observable.expectation_value(result)
        expectation_values.append((scale, exp_val))
        print(f"  Scale {scale:.1f}: <O> = {exp_val:.4f}")
    
    # Extrapolate to zero noise
    print("\nExtrapolating to zero noise...")
    mitigated_result = zne.mitigate_observable(observable, measurements)
    
    print(f"\nResults:")
    print(f"  Mitigated value: {mitigated_result.mitigated_value:.4f}")
    if mitigated_result.error_estimate:
        print(f"  Error estimate: ±{mitigated_result.error_estimate:.4f}")
    print(f"  R² (goodness of fit): {mitigated_result.r_squared:.4f}")
    print(f"  Extrapolation function: {mitigated_result.extrapolation_fn}")
    
    # Plot results
    plot_zne_results(expectation_values, mitigated_result)
    
    return mitigated_result


def demo_circuit_folding():
    """Demonstrate different circuit folding techniques."""
    print("\n" + "=" * 60)
    print("Circuit Folding Demo")
    print("=" * 60)
    
    circuit = create_test_circuit()
    print(f"\nOriginal circuit has {circuit.num_gates} gates")
    
    # Global folding
    print("\nGlobal folding:")
    for scale in [1.0, 2.0, 3.0]:
        folded = CircuitFolding.fold_global(circuit, scale)
        print(f"  Scale {scale}: {folded.num_gates} gates")
    
    # Local folding with custom weights
    print("\nLocal folding (placeholder):")
    weights = [1.0, 0.5, 2.0]  # Different weights for different gates
    folded = CircuitFolding.fold_local(circuit, 2.0, weights)
    print(f"  With custom weights: {folded.num_gates} gates")


def demo_extrapolation_methods():
    """Compare different extrapolation methods."""
    print("\n" + "=" * 60)
    print("Extrapolation Methods Comparison")
    print("=" * 60)
    
    # Generate synthetic data with known zero-noise value
    true_value = 0.85
    noise_rate = 0.1
    scale_factors = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    
    # Exponential decay model: y = a * exp(-b * x)
    values = true_value * np.exp(-noise_rate * (scale_factors - 1))
    # Add some noise
    values += np.random.normal(0, 0.01, len(values))
    
    print(f"\nTrue zero-noise value: {true_value}")
    print(f"Data points: {list(zip(scale_factors, values))}")
    
    # Try different extrapolation methods
    methods = [
        ("Linear", ExtrapolationFitting.fit_linear),
        ("Polynomial (2nd order)", lambda x, y: ExtrapolationFitting.fit_polynomial(x, y, 2)),
        ("Exponential", ExtrapolationFitting.fit_exponential),
        ("Richardson", ExtrapolationFitting.fit_richardson),
        ("Adaptive", ExtrapolationFitting.fit_adaptive),
    ]
    
    results = []
    for name, fit_func in methods:
        try:
            result = fit_func(scale_factors, values)
            results.append((name, result))
            print(f"\n{name}:")
            print(f"  Mitigated value: {result.mitigated_value:.4f}")
            print(f"  Error: {abs(result.mitigated_value - true_value):.4f}")
            print(f"  R²: {result.r_squared:.4f}")
        except Exception as e:
            print(f"\n{name}: Failed - {e}")
    
    # Plot comparison
    plot_extrapolation_comparison(scale_factors, values, results, true_value)


def plot_zne_results(data_points, zne_result):
    """Plot ZNE results."""
    plt.figure(figsize=(8, 6))
    
    # Plot data points
    scales, values = zip(*data_points)
    plt.scatter(scales, values, s=100, label='Measured', zorder=3)
    
    # Plot extrapolation
    x_fit = np.linspace(0, max(scales), 100)
    # Simple linear visualization (actual fit may be different)
    if len(zne_result.fit_params) >= 2:
        y_fit = zne_result.fit_params[0] + zne_result.fit_params[1] * x_fit
        plt.plot(x_fit, y_fit, 'r--', label='Extrapolation', alpha=0.7)
    
    # Mark zero-noise extrapolated value
    plt.scatter([0], [zne_result.mitigated_value], s=200, c='red', 
                marker='*', label=f'Mitigated: {zne_result.mitigated_value:.4f}', zorder=4)
    
    # Error bars if available
    if zne_result.error_estimate:
        plt.errorbar([0], [zne_result.mitigated_value], 
                    yerr=zne_result.error_estimate, 
                    fmt='none', c='red', capsize=5)
    
    plt.xlabel('Noise Scale Factor')
    plt.ylabel('Expectation Value')
    plt.title('Zero-Noise Extrapolation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.2, max(scales) + 0.2)
    
    plt.tight_layout()
    plt.show()


def plot_extrapolation_comparison(x, y, results, true_value):
    """Plot comparison of extrapolation methods."""
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(x, y, s=100, label='Data', zorder=3)
    
    # Plot each extrapolation
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    x_fit = np.linspace(0, max(x), 100)
    
    for (name, result), color in zip(results, colors):
        # Simple visualization (varies by method)
        if "Linear" in name and len(result.fit_params) >= 2:
            y_fit = result.fit_params[0] + result.fit_params[1] * x_fit
            plt.plot(x_fit, y_fit, '--', color=color, alpha=0.7)
        
        # Mark extrapolated value
        plt.scatter([0], [result.mitigated_value], s=150, c=color, 
                   marker='s', label=f'{name}: {result.mitigated_value:.4f}')
    
    # Mark true value
    plt.axhline(y=true_value, color='black', linestyle=':', 
                label=f'True value: {true_value:.4f}')
    
    plt.xlabel('Noise Scale Factor')
    plt.ylabel('Value')
    plt.title('Extrapolation Methods Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.2, max(x) + 0.2)
    
    plt.tight_layout()
    plt.show()


def demo_observable_measurements():
    """Demonstrate different observable measurements."""
    print("\n" + "=" * 60)
    print("Observable Measurements Demo")
    print("=" * 60)
    
    # Create a simple entangled state
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    
    # Different observables
    observables = [
        (Observable.z(0), "Z₀ (first qubit)"),
        (Observable.z(1), "Z₁ (second qubit)"),
        (Observable.zz(0, 1), "Z₀Z₁ (correlation)"),
        (Observable([(0, "X"), (1, "X")], 1.0), "X₀X₁"),
        (Observable([(0, "Y"), (1, "Y")], 1.0), "Y₀Y₁"),
    ]
    
    # Measure each observable
    result = circuit.run()
    sampler = MeasurementSampler()
    measurement = sampler.sample_counts(result, 10000)
    
    print("\nObservable expectation values:")
    for obs, name in observables:
        exp_val = obs.expectation_value(measurement)
        print(f"  <{name}> = {exp_val:+.4f}")


def main():
    """Run all demonstrations."""
    print("Quantum Error Mitigation Demonstration")
    print("=" * 60)
    
    # Run demos
    demo_zero_noise_extrapolation()
    demo_circuit_folding()
    demo_extrapolation_methods()
    demo_observable_measurements()
    
    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == "__main__":
    main()