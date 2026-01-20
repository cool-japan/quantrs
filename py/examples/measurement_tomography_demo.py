#!/usr/bin/env python3
"""
Demonstration of measurement statistics and quantum tomography.

This example shows how to:
1. Sample measurement outcomes from quantum circuits
2. Analyze measurement statistics and correlations
3. Perform quantum state tomography
4. Perform quantum process tomography
5. Apply measurement error mitigation
"""

import numpy as np
from quantrs2 import Circuit, SimulationResult
from quantrs2.measurement import (
    MeasurementSampler, StateTomography, ProcessTomography
)

def measurement_statistics_example():
    """Example of sampling and analyzing measurement outcomes."""
    print("=== Measurement Statistics Example ===")
    
    # Create a Bell state
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    
    # Run circuit to get state vector
    result = circuit.run()
    
    # Create measurement sampler
    sampler = MeasurementSampler()
    
    # Sample measurements (1000 shots)
    measurement = sampler.sample_counts(result, shots=1000)
    
    print("Measurement counts:", measurement.get_counts())
    print("Measurement probabilities:", measurement.get_probabilities())
    print("Most probable outcome:", measurement.most_probable())
    
    # Analyze marginal probabilities
    print(f"\nMarginal probability of qubit 0 being |1⟩: {measurement.marginal_probability(0):.3f}")
    print(f"Marginal probability of qubit 1 being |1⟩: {measurement.marginal_probability(1):.3f}")
    
    # Analyze correlations
    print(f"Correlation between qubits 0 and 1: {measurement.correlation(0, 1):.3f}")
    print("(Perfect correlation = ±1, no correlation = 0)")


def measurement_with_error_example():
    """Example of measurements with readout error."""
    print("\n=== Measurement with Readout Error ===")
    
    # Create a simple state |0⟩
    circuit = Circuit(3)
    # Don't apply any gates - state remains |000⟩
    
    result = circuit.run()
    sampler = MeasurementSampler()
    
    # Sample with 2% readout error
    error_rate = 0.02
    measurement = sampler.sample_with_error(result, shots=1000, error_rate=error_rate)
    
    print(f"Measurements with {error_rate*100}% readout error:")
    print("Counts:", measurement.get_counts())
    print("Expected '000' with perfect readout, but errors flip some bits")
    
    # Create error calibration matrix (simplified)
    n_states = 2**3
    error_matrix = np.eye(n_states) * (1 - error_rate) + \
                   np.ones((n_states, n_states)) * (error_rate / n_states)
    
    # Apply error mitigation
    mitigated = measurement.mitigate_errors(error_matrix)
    print("\nAfter error mitigation:")
    print("Counts:", mitigated.get_counts())


def state_tomography_example():
    """Example of quantum state tomography."""
    print("\n=== Quantum State Tomography ===")
    
    # Create a state to reconstruct (|ψ⟩ = |+⟩)
    circuit = Circuit(1)
    circuit.h(0)
    
    # Create tomography object
    tomography = StateTomography(n_qubits=1)
    
    # Generate measurement circuits
    measurement_circuits = tomography.measurement_circuits()
    print(f"Number of measurement bases: {len(measurement_circuits)}")
    
    # Simulate measurements for each basis
    sampler = MeasurementSampler()
    measurements = []
    
    for circuit_info in measurement_circuits:
        basis = circuit_info['basis']
        meas_circuit = circuit_info['circuit']
        
        # Prepare the state and apply measurement basis
        full_circuit = Circuit(1)
        full_circuit.h(0)  # Prepare |+⟩
        
        # Apply basis transformation from measurement circuit
        # (In practice, would compose circuits properly)
        
        # Run and sample
        result = full_circuit.run()
        measurement = sampler.sample_counts(result, shots=1000)
        
        measurements.append({
            'basis': basis,
            'result': measurement
        })
        
        print(f"Basis {basis}: {measurement.get_counts()}")
    
    # Reconstruct density matrix
    rho = tomography.reconstruct_state(measurements)
    print(f"\nReconstructed density matrix shape: {rho.shape}")
    
    # Calculate fidelity with expected state
    # |+⟩⟨+| = [[0.5, 0.5], [0.5, 0.5]]
    expected_rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
    fidelity = tomography.fidelity(rho, expected_rho)
    print(f"Fidelity with expected |+⟩ state: {fidelity:.3f}")


def process_tomography_example():
    """Example of quantum process tomography."""
    print("\n=== Quantum Process Tomography ===")
    
    # Create process tomography object for 1-qubit processes
    proc_tomo = ProcessTomography(n_qubits=1)
    
    # Generate input states
    input_states = proc_tomo.input_states()
    print(f"Number of input states: {len(input_states)}")
    
    # Show the input states
    print("Input states for process tomography:")
    for state_info in input_states[:6]:  # Show first 6
        print(f"  State: |{state_info['state']}⟩")
    
    # The process we want to characterize (e.g., Hadamard gate)
    def apply_process(circuit):
        """Apply the quantum process (Hadamard gate)."""
        circuit.h(0)
        return circuit
    
    # Collect tomography data
    tomography_data = {}
    state_tomo = StateTomography(n_qubits=1)
    sampler = MeasurementSampler()
    
    for input_info in input_states[:2]:  # Simplified: only use first 2 states
        input_state = input_info['state']
        prep_circuit = input_info['circuit']
        
        # Apply the process
        process_circuit = Circuit(1)
        # Copy preparation (simplified)
        if input_state == '1':
            process_circuit.x(0)
        elif input_state == '+':
            process_circuit.h(0)
        
        # Apply the process being characterized
        process_circuit = apply_process(process_circuit)
        
        # Perform state tomography on output
        result = process_circuit.run()
        measurement = sampler.sample_counts(result, shots=1000)
        
        tomography_data[input_state] = measurement
    
    # Reconstruct process matrix (chi matrix)
    chi_matrix = proc_tomo.reconstruct_process(tomography_data)
    print(f"\nReconstructed chi matrix shape: {chi_matrix.shape}")
    
    # Compare with ideal Hadamard process
    # (In practice, would compute the ideal chi matrix)
    print("Process tomography completed")


def ghz_state_analysis():
    """Analyze correlations in a GHZ state."""
    print("\n=== GHZ State Correlation Analysis ===")
    
    # Create 3-qubit GHZ state: |000⟩ + |111⟩
    circuit = Circuit(3)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.cnot(1, 2)
    
    result = circuit.run()
    sampler = MeasurementSampler()
    measurement = sampler.sample_counts(result, shots=5000)
    
    print("GHZ state measurement outcomes:")
    print(measurement.get_counts())
    
    # Analyze pairwise correlations
    print("\nPairwise correlations:")
    for i in range(3):
        for j in range(i+1, 3):
            corr = measurement.correlation(i, j)
            print(f"Qubits {i}-{j}: {corr:.3f}")
    
    print("\nGHZ states show perfect correlation between all qubit pairs")


def w_state_analysis():
    """Analyze correlations in a W state."""
    print("\n=== W State Analysis ===")
    
    # Create approximate W state: |001⟩ + |010⟩ + |100⟩
    # Using simple construction (not exact W state)
    circuit = Circuit(3)
    
    # Create superposition
    circuit.h(0)
    circuit.h(1)
    
    # Entangle
    circuit.cnot(0, 2)
    circuit.cnot(1, 2)
    
    result = circuit.run()
    sampler = MeasurementSampler()
    measurement = sampler.sample_counts(result, shots=5000)
    
    print("W-like state measurement outcomes:")
    counts = measurement.get_counts()
    # Sort by count for better display
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    for outcome, count in list(sorted_counts.items())[:5]:
        print(f"  {outcome}: {count}")
    
    # Analyze marginal probabilities
    print("\nMarginal probabilities (each should be ~1/3 for ideal W state):")
    for i in range(3):
        prob = measurement.marginal_probability(i)
        print(f"Qubit {i}: {prob:.3f}")


if __name__ == "__main__":
    measurement_statistics_example()
    measurement_with_error_example()
    state_tomography_example()
    process_tomography_example()
    ghz_state_analysis()
    w_state_analysis()
    
    print("\n=== Measurement and Tomography Demo Complete ===")