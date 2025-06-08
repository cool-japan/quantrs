#!/usr/bin/env python3
"""
Demonstration of parametric quantum circuits with automatic differentiation.

This example shows how to:
1. Create parametric quantum circuits
2. Compute gradients using different methods
3. Optimize circuit parameters
4. Build variational quantum algorithms
"""

import numpy as np
from quantrs2 import ParametricCircuit, CircuitOptimizer, ExpectationValue

def vqe_example():
    """Variational Quantum Eigensolver (VQE) example."""
    print("=== VQE Example ===")
    
    # Create a 2-qubit parametric circuit
    circuit = ParametricCircuit(2)
    
    # Add parametric gates
    circuit.ry(0, "theta1", initial_value=0.5)
    circuit.ry(1, "theta2", initial_value=0.5)
    circuit.rx(0, "phi1", initial_value=0.2)
    circuit.rx(1, "phi2", initial_value=0.2)
    
    # Define Hamiltonian (H = 0.5 * Z0 + 0.5 * Z1 + 0.5 * X0X1)
    hamiltonian = [
        (0.5, "ZI"),
        (0.5, "IZ"),
        (0.5, "XX")
    ]
    expectation = ExpectationValue(hamiltonian, 2)
    
    # Create optimizer
    optimizer = CircuitOptimizer(learning_rate=0.1)
    
    # Training loop (simplified)
    print("Initial parameters:", circuit.get_parameters())
    
    for step in range(10):
        # Forward pass (would run quantum circuit in reality)
        # For now, we'll simulate with dummy values
        state = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
        energy = expectation.forward(state)
        
        # Backward pass
        circuit.backward(1.0)  # Compute gradients
        
        # Update parameters
        optimizer.step(circuit)
        
        if step % 2 == 0:
            print(f"Step {step}: Energy = {energy:.4f}")
            print(f"  Gradients: {circuit.get_gradients()}")
    
    print("Final parameters:", circuit.get_parameters())


def gradient_methods_example():
    """Compare different gradient computation methods."""
    print("\n=== Gradient Methods Example ===")
    
    # Parameter shift rule (default)
    circuit_ps = ParametricCircuit(2, gradient_method="parameter_shift")
    circuit_ps.rx(0, "theta", initial_value=np.pi/4)
    
    # Finite differences
    circuit_fd = ParametricCircuit(2, gradient_method="finite_diff")
    circuit_fd.rx(0, "theta", initial_value=np.pi/4)
    
    print("Parameter shift gradient computation...")
    circuit_ps.backward()
    ps_grad = circuit_ps.get_gradients()
    print(f"  Gradient: {ps_grad}")
    
    print("Finite difference gradient computation...")
    circuit_fd.backward()
    fd_grad = circuit_fd.get_gradients()
    print(f"  Gradient: {fd_grad}")


def layered_circuit_example():
    """Example of building layered ansatz circuits."""
    print("\n=== Layered Circuit Example ===")
    
    n_qubits = 4
    n_layers = 3
    
    circuit = ParametricCircuit(n_qubits)
    
    # Build hardware-efficient ansatz
    for layer in range(n_layers):
        # Single-qubit rotation layer
        for q in range(n_qubits):
            circuit.ry(q, f"ry_{layer}_{q}", initial_value=np.random.rand())
            circuit.rz(q, f"rz_{layer}_{q}", initial_value=np.random.rand())
        
        # Entangling layer (would add CNOT gates in full implementation)
        # For now, just add more rotations
        for q in range(0, n_qubits-1, 2):
            circuit.rx(q, f"rx_ent_{layer}_{q}", initial_value=0.1)
    
    params = circuit.get_parameters()
    print(f"Created {n_layers}-layer circuit with {len(params)} parameters")
    print(f"Parameter names: {list(params.keys())[:5]}...")


def u3_gate_example():
    """Example using general U3 gates."""
    print("\n=== U3 Gate Example ===")
    
    circuit = ParametricCircuit(1)
    
    # Add a general U3 gate with three parameters
    circuit.u3(0, "theta", "phi", "lambda", 
               theta_init=np.pi/4,
               phi_init=np.pi/6,
               lambda_init=np.pi/8)
    
    print("U3 gate parameters:", circuit.get_parameters())
    
    # Compute gradients
    circuit.backward()
    print("U3 gate gradients:", circuit.get_gradients())


def optimization_example():
    """Example of parameter optimization with momentum."""
    print("\n=== Optimization Example ===")
    
    circuit = ParametricCircuit(2)
    circuit.ry(0, "theta1", initial_value=0.0)
    circuit.ry(1, "theta2", initial_value=0.0)
    
    # Create optimizer with momentum
    optimizer = CircuitOptimizer(learning_rate=0.1, momentum=0.9)
    
    # Simulated optimization loop
    for step in range(5):
        # Simulate gradients (in reality, these come from quantum measurements)
        circuit.zero_grad()
        # Manually set some gradients for demonstration
        params = circuit.circuit.parameters.lock().unwrap()
        if "theta1" in params:
            params["theta1"].gradient = -0.5
        if "theta2" in params:
            params["theta2"].gradient = 0.3
        
        # Update parameters
        optimizer.step(circuit)
        
        print(f"Step {step}: Parameters = {circuit.get_parameters()}")
    
    # Update learning rate
    optimizer.set_learning_rate(0.05)
    print("Updated learning rate to 0.05")


if __name__ == "__main__":
    vqe_example()
    gradient_methods_example()
    layered_circuit_example()
    u3_gate_example()
    optimization_example()
    
    print("\n=== Parametric Circuit Demo Complete ===")