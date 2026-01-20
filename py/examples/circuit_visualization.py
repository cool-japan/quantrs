"""
Circuit Visualization Example

This example demonstrates the circuit visualization capabilities in QuantRS2.
It shows both text-based and HTML-based circuit visualization for different circuits.
"""

import quantrs2 as qrs
import numpy as np

def main():
    print("QuantRS2 Circuit Visualization Example")
    print("======================================\n")
    
    # Example 1: Simple bell state circuit
    print("Example 1: Bell State Circuit")
    print("----------------------------")
    bell_circuit = qrs.PyCircuit(2)
    bell_circuit.h(0)
    bell_circuit.cnot(0, 1)
    
    # Text-based visualization
    print("\nText-based visualization:")
    print(bell_circuit.draw())
    
    # If in a Jupyter notebook, the circuit can be displayed with HTML
    print("\nTo see the HTML visualization, run this in a Jupyter notebook and execute:")
    print("display(bell_circuit)")
    
    # Example 2: More complex circuit with different gates
    print("\n\nExample 2: Complex Circuit")
    print("-------------------------")
    complex_circuit = qrs.PyCircuit(4)
    
    # Add a variety of gates
    complex_circuit.h(0)
    complex_circuit.x(1)
    complex_circuit.y(2)
    complex_circuit.z(3)
    complex_circuit.cnot(0, 1)
    complex_circuit.cz(1, 2)
    complex_circuit.swap(2, 3)
    complex_circuit.rx(0, np.pi/4)
    complex_circuit.ry(1, np.pi/3)
    complex_circuit.rz(2, np.pi/2)
    complex_circuit.crx(0, 3, np.pi/2)
    complex_circuit.toffoli(0, 1, 2)
    
    # Text-based visualization
    print("\nText-based visualization:")
    print(complex_circuit.draw())
    
    # Example 3: Using the dedicated visualizer
    print("\n\nExample 3: Using PyCircuitVisualizer directly")
    print("--------------------------------------------")
    
    # Create a visualizer
    visualizer = qrs.PyCircuitVisualizer(3)
    
    # Add gates manually
    visualizer.add_gate("H", [0], None)
    visualizer.add_gate("CNOT", [0, 1], None)
    visualizer.add_gate("SWAP", [1, 2], None)
    visualizer.add_gate("RZ", [0], "π/2")
    visualizer.add_gate("Toffoli", [0, 1, 2], None)
    
    # Display the circuit
    print("\nText-based visualization from PyCircuitVisualizer:")
    print(visualizer.to_text())
    
    print("\nTo display HTML visualization in a Jupyter notebook:")
    print("display(visualizer)")
    
    # Example 4: GHZ state preparation
    print("\n\nExample 4: GHZ State Preparation")
    print("------------------------------")
    ghz_circuit = qrs.PyCircuit(5)
    
    # Prepare GHZ state: |00000⟩ + |11111⟩
    ghz_circuit.h(0)
    ghz_circuit.cnot(0, 1)
    ghz_circuit.cnot(1, 2)
    ghz_circuit.cnot(2, 3)
    ghz_circuit.cnot(3, 4)
    
    # Text-based visualization
    print("\nText-based visualization:")
    print(ghz_circuit.draw())
    
if __name__ == "__main__":
    main()