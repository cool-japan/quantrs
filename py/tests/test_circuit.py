#!/usr/bin/env python3
import quantrs2 as qr
import sys

def main():
    """Test the quantrs2 Python bindings."""
    print(f"Python version: {sys.version}")
    print(f"Package location: {qr.__file__}")
    print(f"Module contents: {dir(qr)}")
    
    # Create a circuit
    circuit = qr.PyCircuit(2)
    print(f"Circuit: {circuit}")
    
    # Add gates
    circuit.h(0)
    circuit.cnot(0, 1)
    
    # Run the circuit with debug info
    print("Running circuit...")
    try:
        result = circuit.run()
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Check if PySimulationResult class is accessible
        print(f"PySimulationResult class: {qr.PySimulationResult}")
        
        # Try to create a PySimulationResult instance directly
        # (This is just for testing, not a normal usage pattern)
        try:
            test_result = qr.PySimulationResult()
            print(f"Created test result: {test_result}")
        except Exception as e:
            print(f"Could not create PySimulationResult directly: {e}")
        
        # Try to access result properties and methods
        if result is not None:
            print(f"Result dir: {dir(result)}")
            try:
                n_qubits = result.n_qubits
                print(f"n_qubits: {n_qubits}")
            except Exception as e:
                print(f"Error accessing n_qubits: {e}")
                
            try:
                probs = result.state_probabilities()
                print(f"Probabilities: {probs}")
                for state, prob in probs.items():
                    print(f"|{state}⟩: {prob:.6f}")
            except Exception as e:
                print(f"Error accessing state_probabilities: {e}")
        else:
            print("Result is None")
    except Exception as e:
        print(f"Error running circuit: {e}")

if __name__ == "__main__":
    main()