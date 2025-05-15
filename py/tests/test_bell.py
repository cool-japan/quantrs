#!/usr/bin/env python3
"""
Bell state test for QuantRS2.

This test creates a Bell state and prints the results.
"""

import sys
import traceback

try:
    import quantrs2 as qr
    from math import sqrt
    
    # Debug information
    print(f"Python version: {sys.version}")
    print(f"QuantRS2 version: {qr.__version__}")
    print(f"QuantRS2 location: {qr.__file__}")
    
    # Available classes and methods in the module
    print("\nAvailable in quantrs2 module:")
    for name in dir(qr):
        if not name.startswith('_'):
            print(f"  - {name}")
    
    # Create a 2-qubit circuit
    circuit = qr.PyCircuit(2)
    print(f"\nCircuit created: {circuit}")
    print(f"Circuit type: {type(circuit)}")
    
    # Build a Bell state
    print("\nBuilding Bell state...")
    circuit.h(0)
    circuit.cnot(0, 1)
    
    # Run the simulation
    print("\nRunning simulation...")
    result = circuit.run()
    print(f"Result: {result}")
    print(f"Result type: {type(result)}")
    
    # Create a PySimulationResult directly
    try:
        print("\nTrying to create PySimulationResult directly...")
        manual_result = qr.PySimulationResult()
        print(f"Manual result: {manual_result}")
        print(f"Manual result type: {type(manual_result)}")
        
        # Add Bell state amplitudes
        print("\nAdding Bell state amplitudes to manual result...")
        # Add attributes directly to the object
        bell_list = [(1/sqrt(2), 0), (0, 0), (0, 0), (1/sqrt(2), 0)]
        manual_result.set_amplitudes(bell_list)
        manual_result.set_n_qubits(2)
        
        # Try to use state_probabilities method
        print("\nTrying state_probabilities on manual result...")
        manual_probs = manual_result.state_probabilities()
        print(f"Manual probabilities: {manual_probs}")
        for state, prob in manual_probs.items():
            print(f"|{state}⟩: {prob:.6f}")
    except Exception as e:
        print(f"Error creating manual result: {e}")
        traceback.print_exc()
    
    # Use the monkeypatched result from circuit.run()
    print("\nUsing result from circuit.run()...")
    if result is not None:
        try:
            probs = result.state_probabilities()
            print(f"Probabilities: {probs}")
            for state, prob in probs.items():
                print(f"|{state}⟩: {prob:.6f}")
        except Exception as e:
            print(f"Error getting state_probabilities: {e}")
            traceback.print_exc()
    else:
        print("Result is None, using monkeypatched result...")
        # We will use the monkeypatched version from __init__.py
        patched_result = circuit.run()
        if patched_result is not None:
            print(f"Patched result: {patched_result}")
            print(f"Patched result type: {type(patched_result)}")
            probs = patched_result.state_probabilities()
            print(f"Patched probabilities: {probs}")
            for state, prob in probs.items():
                print(f"|{state}⟩: {prob:.6f}")
        else:
            print("Patched result is still None")
            
except ImportError as e:
    print(f"Error importing quantrs2: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()