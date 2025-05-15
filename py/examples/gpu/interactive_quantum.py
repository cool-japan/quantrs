#!/usr/bin/env python3
"""
QuantRS2 Interactive Quantum Circuit Designer

This script provides an interactive way to create and simulate quantum circuits
using both CPU and GPU backends. It demonstrates the GPU code path even with
our current stub implementation.
"""

import sys
import time
import numpy as np

class QuantumCircuitDemo:
    def __init__(self):
        try:
            import _quantrs2 as qr
            self.qr = qr
            print("✓ Successfully imported QuantRS2")
        except ImportError:
            print("❌ Could not import _quantrs2 module")
            print("Make sure you've built with: ./build_with_gpu_stub.sh")
            print("And activated the virtual environment: source .venv/bin/activate")
            sys.exit(1)
        
        self.circuit = None
        self.n_qubits = 0
        self.gpu_available = self._check_gpu_available()
    
    def _check_gpu_available(self):
        """Check if the GPU code path is available."""
        # Create a minimal circuit to test
        test_circuit = self.qr.PyCircuit(2)
        test_circuit.h(0)
        
        try:
            # Try to run with GPU
            test_circuit.run(use_gpu=True)
            return True
        except Exception as e:
            if "GPU acceleration requested but not compiled in" in str(e):
                return False
            # Some other error occurred, but the GPU feature is compiled in
            return True
    
    def create_circuit(self, n_qubits):
        """Create a new quantum circuit."""
        if n_qubits not in [1, 2, 3, 4, 5, 8, 10, 16]:
            print(f"Error: {n_qubits} qubits not supported.")
            print("Supported qubit counts: 1, 2, 3, 4, 5, 8, 10, 16")
            return False
        
        self.circuit = self.qr.PyCircuit(n_qubits)
        self.n_qubits = n_qubits
        print(f"Created a new {n_qubits}-qubit circuit.")
        return True
    
    def add_gate(self, gate_type, *args):
        """Add a gate to the circuit."""
        if self.circuit is None:
            print("Error: Create a circuit first with 'new <n_qubits>'")
            return
        
        try:
            if gate_type == "h":
                if len(args) != 1:
                    print("Usage: h <qubit>")
                    return
                qubit = int(args[0])
                self.circuit.h(qubit)
                print(f"Added Hadamard gate to qubit {qubit}")
            
            elif gate_type == "x":
                if len(args) != 1:
                    print("Usage: x <qubit>")
                    return
                qubit = int(args[0])
                self.circuit.x(qubit)
                print(f"Added Pauli-X gate to qubit {qubit}")
            
            elif gate_type == "y":
                if len(args) != 1:
                    print("Usage: y <qubit>")
                    return
                qubit = int(args[0])
                self.circuit.y(qubit)
                print(f"Added Pauli-Y gate to qubit {qubit}")
            
            elif gate_type == "z":
                if len(args) != 1:
                    print("Usage: z <qubit>")
                    return
                qubit = int(args[0])
                self.circuit.z(qubit)
                print(f"Added Pauli-Z gate to qubit {qubit}")
            
            elif gate_type == "cnot":
                if len(args) != 2:
                    print("Usage: cnot <control> <target>")
                    return
                control = int(args[0])
                target = int(args[1])
                self.circuit.cnot(control, target)
                print(f"Added CNOT gate with control={control}, target={target}")
            
            elif gate_type == "rx":
                if len(args) != 2:
                    print("Usage: rx <qubit> <angle>")
                    return
                qubit = int(args[0])
                angle = float(args[1])
                self.circuit.rx(qubit, angle)
                print(f"Added Rx({angle}) gate to qubit {qubit}")
            
            elif gate_type == "ry":
                if len(args) != 2:
                    print("Usage: ry <qubit> <angle>")
                    return
                qubit = int(args[0])
                angle = float(args[1])
                self.circuit.ry(qubit, angle)
                print(f"Added Ry({angle}) gate to qubit {qubit}")
            
            elif gate_type == "rz":
                if len(args) != 2:
                    print("Usage: rz <qubit> <angle>")
                    return
                qubit = int(args[0])
                angle = float(args[1])
                self.circuit.rz(qubit, angle)
                print(f"Added Rz({angle}) gate to qubit {qubit}")
            
            elif gate_type == "swap":
                if len(args) != 2:
                    print("Usage: swap <qubit1> <qubit2>")
                    return
                qubit1 = int(args[0])
                qubit2 = int(args[1])
                self.circuit.swap(qubit1, qubit2)
                print(f"Added SWAP gate between qubits {qubit1} and {qubit2}")
            
            else:
                print(f"Unknown gate: {gate_type}")
                self.print_help()
        
        except Exception as e:
            print(f"Error adding gate: {e}")
    
    def run(self, use_gpu=False):
        """Run the quantum circuit and return the result."""
        if self.circuit is None:
            print("Error: Create a circuit first with 'new <n_qubits>'")
            return None
        
        if use_gpu and not self.gpu_available:
            print("GPU code path not available, falling back to CPU.")
            use_gpu = False
        
        try:
            if use_gpu:
                print(f"Running {self.n_qubits}-qubit circuit on GPU...")
            else:
                print(f"Running {self.n_qubits}-qubit circuit on CPU...")
            
            start_time = time.time()
            result = self.circuit.run(use_gpu=use_gpu)
            elapsed = time.time() - start_time
            
            if use_gpu:
                print(f"GPU simulation completed in {elapsed:.4f} seconds")
            else:
                print(f"CPU simulation completed in {elapsed:.4f} seconds")
            
            # Print state probabilities
            probs = result.state_probabilities()
            print("\nState probabilities:")
            if len(probs) > 20:  # Only show top states for large circuits
                print("(Showing top 10 states)")
                top_states = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]
                for state, prob in top_states:
                    binary = '|' + state + '⟩'
                    print(f"{binary:<{self.n_qubits+4}} {prob:.8f}")
            else:
                # Sort by lexical order for smaller circuits
                for state in sorted(probs.keys()):
                    binary = '|' + state + '⟩'
                    print(f"{binary:<{self.n_qubits+4}} {probs[state]:.8f}")
            
            return result
        
        except Exception as e:
            print(f"Error running circuit: {e}")
            return None
    
    def reset(self):
        """Reset the circuit to empty."""
        if self.circuit is not None:
            self.circuit = self.qr.PyCircuit(self.n_qubits)
            print(f"Reset {self.n_qubits}-qubit circuit to empty.")
        else:
            print("No circuit exists yet.")
    
    def print_help(self):
        """Print help information."""
        print("\nQuantRS2 Interactive Quantum Circuit Designer")
        print("=" * 50)
        print("Commands:")
        print("  new <n_qubits>       - Create a new circuit with n qubits (1, 2, 3, 4, 5, 8, 10, 16)")
        print("  h <qubit>            - Add Hadamard gate")
        print("  x <qubit>            - Add Pauli-X gate")
        print("  y <qubit>            - Add Pauli-Y gate")
        print("  z <qubit>            - Add Pauli-Z gate")
        print("  rx <qubit> <angle>   - Add rotation-X gate")
        print("  ry <qubit> <angle>   - Add rotation-Y gate")
        print("  rz <qubit> <angle>   - Add rotation-Z gate")
        print("  cnot <ctrl> <target> - Add CNOT gate")
        print("  swap <q1> <q2>       - Add SWAP gate")
        print("  run                  - Run circuit on CPU")
        print("  run-gpu              - Run circuit on GPU")
        print("  reset                - Reset circuit to empty")
        print("  bell                 - Create Bell state (⟨00⟩ + ⟨11⟩)/√2")
        print("  ghz <n>              - Create GHZ state (⟨00...0⟩ + ⟨11...1⟩)/√2 with n qubits")
        print("  qft <n>              - Create Quantum Fourier Transform circuit with n qubits")
        print("  help                 - Show this help")
        print("  exit, quit           - Exit the program")
        print("\nExample: new 2; h 0; cnot 0 1; run")
    
    def create_bell_state(self):
        """Create a Bell state circuit."""
        self.create_circuit(2)
        self.circuit.h(0)
        self.circuit.cnot(0, 1)
        print("Created Bell state circuit (⟨00⟩ + ⟨11⟩)/√2")
    
    def create_ghz_state(self, n_qubits):
        """Create a GHZ state circuit."""
        if not self.create_circuit(n_qubits):
            return
        
        self.circuit.h(0)
        for i in range(n_qubits - 1):
            self.circuit.cnot(i, i+1)
        
        print(f"Created {n_qubits}-qubit GHZ state circuit (⟨{'0'*n_qubits}⟩ + ⟨{'1'*n_qubits}⟩)/√2")
    
    def create_qft_circuit(self, n_qubits):
        """Create a Quantum Fourier Transform circuit."""
        if not self.create_circuit(n_qubits):
            return
        
        # Apply Hadamard to all qubits
        for i in range(n_qubits):
            self.circuit.h(i)
            
            # Apply controlled phase rotations
            for j in range(i+1, n_qubits):
                # Phase rotation by pi/2^(j-i)
                angle = np.pi / (2 ** (j-i))
                self.circuit.rz(j, angle)
        
        print(f"Created {n_qubits}-qubit Quantum Fourier Transform circuit")
    
    def run_interactive(self):
        """Run the interactive shell."""
        print("\nWelcome to the QuantRS2 Interactive Quantum Circuit Designer")
        print("=" * 60)
        if self.gpu_available:
            print("✓ GPU code path is available (though simulation may use CPU fallback)")
        else:
            print("⚠️  GPU code path is not available (compile with GPU feature flag)")
        
        print("Type 'help' for a list of commands.")
        print("Type multiple commands on one line, separated by semicolons.")
        
        while True:
            try:
                cmd_line = input("\n> ").strip()
                if not cmd_line:
                    continue
                
                # Allow multiple commands separated by semicolons
                commands = cmd_line.split(";")
                
                for cmd in commands:
                    cmd = cmd.strip()
                    if not cmd:
                        continue
                    
                    parts = cmd.split()
                    command = parts[0].lower()
                    args = parts[1:]
                    
                    if command in ["exit", "quit"]:
                        print("Goodbye!")
                        return
                    
                    elif command == "help":
                        self.print_help()
                    
                    elif command == "new":
                        if len(args) != 1:
                            print("Usage: new <n_qubits>")
                            continue
                        try:
                            n_qubits = int(args[0])
                            self.create_circuit(n_qubits)
                        except ValueError:
                            print("Error: n_qubits must be an integer")
                    
                    elif command == "reset":
                        self.reset()
                    
                    elif command == "run":
                        self.run(use_gpu=False)
                    
                    elif command == "run-gpu":
                        self.run(use_gpu=True)
                    
                    elif command == "bell":
                        self.create_bell_state()
                    
                    elif command == "ghz":
                        if len(args) != 1:
                            print("Usage: ghz <n_qubits>")
                            continue
                        try:
                            n_qubits = int(args[0])
                            self.create_ghz_state(n_qubits)
                        except ValueError:
                            print("Error: n_qubits must be an integer")
                    
                    elif command == "qft":
                        if len(args) != 1:
                            print("Usage: qft <n_qubits>")
                            continue
                        try:
                            n_qubits = int(args[0])
                            self.create_qft_circuit(n_qubits)
                        except ValueError:
                            print("Error: n_qubits must be an integer")
                    
                    elif command in ["h", "x", "y", "z", "rx", "ry", "rz", "cnot", "swap"]:
                        self.add_gate(command, *args)
                    
                    else:
                        print(f"Unknown command: {command}")
                        print("Type 'help' for a list of commands")
            
            except KeyboardInterrupt:
                print("\nType 'exit' to quit")
            except EOFError:
                print("\nGoodbye!")
                return
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    demo = QuantumCircuitDemo()
    demo.run_interactive()