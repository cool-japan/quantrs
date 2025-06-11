#!/usr/bin/env python3
"""
Comprehensive demonstration of QuantRS2 Quantum Code Analysis Tools.

This script showcases all the capabilities of the quantum code analysis system including:
- Static analysis of quantum code with quantum-specific patterns
- Code quality metrics and complexity analysis
- Optimization suggestions and performance recommendations
- Circuit pattern detection and anti-pattern identification
- Integration with development workflows
- Analysis reporting in multiple formats
- Historical analysis tracking and statistics
"""

import os
import sys
import time
import tempfile
import logging
from pathlib import Path

# Add the parent directory to the path so we can import quantrs2
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from quantrs2.quantum_code_analysis import (
        QuantumCodeAnalysisManager, QuantumCodeAnalyzer, CodeQualityReporter,
        AnalysisLevel, AnalysisType, IssueSeverity, PatternType, MetricType,
        CodeLocation, AnalysisIssue, CodeMetric, QuantumPattern, OptimizationSuggestion, AnalysisReport,
        get_quantum_code_analysis_manager, analyze_quantum_code, analyze_quantum_project,
        HAS_CLICK, HAS_FLASK, HAS_YAML
    )
    HAS_QUANTUM_CODE_ANALYSIS = True
except ImportError as e:
    print(f"❌ Failed to import quantum code analysis module: {e}")
    HAS_QUANTUM_CODE_ANALYSIS = False


def create_sample_quantum_files(base_dir: Path):
    """Create sample quantum code files for analysis."""
    # Create sample quantum circuit file
    quantum_circuit_code = '''
import numpy as np
import quantrs2

def create_bell_state():
    """Create a Bell state using quantum circuit."""
    circuit = quantrs2.Circuit(2)
    
    # Create Bell state
    circuit.h(0)
    circuit.cx(0, 1)
    
    # Measure qubits
    circuit.measure_all()
    
    return circuit

def inefficient_quantum_algorithm():
    """Example of inefficient quantum algorithm with issues."""
    circuit = quantrs2.Circuit(4)
    
    # Inefficient gate sequence - multiple H gates
    circuit.h(0)
    circuit.h(0)  # Redundant
    
    # No measurement - missing important operation
    
    return circuit

class QuantumVQE:
    """Variational Quantum Eigensolver implementation."""
    
    def __init__(self, num_qubits, ansatz_depth=3):
        self.num_qubits = num_qubits
        self.ansatz_depth = ansatz_depth
        self.parameters = np.random.random(num_qubits * ansatz_depth)
    
    def create_ansatz(self, parameters):
        """Create variational ansatz circuit."""
        circuit = quantrs2.Circuit(self.num_qubits)
        
        param_idx = 0
        for layer in range(self.ansatz_depth):
            # Rotation layer
            for qubit in range(self.num_qubits):
                circuit.rx(qubit, parameters[param_idx])
                param_idx += 1
            
            # Entangling layer
            for qubit in range(self.num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
        
        return circuit
    
    def optimize(self, hamiltonian):
        """Optimize VQE parameters."""
        # Placeholder optimization logic
        best_energy = float('inf')
        best_params = self.parameters.copy()
        
        for iteration in range(100):
            # Create circuit with current parameters
            circuit = self.create_ansatz(self.parameters)
            
            # Simulate and compute energy
            result = circuit.run()
            energy = self.compute_energy(result, hamiltonian)
            
            if energy < best_energy:
                best_energy = energy
                best_params = self.parameters.copy()
            
            # Update parameters (simple gradient descent placeholder)
            self.parameters += np.random.normal(0, 0.01, len(self.parameters))
        
        return best_energy, best_params
    
    def compute_energy(self, result, hamiltonian):
        """Compute energy expectation value."""
        # Placeholder energy computation
        return np.random.random()
'''
    
    (base_dir / "quantum_circuit.py").write_text(quantum_circuit_code)
    
    # Create sample QAOA implementation
    qaoa_code = '''
import numpy as np
import quantrs2

class QAOA:
    """Quantum Approximate Optimization Algorithm implementation."""
    
    def __init__(self, graph, num_layers=3):
        self.graph = graph  # Problem graph
        self.num_layers = num_layers
        self.num_qubits = len(graph.nodes()) if hasattr(graph, 'nodes') else 4
        
        # Initialize parameters
        self.beta = np.random.random(num_layers)  # Mixer parameters
        self.gamma = np.random.random(num_layers)  # Cost parameters
    
    def create_mixer_circuit(self, circuit, beta):
        """Create mixer circuit (X rotations)."""
        for qubit in range(self.num_qubits):
            circuit.rx(qubit, 2 * beta)
    
    def create_cost_circuit(self, circuit, gamma):
        """Create cost circuit based on problem structure."""
        # Example for MaxCut problem
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Sample edges
        
        for edge in edges:
            qubit1, qubit2 = edge
            circuit.cx(qubit1, qubit2)
            circuit.rz(qubit2, 2 * gamma)
            circuit.cx(qubit1, qubit2)
    
    def create_qaoa_circuit(self):
        """Create complete QAOA circuit."""
        circuit = quantrs2.Circuit(self.num_qubits)
        
        # Initial superposition
        for qubit in range(self.num_qubits):
            circuit.h(qubit)
        
        # QAOA layers
        for layer in range(self.num_layers):
            self.create_cost_circuit(circuit, self.gamma[layer])
            self.create_mixer_circuit(circuit, self.beta[layer])
        
        # Measurement
        circuit.measure_all()
        
        return circuit
    
    def optimize_parameters(self):
        """Optimize QAOA parameters using classical optimizer."""
        best_cost = float('inf')
        best_params = (self.beta.copy(), self.gamma.copy())
        
        for iteration in range(50):
            circuit = self.create_qaoa_circuit()
            result = circuit.run()
            
            cost = self.compute_cost(result)
            
            if cost < best_cost:
                best_cost = cost
                best_params = (self.beta.copy(), self.gamma.copy())
            
            # Update parameters
            self.beta += np.random.normal(0, 0.1, len(self.beta))
            self.gamma += np.random.normal(0, 0.1, len(self.gamma))
        
        return best_cost, best_params
    
    def compute_cost(self, result):
        """Compute cost function value."""
        # Placeholder cost computation
        return np.random.random()

# Example usage with potential issues
def run_qaoa_example():
    """Run QAOA example with some code quality issues."""
    
    # Create graph (simplified)
    graph = None  # This could cause issues
    
    qaoa = QAOA(graph, num_layers=5)  # High number of layers might be inefficient
    
    # Run optimization
    best_cost, best_params = qaoa.optimize_parameters()
    
    print(f"Best cost: {best_cost}")
    print(f"Best parameters: {best_params}")
    
    # Missing error handling
    # No documentation for parameters
    # Hardcoded values that should be configurable

if __name__ == "__main__":
    run_qaoa_example()
'''
    
    (base_dir / "qaoa_algorithm.py").write_text(qaoa_code)
    
    # Create sample quantum teleportation implementation
    teleportation_code = '''
import quantrs2
import numpy as np

def quantum_teleportation(state_to_teleport):
    """
    Quantum teleportation protocol implementation.
    
    This function demonstrates quantum teleportation but has some
    potential optimization opportunities and style issues.
    """
    
    # Create 3-qubit circuit (Alice has 2 qubits, Bob has 1)
    circuit = quantrs2.Circuit(3)
    
    # Prepare the state to teleport on qubit 0
    if state_to_teleport == "|0>":
        pass  # Already in |0>
    elif state_to_teleport == "|1>":
        circuit.x(0)
    elif state_to_teleport == "|+>":
        circuit.h(0)
    elif state_to_teleport == "|->" :
        circuit.h(0)
        circuit.z(0)
    
    # Create Bell pair between qubits 1 and 2 (Alice's ancilla and Bob's qubit)
    circuit.h(1)
    circuit.cx(1, 2)
    
    # Alice's operations (Bell measurement)
    circuit.cx(0, 1)  # CNOT between state qubit and ancilla
    circuit.h(0)      # Hadamard on state qubit
    
    # Measure Alice's qubits
    circuit.measure(0)  # This could be optimized
    circuit.measure(1)  # Separate measurements instead of batch
    
    # Bob's conditional operations (normally based on classical bits)
    # For simulation, we'll apply all possible corrections
    # This is not optimal - should be conditional
    circuit.cx(1, 2)  # Conditional X based on qubit 1 measurement
    circuit.cz(0, 2)  # Conditional Z based on qubit 0 measurement
    
    # Measure Bob's qubit
    circuit.measure(2)
    
    return circuit

def demonstrate_teleportation():
    """Demonstrate quantum teleportation for different states."""
    
    states = ["|0>", "|1>", "|+>", "|->"]
    
    for state in states:
        print(f"Teleporting state: {state}")
        
        # Create teleportation circuit
        circuit = quantum_teleportation(state)
        
        # Run simulation
        result = circuit.run()  # Missing error handling
        
        # Analyze results (simplified)
        probabilities = result.state_probabilities()
        print(f"Measurement probabilities: {probabilities}")
        
        print("-" * 40)

# Quantum superdense coding implementation
def superdense_coding(message_bits):
    """
    Quantum superdense coding protocol.
    
    Sends 2 classical bits using 1 qubit.
    Has some inefficiencies and missing documentation.
    """
    
    circuit = quantrs2.Circuit(2)
    
    # Create Bell pair
    circuit.h(0)
    circuit.cx(0, 1)
    
    # Alice encodes message in her qubit (qubit 0)
    if message_bits == "00":
        pass  # Identity operation
    elif message_bits == "01":
        circuit.z(0)  # Z gate
    elif message_bits == "10":
        circuit.x(0)  # X gate  
    elif message_bits == "11":
        circuit.x(0)  # X then Z
        circuit.z(0)
    else:
        raise ValueError("Invalid message bits")  # Good error handling
    
    # Bob's decoding operations
    circuit.cx(0, 1)
    circuit.h(0)
    
    # Measure both qubits
    circuit.measure_all()
    
    return circuit

# Performance testing function with issues
def performance_test():
    """Test performance of quantum protocols."""
    
    import time
    
    # Inefficient loop that could be vectorized
    results = []
    for i in range(1000):  # Large loop without optimization
        state = ["|0>", "|1>", "|+>", "|->"][i % 4]
        
        start_time = time.time()
        circuit = quantum_teleportation(state)
        result = circuit.run()
        end_time = time.time()
        
        execution_time = end_time - start_time
        results.append(execution_time)
        
        # Inefficient string concatenation in loop
        output = ""
        for j, prob in enumerate(result.state_probabilities().values()):
            output += f"Prob_{j}: {prob}, "
    
    # Calculate statistics
    avg_time = sum(results) / len(results)
    max_time = max(results)
    min_time = min(results)
    
    print(f"Performance Results:")
    print(f"Average time: {avg_time:.6f}s")
    print(f"Max time: {max_time:.6f}s") 
    print(f"Min time: {min_time:.6f}s")

if __name__ == "__main__":
    demonstrate_teleportation()
    performance_test()
'''
    
    (base_dir / "quantum_teleportation.py").write_text(teleportation_code)
    
    # Create a file with style and complexity issues
    problematic_code = '''
# Poor quantum code example with multiple issues

import quantrs2
import numpy as np

# Global variables (bad practice)
QUBITS = 8
DEPTH = 20

def bad_function(q,c,d,x,y,z):  # Poor parameter names
    """Function with multiple issues."""
    
    # Very long, complex function
    circuit = quantrs2.Circuit(q)
    
    # Deeply nested loops
    for i in range(d):
        for j in range(q):
            for k in range(3):
                if i % 2 == 0:
                    if j % 2 == 0:
                        if k == 0:
                            circuit.h(j)
                        elif k == 1:
                            circuit.x(j)
                        else:
                            circuit.y(j)
                    else:
                        if k == 0:
                            circuit.z(j)
                        elif k == 1:
                            circuit.s(j)
                        else:
                            circuit.t(j)
                else:
                    if j < q-1:
                        circuit.cx(j, j+1)
    
    # Redundant operations
    circuit.h(0)
    circuit.h(0)
    circuit.x(1)
    circuit.x(1)
    
    # Memory inefficient
    large_array = np.zeros((2**q, 2**q))  # Exponential memory usage
    
    # No error handling
    result = circuit.run()
    
    # Inefficient probability calculation
    probs = {}
    for i in range(2**q):
        basis_state = format(i, f'0{q}b')
        probs[basis_state] = abs(result.amplitudes[i])**2
    
    return probs

class PoorlyDesignedQuantumClass:
    """Class with design issues."""
    
    def __init__(self):
        self.data = []
        self.qubits = 16  # Hardcoded
        self.circuits = {}
        
    def method1(self):
        # Method that does too much
        circuit = quantrs2.Circuit(self.qubits)
        
        # Duplicate code
        for i in range(self.qubits):
            circuit.h(i)
        
        for i in range(self.qubits):
            circuit.x(i)
            
        for i in range(self.qubits):
            circuit.y(i)
            
        for i in range(self.qubits):
            circuit.z(i)
        
        # Should be a separate method
        result = circuit.run()
        probs = result.state_probabilities()
        
        # Should be a separate method  
        max_prob = max(probs.values())
        max_state = [k for k, v in probs.items() if v == max_prob][0]
        
        # Return multiple unrelated things
        return circuit, result, probs, max_prob, max_state

def function_with_security_issues():
    """Function that might have security concerns."""
    
    # Dynamic code execution (security risk)
    gate_name = "h"  # This could come from user input
    qubit = 0
    
    circuit = quantrs2.Circuit(4)
    
    # Eval usage (dangerous)
    # eval(f"circuit.{gate_name}({qubit})")  # Commented out but still an issue
    
    # File operations without validation
    import os
    filename = "temp_circuit.py"  # Could be manipulated
    with open(filename, 'w') as f:
        f.write("circuit.h(0)")
    
    # Command execution
    # os.system(f"python {filename}")  # Commented but concerning
    
    return circuit

# Missing main guard
demonstrate_bad_practices()  # This will cause error if run
'''
    
    (base_dir / "problematic_quantum_code.py").write_text(problematic_code)


def demonstrate_code_analysis_basics():
    """Demonstrate basic quantum code analysis features."""
    print("\n" + "="*60)
    print("🔍 QUANTUM CODE ANALYSIS BASICS")
    print("="*60)
    
    # Create temporary directory with sample files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        sample_dir = temp_path / "quantum_project"
        sample_dir.mkdir()
        
        # Create sample quantum files
        create_sample_quantum_files(sample_dir)
        
        print(f"📁 Created sample quantum project at: {sample_dir}")
        print(f"📄 Files created: {[f.name for f in sample_dir.glob('*.py')]}")
        
        # Initialize analyzer
        analyzer = QuantumCodeAnalyzer()
        
        # Analyze each file
        for file_path in sample_dir.glob("*.py"):
            print(f"\n🔬 Analyzing file: {file_path.name}")
            print("-" * 40)
            
            try:
                report = analyzer.analyze_file(str(file_path), AnalysisLevel.STANDARD)
                
                print(f"✅ Analysis completed in {report.execution_time:.3f}s")
                print(f"📊 Quality Score: {report.summary.get('quality_score', 0):.2f}")
                print(f"⚠️  Total Issues: {len(report.issues)}")
                print(f"📈 Metrics: {len(report.metrics)}")
                print(f"🔍 Patterns: {len(report.patterns)}")
                print(f"🎯 Optimizations: {len(report.optimizations)}")
                
                # Show top issues
                if report.issues:
                    print("\n🚨 Top Issues:")
                    for issue in report.issues[:3]:
                        severity_icon = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}.get(issue.severity.value, "⚪")
                        print(f"  {severity_icon} {issue.title} (Line {issue.location.line_number})")
                        print(f"     {issue.description}")
                
                # Show metrics
                if report.metrics:
                    print("\n📊 Key Metrics:")
                    for metric in report.metrics[:3]:
                        print(f"  • {metric.name}: {metric.value} {metric.unit}")
                
                # Show patterns
                if report.patterns:
                    print("\n🔍 Detected Patterns:")
                    for pattern in report.patterns:
                        print(f"  • {pattern.name}: {pattern.description}")
                
            except Exception as e:
                print(f"❌ Analysis failed: {e}")


def demonstrate_analysis_levels():
    """Demonstrate different analysis depth levels."""
    print("\n" + "="*60)
    print("📊 ANALYSIS LEVELS COMPARISON")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        sample_file = temp_path / "test_circuit.py"
        
        # Create a sample file for analysis
        sample_code = '''
import quantrs2
import numpy as np

def create_vqe_circuit(parameters):
    """Create VQE ansatz circuit with potential optimization opportunities."""
    
    circuit = quantrs2.Circuit(4)
    
    # Initial state preparation
    for qubit in range(4):
        circuit.h(qubit)
    
    # Ansatz layers with redundancies
    for layer in range(3):
        # Rotation layer
        for i, qubit in enumerate(range(4)):
            circuit.rx(qubit, parameters[layer * 4 + i])
            circuit.ry(qubit, parameters[layer * 4 + i] * 0.5)  # Potential optimization
        
        # Entangling layer
        for qubit in range(3):
            circuit.cx(qubit, qubit + 1)
    
    # Redundant operations
    circuit.h(0)
    circuit.h(0)  # Cancels out
    
    return circuit

def inefficient_simulation():
    """Function with performance issues."""
    
    results = []
    for i in range(100):  # Could be vectorized
        params = np.random.random(12)
        circuit = create_vqe_circuit(params)
        result = circuit.run()
        
        # Inefficient probability calculation
        probs = {}
        for j in range(16):
            basis = format(j, '04b')
            probs[basis] = abs(result.amplitudes[j])**2
        
        results.append(probs)
    
    return results
'''
        
        sample_file.write_text(sample_code)
        
        analyzer = QuantumCodeAnalyzer()
        
        # Test different analysis levels
        levels = [AnalysisLevel.BASIC, AnalysisLevel.STANDARD, AnalysisLevel.COMPREHENSIVE, AnalysisLevel.DEEP]
        
        print(f"📄 Analyzing file: {sample_file.name}")
        print()
        
        for level in levels:
            print(f"🔍 {level.value.upper()} Analysis:")
            print("-" * 30)
            
            start_time = time.time()
            report = analyzer.analyze_file(str(sample_file), level)
            analysis_time = time.time() - start_time
            
            print(f"⏱️  Execution Time: {analysis_time:.3f}s")
            print(f"📊 Quality Score: {report.summary.get('quality_score', 0):.2f}")
            print(f"🔢 Issues Found: {len(report.issues)}")
            print(f"📈 Metrics: {len(report.metrics)}")
            print(f"🔍 Patterns: {len(report.patterns)}")
            print(f"🎯 Optimizations: {len(report.optimizations)}")
            
            # Show analysis types
            analysis_types = set(issue.analysis_type.value for issue in report.issues)
            if analysis_types:
                print(f"🏷️  Analysis Types: {', '.join(analysis_types)}")
            
            print()


def demonstrate_project_analysis():
    """Demonstrate full project analysis with reporting."""
    print("\n" + "="*60)
    print("📁 PROJECT-LEVEL ANALYSIS")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_dir = temp_path / "quantum_project"
        project_dir.mkdir()
        
        # Create sample project structure
        create_sample_quantum_files(project_dir)
        
        # Create subdirectories with more files
        algorithms_dir = project_dir / "algorithms"
        algorithms_dir.mkdir()
        
        utils_dir = project_dir / "utils"
        utils_dir.mkdir()
        
        # Create additional files
        (algorithms_dir / "grover.py").write_text('''
import quantrs2

def grovers_algorithm(oracle_function, num_qubits):
    """Grover's algorithm implementation."""
    
    circuit = quantrs2.Circuit(num_qubits)
    
    # Initialize superposition
    for qubit in range(num_qubits):
        circuit.h(qubit)
    
    # Grover iterations
    num_iterations = int(np.pi * np.sqrt(2**num_qubits) / 4)
    
    for iteration in range(num_iterations):
        # Apply oracle
        oracle_function(circuit)
        
        # Diffusion operator
        for qubit in range(num_qubits):
            circuit.h(qubit)
            circuit.x(qubit)
        
        # Multi-controlled Z
        circuit.h(num_qubits - 1)
        for i in range(num_qubits - 1):
            circuit.cx(i, num_qubits - 1)
        circuit.h(num_qubits - 1)
        
        for qubit in range(num_qubits):
            circuit.x(qubit)
            circuit.h(qubit)
    
    circuit.measure_all()
    return circuit
''')
        
        (utils_dir / "quantum_utils.py").write_text('''
import quantrs2
import numpy as np

def create_random_state(num_qubits):
    """Create random quantum state."""
    circuit = quantrs2.Circuit(num_qubits)
    
    for qubit in range(num_qubits):
        # Random rotation
        theta = np.random.random() * 2 * np.pi
        phi = np.random.random() * 2 * np.pi
        
        circuit.rx(qubit, theta)
        circuit.rz(qubit, phi)
    
    return circuit

def measure_fidelity(circuit1, circuit2):
    """Measure fidelity between two quantum states."""
    # Simplified fidelity calculation
    result1 = circuit1.run()
    result2 = circuit2.run()
    
    # Calculate overlap
    fidelity = 0.0
    for i, (amp1, amp2) in enumerate(zip(result1.amplitudes, result2.amplitudes)):
        fidelity += (amp1.conjugate() * amp2).real
    
    return abs(fidelity)**2
''')
        
        print(f"📁 Created quantum project with structure:")
        for file_path in project_dir.rglob("*.py"):
            rel_path = file_path.relative_to(project_dir)
            print(f"  📄 {rel_path}")
        
        # Initialize analysis manager
        manager = QuantumCodeAnalysisManager()
        
        print(f"\n🔬 Analyzing entire project...")
        
        try:
            # Analyze project
            start_time = time.time()
            report_json = manager.analyze_project(
                str(project_dir), 
                AnalysisLevel.COMPREHENSIVE,
                output_format="json"
            )
            analysis_time = time.time() - start_time
            
            print(f"✅ Project analysis completed in {analysis_time:.2f}s")
            
            # Parse JSON report for summary
            import json
            report_data = json.loads(report_json)
            
            print(f"\n📊 PROJECT SUMMARY:")
            print(f"📄 Total files analyzed: {report_data['total_files']}")
            print(f"📈 Average quality score: {report_data['summary']['avg_quality_score']:.2f}")
            print(f"⚠️  Total issues found: {report_data['summary']['total_issues']}")
            print(f"🔴 Critical issues: {report_data['summary']['critical_issues']}")
            print(f"🟠 Error issues: {report_data['summary']['error_issues']}")
            print(f"🟡 Warning issues: {report_data['summary']['warning_issues']}")
            print(f"🔵 Info issues: {report_data['summary']['info_issues']}")
            
            # Generate different report formats
            print(f"\n📝 Generating reports in different formats...")
            
            # HTML report
            html_report = manager.reporter.generate_report(
                {path: manager.analyzer.analyze_file(path, AnalysisLevel.STANDARD) 
                 for path in project_dir.rglob("*.py")}, 
                format="html"
            )
            
            # Text report
            text_report = manager.reporter.generate_report(
                {path: manager.analyzer.analyze_file(path, AnalysisLevel.STANDARD) 
                 for path in project_dir.rglob("*.py")}, 
                format="text"
            )
            
            print(f"✅ Generated HTML report ({len(html_report)} characters)")
            print(f"✅ Generated text report ({len(text_report)} characters)")
            
            # Show text report preview
            print(f"\n📋 TEXT REPORT PREVIEW:")
            print("-" * 50)
            print(text_report[:500] + "..." if len(text_report) > 500 else text_report)
            
        except Exception as e:
            print(f"❌ Project analysis failed: {e}")


def demonstrate_pattern_detection():
    """Demonstrate quantum algorithm pattern detection."""
    print("\n" + "="*60)
    print("🔍 QUANTUM PATTERN DETECTION")
    print("="*60)
    
    # Create files with different quantum algorithm patterns
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # VQE pattern
        vqe_file = temp_path / "vqe_example.py"
        vqe_code = '''
import quantrs2
import numpy as np
from scipy.optimize import minimize

class VariationalQuantumEigensolver:
    """VQE implementation for finding ground state energy."""
    
    def __init__(self, num_qubits, ansatz_depth=3):
        self.num_qubits = num_qubits
        self.ansatz_depth = ansatz_depth
        self.optimizer = minimize
    
    def create_ansatz(self, parameters):
        """Create variational ansatz circuit."""
        circuit = quantrs2.Circuit(self.num_qubits)
        
        param_idx = 0
        for layer in range(self.ansatz_depth):
            # Parametric gates
            for qubit in range(self.num_qubits):
                circuit.ry(qubit, parameters[param_idx])
                param_idx += 1
            
            # Entangling gates
            for qubit in range(self.num_qubits - 1):
                circuit.cx(qubit, qubit + 1)
        
        return circuit
    
    def cost_function(self, parameters, hamiltonian):
        """Cost function for optimization."""
        circuit = self.create_ansatz(parameters)
        result = circuit.run()
        return self.expectation_value(result, hamiltonian)
    
    def expectation_value(self, result, hamiltonian):
        """Calculate expectation value."""
        # Simplified expectation value calculation
        return np.random.random()
'''
        vqe_file.write_text(vqe_code)
        
        # QAOA pattern  
        qaoa_file = temp_path / "qaoa_example.py"
        qaoa_code = '''
import quantrs2
import numpy as np

class QuantumApproximateOptimization:
    """QAOA for combinatorial optimization problems."""
    
    def __init__(self, problem_graph, num_layers=3):
        self.graph = problem_graph
        self.num_layers = num_layers
        self.beta = np.random.random(num_layers)  # Mixer parameters
        self.gamma = np.random.random(num_layers)  # Cost parameters
    
    def mixer_layer(self, circuit, beta):
        """Apply mixer Hamiltonian (X rotations)."""
        for qubit in range(len(self.graph.nodes())):
            circuit.rx(qubit, 2 * beta)
    
    def cost_layer(self, circuit, gamma):
        """Apply cost Hamiltonian."""
        for edge in self.graph.edges():
            qubit1, qubit2 = edge
            circuit.cx(qubit1, qubit2)
            circuit.rz(qubit2, 2 * gamma)
            circuit.cx(qubit1, qubit2)
    
    def create_qaoa_circuit(self):
        """Create QAOA circuit with alternating layers."""
        num_qubits = len(self.graph.nodes())
        circuit = quantrs2.Circuit(num_qubits)
        
        # Initialize superposition
        for qubit in range(num_qubits):
            circuit.h(qubit)
        
        # Alternating cost and mixer layers
        for layer in range(self.num_layers):
            self.cost_layer(circuit, self.gamma[layer])
            self.mixer_layer(circuit, self.beta[layer])
        
        return circuit
'''
        qaoa_file.write_text(qaoa_code)
        
        # Teleportation pattern
        teleport_file = temp_path / "teleportation_example.py"
        teleport_code = '''
import quantrs2

def quantum_teleportation_protocol():
    """Quantum teleportation with Bell state and measurements."""
    
    # 3-qubit system: state to teleport, Alice's ancilla, Bob's qubit
    circuit = quantrs2.Circuit(3)
    
    # Prepare state to teleport (example: |+> state)
    circuit.h(0)
    
    # Create Bell pair between Alice's ancilla (1) and Bob's qubit (2)
    circuit.h(1)
    circuit.cx(1, 2)
    
    # Bell measurement on Alice's qubits
    circuit.cx(0, 1)  # Entangle state with ancilla
    circuit.h(0)      # Hadamard on state qubit
    
    # Measurements (in real implementation, these would be classical bits)
    circuit.measure(0)
    circuit.measure(1)
    
    # Bob's conditional operations (based on measurement results)
    circuit.cx(1, 2)  # Conditional X
    circuit.cz(0, 2)  # Conditional Z
    
    return circuit

def prepare_bell_state(circuit, qubit1, qubit2):
    """Helper function to create Bell state."""
    circuit.h(qubit1)
    circuit.cx(qubit1, qubit2)
'''
        teleport_file.write_text(teleport_code)
        
        # Analyze pattern detection
        analyzer = QuantumCodeAnalyzer()
        
        files_and_expected_patterns = [
            (vqe_file, ["VQE", "variational", "ansatz"]),
            (qaoa_file, ["QAOA", "mixer", "cost_function"]),
            (teleport_file, ["teleportation", "bell_state"])
        ]
        
        for file_path, expected_keywords in files_and_expected_patterns:
            print(f"\n🔬 Analyzing: {file_path.name}")
            print("-" * 40)
            
            try:
                report = analyzer.analyze_file(str(file_path), AnalysisLevel.COMPREHENSIVE)
                
                print(f"📊 Quality Score: {report.summary.get('quality_score', 0):.2f}")
                print(f"🔍 Patterns Detected: {len(report.patterns)}")
                
                if report.patterns:
                    for pattern in report.patterns:
                        confidence_icon = "🟢" if pattern.confidence > 0.7 else "🟡" if pattern.confidence > 0.5 else "🔴"
                        print(f"  {confidence_icon} {pattern.name}: {pattern.description}")
                        print(f"     Confidence: {pattern.confidence:.2f}")
                        if pattern.recommendation:
                            print(f"     💡 {pattern.recommendation}")
                
                # Check if expected patterns were detected
                detected_names = [p.name.lower() for p in report.patterns]
                for keyword in expected_keywords:
                    if any(keyword.lower() in name for name in detected_names):
                        print(f"  ✅ Expected pattern '{keyword}' detected")
                    else:
                        print(f"  ⚠️ Expected pattern '{keyword}' not detected")
                
            except Exception as e:
                print(f"❌ Analysis failed: {e}")


def demonstrate_optimization_suggestions():
    """Demonstrate optimization suggestion capabilities."""
    print("\n" + "="*60)
    print("🎯 OPTIMIZATION SUGGESTIONS")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create file with optimization opportunities
        optimization_file = temp_path / "optimization_example.py"
        optimization_code = '''
import quantrs2
import numpy as np

def inefficient_quantum_circuit():
    """Circuit with multiple optimization opportunities."""
    
    circuit = quantrs2.Circuit(4)
    
    # Redundant gate sequences that could be optimized
    circuit.h(0)
    circuit.h(0)  # H·H = I (identity)
    
    circuit.x(1)
    circuit.x(1)  # X·X = I
    
    # Adjacent rotations that could be combined
    circuit.rx(2, 0.5)
    circuit.ry(2, 0.3)
    circuit.rz(2, 0.7)  # Could be combined into single rotation
    
    # Inefficient CNOT chain
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.cx(3, 0)  # Could be optimized
    
    # Unnecessary deep circuit
    for i in range(20):  # Very deep circuit
        circuit.rz(0, 0.1)
        circuit.rz(1, 0.1)
        circuit.cx(0, 1)
    
    return circuit

def memory_inefficient_simulation():
    """Function with memory optimization opportunities."""
    
    num_qubits = 8
    
    # Inefficient memory usage
    all_states = []
    for i in range(2**num_qubits):  # Storing all possible states
        state_vector = np.zeros(2**num_qubits, dtype=complex)
        state_vector[i] = 1.0
        all_states.append(state_vector)
    
    # Inefficient probability calculation
    circuit = inefficient_quantum_circuit()
    result = circuit.run()
    
    probabilities = {}
    for i in range(2**num_qubits):
        basis_state = format(i, f'0{num_qubits}b')
        probabilities[basis_state] = abs(result.amplitudes[i])**2
    
    return probabilities

def performance_bottleneck():
    """Function with performance optimization opportunities."""
    
    # Inefficient loop that could be vectorized
    results = []
    for angle in np.linspace(0, 2*np.pi, 1000):
        circuit = quantrs2.Circuit(2)
        circuit.rx(0, angle)
        circuit.ry(1, angle/2)
        circuit.cx(0, 1)
        
        result = circuit.run()
        
        # Inefficient string operations in loop
        output_string = ""
        for state, prob in result.state_probabilities().items():
            output_string += f"{state}: {prob:.6f}, "
        
        results.append(output_string)
    
    return results

class QuantumClassWithIssues:
    """Class that could benefit from optimization."""
    
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.circuits = {}
        
        # Inefficient initialization
        for i in range(100):  # Creating many similar circuits
            circuit = quantrs2.Circuit(num_qubits)
            for qubit in range(num_qubits):
                circuit.h(qubit)
            self.circuits[f"circuit_{i}"] = circuit
    
    def run_all_circuits(self):
        """Method that could be parallelized."""
        results = {}
        
        # Sequential execution that could be parallel
        for name, circuit in self.circuits.items():
            result = circuit.run()
            results[name] = result.state_probabilities()
        
        return results
'''
        
        optimization_file.write_text(optimization_code)
        
        print(f"📄 Analyzing file: {optimization_file.name}")
        print("🔍 Looking for optimization opportunities...")
        
        analyzer = QuantumCodeAnalyzer()
        
        try:
            report = analyzer.analyze_file(str(optimization_file), AnalysisLevel.COMPREHENSIVE)
            
            print(f"\n📊 Analysis Results:")
            print(f"  Quality Score: {report.summary.get('quality_score', 0):.2f}")
            print(f"  Total Issues: {len(report.issues)}")
            print(f"  Optimization Suggestions: {len(report.optimizations)}")
            
            # Show optimization suggestions
            if report.optimizations:
                print(f"\n🎯 OPTIMIZATION SUGGESTIONS:")
                for i, opt in enumerate(report.optimizations, 1):
                    effort_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(opt.effort, "⚪")
                    print(f"\n{i}. {opt.title} {effort_icon}")
                    print(f"   📍 Location: Line {opt.location.line_number}")
                    print(f"   📝 Description: {opt.description}")
                    print(f"   💪 Effort: {opt.effort}")
                    print(f"   📈 Impact: {opt.impact}")
                    print(f"   🎁 Potential Improvement: {opt.potential_improvement}")
                    
                    if opt.implementation_hints:
                        print(f"   💡 Implementation Hints:")
                        for hint in opt.implementation_hints:
                            print(f"      • {hint}")
                    
                    if opt.before_code and opt.after_code:
                        print(f"   📋 Code Example:")
                        print(f"      Before: {opt.before_code}")
                        print(f"      After:  {opt.after_code}")
            else:
                print("No optimization suggestions found (this might indicate the analysis needs enhancement)")
            
            # Show performance-related issues
            performance_issues = [issue for issue in report.issues if issue.analysis_type == AnalysisType.PERFORMANCE]
            if performance_issues:
                print(f"\n⚡ PERFORMANCE ISSUES:")
                for issue in performance_issues:
                    severity_icon = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}.get(issue.severity.value, "⚪")
                    print(f"  {severity_icon} {issue.title} (Line {issue.location.line_number})")
                    print(f"     {issue.description}")
                    if issue.suggestion:
                        print(f"     💡 Suggestion: {issue.suggestion}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")


def demonstrate_analysis_manager():
    """Demonstrate the full analysis manager capabilities."""
    print("\n" + "="*60)
    print("⚙️ ANALYSIS MANAGER FEATURES")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize analysis manager with custom workspace
        workspace_dir = temp_path / "analysis_workspace"
        manager = QuantumCodeAnalysisManager(str(workspace_dir))
        
        print(f"📁 Initialized analysis manager")
        print(f"   Workspace: {manager.workspace_dir}")
        print(f"   Database: {manager.db_path}")
        print(f"   Cache: {manager.cache_dir}")
        
        # Create sample project
        project_dir = temp_path / "sample_project"
        project_dir.mkdir()
        create_sample_quantum_files(project_dir)
        
        print(f"\n📊 Running analysis history demonstration...")
        
        # Run multiple analyses to build history
        for i in range(3):
            print(f"   Analysis run {i+1}/3...")
            
            try:
                report = manager.analyze_project(
                    str(project_dir),
                    AnalysisLevel.STANDARD,
                    output_format="json"
                )
                
                # Add small delay to distinguish timestamps
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   ❌ Analysis {i+1} failed: {e}")
        
        # Show analysis history
        print(f"\n📚 Analysis History:")
        history = manager.get_analysis_history(limit=5)
        
        for entry in history:
            timestamp = time.strftime('%H:%M:%S', time.localtime(entry['analysis_timestamp']))
            print(f"  🕒 {timestamp}: {entry['total_files']} files, {entry['total_issues']} issues")
            print(f"     Quality: {entry['avg_quality_score']:.2f}, Time: {entry['execution_time']:.2f}s")
        
        # Show statistics
        print(f"\n📈 Analysis Statistics:")
        stats = manager.get_analysis_statistics()
        
        print(f"  📊 Total analyses performed: {stats['total_analyses']}")
        print(f"  🎯 Average quality score: {stats['average_quality_score']:.2f}")
        
        if stats['top_analyzed_projects']:
            print(f"  🏆 Most analyzed projects:")
            for project in stats['top_analyzed_projects']:
                print(f"     • {project['project']}: {project['count']} times")
        
        print(f"\n✅ Analysis manager demonstration completed!")


def demonstrate_integration_features():
    """Demonstrate integration with development tools."""
    print("\n" + "="*60)
    print("🔗 INTEGRATION FEATURES")
    print("="*60)
    
    # Check available integrations
    print("🔍 Checking available integrations:")
    print(f"  📋 YAML support: {'✅' if HAS_YAML else '❌'}")
    print(f"  🖱️  Click CLI: {'✅' if HAS_CLICK else '❌'}")
    print(f"  🌐 Flask web interface: {'✅' if HAS_FLASK else '❌'}")
    
    # Demonstrate CLI integration (if available)
    if HAS_CLICK:
        print(f"\n💻 CLI Integration Available:")
        print(f"   Commands: analyze, analyze-file, history, stats, web")
        print(f"   Example: python -m quantrs2.quantum_code_analysis analyze ./project")
        print(f"   Example: python -m quantrs2.quantum_code_analysis analyze-file ./circuit.py")
    
    # Demonstrate web interface (if available)
    if HAS_FLASK:
        print(f"\n🌐 Web Interface Available:")
        print(f"   Start server: python -m quantrs2.quantum_code_analysis web")
        print(f"   Endpoints: /, /analyze, /history")
        print(f"   Features: Interactive dashboard, REST API")
    
    # Demonstrate configuration
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        workspace_dir = temp_path / "integration_workspace"
        
        manager = QuantumCodeAnalysisManager(str(workspace_dir))
        
        print(f"\n⚙️ Configuration:")
        for key, value in manager.config.items():
            print(f"   {key}: {value}")
        
        # Demonstrate convenience functions
        print(f"\n🎛️ Convenience Functions:")
        
        # Create sample file
        sample_file = temp_path / "sample.py"
        sample_file.write_text('''
import quantrs2

def simple_circuit():
    circuit = quantrs2.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit
''')
        
        # Use convenience function
        try:
            report = analyze_quantum_code(str(sample_file), AnalysisLevel.BASIC)
            print(f"  ✅ analyze_quantum_code(): {len(report.issues)} issues found")
        except Exception as e:
            print(f"  ❌ analyze_quantum_code() failed: {e}")
        
        # Create mini project
        mini_project = temp_path / "mini_project" 
        mini_project.mkdir()
        (mini_project / "main.py").write_text(sample_file.read_text())
        
        try:
            report_json = analyze_quantum_project(str(mini_project), AnalysisLevel.BASIC)
            print(f"  ✅ analyze_quantum_project(): {len(report_json)} chars generated")
        except Exception as e:
            print(f"  ❌ analyze_quantum_project() failed: {e}")


def run_all_demonstrations():
    """Run all demonstration functions."""
    
    if not HAS_QUANTUM_CODE_ANALYSIS:
        print("❌ Quantum Code Analysis module not available!")
        return
    
    print("🚀 QuantRS2 Quantum Code Analysis Tools - Comprehensive Demo")
    print("=" * 80)
    
    try:
        demonstrate_code_analysis_basics()
        demonstrate_analysis_levels() 
        demonstrate_project_analysis()
        demonstrate_pattern_detection()
        demonstrate_optimization_suggestions()
        demonstrate_analysis_manager()
        demonstrate_integration_features()
        
        print("\n" + "="*80)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n🎉 The QuantRS2 Quantum Code Analysis Tools provide:")
        print("   • Comprehensive static analysis for quantum code")
        print("   • Quantum-specific pattern detection and metrics")
        print("   • Performance optimization suggestions")
        print("   • Multiple analysis depth levels")
        print("   • Project-wide analysis with detailed reporting")
        print("   • Integration with development workflows")
        print("   • Historical analysis tracking")
        print("   • CLI and web interfaces")
        
        print("\n💡 Next steps:")
        print("   • Integrate with your quantum development workflow")
        print("   • Use CLI tools for automated code quality checks")
        print("   • Set up web interface for team collaboration")
        print("   • Configure analysis rules for your specific needs")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    run_all_demonstrations()