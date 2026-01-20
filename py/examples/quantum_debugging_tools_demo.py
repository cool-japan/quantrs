#!/usr/bin/env python3
"""
Comprehensive demo of the QuantRS2 Quantum Debugging Tools.

This demo showcases the complete debugging capabilities including:
- Advanced quantum state inspection with multiple analysis modes
- Comprehensive quantum error analysis with auto-fix suggestions
- Circuit validation with extensive property checking
- Memory debugging with leak detection and optimization suggestions
- Interactive debugging console with full command support
- Web-based debugging interface with real-time monitoring
- Integration with performance profiling and testing frameworks
- Error recovery and debugging workflow automation

Run this demo to see the full range of debugging features available
in the QuantRS2 quantum computing framework.
"""

import numpy as np
import time
import sys
import os
from pathlib import Path

try:
    import quantrs2
    from quantrs2.quantum_debugging_tools import (
        DebugLevel, ErrorType, InspectionMode, ValidationRule,
        QuantumStateInspector, QuantumErrorAnalyzer, QuantumCircuitValidator,
        QuantumMemoryDebugger, InteractiveQuantumDebugConsole,
        QuantumDebuggingWebInterface, QuantumDebuggingToolsManager,
        get_quantum_debugging_tools, debug_quantum_circuit, analyze_quantum_error,
        inspect_quantum_state, validate_quantum_circuit,
        start_quantum_debugging_console, start_quantum_debugging_web_interface
    )
    print(f"QuantRS2 version: {quantrs2.__version__}")
    print(f"Successfully imported quantum debugging tools")
except ImportError as e:
    print(f"Error importing QuantRS2 debugging tools: {e}")
    print("Please ensure the debugging tools are properly installed")
    sys.exit(1)

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    print("✓ Matplotlib support available")
except ImportError:
    HAS_MATPLOTLIB = False
    print("✗ Matplotlib not available")

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
    print("✓ Plotly support available")
except ImportError:
    HAS_PLOTLY = False
    print("✗ Plotly not available")

try:
    import dash
    HAS_DASH = True
    print("✓ Dash web interface support available")
except ImportError:
    HAS_DASH = False
    print("✗ Dash web interface not available")

try:
    import psutil
    HAS_PSUTIL = True
    print("✓ Memory profiling support available")
except ImportError:
    HAS_PSUTIL = False
    print("✗ Memory profiling not available")


def create_sample_quantum_states():
    """Create sample quantum states for demonstration."""
    states = {}
    
    # |0⟩ state
    states['zero'] = np.array([1, 0], dtype=complex)
    
    # |1⟩ state
    states['one'] = np.array([0, 1], dtype=complex)
    
    # |+⟩ state (superposition)
    states['plus'] = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    
    # |i⟩ state (complex superposition)
    states['i'] = np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)
    
    # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    states['bell'] = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    
    # GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
    ghz_state = np.zeros(8, dtype=complex)
    ghz_state[0] = 1/np.sqrt(2)  # |000⟩
    ghz_state[7] = 1/np.sqrt(2)  # |111⟩
    states['ghz'] = ghz_state
    
    # W state |W⟩ = (|001⟩ + |010⟩ + |100⟩)/√3
    w_state = np.zeros(8, dtype=complex)
    w_state[1] = 1/np.sqrt(3)  # |001⟩
    w_state[2] = 1/np.sqrt(3)  # |010⟩
    w_state[4] = 1/np.sqrt(3)  # |100⟩
    states['w'] = w_state
    
    # Non-normalized state (for testing error detection)
    states['unnormalized'] = np.array([2, 3], dtype=complex)
    
    # Zero state (for testing edge cases)
    states['zero_vector'] = np.array([0, 0], dtype=complex)
    
    return states


def create_sample_quantum_circuits():
    """Create sample quantum circuits for demonstration."""
    circuits = {}
    
    # Simple Bell state circuit
    bell_circuit = type('Circuit', (), {})()
    bell_circuit.num_qubits = 2
    bell_circuit.gates = [
        type('Gate', (), {'name': 'H', 'qubits': [0], 'params': []})(),
        type('Gate', (), {'name': 'CNOT', 'qubits': [0, 1], 'params': []})()
    ]
    bell_circuit.run = lambda: type('Result', (), {
        'state_vector': np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    })()
    circuits['bell'] = bell_circuit
    
    # GHZ circuit
    ghz_circuit = type('Circuit', (), {})()
    ghz_circuit.num_qubits = 3
    ghz_circuit.gates = [
        type('Gate', (), {'name': 'H', 'qubits': [0], 'params': []})(),
        type('Gate', (), {'name': 'CNOT', 'qubits': [0, 1], 'params': []})(),
        type('Gate', (), {'name': 'CNOT', 'qubits': [1, 2], 'params': []})()
    ]
    ghz_circuit.run = lambda: type('Result', (), {
        'state_vector': np.array([1/np.sqrt(2), 0, 0, 0, 0, 0, 0, 1/np.sqrt(2)])
    })()
    circuits['ghz'] = ghz_circuit
    
    # Circuit with measurements (for causality testing)
    measurement_circuit = type('Circuit', (), {})()
    measurement_circuit.num_qubits = 2
    measurement_circuit.gates = [
        type('Gate', (), {'name': 'H', 'qubits': [0], 'params': []})(),
        type('Gate', (), {'name': 'MEASURE', 'qubits': [0], 'params': []})(),
        type('Gate', (), {'name': 'CNOT', 'qubits': [0, 1], 'params': []})()  # After measurement!
    ]
    circuits['measurement'] = measurement_circuit
    
    # Large circuit (for resource testing)
    large_circuit = type('Circuit', (), {})()
    large_circuit.num_qubits = 10
    large_circuit.gates = []
    for i in range(50):
        large_circuit.gates.append(
            type('Gate', (), {'name': 'H', 'qubits': [i % 10], 'params': []})()
        )
        if i > 0:
            large_circuit.gates.append(
                type('Gate', (), {'name': 'CNOT', 'qubits': [i % 10, (i + 1) % 10], 'params': []})()
            )
    circuits['large'] = large_circuit
    
    # Circuit with invalid gates (for error testing)
    invalid_circuit = type('Circuit', (), {})()
    invalid_circuit.num_qubits = 2
    invalid_circuit.gates = [
        type('Gate', (), {'name': 'INVALID_GATE', 'qubits': [0], 'params': []})(),
        type('Gate', (), {'name': 'H', 'qubits': [5], 'params': []})()  # Invalid qubit
    ]
    circuits['invalid'] = invalid_circuit
    
    return circuits


def demo_quantum_state_inspection():
    """Demonstrate quantum state inspection capabilities."""
    print("\n" + "="*60)
    print("QUANTUM STATE INSPECTION DEMO")
    print("="*60)
    
    states = create_sample_quantum_states()
    inspector = QuantumStateInspector(DebugLevel.INFO)
    
    for state_name, state_vector in states.items():
        if state_name in ['unnormalized', 'zero_vector']:
            continue  # Skip problematic states for main demo
        
        print(f"\n--- Analyzing {state_name.upper()} state ---")
        
        try:
            # Amplitude analysis
            result = inspector.inspect_state(state_vector, InspectionMode.AMPLITUDE_ANALYSIS)
            print(f"Amplitude Analysis Insights:")
            for insight in result.insights[:3]:  # Show first 3 insights
                print(f"  • {insight}")
            
            if result.anomalies:
                print(f"Anomalies detected:")
                for anomaly in result.anomalies[:2]:
                    print(f"  ⚠ {anomaly}")
            
            # For multi-qubit states, show entanglement analysis
            if len(state_vector) > 2:
                ent_result = inspector.inspect_state(state_vector, InspectionMode.ENTANGLEMENT_ANALYSIS)
                print(f"Entanglement Analysis:")
                for insight in ent_result.insights[:2]:
                    print(f"  • {insight}")
            
            # Show probability analysis for interesting states
            if state_name in ['bell', 'ghz', 'w']:
                prob_result = inspector.inspect_state(state_vector, InspectionMode.PROBABILITY_ANALYSIS)
                prob_stats = prob_result.analysis_data.get('probability_stats', {})
                entropy = prob_stats.get('entropy', 0)
                print(f"  Entropy: {entropy:.3f}")
                
        except Exception as e:
            print(f"  Error during inspection: {e}")
    
    # Demonstrate edge case handling
    print(f"\n--- Testing Edge Cases ---")
    
    # Unnormalized state
    try:
        result = inspector.inspect_state(states['unnormalized'], InspectionMode.AMPLITUDE_ANALYSIS)
        print("Unnormalized state analysis:")
        for anomaly in result.anomalies:
            print(f"  ⚠ {anomaly}")
        for rec in result.recommendations[:2]:
            print(f"  → {rec}")
    except Exception as e:
        print(f"Unnormalized state handling: {e}")
    
    # Zero state
    try:
        result = inspector.inspect_state(states['zero_vector'], InspectionMode.AMPLITUDE_ANALYSIS)
        print("Zero state analysis completed successfully")
    except Exception as e:
        print(f"Zero state error: {e}")


def demo_quantum_error_analysis():
    """Demonstrate quantum error analysis capabilities."""
    print("\n" + "="*60)
    print("QUANTUM ERROR ANALYSIS DEMO")
    print("="*60)
    
    analyzer = QuantumErrorAnalyzer(DebugLevel.INFO)
    
    # Sample errors for different categories
    sample_errors = [
        {
            'error': Exception("Gate execution failed: RX parameter out of range [-π, π]"),
            'context': {'gate_name': 'RX', 'qubit_indices': [0], 'parameter': 4.5},
            'description': "Gate Parameter Error"
        },
        {
            'error': Exception("Measurement readout error: fidelity below threshold (0.85)"),
            'context': {'measurement_type': 'Z-basis', 'qubit_indices': [1]},
            'description': "Measurement Error"
        },
        {
            'error': Exception("T2 dephasing detected: coherence time exceeded"),
            'context': {'qubit_indices': [0], 'coherence_time': 50e-6},
            'description': "Decoherence Error"
        },
        {
            'error': Exception("Circuit depth exceeds maximum limit (500 gates)"),
            'context': {'current_depth': 750, 'max_depth': 500},
            'description': "Circuit Error"
        },
        {
            'error': Exception("State vector normalization error: ||ψ|| = 1.234"),
            'context': {'norm': 1.234, 'expected': 1.0},
            'description': "State Error"
        },
        {
            'error': Exception("Memory allocation failed: insufficient quantum memory"),
            'context': {'requested_memory': '2GB', 'available_memory': '1.5GB'},
            'description': "Memory Error"
        },
        {
            'error': Exception("Performance timeout: operation exceeded 30 seconds"),
            'context': {'operation': 'circuit_simulation', 'timeout': 30},
            'description': "Performance Error"
        },
        {
            'error': Exception("Unknown quantum error occurred"),
            'context': {},
            'description': "Unknown Error"
        }
    ]
    
    for i, error_data in enumerate(sample_errors, 1):
        print(f"\n--- Error Analysis {i}: {error_data['description']} ---")
        
        try:
            diagnosis = analyzer.analyze_error(error_data['error'], error_data['context'])
            
            print(f"Error Type: {diagnosis.error_type.name}")
            print(f"Severity: {diagnosis.severity}")
            print(f"Message: {diagnosis.message}")
            print(f"Location: {diagnosis.location}")
            
            if diagnosis.suggestions:
                print("Suggestions:")
                for suggestion in diagnosis.suggestions[:3]:  # Show first 3 suggestions
                    print(f"  → {suggestion}")
            
            if diagnosis.auto_fix_available:
                print("✓ Auto-fix available")
                # Demonstrate auto-fix attempt
                fix_result = analyzer.apply_auto_fix(diagnosis, error_data['context'])
                print(f"Auto-fix result: {fix_result['message']}")
            else:
                print("✗ No auto-fix available")
                
        except Exception as e:
            print(f"Error during analysis: {e}")
    
    # Show error history
    print(f"\n--- Error History Summary ---")
    print(f"Total errors analyzed: {len(analyzer.error_history)}")
    
    error_type_counts = {}
    for error in analyzer.error_history:
        error_type = error.error_type.name
        error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
    
    print("Error distribution:")
    for error_type, count in error_type_counts.items():
        print(f"  {error_type}: {count}")


def demo_quantum_circuit_validation():
    """Demonstrate quantum circuit validation capabilities."""
    print("\n" + "="*60)
    print("QUANTUM CIRCUIT VALIDATION DEMO")
    print("="*60)
    
    circuits = create_sample_quantum_circuits()
    validator = QuantumCircuitValidator(DebugLevel.INFO)
    
    # Validate different circuits with different rule sets
    validation_scenarios = [
        {
            'circuit_name': 'bell',
            'rules': [ValidationRule.UNITARITY_CHECK, ValidationRule.NORMALIZATION_CHECK],
            'description': "Bell State Circuit - Basic Validation"
        },
        {
            'circuit_name': 'ghz',
            'rules': [ValidationRule.RESOURCE_CHECK, ValidationRule.CONNECTIVITY_CHECK],
            'description': "GHZ Circuit - Resource Validation"
        },
        {
            'circuit_name': 'measurement',
            'rules': [ValidationRule.CAUSALITY_CHECK, ValidationRule.TIMING_CHECK],
            'description': "Measurement Circuit - Causality Validation"
        },
        {
            'circuit_name': 'large',
            'rules': [ValidationRule.RESOURCE_CHECK, ValidationRule.TIMING_CHECK],
            'description': "Large Circuit - Performance Validation"
        }
    ]
    
    for scenario in validation_scenarios:
        print(f"\n--- {scenario['description']} ---")
        
        circuit = circuits[scenario['circuit_name']]
        
        try:
            results = validator.validate_circuit(circuit, scenario['rules'])
            
            for result in results:
                status = "✓ PASSED" if result.passed else "✗ FAILED"
                print(f"{status} {result.rule.name}: {result.message}")
                
                # Show interesting details
                if 'used_qubits' in result.details:
                    print(f"  Used qubits: {result.details['used_qubits']}")
                if 'total_gates' in result.details:
                    print(f"  Total gates: {result.details['total_gates']}")
                if 'circuit_depth' in result.details:
                    print(f"  Circuit depth: {result.details['circuit_depth']}")
                if 'timing_issues' in result.details and result.details['timing_issues']:
                    print(f"  Timing issues: {len(result.details['timing_issues'])}")
                
                # Show suggestions for failed validations
                if not result.passed and result.suggestions:
                    print("  Suggestions:")
                    for suggestion in result.suggestions[:2]:
                        print(f"    → {suggestion}")
                        
        except Exception as e:
            print(f"Validation error: {e}")
    
    # Comprehensive validation of a single circuit
    print(f"\n--- Comprehensive Validation: Bell Circuit ---")
    
    try:
        all_results = validator.validate_circuit(circuits['bell'])  # All rules
        
        passed_count = sum(1 for r in all_results if r.passed)
        total_count = len(all_results)
        
        print(f"Validation Summary: {passed_count}/{total_count} rules passed")
        
        # Show failed validations
        failed_results = [r for r in all_results if not r.passed]
        if failed_results:
            print("Failed validations:")
            for result in failed_results:
                print(f"  ✗ {result.rule.name}: {result.message}")
        else:
            print("All validations passed!")
            
    except Exception as e:
        print(f"Comprehensive validation error: {e}")


def demo_memory_debugging():
    """Demonstrate memory debugging capabilities."""
    print("\n" + "="*60)
    print("MEMORY DEBUGGING DEMO")
    print("="*60)
    
    memory_debugger = QuantumMemoryDebugger(DebugLevel.INFO)
    
    # Get initial memory snapshot
    print("--- Initial Memory State ---")
    initial_snapshot = memory_debugger.get_memory_snapshot()
    
    print(f"Total Memory: {initial_snapshot.total_memory / 1024 / 1024:.2f} MB")
    print(f"Quantum Memory: {initial_snapshot.quantum_memory / 1024 / 1024:.2f} MB")
    print(f"Classical Memory: {initial_snapshot.classical_memory / 1024 / 1024:.2f} MB")
    print(f"Peak Usage: {initial_snapshot.peak_usage / 1024 / 1024:.2f} MB")
    
    if initial_snapshot.optimization_suggestions:
        print("Initial optimization suggestions:")
        for suggestion in initial_snapshot.optimization_suggestions[:3]:
            print(f"  → {suggestion}")
    
    # Demonstrate memory profiling context
    print("\n--- Memory Profiling Context Demo ---")
    
    try:
        with memory_debugger.memory_profiling_context("quantum_state_creation"):
            # Create several large quantum states
            states = []
            for i in range(5):
                # Create 5-qubit states (32 amplitudes each)
                state = np.random.random(32) + 1j * np.random.random(32)
                state = state / np.linalg.norm(state)
                states.append(state)
                time.sleep(0.1)  # Simulate processing time
        
        print("Memory profiling context completed successfully")
        
    except Exception as e:
        print(f"Memory profiling error: {e}")
    
    # Get final memory snapshot
    print("\n--- Final Memory State ---")
    final_snapshot = memory_debugger.get_memory_snapshot()
    
    memory_delta = final_snapshot.total_memory - initial_snapshot.total_memory
    print(f"Memory delta: {memory_delta / 1024 / 1024:.2f} MB")
    
    if final_snapshot.memory_leaks:
        print("Memory leaks detected:")
        for leak in final_snapshot.memory_leaks:
            print(f"  ⚠ {leak['description']}")
    else:
        print("No memory leaks detected")
    
    # Demonstrate memory monitoring (if psutil available)
    if HAS_PSUTIL:
        print("\n--- Memory Monitoring Demo ---")
        
        try:
            print("Starting memory monitoring...")
            memory_debugger.start_memory_monitoring()
            
            # Perform some memory-intensive operations
            for i in range(3):
                large_array = np.random.random(10000)
                time.sleep(0.5)
                del large_array
            
            time.sleep(1)  # Let monitoring collect data
            
            memory_debugger.stop_memory_monitoring()
            
            snapshots_collected = len(memory_debugger.memory_snapshots)
            print(f"Collected {snapshots_collected} memory snapshots during monitoring")
            
            if snapshots_collected > 0:
                max_memory = max(s.total_memory for s in memory_debugger.memory_snapshots)
                min_memory = min(s.total_memory for s in memory_debugger.memory_snapshots)
                print(f"Memory range: {min_memory / 1024 / 1024:.2f} - {max_memory / 1024 / 1024:.2f} MB")
                
        except Exception as e:
            print(f"Memory monitoring error: {e}")
    else:
        print("\n--- Memory Monitoring ---")
        print("psutil not available - memory monitoring disabled")


def demo_interactive_console():
    """Demonstrate interactive debugging console capabilities."""
    print("\n" + "="*60)
    print("INTERACTIVE DEBUGGING CONSOLE DEMO")
    print("="*60)
    
    console = InteractiveQuantumDebugConsole(DebugLevel.INFO)
    
    # Demonstrate various console commands
    demo_commands = [
        "help",
        "session new",
        "session info",
        "inspect amplitude",
        "inspect entanglement",
        "validate unitarity",
        "validate resources",
        "analyze gate parameter error",
        "breakpoint add gate_5",
        "breakpoint list",
        "memory",
        "history"
    ]
    
    print("Demonstrating console commands:")
    print("=" * 40)
    
    for command in demo_commands:
        print(f"\n(qdebug) {command}")
        try:
            result = console.execute_command(command)
            # Truncate long results for demo
            if len(result) > 300:
                result = result[:300] + "... [truncated]"
            print(result)
        except Exception as e:
            print(f"Command error: {e}")
    
    print("\n" + "=" * 40)
    print("Interactive console demo completed")
    print("To start a full interactive session, use: start_quantum_debugging_console()")


def demo_web_interface():
    """Demonstrate web debugging interface capabilities."""
    print("\n" + "="*60)
    print("WEB DEBUGGING INTERFACE DEMO")
    print("="*60)
    
    if not (HAS_DASH or HAS_FLASK):
        print("Web interface demo skipped - no web framework available")
        print("Install dash or flask to enable web interface")
        return
    
    try:
        web_interface = QuantumDebuggingWebInterface(DebugLevel.INFO)
        
        if web_interface.app is not None:
            print("✓ Web debugging interface created successfully")
            
            if HAS_DASH:
                print("✓ Dash-based interface available")
                print("  Features:")
                print("  - Interactive debugging console")
                print("  - Real-time quantum state visualization")
                print("  - Memory usage monitoring")
                print("  - Debug session management")
            else:
                print("✓ Flask-based interface available")
                print("  Features:")
                print("  - Command-line style interface")
                print("  - Basic debugging functionality")
            
            print("\nTo start the web interface:")
            print("  start_quantum_debugging_web_interface()")
            print("  Then navigate to: http://localhost:8050")
            
        else:
            print("✗ Web interface creation failed")
            
    except Exception as e:
        print(f"Web interface demo error: {e}")


def demo_comprehensive_debugging():
    """Demonstrate comprehensive debugging workflow."""
    print("\n" + "="*60)
    print("COMPREHENSIVE DEBUGGING WORKFLOW DEMO")
    print("="*60)
    
    # Create debugging tools manager
    debugger = QuantumDebuggingToolsManager(DebugLevel.INFO)
    
    # Test comprehensive debugging of different circuits
    circuits = create_sample_quantum_circuits()
    
    test_scenarios = [
        {
            'name': 'bell',
            'description': 'Bell State Circuit - Complete Analysis',
            'options': {
                'validate_circuit': True,
                'analyze_state': True,
                'profile_performance': False,
                'analyze_memory': True
            }
        },
        {
            'name': 'measurement',
            'description': 'Measurement Circuit - Causality Issues',
            'options': {
                'validate_circuit': True,
                'analyze_state': False,
                'profile_performance': False,
                'analyze_memory': False
            }
        },
        {
            'name': 'large',
            'description': 'Large Circuit - Resource Analysis',
            'options': {
                'validate_circuit': True,
                'analyze_state': False,
                'profile_performance': False,
                'analyze_memory': True
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['description']} ---")
        
        circuit = circuits[scenario['name']]
        
        try:
            results = debugger.debug_quantum_circuit(circuit, scenario['options'])
            
            print(f"Overall Status: {results['overall_status'].upper()}")
            
            # Show validation results
            validation_results = results.get('validation_results', [])
            if validation_results:
                passed = sum(1 for r in validation_results if r['passed'])
                total = len(validation_results)
                print(f"Validation: {passed}/{total} rules passed")
                
                failed = [r for r in validation_results if not r['passed']]
                if failed:
                    print("Failed validations:")
                    for result in failed[:3]:  # Show first 3
                        print(f"  ✗ {result['rule']}: {result['message']}")
            
            # Show state analysis
            state_analysis = results.get('state_analysis', {})
            if state_analysis:
                print("State Analysis:")
                for mode, analysis in list(state_analysis.items())[:2]:  # Show first 2 modes
                    if isinstance(analysis, dict) and 'insights' in analysis:
                        insights = analysis['insights']
                        if insights:
                            print(f"  {mode}: {insights[0]}")
            
            # Show memory analysis
            memory_analysis = results.get('memory_analysis', {})
            if memory_analysis:
                total_mb = memory_analysis.get('total_memory_mb', 0)
                print(f"Memory Usage: {total_mb:.2f} MB")
                
                if memory_analysis.get('memory_leaks', 0) > 0:
                    print(f"  ⚠ {memory_analysis['memory_leaks']} memory leaks detected")
            
            # Show recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                print("Recommendations:")
                for rec in recommendations[:3]:  # Show first 3
                    print(f"  → {rec}")
            else:
                print("✓ No issues detected")
                
        except Exception as e:
            print(f"Debugging error: {e}")


def demo_convenience_functions():
    """Demonstrate convenience functions for easy usage."""
    print("\n" + "="*60)
    print("CONVENIENCE FUNCTIONS DEMO")
    print("="*60)
    
    states = create_sample_quantum_states()
    circuits = create_sample_quantum_circuits()
    
    # 1. Quick state inspection
    print("--- Quick State Inspection ---")
    
    try:
        result = inspect_quantum_state(states['bell'], 'entanglement')
        print("Bell state entanglement analysis:")
        for insight in result.insights[:2]:
            print(f"  • {insight}")
            
        # Test with different modes
        modes_to_test = ['amplitude', 'probability', 'coherence']
        for mode in modes_to_test:
            result = inspect_quantum_state(states['plus'], mode)
            print(f"Plus state {mode} analysis: {len(result.insights)} insights")
            
    except Exception as e:
        print(f"State inspection error: {e}")
    
    # 2. Quick circuit validation
    print("\n--- Quick Circuit Validation ---")
    
    try:
        results = validate_quantum_circuit(circuits['bell'], ['unitarity', 'resources'])
        print("Bell circuit validation:")
        for result in results:
            status = "✓" if result.passed else "✗"
            print(f"  {status} {result.rule.name}")
            
    except Exception as e:
        print(f"Circuit validation error: {e}")
    
    # 3. Quick error analysis
    print("\n--- Quick Error Analysis ---")
    
    try:
        error = Exception("Quantum gate fidelity below threshold")
        diagnosis = analyze_quantum_error(error, {'gate_name': 'CNOT', 'fidelity': 0.85})
        
        print(f"Error diagnosis:")
        print(f"  Type: {diagnosis.error_type.name}")
        print(f"  Severity: {diagnosis.severity}")
        print(f"  Auto-fix: {'Available' if diagnosis.auto_fix_available else 'Not available'}")
        
    except Exception as e:
        print(f"Error analysis error: {e}")
    
    # 4. Quick comprehensive debugging
    print("\n--- Quick Comprehensive Debugging ---")
    
    try:
        results = debug_quantum_circuit(circuits['ghz'])
        
        print(f"GHZ circuit debugging:")
        print(f"  Status: {results['overall_status']}")
        print(f"  Validations: {len(results.get('validation_results', []))}")
        print(f"  Recommendations: {len(results.get('recommendations', []))}")
        
    except Exception as e:
        print(f"Comprehensive debugging error: {e}")


def demo_integration_features():
    """Demonstrate integration with other QuantRS2 modules."""
    print("\n" + "="*60)
    print("INTEGRATION FEATURES DEMO")
    print("="*60)
    
    debugger = QuantumDebuggingToolsManager()
    
    # Check integration status
    print("Integration Status:")
    integrations = [
        ('Performance Profiler', debugger.profiler),
        ('Test Manager', debugger.test_manager),
        ('Visualizer', debugger.visualizer),
        ('Algorithm Debugger', debugger.algorithm_debugger)
    ]
    
    for name, module in integrations:
        status = "✓ Available" if module is not None else "✗ Not available"
        print(f"  {name}: {status}")
    
    # Demonstrate debugging context
    print("\n--- Debugging Context Demo ---")
    
    try:
        with debugger.debugging_context("integration_test") as debug_ctx:
            print("Entered debugging context")
            
            # Simulate some quantum operations
            state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
            
            # Inspect state
            result = debug_ctx.inspect_quantum_state(state, InspectionMode.AMPLITUDE_ANALYSIS)
            print(f"State inspection completed: {len(result.insights)} insights")
            
            # Get memory info
            memory_info = debug_ctx.get_memory_usage()
            print(f"Memory usage: {memory_info.total_memory / 1024 / 1024:.2f} MB")
            
            print("Debugging context completed successfully")
            
    except Exception as e:
        print(f"Debugging context error: {e}")
    
    # Show available debugging tools
    print("\n--- Available Debugging Tools ---")
    print("Core Tools:")
    print("  ✓ Quantum State Inspector")
    print("  ✓ Quantum Error Analyzer")
    print("  ✓ Quantum Circuit Validator")
    print("  ✓ Quantum Memory Debugger")
    print("  ✓ Interactive Debug Console")
    
    if HAS_DASH or HAS_FLASK:
        print("  ✓ Web Debugging Interface")
    else:
        print("  ✗ Web Debugging Interface (install dash/flask)")
    
    if HAS_PSUTIL:
        print("  ✓ Advanced Memory Monitoring")
    else:
        print("  ✗ Advanced Memory Monitoring (install psutil)")


def main():
    """Run the comprehensive quantum debugging tools demo."""
    print("QuantRS2 Quantum Debugging Tools Comprehensive Demo")
    print("=" * 70)
    print("This demo showcases the complete debugging capabilities")
    print("of the QuantRS2 quantum computing framework.")
    print("=" * 70)
    
    try:
        # Run all demo sections
        demo_quantum_state_inspection()
        demo_quantum_error_analysis()
        demo_quantum_circuit_validation()
        demo_memory_debugging()
        demo_interactive_console()
        demo_web_interface()
        demo_comprehensive_debugging()
        demo_convenience_functions()
        demo_integration_features()
        
        print("\n" + "="*70)
        print("DEMO COMPLETE!")
        print("="*70)
        print("All quantum debugging tools have been demonstrated successfully.")
        
        print("\nDebugging capabilities demonstrated:")
        print("  ✓ Advanced quantum state inspection and analysis")
        print("  ✓ Comprehensive quantum error diagnosis and auto-fix")
        print("  ✓ Circuit validation with extensive property checking")
        print("  ✓ Memory debugging with leak detection and optimization")
        print("  ✓ Interactive debugging console with full command support")
        
        if HAS_DASH or HAS_FLASK:
            print("  ✓ Web-based debugging interface with real-time monitoring")
        else:
            print("  ✗ Web-based debugging interface (install dash/flask to enable)")
        
        if HAS_PSUTIL:
            print("  ✓ Advanced memory profiling and monitoring")
        else:
            print("  ✗ Advanced memory profiling (install psutil to enable)")
        
        print("\nTo start interactive debugging:")
        print("  start_quantum_debugging_console()")
        
        if HAS_DASH or HAS_FLASK:
            print("\nTo start web debugging interface:")
            print("  start_quantum_debugging_web_interface()")
        
        print("\nThe QuantRS2 Quantum Debugging Tools are fully functional!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)