#!/usr/bin/env python3
"""
Quantum Circuit Profiler Demo

This example demonstrates how to use QuantRS2's circuit profiler to analyze
performance characteristics, identify bottlenecks, and optimize quantum circuits.
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import quantrs2
    from quantrs2 import Circuit
    from quantrs2.profiler import (
        CircuitProfiler, ProfilerSession, 
        profile_circuit, compare_circuits
    )
    HAS_QUANTRS2 = True
except ImportError:
    print("QuantRS2 not available. Please install QuantRS2 first.")
    HAS_QUANTRS2 = False
    exit(1)


def demo_basic_profiling():
    """Demonstrate basic circuit profiling."""
    print("="*60)
    print("Basic Circuit Profiling")
    print("="*60)
    
    # Create a Bell state circuit
    circuit = Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    
    print("Profiling Bell state circuit...")
    print("  H gate on qubit 0")
    print("  CNOT gate from qubit 0 to qubit 1")
    
    # Profile the circuit
    report = profile_circuit(circuit, run_simulation=True)
    
    # Display results
    print(f"\nProfiling Results:")
    print(f"  Execution time: {report.metrics.execution_time*1000:.2f} ms")
    print(f"  Memory usage: {report.metrics.peak_memory_mb:.2f} MB")
    print(f"  Gate count: {report.metrics.gate_count}")
    print(f"  Circuit depth: {report.metrics.circuit_depth}")
    print(f"  Gates per second: {report.metrics.gates_per_second:.0f}")
    print(f"  Memory per qubit: {report.metrics.memory_per_qubit:.2f} MB")
    
    print(f"\nCircuit Analysis:")
    print(f"  Gate distribution: {dict(report.analysis.gate_distribution)}")
    print(f"  Qubit utilization: {dict(report.analysis.qubit_utilization)}")
    print(f"  Parallelism factor: {report.analysis.parallelism_factor:.2f}")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")


def demo_circuit_comparison():
    """Demonstrate comparing different circuit implementations."""
    print("\n" + "="*60)
    print("Circuit Comparison Demo")
    print("="*60)
    
    # Create different circuit implementations
    circuits = []
    names = []
    
    # 1. Simple Bell state
    bell_simple = Circuit(2)
    bell_simple.h(0)
    bell_simple.cx(0, 1)
    circuits.append(bell_simple)
    names.append("Bell State (Simple)")
    
    # 2. Bell state with extra operations
    bell_complex = Circuit(2)
    bell_complex.h(0)
    bell_complex.rz(0, np.pi/4)  # Add phase
    bell_complex.cx(0, 1)
    bell_complex.ry(1, np.pi/8)  # Add rotation
    circuits.append(bell_complex)
    names.append("Bell State (Complex)")
    
    # 3. GHZ state (3 qubits)
    ghz_state = Circuit(3)
    ghz_state.h(0)
    ghz_state.cx(0, 1)
    ghz_state.cx(1, 2)
    circuits.append(ghz_state)
    names.append("GHZ State")
    
    # 4. Random circuit
    random_circuit = Circuit(4)
    np.random.seed(42)
    for _ in range(10):
        qubit = np.random.randint(4)
        gate_type = np.random.choice(['h', 'x', 'y', 'z'])
        if gate_type == 'h':
            random_circuit.h(qubit)
        elif gate_type == 'x':
            random_circuit.x(qubit)
        elif gate_type == 'y':
            random_circuit.y(qubit)
        elif gate_type == 'z':
            random_circuit.z(qubit)
    
    # Add some entangling gates
    for i in range(3):
        random_circuit.cx(i, i+1)
    
    circuits.append(random_circuit)
    names.append("Random Circuit")
    
    print(f"Comparing {len(circuits)} different circuits...")
    
    # Compare circuits
    comparison_df = compare_circuits(circuits, names)
    
    print("\nComparison Results:")
    print("=" * 80)
    
    # Format the DataFrame for better display
    display_columns = [
        'circuit_id', 'n_qubits', 'n_gates', 'circuit_depth',
        'execution_time_ms', 'memory_mb', 'gates_per_second'
    ]
    
    display_df = comparison_df[display_columns].copy()
    display_df.columns = [
        'ID', 'Qubits', 'Gates', 'Depth', 
        'Time (ms)', 'Memory (MB)', 'Gates/sec'
    ]
    
    # Round numeric columns
    for col in ['Time (ms)', 'Memory (MB)', 'Gates/sec']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    
    print(display_df.to_string(index=False))
    
    # Find best performers
    best_time = comparison_df.loc[comparison_df['execution_time_ms'].idxmin()]
    best_memory = comparison_df.loc[comparison_df['memory_mb'].idxmin()]
    best_throughput = comparison_df.loc[comparison_df['gates_per_second'].idxmax()]
    
    print(f"\nBest Performers:")
    print(f"  Fastest execution: {names[best_time['circuit_id']]} ({best_time['execution_time_ms']:.2f} ms)")
    print(f"  Lowest memory: {names[best_memory['circuit_id']]} ({best_memory['memory_mb']:.2f} MB)")
    print(f"  Highest throughput: {names[best_throughput['circuit_id']]} ({best_throughput['gates_per_second']:.0f} gates/sec)")


def demo_profiler_session():
    """Demonstrate using ProfilerSession for workflow management."""
    print("\n" + "="*60)
    print("Profiler Session Demo")
    print("="*60)
    
    # Create a profiling session
    session = ProfilerSession("optimization_study")
    
    print("Starting profiling session: 'optimization_study'")
    
    # Profile a series of circuit optimizations
    base_circuit = Circuit(3)
    base_circuit.h(0)
    base_circuit.h(1)
    base_circuit.h(2)
    base_circuit.cx(0, 1)
    base_circuit.cx(1, 2)
    base_circuit.cx(0, 2)
    
    # Version 1: Original circuit
    session.profile(base_circuit, name="original")
    print("  ✓ Profiled original circuit")
    
    # Version 2: Optimized gate order
    optimized_circuit = Circuit(3)
    optimized_circuit.h(0)
    optimized_circuit.h(1)
    optimized_circuit.h(2)
    # Reorder gates for better parallelism
    optimized_circuit.cx(0, 1)
    optimized_circuit.cx(0, 2)  # Can run in parallel with above
    optimized_circuit.cx(1, 2)
    
    session.profile(optimized_circuit, name="optimized_order")
    print("  ✓ Profiled optimized gate order")
    
    # Version 3: Reduced depth version
    reduced_circuit = Circuit(3)
    reduced_circuit.h(0)
    reduced_circuit.h(1)
    reduced_circuit.h(2)
    reduced_circuit.cx(0, 1)
    reduced_circuit.cx(1, 2)
    # Skip the third CNOT to reduce depth
    
    session.profile(reduced_circuit, name="reduced_depth")
    print("  ✓ Profiled reduced depth version")
    
    # Get session summary
    summary = session.summary()
    print(f"\nSession Summary:")
    print("=" * 50)
    
    if not summary.empty:
        # Display key metrics
        key_columns = ['n_gates', 'circuit_depth', 'execution_time_ms', 'parallelism_factor']
        available_columns = [col for col in key_columns if col in summary.columns]
        
        if available_columns:
            display_summary = summary[available_columns].copy()
            display_summary.index = ['Original', 'Optimized Order', 'Reduced Depth']
            print(display_summary.round(3).to_string())
        
        # Analyze improvements
        if len(summary) > 1:
            original_time = summary.iloc[0]['execution_time_ms']
            best_time = summary['execution_time_ms'].min()
            improvement = ((original_time - best_time) / original_time) * 100
            print(f"\nBest optimization achieved {improvement:.1f}% time improvement")
    
    # Save session (commented out to avoid file creation in demo)
    # output_path = session.save_session("profiler_results")
    # print(f"Session saved to {output_path}")


def demo_advanced_profiling():
    """Demonstrate advanced profiling features."""
    print("\n" + "="*60)
    print("Advanced Profiling Features")
    print("="*60)
    
    # Create a complex quantum circuit
    circuit = Circuit(4)
    
    # Add various gate types
    circuit.h(0)
    circuit.rx(1, np.pi/4)
    circuit.ry(2, np.pi/3)
    circuit.rz(3, np.pi/6)
    
    # Add entangling gates
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.cx(3, 0)  # Create a ring
    
    # Add more single-qubit gates
    circuit.h(1)
    circuit.h(2)
    
    print("Profiling complex 4-qubit circuit with various gate types...")
    
    # Create profiler with custom options
    profiler = CircuitProfiler(
        enable_memory_profiling=True,
        enable_timing_breakdown=True,
        enable_resource_tracking=True
    )
    
    # Profile with different simulation backends
    backends = ['cpu']  # GPU would be 'gpu' if available
    
    for backend in backends:
        print(f"\nProfiling with {backend.upper()} backend:")
        
        simulation_params = {'backend': backend}
        report = profiler.profile_circuit(
            circuit, 
            run_simulation=True,
            simulation_params=simulation_params
        )
        
        # Display detailed metrics
        print(f"  Performance Metrics:")
        print(f"    Execution time: {report.metrics.execution_time*1000:.2f} ms")
        print(f"    Memory usage: {report.metrics.peak_memory_mb:.2f} MB")
        print(f"    CPU usage: {report.metrics.cpu_usage_percent:.1f}%")
        print(f"    Gates per second: {report.metrics.gates_per_second:.0f}")
        
        print(f"  Circuit Characteristics:")
        print(f"    Total gates: {report.metrics.gate_count}")
        print(f"    Single-qubit gates: {report.metrics.single_qubit_gates}")
        print(f"    Entangling gates: {report.metrics.entangling_gates}")
        print(f"    Circuit depth: {report.metrics.circuit_depth}")
        print(f"    Parallelism factor: {report.analysis.parallelism_factor:.2f}")
        
        print(f"  Resource Efficiency:")
        print(f"    Time per gate: {report.metrics.time_per_gate*1e6:.2f} μs")
        print(f"    Memory per qubit: {report.metrics.memory_per_qubit:.2f} MB")
        
        if report.analysis.highly_connected_qubits:
            print(f"  Highly connected qubits: {report.analysis.highly_connected_qubits}")
        
        print(f"  Top recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):
            print(f"    {i}. {rec}")


def demo_performance_analysis():
    """Demonstrate performance scaling analysis."""
    print("\n" + "="*60)
    print("Performance Scaling Analysis")
    print("="*60)
    
    # Create circuits of different sizes
    qubit_counts = [2, 3, 4, 5]
    reports = []
    
    print("Analyzing performance scaling with circuit size...")
    
    for n_qubits in qubit_counts:
        print(f"  Testing {n_qubits}-qubit circuit...")
        
        # Create circuit with consistent structure
        circuit = Circuit(n_qubits)
        
        # Add Hadamard gates to all qubits
        for i in range(n_qubits):
            circuit.h(i)
        
        # Add entangling gates in a ring
        for i in range(n_qubits):
            circuit.cx(i, (i + 1) % n_qubits)
        
        # Profile the circuit
        report = profile_circuit(circuit)
        report.circuit_info['n_qubits_label'] = f"{n_qubits}_qubits"
        reports.append(report)
    
    # Create comparison
    profiler = CircuitProfiler()
    comparison_df = profiler.compare_circuits(reports)
    
    print(f"\nScaling Analysis Results:")
    print("=" * 60)
    
    # Display scaling metrics
    scaling_columns = ['n_qubits', 'n_gates', 'execution_time_ms', 'memory_mb', 'gates_per_second']
    available_columns = [col for col in scaling_columns if col in comparison_df.columns]
    
    if available_columns:
        scaling_df = comparison_df[available_columns].copy()
        scaling_df.columns = ['Qubits', 'Gates', 'Time (ms)', 'Memory (MB)', 'Gates/sec']
        
        # Round for display
        for col in ['Time (ms)', 'Memory (MB)', 'Gates/sec']:
            if col in scaling_df.columns:
                scaling_df[col] = scaling_df[col].round(2)
        
        print(scaling_df.to_string(index=False))
        
        # Analyze scaling trends
        if len(comparison_df) > 1:
            time_growth = comparison_df['execution_time_ms'].iloc[-1] / comparison_df['execution_time_ms'].iloc[0]
            memory_growth = comparison_df['memory_mb'].iloc[-1] / comparison_df['memory_mb'].iloc[0]
            qubit_growth = comparison_df['n_qubits'].iloc[-1] / comparison_df['n_qubits'].iloc[0]
            
            print(f"\nScaling Analysis:")
            print(f"  Qubit count increased by: {qubit_growth:.1f}x")
            print(f"  Execution time increased by: {time_growth:.1f}x")
            print(f"  Memory usage increased by: {memory_growth:.1f}x")
            
            # Estimate complexity
            theoretical_memory = 2 ** comparison_df['n_qubits'].iloc[-1] / 2 ** comparison_df['n_qubits'].iloc[0]
            print(f"  Theoretical memory scaling (2^n): {theoretical_memory:.1f}x")
            
            if memory_growth < theoretical_memory * 0.8:
                print("  → Memory scaling better than exponential (good optimization)")
            else:
                print("  → Memory scaling close to exponential (expected for state vectors)")


def main():
    """Run all profiler demos."""
    print("QuantRS2 Quantum Circuit Profiler Demo")
    print("This demo shows how to analyze and optimize quantum circuit performance")
    
    try:
        # Run all demos
        demo_basic_profiling()
        demo_circuit_comparison()
        demo_profiler_session()
        demo_advanced_profiling()
        demo_performance_analysis()
        
        print("\n" + "="*60)
        print("Demo Complete!")
        print("="*60)
        print("\nKey takeaways:")
        print("• Circuit profiler provides comprehensive performance analysis")
        print("• Compare different implementations to find optimal solutions")
        print("• Use sessions to track optimization workflows")
        print("• Memory and timing breakdowns help identify bottlenecks")
        print("• Scaling analysis reveals algorithmic complexity")
        print("• Recommendations guide optimization strategies")
        
        print(f"\nNext steps:")
        print("• Profile your own circuits with profile_circuit()")
        print("• Use ProfilerSession for systematic optimization")
        print("• Compare different algorithms with compare_circuits()")
        print("• Export detailed reports for documentation")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("This may be due to missing dependencies or system limitations.")
        print("The profiler provides fallback implementations for basic functionality.")


if __name__ == "__main__":
    if HAS_QUANTRS2:
        main()
    else:
        print("Please install QuantRS2 to run this demo.")