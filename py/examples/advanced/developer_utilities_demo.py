#!/usr/bin/env python3
"""
Developer Utilities Demo

This example demonstrates the developer utilities for profiling,
debugging, and testing QuantRS2 code.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from quantrs2.dev_utils import (
    profile,
    profile_to_dict,
    retry,
    debug_mode,
    debug_print,
    analyze_circuit,
    print_circuit_analysis,
    run_quick_tests,
    compare_implementations,
    print_comparison,
    safe_execute,
    format_exception
)


# Example functions for demonstration
@profile
def slow_quantum_function():
    """Example function that will be profiled."""
    print("  Simulating quantum computation...")
    # Simulate some work
    result = sum([i**2 for i in range(100000)])
    time.sleep(0.1)  # Simulate I/O
    return result


@profile_to_dict
def quantum_computation_with_stats():
    """Example function that returns profiling data."""
    print("  Running quantum computation...")
    time.sleep(0.05)
    return {"result": "Bell state", "fidelity": 0.99}


@retry(max_attempts=3, delay=0.5)
def unreliable_quantum_operation():
    """Example function that might fail (for retry demo)."""
    import random
    if random.random() < 0.7:  # 70% chance to fail
        raise RuntimeError("Quantum decoherence detected!")
    return "Success!"


def main():
    """Run developer utilities demonstrations."""

    print("\n" + "="*80)
    print("QUANTRS2 DEVELOPER UTILITIES DEMONSTRATIONS")
    print("="*80 + "\n")

    # Example 1: Profiling decorator
    print("Example 1: Function Profiling")
    print("-" * 80)
    result = slow_quantum_function()
    print(f"Result: {result}\n")

    # Example 2: Profile with dict return
    print("\nExample 2: Profiling with Dictionary Return")
    print("-" * 80)
    result_dict = quantum_computation_with_stats()
    print(f"Original result: {result_dict['result']}")
    print(f"Execution time: {result_dict['execution_time_ms']:.2f} ms")
    print(f"Memory delta: {result_dict['memory_delta_mb']:.2f} MB\n")

    # Example 3: Retry decorator
    print("\nExample 3: Retry on Failure")
    print("-" * 80)
    try:
        result = unreliable_quantum_operation()
        print(f"✅ {result}\n")
    except RuntimeError as e:
        print(f"❌ Failed after all retries: {e}\n")

    # Example 4: Debug mode
    print("\nExample 4: Debug Mode")
    print("-" * 80)

    def create_quantum_circuit():
        """Example function that uses debug mode."""
        debug_print("Creating quantum circuit...")
        debug_print("Adding Hadamard gate")
        debug_print("Adding CNOT gate")
        return "circuit_created"

    print("Without debug mode:")
    result = create_quantum_circuit()

    print("\nWith debug mode:")
    with debug_mode():
        result = create_quantum_circuit()

    # Example 5: Circuit analysis
    print("\nExample 5: Circuit Analysis (Simulated)")
    print("-" * 80)

    # Create a mock circuit object
    class MockCircuit:
        def __init__(self):
            self.num_qubits = 5
            self.num_gates = 42
            self.depth = 15

    mock_circuit = MockCircuit()
    print_circuit_analysis(mock_circuit)

    # Example 6: Quick tests
    print("\nExample 6: Quick Sanity Tests")
    print("-" * 80)
    test_results = run_quick_tests()

    # Example 7: Implementation comparison
    print("\nExample 7: Performance Comparison")
    print("-" * 80)

    def implementation_a():
        """Naive implementation."""
        return sum([i**2 for i in range(10000)])

    def implementation_b():
        """Optimized implementation."""
        return sum(i**2 for i in range(10000))

    def implementation_c():
        """List comprehension."""
        squares = [i**2 for i in range(10000)]
        return sum(squares)

    implementations = {
        'Naive (with list)': implementation_a,
        'Generator': implementation_b,
        'List comprehension': implementation_c,
    }

    results = compare_implementations(implementations, n_runs=100)
    print_comparison(results)

    # Example 8: Safe execution
    print("\nExample 8: Safe Execution")
    print("-" * 80)

    def risky_function(x):
        if x < 0:
            raise ValueError("Negative value not allowed!")
        return x ** 2

    # Safe execution
    result, error = safe_execute(risky_function, 5)
    if error is None:
        print(f"✅ Risky function succeeded: {result}")
    else:
        print(f"❌ Error: {error}")

    # This will fail
    result, error = safe_execute(risky_function, -5)
    if error is None:
        print(f"✅ Risky function succeeded: {result}")
    else:
        print(f"❌ Error caught safely: {format_exception(error, include_traceback=False)}")

    # Example 9: Exception formatting
    print("\nExample 9: Exception Formatting")
    print("-" * 80)

    try:
        # Trigger an exception
        raise RuntimeError("Quantum decoherence detected at T2=100ns")
    except Exception as e:
        print("Short format:")
        print(format_exception(e, include_traceback=False))

        print("\nWith traceback:")
        print(format_exception(e, include_traceback=True))

    # Example 10: Development configuration
    print("\nExample 10: Development Configuration")
    print("-" * 80)

    from quantrs2.dev_utils import get_dev_config

    dev_config = get_dev_config()

    # Set some configuration
    dev_config.set('debug_mode', True)
    dev_config.set('verbose_errors', True)
    dev_config.set('custom_setting', 'quantum_advantage')

    print("Development configuration:")
    for key, value in dev_config.config.items():
        print(f"  {key}: {value}")

    print(f"\nConfig file location: {dev_config.config_file}")

    # Summary
    print("\n" + "="*80)
    print("DEVELOPER UTILITIES DEMO COMPLETED")
    print("="*80)
    print("\nAvailable utilities:")
    print("  ✅ @profile - Profile execution time and memory")
    print("  ✅ @profile_to_dict - Profile with dict return")
    print("  ✅ @retry - Automatic retry on failure")
    print("  ✅ debug_mode() - Context manager for debug logging")
    print("  ✅ analyze_circuit() - Circuit statistics")
    print("  ✅ run_quick_tests() - Quick sanity tests")
    print("  ✅ compare_implementations() - Benchmark functions")
    print("  ✅ safe_execute() - Safe function execution")
    print("  ✅ format_exception() - Pretty exception formatting")
    print("  ✅ DevConfig - Development configuration management")
    print("\nUse these utilities to:")
    print("  - Profile and optimize quantum algorithms")
    print("  - Debug complex quantum circuits")
    print("  - Compare different implementations")
    print("  - Handle errors gracefully")
    print("  - Manage development settings")
    print("\n")


if __name__ == "__main__":
    main()
