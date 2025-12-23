#!/usr/bin/env python3
"""
Multi-GPU Quantum Circuit Simulation Demo

This example demonstrates the multi-GPU capabilities of QuantRS2,
including device detection, allocation strategies, and distributed execution.

Requirements:
    - CUDA-capable GPUs (for actual GPU execution)
    - QuantRS2 compiled with GPU support (cargo build --features gpu)

Note: This example will work in mock mode even without GPUs.
"""

import sys
import time
import numpy as np

try:
    from quantrs2 import (
        Circuit,
        PyMultiGpuManager,
        get_gpu_count,
        is_multi_gpu_available,
    )
except ImportError as e:
    print(f"Error importing QuantRS2: {e}")
    print("Please ensure QuantRS2 is properly installed")
    print("Build with: maturin develop --features=gpu")
    sys.exit(1)


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def demo_gpu_detection():
    """Demonstrate GPU detection capabilities."""
    print_section("GPU Detection")

    # Check if multi-GPU support is available
    available = is_multi_gpu_available()
    print(f"Multi-GPU support available: {available}")

    # Get the number of GPUs
    gpu_count = get_gpu_count()
    print(f"Number of GPUs detected: {gpu_count}")

    # Create a multi-GPU manager
    try:
        manager = PyMultiGpuManager()
        print(f"\nMulti-GPU Manager created successfully!")
        print(f"Manager: {manager}")

        # Get detailed device information
        devices = manager.get_devices()
        print(f"\nDetected {len(devices)} GPU device(s):")
        for i, device in enumerate(devices):
            print(f"\n  Device {i}:")
            print(f"    ID: {device['device_id']}")
            print(f"    Name: {device['name']}")
            print(f"    Total Memory: {device['total_memory'] / (1024**3):.2f} GB")
            print(f"    Available Memory: {device['available_memory'] / (1024**3):.2f} GB")
            print(f"    Compute Capability: {device['compute_capability']}")
            print(f"    Multiprocessors: {device['multiprocessor_count']}")
            print(f"    Max Threads/Block: {device['max_threads_per_block']}")
            print(f"    Available: {device['is_available']}")

        return manager

    except Exception as e:
        print(f"Error creating Multi-GPU Manager: {e}")
        return None


def demo_allocation_strategies(manager):
    """Demonstrate different allocation strategies."""
    if manager is None:
        print("Skipping allocation strategies (no manager available)")
        return

    print_section("Allocation Strategies")

    strategies = [
        "single_gpu",
        "round_robin",
        "memory_based",
        "performance_based",
        "adaptive"
    ]

    n_qubits_tests = [10, 20, 30]

    for strategy in strategies:
        print(f"\n{strategy.upper().replace('_', ' ')}:")
        manager.set_strategy(strategy)

        for n_qubits in n_qubits_tests:
            try:
                selected_gpus = manager.select_gpus(n_qubits)
                print(f"  {n_qubits} qubits -> GPUs: {selected_gpus}")
            except Exception as e:
                print(f"  {n_qubits} qubits -> Error: {e}")


def demo_circuit_execution(manager):
    """Demonstrate quantum circuit execution with multi-GPU."""
    if manager is None:
        print("Skipping circuit execution (no manager available)")
        return

    print_section("Multi-GPU Circuit Execution")

    # Create a quantum circuit
    n_qubits = 10
    print(f"\nCreating {n_qubits}-qubit circuit...")

    circuit = Circuit(n_qubits)

    # Build a complex circuit
    # Apply Hadamard gates to create superposition
    for i in range(n_qubits):
        circuit.h(i)

    # Apply CNOT gates to create entanglement
    for i in range(n_qubits - 1):
        circuit.cnot(i, i + 1)

    # Apply rotation gates
    for i in range(n_qubits):
        circuit.rx(i, np.pi / 4)
        circuit.ry(i, np.pi / 3)
        circuit.rz(i, np.pi / 6)

    print(f"Circuit created with {n_qubits} qubits")

    # Execute with different strategies
    strategies = ["single_gpu", "adaptive"]

    for strategy in strategies:
        print(f"\n{strategy.upper().replace('_', ' ')} execution:")
        manager.set_strategy(strategy)

        try:
            # Select GPUs
            selected_gpus = manager.select_gpus(n_qubits)
            print(f"  Selected GPUs: {selected_gpus}")

            # Execute circuit
            start_time = time.time()

            # Run with GPU if available, otherwise fallback to CPU
            if get_gpu_count() > 0:
                result = circuit.run(use_gpu=True)
            else:
                print("  No GPU available, using CPU simulation...")
                result = circuit.run(use_gpu=False)

            elapsed = time.time() - start_time

            # Get results
            probs = result.probabilities()
            print(f"  Execution time: {elapsed * 1000:.2f} ms")
            print(f"  State vector size: {len(probs)} amplitudes")
            print(f"  Top 5 probabilities:")

            # Get indices of top probabilities
            prob_indices = np.argsort(probs)[::-1][:5]
            for idx in prob_indices:
                basis_state = format(idx, f'0{n_qubits}b')
                print(f"    |{basis_state}⟩: {probs[idx]:.6f}")

        except Exception as e:
            print(f"  Error during execution: {e}")


def demo_performance_metrics(manager):
    """Demonstrate performance metrics collection."""
    if manager is None:
        print("Skipping performance metrics (no manager available)")
        return

    print_section("Performance Metrics")

    # Reset metrics
    manager.reset_metrics()
    print("Metrics reset")

    # Execute a circuit (metrics will be collected)
    n_qubits = 8
    circuit = Circuit(n_qubits)

    # Simple Bell state circuit for testing
    circuit.h(0)
    circuit.cnot(0, 1)

    try:
        if get_gpu_count() > 0:
            result = circuit.run(use_gpu=True)
        else:
            result = circuit.run(use_gpu=False)
    except Exception as e:
        print(f"Error executing circuit: {e}")

    # Get metrics
    metrics = manager.get_metrics()
    print("\nCollected Metrics:")
    print(f"  Total execution time: {metrics['total_time_ms']:.2f} ms")
    print(f"  Memory transferred: {metrics['memory_transferred_bytes'] / (1024**2):.2f} MB")
    print(f"  Average GPU utilization: {metrics['avg_gpu_utilization'] * 100:.1f}%")

    if metrics['per_gpu_time_ms']:
        print("\n  Per-GPU execution time:")
        for gpu_id, time_ms in metrics['per_gpu_time_ms'].items():
            print(f"    GPU {gpu_id}: {time_ms:.2f} ms")

    if metrics['gates_per_gpu']:
        print("\n  Gates executed per GPU:")
        for gpu_id, gates in metrics['gates_per_gpu'].items():
            print(f"    GPU {gpu_id}: {gates} gates")


def demo_scalability():
    """Demonstrate scalability with different qubit counts."""
    print_section("Scalability Analysis")

    qubit_counts = [8, 12, 16, 20]

    print("\nExecuting circuits with varying qubit counts...")
    print(f"{'Qubits':<10} {'Time (ms)':<15} {'Memory (MB)':<15} {'GPU Used':<10}")
    print("-" * 50)

    for n_qubits in qubit_counts:
        circuit = Circuit(n_qubits)

        # Create entangled state
        circuit.h(0)
        for i in range(n_qubits - 1):
            circuit.cnot(i, i + 1)

        try:
            start = time.time()

            # Check if GPU is available
            use_gpu = get_gpu_count() > 0 and Circuit.is_gpu_available()
            result = circuit.run(use_gpu=use_gpu)

            elapsed = (time.time() - start) * 1000

            # Estimate memory usage
            state_size = 2 ** n_qubits
            memory_mb = (state_size * 16) / (1024 ** 2)  # Complex64 = 16 bytes

            gpu_status = "Yes" if use_gpu else "No"

            print(f"{n_qubits:<10} {elapsed:<15.2f} {memory_mb:<15.2f} {gpu_status:<10}")

        except Exception as e:
            print(f"{n_qubits:<10} {'Error':<15} {'N/A':<15} {'N/A':<10}")
            print(f"  Error: {e}")


def main():
    """Main demo function."""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║              QuantRS2 Multi-GPU Quantum Simulation Demo               ║
║                                                                       ║
║  This demo showcases the multi-GPU capabilities of QuantRS2:         ║
║    - Automatic GPU detection                                         ║
║    - Multiple allocation strategies                                  ║
║    - Distributed quantum circuit execution                           ║
║    - Performance metrics and monitoring                              ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Demo 1: GPU Detection
    manager = demo_gpu_detection()

    # Demo 2: Allocation Strategies
    demo_allocation_strategies(manager)

    # Demo 3: Circuit Execution
    demo_circuit_execution(manager)

    # Demo 4: Performance Metrics
    demo_performance_metrics(manager)

    # Demo 5: Scalability
    demo_scalability()

    print_section("Demo Complete")
    print("\nMulti-GPU demonstration completed successfully!")
    print("\nKey Takeaways:")
    print("  • QuantRS2 can automatically detect and utilize multiple GPUs")
    print("  • Multiple allocation strategies optimize for different scenarios")
    print("  • Performance metrics help identify bottlenecks")
    print("  • Scalability is limited by available GPU memory")
    print("\nFor production use:")
    print("  1. Compile with GPU support: cargo build --release --features=gpu")
    print("  2. Ensure CUDA drivers are properly installed")
    print("  3. Monitor GPU utilization with nvidia-smi")
    print("  4. Choose allocation strategy based on your workload")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
