#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Suite for QuantRS2

This module provides tools for benchmarking quantum circuit execution across
multiple frameworks, comparing performance, memory usage, and accuracy.

Features:
    - Multi-framework support (QuantRS2, Qiskit, Cirq, PennyLane)
    - Automated benchmark execution and reporting
    - Statistical analysis of performance metrics
    - Visual comparison charts
    - Export to multiple formats (JSON, CSV, HTML)
"""

import time
import sys
import psutil
import gc
import json
import csv
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
from pathlib import Path

import numpy as np

# Try to import all supported frameworks
try:
    from quantrs2 import Circuit as QuantRS2Circuit
    QUANTRS2_AVAILABLE = True
except ImportError:
    QUANTRS2_AVAILABLE = False

try:
    from qiskit import QuantumCircuit, execute, Aer
    from qiskit.circuit.library import QFT
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


class FrameworkType(Enum):
    """Supported quantum computing frameworks."""
    QUANTRS2 = "quantrs2"
    QISKIT = "qiskit"
    CIRQ = "cirq"
    PENNYLANE = "pennylane"


class BenchmarkType(Enum):
    """Types of benchmarks to run."""
    BELL_STATE = "bell_state"
    GHZ_STATE = "ghz_state"
    QFT = "quantum_fourier_transform"
    RANDOM_CIRCUIT = "random_circuit"
    DEEP_CIRCUIT = "deep_circuit"
    WIDE_CIRCUIT = "wide_circuit"
    SUPREMACY_CIRCUIT = "supremacy_circuit"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    framework: str
    benchmark_type: str
    n_qubits: int
    n_gates: int
    execution_time_ms: float
    memory_mb: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    fidelity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkStats:
    """Statistical summary of multiple benchmark runs."""
    framework: str
    benchmark_type: str
    n_qubits: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    mean_memory_mb: float
    success_rate: float
    num_runs: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class PerformanceBenchmark:
    """Main benchmarking class for quantum computing frameworks."""

    def __init__(self, frameworks: Optional[List[FrameworkType]] = None):
        """
        Initialize the benchmarking suite.

        Args:
            frameworks: List of frameworks to benchmark. If None, use all available.
        """
        self.frameworks = frameworks or self._detect_available_frameworks()
        self.results: List[BenchmarkResult] = []

        # Check framework availability
        self.framework_available = {
            FrameworkType.QUANTRS2: QUANTRS2_AVAILABLE,
            FrameworkType.QISKIT: QISKIT_AVAILABLE,
            FrameworkType.CIRQ: CIRQ_AVAILABLE,
            FrameworkType.PENNYLANE: PENNYLANE_AVAILABLE,
        }

        print(f"Initialized benchmark suite with {len(self.frameworks)} framework(s)")
        for fw in self.frameworks:
            status = "✓" if self.framework_available[fw] else "✗"
            print(f"  {status} {fw.value}")

    def _detect_available_frameworks(self) -> List[FrameworkType]:
        """Detect which frameworks are available."""
        available = []
        if QUANTRS2_AVAILABLE:
            available.append(FrameworkType.QUANTRS2)
        if QISKIT_AVAILABLE:
            available.append(FrameworkType.QISKIT)
        if CIRQ_AVAILABLE:
            available.append(FrameworkType.CIRQ)
        if PENNYLANE_AVAILABLE:
            available.append(FrameworkType.PENNYLANE)
        return available

    def run_benchmark(
        self,
        benchmark_type: BenchmarkType,
        n_qubits: int,
        num_runs: int = 5,
        warmup_runs: int = 2,
    ) -> List[BenchmarkResult]:
        """
        Run a benchmark across all frameworks.

        Args:
            benchmark_type: Type of benchmark to run
            n_qubits: Number of qubits
            num_runs: Number of runs for averaging
            warmup_runs: Number of warmup runs (not counted)

        Returns:
            List of benchmark results
        """
        print(f"\n{'='*70}")
        print(f"Running {benchmark_type.value} with {n_qubits} qubits")
        print(f"{'='*70}")

        results = []

        for framework in self.frameworks:
            if not self.framework_available[framework]:
                print(f"Skipping {framework.value} (not available)")
                continue

            print(f"\n{framework.value}:")

            # Warmup runs
            for i in range(warmup_runs):
                try:
                    self._execute_benchmark(framework, benchmark_type, n_qubits)
                except Exception as e:
                    print(f"  Warmup run {i+1} failed: {e}")

            # Actual benchmark runs
            framework_results = []
            for run in range(num_runs):
                try:
                    result = self._execute_benchmark(framework, benchmark_type, n_qubits)
                    framework_results.append(result)
                    results.append(result)
                    print(f"  Run {run+1}/{num_runs}: {result.execution_time_ms:.2f} ms")
                except Exception as e:
                    error_result = BenchmarkResult(
                        framework=framework.value,
                        benchmark_type=benchmark_type.value,
                        n_qubits=n_qubits,
                        n_gates=0,
                        execution_time_ms=0.0,
                        memory_mb=0.0,
                        cpu_percent=0.0,
                        success=False,
                        error_message=str(e),
                    )
                    results.append(error_result)
                    print(f"  Run {run+1}/{num_runs}: FAILED - {e}")

            # Calculate statistics
            if framework_results:
                times = [r.execution_time_ms for r in framework_results if r.success]
                if times:
                    print(f"  Average: {statistics.mean(times):.2f} ms")
                    print(f"  Std Dev: {statistics.stdev(times):.2f} ms" if len(times) > 1 else "  Std Dev: N/A")

        self.results.extend(results)
        return results

    def _execute_benchmark(
        self,
        framework: FrameworkType,
        benchmark_type: BenchmarkType,
        n_qubits: int,
    ) -> BenchmarkResult:
        """Execute a single benchmark run."""
        # Force garbage collection before benchmark
        gc.collect()

        # Get initial memory
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Start timing
        start_time = time.perf_counter()

        # Execute framework-specific benchmark
        n_gates = 0
        try:
            if framework == FrameworkType.QUANTRS2:
                n_gates = self._run_quantrs2(benchmark_type, n_qubits)
            elif framework == FrameworkType.QISKIT:
                n_gates = self._run_qiskit(benchmark_type, n_qubits)
            elif framework == FrameworkType.CIRQ:
                n_gates = self._run_cirq(benchmark_type, n_qubits)
            elif framework == FrameworkType.PENNYLANE:
                n_gates = self._run_pennylane(benchmark_type, n_qubits)

            success = True
            error_message = None
        except Exception as e:
            success = False
            error_message = str(e)

        # End timing
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # ms

        # Get final memory and CPU
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = mem_after - mem_before
        cpu_percent = process.cpu_percent()

        return BenchmarkResult(
            framework=framework.value,
            benchmark_type=benchmark_type.value,
            n_qubits=n_qubits,
            n_gates=n_gates,
            execution_time_ms=execution_time,
            memory_mb=memory_used,
            cpu_percent=cpu_percent,
            success=success,
            error_message=error_message,
        )

    def _run_quantrs2(self, benchmark_type: BenchmarkType, n_qubits: int) -> int:
        """Run benchmark on QuantRS2."""
        circuit = QuantRS2Circuit(n_qubits)

        if benchmark_type == BenchmarkType.BELL_STATE:
            circuit.h(0)
            circuit.cnot(0, 1)
            n_gates = 2

        elif benchmark_type == BenchmarkType.GHZ_STATE:
            circuit.h(0)
            for i in range(n_qubits - 1):
                circuit.cnot(i, i + 1)
            n_gates = 1 + (n_qubits - 1)

        elif benchmark_type == BenchmarkType.QFT:
            n_gates = 0
            # Simplified QFT
            for i in range(n_qubits):
                circuit.h(i)
                for j in range(i + 1, n_qubits):
                    circuit.rz(j, np.pi / (2 ** (j - i)))
                    n_gates += 1
                n_gates += 1

        elif benchmark_type == BenchmarkType.RANDOM_CIRCUIT:
            n_gates = 0
            np.random.seed(42)
            for _ in range(n_qubits * 10):
                qubit = np.random.randint(0, n_qubits)
                gate_type = np.random.randint(0, 4)

                if gate_type == 0:
                    circuit.h(qubit)
                elif gate_type == 1:
                    circuit.x(qubit)
                elif gate_type == 2:
                    circuit.rx(qubit, np.random.uniform(0, 2 * np.pi))
                elif gate_type == 3 and qubit < n_qubits - 1:
                    circuit.cnot(qubit, qubit + 1)

                n_gates += 1

        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}")

        # Run simulation
        result = circuit.run()
        _ = result.probabilities()  # Force computation

        return n_gates

    def _run_qiskit(self, benchmark_type: BenchmarkType, n_qubits: int) -> int:
        """Run benchmark on Qiskit."""
        circuit = QuantumCircuit(n_qubits)

        if benchmark_type == BenchmarkType.BELL_STATE:
            circuit.h(0)
            circuit.cx(0, 1)
            n_gates = 2

        elif benchmark_type == BenchmarkType.GHZ_STATE:
            circuit.h(0)
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)
            n_gates = 1 + (n_qubits - 1)

        elif benchmark_type == BenchmarkType.QFT:
            qft_circuit = QFT(n_qubits)
            circuit.compose(qft_circuit, inplace=True)
            n_gates = circuit.size()

        elif benchmark_type == BenchmarkType.RANDOM_CIRCUIT:
            np.random.seed(42)
            for _ in range(n_qubits * 10):
                qubit = np.random.randint(0, n_qubits)
                gate_type = np.random.randint(0, 4)

                if gate_type == 0:
                    circuit.h(qubit)
                elif gate_type == 1:
                    circuit.x(qubit)
                elif gate_type == 2:
                    circuit.rx(np.random.uniform(0, 2 * np.pi), qubit)
                elif gate_type == 3 and qubit < n_qubits - 1:
                    circuit.cx(qubit, qubit + 1)

            n_gates = circuit.size()

        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}")

        # Run simulation
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circuit, backend)
        result = job.result()
        _ = result.get_statevector()

        return n_gates

    def _run_cirq(self, benchmark_type: BenchmarkType, n_qubits: int) -> int:
        """Run benchmark on Cirq."""
        qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
        circuit = cirq.Circuit()

        if benchmark_type == BenchmarkType.BELL_STATE:
            circuit.append([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])])
            n_gates = 2

        elif benchmark_type == BenchmarkType.GHZ_STATE:
            circuit.append(cirq.H(qubits[0]))
            for i in range(n_qubits - 1):
                circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
            n_gates = 1 + (n_qubits - 1)

        elif benchmark_type == BenchmarkType.QFT:
            # Simplified QFT
            for i in range(n_qubits):
                circuit.append(cirq.H(qubits[i]))
                for j in range(i + 1, n_qubits):
                    circuit.append(cirq.CZ(qubits[i], qubits[j]) ** (1 / (2 ** (j - i))))
            n_gates = len(list(circuit.all_operations()))

        elif benchmark_type == BenchmarkType.RANDOM_CIRCUIT:
            np.random.seed(42)
            for _ in range(n_qubits * 10):
                qubit_idx = np.random.randint(0, n_qubits)
                gate_type = np.random.randint(0, 4)

                if gate_type == 0:
                    circuit.append(cirq.H(qubits[qubit_idx]))
                elif gate_type == 1:
                    circuit.append(cirq.X(qubits[qubit_idx]))
                elif gate_type == 2:
                    circuit.append(cirq.rx(np.random.uniform(0, 2 * np.pi))(qubits[qubit_idx]))
                elif gate_type == 3 and qubit_idx < n_qubits - 1:
                    circuit.append(cirq.CNOT(qubits[qubit_idx], qubits[qubit_idx + 1]))

            n_gates = len(list(circuit.all_operations()))

        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}")

        # Run simulation
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        _ = result.state_vector()

        return n_gates

    def _run_pennylane(self, benchmark_type: BenchmarkType, n_qubits: int) -> int:
        """Run benchmark on PennyLane."""
        dev = qml.device('default.qubit', wires=n_qubits)

        if benchmark_type == BenchmarkType.BELL_STATE:
            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                return qml.probs(wires=range(n_qubits))

            n_gates = 2

        elif benchmark_type == BenchmarkType.GHZ_STATE:
            @qml.qnode(dev)
            def circuit():
                qml.Hadamard(wires=0)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                return qml.probs(wires=range(n_qubits))

            n_gates = 1 + (n_qubits - 1)

        elif benchmark_type == BenchmarkType.QFT:
            @qml.qnode(dev)
            def circuit():
                qml.QFT(wires=range(n_qubits))
                return qml.probs(wires=range(n_qubits))

            # Estimate gate count for QFT
            n_gates = n_qubits * (n_qubits + 1) // 2

        elif benchmark_type == BenchmarkType.RANDOM_CIRCUIT:
            @qml.qnode(dev)
            def circuit():
                np.random.seed(42)
                for _ in range(n_qubits * 10):
                    qubit = np.random.randint(0, n_qubits)
                    gate_type = np.random.randint(0, 4)

                    if gate_type == 0:
                        qml.Hadamard(wires=qubit)
                    elif gate_type == 1:
                        qml.PauliX(wires=qubit)
                    elif gate_type == 2:
                        qml.RX(np.random.uniform(0, 2 * np.pi), wires=qubit)
                    elif gate_type == 3 and qubit < n_qubits - 1:
                        qml.CNOT(wires=[qubit, qubit + 1])

                return qml.probs(wires=range(n_qubits))

            n_gates = n_qubits * 10

        else:
            raise ValueError(f"Unsupported benchmark type: {benchmark_type}")

        # Run simulation
        _ = circuit()

        return n_gates

    def compute_statistics(self) -> List[BenchmarkStats]:
        """Compute statistical summary of all results."""
        stats_list = []

        # Group results by framework, benchmark type, and n_qubits
        grouped = {}
        for result in self.results:
            key = (result.framework, result.benchmark_type, result.n_qubits)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)

        # Compute stats for each group
        for (framework, benchmark_type, n_qubits), results in grouped.items():
            successful = [r for r in results if r.success]

            if not successful:
                continue

            times = [r.execution_time_ms for r in successful]
            memories = [r.memory_mb for r in successful]

            stats = BenchmarkStats(
                framework=framework,
                benchmark_type=benchmark_type,
                n_qubits=n_qubits,
                mean_time_ms=statistics.mean(times),
                std_time_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
                min_time_ms=min(times),
                max_time_ms=max(times),
                median_time_ms=statistics.median(times),
                mean_memory_mb=statistics.mean(memories),
                success_rate=len(successful) / len(results),
                num_runs=len(results),
            )

            stats_list.append(stats)

        return stats_list

    def export_results(self, output_dir: str = "benchmark_results"):
        """
        Export benchmark results to multiple formats.

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Export raw results to JSON
        json_path = output_path / "results.json"
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"Exported results to {json_path}")

        # Export statistics to JSON
        stats = self.compute_statistics()
        stats_json_path = output_path / "statistics.json"
        with open(stats_json_path, 'w') as f:
            json.dump([s.to_dict() for s in stats], f, indent=2)
        print(f"Exported statistics to {stats_json_path}")

        # Export to CSV
        csv_path = output_path / "results.csv"
        if self.results:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
                writer.writeheader()
                writer.writerows([r.to_dict() for r in self.results])
            print(f"Exported results to {csv_path}")

        # Export statistics to CSV
        stats_csv_path = output_path / "statistics.csv"
        if stats:
            with open(stats_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=stats[0].to_dict().keys())
                writer.writeheader()
                writer.writerows([s.to_dict() for s in stats])
            print(f"Exported statistics to {stats_csv_path}")

        # Generate HTML report
        self._generate_html_report(output_path / "report.html", stats)

    def _generate_html_report(self, output_path: Path, stats: List[BenchmarkStats]):
        """Generate an HTML report with visualizations."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>QuantRS2 Benchmark Results</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .best {{
            background-color: #d4edda;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>QuantRS2 Performance Benchmark Results</h1>
        <p>Generated: {timestamp}</p>

        <h2>Summary Statistics</h2>
        <table>
            <tr>
                <th>Framework</th>
                <th>Benchmark</th>
                <th>Qubits</th>
                <th>Mean Time (ms)</th>
                <th>Std Dev (ms)</th>
                <th>Min (ms)</th>
                <th>Max (ms)</th>
                <th>Success Rate</th>
            </tr>
            {rows}
        </table>
    </div>
</body>
</html>
"""

        # Find best times for highlighting
        benchmark_types = set(s.benchmark_type for s in stats)
        best_times = {}
        for bt in benchmark_types:
            bt_stats = [s for s in stats if s.benchmark_type == bt]
            if bt_stats:
                best_times[bt] = min(s.mean_time_ms for s in bt_stats)

        # Generate table rows
        rows = ""
        for stat in sorted(stats, key=lambda s: (s.benchmark_type, s.n_qubits, s.mean_time_ms)):
            is_best = abs(stat.mean_time_ms - best_times.get(stat.benchmark_type, float('inf'))) < 0.01
            row_class = 'class="best"' if is_best else ''

            rows += f"""
            <tr {row_class}>
                <td>{stat.framework}</td>
                <td>{stat.benchmark_type}</td>
                <td>{stat.n_qubits}</td>
                <td>{stat.mean_time_ms:.2f}</td>
                <td>{stat.std_time_ms:.2f}</td>
                <td>{stat.min_time_ms:.2f}</td>
                <td>{stat.max_time_ms:.2f}</td>
                <td>{stat.success_rate * 100:.1f}%</td>
            </tr>
            """

        html = html.format(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            rows=rows
        )

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"Exported HTML report to {output_path}")

    def print_summary(self):
        """Print a summary of benchmark results."""
        stats = self.compute_statistics()

        if not stats:
            print("No benchmark results available")
            return

        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)

        # Group by benchmark type
        benchmark_types = sorted(set(s.benchmark_type for s in stats))

        for bt in benchmark_types:
            print(f"\n{bt.upper()}:")
            bt_stats = [s for s in stats if s.benchmark_type == bt]

            # Group by n_qubits
            for n_qubits in sorted(set(s.n_qubits for s in bt_stats)):
                print(f"\n  {n_qubits} qubits:")
                qubit_stats = [s for s in bt_stats if s.n_qubits == n_qubits]

                # Sort by mean time
                qubit_stats.sort(key=lambda s: s.mean_time_ms)

                for stat in qubit_stats:
                    print(f"    {stat.framework:15s}: {stat.mean_time_ms:8.2f} ms "
                          f"(±{stat.std_time_ms:.2f} ms)")

                # Show speedup
                if len(qubit_stats) > 1:
                    fastest = qubit_stats[0]
                    for stat in qubit_stats[1:]:
                        speedup = stat.mean_time_ms / fastest.mean_time_ms
                        print(f"    → {stat.framework} is {speedup:.2f}x slower than {fastest.framework}")


def main():
    """Run a comprehensive benchmark suite."""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║       QuantRS2 Performance Benchmarking Suite                         ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize benchmark
    benchmark = PerformanceBenchmark()

    # Run benchmarks
    benchmarks_to_run = [
        (BenchmarkType.BELL_STATE, 2),
        (BenchmarkType.GHZ_STATE, 5),
        (BenchmarkType.GHZ_STATE, 10),
        (BenchmarkType.QFT, 5),
        (BenchmarkType.QFT, 10),
        (BenchmarkType.RANDOM_CIRCUIT, 5),
        (BenchmarkType.RANDOM_CIRCUIT, 10),
    ]

    for benchmark_type, n_qubits in benchmarks_to_run:
        benchmark.run_benchmark(benchmark_type, n_qubits, num_runs=5, warmup_runs=2)

    # Print summary
    benchmark.print_summary()

    # Export results
    benchmark.export_results()

    print("\n" + "="*70)
    print("Benchmark complete! Results saved to ./benchmark_results/")
    print("="*70)


if __name__ == "__main__":
    main()
