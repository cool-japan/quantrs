#!/usr/bin/env python3
"""
Developer Utilities for QuantRS2

This module provides helpful utilities for developers working with QuantRS2,
including debugging helpers, profiling decorators, and development tools.

Features:
    - Performance profiling decorators
    - Circuit analysis and debugging tools
    - Memory profiling utilities
    - Development mode configurations
    - Quick testing helpers
    - Circuit validation utilities
    - Debugging context managers

Usage:
    from quantrs2.dev_utils import profile, analyze_circuit, debug_mode

    @profile
    def my_quantum_function():
        # Your code here
        pass

    # Enable debug mode
    with debug_mode():
        # Detailed logging and checks
        circuit = create_circuit()
"""

import time
import functools
import sys
import traceback
import warnings
from typing import Callable, Any, Optional, Dict, List, TypeVar, cast
from contextlib import contextmanager
from pathlib import Path
import json

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


F = TypeVar('F', bound=Callable[..., Any])


# ============================================================================
# Performance Profiling Decorators
# ============================================================================

def profile(func: F) -> F:
    """
    Decorator to profile function execution time and memory usage.

    Usage:
        @profile
        def my_function():
            # code here
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Memory before
        mem_before = 0
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Time execution
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            # Memory after
            mem_after = 0
            if PSUTIL_AVAILABLE:
                mem_after = process.memory_info().rss / 1024 / 1024  # MB

            # Print profile info
            elapsed_ms = (end_time - start_time) * 1000
            print(f"\n{'='*60}")
            print(f"Profile: {func.__name__}")
            print(f"{'='*60}")
            print(f"Execution time: {elapsed_ms:.2f} ms")
            if PSUTIL_AVAILABLE:
                mem_delta = mem_after - mem_before
                print(f"Memory before: {mem_before:.2f} MB")
                print(f"Memory after: {mem_after:.2f} MB")
                print(f"Memory delta: {mem_delta:+.2f} MB")
            print(f"{'='*60}\n")

            return result
        except Exception as e:
            end_time = time.perf_counter()
            elapsed_ms = (end_time - start_time) * 1000
            print(f"\n{'='*60}")
            print(f"Profile: {func.__name__} (FAILED)")
            print(f"{'='*60}")
            print(f"Execution time: {elapsed_ms:.2f} ms")
            print(f"Error: {type(e).__name__}: {e}")
            print(f"{'='*60}\n")
            raise

    return cast(F, wrapper)


def profile_to_dict(func: F) -> F:
    """
    Decorator to profile function and return results in a dictionary.

    The original function's return value is wrapped in a dict with profiling info.

    Usage:
        @profile_to_dict
        def my_function():
            return result

        result = my_function()
        # result = {
        #     'result': ...,
        #     'execution_time_ms': ...,
        #     'memory_delta_mb': ...
        # }
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        mem_before = 0
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        mem_after = 0
        mem_delta = 0
        if PSUTIL_AVAILABLE:
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_delta = mem_after - mem_before

        return {
            'result': result,
            'execution_time_ms': (end_time - start_time) * 1000,
            'memory_before_mb': mem_before,
            'memory_after_mb': mem_after,
            'memory_delta_mb': mem_delta,
            'function_name': func.__name__,
        }

    return cast(F, wrapper)


def retry(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Decorator to retry a function on failure.

    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        exceptions: Tuple of exceptions to catch

    Usage:
        @retry(max_attempts=3, delay=1.0)
        def unstable_function():
            # code that might fail
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        print(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                        print(f"Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"All {max_attempts} attempts failed")
            raise last_exception  # type: ignore

        return cast(F, wrapper)
    return decorator


# ============================================================================
# Debug Mode
# ============================================================================

_debug_mode_enabled = False


@contextmanager
def debug_mode(verbose: bool = True):
    """
    Context manager to enable debug mode with detailed logging.

    Usage:
        with debug_mode():
            # Detailed logging and validation
            circuit = create_circuit()
    """
    global _debug_mode_enabled
    old_value = _debug_mode_enabled
    _debug_mode_enabled = True

    if verbose:
        print("\n" + "="*60)
        print("DEBUG MODE ENABLED")
        print("="*60 + "\n")

    try:
        yield
    finally:
        _debug_mode_enabled = old_value
        if verbose:
            print("\n" + "="*60)
            print("DEBUG MODE DISABLED")
            print("="*60 + "\n")


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return _debug_mode_enabled


def debug_print(*args: Any, **kwargs: Any) -> None:
    """Print only if debug mode is enabled."""
    if _debug_mode_enabled:
        print("[DEBUG]", *args, **kwargs)


# ============================================================================
# Circuit Analysis Utilities
# ============================================================================

def analyze_circuit(circuit: Any) -> Dict[str, Any]:
    """
    Analyze a quantum circuit and return detailed statistics.

    Args:
        circuit: QuantRS2 circuit to analyze

    Returns:
        Dictionary with circuit statistics
    """
    stats = {
        'n_qubits': 0,
        'n_gates': 0,
        'depth': 0,
        'gate_counts': {},
        'qubit_usage': [],
        'warnings': [],
    }

    try:
        # Try to extract basic info
        if hasattr(circuit, 'num_qubits'):
            stats['n_qubits'] = circuit.num_qubits
        elif hasattr(circuit, 'n_qubits'):
            stats['n_qubits'] = circuit.n_qubits

        if hasattr(circuit, 'num_gates'):
            stats['n_gates'] = circuit.num_gates
        elif hasattr(circuit, 'n_gates'):
            stats['n_gates'] = circuit.n_gates

        if hasattr(circuit, 'depth'):
            stats['depth'] = circuit.depth

        # Try to get gate information
        if hasattr(circuit, 'gates'):
            for gate in circuit.gates:
                gate_name = gate.__class__.__name__ if hasattr(gate, '__class__') else str(gate)
                stats['gate_counts'][gate_name] = stats['gate_counts'].get(gate_name, 0) + 1

        # Warnings for potentially inefficient circuits
        if stats['n_qubits'] > 20:
            stats['warnings'].append(f"Large circuit: {stats['n_qubits']} qubits")
        if stats['n_gates'] > 1000:
            stats['warnings'].append(f"Many gates: {stats['n_gates']} gates")
        if stats['depth'] > 100:
            stats['warnings'].append(f"Deep circuit: {stats['depth']} depth")

    except Exception as e:
        stats['warnings'].append(f"Analysis error: {e}")

    return stats


def print_circuit_analysis(circuit: Any) -> None:
    """Print a formatted circuit analysis."""
    stats = analyze_circuit(circuit)

    print("\n" + "="*60)
    print("CIRCUIT ANALYSIS")
    print("="*60)
    print(f"Qubits: {stats['n_qubits']}")
    print(f"Gates: {stats['n_gates']}")
    print(f"Depth: {stats['depth']}")

    if stats['gate_counts']:
        print("\nGate Distribution:")
        for gate, count in sorted(stats['gate_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {gate}: {count}")

    if stats['warnings']:
        print("\n⚠️  Warnings:")
        for warning in stats['warnings']:
            print(f"  - {warning}")

    print("="*60 + "\n")


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_unitary(matrix: Any, tolerance: float = 1e-10) -> bool:
    """
    Validate that a matrix is unitary.

    Args:
        matrix: Matrix to validate (numpy array or similar)
        tolerance: Numerical tolerance

    Returns:
        True if matrix is unitary, False otherwise
    """
    if not NUMPY_AVAILABLE:
        warnings.warn("NumPy not available, cannot validate unitarity")
        return True

    try:
        # Convert to numpy if needed
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        # Check if square
        if matrix.shape[0] != matrix.shape[1]:
            return False

        # Check U * U† = I
        identity = np.eye(matrix.shape[0])
        product = matrix @ matrix.conj().T
        return np.allclose(product, identity, atol=tolerance)

    except Exception:
        return False


def validate_state_vector(state: Any, tolerance: float = 1e-10) -> bool:
    """
    Validate that a state vector is normalized.

    Args:
        state: State vector to validate
        tolerance: Numerical tolerance

    Returns:
        True if state is normalized, False otherwise
    """
    if not NUMPY_AVAILABLE:
        warnings.warn("NumPy not available, cannot validate state")
        return True

    try:
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        # Check normalization
        norm = np.linalg.norm(state)
        return abs(norm - 1.0) < tolerance

    except Exception:
        return False


# ============================================================================
# Quick Testing Helpers
# ============================================================================

def quick_test_bell_state() -> bool:
    """Quick test of Bell state creation."""
    try:
        from quantrs2 import create_bell_state
        result = create_bell_state()
        return result is not None
    except Exception as e:
        print(f"Bell state test failed: {e}")
        return False


def quick_test_ghz_state(n_qubits: int = 3) -> bool:
    """Quick test of GHZ state creation."""
    try:
        from quantrs2 import create_ghz_state
        result = create_ghz_state(n_qubits)
        return result is not None
    except Exception as e:
        print(f"GHZ state test failed: {e}")
        return False


def run_quick_tests() -> Dict[str, bool]:
    """Run all quick tests and return results."""
    results = {
        'bell_state': quick_test_bell_state(),
        'ghz_state': quick_test_ghz_state(),
    }

    print("\n" + "="*60)
    print("QUICK TEST RESULTS")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    print("="*60 + "\n")

    return results


# ============================================================================
# Development Configuration
# ============================================================================

class DevConfig:
    """Development configuration helper."""

    def __init__(self):
        self.config_file = Path.home() / ".quantrs2" / "dev_config.json"
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load development configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'debug_mode': False,
            'verbose_errors': True,
            'auto_profile': False,
            'strict_validation': False,
        }

    def save(self) -> None:
        """Save configuration to file."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to save dev config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
        self.save()


# Global dev config instance
_dev_config: Optional[DevConfig] = None


def get_dev_config() -> DevConfig:
    """Get the global development configuration."""
    global _dev_config
    if _dev_config is None:
        _dev_config = DevConfig()
    return _dev_config


# ============================================================================
# Error Handling Utilities
# ============================================================================

def format_exception(e: Exception, include_traceback: bool = True) -> str:
    """
    Format an exception with optional traceback.

    Args:
        e: Exception to format
        include_traceback: Whether to include full traceback

    Returns:
        Formatted error string
    """
    error_msg = f"{type(e).__name__}: {e}"

    if include_traceback:
        tb = traceback.format_exc()
        error_msg = f"{error_msg}\n\nTraceback:\n{tb}"

    return error_msg


def safe_execute(func: Callable, *args: Any, **kwargs: Any) -> tuple:
    """
    Safely execute a function and return (result, error).

    Returns:
        Tuple of (result, error) where error is None on success
    """
    try:
        result = func(*args, **kwargs)
        return (result, None)
    except Exception as e:
        return (None, e)


# ============================================================================
# Benchmarking Utilities
# ============================================================================

def compare_implementations(
    implementations: Dict[str, Callable],
    *args: Any,
    n_runs: int = 10,
    **kwargs: Any
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple implementations by timing them.

    Args:
        implementations: Dict mapping names to functions
        *args: Arguments to pass to functions
        n_runs: Number of runs for averaging
        **kwargs: Keyword arguments to pass to functions

    Returns:
        Dictionary with timing statistics for each implementation
    """
    results = {}

    for name, func in implementations.items():
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        results[name] = {
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'n_runs': n_runs,
        }

    return results


def print_comparison(results: Dict[str, Dict[str, float]]) -> None:
    """Print comparison results in a formatted table."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Implementation':<30} {'Mean (ms)':<15} {'Min (ms)':<15} {'Max (ms)':<15}")
    print("-"*80)

    for name, stats in results.items():
        print(f"{name:<30} {stats['mean_ms']:<15.2f} {stats['min_ms']:<15.2f} {stats['max_ms']:<15.2f}")

    print("="*80 + "\n")


# ============================================================================
# Module Info
# ============================================================================

def print_module_info() -> None:
    """Print information about the dev_utils module."""
    print("\n" + "="*60)
    print("QUANTRS2 DEVELOPER UTILITIES")
    print("="*60)
    print("\nAvailable decorators:")
    print("  @profile - Profile execution time and memory")
    print("  @profile_to_dict - Profile and return dict")
    print("  @retry - Retry on failure")
    print("\nContext managers:")
    print("  debug_mode() - Enable debug logging")
    print("\nAnalysis functions:")
    print("  analyze_circuit() - Analyze circuit statistics")
    print("  validate_unitary() - Check matrix unitarity")
    print("  validate_state_vector() - Check normalization")
    print("\nTesting helpers:")
    print("  run_quick_tests() - Run quick sanity tests")
    print("\nUtilities:")
    print("  compare_implementations() - Benchmark functions")
    print("  safe_execute() - Execute with error handling")
    print("="*60 + "\n")


if __name__ == "__main__":
    print_module_info()
