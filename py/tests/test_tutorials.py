"""Integration tests for quantrs2 tutorials.

Each tutorial is run as a subprocess so it exercises the full import path
and module execution.  Tutorials must:
  - Exit with code 0 (including when optional deps are missing)
  - Produce non-empty stdout

Heavy tutorials (QEC and QUBO) are marked ``@pytest.mark.slow`` and are
excluded from the default parametrize to keep CI fast.
"""

from __future__ import annotations

import subprocess
import sys
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Tutorial names (module stem only)
# ---------------------------------------------------------------------------

_ALL_TUTORIALS: List[str] = [
    "01_bell_state",
    "02_vqe_h2",
    "03_qaoa_maxcut",
    "04_qec_surface_code",
    "05_qubo_sampling",
    "06_3d_state_visualization",
    "07_parameterized_circuits",
    "08_error_mitigation",
]

# Tutorials that are potentially slow (many Monte Carlo shots, etc.)
_SLOW_TUTORIALS: List[str] = ["04_qec_surface_code", "05_qubo_sampling"]

_FAST_TUTORIALS: List[str] = [t for t in _ALL_TUTORIALS if t not in _SLOW_TUTORIALS]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_tutorial(tutorial: str, timeout: int) -> subprocess.CompletedProcess:
    """Execute a tutorial module and return the completed process."""
    return subprocess.run(
        [sys.executable, "-m", f"quantrs2.tutorials.{tutorial}"],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd="/Users/kitasan/work/quantrs/py/python",
    )


# ---------------------------------------------------------------------------
# Fast tutorials (run in normal CI)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tutorial", _FAST_TUTORIALS)
def test_tutorial_runs(tutorial: str) -> None:
    """Each fast tutorial should exit 0 and produce non-empty stdout."""
    result = _run_tutorial(tutorial, timeout=60)
    assert result.returncode == 0, (
        f"Tutorial {tutorial} failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    assert result.stdout.strip(), (
        f"Tutorial {tutorial} produced no output.\n"
        f"stderr:\n{result.stderr[-500:]}"
    )


# ---------------------------------------------------------------------------
# Slow tutorials (gated behind @pytest.mark.slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("tutorial", _SLOW_TUTORIALS)
def test_tutorial_heavy(tutorial: str) -> None:
    """Heavy tutorials may take up to 120s but must still exit 0."""
    result = _run_tutorial(tutorial, timeout=120)
    assert result.returncode == 0, (
        f"Tutorial {tutorial} failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout[-2000:]}\n"
        f"stderr:\n{result.stderr[-2000:]}"
    )
    assert result.stdout.strip(), (
        f"Tutorial {tutorial} produced no output.\n"
        f"stderr:\n{result.stderr[-500:]}"
    )
