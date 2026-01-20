"""
QuantRS2 - Quantum Computing Framework for Python

A comprehensive Python interface to the QuantRS2 quantum computing framework,
built on high-performance Rust implementations.

Modules:
    core: Core quantum computing primitives, gates, and algorithms
"""

__version__ = "0.1.0b2"
__author__ = "COOLJAPAN OU (Team KitaSan)"
__license__ = "MIT OR Apache-2.0"

# The core module is the compiled Rust extension
# It will be available as quantrs2.core after installation
try:
    from . import core
    __all__ = ['core']
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import quantrs2.core extension module: {e}. "
        "Make sure the package is properly installed.",
        ImportWarning
    )
    __all__ = []