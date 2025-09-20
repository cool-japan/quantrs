# QuantRS2 Documentation

This directory contains all documentation for the QuantRS2 quantum computing framework.

## Directory Structure

### `/project`
Core project documentation including changelog, release notes, and Claude AI guidance.

### `/development`
Development guides, TODO lists, enhancement proposals, and roadmaps.

### `/build`
Build instructions for different platforms and release procedures.

### `/integration`
Integration guides for external systems and frameworks.

### `/implementation`
Detailed implementation guides organized by category:
- **algorithms/** - Quantum algorithm decompositions (Clifford+T, Shannon, Cartan, KAK, eigensolvers)
- **paradigms/** - Quantum computing paradigms (MBQC, topological, variational)
- **operations/** - Quantum operations (fermionic, bosonic, batch)
- **error_correction/** - Quantum error correction and channels
- **optimization/** - Circuit optimization techniques (gate optimization, GPU, tensor networks, ZX-calculus)
- **ml/** - Quantum machine learning implementations

### `/templates`
Templates for module documentation (README and TODO templates).

## Quick Links

- [Project Changelog](project/CHANGELOG.md)
- [Development TODO](development/TODO.md)
- [Development Roadmap](development/ROADMAP_SUMMARY.md)
- [macOS Build Guide](MACOS_BUILD_GUIDE.md)
- [Linux Build Guide](LINUX_BUILD_GUIDE.md)
- [Python Release Guide](build/PYTHON_RELEASE.md)