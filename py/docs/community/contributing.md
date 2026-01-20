# Contributing to QuantRS2

Thank you for your interest in contributing to QuantRS2! This guide will help you understand how to contribute effectively to our quantum computing framework.

## 🌟 Ways to Contribute

There are many ways to contribute to QuantRS2, regardless of your experience level:

### For Everyone
- 🐛 **Report bugs** and issues
- 💡 **Suggest features** and improvements  
- 📝 **Improve documentation**
- 🧪 **Test new features**
- 💬 **Help others** in discussions
- 🎓 **Create tutorials** and examples

### For Developers
- 🔧 **Fix bugs** and implement features
- ⚡ **Optimize performance**
- 🧹 **Refactor code**
- ✅ **Write tests**
- 📦 **Add new algorithms**

### For Researchers
- 📊 **Benchmark algorithms**
- 🔬 **Implement new research**
- 📄 **Contribute papers** and references
- 🎯 **Validate implementations**

### For Educators
- 📚 **Create learning materials**
- 🎥 **Record video tutorials**
- 🧩 **Design exercises**
- 🎯 **Improve examples**

## 🚀 Getting Started

### 1. Set Up Development Environment

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/quantrs2.git
cd quantrs2

# Create virtual environment
python -m venv quantrs2-dev
source quantrs2-dev/bin/activate  # On Windows: quantrs2-dev\Scripts\activate

# Install development dependencies
pip install -e .[dev,test,docs]

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

### 2. Choose Your First Contribution

**For beginners:**
- Look for issues labeled `good first issue`
- Fix documentation typos
- Add code comments
- Write simple tests

**For experienced developers:**
- Issues labeled `help wanted`
- Performance optimizations
- New algorithm implementations
- Advanced features

## 📋 Development Workflow

### 1. Find or Create an Issue

Before starting work:
- **Check existing issues** to avoid duplication
- **Create a new issue** for bugs or feature requests
- **Discuss major changes** before implementing

### 2. Create a Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### 3. Make Your Changes

Follow our coding standards:

#### Code Style
- **Python**: Follow PEP 8 with Black formatting
- **Line length**: 88 characters (Black default)
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public functions

#### Example:
```python
def quantum_fourier_transform(
    circuit: Circuit, 
    qubits: List[int]
) -> None:
    """Apply Quantum Fourier Transform to specified qubits.
    
    Args:
        circuit: The quantum circuit to modify.
        qubits: List of qubit indices to apply QFT to.
        
    Raises:
        ValueError: If qubits list is empty or contains invalid indices.
        
    Example:
        >>> circuit = Circuit(4)
        >>> quantum_fourier_transform(circuit, [0, 1, 2])
    """
    if not qubits:
        raise ValueError("Qubits list cannot be empty")
        
    # Implementation here...
```

### 4. Write Tests

All new code must include tests:

```python
import pytest
import quantrs2

def test_quantum_fourier_transform():
    """Test QFT implementation."""
    circuit = quantrs2.Circuit(3)
    
    # Test normal case
    quantum_fourier_transform(circuit, [0, 1, 2])
    assert circuit.gate_count > 0
    
    # Test edge cases
    with pytest.raises(ValueError):
        quantum_fourier_transform(circuit, [])

def test_qft_correctness():
    """Test QFT produces correct results."""
    circuit = quantrs2.Circuit(2)
    circuit.x(0)  # |01⟩ input
    
    quantum_fourier_transform(circuit, [0, 1])
    result = circuit.run()
    
    # Verify expected output probabilities
    probs = result.state_probabilities()
    assert abs(probs['00'] - 0.25) < 1e-10
    # ... more assertions
```

### 5. Run Quality Checks

Before committing:

```bash
# Format code
black quantrs2/ tests/

# Type checking
mypy quantrs2/

# Run tests
pytest tests/ -v

# Check test coverage
pytest tests/ --cov=quantrs2 --cov-report=html

# Lint code
flake8 quantrs2/ tests/

# Run all pre-commit hooks
pre-commit run --all-files
```

### 6. Commit Your Changes

Use conventional commit messages:

```bash
# Feature
git commit -m "feat: add quantum fourier transform implementation"

# Bug fix
git commit -m "fix: resolve measurement issue in multi-qubit circuits"

# Documentation
git commit -m "docs: add examples for gate operations"

# Tests
git commit -m "test: add comprehensive QFT tests"

# Performance
git commit -m "perf: optimize state vector operations"
```

### 7. Submit Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create PR on GitHub with:
# - Clear title and description
# - Link to related issues
# - Screenshots if UI changes
# - Performance benchmarks if relevant
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Verify performance requirements
4. **Property Tests**: Use hypothesis for edge cases

### Test Structure

```python
# tests/test_new_feature.py
import pytest
import numpy as np
from hypothesis import given, strategies as st
import quantrs2

class TestNewFeature:
    """Test suite for new feature."""
    
    def test_basic_functionality(self):
        """Test basic use case."""
        # Arrange
        circuit = quantrs2.Circuit(2)
        
        # Act
        result = new_feature(circuit)
        
        # Assert
        assert result is not None
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        with pytest.raises(ValueError):
            new_feature(None)
    
    @given(st.integers(min_value=1, max_value=10))
    def test_property_based(self, num_qubits):
        """Property-based test with random inputs."""
        circuit = quantrs2.Circuit(num_qubits)
        result = new_feature(circuit)
        
        # Property: result should always be valid
        assert validate_result(result)
    
    def test_performance(self):
        """Test performance requirements."""
        import time
        
        circuit = quantrs2.Circuit(10)
        start_time = time.time()
        
        new_feature(circuit)
        
        execution_time = time.time() - start_time
        assert execution_time < 1.0  # Should complete in under 1 second
```

## 📝 Documentation Guidelines

### Code Documentation

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """Brief description of what the function does.
    
    Longer description if needed. Explain the algorithm, 
    mathematical background, or implementation details.
    
    Args:
        param1: Description of first parameter.
        param2: Description of second parameter. Defaults to "default".
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When param1 is negative.
        TypeError: When param2 is not a string.
        
    Example:
        Basic usage:
        
        >>> result = example_function(5, "test")
        >>> print(result)
        True
        
        Advanced usage:
        
        >>> for i in range(3):
        ...     result = example_function(i)
        ...     print(f"Result {i}: {result}")
        Result 0: True
        Result 1: True
        Result 2: True
        
    Note:
        This function has O(n) time complexity where n is param1.
        
    References:
        [1] Smith, J. (2023). "Quantum Algorithms". Nature Quantum.
    """
```

### Tutorial Documentation

When writing tutorials:

1. **Start with learning objectives**
2. **Provide conceptual explanations**
3. **Include working code examples**
4. **Add exercises and challenges**
5. **Link to related topics**

### API Documentation

- **Use type hints** for all parameters and returns
- **Include examples** for complex functions
- **Document exceptions** that can be raised
- **Add performance notes** where relevant

## 🎯 Contribution Areas

### High Priority Areas

1. **Algorithm Implementations**
   - Quantum machine learning algorithms
   - Optimization algorithms
   - Cryptographic protocols

2. **Performance Optimizations**
   - Circuit compilation improvements
   - Memory usage optimization
   - Parallel processing enhancements

3. **Hardware Integration**
   - New backend connections
   - Error mitigation techniques
   - Calibration and characterization

4. **Developer Experience**
   - Better error messages
   - Improved debugging tools
   - IDE integrations

### Algorithm Implementation Guide

When implementing new algorithms:

```python
# 1. Create algorithm file
# quantrs2/algorithms/new_algorithm.py

from typing import List, Optional, Dict, Any
import numpy as np
from ..core import Circuit, SimulationResult

class NewQuantumAlgorithm:
    """Implementation of New Quantum Algorithm.
    
    This algorithm solves [problem description] with [complexity].
    
    References:
        [1] Author, "Title", Journal, Year
        [2] Implementation based on: https://arxiv.org/abs/xxxx.xxxx
    """
    
    def __init__(self, num_qubits: int, **kwargs):
        """Initialize algorithm.
        
        Args:
            num_qubits: Number of qubits to use.
            **kwargs: Algorithm-specific parameters.
        """
        self.num_qubits = num_qubits
        self.validate_parameters()
    
    def create_circuit(self) -> Circuit:
        """Create the quantum circuit for this algorithm."""
        circuit = Circuit(self.num_qubits)
        
        # Implementation steps
        self._initialization_step(circuit)
        self._main_algorithm_step(circuit) 
        self._measurement_step(circuit)
        
        return circuit
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the algorithm and return results."""
        circuit = self.create_circuit()
        result = circuit.run(**kwargs)
        
        return {
            'raw_result': result,
            'processed_result': self._process_result(result),
            'success_probability': self._calculate_success_probability(result)
        }

# 2. Add comprehensive tests
# tests/test_new_algorithm.py

# 3. Add examples
# examples/new_algorithm_demo.py

# 4. Update documentation
# docs/api/algorithms.md
```

## 🔍 Review Process

### Pull Request Reviews

All PRs go through review:

1. **Automated checks** must pass
   - Tests, linting, type checking
   - Performance benchmarks
   - Documentation builds

2. **Human review** for:
   - Code quality and clarity
   - Algorithm correctness
   - API design
   - Documentation quality

3. **Approval process**:
   - 1 approval for documentation/tests
   - 2 approvals for new features
   - Maintainer approval for breaking changes

### What Reviewers Look For

**Code Quality:**
- Clear, readable code
- Proper error handling
- Comprehensive tests
- Performance considerations

**Algorithm Correctness:**
- Mathematical accuracy
- Proper quantum circuit construction
- Correct measurement interpretation
- Edge case handling

**Documentation:**
- Clear explanations
- Working examples
- Complete API documentation
- Tutorial integration

## 🏆 Recognition

We value all contributions! Contributors are recognized through:

### GitHub Recognition
- Listed in CONTRIBUTORS.md
- GitHub contributor statistics
- Release acknowledgments

### Special Recognition
- "Contributor of the Month" highlights
- Conference presentation opportunities
- Co-authorship on research papers (for significant algorithmic contributions)

### Community Benefits
- Direct access to maintainers
- Early access to new features
- Invitation to contributor meetings
- QuantRS2 swag and stickers

## 📞 Getting Help

### Communication Channels

**For questions about contributing:**
- GitHub Discussions (preferred)
- Discord #contributors channel
- Email: contributors@quantrs2.dev

**For real-time help:**
- Discord #dev-help channel
- Office hours: Fridays 3-4 PM UTC

**For design discussions:**
- GitHub Issues for feature proposals
- RFC process for major changes

### Mentorship Program

New contributors can request mentorship:
- Paired with experienced contributor
- Guided through first contribution
- Regular check-ins and support

Email mentorship@quantrs2.dev to request a mentor.

## 📜 Code of Conduct

### Our Pledge

We are committed to making participation in QuantRS2 a harassment-free experience for everyone, regardless of:
- Age, body size, disability, ethnicity
- Gender identity and expression
- Level of experience, nationality
- Personal appearance, race, religion
- Sexual identity and orientation

### Our Standards

**Positive behavior:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards other members

**Unacceptable behavior:**
- Trolling, insulting/derogatory comments
- Public or private harassment
- Publishing others' private information
- Conduct inappropriate in professional setting

### Enforcement

Report violations to: conduct@quantrs2.dev

Reports are reviewed by maintainers and may result in:
- Warning
- Temporary ban
- Permanent ban from project

## 🎉 Quick Start Checklist

Ready to contribute? Follow this checklist:

- [ ] ⭐ Star the repository
- [ ] 🍴 Fork the repository  
- [ ] 📥 Clone your fork locally
- [ ] 🐍 Set up Python environment
- [ ] 📦 Install development dependencies
- [ ] ✅ Run tests to verify setup
- [ ] 🔍 Find a good first issue
- [ ] 🌿 Create feature branch
- [ ] 💻 Make your changes
- [ ] ✍️ Write tests
- [ ] 📝 Update documentation
- [ ] 🚀 Submit pull request

**Ready to start?** Check out our [good first issues](https://github.com/cool-japan/quantrs/labels/good%20first%20issue)!

## 🔗 Additional Resources

### Development Resources
- [Development Setup Guide](../dev-tools/development-setup.md)
- [Testing Best Practices](../dev-tools/testing-guide.md)
- [Performance Guidelines](../user-guide/performance.md)

### Learning Resources
- [Quantum Computing Basics](../tutorials/beginner/)
- [Algorithm Implementation Examples](../examples/algorithms/)
- [Research Papers and References](../community/references.md)

### Community Resources
- [Discord Community](https://discord.gg/quantrs2)
- [Monthly Contributor Meetings](https://calendar.google.com/quantrs2)
- [Blog and Updates](https://blog.quantrs2.dev)

---

**Thank you for contributing to QuantRS2!** 🚀

Together, we're building the future of quantum computing. Every contribution, no matter how small, helps advance quantum technology and makes it more accessible to everyone.

*"The best way to predict the future is to invent it."* - Alan Kay