# Framework Integration Examples

This directory contains comprehensive demonstration scripts for QuantRS2's framework integration capabilities.

## 📚 Contents

- **`FRAMEWORK_INTEGRATION_GUIDE.md`** - Complete integration guide
- **`qiskit_converter_demo.py`** - Qiskit integration examples
- **`cirq_converter_demo.py`** - Cirq integration examples
- **`myqlm_converter_demo.py`** - MyQLM/QLM integration examples
- **`projectq_converter_demo.py`** - ProjectQ integration examples

## 🚀 Quick Start

### Prerequisites

```bash
# Install QuantRS2
pip install quantrs2

# Install framework dependencies (optional)
pip install qiskit        # For Qiskit examples
pip install cirq          # For Cirq examples
pip install myqlm         # For MyQLM examples
pip install projectq      # For ProjectQ examples
```

### Running Examples

```bash
# Run all Qiskit examples
python qiskit_converter_demo.py

# Run all Cirq examples
python cirq_converter_demo.py

# Run all MyQLM examples
python myqlm_converter_demo.py

# Run all ProjectQ examples
python projectq_converter_demo.py
```

## 📖 What You'll Learn

### Qiskit Converter (`qiskit_converter_demo.py`)

- ✅ Basic circuit conversion
- ✅ Advanced gate support (iSwap, ECR, RXX, RYY, RZZ)
- ✅ QFT algorithm conversion
- ✅ Variational circuits
- ✅ QASM import/export
- ✅ Multi-controlled gates
- ✅ Circuit optimization
- ✅ Error handling

### Cirq Converter (`cirq_converter_demo.py`)

- ✅ Basic circuit conversion
- ✅ Power gate decomposition
- ✅ Advanced gates (iSwap, FSim, PhasedX, Givens)
- ✅ GridQubit handling
- ✅ Moment preservation
- ✅ Rotation gates
- ✅ Three-qubit gates
- ✅ Grover's algorithm

### MyQLM Converter (`myqlm_converter_demo.py`)

- ✅ Basic circuit conversion
- ✅ Rotation gates
- ✅ Two-qubit gates
- ✅ Three-qubit gates
- ✅ GHZ state preparation
- ✅ Variational circuits
- ✅ Job creation
- ✅ Error handling

### ProjectQ Converter (`projectq_converter_demo.py`)

- ✅ Basic circuit conversion
- ✅ Rotation gates
- ✅ Controlled gates
- ✅ Three-qubit gates
- ✅ GHZ state preparation
- ✅ Quantum Fourier Transform
- ✅ Variational circuits
- ✅ Backend adapter usage

## 🎯 Features Demonstrated

### Universal Features (All Converters)

- **Bidirectional Conversion**: Import and export between frameworks
- **Automatic Decomposition**: Complex gates decomposed to basic operations
- **Error Handling**: Strict and lenient modes
- **Conversion Statistics**: Detailed metrics for every conversion
- **Production Quality**: Robust error handling and validation

### Framework-Specific Features

| Feature | Qiskit | Cirq | MyQLM | ProjectQ |
|---------|--------|------|-------|----------|
| Power Gates | ✅ | ✅ | ❌ | ❌ |
| QASM Support | ✅ | ❌ | ❌ | ❌ |
| Moments | ❌ | ✅ | ❌ | ❌ |
| Job Creation | ❌ | ❌ | ✅ | ❌ |
| Backend Adapter | ❌ | ❌ | ❌ | ✅ |
| Equivalence Test | ✅ | ✅ | ❌ | ❌ |

## 🔧 Advanced Usage

### Custom Conversion

```python
from quantrs2.qiskit_converter import QiskitConverter

# Create converter with strict mode
converter = QiskitConverter(strict_mode=True)

# Convert with optimization
circuit, stats = converter.from_qiskit(qiskit_circuit, optimize=True)

# Check conversion statistics
print(f"Converted {stats.converted_gates} gates")
print(f"Decomposed {stats.decomposed_gates} gates")
```

### Error Handling

```python
from quantrs2.cirq_converter import CirqConverter

converter = CirqConverter(strict_mode=False)  # Lenient mode
circuit, stats = converter.from_cirq(cirq_circuit)

if not stats.success:
    print("Conversion warnings:")
    for warning in stats.warnings:
        print(f"  - {warning}")
```

### Backend Integration

```python
from quantrs2.projectq_converter import ProjectQBackend
from projectq import MainEngine

# Use QuantRS2 as ProjectQ backend
backend = ProjectQBackend()
eng = MainEngine(backend=backend)

# Build circuit
# ...

eng.flush()

# Access converted QuantRS2 circuit
quantrs_circuit = backend._circuit
```

## 📊 Example Output

When you run the demonstrations, you'll see detailed output like:

```
======================================================================
  Basic Circuit Conversion
======================================================================

📋 Original Qiskit Circuit:
        ┌───┐
   q_0: ┤ H ├──■──
        └───┘┌─┴─┐
   q_1: ─────┤ X ├
             └───┘
   c: 2/══════════

📊 Conversion Statistics:
   ✓ Original gates: 3
   ✓ Converted gates: 2
   ✓ Decomposed gates: 0
   ✓ Success: True

✅ Bell state circuit converted successfully!
```

## 🎓 Learning Path

### Beginner
1. Start with **basic conversion** examples
2. Understand **conversion statistics**
3. Learn **error handling**

### Intermediate
4. Explore **advanced gates**
5. Try **optimization**
6. Experiment with **variational circuits**

### Advanced
7. Use **backend adapters**
8. Implement **custom workflows**
9. Contribute **new features**

## 🐛 Troubleshooting

### Import Errors

If you see framework import errors:

```bash
# Install missing frameworks
pip install qiskit cirq myqlm projectq
```

### Conversion Failures

Check the conversion statistics for details:

```python
circuit, stats = converter.from_qiskit(qc)

if not stats.success:
    print(f"Unsupported gates: {stats.unsupported_gates}")
    print(f"Warnings: {stats.warnings}")
```

### Performance Issues

For large circuits, enable optimization:

```python
circuit, stats = converter.from_qiskit(qc, optimize=True)
```

## 📚 Additional Resources

- **Main Documentation**: See `FRAMEWORK_INTEGRATION_GUIDE.md`
- **API Reference**: Check module docstrings
- **GitHub**: https://github.com/cool-japan/quantrs
- **Issues**: Report bugs on GitHub Issues

## 🤝 Contributing

Found a bug or want to add examples? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Add your examples
4. Submit a pull request

## 📝 License

MIT/Apache-2.0 (same as QuantRS2)

---

**Version**: 0.1.2
**Last Updated**: 2026-01-23
**Status**: Production Ready ✅
