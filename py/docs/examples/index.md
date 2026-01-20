# Example Gallery

Comprehensive collection of quantum computing examples using QuantRS2, from basic quantum circuits to advanced applications.

## 🚀 Quick Start Examples

Perfect for getting started with QuantRS2:

### [Bell State Preparation](basic/bell_state.md)
Create and analyze quantum entanglement with the famous Bell states.
```python
circuit = quantrs2.Circuit(2)
circuit.h(0)
circuit.cx(0, 1)
```
**Topics:** Superposition, Entanglement, Bell states  
**Level:** Beginner  
**Runtime:** < 1 second

### [Quantum Random Number Generator](basic/quantum_random.md)
Generate truly random numbers using quantum mechanics.
```python
circuit = quantrs2.Circuit(4)
for i in range(4):
    circuit.h(i)
circuit.measure_all()
```
**Topics:** Superposition, Measurement, Randomness  
**Level:** Beginner  
**Runtime:** < 1 second

### [Quantum Teleportation](basic/teleportation.md)
Teleport quantum information using entanglement and classical communication.
**Topics:** Entanglement, Measurement, Bell states  
**Level:** Intermediate  
**Runtime:** < 1 second

## 🧮 Quantum Algorithms

Core quantum algorithms that demonstrate quantum advantage:

### [Grover's Search](algorithms/grover.md)
Search unsorted databases with quadratic speedup.
```python
# Search 4-item database
oracle = GroverOracle(target_item=3, num_items=4)
result = grovers_algorithm(oracle)
```
**Speedup:** O(√N) vs O(N)  
**Level:** Intermediate  
**Runtime:** < 5 seconds

### [Quantum Fourier Transform](algorithms/qft.md)
The quantum version of the discrete Fourier transform.
**Topics:** Phase estimation, Shor's algorithm foundation  
**Level:** Intermediate  
**Runtime:** < 2 seconds

### [Deutsch-Jozsa Algorithm](algorithms/deutsch_jozsa.md)
Determine if a function is constant or balanced with a single query.
**Speedup:** 1 query vs N/2 queries (classical)  
**Level:** Beginner  
**Runtime:** < 1 second

### [Shor's Algorithm](algorithms/shor.md)
Factor large integers exponentially faster than classical computers.
**Topics:** Period finding, Modular arithmetic  
**Level:** Advanced  
**Runtime:** 1-10 seconds

### [Quantum Phase Estimation](algorithms/phase_estimation.md)
Estimate eigenvalues of unitary operators.
**Topics:** Eigenvalues, QFT, Quantum simulation  
**Level:** Intermediate  
**Runtime:** < 2 seconds

## 🧠 Quantum Machine Learning

Cutting-edge quantum-enhanced machine learning applications:

### [Variational Quantum Classifier](ml/vqc.md)
Classify data using parameterized quantum circuits.
```python
vqc = VariationalQuantumClassifier(num_qubits=4, num_layers=3)
vqc.fit(X_train, y_train)
predictions = vqc.predict(X_test)
```
**Dataset:** Iris classification  
**Level:** Intermediate  
**Runtime:** 30-60 seconds

### [Quantum Neural Network](ml/qnn.md)
Hybrid quantum-classical neural networks for complex pattern recognition.
**Topics:** Hybrid models, Gradient computation  
**Level:** Advanced  
**Runtime:** 2-5 minutes

### [Quantum Support Vector Machine](ml/qsvm.md)
SVM with quantum kernel for exponential feature spaces.
**Topics:** Quantum kernels, Feature maps  
**Level:** Intermediate  
**Runtime:** 1-2 minutes

### [Quantum Feature Maps](ml/feature_maps.md)
Encode classical data into quantum feature spaces.
**Topics:** Data encoding, Kernel methods  
**Level:** Intermediate  
**Runtime:** < 30 seconds

### [Quantum Reinforcement Learning](ml/qrl.md)
Quantum-enhanced reinforcement learning agents.
**Topics:** Policy gradients, Q-learning  
**Level:** Advanced  
**Runtime:** 5-10 minutes

## 🎯 Optimization Problems

Real-world optimization using quantum algorithms:

### [QAOA for Max-Cut](optimization/qaoa_maxcut.md)
Solve graph partitioning problems using QAOA.
```python
qaoa = QAOA(num_layers=3)
result = qaoa.solve_maxcut(graph)
```
**Topics:** Combinatorial optimization  
**Level:** Intermediate  
**Runtime:** 1-3 minutes

### [Variational Quantum Eigensolver](optimization/vqe.md)
Find ground state energies for quantum chemistry.
**Topics:** Quantum chemistry, Molecular simulation  
**Level:** Advanced  
**Runtime:** 2-5 minutes

### [Portfolio Optimization](optimization/portfolio.md)
Optimize investment portfolios with quantum constraints.
**Topics:** Finance, Risk management  
**Level:** Intermediate  
**Runtime:** 1-2 minutes

### [Traveling Salesman Problem](optimization/tsp.md)
Solve TSP using quantum annealing approaches.
**Topics:** Route optimization, Logistics  
**Level:** Intermediate  
**Runtime:** 2-4 minutes

## 🏭 Industry Applications

Real-world quantum applications across industries:

### [Drug Discovery](applications/drug_discovery.md)
Simulate molecular interactions for pharmaceutical research.
**Industry:** Healthcare  
**Topics:** Molecular simulation, Protein folding  
**Level:** Advanced  
**Runtime:** 5-15 minutes

### [Financial Risk Analysis](applications/finance.md)
Quantum Monte Carlo for derivatives pricing and risk assessment.
**Industry:** Finance  
**Topics:** Monte Carlo, Options pricing  
**Level:** Advanced  
**Runtime:** 3-8 minutes

### [Supply Chain Optimization](applications/supply_chain.md)
Optimize complex logistics networks with quantum algorithms.
**Industry:** Logistics  
**Topics:** Network optimization, Scheduling  
**Level:** Intermediate  
**Runtime:** 2-5 minutes

### [Quantum Chemistry](applications/chemistry.md)
Simulate chemical reactions and material properties.
**Industry:** Materials science  
**Topics:** Molecular dynamics, Catalysis  
**Level:** Advanced  
**Runtime:** 10-30 minutes

### [Cryptography](applications/cryptography.md)
Quantum key distribution and post-quantum cryptography.
**Industry:** Cybersecurity  
**Topics:** Quantum protocols, Security  
**Level:** Advanced  
**Runtime:** 1-3 minutes

## 🔧 Hardware Integration

Examples for running on real quantum hardware:

### [IBM Quantum Integration](hardware/ibm_quantum.md)
Run QuantRS2 circuits on IBM Quantum devices.
**Provider:** IBM  
**Topics:** Cloud quantum computing  
**Level:** Intermediate

### [Google Quantum AI](hardware/google_quantum.md)
Execute circuits on Google's quantum processors.
**Provider:** Google  
**Topics:** Sycamore processor  
**Level:** Intermediate

### [AWS Braket Integration](hardware/aws_braket.md)
Multi-provider quantum computing through AWS.
**Provider:** Amazon  
**Topics:** Rigetti, IonQ, D-Wave  
**Level:** Intermediate

### [Quantum Hardware Benchmarking](hardware/benchmarking.md)
Compare performance across different quantum devices.
**Topics:** Performance analysis, Noise characterization  
**Level:** Advanced

## 🛡️ Error Mitigation

Improve results on noisy quantum devices:

### [Zero-Noise Extrapolation](mitigation/zero_noise_extrapolation.md)
Extrapolate to zero-noise limit for better results.
**Topics:** NISQ algorithms, Error correction  
**Level:** Intermediate  
**Runtime:** 2-5 minutes

### [Readout Error Mitigation](mitigation/readout_correction.md)
Correct measurement errors using calibration.
**Topics:** Measurement errors, Calibration  
**Level:** Beginner  
**Runtime:** < 1 minute

### [Dynamical Decoupling](mitigation/dynamical_decoupling.md)
Protect qubits from decoherence during idle periods.
**Topics:** Decoherence, Pulse sequences  
**Level:** Intermediate  
**Runtime:** 1-2 minutes

### [Symmetry Verification](mitigation/symmetry_verification.md)
Use symmetries to detect and correct errors.
**Topics:** Error detection, Post-processing  
**Level:** Intermediate  
**Runtime:** < 1 minute

## 🎨 Visualization and Analysis

Tools for understanding quantum circuits and results:

### [Circuit Visualization](visualization/circuit_plots.md)
Create beautiful quantum circuit diagrams.
**Topics:** Circuit diagrams, Publishing  
**Level:** Beginner

### [State Vector Analysis](visualization/state_analysis.md)
Visualize and analyze quantum states.
**Topics:** Bloch spheres, State tomography  
**Level:** Intermediate

### [Performance Profiling](visualization/profiling.md)
Analyze circuit performance and optimization opportunities.
**Topics:** Optimization, Debugging  
**Level:** Intermediate

### [Interactive Quantum Circuits](visualization/interactive.md)
Build interactive quantum circuit explorers.
**Topics:** Education, Jupyter widgets  
**Level:** Beginner

## 🔬 Research and Advanced Topics

Cutting-edge quantum computing research implementations:

### [Quantum Error Correction](research/error_correction.md)
Implement basic quantum error correction codes.
**Topics:** Surface codes, Stabilizer codes  
**Level:** Expert

### [Topological Quantum Computing](research/topological.md)
Explore topological approaches to quantum computation.
**Topics:** Anyons, Braiding  
**Level:** Expert

### [Quantum Simulation](research/quantum_simulation.md)
Simulate many-body quantum systems.
**Topics:** Hamiltonian simulation, Time evolution  
**Level:** Advanced

### [Hybrid Quantum-Classical Algorithms](research/hybrid_algorithms.md)
Advanced hybrid optimization techniques.
**Topics:** VQAs, NISQ algorithms  
**Level:** Advanced

## 📚 Educational Examples

Perfect for teaching and learning quantum computing:

### [Quantum Computing 101](education/quantum_101.md)
Introduction to quantum computing concepts.
**Audience:** Beginners  
**Topics:** Qubits, Gates, Measurement

### [University Course Examples](education/university_examples.md)
Examples designed for university quantum computing courses.
**Audience:** Students  
**Topics:** Problem sets, Assignments

### [Workshop Materials](education/workshop_materials.md)
Hands-on exercises for quantum computing workshops.
**Audience:** Workshop participants  
**Topics:** Interactive exercises, Group activities

### [Quantum Coding Challenges](education/coding_challenges.md)
Programming challenges to test quantum coding skills.
**Audience:** Developers  
**Topics:** Algorithm implementation, Optimization

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- QuantRS2 installed (`pip install quantrs2`)
- Basic quantum computing knowledge (recommended)

### Running Examples

1. **Clone or download** the example files
2. **Install dependencies** for specific examples:
   ```bash
   pip install quantrs2[examples]  # All example dependencies
   ```
3. **Run examples** in your preferred environment:
   ```bash
   python examples/basic/bell_state.py
   ```

### Example Structure

Each example includes:
- 📝 **Explanation**: Background and theory
- 💻 **Code**: Complete, runnable implementation
- 📊 **Results**: Expected outputs and analysis
- 🎯 **Exercises**: Hands-on extensions and modifications
- 📚 **References**: Further reading and research papers

### Difficulty Levels

- **🟢 Beginner**: Basic quantum concepts, simple circuits
- **🟡 Intermediate**: Quantum algorithms, optimization
- **🔴 Advanced**: Complex applications, research topics
- **⚫ Expert**: Cutting-edge research, experimental features

### Performance Notes

Runtime estimates are for:
- **CPU**: Modern multi-core processor
- **Memory**: 8GB RAM
- **Device**: Classical simulation

Hardware examples require access to quantum cloud services.

## 🤝 Contributing Examples

We welcome contributions! See our [contribution guide](../community/contributing.md) for:
- Adding new examples
- Improving existing examples
- Fixing bugs and issues
- Documentation improvements

### Example Template

Use our [example template](template.md) to create new examples:
```python
"""
Example: [Name]
Description: [Brief description]
Level: [Beginner/Intermediate/Advanced/Expert]
Runtime: [Expected runtime]
Topics: [Relevant topics]
"""
import quantrs2

# Your example code here...
```

## 📖 Additional Resources

### Documentation
- [API Reference](../api/) - Complete API documentation
- [Tutorials](../tutorials/) - Step-by-step learning guides
- [User Guide](../user-guide/) - Best practices and tips

### Community
- [GitHub Discussions](https://github.com/cool-japan/quantrs/discussions) - Ask questions and share ideas
- [Discord Community](https://discord.gg/quantrs2) - Real-time chat and support
- [Stack Overflow](https://stackoverflow.com/questions/tagged/quantrs2) - Technical Q&A

### Learning Resources
- [Quantum Computing Textbooks](../community/references.md) - Recommended reading
- [Online Courses](../community/courses.md) - University and industry courses
- [Research Papers](../community/papers.md) - Latest quantum computing research

---

**Ready to explore quantum computing?** Start with the [Bell State example](basic/bell_state.md) or dive into [Grover's Search](algorithms/grover.md)!

*The future of computing is quantum. Start building it today with QuantRS2.* 🚀