# IBM Quantum Integration

**Complete guide to integrating QuantRS2 with IBM's quantum computing systems**

IBM Quantum provides access to a wide range of quantum processors and simulators through IBM Cloud. QuantRS2 offers seamless integration with IBM's quantum systems, automatic transpilation to IBM's native gate set, and optimization for IBM's hardware topology.

## 🎯 Overview

### IBM Quantum Systems Supported

| System | Qubits | Topology | Gate Set | Typical Queue |
|--------|--------|----------|----------|---------------|
| **ibmq_qasm_simulator** | 32 | Configurable | Universal | < 1 min |
| **ibmq_manila** | 5 | Line | RZ, SX, X, CNOT | 5-15 min |
| **ibmq_lima** | 5 | Line | RZ, SX, X, CNOT | 3-10 min |
| **ibmq_belem** | 5 | Line | RZ, SX, X, CNOT | 5-20 min |
| **ibmq_quito** | 5 | Line | RZ, SX, X, CNOT | 10-30 min |
| **ibmq_montreal** | 27 | Heavy-hex | RZ, SX, X, CNOT | 15-60 min |
| **ibmq_mumbai** | 27 | Heavy-hex | RZ, SX, X, CNOT | 10-45 min |
| **ibmq_kolkata** | 27 | Heavy-hex | RZ, SX, X, CNOT | 20-90 min |
| **ibm_perth** | 7 | Heavy-hex | RZ, SX, X, CNOT | 5-25 min |
| **ibm_lagos** | 7 | Heavy-hex | RZ, SX, X, CNOT | 8-30 min |

## 🚀 Quick Start

### 1. Installation and Setup

```bash
# Install QuantRS2 with IBM support
pip install quantrs2[ibm]

# Or install separately
pip install quantrs2 qiskit-ibm-provider
```

### 2. Authentication

**Option A: Environment Variables**
```bash
export IBMQ_TOKEN="your_ibm_token_here"
export IBMQ_HUB="ibm-q"
export IBMQ_GROUP="open"
export IBMQ_PROJECT="main"
```

**Option B: Configuration File**
```python
import quantrs2

# One-time setup
quantrs2.hardware.configure_ibm(
    token="your_ibm_token_here",
    hub="ibm-q",
    group="open",
    project="main",
    save_credentials=True  # Saves for future use
)
```

**Option C: Runtime Configuration**
```python
import quantrs2

# Configure for this session only
quantrs2.hardware.ibm_auth(
    token="your_ibm_token_here",
    hub="ibm-q-academic",  # If you have academic access
    group="your-group",
    project="your-project"
)
```

### 3. Your First IBM Quantum Circuit

```python
import quantrs2
import numpy as np

# Create a Bell state circuit
circuit = quantrs2.Circuit(2)
circuit.h(0)
circuit.cx(0, 1)

# Run on IBM simulator
result = circuit.run(
    backend="ibmq_qasm_simulator",
    shots=1024
)

print(f"Bell state results: {result.counts}")
print(f"Execution time: {result.execution_time}")
```

## 🔧 Advanced Configuration

### Backend Selection and Management

```python
# List available IBM backends
available = quantrs2.hardware.ibm.list_backends(
    status="online",
    min_qubits=5,
    max_queue_time=30  # minutes
)

for backend in available:
    print(f"{backend.name}: {backend.num_qubits} qubits, "
          f"queue: {backend.queue_length} jobs")

# Get detailed backend information
backend_info = quantrs2.hardware.ibm.get_backend_info("ibmq_montreal")
print(f"Gate error rates: {backend_info.gate_errors}")
print(f"Coherence times: {backend_info.coherence_times}")
print(f"Connectivity: {backend_info.connectivity_map}")
```

### Automatic Backend Selection

```python
# Let QuantRS2 choose the best backend for your circuit
optimal_backend = quantrs2.hardware.ibm.select_optimal_backend(
    circuit=my_circuit,
    criteria=["queue_time", "fidelity", "connectivity"],
    max_queue_time=60,  # minutes
    min_fidelity=0.95
)

result = my_circuit.run(backend=optimal_backend, shots=1024)
```

## 🎛️ Circuit Optimization for IBM Hardware

### Automatic Transpilation

```python
# QuantRS2 automatically optimizes for IBM's native gate set
circuit = quantrs2.Circuit(3)
circuit.ry(0, np.pi/4)  # Becomes RZ-SX-RZ sequence
circuit.rx(1, np.pi/3)  # Optimized decomposition
circuit.rz(2, np.pi/2)  # Native gate, no decomposition
circuit.ccx(0, 1, 2)    # Efficient Toffoli decomposition

# Run with automatic optimization
result = circuit.run(
    backend="ibmq_montreal",
    optimization_level=3,  # Maximum optimization
    shots=1024
)
```

### Manual Optimization Control

```python
# Fine-grained control over transpilation
optimized_circuit = quantrs2.hardware.ibm.optimize_circuit(
    circuit=my_circuit,
    backend="ibmq_montreal",
    optimization_level=2,
    layout_method="sabre",           # Qubit mapping algorithm
    routing_method="stochastic",     # Routing algorithm  
    translation_method="unroller",   # Gate decomposition
    scheduling_method="alap",        # Instruction scheduling
    seed_transpiler=42               # Reproducible results
)

# Inspect optimization results
print(f"Original gates: {my_circuit.gate_count()}")
print(f"Optimized gates: {optimized_circuit.gate_count()}")
print(f"Circuit depth: {optimized_circuit.depth()}")
print(f"Qubit mapping: {optimized_circuit.qubit_mapping}")
```

### Hardware-Aware Circuit Design

```python
# Design circuits that work well on IBM topology
def create_hardware_efficient_ansatz(num_qubits, layers):
    """Create ansatz optimized for IBM heavy-hex topology."""
    circuit = quantrs2.Circuit(num_qubits)
    
    for layer in range(layers):
        # Single-qubit rotations (native on IBM)
        for qubit in range(num_qubits):
            circuit.ry(qubit, f"theta_{layer}_{qubit}")
        
        # Two-qubit gates following IBM connectivity
        connectivity = quantrs2.hardware.ibm.get_coupling_map("ibmq_montreal")
        for qubit1, qubit2 in connectivity:
            if qubit1 < num_qubits and qubit2 < num_qubits:
                circuit.cx(qubit1, qubit2)
    
    return circuit

# Use the hardware-efficient ansatz
ansatz = create_hardware_efficient_ansatz(5, 3)
```

## 📊 Error Mitigation

### Zero-Noise Extrapolation

```python
# Built-in error mitigation for IBM systems
circuit = quantrs2.Circuit(2)
circuit.h(0).cx(0, 1)

# Run with zero-noise extrapolation
result = circuit.run(
    backend="ibmq_montreal",
    shots=8192,  # More shots for better statistics
    error_mitigation="zero_noise_extrapolation",
    extrapolation_factors=[1, 2, 3],  # Noise scaling factors
    fitter="exponential"  # Extrapolation method
)

print(f"Raw results: {result.raw_counts}")
print(f"Mitigated results: {result.mitigated_counts}")
print(f"Error mitigation overhead: {result.mitigation_overhead}")
```

### Readout Error Mitigation

```python
# Correct for measurement errors
result = circuit.run(
    backend="ibmq_mumbai",
    shots=1024,
    readout_mitigation=True,
    calibration_shots=1024  # Shots for calibration matrix
)

print(f"Calibration matrix: {result.calibration_matrix}")
print(f"Corrected counts: {result.corrected_counts}")
```

### Custom Error Mitigation

```python
# Implement custom error mitigation
from quantrs2.mitigation import CustomMitigator

class IBMCustomMitigator(CustomMitigator):
    def __init__(self, backend_name):
        self.backend = quantrs2.hardware.ibm.get_backend(backend_name)
        self.noise_model = self._characterize_noise()
    
    def _characterize_noise(self):
        # Implement noise characterization
        pass
    
    def mitigate(self, counts):
        # Implement mitigation algorithm
        return mitigated_counts

# Use custom mitigation
mitigator = IBMCustomMitigator("ibmq_montreal")
result = circuit.run(
    backend="ibmq_montreal",
    shots=1024,
    error_mitigator=mitigator
)
```

## ⚡ Performance Optimization

### Batch Job Submission

```python
# Submit multiple circuits efficiently
circuits = [create_vqe_circuit(params) for params in parameter_sets]

# Batch submission reduces overhead
batch_job = quantrs2.hardware.ibm.submit_batch(
    circuits=circuits,
    backend="ibmq_montreal",
    shots=1024,
    max_parallel=5  # Maximum concurrent jobs
)

# Monitor progress
for i, job in enumerate(batch_job.jobs):
    print(f"Circuit {i}: {job.status()}")

# Collect results when ready
results = batch_job.results()
```

### Queue Management

```python
# Intelligent queue management
job = circuit.run_async(
    backend="ibmq_montreal",
    shots=1024,
    priority="normal",
    max_wait_time="2 hours",
    fallback_backends=["ibmq_mumbai", "ibmq_kolkata"]
)

# Check status
print(f"Job status: {job.status()}")
print(f"Queue position: {job.queue_position()}")
print(f"Estimated wait: {job.estimated_wait_time()}")

# Get result when ready (blocks until complete)
result = job.result()
```

### Circuit Caching

```python
# Cache compiled circuits for reuse
quantrs2.config.enable_circuit_caching(
    cache_dir="/tmp/quantrs2_cache",
    max_cache_size="1GB",
    cache_transpiled=True
)

# First run compiles and caches
result1 = circuit.run(backend="ibmq_montreal", shots=1024)

# Subsequent runs use cached compilation
result2 = circuit.run(backend="ibmq_montreal", shots=2048)  # Much faster
```

## 🔍 Monitoring and Analysis

### Job Management Dashboard

```python
# Get comprehensive job information
jobs = quantrs2.hardware.ibm.get_jobs(
    backend="ibmq_montreal",
    limit=50,
    status=["RUNNING", "QUEUED", "COMPLETED"]
)

for job in jobs:
    print(f"Job {job.id()}: {job.status()}")
    print(f"  Circuit: {job.circuit_name()}")
    print(f"  Shots: {job.shots()}")
    print(f"  Created: {job.creation_date()}")
    print(f"  Queue time: {job.queue_time()}")
```

### Performance Analytics

```python
# Analyze circuit performance on different backends
performance = quantrs2.hardware.ibm.analyze_performance(
    circuit=my_circuit,
    backends=["ibmq_montreal", "ibmq_mumbai", "ibmq_kolkata"],
    metrics=["fidelity", "execution_time", "queue_time"]
)

print(f"Best backend: {performance.best_backend}")
print(f"Performance scores: {performance.scores}")

# Visualize performance comparison
performance.plot()
```

### Noise Analysis

```python
# Characterize backend noise properties
noise_profile = quantrs2.hardware.ibm.characterize_noise(
    backend="ibmq_montreal",
    characterization_depth="full"
)

print(f"Gate error rates: {noise_profile.gate_errors}")
print(f"Readout errors: {noise_profile.readout_errors}")
print(f"Coherence times: {noise_profile.coherence_times}")
print(f"Cross-talk matrix: {noise_profile.crosstalk}")

# Use noise profile for simulation
noisy_result = circuit.run(
    backend="qasm_simulator",
    noise_model=noise_profile.to_qiskit_noise_model(),
    shots=1024
)
```

## 🔒 Security and Best Practices

### Secure Credential Management

```python
# Use secure credential storage
from quantrs2.security import SecureCredentialStore

# Store credentials securely
store = SecureCredentialStore()
store.set_ibm_credentials(
    token="your_token",
    hub="ibm-q",
    group="open", 
    project="main",
    encrypt=True,
    keyring_service="quantrs2_ibm"
)

# Load credentials securely
credentials = store.get_ibm_credentials()
quantrs2.hardware.configure_ibm(**credentials)
```

### Circuit Privacy

```python
# Enhanced privacy for sensitive circuits
result = circuit.run(
    backend="ibmq_montreal",
    shots=1024,
    privacy_mode=True,           # Encrypt circuit before submission
    anonymous_execution=True,    # Remove identifying metadata
    local_preprocessing=True     # Pre-process locally when possible
)
```

### Compliance Features

```python
# Enable compliance logging
quantrs2.compliance.enable_logging(
    log_level="INFO",
    audit_trail=True,
    data_locality="US",  # Ensure data stays in US
    retention_period="7 years"
)

# Check compliance status
compliance_report = quantrs2.compliance.generate_report(
    timeframe="last_30_days",
    include_circuits=True,
    include_results=False  # Exclude sensitive data
)
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

**Authentication Errors**
```python
# Verify credentials
try:
    quantrs2.hardware.ibm.test_connection()
    print("✅ IBM connection successful")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    # Check token, hub, group, project values
```

**Queue Time Issues**
```python
# Find backends with shorter queues
fast_backends = quantrs2.hardware.ibm.list_backends(
    max_queue_time=15,  # minutes
    min_qubits=circuit.num_qubits,
    status="online"
)

if fast_backends:
    backend = fast_backends[0]
    print(f"Using fast backend: {backend.name}")
else:
    print("No fast backends available, using fallback")
    backend = "ibmq_qasm_simulator"
```

**Circuit Compilation Errors**
```python
# Debug transpilation issues
try:
    optimized = circuit.compile_for("ibmq_montreal")
except quantrs2.TranspilationError as e:
    print(f"Transpilation failed: {e}")
    
    # Try with different optimization levels
    for level in [0, 1, 2]:
        try:
            optimized = circuit.compile_for(
                "ibmq_montreal", 
                optimization_level=level
            )
            print(f"✅ Success with optimization level {level}")
            break
        except:
            continue
```

### Getting Help

- **IBM Quantum Documentation**: [qiskit.org](https://qiskit.org/documentation/)
- **QuantRS2 Discord**: `#ibm-integration` channel
- **GitHub Issues**: Report IBM-specific bugs
- **Office Hours**: Weekly IBM integration Q&A

## 📚 Example Applications

### Variational Quantum Eigensolver (VQE)

```python
import quantrs2
import numpy as np
from scipy.optimize import minimize

# Define H2 molecule Hamiltonian
hamiltonian = quantrs2.chemistry.Hamiltonian.h2_molecule(
    bond_length=0.74,  # Angstroms
    basis_set="sto-3g"
)

# Create parameterized ansatz
def ansatz(params):
    circuit = quantrs2.Circuit(2)
    circuit.ry(0, params[0])
    circuit.ry(1, params[1])
    circuit.cx(0, 1)
    return circuit

# Cost function for VQE
def cost_function(params):
    circuit = ansatz(params)
    expectation = hamiltonian.expectation_value(
        circuit,
        backend="ibmq_montreal",
        shots=4096,
        error_mitigation="zero_noise_extrapolation"
    )
    return expectation

# Optimize parameters
initial_params = [0.1, 0.1]
result = minimize(
    cost_function,
    initial_params,
    method="COBYLA",
    options={"maxiter": 100}
)

print(f"Ground state energy: {result.fun} Hartree")
print(f"Optimal parameters: {result.x}")
```

### Quantum Approximate Optimization Algorithm (QAOA)

```python
# Max-Cut problem with QAOA
graph = quantrs2.optimization.Graph([(0,1), (1,2), (2,3), (3,0)])

def qaoa_circuit(gamma, beta):
    circuit = quantrs2.Circuit(4)
    
    # Initial superposition
    for i in range(4):
        circuit.h(i)
    
    # Problem Hamiltonian
    for edge in graph.edges:
        circuit.zz(edge[0], edge[1], gamma)
    
    # Mixer Hamiltonian  
    for i in range(4):
        circuit.rx(i, beta)
    
    return circuit

# Optimization loop
def qaoa_cost(params):
    gamma, beta = params
    circuit = qaoa_circuit(gamma, beta)
    
    # Measure in computational basis
    counts = circuit.run(
        backend="ibmq_lima",
        shots=2048
    ).counts
    
    # Calculate objective function
    return graph.evaluate_cut(counts)

# Optimize QAOA parameters
optimal_params = minimize(
    qaoa_cost,
    [0.5, 0.5],
    method="NELDER-MEAD"
)

print(f"Best cut value: {-optimal_params.fun}")
print(f"Optimal parameters: {optimal_params.x}")
```

## 🏆 Best Practices Summary

### Circuit Design
- Use native IBM gates (RZ, SX, X, CNOT) when possible
- Design circuits considering IBM's heavy-hex topology
- Keep circuit depth minimal for better fidelity
- Use hardware-efficient ansätze for variational algorithms

### Error Mitigation
- Always use error mitigation for research applications
- Increase shot count for better statistics
- Consider readout error correction for measurement-heavy circuits
- Validate results with noiseless simulation

### Performance
- Use batch submission for multiple circuits
- Cache compiled circuits for repeated use
- Choose backends based on queue time and fidelity
- Monitor job status and use fallback backends

### Security
- Store credentials securely using system keyring
- Enable privacy mode for sensitive circuits
- Use compliance logging for regulated applications
- Regularly rotate access tokens

---

**Ready to run on IBM Quantum?** [Get your IBM Quantum account](https://quantum-computing.ibm.com/) and start executing quantum circuits on IBM's quantum processors!

*Last updated: December 2024 | IBM Quantum integration version: 1.0.2*