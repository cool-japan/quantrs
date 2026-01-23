# Quantum Hardware Integration

**Comprehensive guides for integrating QuantRS2 with major quantum computing platforms**

QuantRS2 provides seamless integration with leading quantum hardware providers, enabling you to run your quantum circuits on real quantum computers with minimal code changes.

## 🎯 Supported Platforms

| Provider | Status | Qubits | Gate Set | Access Method |
|----------|--------|--------|----------|---------------|
| **IBM Quantum** | ✅ Full Support | Up to 127 | Native basis gates | IBMQ Account |
| **Google Quantum AI** | ✅ Full Support | Up to 70 | Sycamore gates | Google Cloud |
| **AWS Braket** | ✅ Full Support | Various | Provider-specific | AWS Account |
| **Azure Quantum** | ✅ Full Support | Various | Provider-specific | Azure Account |
| **Rigetti Computing** | ✅ Full Support | Up to 80 | Native gates | Rigetti Cloud |
| **IonQ** | ✅ Full Support | Up to 32 | All-to-all connectivity | Direct API |
| **Xanadu** | ✅ Full Support | Up to 216 | Photonic gates | Xanadu Cloud |
| **QuEra** | 🔄 Beta | Up to 256 | Neutral atom | QuEra Platform |

## 🚀 Quick Start

### Universal Hardware Interface

```python
import quantrs2

# Create a circuit
circuit = quantrs2.Circuit(2)
circuit.h(0).cx(0, 1)

# Run on any hardware with unified interface
providers = [
    quantrs2.hardware.IBM('ibmq_qasm_simulator'),
    quantrs2.hardware.Google('rainbow'),
    quantrs2.hardware.AWS('sv1'),
    quantrs2.hardware.IonQ('simulator'),
]

for provider in providers:
    result = circuit.run(backend=provider, shots=1024)
    print(f"{provider.name}: {result.counts}")
```

### Automatic Hardware Optimization

```python
# QuantRS2 automatically optimizes for target hardware
circuit = quantrs2.Circuit(4)
# Build your circuit logic here...

# Hardware-specific compilation happens automatically
ibm_backend = quantrs2.hardware.IBM('ibmq_manila')
optimized_circuit = circuit.compile_for(ibm_backend)
result = optimized_circuit.run(shots=1000)
```

## 📚 Integration Guides

### Cloud Providers
- **[IBM Quantum](ibm/index.md)** - Complete integration with IBM's quantum systems
- **[Google Quantum AI](google/index.md)** - Access to Google's Sycamore processors
- **[AWS Braket](aws/index.md)** - Multi-provider access through Amazon Web Services
- **[Azure Quantum](azure/index.md)** - Microsoft's quantum cloud platform

### Hardware Vendors
- **[Rigetti Computing](rigetti/index.md)** - Superconducting quantum processors
- **[IonQ](ionq/index.md)** - Trapped-ion quantum computers
- **[Xanadu](xanadu/index.md)** - Photonic quantum computing
- **[QuEra](quera/index.md)** - Neutral atom quantum systems

### Simulators
- **[Local Simulators](simulators/local.md)** - High-performance local simulation
- **[GPU Simulators](simulators/gpu.md)** - CUDA and OpenCL acceleration
- **[Distributed Simulators](simulators/distributed.md)** - Multi-node simulation

## ⚡ Key Features

### 1. Unified API
```python
# Same code works across all providers
result = circuit.run(
    backend="ibmq_qasm_simulator",  # or "google_rainbow", "aws_sv1", etc.
    shots=1024,
    optimization_level=2
)
```

### 2. Automatic Transpilation
```python
# Hardware-specific gate decomposition
circuit.rx(0, np.pi/4)  # Automatically becomes RZ-SX-RZ on IBM
circuit.ry(1, np.pi/3)  # Optimized for target hardware
```

### 3. Error Mitigation
```python
# Built-in error mitigation
result = circuit.run(
    backend="ibmq_montreal",
    shots=1024,
    error_mitigation="zero_noise_extrapolation"
)
```

### 4. Queue Management
```python
# Intelligent job scheduling
job = circuit.run_async(
    backend="ibmq_montreal",
    priority="normal",
    max_wait_time="1 hour"
)

# Get results when ready
result = job.result()
```

## 🔧 Hardware Configuration

### Provider Authentication

**IBM Quantum**
```python
# Configure IBM credentials
quantrs2.hardware.configure_ibm(
    token="your_ibm_token",
    hub="ibm-q",
    group="open", 
    project="main"
)
```

**Google Cloud**
```python
# Configure Google Cloud credentials
quantrs2.hardware.configure_google(
    project_id="your-project-id",
    service_account_path="path/to/credentials.json"
)
```

**AWS Braket**
```python
# Configure AWS credentials
quantrs2.hardware.configure_aws(
    access_key_id="your_access_key",
    secret_access_key="your_secret_key",
    region="us-east-1"
)
```

### Backend Selection

```python
# List available backends
backends = quantrs2.hardware.list_backends(
    provider="ibm",
    status="online",
    min_qubits=5
)

# Select optimal backend
best_backend = quantrs2.hardware.select_optimal(
    circuit=my_circuit,
    providers=["ibm", "google", "rigetti"],
    criteria=["queue_time", "fidelity", "connectivity"]
)
```

## 📊 Hardware Comparison

### Performance Characteristics

| Provider | Typical Queue | Gate Fidelity | Connectivity | Best For |
|----------|---------------|---------------|--------------|----------|
| **IBM** | 5-30 min | 99.5-99.9% | Heavy-hex | Algorithm research |
| **Google** | 2-15 min | 99.6-99.9% | Grid | Machine learning |
| **Rigetti** | 1-10 min | 99.0-99.5% | Chimera | Optimization |
| **IonQ** | 10-60 min | 99.8-99.9% | All-to-all | Chemistry |
| **Xanadu** | 1-5 min | 99.0-99.5% | Gaussian | Photonic ML |

### Cost Optimization

```python
# Compare costs across providers
cost_estimate = quantrs2.hardware.estimate_cost(
    circuit=my_circuit,
    shots=1024,
    providers=["ibm", "google", "aws"]
)

print(f"IBM: ${cost_estimate.ibm:.2f}")
print(f"Google: ${cost_estimate.google:.2f}")
print(f"AWS: ${cost_estimate.aws:.2f}")

# Run on most cost-effective option
cheapest = min(cost_estimate, key=lambda x: x.cost)
result = circuit.run(backend=cheapest.backend, shots=1024)
```

## 🛠️ Advanced Features

### Circuit Optimization

```python
# Hardware-aware circuit optimization
optimized = quantrs2.optimize_for_hardware(
    circuit=my_circuit,
    backend="ibmq_montreal",
    optimization_level=3,
    layout_method="sabre",
    routing_method="stochastic"
)
```

### Noise Characterization

```python
# Characterize hardware noise
noise_model = quantrs2.hardware.characterize_noise(
    backend="ibmq_montreal",
    characterization_depth=["rb", "process_tomography"]
)

# Use noise model for simulation
result = circuit.run(
    backend="simulator",
    noise_model=noise_model,
    shots=1024
)
```

### Batch Job Management

```python
# Submit multiple jobs efficiently
jobs = quantrs2.hardware.batch_submit([
    (circuit1, "ibmq_montreal"),
    (circuit2, "ibmq_mumbai"), 
    (circuit3, "ibmq_kolkata")
], shots=1024)

# Monitor progress
for job in jobs:
    print(f"Job {job.id}: {job.status()}")

# Collect results
results = [job.result() for job in jobs]
```

## 🔒 Security and Compliance

### Data Privacy

```python
# Enable enhanced privacy mode
quantrs2.config.set_privacy_mode(
    encrypt_circuits=True,
    anonymous_submission=True,
    local_preprocessing=True
)
```

### Compliance Features

- **GDPR Compliance**: Automatic data handling controls
- **ITAR Compliance**: Restricted algorithm detection
- **SOC 2**: Audit logging and access controls
- **HIPAA**: Healthcare data protection modes

## 📚 Documentation and Support

### Getting Help

- **[Hardware FAQ](faq.md)** - Common questions and troubleshooting
- **[Best Practices](best-practices.md)** - Optimization tips and guidelines
- **[Troubleshooting](troubleshooting.md)** - Debugging hardware issues
- **[Performance Tuning](performance.md)** - Maximizing hardware efficiency

### Community Resources

- **Discord**: `#hardware-integration` channel
- **GitHub**: Hardware-specific issue tracking
- **Forums**: Provider-specific discussions
- **Office Hours**: Weekly hardware Q&A sessions

## 🚀 Getting Started

### 1. Choose Your Provider

Start with the provider that best matches your needs:
- **Research/Education**: IBM Quantum or Rigetti
- **Production Applications**: AWS Braket or Azure Quantum
- **Machine Learning**: Google Quantum AI or Xanadu
- **High Fidelity**: IonQ or QuEra

### 2. Set Up Authentication

Follow the provider-specific guides to configure your credentials securely.

### 3. Run Your First Circuit

```python
import quantrs2

# Create a simple circuit
circuit = quantrs2.Circuit(2)
circuit.h(0).cx(0, 1)

# Run on real hardware
result = circuit.run(
    backend="ibmq_qasm_simulator",  # Start with simulator
    shots=1024
)

print(f"Results: {result.counts}")
print(f"Execution time: {result.execution_time}")
```

### 4. Optimize for Production

Once comfortable, explore advanced features like error mitigation, circuit optimization, and cost management.

---

**Ready to run on quantum hardware?** Choose your provider from the guides above and start executing quantum circuits on real quantum computers in minutes!

*Last updated: January 2026 | Next review: April 2026*