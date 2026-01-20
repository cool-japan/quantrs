# Algorithms API Reference

The QuantRS2 algorithms module provides implementations of quantum algorithms and debugging tools.

## Algorithm Debugger

::: quantrs2.algorithm_debugger
    options:
      members:
        - AlgorithmDebugger
        - DebugSession
        - DebugBreakpoint
        - DebuggerState
        - QuantumState
        - create_debug_session
        - debug_algorithm
        - set_breakpoint
        - step_execution
        - inspect_state

### AlgorithmDebugger

The main debugger class for quantum algorithms.

#### Methods

- `create_session(algorithm, initial_state)`: Create a new debugging session
- `step()`: Execute one step of the algorithm
- `run_to_breakpoint()`: Run until next breakpoint
- `inspect_state()`: Get current quantum state information
- `set_breakpoint(line, condition)`: Set conditional breakpoint
- `get_call_stack()`: Get current execution stack

#### Usage Example

```python
from quantrs2.algorithm_debugger import AlgorithmDebugger, create_debug_session

# Create a simple quantum algorithm
def grover_search(oracle, n_qubits):
    circuit = Circuit(n_qubits)
    # Initialization
    for i in range(n_qubits):
        circuit.h(i)
    
    # Grover iterations
    for _ in range(int(np.pi/4 * np.sqrt(2**n_qubits))):
        # Oracle
        oracle(circuit)
        # Diffusion operator
        for i in range(n_qubits):
            circuit.h(i)
            circuit.x(i)
        circuit.multi_controlled_z(list(range(n_qubits)))
        for i in range(n_qubits):
            circuit.x(i)
            circuit.h(i)
    
    return circuit

# Debug the algorithm
debugger = AlgorithmDebugger()
oracle = lambda c: c.z(0)  # Simple oracle
session = debugger.create_session(grover_search, oracle, 3)

# Step through execution
session.step()
state = session.inspect_state()
print(f"Amplitudes: {state.amplitudes}")
```

### DebugSession

A debugging session for a specific algorithm execution.

#### Attributes

- `algorithm`: The algorithm being debugged
- `state`: Current quantum state
- `breakpoints`: List of active breakpoints
- `call_stack`: Current execution stack
- `step_count`: Number of steps executed

#### Methods

- `step()`: Execute single step
- `run()`: Run to completion
- `pause()`: Pause execution
- `reset()`: Reset to initial state

### DebugBreakpoint

Represents a breakpoint in algorithm execution.

#### Attributes

- `location`: Location of breakpoint (line number or gate index)
- `condition`: Optional condition for triggering
- `enabled`: Whether breakpoint is active
- `hit_count`: Number of times hit

## Algorithm Marketplace

::: quantrs2.algorithm_marketplace
    options:
      members:
        - AlgorithmMarketplace
        - AlgorithmRegistry
        - AlgorithmInfo
        - submit_algorithm
        - search_algorithms
        - download_algorithm
        - rate_algorithm

### AlgorithmMarketplace

Central marketplace for sharing and discovering quantum algorithms.

#### Methods

- `search(category, tags, author)`: Search for algorithms
- `submit(algorithm, metadata)`: Submit new algorithm
- `download(algorithm_id)`: Download algorithm by ID
- `rate(algorithm_id, rating)`: Rate an algorithm
- `get_popular()`: Get most popular algorithms
- `get_recent()`: Get recently added algorithms

#### Usage Example

```python
from quantrs2.algorithm_marketplace import AlgorithmMarketplace

marketplace = AlgorithmMarketplace()

# Search for VQE algorithms
vqe_algorithms = marketplace.search(
    category="optimization",
    tags=["VQE", "variational"]
)

for algo in vqe_algorithms:
    print(f"{algo.name}: {algo.description}")
    print(f"Rating: {algo.average_rating}/5")
    print(f"Author: {algo.author}")

# Download and use an algorithm
vqe_impl = marketplace.download(vqe_algorithms[0].id)
result = vqe_impl.run(molecule="H2", basis="sto-3g")
```

### AlgorithmInfo

Metadata for marketplace algorithms.

#### Attributes

- `id`: Unique algorithm identifier
- `name`: Human-readable name
- `description`: Detailed description
- `author`: Algorithm author
- `category`: Algorithm category
- `tags`: List of tags
- `version`: Version string
- `created_date`: Creation timestamp
- `average_rating`: Average user rating
- `download_count`: Number of downloads

## Core Quantum Algorithms

The module includes implementations of fundamental quantum algorithms:

### Grover's Algorithm

```python
def grovers_algorithm(oracle, n_qubits, target_state=None):
    """
    Implement Grover's quantum search algorithm.
    
    Args:
        oracle: Function that marks target state
        n_qubits: Number of qubits
        target_state: Optional target state specification
    
    Returns:
        Circuit implementing Grover's algorithm
    """
```

### Quantum Fourier Transform

```python
def quantum_fourier_transform(n_qubits, inverse=False):
    """
    Implement Quantum Fourier Transform.
    
    Args:
        n_qubits: Number of qubits
        inverse: Whether to implement inverse QFT
    
    Returns:
        Circuit implementing QFT
    """
```

### Variational Quantum Eigensolver (VQE)

```python
def vqe_algorithm(hamiltonian, ansatz, optimizer=None):
    """
    Implement Variational Quantum Eigensolver.
    
    Args:
        hamiltonian: Molecular Hamiltonian
        ansatz: Variational ansatz circuit
        optimizer: Classical optimizer
    
    Returns:
        VQE optimization result
    """
```

### Quantum Approximate Optimization Algorithm (QAOA)

```python
def qaoa_algorithm(cost_hamiltonian, mixer_hamiltonian, p_layers):
    """
    Implement QAOA for optimization problems.
    
    Args:
        cost_hamiltonian: Problem cost function
        mixer_hamiltonian: Mixing Hamiltonian
        p_layers: Number of QAOA layers
    
    Returns:
        QAOA circuit and optimization routine
    """
```

## Error Handling

The algorithms module provides specialized exceptions:

- `AlgorithmError`: Base exception for algorithm errors
- `DebuggerError`: Debugging-specific errors
- `MarketplaceError`: Marketplace operation errors
- `ConvergenceError`: Algorithm convergence failures

## Performance Considerations

- Algorithm debugging adds overhead - disable in production
- Marketplace caches popular algorithms locally
- Large algorithms may require streaming for download
- Consider using compiled variants for performance-critical code

## See Also

- [Core Module](core.md) for basic circuit operations
- [Gates](gates.md) for quantum gate implementations
- [Machine Learning](ml.md) for variational algorithms
- [Testing Tools](testing.md) for algorithm validation