# Migration Guide to QuantRS2-Tytan

This guide helps users migrate from other quantum optimization frameworks to QuantRS2-Tytan.

## Table of Contents

1. [From Python Tytan](#from-python-tytan)
2. [From D-Wave Ocean](#from-d-wave-ocean)
3. [From Qiskit Optimization](#from-qiskit-optimization)
4. [From Microsoft QIO](#from-microsoft-qio)
5. [From PennyLane](#from-pennylane)
6. [General Migration Tips](#general-migration-tips)

## From Python Tytan

QuantRS2-Tytan is inspired by Python Tytan, making migration straightforward.

### Symbol Definition

**Python Tytan:**
```python
from tytan import symbols

q = symbols("q")
x, y, z = symbols("x y z")
```

**QuantRS2-Tytan:**
```rust
use quantrs2_tytan::symbol::symbols;

let q = symbols("q");
let x = symbols("x");
let y = symbols("y");
let z = symbols("z");
```

### Expression Building

**Python Tytan:**
```python
H = (x + y + z - 2)**2
```

**QuantRS2-Tytan:**
```rust
let h = (x + y + z - 2).pow(2);
```

### Compilation

**Python Tytan:**
```python
from tytan import compile

qubo, offset = compile(H).get_qubo()
```

**QuantRS2-Tytan:**
```rust
use quantrs2_tytan::compile::Compile;

let (qubo, offset) = Compile::new(&h).get_qubo()?;
```

### Sampling

**Python Tytan:**
```python
from tytan.sampler import SASampler

solver = SASampler()
result = solver.run(qubo, shots=100)
```

**QuantRS2-Tytan:**
```rust
use quantrs2_tytan::sampler::{SASampler, Sampler};

let solver = SASampler::new(None);
let result = solver.run_qubo(&qubo, 100)?;
```

### Auto Array Processing

**Python Tytan:**
```python
from tytan import Auto_array

auto = Auto_array(result)
for r in auto.calc_score():
    print(r)
```

**QuantRS2-Tytan:**
```rust
use quantrs2_tytan::auto_array::Auto_array;

let auto = Auto_array::new();
let scores = auto.calc_score(result);
for r in scores {
    println!("{:?}", r);
}
```

## From D-Wave Ocean

### Binary Quadratic Model (BQM)

**D-Wave Ocean:**
```python
from dimod import BinaryQuadraticModel

bqm = BinaryQuadraticModel('BINARY')
bqm.add_variable('x', -1)
bqm.add_variable('y', -1)
bqm.add_interaction('x', 'y', 2)
```

**QuantRS2-Tytan:**
```rust
use ndarray::Array2;
use std::collections::HashMap;

let mut qubo = Array2::zeros((2, 2));
qubo[[0, 0]] = -1.0;  // x
qubo[[1, 1]] = -1.0;  // y
qubo[[0, 1]] = 2.0;   // x*y interaction
qubo[[1, 0]] = 2.0;   // symmetric

let mut var_map = HashMap::new();
var_map.insert("x".to_string(), 0);
var_map.insert("y".to_string(), 1);
```

### Constraint Satisfaction Problems (CSP)

**D-Wave Ocean:**
```python
from dimod import ConstraintSatisfactionProblem

csp = ConstraintSatisfactionProblem()
csp.add_constraint(lambda x, y: x + y == 1, ['x', 'y'])
bqm = csp.get_bqm()
```

**QuantRS2-Tytan:**
```rust
use quantrs2_tytan::constraints::add_equality_constraint;

// Add constraint: x + y = 1
add_equality_constraint(&mut qubo, &[0, 1], 1.0, 10.0);
```

### Embedding

**D-Wave Ocean:**
```python
from minorminer import find_embedding

embedding = find_embedding(source_graph, target_graph)
embedded_bqm = embed_bqm(bqm, embedding, target_graph)
```

**QuantRS2-Tytan:**
```rust
// Automatic embedding handled by DWaveSampler
use quantrs2_tytan::sampler::DWaveSampler;

let solver = DWaveSampler::new(None);
let result = solver.run_qubo(&qubo, 100)?;
```

### Hybrid Solvers

**D-Wave Ocean:**
```python
from hybrid import SimulatedAnnealingProblemSampler

sampler = SimulatedAnnealingProblemSampler()
result = sampler.sample(bqm, num_reads=100)
```

**QuantRS2-Tytan:**
```rust
use quantrs2_tytan::hybrid_algorithms::HybridSolver;

let solver = HybridSolver::new(Default::default());
let result = solver.run_qubo(&qubo, 100)?;
```

## From Qiskit Optimization

### Quadratic Program

**Qiskit:**
```python
from qiskit_optimization import QuadraticProgram

qp = QuadraticProgram()
qp.binary_var('x')
qp.binary_var('y')
qp.minimize(quadratic={'x': -1, 'y': -1, ('x', 'y'): 2})
```

**QuantRS2-Tytan:**
```rust
// Using symbolic math (requires 'dwave' feature)
use quantrs2_tytan::{symbols, Compile};

let x = symbols("x");
let y = symbols("y");
let expr = -x - y + 2*x*y;

let (qubo, _) = Compile::new(&expr).get_qubo()?;
```

### VQE Integration

**Qiskit:**
```python
from qiskit.algorithms import VQE
from qiskit.primitives import Estimator

vqe = VQE(estimator=Estimator(), ansatz=ansatz, optimizer=optimizer)
result = vqe.compute_minimum_eigenvalue(operator)
```

**QuantRS2-Tytan:**
```rust
use quantrs2_tytan::quantum_optimization_extensions::VQEOptimizer;

let vqe = VQEOptimizer::new(Default::default());
let result = vqe.minimize(&qubo)?;
```

### Constraint Handling

**Qiskit:**
```python
qp.linear_constraint(['x', 'y'], [1, 1], '==', 1)
```

**QuantRS2-Tytan:**
```rust
use quantrs2_tytan::optimization::constraints::LinearConstraint;

let constraint = LinearConstraint::equality(vec![0, 1], vec![1.0, 1.0], 1.0);
let penalized_qubo = constraint.apply_to_qubo(&qubo, 10.0);
```

## From Microsoft QIO

### Problem Definition

**Microsoft QIO:**
```python
from azure.quantum.optimization import Problem, ProblemType

problem = Problem(name="my-problem", problem_type=ProblemType.ising)
problem.add_term(c=-1, indices=[0])
problem.add_term(c=-1, indices=[1])
problem.add_term(c=2, indices=[0, 1])
```

**QuantRS2-Tytan:**
```rust
use quantrs2_anneal::{IsingModel, QuboModel};

// As QUBO
let mut qubo = QuboModel::new();
qubo.add_term(&[0], -1.0);
qubo.add_term(&[1], -1.0);
qubo.add_term(&[0, 1], 2.0);

// Or as Ising
let mut ising = IsingModel::new();
ising.add_field(0, -1.0);
ising.add_field(1, -1.0);
ising.add_coupling(0, 1, 2.0);
```

### Solver Configuration

**Microsoft QIO:**
```python
from azure.quantum.optimization import SimulatedAnnealing

solver = SimulatedAnnealing(workspace, timeout=100, seed=42)
result = solver.optimize(problem)
```

**QuantRS2-Tytan:**
```rust
use quantrs2_tytan::sampler::{SASampler, Config};

let solver = SASampler::new(Some(Config {
    num_sweeps: 10000,
    temperature_schedule: Schedule::Geometric(100.0, 0.1),
    seed: Some(42),
}));
let result = solver.run_qubo(&qubo_data, 100)?;
```

## From PennyLane

### QAOA

**PennyLane:**
```python
import pennylane as qml

dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def qaoa_circuit(params):
    # QAOA circuit implementation
    return qml.expval(cost_h)
```

**QuantRS2-Tytan:**
```rust
use quantrs2_tytan::quantum_optimization_extensions::QAOA;

let qaoa = QAOA::new(num_layers, Default::default());
let result = qaoa.optimize(&qubo)?;
```

### Variational Circuits

**PennyLane:**
```python
def variational_circuit(params, wires):
    for i in range(len(wires)):
        qml.RY(params[i], wires=i)
    qml.broadcast(qml.CNOT, wires=wires, pattern="chain")
```

**QuantRS2-Tytan:**
```rust
use quantrs2_tytan::variational_quantum_factoring::VariationalCircuit;

let circuit = VariationalCircuit::new(num_qubits);
circuit.add_ry_layer(&params);
circuit.add_entangling_layer();
```

## General Migration Tips

### 1. Data Structure Conversion

Most frameworks use different data structures. Here's how to convert:

```rust
// From dictionary/map representation
fn dict_to_qubo(linear: HashMap<usize, f64>, 
                quadratic: HashMap<(usize, usize), f64>, 
                num_vars: usize) -> Array2<f64> {
    let mut qubo = Array2::zeros((num_vars, num_vars));
    
    // Add linear terms
    for (var, coeff) in linear {
        qubo[[var, var]] = coeff;
    }
    
    // Add quadratic terms
    for ((i, j), coeff) in quadratic {
        qubo[[i, j]] = coeff;
        qubo[[j, i]] = coeff;  // Symmetric
    }
    
    qubo
}
```

### 2. Async/Parallel Execution

QuantRS2-Tytan supports parallel execution natively:

```rust
use rayon::prelude::*;

// Run multiple problems in parallel
let results: Vec<_> = problems
    .par_iter()
    .map(|problem| solver.run_qubo(problem, 100))
    .collect();
```

### 3. Custom Samplers

Create custom samplers by implementing the `Sampler` trait:

```rust
use quantrs2_tytan::sampler::{Sampler, SampleResult};

struct MySampler {
    // Your fields
}

impl Sampler for MySampler {
    fn run_qubo(&self, 
                qubo: &(Array2<f64>, HashMap<String, usize>), 
                num_samples: usize) -> Result<Vec<SampleResult>, Box<dyn Error>> {
        // Your implementation
        todo!()
    }
    
    fn run_hobo(&self,
                hobo: &(ArrayD<f64>, HashMap<String, usize>),
                num_samples: usize) -> Result<Vec<SampleResult>, Box<dyn Error>> {
        // Your implementation
        todo!()
    }
}
```

### 4. Performance Optimization

Enable features for better performance:

```toml
[dependencies]
quantrs2-tytan = { 
    version = "0.1.0-alpha.3",
    features = ["gpu", "scirs", "parallel"]
}
```

### 5. Error Handling

QuantRS2-Tytan uses Rust's Result type for error handling:

```rust
// Python style (exceptions)
# try:
#     result = solver.sample(bqm)
# except Exception as e:
#     print(f"Error: {e}")

// Rust style (Result)
match solver.run_qubo(&qubo, 100) {
    Ok(result) => {
        // Process result
    },
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}

// Or use ? operator
let result = solver.run_qubo(&qubo, 100)?;
```

### 6. Integration with Existing Code

For gradual migration, you can use Python bindings:

```python
# Install the Python package
# pip install quantrs2

import quantrs2

# Use QuantRS2-Tytan from Python
solver = quantrs2.tytan.SASampler()
result = solver.run_qubo(qubo_matrix, var_map, num_samples=100)
```

## Feature Comparison Table

| Feature | D-Wave Ocean | Qiskit | QIO | PennyLane | QuantRS2-Tytan |
|---------|--------------|--------|-----|-----------|----------------|
| Symbolic Math | ❌ | ❌ | ❌ | ✅ | ✅ |
| GPU Support | ❌ | Limited | ❌ | ✅ | ✅ |
| HOBO Support | ❌ | ❌ | ❌ | ❌ | ✅ |
| Native Performance | ❌ | ❌ | ❌ | ❌ | ✅ |
| Cloud Integration | ✅ | ✅ | ✅ | ❌ | ✅ |
| Constraint Handling | ✅ | ✅ | ✅ | ❌ | ✅ |
| Hybrid Algorithms | ✅ | ✅ | ✅ | ✅ | ✅ |
| Problem Decomposition | Limited | ❌ | ❌ | ❌ | ✅ |
| ML Integration | ❌ | Limited | ❌ | ✅ | ✅ |

## Common Pitfalls and Solutions

### 1. Variable Indexing
- Most frameworks use string-based variable names
- QuantRS2-Tytan uses both strings and indices
- Always maintain a variable map for consistency

### 2. Matrix Representation
- Some frameworks use upper-triangular matrices
- QuantRS2-Tytan uses full symmetric matrices
- Convert carefully to avoid doubling coefficients

### 3. Energy Sign Convention
- Check whether the framework minimizes or maximizes
- QuantRS2-Tytan minimizes by default
- Negate your objective if needed

### 4. Constraint Penalties
- Different frameworks use different penalty scaling
- Start with penalty = 10 * max(|coefficients|)
- Tune based on your specific problem

### 5. Result Format
- Results may be in different formats (dict, array, etc.)
- QuantRS2-Tytan returns structured `SampleResult` objects
- Use provided conversion utilities

## Getting Help

1. Check the [API Reference](API_REFERENCE.md)
2. Review [examples](../examples/)
3. Open an issue on [GitHub](https://github.com/cool-japan/quantrs)
4. Join our community discussions