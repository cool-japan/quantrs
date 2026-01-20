# QuantRS2 WebAssembly - Browser-Based Quantum Computing

Run quantum circuit simulations directly in your web browser with near-native performance!

## 🚀 Features

- **Zero Installation**: Works directly in any modern web browser
- **High Performance**: Compiled to WebAssembly for near-native speed
- **Interactive**: Real-time circuit visualization and state inspection
- **Educational**: Perfect for learning quantum computing concepts
- **Cross-Platform**: Works on any device with a modern browser

## 📦 Building

### Prerequisites

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Or with cargo
cargo install wasm-pack
```

### Build the WASM Module

```bash
# From the wasm directory
cd wasm
wasm-pack build --target web --release
```

This will create a `pkg/` directory containing:
- `quantrs2_wasm.js` - JavaScript bindings
- `quantrs2_wasm_bg.wasm` - Compiled WebAssembly module
- `quantrs2_wasm.d.ts` - TypeScript definitions

## 🎮 Running the Demo

### Option 1: Simple HTTP Server

```bash
# Python 3
python3 -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000

# Node.js
npx http-server
```

Then open `http://localhost:8000/demo.html`

### Option 2: Direct File Access

Some browsers support direct file access. Simply open `demo.html` in your browser.

**Note**: Chrome requires `--allow-file-access-from-files` flag for local file access.

## 💻 Usage Examples

### Basic Usage

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>QuantRS2 WASM Example</title>
</head>
<body>
    <script type="module">
        import init, { WasmCircuit } from './pkg/quantrs2_wasm.js';

        async function main() {
            // Initialize the WASM module
            await init();

            // Create a 2-qubit circuit
            const circuit = new WasmCircuit(2);

            // Build a Bell state
            circuit.h(0);      // Hadamard on qubit 0
            circuit.cnot(0, 1); // CNOT from 0 to 1

            // Run the simulation
            const result = circuit.run();

            // Get probabilities
            const probs = result.probabilities();
            console.log('Probabilities:', probs);

            // Get most probable state
            console.log('Most probable state:', result.most_probable_state());
        }

        main();
    </script>
</body>
</html>
```

### Creating Quantum States

```javascript
import init, { create_bell_state, create_ghz_state } from './pkg/quantrs2_wasm.js';

await init();

// Bell state (2 qubits)
const bell = create_bell_state();
const bellResult = bell.run();

// GHZ state (3 qubits)
const ghz = create_ghz_state(3);
const ghzResult = ghz.run();
```

### Custom Circuits

```javascript
const circuit = new WasmCircuit(3);

// Apply gates
circuit.h(0);              // Hadamard
circuit.x(1);              // Pauli-X (NOT)
circuit.y(2);              // Pauli-Y
circuit.z(0);              // Pauli-Z
circuit.s(1);              // S gate
circuit.t(2);              // T gate

// Rotation gates
circuit.rx(0, Math.PI / 4); // RX rotation
circuit.ry(1, Math.PI / 3); // RY rotation
circuit.rz(2, Math.PI / 6); // RZ rotation

// Two-qubit gates
circuit.cnot(0, 1);        // CNOT
circuit.cz(1, 2);          // CZ
circuit.swap(0, 2);        // SWAP

// Run simulation
const result = circuit.run();

// Access results
const probabilities = result.probabilities();
const mostProbable = result.most_probable_state();
const stateProbs = result.state_probabilities();
```

## 📊 API Reference

### WasmCircuit

Create and manipulate quantum circuits.

```javascript
const circuit = new WasmCircuit(n_qubits);
```

#### Single-Qubit Gates

- `h(qubit)` - Hadamard gate
- `x(qubit)` - Pauli-X (NOT) gate
- `y(qubit)` - Pauli-Y gate
- `z(qubit)` - Pauli-Z gate
- `s(qubit)` - S gate (phase)
- `t(qubit)` - T gate (π/8)
- `rx(qubit, theta)` - Rotation around X-axis
- `ry(qubit, theta)` - Rotation around Y-axis
- `rz(qubit, theta)` - Rotation around Z-axis

#### Two-Qubit Gates

- `cnot(control, target)` - Controlled-NOT
- `cz(control, target)` - Controlled-Z
- `swap(qubit1, qubit2)` - SWAP gate

#### Methods

- `run()` - Execute the circuit and return results
- `to_qasm()` - Export circuit as QASM string
- `to_json()` - Export circuit as JSON object
- `n_qubits` - Get number of qubits (property)

### WasmResult

Access simulation results.

```javascript
const result = circuit.run();
```

#### Methods

- `probabilities()` - Get array of probabilities for all basis states
- `get_amplitude(index)` - Get complex amplitude for a specific state
- `amplitudes_flat()` - Get all amplitudes as flat array [re0, im0, re1, im1, ...]
- `state_probabilities()` - Get map of non-zero state probabilities
- `most_probable_state()` - Get binary string of most probable state
- `n_qubits` - Get number of qubits (property)

### Helper Functions

```javascript
create_bell_state()           // Create Bell state circuit
create_ghz_state(n_qubits)    // Create GHZ state circuit
version()                     // Get QuantRS2 version
```

## 🎓 Educational Examples

### Example 1: Quantum Superposition

```javascript
const circuit = new WasmCircuit(2);
circuit.h(0);  // Put qubit 0 in superposition
circuit.h(1);  // Put qubit 1 in superposition

const result = circuit.run();
const probs = result.probabilities();

// All 4 states (|00⟩, |01⟩, |10⟩, |11⟩) should have equal probability (0.25 each)
console.log('Probabilities:', probs);
```

### Example 2: Quantum Entanglement

```javascript
const circuit = new WasmCircuit(2);
circuit.h(0);       // Create superposition
circuit.cnot(0, 1); // Entangle qubits

const result = circuit.run();

// Only |00⟩ and |11⟩ should have non-zero probability (0.5 each)
const stateProbs = result.state_probabilities();
console.log('State probabilities:', stateProbs);
```

### Example 3: Quantum Phase

```javascript
const circuit = new WasmCircuit(1);
circuit.h(0);      // Superposition
circuit.z(0);      // Apply phase
circuit.h(0);      // Hadamard again

const result = circuit.run();
const probs = result.probabilities();

// Should collapse to |1⟩ with probability 1.0
console.log('Final state:', result.most_probable_state());
```

## 🔧 TypeScript Support

TypeScript definitions are automatically generated:

```typescript
import init, { WasmCircuit, WasmResult } from './pkg/quantrs2_wasm';

async function example(): Promise<void> {
    await init();

    const circuit: WasmCircuit = new WasmCircuit(2);
    circuit.h(0);
    circuit.cnot(0, 1);

    const result: WasmResult = circuit.run();
    const probs: number[] = result.probabilities();

    console.log(probs);
}
```

## 📈 Performance Considerations

### Recommended Qubit Limits

- **Interactive demos**: 1-15 qubits (instant feedback)
- **Educational use**: 16-20 qubits (good performance)
- **Research/testing**: 21-25 qubits (slower, may freeze browser)
- **Not recommended**: >25 qubits (will likely crash browser)

### Optimization Tips

1. **Use fewer qubits** for interactive demos
2. **Minimize gate count** where possible
3. **Batch operations** instead of frequent small updates
4. **Profile with browser DevTools** to identify bottlenecks

### Browser Compatibility

Tested on:
- ✅ Chrome/Edge 90+
- ✅ Firefox 89+
- ✅ Safari 15+
- ✅ Mobile browsers (iOS Safari, Chrome Android)

## 🐛 Troubleshooting

### WASM Module Not Loading

```
Error: WebAssembly module failed to load
```

**Solution**: Ensure you're serving the files over HTTP (not `file://`). Use a local web server.

### CORS Errors

```
Access to fetch blocked by CORS policy
```

**Solution**: Serve from a proper web server with correct MIME types, or use `--disable-web-security` flag (development only).

### Memory Errors

```
RuntimeError: memory access out of bounds
```

**Solution**: Reduce the number of qubits. Browser WASM has memory limits.

## 🔗 Integration Examples

### React

```tsx
import { useEffect, useState } from 'react';
import init, { WasmCircuit } from './pkg/quantrs2_wasm';

function QuantumDemo() {
    const [ready, setReady] = useState(false);

    useEffect(() => {
        init().then(() => setReady(true));
    }, []);

    const runCircuit = () => {
        if (!ready) return;

        const circuit = new WasmCircuit(2);
        circuit.h(0);
        circuit.cnot(0, 1);

        const result = circuit.run();
        console.log(result.probabilities());
    };

    return (
        <div>
            <button onClick={runCircuit} disabled={!ready}>
                Run Bell State
            </button>
        </div>
    );
}
```

### Vue.js

```vue
<template>
    <div>
        <button @click="runCircuit" :disabled="!ready">
            Run Bell State
        </button>
    </div>
</template>

<script>
import init, { WasmCircuit } from './pkg/quantrs2_wasm';

export default {
    data() {
        return {
            ready: false
        };
    },
    async mounted() {
        await init();
        this.ready = true;
    },
    methods: {
        runCircuit() {
            const circuit = new WasmCircuit(2);
            circuit.h(0);
            circuit.cnot(0, 1);
            const result = circuit.run();
            console.log(result.probabilities());
        }
    }
};
</script>
```

## 📝 License

Same as QuantRS2 main project.

## 🤝 Contributing

Contributions are welcome! Please follow the main QuantRS2 contribution guidelines.

## 📚 Resources

- [QuantRS2 Documentation](https://github.com/cool-japan/quantrs)
- [WebAssembly MDN Docs](https://developer.mozilla.org/en-US/docs/WebAssembly)
- [wasm-bindgen Guide](https://rustwasm.github.io/wasm-bindgen/)
- [Quantum Computing Primer](https://quantum-computing.ibm.com/)

## 🎉 Acknowledgments

- Built with [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen)
- Powered by [SciRS2](https://github.com/cool-japan/scirs)
- Quantum algorithms inspired by IBM Qiskit and Google Cirq
