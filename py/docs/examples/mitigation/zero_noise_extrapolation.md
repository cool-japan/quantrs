# Zero-Noise Extrapolation (ZNE)

**Level:** 🟡 Intermediate  
**Runtime:** 2-5 minutes  
**Topics:** Error mitigation, NISQ algorithms, Noise characterization  
**Applications:** Improving quantum algorithm results

Learn to mitigate quantum noise using Zero-Noise Extrapolation - one of the most practical and effective error mitigation techniques for near-term quantum devices.

## What is Zero-Noise Extrapolation?

Zero-Noise Extrapolation (ZNE) is a quantum error mitigation technique that extrapolates measurement results to the zero-noise limit without requiring additional quantum resources like extra qubits or complex error correction codes.

**The Key Idea:**
1. **Amplify** noise in quantum circuits by known factors
2. **Measure** expectation values at different noise levels
3. **Extrapolate** to zero noise using classical post-processing
4. **Recover** the ideal noiseless result

**Why ZNE Works:**
- Many quantum algorithms compute expectation values ⟨O⟩
- Noise typically decreases expectation values
- Relationship between noise and ⟨O⟩ is often smooth and predictable
- Classical extrapolation can recover ⟨O⟩_ideal

**Applications:**
- Variational quantum algorithms (VQE, QAOA)
- Quantum simulation and chemistry
- Quantum machine learning
- Any algorithm computing expectation values

## Theoretical Background

### Noise Models

**Depolarizing Noise:**
After each gate, with probability p, replace the state with the maximally mixed state:
```
ρ → (1-p)ρ + p(I/2^n)
```

**Global Noise Scaling:**
ZNE assumes we can scale noise by factor λ ≥ 1:
```
ρ_λ = Λ^λ(ρ_ideal)
```

Where Λ is the noise channel.

### Extrapolation Functions

**Linear Extrapolation:**
```
⟨O⟩(λ) = a + bλ
⟨O⟩(0) = a  (zero-noise limit)
```

**Exponential Extrapolation:**
```
⟨O⟩(λ) = A + B⋅r^λ
⟨O⟩(0) = A + B  (zero-noise limit)
```

**Polynomial Extrapolation:**
```
⟨O⟩(λ) = Σᵢ cᵢ λᵢ
⟨O⟩(0) = c₀  (zero-noise limit)
```

### Noise Amplification Methods

**Digital ZNE (Gate Folding):**
- Replace each gate G with G⁻¹G^(2k+1) = G
- Amplifies noise by factor (2k+1)

**Analog ZNE (Pulse Stretching):**
- Stretch pulse durations to increase decoherence
- Requires pulse-level control

**Identity Insertion:**
- Insert pairs of inverse gates (G, G⁻¹)
- Amplifies noise while preserving logic

## Implementation

### Basic ZNE Framework

```python
import quantrs2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Callable, Tuple, Dict, Any

class ZeroNoiseExtrapolation:
    """
    Zero-Noise Extrapolation implementation for quantum error mitigation.
    """
    
    def __init__(self, circuit_function: Callable, noise_factors: List[float] = None):
        """
        Initialize ZNE with a quantum circuit function.
        
        Args:
            circuit_function: Function that creates quantum circuit
            noise_factors: List of noise amplification factors (λ values)
        """
        self.circuit_function = circuit_function
        self.noise_factors = noise_factors or [1, 3, 5]  # Default noise factors
        
        # Results storage
        self.measurements = {}
        self.extrapolated_value = None
        self.extrapolation_error = None
        self.fit_parameters = None
        
        print(f"🛡️  Zero-Noise Extrapolation Setup:")
        print(f"   Noise factors: {self.noise_factors}")
        print(f"   Extrapolation points: {len(self.noise_factors)}")
    
    def apply_digital_noise_scaling(self, circuit: quantrs2.Circuit, noise_factor: float) -> quantrs2.Circuit:
        """
        Apply digital noise scaling using gate folding.
        
        Args:
            circuit: Original quantum circuit
            noise_factor: Noise amplification factor (must be odd integer ≥ 1)
            
        Returns:
            Circuit with amplified noise
        """
        if noise_factor < 1 or noise_factor % 2 == 0:
            raise ValueError("Noise factor must be odd integer ≥ 1")
        
        if noise_factor == 1:
            return circuit.copy()
        
        # Create new circuit with noise amplification
        scaled_circuit = quantrs2.Circuit(circuit.num_qubits)
        
        # Get gate sequence from original circuit
        gates = circuit.get_gate_sequence()  # This would be a method to extract gates
        
        # For each gate, apply folding: G → G†G^(2k+1) = G
        fold_count = (noise_factor - 1) // 2
        
        for gate in gates:
            # Apply original gate
            self._apply_gate(scaled_circuit, gate)
            
            # Apply folding pairs
            for _ in range(fold_count):
                # Apply inverse gate
                self._apply_inverse_gate(scaled_circuit, gate)
                # Apply original gate
                self._apply_gate(scaled_circuit, gate)
        
        return scaled_circuit
    
    def apply_identity_insertion(self, circuit: quantrs2.Circuit, noise_factor: float) -> quantrs2.Circuit:
        """
        Apply noise scaling using identity insertion.
        
        Args:
            circuit: Original quantum circuit
            noise_factor: Noise amplification factor
            
        Returns:
            Circuit with inserted identities
        """
        scaled_circuit = circuit.copy()
        
        # Number of identity pairs to insert
        num_identities = int((noise_factor - 1) * circuit.gate_count)
        
        # Insert identity operations (X followed by X)
        for _ in range(num_identities):
            for qubit in range(circuit.num_qubits):
                scaled_circuit.x(qubit)
                scaled_circuit.x(qubit)  # X² = I
        
        return scaled_circuit
    
    def _apply_gate(self, circuit: quantrs2.Circuit, gate_info: Dict[str, Any]):
        """Apply a gate to the circuit based on gate information."""
        
        gate_name = gate_info['name']
        qubits = gate_info['qubits']
        params = gate_info.get('parameters', [])
        
        if gate_name == 'H':
            circuit.h(qubits[0])
        elif gate_name == 'X':
            circuit.x(qubits[0])
        elif gate_name == 'Y':
            circuit.y(qubits[0])
        elif gate_name == 'Z':
            circuit.z(qubits[0])
        elif gate_name == 'RX':
            circuit.rx(qubits[0], params[0])
        elif gate_name == 'RY':
            circuit.ry(qubits[0], params[0])
        elif gate_name == 'RZ':
            circuit.rz(qubits[0], params[0])
        elif gate_name == 'CX':
            circuit.cx(qubits[0], qubits[1])
        elif gate_name == 'CZ':
            circuit.cz(qubits[0], qubits[1])
        # Add more gates as needed
    
    def _apply_inverse_gate(self, circuit: quantrs2.Circuit, gate_info: Dict[str, Any]):
        """Apply the inverse of a gate."""
        
        gate_name = gate_info['name']
        qubits = gate_info['qubits']
        params = gate_info.get('parameters', [])
        
        # Most Pauli gates are self-inverse
        if gate_name in ['H', 'X', 'Y', 'Z', 'CX', 'CZ']:
            self._apply_gate(circuit, gate_info)
        elif gate_name == 'RX':
            circuit.rx(qubits[0], -params[0])  # RX†(θ) = RX(-θ)
        elif gate_name == 'RY':
            circuit.ry(qubits[0], -params[0])
        elif gate_name == 'RZ':
            circuit.rz(qubits[0], -params[0])
        # Add more inverse gates as needed
    
    def measure_expectation_value(self, observable: str, num_shots: int = 1000) -> float:
        """
        Measure expectation value of an observable.
        
        Args:
            observable: Observable to measure (e.g., 'Z0', 'X1', 'Z0*Z1')
            num_shots: Number of measurement shots
            
        Returns:
            Expectation value
        """
        circuit = self.circuit_function()
        
        # Add measurement basis rotations for non-Z observables
        measurement_circuit = self._prepare_measurement_circuit(circuit, observable)
        measurement_circuit.measure_all()
        
        total_expectation = 0
        for _ in range(num_shots):
            result = measurement_circuit.run()
            expectation = self._compute_observable_expectation(result, observable)
            total_expectation += expectation
        
        return total_expectation / num_shots
    
    def _prepare_measurement_circuit(self, circuit: quantrs2.Circuit, observable: str) -> quantrs2.Circuit:
        """Prepare circuit for measuring specific observable."""
        
        measurement_circuit = circuit.copy()
        
        # Parse observable string (simplified)
        if 'X' in observable:
            qubit_idx = int(observable.split('X')[1].split('*')[0])
            measurement_circuit.ry(qubit_idx, -np.pi/2)  # Rotate X to Z basis
        elif 'Y' in observable:
            qubit_idx = int(observable.split('Y')[1].split('*')[0])
            measurement_circuit.rx(qubit_idx, np.pi/2)   # Rotate Y to Z basis
        
        return measurement_circuit
    
    def _compute_observable_expectation(self, measurement_result, observable: str) -> float:
        """Compute expectation value from measurement result."""
        
        probabilities = measurement_result.state_probabilities()
        expectation = 0
        
        for state_str, prob in probabilities.items():
            # Compute observable eigenvalue for this state
            eigenvalue = self._compute_observable_eigenvalue(state_str, observable)
            expectation += prob * eigenvalue
        
        return expectation
    
    def _compute_observable_eigenvalue(self, state_str: str, observable: str) -> float:
        """Compute eigenvalue of observable for given computational basis state."""
        
        # For Z observables: |0⟩ → +1, |1⟩ → -1
        if observable.startswith('Z'):
            qubit_idx = int(observable[1:])
            return 1 if state_str[qubit_idx] == '0' else -1
        
        # For multi-qubit observables like Z0*Z1
        if '*' in observable:
            terms = observable.split('*')
            eigenvalue = 1
            for term in terms:
                if term.startswith('Z'):
                    qubit_idx = int(term[1:])
                    eigenvalue *= 1 if state_str[qubit_idx] == '0' else -1
            return eigenvalue
        
        # For X, Y observables after basis rotation, treat as Z
        return 1 if state_str[0] == '0' else -1
    
    def run_zne_experiment(self, observable: str, num_shots: int = 1000, 
                          noise_amplification_method: str = 'digital') -> Dict[str, Any]:
        """
        Run complete ZNE experiment.
        
        Args:
            observable: Observable to measure
            num_shots: Number of shots per noise level
            noise_amplification_method: 'digital', 'identity', or 'analog'
            
        Returns:
            Dictionary with ZNE results
        """
        print(f"\n🚀 Running ZNE Experiment")
        print(f"   Observable: {observable}")
        print(f"   Shots per level: {num_shots}")
        print(f"   Amplification method: {noise_amplification_method}")
        
        # Collect measurements at each noise level
        noise_levels = []
        expectation_values = []
        
        for noise_factor in self.noise_factors:
            print(f"\n   Measuring at noise factor λ = {noise_factor}...")
            
            # Create circuit with amplified noise
            if noise_amplification_method == 'digital':
                noisy_circuit_func = lambda: self.apply_digital_noise_scaling(
                    self.circuit_function(), noise_factor
                )
            elif noise_amplification_method == 'identity':
                noisy_circuit_func = lambda: self.apply_identity_insertion(
                    self.circuit_function(), noise_factor
                )
            else:
                noisy_circuit_func = self.circuit_function  # No amplification for baseline
            
            # Temporarily replace circuit function
            original_func = self.circuit_function
            self.circuit_function = noisy_circuit_func
            
            # Measure expectation value
            expectation = self.measure_expectation_value(observable, num_shots)
            
            # Restore original function
            self.circuit_function = original_func
            
            noise_levels.append(noise_factor)
            expectation_values.append(expectation)
            
            print(f"     ⟨{observable}⟩(λ={noise_factor}) = {expectation:.6f}")
        
        # Store measurements
        self.measurements[observable] = {
            'noise_levels': noise_levels,
            'expectation_values': expectation_values
        }
        
        # Perform extrapolation
        extrapolated_result = self.extrapolate_to_zero_noise(
            noise_levels, expectation_values
        )
        
        print(f"\n✅ ZNE Results:")
        print(f"   Zero-noise extrapolated value: {extrapolated_result['value']:.6f}")
        print(f"   Extrapolation error estimate: ±{extrapolated_result['error']:.6f}")
        print(f"   Fit quality (R²): {extrapolated_result['r_squared']:.4f}")
        
        return {
            'observable': observable,
            'noise_levels': noise_levels,
            'measured_values': expectation_values,
            'extrapolated_value': extrapolated_result['value'],
            'extrapolation_error': extrapolated_result['error'],
            'fit_parameters': extrapolated_result['parameters'],
            'r_squared': extrapolated_result['r_squared']
        }
    
    def extrapolate_to_zero_noise(self, noise_levels: List[float], 
                                 expectation_values: List[float],
                                 extrapolation_type: str = 'exponential') -> Dict[str, Any]:
        """
        Perform extrapolation to zero noise.
        
        Args:
            noise_levels: List of noise amplification factors
            expectation_values: Corresponding expectation values
            extrapolation_type: 'linear', 'exponential', or 'polynomial'
            
        Returns:
            Extrapolation results
        """
        x = np.array(noise_levels)
        y = np.array(expectation_values)
        
        if extrapolation_type == 'linear':
            # Linear fit: y = a + b*x
            def fit_func(x, a, b):
                return a + b * x
            
            popt, pcov = curve_fit(fit_func, x, y)
            zero_noise_value = fit_func(0, *popt)
            
        elif extrapolation_type == 'exponential':
            # Exponential fit: y = A + B*r^x
            def fit_func(x, A, B, r):
                return A + B * (r ** x)
            
            # Initial guess
            p0 = [y[-1], y[0] - y[-1], 0.9]
            
            try:
                popt, pcov = curve_fit(fit_func, x, y, p0=p0, maxfev=2000)
                zero_noise_value = fit_func(0, *popt)
            except:
                # Fall back to linear if exponential fit fails
                print("   Exponential fit failed, using linear extrapolation")
                return self.extrapolate_to_zero_noise(noise_levels, expectation_values, 'linear')
        
        elif extrapolation_type == 'polynomial':
            # Polynomial fit
            degree = min(2, len(x) - 1)  # Use degree 2 or less
            popt = np.polyfit(x, y, degree)
            zero_noise_value = np.polyval(popt, 0)
            pcov = None  # Polynomial fit doesn't provide covariance directly
        
        else:
            raise ValueError(f"Unknown extrapolation type: {extrapolation_type}")
        
        # Calculate error estimate
        if pcov is not None:
            # Use parameter uncertainties to estimate error
            param_errors = np.sqrt(np.diag(pcov))
            
            if extrapolation_type == 'linear':
                error_estimate = param_errors[0]  # Error in intercept
            elif extrapolation_type == 'exponential':
                # Propagate uncertainty through function
                error_estimate = np.sqrt(param_errors[0]**2 + param_errors[1]**2)
            else:
                error_estimate = 0.1 * abs(zero_noise_value)  # 10% estimate
        else:
            error_estimate = 0.1 * abs(zero_noise_value)
        
        # Calculate R-squared
        if extrapolation_type == 'linear':
            y_pred = fit_func(x, *popt)
        elif extrapolation_type == 'exponential':
            y_pred = fit_func(x, *popt)
        else:
            y_pred = np.polyval(popt, x)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'value': zero_noise_value,
            'error': error_estimate,
            'parameters': popt,
            'r_squared': r_squared,
            'fitted_values': y_pred
        }
    
    def plot_extrapolation(self, observable: str, save_figure: bool = False):
        """Plot ZNE extrapolation results."""
        
        if observable not in self.measurements:
            print(f"No measurements found for observable {observable}")
            return
        
        data = self.measurements[observable]
        noise_levels = np.array(data['noise_levels'])
        expectation_values = np.array(data['expectation_values'])
        
        plt.figure(figsize=(10, 6))
        
        # Plot measured data points
        plt.scatter(noise_levels, expectation_values, 
                   color='red', s=100, zorder=5, label='Measured values')
        
        # Plot extrapolation
        x_fine = np.linspace(0, max(noise_levels), 100)
        
        # Get fit parameters
        result = self.extrapolate_to_zero_noise(noise_levels, expectation_values)
        
        # Plot different extrapolation types
        for extrap_type, color in [('linear', 'blue'), ('exponential', 'green')]:
            try:
                extrap_result = self.extrapolate_to_zero_noise(
                    noise_levels, expectation_values, extrap_type
                )
                
                if extrap_type == 'linear':
                    a, b = extrap_result['parameters']
                    y_fine = a + b * x_fine
                elif extrap_type == 'exponential':
                    A, B, r = extrap_result['parameters']
                    y_fine = A + B * (r ** x_fine)
                
                plt.plot(x_fine, y_fine, color=color, linestyle='--', 
                        label=f'{extrap_type.title()} extrapolation')
                
                # Mark zero-noise point
                plt.scatter(0, extrap_result['value'], color=color, 
                           s=150, marker='*', zorder=6,
                           label=f'ZNE value ({extrap_type}): {extrap_result["value"]:.3f}')
                
            except Exception as e:
                print(f"Could not plot {extrap_type} extrapolation: {e}")
        
        # Mark ideal value if known
        ideal_value = self._get_ideal_value(observable)
        if ideal_value is not None:
            plt.axhline(y=ideal_value, color='black', linestyle='-', alpha=0.7,
                       label=f'Ideal value: {ideal_value:.3f}')
        
        plt.xlabel('Noise Amplification Factor (λ)', fontsize=12)
        plt.ylabel(f'Expectation Value ⟨{observable}⟩', fontsize=12)
        plt.title('Zero-Noise Extrapolation', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.5, max(noise_levels) + 0.5)
        
        if save_figure:
            plt.savefig(f'zne_{observable.replace("*", "_")}.png', dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def _get_ideal_value(self, observable: str) -> float:
        """Get ideal (noiseless) value for comparison if known."""
        
        # This would depend on the specific circuit and observable
        # For demonstration, return None (unknown ideal value)
        return None

# Example quantum circuits for ZNE demonstration
def create_sample_circuits():
    """Create sample quantum circuits for ZNE testing."""
    
    circuits = {}
    
    # Bell state circuit
    def bell_state_circuit():
        circuit = quantrs2.Circuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        return circuit
    
    circuits['bell_state'] = {
        'function': bell_state_circuit,
        'observables': ['Z0', 'Z1', 'Z0*Z1'],
        'ideal_values': {'Z0': 0.0, 'Z1': 0.0, 'Z0*Z1': 1.0}
    }
    
    # Parameterized rotation circuit
    def rotation_circuit(theta=np.pi/4):
        circuit = quantrs2.Circuit(1)
        circuit.ry(0, theta)
        return circuit
    
    circuits['rotation'] = {
        'function': lambda: rotation_circuit(np.pi/4),
        'observables': ['Z0', 'X0'],
        'ideal_values': {'Z0': np.cos(np.pi/4), 'X0': np.sin(np.pi/4)}
    }
    
    # GHZ state circuit
    def ghz_state_circuit():
        circuit = quantrs2.Circuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        return circuit
    
    circuits['ghz_state'] = {
        'function': ghz_state_circuit,
        'observables': ['Z0', 'Z1', 'Z2', 'Z0*Z1*Z2'],
        'ideal_values': {'Z0': 0.0, 'Z1': 0.0, 'Z2': 0.0, 'Z0*Z1*Z2': 1.0}
    }
    
    return circuits

# Demonstrate ZNE on sample circuits
def demonstrate_basic_zne():
    """Demonstrate basic ZNE on sample quantum circuits."""
    
    print("🌟 Zero-Noise Extrapolation Demonstration")
    print("=" * 50)
    
    # Create sample circuits
    sample_circuits = create_sample_circuits()
    
    # Test ZNE on Bell state
    print("\n🔔 Bell State ZNE Example")
    bell_circuit = sample_circuits['bell_state']['function']
    
    # Create ZNE instance
    zne = ZeroNoiseExtrapolation(
        circuit_function=bell_circuit,
        noise_factors=[1, 3, 5, 7]
    )
    
    # Run ZNE experiment for Z0*Z1 observable
    result = zne.run_zne_experiment('Z0*Z1', num_shots=1000)
    
    # Plot results
    zne.plot_extrapolation('Z0*Z1')
    
    # Compare with ideal value
    ideal_value = sample_circuits['bell_state']['ideal_values']['Z0*Z1']
    extrapolated_value = result['extrapolated_value']
    improvement = abs(extrapolated_value - ideal_value) - abs(result['measured_values'][0] - ideal_value)
    
    print(f"\n📊 Bell State Results:")
    print(f"   Ideal value: {ideal_value:.3f}")
    print(f"   Noisy measurement: {result['measured_values'][0]:.3f}")
    print(f"   ZNE extrapolated: {extrapolated_value:.3f}")
    print(f"   Error improvement: {improvement:.3f}")
    
    return zne, result

# Run basic ZNE demonstration
zne_demo, demo_result = demonstrate_basic_zne()
```

### Advanced ZNE Techniques

```python
def adaptive_noise_scaling():
    """Implement adaptive noise scaling for improved ZNE."""
    
    print("\n🔄 Adaptive Noise Scaling")
    print("=" * 30)
    
    class AdaptiveZNE(ZeroNoiseExtrapolation):
        """ZNE with adaptive noise factor selection."""
        
        def __init__(self, circuit_function, initial_factors=None):
            super().__init__(circuit_function, initial_factors or [1, 3])
            self.convergence_threshold = 0.001
            self.max_factors = 10
        
        def adaptive_extrapolation(self, observable: str, num_shots: int = 1000):
            """Adaptively select noise factors for better extrapolation."""
            
            print(f"🔄 Adaptive ZNE for {observable}")
            
            # Start with initial measurements
            noise_levels = list(self.noise_factors)
            expectation_values = []
            
            for factor in noise_levels:
                expectation = self._measure_at_noise_level(observable, factor, num_shots)
                expectation_values.append(expectation)
                print(f"   λ = {factor}: ⟨{observable}⟩ = {expectation:.6f}")
            
            # Iteratively add noise factors until convergence
            previous_extrapolation = None
            
            while len(noise_levels) < self.max_factors:
                # Perform current extrapolation
                extrap_result = self.extrapolate_to_zero_noise(noise_levels, expectation_values)
                current_extrapolation = extrap_result['value']
                
                print(f"   Current extrapolation: {current_extrapolation:.6f}")
                
                # Check convergence
                if previous_extrapolation is not None:
                    change = abs(current_extrapolation - previous_extrapolation)
                    print(f"   Change from previous: {change:.6f}")
                    
                    if change < self.convergence_threshold:
                        print(f"   ✅ Converged!")
                        break
                
                # Add next noise factor
                next_factor = max(noise_levels) + 2  # Increment by 2 (keep odd)
                noise_levels.append(next_factor)
                
                # Measure at new noise level
                expectation = self._measure_at_noise_level(observable, next_factor, num_shots)
                expectation_values.append(expectation)
                
                print(f"   Added λ = {next_factor}: ⟨{observable}⟩ = {expectation:.6f}")
                
                previous_extrapolation = current_extrapolation
            
            # Final extrapolation
            final_result = self.extrapolate_to_zero_noise(noise_levels, expectation_values)
            
            print(f"\n📊 Adaptive ZNE Results:")
            print(f"   Noise factors used: {noise_levels}")
            print(f"   Final extrapolation: {final_result['value']:.6f}")
            print(f"   R²: {final_result['r_squared']:.4f}")
            
            return {
                'noise_levels': noise_levels,
                'expectation_values': expectation_values,
                'extrapolated_value': final_result['value'],
                'convergence_achieved': len(noise_levels) < self.max_factors
            }
        
        def _measure_at_noise_level(self, observable: str, noise_factor: float, num_shots: int):
            """Measure expectation value at specific noise level."""
            
            # Create noisy circuit
            noisy_circuit_func = lambda: self.apply_digital_noise_scaling(
                self.circuit_function(), noise_factor
            )
            
            # Temporarily replace circuit function
            original_func = self.circuit_function
            self.circuit_function = noisy_circuit_func
            
            # Measure
            expectation = self.measure_expectation_value(observable, num_shots)
            
            # Restore original function
            self.circuit_function = original_func
            
            return expectation
    
    # Demonstrate adaptive ZNE
    bell_circuit = create_sample_circuits()['bell_state']['function']
    adaptive_zne = AdaptiveZNE(bell_circuit)
    
    result = adaptive_zne.adaptive_extrapolation('Z0*Z1', num_shots=500)
    
    return adaptive_zne, result

def richardson_extrapolation():
    """Implement Richardson extrapolation for ZNE."""
    
    print("\n📐 Richardson Extrapolation")
    print("=" * 30)
    
    def richardson_zne(noise_levels, expectation_values, order=1):
        """
        Apply Richardson extrapolation for improved accuracy.
        
        Richardson extrapolation assumes: f(h) = f(0) + c₁h^p + c₂h^(2p) + ...
        """
        
        print(f"Applying Richardson extrapolation (order {order})")
        
        x = np.array(noise_levels)
        y = np.array(expectation_values)
        
        # Richardson extrapolation formula for two points
        if len(x) >= 2:
            h1, h2 = x[0], x[1]
            f1, f2 = y[0], y[1]
            
            # Assume p = 1 (linear noise scaling)
            p = order
            richardson_value = (f1 * h2**p - f2 * h1**p) / (h2**p - h1**p)
            
            print(f"   Richardson value: {richardson_value:.6f}")
            print(f"   Standard linear: {f1 + (f2 - f1) * h1 / (h2 - h1):.6f}")
            
            return richardson_value
        
        return None
    
    # Test Richardson extrapolation
    noise_levels = [1, 3, 5]
    expectation_values = [0.95, 0.85, 0.75]  # Example decreasing values
    
    richardson_result = richardson_zne(noise_levels, expectation_values)
    
    return richardson_result

def multi_observable_zne():
    """Implement ZNE for multiple observables simultaneously."""
    
    print("\n🎯 Multi-Observable ZNE")
    print("=" * 30)
    
    class MultiObservableZNE(ZeroNoiseExtrapolation):
        """ZNE for multiple observables with correlation analysis."""
        
        def run_multi_observable_experiment(self, observables: List[str], 
                                          num_shots: int = 1000):
            """Run ZNE for multiple observables."""
            
            print(f"Running ZNE for {len(observables)} observables: {observables}")
            
            results = {}
            all_noise_levels = []
            all_measurements = {}
            
            # Initialize measurement storage
            for obs in observables:
                all_measurements[obs] = []
            
            # Measure all observables at each noise level
            for noise_factor in self.noise_factors:
                print(f"\n📊 Noise factor λ = {noise_factor}")
                
                all_noise_levels.append(noise_factor)
                
                for observable in observables:
                    # Create noisy circuit
                    noisy_circuit_func = lambda: self.apply_digital_noise_scaling(
                        self.circuit_function(), noise_factor
                    )
                    
                    # Measure
                    original_func = self.circuit_function
                    self.circuit_function = noisy_circuit_func
                    expectation = self.measure_expectation_value(observable, num_shots)
                    self.circuit_function = original_func
                    
                    all_measurements[observable].append(expectation)
                    print(f"   ⟨{observable}⟩ = {expectation:.6f}")
            
            # Extrapolate each observable
            for observable in observables:
                expectation_values = all_measurements[observable]
                extrap_result = self.extrapolate_to_zero_noise(
                    all_noise_levels, expectation_values
                )
                
                results[observable] = {
                    'measurements': expectation_values,
                    'extrapolated_value': extrap_result['value'],
                    'error': extrap_result['error'],
                    'r_squared': extrap_result['r_squared']
                }
            
            # Analyze correlations between observables
            correlation_analysis = self._analyze_observable_correlations(
                all_measurements, observables
            )
            
            return {
                'results': results,
                'correlations': correlation_analysis,
                'noise_levels': all_noise_levels
            }
        
        def _analyze_observable_correlations(self, measurements, observables):
            """Analyze correlations between different observables under noise."""
            
            print(f"\n🔗 Observable Correlation Analysis")
            
            # Create correlation matrix
            measurement_matrix = np.array([measurements[obs] for obs in observables])
            correlation_matrix = np.corrcoef(measurement_matrix)
            
            print(f"Correlation matrix:")
            print(f"{'':>15}", end="")
            for obs in observables:
                print(f"{obs:>10}", end="")
            print()
            
            for i, obs1 in enumerate(observables):
                print(f"{obs1:>15}", end="")
                for j, obs2 in enumerate(observables):
                    print(f"{correlation_matrix[i,j]:>10.3f}", end="")
                print()
            
            # Find highly correlated pairs
            high_correlations = []
            for i in range(len(observables)):
                for j in range(i+1, len(observables)):
                    corr = correlation_matrix[i, j]
                    if abs(corr) > 0.8:
                        high_correlations.append({
                            'obs1': observables[i],
                            'obs2': observables[j],
                            'correlation': corr
                        })
            
            if high_correlations:
                print(f"\nHighly correlated observable pairs (|r| > 0.8):")
                for pair in high_correlations:
                    print(f"   {pair['obs1']} ↔ {pair['obs2']}: r = {pair['correlation']:.3f}")
            
            return {
                'correlation_matrix': correlation_matrix,
                'high_correlations': high_correlations
            }
    
    # Demonstrate multi-observable ZNE
    ghz_circuit = create_sample_circuits()['ghz_state']['function']
    multi_zne = MultiObservableZNE(ghz_circuit, noise_factors=[1, 3, 5])
    
    observables = ['Z0', 'Z1', 'Z2', 'Z0*Z1', 'Z1*Z2', 'Z0*Z2']
    multi_result = multi_zne.run_multi_observable_experiment(observables, num_shots=500)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot extrapolations for each observable
    for i, obs in enumerate(observables):
        plt.subplot(2, 3, i + 1)
        
        noise_levels = multi_result['noise_levels']
        measurements = multi_result['results'][obs]['measurements']
        extrapolated = multi_result['results'][obs]['extrapolated_value']
        
        # Plot measurements
        plt.scatter(noise_levels, measurements, color='red', s=50)
        
        # Plot extrapolation line
        x_line = np.linspace(0, max(noise_levels), 100)
        # Simple linear extrapolation for visualization
        slope = (measurements[-1] - measurements[0]) / (noise_levels[-1] - noise_levels[0])
        y_line = measurements[0] + slope * (x_line - noise_levels[0])
        plt.plot(x_line, y_line, 'b--', alpha=0.7)
        
        # Mark extrapolated value
        plt.scatter(0, extrapolated, color='blue', s=100, marker='*')
        
        plt.title(f'⟨{obs}⟩')
        plt.xlabel('Noise Factor')
        plt.ylabel('Expectation Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return multi_zne, multi_result

# Run advanced ZNE techniques
adaptive_result = adaptive_noise_scaling()
richardson_result = richardson_extrapolation()
multi_observable_result = multi_observable_zne()
```

### ZNE for Variational Algorithms

```python
def zne_for_vqe():
    """Apply ZNE to Variational Quantum Eigensolver."""
    
    print("\n⚗️ ZNE for Variational Quantum Eigensolver")
    print("=" * 45)
    
    class VQE_with_ZNE:
        """VQE implementation with ZNE error mitigation."""
        
        def __init__(self, hamiltonian_terms, num_qubits):
            """
            Initialize VQE with ZNE.
            
            Args:
                hamiltonian_terms: List of (coefficient, pauli_string) pairs
                num_qubits: Number of qubits
            """
            self.hamiltonian_terms = hamiltonian_terms
            self.num_qubits = num_qubits
            self.parameters = np.random.uniform(0, 2*np.pi, 4)  # Simple ansatz
            
            print(f"🧪 VQE with ZNE:")
            print(f"   Qubits: {num_qubits}")
            print(f"   Hamiltonian terms: {len(hamiltonian_terms)}")
            print(f"   Parameters: {len(self.parameters)}")
        
        def create_ansatz_circuit(self, parameters):
            """Create variational ansatz circuit."""
            
            circuit = quantrs2.Circuit(self.num_qubits)
            
            # Simple hardware-efficient ansatz
            for i in range(self.num_qubits):
                circuit.ry(i, parameters[i % len(parameters)])
            
            for i in range(self.num_qubits - 1):
                circuit.cx(i, i + 1)
            
            return circuit
        
        def compute_energy_with_zne(self, parameters, noise_factors=None, num_shots=1000):
            """Compute energy expectation value using ZNE."""
            
            if noise_factors is None:
                noise_factors = [1, 3, 5]
            
            print(f"Computing energy with ZNE (noise factors: {noise_factors})")
            
            total_energy = 0
            
            # Process each Hamiltonian term
            for coeff, pauli_string in self.hamiltonian_terms:
                print(f"   Processing term: {coeff:.3f} * {pauli_string}")
                
                # Create circuit function for this measurement
                def circuit_func():
                    return self.create_ansatz_circuit(parameters)
                
                # Apply ZNE to this term
                zne = ZeroNoiseExtrapolation(circuit_func, noise_factors)
                
                # Convert Pauli string to observable format
                observable = self._pauli_to_observable(pauli_string)
                
                # Run ZNE
                zne_result = zne.run_zne_experiment(observable, num_shots)
                mitigated_expectation = zne_result['extrapolated_value']
                
                # Add to total energy
                term_energy = coeff * mitigated_expectation
                total_energy += term_energy
                
                print(f"     Raw measurement: {zne_result['measured_values'][0]:.6f}")
                print(f"     ZNE mitigated: {mitigated_expectation:.6f}")
                print(f"     Term contribution: {term_energy:.6f}")
            
            print(f"   Total mitigated energy: {total_energy:.6f}")
            return total_energy
        
        def _pauli_to_observable(self, pauli_string):
            """Convert Pauli string to observable format."""
            
            # Convert 'IZ' to 'Z1', 'ZI' to 'Z0', 'ZZ' to 'Z0*Z1', etc.
            observable_parts = []
            
            for i, pauli in enumerate(pauli_string):
                if pauli != 'I':
                    observable_parts.append(f'{pauli}{i}')
            
            if not observable_parts:
                return 'I'  # Identity
            
            return '*'.join(observable_parts)
        
        def optimize_with_zne(self, max_iterations=20):
            """Optimize VQE parameters with ZNE energy evaluation."""
            
            print(f"\n🚀 VQE Optimization with ZNE")
            
            energy_history = []
            
            for iteration in range(max_iterations):
                # Compute energy with ZNE (reduced shots for speed)
                energy = self.compute_energy_with_zne(self.parameters, num_shots=200)
                energy_history.append(energy)
                
                print(f"Iteration {iteration + 1}: Energy = {energy:.6f}")
                
                # Simple parameter update (gradient-free)
                if iteration < max_iterations - 1:
                    # Random parameter perturbation for demonstration
                    self.parameters += np.random.normal(0, 0.1, len(self.parameters))
                    self.parameters = self.parameters % (2 * np.pi)  # Keep in [0, 2π]
            
            print(f"\n✅ VQE Optimization completed")
            print(f"   Final energy: {energy_history[-1]:.6f}")
            print(f"   Energy change: {energy_history[-1] - energy_history[0]:.6f}")
            
            return energy_history
    
    # Example: H2 molecule Hamiltonian (simplified)
    h2_hamiltonian = [
        (-1.0523732, 'II'),  # Constant term
        (-0.3979374, 'ZI'),  # Z on qubit 0
        (-0.3979374, 'IZ'),  # Z on qubit 1
        (-0.0112801, 'ZZ'),  # ZZ interaction
        (-0.0112801, 'XX'),  # XX interaction
    ]
    
    # Create VQE instance with ZNE
    vqe_zne = VQE_with_ZNE(h2_hamiltonian, num_qubits=2)
    
    # Run optimization
    energy_history = vqe_zne.optimize_with_zne(max_iterations=5)  # Reduced for demo
    
    # Plot energy evolution
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(energy_history) + 1), energy_history, 'bo-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.title('VQE Energy Evolution with ZNE')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return vqe_zne, energy_history

def zne_for_qaoa():
    """Apply ZNE to Quantum Approximate Optimization Algorithm."""
    
    print("\n🎯 ZNE for Quantum Approximate Optimization Algorithm")
    print("=" * 55)
    
    class QAOA_with_ZNE:
        """QAOA implementation with ZNE error mitigation."""
        
        def __init__(self, problem_hamiltonian, num_qubits, num_layers=1):
            self.problem_hamiltonian = problem_hamiltonian
            self.num_qubits = num_qubits
            self.num_layers = num_layers
            
            # QAOA parameters: [beta_1, gamma_1, beta_2, gamma_2, ...]
            self.parameters = np.random.uniform(0, 2*np.pi, 2 * num_layers)
            
            print(f"🎯 QAOA with ZNE:")
            print(f"   Qubits: {num_qubits}")
            print(f"   Layers: {num_layers}")
            print(f"   Parameters: {len(self.parameters)}")
        
        def create_qaoa_circuit(self, parameters):
            """Create QAOA circuit."""
            
            circuit = quantrs2.Circuit(self.num_qubits)
            
            # Initial superposition
            for qubit in range(self.num_qubits):
                circuit.h(qubit)
            
            # QAOA layers
            for layer in range(self.num_layers):
                gamma = parameters[2 * layer]
                beta = parameters[2 * layer + 1]
                
                # Problem unitary (simplified)
                for i in range(self.num_qubits):
                    circuit.rz(i, gamma)
                
                for i in range(self.num_qubits - 1):
                    circuit.cx(i, i + 1)
                    circuit.rz(i + 1, gamma)
                    circuit.cx(i, i + 1)
                
                # Mixing unitary
                for qubit in range(self.num_qubits):
                    circuit.rx(qubit, 2 * beta)
            
            return circuit
        
        def compute_cost_with_zne(self, parameters, noise_factors=None, num_shots=1000):
            """Compute QAOA cost function using ZNE."""
            
            if noise_factors is None:
                noise_factors = [1, 3, 5]
            
            # Create circuit function
            def circuit_func():
                return self.create_qaoa_circuit(parameters)
            
            # Apply ZNE
            zne = ZeroNoiseExtrapolation(circuit_func, noise_factors)
            
            # For simplicity, measure just Z0 (real QAOA would measure problem Hamiltonian)
            zne_result = zne.run_zne_experiment('Z0', num_shots)
            
            return -zne_result['extrapolated_value']  # Negative for minimization
        
        def optimize_qaoa_with_zne(self, max_iterations=10):
            """Optimize QAOA parameters with ZNE."""
            
            print(f"\n🚀 QAOA Optimization with ZNE")
            
            cost_history = []
            
            for iteration in range(max_iterations):
                # Compute cost with ZNE
                cost = self.compute_cost_with_zne(self.parameters, num_shots=200)
                cost_history.append(cost)
                
                print(f"Iteration {iteration + 1}: Cost = {cost:.6f}")
                
                # Simple parameter update
                if iteration < max_iterations - 1:
                    self.parameters += np.random.normal(0, 0.1, len(self.parameters))
                    self.parameters = self.parameters % (2 * np.pi)
            
            return cost_history
    
    # Create simple Max-Cut problem
    problem_hamiltonian = "Simple Max-Cut on 3 qubits"
    
    qaoa_zne = QAOA_with_ZNE(problem_hamiltonian, num_qubits=3, num_layers=2)
    cost_history = qaoa_zne.optimize_qaoa_with_zne(max_iterations=5)
    
    return qaoa_zne, cost_history

# Run ZNE for variational algorithms
vqe_results = zne_for_vqe()
qaoa_results = zne_for_qaoa()
```

### Performance Analysis and Benchmarking

```python
def comprehensive_zne_benchmark():
    """Comprehensive ZNE performance benchmark."""
    
    print("\n🏆 Comprehensive ZNE Benchmark")
    print("=" * 40)
    
    import time
    
    # Test different circuit types and noise levels
    test_circuits = create_sample_circuits()
    noise_methods = ['digital', 'identity']
    extrapolation_types = ['linear', 'exponential']
    
    benchmark_results = []
    
    for circuit_name, circuit_data in test_circuits.items():
        print(f"\n--- Benchmarking {circuit_name} ---")
        
        circuit_func = circuit_data['function']
        observables = circuit_data['observables'][:2]  # Test first 2 observables
        
        for observable in observables:
            for noise_method in noise_methods:
                for extrap_type in extrapolation_types:
                    
                    print(f"  {observable} | {noise_method} | {extrap_type}")
                    
                    start_time = time.time()
                    
                    try:
                        # Create ZNE instance
                        zne = ZeroNoiseExtrapolation(circuit_func, [1, 3, 5])
                        
                        # Run experiment
                        result = zne.run_zne_experiment(
                            observable, 
                            num_shots=200,  # Reduced for benchmarking
                            noise_amplification_method=noise_method
                        )
                        
                        # Perform specific extrapolation
                        extrap_result = zne.extrapolate_to_zero_noise(
                            result['noise_levels'],
                            result['measured_values'],
                            extrap_type
                        )
                        
                        execution_time = time.time() - start_time
                        
                        # Calculate accuracy if ideal value known
                        ideal_value = circuit_data['ideal_values'].get(observable)
                        accuracy = None
                        if ideal_value is not None:
                            raw_error = abs(result['measured_values'][0] - ideal_value)
                            zne_error = abs(extrap_result['value'] - ideal_value)
                            accuracy = (raw_error - zne_error) / raw_error if raw_error > 0 else 0
                        
                        benchmark_results.append({
                            'circuit': circuit_name,
                            'observable': observable,
                            'noise_method': noise_method,
                            'extrapolation': extrap_type,
                            'execution_time': execution_time,
                            'r_squared': extrap_result['r_squared'],
                            'accuracy_improvement': accuracy,
                            'success': True
                        })
                        
                        print(f"    Time: {execution_time:.2f}s, R²: {extrap_result['r_squared']:.3f}")
                        
                    except Exception as e:
                        print(f"    Failed: {e}")
                        benchmark_results.append({
                            'circuit': circuit_name,
                            'observable': observable,
                            'noise_method': noise_method,
                            'extrapolation': extrap_type,
                            'success': False
                        })
    
    # Analyze results
    print(f"\n📊 ZNE Benchmark Summary:")
    print(f"{'Circuit':<12} {'Observable':<10} {'Method':<10} {'Extrap':<12} {'Time (s)':<9} {'R²':<6} {'Accuracy'}")
    print("-" * 85)
    
    successful_results = [r for r in benchmark_results if r['success']]
    
    for result in successful_results:
        accuracy_str = f"{result['accuracy_improvement']:.2f}" if result['accuracy_improvement'] is not None else "N/A"
        
        print(f"{result['circuit']:<12} {result['observable']:<10} {result['noise_method']:<10} "
              f"{result['extrapolation']:<12} {result['execution_time']:<9.2f} "
              f"{result['r_squared']:<6.3f} {accuracy_str}")
    
    # Performance analysis
    print(f"\nPerformance Analysis:")
    
    # Average performance by method
    methods_performance = {}
    for method in noise_methods:
        method_results = [r for r in successful_results if r['noise_method'] == method]
        if method_results:
            avg_time = np.mean([r['execution_time'] for r in method_results])
            avg_r2 = np.mean([r['r_squared'] for r in method_results])
            methods_performance[method] = {'time': avg_time, 'r2': avg_r2}
    
    print(f"Average performance by noise amplification method:")
    for method, perf in methods_performance.items():
        print(f"  {method:<10}: Time = {perf['time']:.2f}s, R² = {perf['r2']:.3f}")
    
    # Extrapolation comparison
    extrap_performance = {}
    for extrap in extrapolation_types:
        extrap_results = [r for r in successful_results if r['extrapolation'] == extrap]
        if extrap_results:
            avg_r2 = np.mean([r['r_squared'] for r in extrap_results])
            extrap_performance[extrap] = avg_r2
    
    print(f"\nAverage fit quality by extrapolation type:")
    for extrap, r2 in extrap_performance.items():
        print(f"  {extrap:<12}: R² = {r2:.3f}")
    
    return benchmark_results

def zne_noise_resilience_study():
    """Study ZNE performance under different noise conditions."""
    
    print("\n🔊 ZNE Noise Resilience Study")
    print("=" * 35)
    
    # Simulate different noise scenarios
    noise_scenarios = {
        'low_noise': {'base_error_rate': 0.01, 'description': 'High-fidelity device'},
        'medium_noise': {'base_error_rate': 0.05, 'description': 'Typical NISQ device'},
        'high_noise': {'base_error_rate': 0.10, 'description': 'Noisy prototype device'}
    }
    
    # Test circuit
    bell_circuit = create_sample_circuits()['bell_state']['function']
    
    resilience_results = {}
    
    for scenario_name, scenario_data in noise_scenarios.items():
        print(f"\n📊 Testing {scenario_name} scenario:")
        print(f"   {scenario_data['description']}")
        print(f"   Base error rate: {scenario_data['base_error_rate']:.1%}")
        
        # Create ZNE with different noise factors
        noise_factors = [1, 3, 5, 7, 9]
        zne = ZeroNoiseExtrapolation(bell_circuit, noise_factors)
        
        # Simulate measurements with realistic noise
        simulated_measurements = []
        
        for factor in noise_factors:
            # Simulate noise effect: higher factor = more noise
            base_value = 1.0  # Ideal Z0*Z1 for Bell state
            noise_strength = scenario_data['base_error_rate'] * factor
            
            # Exponential decay model
            noisy_value = base_value * np.exp(-noise_strength)
            
            # Add measurement uncertainty
            measurement_noise = np.random.normal(0, 0.01)
            measured_value = noisy_value + measurement_noise
            
            simulated_measurements.append(measured_value)
            print(f"     λ = {factor}: ⟨Z0*Z1⟩ = {measured_value:.4f}")
        
        # Perform extrapolation
        extrap_result = zne.extrapolate_to_zero_noise(
            noise_factors, simulated_measurements, 'exponential'
        )
        
        # Calculate performance metrics
        ideal_value = 1.0
        raw_error = abs(simulated_measurements[0] - ideal_value)
        zne_error = abs(extrap_result['value'] - ideal_value)
        improvement_factor = raw_error / zne_error if zne_error > 0 else float('inf')
        
        resilience_results[scenario_name] = {
            'measurements': simulated_measurements,
            'extrapolated_value': extrap_result['value'],
            'raw_error': raw_error,
            'zne_error': zne_error,
            'improvement_factor': improvement_factor,
            'r_squared': extrap_result['r_squared']
        }
        
        print(f"   Raw measurement error: {raw_error:.4f}")
        print(f"   ZNE error: {zne_error:.4f}")
        print(f"   Improvement factor: {improvement_factor:.2f}x")
        print(f"   Fit quality (R²): {extrap_result['r_squared']:.3f}")
    
    # Visualize resilience study
    plt.figure(figsize=(15, 5))
    
    for i, (scenario_name, data) in enumerate(resilience_results.items()):
        plt.subplot(1, 3, i + 1)
        
        # Plot measurements vs noise factors
        plt.scatter(noise_factors, data['measurements'], color='red', s=50, label='Measurements')
        
        # Plot extrapolation
        x_extrap = np.linspace(0, max(noise_factors), 100)
        # Simplified exponential fit for visualization
        y_extrap = data['extrapolated_value'] * np.exp(-0.1 * x_extrap)
        plt.plot(x_extrap, y_extrap, 'b--', alpha=0.7, label='Extrapolation')
        
        # Mark ideal and extrapolated values
        plt.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Ideal value')
        plt.scatter(0, data['extrapolated_value'], color='blue', s=100, marker='*', 
                   label=f'ZNE: {data["extrapolated_value"]:.3f}')
        
        plt.title(f'{scenario_name.replace("_", " ").title()}')
        plt.xlabel('Noise Factor')
        plt.ylabel('⟨Z0*Z1⟩')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary comparison
    print(f"\n📈 Noise Resilience Summary:")
    print(f"{'Scenario':<15} {'Raw Error':<12} {'ZNE Error':<12} {'Improvement':<12} {'R²'}")
    print("-" * 65)
    
    for scenario, data in resilience_results.items():
        print(f"{scenario:<15} {data['raw_error']:<12.4f} {data['zne_error']:<12.4f} "
              f"{data['improvement_factor']:<12.2f} {data['r_squared']:<8.3f}")
    
    return resilience_results

# Run performance analysis
benchmark_data = comprehensive_zne_benchmark()
resilience_data = zne_noise_resilience_study()
```

## Exercises and Extensions

### Exercise 1: Advanced Extrapolation Methods
```python
def exercise_advanced_extrapolation():
    """Exercise: Implement advanced extrapolation methods."""
    
    print("🎯 Exercise: Advanced Extrapolation Methods")
    print("=" * 45)
    
    # TODO: Implement advanced extrapolation techniques:
    # 1. Rational function extrapolation (Padé approximants)
    # 2. Non-linear least squares fitting
    # 3. Machine learning-based extrapolation
    # 4. Bayesian extrapolation with uncertainty quantification
    
    print("Your challenge:")
    print("1. Implement Padé approximant extrapolation")
    print("2. Use ML models (neural networks, Gaussian processes)")
    print("3. Add uncertainty quantification to extrapolations")
    print("4. Compare different methods on various noise models")

exercise_advanced_extrapolation()
```

### Exercise 2: Hardware-Specific ZNE
```python
def exercise_hardware_specific_zne():
    """Exercise: Adapt ZNE for specific quantum hardware."""
    
    print("🎯 Exercise: Hardware-Specific ZNE")
    print("=" * 35)
    
    # TODO: Implement hardware-specific adaptations:
    # 1. Gate set restrictions for different devices
    # 2. Connectivity constraints for noise amplification
    # 3. Device calibration data integration
    # 4. Real-time noise tracking and adaptation
    
    print("Adapt ZNE for real quantum hardware:")
    print("1. Handle limited gate sets (IBM, Google, IonQ)")
    print("2. Work with device topology constraints")
    print("3. Use calibration data for better noise models")
    print("4. Implement real-time noise estimation")

exercise_hardware_specific_zne()
```

### Exercise 3: Composite Error Mitigation
```python
def exercise_composite_mitigation():
    """Exercise: Combine ZNE with other error mitigation techniques."""
    
    print("🎯 Exercise: Composite Error Mitigation")
    print("=" * 40)
    
    # TODO: Combine ZNE with other techniques:
    # 1. ZNE + Readout error mitigation
    # 2. ZNE + Dynamical decoupling
    # 3. ZNE + Symmetry verification
    # 4. Adaptive switching between mitigation methods
    
    print("Combine multiple error mitigation techniques:")
    print("1. ZNE + readout error correction")
    print("2. ZNE + dynamical decoupling sequences")
    print("3. ZNE + symmetry-based error detection")
    print("4. Adaptive method selection based on circuit properties")

exercise_composite_mitigation()
```

## Summary

🎉 **Congratulations!** You've learned:
- The theory and implementation of Zero-Noise Extrapolation
- Different noise amplification methods (digital, identity insertion)
- Various extrapolation functions (linear, exponential, polynomial)
- Advanced techniques: adaptive scaling, Richardson extrapolation, multi-observable ZNE
- Applications to variational algorithms (VQE, QAOA)
- Performance analysis and noise resilience studies

ZNE is one of the most practical error mitigation techniques for near-term quantum devices, offering significant improvements with minimal overhead!

**Next Steps:**
- Explore [Readout Error Mitigation](readout_correction.md)
- Try [Dynamical Decoupling](dynamical_decoupling.md)
- Learn about [Quantum Error Correction](../research/error_correction.md)

## References

### Foundational Papers
- Temme et al. (2017). "Error mitigation for short-depth quantum circuits"
- Li & Benjamin (2017). "Efficient variational quantum simulator incorporating active error minimization"
- Kandala et al. (2019). "Error mitigation extends the computational reach of a noisy quantum processor"

### Advanced Techniques
- Giurgica-Tiron et al. (2020). "Digital zero noise extrapolation for quantum error mitigation"
- Strikis et al. (2021). "Learning-based quantum error mitigation"
- Piveteau & Sutter (2021). "Circuit knitting with classical communication"

### Reviews and Applications
- Endo et al. (2021). "Hybrid quantum-classical algorithms and quantum error mitigation"
- Cai et al. (2023). "Quantum error mitigation"

---

*"Error mitigation bridges the gap between today's noisy quantum devices and tomorrow's fault-tolerant quantum computers."* - Quantum Error Mitigation Researcher

🚀 **Ready to mitigate quantum noise?** Explore more [Error Mitigation Examples](index.md)!