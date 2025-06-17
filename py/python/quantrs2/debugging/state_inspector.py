"""
Quantum State Inspector for QuantRS2 Debugging.

This module provides advanced quantum state inspection and analysis capabilities
for debugging quantum circuits and understanding quantum state properties.
"""

import logging
import numpy as np
from typing import List

from .core import (
    DebugLevel, InspectionMode, StateInspectionResult
)

logger = logging.getLogger(__name__)

class QuantumStateInspector:
    """Advanced quantum state inspection and analysis."""
    
    def __init__(self, debug_level: DebugLevel = DebugLevel.INFO):
        self.debug_level = debug_level
        self.inspection_history = []
        self.anomaly_thresholds = {
            'amplitude_threshold': 1e-10,
            'phase_threshold': 1e-8,
            'entanglement_threshold': 0.1,
            'purity_threshold': 0.95
        }
    
    def inspect_state(self, state_vector: np.ndarray, mode: InspectionMode) -> StateInspectionResult:
        """Perform comprehensive state inspection based on the specified mode."""
        try:
            if mode == InspectionMode.AMPLITUDE_ANALYSIS:
                return self._analyze_amplitudes(state_vector)
            elif mode == InspectionMode.PROBABILITY_ANALYSIS:
                return self._analyze_probabilities(state_vector)
            elif mode == InspectionMode.PHASE_ANALYSIS:
                return self._analyze_phases(state_vector)
            elif mode == InspectionMode.ENTANGLEMENT_ANALYSIS:
                return self._analyze_entanglement(state_vector)
            elif mode == InspectionMode.COHERENCE_ANALYSIS:
                return self._analyze_coherence(state_vector)
            elif mode == InspectionMode.CORRELATION_ANALYSIS:
                return self._analyze_correlations(state_vector)
            elif mode == InspectionMode.PURITY_ANALYSIS:
                return self._analyze_purity(state_vector)
            elif mode == InspectionMode.FIDELITY_ANALYSIS:
                return self._analyze_fidelity(state_vector)
            else:
                raise ValueError(f"Unknown inspection mode: {mode}")
                
        except Exception as e:
            logger.error(f"State inspection failed: {e}")
            return StateInspectionResult(
                mode=mode,
                analysis_data={},
                insights=[],
                anomalies=[f"Inspection failed: {e}"],
                recommendations=["Check state vector validity", "Verify inspection parameters"]
            )
    
    def _analyze_amplitudes(self, state_vector: np.ndarray) -> StateInspectionResult:
        """Analyze state vector amplitudes for anomalies and patterns."""
        analysis_data = {}
        insights = []
        anomalies = []
        recommendations = []
        
        # Calculate amplitude statistics
        amplitudes = np.abs(state_vector)
        analysis_data['amplitude_stats'] = {
            'max': float(np.max(amplitudes)),
            'min': float(np.min(amplitudes)),
            'mean': float(np.mean(amplitudes)),
            'std': float(np.std(amplitudes)),
            'variance': float(np.var(amplitudes))
        }
        
        # Find dominant amplitudes
        threshold = np.max(amplitudes) * 0.1
        dominant_indices = np.where(amplitudes > threshold)[0]
        analysis_data['dominant_states'] = {
            'indices': dominant_indices.tolist(),
            'amplitudes': amplitudes[dominant_indices].tolist(),
            'basis_states': [format(i, f'0{int(np.log2(len(state_vector)))}b') for i in dominant_indices]
        }
        
        # Check for anomalies
        if np.any(amplitudes < self.anomaly_thresholds['amplitude_threshold']):
            anomalies.append("Very small amplitudes detected - possible numerical precision issues")
            recommendations.append("Consider using higher precision arithmetic")
        
        if len(dominant_indices) == 1:
            insights.append("State appears to be in a computational basis state")
        elif len(dominant_indices) == len(state_vector):
            insights.append("State shows uniform superposition")
        else:
            insights.append(f"State has {len(dominant_indices)} dominant components")
        
        # Check normalization
        norm = np.linalg.norm(state_vector)
        analysis_data['normalization'] = {
            'norm': float(norm),
            'normalized': abs(norm - 1.0) < 1e-10
        }
        
        if abs(norm - 1.0) > 1e-6:
            anomalies.append(f"State is not properly normalized (norm = {norm:.6f})")
            recommendations.append("Normalize the state vector")
        
        return StateInspectionResult(
            mode=InspectionMode.AMPLITUDE_ANALYSIS,
            analysis_data=analysis_data,
            insights=insights,
            anomalies=anomalies,
            recommendations=recommendations
        )
    
    def _analyze_probabilities(self, state_vector: np.ndarray) -> StateInspectionResult:
        """Analyze measurement probabilities and distributions."""
        analysis_data = {}
        insights = []
        anomalies = []
        recommendations = []
        
        probabilities = np.abs(state_vector) ** 2
        n_qubits = int(np.log2(len(state_vector)))
        
        analysis_data['probability_stats'] = {
            'max': float(np.max(probabilities)),
            'min': float(np.min(probabilities)),
            'entropy': float(-np.sum(probabilities * np.log2(probabilities + 1e-16))),
            'effective_dimension': float(1.0 / np.sum(probabilities ** 2))
        }
        
        # Analyze individual qubit probabilities
        qubit_probs = []
        for qubit in range(n_qubits):
            prob_0 = 0.0
            prob_1 = 0.0
            for i, prob in enumerate(probabilities):
                if (i >> qubit) & 1:
                    prob_1 += prob
                else:
                    prob_0 += prob
            qubit_probs.append({'qubit': qubit, 'prob_0': prob_0, 'prob_1': prob_1})
        
        analysis_data['qubit_probabilities'] = qubit_probs
        
        # Check for bias
        for i, qprob in enumerate(qubit_probs):
            if abs(qprob['prob_0'] - 0.5) > 0.4:
                if qprob['prob_0'] > 0.9:
                    insights.append(f"Qubit {i} is strongly biased toward |0⟩")
                elif qprob['prob_1'] > 0.9:
                    insights.append(f"Qubit {i} is strongly biased toward |1⟩")
                else:
                    insights.append(f"Qubit {i} shows significant bias")
        
        # Entropy analysis
        entropy = analysis_data['probability_stats']['entropy']
        max_entropy = n_qubits
        if entropy < max_entropy * 0.1:
            insights.append("Low entropy - state is highly localized")
        elif entropy > max_entropy * 0.9:
            insights.append("High entropy - state is highly delocalized")
        
        return StateInspectionResult(
            mode=InspectionMode.PROBABILITY_ANALYSIS,
            analysis_data=analysis_data,
            insights=insights,
            anomalies=anomalies,
            recommendations=recommendations
        )
    
    def _analyze_phases(self, state_vector: np.ndarray) -> StateInspectionResult:
        """Analyze quantum phases and their relationships."""
        analysis_data = {}
        insights = []
        anomalies = []
        recommendations = []
        
        phases = np.angle(state_vector)
        amplitudes = np.abs(state_vector)
        
        # Only analyze phases where amplitude is significant
        significant_mask = amplitudes > self.anomaly_thresholds['amplitude_threshold']
        significant_phases = phases[significant_mask]
        
        if len(significant_phases) > 0:
            analysis_data['phase_stats'] = {
                'mean': float(np.mean(significant_phases)),
                'std': float(np.std(significant_phases)),
                'range': float(np.max(significant_phases) - np.min(significant_phases)),
                'wrapped_variance': float(1 - abs(np.mean(np.exp(1j * significant_phases))))
            }
            
            # Check for phase patterns
            if np.std(significant_phases) < self.anomaly_thresholds['phase_threshold']:
                insights.append("Phases are very uniform - possible global phase factor")
                recommendations.append("Consider removing global phase for canonical representation")
            
            # Check for specific phase relationships
            phase_diffs = np.diff(significant_phases)
            if len(phase_diffs) > 0:
                analysis_data['phase_differences'] = {
                    'mean_diff': float(np.mean(phase_diffs)),
                    'std_diff': float(np.std(phase_diffs))
                }
                
                if np.allclose(phase_diffs, phase_diffs[0], atol=1e-6):
                    insights.append("Linear phase progression detected")
        else:
            anomalies.append("No significant amplitudes found for phase analysis")
        
        # Relative phase analysis
        if len(state_vector) >= 2:
            relative_phases = []
            reference_phase = phases[np.argmax(amplitudes)]
            for i, phase in enumerate(phases):
                if amplitudes[i] > self.anomaly_thresholds['amplitude_threshold']:
                    rel_phase = (phase - reference_phase) % (2 * np.pi)
                    if rel_phase > np.pi:
                        rel_phase -= 2 * np.pi
                    relative_phases.append(rel_phase)
            
            analysis_data['relative_phases'] = relative_phases
        
        return StateInspectionResult(
            mode=InspectionMode.PHASE_ANALYSIS,
            analysis_data=analysis_data,
            insights=insights,
            anomalies=anomalies,
            recommendations=recommendations
        )
    
    def _analyze_entanglement(self, state_vector: np.ndarray) -> StateInspectionResult:
        """Analyze entanglement properties of the quantum state."""
        analysis_data = {}
        insights = []
        anomalies = []
        recommendations = []
        
        n_qubits = int(np.log2(len(state_vector)))
        if n_qubits < 2:
            insights.append("Single qubit state - no entanglement possible")
            return StateInspectionResult(
                mode=InspectionMode.ENTANGLEMENT_ANALYSIS,
                analysis_data={'n_qubits': n_qubits},
                insights=insights,
                anomalies=anomalies,
                recommendations=recommendations
            )
        
        # Calculate entanglement measures for bipartitions
        entanglement_measures = []
        
        for partition_size in range(1, n_qubits):
            # Calculate reduced density matrices and entropy
            entropy = self._calculate_entanglement_entropy(state_vector, partition_size, n_qubits)
            entanglement_measures.append({
                'partition_size': partition_size,
                'entropy': entropy,
                'entanglement_present': entropy > self.anomaly_thresholds['entanglement_threshold']
            })
        
        analysis_data['entanglement_measures'] = entanglement_measures
        
        # Overall entanglement assessment
        max_entropy = max([measure['entropy'] for measure in entanglement_measures])
        analysis_data['max_entanglement_entropy'] = max_entropy
        
        if max_entropy < self.anomaly_thresholds['entanglement_threshold']:
            insights.append("State appears to be separable (no significant entanglement)")
            recommendations.append("Consider adding entangling gates if entanglement is desired")
        elif max_entropy > np.log2(min(2**partition_size, 2**(n_qubits-partition_size))) * 0.9:
            insights.append("State is highly entangled")
        else:
            insights.append("State shows moderate entanglement")
        
        # Check for specific entangled states
        if n_qubits == 2:
            # Check for Bell states
            bell_states = [
                np.array([1, 0, 0, 1]) / np.sqrt(2),  # |Φ+⟩
                np.array([1, 0, 0, -1]) / np.sqrt(2), # |Φ-⟩
                np.array([0, 1, 1, 0]) / np.sqrt(2),  # |Ψ+⟩
                np.array([0, 1, -1, 0]) / np.sqrt(2)  # |Ψ-⟩
            ]
            
            for i, bell_state in enumerate(bell_states):
                fidelity = abs(np.vdot(bell_state, state_vector))**2
                if fidelity > 0.99:
                    bell_names = ["Φ+", "Φ-", "Ψ+", "Ψ-"]
                    insights.append(f"State is very close to Bell state |{bell_names[i]}⟩")
        
        return StateInspectionResult(
            mode=InspectionMode.ENTANGLEMENT_ANALYSIS,
            analysis_data=analysis_data,
            insights=insights,
            anomalies=anomalies,
            recommendations=recommendations
        )
    
    def _calculate_entanglement_entropy(self, state_vector: np.ndarray, partition_size: int, n_qubits: int) -> float:
        """Calculate entanglement entropy for a given bipartition."""
        try:
            # Reshape state vector into tensor form
            shape = [2] * n_qubits
            state_tensor = state_vector.reshape(shape)
            
            # Partition qubits
            partition_A = list(range(partition_size))
            partition_B = list(range(partition_size, n_qubits))
            
            # Calculate reduced density matrix for partition A
            axes_to_trace = tuple(range(partition_size, n_qubits))
            rho_A = np.tensordot(state_tensor, np.conj(state_tensor), axes=(axes_to_trace, axes_to_trace))
            
            # Flatten to matrix form
            dim_A = 2 ** partition_size
            rho_A = rho_A.reshape(dim_A, dim_A)
            
            # Calculate eigenvalues and entropy
            eigenvals = np.real(np.linalg.eigvals(rho_A))
            eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
            entropy = -np.sum(eigenvals * np.log2(eigenvals))
            
            return entropy
            
        except Exception as e:
            logger.warning(f"Entanglement entropy calculation failed: {e}")
            return 0.0
    
    def _analyze_coherence(self, state_vector: np.ndarray) -> StateInspectionResult:
        """Analyze quantum coherence properties."""
        analysis_data = {}
        insights = []
        anomalies = []
        recommendations = []
        
        n_qubits = int(np.log2(len(state_vector)))
        
        # Calculate coherence measures
        density_matrix = np.outer(state_vector, np.conj(state_vector))
        
        # Off-diagonal coherence
        off_diagonal_sum = np.sum(np.abs(density_matrix)) - np.sum(np.abs(np.diag(density_matrix)))
        total_sum = np.sum(np.abs(density_matrix))
        coherence_ratio = off_diagonal_sum / total_sum if total_sum > 0 else 0
        
        analysis_data['coherence_measures'] = {
            'off_diagonal_coherence': float(off_diagonal_sum),
            'coherence_ratio': float(coherence_ratio),
            'l1_coherence': float(off_diagonal_sum),
            'relative_entropy_coherence': self._calculate_relative_entropy_coherence(density_matrix)
        }
        
        # Coherence assessment
        if coherence_ratio < 0.1:
            insights.append("State has low coherence - mostly classical")
            recommendations.append("Consider adding superposition to increase coherence")
        elif coherence_ratio > 0.8:
            insights.append("State has high coherence - strongly quantum")
        else:
            insights.append("State has moderate coherence")
        
        # Check for decoherence patterns
        diagonal_elements = np.abs(np.diag(density_matrix))
        if np.max(diagonal_elements) > 0.95:
            insights.append("State appears to be decohered to a computational basis state")
        
        return StateInspectionResult(
            mode=InspectionMode.COHERENCE_ANALYSIS,
            analysis_data=analysis_data,
            insights=insights,
            anomalies=anomalies,
            recommendations=recommendations
        )
    
    def _calculate_relative_entropy_coherence(self, rho: np.ndarray) -> float:
        """Calculate relative entropy of coherence."""
        try:
            # Diagonal part of density matrix
            rho_diag = np.diag(np.diag(rho))
            
            # Calculate relative entropy
            eigenvals_rho = np.real(np.linalg.eigvals(rho))
            eigenvals_diag = np.real(np.linalg.eigvals(rho_diag))
            
            eigenvals_rho = eigenvals_rho[eigenvals_rho > 1e-12]
            eigenvals_diag = eigenvals_diag[eigenvals_diag > 1e-12]
            
            entropy_rho = -np.sum(eigenvals_rho * np.log2(eigenvals_rho))
            entropy_diag = -np.sum(eigenvals_diag * np.log2(eigenvals_diag))
            
            return entropy_diag - entropy_rho
            
        except Exception:
            return 0.0
    
    def _analyze_correlations(self, state_vector: np.ndarray) -> StateInspectionResult:
        """Analyze quantum correlations and non-local properties."""
        analysis_data = {}
        insights = []
        anomalies = []
        recommendations = []
        
        n_qubits = int(np.log2(len(state_vector)))
        if n_qubits < 2:
            insights.append("Single qubit state - no correlations possible")
            return StateInspectionResult(
                mode=InspectionMode.CORRELATION_ANALYSIS,
                analysis_data={'n_qubits': n_qubits},
                insights=insights,
                anomalies=anomalies,
                recommendations=recommendations
            )
        
        # Calculate two-qubit correlations
        correlations = []
        probabilities = np.abs(state_vector) ** 2
        
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                # Calculate joint probabilities
                p_00 = p_01 = p_10 = p_11 = 0.0
                
                for k, prob in enumerate(probabilities):
                    bit_i = (k >> i) & 1
                    bit_j = (k >> j) & 1
                    
                    if bit_i == 0 and bit_j == 0:
                        p_00 += prob
                    elif bit_i == 0 and bit_j == 1:
                        p_01 += prob
                    elif bit_i == 1 and bit_j == 0:
                        p_10 += prob
                    else:
                        p_11 += prob
                
                # Calculate marginal probabilities
                p_i0 = p_00 + p_01
                p_i1 = p_10 + p_11
                p_j0 = p_00 + p_10
                p_j1 = p_01 + p_11
                
                # Calculate correlation coefficient
                mean_i = p_i1
                mean_j = p_j1
                variance_i = p_i1 * (1 - p_i1)
                variance_j = p_j1 * (1 - p_j1)
                
                if variance_i > 1e-12 and variance_j > 1e-12:
                    covariance = p_11 - mean_i * mean_j
                    correlation = covariance / np.sqrt(variance_i * variance_j)
                else:
                    correlation = 0.0
                
                correlations.append({
                    'qubit_pair': (i, j),
                    'correlation': float(correlation),
                    'joint_probs': {'00': p_00, '01': p_01, '10': p_10, '11': p_11},
                    'marginal_probs': {'i0': p_i0, 'i1': p_i1, 'j0': p_j0, 'j1': p_j1}
                })
        
        analysis_data['two_qubit_correlations'] = correlations
        
        # Identify strong correlations
        strong_correlations = [c for c in correlations if abs(c['correlation']) > 0.5]
        if strong_correlations:
            insights.append(f"Found {len(strong_correlations)} strong correlations")
            for corr in strong_correlations:
                qubits = corr['qubit_pair']
                corr_val = corr['correlation']
                if corr_val > 0.5:
                    insights.append(f"Qubits {qubits[0]} and {qubits[1]} are positively correlated (r={corr_val:.3f})")
                else:
                    insights.append(f"Qubits {qubits[0]} and {qubits[1]} are negatively correlated (r={corr_val:.3f})")
        else:
            insights.append("No strong correlations detected")
        
        return StateInspectionResult(
            mode=InspectionMode.CORRELATION_ANALYSIS,
            analysis_data=analysis_data,
            insights=insights,
            anomalies=anomalies,
            recommendations=recommendations
        )
    
    def _analyze_purity(self, state_vector: np.ndarray) -> StateInspectionResult:
        """Analyze state purity and mixedness."""
        analysis_data = {}
        insights = []
        anomalies = []
        recommendations = []
        
        density_matrix = np.outer(state_vector, np.conj(state_vector))
        purity = np.real(np.trace(np.dot(density_matrix, density_matrix)))
        
        n_qubits = int(np.log2(len(state_vector)))
        max_mixedness = 1.0 / (2 ** n_qubits)
        
        analysis_data['purity_measures'] = {
            'purity': float(purity),
            'mixedness': float(1 - purity),
            'max_mixedness': float(max_mixedness),
            'purity_ratio': float(purity / 1.0),  # Pure state has purity 1
            'is_pure': bool(abs(purity - 1.0) < 1e-10)
        }
        
        if abs(purity - 1.0) < 1e-6:
            insights.append("State is pure")
        elif purity < self.anomaly_thresholds['purity_threshold']:
            insights.append("State is significantly mixed")
            anomalies.append("Low purity detected - state may be decohered")
            recommendations.append("Check for decoherence sources")
        else:
            insights.append("State has moderate purity")
        
        # Linear entropy (alternative purity measure)
        linear_entropy = 1 - purity
        analysis_data['purity_measures']['linear_entropy'] = float(linear_entropy)
        
        return StateInspectionResult(
            mode=InspectionMode.PURITY_ANALYSIS,
            analysis_data=analysis_data,
            insights=insights,
            anomalies=anomalies,
            recommendations=recommendations
        )
    
    def _analyze_fidelity(self, state_vector: np.ndarray) -> StateInspectionResult:
        """Analyze fidelity with respect to common quantum states."""
        analysis_data = {}
        insights = []
        anomalies = []
        recommendations = []
        
        n_qubits = int(np.log2(len(state_vector)))
        
        # Compare with standard states
        fidelities = {}
        
        # Computational basis states
        for i in range(min(2**n_qubits, 8)):  # Limit to prevent memory issues
            basis_state = np.zeros(2**n_qubits, dtype=complex)
            basis_state[i] = 1.0
            fidelity = abs(np.vdot(basis_state, state_vector))**2
            if fidelity > 0.01:  # Only store significant fidelities
                basis_label = format(i, f'0{n_qubits}b')
                fidelities[f'|{basis_label}⟩'] = float(fidelity)
        
        # Uniform superposition
        uniform_state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
        fidelities['uniform_superposition'] = float(abs(np.vdot(uniform_state, state_vector))**2)
        
        # Bell states (for 2-qubit systems)
        if n_qubits == 2:
            bell_states = {
                'Bell_Phi_plus': np.array([1, 0, 0, 1]) / np.sqrt(2),
                'Bell_Phi_minus': np.array([1, 0, 0, -1]) / np.sqrt(2),
                'Bell_Psi_plus': np.array([0, 1, 1, 0]) / np.sqrt(2),
                'Bell_Psi_minus': np.array([0, 1, -1, 0]) / np.sqrt(2)
            }
            
            for name, bell_state in bell_states.items():
                fidelity = abs(np.vdot(bell_state, state_vector))**2
                fidelities[name] = float(fidelity)
        
        # GHZ states (for multi-qubit systems)
        if n_qubits >= 3:
            ghz_state = np.zeros(2**n_qubits, dtype=complex)
            ghz_state[0] = 1/np.sqrt(2)  # |000...0⟩
            ghz_state[-1] = 1/np.sqrt(2)  # |111...1⟩
            fidelities['GHZ_state'] = float(abs(np.vdot(ghz_state, state_vector))**2)
        
        analysis_data['fidelities'] = fidelities
        
        # Find best matches
        best_match = max(fidelities.items(), key=lambda x: x[1])
        analysis_data['best_match'] = {
            'state': best_match[0],
            'fidelity': best_match[1]
        }
        
        if best_match[1] > 0.99:
            insights.append(f"State is very close to {best_match[0]} (F = {best_match[1]:.4f})")
        elif best_match[1] > 0.8:
            insights.append(f"State has good overlap with {best_match[0]} (F = {best_match[1]:.4f})")
        else:
            insights.append("State does not closely match any standard quantum state")
        
        return StateInspectionResult(
            mode=InspectionMode.FIDELITY_ANALYSIS,
            analysis_data=analysis_data,
            insights=insights,
            anomalies=anomalies,
            recommendations=recommendations
        )