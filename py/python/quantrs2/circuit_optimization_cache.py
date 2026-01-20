"""
Circuit Optimization and Intelligent Caching Service for QuantRS2

This module provides intelligent caching and optimization strategies for frequently
used quantum circuits, compilation results, and execution patterns.
"""

import time
import json
import hashlib
import logging
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import weakref

from .connection_pooling import QuantumResultCache, CacheConfig, CacheBackend
from .resource_management import analyze_circuit_resources

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Circuit optimization levels."""
    NONE = 0
    BASIC = 1
    STANDARD = 2
    AGGRESSIVE = 3
    ADAPTIVE = 4


class CircuitPattern(Enum):
    """Common quantum circuit patterns."""
    BELL_STATE = "bell_state"
    GHZ_STATE = "ghz_state"
    QFT = "quantum_fourier_transform"
    GROVER = "grover_search"
    VQE = "variational_quantum_eigensolver"
    QAOA = "quantum_approximate_optimization"
    TELEPORTATION = "quantum_teleportation"
    CUSTOM = "custom"


@dataclass
class CircuitSignature:
    """Unique signature for quantum circuits."""
    gate_sequence_hash: str
    qubit_count: int
    gate_count: int
    depth: int
    gate_types: Set[str] = field(default_factory=set)
    connectivity_hash: str = ""
    parameter_hash: str = ""
    
    def to_cache_key(self) -> str:
        """Generate cache key from signature."""
        components = [
            self.gate_sequence_hash,
            str(self.qubit_count),
            str(self.gate_count),
            str(self.depth),
            sorted(self.gate_types),
            self.connectivity_hash,
            self.parameter_hash
        ]
        content = ":".join(str(c) for c in components)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class OptimizationResult:
    """Result of circuit optimization."""
    original_signature: CircuitSignature
    optimized_signature: CircuitSignature
    optimization_time: float
    improvements: Dict[str, Any] = field(default_factory=dict)
    applied_passes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def improvement_ratio(self, metric: str) -> float:
        """Calculate improvement ratio for a metric."""
        if metric not in self.improvements:
            return 0.0
        
        original = self.improvements[metric].get('original', 0)
        optimized = self.improvements[metric].get('optimized', 0)
        
        if original == 0:
            return 0.0
        
        return (original - optimized) / original


@dataclass
class ExecutionProfile:
    """Execution profile for circuit patterns."""
    pattern: CircuitPattern
    signature: CircuitSignature
    execution_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    success_rate: float = 1.0
    last_executed: float = field(default_factory=time.time)
    preferred_backend: Optional[str] = None
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    
    def update_execution_stats(self, execution_time: float, success: bool):
        """Update execution statistics."""
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.execution_count
        
        # Update success rate with exponential moving average
        alpha = 0.1
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        self.last_executed = time.time()


class CircuitPatternDetector:
    """Detects common quantum circuit patterns."""
    
    def __init__(self):
        self.pattern_signatures = self._initialize_pattern_signatures()
    
    def _initialize_pattern_signatures(self) -> Dict[CircuitPattern, List[str]]:
        """Initialize known pattern signatures."""
        return {
            CircuitPattern.BELL_STATE: [
                "h:cnot",  # Basic Bell state
                "h:cx"     # Alternative notation
            ],
            CircuitPattern.GHZ_STATE: [
                "h:cnot*",  # H followed by multiple CNOTs
                "h:cx*"
            ],
            CircuitPattern.QFT: [
                "h:cp*:swap*",  # Hadamard, controlled phases, swaps
                "h:cu1*:swap*"
            ],
            CircuitPattern.GROVER: [
                "h*:oracle:diffuser",  # Superposition, oracle, diffuser
                "h*:mcx:h*:mcp:h*"
            ],
            CircuitPattern.VQE: [
                "ry*:cnot*:measure",  # Parameterized Y rotations
                "ansatz:measure"
            ],
            CircuitPattern.QAOA: [
                "h*:rzz*:rx*",  # QAOA mixing and cost layers
                "mixer:cost"
            ]
        }
    
    def detect_pattern(self, circuit: Any) -> Tuple[CircuitPattern, float]:
        """
        Detect circuit pattern with confidence score.
        
        Returns:
            Tuple of (detected_pattern, confidence_score)
        """
        try:
            gate_sequence = self._extract_gate_sequence(circuit)
            normalized_sequence = self._normalize_sequence(gate_sequence)
            
            best_pattern = CircuitPattern.CUSTOM
            best_confidence = 0.0
            
            for pattern, signatures in self.pattern_signatures.items():
                for signature in signatures:
                    confidence = self._match_signature(normalized_sequence, signature)
                    if confidence > best_confidence:
                        best_pattern = pattern
                        best_confidence = confidence
            
            return best_pattern, best_confidence
            
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
            return CircuitPattern.CUSTOM, 0.0
    
    def _extract_gate_sequence(self, circuit: Any) -> List[str]:
        """Extract gate sequence from circuit."""
        gates = []
        
        try:
            if hasattr(circuit, 'data'):
                # Qiskit-style circuit
                for instruction in circuit.data:
                    gates.append(instruction[0].name)
            elif hasattr(circuit, 'gates'):
                # QuantRS2-style circuit
                for gate in circuit.gates:
                    gates.append(gate.name if hasattr(gate, 'name') else str(gate))
            elif isinstance(circuit, dict) and 'gates' in circuit:
                # Dictionary representation
                for gate in circuit['gates']:
                    if isinstance(gate, tuple):
                        gates.append(gate[0])
                    else:
                        gates.append(str(gate))
            else:
                logger.warning(f"Unknown circuit format: {type(circuit)}")
                
        except Exception as e:
            logger.warning(f"Failed to extract gate sequence: {e}")
        
        return gates
    
    def _normalize_sequence(self, sequence: List[str]) -> str:
        """Normalize gate sequence for pattern matching."""
        # Group consecutive identical gates
        normalized = []
        current_gate = None
        count = 0
        
        for gate in sequence:
            # Normalize gate names
            gate = gate.lower().replace('_', '').replace('-', '')
            
            if gate == current_gate:
                count += 1
            else:
                if current_gate:
                    if count > 1:
                        normalized.append(f"{current_gate}*")
                    else:
                        normalized.append(current_gate)
                current_gate = gate
                count = 1
        
        # Add the last group
        if current_gate:
            if count > 1:
                normalized.append(f"{current_gate}*")
            else:
                normalized.append(current_gate)
        
        return ":".join(normalized)
    
    def _match_signature(self, sequence: str, signature: str) -> float:
        """Match sequence against pattern signature."""
        # Simple substring matching with wildcards
        # In a real implementation, you'd use more sophisticated pattern matching
        
        sequence_parts = sequence.split(":")
        signature_parts = signature.split(":")
        
        if len(signature_parts) > len(sequence_parts):
            return 0.0
        
        matches = 0
        for i, sig_part in enumerate(signature_parts):
            if i >= len(sequence_parts):
                break
                
            seq_part = sequence_parts[i]
            
            if sig_part.endswith("*"):
                # Wildcard match
                base_sig = sig_part[:-1]
                if seq_part.startswith(base_sig) or seq_part.endswith("*"):
                    matches += 1
            elif sig_part == seq_part:
                matches += 1
        
        return matches / len(signature_parts)


class CircuitOptimizationCache:
    """Advanced caching system for circuit optimizations and execution results."""
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        self.cache_config = cache_config or CacheConfig()
        
        # Initialize caches
        self.result_cache = QuantumResultCache(self.cache_config)
        self.optimization_cache = QuantumResultCache(
            CacheConfig(
                **{**self.cache_config.__dict__, 'max_memory_mb': 256}
            )
        )
        
        # Pattern detection
        self.pattern_detector = CircuitPatternDetector()
        
        # Execution profiles
        self.execution_profiles: Dict[str, ExecutionProfile] = {}
        self.profiles_lock = threading.RLock()
        
        # Optimization statistics
        self.optimization_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'optimizations_performed': 0,
            'total_optimization_time': 0.0,
            'patterns_detected': defaultdict(int),
            'improvement_ratios': defaultdict(list)
        }
        
        # Frequently used circuits
        self.circuit_usage = Counter()
        self.usage_lock = threading.RLock()
        
        logger.info("Circuit optimization cache initialized")
    
    def compute_circuit_signature(self, circuit: Any) -> CircuitSignature:
        """Compute unique signature for a circuit."""
        try:
            # Extract circuit properties
            analysis = analyze_circuit_resources(circuit)
            gate_sequence = self.pattern_detector._extract_gate_sequence(circuit)
            
            # Create gate sequence hash
            gate_sequence_str = ":".join(gate_sequence)
            gate_sequence_hash = hashlib.sha256(gate_sequence_str.encode()).hexdigest()
            
            # Extract gate types
            gate_types = set(gate_sequence)
            
            # Create connectivity hash (simplified)
            connectivity_hash = self._compute_connectivity_hash(circuit)
            
            # Create parameter hash
            parameter_hash = self._compute_parameter_hash(circuit)
            
            return CircuitSignature(
                gate_sequence_hash=gate_sequence_hash,
                qubit_count=analysis.get('qubits', 0),
                gate_count=analysis.get('gates', 0),
                depth=analysis.get('depth', 0),
                gate_types=gate_types,
                connectivity_hash=connectivity_hash,
                parameter_hash=parameter_hash
            )
            
        except Exception as e:
            logger.error(f"Failed to compute circuit signature: {e}")
            # Return a minimal signature
            return CircuitSignature(
                gate_sequence_hash="unknown",
                qubit_count=0,
                gate_count=0,
                depth=0
            )
    
    def _compute_connectivity_hash(self, circuit: Any) -> str:
        """Compute hash of qubit connectivity pattern."""
        try:
            connections = set()
            
            if hasattr(circuit, 'data'):
                # Qiskit-style
                for instruction in circuit.data:
                    qubits = [q.index for q in instruction[1]]
                    if len(qubits) > 1:
                        connections.add(tuple(sorted(qubits)))
            elif hasattr(circuit, 'gates'):
                # QuantRS2-style
                for gate in circuit.gates:
                    if hasattr(gate, 'qubits'):
                        qubits = gate.qubits
                        if len(qubits) > 1:
                            connections.add(tuple(sorted(qubits)))
            
            connections_str = ":".join(sorted(str(c) for c in connections))
            return hashlib.sha256(connections_str.encode()).hexdigest()[:16]
            
        except Exception:
            return ""
    
    def _compute_parameter_hash(self, circuit: Any) -> str:
        """Compute hash of circuit parameters."""
        try:
            parameters = []
            
            if hasattr(circuit, 'parameters'):
                # Parameterized circuit
                for param in circuit.parameters:
                    parameters.append(str(param))
            
            if not parameters:
                return ""
            
            param_str = ":".join(sorted(parameters))
            return hashlib.sha256(param_str.encode()).hexdigest()[:16]
            
        except Exception:
            return ""
    
    def get_cached_result(self, circuit: Any, execution_config: Dict[str, Any]) -> Optional[Any]:
        """Get cached execution result for circuit."""
        try:
            signature = self.compute_circuit_signature(circuit)
            cache_key = self._create_execution_cache_key(signature, execution_config)
            
            result = self.result_cache.get(cache_key)
            
            if result:
                # Update usage statistics
                with self.usage_lock:
                    self.circuit_usage[signature.to_cache_key()] += 1
                
                logger.debug(f"Cache hit for circuit execution: {cache_key[:16]}")
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached result: {e}")
            return None
    
    def cache_execution_result(self, circuit: Any, execution_config: Dict[str, Any], 
                             result: Any, execution_time: float, success: bool) -> bool:
        """Cache execution result."""
        try:
            signature = self.compute_circuit_signature(circuit)
            cache_key = self._create_execution_cache_key(signature, execution_config)
            
            # Detect pattern
            pattern, confidence = self.pattern_detector.detect_pattern(circuit)
            
            # Update execution profile
            self._update_execution_profile(signature, pattern, execution_time, success, execution_config)
            
            # Cache the result
            tags = [
                f"pattern:{pattern.value}",
                f"qubits:{signature.qubit_count}",
                f"gates:{signature.gate_count}",
                f"success:{success}"
            ]
            
            # Determine TTL based on circuit characteristics
            ttl = self._compute_result_ttl(signature, pattern, success)
            
            success = self.result_cache.put(cache_key, result, ttl=ttl, tags=tags)
            
            if success:
                logger.debug(f"Cached execution result: {cache_key[:16]}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache execution result: {e}")
            return False
    
    def get_cached_optimization(self, circuit: Any, 
                              optimization_level: OptimizationLevel) -> Optional[OptimizationResult]:
        """Get cached optimization result."""
        try:
            signature = self.compute_circuit_signature(circuit)
            cache_key = f"opt:{signature.to_cache_key()}:{optimization_level.value}"
            
            result = self.optimization_cache.get(cache_key)
            
            if result:
                self.optimization_stats['cache_hits'] += 1
                logger.debug(f"Optimization cache hit: {cache_key[:16]}")
                return result
            
            self.optimization_stats['cache_misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached optimization: {e}")
            return None
    
    def cache_optimization_result(self, original_circuit: Any, optimized_circuit: Any,
                                optimization_level: OptimizationLevel, 
                                optimization_time: float, applied_passes: List[str],
                                improvements: Dict[str, Any]) -> bool:
        """Cache optimization result."""
        try:
            original_signature = self.compute_circuit_signature(original_circuit)
            optimized_signature = self.compute_circuit_signature(optimized_circuit)
            
            optimization_result = OptimizationResult(
                original_signature=original_signature,
                optimized_signature=optimized_signature,
                optimization_time=optimization_time,
                improvements=improvements,
                applied_passes=applied_passes
            )
            
            cache_key = f"opt:{original_signature.to_cache_key()}:{optimization_level.value}"
            
            # Cache for longer period since optimizations are expensive to compute
            ttl = 86400  # 24 hours
            
            tags = [
                f"optimization_level:{optimization_level.value}",
                f"qubits:{original_signature.qubit_count}",
                f"original_gates:{original_signature.gate_count}",
                f"optimized_gates:{optimized_signature.gate_count}"
            ]
            
            success = self.optimization_cache.put(cache_key, optimization_result, ttl=ttl, tags=tags)
            
            if success:
                self.optimization_stats['optimizations_performed'] += 1
                self.optimization_stats['total_optimization_time'] += optimization_time
                
                # Track improvement ratios
                for metric, improvement in improvements.items():
                    if 'ratio' in improvement:
                        self.optimization_stats['improvement_ratios'][metric].append(improvement['ratio'])
                
                logger.debug(f"Cached optimization result: {cache_key[:16]}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache optimization result: {e}")
            return False
    
    def _create_execution_cache_key(self, signature: CircuitSignature, 
                                  execution_config: Dict[str, Any]) -> str:
        """Create cache key for execution results."""
        config_str = json.dumps(execution_config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return f"exec:{signature.to_cache_key()}:{config_hash}"
    
    def _update_execution_profile(self, signature: CircuitSignature, pattern: CircuitPattern,
                                execution_time: float, success: bool, 
                                execution_config: Dict[str, Any]):
        """Update execution profile for circuit pattern."""
        with self.profiles_lock:
            profile_key = f"{pattern.value}:{signature.qubit_count}:{signature.gate_count}"
            
            if profile_key not in self.execution_profiles:
                self.execution_profiles[profile_key] = ExecutionProfile(
                    pattern=pattern,
                    signature=signature,
                    preferred_backend=execution_config.get('backend')
                )
            
            profile = self.execution_profiles[profile_key]
            profile.update_execution_stats(execution_time, success)
            
            # Update pattern detection statistics
            self.optimization_stats['patterns_detected'][pattern.value] += 1
    
    def _compute_result_ttl(self, signature: CircuitSignature, 
                          pattern: CircuitPattern, success: bool) -> float:
        """Compute appropriate TTL for cached results."""
        base_ttl = self.cache_config.default_ttl
        
        # Adjust based on pattern
        pattern_multipliers = {
            CircuitPattern.BELL_STATE: 2.0,      # Common, cache longer
            CircuitPattern.GHZ_STATE: 2.0,
            CircuitPattern.QFT: 1.5,
            CircuitPattern.GROVER: 1.5,
            CircuitPattern.VQE: 0.5,             # Parameterized, cache shorter
            CircuitPattern.QAOA: 0.5,
            CircuitPattern.CUSTOM: 1.0
        }
        
        multiplier = pattern_multipliers.get(pattern, 1.0)
        
        # Adjust based on circuit size (larger circuits cached longer due to computation cost)
        if signature.gate_count > 1000:
            multiplier *= 2.0
        elif signature.gate_count > 100:
            multiplier *= 1.5
        
        # Successful executions cached longer
        if success:
            multiplier *= 1.2
        else:
            multiplier *= 0.5
        
        return base_ttl * multiplier
    
    def get_frequently_used_circuits(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently used circuits."""
        with self.usage_lock:
            return self.circuit_usage.most_common(limit)
    
    def get_execution_recommendations(self, circuit: Any) -> Dict[str, Any]:
        """Get execution recommendations based on historical data."""
        try:
            signature = self.compute_circuit_signature(circuit)
            pattern, confidence = self.pattern_detector.detect_pattern(circuit)
            
            recommendations = {
                'detected_pattern': pattern.value,
                'pattern_confidence': confidence,
                'recommended_backend': None,
                'estimated_execution_time': None,
                'optimization_recommendations': [],
                'similar_circuits_count': 0
            }
            
            # Find similar execution profiles
            with self.profiles_lock:
                similar_profiles = []
                for profile_key, profile in self.execution_profiles.items():
                    if (profile.pattern == pattern and 
                        abs(profile.signature.qubit_count - signature.qubit_count) <= 2 and
                        abs(profile.signature.gate_count - signature.gate_count) <= 50):
                        similar_profiles.append(profile)
                
                if similar_profiles:
                    # Aggregate recommendations from similar circuits
                    avg_time = sum(p.average_execution_time for p in similar_profiles) / len(similar_profiles)
                    best_backend = Counter(p.preferred_backend for p in similar_profiles if p.preferred_backend).most_common(1)
                    
                    recommendations['estimated_execution_time'] = avg_time
                    recommendations['similar_circuits_count'] = len(similar_profiles)
                    
                    if best_backend:
                        recommendations['recommended_backend'] = best_backend[0][0]
                    
                    # Optimization recommendations
                    if signature.gate_count > 100:
                        recommendations['optimization_recommendations'].append("Consider circuit optimization")
                    
                    if any(p.success_rate < 0.8 for p in similar_profiles):
                        recommendations['optimization_recommendations'].append("Consider noise mitigation")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate execution recommendations: {e}")
            return {}
    
    def invalidate_pattern_cache(self, pattern: CircuitPattern):
        """Invalidate cache entries for a specific pattern."""
        tags = [f"pattern:{pattern.value}"]
        
        result_count = self.result_cache.invalidate_by_tags(tags)
        opt_count = self.optimization_cache.invalidate_by_tags(tags)
        
        logger.info(f"Invalidated {result_count} result cache entries and "
                   f"{opt_count} optimization cache entries for pattern {pattern.value}")
    
    def optimize_cache_performance(self):
        """Optimize cache performance based on usage patterns."""
        try:
            # Get frequently used circuits
            frequent_circuits = self.get_frequently_used_circuits(20)
            
            # Ensure frequently used circuits are cached with longer TTL
            for circuit_key, usage_count in frequent_circuits:
                if usage_count > 10:  # Threshold for "frequent" usage
                    # Would implement pre-warming or TTL extension here
                    logger.debug(f"Circuit {circuit_key[:16]} used {usage_count} times")
            
            # Clean up old execution profiles
            with self.profiles_lock:
                current_time = time.time()
                old_profiles = [
                    key for key, profile in self.execution_profiles.items()
                    if current_time - profile.last_executed > 86400  # 24 hours
                ]
                
                for key in old_profiles:
                    del self.execution_profiles[key]
                
                if old_profiles:
                    logger.debug(f"Cleaned up {len(old_profiles)} old execution profiles")
            
        except Exception as e:
            logger.error(f"Cache performance optimization failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        result_stats = self.result_cache.get_statistics()
        opt_stats = self.optimization_cache.get_statistics()
        
        with self.profiles_lock:
            profile_count = len(self.execution_profiles)
        
        with self.usage_lock:
            total_circuit_executions = sum(self.circuit_usage.values())
            unique_circuits = len(self.circuit_usage)
        
        return {
            'result_cache': result_stats,
            'optimization_cache': opt_stats,
            'optimization_stats': dict(self.optimization_stats),
            'execution_profiles': {
                'total_profiles': profile_count,
                'unique_circuits': unique_circuits,
                'total_executions': total_circuit_executions
            },
            'patterns_detected': dict(self.optimization_stats['patterns_detected']),
            'cache_efficiency': {
                'result_hit_rate': result_stats.get('hit_rate', 0.0),
                'optimization_hit_rate': opt_stats.get('hit_rate', 0.0)
            }
        }
    
    def close(self):
        """Close the circuit optimization cache."""
        logger.info("Closing circuit optimization cache")
        
        self.result_cache.close()
        self.optimization_cache.close()


# Export main classes
__all__ = [
    'OptimizationLevel',
    'CircuitPattern',
    'CircuitSignature',
    'OptimizationResult',
    'ExecutionProfile',
    'CircuitPatternDetector',
    'CircuitOptimizationCache'
]