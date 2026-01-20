#!/usr/bin/env python3
"""
QuantRS2 Quantum Code Analysis Tools.

This module provides comprehensive static analysis, optimization suggestions, and code quality metrics
specifically designed for quantum software development:
- Static analysis of quantum code with quantum-specific patterns
- Code quality metrics and complexity analysis
- Optimization suggestions and performance recommendations
- Circuit pattern detection and anti-pattern identification
- Integration with IDE plugins and development tools
- Quantum algorithm structure analysis
- Resource usage optimization recommendations
- Code style checking for quantum code conventions
"""

import ast
import os
import re
import time
import json
import hashlib
import logging
import sqlite3
import threading
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from contextlib import contextmanager
import numpy as np

# Optional dependencies with graceful fallbacks
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import toml
    HAS_TOML = True
except ImportError:
    HAS_TOML = False

try:
    import click
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

try:
    from flask import Flask, jsonify, request, render_template_string
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# QuantRS2 integration
try:
    import quantrs2
    HAS_QUANTRS2 = True
except ImportError:
    HAS_QUANTRS2 = False


class AnalysisLevel(Enum):
    """Code analysis depth levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"


class AnalysisType(Enum):
    """Types of code analysis."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    PERFORMANCE = "performance"
    QUANTUM_SPECIFIC = "quantum_specific"
    STYLE = "style"
    SECURITY = "security"
    COMPLEXITY = "complexity"
    OPTIMIZATION = "optimization"


class IssueSeverity(Enum):
    """Issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PatternType(Enum):
    """Quantum code pattern types."""
    GATE_SEQUENCE = "gate_sequence"
    CIRCUIT_STRUCTURE = "circuit_structure"
    ALGORITHM_PATTERN = "algorithm_pattern"
    ANTI_PATTERN = "anti_pattern"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"


class MetricType(Enum):
    """Code quality metric types."""
    QUANTUM_DEPTH = "quantum_depth"
    GATE_COUNT = "gate_count"
    ENTANGLEMENT_COMPLEXITY = "entanglement_complexity"
    QUBIT_EFFICIENCY = "qubit_efficiency"
    CIRCUIT_WIDTH = "circuit_width"
    CLASSICAL_COMPLEXITY = "classical_complexity"
    RESOURCE_USAGE = "resource_usage"
    MAINTAINABILITY = "maintainability"


@dataclass
class CodeLocation:
    """Code location information."""
    file_path: str
    line_number: int
    column_number: int = 0
    end_line: int = 0
    end_column: int = 0


@dataclass
class AnalysisIssue:
    """Code analysis issue."""
    issue_id: str
    title: str
    description: str
    severity: IssueSeverity
    analysis_type: AnalysisType
    location: CodeLocation
    suggestion: str = ""
    fix_available: bool = False
    auto_fixable: bool = False
    rule_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeMetric:
    """Code quality metric."""
    name: str
    metric_type: MetricType
    value: float
    description: str
    optimal_range: Tuple[float, float] = (0.0, 1.0)
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumPattern:
    """Detected quantum code pattern."""
    pattern_id: str
    pattern_type: PatternType
    name: str
    description: str
    location: CodeLocation
    confidence: float
    impact: str = ""
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationSuggestion:
    """Code optimization suggestion."""
    suggestion_id: str
    title: str
    description: str
    location: CodeLocation
    impact: str
    effort: str  # low, medium, high
    potential_improvement: str
    implementation_hints: List[str] = field(default_factory=list)
    before_code: str = ""
    after_code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Complete code analysis report."""
    file_path: str
    analysis_timestamp: float
    analysis_level: AnalysisLevel
    issues: List[AnalysisIssue] = field(default_factory=list)
    metrics: List[CodeMetric] = field(default_factory=list)
    patterns: List[QuantumPattern] = field(default_factory=list)
    optimizations: List[OptimizationSuggestion] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumCodeParser:
    """Parser for quantum code structures."""
    
    def __init__(self):
        """Initialize quantum code parser."""
        self.quantum_gate_patterns = {
            'single_qubit': ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz'],
            'two_qubit': ['cx', 'cnot', 'cy', 'cz', 'swap', 'iswap'],
            'multi_qubit': ['ccx', 'toffoli', 'fredkin', 'mcx'],
            'measurement': ['measure', 'measure_all', 'reset']
        }
        
        self.quantum_imports = [
            'quantrs2', 'qiskit', 'cirq', 'pennylane', 'pyquil',
            'quantum', 'qutip', 'strawberryfields'
        ]
        
        self.algorithm_patterns = {
            'vqe': ['variational', 'vqe', 'ansatz', 'optimizer'],
            'qaoa': ['qaoa', 'mixer', 'cost_function', 'beta', 'gamma'],
            'grover': ['grover', 'oracle', 'diffusion', 'amplitude_amplification'],
            'shor': ['shor', 'period_finding', 'quantum_fourier_transform'],
            'teleportation': ['teleport', 'bell_state', 'entanglement'],
            'superdense': ['superdense', 'dense_coding', 'classical_bits']
        }
    
    def parse_file(self, file_path: str) -> ast.AST:
        """Parse Python file into AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return ast.parse(content, filename=file_path)
        except Exception as e:
            raise ValueError(f"Failed to parse file {file_path}: {e}")
    
    def extract_quantum_imports(self, tree: ast.AST) -> List[str]:
        """Extract quantum-related imports."""
        quantum_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(qlib in alias.name.lower() for qlib in self.quantum_imports):
                        quantum_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(qlib in node.module.lower() for qlib in self.quantum_imports):
                    quantum_imports.append(node.module)
        
        return quantum_imports
    
    def find_circuit_definitions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find quantum circuit definitions."""
        circuits = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Check for quantum circuit patterns
                if self._is_quantum_circuit(node):
                    circuits.append({
                        'name': node.name,
                        'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                        'line': node.lineno,
                        'col': node.col_offset,
                        'docstring': ast.get_docstring(node) or "",
                        'quantum_operations': self._extract_quantum_operations(node)
                    })
        
        return circuits
    
    def _is_quantum_circuit(self, node: ast.AST) -> bool:
        """Check if node represents a quantum circuit."""
        if hasattr(node, 'name'):
            name_lower = node.name.lower()
            quantum_keywords = ['circuit', 'quantum', 'qubit', 'gate', 'algorithm']
            if any(keyword in name_lower for keyword in quantum_keywords):
                return True
        
        # Check for quantum operations in the node
        quantum_ops = self._extract_quantum_operations(node)
        return len(quantum_ops) > 0
    
    def _extract_quantum_operations(self, node: ast.AST) -> List[str]:
        """Extract quantum operations from AST node."""
        operations = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if hasattr(child.func, 'attr'):
                    op_name = child.func.attr.lower()
                    if any(op_name in gates for gates in self.quantum_gate_patterns.values()):
                        operations.append(op_name)
                elif hasattr(child.func, 'id'):
                    op_name = child.func.id.lower()
                    if any(op_name in gates for gates in self.quantum_gate_patterns.values()):
                        operations.append(op_name)
        
        return operations
    
    def analyze_algorithm_patterns(self, tree: ast.AST, content: str) -> List[str]:
        """Analyze quantum algorithm patterns."""
        detected_patterns = []
        content_lower = content.lower()
        
        for algorithm, keywords in self.algorithm_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                # Additional verification through AST
                if self._verify_algorithm_pattern(tree, algorithm, keywords):
                    detected_patterns.append(algorithm)
        
        return detected_patterns
    
    def _verify_algorithm_pattern(self, tree: ast.AST, algorithm: str, keywords: List[str]) -> bool:
        """Verify algorithm pattern through AST analysis."""
        # Simple verification - can be enhanced with more sophisticated pattern matching
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                if hasattr(node, 'id') and node.id.lower() in keywords:
                    return True
                if hasattr(node, 'attr') and node.attr.lower() in keywords:
                    return True
        return False


class QuantumCodeAnalyzer:
    """Comprehensive quantum code analyzer."""
    
    def __init__(self):
        """Initialize quantum code analyzer."""
        self.parser = QuantumCodeParser()
        self.analysis_rules = self._load_analysis_rules()
        self.metrics_calculators = self._setup_metrics_calculators()
        self.pattern_detectors = self._setup_pattern_detectors()
        self.optimization_rules = self._load_optimization_rules()
    
    def analyze_file(self, file_path: str, analysis_level: AnalysisLevel = AnalysisLevel.STANDARD) -> AnalysisReport:
        """Analyze a single file."""
        start_time = time.time()
        
        try:
            # Parse file
            tree = self.parser.parse_file(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Initialize report
            report = AnalysisReport(
                file_path=file_path,
                analysis_timestamp=time.time(),
                analysis_level=analysis_level
            )
            
            # Run different types of analysis based on level
            if analysis_level in [AnalysisLevel.BASIC, AnalysisLevel.STANDARD, AnalysisLevel.COMPREHENSIVE, AnalysisLevel.DEEP]:
                report.issues.extend(self._analyze_syntax(tree, content))
                report.metrics.extend(self._calculate_basic_metrics(tree, content))
            
            if analysis_level in [AnalysisLevel.STANDARD, AnalysisLevel.COMPREHENSIVE, AnalysisLevel.DEEP]:
                report.issues.extend(self._analyze_semantics(tree, content))
                report.patterns.extend(self._detect_patterns(tree, content))
                report.metrics.extend(self._calculate_quantum_metrics(tree, content))
            
            if analysis_level in [AnalysisLevel.COMPREHENSIVE, AnalysisLevel.DEEP]:
                report.issues.extend(self._analyze_performance(tree, content))
                report.optimizations.extend(self._suggest_optimizations(tree, content))
                report.metrics.extend(self._calculate_advanced_metrics(tree, content))
            
            if analysis_level == AnalysisLevel.DEEP:
                report.issues.extend(self._deep_analysis(tree, content))
                report.patterns.extend(self._detect_advanced_patterns(tree, content))
            
            # Generate summary
            report.summary = self._generate_summary(report)
            report.execution_time = time.time() - start_time
            
            return report
            
        except Exception as e:
            # Return error report
            error_report = AnalysisReport(
                file_path=file_path,
                analysis_timestamp=time.time(),
                analysis_level=analysis_level,
                execution_time=time.time() - start_time
            )
            
            error_issue = AnalysisIssue(
                issue_id="analysis_error",
                title="Analysis Error",
                description=f"Failed to analyze file: {e}",
                severity=IssueSeverity.ERROR,
                analysis_type=AnalysisType.SYNTAX,
                location=CodeLocation(file_path=file_path, line_number=1)
            )
            
            error_report.issues.append(error_issue)
            return error_report
    
    def analyze_directory(self, directory_path: str, analysis_level: AnalysisLevel = AnalysisLevel.STANDARD) -> Dict[str, AnalysisReport]:
        """Analyze all Python files in a directory."""
        reports = {}
        
        for file_path in Path(directory_path).rglob("*.py"):
            try:
                report = self.analyze_file(str(file_path), analysis_level)
                reports[str(file_path)] = report
            except Exception as e:
                logging.warning(f"Failed to analyze {file_path}: {e}")
        
        return reports
    
    def _load_analysis_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load analysis rules."""
        return {
            'quantum_specific': [
                {
                    'id': 'QS001',
                    'name': 'Inefficient Gate Sequence',
                    'description': 'Detect inefficient quantum gate sequences',
                    'pattern': r'\.h\(\d+\)\.h\(\d+\)',
                    'severity': IssueSeverity.WARNING,
                    'suggestion': 'Remove redundant Hadamard gates'
                },
                {
                    'id': 'QS002',
                    'name': 'Missing Measurement',
                    'description': 'Circuit without measurement operations',
                    'severity': IssueSeverity.INFO,
                    'suggestion': 'Consider adding measurement operations'
                },
                {
                    'id': 'QS003',
                    'name': 'Excessive Circuit Depth',
                    'description': 'Circuit depth exceeds recommended limits',
                    'severity': IssueSeverity.WARNING,
                    'suggestion': 'Consider circuit optimization to reduce depth'
                }
            ],
            'performance': [
                {
                    'id': 'P001',
                    'name': 'Unnecessary Classical Loop',
                    'description': 'Classical loop that could be vectorized',
                    'severity': IssueSeverity.WARNING,
                    'suggestion': 'Consider using vectorized operations'
                },
                {
                    'id': 'P002',
                    'name': 'Memory Inefficient State Storage',
                    'description': 'Inefficient quantum state storage',
                    'severity': IssueSeverity.WARNING,
                    'suggestion': 'Use sparse representations for large state vectors'
                }
            ],
            'style': [
                {
                    'id': 'S001',
                    'name': 'Quantum Variable Naming',
                    'description': 'Non-descriptive quantum variable names',
                    'severity': IssueSeverity.INFO,
                    'suggestion': 'Use descriptive names for quantum registers and circuits'
                }
            ]
        }
    
    def _setup_metrics_calculators(self) -> Dict[str, Callable]:
        """Setup metrics calculators."""
        return {
            'quantum_depth': self._calculate_quantum_depth,
            'gate_count': self._calculate_gate_count,
            'qubit_efficiency': self._calculate_qubit_efficiency,
            'entanglement_complexity': self._calculate_entanglement_complexity,
            'classical_complexity': self._calculate_classical_complexity,
            'maintainability': self._calculate_maintainability
        }
    
    def _setup_pattern_detectors(self) -> Dict[str, Callable]:
        """Setup pattern detectors."""
        return {
            'vqe_pattern': self._detect_vqe_pattern,
            'qaoa_pattern': self._detect_qaoa_pattern,
            'grover_pattern': self._detect_grover_pattern,
            'teleportation_pattern': self._detect_teleportation_pattern,
            'anti_patterns': self._detect_anti_patterns
        }
    
    def _load_optimization_rules(self) -> List[Dict[str, Any]]:
        """Load optimization rules."""
        return [
            {
                'id': 'OPT001',
                'name': 'Gate Fusion',
                'description': 'Combine adjacent single-qubit gates',
                'pattern': r'\.rx\([^)]+\)\.ry\([^)]+\)',
                'impact': 'Reduces gate count and execution time',
                'effort': 'low'
            },
            {
                'id': 'OPT002',
                'name': 'Qubit Reuse',
                'description': 'Reuse qubits to reduce circuit width',
                'impact': 'Reduces hardware requirements',
                'effort': 'medium'
            },
            {
                'id': 'OPT003',
                'name': 'Circuit Depth Reduction',
                'description': 'Parallelize commuting gates',
                'impact': 'Reduces circuit depth and decoherence effects',
                'effort': 'high'
            }
        ]
    
    def _analyze_syntax(self, tree: ast.AST, content: str) -> List[AnalysisIssue]:
        """Analyze syntax issues."""
        issues = []
        
        # Basic syntax validation through AST parsing
        # Additional quantum-specific syntax checks can be added here
        
        return issues
    
    def _analyze_semantics(self, tree: ast.AST, content: str) -> List[AnalysisIssue]:
        """Analyze semantic issues."""
        issues = []
        
        # Check for quantum-specific semantic issues
        quantum_imports = self.parser.extract_quantum_imports(tree)
        circuits = self.parser.find_circuit_definitions(tree)
        
        # Check for missing measurements
        for circuit in circuits:
            if not any('measure' in op for op in circuit['quantum_operations']):
                issues.append(AnalysisIssue(
                    issue_id="missing_measurement",
                    title="Missing Measurement",
                    description="Quantum circuit without measurement operations",
                    severity=IssueSeverity.INFO,
                    analysis_type=AnalysisType.QUANTUM_SPECIFIC,
                    location=CodeLocation(
                        file_path="",
                        line_number=circuit['line'],
                        column_number=0
                    ),
                    suggestion="Add measurement operations to extract classical information"
                ))
        
        return issues
    
    def _analyze_performance(self, tree: ast.AST, content: str) -> List[AnalysisIssue]:
        """Analyze performance issues."""
        issues = []
        
        # Detect performance anti-patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for inefficient classical loops
                if self._is_inefficient_loop(node):
                    issues.append(AnalysisIssue(
                        issue_id="inefficient_loop",
                        title="Inefficient Classical Loop",
                        description="Loop that could be vectorized for better performance",
                        severity=IssueSeverity.WARNING,
                        analysis_type=AnalysisType.PERFORMANCE,
                        location=CodeLocation(
                            file_path="",
                            line_number=node.lineno,
                            column_number=node.col_offset
                        ),
                        suggestion="Consider using vectorized operations or NumPy arrays"
                    ))
        
        return issues
    
    def _deep_analysis(self, tree: ast.AST, content: str) -> List[AnalysisIssue]:
        """Perform deep analysis."""
        issues = []
        
        # Advanced quantum algorithm analysis
        algorithms = self.parser.analyze_algorithm_patterns(tree, content)
        
        for algorithm in algorithms:
            # Algorithm-specific analysis
            if algorithm == 'vqe':
                issues.extend(self._analyze_vqe_implementation(tree, content))
            elif algorithm == 'qaoa':
                issues.extend(self._analyze_qaoa_implementation(tree, content))
        
        return issues
    
    def _calculate_basic_metrics(self, tree: ast.AST, content: str) -> List[CodeMetric]:
        """Calculate basic code metrics."""
        metrics = []
        
        # Lines of code
        lines = content.split('\n')
        loc = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        
        metrics.append(CodeMetric(
            name="Lines of Code",
            metric_type=MetricType.CLASSICAL_COMPLEXITY,
            value=float(loc),
            description="Total lines of code excluding comments and blank lines",
            unit="lines"
        ))
        
        # Function count
        function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        
        metrics.append(CodeMetric(
            name="Function Count",
            metric_type=MetricType.CLASSICAL_COMPLEXITY,
            value=float(function_count),
            description="Total number of functions",
            unit="functions"
        ))
        
        return metrics
    
    def _calculate_quantum_metrics(self, tree: ast.AST, content: str) -> List[CodeMetric]:
        """Calculate quantum-specific metrics."""
        metrics = []
        
        circuits = self.parser.find_circuit_definitions(tree)
        
        if circuits:
            # Average quantum operations per circuit
            total_ops = sum(len(circuit['quantum_operations']) for circuit in circuits)
            avg_ops = total_ops / len(circuits) if circuits else 0
            
            metrics.append(CodeMetric(
                name="Average Quantum Operations",
                metric_type=MetricType.GATE_COUNT,
                value=avg_ops,
                description="Average number of quantum operations per circuit",
                unit="operations"
            ))
            
            # Circuit density
            total_lines = sum(self._estimate_circuit_lines(circuit) for circuit in circuits)
            density = total_ops / total_lines if total_lines > 0 else 0
            
            metrics.append(CodeMetric(
                name="Quantum Code Density",
                metric_type=MetricType.QUANTUM_DEPTH,
                value=density,
                description="Ratio of quantum operations to total circuit code lines",
                optimal_range=(0.1, 0.8),
                unit="ops/line"
            ))
        
        return metrics
    
    def _calculate_advanced_metrics(self, tree: ast.AST, content: str) -> List[CodeMetric]:
        """Calculate advanced metrics."""
        metrics = []
        
        # Quantum algorithm complexity
        algorithms = self.parser.analyze_algorithm_patterns(tree, content)
        
        if algorithms:
            complexity_score = self._calculate_algorithm_complexity(algorithms)
            
            metrics.append(CodeMetric(
                name="Algorithm Complexity",
                metric_type=MetricType.ENTANGLEMENT_COMPLEXITY,
                value=complexity_score,
                description="Estimated complexity of quantum algorithms used",
                optimal_range=(0.0, 1.0),
                unit="complexity"
            ))
        
        return metrics
    
    def _detect_patterns(self, tree: ast.AST, content: str) -> List[QuantumPattern]:
        """Detect quantum code patterns."""
        patterns = []
        
        # Detect algorithm patterns
        algorithms = self.parser.analyze_algorithm_patterns(tree, content)
        
        for algorithm in algorithms:
            pattern = QuantumPattern(
                pattern_id=f"algo_{algorithm}",
                pattern_type=PatternType.ALGORITHM_PATTERN,
                name=algorithm.upper(),
                description=f"Detected {algorithm} quantum algorithm implementation",
                location=CodeLocation(file_path="", line_number=1),
                confidence=0.8,
                impact="Recognized quantum algorithm pattern",
                recommendation="Ensure proper implementation of algorithm components"
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_advanced_patterns(self, tree: ast.AST, content: str) -> List[QuantumPattern]:
        """Detect advanced quantum patterns."""
        patterns = []
        
        # Detect optimization opportunities
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if self._is_optimization_opportunity(node, content):
                    pattern = QuantumPattern(
                        pattern_id="optimization_opportunity",
                        pattern_type=PatternType.OPTIMIZATION_OPPORTUNITY,
                        name="Gate Optimization Opportunity",
                        description="Detected potential for gate sequence optimization",
                        location=CodeLocation(
                            file_path="",
                            line_number=node.lineno,
                            column_number=node.col_offset
                        ),
                        confidence=0.7,
                        recommendation="Consider optimizing gate sequence for better performance"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _suggest_optimizations(self, tree: ast.AST, content: str) -> List[OptimizationSuggestion]:
        """Suggest code optimizations."""
        optimizations = []
        
        # Detect gate fusion opportunities
        gate_sequences = self._find_gate_sequences(tree)
        
        for sequence in gate_sequences:
            if self._can_fuse_gates(sequence):
                optimization = OptimizationSuggestion(
                    suggestion_id="gate_fusion",
                    title="Gate Fusion Opportunity",
                    description="Adjacent single-qubit gates can be combined",
                    location=CodeLocation(
                        file_path="",
                        line_number=sequence.get('line', 1)
                    ),
                    impact="Reduces gate count and execution time",
                    effort="low",
                    potential_improvement="10-30% reduction in gate count",
                    implementation_hints=[
                        "Combine rotation gates using matrix multiplication",
                        "Use circuit optimization libraries",
                        "Verify equivalence with original circuit"
                    ]
                )
                optimizations.append(optimization)
        
        return optimizations
    
    def _generate_summary(self, report: AnalysisReport) -> Dict[str, Any]:
        """Generate analysis summary."""
        summary = {
            'total_issues': len(report.issues),
            'critical_issues': len([i for i in report.issues if i.severity == IssueSeverity.CRITICAL]),
            'error_issues': len([i for i in report.issues if i.severity == IssueSeverity.ERROR]),
            'warning_issues': len([i for i in report.issues if i.severity == IssueSeverity.WARNING]),
            'info_issues': len([i for i in report.issues if i.severity == IssueSeverity.INFO]),
            'total_metrics': len(report.metrics),
            'patterns_detected': len(report.patterns),
            'optimizations_available': len(report.optimizations),
            'analysis_types': list(set(issue.analysis_type.value for issue in report.issues))
        }
        
        # Calculate quality score
        total_issues = summary['total_issues']
        if total_issues == 0:
            quality_score = 1.0
        else:
            weighted_score = (
                summary['critical_issues'] * 1.0 +
                summary['error_issues'] * 0.8 +
                summary['warning_issues'] * 0.4 +
                summary['info_issues'] * 0.1
            ) / total_issues
            quality_score = max(0.0, 1.0 - weighted_score)
        
        summary['quality_score'] = quality_score
        
        return summary
    
    # Helper methods
    def _is_inefficient_loop(self, node: ast.For) -> bool:
        """Check if loop is inefficient."""
        # Simple heuristic - can be enhanced
        return True  # Placeholder
    
    def _analyze_vqe_implementation(self, tree: ast.AST, content: str) -> List[AnalysisIssue]:
        """Analyze VQE implementation."""
        issues = []
        # VQE-specific analysis logic
        return issues
    
    def _analyze_qaoa_implementation(self, tree: ast.AST, content: str) -> List[AnalysisIssue]:
        """Analyze QAOA implementation."""
        issues = []
        # QAOA-specific analysis logic
        return issues
    
    def _estimate_circuit_lines(self, circuit: Dict[str, Any]) -> int:
        """Estimate lines of code for a circuit."""
        return max(10, len(circuit['quantum_operations']) * 2)  # Rough estimate
    
    def _calculate_algorithm_complexity(self, algorithms: List[str]) -> float:
        """Calculate algorithm complexity score."""
        complexity_scores = {
            'vqe': 0.7,
            'qaoa': 0.6,
            'grover': 0.8,
            'shor': 1.0,
            'teleportation': 0.3,
            'superdense': 0.2
        }
        
        if not algorithms:
            return 0.0
        
        return max(complexity_scores.get(algo, 0.5) for algo in algorithms)
    
    def _is_optimization_opportunity(self, node: ast.Call, content: str) -> bool:
        """Check if node represents optimization opportunity."""
        # Placeholder logic
        return False
    
    def _find_gate_sequences(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find gate sequences in the code."""
        sequences = []
        # Placeholder implementation
        return sequences
    
    def _can_fuse_gates(self, sequence: Dict[str, Any]) -> bool:
        """Check if gates in sequence can be fused."""
        # Placeholder logic
        return True
    
    # Additional calculator methods
    def _calculate_quantum_depth(self, tree: ast.AST, content: str) -> float:
        """Calculate quantum circuit depth."""
        return 0.0  # Placeholder
    
    def _calculate_gate_count(self, tree: ast.AST, content: str) -> float:
        """Calculate total gate count."""
        return 0.0  # Placeholder
    
    def _calculate_qubit_efficiency(self, tree: ast.AST, content: str) -> float:
        """Calculate qubit usage efficiency."""
        return 0.0  # Placeholder
    
    def _calculate_entanglement_complexity(self, tree: ast.AST, content: str) -> float:
        """Calculate entanglement complexity."""
        return 0.0  # Placeholder
    
    def _calculate_classical_complexity(self, tree: ast.AST, content: str) -> float:
        """Calculate classical code complexity."""
        return 0.0  # Placeholder
    
    def _calculate_maintainability(self, tree: ast.AST, content: str) -> float:
        """Calculate code maintainability score."""
        return 0.0  # Placeholder
    
    # Pattern detectors
    def _detect_vqe_pattern(self, tree: ast.AST, content: str) -> List[QuantumPattern]:
        """Detect VQE patterns."""
        return []  # Placeholder
    
    def _detect_qaoa_pattern(self, tree: ast.AST, content: str) -> List[QuantumPattern]:
        """Detect QAOA patterns."""
        return []  # Placeholder
    
    def _detect_grover_pattern(self, tree: ast.AST, content: str) -> List[QuantumPattern]:
        """Detect Grover's algorithm patterns."""
        return []  # Placeholder
    
    def _detect_teleportation_pattern(self, tree: ast.AST, content: str) -> List[QuantumPattern]:
        """Detect quantum teleportation patterns."""
        return []  # Placeholder
    
    def _detect_anti_patterns(self, tree: ast.AST, content: str) -> List[QuantumPattern]:
        """Detect quantum anti-patterns."""
        return []  # Placeholder


class CodeQualityReporter:
    """Generate reports for code analysis results."""
    
    def __init__(self):
        """Initialize code quality reporter."""
        self.report_templates = self._load_report_templates()
    
    def generate_report(self, reports: Dict[str, AnalysisReport], format: str = "json") -> str:
        """Generate comprehensive analysis report."""
        if format == "json":
            return self._generate_json_report(reports)
        elif format == "html":
            return self._generate_html_report(reports)
        elif format == "text":
            return self._generate_text_report(reports)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_json_report(self, reports: Dict[str, AnalysisReport]) -> str:
        """Generate JSON report."""
        report_data = {
            'analysis_timestamp': time.time(),
            'total_files': len(reports),
            'files': {}
        }
        
        for file_path, report in reports.items():
            report_data['files'][file_path] = asdict(report)
        
        # Add summary statistics
        report_data['summary'] = self._calculate_overall_summary(reports)
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_html_report(self, reports: Dict[str, AnalysisReport]) -> str:
        """Generate HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quantum Code Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background: #f8f9fa; padding: 20px; border-radius: 5px; }
                .summary { background: #e9ecef; padding: 15px; margin: 20px 0; border-radius: 5px; }
                .file-section { margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }
                .file-header { background: #007bff; color: white; padding: 10px; }
                .issues { margin: 10px; }
                .issue { margin: 10px 0; padding: 10px; border-left: 4px solid #ffc107; background: #fff3cd; }
                .critical { border-left-color: #dc3545; background: #f8d7da; }
                .error { border-left-color: #fd7e14; background: #ffeaa7; }
                .warning { border-left-color: #ffc107; background: #fff3cd; }
                .info { border-left-color: #17a2b8; background: #d1ecf1; }
                .metrics { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px; }
                .metric { background: #f8f9fa; padding: 10px; border-radius: 5px; min-width: 150px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Quantum Code Analysis Report</h1>
                <p>Generated on: {{ timestamp }}</p>
                <p>Total files analyzed: {{ total_files }}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Issues: {{ total_issues }}</p>
                <p>Critical: {{ critical_issues }}, Errors: {{ error_issues }}, Warnings: {{ warning_issues }}, Info: {{ info_issues }}</p>
                <p>Average Quality Score: {{ avg_quality_score:.2f }}</p>
            </div>
            
            {% for file_path, report in files.items() %}
            <div class="file-section">
                <div class="file-header">
                    <h3>{{ file_path }}</h3>
                    <p>Quality Score: {{ report.summary.quality_score:.2f }} | Issues: {{ report.summary.total_issues }} | Execution Time: {{ report.execution_time:.2f }}s</p>
                </div>
                
                {% if report.issues %}
                <div class="issues">
                    <h4>Issues</h4>
                    {% for issue in report.issues %}
                    <div class="issue {{ issue.severity.value }}">
                        <strong>{{ issue.title }}</strong> ({{ issue.severity.value.upper() }})
                        <p>{{ issue.description }}</p>
                        {% if issue.suggestion %}
                        <p><em>Suggestion: {{ issue.suggestion }}</em></p>
                        {% endif %}
                        <small>Line {{ issue.location.line_number }}, {{ issue.analysis_type.value }}</small>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                {% if report.metrics %}
                <div class="metrics">
                    <h4>Metrics</h4>
                    {% for metric in report.metrics %}
                    <div class="metric">
                        <strong>{{ metric.name }}</strong><br>
                        {{ metric.value }} {{ metric.unit }}<br>
                        <small>{{ metric.description }}</small>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </body>
        </html>
        """
        
        # Simple template rendering (in production, use Jinja2)
        summary = self._calculate_overall_summary(reports)
        
        html = html_template.replace("{{ timestamp }}", time.strftime("%Y-%m-%d %H:%M:%S"))
        html = html.replace("{{ total_files }}", str(len(reports)))
        html = html.replace("{{ total_issues }}", str(summary['total_issues']))
        html = html.replace("{{ critical_issues }}", str(summary['critical_issues']))
        html = html.replace("{{ error_issues }}", str(summary['error_issues']))
        html = html.replace("{{ warning_issues }}", str(summary['warning_issues']))
        html = html.replace("{{ info_issues }}", str(summary['info_issues']))
        html = html.replace("{{ avg_quality_score }}", f"{summary['avg_quality_score']:.2f}")
        
        return html
    
    def _generate_text_report(self, reports: Dict[str, AnalysisReport]) -> str:
        """Generate text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("QUANTUM CODE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Files analyzed: {len(reports)}")
        lines.append("")
        
        # Summary
        summary = self._calculate_overall_summary(reports)
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Issues: {summary['total_issues']}")
        lines.append(f"  Critical: {summary['critical_issues']}")
        lines.append(f"  Errors: {summary['error_issues']}")
        lines.append(f"  Warnings: {summary['warning_issues']}")
        lines.append(f"  Info: {summary['info_issues']}")
        lines.append(f"Average Quality Score: {summary['avg_quality_score']:.2f}")
        lines.append("")
        
        # File details
        for file_path, report in reports.items():
            lines.append(f"FILE: {file_path}")
            lines.append("-" * 60)
            lines.append(f"Quality Score: {report.summary['quality_score']:.2f}")
            lines.append(f"Execution Time: {report.execution_time:.2f}s")
            lines.append("")
            
            if report.issues:
                lines.append("Issues:")
                for issue in report.issues:
                    lines.append(f"  [{issue.severity.value.upper()}] {issue.title}")
                    lines.append(f"    Line {issue.location.line_number}: {issue.description}")
                    if issue.suggestion:
                        lines.append(f"    Suggestion: {issue.suggestion}")
                    lines.append("")
            
            if report.metrics:
                lines.append("Metrics:")
                for metric in report.metrics:
                    lines.append(f"  {metric.name}: {metric.value} {metric.unit}")
                lines.append("")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _calculate_overall_summary(self, reports: Dict[str, AnalysisReport]) -> Dict[str, Any]:
        """Calculate overall summary statistics."""
        total_issues = sum(len(report.issues) for report in reports.values())
        critical_issues = sum(len([i for i in report.issues if i.severity == IssueSeverity.CRITICAL]) for report in reports.values())
        error_issues = sum(len([i for i in report.issues if i.severity == IssueSeverity.ERROR]) for report in reports.values())
        warning_issues = sum(len([i for i in report.issues if i.severity == IssueSeverity.WARNING]) for report in reports.values())
        info_issues = sum(len([i for i in report.issues if i.severity == IssueSeverity.INFO]) for report in reports.values())
        
        quality_scores = [report.summary['quality_score'] for report in reports.values() if 'quality_score' in report.summary]
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        return {
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'error_issues': error_issues,
            'warning_issues': warning_issues,
            'info_issues': info_issues,
            'avg_quality_score': avg_quality_score,
            'total_files': len(reports)
        }
    
    def _load_report_templates(self) -> Dict[str, str]:
        """Load report templates."""
        return {
            'html': 'html_template.html',
            'text': 'text_template.txt',
            'json': 'json_template.json'
        }


class QuantumCodeAnalysisManager:
    """Main manager for quantum code analysis operations."""
    
    def __init__(self, workspace_dir: str = "./quantum_analysis"):
        """Initialize quantum code analysis manager."""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.analyzer = QuantumCodeAnalyzer()
        self.reporter = CodeQualityReporter()
        
        # Configuration
        self.config = self._load_configuration()
        
        # Database for analysis history
        self.db_path = self.workspace_dir / "analysis_history.db"
        self._init_database()
        
        # Analysis cache
        self.cache_dir = self.workspace_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def analyze_project(self, project_path: str, analysis_level: AnalysisLevel = AnalysisLevel.STANDARD,
                       output_format: str = "json", output_file: str = None) -> str:
        """Analyze entire project."""
        start_time = time.time()
        
        try:
            # Analyze all files
            reports = self.analyzer.analyze_directory(project_path, analysis_level)
            
            # Generate report
            report_content = self.reporter.generate_report(reports, output_format)
            
            # Save report
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(report_content)
            
            # Store analysis history
            self._store_analysis_history(project_path, reports, time.time() - start_time)
            
            return report_content
            
        except Exception as e:
            logging.error(f"Project analysis failed: {e}")
            raise
    
    def analyze_file(self, file_path: str, analysis_level: AnalysisLevel = AnalysisLevel.STANDARD) -> AnalysisReport:
        """Analyze single file."""
        return self.analyzer.analyze_file(file_path, analysis_level)
    
    def get_analysis_history(self, project_path: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get analysis history."""
        with sqlite3.connect(self.db_path) as conn:
            if project_path:
                cursor = conn.execute("""
                    SELECT project_path, analysis_timestamp, total_files, total_issues, avg_quality_score, execution_time
                    FROM analysis_history
                    WHERE project_path = ?
                    ORDER BY analysis_timestamp DESC
                    LIMIT ?
                """, (project_path, limit))
            else:
                cursor = conn.execute("""
                    SELECT project_path, analysis_timestamp, total_files, total_issues, avg_quality_score, execution_time
                    FROM analysis_history
                    ORDER BY analysis_timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'project_path': row[0],
                    'analysis_timestamp': row[1],
                    'total_files': row[2],
                    'total_issues': row[3],
                    'avg_quality_score': row[4],
                    'execution_time': row[5]
                })
            
            return history
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Total analyses
            cursor = conn.execute("SELECT COUNT(*) FROM analysis_history")
            total_analyses = cursor.fetchone()[0]
            
            # Average quality score
            cursor = conn.execute("SELECT AVG(avg_quality_score) FROM analysis_history")
            avg_quality = cursor.fetchone()[0] or 0.0
            
            # Most analyzed projects
            cursor = conn.execute("""
                SELECT project_path, COUNT(*) as analysis_count
                FROM analysis_history
                GROUP BY project_path
                ORDER BY analysis_count DESC
                LIMIT 5
            """)
            top_projects = cursor.fetchall()
            
            return {
                'total_analyses': total_analyses,
                'average_quality_score': avg_quality,
                'top_analyzed_projects': [{'project': p[0], 'count': p[1]} for p in top_projects]
            }
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load analysis configuration."""
        config_file = self.workspace_dir / "config.yml"
        
        default_config = {
            'analysis_level': 'standard',
            'output_format': 'json',
            'cache_enabled': True,
            'parallel_analysis': True,
            'custom_rules': []
        }
        
        if config_file.exists() and HAS_YAML:
            try:
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                logging.warning(f"Failed to load configuration: {e}")
        
        return default_config
    
    def _init_database(self):
        """Initialize analysis history database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_path TEXT NOT NULL,
                    analysis_timestamp REAL NOT NULL,
                    total_files INTEGER,
                    total_issues INTEGER,
                    avg_quality_score REAL,
                    execution_time REAL,
                    analysis_level TEXT,
                    report_data TEXT
                )
            """)
    
    def _store_analysis_history(self, project_path: str, reports: Dict[str, AnalysisReport], execution_time: float):
        """Store analysis history."""
        summary = self.reporter._calculate_overall_summary(reports)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO analysis_history 
                (project_path, analysis_timestamp, total_files, total_issues, avg_quality_score, execution_time, analysis_level, report_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project_path,
                time.time(),
                summary['total_files'],
                summary['total_issues'],
                summary['avg_quality_score'],
                execution_time,
                'standard',  # Default level for now
                json.dumps(summary)
            ))


# Web interface for analysis results
if HAS_FLASK:
    class AnalysisWebInterface:
        """Web interface for quantum code analysis."""
        
        def __init__(self, manager: QuantumCodeAnalysisManager):
            """Initialize web interface."""
            self.manager = manager
            self.app = Flask(__name__)
            self._setup_routes()
        
        def _setup_routes(self):
            """Setup Flask routes."""
            
            @self.app.route('/')
            def dashboard():
                """Main dashboard."""
                stats = self.manager.get_analysis_statistics()
                return jsonify(stats)
            
            @self.app.route('/analyze', methods=['POST'])
            def analyze():
                """Analyze project endpoint."""
                data = request.get_json()
                project_path = data.get('project_path')
                analysis_level = data.get('analysis_level', 'standard')
                
                try:
                    level = AnalysisLevel(analysis_level)
                    report = self.manager.analyze_project(project_path, level)
                    return jsonify({'success': True, 'report': report})
                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)}), 500
            
            @self.app.route('/history')
            def history():
                """Analysis history."""
                project_path = request.args.get('project_path')
                limit = int(request.args.get('limit', 10))
                
                history = self.manager.get_analysis_history(project_path, limit)
                return jsonify(history)
        
        def run(self, host: str = "localhost", port: int = 8080, debug: bool = False):
            """Run web interface."""
            self.app.run(host=host, port=port, debug=debug)


# CLI Interface
if HAS_CLICK:
    @click.group()
    @click.option('--workspace', default='./quantum_analysis', help='Analysis workspace directory')
    @click.pass_context
    def cli(ctx, workspace):
        """QuantRS2 Quantum Code Analysis CLI."""
        ctx.ensure_object(dict)
        ctx.obj['manager'] = QuantumCodeAnalysisManager(workspace)
    
    @cli.command()
    @click.argument('project_path')
    @click.option('--level', type=click.Choice(['basic', 'standard', 'comprehensive', 'deep']), 
                  default='standard', help='Analysis depth level')
    @click.option('--format', 'output_format', type=click.Choice(['json', 'html', 'text']), 
                  default='json', help='Output format')
    @click.option('--output', help='Output file path')
    @click.pass_context
    def analyze(ctx, project_path, level, output_format, output):
        """Analyze quantum code project."""
        manager = ctx.obj['manager']
        
        click.echo(f"Analyzing project: {project_path}")
        click.echo(f"Analysis level: {level}")
        
        try:
            analysis_level = AnalysisLevel(level)
            report = manager.analyze_project(project_path, analysis_level, output_format, output)
            
            if output:
                click.echo(f"✓ Analysis report saved to: {output}")
            else:
                click.echo("Analysis completed:")
                if output_format == 'json':
                    click.echo(report)
                else:
                    click.echo(report[:1000] + "..." if len(report) > 1000 else report)
                    
        except Exception as e:
            click.echo(f"✗ Analysis failed: {e}", err=True)
    
    @cli.command()
    @click.argument('file_path')
    @click.option('--level', type=click.Choice(['basic', 'standard', 'comprehensive', 'deep']), 
                  default='standard', help='Analysis depth level')
    @click.pass_context
    def analyze_file(ctx, file_path, level):
        """Analyze single quantum code file."""
        manager = ctx.obj['manager']
        
        try:
            analysis_level = AnalysisLevel(level)
            report = manager.analyze_file(file_path, analysis_level)
            
            click.echo(f"File: {file_path}")
            click.echo(f"Quality Score: {report.summary.get('quality_score', 0):.2f}")
            click.echo(f"Issues: {len(report.issues)}")
            click.echo(f"Metrics: {len(report.metrics)}")
            click.echo(f"Patterns: {len(report.patterns)}")
            click.echo(f"Optimizations: {len(report.optimizations)}")
            
            if report.issues:
                click.echo("\nIssues:")
                for issue in report.issues[:5]:  # Show first 5 issues
                    click.echo(f"  [{issue.severity.value.upper()}] {issue.title}")
                    click.echo(f"    Line {issue.location.line_number}: {issue.description}")
                
                if len(report.issues) > 5:
                    click.echo(f"  ... and {len(report.issues) - 5} more issues")
                    
        except Exception as e:
            click.echo(f"✗ File analysis failed: {e}", err=True)
    
    @cli.command()
    @click.option('--project', help='Filter by project path')
    @click.option('--limit', default=10, help='Number of entries to show')
    @click.pass_context
    def history(ctx, project, limit):
        """Show analysis history."""
        manager = ctx.obj['manager']
        
        history = manager.get_analysis_history(project, limit)
        
        if not history:
            click.echo("No analysis history found")
            return
        
        click.echo("Analysis History:")
        click.echo("-" * 80)
        
        for entry in history:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['analysis_timestamp']))
            click.echo(f"Project: {entry['project_path']}")
            click.echo(f"Date: {timestamp}")
            click.echo(f"Files: {entry['total_files']}, Issues: {entry['total_issues']}")
            click.echo(f"Quality Score: {entry['avg_quality_score']:.2f}")
            click.echo(f"Execution Time: {entry['execution_time']:.2f}s")
            click.echo("-" * 80)
    
    @cli.command()
    @click.pass_context
    def stats(ctx):
        """Show analysis statistics."""
        manager = ctx.obj['manager']
        
        stats = manager.get_analysis_statistics()
        
        click.echo("Analysis Statistics:")
        click.echo("-" * 40)
        click.echo(f"Total Analyses: {stats['total_analyses']}")
        click.echo(f"Average Quality Score: {stats['average_quality_score']:.2f}")
        
        if stats['top_analyzed_projects']:
            click.echo("\nTop Analyzed Projects:")
            for project in stats['top_analyzed_projects']:
                click.echo(f"  {project['project']}: {project['count']} analyses")
    
    @cli.command()
    @click.option('--host', default='localhost', help='Host to bind to')
    @click.option('--port', default=8080, help='Port to bind to')
    @click.option('--debug', is_flag=True, help='Enable debug mode')
    @click.pass_context
    def web(ctx, host, port, debug):
        """Start web interface for analysis results."""
        if not HAS_FLASK:
            click.echo("Flask not available for web interface", err=True)
            return
        
        manager = ctx.obj['manager']
        web_interface = AnalysisWebInterface(manager)
        
        click.echo(f"Starting web interface at http://{host}:{port}")
        web_interface.run(host, port, debug)


# Convenience functions
def get_quantum_code_analysis_manager(workspace_dir: str = "./quantum_analysis") -> QuantumCodeAnalysisManager:
    """Get a quantum code analysis manager instance."""
    return QuantumCodeAnalysisManager(workspace_dir)


def analyze_quantum_code(file_path: str, analysis_level: AnalysisLevel = AnalysisLevel.STANDARD) -> AnalysisReport:
    """Analyze quantum code file."""
    analyzer = QuantumCodeAnalyzer()
    return analyzer.analyze_file(file_path, analysis_level)


def analyze_quantum_project(project_path: str, analysis_level: AnalysisLevel = AnalysisLevel.STANDARD,
                           output_format: str = "json") -> str:
    """Analyze quantum code project."""
    manager = QuantumCodeAnalysisManager()
    return manager.analyze_project(project_path, analysis_level, output_format)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("QuantRS2 Quantum Code Analysis Tools")
    print("=" * 60)
    
    # Initialize analysis manager
    analysis_manager = get_quantum_code_analysis_manager()
    
    print("✅ Quantum Code Analysis Manager initialized successfully!")
    print(f"📁 Workspace: {analysis_manager.workspace_dir}")
    
    # Show statistics
    stats = analysis_manager.get_analysis_statistics()
    print(f"📊 Total analyses performed: {stats['total_analyses']}")
    print(f"🎯 Average quality score: {stats['average_quality_score']:.2f}")
    
    print("\n🚀 Quantum Code Analysis Tools are ready!")
    print("   Use the CLI interface or Python API for code analysis operations")
    
    if HAS_CLICK:
        print("   CLI available: python -m quantrs2.quantum_code_analysis --help")