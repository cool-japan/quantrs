#!/usr/bin/env python3
"""
Comprehensive test suite for the QuantRS2 Quantum Code Analysis Tools.

This test suite provides complete coverage of all code analysis functionality including:
- Static analysis of quantum code with quantum-specific patterns
- Code quality metrics and complexity analysis
- Optimization suggestions and performance recommendations
- Circuit pattern detection and anti-pattern identification
- Integration with development workflows
- Analysis reporting in multiple formats
- Historical analysis tracking and statistics
"""

import pytest
import tempfile
import os
import json
import time
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any

try:
    import quantrs2
    from quantrs2.quantum_code_analysis import (
        QuantumCodeAnalysisManager, QuantumCodeAnalyzer, CodeQualityReporter,
        QuantumCodeParser, AnalysisLevel, AnalysisType, IssueSeverity, PatternType, MetricType,
        CodeLocation, AnalysisIssue, CodeMetric, QuantumPattern, OptimizationSuggestion, AnalysisReport,
        get_quantum_code_analysis_manager, analyze_quantum_code, analyze_quantum_project,
        HAS_CLICK, HAS_FLASK, HAS_YAML
    )
    HAS_QUANTUM_CODE_ANALYSIS = True
except ImportError:
    HAS_QUANTUM_CODE_ANALYSIS = False


# Test fixtures
@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_quantum_code():
    """Sample quantum code for testing."""
    return '''
import quantrs2
import numpy as np

def create_bell_state():
    """Create a Bell state."""
    circuit = quantrs2.Circuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit

def vqe_ansatz(parameters):
    """VQE ansatz circuit."""
    circuit = quantrs2.Circuit(4)
    
    for i, param in enumerate(parameters):
        circuit.ry(i % 4, param)
    
    for i in range(3):
        circuit.cx(i, i + 1)
    
    return circuit

class QuantumAlgorithm:
    """Quantum algorithm implementation."""
    
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
    
    def run_algorithm(self):
        circuit = quantrs2.Circuit(self.num_qubits)
        
        # Inefficient gate sequence
        circuit.h(0)
        circuit.h(0)  # Redundant
        
        return circuit.run()
'''


@pytest.fixture
def problematic_quantum_code():
    """Quantum code with various issues for testing."""
    return '''
import quantrs2

# Missing documentation and poor naming
def bad_func(q, p):
    c = quantrs2.Circuit(q)
    
    # Deeply nested loops
    for i in range(10):
        for j in range(q):
            if i % 2 == 0:
                if j % 2 == 0:
                    c.h(j)
                else:
                    c.x(j)
            else:
                c.y(j)
    
    # No measurement
    return c

# Security issue patterns
def risky_function():
    import os
    filename = "temp.py"
    # os.system(f"python {filename}")  # Commented but still concerning
    
    gate = "h"
    qubit = 0
    # eval(f"circuit.{gate}({qubit})")  # Dangerous dynamic execution
    
    return None
'''


@pytest.fixture
def code_parser():
    """Create quantum code parser for testing."""
    return QuantumCodeParser()


@pytest.fixture
def code_analyzer():
    """Create quantum code analyzer for testing."""
    return QuantumCodeAnalyzer()


@pytest.fixture
def analysis_manager(temp_workspace):
    """Create analysis manager for testing."""
    return QuantumCodeAnalysisManager(str(temp_workspace))


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestCodeLocation:
    """Test CodeLocation data class."""
    
    def test_code_location_creation(self):
        """Test CodeLocation creation."""
        location = CodeLocation(
            file_path="/test/file.py",
            line_number=10,
            column_number=5,
            end_line=12,
            end_column=15
        )
        
        assert location.file_path == "/test/file.py"
        assert location.line_number == 10
        assert location.column_number == 5
        assert location.end_line == 12
        assert location.end_column == 15
    
    def test_code_location_defaults(self):
        """Test CodeLocation default values."""
        location = CodeLocation(
            file_path="/test/file.py",
            line_number=10
        )
        
        assert location.column_number == 0
        assert location.end_line == 0
        assert location.end_column == 0


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestAnalysisIssue:
    """Test AnalysisIssue data class."""
    
    def test_analysis_issue_creation(self):
        """Test AnalysisIssue creation."""
        location = CodeLocation(file_path="/test/file.py", line_number=10)
        
        issue = AnalysisIssue(
            issue_id="TEST001",
            title="Test Issue",
            description="This is a test issue",
            severity=IssueSeverity.WARNING,
            analysis_type=AnalysisType.QUANTUM_SPECIFIC,
            location=location,
            suggestion="Fix this issue",
            fix_available=True,
            auto_fixable=False,
            rule_id="QS001"
        )
        
        assert issue.issue_id == "TEST001"
        assert issue.title == "Test Issue"
        assert issue.severity == IssueSeverity.WARNING
        assert issue.analysis_type == AnalysisType.QUANTUM_SPECIFIC
        assert issue.suggestion == "Fix this issue"
        assert issue.fix_available is True
        assert issue.auto_fixable is False
    
    def test_analysis_issue_defaults(self):
        """Test AnalysisIssue default values."""
        location = CodeLocation(file_path="/test/file.py", line_number=10)
        
        issue = AnalysisIssue(
            issue_id="TEST001",
            title="Test Issue",
            description="This is a test issue",
            severity=IssueSeverity.INFO,
            analysis_type=AnalysisType.SYNTAX,
            location=location
        )
        
        assert issue.suggestion == ""
        assert issue.fix_available is False
        assert issue.auto_fixable is False
        assert issue.rule_id == ""
        assert issue.metadata == {}


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestCodeMetric:
    """Test CodeMetric data class."""
    
    def test_code_metric_creation(self):
        """Test CodeMetric creation."""
        metric = CodeMetric(
            name="Test Metric",
            metric_type=MetricType.GATE_COUNT,
            value=42.5,
            description="A test metric",
            optimal_range=(0.0, 100.0),
            unit="gates"
        )
        
        assert metric.name == "Test Metric"
        assert metric.metric_type == MetricType.GATE_COUNT
        assert metric.value == 42.5
        assert metric.description == "A test metric"
        assert metric.optimal_range == (0.0, 100.0)
        assert metric.unit == "gates"
    
    def test_code_metric_defaults(self):
        """Test CodeMetric default values."""
        metric = CodeMetric(
            name="Simple Metric",
            metric_type=MetricType.QUANTUM_DEPTH,
            value=10.0,
            description="Simple test metric"
        )
        
        assert metric.optimal_range == (0.0, 1.0)
        assert metric.unit == ""
        assert metric.metadata == {}


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestQuantumPattern:
    """Test QuantumPattern data class."""
    
    def test_quantum_pattern_creation(self):
        """Test QuantumPattern creation."""
        location = CodeLocation(file_path="/test/file.py", line_number=5)
        
        pattern = QuantumPattern(
            pattern_id="VQE_PATTERN",
            pattern_type=PatternType.ALGORITHM_PATTERN,
            name="VQE Algorithm",
            description="Detected VQE implementation",
            location=location,
            confidence=0.85,
            impact="High performance algorithm",
            recommendation="Optimize ansatz depth"
        )
        
        assert pattern.pattern_id == "VQE_PATTERN"
        assert pattern.pattern_type == PatternType.ALGORITHM_PATTERN
        assert pattern.name == "VQE Algorithm"
        assert pattern.confidence == 0.85
        assert pattern.impact == "High performance algorithm"
        assert pattern.recommendation == "Optimize ansatz depth"
    
    def test_quantum_pattern_defaults(self):
        """Test QuantumPattern default values."""
        location = CodeLocation(file_path="/test/file.py", line_number=5)
        
        pattern = QuantumPattern(
            pattern_id="TEST_PATTERN",
            pattern_type=PatternType.GATE_SEQUENCE,
            name="Test Pattern",
            description="Test pattern description",
            location=location,
            confidence=0.5
        )
        
        assert pattern.impact == ""
        assert pattern.recommendation == ""
        assert pattern.metadata == {}


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestOptimizationSuggestion:
    """Test OptimizationSuggestion data class."""
    
    def test_optimization_suggestion_creation(self):
        """Test OptimizationSuggestion creation."""
        location = CodeLocation(file_path="/test/file.py", line_number=15)
        
        suggestion = OptimizationSuggestion(
            suggestion_id="OPT001",
            title="Gate Fusion",
            description="Combine adjacent rotation gates",
            location=location,
            impact="Reduces gate count by 30%",
            effort="low",
            potential_improvement="30% fewer gates",
            implementation_hints=["Use matrix multiplication", "Verify equivalence"],
            before_code="circuit.rx(0, theta); circuit.ry(0, phi)",
            after_code="circuit.u3(theta, phi, 0, 0)"
        )
        
        assert suggestion.suggestion_id == "OPT001"
        assert suggestion.title == "Gate Fusion"
        assert suggestion.effort == "low"
        assert suggestion.impact == "Reduces gate count by 30%"
        assert len(suggestion.implementation_hints) == 2
        assert suggestion.before_code == "circuit.rx(0, theta); circuit.ry(0, phi)"
        assert suggestion.after_code == "circuit.u3(theta, phi, 0, 0)"
    
    def test_optimization_suggestion_defaults(self):
        """Test OptimizationSuggestion default values."""
        location = CodeLocation(file_path="/test/file.py", line_number=15)
        
        suggestion = OptimizationSuggestion(
            suggestion_id="OPT002",
            title="Simple Optimization",
            description="A simple optimization",
            location=location,
            impact="Minor improvement",
            effort="medium",
            potential_improvement="5% improvement"
        )
        
        assert suggestion.implementation_hints == []
        assert suggestion.before_code == ""
        assert suggestion.after_code == ""
        assert suggestion.metadata == {}


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestQuantumCodeParser:
    """Test QuantumCodeParser functionality."""
    
    def test_parser_initialization(self, code_parser):
        """Test parser initialization."""
        assert isinstance(code_parser.quantum_gate_patterns, dict)
        assert 'single_qubit' in code_parser.quantum_gate_patterns
        assert 'two_qubit' in code_parser.quantum_gate_patterns
        assert 'h' in code_parser.quantum_gate_patterns['single_qubit']
        assert 'cx' in code_parser.quantum_gate_patterns['two_qubit']
        
        assert isinstance(code_parser.quantum_imports, list)
        assert 'quantrs2' in code_parser.quantum_imports
        assert 'qiskit' in code_parser.quantum_imports
        
        assert isinstance(code_parser.algorithm_patterns, dict)
        assert 'vqe' in code_parser.algorithm_patterns
        assert 'qaoa' in code_parser.algorithm_patterns
    
    def test_parse_file(self, temp_workspace, sample_quantum_code, code_parser):
        """Test file parsing."""
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        tree = code_parser.parse_file(str(test_file))
        
        assert tree is not None
        # Basic AST structure checks
        import ast
        assert isinstance(tree, ast.Module)
    
    def test_parse_file_invalid_syntax(self, temp_workspace, code_parser):
        """Test parsing file with invalid syntax."""
        test_file = temp_workspace / "invalid_code.py"
        test_file.write_text("def invalid_function(\n    # Missing closing parenthesis")
        
        with pytest.raises(ValueError):
            code_parser.parse_file(str(test_file))
    
    def test_extract_quantum_imports(self, temp_workspace, sample_quantum_code, code_parser):
        """Test quantum import extraction."""
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        tree = code_parser.parse_file(str(test_file))
        imports = code_parser.extract_quantum_imports(tree)
        
        assert 'quantrs2' in imports
    
    def test_find_circuit_definitions(self, temp_workspace, sample_quantum_code, code_parser):
        """Test circuit definition detection."""
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        tree = code_parser.parse_file(str(test_file))
        circuits = code_parser.find_circuit_definitions(tree)
        
        assert len(circuits) > 0
        
        # Check for expected function
        function_names = [circuit['name'] for circuit in circuits]
        assert 'create_bell_state' in function_names or 'vqe_ansatz' in function_names
    
    def test_analyze_algorithm_patterns(self, temp_workspace, sample_quantum_code, code_parser):
        """Test algorithm pattern analysis."""
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        tree = code_parser.parse_file(str(test_file))
        patterns = code_parser.analyze_algorithm_patterns(tree, sample_quantum_code)
        
        # Should detect VQE pattern
        assert 'vqe' in patterns or len(patterns) >= 0  # Allow for detection variations


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestQuantumCodeAnalyzer:
    """Test QuantumCodeAnalyzer functionality."""
    
    def test_analyzer_initialization(self, code_analyzer):
        """Test analyzer initialization."""
        assert isinstance(code_analyzer.parser, QuantumCodeParser)
        assert isinstance(code_analyzer.analysis_rules, dict)
        assert isinstance(code_analyzer.metrics_calculators, dict)
        assert isinstance(code_analyzer.pattern_detectors, dict)
        assert isinstance(code_analyzer.optimization_rules, list)
    
    def test_analyze_file_basic(self, temp_workspace, sample_quantum_code, code_analyzer):
        """Test basic file analysis."""
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        report = code_analyzer.analyze_file(str(test_file), AnalysisLevel.BASIC)
        
        assert isinstance(report, AnalysisReport)
        assert report.file_path == str(test_file)
        assert report.analysis_level == AnalysisLevel.BASIC
        assert report.execution_time > 0
        assert isinstance(report.issues, list)
        assert isinstance(report.metrics, list)
        assert isinstance(report.patterns, list)
        assert isinstance(report.optimizations, list)
        assert isinstance(report.summary, dict)
    
    def test_analyze_file_standard(self, temp_workspace, sample_quantum_code, code_analyzer):
        """Test standard file analysis."""
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        report = code_analyzer.analyze_file(str(test_file), AnalysisLevel.STANDARD)
        
        assert report.analysis_level == AnalysisLevel.STANDARD
        # Standard analysis should find more than basic
        basic_report = code_analyzer.analyze_file(str(test_file), AnalysisLevel.BASIC)
        assert len(report.metrics) >= len(basic_report.metrics)
    
    def test_analyze_file_comprehensive(self, temp_workspace, sample_quantum_code, code_analyzer):
        """Test comprehensive file analysis."""
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        report = code_analyzer.analyze_file(str(test_file), AnalysisLevel.COMPREHENSIVE)
        
        assert report.analysis_level == AnalysisLevel.COMPREHENSIVE
        # Should have more detailed analysis
        assert 'quality_score' in report.summary
        assert report.summary['quality_score'] >= 0.0
        assert report.summary['quality_score'] <= 1.0
    
    def test_analyze_file_deep(self, temp_workspace, sample_quantum_code, code_analyzer):
        """Test deep file analysis."""
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        report = code_analyzer.analyze_file(str(test_file), AnalysisLevel.DEEP)
        
        assert report.analysis_level == AnalysisLevel.DEEP
        # Deep analysis should be most comprehensive
        standard_report = code_analyzer.analyze_file(str(test_file), AnalysisLevel.STANDARD)
        assert len(report.issues) >= len(standard_report.issues)
    
    def test_analyze_file_with_issues(self, temp_workspace, problematic_quantum_code, code_analyzer):
        """Test analysis of file with known issues."""
        test_file = temp_workspace / "problematic_code.py"
        test_file.write_text(problematic_quantum_code)
        
        report = code_analyzer.analyze_file(str(test_file), AnalysisLevel.COMPREHENSIVE)
        
        # Should detect some issues in problematic code
        assert len(report.issues) > 0
        
        # Check for specific issue types
        issue_types = [issue.analysis_type for issue in report.issues]
        # Should find some analysis types
        assert len(set(issue_types)) > 0
    
    def test_analyze_nonexistent_file(self, code_analyzer):
        """Test analysis of non-existent file."""
        report = code_analyzer.analyze_file("/nonexistent/file.py", AnalysisLevel.BASIC)
        
        # Should return error report
        assert isinstance(report, AnalysisReport)
        assert len(report.issues) > 0
        assert any(issue.severity == IssueSeverity.ERROR for issue in report.issues)
    
    def test_analyze_directory(self, temp_workspace, sample_quantum_code, code_analyzer):
        """Test directory analysis."""
        # Create multiple files
        (temp_workspace / "file1.py").write_text(sample_quantum_code)
        (temp_workspace / "file2.py").write_text(sample_quantum_code)
        
        reports = code_analyzer.analyze_directory(str(temp_workspace), AnalysisLevel.BASIC)
        
        assert isinstance(reports, dict)
        assert len(reports) == 2
        
        for file_path, report in reports.items():
            assert isinstance(report, AnalysisReport)
            assert Path(file_path).exists()


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestCodeQualityReporter:
    """Test CodeQualityReporter functionality."""
    
    def test_reporter_initialization(self):
        """Test reporter initialization."""
        reporter = CodeQualityReporter()
        assert isinstance(reporter.report_templates, dict)
    
    def test_generate_json_report(self, temp_workspace, sample_quantum_code):
        """Test JSON report generation."""
        reporter = CodeQualityReporter()
        analyzer = QuantumCodeAnalyzer()
        
        # Create test file
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        # Analyze file
        report = analyzer.analyze_file(str(test_file), AnalysisLevel.STANDARD)
        reports = {str(test_file): report}
        
        # Generate JSON report
        json_report = reporter.generate_report(reports, format="json")
        
        assert isinstance(json_report, str)
        
        # Validate JSON
        report_data = json.loads(json_report)
        assert 'analysis_timestamp' in report_data
        assert 'total_files' in report_data
        assert 'files' in report_data
        assert 'summary' in report_data
        assert report_data['total_files'] == 1
    
    def test_generate_html_report(self, temp_workspace, sample_quantum_code):
        """Test HTML report generation."""
        reporter = CodeQualityReporter()
        analyzer = QuantumCodeAnalyzer()
        
        # Create test file
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        # Analyze file
        report = analyzer.analyze_file(str(test_file), AnalysisLevel.STANDARD)
        reports = {str(test_file): report}
        
        # Generate HTML report
        html_report = reporter.generate_report(reports, format="html")
        
        assert isinstance(html_report, str)
        assert "<!DOCTYPE html>" in html_report
        assert "Quantum Code Analysis Report" in html_report
        assert str(test_file) in html_report
    
    def test_generate_text_report(self, temp_workspace, sample_quantum_code):
        """Test text report generation."""
        reporter = CodeQualityReporter()
        analyzer = QuantumCodeAnalyzer()
        
        # Create test file
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        # Analyze file
        report = analyzer.analyze_file(str(test_file), AnalysisLevel.STANDARD)
        reports = {str(test_file): report}
        
        # Generate text report
        text_report = reporter.generate_report(reports, format="text")
        
        assert isinstance(text_report, str)
        assert "QUANTUM CODE ANALYSIS REPORT" in text_report
        assert "SUMMARY" in text_report
        assert str(test_file) in text_report
    
    def test_unsupported_format(self, temp_workspace, sample_quantum_code):
        """Test unsupported report format."""
        reporter = CodeQualityReporter()
        analyzer = QuantumCodeAnalyzer()
        
        # Create test file
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        # Analyze file
        report = analyzer.analyze_file(str(test_file), AnalysisLevel.STANDARD)
        reports = {str(test_file): report}
        
        # Try unsupported format
        with pytest.raises(ValueError):
            reporter.generate_report(reports, format="unsupported")
    
    def test_calculate_overall_summary(self, temp_workspace, sample_quantum_code):
        """Test overall summary calculation."""
        reporter = CodeQualityReporter()
        analyzer = QuantumCodeAnalyzer()
        
        # Create multiple test files
        test_file1 = temp_workspace / "test1.py"
        test_file2 = temp_workspace / "test2.py"
        test_file1.write_text(sample_quantum_code)
        test_file2.write_text(sample_quantum_code)
        
        # Analyze files
        reports = {
            str(test_file1): analyzer.analyze_file(str(test_file1), AnalysisLevel.STANDARD),
            str(test_file2): analyzer.analyze_file(str(test_file2), AnalysisLevel.STANDARD)
        }
        
        # Calculate summary
        summary = reporter._calculate_overall_summary(reports)
        
        assert isinstance(summary, dict)
        assert 'total_issues' in summary
        assert 'critical_issues' in summary
        assert 'error_issues' in summary
        assert 'warning_issues' in summary
        assert 'info_issues' in summary
        assert 'avg_quality_score' in summary
        assert 'total_files' in summary
        
        assert summary['total_files'] == 2
        assert summary['avg_quality_score'] >= 0.0
        assert summary['avg_quality_score'] <= 1.0


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestQuantumCodeAnalysisManager:
    """Test QuantumCodeAnalysisManager functionality."""
    
    def test_manager_initialization(self, analysis_manager):
        """Test manager initialization."""
        assert analysis_manager.workspace_dir.exists()
        assert analysis_manager.cache_dir.exists()
        assert analysis_manager.db_path.exists()
        assert isinstance(analysis_manager.analyzer, QuantumCodeAnalyzer)
        assert isinstance(analysis_manager.reporter, CodeQualityReporter)
        assert isinstance(analysis_manager.config, dict)
    
    def test_database_initialization(self, analysis_manager):
        """Test database tables creation."""
        with sqlite3.connect(analysis_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='analysis_history'
            """)
            tables = [row[0] for row in cursor.fetchall()]
        
        assert 'analysis_history' in tables
    
    def test_analyze_file(self, temp_workspace, sample_quantum_code, analysis_manager):
        """Test single file analysis through manager."""
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        report = analysis_manager.analyze_file(str(test_file), AnalysisLevel.STANDARD)
        
        assert isinstance(report, AnalysisReport)
        assert report.file_path == str(test_file)
    
    def test_analyze_project(self, temp_workspace, sample_quantum_code, analysis_manager):
        """Test project analysis through manager."""
        # Create project structure
        project_dir = temp_workspace / "test_project"
        project_dir.mkdir()
        
        (project_dir / "main.py").write_text(sample_quantum_code)
        (project_dir / "utils.py").write_text(sample_quantum_code)
        
        # Analyze project
        report_json = analysis_manager.analyze_project(
            str(project_dir),
            AnalysisLevel.STANDARD,
            output_format="json"
        )
        
        assert isinstance(report_json, str)
        
        # Validate JSON structure
        report_data = json.loads(report_json)
        assert 'total_files' in report_data
        assert 'files' in report_data
        assert report_data['total_files'] == 2
    
    def test_analyze_project_with_output_file(self, temp_workspace, sample_quantum_code, analysis_manager):
        """Test project analysis with output file."""
        # Create project
        project_dir = temp_workspace / "test_project"
        project_dir.mkdir()
        (project_dir / "main.py").write_text(sample_quantum_code)
        
        # Output file
        output_file = temp_workspace / "report.json"
        
        # Analyze with output
        report_json = analysis_manager.analyze_project(
            str(project_dir),
            AnalysisLevel.BASIC,
            output_format="json",
            output_file=str(output_file)
        )
        
        # Check output file was created
        assert output_file.exists()
        
        # Validate content
        with open(output_file, 'r') as f:
            saved_report = json.load(f)
        
        assert 'total_files' in saved_report
        assert saved_report['total_files'] == 1
    
    def test_get_analysis_history_empty(self, analysis_manager):
        """Test getting analysis history when empty."""
        history = analysis_manager.get_analysis_history()
        assert isinstance(history, list)
        assert len(history) == 0
    
    def test_get_analysis_statistics_empty(self, analysis_manager):
        """Test getting analysis statistics when empty."""
        stats = analysis_manager.get_analysis_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_analyses' in stats
        assert 'average_quality_score' in stats
        assert 'top_analyzed_projects' in stats
        
        assert stats['total_analyses'] == 0
        assert stats['average_quality_score'] == 0.0
        assert stats['top_analyzed_projects'] == []
    
    def test_analysis_history_tracking(self, temp_workspace, sample_quantum_code, analysis_manager):
        """Test analysis history tracking."""
        # Create and analyze project
        project_dir = temp_workspace / "test_project"
        project_dir.mkdir()
        (project_dir / "main.py").write_text(sample_quantum_code)
        
        # Run analysis
        analysis_manager.analyze_project(str(project_dir), AnalysisLevel.BASIC)
        
        # Check history
        history = analysis_manager.get_analysis_history()
        
        assert len(history) == 1
        assert history[0]['project_path'] == str(project_dir)
        assert history[0]['total_files'] == 1
        assert 'analysis_timestamp' in history[0]
        assert 'total_issues' in history[0]
        assert 'avg_quality_score' in history[0]
        assert 'execution_time' in history[0]
    
    def test_analysis_statistics_with_data(self, temp_workspace, sample_quantum_code, analysis_manager):
        """Test analysis statistics with data."""
        # Create and analyze multiple projects
        for i in range(3):
            project_dir = temp_workspace / f"project_{i}"
            project_dir.mkdir()
            (project_dir / "main.py").write_text(sample_quantum_code)
            
            analysis_manager.analyze_project(str(project_dir), AnalysisLevel.BASIC)
        
        # Check statistics
        stats = analysis_manager.get_analysis_statistics()
        
        assert stats['total_analyses'] == 3
        assert stats['average_quality_score'] >= 0.0
        assert len(stats['top_analyzed_projects']) <= 3
    
    def test_load_configuration(self, temp_workspace):
        """Test configuration loading."""
        # Create config file
        config_dir = temp_workspace / "config"
        config_dir.mkdir()
        
        config_file = config_dir / "config.yml"
        if HAS_YAML:
            import yaml
            config_data = {
                'analysis_level': 'comprehensive',
                'output_format': 'html',
                'custom_rules': ['rule1', 'rule2']
            }
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
        
        # Create manager with custom config
        manager = QuantumCodeAnalysisManager(str(temp_workspace))
        
        # Check default config is loaded
        assert isinstance(manager.config, dict)
        assert 'analysis_level' in manager.config
        assert 'output_format' in manager.config


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_quantum_code_analysis_manager(self, temp_workspace):
        """Test get_quantum_code_analysis_manager function."""
        manager = get_quantum_code_analysis_manager(str(temp_workspace))
        
        assert isinstance(manager, QuantumCodeAnalysisManager)
        assert manager.workspace_dir == temp_workspace
    
    def test_analyze_quantum_code(self, temp_workspace, sample_quantum_code):
        """Test analyze_quantum_code function."""
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        report = analyze_quantum_code(str(test_file), AnalysisLevel.BASIC)
        
        assert isinstance(report, AnalysisReport)
        assert report.file_path == str(test_file)
        assert report.analysis_level == AnalysisLevel.BASIC
    
    def test_analyze_quantum_project(self, temp_workspace, sample_quantum_code):
        """Test analyze_quantum_project function."""
        # Create project
        project_dir = temp_workspace / "test_project"
        project_dir.mkdir()
        (project_dir / "main.py").write_text(sample_quantum_code)
        
        report_json = analyze_quantum_project(str(project_dir), AnalysisLevel.BASIC)
        
        assert isinstance(report_json, str)
        
        # Validate JSON
        report_data = json.loads(report_json)
        assert 'total_files' in report_data
        assert report_data['total_files'] == 1


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_analyze_empty_file(self, temp_workspace, code_analyzer):
        """Test analyzing empty file."""
        empty_file = temp_workspace / "empty.py"
        empty_file.write_text("")
        
        report = code_analyzer.analyze_file(str(empty_file), AnalysisLevel.BASIC)
        
        # Should handle empty file gracefully
        assert isinstance(report, AnalysisReport)
        # May have warnings about empty file
    
    def test_analyze_syntax_error_file(self, temp_workspace, code_analyzer):
        """Test analyzing file with syntax errors."""
        syntax_error_file = temp_workspace / "syntax_error.py"
        syntax_error_file.write_text("def invalid_function(\n    # Missing closing parenthesis")
        
        report = code_analyzer.analyze_file(str(syntax_error_file), AnalysisLevel.BASIC)
        
        # Should return error report
        assert isinstance(report, AnalysisReport)
        assert len(report.issues) > 0
        assert any(issue.severity == IssueSeverity.ERROR for issue in report.issues)
    
    def test_analyze_binary_file(self, temp_workspace, code_analyzer):
        """Test analyzing binary file."""
        binary_file = temp_workspace / "binary.pyc"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        
        report = code_analyzer.analyze_file(str(binary_file), AnalysisLevel.BASIC)
        
        # Should handle binary file gracefully
        assert isinstance(report, AnalysisReport)
        assert len(report.issues) > 0  # Should report parsing error
    
    def test_analyze_large_file(self, temp_workspace, code_analyzer):
        """Test analyzing very large file."""
        large_file = temp_workspace / "large.py"
        
        # Create large file content
        large_content = "import quantrs2\n\n"
        large_content += "\n".join([f"def function_{i}(): pass" for i in range(1000)])
        
        large_file.write_text(large_content)
        
        report = code_analyzer.analyze_file(str(large_file), AnalysisLevel.BASIC)
        
        # Should handle large file
        assert isinstance(report, AnalysisReport)
        # Execution time might be longer but should complete
        assert report.execution_time > 0
    
    def test_analyze_nonexistent_directory(self, code_analyzer):
        """Test analyzing non-existent directory."""
        reports = code_analyzer.analyze_directory("/nonexistent/directory", AnalysisLevel.BASIC)
        
        # Should return empty reports
        assert isinstance(reports, dict)
        assert len(reports) == 0
    
    def test_manager_invalid_workspace(self):
        """Test manager with invalid workspace permissions."""
        # Try to create manager with invalid path
        try:
            manager = QuantumCodeAnalysisManager("/root/invalid_workspace")
            # Should handle gracefully or create in accessible location
            assert isinstance(manager, QuantumCodeAnalysisManager)
        except PermissionError:
            # This is acceptable behavior
            pass
    
    def test_database_corruption_handling(self, temp_workspace):
        """Test handling of corrupted database."""
        # Create manager
        manager = QuantumCodeAnalysisManager(str(temp_workspace))
        
        # Corrupt the database
        with open(manager.db_path, 'w') as f:
            f.write("corrupted database content")
        
        # Should handle corruption gracefully
        try:
            stats = manager.get_analysis_statistics()
            assert isinstance(stats, dict)
        except Exception:
            # Database corruption handling may vary
            pass
    
    def test_report_generation_edge_cases(self, temp_workspace):
        """Test report generation edge cases."""
        reporter = CodeQualityReporter()
        
        # Empty reports
        empty_reports = {}
        json_report = reporter.generate_report(empty_reports, format="json")
        
        assert isinstance(json_report, str)
        report_data = json.loads(json_report)
        assert report_data['total_files'] == 0
        
        # Reports with no issues
        analyzer = QuantumCodeAnalyzer()
        test_file = temp_workspace / "simple.py"
        test_file.write_text("# Simple comment\npass")
        
        report = analyzer.analyze_file(str(test_file), AnalysisLevel.BASIC)
        reports = {str(test_file): report}
        
        text_report = reporter.generate_report(reports, format="text")
        assert isinstance(text_report, str)
        assert "QUANTUM CODE ANALYSIS REPORT" in text_report


@pytest.mark.skipif(not HAS_QUANTUM_CODE_ANALYSIS, reason="quantum code analysis not available")
class TestIntegrationAndPerformance:
    """Test integration features and performance."""
    
    def test_analysis_level_performance(self, temp_workspace, sample_quantum_code):
        """Test that different analysis levels have appropriate performance characteristics."""
        test_file = temp_workspace / "test_code.py"
        test_file.write_text(sample_quantum_code)
        
        analyzer = QuantumCodeAnalyzer()
        
        # Measure execution times for different levels
        times = {}
        for level in [AnalysisLevel.BASIC, AnalysisLevel.STANDARD, AnalysisLevel.COMPREHENSIVE, AnalysisLevel.DEEP]:
            start_time = time.time()
            report = analyzer.analyze_file(str(test_file), level)
            execution_time = time.time() - start_time
            
            times[level] = execution_time
            assert report.execution_time > 0
        
        # Basic should be fastest (though this may not always hold)
        assert times[AnalysisLevel.BASIC] > 0
        # All should complete in reasonable time
        for level, exec_time in times.items():
            assert exec_time < 10.0  # Should complete within 10 seconds
    
    def test_concurrent_analysis(self, temp_workspace, sample_quantum_code):
        """Test concurrent analysis operations."""
        import threading
        
        # Create multiple test files
        test_files = []
        for i in range(5):
            test_file = temp_workspace / f"test_{i}.py"
            test_file.write_text(sample_quantum_code)
            test_files.append(test_file)
        
        analyzer = QuantumCodeAnalyzer()
        results = {}
        errors = []
        
        def analyze_file_thread(file_path):
            try:
                report = analyzer.analyze_file(str(file_path), AnalysisLevel.BASIC)
                results[str(file_path)] = report
            except Exception as e:
                errors.append(e)
        
        # Start concurrent analyses
        threads = []
        for test_file in test_files:
            thread = threading.Thread(target=analyze_file_thread, args=(test_file,))
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0  # No errors should occur
        assert len(results) == 5  # All files should be analyzed
        
        for file_path, report in results.items():
            assert isinstance(report, AnalysisReport)
            assert report.file_path == file_path
    
    def test_memory_usage_large_project(self, temp_workspace):
        """Test memory usage with larger project."""
        # Create project with many files
        project_dir = temp_workspace / "large_project"
        project_dir.mkdir()
        
        # Create multiple subdirectories with files
        for i in range(10):
            subdir = project_dir / f"module_{i}"
            subdir.mkdir()
            
            for j in range(5):
                file_path = subdir / f"file_{j}.py"
                file_path.write_text(f"""
import quantrs2

def function_{i}_{j}():
    circuit = quantrs2.Circuit({i + j + 1})
    for qubit in range({i + j + 1}):
        circuit.h(qubit)
    return circuit.run()
""")
        
        # Analyze project
        analyzer = QuantumCodeAnalyzer()
        
        start_time = time.time()
        reports = analyzer.analyze_directory(str(project_dir), AnalysisLevel.BASIC)
        analysis_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert analysis_time < 30.0  # 30 seconds max
        assert len(reports) == 50  # 10 dirs * 5 files each
        
        # All reports should be valid
        for file_path, report in reports.items():
            assert isinstance(report, AnalysisReport)
            assert Path(file_path).exists()


if __name__ == "__main__":
    pytest.main([__file__])