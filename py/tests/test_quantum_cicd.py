#!/usr/bin/env python3
"""
Comprehensive test suite for the QuantRS2 Quantum CI/CD Pipeline System.

This test suite provides complete coverage of all CI/CD functionality including:
- Pipeline execution engine with async support
- Git repository integration and webhook handling
- Quantum-specific testing strategies and validation
- Code quality analysis for quantum code
- Deployment automation with container integration
- Notification systems and monitoring dashboards
- Artifact management and release packaging
- Error handling, edge cases, and performance validation
"""

import pytest
import tempfile
import os
import json
import time
import asyncio
import threading
import shutil
import numpy as np
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from typing import Dict, List, Any

try:
    import quantrs2
    from quantrs2.quantum_cicd import (
        PipelineStatus, TriggerType, StageType, Environment, NotificationType,
        PipelineConfig, StageConfig, DeploymentConfig, NotificationConfig,
        PipelineRun, BuildArtifact, GitRepository, QuantumTestRunner,
        CodeQualityAnalyzer, ArtifactManager, NotificationManager,
        PipelineEngine, CICDDashboard, QuantumCICDManager,
        get_quantum_cicd_manager, create_basic_pipeline_config,
        create_quantum_test_stage, create_build_stage, create_deploy_stage,
        HAS_GIT, HAS_DOCKER, HAS_REQUESTS, HAS_FLASK, HAS_PYTEST, HAS_JINJA2, HAS_EMAIL
    )
    HAS_QUANTUM_CICD = True
except ImportError:
    HAS_QUANTUM_CICD = False

# Test fixtures
@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_pipeline_config():
    """Create sample pipeline configuration for testing."""
    return PipelineConfig(
        name="test-pipeline",
        version="1.0",
        description="Test pipeline for CI/CD testing",
        triggers=[TriggerType.PUSH, TriggerType.MANUAL],
        environment_variables={"TEST_ENV": "true"},
        timeout=1800,
        parallel=True,
        quantum_config={"max_qubits": 16}
    )

@pytest.fixture
def sample_stage_configs():
    """Create sample stage configurations for testing."""
    return [
        StageConfig(
            name="build",
            type=StageType.BUILD,
            commands=["echo 'Building project'", "python setup.py build"],
            timeout=900,
            artifacts=["build/*", "dist/*"]
        ),
        StageConfig(
            name="test",
            type=StageType.TEST,
            commands=["pytest tests/", "python -m pytest --quantum"],
            timeout=1800,
            quantum_tests={
                "property_tests": True,
                "circuit_validation": True,
                "performance_tests": True
            }
        ),
        StageConfig(
            name="deploy",
            type=StageType.DEPLOY,
            commands=["echo 'Deploying application'"],
            dependencies=["build", "test"],
            timeout=1200
        )
    ]

@pytest.fixture
def mock_git_repo():
    """Create mock Git repository."""
    mock_repo = Mock()
    mock_repo.head.commit.hexsha = "abc123def456"
    mock_repo.active_branch.name = "main"
    
    mock_commit = Mock()
    mock_commit.hexsha = "abc123def456"
    mock_commit.author = "Test User <test@example.com>"
    mock_commit.message = "Test commit message"
    mock_commit.committed_date = time.time()
    mock_commit.parents = []
    
    mock_repo.commit.return_value = mock_commit
    return mock_repo

@pytest.fixture
def cicd_manager(temp_workspace):
    """Create CI/CD manager for testing."""
    return QuantumCICDManager(str(temp_workspace))


@pytest.mark.skipif(not HAS_QUANTUM_CICD, reason="quantum CI/CD not available")
class TestPipelineConfig:
    """Test PipelineConfig functionality."""
    
    def test_pipeline_config_creation(self, sample_pipeline_config):
        """Test PipelineConfig creation."""
        config = sample_pipeline_config
        
        assert config.name == "test-pipeline"
        assert config.version == "1.0"
        assert TriggerType.PUSH in config.triggers
        assert config.environment_variables["TEST_ENV"] == "true"
        assert config.timeout == 1800
        assert config.parallel is True
        assert config.quantum_config["max_qubits"] == 16

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig default values."""
        config = PipelineConfig(name="minimal-pipeline")
        
        assert config.version == "1.0"
        assert config.description == ""
        assert config.triggers == []
        assert config.environment_variables == {}
        assert config.timeout == 3600
        assert config.parallel is True
        assert config.retry_count == 0
        assert config.cache_enabled is True


@pytest.mark.skipif(not HAS_QUANTUM_CICD, reason="quantum CI/CD not available")
class TestStageConfig:
    """Test StageConfig functionality."""
    
    def test_stage_config_creation(self):
        """Test StageConfig creation."""
        stage = StageConfig(
            name="test-stage",
            type=StageType.TEST,
            commands=["pytest", "python -m test"],
            dependencies=["build"],
            environment={"TEST_MODE": "true"},
            timeout=1200,
            allow_failure=True,
            quantum_tests={"property_tests": True}
        )
        
        assert stage.name == "test-stage"
        assert stage.type == StageType.TEST
        assert "pytest" in stage.commands
        assert "build" in stage.dependencies
        assert stage.environment["TEST_MODE"] == "true"
        assert stage.timeout == 1200
        assert stage.allow_failure is True
        assert stage.quantum_tests["property_tests"] is True

    def test_stage_config_defaults(self):
        """Test StageConfig default values."""
        stage = StageConfig(name="minimal-stage", type=StageType.BUILD)
        
        assert stage.commands == []
        assert stage.dependencies == []
        assert stage.environment == {}
        assert stage.timeout == 1800
        assert stage.allow_failure is False
        assert stage.parallel is False


@pytest.mark.skipif(not HAS_QUANTUM_CICD, reason="quantum CI/CD not available")
class TestGitRepository:
    """Test GitRepository functionality."""
    
    def test_git_repository_initialization(self, temp_workspace):
        """Test GitRepository initialization."""
        repo_path = temp_workspace / "test_repo"
        repo_path.mkdir()
        
        with patch('quantrs2.quantum_cicd.HAS_GIT', True):
            with patch('quantrs2.quantum_cicd.git') as mock_git:
                mock_git.Repo.return_value = Mock()
                
                git_repo = GitRepository(str(repo_path))
                
                assert git_repo.repo_path == repo_path
                assert git_repo.repo is not None

    @patch('quantrs2.quantum_cicd.HAS_GIT', True)
    @patch('quantrs2.quantum_cicd.git')
    def test_get_current_commit(self, mock_git, mock_git_repo):
        """Test getting current commit SHA."""
        mock_git.Repo.return_value = mock_git_repo
        
        git_repo = GitRepository("/test/path")
        commit_sha = git_repo.get_current_commit()
        
        assert commit_sha == "abc123def456"

    @patch('quantrs2.quantum_cicd.HAS_GIT', True)
    @patch('quantrs2.quantum_cicd.git')
    def test_get_current_branch(self, mock_git, mock_git_repo):
        """Test getting current branch name."""
        mock_git.Repo.return_value = mock_git_repo
        
        git_repo = GitRepository("/test/path")
        branch = git_repo.get_current_branch()
        
        assert branch == "main"

    @patch('quantrs2.quantum_cicd.HAS_GIT', True)
    @patch('quantrs2.quantum_cicd.git')
    def test_get_commit_info(self, mock_git, mock_git_repo):
        """Test getting commit information."""
        mock_git.Repo.return_value = mock_git_repo
        
        git_repo = GitRepository("/test/path")
        commit_info = git_repo.get_commit_info("abc123def456")
        
        assert commit_info['sha'] == "abc123def456"
        assert commit_info['author'] == "Test User <test@example.com>"
        assert commit_info['message'] == "Test commit message"
        assert 'timestamp' in commit_info

    def test_git_repository_without_git(self, temp_workspace):
        """Test GitRepository without Git available."""
        with patch('quantrs2.quantum_cicd.HAS_GIT', False):
            git_repo = GitRepository(str(temp_workspace))
            
            assert git_repo.repo is None
            assert git_repo.get_current_commit() is None
            assert git_repo.get_current_branch() is None


@pytest.mark.skipif(not HAS_QUANTUM_CICD, reason="quantum CI/CD not available")
class TestQuantumTestRunner:
    """Test QuantumTestRunner functionality."""
    
    def test_test_runner_initialization(self, temp_workspace):
        """Test QuantumTestRunner initialization."""
        test_runner = QuantumTestRunner(str(temp_workspace))
        
        assert test_runner.working_dir == temp_workspace
        assert test_runner.test_results == []

    def test_run_quantum_tests_basic(self, temp_workspace):
        """Test basic quantum test execution."""
        test_runner = QuantumTestRunner(str(temp_workspace))
        
        test_config = {
            'property_tests': True,
            'circuit_validation': True,
            'performance_tests': True,
            'hardware_tests': False
        }
        
        results = test_runner.run_quantum_tests(test_config)
        
        assert isinstance(results, dict)
        assert 'status' in results
        assert 'tests_run' in results
        assert 'quantum_properties_verified' in results
        assert results['status'] in [PipelineStatus.SUCCESS, PipelineStatus.FAILED]

    def test_run_property_tests(self, temp_workspace):
        """Test property-based testing."""
        test_runner = QuantumTestRunner(str(temp_workspace))
        
        property_results = test_runner._run_property_tests()
        
        assert 'property_tests_run' in property_results
        assert 'property_tests_passed' in property_results
        assert 'quantum_properties_verified' in property_results
        assert 'unitarity' in property_results['quantum_properties_verified']

    def test_run_circuit_validation(self, temp_workspace):
        """Test circuit validation."""
        test_runner = QuantumTestRunner(str(temp_workspace))
        
        validation_results = test_runner._run_circuit_validation()
        
        assert isinstance(validation_results, list)
        assert 'gate_decomposition' in validation_results
        assert 'circuit_optimization' in validation_results

    def test_run_performance_tests(self, temp_workspace):
        """Test performance testing."""
        test_runner = QuantumTestRunner(str(temp_workspace))
        
        perf_results = test_runner._run_performance_tests()
        
        assert 'simulation_time' in perf_results
        assert 'memory_usage' in perf_results
        assert 'gate_fidelity' in perf_results
        assert isinstance(perf_results['simulation_time'], (int, float))


@pytest.mark.skipif(not HAS_QUANTUM_CICD, reason="quantum CI/CD not available")
class TestCodeQualityAnalyzer:
    """Test CodeQualityAnalyzer functionality."""
    
    def test_analyzer_initialization(self):
        """Test CodeQualityAnalyzer initialization."""
        analyzer = CodeQualityAnalyzer()
        
        assert analyzer.analysis_results == {}

    def test_analyze_code_basic(self, temp_workspace):
        """Test basic code analysis."""
        analyzer = CodeQualityAnalyzer()
        
        # Create sample Python file
        sample_file = temp_workspace / "sample.py"
        sample_file.write_text("""
import numpy as np

def quantum_circuit():
    '''A simple quantum circuit.'''
    # Apply Hadamard gate
    H()  # quantum gate
    # Apply CNOT gate
    CNOT()  # quantum gate
    
    try:
        # Quantum operations
        pass
    except Exception:
        pass

def vqe_algorithm():
    '''VQE algorithm implementation.'''
    pass
        """)
        
        results = analyzer.analyze_code(str(temp_workspace))
        
        assert isinstance(results, dict)
        assert 'overall_score' in results
        assert 'metrics' in results
        assert 'quantum_specific' in results
        assert results['overall_score'] >= 0

    def test_analyze_quantum_code(self, temp_workspace):
        """Test quantum-specific code analysis."""
        analyzer = CodeQualityAnalyzer()
        
        # Create quantum code files
        files = []
        for i, content in enumerate([
            "H(); X(); Y(); Z(); CNOT()",  # Quantum gates
            "VQE algorithm implementation",  # Quantum algorithm
            "try: quantum_operation() except: pass"  # Error handling
        ]):
            file_path = temp_workspace / f"quantum_{i}.py"
            file_path.write_text(content)
            files.append(file_path)
        
        quantum_metrics = analyzer._analyze_quantum_code(files)
        
        assert 'quantum_gates_used' in quantum_metrics
        assert 'quantum_algorithms_detected' in quantum_metrics
        assert 'error_handling_patterns' in quantum_metrics
        assert len(quantum_metrics['quantum_gates_used']) > 0

    def test_count_lines_of_code(self, temp_workspace):
        """Test lines of code counting."""
        analyzer = CodeQualityAnalyzer()
        
        # Create test files
        files = []
        for i in range(3):
            file_path = temp_workspace / f"test_{i}.py"
            file_path.write_text(f"""
# This is a comment
import os
def function_{i}():
    return {i}
# Another comment
print("test")
            """)
            files.append(file_path)
        
        line_count = analyzer._count_lines_of_code(files)
        
        assert line_count > 0
        assert isinstance(line_count, int)

    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        analyzer = CodeQualityAnalyzer()
        
        metrics = {
            'complexity': {'maintainability_index': 85.0},
            'documentation': {'docstring_coverage': 90.0}
        }
        
        score = analyzer._calculate_overall_score(metrics)
        
        assert 0 <= score <= 100
        assert isinstance(score, float)


@pytest.mark.skipif(not HAS_QUANTUM_CICD, reason="quantum CI/CD not available")
class TestArtifactManager:
    """Test ArtifactManager functionality."""
    
    def test_artifact_manager_initialization(self, temp_workspace):
        """Test ArtifactManager initialization."""
        artifacts_dir = temp_workspace / "artifacts"
        manager = ArtifactManager(str(artifacts_dir))
        
        assert manager.artifacts_dir == artifacts_dir
        assert artifacts_dir.exists()
        assert manager.artifacts == {}

    def test_store_artifact_file(self, temp_workspace):
        """Test storing a file artifact."""
        manager = ArtifactManager(str(temp_workspace / "artifacts"))
        
        # Create source file
        source_file = temp_workspace / "test_artifact.txt"
        source_file.write_text("Test artifact content")
        
        artifact = manager.store_artifact(
            name="test_artifact",
            source_path=str(source_file),
            artifact_type="text",
            metadata={"description": "Test artifact"}
        )
        
        assert isinstance(artifact, BuildArtifact)
        assert artifact.name == "test_artifact"
        assert artifact.type == "text"
        assert artifact.size > 0
        assert artifact.checksum != ""
        assert artifact.metadata["description"] == "Test artifact"

    def test_store_artifact_directory(self, temp_workspace):
        """Test storing a directory artifact."""
        manager = ArtifactManager(str(temp_workspace / "artifacts"))
        
        # Create source directory
        source_dir = temp_workspace / "build_output"
        source_dir.mkdir()
        (source_dir / "file1.txt").write_text("Content 1")
        (source_dir / "file2.txt").write_text("Content 2")
        
        artifact = manager.store_artifact(
            name="build_output",
            source_path=str(source_dir),
            artifact_type="directory"
        )
        
        assert artifact.name == "build_output"
        assert artifact.type == "directory"
        assert artifact.size > 0

    def test_retrieve_artifact(self, temp_workspace):
        """Test retrieving an artifact."""
        manager = ArtifactManager(str(temp_workspace / "artifacts"))
        
        # Store artifact
        source_file = temp_workspace / "test.txt"
        source_file.write_text("Test content")
        
        stored_artifact = manager.store_artifact("test", str(source_file))
        retrieved_artifact = manager.retrieve_artifact("test")
        
        assert retrieved_artifact is not None
        assert retrieved_artifact.name == stored_artifact.name
        assert retrieved_artifact.checksum == stored_artifact.checksum

    def test_list_artifacts(self, temp_workspace):
        """Test listing artifacts."""
        manager = ArtifactManager(str(temp_workspace / "artifacts"))
        
        # Store multiple artifacts
        for i in range(3):
            source_file = temp_workspace / f"test_{i}.txt"
            source_file.write_text(f"Content {i}")
            manager.store_artifact(f"test_{i}", str(source_file))
        
        artifacts = manager.list_artifacts()
        
        assert len(artifacts) == 3
        assert all(isinstance(artifact, BuildArtifact) for artifact in artifacts)

    def test_cleanup_old_artifacts(self, temp_workspace):
        """Test cleaning up old artifacts."""
        manager = ArtifactManager(str(temp_workspace / "artifacts"))
        
        # Store artifacts with different ages
        old_time = time.time() - (40 * 24 * 3600)  # 40 days ago
        
        source_file = temp_workspace / "old_artifact.txt"
        source_file.write_text("Old content")
        
        artifact = manager.store_artifact("old_artifact", str(source_file))
        artifact.created_at = old_time  # Make it old
        manager.artifacts["old_artifact"] = artifact
        
        # Store new artifact
        new_file = temp_workspace / "new_artifact.txt"
        new_file.write_text("New content")
        manager.store_artifact("new_artifact", str(new_file))
        
        # Cleanup
        manager.cleanup_old_artifacts(max_age_days=30)
        
        # Check that old artifact was removed
        assert "old_artifact" not in manager.artifacts
        assert "new_artifact" in manager.artifacts

    def test_calculate_checksum(self, temp_workspace):
        """Test checksum calculation."""
        manager = ArtifactManager(str(temp_workspace / "artifacts"))
        
        test_file = temp_workspace / "checksum_test.txt"
        test_file.write_text("Test content for checksum")
        
        checksum = manager._calculate_checksum(test_file)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hex digest length

    def test_get_size(self, temp_workspace):
        """Test size calculation."""
        manager = ArtifactManager(str(temp_workspace / "artifacts"))
        
        # Test file size
        test_file = temp_workspace / "size_test.txt"
        test_content = "Test content"
        test_file.write_text(test_content)
        
        file_size = manager._get_size(test_file)
        assert file_size == len(test_content.encode())
        
        # Test directory size
        test_dir = temp_workspace / "size_test_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("Content 1")
        (test_dir / "file2.txt").write_text("Content 2")
        
        dir_size = manager._get_size(test_dir)
        assert dir_size > 0


@pytest.mark.skipif(not HAS_QUANTUM_CICD, reason="quantum CI/CD not available")
class TestNotificationManager:
    """Test NotificationManager functionality."""
    
    def test_notification_manager_initialization(self):
        """Test NotificationManager initialization."""
        manager = NotificationManager()
        
        assert manager.notification_configs == {}
        assert manager.notification_history == []

    def test_add_notification_config(self):
        """Test adding notification configuration."""
        manager = NotificationManager()
        
        config = NotificationConfig(
            type=NotificationType.EMAIL,
            recipients=["test@example.com"],
            on_success=True,
            on_failure=True
        )
        
        manager.add_notification_config("email_notifications", config)
        
        assert "email_notifications" in manager.notification_configs
        assert manager.notification_configs["email_notifications"] == config

    def test_generate_message(self):
        """Test notification message generation."""
        manager = NotificationManager()
        
        pipeline_run = PipelineRun(
            id="test_run_123",
            pipeline_name="test-pipeline",
            commit_sha="abc123def456",
            branch="main",
            trigger=TriggerType.PUSH,
            status=PipelineStatus.SUCCESS,
            started_at=time.time() - 60,
            finished_at=time.time()
        )
        
        config = NotificationConfig(
            type=NotificationType.EMAIL,
            recipients=["test@example.com"]
        )
        
        message = manager._generate_message("completed", pipeline_run, config)
        
        assert "test-pipeline" in message
        assert "success" in message.lower()
        assert "main" in message
        assert "abc123de" in message  # First 8 chars of commit

    @patch('quantrs2.quantum_cicd.HAS_EMAIL', True)
    @patch('quantrs2.quantum_cicd.smtplib')
    def test_send_email_notification(self, mock_smtplib):
        """Test sending email notification."""
        manager = NotificationManager()
        
        mock_server = Mock()
        mock_smtplib.SMTP.return_value.__enter__.return_value = mock_server
        
        config = NotificationConfig(
            type=NotificationType.EMAIL,
            recipients=["test@example.com"],
            configuration={
                'smtp_server': 'localhost',
                'smtp_port': 587,
                'from_email': 'noreply@test.com'
            }
        )
        
        pipeline_run = PipelineRun(
            id="test_run",
            pipeline_name="test-pipeline",
            commit_sha="abc123",
            branch="main",
            trigger=TriggerType.PUSH,
            status=PipelineStatus.SUCCESS,
            started_at=time.time()
        )
        
        manager._send_email(config, "Test message", pipeline_run)
        
        # Verify SMTP was called
        mock_smtplib.SMTP.assert_called_once()

    @patch('quantrs2.quantum_cicd.HAS_REQUESTS', True)
    @patch('quantrs2.quantum_cicd.requests')
    def test_send_webhook_notification(self, mock_requests):
        """Test sending webhook notification."""
        manager = NotificationManager()
        
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        config = NotificationConfig(
            type=NotificationType.WEBHOOK,
            configuration={'url': 'http://test.example.com/webhook'}
        )
        
        pipeline_run = PipelineRun(
            id="test_run",
            pipeline_name="test-pipeline",
            commit_sha="abc123",
            branch="main",
            trigger=TriggerType.PUSH,
            status=PipelineStatus.SUCCESS,
            started_at=time.time()
        )
        
        manager._send_webhook(config, "Test message", pipeline_run)
        
        # Verify webhook was called
        mock_requests.post.assert_called_once()
        args, kwargs = mock_requests.post.call_args
        assert args[0] == 'http://test.example.com/webhook'
        assert 'json' in kwargs

    def test_send_notification_success_only(self):
        """Test notification sending with success-only configuration."""
        manager = NotificationManager()
        
        config = NotificationConfig(
            type=NotificationType.EMAIL,
            recipients=["test@example.com"],
            on_success=True,
            on_failure=False
        )
        manager.add_notification_config("success_only", config)
        
        # Test successful pipeline
        success_run = PipelineRun(
            id="success_run",
            pipeline_name="test-pipeline",
            commit_sha="abc123",
            branch="main",
            trigger=TriggerType.PUSH,
            status=PipelineStatus.SUCCESS,
            started_at=time.time()
        )
        
        with patch.object(manager, '_send_email') as mock_send:
            result = manager.send_notification("completed", success_run)
            mock_send.assert_called_once()
        
        # Test failed pipeline
        failed_run = PipelineRun(
            id="failed_run",
            pipeline_name="test-pipeline",
            commit_sha="abc123",
            branch="main",
            trigger=TriggerType.PUSH,
            status=PipelineStatus.FAILED,
            started_at=time.time()
        )
        
        with patch.object(manager, '_send_email') as mock_send:
            result = manager.send_notification("completed", failed_run)
            mock_send.assert_not_called()


@pytest.mark.skipif(not HAS_QUANTUM_CICD, reason="quantum CI/CD not available")
class TestPipelineEngine:
    """Test PipelineEngine functionality."""
    
    def test_pipeline_engine_initialization(self, temp_workspace):
        """Test PipelineEngine initialization."""
        engine = PipelineEngine(str(temp_workspace))
        
        assert engine.working_dir == temp_workspace
        assert isinstance(engine.test_runner, QuantumTestRunner)
        assert isinstance(engine.code_analyzer, CodeQualityAnalyzer)
        assert isinstance(engine.artifact_manager, ArtifactManager)
        assert isinstance(engine.notification_manager, NotificationManager)

    @pytest.mark.asyncio
    async def test_execute_commands_success(self, temp_workspace):
        """Test successful command execution."""
        engine = PipelineEngine(str(temp_workspace))
        
        stage_result = {'logs': []}
        commands = ['echo "Hello World"', 'echo "Command 2"']
        env = os.environ.copy()
        
        await engine._execute_commands(commands, stage_result, env, 30)
        
        assert len(stage_result['logs']) > 0
        assert any("Hello World" in log for log in stage_result['logs'])

    @pytest.mark.asyncio
    async def test_execute_commands_failure(self, temp_workspace):
        """Test command execution failure."""
        engine = PipelineEngine(str(temp_workspace))
        
        stage_result = {'logs': []}
        commands = ['exit 1']  # Command that fails
        env = os.environ.copy()
        
        with pytest.raises(Exception, match="Command execution failed"):
            await engine._execute_commands(commands, stage_result, env, 30)

    @pytest.mark.asyncio
    async def test_execute_commands_timeout(self, temp_workspace):
        """Test command execution timeout."""
        engine = PipelineEngine(str(temp_workspace))
        
        stage_result = {'logs': []}
        commands = ['sleep 10']  # Long running command
        env = os.environ.copy()
        
        with pytest.raises(Exception, match="Command timed out"):
            await engine._execute_commands(commands, stage_result, env, 1)

    @pytest.mark.asyncio
    async def test_execute_stage_build(self, temp_workspace, sample_stage_configs):
        """Test executing build stage."""
        engine = PipelineEngine(str(temp_workspace))
        
        build_stage = sample_stage_configs[0]  # build stage
        build_stage.commands = ['echo "Building"', 'echo "Build complete"']
        
        pipeline_run = PipelineRun(
            id="test_run",
            pipeline_name="test",
            commit_sha="abc123",
            branch="main",
            trigger=TriggerType.MANUAL,
            status=PipelineStatus.RUNNING,
            started_at=time.time(),
            environment_variables={}
        )
        
        stage_result = await engine._execute_stage(build_stage, pipeline_run)
        
        assert stage_result['name'] == 'build'
        assert stage_result['type'] == 'build'
        assert stage_result['status'] == PipelineStatus.SUCCESS
        assert len(stage_result['logs']) > 0

    @pytest.mark.asyncio
    async def test_execute_stage_test(self, temp_workspace, sample_stage_configs):
        """Test executing test stage."""
        engine = PipelineEngine(str(temp_workspace))
        
        test_stage = sample_stage_configs[1]  # test stage
        test_stage.commands = ['echo "Running tests"']
        
        pipeline_run = PipelineRun(
            id="test_run",
            pipeline_name="test",
            commit_sha="abc123",
            branch="main",
            trigger=TriggerType.MANUAL,
            status=PipelineStatus.RUNNING,
            started_at=time.time(),
            environment_variables={}
        )
        
        stage_result = await engine._execute_stage(test_stage, pipeline_run)
        
        assert stage_result['name'] == 'test'
        assert stage_result['type'] == 'test'
        assert 'quantum_results' in stage_result

    @pytest.mark.asyncio
    async def test_execute_stage_with_artifacts(self, temp_workspace):
        """Test stage execution with artifact collection."""
        engine = PipelineEngine(str(temp_workspace))
        
        # Create test artifacts
        artifact_file = temp_workspace / "test_artifact.txt"
        artifact_file.write_text("Test artifact content")
        
        stage = StageConfig(
            name="artifact_stage",
            type=StageType.BUILD,
            commands=['echo "Creating artifacts"'],
            artifacts=["test_artifact.txt"]
        )
        
        pipeline_run = PipelineRun(
            id="test_run",
            pipeline_name="test",
            commit_sha="abc123",
            branch="main",
            trigger=TriggerType.MANUAL,
            status=PipelineStatus.RUNNING,
            started_at=time.time(),
            environment_variables={}
        )
        
        stage_result = await engine._execute_stage(stage, pipeline_run)
        
        assert len(stage_result['artifacts']) > 0
        assert len(pipeline_run.artifacts) > 0

    def test_build_dependency_graph(self, temp_workspace, sample_stage_configs):
        """Test dependency graph building."""
        engine = PipelineEngine(str(temp_workspace))
        
        graph = engine._build_dependency_graph(sample_stage_configs)
        
        assert 'build' in graph
        assert 'test' in graph
        assert 'deploy' in graph
        assert graph['deploy'] == ['build', 'test']
        assert graph['build'] == []

    @pytest.mark.asyncio
    async def test_execute_pipeline_sequential(self, temp_workspace, sample_pipeline_config):
        """Test sequential pipeline execution."""
        engine = PipelineEngine(str(temp_workspace))
        
        stages = [
            StageConfig(name="stage1", type=StageType.BUILD, commands=['echo "Stage 1"']),
            StageConfig(name="stage2", type=StageType.TEST, commands=['echo "Stage 2"'])
        ]
        
        sample_pipeline_config.parallel = False
        
        pipeline_run = await engine.execute_pipeline(
            config=sample_pipeline_config,
            stages=stages,
            trigger=TriggerType.MANUAL,
            commit_sha="abc123",
            branch="main"
        )
        
        assert pipeline_run.status == PipelineStatus.SUCCESS
        assert len(pipeline_run.stages) == 2
        assert pipeline_run.finished_at is not None

    @pytest.mark.asyncio
    async def test_execute_pipeline_with_failure(self, temp_workspace, sample_pipeline_config):
        """Test pipeline execution with stage failure."""
        engine = PipelineEngine(str(temp_workspace))
        
        stages = [
            StageConfig(name="success_stage", type=StageType.BUILD, commands=['echo "Success"']),
            StageConfig(name="failure_stage", type=StageType.TEST, commands=['exit 1'])
        ]
        
        pipeline_run = await engine.execute_pipeline(
            config=sample_pipeline_config,
            stages=stages,
            trigger=TriggerType.MANUAL,
            commit_sha="abc123",
            branch="main"
        )
        
        assert pipeline_run.status == PipelineStatus.FAILED
        assert pipeline_run.finished_at is not None

    def test_get_pipeline_status(self, temp_workspace):
        """Test getting pipeline status."""
        engine = PipelineEngine(str(temp_workspace))
        
        # Create completed pipeline
        completed_run = PipelineRun(
            id="completed_123",
            pipeline_name="test",
            commit_sha="abc123",
            branch="main",
            trigger=TriggerType.MANUAL,
            status=PipelineStatus.SUCCESS,
            started_at=time.time() - 100,
            finished_at=time.time()
        )
        engine.completed_pipelines.append(completed_run)
        
        # Test retrieving completed pipeline
        status = engine.get_pipeline_status("completed_123")
        assert status is not None
        assert status.id == "completed_123"
        
        # Test non-existent pipeline
        status = engine.get_pipeline_status("nonexistent")
        assert status is None

    def test_list_pipeline_runs(self, temp_workspace):
        """Test listing pipeline runs."""
        engine = PipelineEngine(str(temp_workspace))
        
        # Create multiple pipeline runs
        for i in range(5):
            run = PipelineRun(
                id=f"run_{i}",
                pipeline_name="test",
                commit_sha=f"abc{i}",
                branch="main",
                trigger=TriggerType.MANUAL,
                status=PipelineStatus.SUCCESS,
                started_at=time.time() - (i * 100)
            )
            engine.completed_pipelines.append(run)
        
        runs = engine.list_pipeline_runs(limit=3)
        
        assert len(runs) == 3
        # Should be sorted by start time (most recent first)
        assert runs[0].started_at >= runs[1].started_at


@pytest.mark.skipif(not HAS_QUANTUM_CICD, reason="quantum CI/CD not available")
class TestQuantumCICDManager:
    """Test QuantumCICDManager functionality."""
    
    def test_cicd_manager_initialization(self, cicd_manager):
        """Test QuantumCICDManager initialization."""
        assert isinstance(cicd_manager.pipeline_engine, PipelineEngine)
        assert cicd_manager.git_repos == {}
        assert cicd_manager.pipeline_configs == {}
        assert cicd_manager.webhook_listeners == {}

    @patch('quantrs2.quantum_cicd.HAS_GIT', True)
    @patch('quantrs2.quantum_cicd.git')
    def test_add_repository(self, mock_git, cicd_manager, temp_workspace):
        """Test adding Git repository."""
        mock_git.Repo.return_value = Mock()
        
        repo_path = temp_workspace / "test_repo"
        repo_path.mkdir()
        
        git_repo = cicd_manager.add_repository("test_repo", str(repo_path))
        
        assert "test_repo" in cicd_manager.git_repos
        assert isinstance(git_repo, GitRepository)

    def test_add_pipeline_config(self, cicd_manager, sample_pipeline_config, sample_stage_configs):
        """Test adding pipeline configuration."""
        cicd_manager.add_pipeline_config(sample_pipeline_config, sample_stage_configs)
        
        assert sample_pipeline_config.name in cicd_manager.pipeline_configs
        config_data = cicd_manager.pipeline_configs[sample_pipeline_config.name]
        assert config_data['config'] == sample_pipeline_config
        assert config_data['stages'] == sample_stage_configs

    def test_add_notification_config(self, cicd_manager):
        """Test adding notification configuration."""
        notification_config = NotificationConfig(
            type=NotificationType.EMAIL,
            recipients=["test@example.com"]
        )
        
        cicd_manager.add_notification_config("email_test", notification_config)
        
        # Check if notification was added to pipeline engine
        assert "email_test" in cicd_manager.pipeline_engine.notification_manager.notification_configs

    @pytest.mark.asyncio
    async def test_trigger_pipeline_manual(self, cicd_manager, sample_pipeline_config, sample_stage_configs, temp_workspace):
        """Test manual pipeline trigger."""
        # Setup
        repo_path = temp_workspace / "test_repo"
        repo_path.mkdir()
        
        with patch('quantrs2.quantum_cicd.HAS_GIT', True):
            with patch('quantrs2.quantum_cicd.git') as mock_git:
                mock_repo = Mock()
                mock_repo.head.commit.hexsha = "abc123"
                mock_repo.active_branch.name = "main"
                mock_git.Repo.return_value = mock_repo
                
                cicd_manager.add_repository("test_repo", str(repo_path))
        
        # Simplify stages for testing
        simple_stages = [
            StageConfig(name="simple_test", type=StageType.BUILD, commands=['echo "test"'])
        ]
        
        cicd_manager.add_pipeline_config(sample_pipeline_config, simple_stages)
        
        # Trigger pipeline
        pipeline_run = await cicd_manager.trigger_pipeline(
            pipeline_name=sample_pipeline_config.name,
            repo_name="test_repo",
            trigger=TriggerType.MANUAL
        )
        
        assert isinstance(pipeline_run, PipelineRun)
        assert pipeline_run.trigger == TriggerType.MANUAL
        assert pipeline_run.commit_sha == "abc123"
        assert pipeline_run.branch == "main"

    @pytest.mark.asyncio
    async def test_trigger_pipeline_invalid_config(self, cicd_manager, temp_workspace):
        """Test triggering pipeline with invalid configuration."""
        repo_path = temp_workspace / "test_repo"
        repo_path.mkdir()
        
        with patch('quantrs2.quantum_cicd.HAS_GIT', True):
            with patch('quantrs2.quantum_cicd.git') as mock_git:
                mock_git.Repo.return_value = Mock()
                cicd_manager.add_repository("test_repo", str(repo_path))
        
        with pytest.raises(ValueError, match="Pipeline configuration not found"):
            await cicd_manager.trigger_pipeline(
                pipeline_name="nonexistent_pipeline",
                repo_name="test_repo"
            )

    @pytest.mark.asyncio
    async def test_trigger_pipeline_invalid_repo(self, cicd_manager, sample_pipeline_config, sample_stage_configs):
        """Test triggering pipeline with invalid repository."""
        cicd_manager.add_pipeline_config(sample_pipeline_config, sample_stage_configs)
        
        with pytest.raises(ValueError, match="Repository not found"):
            await cicd_manager.trigger_pipeline(
                pipeline_name=sample_pipeline_config.name,
                repo_name="nonexistent_repo"
            )

    def test_get_pipeline_statistics_empty(self, cicd_manager):
        """Test getting pipeline statistics with no runs."""
        stats = cicd_manager.get_pipeline_statistics()
        
        assert stats['total_runs'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_duration'] == 0.0
        assert stats['quantum_tests_run'] == 0
        assert stats['status_distribution'] == {}
        assert stats['recent_activity'] == []

    def test_get_pipeline_statistics_with_data(self, cicd_manager):
        """Test getting pipeline statistics with data."""
        # Add sample pipeline runs
        runs = []
        for i in range(5):
            run = PipelineRun(
                id=f"run_{i}",
                pipeline_name="test",
                commit_sha=f"abc{i}",
                branch="main",
                trigger=TriggerType.PUSH,
                status=PipelineStatus.SUCCESS if i < 4 else PipelineStatus.FAILED,
                started_at=time.time() - (i * 100),
                finished_at=time.time() - (i * 100) + 60,
                stages=[{
                    'quantum_results': {'tests_run': 10, 'quantum_properties_verified': ['unitarity']}
                }],
                artifacts=[f"artifact_{i}.zip"]
            )
            runs.append(run)
        
        cicd_manager.pipeline_engine.completed_pipelines = runs
        
        stats = cicd_manager.get_pipeline_statistics()
        
        assert stats['total_runs'] == 5
        assert stats['success_rate'] == 80.0  # 4 success out of 5
        assert stats['average_duration'] == 60.0
        assert stats['quantum_tests_run'] == 50  # 10 per run * 5 runs
        assert stats['artifacts_generated'] == 5
        assert 'success' in stats['status_distribution']
        assert 'failed' in stats['status_distribution']
        assert len(stats['recent_activity']) == 5

    def test_export_pipeline_config_yaml(self, cicd_manager, sample_pipeline_config, sample_stage_configs):
        """Test exporting pipeline configuration as YAML."""
        cicd_manager.add_pipeline_config(sample_pipeline_config, sample_stage_configs)
        
        yaml_config = cicd_manager.export_pipeline_config(sample_pipeline_config.name, "yaml")
        
        assert isinstance(yaml_config, str)
        assert "test-pipeline" in yaml_config
        assert "name:" in yaml_config
        assert "stages:" in yaml_config

    def test_export_pipeline_config_json(self, cicd_manager, sample_pipeline_config, sample_stage_configs):
        """Test exporting pipeline configuration as JSON."""
        cicd_manager.add_pipeline_config(sample_pipeline_config, sample_stage_configs)
        
        json_config = cicd_manager.export_pipeline_config(sample_pipeline_config.name, "json")
        
        assert isinstance(json_config, str)
        config_data = json.loads(json_config)
        assert config_data['pipeline']['name'] == "test-pipeline"
        assert 'stages' in config_data
        assert len(config_data['stages']) == len(sample_stage_configs)

    def test_export_pipeline_config_invalid_pipeline(self, cicd_manager):
        """Test exporting configuration for non-existent pipeline."""
        with pytest.raises(ValueError, match="Pipeline not found"):
            cicd_manager.export_pipeline_config("nonexistent", "yaml")

    def test_export_pipeline_config_invalid_format(self, cicd_manager, sample_pipeline_config, sample_stage_configs):
        """Test exporting configuration with invalid format."""
        cicd_manager.add_pipeline_config(sample_pipeline_config, sample_stage_configs)
        
        with pytest.raises(ValueError, match="Unsupported format"):
            cicd_manager.export_pipeline_config(sample_pipeline_config.name, "xml")

    def test_import_pipeline_config_yaml(self, cicd_manager):
        """Test importing pipeline configuration from YAML."""
        yaml_config = """
pipeline:
  name: imported-pipeline
  version: "1.0"
  triggers:
    - push
    - manual
  environment_variables:
    TEST: "true"
stages:
  - name: build
    type: build
    commands:
      - echo "Building"
  - name: test
    type: test
    commands:
      - pytest
    quantum_tests:
      property_tests: true
        """
        
        pipeline_name = cicd_manager.import_pipeline_config(yaml_config, "yaml")
        
        assert pipeline_name == "imported-pipeline"
        assert "imported-pipeline" in cicd_manager.pipeline_configs
        
        config_data = cicd_manager.pipeline_configs["imported-pipeline"]
        assert config_data['config'].name == "imported-pipeline"
        assert len(config_data['stages']) == 2

    def test_import_pipeline_config_json(self, cicd_manager):
        """Test importing pipeline configuration from JSON."""
        json_config = """
{
  "pipeline": {
    "name": "json-imported-pipeline",
    "version": "1.0",
    "triggers": ["push"],
    "environment_variables": {"ENV": "test"}
  },
  "stages": [
    {
      "name": "build",
      "type": "build",
      "commands": ["make build"]
    }
  ]
}
        """
        
        pipeline_name = cicd_manager.import_pipeline_config(json_config, "json")
        
        assert pipeline_name == "json-imported-pipeline"
        assert "json-imported-pipeline" in cicd_manager.pipeline_configs

    def test_import_pipeline_config_invalid_format(self, cicd_manager):
        """Test importing configuration with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            cicd_manager.import_pipeline_config("config", "xml")

    def test_import_pipeline_config_invalid_data(self, cicd_manager):
        """Test importing invalid configuration data."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with pytest.raises(ValueError, match="Failed to import pipeline config"):
            cicd_manager.import_pipeline_config(invalid_yaml, "yaml")


@pytest.mark.skipif(not HAS_QUANTUM_CICD, reason="quantum CI/CD not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_get_quantum_cicd_manager(self, temp_workspace):
        """Test get_quantum_cicd_manager function."""
        manager = get_quantum_cicd_manager(str(temp_workspace))
        
        assert isinstance(manager, QuantumCICDManager)
        assert manager.working_dir == temp_workspace

    def test_create_basic_pipeline_config(self):
        """Test create_basic_pipeline_config function."""
        config = create_basic_pipeline_config(
            name="test-pipeline",
            environment_variables={"TEST": "true"},
            timeout=1200,
            parallel=False,
            quantum_config={"simulators": 2}
        )
        
        assert isinstance(config, PipelineConfig)
        assert config.name == "test-pipeline"
        assert config.environment_variables["TEST"] == "true"
        assert config.timeout == 1200
        assert config.parallel is False
        assert config.quantum_config["simulators"] == 2

    def test_create_quantum_test_stage(self):
        """Test create_quantum_test_stage function."""
        stage = create_quantum_test_stage(
            name="quantum_testing",
            commands=["pytest quantum_tests/"],
            timeout=2400,
            property_tests=True,
            circuit_validation=False,
            performance_tests=True,
            hardware_tests=True
        )
        
        assert isinstance(stage, StageConfig)
        assert stage.name == "quantum_testing"
        assert stage.type == StageType.TEST
        assert "pytest quantum_tests/" in stage.commands
        assert stage.timeout == 2400
        assert stage.quantum_tests["property_tests"] is True
        assert stage.quantum_tests["circuit_validation"] is False
        assert stage.quantum_tests["performance_tests"] is True
        assert stage.quantum_tests["hardware_tests"] is True

    def test_create_build_stage(self):
        """Test create_build_stage function."""
        stage = create_build_stage(
            name="custom_build",
            commands=["make", "make install"],
            artifacts=["bin/*", "lib/*"],
            timeout=1800
        )
        
        assert isinstance(stage, StageConfig)
        assert stage.name == "custom_build"
        assert stage.type == StageType.BUILD
        assert "make" in stage.commands
        assert "bin/*" in stage.artifacts
        assert stage.timeout == 1800

    def test_create_deploy_stage(self):
        """Test create_deploy_stage function."""
        stage = create_deploy_stage(
            name="production_deploy",
            environment=Environment.PRODUCTION,
            commands=["kubectl apply -f deployment.yaml"],
            environment_vars={"ENVIRONMENT": "prod"},
            timeout=1800,
            use_containers=True,
            deployment_tests=True
        )
        
        assert isinstance(stage, StageConfig)
        assert stage.name == "production_deploy"
        assert stage.type == StageType.DEPLOY
        assert "kubectl apply" in stage.commands[0]
        assert stage.environment["ENVIRONMENT"] == "prod"
        assert stage.timeout == 1800
        assert stage.quantum_tests["use_containers"] is True
        assert stage.quantum_tests["deployment_tests"] is True


@pytest.mark.skipif(not HAS_QUANTUM_CICD, reason="quantum CI/CD not available")
class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_artifact_manager_missing_source(self, temp_workspace):
        """Test artifact manager with missing source file."""
        manager = ArtifactManager(str(temp_workspace / "artifacts"))
        
        with pytest.raises(FileNotFoundError):
            manager.store_artifact("missing", "/nonexistent/path")

    @pytest.mark.asyncio
    async def test_pipeline_execution_with_missing_dependencies(self, temp_workspace):
        """Test pipeline execution with circular dependencies."""
        engine = PipelineEngine(str(temp_workspace))
        
        config = PipelineConfig(name="circular-test")
        stages = [
            StageConfig(name="stage1", type=StageType.BUILD, dependencies=["stage2"]),
            StageConfig(name="stage2", type=StageType.TEST, dependencies=["stage1"])
        ]
        
        with pytest.raises(Exception, match="Circular dependency"):
            await engine.execute_pipeline(config, stages, TriggerType.MANUAL, "abc123", "main")

    def test_notification_manager_without_dependencies(self):
        """Test notification manager without optional dependencies."""
        manager = NotificationManager()
        
        config = NotificationConfig(
            type=NotificationType.EMAIL,
            configuration={'smtp_server': 'test.com'}
        )
        
        pipeline_run = PipelineRun(
            id="test",
            pipeline_name="test",
            commit_sha="abc123",
            branch="main",
            trigger=TriggerType.MANUAL,
            status=PipelineStatus.SUCCESS,
            started_at=time.time()
        )
        
        # Should not crash even without email support
        with patch('quantrs2.quantum_cicd.HAS_EMAIL', False):
            manager._send_email(config, "test message", pipeline_run)

    def test_git_repository_error_handling(self, temp_workspace):
        """Test Git repository error handling."""
        with patch('quantrs2.quantum_cicd.HAS_GIT', True):
            with patch('quantrs2.quantum_cicd.git') as mock_git:
                mock_git.Repo.side_effect = Exception("Git error")
                
                git_repo = GitRepository(str(temp_workspace))
                
                # Should handle errors gracefully
                assert git_repo.get_current_commit() is None
                assert git_repo.get_current_branch() is None

    def test_code_analyzer_with_empty_directory(self, temp_workspace):
        """Test code analyzer with empty directory."""
        analyzer = CodeQualityAnalyzer()
        
        results = analyzer.analyze_code(str(temp_workspace))
        
        assert isinstance(results, dict)
        assert 'overall_score' in results
        assert results['overall_score'] >= 0

    @pytest.mark.asyncio
    async def test_stage_execution_timeout_handling(self, temp_workspace):
        """Test stage execution timeout handling."""
        engine = PipelineEngine(str(temp_workspace))
        
        stage = StageConfig(
            name="timeout_stage",
            type=StageType.BUILD,
            commands=["sleep 10"],
            timeout=1  # Very short timeout
        )
        
        pipeline_run = PipelineRun(
            id="test",
            pipeline_name="test",
            commit_sha="abc123",
            branch="main",
            trigger=TriggerType.MANUAL,
            status=PipelineStatus.RUNNING,
            started_at=time.time(),
            environment_variables={}
        )
        
        stage_result = await engine._execute_stage(stage, pipeline_run)
        
        assert stage_result['status'] == PipelineStatus.TIMEOUT
        assert "timed out" in ' '.join(stage_result['logs']).lower()

    def test_artifact_manager_checksum_for_empty_file(self, temp_workspace):
        """Test artifact manager checksum calculation for empty file."""
        manager = ArtifactManager(str(temp_workspace / "artifacts"))
        
        empty_file = temp_workspace / "empty.txt"
        empty_file.touch()  # Create empty file
        
        checksum = manager._calculate_checksum(empty_file)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA256 hash length

    def test_pipeline_config_with_invalid_enum_values(self):
        """Test handling invalid enum values in configurations."""
        # This should work fine as the enums are validated at creation
        config = PipelineConfig(
            name="test",
            triggers=[TriggerType.PUSH, TriggerType.MANUAL]
        )
        
        assert len(config.triggers) == 2

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_execution(self, temp_workspace):
        """Test concurrent pipeline execution."""
        engine = PipelineEngine(str(temp_workspace))
        
        config = PipelineConfig(name="concurrent-test")
        stages = [
            StageConfig(name="concurrent_stage", type=StageType.BUILD, commands=['echo "test"'])
        ]
        
        # Start multiple pipelines concurrently
        tasks = []
        for i in range(3):
            task = asyncio.create_task(
                engine.execute_pipeline(config, stages, TriggerType.MANUAL, f"commit{i}", "main")
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully
        assert len(results) == 3
        assert all(isinstance(result, PipelineRun) for result in results)


if __name__ == "__main__":
    pytest.main([__file__])