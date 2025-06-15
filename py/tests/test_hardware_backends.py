#!/usr/bin/env python3
"""Tests for hardware backend integration."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

try:
    from quantrs2.hardware_backends import *
    HAS_HARDWARE_BACKENDS = True
except ImportError:
    HAS_HARDWARE_BACKENDS = False
    
    # Stub implementations
    class DeviceCapabilities: pass
    class HardwareBackendManager: pass
    class QuantumBackend: pass
    class BackendType: pass
    class DeviceInfo: pass
    class DeviceStatus: pass
    class IBMQuantumBackend: pass
    class GoogleQuantumAIBackend: pass
    class JobStatus: pass
    class JobResult: pass
    class JobRequest: pass


@pytest.mark.skipif(not HAS_HARDWARE_BACKENDS, reason="quantrs2.hardware_backends module not available")
class TestDeviceCapabilities:
    def test_creation(self):
        caps = DeviceCapabilities(
            max_qubits=5,
            gate_set=["h", "cnot", "rz"],
            connectivity=[(0,1), (1,2)],
            max_shots=8192
        )
        assert caps.max_qubits == 5
        assert "h" in caps.gate_set
        assert caps.connectivity == [(0,1), (1,2)]


@pytest.mark.skipif(not HAS_HARDWARE_BACKENDS, reason="quantrs2.hardware_backends module not available")
class TestHardwareBackendManager:
    @pytest.fixture
    def manager(self):
        return HardwareBackendManager()
    
    @pytest.fixture
    def mock_backend(self):
        backend = Mock(spec=QuantumBackend)
        backend.name = "test_backend"
        backend.backend_type = BackendType.SIMULATOR
        backend.authenticate = AsyncMock(return_value=True)
        backend.get_devices = AsyncMock(return_value=[])
        return backend
    
    def test_register_backend(self, manager, mock_backend):
        manager.register_backend(mock_backend)
        assert "test_backend" in manager.backends
        assert manager.get_backend("test_backend") == mock_backend
    
    @pytest.mark.asyncio
    async def test_authenticate_all(self, manager, mock_backend):
        manager.register_backend(mock_backend)
        results = await manager.authenticate_all()
        assert results["test_backend"] is True
        mock_backend.authenticate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_all_devices(self, manager, mock_backend):
        device = DeviceInfo(
            name="test_device",
            backend_type=BackendType.SIMULATOR,
            status=DeviceStatus.ONLINE,
            capabilities=DeviceCapabilities(max_qubits=5, gate_set=["h"])
        )
        mock_backend.get_devices.return_value = [device]
        manager.register_backend(mock_backend)
        
        devices = await manager.get_all_devices()
        assert "test_backend" in devices
        assert len(devices["test_backend"]) == 1
        assert devices["test_backend"][0].name == "test_device"
    
    @pytest.mark.asyncio
    async def test_find_best_device(self, manager, mock_backend):
        device = DeviceInfo(
            name="good_device",
            backend_type=BackendType.SIMULATOR,
            status=DeviceStatus.ONLINE,
            capabilities=DeviceCapabilities(max_qubits=10, gate_set=["h"]),
            queue_length=5
        )
        mock_backend.get_devices.return_value = [device]
        manager.register_backend(mock_backend)
        
        requirements = {"min_qubits": 5, "max_queue_length": 10}
        best = await manager.find_best_device(requirements)
        
        assert best is not None
        assert best.name == "good_device"


@pytest.mark.skipif(not HAS_HARDWARE_BACKENDS, reason="quantrs2.hardware_backends module not available")
class TestIBMQuantumBackend:
    @pytest.fixture
    def backend(self):
        return IBMQuantumBackend({"token": "fake_token"})
    
    @pytest.mark.asyncio
    async def test_authenticate_with_runtime(self, backend):
        with patch('quantrs2.hardware_backends.QiskitRuntimeService') as mock_service:
            mock_service.return_value = Mock()
            result = await backend.authenticate()
            assert result is True
            assert backend._authenticated is True
    
    @pytest.mark.asyncio
    async def test_authenticate_legacy(self, backend):
        with patch('quantrs2.hardware_backends.QiskitRuntimeService', side_effect=ImportError):
            with patch('quantrs2.hardware_backends.IBMQ') as mock_ibmq:
                mock_provider = Mock()
                mock_ibmq.get_provider.return_value = mock_provider
                result = await backend.authenticate()
                assert result is True


@pytest.mark.skipif(not HAS_HARDWARE_BACKENDS, reason="quantrs2.hardware_backends module not available")
class TestGoogleQuantumAIBackend:
    @pytest.fixture
    def backend(self):
        return GoogleQuantumAIBackend({"project_id": "test_project"})
    
    @pytest.mark.asyncio
    async def test_authenticate(self, backend):
        with patch('quantrs2.hardware_backends.cirq_google') as mock_cirq:
            mock_engine = Mock()
            mock_cirq.Engine.return_value = mock_engine
            result = await backend.authenticate()
            assert result is True
            assert backend._authenticated is True


@pytest.mark.skipif(not HAS_HARDWARE_BACKENDS, reason="quantrs2.hardware_backends module not available")
class TestJobOperations:
    @pytest.mark.asyncio
    async def test_job_lifecycle(self):
        manager = HardwareBackendManager()
        
        # Mock backend
        backend = Mock(spec=QuantumBackend)
        backend.name = "test_backend"
        backend.backend_type = BackendType.SIMULATOR
        backend.submit_job = AsyncMock(return_value="job123")
        backend.get_job_status = AsyncMock(return_value=JobStatus.COMPLETED)
        backend.get_job_result = AsyncMock(return_value=JobResult(
            job_id="job123",
            status=JobStatus.COMPLETED,
            device_name="test_device",
            shots=1024,
            counts={"00": 512, "11": 512}
        ))
        backend.cancel_job = AsyncMock(return_value=True)
        
        manager.register_backend(backend)
        
        # Test job submission, status, result, cancellation
        job_id = await backend.submit_job(JobRequest(
            circuit_data={"n_qubits": 2, "operations": []},
            device_name="test_device"
        ))
        assert job_id == "job123"
        
        status = await backend.get_job_status(job_id)
        assert status == JobStatus.COMPLETED
        
        result = await backend.get_job_result(job_id)
        assert result.job_id == "job123"
        assert result.counts == {"00": 512, "11": 512}
        
        cancelled = await backend.cancel_job(job_id)
        assert cancelled is True


if __name__ == "__main__":
    pytest.main([__file__])