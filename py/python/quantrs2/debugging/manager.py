"""
Quantum Debugging Tools Manager for QuantRS2.

This module provides the main manager class that coordinates all
debugging components and provides a unified interface.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

from .core import (
    DebugLevel, DebugSession, InspectionMode, ErrorType, ValidationRule
)
from .state_inspector import QuantumStateInspector
from .error_analyzer import QuantumErrorAnalyzer
from .circuit_validator import QuantumCircuitValidator
from .memory_debugger import QuantumMemoryDebugger
from .interactive_console import InteractiveQuantumDebugConsole
from .web_interface import QuantumDebuggingWebInterface

logger = logging.getLogger(__name__)

class QuantumDebuggingToolsManager:
    """
    Main manager for all quantum debugging tools.
    
    This class coordinates all debugging components and provides
    a unified interface for quantum circuit debugging and analysis.
    """
    
    def __init__(self, debug_level: DebugLevel = DebugLevel.INFO):
        self.debug_level = debug_level
        self.active_sessions: Dict[str, DebugSession] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize debugging components
        self.state_inspector = QuantumStateInspector(debug_level)
        self.error_analyzer = QuantumErrorAnalyzer(debug_level)
        self.circuit_validator = QuantumCircuitValidator(debug_level)
        self.memory_debugger = QuantumMemoryDebugger(debug_level)
        self.interactive_console = InteractiveQuantumDebugConsole(debug_level)
        self.web_interface = QuantumDebuggingWebInterface(debug_level)
        
        # Start memory monitoring
        self.memory_debugger.start_monitoring()
        
        logger.info(f"QuantRS2 Debugging Tools Manager initialized with level: {debug_level.name}")
    
    def create_debug_session(self, circuit: Any, session_id: Optional[str] = None) -> str:
        """
        Create a new debugging session.
        
        Args:
            circuit: The quantum circuit to debug
            session_id: Optional session identifier
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"
        
        session = DebugSession(
            session_id=session_id,
            circuit=circuit,
            start_time=time.time()
        )
        
        self.active_sessions[session_id] = session
        
        # Take initial memory snapshot
        self.memory_debugger.take_snapshot(f"session_start_{session_id}")
        
        logger.info(f"Created debug session: {session_id}")
        return session_id
    
    def end_debug_session(self, session_id: str) -> bool:
        """End a debugging session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        session = self.active_sessions[session_id]
        session.end_time = time.time()
        
        # Take final memory snapshot
        self.memory_debugger.take_snapshot(f"session_end_{session_id}")
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logger.info(f"Ended debug session: {session_id}")
        return True
    
    def debug_circuit_comprehensive(
        self,
        circuit: Any,
        session_id: Optional[str] = None,
        inspect_states: bool = True,
        validate_circuit: bool = True,
        monitor_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive debugging of a quantum circuit.
        
        Args:
            circuit: The quantum circuit to debug
            session_id: Optional session identifier
            inspect_states: Whether to perform state inspection
            validate_circuit: Whether to validate the circuit
            monitor_memory: Whether to monitor memory usage
            
        Returns:
            Comprehensive debugging results
        """
        if session_id is None:
            session_id = self.create_debug_session(circuit)
        
        results = {
            "session_id": session_id,
            "timestamp": time.time(),
            "circuit_info": self._get_circuit_info(circuit),
            "state_inspection": {},
            "circuit_validation": {},
            "memory_analysis": {},
            "errors": [],
            "recommendations": []
        }
        
        try:
            # State inspection
            if inspect_states:
                results["state_inspection"] = self._perform_state_inspection(circuit, session_id)
            
            # Circuit validation
            if validate_circuit:
                results["circuit_validation"] = self._perform_circuit_validation(circuit, session_id)
            
            # Memory monitoring
            if monitor_memory:
                results["memory_analysis"] = self._perform_memory_analysis(session_id)
            
            # Generate overall recommendations
            results["recommendations"] = self._generate_comprehensive_recommendations(results)
            
            logger.info(f"Comprehensive debugging completed for session: {session_id}")
            
        except Exception as e:
            error_diagnosis = self.error_analyzer.analyze_error(e, {"session_id": session_id})
            results["errors"].append(error_diagnosis)
            logger.error(f"Debugging failed for session {session_id}: {e}")
        
        return results
    
    def _get_circuit_info(self, circuit: Any) -> Dict[str, Any]:
        """Extract basic circuit information."""
        try:
            # Placeholder implementation
            return {
                "type": type(circuit).__name__,
                "gates": "unknown",  # Would extract actual gate count
                "qubits": "unknown",  # Would extract qubit count
                "depth": "unknown"   # Would calculate circuit depth
            }
        except Exception as e:
            logger.warning(f"Failed to extract circuit info: {e}")
            return {"error": str(e)}
    
    def _perform_state_inspection(self, circuit: Any, session_id: str) -> Dict[str, Any]:
        """Perform comprehensive state inspection."""
        results = {}
        
        try:
            # Placeholder: In real implementation, would extract state vector from circuit
            # For now, create a simple test state
            import numpy as np
            test_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
            
            # Perform different types of inspections
            inspection_modes = [
                InspectionMode.AMPLITUDE_ANALYSIS,
                InspectionMode.PROBABILITY_ANALYSIS,
                InspectionMode.PHASE_ANALYSIS,
                InspectionMode.ENTANGLEMENT_ANALYSIS
            ]
            
            for mode in inspection_modes:
                inspection_result = self.state_inspector.inspect_state(test_state, mode)
                results[mode.name.lower()] = {
                    "analysis_data": inspection_result.analysis_data,
                    "insights": inspection_result.insights,
                    "anomalies": inspection_result.anomalies,
                    "recommendations": inspection_result.recommendations
                }
            
        except Exception as e:
            logger.error(f"State inspection failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _perform_circuit_validation(self, circuit: Any, session_id: str) -> Dict[str, Any]:
        """Perform circuit validation."""
        results = {}
        
        try:
            # Validate using all available rules
            validation_results = self.circuit_validator.validate_circuit(circuit)
            
            results["validation_results"] = []
            for validation_result in validation_results:
                results["validation_results"].append({
                    "rule": validation_result.rule.name,
                    "passed": validation_result.passed,
                    "message": validation_result.message,
                    "details": validation_result.details,
                    "suggestions": validation_result.suggestions
                })
            
            # Summary
            passed_count = sum(1 for vr in validation_results if vr.passed)
            total_count = len(validation_results)
            
            results["summary"] = {
                "total_rules": total_count,
                "passed_rules": passed_count,
                "failed_rules": total_count - passed_count,
                "overall_passed": passed_count == total_count
            }
            
        except Exception as e:
            logger.error(f"Circuit validation failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _perform_memory_analysis(self, session_id: str) -> Dict[str, Any]:
        """Perform memory analysis."""
        try:
            # Take memory snapshot
            snapshot = self.memory_debugger.take_snapshot(f"analysis_{session_id}")
            
            # Get memory report
            memory_report = self.memory_debugger.get_memory_report()
            
            # Detect potential leaks
            leaks = self.memory_debugger.detect_leaks()
            
            return {
                "current_snapshot": {
                    "used_memory_mb": snapshot.used_memory / 1024 / 1024,
                    "peak_memory_mb": snapshot.peak_memory / 1024 / 1024,
                    "recommendations": snapshot.recommendations
                },
                "memory_report": memory_report,
                "potential_leaks": leaks
            }
            
        except Exception as e:
            logger.error(f"Memory analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_comprehensive_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on all analyses."""
        recommendations = []
        
        try:
            # Analyze state inspection results
            if "state_inspection" in results:
                for mode, data in results["state_inspection"].items():
                    if "recommendations" in data:
                        recommendations.extend(data["recommendations"])
            
            # Analyze validation results
            if "circuit_validation" in results:
                validation_data = results["circuit_validation"]
                if "validation_results" in validation_data:
                    for vr in validation_data["validation_results"]:
                        if not vr["passed"] and "suggestions" in vr:
                            recommendations.extend(vr["suggestions"])
            
            # Analyze memory results
            if "memory_analysis" in results:
                memory_data = results["memory_analysis"]
                if "current_snapshot" in memory_data:
                    recommendations.extend(memory_data["current_snapshot"].get("recommendations", []))
            
            # Remove duplicates while preserving order
            unique_recommendations = []
            seen = set()
            for rec in recommendations:
                if rec not in seen:
                    unique_recommendations.append(rec)
                    seen.add(rec)
            
            return unique_recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return ["Review debugging results manually"]
    
    def start_interactive_console(self):
        """Start the interactive debugging console."""
        try:
            logger.info("Starting interactive quantum debugging console...")
            self.interactive_console.cmdloop()
        except Exception as e:
            logger.error(f"Interactive console failed: {e}")
    
    def start_web_interface(self, host: str = '0.0.0.0', port: int = 5000):
        """Start the web debugging interface."""
        try:
            logger.info(f"Starting web debugging interface on {host}:{port}")
            self.web_interface.start_server(host=host, debug=(self.debug_level == DebugLevel.DEBUG))
        except Exception as e:
            logger.error(f"Web interface failed: {e}")
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a debugging session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session.session_id,
            "start_time": session.start_time,
            "state": session.state.name,
            "breakpoints": len(session.breakpoints),
            "frames": len(session.frames),
            "errors": len(session.errors),
            "validations": len(session.validations)
        }
    
    def cleanup(self):
        """Cleanup debugging tools and resources."""
        try:
            # End all active sessions
            for session_id in list(self.active_sessions.keys()):
                self.end_debug_session(session_id)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Final memory snapshot
            self.memory_debugger.take_snapshot("cleanup")
            
            logger.info("QuantRS2 Debugging Tools Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()