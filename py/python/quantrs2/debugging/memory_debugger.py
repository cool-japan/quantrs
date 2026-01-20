"""
Quantum Memory Debugger for QuantRS2.

This module provides memory debugging and leak detection capabilities
specifically designed for quantum computing applications.
"""

import logging
import gc
import time
from typing import Dict, List, Any

from .core import DebugLevel, MemoryDebugInfo

logger = logging.getLogger(__name__)

class QuantumMemoryDebugger:
    """
    Memory debugging and leak detection for quantum applications.
    
    This class monitors memory usage, detects leaks, and provides
    optimization suggestions for quantum computing workloads.
    """
    
    def __init__(self, debug_level: DebugLevel = DebugLevel.INFO):
        self.debug_level = debug_level
        self.memory_snapshots = []
        self.allocation_tracking = {}
        self.peak_memory = 0
        self.baseline_memory = 0
    
    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        try:
            import psutil
            process = psutil.Process()
            self.baseline_memory = process.memory_info().rss
            logger.info(f"Memory monitoring started. Baseline: {self.baseline_memory / 1024 / 1024:.2f} MB")
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            self.baseline_memory = 0
    
    def take_snapshot(self, label: str = "") -> MemoryDebugInfo:
        """Take a memory snapshot."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            total_memory = memory_info.vms
            used_memory = memory_info.rss
            free_memory = total_memory - used_memory
            
            if used_memory > self.peak_memory:
                self.peak_memory = used_memory
            
            # Force garbage collection
            collected = gc.collect()
            
            debug_info = MemoryDebugInfo(
                total_memory=total_memory,
                used_memory=used_memory,
                free_memory=free_memory,
                peak_memory=self.peak_memory,
                allocation_count=len(gc.get_objects()),
                deallocation_count=collected,
                memory_leaks=[],
                recommendations=self._generate_memory_recommendations(used_memory),
                timestamp=time.time()
            )
            
            self.memory_snapshots.append((label, debug_info))
            
            if self.debug_level in [DebugLevel.DEBUG, DebugLevel.TRACE]:
                logger.debug(f"Memory snapshot '{label}': {used_memory / 1024 / 1024:.2f} MB")
            
            return debug_info
            
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return MemoryDebugInfo(
                total_memory=0,
                used_memory=0,
                free_memory=0,
                peak_memory=0,
                allocation_count=0,
                deallocation_count=0,
                memory_leaks=[],
                recommendations=["Install psutil for memory monitoring"]
            )
        except Exception as e:
            logger.error(f"Memory snapshot failed: {e}")
            return MemoryDebugInfo(
                total_memory=0,
                used_memory=0,
                free_memory=0,
                peak_memory=0,
                allocation_count=0,
                deallocation_count=0,
                memory_leaks=[],
                recommendations=[f"Memory monitoring error: {e}"]
            )
    
    def detect_leaks(self) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        leaks = []
        
        if len(self.memory_snapshots) < 2:
            return leaks
        
        # Compare recent snapshots
        recent_snapshots = self.memory_snapshots[-5:]  # Last 5 snapshots
        
        if len(recent_snapshots) >= 2:
            memory_trend = []
            for _, snapshot in recent_snapshots:
                memory_trend.append(snapshot.used_memory)
            
            # Check for consistent memory growth
            if len(memory_trend) >= 3:
                growth_count = 0
                for i in range(1, len(memory_trend)):
                    if memory_trend[i] > memory_trend[i-1]:
                        growth_count += 1
                
                # If memory consistently grows
                if growth_count >= len(memory_trend) - 2:
                    growth_rate = (memory_trend[-1] - memory_trend[0]) / len(memory_trend)
                    if growth_rate > 1024 * 1024:  # More than 1MB growth per snapshot
                        leaks.append({
                            "type": "consistent_growth",
                            "description": "Memory usage consistently increasing",
                            "growth_rate_mb": growth_rate / 1024 / 1024,
                            "snapshots_analyzed": len(memory_trend)
                        })
        
        return leaks
    
    def _generate_memory_recommendations(self, current_memory: int) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        memory_mb = current_memory / 1024 / 1024
        
        if memory_mb > 1000:  # More than 1GB
            recommendations.append("Consider reducing qubit count or circuit depth")
            recommendations.append("Use sparse matrix representation for large quantum states")
        
        if memory_mb > 2000:  # More than 2GB
            recommendations.append("Critical: Memory usage is very high")
            recommendations.append("Consider distributed quantum simulation")
        
        if self.peak_memory > current_memory * 1.5:
            recommendations.append("Memory usage varies significantly - check for temporary allocations")
        
        return recommendations
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Get memory usage before and after
            snapshot_before = self.take_snapshot("before_optimization")
            
            # Additional optimization steps would go here
            # For now, just garbage collection
            
            snapshot_after = self.take_snapshot("after_optimization")
            
            memory_saved = snapshot_before.used_memory - snapshot_after.used_memory
            
            return {
                "success": True,
                "objects_collected": collected,
                "memory_saved_mb": memory_saved / 1024 / 1024,
                "memory_before_mb": snapshot_before.used_memory / 1024 / 1024,
                "memory_after_mb": snapshot_after.used_memory / 1024 / 1024
            }
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate a comprehensive memory report."""
        if not self.memory_snapshots:
            return {"error": "No memory snapshots available"}
        
        current_snapshot = self.memory_snapshots[-1][1]
        leaks = self.detect_leaks()
        
        return {
            "current_memory_mb": current_snapshot.used_memory / 1024 / 1024,
            "peak_memory_mb": self.peak_memory / 1024 / 1024,
            "baseline_memory_mb": self.baseline_memory / 1024 / 1024,
            "memory_growth_mb": (current_snapshot.used_memory - self.baseline_memory) / 1024 / 1024,
            "snapshots_taken": len(self.memory_snapshots),
            "potential_leaks": len(leaks),
            "leak_details": leaks,
            "recommendations": current_snapshot.recommendations
        }