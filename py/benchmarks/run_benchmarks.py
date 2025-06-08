#!/usr/bin/env python3
"""
Main benchmark runner for QuantRS2

This script runs all benchmarks and generates a comprehensive report.
"""

import sys
import os
import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import platform
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class BenchmarkRunner:
    """Orchestrate all benchmarks and generate reports."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = self.output_dir / f"report_{self.timestamp}"
        self.report_dir.mkdir(exist_ok=True)
        
    def run_benchmark_module(self, module_name: str, script_name: str) -> bool:
        """Run a single benchmark module."""
        print(f"\n{'='*60}")
        print(f"Running {module_name}")
        print('='*60)
        
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            print(f"Warning: {script_path} not found, skipping")
            return False
        
        try:
            # Run as subprocess to isolate environments
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Error running {module_name}:")
                print(result.stderr)
                return False
            
            print(result.stdout)
            return True
            
        except Exception as e:
            print(f"Failed to run {module_name}: {e}")
            return False
    
    def collect_results(self) -> Dict:
        """Collect all benchmark results."""
        results = {
            'timestamp': self.timestamp,
            'system_info': self.get_system_info(),
            'benchmarks': {}
        }
        
        # Collect main benchmark results
        main_results = self.output_dir / f"results_{self.timestamp}.json"
        if main_results.exists():
            with open(main_results) as f:
                results['benchmarks']['main'] = json.load(f)
        
        # Collect memory benchmark results
        memory_results = self.output_dir / "memory" / "memory_results.json"
        if memory_results.exists():
            with open(memory_results) as f:
                results['benchmarks']['memory'] = json.load(f)
        
        # Collect parallel benchmark results
        parallel_results = self.output_dir / "parallel" / "parallel_results.json"
        if parallel_results.exists():
            with open(parallel_results) as f:
                results['benchmarks']['parallel'] = json.load(f)
        
        return results
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        try:
            import quantrs2
            quantrs2_version = getattr(quantrs2, '__version__', 'unknown')
        except:
            quantrs2_version = 'not_installed'
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu': platform.processor() or 'Unknown',
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_gb': psutil.virtual_memory().total / 1e9,
            'quantrs2_version': quantrs2_version,
            'timestamp': self.timestamp
        }
    
    def generate_report(self, results: Dict):
        """Generate comprehensive HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>QuantRS2 Benchmark Report - {self.timestamp}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .system-info {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .benchmark-section {{
            margin-bottom: 30px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .success {{
            color: green;
        }}
        .error {{
            color: red;
        }}
        .warning {{
            color: orange;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 10px 0;
        }}
        .summary-box {{
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>QuantRS2 Benchmark Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="system-info">
            <h2>System Information</h2>
            <table>
                <tr><td><strong>Platform</strong></td><td>{results['system_info']['platform']}</td></tr>
                <tr><td><strong>Python Version</strong></td><td>{results['system_info']['python_version']}</td></tr>
                <tr><td><strong>CPU</strong></td><td>{results['system_info']['cpu']}</td></tr>
                <tr><td><strong>CPU Cores</strong></td><td>{results['system_info']['cpu_count']} physical, {results['system_info']['cpu_count_logical']} logical</td></tr>
                <tr><td><strong>Memory</strong></td><td>{results['system_info']['memory_gb']:.1f} GB</td></tr>
                <tr><td><strong>QuantRS2 Version</strong></td><td>{results['system_info']['quantrs2_version']}</td></tr>
            </table>
        </div>
"""
        
        # Add benchmark results
        if 'main' in results['benchmarks']:
            html_content += self._generate_main_benchmark_section(results['benchmarks']['main'])
        
        if 'memory' in results['benchmarks']:
            html_content += self._generate_memory_benchmark_section(results['benchmarks']['memory'])
        
        if 'parallel' in results['benchmarks']:
            html_content += self._generate_parallel_benchmark_section(results['benchmarks']['parallel'])
        
        # Add summary and recommendations
        html_content += self._generate_summary_section(results)
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Save report
        report_path = self.report_dir / "benchmark_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"\nReport generated: {report_path}")
        
        # Also save raw results
        with open(self.report_dir / "raw_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    def _generate_main_benchmark_section(self, data: Dict) -> str:
        """Generate HTML for main benchmarks."""
        if 'results' not in data:
            return ""
        
        html = """
        <div class="benchmark-section">
            <h2>Performance Benchmarks</h2>
"""
        
        # Group by category
        results_by_category = {}
        for result in data['results']:
            category = result.get('category', 'Unknown')
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)
        
        for category, results in results_by_category.items():
            html += f"<h3>{category}</h3><table>"
            html += "<tr><th>Benchmark</th><th>Parameters</th><th>Time (ms)</th><th>Memory (MB)</th><th>Status</th></tr>"
            
            for r in results:
                status = '<span class="error">Error</span>' if r.get('error') else '<span class="success">Success</span>'
                params = ', '.join(f"{k}={v}" for k, v in r.get('parameters', {}).items())
                time_ms = f"{r.get('execution_time', 0) * 1000:.2f}" if not r.get('error') else 'N/A'
                memory_mb = f"{r.get('memory_usage', 0):.2f}" if not r.get('error') else 'N/A'
                
                html += f"<tr><td>{r['name']}</td><td>{params}</td><td>{time_ms}</td><td>{memory_mb}</td><td>{status}</td></tr>"
            
            html += "</table>"
        
        html += "</div>"
        return html
    
    def _generate_memory_benchmark_section(self, data: List) -> str:
        """Generate HTML for memory benchmarks."""
        html = """
        <div class="benchmark-section">
            <h2>Memory Usage Analysis</h2>
            <table>
                <tr><th>Test</th><th>Parameters</th><th>Peak Memory (MB)</th><th>Status</th></tr>
"""
        
        for test in data:
            status = '<span class="success">Success</span>' if test['success'] else '<span class="error">Error</span>'
            params = ', '.join(f"{k}={v}" for k, v in test.get('parameters', {}).items())
            
            html += f"""
                <tr>
                    <td>{test['name']}</td>
                    <td>{params}</td>
                    <td>{test['peak_traced_mb']:.2f}</td>
                    <td>{status}</td>
                </tr>
"""
        
        html += """
            </table>
        </div>
"""
        return html
    
    def _generate_parallel_benchmark_section(self, data: List) -> str:
        """Generate HTML for parallel benchmarks."""
        html = """
        <div class="benchmark-section">
            <h2>Parallel Performance Analysis</h2>
"""
        
        for benchmark in data:
            html += f"<h3>{benchmark['name']}</h3>"
            
            if benchmark['type'] == 'thread_scaling':
                html += "<table><tr><th>Threads</th><th>Time (s)</th><th>Speedup</th><th>Efficiency</th></tr>"
                for d in benchmark['data']:
                    html += f"""
                        <tr>
                            <td>{d['threads']}</td>
                            <td>{d['time']:.3f}</td>
                            <td>{d['speedup']:.2f}x</td>
                            <td>{d['efficiency']*100:.1f}%</td>
                        </tr>
"""
                html += "</table>"
            
            elif benchmark['type'] == 'gpu_comparison':
                html += "<table><tr><th>Size</th><th>CPU Time (s)</th><th>GPU Time (s)</th><th>Speedup</th></tr>"
                for d in benchmark['data']:
                    gpu_time = f"{d['gpu_time']:.3f}" if d['gpu_available'] else "N/A"
                    speedup = f"{d['speedup']:.2f}x" if d['gpu_available'] else "N/A"
                    html += f"""
                        <tr>
                            <td>{d['size']}</td>
                            <td>{d['cpu_time']:.3f}</td>
                            <td>{gpu_time}</td>
                            <td>{speedup}</td>
                        </tr>
"""
                html += "</table>"
        
        html += "</div>"
        return html
    
    def _generate_summary_section(self, results: Dict) -> str:
        """Generate summary and recommendations."""
        html = """
        <div class="summary-box">
            <h2>Summary and Recommendations</h2>
            <ul>
"""
        
        # Analyze results and provide recommendations
        recommendations = []
        
        # Check GPU availability
        if results['system_info'].get('quantrs2_version') == 'not_installed':
            recommendations.append("QuantRS2 is not installed. Install with: pip install quantrs2")
        
        # Memory recommendations
        if 'memory' in results['benchmarks']:
            memory_tests = results['benchmarks']['memory']
            max_memory = max((t['peak_traced_mb'] for t in memory_tests if t['success']), default=0)
            if max_memory > results['system_info']['memory_gb'] * 1000 * 0.5:
                recommendations.append(f"Memory usage peaked at {max_memory:.0f} MB, which is over 50% of available RAM. Consider using tensor network methods for larger problems.")
        
        # Parallel recommendations
        if 'parallel' in results['benchmarks']:
            for bench in results['benchmarks']['parallel']:
                if bench['type'] == 'thread_scaling' and bench['data']:
                    best_threads = max(bench['data'], key=lambda x: x['speedup'])['threads']
                    if best_threads < results['system_info']['cpu_count_logical']:
                        recommendations.append(f"Optimal thread count ({best_threads}) is less than available cores. This may indicate I/O or memory bottlenecks.")
        
        if not recommendations:
            recommendations.append("All benchmarks completed successfully. Performance appears optimal for your system.")
        
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
            </ul>
        </div>
"""
        return html
    
    def run_all_benchmarks(self, skip_modules: Optional[List[str]] = None):
        """Run all benchmark modules."""
        skip_modules = skip_modules or []
        
        benchmarks = [
            ("Comprehensive Benchmarks", "benchmark_suite.py"),
            ("Memory Benchmarks", "memory_benchmark.py"),
            ("Parallel Benchmarks", "parallel_benchmark.py")
        ]
        
        success_count = 0
        
        for name, script in benchmarks:
            if any(skip in name.lower() for skip in skip_modules):
                print(f"Skipping {name}")
                continue
                
            if self.run_benchmark_module(name, script):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Completed {success_count}/{len(benchmarks)} benchmark modules")
        print('='*60)
        
        # Collect and generate report
        results = self.collect_results()
        self.generate_report(results)
        
        return success_count == len(benchmarks)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run QuantRS2 benchmarks")
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--skip', nargs='*', default=[],
                       help='Skip specific benchmark modules')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmarks only')
    
    args = parser.parse_args()
    
    print("QuantRS2 Benchmark Suite")
    print("=" * 60)
    
    runner = BenchmarkRunner(args.output_dir)
    
    if args.quick:
        print("Running in quick mode (limited benchmarks)")
        args.skip.extend(['memory', 'parallel'])
    
    success = runner.run_all_benchmarks(skip_modules=args.skip)
    
    if success:
        print("\nAll benchmarks completed successfully!")
    else:
        print("\nSome benchmarks failed. Check the report for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()