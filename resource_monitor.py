#!/usr/bin/env python3
"""
Resource Monitoring Utilities
Monitors CPU, GPU, and memory usage during model inference
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import torch
import GPUtil
import json

logger = logging.getLogger(__name__)

@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement"""
    timestamp: float
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_temperature: float = 0.0

@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single model inference"""
    model_name: str
    prompt: str
    response: str
    response_time_seconds: float
    tokens_generated: int
    tokens_per_second: float
    peak_memory_gb: float
    avg_memory_gb: float
    peak_gpu_memory_gb: float
    avg_gpu_memory_gb: float
    avg_gpu_utilization: float
    max_gpu_temperature: float
    resource_snapshots: List[ResourceSnapshot]
    error: Optional[str] = None

class ResourceMonitor:
    """Monitors system resources during model inference"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.snapshots: List[ResourceSnapshot] = []
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Check GPU availability
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            try:
                self.gpus = GPUtil.getGPUs()
                logger.info(f"Found {len(self.gpus)} GPU(s)")
            except Exception as e:
                logger.warning(f"GPU monitoring may not work properly: {e}")
                self.gpus = []
        else:
            self.gpus = []
    
    def _get_cpu_memory_info(self) -> tuple:
        """Get CPU and memory information"""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        return cpu_percent, memory.used / (1024**3), memory.total / (1024**3), memory.percent
    
    def _get_gpu_info(self) -> tuple:
        """Get GPU information"""
        if not self.has_gpu or not self.gpus:
            return 0.0, 0.0, 0.0, 0.0
        
        try:
            # Refresh GPU list
            gpus = GPUtil.getGPUs()
            if not gpus:
                return 0.0, 0.0, 0.0, 0.0
            
            # Use first GPU for monitoring
            gpu = gpus[0]
            memory_used = gpu.memoryUsed / 1024  # Convert MB to GB
            memory_total = gpu.memoryTotal / 1024  # Convert MB to GB
            utilization = gpu.load * 100  # Convert to percentage
            temperature = gpu.temperature
            
            return memory_used, memory_total, utilization, temperature
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a single resource snapshot"""
        timestamp = time.time()
        cpu_percent, mem_used, mem_total, mem_percent = self._get_cpu_memory_info()
        gpu_mem_used, gpu_mem_total, gpu_util, gpu_temp = self._get_gpu_info()
        
        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_used_gb=mem_used,
            memory_total_gb=mem_total,
            memory_percent=mem_percent,
            gpu_memory_used_gb=gpu_mem_used,
            gpu_memory_total_gb=gpu_mem_total,
            gpu_utilization_percent=gpu_util,
            gpu_temperature=gpu_temp
        )
    
    def _monitor_loop(self):
        """Main monitoring loop running in separate thread"""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sampling_interval)
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self.monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring = True
        self.snapshots.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.debug("Resource monitoring started")
    
    def stop_monitoring(self) -> List[ResourceSnapshot]:
        """Stop monitoring and return collected snapshots"""
        if not self.monitoring:
            return self.snapshots
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        logger.debug(f"Resource monitoring stopped, collected {len(self.snapshots)} snapshots")
        return self.snapshots.copy()
    
    @contextmanager
    def monitor_context(self):
        """Context manager for monitoring a code block"""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()

class BenchmarkRunner:
    """Runs benchmarks on models with resource monitoring"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.monitor = ResourceMonitor(sampling_interval)
        self.results: List[BenchmarkResult] = []
    
    def count_tokens(self, text: str) -> int:
        """Simple token counting (approximation)"""
        # This is a rough approximation - in practice you'd use the model's tokenizer
        return len(text.split())
    
    def benchmark_inference(
        self,
        model_name: str,
        inference_function: Callable[[str], str],
        prompt: str,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark a single model inference
        
        Args:
            model_name: Name/identifier of the model
            inference_function: Function that takes prompt and returns response
            prompt: Input prompt for the model
            **kwargs: Additional arguments passed to inference function
        """
        logger.info(f"Benchmarking {model_name} with prompt: {prompt[:100]}...")
        
        with self.monitor.monitor_context():
            start_time = time.time()
            error = None
            response = ""
            
            try:
                response = inference_function(prompt, **kwargs)
            except Exception as e:
                error = str(e)
                logger.error(f"Inference failed for {model_name}: {e}")
            
            end_time = time.time()
            response_time = end_time - start_time
        
        # Get monitoring results
        snapshots = self.monitor.snapshots.copy()
        
        # Calculate metrics
        tokens_generated = self.count_tokens(response) if response else 0
        tokens_per_second = tokens_generated / response_time if response_time > 0 else 0
        
        # Memory metrics
        memory_values = [s.memory_used_gb for s in snapshots]
        gpu_memory_values = [s.gpu_memory_used_gb for s in snapshots]
        gpu_util_values = [s.gpu_utilization_percent for s in snapshots]
        gpu_temp_values = [s.gpu_temperature for s in snapshots]
        
        peak_memory = max(memory_values) if memory_values else 0
        avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0
        peak_gpu_memory = max(gpu_memory_values) if gpu_memory_values else 0
        avg_gpu_memory = sum(gpu_memory_values) / len(gpu_memory_values) if gpu_memory_values else 0
        avg_gpu_util = sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else 0
        max_gpu_temp = max(gpu_temp_values) if gpu_temp_values else 0
        
        result = BenchmarkResult(
            model_name=model_name,
            prompt=prompt,
            response=response,
            response_time_seconds=response_time,
            tokens_generated=tokens_generated,
            tokens_per_second=tokens_per_second,
            peak_memory_gb=peak_memory,
            avg_memory_gb=avg_memory,
            peak_gpu_memory_gb=peak_gpu_memory,
            avg_gpu_memory_gb=avg_gpu_memory,
            avg_gpu_utilization=avg_gpu_util,
            max_gpu_temperature=max_gpu_temp,
            resource_snapshots=snapshots,
            error=error
        )
        
        self.results.append(result)
        return result
    
    def save_results(self, filepath: str):
        """Save benchmark results to JSON file"""
        # Convert to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert snapshots to dicts
            result_dict['resource_snapshots'] = [asdict(snapshot) for snapshot in result.resource_snapshots]
            serializable_results.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all benchmarks"""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.error is None]
        
        if not successful_results:
            return {"error": "No successful benchmarks"}
        
        return {
            "total_benchmarks": len(self.results),
            "successful_benchmarks": len(successful_results),
            "avg_response_time": sum(r.response_time_seconds for r in successful_results) / len(successful_results),
            "avg_tokens_per_second": sum(r.tokens_per_second for r in successful_results) / len(successful_results),
            "avg_peak_memory_gb": sum(r.peak_memory_gb for r in successful_results) / len(successful_results),
            "avg_peak_gpu_memory_gb": sum(r.peak_gpu_memory_gb for r in successful_results) / len(successful_results),
            "models_tested": list(set(r.model_name for r in self.results))
        }

# Utility functions for common monitoring tasks
def quick_resource_check() -> Dict:
    """Quick system resource check"""
    monitor = ResourceMonitor()
    snapshot = monitor._take_snapshot()
    
    return {
        "cpu_percent": snapshot.cpu_percent,
        "memory_used_gb": snapshot.memory_used_gb,
        "memory_total_gb": snapshot.memory_total_gb,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_memory_used_gb": snapshot.gpu_memory_used_gb,
        "gpu_memory_total_gb": snapshot.gpu_memory_total_gb
    }

def main():
    """Demo/test of monitoring functionality"""
    logging.basicConfig(level=logging.INFO)
    
    print("System Resource Check:")
    resources = quick_resource_check()
    for key, value in resources.items():
        print(f"  {key}: {value}")
    
    # Test monitoring
    print("\nTesting resource monitoring for 3 seconds...")
    monitor = ResourceMonitor(sampling_interval=0.5)
    
    with monitor.monitor_context():
        time.sleep(3)
        # Simulate some work
        x = sum(i**2 for i in range(1000000))
    
    snapshots = monitor.snapshots
    print(f"Collected {len(snapshots)} snapshots")
    if snapshots:
        print(f"Memory range: {min(s.memory_used_gb for s in snapshots):.2f} - {max(s.memory_used_gb for s in snapshots):.2f} GB")

if __name__ == "__main__":
    main()