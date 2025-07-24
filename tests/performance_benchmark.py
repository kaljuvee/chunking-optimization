#!/usr/bin/env python3
"""
ADNOC Performance Benchmark
===========================

This module provides performance benchmarking for chunking strategies.
It measures processing speed, memory usage, and scalability.

Author: Data Engineering Team
Purpose: Performance benchmarking of chunking strategies
"""

import json
import time
import psutil
import os
import gc
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import threading
import multiprocessing
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class for performance metrics"""
    strategy_name: str
    processing_time: float
    memory_usage_mb: float
    memory_peak_mb: float
    cpu_usage_percent: float
    throughput_chunks_per_second: float
    scalability_score: float
    resource_efficiency: float

@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarking"""
    test_data_sizes: List[int] = None  # Document sizes in KB
    num_iterations: int = 5
    warmup_iterations: int = 2
    memory_monitoring: bool = True
    cpu_monitoring: bool = True
    parallel_testing: bool = False
    max_workers: int = 4
    
    def __post_init__(self):
        if self.test_data_sizes is None:
            self.test_data_sizes = [10, 50, 100, 500, 1000]  # KB

class PerformanceBenchmark:
    """
    Performance benchmarking for chunking strategies
    """
    
    def __init__(self, config: BenchmarkConfig = None):
        """Initialize the performance benchmark"""
        self.config = config or BenchmarkConfig()
        self.results = {}
        self.monitoring_active = False
        
    def benchmark_strategy(self, 
                          strategy_name: str,
                          chunking_function: Callable,
                          test_data: str,
                          **kwargs) -> PerformanceMetrics:
        """
        Benchmark a single chunking strategy
        
        Args:
            strategy_name: Name of the chunking strategy
            chunking_function: Function to benchmark
            test_data: Test data to process
            **kwargs: Additional arguments for chunking function
            
        Returns:
            PerformanceMetrics object
        """
        logger.info(f"Benchmarking strategy: {strategy_name}")
        
        # Warmup iterations
        for _ in range(self.config.warmup_iterations):
            try:
                chunking_function(test_data, **kwargs)
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
        
        # Clear memory
        gc.collect()
        
        # Start monitoring
        if self.config.memory_monitoring or self.config.cpu_monitoring:
            self._start_monitoring()
        
        # Benchmark iterations
        processing_times = []
        memory_usage = []
        cpu_usage = []
        
        for iteration in range(self.config.num_iterations):
            logger.info(f"  Iteration {iteration + 1}/{self.config.num_iterations}")
            
            # Record initial state
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            initial_cpu = psutil.cpu_percent(interval=0.1)
            
            # Execute chunking
            start_time = time.time()
            try:
                result = chunking_function(test_data, **kwargs)
                processing_time = time.time() - start_time
            except Exception as e:
                logger.error(f"Chunking failed: {e}")
                processing_time = float('inf')
                result = []
            
            # Record final state
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            final_cpu = psutil.cpu_percent(interval=0.1)
            
            # Store metrics
            processing_times.append(processing_time)
            memory_usage.append(final_memory - initial_memory)
            cpu_usage.append((initial_cpu + final_cpu) / 2)
            
            # Clear memory
            del result
            gc.collect()
        
        # Stop monitoring
        if self.config.memory_monitoring or self.config.cpu_monitoring:
            self._stop_monitoring()
        
        # Calculate metrics
        avg_processing_time = np.mean(processing_times)
        avg_memory_usage = np.mean(memory_usage)
        peak_memory_usage = max(memory_usage) if memory_usage else 0.0
        avg_cpu_usage = np.mean(cpu_usage)
        
        # Calculate throughput
        data_size_kb = len(test_data.encode('utf-8')) / 1024
        throughput = data_size_kb / avg_processing_time if avg_processing_time > 0 else 0.0
        
        # Calculate scalability score
        scalability_score = self._calculate_scalability_score(avg_processing_time, data_size_kb)
        
        # Calculate resource efficiency
        resource_efficiency = self._calculate_resource_efficiency(
            avg_processing_time, avg_memory_usage, avg_cpu_usage, data_size_kb
        )
        
        metrics = PerformanceMetrics(
            strategy_name=strategy_name,
            processing_time=avg_processing_time,
            memory_usage_mb=avg_memory_usage,
            memory_peak_mb=peak_memory_usage,
            cpu_usage_percent=avg_cpu_usage,
            throughput_chunks_per_second=throughput,
            scalability_score=scalability_score,
            resource_efficiency=resource_efficiency
        )
        
        self.results[strategy_name] = metrics
        return metrics
    
    def _start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def _stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=1.0)
    
    def _monitor_resources(self):
        """Monitor system resources"""
        while self.monitoring_active:
            try:
                # Monitor memory and CPU
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Log if thresholds exceeded
                if memory_percent > 80:
                    logger.warning(f"High memory usage: {memory_percent}%")
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage: {cpu_percent}%")
                
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                break
    
    def _calculate_scalability_score(self, processing_time: float, data_size_kb: float) -> float:
        """
        Calculate scalability score
        
        Args:
            processing_time: Processing time in seconds
            data_size_kb: Data size in KB
            
        Returns:
            Scalability score between 0 and 1
        """
        if data_size_kb <= 0 or processing_time <= 0:
            return 0.0
        
        # Calculate processing rate (KB/s)
        processing_rate = data_size_kb / processing_time
        
        # Normalize to a 0-1 scale (assuming 1000 KB/s is excellent)
        scalability_score = min(1.0, processing_rate / 1000)
        
        return scalability_score
    
    def _calculate_resource_efficiency(self, 
                                     processing_time: float, 
                                     memory_usage: float, 
                                     cpu_usage: float, 
                                     data_size_kb: float) -> float:
        """
        Calculate resource efficiency score
        
        Args:
            processing_time: Processing time in seconds
            memory_usage: Memory usage in MB
            cpu_usage: CPU usage percentage
            data_size_kb: Data size in KB
            
        Returns:
            Resource efficiency score between 0 and 1
        """
        if data_size_kb <= 0 or processing_time <= 0:
            return 0.0
        
        # Calculate efficiency factors
        time_efficiency = 1.0 / (1.0 + processing_time)  # Lower time is better
        memory_efficiency = 1.0 / (1.0 + memory_usage / 100)  # Lower memory is better
        cpu_efficiency = 1.0 / (1.0 + cpu_usage / 100)  # Lower CPU is better
        
        # Weighted combination
        efficiency_score = (
            0.4 * time_efficiency +
            0.3 * memory_efficiency +
            0.3 * cpu_efficiency
        )
        
        return efficiency_score
    
    def benchmark_scalability(self, 
                            strategy_name: str,
                            chunking_function: Callable,
                            test_data_generator: Callable,
                            **kwargs) -> Dict[str, PerformanceMetrics]:
        """
        Benchmark scalability across different data sizes
        
        Args:
            strategy_name: Name of the chunking strategy
            chunking_function: Function to benchmark
            test_data_generator: Function to generate test data of specified size
            **kwargs: Additional arguments for chunking function
            
        Returns:
            Dictionary mapping data sizes to performance metrics
        """
        logger.info(f"Benchmarking scalability for: {strategy_name}")
        
        scalability_results = {}
        
        for data_size_kb in self.config.test_data_sizes:
            logger.info(f"  Testing data size: {data_size_kb} KB")
            
            # Generate test data
            test_data = test_data_generator(data_size_kb)
            
            # Benchmark this data size
            metrics = self.benchmark_strategy(
                f"{strategy_name}_{data_size_kb}KB",
                chunking_function,
                test_data,
                **kwargs
            )
            
            scalability_results[data_size_kb] = metrics
        
        return scalability_results
    
    def benchmark_parallel(self, 
                          strategies: Dict[str, Callable],
                          test_data: str,
                          **kwargs) -> Dict[str, PerformanceMetrics]:
        """
        Benchmark multiple strategies in parallel
        
        Args:
            strategies: Dictionary mapping strategy names to functions
            test_data: Test data to process
            **kwargs: Additional arguments for chunking functions
            
        Returns:
            Dictionary mapping strategy names to performance metrics
        """
        logger.info(f"Benchmarking {len(strategies)} strategies in parallel")
        
        if not self.config.parallel_testing:
            # Sequential benchmarking
            results = {}
            for strategy_name, chunking_function in strategies.items():
                metrics = self.benchmark_strategy(
                    strategy_name, chunking_function, test_data, **kwargs
                )
                results[strategy_name] = metrics
            return results
        
        # Parallel benchmarking
        with multiprocessing.Pool(processes=self.config.max_workers) as pool:
            # Prepare arguments for parallel execution
            args_list = []
            for strategy_name, chunking_function in strategies.items():
                args_list.append((strategy_name, chunking_function, test_data, kwargs))
            
            # Execute benchmarks in parallel
            results = pool.starmap(self._benchmark_worker, args_list)
            
            # Convert results to dictionary
            return {result.strategy_name: result for result in results}
    
    def _benchmark_worker(self, strategy_name: str, chunking_function: Callable, 
                         test_data: str, kwargs: Dict) -> PerformanceMetrics:
        """
        Worker function for parallel benchmarking
        
        Args:
            strategy_name: Name of the chunking strategy
            chunking_function: Function to benchmark
            test_data: Test data to process
            kwargs: Additional arguments for chunking function
            
        Returns:
            PerformanceMetrics object
        """
        return self.benchmark_strategy(strategy_name, chunking_function, test_data, **kwargs)
    
    def compare_performance(self) -> pd.DataFrame:
        """
        Compare performance across strategies
        
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            logger.warning("No performance benchmark results available")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        data = []
        for strategy_name, metrics in self.results.items():
            data.append({
                'Strategy': strategy_name,
                'Processing_Time_Seconds': metrics.processing_time,
                'Memory_Usage_MB': metrics.memory_usage_mb,
                'Memory_Peak_MB': metrics.memory_peak_mb,
                'CPU_Usage_Percent': metrics.cpu_usage_percent,
                'Throughput_KB_per_Second': metrics.throughput_chunks_per_second,
                'Scalability_Score': metrics.scalability_score,
                'Resource_Efficiency': metrics.resource_efficiency
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_performance_report(self, output_path: str = "performance_benchmark_report.html"):
        """
        Generate performance benchmark report
        
        Args:
            output_path: Path for the output HTML report
        """
        if not self.results:
            logger.warning("No performance benchmark results available")
            return
        
        # Create comparison DataFrame
        df = self.compare_performance()
        
        # Generate HTML report
        html_content = self._generate_performance_html_report(df)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Performance benchmark report saved to: {output_path}")
    
    def _generate_performance_html_report(self, df: pd.DataFrame) -> str:
        """
        Generate HTML report for performance benchmarking
        
        Args:
            df: DataFrame with performance results
            
        Returns:
            HTML content string
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ADNOC Performance Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; color: #333; }
                .score { color: #0066cc; }
                .highlight { background-color: #e6f3ff; }
                .performance-score { background-color: #d4edda; font-weight: bold; }
                .warning { background-color: #fff3cd; }
            </style>
        </head>
        <body>
            <h1>ADNOC Performance Benchmark Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <h2>Performance Summary</h2>
            <p>This report compares {num_strategies} chunking strategies for performance metrics.</p>
            
            <h2>Detailed Performance Metrics</h2>
            {table_html}
            
            <h2>Performance Rankings</h2>
            <ul>
                <li><strong>Fastest Processing:</strong> {fastest_processing}</li>
                <li><strong>Most Memory Efficient:</strong> {most_memory_efficient}</li>
                <li><strong>Lowest CPU Usage:</strong> {lowest_cpu_usage}</li>
                <li><strong>Highest Throughput:</strong> {highest_throughput}</li>
                <li><strong>Best Scalability:</strong> {best_scalability}</li>
                <li><strong>Most Resource Efficient:</strong> {most_efficient}</li>
            </ul>
            
            <h2>Performance Recommendations</h2>
            <p>Based on the performance benchmark results, consider the following recommendations:</p>
            <ul>
                <li>Choose strategies with low processing time for real-time applications</li>
                <li>Prioritize memory efficiency for large-scale deployments</li>
                <li>Consider CPU usage for resource-constrained environments</li>
                <li>Evaluate throughput for high-volume processing requirements</li>
                <li>Assess scalability for growing data volumes</li>
                <li>Balance performance with resource efficiency</li>
            </ul>
            
            <h2>Performance Insights</h2>
            <p>The benchmark reveals important performance characteristics:</p>
            <ul>
                <li>Processing time scales with data size and complexity</li>
                <li>Memory usage varies significantly between strategies</li>
                <li>CPU utilization depends on algorithm complexity</li>
                <li>Throughput is critical for production systems</li>
                <li>Scalability determines long-term viability</li>
            </ul>
            
            <h2>System Requirements</h2>
            <p>Based on the benchmark results, recommended system requirements:</p>
            <ul>
                <li><strong>Minimum Memory:</strong> {min_memory} MB</li>
                <li><strong>Recommended Memory:</strong> {rec_memory} MB</li>
                <li><strong>CPU Cores:</strong> {cpu_cores} cores</li>
                <li><strong>Processing Speed:</strong> {processing_speed} KB/s</li>
            </ul>
        </body>
        </html>
        """
        
        # Generate table HTML
        table_html = df.to_html(index=False, classes='dataframe', float_format='%.3f')
        
        # Find best performers
        fastest_processing = df.loc[df['Processing_Time_Seconds'].idxmin(), 'Strategy'] if not df.empty else 'N/A'
        most_memory_efficient = df.loc[df['Memory_Usage_MB'].idxmin(), 'Strategy'] if not df.empty else 'N/A'
        lowest_cpu_usage = df.loc[df['CPU_Usage_Percent'].idxmin(), 'Strategy'] if not df.empty else 'N/A'
        highest_throughput = df.loc[df['Throughput_KB_per_Second'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_scalability = df.loc[df['Scalability_Score'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        most_efficient = df.loc[df['Resource_Efficiency'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        
        # Calculate system requirements
        min_memory = int(df['Memory_Peak_MB'].max() * 1.2) if not df.empty else 0
        rec_memory = int(df['Memory_Peak_MB'].max() * 2) if not df.empty else 0
        cpu_cores = max(2, int(df['CPU_Usage_Percent'].max() / 25)) if not df.empty else 2
        processing_speed = int(df['Throughput_KB_per_Second'].max()) if not df.empty else 0
        
        # Format the HTML
        html_content = html_template.format(
            timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            num_strategies=len(df),
            table_html=table_html,
            fastest_processing=fastest_processing,
            most_memory_efficient=most_memory_efficient,
            lowest_cpu_usage=lowest_cpu_usage,
            highest_throughput=highest_throughput,
            best_scalability=best_scalability,
            most_efficient=most_efficient,
            min_memory=min_memory,
            rec_memory=rec_memory,
            cpu_cores=cpu_cores,
            processing_speed=processing_speed
        )
        
        return html_content

def generate_test_data(size_kb: int) -> str:
    """
    Generate test data of specified size
    
    Args:
        size_kb: Target size in KB
        
    Returns:
        Generated test data
    """
    # Generate synthetic text content
    words = [
        "oil", "gas", "production", "reservoir", "well", "drilling", "completion",
        "stimulation", "artificial", "lift", "enhanced", "recovery", "carbonate",
        "sandstone", "porosity", "permeability", "saturation", "viscosity",
        "pressure", "temperature", "flowrate", "water", "cut", "formation",
        "damage", "skin", "factor", "efficiency", "optimization", "performance"
    ]
    
    # Calculate target word count (approximately 5 characters per word)
    target_chars = size_kb * 1024
    target_words = target_chars // 5
    
    # Generate text
    text_parts = []
    current_size = 0
    
    while current_size < target_chars:
        # Create a sentence
        sentence_length = random.randint(10, 20)
        sentence_words = random.choices(words, k=sentence_length)
        sentence = " ".join(sentence_words).capitalize() + "."
        
        text_parts.append(sentence)
        current_size += len(sentence) + 1  # +1 for space
    
    return " ".join(text_parts)

def main():
    """Example usage of the performance benchmark"""
    # Initialize benchmark
    config = BenchmarkConfig(
        test_data_sizes=[10, 50, 100],
        num_iterations=3,
        warmup_iterations=1,
        memory_monitoring=True,
        cpu_monitoring=True,
        parallel_testing=False
    )
    
    benchmark = PerformanceBenchmark(config)
    
    # Example: Benchmark a chunking strategy
    # This would typically be called with actual chunking functions
    logger.info("Performance benchmark initialized successfully")
    logger.info("Use benchmark_strategy() to benchmark specific strategies")
    logger.info("Use benchmark_scalability() to test scalability")
    logger.info("Use compare_performance() to compare multiple strategies")

if __name__ == "__main__":
    main() 