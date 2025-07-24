#!/usr/bin/env python3
"""
ADNOC Comprehensive Evaluation Runner
====================================

This script runs all evaluation components for chunking strategies:
- Basic evaluation framework
- RAG-specific evaluation
- Performance benchmarking
- Coherence evaluation
- Synthetic data generation

Author: Data Engineering Team
Purpose: Comprehensive chunking strategy evaluation
"""

import json
import argparse
import os
import sys
from pathlib import Path
import logging
from typing import Dict, List, Any
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from tests.evaluation_framework import ChunkingEvaluator, EvaluationConfig
from tests.rag_evaluator import RAGEvaluator, RAGEvaluationConfig
from tests.performance_benchmark import PerformanceBenchmark, BenchmarkConfig
from tests.coherence_evaluator import CoherenceEvaluator, CoherenceConfig
from tests.synthetic_data_generator import SyntheticDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """
    Comprehensive evaluator that runs all evaluation components
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the comprehensive evaluator"""
        self.config = config or self._get_default_config()
        self.results = {}
        
        # Initialize evaluation components
        self.basic_evaluator = ChunkingEvaluator(
            EvaluationConfig(
                max_chunks=self.config['max_chunks'],
                batch_size=self.config['batch_size'],
                enable_caching=self.config['enable_caching']
            )
        )
        
        self.rag_evaluator = RAGEvaluator(
            RAGEvaluationConfig(
                max_retrieved_chunks=self.config['max_retrieved_chunks'],
                similarity_threshold=self.config['similarity_threshold'],
                evaluation_mode=self.config['evaluation_mode']
            )
        )
        
        self.performance_benchmark = PerformanceBenchmark(
            BenchmarkConfig(
                test_data_sizes=self.config['test_data_sizes'],
                num_iterations=self.config['num_iterations'],
                memory_monitoring=self.config['memory_monitoring'],
                cpu_monitoring=self.config['cpu_monitoring']
            )
        )
        
        self.coherence_evaluator = CoherenceEvaluator(
            CoherenceConfig(
                min_chunk_size=self.config['min_chunk_size'],
                max_chunk_size=self.config['max_chunk_size'],
                similarity_threshold=self.config['similarity_threshold']
            )
        )
        
        self.data_generator = SyntheticDataGenerator()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_chunks': 100,
            'batch_size': 10,
            'enable_caching': True,
            'max_retrieved_chunks': 5,
            'similarity_threshold': 0.7,
            'evaluation_mode': 'automatic',
            'test_data_sizes': [10, 50, 100, 500],
            'num_iterations': 3,
            'memory_monitoring': True,
            'cpu_monitoring': True,
            'min_chunk_size': 20,
            'max_chunk_size': 1000,
            'output_dir': 'evaluation_results',
            'generate_synthetic_data': True,
            'num_synthetic_docs': 5
        }
    
    def run_comprehensive_evaluation(self, 
                                   chunking_strategies: Dict[str, List[Dict]],
                                   test_data_path: str = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of chunking strategies
        
        Args:
            chunking_strategies: Dictionary mapping strategy names to chunk lists
            test_data_path: Path to test data (optional)
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting comprehensive evaluation")
        start_time = time.time()
        
        # Create output directory
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Generate synthetic data if needed
        if self.config['generate_synthetic_data'] and not test_data_path:
            logger.info("Generating synthetic test data")
            synthetic_files = self.data_generator.generate_test_dataset(
                output_dir="data/synthetic",
                num_documents=self.config['num_synthetic_docs']
            )
            test_data_path = "data/synthetic"
        
        # Run each evaluation component
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'strategies': list(chunking_strategies.keys()),
            'basic_evaluation': {},
            'rag_evaluation': {},
            'performance_benchmark': {},
            'coherence_evaluation': {},
            'summary': {}
        }
        
        # 1. Basic evaluation
        logger.info("Running basic evaluation")
        for strategy_name, chunks in chunking_strategies.items():
            metrics = self.basic_evaluator.evaluate_chunking_strategy(strategy_name, chunks)
            evaluation_results['basic_evaluation'][strategy_name] = {
                'num_chunks': metrics.num_chunks,
                'avg_chunk_size': metrics.avg_chunk_size,
                'coherence_score': metrics.coherence_score,
                'retrieval_accuracy': metrics.retrieval_accuracy,
                'rag_performance': metrics.rag_performance,
                'user_satisfaction': metrics.user_satisfaction
            }
        
        # 2. RAG evaluation
        logger.info("Running RAG evaluation")
        for strategy_name, chunks in chunking_strategies.items():
            metrics = self.rag_evaluator.evaluate_rag_performance(strategy_name, chunks)
            evaluation_results['rag_evaluation'][strategy_name] = {
                'retrieval_precision': metrics.retrieval_precision,
                'retrieval_recall': metrics.retrieval_recall,
                'retrieval_f1': metrics.retrieval_f1,
                'answer_accuracy': metrics.answer_accuracy,
                'context_relevance': metrics.context_relevance,
                'response_coherence': metrics.response_coherence,
                'response_completeness': metrics.response_completeness,
                'overall_rag_score': metrics.overall_rag_score
            }
        
        # 3. Performance benchmarking
        logger.info("Running performance benchmarking")
        for strategy_name, chunks in chunking_strategies.items():
            # Create a simple chunking function for benchmarking
            def chunking_function(text, strategy_name=strategy_name, chunks=chunks):
                return chunks
            
            # Use first chunk as test data for benchmarking
            if chunks:
                test_data = chunks[0].get('text', 'Sample text for benchmarking')
                metrics = self.performance_benchmark.benchmark_strategy(
                    strategy_name, chunking_function, test_data
                )
                evaluation_results['performance_benchmark'][strategy_name] = {
                    'processing_time': metrics.processing_time,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'cpu_usage_percent': metrics.cpu_usage_percent,
                    'throughput_kb_per_second': metrics.throughput_chunks_per_second,
                    'scalability_score': metrics.scalability_score,
                    'resource_efficiency': metrics.resource_efficiency
                }
        
        # 4. Coherence evaluation
        logger.info("Running coherence evaluation")
        for strategy_name, chunks in chunking_strategies.items():
            metrics = self.coherence_evaluator.evaluate_coherence(strategy_name, chunks)
            evaluation_results['coherence_evaluation'][strategy_name] = {
                'semantic_coherence': metrics.semantic_coherence,
                'topic_coherence': metrics.topic_coherence,
                'discourse_coherence': metrics.discourse_coherence,
                'context_preservation': metrics.context_preservation,
                'boundary_quality': metrics.boundary_quality,
                'overall_coherence': metrics.overall_coherence
            }
        
        # 5. Generate summary
        evaluation_results['summary'] = self._generate_summary(evaluation_results)
        
        # 6. Save results
        results_file = output_dir / f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        # 7. Generate reports
        self._generate_reports(evaluation_results, output_dir)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Comprehensive evaluation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to: {results_file}")
        
        return evaluation_results
    
    def _generate_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of evaluation results"""
        summary = {
            'best_strategies': {},
            'recommendations': [],
            'key_insights': []
        }
        
        strategies = evaluation_results['strategies']
        
        # Find best strategies for each metric
        if strategies:
            # Basic evaluation
            best_coherence = max(
                evaluation_results['basic_evaluation'].items(),
                key=lambda x: x[1]['coherence_score']
            )[0] if evaluation_results['basic_evaluation'] else None
            
            best_retrieval = max(
                evaluation_results['basic_evaluation'].items(),
                key=lambda x: x[1]['retrieval_accuracy']
            )[0] if evaluation_results['basic_evaluation'] else None
            
            # RAG evaluation
            best_rag = max(
                evaluation_results['rag_evaluation'].items(),
                key=lambda x: x[1]['overall_rag_score']
            )[0] if evaluation_results['rag_evaluation'] else None
            
            # Performance
            best_performance = min(
                evaluation_results['performance_benchmark'].items(),
                key=lambda x: x[1]['processing_time']
            )[0] if evaluation_results['performance_benchmark'] else None
            
            # Coherence
            best_coherence_eval = max(
                evaluation_results['coherence_evaluation'].items(),
                key=lambda x: x[1]['overall_coherence']
            )[0] if evaluation_results['coherence_evaluation'] else None
            
            summary['best_strategies'] = {
                'best_coherence': best_coherence,
                'best_retrieval': best_retrieval,
                'best_rag': best_rag,
                'best_performance': best_performance,
                'best_coherence_eval': best_coherence_eval
            }
            
            # Generate recommendations
            if best_rag:
                summary['recommendations'].append(f"Use {best_rag} for RAG applications")
            if best_performance:
                summary['recommendations'].append(f"Use {best_performance} for high-performance requirements")
            if best_coherence_eval:
                summary['recommendations'].append(f"Use {best_coherence_eval} for content coherence")
            
            # Key insights
            summary['key_insights'].append(f"Evaluated {len(strategies)} chunking strategies")
            summary['key_insights'].append("Performance varies significantly between strategies")
            summary['key_insights'].append("RAG performance is critical for retrieval applications")
            summary['key_insights'].append("Coherence affects user experience and comprehension")
        
        return summary
    
    def _generate_reports(self, evaluation_results: Dict[str, Any], output_dir: Path):
        """Generate individual evaluation reports"""
        logger.info("Generating evaluation reports")
        
        # Generate basic evaluation report
        try:
            self.basic_evaluator.generate_report(
                output_path=str(output_dir / "basic_evaluation_report.html")
            )
        except Exception as e:
            logger.error(f"Error generating basic evaluation report: {e}")
        
        # Generate RAG evaluation report
        try:
            self.rag_evaluator.generate_rag_report(
                output_path=str(output_dir / "rag_evaluation_report.html")
            )
        except Exception as e:
            logger.error(f"Error generating RAG evaluation report: {e}")
        
        # Generate performance benchmark report
        try:
            self.performance_benchmark.generate_performance_report(
                output_path=str(output_dir / "performance_benchmark_report.html")
            )
        except Exception as e:
            logger.error(f"Error generating performance benchmark report: {e}")
        
        # Generate coherence evaluation report
        try:
            self.coherence_evaluator.generate_coherence_report(
                output_path=str(output_dir / "coherence_evaluation_report.html")
            )
        except Exception as e:
            logger.error(f"Error generating coherence evaluation report: {e}")
    
    def load_chunking_results(self, results_dir: str) -> Dict[str, List[Dict]]:
        """
        Load chunking results from directory
        
        Args:
            results_dir: Directory containing chunking result files
            
        Returns:
            Dictionary mapping strategy names to chunk lists
        """
        strategies = {}
        results_path = Path(results_dir)
        
        if not results_path.exists():
            logger.warning(f"Results directory not found: {results_dir}")
            return strategies
        
        # Look for JSON files with chunking results
        for json_file in results_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Determine strategy name from filename or data
                strategy_name = json_file.stem
                if isinstance(data, list) and data:
                    # Check if data contains strategy information
                    if 'strategy' in data[0]:
                        strategy_name = data[0]['strategy']
                
                strategies[strategy_name] = data
                logger.info(f"Loaded {len(data)} chunks for strategy: {strategy_name}")
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        return strategies

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Comprehensive chunking strategy evaluation")
    parser.add_argument("--results-dir", "-r", required=True, help="Directory containing chunking results")
    parser.add_argument("--output-dir", "-o", default="evaluation_results", help="Output directory for evaluation results")
    parser.add_argument("--config", "-c", help="Path to configuration JSON file")
    parser.add_argument("--generate-synthetic", action='store_true', help="Generate synthetic test data")
    parser.add_argument("--num-docs", type=int, default=5, help="Number of synthetic documents to generate")
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return
    
    # Update config with command line arguments
    if config is None:
        config = {}
    
    config['output_dir'] = args.output_dir
    config['generate_synthetic_data'] = args.generate_synthetic
    config['num_synthetic_docs'] = args.num_docs
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(config)
    
    # Load chunking results
    strategies = evaluator.load_chunking_results(args.results_dir)
    
    if not strategies:
        logger.error("No chunking strategies found. Please ensure results directory contains valid JSON files.")
        return
    
    logger.info(f"Found {len(strategies)} chunking strategies: {list(strategies.keys())}")
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(strategies)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*60)
    
    summary = results['summary']
    print(f"\nBest Strategies:")
    for metric, strategy in summary['best_strategies'].items():
        print(f"  {metric}: {strategy}")
    
    print(f"\nRecommendations:")
    for rec in summary['recommendations']:
        print(f"  - {rec}")
    
    print(f"\nKey Insights:")
    for insight in summary['key_insights']:
        print(f"  - {insight}")
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 