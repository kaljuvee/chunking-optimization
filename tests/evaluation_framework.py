#!/usr/bin/env python3
"""
ADNOC Chunking Evaluation Framework
===================================

This module provides a comprehensive evaluation framework for testing and comparing
different chunking strategies. It includes metrics for coherence, retrieval accuracy,
performance, and RAG-specific evaluation.

Author: Data Engineering Team
Purpose: Systematic evaluation of chunking strategies
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkingMetrics:
    """Data class for storing chunking evaluation metrics"""
    strategy_name: str
    num_chunks: int
    avg_chunk_size: float
    chunk_size_std: float
    coherence_score: float
    retrieval_accuracy: float
    processing_time: float
    memory_usage: float
    rag_performance: float
    user_satisfaction: float

@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    max_chunks: int = 100
    batch_size: int = 10
    cache_dir: str = "./cache"
    enable_caching: bool = True
    evaluation_questions: List[str] = None
    
    def __post_init__(self):
        if self.evaluation_questions is None:
            self.evaluation_questions = [
                "What are the key performance indicators?",
                "What are the main challenges?",
                "What are the cost savings?",
                "What are the safety requirements?",
                "What are the environmental standards?"
            ]

class ChunkingEvaluator:
    """
    Comprehensive evaluator for chunking strategies
    """
    
    def __init__(self, config: EvaluationConfig = None):
        """Initialize the evaluator with configuration"""
        self.config = config or EvaluationConfig()
        self.results = {}
        self.cache = {}
        
        # Create cache directory
        Path(self.config.cache_dir).mkdir(exist_ok=True)
        
    def evaluate_chunking_strategy(self, 
                                 strategy_name: str, 
                                 chunks: List[Dict], 
                                 original_text: str = None) -> ChunkingMetrics:
        """
        Evaluate a single chunking strategy
        
        Args:
            strategy_name: Name of the chunking strategy
            chunks: List of chunk dictionaries
            original_text: Original text (optional, for coherence evaluation)
            
        Returns:
            ChunkingMetrics object with evaluation results
        """
        logger.info(f"Evaluating chunking strategy: {strategy_name}")
        
        start_time = time.time()
        
        # Basic metrics
        num_chunks = len(chunks)
        chunk_sizes = [len(chunk.get('text', '').split()) for chunk in chunks]
        avg_chunk_size = np.mean(chunk_sizes)
        chunk_size_std = np.std(chunk_sizes)
        
        # Coherence evaluation
        coherence_score = self._evaluate_coherence(chunks, original_text)
        
        # Retrieval accuracy
        retrieval_accuracy = self._evaluate_retrieval_accuracy(chunks)
        
        # RAG performance
        rag_performance = self._evaluate_rag_performance(chunks)
        
        # Processing time
        processing_time = time.time() - start_time
        
        # Memory usage (approximate)
        memory_usage = self._estimate_memory_usage(chunks)
        
        # User satisfaction (simulated)
        user_satisfaction = self._simulate_user_satisfaction(chunks)
        
        metrics = ChunkingMetrics(
            strategy_name=strategy_name,
            num_chunks=num_chunks,
            avg_chunk_size=avg_chunk_size,
            chunk_size_std=chunk_size_std,
            coherence_score=coherence_score,
            retrieval_accuracy=retrieval_accuracy,
            processing_time=processing_time,
            memory_usage=memory_usage,
            rag_performance=rag_performance,
            user_satisfaction=user_satisfaction
        )
        
        self.results[strategy_name] = metrics
        return metrics
    
    def _evaluate_coherence(self, chunks: List[Dict], original_text: str = None) -> float:
        """
        Evaluate semantic coherence of chunks
        
        Args:
            chunks: List of chunk dictionaries
            original_text: Original text for comparison
            
        Returns:
            Coherence score between 0 and 1
        """
        if not chunks:
            return 0.0
        
        # Extract chunk texts
        chunk_texts = [chunk.get('text', '') for chunk in chunks]
        
        # Calculate semantic similarity between adjacent chunks
        similarities = []
        for i in range(len(chunk_texts) - 1):
            similarity = self._calculate_text_similarity(chunk_texts[i], chunk_texts[i + 1])
            similarities.append(similarity)
        
        # Calculate coherence score
        if similarities:
            coherence_score = np.mean(similarities)
        else:
            coherence_score = 1.0  # Single chunk is perfectly coherent
        
        return coherence_score
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using TF-IDF
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1.strip() or not text2.strip():
            return 0.0
        
        try:
            # Use TF-IDF for similarity calculation
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Combine texts for vectorization
            combined_texts = [text1, text2]
            tfidf_matrix = vectorizer.fit_transform(combined_texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
        except Exception as e:
            logger.warning(f"Error calculating text similarity: {e}")
            return 0.0
    
    def _evaluate_retrieval_accuracy(self, chunks: List[Dict]) -> float:
        """
        Evaluate retrieval accuracy using simulated queries
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Retrieval accuracy score between 0 and 1
        """
        if not chunks:
            return 0.0
        
        # Simulate retrieval queries
        query_results = []
        
        for question in self.config.evaluation_questions:
            # Find most relevant chunk for each question
            best_chunk = self._find_best_chunk(question, chunks)
            if best_chunk:
                # Simulate relevance score based on keyword overlap
                relevance = self._calculate_relevance(question, best_chunk.get('text', ''))
                query_results.append(relevance)
        
        # Calculate average retrieval accuracy
        if query_results:
            return np.mean(query_results)
        else:
            return 0.0
    
    def _find_best_chunk(self, query: str, chunks: List[Dict]) -> Optional[Dict]:
        """
        Find the best chunk for a given query
        
        Args:
            query: Search query
            chunks: List of chunk dictionaries
            
        Returns:
            Best matching chunk or None
        """
        best_chunk = None
        best_score = 0.0
        
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            score = self._calculate_relevance(query, chunk_text)
            
            if score > best_score:
                best_score = score
                best_chunk = chunk
        
        return best_chunk
    
    def _calculate_relevance(self, query: str, text: str) -> float:
        """
        Calculate relevance between query and text
        
        Args:
            query: Search query
            text: Text to evaluate
            
        Returns:
            Relevance score between 0 and 1
        """
        # Simple keyword-based relevance calculation
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _evaluate_rag_performance(self, chunks: List[Dict]) -> float:
        """
        Evaluate RAG-specific performance metrics
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            RAG performance score between 0 and 1
        """
        if not chunks:
            return 0.0
        
        # Calculate various RAG-specific metrics
        metrics = []
        
        # 1. Context preservation (chunk size consistency)
        chunk_sizes = [len(chunk.get('text', '').split()) for chunk in chunks]
        size_consistency = 1.0 - (np.std(chunk_sizes) / np.mean(chunk_sizes)) if np.mean(chunk_sizes) > 0 else 0.0
        metrics.append(max(0.0, size_consistency))
        
        # 2. Information density (non-empty chunks)
        non_empty_chunks = sum(1 for chunk in chunks if chunk.get('text', '').strip())
        density_score = non_empty_chunks / len(chunks) if chunks else 0.0
        metrics.append(density_score)
        
        # 3. Semantic diversity (avoid redundant chunks)
        diversity_score = self._calculate_semantic_diversity(chunks)
        metrics.append(diversity_score)
        
        # 4. Retrieval efficiency (chunk accessibility)
        efficiency_score = self._calculate_retrieval_efficiency(chunks)
        metrics.append(efficiency_score)
        
        # Return average RAG performance score
        return np.mean(metrics)
    
    def _calculate_semantic_diversity(self, chunks: List[Dict]) -> float:
        """
        Calculate semantic diversity among chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Diversity score between 0 and 1
        """
        if len(chunks) < 2:
            return 1.0  # Single chunk has maximum diversity
        
        # Calculate pairwise similarities
        similarities = []
        chunk_texts = [chunk.get('text', '') for chunk in chunks]
        
        for i in range(len(chunk_texts)):
            for j in range(i + 1, len(chunk_texts)):
                similarity = self._calculate_text_similarity(chunk_texts[i], chunk_texts[j])
                similarities.append(similarity)
        
        # Diversity is inverse of average similarity
        if similarities:
            avg_similarity = np.mean(similarities)
            diversity = 1.0 - avg_similarity
            return max(0.0, diversity)
        else:
            return 1.0
    
    def _calculate_retrieval_efficiency(self, chunks: List[Dict]) -> float:
        """
        Calculate retrieval efficiency based on chunk characteristics
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Efficiency score between 0 and 1
        """
        if not chunks:
            return 0.0
        
        # Factors affecting retrieval efficiency
        factors = []
        
        # 1. Optimal chunk size (not too small, not too large)
        chunk_sizes = [len(chunk.get('text', '').split()) for chunk in chunks]
        avg_size = np.mean(chunk_sizes)
        
        # Optimal size range: 100-500 words
        if 100 <= avg_size <= 500:
            size_score = 1.0
        else:
            size_score = max(0.0, 1.0 - abs(avg_size - 300) / 300)
        
        factors.append(size_score)
        
        # 2. Chunk completeness (complete sentences/paragraphs)
        completeness_scores = []
        for chunk in chunks:
            text = chunk.get('text', '')
            # Simple heuristic: check if chunk ends with sentence-ending punctuation
            if text.strip().endswith(('.', '!', '?')):
                completeness_scores.append(1.0)
            else:
                completeness_scores.append(0.5)
        
        factors.append(np.mean(completeness_scores))
        
        # 3. Information richness (non-empty, meaningful content)
        richness_scores = []
        for chunk in chunks:
            text = chunk.get('text', '')
            words = text.split()
            # Richness based on word count and variety
            if len(words) >= 50:
                richness_scores.append(1.0)
            elif len(words) >= 20:
                richness_scores.append(0.7)
            else:
                richness_scores.append(0.3)
        
        factors.append(np.mean(richness_scores))
        
        return np.mean(factors)
    
    def _estimate_memory_usage(self, chunks: List[Dict]) -> float:
        """
        Estimate memory usage for chunks
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Estimated memory usage in MB
        """
        total_size = 0
        for chunk in chunks:
            # Estimate size of chunk data
            chunk_text = chunk.get('text', '')
            chunk_size = len(chunk_text.encode('utf-8'))
            metadata_size = len(json.dumps(chunk).encode('utf-8'))
            total_size += chunk_size + metadata_size
        
        # Convert to MB
        return total_size / (1024 * 1024)
    
    def _simulate_user_satisfaction(self, chunks: List[Dict]) -> float:
        """
        Simulate user satisfaction based on chunk quality
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Simulated satisfaction score between 0 and 1
        """
        if not chunks:
            return 0.0
        
        # Factors affecting user satisfaction
        factors = []
        
        # 1. Chunk readability (appropriate length)
        chunk_sizes = [len(chunk.get('text', '').split()) for chunk in chunks]
        avg_size = np.mean(chunk_sizes)
        
        if 50 <= avg_size <= 300:
            readability_score = 1.0
        else:
            readability_score = max(0.0, 1.0 - abs(avg_size - 175) / 175)
        
        factors.append(readability_score)
        
        # 2. Content completeness
        completeness_scores = []
        for chunk in chunks:
            text = chunk.get('text', '')
            # Check for complete thoughts/sections
            if len(text.split()) >= 20 and text.strip().endswith(('.', '!', '?')):
                completeness_scores.append(1.0)
            elif len(text.split()) >= 10:
                completeness_scores.append(0.7)
            else:
                completeness_scores.append(0.3)
        
        factors.append(np.mean(completeness_scores))
        
        # 3. Information relevance (non-empty, meaningful content)
        relevance_scores = []
        for chunk in chunks:
            text = chunk.get('text', '')
            words = text.split()
            if len(words) >= 15:
                relevance_scores.append(1.0)
            elif len(words) >= 8:
                relevance_scores.append(0.6)
            else:
                relevance_scores.append(0.2)
        
        factors.append(np.mean(relevance_scores))
        
        return np.mean(factors)
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare all evaluated chunking strategies
        
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        data = []
        for strategy_name, metrics in self.results.items():
            data.append({
                'Strategy': strategy_name,
                'Num_Chunks': metrics.num_chunks,
                'Avg_Chunk_Size': metrics.avg_chunk_size,
                'Chunk_Size_Std': metrics.chunk_size_std,
                'Coherence_Score': metrics.coherence_score,
                'Retrieval_Accuracy': metrics.retrieval_accuracy,
                'Processing_Time': metrics.processing_time,
                'Memory_Usage_MB': metrics.memory_usage,
                'RAG_Performance': metrics.rag_performance,
                'User_Satisfaction': metrics.user_satisfaction
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_report(self, output_path: str = "evaluation_report.html"):
        """
        Generate comprehensive evaluation report
        
        Args:
            output_path: Path for the output HTML report
        """
        if not self.results:
            logger.warning("No evaluation results available")
            return
        
        # Create comparison DataFrame
        df = self.compare_strategies()
        
        # Generate HTML report
        html_content = self._generate_html_report(df)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to: {output_path}")
    
    def _generate_html_report(self, df: pd.DataFrame) -> str:
        """
        Generate HTML report from evaluation results
        
        Args:
            df: DataFrame with evaluation results
            
        Returns:
            HTML content string
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ADNOC Chunking Strategy Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; color: #333; }
                .score { color: #0066cc; }
                .highlight { background-color: #e6f3ff; }
            </style>
        </head>
        <body>
            <h1>ADNOC Chunking Strategy Evaluation Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <h2>Evaluation Summary</h2>
            <p>This report compares {num_strategies} chunking strategies across multiple metrics.</p>
            
            <h2>Detailed Results</h2>
            {table_html}
            
            <h2>Key Findings</h2>
            <ul>
                <li><strong>Best Coherence:</strong> {best_coherence}</li>
                <li><strong>Best Retrieval Accuracy:</strong> {best_retrieval}</li>
                <li><strong>Best RAG Performance:</strong> {best_rag}</li>
                <li><strong>Fastest Processing:</strong> {fastest_processing}</li>
                <li><strong>Most Memory Efficient:</strong> {most_efficient}</li>
            </ul>
            
            <h2>Recommendations</h2>
            <p>Based on the evaluation results, consider the following recommendations:</p>
            <ul>
                <li>Choose strategies with high coherence scores for better context preservation</li>
                <li>Prioritize retrieval accuracy for RAG applications</li>
                <li>Balance processing speed with quality for production systems</li>
                <li>Consider memory usage for large-scale deployments</li>
            </ul>
        </body>
        </html>
        """
        
        # Generate table HTML
        table_html = df.to_html(index=False, classes='dataframe', float_format='%.3f')
        
        # Find best performers
        best_coherence = df.loc[df['Coherence_Score'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_retrieval = df.loc[df['Retrieval_Accuracy'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_rag = df.loc[df['RAG_Performance'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        fastest_processing = df.loc[df['Processing_Time'].idxmin(), 'Strategy'] if not df.empty else 'N/A'
        most_efficient = df.loc[df['Memory_Usage_MB'].idxmin(), 'Strategy'] if not df.empty else 'N/A'
        
        # Format the HTML
        html_content = html_template.format(
            timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            num_strategies=len(df),
            table_html=table_html,
            best_coherence=best_coherence,
            best_retrieval=best_retrieval,
            best_rag=best_rag,
            fastest_processing=fastest_processing,
            most_efficient=most_efficient
        )
        
        return html_content

def main():
    """Example usage of the evaluation framework"""
    # Initialize evaluator
    config = EvaluationConfig(
        max_chunks=50,
        batch_size=5,
        evaluation_questions=[
            "What are the key performance indicators?",
            "What are the main challenges?",
            "What are the cost savings?",
            "What are the safety requirements?",
            "What are the environmental standards?"
        ]
    )
    
    evaluator = ChunkingEvaluator(config)
    
    # Example: Evaluate a chunking strategy
    # This would typically be called with actual chunking results
    logger.info("Evaluation framework initialized successfully")
    logger.info("Use evaluate_chunking_strategy() to evaluate specific strategies")
    logger.info("Use compare_strategies() to compare multiple strategies")
    logger.info("Use generate_report() to create HTML reports")

if __name__ == "__main__":
    main() 