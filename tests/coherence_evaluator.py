#!/usr/bin/env python3
"""
ADNOC Coherence Evaluator
=========================

This module provides semantic coherence evaluation for chunking strategies.
It measures how well chunks maintain semantic coherence and context.

Author: Data Engineering Team
Purpose: Semantic coherence evaluation
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CoherenceMetrics:
    """Data class for coherence evaluation metrics"""
    strategy_name: str
    semantic_coherence: float
    topic_coherence: float
    discourse_coherence: float
    context_preservation: float
    boundary_quality: float
    overall_coherence: float

@dataclass
class CoherenceConfig:
    """Configuration for coherence evaluation"""
    min_chunk_size: int = 20
    max_chunk_size: int = 1000
    similarity_threshold: float = 0.3
    topic_model_components: int = 5
    cluster_components: int = 3
    evaluation_methods: List[str] = None
    
    def __post_init__(self):
        if self.evaluation_methods is None:
            self.evaluation_methods = ['semantic', 'topic', 'discourse', 'context', 'boundary']

class CoherenceEvaluator:
    """
    Evaluator for semantic coherence of chunking strategies
    """
    
    def __init__(self, config: CoherenceConfig = None):
        """Initialize the coherence evaluator"""
        self.config = config or CoherenceConfig()
        self.results = {}
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def evaluate_coherence(self, 
                          strategy_name: str, 
                          chunks: List[Dict],
                          original_text: str = None) -> CoherenceMetrics:
        """
        Evaluate coherence of a chunking strategy
        
        Args:
            strategy_name: Name of the chunking strategy
            chunks: List of chunk dictionaries
            original_text: Original text (optional)
            
        Returns:
            CoherenceMetrics object
        """
        logger.info(f"Evaluating coherence for: {strategy_name}")
        
        if not chunks:
            return CoherenceMetrics(
                strategy_name=strategy_name,
                semantic_coherence=0.0,
                topic_coherence=0.0,
                discourse_coherence=0.0,
                context_preservation=0.0,
                boundary_quality=0.0,
                overall_coherence=0.0
            )
        
        # Extract chunk texts
        chunk_texts = [chunk.get('text', '') for chunk in chunks]
        
        # Filter out empty chunks
        valid_chunks = [text for text in chunk_texts if text.strip()]
        if not valid_chunks:
            logger.warning(f"No valid chunks found for {strategy_name}")
            return CoherenceMetrics(
                strategy_name=strategy_name,
                semantic_coherence=0.0,
                topic_coherence=0.0,
                discourse_coherence=0.0,
                context_preservation=0.0,
                boundary_quality=0.0,
                overall_coherence=0.0
            )
        
        # Calculate coherence metrics
        semantic_coherence = self._evaluate_semantic_coherence(valid_chunks)
        topic_coherence = self._evaluate_topic_coherence(valid_chunks)
        discourse_coherence = self._evaluate_discourse_coherence(valid_chunks)
        context_preservation = self._evaluate_context_preservation(valid_chunks, original_text)
        boundary_quality = self._evaluate_boundary_quality(valid_chunks)
        
        # Calculate overall coherence score
        overall_coherence = self._calculate_overall_coherence(
            semantic_coherence, topic_coherence, discourse_coherence,
            context_preservation, boundary_quality
        )
        
        metrics = CoherenceMetrics(
            strategy_name=strategy_name,
            semantic_coherence=semantic_coherence,
            topic_coherence=topic_coherence,
            discourse_coherence=discourse_coherence,
            context_preservation=context_preservation,
            boundary_quality=boundary_quality,
            overall_coherence=overall_coherence
        )
        
        self.results[strategy_name] = metrics
        return metrics
    
    def _evaluate_semantic_coherence(self, chunk_texts: List[str]) -> float:
        """
        Evaluate semantic coherence between adjacent chunks
        
        Args:
            chunk_texts: List of chunk texts
            
        Returns:
            Semantic coherence score
        """
        if len(chunk_texts) < 2:
            return 1.0  # Single chunk is perfectly coherent
        
        # Vectorize chunks
        try:
            tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
            similarities = cosine_similarity(tfidf_matrix)
            
            # Calculate average similarity between adjacent chunks
            adjacent_similarities = []
            for i in range(len(similarities) - 1):
                similarity = similarities[i][i + 1]
                adjacent_similarities.append(similarity)
            
            # Calculate semantic coherence score
            avg_similarity = np.mean(adjacent_similarities)
            
            # Normalize to 0-1 scale
            coherence_score = min(1.0, avg_similarity / self.config.similarity_threshold)
            
            return coherence_score
            
        except Exception as e:
            logger.error(f"Error calculating semantic coherence: {e}")
            return 0.0
    
    def _evaluate_topic_coherence(self, chunk_texts: List[str]) -> float:
        """
        Evaluate topic coherence within chunks
        
        Args:
            chunk_texts: List of chunk texts
            
        Returns:
            Topic coherence score
        """
        if not chunk_texts:
            return 0.0
        
        try:
            # Vectorize chunks
            tfidf_matrix = self.vectorizer.fit_transform(chunk_texts)
            
            # Apply topic modeling
            n_topics = min(self.config.topic_model_components, len(chunk_texts))
            if n_topics < 2:
                return 1.0  # Single topic is perfectly coherent
            
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            
            topic_distributions = lda.fit_transform(tfidf_matrix)
            
            # Calculate topic coherence
            topic_coherences = []
            for i in range(len(chunk_texts)):
                # Get dominant topic for this chunk
                dominant_topic = np.argmax(topic_distributions[i])
                
                # Find chunks with similar topic distributions
                topic_similarities = []
                for j in range(len(chunk_texts)):
                    if i != j:
                        similarity = cosine_similarity(
                            topic_distributions[i:i+1], 
                            topic_distributions[j:j+1]
                        )[0][0]
                        topic_similarities.append(similarity)
                
                if topic_similarities:
                    chunk_coherence = np.mean(topic_similarities)
                    topic_coherences.append(chunk_coherence)
            
            # Calculate overall topic coherence
            if topic_coherences:
                return np.mean(topic_coherences)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating topic coherence: {e}")
            return 0.0
    
    def _evaluate_discourse_coherence(self, chunk_texts: List[str]) -> float:
        """
        Evaluate discourse coherence (logical flow between chunks)
        
        Args:
            chunk_texts: List of chunk texts
            
        Returns:
            Discourse coherence score
        """
        if len(chunk_texts) < 2:
            return 1.0
        
        discourse_scores = []
        
        for i in range(len(chunk_texts) - 1):
            current_chunk = chunk_texts[i]
            next_chunk = chunk_texts[i + 1]
            
            # Check for discourse markers and logical connections
            discourse_score = self._calculate_discourse_score(current_chunk, next_chunk)
            discourse_scores.append(discourse_score)
        
        return np.mean(discourse_scores) if discourse_scores else 0.0
    
    def _calculate_discourse_score(self, chunk1: str, chunk2: str) -> float:
        """
        Calculate discourse score between two chunks
        
        Args:
            chunk1: First chunk text
            chunk2: Second chunk text
            
        Returns:
            Discourse score
        """
        # Discourse markers that indicate logical flow
        discourse_markers = [
            'however', 'therefore', 'consequently', 'furthermore', 'moreover',
            'additionally', 'in addition', 'also', 'besides', 'meanwhile',
            'subsequently', 'then', 'next', 'finally', 'in conclusion',
            'first', 'second', 'third', 'lastly', 'initially'
        ]
        
        # Check for discourse markers in the transition
        transition_text = chunk1[-100:] + " " + chunk2[:100]  # Overlap area
        transition_lower = transition_text.lower()
        
        marker_count = sum(1 for marker in discourse_markers if marker in transition_lower)
        
        # Normalize marker count
        marker_score = min(1.0, marker_count / 3)  # Max 3 markers for perfect score
        
        # Check for semantic similarity as additional factor
        try:
            tfidf_matrix = self.vectorizer.fit_transform([chunk1, chunk2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            similarity_score = min(1.0, similarity / 0.5)  # Normalize to 0-1
        except:
            similarity_score = 0.0
        
        # Combine scores
        discourse_score = 0.6 * marker_score + 0.4 * similarity_score
        
        return discourse_score
    
    def _evaluate_context_preservation(self, chunk_texts: List[str], original_text: str = None) -> float:
        """
        Evaluate how well context is preserved across chunks
        
        Args:
            chunk_texts: List of chunk texts
            original_text: Original text (optional)
            
        Returns:
            Context preservation score
        """
        if not chunk_texts:
            return 0.0
        
        # Calculate context preservation based on chunk characteristics
        context_scores = []
        
        for chunk_text in chunk_texts:
            # Check for complete sentences
            sentences = re.split(r'[.!?]+', chunk_text)
            complete_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            sentence_completeness = len(complete_sentences) / max(1, len(sentences))
            
            # Check for contextual completeness (has subject, verb, object)
            contextual_completeness = self._check_contextual_completeness(chunk_text)
            
            # Check for information density
            words = chunk_text.split()
            information_density = min(1.0, len(words) / 50)  # Normalize to 50 words
            
            # Combine scores
            chunk_context_score = (
                0.4 * sentence_completeness +
                0.4 * contextual_completeness +
                0.2 * information_density
            )
            
            context_scores.append(chunk_context_score)
        
        return np.mean(context_scores) if context_scores else 0.0
    
    def _check_contextual_completeness(self, text: str) -> float:
        """
        Check if text has contextual completeness
        
        Args:
            text: Text to check
            
        Returns:
            Completeness score
        """
        # Simple heuristics for contextual completeness
        words = text.split()
        
        if len(words) < 5:
            return 0.0
        
        # Check for basic sentence structure indicators
        has_capital_letter = any(word[0].isupper() for word in words if word)
        has_ending_punctuation = text.strip().endswith(('.', '!', '?'))
        has_verb_indicators = any(word.lower() in ['is', 'are', 'was', 'were', 'has', 'have', 'had'] for word in words)
        
        # Calculate completeness score
        completeness_factors = [
            1.0 if has_capital_letter else 0.0,
            1.0 if has_ending_punctuation else 0.0,
            1.0 if has_verb_indicators else 0.0
        ]
        
        return np.mean(completeness_factors)
    
    def _evaluate_boundary_quality(self, chunk_texts: List[str]) -> float:
        """
        Evaluate the quality of chunk boundaries
        
        Args:
            chunk_texts: List of chunk texts
            
        Returns:
            Boundary quality score
        """
        if not chunk_texts:
            return 0.0
        
        boundary_scores = []
        
        for chunk_text in chunk_texts:
            # Check if chunk ends with complete sentence
            ends_with_sentence = chunk_text.strip().endswith(('.', '!', '?'))
            
            # Check if chunk starts with proper capitalization
            starts_properly = chunk_text.strip()[0].isupper() if chunk_text.strip() else False
            
            # Check for natural breakpoints
            has_natural_break = self._check_natural_breakpoint(chunk_text)
            
            # Calculate boundary score
            boundary_score = (
                0.4 * (1.0 if ends_with_sentence else 0.0) +
                0.3 * (1.0 if starts_properly else 0.0) +
                0.3 * (1.0 if has_natural_break else 0.0)
            )
            
            boundary_scores.append(boundary_score)
        
        return np.mean(boundary_scores) if boundary_scores else 0.0
    
    def _check_natural_breakpoint(self, text: str) -> bool:
        """
        Check if text has natural breakpoints
        
        Args:
            text: Text to check
            
        Returns:
            True if natural breakpoint exists
        """
        # Check for paragraph breaks
        if '\n\n' in text:
            return True
        
        # Check for section markers
        section_markers = ['##', '###', '####', 'Section', 'Chapter', 'Part']
        if any(marker in text for marker in section_markers):
            return True
        
        # Check for logical connectors at boundaries
        logical_connectors = ['however', 'therefore', 'consequently', 'furthermore', 'moreover']
        text_lower = text.lower()
        if any(connector in text_lower for connector in logical_connectors):
            return True
        
        return False
    
    def _calculate_overall_coherence(self, 
                                   semantic_coherence: float,
                                   topic_coherence: float,
                                   discourse_coherence: float,
                                   context_preservation: float,
                                   boundary_quality: float) -> float:
        """
        Calculate overall coherence score
        
        Args:
            semantic_coherence: Semantic coherence score
            topic_coherence: Topic coherence score
            discourse_coherence: Discourse coherence score
            context_preservation: Context preservation score
            boundary_quality: Boundary quality score
            
        Returns:
            Overall coherence score
        """
        # Weighted combination of coherence factors
        weights = {
            'semantic': 0.25,
            'topic': 0.20,
            'discourse': 0.20,
            'context': 0.20,
            'boundary': 0.15
        }
        
        overall_score = (
            weights['semantic'] * semantic_coherence +
            weights['topic'] * topic_coherence +
            weights['discourse'] * discourse_coherence +
            weights['context'] * context_preservation +
            weights['boundary'] * boundary_quality
        )
        
        return overall_score
    
    def compare_coherence(self) -> pd.DataFrame:
        """
        Compare coherence across strategies
        
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            logger.warning("No coherence evaluation results available")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        data = []
        for strategy_name, metrics in self.results.items():
            data.append({
                'Strategy': strategy_name,
                'Semantic_Coherence': metrics.semantic_coherence,
                'Topic_Coherence': metrics.topic_coherence,
                'Discourse_Coherence': metrics.discourse_coherence,
                'Context_Preservation': metrics.context_preservation,
                'Boundary_Quality': metrics.boundary_quality,
                'Overall_Coherence': metrics.overall_coherence
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_coherence_report(self, output_path: str = "coherence_evaluation_report.html"):
        """
        Generate coherence evaluation report
        
        Args:
            output_path: Path for the output HTML report
        """
        if not self.results:
            logger.warning("No coherence evaluation results available")
            return
        
        # Create comparison DataFrame
        df = self.compare_coherence()
        
        # Generate HTML report
        html_content = self._generate_coherence_html_report(df)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Coherence evaluation report saved to: {output_path}")
    
    def _generate_coherence_html_report(self, df: pd.DataFrame) -> str:
        """
        Generate HTML report for coherence evaluation
        
        Args:
            df: DataFrame with coherence results
            
        Returns:
            HTML content string
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ADNOC Coherence Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; color: #333; }
                .score { color: #0066cc; }
                .highlight { background-color: #e6f3ff; }
                .coherence-score { background-color: #d4edda; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>ADNOC Coherence Evaluation Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <h2>Coherence Summary</h2>
            <p>This report compares {num_strategies} chunking strategies for semantic coherence.</p>
            
            <h2>Detailed Coherence Metrics</h2>
            {table_html}
            
            <h2>Coherence Rankings</h2>
            <ul>
                <li><strong>Best Semantic Coherence:</strong> {best_semantic}</li>
                <li><strong>Best Topic Coherence:</strong> {best_topic}</li>
                <li><strong>Best Discourse Coherence:</strong> {best_discourse}</li>
                <li><strong>Best Context Preservation:</strong> {best_context}</li>
                <li><strong>Best Boundary Quality:</strong> {best_boundary}</li>
                <li><strong>Best Overall Coherence:</strong> {best_overall}</li>
            </ul>
            
            <h2>Coherence Recommendations</h2>
            <p>Based on the coherence evaluation results, consider the following recommendations:</p>
            <ul>
                <li>Choose strategies with high semantic coherence for better context preservation</li>
                <li>Prioritize topic coherence for maintaining thematic consistency</li>
                <li>Consider discourse coherence for logical flow between chunks</li>
                <li>Evaluate context preservation for complete information retention</li>
                <li>Assess boundary quality for natural chunk breaks</li>
                <li>Use overall coherence score as the primary selection criterion</li>
            </ul>
            
            <h2>Coherence Insights</h2>
            <p>The evaluation reveals important coherence characteristics:</p>
            <ul>
                <li>Semantic coherence ensures related information stays together</li>
                <li>Topic coherence maintains thematic consistency across chunks</li>
                <li>Discourse coherence preserves logical flow and transitions</li>
                <li>Context preservation ensures complete information retention</li>
                <li>Boundary quality affects readability and comprehension</li>
            </ul>
        </body>
        </html>
        """
        
        # Generate table HTML
        table_html = df.to_html(index=False, classes='dataframe', float_format='%.3f')
        
        # Find best performers
        best_semantic = df.loc[df['Semantic_Coherence'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_topic = df.loc[df['Topic_Coherence'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_discourse = df.loc[df['Discourse_Coherence'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_context = df.loc[df['Context_Preservation'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_boundary = df.loc[df['Boundary_Quality'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_overall = df.loc[df['Overall_Coherence'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        
        # Format the HTML
        html_content = html_template.format(
            timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            num_strategies=len(df),
            table_html=table_html,
            best_semantic=best_semantic,
            best_topic=best_topic,
            best_discourse=best_discourse,
            best_context=best_context,
            best_boundary=best_boundary,
            best_overall=best_overall
        )
        
        return html_content

def main():
    """Example usage of the coherence evaluator"""
    # Initialize evaluator
    config = CoherenceConfig(
        min_chunk_size=20,
        max_chunk_size=1000,
        similarity_threshold=0.3,
        topic_model_components=5,
        cluster_components=3
    )
    
    evaluator = CoherenceEvaluator(config)
    
    # Example: Evaluate coherence
    # This would typically be called with actual chunking results
    logger.info("Coherence evaluator initialized successfully")
    logger.info("Use evaluate_coherence() to evaluate specific strategies")
    logger.info("Use compare_coherence() to compare multiple strategies")
    logger.info("Use generate_coherence_report() to create HTML reports")

if __name__ == "__main__":
    main() 