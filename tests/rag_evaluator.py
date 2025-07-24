#!/usr/bin/env python3
"""
ADNOC RAG Evaluator
===================

This module provides RAG-specific evaluation metrics for chunking strategies.
It evaluates how well chunks perform in retrieval-augmented generation scenarios.

Author: Data Engineering Team
Purpose: RAG-specific chunking evaluation
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
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
class RAGMetrics:
    """Data class for RAG-specific evaluation metrics"""
    strategy_name: str
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    answer_accuracy: float
    context_relevance: float
    response_coherence: float
    response_completeness: float
    overall_rag_score: float

@dataclass
class RAGEvaluationConfig:
    """Configuration for RAG evaluation parameters"""
    test_questions: List[str] = None
    ground_truth_answers: Dict[str, str] = None
    max_retrieved_chunks: int = 5
    similarity_threshold: float = 0.7
    evaluation_mode: str = 'automatic'  # 'automatic' or 'manual'
    
    def __post_init__(self):
        if self.test_questions is None:
            self.test_questions = [
                "What are the key performance indicators for oil production?",
                "What safety measures are required for offshore operations?",
                "What are the environmental compliance requirements?",
                "What is the cost-benefit analysis of digital transformation?",
                "What are the main challenges in reservoir management?",
                "What technologies are used for enhanced oil recovery?",
                "What are the regulatory standards for emissions?",
                "What is the ROI of AI implementation in oil and gas?",
                "What are the sustainability goals and targets?",
                "What are the best practices for predictive maintenance?"
            ]
        
        if self.ground_truth_answers is None:
            self.ground_truth_answers = {
                "What are the key performance indicators for oil production?": "Oil production rate, water cut, gas-oil ratio, recovery factor, and operational efficiency are key performance indicators for oil production.",
                "What safety measures are required for offshore operations?": "Safety measures include personal protective equipment, emergency response procedures, regular safety training, incident reporting systems, and compliance with safety regulations.",
                "What are the environmental compliance requirements?": "Environmental compliance requirements include emission standards, waste management protocols, water quality standards, and regular environmental monitoring and reporting.",
                "What is the cost-benefit analysis of digital transformation?": "Digital transformation typically shows positive ROI with cost savings of 20-30%, efficiency improvements of 25-40%, and payback periods of 1-3 years.",
                "What are the main challenges in reservoir management?": "Main challenges include reservoir heterogeneity, declining production rates, water management, cost optimization, and maintaining production efficiency."
            }

class RAGEvaluator:
    """
    RAG-specific evaluator for chunking strategies
    """
    
    def __init__(self, config: RAGEvaluationConfig = None):
        """Initialize the RAG evaluator"""
        self.config = config or RAGEvaluationConfig()
        self.results = {}
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def evaluate_rag_performance(self, 
                               strategy_name: str, 
                               chunks: List[Dict],
                               test_questions: Optional[List[str]] = None) -> RAGMetrics:
        """
        Evaluate RAG performance for a chunking strategy
        
        Args:
            strategy_name: Name of the chunking strategy
            chunks: List of chunk dictionaries
            test_questions: Optional list of test questions
            
        Returns:
            RAGMetrics object with evaluation results
        """
        logger.info(f"Evaluating RAG performance for: {strategy_name}")
        
        if test_questions is None:
            test_questions = self.config.test_questions
        
        # Extract chunk texts
        chunk_texts = [chunk.get('text', '') for chunk in chunks]
        
        # Calculate retrieval metrics
        retrieval_precision, retrieval_recall, retrieval_f1 = self._evaluate_retrieval(
            chunk_texts, test_questions
        )
        
        # Calculate answer accuracy
        answer_accuracy = self._evaluate_answer_accuracy(
            chunk_texts, test_questions
        )
        
        # Calculate context relevance
        context_relevance = self._evaluate_context_relevance(
            chunk_texts, test_questions
        )
        
        # Calculate response coherence
        response_coherence = self._evaluate_response_coherence(
            chunk_texts, test_questions
        )
        
        # Calculate response completeness
        response_completeness = self._evaluate_response_completeness(
            chunk_texts, test_questions
        )
        
        # Calculate overall RAG score
        overall_rag_score = self._calculate_overall_rag_score(
            retrieval_precision, retrieval_recall, retrieval_f1,
            answer_accuracy, context_relevance, response_coherence, response_completeness
        )
        
        metrics = RAGMetrics(
            strategy_name=strategy_name,
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            retrieval_f1=retrieval_f1,
            answer_accuracy=answer_accuracy,
            context_relevance=context_relevance,
            response_coherence=response_coherence,
            response_completeness=response_completeness,
            overall_rag_score=overall_rag_score
        )
        
        self.results[strategy_name] = metrics
        return metrics
    
    def _evaluate_retrieval(self, chunk_texts: List[str], questions: List[str]) -> Tuple[float, float, float]:
        """
        Evaluate retrieval performance
        
        Args:
            chunk_texts: List of chunk texts
            questions: List of test questions
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if not chunk_texts or not questions:
            return 0.0, 0.0, 0.0
        
        # Vectorize chunks and questions
        all_texts = chunk_texts + questions
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        chunk_vectors = tfidf_matrix[:len(chunk_texts)]
        question_vectors = tfidf_matrix[len(chunk_texts):]
        
        # Calculate similarities
        similarities = cosine_similarity(question_vectors, chunk_vectors)
        
        # Evaluate retrieval for each question
        precisions = []
        recalls = []
        f1_scores = []
        
        for i, question in enumerate(questions):
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities[i])[::-1][:self.config.max_retrieved_chunks]
            top_similarities = similarities[i][top_indices]
            
            # Determine relevant chunks (above threshold)
            relevant_chunks = top_similarities >= self.config.similarity_threshold
            
            # Calculate metrics
            precision = np.mean(relevant_chunks) if len(relevant_chunks) > 0 else 0.0
            recall = np.sum(relevant_chunks) / len(chunk_texts) if len(chunk_texts) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)
    
    def _evaluate_answer_accuracy(self, chunk_texts: List[str], questions: List[str]) -> float:
        """
        Evaluate answer accuracy based on retrieved chunks
        
        Args:
            chunk_texts: List of chunk texts
            questions: List of test questions
            
        Returns:
            Answer accuracy score
        """
        if not chunk_texts or not questions:
            return 0.0
        
        accuracies = []
        
        for question in questions:
            # Find most relevant chunks for the question
            relevant_chunks = self._find_relevant_chunks(question, chunk_texts)
            
            if not relevant_chunks:
                accuracies.append(0.0)
                continue
            
            # Simulate answer generation from chunks
            simulated_answer = self._simulate_answer_generation(question, relevant_chunks)
            
            # Calculate answer accuracy (simplified)
            accuracy = self._calculate_answer_similarity(question, simulated_answer)
            accuracies.append(accuracy)
        
        return np.mean(accuracies)
    
    def _find_relevant_chunks(self, question: str, chunk_texts: List[str]) -> List[str]:
        """
        Find relevant chunks for a question
        
        Args:
            question: Test question
            chunk_texts: List of chunk texts
            
        Returns:
            List of relevant chunk texts
        """
        # Vectorize question and chunks
        all_texts = [question] + chunk_texts
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        question_vector = tfidf_matrix[0:1]
        chunk_vectors = tfidf_matrix[1:]
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, chunk_vectors)[0]
        
        # Return chunks above threshold
        relevant_indices = np.where(similarities >= self.config.similarity_threshold)[0]
        relevant_chunks = [chunk_texts[i] for i in relevant_indices]
        
        return relevant_chunks[:self.config.max_retrieved_chunks]
    
    def _simulate_answer_generation(self, question: str, relevant_chunks: List[str]) -> str:
        """
        Simulate answer generation from relevant chunks
        
        Args:
            question: Test question
            relevant_chunks: List of relevant chunk texts
            
        Returns:
            Simulated answer
        """
        if not relevant_chunks:
            return ""
        
        # Simple simulation: combine key information from chunks
        combined_text = " ".join(relevant_chunks)
        
        # Extract key phrases based on question keywords
        question_keywords = set(question.lower().split())
        
        # Find sentences containing question keywords
        sentences = combined_text.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in question_keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return ". ".join(relevant_sentences[:3])  # Limit to 3 sentences
        else:
            return combined_text[:200]  # Return first 200 characters
    
    def _calculate_answer_similarity(self, question: str, answer: str) -> float:
        """
        Calculate similarity between question and answer
        
        Args:
            question: Test question
            answer: Generated answer
            
        Returns:
            Similarity score
        """
        if not answer.strip():
            return 0.0
        
        # Vectorize question and answer
        texts = [question, answer]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity
    
    def _evaluate_context_relevance(self, chunk_texts: List[str], questions: List[str]) -> float:
        """
        Evaluate context relevance of retrieved chunks
        
        Args:
            chunk_texts: List of chunk texts
            questions: List of test questions
            
        Returns:
            Context relevance score
        """
        if not chunk_texts or not questions:
            return 0.0
        
        relevance_scores = []
        
        for question in questions:
            relevant_chunks = self._find_relevant_chunks(question, chunk_texts)
            
            if not relevant_chunks:
                relevance_scores.append(0.0)
                continue
            
            # Calculate average relevance of retrieved chunks
            chunk_relevances = []
            for chunk in relevant_chunks:
                relevance = self._calculate_chunk_relevance(question, chunk)
                chunk_relevances.append(relevance)
            
            avg_relevance = np.mean(chunk_relevances)
            relevance_scores.append(avg_relevance)
        
        return np.mean(relevance_scores)
    
    def _calculate_chunk_relevance(self, question: str, chunk: str) -> float:
        """
        Calculate relevance of a chunk to a question
        
        Args:
            question: Test question
            chunk: Chunk text
            
        Returns:
            Relevance score
        """
        # Vectorize question and chunk
        texts = [question, chunk]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity
    
    def _evaluate_response_coherence(self, chunk_texts: List[str], questions: List[str]) -> float:
        """
        Evaluate response coherence
        
        Args:
            chunk_texts: List of chunk texts
            questions: List of test questions
            
        Returns:
            Response coherence score
        """
        if not chunk_texts or not questions:
            return 0.0
        
        coherence_scores = []
        
        for question in questions:
            relevant_chunks = self._find_relevant_chunks(question, chunk_texts)
            
            if len(relevant_chunks) < 2:
                coherence_scores.append(1.0)  # Single chunk is coherent
                continue
            
            # Calculate coherence between retrieved chunks
            chunk_coherences = []
            for i in range(len(relevant_chunks) - 1):
                coherence = self._calculate_chunk_coherence(
                    relevant_chunks[i], relevant_chunks[i + 1]
                )
                chunk_coherences.append(coherence)
            
            avg_coherence = np.mean(chunk_coherences)
            coherence_scores.append(avg_coherence)
        
        return np.mean(coherence_scores)
    
    def _calculate_chunk_coherence(self, chunk1: str, chunk2: str) -> float:
        """
        Calculate coherence between two chunks
        
        Args:
            chunk1: First chunk text
            chunk2: Second chunk text
            
        Returns:
            Coherence score
        """
        # Vectorize chunks
        texts = [chunk1, chunk2]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity
    
    def _evaluate_response_completeness(self, chunk_texts: List[str], questions: List[str]) -> float:
        """
        Evaluate response completeness
        
        Args:
            chunk_texts: List of chunk texts
            questions: List of test questions
            
        Returns:
            Response completeness score
        """
        if not chunk_texts or not questions:
            return 0.0
        
        completeness_scores = []
        
        for question in questions:
            relevant_chunks = self._find_relevant_chunks(question, chunk_texts)
            
            if not relevant_chunks:
                completeness_scores.append(0.0)
                continue
            
            # Simulate answer generation
            answer = self._simulate_answer_generation(question, relevant_chunks)
            
            # Calculate completeness based on answer length and content
            completeness = self._calculate_answer_completeness(question, answer)
            completeness_scores.append(completeness)
        
        return np.mean(completeness_scores)
    
    def _calculate_answer_completeness(self, question: str, answer: str) -> float:
        """
        Calculate completeness of an answer
        
        Args:
            question: Test question
            answer: Generated answer
            
        Returns:
            Completeness score
        """
        if not answer.strip():
            return 0.0
        
        # Simple completeness heuristics
        factors = []
        
        # 1. Answer length (longer answers tend to be more complete)
        length_score = min(1.0, len(answer.split()) / 50)  # Normalize to 50 words
        factors.append(length_score)
        
        # 2. Question keyword coverage
        question_keywords = set(question.lower().split())
        answer_lower = answer.lower()
        keyword_coverage = sum(1 for keyword in question_keywords if keyword in answer_lower)
        coverage_score = keyword_coverage / len(question_keywords) if question_keywords else 0.0
        factors.append(coverage_score)
        
        # 3. Sentence structure (complete sentences)
        sentences = answer.split('.')
        complete_sentences = sum(1 for s in sentences if len(s.strip().split()) >= 5)
        sentence_score = complete_sentences / len(sentences) if sentences else 0.0
        factors.append(sentence_score)
        
        return np.mean(factors)
    
    def _calculate_overall_rag_score(self, 
                                   precision: float, 
                                   recall: float, 
                                   f1: float,
                                   answer_accuracy: float, 
                                   context_relevance: float,
                                   response_coherence: float, 
                                   response_completeness: float) -> float:
        """
        Calculate overall RAG score
        
        Args:
            precision: Retrieval precision
            recall: Retrieval recall
            f1: Retrieval F1 score
            answer_accuracy: Answer accuracy
            context_relevance: Context relevance
            response_coherence: Response coherence
            response_completeness: Response completeness
            
        Returns:
            Overall RAG score
        """
        # Weighted combination of metrics
        weights = {
            'retrieval': 0.3,      # Retrieval performance
            'accuracy': 0.25,      # Answer accuracy
            'relevance': 0.2,      # Context relevance
            'coherence': 0.15,     # Response coherence
            'completeness': 0.1    # Response completeness
        }
        
        retrieval_score = (precision + recall + f1) / 3
        accuracy_score = answer_accuracy
        relevance_score = context_relevance
        coherence_score = response_coherence
        completeness_score = response_completeness
        
        overall_score = (
            weights['retrieval'] * retrieval_score +
            weights['accuracy'] * accuracy_score +
            weights['relevance'] * relevance_score +
            weights['coherence'] * coherence_score +
            weights['completeness'] * completeness_score
        )
        
        return overall_score
    
    def compare_rag_strategies(self) -> pd.DataFrame:
        """
        Compare RAG performance across strategies
        
        Returns:
            DataFrame with comparison results
        """
        if not self.results:
            logger.warning("No RAG evaluation results available")
            return pd.DataFrame()
        
        # Convert results to DataFrame
        data = []
        for strategy_name, metrics in self.results.items():
            data.append({
                'Strategy': strategy_name,
                'Retrieval_Precision': metrics.retrieval_precision,
                'Retrieval_Recall': metrics.retrieval_recall,
                'Retrieval_F1': metrics.retrieval_f1,
                'Answer_Accuracy': metrics.answer_accuracy,
                'Context_Relevance': metrics.context_relevance,
                'Response_Coherence': metrics.response_coherence,
                'Response_Completeness': metrics.response_completeness,
                'Overall_RAG_Score': metrics.overall_rag_score
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_rag_report(self, output_path: str = "rag_evaluation_report.html"):
        """
        Generate RAG-specific evaluation report
        
        Args:
            output_path: Path for the output HTML report
        """
        if not self.results:
            logger.warning("No RAG evaluation results available")
            return
        
        # Create comparison DataFrame
        df = self.compare_rag_strategies()
        
        # Generate HTML report
        html_content = self._generate_rag_html_report(df)
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"RAG evaluation report saved to: {output_path}")
    
    def _generate_rag_html_report(self, df: pd.DataFrame) -> str:
        """
        Generate HTML report for RAG evaluation
        
        Args:
            df: DataFrame with RAG evaluation results
            
        Returns:
            HTML content string
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ADNOC RAG Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { font-weight: bold; color: #333; }
                .score { color: #0066cc; }
                .highlight { background-color: #e6f3ff; }
                .rag-score { background-color: #d4edda; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>ADNOC RAG Evaluation Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <h2>RAG Performance Summary</h2>
            <p>This report compares {num_strategies} chunking strategies for RAG performance.</p>
            
            <h2>Detailed RAG Metrics</h2>
            {table_html}
            
            <h2>Key RAG Findings</h2>
            <ul>
                <li><strong>Best Retrieval Performance:</strong> {best_retrieval}</li>
                <li><strong>Best Answer Accuracy:</strong> {best_accuracy}</li>
                <li><strong>Best Context Relevance:</strong> {best_relevance}</li>
                <li><strong>Best Response Coherence:</strong> {best_coherence}</li>
                <li><strong>Best Overall RAG Score:</strong> {best_overall}</li>
            </ul>
            
            <h2>RAG-Specific Recommendations</h2>
            <p>Based on the RAG evaluation results, consider the following recommendations:</p>
            <ul>
                <li>Prioritize strategies with high retrieval precision for accurate information retrieval</li>
                <li>Choose strategies with good answer accuracy for reliable responses</li>
                <li>Consider context relevance for maintaining conversation flow</li>
                <li>Balance coherence and completeness for user satisfaction</li>
                <li>Use overall RAG score as the primary selection criterion</li>
            </ul>
            
            <h2>RAG Performance Insights</h2>
            <p>The evaluation reveals important insights about chunking strategy performance in RAG applications:</p>
            <ul>
                <li>Retrieval performance directly impacts answer quality</li>
                <li>Context relevance is crucial for maintaining conversation coherence</li>
                <li>Response completeness affects user satisfaction</li>
                <li>Optimal chunk size varies by document type and use case</li>
            </ul>
        </body>
        </html>
        """
        
        # Generate table HTML
        table_html = df.to_html(index=False, classes='dataframe', float_format='%.3f')
        
        # Find best performers
        best_retrieval = df.loc[df['Retrieval_F1'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_accuracy = df.loc[df['Answer_Accuracy'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_relevance = df.loc[df['Context_Relevance'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_coherence = df.loc[df['Response_Coherence'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        best_overall = df.loc[df['Overall_RAG_Score'].idxmax(), 'Strategy'] if not df.empty else 'N/A'
        
        # Format the HTML
        html_content = html_template.format(
            timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            num_strategies=len(df),
            table_html=table_html,
            best_retrieval=best_retrieval,
            best_accuracy=best_accuracy,
            best_relevance=best_relevance,
            best_coherence=best_coherence,
            best_overall=best_overall
        )
        
        return html_content

def main():
    """Example usage of the RAG evaluator"""
    # Initialize RAG evaluator
    config = RAGEvaluationConfig(
        max_retrieved_chunks=5,
        similarity_threshold=0.7,
        evaluation_mode='automatic'
    )
    
    evaluator = RAGEvaluator(config)
    
    # Example: Evaluate RAG performance
    # This would typically be called with actual chunking results
    logger.info("RAG evaluator initialized successfully")
    logger.info("Use evaluate_rag_performance() to evaluate specific strategies")
    logger.info("Use compare_rag_strategies() to compare multiple strategies")
    logger.info("Use generate_rag_report() to create HTML reports")

if __name__ == "__main__":
    main() 