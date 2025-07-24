#!/usr/bin/env python3
"""
Question Clustering and Topic Extraction
========================================

Advanced clustering system that:
1. Clusters questions from chunks based on similarity
2. Extracts representative topics from each cluster
3. Provides cluster analysis and insights
4. Optimizes cluster count for better topic diversity

Author: Data Engineering Team
Purpose: Question clustering and topic extraction for improved index understanding
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from openai import OpenAI
import re
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

@dataclass
class QuestionCluster:
    """Question cluster with metadata"""
    cluster_id: int
    questions: List[str]
    centroid_question: str
    topics: List[str]
    size: int
    coherence_score: float
    representative_chunks: List[str]

@dataclass
class ClusterAnalysis:
    """Complete cluster analysis results"""
    clusters: List[QuestionCluster]
    total_questions: int
    total_clusters: int
    avg_cluster_size: float
    overall_coherence: float
    cluster_metrics: Dict[str, Any]

class QuestionClusterer:
    """
    Advanced question clustering system with topic extraction
    """
    
    def __init__(self):
        """Initialize the question clusterer"""
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        
        # Clustering parameters
        self.min_clusters = 3
        self.max_clusters = 15
        self.min_cluster_size = 2
        self.max_cluster_size = 50
        
        # Vectorization parameters
        self.max_features = 1000
        self.ngram_range = (1, 2)
        
    def cluster_questions(self, 
                         chunks: List[Dict[str, Any]], 
                         method: str = 'auto',
                         n_clusters: Optional[int] = None) -> ClusterAnalysis:
        """
        Cluster questions from chunks and extract topics
        
        Args:
            chunks: List of chunk dictionaries with questions
            method: Clustering method ('kmeans', 'dbscan', 'agglomerative', 'auto')
            n_clusters: Number of clusters (if None, auto-determine)
            
        Returns:
            ClusterAnalysis object with results
        """
        logger.info(f"Clustering questions from {len(chunks)} chunks")
        
        # Extract all questions
        all_questions = []
        question_to_chunk = {}
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', 'unknown')
            questions = chunk.get('questions', [])
            
            for question in questions:
                all_questions.append(question)
                question_to_chunk[question] = chunk_id
        
        if not all_questions:
            logger.warning("No questions found in chunks")
            return self._create_empty_analysis()
        
        logger.info(f"Found {len(all_questions)} questions to cluster")
        
        # Vectorize questions
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        try:
            question_vectors = vectorizer.fit_transform(all_questions)
        except Exception as e:
            logger.error(f"Error vectorizing questions: {e}")
            return self._create_empty_analysis()
        
        # Determine optimal number of clusters
        if n_clusters is None:
            n_clusters = self._determine_optimal_clusters(
                question_vectors, method
            )
        
        # Perform clustering
        cluster_labels = self._perform_clustering(
            question_vectors, method, n_clusters
        )
        
        # Create clusters
        clusters = self._create_clusters(
            all_questions, cluster_labels, question_to_chunk, chunks
        )
        
        # Extract topics from clusters
        clusters_with_topics = self._extract_cluster_topics(clusters)
        
        # Calculate metrics
        metrics = self._calculate_cluster_metrics(
            clusters_with_topics, question_vectors, cluster_labels
        )
        
        return ClusterAnalysis(
            clusters=clusters_with_topics,
            total_questions=len(all_questions),
            total_clusters=len(clusters_with_topics),
            avg_cluster_size=np.mean([c.size for c in clusters_with_topics]),
            overall_coherence=metrics['overall_coherence'],
            cluster_metrics=metrics
        )
    
    def _determine_optimal_clusters(self, vectors, method: str) -> int:
        """Determine optimal number of clusters"""
        if method == 'dbscan':
            return self._optimize_dbscan(vectors)
        else:
            return self._optimize_kmeans(vectors)
    
    def _optimize_kmeans(self, vectors) -> int:
        """Find optimal number of clusters using K-means"""
        silhouette_scores = []
        cluster_counts = range(self.min_clusters, min(self.max_clusters + 1, vectors.shape[0] // 2))
        
        for n_clusters in cluster_counts:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(vectors)
                
                if len(set(cluster_labels)) > 1:
                    score = silhouette_score(vectors, cluster_labels)
                    silhouette_scores.append((n_clusters, score))
                else:
                    silhouette_scores.append((n_clusters, 0))
                    
            except Exception as e:
                logger.warning(f"Error with {n_clusters} clusters: {e}")
                silhouette_scores.append((n_clusters, 0))
        
        # Find best score
        if silhouette_scores:
            best_n_clusters, best_score = max(silhouette_scores, key=lambda x: x[1])
            logger.info(f"Optimal clusters: {best_n_clusters} (silhouette: {best_score:.3f})")
            return best_n_clusters
        
        return self.min_clusters
    
    def _optimize_dbscan(self, vectors) -> int:
        """Find optimal DBSCAN parameters"""
        # Try different eps values
        eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        best_eps = 0.3
        best_n_clusters = 0
        
        for eps in eps_values:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=2)
                cluster_labels = dbscan.fit_predict(vectors)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                
                if self.min_clusters <= n_clusters <= self.max_clusters:
                    best_eps = eps
                    best_n_clusters = n_clusters
                    break
                    
            except Exception as e:
                logger.warning(f"Error with DBSCAN eps={eps}: {e}")
        
        logger.info(f"DBSCAN optimal eps: {best_eps}, clusters: {best_n_clusters}")
        return best_n_clusters
    
    def _perform_clustering(self, vectors, method: str, n_clusters: int) -> np.ndarray:
        """Perform clustering with specified method"""
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'agglomerative':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            eps = 0.3  # Default value
            clusterer = DBSCAN(eps=eps, min_samples=2)
        else:  # auto
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        cluster_labels = clusterer.fit_predict(vectors)
        logger.info(f"Clustering completed with {len(set(cluster_labels))} clusters")
        
        return cluster_labels
    
    def _create_clusters(self, questions: List[str], 
                        cluster_labels: np.ndarray,
                        question_to_chunk: Dict[str, str],
                        chunks: List[Dict[str, Any]]) -> List[QuestionCluster]:
        """Create QuestionCluster objects from clustering results"""
        clusters = []
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise points in DBSCAN
                continue
                
            # Get questions in this cluster
            cluster_questions = [
                q for i, q in enumerate(questions) 
                if cluster_labels[i] == cluster_id
            ]
            
            if len(cluster_questions) < self.min_cluster_size:
                continue
            
            # Find centroid question (most representative)
            centroid_question = self._find_centroid_question(cluster_questions)
            
            # Get representative chunks
            chunk_ids = set(question_to_chunk[q] for q in cluster_questions)
            representative_chunks = [cid for cid in chunk_ids]
            
            # Calculate coherence
            coherence_score = self._calculate_cluster_coherence(cluster_questions)
            
            cluster = QuestionCluster(
                cluster_id=cluster_id,
                questions=cluster_questions,
                centroid_question=centroid_question,
                topics=[],  # Will be filled later
                size=len(cluster_questions),
                coherence_score=coherence_score,
                representative_chunks=representative_chunks
            )
            
            clusters.append(cluster)
        
        # Sort by size (largest first)
        clusters.sort(key=lambda x: x.size, reverse=True)
        
        logger.info(f"Created {len(clusters)} clusters")
        return clusters
    
    def _find_centroid_question(self, questions: List[str]) -> str:
        """Find the most representative question in a cluster"""
        if not questions:
            return ""
        
        if len(questions) == 1:
            return questions[0]
        
        # Use TF-IDF to find most representative question
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            vectors = vectorizer.fit_transform(questions)
            
            # Calculate average vector
            avg_vector = vectors.mean(axis=0)
            
            # Find question closest to average
            similarities = []
            for i in range(vectors.shape[0]):
                sim = np.dot(vectors[i].toarray().flatten(), avg_vector.toarray().flatten())
                similarities.append((sim, questions[i]))
            
            # Return question with highest similarity
            similarities.sort(reverse=True)
            return similarities[0][1]
            
        except Exception as e:
            logger.warning(f"Error finding centroid question: {e}")
            return questions[0]
    
    def _calculate_cluster_coherence(self, questions: List[str]) -> float:
        """Calculate semantic coherence of a cluster"""
        if len(questions) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            vectors = vectorizer.fit_transform(questions)
            
            similarities = []
            for i in range(len(questions)):
                for j in range(i + 1, len(questions)):
                    sim = np.dot(vectors[i].toarray().flatten(), vectors[j].toarray().flatten())
                    similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating cluster coherence: {e}")
            return 0.5
    
    def _extract_cluster_topics(self, clusters: List[QuestionCluster]) -> List[QuestionCluster]:
        """Extract topics from each cluster using LLM"""
        for cluster in clusters:
            try:
                topics = self._extract_topics_from_cluster(cluster)
                cluster.topics = topics
            except Exception as e:
                logger.error(f"Error extracting topics for cluster {cluster.cluster_id}: {e}")
                cluster.topics = []
        
        return clusters
    
    def _extract_topics_from_cluster(self, cluster: QuestionCluster) -> List[str]:
        """Extract topics from a question cluster using LLM"""
        try:
            # Create prompt with cluster questions
            questions_text = "\n".join(cluster.questions[:10])  # Limit to first 10 questions
            
            prompt = f"""
Analyze the following cluster of related questions and extract 3-5 key topics or themes that represent what this cluster is about.

Questions in cluster:
{questions_text}

Extract topics that:
1. Are specific and descriptive (2-4 words each)
2. Capture the main themes of the questions
3. Would be useful for categorizing and searching this type of content
4. Are relevant for information retrieval

Return only the topics, one per line, without numbering or bullet points.
"""
            
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing question clusters and extracting key topics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3,
            )
            
            topics = response.choices[0].message.content.strip().split('\n')
            topics = [t.strip() for t in topics if t.strip()]
            
            return topics[:5]
            
        except Exception as e:
            logger.error(f"Error extracting topics from cluster: {e}")
            return []
    
    def _calculate_cluster_metrics(self, clusters: List[QuestionCluster], 
                                 vectors, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive cluster metrics"""
        if not clusters:
            return {}
        
        # Basic metrics
        cluster_sizes = [c.size for c in clusters]
        coherence_scores = [c.coherence_score for c in clusters]
        
        # Silhouette score (if we have the original labels)
        silhouette = 0.0
        try:
            if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
                silhouette = silhouette_score(vectors, cluster_labels)
        except:
            pass
        
        metrics = {
            'total_clusters': len(clusters),
            'avg_cluster_size': np.mean(cluster_sizes),
            'std_cluster_size': np.std(cluster_sizes),
            'min_cluster_size': min(cluster_sizes),
            'max_cluster_size': max(cluster_sizes),
            'avg_coherence': np.mean(coherence_scores),
            'std_coherence': np.std(coherence_scores),
            'silhouette_score': silhouette,
            'cluster_size_distribution': {
                'small': len([s for s in cluster_sizes if s <= 3]),
                'medium': len([s for s in cluster_sizes if 3 < s <= 10]),
                'large': len([s for s in cluster_sizes if s > 10])
            }
        }
        
        return metrics
    
    def _create_empty_analysis(self) -> ClusterAnalysis:
        """Create empty analysis when no questions are available"""
        return ClusterAnalysis(
            clusters=[],
            total_questions=0,
            total_clusters=0,
            avg_cluster_size=0.0,
            overall_coherence=0.0,
            cluster_metrics={}
        )
    
    def visualize_clusters(self, analysis: ClusterAnalysis, 
                          output_file: Optional[str] = None) -> None:
        """Create visualization of clusters"""
        if not analysis.clusters:
            logger.warning("No clusters to visualize")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cluster size distribution
        sizes = [c.size for c in analysis.clusters]
        ax1.bar(range(len(sizes)), sizes)
        ax1.set_title('Cluster Size Distribution')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Questions')
        
        # 2. Coherence scores
        coherences = [c.coherence_score for c in analysis.clusters]
        ax2.bar(range(len(coherences)), coherences, color='orange')
        ax2.set_title('Cluster Coherence Scores')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Coherence Score')
        
        # 3. Topics per cluster
        topic_counts = [len(c.topics) for c in analysis.clusters]
        ax3.bar(range(len(topic_counts)), topic_counts, color='green')
        ax3.set_title('Topics per Cluster')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Topics')
        
        # 4. Summary statistics
        ax4.axis('off')
        summary_text = f"""
        Cluster Analysis Summary
        
        Total Questions: {analysis.total_questions}
        Total Clusters: {analysis.total_clusters}
        Average Cluster Size: {analysis.avg_cluster_size:.1f}
        Overall Coherence: {analysis.overall_coherence:.3f}
        
        Cluster Size Distribution:
        - Small (≤3): {analysis.cluster_metrics.get('cluster_size_distribution', {}).get('small', 0)}
        - Medium (4-10): {analysis.cluster_metrics.get('cluster_size_distribution', {}).get('medium', 0)}
        - Large (>10): {analysis.cluster_metrics.get('cluster_size_distribution', {}).get('large', 0)}
        """
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster visualization saved to {output_file}")
        
        plt.show()
    
    def export_results(self, analysis: ClusterAnalysis, 
                      output_file: str) -> None:
        """Export cluster analysis results to JSON"""
        results = {
            'metadata': {
                'total_questions': analysis.total_questions,
                'total_clusters': analysis.total_clusters,
                'avg_cluster_size': analysis.avg_cluster_size,
                'overall_coherence': analysis.overall_coherence
            },
            'cluster_metrics': analysis.cluster_metrics,
            'clusters': [
                {
                    'cluster_id': c.cluster_id,
                    'size': c.size,
                    'coherence_score': c.coherence_score,
                    'centroid_question': c.centroid_question,
                    'topics': c.topics,
                    'questions': c.questions,
                    'representative_chunks': c.representative_chunks
                }
                for c in analysis.clusters
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Cluster analysis results exported to {output_file}")

def main():
    """Main function for testing"""
    # Sample chunks with questions
    sample_chunks = [
        {
            'chunk_id': 'chunk_001',
            'questions': [
                'What are the main chunking strategies?',
                'How does semantic chunking work?',
                'What is the difference between fixed and semantic chunking?'
            ]
        },
        {
            'chunk_id': 'chunk_002',
            'questions': [
                'What are the performance metrics for chunking?',
                'How do we measure chunking effectiveness?',
                'What is the F1-score improvement?'
            ]
        },
        {
            'chunk_id': 'chunk_003',
            'questions': [
                'What is the methodology for chunking?',
                'How do we implement hierarchical chunking?',
                'What are the technical requirements?'
            ]
        }
    ]
    
    clusterer = QuestionClusterer()
    analysis = clusterer.cluster_questions(sample_chunks, method='kmeans')
    
    print(f"Clustering Results:")
    print(f"Total questions: {analysis.total_questions}")
    print(f"Total clusters: {analysis.total_clusters}")
    print(f"Average cluster size: {analysis.avg_cluster_size:.1f}")
    
    for cluster in analysis.clusters:
        print(f"\nCluster {cluster.cluster_id}:")
        print(f"  Size: {cluster.size}")
        print(f"  Topics: {cluster.topics}")
        print(f"  Centroid: {cluster.centroid_question}")

if __name__ == "__main__":
    main() 