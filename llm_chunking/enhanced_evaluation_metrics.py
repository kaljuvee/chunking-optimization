#!/usr/bin/env python3
"""
Enhanced Evaluation Metrics for Chunking
========================================

Comprehensive evaluation framework that includes:
1. Context preservation metrics
2. Semantic coherence analysis
3. Information density measures
4. Retrieval effectiveness indicators
5. Structural quality assessment

Author: Data Engineering Team
Purpose: Advanced chunking quality evaluation
"""

import json
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from collections import defaultdict
import math

class EnhancedChunkingEvaluator:
    """
    Enhanced evaluator for chunking quality with multiple metrics
    """
    
    def __init__(self):
        """Initialize the enhanced evaluator"""
        self.metrics = {}
        
    def evaluate_chunks(self, 
                       chunks: List[Dict], 
                       original_text: str,
                       chunking_strategy: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation of chunking quality
        
        Args:
            chunks: List of chunk dictionaries
            original_text: Original text that was chunked
            chunking_strategy: Name of the chunking strategy used
            
        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        print(f"🔍 Evaluating {len(chunks)} chunks from {chunking_strategy}")
        
        evaluation = {
            'strategy': chunking_strategy,
            'total_chunks': len(chunks),
            'original_text_length': len(original_text),
            'original_word_count': len(original_text.split()),
            'metrics': {}
        }
        
        # Basic metrics
        evaluation['metrics']['basic'] = self._calculate_basic_metrics(chunks, original_text)
        
        # Context metrics
        evaluation['metrics']['context'] = self._calculate_context_metrics(chunks, original_text)
        
        # Semantic metrics
        evaluation['metrics']['semantic'] = self._calculate_semantic_metrics(chunks)
        
        # Structural metrics
        evaluation['metrics']['structural'] = self._calculate_structural_metrics(chunks)
        
        # Information density metrics
        evaluation['metrics']['information_density'] = self._calculate_information_density(chunks)
        
        # Retrieval effectiveness metrics
        evaluation['metrics']['retrieval'] = self._calculate_retrieval_metrics(chunks)
        
        # Quality scores
        evaluation['metrics']['quality_scores'] = self._calculate_quality_scores(evaluation['metrics'])
        
        return evaluation
    
    def _calculate_basic_metrics(self, chunks: List[Dict], original_text: str) -> Dict[str, Any]:
        """Calculate basic chunking metrics"""
        metrics = {}
        
        # Chunk size statistics
        chunk_sizes = [len(chunk['text'].split()) for chunk in chunks]
        metrics['avg_chunk_size'] = np.mean(chunk_sizes)
        metrics['std_chunk_size'] = np.std(chunk_sizes)
        metrics['min_chunk_size'] = min(chunk_sizes)
        metrics['max_chunk_size'] = max(chunk_sizes)
        metrics['size_variation'] = metrics['std_chunk_size'] / metrics['avg_chunk_size'] if metrics['avg_chunk_size'] > 0 else 0
        
        # Coverage metrics
        total_chunked_text = sum(len(chunk['text']) for chunk in chunks)
        metrics['text_coverage'] = total_chunked_text / len(original_text) if len(original_text) > 0 else 0
        
        # Word coverage
        original_words = set(original_text.lower().split())
        chunked_words = set()
        for chunk in chunks:
            chunked_words.update(chunk['text'].lower().split())
        metrics['word_coverage'] = len(chunked_words.intersection(original_words)) / len(original_words) if len(original_words) > 0 else 0
        
        return metrics
    
    def _calculate_context_metrics(self, chunks: List[Dict], original_text: str) -> Dict[str, Any]:
        """Calculate context preservation metrics"""
        metrics = {}
        
        # Context window analysis
        context_windows = []
        for chunk in chunks:
            start_idx = chunk.get('start_index', 0)
            end_idx = chunk.get('end_index', len(chunk['text']))
            context_windows.append((start_idx, end_idx))
        
        # Overlap analysis
        overlaps = []
        for i in range(len(context_windows) - 1):
            current_end = context_windows[i][1]
            next_start = context_windows[i + 1][0]
            if next_start < current_end:
                overlap = current_end - next_start
                overlaps.append(overlap)
        
        metrics['avg_overlap'] = np.mean(overlaps) if overlaps else 0
        metrics['overlap_chunks'] = len(overlaps)
        metrics['overlap_ratio'] = len(overlaps) / max(1, len(chunks) - 1)
        
        # Context continuity
        context_continuity_scores = []
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]['text']
            next_chunk = chunks[i + 1]['text']
            
            # Check for sentence continuity
            current_sentences = re.split(r'[.!?]+', current_chunk)
            next_sentences = re.split(r'[.!?]+', next_chunk)
            
            # Look for incomplete sentences
            current_incomplete = any(not sent.strip().endswith(('.', '!', '?')) for sent in current_sentences if sent.strip())
            next_incomplete = any(not sent.strip().endswith(('.', '!', '?')) for sent in next_sentences if sent.strip())
            
            continuity_score = 1.0 if current_incomplete or next_incomplete else 0.0
            context_continuity_scores.append(continuity_score)
        
        metrics['context_continuity'] = np.mean(context_continuity_scores) if context_continuity_scores else 0
        
        # Context boundaries
        boundary_quality_scores = []
        for chunk in chunks:
            text = chunk['text'].strip()
            
            # Check if chunk starts/ends at natural boundaries
            starts_with_capital = text[0].isupper() if text else False
            ends_with_punctuation = text.endswith(('.', '!', '?')) if text else False
            starts_with_paragraph = text.startswith('\n') or text.startswith(' ')
            
            boundary_score = 0
            if starts_with_capital:
                boundary_score += 0.3
            if ends_with_punctuation:
                boundary_score += 0.4
            if starts_with_paragraph:
                boundary_score += 0.3
            
            boundary_quality_scores.append(boundary_score)
        
        metrics['boundary_quality'] = np.mean(boundary_quality_scores)
        
        return metrics
    
    def _calculate_semantic_metrics(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate semantic coherence metrics"""
        metrics = {}
        
        # Topic consistency within chunks
        topic_consistency_scores = []
        for chunk in chunks:
            text = chunk['text']
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                topic_consistency_scores.append(1.0)  # Single sentence is always consistent
                continue
            
            # Simple keyword overlap between consecutive sentences
            keyword_overlaps = []
            for i in range(len(sentences) - 1):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i + 1].lower().split())
                
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                    keyword_overlaps.append(overlap)
            
            avg_overlap = np.mean(keyword_overlaps) if keyword_overlaps else 0
            topic_consistency_scores.append(avg_overlap)
        
        metrics['topic_consistency'] = np.mean(topic_consistency_scores)
        
        # Semantic diversity across chunks
        all_keywords = []
        for chunk in chunks:
            text = chunk['text'].lower()
            # Extract potential keywords (words longer than 4 characters)
            keywords = [word for word in re.findall(r'\b\w+\b', text) if len(word) > 4]
            all_keywords.extend(keywords)
        
        unique_keywords = set(all_keywords)
        metrics['semantic_diversity'] = len(unique_keywords) / len(all_keywords) if all_keywords else 0
        
        # Coherence between consecutive chunks
        chunk_coherence_scores = []
        for i in range(len(chunks) - 1):
            text1 = chunks[i]['text'].lower()
            text2 = chunks[i + 1]['text'].lower()
            
            words1 = set(re.findall(r'\b\w+\b', text1))
            words2 = set(re.findall(r'\b\w+\b', text2))
            
            if len(words1) > 0 and len(words2) > 0:
                coherence = len(words1.intersection(words2)) / len(words1.union(words2))
                chunk_coherence_scores.append(coherence)
        
        metrics['chunk_coherence'] = np.mean(chunk_coherence_scores) if chunk_coherence_scores else 0
        
        return metrics
    
    def _calculate_structural_metrics(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate structural quality metrics"""
        metrics = {}
        
        # Structural completeness
        complete_sentences = 0
        total_sentences = 0
        
        for chunk in chunks:
            text = chunk['text']
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            total_sentences += len(sentences)
            complete_sentences += sum(1 for s in sentences if s.endswith(('.', '!', '?')))
        
        metrics['sentence_completeness'] = complete_sentences / total_sentences if total_sentences > 0 else 0
        
        # Paragraph integrity
        paragraph_scores = []
        for chunk in chunks:
            text = chunk['text']
            paragraphs = text.split('\n\n')
            
            # Check if chunk contains complete paragraphs
            complete_paragraphs = 0
            for para in paragraphs:
                if para.strip() and len(para.strip().split()) > 10:  # Minimum paragraph length
                    complete_paragraphs += 1
            
            paragraph_score = complete_paragraphs / len(paragraphs) if paragraphs else 0
            paragraph_scores.append(paragraph_score)
        
        metrics['paragraph_integrity'] = np.mean(paragraph_scores)
        
        # Structural hierarchy
        hierarchy_scores = []
        for chunk in chunks:
            text = chunk['text']
            
            # Check for structural elements
            has_headers = bool(re.search(r'^[A-Z][^.!?]*$', text, re.MULTILINE))
            has_lists = bool(re.search(r'^\s*[-*•]\s', text, re.MULTILINE))
            has_numbers = bool(re.search(r'^\s*\d+\.', text, re.MULTILINE))
            
            hierarchy_score = sum([has_headers, has_lists, has_numbers]) / 3
            hierarchy_scores.append(hierarchy_score)
        
        metrics['structural_hierarchy'] = np.mean(hierarchy_scores)
        
        return metrics
    
    def _calculate_information_density(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate information density metrics"""
        metrics = {}
        
        # Information density per chunk
        density_scores = []
        for chunk in chunks:
            text = chunk['text']
            
            # Calculate information density indicators
            word_count = len(text.split())
            char_count = len(text)
            
            # Unique words ratio
            unique_words = len(set(text.lower().split()))
            uniqueness_ratio = unique_words / word_count if word_count > 0 else 0
            
            # Named entity density (simple heuristic)
            capitalized_words = len(re.findall(r'\b[A-Z][a-z]+\b', text))
            entity_density = capitalized_words / word_count if word_count > 0 else 0
            
            # Technical term density (words with numbers or special characters)
            technical_terms = len(re.findall(r'\b\w*\d+\w*\b', text))
            technical_density = technical_terms / word_count if word_count > 0 else 0
            
            # Overall density score
            density_score = (uniqueness_ratio + entity_density + technical_density) / 3
            density_scores.append(density_score)
        
        metrics['avg_information_density'] = np.mean(density_scores)
        metrics['density_variation'] = np.std(density_scores)
        
        # Content distribution
        content_distribution = defaultdict(int)
        for chunk in chunks:
            text = chunk['text'].lower()
            
            # Categorize content types
            if any(word in text for word in ['method', 'approach', 'technique']):
                content_distribution['methodological'] += 1
            if any(word in text for word in ['result', 'finding', 'outcome']):
                content_distribution['results'] += 1
            if any(word in text for word in ['conclusion', 'summary', 'recommendation']):
                content_distribution['conclusions'] += 1
            if any(word in text for word in ['introduction', 'background', 'context']):
                content_distribution['introductory'] += 1
        
        metrics['content_distribution'] = dict(content_distribution)
        
        return metrics
    
    def _calculate_retrieval_metrics(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Calculate retrieval effectiveness metrics"""
        metrics = {}
        
        # Query potential (how well chunks could answer questions)
        query_potential_scores = []
        for chunk in chunks:
            text = chunk['text']
            
            # Indicators of query potential
            has_questions = bool(re.search(r'\?', text))
            has_definitions = bool(re.search(r'\bis\b|\bare\b|\bmeans\b', text, re.IGNORECASE))
            has_examples = bool(re.search(r'\bfor example\b|\bsuch as\b|\blike\b', text, re.IGNORECASE))
            has_comparisons = bool(re.search(r'\bcompared to\b|\bversus\b|\bhowever\b|\bwhile\b', text, re.IGNORECASE))
            
            potential_score = sum([has_questions, has_definitions, has_examples, has_comparisons]) / 4
            query_potential_scores.append(potential_score)
        
        metrics['avg_query_potential'] = np.mean(query_potential_scores)
        
        # Retrieval specificity
        specificity_scores = []
        for chunk in chunks:
            text = chunk['text']
            
            # Specificity indicators
            specific_terms = len(re.findall(r'\b\d+%?\b|\b\d+\.\d+\b', text))  # Numbers
            specific_terms += len(re.findall(r'\b[A-Z]{2,}\b', text))  # Acronyms
            specific_terms += len(re.findall(r'\b\w+-\w+\b', text))  # Hyphenated terms
            
            word_count = len(text.split())
            specificity = specific_terms / word_count if word_count > 0 else 0
            specificity_scores.append(specificity)
        
        metrics['avg_specificity'] = np.mean(specificity_scores)
        
        # Context completeness
        context_completeness_scores = []
        for chunk in chunks:
            text = chunk['text']
            
            # Context completeness indicators
            has_subject = bool(re.search(r'\b(the|a|an)\s+\w+', text, re.IGNORECASE))
            has_action = bool(re.search(r'\b(is|are|was|were|has|have|had)\b', text, re.IGNORECASE))
            has_object = bool(re.search(r'\b\w+\s+(the|a|an)\s+\w+', text, re.IGNORECASE))
            
            completeness_score = sum([has_subject, has_action, has_object]) / 3
            context_completeness_scores.append(completeness_score)
        
        metrics['context_completeness'] = np.mean(context_completeness_scores)
        
        return metrics
    
    def _calculate_quality_scores(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality scores"""
        scores = {}
        
        # Context Quality Score
        context_metrics = all_metrics['context']
        context_score = (
            context_metrics['context_continuity'] * 0.3 +
            context_metrics['boundary_quality'] * 0.4 +
            (1 - context_metrics['overlap_ratio']) * 0.3  # Lower overlap is better
        )
        scores['context_quality'] = context_score
        
        # Semantic Quality Score
        semantic_metrics = all_metrics['semantic']
        semantic_score = (
            semantic_metrics['topic_consistency'] * 0.4 +
            semantic_metrics['chunk_coherence'] * 0.3 +
            semantic_metrics['semantic_diversity'] * 0.3
        )
        scores['semantic_quality'] = semantic_score
        
        # Structural Quality Score
        structural_metrics = all_metrics['structural']
        structural_score = (
            structural_metrics['sentence_completeness'] * 0.4 +
            structural_metrics['paragraph_integrity'] * 0.3 +
            structural_metrics['structural_hierarchy'] * 0.3
        )
        scores['structural_quality'] = structural_score
        
        # Information Quality Score
        info_metrics = all_metrics['information_density']
        info_score = (
            info_metrics['avg_information_density'] * 0.5 +
            (1 - info_metrics['density_variation']) * 0.5  # Lower variation is better
        )
        scores['information_quality'] = info_score
        
        # Retrieval Quality Score
        retrieval_metrics = all_metrics['retrieval']
        retrieval_score = (
            retrieval_metrics['avg_query_potential'] * 0.4 +
            retrieval_metrics['avg_specificity'] * 0.3 +
            retrieval_metrics['context_completeness'] * 0.3
        )
        scores['retrieval_quality'] = retrieval_score
        
        # Overall Quality Score
        overall_score = (
            scores['context_quality'] * 0.25 +
            scores['semantic_quality'] * 0.25 +
            scores['structural_quality'] * 0.2 +
            scores['information_quality'] * 0.15 +
            scores['retrieval_quality'] * 0.15
        )
        scores['overall_quality'] = overall_score
        
        return scores

def evaluate_all_chunkers():
    """Evaluate all chunker results with enhanced metrics"""
    
    output_dir = Path("../test-data/all_chunkers")
    evaluator = EnhancedChunkingEvaluator()
    
    print("🔍 Enhanced Evaluation of All Chunkers")
    print("=" * 60)
    
    # Load chunker results
    chunker_files = {
        'Simple Hierarchical': output_dir / "simple_hierarchical_results.json",
        'Content-Aware': output_dir / "content_aware_results.json",
        'Hierarchical': output_dir / "hierarchical_results.json"
    }
    
    evaluations = {}
    
    # Sample text for reference
    sample_text = """
    Technical Implementation Report: Advanced Chunking Strategies
    
    Executive Summary
    This report presents a comprehensive analysis of advanced chunking strategies for document processing systems. The implementation focuses on improving retrieval accuracy and semantic coherence in RAG (Retrieval-Augmented Generation) applications.
    
    Background and Context
    Traditional document chunking approaches rely on fixed-size segmentation, which often breaks semantic units and reduces retrieval effectiveness. Modern approaches leverage machine learning and natural language processing to create more intelligent chunking strategies.
    
    Methodology
    Our approach combines multiple techniques:
    1. Semantic analysis using transformer-based models
    2. Hierarchical document structure analysis
    3. Content-aware boundary detection
    4. Dynamic chunk size optimization
    
    Results and Analysis
    Preliminary results show significant improvements in retrieval accuracy:
    - Semantic chunking: 23% improvement in precision
    - Content-aware chunking: 18% improvement in recall
    - Hierarchical chunking: 31% improvement in F1-score
    
    Technical Implementation
    The hierarchical chunking system implements a multi-level approach:
    - Level 0: Document level (complete document)
    - Level 1: Section level (major topics)
    - Level 2: Subsection level (subtopics)
    - Level 3: Paragraph level (detailed content)
    - Level 4: Sentence level (fine-grained chunks)
    
    Performance Metrics
    System performance was evaluated using standard metrics:
    - Precision: Measures accuracy of retrieved chunks
    - Recall: Measures completeness of retrieval
    - F1-score: Balanced measure of precision and recall
    - Response time: System latency for chunking operations
    
    Conclusion
    The hierarchical chunking approach demonstrates significant improvements over traditional methods. The multi-level structure provides better context preservation and enables more accurate information retrieval.
    """
    
    for name, file_path in chunker_files.items():
        if file_path.exists():
            print(f"\n📊 Evaluating {name}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert hierarchical data to flat chunks if needed
            if 'metadata' in data:  # Hierarchical chunker
                chunks = []
                for chunk_id, chunk_data in data['chunks'].items():
                    flat_chunk = {
                        'text': chunk_data['text'],
                        'start_index': chunk_data.get('start_index', 0),
                        'end_index': chunk_data.get('end_index', len(chunk_data['text'])),
                        'level': chunk_data.get('level', 0),
                        'metadata': chunk_data.get('metadata', {})
                    }
                    chunks.append(flat_chunk)
            else:  # Content-aware chunker
                chunks = data
            
            evaluation = evaluator.evaluate_chunks(chunks, sample_text, name)
            evaluations[name] = evaluation
            
            # Print key metrics
            quality_scores = evaluation['metrics']['quality_scores']
            print(f"   🎯 Overall Quality: {quality_scores['overall_quality']:.3f}")
            print(f"   🔗 Context Quality: {quality_scores['context_quality']:.3f}")
            print(f"   🧠 Semantic Quality: {quality_scores['semantic_quality']:.3f}")
            print(f"   📋 Structural Quality: {quality_scores['structural_quality']:.3f}")
    
    # Save comprehensive evaluation
    evaluation_file = output_dir / "enhanced_evaluation_results.json"
    with open(evaluation_file, 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Enhanced evaluation results saved to: {evaluation_file}")
    
    # Print comparison
    print("\n🏆 Chunker Quality Comparison:")
    print("-" * 50)
    
    for name, evaluation in evaluations.items():
        overall_score = evaluation['metrics']['quality_scores']['overall_quality']
        print(f"{name}: {overall_score:.3f}")
    
    best_chunker = max(evaluations.items(), key=lambda x: x[1]['metrics']['quality_scores']['overall_quality'])
    print(f"\n🥇 Best Quality: {best_chunker[0]} ({best_chunker[1]['metrics']['quality_scores']['overall_quality']:.3f})")

if __name__ == "__main__":
    evaluate_all_chunkers() 