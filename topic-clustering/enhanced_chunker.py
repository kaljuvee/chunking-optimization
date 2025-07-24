#!/usr/bin/env python3
"""
Enhanced Chunker with Question Generation
=========================================

Advanced chunking system that:
1. Creates high-quality, well-sized chunks
2. Generates questions from each chunk
3. Supports overlapping chunks for better context
4. Optimizes chunk boundaries for semantic coherence

Author: Data Engineering Team
Purpose: Enhanced chunking with question generation for improved retrieval
"""

import json
import re
import time
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

@dataclass
class EnhancedChunk:
    """Enhanced chunk with questions and metadata"""
    chunk_id: str
    text: str
    start_index: int
    end_index: int
    word_count: int
    questions: List[str]
    topics: List[str]
    metadata: Dict[str, Any]

class EnhancedChunker:
    """
    Enhanced chunker with question generation and topic extraction
    """
    
    def __init__(self):
        """Initialize the enhanced chunker"""
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        
        # Chunking parameters
        self.min_chunk_size = 100  # Minimum words per chunk
        self.max_chunk_size = 500  # Maximum words per chunk
        self.target_chunk_size = 300  # Target words per chunk
        self.overlap_size = 100  # Overlap between chunks
        self.max_questions_per_chunk = 5  # Maximum questions to generate per chunk
        
    def chunk_text(self, 
                  text: str, 
                  generate_questions: bool = True,
                  extract_topics: bool = True) -> List[EnhancedChunk]:
        """
        Create enhanced chunks with questions and topics
        
        Args:
            text: Input text to chunk
            generate_questions: Whether to generate questions for each chunk
            extract_topics: Whether to extract topics from each chunk
            
        Returns:
            List of enhanced chunks
        """
        logger.info(f"Creating enhanced chunks from text ({len(text.split())} words)")
        
        # Step 1: Create initial chunks with improved boundaries
        initial_chunks = self._create_initial_chunks(text)
        
        # Step 2: Optimize chunk sizes and boundaries
        optimized_chunks = self._optimize_chunks(initial_chunks)
        
        # Step 3: Generate questions and extract topics
        enhanced_chunks = []
        
        for i, chunk in enumerate(optimized_chunks):
            chunk_id = f"chunk_{i:04d}"
            
            # Generate questions if requested
            questions = []
            if generate_questions:
                questions = self._generate_questions(chunk['text'])
            
            # Extract topics if requested
            topics = []
            if extract_topics:
                topics = self._extract_topics(chunk['text'])
            
            # Create enhanced chunk
            enhanced_chunk = EnhancedChunk(
                chunk_id=chunk_id,
                text=chunk['text'],
                start_index=chunk['start_index'],
                end_index=chunk['end_index'],
                word_count=len(chunk['text'].split()),
                questions=questions,
                topics=topics,
                metadata={
                    'chunk_index': i,
                    'strategy': 'enhanced_chunker',
                    'overlap_with_previous': chunk.get('overlap_with_previous', 0),
                    'overlap_with_next': chunk.get('overlap_with_next', 0),
                    'semantic_coherence_score': chunk.get('coherence_score', 0.0),
                    'boundary_quality': chunk.get('boundary_quality', 'good')
                }
            )
            
            enhanced_chunks.append(enhanced_chunk)
            
        logger.info(f"Created {len(enhanced_chunks)} enhanced chunks")
        return enhanced_chunks
    
    def _create_initial_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create initial chunks with improved boundaries"""
        chunks = []
        words = text.split()
        current_pos = 0
        
        while current_pos < len(words):
            # Determine chunk size with some flexibility
            chunk_size = self._determine_optimal_chunk_size(
                words[current_pos:], 
                current_pos, 
                len(words)
            )
            
            # Extract chunk words
            chunk_words = words[current_pos:current_pos + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Find better boundary if possible
            improved_boundary = self._find_better_boundary(
                text, 
                current_pos + len(chunk_text), 
                chunk_text
            )
            
            if improved_boundary > current_pos + len(chunk_text):
                # Extend chunk to better boundary
                extended_words = words[current_pos:improved_boundary]
                chunk_text = ' '.join(extended_words)
                chunk_size = len(extended_words)
            
            # Calculate overlap with previous chunk
            overlap_with_previous = 0
            if chunks:
                overlap_with_previous = self._calculate_overlap(
                    chunks[-1]['text'], chunk_text
                )
            
            chunks.append({
                'text': chunk_text,
                'start_index': current_pos,
                'end_index': current_pos + len(chunk_text),
                'word_count': len(chunk_text.split()),
                'overlap_with_previous': overlap_with_previous,
                'coherence_score': self._calculate_coherence(chunk_text),
                'boundary_quality': self._assess_boundary_quality(chunk_text)
            })
            
            # Move to next position with overlap
            next_pos = current_pos + chunk_size - self.overlap_size
            current_pos = max(next_pos, current_pos + 1)  # Ensure we make progress
        
        return chunks
    
    def _determine_optimal_chunk_size(self, remaining_words: List[str], 
                                    current_pos: int, total_words: int) -> int:
        """Determine optimal chunk size based on context"""
        # Base size
        size = min(self.target_chunk_size, len(remaining_words))
        
        # Adjust based on remaining content
        if current_pos + size > total_words * 0.9:
            # Near end of document, use remaining words
            size = len(remaining_words)
        elif size < self.min_chunk_size and len(remaining_words) > self.min_chunk_size:
            # Too small, try to extend
            size = min(self.max_chunk_size, len(remaining_words))
        
        return size
    
    def _find_better_boundary(self, text: str, current_end: int, chunk_text: str) -> int:
        """Find a better boundary for the chunk"""
        # Look for natural breakpoints within a reasonable range
        search_range = min(200, len(text) - current_end)
        search_text = text[current_end:current_end + search_range]
        
        # Priority: paragraph breaks, sentence breaks, word boundaries
        breakpoints = []
        
        # Paragraph breaks
        para_breaks = [m.start() for m in re.finditer(r'\n\s*\n', search_text)]
        breakpoints.extend([(current_end + pos, 3) for pos in para_breaks])
        
        # Sentence breaks
        sent_breaks = [m.start() for m in re.finditer(r'[.!?]\s+', search_text)]
        breakpoints.extend([(current_end + pos, 2) for pos in sent_breaks])
        
        # Word boundaries (every 50 words)
        word_boundaries = []
        words = search_text.split()
        for i in range(0, len(words), 50):
            if i < len(words):
                word_pos = len(' '.join(words[:i]))
                breakpoints.append((current_end + word_pos, 1))
        
        if breakpoints:
            # Sort by priority and distance
            breakpoints.sort(key=lambda x: (x[1], abs(x[0] - current_end)))
            return breakpoints[0][0]
        
        return current_end
    
    def _calculate_overlap(self, prev_text: str, curr_text: str) -> int:
        """Calculate overlap between consecutive chunks"""
        prev_words = prev_text.split()[-50:]  # Last 50 words
        curr_words = curr_text.split()[:50]   # First 50 words
        
        overlap = 0
        for i in range(min(len(prev_words), len(curr_words))):
            if prev_words[-(i+1)] == curr_words[i]:
                overlap += 1
            else:
                break
        
        return overlap
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate semantic coherence score for a chunk"""
        # Simple TF-IDF based coherence
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 2:
            return 0.5
        
        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._calculate_sentence_similarity(sentences[i], sentences[i+1])
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([sent1, sent2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.5
    
    def _assess_boundary_quality(self, text: str) -> str:
        """Assess the quality of chunk boundaries"""
        # Check if chunk ends with complete sentences
        if text.strip().endswith(('.', '!', '?')):
            return 'excellent'
        elif len(text.split()) >= self.min_chunk_size:
            return 'good'
        else:
            return 'poor'
    
    def _optimize_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize chunks for better quality"""
        optimized = []
        
        for i, chunk in enumerate(chunks):
            # Skip very small chunks unless they're the last one
            if (chunk['word_count'] < self.min_chunk_size and 
                i < len(chunks) - 1):
                # Merge with next chunk
                if i + 1 < len(chunks):
                    merged_text = chunk['text'] + ' ' + chunks[i + 1]['text']
                    if len(merged_text.split()) <= self.max_chunk_size:
                        chunks[i + 1]['text'] = merged_text
                        chunks[i + 1]['word_count'] = len(merged_text.split())
                        continue
            
            # Skip very large chunks
            if chunk['word_count'] > self.max_chunk_size:
                # Split large chunk
                sub_chunks = self._split_large_chunk(chunk)
                optimized.extend(sub_chunks)
            else:
                optimized.append(chunk)
        
        return optimized
    
    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a large chunk into smaller ones"""
        text = chunk['text']
        words = text.split()
        sub_chunks = []
        
        for i in range(0, len(words), self.target_chunk_size):
            sub_words = words[i:i + self.target_chunk_size]
            sub_text = ' '.join(sub_words)
            
            sub_chunk = {
                'text': sub_text,
                'start_index': chunk['start_index'] + i,
                'end_index': chunk['start_index'] + i + len(sub_text),
                'word_count': len(sub_words),
                'overlap_with_previous': 0,
                'coherence_score': self._calculate_coherence(sub_text),
                'boundary_quality': self._assess_boundary_quality(sub_text)
            }
            
            sub_chunks.append(sub_chunk)
        
        return sub_chunks
    
    def _generate_questions(self, text: str) -> List[str]:
        """Generate questions from chunk text"""
        try:
            prompt = f"""
Generate {self.max_questions_per_chunk} diverse questions that could be answered by the following text. 
The questions should cover different aspects and be useful for information retrieval.

Text:
{text[:2000]}  # Limit text length for API

Generate questions that:
1. Are specific and answerable
2. Cover different topics within the text
3. Use various question types (what, how, why, when, where)
4. Are relevant for search and retrieval

Return only the questions, one per line, without numbering.
"""
            
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert at generating searchable questions from text content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7,
            )
            
            questions = response.choices[0].message.content.strip().split('\n')
            questions = [q.strip() for q in questions if q.strip() and q.strip().endswith('?')]
            
            return questions[:self.max_questions_per_chunk]
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from chunk text"""
        try:
            prompt = f"""
Extract 3-5 key topics or themes from the following text. 
Topics should be specific and useful for categorization and search.

Text:
{text[:1500]}  # Limit text length for API

Return only the topics, one per line, without numbering or bullet points.
Topics should be:
1. Specific and descriptive
2. Relevant to the content
3. Useful for search and categorization
4. 2-4 words each
"""
            
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting key topics and themes from text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3,
            )
            
            topics = response.choices[0].message.content.strip().split('\n')
            topics = [t.strip() for t in topics if t.strip()]
            
            return topics[:5]
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    def to_dict(self, chunk: EnhancedChunk) -> Dict[str, Any]:
        """Convert EnhancedChunk to dictionary"""
        return {
            'chunk_id': chunk.chunk_id,
            'text': chunk.text,
            'start_index': chunk.start_index,
            'end_index': chunk.end_index,
            'word_count': chunk.word_count,
            'questions': chunk.questions,
            'topics': chunk.topics,
            'metadata': chunk.metadata
        }

def main():
    """Main function for testing"""
    # Test with sample text
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
    """
    
    chunker = EnhancedChunker()
    chunks = chunker.chunk_text(sample_text, generate_questions=True, extract_topics=True)
    
    print(f"Generated {len(chunks)} enhanced chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Words: {chunk.word_count}")
        print(f"  Questions: {len(chunk.questions)}")
        print(f"  Topics: {chunk.topics}")
        print(f"  Coherence: {chunk.metadata['semantic_coherence_score']:.2f}")

if __name__ == "__main__":
    main() 