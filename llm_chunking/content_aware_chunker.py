#!/usr/bin/env python3
"""
ADNOC Content-Aware Chunker
===========================

This module implements content-aware chunking using LLM to identify natural
breakpoints in text while preserving semantic coherence.

Author: Data Engineering Team
Purpose: LLM-based content boundary detection
"""

import json
import argparse
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

@dataclass
class ChunkBoundary:
    """Data class for chunk boundary information"""
    start_index: int
    end_index: int
    confidence: float
    boundary_type: str  # 'section', 'paragraph', 'sentence', 'topic'
    reasoning: str

class ContentAwareChunker:
    """
    Content-aware chunker using LLM for boundary detection
    """
    
    def __init__(self):
        """Initialize the content-aware chunker"""
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        
    def chunk_text(self, 
                  text: str, 
                  target_chunk_size: int = 500,
                  overlap_size: int = 50,
                  boundary_types: List[str] = None) -> List[Dict]:
        """
        Chunk text using content-aware boundaries
        
        Args:
            text: Input text to chunk
            target_chunk_size: Target chunk size in words
            overlap_size: Overlap size in words
            boundary_types: Types of boundaries to consider
            
        Returns:
            List of chunk dictionaries
        """
        if boundary_types is None:
            boundary_types = ['section', 'paragraph', 'topic']
        
        logger.info(f"Chunking text with content-aware boundaries")
        logger.info(f"Target chunk size: {target_chunk_size} words")
        logger.info(f"Overlap size: {overlap_size} words")
        
        # Step 1: Identify potential boundaries
        boundaries = self._identify_boundaries(text, boundary_types)
        
        # Step 2: Create chunks based on boundaries
        chunks = self._create_chunks_from_boundaries(
            text, boundaries, target_chunk_size, overlap_size
        )
        
        # Step 3: Optimize chunks for coherence
        optimized_chunks = self._optimize_chunks(chunks)
        
        return optimized_chunks
    
    def _identify_boundaries(self, text: str, boundary_types: List[str]) -> List[ChunkBoundary]:
        """
        Identify potential chunk boundaries in text
        
        Args:
            text: Input text
            boundary_types: Types of boundaries to identify
            
        Returns:
            List of chunk boundaries
        """
        boundaries = []
        
        # Split text into manageable segments for LLM processing
        segments = self._split_text_for_analysis(text)
        
        for i, segment in enumerate(segments):
            logger.info(f"Analyzing segment {i+1}/{len(segments)}")
            
            # Use LLM to identify boundaries in this segment
            segment_boundaries = self._analyze_segment_boundaries(segment, boundary_types)
            
            # Adjust indices for global text position
            offset = sum(len(segments[j]) for j in range(i))
            for boundary in segment_boundaries:
                boundary.start_index += offset
                boundary.end_index += offset
                boundaries.append(boundary)
        
        # Sort boundaries by start index
        boundaries.sort(key=lambda x: x.start_index)
        
        logger.info(f"Identified {len(boundaries)} potential boundaries")
        return boundaries
    
    def _split_text_for_analysis(self, text: str, max_segment_size: int = 2000) -> List[str]:
        """
        Split text into segments suitable for LLM analysis
        
        Args:
            text: Input text
            max_segment_size: Maximum segment size in characters
            
        Returns:
            List of text segments
        """
        if len(text) <= max_segment_size:
            return [text]
        
        # Split on paragraph boundaries
        paragraphs = text.split('\n\n')
        segments = []
        current_segment = ""
        
        for paragraph in paragraphs:
            if len(current_segment) + len(paragraph) <= max_segment_size:
                current_segment += paragraph + '\n\n'
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = paragraph + '\n\n'
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    def _analyze_segment_boundaries(self, segment: str, boundary_types: List[str]) -> List[ChunkBoundary]:
        """
        Use LLM to analyze boundaries in a text segment
        
        Args:
            segment: Text segment to analyze
            boundary_types: Types of boundaries to identify
            
        Returns:
            List of chunk boundaries
        """
        prompt = self._create_boundary_analysis_prompt(segment, boundary_types)
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert in text analysis and content structuring."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1,
            )
            
            result = response.choices[0].message.content.strip()
            return self._parse_boundary_analysis(result, segment)
            
        except Exception as e:
            logger.error(f"Error analyzing segment boundaries: {e}")
            return []
    
    def _create_boundary_analysis_prompt(self, segment: str, boundary_types: List[str]) -> str:
        """
        Create prompt for boundary analysis
        
        Args:
            segment: Text segment to analyze
            boundary_types: Types of boundaries to identify
            
        Returns:
            Analysis prompt
        """
        boundary_descriptions = {
            'section': 'major topic changes or section breaks',
            'paragraph': 'paragraph boundaries that represent logical breaks',
            'topic': 'topic shifts within paragraphs',
            'sentence': 'sentence boundaries for fine-grained chunking'
        }
        
        boundary_list = ', '.join([f"{bt} ({boundary_descriptions[bt]})" for bt in boundary_types])
        
        prompt = f"""
Analyze the following text segment and identify natural breakpoints for chunking. Focus on {boundary_list}.

Text segment:
{segment}

Please identify the best chunking boundaries and return your analysis in the following JSON format:
{{
    "boundaries": [
        {{
            "start_index": <character position>,
            "end_index": <character position>,
            "confidence": <0.0-1.0>,
            "boundary_type": "<section|paragraph|topic|sentence>",
            "reasoning": "<explanation of why this is a good boundary>"
        }}
    ]
}}

Guidelines:
1. Start_index and end_index should be character positions within the text
2. Confidence should reflect how strong the boundary is (0.0-1.0)
3. Choose boundaries that preserve semantic coherence
4. Prefer natural breaks over arbitrary positions
5. Consider the context and flow of information

Return only the JSON response, no additional text.
"""
        return prompt
    
    def _parse_boundary_analysis(self, result: str, segment: str) -> List[ChunkBoundary]:
        """
        Parse LLM boundary analysis result
        
        Args:
            result: LLM response
            segment: Original text segment
            
        Returns:
            List of chunk boundaries
        """
        boundaries = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return boundaries
            
            data = json.loads(json_match.group())
            
            for boundary_data in data.get('boundaries', []):
                boundary = ChunkBoundary(
                    start_index=boundary_data.get('start_index', 0),
                    end_index=boundary_data.get('end_index', 0),
                    confidence=boundary_data.get('confidence', 0.5),
                    boundary_type=boundary_data.get('boundary_type', 'paragraph'),
                    reasoning=boundary_data.get('reasoning', '')
                )
                boundaries.append(boundary)
                
        except Exception as e:
            logger.error(f"Error parsing boundary analysis: {e}")
        
        return boundaries
    
    def _create_chunks_from_boundaries(self, 
                                     text: str, 
                                     boundaries: List[ChunkBoundary],
                                     target_chunk_size: int,
                                     overlap_size: int) -> List[Dict]:
        """
        Create chunks based on identified boundaries
        
        Args:
            text: Input text
            boundaries: List of chunk boundaries
            target_chunk_size: Target chunk size in words
            overlap_size: Overlap size in words
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find the best boundary for this chunk
            best_boundary = self._find_best_boundary(
                text, boundaries, current_pos, target_chunk_size
            )
            
            if best_boundary:
                # Create chunk up to the boundary
                chunk_end = best_boundary.end_index
                chunk_text = text[current_pos:chunk_end].strip()
                
                if chunk_text:
                    chunks.append({
                        'text': chunk_text,
                        'start_index': current_pos,
                        'end_index': chunk_end,
                        'boundary_confidence': best_boundary.confidence,
                        'boundary_type': best_boundary.boundary_type,
                        'boundary_reasoning': best_boundary.reasoning
                    })
                
                # Move to next position with overlap
                current_pos = max(current_pos + 1, chunk_end - overlap_size)
            else:
                # No suitable boundary found, create chunk of target size
                words = text[current_pos:].split()
                if len(words) <= target_chunk_size:
                    chunk_text = text[current_pos:].strip()
                    if chunk_text:
                        chunks.append({
                            'text': chunk_text,
                            'start_index': current_pos,
                            'end_index': len(text),
                            'boundary_confidence': 0.0,
                            'boundary_type': 'forced',
                            'boundary_reasoning': 'No natural boundary found'
                        })
                    break
                else:
                    # Create chunk of target size
                    chunk_words = words[:target_chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    chunk_end = current_pos + len(chunk_text)
                    
                    chunks.append({
                        'text': chunk_text,
                        'start_index': current_pos,
                        'end_index': chunk_end,
                        'boundary_confidence': 0.0,
                        'boundary_type': 'forced',
                        'boundary_reasoning': 'Target size reached'
                    })
                    
                    current_pos = chunk_end - overlap_size
        
        return chunks
    
    def _find_best_boundary(self, 
                           text: str, 
                           boundaries: List[ChunkBoundary],
                           current_pos: int, 
                           target_chunk_size: int) -> Optional[ChunkBoundary]:
        """
        Find the best boundary for chunking
        
        Args:
            text: Input text
            boundaries: List of chunk boundaries
            current_pos: Current position in text
            target_chunk_size: Target chunk size in words
            
        Returns:
            Best boundary or None
        """
        target_end = current_pos + (target_chunk_size * 5)  # Approximate character count
        
        # Find boundaries within reasonable range
        candidate_boundaries = [
            b for b in boundaries 
            if b.start_index >= current_pos and b.end_index <= target_end
        ]
        
        if not candidate_boundaries:
            return None
        
        # Score boundaries based on multiple factors
        scored_boundaries = []
        for boundary in candidate_boundaries:
            score = self._score_boundary(boundary, current_pos, target_chunk_size)
            scored_boundaries.append((boundary, score))
        
        # Return boundary with highest score
        if scored_boundaries:
            scored_boundaries.sort(key=lambda x: x[1], reverse=True)
            return scored_boundaries[0][0]
        
        return None
    
    def _score_boundary(self, 
                       boundary: ChunkBoundary, 
                       current_pos: int, 
                       target_chunk_size: int) -> float:
        """
        Score a boundary based on multiple factors
        
        Args:
            boundary: Chunk boundary to score
            current_pos: Current position in text
            target_chunk_size: Target chunk size in words
            
        Returns:
            Boundary score
        """
        # Factor 1: Confidence score
        confidence_score = boundary.confidence
        
        # Factor 2: Distance from target size (closer is better)
        chunk_size = len(text[current_pos:boundary.end_index].split())
        size_diff = abs(chunk_size - target_chunk_size)
        size_score = max(0, 1 - (size_diff / target_chunk_size))
        
        # Factor 3: Boundary type preference
        type_scores = {
            'section': 1.0,
            'paragraph': 0.8,
            'topic': 0.6,
            'sentence': 0.4,
            'forced': 0.1
        }
        type_score = type_scores.get(boundary.boundary_type, 0.5)
        
        # Weighted combination
        total_score = (
            0.4 * confidence_score +
            0.3 * size_score +
            0.3 * type_score
        )
        
        return total_score
    
    def _optimize_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Optimize chunks for better coherence
        
        Args:
            chunks: List of chunks to optimize
            
        Returns:
            Optimized chunks
        """
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Add metadata
            chunk['chunk_index'] = i
            chunk['strategy'] = 'content_aware_llm'
            chunk['word_count'] = len(chunk['text'].split())
            
            # Validate chunk
            if self._validate_chunk(chunk):
                optimized_chunks.append(chunk)
            else:
                logger.warning(f"Invalid chunk {i}: {chunk.get('text', '')[:100]}...")
        
        logger.info(f"Optimized {len(optimized_chunks)} chunks")
        return optimized_chunks
    
    def _validate_chunk(self, chunk: Dict) -> bool:
        """
        Validate a chunk
        
        Args:
            chunk: Chunk to validate
            
        Returns:
            True if chunk is valid
        """
        text = chunk.get('text', '')
        
        # Check minimum size
        if len(text.strip()) < 10:
            return False
        
        # Check for complete sentences (basic validation)
        sentences = text.split('.')
        if len(sentences) < 1:
            return False
        
        return True

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Content-aware chunking using LLM")
    parser.add_argument("--input", "-i", required=True, help="Path to input text file")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
    parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk size in words")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap size in words")
    parser.add_argument("--boundary-types", nargs='+', 
                       default=['section', 'paragraph', 'topic'],
                       help="Types of boundaries to consider")
    args = parser.parse_args()
    
    # Read input text
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize chunker
    chunker = ContentAwareChunker()
    
    # Perform chunking
    chunks = chunker.chunk_text(
        text=text,
        target_chunk_size=args.chunk_size,
        overlap_size=args.overlap,
        boundary_types=args.boundary_types
    )
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Content-aware chunking completed: {len(chunks)} chunks saved to {args.output}")

if __name__ == "__main__":
    main() 