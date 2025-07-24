#!/usr/bin/env python3
"""
ADNOC Hierarchical Chunker
==========================

This module implements hierarchical chunking that creates multi-level
chunking with parent-child relationships for context-aware retrieval.

Author: Data Engineering Team
Purpose: Multi-level chunking with relationships
"""

import json
import argparse
import os
from typing import List, Dict, Optional, Tuple, Any
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
class HierarchicalChunk:
    """Data class for hierarchical chunk information"""
    chunk_id: str
    level: int  # 0=document, 1=section, 2=subsection, 3=paragraph, 4=sentence
    text: str
    parent_id: Optional[str]
    children_ids: List[str]
    metadata: Dict[str, Any]
    start_index: int
    end_index: int

class HierarchicalChunker:
    """
    Hierarchical chunker for multi-level text structuring
    """
    
    def __init__(self):
        """Initialize the hierarchical chunker"""
        self.client = OpenAI(
            api_key=OPENAI_API_KEY
        )
        self.chunk_counter = 0
        
    def chunk_text(self, 
                  text: str, 
                  max_levels: int = 4,
                  min_chunk_size: int = 50,
                  max_chunk_size: int = 1000) -> Dict[str, Any]:
        """
        Create hierarchical chunks from text
        
        Args:
            text: Input text to chunk
            max_levels: Maximum number of hierarchy levels
            min_chunk_size: Minimum chunk size in words
            max_chunk_size: Maximum chunk size in words
            
        Returns:
            Dictionary containing hierarchical chunks and metadata
        """
        logger.info(f"Creating hierarchical chunks with {max_levels} levels")
        
        # Initialize hierarchy
        hierarchy = {
            'document_id': 'doc_0',
            'chunks': {},
            'relationships': {},
            'metadata': {
                'total_chunks': 0,
                'levels': max_levels,
                'document_length': len(text)
            }
        }
        
        # Create document-level chunk
        doc_chunk = HierarchicalChunk(
            chunk_id='doc_0',
            level=0,
            text=text,
            parent_id=None,
            children_ids=[],
            metadata={'type': 'document', 'word_count': len(text.split())},
            start_index=0,
            end_index=len(text)
        )
        
        hierarchy['chunks']['doc_0'] = self._chunk_to_dict(doc_chunk)
        
        # Build hierarchy recursively
        self._build_hierarchy(text, 'doc_0', 1, max_levels, min_chunk_size, max_chunk_size, hierarchy)
        
        # Update metadata
        hierarchy['metadata']['total_chunks'] = len(hierarchy['chunks'])
        
        logger.info(f"Created {hierarchy['metadata']['total_chunks']} hierarchical chunks")
        return hierarchy
    
    def _build_hierarchy(self, 
                        text: str, 
                        parent_id: str, 
                        level: int, 
                        max_levels: int,
                        min_chunk_size: int,
                        max_chunk_size: int,
                        hierarchy: Dict[str, Any]):
        """
        Recursively build hierarchy for a given text segment
        
        Args:
            text: Text to process
            parent_id: ID of parent chunk
            level: Current hierarchy level
            max_levels: Maximum levels to create
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            hierarchy: Hierarchy dictionary to update
        """
        if level > max_levels or len(text.strip()) < min_chunk_size:
            return
        
        # Determine chunking strategy for this level
        if level == 1:  # Section level
            sub_chunks = self._chunk_by_sections(text, level, parent_id, min_chunk_size, max_chunk_size)
        elif level == 2:  # Subsection level
            sub_chunks = self._chunk_by_subsections(text, level, parent_id, min_chunk_size, max_chunk_size)
        elif level == 3:  # Paragraph level
            sub_chunks = self._chunk_by_paragraphs(text, level, parent_id, min_chunk_size, max_chunk_size)
        elif level == 4:  # Sentence level
            sub_chunks = self._chunk_by_sentences(text, level, parent_id, min_chunk_size, max_chunk_size)
        else:
            return
        
        # Add chunks to hierarchy
        for chunk in sub_chunks:
            hierarchy['chunks'][chunk.chunk_id] = self._chunk_to_dict(chunk)
            
            # Update parent's children list
            if parent_id in hierarchy['chunks']:
                hierarchy['chunks'][parent_id]['children_ids'].append(chunk.chunk_id)
            
            # Recursively process children
            if len(chunk.text.split()) >= min_chunk_size * 2:  # Only recurse if enough content
                self._build_hierarchy(
                    chunk.text, chunk.chunk_id, level + 1, max_levels,
                    min_chunk_size, max_chunk_size, hierarchy
                )
    
    def _chunk_by_sections(self, text: str, level: int, parent_id: str, 
                          min_size: int, max_size: int) -> List[HierarchicalChunk]:
        """Chunk text by sections using LLM analysis"""
        return self._llm_chunking(text, level, parent_id, 'sections', min_size, max_size)
    
    def _chunk_by_subsections(self, text: str, level: int, parent_id: str,
                             min_size: int, max_size: int) -> List[HierarchicalChunk]:
        """Chunk text by subsections using LLM analysis"""
        return self._llm_chunking(text, level, parent_id, 'subsections', min_size, max_size)
    
    def _chunk_by_paragraphs(self, text: str, level: int, parent_id: str,
                            min_size: int, max_size: int) -> List[HierarchicalChunk]:
        """Chunk text by paragraphs"""
        paragraphs = text.split('\n\n')
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph.split()) >= min_size:
                chunk_id = f"{parent_id}_para_{i}"
                chunk = HierarchicalChunk(
                    chunk_id=chunk_id,
                    level=level,
                    text=paragraph,
                    parent_id=parent_id,
                    children_ids=[],
                    metadata={'type': 'paragraph', 'paragraph_index': i},
                    start_index=text.find(paragraph),
                    end_index=text.find(paragraph) + len(paragraph)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, level: int, parent_id: str,
                           min_size: int, max_size: int) -> List[HierarchicalChunk]:
        """Chunk text by sentences"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence.split()) >= min_size:
                chunk_id = f"{parent_id}_sent_{i}"
                chunk = HierarchicalChunk(
                    chunk_id=chunk_id,
                    level=level,
                    text=sentence,
                    parent_id=parent_id,
                    children_ids=[],
                    metadata={'type': 'sentence', 'sentence_index': i},
                    start_index=text.find(sentence),
                    end_index=text.find(sentence) + len(sentence)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _llm_chunking(self, text: str, level: int, parent_id: str, 
                     chunk_type: str, min_size: int, max_size: int) -> List[HierarchicalChunk]:
        """
        Use LLM to chunk text by specified type
        
        Args:
            text: Text to chunk
            level: Hierarchy level
            parent_id: Parent chunk ID
            chunk_type: Type of chunking ('sections', 'subsections')
            min_size: Minimum chunk size
            max_size: Maximum chunk size
            
        Returns:
            List of hierarchical chunks
        """
        prompt = self._create_hierarchical_chunking_prompt(text, chunk_type, min_size, max_size)
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an expert in document structure analysis and hierarchical organization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1,
            )
            
            result = response.choices[0].message.content.strip()
            return self._parse_hierarchical_chunks(result, text, level, parent_id, chunk_type)
            
        except Exception as e:
            logger.error(f"Error in LLM chunking: {e}")
            return []
    
    def _create_hierarchical_chunking_prompt(self, text: str, chunk_type: str, 
                                           min_size: int, max_size: int) -> str:
        """Create prompt for hierarchical chunking"""
        
        chunk_descriptions = {
            'sections': 'major sections or topics within the document',
            'subsections': 'subsections or subtopics within each section'
        }
        
        description = chunk_descriptions.get(chunk_type, chunk_type)
        
        prompt = f"""
Analyze the following text and identify {description}. Create logical chunks that maintain semantic coherence.

Text:
{text}

Requirements:
- Minimum chunk size: {min_size} words
- Maximum chunk size: {max_size} words
- Preserve semantic coherence
- Maintain logical flow
- Avoid breaking mid-topic

Return your analysis in the following JSON format:
{{
    "chunks": [
        {{
            "text": "<chunk text>",
            "title": "<chunk title or topic>",
            "start_index": <character position>,
            "end_index": <character position>,
            "word_count": <number of words>,
            "reasoning": "<explanation of why this chunk makes sense>"
        }}
    ]
}}

Guidelines:
1. Each chunk should be a complete, coherent unit
2. Chunks should not overlap
3. Cover the entire text
4. Maintain logical progression
5. Consider natural breakpoints

Return only the JSON response, no additional text.
"""
        return prompt
    
    def _parse_hierarchical_chunks(self, result: str, text: str, level: int, 
                                 parent_id: str, chunk_type: str) -> List[HierarchicalChunk]:
        """Parse LLM hierarchical chunking result"""
        chunks = []
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if not json_match:
                logger.warning("No JSON found in LLM response")
                return chunks
            
            data = json.loads(json_match.group())
            
            for i, chunk_data in enumerate(data.get('chunks', [])):
                chunk_id = f"{parent_id}_{chunk_type}_{i}"
                
                chunk = HierarchicalChunk(
                    chunk_id=chunk_id,
                    level=level,
                    text=chunk_data.get('text', ''),
                    parent_id=parent_id,
                    children_ids=[],
                    metadata={
                        'type': chunk_type,
                        'title': chunk_data.get('title', ''),
                        'word_count': chunk_data.get('word_count', 0),
                        'reasoning': chunk_data.get('reasoning', '')
                    },
                    start_index=chunk_data.get('start_index', 0),
                    end_index=chunk_data.get('end_index', 0)
                )
                chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Error parsing hierarchical chunks: {e}")
        
        return chunks
    
    def _chunk_to_dict(self, chunk: HierarchicalChunk) -> Dict[str, Any]:
        """Convert HierarchicalChunk to dictionary"""
        return {
            'chunk_id': chunk.chunk_id,
            'level': chunk.level,
            'text': chunk.text,
            'parent_id': chunk.parent_id,
            'children_ids': chunk.children_ids,
            'metadata': chunk.metadata,
            'start_index': chunk.start_index,
            'end_index': chunk.end_index
        }
    
    def get_flat_chunks(self, hierarchy: Dict[str, Any], 
                       level_filter: Optional[int] = None) -> List[Dict]:
        """
        Get flat list of chunks from hierarchy
        
        Args:
            hierarchy: Hierarchy dictionary
            level_filter: Optional level filter
            
        Returns:
            List of flat chunks
        """
        flat_chunks = []
        
        for chunk_id, chunk_data in hierarchy['chunks'].items():
            if level_filter is None or chunk_data['level'] == level_filter:
                flat_chunk = {
                    'chunk_index': len(flat_chunks),
                    'text': chunk_data['text'],
                    'strategy': f'hierarchical_level_{chunk_data["level"]}',
                    'level': chunk_data['level'],
                    'parent_id': chunk_data['parent_id'],
                    'children_ids': chunk_data['children_ids'],
                    'metadata': chunk_data['metadata']
                }
                flat_chunks.append(flat_chunk)
        
        return flat_chunks
    
    def get_context_path(self, hierarchy: Dict[str, Any], chunk_id: str) -> List[Dict]:
        """
        Get context path from root to specified chunk
        
        Args:
            hierarchy: Hierarchy dictionary
            chunk_id: Target chunk ID
            
        Returns:
            List of chunks in context path
        """
        path = []
        current_id = chunk_id
        
        while current_id and current_id in hierarchy['chunks']:
            chunk_data = hierarchy['chunks'][current_id]
            path.insert(0, chunk_data)
            current_id = chunk_data['parent_id']
        
        return path
    
    def get_siblings(self, hierarchy: Dict[str, Any], chunk_id: str) -> List[Dict]:
        """
        Get sibling chunks of specified chunk
        
        Args:
            hierarchy: Hierarchy dictionary
            chunk_id: Target chunk ID
            
        Returns:
            List of sibling chunks
        """
        if chunk_id not in hierarchy['chunks']:
            return []
        
        parent_id = hierarchy['chunks'][chunk_id]['parent_id']
        if not parent_id or parent_id not in hierarchy['chunks']:
            return []
        
        siblings = []
        for sibling_id in hierarchy['chunks'][parent_id]['children_ids']:
            if sibling_id != chunk_id:
                siblings.append(hierarchy['chunks'][sibling_id])
        
        return siblings

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Hierarchical chunking with LLM")
    parser.add_argument("--input", "-i", required=True, help="Path to input text file")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
    parser.add_argument("--max-levels", type=int, default=4, help="Maximum hierarchy levels")
    parser.add_argument("--min-chunk-size", type=int, default=50, help="Minimum chunk size in words")
    parser.add_argument("--max-chunk-size", type=int, default=1000, help="Maximum chunk size in words")
    parser.add_argument("--flat-output", action='store_true', help="Output flat chunks instead of hierarchy")
    parser.add_argument("--level-filter", type=int, help="Filter chunks by specific level")
    args = parser.parse_args()
    
    # Read input text
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize chunker
    chunker = HierarchicalChunker()
    
    # Perform hierarchical chunking
    hierarchy = chunker.chunk_text(
        text=text,
        max_levels=args.max_levels,
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size
    )
    
    # Prepare output
    if args.flat_output:
        output_data = chunker.get_flat_chunks(hierarchy, args.level_filter)
    else:
        output_data = hierarchy
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Hierarchical chunking completed: {hierarchy['metadata']['total_chunks']} chunks saved to {args.output}")

if __name__ == "__main__":
    main() 