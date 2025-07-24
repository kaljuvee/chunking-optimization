#!/usr/bin/env python3
"""
Simple Hierarchical Chunker
===========================

A simplified hierarchical chunker that uses rule-based approaches
instead of LLM for sectioning, making it more reliable and faster.

Author: Data Engineering Team
Purpose: Rule-based hierarchical chunking
"""

import json
import re
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleHierarchicalChunk:
    """Data class for hierarchical chunk information"""
    chunk_id: str
    level: int  # 0=document, 1=section, 2=subsection, 3=paragraph, 4=sentence
    text: str
    parent_id: Optional[str]
    children_ids: List[str]
    metadata: Dict[str, Any]
    start_index: int
    end_index: int

class SimpleHierarchicalChunker:
    """
    Simple hierarchical chunker using rule-based approaches
    """
    
    def __init__(self):
        """Initialize the simple hierarchical chunker"""
        self.chunk_counter = 0
        
    def chunk_text(self, 
                  text: str, 
                  max_levels: int = 4,
                  min_chunk_size: int = 50,
                  max_chunk_size: int = 1000) -> Dict[str, Any]:
        """
        Create hierarchical chunks from text using rule-based approaches
        
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
        doc_chunk = SimpleHierarchicalChunk(
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
                          min_size: int, max_size: int) -> List[SimpleHierarchicalChunk]:
        """Chunk text by sections using rule-based detection"""
        chunks = []
        
        # Look for section headers (all caps, numbered, or bold patterns)
        section_patterns = [
            r'^[A-Z\s]{3,}$',  # All caps headers
            r'^\d+\.\s+[A-Z][^.]*$',  # Numbered sections
            r'^[A-Z][a-z\s]+:$',  # Title case with colon
            r'^[A-Z][a-z\s]+$',  # Title case headers
        ]
        
        lines = text.split('\n')
        current_section = []
        current_title = "Introduction"
        section_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if this line is a section header
            is_header = any(re.match(pattern, line) for pattern in section_patterns)
            
            if is_header and current_section:
                # Save current section
                section_text = '\n'.join(current_section).strip()
                if len(section_text.split()) >= min_size:
                    chunk = SimpleHierarchicalChunk(
                        chunk_id=f"{parent_id}_sec_{len(chunks)}",
                        level=level,
                        text=section_text,
                        parent_id=parent_id,
                        children_ids=[],
                        metadata={
                            'type': 'section',
                            'title': current_title,
                            'word_count': len(section_text.split())
                        },
                        start_index=section_start,
                        end_index=section_start + len(section_text)
                    )
                    chunks.append(chunk)
                
                # Start new section
                current_section = [line]
                current_title = line
                section_start = text.find(line, section_start)
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if len(section_text.split()) >= min_size:
                chunk = SimpleHierarchicalChunk(
                    chunk_id=f"{parent_id}_sec_{len(chunks)}",
                    level=level,
                    text=section_text,
                    parent_id=parent_id,
                    children_ids=[],
                    metadata={
                        'type': 'section',
                        'title': current_title,
                        'word_count': len(section_text.split())
                    },
                    start_index=section_start,
                    end_index=section_start + len(section_text)
                )
                chunks.append(chunk)
        
        # If no sections found, create one large chunk
        if not chunks and len(text.split()) >= min_size:
            chunk = SimpleHierarchicalChunk(
                chunk_id=f"{parent_id}_sec_0",
                level=level,
                text=text,
                parent_id=parent_id,
                children_ids=[],
                metadata={
                    'type': 'section',
                    'title': 'Main Content',
                    'word_count': len(text.split())
                },
                start_index=0,
                end_index=len(text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_subsections(self, text: str, level: int, parent_id: str,
                             min_size: int, max_size: int) -> List[SimpleHierarchicalChunk]:
        """Chunk text by subsections using rule-based detection"""
        chunks = []
        
        # Look for subsection patterns
        subsection_patterns = [
            r'^\d+\.\d+\s+[A-Z][^.]*$',  # Numbered subsections (1.1, 1.2, etc.)
            r'^[A-Z][a-z\s]+:$',  # Title case with colon
            r'^[a-z]\s+[A-Z][^.]*$',  # Lettered subsections (a, b, c)
        ]
        
        lines = text.split('\n')
        current_subsection = []
        current_title = "Subsection"
        subsection_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if this line is a subsection header
            is_header = any(re.match(pattern, line) for pattern in subsection_patterns)
            
            if is_header and current_subsection:
                # Save current subsection
                subsection_text = '\n'.join(current_subsection).strip()
                if len(subsection_text.split()) >= min_size:
                    chunk = SimpleHierarchicalChunk(
                        chunk_id=f"{parent_id}_subsec_{len(chunks)}",
                        level=level,
                        text=subsection_text,
                        parent_id=parent_id,
                        children_ids=[],
                        metadata={
                            'type': 'subsection',
                            'title': current_title,
                            'word_count': len(subsection_text.split())
                        },
                        start_index=subsection_start,
                        end_index=subsection_start + len(subsection_text)
                    )
                    chunks.append(chunk)
                
                # Start new subsection
                current_subsection = [line]
                current_title = line
                subsection_start = text.find(line, subsection_start)
            else:
                current_subsection.append(line)
        
        # Add final subsection
        if current_subsection:
            subsection_text = '\n'.join(current_subsection).strip()
            if len(subsection_text.split()) >= min_size:
                chunk = SimpleHierarchicalChunk(
                    chunk_id=f"{parent_id}_subsec_{len(chunks)}",
                    level=level,
                    text=subsection_text,
                    parent_id=parent_id,
                    children_ids=[],
                    metadata={
                        'type': 'subsection',
                        'title': current_title,
                        'word_count': len(subsection_text.split())
                    },
                    start_index=subsection_start,
                    end_index=subsection_start + len(subsection_text)
                )
                chunks.append(chunk)
        
        # If no subsections found, create one large chunk
        if not chunks and len(text.split()) >= min_size:
            chunk = SimpleHierarchicalChunk(
                chunk_id=f"{parent_id}_subsec_0",
                level=level,
                text=text,
                parent_id=parent_id,
                children_ids=[],
                metadata={
                    'type': 'subsection',
                    'title': 'Content',
                    'word_count': len(text.split())
                },
                start_index=0,
                end_index=len(text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, level: int, parent_id: str,
                            min_size: int, max_size: int) -> List[SimpleHierarchicalChunk]:
        """Chunk text by paragraphs"""
        paragraphs = text.split('\n\n')
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph.split()) >= min_size:
                chunk = SimpleHierarchicalChunk(
                    chunk_id=f"{parent_id}_para_{i}",
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
                           min_size: int, max_size: int) -> List[SimpleHierarchicalChunk]:
        """Chunk text by sentences"""
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_word_count = len(sentence.split())
            
            # If adding this sentence would exceed max size, save current chunk
            if current_word_count + sentence_word_count > max_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= min_size:
                    chunk = SimpleHierarchicalChunk(
                        chunk_id=f"{parent_id}_sent_{len(chunks)}",
                        level=level,
                        text=chunk_text,
                        parent_id=parent_id,
                        children_ids=[],
                        metadata={'type': 'sentence_group', 'sentence_count': len(current_chunk)},
                        start_index=text.find(chunk_text),
                        end_index=text.find(chunk_text) + len(chunk_text)
                    )
                    chunks.append(chunk)
                
                current_chunk = [sentence]
                current_word_count = sentence_word_count
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= min_size:
                chunk = SimpleHierarchicalChunk(
                    chunk_id=f"{parent_id}_sent_{len(chunks)}",
                    level=level,
                    text=chunk_text,
                    parent_id=parent_id,
                    children_ids=[],
                    metadata={'type': 'sentence_group', 'sentence_count': len(current_chunk)},
                    start_index=text.find(chunk_text),
                    end_index=text.find(chunk_text) + len(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_to_dict(self, chunk: SimpleHierarchicalChunk) -> Dict[str, Any]:
        """Convert SimpleHierarchicalChunk to dictionary"""
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
        """Get flat list of chunks from hierarchy"""
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
        """Get context path from root to specified chunk"""
        path = []
        current_id = chunk_id
        
        while current_id and current_id in hierarchy['chunks']:
            chunk_data = hierarchy['chunks'][current_id]
            path.insert(0, chunk_data)
            current_id = chunk_data['parent_id']
        
        return path

def main():
    """Main function for testing"""
    # Sample text for testing
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
    
    # Initialize chunker
    chunker = SimpleHierarchicalChunker()
    
    # Create hierarchical chunks
    hierarchy = chunker.chunk_text(
        text=sample_text,
        max_levels=4,
        min_chunk_size=30,
        max_chunk_size=500
    )
    
    print(f"Created hierarchical structure with {hierarchy['metadata']['total_chunks']} chunks")
    
    # Display hierarchy structure
    print("\nHierarchy Structure:")
    for chunk_id, chunk_data in hierarchy['chunks'].items():
        level = chunk_data['level']
        indent = "  " * level
        title = chunk_data['metadata'].get('title', chunk_data['text'][:50] + "...")
        word_count = chunk_data['metadata'].get('word_count', len(chunk_data['text'].split()))
        print(f"{indent}Level {level}: {title} ({word_count} words)")
    
    # Save results
    output_dir = Path("../test-data/hierarchical")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "simple_hierarchical_chunks.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main() 