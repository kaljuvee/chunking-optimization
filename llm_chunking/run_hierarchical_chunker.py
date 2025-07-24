#!/usr/bin/env python3
"""
Run Hierarchical Chunker Test
============================

Simple test script to run the hierarchical chunker with sample text.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hierarchical_chunker import HierarchicalChunker

# Load environment variables
load_dotenv()

def run_hierarchical_chunker_test():
    """Run hierarchical chunker test with sample text"""
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return False
    
    print("✅ OpenAI API key found")
    
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
    
    print("\n🧪 Testing Hierarchical Chunker...")
    
    try:
        # Initialize chunker
        chunker = HierarchicalChunker()
        print("✅ HierarchicalChunker initialized successfully")
        
        # Create hierarchical chunks
        print("📝 Creating hierarchical chunks...")
        hierarchy = chunker.chunk_text(
            text=sample_text,
            max_levels=4,
            min_chunk_size=30,
            max_chunk_size=500
        )
        
        print(f"✅ Created hierarchical structure with {hierarchy['metadata']['total_chunks']} chunks")
        
        # Display hierarchy structure
        print("\n📊 Hierarchy Structure:")
        for chunk_id, chunk_data in hierarchy['chunks'].items():
            level = chunk_data['level']
            indent = "  " * level
            title = chunk_data['metadata'].get('title', chunk_data['text'][:50] + "...")
            word_count = chunk_data['metadata'].get('word_count', len(chunk_data['text'].split()))
            print(f"{indent}Level {level}: {title} ({word_count} words)")
        
        # Get flat chunks for analysis
        print("\n📋 Flat Chunks by Level:")
        for level in range(5):
            flat_chunks = chunker.get_flat_chunks(hierarchy, level_filter=level)
            if flat_chunks:
                print(f"\nLevel {level} chunks ({len(flat_chunks)}):")
                for i, chunk in enumerate(flat_chunks[:3]):  # Show first 3
                    print(f"  {i+1}. {chunk['text'][:100]}...")
                if len(flat_chunks) > 3:
                    print(f"  ... and {len(flat_chunks) - 3} more")
        
        # Save results
        output_dir = Path("../test-data/hierarchical")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "hierarchical_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Show context path example
        if len(hierarchy['chunks']) > 1:
            sample_chunk_id = list(hierarchy['chunks'].keys())[1]
            context_path = chunker.get_context_path(hierarchy, sample_chunk_id)
            print(f"\n🔗 Context path for {sample_chunk_id}:")
            for chunk in context_path:
                print(f"  Level {chunk['level']}: {chunk['metadata'].get('title', chunk['text'][:50])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hierarchical chunker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_hierarchical_chunker_test()
    sys.exit(0 if success else 1) 