#!/usr/bin/env python3
"""
Run All Chunkers Test
====================

Comprehensive test script to run all available chunkers in the llm_chunking directory.
"""

import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all chunkers
from hierarchical_chunker import HierarchicalChunker
from simple_hierarchical_chunker import SimpleHierarchicalChunker
from content_aware_chunker import ContentAwareChunker

# Load environment variables
load_dotenv()

def run_all_chunkers_test():
    """Run all available chunkers with sample text"""
    
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
    
    # Create output directory
    output_dir = Path("../test-data/all_chunkers")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Test 1: Simple Hierarchical Chunker (Rule-based)
    print("\n🧪 Testing Simple Hierarchical Chunker (Rule-based)...")
    try:
        start_time = time.time()
        chunker = SimpleHierarchicalChunker()
        hierarchy = chunker.chunk_text(
            text=sample_text,
            max_levels=4,
            min_chunk_size=30,
            max_chunk_size=500
        )
        end_time = time.time()
        
        results['simple_hierarchical'] = {
            'success': True,
            'chunks': hierarchy['metadata']['total_chunks'],
            'levels': hierarchy['metadata']['levels'],
            'time': end_time - start_time,
            'data': hierarchy
        }
        
        print(f"✅ Success: {hierarchy['metadata']['total_chunks']} chunks, {hierarchy['metadata']['levels']} levels in {end_time - start_time:.2f}s")
        
        # Save results
        with open(output_dir / "simple_hierarchical_results.json", 'w', encoding='utf-8') as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['simple_hierarchical'] = {
            'success': False,
            'error': str(e)
        }
    
    # Test 2: Content-Aware Chunker (LLM-based)
    print("\n🧪 Testing Content-Aware Chunker (LLM-based)...")
    try:
        start_time = time.time()
        chunker = ContentAwareChunker()
        chunks = chunker.chunk_text(
            text=sample_text,
            target_chunk_size=300,
            overlap_size=50,
            boundary_types=['section', 'paragraph', 'topic']
        )
        end_time = time.time()
        
        results['content_aware'] = {
            'success': True,
            'chunks': len(chunks),
            'time': end_time - start_time,
            'data': chunks
        }
        
        print(f"✅ Success: {len(chunks)} chunks in {end_time - start_time:.2f}s")
        
        # Save results
        with open(output_dir / "content_aware_results.json", 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['content_aware'] = {
            'success': False,
            'error': str(e)
        }
    
    # Test 3: Hierarchical Chunker (LLM-based)
    print("\n🧪 Testing Hierarchical Chunker (LLM-based)...")
    try:
        start_time = time.time()
        chunker = HierarchicalChunker()
        hierarchy = chunker.chunk_text(
            text=sample_text,
            max_levels=4,
            min_chunk_size=30,
            max_chunk_size=500
        )
        end_time = time.time()
        
        results['hierarchical'] = {
            'success': True,
            'chunks': hierarchy['metadata']['total_chunks'],
            'levels': hierarchy['metadata']['levels'],
            'time': end_time - start_time,
            'data': hierarchy
        }
        
        print(f"✅ Success: {hierarchy['metadata']['total_chunks']} chunks, {hierarchy['metadata']['levels']} levels in {end_time - start_time:.2f}s")
        
        # Save results
        with open(output_dir / "hierarchical_results.json", 'w', encoding='utf-8') as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        results['hierarchical'] = {
            'success': False,
            'error': str(e)
        }
    
    # Generate summary report
    print("\n📊 Summary Report:")
    print("=" * 50)
    
    successful_tests = 0
    total_tests = len(results)
    
    for test_name, result in results.items():
        if result['success']:
            successful_tests += 1
            if 'chunks' in result:
                print(f"✅ {test_name}: {result['chunks']} chunks in {result['time']:.2f}s")
            else:
                print(f"✅ {test_name}: Completed in {result['time']:.2f}s")
        else:
            print(f"❌ {test_name}: Failed - {result['error']}")
    
    print(f"\n🎯 Success Rate: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
    
    # Save summary
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': successful_tests/total_tests*100,
        'results': results
    }
    
    with open(output_dir / "test_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 All results saved to: {output_dir}")
    
    return successful_tests == total_tests

def compare_chunkers():
    """Compare the results of different chunkers"""
    
    output_dir = Path("../test-data/all_chunkers")
    
    print("\n🔍 Chunker Comparison:")
    print("=" * 50)
    
    comparisons = []
    
    # Load results
    chunker_files = {
        'Simple Hierarchical': output_dir / "simple_hierarchical_results.json",
        'Content-Aware': output_dir / "content_aware_results.json",
        'Hierarchical': output_dir / "hierarchical_results.json"
    }
    
    for name, file_path in chunker_files.items():
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'chunks' in data:  # Hierarchical chunker
                chunk_count = data['metadata']['total_chunks']
                levels = data['metadata']['levels']
                print(f"{name}: {chunk_count} chunks, {levels} levels")
            else:  # Content-aware chunker
                chunk_count = len(data)
                print(f"{name}: {chunk_count} chunks")
            
            comparisons.append({
                'name': name,
                'chunks': chunk_count,
                'file': file_path
            })
    
    if comparisons:
        print(f"\n📈 Most chunks: {max(comparisons, key=lambda x: x['chunks'])['name']}")
        print(f"📉 Least chunks: {min(comparisons, key=lambda x: x['chunks'])['name']}")

if __name__ == "__main__":
    success = run_all_chunkers_test()
    if success:
        compare_chunkers()
    sys.exit(0 if success else 1) 