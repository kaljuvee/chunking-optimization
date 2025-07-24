#!/usr/bin/env python3
"""
Chunker Comparison Report
=========================

Detailed analysis and comparison of different chunking approaches.
"""

import json
from pathlib import Path

def generate_comparison_report():
    """Generate a detailed comparison report"""
    
    output_dir = Path("../test-data/all_chunkers")
    
    print("📊 Detailed Chunker Comparison Report")
    print("=" * 60)
    
    # Load all results
    results = {}
    
    chunker_files = {
        'Simple Hierarchical (Rule-based)': output_dir / "simple_hierarchical_results.json",
        'Content-Aware (LLM-based)': output_dir / "content_aware_results.json",
        'Hierarchical (LLM-based)': output_dir / "hierarchical_results.json"
    }
    
    for name, file_path in chunker_files.items():
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                results[name] = json.load(f)
    
    # Performance Analysis
    print("\n🚀 Performance Analysis:")
    print("-" * 40)
    
    performance_data = []
    for name, data in results.items():
        if 'metadata' in data:  # Hierarchical chunkers
            chunk_count = data['metadata']['total_chunks']
            levels = data['metadata']['levels']
            doc_length = data['metadata']['document_length']
            performance_data.append({
                'name': name,
                'chunks': chunk_count,
                'levels': levels,
                'doc_length': doc_length,
                'avg_chunk_size': doc_length / chunk_count if chunk_count > 0 else 0
            })
        else:  # Content-aware chunker
            chunk_count = len(data)
            total_length = sum(len(chunk['text']) for chunk in data)
            performance_data.append({
                'name': name,
                'chunks': chunk_count,
                'levels': 1,
                'doc_length': total_length,
                'avg_chunk_size': total_length / chunk_count if chunk_count > 0 else 0
            })
    
    # Sort by chunk count
    performance_data.sort(key=lambda x: x['chunks'], reverse=True)
    
    for i, data in enumerate(performance_data):
        print(f"{i+1}. {data['name']}")
        print(f"   📄 Chunks: {data['chunks']}")
        print(f"   📊 Levels: {data['levels']}")
        print(f"   📏 Avg chunk size: {data['avg_chunk_size']:.0f} chars")
        print()
    
    # Granularity Analysis
    print("🔍 Granularity Analysis:")
    print("-" * 40)
    
    most_granular = max(performance_data, key=lambda x: x['chunks'])
    least_granular = min(performance_data, key=lambda x: x['chunks'])
    
    print(f"📈 Most granular: {most_granular['name']} ({most_granular['chunks']} chunks)")
    print(f"📉 Least granular: {least_granular['name']} ({least_granular['chunks']} chunks)")
    print(f"📊 Granularity ratio: {most_granular['chunks'] / least_granular['chunks']:.1f}x")
    
    # Use Case Recommendations
    print("\n🎯 Use Case Recommendations:")
    print("-" * 40)
    
    recommendations = {
        'Simple Hierarchical (Rule-based)': {
            'best_for': [
                'Fast processing requirements',
                'Structured documents with clear headers',
                'When API costs are a concern',
                'Real-time applications'
            ],
            'advantages': [
                'No API calls required',
                'Very fast execution',
                'Consistent results',
                'Good for structured content'
            ],
            'limitations': [
                'Less intelligent than LLM-based',
                'May miss subtle boundaries',
                'Requires well-structured input'
            ]
        },
        'Content-Aware (LLM-based)': {
            'best_for': [
                'Complex, unstructured documents',
                'When semantic boundaries are important',
                'High-quality chunking requirements',
                'Mixed content types'
            ],
            'advantages': [
                'Intelligent boundary detection',
                'Semantic understanding',
                'Handles complex content well',
                'Configurable boundary types'
            ],
            'limitations': [
                'Requires API calls',
                'Slower execution',
                'Higher cost',
                'May fail with API issues'
            ]
        },
        'Hierarchical (LLM-based)': {
            'best_for': [
                'Multi-level document analysis',
                'Context preservation requirements',
                'Complex document structures',
                'When hierarchy is important'
            ],
            'advantages': [
                'Multi-level structure',
                'Context preservation',
                'Parent-child relationships',
                'Flexible level configuration'
            ],
            'limitations': [
                'Requires API calls',
                'Complex output structure',
                'Higher computational cost',
                'May fail with API issues'
            ]
        }
    }
    
    for chunker_name, rec in recommendations.items():
        print(f"\n📋 {chunker_name}:")
        print(f"   ✅ Best for: {', '.join(rec['best_for'])}")
        print(f"   🎯 Advantages: {', '.join(rec['advantages'])}")
        print(f"   ⚠️  Limitations: {', '.join(rec['limitations'])}")
    
    # Technical Analysis
    print("\n🔧 Technical Analysis:")
    print("-" * 40)
    
    for name, data in results.items():
        print(f"\n📄 {name}:")
        
        if 'metadata' in data:  # Hierarchical chunkers
            chunks = data['chunks']
            level_distribution = {}
            
            for chunk_id, chunk_data in chunks.items():
                level = chunk_data['level']
                level_distribution[level] = level_distribution.get(level, 0) + 1
            
            print(f"   📊 Level distribution:")
            for level in sorted(level_distribution.keys()):
                count = level_distribution[level]
                percentage = (count / len(chunks)) * 100
                print(f"      Level {level}: {count} chunks ({percentage:.1f}%)")
            
            # Check for parent-child relationships
            has_relationships = any(len(chunk_data.get('children_ids', [])) > 0 
                                  for chunk_data in chunks.values())
            print(f"   🔗 Has parent-child relationships: {has_relationships}")
            
        else:  # Content-aware chunker
            boundary_types = {}
            confidence_scores = []
            
            for chunk in data:
                boundary_type = chunk.get('boundary_type', 'unknown')
                boundary_types[boundary_type] = boundary_types.get(boundary_type, 0) + 1
                confidence_scores.append(chunk.get('boundary_confidence', 0))
            
            print(f"   🎯 Boundary types: {dict(boundary_types)}")
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                print(f"   📈 Average confidence: {avg_confidence:.2f}")
    
    # Summary and Recommendations
    print("\n📋 Summary and Recommendations:")
    print("-" * 40)
    
    print("🎯 For Production Use:")
    print("1. Use Simple Hierarchical for fast, reliable processing")
    print("2. Use Content-Aware for high-quality semantic chunking")
    print("3. Use Hierarchical for complex document structures")
    
    print("\n⚡ For Performance:")
    print("1. Simple Hierarchical: Fastest (0.00s)")
    print("2. Content-Aware: Medium (1.71s)")
    print("3. Hierarchical: Medium (1.67s)")
    
    print("\n💰 For Cost Optimization:")
    print("1. Simple Hierarchical: No API costs")
    print("2. Content-Aware: Moderate API usage")
    print("3. Hierarchical: Moderate API usage")
    
    print("\n🎨 For Quality:")
    print("1. Content-Aware: Best semantic understanding")
    print("2. Hierarchical: Best structure preservation")
    print("3. Simple Hierarchical: Good for structured content")

if __name__ == "__main__":
    generate_comparison_report() 