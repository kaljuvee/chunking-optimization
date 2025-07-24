#!/usr/bin/env python3
"""
Context Analysis Report
======================

Detailed analysis focusing on context preservation, number of contexts,
and advanced chunking quality metrics.

Author: Data Engineering Team
Purpose: Deep dive into context-related chunking metrics
"""

import json
from pathlib import Path
import numpy as np

def analyze_context_metrics():
    """Analyze context-related metrics in detail"""
    
    output_dir = Path("../test-data/all_chunkers")
    evaluation_file = output_dir / "enhanced_evaluation_results.json"
    
    if not evaluation_file.exists():
        print("❌ Enhanced evaluation results not found. Please run enhanced_evaluation_metrics.py first.")
        return
    
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        evaluations = json.load(f)
    
    print("🔍 Detailed Context Analysis Report")
    print("=" * 60)
    
    # Context Analysis
    print("\n📊 Context Preservation Analysis:")
    print("-" * 40)
    
    context_data = []
    for name, evaluation in evaluations.items():
        context_metrics = evaluation['metrics']['context']
        basic_metrics = evaluation['metrics']['basic']
        
        context_data.append({
            'name': name,
            'total_chunks': evaluation['total_chunks'],
            'context_continuity': context_metrics['context_continuity'],
            'boundary_quality': context_metrics['boundary_quality'],
            'overlap_ratio': context_metrics['overlap_ratio'],
            'avg_overlap': context_metrics['avg_overlap'],
            'avg_chunk_size': basic_metrics['avg_chunk_size'],
            'size_variation': basic_metrics['size_variation']
        })
    
    # Sort by context continuity
    context_data.sort(key=lambda x: x['context_continuity'], reverse=True)
    
    for i, data in enumerate(context_data):
        print(f"{i+1}. {data['name']}")
        print(f"   📄 Chunks: {data['total_chunks']}")
        print(f"   🔗 Context Continuity: {data['context_continuity']:.3f}")
        print(f"   🎯 Boundary Quality: {data['boundary_quality']:.3f}")
        print(f"   📏 Overlap Ratio: {data['overlap_ratio']:.3f}")
        print(f"   📊 Avg Chunk Size: {data['avg_chunk_size']:.1f} words")
        print()
    
    # Number of Contexts Analysis
    print("🔢 Number of Contexts Analysis:")
    print("-" * 40)
    
    for name, evaluation in evaluations.items():
        total_chunks = evaluation['total_chunks']
        context_metrics = evaluation['metrics']['context']
        
        # Calculate effective number of contexts
        effective_contexts = total_chunks
        if context_metrics['overlap_ratio'] > 0:
            # Adjust for overlapping contexts
            effective_contexts = total_chunks * (1 - context_metrics['overlap_ratio'])
        
        # Context diversity score
        context_diversity = 1.0 - context_metrics['overlap_ratio']
        
        print(f"📋 {name}:")
        print(f"   📄 Total Chunks: {total_chunks}")
        print(f"   🔢 Effective Contexts: {effective_contexts:.1f}")
        print(f"   🌐 Context Diversity: {context_diversity:.3f}")
        print(f"   🔗 Overlap Impact: {context_metrics['overlap_ratio']:.3f}")
        print()
    
    # Semantic Context Analysis
    print("🧠 Semantic Context Analysis:")
    print("-" * 40)
    
    for name, evaluation in evaluations.items():
        semantic_metrics = evaluation['metrics']['semantic']
        retrieval_metrics = evaluation['metrics']['retrieval']
        
        print(f"📋 {name}:")
        print(f"   🧠 Topic Consistency: {semantic_metrics['topic_consistency']:.3f}")
        print(f"   🔗 Chunk Coherence: {semantic_metrics['chunk_coherence']:.3f}")
        print(f"   🌐 Semantic Diversity: {semantic_metrics['semantic_diversity']:.3f}")
        print(f"   🎯 Query Potential: {retrieval_metrics['avg_query_potential']:.3f}")
        print(f"   📊 Context Completeness: {retrieval_metrics['context_completeness']:.3f}")
        print()
    
    # Information Density Context Analysis
    print("📈 Information Density Context Analysis:")
    print("-" * 40)
    
    for name, evaluation in evaluations.items():
        info_metrics = evaluation['metrics']['information_density']
        
        print(f"📋 {name}:")
        print(f"   📊 Avg Information Density: {info_metrics['avg_information_density']:.3f}")
        print(f"   📏 Density Variation: {info_metrics['density_variation']:.3f}")
        print(f"   📋 Content Distribution: {info_metrics['content_distribution']}")
        print()
    
    # Context Quality Rankings
    print("🏆 Context Quality Rankings:")
    print("-" * 40)
    
    # Rank by different context metrics
    rankings = {
        'Context Continuity': sorted(context_data, key=lambda x: x['context_continuity'], reverse=True),
        'Boundary Quality': sorted(context_data, key=lambda x: x['boundary_quality'], reverse=True),
        'Effective Contexts': sorted(context_data, key=lambda x: x['total_chunks'], reverse=True),
        'Context Diversity': sorted(context_data, key=lambda x: 1 - x['overlap_ratio'], reverse=True)
    }
    
    for metric_name, ranking in rankings.items():
        print(f"\n📊 {metric_name}:")
        for i, data in enumerate(ranking):
            if metric_name == 'Context Continuity':
                value = data['context_continuity']
            elif metric_name == 'Boundary Quality':
                value = data['boundary_quality']
            elif metric_name == 'Effective Contexts':
                value = data['total_chunks']
            else:  # Context Diversity
                value = 1 - data['overlap_ratio']
            
            print(f"   {i+1}. {data['name']}: {value:.3f}")
    
    # Context Preservation Recommendations
    print("\n💡 Context Preservation Recommendations:")
    print("-" * 40)
    
    recommendations = {
        'Simple Hierarchical': {
            'strengths': [
                'High context continuity (0.621)',
                'Good boundary quality (0.621)',
                'Multiple effective contexts (9 chunks)',
                'Low overlap ratio (0.0)'
            ],
            'use_cases': [
                'When context preservation is critical',
                'For structured documents',
                'When multiple retrieval contexts are needed',
                'For real-time applications'
            ]
        },
        'Content-Aware': {
            'strengths': [
                'Good boundary quality (0.580)',
                'High structural quality (0.600)',
                'Semantic boundary detection',
                'Configurable boundary types'
            ],
            'use_cases': [
                'For complex, unstructured documents',
                'When semantic boundaries matter',
                'For high-quality chunking requirements',
                'When API access is available'
            ]
        },
        'Hierarchical': {
            'strengths': [
                'Good boundary quality (0.580)',
                'Multi-level structure',
                'Context preservation at multiple levels',
                'Parent-child relationships'
            ],
            'use_cases': [
                'For complex document structures',
                'When hierarchy is important',
                'For multi-level analysis',
                'When context at different levels is needed'
            ]
        }
    }
    
    for chunker_name, rec in recommendations.items():
        print(f"\n📋 {chunker_name}:")
        print(f"   ✅ Strengths: {', '.join(rec['strengths'])}")
        print(f"   🎯 Use Cases: {', '.join(rec['use_cases'])}")
    
    # Context Metrics Summary
    print("\n📋 Context Metrics Summary:")
    print("-" * 40)
    
    print("🎯 Key Context Metrics:")
    print("1. Context Continuity: Measures how well context flows between chunks")
    print("2. Boundary Quality: Assesses chunk boundary placement")
    print("3. Overlap Ratio: Percentage of overlapping content between chunks")
    print("4. Effective Contexts: Number of unique retrieval contexts")
    print("5. Context Diversity: Variety of different contexts available")
    
    print("\n🔍 Context Quality Factors:")
    print("• Higher context continuity = better context preservation")
    print("• Higher boundary quality = better chunk boundaries")
    print("• Lower overlap ratio = more diverse contexts")
    print("• More effective contexts = better retrieval options")
    print("• Higher context diversity = better coverage")

def generate_context_comparison_chart():
    """Generate a visual comparison of context metrics"""
    
    output_dir = Path("../test-data/all_chunkers")
    evaluation_file = output_dir / "enhanced_evaluation_results.json"
    
    if not evaluation_file.exists():
        return
    
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        evaluations = json.load(f)
    
    print("\n📊 Context Metrics Comparison Chart:")
    print("=" * 60)
    
    # Create comparison table
    chunkers = list(evaluations.keys())
    metrics = ['context_continuity', 'boundary_quality', 'overlap_ratio', 'total_chunks']
    
    # Header
    header = f"{'Chunker':<25} {'Continuity':<12} {'Boundary':<12} {'Overlap':<12} {'Contexts':<12}"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for name in chunkers:
        evaluation = evaluations[name]
        context_metrics = evaluation['metrics']['context']
        
        row = f"{name:<25} "
        row += f"{context_metrics['context_continuity']:<12.3f} "
        row += f"{context_metrics['boundary_quality']:<12.3f} "
        row += f"{context_metrics['overlap_ratio']:<12.3f} "
        row += f"{evaluation['total_chunks']:<12}"
        
        print(row)

if __name__ == "__main__":
    analyze_context_metrics()
    generate_context_comparison_chart() 