#!/usr/bin/env python3
"""
Comprehensive Evaluation Summary
================================

Complete evaluation summary including all advanced metrics:
- Context preservation and number of contexts
- Semantic coherence and information density
- Structural quality and retrieval effectiveness
- Performance and cost considerations

Author: Data Engineering Team
Purpose: Complete chunking evaluation summary
"""

import json
from pathlib import Path
import numpy as np

def generate_comprehensive_summary():
    """Generate comprehensive evaluation summary"""
    
    output_dir = Path("../test-data/all_chunkers")
    evaluation_file = output_dir / "enhanced_evaluation_results.json"
    
    if not evaluation_file.exists():
        print("❌ Enhanced evaluation results not found. Please run enhanced_evaluation_metrics.py first.")
        return
    
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        evaluations = json.load(f)
    
    print("📊 Comprehensive Chunking Evaluation Summary")
    print("=" * 70)
    
    # Executive Summary
    print("\n🎯 Executive Summary:")
    print("-" * 40)
    
    # Find best performers in different categories
    best_overall = max(evaluations.items(), key=lambda x: x[1]['metrics']['quality_scores']['overall_quality'])
    best_context = max(evaluations.items(), key=lambda x: x[1]['metrics']['quality_scores']['context_quality'])
    best_semantic = max(evaluations.items(), key=lambda x: x[1]['metrics']['quality_scores']['semantic_quality'])
    best_structural = max(evaluations.items(), key=lambda x: x[1]['metrics']['quality_scores']['structural_quality'])
    most_contexts = max(evaluations.items(), key=lambda x: x[1]['total_chunks'])
    
    print(f"🥇 Best Overall Quality: {best_overall[0]} ({best_overall[1]['metrics']['quality_scores']['overall_quality']:.3f})")
    print(f"🔗 Best Context Quality: {best_context[0]} ({best_context[1]['metrics']['quality_scores']['context_quality']:.3f})")
    print(f"🧠 Best Semantic Quality: {best_semantic[0]} ({best_semantic[1]['metrics']['quality_scores']['semantic_quality']:.3f})")
    print(f"📋 Best Structural Quality: {best_structural[0]} ({best_structural[1]['metrics']['quality_scores']['structural_quality']:.3f})")
    print(f"🔢 Most Contexts: {most_contexts[0]} ({most_contexts[1]['total_chunks']} chunks)")
    
    # Detailed Metrics Analysis
    print("\n📈 Detailed Metrics Analysis:")
    print("-" * 40)
    
    metrics_summary = []
    for name, evaluation in evaluations.items():
        metrics = evaluation['metrics']
        quality_scores = metrics['quality_scores']
        
        # Calculate effective contexts
        context_metrics = metrics['context']
        effective_contexts = evaluation['total_chunks'] * (1 - context_metrics['overlap_ratio'])
        
        summary = {
            'name': name,
            'total_chunks': evaluation['total_chunks'],
            'effective_contexts': effective_contexts,
            'overall_quality': quality_scores['overall_quality'],
            'context_quality': quality_scores['context_quality'],
            'semantic_quality': quality_scores['semantic_quality'],
            'structural_quality': quality_scores['structural_quality'],
            'information_quality': quality_scores['information_quality'],
            'retrieval_quality': quality_scores['retrieval_quality'],
            'context_continuity': context_metrics['context_continuity'],
            'boundary_quality': context_metrics['boundary_quality'],
            'overlap_ratio': context_metrics['overlap_ratio'],
            'avg_chunk_size': metrics['basic']['avg_chunk_size'],
            'topic_consistency': metrics['semantic']['topic_consistency'],
            'chunk_coherence': metrics['semantic']['chunk_coherence'],
            'semantic_diversity': metrics['semantic']['semantic_diversity'],
            'sentence_completeness': metrics['structural']['sentence_completeness'],
            'paragraph_integrity': metrics['structural']['paragraph_integrity'],
            'information_density': metrics['information_density']['avg_information_density'],
            'query_potential': metrics['retrieval']['avg_query_potential'],
            'context_completeness': metrics['retrieval']['context_completeness']
        }
        metrics_summary.append(summary)
    
    # Sort by overall quality
    metrics_summary.sort(key=lambda x: x['overall_quality'], reverse=True)
    
    for i, summary in enumerate(metrics_summary):
        print(f"\n{i+1}. {summary['name']}:")
        print(f"   🎯 Overall Quality: {summary['overall_quality']:.3f}")
        print(f"   📄 Total Chunks: {summary['total_chunks']}")
        print(f"   🔢 Effective Contexts: {summary['effective_contexts']:.1f}")
        print(f"   🔗 Context Quality: {summary['context_quality']:.3f}")
        print(f"   🧠 Semantic Quality: {summary['semantic_quality']:.3f}")
        print(f"   📋 Structural Quality: {summary['structural_quality']:.3f}")
    
    # Context Analysis Deep Dive
    print("\n🔍 Context Analysis Deep Dive:")
    print("-" * 40)
    
    print("🎯 Number of Contexts Analysis:")
    for summary in metrics_summary:
        print(f"   {summary['name']}: {summary['effective_contexts']:.1f} effective contexts")
    
    print("\n🔗 Context Continuity Analysis:")
    for summary in metrics_summary:
        print(f"   {summary['name']}: {summary['context_continuity']:.3f} continuity")
    
    print("\n🎯 Boundary Quality Analysis:")
    for summary in metrics_summary:
        print(f"   {summary['name']}: {summary['boundary_quality']:.3f} boundary quality")
    
    # Semantic Analysis
    print("\n🧠 Semantic Analysis:")
    print("-" * 40)
    
    print("📊 Topic Consistency:")
    for summary in metrics_summary:
        print(f"   {summary['name']}: {summary['topic_consistency']:.3f}")
    
    print("\n🔗 Chunk Coherence:")
    for summary in metrics_summary:
        print(f"   {summary['name']}: {summary['chunk_coherence']:.3f}")
    
    print("\n🌐 Semantic Diversity:")
    for summary in metrics_summary:
        print(f"   {summary['name']}: {summary['semantic_diversity']:.3f}")
    
    # Structural Analysis
    print("\n📋 Structural Analysis:")
    print("-" * 40)
    
    print("📄 Sentence Completeness:")
    for summary in metrics_summary:
        print(f"   {summary['name']}: {summary['sentence_completeness']:.3f}")
    
    print("\n📝 Paragraph Integrity:")
    for summary in metrics_summary:
        print(f"   {summary['name']}: {summary['paragraph_integrity']:.3f}")
    
    # Information Density Analysis
    print("\n📈 Information Density Analysis:")
    print("-" * 40)
    
    print("📊 Average Information Density:")
    for summary in metrics_summary:
        print(f"   {summary['name']}: {summary['information_density']:.3f}")
    
    # Retrieval Effectiveness Analysis
    print("\n🎯 Retrieval Effectiveness Analysis:")
    print("-" * 40)
    
    print("📊 Query Potential:")
    for summary in metrics_summary:
        print(f"   {summary['name']}: {summary['query_potential']:.3f}")
    
    print("\n📋 Context Completeness:")
    for summary in metrics_summary:
        print(f"   {summary['name']}: {summary['context_completeness']:.3f}")
    
    # Performance and Cost Analysis
    print("\n⚡ Performance and Cost Analysis:")
    print("-" * 40)
    
    performance_data = {
        'Simple Hierarchical': {
            'speed': 'Very Fast (0.00s)',
            'api_calls': 'None',
            'cost': 'Free',
            'reliability': 'High',
            'scalability': 'High'
        },
        'Content-Aware': {
            'speed': 'Medium (1.71s)',
            'api_calls': 'Required',
            'cost': 'Moderate',
            'reliability': 'Medium (API dependent)',
            'scalability': 'Medium'
        },
        'Hierarchical': {
            'speed': 'Medium (1.67s)',
            'api_calls': 'Required',
            'cost': 'Moderate',
            'reliability': 'Medium (API dependent)',
            'scalability': 'Medium'
        }
    }
    
    for name, perf in performance_data.items():
        print(f"\n📋 {name}:")
        print(f"   ⚡ Speed: {perf['speed']}")
        print(f"   🔌 API Calls: {perf['api_calls']}")
        print(f"   💰 Cost: {perf['cost']}")
        print(f"   🛡️  Reliability: {perf['reliability']}")
        print(f"   📈 Scalability: {perf['scalability']}")
    
    # Recommendations
    print("\n💡 Strategic Recommendations:")
    print("-" * 40)
    
    recommendations = {
        'For Production Systems': {
            'primary': 'Simple Hierarchical',
            'reason': 'High reliability, no API costs, good context preservation',
            'backup': 'Content-Aware (when API is available)'
        },
        'For Research & Development': {
            'primary': 'Content-Aware',
            'reason': 'Best semantic understanding and boundary detection',
            'backup': 'Hierarchical (for complex structures)'
        },
        'For Real-time Applications': {
            'primary': 'Simple Hierarchical',
            'reason': 'Fastest execution, no latency from API calls',
            'backup': 'None (API-based chunkers too slow)'
        },
        'For High-Quality Requirements': {
            'primary': 'Content-Aware',
            'reason': 'Best semantic quality and boundary detection',
            'backup': 'Simple Hierarchical (when API unavailable)'
        }
    }
    
    for use_case, rec in recommendations.items():
        print(f"\n🎯 {use_case}:")
        print(f"   🥇 Primary: {rec['primary']}")
        print(f"   📝 Reason: {rec['reason']}")
        print(f"   🥈 Backup: {rec['backup']}")
    
    # Key Metrics Summary
    print("\n📋 Key Metrics Summary:")
    print("-" * 40)
    
    print("🎯 Context Metrics:")
    print("• Number of Contexts: Total unique retrieval contexts available")
    print("• Context Continuity: How well context flows between chunks")
    print("• Boundary Quality: Quality of chunk boundary placement")
    print("• Overlap Ratio: Percentage of overlapping content")
    
    print("\n🧠 Semantic Metrics:")
    print("• Topic Consistency: Semantic coherence within chunks")
    print("• Chunk Coherence: Semantic similarity between consecutive chunks")
    print("• Semantic Diversity: Variety of topics covered")
    
    print("\n📋 Structural Metrics:")
    print("• Sentence Completeness: Percentage of complete sentences")
    print("• Paragraph Integrity: Quality of paragraph boundaries")
    print("• Structural Hierarchy: Presence of structural elements")
    
    print("\n📈 Information Metrics:")
    print("• Information Density: Amount of information per chunk")
    print("• Content Distribution: Types of content covered")
    print("• Query Potential: Ability to answer questions")
    
    # Final Rankings
    print("\n🏆 Final Rankings by Category:")
    print("-" * 40)
    
    categories = {
        'Overall Quality': 'overall_quality',
        'Context Quality': 'context_quality',
        'Semantic Quality': 'semantic_quality',
        'Structural Quality': 'structural_quality',
        'Number of Contexts': 'effective_contexts',
        'Information Density': 'information_density',
        'Query Potential': 'query_potential'
    }
    
    for category, metric in categories.items():
        print(f"\n📊 {category}:")
        sorted_data = sorted(metrics_summary, key=lambda x: x[metric], reverse=True)
        for i, data in enumerate(sorted_data):
            value = data[metric]
            if metric == 'effective_contexts':
                print(f"   {i+1}. {data['name']}: {value:.1f}")
            else:
                print(f"   {i+1}. {data['name']}: {value:.3f}")

if __name__ == "__main__":
    generate_comprehensive_summary() 