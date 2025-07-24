#!/usr/bin/env python3
"""
Visualize Hierarchical Chunking Results
=======================================

Script to visualize hierarchical chunking results in a tree-like structure.
"""

import json
import sys
from pathlib import Path

def visualize_hierarchy(hierarchy_data, max_text_length=80):
    """Visualize hierarchical chunks in a tree structure"""
    
    print("🌳 Hierarchical Chunking Structure")
    print("=" * 50)
    
    # Get document info
    doc_id = hierarchy_data['document_id']
    total_chunks = hierarchy_data['metadata']['total_chunks']
    levels = hierarchy_data['metadata']['levels']
    doc_length = hierarchy_data['metadata']['document_length']
    
    print(f"Document ID: {doc_id}")
    print(f"Total Chunks: {total_chunks}")
    print(f"Max Levels: {levels}")
    print(f"Document Length: {doc_length} characters")
    print()
    
    # Build tree structure
    chunks = hierarchy_data['chunks']
    
    # Find root chunk
    root_chunk = None
    for chunk_id, chunk_data in chunks.items():
        if chunk_data['parent_id'] is None:
            root_chunk = chunk_data
            break
    
    if root_chunk:
        print_tree_node(root_chunk, chunks, "", max_text_length)
    
    # Show statistics by level
    print("\n📊 Statistics by Level:")
    print("-" * 30)
    
    level_stats = {}
    for chunk_data in chunks.values():
        level = chunk_data['level']
        if level not in level_stats:
            level_stats[level] = {
                'count': 0,
                'total_words': 0,
                'types': set()
            }
        
        level_stats[level]['count'] += 1
        level_stats[level]['total_words'] += chunk_data['metadata'].get('word_count', 0)
        level_stats[level]['types'].add(chunk_data['metadata'].get('type', 'unknown'))
    
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        avg_words = stats['total_words'] / stats['count'] if stats['count'] > 0 else 0
        types_str = ', '.join(sorted(stats['types']))
        print(f"Level {level}: {stats['count']} chunks, {avg_words:.1f} avg words, types: {types_str}")

def print_tree_node(chunk_data, all_chunks, prefix, max_text_length):
    """Print a tree node and its children recursively"""
    
    level = chunk_data['level']
    chunk_id = chunk_data['chunk_id']
    text = chunk_data['text'].strip()
    word_count = chunk_data['metadata'].get('word_count', len(text.split()))
    chunk_type = chunk_data['metadata'].get('type', 'unknown')
    title = chunk_data['metadata'].get('title', '')
    
    # Create display text
    if title and title != text[:len(title)]:
        display_text = title
    else:
        display_text = text[:max_text_length] + "..." if len(text) > max_text_length else text
    
    # Print node
    level_indicator = "📄" if level == 0 else "📝" if level == 1 else "📋" if level == 2 else "📄" if level == 3 else "🔹"
    print(f"{prefix}{level_indicator} Level {level} ({chunk_type}): {display_text}")
    print(f"{prefix}    ID: {chunk_id}, Words: {word_count}")
    
    # Print children
    children = chunk_data.get('children_ids', [])
    for i, child_id in enumerate(children):
        if child_id in all_chunks:
            child_data = all_chunks[child_id]
            is_last = (i == len(children) - 1)
            child_prefix = prefix + ("    " if is_last else "│   ")
            print_tree_node(child_data, all_chunks, child_prefix, max_text_length)

def show_context_paths(hierarchy_data):
    """Show context paths for all chunks"""
    
    print("\n🔗 Context Paths:")
    print("-" * 30)
    
    chunks = hierarchy_data['chunks']
    
    for chunk_id, chunk_data in chunks.items():
        if chunk_data['level'] > 0:  # Skip root
            path = get_context_path(chunk_data, chunks)
            path_text = " → ".join([f"L{chunk['level']}:{chunk['metadata'].get('title', chunk['text'][:20])}" for chunk in path])
            print(f"{chunk_id}: {path_text}")

def get_context_path(chunk_data, all_chunks):
    """Get context path from root to chunk"""
    path = []
    current_id = chunk_data['chunk_id']
    
    while current_id and current_id in all_chunks:
        chunk = all_chunks[current_id]
        path.insert(0, chunk)
        current_id = chunk['parent_id']
    
    return path

def show_flat_chunks_by_level(hierarchy_data):
    """Show flat chunks organized by level"""
    
    print("\n📋 Flat Chunks by Level:")
    print("-" * 30)
    
    chunks = hierarchy_data['chunks']
    
    for level in range(5):  # Assuming max 5 levels
        level_chunks = [chunk for chunk in chunks.values() if chunk['level'] == level]
        
        if level_chunks:
            print(f"\nLevel {level} ({len(level_chunks)} chunks):")
            for i, chunk in enumerate(level_chunks):
                title = chunk['metadata'].get('title', chunk['text'][:50])
                word_count = chunk['metadata'].get('word_count', len(chunk['text'].split()))
                print(f"  {i+1}. {title} ({word_count} words)")

def main():
    """Main function"""
    
    # Try to load the simple hierarchical chunks
    simple_file = Path("../test-data/hierarchical/simple_hierarchical_chunks.json")
    llm_file = Path("../test-data/hierarchical/hierarchical_chunks.json")
    
    if simple_file.exists():
        print("📁 Loading simple hierarchical chunks...")
        with open(simple_file, 'r', encoding='utf-8') as f:
            hierarchy_data = json.load(f)
        visualize_hierarchy(hierarchy_data)
        show_context_paths(hierarchy_data)
        show_flat_chunks_by_level(hierarchy_data)
        
    elif llm_file.exists():
        print("📁 Loading LLM hierarchical chunks...")
        with open(llm_file, 'r', encoding='utf-8') as f:
            hierarchy_data = json.load(f)
        visualize_hierarchy(hierarchy_data)
        show_context_paths(hierarchy_data)
        show_flat_chunks_by_level(hierarchy_data)
        
    else:
        print("❌ No hierarchical chunk files found!")
        print("Please run the hierarchical chunker first:")
        print("  python simple_hierarchical_chunker.py")

if __name__ == "__main__":
    main() 