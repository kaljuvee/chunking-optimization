#!/usr/bin/env python3
"""
Sample Data Generator for Streamlit Dashboard
============================================

Generates sample chunking results for testing the Streamlit dashboard.
Creates realistic JSON files with different chunking strategies.

Author: Data Engineering Team
Purpose: Generate test data for dashboard development
"""

import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any

def generate_sample_text(num_words: int = 100) -> str:
    """Generate sample text with specified word count"""
    sample_words = [
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "up", "about", "into", "through", "during", "before",
        "after", "above", "below", "between", "among", "within", "without",
        "against", "toward", "towards", "upon", "across", "behind", "beneath",
        "beside", "beyond", "inside", "outside", "under", "over", "around",
        "data", "analysis", "machine", "learning", "artificial", "intelligence",
        "semantic", "chunking", "document", "processing", "text", "extraction",
        "natural", "language", "processing", "retrieval", "augmented", "generation",
        "vector", "embeddings", "similarity", "clustering", "classification",
        "regression", "neural", "networks", "deep", "learning", "supervised",
        "unsupervised", "reinforcement", "learning", "algorithm", "model",
        "training", "testing", "validation", "accuracy", "precision", "recall",
        "f1", "score", "performance", "optimization", "hyperparameter", "tuning"
    ]
    
    # Generate sentences
    sentences = []
    current_length = 0
    
    while current_length < num_words:
        sentence_length = random.randint(5, 20)
        sentence_words = random.choices(sample_words, k=sentence_length)
        sentence = " ".join(sentence_words).capitalize() + "."
        sentences.append(sentence)
        current_length += sentence_length
    
    return " ".join(sentences)

def generate_chunk_data(strategy_name: str, num_chunks: int = 50) -> List[Dict[str, Any]]:
    """Generate sample chunk data for a specific strategy"""
    chunks = []
    
    for i in range(num_chunks):
        # Vary chunk sizes based on strategy
        if strategy_name == "semantic":
            chunk_size = random.randint(80, 200)
        elif strategy_name == "fixed_size":
            chunk_size = random.randint(100, 150)
        elif strategy_name == "overlap":
            chunk_size = random.randint(120, 180)
        else:
            chunk_size = random.randint(50, 250)
        
        chunk_text = generate_sample_text(chunk_size)
        
        chunk = {
            "chunk_index": i,
            "text": chunk_text,
            "strategy": strategy_name,
            "tokens": len(chunk_text.split()),
            "metadata": {
                "chunk_size": chunk_size,
                "word_count": len(chunk_text.split()),
                "char_count": len(chunk_text),
                "strategy_type": strategy_name,
                "generated": True
            }
        }
        
        # Add strategy-specific metadata
        if strategy_name == "semantic":
            chunk["metadata"]["semantic_score"] = random.uniform(0.7, 0.95)
            chunk["metadata"]["topic_coherence"] = random.uniform(0.6, 0.9)
        elif strategy_name == "overlap":
            chunk["metadata"]["overlap_tokens"] = random.randint(10, 50)
            chunk["metadata"]["overlap_percentage"] = random.uniform(0.1, 0.3)
        
        chunks.append(chunk)
    
    return chunks

def generate_strategy_comparison() -> Dict[str, List[Dict[str, Any]]]:
    """Generate comparison data for multiple strategies"""
    strategies = {
        "semantic": generate_chunk_data("semantic", 60),
        "fixed_size": generate_chunk_data("fixed_size", 75),
        "overlap": generate_chunk_data("overlap", 65),
        "hierarchical": generate_chunk_data("hierarchical", 55),
        "content_aware": generate_chunk_data("content_aware", 70)
    }
    
    return strategies

def save_sample_data(output_dir: str = "sample_data"):
    """Save sample data to JSON files"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data for each strategy
    strategies = generate_strategy_comparison()
    
    # Save individual strategy files
    for strategy_name, chunks in strategies.items():
        filename = f"{strategy_name}_chunks.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Generated {filename} with {len(chunks)} chunks")
    
    # Save combined file
    combined_filename = "all_strategies.json"
    combined_filepath = os.path.join(output_dir, combined_filename)
    
    all_chunks = []
    for strategy_name, chunks in strategies.items():
        all_chunks.extend(chunks)
    
    with open(combined_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {combined_filename} with {len(all_chunks)} total chunks")
    
    # Create a summary file
    summary = {
        "total_strategies": len(strategies),
        "total_chunks": len(all_chunks),
        "strategies": {
            name: {
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(len(chunk["text"].split()) for chunk in chunks) / len(chunks),
                "file": f"{name}_chunks.json"
            }
            for name, chunks in strategies.items()
        }
    }
    
    summary_filename = "data_summary.json"
    summary_filepath = os.path.join(output_dir, summary_filename)
    
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated {summary_filename} with data summary")
    
    return output_dir

def main():
    """Main function to generate sample data"""
    print("🚀 Generating sample data for Streamlit dashboard...")
    print("📁 Creating sample chunking results...")
    
    output_dir = save_sample_data()
    
    print(f"\n✅ Sample data generated successfully!")
    print(f"📂 Output directory: {output_dir}")
    print(f"\n📊 To test the dashboard:")
    print(f"   1. Run: python run_dashboard.py")
    print(f"   2. Upload files from: {output_dir}")
    print(f"   3. Or specify directory: {output_dir}")
    
    # List generated files
    print(f"\n📋 Generated files:")
    for file in Path(output_dir).glob("*.json"):
        print(f"   - {file.name}")

if __name__ == "__main__":
    main() 