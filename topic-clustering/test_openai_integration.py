#!/usr/bin/env python3
"""
Test OpenAI Integration
======================

Simple test to verify OpenAI API integration works correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_chunker import EnhancedChunker
from question_clustering import QuestionClusterer

# Load environment variables
load_dotenv()

def test_openai_integration():
    """Test OpenAI integration with simple examples"""
    
    # Check if OpenAI API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return False
    
    print("✅ OpenAI API key found")
    
    # Test enhanced chunker
    print("\n🧪 Testing Enhanced Chunker...")
    try:
        chunker = EnhancedChunker()
        print("✅ EnhancedChunker initialized successfully")
        
        # Test with small sample text
        sample_text = """
        This is a test document about artificial intelligence and machine learning.
        The document discusses various AI techniques including neural networks, 
        deep learning, and natural language processing. These technologies are 
        transforming how we approach problem solving in modern applications.
        """
        
        chunks = chunker.chunk_text(
            text=sample_text,
            generate_questions=True,
            extract_topics=True
        )
        
        print(f"✅ Generated {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {chunk.word_count} words, {len(chunk.questions)} questions, {len(chunk.topics)} topics")
        
    except Exception as e:
        print(f"❌ EnhancedChunker test failed: {e}")
        return False
    
    # Test question clusterer
    print("\n🧪 Testing Question Clusterer...")
    try:
        clusterer = QuestionClusterer()
        print("✅ QuestionClusterer initialized successfully")
        
        # Test with sample chunks
        sample_chunks = [
            {
                'chunk_id': 'test_001',
                'questions': [
                    'What is artificial intelligence?',
                    'How does machine learning work?',
                    'What are neural networks?'
                ]
            },
            {
                'chunk_id': 'test_002',
                'questions': [
                    'What is deep learning?',
                    'How do you train a neural network?',
                    'What is natural language processing?'
                ]
            }
        ]
        
        analysis = clusterer.cluster_questions(sample_chunks, method='kmeans')
        
        print(f"✅ Clustering completed: {analysis.total_clusters} clusters from {analysis.total_questions} questions")
        
    except Exception as e:
        print(f"❌ QuestionClusterer test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! OpenAI integration is working correctly.")
    return True

if __name__ == "__main__":
    success = test_openai_integration()
    sys.exit(0 if success else 1) 