#!/usr/bin/env python3
"""
LLM Chunking Test Runner
========================

Runs content-aware and hierarchical chunking tests and saves results to test-data directory.

Author: Data Engineering Team
Purpose: Test LLM-based chunking strategies
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm-chunking'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_documents():
    """Create test documents for chunking"""
    documents = {
        "technical_report": """
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

Technical Implementation Details
The system architecture consists of three main components:
1. Document Preprocessing Module
2. Chunking Strategy Engine
3. Evaluation and Optimization Framework

Each component is designed for scalability and extensibility, allowing for easy integration of new chunking strategies and evaluation metrics.

Performance Considerations
Processing time varies significantly between strategies:
- Traditional fixed-size: 0.5 seconds per document
- Semantic chunking: 2.3 seconds per document
- Content-aware chunking: 3.1 seconds per document
- Hierarchical chunking: 4.2 seconds per document

Future Work
Planned improvements include:
- Multi-modal chunking for documents with images and tables
- Domain-specific optimization for technical and legal documents
- Real-time chunking for streaming document processing
- Advanced evaluation metrics for RAG-specific performance

Conclusion
Advanced chunking strategies provide substantial improvements over traditional approaches. The trade-off between processing time and retrieval accuracy should be carefully considered based on specific use case requirements.
        """,
        
        "executive_summary": """
Executive Summary: Document Processing Optimization Initiative

Business Context
Our organization processes thousands of documents daily, requiring efficient and accurate information retrieval systems. Current approaches result in suboptimal performance and user dissatisfaction.

Strategic Objectives
Primary goals include:
- Improve document retrieval accuracy by 25%
- Reduce processing time by 40%
- Enhance user satisfaction scores
- Support multiple document types and formats

Implementation Strategy
Phase 1: Foundation (Months 1-3)
- Deploy semantic chunking capabilities
- Establish baseline performance metrics
- Train staff on new systems

Phase 2: Optimization (Months 4-6)
- Implement content-aware chunking
- Fine-tune parameters based on usage patterns
- Integrate with existing document management systems

Phase 3: Advanced Features (Months 7-12)
- Deploy hierarchical chunking
- Add multi-modal processing capabilities
- Implement real-time optimization

Expected Outcomes
- 25% improvement in retrieval accuracy
- 40% reduction in processing time
- 30% increase in user satisfaction
- Support for 5 additional document types

Risk Assessment
Primary risks include:
- Technical implementation challenges
- User adoption resistance
- Performance degradation during transition
- Integration complexity with existing systems

Mitigation strategies include comprehensive testing, user training programs, and phased deployment approach.

Resource Requirements
- Technical team: 4 developers, 2 data scientists
- Infrastructure: Cloud computing resources
- Training: User education and support materials
- Timeline: 12-month implementation period

Success Metrics
Key performance indicators include:
- Document retrieval accuracy
- Processing time per document
- User satisfaction scores
- System uptime and reliability
- Cost per document processed

Conclusion
This initiative represents a significant investment in our document processing capabilities. The expected benefits justify the required resources and implementation timeline.
        """,
        
        "research_paper": """
Research Paper: Advanced Document Chunking Strategies for Information Retrieval

Abstract
This paper presents novel approaches to document chunking that leverage machine learning and natural language processing techniques. We evaluate the effectiveness of semantic, content-aware, and hierarchical chunking strategies compared to traditional fixed-size approaches.

Introduction
Information retrieval systems rely heavily on effective document segmentation. Traditional chunking methods use fixed-size windows, which often break semantic units and reduce retrieval effectiveness. Recent advances in natural language processing enable more intelligent chunking strategies.

Related Work
Previous research has explored various chunking approaches:
- Fixed-size chunking (Lewis et al., 2020)
- Sentence boundary detection (Smith & Johnson, 2021)
- Semantic similarity clustering (Brown et al., 2022)
- Hierarchical document structure analysis (Davis & Wilson, 2023)

Methodology
Our experimental setup includes:
- Dataset: 10,000 documents from various domains
- Evaluation metrics: Precision, Recall, F1-score, NDCG
- Baseline: Traditional fixed-size chunking
- Test strategies: Semantic, content-aware, hierarchical chunking

Experimental Design
We conducted controlled experiments with the following parameters:
- Document types: Technical reports, legal documents, news articles
- Chunk sizes: 100-500 words
- Overlap ratios: 0-50%
- Evaluation queries: Domain-specific and general queries

Results
Our experiments show significant improvements:
- Semantic chunking: 23.4% improvement in precision
- Content-aware chunking: 18.7% improvement in recall
- Hierarchical chunking: 31.2% improvement in F1-score
- Combined approach: 28.9% improvement in NDCG

Analysis
The improvements are attributed to:
- Better preservation of semantic units
- Improved context retention
- More appropriate chunk boundaries
- Enhanced topic coherence

Discussion
While advanced chunking strategies show clear benefits, they also introduce computational overhead. The trade-off between accuracy and efficiency must be carefully considered for practical applications.

Limitations
Current limitations include:
- Computational complexity
- Domain-specific performance variations
- Limited evaluation on multi-modal documents
- Scalability concerns for large document collections

Future Work
Planned research directions include:
- Multi-modal chunking strategies
- Real-time optimization algorithms
- Domain adaptation techniques
- Scalable implementation approaches

Conclusion
Advanced chunking strategies provide substantial improvements over traditional approaches. Future work should focus on addressing computational complexity and scalability concerns.

References
[1] Lewis, M., et al. (2020). "Fixed-size chunking for information retrieval"
[2] Smith, A., & Johnson, B. (2021). "Sentence boundary detection in documents"
[3] Brown, C., et al. (2022). "Semantic similarity clustering for chunking"
[4] Davis, E., & Wilson, F. (2023). "Hierarchical document structure analysis"
        """
    }
    
    return documents

def run_content_aware_chunking(documents: Dict[str, str], output_dir: Path):
    """Run content-aware chunking tests"""
    logger.info("Running content-aware chunking tests...")
    
    try:
        from llm_chunking.content_aware_chunker import ContentAwareChunker
        
        chunker = ContentAwareChunker()
        results = {}
        
        for doc_name, doc_content in documents.items():
            logger.info(f"Processing {doc_name} with content-aware chunking...")
            
            try:
                chunks = chunker.chunk_text(
                    text=doc_content,
                    target_chunk_size=300,
                    overlap_size=50,
                    boundary_types=['section', 'paragraph', 'topic']
                )
                
                # Save results
                output_file = output_dir / f"{doc_name}_content_aware_chunks.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks, f, indent=2, ensure_ascii=False)
                
                results[doc_name] = {
                    "chunks_file": str(output_file),
                    "num_chunks": len(chunks),
                    "success": True
                }
                
                logger.info(f"✅ {doc_name}: {len(chunks)} chunks saved to {output_file}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {doc_name}: {e}")
                results[doc_name] = {
                    "error": str(e),
                    "success": False
                }
        
        return results
        
    except ImportError as e:
        logger.error(f"❌ Could not import ContentAwareChunker: {e}")
        return {"error": f"Import error: {e}"}

def run_hierarchical_chunking(documents: Dict[str, str], output_dir: Path):
    """Run hierarchical chunking tests"""
    logger.info("Running hierarchical chunking tests...")
    
    try:
        from llm_chunking.hierarchical_chunker import HierarchicalChunker
        
        chunker = HierarchicalChunker()
        results = {}
        
        for doc_name, doc_content in documents.items():
            logger.info(f"Processing {doc_name} with hierarchical chunking...")
            
            try:
                hierarchy = chunker.chunk_text(
                    text=doc_content,
                    max_levels=4,
                    min_chunk_size=50,
                    max_chunk_size=500
                )
                
                # Save full hierarchy
                hierarchy_file = output_dir / f"{doc_name}_hierarchical_full.json"
                with open(hierarchy_file, 'w', encoding='utf-8') as f:
                    json.dump(hierarchy, f, indent=2, ensure_ascii=False)
                
                # Get flat chunks for comparison
                flat_chunks = chunker.get_flat_chunks(hierarchy)
                flat_file = output_dir / f"{doc_name}_hierarchical_flat.json"
                with open(flat_file, 'w', encoding='utf-8') as f:
                    json.dump(flat_chunks, f, indent=2, ensure_ascii=False)
                
                results[doc_name] = {
                    "hierarchy_file": str(hierarchy_file),
                    "flat_file": str(flat_file),
                    "num_chunks": len(flat_chunks),
                    "total_hierarchy_chunks": hierarchy['metadata']['total_chunks'],
                    "success": True
                }
                
                logger.info(f"✅ {doc_name}: {len(flat_chunks)} flat chunks, {hierarchy['metadata']['total_chunks']} total hierarchy chunks")
                
            except Exception as e:
                logger.error(f"❌ Error processing {doc_name}: {e}")
                results[doc_name] = {
                    "error": str(e),
                    "success": False
                }
        
        return results
        
    except ImportError as e:
        logger.error(f"❌ Could not import HierarchicalChunker: {e}")
        return {"error": f"Import error: {e}"}

def main():
    """Main function to run LLM chunking tests"""
    print("🚀 Starting LLM Chunking Tests...")
    print("📊 Testing content-aware and hierarchical chunking strategies")
    print("📁 Results will be saved to test-data/ directory")
    print("-" * 60)
    
    # Create output directory
    output_dir = Path("test-data")
    output_dir.mkdir(exist_ok=True)
    
    # Create test documents
    documents = create_test_documents()
    
    # Save test documents
    for doc_name, doc_content in documents.items():
        doc_file = output_dir / f"{doc_name}_input.txt"
        with open(doc_file, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        logger.info(f"📄 Saved test document: {doc_file}")
    
    # Run content-aware chunking
    print("\n🔍 Running Content-Aware Chunking...")
    content_aware_results = run_content_aware_chunking(documents, output_dir)
    
    # Run hierarchical chunking
    print("\n🏗️  Running Hierarchical Chunking...")
    hierarchical_results = run_hierarchical_chunking(documents, output_dir)
    
    # Generate summary report
    print("\n📋 Generating Summary Report...")
    summary = {
        "test_metadata": {
            "timestamp": "2024-01-01 12:00:00",
            "total_documents": len(documents),
            "strategies_tested": ["content_aware", "hierarchical"]
        },
        "content_aware_results": content_aware_results,
        "hierarchical_results": hierarchical_results
    }
    
    summary_file = output_dir / "llm_chunking_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n✅ LLM Chunking Tests Completed!")
    print(f"📂 Results saved to: {output_dir}")
    print("\n📋 Generated files:")
    for file in output_dir.glob("*"):
        print(f"   - {file.name}")
    
    print("\n🎯 Next steps:")
    print("   1. Review individual chunk files")
    print("   2. Run comparison tests: python run_chunking_comparison.py")
    print("   3. Launch dashboard: python run_dashboard.py")

if __name__ == "__main__":
    main() 