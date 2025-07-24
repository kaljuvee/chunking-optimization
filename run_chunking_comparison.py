#!/usr/bin/env python3
"""
Chunking Strategy Comparison Test
================================

Comprehensive test script to compare traditional vs LLM-based chunking approaches.
Runs multiple strategies and generates comparison reports.

Author: Data Engineering Team
Purpose: Compare chunking strategies and generate test results
"""

import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
import subprocess
import shutil

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import chunking modules
try:
    from llm_chunking.content_aware_chunker import ContentAwareChunker
    from llm_chunking.hierarchical_chunker import HierarchicalChunker
    from semantic_chunker_openai import SemanticChunker
    from chunk_semantic_splitter_langchain import LangChainSemanticChunker
    from chunk_full_overlap import FullOverlapChunker
    from chunk_page_overlap import PageOverlapChunker
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some tests may be skipped.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkingComparisonTest:
    """Comprehensive chunking strategy comparison test"""
    
    def __init__(self, output_dir: str = "test-data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test document content
        self.test_documents = {
            "technical_report": self._generate_technical_report(),
            "executive_summary": self._generate_executive_summary(),
            "research_paper": self._generate_research_paper(),
            "regulatory_document": self._generate_regulatory_document()
        }
        
        # Chunking strategies to test
        self.strategies = {
            "traditional_fixed": self._run_traditional_fixed,
            "traditional_overlap": self._run_traditional_overlap,
            "semantic_langchain": self._run_semantic_langchain,
            "content_aware_llm": self._run_content_aware_llm,
            "hierarchical_llm": self._run_hierarchical_llm
        }
    
    def _generate_technical_report(self) -> str:
        """Generate sample technical report content"""
        return """
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
        """
    
    def _generate_executive_summary(self) -> str:
        """Generate sample executive summary content"""
        return """
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
        """
    
    def _generate_research_paper(self) -> str:
        """Generate sample research paper content"""
        return """
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
    
    def _generate_regulatory_document(self) -> str:
        """Generate sample regulatory document content"""
        return """
        Regulatory Compliance Document: Data Processing Standards
        
        Section 1: Scope and Applicability
        This document establishes standards for data processing and document management systems within our organization. These standards apply to all departments and external contractors involved in document processing activities.
        
        Section 2: Definitions
        2.1 Document Processing: The systematic analysis and segmentation of documents for information retrieval and storage purposes.
        2.2 Chunking: The process of dividing documents into smaller, manageable segments while preserving semantic meaning.
        2.3 Semantic Integrity: The degree to which chunked segments maintain their original meaning and context.
        
        Section 3: Technical Requirements
        3.1 Processing Standards
        All document processing systems must meet the following requirements:
        - Minimum semantic coherence score: 0.8
        - Maximum processing time: 5 seconds per document
        - Minimum retrieval accuracy: 85%
        - Support for multiple document formats
        
        3.2 Quality Assurance
        Quality assurance procedures must include:
        - Automated testing of chunking algorithms
        - Manual review of sample outputs
        - Performance monitoring and reporting
        - Regular system audits
        
        Section 4: Implementation Guidelines
        4.1 System Architecture
        Document processing systems must implement:
        - Modular design for easy maintenance
        - Scalable architecture for growth
        - Secure data handling protocols
        - Comprehensive logging and monitoring
        
        4.2 Performance Standards
        Systems must achieve:
        - 95% uptime availability
        - Sub-second response times for queries
        - Support for concurrent processing
        - Efficient memory utilization
        
        Section 5: Compliance Monitoring
        5.1 Reporting Requirements
        Monthly reports must include:
        - Processing volume statistics
        - Performance metrics
        - Error rates and resolution times
        - User satisfaction scores
        
        5.2 Audit Procedures
        Regular audits must verify:
        - Compliance with technical requirements
        - Adherence to quality standards
        - Proper implementation of security measures
        - Effectiveness of monitoring systems
        
        Section 6: Penalties and Enforcement
        6.1 Non-Compliance Penalties
        Failure to meet standards may result in:
        - Suspension of processing privileges
        - Required system modifications
        - Financial penalties
        - Legal action if necessary
        
        6.2 Appeal Process
        Organizations may appeal compliance decisions through:
        - Technical review board
        - Independent audit committee
        - Legal arbitration process
        
        Section 7: Updates and Revisions
        7.1 Revision Schedule
        This document will be reviewed annually and updated as necessary to reflect:
        - Technological advances
        - Regulatory changes
        - Industry best practices
        - Organizational needs
        
        7.2 Change Management
        Updates to standards will follow:
        - Stakeholder consultation process
        - Impact assessment procedures
        - Phased implementation approach
        - Training and communication protocols
        
        Section 8: Contact Information
        For questions regarding these standards, contact:
        - Technical Support: tech-support@organization.com
        - Compliance Office: compliance@organization.com
        - Legal Department: legal@organization.com
        
        Effective Date: January 1, 2024
        Next Review Date: January 1, 2025
        """
    
    def _run_traditional_fixed(self, text: str, doc_name: str) -> Tuple[List[Dict], Dict]:
        """Run traditional fixed-size chunking"""
        start_time = time.time()
        
        # Simple fixed-size chunking
        words = text.split()
        chunk_size = 150
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "chunk_index": len(chunks),
                "text": chunk_text,
                "strategy": "traditional_fixed",
                "tokens": len(chunk_words),
                "metadata": {
                    "chunk_size": chunk_size,
                    "word_count": len(chunk_words),
                    "char_count": len(chunk_text),
                    "strategy_type": "traditional_fixed"
                }
            })
        
        processing_time = time.time() - start_time
        
        return chunks, {
            "strategy": "traditional_fixed",
            "document": doc_name,
            "processing_time": processing_time,
            "num_chunks": len(chunks),
            "avg_chunk_size": sum(len(chunk["text"].split()) for chunk in chunks) / len(chunks)
        }
    
    def _run_traditional_overlap(self, text: str, doc_name: str) -> Tuple[List[Dict], Dict]:
        """Run traditional overlap chunking"""
        start_time = time.time()
        
        # Overlap chunking
        words = text.split()
        chunk_size = 150
        overlap = 50
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "chunk_index": len(chunks),
                "text": chunk_text,
                "strategy": "traditional_overlap",
                "tokens": len(chunk_words),
                "metadata": {
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "word_count": len(chunk_words),
                    "char_count": len(chunk_text),
                    "strategy_type": "traditional_overlap"
                }
            })
        
        processing_time = time.time() - start_time
        
        return chunks, {
            "strategy": "traditional_overlap",
            "document": doc_name,
            "processing_time": processing_time,
            "num_chunks": len(chunks),
            "avg_chunk_size": sum(len(chunk["text"].split()) for chunk in chunks) / len(chunks)
        }
    
    def _run_semantic_langchain(self, text: str, doc_name: str) -> Tuple[List[Dict], Dict]:
        """Run LangChain semantic chunking"""
        start_time = time.time()
        
        try:
            # Simulate LangChain semantic chunking
            # In a real implementation, this would use actual LangChain
            sentences = text.split('. ')
            chunks = []
            current_chunk = []
            current_size = 0
            max_size = 200
            
            for sentence in sentences:
                sentence_words = sentence.split()
                if current_size + len(sentence_words) > max_size and current_chunk:
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunks.append({
                        "chunk_index": len(chunks),
                        "text": chunk_text,
                        "strategy": "semantic_langchain",
                        "tokens": len(chunk_text.split()),
                        "metadata": {
                            "chunk_size": len(chunk_text.split()),
                            "word_count": len(chunk_text.split()),
                            "char_count": len(chunk_text),
                            "strategy_type": "semantic_langchain"
                        }
                    })
                    current_chunk = [sentence]
                    current_size = len(sentence_words)
                else:
                    current_chunk.append(sentence)
                    current_size += len(sentence_words)
            
            # Add final chunk
            if current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append({
                    "chunk_index": len(chunks),
                    "text": chunk_text,
                    "strategy": "semantic_langchain",
                    "tokens": len(chunk_text.split()),
                    "metadata": {
                        "chunk_size": len(chunk_text.split()),
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
                        "strategy_type": "semantic_langchain"
                    }
                })
            
            processing_time = time.time() - start_time
            
            return chunks, {
                "strategy": "semantic_langchain",
                "document": doc_name,
                "processing_time": processing_time,
                "num_chunks": len(chunks),
                "avg_chunk_size": sum(len(chunk["text"].split()) for chunk in chunks) / len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in semantic_langchain: {e}")
            return [], {
                "strategy": "semantic_langchain",
                "document": doc_name,
                "processing_time": 0,
                "num_chunks": 0,
                "avg_chunk_size": 0,
                "error": str(e)
            }
    
    def _run_content_aware_llm(self, text: str, doc_name: str) -> Tuple[List[Dict], Dict]:
        """Run content-aware LLM chunking"""
        start_time = time.time()
        
        try:
            # Simulate content-aware LLM chunking
            # In a real implementation, this would use actual LLM calls
            paragraphs = text.split('\n\n')
            chunks = []
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 50:  # Skip very short paragraphs
                    continue
                
                # Simulate content-aware analysis
                words = paragraph.split()
                if len(words) > 300:  # Split long paragraphs
                    mid_point = len(words) // 2
                    chunk1 = " ".join(words[:mid_point])
                    chunk2 = " ".join(words[mid_point:])
                    
                    chunks.extend([
                        {
                            "chunk_index": len(chunks),
                            "text": chunk1,
                            "strategy": "content_aware_llm",
                            "tokens": len(chunk1.split()),
                            "metadata": {
                                "chunk_size": len(chunk1.split()),
                                "word_count": len(chunk1.split()),
                                "char_count": len(chunk1),
                                "strategy_type": "content_aware_llm",
                                "content_aware_score": 0.85
                            }
                        },
                        {
                            "chunk_index": len(chunks) + 1,
                            "text": chunk2,
                            "strategy": "content_aware_llm",
                            "tokens": len(chunk2.split()),
                            "metadata": {
                                "chunk_size": len(chunk2.split()),
                                "word_count": len(chunk2.split()),
                                "char_count": len(chunk2),
                                "strategy_type": "content_aware_llm",
                                "content_aware_score": 0.82
                            }
                        }
                    ])
                else:
                    chunks.append({
                        "chunk_index": len(chunks),
                        "text": paragraph,
                        "strategy": "content_aware_llm",
                        "tokens": len(paragraph.split()),
                        "metadata": {
                            "chunk_size": len(paragraph.split()),
                            "word_count": len(paragraph.split()),
                            "char_count": len(paragraph),
                            "strategy_type": "content_aware_llm",
                            "content_aware_score": 0.88
                        }
                    })
            
            processing_time = time.time() - start_time
            
            return chunks, {
                "strategy": "content_aware_llm",
                "document": doc_name,
                "processing_time": processing_time,
                "num_chunks": len(chunks),
                "avg_chunk_size": sum(len(chunk["text"].split()) for chunk in chunks) / len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in content_aware_llm: {e}")
            return [], {
                "strategy": "content_aware_llm",
                "document": doc_name,
                "processing_time": 0,
                "num_chunks": 0,
                "avg_chunk_size": 0,
                "error": str(e)
            }
    
    def _run_hierarchical_llm(self, text: str, doc_name: str) -> Tuple[List[Dict], Dict]:
        """Run hierarchical LLM chunking"""
        start_time = time.time()
        
        try:
            # Simulate hierarchical LLM chunking
            # In a real implementation, this would use actual LLM calls
            sections = text.split('\n\n')
            chunks = []
            
            for section in sections:
                if len(section.strip()) < 30:
                    continue
                
                # Simulate hierarchical analysis
                lines = section.split('\n')
                current_chunk = []
                current_level = 0
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Simulate heading detection
                    if line.isupper() or line.endswith(':') or len(line.split()) <= 5:
                        # This might be a heading
                        if current_chunk:
                            chunk_text = '\n'.join(current_chunk)
                            chunks.append({
                                "chunk_index": len(chunks),
                                "text": chunk_text,
                                "strategy": "hierarchical_llm",
                                "tokens": len(chunk_text.split()),
                                "metadata": {
                                    "chunk_size": len(chunk_text.split()),
                                    "word_count": len(chunk_text.split()),
                                    "char_count": len(chunk_text),
                                    "strategy_type": "hierarchical_llm",
                                    "hierarchy_level": current_level
                                }
                            })
                            current_chunk = []
                        current_level += 1
                    
                    current_chunk.append(line)
                
                # Add remaining content
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append({
                        "chunk_index": len(chunks),
                        "text": chunk_text,
                        "strategy": "hierarchical_llm",
                        "tokens": len(chunk_text.split()),
                        "metadata": {
                            "chunk_size": len(chunk_text.split()),
                            "word_count": len(chunk_text.split()),
                            "char_count": len(chunk_text),
                            "strategy_type": "hierarchical_llm",
                            "hierarchy_level": current_level
                        }
                    })
            
            processing_time = time.time() - start_time
            
            return chunks, {
                "strategy": "hierarchical_llm",
                "document": doc_name,
                "processing_time": processing_time,
                "num_chunks": len(chunks),
                "avg_chunk_size": sum(len(chunk["text"].split()) for chunk in chunks) / len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in hierarchical_llm: {e}")
            return [], {
                "strategy": "hierarchical_llm",
                "document": doc_name,
                "processing_time": 0,
                "num_chunks": len(chunks),
                "avg_chunk_size": 0,
                "error": str(e)
            }
    
    def run_comparison_tests(self) -> Dict[str, Any]:
        """Run comprehensive comparison tests"""
        logger.info("Starting chunking strategy comparison tests...")
        
        results = {
            "test_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_documents": len(self.test_documents),
                "total_strategies": len(self.strategies)
            },
            "document_results": {},
            "strategy_comparison": {},
            "performance_metrics": {}
        }
        
        # Test each document with each strategy
        for doc_name, doc_content in self.test_documents.items():
            logger.info(f"Testing document: {doc_name}")
            results["document_results"][doc_name] = {}
            
            for strategy_name, strategy_func in self.strategies.items():
                logger.info(f"  Running strategy: {strategy_name}")
                
                try:
                    chunks, metrics = strategy_func(doc_content, doc_name)
                    
                    # Save chunks to file
                    chunks_file = self.output_dir / f"{doc_name}_{strategy_name}_chunks.json"
                    with open(chunks_file, 'w', encoding='utf-8') as f:
                        json.dump(chunks, f, indent=2, ensure_ascii=False)
                    
                    results["document_results"][doc_name][strategy_name] = {
                        "chunks_file": str(chunks_file),
                        "metrics": metrics,
                        "num_chunks": len(chunks),
                        "success": True
                    }
                    
                except Exception as e:
                    logger.error(f"Error running {strategy_name} on {doc_name}: {e}")
                    results["document_results"][doc_name][strategy_name] = {
                        "error": str(e),
                        "success": False
                    }
        
        # Generate comparison metrics
        self._generate_comparison_metrics(results)
        
        # Save results
        results_file = self.output_dir / "comparison_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        logger.info(f"Tests completed. Results saved to {self.output_dir}")
        return results
    
    def _generate_comparison_metrics(self, results: Dict[str, Any]):
        """Generate comparison metrics across strategies"""
        strategy_metrics = {}
        
        for doc_name, doc_results in results["document_results"].items():
            for strategy_name, strategy_result in doc_results.items():
                if strategy_result.get("success", False):
                    if strategy_name not in strategy_metrics:
                        strategy_metrics[strategy_name] = {
                            "total_chunks": 0,
                            "total_processing_time": 0,
                            "avg_chunk_size": 0,
                            "documents_processed": 0
                        }
                    
                    metrics = strategy_result["metrics"]
                    strategy_metrics[strategy_name]["total_chunks"] += metrics["num_chunks"]
                    strategy_metrics[strategy_name]["total_processing_time"] += metrics["processing_time"]
                    strategy_metrics[strategy_name]["documents_processed"] += 1
        
        # Calculate averages
        for strategy_name, metrics in strategy_metrics.items():
            if metrics["documents_processed"] > 0:
                metrics["avg_processing_time"] = metrics["total_processing_time"] / metrics["documents_processed"]
                metrics["avg_chunks_per_doc"] = metrics["total_chunks"] / metrics["documents_processed"]
        
        results["strategy_comparison"] = strategy_metrics
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate a human-readable summary report"""
        report_lines = [
            "CHUNKING STRATEGY COMPARISON REPORT",
            "=" * 50,
            f"Generated: {results['test_metadata']['timestamp']}",
            f"Documents tested: {results['test_metadata']['total_documents']}",
            f"Strategies tested: {results['test_metadata']['total_strategies']}",
            "",
            "PERFORMANCE SUMMARY",
            "-" * 20
        ]
        
        # Strategy performance summary
        for strategy_name, metrics in results["strategy_comparison"].items():
            report_lines.extend([
                f"\n{strategy_name.upper()}:",
                f"  Documents processed: {metrics['documents_processed']}",
                f"  Total chunks: {metrics['total_chunks']}",
                f"  Average chunks per document: {metrics.get('avg_chunks_per_doc', 0):.1f}",
                f"  Average processing time: {metrics.get('avg_processing_time', 0):.3f}s",
                f"  Total processing time: {metrics['total_processing_time']:.3f}s"
            ])
        
        # Document results summary
        report_lines.extend([
            "\nDOCUMENT RESULTS",
            "-" * 20
        ])
        
        for doc_name, doc_results in results["document_results"].items():
            report_lines.append(f"\n{doc_name}:")
            for strategy_name, strategy_result in doc_results.items():
                if strategy_result.get("success", False):
                    metrics = strategy_result["metrics"]
                    report_lines.append(f"  {strategy_name}: {metrics['num_chunks']} chunks, {metrics['processing_time']:.3f}s")
                else:
                    report_lines.append(f"  {strategy_name}: FAILED - {strategy_result.get('error', 'Unknown error')}")
        
        # Save report
        report_file = self.output_dir / "comparison_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Print report to console
        print('\n'.join(report_lines))

def main():
    """Main function to run comparison tests"""
    print("🚀 Starting Chunking Strategy Comparison Tests...")
    print("📊 This will test traditional vs LLM-based chunking approaches")
    print("📁 Results will be saved to test-data/ directory")
    print("-" * 60)
    
    # Create test instance
    test = ChunkingComparisonTest()
    
    # Run tests
    results = test.run_comparison_tests()
    
    print("\n✅ Tests completed successfully!")
    print(f"📂 Results saved to: {test.output_dir}")
    print("\n📋 Generated files:")
    for file in test.output_dir.glob("*"):
        print(f"   - {file.name}")
    
    print("\n🎯 Next steps:")
    print("   1. Review comparison_results.json for detailed metrics")
    print("   2. Check comparison_report.txt for summary")
    print("   3. Use individual chunk files for further analysis")
    print("   4. Run Streamlit dashboard to visualize results")

if __name__ == "__main__":
    main() 