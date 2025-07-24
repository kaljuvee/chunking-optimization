#!/usr/bin/env python3
"""
Enhanced Analysis Runner
========================

Comprehensive test runner that combines:
1. Enhanced chunking with question generation
2. Question clustering and topic extraction
3. Index analysis and insights generation

Author: Data Engineering Team
Purpose: Complete enhanced analysis pipeline
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_chunker import EnhancedChunker
from question_clustering import QuestionClusterer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAnalysisRunner:
    """Comprehensive enhanced analysis runner"""
    
    def __init__(self, output_dir: str = "test-data/enhanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.chunker = EnhancedChunker()
        self.clusterer = QuestionClusterer()
        
        # Test documents
        self.test_documents = self._create_test_documents()
    
    def _create_test_documents(self) -> Dict[str, str]:
        """Create comprehensive test documents"""
        return {
            "meeting_minutes": """
Board Meeting Minutes - Q4 2024 Strategic Review

Meeting Date: December 15, 2024
Attendees: CEO, CTO, CFO, Head of Engineering, Head of Product

Agenda Items:
1. Q4 Performance Review
2. Technology Infrastructure Assessment
3. Product Development Roadmap
4. Budget Allocation for 2025
5. Risk Assessment and Mitigation

Q4 Performance Review:
The company achieved 85% of Q4 targets, with notable successes in customer acquisition and product adoption. Revenue increased by 23% compared to Q3, driven primarily by enterprise sales. Customer satisfaction scores improved to 4.6/5.0, up from 4.2 in Q3.

Technology Infrastructure Assessment:
Current infrastructure supports 10,000 concurrent users with 99.9% uptime. Planned upgrades include migration to cloud-native architecture and implementation of advanced monitoring systems. Estimated cost: $2.5M over 18 months.

Product Development Roadmap:
Q1 2025: Release of AI-powered analytics dashboard
Q2 2025: Mobile application launch
Q3 2025: Enterprise API platform
Q4 2025: Advanced reporting and visualization tools

Budget Allocation for 2025:
Total budget: $15M
- Product Development: 40% ($6M)
- Marketing and Sales: 25% ($3.75M)
- Infrastructure: 20% ($3M)
- Operations: 10% ($1.5M)
- Research and Development: 5% ($0.75M)

Risk Assessment:
Primary risks identified:
1. Cybersecurity threats - Mitigation: Enhanced security protocols
2. Talent retention - Mitigation: Competitive compensation packages
3. Market competition - Mitigation: Accelerated innovation
4. Regulatory changes - Mitigation: Compliance monitoring system

Action Items:
1. Begin cloud migration planning (Owner: CTO, Due: Jan 15)
2. Finalize Q1 product roadmap (Owner: Head of Product, Due: Jan 10)
3. Implement new security measures (Owner: CTO, Due: Feb 1)
4. Review compensation packages (Owner: CFO, Due: Jan 20)

Next Meeting: January 20, 2025
            """,
            
            "deal_analysis": """
Deal Analysis Report - Enterprise Software License Agreement

Deal ID: DEAL-2024-001
Customer: Global Manufacturing Corp
Deal Value: $2.5M (3-year contract)
Sales Representative: Sarah Johnson
Close Date: December 10, 2024

Customer Profile:
Global Manufacturing Corp is a Fortune 500 company with operations in 15 countries. They manufacture automotive components and have 25,000 employees worldwide. Current technology stack includes legacy ERP systems and basic reporting tools.

Business Requirements:
1. Real-time production monitoring and analytics
2. Predictive maintenance capabilities
3. Supply chain optimization
4. Quality control automation
5. Integration with existing ERP systems
6. Multi-site deployment support
7. Advanced reporting and dashboarding
8. Mobile access for field workers

Technical Requirements:
- Support for 5,000 concurrent users
- 99.9% uptime SLA
- Integration with SAP ERP
- Real-time data processing
- Advanced analytics and machine learning
- Mobile-responsive design
- Multi-language support (English, Spanish, German, Chinese)
- Compliance with ISO 9001 and ISO 14001 standards

Solution Proposed:
Enterprise Software Suite with the following components:
1. Production Monitoring Platform
2. Predictive Analytics Engine
3. Supply Chain Optimization Module
4. Quality Management System
5. Mobile Application Suite
6. Advanced Reporting Dashboard
7. API Gateway for ERP Integration

Implementation Timeline:
Phase 1 (Months 1-3): Core platform deployment
Phase 2 (Months 4-6): Advanced analytics implementation
Phase 3 (Months 7-9): Mobile application rollout
Phase 4 (Months 10-12): Full integration and optimization

Pricing Structure:
- License fees: $1.8M (upfront)
- Implementation services: $400K
- Annual maintenance: $300K
- Total 3-year value: $2.5M

Risk Assessment:
Low risk deal with strong customer commitment and clear requirements. Main risks include implementation timeline and integration complexity.

Success Metrics:
- Customer satisfaction score > 4.5/5.0
- System uptime > 99.9%
- User adoption rate > 80%
- ROI achievement within 18 months

Follow-up Actions:
1. Kick-off meeting scheduled for January 5, 2025
2. Technical requirements gathering session
3. Implementation team assignment
4. Customer training program development
            """,
            
            "technical_specification": """
Technical Specification: Advanced Document Processing System

Project: ADNOC Document Intelligence Platform
Version: 2.0
Date: December 2024

System Overview:
The Advanced Document Processing System is designed to handle large volumes of unstructured documents, extract meaningful information, and provide intelligent search and retrieval capabilities. The system leverages machine learning and natural language processing to deliver high-accuracy results.

Architecture Components:

1. Document Ingestion Layer:
   - Support for PDF, DOCX, TXT, and image formats
   - OCR capabilities for scanned documents
   - Batch processing with priority queuing
   - Real-time document streaming
   - Format validation and error handling

2. Processing Engine:
   - Multi-stage document pipeline
   - Content extraction and normalization
   - Metadata generation and enrichment
   - Quality assessment and validation
   - Parallel processing for scalability

3. Chunking and Indexing:
   - Semantic chunking algorithms
   - Hierarchical document structure analysis
   - Question generation for improved retrieval
   - Topic extraction and clustering
   - Vector embedding generation

4. Search and Retrieval:
   - Semantic search capabilities
   - Hybrid search (keyword + semantic)
   - Relevance scoring and ranking
   - Faceted search and filtering
   - Query expansion and suggestion

5. Analytics and Insights:
   - Document analytics dashboard
   - Usage patterns and trends
   - Performance metrics and monitoring
   - Custom reporting capabilities
   - Data visualization tools

Technical Requirements:

Performance:
- Process 10,000 documents per hour
- Support 1,000 concurrent users
- Sub-second search response time
- 99.9% system availability
- Handle documents up to 100MB each

Scalability:
- Horizontal scaling capability
- Load balancing across multiple nodes
- Auto-scaling based on demand
- Geographic distribution support
- Multi-tenant architecture

Security:
- End-to-end encryption
- Role-based access control
- Audit logging and monitoring
- Data privacy compliance
- Secure API authentication

Integration:
- RESTful API endpoints
- Webhook support for real-time updates
- SDK for multiple programming languages
- Database connectivity options
- Third-party system integration

Data Storage:
- Distributed file storage
- Vector database for embeddings
- Relational database for metadata
- Cache layer for performance
- Backup and disaster recovery

Machine Learning Models:
- Document classification models
- Named entity recognition
- Sentiment analysis
- Topic modeling
- Question generation models

Deployment:
- Docker containerization
- Kubernetes orchestration
- CI/CD pipeline integration
- Environment management
- Monitoring and alerting

Testing Strategy:
- Unit testing for all components
- Integration testing for APIs
- Performance testing under load
- Security testing and penetration testing
- User acceptance testing

Implementation Timeline:
Phase 1 (Months 1-3): Core infrastructure and basic processing
Phase 2 (Months 4-6): Advanced ML models and search capabilities
Phase 3 (Months 7-9): Analytics and reporting features
Phase 4 (Months 10-12): Optimization and production deployment

Success Criteria:
- 95% document processing accuracy
- 90% user satisfaction score
- 50% reduction in search time
- 99.9% system uptime
- Successful integration with existing systems
            """
        }
    
    def run_enhanced_analysis(self, 
                             documents: Optional[Dict[str, str]] = None,
                             generate_questions: bool = True,
                             extract_topics: bool = True,
                             clustering_method: str = 'auto') -> Dict[str, Any]:
        """
        Run complete enhanced analysis pipeline
        
        Args:
            documents: Documents to analyze (if None, use test documents)
            generate_questions: Whether to generate questions
            extract_topics: Whether to extract topics
            clustering_method: Clustering method to use
            
        Returns:
            Complete analysis results
        """
        if documents is None:
            documents = self.test_documents
        
        logger.info(f"Starting enhanced analysis for {len(documents)} documents")
        
        results = {
            'metadata': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_documents': len(documents),
                'generate_questions': generate_questions,
                'extract_topics': extract_topics,
                'clustering_method': clustering_method
            },
            'document_results': {},
            'clustering_results': {},
            'summary': {}
        }
        
        # Step 1: Enhanced chunking for each document
        all_chunks = []
        
        for doc_name, doc_content in documents.items():
            logger.info(f"Processing document: {doc_name}")
            
            try:
                # Create enhanced chunks
                chunks = self.chunker.chunk_text(
                    text=doc_content,
                    generate_questions=generate_questions,
                    extract_topics=extract_topics
                )
                
                # Convert to dictionaries
                chunk_dicts = [self.chunker.to_dict(chunk) for chunk in chunks]
                
                # Save individual document results
                doc_file = self.output_dir / f"{doc_name}_enhanced_chunks.json"
                with open(doc_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)
                
                results['document_results'][doc_name] = {
                    'chunks_file': str(doc_file),
                    'num_chunks': len(chunks),
                    'total_questions': sum(len(chunk.questions) for chunk in chunks),
                    'total_topics': sum(len(chunk.topics) for chunk in chunks),
                    'avg_chunk_size': sum(chunk.word_count for chunk in chunks) / len(chunks),
                    'success': True
                }
                
                all_chunks.extend(chunk_dicts)
                
                logger.info(f"✅ {doc_name}: {len(chunks)} chunks, {sum(len(chunk.questions) for chunk in chunks)} questions")
                
            except Exception as e:
                logger.error(f"❌ Error processing {doc_name}: {e}")
                results['document_results'][doc_name] = {
                    'error': str(e),
                    'success': False
                }
        
        # Step 2: Question clustering across all documents
        if all_chunks and generate_questions:
            logger.info("Starting question clustering analysis...")
            
            try:
                cluster_analysis = self.clusterer.cluster_questions(
                    chunks=all_chunks,
                    method=clustering_method
                )
                
                # Save clustering results
                cluster_file = self.output_dir / "question_clustering_results.json"
                self.clusterer.export_results(cluster_analysis, str(cluster_file))
                
                # Create visualization
                viz_file = self.output_dir / "cluster_visualization.png"
                self.clusterer.visualize_clusters(cluster_analysis, str(viz_file))
                
                results['clustering_results'] = {
                    'cluster_file': str(cluster_file),
                    'visualization_file': str(viz_file),
                    'total_questions': cluster_analysis.total_questions,
                    'total_clusters': cluster_analysis.total_clusters,
                    'avg_cluster_size': cluster_analysis.avg_cluster_size,
                    'overall_coherence': cluster_analysis.overall_coherence,
                    'cluster_metrics': cluster_analysis.cluster_metrics,
                    'clusters': [
                        {
                            'cluster_id': c.cluster_id,
                            'size': c.size,
                            'topics': c.topics,
                            'centroid_question': c.centroid_question,
                            'coherence_score': c.coherence_score
                        }
                        for c in cluster_analysis.clusters
                    ],
                    'success': True
                }
                
                logger.info(f"✅ Clustering completed: {cluster_analysis.total_clusters} clusters from {cluster_analysis.total_questions} questions")
                
            except Exception as e:
                logger.error(f"❌ Error in clustering: {e}")
                results['clustering_results'] = {
                    'error': str(e),
                    'success': False
                }
        
        # Step 3: Generate summary
        results['summary'] = self._generate_summary(results)
        
        # Save complete results
        results_file = self.output_dir / "enhanced_analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enhanced analysis completed. Results saved to {self.output_dir}")
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        # Document processing summary
        successful_docs = [doc for doc, res in results['document_results'].items() if res.get('success', False)]
        total_chunks = sum(res.get('num_chunks', 0) for res in results['document_results'].values() if res.get('success', False))
        total_questions = sum(res.get('total_questions', 0) for res in results['document_results'].values() if res.get('success', False))
        total_topics = sum(res.get('total_topics', 0) for res in results['document_results'].values() if res.get('success', False))
        
        # Clustering summary
        clustering_success = results['clustering_results'].get('success', False)
        cluster_summary = {}
        if clustering_success:
            cluster_summary = {
                'total_clusters': results['clustering_results']['total_clusters'],
                'avg_cluster_size': results['clustering_results']['avg_cluster_size'],
                'overall_coherence': results['clustering_results']['overall_coherence']
            }
        
        return {
            'processing_summary': {
                'documents_processed': len(successful_docs),
                'total_chunks': total_chunks,
                'total_questions': total_questions,
                'total_topics': total_topics,
                'avg_chunks_per_doc': total_chunks / len(successful_docs) if successful_docs else 0,
                'avg_questions_per_chunk': total_questions / total_chunks if total_chunks > 0 else 0
            },
            'clustering_summary': cluster_summary,
            'files_generated': [
                f.name for f in self.output_dir.glob("*") if f.is_file()
            ]
        }
    
    def print_summary(self, results: Dict[str, Any]):
        """Print human-readable summary"""
        print("\n" + "="*60)
        print("ENHANCED ANALYSIS SUMMARY")
        print("="*60)
        
        # Processing summary
        summary = results['summary']['processing_summary']
        print(f"\n📄 Document Processing:")
        print(f"   Documents processed: {summary['documents_processed']}")
        print(f"   Total chunks: {summary['total_chunks']}")
        print(f"   Total questions: {summary['total_questions']}")
        print(f"   Total topics: {summary['total_topics']}")
        print(f"   Avg chunks per doc: {summary['avg_chunks_per_doc']:.1f}")
        print(f"   Avg questions per chunk: {summary['avg_questions_per_chunk']:.1f}")
        
        # Clustering summary
        if results['clustering_results'].get('success', False):
            cluster_summary = results['summary']['clustering_summary']
            print(f"\n🔍 Question Clustering:")
            print(f"   Total clusters: {cluster_summary['total_clusters']}")
            print(f"   Avg cluster size: {cluster_summary['avg_cluster_size']:.1f}")
            print(f"   Overall coherence: {cluster_summary['overall_coherence']:.3f}")
            
            # Show top clusters
            print(f"\n📊 Top Clusters:")
            for cluster in results['clustering_results']['clusters'][:5]:
                print(f"   Cluster {cluster['cluster_id']}: {cluster['size']} questions")
                print(f"     Topics: {', '.join(cluster['topics'])}")
                print(f"     Centroid: {cluster['centroid_question']}")
                print()
        
        # Files generated
        print(f"\n📁 Files Generated:")
        for file in results['summary']['files_generated']:
            print(f"   - {file}")

def main():
    """Main function to run enhanced analysis"""
    print("🚀 Starting Enhanced Analysis Pipeline...")
    print("📊 This will run enhanced chunking + question clustering + topic extraction")
    print("📁 Results will be saved to test-data/enhanced/ directory")
    print("-" * 60)
    
    # Create runner
    runner = EnhancedAnalysisRunner()
    
    # Run analysis
    results = runner.run_enhanced_analysis(
        generate_questions=True,
        extract_topics=True,
        clustering_method='auto'
    )
    
    # Print summary
    runner.print_summary(results)
    
    print("\n✅ Enhanced analysis completed successfully!")
    print(f"📂 Results saved to: {runner.output_dir}")
    print("\n🎯 Next steps:")
    print("   1. Review enhanced chunks with questions and topics")
    print("   2. Analyze question clusters for index understanding")
    print("   3. Use topics for agent decision making")
    print("   4. Launch dashboard to visualize results")

if __name__ == "__main__":
    main() 