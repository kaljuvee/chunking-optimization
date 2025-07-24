#!/usr/bin/env python3
"""
ADNOC Synthetic Data Generator
==============================

This module generates synthetic test data for evaluating chunking strategies.
It creates various document types with different characteristics to test
chunking performance across diverse content.

Author: Data Engineering Team
Purpose: Generate test data for chunking evaluation
"""

import json
import random
import string
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentTemplate:
    """Template for generating synthetic documents"""
    doc_type: str
    structure: List[str]
    content_patterns: List[str]
    complexity_levels: List[str]
    languages: List[str]

class SyntheticDataGenerator:
    """
    Generator for synthetic test data
    """
    
    def __init__(self):
        """Initialize the data generator"""
        self.templates = self._initialize_templates()
        self.vocabulary = self._initialize_vocabulary()
        
    def _initialize_templates(self) -> Dict[str, DocumentTemplate]:
        """Initialize document templates"""
        return {
            'technical_report': DocumentTemplate(
                doc_type='technical_report',
                structure=['abstract', 'introduction', 'methodology', 'results', 'conclusion'],
                content_patterns=['data_analysis', 'technical_specs', 'performance_metrics'],
                complexity_levels=['simple', 'moderate', 'complex'],
                languages=['english', 'arabic']
            ),
            'executive_summary': DocumentTemplate(
                doc_type='executive_summary',
                structure=['overview', 'key_points', 'recommendations', 'next_steps'],
                content_patterns=['business_metrics', 'strategic_insights', 'action_items'],
                complexity_levels=['simple', 'moderate'],
                languages=['english', 'arabic']
            ),
            'regulatory_document': DocumentTemplate(
                doc_type='regulatory_document',
                structure=['scope', 'requirements', 'compliance', 'enforcement'],
                content_patterns=['legal_terms', 'compliance_checklists', 'penalties'],
                complexity_levels=['moderate', 'complex'],
                languages=['english', 'arabic']
            ),
            'research_paper': DocumentTemplate(
                doc_type='research_paper',
                structure=['abstract', 'literature_review', 'methodology', 'results', 'discussion'],
                content_patterns=['academic_writing', 'statistical_analysis', 'citations'],
                complexity_levels=['moderate', 'complex'],
                languages=['english']
            ),
            'email_communication': DocumentTemplate(
                doc_type='email_communication',
                structure=['header', 'body', 'signature'],
                content_patterns=['business_communication', 'project_updates', 'meeting_notes'],
                complexity_levels=['simple', 'moderate'],
                languages=['english', 'arabic']
            ),
            'meeting_minutes': DocumentTemplate(
                doc_type='meeting_minutes',
                structure=['header', 'attendees', 'agenda', 'decisions', 'action_items'],
                content_patterns=['meeting_summary', 'decision_records', 'task_assignments'],
                complexity_levels=['simple', 'moderate'],
                languages=['english', 'arabic']
            ),
            'financial_report': DocumentTemplate(
                doc_type='financial_report',
                structure=['executive_summary', 'financial_highlights', 'analysis', 'outlook'],
                content_patterns=['financial_metrics', 'performance_analysis', 'projections'],
                complexity_levels=['moderate', 'complex'],
                languages=['english', 'arabic']
            ),
            'safety_incident_report': DocumentTemplate(
                doc_type='safety_incident_report',
                structure=['incident_details', 'root_cause', 'corrective_actions', 'lessons_learned'],
                content_patterns=['incident_description', 'safety_analysis', 'preventive_measures'],
                complexity_levels=['simple', 'moderate'],
                languages=['english', 'arabic']
            )
        }
    
    def _initialize_vocabulary(self) -> Dict[str, List[str]]:
        """Initialize vocabulary for different content types"""
        return {
            'oil_gas_terms': [
                'reservoir', 'wellbore', 'production', 'injection', 'pressure', 'temperature',
                'flowrate', 'completion', 'stimulation', 'artificial_lift', 'enhanced_recovery',
                'carbonate', 'sandstone', 'porosity', 'permeability', 'saturation', 'viscosity',
                'API_gravity', 'water_cut', 'gas_oil_ratio', 'formation_damage', 'skin_factor'
            ],
            'business_terms': [
                'revenue', 'profitability', 'efficiency', 'optimization', 'strategy', 'performance',
                'investment', 'ROI', 'cost_management', 'budget', 'forecast', 'analysis',
                'stakeholder', 'compliance', 'governance', 'risk_management', 'sustainability'
            ],
            'technical_terms': [
                'algorithm', 'optimization', 'simulation', 'modeling', 'analysis', 'processing',
                'integration', 'deployment', 'monitoring', 'automation', 'digitalization',
                'machine_learning', 'artificial_intelligence', 'data_analytics', 'visualization'
            ],
            'safety_terms': [
                'incident', 'safety', 'risk', 'hazard', 'mitigation', 'prevention', 'compliance',
                'training', 'procedures', 'equipment', 'maintenance', 'inspection', 'audit',
                'emergency', 'response', 'investigation', 'corrective_action'
            ],
            'environmental_terms': [
                'emissions', 'carbon_footprint', 'sustainability', 'environmental_impact',
                'compliance', 'monitoring', 'reduction', 'efficiency', 'renewable_energy',
                'carbon_capture', 'waste_management', 'water_conservation'
            ]
        }
    
    def generate_document(self, 
                         doc_type: str, 
                         complexity: str = 'moderate',
                         language: str = 'english',
                         length: str = 'medium') -> str:
        """
        Generate a synthetic document
        
        Args:
            doc_type: Type of document to generate
            complexity: Complexity level (simple, moderate, complex)
            language: Language (english, arabic)
            length: Document length (short, medium, long)
            
        Returns:
            Generated document text
        """
        if doc_type not in self.templates:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        template = self.templates[doc_type]
        
        if complexity not in template.complexity_levels:
            complexity = template.complexity_levels[0]
        
        if language not in template.languages:
            language = template.languages[0]
        
        # Generate document based on template
        document_parts = []
        
        for section in template.structure:
            section_content = self._generate_section(
                section, doc_type, complexity, language, length
            )
            document_parts.append(section_content)
        
        # Combine sections
        document = '\n\n'.join(document_parts)
        
        return document
    
    def _generate_section(self, 
                         section: str, 
                         doc_type: str, 
                         complexity: str,
                         language: str, 
                         length: str) -> str:
        """
        Generate content for a specific section
        
        Args:
            section: Section name
            doc_type: Document type
            complexity: Complexity level
            language: Language
            length: Length preference
            
        Returns:
            Generated section content
        """
        # Determine section length based on complexity and length preference
        if length == 'short':
            base_length = 100
        elif length == 'medium':
            base_length = 300
        else:  # long
            base_length = 600
        
        # Adjust for complexity
        if complexity == 'simple':
            length_multiplier = 0.7
        elif complexity == 'moderate':
            length_multiplier = 1.0
        else:  # complex
            length_multiplier = 1.5
        
        target_length = int(base_length * length_multiplier)
        
        # Generate section content based on type
        if section == 'abstract':
            return self._generate_abstract(doc_type, target_length, language)
        elif section == 'introduction':
            return self._generate_introduction(doc_type, target_length, language)
        elif section == 'methodology':
            return self._generate_methodology(doc_type, target_length, language)
        elif section == 'results':
            return self._generate_results(doc_type, target_length, language)
        elif section == 'conclusion':
            return self._generate_conclusion(doc_type, target_length, language)
        elif section == 'overview':
            return self._generate_overview(doc_type, target_length, language)
        elif section == 'key_points':
            return self._generate_key_points(doc_type, target_length, language)
        elif section == 'recommendations':
            return self._generate_recommendations(doc_type, target_length, language)
        elif section == 'requirements':
            return self._generate_requirements(doc_type, target_length, language)
        elif section == 'compliance':
            return self._generate_compliance(doc_type, target_length, language)
        else:
            return self._generate_generic_section(section, doc_type, target_length, language)
    
    def _generate_abstract(self, doc_type: str, length: int, language: str) -> str:
        """Generate abstract section"""
        if doc_type == 'technical_report':
            return f"""This {doc_type.replace('_', ' ')} presents the findings of a comprehensive study conducted at ADNOC facilities. The research focuses on optimizing operational efficiency through advanced technology implementation. Key results demonstrate significant improvements in performance metrics, with cost reductions of 25% and efficiency gains of 30%. The methodology employed state-of-the-art analytical techniques and real-time monitoring systems. These findings provide valuable insights for future implementation strategies across the organization."""
        else:
            return f"""This document provides a comprehensive overview of {doc_type.replace('_', ' ')} activities and outcomes. The analysis covers key performance indicators, challenges encountered, and recommendations for improvement. Results indicate positive progress toward organizational objectives with measurable improvements in efficiency and effectiveness."""
    
    def _generate_introduction(self, doc_type: str, length: int, language: str) -> str:
        """Generate introduction section"""
        return f"""The {doc_type.replace('_', ' ')} initiative represents a strategic effort to enhance operational performance and achieve organizational objectives. This document provides a detailed analysis of the current state, challenges, and opportunities for improvement. The scope encompasses multiple operational areas including production optimization, safety management, and environmental compliance. The methodology combines quantitative analysis with qualitative assessment to provide comprehensive insights."""
    
    def _generate_methodology(self, doc_type: str, length: int, language: str) -> str:
        """Generate methodology section"""
        return f"""The methodology employed in this {doc_type.replace('_', ' ')} follows established industry standards and best practices. Data collection involved comprehensive monitoring systems, automated sensors, and manual inspections. Analysis techniques included statistical modeling, machine learning algorithms, and comparative benchmarking. Quality assurance measures ensured data accuracy and reliability throughout the process. The approach prioritized safety, efficiency, and environmental responsibility."""
    
    def _generate_results(self, doc_type: str, length: int, language: str) -> str:
        """Generate results section"""
        return f"""The results of the {doc_type.replace('_', ' ')} demonstrate significant improvements across multiple performance indicators. Key achievements include a 25% increase in operational efficiency, 30% reduction in safety incidents, and 20% improvement in environmental compliance. Cost savings totaled $15 million annually, while productivity gains exceeded initial projections by 15%. These results validate the effectiveness of the implemented strategies and provide a foundation for future initiatives."""
    
    def _generate_conclusion(self, doc_type: str, length: int, language: str) -> str:
        """Generate conclusion section"""
        return f"""In conclusion, the {doc_type.replace('_', ' ')} has successfully achieved its primary objectives and delivered measurable value to the organization. The implemented strategies have demonstrated their effectiveness through improved performance metrics and operational outcomes. Key learnings include the importance of stakeholder engagement, the value of data-driven decision making, and the benefits of continuous improvement processes. Future initiatives should build upon these successes while addressing identified challenges and opportunities."""
    
    def _generate_overview(self, doc_type: str, length: int, language: str) -> str:
        """Generate overview section"""
        return f"""This {doc_type.replace('_', ' ')} provides a comprehensive overview of current operations and strategic initiatives. The overview encompasses key performance metrics, operational challenges, and strategic objectives. Recent developments include technology upgrades, process improvements, and organizational changes. The current state reflects ongoing efforts to enhance efficiency, safety, and environmental performance while maintaining operational excellence."""
    
    def _generate_key_points(self, doc_type: str, length: int, language: str) -> str:
        """Generate key points section"""
        return f"""Key points from the {doc_type.replace('_', ' ')} include: 1) Significant improvements in operational efficiency achieved through technology implementation, 2) Enhanced safety performance with reduced incident rates, 3) Improved environmental compliance and sustainability metrics, 4) Successful stakeholder engagement and communication strategies, 5) Measurable cost savings and productivity gains. These achievements demonstrate the effectiveness of strategic planning and execution."""
    
    def _generate_recommendations(self, doc_type: str, length: int, language: str) -> str:
        """Generate recommendations section"""
        return f"""Based on the analysis of the {doc_type.replace('_', ' ')}, the following recommendations are proposed: 1) Continue investment in technology infrastructure and digital transformation initiatives, 2) Expand safety training programs and enhance monitoring systems, 3) Implement additional environmental protection measures and sustainability programs, 4) Strengthen stakeholder communication and engagement processes, 5) Develop comprehensive performance monitoring and reporting frameworks. These recommendations aim to sustain current improvements and drive future success."""
    
    def _generate_requirements(self, doc_type: str, length: int, language: str) -> str:
        """Generate requirements section"""
        return f"""The {doc_type.replace('_', ' ')} establishes the following requirements: 1) Compliance with all applicable regulatory standards and industry best practices, 2) Implementation of comprehensive monitoring and reporting systems, 3) Regular training and certification for all personnel, 4) Maintenance of equipment and systems according to manufacturer specifications, 5) Documentation of all procedures and processes. These requirements ensure operational excellence and regulatory compliance."""
    
    def _generate_compliance(self, doc_type: str, length: int, language: str) -> str:
        """Generate compliance section"""
        return f"""Compliance with the {doc_type.replace('_', ' ')} requirements is monitored through regular audits, inspections, and performance assessments. Compliance measures include: 1) Quarterly internal audits conducted by qualified personnel, 2) Annual external audits by certified third-party organizations, 3) Continuous monitoring of key performance indicators, 4) Regular review and update of procedures and processes, 5) Comprehensive training and awareness programs. Non-compliance is addressed through corrective action plans and continuous improvement initiatives."""
    
    def _generate_generic_section(self, section: str, doc_type: str, length: int, language: str) -> str:
        """Generate generic section content"""
        return f"""The {section} of this {doc_type.replace('_', ' ')} addresses key aspects of the initiative and provides important information for stakeholders. This section covers relevant details, analysis, and insights that contribute to the overall understanding of the project scope and objectives. The content is structured to facilitate comprehension and support decision-making processes."""
    
    def generate_test_dataset(self, 
                             output_dir: str = "data",
                             num_documents: int = 10,
                             document_types: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Generate a complete test dataset
        
        Args:
            output_dir: Output directory for generated files
            num_documents: Number of documents to generate
            document_types: List of document types to include
            
        Returns:
            Dictionary mapping document names to file paths
        """
        if document_types is None:
            document_types = list(self.templates.keys())
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        generated_files = {}
        
        for i in range(num_documents):
            # Select random document type
            doc_type = random.choice(document_types)
            
            # Select random parameters
            complexity = random.choice(['simple', 'moderate', 'complex'])
            language = random.choice(['english', 'arabic'])
            length = random.choice(['short', 'medium', 'long'])
            
            # Generate document
            document = self.generate_document(doc_type, complexity, language, length)
            
            # Create filename
            filename = f"synthetic_{doc_type}_{i+1:03d}.txt"
            filepath = output_path / filename
            
            # Save document
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(document)
            
            generated_files[filename] = str(filepath)
            
            logger.info(f"Generated: {filename}")
        
        # Save metadata
        metadata = {
            'num_documents': num_documents,
            'document_types': document_types,
            'generated_files': generated_files
        }
        
        metadata_path = output_path / "synthetic_data_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Generated {num_documents} synthetic documents in {output_dir}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return generated_files
    
    def generate_ground_truth(self, documents: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate ground truth annotations for documents
        
        Args:
            documents: Dictionary of document paths
            
        Returns:
            Ground truth annotations
        """
        ground_truth = {}
        
        for doc_name, doc_path in documents.items():
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate ground truth annotations
            annotations = {
                'document_type': self._classify_document_type(content),
                'key_topics': self._extract_key_topics(content),
                'important_sections': self._identify_important_sections(content),
                'technical_terms': self._extract_technical_terms(content),
                'recommended_chunk_boundaries': self._suggest_chunk_boundaries(content)
            }
            
            ground_truth[doc_name] = annotations
        
        return ground_truth
    
    def _classify_document_type(self, content: str) -> str:
        """Classify document type based on content"""
        content_lower = content.lower()
        
        if 'abstract' in content_lower and 'methodology' in content_lower:
            return 'research_paper'
        elif 'executive' in content_lower and 'summary' in content_lower:
            return 'executive_summary'
        elif 'regulatory' in content_lower or 'compliance' in content_lower:
            return 'regulatory_document'
        elif 'technical' in content_lower and 'report' in content_lower:
            return 'technical_report'
        elif 'meeting' in content_lower and 'minutes' in content_lower:
            return 'meeting_minutes'
        elif 'financial' in content_lower and 'report' in content_lower:
            return 'financial_report'
        elif 'safety' in content_lower and 'incident' in content_lower:
            return 'safety_incident_report'
        else:
            return 'general_document'
    
    def _extract_key_topics(self, content: str) -> List[str]:
        """Extract key topics from content"""
        topics = []
        content_lower = content.lower()
        
        # Check for oil and gas terms
        for term in self.vocabulary['oil_gas_terms']:
            if term in content_lower:
                topics.append(term)
        
        # Check for business terms
        for term in self.vocabulary['business_terms']:
            if term in content_lower:
                topics.append(term)
        
        # Check for technical terms
        for term in self.vocabulary['technical_terms']:
            if term in content_lower:
                topics.append(term)
        
        return topics[:5]  # Return top 5 topics
    
    def _identify_important_sections(self, content: str) -> List[str]:
        """Identify important sections in content"""
        sections = []
        lines = content.split('\n')
        
        for line in lines:
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in ['conclusion', 'recommendation', 'key', 'important', 'critical']):
                sections.append(line.strip())
        
        return sections[:3]  # Return top 3 important sections
    
    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical terms from content"""
        terms = []
        content_lower = content.lower()
        
        for category, term_list in self.vocabulary.items():
            for term in term_list:
                if term in content_lower:
                    terms.append(term)
        
        return list(set(terms))[:10]  # Return unique terms, max 10
    
    def _suggest_chunk_boundaries(self, content: str) -> List[int]:
        """Suggest optimal chunk boundaries"""
        sentences = content.split('.')
        boundaries = []
        
        # Suggest boundaries every 3-5 sentences
        for i in range(3, len(sentences), random.randint(3, 5)):
            if i < len(sentences):
                boundaries.append(i)
        
        return boundaries

def main():
    """Example usage of the synthetic data generator"""
    generator = SyntheticDataGenerator()
    
    # Generate a single document
    document = generator.generate_document(
        doc_type='technical_report',
        complexity='moderate',
        language='english',
        length='medium'
    )
    
    print("Generated Document:")
    print("=" * 50)
    print(document)
    print("=" * 50)
    
    # Generate test dataset
    print("\nGenerating test dataset...")
    files = generator.generate_test_dataset(
        output_dir="data/synthetic",
        num_documents=5,
        document_types=['technical_report', 'executive_summary', 'regulatory_document']
    )
    
    print(f"\nGenerated {len(files)} documents:")
    for filename, filepath in files.items():
        print(f"  - {filename}: {filepath}")

if __name__ == "__main__":
    main() 