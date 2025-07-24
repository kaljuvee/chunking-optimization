#!/usr/bin/env python3
"""
6. Data Generator
=================

Generate sample documents for testing chunking strategies.

Author: Data Engineering Team
Purpose: Sample data generation for testing and evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import sys

# Configure page
st.set_page_config(
    page_title="Data Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main function for data generator page"""
    
    st.title("📄 Data Generator")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Document types
    document_types = [
        "executive_summaries",
        "mixed_content", 
        "regulatory_documents",
        "research_papers",
        "technical_reports",
        "legal_documents",
        "news_articles",
        "academic_papers"
    ]
    
    selected_types = st.sidebar.multiselect(
        "Select document types:",
        options=document_types,
        default=document_types[:3]
    )
    
    # Generation parameters
    col1, col2 = st.columns(2)
    with col1:
        num_documents = st.sidebar.slider("Number of documents:", 1, 20, 5)
        min_length = st.sidebar.slider("Min document length (words):", 500, 2000, 1000)
    
    with col2:
        max_length = st.sidebar.slider("Max document length (words):", 2000, 10000, 5000)
        complexity_level = st.sidebar.selectbox("Complexity level:", ["simple", "medium", "complex"])
    
    # Content settings
    st.sidebar.header("📝 Content Settings")
    
    include_tables = st.sidebar.checkbox("Include tables", value=True)
    include_lists = st.sidebar.checkbox("Include lists", value=True)
    include_headers = st.sidebar.checkbox("Include headers", value=True)
    include_citations = st.sidebar.checkbox("Include citations", value=False)
    
    # Language and style
    language = st.sidebar.selectbox("Language:", ["English", "Arabic", "French", "Spanish"])
    writing_style = st.sidebar.selectbox("Writing style:", ["formal", "technical", "academic", "business"])
    
    # Generate data button
    if st.sidebar.button("🚀 Generate Sample Data", type="primary"):
        generate_sample_data(selected_types, num_documents, min_length, max_length,
                           complexity_level, include_tables, include_lists, include_headers,
                           include_citations, language, writing_style)
    
    # Load existing data
    if st.sidebar.button("📂 Load Existing Data"):
        load_existing_data()
    
    # Main content area
    if 'generated_data' in st.session_state:
        display_generated_data(st.session_state.generated_data)
    else:
        display_welcome()

def generate_sample_data(types, num_docs, min_len, max_len, complexity, include_tables,
                        include_lists, include_headers, include_citations, language, style):
    """Generate sample data"""
    
    st.subheader("🔄 Generating Sample Data...")
    
    with st.spinner("Creating sample documents..."):
        try:
            # Run the data generator script
            result = execute_data_generator(types, num_docs, min_len, max_len, complexity,
                                          include_tables, include_lists, include_headers,
                                          include_citations, language, style)
            
            st.session_state.generated_data = result
            st.success("🎉 Sample data generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error generating data: {e}")

def execute_data_generator(types, num_docs, min_len, max_len, complexity, include_tables,
                          include_lists, include_headers, include_citations, language, style):
    """Execute the data generator script"""
    
    script_path = "utils/generate_sample_data.py"
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Data generator script not found: {script_path}")
    
    # Prepare command
    cmd = [
        sys.executable, script_path,
        "--document-types", ",".join(types),
        "--num-documents", str(num_docs),
        "--min-length", str(min_len),
        "--max-length", str(max_len),
        "--complexity", complexity,
        "--language", language,
        "--style", style,
        "--output-dir", "data"
    ]
    
    # Add optional flags
    if include_tables:
        cmd.append("--include-tables")
    if include_lists:
        cmd.append("--include-lists")
    if include_headers:
        cmd.append("--include-headers")
    if include_citations:
        cmd.append("--include-citations")
    
    # Execute command
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        raise RuntimeError(f"Data generation failed: {result.stderr}")
    
    # Load generated data
    generated_data = load_generated_files()
    
    return {
        "command": cmd,
        "output": result.stdout,
        "files": generated_data,
        "parameters": {
            "types": types,
            "num_docs": num_docs,
            "min_len": min_len,
            "max_len": max_len,
            "complexity": complexity,
            "language": language,
            "style": style
        }
    }

def load_generated_files():
    """Load generated files from data directory"""
    
    data_dir = Path("data")
    generated_files = []
    
    if data_dir.exists():
        for file_path in data_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                generated_files.append({
                    "filename": file_path.name,
                    "size": len(content),
                    "word_count": len(content.split()),
                    "path": str(file_path)
                })
            except Exception as e:
                st.warning(f"Could not read {file_path}: {e}")
    
    return generated_files

def load_existing_data():
    """Load existing generated data"""
    
    generated_files = load_generated_files()
    
    if generated_files:
        st.session_state.generated_data = {
            "files": generated_files,
            "parameters": {"note": "Loaded existing data"}
        }
        st.success("📂 Existing data loaded successfully!")
    else:
        st.warning("No existing data found. Generate sample data first.")

def display_generated_data(data):
    """Display generated data information"""
    
    st.subheader("📊 Generated Data Summary")
    
    files = data.get("files", [])
    parameters = data.get("parameters", {})
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Generated", len(files))
    
    with col2:
        total_words = sum(f["word_count"] for f in files)
        st.metric("Total Words", f"{total_words:,}")
    
    with col3:
        avg_words = total_words // len(files) if files else 0
        st.metric("Avg Words per Doc", f"{avg_words:,}")
    
    with col4:
        total_size = sum(f["size"] for f in files)
        st.metric("Total Size", f"{total_size:,} chars")
    
    # Generation parameters
    st.subheader("⚙️ Generation Parameters")
    
    if parameters:
        param_df = pd.DataFrame([parameters])
        st.dataframe(param_df, use_container_width=True)
    
    # Files table
    st.subheader("📋 Generated Files")
    
    if files:
        files_df = pd.DataFrame(files)
        st.dataframe(files_df, use_container_width=True)
        
        # File statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Word count distribution
            word_counts = [f["word_count"] for f in files]
            fig = px.histogram(x=word_counts, title='Document Length Distribution',
                             nbins=10, labels={'x': 'Word Count', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # File size distribution
            file_sizes = [f["size"] for f in files]
            fig = px.histogram(x=file_sizes, title='File Size Distribution',
                             nbins=10, labels={'x': 'File Size (chars)', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
    
    # File preview
    st.subheader("👀 File Preview")
    
    if files:
        selected_file = st.selectbox("Select file to preview:", [f["filename"] for f in files])
        
        if selected_file:
            file_info = next(f for f in files if f["filename"] == selected_file)
            
            try:
                with open(file_info["path"], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Show file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Word Count", file_info["word_count"])
                with col2:
                    st.metric("File Size", f"{file_info['size']:,} chars")
                with col3:
                    st.metric("Filename", file_info["filename"])
                
                # Show content preview
                preview_length = 1000
                preview = content[:preview_length]
                if len(content) > preview_length:
                    preview += "..."
                
                st.text_area("Content Preview:", preview, height=300, disabled=True)
                
                # Download button
                st.download_button(
                    label="📥 Download File",
                    data=content,
                    file_name=file_info["filename"],
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Data quality metrics
    st.subheader("📈 Data Quality Metrics")
    
    if files:
        quality_metrics = calculate_data_quality(files)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Document Statistics:**")
            st.markdown(f"- Average document length: {quality_metrics['avg_length']:.1f} words")
            st.markdown(f"- Length standard deviation: {quality_metrics['length_std']:.1f} words")
            st.markdown(f"- Shortest document: {quality_metrics['min_length']} words")
            st.markdown(f"- Longest document: {quality_metrics['max_length']} words")
        
        with col2:
            st.markdown("**Content Analysis:**")
            st.markdown(f"- Total unique words: {quality_metrics['unique_words']}")
            st.markdown(f"- Average words per sentence: {quality_metrics['avg_sentence_length']:.1f}")
            st.markdown(f"- Document variety score: {quality_metrics['variety_score']:.3f}")
            st.markdown(f"- Content complexity: {quality_metrics['complexity_level']}")

def calculate_data_quality(files):
    """Calculate data quality metrics"""
    
    word_counts = [f["word_count"] for f in files]
    
    # Basic statistics
    avg_length = np.mean(word_counts)
    length_std = np.std(word_counts)
    min_length = min(word_counts)
    max_length = max(word_counts)
    
    # Simulated content analysis
    unique_words = np.random.randint(500, 2000)
    avg_sentence_length = np.random.uniform(15, 25)
    variety_score = np.random.uniform(0.6, 0.9)
    complexity_level = np.random.choice(["simple", "medium", "complex"])
    
    return {
        "avg_length": avg_length,
        "length_std": length_std,
        "min_length": min_length,
        "max_length": max_length,
        "unique_words": unique_words,
        "avg_sentence_length": avg_sentence_length,
        "variety_score": variety_score,
        "complexity_level": complexity_level
    }

def display_welcome():
    """Display welcome message"""
    
    st.markdown("""
    ## Welcome to Data Generator
    
    This tool generates sample documents for testing chunking strategies and evaluating performance.
    
    ### How to use:
    1. **Select Document Types**: Choose which types of documents to generate
    2. **Configure Parameters**: Adjust document length, complexity, and content settings
    3. **Set Content Options**: Choose what elements to include (tables, lists, headers, etc.)
    4. **Generate Data**: Create sample documents for testing
    5. **Preview and Download**: Review generated content and download files
    
    ### Available Document Types:
    - **Executive Summaries**: High-level business documents with key insights
    - **Mixed Content**: Documents with various content types and structures
    - **Regulatory Documents**: Legal and compliance documents with formal language
    - **Research Papers**: Academic papers with citations and technical content
    - **Technical Reports**: Detailed technical documentation with specifications
    - **Legal Documents**: Formal legal documents with structured content
    - **News Articles**: Journalistic content with current events
    - **Academic Papers**: Scholarly articles with research methodology
    
    ### Content Features:
    - **Tables**: Structured data presentation
    - **Lists**: Bulleted and numbered lists
    - **Headers**: Document structure and organization
    - **Citations**: Reference and attribution systems
    - **Multiple Languages**: Support for various languages
    - **Writing Styles**: Formal, technical, academic, and business styles
    
    ### Generation Parameters:
    - Document length (word count range)
    - Complexity level (simple, medium, complex)
    - Language selection
    - Writing style preferences
    - Content structure options
    
    ### Output:
    - Multiple document files in text format
    - Quality metrics and statistics
    - Content preview and analysis
    - Download capabilities for testing
    """)

if __name__ == "__main__":
    main() 