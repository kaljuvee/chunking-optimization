#!/usr/bin/env python3
"""
Chunking Analysis Dashboard - Home
=================================

Main landing page for the multi-page chunking analysis dashboard.
Comprehensive overview of all functionality and tools.

Author: Data Engineering Team
Purpose: Interactive chunking analysis and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os

# Configure page
st.set_page_config(
    page_title="Chunking Analysis - Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_credentials(username, password):
    """Check if the provided credentials are valid"""
    return username == "team@inception.ai" and password == "Optimisation2$2"

def login_page():
    """Display login page"""
    st.title("🔐 Chunking Analysis - Login")
    st.markdown("---")
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3>Welcome to the Chunking Analysis Dashboard</h3>
            <p>Please enter your credentials to continue</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if check_credentials(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
    
    # Add some styling
    st.markdown("""
    <style>
    .stForm {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main function for the home page"""
    
    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Check authentication
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Logout button in sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.rerun()
        st.markdown(f"**Logged in as:** {st.session_state.username}")
    
    # Header
    st.title("🏠 Chunking Strategy Analysis Dashboard")
    st.markdown("---")
    
    # Welcome section
    st.markdown("""
    ## Welcome to the Advanced Chunking Analysis Platform
    
    This comprehensive dashboard provides multiple tools for analyzing, testing, and comparing 
    different chunking strategies for document processing and RAG (Retrieval-Augmented Generation) systems.
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Available Strategies", "6+", "Core chunking methods")
    
    with col2:
        st.metric("Visualization Tools", "4", "Interactive charts")
    
    with col3:
        st.metric("Test Documents", "5", "Sample datasets")
    
    with col4:
        st.metric("Analysis Types", "8", "Different experiments")
    
    st.markdown("---")
    
    # Available pages overview
    st.subheader("📋 Available Experiments & Tools")
    
    # Create a grid layout for the pages
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔬 **Chunking Experiments**
        
        **1. Strategy Comparison**  
        Compare different chunking strategies side-by-side with performance metrics.
        
        **2. Semantic Analysis**  
        Analyze semantic coherence and context preservation across chunks.
        
        **3. RAG Performance**  
        Evaluate retrieval-augmented generation performance with different chunking approaches.
        
        **4. LLM Chunking Tests**  
        Test and compare LLM-based chunking strategies.
        
        **5. Topic Clustering**  
        Explore topic-based chunking and clustering analysis.
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ **Utility Tools**
        
        **6. Data Generator**  
        Generate sample documents for testing chunking strategies.
        
        **7. Visualization Suite**  
        Advanced visualization tools for chunking analysis.
        
        **8. Performance Benchmark**  
        Comprehensive performance benchmarking and optimization tools.
        """)
    
    st.markdown("---")
    
    # Detailed functionality overview
    st.subheader("🔍 Detailed Functionality Overview")
    
    # Strategy Comparison
    with st.expander("📊 1. Strategy Comparison", expanded=False):
        st.markdown("""
        **Purpose**: Compare different chunking strategies side-by-side with performance metrics.
        
        **Features**:
        - Multi-strategy comparison with interactive parameter configuration
        - Performance metrics visualization (coherence, accuracy, processing time)
        - Radar charts and scatter plots for strategy comparison
        - Detailed strategy analysis with chunk-level insights
        - Export capabilities for results and visualizations
        
        **Available Strategies**:
        - Full Overlap: Traditional overlap-based chunking
        - Full Summary: Summary-based chunking approach
        - Page Overlap: Page-aware overlap chunking
        - Page Summary: Page-based summary chunking
        - Semantic LangChain: LangChain semantic splitting
        - Per-Page LangChain: LangChain per-page processing
        """)
    
    # Semantic Analysis
    with st.expander("🔍 2. Semantic Analysis", expanded=False):
        st.markdown("""
        **Purpose**: Analyze semantic coherence and context preservation across chunks.
        
        **Features**:
        - Semantic coherence evaluation using embedding models
        - Context preservation analysis across chunk boundaries
        - Topic consistency measurement between consecutive chunks
        - Semantic similarity calculations for chunk pairs
        - Boundary quality assessment and optimization
        
        **Analysis Types**:
        - Semantic coherence scoring
        - Context preservation metrics
        - Topic consistency analysis
        - Semantic similarity mapping
        - Boundary quality evaluation
        """)
    
    # RAG Performance
    with st.expander("🎯 3. RAG Performance", expanded=False):
        st.markdown("""
        **Purpose**: Evaluate retrieval-augmented generation performance with different chunking approaches.
        
        **Features**:
        - RAG performance metrics (precision, recall, F1-score)
        - Query-based evaluation with custom test queries
        - Retrieval accuracy analysis for different chunking strategies
        - Response quality assessment and scoring
        - Performance benchmarking across multiple metrics
        
        **Evaluation Metrics**:
        - Retrieval precision and recall
        - Answer accuracy and relevance
        - Context relevance scoring
        - Response coherence evaluation
        - Overall RAG performance ranking
        """)
    
    # LLM Chunking Tests
    with st.expander("🤖 4. LLM Chunking Tests", expanded=False):
        st.markdown("""
        **Purpose**: Test and compare LLM-based chunking strategies.
        
        **Features**:
        - Content-aware chunking using LLM understanding
        - Hierarchical chunking with multiple levels
        - Model performance comparison across different LLMs
        - Cost analysis and token efficiency evaluation
        - Processing time and resource usage monitoring
        
        **Available Methods**:
        - Content Aware Chunker: LLM-based intelligent chunking
        - Hierarchical Chunker: Multi-level document structure
        - Model Comparison: GPT-4, GPT-3.5, Claude, etc.
        - Cost Optimization: Token usage and cost analysis
        """)
    
    # Topic Clustering
    with st.expander("🗂️ 5. Topic Clustering", expanded=False):
        st.markdown("""
        **Purpose**: Explore topic-based chunking and clustering analysis.
        
        **Features**:
        - Multiple clustering algorithms (K-means, hierarchical, DBSCAN)
        - Topic distribution analysis and visualization
        - Semantic clustering using embeddings
        - Cluster quality metrics and evaluation
        - Interactive cluster visualization and exploration
        
        **Clustering Methods**:
        - K-Means Clustering: Traditional centroid-based approach
        - Hierarchical Clustering: Tree-based clustering
        - DBSCAN Clustering: Density-based spatial clustering
        - Topic Modeling: Latent topic discovery
        - Semantic Clustering: Embedding-based clustering
        """)
    
    # Data Generator
    with st.expander("📄 6. Data Generator", expanded=False):
        st.markdown("""
        **Purpose**: Generate sample documents for testing chunking strategies.
        
        **Features**:
        - Multiple document types (executive summaries, research papers, etc.)
        - Customizable parameters (length, complexity, style)
        - Content structure options (tables, lists, headers)
        - Language and style selection
        - Quality metrics analysis and validation
        
        **Document Types**:
        - Executive Summaries: High-level business documents
        - Mixed Content: Various content types and structures
        - Regulatory Documents: Legal and compliance documents
        - Research Papers: Academic and technical papers
        - Technical Reports: Detailed technical documentation
        """)
    
    # Visualization Suite
    with st.expander("📊 7. Visualization Suite", expanded=False):
        st.markdown("""
        **Purpose**: Advanced visualization tools for chunking analysis.
        
        **Features**:
        - Multiple visualization types (overlap analysis, coherence heatmaps, etc.)
        - Interactive charts with Plotly integration
        - Export capabilities (PNG, SVG, PDF, HTML)
        - Customizable themes and color schemes
        - Statistical analysis and insights
        
        **Visualization Types**:
        - Chunk Overlap Analysis: Overlap patterns and distributions
        - Semantic Coherence Heatmap: Semantic relationship mapping
        - Performance Comparison: Strategy comparison charts
        - Topic Distribution: Topic clustering visualization
        - Embedding Visualization: 2D/3D embedding plots
        - Interactive Dashboard: Comprehensive multi-view dashboard
        """)
    
    # Performance Benchmark
    with st.expander("⚡ 8. Performance Benchmark", expanded=False):
        st.markdown("""
        **Purpose**: Comprehensive performance benchmarking and optimization tools.
        
        **Features**:
        - Speed and memory benchmarks across strategies
        - Scalability testing with different document sizes
        - Cost analysis and efficiency evaluation
        - Resource monitoring (CPU, memory, processing time)
        - Optimization recommendations and insights
        
        **Benchmark Types**:
        - Speed Benchmark: Processing time and throughput
        - Memory Benchmark: Memory usage and efficiency
        - Scalability Benchmark: Performance with document size
        - Accuracy Benchmark: Quality and accuracy metrics
        - Cost Benchmark: Processing costs and efficiency
        - Resource Benchmark: System resource utilization
        """)
    
    st.markdown("---")
    
    # Quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    ### Getting Started:
    
    1. **Choose an Experiment**: Navigate to any of the numbered pages in the sidebar
    2. **Upload or Generate Data**: Use the data tools to create test datasets
    3. **Run Analysis**: Execute chunking strategies and compare results
    4. **Visualize Results**: Explore interactive charts and metrics
    5. **Export Findings**: Save results and generate reports
    
    ### Data Requirements:
    - JSON files with chunking results
    - PDF documents for processing
    - Text files for analysis
    - Sample data can be generated using the Data Generator tool
    
    ### Launch Command:
    ```bash
    streamlit run Home.py
    ```
    """)
    
    # Recent activity
    st.subheader("📊 Recent Activity")
    
    # Check for recent outputs
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        recent_files = list(outputs_dir.glob("*.json"))[-5:]  # Last 5 files
        if recent_files:
            st.markdown("**Recent Analysis Results:**")
            for file in recent_files:
                st.markdown(f"- {file.name}")
        else:
            st.markdown("*No recent analysis results found.*")
    else:
        st.markdown("*Outputs directory not found.*")
    
    # Technical information
    st.subheader("🔧 Technical Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dependencies**:
        - Streamlit: Web application framework
        - Plotly: Interactive visualizations
        - Pandas: Data manipulation and analysis
        - NumPy: Numerical computing
        - JSON: Data serialization
        
        **File Structure**:
        - `Home.py`: Main dashboard entry point
        - `pages/`: Multi-page structure with 8 experiment pages
        - `outputs/`: Analysis results and visualizations
        - `data/`: Sample data and test documents
        """)
    
    with col2:
        st.markdown("""
        **Key Features**:
        - Interactive parameter configuration
        - Real-time visualization updates
        - Comprehensive performance metrics
        - Export capabilities for results
        - Modular and extensible design
        
        **Navigation**:
        - Use the sidebar to switch between pages
        - Each page has specialized tools and controls
        - Results are automatically saved and can be loaded
        - Cross-page data sharing and analysis
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Chunking Analysis Dashboard | Data Engineering Team</p>
        <p>Use the sidebar to navigate between different experiments and tools.</p>
        <p>🚀 Launch with: <code>streamlit run Home.py</code></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 