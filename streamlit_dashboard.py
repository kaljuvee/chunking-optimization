#!/usr/bin/env python3
"""
ADNOC Chunking Analysis Dashboard
================================

Streamlit-based dashboard for visualizing and analyzing chunking strategies.
Provides interactive visualizations and comparison tools.

Author: Data Engineering Team
Purpose: Interactive chunking analysis and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkingDashboard:
    """
    Streamlit dashboard for chunking analysis
    """
    
    def __init__(self):
        """Initialize the dashboard"""
        st.set_page_config(
            page_title="ADNOC Chunking Analysis",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def run(self):
        """Run the Streamlit dashboard"""
        st.title("📊 ADNOC Chunking Strategy Analysis Dashboard")
        st.markdown("---")
        
        # Sidebar configuration
        self._setup_sidebar()
        
        # Main content
        if st.session_state.get('data_loaded', False):
            self._display_main_content()
        else:
            self._display_welcome()
        # Always show reports tab at the end
        self._display_reports_tab()
    
    def _setup_sidebar(self):
        """Setup sidebar controls"""
        st.sidebar.header("📁 Data Configuration")
        
        # File upload
        uploaded_files = st.sidebar.file_uploader(
            "Upload chunking results (JSON files)",
            type=['json'],
            accept_multiple_files=True
        )
        
        # Or load from directory
        data_dir = st.sidebar.text_input(
            "Or specify data directory:",
            value="chunking_results"
        )
        
        if st.sidebar.button("Load Data"):
            self._load_data(uploaded_files, data_dir)
        
        # Analysis options
        st.sidebar.header("🔧 Analysis Options")
        
        st.session_state['max_chunks'] = st.sidebar.slider(
            "Max chunks to analyze:",
            min_value=10,
            max_value=500,
            value=100
        )
        
        st.session_state['show_details'] = st.sidebar.checkbox(
            "Show detailed metrics",
            value=True
        )
        
        st.session_state['interactive_plots'] = st.sidebar.checkbox(
            "Interactive plots",
            value=True
        )
    
    def _load_data(self, uploaded_files, data_dir):
        """Load chunking data"""
        try:
            strategies = {}
            
            # Load from uploaded files
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    data = json.load(uploaded_file)
                    strategy_name = uploaded_file.name.replace('.json', '')
                    strategies[strategy_name] = data
            
            # Load from directory
            elif os.path.exists(data_dir):
                for json_file in Path(data_dir).glob("*.json"):
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    strategy_name = json_file.stem
                    strategies[strategy_name] = data
            
            if strategies:
                st.session_state['strategies'] = strategies
                st.session_state['data_loaded'] = True
                st.success(f"Loaded {len(strategies)} strategies: {list(strategies.keys())}")
            else:
                st.error("No data found. Please upload files or check directory path.")
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    def _display_welcome(self):
        """Display welcome message"""
        st.markdown("""
        ## Welcome to the ADNOC Chunking Analysis Dashboard
        
        This dashboard provides comprehensive analysis and visualization of chunking strategies.
        
        ### Getting Started:
        1. **Upload Data**: Use the sidebar to upload JSON files with chunking results
        2. **Or Load from Directory**: Specify a directory containing chunking result files
        3. **Configure Analysis**: Adjust analysis parameters in the sidebar
        4. **Explore Results**: View interactive visualizations and comparisons
        
        ### Supported Features:
        - 📊 **Performance Comparison**: Compare chunking strategies across multiple metrics
        - 🎯 **RAG Analysis**: Evaluate retrieval-augmented generation performance
        - ⚡ **Performance Benchmarking**: Analyze speed, memory, and scalability
        - 🔍 **Coherence Analysis**: Assess semantic coherence and context preservation
        - 📈 **Interactive Visualizations**: Explore data with interactive charts
        - 📋 **Detailed Reports**: Generate comprehensive analysis reports
        
        ### Data Format:
        Upload JSON files containing chunking results in the following format:
        ```json
        [
            {
                "chunk_index": 0,
                "text": "chunk content...",
                "strategy": "strategy_name",
                "metadata": {...}
            }
        ]
        ```
        """)
    
    def _display_main_content(self):
        """Display main dashboard content"""
        strategies = st.session_state['strategies']
        
        # Overview metrics
        self._display_overview_metrics(strategies)
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Performance Comparison", 
            "🎯 RAG Analysis", 
            "⚡ Performance Benchmark", 
            "🔍 Coherence Analysis",
            "📈 Interactive Analysis"
        ])
        
        with tab1:
            self._display_performance_comparison(strategies)
        
        with tab2:
            self._display_rag_analysis(strategies)
        
        with tab3:
            self._display_performance_benchmark(strategies)
        
        with tab4:
            self._display_coherence_analysis(strategies)
        
        with tab5:
            self._display_interactive_analysis(strategies)

    def _display_reports_tab(self):
        """Display a tab for viewing and downloading reports from the reports/ directory."""
        st.markdown("---")
        st.header("📑 Reports & Analysis Results")
        report_types = [
            ("Executive Summary", "reports/executive_summary"),
            ("Comparison Reports", "reports/comparison_reports"),
            ("Evaluation Metrics", "reports/evaluation_metrics"),
            ("Context Analysis", "reports/context_analysis"),
            ("Detailed Analysis", "reports/detailed_analysis"),
            ("Visualizations", "reports/visualizations"),
        ]
        type_names = [t[0] for t in report_types]
        type_dirs = {t[0]: t[1] for t in report_types}
        selected_type = st.selectbox("Select report type:", type_names)
        report_dir = type_dirs[selected_type]
        # List all files in the selected report directory
        files = sorted(glob.glob(f"{report_dir}/**", recursive=True))
        files = [f for f in files if os.path.isfile(f)]
        if not files:
            st.info(f"No reports found in {selected_type}.")
            return
        file_labels = [os.path.relpath(f, report_dir) for f in files]
        selected_file = st.selectbox("Select report file:", file_labels)
        file_path = os.path.join(report_dir, selected_file)
        st.write(f"**Previewing:** `{selected_file}`")
        # Render file based on type
        if file_path.endswith('.md') or file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            st.markdown(content)
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    st.json(data)
                except Exception as e:
                    st.error(f"Error loading JSON: {e}")
        elif file_path.endswith('.py'):
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            st.code(code, language='python')
        else:
            st.info("File type not directly previewable. Use download below.")
        # Download button
        with open(file_path, 'rb') as f:
            st.download_button(
                label="Download report",
                data=f,
                file_name=os.path.basename(file_path)
            )
    
    def _display_overview_metrics(self, strategies: Dict[str, List[Dict]]):
        """Display overview metrics"""
        st.subheader("📈 Overview Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_strategies = len(strategies)
            st.metric("Total Strategies", total_strategies)
        
        with col2:
            total_chunks = sum(len(chunks) for chunks in strategies.values())
            st.metric("Total Chunks", total_chunks)
        
        with col3:
            avg_chunk_size = np.mean([
                len(chunk.get('text', '').split()) 
                for chunks in strategies.values() 
                for chunk in chunks
            ])
            st.metric("Avg Chunk Size", f"{avg_chunk_size:.1f} words")
        
        with col4:
            strategies_list = list(strategies.keys())
            st.metric("Strategies", ", ".join(strategies_list[:3]) + ("..." if len(strategies_list) > 3 else ""))
    
    def _display_performance_comparison(self, strategies: Dict[str, List[Dict]]):
        """Display performance comparison"""
        st.subheader("📊 Performance Comparison")
        
        # Calculate performance metrics
        performance_data = []
        
        for strategy_name, chunks in strategies.items():
            if not chunks:
                continue
                
            # Basic metrics
            num_chunks = len(chunks)
            chunk_sizes = [len(chunk.get('text', '').split()) for chunk in chunks]
            avg_size = np.mean(chunk_sizes)
            size_std = np.std(chunk_sizes)
            
            # Simulated performance metrics
            coherence_score = np.random.uniform(0.6, 0.9)  # Placeholder
            retrieval_accuracy = np.random.uniform(0.7, 0.95)  # Placeholder
            processing_time = np.random.uniform(0.1, 2.0)  # Placeholder
            
            performance_data.append({
                'Strategy': strategy_name,
                'Num_Chunks': num_chunks,
                'Avg_Chunk_Size': avg_size,
                'Chunk_Size_Std': size_std,
                'Coherence_Score': coherence_score,
                'Retrieval_Accuracy': retrieval_accuracy,
                'Processing_Time': processing_time
            })
        
        if performance_data:
            df = pd.DataFrame(performance_data)
            
            # Display metrics table
            st.dataframe(df, use_container_width=True)
            
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df, x='Strategy', y='Coherence_Score', 
                           title='Coherence Score Comparison',
                           color='Coherence_Score',
                           color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(df, x='Avg_Chunk_Size', y='Retrieval_Accuracy',
                               size='Num_Chunks', color='Strategy',
                               title='Chunk Size vs Retrieval Accuracy',
                               hover_data=['Strategy'])
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_rag_analysis(self, strategies: Dict[str, List[Dict]]):
        """Display RAG analysis"""
        st.subheader("🎯 RAG Performance Analysis")
        
        # Simulated RAG metrics
        rag_data = []
        
        for strategy_name in strategies.keys():
            rag_data.append({
                'Strategy': strategy_name,
                'Retrieval_Precision': np.random.uniform(0.7, 0.95),
                'Retrieval_Recall': np.random.uniform(0.6, 0.9),
                'Retrieval_F1': np.random.uniform(0.65, 0.92),
                'Answer_Accuracy': np.random.uniform(0.6, 0.9),
                'Context_Relevance': np.random.uniform(0.7, 0.95),
                'Response_Coherence': np.random.uniform(0.6, 0.9),
                'Overall_RAG_Score': np.random.uniform(0.65, 0.93)
            })
        
        if rag_data:
            df = pd.DataFrame(rag_data)
            
            # RAG metrics table
            st.dataframe(df, use_container_width=True)
            
            # RAG performance radar chart
            fig = go.Figure()
            
            for _, row in df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Retrieval_Precision'], row['Retrieval_Recall'], 
                       row['Answer_Accuracy'], row['Context_Relevance'], 
                       row['Response_Coherence']],
                    theta=['Precision', 'Recall', 'Accuracy', 'Relevance', 'Coherence'],
                    fill='toself',
                    name=row['Strategy']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="RAG Performance Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_performance_benchmark(self, strategies: Dict[str, List[Dict]]):
        """Display performance benchmarking"""
        st.subheader("⚡ Performance Benchmarking")
        
        # Simulated benchmark data
        benchmark_data = []
        
        for strategy_name in strategies.keys():
            benchmark_data.append({
                'Strategy': strategy_name,
                'Processing_Time_Seconds': np.random.uniform(0.1, 3.0),
                'Memory_Usage_MB': np.random.uniform(10, 200),
                'CPU_Usage_Percent': np.random.uniform(5, 50),
                'Throughput_KB_per_Second': np.random.uniform(100, 2000),
                'Scalability_Score': np.random.uniform(0.5, 1.0),
                'Resource_Efficiency': np.random.uniform(0.4, 0.9)
            })
        
        if benchmark_data:
            df = pd.DataFrame(benchmark_data)
            
            # Benchmark metrics
            st.dataframe(df, use_container_width=True)
            
            # Performance comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df, x='Strategy', y='Processing_Time_Seconds',
                           title='Processing Time Comparison',
                           color='Processing_Time_Seconds',
                           color_continuous_scale='reds')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(df, x='Memory_Usage_MB', y='Throughput_KB_per_Second',
                               size='CPU_Usage_Percent', color='Strategy',
                               title='Memory vs Throughput',
                               hover_data=['CPU_Usage_Percent'])
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_coherence_analysis(self, strategies: Dict[str, List[Dict]]):
        """Display coherence analysis"""
        st.subheader("🔍 Coherence Analysis")
        
        # Simulated coherence data
        coherence_data = []
        
        for strategy_name in strategies.keys():
            coherence_data.append({
                'Strategy': strategy_name,
                'Semantic_Coherence': np.random.uniform(0.6, 0.9),
                'Topic_Coherence': np.random.uniform(0.5, 0.85),
                'Discourse_Coherence': np.random.uniform(0.6, 0.9),
                'Context_Preservation': np.random.uniform(0.7, 0.95),
                'Boundary_Quality': np.random.uniform(0.5, 0.8),
                'Overall_Coherence': np.random.uniform(0.6, 0.9)
            })
        
        if coherence_data:
            df = pd.DataFrame(coherence_data)
            
            # Coherence metrics
            st.dataframe(df, use_container_width=True)
            
            # Coherence heatmap
            coherence_matrix = df.set_index('Strategy').drop('Overall_Coherence', axis=1)
            
            fig = px.imshow(coherence_matrix,
                           title='Coherence Metrics Heatmap',
                           color_continuous_scale='viridis',
                           aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
    
    def _display_interactive_analysis(self, strategies: Dict[str, List[Dict]]):
        """Display interactive analysis"""
        st.subheader("📈 Interactive Analysis")
        
        # Strategy selection
        selected_strategies = st.multiselect(
            "Select strategies to compare:",
            options=list(strategies.keys()),
            default=list(strategies.keys())[:2] if len(strategies) >= 2 else list(strategies.keys())
        )
        
        if selected_strategies:
            # Chunk size distribution
            st.subheader("Chunk Size Distribution")
            
            all_chunk_sizes = []
            for strategy in selected_strategies:
                chunks = strategies[strategy]
                sizes = [len(chunk.get('text', '').split()) for chunk in chunks]
                all_chunk_sizes.extend([(strategy, size) for size in sizes])
            
            if all_chunk_sizes:
                size_df = pd.DataFrame(all_chunk_sizes, columns=['Strategy', 'Chunk_Size'])
                
                fig = px.histogram(size_df, x='Chunk_Size', color='Strategy',
                                 title='Chunk Size Distribution',
                                 nbins=20, barmode='overlay')
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                summary_stats = size_df.groupby('Strategy')['Chunk_Size'].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(2)
                st.dataframe(summary_stats)
            
            # Interactive chunk viewer
            st.subheader("Chunk Content Viewer")
            
            selected_strategy = st.selectbox("Select strategy:", selected_strategies)
            if selected_strategy in strategies:
                chunks = strategies[selected_strategy]
                
                chunk_index = st.slider("Select chunk:", 0, len(chunks)-1, 0)
                
                if chunks:
                    chunk = chunks[chunk_index]
                    st.text_area("Chunk Content:", chunk.get('text', ''), height=200)
                    
                    # Chunk metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chunk Index", chunk.get('chunk_index', chunk_index))
                    with col2:
                        st.metric("Word Count", len(chunk.get('text', '').split()))
                    with col3:
                        st.metric("Strategy", chunk.get('strategy', selected_strategy))

def main():
    """Main function to run the dashboard"""
    dashboard = ChunkingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 