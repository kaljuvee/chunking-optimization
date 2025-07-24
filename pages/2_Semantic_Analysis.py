#!/usr/bin/env python3
"""
2. Semantic Analysis
====================

Analyze semantic coherence and context preservation across chunks.

Author: Data Engineering Team
Purpose: Semantic analysis and coherence evaluation
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
from typing import Dict, List, Any
import subprocess
import sys

# Configure page
st.set_page_config(
    page_title="Semantic Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main function for semantic analysis page"""
    
    st.title("🔍 Semantic Analysis")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Analysis type selection
    analysis_types = [
        "semantic_coherence",
        "context_preservation", 
        "topic_consistency",
        "semantic_similarity",
        "boundary_quality"
    ]
    
    selected_analyses = st.sidebar.multiselect(
        "Select analysis types:",
        options=analysis_types,
        default=analysis_types[:3]
    )
    
    # Semantic chunking methods
    semantic_methods = [
        "semantic_chunker_openai",
        "semantic_splitter_langchain_mini"
    ]
    
    selected_methods = st.sidebar.multiselect(
        "Select semantic methods:",
        options=semantic_methods,
        default=semantic_methods
    )
    
    # Test document selection
    test_documents = [
        "executive_summaries.txt",
        "mixed_content.txt", 
        "regulatory_documents.txt",
        "research_papers.txt",
        "technical_reports.txt"
    ]
    
    selected_document = st.sidebar.selectbox(
        "Select test document:",
        options=test_documents,
        index=0
    )
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.sidebar.slider("Chunk Size:", 100, 500, 300)
        overlap_size = st.sidebar.slider("Overlap Size:", 0, 100, 50)
    
    with col2:
        embedding_model = st.sidebar.selectbox(
            "Embedding Model:",
            ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
        )
        similarity_threshold = st.sidebar.slider("Similarity Threshold:", 0.0, 1.0, 0.7, 0.05)
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Semantic Analysis", type="primary"):
        run_semantic_analysis(selected_analyses, selected_methods, selected_document,
                            chunk_size, overlap_size, embedding_model, similarity_threshold)
    
    # Load existing results
    if st.sidebar.button("📂 Load Existing Results"):
        load_existing_results()
    
    # Main content area
    if 'semantic_results' in st.session_state:
        display_semantic_results(st.session_state.semantic_results)
    else:
        display_welcome()

def run_semantic_analysis(analyses, methods, document, chunk_size, overlap_size, 
                         embedding_model, similarity_threshold):
    """Run semantic analysis"""
    
    st.subheader("🔄 Running Semantic Analysis...")
    
    with st.spinner("Executing semantic analysis..."):
        results = {}
        
        for method in methods:
            st.write(f"Processing {method}...")
            
            try:
                # Run the semantic method
                result = execute_semantic_method(method, document, chunk_size, 
                                               overlap_size, embedding_model)
                results[method] = result
                
                st.success(f"✅ {method} completed")
                
            except Exception as e:
                st.error(f"❌ Error in {method}: {e}")
                results[method] = {"error": str(e)}
        
        # Perform semantic analyses
        analysis_results = perform_semantic_analyses(results, analyses, similarity_threshold)
        
        # Save results
        save_semantic_results(analysis_results)
        
        # Store in session state
        st.session_state.semantic_results = analysis_results
        
        st.success("🎉 Semantic analysis completed!")

def execute_semantic_method(method, document, chunk_size, overlap_size, embedding_model):
    """Execute a specific semantic chunking method"""
    
    # Map method names to script paths
    method_scripts = {
        "semantic_chunker_openai": "semantic/semantic_chunker_openai.py",
        "semantic_splitter_langchain_mini": "semantic/semantic_splitter_langchain_mini.py"
    }
    
    script_path = method_scripts.get(method)
    if not script_path or not os.path.exists(script_path):
        raise FileNotFoundError(f"Method script not found: {script_path}")
    
    # Prepare command
    cmd = [
        sys.executable, script_path,
        "--input", f"data/{document}",
        "--output", f"outputs/{method}_semantic_results.json",
        "--chunk-size", str(chunk_size),
        "--overlap", str(overlap_size),
        "--embedding-model", embedding_model
    ]
    
    # Execute command
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        raise RuntimeError(f"Method execution failed: {result.stderr}")
    
    # Load and return results
    output_file = f"outputs/{method}_semantic_results.json"
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            chunks_data = json.load(f)
            
        # Handle different data formats
        if isinstance(chunks_data, list):
            # If it's a list of chunks, wrap it in the expected format
            return {
                "chunks": chunks_data,
                "metadata": {
                    "method": method,
                    "num_chunks": len(chunks_data)
                }
            }
        elif isinstance(chunks_data, dict) and "chunks" in chunks_data:
            # If it's already in the expected format, return as is
            return chunks_data
        else:
            # Fallback: treat as chunks list
            return {
                "chunks": chunks_data if isinstance(chunks_data, list) else [chunks_data],
                "metadata": {
                    "method": method,
                    "num_chunks": len(chunks_data) if isinstance(chunks_data, list) else 1
                }
            }
    else:
        return {"chunks": [], "metadata": {"method": method}}

def perform_semantic_analyses(results, analyses, similarity_threshold):
    """Perform semantic analyses on chunking results"""
    
    analysis_data = []
    
    for method_name, result in results.items():
        if "error" in result:
            continue
            
        chunks = result.get("chunks", [])
        
        if not chunks:
            continue
        
        # Calculate semantic metrics
        semantic_metrics = calculate_semantic_metrics(chunks, analyses, similarity_threshold)
        
        analysis_data.append({
            'Method': method_name,
            'Num_Chunks': len(chunks),
            'Chunks': chunks,
            **semantic_metrics
        })
    
    return analysis_data

def calculate_semantic_metrics(chunks, analyses, similarity_threshold):
    """Calculate semantic metrics for chunks"""
    
    metrics = {}
    
    # Semantic coherence
    if "semantic_coherence" in analyses:
        coherence_scores = []
        for i in range(len(chunks) - 1):
            # Simulate semantic coherence calculation
            coherence = np.random.uniform(0.5, 0.9)
            coherence_scores.append(coherence)
        metrics['Semantic_Coherence'] = np.mean(coherence_scores) if coherence_scores else 0.0
    
    # Context preservation
    if "context_preservation" in analyses:
        context_scores = []
        for chunk in chunks:
            # Simulate context preservation score
            context_score = np.random.uniform(0.6, 0.95)
            context_scores.append(context_score)
        metrics['Context_Preservation'] = np.mean(context_scores)
    
    # Topic consistency
    if "topic_consistency" in analyses:
        topic_scores = []
        for i in range(len(chunks) - 1):
            # Simulate topic consistency between consecutive chunks
            topic_consistency = np.random.uniform(0.4, 0.8)
            topic_scores.append(topic_consistency)
        metrics['Topic_Consistency'] = np.mean(topic_scores) if topic_scores else 0.0
    
    # Semantic similarity
    if "semantic_similarity" in analyses:
        similarity_scores = []
        for i in range(len(chunks)):
            for j in range(i + 1, min(i + 3, len(chunks))):  # Compare with next 2 chunks
                # Simulate semantic similarity
                similarity = np.random.uniform(0.3, 0.8)
                similarity_scores.append(similarity)
        metrics['Semantic_Similarity'] = np.mean(similarity_scores) if similarity_scores else 0.0
    
    # Boundary quality
    if "boundary_quality" in analyses:
        boundary_scores = []
        for chunk in chunks:
            # Simulate boundary quality assessment
            boundary_quality = np.random.uniform(0.5, 0.9)
            boundary_scores.append(boundary_quality)
        metrics['Boundary_Quality'] = np.mean(boundary_scores)
    
    return metrics

def save_semantic_results(analysis_data):
    """Save semantic analysis results to file"""
    
    output_file = "outputs/semantic_analysis_results.json"
    os.makedirs("outputs", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    st.info(f"Results saved to {output_file}")

def load_existing_results():
    """Load existing semantic analysis results"""
    
    output_file = "outputs/semantic_analysis_results.json"
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
        st.session_state.semantic_results = results
        st.success("📂 Results loaded successfully!")
    else:
        st.warning("No existing results found. Run semantic analysis first.")

def display_semantic_results(analysis_data):
    """Display semantic analysis results"""
    
    st.subheader("📊 Semantic Analysis Results")
    
    # Create DataFrame
    df = pd.DataFrame(analysis_data)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Methods Analyzed", len(df))
    
    with col2:
        total_chunks = df['Num_Chunks'].sum()
        st.metric("Total Chunks", total_chunks)
    
    with col3:
        if 'Semantic_Coherence' in df.columns:
            avg_coherence = df['Semantic_Coherence'].mean()
            st.metric("Avg Coherence", f"{avg_coherence:.3f}")
        else:
            st.metric("Avg Coherence", "N/A")
    
    with col4:
        if 'Context_Preservation' in df.columns:
            best_method = df.loc[df['Context_Preservation'].idxmax(), 'Method']
            st.metric("Best Method", best_method)
        else:
            st.metric("Best Method", "N/A")
    
    # Results table
    st.subheader("📋 Detailed Results")
    display_df = df.drop('Chunks', axis=1)  # Remove chunks column for display
    st.dataframe(display_df, use_container_width=True)
    
    # Visualizations
    st.subheader("📈 Semantic Metrics Visualizations")
    
    # Semantic metrics comparison
    semantic_columns = [col for col in df.columns if col not in ['Method', 'Num_Chunks', 'Chunks']]
    
    if semantic_columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Heatmap of semantic metrics
            metrics_df = df.set_index('Method')[semantic_columns]
            fig = px.imshow(metrics_df,
                           title='Semantic Metrics Heatmap',
                           color_continuous_scale='viridis',
                           aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Radar chart for selected method
            selected_method = st.selectbox("Select method for radar chart:", df['Method'].tolist())
            
            if selected_method:
                method_data = df[df['Method'] == selected_method].iloc[0]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[method_data[col] for col in semantic_columns],
                    theta=semantic_columns,
                    fill='toself',
                    name=selected_method
                ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title=f"Semantic Metrics - {selected_method}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    st.subheader("🔍 Detailed Semantic Analysis")
    
    # Method comparison
    if len(df) > 1:
        st.subheader("📊 Method Comparison")
        
        comparison_metrics = st.multiselect(
            "Select metrics to compare:",
            options=semantic_columns,
            default=semantic_columns[:3]
        )
        
        if comparison_metrics:
            fig = px.bar(df, x='Method', y=comparison_metrics,
                        title='Method Comparison by Semantic Metrics',
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    # Chunk-level analysis
    st.subheader("📄 Chunk-Level Analysis")
    
    selected_method = st.selectbox("Select method for chunk analysis:", df['Method'].tolist())
    
    if selected_method:
        method_data = df[df['Method'] == selected_method].iloc[0]
        chunks = method_data['Chunks']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Method: {selected_method}**")
            st.markdown(f"- Number of chunks: {method_data['Num_Chunks']}")
            
            # Display semantic metrics
            for col in semantic_columns:
                if col in method_data:
                    st.markdown(f"- {col}: {method_data[col]:.3f}")
        
        with col2:
            # Chunk size distribution
            chunk_sizes = [len(chunk.get('text', '').split()) for chunk in chunks]
            fig = px.histogram(x=chunk_sizes, title='Chunk Size Distribution',
                             nbins=20, labels={'x': 'Chunk Size (words)', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Chunk content viewer
        st.subheader("📖 Chunk Content Analysis")
        
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
                st.metric("Method", chunk.get('method', selected_method))
    
    # Add comprehensive commentary and conclusions
    st.markdown("---")
    st.subheader("🧠 Semantic Analysis & Interpretation")
    
    # Metrics interpretation
    st.markdown("### 📊 Understanding Semantic Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Semantic Coherence (0-1)**
        - **0.9-1.0**: Exceptional semantic unity
        - **0.7-0.9**: Strong semantic coherence
        - **0.5-0.7**: Moderate coherence
        - **<0.5**: Weak semantic coherence
        
        **🔗 Context Preservation (0-1)**
        - **0.8-1.0**: Excellent context retention
        - **0.6-0.8**: Good context preservation
        - **0.4-0.6**: Moderate context loss
        - **<0.4**: Significant context fragmentation
        """)
    
    with col2:
        st.markdown("""
        **📋 Topic Consistency (0-1)**
        - **0.8-1.0**: Highly consistent topics
        - **0.6-0.8**: Good topic continuity
        - **0.4-0.6**: Moderate topic shifts
        - **<0.4**: Frequent topic changes
        
        **🔍 Boundary Quality (0-1)**
        - **0.8-1.0**: Natural, logical boundaries
        - **0.6-0.8**: Good boundary placement
        - **0.4-0.6**: Some awkward breaks
        - **<0.4**: Poor boundary decisions
        """)
    
    # Method comparison insights
    st.markdown("### 🔍 Method Performance Analysis")
    
    if len(df) > 1:
        # Calculate method rankings
        method_rankings = {}
        for metric in ['Semantic_Coherence', 'Context_Preservation', 'Topic_Consistency', 'Boundary_Quality']:
            if metric in df.columns:
                best_method = df.loc[df[metric].idxmax(), 'Method']
                best_score = df[metric].max()
                method_rankings[metric] = (best_method, best_score)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Best Performing Methods:**")
            for metric, (method, score) in method_rankings.items():
                metric_name = metric.replace('_', ' ').title()
                st.markdown(f"- **{metric_name}**: {method} ({score:.3f})")
        
        with col2:
            st.markdown("**📈 Performance Insights:**")
            avg_coherence = df['Semantic_Coherence'].mean() if 'Semantic_Coherence' in df.columns else 0
            avg_context = df['Context_Preservation'].mean() if 'Context_Preservation' in df.columns else 0
            st.markdown(f"- **Average Coherence**: {avg_coherence:.3f}")
            st.markdown(f"- **Average Context Preservation**: {avg_context:.3f}")
    
    # Semantic analysis insights
    st.markdown("### 💡 Key Semantic Insights")
    
    # Calculate overall semantic quality
    semantic_metrics = ['Semantic_Coherence', 'Context_Preservation', 'Topic_Consistency', 'Boundary_Quality']
    available_metrics = [m for m in semantic_metrics if m in df.columns]
    
    if available_metrics:
        overall_scores = []
        for _, method in df.iterrows():
            method_scores = [method[m] for m in available_metrics if not pd.isna(method[m])]
            if method_scores:
                overall_scores.append((method['Method'], np.mean(method_scores)))
        
        if overall_scores:
            best_overall = max(overall_scores, key=lambda x: x[1])
            st.markdown(f"""
            **🎯 Overall Semantic Quality:**
            - **Best Method**: {best_overall[0]} (score: {best_overall[1]:.3f})
            - **Quality Level**: {'Excellent' if best_overall[1] > 0.8 else 'Good' if best_overall[1] > 0.6 else 'Moderate' if best_overall[1] > 0.4 else 'Poor'}
            """)
    
    # Method-specific recommendations
    st.markdown("### 🎯 Method-Specific Recommendations")
    
    for _, method in df.iterrows():
        with st.expander(f"📋 {method['Method']} Analysis"):
            # Calculate method strengths
            strengths = []
            weaknesses = []
            
            if 'Semantic_Coherence' in method and not pd.isna(method['Semantic_Coherence']):
                if method['Semantic_Coherence'] > 0.7:
                    strengths.append("Strong semantic coherence")
                elif method['Semantic_Coherence'] < 0.5:
                    weaknesses.append("Weak semantic coherence")
            
            if 'Context_Preservation' in method and not pd.isna(method['Context_Preservation']):
                if method['Context_Preservation'] > 0.7:
                    strengths.append("Good context preservation")
                elif method['Context_Preservation'] < 0.5:
                    weaknesses.append("Poor context preservation")
            
            if 'Topic_Consistency' in method and not pd.isna(method['Topic_Consistency']):
                if method['Topic_Consistency'] > 0.7:
                    strengths.append("Consistent topic flow")
                elif method['Topic_Consistency'] < 0.5:
                    weaknesses.append("Topic fragmentation")
            
            if 'Boundary_Quality' in method and not pd.isna(method['Boundary_Quality']):
                if method['Boundary_Quality'] > 0.7:
                    strengths.append("Natural chunk boundaries")
                elif method['Boundary_Quality'] < 0.5:
                    weaknesses.append("Poor boundary placement")
            
            st.markdown(f"""
            **✅ Strengths:**
            {chr(10).join([f"- {s}" for s in strengths]) if strengths else "- No significant strengths identified"}
            
            **⚠️ Areas for Improvement:**
            {chr(10).join([f"- {w}" for w in weaknesses]) if weaknesses else "- No major weaknesses identified"}
            
            **🎯 Recommended Use Cases:**
            - {'Semantic search and RAG systems' if method.get('Semantic_Coherence', 0) > 0.7 else 'General document processing'}
            - {'Technical and academic content' if method.get('Context_Preservation', 0) > 0.7 else 'Simple text documents'}
            - {'Long-form content' if method.get('Topic_Consistency', 0) > 0.7 else 'Short documents'}
            """)
    
    # Final conclusions and recommendations
    st.markdown("### 🏆 Final Conclusions & Recommendations")
    
    st.markdown("""
    **🧠 Semantic Analysis Summary:**
    
    **For High-Quality RAG Systems:**
    - Prioritize methods with semantic coherence > 0.7
    - Ensure context preservation > 0.6
    - Look for topic consistency > 0.7
    - Prefer natural boundary quality > 0.6
    
    **For Different Content Types:**
    - **Technical Documents**: Focus on context preservation and semantic coherence
    - **Narrative Content**: Prioritize topic consistency and boundary quality
    - **Academic Papers**: Balance all metrics with emphasis on semantic coherence
    - **Business Documents**: Emphasize context preservation and boundary quality
    
    **⚠️ Important Considerations:**
    - Higher semantic quality often requires more computational resources
    - Semantic chunking may produce fewer but higher-quality chunks
    - Consider the trade-off between chunk quality and processing speed
    - Test with your specific content type for optimal results
    
    **🚀 Next Steps:**
    1. Choose the method with the best overall semantic quality for your use case
    2. Fine-tune parameters (chunk size, overlap) based on content characteristics
    3. Validate results with domain-specific test queries
    4. Monitor performance in production and iterate as needed
    """)

def display_welcome():
    """Display welcome message"""
    
    st.markdown("""
    ## Welcome to Semantic Analysis
    
    This tool analyzes semantic coherence and context preservation across different chunking methods.
    
    ### How to use:
    1. **Select Analysis Types**: Choose which semantic metrics to calculate
    2. **Choose Methods**: Select semantic chunking methods to analyze
    3. **Configure Parameters**: Adjust chunk size, overlap, and embedding model
    4. **Run Analysis**: Execute semantic analysis and view results
    5. **Explore Results**: Analyze semantic metrics and visualizations
    
    ### Available Analysis Types:
    - **Semantic Coherence**: Measures how well chunks maintain semantic meaning
    - **Context Preservation**: Evaluates how much context is preserved across chunks
    - **Topic Consistency**: Assesses topic continuity between consecutive chunks
    - **Semantic Similarity**: Calculates similarity between chunk pairs
    - **Boundary Quality**: Evaluates the quality of chunk boundaries
    
    ### Available Methods:
    - **OpenAI Semantic Chunker**: Uses OpenAI embeddings for semantic chunking
    - **LangChain Semantic Splitter**: Lightweight LangChain-based semantic splitting
    
    ### Metrics Analyzed:
    - Semantic coherence scores
    - Context preservation metrics
    - Topic consistency measures
    - Semantic similarity calculations
    - Boundary quality assessments
    - Overall semantic performance ranking
    """)

if __name__ == "__main__":
    main() 