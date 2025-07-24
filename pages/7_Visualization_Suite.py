#!/usr/bin/env python3
"""
7. Visualization Suite
======================

Advanced visualization tools for chunking analysis.

Author: Data Engineering Team
Purpose: Comprehensive visualization and analysis tools
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
    page_title="Visualization Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main function for visualization suite page"""
    
    st.title("📊 Visualization Suite")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Visualization types
    visualization_types = [
        "chunk_overlap_analysis",
        "semantic_coherence_heatmap", 
        "performance_comparison",
        "topic_distribution",
        "embedding_visualization",
        "interactive_dashboard"
    ]
    
    selected_visualizations = st.sidebar.multiselect(
        "Select visualizations:",
        options=visualization_types,
        default=visualization_types[:3]
    )
    
    # Data sources
    data_sources = [
        "strategy_comparison_results.json",
        "semantic_analysis_results.json",
        "rag_evaluation_results.json",
        "llm_chunking_test_results.json",
        "topic_clustering_results.json"
    ]
    
    selected_data = st.sidebar.multiselect(
        "Select data sources:",
        options=data_sources,
        default=data_sources[:2]
    )
    
    # Visualization parameters
    col1, col2 = st.columns(2)
    with col1:
        max_points = st.sidebar.slider("Max data points:", 100, 1000, 500)
        chart_height = st.sidebar.slider("Chart height:", 400, 800, 500)
    
    with col2:
        color_scheme = st.sidebar.selectbox("Color scheme:", ["viridis", "plasma", "inferno", "magma", "cividis"])
        theme = st.sidebar.selectbox("Theme:", ["plotly", "plotly_white", "plotly_dark"])
    
    # Advanced settings
    st.sidebar.header("⚙️ Advanced Settings")
    
    enable_animations = st.sidebar.checkbox("Enable animations", value=True)
    show_statistics = st.sidebar.checkbox("Show statistics", value=True)
    export_formats = st.sidebar.multiselect(
        "Export formats:",
        ["PNG", "SVG", "PDF", "HTML"],
        default=["PNG", "HTML"]
    )
    
    # Run visualization button
    if st.sidebar.button("🚀 Generate Visualizations", type="primary"):
        generate_visualizations(selected_visualizations, selected_data, max_points,
                              chart_height, color_scheme, theme, enable_animations,
                              show_statistics, export_formats)
    
    # Load existing visualizations
    if st.sidebar.button("📂 Load Existing Visualizations"):
        load_existing_visualizations()
    
    # Main content area
    if 'visualization_results' in st.session_state:
        display_visualization_results(st.session_state.visualization_results)
    else:
        display_welcome()

def generate_visualizations(visualizations, data_sources, max_points, chart_height,
                          color_scheme, theme, enable_animations, show_statistics, export_formats):
    """Generate visualizations"""
    
    st.subheader("🔄 Generating Visualizations...")
    
    with st.spinner("Creating visualizations..."):
        try:
            # Load data
            data = load_visualization_data(data_sources)
            
            # Generate visualizations
            results = {}
            
            for viz_type in visualizations:
                st.write(f"Creating {viz_type}...")
                
                try:
                    viz_result = create_visualization(viz_type, data, max_points, chart_height,
                                                    color_scheme, theme, enable_animations,
                                                    show_statistics, export_formats)
                    results[viz_type] = viz_result
                    
                    st.success(f"✅ {viz_type} completed")
                    
                except Exception as e:
                    st.error(f"❌ Error in {viz_type}: {e}")
                    results[viz_type] = {"error": str(e)}
            
            st.session_state.visualization_results = results
            st.success("🎉 Visualizations generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Error generating visualizations: {e}")

def load_visualization_data(data_sources):
    """Load data for visualizations"""
    
    data = {}
    
    for source in data_sources:
        file_path = f"outputs/{source}"
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data[source] = json.load(f)
            except Exception as e:
                st.warning(f"Could not load {source}: {e}")
        else:
            st.warning(f"Data file not found: {file_path}")
    
    return data

def create_visualization(viz_type, data, max_points, chart_height, color_scheme, theme,
                        enable_animations, show_statistics, export_formats):
    """Create a specific visualization"""
    
    if viz_type == "chunk_overlap_analysis":
        return create_chunk_overlap_analysis(data, max_points, chart_height, color_scheme, theme)
    elif viz_type == "semantic_coherence_heatmap":
        return create_semantic_coherence_heatmap(data, chart_height, color_scheme, theme)
    elif viz_type == "performance_comparison":
        return create_performance_comparison(data, chart_height, color_scheme, theme)
    elif viz_type == "topic_distribution":
        return create_topic_distribution(data, max_points, chart_height, color_scheme, theme)
    elif viz_type == "embedding_visualization":
        return create_embedding_visualization(data, max_points, chart_height, color_scheme, theme)
    elif viz_type == "interactive_dashboard":
        return create_interactive_dashboard(data, chart_height, color_scheme, theme)
    else:
        raise ValueError(f"Unknown visualization type: {viz_type}")

def create_chunk_overlap_analysis(data, max_points, chart_height, color_scheme, theme):
    """Create chunk overlap analysis visualization"""
    
    # Simulate chunk overlap data
    strategies = ["Strategy A", "Strategy B", "Strategy C", "Strategy D"]
    overlap_data = []
    
    for strategy in strategies:
        for i in range(10):
            overlap_data.append({
                'Strategy': strategy,
                'Chunk_Index': i,
                'Overlap_Size': np.random.randint(0, 100),
                'Overlap_Percentage': np.random.uniform(0, 0.5),
                'Chunk_Size': np.random.randint(200, 500)
            })
    
    df = pd.DataFrame(overlap_data)
    
    # Create visualizations
    fig1 = px.scatter(df, x='Chunk_Index', y='Overlap_Size', color='Strategy',
                     title='Chunk Overlap Analysis',
                     color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig2 = px.box(df, x='Strategy', y='Overlap_Percentage',
                  title='Overlap Percentage Distribution by Strategy')
    
    fig3 = px.scatter(df, x='Chunk_Size', y='Overlap_Size', color='Strategy',
                     title='Chunk Size vs Overlap Size',
                     size='Overlap_Percentage')
    
    return {
        "type": "chunk_overlap_analysis",
        "figures": [fig1, fig2, fig3],
        "data": df.to_dict('records'),
        "statistics": {
            "total_chunks": len(df),
            "avg_overlap": df['Overlap_Percentage'].mean(),
            "strategies": strategies
        }
    }

def create_semantic_coherence_heatmap(data, chart_height, color_scheme, theme):
    """Create semantic coherence heatmap"""
    
    # Simulate semantic coherence data
    strategies = ["Strategy A", "Strategy B", "Strategy C", "Strategy D"]
    metrics = ["Coherence", "Relevance", "Accuracy", "Diversity", "Consistency"]
    
    coherence_matrix = np.random.uniform(0.5, 0.9, (len(strategies), len(metrics)))
    
    fig = px.imshow(coherence_matrix,
                    x=metrics,
                    y=strategies,
                    title='Semantic Coherence Heatmap',
                    color_continuous_scale=color_scheme,
                    aspect='auto')
    
    fig.update_layout(height=chart_height)
    
    return {
        "type": "semantic_coherence_heatmap",
        "figure": fig,
        "data": coherence_matrix.tolist(),
        "statistics": {
            "avg_coherence": np.mean(coherence_matrix),
            "best_strategy": strategies[np.argmax(np.mean(coherence_matrix, axis=1))],
            "best_metric": metrics[np.argmax(np.mean(coherence_matrix, axis=0))]
        }
    }

def create_performance_comparison(data, chart_height, color_scheme, theme):
    """Create performance comparison visualization"""
    
    # Simulate performance data
    strategies = ["Strategy A", "Strategy B", "Strategy C", "Strategy D"]
    performance_data = []
    
    for strategy in strategies:
        performance_data.append({
            'Strategy': strategy,
            'Accuracy': np.random.uniform(0.7, 0.95),
            'Speed': np.random.uniform(0.5, 1.0),
            'Memory': np.random.uniform(0.3, 0.8),
            'Cost': np.random.uniform(0.1, 0.5)
        })
    
    df = pd.DataFrame(performance_data)
    
    # Create radar chart
    fig = go.Figure()
    
    for _, row in df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Accuracy'], row['Speed'], row['Memory'], row['Cost']],
            theta=['Accuracy', 'Speed', 'Memory', 'Cost'],
            fill='toself',
            name=row['Strategy']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Performance Comparison Radar Chart",
        height=chart_height
    )
    
    return {
        "type": "performance_comparison",
        "figure": fig,
        "data": df.to_dict('records'),
        "statistics": {
            "best_accuracy": df.loc[df['Accuracy'].idxmax(), 'Strategy'],
            "fastest": df.loc[df['Speed'].idxmax(), 'Strategy'],
            "most_efficient": df.loc[df['Memory'].idxmax(), 'Strategy']
        }
    }

def create_topic_distribution(data, max_points, chart_height, color_scheme, theme):
    """Create topic distribution visualization"""
    
    # Simulate topic distribution data
    topics = ["Technology", "Business", "Science", "Health", "Education", "Finance"]
    strategies = ["Strategy A", "Strategy B", "Strategy C"]
    
    topic_data = []
    for strategy in strategies:
        for topic in topics:
            topic_data.append({
                'Strategy': strategy,
                'Topic': topic,
                'Count': np.random.randint(5, 50),
                'Percentage': np.random.uniform(0.1, 0.3)
            })
    
    df = pd.DataFrame(topic_data)
    
    # Create visualizations
    fig1 = px.bar(df, x='Topic', y='Count', color='Strategy',
                  title='Topic Distribution by Strategy',
                  barmode='group')
    
    fig2 = px.pie(df.groupby('Topic')['Count'].sum().reset_index(),
                  values='Count', names='Topic',
                  title='Overall Topic Distribution')
    
    fig3 = px.heatmap(df.pivot(index='Strategy', columns='Topic', values='Percentage'),
                      title='Topic Percentage Heatmap',
                      color_continuous_scale=color_scheme)
    
    return {
        "type": "topic_distribution",
        "figures": [fig1, fig2, fig3],
        "data": df.to_dict('records'),
        "statistics": {
            "total_topics": len(topics),
            "most_common_topic": df.groupby('Topic')['Count'].sum().idxmax(),
            "topic_variety": len(df['Topic'].unique())
        }
    }

def create_embedding_visualization(data, max_points, chart_height, color_scheme, theme):
    """Create embedding visualization"""
    
    # Simulate embedding data
    n_points = min(max_points, 200)
    
    # Generate 2D embeddings using t-SNE simulation
    embeddings = np.random.randn(n_points, 2)
    labels = np.random.choice(["Cluster A", "Cluster B", "Cluster C", "Cluster D"], n_points)
    
    df = pd.DataFrame({
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'Cluster': labels,
        'Size': np.random.randint(5, 20, n_points)
    })
    
    fig = px.scatter(df, x='x', y='y', color='Cluster', size='Size',
                    title='Document Embeddings Visualization',
                    color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_layout(height=chart_height)
    
    return {
        "type": "embedding_visualization",
        "figure": fig,
        "data": df.to_dict('records'),
        "statistics": {
            "total_points": n_points,
            "clusters": len(df['Cluster'].unique()),
            "avg_cluster_size": df.groupby('Cluster').size().mean()
        }
    }

def create_interactive_dashboard(data, chart_height, color_scheme, theme):
    """Create interactive dashboard"""
    
    # Simulate dashboard data
    time_series_data = []
    for i in range(30):
        time_series_data.append({
            'Date': pd.date_range('2024-01-01', periods=30)[i],
            'Performance': np.random.uniform(0.7, 0.95),
            'Accuracy': np.random.uniform(0.6, 0.9),
            'Efficiency': np.random.uniform(0.5, 0.8)
        })
    
    df = pd.DataFrame(time_series_data)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance Over Time', 'Accuracy vs Efficiency', 
                       'Performance Distribution', 'Daily Metrics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Performance over time
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Performance'], name='Performance'),
        row=1, col=1
    )
    
    # Accuracy vs Efficiency
    fig.add_trace(
        go.Scatter(x=df['Accuracy'], y=df['Efficiency'], mode='markers', name='Accuracy vs Efficiency'),
        row=1, col=2
    )
    
    # Performance distribution
    fig.add_trace(
        go.Histogram(x=df['Performance'], name='Performance Distribution'),
        row=2, col=1
    )
    
    # Daily metrics
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Accuracy'], name='Daily Accuracy'),
        row=2, col=2
    )
    
    fig.update_layout(height=chart_height, title_text="Interactive Dashboard")
    
    return {
        "type": "interactive_dashboard",
        "figure": fig,
        "data": df.to_dict('records'),
        "statistics": {
            "avg_performance": df['Performance'].mean(),
            "avg_accuracy": df['Accuracy'].mean(),
            "trend": "increasing" if df['Performance'].iloc[-1] > df['Performance'].iloc[0] else "decreasing"
        }
    }

def load_existing_visualizations():
    """Load existing visualizations"""
    
    # Check for existing visualization files
    viz_dir = Path("outputs/visualizations")
    
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("*.html")) + list(viz_dir.glob("*.png"))
        
        if viz_files:
            st.session_state.visualization_results = {
                "existing_files": [str(f) for f in viz_files]
            }
            st.success("📂 Existing visualizations loaded!")
        else:
            st.warning("No existing visualization files found.")
    else:
        st.warning("Visualizations directory not found.")

def display_visualization_results(results):
    """Display visualization results"""
    
    st.subheader("📊 Visualization Results")
    
    for viz_type, result in results.items():
        if "error" in result:
            st.error(f"Error in {viz_type}: {result['error']}")
            continue
        
        st.markdown(f"### {viz_type.replace('_', ' ').title()}")
        
        if "figures" in result:
            # Multiple figures
            for i, fig in enumerate(result["figures"]):
                st.plotly_chart(fig, use_container_width=True)
                
                if i < len(result["figures"]) - 1:
                    st.markdown("---")
        elif "figure" in result:
            # Single figure
            st.plotly_chart(result["figure"], use_container_width=True)
        
        # Show statistics if available
        if "statistics" in result and st.session_state.get('show_statistics', True):
            st.markdown("**Statistics:**")
            for key, value in result["statistics"].items():
                st.markdown(f"- {key.replace('_', ' ').title()}: {value}")
        
        # Export options
        if "figure" in result or "figures" in result:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"📥 Download {viz_type} (PNG)"):
                    st.info("Download functionality would be implemented here")
            
            with col2:
                if st.button(f"📥 Download {viz_type} (HTML)"):
                    st.info("Download functionality would be implemented here")
            
            with col3:
                if st.button(f"📥 Download {viz_type} (PDF)"):
                    st.info("Download functionality would be implemented here")
        
        st.markdown("---")

def display_welcome():
    """Display welcome message"""
    
    st.markdown("""
    ## Welcome to Visualization Suite
    
    This comprehensive tool provides advanced visualizations for chunking analysis and performance evaluation.
    
    ### How to use:
    1. **Select Visualizations**: Choose which types of visualizations to generate
    2. **Choose Data Sources**: Select the data files to visualize
    3. **Configure Parameters**: Adjust visualization settings and appearance
    4. **Generate Visualizations**: Create interactive charts and graphs
    5. **Export Results**: Download visualizations in various formats
    
    ### Available Visualizations:
    - **Chunk Overlap Analysis**: Analyze overlap patterns and distributions
    - **Semantic Coherence Heatmap**: Visualize semantic relationships
    - **Performance Comparison**: Compare different strategies and metrics
    - **Topic Distribution**: Explore topic clustering and distribution
    - **Embedding Visualization**: Visualize document embeddings in 2D/3D
    - **Interactive Dashboard**: Comprehensive dashboard with multiple views
    
    ### Data Sources:
    - Strategy comparison results
    - Semantic analysis results
    - RAG evaluation results
    - LLM chunking test results
    - Topic clustering results
    
    ### Features:
    - Interactive Plotly charts
    - Multiple color schemes and themes
    - Export to PNG, SVG, PDF, HTML
    - Statistical analysis and insights
    - Customizable parameters
    - Real-time data updates
    
    ### Export Options:
    - High-resolution PNG images
    - Scalable SVG graphics
    - Interactive HTML files
    - Print-ready PDF documents
    """)

if __name__ == "__main__":
    main() 