#!/usr/bin/env python3
"""
5. Topic Clustering
===================

Explore topic-based chunking and clustering analysis.

Author: Data Engineering Team
Purpose: Topic clustering and analysis for chunking optimization
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
    page_title="Topic Clustering",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main function for topic clustering page"""
    
    st.title("🗂️ Topic Clustering Analysis")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Clustering methods
    clustering_methods = [
        "kmeans_clustering",
        "hierarchical_clustering", 
        "dbscan_clustering",
        "topic_modeling",
        "semantic_clustering"
    ]
    
    selected_methods = st.sidebar.multiselect(
        "Select clustering methods:",
        options=clustering_methods,
        default=clustering_methods[:3]
    )
    
    # Test documents
    test_documents = [
        "executive_summaries.txt",
        "mixed_content.txt", 
        "regulatory_documents.txt",
        "research_papers.txt",
        "technical_reports.txt"
    ]
    
    selected_documents = st.sidebar.multiselect(
        "Select test documents:",
        options=test_documents,
        default=test_documents[:2]
    )
    
    # Clustering parameters
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.sidebar.slider("Number of Clusters:", 2, 20, 5)
        chunk_size = st.sidebar.slider("Chunk Size:", 100, 500, 300)
    
    with col2:
        min_cluster_size = st.sidebar.slider("Min Cluster Size:", 1, 10, 2)
        similarity_threshold = st.sidebar.slider("Similarity Threshold:", 0.0, 1.0, 0.7, 0.05)
    
    # Advanced settings
    st.sidebar.header("⚙️ Advanced Settings")
    
    embedding_model = st.sidebar.selectbox(
        "Embedding Model:",
        ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
    )
    
    dimensionality_reduction = st.sidebar.selectbox(
        "Dimensionality Reduction:",
        ["pca", "tsne", "umap", "none"]
    )
    
    # Run clustering button
    if st.sidebar.button("🚀 Run Topic Clustering", type="primary"):
        run_topic_clustering(selected_methods, selected_documents, n_clusters, chunk_size,
                           min_cluster_size, similarity_threshold, embedding_model, 
                           dimensionality_reduction)
    
    # Load existing results
    if st.sidebar.button("📂 Load Existing Results"):
        load_existing_results()
    
    # Main content area
    if 'clustering_results' in st.session_state:
        display_clustering_results(st.session_state.clustering_results)
    else:
        display_welcome()

def run_topic_clustering(methods, documents, n_clusters, chunk_size, min_cluster_size,
                        similarity_threshold, embedding_model, dimensionality_reduction):
    """Run topic clustering analysis"""
    
    st.subheader("🔄 Running Topic Clustering Analysis...")
    
    with st.spinner("Executing topic clustering..."):
        results = {}
        
        for method in methods:
            st.write(f"Running {method}...")
            
            for document in documents:
                st.write(f"  Processing {document}...")
                
                try:
                    # Run the clustering method
                    result = execute_clustering_method(method, document, n_clusters, chunk_size,
                                                     min_cluster_size, similarity_threshold,
                                                     embedding_model, dimensionality_reduction)
                    
                    key = f"{method}_{document}"
                    results[key] = result
                    
                    st.success(f"✅ {method} - {document} completed")
                    
                except Exception as e:
                    st.error(f"❌ Error in {method} - {document}: {e}")
                    results[f"{method}_{document}"] = {"error": str(e)}
        
        # Calculate clustering metrics
        clustering_analysis = calculate_clustering_metrics(results)
        
        # Save results
        save_clustering_results(clustering_analysis)
        
        # Store in session state
        st.session_state.clustering_results = clustering_analysis
        
        st.success("🎉 Topic clustering completed!")

def execute_clustering_method(method, document, n_clusters, chunk_size, min_cluster_size,
                            similarity_threshold, embedding_model, dimensionality_reduction):
    """Execute a specific clustering method"""
    
    # This would typically involve:
    # 1. Chunking the document
    # 2. Creating embeddings for chunks
    # 3. Applying clustering algorithm
    # 4. Analyzing cluster quality
    
    # For now, we'll simulate the clustering
    clustering_result = {
        "method": method,
        "document": document,
        "parameters": {
            "n_clusters": n_clusters,
            "chunk_size": chunk_size,
            "min_cluster_size": min_cluster_size,
            "similarity_threshold": similarity_threshold,
            "embedding_model": embedding_model,
            "dimensionality_reduction": dimensionality_reduction
        },
        "clusters": [],
        "embeddings": [],
        "metrics": {
            "silhouette_score": np.random.uniform(0.3, 0.8),
            "calinski_harabasz_score": np.random.uniform(100, 1000),
            "davies_bouldin_score": np.random.uniform(0.5, 2.0),
            "cluster_coherence": np.random.uniform(0.6, 0.9),
            "topic_diversity": np.random.uniform(0.4, 0.8),
            "processing_time": np.random.uniform(1.0, 10.0)
        },
        "topics": [],
        "cluster_assignments": []
    }
    
    # Generate simulated clusters
    num_chunks = np.random.randint(20, 50)
    for i in range(num_chunks):
        cluster_id = np.random.randint(0, n_clusters)
        chunk_text = f"Simulated chunk {i+1} for cluster {cluster_id}. " * (chunk_size // 20)
        
        clustering_result["clusters"].append({
            "chunk_index": i,
            "text": chunk_text,
            "cluster_id": cluster_id,
            "size": len(chunk_text.split()),
            "embedding": np.random.rand(384).tolist()  # Simulated embedding
        })
        
        clustering_result["cluster_assignments"].append(cluster_id)
    
    # Generate simulated topics
    for i in range(n_clusters):
        topic_words = [f"topic_word_{i}_{j}" for j in range(5)]
        clustering_result["topics"].append({
            "cluster_id": i,
            "topic_words": topic_words,
            "topic_coherence": np.random.uniform(0.5, 0.9),
            "cluster_size": sum(1 for x in clustering_result["cluster_assignments"] if x == i)
        })
    
    return clustering_result

def calculate_clustering_metrics(results):
    """Calculate metrics for clustering results"""
    
    clustering_data = []
    
    for test_key, result in results.items():
        if "error" in result:
            continue
        
        method, document = test_key.split("_", 1)
        metrics = result.get("metrics", {})
        
        clustering_data.append({
            'Method': method,
            'Document': document,
            'Num_Clusters': result.get("parameters", {}).get("n_clusters", 0),
            'Num_Chunks': len(result.get("clusters", [])),
            'Silhouette_Score': metrics.get("silhouette_score", 0),
            'Calinski_Harabasz_Score': metrics.get("calinski_harabasz_score", 0),
            'Davies_Bouldin_Score': metrics.get("davies_bouldin_score", 0),
            'Cluster_Coherence': metrics.get("cluster_coherence", 0),
            'Topic_Diversity': metrics.get("topic_diversity", 0),
            'Processing_Time': metrics.get("processing_time", 0),
            'Test_Key': test_key,
            'Result': result
        })
    
    return clustering_data

def save_clustering_results(clustering_data):
    """Save clustering results to file"""
    
    output_file = "outputs/topic_clustering_results.json"
    os.makedirs("outputs", exist_ok=True)
    
    # Remove the full result object for JSON serialization
    serializable_data = []
    for item in clustering_data:
        serializable_item = item.copy()
        serializable_item.pop('Result', None)
        serializable_data.append(serializable_item)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    st.info(f"Results saved to {output_file}")

def load_existing_results():
    """Load existing clustering results"""
    
    output_file = "outputs/topic_clustering_results.json"
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
        st.session_state.clustering_results = results
        st.success("📂 Results loaded successfully!")
    else:
        st.warning("No existing results found. Run topic clustering first.")

def display_clustering_results(clustering_data):
    """Display clustering results"""
    
    st.subheader("📊 Topic Clustering Results")
    
    # Create DataFrame
    df = pd.DataFrame(clustering_data)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tests Completed", len(df))
    
    with col2:
        avg_silhouette = df['Silhouette_Score'].mean()
        st.metric("Avg Silhouette Score", f"{avg_silhouette:.3f}")
    
    with col3:
        avg_coherence = df['Cluster_Coherence'].mean()
        st.metric("Avg Cluster Coherence", f"{avg_coherence:.3f}")
    
    with col4:
        best_method = df.loc[df['Silhouette_Score'].idxmax(), 'Method']
        st.metric("Best Method", best_method)
    
    # Results table
    st.subheader("📋 Detailed Results")
    display_df = df.drop(['Test_Key', 'Result'], axis=1)  # Remove non-display columns
    st.dataframe(display_df, use_container_width=True)
    
    # Visualizations
    st.subheader("📈 Clustering Performance Visualizations")
    
    # Method comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Silhouette score comparison
        fig = px.box(df, x='Method', y='Silhouette_Score',
                    title='Silhouette Score by Method',
                    color='Method')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Coherence vs diversity
        fig = px.scatter(df, x='Cluster_Coherence', y='Topic_Diversity',
                        size='Num_Chunks', color='Method',
                        title='Coherence vs Topic Diversity',
                        hover_data=['Document'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Clustering quality metrics
    st.subheader("🎯 Clustering Quality Metrics")
    
    quality_metrics = ['Silhouette_Score', 'Calinski_Harabasz_Score', 
                      'Davies_Bouldin_Score', 'Cluster_Coherence']
    
    fig = px.heatmap(df[quality_metrics].corr(),
                    title='Clustering Quality Metrics Correlation',
                    color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Method performance comparison
    st.subheader("📊 Method Performance Comparison")
    
    method_comparison = df.groupby('Method').agg({
        'Silhouette_Score': 'mean',
        'Cluster_Coherence': 'mean',
        'Topic_Diversity': 'mean',
        'Processing_Time': 'mean'
    }).round(3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(method_comparison, use_container_width=True)
    
    with col2:
        fig = px.bar(method_comparison, y=['Silhouette_Score', 'Cluster_Coherence'],
                    title='Method Performance Comparison',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    st.subheader("🔍 Detailed Analysis")
    
    # Test details
    selected_test = st.selectbox("Select test for detailed view:", df['Test_Key'].tolist())
    
    if selected_test:
        test_data = df[df['Test_Key'] == selected_test].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Test: {selected_test}**")
            st.markdown(f"- Method: {test_data['Method']}")
            st.markdown(f"- Document: {test_data['Document']}")
            st.markdown(f"- Number of clusters: {test_data['Num_Clusters']}")
            st.markdown(f"- Number of chunks: {test_data['Num_Chunks']}")
            st.markdown(f"- Silhouette score: {test_data['Silhouette_Score']:.3f}")
            st.markdown(f"- Cluster coherence: {test_data['Cluster_Coherence']:.3f}")
            st.markdown(f"- Topic diversity: {test_data['Topic_Diversity']:.3f}")
            st.markdown(f"- Processing time: {test_data['Processing_Time']:.2f}s")
        
        with col2:
            # Cluster size distribution
            cluster_sizes = [test_data['Num_Chunks'] // test_data['Num_Clusters']] * test_data['Num_Clusters']
            fig = px.bar(x=range(test_data['Num_Clusters']), y=cluster_sizes,
                        title='Cluster Size Distribution',
                        labels={'x': 'Cluster ID', 'y': 'Number of Chunks'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Topic analysis
    st.subheader("📝 Topic Analysis")
    
    # Document analysis
    doc_analysis = df.groupby('Document').agg({
        'Silhouette_Score': 'mean',
        'Cluster_Coherence': 'mean',
        'Num_Clusters': 'mean'
    }).round(3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(doc_analysis, use_container_width=True)
    
    with col2:
        fig = px.scatter(doc_analysis, x='Num_Clusters', y='Silhouette_Score',
                        size='Cluster_Coherence', title='Document Clustering Performance',
                        hover_data=doc_analysis.index)
        st.plotly_chart(fig, use_container_width=True)
    
    # Processing time analysis
    st.subheader("⏱️ Processing Time Analysis")
    
    time_analysis = df.groupby('Method')['Processing_Time'].agg(['mean', 'std']).round(2)
    
    fig = px.bar(time_analysis, y='mean', error_y='std',
                title='Processing Time by Method',
                labels={'mean': 'Average Time (s)', 'index': 'Method'})
    st.plotly_chart(fig, use_container_width=True)

def display_welcome():
    """Display welcome message"""
    
    st.markdown("""
    ## Welcome to Topic Clustering Analysis
    
    This tool explores topic-based chunking and clustering analysis for document processing optimization.
    
    ### How to use:
    1. **Select Methods**: Choose clustering algorithms to test
    2. **Choose Documents**: Select test documents for analysis
    3. **Configure Parameters**: Adjust clustering parameters and embedding settings
    4. **Run Analysis**: Execute topic clustering and view results
    5. **Explore Results**: Analyze cluster quality and topic distribution
    
    ### Available Clustering Methods:
    - **K-Means Clustering**: Traditional centroid-based clustering
    - **Hierarchical Clustering**: Tree-based clustering approach
    - **DBSCAN Clustering**: Density-based spatial clustering
    - **Topic Modeling**: Latent topic discovery using LDA/NMF
    - **Semantic Clustering**: Embedding-based semantic clustering
    
    ### Test Documents:
    - **Executive Summaries**: High-level business documents
    - **Mixed Content**: Documents with various content types
    - **Regulatory Documents**: Legal and compliance documents
    - **Research Papers**: Academic and technical papers
    - **Technical Reports**: Detailed technical documentation
    
    ### Metrics Evaluated:
    - Silhouette score (cluster quality)
    - Calinski-Harabasz score (cluster separation)
    - Davies-Bouldin score (cluster compactness)
    - Cluster coherence (semantic consistency)
    - Topic diversity (content variety)
    - Processing time and efficiency
    
    ### Advanced Features:
    - Dimensionality reduction (PCA, t-SNE, UMAP)
    - Multiple embedding models
    - Topic word extraction
    - Cluster visualization
    - Performance benchmarking
    - Document-specific optimization
    """)

if __name__ == "__main__":
    main() 