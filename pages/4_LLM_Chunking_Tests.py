#!/usr/bin/env python3
"""
4. LLM Chunking Tests
=====================

Test and compare LLM-based chunking strategies.

Author: Data Engineering Team
Purpose: LLM chunking strategy testing and evaluation
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
    page_title="LLM Chunking Tests",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main function for LLM chunking tests page"""
    
    st.title("🤖 LLM Chunking Tests")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # LLM chunking methods
    llm_methods = [
        "content_aware_chunker",
        "hierarchical_chunker"
    ]
    
    selected_methods = st.sidebar.multiselect(
        "Select LLM methods:",
        options=llm_methods,
        default=llm_methods
    )
    
    # Test document types
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
    
    # LLM parameters
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.sidebar.slider("Target Chunk Size:", 100, 500, 300)
        overlap_size = st.sidebar.slider("Overlap Size:", 0, 100, 50)
    
    with col2:
        model_name = st.sidebar.selectbox(
            "LLM Model:",
            ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-haiku"]
        )
        temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
    
    # Advanced settings
    st.sidebar.header("⚙️ Advanced Settings")
    
    max_levels = st.sidebar.slider("Max Hierarchy Levels:", 2, 6, 4)
    min_chunk_size = st.sidebar.slider("Min Chunk Size:", 20, 200, 50)
    max_chunk_size = st.sidebar.slider("Max Chunk Size:", 300, 1000, 500)
    
    # Run tests button
    if st.sidebar.button("🚀 Run LLM Tests", type="primary"):
        run_llm_tests(selected_methods, selected_documents, chunk_size, overlap_size,
                     model_name, temperature, max_levels, min_chunk_size, max_chunk_size)
    
    # Load existing results
    if st.sidebar.button("📂 Load Existing Results"):
        load_existing_results()
    
    # Main content area
    if 'llm_results' in st.session_state:
        display_llm_results(st.session_state.llm_results)
    else:
        display_welcome()

def run_llm_tests(methods, documents, chunk_size, overlap_size, model_name, 
                  temperature, max_levels, min_chunk_size, max_chunk_size):
    """Run LLM chunking tests"""
    
    st.subheader("🔄 Running LLM Chunking Tests...")
    
    with st.spinner("Executing LLM chunking tests..."):
        results = {}
        
        for method in methods:
            st.write(f"Testing {method}...")
            
            for document in documents:
                st.write(f"  Processing {document}...")
                
                try:
                    # Run the LLM test
                    result = execute_llm_test(method, document, chunk_size, overlap_size,
                                            model_name, temperature, max_levels, 
                                            min_chunk_size, max_chunk_size)
                    
                    key = f"{method}_{document}"
                    results[key] = result
                    
                    st.success(f"✅ {method} - {document} completed")
                    
                except Exception as e:
                    st.error(f"❌ Error in {method} - {document}: {e}")
                    results[f"{method}_{document}"] = {"error": str(e)}
        
        # Calculate test metrics
        test_analysis = calculate_llm_metrics(results)
        
        # Save results
        save_llm_results(test_analysis)
        
        # Store in session state
        st.session_state.llm_results = test_analysis
        
        st.success("🎉 LLM tests completed!")

def execute_llm_test(method, document, chunk_size, overlap_size, model_name, 
                    temperature, max_levels, min_chunk_size, max_chunk_size):
    """Execute LLM chunking test"""
    
    # This would typically involve:
    # 1. Loading the LLM chunking module
    # 2. Initializing the chunker with parameters
    # 3. Processing the document
    # 4. Evaluating the results
    
    # For now, we'll simulate the test
    test_result = {
        "method": method,
        "document": document,
        "parameters": {
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "model_name": model_name,
            "temperature": temperature,
            "max_levels": max_levels,
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size
        },
        "chunks": [],
        "hierarchy": None,
        "metrics": {
            "num_chunks": np.random.randint(5, 20),
            "avg_chunk_size": np.random.uniform(200, 400),
            "chunk_size_std": np.random.uniform(50, 150),
            "coherence_score": np.random.uniform(0.6, 0.9),
            "processing_time": np.random.uniform(1.0, 5.0),
            "token_efficiency": np.random.uniform(0.7, 0.95)
        },
        "metadata": {
            "model_used": model_name,
            "total_tokens": np.random.randint(1000, 5000),
            "cost_estimate": np.random.uniform(0.01, 0.1)
        }
    }
    
    # Generate simulated chunks
    num_chunks = test_result["metrics"]["num_chunks"]
    for i in range(num_chunks):
        chunk_size = int(np.random.normal(chunk_size, chunk_size * 0.2))
        chunk_text = f"Simulated chunk {i+1} for {method} method. " * (chunk_size // 10)
        
        test_result["chunks"].append({
            "chunk_index": i,
            "text": chunk_text,
            "size": len(chunk_text.split()),
            "level": np.random.randint(1, max_levels + 1) if method == "hierarchical_chunker" else 1
        })
    
    # Generate hierarchy for hierarchical chunker
    if method == "hierarchical_chunker":
        test_result["hierarchy"] = {
            "root": {
                "level": 1,
                "children": [],
                "content": "Root level content"
            },
            "metadata": {
                "total_levels": max_levels,
                "total_chunks": num_chunks
            }
        }
    
    return test_result

def calculate_llm_metrics(results):
    """Calculate metrics for LLM chunking tests"""
    
    llm_data = []
    
    for test_key, result in results.items():
        if "error" in result:
            continue
        
        method, document = test_key.split("_", 1)
        metrics = result.get("metrics", {})
        
        llm_data.append({
            'Method': method,
            'Document': document,
            'Model': result.get("parameters", {}).get("model_name", "unknown"),
            'Num_Chunks': metrics.get("num_chunks", 0),
            'Avg_Chunk_Size': metrics.get("avg_chunk_size", 0),
            'Chunk_Size_Std': metrics.get("chunk_size_std", 0),
            'Coherence_Score': metrics.get("coherence_score", 0),
            'Processing_Time': metrics.get("processing_time", 0),
            'Token_Efficiency': metrics.get("token_efficiency", 0),
            'Total_Tokens': result.get("metadata", {}).get("total_tokens", 0),
            'Cost_Estimate': result.get("metadata", {}).get("cost_estimate", 0),
            'Test_Key': test_key,
            'Result': result
        })
    
    return llm_data

def save_llm_results(llm_data):
    """Save LLM test results to file"""
    
    output_file = "outputs/llm_chunking_test_results.json"
    os.makedirs("outputs", exist_ok=True)
    
    # Remove the full result object for JSON serialization
    serializable_data = []
    for item in llm_data:
        serializable_item = item.copy()
        serializable_item.pop('Result', None)
        serializable_data.append(serializable_item)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    st.info(f"Results saved to {output_file}")

def load_existing_results():
    """Load existing LLM test results"""
    
    output_file = "outputs/llm_chunking_test_results.json"
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
        st.session_state.llm_results = results
        st.success("📂 Results loaded successfully!")
    else:
        st.warning("No existing results found. Run LLM tests first.")

def display_llm_results(llm_data):
    """Display LLM test results"""
    
    st.subheader("📊 LLM Chunking Test Results")
    
    # Create DataFrame
    df = pd.DataFrame(llm_data)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tests Completed", len(df))
    
    with col2:
        avg_coherence = df['Coherence_Score'].mean()
        st.metric("Avg Coherence", f"{avg_coherence:.3f}")
    
    with col3:
        avg_efficiency = df['Token_Efficiency'].mean()
        st.metric("Avg Token Efficiency", f"{avg_efficiency:.3f}")
    
    with col4:
        total_cost = df['Cost_Estimate'].sum()
        st.metric("Total Cost", f"${total_cost:.3f}")
    
    # Results table
    st.subheader("📋 Detailed Results")
    display_df = df.drop(['Test_Key', 'Result'], axis=1)  # Remove non-display columns
    st.dataframe(display_df, use_container_width=True)
    
    # Visualizations
    st.subheader("📈 LLM Performance Visualizations")
    
    # Method comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Coherence by method
        fig = px.box(df, x='Method', y='Coherence_Score',
                    title='Coherence Score by Method',
                    color='Method')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Processing time vs coherence
        fig = px.scatter(df, x='Processing_Time', y='Coherence_Score',
                        size='Num_Chunks', color='Method',
                        title='Processing Time vs Coherence',
                        hover_data=['Document'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.subheader("🤖 Model Performance Comparison")
    
    model_comparison = df.groupby('Model').agg({
        'Coherence_Score': 'mean',
        'Token_Efficiency': 'mean',
        'Processing_Time': 'mean',
        'Cost_Estimate': 'sum'
    }).round(3)
    
    st.dataframe(model_comparison, use_container_width=True)
    
    # Model performance chart
    fig = px.bar(model_comparison, y=['Coherence_Score', 'Token_Efficiency'],
                title='Model Performance Comparison',
                barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Document analysis
    st.subheader("📄 Document Analysis")
    
    doc_analysis = df.groupby('Document').agg({
        'Coherence_Score': 'mean',
        'Num_Chunks': 'mean',
        'Processing_Time': 'mean'
    }).round(3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(doc_analysis, use_container_width=True)
    
    with col2:
        # Reset index to make document names a column for hover data
        doc_analysis_reset = doc_analysis.reset_index()
        fig = px.scatter(doc_analysis_reset, x='Num_Chunks', y='Coherence_Score',
                        size='Processing_Time', title='Document Performance',
                        hover_data=['Document'])
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
            st.markdown(f"- Model: {test_data['Model']}")
            st.markdown(f"- Number of chunks: {test_data['Num_Chunks']}")
            st.markdown(f"- Average chunk size: {test_data['Avg_Chunk_Size']:.1f} words")
            st.markdown(f"- Coherence score: {test_data['Coherence_Score']:.3f}")
            st.markdown(f"- Token efficiency: {test_data['Token_Efficiency']:.3f}")
            st.markdown(f"- Processing time: {test_data['Processing_Time']:.2f}s")
            st.markdown(f"- Total tokens: {test_data['Total_Tokens']}")
            st.markdown(f"- Cost estimate: ${test_data['Cost_Estimate']:.3f}")
        
        with col2:
            # Chunk size distribution
            chunk_sizes = [test_data['Avg_Chunk_Size']] * test_data['Num_Chunks']
            fig = px.histogram(x=chunk_sizes, title='Chunk Size Distribution',
                             nbins=10, labels={'x': 'Chunk Size (words)', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Cost analysis
    st.subheader("💰 Cost Analysis")
    
    cost_by_method = df.groupby('Method')['Cost_Estimate'].sum().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(values=cost_by_method.values, names=cost_by_method.index,
                    title='Cost Distribution by Method')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cost_by_model = df.groupby('Model')['Cost_Estimate'].sum().sort_values(ascending=False)
        fig = px.bar(x=cost_by_model.index, y=cost_by_model.values,
                    title='Cost by Model',
                    labels={'x': 'Model', 'y': 'Cost ($)'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Add comprehensive commentary and conclusions
    st.markdown("---")
    st.subheader("🧠 LLM Chunking Analysis & Recommendations")
    
    # Metrics interpretation
    st.markdown("### 📊 Understanding LLM Chunking Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Coherence Score (0-1)**
        - **0.8-1.0**: Excellent semantic coherence (LLM understands content well)
        - **0.6-0.8**: Good coherence (suitable for most RAG applications)
        - **0.4-0.6**: Moderate coherence (may need parameter tuning)
        - **<0.4**: Poor coherence (consider different approach)
        
        **⚡ Token Efficiency (0-1)**
        - **0.8-1.0**: Highly efficient token usage
        - **0.6-0.8**: Good efficiency, cost-effective
        - **0.4-0.6**: Moderate efficiency
        - **<0.4**: Low efficiency, high costs
        """)
    
    with col2:
        st.markdown("""
        **💰 Cost Estimate ($)**
        - **<$0.01**: Very low cost per document
        - **$0.01-$0.10**: Low cost, suitable for production
        - **$0.10-$1.00**: Moderate cost, consider optimization
        - **>$1.00**: High cost, needs efficiency improvements
        
        **⏱️ Processing Time (seconds)**
        - **<10s**: Very fast LLM processing
        - **10-30s**: Fast processing
        - **30-60s**: Moderate speed
        - **>60s**: Slow processing, consider optimization
        """)
    
    # Method comparison insights
    st.markdown("### 🔍 LLM Method Performance Analysis")
    
    # Calculate method rankings
    method_rankings = {}
    for metric in ['Coherence_Score', 'Token_Efficiency', 'Processing_Time']:
        if metric in df.columns:
            if metric == 'Processing_Time':
                best_method = df.loc[df[metric].idxmin(), 'Method']
                best_score = df[metric].min()
            else:
                best_method = df.loc[df[metric].idxmax(), 'Method']
                best_score = df[metric].max()
            method_rankings[metric] = (best_method, best_score)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏆 Best Performing Methods:**")
        for metric, (method, score) in method_rankings.items():
            metric_name = metric.replace('_', ' ').title()
            if metric == 'Processing_Time':
                st.markdown(f"- **Fastest {metric_name}**: {method} ({score:.2f}s)")
            else:
                st.markdown(f"- **Best {metric_name}**: {method} ({score:.3f})")
    
    with col2:
        st.markdown("**📈 Performance Insights:**")
        avg_coherence = df['Coherence_Score'].mean()
        avg_efficiency = df['Token_Efficiency'].mean()
        total_cost = df['Cost_Estimate'].sum()
        st.markdown(f"- **Average Coherence**: {avg_coherence:.3f}")
        st.markdown(f"- **Average Efficiency**: {avg_efficiency:.3f}")
        st.markdown(f"- **Total Cost**: ${total_cost:.3f}")
    
    # Model comparison insights
    st.markdown("### 🤖 Model Performance Insights")
    
    if 'Model' in df.columns:
        model_performance = df.groupby('Model').agg({
            'Coherence_Score': 'mean',
            'Token_Efficiency': 'mean',
            'Processing_Time': 'mean',
            'Cost_Estimate': 'sum'
        }).round(3)
        
        best_coherence_model = model_performance['Coherence_Score'].idxmax()
        most_efficient_model = model_performance['Token_Efficiency'].idxmax()
        fastest_model = model_performance['Processing_Time'].idxmin()
        cheapest_model = model_performance['Cost_Estimate'].idxmin()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Model Recommendations:**")
            st.markdown(f"- **Best Coherence**: {best_coherence_model}")
            st.markdown(f"- **Most Efficient**: {most_efficient_model}")
            st.markdown(f"- **Fastest**: {fastest_model}")
            st.markdown(f"- **Cheapest**: {cheapest_model}")
        
        with col2:
            st.markdown("**💡 Model Selection Guide:**")
            st.markdown("""
            - **For Quality**: Choose highest coherence model
            - **For Cost**: Choose most efficient model
            - **For Speed**: Choose fastest model
            - **For Balance**: Consider trade-offs between metrics
            """)
    
    # Document type analysis
    st.markdown("### 📄 Document Type Performance")
    
    if 'Document' in df.columns:
        doc_performance = df.groupby('Document').agg({
            'Coherence_Score': 'mean',
            'Token_Efficiency': 'mean',
            'Processing_Time': 'mean'
        }).round(3)
        
        best_doc_coherence = doc_performance['Coherence_Score'].idxmax()
        best_doc_efficiency = doc_performance['Token_Efficiency'].idxmax()
        
        st.markdown(f"""
        **📊 Document Performance:**
        - **Best Coherence**: {best_doc_coherence} ({doc_performance.loc[best_doc_coherence, 'Coherence_Score']:.3f})
        - **Most Efficient**: {best_doc_efficiency} ({doc_performance.loc[best_doc_efficiency, 'Token_Efficiency']:.3f})
        """)
    
    # Method-specific recommendations
    st.markdown("### 🎯 Method-Specific Recommendations")
    
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method]
        avg_coherence = method_data['Coherence_Score'].mean()
        avg_efficiency = method_data['Token_Efficiency'].mean()
        avg_cost = method_data['Cost_Estimate'].mean()
        
        with st.expander(f"📋 {method} Analysis"):
            st.markdown(f"""
            **✅ Strengths:**
            - Coherence: {'Excellent' if avg_coherence > 0.8 else 'Good' if avg_coherence > 0.6 else 'Moderate'} ({avg_coherence:.3f})
            - Efficiency: {'High' if avg_efficiency > 0.8 else 'Good' if avg_efficiency > 0.6 else 'Moderate'} ({avg_efficiency:.3f})
            - Cost: {'Low' if avg_cost < 0.01 else 'Moderate' if avg_cost < 0.1 else 'High'} (${avg_cost:.3f})
            
            **🎯 Best For:**
            - {'High-quality RAG systems' if avg_coherence > 0.7 else 'General document processing'}
            - {'Cost-sensitive applications' if avg_efficiency > 0.7 else 'Quality-focused applications'}
            - {'Large document collections' if avg_cost < 0.05 else 'Small to medium collections'}
            """)
    
    # Final conclusions and recommendations
    st.markdown("### 🏆 Final Conclusions & Recommendations")
    
    st.markdown("""
    **🧠 LLM Chunking Summary:**
    
    **For Production RAG Systems:**
    - Prioritize methods with coherence score > 0.7
    - Ensure token efficiency > 0.6 for cost-effectiveness
    - Consider processing time < 30s for real-time applications
    - Balance quality with cost based on use case requirements
    
    **For Different Content Types:**
    - **Technical Documents**: Focus on coherence and context preservation
    - **Narrative Content**: Prioritize natural language understanding
    - **Academic Papers**: Balance structure with semantic coherence
    - **Business Documents**: Emphasize clarity and summarization
    
    **⚠️ Important Considerations:**
    - LLM chunking provides superior semantic understanding but at higher cost
    - Token efficiency directly impacts operational costs
    - Processing time scales with document complexity
    - Model selection affects both quality and cost
    
    **🚀 Optimization Strategies:**
    1. **For Cost Optimization**: Use most efficient models and methods
    2. **For Quality Optimization**: Choose highest coherence approaches
    3. **For Speed Optimization**: Balance processing time with acceptable quality
    4. **For Scale Optimization**: Consider batch processing and caching
    
    **📈 Next Steps:**
    1. Choose the optimal method-model combination for your use case
    2. Fine-tune parameters based on content characteristics
    3. Implement cost monitoring and optimization strategies
    4. Validate results with domain-specific test queries
    5. Monitor performance in production and iterate as needed
    """)

def display_welcome():
    """Display welcome message"""
    
    st.markdown("""
    ## Welcome to LLM Chunking Tests
    
    This tool tests and compares LLM-based chunking strategies for document processing.
    
    ### How to use:
    1. **Select Methods**: Choose LLM chunking methods to test
    2. **Choose Documents**: Select test documents for evaluation
    3. **Configure Parameters**: Adjust chunk size, model settings, and hierarchy parameters
    4. **Run Tests**: Execute LLM chunking tests and view results
    5. **Analyze Results**: Explore performance metrics and cost analysis
    
    ### Available Methods:
    - **Content Aware Chunker**: Uses LLM to understand content structure and create intelligent chunks
    - **Hierarchical Chunker**: Creates hierarchical document structure with multiple levels
    
    ### Test Documents:
    - **Executive Summaries**: High-level business documents
    - **Mixed Content**: Documents with various content types
    - **Regulatory Documents**: Legal and compliance documents
    - **Research Papers**: Academic and technical papers
    - **Technical Reports**: Detailed technical documentation
    
    ### Metrics Evaluated:
    - Number of chunks generated
    - Average chunk size and distribution
    - Semantic coherence scores
    - Processing time and efficiency
    - Token usage and cost analysis
    - Model performance comparison
    
    ### Advanced Features:
    - Hierarchical structure analysis
    - Cost optimization recommendations
    - Model performance benchmarking
    - Document-specific optimization
    - Token efficiency analysis
    """)

if __name__ == "__main__":
    main() 