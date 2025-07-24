# ADNOC Semantic Chunking and Visualization Tool

This project provides advanced semantic chunking capabilities for PDF documents with comprehensive visualization and analysis tools. It's designed to demonstrate the value of semantic vs traditional chunking for RAG (Retrieval-Augmented Generation) systems.

## Features

- **Multiple Chunking Strategies**: Traditional, semantic, and hybrid approaches
- **PDF Processing**: Extract and process text from PDF documents
- **Azure OpenAI Integration**: Leverage Azure OpenAI for embeddings and completions
- **Advanced Visualizations**: t-SNE clustering, topic modeling, similarity heatmaps
- **Performance Analysis**: Compare different chunking strategies
- **Executive Dashboards**: Generate business-ready visualizations

## Project Structure

```
chunking/
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .env                          # Environment variables (create this)
├── Home.py                       # Main Streamlit dashboard application
├── run_chunking_comparison.py    # Main comparison script
├── run_llm_chunking_tests.py     # LLM chunking test runner
├── .gitignore                    # Git ignore file
├── README_enhancements.md        # Enhancement documentation
│
├── visualization/                # Visualization tools
│   ├── README.md
│   ├── visualize-1.py           # 3D t-SNE visualization
│   ├── visualize-2.py           # 2D visualization
│   ├── visualize-3.py           # Cosine distance visualization
│   └── enhanced_visualizer.py   # Main visualization tool
│
├── chunking-strategies/          # Chunking strategy implementations
│   ├── README.md
│   ├── chunk_full_overlap.py    # Full document overlap chunking
│   ├── chunk_page_overlap.py    # Page-based overlap chunking
│   ├── chunk_full_summary.py    # Full document summarization
│   ├── chunk_page_summary.py    # Page-based summarization
│   ├── chunk_semantic_splitter_langchain.py  # LangChain semantic chunking
│   └── chunk_per_page_overlap_langchain.py   # Per-page LangChain chunking
│
├── semantic/                     # Semantic chunking implementations
│   ├── README.md
│   ├── semantic_chunker_openai.py    # Azure OpenAI semantic chunking
│   └── semantic_splitter_langchain_mini.py   # Lightweight semantic chunking
│
├── utils/                        # Utility scripts
│   ├── README.md
│   ├── generate_sample_data.py  # Sample data generator
│   └── full-pdf-chunk-overlap-langchain.py   # LangChain text splitting
│
├── llm_chunking/                 # LLM-based chunking modules
├── topic-clustering/             # Topic clustering implementations
├── tests/                        # Test suite
├── data/                         # Sample data
├── test-data/                    # Test results
├── outputs/                      # Analysis outputs
└── playground/                   # Experimental code
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd chunking
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file with your Azure OpenAI credentials:
   ```env
   AIService__Compass_Key=your_azure_openai_key
   AIService__Compass_Endpoint=your_azure_endpoint
   GPT4O_VLM_API_VERSION=2024-12-01-preview
   AIService__Compass_Models__Embedding=text-embedding-3-large
   AIService__Compass_Models__Completion=gpt-4o
   ```

## Usage Examples

### 1. Semantic Chunking with Azure OpenAI
```bash
python semantic/semantic_chunker_openai.py --input document.pdf --output chunks.json --chunk-size 500 --chunk-overlap 50
```

### 2. Full Document Overlap Summarization
```bash
python chunking-strategies/chunk_full_overlap.py --input document.pdf --output summaries.json --chunk-size 300 --overlap 50 --max-tokens 300
```

### 3. Page-based Chunking
```bash
python chunking-strategies/chunk_page_overlap.py --input document.pdf --output page_chunks.json --chunk-size 200 --overlap 25
```

### 4. Comprehensive Visualization
```bash
python visualization/enhanced_visualizer.py --strategy-files chunks.json summaries.json page_chunks.json --output-dir visualizations --max-chunks 30
```

### 5. Streamlit Dashboard
```bash
# Launch the interactive dashboard
streamlit run Home.py
```

### 6. Generate Sample Data for Dashboard
```bash
# Generate sample data for testing the dashboard
python utils/generate_sample_data.py
```

### 7. 3D t-SNE Visualization
```bash
python visualization/visualize-1.py chunks.json summaries.json --output-dir visualizations --max-chunks 30
```

### 8. Run Chunking Comparison
```bash
# Compare all chunking strategies
python run_chunking_comparison.py
```

### 9. Run LLM Chunking Tests
```bash
# Test LLM-based chunking strategies
python run_llm_chunking_tests.py
```

## Dependencies

### Core Dependencies
- **numpy>=1.21.0**: Numerical computing
- **pandas>=1.3.0**: Data manipulation and analysis
- **matplotlib>=3.5.0**: Basic plotting
- **seaborn>=0.11.0**: Statistical data visualization

### Machine Learning
- **scikit-learn**: Machine learning algorithms (t-SNE, K-means, etc.)
- **plotly**: Interactive visualizations
- **streamlit**: Web application framework for data science

### PDF Processing
- **PyMuPDF**: PDF text extraction and manipulation

### AI/ML Integration
- **openai**: OpenAI API client
- **langchain**: LangChain framework
- **langchain-openai**: LangChain OpenAI integration
- **langchain-experimental**: Experimental LangChain features
- **langchain-text-splitters**: Text splitting utilities
- **langchain-community**: Community LangChain integrations

### Utilities
- **python-dotenv**: Environment variable management
- **tiktoken**: Token counting for OpenAI models
- **nltk**: Natural language processing
- **sentence-transformers**: Sentence embeddings

## Environment Setup

### Azure OpenAI Configuration
This project requires Azure OpenAI services. You'll need:
- Azure OpenAI API key
- Azure OpenAI endpoint
- Appropriate model deployments for embeddings and completions

### Environment Variables
The following environment variables must be set in your `.env` file:
- `AIService__Compass_Key`: Your Azure OpenAI API key
- `AIService__Compass_Endpoint`: Your Azure OpenAI endpoint
- `GPT4O_VLM_API_VERSION`: API version (default: 2024-12-01-preview)
- `AIService__Compass_Models__Embedding`: Embedding model deployment name
- `AIService__Compass_Models__Completion`: Completion model deployment name

## Streamlit Dashboard

The project includes a comprehensive multi-page Streamlit dashboard for interactive analysis of chunking strategies.

### Dashboard Features
- **🏠 Home Page**: Comprehensive overview of all tools and functionality
- **📊 Strategy Comparison**: Compare chunking strategies across multiple metrics
- **🔍 Semantic Analysis**: Analyze semantic coherence and context preservation
- **🎯 RAG Performance**: Evaluate retrieval-augmented generation performance
- **🤖 LLM Chunking Tests**: Test and compare LLM-based chunking strategies
- **🗂️ Topic Clustering**: Explore topic-based chunking and clustering
- **📄 Data Generator**: Generate sample documents for testing
- **📊 Visualization Suite**: Advanced visualization tools
- **⚡ Performance Benchmark**: Comprehensive performance benchmarking

### Dashboard Usage
1. **Launch**: Run `streamlit run Home.py` to start the dashboard
2. **Navigate**: Use the sidebar to switch between different experiment pages
3. **Configure**: Adjust parameters for each experiment in the page-specific sidebars
4. **Execute**: Run experiments and view results with interactive visualizations
5. **Export**: Save and download results in various formats

### Multi-Page Structure
The dashboard consists of 8 specialized pages:
1. **Strategy Comparison** - Side-by-side strategy evaluation
2. **Semantic Analysis** - Semantic coherence and context analysis
3. **RAG Performance** - Retrieval-augmented generation evaluation
4. **LLM Chunking Tests** - LLM-based chunking strategy testing
5. **Topic Clustering** - Topic-based clustering analysis
6. **Data Generator** - Sample document generation
7. **Visualization Suite** - Advanced visualization tools
8. **Performance Benchmark** - Comprehensive performance analysis

### Data Format for Dashboard
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

## Output Formats
All chunking scripts output JSON files with the following structure:
```json
[
  {
    "chunk_index": 0,
    "text": "chunk content...",
    "strategy": "strategy_name",
    "tokens": 150,
    "summary": "summary text...",
    "overlap_tokens": 25
  }
]
```

### Visualization Outputs
- **HTML files**: Interactive Plotly visualizations
- **PNG files**: Static matplotlib/seaborn plots
- **JSON files**: Analysis metrics and data

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Azure authentication**: Verify your `.env` file contains correct Azure OpenAI credentials
3. **Memory issues**: Reduce `--max-chunks` parameter for large documents
4. **PDF processing**: Ensure PDF files are not corrupted or password-protected

### Performance Tips
- Use smaller `--max-chunks` values for faster processing
- Process documents in smaller sections for very large PDFs
- Consider using async versions for better performance with multiple documents

## Contributing

When contributing to this project:
1. Follow the existing code style
2. Add appropriate error handling
3. Update requirements.txt if adding new dependencies
4. Test with sample PDF documents
5. Update documentation as needed

## License

This project is developed for ADNOC internal use. Please ensure compliance with your organization's policies and licensing requirements. 