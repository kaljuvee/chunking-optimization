# Enhanced Chunking and Topic Clustering

This directory contains advanced chunking and analysis tools that improve upon traditional chunking methods by generating questions from chunks and performing topic clustering for better index understanding.

## 🎯 Key Features

### 1. Enhanced Chunking (`enhanced_chunker.py`)
- **Improved Chunk Quality**: Creates well-sized chunks (100-500 words) with better boundaries
- **Overlapping Chunks**: Maintains context between chunks for better retrieval
- **Question Generation**: Automatically generates 5 questions per chunk for improved search
- **Topic Extraction**: Extracts 3-5 key topics from each chunk
- **Semantic Coherence**: Calculates coherence scores to ensure chunk quality

### 2. Question Clustering (`question_clustering.py`)
- **Intelligent Clustering**: Groups similar questions using multiple algorithms (K-means, DBSCAN, Agglomerative)
- **Optimal Cluster Detection**: Automatically determines the best number of clusters
- **Topic Extraction**: Extracts representative topics from each question cluster
- **Cluster Analysis**: Provides comprehensive metrics and visualizations
- **Coherence Scoring**: Measures semantic coherence within clusters

### 3. Enhanced Analysis Runner (`run_enhanced_analysis.py`)
- **Complete Pipeline**: Combines enhanced chunking with question clustering
- **Multiple Document Types**: Handles meetings, deals, technical specifications
- **Comprehensive Output**: Generates detailed analysis reports and visualizations
- **Index Insights**: Provides topics and clusters for agent decision making

## 🚀 Quick Start

### 1. Run Enhanced Analysis
```bash
cd topic-clustering
python run_enhanced_analysis.py
```

This will:
- Process test documents (meeting minutes, deal analysis, technical specs)
- Create enhanced chunks with questions and topics
- Cluster questions and extract representative topics
- Generate visualizations and analysis reports

### 2. Use Individual Components

#### Enhanced Chunking Only
```python
from enhanced_chunker import EnhancedChunker

chunker = EnhancedChunker()
chunks = chunker.chunk_text(
    text="Your document text here...",
    generate_questions=True,
    extract_topics=True
)
```

#### Question Clustering Only
```python
from question_clustering import QuestionClusterer

clusterer = QuestionClusterer()
analysis = clusterer.cluster_questions(
    chunks=your_chunks_with_questions,
    method='auto'  # or 'kmeans', 'dbscan', 'agglomerative'
)
```

## 📊 Output Structure

### Enhanced Chunks
Each enhanced chunk contains:
```json
{
  "chunk_id": "chunk_0001",
  "text": "chunk content...",
  "word_count": 250,
  "questions": [
    "What are the main performance metrics?",
    "How does the system handle scalability?",
    "What are the technical requirements?"
  ],
  "topics": [
    "performance metrics",
    "system scalability",
    "technical requirements"
  ],
  "metadata": {
    "semantic_coherence_score": 0.85,
    "boundary_quality": "excellent",
    "overlap_with_previous": 25
  }
}
```

### Question Clusters
Each cluster contains:
```json
{
  "cluster_id": 0,
  "size": 15,
  "topics": ["performance analysis", "system metrics", "technical evaluation"],
  "centroid_question": "What are the key performance indicators?",
  "coherence_score": 0.78,
  "questions": ["list of questions in cluster"]
}
```

## 🎯 Use Cases

### 1. Meeting Index Understanding
For meeting documents, the system helps agents understand:
- What types of questions are commonly asked about meetings
- What topics are covered in meeting documents
- When to retrieve from meeting vs deal indices

### 2. Deal Index Analysis
For deal documents, the system provides:
- Question patterns for deal-related queries
- Topic clusters for different deal types
- Improved retrieval accuracy for deal-specific questions

### 3. Agent Decision Making
The extracted topics help agents:
- Determine which index to query based on question type
- Understand the scope and content of available documents
- Provide more accurate and relevant responses

## 🔧 Configuration

### Enhanced Chunker Parameters
```python
# In enhanced_chunker.py
self.min_chunk_size = 100      # Minimum words per chunk
self.max_chunk_size = 500      # Maximum words per chunk
self.target_chunk_size = 300   # Target words per chunk
self.overlap_size = 100        # Overlap between chunks
self.max_questions_per_chunk = 5  # Questions per chunk
```

### Question Clusterer Parameters
```python
# In question_clustering.py
self.min_clusters = 3          # Minimum clusters
self.max_clusters = 15         # Maximum clusters
self.min_cluster_size = 2      # Minimum questions per cluster
self.max_cluster_size = 50     # Maximum questions per cluster
```

## 📈 Performance Metrics

### Chunking Quality
- **Chunk Size Distribution**: Ensures chunks are neither too small nor too large
- **Semantic Coherence**: Measures how well chunks maintain semantic meaning
- **Boundary Quality**: Assesses chunk boundary placement

### Clustering Quality
- **Silhouette Score**: Measures cluster separation and cohesion
- **Coherence Score**: Measures semantic similarity within clusters
- **Cluster Size Distribution**: Ensures balanced cluster sizes

## 🎨 Visualizations

The system generates:
1. **Cluster Size Distribution**: Bar chart showing question distribution
2. **Coherence Scores**: Visualization of cluster quality
3. **Topics per Cluster**: Analysis of topic diversity
4. **Summary Statistics**: Comprehensive metrics overview

## 🔄 Integration with Existing System

### Index Integration
```python
# Use extracted topics for index description
index_description = {
    "meeting_index": {
        "topics": ["performance review", "budget allocation", "risk assessment"],
        "question_patterns": ["what was discussed", "what decisions were made"],
        "content_type": "meeting_minutes"
    },
    "deal_index": {
        "topics": ["customer requirements", "technical specifications", "pricing"],
        "question_patterns": ["what are the requirements", "what is the deal value"],
        "content_type": "deal_analysis"
    }
}
```

### Agent Decision Making
```python
# Use topics to determine which index to query
def select_index(user_question, index_descriptions):
    question_topics = extract_topics(user_question)
    
    for index_name, description in index_descriptions.items():
        if any(topic in description["topics"] for topic in question_topics):
            return index_name
    
    return "default_index"
```

## 📁 File Structure

```
topic-clustering/
├── enhanced_chunker.py          # Enhanced chunking with questions
├── question_clustering.py       # Question clustering and topic extraction
├── run_enhanced_analysis.py     # Complete analysis pipeline
├── README.md                    # This file
└── test-data/enhanced/          # Output directory
    ├── meeting_minutes_enhanced_chunks.json
    ├── deal_analysis_enhanced_chunks.json
    ├── question_clustering_results.json
    ├── cluster_visualization.png
    └── enhanced_analysis_results.json
```

## 🚀 Next Steps

1. **Run the enhanced analysis** to see the system in action
2. **Review the generated chunks** to understand question generation quality
3. **Analyze the clusters** to see topic extraction effectiveness
4. **Integrate with your agent system** using the extracted topics
5. **Customize parameters** based on your specific document types

## 🔧 Troubleshooting

### Common Issues
1. **OpenAI API Access**: Ensure your `.env` file has correct OpenAI API credentials
2. **Memory Issues**: For large documents, consider processing in smaller batches
3. **Clustering Quality**: Adjust clustering parameters if results are unsatisfactory

### Performance Tips
1. **Batch Processing**: Process multiple documents together for better clustering
2. **Parameter Tuning**: Adjust chunk sizes based on your document characteristics
3. **Caching**: Cache results for repeated analysis of the same documents 