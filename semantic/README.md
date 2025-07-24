# Semantic Chunking

This directory contains semantic-based chunking implementations using various AI/ML approaches.

## Files

- `semantic_chunker_openai.py` - OpenAI-based semantic chunking implementation
- `semantic_splitter_langchain_mini.py` - Lightweight LangChain semantic splitter

## Usage

Run semantic chunking implementations:

```bash
python semantic/semantic_chunker_openai.py
python semantic/semantic_splitter_langchain_mini.py
```

## Features

- **OpenAI Integration**: Uses OpenAI's embedding models for semantic analysis
- **LangChain Support**: Leverages LangChain's semantic splitting capabilities
- **Intelligent Boundaries**: Identifies semantic boundaries for optimal chunking
- **Context Preservation**: Maintains semantic coherence across chunks

## Requirements

- OpenAI API key (for OpenAI-based chunking)
- LangChain dependencies
- Appropriate model access permissions

## Configuration

Set your API keys in environment variables:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Integration

These semantic chunkers can be integrated with the main comparison framework:

```bash
python run_chunking_comparison.py --include-semantic
``` 