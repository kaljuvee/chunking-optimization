# Chunking Strategies

This directory contains various chunking strategy implementations and algorithms.

## Files

- `chunk_full_overlap.py` - Full document overlap chunking strategy
- `chunk_full_summary.py` - Full document summary-based chunking
- `chunk_page_overlap.py` - Page-based overlap chunking
- `chunk_page_summary.py` - Page-based summary chunking
- `chunk_per_page_overlap_langchain.py` - LangChain-based per-page overlap chunking
- `chunk_semantic_splitter_langchain.py` - LangChain semantic splitter implementation

## Usage

Each strategy can be run independently to test different chunking approaches:

```bash
python chunking-strategies/chunk_full_overlap.py
python chunking-strategies/chunk_semantic_splitter_langchain.py
```

## Strategy Types

1. **Overlap-based**: Maintains context through overlapping chunks
2. **Summary-based**: Uses document summaries for chunk boundaries
3. **Page-based**: Respects document page boundaries
4. **Semantic**: Uses semantic analysis for intelligent chunking
5. **LangChain Integration**: Leverages LangChain's chunking capabilities

## Comparison

Use the main comparison script to evaluate all strategies:

```bash
python run_chunking_comparison.py
``` 