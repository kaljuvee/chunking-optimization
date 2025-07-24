# Utility Scripts

This directory contains utility scripts and helper functions for the chunking project.

## Files

- `generate_sample_data.py` - Generate sample documents for testing chunking strategies
- `full-pdf-chunk-overlap-langchain.py` - PDF-specific chunking with LangChain integration

## Usage

### Generate Sample Data

Create test documents for chunking evaluation:

```bash
python utils/generate_sample_data.py
```

This will generate various document types in the `data/` directory.

### PDF Chunking

Process PDF documents with LangChain:

```bash
python utils/full-pdf-chunk-overlap-langchain.py
```

## Features

- **Sample Data Generation**: Creates realistic test documents
- **PDF Processing**: Handles PDF-specific chunking requirements
- **LangChain Integration**: Leverages LangChain for advanced processing
- **Test Data Management**: Organizes test data for consistent evaluation

## Output

Generated files are typically saved to:
- `data/` - Sample documents
- `test-data/` - Test results and outputs
- `outputs/` - Analysis results

## Integration

These utilities support the main chunking comparison framework and can be used to:
- Generate consistent test datasets
- Process real-world PDF documents
- Prepare data for evaluation 