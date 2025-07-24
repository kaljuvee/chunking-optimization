# API Configuration for Chunking Strategies

## Overview
All chunking strategies now support both OpenAI and Azure OpenAI APIs, automatically detecting which one to use based on the `LLM_MODEL` environment variable.

## Configuration

### Environment Variables

#### For OpenAI (Default)
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults shown)
LLM_MODEL=gpt-4o
OPENAI_MODEL_NAME=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

#### For Azure OpenAI
```bash
# Required
LLM_MODEL=azure-gpt-4o  # Must contain "azure" to trigger Azure mode
AIService__Compass_Key=your_azure_api_key_here
AIService__Compass_Endpoint=your_azure_endpoint_here

# Optional (defaults shown)
GPT4O_VLM_API_VERSION=2024-12-01-preview
AIService__Compass_Models__Completion=gpt-4o
AIService__Compass_Models__Embedding=text-embedding-3-large
```

## Detection Logic

The system automatically detects which API to use based on the `LLM_MODEL` environment variable:

- **Default (OpenAI)**: If `LLM_MODEL` is not set or doesn't contain "azure"
- **Azure OpenAI**: If `LLM_MODEL` contains "azure" (case-insensitive)

### Examples

```bash
# Uses OpenAI (default)
LLM_MODEL=gpt-4o
LLM_MODEL=gpt-4-turbo
LLM_MODEL=claude-3-opus

# Uses Azure OpenAI
LLM_MODEL=azure-gpt-4o
LLM_MODEL=azure-gpt-4-turbo
LLM_MODEL=my-azure-deployment
```

## Supported Files

All chunking strategy files now support this configuration:

### Regular Chunking
- `chunk_full_overlap.py`
- `chunk_full_summary.py`

### Async Chunking
- `chunk_page_overlap.py`
- `chunk_page_summary.py`

### Hierarchical Chunking with Embeddings
- `Hierarchial-pagewise-pdf-semantic-embeddings.py`
- `Hierarchial-pagewise-semantic-embeddings-V1.py`
- `Hierarchial-whole-pdf-semantic-embeddings.py`
- `Hierarchial-whole-pdf-semantic-embeddings-V0.py`

## Usage

1. Set up your `.env` file with the appropriate credentials
2. Set `LLM_MODEL` to control which API to use
3. Run any chunking strategy - it will automatically use the correct API

### Example .env file for OpenAI
```bash
OPENAI_API_KEY=sk-your-openai-key-here
LLM_MODEL=gpt-4o
```

### Example .env file for Azure
```bash
LLM_MODEL=azure-gpt-4o
AIService__Compass_Key=your-azure-key-here
AIService__Compass_Endpoint=https://your-resource.openai.azure.com/
```

## Troubleshooting

- **"Public access is disabled" error**: Make sure you're using the correct API credentials
- **"Authentication failed"**: Check your API keys and endpoints
- **"Model not found"**: Verify the model names in your environment variables

The system will print which API it's using when you run any chunking strategy:
```
🔧 Using OpenAI with model: gpt-4o
```
or
```
🔧 Using Azure OpenAI with embedding model: text-embedding-3-large
``` 