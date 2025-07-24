#!/usr/bin/env python3
"""
Hierarchical Chunker (Page-wise) with Semantic Similarity Filtering
==================================================================

Performs multi-level hierarchical chunking of PDF text per page
and post-processes chunks using GPT embeddings to ensure semantic
coherence using cosine similarity.
"""

import json
import argparse
import os
import fitz  # PyMuPDF
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# OpenAI credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

print(f"🔧 Using OpenAI with embedding model: {EMBEDDING_MODEL}")

# Validate API key
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

import logging
import httpx
import tiktoken
from azure_hierarchical import HierarchicalChunker  # your existing class assumed

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")

def truncate_to_token_limit(text: str, max_tokens: int = 8192) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

def get_embedding(text: str) -> List[float]:
    try:
        safe_text = truncate_to_token_limit(text)
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=safe_text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return [0.0] * 1536

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    dot = np.dot(vec1, vec2)
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

def semantic_filter(chunks: List[Dict], threshold: float = 0.7) -> List[Dict]:
    if not chunks:
        return []

    filtered = []
    prev_chunk = chunks[0]
    prev_emb = get_embedding(prev_chunk['text'])

    for curr_chunk in chunks[1:]:
        curr_emb = get_embedding(curr_chunk['text'])
        sim = cosine_similarity(prev_emb, curr_emb)

        if sim >= threshold:
            prev_chunk['text'] += "\n" + curr_chunk['text']
            prev_chunk['metadata']['word_count'] = len(prev_chunk['text'].split())
            prev_emb = get_embedding(prev_chunk['text'])
        else:
            filtered.append(prev_chunk)
            prev_chunk = curr_chunk
            prev_emb = curr_emb

    filtered.append(prev_chunk)
    return filtered

def main():
    parser = argparse.ArgumentParser(description="Page-wise hierarchical chunking + semantic similarity filter")
    parser.add_argument("--input", "-i", required=True, help="PDF input file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--flat-output", action='store_true', help="Flatten hierarchy before similarity filtering")
    args = parser.parse_args()

    logger.info(f"📄 Reading PDF: {args.input}")
    doc = fitz.open(args.input)
    chunker = HierarchicalChunker()

    all_chunks = []
    for i in range(len(doc)):
        page_text = doc.load_page(i).get_text("text").strip()
        if not page_text:
            continue

        logger.info(f"📃 Chunking Page {i+1}/{len(doc)}")
        try:
            hierarchy = chunker.chunk_text(page_text)
        except Exception as e:
            logger.warning(f"⚠️ Failed to chunk page {i+1}: {e}")
            continue

        flat_chunks = chunker.get_flat_chunks(hierarchy)
        for chunk in flat_chunks:
            chunk['metadata']['page'] = i
        all_chunks.extend(flat_chunks)

    doc.close()

    logger.info(f"🔍 Filtering {len(all_chunks)} chunks using cosine similarity ≥ {0.7}")
    filtered_chunks = semantic_filter(all_chunks, threshold=0.7)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(filtered_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Saved {len(filtered_chunks)} semantically filtered pagewise chunks to {args.output}")

if __name__ == "__main__":
    main()
