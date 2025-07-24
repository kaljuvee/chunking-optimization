#!/usr/bin/env python3
import fitz  # PyMuPDF
import json
import argparse
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# Load environment variables
load_dotenv()

# OpenAI credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

def extract_pdf_text(path):
    doc = fitz.open(path)
    text = "\n\n".join(
        doc.load_page(i).get_text("text").strip()
        for i in range(len(doc))
        if doc.load_page(i).get_text("text").strip()
    )
    doc.close()
    return text

def split_sentences(text):
    """Simple sentence splitting using regex - no NLTK required"""
    # Split on sentence endings followed by whitespace or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty sentences and clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def get_embeddings(texts, client, model):
    """Get embeddings for a list of texts using OpenAI API"""
    if isinstance(texts, str):
        texts = [texts]
    
    try:
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"⚠️ Error getting embeddings: {e}")
        return None

def semantic_chunk_text(text, client, model, chunk_size=500, similarity_threshold=0.7):
    """Simple semantic chunking using OpenAI embeddings"""
    
    # Split text into sentences using simple regex
    sentences = split_sentences(text)
    
    if len(sentences) <= 1:
        return [text]
    
    # Get embeddings for all sentences
    embeddings = get_embeddings(sentences, client, model)
    if embeddings is None:
        # Fallback to simple chunking if embeddings fail
        return [text[i:i+chunk_size*4] for i in range(0, len(text), chunk_size*4)]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence.split())
        
        # If adding this sentence would exceed chunk size, start a new chunk
        if current_length + sentence_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            # Check semantic similarity with previous sentence if available
            if current_chunk and i > 0:
                prev_embedding = embeddings[i-1]
                curr_embedding = embeddings[i]
                
                # Calculate cosine similarity
                similarity = np.dot(prev_embedding, curr_embedding) / (
                    np.linalg.norm(prev_embedding) * np.linalg.norm(curr_embedding)
                )
                
                # If similarity is low, start a new chunk
                if similarity < similarity_threshold and current_length > chunk_size // 2:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                    continue
            
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Semantic chunking using OpenAI embeddings")
    parser.add_argument("--input", "-i", required=True, help="PDF path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    parser.add_argument("--chunk-size", type=int, default=500, help="Approx. chunk size (tokens or characters)")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap between chunks (not used in semantic chunking)")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large", help="OpenAI embedding model to use")
    args = parser.parse_args()

    print(f"📄 Reading PDF: {args.input}")
    text = extract_pdf_text(args.input)

    print(f"🔗 Connecting to OpenAI...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    print(f"🔍 Using OpenAI embedding model: {args.embedding_model}")
    print("🧠 Performing semantic chunking...")
    chunks = semantic_chunk_text(text, client, args.embedding_model, args.chunk_size)

    structured = [{
        "chunk_index": i,
        "text": chunk,
        "strategy": "semantic_split_openai_local"
    } for i, chunk in enumerate(chunks)]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! {len(structured)} semantic chunks saved to {args.output}")

if __name__ == "__main__":
    main()
