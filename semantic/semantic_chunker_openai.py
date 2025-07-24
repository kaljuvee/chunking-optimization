#!/usr/bin/env python3
import fitz  # PyMuPDF
import json
import argparse
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# Load environment variables from .env
load_dotenv()

# Get OpenAI credentials from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

def extract_pdf_text(pdf_path):
    """Extract and concatenate all text from a PDF."""
    doc = fitz.open(pdf_path)
    all_text = "\n\n".join(
        page.get_text("text").strip()
        for page in doc
        if page.get_text("text").strip()
    )
    doc.close()
    return all_text

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Semantic chunking using OpenAI embeddings")
    parser.add_argument("--input", "-i", required=True, help="Path to input PDF file")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=50, help="Token overlap between chunks")
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large", help="Embedding model to use")
    args = parser.parse_args()

    print(f"📄 Reading PDF: {args.input}")
    text = extract_pdf_text(args.input)
    print(f"🧠 Extracted {len(text.split())} words (~{len(text)} characters)")

    print("🔗 Connecting to OpenAI...")
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=args.embedding_model
    )

    print("🧠 Running SemanticChunker...")
    splitter = SemanticChunker(embeddings=embeddings)

    chunks = splitter.split_text(text)

    # Structure output with metadata
    structured = [{
        "chunk_index": i,
        "text": chunk,
        "strategy": "semantic_split_openai"
    } for i, chunk in enumerate(chunks)]

    # Preview first chunk
    if structured:
        print("\n👻 Sample chunk preview:")
        print(structured[0]["text"][:300] + "...")
    else:
        print("⚠️ No chunks were generated.")

    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! {len(structured)} semantic chunks saved to {args.output}")

if __name__ == "__main__":
    main()
