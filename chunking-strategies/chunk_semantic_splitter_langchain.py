#!/usr/bin/env python3
import fitz  # PyMuPDF
import json
import argparse
from langchain_text_splitters import SemanticChunker
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# Load API key from .env if available
load_dotenv()

def extract_pdf_text(path):
    doc = fitz.open(path)
    text = "\n\n".join(
        doc.load_page(i).get_text("text").strip()
        for i in range(len(doc))
        if doc.load_page(i).get_text("text").strip()
    )
    doc.close()
    return text

def main():
    parser = argparse.ArgumentParser(description="Semantic chunking using text-embedding-3-large")
    parser.add_argument("--input", "-i", required=True, help="PDF path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    parser.add_argument("--chunk-size", type=int, default=500, help="Approx. chunk size in tokens")
    args = parser.parse_args()

    print(f"📄 Reading PDF: {args.input}")
    text = extract_pdf_text(args.input)

    # Use OpenAI text-embedding-3-large
    embedder = OpenAIEmbeddings(model="text-embedding-3-large")
    splitter = SemanticChunker(embeddings=embedder, chunk_size=args.chunk_size)

    print("🔍 Performing semantic chunking...")
    chunks = splitter.split_text(text)

    structured = [{
        "chunk_index": i,
        "text": chunk,
        "strategy": "semantic_split_text-embedding-3-large"
    } for i, chunk in enumerate(chunks)]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! {len(chunks)} semantic chunks saved to {args.output}")

if __name__ == "__main__":
    main()
