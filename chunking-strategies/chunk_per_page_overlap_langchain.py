#!/usr/bin/env python3
import fitz  # PyMuPDF
import json
import argparse
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text, chunk_size=500, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def main():
    parser = argparse.ArgumentParser(description="Chunk each PDF page using LangChain TextSplitter (no summarization)")
    parser.add_argument("--input", "-i", required=True, help="Path to input PDF")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    doc = fitz.open(args.input)
    all_chunks = []

    for page_number in range(len(doc)):
        text = doc.load_page(page_number).get_text("text").strip()
        if not text:
            continue
        chunks = split_text(text, args.chunk_size, args.overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "page": page_number + 1,
                "chunk_index": i,
                "text": chunk,
                "strategy": "per_page_overlap_raw"
            })

    doc.close()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(all_chunks)} raw overlapping chunks to {args.output}")

if __name__ == "__main__":
    main()
