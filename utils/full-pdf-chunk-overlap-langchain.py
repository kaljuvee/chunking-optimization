#!/usr/bin/env python3
import fitz  # PyMuPDF
import json
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_pdf_text(path):
    doc = fitz.open(path)
    all_text = "\n\n".join(
        doc.load_page(i).get_text("text").strip()
        for i in range(len(doc))
        if doc.load_page(i).get_text("text").strip()
    )
    doc.close()
    return all_text

def chunk_full_document(text, chunk_size=500, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def main():
    parser = argparse.ArgumentParser(description="Chunk full PDF using LangChain (no summarization)")
    parser.add_argument("--input", "-i", required=True, help="Input PDF path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    print(f"📄 Loading and processing: {args.input}")
    full_text = extract_pdf_text(args.input)
    chunks = chunk_full_document(full_text, args.chunk_size, args.overlap)

    structured = [{
        "chunk_index": i,
        "text": chunk,
        "strategy": "full_doc_langchain"
    } for i, chunk in enumerate(chunks)]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(structured, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(structured)} full-document chunks to {args.output}")

if __name__ == "__main__":
    main()
