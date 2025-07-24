#!/usr/bin/env python3
import fitz  # PyMuPDF
import json
import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# OpenAI credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

print(f"🔧 Using OpenAI with model: {MODEL_NAME}")

# Validate API key
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

# Token counter
try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")
    count_tokens = lambda text: len(enc.encode(text))
except ImportError:
    count_tokens = lambda text: len(text.split())

def split_text_into_chunks(text, chunk_size, overlap_size):
    words = text.split()
    total_words = len(words)
    chunks = []
    start = 0
    chunk_index = 0
    while start < total_words:
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        tokens = count_tokens(chunk_text)
        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text,
            "tokens": tokens,
            "overlap_tokens": overlap_size if chunk_index > 0 else 0
        })
        chunk_index += 1
        start += chunk_size - overlap_size
    return chunks

# New enhanced summarization prompt builder
def build_prompt(text: str) -> str:
    return f"""
You are an expert assistant helping analyze and summarize critical information from strategic business documents.

Carefully summarize the following text. Focus on:
1. Key themes and insights
2. Factual data points and statistics
3. Names of organizations or stakeholders involved
4. Strategic pillars, benefits, and action plans
5. Trends or stages of AI adoption, digital transformation, and impact

Text:
'''
{text.strip()}
'''

Provide your response in bullet-point format when appropriate. Make sure the summary is self-contained and answers questions such as:
- What are the pillars or key themes?
- Which organizations or stakeholders are mentioned?
- What are the benefits or outcomes?
- What current trends or stages are described?
- What evidence (figures, results, initiatives) supports the claims?
""".strip()

def summarize_text(chunk_text, chunk_index, max_tokens=300):
    prompt = build_prompt(chunk_text)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser(description="Summarize full PDF document in overlapping chunks using Azure GPT-4o")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    # Load and merge all text from the PDF
    doc = fitz.open(args.input)
    full_text = "\n\n".join(
        doc.load_page(i).get_text("text").strip() for i in range(len(doc))
    )
    doc.close()

    # Chunk the full document text
    chunks = split_text_into_chunks(full_text, args.chunk_size, args.overlap)

    summarized_chunks = []
    for chunk in chunks:
        summary = summarize_text(chunk["text"], chunk["chunk_index"], args.max_tokens)
        summary_tokens = count_tokens(summary)
        summarized_chunks.append({
            "chunk_index": chunk["chunk_index"],
            "strategy": "full_doc_overlap_summary",
            "original_tokens": chunk["tokens"],
            "summary_tokens": summary_tokens,
            "overlap_tokens": chunk["overlap_tokens"],
            "summary": summary
        })

    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summarized_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Summarized {len(summarized_chunks)} chunks → saved to {args.output}")

if __name__ == "__main__":
    main()
