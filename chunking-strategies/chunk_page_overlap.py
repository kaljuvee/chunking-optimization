#!/usr/bin/env python3
import fitz  # PyMuPDF
import json
import argparse
import asyncio
import os
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env
load_dotenv()

# OpenAI credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")

# Initialize AsyncOpenAI client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

print(f"🔧 Using OpenAI with model: {MODEL_NAME}")

# Validate API key
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

# Token counting
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

# Enhanced prompt builder
def build_prompt(text: str, page_number: int, chunk_index: int) -> str:
    return f"""
You are an expert assistant helping analyze and summarize critical information from strategic business documents.

Carefully summarize the following text extracted from page {page_number + 1}, chunk {chunk_index}. Focus on:
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

async def summarize_text(text, page_number, chunk_index, max_tokens=300, retries=2):
    prompt = build_prompt(text, page_number, chunk_index)
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
            else:
                return "[Error summarizing chunk]"

async def main():
    parser = argparse.ArgumentParser(description="Chunk and summarize PDF using Azure GPT-4o")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ File not found: {args.input}")
        return

    doc = fitz.open(args.input)
    all_chunks = []

    tasks = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text("text").strip()
        if not text:
            continue
        chunks = split_text_into_chunks(text, args.chunk_size, args.overlap)
        for chunk in chunks:
            tasks.append((page_number, chunk))

    async def summarize_chunk_task(page_number, chunk):
        summary = await summarize_text(chunk["text"], page_number, chunk["chunk_index"], args.max_tokens)
        summary_tokens = count_tokens(summary)
        return {
            "page": page_number + 1,
            "chunk_index": chunk["chunk_index"],
            "strategy": "per_page_overlap_summary",
            "original_tokens": chunk["tokens"],
            "summary_tokens": summary_tokens,
            "summary": summary,
            "overlap_tokens": chunk["overlap_tokens"]
        }

    results = await asyncio.gather(*(summarize_chunk_task(pn, chunk) for pn, chunk in tasks))
    doc.close()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Done! Summaries saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
