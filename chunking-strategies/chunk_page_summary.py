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

# Token counting using tiktoken
try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")
    count_tokens = lambda text: len(enc.encode(text))
except ImportError:
    count_tokens = lambda text: len(text.split())

# Enhanced prompt builder for page-level summaries
def build_prompt(text: str, page_number: int) -> str:
    return f"""
You are an expert assistant helping analyze and summarize critical information from strategic business documents.

Carefully summarize the following text from page {page_number + 1} of a document. Focus on:
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

async def summarize_text(text, page_number, max_tokens=300, retries=2):
    prompt = build_prompt(text, page_number)
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
                return "[Error summarizing page]"

async def main():
    parser = argparse.ArgumentParser(description="Asynchronously summarize each page of a PDF using Azure GPT-4o")
    parser.add_argument("--input", "-i", required=True, help="Path to input PDF file")
    parser.add_argument("--output", "-o", required=True, help="Path to output JSON file")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max tokens per summary")
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
        original_tokens = count_tokens(text)
        print(f"Queueing page {page_number + 1} ({original_tokens} tokens) for summarization...")
        tasks.append((page_number, text, original_tokens))

    async def summarize_task(page_number, text, original_tokens):
        summary = await summarize_text(text, page_number, args.max_tokens)
        summary_tokens = count_tokens(summary)
        return {
            "chunk_index": page_number,
            "page": page_number + 1,
            "strategy": "per_page_summary_async",
            "text": summary,
            "tokens": summary_tokens,
            "original_page_tokens": original_tokens
        }

    chunk_results = await asyncio.gather(*(summarize_task(pn, txt, toks) for pn, txt, toks in tasks))

    doc.close()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chunk_results, f, ensure_ascii=False, indent=2)
    print(f"✅ Done! Summaries saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
