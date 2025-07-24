#!/usr/bin/env python3
import fitz  # PyMuPDF
import json
import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env values
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

# New enhanced summarization prompt builder
def build_prompt(text: str, page_range: tuple) -> str:
    return f"""
You are an expert assistant helping analyze and summarize critical information from strategic business documents.

Carefully summarize the following text extracted from pages {page_range[0]} to {page_range[1]} of a report. Focus on:
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

def summarize_text(text, page_range, max_tokens=500):
    prompt = build_prompt(text, page_range)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--pages-per-chunk", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=300)  # Added for compatibility
    parser.add_argument("--overlap", type=int, default=50)      # Added for compatibility
    args = parser.parse_args()

    doc = fitz.open(args.input)
    total_pages = len(doc)
    all_chunks = []
    chunk_index = 0

    for start in range(0, total_pages, args.pages_per_chunk):
        end = min(start + args.pages_per_chunk - 1, total_pages - 1)
        combined_text = "\n\n".join(
            doc.load_page(p).get_text("text").strip()
            for p in range(start, end + 1)
            if doc.load_page(p).get_text("text").strip()
        )
        if not combined_text.strip():
            continue

        page_range = (start + 1, end + 1)
        original_tokens = count_tokens(combined_text)
        summary = summarize_text(combined_text, page_range, args.max_tokens)
        summary_tokens = count_tokens(summary)

        all_chunks.append({
            "chunk_index": chunk_index,
            "pages": list(range(page_range[0], page_range[1] + 1)),
            "strategy": "full_doc_summary_azure",
            "text": summary,
            "tokens": summary_tokens,
            "original_pages_tokens": original_tokens
        })

        chunk_index += 1

    doc.close()

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Saved output to {args.output}")

if __name__ == "__main__":
    main()
