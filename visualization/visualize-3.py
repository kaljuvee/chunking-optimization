#!/usr/bin/env python3
"""
ADNOC Semantic Embedding Visualization (2D & 3D with Optional Raw PDF Embedding)
"""

import os
import json
import argparse
import numpy as np
import fitz  # PyMuPDF
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import plotly.graph_objects as go
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()
AZURE_API_KEY = os.getenv("AIService__Compass_Key")
AZURE_ENDPOINT = os.getenv("AIService__Compass_Endpoint")
AZURE_API_VERSION = os.getenv("GPT4O_VLM_API_VERSION", "2024-12-01-preview")
DEPLOYMENT_NAME = os.getenv("AIService__Compass_Models__Embedding", "text-embedding-3-large")

class SimpleEmbeddingVisualizer:
    def __init__(self):
        self.embeddings_client = AzureOpenAIEmbeddings(
            azure_deployment=DEPLOYMENT_NAME,
            openai_api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            openai_api_version=AZURE_API_VERSION
        )
        self.chunks_data = {}
        self.embeddings_cache = {}
        self.raw_embedding = None

    def load_json_chunks(self, paths):
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.chunks_data[name] = json.load(f)
                print(f"✅ Loaded {name} ({len(self.chunks_data[name])} chunks)")
            else:
                print(f"❌ File not found: {path}")

    def embed(self, strategy_name, max_chunks):
        if strategy_name in self.embeddings_cache:
            return self.embeddings_cache[strategy_name]

        chunks = self.chunks_data[strategy_name][:max_chunks]
        texts = [chunk.get("text") or chunk.get("summary") or str(chunk) for chunk in chunks]
        texts = [t[:8000] for t in texts]
        embeddings = self.embeddings_client.embed_documents(texts)
        self.embeddings_cache[strategy_name] = {"texts": texts, "embeddings": np.array(embeddings)}
        return self.embeddings_cache[strategy_name]

    def extract_pdf_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        full_text = "\n\n".join(page.get_text("text") for page in doc if page.get_text("text"))
        doc.close()
        return full_text.strip()

    def embed_raw_pdf(self, raw_pdf_path):
        print(f"📄 Embedding raw PDF text from {raw_pdf_path}...")
        text = self.extract_pdf_text(raw_pdf_path)
        text = text[:8000]  # Truncate to max token length
        embedding = self.embeddings_client.embed_query(text)
        self.raw_embedding = {"text": text, "embedding": np.array(embedding)}

    def plot_tsne_2d(self, data_dict, output_dir):
        print("📊 Creating 2D t-SNE plot...")
        fig = go.Figure()

        all_coords = []
        legend_map = []

        for name, data in data_dict.items():
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data['embeddings']) - 1))
            coords = tsne.fit_transform(data['embeddings'])
            fig.add_trace(go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                name=name,
                text=[text[:80] for text in data['texts']],
                marker=dict(size=6, opacity=0.7)
            ))
            all_coords.append(coords)
            legend_map.append(name)

        if self.raw_embedding:
            for coords, name in zip(all_coords, legend_map):
                distances = cosine_distances([self.raw_embedding['embedding']], data_dict[name]['embeddings'])[0]
                min_idx = np.argmin(distances)
                fig.add_trace(go.Scatter(
                    x=[coords[min_idx][0], coords[min_idx][0]],
                    y=[coords[min_idx][1], coords[min_idx][1]],
                    mode="lines",
                    line=dict(color="black", dash="dot"),
                    showlegend=False
                ))
            fig.add_trace(go.Scatter(
                x=[0],
                y=[0],
                mode="markers+text",
                name="Original PDF",
                marker=dict(size=10, color="black"),
                text=["Raw Document"],
                textposition="top center"
            ))

        fig.update_layout(
            title="2D t-SNE of Embeddings",
            height=600,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        fig.write_html(os.path.join(output_dir, "tsne_2d_visualize3.html"))
        fig.write_image(os.path.join(output_dir, "tsne_2d_visualize3.png"), scale=2)
        print("✅ Saved: tsne_2d_visualize3.html and tsne_2d_visualize3.png")

    def plot_tsne_3d(self, data_dict, output_dir):
        print("📊 Creating 3D t-SNE plot...")
        fig = go.Figure()

        for name, data in data_dict.items():
            if len(data['embeddings']) < 4:
                continue
            tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(data['embeddings']) - 1))
            coords = tsne.fit_transform(data['embeddings'])
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                name=name,
                text=[text[:80] for text in data['texts']],
                marker=dict(size=4, opacity=0.75)
            ))

        if self.raw_embedding:
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode="markers+text",
                name="Original PDF",
                marker=dict(size=8, color="black"),
                text=["Raw Document"],
                textposition="top center"
            ))

        fig.update_layout(
            title="3D t-SNE of Embeddings",
            height=800,
            paper_bgcolor='white',
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='white'
            )
        )
        fig.write_html(os.path.join(output_dir, "tsne_3d_visualize3.html"))
        fig.write_image(os.path.join(output_dir, "tsne_3d_visualize3.png"), scale=2)
        print("✅ Saved: tsne_3d_visualize3.html and tsne_3d_visualize3.png")

    def run(self, files, output_dir, max_chunks, raw_pdf=None):
        os.makedirs(output_dir, exist_ok=True)
        self.load_json_chunks(files)
        data_dict = {}
        for name in self.chunks_data:
            data = self.embed(name, max_chunks)
            if data:
                data_dict[name] = data
        if raw_pdf:
            self.embed_raw_pdf(raw_pdf)
        self.plot_tsne_2d(data_dict, output_dir)
        self.plot_tsne_3d(data_dict, output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tsne_outputs")
    parser.add_argument("--max-chunks", type=int, default=30)
    parser.add_argument("--raw-text", help="Optional raw PDF path to compare against chunks")
    parser.add_argument("strategy_files", nargs="+", help="Path(s) to JSON chunked files")
    args = parser.parse_args()

    print("🚀 ADNOC Embedding Visualization (2D & 3D, Raw PDF vs Chunks)")
    visualizer = SimpleEmbeddingVisualizer()
    visualizer.run(args.strategy_files, args.output_dir, args.max_chunks, args.raw_text)

if __name__ == "__main__":
    main()
