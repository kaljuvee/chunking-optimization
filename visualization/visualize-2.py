#!/usr/bin/env python3
"""
ADNOC Semantic Embedding Visualization (2D & 3D without Clustering)
"""

import os
import json
import argparse
import numpy as np
from dotenv import load_dotenv
from sklearn.manifold import TSNE
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

    def plot_tsne_2d(self, data_dict, output_dir):
        print("📊 Creating 2D t-SNE plot...")
        fig = go.Figure()

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

        fig.update_layout(title="2D t-SNE of Embeddings", height=600)
        fig.write_html(os.path.join(output_dir, "tsne_2d_embeddings.html"))
        print("✅ Saved: tsne_2d_embeddings.html")

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

        fig.update_layout(title="3D t-SNE of Embeddings", height=800)
        fig.write_html(os.path.join(output_dir, "tsne_3d_embeddings.html"))
        print("✅ Saved: tsne_3d_embeddings.html")

    def run(self, files, output_dir, max_chunks):
        os.makedirs(output_dir, exist_ok=True)
        self.load_json_chunks(files)
        data_dict = {}
        for name in self.chunks_data:
            data = self.embed(name, max_chunks)
            if data:
                data_dict[name] = data
        self.plot_tsne_2d(data_dict, output_dir)
        self.plot_tsne_3d(data_dict, output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tsne_outputs")
    parser.add_argument("--max-chunks", type=int, default=30)
    parser.add_argument("strategy_files", nargs="+", help="Path(s) to JSON chunked files")
    args = parser.parse_args()

    print("🚀 ADNOC Embedding Visualization (2D & 3D, No Topic Modeling)")
    visualizer = SimpleEmbeddingVisualizer()
    visualizer.run(args.strategy_files, args.output_dir, args.max_chunks)

if __name__ == "__main__":
    main()
