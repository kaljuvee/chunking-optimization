#!/usr/bin/env python3
"""
ADNOC Semantic Chunking Visualization Tool
"""

import os
import json
import argparse
import numpy as np
from dotenv import load_dotenv
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()
AZURE_API_KEY = os.getenv("AIService__Compass_Key")
AZURE_ENDPOINT = os.getenv("AIService__Compass_Endpoint")
AZURE_API_VERSION = os.getenv("GPT4O_VLM_API_VERSION", "2024-12-01-preview")
DEPLOYMENT_NAME = os.getenv("AIService__Compass_Models__Embedding", "text-embedding-3-large")

class SemanticChunkingVisualizer:
    def __init__(self):
        self.embeddings_client = AzureOpenAIEmbeddings(
            azure_deployment=DEPLOYMENT_NAME,
            openai_api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            openai_api_version=AZURE_API_VERSION
        )
        self.chunks_data = {}
        self.embeddings_cache = {}

    def load_chunking_strategies(self, strategy_files):
        print("📚 Loading chunking strategies...")
        for path in strategy_files:
            strategy_name = os.path.splitext(os.path.basename(path))[0]
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.chunks_data[strategy_name] = json.load(f)
                print(f"   ✅ {strategy_name}: {len(self.chunks_data[strategy_name])} chunks")
            else:
                print(f"   ❌ {strategy_name}: File not found - {path}")

    def generate_embeddings(self, strategy_name, max_chunks=50):
        print(f"🧠 Generating embeddings for {strategy_name}...")
        if strategy_name in self.embeddings_cache:
            return self.embeddings_cache[strategy_name]

        chunks = self.chunks_data[strategy_name][:max_chunks]
        texts = [chunk.get('text') or chunk.get('summary') or str(chunk) for chunk in chunks]
        texts = [t[:8000] for t in texts]

        try:
            embeddings = self.embeddings_client.embed_documents(texts)
            self.embeddings_cache[strategy_name] = {
                'embeddings': np.array(embeddings),
                'texts': texts,
                'chunks': chunks
            }
            return self.embeddings_cache[strategy_name]
        except Exception as e:
            print(f"   ❌ Error generating embeddings: {e}")
            return None

    def create_tsne_visualization(self, strategy_comparisons, output_dir):
        print("📊 Creating 3D t-SNE visualization...")
        fig = go.Figure()
        colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']

        for idx, (strategy_name, data) in enumerate(strategy_comparisons.items()):
            embeddings = data['embeddings']
            if len(embeddings) < 4:
                print(f"   ⚠️ Skipping {strategy_name} (too few samples for 3D)")
                continue

            tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings)-1))
            coords = tsne.fit_transform(embeddings)
            cluster_labels = KMeans(n_clusters=min(5, len(embeddings)), n_init='auto', random_state=42).fit_predict(embeddings)

            for label in np.unique(cluster_labels):
                mask = cluster_labels == label
                fig.add_trace(go.Scatter3d(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    z=coords[mask, 2],
                    mode='markers',
                    marker=dict(size=5, color=colors[label % len(colors)], opacity=0.75),
                    name=f"{strategy_name} - Topic {label+1}",
                    text=[data['texts'][i][:100] for i in range(len(data['texts'])) if mask[i]]
                ))

        fig.update_layout(title="3D t-SNE Visualization of Semantic Chunking Strategies",
                          scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                          height=800, showlegend=True)
        output_file = os.path.join(output_dir, "tsne_comparison_3d.html")
        fig.write_html(output_file)
        print(f"   ✅ Saved: {output_file}")

    def run(self, strategy_paths, output_dir="visualizations", max_chunks=30):
        os.makedirs(output_dir, exist_ok=True)
        self.load_chunking_strategies(strategy_paths)

        comparisons = {}
        for name in self.chunks_data:
            data = self.generate_embeddings(name, max_chunks)
            if data:
                comparisons[name] = data

        if comparisons:
            self.create_tsne_visualization(comparisons, output_dir)
        else:
            print("❌ No embeddings generated.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="visualizations")
    parser.add_argument("--max-chunks", type=int, default=30)
    parser.add_argument("strategy_files", nargs="+", help="List of JSON chunk files")
    args = parser.parse_args()

    print("🏢 ADNOC Semantic Chunking Analysis Tool")
    visualizer = SemanticChunkingVisualizer()
    visualizer.run(args.strategy_files, args.output_dir, args.max_chunks)

if __name__ == "__main__":
    main()
