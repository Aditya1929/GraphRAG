import json
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class VectorSearch:
    def __init__(self, chunks_file):
        
        with open(chunks_file, 'r') as f:
            self.chunks = json.load(f)

        print(f"Loaded {len(self.chunks)} chunks")

        embeddings = np.array([chunk["embedding"] for chunk in self.chunks])
        embeddings = embeddings.astype('float32')

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        print(f"Created FAISS index with {self.index.ntotal} vectors")
        print(f"Dimension: {dimension}")

    def search(self, query, top_k = 5):
        
        query_embedding = client.embeddings.create(
            input=query.replace("\n", " "),
            model="text-embedding-3-small"
        ).data[0].embedding

        query_vector = np.array([query_embedding]).astype('float32')

        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'chunk': self.chunks[idx],
                'distance': float(distances[0][i]),
                'rank': i + 1
            })
        
        return results
    
if __name__ == "__main__":

    search = VectorSearch("app\data\chunks_with_embeddings.json")

    query = "What are exosomes?"
    print(f"\nQuery: {query}")

    results = search.search(query, top_k=5)
    print(f"NUmebr of results: {len(results)}")

    for result in results:
        print(f"\nRank {result['rank']} (distance: {result['distance']:.4f})")
        print(f"Soruce: {result['chunk']['source_file']}")
        print(f"Text preview:")
        print(result['chunk']['text'][:300])