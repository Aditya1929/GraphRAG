import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_embedding(text, model="text-embedding-3-small"):

    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=text,
        model=model
    )

    return response.data[0].embedding

def generate_all_embeddings(chunks_file, output_file):

    with open(chunks_file, 'r') as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")
    print(f"Generating embeddings...")
    print()

    for i, chunk in enumerate(chunks):
        try:
            embedding = generate_embedding(chunk["text"])
            chunk['embedding'] = embedding

            time.sleep(0.1)
        except Exception as e:
            print(f"Error on hcunk {i}: {e}")
            chunk['embedding'] = None

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(chunks, f)

        print(f"Generated embeddings for {len(chunks)} chunks")
        print(f"Saved to: {output_path}")
        print(f"Embedding dimension: {len(chunks[0]['embedding'])}")

        return chunks
    
if __name__ == "__main__":
    chunks_with_embeddings = generate_all_embeddings(
        chunks_file="app\data\chunks.json",
        output_file="app\data\chunks_with_embeddings.json"
    )

    

    
