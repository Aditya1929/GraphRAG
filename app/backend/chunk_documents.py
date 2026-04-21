from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_document(text, filename, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_text(text)

    chunk_dicts = []

    for i, chunk_text in enumerate(chunks):
        chunk_dicts.append({
            "chunk_id": f"{filename}_chunk_{i}",
            "source_file": filename,
            "text": chunk_text,
            "char_count": len(chunk_text)
        })

    return chunk_dicts

def process_all_documents(input_dir, output_file, chunk_size=1000, chunk_overlap=200):
    
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    print(f"Found {len(json_files)} documents to chunk")
    print(f"Chunk size: {chunk_size} characters")
    print(f"Overlap: {chunk_overlap} characters")

    all_chunks = []

    for json_file in json_files:
        print(f"Processing: {json_file.name}")

        with open(json_file, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        chunks = chunk_document(
            text=data["full_text"],
            filename=data["metadata"]["filename"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        all_chunks.extend(chunks)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_chunks, f, indent=2)
    
    return all_chunks

if __name__ == "__main__":
    
    chunks = process_all_documents(
        input_dir="app\data\processed",
        output_file="app\data\chunks.json",
        chunk_size=1000,
        chunk_overlap=200
    )

    


