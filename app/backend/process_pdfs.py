import json
from pathlib import Path
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)

        text_by_page = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            text_by_page.append({
                "page_number": page_num,
                "text": text
            })

        full_text = "\n\n".join([p["text"] for p in text_by_page])

        metadata = {
            "filename": Path(pdf_path).name,
            "num_pages": len(reader.pages),
            "has_text": len(full_text.strip()) > 0
        }

        return {
            "metadata": metadata,
            "full_text": full_text,
            "pages": text_by_page
        }
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")


def process_all_pdfs(input_dir, output_dir):

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    results = []

    for pdf_file in pdf_files:
        extracted = extract_text_from_pdf(pdf_file)
        
        if extracted:
            output_file = output_path / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted, f, indent=2, ensure_ascii=False)
            
            results.append(extracted["metadata"])
        else:
            print("Failed to extract text from PDF")
    
    return results

if __name__ == "__main__":
    process_all_pdfs(r"app\data\raw_pdfs", r"app\data\processed")

