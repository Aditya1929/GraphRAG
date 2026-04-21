"""
3-Stage PDF → Structured JSON Extraction Pipeline

Stage 1 — Text extraction:
    pdfplumber pulls structured text, preserving headings, paragraphs,
    lists, and page numbers. Fast and free.

Stage 2 — Layout analysis:
    Detect regions where text extraction fails or produces garbage:
    tables (pdfplumber table detector), images/figures (image bbox list),
    and scanned pages (low text-density heuristic).

Stage 3 — Targeted GPT Vision:
    ONLY for problem regions identified in Stage 2. Tables → extract as
    structured JSON rows/columns. Figures → describe data + relationships.
    Scanned pages → OCR transcript. Each vision call is scoped to the
    specific region, not the full page.

Every content node carries a provenance flag: which doc, which page, and
how it was extracted (text_extraction | vision_table_parse | vision_ocr |
vision_figure).
"""

import asyncio
import base64
import json
import re
from pathlib import Path
from typing import Any, Optional

import fitz  # pymupdf — for rasterizing specific page regions
import pdfplumber
from openai import AsyncOpenAI


# ── Vision helpers ────────────────────────────────────────────────────────────

def _rasterize_region(pdf_path: str, page_index: int, bbox: Optional[tuple] = None,
                      dpi: int = 150) -> str:
    """
    Rasterize a page (or a bbox sub-region) using PyMuPDF.
    Returns a base64-encoded PNG string.

    bbox: (x0, y0, x1, y1) in pdfplumber coordinates (top-left origin, pts).
          If None, rasterize the entire page.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    if bbox:
        # pdfplumber uses top-left origin; fitz uses bottom-left.
        # Convert: fitz_y = page_height - pdfplumber_y
        ph = page.rect.height
        x0, top, x1, bottom = bbox
        rect = fitz.Rect(x0, ph - bottom, x1, ph - top)
        clip = rect
    else:
        clip = None

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode()


async def _vision_call(client: AsyncOpenAI, b64_image: str, prompt: str) -> str:
    """Send a single image + prompt to GPT-4o (vision) and return the text response."""
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}", "detail": "high"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=2000,
        temperature=0,
    )
    return response.choices[0].message.content or ""


async def _extract_table_via_vision(client: AsyncOpenAI, pdf_path: str,
                                     page_index: int, bbox: tuple) -> dict:
    b64 = _rasterize_region(pdf_path, page_index, bbox)
    prompt = (
        "Extract the table from this image as structured JSON. "
        "Return ONLY valid JSON in this format:\n"
        '{"headers": ["col1", "col2", ...], "rows": [["val1", "val2", ...], ...]}\n'
        "If there are no clear headers, use the first row as headers. "
        "Preserve all numeric values exactly."
    )
    raw = await _vision_call(client, b64, prompt)
    try:
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_text": raw}


async def _extract_figure_via_vision(client: AsyncOpenAI, pdf_path: str,
                                      page_index: int, bbox: tuple) -> str:
    b64 = _rasterize_region(pdf_path, page_index, bbox)
    prompt = (
        "Describe the data, trends, relationships, and key values shown in this figure or chart. "
        "Be specific: mention axis labels, legend entries, and any quantitative claims visible. "
        "Write 2-5 sentences."
    )
    return await _vision_call(client, b64, prompt)


async def _ocr_page_via_vision(client: AsyncOpenAI, pdf_path: str, page_index: int) -> str:
    b64 = _rasterize_region(pdf_path, page_index, dpi=200)
    prompt = (
        "Transcribe all text visible in this scanned document page. "
        "Preserve paragraph breaks and headings. Output plain text only, no commentary."
    )
    return await _vision_call(client, b64, prompt)


# ── Table quality check ───────────────────────────────────────────────────────

def _is_table_well_extracted(table_data: list) -> bool:
    """
    Heuristic: a table is 'well extracted' if it has at least 2 rows,
    at least 2 columns, and fewer than 40% of cells are None/empty.
    """
    if not table_data or len(table_data) < 2:
        return False
    col_count = max(len(row) for row in table_data)
    if col_count < 2:
        return False
    total_cells = sum(len(row) for row in table_data)
    empty_cells = sum(1 for row in table_data for cell in row if not cell)
    return (empty_cells / max(total_cells, 1)) < 0.4


# ── Section builder ───────────────────────────────────────────────────────────

def _build_sections(pages_detail: list) -> list:
    """
    Convert the flat per-page content list into a list of sections.
    A new section starts whenever a content node of type 'heading' is found.
    """
    sections = []
    current_section: dict = {"heading": None, "page": 1, "content": "", "content_nodes": []}

    for page_data in pages_detail:
        for node in page_data["content_nodes"]:
            if node["type"] == "heading":
                # Save previous section if it has content
                if current_section["content"] or current_section["content_nodes"]:
                    sections.append(current_section)
                current_section = {
                    "heading": node["content"],
                    "page": node["page"],
                    "content": "",
                    "content_nodes": [],
                }
            else:
                current_section["content_nodes"].append(node)
                if isinstance(node["content"], str):
                    current_section["content"] += node["content"] + "\n"
                elif isinstance(node["content"], dict):
                    # Tables: stringify for embedding/search
                    current_section["content"] += json.dumps(node["content"]) + "\n"

    if current_section["content"] or current_section["content_nodes"]:
        sections.append(current_section)

    return sections


# ── Heading detection ─────────────────────────────────────────────────────────

def _classify_line(line: str, chars: list) -> str:
    """
    Classify a text line as 'heading' or 'paragraph'.
    Uses font-size heuristic from pdfplumber character data.
    """
    if not line.strip():
        return "empty"

    # Find chars that match this line (approximate)
    line_stripped = line.strip()
    line_chars = [c for c in chars if line_stripped[:10] in (c.get("text", ""))]

    if line_chars:
        sizes = [c.get("size", 10) for c in line_chars if c.get("size")]
        avg_size = sum(sizes) / len(sizes) if sizes else 10
        if avg_size >= 13:  # larger than body text
            return "heading"

    # Fallback heuristics: short, ALL CAPS, or ends without period
    if len(line_stripped) < 80 and line_stripped == line_stripped.upper() and len(line_stripped) > 3:
        return "heading"

    return "paragraph"


# ── Main extraction function ──────────────────────────────────────────────────

async def extract_pdf_to_json(pdf_path: str, openai_client: AsyncOpenAI) -> dict:
    """
    Full 3-stage extraction. Returns a structured document dict with provenance.
    Vision calls are parallelized across pages.
    """
    doc_name = Path(pdf_path).name
    pages_detail = []

    # Stage 1 + 2: text extraction and layout analysis (synchronous, fast)
    pages_needing_vision: list = []  # list of coroutines to run in parallel

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            chars = page.chars  # character-level metadata (includes font size)

            # Scanned page detection: text too sparse
            is_scanned = len(raw_text.strip()) < 20 and page_num > 0

            content_nodes: list = []

            if is_scanned:
                pages_needing_vision.append(
                    _process_scanned_page(openai_client, pdf_path, page_num - 1, content_nodes)
                )
            else:
                # Parse text lines into headings / paragraphs
                for line in raw_text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    node_type = _classify_line(line, chars)
                    content_nodes.append({
                        "type": node_type,
                        "content": line,
                        "page": page_num,
                        "extraction_method": "text_extraction",
                        "confidence": 1.0,
                    })

                # Stage 2: table detection
                try:
                    tables = page.find_tables()
                    for tbl in tables:
                        table_data = tbl.extract()
                        if table_data and _is_table_well_extracted(table_data):
                            content_nodes.append({
                                "type": "table",
                                "content": {"headers": table_data[0], "rows": table_data[1:]},
                                "page": page_num,
                                "extraction_method": "text_extraction",
                                "confidence": 0.8,
                            })
                        else:
                            # Bad extraction → vision fallback
                            bbox = tbl.bbox
                            pages_needing_vision.append(
                                _process_table_region(
                                    openai_client, pdf_path, page_num - 1,
                                    bbox, page_num, content_nodes
                                )
                            )
                except Exception:
                    pass  # table detection not critical

                # Stage 2: figure/image detection
                try:
                    for img in page.images:
                        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                        # Only process if the image is sizeable (not decorative icons)
                        width = img["x1"] - img["x0"]
                        height = img["bottom"] - img["top"]
                        if width > 80 and height > 80:
                            pages_needing_vision.append(
                                _process_figure_region(
                                    openai_client, pdf_path, page_num - 1,
                                    bbox, page_num, content_nodes
                                )
                            )
                except Exception:
                    pass

            pages_detail.append({
                "page_number": page_num,
                "content_nodes": content_nodes,
            })

    # Stage 3: run all vision coroutines in parallel (batched to avoid rate limits)
    if pages_needing_vision:
        batch_size = 5
        for i in range(0, len(pages_needing_vision), batch_size):
            await asyncio.gather(*pages_needing_vision[i : i + batch_size])

    sections = _build_sections(pages_detail)

    return {
        "source": doc_name,
        "pages": total_pages,
        "sections": sections,
        "pages_detail": pages_detail,
    }


# ── Vision coroutines (mutate content_nodes list in place) ───────────────────

async def _process_scanned_page(client, pdf_path, page_index, content_nodes):
    page_num = page_index + 1
    try:
        text = await _ocr_page_via_vision(client, pdf_path, page_index)
        content_nodes.append({
            "type": "text",
            "content": text,
            "page": page_num,
            "extraction_method": "vision_ocr",
            "confidence": 0.85,
        })
    except Exception as e:
        content_nodes.append({
            "type": "text",
            "content": f"[Vision OCR failed: {e}]",
            "page": page_num,
            "extraction_method": "vision_ocr",
            "confidence": 0.0,
        })


async def _process_table_region(client, pdf_path, page_index, bbox, page_num, content_nodes):
    try:
        table_json = await _extract_table_via_vision(client, pdf_path, page_index, bbox)
        content_nodes.append({
            "type": "table",
            "content": table_json,
            "page": page_num,
            "extraction_method": "vision_table_parse",
            "confidence": 0.9,
        })
    except Exception as e:
        content_nodes.append({
            "type": "table",
            "content": {"error": str(e)},
            "page": page_num,
            "extraction_method": "vision_table_parse",
            "confidence": 0.0,
        })


async def _process_figure_region(client, pdf_path, page_index, bbox, page_num, content_nodes):
    try:
        description = await _extract_figure_via_vision(client, pdf_path, page_index, bbox)
        content_nodes.append({
            "type": "figure",
            "content": description,
            "page": page_num,
            "extraction_method": "vision_figure",
            "confidence": 0.9,
        })
    except Exception as e:
        pass  # Figure description is nice-to-have; skip on failure
