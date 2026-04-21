"""
RLM-based document analysis using the `rlm` library (pip install rlm).

Two passes:
  1. Per-document: entities, relationships, key claims, cross-doc signals.
  2. Cross-document: shared entities, contradictions, complementary info.

The RLM library's REPL scaffolding lets the model interact with the document
JSON programmatically — iterating through sections, delegating sub-tasks to
fresh model instances — rather than dumping everything into one context window.

Model: gpt-4o-mini for all calls (development budget).
"""

import json
import re
import asyncio
from typing import Any

from openai import AsyncOpenAI


# ── System prompts ────────────────────────────────────────────────────────────

ENTITY_EXTRACTION_PROMPT = """You are a knowledge graph construction expert analyzing a document.

Extract the following from the provided document JSON:
1. A 2-3 sentence summary of the document's purpose and main findings.
2. Up to 10 key claims or assertions made in the document.
3. All significant entities (people, organizations, concepts, dates, metrics, policies, events, locations).
4. Relationships between entities WITH provenance (which page/section they appear on).
5. Cross-document signals: topics or concepts that might connect to other documents in a corpus.

CRITICAL: Return ONLY valid JSON. No prose, no markdown fences, no explanation.

Required format:
{
  "summary": "...",
  "key_claims": ["claim1", "claim2"],
  "entities": [
    {
      "id": "E1",
      "name": "exact name as it appears in text",
      "type": "Person|Organization|Concept|Claim|Date|Metric|Policy|Event|Location",
      "description": "one-line description",
      "source_page": 1,
      "source_section": "section heading or null"
    }
  ],
  "relationships": [
    {
      "source_id": "E1",
      "target_id": "E2",
      "relation_type": "references|contradicts|supports|authored|mentions|is_part_of|caused_by|measures|occurred_before|occurred_after",
      "description": "brief relationship description",
      "source_page": 1,
      "confidence": 0.95
    }
  ],
  "cross_doc_signals": [
    "This document discusses X methodology, which may appear under different names in related documents."
  ]
}"""

CROSS_DOC_PROMPT = """You are analyzing a set of per-document knowledge extractions to identify cross-document connections.

Given the list of document analyses, identify:
1. Shared entities (same real-world entity appearing across multiple documents, possibly with different names or IDs).
2. Contradictions (document A claims X, document B claims the opposite or incompatible thing).
3. Complementary connections (document A provides context/background that explains or extends a claim in document B).
4. Temporal relationships (chronological ordering of events, publications, or findings across documents).

CRITICAL: Return ONLY valid JSON. No prose, no markdown fences.

Required format:
{
  "shared_entities": [
    {
      "canonical_name": "unified entity name",
      "type": "entity type",
      "documents": [
        {"doc": "source_filename", "entity_id": "E1", "local_name": "name in this doc"}
      ]
    }
  ],
  "contradictions": [
    {
      "topic": "what the contradiction concerns",
      "doc_a": {"source": "filename", "claim": "what doc A says", "entity_ids": ["E1"]},
      "doc_b": {"source": "filename", "claim": "what doc B says", "entity_ids": ["E2"]},
      "description": "nature and significance of the contradiction"
    }
  ],
  "complementary_connections": [
    {
      "topic": "shared topic or concept",
      "docs": [{"source": "filename", "contribution": "what this document contributes"}],
      "description": "how these documents complement each other"
    }
  ],
  "temporal_relationships": [
    {
      "doc_a": "filename",
      "doc_b": "filename",
      "relation": "before|after|concurrent|references",
      "description": "temporal relationship description"
    }
  ]
}"""


# ── RLM wrapper ───────────────────────────────────────────────────────────────

def _try_import_rlm():
    """Try to import the RLM library. Return None if unavailable."""
    try:
        from rlm import RLM
        return RLM
    except ImportError:
        return None


def _parse_json_response(raw: str) -> dict:
    """Extract and parse JSON from a model response, stripping markdown fences."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    # Find the outermost JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)


async def _openai_fallback(client: AsyncOpenAI, system_prompt: str, content: str) -> dict:
    """
    Direct OpenAI call as a fallback when the RLM library is unavailable or
    the document is small enough that RLM's overhead isn't needed.
    """
    # Truncate content to stay within token budget
    MAX_CHARS = 80_000  # ~20k tokens at 4 chars/token
    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS] + "\n...[truncated]"

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document JSON:\n{content}"},
        ],
        temperature=0,
        max_tokens=4000,
    )
    raw = response.choices[0].message.content or "{}"
    return _parse_json_response(raw)


async def analyze_document(doc_json: dict, openai_api_key: str, source_filename: str) -> dict:
    """
    Per-document analysis: extract entities, relationships, claims.
    Uses RLM library if available; falls back to direct OpenAI call.
    """
    client = AsyncOpenAI(api_key=openai_api_key)
    content_str = json.dumps(doc_json, ensure_ascii=False)

    RLM = _try_import_rlm()

    if RLM is not None:
        try:
            # RLM handles chunking and recursive sub-LLM delegation automatically.
            # API: rlm.completion(prompt) -> RLMChatCompletion with .response field
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "gpt-4o-mini", "api_key": openai_api_key},
            )
            combined_prompt = (
                f"{ENTITY_EXTRACTION_PROMPT}\n\nDocument JSON:\n{content_str[:60000]}"
            )
            # run_in_executor to avoid blocking the async event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: rlm.completion(prompt=combined_prompt),
            )
            analysis = _parse_json_response(result.response)
        except Exception:
            # RLM call failed; fall back to direct OpenAI
            analysis = await _openai_fallback(client, ENTITY_EXTRACTION_PROMPT, content_str)
    else:
        analysis = await _openai_fallback(client, ENTITY_EXTRACTION_PROMPT, content_str)

    # Attach source document to every entity and relationship
    for entity in analysis.get("entities", []):
        entity["source_doc"] = source_filename
    for rel in analysis.get("relationships", []):
        rel["source_doc"] = source_filename

    analysis["document_source"] = source_filename
    return analysis


async def analyze_cross_document(
    per_doc_analyses: list[dict], openai_api_key: str
) -> dict:
    """
    Cross-document pass: find shared entities, contradictions, complements.
    Runs after all per-document analyses are complete.
    """
    client = AsyncOpenAI(api_key=openai_api_key)

    # Build a compact summary of all analyses to fit within context window.
    # Each analysis is reduced to its entities and key claims.
    compact = []
    for analysis in per_doc_analyses:
        compact.append({
            "document_source": analysis.get("document_source", "unknown"),
            "summary": analysis.get("summary", ""),
            "key_claims": analysis.get("key_claims", [])[:5],
            "entities": [
                {"id": e["id"], "name": e["name"], "type": e["type"]}
                for e in analysis.get("entities", [])[:30]
            ],
            "cross_doc_signals": analysis.get("cross_doc_signals", []),
        })

    compact_str = json.dumps(compact, ensure_ascii=False)
    RLM = _try_import_rlm()

    if RLM is not None:
        try:
            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "gpt-4o-mini", "api_key": openai_api_key},
            )
            combined_prompt = (
                f"{CROSS_DOC_PROMPT}\n\nDocument analyses:\n{compact_str[:60000]}"
            )
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: rlm.completion(prompt=combined_prompt),
            )
            cross_analysis = _parse_json_response(result.response)
        except Exception:
            cross_analysis = await _openai_fallback(client, CROSS_DOC_PROMPT, compact_str)
    else:
        cross_analysis = await _openai_fallback(client, CROSS_DOC_PROMPT, compact_str)

    return cross_analysis
