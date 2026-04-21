"""
Chat engine: question routing + answer synthesis.

Two retrieval paths:
  GraphRAG  — traverses Neo4j knowledge graph. Best for relationship,
               entity-centric, structural, and contradiction questions.
  RLM       — sends relevant document JSON chunks to GPT-4o-mini for
               deep reasoning. Best for interpretation, methodology deep-dives,
               nuanced synthesis.

Routing: GPT-4o-mini classifies the question and decides which path(s) to use.
Both can run in parallel; results are merged into a single cited answer.
"""

import json
from typing import Any

from openai import AsyncOpenAI
import asyncio

from graph_retrieval import graph_search, get_full_graph_context, find_contradictions
from database import get_chat_documents, get_chat_messages


# ── Routing ───────────────────────────────────────────────────────────────────

ROUTING_PROMPT = """Classify this question to determine the best retrieval strategy.

Question: "{question}"

Choose ONE of:
  graph   — relationship questions, entity lookups, cross-document connections, contradictions, structure
  rlm     — deep analysis, methodology explanation, interpretation, nuanced reasoning over specific passages
  hybrid  — requires both graph structure AND deep passage analysis

Respond with ONLY one word: graph, rlm, or hybrid."""

SYNTHESIS_PROMPT = """You are GraphReason, an intelligent multi-document analysis assistant.

Answer the user's question based on the provided context. The context may include:
- Knowledge graph data (structured entity and relationship information)
- Direct document passages (raw text from the source documents)

Instructions:
1. Synthesize a clear, accurate answer using the provided context.
2. Cite sources using [Doc: filename, p.X] format.
3. If context includes graph data, explicitly mention entity relationships.
4. If sources contradict each other, note the contradiction.
5. Be transparent about retrieval method: start with "Based on the knowledge graph..." or
   "After analyzing the relevant sections..." or "Combining graph and passage analysis..."
6. If the context is insufficient, say so clearly.

Question: {question}

Context:
{context}

Answer:"""


async def _classify_question(question: str, client: AsyncOpenAI) -> str:
    """Classify question type for routing. Returns 'graph', 'rlm', or 'hybrid'."""
    # Quick heuristic to avoid an API call for obvious cases
    q_lower = question.lower()
    graph_keywords = [
        "relate", "connection", "link", "contradict", "conflict", "disagree",
        "same as", "reference", "cite", "mention", "cross", "between documents",
        "which document", "how many", "structure", "graph", "entity"
    ]
    rlm_keywords = [
        "explain", "methodology", "how does", "what does it mean",
        "interpret", "analyze", "describe in detail", "elaborate",
        "nuance", "what is the reasoning"
    ]

    graph_score = sum(1 for kw in graph_keywords if kw in q_lower)
    rlm_score = sum(1 for kw in rlm_keywords if kw in q_lower)

    if q_lower.startswith("how are") or "big picture" in q_lower or "overview" in q_lower:
        return "hybrid"
    if graph_score > rlm_score:
        return "graph"
    if rlm_score > graph_score:
        return "rlm"

    # Fall back to LLM classification for ambiguous cases
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": ROUTING_PROMPT.format(question=question)}],
        temperature=0,
        max_tokens=10,
    )
    decision = response.choices[0].message.content.strip().lower()
    if decision not in ("graph", "rlm", "hybrid"):
        return "hybrid"
    return decision


# ── RLM-based direct reasoning ────────────────────────────────────────────────

async def _rlm_answer(
    question: str,
    chat_id: str,
    openai_api_key: str,
    client: AsyncOpenAI,
) -> tuple[str, list[dict]]:
    """
    Retrieve relevant document sections and use GPT-4o-mini to reason over them.
    Returns (context_text, sources_list).
    """
    # Get all documents for this chat
    docs = await get_chat_documents(chat_id)

    # Simple relevance: search for question keywords in document text
    question_words = set(question.lower().split())
    scored_sections = []

    for doc in docs:
        doc_json = doc.json_content or {}
        for section in doc_json.get("sections", []):
            content = section.get("content", "").lower()
            score = sum(1 for w in question_words if w in content and len(w) > 3)
            if score > 0:
                scored_sections.append({
                    "score": score,
                    "source": doc.original_filename,
                    "heading": section.get("heading", ""),
                    "content": section.get("content", "")[:2000],
                })

    # Sort by relevance and take top sections
    scored_sections.sort(key=lambda x: x["score"], reverse=True)
    top_sections = scored_sections[:8]

    if not top_sections:
        # Fall back to first sections of each document
        for doc in docs[:3]:
            doc_json = doc.json_content or {}
            sections = doc_json.get("sections", [])
            if sections:
                top_sections.append({
                    "source": doc.original_filename,
                    "heading": sections[0].get("heading", ""),
                    "content": sections[0].get("content", "")[:1500],
                })

    context_parts = []
    sources = []
    for s in top_sections:
        heading = f" — {s['heading']}" if s.get("heading") else ""
        context_parts.append(f"[{s['source']}{heading}]\n{s['content']}")
        if s["source"] not in [src["file"] for src in sources]:
            sources.append({"file": s["source"], "retrieval": "rlm"})

    context_text = "\n\n".join(context_parts) if context_parts else "No relevant sections found."
    return context_text, sources


# ── Main chat function ────────────────────────────────────────────────────────

async def answer_question(
    question: str,
    chat_id: str,
    openai_api_key: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Route the question, retrieve context, synthesize an answer.
    Returns dict with message, retrieval_method, sources, graph_insights.
    """
    client = AsyncOpenAI(api_key=openai_api_key)

    # Classify the question
    retrieval_method = await _classify_question(question, client)

    graph_context = ""
    rlm_context = ""
    all_sources: list[dict] = []
    graph_insights = None

    # Run retrieval (parallelized for hybrid)
    if retrieval_method == "hybrid":
        # Big-picture and overview questions → full graph context
        q_lower = question.lower()
        if "big picture" in q_lower or "overview" in q_lower or "related" in q_lower:
            graph_data_task = asyncio.create_task(get_full_graph_context(chat_id))
        else:
            graph_data_task = asyncio.create_task(
                graph_search(chat_id, question, openai_api_key)
            )
        rlm_data_task = asyncio.create_task(
            _rlm_answer(question, chat_id, openai_api_key, client)
        )
        graph_raw, (rlm_context, rlm_sources) = await asyncio.gather(
            graph_data_task, rlm_data_task
        )
        if isinstance(graph_raw, str):
            graph_context = graph_raw
        else:
            graph_context = graph_raw.get("context", "")
            all_sources.extend(graph_raw.get("sources", []))
        all_sources.extend(rlm_sources)

    elif retrieval_method == "graph":
        graph_raw = await graph_search(chat_id, question, openai_api_key)
        graph_context = graph_raw.get("context", "")
        all_sources.extend(graph_raw.get("sources", []))

        # Check for contradictions if the question is about conflicts
        q_lower = question.lower()
        if any(w in q_lower for w in ["contradict", "disagree", "conflict", "oppose"]):
            contradictions = await find_contradictions(chat_id)
            if contradictions:
                graph_insights = json.dumps(contradictions[:5], indent=2)
    else:
        # rlm only
        rlm_context, rlm_sources = await _rlm_answer(question, chat_id, openai_api_key, client)
        all_sources.extend(rlm_sources)

    # Build the combined context for synthesis
    context_sections = []
    if graph_context and graph_context != "No relevant graph data found.":
        context_sections.append(f"=== Knowledge Graph Data ===\n{graph_context}")
    if rlm_context and rlm_context != "No relevant sections found.":
        context_sections.append(f"=== Document Passages ===\n{rlm_context}")

    combined_context = "\n\n".join(context_sections) if context_sections else "No relevant context found."

    # Build conversation history for the synthesis call
    messages = []
    if conversation_history:
        for msg in conversation_history[-6:]:  # last 3 exchanges
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": SYNTHESIS_PROMPT.format(question=question, context=combined_context),
    })

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=2000,
    )
    answer = response.choices[0].message.content or "I couldn't generate an answer."

    # Deduplicate sources
    seen = set()
    unique_sources = []
    for s in all_sources:
        key = s.get("file", "")
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {
        "message": answer,
        "retrieval_method": retrieval_method,
        "sources": unique_sources,
        "graph_insights": graph_insights,
    }
