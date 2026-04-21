"""
Neo4j GraphRAG retrieval layer.

Translates natural-language questions into Cypher queries and returns
structured context suitable for answer synthesis.

GraphRAG is best for:
  - Relationship questions ("how does X relate to Y")
  - Multi-hop reasoning ("who authored the paper that contradicts X")
  - Entity-centric queries ("tell me everything about Company X")
  - Structural questions ("which documents reference each other")
  - Contradiction detection ("does any source contradict claim X")
"""

import os
import json
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver
from openai import AsyncOpenAI
from dotenv import load_dotenv
from graph_builder import get_driver

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


# ── Cypher generation ─────────────────────────────────────────────────────────

CYPHER_GEN_PROMPT = """You are a Neo4j Cypher expert. Given a user question and the graph schema,
write a Cypher query to retrieve the relevant information.

Graph schema:
  Nodes: :Document (source, summary, key_claims), :Entity (name, type, description, source_doc, source_page)
  Relationships: :CONTAINS (Document→Entity), :IS_SAME_AS, :CONTRADICTS, :REFERENCES, :SUPPORTS,
                 and dynamic relation types from entity analysis.
  All nodes have chat_id property for isolation.

Rules:
1. ALWAYS filter by chat_id: $chat_id
2. Use OPTIONAL MATCH for optional traversals
3. Limit results: LIMIT 20 unless the question requires more
4. Return meaningful fields (name, description, source_doc, source_page, type)
5. For contradiction queries, match CONTRADICTS relationships
6. For entity lookups, use case-insensitive CONTAINS: toLower(e.name) CONTAINS toLower($term)
7. Return ONLY the Cypher query, no explanation, no markdown fences.

User question: {question}

Cypher query:"""


async def _generate_cypher(question: str, openai_api_key: str) -> str:
    """Use GPT-4o-mini to translate a question into a Cypher query."""
    client = AsyncOpenAI(api_key=openai_api_key)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": CYPHER_GEN_PROMPT.format(question=question),
            }
        ],
        temperature=0,
        max_tokens=500,
    )
    raw = response.choices[0].message.content or ""
    # Strip markdown fences if present
    raw = raw.strip().lstrip("```cypher").lstrip("```").rstrip("```").strip()
    return raw


async def graph_search(chat_id: str, question: str, openai_api_key: str) -> dict:
    """
    Full GraphRAG retrieval:
    1. Generate Cypher from the question
    2. Execute against Neo4j
    3. Return structured results + a text summary for synthesis
    """
    driver = get_driver()

    # Generate Cypher
    try:
        cypher = await _generate_cypher(question, openai_api_key)
    except Exception as e:
        return {"error": f"Cypher generation failed: {e}", "results": [], "context": ""}

    # Execute Cypher
    records = []
    try:
        async with driver.session() as session:
            result = await session.run(cypher, chat_id=chat_id)
            records = [dict(r) async for r in result]
    except Exception as e:
        # If the generated Cypher is invalid, fall back to a basic entity search
        try:
            async with driver.session() as session:
                fallback_result = await session.run(
                    """
                    MATCH (e:Entity {chat_id: $chat_id})
                    WHERE toLower(e.name) CONTAINS toLower($term)
                       OR toLower(e.description) CONTAINS toLower($term)
                    OPTIONAL MATCH (e)-[r]-(related:Entity {chat_id: $chat_id})
                    RETURN e.name AS name, e.type AS type, e.description AS description,
                           e.source_doc AS source_doc, e.source_page AS source_page,
                           collect(related.name)[..5] AS related_entities
                    LIMIT 15
                    """,
                    chat_id=chat_id,
                    term=question[:50],
                )
                records = [dict(r) async for r in fallback_result]
        except Exception:
            records = []

    # Format results as readable context
    context_lines = []
    sources = []

    for record in records:
        line_parts = []
        for key, value in record.items():
            if value is not None and value != [] and value != "":
                line_parts.append(f"{key}: {value}")
        if line_parts:
            context_lines.append(" | ".join(line_parts))

        # Track unique sources
        source_doc = record.get("source_doc") or record.get("e.source_doc")
        if source_doc and source_doc not in [s["file"] for s in sources]:
            sources.append({
                "file": source_doc,
                "page": record.get("source_page") or record.get("e.source_page"),
                "retrieval": "graph",
            })

    context_text = "\n".join(context_lines) if context_lines else "No relevant graph data found."

    return {
        "cypher_used": cypher,
        "records": records,
        "context": context_text,
        "sources": sources,
        "result_count": len(records),
    }


async def get_full_graph_context(chat_id: str) -> str:
    """
    Return a high-level narrative of the graph (all documents and their connections).
    Used for 'big picture' questions.
    """
    driver = get_driver()
    async with driver.session() as session:
        # Get all documents and their top entities
        doc_result = await session.run(
            """
            MATCH (d:Document {chat_id: $chat_id})
            OPTIONAL MATCH (d)-[:CONTAINS]->(e:Entity {chat_id: $chat_id})
            RETURN d.source AS doc, d.summary AS summary,
                   collect(e.name + ' (' + e.type + ')')[..10] AS entities
            """,
            chat_id=chat_id,
        )
        docs = [dict(r) async for r in doc_result]

        # Get cross-document connections
        cross_result = await session.run(
            """
            MATCH (a:Entity {chat_id: $chat_id})-[r:CONTRADICTS|IS_SAME_AS]->(b:Entity {chat_id: $chat_id})
            WHERE a.source_doc <> b.source_doc
            RETURN a.name AS entity_a, a.source_doc AS doc_a,
                   type(r) AS relationship, b.name AS entity_b, b.source_doc AS doc_b
            LIMIT 20
            """,
            chat_id=chat_id,
        )
        cross_connections = [dict(r) async for r in cross_result]

    lines = ["=== Document Overview ==="]
    for doc in docs:
        lines.append(f"\n[{doc['doc']}]\nSummary: {doc['summary']}")
        if doc["entities"]:
            lines.append(f"Key entities: {', '.join(doc['entities'])}")

    if cross_connections:
        lines.append("\n=== Cross-Document Connections ===")
        for conn in cross_connections:
            lines.append(
                f"{conn['entity_a']} ({conn['doc_a']}) "
                f"--{conn['relationship']}--> "
                f"{conn['entity_b']} ({conn['doc_b']})"
            )

    return "\n".join(lines)


async def find_contradictions(chat_id: str) -> list[dict]:
    """Return all contradiction relationships in the graph."""
    driver = get_driver()
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Entity {chat_id: $chat_id})-[r:CONTRADICTS]->(b:Entity {chat_id: $chat_id})
            RETURN a.name AS entity_a, a.source_doc AS doc_a, a.source_page AS page_a,
                   b.name AS entity_b, b.source_doc AS doc_b, b.source_page AS page_b,
                   r.topic AS topic, r.description AS description
            """,
            chat_id=chat_id,
        )
        return [dict(r) async for r in result]
