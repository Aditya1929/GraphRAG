"""
Neo4j knowledge graph construction.

Graph schema adapts to document content — the RLM analysis determines what
node types and edge types exist. Every node and edge carries provenance back
to the specific document and section it came from.

Node types (dynamic, based on entity types from RLM):
  :Document  — the source PDF itself
  :Entity    — any extracted entity (person, org, concept, etc.)

Edge types (dynamic, based on relation_types from RLM):
  :CONTAINS           — Document → Entity (entity appears in this document)
  :REFERENCES         — Entity → Entity
  :CONTRADICTS        — Entity → Entity (cross-document)
  :SUPPORTS           — Entity → Entity
  :IS_SAME_AS         — Entity → Entity (cross-document deduplication)
  + any relation_type from the RLM extraction

Isolation: each chat has its own namespace via a `chat_id` property on every
node. All queries filter by `chat_id` so separate chats don't bleed together.
"""

import os
from typing import Any
from neo4j import AsyncGraphDatabase, AsyncDriver
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# AuraDB free doesn't support cluster routing — use direct bolt connection
if NEO4J_URI.startswith("neo4j+s://"):
    NEO4J_URI = NEO4J_URI.replace("neo4j+s://", "bolt+s://", 1)

_driver: AsyncDriver | None = None


def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _driver


async def close_driver():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


# ── Graph construction ────────────────────────────────────────────────────────

async def build_graph(
    chat_id: str,
    per_doc_analyses: list[dict],
    cross_doc_analysis: dict,
):
    """
    Construct the knowledge graph for a chat from RLM analysis output.
    Called once after all documents have been analyzed.
    """
    driver = get_driver()
    async with driver.session() as session:
        # Clear any existing graph for this chat
        await session.run(
            "MATCH (n {chat_id: $chat_id}) DETACH DELETE n",
            chat_id=chat_id,
        )

        # ── Step 1: Create Document nodes ─────────────────────────────────────
        for analysis in per_doc_analyses:
            doc_source = analysis.get("document_source", "unknown")
            await session.run(
                """
                MERGE (d:Document {chat_id: $chat_id, source: $source})
                SET d.summary = $summary,
                    d.page_count = $page_count,
                    d.key_claims = $key_claims
                """,
                chat_id=chat_id,
                source=doc_source,
                summary=analysis.get("summary", ""),
                page_count=0,
                key_claims=analysis.get("key_claims", []),
            )

        # ── Step 2: Create Entity nodes and CONTAINS edges ────────────────────
        for analysis in per_doc_analyses:
            doc_source = analysis.get("document_source", "unknown")
            for entity in analysis.get("entities", []):
                node_id = f"{chat_id}_{doc_source}_{entity['id']}"
                await session.run(
                    """
                    MERGE (e:Entity {node_id: $node_id})
                    SET e.chat_id = $chat_id,
                        e.name = $name,
                        e.type = $type,
                        e.description = $description,
                        e.source_doc = $source_doc,
                        e.source_page = $source_page,
                        e.source_section = $source_section,
                        e.local_id = $local_id
                    WITH e
                    MATCH (d:Document {chat_id: $chat_id, source: $source_doc})
                    MERGE (d)-[:CONTAINS]->(e)
                    """,
                    node_id=node_id,
                    chat_id=chat_id,
                    name=entity.get("name", ""),
                    type=entity.get("type", "Concept"),
                    description=entity.get("description", ""),
                    source_doc=doc_source,
                    source_page=entity.get("source_page"),
                    source_section=entity.get("source_section"),
                    local_id=entity.get("id", ""),
                )

        # ── Step 3: Create within-document relationship edges ─────────────────
        for analysis in per_doc_analyses:
            doc_source = analysis.get("document_source", "unknown")
            for rel in analysis.get("relationships", []):
                src_node_id = f"{chat_id}_{doc_source}_{rel['source_id']}"
                tgt_node_id = f"{chat_id}_{doc_source}_{rel['target_id']}"
                rel_type = rel.get("relation_type", "RELATED_TO").upper().replace(" ", "_")

                # Dynamic relationship type (Cypher requires literals for rel types,
                # so we use APOC's apoc.create.relationship or a parameterized workaround)
                await session.run(
                    f"""
                    MATCH (a:Entity {{node_id: $src_id}})
                    MATCH (b:Entity {{node_id: $tgt_id}})
                    MERGE (a)-[r:`{rel_type}` {{chat_id: $chat_id}}]->(b)
                    SET r.description = $description,
                        r.source_doc = $source_doc,
                        r.source_page = $source_page,
                        r.confidence = $confidence
                    """,
                    src_id=src_node_id,
                    tgt_id=tgt_node_id,
                    chat_id=chat_id,
                    description=rel.get("description", ""),
                    source_doc=doc_source,
                    source_page=rel.get("source_page"),
                    confidence=rel.get("confidence", 1.0),
                )

        # ── Step 4: Cross-document IS_SAME_AS edges (entity deduplication) ────
        for shared in cross_doc_analysis.get("shared_entities", []):
            docs = shared.get("documents", [])
            if len(docs) < 2:
                continue
            # Link the first occurrence to all others with IS_SAME_AS
            first = docs[0]
            first_node_id = f"{chat_id}_{first['doc']}_{first['entity_id']}"
            for other in docs[1:]:
                other_node_id = f"{chat_id}_{other['doc']}_{other['entity_id']}"
                await session.run(
                    """
                    MATCH (a:Entity {node_id: $a_id})
                    MATCH (b:Entity {node_id: $b_id})
                    MERGE (a)-[:IS_SAME_AS {chat_id: $chat_id, canonical: $canonical}]->(b)
                    """,
                    a_id=first_node_id,
                    b_id=other_node_id,
                    chat_id=chat_id,
                    canonical=shared.get("canonical_name", ""),
                )

        # ── Step 5: Cross-document CONTRADICTS / SUPPORTS edges ───────────────
        for contradiction in cross_doc_analysis.get("contradictions", []):
            doc_a_src = contradiction["doc_a"]["source"]
            doc_b_src = contradiction["doc_b"]["source"]
            for eid_a in contradiction["doc_a"].get("entity_ids", []):
                for eid_b in contradiction["doc_b"].get("entity_ids", []):
                    node_a = f"{chat_id}_{doc_a_src}_{eid_a}"
                    node_b = f"{chat_id}_{doc_b_src}_{eid_b}"
                    await session.run(
                        """
                        MATCH (a:Entity {node_id: $a_id})
                        MATCH (b:Entity {node_id: $b_id})
                        MERGE (a)-[r:CONTRADICTS {chat_id: $chat_id}]->(b)
                        SET r.topic = $topic, r.description = $description
                        """,
                        a_id=node_a,
                        b_id=node_b,
                        chat_id=chat_id,
                        topic=contradiction.get("topic", ""),
                        description=contradiction.get("description", ""),
                    )

    print(f"[graph_builder] Graph built for chat {chat_id}")


async def get_graph_summary(chat_id: str) -> dict:
    """Return high-level statistics and top entities for the chat's graph."""
    driver = get_driver()
    async with driver.session() as session:
        entity_result = await session.run(
            "MATCH (e:Entity {chat_id: $chat_id}) RETURN count(e) AS cnt",
            chat_id=chat_id,
        )
        entity_count = (await entity_result.single())["cnt"]

        rel_result = await session.run(
            """
            MATCH (a {chat_id: $chat_id})-[r]->(b {chat_id: $chat_id})
            RETURN count(r) AS cnt
            """,
            chat_id=chat_id,
        )
        rel_count = (await rel_result.single())["cnt"]

        doc_result = await session.run(
            "MATCH (d:Document {chat_id: $chat_id}) RETURN count(d) AS cnt",
            chat_id=chat_id,
        )
        doc_count = (await doc_result.single())["cnt"]

        top_result = await session.run(
            """
            MATCH (e:Entity {chat_id: $chat_id})-[r]->()
            RETURN e.name AS name, e.type AS type, count(r) AS degree
            ORDER BY degree DESC LIMIT 10
            """,
            chat_id=chat_id,
        )
        top_entities = [
            {"name": r["name"], "type": r["type"], "connections": r["degree"]}
            async for r in top_result
        ]

    return {
        "chat_id": chat_id,
        "entity_count": entity_count,
        "relationship_count": rel_count,
        "document_count": doc_count,
        "top_entities": top_entities,
    }
