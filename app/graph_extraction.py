# import json
# import os
# import time
# from neo4j import GraphDatabase
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# driver = GraphDatabase.driver(
#     os.getenv("NEO4J_URI"),
#     auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

# PROMPT = """
# Extract entities and relationships from the text.

# Return JSON ONLY in this format. id has to be of the same form with E then number starting with 1:
# {{
#   "entities": [
#     {{"id": "E1", "name": "X", "type": "Concept"}}
#   ],
#   "relations": [
#     {{"source": "E1", "relation": "relates_to", "target": "E2"}}
#   ]
# }}

# Text:
# \"\"\"{text}\"\"\"
# """

# def extract_graph_data(text):
#     response = client.chat.completions.create( 
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": PROMPT.format(text=text)}],
#         temperature=0,
#     )
    
#     content = response.choices[0].message.content.strip()
#     if content.startswith("```"):
#         lines = content.split('\n')
#         lines = lines[1:] if lines[0].startswith("```") else lines
#         lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
#         content = '\n'.join(lines)
    
#     return json.loads(content)

# def write_graph(tx, graph, chunk):
#     tx.run(
#             """
#             MERGE (c:Chunk{id: $cid})
#             SET c.source = $file
#             """, cid=chunk.get("chunk_id", chunk.get("id")), file=chunk.get("source_file", chunk.get("source"))
#         )
    
#     for e in graph["entities"]:
#         tx.run(
#             """
#             MERGE (n:Entity{id: $id})
#             SET n.name = $name, n.type = $type
#             WITH n
#             MATCH (c:Chunk{id: $cid})
#             MERGE (n)-[:MENTIONED_IN]->(c)
#             """, id=e["id"], name=e["name"], type=e["type"], cid=chunk.get("chunk_id", chunk.get("id"))
#         )
    
#     for r in graph["relations"]:
#         tx.run(
#             """
#             MATCH (a:Entity{id: $source})
#             MATCH (b:Entity{id: $target})
#             MERGE (a)-[:RELATED_TO {type: $relation}]->(b)
#             """, **r
#         )

# def build_graph(chunks_file):
#     with open(chunks_file) as f:
#         chunks = json.load(f)

#     with driver.session() as session:
#         for chunk in chunks:
#             try:
#                 graph_data = extract_graph_data(chunk["text"])
#                 session.execute_write(write_graph, graph_data, chunk)
#                 time.sleep(0.3)
#             except Exception as e:
#                 print(f"Graph extraction failed for {chunk['chunk_id']}: {e}")

# if __name__ == "__main__":
#     build_graph("app/data/chunks.json")