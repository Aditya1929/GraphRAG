# # recreate_index.py
# from neo4j import GraphDatabase
# import os
# from dotenv import load_dotenv

# load_dotenv()

# driver = GraphDatabase.driver(
#     os.getenv("NEO4J_URI"),
#     auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
# )

# with driver.session() as session:
#     # Drop old index
#     try:
#         session.run("DROP INDEX entity_embeddings IF EXISTS")
#         print("Dropped old index")
#     except:
#         pass
    
#     # Create new index
#     session.run("""
#         CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
#         FOR (e:Entity)
#         ON e.embedding
#         OPTIONS {indexConfig: {
#             `vector.dimensions`: 1536,
#             `vector.similarity_function`: 'cosine'
#         }}
#     """)
#     print("✅ Index created with 1536 dimensions!")

# driver.close()