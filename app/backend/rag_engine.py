import os
from anthropic import Anthropic
from dotenv import load_dotenv
from vector_search import VectorSearch
# from neo4j import GraphDatabase
# from neo4j_graphrag.retrievers import VectorRetriever
# from neo4j_graphrag.llm import OpenAILLM
# from neo4j_graphrag.generation import GraphRAG
# from neo4j_graphrag.embeddings import OpenAIEmbeddings

load_dotenv()

# # driver = GraphDatabase.driver(
# #     os.getenv("NEO4J_URI"),
# #     auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

# embedder = OpenAIEmbeddings(model="text-embedding-3-large")
# # retriever = VectorRetriever(driver, "entity_embeddings", embedder)
# llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})

class RagEngine:
    def __init__(self, chunks_file):
        self.search = VectorSearch(chunks_file)
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # self.driver = GraphDatabase.driver(
        #     os.getenv("NEO4J_URI"),
        #     auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")))

        # self.embedder = OpenAIEmbeddings(model="text-embedding-3-small")
        # self.retriever = VectorRetriever(self.driver, "entity_embeddings", embedder)
        #self.llm = OpenAILLM(model_name="gpt-4o-mini", model_params={"temperature": 0})
        # self.graph_rag = GraphRAG(retriever=self.retriever, llm=self.llm)
        print('RAG Engine initialized')
    
    def query(self, question, top_k=5):
        
        print(f"Processing query: {question}")
        print(f"Retreiving top {top_k} relevant chunks...")

        results = self.search.search(question, top_k=top_k)
        # results1 = self.graph_rag.search(query_text=question, retriever_config={"top_k": 5})

        vector_context = "\n\n".join([
            f"[Source {i+1}: {r['chunk']['source_file']}]\n{r['chunk']['text']}"
            for i, r in enumerate(results)
        ])

        # graph_context = f"""[Graph Knowledge Base]
        # {results1.answer}"""

        context = f"""{vector_context}"""

        prompt = f"""You are a medical research assistant / clinician. Answer the question based ONLY on the provided report papers.
                Context from report papers:
                {context}

                Question: {question}

                Instructions:
                - Answer based only on the provided context
                - Cite sources using [Source 1], [Source 2], etc. for paper references
                - If the context doesn't contain enough information, say so
                - Be precise and include additional information only when required

                Answer:"""


        print("Generating answer...")
        response = self.client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        answer = response.content[0].text

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "rank": i + 1,
                    "file": r['chunk']['source_file'],
                    "relevance": f"{(1 - r['distance']):.2%}",
                    "text_preview": r['chunk']['text'][:200] + "..."
                }
                for i, r in enumerate(results)
            ],
            "num_sources": len(results),
            # "graph_insights": results1.answer
        }

if __name__ == "__main__":
    # Initialize
    rag = RagEngine("app/data/chunks_with_embeddings.json")
    
    # Test queries
    question = "What temperature do heated tobacco products operate at?"

    print("\n" + "="*80)
    result = rag.query(question, top_k=5)
    
    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources used: {result['num_sources']}")
    for source in result['sources']:
        print(f"  [{source['rank']}] {source['file']} (Relevance: {source['relevance']})")

