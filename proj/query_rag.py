import os
import sys
from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"


def query_graph_rag(user_question: str, top_k: int = 5):
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        raise SystemExit("Missing Neo4j credentials in .env.")
    if not GEMINI_API_KEY:
        raise SystemExit("Missing GEMINI_API_KEY in .env.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # 1. Embed user question
    print(f"\n[1/3] Embedding question with {EMBED_MODEL}...")
    embed_resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[user_question],
    )
    query_vector = embed_resp.embeddings[0].values

    # 2. Vector search + Graph traversal
    print(f"[2/3] Searching Neo4j vector index & traversing knowledge graph...")
    cypher_query = """
    CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $vector)
    YIELD node AS chunk, score
    MATCH (p:Paper)-[:HAS_CHUNK]->(chunk)
    OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
    OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
    OPTIONAL MATCH (p)-[:USES_DATASET]->(d:Dataset)
    OPTIONAL MATCH (p)-[:ADDRESSES]->(t:Task)
    RETURN 
        p.title AS paper_title,
        p.year AS year,
        p.arxiv_url AS arxiv_url,
        chunk.section AS section,
        chunk.text AS chunk_text,
        collect(DISTINCT a.name) AS authors,
        collect(DISTINCT m.name) AS methods,
        collect(DISTINCT d.name) AS datasets,
        collect(DISTINCT t.name) AS tasks,
        score
    ORDER BY score DESC
    """

    with driver.session(database=NEO4J_DATABASE) as session:
        records = session.run(cypher_query, top_k=top_k, vector=query_vector).data()

    driver.close()

    if not records:
        print("No matching records found in Neo4j. Have you ingested any papers yet?")
        return

    # Build context for LLM
    context_blocks = []
    print(f"  Found {len(records)} relevant chunks from graph:")
    for idx, r in enumerate(records, 1):
        print(f"    {idx}. '{r['paper_title']}' (Score: {r['score']:.4f})")
        context_blocks.append(
            f"--- Document Source {idx} ---\n"
            f"Paper: {r['paper_title']} ({r['year']})\n"
            f"URL: {r['arxiv_url']}\n"
            f"Section: {r['section']}\n"
            f"Authors: {', '.join(r['authors']) or 'N/A'}\n"
            f"Methods: {', '.join(r['methods']) or 'N/A'}\n"
            f"Datasets: {', '.join(r['datasets']) or 'N/A'}\n"
            f"Tasks: {', '.join(r['tasks']) or 'N/A'}\n"
            f"Excerpt:\n{r['chunk_text']}"
        )

    full_context = "\n\n".join(context_blocks)

    # 3. Generate answer
    print(f"[3/3] Generating answer with {LLM_MODEL}...")
    prompt = f"""You are a research assistant answering questions using a research paper Knowledge Graph and vector database.

Use the provided graph metadata and text excerpts to answer the question accurately. If the context does not contain enough info, state what is missing.

Context from Knowledge Graph:
{full_context}

Question:
{user_question}

Answer:"""

    resp = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(resp.text)
    print("=" * 60)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip() or input("\nEnter your question: ").strip()
    if question:
        query_graph_rag(question)
