import os
import threading
import traceback
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase

# Import ingestion logic from filo.py
import filo

load_dotenv()

app = Flask(__name__)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "28ea31f4")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "28ea31f4")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

LLM_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

ingest_status = {
    "is_running": False,
    "current_paper": "",
    "log": [],
    "total": 0,
    "completed": 0,
    "failed": []
}


def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def get_genai_client():
    return genai.Client(api_key=GEMINI_API_KEY)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def get_stats():
    driver = get_neo4j_driver()
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            node_counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count"
            ).data()
            rel_count = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS count"
            ).single()["count"]

        counts = {r["label"]: r["count"] for r in node_counts if r["label"]}
        return jsonify({
            "papers": counts.get("Paper", 0),
            "authors": counts.get("Author", 0),
            "methods": counts.get("Method", 0),
            "datasets": counts.get("Dataset", 0),
            "tasks": counts.get("Task", 0),
            "concepts": counts.get("Concept", 0),
            "chunks": counts.get("Chunk", 0),
            "total_relationships": rel_count,
        })
    finally:
        driver.close()


@app.route("/api/graph")
def get_graph():
    """
    Returns graph nodes and edges (excluding text Chunk nodes to keep visualization crisp).
    """
    driver = get_neo4j_driver()
    try:
        query = """
        MATCH (p:Paper)
        OPTIONAL MATCH (p)-[r]->(target)
        WHERE NOT target:Chunk
        RETURN 
            p.paper_id AS paper_id,
            p.title AS paper_title,
            p.year AS paper_year,
            p.arxiv_url AS paper_url,
            type(r) AS rel_type,
            labels(target)[0] AS target_label,
            target.key AS target_key,
            target.name AS target_name
        """
        with driver.session(database=NEO4J_DATABASE) as session:
            records = session.run(query).data()

        nodes_dict = {}
        edges = []
        edge_keys = set()

        for row in records:
            p_id = f"paper_{row['paper_id']}"
            if p_id not in nodes_dict:
                nodes_dict[p_id] = {
                    "id": p_id,
                    "label": row["paper_title"],
                    "group": "Paper",
                    "paper_id": row["paper_id"],
                    "title": row["paper_title"],
                    "year": row["paper_year"],
                    "url": row["paper_url"],
                    "node_type": "Paper"
                }

            if row.get("target_key") and row.get("target_label"):
                t_label = row["target_label"]
                t_key = row["target_key"]
                t_name = row.get("target_name") or t_key
                t_id = f"{t_label.lower()}_{t_key}"

                if t_id not in nodes_dict:
                    nodes_dict[t_id] = {
                        "id": t_id,
                        "label": t_name,
                        "group": t_label,
                        "name": t_name,
                        "key": t_key,
                        "node_type": t_label
                    }

                edge_key = f"{p_id}->{row['rel_type']}->{t_id}"
                if edge_key not in edge_keys:
                    edge_keys.add(edge_key)
                    edges.append({
                        "id": edge_key,
                        "from": p_id,
                        "to": t_id,
                        "label": row["rel_type"],
                        "title": row["rel_type"]
                    })

        return jsonify({
            "nodes": list(nodes_dict.values()),
            "edges": edges
        })
    finally:
        driver.close()


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    question = data.get("query", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    client = get_genai_client()
    driver = get_neo4j_driver()

    try:
        # 1. Embed query
        embed_resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[question]
        )
        query_vector = embed_resp.embeddings[0].values

        # 2. Hybrid Graph Retrieval: vector search chunks + graph context
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embedding', 4, $vector)
        YIELD node AS chunk, score
        MATCH (p:Paper)-[:HAS_CHUNK]->(chunk)
        OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
        OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
        OPTIONAL MATCH (p)-[:USES_DATASET]->(d:Dataset)
        OPTIONAL MATCH (p)-[:ADDRESSES]->(t:Task)
        OPTIONAL MATCH (p)-[:MENTIONS]->(c:Concept)
        RETURN 
            p.paper_id AS paper_id,
            p.title AS paper_title,
            p.year AS year,
            chunk.section AS section,
            chunk.text AS text,
            collect(DISTINCT a.name) AS authors,
            collect(DISTINCT m.name) AS methods,
            collect(DISTINCT d.name) AS datasets,
            collect(DISTINCT t.name) AS tasks,
            collect(DISTINCT c.name) AS concepts,
            score
        ORDER BY score DESC
        """

        with driver.session(database=NEO4J_DATABASE) as session:
            records = session.run(cypher, vector=query_vector).data()

            # Cross-paper connections query
            interconnects = session.run("""
                MATCH (p1:Paper)-[r1]->(shared)<-[r2]-(p2:Paper)
                WHERE p1.paper_id < p2.paper_id AND NOT shared:Chunk
                RETURN 
                    p1.title AS paper1,
                    p2.title AS paper2,
                    labels(shared)[0] AS entity_type,
                    shared.name AS shared_name,
                    shared.key AS shared_key,
                    p1.paper_id AS p1_id,
                    p2.paper_id AS p2_id
                LIMIT 30
            """).data()

        highlight_nodes = set()
        context_blocks = []

        for r in records:
            p_node_id = f"paper_{r['paper_id']}"
            highlight_nodes.add(p_node_id)
            context_blocks.append(
                f"[Source: '{r['paper_title']}' ({r['year']}) | Section: {r['section']}]\n"
                f"Authors: {', '.join(r['authors']) or 'N/A'}\n"
                f"Methods: {', '.join(r['methods']) or 'N/A'}\n"
                f"Datasets: {', '.join(r['datasets']) or 'N/A'}\n"
                f"Tasks: {', '.join(r['tasks']) or 'N/A'}\n"
                f"Text Excerpt:\n{r['text']}\n"
            )

        connections_text = []
        for ic in interconnects:
            connections_text.append(
                f"• '{ic['paper1']}' and '{ic['paper2']}' are connected via {ic['entity_type']}: '{ic['shared_name']}'"
            )
            highlight_nodes.add(f"paper_{ic['p1_id']}")
            highlight_nodes.add(f"paper_{ic['p2_id']}")
            highlight_nodes.add(f"{ic['entity_type'].lower()}_{ic['shared_key']}")

        prompt = f"""You are a precise research assistant with access to a research paper Knowledge Graph and vector database.
Use the provided graph connections and paper excerpts to answer the user's question clearly, objectively, and accurately.
Cite the relevant papers, methods, datasets, or shared connections directly.

KNOWN GRAPH CONNECTIONS BETWEEN PAPERS:
{chr(10).join(connections_text) if connections_text else "No shared entities found."}

TEXT CHUNKS & METADATA FROM PAPERS:
{chr(10).join(context_blocks)}

USER QUESTION:
{question}

Provide a direct, informative answer. Avoid pleasantries."""

        resp = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
        )

        return jsonify({
            "answer": resp.text,
            "highlight_nodes": list(highlight_nodes)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        driver.close()


def run_batch_ingest(paper_ids):
    global ingest_status
    ingest_status["is_running"] = True
    ingest_status["total"] = len(paper_ids)
    ingest_status["completed"] = 0
    ingest_status["failed"] = []
    ingest_status["log"].append(f"Starting ingestion of {len(paper_ids)} papers...")

    for pid in paper_ids:
        pid = pid.strip()
        if not pid:
            continue
        ingest_status["current_paper"] = pid
        ingest_status["log"].append(f"Processing paper: {pid}")
        try:
            filo.process_paper(pid)
            ingest_status["completed"] += 1
            ingest_status["log"].append(f"✓ Completed {pid}")
        except Exception as e:
            ingest_status["failed"].append({"paper_id": pid, "error": str(e)})
            ingest_status["log"].append(f"✗ Failed {pid}: {e}")

    ingest_status["is_running"] = False
    ingest_status["current_paper"] = ""
    ingest_status["log"].append("Batch ingestion finished.")


@app.route("/api/ingest", methods=["POST"])
def ingest():
    data = request.json or {}
    papers = data.get("paper_ids", [])
    if isinstance(papers, str):
        papers = [p.strip() for p in papers.split() if p.strip()]

    if not papers:
        return jsonify({"error": "No paper IDs provided"}), 400

    if ingest_status["is_running"]:
        return jsonify({"error": "An ingestion job is already running."}), 409

    t = threading.Thread(target=run_batch_ingest, args=(papers,), daemon=True)
    t.start()

    return jsonify({"message": f"Started ingestion for {len(papers)} papers.", "count": len(papers)})


@app.route("/api/ingest/status")
def get_ingest_status():
    return jsonify(ingest_status)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
