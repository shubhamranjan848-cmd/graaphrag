"""
End-to-End arXiv Knowledge Graph & Vector RAG Ingestion Pipeline
"""

import json
import os
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import certifi
from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
import pymupdf4llm

# Load .env variables
load_dotenv()

# --- Configurations ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PAPERS_DIR = Path("papers")
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

LLM_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 1000
OVERLAP = 200
BATCH_SIZE = 20

SSL_CTX = ssl.create_default_context(cafile=certifi.where())
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

CATEGORY_NAMES = {
    "cs.CL": "Natural Language Processing",
    "cs.LG": "Machine Learning",
    "cs.CV": "Computer Vision",
    "cs.AI": "Artificial Intelligence",
    "cs.IR": "Information Retrieval",
    "cs.DC": "Distributed Computing",
    "cs.AR": "Hardware Architecture",
    "stat.ML": "Machine Learning (Statistics)",
    "cs.NE": "Neural and Evolutionary Computing",
}


def get_genai_client():
    if not GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY in environment or .env file.")
    return genai.Client(api_key=GEMINI_API_KEY)


# ==============================================================================
# 1. TEXT HELPERS & NORMALIZATION
# ==============================================================================

def normalize(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[-_]+", " ", name)
    return re.sub(r"\s+", " ", name)


def as_entities(items):
    """Normalize list of strings or {'key': ..., 'name': ...} into unique entity dicts."""
    output = []
    seen = set()
    for item in items or []:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            key = str(item.get("key") or normalize(name)).strip()
        else:
            name = str(item).strip()
            key = normalize(name)

        if name and key and key not in seen:
            seen.add(key)
            output.append({"key": key, "name": name})
    return output


# ==============================================================================
# 2. ARXIV SEARCH, METADATA & PDF RETRIEVAL
# ==============================================================================

def fetch_arxiv_metadata(paper_id: str = None, title: str = None):
    if paper_id:
        params = {"id_list": paper_id}
    else:
        params = {"search_query": f'ti:"{title}"', "max_results": 1}

    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, context=SSL_CTX) as r:
        root = ET.fromstring(r.read())

    entry = root.find("atom:entry", NS)
    if entry is None:
        return None

    raw_id = entry.find("atom:id", NS).text.strip().split("/abs/")[-1]
    clean_id = re.sub(r"v\d+$", "", raw_id)
    cats = [c.attrib["term"] for c in entry.findall("atom:category", NS)]
    primary = entry.find("arxiv:primary_category", NS)

    return {
        "paper_id": clean_id,
        "title": " ".join(entry.find("atom:title", NS).text.split()),
        "authors": [a.find("atom:name", NS).text for a in entry.findall("atom:author", NS)],
        "published": entry.find("atom:published", NS).text[:10],
        "year": int(entry.find("atom:published", NS).text[:4]),
        "abstract": " ".join(entry.find("atom:summary", NS).text.split()),
        "primary_category": primary.attrib["term"] if primary is not None else None,
        "categories": cats,
        "topics": [CATEGORY_NAMES.get(c, c) for c in cats],
        "pdf_url": f"https://arxiv.org/pdf/{clean_id}.pdf",
    }


def download_pdf(paper_meta: dict, folder: Path = PAPERS_DIR) -> Path:
    safe_id = paper_meta["paper_id"].replace("/", "_")
    path = folder / f"{safe_id}.pdf"
    if path.exists():
        print(f"  PDF already exists: {path}")
        return path

    req = urllib.request.Request(
        paper_meta["pdf_url"], headers={"User-Agent": "ResearchRAG/1.0 (academic-pipeline)"}
    )
    with urllib.request.urlopen(req, context=SSL_CTX) as web, open(path, "wb") as f:
        f.write(web.read())
    print(f"  Downloaded: {path}")
    return path


# ==============================================================================
# 3. LLM ENTITY EXTRACTION
# ==============================================================================

class PaperEntities(BaseModel):
    keywords: list[str] = Field(description="5-8 key concepts named in the text")
    methods: list[str] = Field(description="algorithms/architectures/techniques, max 8")
    datasets: list[str] = Field(description="datasets or benchmarks named, max 8")
    tasks: list[str] = Field(description="problems the paper addresses, max 4")


def extract_entities(client: genai.Client, meta: dict) -> dict:
    prompt = f"""Extract information from this research paper abstract.
Rules:
- Only include items explicitly named in the text. Do not hallucinate or guess.
- Use the paper's own wording and standard names (e.g., 'Transformer', 'BLEU').
- Return empty lists if nothing applies.

Title: {meta['title']}
Abstract: {meta['abstract']}
"""
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": PaperEntities,
                    "temperature": 0.0,
                },
            )
            ents = resp.parsed.model_dump()
            return {
                **meta,
                "authors_norm": as_entities(meta["authors"]),
                "keywords": as_entities(ents["keywords"]),
                "methods": as_entities(ents["methods"]),
                "datasets": as_entities(ents["datasets"]),
                "tasks": as_entities(ents["tasks"]),
            }
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  Gemini API error ({e}); retrying in {wait}s ({attempt+1}/5)...")
            time.sleep(wait)
    raise RuntimeError("Gemini API failed after 5 retries.")


# ==============================================================================
# 4. PDF PARSING, CHUNKING & EMBEDDINGS
# ==============================================================================

REF_PATTERN = re.compile(
    r"^\s*(#{1,6}\s*)?\**\s*(\d+\.?\s*)?(references|bibliography)\s*\**\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_markdown(md: str) -> str:
    matches = list(REF_PATTERN.finditer(md))
    if matches:
        md = md[: matches[-1].start()]
    md = re.sub(r"-{3,}\s*PAGE\s*\d+\s*-{3,}", "", md)
    md = re.sub(r"^\s*\d{1,3}\s*$", "", md, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def split_sections(md: str):
    sections, current_title, buf = [], "Front matter", []
    for line in md.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                sections.append((current_title, "\n".join(buf).strip()))
            current_title = m.group(1).strip("* ").strip() or "Untitled"
            buf = []
        else:
            buf.append(line)
    if "".join(buf).strip():
        sections.append((current_title, "\n".join(buf).strip()))
    return sections or [("Full text", md)]


def chunk_sections(sections, paper_id: str):
    chunks, idx = [], 0
    for title, text in sections:
        start = 0
        while start < len(text):
            piece = text[start : start + CHUNK_SIZE].strip()
            if len(piece) > 50:
                chunks.append({
                    "chunk_id": f"{paper_id}_{idx}",
                    "paper_id": paper_id,
                    "section": title,
                    "index": idx,
                    "text": piece,
                })
                idx += 1
            start += CHUNK_SIZE - OVERLAP
    return chunks


def embed_chunks(client: genai.Client, chunks: list[dict]):
    texts = [c["text"] for c in chunks]
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        for attempt in range(3):
            try:
                resp = client.models.embed_content(model=EMBED_MODEL, contents=batch)
                vectors.extend([e.values for e in resp.embeddings])
                break
            except Exception as e:
                print(f"  Batch {i // BATCH_SIZE} failed ({e}); retrying {attempt + 1}/3...")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Embedding generation failed for batch starting at index {i}")

    for c, v in zip(chunks, vectors):
        c["embedding"] = [float(val) for val in v]
    return chunks


# ==============================================================================
# 5. NEO4J GRAPH & VECTOR STORAGE
# ==============================================================================

def create_constraints(driver, database: str):
    queries = [
        "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT author_key_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.key IS UNIQUE",
        "CREATE CONSTRAINT concept_key_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.key IS UNIQUE",
        "CREATE CONSTRAINT method_key_unique IF NOT EXISTS FOR (m:Method) REQUIRE m.key IS UNIQUE",
        "CREATE CONSTRAINT dataset_key_unique IF NOT EXISTS FOR (d:Dataset) REQUIRE d.key IS UNIQUE",
        "CREATE CONSTRAINT task_key_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.key IS UNIQUE",
        "CREATE CONSTRAINT topic_key_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.key IS UNIQUE",
    ]
    with driver.session(database=database) as session:
        for q in queries:
            session.run(q).consume()


def write_paper_transaction(tx, paper_meta: dict, chunks: list[dict]):
    paper_id = paper_meta["paper_id"]

    # 1. Paper node
    tx.run(
        """
        MERGE (p:Paper {paper_id: $paper_id})
        SET p.title = $title,
            p.abstract = $abstract,
            p.published = $published,
            p.year = $year,
            p.arxiv_url = $arxiv_url,
            p.pdf_url = $pdf_url,
            p.primary_category = $primary_category,
            p.categories = $categories,
            p.topics = $topics
        """,
        paper_id=paper_id,
        title=paper_meta.get("title", ""),
        abstract=paper_meta.get("abstract", ""),
        published=paper_meta.get("published"),
        year=paper_meta.get("year"),
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=paper_meta.get("pdf_url"),
        primary_category=paper_meta.get("primary_category"),
        categories=paper_meta.get("categories", []),
        topics=paper_meta.get("topics", []),
    ).consume()

    # 2. Authors
    tx.run(
        """
        MATCH (p:Paper {paper_id: $paper_id})
        UNWIND $items AS item
        MERGE (a:Author {key: item.key})
        SET a.name = item.name
        MERGE (p)-[:AUTHORED_BY]->(a)
        """,
        paper_id=paper_id,
        items=paper_meta.get("authors_norm", []),
    ).consume()

    # 3. Entity Relationships
    entity_specs = [
        ("keywords", "Concept", "MENTIONS"),
        ("methods", "Method", "USES_METHOD"),
        ("datasets", "Dataset", "USES_DATASET"),
        ("tasks", "Task", "ADDRESSES"),
        ("topics", "Topic", "IN_TOPIC"),
    ]
    for field, label, rel in entity_specs:
        items = as_entities(paper_meta.get(field, []))
        if items:
            tx.run(
                f"""
                MATCH (p:Paper {{paper_id: $paper_id}})
                UNWIND $items AS item
                MERGE (e:{label} {{key: item.key}})
                SET e.name = item.name
                MERGE (p)-[:{rel}]->(e)
                """,
                paper_id=paper_id,
                items=items,
            ).consume()

    # 4. Chunks with Embeddings
    tx.run(
        """
        MATCH (p:Paper {paper_id: $paper_id})
        UNWIND $chunks AS item
        MERGE (c:Chunk {chunk_id: item.chunk_id})
        SET c.paper_id = $paper_id,
            c.text = item.text,
            c.section = item.section,
            c.chunk_index = item.index,
            c.embedding = item.embedding
        MERGE (p)-[:HAS_CHUNK]->(c)
        """,
        paper_id=paper_id,
        chunks=chunks,
    ).consume()


def create_vector_index(driver, database: str, dimensions: int):
    query = f"""
    CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
    FOR (c:Chunk) ON c.embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dimensions},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    with driver.session(database=database) as session:
        session.run(query).consume()


# ==============================================================================
# 6. MAIN ORCHESTRATOR
# ==============================================================================

def process_paper(query_or_id: str):
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        sys.exit("Error: Neo4j credentials missing in environment / .env file.")

    client = get_genai_client()

    print(f"\n[1/5] Fetching arXiv metadata for '{query_or_id}'...")
    is_id = bool(re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", query_or_id))
    raw_meta = fetch_arxiv_metadata(paper_id=query_or_id) if is_id else fetch_arxiv_metadata(title=query_or_id)

    if not raw_meta:
        print(f"No paper found for query: {query_or_id}")
        return

    print(f"  Title: {raw_meta['title']}")
    print(f"  arXiv ID: {raw_meta['paper_id']}")

    print("\n[2/5] Extracting entities with Gemini...")
    meta = extract_entities(client, raw_meta)

    print("\n[3/5] Downloading and parsing PDF...")
    pdf_path = download_pdf(meta)
    raw_md = pymupdf4llm.to_markdown(str(pdf_path))
    cleaned_md = clean_markdown(raw_md)

    sections = split_sections(cleaned_md)
    chunks = chunk_sections(sections, meta["paper_id"])
    print(f"  Extracted {len(sections)} sections, created {len(chunks)} text chunks.")

    print("\n[4/5] Embedding chunks...")
    chunks = embed_chunks(client, chunks)
    embedding_dim = len(chunks[0]["embedding"])

    # Persist JSON backups locally
    safe_id = meta["paper_id"].replace("/", "_")
    with open(PAPERS_DIR / f"{safe_id}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(PAPERS_DIR / f"{safe_id}.json", "w", encoding="utf-8") as f:
        json.dump({"paper": meta, "chunks": chunks}, f)

    print("\n[5/5] Ingesting into Neo4j...")
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        driver.verify_connectivity()
        create_constraints(driver, NEO4J_DATABASE)

        with driver.session(database=NEO4J_DATABASE) as session:
            session.execute_write(write_paper_transaction, meta, chunks)

        create_vector_index(driver, NEO4J_DATABASE, embedding_dim)

    print(f"\n[DONE] Successfully ingested '{meta['title']}' ({len(chunks)} chunks, {embedding_dim} dim vectors) into Neo4j!")


if __name__ == "__main__":
    paper_input = " ".join(sys.argv[1:]).strip() or input("Enter arXiv ID or paper title: ").strip()
    if paper_input:
        process_paper(paper_input)