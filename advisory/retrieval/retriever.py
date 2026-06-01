import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection("coffee_kb")


def check_zone_exists(zone: str) -> bool:
    """
    Returns True if the KB has at least one Excel record for this zone.
    Uses a metadata-only filter (no embedding needed) — fast and reliable.
    If zone is None/empty, returns False immediately.
    """
    if not zone or not zone.strip():
        return False
    try:
        results = collection.get(
            where={"$and": [
                {"source": {"$eq": "excel"}},
                {"zone":   {"$eq": zone.strip()}}
            ]},
            limit=1,          # we only need to know if at least 1 exists
            include=["metadatas"]
        )
        return len(results["ids"]) > 0
    except Exception:
        # If the query itself errors (e.g. zone string has special chars),
        # treat as no data — safe default.
        return False


def retrieve(query: str, zone: str = None, crop: str = None, n_results: int = 8) -> list[str]:
    query_embedding = embedding_model.encode([query]).tolist()[0]

    # Step 1: Try filtered search (zone + crop specific Excel records)
    filtered_docs = []
    if zone and crop:
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where={"$and": [
                    {"zone": {"$eq": zone}},
                    {"crop": {"$eq": crop}}
                ]},
                include=["documents", "distances"]
            )
            # Only accept results with good similarity (distance < 0.6)
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                if dist < 0.6:
                    filtered_docs.append(doc)
        except Exception:
            pass

    # Step 2: Always add MD rule/band chunks (source = all zones)
    try:
        rule_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4,
            where={"zone": {"$eq": "all"}},
            include=["documents", "distances"]
        )
        rule_docs = [
            doc for doc, dist in zip(rule_results["documents"][0], rule_results["distances"][0])
            if dist < 0.7
        ]
    except Exception:
        rule_docs = []

    combined = filtered_docs + rule_docs

    # Step 3: Fallback — unfiltered search if nothing retrieved
    if not combined:
        fallback = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents"]
        )
        combined = fallback["documents"][0]

    return combined[:n_results]


if __name__ == "__main__":
    print("Testing retriever...")
    print("Zone 'Kodagu' exists:", check_zone_exists("Kodagu"))
    print("Zone 'RandomPlace' exists:", check_zone_exists("RandomPlace"))
    res = retrieve("acidity", zone="Kodagu", crop="Robusta")
    print(f"Results: {len(res)} chunks found.")
    if res:
        print(f"First chunk snippet: {res[0][:100]}...")