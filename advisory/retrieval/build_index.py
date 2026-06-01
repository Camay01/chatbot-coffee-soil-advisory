import os
import re
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    chroma_client.delete_collection("coffee_kb")
except:
    pass

collection = chroma_client.get_or_create_collection(
    name="coffee_kb",
    metadata={"hnsw:space": "cosine"}
)

documents, ids, metadatas = [], [], []

# ── 1. Excel rows ──────────────────────────────────────────────────────────────
print("Indexing Excel records...")
ROOT_DIR = os.path.dirname(BASE_DIR)
KB_DIR = os.path.join(ROOT_DIR, "kb")
df = pd.read_excel(os.path.join(KB_DIR, "coffee_data.xlsx"))  # correct filename

for _, row in df.iterrows():
    text = f"""Coffee Soil Advisory Record
Zone: {row['Zone']} | State: {row['State']} | Crop: {row['Crop']} | Variety: {row['Variety']}
pH: {row['pH']} | OC: {row['Organic_C_percent']}% | N: {row['Available_N_kg_ha']} kg/ha | P: {row['Available_P_kg_ha']} kg/ha | K: {row['Available_K_kg_ha']} kg/ha
Zn: {row['Zn_mg_kg']} mg/kg | B: {row['B_mg_kg']} mg/kg
Micronutrient flag: {row['Micronutrient_flag']}
Major limiting factor: {row['Major_limiting_factor']}
Risk Level: {row['Risk Level']}
Technical Interpretation: {row['Technical Interpretation']}
Suggested Intervention: {row['Suggested Intervention']}
Priority Action: {row['Priority Action']}"""

    documents.append(text)
    ids.append(f"excel_{row['Record_ID']}")
    metadatas.append({
        "source": "excel",
        "zone": str(row['Zone']),
        "crop": str(row['Crop']),
        "variety": str(row['Variety']),
        "risk_level": str(row['Risk Level'])
    })

print(f"  Excel: {len(documents)} records")

# ── 2. MD files ────────────────────────────────────────────────────────────────
MD_FILES = {
    "advisory_rules.md":       "advisory_rules",
    "interpretation_bands.md": "interp_bands",
    "parameter_dictionary.md": "param_dict",
}

print("Indexing MD files...")
for filename, tag in MD_FILES.items():
    path = os.path.join(KB_DIR, filename)
    if not os.path.exists(path):
        print(f"  WARNING: {filename} not found, skipping")
        continue
    with open(path, "r") as f:
        content = f.read()
    # Split on headers or table rows
    sections = re.split(r'\n(?=#{1,3} |\| [A-Z])', content)
    count = 0
    for i, section in enumerate(sections):
        section = section.strip()
        if len(section) < 50:
            continue
        documents.append(section)
        ids.append(f"{tag}_{i}")
        metadatas.append({"source": tag, "zone": "all", "crop": "all"})
        count += 1
    print(f"  {filename}: {count} chunks")

# ── 3. Embed and store ─────────────────────────────────────────────────────────
print("\nGenerating embeddings...")
embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()

print("Storing in ChromaDB...")
batch_size = 100
for i in range(0, len(documents), batch_size):
    collection.add(
        ids=ids[i:i+batch_size],
        documents=documents[i:i+batch_size],
        embeddings=embeddings[i:i+batch_size],
        metadatas=metadatas[i:i+batch_size]
    )

print(f"\n Done! Total chunks indexed: {collection.count()}")