import chromadb
from notes import KNOWLEDGE_BASE

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="it_notes_v2",
    metadata={"hnsw:space": "cosine"},
)

documents = [entry["content"] for entry in KNOWLEDGE_BASE]
metadatas = [{"topic": entry["topic"]} for entry in KNOWLEDGE_BASE]
ids = [f"note_{i}" for i in range(len(KNOWLEDGE_BASE))]

collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

print(f"Ingested {len(documents)} notes into the knowledge base.")