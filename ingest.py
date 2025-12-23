import os
import trafilatura
import cohere
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "physical-ai-textbook"  # updated collection name

cohere_client = cohere.Client(COHERE_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# --- DELETE OLD COLLECTION ---
try:
    qdrant.delete_collection(COLLECTION_NAME)
    print(f"Old collection {COLLECTION_NAME} deleted!")
except Exception as e:
    print(f"No old collection to delete or error: {e}")

# --- FUNCTIONS ---

def read_chapter(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, max_chars=1200):
    chunks = []
    while len(text) > max_chars:
        split_pos = text[:max_chars].rfind(".")
        if split_pos == -1:
            split_pos = max_chars
        chunks.append(text[:split_pos])
        text = text[split_pos:]
    chunks.append(text)
    return chunks

def embed_text(text):
    res = cohere_client.embed(
        model="embed-english-v3.0",
        texts=[text],
        input_type="search_document"  # required for v3 models
    )
    return res.embeddings[0]

def create_collection():
    print(f"Creating Qdrant collection: {COLLECTION_NAME}")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,  # Cohere embed-english-v3.0 dimension
            distance=Distance.COSINE
        )
    )

def ingest_book():
    create_collection()
    global_id = 1
    data_folder = os.path.join(os.getcwd(), "data")

    # --- Explicit file order ---
    file_names = [f"chapter{i}.txt" for i in range(1, 9)]

    for file_name in file_names:
        file_path = os.path.join(data_folder, file_name)
        if os.path.exists(file_path):
            text = read_chapter(file_path)
            chunks = chunk_text(text)
            for chunk in chunks:
                vector = embed_text(chunk)
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[{
                        "id": global_id,
                        "vector": vector,
                        "payload": {"text": chunk}
                    }]
                )
                print(f"Saved chunk {global_id}")
                global_id += 1
        else:
            print(f"File not found: {file_name}")

    print(f"Ingestion completed! Total chunks: {global_id-1}")

# --- MAIN ---
if __name__ == "__main__":
    ingest_book()
