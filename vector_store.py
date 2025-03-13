# vector_store.py
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from typing import List
import os

CHROMA_DB_DIR = "C:\\CS Tech\\Agents\\Daily\\GPT\\latest_chroma_db_insurance"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def split_text(text: str, chunk_size=500, overlap=50) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_chroma_collection():
    client = PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(name="insurance_docs")
    return collection

def store_chunks_in_chroma(file_id: str, text: str):
    collection = get_chroma_collection()

    chunks = split_text(text)
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{file_id}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"file_id": file_id}] * len(chunks)
    )
