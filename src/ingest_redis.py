import ollama
import redis
import numpy as np
import os
import fitz
import re
import string
from nltk.corpus import stopwords
import nltk
import time
import argparse
from sentence_transformers import SentenceTransformer

# Download stopwords
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# Redis config
redis_client = redis.Redis(host="localhost", port=6379, db=0)
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text

def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass
    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA file TEXT page TEXT chunk TEXT chunk_text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

def get_embedding(text: str, model_name: str, model_instance=None):
    if model_name == "nomic-embed-text":
        response = ollama.embeddings(model=model_name, prompt=text)
        return response["embedding"]
    elif model_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
        return model_instance.encode(text).tolist()

def store_embedding(file: str, page: str, chunk: str, embedding: list, chunk_text: str):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "chunk_text": chunk_text,
            "embedding": np.array(embedding, dtype=np.float32).tobytes()
        },
    )
    print(f"Stored embedding → file: {file} | page: {page} | chunk: {chunk} | text: {chunk_text[:100]}...")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        raw_text = page.get_text()
        cleaned_text = clean_text(raw_text)
        text_by_page.append((page_num, cleaned_text))
    return text_by_page

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdfs(data_dir, chunk_size, overlap, model_name):
    if model_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
        model_instance = SentenceTransformer(f"sentence-transformers/{model_name}")
    else:
        model_instance = None

    total_chunks = 0
    start = time.time()

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, model_name, model_instance)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk_index),
                        embedding=embedding,
                        chunk_text=chunk
                    )
                    total_chunks += 1
            print(f" -----> Processed {file_name}")

    elapsed = time.time() - start
    print(f"\n⏱ Model: {model_name} | Chunks: {total_chunks} | Time: {elapsed:.2f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--model", type=str, default="nomic-embed-text",
        choices=["nomic-embed-text", "all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    )
    args = parser.parse_args()

    clear_redis_store()
    create_hnsw_index()
    process_pdfs("../data/raw_data", chunk_size=args.chunk_size, overlap=args.overlap, model_name=args.model)
    print("\n---Done processing PDFs---\n")

if __name__ == "__main__":
    main()
