## DS 4300 Example - from docs

import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import re
import string
from nltk.corpus import stopwords
import nltk
import argparse
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


def clean_text(text):
    """Clean text by removing extra whitespace, punctuation, and stopwords."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra whitespaces
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join(word for word in text.split() if word not in STOPWORDS)  # Remove stopwords
    return text

# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


# Generate an embedding using the specified model
def get_embedding(text: str, model_instance) -> list:
    if isinstance(model_instance, SentenceTransformer):
        # Use sentence-transformers model
        embedding = model_instance.encode(text)
    elif isinstance(model_instance, INSTRUCTOR):
        # Use InstructorXL model
        instruction = "Represent the text for retrieval:"
        embedding = model_instance.encode([[instruction, text]]).flatten().tolist()
    else:
        # Default to the original Ollama model
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        embedding = response["embedding"]
    return embedding



# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list, chunk_text: str):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "text": chunk_text,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {chunk}")


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        raw_text = page.get_text()
        cleaned_text = clean_text(raw_text)
        text_by_page.append((page_num, cleaned_text))

    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir, model: str, chunk_size: int, overlap: int, model_instance):

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk, model_instance=model_instance)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk_index),
                        chunk_text=chunk,
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


def query_redis(query_text: str, model: str, model_instance):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance", "chunk_text", "file", "page")
        .dialect(2)
    )
    embedding = get_embedding(query_text, model_instance=model_instance)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    # print(res.docs)
    print(f"\nQuery results using model: {model}")
    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")
        print(f"File: {doc.file}")
        print(f"Page: {doc.page}")
        print(f"Chunk Text: {doc.chunk_text}")
        print("------")
    


# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process PDFs and generate embeddings.")
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        choices=["nomic-embed-text", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "InstructorXL"],
        help="Embedding model to use.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=300,
        help="Size of each text chunk (in words).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap between text chunks (in words).",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="What is the capital of France?",
        help="Query text for similarity search.",
    )
    args = parser.parse_args()

    # Load the model once (improves efficiency)
    if args.model == "all-MiniLM-L6-v2":
        model_instance = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    elif args.model == "all-mpnet-base-v2":
        model_instance = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    elif args.model == "InstructorXL":
        model_instance = INSTRUCTOR('hkunlp/instructor-xl')
    else:
        model_instance = None  # Use Ollama by default

    # Clear Redis and create index
    clear_redis_store()
    create_hnsw_index()

    # Process PDFs using the selected model and chunking parameters
    process_pdfs("../data/raw_data", model=args.model, chunk_size=args.chunk_size, overlap=args.overlap, model_instance=model_instance)
    print(f"\n---Done processing PDFs using model: {args.model}---\n")

    # Query Redis using the same model
    query_redis(args.query, model=args.model, model_instance=model_instance)


if __name__ == "__main__":
    main()
