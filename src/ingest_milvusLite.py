import numpy as np
from pymilvus import MilvusClient
import os
import fitz
import re
from nltk.corpus import stopwords
import nltk
import argparse
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
import ollama

# Download NLTK stopwords if not already present
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

VECTOR_DIM = 768  # This works for both nomic-embed-text and all-MiniLM-L6-v2
COLLECTION_NAME = "document_store"

def init_milvus_lite():
    """Initialize MilvusLite client"""
    os.makedirs("milvuslite_data", exist_ok=True)
    client = MilvusClient("./milvuslite_data/milvus.db")
    print("MilvusLite initialized successfully")
    return client

def setup_collection(client):
    """Create collection if it doesn't exist, or drop and recreate if it does"""
    # Drop collection if it exists
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    
    # Create new collection with proper field types and dimension
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=VECTOR_DIM,  # Use the required dimension parameter
        primary_field_name="id",
        vector_field_name="embedding"
    )
    
    print("Collection created successfully")

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return ' '.join(words)

def extract_text_from_pdf(pdf_path: str) -> list:
    """Extract text from PDF file, returning list of (page_num, text) tuples"""
    doc = fitz.open(pdf_path)
    text_by_page = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        text = clean_text(text)
        if text.strip():
            text_by_page.append((page_num + 1, text))
    
    return text_by_page

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def get_embedding(text: str, model_name: str, model_instance=None) -> list:
    """Generate embedding for text using specified model"""
    if model_name == "nomic-embed-text":
        response = ollama.embeddings(model=model_name, prompt=text)
        return response["embedding"]
    elif model_name == "all-MiniLM-L6-v2":
        return model_instance.encode(text).tolist()

def store_embedding(client, file: str, page: str, chunk: str, embedding: list, chunk_text: str):
    """Store document chunk and its embedding in MilvusLite"""

    embedding_vector = np.array(embedding, dtype=np.float32)

    data = {
        # "file": [file],
        # "page": [page],
        # "chunk": [chunk],
        # "text": [chunk_text],
        "embedding": [embedding_vector]
    }
    client.insert(
        collection_name=COLLECTION_NAME,
        data=data
    )
    print(f"Stored embedding → file: {file} | page: {page} | chunk: {chunk} | text: {chunk_text[:100]}...")

def process_pdfs(client, data_dir: str, model: str, chunk_size: int, overlap: int, model_instance):
    """Process all PDFs in directory and store embeddings"""
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            print(f"Processing {file_name}...")
            
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, model_name=model, model_instance=model_instance)
                    store_embedding(
                        client=client,
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk_index),
                        embedding=embedding,
                        chunk_text=chunk
                    )
            print(f"Completed processing {file_name}")

def query_milvus(client, query_text: str, model: str, model_instance):
    """Search for similar documents using query text"""
    query_embedding = get_embedding(query_text, model_name=model, model_instance=model_instance)
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=5,
        output_fields=["file", "page", "text"]
    )

    print(f"\nQuery results using model: {model}")
    for hit in results[0]:
        print(f"\nSimilarity Score: {1 - hit.distance:.4f}")
        print(f"File: {hit.entity.get('file')}")
        print(f"Page: {hit.entity.get('page')}")
        print(f"Text: {hit.entity.get('text')}")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Process PDFs and store embeddings in MilvusLite")
    parser.add_argument("--model", type=str, default="nomic-embed-text",
                      choices=["nomic-embed-text", "all-MiniLM-L6-v2"],
                      help="Model to use for embeddings")
    parser.add_argument("--chunk-size", type=int, default=300,
                      help="Size of text chunks")
    parser.add_argument("--overlap", type=int, default=50,
                      help="Overlap between chunks")
    parser.add_argument("--query", type=str, default=None,
                      help="Query text to search for similar documents")
    
    args = parser.parse_args()

    # Initialize the embedding model
    if args.model == "nomic-embed-text":
        model_instance = None
    else:  # all-MiniLM-L6-v2
        model_instance = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Initialize MilvusLite client
    client_ = init_milvus_lite()
    
    # Setup collection
    setup_collection(client_)

    # Process PDFs
    process_pdfs(data_dir="data/raw_data", client=client_, model=args.model, 
                 chunk_size=args.chunk_size, overlap=args.overlap, 
                 model_instance=model_instance)
    print(f"\n---Done processing PDFs using model: {args.model}---\n")

    # Perform query if provided
    if args.query:
        query_milvus(client_, args.query, model=args.model, model_instance=model_instance)

if __name__ == "__main__":
    main()
