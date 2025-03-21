import os
import fitz
import re
import string
import uuid
import argparse
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

# Clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text

# Chunk text into overlapping segments
def extract_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Extract cleaned text from PDF
def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        raw = page.get_text()
        if raw:
            cleaned = clean_text(raw)
            text_by_page.append((page_num, cleaned))
    return text_by_page

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk_size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--data_dir", type=str, default="../data/raw_data")
    parser.add_argument("--persist_dir", type=str, default="./chroma_store")
    parser.add_argument("--collection", type=str, default="course_notes")
    args = parser.parse_args()

    # Load model
    model = SentenceTransformer(f"sentence-transformers/{args.model}")

    # Initialize ChromaDB
    client = PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(name=args.collection)

    # Process PDFs
    for filename in os.listdir(args.data_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(args.data_dir, filename)
            text_by_page = extract_text_from_pdf(path)

            for page_num, text in text_by_page:
                chunks = extract_chunks(text, args.chunk_size, args.overlap)
                for idx, chunk in enumerate(chunks):
                    embedding = model.encode(chunk).tolist()
                    uid = str(uuid.uuid4())
                    metadata = {
                        "file": filename,
                        "page": str(page_num),
                        "chunk_index": str(idx)
                    }
                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        ids=[uid]
                    )
                    print(f"Stored chunk: {chunk[:100]}...")

    print("✅ Done ingesting into ChromaDB.")

if __name__ == "__main__":
    main()
