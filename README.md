# Ollama RAG Ingest and Search

## Prerequisites

- Ollama app set up ([Ollama.com](Ollama.com))
- Python with Ollama, Redis-py, and Numpy installed (`pip install ollama redis numpy`)
- Redis Stack running (Docker container is fine) on port 6379.  If that port is mapped to another port in 
Docker, change the port number in the creation of the Redis client in both python files in `src`.

docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest


## Source Code
- `src/ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack
- `src/search.py` - simple question answering using the LLM
- `src/preprocess_text.py` - Extracts and cleans text from pdf documents in `raw_data` folder and creates corresponding txt files in the `cleaned_data` folder

## How to Execute

Configuration Options
- Embedding Models
  - nomic-embed-text: Lightweight and fast, but less accurate.
  - all-MiniLM-L6-v2: Balanced speed and accuracy (384 dimensions).
  - all-mpnet-base-v2: High accuracy but slower (768 dimensions).

- Chunking Strategies
  - Chunk Size: Controls the size of text chunks (e.g., 200, 500, 1000 tokens).

- Overlap: Ensures continuity between chunks (e.g., 0, 50, 100 tokens).

- Vector Databases
  - Redis: High performance and scalability. Requires Docker.
  - Chroma: Easy to set up and use locally. No Docker required.
  - MilvusLite: Lightweight but less feature-rich.
