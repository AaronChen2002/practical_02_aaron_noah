# Ollama RAG Ingest and Search

## Prerequisites

- Ollama app set up ([Ollama.com](Ollama.com))
- Python with Ollama, Redis-py, and Numpy installed (`pip install ollama redis numpy`)
- Redis Stack running (Docker container is fine) on port 6379.  If that port is mapped to another port in 
Docker, change the port number in the creation of the Redis client in both python files in `src`.

docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest


## Source Code
- `src/ingest_redis.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack
- `src/search_redis.py` - simple question answering using the LLM
- `src/preprocess_text.py` - Extracts and cleans text from pdf documents in `raw_data` folder and creates corresponding txt files in the `cleaned_data` folder

- src/chroma_ingest.py: Ingests and processes PDF files from the ./data/raw_data folder. Generates embeddings using a specified model and stores them in ChromaDB.
- src/chroma_search.py: Provides an interactive interface for querying the RAG system. Retrieves relevant context from ChromaDB and generates responses using a locally-running LLM.
- src/milvus_ingest.py: Ingests and processes PDF files from the ./data/raw_data folder. Generates embeddings using a specified model and stores them in MilvusLite.
- src/milvus_search.py: Provides an interactive interface for querying the RAG system. Retrieves relevant context from MilvusLite and generates responses using a locally-running LLM.
