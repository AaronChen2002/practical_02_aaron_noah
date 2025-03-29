import argparse
from chromadb import PersistentClient
import ollama
from sentence_transformers import SentenceTransformer

# Load embedding model
def load_model(model_name):
    return SentenceTransformer(f"sentence-transformers/{model_name}")

# Embed query
def get_embedding(model, text):
    return model.encode(text).tolist()

# Run RAG response using Ollama (Mistral)
def generate_rag_response(query, context_chunks):
    context_str = "\n\n".join(
        f"From {meta.get('file')} (page {meta.get('page')}, chunk {meta.get('chunk_index')}):\n{chunk}"
        for chunk, meta in context_chunks
    )

    prompt = f"""You are a helpful AI assistant. 
Use the following context to answer the query. If the context isn't relevant, say "I don't know".

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model="mistral:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def interactive_search(model, collection, top_k=5):
    print("🔍 RAG Search Interface (ChromaDB)")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        query_embedding = get_embedding(model, query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        context_chunks = list(zip(results["documents"][0], results["metadatas"][0]))

        print(f"\n📄 Top {top_k} Retrieved Chunks:")
        for i, (chunk, meta) in enumerate(context_chunks):
            print(f"{i + 1}. File: {meta.get('file')} | Page: {meta.get('page')} | Chunk: {meta.get('chunk_index')}")
            print(f"Text: {chunk[:200]}...\n")

        print("💬 Generating answer...\n")
        response = generate_rag_response(query, context_chunks)
        print("🧠 RAG Response:\n")
        print(response)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--persist_dir", type=str, default="./chroma_store")
    parser.add_argument("--collection", type=str, default="course_notes")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Load model and Chroma collection
    model = load_model(args.model)
    client = PersistentClient(path="./chroma_store")
    collection = client.get_or_create_collection(name=args.collection)

    # interactive CLI
    interactive_search(model, collection, top_k=args.top_k)

if __name__ == "__main__":
    main()
