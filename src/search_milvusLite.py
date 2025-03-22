from pymilvus import MilvusClient
import ollama
from sentence_transformers import SentenceTransformer
import argparse

COLLECTION_NAME = "document_store"

def load_model(model_name):
    """Load the embedding model"""
    if model_name == "nomic-embed-text":
        return None
    return SentenceTransformer(f"sentence-transformers/{model_name}")

def get_embedding(text: str, model_name: str, model_instance=None) -> list:
    """Generate embedding for text using specified model"""
    if model_name == "nomic-embed-text":
        response = ollama.embeddings(model=model_name, prompt=text)
        return response["embedding"]
    return model_instance.encode(text).tolist()

def generate_rag_response(query, context_chunks):
    """Generate RAG response using Ollama"""
    context_str = "\n\n".join(
        f"From {chunk['file']} (page {chunk['page']}):\n{chunk['text']}"
        for chunk in context_chunks
    )

    prompt = f"""You are a helpful AI assistant. 
Use the following context to answer the query. If the context isn't relevant, say "I don't know".

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model="llama2",  # or any other model you have pulled
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def interactive_search(client, model_name, model_instance, top_k=5):
    """Interactive search interface for MilvusLite"""
    print("🔍 RAG Search Interface (MilvusLite)")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Generate query embedding
        query_embedding = get_embedding(query, model_name=model_name, model_instance=model_instance)

        # Search in Milvus
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=top_k,
            output_fields=["file", "page", "text"]
        )

        # Process and display results
        print(f"\n📄 Top {top_k} Retrieved Chunks:")
        context_chunks = []
        for i, hit in enumerate(results[0]):
            chunk = hit['entity']
            similarity = 1 - hit['distance']
            context_chunks.append(chunk)
            
            print(f"{i + 1}. File: {chunk['file']} | Page: {chunk['page']}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Text: {chunk['text'][:200]}...\n")

        print("💬 Generating answer...\n")
        response = generate_rag_response(query, context_chunks)
        print("🧠 RAG Response:\n")
        print(response)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="nomic-embed-text",
                      choices=["nomic-embed-text", "all-MiniLM-L6-v2"])
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Initialize MilvusLite client
    client = MilvusClient("./milvuslite_data/milvus.db")
    
    # Load model
    model_instance = load_model(args.model)

    # Start interactive search
    interactive_search(client, args.model, model_instance, top_k=args.top_k)

if __name__ == "__main__":
    main()