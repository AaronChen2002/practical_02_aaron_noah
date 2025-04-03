import subprocess
import time
import os

VECTOR_DBS = ["chroma", "redis", "milvusLite"]
EMBED_MODELS = ["all-MiniLM-L6-v2", "nomic-embed-text"]
CHUNK_SIZES = [300, 500]
OVERLAPS = [0, 50]
LLMS = ["mistral", "llama2"]


def get_user_choice(options, prompt):
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        choice = input("Enter number: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            print("Invalid input. Try again.")


def run_ingest(vector_db, model, chunk_size, overlap):
    script = os.path.join(os.path.dirname(__file__), f"ingest_{vector_db}.py")
    print(f"Running ingest script: {script}")

# chroma uses hyphens, redis, milvusLite use underscores
    if vector_db in ["redis", "milvusLite"]:
        chunk_arg = "--chunk_size"
        overlap_arg = "--overlap"
    else:
        chunk_arg = "--chunk-size"
        overlap_arg = "--overlap"

    start = time.time()
    subprocess.run([
        "python", script,
        chunk_arg, str(chunk_size),
        overlap_arg, str(overlap),
        "--model", model
    ], check=True)
    return round(time.time() - start, 2)



def run_search(vector_db, query, model, llm):
    script = os.path.join(os.path.dirname(__file__), f"search_{vector_db}.py")
    print(f"\nRunning search script: {script}")
    start = time.time()
    result = subprocess.run([
        "python", script,
        "--query", query,
        "--model", model,
        "--llm", llm
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip(), round(time.time() - start, 2)


def main():
    print("Welcome to the Interactive RAG Driver")

    vector_db = get_user_choice(VECTOR_DBS, "Choose a Vector DB:")
    embed_model = get_user_choice(EMBED_MODELS, "Choose an Embedding Model:")
    chunk_size = int(get_user_choice(list(map(str, CHUNK_SIZES)), "Choose a Chunk Size:"))
    overlap = int(get_user_choice(list(map(str, OVERLAPS)), "Choose an Overlap:"))
    llm = get_user_choice(LLMS, "Choose an LLM:")

    query = input("\nEnter your query: ")

    try:
        ingest_time = run_ingest(vector_db, embed_model, chunk_size, overlap)
        response, query_time = run_search(vector_db, query, embed_model, llm)

        print("\nRAG Response:")
        print(response)

        print("\nMetadata:")
        print(f"Vector DB: {vector_db}")
        print(f"Embedding Model: {embed_model}")
        print(f"Chunk Size: {chunk_size}")
        print(f"Overlap: {overlap}")
        print(f"LLM: {llm}")
        print(f"Query: {query}")
        print(f"Ingest Time (sec): {ingest_time}")
        print(f"Query Time (sec): {query_time}")

    except subprocess.CalledProcessError as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()