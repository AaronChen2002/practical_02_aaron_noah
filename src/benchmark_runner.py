import subprocess
import csv
import time
import os

# ==== CONFIGURATION ====
VECTOR_DBS = ["chroma", "redis", "milvus"]
EMBED_MODELS = ["nomic-embed-text", "all-MiniLM-L6-v2"]
CHUNK_SIZES = [300, 500]
OVERLAPS = [0, 50]
LLMS = ["mistral", "llama2"]  # Used during prompting

# List of test questions to evaluate
TEST_QUERIES = [
    "What is an AVL tree?",
    "What is the CAP theorem?",
    "How does Redis handle hash values?",
    "Explain the purpose of Cypher in Neo4j."
]

# Where results will be saved
RESULTS_FILE = "benchmark_results.csv"

# ==== UTILITY FUNCTIONS ====
def run_ingest(vector_db, model, chunk_size, overlap):
    script = os.path.join(os.path.dirname(__file__), f"ingest_{vector_db}.py")
    subprocess.run([
        "python", script,
        "--chunk_size", str(chunk_size),
        "--overlap", str(overlap),
        "--model", model
    ], check=True)

def run_search(vector_db, query, model, llm):
    script = os.path.join(os.path.dirname(__file__), f"search_{vector_db}.py")
    result = subprocess.run(
        ["python", script, "--query", query, "--model", model, "--llm", llm],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()

def write_result(row):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ==== MAIN LOOP ====
for vector_db in VECTOR_DBS:
    for model in EMBED_MODELS:
        for chunk_size in CHUNK_SIZES:
            for overlap in OVERLAPS:
                for llm in LLMS:
                    print(f"\n[⚙️ RUNNING] DB={vector_db}, Model={model}, LLM={llm}, Chunk={chunk_size}, Overlap={overlap}")
                    try:
                        start_ingest = time.time()
                        run_ingest(vector_db, model, chunk_size, overlap)
                        ingest_time = time.time() - start_ingest

                        for query in TEST_QUERIES:
                            print(f"\n🔍 Query: {query}")
                            start_query = time.time()
                            response = run_search(vector_db, query, model, llm)
                            query_time = time.time() - start_query

                            write_result({
                                "vector_db": vector_db,
                                "embed_model": model,
                                "chunk_size": chunk_size,
                                "overlap": overlap,
                                "llm": llm,
                                "query": query,
                                "response": response,
                                "ingest_time": round(ingest_time, 2),
                                "query_time": round(query_time, 2),
                            })
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Error during processing: {e}")
