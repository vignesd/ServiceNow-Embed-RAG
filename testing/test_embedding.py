from langchain_ollama import OllamaEmbeddings

MODEL = "nomic-embed-text"

embeddings = OllamaEmbeddings(
    model=MODEL,
    base_url="http://127.0.0.1:11434",
)

print("Test 1: Embed a single query")

try:
    vector = embeddings.embed_query(
        "How do I request software access in ServiceNow?"
    )
    print(f"SUCCESS: Embedding dimensions = {len(vector)}")

except Exception:
    print("FAILED: Query embedding")
    raise


print("\nTest 2: Embed multiple documents")

try:
    vectors = embeddings.embed_documents(
        [
            "How to request software access.",
            "How to configure a catalog item.",
            "How to troubleshoot access issues.",
        ]
    )

    print(f"SUCCESS: Documents embedded = {len(vectors)}")
    print(f"Embedding dimensions = {len(vectors[0])}")

except Exception:
    print("FAILED: Document embedding")
    raise
