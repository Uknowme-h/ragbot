# Embeddings and Vector Databases

An embedding model maps text to a dense vector so that semantically similar
sentences land near each other. Vector databases store those vectors and
answer nearest-neighbor queries.

## Sentence-Transformers MiniLM

`all-MiniLM-L6-v2` is a 384-dimensional MiniLM model. It runs locally on
CPU, needs no API key, and is accurate enough for a small technical
knowledge base. Always embed documents and queries with the **same** model
and the **same** normalization.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts, batch_size=32, normalize_embeddings=True)
```

Batch encoding (32 chunks at a time) keeps peak RAM bounded. Do not load a
new `SentenceTransformer` on every request; keep one process-wide instance.

## Distance vs similarity

ChromaDB in cosine space returns **distance** = `1 - cosine_similarity`.
Cosine similarity ranges from -1 to 1, so distance ranges from 0 (identical)
to 2 (opposite). Convert for a 0–1 style score:

```
similarity = 1 - (distance / 2)
```

A typical in-domain technical question against a well-chunked corpus
scores 0.5–0.85. Scores below ~0.35 usually mean the index has no related
passage. That is the fallback threshold used by this service.

## ChromaDB locally

ChromaDB can persist to a directory with `PersistentClient(path=...)`.
Collections are created with an explicit metric:

```python
collection = client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"},
)
```

Store metadata beside each vector: `source_file`, `page_number`,
`chunk_index`, `file_hash`. Query with `include=["documents", "metadatas",
"distances"]`.

## Top-k retrieval

`n_results=5` is a solid default for an 8-document corpus. Larger k
increases recall and prompt tokens. If the LLM context is small, drop
chunks below the similarity threshold rather than stuffing low-quality
context.

## Duplicate prevention

Hash the raw file bytes with MD5. If the hash is already in the ingest
manifest, skip the file. Chunk IDs should be `{file_hash}:{chunk_index}` so
a retry is an upsert, not a duplicate.

## What MiniLM is not

MiniLM is not a multilingual specialist, and it is not the best model for
code-only corpora. If queries systematically miss, inspect the top-k
distances before swapping models. Often the issue is chunking, not the
embedding architecture.
