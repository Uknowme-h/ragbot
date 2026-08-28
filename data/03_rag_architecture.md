# Retrieval-Augmented Generation Architecture

Retrieval-Augmented Generation (RAG) grounds a language model in documents
you control, instead of relying only on the model's training data.

## Why RAG exists

LLMs hallucinate when they lack facts. They also cannot see private or
recent documents. RAG fixes both problems by retrieving relevant text at
query time and putting that text into the prompt as **context**.

A production RAG system has two pipelines that share a vector index.

## Ingestion pipeline (offline)

1. **Load** files (PDF, Markdown, HTML).
2. **Parse** text. For PDFs, extract per page; fall back to OCR on scanned
   pages that have no text layer.
3. **Clean** headers, page numbers, null bytes, and collapsed whitespace.
4. **Chunk** into overlapping windows so a fact is unlikely to be split
   across two unrelated embeddings.
5. **Embed** each chunk with the same model that will later embed queries.
6. **Store** vectors plus metadata (`source_file`, `page_number`,
   `chunk_index`) in a vector database.

Ingestion is idempotent when each chunk ID is derived from a content hash.
Re-running ingest on an unchanged file should insert nothing.

## Query pipeline (online)

1. **Guard** the user question for prompt injection.
2. **Embed** the question with the ingestion embedding model.
3. **Retrieve** the top-k nearest chunks (typically k = 4 to 8).
4. **Score** the best match. If similarity is below a threshold, do **not**
   call the LLM — return a structured fallback instead. This is cheaper and
   more honest than letting the model guess.
5. **Generate** an answer that is allowed to use only the retrieved chunks,
   with citations back to source file and page.
6. **Log** retrieval latency, generation latency, and token counts.

## Citations

Every factual claim should point at a chunk. A citation at least includes:

- `source_file` — the ingested document name
- `page_number` — PDF page, or 1 for Markdown
- `chunk_index` — position inside that file

If the model cannot find the answer in the chunks, it must say so. The
system prompt must forbid inventing sources.

## Failure modes RAG does not solve

- **Bad chunking** (too small loses context, too large dilutes similarity).
- **Domain mismatch** (embedding model trained on news, documents are code).
- **Stale index** (files changed on disk but ingest was not re-run).
- **Prompt injection** inside retrieved documents (indirect injection).
  Treat retrieved text as untrusted data, never as instructions.

RAG is a retrieval system plus a constrained generator, not a search
engine with a chat skin. The fallback path is part of the product, not an
error.
