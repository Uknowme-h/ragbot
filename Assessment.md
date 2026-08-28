Assessment
Mid-Level AI Engineer
Project Brief
Retrieval-Augmented Generation (RAG) & Agent API
• Time Commitment: 3–4 hours (or a 48-hour take-home window)
• Deliverables: Public GitHub Repository with runnable code, a sample dataset, 
and an evaluation report in README.md.
Task Scenario
Build an API service that ingests a mini-knowledge base (e.g., 5-10 technical 
documentations or PDFs), exposes a query endpoint using RAG with an LLM, and 
includes a fallback tool for missing context.
Technical Requirements
1. Ingestion & Indexing Pipeline
• Parse and chunk a small document set (provided by candidate or markdown 
docs included in the repository).
• Generate embeddings (OpenAI, HuggingFace, or Cohere) and store them in a 
local vector database (ChromaDB, FAISS, or Qdrant).
2. Core Inference Engine & Tool Integration
• Implement a query service using LangChain, LlamaIndex, or raw SDK 
integrations (Python preferred).
• Logic Flow:
1. Retrieve relevant chunks based on semantic similarity.
2. Synthesize an accurate response using a specified LLM, citing the source 
document section.
3. Fallback Guardrail: If confidence or semantic similarity score falls below 
a threshold, trigger a secondary action (e.g., formatted response stating 
insufficient internal context or routing to a web search mock tool).
3. API Layer
• Expose a FastAPI or Express endpoint: POST /api/v1/query accepting 
{ "question": string, "stream": boolean }.
• Support streaming responses (Server-Sent Events or WebSockets) for generated 
outputs

4. Engineering & Safety Standards
• Prompt engineering that prevents hallucination and guards basic prompt 
injection attempts.
• Structured logging tracking latency (chunk retrieval time vs. LLM generation 
time) and token usage per request