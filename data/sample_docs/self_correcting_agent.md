# Self-Correcting Agentic RAG Technical Specifications

## Problem Statement
Standard naive RAG systems often fail due to:
1. Low retrieval precision (retrieving irrelevant chunks that distract the generator)
2. Ungrounded answers / hallucinations
3. Vague or poorly formatted initial user queries

## The Self-Correction Loop
The system evaluates the retrieved chunks before generation.
- If irrelevant chunks are detected, it rewrites the query and executes re-retrieval.
- Maximum retry limits prevent infinite execution cycles.
- After generation, the answer is checked against source facts. If hallucination is detected, the query is rewritten and re-evaluated.
- Every state transition is recorded into persistent PostgreSQL checkpoints using the thread_id.
