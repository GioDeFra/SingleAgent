# SingleAgent

Single-agent legal RAG using the existing `legal-rag` Pinecone corpus.

## Setup

Install dependencies with `python -m pip install -r requirements.txt`.
Keep your existing `Apikey.env` provider settings and add `PINECONE_API_KEY`
if it is not already there. Optional settings: `PINECONE_INDEX_NAME` (default
`legal-rag`) and `PINECONE_NAMESPACE` (default empty, matching the old project).
Query embeddings use `BAAI/bge-m3` with normalization, matching the old retriever.
The first retrieval may download this embedding model.

Run `python main.py`. Add `--long-term` to enable Chroma memory; this may also
download `all-mpnet-base-v2`. Short-term context and SQLite history are always on.

## Flow

One triage call either generates a concise direct answer or rewrites the question
as a standalone retrieval query. Retrieval uses the rewritten question, while
answer generation receives both the original question and the rewritten version.
Invalid triage responses fall back to retrieval with the original question.
No-source results produce an explicit explanation instead of an unsupported
legal answer. Service failures are raised, not treated as empty search results.

`SingleAgentRAG.ask(query)` returns a dictionary with `answer`, `needs_retrieval`,
`search_query`, `retrieved_documents`, `session_id`, and `turn_id`.
Use `new_session()` or `load_session(session_id)` to change conversations.
Use one instance per conversation. History can be listed via
`rag.history.list_sessions()`.

## Memory reuse

The short-term and chat-history implementations are copied from MultiAgentRag.
Their agent-list fields are retained for compatibility: retrieval turns record
`single_agent`, and direct answers record an empty list. History saves original
questions and full answers; short-term context uses the last three turns.

Long-term memory is optional and stores rewritten questions and summarized RAG
answers. Recall and deduplication are scoped by both agent and exact country set.
Countries currently come from retrieved source metadata, not a jurisdiction
classifier. Memory is background only, never a substitute for document evidence.
Only turns with retrieved documents are stored, but there is no independent
grounding validator yet.

Storage defaults to this project's `data/` directory. Existing multi-agent
databases are not copied or changed. Reusing implementation does not automatically
import old conversations. Old agent-tagged semantic memories will not match the
`single_agent` filter without an explicit migration.

## Retrieval limits

The retriever reads eight nearest chunks across the corpus by default. An optional
`metadata_filter` can be passed to `PineconeRetriever`. It preserves ingestion
`citation_label` values and skips missing or ambiguous labels. This initial
version has no reranker, automatic country filters, or independent citation
verification; evaluate retrieval coverage, especially for country comparisons.

Run offline checks with `python -m unittest discover -s tests -v`.
