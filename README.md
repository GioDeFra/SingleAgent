# SingleAgent

A single-agent legal retrieval-augmented generation (RAG) application for
divorce and inheritance questions concerning Italy, Estonia, and Slovenia.
It uses an existing Pinecone corpus, local embedding and reranking models,
and an OpenAI-compatible language model provider.

## Setup and launch

Run the following commands from this project folder, with your Python
environment activated:

```bash
python -m pip install -r requirements.txt
```

Configure `Apikey.env` in the project root. For example, to use DeepSeek:

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=legal-rag
PINECONE_NAMESPACE=
```

The configured providers are `deepseek`, `groq`, and `gemini`. Use the
corresponding `DEEPSEEK_API_KEY`, `GROQ_API_KEY`, or `GEMINI_API_KEY`.
Provider endpoints and model names are defined in `llm_client.py`.
Keep real API keys private.

The Pinecone index must already contain compatible documents and embeddings.
This project reads the existing corpus; it does not ingest documents.
The index defaults to `legal-rag` and the namespace defaults to empty.

Start the Gradio interface:

```bash
python ui.py
```

Open the local address printed in the terminal. The interface provides chat,
previous conversations, new-chat controls, memory status, and retrieved sources.
It binds to `127.0.0.1` and shares one active conversation across browser tabs.
The terminal chat entry point `main.py` has been removed.

First startup may download the long-term memory model. The first retrieval
may download the embedding and reranking models.

## Project structure

| File or directory | Responsibility |
| --- | --- |
| `ui.py` | Gradio interface, conversation selection, and source display. |
| `agent.py` | `SingleAgentRAG`: routing, retrieval, generation, verification, and memory updates. |
| `prompts.py` | Prompts for routing, grounded answers, and answers without sources. |
| `retrieval.py` | Metadata filters, Pinecone search, candidate validation, and reranking. |
| `llm_client.py` | Provider configuration and shared client creation. |
| `guardrails/output_guard.py` | Citation checks and one possible corrective pass. |
| `memory/short_term.py` | Recent conversation context held in RAM. |
| `memory/chat_history.py` | SQLite conversation history and JSON export. |
| `memory/long_term.py` | Persistent semantic memory in ChromaDB. |
| `ragas_evaluation.py` | Independent dataset collection and Ragas scoring. |
| `questions.json` | Evaluation questions and reference answers. |
| `requirements.txt` | Application dependencies. |
| `requirements-evaluation.txt` | Application and evaluation dependencies. |
| `tests/check_agent_refactor.py` | Offline comparisons against a supplied pre-refactor agent snapshot. |

To study the application, start with `ui.py`, then follow `agent.py` and
`prompts.py`, `retrieval.py`, the output guard, and the memory modules.
Read `llm_client.py` alongside these modules to understand model configuration.

## Answer pipeline

1. **Read conversation context.** The agent uses the last three turns to
   interpret follow-up questions.
2. **Route the question.** One model call chooses a direct answer or retrieval.
   For retrieval, it produces a standalone question and selects countries,
   legal areas, and document types. Questions needing a jurisdiction ask for
   clarification when no country is specified.
3. **Retrieve and rerank.** The standalone question is embedded and submitted
   to Pinecone with metadata filters. Invalid candidates are removed, and a
   cross-encoder reranks the remaining passages.
4. **Check country coverage.** Every retained passage must have country
   metadata, and the combined country set must exactly match the requested set.
   Empty results or incomplete coverage trigger an answer without sources.
   Requests including unsupported countries also use this fallback.
5. **Generate the answer.** The grounded path receives the original question,
   standalone query, recent conversation, retrieved documents, and relevant
   semantic memories. Retrieved documents provide evidence; memories provide
   background.
6. **Verify citations.** The output guard checks cited claims, may request one
   revision, and returns the final answer with any verification notices.
7. **Save the turn.** Short-term context and chat history are updated.
   Only answers with retrieved documents and successful citation verification
   are eligible for long-term memory storage.

The fallback prompt requests an answer explicitly labeled as outside the
retrieved corpus, without source citations. Direct and fallback answers bypass
the citation guard.

Invalid routing responses or filters, and exceptions during routing, produce
a clarification response. Retrieval service errors propagate rather than being
treated as empty results. Empty filtered results are not retried with broader
filters.

## Retrieval configuration

| Setting | Default |
| --- | --- |
| Query embedding model | `BAAI/bge-m3`, normalized embeddings |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` |
| Candidates per Pinecone search | Up to 20 |
| Retained passages after reranking | Up to 5 |
| Countries | `Italy`, `Estonia`, `Slovenia` |
| Legal areas (`law`) | `Divorce`, `Inheritance` |
| Document types (`doc_type`) | `Legal Cases`, `Civil Codes` |

Query embeddings must match the model used to build the corpus.
`PineconeRetriever` accepts configurable `n_retrieve`, `top_k`, namespace,
and an optional base metadata filter.

Multiple values in one field use `$in`; different fields use `$and`.
Unspecified legal areas and document types remain unrestricted.
Caller-configured filters are combined with routing filters.

Candidates with missing text or citation labels are excluded. Labels shared
by different sources are also excluded. Ingestion citation labels are preserved.
Returned documents retain Pinecone `score` and add `rerank_score`.
Answer generation, source display, and evaluation use the final reranked order.

All requested countries and document types share one search budget.
There are no separate searches or quotas per country. Exact country coverage
does not guarantee that the passages answer every part of the question.

## Memory and citation checks

Storage defaults to the project's `data/` directory:

- `chat_history.db`: persistent conversation history.
- `chat_history.json`: readable history and evaluation fields.
- `chroma_db/`: long-term semantic memory using `all-mpnet-base-v2`.

Short-term memory retains up to ten turns and supplies the latest three to
the answer pipeline. Starting a new session clears that conversation context
while retaining saved chats and long-term memory.

Long-term recall and deduplication are scoped to `single_agent` and the exact
country set. Verified RAG answers are stored with their standalone questions.
Memory failures are logged without stopping answer generation. Existing
multi-agent databases are not automatically imported.

`check_rag_answer()` checks up to ten citation/claim pairs per pass against
retrieved text and metadata. It can attempt one correction for unknown or
unsupported citations, then check the revision. Missing citations, skipped
checks, API failures, or unresolved problems produce a notice and prevent
long-term storage. Full chat history still retains these answers.

A label mentioned inside another source is not enough to establish a retrieved
citation; a subsection citation may match its retrieved base article.
The guard checks parsed citations, not every uncited assertion, and its model
judgments are not a guarantee of factual correctness.

## Python interface

```python
from agent import SingleAgentRAG

rag = SingleAgentRAG()
result = rag.ask("What are the legal conditions for divorce in Italy?")
print(result["answer"])
```

Each instance represents one conversation. Use `new_session()` to start a
fresh conversation, `load_session(session_id)` to resume one, and
`rag.history.list_sessions()` to list saved sessions.

`ask()` returns:

| Field | Meaning |
| --- | --- |
| `answer` | Final answer, including any verification notice. |
| `needs_retrieval` | Whether routing selected retrieval; may remain true when retrieval leads to fallback. |
| `search_query` | Standalone retrieval query, or null for a direct route. |
| `metadata_filter` | Query-specific metadata filter. |
| `retrieved_documents` | Retained sources, or an empty list when no sources are used. |
| `citations_verified` | Citation verification result; null when the guard was not invoked. |
| `evaluation` | Evaluation record prepared without running Ragas scoring. |
| `session_id`, `turn_id` | Conversation and turn identifiers. |

## Ragas evaluation

Evaluation has two separate stages: collect answers, then score the saved dataset.

### Collect answers

`questions.json` contains a list of objects with non-empty `question` and
`reference` strings. Reference answers are attached after generation and are
never passed to the RAG as answer evidence.

```bash
python ragas_evaluation.py --input questions.json
```

Collection creates a timestamped directory under `ragas_results/` and saves
`dataset.json` after every question. Each case starts a fresh session and
long-term memory is disabled for this run. Normal UI memory is unaffected.
Generation failures are recorded and collection continues.

Successful records contain `id`, `user_input`, `reference`, `response`,
`retrieved_contexts`, `contexts_by_agent`, and `status`. Contexts include
full passage text and metadata in reranked order. When sources exist,
`contexts_by_agent` uses the key `single_agent`.

### Score the dataset

```bash
python -m pip install -r requirements-evaluation.txt
python ragas_evaluation.py --score-only ragas_results/YOUR_RUN/dataset.json
```

Replace `YOUR_RUN` with the collection directory. Scoring creates a new
results directory containing the copied dataset, `scores.json`,
`scores.csv`, and `summary.json`. Use `--output PATH` with either stage
to choose a new output directory; the directory must not already exist.

The evaluator uses Ragas 0.3.2, a DeepSeek judge configured in
`ragas_evaluation.py`, and local BGE-M3 embeddings. It requires
`DEEPSEEK_API_KEY` even if the chat uses another provider. BGE-M3 must already
be cached locally for scoring. Embeddings use CUDA when available, otherwise CPU.

The five metrics are context precision with reference, context recall,
faithfulness, answer relevancy, and answer correctness. Context-dependent
metrics are skipped when no contexts were retrieved. Metric failures are
recorded separately.

Chat history also includes evaluation fields, but its JSON is grouped by
session. Pass the batch `dataset.json` to the scoring command.
Questions that do not match `questions.json` have null reference and case ID
in their automatically prepared chat evaluation records.
