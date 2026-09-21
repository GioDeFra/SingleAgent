"""Coordinate one conversation through routing, retrieval, verification, and memory."""

import json
import logging
import re
from typing import Optional, Tuple

from llm_client import LLM_PROVIDER, get_llm_client, model_names
from retrieval import FILTER_VALUES, build_metadata_filter
from prompts import FALLBACK_PROMPT, GROUNDED_ANSWER_PROMPT, TRIAGE_PROMPT

logger = logging.getLogger(__name__)
_MODELS = model_names()

COUNTRY_CLARIFICATION = "Which country or countries are you referring to?"
ROUTING_CLARIFICATION = (
    "Please clarify which country or countries your question concerns and restate the question."
)


class SingleAgentRAG:
    def __init__(self, retriever=None, long_term=None, history=None, data_dir=None):
        """Initialize conversation memory; create the retriever only when needed."""
        self.llm_client = get_llm_client()
        from pathlib import Path
        from memory.short_term import ShortTermMemory
        from memory.chat_history import ChatHistoryStore

        self.retriever = retriever
        self.ltm = long_term
        self.data_dir = Path(data_dir or Path(__file__).parent / "data")
        if self.ltm is None:
            from memory.long_term import LongTermMemory
            self.ltm = LongTermMemory(db_dir=str(self.data_dir / "chroma_db"))
        self.stm = ShortTermMemory()
        self.history = history if history is not None else ChatHistoryStore(
            db_path=str(self.data_dir / "chat_history.db")
        )
        self.session_id = self.history.start_session()

    def _triage_and_route(
        self, query: str, session_context: str
    ) -> Tuple[bool, Optional[str], str, dict]:
        """
        Decide whether to retrieve sources or answer directly.

        Returns:
            (True, None, search_query, metadata_filter): filtered retrieval.
            (False, answer, query, {}): return the direct answer.
        """
        try:
            response = self._request_route(query, session_context)
            return self._parse_route(response, query, session_context)

        except Exception as exc:
            # Preserve the existing clarification response for all routing failures.
            logger.warning(
                "Routing failed (%s); requesting jurisdiction clarification",
                exc,
            )
            return False, ROUTING_CLARIFICATION, query, {}

    def _request_route(self, query, session_context):
        """Ask the model for a routing decision using recent conversation context."""
        return self.llm_client.chat.completions.create(
            model=_MODELS["main"],
            # Gemini needs room for reasoning as well as the final routing JSON.
            max_tokens=4096 if LLM_PROVIDER == "gemini" else 600,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": TRIAGE_PROMPT,
                },
                {
                    "role": "user",
                    "content": (
                        f"Recent conversation context:\n{session_context}\n\n"
                        f"Current question:\n{query}"
                    ),
                },
            ],
        )

    def _parse_route(self, response, query, session_context):
        """Validate the model response and resolve the direct or retrieval route."""
        if getattr(response.choices[0], "finish_reason", None) == "length":
            raise ValueError("Router output exceeded the token limit")
        raw = response.choices[0].message.content
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("router response must contain text")

        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        if not isinstance(data, dict):
            raise ValueError("router response must be a JSON object")

        needs_retrieval = data.get("retrieval")
        if type(needs_retrieval) is not bool:
            raise ValueError("'retrieval' must be a JSON boolean")

        out_of_scope = data.get("out_of_scope", False)
        if type(out_of_scope) is not bool:
            raise ValueError("'out_of_scope' must be a JSON boolean")
        if out_of_scope:
            if needs_retrieval:
                raise ValueError("Out-of-scope topics cannot request corpus retrieval")
            search_query = data.get("search_query")
            if not isinstance(search_query, str) or not search_query.strip():
                raise ValueError("Out-of-scope routing requires a standalone question")
            answer = self._answer_without_sources(query, search_query.strip(), session_context)
            notice = "**Topic outside the corpus: LLM answer, not verified against RAG sources.**"
            return False, f"{notice}\n\n{answer}", query, {}

        reasoning = data.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = ""

        logger.info(
            "Router retrieval=%s; reason=%s",
            needs_retrieval,
            reasoning,
        )

        if needs_retrieval:
            requested = data.get("requested_countries")
            if not isinstance(requested, list) or not all(
                isinstance(country, str) and country.strip() for country in requested
            ):
                raise ValueError("retrieval requires requested_countries")
            requested = sorted(set(country.strip() for country in requested))
            if not requested:
                return False, COUNTRY_CLARIFICATION, query, {}
            search_query = data.get("search_query")
            if not isinstance(search_query, str) or not search_query.strip():
                raise ValueError("retrieval requires a non-empty search_query")
            if not set(requested).issubset(FILTER_VALUES["country"]):
                return False, self._answer_without_sources(query, search_query, session_context), query, {}
            selection = dict(data.get("filters", {}))
            selection["country"] = requested
            metadata_filter = build_metadata_filter(selection)
            logger.info("Retrieval filter: %s", metadata_filter)
            return True, None, search_query.strip(), metadata_filter

        direct_answer = data.get("direct_answer")
        if not isinstance(direct_answer, str) or not direct_answer.strip():
            raise ValueError(
                "direct route requires a non-empty 'direct_answer'"
            )

        return False, direct_answer.strip(), query, {}

    def _answer_without_sources(self, query, search_query, context):
        """Generate a fallback explicitly labeled as outside the retrieved corpus."""
        return self._complete(
            FALLBACK_PROMPT,
            {"question": query, "standalone_question": search_query,
             "recent_conversation": context},
        )

    def _complete(self, system, payload, max_tokens=None):
        """Send a generation request and reject empty or truncated answers."""
        if max_tokens is None:
            # Reserve room for Gemini reasoning and the final answer.
            max_tokens = 8192 if LLM_PROVIDER == "gemini" else 2000
        response = self.llm_client.chat.completions.create(
            model=_MODELS["main"], temperature=0, max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise ValueError("Answer exceeded the token limit")
        content = choice.message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Model returned no answer")
        return content.strip()

    def _answer_with_rag(self, query, search_query, context, documents, memories):
        """Use retrieved sources as evidence and memories only as background."""
        return self._complete(
            GROUNDED_ANSWER_PROMPT,
            {"original_question": query, "search_query": search_query,
             "recent_conversation": context, "past_qa_background": memories,
             "retrieved_documents": documents},
        )

    def ask(self, query):
        """Complete one turn. Each instance represents one conversation."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        query = query.strip()
        context = self.stm.as_context_string()
        needs_retrieval, answer, search_query, metadata_filter = self._triage_and_route(query, context)
        documents, countries = [], []
        citations_verified = None
        if needs_retrieval:
            documents, countries = self._retrieve_sources(search_query, metadata_filter)
            if documents:
                memories = self._recall_memories(search_query, countries)
                answer = self._answer_with_rag(
                    query, search_query, context, documents, memories
                )
                answer, citations_verified = self._verify_answer(answer, documents)
            else:
                answer = self._answer_without_sources(query, search_query, context)

        return self._save_and_format_result(
            query, answer, needs_retrieval, search_query, metadata_filter,
            documents, countries, citations_verified,
        )

    def _retrieve_sources(self, search_query, metadata_filter):
        """Retrieve reranked sources and require exact requested-country coverage."""
        if self.retriever is None:
            from retrieval import PineconeRetriever
            self.retriever = PineconeRetriever()
        # Service errors propagate instead of being mistaken for no results.
        documents = self.retriever.retrieve(search_query, metadata_filter=metadata_filter)
        countries = sorted({d["country"] for d in documents if d.get("country")})
        conditions = metadata_filter.get("$and", [metadata_filter])
        country_filter = next(condition["country"] for condition in conditions if "country" in condition)
        requested = country_filter.get("$in", [country_filter.get("$eq")])
        if set(countries) != set(requested) or any(not d.get("country") for d in documents):
            documents = []
        return documents, countries

    def _recall_memories(self, search_query, countries):
        """Recall background context; memory failures must not stop generation."""
        if self.ltm is not None:
            try:
                return self.ltm.recall_similar(
                    search_query, agent_id="single_agent", countries_used=countries
                )
            except Exception as exc:
                logger.warning("Long-term recall failed: %s", exc)
        return []

    def _verify_answer(self, answer, documents):
        """Use the guard's final answer, including any correction or notice."""
        from guardrails.output_guard import check_rag_answer

        checked = check_rag_answer(answer, documents, self.llm_client)
        return checked.answer, checked.citations_verified

    def _save_and_format_result(
        self, query, answer, needs_retrieval, search_query, metadata_filter,
        documents, countries, citations_verified,
    ):
        """Update short-term context, save history, then store verified memories."""
        # Keep this compatibility field even when retrieval ends in a fallback.
        agents = ["single_agent"] if needs_retrieval else []
        turn = self.stm.add_turn(query=query, agents_activated=agents, answer=answer)
        self.history.save_turn(
            self.session_id, turn.turn_id, query, answer, agents, documents
        )
        if documents and citations_verified is True and self.ltm is not None:
            try:
                self.ltm.store(search_query, answer, agents, countries_used=countries)
            except Exception as exc:
                logger.warning("Long-term memory save failed: %s", exc)
        # Formatting an evaluation record does not run Ragas scoring.
        from ragas_evaluation import evaluation_record
        return {"answer": answer, "needs_retrieval": needs_retrieval,
                "evaluation": evaluation_record(query, answer, documents),
                "citations_verified": citations_verified,
                "metadata_filter": metadata_filter,
                "search_query": search_query if needs_retrieval else None,
                "retrieved_documents": documents, "session_id": self.session_id,
                "turn_id": turn.turn_id}

    def new_session(self):
        """Start fresh short-term context without deleting stored conversations."""
        self.session_id = self.history.start_session()
        self.stm.reset()
        return self.session_id

    def load_session(self, session_id):
        """Restore a saved conversation and its short-term context."""
        from memory.short_term import Turn
        rows = self.history.load_session(session_id)
        self.stm.load_history([Turn(
            turn_id=row["turn_id"], query=row["query"], answer=row["answer"],
            agents_activated=row["agents_activated"],
        ) for row in rows])
        self.session_id = session_id
        return rows
