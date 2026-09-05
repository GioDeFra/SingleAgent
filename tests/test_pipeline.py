"""Offline behavioral tests: no provider keys, model downloads, or network."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
fake_config = types.ModuleType("llm_client")
fake_config.get_llm_client = Mock()
fake_config.model_names = lambda: {"main": "test", "light": "test"}
with patch.dict(sys.modules, {"llm_client": fake_config}):
    spec = importlib.util.spec_from_file_location("agent_under_test", ROOT / "agent.py")
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)

from retrieval import PineconeRetriever


def completion(content, finish_reason="stop"):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content=content), finish_reason=finish_reason
    )])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = Mock()
        fake_config.get_llm_client.return_value = self.client
        self.retriever = Mock()
        self.ltm = Mock()
        self.ltm.recall_similar.return_value = []
        self.rag = agent.SingleAgentRAG(
            retriever=self.retriever, long_term=self.ltm, data_dir=self.temp.name
        )

    def test_direct_answer_saved_without_retrieval_or_long_term(self):
        self.client.chat.completions.create.return_value = completion(json.dumps(
            {"retrieval": False, "direct_answer": "Hello!"}
        ))
        result = self.rag.ask("Hello")
        self.assertEqual(result["answer"], "Hello!")
        self.retriever.retrieve.assert_not_called()
        self.ltm.store.assert_not_called()
        self.assertEqual(self.rag.history.load_session(result["session_id"])[0]["query"], "Hello")

    def test_followup_rewrite_retrieval_original_question_and_session_restore(self):
        self.rag.stm.add_turn("Divorce in Italy?", [], "Earlier answer")
        rewrite = "What are the divorce rules in France?"
        self.client.chat.completions.create.side_effect = [
            completion(json.dumps({"retrieval": True, "search_query": rewrite})),
            completion("Supported answer [FR-1]"),
        ]
        self.retriever.retrieve.return_value = [
            {"text": "Evidence", "citation_label": "FR-1", "country": "France"}
        ]
        result = self.rag.ask("And in France?")
        self.retriever.retrieve.assert_called_once_with(rewrite)
        payload = json.loads(self.client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
        self.assertEqual(payload["original_question"], "And in France?")
        self.assertIn("Italy", payload["recent_conversation"])
        self.ltm.recall_similar.assert_called_once_with(
            rewrite, agent_id="single_agent", countries_used=["France"]
        )
        self.ltm.store.assert_called_once_with(
            rewrite, result["answer"], ["single_agent"], countries_used=["France"]
        )
        self.rag.new_session()
        self.assertEqual(len(self.rag.stm), 0)
        self.rag.load_session(result["session_id"])
        self.assertIn("And in France?", self.rag.stm.as_context_string())
        self.assertEqual(self.rag.stm.add_turn("Next", [], "Answer").turn_id, 2)

    def test_invalid_router_and_truncation_fall_back(self):
        for response in [completion("not json"), completion('{"retrieval": "true"}'),
                         completion('{"retrieval": true}'), completion("{}", "length")]:
            with self.subTest(response=response):
                self.client.chat.completions.create.return_value = response
                self.assertEqual(self.rag._triage_and_route("Original", ""), (True, None, "Original"))

    def test_empty_retrieval_not_stored_in_long_term(self):
        self.client.chat.completions.create.side_effect = [
            completion('{"retrieval": true, "search_query": "Legal question"}'),
            completion("The available sources do not support an answer."),
        ]
        self.retriever.retrieve.return_value = []
        result = self.rag.ask("Legal question")
        self.assertEqual(result["retrieved_documents"], [])
        self.ltm.store.assert_not_called()
        self.assertIn("No citable documents", self.client.chat.completions.create.call_args.kwargs["messages"][0]["content"])

    def test_service_failure_is_not_empty_retrieval(self):
        self.client.chat.completions.create.return_value = completion(
            '{"retrieval": true, "search_query": "Question"}'
        )
        self.retriever.retrieve.side_effect = RuntimeError("Service unavailable")
        with self.assertRaisesRegex(RuntimeError, "Service unavailable"):
            self.rag.ask("Question")
        self.assertEqual(len(self.rag.stm), 0)

    def test_memory_failure_does_not_lose_answer(self):
        self.client.chat.completions.create.side_effect = [
            completion('{"retrieval": true, "search_query": "Question"}'), completion("Answer [A]")
        ]
        self.retriever.retrieve.return_value = [{"country": "Italy", "citation_label": "A", "text": "Source"}]
        self.ltm.recall_similar.side_effect = RuntimeError("Memory down")
        self.ltm.store.side_effect = RuntimeError("Memory down")
        self.assertEqual(self.rag.ask("Question")["answer"], "Answer [A]")


class RetrievalTests(unittest.TestCase):
    def test_embedding_namespace_filter_and_citation_integrity(self):
        model, index = Mock(), Mock()
        vector = Mock()
        vector.tolist.return_value = [0.1, 0.2]
        model.encode.return_value = [vector]
        index.query.return_value = {"matches": [
            {"id": "1", "metadata": {"text": "Valid", "citation_label": "A", "source": "one"}},
            {"id": "2", "metadata": {"text": "No label"}},
            {"id": "3", "metadata": {"text": "Conflict", "citation_label": "B", "source": "two"}},
            {"id": "4", "metadata": {"text": "Conflict", "citation_label": "B", "source": "three"}},
        ]}
        retriever = PineconeRetriever(index=index, embed_model=model,
                                      namespace="existing", metadata_filter={"country": "Italy"})
        documents = retriever.retrieve("Question")
        self.assertEqual([d["id"] for d in documents], ["1"])
        model.encode.assert_called_once_with(["Question"], normalize_embeddings=True)
        index.query.assert_called_once_with(vector=[0.1, 0.2], namespace="existing", top_k=8,
                                            include_metadata=True, filter={"country": "Italy"})


class LongTermScopeTests(unittest.TestCase):
    def test_recall_and_dedup_require_country_scope(self):
        import threading
        fake_chroma = types.ModuleType("chromadb")
        fake_utils = types.ModuleType("chromadb.utils")
        fake_utils.embedding_functions = Mock()
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = Mock()
        with patch.dict(sys.modules, {
            "llm_client": fake_config, "chromadb": fake_chroma,
            "chromadb.utils": fake_utils, "dotenv": fake_dotenv,
        }):
            spec = importlib.util.spec_from_file_location("ltm_under_test", ROOT / "memory/long_term.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        ltm = module.LongTermMemory.__new__(module.LongTermMemory)
        ltm._db_lock = threading.Lock()
        ltm._qa_col = Mock()
        ltm._qa_col.count.return_value = 1
        ltm.qa_threshold = 0.25
        ltm.dedup_threshold = 0.05
        ltm._qa_col.query.return_value = {
            "ids": [["old"]], "documents": [["Q"]], "distances": [[0.01]],
            "metadatas": [[{"answer": "A", "country_scope": "France"}]],
        }
        ltm.recall_similar("Q", agent_id="single_agent", countries_used=["France"])
        self.assertEqual(ltm._qa_col.query.call_args.kwargs["where"],
                         {"$and": [{"agent_single_agent": True}, {"country_scope": "France"}]})
        self.assertEqual(ltm._find_duplicate_locked("Q", "single_agent", "Italy"), "old")
        self.assertEqual(ltm._qa_col.query.call_args.kwargs["where"],
                         {"$and": [{"agent_scope": "single_agent"}, {"country_scope": "Italy"}]})


if __name__ == "__main__":
    unittest.main()
