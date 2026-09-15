"""Read the existing multi-agent corpus without re-ingesting it."""

import os
import math


def _scoring_text(document):
    """Use the same metadata enrichment as the multi-agent reranker."""
    skip = {"CASE_ID", "citation_label", "country", "doc_type", "law", "source",
            "text", "chunk_index", "n_chunks"}
    placeholders = {"", "no data", "not specified", "n/a", "unknown"}
    parts = []
    for key, value in document["metadata"].items():
        if key in skip or value in (None, [], {}):
            continue
        if isinstance(value, str) and value.strip().lower() in placeholders:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        parts.append(f"{key}: {value}")
    return "; ".join(parts) + "\n" + document["text"] if parts else document["text"]


FILTER_VALUES = {
    "country": {"Italy", "Estonia", "Slovenia"},
    "law": {"Divorce", "Inheritance"},
    "doc_type": {"Legal Cases", "Civil Codes"},
}


def build_metadata_filter(selection):
    """Convert validated router selections to Pinecone conditions."""
    if not isinstance(selection, dict) or set(selection) - set(FILTER_VALUES):
        raise ValueError("Invalid retrieval filter fields")
    conditions = []
    for field, allowed in FILTER_VALUES.items():
        values = selection.get(field, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value in allowed for value in values
        ):
            raise ValueError(f"Invalid values for retrieval filter {field}")
        values = list(dict.fromkeys(values))
        if len(values) == 1:
            conditions.append({field: {"$eq": values[0]}})
        elif values:
            conditions.append({field: {"$in": values}})
    if not conditions:
        return {}
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class PineconeRetriever:
    def __init__(self, index_name=None, namespace=None, top_k=5,
                 metadata_filter=None, index=None, embed_model=None,
                 n_retrieve=20, reranker=None):
        if not (1 <= top_k <= n_retrieve):
            raise ValueError("Require 1 <= top_k <= n_retrieve")
        if index is None:
            from pinecone import Pinecone
            key = os.getenv("PINECONE_API_KEY")
            if not key:
                raise RuntimeError("Add PINECONE_API_KEY to Apikey.env")
            index = Pinecone(api_key=key).Index(
                index_name or os.getenv("PINECONE_INDEX_NAME", "legal-rag")
            )
        if embed_model is None:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer("BAAI/bge-m3")
        self.index = index
        self.embed_model = embed_model
        self.namespace = namespace if namespace is not None else os.getenv("PINECONE_NAMESPACE", "")
        self.top_k = top_k
        self.metadata_filter = metadata_filter
        self.n_retrieve = n_retrieve
        self.reranker = reranker

    def retrieve(self, query, metadata_filter=None):
        """Embed the query, combine filters, validate candidates, and rerank."""
        vector = self.embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        kwargs = dict(vector=vector, namespace=self.namespace, top_k=self.n_retrieve,
                      include_metadata=True)
        filters = [f for f in (self.metadata_filter, metadata_filter) if f]
        if filters:
            kwargs["filter"] = filters[0] if len(filters) == 1 else {"$and": filters}
        result = self.index.query(**kwargs)
        documents = self._prepare_documents(result.get("matches", []))
        return self._rerank(query, documents)

    @staticmethod
    def _prepare_documents(matches):
        """Keep usable text and citation labels with unambiguous source identity."""
        documents = []
        for match in matches:
            meta = match.get("metadata", {}) or {}
            text, label = meta.get("text"), meta.get("citation_label")
            if not isinstance(text, str) or not text.strip():
                continue
            if not isinstance(label, str) or not label.strip():
                continue
            documents.append({
                "id": match.get("id", ""), "text": text,
                "citation_label": label, "source": str(meta.get("source") or ""),
                "country": str(meta.get("country") or ""),
                "metadata": meta, "score": float(match.get("score") or 0),
            })
        # Preserve ingestion labels and exclude labels shared by different sources.
        sources = {}
        for document in documents:
            sources.setdefault(document["citation_label"], set()).add(document["source"])
        documents = [d for d in documents if len(sources[d["citation_label"]]) == 1]
        return documents

    def _rerank(self, query, documents):
        """Score valid candidates and retain the highest-ranked passages."""
        if not documents:
            return []
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
        scores = self.reranker.predict([(query, _scoring_text(d)) for d in documents])
        if len(scores) != len(documents):
            raise ValueError("Reranker returned an unexpected number of scores")
        for document, score in zip(documents, scores):
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("Reranker returned a non-finite score")
            document["rerank_score"] = score
        return sorted(documents, key=lambda d: d["rerank_score"], reverse=True)[:self.top_k]
