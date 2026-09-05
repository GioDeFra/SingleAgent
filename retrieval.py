"""Read the existing multi-agent corpus without re-ingesting it."""

import os


class PineconeRetriever:
    def __init__(self, index_name=None, namespace=None, top_k=8,
                 metadata_filter=None, index=None, embed_model=None):
        if top_k < 1:
            raise ValueError("top_k must be positive")
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

    def retrieve(self, query):
        vector = self.embed_model.encode([query], normalize_embeddings=True)[0].tolist()
        kwargs = dict(vector=vector, namespace=self.namespace, top_k=self.top_k,
                      include_metadata=True)
        if self.metadata_filter:
            kwargs["filter"] = self.metadata_filter
        result = self.index.query(**kwargs)
        documents = []
        for match in result.get("matches", []):
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
        return [d for d in documents if len(sources[d["citation_label"]]) == 1]
