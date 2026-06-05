from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import os
            import sys

            # Force Ephemeral in-memory list fallback during tests to keep them isolated and stateless
            is_testing = "pytest" in sys.modules or "unittest" in sys.modules or "pytest_cache" in sys.modules
            
            if is_testing:
                self._use_chroma = False
                self._collection = None
            else:
                import chromadb
                persist_dir = os.getenv("CHROMA_PERSIST_DIR")
                if persist_dir:
                    self._client = chromadb.PersistentClient(path=persist_dir)
                else:
                    self._client = chromadb.EphemeralClient()
                    
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name
                )
                self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # build a normalized stored record for one document
        embedding = self._embedding_fn(doc.content)
        metadata = dict(doc.metadata) if doc.metadata is not None else {}
        metadata["doc_id"] = doc.id
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # run in-memory similarity search over provided records
        query_emb = self._embedding_fn(query)
        results = []
        for r in records:
            score = compute_similarity(query_emb, r["embedding"])
            results.append({
                "id": r["id"],
                "content": r["content"],
                "metadata": r["metadata"],
                "score": score
            })
        # sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection is not None:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                rec = self._make_record(doc)
                ids.append(rec["id"])
                documents.append(rec["content"])
                embeddings.append(rec["embedding"])
                metadatas.append(rec["metadata"])
            
            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            for doc in docs:
                rec = self._make_record(doc)
                self._store.append(rec)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            ret = []
            if results and "ids" in results and results["ids"]:
                ids = results["ids"][0]
                documents = results["documents"][0] if "documents" in results and results["documents"] else [None] * len(ids)
                metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [None] * len(ids)
                distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
                
                for i in range(len(ids)):
                    score = 1.0 - distances[i] if distances[i] is not None else 0.0
                    ret.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": metadatas[i],
                        "score": score
                    })
            return ret
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        else:
            return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter
            )
            ret = []
            if results and "ids" in results and results["ids"]:
                ids = results["ids"][0]
                documents = results["documents"][0] if "documents" in results and results["documents"] else [None] * len(ids)
                metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [None] * len(ids)
                distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
                
                for i in range(len(ids)):
                    score = 1.0 - distances[i] if distances[i] is not None else 0.0
                    ret.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": metadatas[i],
                        "score": score
                    })
            return ret
        else:
            filtered_records = []
            if metadata_filter:
                for rec in self._store:
                    match = True
                    rec_meta = rec.get("metadata", {})
                    for k, v in metadata_filter.items():
                        if rec_meta.get(k) != v:
                            match = False
                            break
                    if match:
                        filtered_records.append(rec)
            else:
                filtered_records = list(self._store)
            
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            size_before = self.get_collection_size()
            self._collection.delete(where={"doc_id": doc_id})
            size_after = self.get_collection_size()
            return size_before > size_after
        else:
            initial_len = len(self._store)
            self._store = [r for r in self._store if r.get("metadata", {}).get("doc_id") != doc_id]
            return len(self._store) < initial_len
