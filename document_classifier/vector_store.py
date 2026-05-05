"""Vector Store for the document-classifier pipeline.

Wraps ChromaDB PersistentClient to persist document embeddings and metadata.
A single collection named "documents" is used, created with cosine distance.

None values in metadata are stored as the sentinel string "__none__" because
ChromaDB metadata values must be strings, ints, floats, or booleans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from document_classifier.config import PipelineConfig

# ---------------------------------------------------------------------------
# Sentinel for None metadata values
# ---------------------------------------------------------------------------

_NONE_SENTINEL = "__none__"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DocumentRecord:
    """A single document stored in the Vector Store.

    Attributes:
        id: UUID string uniquely identifying the document.
        text: Extracted text content.
        vector: 1024-dimensional embedding vector.
        source_filename: Original filename of the source document.
        file_format: One of "pdf", "image", or "docx".
        ingestion_timestamp: ISO-8601 UTC timestamp of when the document was ingested.
        cluster_id: HDBSCAN cluster assignment; -1 = outlier; None = not yet clustered.
        label: Human-readable cluster label; None if not yet labelled.
        corrected_label: Operator-provided override label; None if not corrected.
    """

    id: str
    text: str
    vector: list[float]
    source_filename: str
    file_format: str
    ingestion_timestamp: str
    cluster_id: int | None
    label: str | None
    corrected_label: str | None = None


@dataclass
class QueryResult:
    """Result of a nearest-neighbour query against the Vector Store.

    Attributes:
        document_id: ID of the matched document.
        label: Cluster label of the matched document; None if not yet labelled.
        distance: Cosine distance to the query vector (0 = identical, 1 = orthogonal).
                  Confidence score = 1 - distance.
    """

    document_id: str
    label: str | None
    distance: float


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _to_meta(value: int | str | None) -> int | str:
    """Convert a Python value to a ChromaDB-safe metadata value."""
    if value is None:
        return _NONE_SENTINEL
    return value


def _from_meta_str(value: str) -> str | None:
    """Convert a ChromaDB metadata string back to Python, restoring None."""
    if value == _NONE_SENTINEL:
        return None
    return value


def _from_meta_cluster_id(value: int | str) -> int | None:
    """Convert a ChromaDB metadata cluster_id back to Python int or None."""
    if value == _NONE_SENTINEL:
        return None
    return int(value)


def _record_from_chroma(
    doc_id: str,
    text: str,
    metadata: dict,
    embedding: list[float],
) -> DocumentRecord:
    """Reconstruct a DocumentRecord from raw ChromaDB result fields."""
    return DocumentRecord(
        id=doc_id,
        text=text,
        vector=list(embedding),
        source_filename=str(metadata.get("source_filename", "")),
        file_format=str(metadata.get("file_format", "")),
        ingestion_timestamp=str(metadata.get("ingestion_timestamp", "")),
        cluster_id=_from_meta_cluster_id(metadata.get("cluster_id", _NONE_SENTINEL)),
        label=_from_meta_str(str(metadata.get("label", _NONE_SENTINEL))),
        corrected_label=_from_meta_str(
            str(metadata.get("corrected_label", _NONE_SENTINEL))
        ),
    )


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """Wraps a ChromaDB PersistentClient to store and query document embeddings.

    Args:
        config: PipelineConfig instance providing chroma_persist_dir.
    """

    def __init__(self, config: "PipelineConfig") -> None:
        import chromadb

        if config.chroma_persist_dir == ":memory:":
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=config.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
        self._new_doc_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: DocumentRecord) -> None:
        """Add a DocumentRecord to the store.

        The text is stored as the ChromaDB document body; all other fields
        (except vector) are stored as metadata; the vector is stored as the
        embedding.

        Args:
            record: The DocumentRecord to persist.
        """
        metadata = {
            "source_filename": record.source_filename,
            "file_format": record.file_format,
            "ingestion_timestamp": record.ingestion_timestamp,
            "cluster_id": _to_meta(record.cluster_id),
            "label": _to_meta(record.label),
            "corrected_label": _to_meta(record.corrected_label),
        }
        self._collection.add(
            ids=[record.id],
            documents=[record.text],
            embeddings=[record.vector],
            metadatas=[metadata],
        )
        self._new_doc_count += 1

    def get_all_vectors(self) -> list[tuple[str, list[float]]]:
        """Return (id, vector) tuples for every document in the store.

        Returns:
            A list of (document_id, embedding_vector) pairs.
        """
        result = self._collection.get(include=["embeddings"])
        ids = result.get("ids") or []
        embeddings = result.get("embeddings")
        if embeddings is None:
            return []
        return [(doc_id, list(vec)) for doc_id, vec in zip(ids, embeddings)]

    def update_cluster_assignments(self, assignments: dict[str, int]) -> None:
        """Update the cluster_id metadata for each document in *assignments*.

        Args:
            assignments: Mapping of document_id → cluster_id.
        """
        if not assignments:
            return
        ids = list(assignments.keys())
        # Fetch current metadata so we only change cluster_id
        result = self._collection.get(ids=ids, include=["metadatas"])
        existing_meta = {
            doc_id: meta
            for doc_id, meta in zip(result["ids"], result["metadatas"])
        }
        updated_metadatas = []
        for doc_id in ids:
            meta = dict(existing_meta.get(doc_id, {}))
            meta["cluster_id"] = _to_meta(assignments[doc_id])
            updated_metadatas.append(meta)
        self._collection.update(ids=ids, metadatas=updated_metadatas)

    def update_label(self, cluster_id: int, label: str) -> None:
        """Update the label metadata for all documents with the given cluster_id.

        Args:
            cluster_id: The cluster whose documents should be re-labelled.
            label: The new human-readable label.
        """
        result = self._collection.get(
            where={"cluster_id": _to_meta(cluster_id)},
            include=["metadatas"],
        )
        ids = result.get("ids", [])
        if not ids:
            return
        updated_metadatas = []
        for meta in result["metadatas"]:
            updated = dict(meta)
            updated["label"] = label
            updated_metadatas.append(updated)
        self._collection.update(ids=ids, metadatas=updated_metadatas)

    def get_cluster_members(self, cluster_id: int) -> list[DocumentRecord]:
        """Return all DocumentRecords belonging to *cluster_id*.

        Args:
            cluster_id: The cluster to retrieve members for.

        Returns:
            List of DocumentRecord instances.
        """
        result = self._collection.get(
            where={"cluster_id": _to_meta(cluster_id)},
            include=["documents", "embeddings", "metadatas"],
        )
        records = []
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        embeddings = result.get("embeddings")
        if embeddings is None:
            embeddings = [[] for _ in ids]
        metadatas = result.get("metadatas") or []
        for doc_id, text, embedding, metadata in zip(
            ids, documents, embeddings, metadatas
        ):
            records.append(
                _record_from_chroma(doc_id, text, metadata, embedding)
            )
        return records

    def get_cluster_centroids(self) -> dict[int, list[float]]:
        """Compute cluster centroids on-the-fly by averaging member vectors.

        Excludes outliers (cluster_id == -1) and unclustered documents
        (cluster_id == None / "__none__").

        Returns:
            Mapping of cluster_id → centroid vector.
        """
        result = self._collection.get(include=["embeddings", "metadatas"])

        # Group vectors by cluster_id
        cluster_vectors: dict[int, list[list[float]]] = {}
        embeddings = result.get("embeddings")
        if embeddings is None:
            return {}
        metadatas = result.get("metadatas") or []
        for embedding, metadata in zip(embeddings, metadatas):
            raw = metadata.get("cluster_id", _NONE_SENTINEL)
            cid = _from_meta_cluster_id(raw)
            # Skip outliers and unclustered
            if cid is None or cid == -1:
                continue
            cluster_vectors.setdefault(cid, []).append(list(embedding))

        # Compute mean vector for each cluster
        centroids: dict[int, list[float]] = {}
        for cid, vectors in cluster_vectors.items():
            n = len(vectors)
            dim = len(vectors[0])
            centroid = [sum(v[i] for v in vectors) / n for i in range(dim)]
            centroids[cid] = centroid
        return centroids

    def query_nearest(
        self, vector: list[float], n_results: int = 5
    ) -> list[QueryResult]:
        """Query ChromaDB for the *n_results* nearest neighbours.

        Args:
            vector: The query embedding vector.
            n_results: Number of nearest neighbours to return.

        Returns:
            List of QueryResult ordered by ascending cosine distance.
        """
        total = self._collection.count()
        if total == 0:
            return []
        actual_n = min(n_results, total)
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=actual_n,
            include=["metadatas", "distances"],
        )
        query_results = []
        ids_list = result.get("ids", [[]])[0]
        distances_list = result.get("distances", [[]])[0]
        metadatas_list = result.get("metadatas", [[]])[0]
        for doc_id, distance, metadata in zip(
            ids_list, distances_list, metadatas_list
        ):
            label = _from_meta_str(str(metadata.get("label", _NONE_SENTINEL)))
            query_results.append(
                QueryResult(
                    document_id=doc_id,
                    label=label,
                    distance=max(0.0, float(distance)),
                )
            )
        return query_results

    def get_new_doc_count_since_last_cluster(self) -> int:
        """Return the number of documents added since the last cluster reset.

        Returns:
            Count of new documents.
        """
        return self._new_doc_count

    def reset_new_doc_counter(self) -> None:
        """Reset the new-document counter to zero."""
        self._new_doc_count = 0

    def update_corrected_label(self, document_id: str, corrected_label: str) -> None:
        """Update the corrected_label for a specific document.

        Args:
            document_id: The ID of the document to update.
            corrected_label: The operator-provided corrected label to store.
        """
        result = self._collection.get(ids=[document_id], include=["metadatas"])
        if not result["ids"]:
            return
        meta = dict(result["metadatas"][0])
        meta["corrected_label"] = corrected_label
        self._collection.update(ids=[document_id], metadatas=[meta])


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_vector_store(config: "PipelineConfig | None" = None) -> VectorStore:
    """Create and return a VectorStore instance.

    Args:
        config: Optional PipelineConfig. If None, a default PipelineConfig is used.

    Returns:
        A configured VectorStore instance.
    """
    if config is None:
        from document_classifier.config import PipelineConfig

        config = PipelineConfig()
    return VectorStore(config)
