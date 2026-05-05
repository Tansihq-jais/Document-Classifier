"""Orchestrator for the document-classifier pipeline.

Wires together ingestion, embedding, clustering, and classification into a
single ``Pipeline`` class.

Labelling is **manual**: after documents are ingested and clustered, call
``pipeline.label_cluster(cluster_id, label)`` to assign a human-readable label
to each cluster.  Use ``pipeline.get_cluster_members(cluster_id)`` or
``pipeline.list_clusters()`` to inspect what ended up in each cluster before
deciding on a label.

The feedback loop counter is managed here: re-clustering is triggered
automatically after every ``config.recluster_threshold`` (default 20) new
documents.

Usage::

    from document_classifier.pipeline import Pipeline
    from document_classifier.config import PipelineConfig

    pipeline = Pipeline()

    # Ingest training documents
    for path in training_files:
        pipeline.ingest(path)

    # Inspect clusters and assign labels manually
    for cluster_id in pipeline.list_clusters():
        samples = pipeline.get_cluster_samples(cluster_id)
        print(f"Cluster {cluster_id}:", samples)
        pipeline.label_cluster(cluster_id, input("Label: "))

    # Classify a new document
    result = pipeline.classify("path/to/new_document.pdf")

    # Correct a misclassification
    pipeline.apply_correction(record.id, "Corrected Label")
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from document_classifier import clustering as clustering_module
from document_classifier import embedding as embedding_module
from document_classifier import ingestion as ingestion_module
from document_classifier import label_service as label_service_module
from document_classifier.classifier import ClassificationError, ClassificationResult
from document_classifier.classifier import classify as _classify
from document_classifier.config import PipelineConfig
from document_classifier.ingestion import IngestionError
from document_classifier.label_service import LabelResult
from document_classifier.vector_store import DocumentRecord, VectorStore


class Pipeline:
    """End-to-end document intelligence pipeline.

    Orchestrates ingestion, embedding, storage, clustering, and classification.
    Labelling is manual: call ``label_cluster()`` after clustering to assign
    human-readable labels to each cluster.

    A feedback loop re-clusters the corpus every time
    ``config.recluster_threshold`` (default 20) new documents have been
    ingested.

    Args:
        config: Optional PipelineConfig.  A default PipelineConfig is used if None.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config if config is not None else PipelineConfig()
        self._vector_store = VectorStore(self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, file_path: str) -> DocumentRecord | IngestionError:
        """Ingest a document file into the pipeline.

        Steps:
        1. Extract text via ``ingestion.extract()``.
        2. On ``IngestionError``, return it immediately.
        3. Embed the extracted text via ``embedding.embed()``.
        4. Build a ``DocumentRecord`` with a UUID4 id and ISO-8601 timestamp.
        5. Add the record to the Vector Store.
        6. Trigger re-clustering if the threshold has been reached.
        7. Return the ``DocumentRecord``.

        Args:
            file_path: Path to the document file to ingest.

        Returns:
            The stored ``DocumentRecord`` on success, or an ``IngestionError``
            if extraction fails.
        """
        # Step 1: Extract text
        extraction = ingestion_module.extract(file_path)
        if isinstance(extraction, IngestionError):
            return extraction

        # Step 2: Embed text
        document_id = str(uuid.uuid4())
        embedding_result = embedding_module.embed(extraction.text, document_id)

        # Step 3: Build DocumentRecord
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        record = DocumentRecord(
            id=document_id,
            text=extraction.text,
            vector=embedding_result.vector,
            source_filename=extraction.source_file,
            file_format=extraction.file_format,
            ingestion_timestamp=timestamp,
            cluster_id=None,
            label=None,
            corrected_label=None,
        )

        # Step 4: Persist to Vector Store
        self._vector_store.add(record)

        # Step 5: Maybe trigger re-clustering
        self._maybe_trigger_recluster()

        return record

    def classify(
        self, file_path: str
    ) -> ClassificationResult | ClassificationError:
        """Classify a document by nearest-centroid lookup.

        Delegates to ``classifier.classify()`` using the pipeline's Vector Store
        and config.

        Args:
            file_path: Path to the document file to classify.

        Returns:
            A ``ClassificationResult`` on success, or a ``ClassificationError``
            on failure.
        """
        return _classify(file_path, self._vector_store, self._config)

    def label_cluster(self, cluster_id: int, label: str) -> LabelResult:
        """Manually assign a human-readable label to a cluster.

        Call this after ingesting and clustering documents.  Use
        ``list_clusters()`` and ``get_cluster_samples()`` to inspect cluster
        contents before deciding on a label.

        Args:
            cluster_id: The cluster to label.
            label: Human-readable label (e.g. ``"GST Invoice"``).

        Returns:
            A :class:`~document_classifier.label_service.LabelResult` confirming
            the label was stored.
        """
        return label_service_module.assign_label(
            cluster_id=cluster_id,
            label=label,
            vector_store=self._vector_store,
        )

    def list_clusters(self) -> list[int]:
        """Return the IDs of all non-outlier clusters currently in the store.

        Returns:
            Sorted list of cluster IDs (excludes -1 outliers and unclustered docs).
        """
        centroids = self._vector_store.get_cluster_centroids()
        return sorted(centroids.keys())

    def get_cluster_samples(self, cluster_id: int, n: int = 5) -> list[str]:
        """Return up to *n* representative text samples from a cluster.

        Useful for reviewing cluster contents before assigning a label.

        Args:
            cluster_id: The cluster to sample from.
            n: Maximum number of text samples to return (default 5).

        Returns:
            List of up to *n* text strings.
        """
        members = self._vector_store.get_cluster_members(cluster_id)
        return label_service_module.select_representatives(members, n=n)

    def classify_batch(
        self, file_paths: list[str]
    ) -> list[ClassificationResult | ClassificationError]:
        """Classify multiple documents in one call.

        Args:
            file_paths: List of paths to document files.

        Returns:
            List of results in the same order as *file_paths*.  Each entry is
            either a :class:`~document_classifier.classifier.ClassificationResult`
            or a :class:`~document_classifier.classifier.ClassificationError`.
        """
        return [_classify(fp, self._vector_store, self._config) for fp in file_paths]

    def get_cluster_label(self, cluster_id: int) -> str | None:
        """Return the current label for a cluster, or None if unlabelled.

        Args:
            cluster_id: The cluster to look up.

        Returns:
            The label string, or ``None`` if the cluster has no label yet.
        """
        members = self._vector_store.get_cluster_members(cluster_id)
        for member in members:
            if member.label is not None:
                return member.label
        return None

    def export_labels(self) -> dict[int, str]:
        """Export all cluster labels as a plain dict.

        Returns:
            Mapping of ``cluster_id → label`` for every labelled cluster.
            Unlabelled clusters are omitted.
        """
        result: dict[int, str] = {}
        for cluster_id in self.list_clusters():
            label = self.get_cluster_label(cluster_id)
            if label is not None:
                result[cluster_id] = label
        return result

    def import_labels(self, labels: dict[int, str]) -> None:
        """Import cluster labels from a dict, overwriting any existing labels.

        Args:
            labels: Mapping of ``cluster_id → label``.
        """
        for cluster_id, label in labels.items():
            label_service_module.assign_label(
                cluster_id=cluster_id,
                label=label,
                vector_store=self._vector_store,
            )

    def apply_correction(self, document_id: str, corrected_label: str) -> None:
        """Store an operator-provided corrected label for a document.

        The corrected label is persisted in the Vector Store and will be used
        as a seed label in the next clustering run.

        Args:
            document_id: The ID of the document to correct.
            corrected_label: The operator-provided label override.
        """
        self._vector_store.update_corrected_label(document_id, corrected_label)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_trigger_recluster(self) -> None:
        """Trigger re-clustering if the new-document threshold has been reached.

        Checks ``vector_store.get_new_doc_count_since_last_cluster()``.  If the
        count is >= ``config.recluster_threshold`` (default 20):

        1. Retrieve all vectors from the Vector Store.
        2. Run HDBSCAN clustering via ``clustering.run_clustering()``.
        3. Update cluster assignments in the Vector Store.
        4. Reset the new-document counter.

        Note: Labels are NOT automatically reassigned after re-clustering.
        Call ``label_cluster()`` to label any new or changed clusters.
        """
        count = self._vector_store.get_new_doc_count_since_last_cluster()
        if count < self._config.recluster_threshold:
            return

        # Step 1: Get all vectors
        all_vectors = self._vector_store.get_all_vectors()
        if not all_vectors:
            self._vector_store.reset_new_doc_counter()
            return

        # Step 2: Run clustering
        clustering_result = clustering_module.run_clustering(
            vectors=all_vectors,
            min_cluster_size=self._config.hdbscan_min_cluster_size,
            min_samples=self._config.hdbscan_min_samples,
            previous_assignments=getattr(self, "_previous_assignments", None),
        )

        new_assignments = clustering_result.assignments

        # Step 3: Update cluster assignments in the Vector Store
        self._vector_store.update_cluster_assignments(new_assignments)

        # Step 4: Store new assignments for the next run
        self._previous_assignments = new_assignments

        # Step 5: Reset counter
        self._vector_store.reset_new_doc_counter()
