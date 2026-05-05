"""Clustering engine for the document-classifier pipeline.

Wraps HDBSCAN to group document embeddings into clusters without requiring
a pre-specified cluster count.  Outlier documents (HDBSCAN label -1) are
preserved as-is rather than being forced into any cluster.

Usage::

    from document_classifier.clustering import run_clustering

    result = run_clustering(vectors, min_cluster_size=3, min_samples=1)
    print(result.assignments)   # {doc_id: cluster_id, ...}
    print(result.n_clusters)    # number of distinct non-outlier clusters
    print(result.n_outliers)    # number of documents assigned to -1
"""

from __future__ import annotations

from dataclasses import dataclass

import hdbscan
import numpy as np

from document_classifier.logger import log_stage_error


@dataclass
class ClusteringResult:
    """Result of a single clustering run.

    Attributes:
        assignments: Mapping of document_id → cluster_id.  A cluster_id of -1
            indicates that the document is an outlier (not part of any dense cluster).
        n_clusters: Number of distinct clusters found, excluding the outlier group.
        n_outliers: Number of documents assigned to the outlier group (cluster_id -1).
        previous_assignments: The assignments dict from the previous clustering run,
            or None if this is the first run.  Included so the orchestrator can
            compute membership-change percentages for the feedback loop.
    """

    assignments: dict[str, int]
    n_clusters: int
    n_outliers: int
    previous_assignments: dict[str, int] | None


def run_clustering(
    vectors: list[tuple[str, list[float]]],
    min_cluster_size: int = 3,
    min_samples: int = 1,
    previous_assignments: dict[str, int] | None = None,
) -> ClusteringResult:
    """Cluster document embeddings using HDBSCAN.

    Args:
        vectors: A list of ``(document_id, vector)`` tuples.  Each vector must
            have the same dimensionality.
        min_cluster_size: Minimum number of documents required to form a cluster.
            Passed directly to HDBSCAN.
        min_samples: Controls noise sensitivity.  Lower values produce fewer
            outliers.  Passed directly to HDBSCAN.
        previous_assignments: Cluster assignments from the previous run, used by
            the orchestrator to detect membership changes.  Stored in the result
            unchanged.

    Returns:
        A :class:`ClusteringResult` containing the new assignments, cluster
        counts, and the previous assignments for change detection.

    Notes:
        - Documents assigned to cluster ``-1`` by HDBSCAN are outliers and are
          preserved as-is; they are NOT forced into any named cluster.
        - If fewer than 2 clusters are found from a corpus of 10 or more
          documents, a warning is logged via :func:`~document_classifier.logger.log_stage_error`.
    """
    if not vectors:
        return ClusteringResult(
            assignments={},
            n_clusters=0,
            n_outliers=0,
            previous_assignments=previous_assignments,
        )

    doc_ids = [doc_id for doc_id, _ in vectors]
    matrix = np.array([vec for _, vec in vectors], dtype=np.float64)

    # Compute cosine distance matrix manually — avoids hdbscan/sklearn
    # metric compatibility issues across versions.
    # Normalise rows then use euclidean distance on unit vectors
    # (euclidean on unit vectors == sqrt(2*(1-cosine_sim)), monotone with cosine distance).
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms == 0, 1.0, norms)
    normed = matrix / norms

    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels: np.ndarray = clusterer.fit_predict(normed)

    assignments: dict[str, int] = {
        doc_id: int(label) for doc_id, label in zip(doc_ids, labels)
    }

    unique_labels = set(labels.tolist())
    n_clusters = len(unique_labels - {-1})
    n_outliers = int((labels == -1).sum())

    corpus_size = len(vectors)
    if n_clusters < 2 and corpus_size >= 10:
        log_stage_error(
            stage="clustering",
            document_id="",
            error_type="low_cluster_count",
            message=(
                f"Clustering produced {n_clusters} cluster(s) from a corpus of "
                f"{corpus_size} documents. The corpus may be too homogeneous or "
                "too small for meaningful clustering."
            ),
        )

    return ClusteringResult(
        assignments=assignments,
        n_clusters=n_clusters,
        n_outliers=n_outliers,
        previous_assignments=previous_assignments,
    )
