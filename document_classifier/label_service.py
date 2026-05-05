"""Label Service for the document-classifier pipeline.

Provides manual labelling of document clusters.  An operator calls
``assign_label()`` to attach a human-readable label to a cluster; the label
is persisted in the Vector Store and used during classification.

There is no automatic LLM-based labelling.  Labels are always supplied by a
human operator.

Usage::

    from document_classifier.label_service import assign_label, select_representatives

    # Pick representative texts to review before deciding on a label
    texts = select_representatives(vector_store.get_cluster_members(cluster_id))
    print(texts)

    # Assign the label you choose
    assign_label(cluster_id=0, label="GST Invoice", vector_store=vector_store)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from document_classifier import logger

if TYPE_CHECKING:
    from document_classifier.vector_store import DocumentRecord, VectorStore


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LabelResult:
    """Result of a manual cluster labelling operation.

    Attributes:
        cluster_id: The cluster that was labelled.
        label: The label string that was stored.
        success: Always ``True`` for manual labelling; kept for API compatibility.
        error: Always ``None`` for manual labelling.
    """

    cluster_id: int
    label: str
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_representatives(
    records: "list[DocumentRecord]",
    n: int = 5,
) -> list[str]:
    """Pick up to *n* representative text samples from a cluster.

    Useful for reviewing cluster contents before deciding on a label.

    Args:
        records: All DocumentRecord instances in the cluster.
        n: Maximum number of representative texts to return (default 5).

    Returns:
        A list of up to *n* text strings from the first *n* records.
        Returns an empty list if *records* is empty.
    """
    return [r.text for r in records[:n]]


def assign_label(
    cluster_id: int,
    label: str,
    vector_store: "VectorStore",
) -> LabelResult:
    """Assign a human-provided label to a cluster and persist it in the Vector Store.

    Args:
        cluster_id: The cluster to label.
        label: The human-readable label to assign (e.g. ``"GST Invoice"``).
        vector_store: VectorStore instance used to persist the label.

    Returns:
        A :class:`LabelResult` confirming the label was stored.
    """
    vector_store.update_label(cluster_id, label)

    logger.log_stage_complete(
        stage="labelling",
        document_id=f"cluster_{cluster_id}",
        duration_ms=0.0,
        metadata={"cluster_id": cluster_id, "label": label},
    )

    return LabelResult(cluster_id=cluster_id, label=label, success=True, error=None)
