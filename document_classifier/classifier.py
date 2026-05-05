"""Classifier module for the document-classifier pipeline.

Classifies a new document by:
1. Extracting text via ingestion.extract()
2. Checking rule-based keyword matches (fuzzy matching with 85% threshold)
3. If no rule matches, embedding via embedding.embed()
4. Querying the Vector Store for nearest neighbours
5. Finding the cluster centroid with the smallest cosine distance
6. Returning a ClassificationResult with label, confidence_score, and low_confidence flag

Returns ClassificationError on failure (no trained clusters, or embedding/ingestion failure).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

import numpy as np

from document_classifier import embedding as embedding_module
from document_classifier import ingestion as ingestion_module
from document_classifier.config import PipelineConfig
from document_classifier.ingestion import IngestionError

if TYPE_CHECKING:
    from document_classifier.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

_config: PipelineConfig = PipelineConfig()


def configure_classifier(config: PipelineConfig) -> None:
    """Override the module-level PipelineConfig.

    Args:
        config: A PipelineConfig instance to use for all subsequent classify() calls.
    """
    global _config
    _config = config


# ---------------------------------------------------------------------------
# Rule-based classification
# ---------------------------------------------------------------------------

# Define keyword rules for each document type
# Each rule has a label, list of keyword sets, and fuzzy matching threshold
CLASSIFICATION_RULES = [
    {
        "label": "Aadhaar Card",
        "keyword_sets": [
            ["Unique Identification Authority of India"],
            ["UIDAI"],
            ["आधार", "Government of India"],
            ["Your Aadhaar No"],  # Physical card marker
            ["आधार", "आम आदमी का अधिकार"],  # Hindi slogan
            ["मेरा आधार", "मेरी पहचान"],  # Another Hindi slogan
        ],
        "fuzzy_threshold": 0.85,
    },
    {
        "label": "PAN Card",
        "keyword_sets": [
            ["Income Tax Department", "Permanent Account"],
            ["आयकर विभाग", "PAN"],
            ["INCOME TAX DEPARTMENT", "Permanent Account Number"],
        ],
        "fuzzy_threshold": 0.85,
    },
    {
        "label": "Indian Passport",
        "keyword_sets": [
            ["Republic of India", "Passport"],
            ["भारत गणराज्य", "Passport"],
            ["REPUBLIC OF INDIA", "Passport"],
            ["Passport", "INDIAN"],  # Fallback for garbled OCR
            ["Passport", "IND"],  # MRZ code pattern
            ["P<IND"],  # Machine Readable Zone (MRZ) pattern
        ],
        "fuzzy_threshold": 0.85,
    },
]


def _fuzzy_contains(text: str, keyword: str, threshold: float = 0.85) -> bool:
    """Check if keyword appears in text with fuzzy matching to handle OCR errors.

    Uses sliding window approach to find the best match for the keyword in the text.

    Args:
        text: The text to search in.
        keyword: The keyword to search for.
        threshold: Minimum similarity ratio (0.0-1.0) to consider a match.

    Returns:
        True if keyword is found with similarity >= threshold, False otherwise.
    """
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    keyword_words = keyword_lower.split()
    text_words = text_lower.split()

    # Try exact match first (fast path)
    if keyword_lower in text_lower:
        return True

    # Sliding window fuzzy match
    for i in range(len(text_words) - len(keyword_words) + 1):
        window = " ".join(text_words[i : i + len(keyword_words)])
        similarity = SequenceMatcher(None, window, keyword_lower).ratio()
        if similarity >= threshold:
            return True

    return False


def _check_rules(text: str) -> tuple[str | None, list[str]]:
    """Check if text matches any classification rules.

    Args:
        text: Extracted document text to check against rules.

    Returns:
        Tuple of (label, matched_keywords) if a rule matches, (None, []) otherwise.
    """
    for rule in CLASSIFICATION_RULES:
        label = rule["label"]
        threshold = rule["fuzzy_threshold"]

        # Check each keyword set - if ANY set matches completely, return that label
        for keyword_set in rule["keyword_sets"]:
            # ALL keywords in the set must be present (with fuzzy matching)
            if all(_fuzzy_contains(text, kw, threshold) for kw in keyword_set):
                return label, keyword_set

    return None, []


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Successful classification result.

    Attributes:
        label: Human-readable cluster label assigned to the document.
        confidence_score: Value in [0.0, 1.0]; computed as 1 - cosine_distance
            between the document embedding and the nearest cluster centroid.
        low_confidence: True if confidence_score < low_confidence_threshold (default 0.5).
        document_id: Identifier used during embedding (basename of file_path).
    """

    label: str
    confidence_score: float
    low_confidence: bool
    document_id: str


@dataclass
class ClassificationError:
    """Structured error returned when classification cannot be completed.

    Attributes:
        error_type: One of "no_trained_clusters" | "embedding_failed".
        message: Human-readable description of the failure.
    """

    error_type: str
    message: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify(
    file_path: str,
    vector_store: "VectorStore",
    config: PipelineConfig | None = None,
) -> ClassificationResult | ClassificationError:
    """Classify a document by nearest-centroid lookup in the Vector Store.

    Algorithm:
    1. Extract text via ingestion.extract(). On IngestionError, return
       ClassificationError(error_type="embedding_failed").
    2. Embed via embedding.embed(). On any exception, return
       ClassificationError(error_type="embedding_failed").
    3. Query vector_store.query_nearest(vector, n_results=5) for 5 neighbours.
    4. Get vector_store.get_cluster_centroids(). If empty, return
       ClassificationError(error_type="no_trained_clusters").
    5. Find the cluster with the smallest cosine distance to the document embedding.
    6. confidence_score = 1 - nearest_centroid_distance (clamped to [0.0, 1.0]).
    7. low_confidence = confidence_score < threshold (default 0.5).
    8. Return ClassificationResult.

    Args:
        file_path: Path to the document file to classify.
        vector_store: VectorStore instance to query for centroids and neighbours.
        config: Optional PipelineConfig override. Falls back to module-level _config.

    Returns:
        ClassificationResult on success, ClassificationError on failure.
    """
    import os

    effective_config = config if config is not None else _config
    document_id = os.path.basename(file_path)

    # Step 1: Extract text
    extraction = ingestion_module.extract(file_path)
    if isinstance(extraction, IngestionError):
        return ClassificationError(
            error_type="embedding_failed",
            message=f"Ingestion failed for '{file_path}': {extraction.message}",
        )

    # Step 2: Check rule-based classification first
    rule_label, matched_keywords = _check_rules(extraction.text)
    
    # DEBUG: Log extracted text and rule check result
    from document_classifier import logger as _logger
    _logger.log_stage_complete(
        stage="classification_text_debug",
        document_id=document_id,
        duration_ms=0.0,
        metadata={
            "text_snippet": extraction.text[:500].replace('\n', ' '),
            "text_length": len(extraction.text),
            "rule_matched": rule_label is not None,
            "rule_label": rule_label,
        },
    )
    
    if rule_label:
        # Rule matched - return immediately with high confidence
        _logger.log_stage_complete(
            stage="classification_rule_match",
            document_id=document_id,
            duration_ms=0.0,
            metadata={
                "label": rule_label,
                "matched_keywords": matched_keywords,
                "method": "rule-based",
            },
        )
        return ClassificationResult(
            label=rule_label,
            confidence_score=1.0,  # 100% confidence for rule-based matches
            low_confidence=False,
            document_id=document_id,
        )

    # Step 3: No rule matched - fall back to ML-based classification
    # Embed text
    try:
        embedding_result = embedding_module.embed(extraction.text, document_id)
        vector = embedding_result.vector
    except Exception as exc:  # noqa: BLE001
        return ClassificationError(
            error_type="embedding_failed",
            message=f"Embedding failed for '{file_path}': {exc}",
        )

    # Step 4: Query nearest neighbours (used for context; centroid comparison is primary)
    vector_store.query_nearest(vector, n_results=5)

    # Step 5: Get cluster centroids
    centroids = vector_store.get_cluster_centroids()
    if not centroids:
        return ClassificationError(
            error_type="no_trained_clusters",
            message=(
                "No labelled clusters found in the Vector Store. "
                "Run the pipeline on a corpus first to train clusters."
            ),
        )

    # Step 6: Find nearest centroid by cosine distance
    doc_vector = np.array(vector, dtype=float)
    nearest_label: str | None = None
    nearest_distance: float = float("inf")

    # Collect all distances for debug logging
    cluster_distances: list[tuple[int, float]] = []

    for cluster_id, centroid in centroids.items():
        centroid_vector = np.array(centroid, dtype=float)
        distance = _cosine_distance(doc_vector, centroid_vector)
        cluster_distances.append((cluster_id, distance))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_cluster_id = cluster_id

    # Log top-3 candidates so misclassifications are easy to diagnose
    cluster_distances.sort(key=lambda x: x[1])
    top3 = []
    for cid, dist in cluster_distances[:3]:
        members = vector_store.get_cluster_members(cid)
        lbl = next((m.label for m in members if m.label), f"Cluster {cid}")
        top3.append({"cluster_id": cid, "label": lbl, "distance": round(dist, 4)})

    from document_classifier import logger as _logger
    _logger.log_stage_complete(
        stage="classification_ml_debug",
        document_id=document_id,
        duration_ms=0.0,
        metadata={"top3_candidates": top3, "method": "ml-based"},
    )

    # Retrieve the label for the nearest cluster
    members = vector_store.get_cluster_members(nearest_cluster_id)
    if members:
        # Use the label from the first member that has one
        label = None
        for member in members:
            if member.label is not None:
                label = member.label
                break
        if label is None:
            label = f"Cluster {nearest_cluster_id}"
    else:
        label = f"Cluster {nearest_cluster_id}"

    # Step 7: Compute confidence score
    confidence_score = float(np.clip(1.0 - nearest_distance, 0.0, 1.0))

    # Step 8: Determine low_confidence flag
    threshold = effective_config.low_confidence_threshold
    low_confidence = confidence_score < threshold

    return ClassificationResult(
        label=label,
        confidence_score=confidence_score,
        low_confidence=low_confidence,
        document_id=document_id,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors, clamped to [0.0, 1.0].

    cosine_distance = 1 - dot(a, b) / (norm(a) * norm(b))

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine distance in [0.0, 1.0].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        # Zero vector has undefined cosine distance; treat as maximally distant
        return 1.0
    cosine_similarity = float(np.dot(a, b) / (norm_a * norm_b))
    distance = 1.0 - cosine_similarity
    # Clamp to [0.0, 1.0] to handle floating-point rounding
    return float(np.clip(distance, 0.0, 1.0))
