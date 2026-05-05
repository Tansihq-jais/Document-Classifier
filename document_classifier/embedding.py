"""Embedding module for the document-classifier pipeline.

Wraps ``sentence-transformers`` to load ``multilingual-e5-large`` locally and
produce 1024-dimensional float vectors for arbitrary text.

The model is loaded lazily on the first call to :func:`embed` (or via
:func:`_get_model`).  A single model instance is reused across all calls.

Usage::

    from document_classifier.embedding import embed, configure_embedding
    from document_classifier.config import PipelineConfig

    # Optional: override config before first call
    configure_embedding(PipelineConfig(embedding_model_name="intfloat/multilingual-e5-large"))

    result = embed("Some document text", document_id="doc-uuid-123")
    print(len(result.vector))   # 1024
    print(result.truncated)     # True if token_count > 512
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from document_classifier.config import PipelineConfig
from document_classifier.logger import log_stage_complete, log_stage_error

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_config: PipelineConfig = PipelineConfig()
_model: "SentenceTransformer | None" = None

# Maximum token length supported by the E5 model family.
_MAX_TOKENS: int = 512


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingResult:
    """Result of embedding a single document.

    Attributes:
        vector: 1024-dimensional float vector produced by the model.
        token_count: Number of tokens in the prefixed input text.
        truncated: True if ``token_count > 512``; the model silently truncated
            the input during encoding.
        tokens_truncated: Number of tokens that were dropped (``token_count - 512``
            when truncated, otherwise 0).
    """

    vector: list[float]
    token_count: int
    truncated: bool
    tokens_truncated: int


def configure_embedding(config: PipelineConfig) -> None:
    """Override the module-level configuration.

    Must be called *before* the first :func:`embed` call if a non-default
    model name is required.  Calling this after the model has been loaded
    will NOT reload the model; restart the process to pick up a new model.

    Args:
        config: A :class:`~document_classifier.config.PipelineConfig` instance
            whose ``embedding_model_name`` field controls which model is loaded.
    """
    global _config
    _config = config


def _get_model() -> "SentenceTransformer":
    """Return the module-level model, loading it on first access (lazy init).

    Returns:
        The loaded :class:`sentence_transformers.SentenceTransformer` instance.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        _model = SentenceTransformer(_config.embedding_model_name)
    return _model


def embed(text: str, document_id: str) -> EmbeddingResult:
    """Embed *text* and return a 1024-dimensional :class:`EmbeddingResult`.

    The input is prefixed with ``"passage: "`` as required by the E5 model
    family for document (as opposed to query) embedding.

    If the token count of the prefixed text exceeds 512, ``truncated`` is set
    to ``True``, ``tokens_truncated`` is set to the excess, and a warning is
    logged via :func:`~document_classifier.logger.log_stage_error` with
    ``error_type="truncation_warning"``.

    Args:
        text: Raw extracted text to embed.
        document_id: Unique identifier of the document (used in log entries).

    Returns:
        An :class:`EmbeddingResult` with a 1024-dimensional float vector.
    """
    start_ms = time.monotonic() * 1000

    model = _get_model()

    # E5 models require a task prefix for document embedding.
    prefixed_text = f"passage: {text}"

    # Count tokens using the model's tokenizer BEFORE encoding.
    token_ids = model.tokenizer.encode(prefixed_text)
    token_count = len(token_ids)

    truncated = token_count > _MAX_TOKENS
    tokens_truncated = (token_count - _MAX_TOKENS) if truncated else 0

    if truncated:
        log_stage_error(
            stage="embedding",
            document_id=document_id,
            error_type="truncation_warning",
            message=(
                f"Input truncated for document '{document_id}': "
                f"token_count={token_count}, tokens_truncated={tokens_truncated}"
            ),
        )

    # sentence-transformers handles truncation internally during encode().
    vector_array = model.encode(prefixed_text, convert_to_numpy=True)
    vector: list[float] = vector_array.tolist()

    duration_ms = time.monotonic() * 1000 - start_ms

    log_stage_complete(
        stage="embedding",
        document_id=document_id,
        duration_ms=duration_ms,
        metadata={
            "token_count": token_count,
            "truncated": truncated,
            "tokens_truncated": tokens_truncated,
        },
    )

    return EmbeddingResult(
        vector=vector,
        token_count=token_count,
        truncated=truncated,
        tokens_truncated=tokens_truncated,
    )
