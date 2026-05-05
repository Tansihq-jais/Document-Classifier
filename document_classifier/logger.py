"""Structured JSON logger for the document-classifier pipeline.

All log entries are emitted as single-line JSON objects.  Raw document text and
embedding vectors are NEVER included in any log entry.

Output destination is controlled by ``configure_logger(config)`` and defaults to
stdout.  Supported values for ``PipelineConfig.log_output``:
    - "stdout"  — write to sys.stdout only (default)
    - "file"    — write to the path given by ``PipelineConfig.log_file_path`` only
    - "both"    — write to both stdout and the file
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from document_classifier.config import PipelineConfig

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_log_output: str = "stdout"
_log_file_path: str = "./pipeline.log"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logger(config: "PipelineConfig") -> None:
    """Set the output mode from a ``PipelineConfig`` instance.

    Must be called before any logging functions are used if non-default
    behaviour is required.  Safe to call multiple times (last call wins).

    Args:
        config: A ``PipelineConfig`` instance whose ``log_output`` and
            ``log_file_path`` fields control where log entries are written.
    """
    global _log_output, _log_file_path
    _log_output = config.log_output
    _log_file_path = config.log_file_path


def log_stage_complete(
    stage: str,
    document_id: str,
    duration_ms: float,
    metadata: dict,
) -> None:
    """Emit a structured JSON log entry for a successfully completed pipeline stage.

    The entry contains ``timestamp``, ``level``, ``stage``, ``document_id``,
    ``status``, ``duration_ms``, and any additional ``metadata`` supplied by the
    caller.  Raw document text and embedding vectors MUST NOT appear in
    ``metadata``.

    Args:
        stage: Name of the pipeline stage (e.g. "ingestion", "embedding").
        document_id: Unique identifier of the document being processed.
        duration_ms: Wall-clock duration of the stage in milliseconds (must be ≥ 0).
        metadata: Arbitrary key/value pairs with stage-specific diagnostics.
            Must not contain raw text or float arrays.
    """
    entry = {
        "timestamp": _utc_now(),
        "level": "INFO",
        "stage": stage,
        "document_id": document_id,
        "status": "complete",
        "duration_ms": duration_ms,
        "metadata": metadata,
    }
    _emit(entry)


def log_stage_error(
    stage: str,
    document_id: str,
    error_type: str,
    message: str,
) -> None:
    """Emit a structured JSON log entry for a pipeline stage failure.

    The entry contains ``timestamp``, ``level``, ``stage``, ``document_id``,
    ``status``, ``error_type``, and ``message``.

    Args:
        stage: Name of the pipeline stage where the error occurred.
        document_id: Unique identifier of the document being processed.
        error_type: Machine-readable error category (e.g. "corrupt_file").
        message: Human-readable description of the failure.
    """
    entry = {
        "timestamp": _utc_now(),
        "level": "ERROR",
        "stage": stage,
        "document_id": document_id,
        "status": "error",
        "error_type": error_type,
        "message": message,
    }
    _emit(entry)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string with millisecond precision."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _emit(entry: dict) -> None:
    """Serialise *entry* to JSON and write it to the configured destination(s)."""
    line = json.dumps(entry, ensure_ascii=False)

    if _log_output in ("stdout", "both"):
        print(line, file=sys.stdout, flush=True)

    if _log_output in ("file", "both"):
        try:
            with open(_log_file_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            # Fall back to stderr so the error is visible without crashing the pipeline.
            print(
                f"[logger] Failed to write to log file '{_log_file_path}': {exc}",
                file=sys.stderr,
                flush=True,
            )
