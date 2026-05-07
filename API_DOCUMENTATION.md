# Document Classifier — FastAPI Developer Reference

This document is the complete reference for converting the document classifier
into a FastAPI service. It covers every function that needs to be exposed,
the exact inputs/outputs, error handling, and implementation notes.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Startup — Pipeline Singleton](#2-startup--pipeline-singleton)
3. [File Handling Pattern](#3-file-handling-pattern)
4. [Endpoint Reference](#4-endpoint-reference)
   - [POST /ingest](#post-ingest)
   - [POST /classify](#post-classify)
   - [POST /classify/batch](#post-classifybatch)
   - [GET /clusters](#get-clusters)
   - [GET /clusters/{cluster_id}/samples](#get-clusterscluster_idsamples)
   - [GET /clusters/{cluster_id}/label](#get-clusterscluster_idlabel)
   - [PUT /clusters/{cluster_id}/label](#put-clusterscluster_idlabel)
   - [GET /labels](#get-labels)
   - [POST /labels](#post-labels)
   - [POST /corrections/{document_id}](#post-correctionsdocument_id)
5. [Return Types Reference](#5-return-types-reference)
6. [Error Types Reference](#6-error-types-reference)
7. [Configuration via Environment Variables](#7-configuration-via-environment-variables)
8. [Performance Notes](#8-performance-notes)

---

## 1. Architecture Overview

```
FastAPI app
    │
    ├── Startup: Pipeline() instantiated ONCE, held in app.state
    │
    └── Endpoints call Pipeline methods
            │
            ├── pipeline.ingest()        → OCR + embed + store in ChromaDB
            ├── pipeline.classify()      → OCR + embed + nearest-centroid lookup
            ├── pipeline.classify_batch()→ classify() called per file
            ├── pipeline.list_clusters() → read ChromaDB centroids
            ├── pipeline.label_cluster() → write label to ChromaDB
            ├── pipeline.export_labels() → read all labels from ChromaDB
            ├── pipeline.import_labels() → bulk write labels to ChromaDB
            └── pipeline.apply_correction() → write corrected_label to ChromaDB
```

The `Pipeline` class lives in `document_classifier/pipeline.py`.
It is the **only** object the API needs to interact with.

---

## 2. Startup — Pipeline Singleton

**Critical:** The pipeline must be created **once** at app startup and reused
across all requests. Creating it per-request will reload the embedding model
(~10 seconds) on every call.

```python
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from document_classifier.pipeline import Pipeline
from document_classifier.config import PipelineConfig
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = PipelineConfig(
        chroma_persist_dir=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
        log_output="file",
        hdbscan_min_cluster_size=5,
        hdbscan_min_samples=2,
    )
    app.state.pipeline = Pipeline(config=config)
    yield
    # cleanup if needed

app = FastAPI(lifespan=lifespan)
```

Access the pipeline in any endpoint via:

```python
from fastapi import Request

def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline
```

---

## 3. File Handling Pattern

All endpoints that accept file uploads must:
1. Save the upload to a temp file
2. Pass the temp file path to the pipeline
3. Delete the temp file in a `finally` block

```python
import tempfile
import os
from pathlib import Path
from fastapi import UploadFile

async def save_upload(file: UploadFile) -> str:
    """Save an UploadFile to a temp path and return the path."""
    suffix = Path(file.filename).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name
```

Supported file extensions: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.docx`

---

## 4. Endpoint Reference

---

### POST /ingest

Ingest a document into the pipeline. Extracts text via OCR/parsing, embeds it,
stores it in ChromaDB, and triggers re-clustering if the threshold is reached.

**Source:** `Pipeline.ingest(file_path: str)`

**Request:**
```
Content-Type: multipart/form-data
Body: file  (UploadFile)
```

**Success Response — 200:**
```json
{
  "id": "d1bb2436-fc1a-468c-98dd-f03d2192539b",
  "source_filename": "invoice_001.jpg",
  "file_format": "image",
  "ingestion_timestamp": "2026-05-05T11:23:14.192Z",
  "cluster_id": null,
  "label": null
}
```

**Error Response — 422:**
```json
{
  "error_type": "unsupported_format",
  "message": "Unsupported file format '.xlsx'. Supported formats: .pdf, .png, .jpg, .jpeg, .docx"
}
```

**Error Response — 422:**
```json
{
  "error_type": "corrupt_file",
  "message": "Failed to extract text from 'file.pdf': ..."
}
```

**Error Response — 422:**
```json
{
  "error_type": "ocr_failure",
  "message": "OCR produced empty text for 'scan.jpg'"
}
```

**Implementation:**
```python
from fastapi import APIRouter, UploadFile, File, Request, HTTPException
from document_classifier.ingestion import IngestionError

router = APIRouter()

@router.post("/ingest")
async def ingest(request: Request, file: UploadFile = File(...)):
    pipeline = request.app.state.pipeline
    tmp_path = await save_upload(file)
    try:
        result = pipeline.ingest(tmp_path)
    finally:
        os.unlink(tmp_path)

    if isinstance(result, IngestionError):
        raise HTTPException(status_code=422, detail={
            "error_type": result.error_type,
            "message": result.message,
        })

    return {
        "id": result.id,
        "source_filename": result.source_filename,
        "file_format": result.file_format,
        "ingestion_timestamp": result.ingestion_timestamp,
        "cluster_id": result.cluster_id,
        "label": result.label,
    }
```

**Notes:**
- This is a **slow endpoint** — OCR on an image takes ~5–6 seconds.
  Run it in a thread pool: `await asyncio.get_event_loop().run_in_executor(None, pipeline.ingest, tmp_path)`
- Re-clustering fires automatically every `recluster_threshold` (default 20) new documents.
  This adds ~1–2 seconds to the response when it triggers.

---

### POST /classify

Classify a single document by finding the nearest cluster centroid.

**Source:** `Pipeline.classify(file_path: str)`

**Request:**
```
Content-Type: multipart/form-data
Body: file  (UploadFile)
```

**Success Response — 200:**
```json
{
  "label": "Invoice / Receipt",
  "confidence_score": 0.87,
  "low_confidence": false,
  "document_id": "invoice_042.jpg"
}
```

**Error Response — 422 (no trained clusters):**
```json
{
  "error_type": "no_trained_clusters",
  "message": "No labelled clusters found in the Vector Store. Run the pipeline on a corpus first."
}
```

**Error Response — 422 (ingestion/embedding failed):**
```json
{
  "error_type": "embedding_failed",
  "message": "Ingestion failed for 'file.jpg': OCR produced empty text"
}
```

**Implementation:**
```python
from document_classifier.classifier import ClassificationError

@router.post("/classify")
async def classify(request: Request, file: UploadFile = File(...)):
    pipeline = request.app.state.pipeline
    tmp_path = await save_upload(file)
    try:
        result = pipeline.classify(tmp_path)
    finally:
        os.unlink(tmp_path)

    if isinstance(result, ClassificationError):
        raise HTTPException(status_code=422, detail={
            "error_type": result.error_type,
            "message": result.message,
        })

    return {
        "label": result.label,
        "confidence_score": result.confidence_score,
        "low_confidence": result.low_confidence,
        "document_id": result.document_id,
    }
```

**Notes:**
- `confidence_score` is `1 - cosine_distance` between the document embedding
  and the nearest cluster centroid. Range: `0.0` (no match) to `1.0` (identical).
- `low_confidence = True` when `confidence_score < 0.5` (configurable via
  `PipelineConfig.low_confidence_threshold`).
- Rule-based classification fires first (keyword matching). If a rule matches,
  `confidence_score` is always `1.0`. Only falls back to ML if no rule matches.

---

### POST /classify/batch

Classify multiple documents in one request.

**Source:** `Pipeline.classify_batch(file_paths: list[str])`

**Request:**
```
Content-Type: multipart/form-data
Body: files  (list[UploadFile])
```

**Success Response — 200:**
```json
[
  {
    "filename": "invoice_001.jpg",
    "label": "Invoice / Receipt",
    "confidence_score": 0.91,
    "low_confidence": false,
    "document_id": "invoice_001.jpg"
  },
  {
    "filename": "unknown.jpg",
    "error_type": "no_trained_clusters",
    "message": "..."
  }
]
```

**Implementation:**
```python
from fastapi import UploadFile, File
from typing import List

@router.post("/classify/batch")
async def classify_batch(request: Request, files: List[UploadFile] = File(...)):
    pipeline = request.app.state.pipeline
    tmp_paths = []
    for f in files:
        tmp_paths.append((f.filename, await save_upload(f)))

    try:
        results = pipeline.classify_batch([p for _, p in tmp_paths])
    finally:
        for _, p in tmp_paths:
            os.unlink(p)

    output = []
    for (filename, _), result in zip(tmp_paths, results):
        if isinstance(result, ClassificationError):
            output.append({
                "filename": filename,
                "error_type": result.error_type,
                "message": result.message,
            })
        else:
            output.append({
                "filename": filename,
                "label": result.label,
                "confidence_score": result.confidence_score,
                "low_confidence": result.low_confidence,
                "document_id": result.document_id,
            })
    return output
```

**Notes:**
- Results are returned in the **same order** as the uploaded files.
- Each file is classified independently — one failure does not affect others.

---

### GET /clusters

List all cluster IDs currently in the vector store.

**Source:** `Pipeline.list_clusters()`

**Response — 200:**
```json
{
  "cluster_ids": [0, 1, 2, 3]
}
```

**Implementation:**
```python
@router.get("/clusters")
def list_clusters(request: Request):
    pipeline = request.app.state.pipeline
    return {"cluster_ids": pipeline.list_clusters()}
```

**Notes:**
- Returns only non-outlier clusters. Outlier cluster `-1` is excluded.
- Returns an empty list `[]` if no clustering has run yet.

---

### GET /clusters/{cluster_id}/samples

Get up to `n` representative text samples from a cluster. Use this to review
what documents are in a cluster before assigning a label.

**Source:** `Pipeline.get_cluster_samples(cluster_id: int, n: int = 5)`

**Query params:** `n` (optional, default `5`) — max number of samples to return.

**Response — 200:**
```json
{
  "cluster_id": 2,
  "samples": [
    "INVOICE\nDate: 12/03/2024\nBill To: Acme Corp...",
    "TAX INVOICE\nGST No: 27AABCU9603R1ZX..."
  ]
}
```

**Error Response — 404:**
```json
{"detail": "Cluster 99 not found"}
```

**Implementation:**
```python
@router.get("/clusters/{cluster_id}/samples")
def get_cluster_samples(request: Request, cluster_id: int, n: int = 5):
    pipeline = request.app.state.pipeline
    if cluster_id not in pipeline.list_clusters():
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
    samples = pipeline.get_cluster_samples(cluster_id, n=n)
    return {"cluster_id": cluster_id, "samples": samples}
```

---

### GET /clusters/{cluster_id}/label

Get the current label assigned to a cluster.

**Source:** `Pipeline.get_cluster_label(cluster_id: int)`

**Response — 200 (labelled):**
```json
{
  "cluster_id": 2,
  "label": "Invoice / Receipt"
}
```

**Response — 200 (not yet labelled):**
```json
{
  "cluster_id": 2,
  "label": null
}
```

**Implementation:**
```python
@router.get("/clusters/{cluster_id}/label")
def get_cluster_label(request: Request, cluster_id: int):
    pipeline = request.app.state.pipeline
    label = pipeline.get_cluster_label(cluster_id)
    return {"cluster_id": cluster_id, "label": label}
```

---

### PUT /clusters/{cluster_id}/label

Assign or update the human-readable label for a cluster. This writes the label
to every document in the cluster inside ChromaDB.

**Source:** `Pipeline.label_cluster(cluster_id: int, label: str)`

**Request body:**
```json
{
  "label": "Invoice / Receipt"
}
```

**Response — 200:**
```json
{
  "cluster_id": 2,
  "label": "Invoice / Receipt",
  "success": true
}
```

**Error Response — 422 (empty label):**
```json
{"detail": "Label cannot be empty"}
```

**Implementation:**
```python
from pydantic import BaseModel

class LabelRequest(BaseModel):
    label: str

@router.put("/clusters/{cluster_id}/label")
def label_cluster(request: Request, cluster_id: int, body: LabelRequest):
    if not body.label.strip():
        raise HTTPException(status_code=422, detail="Label cannot be empty")
    pipeline = request.app.state.pipeline
    result = pipeline.label_cluster(cluster_id, body.label.strip())
    return {
        "cluster_id": result.cluster_id,
        "label": result.label,
        "success": result.success,
    }
```

**Notes:**
- `LabelResult.success` is always `True` for manual labelling — there is no
  failure path. The pipeline writes directly to ChromaDB.
- Calling this on a cluster that doesn't exist is a no-op (ChromaDB returns
  zero rows to update). Add a `list_clusters()` check if you want a 404.

---

### GET /labels

Export all cluster labels as a JSON dict. Unlabelled clusters are omitted.

**Source:** `Pipeline.export_labels()`

**Response — 200:**
```json
{
  "labels": {
    "0": "Aadhaar Card",
    "1": "PAN Card",
    "2": "Invoice / Receipt"
  }
}
```

**Implementation:**
```python
@router.get("/labels")
def export_labels(request: Request):
    pipeline = request.app.state.pipeline
    labels = pipeline.export_labels()
    # Keys are int in Python — convert to str for JSON
    return {"labels": {str(k): v for k, v in labels.items()}}
```

---

### POST /labels

Bulk import cluster labels, overwriting any existing ones.

**Source:** `Pipeline.import_labels(labels: dict[int, str])`

**Request body:**
```json
{
  "labels": {
    "0": "Aadhaar Card",
    "1": "PAN Card",
    "2": "Invoice / Receipt"
  }
}
```

**Response — 200:**
```json
{
  "imported": 3
}
```

**Implementation:**
```python
class LabelsImportRequest(BaseModel):
    labels: dict[str, str]  # JSON keys are always strings

@router.post("/labels")
def import_labels(request: Request, body: LabelsImportRequest):
    pipeline = request.app.state.pipeline
    # Convert string keys back to int
    labels_int = {int(k): v for k, v in body.labels.items()}
    pipeline.import_labels(labels_int)
    return {"imported": len(labels_int)}
```

---

### POST /corrections/{document_id}

Store an operator-provided corrected label for a specific document. Used when
the classifier returns a wrong label and the operator wants to override it.
The correction is persisted in ChromaDB and used as a seed in the next
clustering run.

**Source:** `Pipeline.apply_correction(document_id: str, corrected_label: str)`

**Request body:**
```json
{
  "corrected_label": "PAN Card"
}
```

**Response — 200:**
```json
{
  "document_id": "d1bb2436-fc1a-468c-98dd-f03d2192539b",
  "corrected_label": "PAN Card"
}
```

**Implementation:**
```python
class CorrectionRequest(BaseModel):
    corrected_label: str

@router.post("/corrections/{document_id}")
def apply_correction(request: Request, document_id: str, body: CorrectionRequest):
    pipeline = request.app.state.pipeline
    pipeline.apply_correction(document_id, body.corrected_label)
    return {"document_id": document_id, "corrected_label": body.corrected_label}
```

**Notes:**
- `apply_correction()` returns `None` — there is no success/failure signal.
  If `document_id` doesn't exist in ChromaDB, it silently does nothing.
  Add a lookup if you need a 404.

---

## 5. Return Types Reference

### DocumentRecord (from `vector_store.py`)
Returned by `pipeline.ingest()` on success.

| Field | Type | Description |
|---|---|---|
| `id` | `str` | UUID4 string — unique document identifier |
| `text` | `str` | Extracted text content |
| `vector` | `list[float]` | 1024-dim embedding — **do not expose in API response** (too large) |
| `source_filename` | `str` | Original filename of the uploaded file |
| `file_format` | `str` | One of `"pdf"`, `"image"`, `"docx"` |
| `ingestion_timestamp` | `str` | ISO-8601 UTC timestamp e.g. `"2026-05-05T11:23:14.192Z"` |
| `cluster_id` | `int \| None` | HDBSCAN cluster assignment; `-1` = outlier; `None` = not yet clustered |
| `label` | `str \| None` | Human-readable cluster label; `None` if not yet labelled |
| `corrected_label` | `str \| None` | Operator override label; `None` if not corrected |

### ClassificationResult (from `classifier.py`)
Returned by `pipeline.classify()` on success.

| Field | Type | Description |
|---|---|---|
| `label` | `str` | Predicted human-readable label |
| `confidence_score` | `float` | `1 - cosine_distance` to nearest centroid. Range `[0.0, 1.0]` |
| `low_confidence` | `bool` | `True` if `confidence_score < 0.5` |
| `document_id` | `str` | Basename of the file path passed in |

### LabelResult (from `label_service.py`)
Returned by `pipeline.label_cluster()`.

| Field | Type | Description |
|---|---|---|
| `cluster_id` | `int` | The cluster that was labelled |
| `label` | `str` | The label string that was stored |
| `success` | `bool` | Always `True` for manual labelling |
| `error` | `str \| None` | Always `None` for manual labelling |

---

## 6. Error Types Reference

### IngestionError (from `ingestion.py`)
Returned by `pipeline.ingest()` on failure.

| `error_type` | Cause | Suggested HTTP status |
|---|---|---|
| `"unsupported_format"` | File extension not in `.pdf .png .jpg .jpeg .docx` | 422 |
| `"corrupt_file"` | File could not be parsed/opened | 422 |
| `"ocr_failure"` | OCR ran but returned empty text | 422 |

### ClassificationError (from `classifier.py`)
Returned by `pipeline.classify()` / `pipeline.classify_batch()` on failure.

| `error_type` | Cause | Suggested HTTP status |
|---|---|---|
| `"no_trained_clusters"` | ChromaDB has no labelled clusters yet | 503 (service not ready) |
| `"embedding_failed"` | Ingestion or embedding step failed | 422 |

---

## 7. Configuration via Environment Variables

The `PipelineConfig` dataclass (in `config.py`) controls all pipeline behaviour.
Map these to env vars in the FastAPI app:

| Env var | Config field | Default | Description |
|---|---|---|---|
| `CHROMA_DB_PATH` | `chroma_persist_dir` | `"./chroma_db"` | Path to ChromaDB storage directory |
| `EMBEDDING_MODEL` | `embedding_model_name` | `"intfloat/multilingual-e5-large"` | HuggingFace model ID |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `hdbscan_min_cluster_size` | `5` | Min docs to form a cluster |
| `HDBSCAN_MIN_SAMPLES` | `hdbscan_min_samples` | `2` | HDBSCAN noise sensitivity |
| `RECLUSTER_THRESHOLD` | `recluster_threshold` | `20` | New docs before auto re-cluster |
| `LOW_CONFIDENCE_THRESHOLD` | `low_confidence_threshold` | `0.5` | Below this → `low_confidence=True` |
| `LOG_OUTPUT` | `log_output` | `"stdout"` | `"stdout"` \| `"file"` \| `"both"` |

---

## 8. Performance Notes

| Operation | Typical duration | Notes |
|---|---|---|
| Model load (first request) | ~10s | Happens once at startup, not per request |
| OCR on image | ~5–6s | CPU-bound; use `run_in_executor` |
| Embedding | ~0.5–1s | CPU-bound after model is warm |
| ChromaDB read/write | <100ms | Fast local SQLite |
| Re-clustering (200 docs) | ~1–2s | Fires automatically every N ingestions |

**For production**, wrap all `pipeline.ingest()` and `pipeline.classify()` calls
in `asyncio.get_event_loop().run_in_executor(None, ...)` to avoid blocking the
event loop, or offload to a Celery task queue and return a job ID immediately.
