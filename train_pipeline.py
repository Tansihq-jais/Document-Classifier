"""Supervised training pipeline for the document classifier.

Ingests all 7 document types with CORRECT labels assigned directly at ingest
time — no unsupervised clustering, no majority-vote guessing.

Each document type gets a fixed cluster_id so the classifier's centroid lookup
works correctly:

    0  → Aadhaar Card
    1  → PAN Card
    2  → Indian Passport
    3  → Invoice / Receipt
    4  → Purchase Order
    5  → Inventory Report
    6  → Shipping Order

Directory structure expected (200 files each):
    training_data/
    ├── aadhaar/           → label: "Aadhaar Card"
    ├── pan_card/          → label: "PAN Card"
    ├── passport/          → label: "Indian Passport"
    ├── invoices_receipts/ → label: "Invoice / Receipt"
    ├── purchase_order/    → label: "Purchase Order"
    ├── inventory_report/  → label: "Inventory Report"
    └── shipping_order/    → label: "Shipping Order"

Usage:
    python train_pipeline.py
    python train_pipeline.py --data-dir path/to/data --db-dir ./chroma_db --max 200
    python train_pipeline.py --no-confirm   # skip the Y/n prompt
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from document_classifier.config import PipelineConfig
from document_classifier.ingestion import IngestionError, SUPPORTED_FORMATS
from document_classifier.vector_store import DocumentRecord, VectorStore
from document_classifier import embedding as embedding_module
from document_classifier import ingestion as ingestion_module

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("training_data")
DEFAULT_DB_DIR = "./chroma_db"
DEFAULT_MAX_PER_TYPE = 200

# Map subdirectory name → (human-readable label, fixed cluster_id)
# cluster_id is fixed per type so centroid lookup is deterministic.
LABEL_MAP: dict[str, tuple[str, int]] = {
    "aadhaar":           ("Aadhaar Card",      0),
    "pan_card":          ("PAN Card",           1),
    "pan":               ("PAN Card",           1),
    "passport":          ("Indian Passport",    2),
    "invoices_receipts": ("Invoice / Receipt",  3),
    "purchase_order":    ("Purchase Order",     4),
    "inventory_report":  ("Inventory Report",   5),
    "shipping_order":    ("Shipping Order",     6),
}

SUPPORTED_EXTS = {ext.lstrip(".") for ext in SUPPORTED_FORMATS}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_files(
    data_dir: Path, max_per_type: int
) -> list[tuple[Path, str, int]]:
    """Collect up to max_per_type supported files per labelled subdirectory.

    Returns:
        List of (file_path, label, cluster_id) tuples.
    """
    files: list[tuple[Path, str, int]] = []

    if not data_dir.exists():
        print(f"❌ Training data directory not found: {data_dir}")
        return files

    skipped_dirs: list[str] = []

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue

        key = subdir.name.lower()
        if key not in LABEL_MAP:
            skipped_dirs.append(subdir.name)
            continue

        label, cluster_id = LABEL_MAP[key]

        subdir_files = sorted(
            f for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower().lstrip(".") in SUPPORTED_EXTS
        )

        selected = subdir_files[:max_per_type]

        if not selected:
            print(f"⚠️  No supported files in: {subdir.name}/")
            continue

        for f in selected:
            files.append((f, label, cluster_id))

    if skipped_dirs:
        print(f"ℹ️  Skipped unknown subdirs: {', '.join(skipped_dirs)}")

    return files


def _print_plan(files: list[tuple[Path, str, int]]) -> None:
    """Print a summary of what will be ingested."""
    label_counts: Counter = Counter(label for _, label, _ in files)

    print("📋 Training Plan (supervised — labels assigned directly):")
    print("-" * 55)
    for label, count in sorted(label_counts.items()):
        print(f"  {label:<25}: {count:>4} files")
    print("-" * 55)
    print(f"  {'TOTAL':<25}: {len(files):>4} files")
    print()


def _now_iso() -> str:
    return (
        datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )


# ---------------------------------------------------------------------------
# Core supervised ingest
# ---------------------------------------------------------------------------


def _ingest_supervised(
    file_path: Path,
    label: str,
    cluster_id: int,
    vector_store: VectorStore,
) -> bool:
    """Extract, embed, and store a document with a pre-assigned label.

    Returns True on success, False on failure.
    """
    # 1. Extract text
    extraction = ingestion_module.extract(str(file_path))
    if isinstance(extraction, IngestionError):
        return False

    # 2. Embed
    doc_id = str(uuid.uuid4())
    try:
        embedding_result = embedding_module.embed(extraction.text, doc_id)
    except Exception:
        return False

    # 3. Build record with label + cluster_id already set
    record = DocumentRecord(
        id=doc_id,
        text=extraction.text,
        vector=embedding_result.vector,
        source_filename=extraction.source_file,
        file_format=extraction.file_format,
        ingestion_timestamp=_now_iso(),
        cluster_id=cluster_id,
        label=label,
        corrected_label=None,
    )

    # 4. Persist
    vector_store.add(record)
    return True


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(data_dir: Path, db_dir: str, max_per_type: int, no_confirm: bool) -> None:
    print("=" * 60)
    print("Document Classifier — Supervised Training Pipeline")
    print("=" * 60)
    print()

    # Step 1: Collect files
    files = _collect_files(data_dir, max_per_type)
    if not files:
        print("❌ No files found. Check your training_data/ directory.")
        sys.exit(1)

    _print_plan(files)

    # Warn if any type is missing
    present_labels = {label for _, label, _ in files}
    all_labels = {v[0] for v in LABEL_MAP.values()}
    missing = all_labels - present_labels
    if missing:
        print(f"⚠️  Missing document types: {', '.join(sorted(missing))}")
        print()

    # Confirm
    if not no_confirm:
        answer = input(
            f"Start ingesting {len(files)} documents into '{db_dir}'? [Y/n]: "
        ).strip().lower()
        if answer == "n":
            print("Aborted.")
            return
        print()

    # Step 2: Initialise vector store directly (no pipeline / no clustering)
    config = PipelineConfig(
        chroma_persist_dir=db_dir,
        log_output="file",
    )
    vector_store = VectorStore(config)

    # Step 3: Ingest all documents with correct labels
    print("📥 Ingesting documents with supervised labels...")
    print("-" * 60)

    ok_count = 0
    fail_count = 0
    start_time = time.monotonic()
    current_label: str | None = None

    for i, (file_path, label, cluster_id) in enumerate(files, 1):
        if label != current_label:
            current_label = label
            print(f"\n  [{label}]  (cluster_id={cluster_id})")

        success = _ingest_supervised(file_path, label, cluster_id, vector_store)

        if success:
            print(f"  ✅ [{i:3}/{len(files)}] {file_path.name}")
            ok_count += 1
        else:
            print(f"  ❌ [{i:3}/{len(files)}] {file_path.name}  — ingestion/embedding failed")
            fail_count += 1

    elapsed = time.monotonic() - start_time

    # Step 4: Summary
    print()
    print("=" * 60)
    print("✅ Supervised Training Complete!")
    print("=" * 60)
    print(f"  Documents ingested : {ok_count}")
    print(f"  Documents failed   : {fail_count}")
    print(f"  Total time         : {elapsed:.1f}s")
    print()

    # Per-label breakdown
    label_ok: Counter = Counter()
    label_fail: Counter = Counter()
    for file_path, label, _ in files:
        # We don't track per-file success here, but show totals
        pass

    print("  Label assignments (cluster_id → label):")
    for key, (lbl, cid) in sorted(LABEL_MAP.items(), key=lambda x: x[1][1]):
        print(f"    {cid}  →  {lbl}")
    print()

    if ok_count == 0:
        print("❌ No documents were ingested. Cannot classify.")
        sys.exit(1)

    print("Run the UI to start classifying:")
    print("  python -m streamlit run document_classifier/app.py")
    print()
    print("Or classify from CLI:")
    print("  python -m document_classifier classify your_document.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised training for the document classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Training data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=DEFAULT_DB_DIR,
        help=f"ChromaDB directory (default: {DEFAULT_DB_DIR})",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_PER_TYPE,
        dest="max_per_type",
        help=f"Max files per document type (default: {DEFAULT_MAX_PER_TYPE})",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the Y/n confirmation prompt",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        db_dir=args.db_dir,
        max_per_type=args.max_per_type,
        no_confirm=args.no_confirm,
    )


if __name__ == "__main__":
    main()
