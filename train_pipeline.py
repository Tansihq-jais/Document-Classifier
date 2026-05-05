"""Train the document classifier pipeline on a corpus of labelled documents.

Walks the training_data/ directory, ingests all supported documents into the
vector store, triggers clustering, and auto-labels clusters based on the
subdirectory name (which acts as the ground-truth label).

Directory structure expected:
    training_data/
    ├── aadhaar/        → label: "Aadhaar Card"
    ├── pan_card/       → label: "PAN Card"
    ├── passport/       → label: "Indian Passport"
    └── other/          → label: "Other"

Usage:
    python train_pipeline.py
    python train_pipeline.py --data-dir path/to/data --db-dir path/to/chroma_db
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from document_classifier.config import PipelineConfig
from document_classifier.ingestion import IngestionError, SUPPORTED_FORMATS
from document_classifier.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("training_data")
DEFAULT_DB_DIR = "./chroma_db_paddle"

# Map subdirectory name → human-readable label
LABEL_MAP: dict[str, str] = {
    "aadhaar": "Aadhaar Card",
    "pan_card": "PAN Card",
    "pan": "PAN Card",
    "passport": "Indian Passport",
    "other": "Other",
}

SUPPORTED_EXTS = {ext.lstrip(".") for ext in SUPPORTED_FORMATS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_files(data_dir: Path) -> list[tuple[Path, str]]:
    """Collect all supported files with their labels from data_dir.

    Args:
        data_dir: Root training data directory with labelled subdirectories.

    Returns:
        List of (file_path, label) tuples.
    """
    files: list[tuple[Path, str]] = []

    if not data_dir.exists():
        print(f"❌ Training data directory not found: {data_dir}")
        print()
        print("Run first: python download_datasets.py")
        print("Or create the directory and add documents manually:")
        print(f"  {data_dir}/aadhaar/   → Aadhaar card images/PDFs")
        print(f"  {data_dir}/pan_card/  → PAN card images/PDFs")
        print(f"  {data_dir}/passport/  → Passport images/PDFs")
        print(f"  {data_dir}/other/     → Other documents")
        return files

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue

        label = LABEL_MAP.get(subdir.name.lower(), subdir.name.replace("_", " ").title())

        subdir_files = [
            f for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower().lstrip(".") in SUPPORTED_EXTS
        ]

        if not subdir_files:
            print(f"⚠️  No supported files in: {subdir.name}/")
            continue

        for f in sorted(subdir_files):
            files.append((f, label))

    return files


def _print_plan(files: list[tuple[Path, str]]) -> None:
    """Print a summary of what will be ingested."""
    from collections import Counter
    label_counts = Counter(label for _, label in files)

    print("📋 Training Plan:")
    print("-" * 40)
    for label, count in sorted(label_counts.items()):
        print(f"  {label:<25}: {count} files")
    print("-" * 40)
    print(f"  {'TOTAL':<25}: {len(files)} files")
    print()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(data_dir: Path, db_dir: str) -> None:
    print("=" * 55)
    print("Document Classifier — Training Pipeline")
    print("=" * 55)
    print()

    # Step 1: Collect files
    files = _collect_files(data_dir)
    if not files:
        sys.exit(1)

    _print_plan(files)

    # Confirm before starting
    answer = input(f"Start ingesting {len(files)} documents? [Y/n]: ").strip().lower()
    if answer == "n":
        print("Aborted.")
        return
    print()

    # Step 2: Initialise pipeline
    config = PipelineConfig(
        chroma_persist_dir=db_dir,
        log_output="file",
        hdbscan_min_cluster_size=3,
        hdbscan_min_samples=1,
        recluster_threshold=10,  # Cluster more frequently during training
    )
    pipeline = Pipeline(config=config)

    # Step 3: Ingest all documents
    print("📥 Ingesting documents...")
    print("-" * 55)

    ok_count = 0
    fail_count = 0
    start_time = time.monotonic()

    # Group files by label for organised output
    current_label = None

    for i, (file_path, label) in enumerate(files, 1):
        if label != current_label:
            current_label = label
            print(f"\n  [{label}]")

        result = pipeline.ingest(str(file_path))

        if isinstance(result, IngestionError):
            print(f"  ❌ [{i:3}/{len(files)}] {file_path.name} — {result.error_type}: {result.message}")
            fail_count += 1
        else:
            print(f"  ✅ [{i:3}/{len(files)}] {file_path.name}")
            ok_count += 1

    elapsed = time.monotonic() - start_time
    print()
    print("-" * 55)
    print(f"Ingestion complete in {elapsed:.1f}s")
    print(f"  ✅ Success : {ok_count}")
    print(f"  ❌ Failed  : {fail_count}")
    print()

    if ok_count == 0:
        print("❌ No documents were ingested successfully. Cannot train.")
        sys.exit(1)

    # Step 4: Force final clustering
    print("🔄 Running final clustering...")
    pipeline._vector_store.reset_new_doc_counter()
    # Force recluster by temporarily lowering threshold
    pipeline._config.recluster_threshold = 1
    pipeline._maybe_trigger_recluster()
    pipeline._config.recluster_threshold = config.recluster_threshold

    clusters = pipeline.list_clusters()
    print(f"   Found {len(clusters)} cluster(s)")
    print()

    if not clusters:
        print("⚠️  No clusters formed. Try ingesting more documents.")
        return

    # Step 5: Auto-label clusters based on majority document label
    print("🏷️  Auto-labelling clusters...")
    print("-" * 55)

    # Build a map of document_id → ground truth label
    doc_label_map: dict[str, str] = {}
    for file_path, label in files:
        # Match by source filename
        doc_label_map[file_path.name] = label

    labelled = 0
    for cluster_id in clusters:
        members = pipeline._vector_store.get_cluster_members(cluster_id)
        if not members:
            continue

        # Count labels among members
        from collections import Counter
        label_votes: Counter = Counter()
        for member in members:
            src = Path(member.source_filename).name if member.source_filename else ""
            gt_label = doc_label_map.get(src)
            if gt_label:
                label_votes[gt_label] += 1

        if label_votes:
            best_label, votes = label_votes.most_common(1)[0]
            pipeline.label_cluster(cluster_id, best_label)
            print(f"  Cluster {cluster_id:2d} → '{best_label}' ({votes}/{len(members)} votes)")
            labelled += 1
        else:
            print(f"  Cluster {cluster_id:2d} → ⚠️  Could not determine label ({len(members)} members)")

    print()
    print("=" * 55)
    print("✅ Training Complete!")
    print("=" * 55)
    print(f"  Clusters formed  : {len(clusters)}")
    print(f"  Clusters labelled: {labelled}")
    print(f"  Documents ingested: {ok_count}")
    print(f"  Total time       : {elapsed:.1f}s")
    print()
    print("Run the UI to start classifying:")
    print("  python -m streamlit run document_classifier/app.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the document classifier pipeline."
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
    args = parser.parse_args()

    train(data_dir=args.data_dir, db_dir=args.db_dir)


if __name__ == "__main__":
    main()
