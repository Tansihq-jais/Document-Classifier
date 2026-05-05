"""Download training datasets for the document classifier.

Downloads sample Indian identity documents (Aadhaar, PAN, Passport) from
publicly available Kaggle datasets and organises them into the training_data/
directory structure expected by train_pipeline.py.

Usage:
    python download_datasets.py

Requirements:
    - kaggle CLI configured (~/.kaggle/kaggle.json)
    - pip install kaggle datasets

Directory structure created:
    training_data/
    ├── aadhaar/
    ├── pan_card/
    ├── passport/
    └── other/
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAINING_DATA_DIR = Path("training_data")

CATEGORIES = {
    "aadhaar": TRAINING_DATA_DIR / "aadhaar",
    "pan_card": TRAINING_DATA_DIR / "pan_card",
    "passport": TRAINING_DATA_DIR / "passport",
    "other": TRAINING_DATA_DIR / "other",
}

# Kaggle datasets to download
# Format: (dataset_slug, target_category)
KAGGLE_DATASETS = [
    ("quadeer15sh/augmented-indian-id-cards-and-documents", None),  # Mixed dataset
]

# Supported file extensions
SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg"}

# Keywords to auto-categorise files by filename
FILENAME_KEYWORDS = {
    "aadhaar": ["aadhaar", "aadhar", "uid", "uidai"],
    "pan_card": ["pan", "income_tax", "incometax"],
    "passport": ["passport", "travel"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_dirs() -> None:
    """Create training data directory structure."""
    for path in CATEGORIES.values():
        path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created directory structure under: {TRAINING_DATA_DIR}/")


def _guess_category(filename: str) -> str:
    """Guess document category from filename keywords."""
    name_lower = filename.lower()
    for category, keywords in FILENAME_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return category
    return "other"


def _move_files_to_categories(source_dir: Path) -> dict[str, int]:
    """Walk source_dir and move supported files into category folders."""
    counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}

    for file_path in source_dir.rglob("*"):
        if file_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        if file_path.stat().st_size == 0:
            continue

        category = _guess_category(file_path.name)
        dest_dir = CATEGORIES[category]
        dest_path = dest_dir / file_path.name

        # Avoid overwriting — append a counter if needed
        counter = 1
        while dest_path.exists():
            stem = file_path.stem
            dest_path = dest_dir / f"{stem}_{counter}{file_path.suffix}"
            counter += 1

        shutil.copy2(file_path, dest_path)
        counts[category] += 1

    return counts


def _download_kaggle(dataset_slug: str, dest_dir: Path) -> bool:
    """Download a Kaggle dataset and extract it to dest_dir."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("❌ kaggle package not installed. Run: pip install kaggle")
        return False

    try:
        print(f"   Downloading: {dataset_slug}")
        os.makedirs(dest_dir, exist_ok=True)
        os.system(
            f'kaggle datasets download -d "{dataset_slug}" -p "{dest_dir}" --unzip'
        )
        return True
    except Exception as exc:
        print(f"   ❌ Failed to download {dataset_slug}: {exc}")
        return False


def _download_huggingface() -> bool:
    """Download datasets from HuggingFace as a fallback."""
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        print("❌ datasets package not installed. Run: pip install datasets")
        return False

    print("   Trying HuggingFace datasets...")

    # Example: download a document classification dataset
    try:
        dataset = load_dataset("maveriq/typographic-text-dataset", split="train[:200]")
        dest = TRAINING_DATA_DIR / "other"
        dest.mkdir(parents=True, exist_ok=True)

        count = 0
        for i, item in enumerate(dataset):
            if "image" in item and item["image"] is not None:
                img_path = dest / f"hf_sample_{i}.png"
                item["image"].save(img_path)
                count += 1

        print(f"   ✅ Downloaded {count} samples from HuggingFace")
        return True
    except Exception as exc:
        print(f"   ❌ HuggingFace download failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 50)
    print("Document Classifier — Dataset Downloader")
    print("=" * 50)
    print()

    # Step 1: Create directory structure
    _create_dirs()
    print()

    # Step 2: Check for existing files
    existing = sum(
        len(list(path.glob("*"))) for path in CATEGORIES.values()
    )
    if existing > 0:
        print(f"ℹ️  Found {existing} existing files in training_data/")
        answer = input("Re-download and add more? [y/N]: ").strip().lower()
        if answer != "y":
            print("Skipping download.")
            _print_summary()
            return
    print()

    # Step 3: Download from Kaggle
    print("📥 Downloading from Kaggle...")
    tmp_dir = TRAINING_DATA_DIR / "_tmp_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    kaggle_ok = False
    for dataset_slug, _ in KAGGLE_DATASETS:
        if _download_kaggle(dataset_slug, tmp_dir):
            kaggle_ok = True

    if kaggle_ok:
        print()
        print("📂 Organising files into categories...")
        counts = _move_files_to_categories(tmp_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        for cat, count in counts.items():
            print(f"   {cat}: {count} files")
    else:
        print()
        print("⚠️  Kaggle download failed. Trying HuggingFace fallback...")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _download_huggingface()

    print()
    _print_summary()


def _print_summary() -> None:
    print("=" * 50)
    print("📊 Training Data Summary")
    print("=" * 50)
    total = 0
    for category, path in CATEGORIES.items():
        if path.exists():
            count = len([f for f in path.iterdir() if f.suffix.lower() in SUPPORTED_EXTS])
            total += count
            status = "✅" if count > 0 else "⚠️ "
            print(f"  {status} {category:<12}: {count} files")
        else:
            print(f"  ❌ {category:<12}: directory missing")
    print(f"  {'TOTAL':<14}: {total} files")
    print()

    if total == 0:
        print("⚠️  No training files found!")
        print()
        print("Manual setup:")
        print("  1. Add your own documents to training_data/")
        print("     training_data/aadhaar/   → Aadhaar card images/PDFs")
        print("     training_data/pan_card/  → PAN card images/PDFs")
        print("     training_data/passport/  → Passport images/PDFs")
        print("     training_data/other/     → Other documents")
        print()
        print("  2. Then run: python train_pipeline.py")
    else:
        print("✅ Ready to train! Run: python train_pipeline.py")


if __name__ == "__main__":
    main()
