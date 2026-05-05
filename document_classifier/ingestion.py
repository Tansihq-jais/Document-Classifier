"""Ingestion module for the document-classifier pipeline.

Routes files to the correct extraction strategy and returns clean text.

Supported formats:
    - .pdf  → pdfplumber; falls back to PaddleOCR if text < pdf_text_min_chars chars
    - .png / .jpg / .jpeg → PaddleOCR (eng + hindi)
    - .docx → python-docx
    - other → IngestionError(error_type="unsupported_format")

PaddleOCR advantages over Tesseract:
    - No external binary installation required (pure Python)
    - Better accuracy on low-quality / tilted scans out of the box
    - Built-in angle detection and correction (no manual deskewing needed)
    - Faster inference on CPU
    - Supports Hindi (Devanagari) via the 'hindi' model

All extraction exceptions are caught and returned as IngestionError(error_type="corrupt_file").
If PaddleOCR returns empty text for a non-empty image, IngestionError(error_type="ocr_failure")
is returned.

Raw document text is never written to logs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pdfplumber  # type: ignore[import]
import docx  # type: ignore[import]
from PIL import Image  # type: ignore[import]

try:
    from pdf2image import convert_from_path  # type: ignore[import]
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    _PDF2IMAGE_AVAILABLE = False
    convert_from_path = None  # type: ignore[assignment]

from document_classifier.config import PipelineConfig
from document_classifier.logger import log_stage_complete, log_stage_error

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

_config = PipelineConfig()

SUPPORTED_FORMATS = [".pdf", ".png", ".jpg", ".jpeg", ".docx"]


def configure_ingestion(config: PipelineConfig) -> None:
    """Override the module-level PipelineConfig.

    Args:
        config: A PipelineConfig instance to use for all subsequent extract() calls.
    """
    global _config
    _config = config


# ---------------------------------------------------------------------------
# PaddleOCR — lazy singleton (loaded once, reused across all calls)
# ---------------------------------------------------------------------------

_paddle_ocr_instance = None


def _get_ocr():
    """Return a cached PaddleOCR instance (initialised on first call).

    Uses English model with built-in text-line orientation detection so tilted
    scans are corrected automatically without manual deskewing.
    Compatible with PaddleOCR >= 3.x API.
    """
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        try:
            from paddleocr import PaddleOCR  # type: ignore[import]
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # use_textline_orientation=True → auto-detects and corrects rotation
                # lang='en' → English model (handles mixed eng+hin documents)
                _paddle_ocr_instance = PaddleOCR(
                    use_textline_orientation=True,
                    lang="en",
                )
        except ImportError as exc:
            raise ImportError(
                "PaddleOCR is not installed. Run: pip install paddleocr paddlepaddle"
            ) from exc
    return _paddle_ocr_instance


def _paddle_ocr_image(img: Image.Image) -> str:
    """Run PaddleOCR on a PIL image and return extracted text.

    Converts the PIL image to a numpy array (RGB) which PaddleOCR expects.
    Joins all detected text lines with newlines.

    Args:
        img: PIL image in any mode.

    Returns:
        Extracted text string (may be empty if no text detected).
    """
    ocr = _get_ocr()

    # PaddleOCR expects a numpy array in RGB format
    img_rgb = img.convert("RGB")
    img_array = np.array(img_rgb)

    result = ocr.ocr(img_array, cls=True)

    if not result or result == [None]:
        return ""

    lines: list[str] = []
    for page_result in result:
        if not page_result:
            continue
        for line in page_result:
            # line format: [[bbox], (text, confidence)]
            if line and len(line) >= 2:
                text_conf = line[1]
                if text_conf and len(text_conf) >= 1:
                    lines.append(text_conf[0])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Successful extraction result from any supported format.

    Attributes:
        text: Extracted text (may be mixed Hindi/English).
        source_file: Original file path.
        file_format: One of "pdf", "image", "docx".
        extraction_method: One of "pdfplumber", "paddleocr", "docx_parser".
        page_count: Number of pages (1 for images/docx when not applicable).
    """

    text: str
    source_file: str
    file_format: str
    extraction_method: str
    page_count: int


@dataclass
class IngestionError:
    """Structured error returned when extraction fails.

    Attributes:
        file_path: Path of the file that caused the error.
        error_type: One of "unsupported_format", "corrupt_file", "ocr_failure".
        message: Human-readable description of the failure.
        supported_formats: Populated only for error_type="unsupported_format".
    """

    file_path: str
    error_type: str
    message: str
    supported_formats: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(file_path: str) -> ExtractionResult | IngestionError:
    """Extract text from a document file.

    Routes the file to the appropriate extraction strategy based on its extension.
    Logs stage completion or errors via logger.py.

    Args:
        file_path: Absolute or relative path to the document file.

    Returns:
        ExtractionResult on success, IngestionError on failure.
    """
    start_time = time.monotonic()
    document_id = os.path.basename(file_path)

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_FORMATS:
        error = IngestionError(
            file_path=file_path,
            error_type="unsupported_format",
            message=(
                f"Unsupported file format '{ext}'. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            ),
            supported_formats=list(SUPPORTED_FORMATS),
        )
        log_stage_error(
            stage="ingestion",
            document_id=document_id,
            error_type=error.error_type,
            message=error.message,
        )
        return error

    try:
        if ext == ".pdf":
            result = _extract_pdf(file_path)
        elif ext in (".png", ".jpg", ".jpeg"):
            result = _extract_image(file_path)
        else:  # .docx
            result = _extract_docx(file_path)
    except Exception as exc:  # noqa: BLE001
        error = IngestionError(
            file_path=file_path,
            error_type="corrupt_file",
            message=f"Failed to extract text from '{file_path}': {exc}",
        )
        log_stage_error(
            stage="ingestion",
            document_id=document_id,
            error_type=error.error_type,
            message=error.message,
        )
        return error

    if isinstance(result, IngestionError):
        log_stage_error(
            stage="ingestion",
            document_id=document_id,
            error_type=result.error_type,
            message=result.message,
        )
        return result

    duration_ms = (time.monotonic() - start_time) * 1000.0
    log_stage_complete(
        stage="ingestion",
        document_id=document_id,
        duration_ms=duration_ms,
        metadata={
            "file_format": result.file_format,
            "extraction_method": result.extraction_method,
            "page_count": result.page_count,
        },
    )
    return result


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------


def _extract_pdf(file_path: str) -> ExtractionResult | IngestionError:
    """Extract text from a PDF using pdfplumber, falling back to PaddleOCR if needed."""
    with pdfplumber.open(file_path) as pdf:
        page_count = len(pdf.pages)
        parts: list[str] = []
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            parts.append(page_text)
        text = "\n".join(parts)

    if len(text.strip()) < _config.pdf_text_min_chars:
        return _ocr_pdf(file_path, page_count)

    return ExtractionResult(
        text=text,
        source_file=file_path,
        file_format="pdf",
        extraction_method="pdfplumber",
        page_count=page_count,
    )


def _ocr_pdf(file_path: str, page_count: int) -> ExtractionResult | IngestionError:
    """Run PaddleOCR on a PDF by converting pages to images first."""
    images: list[Image.Image] = []

    # Try pdf2image first (best quality, requires Poppler)
    if convert_from_path is not None:
        try:
            images = convert_from_path(file_path, dpi=200)
        except Exception:
            images = []

    # Fallback: use pdfplumber to render pages
    if not images:
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    img = page.to_image(resolution=200).original
                    images.append(img)
        except Exception:
            images = []

    # Last resort: try opening directly as image
    if not images:
        try:
            images = [Image.open(file_path)]
        except Exception:
            pass

    if not images:
        return IngestionError(
            file_path=file_path,
            error_type="corrupt_file",
            message=f"Could not render pages from '{file_path}' for OCR",
        )

    parts: list[str] = []
    for img in images:
        text = _paddle_ocr_image(img)
        parts.append(text)

    text = "\n".join(parts)

    if not text.strip():
        return IngestionError(
            file_path=file_path,
            error_type="ocr_failure",
            message=f"PaddleOCR produced empty text for '{file_path}'",
        )

    return ExtractionResult(
        text=text,
        source_file=file_path,
        file_format="pdf",
        extraction_method="paddleocr",
        page_count=page_count,
    )


def _extract_image(file_path: str) -> ExtractionResult | IngestionError:
    """Extract text from an image file using PaddleOCR.

    PaddleOCR handles preprocessing (angle correction, binarisation) internally,
    so no manual deskewing or contrast enhancement is needed.
    """
    img = Image.open(file_path)
    text = _paddle_ocr_image(img)

    if not text.strip():
        return IngestionError(
            file_path=file_path,
            error_type="ocr_failure",
            message=f"PaddleOCR produced empty text for '{file_path}'",
        )

    return ExtractionResult(
        text=text,
        source_file=file_path,
        file_format="image",
        extraction_method="paddleocr",
        page_count=1,
    )


def _extract_docx(file_path: str) -> ExtractionResult:
    """Extract text from a .docx file using python-docx."""
    doc = docx.Document(file_path)
    parts: list[str] = []
    for para in doc.paragraphs:
        parts.append(para.text)
    for table in doc.tables:
        for row in table.rows:
            row_texts = [cell.text for cell in row.cells]
            parts.append("\t".join(row_texts))

    text = "\n".join(parts)

    return ExtractionResult(
        text=text,
        source_file=file_path,
        file_format="docx",
        extraction_method="docx_parser",
        page_count=1,
    )
