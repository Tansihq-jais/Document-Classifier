"""Central configuration for the document classifier pipeline."""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for all pipeline stages.

    Attributes:
        chroma_persist_dir:      ChromaDB storage directory. Use ":memory:" for tests.
        embedding_model_name:    HuggingFace sentence-transformers model identifier.
        hdbscan_min_cluster_size: Minimum documents required to form a cluster.
        hdbscan_min_samples:     HDBSCAN noise sensitivity (lower = fewer outliers).
        recluster_threshold:     New documents ingested before auto re-clustering fires.
        low_confidence_threshold: Confidence score below which a result is flagged.
        pdf_text_min_chars:      Min chars from pdfplumber before falling back to OCR.
        log_output:              Log destination: "stdout" | "file" | "both".
        log_file_path:           Log file path (used when log_output is "file" or "both").
        extraction_api_url:      Remote text extraction API endpoint. Set to "" to disable.
        extraction_api_timeout:  Timeout in seconds for the extraction API call.
    """

    chroma_persist_dir: str = "./chroma_db"
    embedding_model_name: str = "intfloat/multilingual-e5-large"
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 2
    recluster_threshold: int = 20
    low_confidence_threshold: float = 0.5
    pdf_text_min_chars: int = 50
    log_output: str = "stdout"
    log_file_path: str = "./pipeline.log"
    extraction_api_url: str = "https://staging-api.gipdataboard.io/api/extract_text_from_stream"
    extraction_api_timeout: int = 30
