"""
Document Classifier — Streamlit Web UI

Run with:
    venv\\Scripts\\python.exe -m streamlit run document_classifier/app.py

Pages:
    🗂 Train    — Upload documents and ingest them into the pipeline
    🏷 Label    — Review cluster samples and assign labels
    🔍 Classify — Upload a document and see its predicted label
    💾 Labels   — Export / import labels as JSON
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure the project root is on sys.path when launched via streamlit directly
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from document_classifier.classifier import ClassificationError
from document_classifier.config import PipelineConfig
from document_classifier.ingestion import IngestionError, SUPPORTED_FORMATS
from document_classifier.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Document Classifier",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Pipeline — one instance shared across all reruns via cache
# ---------------------------------------------------------------------------

_SUPPORTED_EXTS = [e.lstrip(".") for e in SUPPORTED_FORMATS]


@st.cache_resource
def _get_pipeline(db_path: str) -> Pipeline:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    config = PipelineConfig(
        chroma_persist_dir=db_path,
        log_output="file",
        hdbscan_min_cluster_size=5,
        hdbscan_min_samples=2,
        chroma_api_key=os.environ.get("CHROMA_API_KEY", ""),
        chroma_tenant=os.environ.get("CHROMA_TENANT", ""),
        chroma_database=os.environ.get("CHROMA_DATABASE", ""),
    )
    return Pipeline(config=config)


def _save_upload(uploaded_file) -> str:
    """Save an uploaded file to a temp path and return the path."""
    suffix = Path(uploaded_file.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def _is_error(result) -> bool:
    """Check if a pipeline result is an error (works across module reloads)."""
    return hasattr(result, "error_type")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📄 Document Classifier")
    st.divider()
    db_path = st.text_input("ChromaDB directory", value="./chroma_db")
    pipeline = _get_pipeline(db_path)
    st.caption(f"Supported: {', '.join(SUPPORTED_FORMATS)}")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🗂 Train", "🏷 Label", "🔍 Classify", "💾 Labels"],
        label_visibility="collapsed",
    )

# ===========================================================================
# 🗂 Train
# ===========================================================================

if page == "🗂 Train":
    st.header("🗂 Train — Ingest Documents")
    st.write(
        "Upload documents to embed and store in the vector database. "
        "Clustering runs automatically every 20 new documents."
    )

    uploaded = st.file_uploader(
        "Choose files",
        type=_SUPPORTED_EXTS,
        accept_multiple_files=True,
    )

    if uploaded and st.button("Ingest", type="primary"):
        progress = st.progress(0, text="Starting…")
        ok_count = fail_count = 0
        log_lines: list[str] = []

        for i, uf in enumerate(uploaded):
            progress.progress((i + 1) / len(uploaded), text=f"Processing {uf.name}…")
            tmp_path = _save_upload(uf)
            try:
                result = pipeline.ingest(tmp_path)
            finally:
                os.unlink(tmp_path)

            if _is_error(result):
                log_lines.append(f"❌ **{uf.name}** — {result.message}")
                fail_count += 1
            else:
                log_lines.append(f"✅ **{uf.name}** — ingested (`{result.id[:8]}…`)")
                ok_count += 1

        progress.empty()
        st.success(f"Done: {ok_count} ingested, {fail_count} failed.")
        for line in log_lines:
            st.markdown(line)

        clusters = pipeline.list_clusters()
        if clusters:
            st.info(f"**{len(clusters)} cluster(s)** found. Go to **🏷 Label** to assign labels.")

# ===========================================================================
# 🏷 Label
# ===========================================================================

elif page == "🏷 Label":
    st.header("🏷 Label — Assign Labels to Clusters")

    clusters = pipeline.list_clusters()
    if not clusters:
        st.warning("No clusters yet. Ingest documents on the **🗂 Train** page first.")
    else:
        st.write(f"**{len(clusters)} cluster(s)** found. Review samples and enter a label for each.")

        for cluster_id in clusters:
            current_label = pipeline.get_cluster_label(cluster_id)
            header = (
                f"Cluster {cluster_id}  —  `{current_label}`"
                if current_label
                else f"Cluster {cluster_id}  —  *unlabelled*"
            )
            with st.expander(header, expanded=(current_label is None)):
                samples = pipeline.get_cluster_samples(cluster_id, n=5)
                if samples:
                    for j, text in enumerate(samples, 1):
                        snippet = text.replace("\n", " ").strip()[:300]
                        st.markdown(f"**Sample {j}:** {snippet}")
                else:
                    st.caption("No text samples available.")

                new_label = st.text_input(
                    "Label",
                    value=current_label or "",
                    key=f"label_{cluster_id}",
                    placeholder="e.g. GST Invoice, Aadhaar Card, PAN Card…",
                )
                if st.button("Save", key=f"save_{cluster_id}"):
                    if new_label.strip():
                        pipeline.label_cluster(cluster_id, new_label.strip())
                        st.success(f"Saved '{new_label.strip()}'")
                        st.rerun()
                    else:
                        st.error("Label cannot be empty.")

# ===========================================================================
# 🔍 Classify
# ===========================================================================

elif page == "🔍 Classify":
    st.header("🔍 Classify — Identify a Document")
    st.write("Upload one or more documents to classify them.")

    uploaded = st.file_uploader(
        "Choose files",
        type=_SUPPORTED_EXTS,
        accept_multiple_files=True,
        key="classify_uploader",
    )

    if uploaded and st.button("Classify", type="primary"):
        tmp_paths: list[tuple[str, str]] = []
        for uf in uploaded:
            tmp_paths.append((uf.name, _save_upload(uf)))

        try:
            results = pipeline.classify_batch([p for _, p in tmp_paths])
        finally:
            for _, p in tmp_paths:
                os.unlink(p)

        st.divider()
        for (name, _), result in zip(tmp_paths, results):
            col1, col2, col3 = st.columns([3, 2, 1])
            col1.markdown(f"**{name}**")
            if _is_error(result):
                col2.error(result.message)
            else:
                col2.markdown(f"🏷 `{result.label}`")
                col3.metric("Confidence", f"{result.confidence_score:.1%}")
                if result.low_confidence:
                    st.warning(f"⚠️ Low confidence for **{name}** — consider manual review.")

# ===========================================================================
# 💾 Labels
# ===========================================================================

elif page == "💾 Labels":
    st.header("💾 Labels — Export & Import")

    col_export, col_import = st.columns(2)

    with col_export:
        st.subheader("Export")
        labels = pipeline.export_labels()
        if not labels:
            st.info("No labelled clusters to export yet.")
        else:
            st.write(f"**{len(labels)} labelled cluster(s):**")
            for cid, lbl in sorted(labels.items()):
                st.markdown(f"- Cluster **{cid}**: `{lbl}`")
            st.download_button(
                label="⬇ Download labels.json",
                data=json.dumps(labels, indent=2, ensure_ascii=False),
                file_name="labels.json",
                mime="application/json",
            )

    with col_import:
        st.subheader("Import")
        uploaded_json = st.file_uploader("Upload labels.json", type=["json"], key="import_uploader")
        if uploaded_json:
            try:
                raw = json.loads(uploaded_json.read().decode("utf-8"))
                incoming: dict[int, str] = {int(k): v for k, v in raw.items()}
            except (json.JSONDecodeError, ValueError) as exc:
                st.error(f"Invalid JSON: {exc}")
                incoming = {}

            if incoming:
                st.write(f"**{len(incoming)} label(s) in file:**")
                for cid, lbl in sorted(incoming.items()):
                    st.markdown(f"- Cluster **{cid}**: `{lbl}`")
                if st.button("Import labels", type="primary"):
                    pipeline.import_labels(incoming)
                    st.success(f"Imported {len(incoming)} label(s).")
                    st.rerun()
