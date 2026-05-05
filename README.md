# Document Classifier

> Intelligent document classification system with OCR support for Indian identity documents (Aadhaar, PAN, Passport)

An intelligent document classification system that automatically identifies Indian government documents (Aadhaar Card, PAN Card, Indian Passport) using OCR and machine learning.

## Quick Start

See [SETUP.md](SETUP.md) for detailed installation instructions.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the web interface
python -m streamlit run document_classifier/app.py

# 3. Open browser to http://localhost:8501
```

## Features

- **Hybrid Classification**: Combines rule-based keyword matching with ML semantic analysis
- **Robust OCR**: Multi-strategy text extraction with automatic deskewing
- **Multilingual**: Supports English and Hindi
- **Web Interface**: Easy-to-use Streamlit UI
- **Clustering**: Automatic document grouping using HDBSCAN
- **Feedback Loop**: Continuous improvement through manual corrections

## How It Works

1. **OCR**: Extracts text from document images using Tesseract with 8 different strategies
2. **Rule Matching**: Checks for definitive keywords (e.g., "Income Tax Department" → PAN Card)
3. **ML Fallback**: If no rule matches, uses semantic embeddings to find nearest cluster
4. **Confidence Scoring**: Returns classification with confidence score (0-1)

## Architecture

```
Document → OCR (8 strategies) → Best Text
                                    ↓
                          Rule Check (fuzzy matching)
                                    ↓
                          Match? → Return Label (conf=1.0)
                                    ↓
                          No Match → Embed Text
                                    ↓
                          Find Nearest Cluster → Return Label (conf=0-1)
```

## Requirements

- Python 3.11+
- Tesseract OCR (with Hindi language pack)
- 4 GB RAM minimum
- 5 GB disk space

## Documentation

- [SETUP.md](SETUP.md) - Installation and configuration guide
- [document_classifier/](document_classifier/) - Source code with inline documentation

## Training Data

Download training datasets:
```bash
python download_datasets.py --all
```

Train the model:
```bash
python train_pipeline.py --dir training_data/passport --label "Indian Passport"
python train_pipeline.py --dir training_data/aadhaar --label "Aadhaar Card"
python train_pipeline.py --dir training_data/pan_card --label "PAN Card"
```

## Classification Rules

The system uses fuzzy keyword matching (85% similarity threshold) for:

**Aadhaar Card:**
- "Unique Identification Authority of India"
- "Your Aadhaar No"
- "आधार" + "आम आदमी का अधिकार"

**PAN Card:**
- "Income Tax Department" + "Permanent Account"
- "आयकर विभाग" + "PAN"

**Indian Passport:**
- "Republic of India" + "Passport"
- "भारत गणराज्य" + "Passport"

## Performance

- **Accuracy**: 95%+ on clean scans, 85%+ on phone photos
- **Speed**: 10-20 seconds per image (8 OCR strategies)
- **Supported Formats**: PDF, PNG, JPG, JPEG, DOCX

## Project Structure

```
document_classifier/
├── app.py               # Streamlit web UI
├── classifier.py        # Classification logic (rules + ML)
├── clustering.py        # HDBSCAN clustering
├── embedding.py         # Text embedding (sentence-transformers)
├── ingestion.py         # OCR and text extraction
├── label_service.py     # Manual labeling
├── pipeline.py          # End-to-end orchestration
├── vector_store.py      # ChromaDB interface
├── config.py            # Configuration
└── logger.py            # Structured logging
```

## License

Apache 2.0
