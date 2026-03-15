# FRIDGE-RAG 🧊🍳

Production-ready blueprint for a **multimodal recipe recommendation system** that turns a fridge photo into meal ideas using computer vision + retrieval-augmented generation (RAG).

> **Concept:** Upload a fridge image → detect available ingredients → retrieve matching recipes from a vector database → generate practical cooking suggestions.

---

## Table of Contents

- [Overview](#overview)
- [Core Capabilities](#core-capabilities)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Download Dataset](#download-dataset)
  - [Build Vector Database](#build-vector-database)
- [Run the Application](#run-the-application)
  - [Start API](#start-api)
  - [Start Dashboard](#start-dashboard)
- [Testing](#testing)
- [Operational Guidance](#operational-guidance)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

FRIDGE-RAG combines:

1. **Vision models** (YOLOv8 + DETR + CLIP) to infer ingredients from fridge images.
2. **Semantic retrieval** (SentenceTransformer embeddings + ChromaDB) to fetch relevant recipes.
3. **LLM reasoning** (OpenAI GPT models) to rank, explain, and personalize recipe recommendations.

The repository currently provides a clean project scaffold with scripts, API, dashboard, and test folders ready for implementation.

---

## Core Capabilities

- 📷 **Image-to-ingredient extraction** from fridge snapshots.
- 🧠 **RAG-based recipe retrieval** over a large recipe dataset.
- 🥗 **Context-aware suggestions** (e.g., based on detected ingredients).
- ⚡ **FastAPI backend** for model + retrieval orchestration.
- 📊 **Streamlit UI** for rapid experimentation and demos.

---

## System Architecture

```text
[Fridge Image]
     │
     ▼
[Vision Ensemble: YOLOv8 + DETR + CLIP]
     │
     ▼
[Detected Ingredients]
     │
     ▼
[Embedding Model: all-MiniLM-L6-v2]
     │
     ▼
[ChromaDB Recipe Index]
     │
     ▼
[Top-K Candidate Recipes]
     │
     ▼
[LLM Re-ranker / Explainer]
     │
     ▼
[Final Recommended Recipes]
```

---

## Project Structure

```bash
FRIDGE-RAG/
├── api/                  # FastAPI app and request/response schemas
├── dashboard/            # Streamlit dashboard
├── scripts/              # Data download + vector DB build scripts
├── src/                  # Core pipeline/config modules
├── tests/                # Unit/integration test suite
├── .env.example          # Environment variable template
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Tech Stack

- **Vision:** YOLOv8, DETR, CLIP (open-clip)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **LLM:** OpenAI API (e.g., GPT-4o-mini)
- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit (+ Plotly for visualization)
- **Data:** Food.com Recipes and Reviews (Kaggle)

Dataset source:
- https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews/data

---

## Getting Started

### Prerequisites

- Python **3.10+** recommended
- `pip` / virtual environment tooling
- Kaggle API credentials (`~/.kaggle/kaggle.json`) for dataset download
- OpenAI API key

### Installation

```bash
# 1) Clone repository
git clone <your-repo-url>
cd FRIDGE-RAG

# 2) (Recommended) Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

### Environment Variables

```bash
cp .env.example .env
```

Update `.env`:

```env
OPENAI_API_KEY=your_real_openai_key
```

### Download Dataset

```bash
python scripts/download_data.py
```

### Build Vector Database

```bash
python scripts/build_vectordb.py
```

> First indexing run can take several minutes depending on dataset size and machine resources.

---

## Run the Application

### Start API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Dashboard

In another terminal:

```bash
streamlit run dashboard/app.py
```

---

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Useful variants:

```bash
pytest tests/test_pipeline.py -v
pytest -q
```

---

## Operational Guidance

For production readiness, consider the following:

- **Model lifecycle:** version and pin all model artifacts.
- **Data contracts:** enforce strict request/response schemas in API.
- **Observability:** add structured logging, request tracing, and metrics.
- **Caching:** cache embeddings and frequent retrieval queries.
- **Guardrails:** validate image types/sizes and sanitize user text input.
- **Deployment:** containerize API/dashboard and deploy behind a reverse proxy.
- **Security:** store secrets in a vault/secret manager (not plaintext files).

---

## Troubleshooting

- **Kaggle download fails**
  - Ensure `~/.kaggle/kaggle.json` exists and has proper permissions.

- **OpenAI errors (`401` / auth)**
  - Verify `OPENAI_API_KEY` in `.env`.

- **Slow retrieval/indexing**
  - Rebuild vector DB on SSD and reduce dataset size for local testing.

- **Torch/vision install issues**
  - Match Torch build to your CUDA/CPU environment.

---

## Roadmap

- [ ] Implement ingredient confidence calibration across ensemble models.
- [ ] Add recipe filtering by dietary constraints/allergens.
- [ ] Add evaluation harness for retrieval precision/recall.
- [ ] Add Docker + CI/CD pipeline.
- [ ] Add end-to-end integration tests and benchmark suite.

---

## License

Add your license here (e.g., MIT, Apache-2.0).
